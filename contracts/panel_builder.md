# Contract: panel_builder

## Purpose
Reconstructs a longitudinal project-month panel from the raw MoSPI PAIMANA
harvest (`data/paimana_live_raw.json`), assigns each snapshot a defensible
temporal order and an approximate calendar stamp, and emits one row per
observed *transition* carrying leak-free features plus the supervised label
`slip_next`.

It explicitly hands off to:
- `model_training` — consumes the emitted panel; owns all fitting, splitting,
  calibration and evaluation.
- `paimana_rules` / `paimana_engine` — own statutory scoring; this module
  never computes a GovScore.

## Two time-axis modes

### OBSERVED mode (current, preferred)
`scripts/harvest_monthly.py` queries `GetTileData` once per freeze month
(`MonthYear=YYYY-MM`) and tags every record with the reporting month it came
from. When **every** record for a project carries a `freeze_month`, the builder
reads order off the calendar and **assumptions A1 and A2 below do not apply**:

- snapshots are ordered by real month; `as_of_month` is the portal's own value;
- horizon windows are real calendar months, and `gap_months` records the actual
  spacing between consecutive reports;
- no project is ever excluded for "unreliable ordering" — there is nothing left
  to falsify. A project whose *reported* progress moves backwards is counted as
  a feed-quality observation (`dated_projects_with_backward_progress`) and is
  still trained on.

**The reporting month is part of a snapshot's identity.** Two distinct months in
which nothing changed on site are byte-identical on every measurement field; if
the month is left out of the dedupe key they collapse into one snapshot,
deleting a real observation. Frozen by
`tests/golden/test_dated_panel_golden.py::test_golden_two_quiet_months_are_not_collapsed`.

### RECONSTRUCTED mode (legacy fallback)
The original harvest left `MonthYear` empty, and the portal then returned every
record undated (`Month: null` / `Year: null`). Order had to be *reconstructed*
from monotone quantities under A1/A2. That path is retained so the legacy
`paimana_live_raw.json` still builds, and it is treated as an explicit,
measurable assumption — never as ground truth.

Measurement has since shown A1 is not merely unverifiable but **false for some
projects**: querying one project across freeze months showed expenditure
falling from 160.48 (2025-07) to 155.72 (2025-10). Reconstructed ordering
silently mis-sequenced those projects. Observed mode fixes this outright.

### Ordering assumption (A1) — reconstructed mode only
Within one project, cumulative expenditure and cumulative physical progress are
non-decreasing in real time. Revised completion dates are non-decreasing in real
time in the large majority of cases (a date is pushed out far more often than
pulled in).

### Ordering rule
1. Drop exact-duplicate records (same project appearing in several query slices).
2. Rank surviving snapshots by `(expenditure, physical_progress, revised_date)`
   lexicographically ascending.
3. Compute a per-project `order_confidence` = fraction of adjacent pairs in
   which physical progress is non-decreasing. This is a *diagnostic that can
   falsify A1 per project*, and is emitted on every row.
4. Projects whose `order_confidence < 0.60` are flagged `order_reliable = False`
   and excluded from training (retained in diagnostics).

### Calendar stamping assumption (A2) — reconstructed mode only
Snapshots are monthly and the newest snapshot corresponds to the harvest month
(`HARVEST_MONTH`, read from the raw file's mtime, default 2026-09). Snapshot `i`
counted backwards from the newest is stamped `HARVEST_MONTH - i months`. Stamps
are approximate and are used only for (a) out-of-time splitting and (b)
expressing lead time in months. They are never presented as observed dates.

## Inputs
- `raw_path`: defaults to `data/paimana_monthly_raw.json` (dated) and falls back
  to `data/paimana_live_raw.json` (legacy undated) only when the former is
  absent. A JSON list of raw PAIMANA records. Required keys per record: `ProjectId` (int), `ProjectName` (str),
  `SectorName` (str|null), `OriginalCost` (num|str), `RevisedCost` (num|str|""),
  `Expenditure` (num|str|null), `PhysicalProgress` (num|str|null),
  `SanctionDate`/`OriginalEndDate`/`RevisedDate` (`dd/mm/YYYY` str | null).
- `slip_threshold_days`: int, default 15. A revised-date movement strictly
  greater than this many days counts as a slip.
- `min_order_confidence`: float in [0,1], default 0.60.

## Outputs
- `data/paimana_panel.csv` — one row per transition `t -> t+1`.
- `data/paimana_panel_meta.json` — build diagnostics (counts, positive rate,
  concordance statistics, assumptions, exclusion tallies).

### Row schema (per transition)
Identity / bookkeeping (never fed to the model):
`project_id, project_name, sector, seq_index, n_snapshots, as_of_month,
 order_confidence, order_reliable, revised_date_t, revised_date_next,
 time_axis ("observed" | "reconstructed"), gap_months`

Leak-free features (observable at time `t`, before the outcome exists):
`original_cost, expenditure, expenditure_pct, expenditure_delta,
 expenditure_delta_pct, physical_progress, progress_delta,
 progress_delta_3m, remaining_progress_pct, spend_progress_gap,
 burn_to_progress_ratio, months_elapsed, original_duration_months,
 elapsed_ratio, schedule_pressure, months_to_revised_date,
 revised_date_already_slipped_months, stall_streak, sector_freq_encoded,
 log_original_cost, progress_velocity, required_velocity, velocity_deficit`

Label:
`slip_next` — 1 if `revised_date_next - revised_date_t > slip_threshold_days`,
else 0. Rows where either revised date is missing are dropped.

## Behavior cases (input → expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | Project with snapshots exp=[10,20,30], revdate=[2026-01-31, 2026-01-31, 2026-06-30] | 2 rows; `slip_next` = [0, 1] | Label reads the *next* revised date only |
| 2 | Project with 1 snapshot | 0 rows | No transition exists |
| 3 | Snapshot pair with `revised_date_next` null | row dropped | Cannot label |
| 4 | Snapshot pair `revdate_t = 2026-03-31`, `revdate_next = 2026-04-10` (10 days) | `slip_next = 0` | 10 ≤ threshold 15; absorbs month-end jitter |
| 5 | Snapshot pair where revised date is pulled *earlier* | `slip_next = 0` | Negative movement is not a slip |
| 6 | Project whose progress decreases on 3 of 4 adjacent pairs | `order_reliable = False`, rows excluded from training set | A1 falsified for that project |
| 7 | Exact duplicate records for a project | deduplicated before ordering; `n_snapshots` counts distinct only | Portal returns a project in several query slices |
| 8 | `OriginalCost <= 0` | project dropped entirely | Cannot compute cost-normalised features |
| 9 | `physical_progress` = 0 and `expenditure` = 0 across all snapshots | rows emitted with `burn_to_progress_ratio = 0.0` | No division by zero |
| 10 | `SanctionDate` null | `months_elapsed` imputed from `OriginalEndDate - original_duration`; `sanction_imputed = True` | Never silently defaults to 12.0 |

## Edge cases that must be covered
- Expenditure reported in rupees rather than crore (value > 20x original cost) —
  rescale by 100, matching the existing harvest heuristic, and count the fixes.
- Ties on all three ordering keys — order must be deterministic and stable, so
  the same input always yields byte-identical output.
- A project appearing with two different `SanctionDate` values across snapshots
  (observed in the real feed) — take the modal value, not the latest.
- `RevisedCost` as empty string `""` rather than null.
- Revised date equal to original end date (never revised) — valid, labelled by
  movement at the next snapshot like any other.
- A project whose every snapshot has an identical revised date — contributes
  only negatives; must not be dropped (it is the majority class).

## Explicitly out of scope
- Model fitting, splitting, calibration, metrics — `model_training` owns these.
- Statutory flag evaluation — `paimana_rules` owns this.
- Any imputation of `Remarks`, `StateName`, `DELAYED_TIME` or
  `COST_OVERRUN_PERC`. These are 100% null/zero in the live feed; this module
  reports their absence in the meta file and fabricates nothing.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [x] Implementation matches this contract (`scripts/build_panel.py`)
- [x] Golden tests exist for every behavior case above
      (`tests/golden/test_panel_builder_golden.py`, 19 cases for reconstructed
      mode; `tests/golden/test_dated_panel_golden.py`, 9 cases for observed mode)
