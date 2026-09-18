# Contract: lead_time_backtest

## Purpose
Measures, on real MoSPI data, the single claim the whole platform rests on:
**how many months before the ministry officially filed a revised completion
date did the model already flag the project?**

An earlier version of the README asserted "12 to 18 months of early warning"
with nothing behind it. This module exists to replace that assertion with a
measured distribution, including the cases where the lead time is zero or
negative. It is the evidence layer for the product claim, and it is permitted
to produce an unflattering answer — which it did: the measured median is 2
months, and the claim was removed from the README rather than the measurement
being adjusted.

Hands off to:
- `model_training` — supplies the fitted, calibrated model and the split.
- `analytics` endpoint — reads the emitted artifact verbatim.

## Method
The cohort is **out-of-fold**: every project is scored by a model fitted on
GroupKFold folds that never contained that project, using the exported
`model/paimana_oof_predictions.csv`. (An earlier draft of this contract named an
out-of-time cohort; that split was rejected as confounded — see
`contracts/model_training.md`.)

For each project that experiences its first observed slip event at reconstructed
month `m_slip`:

1. Walk its snapshots forward from the earliest, scoring each with the model
   using only features observable at that snapshot.
2. Let `m_alert` be the first month at which `p_model >= alert_threshold`.
3. `lead_time_months = (m_slip + 1) - m_alert`.
   The `+1` is load-bearing: a transition at `seq_index i` is scored from the
   report *at* `i` and describes what the report at `i+1` will say, so the alert
   lands at report `m_alert` while the revision first appears on the record at
   report `m_slip + 1`. Getting this wrong understates lead time by exactly one
   reporting cycle (see `BUGLOG.md`).
   - Positive → the model warned before the paperwork.
   - Zero → the model warned in the same reporting month.
   - Negative → the model only warned after the revision was already filed.
   - Never alerted → recorded as `alerted: false` and counted in the miss rate;
     **excluded from the median rather than imputed as a large value**, with
     the exclusion count reported alongside the median.

Projects that never slip are scored the same way to produce the **false-alarm
rate** and the **alert-to-slip precision** at the chosen threshold. A lead-time
figure quoted without its false-alarm rate is meaningless and must not be
emitted alone.

`alert_threshold` is swept over a grid so the ministry can choose an operating
point; the artifact stores the full curve, not a single cherry-picked point.

## Inputs
- `model/paimana_oof_predictions.csv` (out-of-fold scores, from `train_model.py`)
- `data/paimana_panel_meta.json` (for the assumption text restated in `caveats`)
- `alert_thresholds`: list of floats, default `[0.20, 0.30, 0.40, 0.50, 0.60, 0.70]`

## Outputs
- `model/paimana_lead_time.json`:
  - `per_threshold`: for each threshold — `median_lead_months`,
    `p25_lead_months`, `p75_lead_months`, `mean_lead_months`,
    `n_slip_projects`, `n_alerted_before_slip`, `n_never_alerted`,
    `false_alarm_rate`, `alert_precision`, `recall`
  - `recommended_threshold` — the threshold maximising F1 on the test cohort,
    stated as a recommendation with its trade-offs, not as an optimum
  - `caveats` — the reconstructed-month assumption (A1/A2 from `panel_builder`)
    restated verbatim, and the observation-window truncation below
  - `provenance: "measured"`

## Behavior cases (input → expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | Project slips at transition 8, model first crosses threshold at transition 3 | `lead_time_months = 6` | `(8 + 1) - 3`; the core measurement |
| 2 | Project slips at transition 4, model first crosses at transition 4 | `lead_time_months = 1` | Detected one report before the revision lands |
| 3 | Project slips at transition 2, model first crosses at transition 6 | `lead_time_months = -3` | Negative lead times are reported, not clipped to 0 |
| 4 | Project slips, model never crosses threshold | `alerted = false`, counted in `n_never_alerted`, excluded from median | Must not be imputed |
| 5 | Project never slips, model never crosses | true negative; contributes to false-alarm denominator | |
| 6 | Project never slips, model crosses at month 5 | false alarm | Required to make lead time interpretable |
| 7 | Project with only 2 snapshots that slips at the 2nd | lead time can be at most 1; included with `window_truncated = true` | Short histories bias lead time *downward* |
| 8 | Threshold so low every project alerts at month 0 | large median lead time **and** false-alarm rate ≈ 1.0 | The artifact must make this degenerate point obvious |

## Edge cases that must be covered
- **Observation-window truncation (mandatory caveat).** The panel holds ~13
  monthly snapshots. A lead time longer than a project's observed history
  cannot be measured, so the reported median is a *lower bound* and must be
  labelled as such. Any claim of "12–18 months" that exceeds the observation
  window is unsupportable from this data and must not be made.
- A project whose first snapshot already shows a slipped revised date (slip
  occurred before the window opened) — excluded from lead-time measurement,
  counted separately as `n_left_censored`.
- Ties: model crosses threshold and slip is filed in the same reconstructed
  month — lead time 0, credited to neither side.
- Zero slip projects in the cohort — emit the artifact with
  `status: "insufficient_events"` rather than dividing by zero.

## Explicitly out of scope
- Changing the model or threshold used in production serving — this module
  measures, it does not tune the deployed system.
- Cost-of-delay valuation — `paimana_car` owns Capital-at-Risk.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [x] Implementation matches this contract (`scripts/lead_time_backtest.py`)
- [x] Golden tests exist for every behavior case above
      (`tests/golden/test_lead_time_golden.py`, 12 cases). Case 1-3 pin the
      `+1` arithmetic that was previously wrong (see `BUGLOG.md`); case 4 pins
      the refusal to impute never-alerted projects.
