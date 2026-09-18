# Contract: monthly_harvest

## Purpose
Harvests the MoSPI PAIMANA portal **one freeze month at a time**, so every
record arrives carrying the reporting month it belongs to. This replaces the
undated bulk harvest and, with it, the two weakest assumptions in the whole
platform.

## Why this exists
The original harvester called `GetTileData` with `Month`/`Year`/`MonthYear`
left empty. The portal then returned every monthly record it holds for a
project in one undated blob, with `Month: null` and `Year: null` on every row.
`panel_builder` therefore had to *reconstruct* the time axis from monotone
quantities under assumptions **A1** (expenditure and progress only move
forward) and **A2** (snapshots are monthly, newest = harvest month).

Two live endpoints make that unnecessary:

* `GET /Home/GetFreezeDates` → `{"firstFreeze":"2025-07","lastFreeze":"2026-07"}`
  — the portal publishes its own monthly freeze range.
* `POST /Home/GetTileData` with `MonthYear=YYYY-MM` → that month's snapshot only.

Verified behaviour: querying one project across freeze months returns genuinely
different expenditure and progress per month, and **expenditure is not always
monotone** (a probe project fell from 160.48 in 2025-07 to 155.72 in 2025-10).
A1 is therefore not merely unverifiable on the old data — it is **false** for
some projects, and the old ordering silently mis-sequenced them.

A second benefit: `MonthYear` shrinks each response enough that sectors which
previously blew the ASP.NET `maxJsonLength` limit (Roads & Highways returned
HTTP 500 unpartitioned) now return in one call, so the state-partitioning
fallback is rarely needed.

## Inputs
- `sectors`: the portal's sector catalog (id, name).
- `months`: derived from `GetFreezeDates`, inclusive of both ends.
- `out_path`: default `data/paimana_monthly_raw.json`.
- `resume`: if the output file exists, already-harvested (month, sector) pairs
  are skipped unless `--force`.

## Outputs
- `data/paimana_monthly_raw.json` — a JSON list of raw PAIMANA records, each
  annotated with two added keys:
  - `freeze_month`: `"YYYY-MM"`, the month the record was queried under.
  - `harvest_sector_id`: the sector slice it arrived in.
- `data/paimana_monthly_meta.json` — per-(month, sector) record counts, the
  freeze range reported by the portal, failures, and partitioning fallbacks used.

## Behaviour
1. Read the freeze range from `GetFreezeDates`. Never hardcode it.
2. For each (month, sector): request unpartitioned first.
3. On HTTP 500 (payload over `maxJsonLength`), fall back to `CostRange` 2 then 1;
   if a cost slice still fails, fall back to per-state partitioning.
4. Records are tagged and appended; duplicates *within* a (month, sector, project)
   are collapsed, duplicates *across* months are preserved — they are the panel.
5. A failed (month, sector) is recorded in the meta file as a failure and does
   **not** silently produce an empty month.

## Behaviour cases (input → expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | Sector returns 993 records for `2026-07` | 993 records tagged `freeze_month="2026-07"` | The normal path |
| 2 | Sector returns HTTP 500 unpartitioned | retried by `CostRange`, union emitted, fallback logged | Observed on Roads & Highways |
| 3 | Sector returns an empty list for a month | month recorded with `count: 0`, **not** an error | Some sectors genuinely have no rows in early freezes |
| 4 | Network error on one (month, sector) | recorded in `failures`, harvest continues | One bad slice must not void the run |
| 5 | Re-run with the output present | already-harvested pairs skipped | Harvest is resumable; the portal is slow |
| 6 | Same project appears twice in one (month, sector) | collapsed to one record | Query-slice duplication |
| 7 | Same project appears in 13 months | **13 records kept** | This is the panel; never deduplicate across months |

## Edge cases that must be covered
- `GetFreezeDates` unreachable → abort with a clear message rather than
  guessing a range.
- A freeze month present in the range but rejected by `GetTileData` → logged as
  a failure, not silently dropped.
- The anti-forgery token expiring mid-run → re-fetch it and continue.
- Record counts must be emitted per month so a truncated harvest is visible
  rather than appearing as a real decline in project count.

## Explicitly out of scope
- Panel construction, labelling, features — `panel_builder` owns those.
- Any imputation. If the portal does not return a field, it stays absent.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [x] Implementation matches this contract (`scripts/harvest_monthly.py`)
- [x] Golden tests exist for the downstream contract this enables
      (`tests/golden/test_dated_panel_golden.py`, 9 cases). Case 7 (never
      deduplicate across months) is pinned by
      `test_golden_two_quiet_months_are_not_collapsed`, which fails loudly if
      the reporting month is dropped from the dedupe key.

## Measured outcome of the first full run
18,601 records, 2,243 unique projects, 13 freeze months (2025-07 .. 2026-07),
**0 slice failures**. Panel grew from 8,838 reconstructed transitions to
**11,531 observed** ones, with 0 projects excluded for unreliable ordering
(previously 12) and `gap_months == 1` on 11,506 of 11,531 transitions,
confirming the reporting cadence really is monthly.
