# Contract: inference_adapter

## Purpose
Translates a single `ProjectInput` (the CUF monthly submission schema the API
accepts) into the exact 22-column design matrix the v2 bundle was trained on,
so that a project scored through the API gets the same features it would have
got in the training panel.

This module exists because the training panel and the serving schema are
different shapes, and the v1 failure was precisely a train/serve mismatch.
Any derivation here that cannot be computed from `ProjectInput` must be stated
as an explicit, documented approximation — never silently defaulted.

Hands off to:
- `model_service` — owns loading, prediction and SHAP; calls this adapter.
- `panel_builder` — owns the training-time derivation; this module must mirror
  its formulas exactly.

## Inputs
- `project`: a validated `ProjectInput`.
- `feature_columns`: the bundle's ordered feature list.
- `sector_frequency`: accepted for signature compatibility and currently unused.
  A sector frequency encoding was tried and **removed** — it correlated -0.703
  with project history length and was reading that confound rather than sector
  behaviour (see `BUGLOG.md`). The parameter is retained so a future, properly
  fold-fitted sector encoding can be reintroduced without a signature change.
- `medians`: training-fold medians, read from the bundle, used for any feature
  that cannot be derived.

## Outputs
- A `pandas.DataFrame` of one row, columns in `feature_columns` order, no NaN.
- An `approximations` dict naming every feature that was approximated rather
  than derived, so the API can surface it.

## Exact derivations (must mirror panel_builder)
| feature | derivation |
|---|---|
| `original_cost`, `expenditure`, `physical_progress`, `months_elapsed`, `original_duration_months` | passed through |
| `log_original_cost` | `log1p(original_cost)` |
| `expenditure_pct` | `100 * expenditure / max(1, original_cost)` |
| `expenditure_delta` | `expenditure_change_recent` |
| `expenditure_delta_pct` | `100 * expenditure_delta / max(1, original_cost)` |
| `progress_delta` | `progress_change_recent` |
| `remaining_progress_pct` | `max(0, 100 - physical_progress)` |
| `spend_progress_gap` | `expenditure_pct - physical_progress` |
| `burn_to_progress_ratio` | `expenditure_pct / physical_progress` if progress > 0.5 else 0 |
| `elapsed_ratio` | `months_elapsed / max(1, original_duration_months)` |
| `schedule_pressure` | `elapsed_ratio - physical_progress / 100` |
| `months_to_revised_date` | `original_duration_months + current_delay_months - months_elapsed` |
| `revised_date_already_slipped_months` | `current_delay_months` |
| `required_velocity` | `remaining_progress_pct / max(1, months_to_revised_date)`, else `remaining_progress_pct` when the date has passed |
| `velocity_deficit` | `required_velocity - progress_velocity` |

## Documented approximations (three, and they must be reported)
`ProjectInput` is a **single monthly snapshot**; the panel had a project's
history. Three features need history and are approximated:

1. `progress_delta_3m` ← `3 * progress_change_recent`. Assumes the most recent
   month's rate held for three months. Over-estimates for a project that just
   restarted after a stall, under-estimates for one that just stopped.
2. `progress_velocity` ← `progress_change_recent`. The panel computed this over
   a 3-month window; here only one month is observable.
3. `stall_streak` ← `1.0` if `progress_change_recent <= 0.1` else `0.0`. The
   panel counted consecutive stalled months; a single snapshot can only
   distinguish stalled from not-stalled, so the served value is capped at 1
   where training values ranged higher.

Callers that hold real history should pass it rather than rely on these.

## Behavior cases (input → expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | `original_cost=1000, expenditure=600, physical_progress=30` | `expenditure_pct=60`, `spend_progress_gap=30` | The F1 decoupling signal |
| 2 | `physical_progress=0` | `burn_to_progress_ratio=0.0` | No division by zero |
| 3 | `months_elapsed=60, original_duration_months=36, current_delay_months=0` | `months_to_revised_date=-24` | Deadline already passed; negative is meaningful and is not clipped |
| 4 | deadline passed (`months_to_revised_date <= 0`) | `required_velocity = remaining_progress_pct` | Everything remaining is required immediately |
| 5 | `sector="Nonexistent Sector"` | row built normally | Sector is not a model input; an unseen value must never raise |
| 6 | `progress_change_recent=0.0` | `stall_streak=1.0`, listed in `approximations` | Stall detected from a single snapshot |
| 7 | any input | output column order == `feature_columns`, no NaN | Column order mismatch silently corrupts XGBoost input |
| 8 | a feature in `feature_columns` that the adapter cannot derive | filled from `medians`, named in `approximations` | Never silently zero-filled |

## Edge cases that must be covered
- `original_cost` of 0 or negative — `ProjectInput` forbids it, but the adapter
  must still guard the divisions rather than trust upstream validation.
- Bundle `feature_columns` containing a name the adapter has never seen (model
  retrained with new features) — fill from medians and report, do not crash.
- `physical_progress = 100` — `remaining_progress_pct = 0`,
  `required_velocity = 0`, no NaN.

## Explicitly out of scope
- Any leakage decision: the exclusion list lives in `model_training`.
- Scoring, calibration, SHAP, GovScore.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [x] Implementation matches this contract (`backend/app/services/feature_adapter.py`)
- [x] Golden tests exist for every behavior case above (`tests/golden/test_inference_adapter_golden.py`, 12 cases including train/serve parity)
