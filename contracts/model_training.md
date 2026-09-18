# Contract: model_training

## Purpose
Fits, calibrates and *honestly evaluates* the early-warning classifier on the
reconstructed panel, against classical statistical baselines, and emits both a
servable model bundle and a machine-readable metrics artifact. Every number the
API later reports about model quality must originate here — the API is
forbidden from carrying hardcoded performance constants.

Hands off to:
- `panel_builder` — supplies the labelled panel; this module never re-derives it.
- `model_service` — consumes the emitted bundle for inference only.
- `analytics` endpoint — reads the emitted metrics artifact verbatim.

## Inputs
- `data/paimana_panel.csv` (from `panel_builder`), rows with `order_reliable = True`.
- `seed`: int, default 20260917. All randomness is seeded; two runs on the same
  panel must produce identical metrics.
- `test_horizon_months`: int, default 3. Out-of-time cutoff: the newest N
  months of `as_of_month` form the test cohort.

## Outputs
- `model/paimana_schedule_risk_v2.pkl` — bundle dict with keys:
  `model, calibrator, feature_columns, classification_threshold,
   model_name, model_version, target, target_description, trained_at,
   panel_fingerprint, class_prior, leakage_excluded_features`
- `model/paimana_model_metrics.json` — the sole source of truth for reported
  performance. Contains per-model rows (AUC, PR-AUC, Brier, log-loss, ECE),
  DeLong test results, the split definition, and `provenance: "measured"`.

## Leakage policy (hard requirement)
Two feature classes are **excluded from the design matrix entirely** — not
zeroed at inference. Zeroing a feature the booster was trained to split on
creates train/serve skew and is the defect this contract exists to remove.

Excluded:
1. `has_revised_doc` / any indicator that a revision was filed — this is the
   label's own administrative footprint.
2. `current_delay_months` and anything derived from `revised_date_next` — the
   outcome.

`revised_date_already_slipped_months` (slippage *already on the record at time
t*) is **permitted**: it is observable before the next revision and is the
legitimate "this project has a slipping history" signal. It must be computed
from `revised_date_t` only.

## Evaluation protocol
Two splits are computed and both are reported; disagreement between them is
itself a finding.

- **Split A — GroupKFold by `project_id`, k=5.** No project contributes rows to
  both train and validation. This is the primary guard against the panel's
  dominant leak (same project, adjacent months, near-identical features), and
  it measures generalisation to a project the system has never seen.
- **Split B — Last-transition-per-project holdout.** For every project, the most
  recent transition is held out and all earlier ones are used for training.
  This is the deployment-realistic question: "for the projects we already
  monitor, can we call next month?"

### Split C — calendar out-of-time (ENABLED on an observed time axis)
Now that every record carries the portal's own freeze month, a genuine
calendar split is meaningful and is run per horizon: train on the earlier
months, test on the latest three. Measured at the 1-month horizon:

| | train | test |
|---|---|---|
| months | 2025-07 .. 2026-03 | 2026-04 .. 2026-06 |
| rows | 7,022 | 4,509 |
| projects | 1,696 | 1,667 |
| positive rate | 0.1631 | 0.2040 |

XGBoost **0.8296** vs logistic 0.7517, survival 0.7536, deadline rule 0.7370.

It is not a free lunch and the artifact says so: the portal onboarded projects
across the window, so later months hold more projects, and the base rate moves
month to month (a mass revision wave is visible in 2026-02). Those are real
properties of the programme, not artifacts of our reconstruction.

At the **6-month** horizon the split is structurally impossible and reports why
rather than vanishing: a recent row survives censoring only if a slip was
already observed inside its window, so the out-of-time cohort is 100% positive
and AUC is undefined (`unavailable_single_class_test_cohort`).

### Why this split was previously rejected (historical, reconstructed axis only)
A naive split on `as_of_month` is **confounded and must not be used on this
panel**. Because assumption A2 stamps every project's newest snapshot as the
harvest month, a project's history length determines how far back its rows
reach. Measured on the built panel:

| history length | rows | positive rate |
|---|---|---|
| 2–8 snapshots | 4,715 | ~33% |
| 9–13 snapshots | 4,151 | ~5–16% |

So the oldest calendar months contain *only* long-history projects, which are
the stable ones. Positive rate by reconstructed month therefore swings from
6% to 34% for reasons that are structural, not temporal. Training on early
months and testing on late months would measure that artifact, not forecasting
skill. The confound is recorded here so no later contributor reintroduces it.

### Confound guard (required)
Because history length correlates strongly with the label, the run must report
AUC **stratified by `n_snapshots` bucket**. If discrimination survives within
strata, the model has learned project dynamics; if it collapses, the headline
AUC was reading the confound. `n_snapshots` itself is an identity column and is
never in the design matrix.

### Baselines (MoSPI Dimension (b))
All fitted on the identical design matrix and split:
1. `Logistic Regression (ElasticNet)` — classical econometric baseline.
2. `Discrete-Time Proportional Hazards (complementary log-log)` — the survival
   baseline. Continuous-time Cox is the wrong estimator for a monthly panel
   with heavy ties; its discrete-time equivalent (Prentice & Gloeckler 1978)
   is the correct classical specification and is fitted directly with scipy,
   so it needs no extra dependency and runs offline for reproducibility.
   If `lifelines` is installed, a continuous-time Cox row is fitted too;
   when it is absent the row is emitted with `status: "unavailable"` and is
   **omitted from the comparison table** — never filled with a
   plausible-looking constant.
3. `Stratified prior` — predicts the class base rate. Sanity floor: AUC ≈ 0.50.
4. `Naive rule baseline` — the existing RuleFloor, min-max scaled to [0,1].
   Answers "does the ML add anything over the rules we already ship?"
5. `Stage-Aware XGBoost (proposed)`.

### Statistical test
DeLong's paired AUC test (`app.paimana_statistics.delong_test_paired_auc`,
already implemented and unit-tested) between XGBoost and every baseline, on the
same test cohort. Reported with Z, two-sided p, and both AUCs.

### Calibration (required, not optional)
Capital-at-Risk multiplies `p_model` by rupees, so a miscalibrated probability
is a miscalibrated rupee figure. A calibrator (isotonic, or sigmoid when the
positive count is under 500) is fitted on a held-out calibration fold and
stored in the bundle. Expected Calibration Error is reported pre- and
post-calibration; post must not be worse.

## Behavior cases (input → expected output)
| # | Input | Expected output | Notes |
|---|-------|------------------|-------|
| 1 | Panel with a feature perfectly correlated with the label | that feature must not be in `feature_columns` if it is on the exclusion list | Leakage policy is enforced in code, not by convention |
| 2 | Two runs, same panel, same seed | byte-identical `paimana_model_metrics.json` except `trained_at` | Reproducibility |
| 3 | `lifelines` not installed | Cox row `status: "unavailable"`, absent from comparison table, run still succeeds | Zero-confabulation |
| 4 | Test cohort with a single class | DeLong raises `ValueError`; run aborts with a clear message | Never emit an AUC on a degenerate cohort |
| 5 | XGBoost AUC < baseline AUC | metrics written faithfully, exit code 0 | The pipeline reports losses, it does not hide them |
| 6 | Post-calibration ECE > pre-calibration ECE | calibrator discarded, `calibrator: null`, noted in metrics | Never ship a calibrator that hurts |
| 7 | Predicting on a project with all features at training median | probability within [0,1], not NaN | Serving sanity |

## Edge cases that must be covered
- A sector present in test but unseen in train (frequency encoding must map it
  to 0, not raise).
- Panel with fewer than 200 usable rows — abort with a clear message rather
  than emitting a metrics file that looks authoritative.
- Class imbalance: positives are ~21%; `scale_pos_weight` is set from the train
  fold only, never from the full panel.
- NaN in any feature column at fit time — imputed with the *train-fold* median,
  and the imputer is stored in the bundle.

## Explicitly out of scope
- Serving, HTTP, caching — `model_service` and the API layer own these.
- The GovScore lattice and RuleFloor — `paimana_engine` owns these.
- Lead-time measurement — `lead_time_backtest` owns it.

## Status
- [x] Drafted
- [ ] Reviewed by a human
- [x] Implementation matches this contract (`scripts/train_model.py`)
- [x] Cases 1-3, 5-7 covered by the run itself (`assert_no_leakage`, `_cox_status`, reproducibility check) and by `backend/tests/test_endpoints.py::test_statistical_benchmark_delong`.
      Case 4 (single-class cohort) is enforced by `score_report` raising `ValueError`.

### Deviation from this contract, recorded
Split B was specified as a calendar out-of-time split. It was **replaced** by a
last-transition-per-project holdout after measurement showed the calendar split
is confounded on this panel; the evidence and reasoning are in the "Why not a
calendar out-of-time split" section above and in the metrics artifact under
`rejected_split`.
