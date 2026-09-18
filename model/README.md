# Model Card: PAIMANA Stage-Aware Schedule-Slip Early Warning

**Artifact:** `paimana_schedule_risk_v2.pkl`
**Version:** 2.0.0
**Produced by:** `scripts/train_model.py` (contract: `contracts/model_training.md`)
**Reproduce:** `python scripts/build_panel.py && python scripts/train_model.py && python scripts/lead_time_backtest.py`

Every number in this card is read from `paimana_model_metrics.json`, emitted by the training run. Nothing here is asserted independently of that artifact.

---

## 1. What it predicts

**Target:** `slip_next` — will the project's *officially reported revised completion date* be pushed further out at the next monthly PAIMANA report?

Three horizons are trained and shipped in one bundle:

| Key | Question |
|---|---|
| `1m` | Does the date move at the next monthly report? |
| `3m` | Does it move at any report within 3 months? |
| `6m` | Does it move at any report within 6 months? |

The label is **observed, not constructed**. The PAIMANA `GetTileData` endpoint returns every monthly record it holds per project, and the declared completion date is visibly pushed later over time. See `contracts/panel_builder.md`.

---

## 2. Training data

| | |
|---|---|
| Source | [paimana-proj.mospi.gov.in](https://paimana-proj.mospi.gov.in/) — 14,917 raw monthly records, 2,155 projects |
| Panel | **11,531 labelled transitions** from **1,807 projects** |
| Panel fingerprint | SHA-256 prefix `4a29176a6eeafa13` |
| Excluded | 28 rows / 12 projects whose reconstructed ordering failed the `order_confidence` check |
| Base rate | 22.4% (1m), 50.3% (3m), 73.7% (6m) |
| Seed | 20260917 — panel and metrics are byte-reproducible across runs |

**Known constraint on the data.** The portal returns `Month: null` / `Year: null` on every record, so snapshot order is *reconstructed* from cumulative expenditure and physical progress (assumptions A1/A2 in `contracts/panel_builder.md`), not read from a timestamp. A per-project `order_confidence` diagnostic can falsify that assumption and excludes projects where it does.

---

## 3. Feature space (22 features, leak-free by exclusion)

All features are observable strictly at time *t*.

| Group | Features |
|---|---|
| Financial | `original_cost`, `log_original_cost`, `expenditure`, `expenditure_pct`, `expenditure_delta`, `expenditure_delta_pct` |
| Physical | `physical_progress`, `progress_delta`, `progress_delta_3m`, `remaining_progress_pct`, `stall_streak` |
| Decoupling | `spend_progress_gap`, `burn_to_progress_ratio`, `schedule_pressure` |
| Schedule | `months_elapsed`, `original_duration_months`, `elapsed_ratio`, `months_to_revised_date`, `revised_date_already_slipped_months` |
| Velocity | `progress_velocity`, `required_velocity`, `velocity_deficit` |

### Excluded entirely (not zeroed at inference)

`has_revised_doc`, `current_delay_months_robust`, `revised_date_next`, `slip_next`.

This is the central correction from v1. The v1 bundle **trained on** `has_revised_doc` (34.2% of booster gain) and then **hard-zeroed it at inference**. That is train/serve skew, not a leakage fix: every live input landed in leaf regions whose training base rate was ~0.

| Measured across all 2,155 live projects | v1 | v2 |
|---|---|---|
| `p_model` range | 0.0001 – 0.0345 | 0.0103 – 0.8533 |
| median `p_model` | 0.0040 | 0.1231 |
| projects where ML drives the score | 0 | 1,222 |
| portfolio Capital-at-Risk | ₹2,985 Cr | ₹1,48,774 Cr |

`scripts/train_model.py::assert_no_leakage` enforces the exclusion list in code and raises if a banned column reaches the design matrix.

### Where the leakage line is drawn

`current_delay_months` **is used**, via `months_to_revised_date` and `revised_date_already_slipped_months`. It is the slippage *already on the record* at time *t* — the gap between the original completion date and the date the ministry is currently declaring. A desk officer reading the file that day can see it, so it is **state, not outcome**. What is excluded is whether the date moves *again*, which is the label.

### Removed during development

`sector_freq_encoded` was the highest-gain feature (0.162) in the first v2 model but correlated **−0.703** with a project's history length, which is itself correlated with the label. Within history-length strata its AUC collapsed to 0.34–0.56 and flipped direction. It was reading a confound, not sector behaviour, and was removed (cost: 1m AUC 0.869 → 0.854). See `BUGLOG.md`.

---

## 4. Measured performance

**Split A (primary) — GroupKFold(k=5) grouped by `project_id`.** No project appears in both train and validation; this measures generalisation to an unseen project.

| Model | 1m AUC | 3m AUC | 6m AUC |
|---|---|---|---|
| **Stage-Aware XGBoost (proposed)** | **0.854** | **0.911** | **0.946** |
| Logistic Regression (ElasticNet) | 0.798 | 0.883 | 0.920 |
| Discrete-Time PH (cloglog) | 0.793 | 0.881 | 0.923 |
| Stratified class prior | 0.484 | 0.489 | 0.473 |
| Deterministic RuleFloor (F1–F2) | 0.433 | 0.430 | 0.378 |

**Split B (deployment) — last transition per project held out.** 1m AUC 0.847, 3m 0.903, 6m 0.968.

**DeLong paired AUC tests, 1-month horizon:** vs logistic Z = 14.87; vs discrete-time PH Z = 15.15; both p < 1e-15.

**Operating points** (row-level, project-month):

| Horizon | Threshold | Precision | Recall | Base rate | Lift |
|---|---|---|---|---|---|
| 1m | 0.55 | 0.538 | 0.726 | 0.224 | 2.4× |
| 3m | 0.40 | 0.814 | 0.867 | 0.503 | 1.6× |
| 6m | 0.30 | 0.918 | 0.932 | 0.737 | 1.25× |

**Calibration** (isotonic, Expected Calibration Error cross-fitted over project-grouped folds):

| Horizon | ECE before | ECE after |
|---|---|---|
| 1m | 0.1337 | **0.0068** |
| 3m | 0.0251 | **0.0120** |
| 6m | 0.0454 | **0.0226** |

Calibration is not cosmetic here: Capital-at-Risk multiplies `p_model` by crores, so a miscalibrated probability is a miscalibrated rupee figure.

**Measured early warning:** median **2 months** (IQR 1–3) before the revised date is filed, at 81% project-level precision / 91% recall / 37% false alarms (threshold 0.30). This is a **lower bound** — the public feed exposes only ~13 monthly snapshots, so longer leads are unobservable. Full sweep in `paimana_lead_time.json`.

---

## 5. Confound guard

History length correlates strongly with the label on this panel, so every training run reports AUC *within* history-length strata:

| History length | Rows | Base rate | 1m AUC |
|---|---|---|---|
| 2–5 snapshots | 431 | 0.181 | 0.8694 |
| 6–8 snapshots | 5,185 | 0.289 | 0.8408 |
| 9–11 snapshots | 640 | 0.133 | 0.8542 |
| 12–20 snapshots | 5,275 | 0.077 | 0.8360 |

Discrimination holds at 0.81–0.83 inside every stratum, so the model reads project dynamics rather than the artifact.

---

## 5b. Self-challenges (measured, not argued)

Three objections a reviewer will raise, each answered by its own artifact.

**"Isn't this just a deadline rule?"** `months_to_revised_date` scores AUC 0.774 alone, so a fitted deadline heuristic is a benchmark row and every model is re-scored where the declared date has *not yet* passed — the only cohort where a warning can still change anything (`paimana_model_metrics.json` → `horizons.*.actionable_cohort`):

| Cohort (1m) | n | base rate | XGBoost | Logistic | Deadline rule | Margin |
|---|---|---|---|---|---|---|
| Already overdue | 2,835 | 0.467 | 0.7625 | 0.6235 | 0.4295 | +0.333 |
| **Not yet overdue** | 6,003 | 0.109 | **0.8034** | 0.7283 | 0.7331 | +0.0703 |
| ≥3 months runway | 4,544 | 0.067 | 0.7554 | 0.6319 | 0.6550 | +0.1004 |
| ≥6 months runway | 3,538 | 0.055 | 0.7492 | 0.6150 | 0.6582 | +0.0910 |

The margin over both the rule and logistic regression **widens** as runway increases — the model earns its keep precisely where forecasting is hard.

**"Your time axis is reconstructed."** The panel was rebuilt under four plausible ordering rules plus an inverted negative control, retraining each (`paimana_ordering_sensitivity.json`): AUC spread across plausible orderings **0.0036**; inverting the axis costs **0.1711**. The conclusion does not rest on which monotone quantity is treated as authoritative.

**"You're predicting paperwork."** Partly true and reported as such (`paimana_label_validity.json`). Schedule-relative distress precedes a filing (`schedule_pressure` d=+0.2192, `velocity_deficit` d=+0.0854, holding on active-only months); absolute activity does not — slipping projects are further along (74.49% vs 49.51% progress), not less active. Projects whose agencies never revise are invisible to the label, uncorrected.

That analysis also produced the platform's most actionable finding: **rule F1's statutory premise is inverted for schedule forecasting** — slip rate 0.0968 where F1 fires vs 0.3764 in the opposite condition, a 3.9x inversion. F1 was deliberately left unchanged (it is a valid financial-irregularity test); only the claim about it changed.

---

## 6. Known limitations

- **Reconstructed time axis.** Snapshot order and month stamps are inferred (A1/A2), not observed. All month-denominated figures inherit that approximation.
- **A calendar out-of-time split is invalid on this panel** and is deliberately not used; the reason is recorded in the metrics artifact under `rejected_split`.
- **Horizon cohorts are censored and the censoring is informative.** 3m drops 2,210 rows and 6m drops 3,677 — systematically the latest transition of each project. Metrics at those horizons are conditional on the surviving cohort. The 1-month horizon has zero censoring and is the cleanest comparison.
- **Serving approximates three features.** A CUF snapshot has no history, so `progress_delta_3m`, `progress_velocity` and `stall_streak` are approximated at inference. Every response names them in `feature_approximations`. See `contracts/inference_adapter.md`.
- **Trained on public-portal fields only.** The model never sees clearance pendency, dispute status, reporting staleness or delay narratives, because the portal does not publish them.
- **Not a causal model.** It ranks which projects are about to have their dates revised. It does not estimate the effect of an intervention.

---

## 7. Intended use

Prioritising which central-sector projects a review committee (PRAGATI / Cabinet Secretariat) should examine next, combined with the deterministic statutory floor via `GovScore = max(100 · P_model, RuleFloor)`.

**Not** intended as an automated basis for contractual penalty, payment withholding, or any adverse action against an implementing agency without human review of the underlying file. At the 1-month operating point roughly one flagged project-month in two is a false positive.

## 8. Explainability

TreeSHAP local attribution (`pred_contribs=True`) per prediction, translated into administrative directives. Attribution is instance-level: it explains why *this* project scores as it does, not the model in aggregate.

---

## Legacy artifact

`paimana_schedule_risk_xgboost.pkl` (v1, 12 features) is retained **only** as a fallback so a fresh checkout starts before the pipeline has been run. When it loads, `ModelService` sets `is_degraded = True` and `/api/v1/health` reports the reason. It should not be used for evaluation.
