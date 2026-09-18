# Bug Log

Every entry here must result in a permanent case added to `tests/golden/`
before it's marked resolved. A patched bug without a regression case is not
resolved — it's just hidden until the next rewrite.

---

## 2026-09-17 — ML branch dead on real data (train/serve skew)
- **Symptom:** Across all 2,155 live MoSPI projects, `p_model` spanned only
  0.0001–0.0345 (median 0.0040). Every project in the HIGH tier was there
  purely because of the RuleFloor; `dominant_source` was never
  `MACHINE_LEARNING` for any elevated project. Total Capital-at-Risk across
  ₹41.85 lakh crore of monitored capex computed to ₹2,985 Cr.
- **Root cause:** `has_revised_doc` carried 34.22% of the v1 booster's gain,
  and `model_service.build_leak_free_frame()` hard-zeroed it at inference.
  That is not a leakage fix — it is train/serve skew. Zeroing a feature the
  model was trained to split on pushes every input into leaf regions whose
  training base rate was ~0, so the booster returns near-zero for everything.
  The README presented this zeroing as the platform's headline innovation.
- **Stage/module:** `backend/app/services/model_service.py` (inference), with
  the true origin in an untracked training step — the repo contained a `.pkl`
  and no training script at all.
- **Fix:** Rebuilt the pipeline so leakage is handled by *exclusion at training
  time* (`scripts/train_model.py`, `LEAKAGE_EXCLUDED_FEATURES`). `p_model` now
  spans 0.0103–0.8533 (median 0.1231) and portfolio CaR is ₹1,48,774 Cr.
- **Regression case added:** `backend/tests/test_endpoints.py::test_v2_probabilities_are_not_degenerate`
  (asserts spread > 0.5 and a populated HIGH/CRITICAL tier), plus
  `::test_v2_excludes_leakage_proxies_from_design_matrix_entirely`.
- **Status:** fixed / verified

---

## 2026-09-17 — Fabricated model-performance constants in the serving layer
- **Symptom:** `GET /api/v1/analytics/benchmark-baseline` returned a benchmark
  table (Stage-Aware XGBoost AUC 0.814 vs Cox PH 0.732, Z = 4.12, p = 0.00001)
  that had never been computed. `README.md` quoted a *third*, contradictory set
  for the same comparison (0.814 vs 0.712, Z = 3.42). `cuf_analytics.py`
  carried the same defect in a second place (ceiling AUC 0.81; NLP proxy gain
  0.756 → 0.814), labelled "protocol-level reference values".
- **Root cause:** Performance numbers were authored as source-code constants
  rather than emitted by an evaluation run. With no training script in the
  repo there was nothing to reconcile them against, so the API, the README and
  the CUF module drifted into three mutually inconsistent stories.
- **Stage/module:** `backend/app/api/v1/endpoints/analytics.py` (`_BENCHMARK_ROWS`),
  `backend/app/services/cuf_analytics.py` (`_OBSERVABLE_CEILING`, `_PROXY_AUGMENTATION`).
- **Fix:** Deleted every constant. Both endpoints now read
  `model/paimana_model_metrics.json`, written by `scripts/train_model.py`, and
  return HTTP 503 with reproduction instructions when it is absent.
- **Regression case added:** `backend/tests/test_endpoints.py::test_statistical_benchmark_delong`
  asserts `provenance == "measured"` and that the XGBoost-vs-baseline margin is
  certified by a live DeLong test on the same cohort.
- **Status:** fixed / verified

---

## 2026-09-17 — NLP delay-reason pillar has no input on the public feed
- **Symptom:** The platform reported a +11.4% (README) / +5.8pp AUC (API) gain
  from mining "Reasons for Delay" free text. Zero projects in the served
  portfolio had `reasons_for_delay` populated.
- **Root cause:** `Remarks`, `RevisedDateReason` and `RevisedCostReason` are
  null on **14,917 / 14,917** records returned by the PAIMANA public portal.
  The extractor is correct and unit-tested; it simply has no source text. The
  reported gain described data that does not exist. (`StateName` is likewise
  null on all records, contradicting the README's "all 36 States/UTs" claim,
  and `DELAYED_TIME` / `COST_OVERRUN_PERC` are zero on every record.)
- **Stage/module:** `scripts/download_real_paimana_data.py` (ingest) →
  `backend/app/services/cuf_analytics.py` (reporting).
- **Fix:** All proxy-augmentation figures now return the sentinel `-1.0` with a
  `proxy_augmentation_provenance` string naming the 14,917/14,917 null count.
  `scripts/build_panel.py` emits `fields_absent_from_live_feed` in its meta so
  the absence is visible at the pipeline's first stage.
- **Regression case added:** `tests/golden/test_panel_builder_golden.py::test_golden_absent_feed_fields_are_reported_not_imputed`;
  `backend/tests/test_endpoints.py::test_cuf_gap_analysis_zero_confabulation`
  now asserts every proxy field is the sentinel.
- **Status:** fixed / verified

---

## 2026-09-17 — Statutory flags F3/F4/F5 structurally inert on public data
- **Symptom:** On the live 2,155-project portfolio, F3 (clearance pendency),
  F4 (stale reporting) and F5 (litigation) fired on exactly **zero** projects,
  while the dashboard presented all five flags as active monitoring.
- **Root cause:** The ingest script hardcodes `days_since_last_cuf_update = 15`
  for every project, so F4's `> 60` condition is unreachable by construction.
  `clearance_pending_days` and dispute status are derived from `Remarks`, which
  is null on every record.
- **Stage/module:** `scripts/download_real_paimana_data.py` → `paimana_rules.py`.
- **Fix:** Documented as a known limitation in `contracts/rule_floor.md` rather
  than faked. The rules are correct against the full internal CUF; their
  inertness on public data is now the concrete, evidenced argument for the
  CUF 2.0 field proposal instead of a hidden gap.
- **Status:** fixed (documented) / verified

---

## 2026-09-17 — Shipped rule set is anti-predictive for schedule slip; F6 missing
- **Symptom:** Scored against the real outcome (revised completion date moves
  at the next report), the shipped rules F1+F2 achieve **AUC 0.433** — worse
  than random — degrading to 0.378 at a 6-month horizon.
- **Root cause:** F1 and F2 fire on early-stage projects with distant declared
  deadlines, which are *less* likely to be revised imminently. They detect
  chronic governance breaches, which is a legitimate but different question
  from forecasting. Meanwhile the strongest simple signal in the data was
  absent from the rule set entirely: "the declared completion date has already
  passed" covers 32.1% of project-months at a 4.3x lift (46.7% vs 10.9% slip
  rate), AUC 0.725 standing alone.
- **Stage/module:** `backend/app/paimana_rules.py`.
- **Fix:** Added flag **F6 — Declared Completion Date Elapsed** (weight 25,
  critical escalation at ≥ 12 months overdue), specified in
  `contracts/rule_floor.md` with the measured evidence table.
- **Regression case added:** `tests/golden/test_rule_floor_golden.py` — 8 F6
  cases plus a 5-case parametrised guard
  (`test_golden_f6_does_not_change_floors_where_it_is_inactive`) proving F6 is
  inert for every project whose deadline has not passed.
- **Status:** fixed / verified

---

## 2026-09-17 — Isotonic calibration scored on its own training data
- **Symptom:** The first training run reported Expected Calibration Error
  improving 0.1246 → **0.0000**. A perfect zero is not a result; it is a
  fingerprint of in-sample evaluation.
- **Root cause:** The calibrator was fitted on the out-of-fold scores and then
  scored on those same scores. Isotonic regression is flexible enough to drive
  in-sample ECE to exactly zero, which measures nothing.
- **Stage/module:** `scripts/train_model.py::fit_calibrator`.
- **Fix:** Post-calibration ECE is now cross-fitted over 5 project-grouped
  folds. Honest figures: 0.1337 → 0.0068 (1m), 0.0251 → 0.0120 (3m),
  0.0454 → 0.0226 (6m). The metrics key was renamed to
  `ece_after_cross_fitted` so the estimator is visible in the artifact.
- **Status:** fixed / verified

---

## 2026-09-17 — Lead-time measurement off by one reporting cycle
- **Symptom:** The backtest reported a median early-warning lead of 0 months,
  implying the model never warns before the revision is filed.
- **Root cause:** A transition at `seq_index i` is scored from the report *at*
  `i` and predicts what the report at `i+1` will say. Lead time was computed as
  `m_slip - m_alert`, which ignores that the revision first appears on the
  record at report `m_slip + 1`.
- **Stage/module:** `scripts/lead_time_backtest.py::analyse_threshold`.
- **Fix:** `lead = (m_slip + 1) - m_alert`, with the reasoning recorded inline
  and the definition spelled out in the emitted artifact. Median lead at the
  recommended operating point is now 2.0 months (a *lower bound*: the public
  feed exposes only ~13 monthly snapshots, so longer leads are unobservable).
- **Status:** fixed / verified

---

## 2026-09-17 — Sector frequency encoding was reading the history-length confound
- **Symptom:** `sector_freq_encoded` ranked as the single highest-gain feature
  (0.162) in the first v2 model and scored AUC 0.712 standing alone.
- **Root cause:** Found during the adversarial pass, in code written earlier in
  the same session. Sector frequency correlates **−0.703** with a project's
  number of snapshots, and history length is itself strongly correlated with
  the label (short histories ~33% positive, long histories ~5–16%). Within
  history-length strata the feature's AUC collapses to 0.34–0.56 and *flips
  direction* between strata — it was encoding the confound, not sector
  behaviour. Roads & Highways (979 projects, median 7 snapshots, 37% slip) vs
  Railways (200 projects, median 11 snapshots, 4% slip) drove the whole effect.
- **Stage/module:** `scripts/build_panel.py` (feature construction).
- **Fix:** Feature removed. Cost: 1-month AUC 0.8685 → 0.8541. The sector
  frequency table is retained in the panel meta under
  `sector_frequency_reference_only` for reporting, and is not a model input.
  The stratified confound guard that caught it is now a permanent part of every
  training run (`stratified_confound_check`, reported per horizon).
- **Status:** fixed / verified

---

## 2026-09-17 — Horizon probability curve could invert
- **Symptom:** For some projects the API returned P(slip within 1m) = 0.052,
  P(3m) = 0.231, P(6m) = 0.110 — a 6-month probability *below* the 3-month one.
- **Root cause:** "Slips within 6 months" strictly contains "slips within 3
  months", so P(6m) ≥ P(3m) ≥ P(1m) is a mathematical necessity. The horizon
  models are fitted independently on differently-censored cohorts, and nothing
  in the fitting enforces the nesting.
- **Stage/module:** `backend/app/services/model_service.py::predict_horizons`.
- **Fix:** A running maximum projects the curve back onto the feasible set —
  the minimal correction. Where it binds, the response sets
  `monotonicity_enforced: true` and retains `independent_probability`, so the
  adjustment is auditable rather than hidden.
- **Status:** fixed / verified

---

## 2026-09-17 — Portfolio fast path bypassed probability calibration
- **Symptom:** The portfolio leaderboard and the single-project endpoint could
  return different probabilities for the same project.
- **Root cause:** `evaluate_project(include_drivers=False)` reached into the
  raw booster via `predict_proba` to skip TreeSHAP, bypassing the isotonic
  calibrator. It also hardcoded `base_rate_probability = 0.5`. Since Capital-
  at-Risk multiplies `p_model` by rupees, the two views would have disagreed in
  crores on the same project.
- **Stage/module:** `backend/app/services/governance_service.py`.
- **Fix:** The fast path now calls `model_service.predict()` (still skipping
  TreeSHAP) so calibration always applies, and the base rate comes from the
  bundle's measured `class_prior` rather than a placeholder.
- **Status:** fixed / verified

---

## 2026-09-17 — Rule F1's statutory premise is inverted for schedule forecasting
- **Symptom:** The shipped rule set scored AUC 0.433 against schedule slippage —
  worse than random — with no mechanical explanation for *why*.
- **Root cause:** Traced by `scripts/label_validity.py`. F1 (GFR 2017 Rule 159)
  assumes disbursement running ahead of physical progress predicts trouble.
  Measured against the real outcome, the relationship runs the other way:

  | spend-vs-progress band | project-months | slip rate |
  |---|---|---|
  | spend ≥25pp ahead (**F1 fires**) | 1,157 | **0.0968** |
  | roughly aligned | 5,056 | 0.1733 |
  | progress ≥25pp ahead (**opposite**) | 2,625 | **0.3764** |

  A 3.9x inversion. F1 fires hardest on early-stage projects whose declared
  deadlines are still distant — precisely the ones least likely to be revised
  next month.
- **Stage/module:** `backend/app/paimana_rules.py` (the rule), surfaced by
  `scripts/label_validity.py` (the measurement).
- **Fix:** **F1 was deliberately NOT changed.** It encodes a real statutory
  test for unearned contractor advances, which is a financial-irregularity
  question, not a schedule question. Flipping a statutory rule to chase a
  correlation would be the exact category error the platform exists to prevent.
  What changed is the claim: `contracts/rule_floor.md` now records the
  inversion and states that F1 is never to be presented as a schedule
  predictor, and `/api/v1/analytics/label-validity` publishes the evidence.
- **Regression case added:** covered indirectly by
  `tests/golden/test_rule_floor_golden.py` (F1 behaviour is frozen, so a silent
  "fix" that flips it breaks the golden suite).
- **Status:** fixed (reclassified, not altered) / verified

---

## 2026-09-17 — First label-validity analysis encoded a disproven folk theory
- **Symptom:** The first run of `scripts/label_validity.py` returned
  "LABEL LOOKS ADMINISTRATIVE: only 2 of 6 indicators corroborate it",
  implying the platform predicts paperwork rather than project distress.
- **Root cause:** The analysis hardcoded an `expected_direction` for every
  indicator, and for `spend_progress_gap` that expectation was F1's own
  premise — which the same data disproves. It also lumped schedule-relative
  distress together with absolute activity, so "slipping projects are not less
  active" was scored as evidence against the label when it is actually a
  separate and true finding about project stage.
- **Stage/module:** `scripts/label_validity.py`.
- **Fix:** Indicators are now grouped by what they test (schedule-relative /
  absolute activity / spend pattern); no direction is asserted for the
  hypothesis under test; the comparison is repeated on physically-active
  months as a control; and stage-coverage bias is measured separately. The
  corrected verdict is that the label **is** schedule-grounded, with two
  explicit qualifications carried alongside it.
- **Note:** The initial dormancy hypothesis ("non-revising projects are
  dormant") was also tested and **rejected** — their monthly progress rate is
  indistinguishable (1.997 vs 2.060). They are simply earlier-stage
  (49.51% vs 74.49% mean progress).
- **Status:** fixed / verified

---

## 2026-09-18 — Time axis was reconstructed when the portal publishes it
- **Symptom:** The panel inferred snapshot order from monotone quantities under
  assumptions A1/A2, because every harvested record carried `Month: null` and
  `Year: null`. Assumption A1 was the platform's deepest methodological
  weakness and drew a dedicated robustness study to defend it.
- **Root cause:** `scripts/download_real_paimana_data.py` called `GetTileData`
  with `Month`/`Year`/`MonthYear` **left empty**, so the portal returned every
  monthly record for a project in one undated blob. Two endpoints on the same
  dashboard made this unnecessary the whole time:
  `GET /Home/GetFreezeDates` (the portal's own freeze range, 2025-07..2026-07)
  and `POST /Home/GetTileData` with `MonthYear=YYYY-MM`.
- **Stage/module:** ingest (`scripts/download_real_paimana_data.py`) →
  `scripts/build_panel.py`.
- **Fix:** `scripts/harvest_monthly.py` harvests one freeze month at a time and
  tags every record with `freeze_month`; `build_panel.py` gained an OBSERVED
  mode that reads order off the calendar. A1/A2 no longer apply to the shipped
  panel. Measured effect: 8,838 → **11,531** transitions, 1,631 → 1,807
  projects, 12 → **0** projects excluded for unreliable ordering, 1-month AUC
  0.8541 → **0.8665**, margin over logistic +0.056 → **+0.103**, and a calendar
  out-of-time split became valid (**0.8296**).
- **Also revealed:** A1 was not merely unverifiable but **false** for some
  projects — a probe project's cumulative expenditure fell from 160.48
  (2025-07) to 155.72 (2025-10), so expenditure-ordering had mis-sequenced it.
- **Regression case added:** `tests/golden/test_dated_panel_golden.py` (9 cases).
- **Status:** fixed / verified

---

## 2026-09-18 — Dedupe key would have deleted real months (caught before shipping)
- **Symptom:** Caught while implementing the dated harvest, before any data was
  published from it.
- **Root cause:** The snapshot dedupe key was built from measurement fields
  only. Two genuinely distinct reporting months in which nothing changed on
  site are byte-identical on every one of those fields, so they would have
  collapsed into a single snapshot — silently deleting a real observation and
  corrupting every month-denominated feature downstream.
- **Stage/module:** `scripts/build_panel.py::dedupe_key`.
- **Fix:** `freeze_month` is now part of a snapshot's identity. Duplicates
  *within* a (month, sector) slice still collapse; duplicates *across* months
  never do — they are the panel.
- **Regression case added:**
  `tests/golden/test_dated_panel_golden.py::test_golden_two_quiet_months_are_not_collapsed`.
  Verified to have teeth by reverting the fix and confirming it fails with
  "a quiet month was collapsed away".
- **Status:** fixed / verified

---

## 2026-09-18 — Flag F6's supporting evidence was inflated by the reconstruction
- **Symptom:** F6 ("declared completion date has elapsed") was introduced on
  measured evidence of a **4.3× lift** covering 32.1% of project-months with a
  standalone AUC of 0.725. On the corrected observed time axis those figures
  fall to a **1.38× lift**, 8.3% coverage, AUC **0.5170**.
- **Root cause:** `months_to_revised_date` was computed against a reconstructed
  `as_of` stamp, which systematically placed projects further past their
  declared deadlines than they actually were. The rule's justification was
  measured honestly at the time, but on a time axis that was itself wrong.
- **Stage/module:** `scripts/build_panel.py` (feature) → `contracts/rule_floor.md`
  (the claim).
- **Fix:** **F6 was NOT retuned or removed.** "The declared date has elapsed
  while work is incomplete" remains a true statutory fact, and quietly
  rewriting a rule to chase a freshly-measured correlation is the same error
  refused for F1. The *claim* was corrected in `contracts/rule_floor.md` and
  the README.
- **What the corrected data shows instead:** risk peaks *just before* the
  deadline, not after it — the 0–3 months-remaining band carries a 0.4655 slip
  rate against 0.0128 for >24 months out, a 36× spread. An imminent-deadline
  flag is recorded as a **candidate F7 for human review**, deliberately not
  added by the model author.
- **Regression case added:** F6's behaviour is frozen by
  `tests/golden/test_rule_floor_golden.py` (8 cases), so any silent retune
  breaks the golden suite.
- **Status:** fixed (claim corrected, rule unchanged) / verified

---

## 2026-09-18 — Portfolio leaderboard shipped a 7.7 MB response
- **Symptom:** Found by running the server and measuring, not by reading code.
  `GET /api/v1/portfolio/risk-ranking` returned **7,704,070 bytes in 7.89s**,
  and the dashboard fetches it unconditionally on page load.
- **Root cause:** The endpoint returns the full `ProjectGovernanceAssessment`
  for all 2,155 projects. `rule_signals` is ~82% of each record (2,918 of
  3,563 bytes) -- six statutory signals each carrying a paragraph of rationale.
  The leaderboard renders none of them; it renders only 300 rows and filters
  client-side on name/agency/sector/id.
- **Stage/module:** `backend/app/api/v1/endpoints/portfolio.py`,
  `frontend/index.html::fetchPortfolioRiskRanking`.
- **Fix:** Added `?slim=true`, which omits `rule_signals`, `shap_drivers` and
  `prescriptive_interventions` and returns a `JSONResponse` directly so
  FastAPI's `response_model` does not re-populate them. The dashboard now uses
  it. Measured: **7.70 MB / 7.89s -> 1.22 MB / 0.015s** (84% smaller). Full
  detail remains available unslimmed and on `/predict/project`.
- **Regression case added:**
  `backend/tests/test_endpoints.py::test_risk_ranking_slim_mode_drops_only_unused_payload`
  -- asserts the saving, that every field the table renders survives, and that
  ordering is unchanged (slim is a projection, not a different query).
- **Status:** fixed / verified

---

## 2026-09-18 — Dashboard "Progress" column was dead for every project
- **Symptom:** Surfaced while verifying the slim payload: `physical_progress`
  was absent from the risk-ranking response. Checking the *unslimmed* response
  showed it had never been there either.
- **Root cause:** `ProjectGovernanceAssessment` did not carry
  `physical_progress`. The dashboard renders
  `p.physical_progress != null ? p.physical_progress + '%' : '-'`, so the
  guard silently took the null branch and the column showed "-" for all 2,155
  rows. A defensive null-check hid a missing field instead of exposing it.
- **Stage/module:** `backend/app/paimana_contracts.py` (schema),
  `backend/app/paimana_engine.py` (population).
- **Fix:** `physical_progress` added to the assessment and populated from the
  `ProjectInput`. The leaderboard now shows real execution state (96.0%, 86.0%,
  92.0% for the top three by Capital-at-Risk).
- **Regression case added:**
  `backend/tests/test_endpoints.py::test_assessment_carries_physical_progress`
  -- asserts the value round-trips and that ranked rows are not all zero, so a
  defaulted field cannot pass as a populated one.
- **Status:** fixed / verified

---

## 2026-09-18 — Dashboard KPI captions contradicted the numbers above them
- **Symptom:** Found by screenshotting the running dashboard. The four headline
  tiles showed the real portfolio (Rs 4,184,701 Cr capex, 34 Critical) while
  the captions directly beneath read **"8 Sample Megaprojects"**, **"3 of 8
  Projects governed by statutory floor"** and **"1 Crit / 2 Mod / 5 Low"**.
- **Root cause:** `fetchPortfolioSummary()` updated only the four big numbers
  by id. The sub-captions were static HTML left over from an early 8-project
  mock and were never wired to the API. The placeholder values baked into the
  tiles (Rs 97,266 Cr, Rs 103.11 Cr, 37.5%, "1 Critical") came from the same
  mock and were visible until the first fetch resolved.
- **Why it matters:** a reviewer reading the small print would conclude the
  platform runs on eight rows rather than 2,155.
- **Stage/module:** `frontend/index.html`.
- **Fix:** Captions given ids and populated from the same payload:
  "2,155 monitored projects", "933 of 2,155 projects", and a full live tier
  breakdown (34 Crit / 453 High / 933 Mod / 735 Low). Static placeholders
  replaced with em-dashes so a failed fetch shows nothing rather than fiction.
- **Regression case added:**
  `test_dashboard_kpi_captions_are_not_hardcoded_sample_text` — asserts the
  stale strings are gone and the live elements exist.
- **Status:** fixed / verified (screenshot confirms all four tiles agree)

---

## 2026-09-18 — Dashboard rendered unstyled without internet
- **Symptom:** The page sourced its entire layout from `cdn.tailwindcss.com`
  plus Google Fonts, with only a 41-line inline `<style>` block locally. With
  no connection the dashboard degrades to an unstyled wall of text — silently,
  with no error.
- **Root cause:** Play-CDN Tailwind was never vendored. At a demo venue with
  blocked or absent wifi this is a total presentation failure.
- **Stage/module:** `frontend/index.html`, `backend/app/main.py`.
- **Fix:** Tailwind vendored to `frontend/vendor/tailwind.play.js` (407 KB) and
  served from a new `/static` mount; the CDN remains only as a guarded fallback
  behind a `window.tailwind` check. Font stacks given real local fallbacks
  (`system-ui`, `ui-monospace`) so blocked webfonts cost typography, never
  layout.
- **Verified:** screenshotted headless Chromium with `cdn.tailwindcss.com`,
  `fonts.googleapis.com` and `fonts.gstatic.com` all resolved to NOTFOUND — the
  dashboard renders fully styled.
- **Regression case added:**
  `test_dashboard_styling_is_served_locally_not_from_a_cdn`.
- **Status:** fixed / verified

---

## 2026-09-18 — Project detail contradicted the list it was opened from
- **Symptom:** Found by checking my own new front end rather than waiting for
  it to be reported. Opening any project from the leaderboard showed a slip
  forecast of ~1% regardless of the project, while the row that launched the
  sheet showed p_model of 0.15, 0.21 or 0.51. The detail view disagreed with
  the list on every project.
- **Root cause:** The sheet rebuilt a `ProjectInput` from the slim leaderboard
  row and re-scored it. The slim payload carries no schedule fields, so
  `months_elapsed`, `original_duration_months` and `current_delay_months` fell
  back to `ProjectInput` defaults (12 / 36 / 0). Those drive
  `months_to_revised_date`, `elapsed_ratio` and `schedule_pressure` — the
  model's strongest features — so every project was scored as a generic
  early-stage project and p_model collapsed to 0.0103.
- **Stage/module:** `hackathon-frontend-starter/lib/api.ts`
  (`toAssessmentInput`) → `components/project-sheet.tsx`.
- **Fix:** Deleted the reconstruction entirely rather than patching it. Added
  `GET /api/v1/portfolio/project/{project_id}`, which returns the full
  assessment plus the horizon curve scored from the project's **own stored
  record**. The two views now agree by construction. Verified exact equality
  (delta 0.0000) on p_model, gov_score, rule_floor and Capital-at-Risk.
- **Regression case added:**
  `test_project_detail_matches_the_leaderboard_row_exactly` — asserts field
  equality against the leaderboard and that the 1-month horizon tracks
  p_model, so the sheet's headline and its own forecast cannot drift apart.
  Plus `test_project_detail_404s_on_unknown_id`.
- **Status:** fixed / verified
