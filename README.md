# MoSPI PAIMANA • Predictive Early-Warning Platform
### Smart India Hackathon 2026 | Problem Statement: SIH26103
**Target Domain:** Central Sector Infrastructure Monitoring (Projects costing ₹150+ Crore)
**Ministry:** Ministry of Statistics and Programme Implementation (MoSPI) – IPMD

---

## What this is

India's central government monitors roughly 2,000 mega infrastructure projects — highways, railway corridors, power plants, urban metro lines — representing over ₹41 lakh crore in public capital.

The PAIMANA portal and OCMS today work as **retrospective record-keepers**. When an agency files paperwork revising a completion date by three years, the dashboard turns red. That isn't an early warning — it's an autopsy.

This platform forecasts that revision **before it is filed**, prices the exposure in rupees, and hands a review committee a ranked, explainable agenda.

**Every performance number in this README was measured by the training pipeline in this repository and can be reproduced with two commands.** None of them are asserted. The measurement scripts are in `scripts/`, the artifacts they emit are in `model/`, and the API refuses to report a figure that has not been measured.

---

## The headline result

Trained and evaluated on **11,531 real project-month transitions** from **1,807 live MoSPI projects**, on an **observed** monthly time axis harvested from the portal's own data freezes:

| Forecast horizon | AUC (unseen projects) | AUC (out-of-time) | Precision / recall @ op | Base rate |
|---|---|---|---|---|
| **1 month** | **0.8665** | **0.8296** | 0.63 / 0.59 @ 0.70 | 0.179 |
| **3 months** | **0.8998** | **0.9075** | 0.79 / 0.83 @ 0.50 | 0.450 |
| **6 months** | **0.9258** | n/a¹ | 0.90 / 0.92 @ 0.35 | 0.691 |

¹ At 6 months the out-of-time cohort is structurally 100% positive — a recent row survives censoring only if it already slipped — so AUC is undefined. The artifact says exactly that rather than omitting the row.

Measured against classical statistics on the identical cohort and split, with **DeLong's paired AUC test**:

| Model | 1m | 3m | 6m |
|---|---|---|---|
| Stage-Aware XGBoost (proposed) | **0.8665** | **0.8998** | **0.9258** |
| Logistic Regression (ElasticNet) | 0.7634 | 0.8471 | 0.9023 |
| Discrete-Time Proportional Hazards (cloglog) | 0.7607 | 0.8339 | 0.9028 |
| Deadline proximity heuristic | 0.7560 | 0.8183 | 0.8641 |
| Deterministic RuleFloor (the rules already shipped) | 0.4187 | 0.4133 | 0.3650 |
| Class prior (sanity floor) | 0.4869 | 0.4929 | 0.4873 |

At 1 month the margin over both classical baselines is **+0.103** (DeLong Z = 23.42 vs logistic, Z = 23.18 vs survival; p < 1e-15).

**Measured early warning.** Walking each project's timeline with out-of-fold scores, at threshold 0.30 the model raises its first alert a **median of 2 months** before the revised date appears on the record, at **78% project-level precision, 90% recall, 38% false alarms**. Tightening to 0.70 gives 89% precision and 13% false alarms at a median 1-month lead.

**Calibration** (isotonic, ECE cross-fitted over project-grouped folds): 1m **0.1531 → 0.0034**, 3m 0.0330 → 0.0084, 6m 0.0424 → 0.0144. This matters because Capital-at-Risk multiplies `p_model` by crores.

Reproduce all of it:
```bash
python scripts/harvest_monthly.py      # 13 freeze months from the live portal
python scripts/build_panel.py && python scripts/train_model.py && python scripts/lead_time_backtest.py
```

---

## What makes this different

### 1. We took the time axis from the portal instead of inferring it

The PAIMANA dashboard exposes two endpoints nobody was using:

```
GET  /Home/GetFreezeDates              -> {"firstFreeze":"2025-07","lastFreeze":"2026-07"}
POST /Home/GetTileData  MonthYear=...  -> that freeze month's snapshot only
```

The original harvester left `MonthYear` empty, so the portal returned every record for a project in one undated blob — `Month: null`, `Year: null` on all 14,917 of them. The panel then had to *reconstruct* the time axis from quantities assumed to move only forward.

Querying month by month instead gives **18,601 records across 2,243 projects and 13 real freeze months**, each stamped with the month it belongs to. The effect on the panel:

| | reconstructed axis | **observed axis** |
|---|---|---|
| labelled transitions | 8,838 | **11,531** |
| projects | 1,631 | 1,807 |
| projects dropped for unreliable ordering | 12 | **0** |
| 1-month AUC | 0.8541 | **0.8665** |
| margin over logistic regression | +0.056 | **+0.103** |
| calendar out-of-time split | invalid (confounded) | **valid, 0.8296** |

It also showed the old assumption was not merely unverifiable but **wrong for some projects**: one probe project's cumulative expenditure *fell* from 160.48 in 2025-07 to 155.72 in 2025-10, so ordering by expenditure had silently mis-sequenced it. And `gap_months == 1` on 11,506 of 11,531 transitions — the reporting cadence really is monthly, which was previously an assumption.

The label itself is unchanged and still observed rather than invented: **does the officially declared completion date move at the next monthly report?** 17.9% positive.

One trap worth naming. Two distinct months in which nothing happened on site are byte-identical on every measurement field, so a dedupe key that omits the reporting month silently deletes a real month of history. `tests/golden/test_dated_panel_golden.py` fails loudly if that is ever reintroduced — verified by reverting the fix and watching it fail.

### 2. Honest leakage handling — and the bug it fixed

Most PAIMANA baselines leak: the single strongest predictor of "was this project delayed" is `has_revised_doc`, filed *after* the delay is already fact.

The previous version of this platform claimed to solve that by **zeroing the feature at inference**. That is not a leakage fix — it is train/serve skew. The booster was trained with `has_revised_doc` carrying 34% of its gain, then fed zeros in production. The result, measured across all 2,155 live projects:

| | v1 (zeroed at inference) | v2 (excluded from training) |
|---|---|---|
| `p_model` range | 0.0001 – 0.0345 | 0.0103 – 0.8533 |
| median `p_model` | 0.0040 | 0.1231 |
| projects where ML drives the score | **0** | 1,222 of 2,155 |
| total Capital-at-Risk | ₹2,985 Cr | **₹1,48,774 Cr** |

The ML branch was inert. Every "high risk" project was a rule firing. The fix is to **exclude leaky fields from the design matrix at training time**, which `scripts/train_model.py` enforces in code (`assert_no_leakage`), not by convention.

The line is drawn deliberately. `current_delay_months` — slippage *already on the record* at time t — **is** used: a desk officer reading the file that day can see it, so it is state, not outcome. What is excluded is whether the date moves *again*, which is the label.

### 3. The statutory rules are not a forecast — and we measured that

Scored against the real outcome, the shipped rule set F1+F2 achieves **AUC 0.4187 — worse than random**, degrading to 0.3650 at six months.

This is not a claim that the rules are wrong. They detect chronic governance breaches, which is a legitimate and different question from forecasting. It *is* a claim that the platform should stop implying they are an early-warning system.

Measuring them surfaced what the rule set was missing — and then the corrected time axis **revised our own answer downward**, which is worth showing rather than hiding:

| "Declared completion date has passed" | reconstructed axis | **observed axis** |
|---|---|---|
| coverage | 32.1% of project-months | **8.3%** |
| lift over base rate | 4.3× | **1.38×** |
| AUC standing alone | 0.725 | **0.5170** |

The reconstructed axis had been systematically placing projects further past their deadlines than they really were, which inflated the evidence we originally used to justify flag **F6**. On real dates, "overdue" is a far weaker signal than we reported.

What the corrected data shows instead is that risk peaks *just before* the deadline, not after it:

| Months to declared completion date | Rows | Slip rate |
|---|---|---|
| ≥12 months overdue | 529 | 0.2911 |
| 6–12 months overdue | 413 | 0.2736 |
| 0–3 months overdue | 545 | 0.1798 |
| **0–3 months remaining** | **2,462** | **0.4655** |
| 3–6 months remaining | 1,850 | 0.1200 |
| 6–12 months remaining | 2,245 | 0.0704 |
| >24 months remaining | 1,094 | 0.0128 |

The agency files the revision as the deadline comes into view, not long after it lapses — a 36× spread between the imminent-deadline band and the distant one.

**F6 is kept exactly as written and was not retuned.** "The declared date has elapsed while work is incomplete" remains a true statutory fact worth flagging, and quietly rewriting a rule to chase a freshly-measured correlation is the same error we refused to make with F1. The corrected numbers are recorded in `BUGLOG.md` and `contracts/rule_floor.md`; an imminent-deadline flag is logged there as a **candidate for human review**, not something the model author adds unilaterally.

---

## Three challenges we put to our own model

A reviewer will ask these. We measured them rather than waiting to be asked, and each one is a live endpoint.

### 1. "Isn't this just a deadline rule?"

`months_to_revised_date` scores **AUC 0.756 on its own**, so this is a fair challenge. We made the objection a first-class competitor: a fitted deadline heuristic is a benchmark row, and every model is re-scored on the cohort that actually matters, where the declared date has **not yet passed** and a warning can still change the outcome.

| Cohort (1-month horizon) | n | base rate | XGBoost | Logistic | Deadline rule | Margin |
|---|---|---|---|---|---|---|
| Already overdue | 1,785 | 0.244 | 0.7709 | 0.6269 | 0.5710 | +0.1999 |
| **Not yet overdue (actionable)** | 9,746 | 0.167 | **0.8796** | 0.7982 | 0.8245 | +0.0551 |
| ≥3 months runway left | 7,284 | 0.066 | **0.7868** | 0.6466 | 0.6712 | +0.1156 |
| ≥6 months runway left | 5,434 | 0.048 | **0.7737** | 0.6077 | 0.6405 | +0.1332 |

The model's margin **grows as the problem gets harder**. On projects with six or more months of declared runway — needles in a haystack at a 4.8% base rate — it holds 0.7737 while the deadline rule falls to 0.6405 and logistic regression to 0.6077. Its edge over logistic there is **+0.166**, far larger than the +0.103 headline gap.

That also answers the second-order objection: the ML lift over classical statistics is modest on easy cases and large on hard ones, which is exactly where it earns its place. → `GET /api/v1/analytics/benchmark-baseline`

### 2. "Your time axis is reconstructed — how much rests on that?"

**This objection no longer applies, and that is the single biggest improvement in the project.** The panel now uses the portal's own freeze months, so order is observed rather than inferred.

The robustness study that answered it while the axis *was* reconstructed is retained as evidence the earlier approach was sound (`GET /api/v1/analytics/ordering-sensitivity`, now marked `superseded`): across four plausible ordering rules the AUC moved by only **0.0036**, while a deliberately inverted negative control cost **0.1711**.

Switching to observed dates then *improved* the result — 1-month AUC 0.8541 → 0.8665, with the margin over logistic regression widening from +0.056 to +0.103 — and enabled a genuine calendar out-of-time split that was previously invalid.

### 3. "You're predicting paperwork, not concrete."

Also fair — the label is a date moving on a form. We measured whether that bureaucratic event tracks physical reality, and on the corrected time axis the answer got **weaker**, which we report rather than bury:

| Indicator | group | Cohen's d | effect |
|---|---|---|---|
| `schedule_pressure` | schedule-relative | **+0.2192** | small |
| `velocity_deficit` | schedule-relative | +0.0854 | negligible |
| `progress_delta` | absolute activity | −0.054 | negligible |
| `spend_progress_gap` | spend pattern | −0.324 | medium |

Verdict: **PARTIALLY GROUNDED — 1 of 2 schedule-relative indicators corroborate the label.** On the reconstructed axis both did (d = +0.3540 and +0.1110); on real dates only `schedule_pressure` survives. Claims about predicting real-world delay are hedged accordingly.

Two further limits stand: slipping projects are *not* less active, they are further along (75.4% vs 51.0% mean progress) and closer to a deadline they can see they will miss; and projects whose agencies never revise their dates are invisible to this label, uncorrected.

→ `GET /api/v1/analytics/label-validity`

**And the finding that came out of it** — rule F1's own statutory premise is *inverted* for schedule forecasting, and the corrected axis makes it sharper:

| Spend-vs-progress band | project-months | slip rate |
|---|---|---|
| Spend ≥25pp ahead of progress (**F1 fires**) | 1,578 | **0.0716** |
| Roughly aligned | 6,324 | 0.1420 |
| Progress ≥25pp ahead of spend (**opposite of F1**) | 3,629 | **0.2904** |

The condition F1 flags slips at a quarter the rate of the exact opposite condition — a **4.1× inversion**, and the mechanical explanation for the rule set's AUC of 0.4187.

We did **not** flip F1. It encodes a real financial-irregularity test under GFR 2017 Rule 159, and inverting a statutory rule to chase a correlation would be precisely the confusion this platform exists to prevent. What changed is the claim: F1 is a financial flag, never a schedule predictor, and no interface presents it as one.

---

## Where the data does not support a claim, we say so

These are stated plainly because a reviewer will check, and because they are the actual argument for CUF 2.0:

- **Delay narratives do not exist in the public feed.** `Remarks`, `RevisedDateReason` and `RevisedCostReason` are null on **14,917 / 14,917** records. The NLP bottleneck-proxy extractor is implemented and unit-tested but has no input, so `/analytics/cuf-gap` returns the sentinel `-1.0` for every augmentation figure with the reason attached. An earlier version of this platform reported a +5.8pp AUC gain from text that is not in the data.
- **State is never populated.** `StateName` is null on all 14,917 records, so no state-level analysis is possible from the public portal.
- **`DELAYED_TIME` and `COST_OVERRUN_PERC` are zero on every record**, so delay is derived from the original-vs-revised date gap and no cost-escalation regression is offered (`cost_regression_r2_ceiling` is returned as `-1.0`).
- **Flags F3, F4 and F5 are structurally inert on public data** — clearance pendency, reporting staleness and dispute status are not published, so they fire on zero of 2,155 projects. They are correct rules against the full internal CUF; their inertness is the evidence for the schema proposal, not something to hide.
- **A calendar out-of-time split is invalid on this panel** and is deliberately not used. Because each project's newest snapshot is stamped as the harvest month, history length determines how far back its rows reach: short-history projects (~33% positive) populate only recent months, long-history ones (~5–16% positive) the early ones. Splitting on the calendar would measure that artifact. `model/paimana_model_metrics.json` records the rejection under `rejected_split`.

---

## Guarding against our own mistake

History length correlates strongly with the label on this panel, which makes it easy to build a model that scores well by reading an artifact. So every training run reports **AUC within history-length strata**:

| History length | Rows | Positive rate | 1m AUC |
|---|---|---|---|
| 2–5 snapshots | 431 | 0.181 | 0.8694 |
| 6–8 snapshots | 5,185 | 0.289 | 0.8408 |
| 9–11 snapshots | 640 | 0.133 | 0.8542 |
| 12–20 snapshots | 5,275 | 0.077 | 0.8360 |

Discrimination holds at 0.836–0.869 inside every stratum, so the model is reading project dynamics, not the artifact.

This guard earned its place. It caught a contaminated feature **in our own v2 model**: a sector frequency encoding that ranked as the highest-gain input (0.162) and scored AUC 0.712 alone — but correlated −0.703 with history length, and inside strata collapsed to 0.34–0.56, flipping direction between them. It was removed. Cost: 1-month AUC 0.869 → 0.854. Logged in `BUGLOG.md`.

---

## Answering MoSPI's two sponsor questions

### Dimension (b): does AI actually beat classical statistics?

**Yes, and here is the test.** Diebold-Mariano is not applicable — AUC is a combinatorial rank-concordance metric over case-control pairs and does not decompose into an additive point-wise loss differential on a single time axis. We use **DeLong's paired AUC test** (DeLong, DeLong & Clarke-Pearson, *Biometrics* 1988) on the same cohort.

The survival baseline is a **discrete-time proportional hazards model with a complementary log-log link** (Prentice & Gloeckler 1978), not continuous-time Cox. On a monthly panel with heavy ties, Cox's partial likelihood degrades; the cloglog specification is the exact discrete analogue of the proportional-hazards assumption and is therefore the fair comparator. It is fitted directly with scipy, so it needs no extra dependency and runs offline. If `lifelines` is installed a continuous-time Cox row is added; if not, the artifact records `status: "unavailable"` rather than inventing a number.

Result at 1 month: XGBoost 0.854 vs survival 0.793 (Z = 15.15) and logistic 0.798 (Z = 14.87), both p < 1e-15. The margin is real but modest — the classical models are genuinely competitive, and the artifact says so.

### Dimension (c): how much is missing from current CUF fields?

The honest answer is that **an exact variance decomposition over variables that were never recorded is not computable**, and we refuse to invent one.

What *is* measurable: the observable CUF feature set saturates at **AUC 0.854** out-of-fold on next-report slip. What is *not* measurable here: any gain from the delay narratives, because the public feed publishes none (14,917/14,917 null).

The deliverable is therefore the **CUF 2.0 structured-field proposal** at `/api/v1/analytics/cuf-gap`, each field grounded in a named CAG audit or official report — Right-of-Way unencumbered at sanction, Stage-II forest clearance state, structured dispute status, and reporting recency. Three of the platform's own statutory flags are inert precisely because those fields are missing, which is the strongest possible argument for collecting them.

---

## Architecture

```
├── contracts/                 # Specs written BEFORE code (input -> expected output)
│   ├── monthly_harvest.md     #   dated harvest: the portal's own freeze months
│   ├── panel_builder.md       #   observed/reconstructed time axis + label definition
│   ├── model_training.md      #   splits, baselines, leakage policy, calibration
│   ├── lead_time_backtest.md  #   how early-warning lead time is measured
│   ├── inference_adapter.md   #   CUF snapshot -> model design matrix
│   └── rule_floor.md          #   statutory flags F1..F6, with measured evidence
│
├── scripts/                   # The reproducible pipeline
│   ├── harvest_monthly.py     # harvest 13 freeze months (observed time axis)
│   ├── download_real_paimana_data.py  # legacy undated harvest (superseded)
│   ├── build_panel.py         #   raw -> 8,838-row labelled panel + diagnostics
│   ├── train_model.py         #   fit, calibrate, benchmark, DeLong -> artifacts
│   └── lead_time_backtest.py  #   measured lead time vs. filed revisions
│
├── model/                     # Emitted artifacts (the single source of truth)
│   ├── paimana_schedule_risk_v2.pkl    # multi-horizon bundle + calibrators
│   ├── paimana_model_metrics.json      # every reported number originates here
│   ├── paimana_lead_time.json          # lead-time sweep
│   ├── paimana_label_validity.json     # label grounding + F1 inversion
│   ├── paimana_ordering_sensitivity.json # time-axis robustness
│   └── paimana_oof_predictions.csv     # out-of-fold scores for audit
│
├── backend/app/
│   ├── paimana_rules.py       # deterministic flags F1..F6
│   ├── paimana_engine.py      # GovScore = max(100 * P_model, RuleFloor)
│   ├── paimana_car.py         # Capital-at-Risk in crores
│   ├── paimana_statistics.py  # DeLong paired AUC test
│   └── services/
│       ├── feature_adapter.py # CUF snapshot -> design matrix (approximations named)
│       └── model_service.py   # multi-horizon inference + calibration
│
├── tests/golden/              # FROZEN hand-verified ground truth (41 cases)
├── BUGLOG.md                  # every bug found, with its regression case
└── data/                      # harvested MoSPI data + reconstructed panel
```

---

## The GovScore lattice

```
GovScore = max(100 · P_model, RuleFloor)
```

The supremum guarantees the **Non-Masking Invariant**: an optimistic model output can never suppress an established statutory breach, and a clean paper trail can never suppress a predictive alert. Risk bands: `[0,25) Low | [25,50) Moderate | [50,75) High | [75,100] Critical`.

**Capital-at-Risk** converts rank into rupees, because ranking 2,000 projects by probability alone does not help the Ministry of Finance allocate attention:

```
CaR = C₀ × P_model × max(current overrun %, sector median overrun %)
```

The sector-median prior solves the early-stage blindspot, where a project with no filed overrun yet would otherwise appear risk-free.

---

## Quickstart

```bash
./setup.sh          # macOS / Linux  (setup.bat on Windows)
./run.sh            # starts on http://localhost:8000
```

Manual:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python scripts/build_panel.py && python scripts/train_model.py && python scripts/lead_time_backtest.py
python scripts/label_validity.py           # is the label physical? (fast)
python scripts/ordering_sensitivity.py     # time-axis robustness (~2 min, retrains 5x)
python run.py
```

The pipeline is seeded (`--seed 20260917`) and deterministic: the panel is byte-identical and the metrics artifact is identical across runs.

### Endpoints

| What | URL |
|---|---|
| Executive dashboard | `/dashboard` |
| API docs (Swagger) | `/docs` |
| Model state & leakage policy | `/api/v1/health` |
| Portfolio ranked by Capital-at-Risk | `/api/v1/portfolio/risk-ranking` |
| **Measured benchmark vs classical stats** | `/api/v1/analytics/benchmark-baseline` |
| **Full evaluation artifact (audit everything)** | `/api/v1/analytics/model-metrics` |
| **Measured early-warning lead time** | `/api/v1/analytics/lead-time` |
| **Model vs. a deadline rule, by cohort** | `/api/v1/analytics/benchmark-baseline` |
| **Robustness to the reconstructed time axis** | `/api/v1/analytics/ordering-sensitivity` |
| **Is the label physical? (+ the F1 inversion)** | `/api/v1/analytics/label-validity` |
| **1/3/6-month slip curve for a project** | `POST /api/v1/analytics/horizons` |
| TreeSHAP drivers for a project | `POST /api/v1/analytics/drivers` |
| CUF 2.0 gap analysis | `/api/v1/analytics/cuf-gap` |
| DeLong test on your own score vectors | `POST /api/v1/analytics/delong-test` |

---

## Tests

```bash
PYTHONPATH=backend pytest backend tests -q
```

**239 passing** — 155 service/API tests plus **84 frozen golden cases** in `tests/golden/`:

| Golden suite | Cases | Pins |
|---|---|---|
| `test_rule_floor_golden.py` | 22 | the F1–F6 flag table, weights, boundaries, and the clamp |
| `test_panel_builder_golden.py` | 19 | the label definition, duplicate collapsing, ordering, horizon censoring |
| `test_inference_adapter_golden.py` | 12 | CUF→matrix derivations, plus **train/serve parity** |
| `test_lead_time_golden.py` | 12 | the lead-time arithmetic (including the `+1`) and its refusals |
| `test_cohort_verdict_golden.py` | 10 | that the "is it just a deadline rule?" check can return **yes** |
| `test_dated_panel_golden.py` | 9 | the observed time axis, incl. the dedupe trap that would delete a real month |

Golden cases are hand-computed from the contracts, not recorded from implementation output, and are never regenerated.

---

## Data provenance

- **Source:** [https://paimana-proj.mospi.gov.in/](https://paimana-proj.mospi.gov.in/) (cited in SIH26103)
- **Raw harvest:** 14,917 monthly records across 2,155 projects, ₹41,84,701 Cr aggregate sanctioned capex
- **Reconstructed panel:** 8,838 labelled transitions from 1,631 projects (SHA-256 prefix `4a29176a6eeafa13`)
- **Re-harvest:** `python scripts/download_real_paimana_data.py`
