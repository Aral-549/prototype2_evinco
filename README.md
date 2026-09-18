# MoSPI PAIMANA • Predictive AI Early-Warning & Capital-at-Risk Platform
### Smart India Hackathon 2026 • Problem Statement: SIH26103
**Target Domain:** Central Sector Infrastructure Mega-Projects (Costing ₹150+ Crore)  
**Beneficiary:** Ministry of Statistics and Programme Implementation (MoSPI) – Infrastructure and Project Monitoring Division (IPMD) & Ministry of Finance  
**Team:** Evinco  

---

## 🌐 Live Cloud Deployment (Try It Live)

The platform is deployed live on an Azure cloud instance with SSL encryption and full mobile/desktop responsiveness:

* **Unified Next.js Decision-Support Platform:** [https://evinco-sih.centralindia.cloudapp.azure.com/paimana](https://evinco-sih.centralindia.cloudapp.azure.com/paimana)
  * `Overview (/paimana)`: Portfolio Capital-at-Risk summary, macro exposure, and ranked priority review list with slide-over project sheets.
  * `Simulator (/paimana/simulator)`: Interactive What-If sandbox to adjust project parameters (spend velocity, progress, disputes) and observe real-time risk scores.
  * `Evidence (/paimana/evidence)`: Statistical audit proving AI superiority via DeLong's paired AUC tests, out-of-time validation, and lead-time analysis.
  * `CUF 2.0 (/paimana/cuf-gap)`: Information-theoretic feature ceiling and structured data reform proposals grounded in CAG audits.
* **Institutional Executive Dashboard (Static / Low-Bandwidth):** [https://evinco-sih.centralindia.cloudapp.azure.com/dashboard](https://evinco-sih.centralindia.cloudapp.azure.com/dashboard) (Direct FastAPI single-page view designed for air-gapped institutional environments).
* **Interactive OpenAPI / Swagger Documentation:** [https://evinco-sih.centralindia.cloudapp.azure.com/docs](https://evinco-sih.centralindia.cloudapp.azure.com/docs)
* **API Portfolio Endpoint:** [https://evinco-sih.centralindia.cloudapp.azure.com/api/v1/portfolio/summary](https://evinco-sih.centralindia.cloudapp.azure.com/api/v1/portfolio/summary)

---

## 1. What This Project Is

India’s central government monitors over **2,000 mega infrastructure projects**—highways, freight corridors, nuclear and thermal power stations, railway links, and urban metro systems—representing more than **₹41 lakh crore in public taxpayer capital**.

Today, the official MoSPI PAIMANA portal and OCMS operate as **retrospective record-keepers**. When an implementing agency files paperwork admitting that a railway line is delayed by three years, the portal turns red. That is not an early warning—**it is an autopsy**. The cost escalation has already happened, contractor claims have mounted, and public capital has been locked up for months.

**Our platform transforms PAIMANA from a retroactive ledger into a predictive early-warning radar:**
1. **Forecasts Schedule Revisions 1 to 6 Months in Advance:** Predicts whether an agency will file a project delay *before* the paperwork is submitted.
2. **Translates Delay into Rupees (Capital-at-Risk):** Converts abstract risk percentages into concrete rupee figures ($\text{CaR}$ in Crores) so the Ministry of Finance and review committees know exactly where fiscal exposure is concentrated.
3. **Explains the Root Causes (TreeSHAP):** Deconstructs each prediction into plain-English administrative factors (e.g. expenditure velocity vs timeline elapsed, agency track record, milestone stall).
4. **Enforces Non-Masking Statutory Redlines:** Combines machine learning with deterministic statutory rule floors—meaning an algorithm can never "pardon" or dilute a mandatory statutory violation.
5. **Formulates the CUF 2.0 Policy Reform:** Identifies institutional reporting blindspots and proposes structural data collection improvements grounded in CAG audits.

---

## 2. The Headline Results (Empirically Measured & Replicable)

Every performance figure in this documentation was measured by the training pipeline against **11,531 observed project-month transitions** across **1,807 live MoSPI mega-projects**, harvested from actual portal freeze snapshots:

### A. Multi-Horizon Discrimination
| Forecast Horizon | AUC (Unseen Future Projects) | Precision / Recall @ Operational Threshold | Base Rate |
|---|---|---|---|
| **1 Month Ahead** | **0.8665** | 0.63 / 0.59 @ 0.70 | 0.179 |
| **3 Months Ahead** | **0.8998** | 0.79 / 0.83 @ 0.50 | 0.450 |
| **6 Months Ahead** | **0.9258** | 0.90 / 0.92 @ 0.35 | 0.691 |

### B. Proving AI Beats Classical Statistics (DeLong's Paired AUC Test)
We benchmarked our Stage-Aware Gradient Boosting model against standard statistical and actuarial baselines on the exact same project cohorts using **DeLong’s paired AUC test**:

| Model Architecture | 1-Month AUC | 3-Month AUC | 6-Month AUC | DeLong Test vs Proposed ($p$-value) |
|---|---|---|---|---|
| **Stage-Aware XGBoost (Proposed)** | **0.8665** | **0.8998** | **0.9258** | — |
| **Logistic Regression (ElasticNet)** | 0.7634 | 0.8471 | 0.9023 | $Z = 23.42$ ($p < 10^{-15}$) |
| **Discrete Proportional Hazards (cloglog)** | 0.7607 | 0.8339 | 0.9028 | $Z = 23.18$ ($p < 10^{-15}$) |
| **Deadline Proximity Heuristic** | 0.7560 | 0.8183 | 0.8641 | $Z = 21.05$ ($p < 10^{-15}$) |
| **Deterministic RuleFloor Alone** | 0.4187 | 0.4133 | 0.3650 | $Z = 38.12$ ($p < 10^{-15}$) |

*At 1 month, the machine learning model achieves a **+0.103 AUC lift** over both classical statistical baselines.*

### C. Measured Early-Warning Lead Time
Walking each project's timeline with out-of-fold predictions:
* At operational threshold 0.30, the model fires its first warning a **median of 2 months before** the revised completion date appears on official MoSPI records.
* Performance: **78% project-level precision, 90% recall, 38% false alarm rate**.
* At stricter threshold 0.70, precision climbs to **89%** with a **13% false alarm rate** at a median 1-month lead time.

### D. True Probability Calibration (ECE < 0.01)
Raw machine learning scores often suffer from overconfidence. Because Capital-at-Risk directly multiplies predicted probability by crores of expenditure ($\text{CaR} = C_0 \times P \times \dots$), probability miscalibration would directly distort fiscal risk allocation.
* We applied **Isotonic Regression calibration** cross-fitted over project-grouped folds.
* **1-Month Expected Calibration Error (ECE):** Reduced from **0.1531 $\rightarrow$ 0.0034** (a 45-fold improvement).
* **3-Month ECE:** Reduced from **0.0330 $\rightarrow$ 0.0084**.

---

## 3. What We Did in the Generated Reports

The platform produces six distinct analytical reports and decision artifacts designed specifically for senior policymakers, project monitoring committees, and auditors:

```
                                  [ EXECUTIVE DECISION COCKPIT ]
                                                 │
      ┌──────────────────┬───────────────────────┼───────────────────────┬──────────────────┐
      ▼                  ▼                       ▼                       ▼                  ▼
 [ Portfolio CaR ]  [ 4-Tier Risk ]    [ Multi-Horizon ]          [ TreeSHAP ]       [ CUF 2.0 Reform ]
  ₹1.48 Lakh Cr      Stratification     Slip Curves (1/3/6m)       Root Causes        CAG Gap Analysis
  Fiscal Exposure   CRITICAL/HIGH/...   Calibrated Probabilities   Explainable AI     Policy Roadmap
```

### Report 1: Executive Portfolio Risk & Capital-at-Risk (CaR) Rollup
* **Monitored Scope:** Aggregates ₹41,12,000+ Crore in sanctioned Capex across 2,155 live central projects.
* **Priced Exposure:** Computes total national Capital-at-Risk at **₹1,48,774 Crore**.
* **Formula Grounding:**
  $$\text{CaR} = C_{\text{original}} \times P_{\text{model}} \times \max\left(\text{Current Overrun \%}, \, \text{Sector Median Overrun \%}\right)$$
  *(Using the sector-median overrun solves the early-stage blindspot where a brand-new project with no filed overrun yet would otherwise appear deceptively risk-free).*
* **Sector Rollups:** Instant capital-at-risk ranking across Railways, Road Transport & Highways, Power, Petroleum, and Urban Development.

### Report 2: Ranked Project Leaderboard with 4-Tier Stratification
Ranks all monitored infrastructure projects into actionable management tiers:
* **CRITICAL ($\text{GovScore} \ge 75$):** Immediate cabinet/secretariat intervention required.
* **HIGH ($50 \le \text{GovScore} < 75$):** Active schedule distress; project committee review needed.
* **MODERATE ($25 \le \text{GovScore} < 50$):** Minor milestone drift; automated monitoring.
* **LOW ($\text{GovScore} < 25$):** On-track; healthy capex expenditure velocity.

### Report 3: Multi-Horizon Slip Curves
For every individual project (e.g. *Chenab Rail Bridge, Mumbai Metro Line 3, NH-66 Four-Laning*), the platform plots a calibrated probability progression:
* Probability of schedule slip within **30 days** ($P_{1m}$).
* Probability of schedule slip within **90 days** ($P_{3m}$).
* Probability of schedule slip within **180 days** ($P_{6m}$).

### Report 4: TreeSHAP Feature Attribution (Explainable AI)
A black-box prediction is inadmissible in inter-ministerial disputes. For every flagged project, our report breaks down the exact marginal contribution of each engineering feature:
* **Expenditure vs Time Pacing Gap:** Ratio of cumulative financial spend to elapsed contractual duration.
* **Agency Historical Latency:** Track record of the executing PSU/contractor across prior completed projects.
* **Milestone Inaction Run:** Number of consecutive months with zero physical progress.
* **Statutory Rule Overrides:** Explicit logging of any triggered statutory boundaries.

### Report 5: The GovScore Non-Masking Lattice
$$\text{GovScore} = \max\left(100 \cdot P_{\text{model}}, \, \text{RuleFloor}\right)$$
The supremum guarantees the **Non-Masking Invariant**: an optimistic machine learning model output can never dilute or mask an established statutory breach (such as a project exceeding 150% of its sanctioned completion timeline or 6 months of zero expenditure). Conversely, a clean administrative paper trail cannot suppress a predictive early warning.

### Report 6: CUF 2.0 Policy Reform Audit (MoSPI Data Gap Analysis)
An audit deliverable accessible at `/api/v1/analytics/cuf-gap`:
* Audited 14,917 raw public records and discovered that narrative fields (`Remarks`, `RevisedDateReason`, `RevisedCostReason`) are 100% null in the public feed.
* Outlines the **CUF 2.0 schema specification**, detailing four high-value fields recommended for mandatory collection:
  1. *Right-of-Way (RoW) unencumbered percentage at financial sanction.*
  2. *Stage-II Forest and Wildlife clearance status.*
  3. *Structured dispute and arbitration status.*
  4. *Reporting recency and contractor milestone sign-off timestamp.*

---

## 4. Engineering Hardening & Traps Avoided

### The "Hidden Snapshots" Breakthrough: An Observed Time Axis
The standard PAIMANA portal export returned all records for a project in an undated blob (`Month: null`, `Year: null` on 14,917 records). Previous teams attempted to reconstruct the time axis by sorting by cumulative expenditure.
* **The Flaw:** We proved that cumulative expenditure *is not monotonically increasing*. In several audited projects, expenditure was revised downward due to de-scoping or accounting adjustments, silently corrupting the time sequence!
* **The Solution:** We inspected network traffic and uncovered two unadvertised portal endpoints:
  `GET /Home/GetFreezeDates` and `POST /Home/GetTileData?MonthYear=...`
  This yielded **18,601 records across 13 verified monthly freezes**, giving India's infrastructure monitoring system a clean, ground-truth observed monthly time axis.

### Honest Target Leakage Policy
In typical hackathon models, the single strongest feature predicting "will this project be delayed" is `has_revised_doc`. But that document is only filed *after* the delay is already a documented fact.
* **The V1 Trap:** The early prototype zeroed this feature at inference time. This caused severe train/serve skew: the model was trained relying on `has_revised_doc` for 34% of its gain, then starved of it in production, causing predicted risk to collapse to near-zero.
* **The V2 Resolution:** We strictly purged all post-facto fields from the design matrix at training time. This is verified by an automated CI assertion (`assert_no_leakage`) in `scripts/train_model.py`.

---

## 5. Technology Stack

| Component | Technologies | Implementation |
|---|---|---|
| **Predictive Engine** | Python 3.12, XGBoost, Scikit-learn, SciPy | Multi-horizon gradient boosting with isotonic calibrators. |
| **Statistical Baselines** | DeLong Paired AUC, Discrete Hazard Model (cloglog) | Formulated directly in NumPy/SciPy without external bloat. |
| **Explainability** | TreeSHAP, SHAP TreeExplainer | Generates fast, game-theoretic feature contribution vectors. |
| **Backend & APIs** | FastAPI, Pydantic v2, Uvicorn | Sub-15ms inference latency; auto-generated Swagger UI. |
| **Frontend Platform** | Next.js 16, React 19, TypeScript, Tailwind CSS, Framer Motion | High-density executive command cockpit mounted at `/paimana`. |
| **Data Ingestion** | Requests, Pandas, JSON schema validators | Automated monthly harvester querying MoSPI freeze snapshots. |

---

## 6. Zero-Friction Local Setup (Run It Locally)

If you wish to run and evaluate the platform locally on your own machine:

### Prerequisites
* Python 3.10+ (tested on Python 3.11 & 3.12)
* Node.js 18+ & npm
* Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Aral-549/prototype2_evinco.git
cd prototype2_evinco
```

### Step 2: Setup and Start the FastAPI Backend
```bash
# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt

# Start the FastAPI backend on port 8001
python run.py
```
*The backend starts at **`http://localhost:8001`** with all models, pre-computed DeLong test matrices, and 2,155 live MoSPI projects loaded into memory.*

### Step 3: Start the Next.js Frontend
In a new terminal window:
```bash
cd prototype2_evinco/hackathon-frontend-starter
npm install
npm run dev
```

Open your browser at **`http://localhost:3000/paimana`** (or **`http://localhost:8001/dashboard`** for the built-in HTML executive dashboard).

### Step 4: Run Automated Tests
```bash
PYTHONPATH=backend pytest backend tests -q
```
*Executes all 41 golden test fixtures, verifying statistical invariants, non-masking boundaries, and leak-free transformations.*

---

## 7. Repository Structure

```
prototype2_evinco/
├── backend/app/                  # FastAPI Application Core
│   ├── api/v1/                   # REST endpoints (portfolio, predict, analytics, health)
│   ├── paimana_car.py            # Capital-at-Risk economic pricing model
│   ├── paimana_engine.py         # GovScore lattice & non-masking supremum
│   ├── paimana_rules.py          # Deterministic statutory flags (F1 through F6)
│   ├── paimana_shap.py           # TreeSHAP root-cause driver extractor
│   ├── paimana_statistics.py     # DeLong paired AUC test implementation
│   └── services/                 # Multi-horizon inference & calibration services
├── hackathon-frontend-starter/   # Modern Next.js 16 / React 19 Executive UI
│   ├── app/                      # App router (Overview, Evidence, Project detail sheet)
│   ├── components/               # Attention lists, metrics, risk tier chips
│   └── lib/                      # Type-safe API client
├── model/                        # Measured Empirical Artifacts
│   ├── paimana_model_metrics.json       # Ground-truth AUC, Brier & DeLong test values
│   ├── paimana_lead_time.json           # Backtested lead-time distributions
│   ├── paimana_schedule_risk_v2.pkl     # Serialized XGBoost multi-horizon models
│   └── paimana_oof_predictions.csv      # Out-of-fold predictions for external auditing
├── data/                         # Harvested live MoSPI dataset & panel transitions
├── scripts/                      # Complete reproducible pipeline scripts
├── contracts/                    # Design specifications written before code
├── BUGLOG.md                     # Engineering hardening & bugfix log
└── README.md                     # This documentation
```

---

## 8. Hackathon Submission Summary

* **Project:** MoSPI PAIMANA Predictive Early-Warning Platform (PS SIH26103)
* **Team Name:** Evinco
* **Live Online Demo:** [https://evinco-sih.centralindia.cloudapp.azure.com/paimana](https://evinco-sih.centralindia.cloudapp.azure.com/paimana)
* **Executive Cockpit:** [https://evinco-sih.centralindia.cloudapp.azure.com/dashboard](https://evinco-sih.centralindia.cloudapp.azure.com/dashboard)
* **Contact Email:** `shaik2.mitmpl2025@learner.manipal.edu`
