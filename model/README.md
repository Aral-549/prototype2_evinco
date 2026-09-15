# Model Subsystem: PAIMANA Schedule Risk Ensemble

### Artifact: `paimana_schedule_risk_xgboost.pkl`
**Architecture:** Stage-Aware Gradient Boosted Trees (XGBoost Classifier)  
**Binary Classification Target:** `future_schedule_risk` (whether project schedule slips $\ge 6$ months beyond statutory milestone)  
**Optimal Decision Threshold:** $\tau = 0.45$ (maximizing F1/balanced accuracy over Central Sector historical base-rate)

---

## 1. Feature Space & Quarantined Proxies (12 Features)

To withstand scrutiny from Indian Statistical Service (ISS) evaluators and prevent post-facto outcome leakage, the model enforces strict feature segregation:

| # | Feature Name | Description | In-Model Role | Quarantine Status |
|---|---|---|---|---|
| 1 | `physical_progress` | Cumulative physical completion percentage ($0.0 - 100.0\%$) | Active Lead Indicator | Normal |
| 2 | `progress_change` | Monthly physical velocity delta | Active Lead Indicator | Normal |
| 3 | `expenditure` | Cumulative capex disbursement in ₹ Crore | Scale Normalizer | Normal |
| 4 | `expenditure_change_robust` | Recent monthly capital deployment rate | Active Lead Indicator | Normal |
| 5 | `expenditure_pct_robust` | Ratio of disbursement to original sanctioned capex | Spend-to-progress ratio | Normal |
| 6 | `original_cost` | Sanctioned baseline capex ($C_0$) in ₹ Crore | Scale Factor | Normal |
| 7 | `revised_cost` | Anticipated/revised project completion cost | Cost Variance | Normal |
| 8 | `cost_overrun_pct_robust` | Cumulative cost escalation percentage | Financial Strain | Normal |
| 9 | `cost_change_robust` | Recent revision escalation rate | Cost Velocity | Normal |
| 10 | `remaining_progress_pct` | $100.0 - \text{physical\_progress}$ | Execution Float | Normal |
| 11 | `current_delay_months_robust` | Post-facto recorded delay in months | **QUARANTINED** | **Hard-zeroed at inference** |
| 12 | `has_revised_doc` | Bureaucratic revision flag (Date of Completion) | **QUARANTINED** | **Hard-zeroed at inference** |

---

## 2. Leakage Quarantine Guarantee

In administrative project monitoring, an implementing CPSE files a revised Date of Completion (DOC) *after* delay is already acknowledged. A model relying on `has_revised_doc` (34.2% of baseline booster gain) or `current_delay_months_robust` acts as an autopsy report rather than a predictive early warning system.

At inference time, `ModelService.predict_proba()` hard-zeroes both quarantined features. This forces the booster to evaluate structural early indicators (spend-to-physical decoupling and velocity stalls) rather than memorizing administrative filings.

---

## 3. TreeSHAP Local Attribution

Predictions are decomposed using Lundberg & Lee's TreeSHAP algorithm:
$$\ln\left(\frac{P}{1-P}\right) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$
SHAP values ($\phi_j$) are translated in real-time into human-readable English and Hindi administrative directives for senior monitoring committees (e.g. PRAGATI / Cabinet Secretariat).
