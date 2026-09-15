# Frontend Subsystem: MoSPI PAIMANA Executive Command Center

The frontend provides an institutional, high-density light dashboard for senior secretaries, NITI Aayog advisors, and MoSPI IPMD directors to review project health and run live what-if stress tests.

---

## 1. Design & Typography Standards
- **Institutional Light Aesthetic:** Crisp white (`#ffffff`) surfaces on slate (`#f8fafc`) with sharp 1px structural borders (`#e2e8f0`). Optimized for daylight clarity, projectors, and black-and-white memo printing.
- **Typography:** Inter for system prose and JetBrains Mono for financial figures, percentages, and statistical test outputs.
- **Zero Heavy Framework Footprint:** Zero Node.js / NPM build pipeline required. Native ES6 + Tailwind CDN delivers instant rendering on any standard browser.

---

## 2. Integrated Interactive Workstations
1. **Executive KPI Ribbon:** Portfolio aggregates (Total Sanctioned Capex, Capital-at-Risk, Statutory Override Rate, Distress Distribution).
2. **Interactive What-If Sandbox:** Sliders for Physical Progress, Cumulative Expenditure, Schedule Elapsed Ratio, and Days Since Last Report, plus active legal stay toggles and NLP remarks.
3. **Real-Time Governance Dial:** Displays the non-masking invariant $\text{GovScore} = \max(100 \cdot P_{\text{model}}, \text{RuleFloor})$, Capital-at-Risk, active statutory flags ($F_1$ to $F_5$), and top-5 TreeSHAP factor attributions.
4. **Portfolio CaR Leaderboard:** Filterable and sortable risk ranking table with 1-click project inspection.
5. **DeLong Statistical Protocol:** Interactive test runner for paired AUC comparisons between Stage-Aware XGBoost and Cox-PH baseline.
6. **PRAGATI 1-Page Risk Dossier Modal:** Executive memo formatted for Cabinet Secretariat review with full print stylesheet (`window.print()`).

---

## 3. Connecting to the Backend
The frontend communicates directly with the FastAPI backend at `http://localhost:8000/api/v1/`:
- `GET  /api/v1/portfolio/summary`: Aggregated portfolio statistics
- `GET  /api/v1/portfolio/risk-ranking`: Ranked project table
- `POST /api/v1/predict/project`: Real-time GovScore and CaR computation
- `POST /api/v1/analytics/delong-test`: Paired AUC statistical significance test
