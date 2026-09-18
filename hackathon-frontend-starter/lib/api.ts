/**
 * Typed client for the PAIMANA FastAPI backend.
 *
 * Every figure this UI shows comes from a measured artifact on the server.
 * Nothing here invents, defaults or interpolates a number: when the backend
 * has not measured something it returns 503 or a -1 sentinel, and the UI is
 * expected to say so rather than render a plausible-looking blank.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ??
  (typeof window === "undefined" ? "http://127.0.0.1:8001" : "");

export type RiskTier = "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
export type DominantSource = "MACHINE_LEARNING" | "RULE_FLOOR_OVERRIDE";

export interface PortfolioSummary {
  total_projects: number;
  total_monitored_capex_crores: number;
  total_expenditure_crores: number;
  total_capital_at_risk_crores: number;
  capital_at_risk_percentage: number;
  risk_tier_counts: Partial<Record<RiskTier, number>>;
  dominant_source_counts: Partial<Record<DominantSource, number>>;
  sectors: SectorRollup[];
}

export interface SectorRollup {
  sector: string;
  project_count: number;
  total_original_cost_crores: number;
  total_expenditure_crores: number;
  total_capital_at_risk_crores: number;
  avg_gov_score: number;
  critical_high_projects: number;
}

/** A leaderboard row. Matches `?slim=true`, which omits the heavy nested fields. */
export interface ProjectRow {
  project_id: string;
  project_name: string;
  sector: string;
  implementing_agency: string;
  physical_progress: number;
  original_cost_crores: number;
  revised_cost_crores: number | null;
  p_model: number;
  p_model_score: number;
  rule_floor: number;
  gov_score: number;
  risk_tier: RiskTier;
  dominant_source: DominantSource;
  capital_at_risk_crores: number;
  sector_median_overrun_pct: number;
  effective_overrun_pct: number;
  base_rate_probability: number;
}

export interface RuleSignal {
  flag_id: string;
  signal_code: string;
  rule_name: string;
  severity: string;
  weight: number;
  is_active: boolean;
  metric_name: string;
  metric_value: number;
  threshold_value: number;
  statutory_rationale: string;
}

export interface ShapDriver {
  rank: number;
  feature_name: string;
  feature_value: number;
  shap_value: number;
  direction: string;
  administrative_interpretation: string;
}

export interface Assessment extends ProjectRow {
  rule_signals: RuleSignal[];
  shap_drivers: ShapDriver[];
  prescriptive_interventions: string[];
}

export interface HorizonEntry {
  probability: number;
  independent_probability: number;
  probability_uncalibrated: number;
  calibrated: boolean;
  monotonicity_enforced: boolean;
}

export interface HorizonResponse {
  project_id: string;
  project_name: string;
  sector: string;
  horizons: Record<string, HorizonEntry>;
  horizon_definitions: Record<string, string>;
  feature_approximations: Record<string, string>;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  model_name: string;
  feature_count: number;
  features: string[];
  quarantined_features: string[];
}

/** Minimal shape of the measured metrics artifact this UI reads. */
export interface ModelMetrics {
  provenance: string;
  model_version: string;
  panel: {
    rows_used: number;
    projects: number;
    source_sha256: string;
    time_axis?: { mode?: string; explanation?: string };
  };
  horizons: Record<string, HorizonMetrics>;
  statistical_test: string;
  calendar_split_status?: string;
}

export interface HorizonMetrics {
  status: string;
  horizon_months: number;
  rows_labelled: number;
  positive_rate: number;
  splits: {
    primary: { name: string; question: string; results: ModelRow[]; delong_tests: DelongRow[] };
    out_of_time?: {
      status: string;
      cutoff?: string;
      n_test?: number;
      test_projects?: number;
      results?: ModelRow[];
    };
  };
  calibration: { method: string; ece_before: number; ece_after_cross_fitted: number };
  actionable_cohort: {
    verdict: string;
    cohorts: CohortRow[];
  };
}

export interface ModelRow {
  key: string;
  model_architecture?: string;
  model_class?: string;
  auc?: number;
  pr_auc?: number;
  brier?: number;
}

export interface DelongRow {
  comparison: string;
  auc_proposed?: number;
  auc_baseline?: number;
  auc_delta?: number;
  z_statistic?: number;
  p_value?: number;
}

export interface CohortRow {
  cohort: string;
  n: number;
  positive_rate: number | null;
  auc_xgboost?: number | null;
  auc_logit?: number | null;
  auc_deadline_continuous?: number | null;
  xgboost_minus_deadline_rule?: number;
  status?: string;
}

export interface LeadTime {
  provenance: string;
  recommended_threshold: {
    threshold: number;
    median_lead_months: number | null;
    alert_precision: number;
    recall: number;
    false_alarm_rate: number;
  };
  per_threshold: Array<{
    threshold: number;
    median_lead_months: number | null;
    alert_precision: number;
    recall: number;
    false_alarm_rate: number;
    n_never_alerted: number;
  }>;
  caveats: Record<string, unknown>;
}

/** Thrown when the backend is reachable but has no measured artifact yet. */
export class NotMeasuredError extends Error {
  constructor(public readonly detail: string) {
    super(detail);
    this.name = "NotMeasuredError";
  }
}

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (res.status === 503) {
    const body = await res.json().catch(() => ({}));
    throw new NotMeasuredError(body?.detail ?? "This figure has not been measured yet.");
  }
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return (await res.json()) as T;
}

export const api = {
  health: () => get<HealthResponse>("/api/v1/health"),
  summary: () => get<PortfolioSummary>("/api/v1/portfolio/summary"),

  /** Slim omits rule_signals/shap/interventions — 84% of the payload. */
  ranking: (limit = 250) =>
    get<ProjectRow[]>(`/api/v1/portfolio/risk-ranking?slim=true&limit=${limit}`),

  metrics: () => get<ModelMetrics>("/api/v1/analytics/model-metrics"),
  leadTime: () => get<LeadTime>("/api/v1/analytics/lead-time"),

  assess: (project: Record<string, unknown>) =>
    get<Assessment>("/api/v1/predict/project", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(project),
    }),

  /**
   * Detail for a monitored project, scored from its own stored record.
   *
   * Do NOT rebuild an input from a slim leaderboard row and call `assess`
   * instead: the slim payload omits the schedule fields that drive the
   * model's strongest features, and re-scoring without them returned
   * p_model ~= 0.01 for every project while the row said 0.15-0.51.
   */
  projectDetail: (projectId: string) =>
    get<ProjectDetail>(`/api/v1/portfolio/project/${encodeURIComponent(projectId)}`),

  horizons: (project: Record<string, unknown>) =>
    get<HorizonResponse>("/api/v1/analytics/horizons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(project),
    }),

  orderingSensitivity: () =>
    get<OrderingSensitivity>("/api/v1/analytics/ordering-sensitivity"),

  labelValidity: () =>
    get<LabelValidity>("/api/v1/analytics/label-validity"),

  cufGap: () =>
    get<CufGap>("/api/v1/analytics/cuf-gap"),

  delongTest: (payload: DelongTestInput) =>
    get<DelongTestResult>("/api/v1/analytics/delong-test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};

/** Full assessment plus horizon curve for one monitored project. */
export interface ProjectDetail {
  assessment: Assessment;
  horizons: Record<string, HorizonEntry>;
  horizon_definitions: Record<string, string>;
  feature_approximations: Record<string, string>;
}

/* ── Accurate endpoint types for Simulator / Audit / CUF-Gap pages ───────── */

export interface OrderingSensitivity {
  provenance: string;
  question?: string;
  method?: string;
  shipped_ordering?: string;
  negative_control?: string;
  results: Record<
    string,
    {
      status: string;
      rows: number;
      projects: number;
      positive_rate: number;
      auc: number;
      pr_auc: number;
      brier: number;
      auc_delta_vs_shipped?: number;
    }
  >;
}

export interface LabelValidity {
  provenance: string;
  question?: string;
  method?: string;
  n_slip_months?: number;
  n_hold_months?: number;
  indicators_all_months?: Array<{
    indicator: string;
    group: string;
    meaning: string;
    mean_before_slip: number;
    mean_before_hold: number;
    cohens_d: number;
    effect_size: string;
    observed_direction: string;
    corroborates?: boolean;
  }>;
  rule_f1_premise_test?: {
    premise: string;
    bands: Array<{
      band: string;
      n: number;
      slip_rate: number;
    }>;
    slip_rate_when_f1_fires: number;
    slip_rate_in_opposite_condition: number;
    premise_inverted_for_schedule_slip: boolean;
    interpretation: string;
  };
  stage_coverage_bias?: {
    finding: string;
    implication: string;
  };
  verdict: string;
  caveat?: string;
}

export interface CUFObservableCeiling {
  discriminative_auc_ceiling: number;
  brier_score_at_ceiling: number;
  cost_regression_r2_ceiling: number;
  residual_classification_error: number;
}

export interface CUFProxyAugmentation {
  proxy_names: string[];
  auc_before: number;
  auc_after: number;
  auc_delta: number;
  rmse_before_pct: number;
  rmse_after_pct: number;
  rmse_delta_pct: number;
  delayed_projects_citing_top_three_factors_pct: number;
}

export interface CUFMissingVariable {
  variable_name: string;
  suggested_field_name: string;
  operational_rationale: string;
  recommended_input_type: string;
  primary_source_grounding: string;
  cuf_2_0_priority: string;
}

export interface CufGap {
  methodology_statement: string;
  observable_ceiling: CUFObservableCeiling;
  observable_ceiling_provenance: string;
  proxy_augmentation: CUFProxyAugmentation;
  proxy_augmentation_provenance: string;
  missing_variables_recommended: CUFMissingVariable[];
  policy_action_items: string[];
}

export interface DelongTestInput {
  y_true: number[];
  scores_1: number[];
  scores_2: number[];
}

export interface DelongTestResult {
  auc_1: number;
  auc_2: number;
  z_statistic: number;
  p_value: number;
  variance_auc_1: number;
  covariance_12: number;
}

