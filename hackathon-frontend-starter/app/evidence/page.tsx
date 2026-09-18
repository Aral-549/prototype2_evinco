"use client";

import { useEffect, useState } from "react";
import { api, NotMeasuredError, type LeadTime, type ModelMetrics, type OrderingSensitivity, type LabelValidity } from "@/lib/api";
import { count, humanise, percent } from "@/lib/format";
import {
  NotMeasured,
  SectionHeader,
  Separator,
  Surface,
} from "@/components/ui/primitives";
import { SiteHeader } from "@/components/site-header";
import { cn } from "@/lib/utils";

const MODEL_ORDER = ["xgboost", "logit", "dtph", "deadline_continuous", "rulefloor", "prior"];

/**
 * Evidence lives on its own route.
 *
 * Benchmarks, cohort analysis and lead time are what a reviewer audits, not
 * what an operator acts on. Keeping them off the overview is most of the
 * density fix: the main screen answers "what do I look at today", this one
 * answers "why should I believe it".
 */
export default function EvidencePage() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [lead, setLead] = useState<LeadTime | null>(null);
  const [ordering, setOrdering] = useState<OrderingSensitivity | null>(null);
  const [validity, setValidity] = useState<LabelValidity | null>(null);
  const [missing, setMissing] = useState<string | null>(null);

  useEffect(() => {
    api
      .metrics()
      .then(setMetrics)
      .catch((e) =>
        setMissing(
          e instanceof NotMeasuredError
            ? e.detail
            : "Could not reach the PAIMANA API. Start the backend and reload.",
        ),
      );
    api.leadTime().then(setLead).catch(() => undefined);
    api.orderingSensitivity().then(setOrdering).catch(() => undefined);
    api.labelValidity().then(setValidity).catch(() => undefined);
  }, []);

  const h1 = metrics?.horizons?.["1m"];
  const primary = h1?.splits?.primary;
  const oot = h1?.splits?.out_of_time;
  const cohorts = h1?.actionable_cohort;

  return (
    <div className="min-h-dvh">
      <SiteHeader />

      <main className="mx-auto w-full max-w-5xl px-4 pb-24 pt-8 sm:px-6 sm:pt-12">
        <header>
          <h1 className="t-large-title">Evidence</h1>
          <p className="t-body mt-2 max-w-prose text-fg-secondary">
            Every figure here is read from the training run&rsquo;s metrics artifact. Nothing on
            this page is a stored constant; when something has not been measured, it says so.
          </p>
          {metrics ? (
            <p className="t-footnote mt-4 text-fg-tertiary">
              {metrics.provenance} · model {metrics.model_version} ·{" "}
              {count(metrics.panel.rows_used)} transitions from {count(metrics.panel.projects)}{" "}
              projects
              {metrics.panel.time_axis?.mode
                ? ` · ${metrics.panel.time_axis.mode} time axis`
                : ""}
            </p>
          ) : null}
        </header>

        {missing ? (
          <div className="mt-8">
            <NotMeasured title="No measured metrics available" detail={missing} />
          </div>
        ) : null}

        {/* Headline comparison */}
        {primary ? (
          <section className="mt-12">
            <SectionHeader trailing="1-month horizon">
              Against classical statistics
            </SectionHeader>
            <Surface padded={false}>
              {MODEL_ORDER.map((key, i) => {
                const row = primary.results.find((r) => r.key === key);
                if (!row?.auc) return null;
                const proposed = key === "xgboost";
                return (
                  <div key={key}>
                    {i > 0 ? <Separator className="ml-5" /> : null}
                    <div
                      className={cn(
                        "flex items-center gap-4 px-5 py-3.5",
                        proposed && "bg-accent-wash",
                      )}
                    >
                      <div className="min-w-0 flex-1">
                        <div
                          className={cn(
                            "t-subhead truncate",
                            proposed ? "font-semibold text-accent" : "",
                          )}
                        >
                          {row.model_architecture}
                        </div>
                        <div className="t-caption truncate text-fg-tertiary">
                          {row.model_class}
                        </div>
                      </div>
                      <AucBar value={row.auc} highlight={proposed} />
                      <div
                        className={cn(
                          "tabular t-subhead w-14 shrink-0 text-right font-semibold",
                          proposed ? "text-accent" : "text-fg-secondary",
                        )}
                      >
                        {row.auc.toFixed(3)}
                      </div>
                    </div>
                  </div>
                );
              })}
            </Surface>
            <p className="t-footnote mt-3 text-fg-tertiary">{metrics?.statistical_test}</p>
          </section>
        ) : null}

        {/* Out-of-time */}
        {oot ? (
          <section className="mt-12">
            <SectionHeader
              trailing={oot.status === "ok" && oot.cutoff ? `cutoff ${oot.cutoff}` : undefined}
            >
              Forecasting months it never saw
            </SectionHeader>
            {oot.status === "ok" && oot.results ? (
              <Surface>
                <div className="flex flex-wrap items-end gap-x-10 gap-y-5">
                  {oot.results
                    .filter((r) => ["xgboost", "logit", "dtph"].includes(r.key) && r.auc)
                    .map((r) => (
                      <div key={r.key}>
                        <div className="t-section">{r.model_architecture?.split(" (")[0]}</div>
                        <div
                          className={cn(
                            "tabular t-title-2 mt-1",
                            r.key === "xgboost" ? "text-accent" : "text-fg-secondary",
                          )}
                        >
                          {r.auc!.toFixed(4)}
                        </div>
                      </div>
                    ))}
                </div>
                <p className="t-footnote mt-5 text-fg-secondary">
                  Trained only on earlier months and tested on {count(oot.n_test ?? 0)} rows across{" "}
                  {count(oot.test_projects ?? 0)} projects from months held out entirely.
                </p>
              </Surface>
            ) : (
              <NotMeasured
                title="Not available at this horizon"
                detail={oot.status}
              />
            )}
          </section>
        ) : null}

        {/* Actionable cohort — the "isn't this just a deadline rule" answer */}
        {cohorts ? (
          <section className="mt-12">
            <SectionHeader>Is it more than a deadline rule?</SectionHeader>
            <Surface padded={false}>
              {cohorts.cohorts.map((c, i) => (
                <div key={c.cohort}>
                  {i > 0 ? <Separator className="ml-5" /> : null}
                  <div className="flex items-center gap-4 px-5 py-3.5">
                    <div className="min-w-0 flex-1">
                      <div className="t-subhead truncate">{humanise(c.cohort)}</div>
                      <div className="t-caption text-fg-tertiary">
                        {count(c.n)} project-months
                        {c.positive_rate !== null ? ` · base ${percent(c.positive_rate * 100, 1)}` : ""}
                      </div>
                    </div>
                    {c.status ? (
                      <span className="t-caption text-fg-tertiary">{c.status}</span>
                    ) : (
                      <>
                        <div className="hidden text-right sm:block">
                          <div className="tabular t-subhead font-semibold">
                            {c.auc_xgboost?.toFixed(3) ?? "—"}
                          </div>
                          <div className="t-caption text-fg-tertiary">model</div>
                        </div>
                        <div className="text-right">
                          <div className="tabular t-subhead text-fg-secondary">
                            {c.auc_deadline_continuous?.toFixed(3) ?? "—"}
                          </div>
                          <div className="t-caption text-fg-tertiary">deadline rule</div>
                        </div>
                        <div className="w-16 shrink-0 text-right">
                          <div
                            className={cn(
                              "tabular t-subhead font-semibold",
                              (c.xgboost_minus_deadline_rule ?? 0) > 0
                                ? "text-low"
                                : "text-critical",
                            )}
                          >
                            {c.xgboost_minus_deadline_rule !== undefined
                              ? `${c.xgboost_minus_deadline_rule > 0 ? "+" : ""}${c.xgboost_minus_deadline_rule.toFixed(3)}`
                              : "—"}
                          </div>
                          <div className="t-caption text-fg-tertiary">margin</div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </Surface>
            <p className="t-footnote mt-3 max-w-prose text-fg-secondary">{cohorts.verdict}</p>
          </section>
        ) : null}

        {/* Lead time */}
        {lead ? (
          <section className="mt-12">
            <SectionHeader trailing={`threshold ${lead.recommended_threshold.threshold}`}>
              Measured early warning
            </SectionHeader>
            <Surface>
              <div className="flex flex-wrap items-end gap-x-10 gap-y-5">
                <div>
                  <div className="t-section">Median lead</div>
                  <div className="tabular t-large-title mt-1 text-low">
                    {lead.recommended_threshold.median_lead_months ?? "—"}
                    <span className="t-title-3 ml-1 font-normal text-fg-secondary">months</span>
                  </div>
                </div>
                <div>
                  <div className="t-section">Precision</div>
                  <div className="tabular t-title-2 mt-1">
                    {percent(lead.recommended_threshold.alert_precision * 100)}
                  </div>
                </div>
                <div>
                  <div className="t-section">Recall</div>
                  <div className="tabular t-title-2 mt-1">
                    {percent(lead.recommended_threshold.recall * 100)}
                  </div>
                </div>
                <div>
                  <div className="t-section">False alarms</div>
                  <div className="tabular t-title-2 mt-1 text-fg-secondary">
                    {percent(lead.recommended_threshold.false_alarm_rate * 100)}
                  </div>
                </div>
              </div>
              <p className="t-footnote mt-5 max-w-prose text-fg-secondary">
                How far ahead of the ministry&rsquo;s own filing the model raises its first alert.
                A lower bound: the public feed exposes a limited monthly window, so longer leads
                cannot be observed.
              </p>
            </Surface>
          </section>
        ) : null}

        {/* DeLong Paired Tests */}
        {primary?.delong_tests?.length ? (
          <section className="mt-12">
            <SectionHeader>DeLong paired tests</SectionHeader>
            <Surface padded={false}>
              {primary.delong_tests.map((test, i) => {
                const significant = (test.p_value ?? 1) < 0.05;
                return (
                  <div key={test.comparison}>
                    {i > 0 ? <Separator className="ml-5" /> : null}
                    <div className={cn("flex items-center gap-4 px-5 py-3.5", significant && "bg-accent-wash")}>
                      <div className="min-w-0 flex-1">
                        <div className={cn("t-subhead truncate", significant ? "font-semibold text-accent" : "")}>
                          {test.comparison}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="tabular t-subhead text-fg-secondary">
                          {test.auc_delta !== undefined ? (test.auc_delta > 0 ? "+" : "") + test.auc_delta.toFixed(4) : "—"}
                        </div>
                        <div className="t-caption text-fg-tertiary">Δ AUC</div>
                      </div>
                      <div className="text-right">
                        <div className="tabular t-subhead text-fg-secondary">
                          {test.z_statistic !== undefined ? test.z_statistic.toFixed(3) : "—"}
                        </div>
                        <div className="t-caption text-fg-tertiary">Z-stat</div>
                      </div>
                      <div className="w-16 shrink-0 text-right">
                        <div className={cn("tabular t-subhead", significant ? "font-semibold text-accent" : "text-fg-secondary")}>
                          {test.p_value !== undefined ? test.p_value.toFixed(4) : "—"}
                        </div>
                        <div className="t-caption text-fg-tertiary">p-value</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </Surface>
          </section>
        ) : null}

        {/* Calibration */}
        {h1?.calibration ? (
          <section className="mt-12">
            <SectionHeader trailing={h1.calibration.method}>Calibration</SectionHeader>
            <Surface>
              <div className="flex flex-wrap items-end gap-x-10 gap-y-5">
                <div>
                  <div className="t-section">ECE before</div>
                  <div className="tabular t-title-2 mt-1">
                    {h1.calibration.ece_before.toFixed(4)}
                  </div>
                </div>
                <div>
                  <div className="t-section">ECE after cross-fitting</div>
                  <div className="tabular t-title-2 mt-1 text-accent">
                    {h1.calibration.ece_after_cross_fitted.toFixed(4)}
                  </div>
                </div>
              </div>
            </Surface>
          </section>
        ) : null}

        {/* Ordering Sensitivity */}
        {ordering ? (
          <section className="mt-12">
            <SectionHeader>Ordering sensitivity</SectionHeader>
            <Surface padded={false}>
              {ordering.rules.map((rule, i) => (
                <div key={rule.ordering_rule}>
                  {i > 0 ? <Separator className="ml-5" /> : null}
                  <div className={cn("flex items-center gap-4 px-5 py-3.5", rule.is_negative_control && "bg-critical-wash")}>
                    <div className="min-w-0 flex-1">
                      <div className={cn("t-subhead truncate", rule.is_negative_control ? "text-critical font-semibold" : "")}>
                        {rule.ordering_rule}
                      </div>
                      <div className="t-caption text-fg-tertiary truncate">
                        {rule.description}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="tabular t-subhead font-semibold text-fg-secondary">
                        {rule.auc_1m.toFixed(3)}
                      </div>
                      <div className="t-caption text-fg-tertiary">AUC (1m)</div>
                    </div>
                    <div className="w-24 shrink-0 text-right">
                      <div className={cn("t-subhead", rule.is_negative_control ? "text-critical" : "text-fg-secondary")}>
                        {rule.is_negative_control ? "Yes" : "No"}
                      </div>
                      <div className="t-caption text-fg-tertiary">Neg. control</div>
                    </div>
                  </div>
                </div>
              ))}
            </Surface>
            <p className="t-footnote mt-3 max-w-prose text-fg-secondary">{ordering.stability_verdict}</p>
          </section>
        ) : null}

        {/* Label Validity */}
        {validity ? (
          <section className="mt-12">
            <SectionHeader>Label validity</SectionHeader>
            <Surface padded={false}>
              {validity.bands.map((band, i) => (
                <div key={band.band}>
                  {i > 0 ? <Separator className="ml-5" /> : null}
                  <div className="flex items-center gap-4 px-5 py-3.5">
                    <div className="min-w-0 flex-1">
                      <div className="t-subhead truncate">{band.band}</div>
                    </div>
                    <div className="text-right">
                      <div className="tabular t-subhead text-fg-secondary">
                        {count(band.n)}
                      </div>
                      <div className="t-caption text-fg-tertiary">n</div>
                    </div>
                    <div className="w-16 shrink-0 text-right">
                      <div className="tabular t-subhead font-semibold text-accent">
                        {percent(band.slip_rate * 100, 1)}
                      </div>
                      <div className="t-caption text-fg-tertiary">slip rate</div>
                    </div>
                  </div>
                </div>
              ))}
            </Surface>
            <p className="t-footnote mt-3 max-w-prose text-fg-secondary">{validity.verdict}</p>
          </section>
        ) : null}
      </main>
    </div>
  );
}

/** AUC rail, scaled from 0.5 (chance) so differences are legible. */
function AucBar({ value, highlight }: { value: number; highlight?: boolean }) {
  const pct = Math.max(0, Math.min(100, ((value - 0.35) / 0.65) * 100));
  return (
    <div className="hidden h-1.5 w-28 overflow-hidden rounded-full bg-surface-sunken sm:block lg:w-40">
      <div
        className={cn("h-full rounded-full", highlight ? "bg-accent" : "bg-fg-tertiary")}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
