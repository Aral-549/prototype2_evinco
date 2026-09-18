"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X, ChevronDown, TriangleAlert, ShieldCheck } from "lucide-react";
import { api, type Assessment, type HorizonEntry, type ProjectRow } from "@/lib/api";
import { crore, probability, sourceLabel, featureLabel, TIER_TEXT } from "@/lib/format";
import { Chip, Meter, RiskChip, Separator, SectionHeader } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

const HORIZON_LABEL: Record<string, string> = {
  "1m": "Next report",
  "3m": "Within 3 months",
  "6m": "Within 6 months",
};

/** Risk colour for a probability, using the same bands as the GovScore tiers. */
function probTone(p: number): { text: string; bar: string } {
  if (p >= 0.75) return { text: "text-critical", bar: "bg-critical" };
  if (p >= 0.5) return { text: "text-high", bar: "bg-high" };
  if (p >= 0.25) return { text: "text-moderate", bar: "bg-moderate" };
  return { text: "text-low", bar: "bg-low" };
}

/**
 * Project detail, presented as a sheet.
 *
 * This is where the density went. The old dashboard showed the forecast, all
 * six statutory flags, five SHAP drivers and the directives permanently on the
 * main screen. Here they are one tap away, and within the sheet the same
 * discipline applies: only *triggered* flags and the top three drivers are
 * shown until asked for more.
 */
export function ProjectSheet({
  row,
  onClose,
}: {
  row: ProjectRow | null;
  onClose: () => void;
}) {
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [horizons, setHorizons] = useState<Record<string, HorizonEntry> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAllFlags, setShowAllFlags] = useState(false);
  const [showAllDrivers, setShowAllDrivers] = useState(false);
  const [prevRowId, setPrevRowId] = useState<string | null>(null);

  if (row && row.project_id !== prevRowId) {
    setPrevRowId(row.project_id);
    setAssessment(null);
    setHorizons(null);
    setError(null);
    setShowAllFlags(false);
    setShowAllDrivers(false);
  }

  useEffect(() => {
    if (!row) return;

    // One call, scored from the project's own stored record. Rebuilding an
    // input from the slim row and re-scoring it produced a detail view that
    // contradicted the list it was opened from (see lib/api.ts).
    let cancelled = false;
    api
      .projectDetail(row.project_id)
      .then((detail) => {
        if (cancelled) return;
        setAssessment(detail.assessment);
        setHorizons(detail.horizons ?? null);
      })
      .catch(() => {
        if (!cancelled) setError("Could not load the full assessment for this project.");
      });
    return () => {
      cancelled = true;
    };
  }, [row]);

  // Escape closes, and the page behind must not scroll while the sheet is up.
  useEffect(() => {
    if (!row) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [row, onClose]);

  const triggered = assessment?.rule_signals.filter((s) => s.is_active) ?? [];
  const flagsToShow = showAllFlags ? (assessment?.rule_signals ?? []) : triggered;
  const drivers = assessment?.shap_drivers ?? [];
  const driversToShow = showAllDrivers ? drivers : drivers.slice(0, 3);

  return (
    <AnimatePresence>
      {row ? (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-black/25 backdrop-blur-[2px]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            aria-hidden
          />
          <motion.aside
            role="dialog"
            aria-modal="true"
            aria-label={`${row.project_name} detail`}
            className={cn(
              "fixed z-50 bg-surface",
              "inset-x-0 bottom-0 max-h-[92vh] rounded-t-[20px]",
              "sm:inset-y-0 sm:left-auto sm:right-0 sm:max-h-none sm:w-[min(30rem,100vw)] sm:rounded-none sm:rounded-l-[20px]",
              "flex flex-col overflow-hidden",
            )}
            style={{ boxShadow: "var(--shadow-sheet)" }}
            initial={{ y: "100%", opacity: 0.6 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: "100%", opacity: 0.6 }}
            transition={{ type: "spring", stiffness: 420, damping: 40 }}
          >
            {/* Grabber — the affordance people expect on a sheet. */}
            <div className="flex justify-center pt-2.5 sm:hidden">
              <div className="h-1 w-9 rounded-full bg-hairline" />
            </div>

            <header className="flex items-start gap-3 px-5 pb-4 pt-4 sm:px-6">
              <div className="min-w-0 flex-1">
                <h2 className="t-title-2 text-balance">{row.project_name}</h2>
                <p className="t-footnote mt-1 text-fg-secondary">
                  {row.sector} · {row.implementing_agency}
                </p>
              </div>
              <button
                onClick={onClose}
                aria-label="Close"
                className="grid size-8 shrink-0 place-items-center rounded-full bg-surface-sunken text-fg-secondary transition hover:text-fg"
              >
                <X className="size-4" />
              </button>
            </header>
            <Separator />

            <div className="flex-1 overflow-y-auto px-5 pb-10 pt-5 sm:px-6">
              {/* Headline: the score, and — just as important — what set it. */}
              <div className="flex items-end justify-between gap-4">
                <div>
                  <div className="t-section">Governance risk score</div>
                  <div className={cn("tabular t-large-title mt-1", TIER_TEXT[row.risk_tier])}>
                    {row.gov_score.toFixed(1)}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <RiskChip tier={row.risk_tier} />
                  <Chip>{sourceLabel(row.dominant_source)}</Chip>
                </div>
              </div>

              <p className="t-footnote mt-3 text-fg-secondary">
                {row.dominant_source === "RULE_FLOOR_OVERRIDE" ? (
                  <>
                    A statutory breach set this score. The model put the risk at{" "}
                    <span className="tabular font-semibold text-fg">
                      {probability(row.p_model)}
                    </span>
                    , but a floor of{" "}
                    <span className="tabular font-semibold text-fg">
                      {row.rule_floor.toFixed(0)}
                    </span>{" "}
                    applies and cannot be overridden by an optimistic forecast.
                  </>
                ) : (
                  <>
                    The model set this score at{" "}
                    <span className="tabular font-semibold text-fg">
                      {probability(row.p_model)}
                    </span>
                    , above the statutory floor of{" "}
                    <span className="tabular font-semibold text-fg">
                      {row.rule_floor.toFixed(0)}
                    </span>
                    .
                  </>
                )}
              </p>

              <div className="mt-5 grid grid-cols-2 gap-4">
                <div>
                  <div className="t-section">Capital at risk</div>
                  <div className="tabular t-title-3 mt-1 text-critical">
                    {crore(row.capital_at_risk_crores)}
                  </div>
                </div>
                <div>
                  <div className="t-section">Sanctioned</div>
                  <div className="tabular t-title-3 mt-1">{crore(row.original_cost_crores)}</div>
                </div>
              </div>

              <div className="mt-5">
                <div className="mb-2 flex items-baseline justify-between">
                  <span className="t-section">Physical progress</span>
                  <span className="tabular t-footnote text-fg-secondary">
                    {row.physical_progress.toFixed(0)}%
                  </span>
                </div>
                <Meter value={row.physical_progress} tone="bg-accent" />
              </div>

              {/* Forecast curve */}
              <div className="mt-8">
                <SectionHeader trailing={horizons ? "calibrated" : undefined}>
                  Slip forecast
                </SectionHeader>
                {horizons && Object.keys(horizons).length ? (
                  <div className="space-y-3.5">
                    {["1m", "3m", "6m"].map((key) => {
                      const h = horizons[key];
                      if (!h) return null;
                      const tone = probTone(h.probability);
                      return (
                        <div key={key}>
                          <div className="flex items-baseline justify-between">
                            <span className="t-subhead">{HORIZON_LABEL[key]}</span>
                            <span className={cn("tabular t-headline", tone.text)}>
                              {probability(h.probability)}
                            </span>
                          </div>
                          <Meter className="mt-1.5" value={h.probability * 100} tone={tone.bar} />
                        </div>
                      );
                    })}
                    <p className="t-caption pt-1 text-fg-tertiary">
                      Probability the declared completion date is pushed further out within each
                      window.
                    </p>
                  </div>
                ) : (
                  <SkeletonRows rows={3} />
                )}
              </div>

              {/* Statutory flags — triggered only, by default */}
              <div className="mt-8">
                <SectionHeader
                  trailing={
                    assessment ? `${triggered.length} of ${assessment.rule_signals.length}` : undefined
                  }
                >
                  Statutory flags
                </SectionHeader>

                {!assessment ? (
                  <SkeletonRows rows={2} />
                ) : triggered.length === 0 && !showAllFlags ? (
                  <div className="flex items-center gap-2.5 rounded-[var(--radius-card)] bg-low-wash px-4 py-3">
                    <ShieldCheck className="size-4 shrink-0 text-low" />
                    <span className="t-subhead text-low">No statutory flag is triggered.</span>
                  </div>
                ) : (
                  <ul className="space-y-2.5">
                    {flagsToShow.map((s) => (
                      <li
                        key={s.flag_id}
                        className={cn(
                          "rounded-[var(--radius-card)] px-4 py-3",
                          s.is_active ? "bg-critical-wash" : "bg-surface-sunken",
                        )}
                      >
                        <div className="flex items-start gap-2.5">
                          {s.is_active ? (
                            <TriangleAlert className="mt-0.5 size-4 shrink-0 text-critical" />
                          ) : (
                            <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-fg-tertiary" />
                          )}
                          <div className="min-w-0">
                            <div
                              className={cn(
                                "t-subhead font-semibold",
                                s.is_active ? "text-critical" : "text-fg-secondary",
                              )}
                            >
                              {s.flag_id} · {s.rule_name}
                            </div>
                            {s.is_active ? (
                              <p className="t-caption mt-1 text-fg-secondary">
                                {s.statutory_rationale}
                              </p>
                            ) : null}
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}

                {assessment && assessment.rule_signals.length > triggered.length ? (
                  <Disclosure
                    open={showAllFlags}
                    onToggle={() => setShowAllFlags((v) => !v)}
                    label={showAllFlags ? "Show triggered only" : "Show all six checks"}
                  />
                ) : null}
              </div>

              {/* Drivers */}
              <div className="mt-8">
                <SectionHeader>Why</SectionHeader>
                {!assessment ? (
                  <SkeletonRows rows={3} />
                ) : (
                  <ul className="space-y-3">
                    {driversToShow.map((d) => {
                      const up = d.shap_value > 0;
                      const magnitude = Math.min(100, Math.abs(d.shap_value) * 60);
                      return (
                        <li key={d.rank}>
                          <div className="flex items-baseline justify-between gap-3">
                            <span className="t-subhead truncate">
                              {featureLabel(d.feature_name)}
                            </span>
                            <span
                              className={cn(
                                "tabular t-footnote shrink-0 font-semibold",
                                up ? "text-critical" : "text-low",
                              )}
                            >
                              {up ? "+" : ""}
                              {d.shap_value.toFixed(2)}
                            </span>
                          </div>
                          <Meter
                            className="mt-1.5"
                            value={magnitude}
                            tone={up ? "bg-critical" : "bg-low"}
                          />
                        </li>
                      );
                    })}
                  </ul>
                )}
                {drivers.length > 3 ? (
                  <Disclosure
                    open={showAllDrivers}
                    onToggle={() => setShowAllDrivers((v) => !v)}
                    label={showAllDrivers ? "Show top three" : `Show all ${drivers.length}`}
                  />
                ) : null}
              </div>

              {/* Directives */}
              {assessment?.prescriptive_interventions?.length ? (
                <div className="mt-8">
                  <SectionHeader>Recommended action</SectionHeader>
                  <ul className="space-y-2.5">
                    {assessment.prescriptive_interventions.map((text, i) => (
                      <li
                        key={i}
                        className="rounded-[var(--radius-card)] bg-surface-sunken px-4 py-3"
                      >
                        <p className="t-subhead text-fg-secondary">{text}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {error ? <p className="t-footnote mt-6 text-critical">{error}</p> : null}
            </div>
          </motion.aside>
        </>
      ) : null}
    </AnimatePresence>
  );
}

function Disclosure({
  open,
  onToggle,
  label,
}: {
  open: boolean;
  onToggle: () => void;
  label: string;
}) {
  return (
    <button
      onClick={onToggle}
      className="t-footnote mt-3 inline-flex items-center gap-1 font-medium text-accent transition hover:opacity-70"
    >
      {label}
      <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
    </button>
  );
}

function SkeletonRows({ rows }: { rows: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-9 animate-pulse rounded-lg bg-surface-sunken" />
      ))}
    </div>
  );
}
