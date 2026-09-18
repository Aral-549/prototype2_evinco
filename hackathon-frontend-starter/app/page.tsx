"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, type PortfolioSummary, type ProjectRow } from "@/lib/api";
import {
  count,
  crore,
  croreCompact,
  percent,
  TIER_DOT,
  TIER_LABEL,
  TIER_ORDER,
} from "@/lib/format";
import { AttentionList } from "@/components/attention-list";
import { ProjectSheet } from "@/components/project-sheet";
import { SectionHeader, Separator, Stat, Surface } from "@/components/ui/primitives";
import { SiteHeader } from "@/components/site-header";
import { cn } from "@/lib/utils";

export default function OverviewPage() {
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [rows, setRows] = useState<ProjectRow[]>([]);
  const [selected, setSelected] = useState<ProjectRow | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.summary(), api.ranking(250)])
      .then(([s, r]) => {
        setSummary(s);
        setRows(r);
        // Deep link: ?project=<id> opens that project's sheet directly, so a
        // reviewer can be sent straight to the project under discussion.
        const wanted = new URLSearchParams(window.location.search).get("project");
        if (wanted) {
          const match = r.find((row) => row.project_id === wanted);
          if (match) setSelected(match);
        }
      })
      .catch(() =>
        setError(
          "Could not reach the PAIMANA API. Start the backend, then reload this page.",
        ),
      );
  }, []);

  const tierCounts = summary?.risk_tier_counts ?? {};
  const needsAttention =
    (tierCounts.CRITICAL ?? 0) + (tierCounts.HIGH ?? 0);
  const overrides = summary?.dominant_source_counts?.RULE_FLOOR_OVERRIDE ?? 0;

  return (
    <div className="min-h-dvh">
      <SiteHeader />

      <main className="mx-auto w-full max-w-5xl px-4 pb-24 pt-8 sm:px-6 sm:pt-12">
        {error ? (
          <Surface className="border-dashed">
            <h1 className="t-title-3">Backend unavailable</h1>
            <p className="t-subhead mt-2 text-fg-secondary">{error}</p>
          </Surface>
        ) : null}

        {/*
          One question per view: how much public capital is exposed, and which
          projects account for it. Everything else is one tap away.
        */}
        <motion.section
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        >
          <h1 className="t-section">Capital at risk</h1>
          <div className="tabular mt-2 text-[44px] font-bold leading-[1.05] tracking-[-0.022em] text-critical sm:text-[64px]">
            {summary ? croreCompact(summary.total_capital_at_risk_crores) : "—"}
          </div>
          <p className="t-body mt-3 max-w-prose text-fg-secondary">
            {summary ? (
              <>
                {percent(summary.capital_at_risk_percentage, 2)} of{" "}
                {croreCompact(summary.total_monitored_capex_crores)} monitored across{" "}
                {count(summary.total_projects)} central infrastructure projects.
              </>
            ) : (
              "Loading portfolio…"
            )}
          </p>
        </motion.section>

        {/* Three supporting figures. Deliberately quieter than the hero. */}
        <section className="mt-10">
          <Surface padded={false}>
            <div className="grid grid-cols-1 divide-y divide-hairline sm:grid-cols-3 sm:divide-x sm:divide-y-0">
              <div className="p-5 sm:p-6">
                <Stat
                  label="Needs attention"
                  value={summary ? count(needsAttention) : "—"}
                  detail="Critical or high risk"
                  tone="text-critical"
                />
              </div>
              <div className="p-5 sm:p-6">
                <Stat
                  label="Statutory overrides"
                  value={summary ? count(overrides) : "—"}
                  detail="Score set by a legal breach, not the model"
                />
              </div>
              <div className="p-5 sm:p-6">
                <Stat
                  label="Monitored capex"
                  value={summary ? croreCompact(summary.total_monitored_capex_crores) : "—"}
                  detail="₹150+ Cr central tier"
                />
              </div>
            </div>
          </Surface>

          {/* Tier distribution as a single proportional rail, not four tiles. */}
          {summary ? <TierBar counts={tierCounts} total={summary.total_projects} /> : null}
        </section>

        <section className="mt-12">
          <SectionHeader trailing={rows.length ? `${count(rows.length)} ranked by exposure` : undefined}>
            Priority review
          </SectionHeader>
          {rows.length ? (
            <AttentionList
              rows={rows}
              onSelect={(row) => {
                setSelected(row);
                window.history.replaceState(null, "", `?project=${row.project_id}`);
              }}
            />
          ) : !error ? (
            <div className="space-y-2">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="h-16 animate-pulse rounded-[var(--radius-card)] bg-surface" />
              ))}
            </div>
          ) : null}
        </section>
      </main>

      <ProjectSheet
        row={selected}
        onClose={() => {
          setSelected(null);
          window.history.replaceState(null, "", window.location.pathname);
        }}
      />
    </div>
  );
}

/**
 * The risk mix as one proportional bar. Four separate count tiles made the eye
 * do the division; a single rail shows the shape of the portfolio at a glance.
 */
function TierBar({
  counts,
  total,
}: {
  counts: Partial<Record<string, number>>;
  total: number;
}) {
  return (
    <div className="mt-6">
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-surface-sunken">
        {TIER_ORDER.map((tier) => {
          const n = counts[tier] ?? 0;
          if (!n) return null;
          return (
            <div
              key={tier}
              className={cn(TIER_DOT[tier], "h-full")}
              style={{ width: `${(n / total) * 100}%` }}
              title={`${TIER_LABEL[tier]}: ${n}`}
            />
          );
        })}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
        {TIER_ORDER.map((tier) => (
          <span key={tier} className="inline-flex items-center gap-2">
            <span className={cn("size-1.5 rounded-full", TIER_DOT[tier])} aria-hidden />
            <span className="t-caption text-fg-secondary">
              {TIER_LABEL[tier]}{" "}
              <span className="tabular font-semibold text-fg">{count(counts[tier] ?? 0)}</span>
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
