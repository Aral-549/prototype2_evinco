"use client";

import { useMemo, useState } from "react";
import { ChevronRight, Search } from "lucide-react";
import type { ProjectRow, RiskTier } from "@/lib/api";
import { crore, TIER_DOT, TIER_ORDER, TIER_LABEL, sourceLabel } from "@/lib/format";
import { Separator } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

const PAGE = 12;

/**
 * The ranked "what should I look at today" list.
 *
 * The old leaderboard was a nine-column table rendering 300 rows at once. A
 * table that wide forces the eye to scan horizontally before it can rank
 * anything, which is the opposite of what this view is for. Each row here
 * carries four things — identity, risk, exposure, and a way in — and the rest
 * moves into the detail sheet.
 */
export function AttentionList({
  rows,
  onSelect,
}: {
  rows: ProjectRow[];
  onSelect: (row: ProjectRow) => void;
}) {
  const [query, setQuery] = useState("");
  const [tier, setTier] = useState<RiskTier | "ALL">("ALL");
  const [limit, setLimit] = useState(PAGE);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rows.filter((r) => {
      const matchesTier = tier === "ALL" || r.risk_tier === tier;
      if (!matchesTier) return false;
      if (!q) return true;
      return (
        r.project_name.toLowerCase().includes(q) ||
        r.sector.toLowerCase().includes(q) ||
        r.implementing_agency.toLowerCase().includes(q)
      );
    });
  }, [rows, query, tier]);

  const visible = filtered.slice(0, limit);

  return (
    <div>
      {/* Controls stay light: one search field and one segmented filter. */}
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <label className="relative flex-1 sm:max-w-xs">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-fg-tertiary" />
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setLimit(PAGE);
            }}
            placeholder="Search projects, sectors, agencies"
            className={cn(
              "t-subhead w-full rounded-[var(--radius-chip)] bg-surface-sunken py-2 pl-9 pr-3",
              "ring-1 ring-hairline outline-none transition",
              "placeholder:text-fg-tertiary focus:ring-2 focus:ring-accent",
            )}
          />
        </label>

        <div
          role="tablist"
          aria-label="Filter by risk tier"
          className="flex gap-1 rounded-[var(--radius-chip)] bg-surface-sunken p-1 ring-1 ring-hairline"
        >
          {(["ALL", ...TIER_ORDER] as const).map((t) => {
            const active = tier === t;
            return (
              <button
                key={t}
                role="tab"
                aria-selected={active}
                onClick={() => {
                  setTier(t);
                  setLimit(PAGE);
                }}
                className={cn(
                  "t-caption rounded-[var(--radius-chip)] px-2.5 py-1.5 font-medium transition",
                  active
                    ? "bg-surface text-fg shadow-sm"
                    : "text-fg-secondary hover:text-fg",
                )}
              >
                {t === "ALL" ? "All" : TIER_LABEL[t]}
              </button>
            );
          })}
        </div>
      </div>

      <div className="overflow-hidden rounded-[var(--radius-card)] bg-surface ring-1 ring-hairline"
        style={{ boxShadow: "var(--shadow-raised)" }}>
        {visible.length === 0 ? (
          <p className="t-subhead px-5 py-10 text-center text-fg-secondary">
            No projects match that filter.
          </p>
        ) : (
          visible.map((row, i) => (
            <div key={row.project_id}>
              {i > 0 ? <Separator className="ml-5" /> : null}
              <button
                onClick={() => onSelect(row)}
                className="group flex w-full items-center gap-4 px-5 py-3.5 text-left transition hover:bg-surface-sunken"
              >
                <span
                  className={cn("size-2 shrink-0 rounded-full", TIER_DOT[row.risk_tier])}
                  aria-label={TIER_LABEL[row.risk_tier]}
                />

                <span className="min-w-0 flex-1">
                  <span className="t-subhead block truncate font-medium">{row.project_name}</span>
                  <span className="t-caption block truncate text-fg-tertiary">
                    {row.sector} · {sourceLabel(row.dominant_source)}
                  </span>
                </span>

                <span className="hidden shrink-0 text-right sm:block">
                  <span className="tabular t-subhead block font-semibold">
                    {row.gov_score.toFixed(0)}
                  </span>
                  <span className="t-caption block text-fg-tertiary">score</span>
                </span>

                <span className="shrink-0 text-right">
                  <span className="tabular t-subhead block font-semibold text-critical">
                    {crore(row.capital_at_risk_crores)}
                  </span>
                  <span className="t-caption block text-fg-tertiary">at risk</span>
                </span>

                <ChevronRight className="size-4 shrink-0 text-fg-tertiary transition group-hover:translate-x-0.5 group-hover:text-fg-secondary" />
              </button>
            </div>
          ))
        )}
      </div>

      {limit < filtered.length ? (
        <div className="mt-4 text-center">
          <button
            onClick={() => setLimit((n) => n + PAGE * 2)}
            className="t-subhead rounded-[var(--radius-chip)] bg-surface px-4 py-2 font-medium text-accent ring-1 ring-hairline transition hover:bg-surface-sunken"
          >
            Show more
          </button>
          <p className="t-caption mt-2 text-fg-tertiary">
            Showing {visible.length} of {filtered.length}
          </p>
        </div>
      ) : null}
    </div>
  );
}
