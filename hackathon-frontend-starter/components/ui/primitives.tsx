import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { TIER_DOT, TIER_LABEL, TIER_WASH } from "@/lib/format";
import type { RiskTier } from "@/lib/api";

/**
 * A raised surface. Deference: one hairline and a soft shadow, never a heavy
 * border. Stacking borders is what makes a dashboard feel like a spreadsheet.
 */
export function Surface({
  className,
  children,
  padded = true,
}: {
  className?: string;
  children: ReactNode;
  padded?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-[var(--radius-card)] bg-surface ring-1 ring-hairline",
        padded && "p-5 sm:p-6",
        className,
      )}
      style={{ boxShadow: "var(--shadow-raised)" }}
    >
      {children}
    </div>
  );
}

/** Uppercase section header, as used above grouped lists in Settings. */
export function SectionHeader({
  children,
  trailing,
  className,
}: {
  children: ReactNode;
  trailing?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mb-3 flex items-end justify-between gap-4", className)}>
      <h2 className="t-section">{children}</h2>
      {trailing ? <div className="t-footnote text-fg-tertiary">{trailing}</div> : null}
    </div>
  );
}

export function RiskChip({ tier, className }: { tier: RiskTier; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-[var(--radius-chip)] px-2.5 py-1",
        "t-caption font-semibold",
        TIER_WASH[tier],
        className,
      )}
    >
      <span className={cn("size-1.5 rounded-full", TIER_DOT[tier])} aria-hidden />
      {TIER_LABEL[tier]}
    </span>
  );
}

/** Neutral metadata chip. Deliberately colourless so risk colour stays meaningful. */
export function Chip({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-[var(--radius-chip)] bg-surface-sunken px-2.5 py-1",
        "t-caption text-fg-secondary ring-1 ring-hairline",
        className,
      )}
    >
      {children}
    </span>
  );
}

/**
 * A labelled figure. `emphasis` controls the type ramp so one number per view
 * can carry real weight while its neighbours stay quiet — the hierarchy the
 * old dashboard lacked when four tiles all shouted at the same volume.
 */
export function Stat({
  label,
  value,
  detail,
  emphasis = "normal",
  tone,
  className,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  emphasis?: "hero" | "normal";
  tone?: string;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <div className="t-section">{label}</div>
      <div
        className={cn(
          "tabular mt-1.5 truncate",
          emphasis === "hero" ? "t-large-title sm:text-[44px] sm:leading-[48px]" : "t-title-2",
          tone,
        )}
      >
        {value}
      </div>
      {detail ? <div className="t-footnote mt-1 text-fg-secondary">{detail}</div> : null}
    </div>
  );
}

/** Hairline divider matching the Apple separator inset. */
export function Separator({ className }: { className?: string }) {
  return <div className={cn("h-px bg-hairline", className)} aria-hidden />;
}

/**
 * Honest empty state. Used when the backend reports a figure has not been
 * measured — the UI says so instead of rendering a zero.
 */
export function NotMeasured({ title, detail }: { title: string; detail?: string }) {
  return (
    <Surface className="border-dashed">
      <div className="t-headline">{title}</div>
      {detail ? <p className="t-footnote mt-2 max-w-prose text-fg-secondary">{detail}</p> : null}
    </Surface>
  );
}

/** Quiet progress rail. Meter, not decoration — so it carries risk colour. */
export function Meter({
  value,
  tone = "bg-accent",
  className,
}: {
  value: number;
  tone?: string;
  className?: string;
}) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-surface-sunken", className)}>
      <div
        className={cn("h-full rounded-full transition-[width] duration-500", tone)}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
