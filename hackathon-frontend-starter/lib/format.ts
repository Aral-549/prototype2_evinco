import type { RiskTier } from "./api";

/**
 * Indian numbering. ₹1,48,774 Cr groups as lakh-crore, not 148,774 — using
 * en-US grouping here would read as wrong to the audience this is built for.
 */
const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const inr1 = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 1 });

export function crore(value: number): string {
  return `₹${inr.format(Math.round(value))} Cr`;
}

/**
 * Large rupee amounts, rendered the way the ministry states them: anything
 * at or above one lakh crore reads as "₹4.18 lakh Cr" rather than a
 * seven-digit string nobody can parse at a glance.
 */
export function croreCompact(value: number): string {
  if (Math.abs(value) >= 100_000) return `₹${inr1.format(value / 100_000)} lakh Cr`;
  if (Math.abs(value) >= 1_000) return `₹${inr1.format(value / 1_000)}k Cr`;
  return `₹${inr.format(Math.round(value))} Cr`;
}

export function percent(value: number, digits = 0): string {
  return `${value.toFixed(digits)}%`;
}

/** 0–1 probability as a percentage. */
export function probability(p: number, digits = 1): string {
  return `${(p * 100).toFixed(digits)}%`;
}

export function count(value: number): string {
  return inr.format(value);
}

export const TIER_LABEL: Record<RiskTier, string> = {
  CRITICAL: "Critical",
  HIGH: "High",
  MODERATE: "Moderate",
  LOW: "Low",
};

/** Semantic colour only: these classes always mean risk level, never decoration. */
export const TIER_TEXT: Record<RiskTier, string> = {
  CRITICAL: "text-critical",
  HIGH: "text-high",
  MODERATE: "text-moderate",
  LOW: "text-low",
};

export const TIER_WASH: Record<RiskTier, string> = {
  CRITICAL: "bg-critical-wash text-critical",
  HIGH: "bg-high-wash text-high",
  MODERATE: "bg-moderate-wash text-moderate",
  LOW: "bg-low-wash text-low",
};

export const TIER_DOT: Record<RiskTier, string> = {
  CRITICAL: "bg-critical",
  HIGH: "bg-high",
  MODERATE: "bg-moderate",
  LOW: "bg-low",
};

export const TIER_ORDER: RiskTier[] = ["CRITICAL", "HIGH", "MODERATE", "LOW"];

/**
 * Why a project scored the way it did, in one phrase. The distinction matters:
 * a RuleFloor override means a statutory breach set the score and the model's
 * opinion was irrelevant.
 */
export function sourceLabel(source: string): string {
  return source === "RULE_FLOOR_OVERRIDE" ? "Statutory floor" : "Model forecast";
}

/** Turn `runway_over_6_months` into `Runway over 6 months`. */
export function humanise(key: string): string {
  const spaced = key.replace(/_/g, " ").trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

export function featureLabel(name: string): string {
  return humanise(name);
}
