#!/usr/bin/env python3
"""Reconstruct the MoSPI PAIMANA longitudinal project-month panel.

Implements contracts/panel_builder.md.

The PAIMANA GetTileData endpoint returns every monthly record it holds for a
project but strips the reporting month (`Month`/`Year` are null on every
record). Temporal order is therefore *reconstructed* from monotone physical
quantities under the stated assumptions A1/A2, and a per-project confidence
diagnostic is emitted so the assumption can be falsified rather than trusted.

Usage:
    python scripts/build_panel.py
    python scripts/build_panel.py --raw data/paimana_live_raw.json --slip-days 15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paimana.panel")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# Prefer the dated monthly harvest (real reporting months, no A1/A2). Fall
# back to the legacy undated bulk harvest only if it is the only file present.
MONTHLY_RAW_PATH = DATA_DIR / "paimana_monthly_raw.json"
LEGACY_RAW_PATH = DATA_DIR / "paimana_live_raw.json"
RAW_PATH = MONTHLY_RAW_PATH if MONTHLY_RAW_PATH.exists() else LEGACY_RAW_PATH
PANEL_PATH = DATA_DIR / "paimana_panel.csv"
META_PATH = DATA_DIR / "paimana_panel_meta.json"

DEFAULT_SLIP_DAYS = 15
DEFAULT_MIN_ORDER_CONFIDENCE = 0.60
DAYS_PER_MONTH = 30.44

# Assumption A2: the newest snapshot corresponds to the harvest month.
DEFAULT_HARVEST_MONTH = (2026, 9)

# Identity / bookkeeping columns. Never fed to the model.
ID_COLUMNS = [
    "project_id",
    "project_name",
    "sector",
    "seq_index",
    "n_snapshots",
    "as_of_month",
    "order_confidence",
    "order_reliable",
    "revised_date_t",
    "revised_date_next",
    "sanction_imputed",
    "window_truncated",
    "time_axis",
    "gap_months",
]

# Leak-free feature columns, observable strictly at time t.
FEATURE_COLUMNS = [
    "original_cost",
    "log_original_cost",
    "expenditure",
    "expenditure_pct",
    "expenditure_delta",
    "expenditure_delta_pct",
    "physical_progress",
    "progress_delta",
    "progress_delta_3m",
    "remaining_progress_pct",
    "spend_progress_gap",
    "burn_to_progress_ratio",
    "months_elapsed",
    "original_duration_months",
    "elapsed_ratio",
    "schedule_pressure",
    "months_to_revised_date",
    "revised_date_already_slipped_months",
    "stall_streak",
    "progress_velocity",
    "required_velocity",
    "velocity_deficit",
]

LABEL_COLUMN = "slip_next"

# Multi-horizon labels. `slip_next` answers "does the date move at the next
# report?" -- a 1-month question, which caps measurable lead time at ~1 month.
# The horizon labels ask "does it move at any report within the next K months?",
# which is the question a ministry actually plans against, and which permits
# genuinely longer early warning.
#
# Right-censoring: when a project's remaining observation window is shorter
# than K and no slip has occurred inside it, the answer is unknown -- the label
# is written as empty (NaN), never as 0. Coding an unknown as "no slip" would
# manufacture negatives out of missing data and inflate every metric.
HORIZONS: List[int] = [3, 6]
HORIZON_LABELS = [f"slip_within_{k}m" for k in HORIZONS]
CENSOR_FLAGS = [f"censored_{k}m" for k in HORIZONS]

PANEL_COLUMNS = (
    ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN] + HORIZON_LABELS + CENSOR_FLAGS
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def to_float(value: Any, default: float = 0.0) -> float:
    """Coerce PAIMANA's mixed str/num/null fields to float."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_date(value: Any) -> Optional[datetime]:
    """Parse the Indian government date formats used across the PAIMANA feed."""
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def canonical_sector(raw: Optional[str]) -> str:
    """Map the portal's 26 sector labels onto the platform's baseline groups.

    Mirrors scripts/download_real_paimana_data.py so the panel and the served
    portfolio agree on sector naming.
    """
    name = str(raw or "Other").strip()
    lowered = name.lower()
    if "railway" in lowered:
        return "Railways"
    if "road" in lowered or "highway" in lowered:
        return "Roads & Highways"
    if "electricity" in lowered or "energy" in lowered:
        return "Power"
    if "oil" in lowered or "gas" in lowered or "petroleum" in lowered:
        return "Petroleum"
    if "coal" in lowered:
        return "Coal"
    if "steel" in lowered or "metal" in lowered:
        return "Steel"
    if "aviation" in lowered:
        return "Aviation"
    if "urban" in lowered or "transport" in lowered:
        return "Urban Transport"
    if "water" in lowered:
        return "Water Resources"
    if "telecommunication" in lowered:
        return "Telecommunications"
    return name


def modal(values: Iterable[Any]) -> Any:
    """Most common non-null value, deterministic on ties (first-seen wins)."""
    present = [v for v in values if v not in (None, "")]
    if not present:
        return None
    return Counter(present).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Snapshot model
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    expenditure: float
    physical_progress: float
    revised_date: Optional[datetime]
    original_cost: float
    revised_cost: float
    sanction_date: Optional[datetime]
    original_end_date: Optional[datetime]
    # Real reporting month ("YYYY-MM") when the harvest carried one. Present
    # for records collected by scripts/harvest_monthly.py; None for the legacy
    # undated bulk harvest.
    freeze_month: Optional[str] = None
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    def sort_key(self) -> Tuple[float, float, float]:
        """Lexicographic ordering key under assumption A1.

        Used ONLY in reconstructed mode. Missing revised dates sort first so
        they do not jump a project's ordering around; the value is only a
        tie-breaker behind the two monotone quantities.
        """
        rev = self.revised_date.timestamp() if self.revised_date else float("-inf")
        return (self.expenditure, self.physical_progress, rev)

    def month_ordinal(self) -> int:
        """Calendar month as a sortable integer. Dated mode only."""
        y, m = (int(x) for x in self.freeze_month.split("-"))
        return y * 12 + (m - 1)

    def as_of_date(self) -> Optional[datetime]:
        if not self.freeze_month:
            return None
        y, m = (int(x) for x in self.freeze_month.split("-"))
        return datetime(y, m, 15)


def dedupe_key(record: Dict[str, Any]) -> Tuple:
    """Identity of a snapshot, used to collapse duplicate query slices.

    The harvester queries the portal by sector, then by cost band, then by
    state; a single project is returned by several of those slices. Those
    repeats are byte-identical on the measurement fields and must not be
    mistaken for extra months of history.
    """
    return (
        # DATED MODE: the reporting month is part of a snapshot's identity.
        # Without it, two genuinely distinct months in which nothing happened
        # on site would collapse into one, silently deleting real observations
        # and corrupting every month-denominated feature downstream.
        str(record.get("freeze_month") or ""),
        str(record.get("Expenditure")),
        str(record.get("PhysicalProgress")),
        str(record.get("RevisedDate")),
        str(record.get("RevisedCost")),
        str(record.get("SanctionDate")),
        str(record.get("OriginalEndDate")),
        str(record.get("OriginalCost")),
    )


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_snapshots(records: List[Dict[str, Any]], stats: Counter) -> List[Snapshot]:
    """Dedupe, parse and temporally order one project's raw records."""
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for rec in records:
        key = dedupe_key(rec)
        if key in seen:
            stats["duplicate_snapshots_dropped"] += 1
            continue
        seen.add(key)
        unique.append(rec)

    # Modal sanction / original-end dates: the live feed carries clerical
    # corrections across snapshots (observed on 1,761 records), and taking the
    # latest value would let a correction rewrite the project's whole timeline.
    modal_sanction = parse_date(modal(r.get("SanctionDate") for r in unique))
    modal_orig_end = parse_date(modal(r.get("OriginalEndDate") for r in unique))
    modal_orig_cost = to_float(modal(r.get("OriginalCost") for r in unique))

    snapshots: List[Snapshot] = []
    for rec in unique:
        expenditure = to_float(rec.get("Expenditure"))
        # Some agencies report expenditure in rupees rather than crore.
        if modal_orig_cost > 0 and expenditure > 20 * modal_orig_cost:
            expenditure /= 100.0
            stats["expenditure_unit_rescaled"] += 1

        progress = max(0.0, min(100.0, to_float(rec.get("PhysicalProgress"))))
        revised_cost = to_float(rec.get("RevisedCost"), default=modal_orig_cost)
        if revised_cost <= 0:
            revised_cost = modal_orig_cost

        snapshots.append(
            Snapshot(
                expenditure=expenditure,
                physical_progress=progress,
                revised_date=parse_date(rec.get("RevisedDate")),
                original_cost=modal_orig_cost,
                revised_cost=revised_cost,
                sanction_date=modal_sanction,
                original_end_date=modal_orig_end,
                freeze_month=rec.get("freeze_month"),
                raw=rec,
            )
        )

    # DATED MODE when every record carries a real reporting month: order by the
    # calendar, not by a reconstruction. Assumptions A1 and A2 do not apply.
    if snapshots and all(x.freeze_month for x in snapshots):
        # One project can legitimately appear twice inside a month only if the
        # portal returned it in two slices; keep the last by measurement.
        by_month: Dict[str, Snapshot] = {}
        for snap in sorted(snapshots, key=lambda x: (x.expenditure, x.physical_progress)):
            by_month[snap.freeze_month] = snap
        ordered = sorted(by_month.values(), key=Snapshot.month_ordinal)
        stats["dated_projects"] += 1
        return ordered

    stats["reconstructed_projects"] += 1
    snapshots.sort(key=Snapshot.sort_key)
    return snapshots


def order_confidence(snapshots: List[Snapshot]) -> float:
    """Fraction of adjacent pairs where physical progress is non-decreasing.

    This is the diagnostic that can falsify assumption A1 for a given project:
    if expenditure-order and progress-order disagree badly, the reconstructed
    timeline for that project is not trustworthy.
    """
    if len(snapshots) < 2:
        return 1.0
    ok = sum(
        1
        for a, b in zip(snapshots, snapshots[1:])
        if b.physical_progress >= a.physical_progress
    )
    return ok / (len(snapshots) - 1)


def month_index(harvest: Tuple[int, int], months_back: int) -> str:
    """Stamp a snapshot with its approximate reporting month (assumption A2)."""
    year, month = harvest
    total = year * 12 + (month - 1) - months_back
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def compute_features(
    snaps: List[Snapshot],
    idx: int,
    as_of: datetime,
) -> Dict[str, float]:
    """Features observable strictly at snapshot `idx`.

    Nothing here may read snapshot idx+1: that is the outcome.
    """
    cur = snaps[idx]
    prev = snaps[idx - 1] if idx > 0 else None
    prev3 = snaps[max(0, idx - 3)]

    original_cost = max(1.0, cur.original_cost)
    expenditure_pct = 100.0 * cur.expenditure / original_cost
    progress = cur.physical_progress

    expenditure_delta = max(0.0, cur.expenditure - prev.expenditure) if prev else 0.0
    progress_delta = (cur.physical_progress - prev.physical_progress) if prev else 0.0
    progress_delta_3m = cur.physical_progress - prev3.physical_progress

    # Duration from the project's own sanction window, not a global default.
    if cur.sanction_date and cur.original_end_date and cur.original_end_date > cur.sanction_date:
        duration = max(1.0, (cur.original_end_date - cur.sanction_date).days / DAYS_PER_MONTH)
    else:
        duration = 36.0

    if cur.sanction_date:
        months_elapsed = max(0.0, (as_of - cur.sanction_date).days / DAYS_PER_MONTH)
    elif cur.original_end_date:
        months_elapsed = max(0.0, (as_of - cur.original_end_date).days / DAYS_PER_MONTH + duration)
    else:
        months_elapsed = duration * 0.5

    elapsed_ratio = months_elapsed / duration

    # Slippage already on the record at time t. Permitted: it is visible before
    # the next revision is filed, unlike the outcome itself.
    if cur.revised_date and cur.original_end_date and cur.revised_date > cur.original_end_date:
        already_slipped = (cur.revised_date - cur.original_end_date).days / DAYS_PER_MONTH
    else:
        already_slipped = 0.0

    months_to_revised = (
        (cur.revised_date - as_of).days / DAYS_PER_MONTH if cur.revised_date else 0.0
    )

    # Consecutive snapshots ending at idx with effectively no physical movement.
    stall_streak = 0
    for j in range(idx, 0, -1):
        if snaps[j].physical_progress - snaps[j - 1].physical_progress <= 0.1:
            stall_streak += 1
        else:
            break

    # Velocity: achieved %/month vs the %/month still required to hit the
    # currently-declared date. The deficit is the core early-warning signal.
    progress_velocity = progress_delta_3m / max(1.0, min(3.0, idx)) if idx > 0 else 0.0
    remaining = max(0.0, 100.0 - progress)
    required_velocity = remaining / max(1.0, months_to_revised) if months_to_revised > 0 else remaining
    velocity_deficit = required_velocity - progress_velocity

    return {
        "original_cost": round(cur.original_cost, 2),
        "log_original_cost": round(_log1p(cur.original_cost), 4),
        "expenditure": round(cur.expenditure, 2),
        "expenditure_pct": round(expenditure_pct, 3),
        "expenditure_delta": round(expenditure_delta, 2),
        "expenditure_delta_pct": round(100.0 * expenditure_delta / original_cost, 4),
        "physical_progress": round(progress, 2),
        "progress_delta": round(progress_delta, 3),
        "progress_delta_3m": round(progress_delta_3m, 3),
        "remaining_progress_pct": round(remaining, 2),
        "spend_progress_gap": round(expenditure_pct - progress, 3),
        "burn_to_progress_ratio": round(expenditure_pct / progress, 4) if progress > 0.5 else 0.0,
        "months_elapsed": round(months_elapsed, 2),
        "original_duration_months": round(duration, 2),
        "elapsed_ratio": round(elapsed_ratio, 4),
        "schedule_pressure": round(elapsed_ratio - progress / 100.0, 4),
        "months_to_revised_date": round(months_to_revised, 2),
        "revised_date_already_slipped_months": round(already_slipped, 2),
        "stall_streak": float(stall_streak),
        "progress_velocity": round(progress_velocity, 4),
        "required_velocity": round(required_velocity, 4),
        "velocity_deficit": round(velocity_deficit, 4),
    }


def _log1p(x: float) -> float:
    import math

    return math.log1p(max(0.0, x))


def build_panel(
    raw_path: Path,
    slip_days: int = DEFAULT_SLIP_DAYS,
    min_conf: float = DEFAULT_MIN_ORDER_CONFIDENCE,
    harvest: Tuple[int, int] = DEFAULT_HARVEST_MONTH,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the transition-level panel. Returns (rows, meta)."""
    with open(raw_path, "r", encoding="utf-8") as fh:
        raw_records = json.load(fh)

    by_project: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in raw_records:
        pid = rec.get("ProjectId")
        if pid:
            by_project[pid].append(rec)

    stats: Counter = Counter()
    stats["raw_records"] = len(raw_records)
    stats["unique_projects"] = len(by_project)

    # Sector frequency encoding, computed once over the whole project universe.
    sector_counts = Counter(
        canonical_sector(recs[0].get("SectorName")) for recs in by_project.values()
    )
    total_projects = max(1, sum(sector_counts.values()))
    sector_freq = {k: v / total_projects for k, v in sector_counts.items()}

    rows: List[Dict[str, Any]] = []
    concordance = Counter()

    for pid, records in sorted(by_project.items()):
        snaps = build_snapshots(records, stats)

        if not snaps or snaps[0].original_cost <= 0:
            stats["projects_dropped_no_cost"] += 1
            continue
        if len(snaps) < 2:
            stats["projects_dropped_single_snapshot"] += 1
            continue

        dated = bool(snaps and all(x.freeze_month for x in snaps))
        if dated:
            # Ordering is READ from the portal's own freeze months, so there is
            # no A1 assumption left to falsify. The progress-monotonicity ratio
            # is still computed, but it is now a DATA-QUALITY signal about the
            # feed (how often reported progress goes backwards), not a
            # confidence score in our reconstruction, and it never excludes a
            # project.
            conf = order_confidence(snaps)
            reliable = True
            if conf < min_conf:
                stats["dated_projects_with_backward_progress"] += 1
        else:
            conf = order_confidence(snaps)
            reliable = conf >= min_conf
            if not reliable:
                stats["projects_order_unreliable"] += 1

        sector = canonical_sector(records[0].get("SectorName"))
        project_name = str(modal(r.get("ProjectName") for r in records) or f"Central Project {pid}")
        n_snap = len(snaps)
        sanction_imputed = snaps[0].sanction_date is None

        for idx in range(n_snap - 1):
            cur, nxt = snaps[idx], snaps[idx + 1]

            # Cannot label a transition without both revised dates.
            if cur.revised_date is None or nxt.revised_date is None:
                stats["transitions_dropped_no_label"] += 1
                continue

            delta_days = (nxt.revised_date - cur.revised_date).days
            label = 1 if delta_days > slip_days else 0
            concordance["pushed" if delta_days > 0 else ("pulled" if delta_days < 0 else "flat")] += 1

            # Multi-horizon labels, measured against the date on the record at
            # time t. A slip anywhere in the window counts; an exhausted window
            # with no slip is censored (unknown), not a negative.
            #
            # In dated mode the window is a real CALENDAR window: "does the
            # date move within k months?" In reconstructed mode it can only be
            # "within the next k reports", which is the same thing only when
            # reporting is perfectly monthly and gapless.
            horizon_values: Dict[str, Any] = {}
            if dated:
                cur_ord = cur.month_ordinal()
                last_ord = snaps[-1].month_ordinal()
            for k in HORIZONS:
                if dated:
                    window = [
                        x for x in snaps[idx + 1:]
                        if cur_ord < x.month_ordinal() <= cur_ord + k
                    ]
                    window_complete = last_ord >= cur_ord + k
                else:
                    window = snaps[idx + 1: idx + 1 + k]
                    window_complete = len(window) >= k
                observed = [x for x in window if x.revised_date is not None]
                slipped = any(
                    (x.revised_date - cur.revised_date).days > slip_days for x in observed
                )
                censored = (not slipped) and not window_complete
                horizon_values[f"slip_within_{k}m"] = (
                    "" if censored else (1 if slipped else 0)
                )
                horizon_values[f"censored_{k}m"] = int(censored)
                if censored:
                    stats[f"censored_{k}m"] += 1

            if dated:
                as_of_month = cur.freeze_month
                as_of = cur.as_of_date()
                gap_months = nxt.month_ordinal() - cur.month_ordinal()
            else:
                months_back = (n_snap - 1) - idx
                as_of_month = month_index(harvest, months_back)
                as_of = datetime(int(as_of_month[:4]), int(as_of_month[5:]), 15)
                gap_months = 1

            feats = compute_features(snaps, idx, as_of)

            row: Dict[str, Any] = {
                "project_id": f"MOSPI_{pid}",
                "project_name": project_name,
                "sector": sector,
                "seq_index": idx,
                "n_snapshots": n_snap,
                "as_of_month": as_of_month,
                "order_confidence": round(conf, 4),
                "order_reliable": int(reliable),
                "revised_date_t": cur.revised_date.strftime("%Y-%m-%d"),
                "revised_date_next": nxt.revised_date.strftime("%Y-%m-%d"),
                "sanction_imputed": int(sanction_imputed),
                "window_truncated": int(n_snap <= 3),
                "time_axis": "observed" if dated else "reconstructed",
                "gap_months": gap_months,
                LABEL_COLUMN: label,
            }
            row.update(horizon_values)
            row.update(feats)
            rows.append(row)
            stats["transitions_emitted"] += 1
            stats["positives"] += label

    positives = stats["positives"]
    emitted = stats["transitions_emitted"]
    time_axis_mode = (
        "observed"
        if stats.get("dated_projects", 0) and not stats.get("reconstructed_projects", 0)
        else ("mixed" if stats.get("dated_projects", 0) else "reconstructed")
    )

    meta: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(raw_path.relative_to(ROOT)) if raw_path.is_relative_to(ROOT) else str(raw_path),
        "source_sha256": _sha256(raw_path),
        "slip_threshold_days": slip_days,
        "min_order_confidence": min_conf,
        "harvest_month": f"{harvest[0]:04d}-{harvest[1]:02d}",
        "counts": dict(stats),
        "positive_rate": round(positives / emitted, 4) if emitted else 0.0,
        "revised_date_movement": dict(concordance),
        "time_axis": {
            "mode": time_axis_mode,
            "dated_projects": stats.get("dated_projects", 0),
            "reconstructed_projects": stats.get("reconstructed_projects", 0),
            "explanation": (
                "OBSERVED: every record carries the portal's own freeze month "
                "(scripts/harvest_monthly.py), so snapshot order is read from the calendar "
                "and assumptions A1/A2 below do not apply. Horizon windows are real calendar "
                "months and `gap_months` records the actual spacing between consecutive "
                "reports."
                if time_axis_mode == "observed" else
                "RECONSTRUCTED: records carry no reporting month, so order is inferred under "
                "assumptions A1/A2 below. Run scripts/harvest_monthly.py to replace this with "
                "an observed time axis."
            ),
            "projects_with_backward_reported_progress": stats.get(
                "dated_projects_with_backward_progress", 0
            ),
        },
        "assumptions_apply": time_axis_mode != "observed",
        "assumptions": {
            "A1_monotone_ordering": (
                "Cumulative expenditure and physical progress are non-decreasing in real "
                "time within a project; snapshots are ordered by (expenditure, progress, "
                "revised_date). The PAIMANA feed returns Month=null/Year=null on every "
                "record, so this order is reconstructed, not observed."
            ),
            "A2_monthly_calendar_stamp": (
                f"Snapshots are monthly and the newest corresponds to {harvest[0]:04d}-"
                f"{harvest[1]:02d}. Earlier snapshots are stamped one month apart. Stamps "
                "are approximate and used only for out-of-time splitting and for "
                "expressing lead time in months."
            ),
        },
        "fields_absent_from_live_feed": _absent_field_report(raw_records),
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        # Retained for reporting only -- NOT a model feature. A sector
        # frequency encoding was tried and removed: it correlates -0.70 with
        # n_snapshots, and within history-length strata its AUC collapses to
        # 0.34-0.56 (flipping direction between strata), so it was reading the
        # history-length confound rather than sector behaviour. Dropping it
        # costs 0.014 AUC and removes a contaminated input.
        "sector_frequency_reference_only": {k: round(v, 6) for k, v in sorted(sector_freq.items())},
        "horizon_labels": HORIZON_LABELS,
        "horizon_months": HORIZONS,
        "censoring_policy": (
            "Horizon labels are left empty (NaN) when the remaining observation "
            "window is shorter than the horizon and no slip occurred inside it. "
            "Censored rows are dropped at training time for that horizon; they "
            "are never coded as negatives."
        ),
    }
    return rows, meta


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _absent_field_report(raw_records: List[Dict[str, Any]]) -> Dict[str, str]:
    """Record which CUF fields the public feed does not actually populate.

    This is deliberately surfaced rather than papered over: several downstream
    features (NLP delay-reason mining, state-level rollups, statutory clearance
    pendency) have no input in the public feed, and the platform must say so
    instead of imputing plausible values.
    """
    watched = [
        "Remarks",
        "RevisedDateReason",
        "RevisedCostReason",
        "StateName",
        "DELAYED_TIME",
        "COST_OVERRUN_PERC",
        "COR_PERC",
        "TOR_PERC",
        "StartDate",
        "AgencyName",
    ]
    total = max(1, len(raw_records))
    report: Dict[str, str] = {}
    for key in watched:
        populated = sum(1 for r in raw_records if r.get(key) not in (None, "", 0))
        report[key] = f"{populated}/{total} populated ({populated / total:.1%})"
    return report


def write_panel(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PANEL_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PANEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PAIMANA longitudinal panel.")
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--slip-days", type=int, default=DEFAULT_SLIP_DAYS)
    parser.add_argument("--min-order-confidence", type=float, default=DEFAULT_MIN_ORDER_CONFIDENCE)
    args = parser.parse_args()

    if not args.raw.exists():
        raise SystemExit(
            f"Raw harvest not found at {args.raw}. "
            "Run scripts/download_real_paimana_data.py first."
        )

    logger.info("Reconstructing panel from %s", args.raw)
    rows, meta = build_panel(args.raw, args.slip_days, args.min_order_confidence)
    write_panel(rows, meta)

    counts = meta["counts"]
    logger.info("Panel written to %s", PANEL_PATH)
    logger.info(
        "  %d transitions from %d projects (%d duplicate snapshots dropped)",
        counts.get("transitions_emitted", 0),
        counts.get("unique_projects", 0),
        counts.get("duplicate_snapshots_dropped", 0),
    )
    logger.info(
        "  positive rate (revised date pushed > %dd): %.2f%%",
        args.slip_days,
        100 * meta["positive_rate"],
    )
    logger.info(
        "  order-unreliable projects excluded from training: %d",
        counts.get("projects_order_unreliable", 0),
    )
    logger.info("  diagnostics written to %s", META_PATH)


if __name__ == "__main__":
    main()
