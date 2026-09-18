"""Translate a CUF `ProjectInput` snapshot into the v2 model's design matrix.

Implements contracts/inference_adapter.md.

This module exists because the v1 failure was a train/serve mismatch: the
booster was trained with `has_revised_doc` carrying 34% of its gain, then that
column was zeroed at inference, pushing every live project into leaf regions
whose training base rate was near zero. The formulas below therefore mirror
scripts/build_panel.py exactly, and every value that *cannot* be derived from a
single monthly snapshot is reported as an approximation rather than quietly
defaulted.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pandas as pd

from app.paimana_contracts import ProjectInput

# Features that need project history the CUF snapshot does not carry.
# See contracts/inference_adapter.md "Documented approximations".
HISTORY_APPROXIMATIONS: Dict[str, str] = {
    "progress_delta_3m": (
        "Approximated as 3x the most recent month's progress delta; the training "
        "panel measured an actual 3-month window."
    ),
    "progress_velocity": (
        "Approximated from the single most recent month; the training panel "
        "averaged over a 3-month window."
    ),
    "stall_streak": (
        "Capped at 1 (stalled / not stalled). The training panel counted "
        "consecutive stalled months, so served values cannot exceed 1."
    ),
}


def build_feature_row(
    project: ProjectInput,
    feature_columns: List[str],
    sector_frequency: Dict[str, float],
    medians: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return (one-row frame in `feature_columns` order, approximations used)."""
    original_cost = max(1.0, float(project.original_cost))
    progress = float(project.physical_progress)
    expenditure = float(project.expenditure)
    duration = max(1.0, float(project.original_duration_months))
    elapsed = float(project.months_elapsed)
    delay = float(project.current_delay_months)

    expenditure_pct = 100.0 * expenditure / original_cost
    expenditure_delta = float(project.expenditure_change_recent or 0.0)
    progress_delta = float(project.progress_change_recent or 0.0)
    remaining = max(0.0, 100.0 - progress)
    elapsed_ratio = elapsed / duration

    # The currently-declared completion date, expressed as months from now:
    # sanction + original duration + accumulated slippage - elapsed. Negative
    # means the declared date has already passed, which is meaningful and is
    # deliberately not clipped.
    months_to_revised = duration + delay - elapsed

    progress_velocity = progress_delta
    required_velocity = remaining / months_to_revised if months_to_revised > 0 else remaining

    derived: Dict[str, float] = {
        "original_cost": float(project.original_cost),
        "log_original_cost": math.log1p(max(0.0, float(project.original_cost))),
        "expenditure": expenditure,
        "expenditure_pct": expenditure_pct,
        "expenditure_delta": expenditure_delta,
        "expenditure_delta_pct": 100.0 * expenditure_delta / original_cost,
        "physical_progress": progress,
        "progress_delta": progress_delta,
        "progress_delta_3m": 3.0 * progress_delta,
        "remaining_progress_pct": remaining,
        "spend_progress_gap": expenditure_pct - progress,
        "burn_to_progress_ratio": (expenditure_pct / progress) if progress > 0.5 else 0.0,
        "months_elapsed": elapsed,
        "original_duration_months": duration,
        "elapsed_ratio": elapsed_ratio,
        "schedule_pressure": elapsed_ratio - progress / 100.0,
        "months_to_revised_date": months_to_revised,
        "revised_date_already_slipped_months": delay,
        "stall_streak": 1.0 if progress_delta <= 0.1 else 0.0,
        "progress_velocity": progress_velocity,
        "required_velocity": required_velocity,
        "velocity_deficit": required_velocity - progress_velocity,
    }

    approximations = {
        name: note
        for name, note in HISTORY_APPROXIMATIONS.items()
        if name in feature_columns
    }

    row: Dict[str, Any] = {}
    for col in feature_columns:
        if col in derived:
            row[col] = derived[col]
        else:
            # The bundle asks for a column this adapter does not know how to
            # derive (model retrained with new features). Fill from the
            # training-fold median and say so -- never silently zero-fill,
            # which is the v1 mistake in a new costume.
            row[col] = float(medians.get(col, 0.0))
            approximations[col] = (
                "Not derivable from a CUF snapshot; filled with the training-fold median."
            )

    frame = pd.DataFrame([row])[feature_columns]
    # Guard against NaN reaching the booster from an unexpected division.
    frame = frame.fillna(pd.Series(medians)).fillna(0.0)
    return frame, approximations
