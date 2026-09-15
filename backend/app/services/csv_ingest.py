"""CSV ingestion for batch scoring of PAIMANA monthly report exports.

Header matching is case/space-insensitive with a synonym table covering
common PAIMANA / CUF export column names. The governance trigger columns
(milestones, clearance pendency, reporting staleness, dispute status)
feed the RuleFloor engine (Spec Section 3.2.1).
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import pandas as pd
from pydantic import ValidationError

from app.paimana_contracts import ProjectInput


class CsvIngestError(ValueError):
    """Raised when an uploaded CSV cannot be interpreted as a project report."""


# Exact-match synonym table: PAIMANA export header -> ProjectInput field.
# Keys are matched after stripping; matching is case- and space-insensitive.
COLUMN_SYNONYMS: Dict[str, str] = {
    "projectid": "project_id",
    "projectcode": "project_id",
    "projectname": "project_name",
    "sector": "sector",
    "sectorname": "sector",
    "implementingagency": "implementing_agency",
    "agency": "implementing_agency",
    "state": "state",
    "originalcost": "original_cost",
    "sanctionedcost": "original_cost",
    "revisedcost": "revised_cost",
    "expenditure": "expenditure",
    "originalexp": "expenditure",
    "expenditurechangerecent": "expenditure_change_recent",
    "recentexpenditure": "expenditure_change_recent",
    "physicalprogress": "physical_progress",
    "progress": "physical_progress",
    "progresschange": "progress_change_recent",
    "progresschangerecent": "progress_change_recent",
    "originaldurationmonths": "original_duration_months",
    "monthselapsed": "months_elapsed",
    "currentdelaymonths": "current_delay_months",
    "delaymonths": "current_delay_months",
    "overduemilestones": "overdue_milestones",
    "totalscheduledmilestones": "total_scheduled_milestones",
    "scheduledmilestones": "total_scheduled_milestones",
    "clearancependingdays": "clearance_pending_days",
    "dayssincelastupdate": "days_since_last_update",
    "disputestatus": "dispute_status",
    "delayremarks": "delay_remarks",
    "reasonsfordelay": "delay_remarks",
    "remarks": "delay_remarks",
}

NUMERIC_FIELDS = {
    "original_cost",
    "revised_cost",
    "expenditure",
    "expenditure_change_recent",
    "physical_progress",
    "progress_change_recent",
    "original_duration_months",
    "months_elapsed",
    "current_delay_months",
    "overdue_milestones",
    "total_scheduled_milestones",
    "clearance_pending_days",
    "days_since_last_update",
}

TEXT_FIELDS = {
    "project_id",
    "project_name",
    "sector",
    "implementing_agency",
    "state",
    "dispute_status",
    "delay_remarks",
}

INT_FIELDS = {"overdue_milestones", "total_scheduled_milestones", "clearance_pending_days", "days_since_last_update"}

VALID_DISPUTE_STATUSES = {"NONE", "CONCILIATION", "ARBITRATION", "HIGH_COURT_STAY", "TERMINATION_NOTICE"}


def _normalize_header(col: str) -> str:
    return str(col).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def csv_to_project_inputs(file_bytes: bytes) -> Tuple[List[ProjectInput], Dict[str, Any]]:
    """Parse a PAIMANA report CSV into ProjectInput items.

    - Header matching is case/space-insensitive with a synonym table for
      common PAIMANA column names.
    - Unknown columns are ignored and reported back in `columns_ignored`.
    - Numeric cells that fail coercion are treated as field defaults.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise CsvIngestError(f"could not parse CSV: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise CsvIngestError("CSV contains no data rows")

    # Map normalized header -> actual column name (first occurrence wins)
    header_map: Dict[str, str] = {}
    for col in df.columns:
        norm = _normalize_header(col)
        field = COLUMN_SYNONYMS.get(norm, norm if norm in NUMERIC_FIELDS | TEXT_FIELDS else None)
        if field and field not in header_map.values():
            header_map[col] = field

    if "project_id" not in header_map.values() and "original_cost" not in header_map.values():
        raise CsvIngestError(
            "CSV header not recognized: need at least a project id (or original_cost) column"
        )

    columns_ignored = sorted(set(df.columns) - set(header_map))

    items: List[ProjectInput] = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        kwargs: Dict[str, Any] = {}
        for col, field in header_map.items():
            value = row[col]
            if field in TEXT_FIELDS:
                if pd.notna(value):
                    text = str(value).strip()
                    if field == "dispute_status":
                        text = text.upper().replace(" ", "_")
                    if text:
                        kwargs[field] = text
            elif field in NUMERIC_FIELDS:
                coerced = pd.to_numeric(value, errors="coerce")
                if pd.notna(coerced):
                    kwargs[field] = int(coerced) if field in INT_FIELDS else float(coerced)
        try:
            items.append(ProjectInput(**kwargs))
        except ValidationError as exc:
            missing = [
                e["loc"][0] if e["loc"] else "?"
                for e in exc.errors()
                if e.get("type") == "missing"
            ]
            raise CsvIngestError(
                f"CSV row {row_idx + 1}: missing required fields {missing} "
                "(required: project_id, project_name, original_cost, expenditure, physical_progress)"
            ) from exc

    return items, {"columns_ignored": columns_ignored}
