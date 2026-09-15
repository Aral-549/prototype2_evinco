"""Loader and cache for real MoSPI PAIMANA infrastructure projects.

Reads the ingested national infrastructure monitoring panel from
data/paimana_real_projects.json (harvested directly from https://paimana-proj.mospi.gov.in/).
"""

import json
import logging
from typing import List, Optional

from app.config import settings
from app.paimana_contracts import ProjectGovernanceAssessment, ProjectInput
from app.services.governance_service import evaluate_project
from app.services.mock_data import MOCK_PAIMANA_PROJECTS

logger = logging.getLogger("paimana_real_data")

_CACHED_PROJECTS: Optional[List[ProjectInput]] = None
_CACHED_EVALUATED: Optional[List[ProjectGovernanceAssessment]] = None


def load_real_projects() -> List[ProjectInput]:
    """Load and validate real infrastructure projects from the harvested dataset."""
    global _CACHED_PROJECTS
    if _CACHED_PROJECTS is not None:
        return _CACHED_PROJECTS

    data_file = settings.REAL_DATA_PATH
    if not data_file.exists():
        logger.warning(
            f"Real PAIMANA dataset not found at {data_file}. Falling back to baseline projects."
        )
        _CACHED_PROJECTS = list(MOCK_PAIMANA_PROJECTS)
        return _CACHED_PROJECTS

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        projects: List[ProjectInput] = []
        for item in raw_list:
            try:
                projects.append(ProjectInput(**item))
            except Exception as exc:
                logger.debug(f"Skipping invalid project entry {item.get('project_id')}: {exc}")

        if projects:
            logger.info(f"Loaded {len(projects)} real MoSPI PAIMANA projects from {data_file}")
            _CACHED_PROJECTS = projects
            return _CACHED_PROJECTS
        else:
            logger.warning("Real PAIMANA dataset empty. Falling back to baseline projects.")
            _CACHED_PROJECTS = list(MOCK_PAIMANA_PROJECTS)
            return _CACHED_PROJECTS
    except Exception as exc:
        logger.error(f"Failed to load real PAIMANA dataset: {exc}. Falling back to baseline.")
        _CACHED_PROJECTS = list(MOCK_PAIMANA_PROJECTS)
        return _CACHED_PROJECTS


def get_evaluated_portfolio() -> List[ProjectGovernanceAssessment]:
    """Return pre-evaluated assessments for the entire monitored national portfolio."""
    global _CACHED_EVALUATED
    if _CACHED_EVALUATED is not None:
        return _CACHED_EVALUATED

    projects = load_real_projects()
    evaluated = [evaluate_project(p, include_drivers=False) for p in projects]
    # Sort descending by Capital-at-Risk
    evaluated.sort(key=lambda x: x.capital_at_risk_crores, reverse=True)
    _CACHED_EVALUATED = evaluated
    return _CACHED_EVALUATED


def invalidate_cache():
    """Clear in-memory cache to reload freshly ingested project datasets."""
    global _CACHED_PROJECTS, _CACHED_EVALUATED
    _CACHED_PROJECTS = None
    _CACHED_EVALUATED = None
