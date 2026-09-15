"""
MoSPI PAIMANA - NLP Bottleneck Proxy Mining (Spec Section 2.9)

Deterministic regular-expression ontology parser over the free-text
"Reasons for Delay" remarks submitted in the monthly CUF. Extracts five
binary proxy indicators that empirically bridge the CUF information gap
(MoSPI Dimension (c)) without fabricating synthetic statistics.
"""

from __future__ import annotations

import re
from typing import Dict

NLP_BOTTLENECK_RULES: Dict[str, str] = {
    "LAND_ROW": (
        r"(?i)\b(land\s+acquisition|row|right\s+of\s+way|compensation|"
        r"slao|district\s+collector|encroachment|possession|land\s+handover)\b"
    ),
    "ENV_FOREST": (
        r"(?i)\b(forest|wildlife|moef|moefcc|stage[- ]?[iI]{1,2}|"
        r"tree\s+cutting|nbwl|clearance|afforestation|ca\s+land)\b"
    ),
    "LEGAL_CONTRACTOR": (
        r"(?i)\b(arbitrat\w*|court|litigat\w*|stay\s+order|dispute|"
        r"contractor\s+default|termination|liquidity|nclt|insolvency)\b"
    ),
    "UTILITY_INTERAGENCY": (
        r"(?i)\b(utility\s+shifting|transmission\s+line|powergrid|"
        r"railway\s+crossing|crs|pipeline\s+shifting|water\s+pipeline)\b"
    ),
    "LOCAL_GEOLOGY": (
        r"(?i)\b(law\s+and\s+order|local\s+agitation|protest|strike|"
        r"monsoon|flash\s+flood|geological\s+strata|rockfall|landslide)\b"
    ),
}

# Compiled once at import for deterministic, fast repeated evaluation.
_COMPILED_RULES: Dict[str, "re.Pattern[str]"] = {
    tag: re.compile(pattern) for tag, pattern in NLP_BOTTLENECK_RULES.items()
}


def extract_bottleneck_proxies(remarks: str | None) -> Dict[str, int]:
    """
    Extracts structured binary proxy indicators from free-text delay remarks.

    Args:
        remarks: Raw free-text delay narrative (may be None / empty).

    Returns:
        Dict mapping each of the five ontology tags to 0 or 1.
    """
    if not remarks or not isinstance(remarks, str):
        return {k: 0 for k in NLP_BOTTLENECK_RULES}
    return {
        tag: 1 if compiled.search(remarks) else 0
        for tag, compiled in _COMPILED_RULES.items()
    }
