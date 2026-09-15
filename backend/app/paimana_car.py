"""
MoSPI PAIMANA - Capital-at-Risk Engine (Spec Section 3.2.2 + 2.3 + 2.4)

    CaR_i = C0_i * P_model_i * max(current_overrun_i, mu_sector(i))

with a hard fiscal exposure cap of 300% of original sanctioned cost and the
empirical 22-sector median overrun lookup table grounded in MoSPI IPMD Flash
Reports and Ram Singh (2010, EPW Table 2).
"""

from typing import Dict, Optional

from pydantic import BaseModel

from app.paimana_contracts import SECTOR_SYNONYMS

# ---------------------------------------------------------------------------
# Empirical 22-Sector Median Overrun Lookup Table (Spec Section 2.4)
# Values are sector-median cost overrun percentages (%) from MoSPI IPMD
# longitudinal panels (461st Flash Report, March 2024) and Ram Singh (2010).
# ---------------------------------------------------------------------------

SECTOR_MEDIAN_OVERRUN_LOOKUP: Dict[str, float] = {
    "Railways": 25.0,                       # SEC-01
    "Road Transport & Highways": 12.0,      # SEC-02
    "Power": 18.0,                          # SEC-03
    "Urban Development": 15.0,              # SEC-04
    "Water Resources": 20.0,                # SEC-05
    "Petroleum & Natural Gas": 8.0,         # SEC-06
    "Coal": 10.0,                           # SEC-07
    "Atomic Energy": 15.0,                  # SEC-08
    "Civil Aviation": 10.0,                 # SEC-09
    "Shipping & Ports": 10.0,               # SEC-10
    "Telecommunications": 5.0,              # SEC-11
    "Steel": 8.0,                           # SEC-12
    "Fertilizers": 8.0,                     # SEC-13
    "Mines": 8.0,                           # SEC-14
    "Information & Broadcasting": 10.0,     # SEC-15
    "Health & Family Welfare": 20.0,        # SEC-16
    "Petrochemicals": 8.0,                  # SEC-17
    "Heavy Industry": 10.0,                 # SEC-18
    "Defence Production": 12.0,             # SEC-19
    "Higher Education": 12.0,               # SEC-20
    "Renewable Energy": 6.0,                # SEC-21
    "Other": 15.0,                          # SEC-22 DEFAULT / NATIONAL FALLBACK
}
NATIONAL_MACRO_MEDIAN_OVERRUN: float = 15.0  # MoSPI aggregate central portfolio median

# Sector synonyms beyond the contract-level alias table (CaR-specific)
_CAR_SECTOR_SYNONYMS: Dict[str, str] = dict(SECTOR_SYNONYMS)
_CAR_SECTOR_SYNONYMS.update(
    {
        "urban development & metro rail transit": "Urban Development",
        "urban": "Urban Development",
        "metro": "Urban Development",
        "water resources / irrigation & flood control": "Water Resources",
        "irrigation": "Water Resources",
        "shipping, ports & inland waterways": "Shipping & Ports",
        "ports": "Shipping & Ports",
        "shipping": "Shipping & Ports",
        "telecom": "Telecommunications",
        "health & family welfare": "Health & Family Welfare",
        "health": "Health & Family Welfare",
        "defence": "Defence Production",
        "education": "Higher Education",
        "renewables": "Renewable Energy",
    }
)

# Hard fiscal exposure cap factor (Spec 2.3 boundary invariant #4)
CAR_FISCAL_CAP_FACTOR: float = 3.0


class CaRResult(BaseModel):
    """Structured result of the Capital-at-Risk computation."""
    capital_at_risk_crores: float
    sector_median_overrun_pct: float
    effective_overrun_pct: float
    is_sector_fallback_used: bool
    is_clamped_to_cap: bool


def resolve_sector_median(sector: str, lookup_table: Optional[Dict[str, float]] = None) -> float:
    """
    Resolves the empirical sector-median overrun percentage for a raw sector
    string, applying alias normalization and the national fallback.
    """
    table = lookup_table or SECTOR_MEDIAN_OVERRUN_LOOKUP
    clean_sector = sector.strip() if isinstance(sector, str) else str(sector)
    canonical = _CAR_SECTOR_SYNONYMS.get(clean_sector.lower(), clean_sector)
    table_lower = {k.lower(): v for k, v in table.items()}
    return table.get(
        canonical,
        table.get(
            clean_sector,
            table_lower.get(
                canonical.lower(),
                table_lower.get(clean_sector.lower(), NATIONAL_MACRO_MEDIAN_OVERRUN),
            ),
        ),
    )


def calculate_capital_at_risk(
    original_cost: float,
    p_model: float,
    cost_overrun_pct_current: float,
    sector: str,
    lookup_table: Optional[Dict[str, float]] = None,
) -> CaRResult:
    """
    Computes the expected financial exposure of the public exchequer in ₹ Crore.

    Args:
        original_cost: Sanctioned original capital outlay in ₹ Crore.
        p_model: Calibrated schedule failure probability [0.0 - 1.0].
        cost_overrun_pct_current: Current reported cost overrun percentage.
        sector: Infrastructure sector name.
        lookup_table: Optional custom lookup table for sector medians.

    Returns:
        CaRResult with rupee exposure, sector median, and effective overrun percentage.
    """
    sector_median = resolve_sector_median(sector, lookup_table)

    if original_cost <= 0.0 or p_model <= 0.0:
        return CaRResult(
            capital_at_risk_crores=0.0,
            sector_median_overrun_pct=round(sector_median, 2),
            effective_overrun_pct=0.0,
            is_sector_fallback_used=False,
            is_clamped_to_cap=False,
        )

    # Current overrun clamped to non-negative (rejects interim contractor billing discounts)
    reported_overrun = max(0.0, cost_overrun_pct_current)

    # Master CaR Operator: max(current_overrun, sector_median)
    effective_overrun = max(reported_overrun, sector_median)
    fallback_used = bool(reported_overrun < sector_median)

    # Raw expected capital exposure
    car_raw = original_cost * p_model * (effective_overrun / 100.0)

    # Hard fiscal exposure cap: CaR cannot exceed 300% of original sanctioned cost
    fiscal_cap = original_cost * CAR_FISCAL_CAP_FACTOR
    clamped_to_cap = bool(car_raw > fiscal_cap)
    final_car = min(fiscal_cap, car_raw)

    return CaRResult(
        capital_at_risk_crores=round(final_car, 2),
        sector_median_overrun_pct=round(sector_median, 2),
        effective_overrun_pct=round(effective_overrun, 2),
        is_sector_fallback_used=fallback_used,
        is_clamped_to_cap=clamped_to_cap,
    )


def calculate_capital_at_risk_gov(
    original_cost: float,
    gov_score: float,
    cost_overrun_pct_current: float,
    sector: str,
    lookup_table: Optional[Dict[str, float]] = None,
) -> CaRResult:
    """
    Executive governance variant (Spec 2.3 item 5): CaR driven by the composite
    GovScore rather than the raw ML probability, for Cabinet Secretariat /
    PRAGATI review contexts where administrative compliance takes precedence.
    """
    return calculate_capital_at_risk(
        original_cost=original_cost,
        p_model=max(0.0, min(1.0, gov_score / 100.0)),
        cost_overrun_pct_current=cost_overrun_pct_current,
        sector=sector,
        lookup_table=lookup_table,
    )
