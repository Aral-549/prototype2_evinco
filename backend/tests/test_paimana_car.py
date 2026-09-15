"""Unit tests for the Capital-at-Risk engine (Spec Sections 2.3, 2.4, 3.2.2)."""

import pytest

from app.paimana_car import (
    NATIONAL_MACRO_MEDIAN_OVERRUN,
    SECTOR_MEDIAN_OVERRUN_LOOKUP,
    calculate_capital_at_risk,
    calculate_capital_at_risk_gov,
    resolve_sector_median,
)


class TestSectorMedianResolution:
    def test_all_22_sectors_present(self):
        assert len(SECTOR_MEDIAN_OVERRUN_LOOKUP) == 22
        assert "Other" in SECTOR_MEDIAN_OVERRUN_LOOKUP

    def test_spec_values(self):
        assert SECTOR_MEDIAN_OVERRUN_LOOKUP["Railways"] == 25.0
        assert SECTOR_MEDIAN_OVERRUN_LOOKUP["Road Transport & Highways"] == 12.0
        assert SECTOR_MEDIAN_OVERRUN_LOOKUP["Power"] == 18.0
        assert SECTOR_MEDIAN_OVERRUN_LOOKUP["Telecommunications"] == 5.0
        assert SECTOR_MEDIAN_OVERRUN_LOOKUP["Other"] == 15.0

    def test_unknown_sector_falls_back_to_national_median(self):
        assert resolve_sector_median("Interstellar Tunnels") == NATIONAL_MACRO_MEDIAN_OVERRUN == 15.0

    def test_alias_resolution(self):
        assert resolve_sector_median("railway") == 25.0
        assert resolve_sector_median("roads") == 12.0
        assert resolve_sector_median("telecom") == 5.0

    def test_case_insensitive_resolution(self):
        assert resolve_sector_median("railways") == 25.0
        assert resolve_sector_median("COAL") == 10.0


class TestCaRCalculation:
    def test_month6_zero_overrun_blindspot_resolved(self):
        # Spec 2.3 item 1: zero reported overrun must NOT yield zero CaR
        r = calculate_capital_at_risk(
            original_cost=10000.0,
            p_model=0.9,
            cost_overrun_pct_current=0.0,
            sector="Railways",
        )
        assert r.effective_overrun_pct == 25.0
        assert r.is_sector_fallback_used is True
        assert r.capital_at_risk_crores == pytest.approx(10000.0 * 0.9 * 0.25, abs=0.01)

    def test_negative_overrun_clamped_to_sector_median(self):
        # Spec 2.3 item 2: interim billing discounts cannot erase exposure
        r = calculate_capital_at_risk(
            original_cost=1000.0,
            p_model=0.5,
            cost_overrun_pct_current=-5.0,
            sector="Coal",
        )
        assert r.effective_overrun_pct == 10.0
        assert r.is_sector_fallback_used is True

    def test_runaway_escalation_scales(self):
        # Spec 2.3 item 3: current overrun above median takes precedence
        r = calculate_capital_at_risk(
            original_cost=1000.0,
            p_model=0.5,
            cost_overrun_pct_current=180.0,
            sector="Railways",
        )
        assert r.effective_overrun_pct == 180.0
        assert r.is_sector_fallback_used is False
        assert r.capital_at_risk_crores == pytest.approx(1000.0 * 0.5 * 1.80, abs=0.01)

    def test_fiscal_cap_300pct(self):
        # Spec 2.3 item 4: pathological data entry must clamp to 3x C0
        r = calculate_capital_at_risk(
            original_cost=1000.0,
            p_model=1.0,
            cost_overrun_pct_current=500.0,
            sector="Railways",
        )
        assert r.is_clamped_to_cap is True
        assert r.capital_at_risk_crores == 3000.0

    def test_zero_probability_zero_car(self):
        r = calculate_capital_at_risk(
            original_cost=1000.0,
            p_model=0.0,
            cost_overrun_pct_current=50.0,
            sector="Railways",
        )
        assert r.capital_at_risk_crores == 0.0
        assert r.effective_overrun_pct == 0.0

    def test_non_positive_cost_zero_car(self):
        r = calculate_capital_at_risk(
            original_cost=0.0,
            p_model=0.8,
            cost_overrun_pct_current=50.0,
            sector="Railways",
        )
        assert r.capital_at_risk_crores == 0.0

    def test_boundary_cap_not_exceeded_exactly_300(self):
        # effective overrun 300% exactly -> raw = 1.0 * 1.0 * 3.0 = cap, not clamped
        r = calculate_capital_at_risk(
            original_cost=1000.0,
            p_model=1.0,
            cost_overrun_pct_current=300.0,
            sector="Railways",
        )
        assert r.is_clamped_to_cap is False
        assert r.capital_at_risk_crores == 3000.0

    def test_custom_lookup_table(self):
        r = calculate_capital_at_risk(
            original_cost=1000.0,
            p_model=0.5,
            cost_overrun_pct_current=0.0,
            sector="Railways",
            lookup_table={"Railways": 40.0},
        )
        assert r.sector_median_overrun_pct == 40.0
        assert r.capital_at_risk_crores == 200.0


class TestGovVariant:
    def test_gov_score_drives_exposure(self):
        r = calculate_capital_at_risk_gov(
            original_cost=1000.0,
            gov_score=85.0,
            cost_overrun_pct_current=0.0,
            sector="Railways",
        )
        assert r.capital_at_risk_crores == pytest.approx(1000.0 * 0.85 * 0.25, abs=0.01)

    def test_gov_score_clamped_to_unit_interval(self):
        r = calculate_capital_at_risk_gov(
            original_cost=1000.0,
            gov_score=150.0,
            cost_overrun_pct_current=0.0,
            sector="Coal",
        )
        assert r.capital_at_risk_crores == pytest.approx(1000.0 * 1.0 * 0.10, abs=0.01)
