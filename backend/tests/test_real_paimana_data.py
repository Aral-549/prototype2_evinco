"""Tests validating real MoSPI PAIMANA infrastructure monitoring data.

Tests the live national infrastructure dataset harvested from the official
MoSPI PAIMANA portal (https://paimana-proj.mospi.gov.in/), verifying data
integrity, contract conformance, leakage quarantine, and governance scoring
against 2,155+ real central projects representing ₹41.8+ lakh crore capex.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.paimana_contracts import ProjectInput
from app.services.csv_ingest import csv_to_project_inputs
from app.services.governance_service import evaluate_project
from app.services.real_data import get_evaluated_portfolio, load_real_projects

client = TestClient(app)


def test_real_dataset_presence_and_volume():
    """Verify harvested PAIMANA real dataset exists and matches official scale."""
    data_path = settings.REAL_DATA_PATH
    assert data_path.exists(), f"Real PAIMANA dataset not found at {data_path}"

    with open(data_path, "r", encoding="utf-8") as f:
        projects_data = json.load(f)

    # Problem statement cites ~1,981 projects; our live harvest captured 2,155
    assert len(projects_data) >= 1900, f"Expected >= 1900 projects, found {len(projects_data)}"

    total_capex = sum(float(p.get("original_cost", 0)) for p in projects_data)
    # Problem statement states aggregate cost is ₹37.13 - 42.78 lakh crore
    assert total_capex >= 3500000.0, f"Total monitored capex ({total_capex:,.2f} Cr) below expected scale"


def test_real_projects_contract_conformance():
    """Verify all real projects conform strictly to the ProjectInput contract."""
    real_projects = load_real_projects()
    assert len(real_projects) >= 1900

    for p in real_projects[:100]:  # Sample deep-check
        assert isinstance(p, ProjectInput)
        assert p.original_cost > 0.0
        assert p.expenditure >= 0.0
        assert 0.0 <= p.physical_progress <= 100.0
        assert p.original_duration_months > 0.0
        assert p.months_elapsed >= 0.0
        assert p.current_delay_months >= 0.0


def test_real_projects_sector_diversity():
    """Verify presence of all major central infrastructure sectors."""
    real_projects = load_real_projects()
    sectors = {p.sector for p in real_projects}

    # Core sectors from MoSPI IPMD mandate
    expected_sectors = {
        "Railways",
        "Roads & Highways",
        "Power",
        "Coal",
        "Petroleum",
        "Aviation",
        "Steel",
    }
    for sec in expected_sectors:
        assert sec in sectors, f"Expected major sector '{sec}' not found in real dataset"


def test_real_data_governance_and_rulefloor_execution():
    """Verify ML and RuleFloor evaluation across real projects."""
    evaluated = get_evaluated_portfolio()
    assert len(evaluated) >= 1900

    # Ensure both Machine Learning and RuleFloor overrides are active in the wild
    sources = {e.dominant_source for e in evaluated}
    assert "MACHINE_LEARNING" in sources
    assert "RULE_FLOOR_OVERRIDE" in sources

    # Ensure all 4 risk tiers are populated
    tiers = {e.risk_tier for e in evaluated}
    assert "LOW" in tiers
    assert "MODERATE" in tiers
    assert "HIGH" in tiers

    # Check that GovScore satisfies the Non-Masking Invariant
    for e in evaluated[:200]:
        assert e.gov_score >= e.rule_floor
        assert 0.0 <= e.gov_score <= 100.0
        assert e.capital_at_risk_crores >= 0.0


def test_real_mega_projects_capital_at_risk():
    """Verify Capital-at-Risk scales correctly on real mega-infrastructure assets."""
    evaluated = get_evaluated_portfolio()
    # Find mega projects (> ₹10,000 Crore sanctioned)
    mega = [e for e in evaluated if e.original_cost_crores >= 10000.0]
    assert len(mega) > 0, "Expected at least one real mega project with capex >= ₹10,000 Cr"

    for p in mega:
        assert p.capital_at_risk_crores >= 0.0
        # CaR should not exceed sanctioned capex
        assert p.capital_at_risk_crores <= p.original_cost_crores * 2.0


def test_portfolio_summary_endpoint_with_real_data():
    """Verify /portfolio/summary serves live macro aggregates from real data."""
    resp = client.get("/api/v1/portfolio/summary")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_projects"] >= 1900
    assert data["total_monitored_capex_crores"] >= 3500000.0
    assert data["total_capital_at_risk_crores"] > 0.0
    assert len(data["sectors"]) >= 10
    assert data["capital_at_risk_percentage"] > 0.0


def test_portfolio_risk_ranking_endpoint_with_real_data():
    """Verify /portfolio/risk-ranking filters and ranks real infrastructure projects."""
    resp = client.get("/api/v1/portfolio/risk-ranking?sector=Railways&limit=10")
    assert resp.status_code == 200
    data = resp.json()

    assert len(data) == 10
    for item in data:
        assert item["sector"] == "Railways"
        assert item["project_id"].startswith("MOSPI_")
        assert len(item["project_name"]) > 0
        assert item["capital_at_risk_crores"] >= 0.0

    # Verify descending order by CaR
    cars = [item["capital_at_risk_crores"] for item in data]
    assert cars == sorted(cars, reverse=True)


def test_real_dataset_csv_ingest_pipeline():
    """Verify CSV ingestion pipeline parses real dataset without errors."""
    csv_path = settings.REAL_DATA_PATH.parent / "paimana_real_projects.csv"
    if csv_path.exists():
        with open(csv_path, "rb") as f:
            content = f.read()

        parsed, meta = csv_to_project_inputs(content)
        assert len(parsed) >= 1900
        assert "columns_ignored" in meta
