#!/usr/bin/env python3
"""Downloader and parser for real MoSPI PAIMANA project data.

Fetches the live national infrastructure project registry directly from
the official MoSPI PAIMANA portal:
https://paimana-proj.mospi.gov.in/

Transforms raw government records into typed ProjectInput models ready
for ML inference, RuleFloor governance, and Capital-at-Risk evaluation.
"""

import csv
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paimana_downloader")

BASE_URL = "https://paimana-proj.mospi.gov.in"
DASHBOARD_URL = f"{BASE_URL}/Home/PublicDashboardNew"
SECTOR_LIST_URL = f"{BASE_URL}/Home/GetSectorList"
STATE_LIST_URL = f"{BASE_URL}/Home/GetStateList"
TILE_DATA_URL = f"{BASE_URL}/Home/GetTileData"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "paimana_live_raw.json"
PROJECTS_JSON_PATH = DATA_DIR / "paimana_real_projects.json"
PROJECTS_CSV_PATH = DATA_DIR / "paimana_real_projects.csv"

# Pre-verified sector catalog from live PAIMANA endpoint
FALLBACK_SECTORS = [
    {"Value": 110, "Text": "Aviation & Aviation Infrastructure"},
    {"Value": 121, "Text": "Chemicals"},
    {"Value": 123, "Text": "Coal"},
    {"Value": 115, "Text": "Construction "},
    {"Value": 31, "Text": "Education"},
    {"Value": 3, "Text": "Electricity Generation"},
    {"Value": 103, "Text": "Energy Storage"},
    {"Value": 1, "Text": "Food Processing & Agriculture"},
    {"Value": 113, "Text": "Healthcare"},
    {"Value": 10, "Text": "Inland Waterways"},
    {"Value": 7, "Text": "Logistics Infrastructure"},
    {"Value": 122, "Text": "Metals & Mining"},
    {"Value": 12, "Text": "Oil & Gas"},
    {"Value": 108, "Text": "Railways"},
    {"Value": 9, "Text": "Real Estate"},
    {"Value": 107, "Text": "Roads & Highways"},
    {"Value": 109, "Text": "Shipping"},
    {"Value": 114, "Text": "Sports"},
    {"Value": 104, "Text": "Steel"},
    {"Value": 30, "Text": "Telecommunication"},
    {"Value": 8, "Text": "Tourism, Hospitality & Wellness"},
    {"Value": 102, "Text": "Transmission & Distribution"},
    {"Value": 111, "Text": "Urban Public Transport"},
    {"Value": 106, "Text": "Utility and Resources Pipelines"},
    {"Value": 18, "Text": "Waste & Water"},
    {"Value": 112, "Text": "Water Resources"},
]

FALLBACK_STATES = [
    {"Value": 47, "Text": "Andaman & Nicobar"},
    {"Value": 1, "Text": "Andhra Pradesh"},
    {"Value": 2, "Text": "Arunachal Pradesh"},
    {"Value": 3, "Text": "Assam"},
    {"Value": 4, "Text": "Bihar"},
    {"Value": 5, "Text": "Chandigarh"},
    {"Value": 6, "Text": "Chhattisgarh"},
    {"Value": 7, "Text": "Dadra & Nagar Haveli and Daman & Diu"},
    {"Value": 8, "Text": "Delhi"},
    {"Value": 9, "Text": "Goa"},
    {"Value": 10, "Text": "Gujarat"},
    {"Value": 11, "Text": "Haryana"},
    {"Value": 12, "Text": "Himachal Pradesh"},
    {"Value": 13, "Text": "Jammu & Kashmir"},
    {"Value": 14, "Text": "Jharkhand"},
    {"Value": 15, "Text": "Karnataka"},
    {"Value": 16, "Text": "Kerala"},
    {"Value": 17, "Text": "Ladakh"},
    {"Value": 18, "Text": "Lakshadweep"},
    {"Value": 19, "Text": "Madhya Pradesh"},
    {"Value": 20, "Text": "Maharashtra"},
    {"Value": 21, "Text": "Manipur"},
    {"Value": 22, "Text": "Meghalaya"},
    {"Value": 23, "Text": "Mizoram"},
    {"Value": 24, "Text": "Nagaland"},
    {"Value": 25, "Text": "Odisha"},
    {"Value": 26, "Text": "Puducherry"},
    {"Value": 27, "Text": "Punjab"},
    {"Value": 28, "Text": "Rajasthan"},
    {"Value": 29, "Text": "Sikkim"},
    {"Value": 30, "Text": "Tamil Nadu"},
    {"Value": 31, "Text": "Telangana"},
    {"Value": 32, "Text": "Tripura"},
    {"Value": 33, "Text": "Uttar Pradesh"},
    {"Value": 34, "Text": "Uttarakhand"},
    {"Value": 35, "Text": "West Bengal"},
]


def create_session() -> tuple[requests.Session, str]:
    """Initialize resilient session and extract ASP.NET RequestVerificationToken."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": DASHBOARD_URL,
    }
    session.headers.update(headers)

    logger.info(f"Connecting to MoSPI PAIMANA portal at {DASHBOARD_URL}...")
    try:
        resp = session.get(DASHBOARD_URL, verify=False, timeout=30)
        resp.raise_for_status()
        match = re.search(r'name="__RequestVerificationToken"\s+type="hidden"\s+value="([^"]+)"', resp.text)
        if not match:
            match = re.search(r'value="([^"]+)"\s+name="__RequestVerificationToken"', resp.text)
        token = match.group(1) if match else ""
        logger.info(f"Retrieved Anti-Forgery Token (prefix: {token[:12]}...)")
    except Exception as exc:
        logger.warning(f"Could not load dashboard page ({exc}); proceeding with empty token")
        token = ""

    return session, token


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date from common Indian government formats."""
    if not date_str or not isinstance(date_str, str):
        return None
    cleaned = date_str.strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass
    return None


def fetch_tile_data(session: requests.Session, token: str, sector_id: str,
                    state_id: str = "", cost_range: str = "") -> Optional[List[Dict[str, Any]]]:
    """Query GetTileData endpoint for a given sector/state/cost slice."""
    data = {
        "__RequestVerificationToken": token,
        "SectorId": str(sector_id),
        "PROJ_MINISTRY_ID": "",
        "StateId": str(state_id),
        "CostRange": str(cost_range),
        "Month": "",
        "Year": "",
        "MonthYear": "",
    }
    for attempt in range(2):
        try:
            resp = session.post(TILE_DATA_URL, data=data, verify=False, timeout=30)
            if resp.status_code == 200:
                payload = resp.json()
                return payload.get("data", {}).get("ProjectsCountTabDetails", []) or []
            elif resp.status_code == 500:
                # Payload exceeded maxJsonLength in ASP.NET
                return None
            else:
                logger.warning(f"Sector {sector_id} returned HTTP {resp.status_code}")
                return []
        except Exception as exc:
            logger.warning(f"Attempt {attempt + 1} failed for sector {sector_id}, state {state_id}: {exc}")
            time.sleep(1)
    return []


def download_all_live_records() -> List[Dict[str, Any]]:
    """Harvest real project records across all 26 infrastructure sectors."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session, token = create_session()

    sectors = FALLBACK_SECTORS
    states = FALLBACK_STATES
    logger.info(f"Targeting {len(sectors)} infrastructure sectors on PAIMANA portal.")

    all_records: List[Dict[str, Any]] = []

    for sec in sectors:
        sec_id = str(sec["Value"])
        sec_name = sec["Text"]
        logger.info(f"Querying Sector: {sec_name} (ID: {sec_id})...")

        items = fetch_tile_data(session, token, sec_id)

        if items is None:
            logger.info(f"  Sector {sec_name} payload exceeded maxJsonLength. Partitioning by CostRange...")
            # Partitioning by CostRange: 2 (<=500 Cr) and 1 (>500 Cr)
            items_le500 = fetch_tile_data(session, token, sec_id, cost_range="2") or []
            logger.info(f"    Sector {sec_name} [<=500 Cr]: {len(items_le500)} records")
            all_records.extend(items_le500)

            items_gt500 = fetch_tile_data(session, token, sec_id, cost_range="1")
            if items_gt500 is not None:
                logger.info(f"    Sector {sec_name} [>500 Cr]: {len(items_gt500)} records")
                all_records.extend(items_gt500)
            else:
                logger.info(f"    Sector {sec_name} [>500 Cr] still too large. Partitioning by {len(states)} States...")
                for st in states:
                    st_id = str(st["Value"])
                    st_name = st["Text"]
                    st_items = fetch_tile_data(session, token, sec_id, state_id=st_id, cost_range="1") or []
                    if st_items:
                        logger.info(f"      State {st_name}: {len(st_items)} records")
                        all_records.extend(st_items)
                    time.sleep(0.05)
        else:
            logger.info(f"  Retrieved {len(items)} records.")
            all_records.extend(items)

        time.sleep(0.1)

    logger.info(f"Finished download. Total raw records fetched: {len(all_records)}")
    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    logger.info(f"Saved raw records to {RAW_DATA_PATH}")

    return all_records


def clean_project_records(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate longitudinal snapshots and map to ProjectInput contract."""
    projects_by_id: Dict[int, List[Dict[str, Any]]] = {}
    for r in raw_records:
        pid = r.get("ProjectId")
        if pid:
            projects_by_id.setdefault(pid, []).append(r)

    logger.info(f"Consolidating {len(projects_by_id)} unique infrastructure projects from {len(raw_records)} snapshots...")

    cleaned_projects: List[Dict[str, Any]] = []
    now = datetime(2026, 9, 15)

    for pid, snapshots in projects_by_id.items():
        def get_exp(item):
            try:
                return float(item.get("Expenditure") or 0.0)
            except (ValueError, TypeError):
                return 0.0

        snapshots.sort(key=get_exp)
        latest = snapshots[-1]
        prev = snapshots[-2] if len(snapshots) > 1 else None

        # Sanction & End Dates
        sanction_dt = parse_date(latest.get("SanctionDate"))
        orig_end_dt = parse_date(latest.get("OriginalEndDate"))
        revised_end_dt = parse_date(latest.get("RevisedDate"))

        # Costs
        try:
            orig_cost = float(latest.get("OriginalCost") or 0.0)
        except (ValueError, TypeError):
            orig_cost = 0.0

        if orig_cost <= 0.0:
            continue

        try:
            rev_cost = float(latest.get("RevisedCost") or 0.0)
            if rev_cost <= 0.0:
                rev_cost = orig_cost
        except (ValueError, TypeError):
            rev_cost = orig_cost

        # Expenditure
        exp = get_exp(latest)
        if orig_cost > 0 and exp > 20 * orig_cost:
            exp = exp / 100.0
        exp_recent = max(0.0, exp - get_exp(prev)) if prev else 0.0

        # Physical Progress
        try:
            prog = float(latest.get("PhysicalProgress") or 0.0)
            prog = max(0.0, min(100.0, prog))
        except (ValueError, TypeError):
            prog = 0.0

        try:
            prev_prog = float(prev.get("PhysicalProgress") or 0.0) if prev else prog
            prog_recent = max(-5.0, min(100.0, prog - prev_prog))
        except (ValueError, TypeError):
            prog_recent = 0.0

        # Schedule Durations
        if sanction_dt and orig_end_dt and orig_end_dt > sanction_dt:
            orig_duration = max(1.0, (orig_end_dt - sanction_dt).days / 30.44)
        else:
            orig_duration = 36.0

        if sanction_dt:
            months_elapsed = max(0.0, (now - sanction_dt).days / 30.44)
        else:
            months_elapsed = 12.0

        # Delay
        try:
            delayed_time = float(latest.get("DELAYED_TIME") or 0.0)
        except (ValueError, TypeError):
            delayed_time = 0.0

        if delayed_time > 0:
            current_delay = delayed_time
        elif orig_end_dt and revised_end_dt and revised_end_dt > orig_end_dt:
            current_delay = max(0.0, (revised_end_dt - orig_end_dt).days / 30.44)
        else:
            current_delay = 0.0

        # Sector mapping to standard MoSPI baseline groups
        raw_sector = str(latest.get("SectorName") or "Other").strip()
        if "Railway" in raw_sector:
            sector = "Railways"
        elif "Road" in raw_sector or "Highway" in raw_sector:
            sector = "Roads & Highways"
        elif "Electricity" in raw_sector or "Energy" in raw_sector:
            sector = "Power"
        elif "Oil" in raw_sector or "Gas" in raw_sector or "Petroleum" in raw_sector:
            sector = "Petroleum"
        elif "Coal" in raw_sector:
            sector = "Coal"
        elif "Steel" in raw_sector or "Metal" in raw_sector:
            sector = "Steel"
        elif "Aviation" in raw_sector:
            sector = "Aviation"
        elif "Urban" in raw_sector or "Transport" in raw_sector:
            sector = "Urban Transport"
        elif "Water" in raw_sector:
            sector = "Water Resources"
        elif "Telecommunication" in raw_sector:
            sector = "Telecommunications"
        else:
            sector = raw_sector

        # Remarks & Litigation
        remarks = str(latest.get("Remarks") or latest.get("RevisedDateReason") or "").strip()
        pname = str(latest.get("ProjectName") or "").strip()
        litigation_keywords = ["court", "arbitration", "stay", "litigation", "dispute", "tribunal", "injunction"]
        has_litigation = any(k in remarks.lower() or k in pname.lower() for k in litigation_keywords)

        # Clearances
        clearance_days = 0
        if any(c in remarks.lower() for c in ["forest", "wildlife", "environmental clearance", "stage-2"]):
            clearance_days = 195

        agency = str(latest.get("COMPANYNAME") or latest.get("LineMinistry") or "CPSE").strip()
        if len(agency) > 60:
            agency = agency[:57] + "..."

        p_dict = {
            "project_id": f"MOSPI_{pid}",
            "project_name": pname or f"Central Project {pid}",
            "sector": sector,
            "implementing_agency": agency or "Central Agency",
            "state": latest.get("StateName"),
            "original_cost": round(orig_cost, 2),
            "revised_cost": round(rev_cost, 2) if rev_cost > 0 else None,
            "expenditure": round(exp, 2),
            "expenditure_change_recent": round(exp_recent, 2),
            "physical_progress": round(prog, 2),
            "progress_change_recent": round(prog_recent, 2),
            "original_duration_months": round(orig_duration, 1),
            "months_elapsed": round(months_elapsed, 1),
            "current_delay_months": round(current_delay, 1),
            "overdue_milestones": 1 if current_delay > 6.0 else 0,
            "days_since_last_cuf_update": 15,
            "clearance_pending_days": clearance_days,
            "active_litigation": has_litigation,
            "reasons_for_delay": remarks if remarks else None,
        }
        cleaned_projects.append(p_dict)

    cleaned_projects.sort(key=lambda x: x["original_cost"], reverse=True)

    with open(PROJECTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_projects, f, indent=2)
    logger.info(f"Saved {len(cleaned_projects)} normalized projects to {PROJECTS_JSON_PATH}")

    if cleaned_projects:
        keys = list(cleaned_projects[0].keys())
        with open(PROJECTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(cleaned_projects)
        logger.info(f"Saved CSV dataset to {PROJECTS_CSV_PATH}")

    return cleaned_projects


def main():
    logger.info("=== MoSPI PAIMANA Real Data Pipeline ===")
    raw_records = download_all_live_records()
    clean_projects = clean_project_records(raw_records)
    logger.info(f"=== Complete! Successfully ingested {len(clean_projects)} live government infrastructure projects. ===")


if __name__ == "__main__":
    main()
