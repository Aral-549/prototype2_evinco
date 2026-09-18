#!/usr/bin/env python3
"""Harvest MoSPI PAIMANA one freeze month at a time, so records carry real dates.

Implements contracts/monthly_harvest.md.

The original harvester left `MonthYear` empty and got back every monthly record
for a project in one undated blob (`Month: null`, `Year: null` on every row),
which forced the panel builder to reconstruct the time axis under assumptions
A1/A2. Two endpoints make that unnecessary:

    GET  /Home/GetFreezeDates              -> the portal's own freeze range
    POST /Home/GetTileData  MonthYear=...  -> that month's snapshot only

Usage:
    python scripts/harvest_monthly.py
    python scripts/harvest_monthly.py --force          # ignore existing output
    python scripts/harvest_monthly.py --sectors 108 107
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paimana.harvest")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "paimana_monthly_raw.json"
META_PATH = DATA_DIR / "paimana_monthly_meta.json"

BASE = "https://paimana-proj.mospi.gov.in"
DASHBOARD = f"{BASE}/Home/PublicDashboardNew"
FREEZE_URL = f"{BASE}/Home/GetFreezeDates"
TILE_URL = f"{BASE}/Home/GetTileData"

sys.path.insert(0, str(ROOT / "scripts"))
from download_real_paimana_data import FALLBACK_SECTORS, FALLBACK_STATES  # noqa: E402


def make_session() -> Tuple[requests.Session, str]:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                                                      status_forcelist=[502, 503, 504])))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": DASHBOARD,
    })
    return s, refresh_token(s)


def refresh_token(session: requests.Session) -> str:
    """Fetch a fresh ASP.NET anti-forgery token (they expire mid-run)."""
    try:
        r = session.get(DASHBOARD, verify=False, timeout=30)
        r.raise_for_status()
        m = (re.search(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', r.text)
             or re.search(r'value="([^"]+)"[^>]*name="__RequestVerificationToken"', r.text))
        return m.group(1) if m else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not refresh anti-forgery token: %s", exc)
        return ""


def freeze_months(session: requests.Session) -> List[str]:
    """Read the portal's own freeze range. Never hardcode it."""
    r = session.get(FREEZE_URL, verify=False, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("success"):
        raise SystemExit(f"GetFreezeDates returned success=false: {payload}")
    first, last = payload["firstFreeze"], payload["lastFreeze"]
    fy, fm = (int(x) for x in first.split("-"))
    ly, lm = (int(x) for x in last.split("-"))
    months, y, m = [], fy, fm
    while (y, m) <= (ly, lm):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    logger.info("Portal freeze range: %s .. %s (%d months) — display: %s",
                first, last, len(months), payload.get("displayFreeze"))
    return months


def fetch(session: requests.Session, token: str, sector: str, month: str,
          cost: str = "", state: str = "") -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """One GetTileData call. Returns (records, status)."""
    data = {
        "__RequestVerificationToken": token,
        "SectorId": str(sector),
        "PROJ_MINISTRY_ID": "",
        "StateId": str(state),
        "CostRange": str(cost),
        "Month": "",
        "Year": "",
        "MonthYear": month,
    }
    try:
        resp = session.post(TILE_URL, data=data, verify=False, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return None, f"exception:{type(exc).__name__}"
    if resp.status_code == 500:
        # ASP.NET maxJsonLength exceeded -- caller partitions and retries.
        return None, "payload_too_large"
    if resp.status_code != 200:
        return None, f"http_{resp.status_code}"
    try:
        return resp.json().get("data", {}).get("ProjectsCountTabDetails", []) or [], "ok"
    except Exception:  # noqa: BLE001
        return None, "parse_error"


def harvest_slice(session: requests.Session, token: str, sector_id: str, sector_name: str,
                  month: str, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Harvest one (month, sector), partitioning only if the payload is too large."""
    records, status = fetch(session, token, sector_id, month)
    if status == "ok":
        return records

    if status != "payload_too_large":
        stats["failures"].append({"month": month, "sector": sector_name, "reason": status})
        logger.warning("  %s / %s -> %s", month, sector_name, status)
        return []

    # Partition by cost band.
    stats["partitioned"].append({"month": month, "sector": sector_name, "by": "CostRange"})
    out: List[Dict[str, Any]] = []
    for cost in ("2", "1"):
        part, st = fetch(session, token, sector_id, month, cost=cost)
        if st == "ok":
            out.extend(part)
            continue
        if st != "payload_too_large":
            stats["failures"].append(
                {"month": month, "sector": sector_name, "cost_range": cost, "reason": st})
            continue
        # Still too large: partition that cost band by state.
        stats["partitioned"].append(
            {"month": month, "sector": sector_name, "by": f"State(cost={cost})"})
        for st_row in FALLBACK_STATES:
            sub, sub_st = fetch(session, token, sector_id, month,
                                cost=cost, state=str(st_row["Value"]))
            if sub_st == "ok" and sub:
                out.extend(sub)
            time.sleep(0.05)
    return out


def collapse(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse duplicates WITHIN one (month, sector) slice.

    Duplicates across months are the panel itself and must never be collapsed.
    """
    seen, out = set(), []
    for r in records:
        pid = r.get("ProjectId")
        if pid in seen:
            continue
        seen.add(pid)
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest PAIMANA by freeze month.")
    ap.add_argument("--out", type=Path, default=RAW_PATH)
    ap.add_argument("--force", action="store_true", help="ignore any existing output")
    ap.add_argument("--sectors", nargs="*", default=None, help="restrict to these sector ids")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session, token = make_session()
    months = freeze_months(session)

    sectors = FALLBACK_SECTORS
    if args.sectors:
        wanted = {str(x) for x in args.sectors}
        sectors = [s for s in sectors if str(s["Value"]) in wanted]

    existing: List[Dict[str, Any]] = []
    done: set = set()
    if args.out.exists() and not args.force:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        done = {(r.get("freeze_month"), r.get("harvest_sector_id")) for r in existing}
        logger.info("Resuming: %d records already present, %d (month,sector) pairs done",
                    len(existing), len(done))

    stats: Dict[str, Any] = {
        "failures": [], "partitioned": [],
        "counts_by_month": defaultdict(int), "counts_by_month_sector": {},
    }
    all_records = list(existing)
    total_pairs = len(months) * len(sectors)
    pair_i = 0

    for month in months:
        month_total = 0
        for sec in sectors:
            pair_i += 1
            sid, sname = str(sec["Value"]), sec["Text"].strip()
            if (month, sid) in done:
                continue
            recs = collapse(harvest_slice(session, token, sid, sname, month, stats))
            for r in recs:
                r["freeze_month"] = month
                r["harvest_sector_id"] = sid
            all_records.extend(recs)
            month_total += len(recs)
            stats["counts_by_month_sector"][f"{month}|{sname}"] = len(recs)
            time.sleep(args.sleep)

        stats["counts_by_month"][month] = month_total
        logger.info("[%3d/%3d pairs] %s -> %d records (running total %d)",
                    pair_i, total_pairs, month, month_total, len(all_records))

        # Persist after every month so a long run is never lost, and refresh
        # the anti-forgery token which expires on this portal.
        args.out.write_text(json.dumps(all_records), encoding="utf-8")
        token = refresh_token(session) or token

    by_month = dict(stats["counts_by_month"])
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": BASE,
        "freeze_months": months,
        "n_records": len(all_records),
        "n_unique_projects": len({r.get("ProjectId") for r in all_records}),
        "counts_by_month": by_month,
        "counts_by_month_sector": stats["counts_by_month_sector"],
        "partitioning_fallbacks": stats["partitioned"],
        "failures": stats["failures"],
        "note": (
            "Each record carries freeze_month, the reporting month the portal returned it "
            "under. This removes the need for the reconstructed time axis (assumptions A1/A2 "
            "in contracts/panel_builder.md)."
        ),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("")
    logger.info("Harvest complete: %d records, %d unique projects",
                meta["n_records"], meta["n_unique_projects"])
    for m in months:
        logger.info("  %s -> %d", m, by_month.get(m, 0))
    if stats["failures"]:
        logger.warning("  %d slice failures recorded in %s", len(stats["failures"]), META_PATH)
    logger.info("Raw -> %s", args.out)
    logger.info("Meta -> %s", META_PATH)


if __name__ == "__main__":
    main()
