#!/usr/bin/env python3
"""
scrape_fpl_data.py
==================
Fetch Fantasy Premier League data for one or more seasons, including historical seasons via the Wayback Machine:
  - bootstrap-static (metadata: players, teams, events list, element_types)
  - fixtures (match list)
  - events (gameweek metadata list)
  - event-live (per-gameweek live player stats)
  - element-summary (per-player history for active players)

Note: The official FPL API only returns the current season. Historical seasons are fetched
from archived snapshots in the Internet Archive (Wayback Machine).

Usage:
    python scrape_fpl_data.py \
        --season 2015-16 2016-17 ... 2024-25 \
        [--output-root data/raw/fpl] \
        [--skip-existing] \
        [--delay 0.5]
"""
import argparse
import logging
import time
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter, Retry
import datetime

# ── CONFIGURATION ─────────────────────────────────────────────────────────
API_ROOT = "https://fantasy.premierleague.com/api"
ENDPOINTS = {
    "bootstrap_static": "/bootstrap-static/",
    "fixtures":        "/fixtures/",
    "events":          "/events/",
    # live stats for a given gameweek
    "event_live":      "/event/{event_id}/live/",
    "element_summary": "/element-summary/{element_id}/",
}
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        " AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://fantasy.premierleague.com/",
    "Accept": "application/json, text/plain, */*",
}
# Map each season to its approximate archive date (YYYYMMDD) at season end
SEASON_END = {
    "2015-16": "20160601",
    "2016-17": "20170601",
    "2017-18": "20180601",
    "2018-19": "20190601",
    "2019-20": "20200601",
    "2020-21": "20210601",
    "2021-22": "20220601",
    "2022-23": "20230601",
    "2023-24": "20240601",
}
# Maximum days to look back if the snapshot isn't exactly on the target date
MAX_LOOKBACK_DAYS = 14
# ────────────────────────────────────────────────────────────────────────────

def detect_current_season():
    today = datetime.date.today()
    year = today.year
    if today.month < 8:
        start = year - 1
    else:
        start = year
    end = (start + 1) % 100
    return f"{start}-{end:02d}"


def create_session(headers: dict) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(headers)
    return session


def fetch_json(session: requests.Session, path: str) -> dict:
    url = API_ROOT + path
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_historical(session: requests.Session, path: str, season: str) -> dict:
    original_url = API_ROOT + path
    date_str = SEASON_END.get(season)
    if not date_str:
        raise ValueError(f"No archive date defined for season {season}")

    # Try decreasing the date up to MAX_LOOKBACK_DAYS if no exact snapshot
    base_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
    for delta in range(0, MAX_LOOKBACK_DAYS + 1):
        try_date = base_date - datetime.timedelta(days=delta)
        timestamp = try_date.strftime("%Y%m%d")
        logging.debug("Looking for archive snapshot for %s on %s", original_url, timestamp)

        avail = requests.get(
            "https://archive.org/wayback/available",
            params={"url": original_url, "timestamp": timestamp},
            timeout=10
        ).json()
        snap = avail.get("archived_snapshots", {}).get("closest")
        if snap:
            archive_url = snap["url"]
            logging.info("Found snapshot for %s on %s (requested %s)", original_url, snap.get("timestamp"), timestamp)
            resp = session.get(archive_url, timeout=20)
            resp.raise_for_status()
            return resp.json()

    raise RuntimeError(f"No archive snapshot for {original_url} within {MAX_LOOKBACK_DAYS} days of {date_str}")


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Scrape FPL data for one or more seasons"
    )
    ap.add_argument(
        "--season", nargs="+", required=True,
        help="Season strings (format YYYY-YY), e.g. 2023-24"
    )
    ap.add_argument(
        "--output-root", type=Path,
        default=Path("data/raw/fantasy"),
        help="Root directory for output JSON files"
    )
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="Skip download if target file already exists"
    )
    ap.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to sleep between requests"
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s"
    )
    session = create_session(DEFAULT_HEADERS)
    current_season = detect_current_season()
    logging.info("Detected current season as %s", current_season)

    for season in args.season:
        season_dir = args.output_root / season
        season_dir.mkdir(parents=True, exist_ok=True)
        is_current = (season == current_season)

        # 1) bootstrap-static
        bs_path = season_dir / "bootstrap-static.json"
        if args.skip_existing and bs_path.exists():
            logging.info("Skipping bootstrap-static for %s", season)
            bs = json.loads(bs_path.read_text())
        else:
            logging.info("Fetching bootstrap-static for %s", season)
            bs = fetch_json(session, ENDPOINTS["bootstrap_static"]) if is_current else \
                 fetch_historical(session, ENDPOINTS["bootstrap_static"], season)
            save_json(bs, bs_path)
            time.sleep(args.delay)

        # 2) element_types
        et_path = season_dir / "element_types.json"
        if not (args.skip_existing and et_path.exists()):
            logging.info("Saving element_types for %s", season)
            save_json(bs.get("element_types", []), et_path)

        # 3) fixtures
        fx_path = season_dir / "fixtures.json"
        if not (args.skip_existing and fx_path.exists()):
            logging.info("Fetching fixtures for %s", season)
            fixtures = fetch_json(session, ENDPOINTS["fixtures"]) if is_current else \
                       fetch_historical(session, ENDPOINTS["fixtures"], season)
            save_json(fixtures, fx_path)
            time.sleep(args.delay)

        # 4) events
        ev_path = season_dir / "events.json"
        if args.skip_existing and ev_path.exists():
            events = json.loads(ev_path.read_text())
        else:
            logging.info("Fetching events list for %s", season)
            events = fetch_json(session, ENDPOINTS["events"]) if is_current else \
                     fetch_historical(session, ENDPOINTS["events"], season)
            save_json(events, ev_path)
            time.sleep(args.delay)

        # 5) gameweek_live
        live_dir = season_dir / "gameweek_live"
        live_dir.mkdir(parents=True, exist_ok=True)
        for ev in events:
            ev_id = ev.get("id")
            out_file = live_dir / f"event_{ev_id}_live.json"
            if args.skip_existing and out_file.exists():
                continue
            logging.info("Fetching live event data for GW %s/%s", ev_id, season)
            data = fetch_json(session, ENDPOINTS["event_live"].format(event_id=ev_id)) if is_current else \
                   fetch_historical(session, ENDPOINTS["event_live"].format(event_id=ev_id), season)
            save_json(data, out_file)
            time.sleep(args.delay)

        # 6) player_summaries
        players_dir = season_dir / "player_summaries"
        players_dir.mkdir(parents=True, exist_ok=True)
        for elem in bs.get("elements", []):
            eid = elem.get("id")
            ps_path = players_dir / f"{eid}.json"
            if args.skip_existing and ps_path.exists():
                continue
            logging.info("Fetching element-summary for player %s/%s", eid, season)
            summary = fetch_json(session, ENDPOINTS["element_summary"].format(element_id=eid)) if is_current else \
                      fetch_historical(session, ENDPOINTS["element_summary"].format(element_id=eid), season)
            save_json(summary, ps_path)
            time.sleep
