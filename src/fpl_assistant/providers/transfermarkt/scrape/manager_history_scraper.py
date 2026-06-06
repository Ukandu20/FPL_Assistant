#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_MANIFEST = Path("data/config/transfermarkt_premier_league_clubs.json")
DEFAULT_OUT_DIR = Path("data/raw/transfermarkt/premier_league/managers")
DEFAULT_TIMEOUT = 30
DEFAULT_DELAY_SECONDS = 1.0
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)
CSV_FIELDS = [
    "name",
    "nationality",
    "appointment_date",
    "end_date",
    "tenure",
    "matches",
]
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
TENURE_DAYS_RE = re.compile(r"(\d+)\s*day", re.IGNORECASE)
INTEGER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class ClubManifestEntry:
    team_code: str
    club_name: str
    transfermarkt_slug: str
    verein_id: int
    url: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ClubManifestEntry":
        required = {"team_code", "club_name", "transfermarkt_slug", "verein_id", "url"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Manifest entry missing required keys: {', '.join(missing)}")
        return cls(
            team_code=str(raw["team_code"]).upper(),
            club_name=str(raw["club_name"]).strip(),
            transfermarkt_slug=str(raw["transfermarkt_slug"]).strip(),
            verein_id=int(raw["verein_id"]),
            url=str(raw["url"]).strip(),
        )


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.transfermarkt.com/",
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def load_manifest(path: Path) -> list[ClubManifestEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in manifest: {path}")
    return [ClubManifestEntry.from_dict(item) for item in raw]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "stripped_strings"):
        return " ".join(value.stripped_strings)
    return str(value).strip()


def _normalize_date_text(text: str) -> str:
    clean = text.strip()
    if not clean:
        return ""
    if not DATE_RE.match(clean):
        raise ValueError(f"Unsupported date format: {clean}")
    return datetime.strptime(clean, "%d/%m/%Y").date().isoformat()


def _extract_nationality(cell: Any) -> str:
    names: list[str] = []
    for image in cell.select("img"):
        label = image.get("title") or image.get("alt") or ""
        label = label.strip()
        if label and label not in names:
            names.append(label)
    fallback = _clean_text(cell)
    if fallback and fallback not in names:
        names.append(fallback)
    return "; ".join(names)


def _extract_name(cell: Any) -> str:
    link = cell.select_one("td.hauptlink a") or cell.select_one("a[title]") or cell.select_one("a")
    if link is None:
        return ""
    return _clean_text(link)


def _parse_tenure_days(text: str) -> int | None:
    clean = text.strip()
    if not clean:
        return None
    match = TENURE_DAYS_RE.search(clean)
    if match:
        return int(match.group(1))
    return None


def _compute_tenure_days(appointment_date: str, end_date: str, as_of_date: date) -> int | None:
    if not appointment_date:
        return None
    start = date.fromisoformat(appointment_date)
    end = date.fromisoformat(end_date) if end_date else as_of_date
    return (end - start).days


def _parse_matches(text: str) -> int:
    clean = text.strip()
    if not clean or clean == "-":
        return 0
    match = INTEGER_RE.search(clean.replace(",", ""))
    if match is None:
        raise ValueError(f"Unable to parse matches value: {clean}")
    return int(match.group(0))


def find_manager_history_table(soup: BeautifulSoup) -> Any:
    for table in soup.select("table.items"):
        table_text = " ".join(table.stripped_strings)
        if (
            "Name/Date of birth" in table_text
            and "Appointed" in table_text
            and "Matches" in table_text
        ):
            return table
    raise ValueError("Could not find Transfermarkt manager history table")


def parse_manager_history_html(html: str, as_of_date: date | None = None) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = find_manager_history_table(soup)
    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError("Manager history table is missing a tbody")

    run_date = as_of_date or date.today()
    rows: list[dict[str, Any]] = []

    for tr in tbody.find_all("tr", recursive=False):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 6:
            continue

        name = _extract_name(cells[0])
        if not name:
            continue

        appointment_date = _normalize_date_text(_clean_text(cells[2]))
        end_date = _normalize_date_text(_clean_text(cells[3])) if len(cells) > 3 else ""
        tenure = _parse_tenure_days(_clean_text(cells[4]))
        if tenure is None:
            tenure = _compute_tenure_days(appointment_date, end_date, run_date)
        if tenure is None:
            raise ValueError(f"Could not determine tenure for {name}")

        rows.append(
            {
                "name": name,
                "nationality": _extract_nationality(cells[1]),
                "appointment_date": appointment_date,
                "end_date": end_date,
                "tenure": tenure,
                "matches": _parse_matches(_clean_text(cells[5])),
            }
        )

    if not rows:
        raise ValueError("No manager rows were parsed from the Transfermarkt page")
    return rows


def build_output_path(team_code: str, out_dir: Path) -> Path:
    return out_dir / f"{team_code.upper()}.csv"


def fetch_manager_history_html(session: requests.Session, url: str, timeout: int) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content.decode("utf-8", errors="replace")


def write_manager_history_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _parse_cli_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --as-of-date value '{value}'. Use YYYY-MM-DD."
        ) from exc


def _select_entries(
    manifest: list[ClubManifestEntry],
    team_code: str | None,
    scrape_all: bool,
) -> list[ClubManifestEntry]:
    if scrape_all:
        return manifest
    wanted = (team_code or "").upper()
    for entry in manifest:
        if entry.team_code == wanted:
            return [entry]
    available = ", ".join(sorted(item.team_code for item in manifest))
    raise SystemExit(f"Unknown --team-code '{wanted}'. Available codes: {available}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Transfermarkt manager-history CSVs for current Premier League clubs."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--team-code", help="Single club team code to scrape, e.g. MUN")
    selector.add_argument(
        "--all-current-pl",
        action="store_true",
        help="Scrape all clubs listed in the manifest",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--as-of-date",
        type=_parse_cli_date,
        default=None,
        help="Override the run date used for current-manager tenure fallback (YYYY-MM-DD).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest = load_manifest(args.manifest)
    entries = _select_entries(manifest, args.team_code, args.all_current_pl)
    session = build_session()
    run_date = args.as_of_date or date.today()

    for index, entry in enumerate(entries):
        if index:
            time.sleep(DEFAULT_DELAY_SECONDS)
        html = fetch_manager_history_html(session, entry.url, timeout=args.timeout)
        rows = parse_manager_history_html(html, as_of_date=run_date)
        out_path = build_output_path(entry.team_code, args.out_dir)
        write_manager_history_csv(rows, out_path)
        logging.info("Wrote %s rows for %s -> %s", len(rows), entry.team_code, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
