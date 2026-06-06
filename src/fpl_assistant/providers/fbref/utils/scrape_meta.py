# scripts/fbref_pipeline/utils/scrape_meta.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional


ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"


@dataclass
class ScrapeJobId:
    scraper: str      # e.g. "match", "season"
    league: str       # e.g. "ENG-Premier League"
    season: str       # e.g. "2025-2026"
    levels: str       # e.g. "player", "team", "both"

    def key(self) -> str:
        # Normalise spaces to avoid surprises in JSON keys
        league_clean = self.league.replace(":", "_")
        return f"{self.scraper}:{league_clean}:{self.season}:{self.levels}"


def _load_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupt file? Don't kill the run; start fresh.
        return {}


def _save_meta(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def record_last_run(meta_path: Path, job: ScrapeJobId, when: Optional[datetime] = None) -> None:
    """
    Record that a given scrape job (scraper+league+season+levels) ran successfully at 'when'.
    If 'when' is None, use now() in UTC.
    """
    if when is None:
        when = datetime.now(timezone.utc)
    data = _load_meta(meta_path)
    data[job.key()] = when.strftime(ISO_FMT)
    _save_meta(meta_path, data)


def get_last_run(meta_path: Path, job: ScrapeJobId) -> Optional[datetime]:
    data = _load_meta(meta_path)
    raw = data.get(job.key())
    if not raw:
        return None
    try:
        # Backwards-compatible: if no timezone, assume UTC
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if "+" in raw or "-" in raw[10:]:
            return datetime.strptime(raw, ISO_FMT)
        return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def should_run(
    meta_path: Path,
    job: ScrapeJobId,
    min_interval: timedelta,
) -> bool:
    """
    Decide whether a job should run again now, given a minimum spacing between runs.
    """
    last = get_last_run(meta_path, job)
    if last is None:
        return True
    now = datetime.now(timezone.utc)
    return (now - last) >= min_interval
