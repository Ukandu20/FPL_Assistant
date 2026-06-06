# scripts/fbref_pipeline/automation/auto_scrape.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

import json
import logging


@dataclass(frozen=True)
class ScrapeJobId:
    """
    Identifies a single scrape job for meta tracking.

    Example:
        ScrapeJobId(
            scraper="match",
            league="ENG-Premier League",
            season="2025-2026",
            levels="both",
        )
    """
    scraper: str
    league: str
    season: str
    levels: str

    def key(self) -> str:
        """Stable key for use in scraper_runs.json."""
        return f"{self.scraper}|{self.league}|{self.season}|{self.levels}"


def _load_meta(path: Path) -> Dict[str, Any]:
    """
    Load existing meta JSON from disk. If file does not exist or is invalid,
    return an empty dict.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger("fbref").warning(
            "Failed to load scraper meta from %s: %s (resetting).",
            path,
            e,
        )
        return {}


def _save_meta(path: Path, data: Dict[str, Any]) -> None:
    """
    Save meta JSON to disk with a simple atomic write (write to tmp, then move).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def record_last_run(meta_path: Path, job: ScrapeJobId, run_info: Dict[str, Any]) -> None:
    """
    Update the meta JSON with information about the last run for a given job.

    The file is a dict keyed by job.key(), each containing:
        {
          "job": { ... job fields ... },
          "last_run": {
              "scrape_ts": ISO timestamp (UTC),
              "mode": "manual" | "automation",
              "cutoff_date": "YYYY-MM-DD",
              "last_match_date": "YYYY-MM-DD" | null,
              "latest_fixture": {
                  "date": "YYYY-MM-DD",
                  "home_team": "...",
                  "away_team": "...",
                  "score": "...",
                  "round": "GW14" | "" | ...
              },
              "stats_summary": {
                  "player_match": {
                      "skipped_up_to_date": [...],
                      "scraped_ok": [...],
                      "incomplete_existing": [...],
                      "schema_only": [...]
                  },
                  "team_match": {
                      "skipped_up_to_date": [...],
                      "scraped_ok": [...],
                      "incomplete_existing": [...],
                      "schema_only": [...]
                  }
              }
          }
        }

    This structure gives you, for each (scraper, league, season, levels):
    - When the last run happened (scrape_ts)
    - Whether it was automation or manual
    - Up to which fixture date the stats are expected to be complete (cutoff_date)
    - The latest fixture (as known at that time)
    - Which stats were skipped, fully scraped, incomplete, or schema-only
    """
    log = logging.getLogger("fbref")

    meta = _load_meta(meta_path)
    key = job.key()

    job_block = meta.get(key, {})
    job_block["job"] = asdict(job)
    job_block["last_run"] = run_info

    meta[key] = job_block

    _save_meta(meta_path, meta)
    log.info(
        "Updated scraper meta for %s at %s",
        key,
        meta_path,
    )
