# scripts/fbref_pipeline/automation/run_fbref_automated_scrapes.py

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Literal

from scripts.fbref_pipeline.automation.auto_scrape import ScrapeJobId
from scripts.fbref_pipeline.utils.scrape_meta import should_run  # if you moved should_run elsewhere, adjust import

META_PATH = Path("data/meta/scraper_runs.json")

ScraperType = Literal["match", "season"]


@dataclass
class AutoJob:
    scraper: ScraperType
    league: str
    season: str            # e.g. "2025-2026"
    levels: str            # "player" | "team" | "both"
    min_interval_days: int
    team_mode: str = "auto"
    refresh: bool = True
    verbose: bool = True


# ---- define the jobs you want automated ----
JOBS: List[AutoJob] = [
    AutoJob(
        scraper="match",
        league="ENG-Premier League",
        season="2025-2026",
        levels="player",
        min_interval_days=3,
        team_mode="auto",
        refresh=True,
    ),
    AutoJob(
        scraper="season",
        league="ENG-Premier League",
        season="2025-2026",
        levels="player",
        min_interval_days=7,
        team_mode="aggregate",
        refresh=True,
    ),
]


def build_command(job: AutoJob) -> List[str]:
    base = ["py", "-m"]

    if job.scraper == "match":
        module = "scripts.fbref_pipeline.scrape.match_stats_scraper"
        cmd = base + [module]
        cmd += [
            "--league", job.league,
            "--seasons", job.season,
            "--levels", job.levels,
            "--team-mode", job.team_mode,
            "--meta-path", str(META_PATH),
        ]
        if job.refresh:
            cmd.append("--refresh")
        if job.verbose:
            cmd.append("--verbose")
        return cmd

    if job.scraper == "season":
        module = "scripts.fbref_pipeline.scrape.season_stats_scraper"
        cmd = base + [module]
        cmd += [
            "--league", job.league,
            "--seasons", job.season,
            "--levels", job.levels,
            "--team-mode", job.team_mode,
            "--layout", "flat",
            "--meta-path", str(META_PATH),
        ]
        if job.refresh:
            cmd.append("--refresh")
        if job.verbose:
            cmd.append("--verbose")
        return cmd

    raise ValueError(f"Unknown scraper type: {job.scraper}")


def run_job_if_due(job: AutoJob) -> None:
    job_id = ScrapeJobId(
        scraper=job.scraper,
        league=job.league,
        season=job.season,
        levels=job.levels,
    )
    interval = timedelta(days=job.min_interval_days)

    if not should_run(META_PATH, job_id, interval):
        print(f"[SKIP] {job_id.key()} — ran recently (interval={interval.days}d)")
        return

    cmd = build_command(job)
    print(f"[RUN ] {job_id.key()} -> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[FAIL] {job_id.key()} (exit {result.returncode})")
        print("stdout:\n", result.stdout)
        print("stderr:\n", result.stderr)
        # propagate error so scheduler shows 0x1 when something breaks
        sys.exit(result.returncode)

    print(f"[OK  ] {job_id.key()} (scraper recorded meta itself)")


def main() -> None:
    print("=== FBref automated scrapes ===")
    print(f"Meta path: {META_PATH}")
    for job in JOBS:
        run_job_if_due(job)


if __name__ == "__main__":
    main()
