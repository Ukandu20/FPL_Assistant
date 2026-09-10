"""Schedule and missing-player CLI using the main WhoScored implementation."""
from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from .whoscored_match_stats_scraper import main as whoscored_main


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        "Scrape WhoScored schedule and missing players (injuries/suspensions)"
    )
    parser.add_argument("--league", default="ENG-Premier League")
    parser.add_argument("--out-dir", default="data/raw/whoscored")
    parser.add_argument("--seasons", nargs="*", help="Accepts 'YYYY-YYYY', 'YYYY-YY', 'YY-YY' or a single year like 2025")
    parser.add_argument("--delay", type=float, default=0.75)
    parser.add_argument("--proxy", default=None, help="e.g. 'tor' or http://user:pass@host:port")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument("--browser", dest="path_to_browser", default=None, help="Path to Chrome executable")
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--headed", dest="headless", action="store_false", help="Run Chrome with a visible window")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    forwarded: List[str] = [
        "--league",
        args.league,
        "--out-dir",
        args.out_dir,
        "--tables",
        "schedule",
        "missing_players",
        "--no-derived-tables",
        "--no-archive-raw-events",
        "--delay",
        str(args.delay),
    ]
    if args.seasons:
        forwarded.extend(["--seasons", *args.seasons])
    if args.proxy:
        forwarded.extend(["--proxy", args.proxy])
    if args.no_cache:
        forwarded.append("--no-cache")
    if args.no_store:
        forwarded.append("--no-store")
    if args.path_to_browser:
        forwarded.extend(["--browser", args.path_to_browser])
    forwarded.append("--headless" if args.headless else "--headed")
    if args.verbose:
        forwarded.append("--verbose")
    whoscored_main(forwarded)

if __name__ == "__main__":
    main()
