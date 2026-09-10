from __future__ import annotations

import sys

from typing import Optional, Sequence

from fpl_assistant.providers.whoscored.scrape.whoscored_match_stats_scraper import main as whoscored_main


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]
    if not any(arg == "--backend" or arg.startswith("--backend=") for arg in args):
        args = ["--backend", "soccerdata", *args]
    whoscored_main(args)


if __name__ == "__main__":
    main()
