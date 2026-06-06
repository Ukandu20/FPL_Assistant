from __future__ import annotations

from typing import Optional, Sequence

from scripts.fbref_pipeline.scrape.whoscored_match_stats_scraper import main as whoscored_main


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(argv) if argv is not None else []
    if "--backend" not in args:
        args = ["--backend", "soccerdata", *args]
    whoscored_main(args)


if __name__ == "__main__":
    main()
