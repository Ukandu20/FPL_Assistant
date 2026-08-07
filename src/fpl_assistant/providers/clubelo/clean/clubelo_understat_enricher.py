"""Provider import wrapper for the script-level ClubElo Understat enricher."""

from scripts.clubelo_pipeline.clean.clubelo_understat_enricher import *  # noqa: F403


if __name__ == "__main__":
    raise SystemExit(main())  # noqa: F405
