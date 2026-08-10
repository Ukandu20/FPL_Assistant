from __future__ import annotations

from pathlib import Path


DEFAULT_FPL_LEAGUE = "ENG-Premier League"


def league_scoped_root(root: Path, league: str = DEFAULT_FPL_LEAGUE) -> Path:
    """Return ``root/<league>`` without duplicating an existing league suffix."""
    root = Path(root)
    return root if root.name == league else root / league
