from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fpl_assistant.canonical.contracts import validate_contract

from .persistence import persist_snapshot
from .pipeline import build_archetype_snapshot
from .schema import ARCHETYPE_OUTPUT_CONTRACT, ARCHETYPE_PLAYER_MATCH_CONTRACT


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if source.suffix.lower() == ".jsonl":
        return pd.read_json(source, lines=True)
    return pd.read_csv(source)


def _optional_table(path: str | None) -> pd.DataFrame | None:
    return _read_table(path) if path else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe V1 FPL player-archetype snapshot."
    )
    parser.add_argument("--player-matches", required=True)
    parser.add_argument("--team-matches")
    parser.add_argument("--player-values")
    parser.add_argument("--previous-snapshot")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--current-season", required=True)
    parser.add_argument("--output-root", default="data/processed/archetypes")
    parser.add_argument(
        "--strict-input-contract", action="store_true",
        help="Require every canonical optional field to be present before scoring.",
    )
    args = parser.parse_args()

    player_matches = _read_table(args.player_matches)
    if args.strict_input_contract:
        validate_contract(player_matches, ARCHETYPE_PLAYER_MATCH_CONTRACT)
    result = build_archetype_snapshot(
        player_matches,
        as_of=args.as_of,
        current_season=args.current_season,
        team_matches=_optional_table(args.team_matches),
        player_values=_optional_table(args.player_values),
        previous_states=_optional_table(args.previous_snapshot),
    )
    if not result.archetypes.empty:
        validate_contract(result.archetypes, ARCHETYPE_OUTPUT_CONTRACT)
        model_version = str(result.archetypes["model_version"].iloc[0])
    else:
        model_version = "1.0.0"
    target = persist_snapshot(
        result.archetypes,
        result.team_ratings,
        output_root=args.output_root,
        snapshot_date=args.as_of,
        model_version=model_version,
        evidence_tables=result.evidence_tables,
    )
    print(f"Wrote archetype snapshot to {target}")


if __name__ == "__main__":
    main()
