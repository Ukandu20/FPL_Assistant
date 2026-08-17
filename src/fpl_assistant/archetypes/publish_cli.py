from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fpl_assistant.canonical.contracts import validate_contract

from .config import load_config
from .input_builder import build_canonical_archetype_inputs, discover_joinable_seasons
from .persistence import persist_snapshot
from .pipeline import build_archetype_snapshot
from .schema import ARCHETYPE_OUTPUT_CONTRACT, ARCHETYPE_PLAYER_MATCH_CONTRACT


def _latest_previous_snapshot(
    output_root: Path, *, model_version: str, before: pd.Timestamp
) -> Path | None:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    version_root = output_root / f"model_version={model_version}"
    for directory in version_root.glob("snapshot=*") if version_root.is_dir() else ():
        path = directory / "archetypes.jsonl"
        if not path.is_file():
            continue
        try:
            stamp = pd.to_datetime(
                directory.name.removeprefix("snapshot="),
                format="%Y-%m-%dT%H-%M-%SZ",
                utc=True,
            )
        except ValueError:
            continue
        if stamp < before:
            candidates.append((stamp, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical provider joins and publish an app-visible V1 "
            "archetype snapshot in one command."
        )
    )
    parser.add_argument("--league", default="ENG-Premier League")
    parser.add_argument("--current-season", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--seasons",
        nargs="+",
        help="Evidence seasons; defaults to every joinable processed season.",
    )
    parser.add_argument("--output-root", default="data/processed/archetypes")
    parser.add_argument(
        "--previous-snapshot",
        help="Prior archetypes.jsonl. Omit to auto-select the newest earlier snapshot.",
    )
    parser.add_argument(
        "--strict-input-contract",
        action="store_true",
        help="Validate the fully assembled canonical player-match table.",
    )
    args = parser.parse_args()

    config = load_config()
    snapshot = pd.Timestamp(args.as_of)
    snapshot = snapshot.tz_localize("UTC") if snapshot.tzinfo is None else snapshot.tz_convert("UTC")
    seasons = args.seasons or discover_joinable_seasons(args.league)
    if not seasons:
        raise RuntimeError(f"No joinable FPL/Understat/WhoScored seasons found for {args.league}")

    inputs = build_canonical_archetype_inputs(
        league=args.league,
        seasons=seasons,
        current_season=args.current_season,
        as_of=snapshot,
    )
    if inputs.player_matches.empty:
        raise RuntimeError("Canonical player-match build produced no rows")
    if args.strict_input_contract:
        validate_contract(inputs.player_matches, ARCHETYPE_PLAYER_MATCH_CONTRACT)

    output_root = Path(args.output_root)
    if args.previous_snapshot:
        previous_path = Path(args.previous_snapshot)
    else:
        previous_path = _latest_previous_snapshot(
            output_root,
            model_version=config.model_version,
            before=snapshot,
        )
    previous = (
        pd.read_json(previous_path, lines=True)
        if previous_path is not None and previous_path.is_file()
        else None
    )

    result = build_archetype_snapshot(
        inputs.player_matches,
        as_of=snapshot,
        current_season=args.current_season,
        team_matches=inputs.team_matches,
        player_values=inputs.player_values,
        config=config,
        previous_states=previous,
    )
    if result.archetypes.empty:
        raise RuntimeError("Archetype scoring produced no rows")
    validate_contract(result.archetypes, ARCHETYPE_OUTPUT_CONTRACT)
    evidence = {
        **result.evidence_tables,
        "field_provenance": inputs.field_provenance,
        "input_build_audit": inputs.build_audit,
    }
    target = persist_snapshot(
        result.archetypes,
        result.team_ratings,
        output_root=output_root,
        snapshot_date=snapshot,
        model_version=config.model_version,
        evidence_tables=evidence,
    )
    active = result.archetypes["active_label"].fillna(False).astype(bool)
    print(f"Published archetype snapshot: {target}")
    print(
        f"Players: {result.archetypes['player_id'].nunique():,}; "
        f"rows: {len(result.archetypes):,}; active labels: {int(active.sum()):,}"
    )
    print(f"Evidence seasons: {', '.join(seasons)}")
    print(f"Previous snapshot: {previous_path or 'none'}")


if __name__ == "__main__":
    main()
