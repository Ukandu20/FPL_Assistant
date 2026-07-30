from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .dnp import build_complete_player_fixture_panel
from .contracts import PLAYER_FIXTURE_CONTRACT, PLAYER_MATCH_FACT_CONTRACT
from .features import build_feature_snapshot
from .facts import (
    DEFAULT_PLAYER_SOURCE_POLICY,
    DEFAULT_TEAM_SOURCE_POLICY,
    build_canonical_facts,
)
from .identity import build_identity_registry
from .matches import build_match_registry
from .manifest import RunManifest, determine_run_mode
from .staging import stage_match_facts
from .whoscored_defense import aggregate_whoscored_defensive_events
from .whoscored_defense import canonicalize_whoscored_event_ids


def _read(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _load_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Aliases file must contain a JSON object.")
    return {str(key): str(target) for key, target in value.items()}


def _load_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _identity(args: argparse.Namespace) -> None:
    result = build_identity_registry(
        _read(args.records),
        entity_type=args.entity_type,
        existing_dimensions=_read(args.existing_dimensions),
        existing_bridges=_read(args.existing_bridges),
        aliases=_load_aliases(args.aliases),
        generate_missing=not args.no_generate_missing,
    )
    _write(result.dimensions, args.out_dir / f"dim_{args.entity_type}.csv")
    _write(
        result.bridges,
        args.out_dir / f"bridge_{args.entity_type}_provider_id.csv",
    )
    _write(result.review, args.out_dir / f"{args.entity_type}_identity_review.csv")


def _matches(args: argparse.Namespace) -> None:
    result = build_match_registry(
        _read(args.schedules),
        team_bridges=_read(args.team_bridges),
        existing_matches=_read(args.existing_matches),
        existing_bridges=_read(args.existing_bridges),
    )
    _write(result.matches, args.out_dir / "dim_match.csv")
    _write(result.bridges, args.out_dir / "bridge_match_provider_id.csv")
    _write(result.unresolved, args.out_dir / "match_identity_review.csv")


def _facts(args: argparse.Namespace) -> None:
    entity_column = "player_id" if args.entity_type == "player" else "team_id"
    policy = (
        DEFAULT_PLAYER_SOURCE_POLICY
        if args.entity_type == "player"
        else DEFAULT_TEAM_SOURCE_POLICY
    )
    result = build_canonical_facts(
        _read(args.records),
        entity_column=entity_column,
        source_policy=policy,
    )
    stem = f"fact_{args.entity_type}_match"
    _write(result.facts, args.out_dir / f"{stem}.csv")
    _write(result.provenance, args.out_dir / f"{stem}_provenance.csv")
    _write(result.conflicts, args.out_dir / f"{stem}_conflicts.csv")


def _stage_facts(args: argparse.Namespace) -> None:
    frame = stage_match_facts(
        _read(args.records),
        provider=args.provider,
        entity_type=args.entity_type,
        entity_bridges=_read(args.entity_bridges),
        match_bridges=_read(args.match_bridges),
        provider_entity_id_column=args.provider_entity_id_column,
        provider_match_id_column=args.provider_match_id_column,
        metric_columns=_load_json_object(args.metric_map),
        retrieved_at=args.retrieved_at,
        strict=not args.allow_unresolved,
    )
    _write(frame, args.output)


def _dnp(args: argparse.Namespace) -> None:
    panel = build_complete_player_fixture_panel(
        _read(args.roster),
        _read(args.fixtures),
        observations=_read(args.observations),
        availability=_read(args.availability),
        as_of_timestamp=args.as_of,
    )
    _write(panel, args.output)


def _whoscored_defense(args: argparse.Namespace) -> None:
    events = _read(args.events)
    bridges = [args.player_bridges, args.team_bridges, args.match_bridges]
    if any(bridges) and not all(bridges):
        raise ValueError(
            "WhoScored ID resolution requires player, team, and match bridges."
        )
    if all(bridges):
        events = canonicalize_whoscored_event_ids(
            events,
            player_bridges=_read(args.player_bridges),
            team_bridges=_read(args.team_bridges),
            match_bridges=_read(args.match_bridges),
            strict=not args.allow_unresolved,
        )
    result = aggregate_whoscored_defensive_events(events)
    _write(result.player_match, args.out_dir / "player_match_defensive_events.csv")
    _write(result.event_audit, args.out_dir / "event_coverage_audit.csv")


def _feature_snapshot(args: argparse.Namespace) -> None:
    contracts = {
        "player-fixture": PLAYER_FIXTURE_CONTRACT,
        "player-match": PLAYER_MATCH_FACT_CONTRACT,
    }
    snapshot = build_feature_snapshot(
        _read(args.records),
        contract=contracts[args.contract],
        feature_version=args.feature_version,
        as_of_timestamp=args.as_of,
    )
    _write(snapshot.frame, args.output)


def _manifest(args: argparse.Namespace) -> None:
    provider_status = _load_json_object(args.provider_status)
    mode = determine_run_mode(provider_status) if args.mode == "auto" else args.mode
    manifest = RunManifest.create(
        run_id=args.run_id,
        mode=mode,
        configuration=_load_json_object(args.configuration)
        if args.configuration
        else {},
    )
    manifest.provider_status.update(provider_status)
    if args.contracts:
        manifest.contracts.update(_load_json_object(args.contracts))
    if args.feature_versions:
        manifest.feature_versions.update(
            _load_json_object(args.feature_versions)
        )
    manifest.warnings.extend(args.warnings)
    for path in args.inputs:
        manifest.add_artifact(path, output=False)
    for path in args.outputs:
        manifest.add_artifact(path, output=True)
    manifest.finish()
    manifest.write(args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build provider-independent canonical football data."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    identity = commands.add_parser("identities")
    identity.add_argument("--records", type=Path, required=True)
    identity.add_argument("--entity-type", choices=["player", "team"], required=True)
    identity.add_argument("--existing-dimensions", type=Path)
    identity.add_argument("--existing-bridges", type=Path)
    identity.add_argument("--aliases", type=Path)
    identity.add_argument("--no-generate-missing", action="store_true")
    identity.add_argument("--out-dir", type=Path, required=True)
    identity.set_defaults(handler=_identity)

    matches = commands.add_parser("matches")
    matches.add_argument("--schedules", type=Path, required=True)
    matches.add_argument("--team-bridges", type=Path, required=True)
    matches.add_argument("--existing-matches", type=Path)
    matches.add_argument("--existing-bridges", type=Path)
    matches.add_argument("--out-dir", type=Path, required=True)
    matches.set_defaults(handler=_matches)

    facts = commands.add_parser("facts")
    facts.add_argument("--records", type=Path, required=True)
    facts.add_argument("--entity-type", choices=["player", "team"], required=True)
    facts.add_argument("--out-dir", type=Path, required=True)
    facts.set_defaults(handler=_facts)

    stage = commands.add_parser("stage-facts")
    stage.add_argument("--records", type=Path, required=True)
    stage.add_argument("--provider", required=True)
    stage.add_argument("--entity-type", choices=["player", "team"], required=True)
    stage.add_argument("--entity-bridges", type=Path, required=True)
    stage.add_argument("--match-bridges", type=Path, required=True)
    stage.add_argument("--provider-entity-id-column", required=True)
    stage.add_argument("--provider-match-id-column", required=True)
    stage.add_argument("--metric-map", type=Path, required=True)
    stage.add_argument("--retrieved-at")
    stage.add_argument("--allow-unresolved", action="store_true")
    stage.add_argument("--output", type=Path, required=True)
    stage.set_defaults(handler=_stage_facts)

    dnp = commands.add_parser("dnp-panel")
    dnp.add_argument("--roster", type=Path, required=True)
    dnp.add_argument("--fixtures", type=Path, required=True)
    dnp.add_argument("--observations", type=Path)
    dnp.add_argument("--availability", type=Path)
    dnp.add_argument("--as-of", required=True)
    dnp.add_argument("--output", type=Path, required=True)
    dnp.set_defaults(handler=_dnp)

    defense = commands.add_parser("whoscored-defense")
    defense.add_argument("--events", type=Path, required=True)
    defense.add_argument("--player-bridges", type=Path)
    defense.add_argument("--team-bridges", type=Path)
    defense.add_argument("--match-bridges", type=Path)
    defense.add_argument("--allow-unresolved", action="store_true")
    defense.add_argument("--out-dir", type=Path, required=True)
    defense.set_defaults(handler=_whoscored_defense)

    features = commands.add_parser("feature-snapshot")
    features.add_argument("--records", type=Path, required=True)
    features.add_argument(
        "--contract",
        choices=["player-fixture", "player-match"],
        required=True,
    )
    features.add_argument("--feature-version", required=True)
    features.add_argument("--as-of", required=True)
    features.add_argument("--output", type=Path, required=True)
    features.set_defaults(handler=_feature_snapshot)

    manifest = commands.add_parser("run-manifest")
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument(
        "--mode",
        choices=["auto", "full", "standard", "minimum"],
        default="auto",
    )
    manifest.add_argument("--provider-status", type=Path, required=True)
    manifest.add_argument("--configuration", type=Path)
    manifest.add_argument("--contracts", type=Path)
    manifest.add_argument("--feature-versions", type=Path)
    manifest.add_argument("--warnings", nargs="*", default=[])
    manifest.add_argument("--inputs", nargs="*", type=Path, default=[])
    manifest.add_argument("--outputs", nargs="*", type=Path, default=[])
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(handler=_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
