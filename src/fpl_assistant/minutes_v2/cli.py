from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ablations import ABLATIONS, get_ablation
from .artifacts import (
    deterministic_run_id, load_training_artifacts, write_prediction_artifact,
)
from .backtest import run_walk_forward
from .config import load_config
from .data import canonical_training_rows, inference_rows, load_registry
from .features import build_features
from .inference import add_legacy_compatibility, predict
from .legacy import prepare_v1_inputs, prepare_v1_shadow_artifact
from .overrides import apply_confirmed_absence_overrides
from .shadow import assemble_shadow, evaluate_live_shadow
from .workflow import train_production


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _training_data(config_path: str, cutoff: str):
    config = load_config(config_path)
    rows, audit = load_registry(config, config.canonical_label_seasons)
    canonical = canonical_training_rows(rows, config, cutoff, audit)
    featured = build_features(canonical, cutoff)
    # build_features retains only past labels because canonical was cutoff-filtered.
    return config, featured, audit


def command_audit(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rows, audit = load_registry(config, (*config.diagnostic_only_seasons, *config.canonical_label_seasons, config.forward_season))
    canonical_training_rows(rows, config, args.prediction_cutoff, audit)
    payload = {
        "architecture_version": config.architecture_version,
        "prediction_cutoff": pd.to_datetime(args.prediction_cutoff, utc=True).isoformat(),
        "audit": audit.to_dict(),
        "forward_timestamp_safe_rows": int(
            (rows["season"].eq(config.forward_season) & rows["eligibility_timestamp_safe"]).sum()
        ),
        "status": "pass",
    }
    _write_json(Path(args.output), payload)
    return 0


def command_backtest(args: argparse.Namespace) -> int:
    config, featured, audit = _training_data(args.config, args.prediction_cutoff)
    v1 = pd.read_csv(args.v1_predictions, low_memory=False) if args.v1_predictions else None
    ablation = get_ablation(args.ablation)
    predictions, report = run_walk_forward(
        featured, config, v1_predictions=v1,
        feature_overrides=ablation.feature_overrides if ablation else None,
    )
    report["dataset_audit"] = audit.to_dict()
    report["experiment"] = ablation.name if ablation else "canonical_v2.0"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    _write_json(Path(args.report), report)
    return 0


def command_train(args: argparse.Namespace) -> int:
    config, featured, audit = _training_data(args.config, args.prediction_cutoff)
    v1 = pd.read_csv(args.v1_predictions, low_memory=False) if args.v1_predictions else None
    input_ids = audit.source_files
    destination, _ = train_production(
        featured, config, args.prediction_cutoff, audit, input_ids,
        v1_predictions=v1, artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
    )
    print(destination)
    return 0


def command_forecast(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    cutoff = pd.to_datetime(args.prediction_cutoff, utc=True)
    rows, audit = load_registry(config, (*config.canonical_label_seasons, args.season))
    historical = canonical_training_rows(rows, config, cutoff, audit)
    future = inference_rows(rows, args.season, cutoff)
    if args.gws:
        target_gws = {int(value.strip()) for value in args.gws.split(",") if value.strip()}
    else:
        ordered_gws = (
            future[["gw_orig", "date_sched"]].dropna().drop_duplicates()
            .groupby("gw_orig", as_index=False)["date_sched"].min()
            .sort_values(["date_sched", "gw_orig"])["gw_orig"].astype(int).tolist()
        )
        target_gws = set(ordered_gws[: args.next_k])
    future = future[pd.to_numeric(future["gw_orig"], errors="coerce").isin(target_gws)].copy()
    if future.empty:
        raise ValueError(f"No timestamp-safe inference rows for target GWs {sorted(target_gws)}")
    combined = pd.concat([historical, future], ignore_index=True, sort=False)
    featured = build_features(combined, cutoff)
    future_keys = future[["match_id", "player_id"]].drop_duplicates()
    forward_featured = featured[featured["season"].eq(args.season)].copy()
    forecast_rows = forward_featured.merge(
        future_keys, on=["match_id", "player_id"], how="inner", validate="one_to_one"
    )
    models, calibrators, card = load_training_artifacts(Path(args.artifact_dir))
    run_id = deterministic_run_id(config, cutoff.isoformat(), audit.source_files)
    predictions = predict(forecast_rows, models, calibrators, cutoff, run_id)
    overrides = pd.read_csv(args.overrides, low_memory=False) if args.overrides else None
    predictions = apply_confirmed_absence_overrides(predictions, overrides, cutoff)
    if args.legacy_compatibility:
        predictions = add_legacy_compatibility(predictions)
    metadata = {
        "architecture_version": config.architecture_version,
        "model_artifact_run_id": card.get("run_id"),
        "prediction_run_id": run_id,
        "prediction_cutoff": cutoff.isoformat(),
        "season": args.season,
        "input_data_identifiers": audit.source_files,
        "row_count": len(predictions),
        "override_count": int(predictions["override_applied"].sum()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_prediction_artifact(predictions, Path(args.output), metadata)
    return 0


def command_shadow(args: argparse.Namespace) -> int:
    assemble_shadow(Path(args.v1), Path(args.v2), Path(args.output), args.prediction_cutoff)
    return 0


def command_prepare_v1_inputs(args: argparse.Namespace) -> int:
    gws = {int(value.strip()) for value in args.gws.split(",") if value.strip()}
    prepare_v1_inputs(
        Path(args.player_calendar), Path(args.fixture_calendar), args.prediction_cutoff,
        gws, Path(args.fixtures_output), Path(args.squads_output),
        Path(args.registry_root) if args.registry_root else None,
        Path(args.history_root) if args.history_root else None,
        [value.strip() for value in args.history_seasons.split(",") if value.strip()],
    )
    return 0


def command_prepare_v1_shadow(args: argparse.Namespace) -> int:
    prepare_v1_shadow_artifact(
        Path(args.raw_v1), Path(args.v2), Path(args.output), args.prediction_cutoff
    )
    return 0


def command_evaluate_shadow(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    shadow = pd.read_csv(args.shadow, low_memory=False)
    outcomes = pd.read_csv(args.outcomes, low_memory=False) if args.outcomes else None
    report = evaluate_live_shadow(
        shadow, outcomes, config.bootstrap_iterations, config.bootstrap_confidence,
        config.random_seed, config.acceptance_margins,
    )
    _write_json(Path(args.output), report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hardened FPL Expected-Minutes V2.0")
    parser.add_argument("--config", default="config/minutes_v2.json")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--prediction-cutoff", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=command_audit)
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--prediction-cutoff", required=True)
    backtest.add_argument("--v1-predictions", default="")
    backtest.add_argument("--ablation", choices=sorted(ABLATIONS), default="")
    backtest.add_argument("--output", required=True)
    backtest.add_argument("--report", required=True)
    backtest.set_defaults(func=command_backtest)
    train = sub.add_parser("train")
    train.add_argument("--prediction-cutoff", required=True)
    train.add_argument("--v1-predictions", default="")
    train.add_argument("--artifact-dir", default="")
    train.set_defaults(func=command_train)
    forecast = sub.add_parser("forecast")
    forecast.add_argument("--artifact-dir", required=True)
    forecast.add_argument("--prediction-cutoff", required=True)
    forecast.add_argument("--season", required=True)
    forecast.add_argument("--gws", default="", help="Comma-separated target GWs; defaults to the next GW")
    forecast.add_argument("--next-k", type=int, default=1, help="Number of chronological GWs when --gws is omitted")
    forecast.add_argument("--overrides", default="")
    forecast.add_argument("--legacy-compatibility", action="store_true")
    forecast.add_argument("--output", required=True)
    forecast.set_defaults(func=command_forecast)
    shadow = sub.add_parser("shadow")
    shadow.add_argument("--v1", required=True)
    shadow.add_argument("--v2", required=True)
    shadow.add_argument("--prediction-cutoff", required=True)
    shadow.add_argument("--output", required=True)
    shadow.set_defaults(func=command_shadow)
    legacy_inputs = sub.add_parser("prepare-v1-inputs")
    legacy_inputs.add_argument("--player-calendar", required=True)
    legacy_inputs.add_argument("--fixture-calendar", required=True)
    legacy_inputs.add_argument("--prediction-cutoff", required=True)
    legacy_inputs.add_argument("--gws", required=True)
    legacy_inputs.add_argument("--fixtures-output", required=True)
    legacy_inputs.add_argument("--squads-output", required=True)
    legacy_inputs.add_argument("--registry-root", default="")
    legacy_inputs.add_argument("--history-root", default="")
    legacy_inputs.add_argument("--history-seasons", default="")
    legacy_inputs.set_defaults(func=command_prepare_v1_inputs)
    legacy_shadow = sub.add_parser("prepare-v1-shadow")
    legacy_shadow.add_argument("--raw-v1", required=True)
    legacy_shadow.add_argument("--v2", required=True)
    legacy_shadow.add_argument("--prediction-cutoff", required=True)
    legacy_shadow.add_argument("--output", required=True)
    legacy_shadow.set_defaults(func=command_prepare_v1_shadow)
    live = sub.add_parser("evaluate-shadow")
    live.add_argument("--shadow", required=True)
    live.add_argument("--outcomes", default="")
    live.add_argument("--output", required=True)
    live.set_defaults(func=command_evaluate_shadow)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
