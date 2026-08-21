from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import deterministic_run_id, save_training_artifacts
from .backtest import run_walk_forward
from .calibration import fit_hierarchical_calibrators
from .config import MinutesV2Config
from .data import DatasetAudit, canonical_training_rows
from .features import (
    CAMEO_FEATURES, CAMEO_MIN_FEATURES, P60_FEATURES, START_FEATURES,
    START_MIN_FEATURES,
)
from .folds import fixture_block_key
from .inference import CalibrationBundle
from .models import model_family, train_bundle


def train_production(
    featured_rows: pd.DataFrame,
    config: MinutesV2Config,
    prediction_cutoff: str | pd.Timestamp,
    audit: DatasetAudit,
    input_identifiers: dict[str, object],
    v1_predictions: pd.DataFrame | None = None,
    artifact_dir: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    canonical = canonical_training_rows(featured_rows, config, prediction_cutoff, audit)
    backtest_predictions, backtest_report = run_walk_forward(canonical, config, v1_predictions=v1_predictions)
    canonical = canonical.copy()
    canonical["_block_key"] = fixture_block_key(canonical)
    blocks = (
        canonical[["_block_key", "date_sched"]].drop_duplicates()
        .groupby("_block_key", as_index=False)["date_sched"].min()
        .sort_values("date_sched")["_block_key"].tolist()
    )
    if len(blocks) <= config.calibration_gws:
        raise ValueError("Insufficient blocks for final six-GW calibration")
    calibration_keys = blocks[-config.calibration_gws:]
    train = canonical[~canonical["_block_key"].isin(calibration_keys)].copy()
    calibration = canonical[canonical["_block_key"].isin(calibration_keys)].copy()
    models = train_bundle(train, config.random_seed)
    selection = backtest_report["calibration_selection"]

    calibrators = {}
    for head in ("start", "cameo", "p60"):
        raw, _ = models.predict_raw(head, calibration)
        if head == "cameo":
            mask = calibration["is_starter"].eq(0).to_numpy()
            y = calibration.loc[mask, "minutes"].gt(0).astype(int).to_numpy()
        elif head == "start":
            mask = np.ones(len(calibration), dtype=bool)
            y = calibration["is_starter"].astype(int).to_numpy()
        else:
            mask = np.ones(len(calibration), dtype=bool)
            y = calibration["minutes"].ge(60).astype(int).to_numpy()
        calibrators[head] = fit_hierarchical_calibrators(
            str(selection[head]["method"]), raw[mask], y,
            model_family(calibration.loc[mask, "pos"]), config,
        )
    calibration_bundle = CalibrationBundle(
        start=calibrators["start"], cameo=calibrators["cameo"], p60=calibrators["p60"]
    )
    cutoff_text = pd.to_datetime(prediction_cutoff, utc=True).isoformat()
    run_id = deterministic_run_id(config, cutoff_text, input_identifiers)
    destination = artifact_dir or config.artifact_root / run_id
    model_card = {
        "architecture_version": config.architecture_version,
        "feature_version": config.feature_version,
        "run_id": run_id,
        "prediction_cutoff": cutoff_text,
        "training_seasons": list(config.canonical_label_seasons),
        "excluded_seasons": audit.excluded_seasons,
        "starter_source_distribution": audit.starter_source_distribution,
        "eligibility_timestamp_safe_distribution": audit.eligibility_timestamp_safe_distribution,
        "season_rows_before_filter": audit.season_rows_before_filter,
        "season_rows_after_filter": audit.season_rows_after_filter,
        "excluded_label_rows": audit.excluded_label_rows,
        "historical_eligibility_limitation": "eligibility_timestamp_safe=False is a retrospective proxy, never timestamp-safe evidence",
        "feature_lists": {
            "start": START_FEATURES, "start_minutes": START_MIN_FEATURES,
            "cameo": CAMEO_FEATURES, "cameo_minutes": CAMEO_MIN_FEATURES,
            "p60": P60_FEATURES,
        },
        "model_families": ["GK", "OUTFIELD"],
        "position_encoding": "native_categorical",
        "hyperparameters": models.hyperparameters,
        "random_seed": config.random_seed,
        "folds": backtest_report["folds"],
        "calibration_selection": selection,
        "final_calibration_keys": calibration_keys,
        "sample_counts": {
            head: {family: model.sample_count for family, model in family_models.items()}
            for head, family_models in models.heads.items()
        },
        "positive_rates": {
            head: {family: model.positive_rate for family, model in family_models.items()}
            for head, family_models in models.heads.items()
        },
        "input_data_identifiers": input_identifiers,
        "backtest_report": backtest_report,
        "engineering_status": "implemented",
        "offline_acceptance_status": "requires review of generated gates",
        "shadow_readiness": "ready",
        "production_approval": "deferred pending sufficient completed live 2026-2027 gameweeks",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_training_artifacts(destination, models, calibration_bundle, model_card)
    backtest_predictions.to_csv(destination / "backtest_predictions.csv", index=False)
    return destination, model_card
