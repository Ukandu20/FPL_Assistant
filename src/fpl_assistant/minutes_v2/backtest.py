from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .calibration import (
    CalibrationFoldData, fit_hierarchical_calibrators, select_calibration,
)
from .config import MinutesV2Config
from .evaluation import (
    mandatory_baselines, minutes_metrics, offline_acceptance_report, probability_bands,
    probability_metrics, starter_duration_bands, state_log_loss, subgroup_metrics,
)
from .folds import ChronologicalFold, assert_holdout_isolated, fixture_block_key, generate_folds
from .models import ModelBundle, model_family, train_bundle
from .schema import state_components, state_entropy


@dataclass
class FoldRawPredictions:
    definition: ChronologicalFold
    train: pd.DataFrame
    calibration: pd.DataFrame
    evaluation: pd.DataFrame
    models: ModelBundle
    calibration_raw: dict[str, np.ndarray]
    evaluation_raw: dict[str, np.ndarray]


def _head_raw(models: ModelBundle, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {head: models.predict_raw(head, rows)[0] for head in models.heads}


def _target(rows: pd.DataFrame, head: str) -> tuple[np.ndarray, np.ndarray]:
    if head == "cameo":
        mask = rows["is_starter"].eq(0).to_numpy()
        return mask, rows.loc[mask, "minutes"].gt(0).astype(int).to_numpy()
    if head == "start":
        mask = np.ones(len(rows), dtype=bool)
        return mask, rows["is_starter"].astype(int).to_numpy()
    if head == "p60":
        mask = np.ones(len(rows), dtype=bool)
        return mask, rows["minutes"].ge(60).astype(int).to_numpy()
    raise ValueError(head)


def _merge_v1(eval_rows: pd.DataFrame, v1_predictions: pd.DataFrame | None) -> np.ndarray | None:
    if v1_predictions is None:
        return None
    if {"match_id", "player_id"} <= set(eval_rows.columns) & set(v1_predictions.columns):
        keys = ["match_id", "player_id"]
    else:
        keys = [key for key in ("season", "gw_orig", "player_id") if key in eval_rows and key in v1_predictions]
    prediction_col = next((c for c in ("pred_minutes", "expected_minutes", "pred_exp_minutes") if c in v1_predictions), None)
    if len(keys) < 3 or prediction_col is None:
        raise ValueError("V1 comparison requires match/player or season/GW/player keys and a minutes field")
    right = v1_predictions[keys + [prediction_col]].drop_duplicates(keys, keep=False)
    return eval_rows[keys].merge(right, on=keys, how="left")[prediction_col].to_numpy(dtype=float)


def run_walk_forward(
    rows: pd.DataFrame,
    config: MinutesV2Config,
    v1_predictions: pd.DataFrame | None = None,
    hyperparameters: dict[str, dict[str, object]] | None = None,
    feature_overrides: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    work = rows.copy()
    work["_block_key"] = fixture_block_key(work)
    folds = generate_folds(work, config)
    assert_holdout_isolated(folds)
    raw_folds: list[FoldRawPredictions] = []
    for fold in folds:
        train = work[work["_block_key"].isin(fold.train_keys)].copy()
        calibration = work[work["_block_key"].isin(fold.calibration_keys)].copy()
        evaluation = work[work["_block_key"].isin(fold.evaluation_keys)].copy()
        models = train_bundle(train, config.random_seed, hyperparameters, feature_overrides)
        raw_folds.append(FoldRawPredictions(
            fold, train, calibration, evaluation, models,
            _head_raw(models, calibration), _head_raw(models, evaluation),
        ))

    selections = {}
    selectable = [item for item in raw_folds if not item.definition.is_final_holdout]
    for head in ("start", "cameo", "p60"):
        evidence: list[CalibrationFoldData] = []
        for item in selectable:
            cal_mask, cal_y = _target(item.calibration, head)
            eval_mask, eval_y = _target(item.evaluation, head)
            evidence.append(CalibrationFoldData(
                fold_id=item.definition.fold_id,
                calibration_raw=item.calibration_raw[head][cal_mask], calibration_y=cal_y,
                evaluation_raw=item.evaluation_raw[head][eval_mask], evaluation_y=eval_y,
            ))
        selections[head] = select_calibration(evidence, config)

    prediction_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, object]] = []
    for item in raw_folds:
        calibrated: dict[str, np.ndarray] = {}
        fallback_flags: dict[str, list[list[str]]] = {}
        for head in ("start", "cameo", "p60"):
            cal_mask, cal_y = _target(item.calibration, head)
            hierarchy = fit_hierarchical_calibrators(
                selections[head].method,
                item.calibration_raw[head][cal_mask], cal_y,
                model_family(item.calibration.loc[cal_mask, "pos"]), config,
            )
            calibrated[head], fallback_flags[head] = hierarchy.transform(
                item.evaluation_raw[head], model_family(item.evaluation["pos"])
            )
        ps_raw = np.clip(item.evaluation_raw["start"], 0, 1)
        pc_raw = np.clip(item.evaluation_raw["cameo"], 0, 1)
        mu_start = np.clip(item.evaluation_raw["start_minutes"], 1, 90)
        mu_cameo = np.clip(item.evaluation_raw["cameo_minutes"], 1, 90)
        raw_mixture = ps_raw * mu_start + (1 - ps_raw) * pc_raw * mu_cameo
        ps, pc, p60 = calibrated["start"], calibrated["cameo"], calibrated["p60"]
        expected = np.clip(ps * mu_start + (1 - ps) * pc * mu_cameo, 0, 90)
        p_start_state, p_cameo_state, p_dnp_state = state_components(ps, pc)
        pred = item.evaluation.copy()
        pred["fold_id"] = item.definition.fold_id
        pred["is_final_holdout"] = item.definition.is_final_holdout
        pred["p_start_raw"] = ps_raw
        pred["p_start_cal"] = ps
        pred["p_cameo_raw"] = pc_raw
        pred["p_cameo_cal"] = pc
        pred["p60_raw"] = np.clip(item.evaluation_raw["p60"], 0, 1)
        pred["p60_cal"] = p60
        pred["pred_minutes_if_start"] = mu_start
        pred["pred_minutes_if_cameo"] = mu_cameo
        pred["p_start_state"] = p_start_state
        pred["p_cameo_state"] = p_cameo_state
        pred["p_dnp_state"] = p_dnp_state
        pred["state_entropy"] = state_entropy(p_start_state, p_cameo_state, p_dnp_state)
        pred["pred_exp_minutes_uncalibrated"] = raw_mixture
        pred["pred_exp_minutes"] = expected
        actual_state = np.where(pred["is_starter"].eq(1), 0, np.where(pred["minutes"].gt(0), 1, 2))
        v1 = _merge_v1(pred, v1_predictions)
        baselines = mandatory_baselines(item.train, pred, ps, raw_mixture, expected, v1)
        baseline_metrics = {name: minutes_metrics(pred["minutes"], values) for name, values in baselines.items()}
        for name, values in baselines.items():
            pred[f"baseline_{name}"] = values
        minutes_error = expected - pred["minutes"].to_numpy(dtype=float)
        start_metrics = probability_metrics(pred["is_starter"], ps)
        start_metrics["reliability_bins"] = probability_bands(
            pred["is_starter"], ps, minutes_error
        ).to_dict(orient="records")
        nonstarter = pred["is_starter"].eq(0).to_numpy()
        cameo_metrics = probability_metrics(pred.loc[nonstarter, "minutes"].gt(0), pc[nonstarter])
        cameo_metrics["reliability_bins"] = probability_bands(
            pred.loc[nonstarter, "minutes"].gt(0), pc[nonstarter], minutes_error[nonstarter]
        ).to_dict(orient="records")
        p60_metrics = probability_metrics(pred["minutes"].ge(60), p60)
        p60_metrics["reliability_bins"] = probability_bands(
            pred["minutes"].ge(60), p60, minutes_error
        ).to_dict(orient="records")
        fold_reports.append({
            "fold": item.definition.to_dict(),
            "baselines": baseline_metrics,
            "start": start_metrics,
            "cameo": cameo_metrics,
            "p60": p60_metrics,
            "state_log_loss": state_log_loss(actual_state, pred[["p_start_state", "p_cameo_state", "p_dnp_state"]].to_numpy()),
            "starter_duration_by_p_start_band": starter_duration_bands(pred).to_dict(orient="records"),
            "subgroups": subgroup_metrics(pred),
            "historical_eligibility_proxy_rows": int((~pred["eligibility_timestamp_safe"]).sum()) if "eligibility_timestamp_safe" in pred else None,
        })
        prediction_frames.append(pred)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    report = {
        "protocol": {
            "calibration_gws": 6, "evaluation_gws": 6, "step_gws": 6,
            "final_holdout_gws": 6, "final_holdout_used_for_selection": False,
        },
        "folds": [fold.to_dict() for fold in folds],
        "calibration_selection": {head: asdict(value) for head, value in selections.items()},
        "fold_reports": fold_reports,
        "ablation_feature_overrides": feature_overrides or {},
        "offline_acceptance": offline_acceptance_report(
            all_predictions, config.bootstrap_iterations, config.bootstrap_confidence,
            config.random_seed,
        ),
    }
    return all_predictions, report
