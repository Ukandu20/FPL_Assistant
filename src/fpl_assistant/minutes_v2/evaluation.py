from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-9


def minutes_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    error = p[valid] - y[valid]
    return {
        "n": int(valid.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    positive = y == 1
    negative = y == 0
    if not positive.any() or not negative.any():
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def probability_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    y = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(predicted, dtype=float), EPS, 1.0 - EPS)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0:
        return {"n": 0}
    prevalence = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    prevalence_brier = prevalence * (1.0 - prevalence)
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    logit = np.log(p / (1.0 - p))
    design = np.column_stack([np.ones(len(p)), logit])
    beta = np.zeros(2)
    if len(np.unique(y)) > 1:
        for _ in range(100):
            fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -35, 35)))
            weights = np.clip(fitted * (1 - fitted), 1e-6, None)
            step = np.linalg.solve(design.T @ (weights[:, None] * design) + np.eye(2) * 1e-6, design.T @ (y - fitted))
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break
    else:
        beta[:] = np.nan
    return {
        "n": int(len(y)), "positive_count": int(y.sum()), "prevalence": prevalence,
        "auc": _auc(y, p), "brier": brier, "prevalence_brier": prevalence_brier,
        "brier_skill": float(1.0 - brier / prevalence_brier) if prevalence_brier > 0 else float("nan"),
        "log_loss": log_loss, "calibration_intercept": float(beta[0]),
        "calibration_slope": float(beta[1]),
        "reliability_bins": probability_bands(y, p).to_dict(orient="records"),
    }


def probability_bands(actual: np.ndarray, predicted: np.ndarray, minutes_error: np.ndarray | None = None) -> pd.DataFrame:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    errors = np.asarray(minutes_error, dtype=float) if minutes_error is not None else np.full(len(y), np.nan)
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
    band = pd.cut(np.clip(p, 0, 1), bins=bins, labels=labels, include_lowest=True, right=False)
    # Include p=1 in the last fixed band.
    band = pd.Series(band).astype("string")
    band[p >= 1.0] = labels[-1]
    frame = pd.DataFrame({"band": band, "actual": y, "predicted": p, "minutes_error": errors})
    rows = []
    for label in labels:
        sample = frame[frame["band"].eq(label)]
        rows.append({
            "band": label, "n": len(sample),
            "mean_predicted": float(sample["predicted"].mean()) if len(sample) else float("nan"),
            "actual_rate": float(sample["actual"].mean()) if len(sample) else float("nan"),
            "minutes_mae": float(sample["minutes_error"].abs().mean()) if sample["minutes_error"].notna().any() else float("nan"),
        })
    return pd.DataFrame(rows)


def state_log_loss(actual_state: np.ndarray, probabilities: np.ndarray) -> float:
    states = np.asarray(actual_state, dtype=int)
    probs = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("State probabilities must have columns START, CAMEO, DNP")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("State probabilities must sum to one")
    return float(-np.mean(np.log(probs[np.arange(len(states)), states])))


def starter_duration_bands(frame: pd.DataFrame) -> pd.DataFrame:
    starters = frame[frame["is_starter"].eq(1)].copy()
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
    starters["band"] = pd.cut(
        starters["p_start_cal"].clip(0, 1), bins=bins, labels=labels,
        include_lowest=True, right=False,
    ).astype("string")
    starters.loc[starters["p_start_cal"].ge(1), "band"] = labels[-1]
    rows = []
    for label in labels:
        sample = starters[starters["band"].eq(label)]
        rows.append({
            "band": label, "n": len(sample),
            "mean_actual_starter_minutes": float(sample["minutes"].mean()) if len(sample) else float("nan"),
            "mean_starter_head_prediction": float(sample["pred_minutes_if_start"].mean()) if len(sample) else float("nan"),
            "mae": float((sample["pred_minutes_if_start"] - sample["minutes"]).abs().mean()) if len(sample) else float("nan"),
        })
    return pd.DataFrame(rows)


def subgroup_metrics(frame: pd.DataFrame, prediction_col: str = "pred_exp_minutes") -> list[dict[str, object]]:
    dimensions = [column for column in ("pos", "season", "gw_orig", "cold_start", "eligibility_timestamp_safe") if column in frame]
    result: list[dict[str, object]] = []
    for dimension in dimensions:
        for value, sample in frame.groupby(dimension, dropna=False, observed=True):
            result.append({"dimension": dimension, "value": str(value), **minutes_metrics(sample["minutes"], sample[prediction_col])})
    return result


def mandatory_baselines(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    p_start: np.ndarray,
    uncalibrated_mixture: np.ndarray,
    calibrated_v2: np.ndarray,
    v1: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    position_means = train.groupby("pos", observed=True)["minutes"].mean()
    result = {
        "position_mean": evaluation["pos"].map(position_means).astype(float).to_numpy(),
        "previous_match_minutes": evaluation["min_lag1"].astype(float).to_numpy(),
        "leak_free_ewma": evaluation["min_ewm_hl2"].astype(float).to_numpy(),
        "90_times_p_start": 90.0 * np.asarray(p_start, dtype=float),
        "uncalibrated_soft_mixture": np.asarray(uncalibrated_mixture, dtype=float),
        "calibrated_v2": np.asarray(calibrated_v2, dtype=float),
    }
    result["v1_current"] = np.asarray(v1, dtype=float) if v1 is not None else np.full(len(evaluation), np.nan)
    return result


@dataclass
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    confidence: float
    iterations: int
    block_column: str


def paired_block_bootstrap(
    frame: pd.DataFrame,
    v2_loss_col: str,
    v1_loss_col: str,
    block_col: str,
    iterations: int,
    confidence: float,
    seed: int,
) -> BootstrapResult:
    if block_col not in frame:
        raise ValueError(f"Bootstrap block column missing: {block_col}")
    grouped = frame.assign(_difference=frame[v2_loss_col] - frame[v1_loss_col]).groupby(block_col)["_difference"]
    blocks = [values.to_numpy(dtype=float) for _, values in grouped]
    if len(blocks) < 2:
        raise ValueError("Paired block bootstrap requires at least two player/fixture blocks")
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for i in range(iterations):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        draws[i] = np.concatenate([blocks[j] for j in selected]).mean()
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        estimate=float(frame[v2_loss_col].sub(frame[v1_loss_col]).mean()),
        lower=float(np.quantile(draws, alpha)), upper=float(np.quantile(draws, 1.0 - alpha)),
        confidence=confidence, iterations=iterations, block_column=block_col,
    )


def acceptance_gate(name: str, result: BootstrapResult | None, margin: float, unavailable_reason: str | None = None) -> dict[str, object]:
    if result is None:
        return {"gate": name, "status": "deferred", "margin": margin, "evidence": unavailable_reason or "required paired outcomes unavailable"}
    return {
        "gate": name, "status": "pass" if result.upper <= margin else "fail",
        "margin": margin, "paired_difference": result.estimate,
        "ci_lower": result.lower, "ci_upper": result.upper,
        "confidence": result.confidence, "iterations": result.iterations,
        "block_column": result.block_column,
    }


def offline_acceptance_report(
    predictions: pd.DataFrame,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    """Evaluate implementable offline gates; never promote production itself."""
    holdout = predictions[predictions["is_final_holdout"]].copy()
    if holdout.empty:
        raise ValueError("Offline acceptance requires the untouched final holdout")
    actual = holdout["minutes"].to_numpy(dtype=float)
    v2 = holdout["pred_exp_minutes"].to_numpy(dtype=float)
    simple_names = [
        "position_mean", "previous_match_minutes", "leak_free_ewma", "90_times_p_start",
        "uncalibrated_soft_mixture",
    ]
    simple = {
        name: minutes_metrics(actual, holdout[f"baseline_{name}"].to_numpy(dtype=float))
        for name in simple_names
    }
    strongest_name = min(simple, key=lambda name: float(simple[name]["mae"]) if np.isfinite(simple[name]["mae"]) else float("inf"))
    v2_metrics = minutes_metrics(actual, v2)
    v1_values = holdout["baseline_v1_current"].to_numpy(dtype=float)
    v1_available = np.isfinite(v1_values).all()
    if v1_available:
        holdout["_v2_minutes_loss"] = np.abs(v2 - actual)
        holdout["_v1_minutes_loss"] = np.abs(v1_values - actual)
        block = "match_id" if "match_id" in holdout else "player_id"
        minutes_ci = paired_block_bootstrap(
            holdout, "_v2_minutes_loss", "_v1_minutes_loss", block,
            iterations, confidence, seed,
        )
        beats_v1 = float(v2_metrics["mae"]) < float(minutes_metrics(actual, v1_values)["mae"])
        v1_gate: dict[str, object] = {
            "gate": "minutes_vs_v1", "status": "pass" if beats_v1 else "fail",
            "v2_mae": v2_metrics["mae"], "v1_mae": minutes_metrics(actual, v1_values)["mae"],
            "paired_ci": minutes_ci.__dict__,
        }
    else:
        v1_gate = {"gate": "minutes_vs_v1", "status": "deferred", "evidence": "paired V1 holdout predictions unavailable"}
    start_metrics = probability_metrics(holdout["is_starter"], holdout["p_start_cal"])
    strongest_mae = float(simple[strongest_name]["mae"])
    minutes_simple_gate = {
        "gate": "minutes_vs_strongest_simple", "status": "pass" if float(v2_metrics["mae"]) < strongest_mae else "fail",
        "v2_mae": v2_metrics["mae"], "strongest_baseline": strongest_name,
        "strongest_baseline_mae": strongest_mae,
    }
    return {
        "scope": "untouched_final_historical_holdout",
        "engineering_readiness": "implemented",
        "offline_model_acceptance": "requires all non-deferred offline gates and review diagnostics",
        "shadow_readiness": "ready after timestamp-matched V1/V2 run",
        "production_approval": "prohibited_from_offline_evidence",
        "gates": [
            minutes_simple_gate,
            v1_gate,
            {
                "gate": "positive_start_brier_skill",
                "status": "pass" if float(start_metrics.get("brier_skill", float("nan"))) > 0 else "fail",
                "brier_skill": start_metrics.get("brier_skill"),
            },
            {"gate": "p60_non_inferiority", "status": "deferred", "evidence": "V1 P60 probabilities not present in legacy minutes artifact"},
            {"gate": "state_non_inferiority", "status": "deferred", "evidence": "V1 has no three-state probability contract"},
            {"gate": "downstream_xpoints_non_inferiority", "status": "deferred", "evidence": "paired downstream xPoints outcomes/predictions not supplied"},
            {"gate": "probability_band_continuity", "status": "review", "evidence": start_metrics.get("reliability_bins", [])},
            {"gate": "subgroup_collapse", "status": "review", "evidence": subgroup_metrics(holdout)},
        ],
        "v2_minutes": v2_metrics,
        "simple_baselines": simple,
        "start_probability": start_metrics,
    }
