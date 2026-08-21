from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


MODEL_VERSION = "minutes/v2.0"
REQUIRED_OUTPUT_COLUMNS = [
    "p_start_raw", "p_start_cal", "pred_minutes_if_start",
    "p_cameo_raw", "p_cameo_cal", "pred_minutes_if_cameo",
    "p_start_state", "p_cameo_state", "p_dnp_state",
    "state_entropy", "p60_raw", "p60_cal", "pred_exp_minutes", "history_matches",
    "season_history_matches", "cold_start", "model_version", "fallback_flags",
]
OVERRIDE_COLUMNS = [
    "pred_exp_minutes_raw", "pred_exp_minutes_final", "override_applied",
    "override_reason", "override_source", "override_information_timestamp",
]


def state_components(p_start: np.ndarray, p_cameo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ps = np.clip(np.asarray(p_start, dtype=float), 0.0, 1.0)
    pc = np.clip(np.asarray(p_cameo, dtype=float), 0.0, 1.0)
    return ps, (1.0 - ps) * pc, (1.0 - ps) * (1.0 - pc)


def state_entropy(p_start_state: np.ndarray, p_cameo_state: np.ndarray, p_dnp_state: np.ndarray) -> np.ndarray:
    probs = np.column_stack([p_start_state, p_cameo_state, p_dnp_state])
    terms = np.zeros_like(probs, dtype=float)
    positive = probs > 0
    terms[positive] = probs[positive] * np.log(probs[positive])
    return -np.sum(terms, axis=1)


def _flags(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        decoded = json.loads(str(value))
        return decoded if isinstance(decoded, list) else [str(decoded)]
    except json.JSONDecodeError:
        return [str(value)]


def validate_predictions(frame: pd.DataFrame, require_override_audit: bool = False) -> None:
    required = REQUIRED_OUTPUT_COLUMNS + ([*OVERRIDE_COLUMNS] if require_override_audit else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction schema missing fields: {sorted(missing)}")
    for col in ("p_start_raw", "p_start_cal", "p_cameo_raw", "p_cameo_cal", "p60_raw", "p60_cal",
                "p_start_state", "p_cameo_state", "p_dnp_state"):
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{col} must be finite and in [0,1]")
    for col in ("pred_minutes_if_start", "pred_minutes_if_cameo"):
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or ((values < 1) | (values > 90)).any():
            raise ValueError(f"{col} must be finite and in [1,90]")
    expected = pd.to_numeric(frame["pred_exp_minutes"], errors="coerce")
    if expected.isna().any() or ((expected < 0) | (expected > 90)).any():
        raise ValueError("pred_exp_minutes must be finite and in [0,90]")
    state_sum = frame[["p_start_state", "p_cameo_state", "p_dnp_state"]].sum(axis=1)
    if not np.allclose(state_sum, 1.0, atol=1e-9):
        raise ValueError("START/CAMEO/DNP probabilities must sum to one")
    if not (frame["model_version"].astype(str) == MODEL_VERSION).all():
        raise ValueError(f"model_version must be {MODEL_VERSION}")
    for flags in frame["fallback_flags"]:
        _flags(flags)
