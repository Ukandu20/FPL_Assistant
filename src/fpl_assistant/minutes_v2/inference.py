from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import HierarchicalCalibrators
from .models import ModelBundle, model_family
from .schema import MODEL_VERSION, state_components, state_entropy, validate_predictions


@dataclass
class CalibrationBundle:
    start: HierarchicalCalibrators
    cameo: HierarchicalCalibrators
    p60: HierarchicalCalibrators


def _merge_flags(*flag_sets: list[list[str]]) -> list[str]:
    merged: list[str] = []
    for flags in flag_sets:
        merged.extend(flags)
    return sorted(set(merged))


def predict(
    rows: pd.DataFrame,
    models: ModelBundle,
    calibrators: CalibrationBundle,
    prediction_cutoff: str | pd.Timestamp,
    run_id: str,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(prediction_cutoff, utc=True)
    if "date_sched" in rows and (pd.to_datetime(rows["date_sched"], utc=True) < cutoff).any():
        raise ValueError("Inference rows must not precede prediction_cutoff")
    families = model_family(rows["pos"])
    p_start_raw, start_model_flags = models.predict_raw("start", rows)
    p_cameo_raw, cameo_model_flags = models.predict_raw("cameo", rows)
    p60_raw, p60_model_flags = models.predict_raw("p60", rows)
    mu_start_raw, start_min_flags = models.predict_raw("start_minutes", rows)
    mu_cameo_raw, cameo_min_flags = models.predict_raw("cameo_minutes", rows)

    p_start_raw = np.clip(p_start_raw, 0.0, 1.0)
    p_cameo_raw = np.clip(p_cameo_raw, 0.0, 1.0)
    p60_raw = np.clip(p60_raw, 0.0, 1.0)
    mu_start = np.clip(mu_start_raw, 1.0, 90.0)
    mu_cameo = np.clip(mu_cameo_raw, 1.0, 90.0)

    p_start, start_cal_flags = calibrators.start.transform(p_start_raw, families)
    p_cameo, cameo_cal_flags = calibrators.cameo.transform(p_cameo_raw, families)
    p60, p60_cal_flags = calibrators.p60.transform(p60_raw, families)
    p_start_state, p_cameo_state, p_dnp_state = state_components(p_start, p_cameo)
    expected_raw = p_start * mu_start + (1.0 - p_start) * p_cameo * mu_cameo
    expected = np.clip(expected_raw, 0.0, 90.0)

    out = rows.copy().reset_index(drop=True)
    out["p_start_raw"] = p_start_raw
    out["p_start_cal"] = p_start
    out["pred_minutes_if_start"] = mu_start
    out["p_cameo_raw"] = p_cameo_raw
    out["p_cameo_cal"] = p_cameo
    out["pred_minutes_if_cameo"] = mu_cameo
    out["p_start_state"] = p_start_state
    out["p_cameo_state"] = p_cameo_state
    out["p_dnp_state"] = p_dnp_state
    out["state_entropy"] = state_entropy(p_start_state, p_cameo_state, p_dnp_state)
    out["p60_raw"] = p60_raw
    out["p60_cal"] = p60
    out["pred_exp_minutes"] = expected
    out["model_version"] = MODEL_VERSION
    out["prediction_cutoff"] = cutoff.isoformat()
    out["run_id"] = run_id

    all_flags: list[str] = []
    for i in range(len(out)):
        flags = _merge_flags(
            start_model_flags[i], cameo_model_flags[i], p60_model_flags[i],
            start_min_flags[i], cameo_min_flags[i], start_cal_flags[i],
            cameo_cal_flags[i], p60_cal_flags[i],
        )
        if mu_start_raw[i] < 1 or mu_start_raw[i] > 90:
            flags.append("physical_clip:pred_minutes_if_start")
        if mu_cameo_raw[i] < 1 or mu_cameo_raw[i] > 90:
            flags.append("physical_clip:pred_minutes_if_cameo")
        if expected_raw[i] < 0 or expected_raw[i] > 90:
            flags.append("physical_clip:pred_exp_minutes")
        if p_start[i] < 0.10 and expected[i] > 30:
            flags.append("warning:low_start_high_minutes")
        if abs(p_start_state[i] + p_cameo_state[i] + p_dnp_state[i] - 1.0) > 1e-9:
            flags.append("warning:state_probability_sum")
        all_flags.append(json.dumps(sorted(set(flags)), separators=(",", ":")))
    out["fallback_flags"] = all_flags
    validate_predictions(out)
    return out


def add_legacy_compatibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit migration adapter; canonical V2 fields remain unchanged."""
    out = frame.copy()
    out["pred_minutes"] = out["pred_exp_minutes"]
    out["expected_minutes"] = out["pred_exp_minutes"]
    out["p_start"] = out["p_start_cal"]
    out["p_cameo"] = out["p_cameo_cal"]
    out["p60"] = out["p60_cal"]
    out["p_play"] = out["p_start_state"] + out["p_cameo_state"]
    out["exp_minutes_points"] = out["p_play"] + out["p60"]
    return out
