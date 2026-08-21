from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluation import acceptance_gate, paired_block_bootstrap


def assemble_shadow(
    v1_path: Path,
    v2_path: Path,
    output: Path,
    prediction_cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(prediction_cutoff, utc=True)
    metadata = []
    for path in (v1_path, v2_path):
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        if not sidecar.exists():
            raise ValueError(f"Shadow input lacks reproducibility sidecar: {sidecar}")
        metadata.append(json.loads(sidecar.read_text(encoding="utf-8")))
    for name, meta in zip(("V1", "V2"), metadata, strict=True):
        if pd.to_datetime(meta.get("prediction_cutoff"), utc=True) != cutoff:
            raise ValueError(f"{name} sidecar prediction cutoff differs")
        if not meta.get("input_data_identifiers"):
            raise ValueError(f"{name} sidecar lacks input_data_identifiers")
    if metadata[0]["input_data_identifiers"] != metadata[1]["input_data_identifiers"]:
        raise ValueError("V1 and V2 were not generated from the same input snapshot identifiers")
    v1 = pd.read_csv(v1_path, low_memory=False)
    v2 = pd.read_csv(v2_path, low_memory=False)
    if {"match_id", "player_id"} <= set(v1.columns) & set(v2.columns):
        keys = [key for key in ("season", "match_id", "player_id") if key in v1 and key in v2]
    else:
        keys = [key for key in ("season", "gw_orig", "player_id") if key in v1 and key in v2]
    if len(keys) < 3:
        raise ValueError("Shadow artifacts require match/player or season/GW/player keys")
    for name, frame in (("V1", v1), ("V2", v2)):
        if "prediction_cutoff" not in frame:
            raise ValueError(f"{name} artifact lacks prediction_cutoff")
        timestamps = pd.to_datetime(frame["prediction_cutoff"], utc=True, errors="coerce")
        if timestamps.isna().any() or not (timestamps == cutoff).all():
            raise ValueError(f"{name} artifact was not generated at the shared prediction cutoff")
    v1_col = next((c for c in ("pred_minutes", "expected_minutes", "pred_exp_minutes") if c in v1), None)
    if v1_col is None or "pred_exp_minutes_raw" not in v2 or "pred_exp_minutes_final" not in v2:
        raise ValueError("Shadow artifacts lack raw/final prediction fields")
    v1_audit = [column for column in ("v1_operational_valid", "v1_operational_warning") if column in v1]
    left = v1[keys + [v1_col] + v1_audit].rename(columns={v1_col: "v1_pred_exp_minutes"})
    v2_cols = keys + [c for c in (
        "p_start_cal", "p_cameo_cal", "p60_cal", "p_start_state", "p_cameo_state",
        "p_dnp_state", "state_entropy", "pred_exp_minutes_raw", "pred_exp_minutes_final",
        "override_applied", "override_reason", "override_source", "override_information_timestamp",
        "prediction_cutoff", "run_id", "model_version",
    ) if c in v2]
    combined = left.merge(v2[v2_cols], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not combined["_merge"].eq("both").all():
        raise ValueError("V1 and V2 shadow rosters differ; refusing an unpaired comparison")
    combined = combined.drop(columns="_merge")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps({
        "prediction_cutoff": cutoff.isoformat(),
        "input_data_identifiers": metadata[1]["input_data_identifiers"],
        "v1_operational_valid": bool(metadata[0].get("operational_valid", False)),
        "v1_operational_warning": metadata[0].get("operational_warning", ""),
        "v2_operational_valid": True,
        "paired_rows": len(combined),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return combined


def evaluate_live_shadow(
    shadow: pd.DataFrame,
    outcomes: pd.DataFrame | None,
    iterations: int,
    confidence: float,
    seed: int,
    margins: dict[str, float],
) -> dict[str, object]:
    if "v1_operational_valid" in shadow and not shadow["v1_operational_valid"].astype(bool).all():
        return {
            "production_approval": "deferred",
            "evidence": "V1 rollback/shadow artifact is operationally invalid",
            "operational_warning": shadow.get("v1_operational_warning", pd.Series([""])).iloc[0],
            "gates": [
                acceptance_gate("p60_brier", None, margins["p60_brier"], "valid V1 shadow unavailable"),
                acceptance_gate("state_log_loss", None, margins["state_log_loss"], "valid V1 shadow unavailable"),
                acceptance_gate("xpoints_mae", None, margins["xpoints_mae"], "valid V1 shadow unavailable"),
            ],
        }
    if outcomes is None or outcomes.empty:
        return {
            "production_approval": "deferred",
            "evidence": "No completed live outcomes supplied",
            "gates": [
                acceptance_gate("p60_brier", None, margins["p60_brier"]),
                acceptance_gate("state_log_loss", None, margins["state_log_loss"]),
                acceptance_gate("xpoints_mae", None, margins["xpoints_mae"]),
            ],
        }
    keys = [key for key in ("match_id", "player_id") if key in shadow and key in outcomes]
    merged = shadow.merge(outcomes, on=keys, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No paired shadow predictions and outcomes")
    gates: list[dict[str, object]] = []
    # V1 probability/state fields may not be available in legacy output. Gates
    # remain explicitly deferred instead of fabricating a comparator.
    for gate_name, v2_col, v1_col, target_col in (
        ("p60_brier", "p60_cal", "v1_p60", "actual_p60"),
        ("state_log_loss", "v2_state_log_loss", "v1_state_log_loss", None),
        ("xpoints_mae", "v2_xpoints", "v1_xpoints", "actual_xpoints"),
    ):
        margin = margins[gate_name]
        needed = [v2_col, v1_col] + ([target_col] if target_col else [])
        if any(column not in merged for column in needed):
            gates.append(acceptance_gate(gate_name, None, margin, f"missing live fields: {needed}"))
            continue
        if target_col:
            merged[f"_{gate_name}_v2_loss"] = (merged[v2_col] - merged[target_col]).abs()
            merged[f"_{gate_name}_v1_loss"] = (merged[v1_col] - merged[target_col]).abs()
            if gate_name == "p60_brier":
                merged[f"_{gate_name}_v2_loss"] **= 2
                merged[f"_{gate_name}_v1_loss"] **= 2
        else:
            merged[f"_{gate_name}_v2_loss"] = merged[v2_col]
            merged[f"_{gate_name}_v1_loss"] = merged[v1_col]
        block = "match_id" if "match_id" in merged else "player_id"
        result = paired_block_bootstrap(
            merged, f"_{gate_name}_v2_loss", f"_{gate_name}_v1_loss", block,
            iterations, confidence, seed,
        )
        gates.append(acceptance_gate(gate_name, result, margin))
    return {
        "production_approval": "pass" if all(g["status"] == "pass" for g in gates) else "deferred_or_failed",
        "paired_rows": len(merged), "gates": gates,
    }
