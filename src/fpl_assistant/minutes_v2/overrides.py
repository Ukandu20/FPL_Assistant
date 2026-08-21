from __future__ import annotations

import pandas as pd

from .schema import validate_predictions


REQUIRED_OVERRIDE_INPUT = {
    "player_id", "confirmed_unavailable", "override_reason",
    "override_source", "information_timestamp",
}


def apply_confirmed_absence_overrides(
    predictions: pd.DataFrame,
    overrides: pd.DataFrame | None,
    prediction_cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(prediction_cutoff, utc=True)
    out = predictions.copy()
    out["pred_exp_minutes_raw"] = out["pred_exp_minutes"]
    out["pred_exp_minutes_final"] = out["pred_exp_minutes"]
    out["override_applied"] = False
    out["override_reason"] = pd.NA
    out["override_source"] = pd.NA
    out["override_information_timestamp"] = pd.Series(
        pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
    )
    if overrides is None or overrides.empty:
        validate_predictions(out, require_override_audit=True)
        return out
    missing = REQUIRED_OVERRIDE_INPUT - set(overrides.columns)
    if missing:
        raise ValueError(f"Override data missing auditable fields: {sorted(missing)}")
    ov = overrides.copy()
    ov["information_timestamp"] = pd.to_datetime(ov["information_timestamp"], utc=True, errors="coerce")
    if ov["information_timestamp"].isna().any():
        raise ValueError("Override information_timestamp may not be missing")
    late = ov["information_timestamp"] > cutoff
    if late.any():
        # Post-cutoff knowledge is rejected, never quietly applied or used as history.
        ov = ov[~late].copy()
    ov = ov[ov["confirmed_unavailable"].astype("string").str.lower().isin({"1", "true", "yes"})]
    ov = ov.sort_values("information_timestamp").drop_duplicates("player_id", keep="last")
    if ov.empty:
        validate_predictions(out, require_override_audit=True)
        return out
    indexed = ov.set_index("player_id")
    for idx, player_id in out["player_id"].items():
        if player_id not in indexed.index:
            continue
        record = indexed.loc[player_id]
        if isinstance(record, pd.DataFrame):
            record = record.iloc[-1]
        out.at[idx, "pred_exp_minutes_final"] = 0.0
        out.at[idx, "override_applied"] = True
        out.at[idx, "override_reason"] = str(record["override_reason"])
        out.at[idx, "override_source"] = str(record["override_source"])
        out.at[idx, "override_information_timestamp"] = record["information_timestamp"]
    validate_predictions(out, require_override_audit=True)
    return out
