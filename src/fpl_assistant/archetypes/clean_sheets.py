from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import ArchetypeConfig, load_config
from .evidence import confidence_band, confidence_score
from .preprocessing import aggregate_rate, normalize_position, percentile_within_position, zscore_within_group


def score_clean_sheet_specialist(
    observations: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    config: ArchetypeConfig | None = None,
) -> pd.DataFrame:
    required = {
        "player_id", "kickoff_utc", "fpl_position", "minutes", "started",
        "clean_sheet", "goals_conceded", "xga", "team_defence_index_pre_match",
    }
    missing = required - set(observations)
    if missing:
        raise KeyError(f"Clean-sheet observations missing: {sorted(missing)}")
    selected = config or load_config()
    snapshot = pd.Timestamp(as_of)
    if snapshot.tzinfo is None:
        snapshot = snapshot.tz_localize("UTC")
    work = observations.copy()
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], utc=True, errors="raise")
    work["fpl_position"] = normalize_position(work["fpl_position"])
    work = work[
        work["kickoff_utc"].lt(snapshot)
        & work["fpl_position"].isin({"DEF", "GKP"})
        & pd.to_numeric(work["minutes"], errors="coerce").ge(30)
    ].copy()
    if "red_card" in work:
        work = work[~work["red_card"].fillna(False).astype(bool)]
    rows: list[dict[str, object]] = []
    for player_id, group in work.groupby("player_id", sort=False):
        position = str(group["fpl_position"].iloc[-1])
        starts = group[group["started"].fillna(False).astype(bool)]
        opportunities = starts[pd.to_numeric(starts["minutes"], errors="coerce").ge(60)]
        core_missing = [
            field for field in ("clean_sheet", "goals_conceded", "xga", "team_defence_index_pre_match")
            if field not in group or group[field].isna().any()
        ]
        clean_sheet_probability = (
            float(opportunities["clean_sheet"].fillna(False).astype(bool).mean())
            if not opportunities.empty else np.nan
        )
        # Ratio below one represents conceding fewer goals than the chance quality allowed.
        xga = pd.to_numeric(group["xga"], errors="coerce")
        goals_conceded = pd.to_numeric(group["goals_conceded"], errors="coerce")
        prevention = aggregate_rate(xga - goals_conceded, group["minutes"])
        rows.append(
            {
                "player_id": str(player_id), "fpl_position": position,
                "clean_sheet_probability": clean_sheet_probability,
                "goals_prevention": prevention,
                "defence_index": float(pd.to_numeric(group["team_defence_index_pre_match"], errors="coerce").mean()),
                "evidence_minutes": float(pd.to_numeric(group["minutes"], errors="coerce").sum()),
                "eligible_appearances": int(len(group)), "context_starts": int(len(opportunities)),
                "missing_flags": core_missing,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    groups = result["fpl_position"]
    result["raw"] = (
        0.50 * zscore_within_group(result["clean_sheet_probability"], groups)
        + 0.30 * zscore_within_group(result["goals_prevention"], groups)
        + 0.20 * zscore_within_group(result["defence_index"], groups)
    )
    result["score_0_100"] = percentile_within_position(
        result["raw"], result["fpl_position"], result["evidence_minutes"]
    ).round(1)
    output: list[dict[str, object]] = []
    for row in result.to_dict("records"):
        context = min(1.0, row["context_starts"] / 10.0)
        confidence = confidence_score(
            eligible_minutes=row["evidence_minutes"], required_full_minutes=900,
            eligible_appearances=row["eligible_appearances"], required_appearances=10,
            context_coverage=context, data_quality=0.0 if row["missing_flags"] else 1.0,
        )
        active = bool(
            not row["missing_flags"] and row["evidence_minutes"] >= 450
            and row["score_0_100"] >= 80 and confidence >= 0.35
        )
        output.append(
            {
                "player_id": row["player_id"], "snapshot_date": snapshot.isoformat(),
                "fpl_position": row["fpl_position"], "archetype_id": "CLEAN_SHEET_SPECIALIST",
                "display_name": "Clean-Sheet Specialist", "family": "Production",
                "score_0_100": row["score_0_100"] if not row["missing_flags"] else None,
                "active_label": active, "confidence_0_1": confidence,
                "confidence_band": confidence_band(confidence), "trend": None,
                "status": "Established" if active and row["evidence_minutes"] >= 900 else "Provisional" if active else "Insufficient Evidence",
                "evidence_minutes": row["evidence_minutes"],
                "eligible_appearances": row["eligible_appearances"],
                "component_scores": json.dumps(
                    {"clean_sheet_probability": row["clean_sheet_probability"], "goals_prevention": row["goals_prevention"], "defence_index": row["defence_index"]},
                    sort_keys=True,
                ),
                "missing_data_flags": json.dumps(row["missing_flags"]),
                "model_version": selected.model_version,
            }
        )
    return pd.DataFrame(output)


__all__ = ["score_clean_sheet_specialist"]
