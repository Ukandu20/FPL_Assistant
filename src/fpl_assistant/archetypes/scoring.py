from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .config import ArchetypeConfig, load_config
from .evidence import confidence_band, confidence_score, hysteresis_state, status_label, trend_label
from .preprocessing import aggregate_rate, normalize_position, percentile_within_position, winsorize_within_group, zscore_within_group
from .temporal import assign_evidence_windows, combine_window_scores, shrink_toward_mean


DISPLAY_NAMES = {
    "GOAL_THREAT": "Goal Threat",
    "CREATOR": "Creator",
    "DEFENSIVE_ENGINE": "Defensive Engine",
    "SAVES_MACHINE": "Saves Machine",
}


@dataclass(frozen=True)
class BaseScoringResult:
    """Final component scores and the reproducible calculation ledger."""

    scores: pd.DataFrame
    evidence: pd.DataFrame


def _weights_for(component: Mapping[str, object], position: str) -> dict[str, float]:
    by_position = component.get("weights_by_position")
    if isinstance(by_position, Mapping):
        return {str(key): float(value) for key, value in by_position[position].items()}
    return {str(key): float(value) for key, value in component["weights"].items()}


def _aggregate_metric(group: pd.DataFrame, metric: str, position: str) -> float:
    minutes = group["minutes"]
    if metric == "defcon_hit":
        if "defcon_hit" in group and group["defcon_hit"].notna().all():
            return float(pd.to_numeric(group["defcon_hit"], errors="coerce").mean())
        action_fields = ["tackles_won", "interceptions", "clearances", "blocks"]
        if position in {"MID", "FWD"}:
            action_fields.append("recoveries")
        if any(field not in group or group[field].isna().any() for field in action_fields):
            return float("nan")
        actions = sum(pd.to_numeric(group[field], errors="coerce") for field in action_fields)
        threshold = 10 if position == "DEF" else 12
        return float(actions.ge(threshold).mean())
    if metric == "goals_prevented":
        if "post_shot_xg" in group and group["post_shot_xg"].notna().all():
            return aggregate_rate(group["post_shot_xg"] - group["goals_conceded"], minutes)
        faced = pd.to_numeric(group["shots_on_target_faced"], errors="coerce").sum(min_count=1)
        saves = pd.to_numeric(group["saves"], errors="coerce").sum(min_count=1)
        return float(saves / faced) if pd.notna(faced) and faced > 0 else float("nan")
    if metric == "penalty_save_rate":
        if "penalties_saved" not in group or "penalties_faced" not in group:
            return float("nan")
        saved = pd.to_numeric(group["penalties_saved"], errors="coerce").sum(min_count=1)
        faced = pd.to_numeric(group["penalties_faced"], errors="coerce").sum(min_count=1)
        if pd.isna(saved) or pd.isna(faced):
            return float("nan")
        # Strong beta prior prevents one penalty from dominating the component.
        return float((saved + 1.0) / (faced + 10.0))
    if metric not in group or group[metric].isna().any():
        return float("nan")
    return aggregate_rate(group[metric], minutes)


def _prepare_observations(
    observations: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    current_season: str,
    config: ArchetypeConfig,
) -> pd.DataFrame:
    required = {"player_id", "season", "kickoff_utc", "fpl_position", "minutes"}
    missing = required - set(observations)
    if missing:
        raise KeyError(f"Archetype observations missing: {sorted(missing)}")
    work = observations.copy()
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], utc=True, errors="raise")
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    work = work.loc[work["kickoff_utc"].lt(as_of)].copy()
    work["fpl_position"] = normalize_position(work["fpl_position"])
    work["minutes"] = pd.to_numeric(work["minutes"], errors="coerce")
    red_card = pd.Series(False, index=work.index)
    if "red_card" in work:
        red_card |= work["red_card"].fillna(False).astype(bool)
    if "red_cards" in work:
        red_card |= pd.to_numeric(work["red_cards"], errors="coerce").fillna(0).gt(0)
    if red_card.any():
        valid_pre_card = work.get(
            "pre_red_card_evidence_valid", pd.Series(False, index=work.index)
        ).fillna(False).astype(bool)
        for column in list(work.columns):
            prefix = f"pre_red_card_{column}"
            if prefix in work:
                work.loc[red_card & valid_pre_card, column] = work.loc[
                    red_card & valid_pre_card, prefix
                ]
        work = work.loc[~red_card | valid_pre_card].copy()
    work["evidence_window"] = assign_evidence_windows(
        work,
        current_season=current_season,
        recent_appearances=int(config.values["recent_appearances"]),
        eligible_minutes=int(config.values["eligible_appearance_minutes"]),
    )
    return work


def score_base_components(
    observations: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    current_season: str,
    config: ArchetypeConfig | None = None,
    previous_states: pd.DataFrame | None = None,
    minimum_peers: int = 1,
) -> pd.DataFrame:
    """Score released V1 production components without using future rows."""
    return score_base_components_with_evidence(
        observations,
        as_of=as_of,
        current_season=current_season,
        config=config,
        previous_states=previous_states,
        minimum_peers=minimum_peers,
    ).scores


def score_base_components_with_evidence(
    observations: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    current_season: str,
    config: ArchetypeConfig | None = None,
    previous_states: pd.DataFrame | None = None,
    minimum_peers: int = 1,
) -> BaseScoringResult:
    """Score components and retain every intermediate production value."""
    selected = config or load_config()
    snapshot = pd.Timestamp(as_of)
    work = _prepare_observations(
        observations, as_of=snapshot, current_season=current_season, config=selected
    )
    eligible_floor = int(selected.values["eligible_appearance_minutes"])
    work = work.loc[work["minutes"].ge(eligible_floor) & work["evidence_window"].notna()].copy()
    if work.empty:
        return BaseScoringResult(pd.DataFrame(), pd.DataFrame())
    latest_positions = (
        work.sort_values("kickoff_utc", kind="stable")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["fpl_position"]
    )
    work["fpl_position"] = work["player_id"].map(latest_positions)
    previous_lookup: dict[tuple[str, str], Mapping[str, object]] = {}
    if previous_states is not None and not previous_states.empty:
        previous_lookup = {
            (str(row["player_id"]), str(row["archetype_id"])): row
            for row in previous_states.to_dict("records")
        }

    component_results: list[pd.DataFrame] = []
    evidence_results: list[pd.DataFrame] = []
    for component_id, definition_object in selected.values["components"].items():
        definition = dict(definition_object)
        positions = set(definition["positions"])
        component_work = work.loc[work["fpl_position"].isin(positions)].copy()
        if component_work.empty:
            continue
        aggregate_rows: list[dict[str, object]] = []
        for (player_id, window), group in component_work.groupby(
            ["player_id", "evidence_window"], sort=False
        ):
            position = str(group["fpl_position"].iloc[-1])
            weights = _weights_for(definition, position)
            values = {metric: _aggregate_metric(group, metric, position) for metric in weights}
            available_weight = sum(weight for metric, weight in weights.items() if np.isfinite(values[metric]))
            core_fields = set(definition.get("core_fields", [])) - {"minutes"}
            missing_core = sorted(field for field in core_fields if field not in group or group[field].isna().any())
            aggregate_rows.append(
                {
                    "player_id": str(player_id), "window": str(window), "fpl_position": position,
                    "minutes": float(group["minutes"].sum()), "appearances": int(len(group)),
                    "available_weight": available_weight, "missing_core": missing_core,
                    **values,
                }
            )
        aggregates = pd.DataFrame(aggregate_rows)
        metric_z_columns: dict[str, str] = {}
        group_key = aggregates["fpl_position"].astype(str) + "|" + aggregates["window"].astype(str)
        for metric in sorted({metric for position in positions for metric in _weights_for(definition, position)}):
            if metric not in aggregates:
                continue
            values = aggregates[metric]
            if metric in set(definition.get("log1p_fields", [])):
                values = np.log1p(values.clip(lower=0))
            aggregates[f"{metric}_transformed"] = values
            winsorized = winsorize_within_group(
                values, group_key,
                lower=float(selected.values["winsor_limits"][0]),
                upper=float(selected.values["winsor_limits"][1]),
            )
            aggregates[f"{metric}_winsorized"] = winsorized
            z_column = f"{metric}_z"
            aggregates[z_column] = zscore_within_group(winsorized, group_key)
            metric_z_columns[metric] = z_column

        raw_scores: list[float] = []
        for row in aggregates.to_dict("records"):
            weights = _weights_for(definition, str(row["fpl_position"]))
            available = {
                metric: weight for metric, weight in weights.items()
                if metric in metric_z_columns and pd.notna(row.get(metric_z_columns[metric]))
            }
            available_weight = sum(available.values())
            if row["missing_core"] or available_weight < float(selected.values["secondary_weight_floor"]):
                raw_scores.append(float("nan"))
            else:
                raw_scores.append(sum(row[metric_z_columns[metric]] * weight for metric, weight in available.items()) / available_weight)
        aggregates["raw_score"] = raw_scores
        window_peer_counts = aggregates.groupby(["fpl_position", "window"])["raw_score"].transform("count")
        aggregates.loc[window_peer_counts.lt(minimum_peers), "raw_score"] = np.nan
        aggregates["window_percentile"] = percentile_within_position(
            aggregates["raw_score"], group_key, aggregates["minutes"]
        )

        final_rows: list[dict[str, object]] = []
        for player_id, group in aggregates.groupby("player_id", sort=False):
            position = str(group["fpl_position"].iloc[-1])
            by_window = group.set_index("window")
            window_raw = {
                window: (float(by_window.at[window, "raw_score"]) if window in by_window.index and pd.notna(by_window.at[window, "raw_score"]) else None)
                for window in ("previous_season", "earlier_current", "recent")
            }
            temporal_raw, applied_weights = combine_window_scores(
                window_raw, selected.values["temporal_weights"]
            )
            evidence_minutes = float(group["minutes"].sum())
            appearances = int(group["appearances"].sum())
            missing_flags = sorted({item for items in group["missing_core"] for item in items})
            quality = float(np.average(group["available_weight"], weights=group["minutes"])) if evidence_minutes else 0.0
            final_rows.append(
                {
                    "player_id": str(player_id), "fpl_position": position,
                    "temporal_raw": temporal_raw, "evidence_minutes": evidence_minutes,
                    "eligible_appearances": appearances, "data_quality": quality,
                    "window_weights": applied_weights, "missing_flags": missing_flags,
                    "recent_percentile": (float(by_window.at["recent", "window_percentile"]) if "recent" in by_window.index and pd.notna(by_window.at["recent", "window_percentile"]) else None),
                    "baseline_percentile": combine_window_scores(
                        {
                            "previous_season": (float(by_window.at["previous_season", "window_percentile"]) if "previous_season" in by_window.index and pd.notna(by_window.at["previous_season", "window_percentile"]) else None),
                            "earlier_current": (float(by_window.at["earlier_current", "window_percentile"]) if "earlier_current" in by_window.index and pd.notna(by_window.at["earlier_current", "window_percentile"]) else None),
                            "recent": None,
                        },
                        selected.values["temporal_weights"],
                    )[0],
                }
            )
        final = pd.DataFrame(final_rows)
        means = final.groupby("fpl_position")["temporal_raw"].transform("mean").fillna(0.0)
        final["shrunk_raw"] = [
            shrink_toward_mean(value, evidence_minutes=minutes, position_mean=mean, prior_minutes=float(selected.values["shrinkage_prior_minutes"]))
            if value is not None and pd.notna(value) else np.nan
            for value, minutes, mean in zip(final["temporal_raw"], final["evidence_minutes"], means)
        ]
        final["score"] = percentile_within_position(
            final["shrunk_raw"], final["fpl_position"], final["evidence_minutes"]
        ).round(1)
        output_rows: list[dict[str, object]] = []
        for row in final.to_dict("records"):
            precision_width = 40.0 * float(selected.values["shrinkage_prior_minutes"]) / (
                float(row["evidence_minutes"]) + float(selected.values["shrinkage_prior_minutes"])
            )
            confidence = confidence_score(
                eligible_minutes=row["evidence_minutes"],
                required_full_minutes=float(selected.values["established_minutes"]),
                eligible_appearances=row["eligible_appearances"],
                required_appearances=int(selected.values["required_appearances"]),
                interval_width=precision_width,
                data_quality=row["data_quality"],
            )
            score = float(row["score"]) if pd.notna(row["score"]) else None
            previous = previous_lookup.get((row["player_id"], component_id), {})
            active, below_count = hysteresis_state(
                score,
                was_active=bool(previous.get("active_label", False)),
                below_exit_updates=int(previous.get("below_exit_updates", 0) or 0),
                entry=float(selected.values["entry_percentile"]),
                exit=float(selected.values["exit_percentile"]),
                required_exit_updates=int(selected.values["exit_updates"]),
            )
            has_provisional_evidence = row["evidence_minutes"] >= float(selected.values["provisional_minutes"])
            active = bool(active and confidence >= 0.35 and has_provisional_evidence and not row["missing_flags"])
            delta = (
                row["recent_percentile"] - row["baseline_percentile"]
                if row["recent_percentile"] is not None and row["baseline_percentile"] is not None
                else None
            )
            direction = "rising" if delta is not None and delta >= 10 else "declining" if delta is not None and delta <= -10 else "stable"
            prior_direction = previous.get("trend_direction")
            persistence = int(previous.get("trend_persistence", 0) or 0) + 1 if prior_direction == direction else 1
            trend = trend_label(delta, persistent_updates=persistence)
            output_rows.append(
                {
                    "player_id": row["player_id"], "snapshot_date": snapshot.isoformat(),
                    "fpl_position": row["fpl_position"], "archetype_id": component_id,
                    "display_name": DISPLAY_NAMES[component_id], "family": "Production Style",
                    "score_0_100": score, "active_label": active,
                    "confidence_0_1": confidence, "confidence_band": confidence_band(confidence),
                    "trend": trend,
                    "status": status_label(qualifies=active, confidence=confidence, trend=trend, historical_qualified=bool(previous.get("active_label", False))),
                    "evidence_minutes": row["evidence_minutes"],
                    "eligible_appearances": row["eligible_appearances"],
                    "component_scores": json.dumps({"window_weights": row["window_weights"]}, sort_keys=True),
                    "missing_data_flags": json.dumps(row["missing_flags"]),
                    "below_exit_updates": below_count, "trend_direction": direction,
                    "trend_persistence": persistence, "model_version": selected.model_version,
                }
            )
        output_frame = pd.DataFrame(output_rows)
        final_lookup = final.set_index("player_id").to_dict("index")
        output_lookup = output_frame.set_index("player_id").to_dict("index")
        evidence_rows: list[dict[str, object]] = []
        for row in aggregates.to_dict("records"):
            player_id = str(row["player_id"])
            position = str(row["fpl_position"])
            weights = _weights_for(definition, position)
            metrics = sorted(weights)
            summary = final_lookup[player_id]
            decision = output_lookup[player_id]

            def metric_values(suffix: str = "") -> dict[str, float | None]:
                values: dict[str, float | None] = {}
                for metric in metrics:
                    value = row.get(f"{metric}{suffix}")
                    values[metric] = (
                        None if value is None or pd.isna(value) else float(value)
                    )
                return values

            evidence_rows.append(
                {
                    "player_id": player_id,
                    "snapshot_date": snapshot.isoformat(),
                    "fpl_position": position,
                    "archetype_id": component_id,
                    "evidence_window": str(row["window"]),
                    "window_minutes": float(row["minutes"]),
                    "window_appearances": int(row["appearances"]),
                    "metric_values": json.dumps(metric_values(), sort_keys=True),
                    "transformed_values": json.dumps(
                        metric_values("_transformed"), sort_keys=True
                    ),
                    "winsorized_values": json.dumps(
                        metric_values("_winsorized"), sort_keys=True
                    ),
                    "position_z_scores": json.dumps(
                        metric_values("_z"), sort_keys=True
                    ),
                    "metric_weights": json.dumps(weights, sort_keys=True),
                    "available_weight": float(row["available_weight"]),
                    "missing_core_fields": json.dumps(row["missing_core"]),
                    "window_raw_score": (
                        None if pd.isna(row["raw_score"]) else float(row["raw_score"])
                    ),
                    "window_percentile": (
                        None
                        if pd.isna(row["window_percentile"])
                        else float(row["window_percentile"])
                    ),
                    "applied_window_weights": json.dumps(
                        summary["window_weights"], sort_keys=True
                    ),
                    "temporal_combined_raw": (
                        None
                        if pd.isna(summary["temporal_raw"])
                        else float(summary["temporal_raw"])
                    ),
                    "shrunk_raw": (
                        None
                        if pd.isna(summary["shrunk_raw"])
                        else float(summary["shrunk_raw"])
                    ),
                    "data_quality": float(summary["data_quality"]),
                    "recent_percentile": summary["recent_percentile"],
                    "baseline_percentile": summary["baseline_percentile"],
                    "final_score_0_100": decision["score_0_100"],
                    "confidence_0_1": float(decision["confidence_0_1"]),
                    "active_label": bool(decision["active_label"]),
                    "status": str(decision["status"]),
                    "model_version": selected.model_version,
                }
            )
        evidence_results.append(pd.DataFrame(evidence_rows))
        component_results.append(pd.DataFrame(output_rows))
    if not component_results:
        return BaseScoringResult(pd.DataFrame(), pd.DataFrame())
    scores = pd.concat(component_results, ignore_index=True).sort_values(
        ["player_id", "archetype_id"], kind="stable"
    ).reset_index(drop=True)
    evidence = pd.concat(evidence_results, ignore_index=True).sort_values(
        ["player_id", "archetype_id", "evidence_window"], kind="stable"
    ).reset_index(drop=True)
    return BaseScoringResult(scores=scores, evidence=evidence)


__all__ = [
    "BaseScoringResult",
    "DISPLAY_NAMES",
    "score_base_components",
    "score_base_components_with_evidence",
]
