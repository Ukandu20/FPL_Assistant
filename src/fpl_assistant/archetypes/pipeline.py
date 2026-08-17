from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .clean_sheets import score_clean_sheet_specialist
from .composites import resolve_production_composite
from .config import ArchetypeConfig, load_config
from .evidence import confidence_band, confidence_score
from .families import (
    estimate_fixture_effect, estimate_venue_effect, fixture_labels,
    points_hazard_score, return_shape_scores, temporal_context_effect,
    value_state_hysteresis, value_states,
)
from .preprocessing import percentile_within_position
from .scoring import score_base_components_with_evidence
from .team_ratings import TeamRatingParameters, build_pre_match_ratings
from .temporal import assign_evidence_windows
from .transitions import apply_context_and_transitions
from .usage import estimate_usage, stabilize_usage_state


@dataclass(frozen=True)
class SnapshotBuildResult:
    archetypes: pd.DataFrame
    team_ratings: pd.DataFrame
    evidence_tables: dict[str, pd.DataFrame]


def _row(
    *, player_id: object, snapshot: pd.Timestamp, position: object,
    archetype_id: str, display_name: str, family: str,
    score: float | None, active: bool, confidence: float,
    status: str, evidence_minutes: float, appearances: int,
    model_version: str, component_scores: dict[str, object] | None = None,
    missing_flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "player_id": str(player_id), "snapshot_date": snapshot.isoformat(),
        "fpl_position": str(position), "archetype_id": archetype_id,
        "display_name": display_name, "family": family,
        "score_0_100": None if score is None or pd.isna(score) else round(float(score), 1),
        "active_label": bool(active), "confidence_0_1": round(float(confidence), 6),
        "confidence_band": confidence_band(confidence), "trend": None,
        "status": status, "evidence_minutes": float(evidence_minutes),
        "eligible_appearances": int(appearances),
        "component_scores": json.dumps(component_scores or {}, sort_keys=True),
        "missing_data_flags": json.dumps(missing_flags or []),
        "model_version": model_version,
    }


def _team_parameters(config: ArchetypeConfig) -> TeamRatingParameters:
    values = config.section("team_ratings")
    return TeamRatingParameters(
        season_retention=float(values["season_retention"]),
        prior_equivalent_matches=float(values["prior_equivalent_matches"]),
        xg_weight=float(values["xg_weight"]),
        elo_update_k=float(values["elo_update_k"]),
        elo_home_advantage_points=float(values["elo_home_advantage_points"]),
        home_log_effect=float(values["home_log_effect"]),
        information_floor=float(values["information_floor"]),
        loss_floor=float(values["loss_floor"]),
    )


def _player_match_evidence(
    work: pd.DataFrame,
    *,
    snapshot: pd.Timestamp,
    current_season: str,
    config: ArchetypeConfig,
) -> pd.DataFrame:
    """Annotate the canonical pre-snapshot rows that fed the classifiers."""
    evidence = work.copy()
    evidence["evidence_window"] = assign_evidence_windows(
        evidence,
        current_season=current_season,
        recent_appearances=int(config.values["recent_appearances"]),
        eligible_minutes=int(config.values["eligible_appearance_minutes"]),
    )
    minutes = pd.to_numeric(evidence["minutes"], errors="coerce")
    red_card = pd.Series(False, index=evidence.index)
    if "red_card" in evidence:
        red_card |= evidence["red_card"].fillna(False).astype(bool)
    if "red_cards" in evidence:
        red_card |= pd.to_numeric(
            evidence["red_cards"], errors="coerce"
        ).fillna(0).gt(0)
    valid_pre_card = evidence.get(
        "pre_red_card_evidence_valid", pd.Series(False, index=evidence.index)
    ).fillna(False).astype(bool)
    reasons = pd.Series("", index=evidence.index, dtype="string")
    reasons.loc[minutes.lt(int(config.values["eligible_appearance_minutes"]))] = (
        "short_appearance"
    )
    reasons.loc[red_card & ~valid_pre_card] = "red_card_without_valid_event_slice"
    reasons.loc[evidence["evidence_window"].isna() & reasons.eq("")] = (
        "outside_evidence_windows"
    )
    evidence["eligible_for_production"] = reasons.eq("")
    evidence["production_exclusion_reason"] = reasons.mask(reasons.eq(""), pd.NA)
    evidence["snapshot_date"] = snapshot.isoformat()
    evidence["model_version"] = config.model_version
    return evidence.sort_values(
        [column for column in ("kickoff_utc", "match_id", "player_id") if column in evidence],
        kind="stable",
    ).reset_index(drop=True)


def build_archetype_snapshot(
    player_matches: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    current_season: str,
    team_matches: pd.DataFrame | None = None,
    player_values: pd.DataFrame | None = None,
    config: ArchetypeConfig | None = None,
    previous_states: pd.DataFrame | None = None,
) -> SnapshotBuildResult:
    selected = config or load_config()
    snapshot = pd.Timestamp(as_of)
    if snapshot.tzinfo is None:
        snapshot = snapshot.tz_localize("UTC")
    else:
        snapshot = snapshot.tz_convert("UTC")
    work = player_matches.copy()
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], utc=True, errors="raise")
    work = work[work["kickoff_utc"].lt(snapshot)].copy()
    context_columns = [
        column for column in (
            "player_id", "transferred", "appearances_since_transfer",
            "position_changed", "appearances_since_position_change",
            "minutes_since_position_change", "days_since_eligible_appearance",
            "availability_status", "current_availability_status",
            "current_availability_reason",
        ) if column in work
    ]
    player_context = None
    if len(context_columns) > 1:
        player_context = (
            work.sort_values("kickoff_utc", kind="stable")
            .drop_duplicates("player_id", keep="last")[context_columns]
        )
        if "current_availability_status" in player_context:
            player_context["availability_status"] = player_context[
                "current_availability_status"
            ].fillna(player_context.get("availability_status"))

    team_ratings = pd.DataFrame()
    team_match_evidence = pd.DataFrame()
    if team_matches is not None and not team_matches.empty:
        team_work = team_matches.copy()
        team_work["kickoff_utc"] = pd.to_datetime(team_work["kickoff_utc"], utc=True, errors="raise")
        team_work = team_work[team_work["kickoff_utc"].lt(snapshot)].copy()
        team_match_evidence = team_work.copy()
        team_match_evidence["snapshot_date"] = snapshot.isoformat()
        team_match_evidence["model_version"] = selected.model_version
        team_ratings = build_pre_match_ratings(team_work, parameters=_team_parameters(selected))
        if {"match_id", "team_id"}.issubset(work) and not team_ratings.empty:
            context = team_ratings[[
                "match_id", "team_id", "overall_elo_pre_match", "attack_rating_pre_match",
                "defence_rating_pre_match", "attack_index_pre_match", "defence_index_pre_match",
            ]].rename(columns={
                "attack_index_pre_match": "team_attack",
                "defence_index_pre_match": "team_defence_index_pre_match",
            })
            work = work.merge(context, on=["match_id", "team_id"], how="left", validate="many_to_one")
            if "opponent_id" in work:
                opponent_context = team_ratings[[
                    "match_id", "team_id", "overall_elo_pre_match",
                    "attack_index_pre_match", "defence_index_pre_match",
                ]].rename(columns={
                    "team_id": "opponent_id",
                    "overall_elo_pre_match": "opponent_elo_pre_match",
                    "attack_index_pre_match": "opponent_attack",
                    "defence_index_pre_match": "opponent_defence",
                })
                work = work.merge(
                    opponent_context,
                    on=["match_id", "opponent_id"],
                    how="left",
                    validate="many_to_one",
                )

    parts: list[pd.DataFrame] = []
    family_evidence_parts: list[pd.DataFrame] = []
    base_result = score_base_components_with_evidence(
        work, as_of=snapshot, current_season=current_season,
        config=selected, previous_states=None,
    )
    base = base_result.scores
    component_evidence = base_result.evidence
    if not base.empty:
        base = apply_context_and_transitions(
            base, player_context=player_context, previous=previous_states
        )
        if not component_evidence.empty:
            final_decisions = base[
                [
                    "player_id", "archetype_id", "score_0_100", "confidence_0_1",
                    "active_label", "status", "trend",
                ]
            ].rename(
                columns={
                    "score_0_100": "post_transition_score_0_100",
                    "confidence_0_1": "post_transition_confidence_0_1",
                    "active_label": "post_transition_active_label",
                    "status": "post_transition_status",
                    "trend": "post_transition_trend",
                }
            )
            component_evidence = component_evidence.merge(
                final_decisions,
                on=["player_id", "archetype_id"],
                how="left",
                validate="many_to_one",
            )
        base["_transitions_complete"] = True
        parts.append(base)
        composites: list[dict[str, object]] = []
        for player_id, group in base.groupby("player_id", sort=False):
            components = {
                row["archetype_id"]: {
                    "active": row["active_label"], "score": row["score_0_100"],
                    "confidence": row["confidence_0_1"],
                }
                for row in group.to_dict("records") if row["score_0_100"] is not None
            }
            position = str(group["fpl_position"].iloc[-1])
            composite = resolve_production_composite(position, components)
            if composite:
                composites.append(_row(
                    player_id=player_id, snapshot=snapshot, position=position,
                    archetype_id=str(composite["id"]), display_name=str(composite["display_name"]),
                    family="Production Composite", score=float(composite["score"]), active=True,
                    confidence=float(composite["confidence"]), status="Established" if float(composite["confidence"]) >= 0.70 else "Provisional",
                    evidence_minutes=float(group["evidence_minutes"].min()),
                    appearances=int(group["eligible_appearances"].min()),
                    model_version=selected.model_version,
                    component_scores={"required": composite["components"]},
                ))
                composites[-1]["_transitions_complete"] = True
        if composites:
            parts.append(pd.DataFrame(composites))

    usage_rows: list[dict[str, object]] = []
    usage_required = {"availability_status", "started", "minutes"}
    if usage_required.issubset(work):
        for player_id, group in work.groupby("player_id", sort=False):
            usage_group = group
            if "team_id" in group and group["team_id"].notna().any():
                current_team = group.sort_values("kickoff_utc", kind="stable").iloc[-1]["team_id"]
                usage_group = group[group["team_id"].eq(current_team)]
            usage = estimate_usage(usage_group)
            previous_usage = None
            if previous_states is not None and not previous_states.empty:
                matches_previous = previous_states[
                    previous_states["player_id"].astype(str).eq(str(player_id))
                    & previous_states["family"].eq("Usage")
                ]
                if not matches_previous.empty:
                    previous_usage = matches_previous.iloc[-1]
            immediate_reset = bool(
                group.get("transferred", pd.Series(False, index=group.index)).fillna(False).astype(bool).iloc[-1]
                or group.get("manager_changed", pd.Series(False, index=group.index)).fillna(False).astype(bool).iloc[-1]
            )
            stable_state, pending_state, pending_updates = stabilize_usage_state(
                str(usage["state"]),
                previous_state=(str(previous_usage["display_name"]) if previous_usage is not None else None),
                pending_state=(str(previous_usage.get("pending_usage_state")) if previous_usage is not None and pd.notna(previous_usage.get("pending_usage_state")) else None),
                pending_updates=(int(previous_usage.get("pending_usage_updates", 0) or 0) if previous_usage is not None else 0),
                immediate_reset=immediate_reset,
            )
            minutes = float(pd.to_numeric(group["minutes"], errors="coerce").sum())
            observations = int(usage["observations"])
            availability = group["availability_status"].astype("string").str.lower()
            known_availability = availability.isin(
                {"available", "injured", "suspended", "unavailable", "illness"}
            )
            availability_quality = float(known_availability.mean())
            confidence = confidence_score(
                eligible_minutes=minutes, required_full_minutes=450,
                eligible_appearances=observations, required_appearances=8,
                interval_width=0.40 / max(1, observations), maximum_useful_width=0.40,
                data_quality=availability_quality,
            )
            position = str(group.sort_values("kickoff_utc").iloc[-1]["fpl_position"])
            usage_rows.append(_row(
                player_id=player_id, snapshot=snapshot, position=position,
                archetype_id=stable_state.upper().replace(" ", "_"),
                display_name=stable_state, family="Usage",
                score=float(usage["start_probability"]) * 100, active=True,
                confidence=confidence,
                status="Insufficient Evidence" if confidence < 0.35 else "Provisional" if confidence < 0.55 else "Established",
                evidence_minutes=minutes, appearances=observations,
                model_version=selected.model_version,
                component_scores={key: usage[key] for key in ("start_probability", "cameo_probability", "expected_minutes")},
            ))
            usage_rows[-1]["pending_usage_state"] = pending_state
            usage_rows[-1]["pending_usage_updates"] = pending_updates
            family_evidence_parts.append(pd.DataFrame([{
                "player_id": str(player_id), "evidence_type": "usage",
                "proposed_state": str(usage["state"]), "stable_state": stable_state,
                "start_probability": usage["start_probability"],
                "cameo_probability": usage["cameo_probability"],
                "expected_minutes": usage["expected_minutes"],
                "observations": observations, "total_history_minutes": minutes,
                "confidence_0_1": confidence, "pending_state": pending_state,
                "pending_updates": pending_updates, "immediate_reset": immediate_reset,
            }]))
    if usage_rows:
        parts.append(pd.DataFrame(usage_rows))

    clean_required = {"clean_sheet", "goals_conceded", "xga", "team_defence_index_pre_match", "started"}
    if clean_required.issubset(work):
        clean = score_clean_sheet_specialist(work, as_of=snapshot, config=selected)
        if not clean.empty:
            parts.append(clean)
            clean_evidence = clean.copy()
            clean_evidence["evidence_type"] = "clean_sheet"
            family_evidence_parts.append(clean_evidence)

    if {"production_response", "matchup_difficulty"}.issubset(work):
        effect_rows: list[dict[str, object]] = []
        for player_id, group in work.groupby("player_id", sort=False):
            effect = temporal_context_effect(
                group, estimator=estimate_fixture_effect,
                current_season=current_season, as_of=snapshot,
                base_weights=dict(selected.values["temporal_weights"]),
            )
            effect_rows.append({
                "player_id": str(player_id), "fpl_position": group.iloc[-1]["fpl_position"],
                "effect_sd": effect.effect_sd, "confidence": effect.confidence,
                "context_coverage": effect.context_coverage,
                "evidence_minutes": effect.evidence_minutes,
                "eligible_appearances": effect.eligible_appearances,
            })
        effects = fixture_labels(pd.DataFrame(effect_rows))
        fixture_evidence = effects.copy()
        fixture_evidence["evidence_type"] = "fixture_behaviour"
        family_evidence_parts.append(fixture_evidence)
        family_rows: list[dict[str, object]] = []
        for item in effects.to_dict("records"):
            for archetype_id, display_name, score_column in (
                ("FODDER_HUNTER", "Fodder Hunter", "fodder_score"),
                ("MATCHUP_PROOF", "Matchup Proof", "matchup_proof_score"),
            ):
                active = item["fixture_label"] == display_name
                family_rows.append(_row(
                    player_id=item["player_id"], snapshot=snapshot, position=item["fpl_position"],
                    archetype_id=archetype_id, display_name=display_name, family="Fixture Behaviour",
                    score=item[score_column], active=active, confidence=item["confidence"],
                    status="Established" if active else "Insufficient Evidence",
                    evidence_minutes=item["evidence_minutes"], appearances=item["eligible_appearances"],
                    model_version=selected.model_version,
                    component_scores={"effect_sd": item["effect_sd"], "context_coverage": item["context_coverage"]},
                ))
        parts.append(pd.DataFrame(family_rows))

    if {"production_response", "is_home"}.issubset(work):
        venue_effects: list[dict[str, object]] = []
        for player_id, group in work.groupby("player_id", sort=False):
            effect = temporal_context_effect(
                group, estimator=estimate_venue_effect,
                current_season=current_season, as_of=snapshot,
                base_weights=dict(selected.values["temporal_weights"]),
            )
            venue_effects.append({
                "player_id": str(player_id), "fpl_position": group.iloc[-1]["fpl_position"],
                "effect_sd": effect.effect_sd, "confidence": effect.confidence,
                "context_coverage": effect.context_coverage, "minutes": effect.evidence_minutes,
                "appearances": effect.eligible_appearances,
            })
        venue = pd.DataFrame(venue_effects)
        venue["home_score"] = percentile_within_position(venue["effect_sd"], venue["fpl_position"], venue["minutes"])
        venue["road_score"] = percentile_within_position(-venue["effect_sd"], venue["fpl_position"], venue["minutes"])
        venue["anywhere_score"] = percentile_within_position(-venue["effect_sd"].abs(), venue["fpl_position"], venue["minutes"])
        venue_evidence = venue.copy()
        venue_evidence["evidence_type"] = "venue_behaviour"
        family_evidence_parts.append(venue_evidence)
        venue_rows: list[dict[str, object]] = []
        for item in venue.to_dict("records"):
            rules = (
                ("HOME_FAVORITE", "Home Favorite", "home_score", item["effect_sd"] >= 0.35),
                ("ROAD_WARRIOR", "Road Warrior", "road_score", item["effect_sd"] <= -0.35),
                ("ANYWHERE_THREAT", "Anywhere Threat", "anywhere_score", abs(item["effect_sd"]) <= 0.20),
            )
            for archetype_id, display_name, score_column, practical in rules:
                active = bool(practical and item[score_column] >= 80 and item["confidence"] >= 0.70 and item["context_coverage"] >= 1)
                venue_rows.append(_row(
                    player_id=item["player_id"], snapshot=snapshot, position=item["fpl_position"],
                    archetype_id=archetype_id, display_name=display_name, family="Venue Behaviour",
                    score=item[score_column], active=active, confidence=item["confidence"],
                    status="Established" if active else "Insufficient Evidence",
                    evidence_minutes=item["minutes"], appearances=item["appearances"],
                    model_version=selected.model_version,
                    component_scores={"effect_sd": item["effect_sd"], "context_coverage": item["context_coverage"]},
                ))
        parts.append(pd.DataFrame(venue_rows))

    if {"fpl_points", "return_event"}.issubset(work):
        shapes = return_shape_scores(work)
        shape_evidence = shapes.copy()
        shape_evidence["evidence_type"] = "return_shape"
        family_evidence_parts.append(shape_evidence)
        shape_rows: list[dict[str, object]] = []
        for item in shapes.to_dict("records"):
            confidence = confidence_score(
                eligible_minutes=item["minutes"], required_full_minutes=900,
                eligible_appearances=item["appearances"], required_appearances=25,
            )
            for archetype_id, display_name, score_column, active_column in (
                ("EXPLOSIVE_RETURNER", "Explosive Returner", "explosive_score", "explosive_active"),
                ("STEADY_RETURNER", "Steady Returner", "steady_score", "steady_active"),
            ):
                active = bool(item[active_column] and confidence >= (0.50 if archetype_id == "EXPLOSIVE_RETURNER" else 0.70))
                shape_rows.append(_row(
                    player_id=item["player_id"], snapshot=snapshot, position=item["fpl_position"],
                    archetype_id=archetype_id, display_name=display_name, family="Return Shape",
                    score=item[score_column], active=active, confidence=confidence,
                    status="Established" if active else "Insufficient Evidence",
                    evidence_minutes=item["minutes"], appearances=item["appearances"],
                    model_version=selected.model_version,
                    component_scores={"haul_probability": item["haul_probability"], "return_rate": item["return_rate"]},
                ))
        parts.append(pd.DataFrame(shape_rows))

    if {"yellow_cards", "red_cards"}.issubset(work):
        hazards = points_hazard_score(work)
        hazard_evidence = hazards.copy()
        hazard_evidence["evidence_type"] = "points_hazard"
        family_evidence_parts.append(hazard_evidence)
        hazard_rows = [
            _row(
                player_id=item["player_id"], snapshot=snapshot, position=item["fpl_position"],
                archetype_id="POINTS_HAZARD", display_name="Points Hazard", family="Risk Badge",
                score=item["score"], active=item["active"],
                confidence=min(1.0, item["minutes"] / 900),
                status="Established" if item["active"] else "Insufficient Evidence",
                evidence_minutes=item["minutes"], appearances=0,
                model_version=selected.model_version,
            )
            for item in hazards.to_dict("records")
        ]
        parts.append(pd.DataFrame(hazard_rows))

    if player_values is not None and not player_values.empty:
        for view, column in (("HISTORICAL", "historical_value_over_replacement"), ("FORWARD", "expected_next5_value_over_replacement")):
            if column not in player_values:
                continue
            values = value_states(player_values, value_column=column)
            values_evidence = values.copy()
            values_evidence["evidence_type"] = f"value_{view.lower()}"
            values_evidence["value_source_column"] = column
            family_evidence_parts.append(values_evidence)
            value_rows: list[dict[str, object]] = []
            labels = {
                "Hidden Gem": "HIDDEN_GEM", "Premium Pick": "PREMIUM_PICK",
                "High Maintenance": "HIGH_MAINTENANCE", "Bench Auto-fill": "BENCH_AUTOFILL",
            }
            for item in values.to_dict("records"):
                for display_name, archetype_id in labels.items():
                    output_id = f"{archetype_id}_{view}"
                    previous_value = None
                    if previous_states is not None and not previous_states.empty:
                        match_previous = previous_states[
                            previous_states["player_id"].astype(str).eq(str(item["player_id"]))
                            & previous_states["archetype_id"].eq(output_id)
                        ]
                        if not match_previous.empty:
                            previous_value = match_previous.iloc[-1]
                    active, value_exit_updates = value_state_hysteresis(
                        display_name,
                        price_percentile=float(item["price_percentile"]),
                        value_percentile=float(item["value_percentile"]),
                        was_active=bool(previous_value.get("active_label", False)) if previous_value is not None else False,
                        exit_updates=int(previous_value.get("below_exit_updates", 0) or 0) if previous_value is not None else 0,
                    )
                    high_value = display_name in {"Hidden Gem", "Premium Pick"}
                    score = item["value_percentile"] if high_value else 100 - item["value_percentile"]
                    historical_value = pd.to_numeric(
                        item.get("historical_minutes", 0), errors="coerce"
                    )
                    expected_value = pd.to_numeric(
                        item.get("expected_next5_minutes", 0), errors="coerce"
                    )
                    appearances_value = pd.to_numeric(
                        item.get("historical_appearances", 0), errors="coerce"
                    )
                    evidence_minutes = 0.0 if pd.isna(historical_value) else float(historical_value)
                    expected_minutes = 0.0 if pd.isna(expected_value) else float(expected_value)
                    historical_appearances = 0 if pd.isna(appearances_value) else int(appearances_value)
                    evidence_ok = evidence_minutes >= 450 if view == "HISTORICAL" else expected_minutes >= 50
                    confidence = min(1.0, evidence_minutes / 900) if view == "HISTORICAL" else min(0.69, expected_minutes / 450)
                    value_rows.append(_row(
                        player_id=item["player_id"], snapshot=snapshot, position=item["fpl_position"],
                        archetype_id=output_id, display_name=display_name,
                        family=f"Value {view.title()}", score=score,
                        active=bool(active and evidence_ok), confidence=confidence,
                        status="Established" if active and evidence_ok and confidence >= 0.70 else "Provisional" if active and evidence_ok else "Insufficient Evidence",
                        evidence_minutes=evidence_minutes, appearances=historical_appearances,
                        model_version=selected.model_version,
                        component_scores={"price_percentile": item["price_percentile"], "value_percentile": item["value_percentile"]},
                    ))
                    value_rows[-1]["below_exit_updates"] = value_exit_updates
                    value_rows[-1]["transition_applied"] = True
            parts.append(pd.DataFrame(value_rows))

    archetypes = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not archetypes.empty:
        completed = archetypes.get(
            "_transitions_complete", pd.Series(False, index=archetypes.index)
        ).fillna(False).astype(bool)
        transitioned = archetypes.loc[completed].copy()
        pending = apply_context_and_transitions(
            archetypes.loc[~completed].copy(),
            player_context=player_context,
            previous=previous_states,
        )
        archetypes = pd.concat([transitioned, pending], ignore_index=True)
        archetypes = archetypes.drop(columns=["_transitions_complete"], errors="ignore")
        archetypes = archetypes.sort_values(["player_id", "family", "archetype_id"], kind="stable").reset_index(drop=True)
    value_evidence = pd.DataFrame()
    if player_values is not None and not player_values.empty:
        value_evidence = player_values.copy()
        value_evidence["snapshot_date"] = snapshot.isoformat()
        value_evidence["model_version"] = selected.model_version
    family_evidence = (
        pd.concat(family_evidence_parts, ignore_index=True, sort=False)
        if family_evidence_parts
        else pd.DataFrame()
    )
    if not family_evidence.empty:
        family_evidence["snapshot_date"] = snapshot.isoformat()
        family_evidence["model_version"] = selected.model_version
    evidence_tables = {
        "player_match_evidence": _player_match_evidence(
            work,
            snapshot=snapshot,
            current_season=current_season,
            config=selected,
        ),
        "production_component_evidence": component_evidence,
        "family_calculation_evidence": family_evidence,
        "team_match_evidence": team_match_evidence,
        "player_value_evidence": value_evidence,
    }
    return SnapshotBuildResult(
        archetypes=archetypes,
        team_ratings=team_ratings,
        evidence_tables=evidence_tables,
    )


__all__ = ["SnapshotBuildResult", "build_archetype_snapshot"]
