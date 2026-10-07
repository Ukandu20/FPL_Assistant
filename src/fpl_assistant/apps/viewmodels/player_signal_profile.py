from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

from fpl_assistant.archetypes.config import ArchetypeConfig, load_config
from fpl_assistant.archetypes.scoring import score_base_components_with_evidence


def current_season_signal_profiles(
    match_evidence: pd.DataFrame, season: str, gameweeks: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute display-only season totals without changing published models.

    Use one season-wide window so raw components, peer rankings, headline and
    confidence all describe the same sample. Retain the model's eligibility,
    missing-data and shrinkage rules, but never inherit historical states.
    """
    if gameweeks is not None:
        return _fpl_eligible_signal_profiles(match_evidence, season, gameweeks)
    required = {
        "season", "snapshot_date", "player_id", "kickoff_utc",
        "fpl_position", "minutes",
    }
    if match_evidence.empty or not required.issubset(match_evidence):
        return pd.DataFrame(), pd.DataFrame()
    current = match_evidence.loc[
        match_evidence["season"].astype("string").eq(str(season))
    ].copy()
    if current.empty:
        return pd.DataFrame(), pd.DataFrame()
    cutoff = pd.to_datetime(current["snapshot_date"], utc=True, errors="coerce").max()
    if pd.isna(cutoff):
        return pd.DataFrame(), pd.DataFrame()
    config = load_config()
    season_config = ArchetypeConfig(
        values={**config.values, "recent_appearances": len(current)},
        source_path=config.source_path,
    )
    result = score_base_components_with_evidence(
        current, as_of=cutoff, current_season=season, config=season_config,
    )
    if not result.scores.empty:
        # A season-wide total has no separate recent-versus-baseline trend.
        result.scores["trend"] = None
    if not result.evidence.empty:
        result.evidence["evidence_window"] = "current_season"
        result.evidence["applied_window_weights"] = '{"current_season": 1.0}'
    return result.scores, result.evidence


def _eligible_fpl_matches(gameweeks: pd.DataFrame) -> pd.DataFrame:
    """The shared appearance rule for Overview signals and FPL Output."""
    if not {"player_id", "fpl_pos", "minutes", "total_points"}.issubset(gameweeks):
        return pd.DataFrame()
    work = gameweeks.copy()
    work["minutes"] = pd.to_numeric(work["minutes"], errors="coerce").fillna(0)
    return work.loc[work["minutes"].ge(30)].copy()


def _fpl_eligible_signal_profiles(
    match_evidence: pd.DataFrame, season: str, gameweeks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Left-join provider statistics onto the official eligible fixture list.

    Missing provider rows stay missing. Red-card appearances remain eligible,
    just as they do for FPL Output; this never changes model scoring rules.
    """
    eligible = _eligible_fpl_matches(gameweeks)
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible["player_id"] = eligible["player_id"].astype("string")
    config = load_config()
    metrics = set(config.values["provider_fields"]) - {"minutes"}
    metrics.update({"defcon_hit", "non_penalty_goals"})
    # Only bring metric fields across: FPL owns membership, minutes and position.
    provider = match_evidence.copy()
    if "season" in provider:
        provider = provider.loc[provider["season"].astype("string").eq(season)].copy()
    else:
        provider = pd.DataFrame()
    key_pair = next((
        (left, right) for left, right in (
            ("fixture", "fpl_fixture_id"), ("match_id", "match_id"),
            ("game_id", "match_id"),
        ) if left in eligible and right in provider and "player_id" in provider
    ), None)
    if key_pair is not None:
        left, right = key_pair
        provider["player_id"] = provider["player_id"].astype("string")
        if left == "fixture":
            eligible["_fixture_key"] = pd.to_numeric(eligible[left], errors="coerce")
            provider["_fixture_key"] = pd.to_numeric(provider[right], errors="coerce")
        else:
            eligible["_fixture_key"] = eligible[left].astype("string")
            provider["_fixture_key"] = provider[right].astype("string")
        keys = ["player_id", "_fixture_key"]
        provider = provider.dropna(subset=keys)
        # Ambiguous rows cannot safely establish provider coverage.
        provider = provider.loc[~provider.duplicated(keys, keep=False)]
        fields = sorted(metrics.intersection(provider.columns) - {"red_cards"})
        aligned = eligible[keys].merge(
            provider[keys + fields], on=keys, how="left", validate="many_to_one"
        )
    else:
        aligned = pd.DataFrame(index=range(len(eligible)))
    for metric in metrics:
        if metric not in aligned:
            aligned[metric] = float("nan")
    aligned["player_id"] = eligible["player_id"].to_numpy()
    aligned["minutes"] = eligible["minutes"].to_numpy()
    aligned["fpl_position"] = eligible["fpl_pos"].to_numpy()
    aligned["season"] = season
    date_column = next((name for name in ("kickoff_time", "kickoff_utc", "date_played") if name in eligible), None)
    if date_column is None:
        return pd.DataFrame(), pd.DataFrame()
    kickoff = pd.to_datetime(eligible[date_column], utc=True, errors="coerce")
    if kickoff.isna().any():
        return pd.DataFrame(), pd.DataFrame()
    aligned["kickoff_utc"] = kickoff.to_numpy()
    # Include the complete official appearance set, even if provider data lags.
    aligned["snapshot_date"] = kickoff.max() + pd.Timedelta(nanoseconds=1)
    aligned["red_cards"] = 0
    if "defensive_contribution" in eligible:
        actions = pd.to_numeric(eligible["defensive_contribution"], errors="coerce")
        threshold = eligible["fpl_pos"].eq("DEF").map({True: 10, False: 12})
        aligned["defcon_hit"] = actions.ge(threshold).where(actions.notna()).to_numpy()
    scores, evidence = current_season_signal_profiles(aligned, season)
    for index, row in scores.iterrows():
        if row["archetype_id"] not in COMPONENT_CATALOGUE:
            continue  # Goalkeepers currently receive only the FPL Output card.
        definition = config.values["components"][row["archetype_id"]]
        weights = definition.get("weights") or definition["weights_by_position"][row["fpl_position"]]
        player_rows = aligned.loc[aligned["player_id"].eq(row["player_id"])]
        available = player_rows[list(weights)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        scores.loc[index, "covered_appearances"] = int(available.sum())
        if pd.isna(row["score_0_100"]):
            scores.loc[index, "confidence_band"] = None
            scores.loc[index, "status"] = "Provider evidence incomplete"
    return scores, evidence


PROFILE_ARCHETYPE_CARDS = (
    ("GOAL_THREAT", "goal_threat", "Goal Scoring"),
    ("CREATOR", "assist_potential", "Assist Potential"),
    ("DEFENSIVE_ENGINE", "defensive_contribution", "Defensive Contribution"),
)

COMPONENT_CATALOGUE: dict[str, dict[str, dict[str, str]]] = {
    "GOAL_THREAT": {
        "npxg": {
            "label": "Non-penalty xG / 90",
            "unit": "per_90",
            "provider": "Understat",
        },
        "shots_in_box": {
            "label": "Shots in box / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
        "shots_on_target": {
            "label": "Shots on target / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
        "non_penalty_goals": {
            "label": "Non-penalty goals / 90",
            "unit": "per_90",
            "provider": "Understat",
        },
    },
    "CREATOR": {
        "xa": {
            "label": "Expected assists / 90",
            "unit": "per_90",
            "provider": "Understat",
        },
        "key_passes": {
            "label": "Key passes / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
        "big_chances_created": {
            "label": "Big chances created / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
        "shot_creating_actions": {
            "label": "Shot-creating actions / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
    },
    "DEFENSIVE_ENGINE": {
        "defcon_hit": {
            "label": "DefCon hit rate",
            "unit": "ratio",
            "provider": "FPL + WhoScored",
        },
        "tackles_won": {
            "label": "Tackles won / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
        "interceptions": {
            "label": "Interceptions / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
        "clearances": {
            "label": "Clearances / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
        "blocks": {
            "label": "Blocks / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
        "recoveries": {
            "label": "Recoveries / 90",
            "unit": "per_90",
            "provider": "WhoScored",
        },
    },
}

COMPONENT_DEFINITIONS = {
    "npxg": "Expected goals excluding penalties.",
    "shots_in_box": "Shots attempted from inside the penalty area.",
    "shots_on_target": "Attempts that would score without a save or block on the goal line.",
    "non_penalty_goals": "Goals scored excluding penalties.",
    "xa": "Expected assists from the scoring probability of chances created.",
    "key_passes": "Passes that directly create a shot.",
    "big_chances_created": "High-quality scoring chances created for teammates.",
    "shot_creating_actions": "Attacking actions directly involved in creating a shot.",
    "defcon_hit": "Share of eligible matches reaching the FPL DefCon action threshold.",
    "tackles_won": "Tackles where the player's team gained possession.",
    "interceptions": "Opponent passes cut out by the player.",
    "clearances": "Defensive actions that remove the ball from danger.",
    "blocks": "Opponent shots or passes blocked by the player.",
    "recoveries": "Loose balls recovered for the player's team.",
}

FPL_COMPONENT_DEFINITIONS = {
    "total_points": "All official FPL points in the selected season.",
    "points_per_appearance": "FPL points per appearance of at least 30 minutes.",
    "points_per_90": "FPL points scaled to 90 minutes across eligible appearances.",
    "return_rate": "Eligible appearances with a goal, assist, or position-relevant defensive return.",
    "haul_rate": "Eligible appearances producing at least 10 FPL points.",
    "blank_rate": "Eligible appearances without a goal, assist, or position-relevant defensive return.",
    "bonus_points": "Official FPL bonus points in the selected season.",
}


def _number(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    if not isinstance(value, str):
        try:
            if bool(pd.isna(value)):
                return {}
        except (TypeError, ValueError):
            return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def interpretation_band(percentile: object) -> str:
    value = _number(percentile)
    if value is None:
        return "Unavailable"
    if value < 20:
        return "Very low"
    if value < 40:
        return "Low"
    if value < 60:
        return "Average"
    if value < 80:
        return "Strong"
    return "Elite"


def _dominant_evidence_row(evidence: pd.DataFrame) -> pd.Series | None:
    if evidence.empty:
        return None
    weights = _mapping(evidence.iloc[0].get("applied_window_weights"))
    priority = {"previous_season": 0, "earlier_current": 1, "recent": 2}
    if weights:
        window = max(
            weights,
            key=lambda name: (float(weights.get(name) or 0), priority.get(name, -1)),
        )
        matching = evidence.loc[evidence["evidence_window"].eq(window)]
        if not matching.empty:
            return matching.iloc[-1]
    ranked = evidence.assign(
        _window_priority=evidence.get(
            "evidence_window", pd.Series("", index=evidence.index)
        ).map(priority).fillna(-1)
    ).sort_values("_window_priority")
    return ranked.iloc[-1]


def _peer_component_percentiles(
    archetypes: pd.DataFrame,
    component_evidence: pd.DataFrame,
    *,
    archetype_id: str,
    evidence_window: str,
    fpl_position: str,
    player_id: str,
) -> dict[str, float]:
    required_archetypes = {"player_id", "fpl_position"}
    required_evidence = {
        "player_id",
        "archetype_id",
        "evidence_window",
        "position_z_scores",
    }
    if not required_archetypes.issubset(archetypes) or not required_evidence.issubset(
        component_evidence
    ):
        return {}
    peer_rows = component_evidence.loc[
        component_evidence["archetype_id"].eq(archetype_id)
        & component_evidence["evidence_window"].eq(evidence_window)
    ].copy()
    peer_rows["player_id"] = peer_rows["player_id"].astype("string")
    if "fpl_position" not in peer_rows:
        positions = (
            archetypes[["player_id", "fpl_position"]]
            .dropna()
            .drop_duplicates("player_id", keep="last")
        )
        positions["player_id"] = positions["player_id"].astype("string")
        peer_rows = peer_rows.merge(positions, on="player_id", how="inner")
    peer_rows = peer_rows.loc[
        peer_rows["fpl_position"].astype("string").str.upper().eq(fpl_position)
    ]
    values: dict[str, dict[str, float]] = {}
    for row in peer_rows.to_dict("records"):
        raw_values = _mapping(row.get("metric_values"))
        for component_id, value in _mapping(row.get("position_z_scores")).items():
            if _number(raw_values.get(component_id)) is None:
                continue
            number = _number(value)
            if number is not None:
                values.setdefault(component_id, {})[str(row["player_id"])] = number
    percentiles = {}
    for component_id, player_values in values.items():
        series = pd.Series(player_values, dtype="float64")
        ranks = series.rank(method="average", pct=True).mul(100)
        if str(player_id) in ranks:
            percentiles[component_id] = round(float(ranks[str(player_id)]), 1)
    return percentiles


def _component_rows(
    archetype_id: str,
    evidence: pd.DataFrame,
    percentiles: Mapping[str, float],
) -> list[dict[str, Any]]:
    selected = _dominant_evidence_row(evidence)
    if selected is None:
        return []
    raw_values = _mapping(selected.get("metric_values"))
    weights = _mapping(selected.get("metric_weights"))
    raw_missing = selected.get("missing_core_fields")
    missing = (
        {str(value) for value in raw_missing}
        if isinstance(raw_missing, list)
        else set(_mapping(raw_missing))
    )

    rows = []
    for component_id, metadata in COMPONENT_CATALOGUE[archetype_id].items():
        raw_value = _number(raw_values.get(component_id))
        percentile = _number(percentiles.get(component_id))
        status = "available"
        if component_id in missing:
            status = "Core field missing"
        elif raw_value is None:
            status = "Provider data unavailable"
        elif percentile is None:
            status = "Peer percentile unavailable"
        display_value = None
        if raw_value is not None:
            display_value = (
                f"{raw_value * 100:.1f}%"
                if metadata["unit"] == "ratio"
                else f"{raw_value:.2f}"
            )
        rows.append(
            {
                "id": component_id,
                "label": metadata["label"],
                "raw_value": raw_value,
                "display_value": display_value,
                "unit": metadata["unit"],
                "percentile": percentile,
                "interpretation": interpretation_band(percentile),
                "weight": _number(weights.get(component_id)),
                "status": status,
                "provider": metadata["provider"],
                "help": (
                    f"{COMPONENT_DEFINITIONS[component_id]} Denominator: "
                    f"{'eligible appearances' if metadata['unit'] == 'ratio' else '90 minutes'}. "
                    "Peer group: players in the same FPL position and evidence window. "
                    "Higher is better. "
                    + (
                        "Model weight unavailable. "
                        if _number(weights.get(component_id)) is None
                        else f"Persisted model weight: {_number(weights.get(component_id)) * 100:.0f}%. "
                    )
                    + f"Provider: {metadata['provider']}."
                ),
            }
        )
    return rows


def _profile_card(
    archetypes: pd.DataFrame,
    component_evidence: pd.DataFrame,
    *,
    archetype_id: str,
    card_id: str,
    title: str,
    player_id: str,
    fpl_position: str,
) -> dict[str, Any]:
    rows = (
        archetypes.loc[
            archetypes["archetype_id"].eq(archetype_id)
            & archetypes.get(
                "player_id", pd.Series(player_id, index=archetypes.index)
            ).astype("string").eq(str(player_id))
        ]
        if "archetype_id" in archetypes
        else pd.DataFrame()
    )
    if rows.empty:
        return {
            "id": card_id,
            "title": title,
            "headline_value": None,
            "headline_type": "position_percentile",
            "scale_label": "Same-position profile percentile",
            "confidence_band": None,
            "trend": None,
            "evidence_minutes": None,
            "eligible_appearances": None,
            "evidence_window": None,
            "status": "Profile not published",
            "components": [],
        }
    headline = rows.iloc[-1]
    player_evidence = (
        component_evidence.loc[
            component_evidence["archetype_id"].eq(archetype_id)
            & component_evidence.get(
                "player_id", pd.Series(player_id, index=component_evidence.index)
            ).astype("string").eq(str(player_id))
        ]
        if "archetype_id" in component_evidence
        else pd.DataFrame()
    )
    dominant = _dominant_evidence_row(player_evidence)
    evidence_window = None if dominant is None else dominant.get("evidence_window")
    percentiles = (
        {}
        if evidence_window is None
        else _peer_component_percentiles(
            archetypes,
            component_evidence,
            archetype_id=archetype_id,
            evidence_window=str(evidence_window),
            fpl_position=fpl_position,
            player_id=player_id,
        )
    )
    return {
        "id": card_id,
        "title": title,
        "headline_value": _number(headline.get("score_0_100")),
        "headline_type": "position_percentile",
        "scale_label": "Same-position profile percentile",
        "confidence_band": headline.get("confidence_band"),
        "trend": headline.get("trend"),
        "evidence_minutes": _number(headline.get("evidence_minutes")),
        "eligible_appearances": _number(headline.get("eligible_appearances")),
        "covered_appearances": _number(headline.get("covered_appearances")),
        "evidence_window": evidence_window,
        "status": headline.get("status"),
        "components": _component_rows(
            archetype_id, player_evidence, percentiles
        ),
    }


def _select_player_matches(
    gameweeks: pd.DataFrame, player_id: str, player_name: str
) -> pd.DataFrame:
    if gameweeks.empty:
        return pd.DataFrame()
    if "player_id" in gameweeks:
        selected = gameweeks.loc[
            gameweeks["player_id"].astype("string").eq(str(player_id))
        ].copy()
        if not selected.empty:
            return selected
    if "name" in gameweeks:
        return gameweeks.loc[
            gameweeks["name"].astype("string").eq(str(player_name))
        ].copy()
    return pd.DataFrame()


def _fpl_player_summaries(gameweeks: pd.DataFrame) -> pd.DataFrame:
    required = {"player_id", "fpl_pos", "minutes", "total_points"}
    if gameweeks.empty or not required.issubset(gameweeks):
        return pd.DataFrame()
    work = _eligible_fpl_matches(gameweeks)
    work["total_points"] = pd.to_numeric(work["total_points"], errors="coerce")
    work["fpl_pos"] = work["fpl_pos"].astype("string").str.upper().replace("GK", "GKP")
    work = work.loc[work["minutes"].ge(30)].copy()
    if work.empty:
        return pd.DataFrame()
    def numeric_column(column: str) -> pd.Series:
        return pd.to_numeric(
            work.get(column, pd.Series(0, index=work.index)), errors="coerce"
        ).fillna(0)

    goals = numeric_column("goals_scored")
    assists = numeric_column("assists")
    clean_sheets = numeric_column("clean_sheets")
    saves = numeric_column("saves")
    work["return_event"] = goals.gt(0) | assists.gt(0)
    work["return_event"] |= work["fpl_pos"].isin(["DEF", "GKP"]) & clean_sheets.gt(0)
    work["return_event"] |= work["fpl_pos"].eq("GKP") & saves.ge(3)
    work["haul"] = work["total_points"].ge(10)
    work["blank"] = ~work["return_event"]
    work["bonus"] = numeric_column("bonus")
    summaries = (
        work.groupby("player_id", as_index=False)
        .agg(
            fpl_pos=("fpl_pos", "last"),
            evidence_minutes=("minutes", "sum"),
            eligible_appearances=("player_id", "size"),
            eligible_points=("total_points", "sum"),
            return_rate=("return_event", "mean"),
            haul_rate=("haul", "mean"),
            blank_rate=("blank", "mean"),
            bonus_points=("bonus", "sum"),
        )
    )
    summaries["points_per_appearance"] = summaries["eligible_points"].div(
        summaries["eligible_appearances"]
    )
    summaries["points_per_90"] = summaries["eligible_points"].div(
        summaries["evidence_minutes"]
    ).mul(90)
    for metric in (
        "points_per_appearance",
        "points_per_90",
        "return_rate",
        "haul_rate",
        "bonus_points",
    ):
        summaries[f"{metric}_percentile"] = summaries[metric].groupby(
            summaries["fpl_pos"]
        ).rank(method="average", pct=True).mul(100)
    summaries["blank_rate_percentile"] = summaries["blank_rate"].groupby(
        summaries["fpl_pos"]
    ).rank(method="average", pct=True, ascending=False).mul(100)
    return summaries


def _fpl_output_card(
    selected_record: pd.Series,
    gameweeks: pd.DataFrame,
    *,
    player_id: str,
    player_name: str,
) -> dict[str, Any]:
    matches = _select_player_matches(gameweeks, player_id, player_name)
    summaries = _fpl_player_summaries(gameweeks)
    summary_rows = (
        summaries.loc[
            summaries["player_id"].astype("string").eq(str(player_id))
        ]
        if "player_id" in summaries
        else pd.DataFrame()
    )
    summary = None if summary_rows.empty else summary_rows.iloc[-1]
    component_specs = (
        ("total_points", "Total points", "points", "Points", "Points Position Percentile"),
        ("points_per_appearance", "Points / eligible appearance", "number", None, None),
        ("points_per_90", "Points / 90", "number", None, None),
        ("return_rate", "Return rate", "ratio", None, None),
        ("haul_rate", "Haul rate", "ratio", None, None),
        ("blank_rate", "Blank rate", "ratio", None, None),
        ("bonus_points", "Bonus points", "points", "Bonus", "Bonus Position Percentile"),
    )
    components = []
    for metric, label, unit, record_value, record_percentile in component_specs:
        value = (
            _number(selected_record.get(record_value))
            if record_value is not None
            else None if summary is None else _number(summary.get(metric))
        )
        percentile = (
            _number(selected_record.get(record_percentile))
            if record_percentile is not None
            else None if summary is None else _number(summary.get(f"{metric}_percentile"))
        )
        display_value = None
        if value is not None:
            if unit == "ratio":
                display_value = f"{value * 100:.1f}%"
            elif unit == "number":
                display_value = f"{value:.2f}"
            else:
                display_value = f"{value:.0f}"
        components.append(
            {
                "id": metric,
                "label": label,
                "raw_value": value,
                "display_value": display_value,
                "unit": unit,
                "percentile": percentile,
                "interpretation": interpretation_band(percentile),
                "weight": None,
                "status": "available" if value is not None else "Insufficient appearances",
                "provider": "FPL",
                "help": (
                    f"{FPL_COMPONENT_DEFINITIONS[metric]} "
                    "Peer group: current-season players in the same FPL position. "
                    f"{'Lower' if metric == 'blank_rate' else 'Higher'} is better. "
                    "Provider: FPL."
                ),
            }
        )
    headline = None if summary is None else _number(
        summary.get("points_per_appearance_percentile")
    )
    appearances = None if summary is None else _number(summary.get("eligible_appearances"))
    return {
        "id": "fpl_output",
        "title": "FPL Points Output",
        "headline_value": headline,
        "headline_type": "position_percentile",
        "scale_label": "Current-season points / eligible appearance percentile",
        "confidence_band": None,
        "trend": None,
        "evidence_minutes": (
            None if summary is None else _number(summary.get("evidence_minutes"))
        ),
        "eligible_appearances": appearances,
        "evidence_window": "current_season",
        "status": "Current season" if summary is not None else "Insufficient appearances",
        "components": components,
        "match_rows": len(matches),
    }


def build_player_signal_cards(
    archetypes: pd.DataFrame,
    component_evidence: pd.DataFrame,
    selected_record: pd.Series,
    gameweeks: pd.DataFrame,
    *,
    player_id: str,
    player_name: str,
    fpl_position: str,
    current_season_only: bool = False,
) -> list[dict[str, Any]]:
    """Build the four Overview signal cards from persisted and official data."""
    cards = []
    position = str(fpl_position).strip().upper()
    if position == "GK":
        position = "GKP"
    if position != "GKP":
        cards.extend(
            _profile_card(
                archetypes,
                component_evidence,
                archetype_id=archetype_id,
                card_id=card_id,
                title=title,
                player_id=player_id,
                fpl_position=position,
            )
            for archetype_id, card_id, title in PROFILE_ARCHETYPE_CARDS
        )
    cards.append(
        _fpl_output_card(
            selected_record,
            gameweeks,
            player_id=player_id,
            player_name=player_name,
        )
    )
    if current_season_only:
        summaries = _fpl_player_summaries(gameweeks)
        selected_summary = (
            summaries.loc[summaries["player_id"].astype("string").eq(str(player_id))]
            if "player_id" in summaries else pd.DataFrame()
        )
        for card in cards:
            if card["id"] == "fpl_output":
                continue
            card["scale_label"] = "Current-season same-position percentile"
            card["evidence_window"] = "current_season"
            if not selected_summary.empty:
                official = selected_summary.iloc[-1]
                card["eligible_appearances"] = int(official["eligible_appearances"])
                card["evidence_minutes"] = float(official["evidence_minutes"])
            covered = card.get("covered_appearances")
            card["covered_appearances"] = 0 if covered is None else covered
            if card["status"] == "Profile not published":
                card["status"] = "Current-season evidence unavailable"
    return cards
