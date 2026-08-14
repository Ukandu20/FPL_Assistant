"""Production-style profiles derived from season-level player statistics.

The profile model deliberately separates *style* (archetype) from *quality*
(tier).  Per-90 metrics are shrunk toward a minutes-weighted positional mean
before positional standardisation, so small samples do not dominate the
rankings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


MODEL_VERSION = "player-production-profile-v1"
OUTFIELD_POSITIONS = {"DEF", "MID", "FWD"}
GOALKEEPER_POSITIONS = {"GK", "GKP"}

OUTFIELD_ARCHETYPES: dict[str, dict[frozenset[str], str]] = {
    "FWD": {
        frozenset({"G"}): "Finisher",
        frozenset({"C"}): "Creative Forward",
        frozenset({"D"}): "Pressing Forward",
        frozenset({"G", "C"}): "Complete Forward",
        frozenset({"G", "D"}): "Pressing Finisher",
        frozenset({"C", "D"}): "Pressing Link Forward",
        frozenset({"G", "C", "D"}): "Complete Two-Way Forward",
    },
    "MID": {
        frozenset({"G"}): "Goal-Scoring Midfielder",
        frozenset({"C"}): "Playmaker",
        frozenset({"D"}): "Ball-Winning Midfielder",
        frozenset({"G", "C"}): "Attacking Playmaker",
        frozenset({"G", "D"}): "Box-to-Box Midfielder",
        frozenset({"C", "D"}): "Deep-Lying Playmaker",
        frozenset({"G", "C", "D"}): "Complete Midfielder",
    },
    "DEF": {
        frozenset({"G"}): "Goal-Threat Defender",
        frozenset({"C"}): "Creative Defender",
        frozenset({"D"}): "Defensive Stopper",
        frozenset({"G", "C"}): "Attacking Defender",
        frozenset({"G", "D"}): "Two-Way Defender",
        frozenset({"C", "D"}): "Defensive Creator",
        frozenset({"G", "C", "D"}): "Complete Defender",
    },
}

GOALKEEPER_ARCHETYPES: dict[frozenset[str], str] = {
    frozenset({"S"}): "Shot Stopper",
    frozenset({"W"}): "Sweeper Keeper",
    frozenset({"P"}): "Distributor",
    frozenset({"S", "W"}): "Proactive Shot Stopper",
    frozenset({"S", "P"}): "Ball-Playing Shot Stopper",
    frozenset({"W", "P"}): "Ball-Playing Sweeper",
    frozenset({"S", "W", "P"}): "Complete Goalkeeper",
}

STRENGTH_NAMES = {
    "G": "Goal Threat",
    "C": "Creativity",
    "D": "Defensive Activity",
    "S": "Shot Stopping",
    "W": "Sweeping",
    "P": "Distribution",
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _normalise_positions(series: pd.Series) -> pd.Series:
    positions = series.astype("string").str.strip().str.upper()
    return positions.replace({"GK": "GKP"})


def _position_zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    standard_deviation = numeric.std(ddof=0)
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean()) / standard_deviation


def _scale_0_100(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    spread = numeric.max() - numeric.min()
    if pd.isna(spread) or spread == 0:
        return pd.Series(50.0, index=series.index)
    return (numeric - numeric.min()) / spread * 100


def _merge_provider(
    base: pd.DataFrame,
    provider: pd.DataFrame | None,
    columns: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    result = base.copy()
    marker = f"_{prefix}_matched"
    if provider is None or provider.empty or "player_id" not in provider:
        result[marker] = False
        for column in columns:
            result[f"{prefix}_{column}"] = np.nan
        return result

    available = [column for column in columns if column in provider]
    lookup = provider[["player_id", *available]].copy()
    lookup["player_id"] = lookup["player_id"].astype("string")
    lookup = lookup.dropna(subset=["player_id"]).drop_duplicates(
        "player_id", keep="last"
    )
    lookup[marker] = True
    lookup = lookup.rename(
        columns={column: f"{prefix}_{column}" for column in available}
    )
    result = result.merge(lookup, on="player_id", how="left", validate="one_to_one")
    result[marker] = result[marker].fillna(False).astype(bool)
    for column in columns:
        target = f"{prefix}_{column}"
        if target not in result:
            result[target] = np.nan
    return result


def _add_shrunk_metric(
    frame: pd.DataFrame,
    source: str,
    output: str,
    *,
    eligible: pd.Series,
    prior_minutes: float,
    per_90: bool = True,
    invert: bool = False,
) -> None:
    values = _numeric(frame, source)
    reliability = frame["reliability"]
    minutes = frame["minutes"]

    if per_90:
        observed = values.div(frame["90s"].replace(0, np.nan))
        numerator = values.where(eligible).groupby(frame["fpl_pos"]).transform("sum")
        denominator = (
            frame["90s"].where(eligible).groupby(frame["fpl_pos"]).transform("sum")
        )
        position_mean = numerator.div(denominator.replace(0, np.nan))
    else:
        observed = values
        weighted_values = values.mul(minutes).where(eligible)
        numerator = weighted_values.groupby(frame["fpl_pos"]).transform("sum")
        denominator = minutes.where(eligible).groupby(frame["fpl_pos"]).transform("sum")
        position_mean = numerator.div(denominator.replace(0, np.nan))

    shrunk = reliability.mul(observed).add((1 - reliability).mul(position_mean))
    frame[f"{output}_observed"] = observed
    frame[f"{output}_shrunk"] = shrunk
    zscore = (
        shrunk.where(eligible)
        .groupby(frame["fpl_pos"])
        .transform(_position_zscore)
    )
    frame[f"{output}_z"] = -zscore if invert else zscore


def _weighted_composite(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
    output: str,
    eligible: pd.Series,
) -> None:
    components = pd.DataFrame(
        {column: _numeric(frame, column) for column in weights}, index=frame.index
    )
    weight_series = pd.Series(weights, dtype="float64")
    available_weight = components.notna().mul(weight_series, axis=1).sum(axis=1)
    weighted_sum = components.mul(weight_series, axis=1).sum(axis=1, min_count=1)
    raw = weighted_sum.div(available_weight.replace(0, np.nan)).where(eligible)
    frame[f"{output}_raw"] = raw
    frame[f"{output}_score"] = (
        raw.groupby(frame["fpl_pos"]).transform(_scale_0_100).round(2)
    )
    frame[f"{output}_percentile"] = (
        raw.groupby(frame["fpl_pos"])
        .rank(method="average", pct=True)
        .mul(100)
        .round(2)
    )
    frame[f"{output}_rank"] = (
        raw.groupby(frame["fpl_pos"])
        .rank(method="min", ascending=False)
        .astype("Int64")
    )


def _profile_tier(primary_score: float) -> str:
    if primary_score >= 90:
        return "Elite"
    if primary_score >= 75:
        return "High"
    if primary_score >= 50:
        return "Average"
    return "Low"


def _assign_style(
    row: pd.Series,
    dimensions: Mapping[str, str],
    archetypes: Mapping[frozenset[str], str],
) -> tuple[str, str, str, str]:
    scores = {
        code: pd.to_numeric(row.get(column), errors="coerce")
        for code, column in dimensions.items()
    }
    if any(pd.isna(score) for score in scores.values()):
        return "Insufficient Data", "Unrated", "", ""

    ordered = sorted(scores, key=scores.get, reverse=True)
    primary, secondary = ordered[0], ordered[1]
    primary_score = float(scores[primary])
    spread = max(scores.values()) - min(scores.values())

    if primary_score < 40:
        archetype = "Low-Production"
    elif spread <= 10:
        archetype = "Balanced Profile"
    else:
        traits = {primary}
        for code, score in scores.items():
            if code != primary and score >= 60 and primary_score - score <= 15:
                traits.add(code)
        archetype = archetypes[frozenset(traits)]

    secondary_strength = (
        STRENGTH_NAMES[secondary]
        if scores[secondary] >= 60 and primary_score - scores[secondary] <= 15
        else ""
    )
    return (
        archetype,
        _profile_tier(primary_score),
        STRENGTH_NAMES[primary],
        secondary_strength,
    )


def _characteristics(row: pd.Series, candidates: Sequence[tuple[str, str]]) -> str:
    ranked = []
    for column, label in candidates:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value) and value >= 0.5:
            ranked.append((float(value), label))
    return "; ".join(label for _, label in sorted(ranked, reverse=True)[:3])


def build_player_profiles(
    fpl_players: pd.DataFrame,
    *,
    shooting: pd.DataFrame | None = None,
    passing: pd.DataFrame | None = None,
    defense: pd.DataFrame | None = None,
    keepers: pd.DataFrame | None = None,
    season: str | None = None,
    minimum_minutes: int = 450,
    established_minutes: int = 900,
    prior_minutes: int = 900,
) -> pd.DataFrame:
    """Return one production-style profile per FPL player-season."""
    required = {"player_id", "name", "fpl_pos", "minutes"}
    missing = required - set(fpl_players.columns)
    if missing:
        raise ValueError(f"FPL profile input is missing columns: {sorted(missing)}")

    players = fpl_players.copy()
    players["player_id"] = players["player_id"].astype("string")
    players = players.dropna(subset=["player_id"]).copy()
    if players["player_id"].duplicated().any():
        duplicates = players.loc[players["player_id"].duplicated(), "player_id"].tolist()
        raise ValueError(f"Duplicate player IDs in FPL profile input: {duplicates[:5]}")

    players["fpl_pos"] = _normalise_positions(players["fpl_pos"])
    players["minutes"] = _numeric(players, "minutes").fillna(0)
    players["90s"] = players["minutes"] / 90
    players["reliability"] = players["minutes"].div(
        players["minutes"] + prior_minutes
    )
    players["season"] = str(season) if season is not None else players.get("season", "")
    players["model_version"] = MODEL_VERSION
    players["prior_minutes"] = prior_minutes

    players = _merge_provider(
        players,
        shooting,
        ["shots_total", "shots_on_target"],
        "shooting",
    )
    players = _merge_provider(
        players,
        passing,
        [
            "key_passes",
            "big_chances_created",
            "passes_attempted",
            "pass_completion_pct",
            "progressive_passes",
        ],
        "passing",
    )
    players = _merge_provider(
        players,
        defense,
        ["tackles_won", "interceptions", "clearances", "blocks", "recoveries"],
        "defense",
    )
    players = _merge_provider(
        players,
        keepers,
        [
            "saves",
            "save_pct",
            "goals_against",
            "keeper_sweeper_actions",
            "smothers",
        ],
        "keepers",
    )

    outfield = players["fpl_pos"].isin(OUTFIELD_POSITIONS)
    goalkeeper = players["fpl_pos"].eq("GKP")
    outfield_coverage = (
        players["_shooting_matched"]
        & players["_passing_matched"]
        & players["_defense_matched"]
    )
    goalkeeper_coverage = players["_keepers_matched"] & players["_passing_matched"]
    covered = (outfield & outfield_coverage) | (goalkeeper & goalkeeper_coverage)
    eligible = covered & players["minutes"].ge(minimum_minutes)

    players["profile_status"] = "Data unavailable"
    players.loc[covered & players["minutes"].lt(minimum_minutes), "profile_status"] = (
        "Insufficient data"
    )
    players.loc[eligible, "profile_status"] = "Provisional"
    players.loc[eligible & players["minutes"].ge(established_minutes), "profile_status"] = (
        "Established"
    )

    outfield_eligible = eligible & outfield
    for source, output, invert in [
        ("xg", "xg_p90", False),
        ("goals_scored", "goals_scored_p90", False),
        ("shooting_shots_total", "shots_total_p90", False),
        ("shooting_shots_on_target", "shots_on_target_p90", False),
        ("xa", "xa_p90", False),
        ("assists", "assists_p90", False),
        ("passing_key_passes", "key_passes_p90", False),
        ("passing_big_chances_created", "big_chances_created_p90", False),
        ("defense_tackles_won", "tackles_won_p90", False),
        ("defense_interceptions", "interceptions_p90", False),
        ("defense_clearances", "clearances_p90", False),
        ("defense_blocks", "blocks_p90", False),
        ("defense_recoveries", "recoveries_p90", False),
    ]:
        _add_shrunk_metric(
            players,
            source,
            output,
            eligible=outfield_eligible,
            prior_minutes=prior_minutes,
            invert=invert,
        )

    players["gls_performance_shrunk"] = (
        players["goals_scored_p90_shrunk"] - players["xg_p90_shrunk"]
    )
    players["gls_performance_z"] = (
        players["gls_performance_shrunk"]
        .where(outfield_eligible)
        .groupby(players["fpl_pos"])
        .transform(_position_zscore)
    )

    _weighted_composite(
        players,
        {
            "xg_p90_z": 0.55,
            "shots_on_target_p90_z": 0.20,
            "shots_total_p90_z": 0.15,
            "gls_performance_z": 0.10,
        },
        "goal_threat",
        outfield_eligible,
    )
    _weighted_composite(
        players,
        {
            "key_passes_p90_z": 0.25,
            "big_chances_created_p90_z": 0.20,
            "assists_p90_z": 0.05,
            "xa_p90_z": 0.50,
        },
        "creativity",
        outfield_eligible,
    )
    _weighted_composite(
        players,
        {
            "tackles_won_p90_z": 0.20,
            "interceptions_p90_z": 0.25,
            "clearances_p90_z": 0.25,
            "blocks_p90_z": 0.15,
            "recoveries_p90_z": 0.15,
        },
        "defensive_threat",
        outfield_eligible,
    )

    goalkeeper_eligible = eligible & goalkeeper
    for source, output, per_90, invert in [
        ("keepers_saves", "saves_p90", True, False),
        ("keepers_save_pct", "save_pct", False, False),
        ("keepers_goals_against", "goals_against_p90", True, True),
        ("keepers_keeper_sweeper_actions", "sweeper_actions_p90", True, False),
        ("keepers_smothers", "smothers_p90", True, False),
        ("passing_progressive_passes", "progressive_passes_p90", True, False),
        ("passing_passes_attempted", "passes_attempted_p90", True, False),
        ("passing_pass_completion_pct", "pass_completion_pct", False, False),
    ]:
        _add_shrunk_metric(
            players,
            source,
            output,
            eligible=goalkeeper_eligible,
            prior_minutes=prior_minutes,
            per_90=per_90,
            invert=invert,
        )

    _weighted_composite(
        players,
        {"save_pct_z": 0.45, "saves_p90_z": 0.30, "goals_against_p90_z": 0.25},
        "shot_stopping",
        goalkeeper_eligible,
    )
    _weighted_composite(
        players,
        {"sweeper_actions_p90_z": 0.70, "smothers_p90_z": 0.30},
        "sweeping",
        goalkeeper_eligible,
    )
    _weighted_composite(
        players,
        {
            "progressive_passes_p90_z": 0.45,
            "passes_attempted_p90_z": 0.25,
            "pass_completion_pct_z": 0.30,
        },
        "distribution",
        goalkeeper_eligible,
    )

    players["production_archetype"] = "Insufficient Data"
    players["production_tier"] = "Unrated"
    players["primary_strength"] = ""
    players["secondary_strength"] = ""
    players["playing_characteristics"] = ""

    for index in players.index[outfield_eligible]:
        position = str(players.at[index, "fpl_pos"])
        archetype, tier, primary, secondary = _assign_style(
            players.loc[index],
            {
                "G": "goal_threat_percentile",
                "C": "creativity_percentile",
                "D": "defensive_threat_percentile",
            },
            OUTFIELD_ARCHETYPES[position],
        )
        players.loc[index, [
            "production_archetype",
            "production_tier",
            "primary_strength",
            "secondary_strength",
        ]] = [archetype, tier, primary, secondary]
        players.at[index, "playing_characteristics"] = _characteristics(
            players.loc[index],
            [
                ("xg_p90_z", "Gets high-quality chances"),
                ("shots_on_target_p90_z", "Tests the goalkeeper frequently"),
                ("key_passes_p90_z", "Creates chances frequently"),
                ("xa_p90_z", "Produces high-value final passes"),
                ("tackles_won_p90_z", "Wins tackles frequently"),
                ("interceptions_p90_z", "Reads passing lanes well"),
                ("clearances_p90_z", "Provides strong box defending"),
                ("recoveries_p90_z", "Recovers possession frequently"),
            ],
        )

    for index in players.index[goalkeeper_eligible]:
        archetype, tier, primary, secondary = _assign_style(
            players.loc[index],
            {
                "S": "shot_stopping_percentile",
                "W": "sweeping_percentile",
                "P": "distribution_percentile",
            },
            GOALKEEPER_ARCHETYPES,
        )
        players.loc[index, [
            "production_archetype",
            "production_tier",
            "primary_strength",
            "secondary_strength",
        ]] = [archetype, tier, primary, secondary]
        players.at[index, "playing_characteristics"] = _characteristics(
            players.loc[index],
            [
                ("save_pct_z", "Strong shot-stopping rate"),
                ("saves_p90_z", "Handles a high shot volume"),
                ("sweeper_actions_p90_z", "Active outside the goalmouth"),
                ("smothers_p90_z", "Closes down danger proactively"),
                ("progressive_passes_p90_z", "Advances possession through passing"),
                ("pass_completion_pct_z", "Retains possession reliably"),
            ],
        )

    players.loc[players["profile_status"].eq("Data unavailable"), "production_archetype"] = (
        "Data Unavailable"
    )

    public_columns = [
        "season",
        "player_id",
        "name",
        "team",
        "fpl_pos",
        "minutes",
        "90s",
        "reliability",
        "profile_status",
        "production_archetype",
        "production_tier",
        "primary_strength",
        "secondary_strength",
        "playing_characteristics",
        "goal_threat_raw",
        "goal_threat_score",
        "goal_threat_percentile",
        "goal_threat_rank",
        "creativity_raw",
        "creativity_score",
        "creativity_percentile",
        "creativity_rank",
        "defensive_threat_raw",
        "defensive_threat_score",
        "defensive_threat_percentile",
        "defensive_threat_rank",
        "shot_stopping_raw",
        "shot_stopping_score",
        "shot_stopping_percentile",
        "shot_stopping_rank",
        "sweeping_raw",
        "sweeping_score",
        "sweeping_percentile",
        "sweeping_rank",
        "distribution_raw",
        "distribution_score",
        "distribution_percentile",
        "distribution_rank",
        "model_version",
        "prior_minutes",
    ]
    return players.reindex(columns=public_columns).sort_values(
        ["fpl_pos", "production_tier", "name"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "MODEL_VERSION",
    "build_player_profiles",
]
