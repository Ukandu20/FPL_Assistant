import importlib
import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit_shadcn_ui as ui

from apps.fpl.catalog import (
    PREDICTIONS_ROOT,
    FPL_ROOT,
    PLAYER_IMAGE_CONFIG_PATH,
    PRICE_CATEGORY_CONFIG_PATH,
    discover_leagues as catalog_discover_leagues,
    discover_seasons as catalog_discover_seasons,
    file_version,
    fpl_fixture_metadata_path,
    fpl_gameweeks_path,
    fpl_player_profiles_path,
    fpl_raw_fixtures_path,
    fpl_raw_players_path,
    fpl_raw_teams_path,
    fpl_season_path,
)
from fpl_assistant.apps.viewmodels import player_card as player_card_viewmodels


# Streamlit reruns page modules without necessarily reloading their imported
# dependencies. Reload this small, pure presentation module so newly added
# helpers cannot remain missing in a long-running development server.
player_card_viewmodels = importlib.reload(player_card_viewmodels)
comparable_players = player_card_viewmodels.comparable_players
decision_factors = player_card_viewmodels.decision_factors
forecast_summary = player_card_viewmodels.forecast_summary
latest_forecast_path = player_card_viewmodels.latest_forecast_path
prepare_player_forecast = player_card_viewmodels.prepare_player_forecast
prepare_player_fixtures = player_card_viewmodels.prepare_player_fixtures
player_photo_urls = player_card_viewmodels.player_photo_urls
player_placeholder_url = player_card_viewmodels.player_placeholder_url
profile_dimensions = player_card_viewmodels.profile_dimensions
profile_trend = player_card_viewmodels.profile_trend
recent_form_summary = player_card_viewmodels.recent_form_summary
select_profile_snapshot = player_card_viewmodels.select_profile_snapshot
team_badge_url = player_card_viewmodels.team_badge_url


st.set_page_config(page_title="Fantasy Premier League dashboard", layout="wide")

MIN_PER_90_MINUTES = 450

PLAYER_HISTORY_COLUMNS = {
    "season": "Season",
    "name": "Player",
    "team": "Team",
    "fpl_pos": "Position",
    "minutes": "Minutes",
    "total_points": "Points",
    "goals_scored": "Goals",
    "assists": "Assists",
    "goals_conceded": "Goals conceded",
    "creativity": "Creativity",
    "influence": "Influence",
    "threat": "Threat",
    "ict_index": "ICT index",
    "clean_sheets": "Clean sheets",
    "bonus": "Bonus",
    "bps": "BPS",
    "yellow_cards": "Yellow cards",
    "red_cards": "Red cards",
    "blocks": "Blocks",
    "interceptions": "Interceptions",
    "clearances": "Clearances",
    "tackles_won": "Tackles won",
    "recoveries": "Recoveries",
    "defcon": "Def Con",
    "xg": "xG",
    "xa": "xA",
    "shots_on_target_against": "Shots on target against",
    "saves": "Saves",
    "goals_against": "Goals against",
    "save_pct": "Save %",
    "penalties_faced": "Penalties faced",
    "penalties_allowed": "Penalties allowed",
    "penalties_saved": "Penalties saved",
    "penalties_missed": "Penalties missed",
    "penalty_save_pct": "Penalty save %",
    "selected_by_percent": "Selected by %",
    "now_cost": "Price",
}

RANKED_METRICS = {
    source: label
    for source, label in PLAYER_HISTORY_COLUMNS.items()
    if source not in {"season", "name", "team", "fpl_pos"}
}

PER_90_METRICS = set(RANKED_METRICS) - {
    "minutes",
    "selected_by_percent",
    "now_cost",
    "save_pct",
    "penalty_save_pct",
}

# These metrics describe adverse outcomes. A lower value therefore deserves a
# higher performance percentile.
LOWER_IS_BETTER_METRICS = {
    "goals_conceded",
    "yellow_cards",
    "red_cards",
    "shots_on_target_against",
    "goals_against",
    "penalties_allowed",
    "penalties_missed",
}

GAMEWEEK_POINT_COMPONENTS = [
    "Appearance points",
    "Goal points",
    "Assist points",
    "Bonus points",
    "Defensive contribution points",
    "Save points",
    "Card points",
]

GAMEWEEK_COMPONENT_COLORS = {
    "Appearance points": "#3B82F6",
    "Goal points": "#22C55E",
    "Assist points": "#14B8A6",
    "Bonus points": "#F59E0B",
    "Defensive contribution points": "#8B5CF6",
    "Save points": "#06B6D4",
    "Card points": "#EF4444",
}

PRICE_CATEGORY_COLORS = {
    "Premium": ("#D4AF37", "#1F2937"),
    "Top-tier": ("#8B5CF6", "#FFFFFF"),
    "Mid-tier": ("#22C55E", "#052E16"),
    "Budget": ("#06B6D4", "#083344"),
    "Fodder": ("#6B7280", "#FFFFFF"),
    "Uncategorized": ("#6B7280", "#FFFFFF"),
}

PRICE_CATEGORY_ORDER = [
    "Fodder",
    "Budget",
    "Mid-tier",
    "Top-tier",
    "Premium",
]


@st.cache_data(show_spinner=False)
def discover_leagues() -> list[str]:
    return catalog_discover_leagues()


@st.cache_data(show_spinner=False)
def discover_seasons(league: str) -> list[str]:
    return catalog_discover_seasons(league)


@st.cache_data(show_spinner=False)
def load_price_category_config(
    data_version: tuple[int, int] | None = None,
) -> dict:
    """Load the stored, frozen price-category boundaries."""
    del data_version  # Used by Streamlit as a file-change-aware cache key.
    if not PRICE_CATEGORY_CONFIG_PATH.is_file():
        return {}
    with PRICE_CATEGORY_CONFIG_PATH.open(encoding="utf-8") as config_file:
        return json.load(config_file)


@st.cache_data(show_spinner=False)
def load_player_image_config(
    data_version: tuple[int, int] | None = None,
) -> dict:
    """Load the configurable official portrait collection version."""
    del data_version
    if not PLAYER_IMAGE_CONFIG_PATH.is_file():
        return {"asset_version": "25"}
    with PLAYER_IMAGE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def format_price_category(
    now_cost: object, position: object, season: object
) -> str:
    """Classify a raw FPL price using the frozen configuration for its season."""
    price = pd.to_numeric(now_cost, errors="coerce")
    normalized_position = str(position).strip().upper()
    if normalized_position == "GK":
        normalized_position = "GKP"
    if pd.isna(price):
        return "Uncategorized"

    stored_config = load_price_category_config(
        file_version(PRICE_CATEGORY_CONFIG_PATH)
    )
    season_config = stored_config.get("seasons", {}).get(str(season))
    calibration = season_config or stored_config.get("default", {})
    position_config = calibration.get("positions", {}).get(normalized_position)
    if not position_config:
        return "Uncategorized"

    labels = position_config.get("labels", [])
    raw_edges = position_config.get("edges", [])
    edges = [
        float("inf") if str(edge).lower() == "inf" else float(edge)
        for edge in raw_edges
    ]
    if len(edges) != len(labels) + 1:
        return "Uncategorized"

    for index, label in enumerate(labels):
        lower_bound = edges[index]
        upper_bound = edges[index + 1]
        if lower_bound <= float(price) < upper_bound:
            if index == 0:
                price_range = f"< £{upper_bound / 10:.1f}m"
            elif upper_bound == float("inf"):
                price_range = f"≥ £{lower_bound / 10:.1f}m"
            else:
                inclusive_upper = (upper_bound - 1) / 10
                price_range = (
                    f"£{lower_bound / 10:.1f}m–£{inclusive_upper:.1f}m"
                )
            return f"{label} · {price_range}"

    return "Uncategorized"


def render_metric_card(
    label: object,
    value: object,
    badge: object,
    *,
    badge_background: str = "#475569",
    badge_color: str = "#FFFFFF",
    tooltip: str | None = None,
) -> None:
    """Render a uniformly styled metric card with a footer badge."""
    badge_text = str(badge)
    tooltip_text = tooltip or badge_text
    cursor = "help" if tooltip else "default"
    st.markdown(
        (
            '<div style="border:1px solid rgba(128,128,128,0.28);'
            'border-radius:0.75rem;overflow:hidden;min-height:116px;'
            'display:flex;flex-direction:column;'
            'background:var(--secondary-background-color,transparent)">'
            '<div style="padding:0.75rem 1rem 0.65rem;flex:1">'
            '<div style="font-size:0.875rem;opacity:0.68;margin-bottom:0.2rem">'
            f'{escape(str(label))}</div>'
            '<div style="font-size:1.5rem;font-weight:600;line-height:1.2">'
            f'{escape(str(value))}</div></div>'
            '<div style="border-top:1px solid rgba(128,128,128,0.22);'
            'padding:0.55rem 1rem">'
            f'<span title="{escape(tooltip_text)}" '
            f'aria-label="{escape(tooltip_text)}" '
            'style="display:inline-flex;align-items:center;border-radius:9999px;'
            'padding:0.18rem 0.55rem;font-size:0.75rem;font-weight:600;'
            f'cursor:{cursor};white-space:nowrap;'
            f'background:{badge_background};color:{badge_color}">'
            f'{escape(badge_text)}</span></div></div>'
        ),
        unsafe_allow_html=True,
    )


def render_price_metric_card(price: object, category_detail: object) -> None:
    """Render the Price card with a colored, range-aware category badge."""
    category, separator, price_range = str(category_detail).partition(" · ")
    category = category or "Uncategorized"
    background_color, text_color = PRICE_CATEGORY_COLORS.get(
        category, PRICE_CATEGORY_COLORS["Uncategorized"]
    )
    tooltip = f"Price range: {price_range}" if separator else "Price range unavailable"
    render_metric_card(
        "Price",
        price,
        category,
        badge_background=background_color,
        badge_color=text_color,
        tooltip=tooltip,
    )


def historical_fpl_version(
    league: str,
) -> tuple[tuple[str, tuple[int, int] | None], ...]:
    """Build the cache version for all season-level player files."""
    return tuple(
        (
            season,
            file_version(
                fpl_season_path(league, season)
            ),
        )
        for season in discover_seasons(league)
    )


@st.cache_data(show_spinner=False)
def load_historical_fpl(
    league: str,
    data_version: tuple[tuple[str, tuple[int, int] | None], ...] = (),
) -> pd.DataFrame:
    """Load and combine all player-season files for the selected league."""
    del data_version  # Used by Streamlit as a file-change-aware cache key.
    season_frames = []

    for season in discover_seasons(league):
        csv_path = fpl_season_path(league, season)
        season_data = pd.read_csv(csv_path)
        season_data["season"] = season
        season_frames.append(season_data)

    if not season_frames:
        return pd.DataFrame()

    players = pd.concat(season_frames, ignore_index=True, sort=False)
    players["player_id"] = players["player_id"].astype("string")
    players = deduplicate_player_seasons(players)

    base_defcon_columns = [
        "clearances",
        "blocks",
        "interceptions",
        "tackles_won",
    ]
    if all(column in players.columns for column in base_defcon_columns):
        defensive_stats = players[base_defcon_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        calculated_defcon = defensive_stats.sum(axis=1, min_count=4)
        normalized_positions = players.get(
            "fpl_pos", pd.Series(index=players.index, dtype="string")
        ).astype("string").str.upper()
        midfield_forward = normalized_positions.isin(["MID", "FWD"])
        if "recoveries" in players.columns:
            recoveries = pd.to_numeric(players["recoveries"], errors="coerce")
            calculated_defcon.loc[midfield_forward] = (
                calculated_defcon.loc[midfield_forward]
                + recoveries.loc[midfield_forward]
            )
        existing_defcon = pd.to_numeric(
            players.get(
                "defcon", pd.Series(index=players.index, dtype="float64")
            ),
            errors="coerce",
        )
        players["defcon"] = calculated_defcon.combine_first(existing_defcon)

    return players


def deduplicate_player_seasons(players: pd.DataFrame) -> pd.DataFrame:
    """Resolve provider duplicates to one deterministic player-season row."""
    if players.empty:
        return players.copy()
    required = {"season", "player_id", "name"}
    if not required.issubset(players.columns):
        missing = sorted(required - set(players.columns))
        raise ValueError(f"Player-season data is missing columns: {missing}")

    resolved = players.dropna(subset=["player_id", "name"]).copy()
    resolved["player_id"] = resolved["player_id"].astype("string")

    # Older provider files can contain multiple rows for the same stable ID in
    # one season (transfers and historical identity collisions). Player cards
    # require exactly one record per player-season, so retain the best-supported
    # row rather than double-counting two potentially different people.
    resolved["_minutes_sort"] = pd.to_numeric(
        resolved.get(
            "minutes", pd.Series(index=resolved.index, dtype="float64")
        ),
        errors="coerce",
    ).fillna(-1)
    resolved["_points_sort"] = pd.to_numeric(
        resolved.get(
            "total_points", pd.Series(index=resolved.index, dtype="float64")
        ),
        errors="coerce",
    ).fillna(-1)
    return (
        resolved.sort_values(
            ["season", "player_id", "_minutes_sort", "_points_sort"],
            ascending=[True, True, False, False],
            kind="mergesort",
        )
        .drop_duplicates(["season", "player_id"], keep="first")
        .drop(columns=["_minutes_sort", "_points_sort"])
        .reset_index(drop=True)
    )

@st.cache_data(show_spinner=False)
def load_gameweek_fpl(
    league: str,
    season: str,
    data_version: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Load gameweek-level FPL data for one league and season."""
    del data_version  # Used by Streamlit as a file-change-aware cache key.
    csv_path = fpl_gameweeks_path(league, season)
    if not csv_path.is_file():
        return pd.DataFrame()

    gameweeks = pd.read_csv(csv_path)
    if "player_id" in gameweeks.columns:
        gameweeks["player_id"] = gameweeks["player_id"].astype("string")
    return gameweeks


@st.cache_data(show_spinner=False)
def load_player_profiles(
    csv_path: str, data_version: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Load the published profile artifact without recalculating league scores."""
    del data_version
    path = Path(csv_path)
    if not path.is_file():
        return pd.DataFrame()
    profiles = pd.read_csv(path)
    if "player_id" in profiles:
        profiles["player_id"] = profiles["player_id"].astype("string")
    return profiles


@st.cache_data(show_spinner=False)
def load_raw_player_details(
    csv_path: str, data_version: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Load current FPL market, availability, set-piece and image metadata."""
    del data_version
    path = Path(csv_path)
    if not path.is_file():
        return pd.DataFrame()
    details = pd.read_csv(path)
    if "id" in details:
        details["id"] = pd.to_numeric(details["id"], errors="coerce")
    return details


@st.cache_data(show_spinner=False)
def load_raw_teams(
    csv_path: str, data_version: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Load the raw FPL team IDs and official badge codes."""
    del data_version
    path = Path(csv_path)
    if not path.is_file():
        return pd.DataFrame()
    teams = pd.read_csv(path)
    for column in ["id", "code"]:
        if column in teams:
            teams[column] = pd.to_numeric(teams[column], errors="coerce")
    return teams


@st.cache_data(show_spinner=False)
def load_fixture_schedule(
    metadata_path: str,
    fixtures_path: str,
    metadata_version: tuple[int, int] | None = None,
    fixtures_version: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Build one factual upcoming-fixture row per team, independent of models."""
    del metadata_version, fixtures_version
    metadata_file = Path(metadata_path)
    fixtures_file = Path(fixtures_path)
    if not metadata_file.is_file() or not fixtures_file.is_file():
        return pd.DataFrame()

    metadata = pd.read_csv(metadata_file)
    fixtures = pd.read_csv(fixtures_file)
    required_metadata = {
        "fpl_id", "team", "opp", "opp_short", "venue", "date_sched"
    }
    required_fixtures = {"id", "event", "team_h_difficulty", "team_a_difficulty"}
    if not required_metadata.issubset(metadata) or not required_fixtures.issubset(fixtures):
        return pd.DataFrame()

    if "finished" in fixtures:
        finished = fixtures["finished"].astype("string").str.lower().eq("true")
        fixtures = fixtures.loc[~finished].copy()
    fixture_columns = [
        column
        for column in [
            "id",
            "event",
            "kickoff_time",
            "team_h_difficulty",
            "team_a_difficulty",
        ]
        if column in fixtures
    ]
    fixture_data = fixtures[fixture_columns].rename(
        columns={"id": "fpl_id", "event": "gw_orig"}
    )
    schedule = metadata.merge(
        fixture_data, on="fpl_id", how="inner", validate="many_to_one"
    )
    schedule["is_home"] = schedule["venue"].astype("string").str.lower().eq("home")
    schedule["fdr"] = schedule["team_a_difficulty"]
    schedule.loc[schedule["is_home"], "fdr"] = schedule.loc[
        schedule["is_home"], "team_h_difficulty"
    ]
    schedule = schedule.rename(
        columns={
            "opp_short": "opponent",
            "opp_name": "opponent_name",
            "opp": "opponent_team_id",
            "team_short": "team_code",
        }
    )
    return schedule[
        [
            column
            for column in [
                "fpl_id",
                "gw_orig",
                "kickoff_time",
                "date_sched",
                "team",
                "team_code",
                "opponent",
                "opponent_name",
                "opponent_team_id",
                "is_home",
                "fdr",
            ]
            if column in schedule
        ]
    ].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_expected_points(
    forecast_path: str, data_version: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Load one season-specific forecast window."""
    del data_version
    path = Path(forecast_path)
    if not path.is_file():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_player_gameweek_history(
    gameweeks: pd.DataFrame, player_id: str, player_name: str
) -> pd.DataFrame:
    """Return selected FPL scoring components per gameweek."""
    if gameweeks.empty or "round" not in gameweeks.columns:
        return pd.DataFrame()

    player_gameweeks = pd.DataFrame()
    if "player_id" in gameweeks.columns:
        player_gameweeks = gameweeks.loc[
            gameweeks["player_id"].eq(str(player_id))
        ].copy()

    # Exact-name fallback supports seasons whose generated stable ID changed.
    if player_gameweeks.empty and "name" in gameweeks.columns:
        player_gameweeks = gameweeks.loc[
            gameweeks["name"].astype("string").eq(str(player_name))
        ].copy()

    if player_gameweeks.empty or "total_points" not in player_gameweeks.columns:
        return pd.DataFrame()

    player_gameweeks["round"] = pd.to_numeric(
        player_gameweeks["round"], errors="coerce"
    )
    player_gameweeks["total_points"] = pd.to_numeric(
        player_gameweeks["total_points"], errors="coerce"
    )

    def numeric_column(column: str) -> pd.Series:
        return pd.to_numeric(
            player_gameweeks.get(
                column, pd.Series(0, index=player_gameweeks.index)
            ),
            errors="coerce",
        ).fillna(0)

    positions = player_gameweeks.get(
        "fpl_pos", pd.Series(index=player_gameweeks.index, dtype="string")
    ).astype("string").str.upper().replace("GK", "GKP")
    goal_multiplier = positions.map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4})
    player_gameweeks["goal_points"] = (
        numeric_column("goals_scored") * goal_multiplier.fillna(0)
    )
    player_gameweeks["assist_points"] = numeric_column("assists") * 3
    player_gameweeks["returns"] = (
        numeric_column("goals_scored") + numeric_column("assists")
    )
    player_gameweeks["bonus_points"] = numeric_column("bonus")
    defensive_contribution = numeric_column("defensive_contribution")
    earned_defensive_points = (
        (positions.eq("DEF") & defensive_contribution.ge(10))
        | (
            positions.isin(["MID", "FWD"])
            & defensive_contribution.ge(12)
        )
    )
    player_gameweeks["defensive_contribution_points"] = (
        earned_defensive_points.astype(int) * 2
    )
    player_gameweeks["minutes"] = pd.to_numeric(
        player_gameweeks.get(
            "minutes", pd.Series(0, index=player_gameweeks.index)
        ),
        errors="coerce",
    ).fillna(0)
    player_gameweeks["appearance_points"] = 0
    player_gameweeks.loc[
        player_gameweeks["minutes"].between(1, 59), "appearance_points"
    ] = 1
    player_gameweeks.loc[
        player_gameweeks["minutes"].ge(60), "appearance_points"
    ] = 2
    player_gameweeks["save_points"] = (
        numeric_column("saves").floordiv(3) * positions.eq("GKP").astype(int)
    )
    player_gameweeks["card_points"] = (
        numeric_column("yellow_cards").mul(-1)
        + numeric_column("red_cards").mul(-3)
    )
    opponent_column = next(
        (
            column
            for column in ("opp_code", "opponent_team")
            if column in player_gameweeks.columns
        ),
        None,
    )
    if opponent_column is None:
        player_gameweeks["opponent"] = ""
        opponent_column = "opponent"

    def combine_opponents(values: pd.Series) -> str:
        opponents = [
            str(value) for value in values if pd.notna(value) and str(value).strip()
        ]
        return ", ".join(dict.fromkeys(opponents))

    history = (
        player_gameweeks.dropna(subset=["round"])
        .groupby("round", as_index=False)
        .agg(
            Total_Points=(
                "total_points", lambda values: values.sum(min_count=1)
            ),
            Appearance_Points=("appearance_points", "sum"),
            Goal_Points=("goal_points", "sum"),
            Assist_Points=("assist_points", "sum"),
            Bonus_Points=("bonus_points", "sum"),
            Defensive_Contribution_Points=(
                "defensive_contribution_points", "sum"
            ),
            Save_Points=("save_points", "sum"),
            Card_Points=("card_points", "sum"),
            Minutes=("minutes", "sum"),
            Starts=("minutes", lambda values: int((values >= 60).sum())),
            Returns=("returns", "sum"),
            Opponent=(opponent_column, combine_opponents),
            Fixtures=("round", "size"),
        )
        .rename(
            columns={
                "round": "GW",
                "Total_Points": "Total FPL points",
                "Appearance_Points": "Appearance points",
                "Goal_Points": "Goal points",
                "Assist_Points": "Assist points",
                "Bonus_Points": "Bonus points",
                "Defensive_Contribution_Points": (
                    "Defensive contribution points"
                ),
                "Save_Points": "Save points",
                "Card_Points": "Card points",
            }
        )
        .sort_values("GW")
        .reset_index(drop=True)
    )
    history["GW"] = history["GW"].astype(int)
    return history


def build_player_history(
    players: pd.DataFrame, player_id: str, per_90: bool = False
) -> pd.DataFrame:
    """Return one row per season with league and position percentiles."""
    ranked_players = add_metric_percentiles(players, per_90=per_90)
    history = ranked_players.loc[
        ranked_players["player_id"] == str(player_id)
    ].copy()

    output_columns = []
    output_names = {}
    for source, label in PLAYER_HISTORY_COLUMNS.items():
        if source not in history.columns:
            continue

        output_columns.append(source)
        output_names[source] = label
        if source in RANKED_METRICS:
            for scope, scope_label in (
                ("position", "Position Percentile"),
                ("league", "League Percentile"),
            ):
                percentile_column = f"{source}_{scope}_percentile"
                if percentile_column in history.columns:
                    output_columns.append(percentile_column)
                    output_names[percentile_column] = f"{label} {scope_label}"

    history = history[output_columns].rename(columns=output_names)

    if "Price" in history.columns:
        raw_prices = pd.to_numeric(history["Price"], errors="coerce")
        history["Price Category"] = [
            format_price_category(price, position, season)
            for price, position, season in zip(
                raw_prices,
                history["Position"],
                history["Season"],
            )
        ]
        history["Price"] = raw_prices / 10
        history["Price"] = history["Price"].map(
            lambda price: f"£{price:.1f}m" if pd.notna(price) else ""
        )

    return history.sort_values("Season", ascending=False).reset_index(drop=True)


def add_metric_percentiles(
    players: pd.DataFrame, per_90: bool = False
) -> pd.DataFrame:
    """Calculate metric percentiles within each season and FPL position.

    The highest value receives P100 and ties share their average percentile.
    Missing metric values or positions remain without a percentile.
    """
    ranked = players.copy()
    if "season" not in ranked.columns:
        return ranked

    minutes = pd.to_numeric(
        ranked.get("minutes", pd.Series(index=ranked.index, dtype="float64")),
        errors="coerce",
    )
    valid_minutes = minutes.where(minutes.gt(0))

    for metric in RANKED_METRICS:
        if metric not in ranked.columns:
            continue

        values = pd.to_numeric(ranked[metric], errors="coerce")
        if per_90 and metric in PER_90_METRICS:
            values = values.div(valid_minutes).mul(90)
            values = values.where(minutes.ge(MIN_PER_90_MINUTES))
            ranked[metric] = values
        ascending = metric not in LOWER_IS_BETTER_METRICS
        ranked[f"{metric}_league_percentile"] = (
            values.groupby(ranked["season"]).rank(
                method="average",
                pct=True,
                ascending=ascending,
                na_option="keep",
            )
            * 100
        ).round(1)

        if "fpl_pos" in ranked.columns:
            ranked[f"{metric}_position_percentile"] = (
                values.groupby([ranked["season"], ranked["fpl_pos"]]).rank(
                    method="average",
                    pct=True,
                    ascending=ascending,
                    na_option="keep",
                )
                * 100
            ).round(1)

    return ranked


def build_metric_percentile_table(
    history: pd.DataFrame, per_90: bool = False
) -> pd.DataFrame:
    """Convert wide player history into a readable metric/percentile table."""
    rows = []
    for record in history.to_dict("records"):
        for label in RANKED_METRICS.values():
            if label not in record:
                continue
            source = next(
                source for source, source_label in RANKED_METRICS.items()
                if source_label == label
            )
            rows.append(
                {
                    "Season": record.get("Season", ""),
                    "Metric": (
                        f"{label} /90"
                        if per_90 and source in PER_90_METRICS
                        else label
                    ),
                    "Value": record[label],
                    "Position Percentile": record.get(
                        f"{label} Position Percentile", pd.NA
                    ),
                    "League Percentile": record.get(
                        f"{label} League Percentile", pd.NA
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "Season",
            "Metric",
            "Value",
            "Position Percentile",
            "League Percentile",
        ],
    )


def format_metric_percentile_delta(record: pd.Series, metric: str) -> str:
    """Format a metric card delta with position and league percentiles."""
    position_percentile = record.get(
        f"{metric} Position Percentile", pd.NA
    )
    league_percentile = record.get(f"{metric} League Percentile", pd.NA)

    def display(percentile: object) -> str:
        return "–" if pd.isna(percentile) else f"P{float(percentile):.0f}"

    return (
        f"Pos {display(position_percentile)} · "
        f"League {display(league_percentile)}"
    )


def is_goalkeeper_position(position: object) -> bool:
    """Return whether an FPL position value represents a goalkeeper."""
    return str(position).strip().upper() in {"GK", "GKP"}


def format_percentage(value: object) -> str:
    """Format a percentage metric without applying per-90 conversion."""
    return "–" if pd.isna(value) else f"{float(value):.1f}%"


def format_metric_card_value(value: object, per_90: bool) -> str:
    """Format total and per-90 card values, including unavailable rates."""
    if pd.isna(value):
        return "–"
    return format(value, ",.2f" if per_90 else ",.0f")


def build_player_labels(players: pd.DataFrame) -> dict[str, str]:
    """Create selector labels for the supplied player pool."""
    selectable_players = players.drop_duplicates(subset="player_id").sort_values("name")

    labels = {}
    for player in selectable_players.itertuples():
        team = (
            str(player.team)
            if pd.notna(player.team) and str(player.team).strip()
            else "No team"
        )
        labels[str(player.player_id)] = f"{player.name}"

    return labels


def search_player_pool(players: pd.DataFrame, query: str) -> pd.DataFrame:
    """Search an already-filtered player pool by player or team name."""
    search_text = str(query).strip()
    if not search_text or players.empty:
        return players

    name_matches = players["name"].astype("string").str.contains(
        search_text, case=False, na=False, regex=False
    )
    team_matches = players["team"].astype("string").str.contains(
        search_text, case=False, na=False, regex=False
    )
    return players.loc[name_matches | team_matches]


def filter_player_pool(
    players: pd.DataFrame,
    season: str,
    positions: list[str],
    teams: list[str],
    price_range: tuple[float, float],
    price_category: str = "All categories",
) -> pd.DataFrame:
    """Filter the searchable player pool for one season."""
    pool = players.loc[players["season"].eq(season)].copy()

    if "fpl_pos" in pool.columns:
        pool = pool.loc[pool["fpl_pos"].isin(positions)]
    if "team" in pool.columns:
        pool = pool.loc[pool["team"].isin(teams)]
    if "now_cost" in pool.columns:
        prices = pd.to_numeric(pool["now_cost"], errors="coerce") / 10
        pool = pool.loc[prices.between(*price_range, inclusive="both")]
    if price_category != "All categories" and {
        "now_cost",
        "fpl_pos",
    }.issubset(pool.columns):
        category_labels = [
            format_price_category(price, position, season).partition(" · ")[0]
            for price, position in zip(pool["now_cost"], pool["fpl_pos"])
        ]
        category_mask = pd.Series(category_labels, index=pool.index).eq(
            price_category
        )
        pool = pool.loc[category_mask]

    return pool


def _render_legacy_overview_tab(
    selected_record: pd.Series,
    total_record: pd.Series,
    raw_record: pd.Series,
    gameweek_history: pd.DataFrame,
    forecast: pd.DataFrame,
    profile: pd.Series | None,
    *,
    per_90: bool,
) -> None:
    """Render the decision-oriented summary for a selected player-season."""
    position = selected_record["Position"]
    is_goalkeeper = is_goalkeeper_position(position)
    metric_suffix = " /90" if per_90 else ""
    selected_points = pd.to_numeric(selected_record["Points"], errors="coerce")
    selected_goals = pd.to_numeric(selected_record["Goals"], errors="coerce")
    selected_assists = pd.to_numeric(selected_record["Assists"], errors="coerce")
    selected_saves = pd.to_numeric(selected_record.get("Saves"), errors="coerce")
    selected_defcon = pd.to_numeric(selected_record.get("Def Con"), errors="coerce")

    metrics = st.columns(5)
    with metrics[0]:
        render_price_metric_card(
            selected_record["Price"],
            selected_record.get("Price Category", "Uncategorized"),
        )
    with metrics[1]:
        # Overview always shows decision-friendly season points, not a rate.
        total_points = pd.to_numeric(raw_record.get("total_points"), errors="coerce")
        render_metric_card(
            f"{selected_record['Season']} points",
            format_metric_card_value(total_points, False),
            format_metric_percentile_delta(total_record, "Points"),
        )
    with metrics[2]:
        metric_name = "Saves" if is_goalkeeper else "Goals"
        metric_value = selected_saves if is_goalkeeper else selected_goals
        render_metric_card(
            f"{metric_name}{metric_suffix}",
            format_metric_card_value(metric_value, per_90),
            format_metric_percentile_delta(selected_record, metric_name),
        )
    with metrics[3]:
        if is_goalkeeper:
            save_pct = pd.to_numeric(selected_record.get("Save %"), errors="coerce")
            render_metric_card(
                "Save %",
                format_percentage(save_pct),
                format_metric_percentile_delta(selected_record, "Save %"),
            )
        else:
            render_metric_card(
                f"Assists{metric_suffix}",
                format_metric_card_value(selected_assists, per_90),
                format_metric_percentile_delta(selected_record, "Assists"),
            )
    with metrics[4]:
        render_metric_card(
            f"Def Con{metric_suffix}",
            format_metric_card_value(selected_defcon, per_90),
            format_metric_percentile_delta(selected_record, "Def Con"),
        )

    recent_points = pd.to_numeric(
        gameweek_history.tail(3).get("Total FPL points"), errors="coerce"
    ).sum(min_count=1) if not gameweek_history.empty else pd.NA
    summary = forecast_summary(forecast)
    decision_cards = st.columns(5)
    decision_cards[0].metric(
        "Last 3 GWs",
        "—" if pd.isna(recent_points) else f"{recent_points:.0f} pts",
    )
    decision_cards[1].metric(
        "Forecast window",
        "—" if not summary else f"{summary['expected_points']:.1f} xPts",
    )
    decision_cards[2].metric(
        "Predicted minutes",
        "—" if not summary else f"{summary['predicted_minutes']:.0f}",
    )
    decision_cards[3].metric(
        "Next fixture", "—" if not summary else str(summary["next_fixture"])
    )
    ownership = pd.to_numeric(raw_record.get("selected_by_percent"), errors="coerce")
    decision_cards[4].metric(
        "Ownership", "—" if pd.isna(ownership) else f"{ownership:.1f}%"
    )

    status = str(raw_record.get("status", ""))
    news = str(raw_record.get("news", "")).strip()
    if status and status.lower() not in {"a", "available", "nan"}:
        st.warning(f"Availability status: {status}. {news}".strip())
    elif news and news.lower() != "nan":
        st.info(news)

    if profile is not None and str(profile.get("profile_status")) in {
        "Established",
        "Provisional",
    }:
        st.caption(
            f"Profile: {profile.get('production_tier', 'Unrated')} "
            f"{profile.get('production_archetype', 'profile')} · "
            f"Primary strength: {profile.get('primary_strength', '—')}"
        )


def render_overview_tab(
    player_name: str,
    selected_season: str,
    selected_record: pd.Series,
    raw_record: pd.Series,
    gameweek_history: pd.DataFrame,
    fixtures: pd.DataFrame,
    forecast: pd.DataFrame,
    profile: pd.Series | None,
    profile_season: str | None,
    profile_is_carryover: bool,
    profile_trend_data: dict[str, float | str],
    player_details: pd.Series | None,
    team_badges: dict[int, str],
    alternatives: pd.DataFrame,
    data_freshness: dict[str, str],
) -> None:
    """Render a player-first, decision-oriented landing page."""

    def number(value: object) -> float:
        return pd.to_numeric(value, errors="coerce")

    position = str(selected_record["Position"])
    team = str(selected_record["Team"])
    status_code = str(raw_record.get("status", "a")).strip().lower()
    availability = {
        "a": "Available",
        "d": "Doubtful",
        "i": "Injured",
        "s": "Suspended",
        "u": "Unavailable",
        "n": "Unavailable",
    }.get(status_code, "Status unknown")
    status_color = "#16A34A" if availability == "Available" else "#D97706"
    news = str(raw_record.get("news", "")).strip()
    if news.lower() == "nan":
        news = ""
    ownership = number(raw_record.get("selected_by_percent"))
    if player_details is not None:
        ownership = number(player_details.get("selected_by_percent"))
    selected_team_id = number(raw_record.get("fpl_team_numeric_id"))
    selected_team_badge = (
        None if pd.isna(selected_team_id) else team_badges.get(int(selected_team_id))
    )
    image_config = load_player_image_config(
        file_version(PLAYER_IMAGE_CONFIG_PATH)
    )
    image_asset_version = str(image_config.get("asset_version", "25"))
    placeholder_url = player_placeholder_url(asset_version=image_asset_version)

    image_column, identity_column, market_column = st.columns([1, 4, 2])
    with image_column:
        photo = "" if player_details is None else str(player_details.get("photo", "")).strip()
        photo_urls = player_photo_urls(
            photo, asset_version=image_asset_version
        )
        initials = "".join(part[0] for part in player_name.split()[:2]).upper()
        if photo_urls and placeholder_url:
            st.markdown(
                '<object type="image/png" width="110" height="140" '
                f'data="{escape(photo_urls["current"])}" '
                f'aria-label="{escape(player_name)} portrait">'
                '<object type="image/png" width="110" height="140" '
                f'data="{escape(photo_urls["legacy"])}" '
                f'aria-label="{escape(player_name)} legacy portrait">'
                f'<img src="{escape(placeholder_url)}" '
                f'alt="{escape(player_name)} portrait unavailable" '
                'width="110" height="140" style="object-fit:contain" />'
                '</object>'
                '</object>',
                unsafe_allow_html=True,
            )
        elif placeholder_url:
            st.markdown(
                f'<img src="{escape(placeholder_url)}" '
                f'alt="{escape(player_name)} portrait unavailable" '
                'width="110" height="140" style="object-fit:contain" />',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="width:96px;height:96px;border-radius:50%;display:flex;'
                'align-items:center;justify-content:center;background:#334155;color:white;'
                f'font-size:1.75rem;font-weight:700">{escape(initials)}</div>',
                unsafe_allow_html=True,
            )
    with identity_column:
        st.markdown(f"## {player_name}")
        badge_column, team_column = st.columns(
            [1, 9], vertical_alignment="center"
        )
        if selected_team_badge:
            badge_column.image(selected_team_badge, width=34)
        team_column.markdown(f"**{team}** · {position} · {selected_season}")
        st.markdown(
            '<span style="display:inline-block;padding:.2rem .6rem;border-radius:999px;'
            f'background:{status_color};color:white;font-size:.8rem;font-weight:600">'
            f'{escape(availability)}</span>',
            unsafe_allow_html=True,
        )
        if news:
            st.caption(news)
    with market_column:
        market_metrics = st.columns(2)
        market_metrics[0].metric("Price", str(selected_record["Price"]))
        market_metrics[1].metric(
            "Ownership", "—" if pd.isna(ownership) else f"{ownership:.1f}%"
        )
        st.caption(str(selected_record.get("Price Category", "Uncategorized")).replace(" Â· ", " · "))

    st.markdown("### Next fixture")
    with st.container(border=True):
        if fixtures.empty:
            st.info("No upcoming fixture is currently published for this team.")
        else:
            next_fixture = fixtures.iloc[0]
            opponent = next_fixture.get("opponent_name", next_fixture.get("opponent", "—"))
            opponent = str(opponent) if pd.notna(opponent) else str(next_fixture.get("opponent", "—"))
            venue = "H" if bool(next_fixture.get("is_home")) else "A"
            gw = number(next_fixture.get("gw_orig"))
            fdr = number(next_fixture.get("fdr"))
            predicted_minutes = number(next_fixture.get("pred_minutes"))
            expected_points = number(next_fixture.get("xPts"))
            date_value = pd.to_datetime(
                next_fixture.get("kickoff_time", next_fixture.get("date_sched")),
                errors="coerce",
                utc=True,
            )
            date_label = (
                "Date not confirmed"
                if pd.isna(date_value)
                else date_value.strftime("%a %d %b · %H:%M UTC")
            )
            opponent_team_id = number(next_fixture.get("opponent_team_id"))
            opponent_badge = (
                None
                if pd.isna(opponent_team_id)
                else team_badges.get(int(opponent_team_id))
            )
            fixture_metrics = st.columns(
                [1, 2, 2, 1, 1, 1], vertical_alignment="center"
            )
            if opponent_badge:
                fixture_metrics[0].image(opponent_badge, width=48)
            fixture_metrics[1].metric("Opponent", f"{opponent} ({venue})")
            fixture_metrics[2].metric("Kick-off", date_label)
            fixture_metrics[3].metric("Gameweek", "—" if pd.isna(gw) else f"GW{gw:.0f}")
            fixture_metrics[4].metric("FDR", "—" if pd.isna(fdr) else f"{fdr:.0f}/5")
            fixture_metrics[5].metric(
                "Next xPts",
                "Not published" if pd.isna(expected_points) else f"{expected_points:.1f}",
            )
            st.caption(
                "Predicted minutes: "
                + ("not published" if pd.isna(predicted_minutes) else f"{predicted_minutes:.0f}")
                + ". Fixture facts remain visible independently of model availability."
            )

    recent = recent_form_summary(gameweek_history)
    st.markdown("### Decision snapshot")
    snapshot = st.columns(4)
    snapshot[0].metric(
        "Recent form",
        "Preseason / no games"
        if not recent
        else f"{recent['points']:.0f} pts · last {recent['gameweeks']} GWs",
    )
    next_minutes = pd.NA if fixtures.empty else number(fixtures.iloc[0].get("pred_minutes"))
    snapshot[1].metric(
        "Minutes security",
        "Forecast not published" if pd.isna(next_minutes) else f"{next_minutes:.0f} next match",
        None if not recent else f"{recent['starts']} starts · last {recent['gameweeks']} GWs",
    )
    if player_details is None:
        snapshot[2].metric("Market movement", "Unavailable")
        snapshot[3].metric("FPL next estimate", "Unavailable")
    else:
        net_transfers = number(player_details.get("transfers_in_event")) - number(
            player_details.get("transfers_out_event")
        )
        price_change = number(player_details.get("cost_change_event"))
        snapshot[2].metric(
            "Net transfers this GW",
            "—" if pd.isna(net_transfers) else f"{net_transfers:+,.0f}",
            None if pd.isna(price_change) else f"{price_change / 10:+.1f}m price change",
        )
        ep_next = number(player_details.get("ep_next"))
        snapshot[3].metric(
            "FPL next estimate", "—" if pd.isna(ep_next) else f"{ep_next:.1f} pts"
        )

    st.markdown("#### Current-season snapshot")
    season_minutes = number(raw_record.get("minutes"))
    season_points = number(raw_record.get("total_points"))
    price_tenths = number(raw_record.get("now_cost"))
    points_per_million = (
        pd.NA
        if pd.isna(season_points) or pd.isna(price_tenths) or price_tenths <= 0
        else season_points / (price_tenths / 10)
    )
    season_metrics = st.columns(5)
    season_metrics[0].metric(
        "FPL points", "—" if pd.isna(season_points) else f"{season_points:.0f}"
    )
    if is_goalkeeper_position(position):
        season_metrics[1].metric(
            "Saves", "—" if pd.isna(number(raw_record.get("saves"))) else f"{number(raw_record.get('saves')):.0f}"
        )
        save_pct = number(raw_record.get("save_pct"))
        season_metrics[2].metric("Save %", "—" if pd.isna(save_pct) else f"{save_pct:.1f}%")
    else:
        goals = number(raw_record.get("goals_scored"))
        assists = number(raw_record.get("assists"))
        season_metrics[1].metric("Goals", "—" if pd.isna(goals) else f"{goals:.0f}")
        season_metrics[2].metric("Assists", "—" if pd.isna(assists) else f"{assists:.0f}")
    expected_involvement = number(raw_record.get("xg")) + number(raw_record.get("xa"))
    season_metrics[3].metric(
        "xGI", "—" if pd.isna(expected_involvement) else f"{expected_involvement:.2f}"
    )
    season_metrics[4].metric(
        "Points / £m",
        "—" if pd.isna(points_per_million) else f"{points_per_million:.1f}",
    )
    if pd.isna(season_minutes) or season_minutes == 0:
        st.caption(
            "Preseason state: current-season production is shown as zero; use the "
            "labelled prior profile and upcoming fixtures for context."
        )

    profile_column, fixture_column = st.columns([3, 2])
    with profile_column:
        st.markdown("### Player DNA")
        with st.container(border=True):
            if profile is None or str(profile.get("profile_status")) not in {
                "Established", "Provisional"
            }:
                st.info("No reliable current or prior-season production profile is available.")
            else:
                provenance = (
                    f"Previous-season baseline ({profile_season})"
                    if profile_is_carryover
                    else f"Current-season evidence ({profile_season})"
                )
                reliability = number(profile.get("reliability"))
                st.markdown(
                    f"#### {profile.get('production_tier', 'Unrated')} "
                    f"{profile.get('production_archetype', 'profile')}"
                )
                st.caption(
                    f"{provenance} · {profile.get('profile_status', 'Unknown')}"
                    + ("" if pd.isna(reliability) else f" · {reliability * 100:.0f}% reliability")
                )
                for dimension in profile_dimensions(profile):
                    percentile = float(dimension["Percentile"])
                    st.markdown(
                        f"**{dimension['Dimension']}** · P{percentile:.0f}"
                    )
                    st.progress(percentile / 100)
                characteristics = [
                    item.strip()
                    for item in str(profile.get("playing_characteristics", "")).split(";")
                    if item.strip()
                ]
                set_piece_roles = []
                if player_details is not None:
                    for column, label in [
                        ("penalties_order", "First-choice penalties"),
                        ("corners_and_indirect_freekicks_order", "First-choice corners"),
                        ("direct_freekicks_order", "First-choice direct free-kicks"),
                    ]:
                        if number(player_details.get(column)) == 1:
                            set_piece_roles.append(label)
                for characteristic in [*characteristics[:3], *set_piece_roles]:
                    st.markdown(f"- {characteristic}")
                if profile_trend_data:
                    trend_delta = float(profile_trend_data["delta"])
                    st.caption(
                        f"Season-over-season profile trend: {profile_trend_data['label']} "
                        f"({trend_delta:+.1f} average percentile points)."
                    )
                else:
                    st.caption(
                        "Profile trend requires compatible, reliable profiles in "
                        "both the current and previous seasons."
                    )

    with fixture_column:
        st.markdown("### Fixture run")
        if fixtures.empty:
            st.info("Fixture schedule unavailable.")
        else:
            fixture_table = fixtures.copy()
            fixture_table["GW"] = pd.to_numeric(
                fixture_table["gw_orig"], errors="coerce"
            ).map(lambda value: f"GW{value:.0f}" if pd.notna(value) else "—")
            fixture_table["Fixture"] = fixture_table["opponent"].astype(str) + fixture_table[
                "is_home"
            ].map({True: " (H)", False: " (A)"})
            fixture_table["Badge"] = pd.to_numeric(
                fixture_table.get("opponent_team_id"), errors="coerce"
            ).map(
                lambda value: (
                    team_badges.get(int(value), "") if pd.notna(value) else ""
                )
            )
            fixture_table["Date"] = pd.to_datetime(
                fixture_table["date_sched"], errors="coerce"
            ).dt.strftime("%d %b")
            fixture_table["FDR"] = pd.to_numeric(fixture_table.get("fdr"), errors="coerce")
            fixture_table["xPts"] = pd.to_numeric(fixtures.get("xPts"), errors="coerce")
            fixture_table["Pred mins"] = pd.to_numeric(
                fixtures.get("pred_minutes"), errors="coerce"
            )
            st.dataframe(
                fixture_table[
                    ["Badge", "GW", "Fixture", "Date", "FDR", "xPts", "Pred mins"]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "Badge": st.column_config.ImageColumn("", width="small")
                },
            )
            summary = forecast_summary(forecast)
            if summary:
                forecast_gws = pd.to_numeric(forecast["gw_orig"], errors="coerce")
                st.caption(
                    f"GW{forecast_gws.min():.0f}–GW{forecast_gws.max():.0f}: "
                    f"{summary['expected_points']:.1f} xPts and "
                    f"{summary['predicted_minutes']:.0f} projected minutes."
                )
            else:
                st.caption("Forecast values will appear here when the model publishes them.")

    if not gameweek_history.empty:
        st.markdown("### Recent evidence")
        trend = gameweek_history.tail(5).copy()
        trend["GW label"] = "GW" + trend["GW"].astype(str)
        figure = px.bar(
            trend,
            x="GW label",
            y="Total FPL points",
            text_auto=".0f",
            custom_data=["Minutes", "Opponent", "Starts", "Returns"],
            title="Points and playing time across the last five gameweeks",
        )
        figure.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>Points: %{y:.0f}<br>Minutes: %{customdata[0]:.0f}"
                "<br>Opponent: %{customdata[1]}<br>Starts: %{customdata[2]:.0f}"
                "<br>Returns: %{customdata[3]:.0f}<extra></extra>"
            )
        )
        figure.update_layout(height=280, margin=dict(t=45, b=15, l=15, r=15))
        st.plotly_chart(figure, width="stretch")

    positives, risks = decision_factors(
        fixtures,
        recent,
        profile,
        player_details,
        profile_is_carryover=profile_is_carryover,
    )
    positive_column, risk_column = st.columns(2)
    with positive_column:
        st.markdown("### Case for")
        with st.container(border=True):
            if positives:
                for reason in positives:
                    st.markdown(f"- {reason}")
            else:
                st.caption("No strong positive signal is supported by the available data yet.")
    with risk_column:
        st.markdown("### Risks and uncertainty")
        with st.container(border=True):
            if risks:
                for risk in risks:
                    st.markdown(f"- {risk}")
            else:
                st.caption("No material risk flag is present in the available data.")

    st.markdown("### Comparable alternatives")
    st.caption("Same-position players within £0.5m, ranked by current points then ownership.")
    if alternatives.empty:
        st.info("No comparable alternatives are available in the selected player pool.")
    else:
        st.dataframe(alternatives, hide_index=True, width="stretch")

    freshness = " · ".join(
        f"{label}: {value}" for label, value in data_freshness.items()
    )
    if freshness:
        st.caption(f"Data provenance — {freshness}")


def render_performance_tab(
    player_name: str,
    selected_season: str,
    gameweek_history: pd.DataFrame,
    percentile_table: pd.DataFrame,
) -> None:
    """Render selected-season performance and gameweek trends."""
    st.subheader("Selected-season performance")
    if gameweek_history.empty:
        st.info(f"No gameweek data is available for {player_name} in {selected_season}.")
    else:
        stacked = gameweek_history.melt(
            id_vars=["GW", "Total FPL points", "Opponent", "Minutes", "Fixtures"],
            value_vars=GAMEWEEK_POINT_COMPONENTS,
            var_name="Contribution",
            value_name="Points",
        )
        figure = px.bar(
            stacked,
            x="GW",
            y="Points",
            color="Contribution",
            color_discrete_map=GAMEWEEK_COMPONENT_COLORS,
            barmode="stack",
            custom_data=[
                "Contribution",
                "Total FPL points",
                "Opponent",
                "Minutes",
                "Fixtures",
            ],
            title=f"{player_name} points contributions by gameweek — {selected_season}",
        )
        figure.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>GW: %{x}<br>Points: %{y:.0f}<br>"
                "Total FPL points: %{customdata[1]:.0f}<br>"
                "Opponent: %{customdata[2]}<br>Minutes: %{customdata[3]:.0f}<br>"
                "Fixtures: %{customdata[4]}<extra></extra>"
            )
        )
        figure.update_layout(
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
            xaxis_title="Gameweek",
            yaxis_title="FPL points from selected contributions",
            legend_title_text="Contribution",
        )
        figure.update_xaxes(dtick=1)
        st.plotly_chart(figure, width="stretch")

    season_percentiles = percentile_table.loc[
        percentile_table["Season"].eq(selected_season)
    ] if not percentile_table.empty else percentile_table
    ui.table(
        data=season_percentiles,
        caption=f"Metric values and percentiles for {player_name} in {selected_season}",
        key=f"performance_{player_name}_{selected_season}",
        max_height=520,
    )


def render_forecast_tab(
    player_name: str, selected_season: str, forecast: pd.DataFrame
) -> None:
    """Render the latest exact-season forecast window."""
    st.subheader("Future gameweeks")
    if forecast.empty:
        st.info(
            f"No current forecast artifact is available for {player_name} in "
            f"{selected_season}. Forecasts never fall back to another season."
        )
        return

    chart = forecast.copy()
    chart["GW"] = chart.get("gw_orig")
    chart["Expected points"] = pd.to_numeric(chart.get("xPts"), errors="coerce")
    chart["Fixture"] = (
        chart.get("opponent", pd.Series("—", index=chart.index)).astype(str)
        + chart.get("is_home", pd.Series(False, index=chart.index)).map(
            {True: " (H)", False: " (A)", 1: " (H)", 0: " (A)"}
        ).fillna("")
    )
    figure = px.bar(
        chart,
        x="GW",
        y="Expected points",
        color="Fixture",
        text_auto=".1f",
        title=f"{player_name} expected points by upcoming fixture",
    )
    figure.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20))
    figure.update_xaxes(dtick=1)
    st.plotly_chart(figure, width="stretch")

    display_columns = {
        "gw_orig": "GW",
        "date_sched": "Date",
        "opponent": "Opponent",
        "is_home": "Home",
        "fdr": "FDR",
        "pred_minutes": "Predicted minutes",
        "p_goal": "Goal probability",
        "p_assist": "Assist probability",
        "xg_mean": "xG",
        "xa_mean": "xA",
        "xPts": "Expected points",
    }
    available = [column for column in display_columns if column in forecast]
    table = forecast[available].rename(columns=display_columns)
    ui.table(
        data=table,
        caption="Latest published season-specific forecast window",
        key=f"forecast_{player_name}_{selected_season}",
        max_height=420,
    )


def render_history_tab(
    player_name: str,
    player_id: str,
    history: pd.DataFrame,
    percentile_table: pd.DataFrame,
    *,
    per_90: bool,
) -> None:
    """Render prior seasons and historical percentile context."""
    metric_suffix = " /90" if per_90 else ""
    chart_data = history.copy()
    chart_data["Points"] = pd.to_numeric(chart_data["Points"], errors="coerce")
    chart_data = chart_data.sort_values("Season")
    figure = px.line(
        chart_data,
        x="Season",
        y="Points",
        markers=True,
        title=f"FPL points{metric_suffix} by season",
    )
    figure.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(figure, width="stretch")

    regular_columns = [
        label for label in PLAYER_HISTORY_COLUMNS.values() if label in history.columns
    ]
    regular_history = history[regular_columns].copy()
    if per_90:
        regular_history = regular_history.rename(
            columns={
                label: f"{label} /90"
                for source, label in RANKED_METRICS.items()
                if source in PER_90_METRICS and label in regular_history.columns
            }
        )
    ui.table(
        data=regular_history,
        caption=f"All available FPL seasons for {player_name}",
        key=f"regular_history_{player_id}",
        max_height=500,
    )
    st.caption(
        "Percentiles are calculated separately for each season. P100 is best "
        "for the metric, and per-90 percentiles require at least "
        f"{MIN_PER_90_MINUTES} minutes."
    )
    ui.table(
        data=percentile_table,
        caption=f"Metric values and percentiles by season for {player_name}",
        key=f"history_{player_id}",
        max_height=500,
    )


def render_profile_tab(
    player_name: str, selected_season: str, profile: pd.Series | None
) -> None:
    """Render archetype, quality tier, confidence and playing characteristics."""
    st.subheader("Production style")
    if profile is None:
        st.info(
            f"No published player-profile artifact is available for {selected_season}."
        )
        return

    status = str(profile.get("profile_status", "Data unavailable"))
    if status not in {"Established", "Provisional"}:
        st.info(
            f"{player_name}: {status}. The app will not infer a style from "
            "missing provider data or an undersized sample."
        )
        return

    reliability = pd.to_numeric(profile.get("reliability"), errors="coerce")
    cards = st.columns(4)
    cards[0].metric("Archetype", str(profile.get("production_archetype", "—")))
    cards[1].metric("Production tier", str(profile.get("production_tier", "—")))
    cards[2].metric("Primary strength", str(profile.get("primary_strength", "—")))
    cards[3].metric(
        "Profile reliability",
        "—" if pd.isna(reliability) else f"{reliability * 100:.0f}%",
        status,
    )

    dimensions = profile_dimensions(profile)
    if dimensions:
        with st.container(border=True):
            st.markdown("#### Production")
            st.caption("Compared with players in the same FPL position.")
            for dimension in dimensions:
                label = str(dimension["Dimension"])
                percentile = float(dimension["Percentile"])
                ui.progress(
                    value=percentile,
                    label=label,
                    show_value=True,
                    width="stretch",
                    key=(
                        f"profile_percentile_{selected_season}_"
                        f"{player_name}_{label}"
                    ),
                )

            average_percentile = sum(
                float(dimension["Percentile"]) for dimension in dimensions
            ) / len(dimensions)
            ui.separator(
                key=f"profile_percentile_separator_{player_name}_{selected_season}"
            )
            ui.progress(
                value=average_percentile,
                label="Average percentile across all dimensions",
                show_value=True,
                width="stretch",
                key=f"profile_average_percentile_{player_name}_{selected_season}",
            )

    characteristics = [
        value.strip()
        for value in str(profile.get("playing_characteristics", "")).split(";")
        if value.strip()
    ]
    if characteristics:
        st.markdown("**Playing characteristics**")
        for characteristic in characteristics:
            st.markdown(f"- {characteristic}")
    secondary = str(profile.get("secondary_strength", "")).strip()
    if secondary:
        st.caption(f"Secondary strength: {secondary}")
    st.caption(
        "Rates are shrunk toward the minutes-weighted positional mean before "
        "ranking. Established profiles require 900 minutes; 450–899 minutes "
        "are provisional. Goalkeepers are evaluated on shot stopping, sweeping, "
        "and distribution."
    )


def main() -> None:
    st.title("Fantasy Premier League Player Dashboard")

    leagues = discover_leagues()
    if not leagues:
        st.error(f"No processed FPL data was found in {FPL_ROOT}.")
        return

    league = st.sidebar.selectbox("League", leagues, key="fpl_league")
    players = load_historical_fpl(league, historical_fpl_version(league))

    if players.empty:
        st.warning(f"No historical player data was found for {league}.")
        return

    seasons = discover_seasons(league)
    saved_filters = st.session_state.get("player_filter_state", {})
    reset_version = st.session_state.get("player_filter_reset_version", 0)
    saved_season = saved_filters.get("season", seasons[0])
    if saved_season not in seasons:
        saved_season = seasons[0]

    with st.expander(
        "Find or switch player",
        expanded=not bool(
            st.session_state.get(f"historical_player_{reset_version}")
        ),
    ):
        season_column, team_column, category_column, price_column = st.columns(4)

        with season_column:
            selected_season = (
                ui.select(
                    "Season",
                    options=seasons,
                    value=saved_season,
                    key=f"player_filter_season_{reset_version}",
                )
                or saved_season
            )

        season_players = players.loc[
            players["season"].eq(selected_season)
        ].copy()
        positions = sorted(season_players["fpl_pos"].dropna().unique().tolist())
        teams = sorted(season_players["team"].dropna().unique().tolist())
        season_prices = (
            pd.to_numeric(season_players["now_cost"], errors="coerce") / 10
        ).dropna()

        position_options = ["All", *positions]
        saved_position = saved_filters.get("position", "All")
        if saved_position not in position_options:
            saved_position = "All"
        selected_position = ui.tabs(
            options=position_options,
            value=saved_position,
            label="Position",
            key=(
                f"player_filter_position_{selected_season}_{reset_version}"
            ),
        )
        selected_positions = (
            positions if selected_position == "All" else [selected_position]
        )

        with team_column:
            team_options = ["All teams", *teams]
            saved_team = saved_filters.get("team", "All teams")
            if saved_team not in team_options:
                saved_team = "All teams"
            selected_team = ui.select(
                "Team",
                options=team_options,
                value=saved_team,
                key=f"player_filter_team_{selected_season}_{reset_version}",
            )
            selected_teams = (
                teams if selected_team in {None, "All teams"} else [selected_team]
            )

        with category_column:
            price_category_options = ["All categories", *PRICE_CATEGORY_ORDER]
            saved_price_category = saved_filters.get(
                "price_category", "All categories"
            )
            if saved_price_category not in price_category_options:
                saved_price_category = "All categories"
            selected_price_category = ui.select(
                "Price category",
                options=price_category_options,
                value=saved_price_category,
                key=(
                    f"player_filter_price_category_"
                    f"{selected_season}_{reset_version}"
                ),
            ) or "All categories"

        with price_column:
            minimum_price = float(season_prices.min())
            maximum_price = float(season_prices.max())
            saved_price_range = saved_filters.get(
                "price_range", (minimum_price, maximum_price)
            )
            try:
                saved_minimum, saved_maximum = map(float, saved_price_range)
                saved_minimum = max(minimum_price, saved_minimum)
                saved_maximum = min(maximum_price, saved_maximum)
                if saved_minimum > saved_maximum:
                    raise ValueError
            except (TypeError, ValueError):
                saved_minimum, saved_maximum = minimum_price, maximum_price

            if minimum_price == maximum_price:
                st.number_input(
                    "Price (£m)",
                    value=minimum_price,
                    disabled=True,
                    key=f"player_filter_price_{selected_season}_{reset_version}",
                )
                selected_price_range = (minimum_price, maximum_price)
            else:
                selected_price_range = ui.slider(
                    "Price range (£m)",
                    min_value=minimum_price,
                    max_value=maximum_price,
                    value=(saved_minimum, saved_maximum),
                    step=0.1,
                    key=f"player_filter_price_{selected_season}_{reset_version}",
                )

        filtered_players = filter_player_pool(
            players,
            selected_season,
            selected_positions,
            selected_teams,
            selected_price_range,
            selected_price_category,
        )
        labels = build_player_labels(filtered_players)

        search_column, player_column, mode_column, reset_column = st.columns(4)
        with search_column:
            player_search = ui.input(
                "Search players",
                value=str(saved_filters.get("player_search", "")),
                key=f"player_search_{selected_season}_{reset_version}",
                placeholder="Search by player or team...",
            ).strip()

        if player_search:
            searched_players = search_player_pool(filtered_players, player_search)
            matching_ids = set(
                searched_players["player_id"].astype(str)
            )
            matching_labels = {
                player_id: label
                for player_id, label in labels.items()
                if player_id in matching_ids
            }
        else:
            matching_labels = labels

        with player_column:
            if matching_labels:
                selected_player_id = ui.select(
                    "Select a player to view",
                    options=list(matching_labels),
                    format_func=lambda player_id: matching_labels[player_id],
                    index=None,
                    key=f"historical_player_{reset_version}",
                    placeholder="Select a matching player...",
                )
            else:
                selected_player_id = None
                st.caption("No matching players")

        with mode_column:
            per_90 = ui.switch(
                "Per 90",
                value=bool(saved_filters.get("per_90", False)),
                key=f"player_metric_mode_{reset_version}",
            )
            st.caption("Per-90 metrics" if per_90 else "Total metrics")

        with reset_column:
            reset_filters = ui.button(
                "Reset filters",
                key=f"reset_player_filters_{reset_version}",
                variant="outline",
            )

    if reset_filters:
        st.session_state.pop("player_filter_state", None)
        st.session_state.pop("historical_player", None)
        st.session_state.pop(f"historical_player_{reset_version}", None)
        st.session_state["player_filter_reset_version"] = reset_version + 1
        st.rerun()

    st.session_state["player_filter_state"] = {
        "season": selected_season,
        "position": selected_position,
        "team": selected_team,
        "price_category": selected_price_category,
        "price_range": tuple(selected_price_range),
        "player_search": player_search,
        "per_90": per_90,
    }

    st.caption(
        f"Showing {len(matching_labels):,} of {len(labels):,} filtered players "
        f"in {selected_season}."
    )

    if not labels:
        st.warning("No players match the selected filters.")
        return

    if not matching_labels:
        st.warning("No players match the current search.")
        return

    if selected_player_id is None:
        st.info("Select a player to see their FPL history.")
        return

    history = build_player_history(players, selected_player_id, per_90=per_90)
    if history.empty:
        st.warning("No historical records were found for the selected player.")
        return
    player_name = history.iloc[0]["Player"]
    percentile_table = build_metric_percentile_table(history, per_90=per_90)

    selected_record = history.loc[history["Season"].eq(selected_season)].iloc[0]
    raw_rows = season_players.loc[
        season_players["player_id"].astype("string").eq(str(selected_player_id))
    ]
    if raw_rows.empty:
        st.warning("The selected player is missing from the season roster.")
        return
    raw_record = raw_rows.iloc[0]

    gameweek_path = fpl_gameweeks_path(league, selected_season)
    gameweeks = load_gameweek_fpl(
        league, selected_season, file_version(gameweek_path)
    )
    gameweek_history = build_player_gameweek_history(
        gameweeks, selected_player_id, player_name
    )

    forecast_path = (
        latest_forecast_path(PREDICTIONS_ROOT / "expected_points", selected_season)
        if selected_season == seasons[0]
        else None
    )
    if forecast_path is None:
        player_forecast = pd.DataFrame()
    else:
        forecasts = load_expected_points(
            str(forecast_path), file_version(forecast_path)
        )
        player_forecast = prepare_player_forecast(
            forecasts, selected_player_id, season=selected_season
        )

    profile_path = fpl_player_profiles_path(league, selected_season)
    profiles = load_player_profiles(str(profile_path), file_version(profile_path))
    season_index = seasons.index(selected_season)
    previous_season = (
        seasons[season_index + 1] if season_index + 1 < len(seasons) else None
    )
    if previous_season is None:
        previous_profile_path = None
        previous_profiles = pd.DataFrame()
    else:
        previous_profile_path = fpl_player_profiles_path(league, previous_season)
        previous_profiles = load_player_profiles(
            str(previous_profile_path), file_version(previous_profile_path)
        )
    selected_profile, selected_profile_season, profile_is_carryover = (
        select_profile_snapshot(
            profiles,
            previous_profiles,
            str(selected_player_id),
            current_season=selected_season,
            previous_season=previous_season,
        )
    )
    selected_profile_trend = profile_trend(
        profiles, previous_profiles, str(selected_player_id)
    )

    details_path = fpl_raw_players_path(league, selected_season)
    raw_details = load_raw_player_details(
        str(details_path), file_version(details_path)
    )
    fpl_element_id = pd.to_numeric(raw_record.get("fpl_element_id"), errors="coerce")
    if raw_details.empty or "id" not in raw_details or pd.isna(fpl_element_id):
        selected_details = None
    else:
        detail_rows = raw_details.loc[raw_details["id"].eq(fpl_element_id)]
        selected_details = None if detail_rows.empty else detail_rows.iloc[0]

    teams_path = fpl_raw_teams_path(league, selected_season)
    raw_teams = load_raw_teams(str(teams_path), file_version(teams_path))
    image_config = load_player_image_config(
        file_version(PLAYER_IMAGE_CONFIG_PATH)
    )
    asset_version = str(image_config.get("asset_version", "25"))
    team_badges: dict[int, str] = {}
    if {"id", "code"}.issubset(raw_teams.columns):
        for team_row in raw_teams[["id", "code"]].dropna().itertuples(index=False):
            badge = team_badge_url(team_row.code, asset_version=asset_version)
            if badge:
                team_badges[int(team_row.id)] = badge

    metadata_path = fpl_fixture_metadata_path(league, selected_season)
    raw_fixtures_path = fpl_raw_fixtures_path(league, selected_season)
    fixture_schedule = load_fixture_schedule(
        str(metadata_path),
        str(raw_fixtures_path),
        file_version(metadata_path),
        file_version(raw_fixtures_path),
    )
    player_fixtures = prepare_player_fixtures(
        fixture_schedule,
        raw_record.get("fpl_team_numeric_id"),
        forecast=player_forecast,
        limit=5,
    )
    alternatives = comparable_players(
        season_players, str(selected_player_id), price_tolerance=5, limit=3
    )

    def updated_label(path: Path | None) -> str:
        if path is None or not path.is_file():
            return "unavailable"
        return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%d %b %Y %H:%M")

    profile_freshness_path = (
        previous_profile_path if profile_is_carryover else profile_path
    )
    data_freshness = {
        "roster": updated_label(fpl_season_path(league, selected_season)),
        "fixtures": updated_label(raw_fixtures_path),
        "profile": updated_label(profile_freshness_path),
        "forecast": updated_label(forecast_path),
    }

    overview_tab, performance_tab, forecast_tab, history_tab, profile_tab = st.tabs(
        ["Overview", "Performance", "Forecast", "History", "Profile"]
    )
    with overview_tab:
        render_overview_tab(
            player_name,
            selected_season,
            selected_record,
            raw_record,
            gameweek_history,
            player_fixtures,
            player_forecast,
            selected_profile,
            selected_profile_season,
            profile_is_carryover,
            selected_profile_trend,
            selected_details,
            team_badges,
            alternatives,
            data_freshness,
        )
    with performance_tab:
        render_performance_tab(
            player_name, selected_season, gameweek_history, percentile_table
        )
    with forecast_tab:
        render_forecast_tab(player_name, selected_season, player_forecast)
    with history_tab:
        render_history_tab(
            player_name,
            selected_player_id,
            history,
            percentile_table,
            per_90=per_90,
        )
    with profile_tab:
        if profile_is_carryover:
            st.info(
                f"Showing the {selected_profile_season} profile as a preseason "
                "baseline because current-season evidence is not yet sufficient."
            )
        render_profile_tab(
            player_name,
            selected_profile_season or selected_season,
            selected_profile,
        )


if __name__ == "__main__":
    main()
