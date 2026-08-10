import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit_shadcn_ui as ui

from apps.fpl.catalog import (
    FPL_ROOT,
    PRICE_CATEGORY_CONFIG_PATH,
    discover_leagues as catalog_discover_leagues,
    discover_seasons as catalog_discover_seasons,
    file_version,
    fpl_gameweeks_path,
    fpl_season_path,
)


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

    st.subheader("Filters")
    with st.container(border=True):
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
                value=bool(saved_filters.get("per_90", True)),
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
    selected_points = pd.to_numeric(selected_record["Points"], errors="coerce")
    selected_goals = pd.to_numeric(selected_record["Goals"], errors="coerce")
    selected_assists = pd.to_numeric(selected_record["Assists"], errors="coerce")
    selected_saves = pd.to_numeric(
        selected_record.get("Saves", pd.NA), errors="coerce"
    )
    selected_save_pct = pd.to_numeric(
        selected_record.get("Save %", pd.NA), errors="coerce"
    )
    selected_defcon = pd.to_numeric(
        selected_record.get("Def Con", pd.NA), errors="coerce"
    )
    player_team = selected_record["Team"]
    player_price = selected_record["Price"]
    player_position = selected_record["Position"]
    is_goalkeeper = is_goalkeeper_position(player_position)
    player_season = selected_record["Season"]
    metric_suffix = " /90" if per_90 else ""

    st.markdown(
        (
            '<div style="display:flex;align-items:baseline;gap:0.75rem;'
            'flex-wrap:wrap;margin-bottom:0.5rem">'
            f'<h3 style="margin:0">{escape(str(player_name))}</h3>'
            '<span style="color:var(--text-color);opacity:0.65;font-size:0.9rem">'
            f'{escape(str(player_team))} &bull; {escape(str(player_position))}'
            "</span></div>"
        ),
        unsafe_allow_html=True,
    )

    metrics = st.columns(5)
    with metrics[0]:
        render_price_metric_card(
            player_price,
            selected_record.get("Price Category", "Uncategorized"),
        )
    with metrics[1]:
        render_metric_card(
            f"{player_season} points{metric_suffix}",
            format_metric_card_value(selected_points, per_90),
            format_metric_percentile_delta(selected_record, "Points"),
        )
    with metrics[2]:
        if is_goalkeeper:
            render_metric_card(
                f"Saves{metric_suffix}",
                format_metric_card_value(selected_saves, per_90),
                format_metric_percentile_delta(selected_record, "Saves"),
            )
        else:
            render_metric_card(
                f"Goals{metric_suffix}",
                format_metric_card_value(selected_goals, per_90),
                format_metric_percentile_delta(selected_record, "Goals"),
            )
    with metrics[3]:
        if is_goalkeeper:
            render_metric_card(
                "Save %",
                format_percentage(selected_save_pct),
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

    gameweek_path = fpl_gameweeks_path(league, selected_season)
    gameweeks = load_gameweek_fpl(
        league, selected_season, file_version(gameweek_path)
    )
    gameweek_history = build_player_gameweek_history(
        gameweeks, selected_player_id, player_name
    )
    if gameweek_history.empty:
        st.info(
            f"No gameweek data is available for {player_name} in "
            f"{selected_season}."
        )
    else:
        stacked_gameweek_history = gameweek_history.melt(
            id_vars=[
                "GW",
                "Total FPL points",
                "Opponent",
                "Minutes",
                "Fixtures",
            ],
            value_vars=GAMEWEEK_POINT_COMPONENTS,
            var_name="Contribution",
            value_name="Points",
        )
        gameweek_figure = px.bar(
            stacked_gameweek_history,
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
            title=(
                f"{player_name} points contributions by gameweek "
                f"— {selected_season}"
            ),
        )
        gameweek_figure.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "GW: %{x}<br>"
                "Points: %{y:.0f}<br>"
                "Total FPL points: %{customdata[1]:.0f}<br>"
                "Opponent: %{customdata[2]}<br>"
                "Minutes: %{customdata[3]:.0f}<br>"
                "Fixtures: %{customdata[4]}<extra></extra>"
            )
        )
        gameweek_figure.update_layout(
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
            xaxis_title="Gameweek",
            yaxis_title="FPL points from selected contributions",
            legend_title_text="Contribution",
        )
        gameweek_figure.update_xaxes(dtick=1)
        st.plotly_chart(gameweek_figure, width="stretch")

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

    regular_history_columns = [
        label for label in PLAYER_HISTORY_COLUMNS.values() if label in history.columns
    ]
    regular_history = history[regular_history_columns].copy()
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
        caption=(
            f"All available FPL seasons for {player_name} "
            f"({'per 90' if per_90 else 'totals'})"
        ),
        key=f"regular_history_{selected_player_id}",
        max_height=500,
    )

    st.caption(
        "Percentiles are calculated separately for each season. P100 is best "
        "for the metric (lower is better for adverse outcomes), and ties share "
        "their average percentile. Per-90 percentiles require at least "
        f"{MIN_PER_90_MINUTES} minutes."
    )
    ui.table(
        data=percentile_table,
        caption=f"Metric values and percentiles by season for {player_name}",
        key=f"history_{selected_player_id}",
        max_height=500,
    )


if __name__ == "__main__":
    main()
