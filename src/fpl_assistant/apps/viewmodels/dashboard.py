"""Pure view models shared by the FPL hub, compare, team and player pages."""

from __future__ import annotations

import json

import pandas as pd


FAMILY_ORDER = {
    family: index
    for index, family in enumerate(
        [
            "Production Composite",
            "Production Style",
            "Production",
            "Usage",
            "Return Shape",
            "Venue Behaviour",
            "Value Historical",
            "Risk Badge",
        ]
    )
}

FAMILY_DESCRIPTIONS = {
    "Production Composite": "The player's combined attacking or defensive role.",
    "Production Style": "The individual production traits driving the combined role.",
    "Production": "Position-specific output such as clean sheets or saves.",
    "Usage": "How consistently the player starts and accumulates meaningful minutes.",
    "Return Shape": "Whether FPL returns tend to arrive steadily or in bursts.",
    "Venue Behaviour": "Whether production changes materially between home and away matches.",
    "Value Historical": "Historical points return relative to FPL price.",
    "Risk Badge": "A material reliability or points-risk signal supported by the evidence.",
}


def archetype_reason(row: pd.Series | dict[str, object], *, limit: int = 3) -> str:
    """Summarise the strongest persisted calculation components for a label."""
    raw = row.get("component_scores") if hasattr(row, "get") else None
    if raw is None:
        return ""
    if not isinstance(raw, (dict, list, str)):
        if not pd.api.types.is_scalar(raw) or bool(pd.isna(raw)):
            return ""
    elif isinstance(raw, str) and not raw.strip():
        return ""
    try:
        values = json.loads(str(raw)) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(values, dict):
        return ""
    if str(row.get("family", "")) == "Usage":
        usage_parts: list[str] = []
        usage_fields = (
            ("expected_minutes", "Avg min / available team match", False),
            ("start_probability", "Historical start rate", True),
            ("cameo_probability", "Historical cameo rate when benched", True),
        )
        for field, label, as_percentage in usage_fields:
            number = pd.to_numeric(values.get(field), errors="coerce")
            if bool(pd.notna(number)):
                display = f"{float(number) * 100:.1f}%" if as_percentage else f"{float(number):.1f}"
                usage_parts.append(f"{label}: {display}")
        return " · ".join(usage_parts[:limit])
    ranked: list[tuple[str, float]] = []
    for name, value in values.items():
        # Calculation payloads also persist structural metadata such as
        # ``window_weights`` dictionaries and ``required`` archetype lists.
        # Only scalar numeric components belong in the concise UI reason.
        if isinstance(value, bool) or not pd.api.types.is_scalar(value):
            continue
        number = pd.to_numeric(value, errors="coerce")
        if bool(pd.notna(number)):
            ranked.append((str(name), float(number)))
    ranked.sort(key=lambda item: abs(item[1]), reverse=True)
    return " · ".join(
        f"{name.replace('_', ' ').title()}: {value:.1f}" for name, value in ranked[:limit]
    )


def active_archetype_tags(
    archetypes: pd.DataFrame, *, overview: bool = False
) -> pd.DataFrame:
    """Return active labels in stable family/score order."""
    required = {"family", "display_name", "active_label"}
    if archetypes.empty or not required.issubset(archetypes.columns):
        return pd.DataFrame(columns=list(archetypes.columns))

    values = archetypes["active_label"]
    mask = values.eq(True) | values.astype("string").str.lower().eq("true")
    active = archetypes.loc[mask].copy()
    if active.empty:
        return active
    active["_score"] = pd.to_numeric(active.get("score_0_100"), errors="coerce")
    active["_family_order"] = active["family"].map(FAMILY_ORDER).fillna(len(FAMILY_ORDER))
    active = active.sort_values(
        ["_family_order", "_score", "display_name"],
        ascending=[True, False, True],
        kind="stable",
    )

    if overview:
        selected: list[pd.DataFrame] = []
        production = active.loc[active["family"].eq("Production Composite")]
        if production.empty:
            production = active.loc[active["family"].isin(["Production Style", "Production"])]
        if not production.empty:
            selected.append(production.head(1))
        for family in ("Usage", "Risk Badge"):
            rows = active.loc[active["family"].eq(family)]
            if not rows.empty:
                selected.append(rows.head(1))
        active = pd.concat(selected, ignore_index=True) if selected else active.head(0)

    archetype_key = "archetype_id" if "archetype_id" in active else "display_name"
    dedupe = [archetype_key]
    if "player_id" in active:
        # Season-wide discovery frames contain the same archetype for many
        # players. De-duplicate repeated records for a player, never the tag
        # across the whole league.
        dedupe.insert(0, "player_id")
    return (
        active.drop_duplicates(subset=dedupe, keep="first")
        .drop(columns=["_score", "_family_order"], errors="ignore")
        .reset_index(drop=True)
    )


def archetype_wide(archetypes: pd.DataFrame) -> pd.DataFrame:
    """Return one row per player with the most relevant active family labels."""
    active = active_archetype_tags(archetypes)
    if active.empty or "player_id" not in active:
        return pd.DataFrame(columns=["player_id"])
    active["_score"] = pd.to_numeric(active.get("score_0_100"), errors="coerce")
    active = active.sort_values("_score", ascending=False, kind="stable")
    active = active.drop_duplicates(["player_id", "family"], keep="first")
    labels = active.pivot(index="player_id", columns="family", values="display_name")
    labels = labels.rename(
        columns={
            "Production Composite": "Production profile",
            "Production Style": "Production component",
            "Production": "Position production",
            "Usage": "Usage",
            "Return Shape": "Return shape",
            "Venue Behaviour": "Venue behaviour",
            "Value Historical": "Value profile",
            "Risk Badge": "Risk",
        }
    )
    return labels.reset_index()


def archetype_changes(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Describe active-label additions and removals between two snapshots."""
    if current.empty or previous.empty:
        return pd.DataFrame()
    required = {"player_id", "archetype_id", "display_name", "active_label"}
    if not required.issubset(current.columns) or not required.issubset(previous.columns):
        return pd.DataFrame()
    current_active = active_archetype_tags(current)
    previous_active = active_archetype_tags(previous)
    current_keys = set(zip(current_active["player_id"].astype(str), current_active["archetype_id"].astype(str)))
    previous_keys = set(zip(previous_active["player_id"].astype(str), previous_active["archetype_id"].astype(str)))
    rows: list[dict[str, object]] = []
    for change, keys, source in (
        ("Activated", current_keys - previous_keys, current_active),
        ("Removed", previous_keys - current_keys, previous_active),
    ):
        lookup = source.copy()
        lookup["player_id"] = lookup["player_id"].astype(str)
        lookup["archetype_id"] = lookup["archetype_id"].astype(str)
        lookup = lookup.set_index(["player_id", "archetype_id"])
        for key in sorted(keys):
            row = lookup.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rows.append(
                {
                    "player_id": key[0],
                    "Change": change,
                    "Archetype": row.get("display_name"),
                    "Family": row.get("family"),
                    "Confidence": row.get("confidence_band"),
                }
            )
    output = pd.DataFrame(rows)
    if output.empty or players is None or players.empty or "player_id" not in players:
        return output
    names = players[[column for column in ["player_id", "name", "team"] if column in players]].copy()
    names["player_id"] = names["player_id"].astype(str)
    names = names.drop_duplicates("player_id", keep="last")
    return output.merge(names, on="player_id", how="left", validate="many_to_one")


def enrich_current_players(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    archetypes: pd.DataFrame,
) -> pd.DataFrame:
    """Attach historical baselines and active archetype summaries to a roster."""
    if current.empty or "player_id" not in current:
        return pd.DataFrame()
    output = current.copy()
    output["player_id"] = output["player_id"].astype("string")
    if not previous.empty and "player_id" in previous:
        baseline_columns = [
            column
            for column in [
                "player_id", "total_points", "minutes", "goals_scored", "assists",
                "clean_sheets", "xg", "xa", "defcon",
            ]
            if column in previous
        ]
        baseline = previous[baseline_columns].copy()
        baseline["player_id"] = baseline["player_id"].astype("string")
        baseline = baseline.drop_duplicates("player_id", keep="last").rename(
            columns={column: f"baseline_{column}" for column in baseline_columns if column != "player_id"}
        )
        output = output.merge(baseline, on="player_id", how="left", validate="one_to_one")
    wide = archetype_wide(archetypes)
    if not wide.empty:
        wide["player_id"] = wide["player_id"].astype("string")
        output = output.merge(wide, on="player_id", how="left", validate="one_to_one")
    return output


def player_watchlist(players: pd.DataFrame, *, limit: int = 8) -> pd.DataFrame:
    """Rank a current roster using live totals or an explicit historical baseline."""
    if players.empty:
        return pd.DataFrame()
    work = players.copy()
    live_minutes = pd.to_numeric(work.get("minutes"), errors="coerce").fillna(0)
    has_live = live_minutes.gt(0).any()
    points_column = "total_points" if has_live else "baseline_total_points"
    minutes_column = "minutes" if has_live else "baseline_minutes"
    work["_points"] = pd.to_numeric(work.get(points_column), errors="coerce")
    work["_minutes"] = pd.to_numeric(work.get(minutes_column), errors="coerce")
    work["_price"] = pd.to_numeric(work.get("now_cost"), errors="coerce").div(10)
    work["_value"] = work["_points"].div(work["_price"].where(work["_price"].gt(0)))
    work = work.sort_values(["_points", "_value", "_minutes"], ascending=False, na_position="last")
    columns = [
        column for column in [
            "player_id", "name", "team", "fpl_pos", "status", "now_cost",
            "selected_by_percent", "Production profile", "Usage", "Risk",
        ] if column in work
    ]
    result = work.head(limit)[columns].copy()
    result["Points"] = work.head(limit)["_points"].values
    result["Value"] = work.head(limit)["_value"].values
    result["Evidence"] = "Current season" if has_live else "Previous-season baseline"
    return result.reset_index(drop=True)


def forecast_watchlist(
    players: pd.DataFrame, forecasts: pd.DataFrame, *, limit: int = 12
) -> pd.DataFrame:
    """Rank players by a published forecast window with minutes and value context."""
    if players.empty or forecasts.empty or "player_id" not in forecasts:
        return pd.DataFrame()
    work = forecasts.copy()
    work["player_id"] = work["player_id"].astype(str)
    work["xPts"] = pd.to_numeric(work.get("xPts"), errors="coerce")
    work["pred_minutes"] = pd.to_numeric(work.get("pred_minutes"), errors="coerce")
    summary = work.groupby("player_id", as_index=False).agg(
        **{
            "Forecast xPts": ("xPts", "sum"),
            "Forecast minutes": ("pred_minutes", "sum"),
            "Fixtures": ("gw_orig", "nunique"),
        }
    )
    roster = players.copy()
    roster["player_id"] = roster["player_id"].astype(str)
    output = roster.merge(summary, on="player_id", how="inner", validate="one_to_one")
    output["Price"] = pd.to_numeric(output.get("now_cost"), errors="coerce").div(10)
    output["Forecast value"] = output["Forecast xPts"].div(output["Price"].where(output["Price"].gt(0)))
    output = output.sort_values(
        ["Forecast xPts", "Forecast minutes", "Forecast value"], ascending=False, na_position="last"
    )
    columns = [
        column for column in [
            "player_id", "name", "team", "fpl_pos", "status", "Price",
            "selected_by_percent", "Production profile", "Usage", "Risk",
            "Forecast xPts", "Forecast minutes", "Forecast value", "Fixtures",
        ] if column in output
    ]
    return output.head(limit)[columns].reset_index(drop=True)


def fixture_rows(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Expand FPL fixtures to one row per team with venue and difficulty."""
    required = {"id", "team_h", "team_a"}
    if fixtures.empty or not required.issubset(fixtures.columns):
        return pd.DataFrame()
    names = {}
    short_names = {}
    if not teams.empty and "id" in teams:
        ids = pd.to_numeric(teams["id"], errors="coerce")
        names = dict(zip(ids, teams.get("name", teams.get("short_name", ids))))
        short_names = dict(zip(ids, teams.get("short_name", teams.get("name", ids))))
    rows: list[dict[str, object]] = []
    for fixture in fixtures.to_dict("records"):
        home = pd.to_numeric(fixture.get("team_h"), errors="coerce")
        away = pd.to_numeric(fixture.get("team_a"), errors="coerce")
        if pd.isna(home) or pd.isna(away):
            continue
        common = {
            "fixture_id": fixture.get("id"),
            "GW": fixture.get("event"),
            "Kickoff": fixture.get("kickoff_time"),
            "finished": fixture.get("finished", False),
        }
        rows.extend(
            [
                {
                    **common, "team_numeric_id": int(home), "Team": short_names.get(home, names.get(home, home)),
                    "Opponent": short_names.get(away, names.get(away, away)), "Venue": "H",
                    "FDR": fixture.get("team_h_difficulty"),
                },
                {
                    **common, "team_numeric_id": int(away), "Team": short_names.get(away, names.get(away, away)),
                    "Opponent": short_names.get(home, names.get(home, home)), "Venue": "A",
                    "FDR": fixture.get("team_a_difficulty"),
                },
            ]
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["GW"] = pd.to_numeric(output["GW"], errors="coerce")
    output["FDR"] = pd.to_numeric(output["FDR"], errors="coerce")
    output["Kickoff"] = pd.to_datetime(output["Kickoff"], errors="coerce", utc=True)
    return output.sort_values(["GW", "Kickoff", "fixture_id"], kind="stable").reset_index(drop=True)


def team_fixture_outlook(rows: pd.DataFrame, *, fixtures_per_team: int = 5) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    upcoming = rows.loc[~rows["finished"].fillna(False).astype(bool)].copy()
    upcoming = upcoming.groupby("Team", group_keys=False).head(fixtures_per_team)
    summary = (
        upcoming.groupby("Team", as_index=False)
        .agg(
            **{
                "Average FDR": ("FDR", "mean"),
                "Next opponents": ("Opponent", lambda values: " · ".join(values.astype(str))),
                "Fixtures": ("fixture_id", "count"),
            }
        )
        .sort_values(["Average FDR", "Team"], kind="stable")
    )
    return summary.reset_index(drop=True)


def comparison_table(players: pd.DataFrame) -> pd.DataFrame:
    """Build an exact-value comparison table for two to four enriched players."""
    if players.empty:
        return pd.DataFrame()
    rows = []
    for row in players.to_dict("records"):
        live = pd.to_numeric(row.get("minutes"), errors="coerce")
        use_baseline = pd.isna(live) or float(live) == 0
        prefix = "baseline_" if use_baseline else ""
        price = pd.to_numeric(row.get("now_cost"), errors="coerce")
        rows.append(
            {
                "Player": row.get("name"),
                "Team": row.get("team"),
                "Position": row.get("fpl_pos"),
                "Price": None if pd.isna(price) else float(price) / 10,
                "Points": pd.to_numeric(row.get(f"{prefix}total_points"), errors="coerce"),
                "Minutes": pd.to_numeric(row.get(f"{prefix}minutes"), errors="coerce"),
                "Goals": pd.to_numeric(row.get(f"{prefix}goals_scored"), errors="coerce"),
                "Assists": pd.to_numeric(row.get(f"{prefix}assists"), errors="coerce"),
                "xG": pd.to_numeric(row.get(f"{prefix}xg"), errors="coerce"),
                "xA": pd.to_numeric(row.get(f"{prefix}xa"), errors="coerce"),
                "Ownership": pd.to_numeric(row.get("selected_by_percent"), errors="coerce"),
                "Production profile": row.get("Production profile"),
                "Usage": row.get("Usage"),
                "Risk": row.get("Risk"),
                "Evidence": "Previous season" if use_baseline else "Current season",
            }
        )
    return pd.DataFrame(rows)
