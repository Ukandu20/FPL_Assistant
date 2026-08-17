from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class CanonicalArchetypeInputs:
    player_matches: pd.DataFrame
    team_matches: pd.DataFrame
    player_values: pd.DataFrame
    field_provenance: pd.DataFrame
    build_audit: pd.DataFrame


def discover_joinable_seasons(
    league: str = "ENG-Premier League",
    *,
    processed_root: Path = PROCESSED_ROOT,
) -> list[str]:
    """Find seasons with FPL match rows plus Understat and WhoScored evidence."""
    fpl_root = processed_root / "fpl" / league
    if not fpl_root.is_dir():
        return []
    seasons: list[str] = []
    for season_path in fpl_root.iterdir():
        season = season_path.name
        required = (
            season_path / "gws" / "merged_gws.csv",
            processed_root / "understat" / league / season / "player_match.csv",
            processed_root / "whoscored" / league / season / "player_match" / "summary.csv",
        )
        if all(path.is_file() for path in required):
            seasons.append(season)
    return sorted(seasons)


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    source = frame[column] if column in frame else pd.Series(default, index=frame.index)
    return pd.to_numeric(source, errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    source = frame[column] if column in frame else pd.Series(False, index=frame.index)
    if pd.api.types.is_bool_dtype(source):
        return source.fillna(False).astype(bool)
    return _numeric(frame, column, 0).fillna(0).gt(0)


def _normalize_provider_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("Int64").astype("string")


def _fpl_match_id(frame: pd.DataFrame) -> pd.Series:
    for column in ("match_id", "game_id"):
        if column in frame:
            return frame[column].astype("string")
    raise KeyError("FPL gameweek input has neither match_id nor game_id")


def _matchup_key(home: object, away: object) -> str:
    return f"{str(home).strip().upper()}|{str(away).strip().upper()}"


def _fpl_matchup_map(frame: pd.DataFrame) -> dict[str, str]:
    match_ids = _fpl_match_id(frame)
    is_home = _boolean(frame, "was_home")
    team = frame["team"].astype("string").str.upper()
    opponent = frame["opp_code"].astype("string").str.upper()
    home = team.where(is_home, opponent)
    away = opponent.where(is_home, team)
    matches = pd.DataFrame({"match_id": match_ids, "home": home, "away": away})
    matches = matches.dropna().drop_duplicates()
    duplicates = matches.groupby(["home", "away"])["match_id"].nunique()
    ambiguous = set(duplicates[duplicates.gt(1)].index)
    return {
        _matchup_key(row.home, row.away): str(row.match_id)
        for row in matches.itertuples(index=False)
        if (row.home, row.away) not in ambiguous
    }


def _provider_match_ids(
    frame: pd.DataFrame,
    *,
    fpl_match_ids: set[str],
    matchup_map: dict[str, str],
) -> pd.Series:
    if "match_id" in frame:
        existing = frame["match_id"].astype("string")
        if existing.isin(fpl_match_ids).mean() >= 0.95:
            return existing
    if "game" not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    pairs = frame["game"].astype("string").str.split(" - ", n=1, expand=True)
    if pairs.shape[1] != 2:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    keys = [
        _matchup_key(home, away)
        for home, away in zip(pairs.iloc[:, 0], pairs.iloc[:, 1])
    ]
    return pd.Series(keys, index=frame.index, dtype="string").map(matchup_map)


def _deduplicate_fpl(
    frame: pd.DataFrame, season: str
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    work = frame.copy()
    work["_match_id"] = _fpl_match_id(work)
    key = ["_match_id", "player_id"]
    duplicated = work.duplicated(key, keep=False)
    audit: list[dict[str, object]] = []
    ambiguous_keys: set[tuple[str, str]] = set()
    for values, group in work.loc[duplicated].groupby(key, sort=False):
        identity_columns = [column for column in ("name", "team") if column in group]
        ambiguous = any(group[column].astype(str).nunique() > 1 for column in identity_columns)
        if ambiguous:
            ambiguous_keys.add((str(values[0]), str(values[1])))
        audit.append(
            {
                "season": season,
                "audit_type": "ambiguous_identity_excluded" if ambiguous else "duplicate_resolved",
                "match_id": str(values[0]),
                "player_id": str(values[1]),
                "rows": int(len(group)),
                "details": "; ".join(
                    f"{column}={','.join(sorted(group[column].dropna().astype(str).unique()))}"
                    for column in identity_columns
                ),
            }
        )
    if ambiguous_keys:
        keys = pd.MultiIndex.from_frame(work[key].astype(str))
        work = work.loc[~keys.isin(ambiguous_keys)].copy()
    work["_minutes_sort"] = _numeric(work, "minutes", -1).fillna(-1)
    work["_points_sort"] = _numeric(work, "total_points", -1).fillna(-1)
    work = (
        work.sort_values(key + ["_minutes_sort", "_points_sort"], kind="stable")
        .drop_duplicates(key, keep="last")
        .drop(columns=["_minutes_sort", "_points_sort"])
    )
    return work, audit


def _fixture_difficulty(
    frame: pd.DataFrame,
    *,
    league: str,
    season: str,
    raw_root: Path,
) -> pd.Series:
    path = raw_root / "fpl" / league / season / "season" / "fixtures.csv"
    if "fixture" not in frame or not path.is_file():
        return pd.Series(np.nan, index=frame.index)
    fixtures = pd.read_csv(path, low_memory=False)
    required = {"id", "team_h_difficulty", "team_a_difficulty"}
    if not required.issubset(fixtures):
        return pd.Series(np.nan, index=frame.index)
    fixture_map = fixtures.set_index("id")
    ids = _numeric(frame, "fixture").astype("Int64")
    home = ids.map(fixture_map["team_h_difficulty"])
    away = ids.map(fixture_map["team_a_difficulty"])
    difficulty = home.where(_boolean(frame, "was_home"), away)
    return (pd.to_numeric(difficulty, errors="coerce") - 1.0) / 4.0


def _canonical_fpl_rows(
    frame: pd.DataFrame,
    *,
    league: str,
    season: str,
    raw_root: Path,
) -> pd.DataFrame:
    is_home = _boolean(frame, "was_home")
    position = frame.get("fpl_pos", frame.get("position")).astype("string").str.upper()
    position = position.replace({"GK": "GKP"})
    goals = _numeric(frame, "goals_scored", 0).fillna(0)
    assists = _numeric(frame, "assists", 0).fillna(0)
    clean = _numeric(frame, "clean_sheets", 0).fillna(0)
    saves = _numeric(frame, "saves", 0).fillna(0)
    return_event = goals.gt(0) | assists.gt(0)
    return_event |= position.isin(["DEF", "GKP"]) & clean.gt(0)
    return_event |= position.eq("GKP") & saves.ge(3)
    output = pd.DataFrame(
        {
            "match_id": frame["_match_id"].astype("string"),
            "player_id": frame["player_id"].astype("string"),
            "team_id": frame["team_id"].astype("string"),
            "opponent_id": frame.get("opp_id", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "season": str(season),
            "gameweek": _numeric(frame, "round"),
            "kickoff_utc": pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce"),
            "fpl_position": position,
            "historical_fpl_position": position,
            "minutes": _numeric(frame, "minutes", 0).fillna(0),
            "started": _boolean(frame, "starts"),
            "named_on_bench": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "availability_status": "unknown",
            "availability_reason": pd.NA,
            "fpl_points": _numeric(frame, "total_points"),
            "price": _numeric(frame, "value"),
            "fpl_goals": goals,
            "fpl_assists": assists,
            "clean_sheet": clean.gt(0),
            "goals_conceded": _numeric(frame, "goals_conceded"),
            "xga": _numeric(frame, "expected_goals_conceded"),
            "fpl_saves": saves,
            "yellow_cards": _numeric(frame, "yellow_cards", 0).fillna(0),
            "red_cards": _numeric(frame, "red_cards", 0).fillna(0),
            "second_yellow_cards": 0.0,
            "penalties_saved": _numeric(frame, "penalties_saved"),
            "is_home": is_home,
            "venue": is_home.map({True: "Home", False: "Away"}),
            "return_event": return_event,
            "matchup_difficulty": _fixture_difficulty(
                frame, league=league, season=season, raw_root=raw_root
            ),
            "fpl_fixture_id": _numeric(frame, "fixture"),
        }
    )
    return output


def _understat_shot_evidence(
    season: str,
    *,
    raw_root: Path,
) -> pd.DataFrame:
    provider_season = season.split("-")[0]
    path = raw_root / "understat" / "EPL" / provider_season / "shot_events.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["game_id", "understat_player_id", "npxg", "non_penalty_goals"])
    shots = pd.read_csv(path, low_memory=False)
    if not {"game_id", "player_id", "xg"}.issubset(shots):
        return pd.DataFrame(columns=["game_id", "understat_player_id", "npxg", "non_penalty_goals"])
    situation = shots.get(
        "situation", shots.get("situation_raw", pd.Series("", index=shots.index))
    ).astype("string").str.lower()
    result = shots.get(
        "result", shots.get("result_raw", pd.Series("", index=shots.index))
    ).astype("string").str.lower()
    shots["_non_penalty_xg"] = _numeric(shots, "xg", 0).where(
        ~situation.str.contains("penalty", na=False), 0.0
    )
    shots["_non_penalty_goal"] = (
        result.str.contains("goal", na=False)
        & ~situation.str.contains("penalty", na=False)
    ).astype(int)
    shots["understat_player_id"] = _normalize_provider_id(shots["player_id"])
    return (
        shots.groupby(["game_id", "understat_player_id"], as_index=False)
        .agg(npxg=("_non_penalty_xg", "sum"), non_penalty_goals=("_non_penalty_goal", "sum"))
    )


def _understat_player_metrics(
    frame: pd.DataFrame,
    *,
    season: str,
    matchup_map: dict[str, str],
    fpl_match_ids: set[str],
    raw_root: Path,
) -> pd.DataFrame:
    work = frame.copy()
    work["match_id"] = _provider_match_ids(
        work, fpl_match_ids=fpl_match_ids, matchup_map=matchup_map
    )
    work["understat_player_id"] = _normalize_provider_id(
        work.get("understat_player_id", work["player_id"])
    )
    shots = _understat_shot_evidence(season, raw_root=raw_root)
    work = work.merge(
        shots,
        on=["game_id", "understat_player_id"],
        how="left",
        validate="many_to_one",
    )
    # A matched Understat appearance with no shot event has genuine zero shot output.
    work[["npxg", "non_penalty_goals"]] = work[
        ["npxg", "non_penalty_goals"]
    ].fillna(0.0)
    output = work[["match_id", "player_id", "npxg", "non_penalty_goals", "xa"]].copy()
    output["player_id"] = output["player_id"].astype("string")
    return output.dropna(subset=["match_id"]).drop_duplicates(["match_id", "player_id"])


def _who_scored_metrics(root: Path) -> pd.DataFrame:
    tables: list[tuple[str, dict[str, str]]] = [
        (
            "summary.csv",
            {
                "shots_on_target": "shots_on_target",
                "key_passes": "key_passes",
                "big_chances_created": "big_chances_created",
                "shot_creating_actions": "shot_creating_actions",
            },
        ),
        ("shooting.csv", {"shots_box": "shots_in_box"}),
        (
            "defense.csv",
            {
                "tackles_won": "tackles_won",
                "interceptions": "interceptions",
                "clearances": "clearances",
                "blocks": "blocks",
                "recoveries": "recoveries",
            },
        ),
        (
            "keepers.csv",
            {
                "shots_on_target_against": "shots_on_target_faced",
                "saves": "saves",
                "penalties_faced": "penalties_faced",
            },
        ),
    ]
    result: pd.DataFrame | None = None
    for filename, mapping in tables:
        path = root / filename
        if not path.is_file():
            continue
        frame = pd.read_csv(path, low_memory=False)
        available = {source: target for source, target in mapping.items() if source in frame}
        selected = frame[["match_id", "player_id", *available]].rename(columns=available)
        selected["match_id"] = selected["match_id"].astype("string")
        selected["player_id"] = selected["player_id"].astype("string")
        selected = selected.drop_duplicates(["match_id", "player_id"])
        result = selected if result is None else result.merge(
            selected, on=["match_id", "player_id"], how="outer", validate="one_to_one"
        )
    return result if result is not None else pd.DataFrame(columns=["match_id", "player_id"])


def _merge_metrics(
    base: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    provider: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    value_columns = [column for column in metrics if column not in {"match_id", "player_id"}]
    merged = base.merge(
        metrics,
        on=["match_id", "player_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", f"__{provider}"),
    )
    provenance: list[dict[str, object]] = []
    for column in value_columns:
        provider_column = f"{column}__{provider}" if column in base else column
        if provider_column != column:
            merged[column] = merged[provider_column]
            merged = merged.drop(columns=[provider_column])
        provenance.append(
            {
                "provider": provider,
                "canonical_field": column,
                "source_field": column,
                "non_null_rows": int(merged[column].notna().sum()),
                "total_rows": int(len(merged)),
            }
        )
    return merged, provenance


def _team_matches(
    seasons: Iterable[str],
    *,
    league: str,
    processed_root: Path,
    matchup_maps: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in seasons:
        path = processed_root / "understat" / league / season / "team_match.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path, low_memory=False)
        for game_id, group in frame.groupby("game_id", sort=False):
            venue = group["venue"].astype("string").str.upper()
            home = group.loc[venue.eq("H")]
            away = group.loc[venue.eq("A")]
            if len(home) != 1 or len(away) != 1:
                continue
            home_row, away_row = home.iloc[0], away.iloc[0]
            match_id = matchup_maps.get(season, {}).get(
                _matchup_key(home_row["team"], away_row["team"])
            )
            if match_id is None and "match_id" in group:
                match_id = str(home_row["match_id"])
            if match_id is None or match_id == "<NA>":
                continue
            rows.append(
                {
                    "match_id": match_id,
                    "kickoff_utc": pd.to_datetime(
                        f"{home_row['game_date']} {home_row.get('game_time', '00:00:00')}",
                        utc=True,
                        errors="coerce",
                    ),
                    "season": season,
                    "gameweek": home_row.get("round", pd.NA),
                    "home_team_id": str(home_row["team_id"]),
                    "away_team_id": str(away_row["team_id"]),
                    "home_goals": float(home_row["goals"]),
                    "away_goals": float(away_row["goals"]),
                    "home_xg": float(home_row["xg"]),
                    "away_xg": float(away_row["xg"]),
                    "provider": "understat",
                }
            )
    return pd.DataFrame(rows).sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True) if rows else pd.DataFrame()


def _current_roster(
    league: str,
    current_season: str,
    *,
    processed_root: Path,
) -> pd.DataFrame:
    path = processed_root / "fpl" / league / current_season / "season" / "cleaned_players.csv"
    if not path.is_file():
        return pd.DataFrame()
    roster = pd.read_csv(path, low_memory=False)
    roster["player_id"] = roster["player_id"].astype("string")
    return roster.drop_duplicates("player_id", keep="last")


def _apply_current_roster_context(
    matches: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    if roster.empty:
        return matches
    work = matches[matches["player_id"].isin(roster["player_id"])].copy()
    roster_index = roster.set_index("player_id")
    current_position = roster_index.get("fpl_pos", roster_index.get("position"))
    current_position = current_position.astype("string").str.upper().replace({"GK": "GKP"})
    current_team = roster_index["team_id"].astype("string")
    latest = (
        work.sort_values("kickoff_utc", kind="stable")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")
    )
    work["fpl_position"] = work["player_id"].map(current_position).fillna(work["fpl_position"])
    work["current_team_id"] = work["player_id"].map(current_team)
    latest_team = latest["team_id"].astype("string")
    latest_position = latest["historical_fpl_position"].astype("string")
    transferred = current_team.reindex(latest.index).ne(latest_team)
    position_changed = current_position.reindex(latest.index).ne(latest_position)
    work["transferred"] = work["player_id"].map(transferred).fillna(False)
    work["position_changed"] = work["player_id"].map(position_changed).fillna(False)
    work["appearances_since_transfer"] = 0
    work["appearances_since_position_change"] = 0
    work["minutes_since_position_change"] = 0.0
    status_map = {
        "a": "available", "d": "doubtful", "i": "injured",
        "s": "suspended", "u": "unavailable",
    }
    if "status" in roster_index:
        current_status = roster_index["status"].astype("string").str.lower().map(status_map)
        work["current_availability_status"] = work["player_id"].map(current_status)
    if "news" in roster_index:
        work["current_availability_reason"] = work["player_id"].map(roster_index["news"])
    eligible = work.loc[work["minutes"].ge(30)].groupby("player_id")["kickoff_utc"].max()
    work["days_since_eligible_appearance"] = work["player_id"].map(
        ((as_of - eligible).dt.total_seconds() / 86400.0).clip(lower=0)
    )
    return work


def _player_values(matches: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if roster.empty:
        return pd.DataFrame()
    values = roster[[column for column in ("player_id", "fpl_pos", "position", "now_cost") if column in roster]].copy()
    values["fpl_position"] = values.get("fpl_pos", values.get("position")).astype("string").str.upper().replace({"GK": "GKP"})
    values["price"] = _numeric(values, "now_cost")
    history = (
        matches.groupby("player_id", as_index=False)
        .agg(
            historical_points=("fpl_points", "sum"),
            historical_minutes=("minutes", "sum"),
            historical_appearances=("minutes", lambda value: int(pd.to_numeric(value, errors="coerce").ge(30).sum())),
        )
    )
    values = values.merge(history, on="player_id", how="left", validate="one_to_one")
    values[["historical_points", "historical_minutes", "historical_appearances"]] = values[
        ["historical_points", "historical_minutes", "historical_appearances"]
    ].fillna(0)
    price_millions = values["price"] / 10.0
    values["points_per_million"] = values["historical_points"] / price_millions.replace(0, np.nan)
    regular = values["historical_minutes"].ge(450)
    replacement = values.loc[regular].groupby("fpl_position")["points_per_million"].quantile(0.25)
    values["replacement_points_per_million"] = values["fpl_position"].map(replacement)
    values["historical_value_over_replacement"] = (
        values["points_per_million"] - values["replacement_points_per_million"]
    )
    return values[
        [
            "player_id", "fpl_position", "price", "historical_value_over_replacement",
            "historical_minutes", "historical_appearances", "historical_points",
            "points_per_million", "replacement_points_per_million",
        ]
    ]


def build_canonical_archetype_inputs(
    *,
    league: str,
    seasons: Iterable[str],
    current_season: str,
    as_of: str | pd.Timestamp,
    processed_root: Path = PROCESSED_ROOT,
    raw_root: Path = RAW_ROOT,
) -> CanonicalArchetypeInputs:
    """Join processed provider data into leakage-safe archetype inputs."""
    snapshot = pd.Timestamp(as_of)
    snapshot = snapshot.tz_localize("UTC") if snapshot.tzinfo is None else snapshot.tz_convert("UTC")
    player_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    matchup_maps: dict[str, dict[str, str]] = {}

    for season in sorted(set(str(value) for value in seasons)):
        fpl_path = processed_root / "fpl" / league / season / "gws" / "merged_gws.csv"
        understat_path = processed_root / "understat" / league / season / "player_match.csv"
        whoscored_root = processed_root / "whoscored" / league / season / "player_match"
        if not fpl_path.is_file():
            audit_rows.append({"season": season, "audit_type": "missing_provider", "details": "fpl"})
            continue
        fpl = pd.read_csv(fpl_path, low_memory=False)
        fpl, duplicates = _deduplicate_fpl(fpl, season)
        audit_rows.extend(duplicates)
        matchup_map = _fpl_matchup_map(fpl)
        matchup_maps[season] = matchup_map
        base = _canonical_fpl_rows(fpl, league=league, season=season, raw_root=raw_root)
        base = base.loc[base["kickoff_utc"].lt(snapshot)].copy()

        if understat_path.is_file():
            understat = pd.read_csv(understat_path, low_memory=False)
            metrics = _understat_player_metrics(
                understat,
                season=season,
                matchup_map=matchup_map,
                fpl_match_ids=set(base["match_id"].dropna().astype(str)),
                raw_root=raw_root,
            )
            base, provenance = _merge_metrics(base, metrics, provider="understat")
            provenance_rows.extend({"season": season, **row} for row in provenance)
        else:
            audit_rows.append({"season": season, "audit_type": "missing_provider", "details": "understat"})

        if (whoscored_root / "summary.csv").is_file():
            metrics = _who_scored_metrics(whoscored_root)
            base, provenance = _merge_metrics(base, metrics, provider="whoscored")
            provenance_rows.extend({"season": season, **row} for row in provenance)
        else:
            audit_rows.append({"season": season, "audit_type": "missing_provider", "details": "whoscored"})

        base["post_shot_xg"] = np.nan
        exposure = base["minutes"].replace(0, np.nan)
        attacking_response = (base.get("npxg") + base.get("xa")) * 90.0 / exposure
        defensive_response = base["clean_sheet"].astype(float)
        goalkeeper_response = defensive_response + base["fpl_saves"].fillna(0) / 3.0
        base["production_response"] = attacking_response
        base.loc[base["fpl_position"].eq("DEF"), "production_response"] = defensive_response
        base.loc[base["fpl_position"].eq("GKP"), "production_response"] = goalkeeper_response
        player_parts.append(base)

    matches = pd.concat(player_parts, ignore_index=True) if player_parts else pd.DataFrame()
    if not matches.empty:
        roster = _current_roster(league, current_season, processed_root=processed_root)
        matches = _apply_current_roster_context(matches, roster, as_of=snapshot)
        matches = matches.sort_values(["kickoff_utc", "match_id", "player_id"], kind="stable").reset_index(drop=True)
    else:
        roster = _current_roster(league, current_season, processed_root=processed_root)
    teams = _team_matches(
        seasons,
        league=league,
        processed_root=processed_root,
        matchup_maps=matchup_maps,
    )
    if not teams.empty:
        teams = teams.loc[teams["kickoff_utc"].lt(snapshot)].reset_index(drop=True)
    values = _player_values(matches, roster) if not matches.empty else pd.DataFrame()
    audit_rows.append(
        {
            "season": current_season,
            "audit_type": "build_summary",
            "details": (
                f"player_match_rows={len(matches)}; players={matches['player_id'].nunique() if not matches.empty else 0}; "
                f"team_matches={len(teams)}; value_rows={len(values)}"
            ),
        }
    )
    return CanonicalArchetypeInputs(
        player_matches=matches,
        team_matches=teams,
        player_values=values,
        field_provenance=pd.DataFrame(provenance_rows),
        build_audit=pd.DataFrame(audit_rows),
    )


__all__ = [
    "CanonicalArchetypeInputs",
    "build_canonical_archetype_inputs",
    "discover_joinable_seasons",
]
