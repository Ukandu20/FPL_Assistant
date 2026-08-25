#!/usr/bin/env python3
r"""fixtures_meta_builder.py – Batch-capable builder for **fixture_calendar.csv**
───────────────────────────────────────────────────────────────────────────────
Adds **home** and **away** (three-letter short codes) to the final CSV,
derives hex IDs (`home_id`, `away_id`) **from team_id/opponent_id + is_home/is_away**.

This script keeps fixtures **pure** (no FDR columns). If you want a
denormalized calendar that includes Fixture Difficulty Ratings (FDR),
use the `--attach-fdr <version|latest>` flag to write a view:
features/<views-subdir>/<SEASON>/fixture_calendar_with_fdr__<version>.csv

Batch rules
• `--season` → single season; omit → loop over every folder in `--fpl-root`.
• `--force`  → overwrite existing outputs.

Output columns (order)
----------------------
fpl_id, fbref_id, gw_orig, gw_played,
date_sched, date_played,
days_since_last_game,
team, team_id, opponent_id,
home, away, home_id, away_id,
status, sched_missing,
venue, gf, ga, xga, xg, result, is_promoted, is_relegated
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import numpy as np

from fpl_assistant.canonical.identity import stable_canonical_id


# ───────────────────── helpers ──────────────────────────────────────────────

def load_json(p: Path) -> dict:
    return json.loads(p.read_text("utf-8"))

def canon(s: str) -> str:
    return " ".join(str(s).lower().split())

def build_maps(long2hex: Dict[str, str], long2code: Dict[str, str]):
    name2hex = {canon(k): str(v).lower() for k, v in long2hex.items()}
    name2code = {canon(k): str(v).upper() for k, v in long2code.items()}
    code2hex = {name2code[k]: v for k, v in name2hex.items() if k in name2code}
    # Alias keys can legitimately point at an obsolete duplicate registry ID.
    # When the lookup contains the short code itself (e.g. ``nfo``), that is
    # the canonical identity and must win over aliases such as ``nottingham``.
    for code in set(name2code.values()):
        direct = name2hex.get(canon(code))
        if direct:
            code2hex[code] = direct
    return name2hex, name2code, code2hex

def normalise_date(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series, errors="coerce")
    if hasattr(series.dt, "tz") and series.dt.tz is not None:
        series = series.dt.tz_convert(None)
    return series.dt.floor("D")

def _to_bool_mask(s: pd.Series) -> pd.Series:
    """Convert various truthy encodings to boolean mask (handles numeric floats like 1.0/0.0)."""
    if s is None:
        return pd.Series(False, index=pd.RangeIndex(0))
    if is_bool_dtype(s):
        return s.fillna(False)
    if is_numeric_dtype(s):
        # 0, 0.0 -> False; any non-zero -> True
        return s.fillna(0).astype(float).ne(0.0)
    # string-like: accept common truthy tokens, including "1.0"
    tokens_true = {"1","1.0","true","t","yes","y"}
    return s.astype(str).str.strip().str.lower().isin(tokens_true)


def _fixture_finished_mask(fixtures: pd.DataFrame) -> pd.Series:
    """Recognize finalized and provisionally completed FPL fixtures."""
    finished = pd.Series(False, index=fixtures.index)
    for column in ("finished", "finished_provisional"):
        if column in fixtures:
            finished |= _to_bool_mask(fixtures[column])
    if {"started", "minutes"}.issubset(fixtures.columns):
        finished |= _to_bool_mask(fixtures["started"]) & pd.to_numeric(
            fixtures["minutes"], errors="coerce"
        ).ge(90)
    return finished

def _venue_to_is_home_int8(venue: pd.Series) -> pd.Series:
    """Map FBref venue → is_home:Int8 (1=home, 0=away, <NA>=neutral/unknown)."""
    v = venue.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=venue.index, dtype="Int8")
    out[v.isin({"home", "h"})] = 1
    out[v.isin({"away", "a"})] = 0
    return out

def _naeq(a: pd.Series, b: pd.Series) -> pd.Series:
    """NA-safe equality: True if equal OR both NA."""
    return (a == b) | (a.isna() & b.isna())


def _observed_match_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify rows backed by a played result, not a provisional schedule."""
    observed = pd.Series(False, index=frame.index)
    for column in ("is_result", "has_data"):
        if column in frame.columns:
            observed |= _to_bool_mask(frame[column])
    result_columns = [
        column
        for column in ("team_goals", "opp_goals", "team_xg", "opp_xg")
        if column in frame.columns
    ]
    if result_columns:
        observed |= frame[result_columns].notna().any(axis=1)
    return observed

def read_fixture_calendar(out_dir: Path, season: str) -> pd.DataFrame:
    fp = out_dir / season / "fixture_calendar.csv"
    return pd.read_csv(fp, parse_dates=["date_sched", "date_played"])


def build_bootstrap_fixture_calendar(
    *,
    season: str,
    league: str,
    fpl_csv: Path,
    teams_csv: Path,
    team_map_fp: Path,
    short_map_fp: Path,
    out_dir: Path,
    force: bool = False,
) -> bool:
    """Publish a schedule-only canonical calendar from official FPL fixtures.

    This is deliberately independent of WhoScored, Understat, and FBref so
    provider cleaners can resolve match identities before any match is played.
    Domestic league home/away pairings are unique within a season, therefore
    the stable ID intentionally excludes the mutable kickoff date.
    """
    dst_dir = out_dir / season
    out_csv = dst_dir / "fixture_calendar.csv"
    if out_csv.exists() and not force:
        logging.info("%s • bootstrap calendar already exists – skip (use --force)", season)
        return False
    if not fpl_csv.is_file():
        raise FileNotFoundError(fpl_csv)
    if not teams_csv.is_file():
        raise FileNotFoundError(teams_csv)

    _, _, code2hex = build_maps(load_json(team_map_fp), load_json(short_map_fp))
    fixtures = pd.read_csv(fpl_csv, parse_dates=["kickoff_time"])
    fixtures["is_finished_app"] = _fixture_finished_mask(fixtures)
    teams = pd.read_csv(teams_csv)
    required_fixtures = {"id", "event", "kickoff_time", "team_h", "team_a"}
    required_teams = {"id", "name", "short_name"}
    if missing := sorted(required_fixtures - set(fixtures.columns)):
        raise ValueError(f"{fpl_csv} lacks required columns: {missing}")
    if missing := sorted(required_teams - set(teams.columns)):
        raise ValueError(f"{teams_csv} lacks required columns: {missing}")

    team_rows = teams[["id", "name", "short_name"]].copy()
    team_rows["short_name"] = team_rows["short_name"].astype("string").str.upper()
    numeric_to_code = dict(zip(team_rows["id"], team_rows["short_name"]))
    numeric_to_name = dict(zip(team_rows["id"], team_rows["name"]))
    generated_codes: list[str] = []
    for code in sorted(set(numeric_to_code.values())):
        if code and code not in code2hex:
            code2hex[code] = stable_canonical_id(
                "team", "fpl", code, length=12
            )
            generated_codes.append(code)
    if generated_codes:
        logging.warning(
            "%s • generated canonical team IDs from FPL codes absent from the "
            "registry: %s",
            season,
            ", ".join(generated_codes),
        )

    records: list[dict] = []
    for fixture in fixtures.itertuples(index=False):
        home = str(numeric_to_code.get(fixture.team_h, "")).upper()
        away = str(numeric_to_code.get(fixture.team_a, "")).upper()
        home_id = code2hex.get(home)
        away_id = code2hex.get(away)
        if not home or not away or not home_id or not away_id:
            raise ValueError(
                f"Unable to resolve canonical teams for FPL fixture {fixture.id}: "
                f"home={home!r}/{home_id!r}, away={away!r}/{away_id!r}"
            )
        match_id = stable_canonical_id(
            "match", league, season, home_id, away_id, length=16
        )
        kickoff = pd.to_datetime(fixture.kickoff_time, utc=True, errors="coerce")
        date_sched = kickoff.tz_convert(None).floor("D") if pd.notna(kickoff) else pd.NaT
        status = "finished" if bool(fixture.is_finished_app) else "scheduled"
        base = {
            "fpl_id": fixture.id,
            "match_id": match_id,
            # Compatibility alias for downstream code that predates canonical
            # match IDs. It is not evidence that FBref supplied this identity.
            "fbref_id": match_id,
            "gw_orig": fixture.event,
            "gw_played": fixture.event if status == "finished" else pd.NA,
            "date_sched": date_sched,
            "date_played": date_sched if status == "finished" else pd.NaT,
            "days_since_last_game": pd.NA,
            "home": home,
            "away": away,
            "home_id": home_id,
            "away_id": away_id,
            "status": status,
            "sched_missing": 0,
            "venue": pd.NA,
            "gf": getattr(fixture, "team_h_score", pd.NA),
            "ga": getattr(fixture, "team_a_score", pd.NA),
            "xga": pd.NA,
            "xg": pd.NA,
            "poss": pd.NA,
            "result": pd.NA,
            "is_promoted": pd.NA,
            "is_relegated": pd.NA,
            "home_name": numeric_to_name.get(fixture.team_h, home),
            "away_name": numeric_to_name.get(fixture.team_a, away),
        }
        for is_home in (True, False):
            row = dict(base)
            row.update(
                {
                    "team": home if is_home else away,
                    "team_id": home_id if is_home else away_id,
                    "opponent_id": away_id if is_home else home_id,
                    "is_home": int(is_home),
                }
            )
            if not is_home:
                row["gf"], row["ga"] = base["ga"], base["gf"]
            records.append(row)

    out = pd.DataFrame(records)
    key = ["match_id", "team_id"]
    if out.duplicated(key).any():
        raise ValueError("Bootstrap calendar contains duplicate (match_id, team_id) rows")
    expected_rows = 2 * len(fixtures)
    if len(out) != expected_rows:
        raise ValueError(f"Bootstrap calendar has {len(out)} rows; expected {expected_rows}")

    ordered = [
        "fpl_id", "match_id", "fbref_id", "gw_orig", "gw_played",
        "date_sched", "date_played", "days_since_last_game", "team",
        "team_id", "opponent_id", "is_home", "home", "away", "home_id",
        "away_id", "status", "sched_missing", "venue", "gf", "ga", "xga",
        "xg", "poss", "result", "is_promoted", "is_relegated",
    ]
    out = out[ordered].sort_values(["date_sched", "fpl_id", "is_home"], ascending=[True, True, False])
    dst_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out.copy()
    for column in ("date_sched", "date_played"):
        csv_out[column] = pd.to_datetime(csv_out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    csv_out.to_csv(out_csv, index=False)
    logging.info("%s • bootstrap fixture_calendar.csv (%d rows)", season, len(out))
    return True

# ──────────────── NEW: sticky locking of schedule fields ─────────────────────

def _lock_sched_fields(out_new: pd.DataFrame, out_dir: Path, season: str) -> pd.DataFrame:
    """
    If a prior fixture_calendar.csv exists, keep previously-written
    date_sched and gw_orig for matching rows (keyed by fpl_id+team).
    This makes date_sched/gw_orig **sticky** so reschedules are measurable.
    """
    prev_path = out_dir / season / "fixture_calendar.csv"
    if not prev_path.exists():
        return out_new

    try:
        prev = pd.read_csv(prev_path, parse_dates=["date_sched", "date_played"])
    except Exception:
        logging.warning("%s • failed to read prior fixture_calendar.csv; skipping sticky restore.", season)
        return out_new

    keep = prev[["fpl_id", "team", "date_sched", "gw_orig"]].copy()
    keep["date_sched"] = normalise_date(keep["date_sched"])
    # prefer Int8 for gw_orig if possible
    try:
        keep["gw_orig"] = pd.to_numeric(keep["gw_orig"], errors="coerce").astype("Int8")
    except Exception:
        pass

    merged = out_new.merge(
        keep.rename(columns={"date_sched": "_date_sched_prev", "gw_orig": "_gw_orig_prev"}),
        on=["fpl_id", "team"],
        how="left",
        validate="many_to_one"
    )

    # restore when a previous value exists
    have_prev_date = merged["_date_sched_prev"].notna()
    have_prev_gw   = merged["_gw_orig_prev"].notna()

    merged.loc[have_prev_date, "date_sched"] = merged.loc[have_prev_date, "_date_sched_prev"]
    merged.loc[have_prev_gw,   "gw_orig"]    = merged.loc[have_prev_gw,   "_gw_orig_prev"]

    merged.drop(columns=["_date_sched_prev","_gw_orig_prev"], inplace=True)

    # normalize & dtype after restore
    merged["date_sched"] = normalise_date(merged["date_sched"])
    try:
        merged["gw_orig"] = pd.to_numeric(merged["gw_orig"], errors="coerce").astype("Int8")
    except Exception:
        pass

    n_locked = int(have_prev_date.sum())
    if n_locked:
        logging.info("%s • sticky lock applied: restored %d date_sched/gw_orig rows from previous calendar.", season, n_locked)
    return merged

# ─────────────── FDR view materializer (optional, behind a flag) ────────────

def maybe_write_fdr_view(
    calendar_df: pd.DataFrame,
    season: str,
    features_root: Path,
    team_version: str,
    views_subdir: str = "views"
) -> None:
    calendar_df = calendar_df.copy()
    tf_dir = Path(features_root) / team_version
    tf_path = tf_dir / season / "team_form.csv"
    if not tf_path.exists():
        logging.warning("%s • team_form.csv not found at %s; skip FDR view",
                        season, tf_path)
        return

    tf = pd.read_csv(tf_path, low_memory=False)

    if "game_date" in tf.columns and "date_played" not in tf.columns:
        tf = tf.rename(columns={"game_date": "date_played"})

    # Both sides must use the same date dtype.  CSV-loaded team_form dates are
    # strings, while the in-memory fixture calendar carries datetime64.
    if "date_played" in tf.columns:
        tf["date_played"] = normalise_date(tf["date_played"])
    if "date_played" in calendar_df.columns:
        calendar_df["date_played"] = normalise_date(calendar_df["date_played"])

    for c in ("home_id", "away_id", "team_id"):
        if c in tf.columns:
            tf[c] = tf[c].astype("string").str.lower()
    for c in ("home_id", "away_id"):
        if c in calendar_df.columns:
            calendar_df[c] = calendar_df[c].astype("string").str.lower()

    need_A = {"date_played", "home_id", "away_id", "fdr_home", "fdr_away"}
    need_B = {"fpl_id", "team_id", "fdr_home", "fdr_away"}

    merged = None
    if need_A.issubset(set(tf.columns)):
        key = ["date_played", "home_id", "away_id"]
        subset = tf[key + ["fdr_home", "fdr_away"]].drop_duplicates(key)
        try:
            merged = calendar_df.merge(subset, on=key, how="left", validate="many_to_one")
            logging.info("%s • FDR view join (A: date+ids) OK; null FDR rows=%d",
                         season, int(merged["fdr_home"].isna().sum()))
        except Exception:
            logging.exception("%s • join (A) failed; falling back to (B) if possible", season)

    if merged is None and need_B.issubset(set(tf.columns)) and "fpl_id" in calendar_df.columns:
        tf_b = tf[["fpl_id", "team_id", "fdr_home", "fdr_away"]].copy()
        tf_b["team_id"] = tf_b["team_id"].astype("string").str.lower()
        if tf_b.duplicated(["fpl_id", "team_id"]).any():
            raise ValueError(
                f"{tf_path} has duplicate (fpl_id, team_id) rows; refusing an "
                "FDR merge that could multiply fixture rows"
            )
        left = calendar_df.merge(
            tf_b[["fpl_id", "team_id", "fdr_home"]].rename(columns={"team_id": "home_id"}),
            on=["fpl_id", "home_id"], how="left", validate="many_to_one"
        )
        merged = left.merge(
            tf_b[["fpl_id", "team_id", "fdr_away"]].rename(columns={"team_id": "away_id"}),
            on=["fpl_id", "away_id"], how="left", validate="many_to_one"
        )
        logging.info("%s • FDR view join (B: fpl_id+team_id) OK; null FDR rows=%d",
                     season, int(merged["fdr_home"].isna().sum()))
    if merged is None:
        logging.error(
            "%s • team_form.csv lacks required columns for join. "
            "Expected either %s or %s.",
            season, sorted(need_A), sorted(need_B)
        )
        return

    if len(merged) != len(calendar_df):
        raise ValueError(
            f"FDR merge changed fixture-calendar row count from "
            f"{len(calendar_df)} to {len(merged)}"
        )

    out_dir = Path(features_root) / views_subdir / season
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fixture_calendar_with_fdr__{team_version}.csv"
    merged.to_csv(out_path, index=False)
    logging.info("%s • wrote FDR view → %s", season, out_path)


# ─────────────────── single-season builder ─────────────────────────────────

def build_fixture_calendar(
    season: str,
    fpl_csv: Path,
    fb_csv: Path,
    ws_csv: Path,
    und_csv: Path,
    team_map_fp: Path,
    short_map_fp: Path,
    out_dir: Path,
    *,
    attach_fdr: str | None,
    features_root: Path,
    views_subdir: str,
    teams_csv: Path | None = None,
    match_tolerance_days: int = 100,
    force: bool = False,
) -> bool:
    dst_dir = out_dir / season
    out_csv = dst_dir / "fixture_calendar.csv"
    if out_csv.exists() and not force:
        logging.info("%s • already done – skip (use --force)", season)
        if attach_fdr:
            cal = read_fixture_calendar(out_dir, season)
            maybe_write_fdr_view(cal, season, features_root, attach_fdr, views_subdir)
        return False
    required_inputs = [fpl_csv, ws_csv, und_csv]
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            f"{season} • missing fixture/provider inputs: {missing_inputs}"
        )

    # -- lookups
    name2hex, name2code, code2hex = build_maps(
        load_json(team_map_fp), load_json(short_map_fp)
    )

    # -- FPL data (scheduled)
    fpl = pd.read_csv(fpl_csv, parse_dates=["kickoff_time"])
    fpl = fpl.rename(
        columns={"id": "fpl_id", "event": "gw_orig", "team_h": "home_id_fpl", "team_a": "away_id_fpl"}
    )
    fpl["_fpl_row"]   = np.arange(len(fpl), dtype=np.int64)
    fpl["status"]     = np.where(
        _fixture_finished_mask(fpl), "finished", "scheduled"
    )
    fpl["date_sched"] = normalise_date(fpl["kickoff_time"])
    fpl["sched_missing"] = 1  # will be corrected after joining

    if teams_csv is None:
        teams_csv = fpl_csv.with_name("teams.csv")
        if not teams_csv.exists():
            logging.warning("%s • teams.csv missing – skipped", season)
            return False
    teams_df = pd.read_csv(teams_csv, usecols=["id", "name"])
    id2name = dict(zip(teams_df.id, teams_df.name.map(canon)))

    # map FPL numeric ids -> normalized long names -> short codes for merge keys
    for side in ("home", "away"):
        fpl[f"{side}_long"] = fpl[f"{side}_id_fpl"].map(id2name)
        fpl[side] = fpl[f"{side}_long"].map(name2code)

    # WhoScored supplies fixture identity, team perspectives, home/away and
    # stadium. Understat supplies the observed goals and xG metrics. Both are
    # team-perspective datasets, so each match correctly appears twice.
    ws = pd.read_csv(ws_csv, low_memory=False)
    und = pd.read_csv(und_csv, low_memory=False)

    required_ws = {
        "match_id", "team_id", "opponent_id", "team", "home", "away",
        "game_date", "venue", "is_home",
    }
    required_und = {
        "match_id", "team_id", "opp_id", "team_goals", "opp_goals",
        "team_xg", "opp_xg", "result", "game_date",
    }
    missing_ws = sorted(required_ws - set(ws.columns))
    missing_und = sorted(required_und - set(und.columns))
    if missing_ws:
        raise ValueError(f"{ws_csv} lacks required columns: {missing_ws}")
    if missing_und:
        raise ValueError(f"{und_csv} lacks required columns: {missing_und}")

    for frame in (ws, und):
        for column in ("match_id", "team_id"):
            frame[column] = frame[column].astype("string").str.strip().str.lower()
    ws["opponent_id"] = ws["opponent_id"].astype("string").str.strip().str.lower()
    und["opp_id"] = und["opp_id"].astype("string").str.strip().str.lower()

    join_key = ["match_id", "team_id"]
    if ws.duplicated(join_key).any():
        raise ValueError(f"{ws_csv} has duplicate (match_id, team_id) rows")
    if und.duplicated(join_key).any():
        raise ValueError(f"{und_csv} has duplicate (match_id, team_id) rows")

    # FBref's surviving player-match schedule remains the canonical match-ID
    # registry even though its team-match schedule was deprecated. Validate
    # provider IDs against it when that artifact is available.
    fbref_schedule_path = fb_csv.parent.parent / "player_match" / "schedule.csv"
    if fbref_schedule_path.is_file():
        fbref_schedule = pd.read_csv(fbref_schedule_path, low_memory=False)
        id_column = "match_id" if "match_id" in fbref_schedule.columns else "game_id"
        if id_column in fbref_schedule.columns:
            canonical_ids = set(
                fbref_schedule[id_column].dropna().astype("string").str.lower()
            )
            provider_ids = set(ws["match_id"].dropna())
            unknown_ids = sorted(provider_ids - canonical_ids)
            if unknown_ids:
                raise ValueError(
                    f"{ws_csv} contains {len(unknown_ids)} match IDs absent from "
                    f"FBref canonical schedule {fbref_schedule_path}"
                )
            logging.info(
                "%s • FBref canonical match-ID validation: %d matches",
                season,
                len(provider_ids),
            )

    metric_columns = [
        "opp_id", "team_goals", "opp_goals", "team_xg", "opp_xg",
        "result", "game_date",
    ]
    metric_columns.extend(
        column for column in ("is_result", "has_data") if column in und.columns
    )
    metrics = und[join_key + metric_columns].rename(
        columns={"game_date": "_understat_game_date"}
    )
    fb = ws.merge(metrics, on=join_key, how="left", validate="one_to_one")

    missing_metrics = fb["opp_id"].isna()
    if missing_metrics.any():
        raise ValueError(
            f"{und_csv} is missing {int(missing_metrics.sum())} WhoScored "
            "(match_id, team_id) rows"
        )
    opponent_conflicts = fb["opponent_id"].ne(fb["opp_id"])
    if opponent_conflicts.any():
        raise ValueError(
            "WhoScored/Understat opponent IDs conflict on "
            f"{int(opponent_conflicts.sum())} rows"
        )

    ws_dates = normalise_date(fb["game_date"])
    und_dates = normalise_date(fb["_understat_game_date"])
    all_date_conflicts = ws_dates.notna() & und_dates.notna() & ws_dates.ne(und_dates)
    observed_rows = _observed_match_mask(fb)
    date_conflicts = all_date_conflicts & observed_rows
    if date_conflicts.any():
        raise ValueError(
            "WhoScored/Understat game dates conflict on "
            f"{int(date_conflicts.sum())} rows"
        )
    provisional_conflicts = all_date_conflicts & ~observed_rows
    if provisional_conflicts.any():
        logging.info(
            "%s • allowing %d provisional WhoScored/Understat date conflicts",
            season,
            int(provisional_conflicts.sum()),
        )
    fb["date_played"] = ws_dates.fillna(und_dates)
    fb["is_away"] = (~_to_bool_mask(fb["is_home"])).astype("Int8")

    # Publish the schema consumed by team_form_builder. The current WhoScored
    # schedule does not expose possession, so preserve it as unknown; the team
    # form builder applies its documented neutral fallback.
    fb["gf"] = pd.to_numeric(fb["team_goals"], errors="coerce")
    fb["ga"] = pd.to_numeric(fb["opp_goals"], errors="coerce")
    fb["xg"] = pd.to_numeric(fb["team_xg"], errors="coerce")
    fb["xga"] = pd.to_numeric(fb["opp_xg"], errors="coerce")
    fb["poss"] = pd.NA
    fb["is_promoted"] = pd.NA
    fb["is_relegated"] = pd.NA

    fb_match = fb[[
        "match_id", "team", "team_id", "opponent_id",
        "home", "away", "date_played", "venue", "gf", "ga", "xg", "xga",
        "poss", "result", "is_promoted", "is_relegated", "is_home", "is_away",
    ]].copy()
    logging.info(
        "%s • provider reconciliation: %d WhoScored rows joined one-to-one "
        "to Understat",
        season,
        len(fb_match),
    )

    # -- First pass: strict date match (home, away, date_sched == date_played)
    cal = fpl.merge(
        fb_match,
        left_on=["home", "away", "date_sched"],
        right_on=["home", "away", "date_played"],
        how="left",
        validate="one_to_many",  # one FPL row -> two FB rows (one per team)
    )
    strict_matched = cal["match_id"].notna().sum()
    logging.info("%s • strict matches: %d", season, strict_matched)

    # -- Second pass: nearest-date recovery for reschedules within tolerance
    missing_mask = cal["match_id"].isna()
    if missing_mask.any():
        miss = cal.loc[missing_mask, ["_fpl_row", "home", "away", "date_sched"]].drop_duplicates("_fpl_row")
        fb_dates = fb_match[["home", "away", "date_played"]].drop_duplicates()

        cand = miss.merge(fb_dates, on=["home", "away"], how="left")
        cand["absdiff"] = (cand["date_played"] - cand["date_sched"]).abs().dt.days
        cand = cand[cand["absdiff"] <= int(match_tolerance_days)]
        # Prefer postponements (date_played >= date_sched) before bring-forwards,
        # then choose smallest absolute delta, then earliest actual date.
        nearest = (
            cand.sort_values(["_fpl_row", "absdiff", "date_played"])
                .drop_duplicates("_fpl_row", keep="first")
        )

        # Expand to both team rows for that nearest date
        fb_rows = nearest[["_fpl_row", "home", "away", "date_played"]].merge(
            fb_match, on=["home", "away", "date_played"], how="left", validate="one_to_many"
        )

        # Bring back FPL columns for these rows
        fpl_base = fpl[["_fpl_row", "fpl_id", "gw_orig", "date_sched", "home", "away", "status"]].copy()
        repair = fpl_base.merge(fb_rows, on=["_fpl_row", "home", "away"], how="right")

        # shape repair like 'cal'
        common_cols = list(set(cal.columns).intersection(set(repair.columns)))
        repair = repair[common_cols].copy()
        repair["gw_played"] = repair.get("gw_orig", np.nan)

        # combine: keep strict matches, drop the single unmatched shell rows, add full repaired rows
        still_unresolved = cal.loc[missing_mask & ~cal["_fpl_row"].isin(nearest["_fpl_row"])]
        cal = pd.concat([cal.loc[~missing_mask], repair, still_unresolved], ignore_index=True)

        logging.info("%s • recovered via nearest-date: %d; unresolved: %d",
                     season, int(repair["match_id"].notna().sum()),
                     int(still_unresolved.shape[0]))

    # Ensure team/opponent ids are lowercase strings
    for c in ("team_id", "opponent_id"):
        if c in cal.columns:
            cal[c] = cal[c].astype(str).str.lower()

    # If team_id missing (rare), attempt resolve from short code
    if "team_id" not in cal.columns:
        cal["team_id"] = cal["team"].map(code2hex)
    else:
        cal["team_id"] = cal["team_id"].fillna(cal["team"].map(code2hex))

    # ── Derive home/away using IDs first, then venue, then legacy flags ──
    cal["home_hex_expected"] = cal["home"].astype(str).map(code2hex).astype("string")
    cal["away_hex_expected"] = cal["away"].astype(str).map(code2hex).astype("string")
    id_cmp_valid = cal["team_id"].notna() & cal["home_hex_expected"].notna()
    is_home_by_ids = pd.Series(pd.NA, index=cal.index, dtype="Int8")
    is_home_by_ids[id_cmp_valid] = cal.loc[id_cmp_valid, "team_id"].eq(
        cal.loc[id_cmp_valid, "home_hex_expected"]
    ).astype("Int8")
    is_home_by_venue = _venue_to_is_home_int8(cal["venue"])
    if "is_home" in cal.columns:
        is_home_flag = _to_bool_mask(cal["is_home"]).astype("Int8")
    elif "is_away" in cal.columns:
        is_home_flag = (~_to_bool_mask(cal["is_away"])).astype("Int8")
    else:
        is_home_flag = pd.Series(pd.NA, index=cal.index, dtype="Int8")

    cal["is_home"] = is_home_by_ids
    need_fill = cal["is_home"].isna()
    if need_fill.any():
        cal.loc[need_fill, "is_home"] = is_home_by_venue.loc[need_fill]
    need_fill = cal["is_home"].isna()
    if need_fill.any():
        cal.loc[need_fill, "is_home"] = is_home_flag.loc[need_fill]
    cal["is_home"] = cal["is_home"].astype("Int8")
    cal["is_away"] = (1 - cal["is_home"]).astype("Int8")

    # Build mask and assign ids
    hmask = (cal["is_home"] == 1)
    cal["home_id"] = np.where(hmask, cal["team_id"], cal["opponent_id"])
    cal["away_id"] = np.where(hmask, cal["opponent_id"], cal["team_id"])

    # enforce lowercase hex strings (keeps <NA> if missing)
    for c in ("home_id", "away_id"):
        cal[c] = cal[c].astype("string").str.lower()

    # days since last match (by FBref 'team' rows; NaNs will be ignored)
    cal = cal.sort_values(["team", "date_played"]).copy()
    cal["days_since_last_game"] = (
        cal.groupby("team")["date_played"].diff().dt.days.fillna(0).astype(int)
    )

    # Default: gw_played := gw_orig if not set
    if "gw_played" not in cal.columns:
        cal["gw_played"] = np.nan
    cal["gw_played"] = cal["gw_played"].fillna(cal["gw_orig"])

    # final flags
    cal["gw_played"] = cal.get("gw_played", cal.get("gw_orig"))
    cal["sched_missing"] = cal["match_id"].isna().astype("Int8")

    # ── select & order for output (PURE schedule; no FDR here) ──
    out = cal[[
        "fpl_id", "match_id", "gw_orig", "gw_played",
        "date_sched", "date_played", "days_since_last_game",
        "team", "team_id", "opponent_id", "is_home",
        "home", "away", "home_id", "away_id",
        "status", "sched_missing", "venue",
        "gf", "ga", "xga", "xg", "poss", "result", "is_promoted", "is_relegated",
    ]].copy()
    # Compatibility alias for older consumers. ``match_id`` is canonical and
    # provider-neutral; this value is not necessarily an FBref-native ID.
    out.insert(2, "fbref_id", out["match_id"])

    # ── NEW: apply sticky schedule lock from previous calendar ──
    out = _lock_sched_fields(out, out_dir, season)

    # guarantee integer 0/1 in CSV
    out["is_home"] = out["is_home"].astype("Int8")
    out["gw_played"] = out["gw_played"].astype("Int8")

    # Normalize dates (date-only) just before write
    out["date_sched"]  = normalise_date(out["date_sched"])
    out["date_played"] = normalise_date(out["date_played"])

    # ── write base calendar ──
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_csv = dst_dir / "fixture_calendar.csv"

    csv_out = out.copy()
    csv_out["date_sched"]  = csv_out["date_sched"].dt.strftime("%Y-%m-%d")
    csv_out["date_played"] = csv_out["date_played"].dt.strftime("%Y-%m-%d")
    csv_out.to_csv(out_csv, index=False)
    logging.info("%s • fixture_calendar.csv (%d rows)", season, len(out))

    # Diagnostics
    missing = out[out["match_id"].isna()]
    missing_audit_path = dst_dir / "_manual_match_identity.csv"
    if not missing.empty:
        missing.to_csv(missing_audit_path, index=False)
        logging.warning("%s • %d rows lack match_id (see _manual_match_identity.csv)", season, len(missing))
    else:
        missing_audit_path.unlink(missing_ok=True)
    null_ids = out[out["home_id"].isna() | out["away_id"].isna()]
    null_ids_audit_path = dst_dir / "_missing_home_away_ids.csv"
    if not null_ids.empty:
        null_ids.to_csv(null_ids_audit_path, index=False)
        logging.warning("%s • %d rows lack home_id/away_id (see _missing_home_away_ids.csv)",
                        season, len(null_ids))
    else:
        null_ids_audit_path.unlink(missing_ok=True)
    # NA-safe alignment audit (only on rows with all IDs present)
    cal_ids_ok = cal[["team_id","opponent_id","home_id","away_id","is_home"]].dropna().copy()
    bad_align = cal_ids_ok[
        ((cal_ids_ok["is_home"]==True) & (~_naeq(cal_ids_ok["home_id"], cal_ids_ok["team_id"]) |
                                      ~_naeq(cal_ids_ok["away_id"], cal_ids_ok["opponent_id"]))) |
        ((cal_ids_ok["is_home"]==False) & (~_naeq(cal_ids_ok["home_id"], cal_ids_ok["opponent_id"]) |
                                      ~_naeq(cal_ids_ok["away_id"], cal_ids_ok["team_id"])))
    ]
    if not bad_align.empty:
        cols = ["home","away","team","venue","date_played","team_id","opponent_id","home_id","away_id","is_home"]
        (dst_dir / "_home_alignment_audit.csv").write_text("")
        cal.loc[bad_align.index, cols].to_csv(dst_dir / "_home_alignment_audit.csv", index=False)
        logging.error("%s • %d rows fail home/away ID alignment (see _home_alignment_audit.csv)", season, len(bad_align))

    # ── reschedule audit (only where we *did* match) ──
    if bad_align.empty:
        (dst_dir / "_home_alignment_audit.csv").unlink(missing_ok=True)

    audit = out.loc[out["match_id"].notna() & out["date_sched"].notna() & out["date_played"].notna()].copy()
    audit = audit[audit["date_sched"] != audit["date_played"]]
    if not audit.empty:
        audit["delta_days"] = (audit["date_played"] - audit["date_sched"]).dt.days.astype(int)
        audit["abs_delta_days"] = audit["delta_days"].abs()
        audit_cols = [
            "fpl_id", "match_id", "fbref_id", "gw_orig", "gw_played",
            "home", "away", "team", "team_id", "opponent_id",
            "date_sched", "date_played", "delta_days", "abs_delta_days", "status"
        ]
        (dst_dir / "_reschedule_audit.csv").write_text("")
        audit[audit_cols].to_csv(dst_dir / "_reschedule_audit.csv", index=False)
        logging.info("%s • reschedule audit: %d rows moved (see _reschedule_audit.csv)", season, len(audit))
    else:
        (dst_dir / "_reschedule_audit.csv").unlink(missing_ok=True)
        logging.info("%s • no reschedules detected", season)

    # ── optionally write an FDR-attached view ──
    if attach_fdr:
        maybe_write_fdr_view(out, season, features_root, attach_fdr, views_subdir)

    return True


# ───────────────────── batch driver ─────────────────────────────────────────

def run_batch(
    seasons: List[str],
    fpl_root: Path,
    fbref_league: Path,
    understat_league: Path,
    whoscored_league: Path,
    team_map: Path,
    short_map: Path,
    out_dir: Path,
    *,
    attach_fdr: str | None,
    features_root: Path,
    views_subdir: str,
    match_tolerance_days: int,
    force: bool,
    bootstrap: bool = False,
    league: str = "ENG-Premier League",
):
    failures: list[tuple[str, Exception]] = []
    for season in seasons:
        fpl_csv = fpl_root / season / "season" / "fixtures.csv"
        fb_csv = fbref_league / season / "team_match" / "schedule.csv"
        ws_csv = whoscored_league / season / "team_match" / "schedule.csv"
        und_csv = understat_league / season / "schedule.csv"
        teams_csv = fpl_root / season / "season" / "teams.csv"
        if not teams_csv.is_file():
            teams_csv = fpl_root / season / "teams.csv"
        try:
            if bootstrap:
                build_bootstrap_fixture_calendar(
                    season=season,
                    league=league,
                    fpl_csv=fpl_csv,
                    teams_csv=teams_csv,
                    team_map_fp=team_map,
                    short_map_fp=short_map,
                    out_dir=out_dir,
                    force=force,
                )
                continue
            build_fixture_calendar(
                season=season,
                fpl_csv=fpl_csv,
                fb_csv=fb_csv,
                ws_csv=ws_csv,
                und_csv=und_csv,
                team_map_fp=team_map,
                short_map_fp=short_map,
                out_dir=out_dir,
                attach_fdr=attach_fdr,
                features_root=features_root,
                views_subdir=views_subdir,
                teams_csv=teams_csv,
                match_tolerance_days=match_tolerance_days,
                force=force,
            )
        except Exception as exc:
            logging.exception("%s • unhandled error", season)
            failures.append((season, exc))
    if failures:
        failed_seasons = ", ".join(season for season, _ in failures)
        raise RuntimeError(f"Fixture calendar build failed for: {failed_seasons}")


# ───────────────────────────── CLI ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season")
    ap.add_argument("--league", default="ENG-Premier League")
    ap.add_argument(
        "--bootstrap",
        action="store_true",
        help=(
            "Build a schedule-only canonical fixture calendar from FPL fixtures. "
            "Does not require cleaned WhoScored, Understat, or FBref data."
        ),
    )
    ap.add_argument("--fpl-root", type=Path, default=Path("data/raw/fpl/ENG-Premier League"))
    ap.add_argument(
        "--fbref-league-dir",
        type=Path,
        default=Path("data/processed/fbref/ENG-Premier League"),
        help="Optional FBref root used only for schedule validation when present",
    )
    ap.add_argument(
        "--whoscored-league-dir",
        type=Path,
        default=Path("data/processed/whoscored/ENG-Premier League"),
        help="Required for the enriched (non-bootstrap) calendar",
    )
    ap.add_argument(
        "--understat-league-dir",
        type=Path,
        default=Path("data/processed/understat/ENG-Premier League"),
        help="Required for the enriched (non-bootstrap) calendar",
    )
    ap.add_argument("--team-map", type=Path, default=Path("data/processed/registry/_id_lookup_teams.json"))
    ap.add_argument("--short-map", type=Path, default=Path("data/config/teams.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/registry/fixtures"))
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"))
    ap.add_argument("--attach-fdr", default=None,
                    help="If set (e.g., 'latest' or 'v7'), also write "
                         "features/<views-subdir>/<SEASON>/fixture_calendar_with_fdr__<version>.csv")
    ap.add_argument("--views-subdir", default="views",
                    help="Subfolder under features/ to store materialized views (default: 'views').")
    ap.add_argument("--match-tolerance-days", type=int, default=21,
                    help="Max days between scheduled FPL and played provider dates for nearest-date recovery.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    if not args.fpl_root.exists():
        raise SystemExit(f"FPL root not found: {args.fpl_root}")
    try:
        season_dirs = [d.name for d in args.fpl_root.iterdir() if d.is_dir()]
    except FileNotFoundError:
        season_dirs = []

    seasons = [args.season] if args.season else sorted(season_dirs)
    if not seasons:
        raise SystemExit("No seasons found")

    run_batch(
        seasons=seasons,
        fpl_root=args.fpl_root,
        fbref_league=args.fbref_league_dir,
        understat_league=args.understat_league_dir,
        whoscored_league=args.whoscored_league_dir,
        team_map=args.team_map,
        short_map=args.short_map,
        out_dir=args.out_dir,
        attach_fdr=args.attach_fdr,
        features_root=args.features_root,
        views_subdir=args.views_subdir,
        match_tolerance_days=args.match_tolerance_days,
        force=args.force,
        bootstrap=args.bootstrap,
        league=args.league,
    )

if __name__ == "__main__":
    main()
