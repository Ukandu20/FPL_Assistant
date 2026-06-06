#!/usr/bin/env python3
"""
scripts.fpl_pipeline.clean.assign_game_ids

Assign FBref game_id (from summary.csv) to processed FPL GW rows.
Build matches.csv from the FPL-joined rows so ROUND comes from FPL.

Join tiers:
  1) (date_played, home_id, away_id)
  2) (round,       home_id, away_id)
  3) (date_played, home,    away)
  4) (round,       home,    away)

Also derives on FPL rows:
  - date_played: prefer existing 'date_played' or 'game_date'; else from kickoff (tz via --tz)
  - time:        keep existing 'time' if present; else derive from kickoff (tz via --tz)
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import tz as dateutil_tz
from unidecode import unidecode

# ────────── season helpers ──────────

SEASON_RE_SHORT = re.compile(r"^\d{4}-\d{2}$")
SEASON_RE_LONG  = re.compile(r"^\d{4}-\d{4}$")
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")

def norm_dashes(s: str) -> str:
    return (s or "").translate(_DASHES)

def season_short(s: str) -> str:
    s = norm_dashes(s.strip())
    if SEASON_RE_SHORT.fullmatch(s):
        return s
    if SEASON_RE_LONG.fullmatch(s):
        return f"{s[:4]}-{s[-2:]}"
    return s

def season_long(s: str) -> str:
    s = norm_dashes(s.strip())
    if SEASON_RE_LONG.fullmatch(s):
        return s
    if SEASON_RE_SHORT.fullmatch(s):
        start = int(s[:4]); end = int(str(start)[:2] + s[-2:])
        return f"{start}-{end}"
    return s

# ────────── IO helpers ──────────

def read_json(p: Path) -> dict:
    data = p.read_bytes()
    if data.startswith(codecs.BOM_UTF8):
        return json.loads(data.decode("utf-8-sig"))
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return json.loads(data.decode(enc))
        except Exception:
            continue
    return json.loads(data.decode("utf-8", errors="replace"))

def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

def write_csv(p: Path, df: pd.DataFrame) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

# ────────── name / code normalisation ──────────

def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = unidecode(str(s))
    s = re.sub(r"[^\w\s\|]", " ", s, flags=re.UNICODE).lower()
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s

# ────────── kickoff parsing ──────────

def split_kickoff_into_date_time(df: pd.DataFrame, tz_name: str) -> Tuple[pd.Series, pd.Series]:
    """
    Parse kickoff from common FPL columns and return (date_played, time).
    Priority: kickoff_time > kickoff_time_formatted > kickoff_date.
    Inputs assumed UTC if timezone not specified; convert to tz_name.
    """
    src = None
    if "kickoff_time" in df.columns:
        src = df["kickoff_time"].astype(str)
    elif "kickoff_time_formatted" in df.columns:
        src = df["kickoff_time_formatted"].astype(str)
    elif "kickoff_date" in df.columns:
        src = df["kickoff_date"].astype(str)
    else:
        n = len(df)
        return pd.Series([None]*n, dtype="object"), pd.Series([None]*n, dtype="object")

    dt_utc = pd.to_datetime(src, errors="coerce", utc=True)
    # If only a YYYY-MM-DD date is provided, force midnight UTC
    mask_naive = dt_utc.isna() & src.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    if mask_naive.any():
        dt_utc.loc[mask_naive] = pd.to_datetime(src[mask_naive] + "T00:00:00Z", utc=True, errors="coerce")

    tzinfo = dateutil_tz.gettz(tz_name) or dateutil_tz.UTC
    try:
        dt_local = dt_utc.dt.tz_convert(tzinfo)
    except Exception:
        dt_local = dt_utc

    date_played = dt_local.dt.strftime("%Y-%m-%d").where(dt_local.notna(), None)
    time_local  = dt_local.dt.strftime("%H:%M").where(dt_local.notna(), None)
    return date_played, time_local

# ────────── FBref ingestion (summary.csv) ──────────

def locate_fbref_summary(fbref_root: Path, league: str, season_long_str: str, summary_name: str) -> Optional[Path]:
    base = fbref_root / league / season_long_str
    candidates = [base / "summary.csv", base / summary_name]
    if base.is_dir():
        for p in base.glob("**/*summary*.csv"):
            candidates.append(p)
    for p in candidates:
        if p.is_file():
            return p
    return None

def _pick(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None

def read_fbref_summary(fpath: Path) -> pd.DataFrame:
    """
    Normalize FBref summary.csv to per-game rows with required columns:
      ['round','date_played','home','away','home_id','away_id','game_id']
    (round here is informational; FPL round is authoritative later)
    """
    df = read_csv(fpath)

    # locate columns
    round_col = _pick(df, ["game", "round", "gw", "matchweek", "mw", "gameweek"])
    date_col  = _pick(df, ["game_date", "date", "match_date"])
    home_col  = _pick(df, ["home"])
    away_col  = _pick(df, ["away"])
    gid_col   = _pick(df, ["game_id", "match_id", "fbref_game_id", "fb_match_id"])
    if not gid_col:
        raise ValueError(f"{fpath} does not contain a recognizable game_id column")
    if not (home_col and away_col):
        raise ValueError(f"{fpath} missing 'home'/'away' columns")

    team_hex_col = _pick(df, ["team_id", "team_hex", "team_fbref_id"])
    opp_hex_col  = _pick(df, ["opponent_id", "opp_id", "opponent_hex"])
    is_home_col  = _pick(df, ["is_home", "home_flag", "home_bool"])
    is_away_col  = _pick(df, ["is_away", "away_flag", "away_bool"])

    # basic series
    rounds = pd.to_numeric(df[round_col], errors="coerce").astype("Int64") if round_col else pd.Series(pd.NA, index=df.index, dtype="Int64")
    date_played = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d").where(pd.notna(df[date_col]), None) if date_col else pd.Series([None]*len(df))
    home_code = df[home_col].astype(str).str.upper()
    away_code = df[away_col].astype(str).str.upper()
    gid = df[gid_col].astype(str)

    team_hex = df[team_hex_col].astype(str).str.lower() if team_hex_col else pd.Series([None]*len(df))
    opp_hex  = df[opp_hex_col].astype(str).str.lower()  if opp_hex_col  else pd.Series([None]*len(df))
    is_home  = df[is_home_col].astype(bool) if is_home_col else pd.Series([False]*len(df))
    is_away  = df[is_away_col].astype(bool) if is_away_col else pd.Series([False]*len(df))

    # row-level home/away ids
    home_id = np.where(is_home, team_hex, np.where(is_away, opp_hex, None))
    away_id = np.where(is_home, opp_hex,  np.where(is_away, team_hex, None))

    per_row = pd.DataFrame({
        "round": rounds,
        "date_played": date_played,
        "home": home_code,
        "away": away_code,
        "home_id": home_id,
        "away_id": away_id,
        "game_id": gid,
    })

    # robust per-game aggregation
    def first_not_null(s: pd.Series):
        idx = s.first_valid_index()
        return s.loc[idx] if idx is not None else (pd.NA if s.dtype.kind in "iu" else None)

    games = (
        per_row
        .groupby("game_id", as_index=False)
        .agg({
            "round": first_not_null,
            "date_played": first_not_null,
            "home": first_not_null,
            "away": first_not_null,
            "home_id": first_not_null,
            "away_id": first_not_null
        })
    )

    games["home"]    = games["home"].astype(str).str.upper()
    games["away"]    = games["away"].astype(str).str.upper()
    games["home_id"] = games["home_id"].astype(str).str.lower()
    games["away_id"] = games["away_id"].astype(str).str.lower()
    return games

# ────────── FPL helpers ──────────

def ensure_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def normalize_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda x: str(x).strip().lower() in {"true","1","yes","y","t"})

def derive_fpl_keys(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    """
    Ensure round/home/away ids & codes exist; derive date_played/time.
    - If 'date_played' exists, use it.
    - Else if 'game_date' exists, use it.
    - Else derive from kickoff fields (kickoff_time, kickoff_time_formatted, kickoff_date).
    - If 'time' already exists, keep it; else derive from kickoff.
    """
    out = df.copy()

    # round
    out["round"] = ensure_int(out["round"]) if "round" in out.columns else pd.Series(pd.NA, index=out.index, dtype="Int64")

    # home/away from was_home
    was_home = normalize_bool(out["was_home"]) if "was_home" in out.columns else pd.Series(False, index=out.index)
    team_id = out["team_id"] if "team_id" in out.columns else pd.Series([None]*len(out))
    opp_id  = out["opp_id"]  if "opp_id"  in out.columns else pd.Series([None]*len(out))
    team_c  = out["team_code"] if "team_code" in out.columns else pd.Series([None]*len(out))
    opp_c   = out["opp_code"]  if "opp_code"  in out.columns else pd.Series([None]*len(out))

    out["home_id"] = np.where(was_home, team_id, opp_id)
    out["away_id"] = np.where(was_home, opp_id,  team_id)
    out["home"]    = np.where(was_home, team_c,  opp_c)
    out["away"]    = np.where(was_home, opp_c,   team_c)

    # date_played preference: existing 'date_played' → 'game_date' → kickoff-derived
    if "date_played" in out.columns and out["date_played"].notna().any():
        dp = pd.to_datetime(out["date_played"], errors="coerce", utc=False).dt.strftime("%Y-%m-%d")
        out["date_played"] = dp.where(pd.notna(dp), None)
    elif "game_date" in out.columns and out["game_date"].notna().any():
        gd = pd.to_datetime(out["game_date"], errors="coerce", utc=False).dt.strftime("%Y-%m-%d")
        out["date_played"] = gd.where(pd.notna(gd), None)
    else:
        dp, tm = split_kickoff_into_date_time(out, tz_name)
        out["date_played"] = dp
        # only set time if it doesn't already exist
        if "time" not in out.columns or out["time"].isna().all():
            out["time"] = tm

    # if 'time' still missing and kickoff available, try derive it
    if "time" not in out.columns or out["time"].isna().all():
        _dp, tm2 = split_kickoff_into_date_time(out, tz_name)
        out["time"] = tm2

    return out

# ────────── joins & outputs ──────────

def attach_game_ids(merged_keys: pd.DataFrame, fbref_games: pd.DataFrame) -> pd.DataFrame:
    """
    Tiered join from FPL-derived keys to FBref per-game rows.
    Ensures a 'game_id' column exists and fills it across tiers.
    """
    df = merged_keys.copy()
    if "game_id" not in df.columns:
        df["game_id"] = pd.NA

    # 1) date + hex ids
    key1 = ["date_played","home_id","away_id"]
    m1 = fbref_games.dropna(subset=[c for c in ["home_id","away_id"] if c in fbref_games.columns])[key1 + ["game_id"]].drop_duplicates()
    df = df.merge(m1, on=key1, how="left", suffixes=("", "_fb1"))
    if "game_id_fb1" in df.columns:
        df["game_id"] = df["game_id"].fillna(df["game_id_fb1"])
        df.drop(columns=["game_id_fb1"], inplace=True)

    # 2) round + hex ids
    need = df["game_id"].isna()
    if need.any():
        key2 = ["round","home_id","away_id"]
        m2 = fbref_games.dropna(subset=[c for c in ["home_id","away_id"] if c in fbref_games.columns])[key2 + ["game_id"]].drop_duplicates()
        joined = df.loc[need, key2].merge(m2, on=key2, how="left")
        df.loc[need, "game_id"] = joined["game_id"].values

    # 3) date + codes
    need = df["game_id"].isna()
    if need.any():
        key3 = ["date_played","home","away"]
        m3 = fbref_games[key3 + ["game_id"]].dropna().drop_duplicates()
        joined = df.loc[need, key3].merge(m3, on=key3, how="left")
        df.loc[need, "game_id"] = joined["game_id"].values

    # 4) round + codes
    need = df["game_id"].isna()
    if need.any():
        key4 = ["round","home","away"]
        m4 = fbref_games[key4 + ["game_id"]].dropna().drop_duplicates()
        joined = df.loc[need, key4].merge(m4, on=key4, how="left")
        df.loc[need, "game_id"] = joined["game_id"].values

    return df

def build_matches_from_fpl(merged_with_gid: pd.DataFrame, season_short_str: str) -> pd.DataFrame:
    """
    Build matches.csv using FPL rounds (authoritative), after game_id has been attached.
    For each game_id:
      - round: mode from FPL rows (warn on ties)
      - date_played/home/away/home_id/away_id: first non-null
    """
    df = merged_with_gid[merged_with_gid["game_id"].notna()].copy()

    def mode_or_na(s: pd.Series):
        vc = s.dropna().value_counts()
        if vc.empty:
            return pd.NA
        if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
            logging.warning("matches: tie in FPL round mode for a game_id; taking first")
        return vc.index[0]

    grouped = df.groupby("game_id", as_index=False)
    out = grouped.agg({
        "round": mode_or_na,
        "date_played": lambda x: x.dropna().iloc[0] if x.notna().any() else None,
        "home":        lambda x: x.dropna().iloc[0] if x.notna().any() else None,
        "away":        lambda x: x.dropna().iloc[0] if x.notna().any() else None,
        "home_id":     lambda x: x.dropna().iloc[0] if x.notna().any() else None,
        "away_id":     lambda x: x.dropna().iloc[0] if x.notna().any() else None,
    })

    out["season_short"] = season_short_str
    cols = ["season_short","round","date_played","home","away","home_id","away_id","game_id"]
    return out[cols]

# ────────── per-season pipeline ──────────

def process_season(proc_season_dir: Path,
                   fbref_root: Path,
                   league: str,
                   summary_name: str,
                   tz_name: str) -> None:
    season = proc_season_dir.name
    short = season_short(season)
    longf = season_long(season)

    merged_csv = proc_season_dir / "gws" / "merged_gws.csv"
    if not merged_csv.is_file():
        logging.warning("[%s] missing %s; skipping", season, merged_csv)
        return

    fb_summary = locate_fbref_summary(fbref_root, league, longf, summary_name)
    if not fb_summary:
        logging.warning("[%s] FBref summary.csv not found under %s/%s/%s", season, fbref_root, league, longf)
        return
    logging.info("[%s] FBref summary: %s", season, fb_summary)

    # FBref per-game (for game_id only)
    fb_games = read_fbref_summary(fb_summary)

    # FPL merged with derived keys
    merged = read_csv(merged_csv)
    merged_keys = derive_fpl_keys(merged, tz_name=tz_name)

    # Attach game_id to merged
    merged_with_gid = attach_game_ids(merged_keys, fb_games)
    missing = int(merged_with_gid["game_id"].isna().sum())
    if missing:
        logging.warning("[%s] merged_gws: %d rows still missing game_id after all joins", season, missing)

    # reorder: insert game_id after round; date/time after it
    cols = list(merged_with_gid.columns)
    def insert_after(cols, col_to_move, after):
        if col_to_move in cols and after in cols:
            cols.insert(cols.index(after) + 1, cols.pop(cols.index(col_to_move)))
        return cols
    cols = insert_after(cols, "game_id", "round")
    cols = insert_after(cols, "date_played", "game_id")
    cols = insert_after(cols, "time", "date_played")
    merged_with_gid = merged_with_gid[cols]
    write_csv(merged_csv, merged_with_gid)
    logging.info("[%s] updated merged_gws.csv with game_id/date_played/time", season)

    # Build a compact per-game mapping from the merged itself for GW files
    keymap = merged_with_gid[["round","date_played", "time","home_id","away_id","home","away","game_id"]].drop_duplicates()

    # per-GW files — use the SAME attach function (prevents KeyError on 'game_id')
    gws_dir = proc_season_dir / "gws"
    for fp in sorted(gws_dir.glob("gw*.csv")):
        gw = read_csv(fp)
        if gw.empty:
            continue
        gwk = derive_fpl_keys(gw, tz_name=tz_name)
        res = attach_game_ids(gwk, keymap)  # keymap already has keys + game_id

        # order columns in each gw file
        cols = list(res.columns)
        cols = insert_after(cols, "game_id", "round") if "round" in cols else cols
        cols = insert_after(cols, "date_played", "game_id") if "game_id" in cols else cols
        cols = insert_after(cols, "time", "date_played") if "date_played" in cols else cols
        res = res[cols]
        write_csv(fp, res)

    # matches.csv — round from FPL
    matches = build_matches_from_fpl(merged_with_gid, short)
    matches_out = proc_season_dir / "matches" / "matches.csv"
    write_csv(matches_out, matches)
    logging.info("[%s] wrote matches table (FPL round): %s", season, matches_out)

# ────────── CLI ──────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Assign FBref game_id to FPL GW rows using summary.csv; matches.csv uses FPL round.")
    ap.add_argument("--proc-root",  required=True, type=Path, help="data/processed/fpl")
    ap.add_argument("--fbref-root", required=True, type=Path, help="data/processed/fbref")
    ap.add_argument("--league",     default="ENG-Premier League", help="League folder under fbref-root")
    ap.add_argument("--summary-name", default="summary.csv", help="If your file isn’t named summary.csv")
    ap.add_argument("--season", help="Only process one season (e.g., '2025-26' or '2025-2026').")
    ap.add_argument("--tz", default="UTC", help="Timezone for date_played/time (e.g., 'Africa/Lagos').")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    seasons = [d for d in sorted(args.proc_root.iterdir()) if d.is_dir()]
    if args.season:
        target = season_short(args.season.strip())
        seasons = [d for d in seasons if season_short(d.name) == target]
    if not seasons:
        logging.warning("No seasons found under %s", args.proc_root)
        return

    for sdir in seasons:
        logging.info("Season %s …", sdir.name)
        process_season(
            proc_season_dir=sdir,
            fbref_root=Path(args.fbref_root),
            league=args.league,
            summary_name=args.summary_name,
            tz_name=args.tz
        )

if __name__ == "__main__":
    main()
