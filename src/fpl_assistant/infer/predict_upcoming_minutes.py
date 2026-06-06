#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_upcoming_minutes.py

Forward inference for upcoming fixtures using v1.x minutes models.

New in this version:
- Supports building squads from a master teams JSON (via --teams-json) instead of a squads CSV.
- Optional --team-id-map to reconcile fixture team_id with JSON team keys.
- Pulls FDR from team_form.csv (via --team-form-csv or default fix_root/<season>/team_form.csv).

Outputs:
  1) consolidated per-player predictions CSV (--out)
  2) optional per-club summaries CSV (--club-out)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

# ----------------- Data loading -----------------

KEEP_COLS = [
    "player_id","player","pos","gw_orig","date_played","minutes",
    "is_starter","days_since_last","is_active","venue","fdr_home","fdr_away",
    "team_id","season"
]

def load_minutes(fix_root: Path, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        fp = fix_root / season / "player_minutes_calendar.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, parse_dates=["date_played"])
        if "minutes" not in df.columns and "min" in df.columns:
            df = df.rename(columns={"min":"minutes"})
        if "season" not in df.columns:
            df["season"] = season
        cols = [c for c in KEEP_COLS if c in df.columns]
        frames.append(df[cols].copy())
    if not frames:
        raise FileNotFoundError("No player_minutes_calendar.csv found for supplied seasons.")
    out = pd.concat(frames, ignore_index=True)
    out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0).clip(0, 120)
    out["gw_orig"] = pd.to_numeric(out["gw_orig"], errors="coerce")
    for c in ("is_starter","days_since_last","is_active"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # normalize ids as strings for robust joins against JSON keys
    out["player_id_str"] = out["player_id"].astype(str)
    if "pos" not in out.columns:
        out["pos"] = "MID"
    return out

def _coerce_date_played(df: pd.DataFrame) -> pd.DataFrame:
    if "date_played" in df.columns:
        df["date_played"] = pd.to_datetime(df["date_played"], errors="coerce")
        if df["date_played"].notna().any():
            return df
    for cand in ["kickoff_time", "kickoff", "date", "datetime"]:
        if cand in df.columns:
            df["date_played"] = pd.to_datetime(df[cand], errors="coerce")
            break
    if "date_played" not in df.columns or df["date_played"].isna().all():
        raise ValueError("fixtures CSV must have a date column (date_played / kickoff_time / date).")
    return df

def _normalize_team_form(tf: pd.DataFrame, season: str) -> pd.DataFrame:
    """Ensure tf has: season, gw_orig, team_id, fdr_home, fdr_away."""
    need = {"gw_orig","team_id"}
    if missing := (need - set(tf.columns)):
        raise ValueError(f"team_form CSV missing columns: {missing}")
    tf = tf.copy()
    if "season" not in tf.columns:
        tf["season"] = season
    # Handle FDR columns flexibly
    has_home = "fdr_home" in tf.columns
    has_away = "fdr_away" in tf.columns
    if not (has_home and has_away):
        if "fdr" in tf.columns:
            tf["fdr_home"] = pd.to_numeric(tf["fdr"], errors="coerce").fillna(0.0)
            tf["fdr_away"] = pd.to_numeric(tf["fdr"], errors="coerce").fillna(0.0)
        else:
            tf["fdr_home"] = 0.0
            tf["fdr_away"] = 0.0
    else:
        tf["fdr_home"] = pd.to_numeric(tf["fdr_home"], errors="coerce").fillna(0.0)
        tf["fdr_away"] = pd.to_numeric(tf["fdr_away"], errors="coerce").fillna(0.0)

    tf["gw_orig"] = pd.to_numeric(tf["gw_orig"], errors="coerce")
    tf["team_id"] = pd.to_numeric(tf["team_id"], errors="coerce")
    return tf[["season","gw_orig","team_id","fdr_home","fdr_away"]]

def load_team_form(fix_root: Path, season: str, team_form_csv: Optional[str]) -> Optional[pd.DataFrame]:
    fp = Path(team_form_csv) if team_form_csv else (fix_root / season / "team_form.csv")
    if not fp.exists():
        print(f"[warn] team_form not found at {fp}; FDR will default to 0.0")
        return None
    tf = pd.read_csv(fp)
    return _normalize_team_form(tf, season)

def load_fixtures(fix_root: Path, season: str, fixtures_csv: Optional[str],
                  team_form_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if fixtures_csv:
        fx = pd.read_csv(fixtures_csv)
    else:
        fp = fix_root / season / "fixture_calendar.csv"
        fx = pd.read_csv(fp)
    fx = _coerce_date_played(fx)

    # Ensure season
    if "season" not in fx.columns:
        fx["season"] = season

    # Ensure venue (derive from was_home if available)
    if "venue" not in fx.columns and "was_home" in fx.columns:
        fx["venue"] = np.where(fx["was_home"].astype(bool), "home", "away")
    if "venue" not in fx.columns:
        if {"home_team_id","away_team_id","team_id"}.issubset(fx.columns):
            fx["venue"] = np.where(fx["team_id"].eq(fx["home_team_id"]), "home", "away")
        else:
            fx["venue"] = "home"

    fx["gw_orig"] = pd.to_numeric(fx["gw_orig"], errors="coerce")
    fx["team_id"] = pd.to_numeric(fx["team_id"], errors="coerce")
    if "opponent_id" in fx.columns:
        fx["opponent_id"] = pd.to_numeric(fx["opponent_id"], errors="coerce")
    else:
        # derive opponent_id if both home/away team ids exist
        if {"home_team_id","away_team_id","team_id"}.issubset(fx.columns):
            is_home = fx["team_id"].eq(fx["home_team_id"])
            fx["opponent_id"] = np.where(is_home, fx["away_team_id"], fx["home_team_id"])
        else:
            fx["opponent_id"] = np.nan

    # Merge team_form FDRs
    if team_form_df is not None:
        fx = fx.merge(team_form_df, on=["season","gw_orig","team_id"], how="left")
    if "fdr_home" not in fx.columns: fx["fdr_home"] = 0.0
    if "fdr_away" not in fx.columns: fx["fdr_away"] = 0.0
    fx["fdr_home"] = pd.to_numeric(fx["fdr_home"], errors="coerce").fillna(0.0)
    fx["fdr_away"] = pd.to_numeric(fx["fdr_away"], errors="coerce").fillna(0.0)

    req = {"season","gw_orig","date_played","team_id","opponent_id","venue","fdr_home","fdr_away"}
    missing = req - set(fx.columns)
    if missing:
        raise ValueError(f"fixtures CSV missing columns even after patching: {missing}")
    return fx

# -------- Teams JSON -> squads --------

def load_team_id_map(csv_path: str) -> Dict[str, str]:
    """
    CSV columns: fixture_team_id,json_team_key
    Both are treated as strings. Returns dict {fixture_team_id_str -> json_team_key_str}
    """
    mp = pd.read_csv(csv_path, dtype=str)
    need = {"fixture_team_id","json_team_key"}
    if missing := (need - set(mp.columns)):
        raise ValueError(f"team-id-map CSV missing columns: {missing}")
    mp = mp.dropna(subset=["fixture_team_id","json_team_key"])
    return dict(zip(mp["fixture_team_id"].astype(str), mp["json_team_key"].astype(str)))

def build_squads_from_teams_json(teams_json_path: str,
                                 season: str,
                                 fixtures: pd.DataFrame,
                                 df_hist: pd.DataFrame,
                                 team_id_map_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Build a squads table: columns (player_id, player, pos, team_id)
    - Pull players from teams JSON for the given season.
    - Map JSON team keys to fixtures team_id:
        1) explicit --team-id-map CSV, or
        2) if fixtures has 'team' (abbr) column matching JSON 'name', use that, or
        3) fall back: treat fixtures.team_id (as str) == JSON key
    - Backfill pos from df_hist (mode per player_id_str). Default 'MID' if unknown.
    """
    with open(teams_json_path, "r", encoding="utf-8") as f:
        teams = json.load(f)

    # 1) collect JSON squads for season
    rows = []
    for json_key, meta in teams.items():
        career = meta.get("career", {})
        if season not in career:
            continue
        team_abbr = meta.get("name", None)  # e.g., "ARS"
        for p in career[season].get("players", []):
            rows.append({
                "json_team_key": str(json_key),
                "team_abbr": team_abbr,
                "player_id_str": str(p.get("id")),
                "player": p.get("name")
            })
    if not rows:
        raise ValueError(f"No players found in teams JSON for season {season}.")
    js = pd.DataFrame(rows)

    # 2) derive positions from history (mode pos per player)
    pos_map = (df_hist.dropna(subset=["player_id_str","pos"])
                     .groupby("player_id_str")["pos"]
                     .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1])
                     .to_dict())

    js["pos"] = js["player_id_str"].map(pos_map).fillna("MID")

    # 3) map JSON teams to fixtures team_id
    # strategy A: explicit map
    explicit_map = {}
    if team_id_map_csv:
        explicit_map = load_team_id_map(team_id_map_csv)  # fixture_team_id_str -> json_team_key_str
        # invert for lookup from json_key to fixture_team_id
        inv = {v: k for k, v in explicit_map.items()}
        js["fixture_team_id_str"] = js["json_team_key"].map(inv)

    # strategy B: via team abbr if fixtures carry 'team' (abbr)
    if "team" in fixtures.columns and js["team_abbr"].notna().any():
        # Build map from team abbr -> fixture team_id (string)
        fx_map = (fixtures[["team_id","team"]]
                  .dropna()
                  .drop_duplicates()
                  .assign(team=lambda d: d["team"].astype(str).str.upper())
                  .assign(team_id_str=lambda d: d["team_id"].astype(str)))
        abbr_to_tid = dict(zip(fx_map["team"], fx_map["team_id_str"]))
        # Only fill where missing
        js.loc[js["fixture_team_id_str"].isna() | (js["fixture_team_id_str"] == ""), "fixture_team_id_str"] = \
            js.loc[js["fixture_team_id_str"].isna() | (js["fixture_team_id_str"] == ""), "team_abbr"].map(abbr_to_tid)

    # strategy C: assume equality of keys
    js["fixture_team_id_str"] = js["fixture_team_id_str"].fillna(js["json_team_key"]).astype(str)

    # 4) finalize squads
    # align dtype to fixtures["team_id"] dtype
    fx_dtype = fixtures["team_id"].dtype
    try:
        squads = pd.DataFrame({
            "player_id": js["player_id_str"],   # keep string id; downstream code treats ids as labels only
            "player": js["player"],
            "pos": js["pos"],
            "team_id": js["fixture_team_id_str"].astype(fx_dtype, errors="ignore")
        })
    except Exception:
        squads = pd.DataFrame({
            "player_id": js["player_id_str"],
            "player": js["player"],
            "pos": js["pos"],
            "team_id": js["fixture_team_id_str"]  # string fallback
        })

    # sanity: drop empty team_id rows
    squads = squads.dropna(subset=["team_id"]).copy()
    return squads

def load_squads(squads_csv: str,
                teams_json: str,
                season: str,
                fixtures: pd.DataFrame,
                df_hist: pd.DataFrame,
                team_id_map_csv: Optional[str]) -> pd.DataFrame:
    if squads_csv:
        sq = pd.read_csv(squads_csv)
        req = {"player_id","player","pos","team_id"}
        missing = req - set(sq.columns)
        if missing:
            raise ValueError(f"squads CSV missing columns: {missing}")
        return sq
    if not teams_json:
        raise ValueError("Provide either --squads-csv or --teams-json.")
    return build_squads_from_teams_json(teams_json, season, fixtures, df_hist, team_id_map_csv)

# ----------------- Feature engineering (mirror training) -----------------

def make_hist_features(df: pd.DataFrame,
                       halflife_min: float = 2.0,
                       halflife_start: float = 3.0,
                       use_fdr: bool = True,
                       days_cap: Optional[int] = 14,
                       use_log_days: bool = False) -> pd.DataFrame:
    df = df.sort_values(["player_id","season","date_played","gw_orig"]).copy()

    if use_fdr and {"venue","fdr_home","fdr_away"}.issubset(df.columns):
        df["fdr"] = np.where(df["venue"].astype(str).str.lower().eq("home"),
                             df["fdr_home"].fillna(0.0), df["fdr_away"].fillna(0.0))
    else:
        df["fdr"] = 0.0

    df["min_lag1"] = df.groupby(["player_id","season"], sort=False)["minutes"].shift(1)
    df["played_last"] = (df["min_lag1"].fillna(0) >= 1).astype(int)

    df["min_ewm_hl2"] = (
        df.groupby(["player_id","season"], sort=False)["minutes"]
          .transform(lambda s: s.shift(1).ewm(halflife=halflife_min, adjust=False).mean())
    )

    prev_date = df.groupby(["player_id","season"], sort=False)["date_played"].shift(1)
    ds = (df["date_played"] - prev_date).dt.days
    ds = ds.clip(lower=0).fillna(14)
    df["days_feat"] = np.log1p(ds) if use_log_days else (ds if days_cap is None else ds.clip(upper=days_cap))
    df["long_gap14"] = (ds > 14).astype(int)

    df["is_starter"] = pd.to_numeric(df["is_starter"], errors="coerce").fillna(0).astype(int)
    prev_start_raw = df.groupby(["player_id","season"], sort=False)["is_starter"].shift(1)
    had_prev = prev_start_raw.notna().astype(int)
    prev_start = prev_start_raw.fillna(0).astype(int) * had_prev
    prev_bench  = (1 - prev_start) * had_prev

    df["start_lag1"] = prev_start
    df["start_rate_hl3"] = (
        df.groupby(["player_id","season"], sort=False)["is_starter"]
          .transform(lambda s: s.shift(1).ewm(halflife=halflife_start, adjust=False).mean())
    ).fillna(0.0).clip(0, 1)

    def _consecutive_ones(s: pd.Series) -> pd.Series:
        s = s.astype(int)
        grp = (s == 0).cumsum()
        return s.groupby(grp).cumsum()

    keys = [df["player_id"], df["season"]]
    df["start_streak"] = prev_start.groupby(keys, sort=False).transform(_consecutive_ones).astype(int)
    df["bench_streak"] = prev_bench.groupby(keys, sort=False).transform(_consecutive_ones).astype(int)

    pos_map = {"GK":0,"DEF":1,"MID":2,"FWD":3}
    df["pos_enc"] = df["pos"].map(pos_map).fillna(2).astype(int)

    df["team_rot3"] = 0.0  # placeholder (off by default)
    df["y60"] = (df["minutes"] >= 60).astype(int)
    df["y_played"] = (df["minutes"] > 0).astype(int)
    return df

def last_state_asof(df_hist: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    dff = df_hist[df_hist["date_played"] <= asof_date].copy()
    if dff.empty:
        dff = df_hist.copy()
    idx = (dff.sort_values(["player_id","date_played","gw_orig"])
               .groupby("player_id", as_index=False).tail(1).index)
    return dff.loc[idx].copy()

# ----------------- Upcoming grid -----------------

def select_target_gws(fixtures: pd.DataFrame, gws_str: str, next_k: int, asof_date: pd.Timestamp) -> List[int]:
    if gws_str:
        return [int(x.strip()) for x in gws_str.split(",") if x.strip()]
    fut = fixtures[fixtures["date_played"] >= asof_date].copy()
    if fut.empty:
        uniq = sorted(fixtures["gw_orig"].unique())[-next_k:]
        return uniq
    uniq = fut.sort_values("date_played")["gw_orig"].drop_duplicates().tolist()
    return uniq[:next_k]

def expand_upcoming_rows(fixtures: pd.DataFrame,
                         squads: pd.DataFrame,
                         target_gws: List[int]) -> pd.DataFrame:
    fx = fixtures[fixtures["gw_orig"].isin(target_gws)].copy()
    fx = fx[["season","gw_orig","date_played","team_id","opponent_id","venue","fdr_home","fdr_away","team"] if "team" in fx.columns
            else ["season","gw_orig","date_played","team_id","opponent_id","venue","fdr_home","fdr_away"]]
    # align dtypes for merge
    try:
        squads = squads.copy()
        squads["team_id"] = squads["team_id"].astype(fx["team_id"].dtype, errors="ignore")
    except Exception:
        pass
    up = fx.merge(squads[["player_id","player","pos","team_id"]], on="team_id", how="left")
    up = up.dropna(subset=["player_id"]).copy()
    return up

def compute_future_features(upcoming: pd.DataFrame,
                            last_state: pd.DataFrame,
                            days_cap: Optional[int] = 14,
                            use_log_days: bool = False) -> pd.DataFrame:
    carry_cols = [
        "player_id","player","pos","team_id",
        "min_lag1","min_ewm_hl2","played_last",
        "start_lag1","start_rate_hl3","start_streak","bench_streak",
        "pos_enc","date_played"
    ]
    ls = last_state[carry_cols].rename(columns={"date_played":"prev_date_played"})
    feat = upcoming.merge(ls, on=["player_id","player","pos","team_id"], how="left")

    feat["fdr"] = np.where(feat["venue"].astype(str).str.lower().eq("home"),
                           feat["fdr_home"].fillna(0.0), feat["fdr_away"].fillna(0.0))

    ds = (feat["date_played"] - feat["prev_date_played"]).dt.days
    ds = ds.clip(lower=0).fillna(14)
    feat["days_feat"] = np.log1p(ds) if use_log_days else (ds if days_cap is None else ds.clip(upper=days_cap))
    feat["long_gap14"] = (ds > 14).astype(int)

    defaults = {
        "min_lag1":0.0, "min_ewm_hl2":0.0, "played_last":0,
        "start_lag1":0, "start_rate_hl3":0.0, "start_streak":0, "bench_streak":0,
        "pos_enc":feat["pos"].map({"GK":0,"DEF":1,"MID":2,"FWD":3}).fillna(2).astype(int)
    }
    for k,v in defaults.items():
        if k != "pos_enc":
            feat[k] = pd.to_numeric(feat[k], errors="coerce").fillna(v)
        else:
            feat[k] = defaults["pos_enc"]

    feat["team_rot3"] = 0.0
    for c in ["days_feat","long_gap14","fdr"]:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0)
    return feat

# ----------------- Model loading -----------------

def load_booster(path: Path) -> Optional[lgb.Booster]:
    return lgb.Booster(model_file=str(path)) if path.exists() else None

def load_calibrator(path: Path) -> Optional[object]:
    return joblib.load(path) if path.exists() else None

def per_pos_paths(root: Path, stem: str, suffix: str) -> Dict[str, Path]:
    return {pos: root / f"{stem}_{pos}{suffix}" for pos in ["GK","DEF","MID","FWD"]}

def _resolve_store(model_root: Path, use_version: str) -> Path:
    model_root = Path(model_root)
    return (model_root / "versions" / use_version) if use_version else (model_root / "latest")

# ----------------- Prediction primitives -----------------

def predict_proba_booster(booster: lgb.Booster, X: pd.DataFrame, features: List[str]) -> np.ndarray:
    A = X[features].fillna(0).to_numpy(dtype=np.float32)
    return booster.predict(A)

def predict_reg_booster(booster: lgb.Booster, X: pd.DataFrame, features: List[str]) -> np.ndarray:
    A = X[features].fillna(0).to_numpy(dtype=np.float32)
    return booster.predict(A)

# ----------------- Summaries -----------------

def summarize_by_club(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tid, gw), g in df.groupby(["team_id","gw_orig"]):
        g = g.sort_values("pred_minutes", ascending=False).reset_index(drop=True)
        expected_xi = g.head(11)[["player","pred_minutes","p_start","p60"]]
        expected_xi_str = "; ".join([f"{r.player} ({r.pred_minutes:.1f}m, ps={r.p_start:.2f}, p60={r.p60:.2f})"
                                     for r in expected_xi.itertuples(index=False)])

        secure = g[g["p_start"] >= 0.80].sort_values("p_start", ascending=False)
        secure_str = "; ".join([f"{r.player} ({r.p_start:.2f})" for r in secure.itertuples(index=False)])

        risky = g[(g["p_start"] >= 0.70) & (g["p60"] < 0.70)].copy()
        risky["risk"] = risky["p_start"] - risky["p60"]
        risky = risky.sort_values("risk", ascending=False).head(8)
        risky_str = "; ".join([f"{r.player} (risk={r.risk:.2f})" for r in risky.itertuples(index=False)])

        bench_cam = g[(g["p_start"] < 0.50)].copy()
        bench_cam["cameo_score"] = (1.0 - bench_cam["p_start"]) * bench_cam.get("p_cameo", 1.0)
        bench_cam = bench_cam.sort_values("cameo_score", ascending=False).head(5)
        bench_cam_str = "; ".join([f"{r.player} (p_cameo={getattr(r,'p_cameo',np.nan):.2f})"
                                   for r in bench_cam.itertuples(index=False)])

        rows.append(dict(
            team_id=tid, gw_orig=gw,
            n_players=int(len(g)),
            mean_pred_minutes=float(g["pred_minutes"].mean()),
            mean_p_start=float(g["p_start"].mean()),
            expected_xi=expected_xi_str,
            secure_starters=secure_str,
            risky_starters=risky_str,
            bench_cameo_candidates=bench_cam_str
        ))
    return pd.DataFrame(rows)

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True, help="e.g., data/models/minutes")
    ap.add_argument("--use-version", default="", help="e.g., v1; if empty, uses 'latest'")
    ap.add_argument("--fix-root", required=True)
    ap.add_argument("--season", required=True)
    ap.add_argument("--squads-csv", default="", help="CSV with player_id, player, pos, team_id (optional if --teams-json used)")
    ap.add_argument("--teams-json", default="", help="Master teams JSON with per-season player lists")
    ap.add_argument("--team-id-map", default="", help="CSV mapping fixture_team_id -> json_team_key (optional)")
    ap.add_argument("--fixtures-csv", default="", help="Optional fixtures CSV path for the season")
    ap.add_argument("--team-form-csv", default="", help="Optional team_form.csv path for the season (FDR source)")
    ap.add_argument("--gws", default="", help="Comma-separated GW list; ignored if --next-k > 0")
    ap.add_argument("--next-k", type=int, default=0, help="If >0, pick next-K GWs as of --asof-date")
    ap.add_argument("--asof-date", default="", help="YYYY-MM-DD; if empty uses today()")
    ap.add_argument("--out", required=True, help="Output CSV for consolidated predictions")
    ap.add_argument("--club-out", default="", help="Output CSV for per-club summaries (optional)")

    # knobs (mirror training)
    ap.add_argument("--gate-blend", type=float, default=0.3)
    ap.add_argument("--pos-thresholds", default="GK:0.15,0.55;DEF:0.25,0.70;MID:0.25,0.70;FWD:0.25,0.70")
    ap.add_argument("--no-mix-gk", action="store_true")
    ap.add_argument("--use-taper", action="store_true")
    ap.add_argument("--use-pos-bench-caps", action="store_true")

    args = ap.parse_args()

    # resolve model store
    root = Path(args.model_root)
    store = (root / "versions" / args.use_version) if args.use_version else (root / "latest")
    if not store.exists():
        raise FileNotFoundError(f"Model store not found: {store}.")
    print(f"[info] Using model store: {store}")

    # dates
    asof = pd.to_datetime(args.asof_date).normalize() if args.asof_date else pd.Timestamp.today().normalize()

    # load inputs
    fix_root = Path(args.fix_root)
    season = args.season

    # FDR source
    tf = load_team_form(fix_root, season, args.team_form_csv if args.team_form_csv else None)

    fixtures = load_fixtures(fix_root, season, args.fixtures_csv if args.fixtures_csv else None, tf)

    # choose historical seasons <= season (to compute lags & derive positions)
    seasons_hist = sorted([p.name for p in fix_root.iterdir() if p.is_dir() and p.name <= season])
    df_hist = load_minutes(fix_root, seasons_hist)
    df_hist = make_hist_features(df_hist)
    last_state = last_state_asof(df_hist, asof)

    # squads from CSV or teams JSON
    team_id_map_csv = args.team_id_map if args.team_id_map else None
    squads = load_squads(args.squads_csv, args.teams_json, season, fixtures, df_hist, team_id_map_csv)

    # target GWs & upcoming grid
    target_gws = select_target_gws(fixtures, args.gws, args.next_k, asof)
    upcoming = expand_upcoming_rows(fixtures, squads, target_gws)
    feat = compute_future_features(upcoming, last_state)

    # feature lists (mirror training)
    feat_start = ["start_lag1","start_rate_hl3","min_lag1","min_ewm_hl2",
                  "played_last","days_feat","long_gap14","start_streak","bench_streak",
                  "pos_enc","team_rot3","fdr"]
    feat_reg   = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                  "start_lag1","start_rate_hl3","pos_enc","team_rot3","fdr"]
    feat_cameo = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                  "start_rate_hl3","bench_streak","pos_enc","team_rot3","fdr"]
    feat_p60   = feat_start  # direct mode

    # load models / calibrators
    def _booster(path): return lgb.Booster(model_file=str(path)) if path.exists() else None
    def _cal(path): return joblib.load(path) if path.exists() else None

    gate_global = _booster(store / "start_classifier_global.txt")
    gate_global_iso = _cal(store / "start_classifier_global_iso.joblib")
    gate_pos = {k: _booster(store / f"start_classifier_{k}.txt") for k in ["GK","DEF","MID","FWD"]}
    gate_pos_iso = {k: _cal(store / f"start_classifier_{k}_iso.joblib") for k in ["GK","DEF","MID","FWD"]}

    reg_start = _booster(store / "reg_start.txt")
    reg_bench = _booster(store / "reg_bench.txt")

    cameo_prob_global = _booster(store / "cameo_given_bench_global.txt")
    cameo_prob_pos    = {k: _booster(store / f"cameo_given_bench_{k}.txt") for k in ["GK","DEF","MID","FWD"]}
    cameo_prob_iso_global = _cal(store / "cameo_given_bench_global_iso.joblib")
    cameo_prob_iso_pos    = {k: _cal(store / f"cameo_given_bench_{k}_iso.joblib") for k in ["GK","DEF","MID","FWD"]}

    cameo_min_global = _booster(store / "cameo_minutes_global.txt")
    cameo_min_pos    = {k: _booster(store / f"cameo_minutes_{k}.txt") for k in ["GK","DEF","MID","FWD"]}

    p60_direct = _booster(store / "p60_direct.txt")
    p60_iso    = _cal(store / "p60_direct_calib.joblib")

    caps_json = store / "bench_caps.json"
    bench_caps = None
    if caps_json.exists():
        bench_caps = pd.read_json(caps_json, typ="series").to_dict()

    # --------- p_start (per-pos blended with global) ---------
    X_all = feat
    n = len(X_all)

    def proba(model, X, feats): 
        if model is None or len(X) == 0:
            return np.zeros(len(X), dtype=float)
        A = X[feats].fillna(0).to_numpy(dtype=np.float32)
        return model.predict(A)

    p_pos = np.full(n, np.nan, dtype=float)
    for pos in ["GK","DEF","MID","FWD"]:
        m = gate_pos.get(pos)
        if m is None: continue
        mask = (X_all["pos"] == pos).values
        if not mask.any(): continue
        pv = proba(m, X_all.loc[mask], feat_start)
        iso = gate_pos_iso.get(pos)
        if iso is not None: pv = iso.transform(pv)
        p_pos[mask] = pv

    p_glob = proba(gate_global, X_all, feat_start) if gate_global is not None else np.full(n, np.nan)
    if gate_global_iso is not None and gate_global is not None:
        p_glob = gate_global_iso.transform(p_glob)

    alpha = float(np.clip(args.gate_blend, 0.0, 1.0))
    # where p_pos is available, blend; else use p_glob
    p_start = np.where(~np.isnan(p_pos), alpha * p_pos + (1.0 - alpha) * p_glob, p_glob)
    # if any remaining NaNs (rare), set to 0.5
    p_start = np.nan_to_num(p_start, nan=0.5)
    p_start = np.clip(p_start, 0, 1)

    # --------- starter & bench minutes heads ---------
    def predict_reg(model, X, feats):
        if model is None or len(X) == 0:
            return np.zeros(len(X), dtype=float)
        A = X[feats].fillna(0).to_numpy(dtype=np.float32)
        return model.predict(A)

    mu_start = np.clip(predict_reg(reg_start, X_all, feat_reg), 0, 120)
    mu_bench_min = np.clip(predict_reg(reg_bench, X_all, feat_reg), 0, 120)

    # cameo probability (per-pos, fallback global)
    p_cameo = np.zeros(n, dtype=float)
    used = np.zeros(n, dtype=bool)
    for pos in ["GK","DEF","MID","FWD"]:
        m = cameo_prob_pos.get(pos)
        if m is None: continue
        mask = (X_all["pos"] == pos).values
        if not mask.any(): continue
        pv = proba(m, X_all.loc[mask], feat_cameo)
        iso = cameo_prob_iso_pos.get(pos)
        if iso is not None: pv = iso.transform(pv)
        p_cameo[mask] = pv
        used[mask] = True
    if (~used).any() and cameo_prob_global is not None:
        pv = proba(cameo_prob_global, X_all.loc[~used], feat_cameo)
        if cameo_prob_iso_global is not None:
            pv = cameo_prob_iso_global.transform(pv)
        p_cameo[~used] = pv
    p_cameo = np.clip(p_cameo, 0, 1)

    # cameo minutes (if trained); else use bench regressor minutes
    if any(cameo_min_pos.values()) or (cameo_min_global is not None):
        mu_cameo = np.zeros(n, dtype=float)
        used = np.zeros(n, dtype=bool)
        for pos in ["GK","DEF","MID","FWD"]:
            m = cameo_min_pos.get(pos)
            if m is None: continue
            mask = (X_all["pos"] == pos).values
            if mask.any():
                mu_cameo[mask] = np.clip(predict_reg(m, X_all.loc[mask], feat_reg), 0, 120)
                used[mask] = True
        if (~used).any() and cameo_min_global is not None:
            mu_cameo[~used] = np.clip(predict_reg(cameo_min_global, X_all.loc[~used], feat_reg), 0, 120)
        mu_bench = p_cameo * mu_cameo
    else:
        mu_bench = mu_bench_min.copy()

    # per-pos bench caps
    if args.use_pos_bench_caps and bench_caps is not None:
        cap = X_all["pos"].map(bench_caps).fillna(25.0).to_numpy()
        mu_bench = np.minimum(mu_bench, cap)

    # taper starter minutes if requested
    if args.use_taper:
        w = np.clip((p_start - 0.2) / (0.6 - 0.2), 0.0, 1.0)
        mu_start = (0.5 + 0.5 * w) * mu_start

    # --------- per-position thresholds & GK routing ---------
    t_map: Dict[str, Tuple[float,float]] = {}
    for part in args.pos_thresholds.split(";"):
        if not part.strip(): continue
        tag, pair = part.split(":")
        lo, hi = pair.split(",")
        t_map[tag.strip()] = (float(lo), float(hi))
    pos_arr = X_all["pos"].values
    t_lo = np.vectorize(lambda p: t_map.get(p, (0.25, 0.70))[0])(pos_arr).astype(float)
    t_hi = np.vectorize(lambda p: t_map.get(p, (0.25, 0.70))[1])(pos_arr).astype(float)

    mu_mix = p_start * mu_start + (1.0 - p_start) * mu_bench
    hard_bench = p_start <= t_lo
    hard_start = p_start >= t_hi
    routed = mu_mix.copy()
    routed[hard_bench] = mu_bench[hard_bench]
    routed[hard_start] = mu_start[hard_start]
    if args.no_mix_gk:
        is_gk = (pos_arr == "GK")
        take_start = is_gk & (p_start > t_lo)
        routed[take_start] = mu_start[take_start]
    pred_minutes = np.clip(routed, 0, 120)

    # --------- p60 and EV minutes points ---------
    if p60_direct is not None:
        p60 = proba(p60_direct, X_all, feat_p60)
        if p60_iso is not None:
            p60 = p60_iso.transform(p60)
    else:
        p60 = np.clip(p_start, 0, 1)  # fallback

    p_play = np.clip(p_start + (1.0 - p_start) * p_cameo, 0, 1)
    exp_minutes_points = np.clip(p_play + p60, 0, 2)

    # --------- write consolidated ---------
    out_df = pd.DataFrame({
        "season": X_all["season"].values,
        "gw_orig": X_all["gw_orig"].values,
        "date_played": X_all["date_played"].values,
        "team_id": X_all["team_id"].values,
        "opponent_id": X_all["opponent_id"].values,
        "player_id": X_all["player_id"].values,
        "player": X_all["player"].values,
        "pos": X_all["pos"].values,
        "p_start": p_start,
        "p_cameo": p_cameo,
        "p_play": p_play,
        "p60": p60,
        "pred_minutes": pred_minutes,
        "exp_minutes_points": exp_minutes_points
    }).sort_values(["gw_orig","team_id","player_id"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    # --------- write per-club summary ---------
    if args.club_out:
        club_df = summarize_by_club(out_df[["team_id","gw_orig","player","pos","p_start","p60","pred_minutes","exp_minutes_points"]].copy())
        Path(args.club_out).parent.mkdir(parents=True, exist_ok=True)
        club_df.sort_values(["gw_orig","team_id"]).to_csv(args.club_out, index=False)

    print(f"Wrote {len(out_df)} rows to {args.out}")
    if args.club_out:
        print(f"Wrote club summaries to {args.club_out}")

if __name__ == "__main__":
    main()
