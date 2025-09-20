#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minutes_forecast.py — score future GWs using fixed v1 gates/calibration + roster gating.

• Trains minutes regressors on: all history + current-season rows strictly before an as-of timestamp.
• Predicts for a GW window (default next 3 GWs).
• If player future rows are missing, synthesizes them from the team fixture calendar + an as-of squad snapshot.
• Optional roster gate (master_teams.json): only predict players on the future-season roster (optionally filtered by league).
• Robust GW selection:
    - Prefer gw_played only if >0, else fall back to gw_orig, else gw.
    - Soft-selects the next available GWs ≥ --as-of-gw (up to --n-future).

Reads fixtures from:
  <fix-root>/<season>/player_fixture_calendar.csv
Fallback for future rows from:
  <fix-root>/<season>/fixture_calendar.csv
Reads FDR (optional) from:
  <form-root>/<version>/<season>/team_form.csv

NEW:
• Auto-name outputs from the GW window and dual writer (CSV/Parquet):
  - --out-format {csv,parquet,both} (default: csv)
  - --zero-pad-filenames to emit GW05_07 instead of GW5_7
• Legacy metadata in output (sourced from fixture_calendar.csv):
  - game_id (fbref_id), team, opponent_id, opponent
• FDR:
  - Venue-consistent, DGW-safe attach using (season, team_id, GW, is_home).
  - fdr emitted as INT and considered legacy metadata (with gw_played/gw_orig).
"""

import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from typing import Optional, Dict, List, Tuple, Set

from scripts.models.minutes_model_builder import (
    _pick_gw_col, load_minutes, make_features,
    train_regressor, train_cameo_minutes_by_pos,
    parse_pos_thresholds, per_position_bench_cap_from_train,
    predict_with_model, taper_start_minutes
)

# ----------------------------- utils ------------------------------------------

def load_booster(path: Path) -> lgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return lgb.Booster(model_file=str(path))

def predict_booster_prob(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    Xn = X.select_dtypes(include=[np.number]).fillna(0)
    p = booster.predict(Xn)  # binary → prob
    return np.clip(p.astype(float), 0.0, 1.0)

def coerce_ts(x, tz: str | None) -> pd.Timestamp:
    t = pd.to_datetime(x, errors="coerce")
    if t.tzinfo is None:
        return t.tz_localize(tz) if tz else t
    return t.tz_convert(tz) if tz else t

def build_date_used(df: pd.DataFrame, tz: str | None) -> pd.Series:
    """Prefer date_played; fallback to date_sched; localize/convert to tz."""
    dp = pd.to_datetime(df.get("date_played"), errors="coerce")
    ds = pd.to_datetime(df.get("date_sched"), errors="coerce")
    out = dp.where(dp.notna(), ds)
    if tz:
        if out.dt.tz is None:
            out = out.dt.tz_localize(tz)
        else:
            out = out.dt.tz_convert(tz)
    return out

def numeric_series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series aligned to df.index if column missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")

def gw_coalesce_for_future(df: pd.DataFrame) -> pd.Series:
    """GW key for selection: gw_played>0 → gw_orig>0 → gw>0."""
    gw_played = numeric_series_or_nan(df, "gw_played")
    gw_orig   = numeric_series_or_nan(df, "gw_orig")
    gw_any    = numeric_series_or_nan(df, "gw")
    gw_played_valid = gw_played.where(gw_played > 0)
    gw_orig_valid   = gw_orig.where(gw_orig > 0)
    gw_any_valid    = gw_any.where(gw_any > 0)
    gwn_pred = gw_played_valid.where(
        gw_played_valid.notna(),
        gw_orig_valid.where(gw_orig_valid.notna(), gw_any_valid)
    )
    return gwn_pred

def load_team_fixtures(fix_root: Path, season: str, filename: str) -> pd.DataFrame:
    """Load team fixtures (scheduled + played) for a season."""
    path = fix_root / season / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing team fixtures: {path}")
    tf = pd.read_csv(path)

    # tolerant date parsing
    for dc in ("date_sched", "date_played"):
        if dc in tf.columns:
            tf[dc] = pd.to_datetime(tf[dc], errors="coerce")

    # normalize is_home
    if "is_home" not in tf.columns:
        if "was_home" in tf.columns:
            tf["is_home"] = tf["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype(int)
        elif "venue" in tf.columns:
            tf["is_home"] = tf["venue"].astype(str).str.lower().eq("home").astype(int)
        else:
            tf["is_home"] = 0

    # Ensure numeric GWs
    for c in ("gw_played","gw_orig","gw"):
        if c in tf.columns:
            tf[c] = pd.to_numeric(tf[c], errors="coerce")

    # Team id column naming tolerance
    if "team_id" not in tf.columns:
        for alt in ["team", "teamId", "team_code"]:
            if alt in tf.columns:
                tf = tf.rename(columns={alt: "team_id"})
                break

    tf["season"] = season
    # ensure string team_id for stable joins
    if "team_id" in tf.columns:
        tf["team_id"] = tf["team_id"].astype(str)
    return tf

def build_asof_squad(hist: pd.DataFrame, as_of_ts: pd.Timestamp, tz: str | None) -> pd.DataFrame:
    """
    From historical minutes rows strictly before as_of_ts, build the latest-known
    (player_id → player, pos, team_id, is_active) snapshot.
    """
    h = hist.copy()
    du = build_date_used(h, tz)
    h = h[du < as_of_ts].copy()
    if h.empty:
        return pd.DataFrame(columns=["player_id","player","pos","team_id","is_active"])

    gw_key = _pick_gw_col(h.columns.tolist()) or "gw_orig"
    h = h.sort_values(["player_id", "season", du.name, gw_key])
    last = h.groupby("player_id", as_index=False).tail(1)

    for c in ["player","pos","team_id","is_active"]:
        if c not in last.columns:
            last[c] = 1 if c == "is_active" else np.nan

    return last[["player_id","player","pos","team_id","is_active"]].dropna(subset=["team_id"])

def synthesize_future_player_rows(team_fix: pd.DataFrame,
                                  squad: pd.DataFrame,
                                  target_gws: list[int]) -> pd.DataFrame:
    """
    Cross-join each target team fixture with as-of squad rows of the same team_id
    to create player-level future rows (minutes/is_starter unknown).
    """
    tf = team_fix[team_fix["season"].notna()].copy()
    gwn_team = gw_coalesce_for_future(tf)
    tf = tf[gwn_team.isin(target_gws)].copy()
    if tf.empty or squad.empty:
        return pd.DataFrame(columns=[
            "season","gw_orig","gw_played","gw","date_sched","player_id","player","team_id","pos","is_home",
            "minutes","is_starter","is_active","_is_synth"
        ])

    keep_cols = [c for c in ["season","gw_orig","gw_played","gw","date_sched","team_id","is_home"] if c in tf.columns]
    tf_small = tf[keep_cols].copy()

    synth = tf_small.merge(squad, how="left", on="team_id", suffixes=("", ""))
    synth["minutes"] = np.nan
    synth["is_starter"] = np.nan
    if "is_active" not in synth.columns:
        synth["is_active"] = 1
    synth["_is_synth"] = 1

    for c in ["gw_orig","gw_played","gw"]:
        if c in synth.columns:
            synth[c] = pd.to_numeric(synth[c], errors="coerce")

    return synth

# ----------------------------- roster gating ----------------------------------

def _load_roster_pairs(teams_json: Optional[Path],
                       season: str,
                       league_filter: Optional[str]) -> Optional[Set[Tuple[str, str]]]:
    """
    Returns a set of allowed (team_id, player_id) pairs for the given season
    from master_teams-like JSON (career-season structure).
    If teams_json is None or file missing, returns None (no explicit gating).
    """
    if not teams_json:
        return None
    p = Path(teams_json)
    if not p.exists():
        logging.warning("teams_json not found at %s — skipping roster gate.", p)
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning("Failed to parse teams_json (%s): %s — skipping roster gate.", p, e)
        return None

    allowed: Set[Tuple[str, str]] = set()
    # master_teams: { team_id: { name, career: { "<season>": {league, players:[{id,name},..] } } } }
    for team_id, obj in (data or {}).items():
        career = (obj or {}).get("career", {})
        season_info = career.get(season)
        if not season_info:
            continue
        if league_filter:
            if str(season_info.get("league", "")).strip().lower() != str(league_filter).strip().lower():
                continue
        players = season_info.get("players", []) or []
        for pl in players:
            pid = str(pl.get("id", "")).strip()
            if pid:
                allowed.add((str(team_id), pid))
    if not allowed:
        logging.warning("Roster map for %s produced 0 allowed pairs (league=%r).", season, league_filter)
    return allowed or None

def _implicit_future_roster(df_all: pd.DataFrame,
                            future_season: str,
                            as_of_ts: pd.Timestamp,
                            tz: str | None) -> Set[Tuple[str, str]]:
    """
    Derive implicit (team_id, player_id) pairs seen for the FUTURE season
    in rows strictly before as_of_ts (guards against stale JSON).
    """
    du = build_date_used(df_all, tz)
    m = (df_all["season"] == future_season) & (du < as_of_ts)
    sub = df_all.loc[m, ["team_id", "player_id"]].dropna()
    return set((str(t), str(p)) for t, p in sub.to_records(index=False))

def _apply_roster_gate(df: pd.DataFrame,
                       allowed_pairs: Optional[Set[Tuple[str, str]]],
                       season: str,
                       where: str,
                       out_artifacts_dir: Optional[Path] = None,
                       require_on_roster: bool = False) -> pd.DataFrame:
    """
    Keep only rows with (team_id, player_id) inside allowed_pairs for the season.
    If require_on_roster is True and any rows are dropped, raise RuntimeError.
    """
    if allowed_pairs is None or df.empty:
        return df

    # Normalize to strings for stable matching
    tid = df.get("team_id").astype(str)
    pid = df.get("player_id").astype(str)
    mask_ok = tid.combine(pid, lambda a, b: (a, b) in allowed_pairs)

    dropped = int((~mask_ok).sum())
    if dropped:
        logging.info("Roster gate dropped %d %s row(s) not present on the %s roster.", dropped, where, season)
        if out_artifacts_dir is not None:
            out_artifacts_dir.mkdir(parents=True, exist_ok=True)
            df.loc[~mask_ok].to_csv(out_artifacts_dir / f"roster_dropped_{where}.csv", index=False)
        if require_on_roster:
            raise RuntimeError(f"--require-on-roster set: {dropped} {where} rows are not on the {season} roster.")
    return df.loc[mask_ok].copy()

# ----------------------------- future feature freeze --------------------------

def _freeze_future_features(df: pd.DataFrame,
                            date_used: pd.Series,
                            as_of_ts: pd.Timestamp) -> pd.DataFrame:
    """
    For synthetic rows (_is_synth==1), freeze lag/streak features to the
    last real game snapshot per player before as_of_ts. Only 'days_feat'
    and 'long_gap14' evolve with the scheduled date difference.
    """
    if "_is_synth" not in df.columns or not df["_is_synth"].fillna(0).eq(1).any():
        return df

    # Collect last real snapshot per player pre-as-of
    mask_hist = df["minutes"].notna() & (date_used < as_of_ts)
    if not mask_hist.any():
        return df

    gw_key = _pick_gw_col(df.columns.tolist()) or "gw_orig"
    snap = (df.loc[mask_hist, ["player_id", "start_lag1","played_last","start_rate_hl3",
                               "min_lag1","min_ewm_hl2","start_streak","bench_streak", gw_key]]
              .assign(_date_used=date_used.loc[mask_hist])
              .sort_values(["player_id","_date_used", gw_key])
              .groupby("player_id", as_index=False).tail(1)
              .set_index("player_id"))

    out = df.copy()
    m_syn = out["_is_synth"].fillna(0).eq(1)

    freeze_cols = [c for c in
                   ["start_lag1","played_last","start_rate_hl3","min_lag1","min_ewm_hl2","start_streak","bench_streak"]
                   if c in out.columns and c in snap.columns]

    for c in freeze_cols:
        out.loc[m_syn, c] = out.loc[m_syn, "player_id"].map(snap[c])

    # Recompute days_feat / long_gap14 on synthetic rows from last real played date
    last_date = date_used.loc[mask_hist].groupby(df.loc[mask_hist, "player_id"]).max()
    ds = (pd.to_datetime(date_used.loc[m_syn], errors="coerce")
          - out.loc[m_syn, "player_id"].map(last_date)).dt.days.clip(lower=0)
    out.loc[m_syn, "days_feat"] = ds.clip(upper=14)
    out.loc[m_syn, "long_gap14"] = (ds > 14).astype(int)

    return out

# ----------------------------- player-specific bench caps ---------------------

def _weighted_quantile(values: np.ndarray,
                       weights: np.ndarray,
                       q: float) -> float:
    """Compute weighted quantile (0..1) with stable handling of edge cases."""
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    v = values[order]
    w = np.asarray(weights, dtype=float)[order]
    w = np.clip(w, 0.0, np.inf)
    if not np.isfinite(w).any() or w.sum() <= 0:
        return float(v[-1])
    cw = np.cumsum(w)
    t = q * w.sum()
    idx = np.searchsorted(cw, t, side="right")
    idx = int(np.clip(idx, 0, len(v)-1))
    return float(v[idx])

def _build_player_bench_caps(cameo_hist: pd.DataFrame,
                             pos_caps: Dict[str, float],
                             halflife_matches: float = 8.0,
                             shrink_k: float = 6.0,
                             winsor_to_pos: bool = True) -> Dict[str, float]:
    """
    For each player, compute a recency-weighted 95th percentile (bench-only minutes),
    shrink toward the position cap with strength 'shrink_k', and optionally winsor
    to the position-level 95th cap.
    Returns: {player_id: cap_minutes}
    """
    if cameo_hist.empty:
        return {}

    # Ensure we have ordering
    if "_du_for_caps" not in cameo_hist.columns:
        cameo_hist = cameo_hist.copy()
        cameo_hist["_du_for_caps"] = pd.to_datetime(
            cameo_hist.get("date_played"), errors="coerce"
        ).where(lambda s: s.notna(),
                pd.to_datetime(cameo_hist.get("date_sched"), errors="coerce"))

    out: Dict[str, float] = {}
    for pid, grp in cameo_hist.sort_values(["player_id", "_du_for_caps"]).groupby("player_id"):
        g = grp.dropna(subset=["minutes", "pos"]).copy()
        if g.empty:
            continue
        minutes = g["minutes"].to_numpy(dtype=float)
        n = len(minutes)

        # recency weights: newest has weight 1.0; decay by halflife in matches backwards
        age = np.arange(n, 0, -1) - 1  # 0 for newest, (n-1) for oldest
        weights = np.power(0.5, age / max(1e-6, float(halflife_matches)))

        q95 = _weighted_quantile(minutes, weights, 0.95)
        pos = str(g["pos"].iloc[-1])
        pos_cap = float(pos_caps.get(pos, 45.0))

        if winsor_to_pos and np.isfinite(q95):
            q95 = float(np.minimum(q95, pos_cap))

        # shrink toward position cap by weighted "effective sample size"
        n_eff = float(weights.sum())
        cap = (n_eff * q95 + float(shrink_k) * pos_cap) / (n_eff + float(shrink_k))
        out[str(pid)] = float(np.clip(cap, 0.0, pos_cap if winsor_to_pos else np.inf))
    return out

# ----------------------------- I/O helpers (NEW) ------------------------------

def _fmt_gw(n: int, zero_pad: bool) -> str:
    return f"{int(n):02d}" if zero_pad else f"{int(n)}"

def _out_paths(base_dir: Path, season: str, gw_from: int, gw_to: int, zero_pad: bool, out_format: str) -> List[Path]:
    """
    Build one or two output paths depending on out_format: csv|parquet|both.
    """
    a = _fmt_gw(gw_from, zero_pad); b = _fmt_gw(gw_to, zero_pad)
    season_dir = base_dir / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    stem = season_dir / f"GW{a}_{b}"
    if out_format == "csv":
        return [Path(str(stem) + ".csv")]
    if out_format == "parquet":
        return [Path(str(stem) + ".parquet")]
    # both
    return [Path(str(stem) + ".csv"), Path(str(stem) + ".parquet")]

def _write_minutes(out: pd.DataFrame, paths: List[Path]) -> List[str]:
    """
    Write CSV and/or Parquet. Ensures date columns are written cleanly.
    Returns list of stringified paths written.
    """
    written: List[str] = []
    for p in paths:
        if p.suffix.lower() == ".csv":
            # normalize date column to date-only string for stability
            if "date_sched" in out.columns:
                tmp = out.copy()
                tmp["date_sched"] = pd.to_datetime(tmp["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                tmp = out
            tmp.to_csv(p, index=False)
            written.append(str(p))
        elif p.suffix.lower() == ".parquet":
            out.to_parquet(p, index=False)
            written.append(str(p))
        else:
            raise ValueError(f"Unsupported output extension: {p.suffix}")
    return written

# ----------------------------- Legacy metadata attach (NEW) -------------------

def _attach_legacy_meta(df_pred: pd.DataFrame, team_fix: pd.DataFrame) -> pd.DataFrame:
    """
    Attach legacy metadata to df_pred from team fixtures:
      - game_id (fbref_id), team, opponent_id, opponent (home/away derived)
    Join keys: (season, team_id, gw_key), where gw_key is gw_played>0 → gw_orig>0 → gw>0.
    """
    if df_pred.empty or team_fix.empty:
        # Ensure columns exist even if empty
        for col in ["fbref_id", "team", "opponent_id", "opponent"]:
            if col not in df_pred.columns:
                df_pred[col] = np.nan
        return df_pred

    df = df_pred.copy()

    # normalize join keys
    gw_df = _pick_gw_col(df.columns.tolist()) or "gw_orig"
    gw_tf = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"

    for c in [gw_df]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    tf = team_fix.copy()
    for c in [gw_tf]:
        if c in tf.columns:
            tf[c] = pd.to_numeric(tf[c], errors="coerce")

    if "team_id" in df.columns:
        df["team_id"] = df["team_id"].astype(str)
    if "team_id" in tf.columns:
        tf["team_id"] = tf["team_id"].astype(str)

    # select and rename for merge
    keep = ["season", "team_id", gw_tf, "fbref_id", "team", "opponent_id", "is_home", "home", "away"]
    keep = [c for c in keep if c in tf.columns]
    tf_small = (tf[keep]
                .dropna(subset=[gw_tf, "team_id"])
                .drop_duplicates())
    if gw_tf != gw_df:
        tf_small = tf_small.rename(columns={gw_tf: gw_df})

    merged = df.merge(
        tf_small,
        how="left",
        on=["season", "team_id", gw_df],
        validate="many_to_one",
        copy=False,
        suffixes=("", "_fix")
    )

    # Compute opponent if missing; prefer fixture names for team/opponent
    if "opponent" not in merged.columns or merged["opponent"].isna().all():
        if {"home", "away", "is_home"}.issubset(merged.columns):
            ih = pd.to_numeric(merged["is_home"], errors="coerce").fillna(0).astype(int)
            merged["opponent"] = np.where(ih == 1, merged.get("away"), merged.get("home"))

    # Ensure columns exist
    for col in ["fbref_id", "team", "opponent_id", "opponent"]:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged

# ----------------------------- Venue-consistent, DGW-safe FDR attach ----------

def _find_fdr_cols(cols: set[str]) -> tuple[str, str]:
    home_aliases = ["fdr_home", "team_fdr_home", "def_fdr_home", "fdrH"]
    away_aliases = ["fdr_away", "team_fdr_away", "def_fdr_away", "fdrA"]
    home = next((c for c in home_aliases if c in cols), None)
    away = next((c for c in away_aliases if c in cols), None)
    if not home or not away:
        raise RuntimeError(
            f"FDR columns not found in team_form. "
            f"Tried {home_aliases} and {away_aliases}. Got: {sorted(cols)}"
        )
    return home, away

def _ensure_is_home(df: pd.DataFrame) -> pd.DataFrame:
    if "is_home" in df.columns:
        out = df.copy()
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype(int)
        return out
    out = df.copy()
    if "was_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["was_home"], errors="coerce").fillna(0).astype(int)
        return out
    if "venue" in out.columns:
        out["is_home"] = out["venue"].astype(str).str.lower().eq("home").astype(int)
        return out
    raise RuntimeError("No venue columns found to compute is_home.")

def attach_fdr_consistent(df: pd.DataFrame,
                          seasons_all: List[str],
                          form_root: Path,
                          version: str,
                          team_fix: pd.DataFrame) -> pd.DataFrame:
    """
    DGW-safe venue-consistent FDR attach:

    1) Ensure df has is_home (derive/merge from team_fix if needed).
    2) Load team_form for each season and pick GW key like scorer.
    3) Expand team_form rows into a long table keyed by
       (season, team_id, GW, is_home) with a single 'fdr_side':
         - emit (is_home=1, fdr_side=fdr_home) and (is_home=0, fdr_side=fdr_away).
    4) Collapse duplicates for the same key by worst-case (max fdr_side).
    5) Merge many-to-one on (season, team_id, GW, is_home) and set df["fdr"] as INT.
    """
    if df.empty:
        return df

    # Ensure df has is_home; if missing, derive from team_fix
    if "is_home" not in df.columns:
        gw_df = _pick_gw_col(df.columns.tolist()) or "gw_orig"
        gw_tf = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"
        tf_small = team_fix[["season","team_id",gw_tf,"is_home"]].dropna(subset=[gw_tf,"team_id"]).drop_duplicates()
        if gw_tf != gw_df:
            tf_small = tf_small.rename(columns={gw_tf: gw_df})
        df = df.merge(tf_small, how="left", on=["season","team_id",gw_df], validate="many_to_one")

    df = _ensure_is_home(df)
    df["team_id"] = df["team_id"].astype(str)
    for c in ("gw_played","gw_orig","gw"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    gw_df = _pick_gw_col(df.columns.tolist())
    if gw_df is None:
        raise RuntimeError("attach_fdr_consistent: no GW column among ['gw_played','gw_orig','gw'].")

    seasons_to_load = sorted(set(df["season"].astype(str)) & set(map(str, seasons_all)))
    if not seasons_to_load:
        raise RuntimeError("attach_fdr_consistent: df seasons do not overlap seasons_all.")

    long_parts = []
    for season in seasons_to_load:
        fp = form_root / version / season / "team_form.csv"
        if not fp.exists():
            raise FileNotFoundError(f"attach_fdr_consistent: missing team_form: {fp}")
        tf = pd.read_csv(fp)
        tf["team_id"] = tf.get("team_id", pd.Series(index=tf.index, dtype=object)).astype(str)
        for c in ("gw_played","gw_orig","gw"):
            if c in tf.columns:
                tf[c] = pd.to_numeric(tf[c], errors="coerce")

        home_col, away_col = _find_fdr_cols(set(tf.columns))
        gw_tf = _pick_gw_col(tf.columns.tolist())
        if gw_tf is None:
            raise RuntimeError(f"attach_fdr_consistent: no GW column in team_form: {fp}")

        base = tf[["season","team_id",gw_tf,home_col,away_col]].dropna(subset=["team_id", gw_tf])
        # expand to long: is_home=1 (home_col), is_home=0 (away_col)
        home_rows = base.rename(columns={home_col: "fdr_side"}).assign(is_home=1)[["season","team_id",gw_tf,"is_home","fdr_side"]]
        home_rows = home_rows.rename(columns={away_col: "_drop"})
        away_rows = base.rename(columns={away_col: "fdr_side"}).assign(is_home=0)[["season","team_id",gw_tf,"is_home","fdr_side"]]
        away_rows = away_rows.rename(columns={home_col: "_drop"})
        part = pd.concat([home_rows, away_rows], ignore_index=True)

        if gw_tf != gw_df:
            part = part.rename(columns={gw_tf: gw_df})

        long_parts.append(part)

    form_long = pd.concat(long_parts, ignore_index=True)

    # Collapse duplicates by worst-case difficulty (conservative)
    form_long = (form_long
                 .groupby(["season","team_id",gw_df,"is_home"], as_index=False)["fdr_side"]
                 .max())

    # Merge many-to-one on (season, team_id, GW, is_home)
    merged = df.merge(
        form_long,
        how="left",
        on=["season","team_id",gw_df,"is_home"],
        validate="many_to_one",
        copy=False
    )

    # Coerce to integer, fail loudly if any missing
    if merged["fdr_side"].isna().any():
        miss = merged.loc[merged["fdr_side"].isna(), ["season","team_id",gw_df,"is_home"]].drop_duplicates()
        logging.error("attach_fdr_consistent: missing FDR for %d rows. Examples:\n%s",
                      len(miss), miss.head(20).to_string(index=False))
        raise RuntimeError("attach_fdr_consistent: FDR merge produced NaNs. Check keys/coverage.")

    merged["fdr"] = pd.to_numeric(merged["fdr_side"], errors="coerce").astype("Int64")
    merged = merged.drop(columns=["fdr_side"], errors="ignore")
    return merged

# ----------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Windows
    ap.add_argument("--history-seasons", required=True, help="Comma list of past seasons")
    ap.add_argument("--future-season", required=True, help="Season to score (contains future GWs)")
    ap.add_argument("--as-of", default="now", help='ISO timestamp or "now" (default)')
    ap.add_argument("--as-of-tz", default="Africa/Lagos", help="Timezone to localize naive dates")
    ap.add_argument("--as-of-gw", type=int, required=True, help="First GW not yet played at --as-of (the next GW)")
    # Range control
    ap.add_argument("--n-future", type=int, default=3, help="Number of future GWs to score (default 3)")
    ap.add_argument("--gw-from", type=int, default=None, help="Override start GW; default = --as-of-gw")
    ap.add_argument("--gw-to", type=int, default=None, help="Override end GW; default = gw-from + n-future - 1")
    ap.add_argument("--strict-n-future", action="store_true",
                    help="If set, error out when fewer than --n-future GWs are available (default: proceed with what's available)")

    # IO
    ap.add_argument("--fix-root", default="data/processed/registry/fixtures")
    ap.add_argument("--team-fixtures-filename", default="fixture_calendar.csv",
                    help="Filename of team fixtures under <fix-root>/<season>/ (default: fixture_calendar.csv)")
    ap.add_argument("--out-dir", default="data/predictions/minutes")
    ap.add_argument("--out-format", choices=["csv","parquet","both"], default="csv",
                    help="Output format for minutes (default: csv)")
    ap.add_argument("--zero-pad-filenames", action="store_true",
                    help="Write filenames as GW05_07 instead of GW5_7")

    ap.add_argument("--model-dir", default="data/models/minutes/versions/v1")

    # Features (mirror v1)
    ap.add_argument("--use-fdr", action="store_true")
    ap.add_argument("--form-root", default="data/processed/registry/features")
    ap.add_argument("--form-version", default="v2")
    ap.add_argument("--form-source", choices=["team","player"], default="team")

    # Routing config (v1)
    ap.add_argument("--gate-blend", type=float, default=0.25)
    ap.add_argument("--t-lo", type=float, default=0.25)
    ap.add_argument("--t-hi", type=float, default=0.65)
    ap.add_argument("--pos-thresholds", type=str, default="")
    ap.add_argument("--no-mix-gk", action="store_true")

    # Taper + caps
    ap.add_argument("--use-taper", action="store_true")
    ap.add_argument("--taper-lo", type=float, default=0.40)
    ap.add_argument("--taper-hi", type=float, default=0.70)
    ap.add_argument("--taper-min-scale", type=float, default=0.80)
    ap.add_argument("--use-pos-bench-caps", action="store_true")
    ap.add_argument("--bench-cap", type=float, default=45.0)

    # Roster gating
    ap.add_argument("--teams-json", type=Path, help="Path to master_teams.json containing per-season rosters")
    ap.add_argument("--league-filter", type=str, default="",
                    help="Optional league name to filter roster for the future season (e.g., 'ENG-Premier League')")
    ap.add_argument("--require-on-roster", action="store_true",
                    help="If set, error out when any scoring rows are not on the future-season roster")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    fix_root = Path(args.fix_root)

    # Resolve GW range requested
    gw_from_req = args.gw_from if args.gw_from is not None else args.as_of_gw
    gw_to_req = args.gw_to if args.gw_to is not None else (gw_from_req + max(1, args.n_future) - 1)

    # Load fixtures for history + (whatever player rows exist) future
    history = [s.strip() for s in args.history_seasons.split(",") if s.strip()]
    seasons_all = history + [args.future_season]
    df = load_minutes(seasons_all, fix_root)

    # Determine as-of
    as_of_ts = (pd.Timestamp.now(tz=args.as_of_tz)
                if str(args.as_of).lower() in ("now", "auto", "today")
                else coerce_ts(args.as_of, args.as_of_tz))

    # If player future rows are missing for future season, synthesize from team fixtures
    team_fix = load_team_fixtures(fix_root, args.future_season,
                                  args.team_fixtures_filenames if hasattr(args, "team_fixtures_filenames") else args.team_fixtures_filename)
    gwn_team = gw_coalesce_for_future(team_fix)
    avail_gws = sorted(pd.unique(gwn_team.dropna().astype(int)))
    target_gws = [g for g in avail_gws if g >= gw_from_req][:args.n_future]
    if not target_gws:
        diag = {"requested_from_to": [int(gw_from_req), int(gw_to_req)], "available_team_gws": avail_gws}
        raise RuntimeError(f"No future GWs available in team fixtures for selection. Diagnostics: {diag}")
    if args.strict_n_future and len(target_gws) < args.n_future:
        raise RuntimeError(
            f"Only {len(target_gws)} GW(s) available ≥ {gw_from_req}, "
            f"but --n-future={args.n_future} and --strict-n-future was set. Available: {avail_gws}"
        )

    # Build as-of squad from history (strictly before as_of_ts)
    date_used_full = build_date_used(df, args.as_of_tz)
    hist_mask = (df["season"].isin(history)) | ((df["season"] == args.future_season) & (date_used_full < as_of_ts))
    squad = build_asof_squad(df.loc[hist_mask], as_of_ts, args.as_of_tz)

    # --- Roster gating sets: explicit (JSON) ∪ implicit (seen pre-as-of in future season)
    league = args.league_filter.strip() or None
    explicit_pairs = _load_roster_pairs(args.teams_json, args.future_season, league)
    implicit_pairs = _implicit_future_roster(df, args.future_season, as_of_ts, args.as_of_tz)
    allowed_pairs = (explicit_pairs or set()) | implicit_pairs if (explicit_pairs or implicit_pairs) else None

    # (Optional) filter the as-of squad used for synthesis — but only if we have a roster set
    if allowed_pairs is not None and not squad.empty:
        mask_squad = squad["team_id"].astype(str).combine(squad["player_id"].astype(str),
                                                          lambda a, b: (a, b) in allowed_pairs)
        dropped_squad = int((~mask_squad).sum())
        if dropped_squad:
            logging.info("Roster gate dropped %d as-of squad row(s) not on the %s roster.", dropped_squad, args.future_season)
        squad = squad.loc[mask_squad].copy()

    # Synthesize player rows for target_gws that aren't already present
    synth = synthesize_future_player_rows(team_fix, squad, target_gws)

    # Avoid duplicating real rows: drop any synth rows that already exist in df
    if not synth.empty:
        gw_key = _pick_gw_col(df.columns.tolist()) or "gw_orig"
        if gw_key not in synth.columns:
            if "gw_orig" in synth.columns: gw_key = "gw_orig"
            elif "gw_played" in synth.columns: gw_key = "gw_played"
        existing_keys = set(
            tuple(x) for x in df[["season", "player_id", gw_key, "team_id"]]
            .dropna(subset=[gw_key]).to_records(index=False)
        )
        keep_mask = ~synth.apply(
            lambda r: (r["season"], r["player_id"], r.get(gw_key, np.nan), r["team_id"]) in existing_keys,
            axis=1
        )
        synth = synth.loc[keep_mask].copy()

    # Extend df with synthesized rows (if any)
    if not synth.empty:
        df = pd.concat([df, synth], ignore_index=True)

    # -------- VENUE-CONSISTENT, DGW-SAFE FDR ATTACH --------
    if args.use_fdr:
        df = attach_fdr_consistent(
            df=df,
            seasons_all=seasons_all,
            form_root=Path(args.form_root),
            version=args.form_version,
            team_fix=team_fix
        )
        # hard cast to int now that attach_fdr_consistent guarantees no NaNs
        df["fdr"] = pd.to_numeric(df["fdr"], errors="raise").astype(int)

    # Leakage control mask computed on the extended df
    date_used = build_date_used(df, args.as_of_tz)
    pre_asof_mask = (df["season"].isin(history)) | ((df["season"] == args.future_season) & (date_used < as_of_ts))
    pre_asof_mask &= df["minutes"].notna()

    # Feature engineering (compat with old/new signatures)
    try:
        df = make_features(
            df,
            halflife_min=2.0, halflife_start=3.0,
            days_cap=14, use_log_days=False,
            use_fdr=args.use_fdr, add_team_rotation=True,
            taper_lo=args.taper_lo, taper_hi=args.taper_hi, taper_min_scale=args.taper_min_scale
        )
    except TypeError:
        df = make_features(
            df,
            halflife_min=2.0, halflife_start=3.0,
            days_cap=14, use_log_days=False,
            use_fdr=args.use_fdr, add_team_rotation=True
        )

    # Neutralize team_rot3 leakage on synthesized rows
    if "_is_synth" in df.columns and df["_is_synth"].fillna(0).eq(1).any():
        gw_key = _pick_gw_col(df.columns.tolist()) or "gw_orig"
        idx_pre = df.index[pre_asof_mask]
        tmp = df.loc[idx_pre, ["team_id", "team_rot3", "season", gw_key]].copy()
        tmp["_du"] = date_used.loc[idx_pre]
        tmp = tmp.sort_values(["team_id", "_du", gw_key])
        last_rot = tmp.groupby("team_id")["team_rot3"].last()
        mask_synth = df["_is_synth"].fillna(0).eq(1)
        df.loc[mask_synth, "team_rot3"] = df.loc[mask_synth, "team_id"].map(last_rot).fillna(0.0)

    # Freeze lag/streak features on synthetic rows to pre-as-of snapshot
    df = _freeze_future_features(df, date_used, as_of_ts)

    # ----- Build prediction slice from extended df -----
    gwn_pred = gw_coalesce_for_future(df)
    future_mask = (df["season"] == args.future_season)
    df_pred = df.loc[future_mask & gwn_pred.isin(target_gws)].copy()
    if df_pred.empty:
        diag = {
            "target_gws": target_gws,
            "future_unique_gws_in_df": sorted(pd.unique(gwn_pred[future_mask].dropna().astype(int)))
        }
        raise RuntimeError(f"No rows to score after synthesis. Diagnostics: {diag}")

    # Apply roster gate to the scoring slice (authoritative filter)
    season_dir = (out_dir / f"{args.future_season}")
    artifacts_dir = season_dir / "artifacts"
    df_pred = _apply_roster_gate(
        df_pred,
        allowed_pairs=allowed_pairs,
        season=args.future_season,
        where="minutes_scoring",
        out_artifacts_dir=artifacts_dir,
        require_on_roster=args.require_on_roster
    )
    if df_pred.empty:
        raise RuntimeError("All rows were dropped by roster gating; nothing to score.")

    # --------- attach legacy metadata (from team fixture calendar) -------------
    df_pred = _attach_legacy_meta(df_pred, team_fix)

    # Final FDR integrity check on the scored slice
    if args.use_fdr:
        if "fdr" not in df_pred.columns or df_pred["fdr"].isna().any():
            bad = df_pred.loc[df_pred.get("fdr").isna() if "fdr" in df_pred.columns else slice(None),
                              [c for c in ["season","team_id","gw_played","gw_orig","gw","is_home"] if c in df_pred.columns]].drop_duplicates()
            (artifacts_dir).mkdir(parents=True, exist_ok=True)
            bad.to_csv(artifacts_dir / "missing_fdr_rows.csv", index=False)
            raise RuntimeError("FDR required but missing in scoring slice; wrote artifacts/missing_fdr_rows.csv")
        # enforce int dtype
        df_pred["fdr"] = pd.to_numeric(df_pred["fdr"], errors="raise").astype(int)

    # Bench caps from historical bench cameos (position)
    hist = df.loc[pre_asof_mask].copy()
    cameo_hist = hist[(hist["is_starter"] == 0) & (hist["minutes"] > 0)]
    if cameo_hist.empty:
        bench_caps = {"GK": 5.0, "DEF": 20.0, "MID": 30.0, "FWD": 30.0}
    else:
        bench_caps = cameo_hist.groupby("pos")["minutes"].quantile(0.95).to_dict()

    # Load v1 gates + calibrations
    start_glob = load_booster(model_dir / "start_classifier_global.txt")
    start_pos = {p: load_booster(model_dir / f"start_classifier_{p}.txt")
                 for p in ["GK","DEF","MID","FWD"] if (model_dir / f"start_classifier_{p}.txt").exists()}
    iso_glob = joblib.load(model_dir / "start_classifier_global_iso.joblib") if (model_dir / "start_classifier_global_iso.joblib").exists() else None
    iso_pos = {p: joblib.load(model_dir / f"start_classifier_{p}_iso.joblib")
               for p in ["GK","DEF","MID","FWD"] if (model_dir / f"start_classifier_{p}_iso.joblib").exists()}

    p60_boost = load_booster(model_dir / "p60_direct.txt")
    p60_iso = joblib.load(model_dir / "p60_direct_calib.joblib") if (model_dir / "p60_direct_calib.joblib").exists() else None

    cameo_pos_boost = {p: load_booster(model_dir / f"cameo_given_bench_{p}.txt")
                       for p in ["GK","DEF","MID","FWD"] if (model_dir / f"cameo_given_bench_{p}.txt").exists()}
    cameo_glob_boost = load_booster(model_dir / "cameo_given_bench_global.txt") if (model_dir / "cameo_given_bench_global.txt").exists() else None
    cameo_iso_pos = {p: joblib.load(model_dir / f"cameo_given_bench_{p}_iso.joblib")
                     for p in ["GK","DEF","MID","FWD"] if (model_dir / f"cameo_given_bench_{p}_iso.joblib").exists() }
    cameo_iso_glob = joblib.load(model_dir / "cameo_given_bench_global_iso.joblib") if (model_dir / "cameo_given_bench_global_iso.joblib").exists() else None

    # Feature sets
    feat_start = ["start_lag1","start_rate_hl3","min_lag1","min_ewm_hl2",
                  "played_last","days_feat","long_gap14","start_streak","bench_streak",
                  "pos_enc","team_rot3","fdr"]
    feat_reg = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                "start_lag1","start_rate_hl3","pos_enc","team_rot3","fdr"]
    feat_p60 = feat_start
    feat_cameo = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                  "start_rate_hl3","bench_streak","pos_enc","team_rot3","fdr"]
    feat_cameo_min = ["min_lag1","min_ewm_hl2","played_last","days_feat","long_gap14",
                      "start_rate_hl3","bench_streak","pos_enc","team_rot3","fdr"]

    # Gates (v1)
    def predict_p_start(df_in: pd.DataFrame) -> np.ndarray:
        X = df_in[feat_start].fillna(0)
        n = len(df_in); p_pos = np.full(n, np.nan)
        for pos in ["GK","DEF","MID","FWD"]:
            m = start_pos.get(pos)
            if m is None: continue
            rows = (df_in["pos"].values == pos)
            if not rows.any(): continue
            pv = predict_booster_prob(m, X.loc[rows])
            iso = iso_pos.get(pos)
            if iso is not None: pv = np.clip(iso.transform(pv), 0, 1)
            p_pos[rows] = pv
        p_glob = predict_booster_prob(start_glob, X)
        if iso_glob is not None: p_glob = np.clip(iso_glob.transform(p_glob), 0, 1)
        blend = float(np.clip(args.gate_blend, 0, 1))
        p = np.where(~np.isnan(p_pos), blend*p_pos + (1-blend)*p_glob, p_glob)
        return np.clip(p, 0, 1)

    def predict_p60(df_in: pd.DataFrame) -> np.ndarray:
        p = predict_booster_prob(p60_boost, df_in[feat_p60].fillna(0))
        if p60_iso is not None: p = np.clip(p60_iso.transform(p), 0, 1)
        return p

    def predict_p_cameo(df_in: pd.DataFrame) -> np.ndarray:
        X = df_in[feat_cameo].fillna(0)
        n = len(df_in); p = np.zeros(n, dtype=float); used = np.zeros(n, dtype=bool)
        for pos in ["GK","DEF","MID","FWD"]:
            m = cameo_pos_boost.get(pos)
            if m is None: continue
            rows = (df_in["pos"].values == pos)
            if not rows.any(): continue
            pv = predict_booster_prob(m, X.loc[rows])
            iso = cameo_iso_pos.get(pos)
            if iso is not None: pv = np.clip(iso.transform(pv), 0, 1)
            p[rows] = pv; used[rows] = True
        if (~used).any() and cameo_glob_boost is not None:
            pv = predict_booster_prob(cameo_glob_boost, X.loc[~used])
            if cameo_iso_glob is not None: pv = np.clip(cameo_iso_glob.transform(pv), 0, 1)
            p[~used] = pv
        return np.clip(p, 0, 1)

    # Refit minute regressors on leakage-safe history
    hist = df.loc[pre_asof_mask].copy()
    reg_start, _ = train_regressor(hist[hist["is_starter"] == 1], feat_reg, objective="regression_l2")
    cameo_min_by_pos, _, cameo_min_global = train_cameo_minutes_by_pos(hist, feat_cameo_min, min_rows=150)

    def predict_cameo_minutes(df_in: pd.DataFrame) -> np.ndarray:
        Xc = df_in[feat_cameo_min].fillna(0)
        mu = np.zeros(len(df_in), dtype=float); used = np.zeros(len(df_in), dtype=bool)
        for pos in ["GK","DEF","MID","FWD"]:
            model = cameo_min_by_pos.get(pos)
            if model is None: continue
            rows = (df_in["pos"].values == pos)
            if not rows.any(): continue
            mu[rows] = predict_with_model(model, Xc.loc[rows], default_val=15.0, clip_hi=None)
            used[rows] = True
        if (~used).any() and cameo_min_global is not None:
            mu[~used] = predict_with_model(cameo_min_global, Xc.loc[~used], default_val=15.0, clip_hi=None)
        return np.clip(mu, 0, 60)

    # Score future rows
    p_start = predict_p_start(df_pred)
    p60 = predict_p60(df_pred)
    p_cameo = predict_p_cameo(df_pred)
    mu_cameo = predict_cameo_minutes(df_pred)

    # STARTER MINUTES: keep your original taper (single policy from args)
    pred_start = predict_with_model(reg_start, df_pred[feat_reg], default_val=60.0, clip_hi=120)
    if args.use_taper:
        try:
            pred_start = taper_start_minutes(pred_start, p_start, df_pred["pos"],
                                             args.taper_lo, args.taper_hi, args.taper_min_scale)
        except TypeError:
            pred_start = taper_start_minutes(pred_start, p_start, df_pred["pos"])

    # BENCH: player-specific caps (recency-weighted 95th) shrunk to position cap
    if args.use_pos_bench_caps and not cameo_hist.empty:
        player_caps = _build_player_bench_caps(
            cameo_hist=cameo_hist.assign(),
            pos_caps=bench_caps,
            halflife_matches=8.0,
            shrink_k=6.0,
            winsor_to_pos=True
        )
        cap_vec = df_pred["player_id"].astype(str).map(player_caps).to_numpy()
        # Fallback to position cap if a player has no history
        pos_cap_vec = per_position_bench_cap_from_train(df_pred["pos"], bench_caps)
        cap_vec = np.where(np.isfinite(cap_vec), cap_vec, pos_cap_vec)
    else:
        # Position-only cap or flat cap
        if args.use_pos_bench_caps:
            cap_vec = per_position_bench_cap_from_train(df_pred["pos"], bench_caps)
        else:
            cap_vec = np.full(len(df_pred), float(args.bench_cap), dtype=float)

    pred_bench_cameo = np.minimum(mu_cameo, cap_vec)
    pred_bench = np.clip(p_cameo * pred_bench_cameo, 0.0, cap_vec)

    # thresholds
    tlo = np.full(len(df_pred), args.t_lo, dtype=float)
    thi = np.full(len(df_pred), args.t_hi, dtype=float)
    for pos, (a, b) in parse_pos_thresholds(args.pos_thresholds).items():
        rows = (df_pred["pos"].values == pos)
        if rows.any(): tlo[rows], thi[rows] = a, b

    mix = p_start * pred_start + (1.0 - p_start) * pred_bench
    is_gk = (df_pred["pos"].values == "GK")
    minutes_pred = np.empty(len(df_pred), dtype=float)
    if args.no_mix_gk and is_gk.any():
        minutes_pred[is_gk] = np.where(
            p_start[is_gk] >= thi[is_gk], pred_start[is_gk],
            np.where(p_start[is_gk] <= tlo[is_gk], pred_bench[is_gk], pred_start[is_gk])
        )
    if (~is_gk).any():
        idx = ~is_gk
        minutes_pred[idx] = np.where(
            p_start[idx] >= thi[idx], pred_start[idx],
            np.where(p_start[idx] <= tlo[idx], pred_bench[idx], mix[idx])
        )
    minutes_pred = np.clip(minutes_pred, 0, 120)

    p_play = np.clip(p_start + (1.0 - p_start) * p_cameo, 0, 1)
    exp_minutes_points = np.clip(p_play + p60, 0, 2)

    # -------- Build output (legacy metadata + predictions) --------
    # Legacy metadata first
    out_cols = {
        "season": df_pred["season"].values,
        "game_id": df_pred.get("fbref_id", pd.Series([np.nan]*len(df_pred))).values,
        "team_id": df_pred.get("team_id", np.nan).values,
        "team": df_pred.get("team", pd.Series([np.nan]*len(df_pred))).values,
        "opponent_id": df_pred.get("opponent_id", pd.Series([np.nan]*len(df_pred))).values,
        "opponent": df_pred.get("opponent", pd.Series([np.nan]*len(df_pred))).values,
        # gw columns (if present)
    }
    if "gw_played" in df_pred.columns: out_cols["gw_played"] = df_pred["gw_played"].values
    if "gw_orig"   in df_pred.columns: out_cols["gw_orig"]   = df_pred["gw_orig"].values
    if "gw"        in df_pred.columns: out_cols["gw"]        = df_pred["gw"].values
    # fdr as integer in legacy metadata block
    if args.use_fdr:
        out_cols["fdr"] = df_pred["fdr"].astype(int).values

    # Row identity and features
    out_cols.update({
        "player_id": df_pred["player_id"].values,
        "player": df_pred.get("player", np.nan).values,
        "pos": df_pred["pos"].values,
        "date_sched": df_pred.get("date_sched", pd.NaT).values,
        "p_start": p_start, "p60": p60, "p_cameo": p_cameo, "p_play": p_play,
        "pred_start_head": pred_start,
        "pred_bench_cameo_head": pred_bench_cameo,
        "pred_bench_head": pred_bench,
        "pred_minutes": minutes_pred,
        "exp_minutes_points": exp_minutes_points,
    })
    if "_is_synth" in df_pred.columns:
        out_cols["_is_synth"] = df_pred["_is_synth"].fillna(0).astype(int).values

    out = pd.DataFrame(out_cols)

    # Optional: sort for readability
    sort_keys = [k for k in ["gw_played","gw_orig","gw","team_id","player_id"] if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys).reset_index(drop=True)

    # ---------- auto-name output + dual writer ----------
    gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
    out_paths = _out_paths(
        base_dir=out_dir,
        season=args.future_season,
        gw_from=gw_from_eff,
        gw_to=gw_to_eff,
        zero_pad=args.zero_pad_filenames,
        out_format=args.out_format
    )
    written_paths = _write_minutes(out, out_paths)

    print(json.dumps({
        "rows": int(len(out)),
        "season": args.future_season,
        "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
        "available_team_gws": [int(g) for g in avail_gws],
        "scored_gws": [int(g) for g in target_gws],
        "as_of": str(as_of_ts),
        "out": written_paths
    }, indent=2))

if __name__ == "__main__":
    main()
