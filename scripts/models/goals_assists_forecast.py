#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
r"""
goals_assists_forecast.py — leak-free scorer for future GWs using trained G/A heads,
with roster gating, venue mapping from team fixtures, serve-time feature parity, and safer handling.

Key fixes vs prior rev:
• Robust fixtures join: coerce key dtypes (gw numeric, team_id string), ensure `is_home`,
  and compute `venue_bin` AFTER the merge to avoid KeyError paths.
• Serve-time feature parity: MERGE last-known per-player EWMs/rolls into `fut` before building X.
  (This removes the NaN-heavy LGBM vector that caused Salah underestimates and Thiago 5.99 a90 explosions.)
• Keep NaNs for LGBM at inference (no blanket fillna(0)); LightGBM routes missing values.
• Use training medians (artifacts/features_median.json) for Poisson/Tweedie imputations; avoid leakage.
• Optional safety clamp: --cap-per90 "GK:0.10,DEF:0.35,MID:1.00,FWD:1.40" (applies to LGBM per-90s only).
• Apply isotonic calibration for probabilities when --apply-calibration is set.
• Normalize league labels and write date-only for dates in all persisted files (CSV+Parquet).
• Output includes rich minutes metadata in a fixed order.
• Auto-resolve minutes file from GW window (+ dual CSV/Parquet loader), with helpful failures.
• NEW: Output format control (--out-format csv|parquet|both) + optional zero-padded filenames.
• NEW: Guaranteed date-only persistence (YYYY-MM-DD strings) — never writes time.

NEW (legacy meta like minutes_forecast):
• Attach game_id (fbref_id), team, opponent_id, opponent from fixture_calendar.csv.
• Venue-consistent, DGW-safe FDR attach; `fdr` is INT and part of legacy metadata.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.utils.validate import validate_df


# ───────────────────────────── Helpers ─────────────────────────────

def _load_json(p: Path) -> dict:
    if not p or not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _pick_gw_col(cols: List[str]) -> Optional[str]:
    for k in ("gw_played", "gw_orig", "gw"):
        if k in cols:
            return k
    return None


def _coerce_ts(s: pd.Series, tz: Optional[str]) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if tz:
        if out.dt.tz is None:
            out = out.dt.tz_localize(tz)
        else:
            out = out.dt.tz_convert(tz)
    return out


def _to_naive_date(s: pd.Series, tz: Optional[str]) -> pd.Series:
    """
    Ensure date-only, tz-naive (datetime64[ns]) in local calendar.
    If tz-aware, convert to `tz`, drop tz, then normalize.
    """
    out = pd.to_datetime(s, errors="coerce")
    if hasattr(out.dt, "tz") and out.dt.tz is not None:
        # align to target tz calendar day before dropping tz
        if tz:
            out = out.dt.tz_convert(tz)
        out = out.dt.tz_localize(None)
    return out.dt.normalize()



def _load_players_form(features_root: Path, form_version: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "players_form.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Missing players_form: {fp}")
        t = pd.read_csv(fp, parse_dates=["date_played"])
        t["date_played"] = pd.to_datetime(t["date_played"], errors="coerce")  # keep datetime for in-memory ops
        t["season"] = s
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    need = {"season", "gw_orig", "date_played", "player_id", "team_id", "player", "pos", "venue", "minutes"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"players_form missing required columns: {miss}")
    return df


def _load_team_form(features_root: Path, form_version: str, seasons: List[str]) -> Optional[pd.DataFrame]:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "team_form.csv"
        if fp.exists():
            t = pd.read_csv(fp, parse_dates=["date_played"])
            t["date_played"] = pd.to_datetime(t["date_played"], errors="coerce")
            t["season"] = s
            frames.append(t)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _team_z_venue(team_form: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return (season, gw_orig, team_id, team_att_z_venue, opp_def_z_venue)."""
    if team_form is None:
        return None

    if {"team_att_z_venue", "opp_def_z_venue"}.issubset(team_form.columns):
        t = team_form[["season", "gw_orig", "team_id", "team_att_z_venue", "opp_def_z_venue"]].drop_duplicates()
        for c in ["team_att_z_venue", "opp_def_z_venue"]:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        return t

    need = {
        "season", "gw_orig", "team_id", "venue",
        "att_xg_home_roll_z", "att_xg_away_roll_z",
        "def_xga_home_roll_z", "def_xga_away_roll_z"
    }
    if need.issubset(team_form.columns):
        t = team_form[list(need)].copy()
        v = t["venue"].astype(str).str.lower()
        t["team_att_z_venue"] = np.where(v.eq("home"), t["att_xg_home_roll_z"], t["att_xg_away_roll_z"])
        t["opp_def_z_venue"] = np.where(v.eq("home"), t["def_xga_away_roll_z"], t["def_xga_home_roll_z"])
        t = t.drop(columns=["venue", "att_xg_home_roll_z", "att_xg_away_roll_z", "def_xga_home_roll_z", "def_xga_away_roll_z"])
        t = t.drop_duplicates(subset=["season", "gw_orig", "team_id"])
        for c in ["team_att_z_venue", "opp_def_z_venue"]:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        return t

    return None


def _load_team_fixtures(fix_root: Path, season: str, filename: str) -> pd.DataFrame:
    path = fix_root / season / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing team fixtures: {path}")
    tf = pd.read_csv(path)
    # Parse → normalize (keep dtype datetime64[ns], zero time component)
    for dc in ("date_sched", "date_played"):
        if dc in tf.columns:
            tf[dc] = pd.to_datetime(tf[dc], errors="coerce").dt.normalize()
    if "is_home" not in tf.columns:
        if "was_home" in tf.columns:
            tf["is_home"] = tf["was_home"].astype(str).str.lower().isin(["1", "true", "yes"]).astype("Int8")
        elif "venue" in tf.columns:
            tf["is_home"] = tf["venue"].astype(str).str.lower().eq("home").astype("Int8")
        else:
            tf["is_home"] = 0
    for c in ("gw_played", "gw_orig", "gw"):
        if c in tf.columns:
            tf[c] = pd.to_numeric(tf[c], errors="coerce")
    if "team_id" not in tf.columns:
        for alt in ["team", "teamId", "team_code"]:
            if alt in tf.columns:
                tf = tf.rename(columns={alt: "team_id"})
                break
    if "team_id" in tf.columns:
        tf["team_id"] = tf["team_id"].astype(str)
    tf["season"] = season
    return tf


def _gw_for_selection(df: pd.DataFrame) -> pd.Series:
    def num(s): return pd.to_numeric(df.get(s), errors="coerce")
    gwp = num("gw_played"); gwo = num("gw_orig"); gwa = num("gw")
    return gwo.where(gwp.isna() | (gwp <= 0), gwp).where(lambda x: x.notna(), gwa)


def _ewm_shots_per_pos(hist: pd.DataFrame,
                       hl_map: Dict[str, float],
                       min_periods: int,
                       adjust: bool) -> pd.DataFrame:
    df = hist.copy()
    shots = next((c for c in ("shots", "sh") if c in df.columns), None)
    sot   = next((c for c in ("sot", "shots_on_target") if c in df.columns), None)

    m = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0).clip(lower=0)
    denom = (m / 90.0).replace(0, np.nan)

    if shots is not None:
        df["_shots_p90_raw"] = pd.to_numeric(df[shots], errors="coerce") / denom
    if sot is not None:
        df["_sot_p90_raw"] = pd.to_numeric(df[sot], errors="coerce") / denom

    df["pos"] = df["pos"].astype(str).str.upper()
    for col in ["shots_p90_ewm", "sot_p90_ewm", "shots_p90_home_ewm", "shots_p90_away_ewm",
                "sot_p90_home_ewm", "sot_p90_away_ewm"]:
        df[col] = np.nan

    def _ewm(series: pd.Series, hl: float) -> pd.Series:
        return series.shift(1).ewm(halflife=float(hl), min_periods=min_periods, adjust=adjust).mean()

    v = df.get("venue", pd.Series([""] * len(df))).astype(str).str.lower()
    mask_home = v.eq("home"); mask_away = v.eq("away")

    for tag, hl in hl_map.items():
        mask_pos = df["pos"].eq(tag)

        if "_shots_p90_raw" in df.columns:
            df.loc[mask_pos, "shots_p90_ewm"] = (
                df.loc[mask_pos].groupby(["player_id", "season"], sort=False)["_shots_p90_raw"]
                  .transform(lambda s: _ewm(s, hl))
            )
            if mask_home.any():
                sub = mask_pos & mask_home
                df.loc[sub, "shots_p90_home_ewm"] = (
                    df.loc[sub].groupby(["player_id", "season"], sort=False)["_shots_p90_raw"]
                      .transform(lambda s: _ewm(s, hl))
                )
            if mask_away.any():
                sub = mask_pos & mask_away
                df.loc[sub, "shots_p90_away_ewm"] = (
                    df.loc[sub].groupby(["player_id", "season"], sort=False)["_shots_p90_raw"]
                      .transform(lambda s: _ewm(s, hl))
                )

        if "_sot_p90_raw" in df.columns:
            df.loc[mask_pos, "sot_p90_ewm"] = (
                df.loc[mask_pos].groupby(["player_id", "season"], sort=False)["_sot_p90_raw"]
                  .transform(lambda s: _ewm(s, hl))
            )
            if mask_home.any():
                sub = mask_pos & mask_home
                df.loc[sub, "sot_p90_home_ewm"] = (
                    df.loc[sub].groupby(["player_id", "season"], sort=False)["_sot_p90_raw"]
                      .transform(lambda s: _ewm(s, hl))
                )
            if mask_away.any():
                sub = mask_pos & mask_away
                df.loc[sub, "sot_p90_away_ewm"] = (
                    df.loc[sub].groupby(["player_id", "season"], sort=False)["_sot_p90_raw"]
                      .transform(lambda s: _ewm(s, hl))
                )

    drop = [c for c in ["_shots_p90_raw", "_sot_p90_raw"] if c in df.columns]
    return df.drop(columns=drop)


def _last_snapshot_per_player(df: pd.DataFrame, feature_cols: List[str],
                              as_of_ts: pd.Timestamp, tz: Optional[str]) -> pd.DataFrame:
    du = pd.to_datetime(df["date_played"], errors="coerce")
    if tz:
        if du.dt.tz is None:
            du = du.dt.tz_localize(tz)
        else:
            du = du.dt.tz_convert(tz)
    hist = df[du < as_of_ts].copy()
    if hist.empty:
        return pd.DataFrame(columns=["season", "player_id"] + feature_cols)

    gw_key = _pick_gw_col(hist.columns.tolist())
    sort_cols = ["player_id", "season", "date_played"] + ([gw_key] if gw_key else [])
    hist = hist.sort_values(sort_cols)
    last = hist.groupby(["season", "player_id"], as_index=False).tail(1).copy()

    keep = ["season", "player_id"] + [c for c in feature_cols if c in last.columns]
    for c in feature_cols:
        if c not in keep:
            last[c] = np.nan
    return last[["season", "player_id"] + feature_cols].copy()


def _load_booster(p: Path) -> Optional[lgb.Booster]:
    if not p.exists():
        return None
    return lgb.Booster(model_file=str(p))


def _predict_reg(booster: Optional[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    if booster is None or X.empty:
        return np.zeros(len(X), dtype=float)
    Xn = X.astype(float)
    return np.clip(booster.predict(Xn), 0, None)


def _predict_per_pos(goals_or_assists: str,
                     X: pd.DataFrame,
                     pos: pd.Series,
                     model_dir: Path) -> np.ndarray:
    glob = _load_booster(model_dir / f"{goals_or_assists}_global_lgbm.txt")
    out = np.zeros(len(X), dtype=float)
    used = np.zeros(len(X), dtype=bool)
    for tag in ["GK", "DEF", "MID", "FWD"]:
        m = _load_booster(model_dir / f"{goals_or_assists}_{tag}_lgbm.txt")
        idx = pos.str.upper().eq(tag).to_numpy()
        if idx.any() and m is not None:
            out[idx] = _predict_reg(m, X.iloc[idx])
            used[idx] = True
    if (~used).any() and glob is not None:
        out[~used] = _predict_reg(glob, X.iloc[~used])
    return out


def _read_features_median(model_dir: Path) -> Optional[pd.Series]:
    p = model_dir / "artifacts" / "features_median.json"
    if p.exists():
        try:
            dct = _load_json(p)
            if dct:
                return pd.Series(dct, dtype=float)
        except Exception:
            pass
    return None


def _predict_poisson_per_pos(name: str,
                             X: pd.DataFrame,
                             pos: pd.Series,
                             model_dir: Path,
                             med: Optional[pd.Series]) -> np.ndarray:
    def _load(path: Path):
        try:
            return joblib.load(path) if path.exists() else None
        except Exception:
            return None

    glob = _load(model_dir / f"{name}_global_poisson.joblib")
    out = np.full(len(X), np.nan, dtype=float)
    used = np.zeros(len(X), dtype=bool)

    Xp = X.apply(pd.to_numeric, errors="coerce")
    if med is not None:
        Xp = Xp.fillna(med.reindex(Xp.columns))
    Xp = Xp.fillna(0.0)

    for tag in ["GK", "DEF", "MID", "FWD"]:
        mdl = _load(model_dir / f"{name}_{tag}_poisson.joblib")
        idx = pos.str.upper().eq(tag).to_numpy()
        if mdl is not None and idx.any():
            out[idx] = np.clip(mdl.predict(Xp.iloc[idx].to_numpy(dtype=float)), 0, None)
            used[idx] = True
    if (~used).any() and glob is not None:
        out[~used] = np.clip(glob.predict(Xp.iloc[~used].to_numpy(dtype=float)), 0, None)
    return out


# ───────────────────────────── Minutes resolver & dual loader ─────────────────────────────

def _fmt_gw(n: int, zero_pad: bool) -> str:
    return f"{int(n):02d}" if zero_pad else f"{int(n)}"


def _candidate_minutes_paths(minutes_root: Path, future_season: str, gw_from: int, gw_to: int) -> List[Path]:
    season_dir = minutes_root / str(future_season)
    cands: List[Path] = []
    for zp in (False, True):
        a = _fmt_gw(gw_from, zp)
        b = _fmt_gw(gw_to, zp)
        cands.append(season_dir / f"GW{a}_{b}.csv")
        cands.append(season_dir / f"GW{a}_{b}.parquet")
    return cands


def _glob_fallback(minutes_root: Path, future_season: str, gw_from: int, gw_to: int) -> Optional[Path]:
    season_dir = minutes_root / str(future_season)
    if not season_dir.exists():
        return None
    patterns = [f"GW{gw_from}_*.csv", f"GW{gw_from}_*.parquet",
                f"GW{gw_from:02d}_*.csv", f"GW{gw_from:02d}_*.parquet"]
    for pat in patterns:
        for p in sorted(season_dir.glob(pat)):
            try:
                stem = p.stem
                to_str = stem.split("_")[-1].replace("GW", "")
                to_val = int(to_str)
                if to_val == int(gw_to):
                    return p
            except Exception:
                continue
    return None


def _resolve_minutes_path(args: argparse.Namespace, gw_from: int, gw_to: int) -> Path:
    if args.minutes_csv:
        p = Path(args.minutes_csv)
        if not p.exists():
            raise FileNotFoundError(f"--minutes-csv not found: {p}")
        return p

    for cand in _candidate_minutes_paths(args.minutes_root, args.future_season, gw_from, gw_to):
        if cand.exists():
            return cand

    fb = _glob_fallback(args.minutes_root, args.future_season, gw_from, gw_to)
    if fb:
        return fb

    season_dir = args.minutes_root / str(args.future_season)
    msg = [
        f"Minutes file not found for GW window {gw_from}-{gw_to}.",
        f"Looked under: {season_dir}",
        "Tried candidates:",
    ]
    for c in _candidate_minutes_paths(args.minutes_root, args.future_season, gw_from, gw_to):
        msg.append(f"  - {c}")
    msg += [
        "Also tried glob fallback: GW{from}_*.{csv,parquet} (with and without zero-padding).",
        "Fix by either:",
        "  • generating that file, or",
        "  • pointing directly via --minutes-csv <path>, or",
        "  • adjusting --minutes-root / GW window."
    ]
    raise FileNotFoundError("\n".join(msg))


def _load_minutes_dual(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        hdr = pd.read_csv(path, nrows=0)
        parse_cols = ["date_sched"] if "date_sched" in hdr.columns else None
        df = pd.read_csv(path, parse_dates=parse_cols)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        if "date_sched" in df.columns:
            df["date_sched"] = pd.to_datetime(df["date_sched"], errors="coerce")
    else:
        raise ValueError(f"Unsupported minutes file extension: {suffix}. Use .csv or .parquet")

    # enforce date-only (datetime64[ns] at midnight) for safety in-memory
    if "date_sched" in df.columns:
        df["date_sched"] = pd.to_datetime(df["date_sched"], errors="coerce").dt.normalize()
    if "date_played" in df.columns:
        df["date_played"] = pd.to_datetime(df["date_played"], errors="coerce").dt.normalize()

    return df


# ───────────────────────────── Roster gating ─────────────────────────────

def _norm_label(s: str) -> str:
    return str(s or "").lower().replace("-", " ").replace("_", " ").strip()


def _load_roster_pairs(teams_json: Optional[Path],
                       season: str,
                       league_filter: Optional[str]) -> Optional[Set[Tuple[str, str]]]:
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

    lf = _norm_label(league_filter) if league_filter else ""
    allowed: Set[Tuple[str, str]] = set()

    for team_id, obj in (data or {}).items():
        season_info = (obj or {}).get("career", {}).get(season)
        if not season_info:
            continue
        if lf and _norm_label(season_info.get("league", "")) != lf:
            continue
        players = season_info.get("players", []) or []
        for pl in players:
            pid = str(pl.get("id", "")).strip()
            if pid:
                allowed.add((str(team_id), pid))

    if not allowed:
        logging.warning("Roster map for %s produced 0 allowed pairs (league=%r).", season, league_filter)
    return allowed or None


def _apply_roster_gate(df: pd.DataFrame,
                       allowed_pairs: Optional[Set[Tuple[str, str]]],
                       season: str,
                       where: str,
                       out_artifacts_dir: Optional[Path] = None,
                       require_on_roster: bool = False) -> pd.DataFrame:
    if allowed_pairs is None or df.empty:
        return df

    tid = df.get("team_id").astype(str)
    pid = df.get("player_id").astype(str)
    pairs = list(zip(tid.to_numpy(), pid.to_numpy()))
    mask_ok = np.fromiter(((a, b) in allowed_pairs for (a, b) in pairs), count=len(pairs), dtype=bool)

    dropped = int((~mask_ok).sum())
    if dropped:
        logging.info("Roster gate dropped %d %s row(s) not present on the %s roster.", dropped, where, season)
        if out_artifacts_dir is not None:
            out_artifacts_dir.mkdir(parents=True, exist_ok=True)
            df.loc[~mask_ok].to_csv(out_artifacts_dir / f"roster_dropped_{where}.csv", index=False)
        if require_on_roster:
            raise RuntimeError(f"--require-on-roster set: {dropped} {where} rows are not on the {season} roster.")
    return df.loc[mask_ok].copy()


# ───────────────────────────── GA output helpers ─────────────────────────────

def _ga_out_paths(base_dir: Path, season: str, gw_from: int, gw_to: int,
                  zero_pad: bool, out_format: str) -> List[Path]:
    a = _fmt_gw(gw_from, zero_pad); b = _fmt_gw(gw_to, zero_pad)
    season_dir = base_dir / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    stem = season_dir / f"GW{a}_{b}"
    if out_format == "csv":
        return [Path(str(stem) + ".csv")]
    if out_format == "parquet":
        return [Path(str(stem) + ".parquet")]
    return [Path(str(stem) + ".csv"), Path(str(stem) + ".parquet")]


def _write_ga(df: pd.DataFrame, paths: List[Path]) -> List[str]:
    """
    Persist outputs guaranteeing date-only strings (YYYY-MM-DD) for all date columns,
    for BOTH CSV and Parquet. In-memory df remains untouched (datetime64[ns] ok).
    """
    written: List[str] = []
    tmp = df.copy()

    # Identify date-like cols to stringify; GA output uses date_sched, but handle any present.
    date_cols = [c for c in ["date_sched", "date_played"] if c in tmp.columns]
    for dc in date_cols:
        tmp[dc] = pd.to_datetime(tmp[dc], errors="coerce").dt.strftime("%Y-%m-%d")

    for p in paths:
        if p.suffix.lower() == ".csv":
            tmp.to_csv(p, index=False)
            written.append(str(p))
        elif p.suffix.lower() == ".parquet":
            # Store as strings in Parquet to avoid implicit time components
            tmp.to_parquet(p, index=False)
            written.append(str(p))
        else:
            raise ValueError(f"Unsupported output extension: {p.suffix}")
    return written


# ===================== CONSOLIDATED WRITER (season-level) =====================
def _read_any(path: Path) -> pd.DataFrame:
    """
    Read CSV/Parquet without forcing date parsing — keep strings as-is.
    Older files may contain datetimes; downstream code will handle both.
    """
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file to read: {path}")


def _update_consolidated_ga(
    out_df: pd.DataFrame,
    season_dir: Path,
    out_format: str,
    desired_order: List[str],
    run_gw_from: int,
    run_gw_to: int,
) -> List[str]:
    """
    Maintain season-level consolidated files:
      <season_dir>/expected_goals_assists.csv and/or .parquet (per --out-format).
    DGW-safe: de-duplicate on a robust identity.
    Persist dates as 'YYYY-MM-DD' strings (never time).
    """
    cons_stem = season_dir / "expected_goals_assists"
    targets: List[Path] = []
    if out_format in ("csv", "both"):    targets.append(Path(str(cons_stem) + ".csv"))
    if out_format in ("parquet", "both"): targets.append(Path(str(cons_stem) + ".parquet"))

    # annotate this batch with run window (persisted for precedence on future merges)
    out_df = out_df.copy()
    out_df["_run_gw_from"] = int(run_gw_from)
    out_df["_run_gw_to"] = int(run_gw_to)

    # load existing (union csv/parquet if both exist)
    old = pd.DataFrame()
    for p in [Path(str(cons_stem) + ".csv"), Path(str(cons_stem) + ".parquet")]:
        if p.exists():
            dfp = _read_any(p)
            old = dfp if old.empty else pd.concat([old, dfp], ignore_index=True)

    # normalize key dtypes
    for c in ("season","team_id","player_id","opponent_id","game_id"):
        if c in out_df.columns: out_df[c] = out_df[c].astype(str)
        if c in old.columns:    old[c]    = old[c].astype(str)

    # ensure batch meta columns exist on old (if file pre-dates this change)
    if "_run_gw_from" not in old.columns:
        old["_run_gw_from"] = -1
    if "_run_gw_to" not in old.columns:
        old["_run_gw_to"] = -1

    all_cols = list(dict.fromkeys([*desired_order, *old.columns.tolist(), *out_df.columns.tolist()]))
    new = out_df.reindex(columns=all_cols)
    old = old.reindex(columns=all_cols)
    merged = pd.concat([old, new], ignore_index=True)

    # Build a temporary datetime for robust dedup/sort, but DO NOT persist it
    if "date_sched" in merged.columns:
        merged["_date_sched_dt"] = pd.to_datetime(merged["date_sched"], errors="coerce")
    else:
        merged["_date_sched_dt"] = pd.NaT

    # De-dup rule (points-forecast style):
    # for the same (season, player_id, gw_orig), keep the row from the batch
    # with the highest _run_gw_to (latest window). Tie-breaker: later date_sched.
    dedup_key = [c for c in ["season", "player_id", "gw_orig"] if c in merged.columns]
    if not dedup_key:
        # fall back if columns are missing in unexpected data
        dedup_key = [c for c in ["season", "player_id", "team_id", "gw_orig"] if c in merged.columns]

    merged = merged.sort_values(
        dedup_key
        + ["_run_gw_to"] 
        + (["_date_sched_dt"] if "_date_sched_dt" in merged.columns else []),
        kind="mergesort"
    ).drop_duplicates(subset=dedup_key, keep="last")

    
    # Keep desired output order, but also persist batch columns for future precedence
    keep_cols = [c for c in desired_order if c in merged.columns] + ["_run_gw_from","_run_gw_to","_date_sched_dt"]
    merged = merged[keep_cols].copy()

    sort_keys = [k for k in ["gw_orig","team_id","player_id"] if k in merged.columns]
    if sort_keys:
        merged = merged.sort_values(sort_keys + ["_date_sched_dt"], kind="mergesort").reset_index(drop=True)

    # Before writing, stringify dates (no time) and drop temp
    if "date_sched" in merged.columns:
        merged["date_sched"] = pd.to_datetime(merged["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = merged.drop(columns=["_date_sched_dt"], errors="ignore")

    written: List[str] = []
    for tgt in targets:
        if tgt.suffix.lower() == ".csv":
            merged.to_csv(tgt, index=False)
        else:
            merged.to_parquet(tgt, index=False)
        written.append(str(tgt))
    return written


# ───────────────────────────── Legacy metadata attach (NEW) ─────────────────────────────

def _attach_legacy_meta(df_pred: pd.DataFrame, team_fix: pd.DataFrame) -> pd.DataFrame:
    """
    Attach legacy metadata to df_pred from team fixtures:
      - game_id (from fbref_id), team, opponent_id, opponent (home/away derived)
    Join keys: (season, team_id, gw_key), preferring gw_played>0 → gw_orig>0 → gw>0.
    """
    if df_pred.empty or team_fix.empty:
        for col in ["fbref_id", "team", "opponent_id", "opponent"]:
            if col not in df_pred.columns:
                df_pred[col] = np.nan
        return df_pred

    df = df_pred.copy()
    gw_df = _pick_gw_col(df.columns.tolist()) or "gw_orig"
    gw_tf = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"

    if gw_df in df.columns:
        df[gw_df] = pd.to_numeric(df[gw_df], errors="coerce")
    tf = team_fix.copy()
    if gw_tf in tf.columns:
        tf[gw_tf] = pd.to_numeric(tf[gw_tf], errors="coerce")

    if "team_id" in df.columns:
        df["team_id"] = df["team_id"].astype(str)
    if "team_id" in tf.columns:
        tf["team_id"] = tf["team_id"].astype(str)

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

    if "opponent" not in merged.columns or merged["opponent"].isna().all():
        if {"home", "away", "is_home"}.issubset(merged.columns):
            ih = pd.to_numeric(merged["is_home"], errors="coerce").fillna(0).astype("Int8")
            merged["opponent"] = np.where(ih == 1, merged.get("away"), merged.get("home"))

    for col in ["fbref_id", "team", "opponent_id", "opponent"]:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged


# ───────────────────────────── Venue-consistent, DGW-safe FDR attach ─────────────────────────────

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
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype("Int8")
        return out
    out = df.copy()
    if "was_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["was_home"], errors="coerce").fillna(0).astype("Int8")
        return out
    if "venue" in out.columns:
        out["is_home"] = out["venue"].astype(str).str.lower().eq("home").astype("Int8")
        return out
    raise RuntimeError("No venue columns found to compute is_home.")


def attach_fdr_consistent(df: pd.DataFrame,
                          seasons_all: List[str],
                          features_root: Path,
                          version: str,
                          team_form: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Attach integer FDR using (season, team_id, GW, is_home). DGW-safe: collapse duplicates by max.
    """
    if df.empty:
        return df

    df = _ensure_is_home(df)
    df["team_id"] = df["team_id"].astype(str)
    for c in ("gw_played", "gw_orig", "gw"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    gw_df = _pick_gw_col(df.columns.tolist())
    if gw_df is None:
        raise RuntimeError("attach_fdr_consistent: no GW column among ['gw_played','gw_orig','gw'].")

    if team_form is None:
        tf_all = _load_team_form(features_root, version, seasons_all)
    else:
        tf_all = team_form.copy()

    if tf_all is None or tf_all.empty:
        raise FileNotFoundError("attach_fdr_consistent: team_form not available to attach FDR.")

    tf_all["team_id"] = tf_all.get("team_id", pd.Series(index=tf_all.index, dtype=object)).astype(str)
    for c in ("gw_played", "gw_orig", "gw"):
        if c in tf_all.columns:
            tf_all[c] = pd.to_numeric(tf_all[c], errors="coerce")

    home_col, away_col = _find_fdr_cols(set(tf_all.columns))
    gw_tf = _pick_gw_col(tf_all.columns.tolist())
    if gw_tf is None:
        raise RuntimeError("attach_fdr_consistent: no GW column in team_form.")

    base = tf_all[["season", "team_id", gw_tf, home_col, away_col]].dropna(subset=["team_id", gw_tf])
    home_rows = base.rename(columns={home_col: "fdr_side"}).assign(is_home=1)[["season", "team_id", gw_tf, "is_home", "fdr_side"]]
    away_rows = base.rename(columns={away_col: "fdr_side"}).assign(is_home=0)[["season", "team_id", gw_tf, "is_home", "fdr_side"]]
    form_long = pd.concat([home_rows, away_rows], ignore_index=True)

    if gw_tf != gw_df:
        form_long = form_long.rename(columns={gw_tf: gw_df})

    # Collapse DGW duplicates conservatively
    form_long = (form_long
                 .groupby(["season", "team_id", gw_df, "is_home"], as_index=False)["fdr_side"]
                 .max())

    merged = df.merge(
        form_long,
        how="left",
        on=["season", "team_id", gw_df, "is_home"],
        validate="many_to_one",
        copy=False
    )

    if merged["fdr_side"].isna().any():
        miss = merged.loc[merged["fdr_side"].isna(), ["season", "team_id", gw_df, "is_home"]].drop_duplicates()
        logging.error("attach_fdr_consistent: missing FDR for %d rows. Examples:\n%s",
                      len(miss), miss.head(20).to_string(index=False))
        raise RuntimeError("attach_fdr_consistent: FDR merge produced NaNs. Check keys/coverage.")

    merged["fdr"] = pd.to_numeric(merged["fdr_side"], errors="raise").astype("Int8")
    merged.drop(columns=["fdr_side"], inplace=True, errors="ignore")
    return merged


# ───────────────────────────── Main ─────────────────────────────

def _parse_cap_per90(s: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in str(s).split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            k = k.strip().upper()
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    # Windows-friendly args
    ap.add_argument("--history-seasons", required=True, help="Comma list of past seasons")
    ap.add_argument("--future-season", required=True, help="Season to score (contains future GWs)")
    ap.add_argument("--as-of", default="now", help='ISO timestamp or "now"')
    ap.add_argument("--as-of-tz", default="Africa/Lagos")
    ap.add_argument("--as-of-gw", type=int, required=True, help="First GW not yet played at --as-of (next GW)")
    ap.add_argument("--n-future", type=int, default=3)
    ap.add_argument("--gw-from", type=int, default=None)
    ap.add_argument("--gw-to", type=int, default=None)
    ap.add_argument("--strict-n-future", action="store_true")

    # IO
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"))
    ap.add_argument("--form-version", required=True)
    ap.add_argument("--fix-root", type=Path, default=Path("data/processed/registry/fixtures"))
    ap.add_argument("--team-fixtures-filename", default="fixture_calendar.csv")

    # Minutes input
    ap.add_argument("--minutes-csv", type=Path, help="Explicit minutes file (CSV or Parquet). Overrides auto-resolution.")
    ap.add_argument("--minutes-root", type=Path, default=Path("data/predictions/minutes"),
                    help="Root containing <season>/GW<from>_<to>.csv|parquet")

    # GA output
    ap.add_argument("--out-dir", type=Path, default=Path("data/predictions/goals_assists"))
    ap.add_argument("--out-format", choices=["csv", "parquet", "both"], default="csv",
                    help="Output format for G/A predictions (default: csv)")
    ap.add_argument("--zero-pad-filenames", action="store_true",
                    help="Write filenames as GW05_07 instead of GW5_7")

    ap.add_argument("--model-dir", type=Path, required=True, help="Folder with trained G/A artifacts (a specific version)")
    ap.add_argument("--apply-calibration", action="store_true")
    ap.add_argument("--skip-gk", action="store_true")
    ap.add_argument("--cap-per90", type=str, default="", help='Optional per-POS cap, e.g. "GK:0.10,DEF:0.35,MID:1.00,FWD:1.40"')
    ap.add_argument("--log-level", default="INFO")

    # Roster gating
    ap.add_argument("--teams-json", type=Path, help="Path to master_teams.json containing per-season rosters")
    ap.add_argument("--league-filter", type=str, default="", help="Optional league name (e.g., 'ENG-Premier League')")
    ap.add_argument("--require-on-roster", action="store_true",
                    help="If set, error out when any scoring rows are not on the future-season roster")

    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    # Seasons & GW window
    history = [s.strip() for s in args.history_seasons.split(",") if s.strip()]
    seasons_all = history + [args.future_season]
    gw_from_req = args.gw_from if args.gw_from is not None else args.as_of_gw
    gw_to_req = args.gw_to if args.gw_to is not None else (gw_from_req + max(1, args.n_future) - 1)

    # Pre-compute season dirs (for artifacts)
    season_dir = args.out_dir / f"{args.future_season}"
    artifacts_dir = season_dir / "artifacts"

    # as-of
    as_of_ts = (pd.Timestamp.now(tz=args.as_of_tz)
                if str(args.as_of).lower() in ("now", "auto", "today")
                else pd.Timestamp(args.as_of, tz=args.as_of_tz))

    # --- Load artifacts (features & meta) ---
    feat_path = args.model_dir / "artifacts" / "features.json"
    feat_cols: List[str] = _load_json(feat_path) or []
    if not feat_cols:
        raise FileNotFoundError(f"features.json not found or empty: {feat_path}")

    meta = _load_json(args.model_dir / "meta.json")
    train_args = meta.get("args", {})
    hl_default = float(train_args.get("ewm_halflife", 3.0))
    hl_pos_map = {"GK": hl_default, "DEF": hl_default, "MID": hl_default, "FWD": hl_default}
    key = "ewm_halflife_pos"
    if isinstance(train_args.get(key), str) and ":" in train_args[key]:
        try:
            parts = dict(p.split(":") for p in train_args[key].split(","))
            for k in ["GK", "DEF", "MID", "FWD"]:
                if k in parts:
                    hl_pos_map[k] = float(parts[k])
        except Exception:
            pass
    ewm_min_periods = int(train_args.get("ewm_min_periods", 1))
    ewm_adjust = bool(train_args.get("ewm_adjust", False))

    med_train = _read_features_median(args.model_dir)

    # --- Load registry features up to --as-of ---
    pf = _load_players_form(args.features_root, args.form_version, seasons_all)
    tf = _load_team_form(args.features_root, args.form_version, seasons_all)
    tz = args.as_of_tz

    du = _coerce_ts(pf["date_played"], tz)
    pf_hist = pf[(pf["season"].isin(history)) |
                 ((pf["season"] == args.future_season) & (du < as_of_ts))].copy()
    pf_hist = _ewm_shots_per_pos(pf_hist, hl_map=hl_pos_map, min_periods=ewm_min_periods, adjust=ewm_adjust)

    # --- Minutes forecast for target GWs (auto-resolve + dual loader) ---
    minutes_path = _resolve_minutes_path(args, gw_from_req, gw_to_req)
    minutes = _load_minutes_dual(minutes_path)
    if "season" not in minutes.columns:
        minutes["season"] = args.future_season

    gw_sel = _gw_for_selection(minutes)
    avail_gws = sorted(pd.unique(gw_sel.dropna().astype("Int8")))
    avail_gws = [int(x) for x in avail_gws]
    target_gws = [int(g) for g in avail_gws if g >= int(gw_from_req)][:int(args.n_future)]
    if not target_gws:
        raise RuntimeError(f"No target GWs >= {gw_from_req} in minutes file ({minutes_path}). Available: {avail_gws}")
    if args.strict_n_future and len(target_gws) < args.n_future:
        raise RuntimeError(f"Only {len(target_gws)} GW(s) available; wanted {args.n_future}. Available: {avail_gws}")

    minutes = minutes[gw_sel.isin(target_gws)].copy()
    if minutes.empty:
        raise RuntimeError("No minute rows after filtering target GWs.")

    # Merge team fixtures (for venue + legacy metadata)
    team_fix = _load_team_fixtures(args.fix_root, args.future_season, args.team_fixtures_filename)

    # Harmonize join keys
    gw_key_m = _pick_gw_col(minutes.columns.tolist()) or "gw_orig"
    gw_key_t = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"
    if gw_key_m in minutes.columns:
        minutes[gw_key_m] = pd.to_numeric(minutes[gw_key_m], errors="coerce")
    if gw_key_t in team_fix.columns:
        team_fix[gw_key_t] = pd.to_numeric(team_fix[gw_key_t], errors="coerce")
    if "team_id" in minutes.columns:
        minutes["team_id"] = minutes["team_id"].astype(str)
    if "team_id" in team_fix.columns:
        team_fix["team_id"] = team_fix["team_id"].astype(str)

    # Bring in is_home
    venue_cols = ["season", "team_id", gw_key_t, "is_home"]
    vmap = (team_fix[venue_cols]
            .dropna(subset=[gw_key_t, "team_id"])
            .drop_duplicates()
            .rename(columns={gw_key_t: gw_key_m}))
    minutes = minutes.merge(
        vmap, how="left", on=["season", "team_id", gw_key_m],
        validate="many_to_one", copy=False,
    )

    if "is_home" not in minutes.columns:
        if "venue" in minutes.columns:
            minutes["is_home"] = minutes["venue"].astype(str).str.lower().eq("home").astype("Int8")
        else:
            minutes["is_home"] = np.nan

    venue_fallback = (
        minutes["venue"].astype(str).str.lower().eq("home").astype("Int8")
        if "venue" in minutes.columns else 0
    )
    minutes["venue_bin"] = (
        pd.to_numeric(minutes["is_home"], errors="coerce")
          .fillna(venue_fallback)
          .fillna(0)
          .astype("Int8")
    )

    # Build future scoring frame
    fut = minutes.copy()

    # -------- NEW: Venue-consistent, DGW-safe FDR attach ("Int8") --------
    fut = attach_fdr_consistent(
        df=fut,
        seasons_all=seasons_all,
        features_root=args.features_root,
        version=args.form_version,
        team_form=tf
    )
    # Ensure int dtype (attach_fdr_consistent guarantees no NaNs)
    fut["fdr"] = pd.to_numeric(fut["fdr"], errors="raise").astype("Int8")

    # Roster gate
    allowed_pairs = _load_roster_pairs(
        teams_json=args.teams_json,
        season=args.future_season,
        league_filter=(args.league_filter.strip() or None)
    )
    fut = _apply_roster_gate(
        fut,
        allowed_pairs=allowed_pairs,
        season=args.future_season,
        where="ga_scoring",
        out_artifacts_dir=artifacts_dir,
        require_on_roster=args.require_on_roster
    )
    if fut.empty:
        raise RuntimeError("All rows were dropped by roster gating; nothing to score.")

    # Team z-venue merge (by GW)
    tz_map = _team_z_venue(tf)
    if tz_map is not None and (gw_key_m in fut.columns):
        fut = fut.merge(
            tz_map.rename(columns={"gw_orig": gw_key_m}),
            how="left",
            on=["season", gw_key_m, "team_id"],
            validate="many_to_one"
        )
    else:
        fut["team_att_z_venue"] = np.nan
        fut["opp_def_z_venue"] = np.nan

    # --- Attach legacy metadata from fixtures (fbref_id→game_id, team, opponent_id, opponent)
    fut = _attach_legacy_meta(fut, team_fix)

    # Last-known snapshot features
    pull_cols = {c for c in pf_hist.columns if ("_ewm" in c or "_roll" in c)}
    pull_cols |= {"minutes"}
    snap_cols = list(pull_cols)

    gw_cols = [c for c in ("gw_played", "gw_orig", "gw") if c in pf_hist.columns]
    base_cols = ["season", "player_id", "date_played"] + snap_cols + gw_cols

    last = _last_snapshot_per_player(
        pf_hist[base_cols].copy(),
        feature_cols=list(pull_cols),
        as_of_ts=as_of_ts,
        tz=tz
    )

    # days_since_last
    last_play = (
        pf_hist.sort_values(["player_id", "season", "date_played"])
        .groupby(["season", "player_id"], as_index=False)
        .tail(1)
        .loc[:, ["season", "player_id", "date_played"]]
    )
    last_play["date_played"] = _coerce_ts(last_play["date_played"], tz)
    fut["date_sched"] = _coerce_ts(fut["date_sched"], tz)

    # use tz-aware for the subtraction, THEN strip tz for persisted field
    fut = fut.merge(last_play, how="left", on=["season", "player_id"], validate="many_to_one")
    fut["days_since_last"] = (fut["date_sched"] - fut["date_played"]).dt.days.clip(lower=0)
    fut.drop(columns=["date_played"], inplace=True)

    # >>> canonicalize to tz-naive, date-only (pre-validation!)
    fut["date_sched"] = _to_naive_date(fut["date_sched"], tz)


    if "prev_minutes" not in fut.columns:
        last_prev = last[["season", "player_id", "minutes"]].rename(columns={"minutes": "prev_minutes"})
        fut = fut.merge(last_prev, how="left", on=["season", "player_id"], validate="many_to_one")

    last_feat = last.drop(columns=["minutes"], errors="ignore")
    feat_like = [c for c in last_feat.columns if c not in ("season", "player_id")]
    if feat_like:
        fut = fut.merge(
            last_feat[["season", "player_id"] + feat_like],
            how="left",
            on=["season", "player_id"],
            validate="many_to_one"
        )

    if fut.columns.duplicated().any():
        dups = fut.columns[fut.columns.duplicated()].tolist()
        logging.warning("Duplicate columns detected; keeping last occurrence: %s", set(dups))
        fut = fut.loc[:, ~fut.columns.duplicated(keep="last")]

    # --- Feature matrix in exact training order ---
    def _get_unique_col(df: pd.DataFrame, name: str) -> pd.Series:
        obj = df[name]
        if isinstance(obj, pd.DataFrame):
            return obj[obj.columns[-1]]
        return obj

    feat_path_cols = _load_json(args.model_dir / "artifacts" / "features.json") or []
    X = pd.DataFrame(index=fut.index)
    for c in feat_path_cols:
        if c in fut.columns:
            X[c] = pd.to_numeric(_get_unique_col(fut, c), errors="coerce")
        else:
            X[c] = np.nan

    na_rate = X.isna().mean()
    bad = na_rate[na_rate > 0.5]
    if not bad.empty:
        logging.warning("High missingness in features used by LGBM (top offenders):\n%s",
                        bad.sort_values(ascending=False).head(15))

    if list(X.columns) != list(feat_path_cols):
        raise AssertionError(
            "Feature order mismatch.\n"
            f"Expected (from features.json): {feat_path_cols}\n"
            f"Got: {list(X.columns)}"
        )

    pos_ser = fut["pos"].astype(str).str.upper()

    # Predict per-90 (LGBM)
    g_p90_mean = _predict_per_pos("goals", X, pos_ser, args.model_dir)
    a_p90_mean = _predict_per_pos("assists", X, pos_ser, args.model_dir)

    # Optional per-POS clamp
    caps = _parse_cap_per90(args.cap_per90)
    if caps:
        for tag, cap in caps.items():
            mask = pos_ser.eq(tag).to_numpy()
            if mask.any():
                g_p90_mean[mask] = np.clip(g_p90_mean[mask], 0.0, cap)
                a_p90_mean[mask] = np.clip(a_p90_mean[mask], 0.0, cap)

    # Optional Poisson/Tweedie heads
    med_train = _read_features_median(args.model_dir)
    g_p90_pois = _predict_poisson_per_pos("goals", X, pos_ser, args.model_dir, med=med_train)
    a_p90_pois = _predict_poisson_per_pos("assists", X, pos_ser, args.model_dir, med=med_train)

    # Scale to per-match using pred_minutes
    scale = pd.to_numeric(fut["pred_minutes"], errors="coerce").fillna(0.0).to_numpy() / 90.0
    pred_goals_mean   = g_p90_mean * scale
    pred_assists_mean = a_p90_mean * scale
    pred_goals_pois   = g_p90_pois * scale
    pred_assists_pois = a_p90_pois * scale

    # Combine to probabilities
    rg90 = np.where(~np.isnan(g_p90_pois), g_p90_pois, g_p90_mean)
    ra90 = np.where(~np.isnan(a_p90_pois), a_p90_pois, a_p90_mean)

    have_mix = all(c in fut.columns for c in ["p_start", "p_cameo", "pred_start_head", "pred_bench_cameo_head"])
    if have_mix:
        ps = fut["p_start"].clip(0, 1).to_numpy()
        pc = fut["p_cameo"].clip(0, 1).to_numpy()
        ms = np.clip(pd.to_numeric(fut["pred_start_head"], errors="coerce").fillna(0).to_numpy(), 0, None)
        mb = np.clip(pd.to_numeric(fut["pred_bench_cameo_head"], errors="coerce").fillna(0).to_numpy(), 0, None)

        lam_g_s = rg90 * (ms / 90.0); lam_g_b = rg90 * (mb / 90.0)
        lam_a_s = ra90 * (ms / 90.0); lam_a_b = ra90 * (mb / 90.0)

        p_goal_raw   = ps*(1.0 - np.exp(-lam_g_s)) + (1.0-ps)*pc*(1.0 - np.exp(-lam_g_b))
        p_assist_raw = ps*(1.0 - np.exp(-lam_a_s)) + (1.0-ps)*pc*(1.0 - np.exp(-lam_a_b))
    else:
        lam_g_eff = rg90 * scale
        lam_a_eff = ra90 * scale
        p_goal_raw   = 1.0 - np.exp(-lam_g_eff)
        p_assist_raw = 1.0 - np.exp(-lam_a_eff)

    p_goal   = np.clip(p_goal_raw, 0, 1)
    p_assist = np.clip(p_assist_raw, 0, 1)

    if args.apply_calibration:
        try:
            goal_cal = joblib.load(args.model_dir / "artifacts" / "p_goal_isotonic_per_pos.joblib")
            ass_cal  = joblib.load(args.model_dir / "artifacts" / "p_assist_isotonic_per_pos.joblib")
            def _apply_iso(p: np.ndarray, pos: pd.Series, cal_map: dict) -> np.ndarray:
                pout = p.copy()
                for tag, iso in (cal_map or {}).items():
                    m = pos.str.upper().eq(tag).to_numpy()
                    if m.any():
                        pout[m] = iso.transform(pout[m])
                return np.clip(pout, 0, 1)
            p_goal   = _apply_iso(p_goal,   pos_ser, goal_cal)
            p_assist = _apply_iso(p_assist, pos_ser, ass_cal)
        except Exception as e:
            logging.warning("Calibration artifacts missing or failed: %s", e)

    p_return_any = (1.0 - (1.0 - p_goal) * (1.0 - p_assist)) * (scale > 0)

    if args.skip_gk and pos_ser.eq("GK").any():
        m = pos_ser.eq("GK").to_numpy()
        for arr in (g_p90_mean, a_p90_mean, g_p90_pois, a_p90_pois,
                    pred_goals_mean, pred_assists_mean, pred_goals_pois, pred_assists_pois,
                    p_goal, p_assist, p_return_any):
            arr[m] = 0.0

    # ---------------------- Assemble output with legacy metadata ----------------------
    def pick_col(df: pd.DataFrame, *names: str):
        for n in names:
            if n in df.columns:
                return df[n].values
        return np.full(len(df), np.nan, dtype=object)

    opponent_id_vals    = pick_col(fut, "opponent_id", "opp_team_id", "opp_id")
    opponent_name_vals  = pick_col(fut, "opponent", "opp_team", "opp_name")
    game_id_vals        = pick_col(fut, "fbref_id", "fixture_id", "game_id")  # legacy: expose as game_id
    team_name_vals      = pick_col(fut, "team", "team_name", "team_short", "team_long")

    gw_played_vals = pd.to_numeric(fut.get("gw_played", np.nan), errors="coerce").values
    gw_orig_vals   = pd.to_numeric(fut.get("gw_orig",   fut.get("gw", np.nan)), errors="coerce").values

    # fdr now guaranteed non-null int
    fdr_vals       = pd.to_numeric(fut.get("fdr", np.nan), errors="raise").astype("Int8").values

    p_start_vals = pd.to_numeric(fut.get("p_start", np.nan), errors="coerce").clip(0, 1).values
    p60_vals     = pd.to_numeric(fut.get("p60", np.nan), errors="coerce").clip(0, 1).values
    p_cameo_vals = pd.to_numeric(fut.get("p_cameo", np.nan), errors="coerce").clip(0, 1).values
    p_play_vals  = pd.to_numeric(fut.get("p_play", np.nan), errors="coerce").clip(0, 1).values

    pred_start_head_vals       = pd.to_numeric(fut.get("pred_start_head", np.nan), errors="coerce").values
    pred_bench_cameo_head_vals = pd.to_numeric(fut.get("pred_bench_cameo_head", np.nan), errors="coerce").values
    pred_bench_head_vals       = pd.to_numeric(fut.get("pred_bench_head", np.nan), errors="coerce").values
    exp_minutes_points_vals    = pd.to_numeric(fut.get("exp_minutes_points", np.nan), errors="coerce").values
    is_synth_vals              = fut.get("_is_synth", pd.Series([False]*len(fut))).astype(bool).values

    out = pd.DataFrame({
        # --- Legacy/fixture metadata first (match minutes_forecast) ---
        "season":            fut["season"].values,
        "gw_played":         gw_played_vals,
        "gw_orig":           gw_orig_vals,
        "date_sched":        _to_naive_date(fut["date_sched"], tz).values,
        "game_id":           game_id_vals,       # fbref_id → game_id
        "team_id":           fut.get("team_id", pd.Series([np.nan]*len(fut))).astype(str).values,
        "team":              team_name_vals,
        "opponent_id":       opponent_id_vals,
        "opponent":          opponent_name_vals,
        "is_home":           pd.to_numeric(fut.get("is_home", np.nan), errors="coerce").fillna(0).astype("Int8").values,

        # --- Player context ---
        "player_id":         fut["player_id"].astype(str).values,
        "player":            fut.get("player", pd.Series([np.nan]*len(fut))).values,
        "pos":               fut["pos"].astype(str).values,
        "fdr":               fdr_vals,
        "venue_bin":         pd.to_numeric(fut.get("venue_bin", 0), errors="coerce").astype("Int8").values,

        # --- Minutes heads & gates (for consumers) ---
        "p_start":           p_start_vals,
        "p60":               p60_vals,
        "p_cameo":           p_cameo_vals,
        "p_play":            p_play_vals,
        "pred_start_head":       pred_start_head_vals,
        "pred_bench_cameo_head": pred_bench_cameo_head_vals,
        "pred_bench_head":       pred_bench_head_vals,
        "pred_minutes":      pd.to_numeric(fut["pred_minutes"], errors="coerce").values,
        "exp_minutes_points": exp_minutes_points_vals,
        "_is_synth":         is_synth_vals,

        # --- Context z-scores (diagnostics) ---
        "team_att_z_venue":  fut.get("team_att_z_venue", pd.Series([np.nan]*len(fut))).values,
        "opp_def_z_venue":   fut.get("opp_def_z_venue", pd.Series([np.nan]*len(fut))).values,

        # --- GA predictions ---
        "pred_goals_p90_mean":    g_p90_mean,
        "pred_assists_p90_mean":  a_p90_mean,
        "pred_goals_mean":        pred_goals_mean,
        "pred_assists_mean":      pred_assists_mean,
        "pred_goals_p90_poisson": g_p90_pois,
        "pred_assists_p90_poisson": a_p90_pois,
        "pred_goals_poisson":     pred_goals_pois,
        "pred_assists_poisson":   pred_assists_pois,
        "p_goal":                 p_goal,
        "p_assist":               p_assist,
        "p_return_any":           p_return_any,
    })

    # Desired column order: meta → player/gates → context → preds
    desired_order = [
        "season","gw_played","gw_orig","date_sched","game_id",
        "team_id","team","opponent_id","opponent","is_home",
        "player_id","player","pos","fdr","venue_bin",
        "p_start","p60","p_cameo","p_play",
        "pred_start_head","pred_bench_cameo_head","pred_bench_head",
        "pred_minutes","exp_minutes_points","_is_synth",
        "team_att_z_venue","opp_def_z_venue",
        "pred_goals_p90_mean","pred_assists_p90_mean",
        "pred_goals_mean","pred_assists_mean",
        "pred_goals_p90_poisson","pred_assists_p90_poisson",
        "pred_goals_poisson","pred_assists_poisson",
        "p_goal","p_assist","p_return_any"
    ]
    keep_cols = [c for c in desired_order if c in out.columns]
    out = out[keep_cols].copy()

    # enforce integer dtypes (nullable ints preserve NA)
    if "gw_played" in out.columns:
        out["gw_played"] = pd.to_numeric(out["gw_played"], errors="coerce").astype("Int8")
    if "gw_orig" in out.columns:
        out["gw_orig"] = pd.to_numeric(out["gw_orig"], errors="coerce").astype("Int8")
    if "fdr" in out.columns:
        out["fdr"] = pd.to_numeric(out["fdr"], errors="coerce").astype("Int8")
    if "is_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").astype("Int8")
    if "venue_bin" in out.columns:
        out["venue_bin"] = pd.to_numeric(out["venue_bin"], errors="coerce").astype("Int8")

    sort_keys = [k for k in ["gw_orig", "team_id", "player_id"] if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    # --- schema validation before write ---
    GA_SCHEMA = {
        "required": ["season","gw_played","gw_orig","date_sched","game_id","team_id","team",
                     "opponent_id","opponent","is_home","player_id","player","pos","fdr",
                     "p_start","p60","p_cameo","p_play","pred_start_head","pred_bench_cameo_head",
                     "pred_bench_head","pred_minutes","exp_minutes_points",
                     "team_att_z_venue","opp_def_z_venue",
                     "pred_goals_p90_mean","pred_assists_p90_mean",
                     "pred_goals_mean","pred_assists_mean",
                     "pred_goals_p90_poisson","pred_assists_p90_poisson",
                     "pred_goals_poisson","pred_assists_poisson",
                     "p_goal","p_assist","p_return_any"],
        "dtypes": {
            "season":"string",
            "gw_played":"Int8",          # align with cast above
            "gw_orig":"Int8",            # align with cast above
            "date_sched":"datetime64[ns]",
            "game_id":"object",
            "team_id":"string",
            "team":"object",
            "opponent_id":"object",
            "opponent":"object",
            "is_home":"Int8",             # align with cast above
            "player_id":"string",
            "player":"object",
            "pos":"string",
            "fdr":"Int8",                 # align with cast above
            "venue_bin":"Int8",           # explicitly declare for consistency
            "p_start":"float","p60":"float","p_cameo":"float","p_play":"float",
            "pred_start_head":"float","pred_bench_cameo_head":"float","pred_bench_head":"float",
            "pred_minutes":"float","exp_minutes_points":"float",
            "team_att_z_venue":"float","opp_def_z_venue":"float",
            "pred_goals_p90_mean":"float","pred_assists_p90_mean":"float",
            "pred_goals_mean":"float","pred_assists_mean":"float",
            "pred_goals_p90_poisson":"float","pred_assists_p90_poisson":"float",
            "pred_goals_poisson":"float","pred_assists_poisson":"float",
            "p_goal":"float","p_assist":"float","p_return_any":"float",
        },
        "na": {"gw_orig": False, "fdr": False, "is_home": False},
        "ranges": {
            "p_start":{"min":0.0,"max":1.0}, "p60":{"min":0.0,"max":1.0},
            "p_cameo":{"min":0.0,"max":1.0}, "p_play":{"min":0.0,"max":1.0},
            "exp_minutes_points":{"min":0.0,"max":2.0},
            "p_goal":{"min":0.0,"max":1.0}, "p_assist":{"min":0.0,"max":1.0}, "p_return_any":{"min":0.0,"max":1.0},
            "pred_minutes":{"min":0.0}
        },
        "choices": {"pos":{"in":["GK","DEF","MID","FWD"]}},
        "logic": [("is_home in {0,1}", ["is_home"])],
        "date_rules": {"normalize":["date_sched"]},
        # DGW-safe uniqueness (two fixtures same GW → distinct game_id)
        "unique": ["season","gw_orig","team_id","player_id","game_id"]
    }
    validate_df(out, GA_SCHEMA, name="goals_assists_forecast")

    gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
    out_paths = _ga_out_paths(
        base_dir=args.out_dir,
        season=args.future_season,
        gw_from=gw_from_eff,
        gw_to=gw_to_eff,
        zero_pad=args.zero_pad_filenames,
        out_format=args.out_format,
    )
    written_paths = _write_ga(out, out_paths)

    # >>> NEW: update consolidated season-level files (also string dates)
    season_dir.mkdir(parents=True, exist_ok=True)
    consolidated_paths = _update_consolidated_ga(
        out_df=out,
        season_dir=season_dir,
        out_format=args.out_format,
        desired_order=desired_order,
        run_gw_from=gw_from_eff,
        run_gw_to=gw_to_eff,
    )
    try:
        def q995(x): return float(pd.Series(x).quantile(0.995))
        for nm, arr in [("g90_LGBM", g_p90_mean), ("a90_LGBM", a_p90_mean),
                        ("g90_POIS", g_p90_pois), ("a90_POIS", a_p90_pois)]:
            logging.info("%s 99.5th percentile: %.3f", nm, q995(arr))
    except Exception:
        pass

    diag = {
        "rows": int(len(out)),
        "season": str(args.future_season),
        "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
        "available_team_gws": [int(x) for x in avail_gws],
        "scored_gws": [int(x) for x in target_gws],
        "as_of": str(as_of_ts),
        "minutes_in": str(minutes_path),
        "out": written_paths,
        "consolidated_out": consolidated_paths
    }
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
