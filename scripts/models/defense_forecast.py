#!/usr/bin/env python3
r"""
defense_forecast.py — leak-free forecaster for CS, DCP, and exp_gc using trained defense heads
Now with minutes auto-resolve (CSV/Parquet), legacy fixture metadata, roster gating, and dual writer.
Also writes/maintains a season-level consolidated file, e.g.:
  <out-dir>/<SEASON>/defense.csv  and/or  defense.parquet  (according to --out-format)

[see original docstring for details]
"""

from __future__ import annotations
import argparse, json, logging, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from scripts.utils.validate import validate_df
# ----------------------------- helpers ----------------------------------------

def _load_json(p: Path) -> list | dict:
    if not p.exists():
        raise FileNotFoundError(f"Missing artifact: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def _try_load_json(p: Path) -> Optional[list | dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("Failed to parse %s; falling back if possible.", p)
        return None

def _pick_gw_col(cols: List[str]) -> Optional[str]:
    for k in ("gw_played","gw_orig","gw"):
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

def _load_players_form(features_root: Path, form_version: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "players_form.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Missing players_form: {fp}")
        t = pd.read_csv(fp, parse_dates=["date_played"])
        t["season"] = s
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    need = {"season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"players_form missing: {miss}")
    df["pos"] = df["pos"].astype(str).str.upper()
    return df

def _load_team_form(features_root: Path, form_version: str, seasons: List[str]) -> Optional[pd.DataFrame]:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "team_form.csv"
        if fp.exists():
            t = pd.read_csv(fp, parse_dates=["date_played"])
            t["season"] = s
            frames.append(t)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def _map_team_context(tf: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Build per (season, gw_orig, team_id, is_home) table with:
      team_def_xga_venue, team_def_xga_venue_z, team_possession_venue,
      team_att_z_venue, opp_att_z_venue, is_home
    """
    if tf is None:
        return None

    t = tf.copy()

    # Mark is_home if venue exists
    if "venue" in t.columns:
        t["is_home"] = t["venue"].astype(str).str.lower().eq("home").astype("Int8")
    elif "was_home" in t.columns:
        t["is_home"] = t["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype("Int8")
    else:
        t["is_home"] = np.nan

    # team_def_xga_venue[_z]
    if "venue" in t.columns and {"def_xga_home_roll","def_xga_away_roll"}.issubset(t.columns):
        t["team_def_xga_venue"] = np.where(
            t["venue"].astype(str).str.lower().eq("home"),
            t["def_xga_home_roll"], t["def_xga_away_roll"]
        )
    elif "def_xga_roll" in t.columns:
        t["team_def_xga_venue"] = t["def_xga_roll"]
    else:
        t["team_def_xga_venue"] = np.nan

    if "venue" in t.columns and {"def_xga_home_roll_z","def_xga_away_roll_z"}.issubset(t.columns):
        t["team_def_xga_venue_z"] = np.where(
            t["venue"].astype(str).str.lower().eq("home"),
            t["def_xga_home_roll_z"], t["def_xga_away_roll_z"]
        )
    elif "def_xga_roll_z" in t.columns:
        t["team_def_xga_venue_z"] = t["def_xga_roll_z"]
    else:
        t["team_def_xga_venue_z"] = np.nan

    # possession (venue-aware if available)
    if "venue" in t.columns and {"possession_home_roll","possession_away_roll"}.issubset(t.columns):
        t["team_possession_venue"] = np.where(
            t["venue"].astype(str).str.lower().eq("home"),
            t["possession_home_roll"], t["possession_away_roll"]
        )
    elif "possession_roll" in t.columns:
        t["team_possession_venue"] = t["possession_roll"]
    elif "possession" in t.columns:
        t["team_possession_venue"] = t["possession"]
    else:
        t["team_possession_venue"] = np.nan

    # team_att_z_venue (for opp map and optional output)
    if "venue" in t.columns and {"att_xg_home_roll_z","att_xg_away_roll_z"}.issubset(t.columns):
        t["team_att_z_venue"] = np.where(
            t["venue"].astype(str).str.lower().eq("home"),
            t["att_xg_home_roll_z"], t["att_xg_away_roll_z"]
        )
    elif "att_xg_roll_z" in t.columns:
        t["team_att_z_venue"] = t["att_xg_roll_z"]
    else:
        t["team_att_z_venue"] = np.nan

    # opponent attacking z (from opponent's team_att_z_venue)
    if {"home_id","away_id","team_id"}.issubset(t.columns):
        t["opp_id"] = np.where(t["team_id"]==t["home_id"], t["away_id"], t["home_id"])
    else:
        t["opp_id"] = np.nan
    opp = (t[["season","gw_orig","team_id","team_att_z_venue"]]
           .drop_duplicates()
           .rename(columns={"team_id":"opp_id","team_att_z_venue":"opp_att_z_venue"}))

    out = (t.merge(opp, on=["season","gw_orig","opp_id"], how="left")
             [["season","gw_orig","team_id","is_home",
               "team_def_xga_venue","team_def_xga_venue_z",
               "team_possession_venue","team_att_z_venue","opp_att_z_venue"]])

    for c in ["team_def_xga_venue","team_def_xga_venue_z","team_possession_venue","team_att_z_venue","opp_att_z_venue"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    dups = out.duplicated(["season","gw_orig","team_id","is_home"]).sum()
    if dups:
        logging.warning("team_ctx has %d duplicate rows across (season, gw_orig, team_id, is_home); keeping last.", dups)
        out = (out.sort_values(["season","gw_orig","team_id","is_home"])
                  .drop_duplicates(subset=["season","gw_orig","team_id","is_home"], keep="last"))

    return out

def _load_team_fixtures(fix_root: Path, season: str, filename: str) -> pd.DataFrame:
    path = fix_root / season / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing team fixtures: {path}")
    tf = pd.read_csv(path)
    for dc in ("date_sched","date_played"):
        if dc in tf.columns:
            tf[dc] = pd.to_datetime(tf[dc], errors="coerce")
    for c in ("gw_played","gw_orig","gw"):
        if c in tf.columns:
            tf[c] = pd.to_numeric(tf[c], errors="coerce")
    if "is_home" not in tf.columns:
        if "was_home" in tf.columns:
            tf["is_home"] = tf["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype("Int8")
        elif "venue" in tf.columns:
            tf["is_home"] = tf["venue"].astype(str).str.lower().eq("home").astype("Int8")
        else:
            tf["is_home"] = 0
    if "team_id" not in tf.columns:
        for alt in ("team","teamId","team_code"):
            if alt in tf.columns:
                tf = tf.rename(columns={alt:"team_id"})
                break
    tf["team_id"] = tf["team_id"].astype(str)
    tf["season"] = season
    return tf

def _gw_for_selection(df: pd.DataFrame) -> pd.Series:
    def num(s): return pd.to_numeric(df.get(s), errors="coerce")
    gwp = num("gw_played"); gwo = num("gw_orig"); gwa = num("gw")
    return gwo.where(gwp.isna() | (gwp <= 0), gwp).where(lambda x: x.notna(), gwa)

def _last_snapshot_per_player(df: pd.DataFrame, feature_cols: List[str], as_of_ts: pd.Timestamp, tz: Optional[str]) -> pd.DataFrame:
    du = pd.to_datetime(df["date_played"], errors="coerce")
    if tz:
        if du.dt.tz is None:
            du = du.dt.tz_localize(tz)
        else:
            du = du.dt.tz_convert(tz)
    hist = df[du < as_of_ts].copy()
    if hist.empty:
        return pd.DataFrame(columns=["season","player_id"] + feature_cols)
    gw_key = _pick_gw_col(hist.columns.tolist()) or "gw_orig"
    hist = hist.sort_values(["player_id","season","date_played", gw_key])
    last = hist.groupby(["season","player_id"], as_index=False).tail(1).copy()
    keep = ["season","player_id"] + [c for c in feature_cols if c in last.columns]
    for c in feature_cols:
        if c not in last.columns:
            last[c] = np.nan
    return last[["season","player_id"] + feature_cols].copy()

def _poisson_tail_prob_vec(lam: np.ndarray, k: np.ndarray) -> np.ndarray:
    """P(K >= k) for K ~ Poisson(lam)."""
    lam = np.asarray(lam, dtype=float); k = np.asarray(k, dtype=int)
    out = np.zeros_like(lam, dtype=float)

    def tail_one(l, kk):
        if not np.isfinite(l) or l <= 0: return 0.0 if kk > 0 else 1.0
        if kk <= 0: return 1.0
        term = math.exp(-l) * (l ** kk) / math.factorial(kk)
        s = term; i = kk + 1
        for _ in range(200):
            term *= (l / i); s += term
            if term < 1e-12: break
            i += 1
        return float(min(max(s, 0.0), 1.0))
    for i in range(lam.shape[0]): out[i] = tail_one(lam[i], int(k[i]))
    return out

def _parse_k90(s: str) -> Dict[str, int]:
    out = {"DEF":10, "MID":12, "FWD":12}
    if not s: return out
    for part in s.split(";"):
        if not part.strip(): continue
        k, v = part.split(":"); out[k.strip().upper()] = int(v)
    return out

# ---------- roster gating ----------
def _norm_label(s: str) -> str:
    return str(s or "").lower().replace("-", " ").replace("_", " ").strip()

def _load_roster_pairs(teams_json: Optional[Path],
                       season: str,
                       league_filter: Optional[str]) -> Optional[Set[Tuple[str, str]]]:
    if not teams_json: return None
    p = Path(teams_json)
    if not p.exists():
        logging.warning("teams_json not found at %s — skipping roster gate."); return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning("Failed to parse teams_json (%s): %s — skipping roster gate.", p, e); return None

    lf = _norm_label(league_filter) if league_filter else ""
    allowed: Set[Tuple[str, str]] = set()
    for team_id, obj in (data or {}).items():
        season_info = (obj or {}).get("career", {}).get(season)
        if not season_info: continue
        if lf and _norm_label(season_info.get("league", "")) != lf: continue
        for pl in season_info.get("players", []) or []:
            pid = str(pl.get("id", "")).strip()
            if pid: allowed.add((str(team_id), pid))
    if not allowed:
        logging.warning("Roster map for %s produced 0 allowed pairs (league=%r).", season, league_filter)
    return allowed or None

def _apply_roster_gate(df: pd.DataFrame,
                       allowed_pairs: Optional[Set[Tuple[str, str]]],
                       season: str,
                       where: str,
                       out_artifacts_dir: Optional[Path] = None,
                       require_on_roster: bool = False) -> pd.DataFrame:
    if allowed_pairs is None or df.empty: return df
    tid = df.get("team_id").astype(str); pid = df.get("player_id").astype(str)
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

# ---------- minutes resolver (auto) & dual loader ----------
def _fmt_gw(n: int, zero_pad: bool) -> str:
    return f"{int(n):02d}" if zero_pad else f"{int(n)}"

def _candidate_minutes_paths(minutes_root: Path, future_season: str, gw_from: int, gw_to: int) -> List[Path]:
    season_dir = minutes_root / str(future_season)
    cands: List[Path] = []
    for zp in (False, True):
        a = _fmt_gw(gw_from, zp); b = _fmt_gw(gw_to,   zp)
        cands.append(season_dir / f"GW{a}_{b}.csv")
        cands.append(season_dir / f"GW{a}_{b}.parquet")
    return cands

def _glob_fallback(minutes_root: Path, future_season: str, gw_from: int, gw_to: int) -> Optional[Path]:
    season_dir = minutes_root / str(future_season)
    if not season_dir.exists(): return None
    patterns = [f"GW{gw_from}_*.csv", f"GW{gw_from}_*.parquet",
                f"GW{gw_from:02d}_*.csv", f"GW{gw_from:02d}_*.parquet"]
    for pat in patterns:
        for p in sorted(season_dir.glob(pat)):
            try:
                stem = p.stem; to_val = int(stem.split("_")[-1].replace("GW",""))
                if to_val == int(gw_to): return p
            except Exception: continue
    return None

def _resolve_minutes_path(args: argparse.Namespace, gw_from: int, gw_to: int) -> Path:
    if args.minutes_csv:
        p = Path(args.minutes_csv)
        if not p.exists(): raise FileNotFoundError(f"--minutes-csv not found: {p}")
        return p
    for cand in _candidate_minutes_paths(args.minutes_root, args.future_season, gw_from, gw_to):
        if cand.exists(): return cand
    fb = _glob_fallback(args.minutes_root, args.future_season, gw_from, gw_to)
    if fb: return fb
    season_dir = args.minutes_root / str(args.future_season)
    msg = [f"Minutes file not found for GW window {gw_from}-{gw_to}.",
           f"Looked under: {season_dir}", "Tried candidates:"]
    for c in _candidate_minutes_paths(args.minutes_root, args.future_season, gw_from, gw_to):
        msg.append(f"  - {c}")
    msg += ["Also tried glob fallback: GW{from}_*.{csv,parquet} (with and without zero-padding).",
            "Fix by either:",
            "  • generating that file, or",
            "  • pointing directly via --minutes-csv <path>, or",
            "  • adjusting --minutes-root / GW window."]
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
    return df

# ---------- FDR (venue-consistent, DGW-safe) ----------
def _find_fdr_cols(cols: set[str]) -> tuple[str, str]:
    home_aliases = ["fdr_home", "team_fdr_home", "def_fdr_home", "fdrH"]
    away_aliases = ["fdr_away", "team_fdr_away", "def_fdr_away", "fdrA"]
    home = next((c for c in home_aliases if c in cols), None)
    away = next((c for c in away_aliases if c in cols), None)
    if not home or not away:
        raise RuntimeError(
            f"FDR columns not found in team_form. Tried {home_aliases} and {away_aliases}. Got: {sorted(cols)}"
        )
    return home, away

def _ensure_is_home(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype("Int8")
        return out
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
    for c in ("gw_played","gw_orig","gw"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    gw_df = _pick_gw_col(df.columns.tolist())
    if gw_df is None:
        raise RuntimeError("attach_fdr_consistent: no GW column among ['gw_played','gw_orig','gw'].")

    tf_all = team_form.copy() if team_form is not None else _load_team_form(features_root, version, seasons_all)
    if tf_all is None or tf_all.empty:
        raise FileNotFoundError("attach_fdr_consistent: team_form not available to attach FDR.")

    tf_all["team_id"] = tf_all.get("team_id", pd.Series(index=tf_all.index, dtype=object)).astype(str)
    for c in ("gw_played","gw_orig","gw"):
        if c in tf_all.columns:
            tf_all[c] = pd.to_numeric(tf_all[c], errors="coerce")

    home_col, away_col = _find_fdr_cols(set(tf_all.columns))
    gw_tf = _pick_gw_col(tf_all.columns.tolist())
    if gw_tf is None:
        raise RuntimeError("attach_fdr_consistent: no GW column in team_form.")

    base = tf_all[["season","team_id",gw_tf,home_col,away_col]].dropna(subset=["team_id",gw_tf])
    home_rows = base.rename(columns={home_col: "fdr_side"}).assign(is_home=1)[["season","team_id",gw_tf,"is_home","fdr_side"]]
    away_rows = base.rename(columns={away_col: "fdr_side"}).assign(is_home=0)[["season","team_id",gw_tf,"is_home","fdr_side"]]
    form_long = pd.concat([home_rows, away_rows], ignore_index=True)
    if gw_tf != gw_df:
        form_long = form_long.rename(columns={gw_tf: gw_df})

    form_long = (form_long.groupby(["season","team_id",gw_df,"is_home"], as_index=False)["fdr_side"].max())

    merged = df.merge(form_long, how="left", on=["season","team_id",gw_df,"is_home"], validate="many_to_one", copy=False)
    if merged["fdr_side"].isna().any():
        miss = merged.loc[merged["fdr_side"].isna(), ["season","team_id",gw_df,"is_home"]].drop_duplicates()
        logging.error("attach_fdr_consistent: missing FDR for %d rows. Examples:\n%s",
                      len(miss), miss.head(20).to_string(index=False))
        raise RuntimeError("attach_fdr_consistent: FDR merge produced NaNs. Check keys/coverage.")

    merged["fdr"] = pd.to_numeric(merged["fdr_side"], errors="raise").astype("Int8")
    merged.drop(columns=["fdr_side"], inplace=True, errors="ignore")
    return merged

# ---------- file writing helpers ----------
def _def_out_paths(base_dir: Path, season: str, gw_from: int, gw_to: int,
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

def _write_def(df: pd.DataFrame, paths: List[Path]) -> List[str]:
    written: List[str] = []
    for p in paths:
        if p.suffix.lower() == ".csv":
            tmp = df.copy()
            if "date_sched" in tmp.columns:
                tmp["date_sched"] = pd.to_datetime(tmp["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
            tmp.to_csv(p, index=False)
            written.append(str(p))
        elif p.suffix.lower() == ".parquet":
            df.to_parquet(p, index=False)
            written.append(str(p))
        else:
            raise ValueError(f"Unsupported output extension: {p.suffix}")
    return written

# ===================== CONSOLIDATED WRITER (season-level) =====================

def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        # date_sched may be already string; coerce later
        return pd.read_csv(path, parse_dates=["date_sched"]) if "date_sched" in pd.read_csv(path, nrows=0).columns else pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file to read: {path}")

def _update_consolidated(
    out_df: pd.DataFrame,
    season_dir: Path,
    out_format: str,
    desired_order: List[str],
) -> List[str]:
    """
    Maintain season-level consolidated files:
      <season_dir>/defense.csv and/or defense.parquet (per --out-format).
    DGW-safe: de-duplicate on a robust identity.
    """
    cons_stem = season_dir / "expected_defense"
    targets: List[Path] = []
    if out_format in ("csv", "both"): targets.append(Path(str(cons_stem) + ".csv"))
    if out_format in ("parquet", "both"): targets.append(Path(str(cons_stem) + ".parquet"))

    # Load whichever consolidated exists (prefer union if both exist)
    old_csv = Path(str(cons_stem) + ".csv")
    old_pq  = Path(str(cons_stem) + ".parquet")

    old = pd.DataFrame()
    if old_csv.exists(): old = _read_any(old_csv)
    if old_pq.exists():
        old_p = _read_any(old_pq)
        if old.empty: old = old_p
        else:
            # union in case they diverged
            old = pd.concat([old, old_p], ignore_index=True)

    # Normalize dtypes for keys
    for c in ("season","team_id","player_id","opponent_id","game_id"):
        if c in out_df.columns:
            out_df[c] = out_df[c].astype(str)
        if c in old.columns:
            old[c] = old[c].astype(str)

    # Align columns (full outer schema)
    all_cols = list(dict.fromkeys([*desired_order, *old.columns.tolist(), *out_df.columns.tolist()]))
    new = out_df.reindex(columns=all_cols)
    old = old.reindex(columns=all_cols)

    # Robust dedup key (DGW-safe): prefer game_id when present, otherwise gw_orig+date_sched+opponent_id
    key_candidates = [c for c in ["season","player_id","team_id","game_id","gw_orig","date_sched","opponent_id"] if c in all_cols]
    if not key_candidates:
        # fallback (shouldn't happen)
        key_candidates = [c for c in ["season","player_id","team_id","gw_orig"] if c in all_cols]

    merged = pd.concat([old, new], ignore_index=True)

    # Ensure date_sched is datetime for sorting & CSV formatting later
    if "date_sched" in merged.columns:
        merged["date_sched"] = pd.to_datetime(merged["date_sched"], errors="coerce")

    merged = merged.drop_duplicates(subset=key_candidates, keep="last")

    # Sort and project to the desired output column order we already use for window files
    keep_cols = [c for c in desired_order if c in merged.columns]
    merged = merged[keep_cols].copy()

    sort_keys = [k for k in ["gw_orig","team_id","player_id"] if k in merged.columns]
    if sort_keys:
        merged = merged.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    # Write back according to --out-format
    written: List[str] = []
    for tgt in targets:
        if tgt.suffix.lower() == ".csv":
            tmp = merged.copy()
            if "date_sched" in tmp.columns:
                tmp["date_sched"] = pd.to_datetime(tmp["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
            tmp.to_csv(tgt, index=False)
            written.append(str(tgt))
        else:
            merged.to_parquet(tgt, index=False)
            written.append(str(tgt))
    return written

# ----------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # windowing
    ap.add_argument("--history-seasons", required=True, help="Comma list of past seasons")
    ap.add_argument("--future-season", required=True, help="Season to score (contains future GWs)")
    ap.add_argument("--as-of", default="now", help='ISO timestamp or "now"')
    ap.add_argument("--as-of-tz", default="Africa/Lagos")
    ap.add_argument("--as-of-gw", type=int, required=True, help="First unplayed GW at --as-of (next GW)")
    ap.add_argument("--n-future", type=int, default=3)
    ap.add_argument("--gw-from", type=int, default=None)
    ap.add_argument("--gw-to", type=int, default=None)
    ap.add_argument("--strict-n-future", action="store_true")

    # IO
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"))
    ap.add_argument("--form-version", required=True)
    ap.add_argument("--fix-root", type=Path, default=Path("data/processed/registry/fixtures"))
    ap.add_argument("--team-fixtures-filename", default="fixture_calendar.csv")

    # Minutes input (auto-resolve)
    ap.add_argument("--minutes-csv", type=Path, help="Explicit minutes file (CSV or Parquet). Overrides auto-resolution.")
    ap.add_argument("--minutes-root", type=Path, default=Path("data/predictions/minutes"),
                    help="Root containing <season>/GW<from>_<to>.csv|parquet")

    # Output
    ap.add_argument("--out-dir", type=Path, default=Path("data/predictions/defense"))
    ap.add_argument("--out-format", choices=["csv","parquet","both"], default="csv")
    ap.add_argument("--zero-pad-filenames", action="store_true")

    # Policy / inference tweaks
    ap.add_argument("--dcp-k90", type=str, default="DEF:10;MID:12;FWD:12")
    ap.add_argument("--require-pred-minutes", action="store_true")
    ap.add_argument("--include-gk-dcp", action="store_true", help="If set, computes DCP for GK (otherwise NaN)")
    ap.add_argument("--dump-lambdas", action="store_true", help="Include lambda_match column")
    ap.add_argument("--log-level", default="INFO")

    # Artifacts
    ap.add_argument("--model-dir", type=Path, required=True, help="Folder with trained DEFENSE artifacts (a specific version)")

    # Roster gating
    ap.add_argument("--teams-json", type=Path, help="Path to master_teams.json containing per-season rosters")
    ap.add_argument("--league-filter", type=str, default="", help="Optional league name (e.g., 'ENG-Premier League')")
    ap.add_argument("--require-on-roster", action="store_true",
                    help="If set, error out when any scoring rows are not on the future-season roster")

    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    # seasons & as-of
    history = [s.strip() for s in args.history_seasons.split(",") if s.strip()]
    seasons_all = history + [args.future_season]
    tz = args.as_of_tz
    as_of_ts = (pd.Timestamp.now(tz=tz)
                if str(args.as_of).lower() in ("now","auto","today")
                else pd.Timestamp(args.as_of, tz=tz))

    # output dirs for artifacts
    season_dir = args.out_dir / f"{args.future_season}"
    artifacts_dir = season_dir / "artifacts"

    # --- Load artifacts ---
    cs_feats = _load_json(args.model_dir / "artifacts" / "cs_features_team.json")  # list[str]
    dcp_feats = _load_json(args.model_dir / "artifacts" / "dcp_features.json")     # list[str]
    gc_feats = _try_load_json(args.model_dir / "artifacts" / "gc_features_team.json")  # optional
    if gc_feats is None:
        gc_feats = cs_feats
        logging.info("gc_features_team.json not found; using cs_features_team.json order for GC.")

    cs_booster = lgb.Booster(model_file=str(args.model_dir / "team_cs_lgbm.txt"))
    iso_cs = None
    iso_path = args.model_dir / "cs_isotonic.joblib"
    if iso_path.exists():
        try:
            iso_cs = joblib.load(iso_path)
            logging.info("Loaded isotonic calibration for team CS.")
        except Exception:
            logging.warning("Failed loading isotonic calibration; proceeding without.")

    gc_model_path = args.model_dir / "team_gc_lgbm.joblib"
    gc_model = None
    if gc_model_path.exists():
        try:
            gc_model = joblib.load(gc_model_path)
            logging.info("Loaded team GC regressor (expected goals conceded).")
        except Exception as e:
            logging.warning("Failed loading GC regressor: %s — exp_gc will be NaN.", e)
            gc_model = None
    else:
        logging.info("team_gc_lgbm.joblib not found — exp_gc will be NaN.")

    dcp_models: Dict[str, object] = {}
    for tag in ("DEF","MID","FWD"):
        p = args.model_dir / f"dcp_{tag}_lgbm.joblib"
        if p.exists():
            dcp_models[tag] = joblib.load(p)
    if not dcp_models:
        raise FileNotFoundError("No DCP per-pos models found (expected dcp_DEF/MID/FWD_lgbm.joblib).")

    # --- Load registry up to as-of ---
    pf = _load_players_form(args.features_root, args.form_version, seasons_all)
    tf = _load_team_form(args.features_root, args.form_version, seasons_all)
    team_ctx = _map_team_context(tf)

    # --- Resolve minutes file & window (auto) ---
    gw_from_req = args.gw_from if args.gw_from is not None else args.as_of_gw
    gw_to_req   = args.gw_to   if args.gw_to   is not None else (gw_from_req + max(1, args.n_future) - 1)
    minutes_path = _resolve_minutes_path(args, int(gw_from_req), int(gw_to_req))
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
        raise RuntimeError("No rows after GW filtering.")
    if minutes.columns.duplicated().any():
        dups = minutes.columns[minutes.columns.duplicated()].tolist()
        logging.warning("Minutes had duplicate columns; dropping earlier duplicates: %s", set(dups))
        minutes = minutes.loc[:, ~minutes.columns.duplicated(keep="last")]

    # --- Roster gating ---
    allowed_pairs = _load_roster_pairs(
        teams_json=args.teams_json,
        season=args.future_season,
        league_filter=(args.league_filter.strip() or None)
    )
    minutes = _apply_roster_gate(
        minutes,
        allowed_pairs=allowed_pairs,
        season=args.future_season,
        where="def_scoring",
        out_artifacts_dir=artifacts_dir,
        require_on_roster=args.require_on_roster
    )
    if minutes.empty:
        raise RuntimeError("All rows were dropped by roster gating; nothing to score.")

    # --- Merge team fixtures for venue + legacy metadata ---
    team_fix = _load_team_fixtures(args.fix_root, args.future_season, args.team_fixtures_filename)

    gw_key_m = _pick_gw_col(minutes.columns.tolist()) or "gw_orig"
    gw_key_t = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"

    minutes[gw_key_m] = pd.to_numeric(minutes[gw_key_m], errors="coerce")
    team_fix[gw_key_t] = pd.to_numeric(team_fix[gw_key_t], errors="coerce")
    minutes["team_id"] = minutes["team_id"].astype(str)
    team_fix["team_id"] = team_fix["team_id"].astype(str)

    # Map is_home from fixtures
    vmap = (team_fix[["season","team_id",gw_key_t,"is_home"]]
            .dropna(subset=[gw_key_t,"team_id"])
            .drop_duplicates()
            .rename(columns={gw_key_t: gw_key_m}))
    minutes = minutes.merge(vmap, how="left", on=["season","team_id",gw_key_m], validate="many_to_one")

    # venue_bin AFTER is_home
    venue_fallback = (minutes["venue"].astype(str).str.lower().eq("home").astype("Int8")
                      if "venue" in minutes.columns else 0)
    minutes["venue_bin"] = (
        pd.to_numeric(minutes.get("is_home"), errors="coerce")
          .fillna(venue_fallback)
          .fillna(0)
          .astype("Int8")
    )

    # -------- Venue-consistent, DGW-safe FDR attach ("Int8") --------
    minutes = attach_fdr_consistent(
        df=minutes,
        seasons_all=seasons_all,
        features_root=args.features_root,
        version=args.form_version,
        team_form=tf
    )
    minutes["fdr"] = pd.to_numeric(minutes["fdr"], errors="raise").astype("Int8")

    # --- Team rows for CS/GC features (uses minutes' GW key) ---
    team_rows = (minutes[["season", gw_key_m, "team_id", "is_home", "venue_bin", "fdr", "date_sched"]]
                 .drop_duplicates()
                 .rename(columns={gw_key_m: "gw_orig"}))

    if team_ctx is not None:
        team_rows = team_rows.merge(
            team_ctx,
            on=["season","gw_orig","team_id","is_home"],
            how="left",
            validate="many_to_one",
        )
    else:
        for c in ["team_def_xga_venue","team_def_xga_venue_z","team_possession_venue","team_att_z_venue","opp_att_z_venue"]:
            team_rows[c] = np.nan

    # Assemble X_cs in artifact order
    Xcs = pd.DataFrame(index=team_rows.index)
    for f in cs_feats:
        if f in team_rows.columns:
            Xcs[f] = pd.to_numeric(team_rows[f], errors="coerce")
        elif f == "venue_bin":
            Xcs[f] = team_rows["venue_bin"]
        elif f == "fdr":
            Xcs[f] = pd.to_numeric(team_rows["fdr"], errors="coerce")
        else:
            Xcs[f] = np.nan
    Xcs = Xcs.fillna(0.0)

    # Predict team CS
    p_team_raw = np.clip(cs_booster.predict(Xcs), 0, 1)
    p_team = np.clip(iso_cs.transform(p_team_raw), 0, 1) if iso_cs is not None else p_team_raw
    team_rows["p_teamCS"] = p_team

    # GC regressor (optional)
    if gc_model is not None:
        Xgc = pd.DataFrame(index=team_rows.index)
        for f in gc_feats:
            if f in team_rows.columns:
                Xgc[f] = pd.to_numeric(team_rows[f], errors="coerce")
            elif f == "venue_bin":
                Xgc[f] = team_rows["venue_bin"]
            elif f == "fdr":
                Xgc[f] = pd.to_numeric(team_rows["fdr"], errors="coerce")
            else:
                Xgc[f] = np.nan
        Xgc = Xgc.fillna(0.0)
        exp_gc_team = np.clip(gc_model.predict(Xgc), 0.0, None)
        team_rows["exp_gc_team"] = exp_gc_team
    else:
        team_rows["exp_gc_team"] = np.nan

    # --- Player last snapshot for DCP features (leak-free) ---
    pull_cols = [c for c in dcp_feats if c not in ("venue_bin","fdr",
                                                   "team_possession_venue","opp_att_z_venue",
                                                   "team_def_xga_venue","team_def_xga_venue_z","team_att_z_venue")]
    pf_hist = pf[(pf["season"].isin(history)) |
                 ((pf["season"] == args.future_season) & (_coerce_ts(pf["date_played"], tz) < as_of_ts))].copy()
    last = _last_snapshot_per_player(pf_hist, feature_cols=pull_cols, as_of_ts=as_of_ts, tz=tz)

    # --- Build per-player future frame ---
    fut = minutes.copy()
    fut = fut.merge(
        team_rows.rename(columns={"gw_orig": gw_key_m})[["season", gw_key_m, "team_id", "p_teamCS", "exp_gc_team",
                                                         "team_def_xga_venue","team_def_xga_venue_z",
                                                         "team_possession_venue","team_att_z_venue","opp_att_z_venue"]],
        on=["season", gw_key_m, "team_id"], how="left", validate="many_to_one"
    )
    fut = fut.merge(last, on=["season","player_id"], how="left", validate="many_to_one")

    if fut.columns.duplicated().any():
        dups = fut.columns[fut.columns.duplicated()].tolist()
        logging.warning("Future frame had duplicate columns; keeping last occurrence: %s", set(dups))
        fut = fut.loc[:, ~fut.columns.duplicated(keep="last")]

    # --- Legacy metadata attach from fixtures: game_id/team/opponent_id/opponent ---
    fix_keep = ["season","team_id",gw_key_t,"fbref_id","team","opponent_id","is_home","home","away"]
    fix_keep = [c for c in fix_keep if c in team_fix.columns]
    fix_small = (team_fix[fix_keep]
                 .dropna(subset=[gw_key_t,"team_id"])
                 .drop_duplicates()
                 .rename(columns={gw_key_t: gw_key_m}))
    fut = fut.merge(
        fix_small,
        how="left",
        on=["season","team_id", gw_key_m],
        validate="many_to_one",
        suffixes=("", "_fix")
    )
    if "opponent" not in fut.columns or fut["opponent"].isna().all():
        if {"home","away","is_home"}.issubset(fut.columns):
            ih = pd.to_numeric(fut["is_home"], errors="coerce").fillna(0).astype("Int8")
            fut["opponent"] = np.where(ih == 1, fut.get("away"), fut.get("home"))

    # --- p60: prefer columns; else fallback from E[min] ---
    if "prob_played60_use" not in fut.columns:
        p60_col = None
        for c in ("prob_played60_cal","prob_played60_raw","prob_played60","p60"):
            if c in fut.columns:
                p60_col = c; break
        if p60_col is None:
            fut["prob_played60_use"] = np.clip((pd.to_numeric(fut["pred_minutes"], errors="coerce") - 30.0) / 60.0, 0.0, 1.0)
        else:
            fut["prob_played60_use"] = pd.to_numeric(fut[p60_col], errors="coerce")
            miss = fut["prob_played60_use"].isna()
            if miss.any():
                fut.loc[miss, "prob_played60_use"] = np.clip((pd.to_numeric(fut.loc[miss,"pred_minutes"], errors="coerce") - 30.0) / 60.0, 0.0, 1.0)

    # Player CS probability
    fut["prob_cs"] = pd.to_numeric(fut["p_teamCS"], errors="coerce") * pd.to_numeric(fut["prob_played60_use"], errors="coerce")

    # --- DCP per-90 features in exact artifact order ---
    Xdcp = pd.DataFrame(index=fut.index)
    for f in dcp_feats:
        if f in fut.columns:
            Xdcp[f] = pd.to_numeric(fut[f], errors="coerce")
        elif f == "venue_bin":
            Xdcp[f] = fut["venue_bin"]
        elif f == "fdr":
            Xdcp[f] = pd.to_numeric(fut["fdr"], errors="coerce")
        else:
            Xdcp[f] = np.nan
    Xdcp = Xdcp.fillna(0.0)

    # Predict per-90 λ for DEF/MID/FWD (GK optional)
    pos = fut["pos"].astype(str).str.upper().to_numpy()
    lam90 = np.full(len(fut), np.nan, dtype=float)
    for tag, mdl in dcp_models.items():
        mask = (pos == tag)
        if mask.any():
            lam90[mask] = np.clip(mdl.predict(Xdcp.loc[mask]), 0, None)
    if not args.include_gk_dcp:
        lam90[(pos == "GK")] = np.nan

    # --- DCP scaling & probabilities (mixture-aware if available) ---
    m_pred = np.clip(pd.to_numeric(fut["pred_minutes"], errors="coerce").fillna(0).to_numpy(), 0, None)
    k90_map = _parse_k90(args.dcp_k90)

    def _k_from_minutes(pos_arr, mins_arr):
        return np.array([
            int(np.ceil(k90_map.get(p, 12) * (mm / 90.0))) if p in ("DEF","MID","FWD") else 9_999
            for p, mm in zip(pos_arr, mins_arr)
        ], dtype=int)

    have_mix = all(c in fut.columns for c in ["p_start","p_cameo","pred_start_head","pred_bench_cameo_head"])
    if have_mix:
        ps = pd.to_numeric(fut["p_start"], errors="coerce").fillna(0).clip(0,1).to_numpy()
        pc = pd.to_numeric(fut["p_cameo"], errors="coerce").fillna(0).clip(0,1).to_numpy()
        ms = np.clip(pd.to_numeric(fut["pred_start_head"], errors="coerce").fillna(0).to_numpy(), 0, None)
        mb = np.clip(pd.to_numeric(fut["pred_bench_cameo_head"], errors="coerce").fillna(0).to_numpy(), 0, None)

        lam_s = lam90 * (ms / 90.0); lam_b = lam90 * (mb / 90.0)
        k_s = _k_from_minutes(pos, ms); k_b = _k_from_minutes(pos, mb)
        p_s = _poisson_tail_prob_vec(np.where(np.isfinite(lam_s), lam_s, 0.0), k_s)
        p_b = _poisson_tail_prob_vec(np.where(np.isfinite(lam_b), lam_b, 0.0), k_b)

        prob_dcp = ps * p_s + (1.0 - ps) * pc * p_b
        expected_dc = ps * lam_s + (1.0 - ps) * pc * lam_b
        lam_match = ps * lam_s + (1.0 - ps) * pc * lam_b
    else:
        lam_match = lam90 * (m_pred / 90.0)
        k_match = _k_from_minutes(pos, m_pred)
        prob_dcp = _poisson_tail_prob_vec(np.where(np.isfinite(lam_match), lam_match, 0.0), k_match)
        expected_dc = lam_match

    if not args.include_gk_dcp:
        gk_mask = (pos == "GK")
        prob_dcp[gk_mask] = np.nan
        expected_dc[gk_mask] = np.nan
        lam_match[gk_mask] = np.nan

    # ---------------------- Assemble output (legacy metadata first) ----------------------
    def pick_col(df: pd.DataFrame, *names: str):
        for n in names:
            if n in df.columns:
                return df[n].values
        return np.full(len(df), np.nan, dtype=object)

    opponent_id_vals    = pick_col(fut, "opponent_id", "opp_team_id", "opp_id")
    opponent_name_vals  = pick_col(fut, "opponent", "opp_team", "opp_name")
    game_id_vals        = pick_col(fut, "fbref_id", "fixture_id", "game_id")
    team_name_vals      = pick_col(fut, "team", "team_name", "team_short", "team_long")

    gw_played_vals = pd.to_numeric(fut.get("gw_played", np.nan), errors="coerce").values
    gw_orig_vals   = pd.to_numeric(fut.get(gw_key_m, np.nan), errors="coerce").values

    out = pd.DataFrame({
        # --- Legacy/fixture metadata (match minutes & GA) ---
        "season":            fut["season"].values,
        "gw_played":         gw_played_vals,
        "gw_orig":           gw_orig_vals,
        "date_sched":        pd.to_datetime(fut["date_sched"], errors="coerce").dt.normalize().values,
        "game_id":           game_id_vals,
        "team_id":           fut.get("team_id", pd.Series([np.nan]*len(fut))).astype(str).values,
        "team":              team_name_vals,
        "opponent_id":       opponent_id_vals,
        "opponent":          opponent_name_vals,
        "is_home":           pd.to_numeric(fut.get("is_home", np.nan), errors="coerce").fillna(0).astype("Int8").values,

        # --- Player context ---
        "player_id":         fut["player_id"].astype(str).values,
        "player":            fut.get("player", pd.Series([np.nan]*len(fut))).values,
        "pos":               fut["pos"].astype(str).values,
        "fdr":               pd.to_numeric(fut.get("fdr", np.nan), errors="raise").astype("Int8").values,
        "venue_bin":         pd.to_numeric(fut.get("venue_bin", np.nan), errors="coerce").values,

        # --- Team defense heads & player CS ---
        "p_teamCS":          pd.to_numeric(fut["p_teamCS"], errors="coerce").clip(0,1).values,
        "prob_cs":           pd.to_numeric(fut["prob_cs"], errors="coerce").clip(0,1).values,

        # --- DCP heads ---
        "lambda90":          lam90,
        "expected_dc":       expected_dc,
        "prob_dcp":          prob_dcp,

        # --- Team exp GC repeated per player ---
        "exp_gc":            pd.to_numeric(fut.get("exp_gc_team", np.nan), errors="coerce").values,
    })


    desired_order = [
        "season","gw_played","gw_orig","date_sched","game_id",
        "team_id","team","opponent_id","opponent","is_home",
        "player_id","player","pos","fdr","venue_bin",
        "p_teamCS","prob_cs",
        "lambda90","expected_dc","prob_dcp",
        "exp_gc"
    ]
    keep_cols = [c for c in desired_order if c in out.columns]
    out = out[keep_cols].copy()

        # Enforce integer dtypes (nullable ints so NA is preserved)
    if "gw_played" in out.columns:
        out["gw_played"] = pd.to_numeric(out["gw_played"], errors="coerce").astype("Int16")
    if "gw_orig" in out.columns:
        out["gw_orig"] = pd.to_numeric(out["gw_orig"], errors="coerce").astype("Int16")
    if "fdr" in out.columns:
        out["fdr"] = pd.to_numeric(out["fdr"], errors="coerce").astype("Int8")
    if "is_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").astype("Int8")
    if "venue_bin" in out.columns:
        out["venue_bin"] = pd.to_numeric(out["venue_bin"], errors="coerce").astype("Int8")

    sort_keys = [k for k in ["gw_orig","team_id","player_id"] if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    # … just before computing out_paths / writing:
    DEF_SCHEMA = {
    "required": ["season","gw_played","gw_orig","date_sched","game_id","team_id","team",
                "opponent_id","opponent","is_home","player_id","player","pos","fdr",
                "venue_bin","p_teamCS","prob_cs","lambda90","expected_dc","prob_dcp","exp_gc"],
    "dtypes": {
        "season":"string","gw_played":"Int64","gw_orig":"Int64","date_sched":"datetime64[ns]",
        "game_id":"object","team_id":"string","team":"object","opponent_id":"object","opponent":"object",
        "is_home":"Int64","player_id":"string","player":"object","pos":"string","fdr":"Int64","venue_bin":"Int64",
        "p_teamCS":"float","prob_cs":"float","lambda90":"float","expected_dc":"float","prob_dcp":"float","exp_gc":"float",
    },
    "na": {"gw_orig": False, "fdr": False, "is_home": False, "venue_bin": False},
    "ranges": {
        "p_teamCS":{"min":0.0,"max":1.0},
        "prob_cs":{"min":0.0,"max":1.0},
        "prob_dcp":{"min":0.0,"max":1.0},
        "lambda90":{"min":0.0}, "expected_dc":{"min":0.0}, "exp_gc":{"min":0.0},
    },
    "choices": {"pos":{"in":["GK","DEF","MID","FWD"]}},
    "logic": [("venue_bin in {0,1}", ["venue_bin"]), ("is_home in {0,1}", ["is_home"])],
    "date_rules": {"normalize":["date_sched"]},
    "unique": ["season","gw_orig","team_id","player_id"]
    }

    validate_df(out, DEF_SCHEMA, name="defense_forecast")

    gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
    out_paths = _def_out_paths(
        base_dir=args.out_dir,
        season=args.future_season,
        gw_from=gw_from_eff,
        gw_to=gw_to_eff,
        zero_pad=args.zero_pad_filenames,
        out_format=args.out_format,
    )
    written_paths = _write_def(out, out_paths)

    # >>> NEW: update consolidated season-level files
    season_dir.mkdir(parents=True, exist_ok=True)
    consolidated_paths = _update_consolidated(
        out_df=out,
        season_dir=season_dir,
        out_format=args.out_format,
        desired_order=desired_order,
    )

    diag = {
        "rows": int(len(out)),
        "season": str(args.future_season),
        "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
        "available_gws": [int(x) for x in avail_gws],
        "scored_gws": [int(x) for x in target_gws],
        "as_of": str(as_of_ts),
        "minutes_in": str(minutes_path),
        "out": written_paths,
        "consolidated_out": consolidated_paths,
        "has_iso_cs": bool(iso_cs is not None),
        "has_gc": bool(gc_model is not None),
        "dcp_models": list(dcp_models.keys()),
        "mixture_used": bool(have_mix),
        "roster_gate": {
            "enabled": bool(args.teams_json),
            "league_filter": args.league_filter or None,
            "require_on_roster": bool(args.require_on_roster),
        }
    }
    print(json.dumps(diag, indent=2))

if __name__ == "__main__":
    main()
