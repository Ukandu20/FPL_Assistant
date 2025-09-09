#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
points_forecast.py — upcoming fixtures — v1.5

What's new vs v1.4
------------------
• Expose DCP signals (prob_dcp, expected_dc) for all OUTFIELD positions (DEF/MID/FWD); GKs → NaN.
• Add DCP bonus into xPts: +2 * prob_dcp_outfield, with thresholds 10 (DEF) / 12 (MID,FWD).
• Rename internal/team defense columns for readability in OUTPUT:
    __p_cs__      → team_prob_cs
    __lambda90__  → team_ga_lambda90
    __exp_gc__    → team_exp_gc
• Auto-maintained merged CSV with overwrite-by-key semantics.
• date_sched placed immediately after season in the header.
"""

from __future__ import annotations
import argparse, logging, json, os, re, datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

SCHEMA_VERSION = "future.v1.5"

# ───────────────────────── versioning & pointers ─────────────────────────

def _resolve_version(base_dir: Path, requested: Optional[str], auto: bool) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    if auto or not requested or requested.lower() == "auto":
        existing = [p.name for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        nxt = (max(int(s[1:]) for s in existing) + 1) if existing else 1
        ver = f"v{nxt}"
        logging.info("[version] auto -> %s", ver)
        return ver
    if re.fullmatch(r"v\d+", requested):
        return requested
    if requested.isdigit():
        return f"v{requested}"
    raise ValueError("--version must look like v3 (or use --auto-version)")

def _write_latest_pointer(root: Path, version: str) -> None:
    latest = root / "latest"
    target = root / version
    try:
        if latest.exists() or latest.is_symlink():
            try:
                latest.unlink()
            except Exception:
                pass
        os.symlink(target.name, latest, target_is_directory=True)
        logging.info("[latest] symlink -> %s", version)
    except (OSError, NotImplementedError):
        (root / "LATEST_VERSION.txt").write_text(version, encoding="utf-8")
        logging.info("[latest] wrote LATEST_VERSION.txt -> %s", version)

# ───────────────────────── IO & normalization ─────────────────────────

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    for c in ("date_sched", "date_played", "kickoff_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    if "gw_orig" in df.columns:
        df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")
    for c in ("player_id", "team_id"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def _read_csv(path: Path, need: List[str] | None = None, tag: str = "") -> pd.DataFrame:
    try:
        head = pd.read_csv(path, nrows=0)
        parse_dates = [c for c in ["date_sched", "date_played", "kickoff_time"] if c in head.columns]
        df = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)
    df = _norm_cols(df)
    if need:
        miss = [c for c in need if c not in df.columns]
        if miss:
            logging.warning("[%s] missing expected columns: %s", tag or path.name, miss)
    return df

def _coverage(df: pd.DataFrame, cols: List[str], tag: str) -> None:
    have = [c for c in cols if c in df.columns]
    if not have:
        logging.info("[%s] coverage: (none found)", tag)
        return
    stats = {c: float(df[c].notna().mean()) for c in have}
    logging.info("[%s] coverage: %s", tag, ", ".join(f"{k}={v:.1%}" for k, v in stats.items()))

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# ───────────────────────── math helpers ─────────────────────────

def _round_for_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(1 if c == "xPts" else 2)
    return out

def _pos_points_vector(pos: pd.Series, kind: str) -> np.ndarray:
    p = pos.fillna("").astype(str).str.upper().str[:3]
    if kind == "goal":
        return np.where(p.isin(["GKP","GK","DEF"]), 6,
                        np.where(p.isin(["MID"]), 5,
                                 np.where(p.isin(["FWD"]), 4, 0.0))).astype(float)
    if kind == "cs":
        return np.where(p.isin(["GKP","GK","DEF"]), 4,
                        np.where(p.isin(["MID"]), 1, 0.0)).astype(float)
    raise ValueError("kind must be 'goal' or 'cs'")

def _is_gk(pos: pd.Series) -> np.ndarray:
    p = pos.fillna("").astype(str).str.upper().str[:3]
    return p.isin(["GK","GKP"]).to_numpy()

def _is_def(pos: pd.Series) -> np.ndarray:
    p = pos.fillna("").astype(str).str.upper().str[:3]
    return p.eq("DEF").to_numpy()

def _is_outfield(pos: pd.Series) -> np.ndarray:
    p = pos.fillna("").astype(str).str.upper().str[:3]
    return ~p.isin(["GK","GKP"]).to_numpy()

def _choose_first_num(df: pd.DataFrame, candidates: List[str], default: float | None = None) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")

def _select_prob_series(df: pd.DataFrame, candidates: List[str], default: float = np.nan) -> pd.Series:
    s = _choose_first_num(df, candidates, default=default)
    return s.astype(float).clip(0.0, 1.0)

def _expected_gc_pairs_lambda_on(lam_on: np.ndarray) -> np.ndarray:
    lam = np.clip(lam_on.astype(float), 0.0, None)
    return 0.5 * lam - 0.25 * (1.0 - np.exp(-2.0 * lam))

def _expected_floor_div3_poisson(lam: np.ndarray, kmax: int | None = None) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    lam = np.where(np.isfinite(lam) & (lam >= 0.0), lam, 0.0)
    if lam.size == 0:
        return np.array([], dtype=float)
    if kmax is None:
        lam_max = float(lam.max(initial=0.0))
        if lam_max == 0.0:
            return np.zeros_like(lam)
        kmax = int(np.clip(np.ceil(lam_max + 8.0 * np.sqrt(max(lam_max, 1e-9)) + 30.0), 40, 120))
    out = np.zeros_like(lam)
    pmf = np.exp(-lam)
    for i in range(1, kmax + 1):
        pmf = pmf * lam / i
        out += (i // 3) * pmf
    return out

def _poisson_sf_vectorized(lam: np.ndarray, k_threshold: int) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    lam = np.where(np.isfinite(lam) & (lam >= 0.0), lam, 0.0)
    pmf0 = np.exp(-lam)
    cdf = pmf0.copy()
    pmf = pmf0.copy()
    for i in range(1, k_threshold):
        pmf = pmf * lam / i
        cdf += pmf
    sf = 1.0 - cdf
    return np.clip(sf, 0.0, 1.0)

# ───────────────────────── KEY & merge helpers ─────────────────────────

FORCED_KEY = ["season", "gw_orig", "player_id", "team_id"]

LEFT_DATE_PREFS  = ("date_sched", "date_played", "kickoff_time")
RIGHT_DATE_PREFS = ("date_sched", "date_played", "kickoff_time")

def _pick_first_date_col(df: pd.DataFrame, prefs: tuple[str, ...]) -> str | None:
    for c in prefs:
        if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def _drop_exact_dupes(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates()
    d = before - len(out)
    if d:
        logging.warning("[%s] dropped %d exact duplicate rows", tag, d)
    return out

def _merge_with_nearest_date(base: pd.DataFrame,
                             right: pd.DataFrame | None,
                             key: list[str],
                             keep_cols: list[str],
                             tag: str) -> pd.DataFrame:
    if right is None or not len(right):
        return base
    ldate = _pick_first_date_col(base, LEFT_DATE_PREFS)
    rdate = _pick_first_date_col(right, RIGHT_DATE_PREFS)
    rkeep = [c for c in keep_cols if c in right.columns]
    cols = list(dict.fromkeys([*key, *rkeep, *( [rdate] if rdate else [] )]))
    rsub = right[cols].copy()
    if rdate:
        rsub = rsub.rename(columns={rdate: "__r_date__"})
    base = base.copy()
    base["__rowid__"] = np.arange(len(base))
    m = base.merge(rsub, on=key, how="left", suffixes=("", "_r"))
    if "__r_date__" not in m.columns or ldate is None or ldate not in m.columns:
        if m["__rowid__"].duplicated().any():
            before = len(m)
            m = m.drop_duplicates(subset="__rowid__", keep="last")
            logging.warning("[%s] duplicates after merge but no date tie-break; kept last (dropped %d)", tag, before - len(m))
        return m.drop(columns="__rowid__")
    dup_counts = m["__rowid__"].value_counts()
    if int((dup_counts > 1).sum()):
        logging.info("[%s] duplicate 4-col matches; resolving via nearest-date", tag)
    m["__delta__"] = (m["__r_date__"] - m[ldate]).abs()
    m = m.sort_values(["__rowid__", "__delta__"]).drop_duplicates(subset="__rowid__", keep="first")
    drop_cols = ["__rowid__", "__delta__", "__r_date__"] if "__r_date__" in m.columns else ["__rowid__", "__delta__"]
    return m.drop(columns=drop_cols)

# ───────────────────────── roster gate helpers ─────────────────────────

def _build_roster_keyset(teams_json: Optional[Path],
                         seasons: List[str],
                         league_filter: Optional[str]) -> Optional[Set[str]]:
    if teams_json is None:
        return None
    if not teams_json.exists():
        raise FileNotFoundError(f"--teams-json points to a missing file: {teams_json}")
    raw = json.loads(teams_json.read_text(encoding="utf-8"))

    def norm(x: object) -> str:
        return str(x).strip().lower()

    seasons_norm = {norm(s) for s in seasons}
    allow: Set[str] = set()
    for team_id, blob in (raw or {}).items():
        career = (blob or {}).get("career", {})
        for season, sblob in career.items():
            if norm(season) not in seasons_norm:
                continue
            if league_filter:
                if str(sblob.get("league", "")).strip() != league_filter:
                    continue
            for p in (sblob.get("players", []) or []):
                pid = p.get("id")
                if pid:
                    allow.add(f"{norm(season)}|{norm(team_id)}|{norm(pid)}")
    return allow

def _apply_roster_gate(base: pd.DataFrame,
                       allow_keys: Optional[Set[str]],
                       artifacts_dir: Path,
                       require_on_roster: bool) -> pd.DataFrame:
    if allow_keys is None:
        return base
    need = {"season", "team_id", "player_id"}
    if not need.issubset(base.columns):
        logging.warning("[roster] missing columns for roster gate (%s); skipping.", need - set(base.columns))
        return base

    def norm_s(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower()

    k = norm_s(base["season"]) + "|" + norm_s(base["team_id"]) + "|" + norm_s(base["player_id"])
    keep = k.isin(list(allow_keys))
    dropped = int((~keep).sum())
    if dropped:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        outp = artifacts_dir / "roster_dropped_points.csv"
        cols = [c for c in ["season","gw_orig","player_id","player","team_id","pos","date_sched","date_played","kickoff_time"] if c in base.columns]
        try:
            base.loc[~keep, cols].to_csv(outp, index=False)
        except Exception:
            pass
        logging.info("Roster gate dropped %d rows not present on the provided season roster(s).", dropped)
        if require_on_roster:
            raise RuntimeError(f"--require-on-roster set: {dropped} rows are not on the roster.")
    return base.loc[keep].copy()

# ───────────────────────── merged writer ─────────────────────────

def _align_union_columns(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    all_cols = list(dict.fromkeys([*a.columns, *b.columns]))
    a2 = a.reindex(columns=all_cols)
    b2 = b.reindex(columns=all_cols)
    if "season" in all_cols and "date_sched" in all_cols:
        all_cols.remove("date_sched")
        all_cols.insert(all_cols.index("season") + 1, "date_sched")
    return a2[all_cols], b2[all_cols], all_cols

def _update_merged_csv(new_df: pd.DataFrame,
                       merged_path: Path,
                       key_cols: List[str],
                       sort_cols: List[str]) -> None:
    if merged_path.exists():
        old = _read_csv(merged_path, tag="MERGED")
    else:
        old = pd.DataFrame(columns=new_df.columns)
    old, new, all_cols = _align_union_columns(old, new_df)
    key_df = new[key_cols].drop_duplicates()
    old_mark = old.merge(key_df.assign(__hit__=1), on=key_cols, how="left")
    old_kept = old_mark[old_mark["__hit__"].isna()].drop(columns="__hit__")
    merged = pd.concat([old_kept, new], ignore_index=True, sort=False)
    for c in sort_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged.sort_values(by=[c for c in sort_cols if c in merged.columns])
    merged = _round_for_output(merged)
    _atomic_write_csv(merged, merged_path)
    logging.info("[merged] updated %s (rows=%d)", merged_path.resolve(), len(merged))

# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Forecast expected FPL points for upcoming fixtures.")
    ap.add_argument("--minutes", required=True, type=Path)
    ap.add_argument("--goals-assists", required=True, type=Path)
    ap.add_argument("--saves", type=Path, default=None)
    ap.add_argument("--defense", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--version", type=str, default=None)
    ap.add_argument("--auto-version", action="store_true")
    ap.add_argument("--write-latest", action="store_true")
    ap.add_argument("--discipline-priors", type=str, default=None,
                    help='Override per-90 priors, e.g. "GK:-0.05,DEF:-0.20,MID:-0.15,FWD:-0.10"')
    ap.add_argument("--teams-json", type=Path, default=None)
    ap.add_argument("--league-filter", type=str, default=None)
    ap.add_argument("--require-on-roster", action="store_true")
    ap.add_argument("--no-merged", action="store_true")
    ap.add_argument("--log-level", default="INFO", type=str)
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s"
    )

    version = _resolve_version(args.out_dir, args.version, args.auto_version)
    out_dir = args.out_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"

    # Read inputs
    min_need = ["season","player_id","team_id","pos","gw_orig","pred_minutes"]
    min_df = _read_csv(args.minutes, min_need, tag="MIN")
    _coverage(min_df, ["season","gw_orig","player_id","team_id","p_play","p60","pred_minutes","exp_minutes_points","pos","player","date_sched","date_played","kickoff_time"], "MIN")

    ga_need = ["season","player_id","team_id","gw_orig"]
    ga_df = _read_csv(args.goals_assists, ga_need, tag="GA")
    _coverage(ga_df, ["season","gw_orig","player_id","team_id","pred_goals_mean","pred_assists_mean","player","pos","date_sched","date_played","kickoff_time"], "GA")

    def_df = None
    if args.defense:
        def_df = _read_csv(args.defense, tag="DEF")
        _coverage(def_df, [
            "season","gw_orig","player_id","team_id",
            "prob_cs","p_teamCS","lambda90","exp_gc",
            "prob_dcp","p_dcp","dcp_prob",
            "lambda90_dcp","dcp_lambda90","dcp_p90","lambda_dcp90",
            "expected_dc",
            "date_sched","date_played","kickoff_time"
        ], "DEF")

    sav_df = None
    if args.saves:
        sav_df = _read_csv(args.saves, tag="SAV")
        _coverage(sav_df, [
            "season","gw_orig","player_id","team_id","pred_saves_mean","pred_saves_poisson",
            "pred_saves_p90_mean","pred_saves_p90_poisson","pred_minutes","player","pos","date_sched","date_played","kickoff_time"
        ], "SAV")

    KEY = FORCED_KEY
    logging.info("[key] FORCED -> %s (date columns not used in key)", KEY)

    # De-dupe
    min_df = _drop_exact_dupes(min_df, "MIN")
    ga_df  = _drop_exact_dupes(ga_df,  "GA")
    if def_df is not None: def_df = _drop_exact_dupes(def_df, "DEF")
    if sav_df is not None: sav_df = _drop_exact_dupes(sav_df, "SAV")

    # Base
    if not set(KEY).issubset(min_df.columns):
        raise ValueError(f"MINUTES CSV missing key columns for required KEY: {KEY}")
    base = min_df.copy()

    # Roster gate
    seasons_in_minutes = sorted(base["season"].dropna().astype(str).unique().tolist())
    allow_keys = _build_roster_keyset(args.teams_json, seasons_in_minutes, args.league_filter)
    base = _apply_roster_gate(base, allow_keys, artifacts_dir, args.require_on_roster)
    if base.empty:
        raise RuntimeError("No rows remain after roster gating.")

    # Fill identity from GA if needed
    for c in ("player","pos"):
        if c not in base.columns and ga_df is not None and c in ga_df.columns:
            right = ga_df[KEY + [c] + [col for col in RIGHT_DATE_PREFS if col in ga_df.columns]]
            base = _merge_with_nearest_date(base, right, KEY, [c], tag=f"GA:{c}")
    for c in ("player","pos"):
        if c not in base.columns:
            base[c] = np.nan

    # GA expectations
    if ga_df is not None and len(ga_df):
        keep = [col for col in ["pred_goals_mean","pred_assists_mean"] if col in ga_df.columns]
        if keep:
            right = ga_df[KEY + keep + [col for col in RIGHT_DATE_PREFS if col in ga_df.columns]]
            base = _merge_with_nearest_date(base, right, KEY, keep, tag="GA:means")
    base["pred_goals_mean"]   = pd.to_numeric(base.get("pred_goals_mean", 0.0), errors="coerce").fillna(0.0)
    base["pred_assists_mean"] = pd.to_numeric(base.get("pred_assists_mean", 0.0), errors="coerce").fillna(0.0)

    # Defense signals (CS/GA + DCP)
    if def_df is not None and len(def_df):
        tmp = def_df.copy()
        tmp["__p_cs__"]         = _select_prob_series(tmp, ["prob_cs","p_teamCS"])
        tmp["__lambda90__"]     = pd.to_numeric(tmp.get("lambda90", np.nan), errors="coerce")
        tmp["__exp_gc__"]       = pd.to_numeric(tmp.get("exp_gc",   np.nan), errors="coerce")

        # DCP pieces (for all rows; will mask GKs later)
        tmp["__prob_dcp__"]     = _select_prob_series(tmp, ["prob_dcp","p_dcp","dcp_prob"])
        tmp["__lambda90_dcp__"] = _choose_first_num(tmp, ["lambda90_dcp","dcp_lambda90","dcp_p90","lambda_dcp90"], default=np.nan)
        if "expected_dc" in tmp.columns:
            tmp["__expected_dc__"] = pd.to_numeric(tmp["expected_dc"], errors="coerce")
        else:
            tmp["__expected_dc__"] = np.nan

        keep = ["__p_cs__","__lambda90__","__exp_gc__","__prob_dcp__","__lambda90_dcp__","__expected_dc__"]
        right = tmp[KEY + keep + [col for col in RIGHT_DATE_PREFS if col in tmp.columns]]
        base = _merge_with_nearest_date(base, right, KEY, keep, tag="DEF")
    else:
        base["__p_cs__"] = np.nan
        base["__lambda90__"] = np.nan
        base["__exp_gc__"] = np.nan
        base["__prob_dcp__"] = np.nan
        base["__lambda90_dcp__"] = np.nan
        base["__expected_dc__"] = np.nan

    # Saves (GK) — prefer per-match λ; else per-90 scaled by minutes
    if sav_df is not None and len(sav_df):
        tmp = sav_df.copy()
        lam_pm  = _choose_first_num(tmp, ["pred_saves_mean","pred_saves_poisson"], default=np.nan)
        lam_p90 = _choose_first_num(tmp, ["pred_saves_p90_mean","pred_saves_p90_poisson"], default=np.nan)
        tmp["__saves_lambda_raw__"] = lam_pm.where(lam_pm.notna(), lam_p90)
        keep = ["__saves_lambda_raw__"]
        right = tmp[KEY + keep + [col for col in RIGHT_DATE_PREFS if col in tmp.columns]]
        base = _merge_with_nearest_date(base, right, KEY, keep, tag="SAV")
    else:
        base["__saves_lambda_raw__"] = np.nan

    # Minutes & appearance
    m = pd.to_numeric(base.get("pred_minutes", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    base["pred_minutes"] = m
    base["p1"]  = pd.to_numeric(base.get("p_play", np.nan), errors="coerce")
    base["p60"] = pd.to_numeric(base.get("p60",     np.nan), errors="coerce")
    base["p1"]  = base["p1"].where(base["p1"].notna(), (m > 0).astype(float))
    base["p60"] = base["p60"].where(base["p60"].notna(), np.clip(m/90.0, 0.0, 1.0))
    base["xp_appearance"] = pd.to_numeric(base.get("exp_minutes_points", np.nan), errors="coerce")
    base["xp_appearance"] = base["xp_appearance"].where(base["xp_appearance"].notna(), base["p1"] + base["p60"])

    # Goals & assists EP
    goal_pts_vec = _pos_points_vector(base["pos"], kind="goal")
    base["xp_goals"]   = base["pred_goals_mean"]   * goal_pts_vec
    base["xp_assists"] = base["pred_assists_mean"] * 3.0

    # Clean sheet EP (≥60)
    cs_pts_vec = _pos_points_vector(base["pos"], kind="cs")
    p_cs = pd.to_numeric(base.get("__p_cs__", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    base["xp_clean_sheets"] = base["p60"] * p_cs * cs_pts_vec

    # Goals-conceded penalty (GK/DEF-like)
    lam90   = pd.to_numeric(base.get("__lambda90__", np.nan), errors="coerce")
    exp_gc  = pd.to_numeric(base.get("__exp_gc__",   np.nan), errors="coerce")
    lam_team = lam90.where(lam90.notna(), exp_gc)
    lam_team = lam_team.where(
        lam_team.notna(),
        (-np.log(np.clip(p_cs.replace(0, np.nan), 1e-6, 1 - 1e-6))).replace([np.inf, -np.inf], np.nan)
    )
    lam_on = lam_team.fillna(0.0) * (m / 90.0)
    e_pairs = _expected_gc_pairs_lambda_on(lam_on.to_numpy())
    is_def_like_mask = base["pos"].fillna("").astype(str).str.upper().str[:3].isin(["GKP","GK","DEF"]).to_numpy()
    base["xp_concede_penalty"] = np.where(is_def_like_mask, -e_pairs, 0.0)

    # Saves EP (GK only)
    raw_lambda = pd.to_numeric(base.get("__saves_lambda_raw__"), errors="coerce")
    had_pm = bool(sav_df is not None and (
        ("pred_saves_mean" in sav_df.columns and pd.to_numeric(sav_df["pred_saves_mean"], errors="coerce").notna().any()) or
        ("pred_saves_poisson" in sav_df.columns and pd.to_numeric(sav_df["pred_saves_poisson"], errors="coerce").notna().any())
    ))
    lam_saves = raw_lambda if had_pm else raw_lambda * np.clip(m/90.0, 0.0, None)
    lam_saves = lam_saves.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    exp_floor_s3 = _expected_floor_div3_poisson(lam_saves.to_numpy(), kmax=None)
    base["xp_saves_points"] = np.where(_is_gk(base["pos"]), exp_floor_s3, 0.0)

    # Discipline v1 prior (per-90 × minutes)
    priors = {"GK": -0.05, "GKP": -0.05, "DEF": -0.18, "MID": -0.14, "FWD": -0.10}
    pos_upper = base["pos"].fillna("").astype(str).str.upper()
    prior_per90 = pos_upper.map(lambda x: priors.get(x, priors.get(x[:3], -0.12))).astype(float).fillna(-0.12)
    base["xp_discipline_prior"] = prior_per90 * (m / 90.0)

    # DCP exposure for outfield & fallback prob if needed
    is_outfield = _is_outfield(base["pos"])
    is_def = _is_def(base["pos"])

    prob_dcp_in = pd.to_numeric(base.get("__prob_dcp__"), errors="coerce").clip(0.0, 1.0)
    lam90_dcp   = pd.to_numeric(base.get("__lambda90_dcp__"), errors="coerce")
    exp_dc_dir  = pd.to_numeric(base.get("__expected_dc__"), errors="coerce")

    expected_dc = exp_dc_dir.where(exp_dc_dir.notna(), lam90_dcp * (m / 90.0))

    # Thresholds: DEF=10, MID/FWD=12
    lam_on_dcp = lam90_dcp * (m / 90.0)
    sf10 = _poisson_sf_vectorized(lam_on_dcp.to_numpy(), 10)
    sf12 = _poisson_sf_vectorized(lam_on_dcp.to_numpy(), 12)
    sf_fallback = np.where(is_def, sf10, sf12)

    prob_dcp_final = prob_dcp_in.where(prob_dcp_in.notna(), sf_fallback)
    prob_dcp_final = np.where(is_outfield, prob_dcp_final, np.nan)

    base["prob_dcp"]    = np.where(is_outfield, prob_dcp_final, np.nan)
    base["expected_dc"] = np.where(is_outfield, expected_dc, np.nan)

    # DCP bonus → xPts
    base["xp_dcp_bonus"] = np.where(is_outfield, 2.0 * prob_dcp_final, 0.0)

    # Sum components → xPts
    comp_cols = [
        "xp_appearance",
        "xp_goals",
        "xp_assists",
        "xp_clean_sheets",
        "xp_saves_points",
        "xp_discipline_prior",
        "xp_dcp_bonus",
    ]
    for c in comp_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    base["xPts"] = base[comp_cols].sum(axis=1)

    # Friendly output aliases for team defense signals
    base["team_prob_cs"]     = pd.to_numeric(base.get("__p_cs__"), errors="coerce")
    base["team_ga_lambda90"] = pd.to_numeric(base.get("__lambda90__"), errors="coerce")
    base["team_exp_gc"]      = pd.to_numeric(base.get("__exp_gc__"), errors="coerce")

    # Output columns (place date_sched right after season)
    out_cols = list(dict.fromkeys([
        *FORCED_KEY, "player", "pos",
        *[c for c in LEFT_DATE_PREFS if c in base.columns],
        "pred_minutes", "p1", "p60",
        "pred_goals_mean","pred_assists_mean",
        "team_prob_cs","team_ga_lambda90","team_exp_gc",
        "prob_dcp","expected_dc",
        "xp_appearance","xp_goals","xp_assists","xp_clean_sheets","xp_concede_penalty","xp_saves_points","xp_discipline_prior","xp_dcp_bonus",
        "xPts",
    ]))
    out_cols = [c for c in out_cols if c in base.columns]
    if "season" in out_cols and "date_sched" in out_cols:
        out_cols.remove("date_sched")
        out_cols.insert(out_cols.index("season") + 1, "date_sched")

    # Sort order
    sort_cols = [c for c in ["season", "gw_orig", "team_id", "player_id"] if c in base.columns]
    for dc in LEFT_DATE_PREFS:
        if dc in base.columns and dc not in sort_cols:
            sort_cols.append(dc)

    out = base[out_cols].copy().sort_values(by=sort_cols)
    out_rounded = _round_for_output(out)

    # Write per-run (versioned)
    fp = out_dir / "expected_points.csv"
    _atomic_write_csv(out_rounded, fp)
    logging.info("[write] upcoming expected points -> %s (rows=%d)", fp.resolve(), len(out_rounded))

    # Update merged unless suppressed
    if not args.no_merged:
        merged_fp = args.out_dir / "expected_points_merged.csv"
        _update_merged_csv(out_rounded, merged_fp, FORCED_KEY, sort_cols)

    # Logs
    nonzero_ga = int((base["pred_goals_mean"] > 0).sum())
    nonzero_as = int((base["pred_assists_mean"] > 0).sum())
    logging.info("[GA] non-zero means -> goals=%d, assists=%d", nonzero_ga, nonzero_as)

    meta = {
        "schema": SCHEMA_VERSION,
        "build_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "version": version,
        "key": FORCED_KEY,
        "inputs": {
            "minutes": str(args.minutes),
            "goals_assists": str(args.goals_assists),
            "saves": str(args.saves) if args.saves else None,
            "defense": str(args.defense) if args.defense else None,
        },
        "component_columns": comp_cols,
        "signals_used": {
            "appearance": ["exp_minutes_points", "p_play + p60 (fallback)"],
            "goals": ["pred_goals_mean"],
            "assists": ["pred_assists_mean"],
            "clean_sheet_prob": ["team_prob_cs ← (prob_cs | p_teamCS)"],
            "ga_lambda": ["team_ga_lambda90 ← lambda90 (per90)", "team_exp_gc ← exp_gc (per match)", "fallback: -log(prob_cs)"],
            "saves_points_formula": "Exact E[floor(S/3)] with per-match saves λ",
            "dcp": [
                "prob_dcp ← (prob_dcp|p_dcp|dcp_prob); fallback P(DC≥thr) via Poisson with λ_on=lambda90_dcp×min/90",
                "expected_dc ← defense.expected_dc or lambda90_dcp×min/90",
                "DCP bonus: +2 * prob_dcp_outfield; thr DEF=10, MID/FWD=12"
            ],
        },
        "output_fields": {"total": "xPts (1 dp)", "others": "2 dp"}
    }
    (out_dir / "expected_points.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.write_latest:
        _write_latest_pointer(args.out_dir, version)

if __name__ == "__main__":
    main()
