#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
points_forecast.py — upcoming fixtures — v1.8 (output rev, with FDR harmonization)

Output changes vs v1.7
----------------------
• No versioned run folders. Writes GW-windowed files mirroring defense_forecast style:
    <out-dir>/<FUTURE_SEASON>/GW{from}_{to}.csv|parquet   (zero-pad controlled by --zero-pad-filenames)
• Select format via --out-format {csv,parquet,both}.
• The cumulative stack file is now <out-dir>/expected_points.{csv,parquet} (was expected_points_merged.csv).
• Artifacts (e.g., roster drop audits) go under <out-dir>/<FUTURE_SEASON>/artifacts.

New in this revision
--------------------
• fdr (fixture difficulty rating) is included in the output metadata and harmonized across inputs:
  - Pulls fdr from GA, MIN, DEF, and SAV files when present.
  - Per-row conflict resolution: GA → MIN → DEF → SAV priority; if sources disagree, choose the mode; tie → max.
  - Stored as pandas nullable Int64, preserving missing values cleanly in CSV/Parquet.

Additional updates in this patch
--------------------------------
• Merged stack now mirrors --out-format:
  - If `csv` → writes expected_points.csv
  - If `parquet` → writes expected_points.parquet
  - If `both` → writes both, kept in sync.
• Louder warnings when DEF/SAV are omitted so users don’t accidentally run GA-only points.
"""

from __future__ import annotations
import argparse, logging, json, os, re, datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from scripts.utils.validate import validate_df
SCHEMA_VERSION = "future.v1.8"

# ───────────────────────── IO & normalization ─────────────────────────

LEFT_DATE_PREFS  = ("date_sched", "date_played", "kickoff_time")
RIGHT_DATE_PREFS = ("date_sched", "date_played", "kickoff_time")
FORCED_KEY = ["season", "gw_orig", "player_id", "team_id"]

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    for c in ("date_sched", "date_played", "kickoff_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    if "gw_orig" in df.columns:
        df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce")
    for c in ("player_id", "team_id"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def _read_any(path: Path, tag: str = "") -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            head = pd.read_csv(path, nrows=0)
            parse_dates = [c for c in LEFT_DATE_PREFS if c in head.columns]
            df = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
        except Exception:
            df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"[{tag}] unsupported extension: {path.suffix}")
    return _norm_cols(df)

def _read_csv(path: Path, need: List[str] | None = None, tag: str = "") -> pd.DataFrame:
    df = _read_any(path, tag=tag or path.name)
    if need:
        miss = [c for c in need if c not in df.columns]
        if miss:
            logging.warning("[%s] missing expected columns: %s", tag or path.name, miss)
    return df

def _coverage(df: pd.DataFrame, cols: List[str], tag: str) -> None:
    have = [c for c in cols if c in df.columns]
    if not have:
        logging.info("[%s] coverage: (none found)", tag); return
    stats = {c: float(df[c].notna().mean()) for c in have}
    logging.info("[%s] coverage: %s", tag, ", ".join(f"{k}={v:.1%}" for k, v in stats.items()))

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# ───────────────────────── GW window auto-resolve (inputs) ─────────────────────────

def _fmt_gw(n: int, zero_pad: bool) -> str:
    return f"{int(n):02d}" if zero_pad else f"{int(n)}"

def _candidate_paths(root: Path, season: str, gw_from: int, gw_to: int, zero_pad: bool) -> List[Path]:
    season_dir = root / str(season)
    a, b = _fmt_gw(gw_from, zero_pad), _fmt_gw(gw_to, zero_pad)
    return [season_dir / f"GW{a}_{b}.csv", season_dir / f"GW{a}_{b}.parquet"]

def _glob_fallback(root: Path, season: str, gw_from: int, gw_to: int) -> Optional[Path]:
    season_dir = root / str(season)
    if not season_dir.exists(): return None
    pats = [f"GW{gw_from}_*.csv", f"GW{gw_from}_*.parquet",
            f"GW{gw_from:02d}_*.csv", f"GW{gw_from:02d}_*.parquet"]
    for pat in pats:
        for p in sorted(season_dir.glob(pat)):
            try:
                to_str = p.stem.split("_")[-1].replace("GW", "")
                if int(to_str) == int(gw_to):
                    return p
            except Exception:
                continue
    return None

def _resolve_input_path(explicit: Optional[Path],
                        root: Optional[Path],
                        season: Optional[str],
                        gw_from: Optional[int],
                        gw_to: Optional[int],
                        zero_pad: bool,
                        tag: str) -> Path:
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(f"[{tag}] file not found: {explicit}")
        return explicit
    if root is None or season is None or gw_from is None or gw_to is None:
        raise ValueError(f"[{tag}] need --{tag}-root, --future-season, and GW window to auto-resolve.")
    for cand in _candidate_paths(root, season, gw_from, gw_to, zero_pad):
        if cand.exists(): return cand
    fb = _glob_fallback(root, season, gw_from, gw_to)
    if fb: return fb
    season_dir = root / str(season)
    tried = "\n".join(f"  - {p}" for p in _candidate_paths(root, season, gw_from, gw_to, zero_pad))
    raise FileNotFoundError(
        f"[{tag}] Could not locate GW{gw_from}_{gw_to} under {season_dir}\n"
        f"Tried:\n{tried}\nAlso tried glob fallback."
    )

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

def _lambda_from_p(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").clip(0.0, 1.0)
    eps = 1e-12
    return -np.log(np.clip(1.0 - p, eps, 1.0))

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
        try: base.loc[~keep, cols].to_csv(outp, index=False)
        except Exception: pass
        logging.info("Roster gate dropped %d rows not present on the provided season roster(s).", dropped)
        if require_on_roster:
            raise RuntimeError(f"--require-on-roster set: {dropped} rows are not on the roster.")
    return base.loc[keep].copy()

# ───────────────────────── merged writer (format-aware) ─────────────────────────

def _align_union_columns(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    all_cols = list(dict.fromkeys([*a.columns, *b.columns]))
    a2 = a.reindex(columns=all_cols)
    b2 = b.reindex(columns=all_cols)
    if "season" in all_cols and "date_sched" in all_cols:
        all_cols.remove("date_sched")
        all_cols.insert(all_cols.index("season") + 1, "date_sched")
    return a2[all_cols], b2[all_cols], all_cols

def _read_existing_merged(stem: Path) -> pd.DataFrame:
    """Try to read existing merged stack from <stem>.parquet then <stem>.csv; else empty."""
    pq = stem.with_suffix(".parquet")
    cs = stem.with_suffix(".csv")
    if pq.exists():
        try:
            return _read_any(pq, tag="MERGED-PQ")
        except Exception:
            logging.warning("[merged] failed to read parquet; will try csv.")
    if cs.exists():
        try:
            return _read_any(cs, tag="MERGED-CSV")
        except Exception:
            logging.warning("[merged] failed to read csv; starting fresh.")
    return pd.DataFrame()

def _update_merged(
    new_df: pd.DataFrame,
    merged_stem: Path,           # e.g., out_dir / "expected_points" (no extension)
    key_cols: List[str],
    sort_cols: List[str],
    formats: List[str],          # subset of ["csv","parquet"]
) -> List[str]:
    """
    Union-merge new_df into existing merged stack and write to every requested format.
    Returns list of written file paths.
    """
    old = _read_existing_merged(merged_stem)
    old, new, all_cols = _align_union_columns(old, new_df)

    key_df = new[key_cols].drop_duplicates()
    old_mark = old.merge(key_df.assign(__hit__=1), on=key_cols, how="left")
    old_kept = old_mark[old_mark["__hit__"].isna()].drop(columns="__hit__")
    merged = pd.concat([old_kept, new], ignore_index=True, sort=False)

    # ensure sort_cols exist
    for c in sort_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged.sort_values(by=[c for c in sort_cols if c in merged.columns])
    merged = _round_for_output(merged)

    written: List[str] = []
    # CSV writer with date formatting
    if "csv" in formats:
        out_csv = merged_stem.with_suffix(".csv")
        if "date_sched" in merged.columns and pd.api.types.is_datetime64_any_dtype(merged["date_sched"]):
            merged_csv = merged.copy()
            merged_csv["date_sched"] = pd.to_datetime(merged_csv["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_csv.with_suffix(".csv.tmp")
            merged_csv.to_csv(tmp, index=False)
            os.replace(tmp, out_csv)
        else:
            _atomic_write_csv(merged, out_csv)
        written.append(str(out_csv))
        logging.info("[merged] updated %s (rows=%d)", out_csv.resolve(), len(merged))

    # Parquet writer (preserves nullable integers, categoricals, etc.)
    if "parquet" in formats:
        out_pq = merged_stem.with_suffix(".parquet")
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_pq, index=False)
        written.append(str(out_pq))
        logging.info("[merged] updated %s (rows=%d)", out_pq.resolve(), len(merged))

    return written

# ───────────────────────── FDR harmonization helpers ─────────────────────────

def _to_int_series(s: pd.Series) -> pd.Series:
    """Coerce to pandas nullable Int64, preserving NA."""
    x = pd.to_numeric(s, errors="coerce").round(0)
    try:
        return x.astype("Int64")
    except Exception:
        return x.astype("float").astype("Int64")

def _resolve_fdr_rowwise(df: pd.DataFrame) -> pd.Series:
    """
    Choose fdr per row with priority: GA → MIN → DEF → SAV.
    If multiple disagree, pick the mode; if still tied, pick max.
    Returns Int64 series.
    """
    cand_cols = [c for c in ["fdr_ga","fdr_min","fdr_def","fdr_sav"] if c in df.columns]
    if not cand_cols:
        return pd.Series([pd.NA]*len(df), index=df.index, dtype="Int64")

    chosen = df[cand_cols[0]].copy()
    for c in cand_cols[1:]:
        mask = chosen.isna()
        if mask.any():
            chosen[mask] = df.loc[mask, c]

    # detect conflicts via rowwise min/max ignoring NAs
    arrs = [df[c].astype("Int64") for c in cand_cols]
    minv = arrs[0]
    maxv = arrs[0]
    for a in arrs[1:]:
        minv = minv.combine(a, lambda l, r: r if pd.isna(l) else (l if pd.isna(r) else min(l, r)))
        maxv = maxv.combine(a, lambda l, r: r if pd.isna(l) else (l if pd.isna(r) else max(l, r)))
    conflict_mask = (~minv.isna()) & (~maxv.isna()) & (minv != maxv)
    n_conflict = int(conflict_mask.sum())
    if n_conflict:
        logging.warning("[FDR] %d row(s) had conflicting FDR across inputs; resolving by mode→max with priority GA>MIN>DEF>SAV.", n_conflict)
        for idx in df.index[conflict_mask]:
            vals = [v for v in (df.at[idx, c] for c in cand_cols) if pd.notna(v)]
            if vals:
                counts: Dict[int,int] = {}
                for v in vals:
                    counts[int(v)] = counts.get(int(v), 0) + 1
                best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]  # freq then value
                chosen.at[idx] = best
    return _to_int_series(chosen)

# ───────────────────────── output helpers (mirror defense_forecast) ─────────────────────────

def _out_paths(base_dir: Path, season: str, gw_from: int, gw_to: int,
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

def _write_points(df: pd.DataFrame, paths: List[Path]) -> List[str]:
    written: List[str] = []
    for p in paths:
        if p.suffix.lower() == ".csv":
            tmp = df.copy()
            if "date_sched" in tmp.columns:
                tmp["date_sched"] = pd.to_datetime(tmp["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp.to_csv(p, index=False)
            written.append(str(p))
        elif p.suffix.lower() == ".parquet":
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
            written.append(str(p))
        else:
            raise ValueError(f"Unsupported output extension: {p.suffix}")
    return written

# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Forecast expected FPL points for upcoming fixtures.")

    # Existing explicit-file flags (remain supported)
    ap.add_argument("--minutes", type=Path, help="Explicit minutes CSV/Parquet")
    ap.add_argument("--goals-assists", type=Path, help="Explicit goals_assists CSV/Parquet")
    ap.add_argument("--saves", type=Path, default=None, help="Explicit saves CSV/Parquet")
    ap.add_argument("--defense", type=Path, default=None, help="Explicit defense CSV/Parquet")

    # Auto-resolve roots + window (optional if explicit files are given)
    ap.add_argument("--minutes-root", type=Path, help="Root dir for minutes (…/<season>/GWx_y.{csv,parquet})")
    ap.add_argument("--ga-root", type=Path, help="Root dir for goals_assists")
    ap.add_argument("--saves-root", type=Path, help="Root dir for saves")
    ap.add_argument("--defense-root", type=Path, help="Root dir for defense")

    ap.add_argument("--future-season", type=str, help="Season for auto-resolve (e.g., 2025-2026)")
    ap.add_argument("--gw-from", type=int, help="GW window start (for auto-resolve)")
    ap.add_argument("--gw-to", type=int, help="GW window end (for auto-resolve)")
    ap.add_argument("--zero-pad-filenames", action="store_true", help="Use GW05_07 instead of GW5_7 when resolving & writing")

    # Output (mirrors defense_forecast): no versioning
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out-format", choices=["csv","parquet","both"], default="both")

    # Policy / roster
    ap.add_argument("--discipline-priors", type=str, default=None,
                    help='Override per-90 priors, e.g. "GK:-0.05,DEF:-0.20,MID:-0.15,FWD:-0.10"')
    ap.add_argument("--teams-json", type=Path, default=None)
    ap.add_argument("--league-filter", type=str, default=None)
    ap.add_argument("--require-on-roster", action="store_true")

    ap.add_argument("--no-merged", action="store_true")
    ap.add_argument("--log-level", default="INFO", type=str)
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    # Season output root (defense style)
    season_dir = args.out_dir / f"{args.future_season}" if args.future_season else args.out_dir
    artifacts_dir = season_dir / "artifacts"

    # ---- Resolve input files (explicit > auto-resolve) ----
    def res(tag, explicit, root):
        return _resolve_input_path(explicit, root, args.future_season, args.gw_from, args.gw_to,
                                   args.zero_pad_filenames, tag)

    minutes_path = args.minutes if args.minutes else res("minutes", args.minutes, args.minutes_root)
    ga_path      = args.goals_assists if args.goals_assists else res("goals-assists", args.goals_assists, args.ga_root)
    sav_path     = args.saves if args.saves else (res("saves", args.saves, args.saves_root) if args.saves_root else None)
    def_path     = args.defense if args.defense else (res("defense", args.defense, args.defense_root) if args.defense_root else None)

    # Louder guardrails if missing DEF/SAV
    if def_path is None:
        logging.warning("[DEF] not provided → clean sheets, GA penalties, and DCP will default to 0.")
    if sav_path is None:
        logging.warning("[SAV] not provided → GK saves points will default to 0.")

    # ---- Read inputs (dual loader) ----
    min_need = ["season","player_id","team_id","pos","gw_orig","pred_minutes"]
    min_df = _read_csv(minutes_path, min_need, tag="MIN")
    _coverage(min_df, ["season","gw_orig","player_id","team_id","p_play","p60","pred_minutes","exp_minutes_points",
                       "pos","player","date_sched","date_played","kickoff_time",
                       "team","opponent","opponent_id","is_home","fbref_id","game_id","fdr"], "MIN")

    ga_need = ["season","player_id","team_id","gw_orig"]
    ga_df = _read_csv(ga_path, ga_need, tag="GA")
    _coverage(ga_df, ["season","gw_orig","player_id","team_id",
                      "p_goal","p_assist","prob_goal","prob_assist","p_goal_cal","prob_assist_cal","prob_goal_cal","prob_assist_cal",
                      "pred_goals_mean","pred_assists_mean","fdr",
                      "player","pos","date_sched","date_played","kickoff_time",
                      "team","opponent","opponent_id","is_home","fbref_id","game_id"], "GA")

    def_df = None
    if def_path:
        def_df = _read_csv(def_path, tag="DEF")
        _coverage(def_df, ["season","gw_orig","player_id","team_id","fdr",
                           "prob_cs","p_teamCS","lambda90","exp_gc",
                           "prob_dcp","p_dcp","dcp_prob",
                           "lambda90_dcp","dcp_lambda90","dcp_p90","lambda_dcp90",
                           "expected_dc",
                           "date_sched","date_played","kickoff_time"], "DEF")

    sav_df = None
    if sav_path:
        sav_df = _read_csv(sav_path, tag="SAV")
        _coverage(sav_df, ["season","gw_orig","player_id","team_id","fdr","pred_saves_mean","pred_saves_poisson",
                           "pred_saves_p90_mean","pred_saves_p90_poisson","pred_minutes","player","pos",
                           "date_sched","date_played","kickoff_time"], "SAV")

    KEY = FORCED_KEY
    logging.info("[key] FORCED -> %s (date columns not used in key)", KEY)

    # De-dupe
    min_df = _drop_exact_dupes(min_df, "MIN")
    ga_df  = _drop_exact_dupes(ga_df,  "GA")
    if def_df is not None: def_df = _drop_exact_dupes(def_df, "DEF")
    if sav_df is not None: sav_df = _drop_exact_dupes(sav_df, "SAV")

    # Base
    if not set(KEY).issubset(min_df.columns):
        raise ValueError(f"MINUTES missing key columns: {KEY}")
    base = min_df.copy()

    # Roster gate
    seasons_in_minutes = sorted(base["season"].dropna().astype(str).unique().tolist())
    allow_keys = _build_roster_keyset(args.teams_json, seasons_in_minutes, args.league_filter)
    base = _apply_roster_gate(base, allow_keys, artifacts_dir, args.require_on_roster)
    if base.empty:
        raise RuntimeError("No rows remain after roster gating.")

    # Identity & legacy meta from GA first, then MIN
    if ga_df is not None and len(ga_df):
        id_keep = ["player","pos","team","opponent","opponent_id","is_home","fbref_id","game_id","date_sched","date_played","kickoff_time","fdr"]
        right = ga_df[KEY + [c for c in id_keep if c in ga_df.columns]]
        base = _merge_with_nearest_date(base, right, KEY, [c for c in id_keep if c in right.columns], tag="GA:identity+meta")
    fill_keep = ["player","pos","team","opponent","opponent_id","is_home","fbref_id","game_id","date_sched","date_played","kickoff_time","fdr"]
    right2 = min_df[KEY + [c for c in fill_keep if c in min_df.columns]]
    base = _merge_with_nearest_date(base, right2, KEY, [c for c in fill_keep if c in right2.columns], tag="MIN:meta-fallback")
    for c in ("player","pos"):
        if c not in base.columns: base[c] = np.nan

    # Collect FDR from each source into separate columns for consistency checks
    # GA fdr (may have been merged in under name 'fdr')
    if "fdr" in base.columns:
        # If this 'fdr' came from GA merge (most recent), park as fdr_ga; we will pull minutes/def/sav next
        base = base.rename(columns={"fdr": "fdr_ga"})
    # Pull minutes fdr explicitly
    if "fdr" in min_df.columns:
        fmin = right2[KEY + ["fdr"]].rename(columns={"fdr": "fdr_min"})
        base = _merge_with_nearest_date(base, fmin, KEY, ["fdr_min"], tag="MIN:fdr")
    # DEF fdr
    if def_df is not None and "fdr" in def_df.columns:
        fdef = def_df[KEY + ["fdr"]].copy().rename(columns={"fdr":"fdr_def"})
        base = _merge_with_nearest_date(base, fdef, KEY, ["fdr_def"], tag="DEF:fdr")
    # SAV fdr
    if sav_df is not None and "fdr" in sav_df.columns:
        fsav = sav_df[KEY + ["fdr"]].copy().rename(columns={"fdr":"fdr_sav"})
        base = _merge_with_nearest_date(base, fsav, KEY, ["fdr_sav"], tag="SAV:fdr")

    # Normalize candidate FDRs to Int64 and resolve to a single 'fdr'
    for cc in ["fdr_ga","fdr_min","fdr_def","fdr_sav"]:
        if cc in base.columns:
            base[cc] = _to_int_series(base[cc])
    base["fdr"] = _resolve_fdr_rowwise(base)

    # GA probabilities & expectations
    if ga_df is not None and len(ga_df):
        tmp = ga_df.copy()
        tmp["p_goal"]   = _select_prob_series(tmp, ["p_goal","prob_goal","p_goal_cal","prob_goal_cal"])
        tmp["p_assist"] = _select_prob_series(tmp, ["p_assist","prob_assist","p_assist_cal","prob_assist_cal"])
        tmp["xg_mean"]  = _choose_first_num(tmp, ["pred_goals_mean"], default=np.nan)
        tmp["xa_mean"]  = _choose_first_num(tmp, ["pred_assists_mean"], default=np.nan)
        keep = ["p_goal","p_assist","xg_mean","xa_mean"]
        right = tmp[KEY + keep + [c for c in RIGHT_DATE_PREFS if c in tmp.columns]]
        base = _merge_with_nearest_date(base, right, KEY, keep, tag="GA:probs_means")
    else:
        base["p_goal"] = np.nan; base["p_assist"] = np.nan; base["xg_mean"] = np.nan; base["xa_mean"] = np.nan

    p_goal = pd.to_numeric(base.get("p_goal"), errors="coerce"); p_asst = pd.to_numeric(base.get("p_assist"), errors="coerce")
    lam_goal = _lambda_from_p(p_goal); lam_asst = _lambda_from_p(p_asst)
    lam_goal = lam_goal.where(lam_goal.notna() & (lam_goal > 0), pd.to_numeric(base.get("xg_mean"), errors="coerce"))
    lam_asst = lam_asst.where(lam_asst.notna() & (lam_asst > 0), pd.to_numeric(base.get("xa_mean"), errors="coerce"))
    lam_goal = lam_goal.fillna(0.0).clip(lower=0.0); lam_asst = lam_asst.fillna(0.0).clip(lower=0.0)

    # Defense signals (CS/GA + DCP)
    if def_df is not None and len(def_df):
        tmp = def_df.copy()
        tmp["p_cs"]         = _select_prob_series(tmp, ["prob_cs","p_teamCS"])
        tmp["lambda90"]     = pd.to_numeric(tmp.get("lambda90", np.nan), errors="coerce")
        tmp["exp_gc"]       = pd.to_numeric(tmp.get("exp_gc",   np.nan), errors="coerce")
        tmp["p_dcp"]        = _select_prob_series(tmp, ["prob_dcp","p_dcp","dcp_prob"])
        tmp["lambda90_dcp"] = _choose_first_num(tmp, ["lambda90_dcp","dcp_lambda90","dcp_p90","lambda_dcp90"], default=np.nan)
        tmp["exp_dc"]       = pd.to_numeric(tmp.get("expected_dc", np.nan), errors="coerce")
        keep = ["p_cs","lambda90","exp_gc","p_dcp","lambda90_dcp","exp_dc"]
        right = tmp[KEY + keep + [c for c in RIGHT_DATE_PREFS if c in tmp.columns]]
        base = _merge_with_nearest_date(base, right, KEY, keep, tag="DEF")
    else:
        for c in ["p_cs","lambda90","exp_gc","p_dcp","lambda90_dcp","exp_dc"]:
            base[c] = np.nan

    # Saves (GK) — prefer per-match λ; else per-90 scaled by minutes
    if sav_df is not None and len(sav_df):
        tmp = sav_df.copy()
        lam_pm  = _choose_first_num(tmp, ["pred_saves_mean","pred_saves_poisson"], default=np.nan)
        lam_p90 = _choose_first_num(tmp, ["pred_saves_p90_mean","pred_saves_p90_poisson"], default=np.nan)
        tmp["saves_lambda_raw"] = lam_pm.where(lam_pm.notna(), lam_p90)
        keep = ["saves_lambda_raw"]
        right = tmp[KEY + keep + [c for c in RIGHT_DATE_PREFS if c in tmp.columns]]
        base = _merge_with_nearest_date(base, right, KEY, keep, tag="SAV")
    else:
        base["saves_lambda_raw"] = np.nan

    # Minutes & appearance
    m = pd.to_numeric(base.get("pred_minutes", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    base["pred_minutes"] = m
    base["p1"]  = pd.to_numeric(base.get("p_play", np.nan), errors="coerce")
    base["p60"] = pd.to_numeric(base.get("p60",     np.nan), errors="coerce")
    base["p1"]  = base["p1"].where(base["p1"].notna(), (m > 0).astype(float))
    base["p60"] = base["p60"].where(base["p60"].notna(), np.clip(m/90.0, 0.0, 1.0))
    base["p60"] = np.minimum(base["p60"], base["p1"])
    base["xp_appearance"] = pd.to_numeric(base.get("exp_minutes_points"), errors="coerce")
    base["xp_appearance"] = base["xp_appearance"].where(base["xp_appearance"].notna(), base["p1"] + base["p60"])

    # Goals & assists EP
    goal_pts_vec = _pos_points_vector(base["pos"], kind="goal")
    base["xp_goals"]   = goal_pts_vec * lam_goal.to_numpy()
    base["xp_assists"] = 3.0 * lam_asst.to_numpy()

    # Clean sheet EP (≥60)
    cs_pts_vec = _pos_points_vector(base["pos"], kind="cs")
    p_cs = pd.to_numeric(base.get("p_cs", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    base["xp_clean_sheets"] = base["p60"] * p_cs * cs_pts_vec

    # Goals-conceded penalty
    lam90   = pd.to_numeric(base.get("lambda90", np.nan), errors="coerce")
    exp_gc  = pd.to_numeric(base.get("exp_gc",   np.nan), errors="coerce")
    lam_team = lam90.where(lam90.notna(), exp_gc)
    lam_team = lam_team.where(
        lam_team.notna(),
        (-np.log(np.clip(p_cs.replace(0, np.nan), 1e-6, 1 - 1e-6))).replace([np.inf, -np.inf], np.nan)
    )
    lam_on = lam_team.fillna(0.0) * (m / 90.0)
    e_pairs = _expected_gc_pairs_lambda_on(lam_on.to_numpy())
    is_def_like_mask = base["pos"].fillna("").astype(str).str.upper().str[:3].isin(["GKP","GK","DEF"]).to_numpy()
    base["xp_concede_penalty"] = np.where(is_def_like_mask, -e_pairs, 0.0)

    # Saves EP (GK only) — gated by p1
    raw_lambda = pd.to_numeric(base.get("saves_lambda_raw"), errors="coerce")
    had_pm = bool(sav_df is not None and (
        ("pred_saves_mean" in (sav_df.columns if sav_df is not None else [])) or
        ("pred_saves_poisson" in (sav_df.columns if sav_df is not None else []))
    ))
    lam_saves = raw_lambda if had_pm else raw_lambda * np.clip(m/90.0, 0.0, None)
    lam_saves = lam_saves.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    exp_floor_s3 = _expected_floor_div3_poisson(lam_saves.to_numpy(), kmax=None)
    base["xp_saves_points"] = np.where(_is_gk(base["pos"]), base["p1"].to_numpy() * exp_floor_s3, 0.0)

    # Discipline priors
    priors = {"GK": -0.05, "GKP": -0.05, "DEF": -0.18, "MID": -0.14, "FWD": -0.10}
    if args.discipline_priors:
        try:
            upd = dict(x.split(":") for x in args.discipline_priors.split(","))
            priors.update({k.strip().upper(): float(v) for k, v in upd.items()})
        except Exception:
            logging.warning("Failed to parse --discipline-priors; using defaults.")
    pos_upper = base["pos"].fillna("").astype(str).str.upper()
    prior_per90 = pos_upper.map(lambda x: priors.get(x, priors.get(x[:3], -0.12))).astype(float).fillna(-0.12)
    base["xp_discipline_prior"] = prior_per90 * (m / 90.0)

    # DCP exposure (outfield)
    is_outfield = _is_outfield(base["pos"])
    is_def = _is_def(base["pos"])

    prob_dcp_in = pd.to_numeric(base.get("p_dcp"), errors="coerce").clip(0.0, 1.0)
    lam90_dcp   = pd.to_numeric(base.get("lambda90_dcp"), errors="coerce")
    exp_dc_dir  = pd.to_numeric(base.get("exp_dc"), errors="coerce")

    expected_dc = exp_dc_dir.where(exp_dc_dir.notna(), lam90_dcp * (m / 90.0))

    lam_on_dcp = lam90_dcp * (m / 90.0)
    sf10 = _poisson_sf_vectorized(lam_on_dcp.to_numpy(), 10)
    sf12 = _poisson_sf_vectorized(lam_on_dcp.to_numpy(), 12)
    sf_fallback = np.where(is_def, sf10, sf12)

    prob_dcp_final = prob_dcp_in.where(prob_dcp_in.notna(), sf_fallback)
    prob_dcp_final = np.where(is_outfield, prob_dcp_final, np.nan)

    base["prob_dcp"]    = np.where(is_outfield, prob_dcp_final, np.nan)
    base["expected_dc"] = np.where(is_outfield, expected_dc, np.nan)
    base["xp_dcp_bonus"] = np.where(is_outfield, 2.0 * prob_dcp_final, 0.0)

    # Sum → xPts
    comp_cols = [
        "xp_appearance",
        "xp_goals",
        "xp_assists",
        "xp_clean_sheets",
        "xp_saves_points",
        "xp_dcp_bonus",
    ]
    for c in comp_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    base["xPts"] = base[comp_cols].sum(axis=1)

    # Friendly aliases
    base["team_prob_cs"]     = pd.to_numeric(base.get("p_cs"), errors="coerce")
    base["team_ga_lambda90"] = pd.to_numeric(base.get("lambda90"), errors="coerce")
    base["team_exp_gc"]      = pd.to_numeric(base.get("exp_gc"), errors="coerce")

    # Normalize date_sched to date-only in memory (writer keeps date)
    if "date_sched" in base.columns:
        base["date_sched"] = pd.to_datetime(base["date_sched"], errors="coerce").dt.normalize()

    # Output columns
    out_cols = list(dict.fromkeys([
        *FORCED_KEY, "player", "pos",
        "date_sched", "date_played", "kickoff_time",
        "team", "opponent", "opponent_id", "is_home", "fbref_id", "game_id", "fdr",
        "pred_minutes", "p1", "p60",
        "p_goal", "p_assist", "xg_mean", "xa_mean",
        "team_prob_cs", "team_ga_lambda90", "team_exp_gc",
        "prob_dcp", "expected_dc",
        "xp_appearance","xp_goals","xp_assists","xp_clean_sheets",
        "xp_concede_penalty","xp_saves_points","xp_discipline_prior","xp_dcp_bonus",
        "xPts",
    ]))
    out_cols = [c for c in out_cols if c in base.columns]

    # Sort order
    sort_cols = [c for c in ["season", "gw_orig", "team_id", "player_id"] if c in base.columns]
    for dc in LEFT_DATE_PREFS:
        if dc in base.columns and dc not in sort_cols:
            sort_cols.append(dc)

    out = base[out_cols].copy().sort_values(by=sort_cols)
    # ensure 'fdr' stays Int64 in memory; writer will preserve it (CSV writes as int or blank)
    if "fdr" in out.columns:
        out["fdr"] = _to_int_series(out["fdr"])
    out_rounded = _round_for_output(out)

    # Determine GW window for output naming (robust even with explicit inputs)
    gw_series = pd.to_numeric(out_rounded.get("gw_orig"), errors="coerce").dropna().astype(int)
    if len(gw_series):
        gw_from_eff, gw_to_eff = int(gw_series.min()), int(gw_series.max())
    else:
        gw_from_eff = int(args.gw_from) if args.gw_from is not None else 0
        gw_to_eff   = int(args.gw_to)   if args.gw_to   is not None else 0

    # Write per-window file(s) (defense_forecast style)
    out_paths = _out_paths(
        base_dir=args.out_dir,             # ← just the base
        season=args.future_season or "",   # ← let _out_paths add the season
        gw_from=gw_from_eff,
        gw_to=gw_to_eff,
        zero_pad=args.zero_pad_filenames,
        out_format=args.out_format,
    )

    written = _write_points(out_rounded, out_paths)
    logging.info("[write] upcoming expected points -> %s (rows=%d)", ", ".join(written), len(out_rounded))

    # Update cumulative combined file(s) unless suppressed (format-aware)
    merged_written: Optional[List[str]] = None
    if not args.no_merged:
        fmt_map = {"csv": ["csv"], "parquet": ["parquet"], "both": ["csv", "parquet"]}
        merged_formats = fmt_map[args.out_format]
        merged_stem = args.out_dir / "expected_points"
        merged_written = _update_merged(out_rounded, merged_stem, FORCED_KEY, sort_cols, merged_formats)

    # Logs
    nonzero_ga = int((pd.to_numeric(base.get('xg_mean'), errors='coerce').fillna(0) > 0).sum() |
                     (pd.to_numeric(base.get('p_goal'), errors='coerce').fillna(0) > 0).sum())
    nonzero_as = int((pd.to_numeric(base.get('xa_mean'), errors='coerce').fillna(0) > 0).sum() |
                     (pd.to_numeric(base.get('p_assist'), errors='coerce').fillna(0) > 0).sum())
    logging.info("[GA] non-zero λ -> goals=%d, assists=%d", nonzero_ga, nonzero_as)

    # Simple diagnostic print (defense_forecast style)
    diag = {
        "rows": int(len(out_rounded)),
        "season": str(args.future_season),
        "gw_window": {"from": int(gw_from_eff), "to": int(gw_to_eff)},
        "inputs": {
            "minutes": str(minutes_path),
            "goals_assists": str(ga_path),
            "saves": str(sav_path) if sav_path else None,
            "defense": str(def_path) if def_path else None,
        },
        "out": written,
        "merged_out": merged_written,
        "schema": SCHEMA_VERSION,
    }
    print(json.dumps(diag, indent=2))

if __name__ == "__main__":
    main()
