#!/usr/bin/env python3 
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# ===============================
# Contract & Normalization
# ===============================

POS_ALLOWED = {"GK", "DEF", "MID", "FWD"}
_POS_MAP = {
    "GKP": "GK", "GK": "GK", "G": "GK",
    "DEF": "DEF", "DF": "DEF", "D": "DEF",
    "MID": "MID", "MF": "MID", "M": "MID",
    "FWD": "FWD", "FW": "FWD", "F": "FWD", "ST": "FWD",
}

# Strict required columns for the solver
REQUIRED_OUT_COLS = [
    "season", "gw", "player_id", "team_id", "pos", "price", "sell_price", "p60",
    "xPts", "exp_pts_var", "cs_prob", "is_dgw", "team", "captain_uplift",
]

# Accepted upstream cols (plus metadata we passthrough)
ACCEPTED_IN_COLS = {
    "season", "gw", "gw_orig",
    "player_id", "team_id", "team_code", "player", "pos",
    "p1","p60", "pred_minutes", "xMins_mean",
    "xPts", "exp_pts_var",
    "__p_cs__", "team_prob_cs", "cs_prob",
    "team_ga_lambda90", "__lambda90__", "team_exp_gc",
    "is_dgw", "fixture_count",
    # legacy/meta
    "date_sched", "date_played", "kickoff_time",
    "team", "opponent", "opponent_id", "is_home", "fbref_id", "game_id",
    "fdr",
    # prices
    "price", "now_price", "current_price", "now_cost", "cost",
    # other
    "captain_uplift",
}

# Legacy metadata we group to the left (after season)
LEGACY_META_COLS = [
    "date_sched", "date_played", "kickoff_time",
    "team", "opponent", "opponent_id", "is_home", "fbref_id", "game_id", "fdr",
]

KEY_FOR_STACK = ["season", "gw", "player_id"]  # consolidated unique key
BANNED_COLS_IN_STACK = {"exp_pts_mean"}  # kill legacy/stray columns on stack

# ===============================
# IO Helpers
# ===============================

def _read_any(path: str | Path) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet") or p.endswith(".pq"):
        return pd.read_parquet(p)
    if p.endswith(".csv"):
        try:
            head = pd.read_csv(p, nrows=0)
            parse_dates = [c for c in ("date_sched","date_played","kickoff_time") if c in head.columns]
            return pd.read_csv(p, parse_dates=parse_dates, low_memory=False)
        except Exception:
            return pd.read_csv(p, low_memory=False)
    if p.endswith(".json") or p.endswith(".jsonl"):
        return pd.read_json(p, lines=p.endswith(".jsonl"))
    raise ValueError("Unsupported predictions format. Use .csv, .parquet, .json, or .jsonl")

def _detect_parquet_engine() -> Optional[str]:
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception:
            return None

# ---------- autoloader (season + GW window) ----------

def _fmt_gw(n: int, zero_pad: bool) -> str:
    return f"{int(n):02d}" if zero_pad else f"{int(n)}"

def _candidate_paths(root: Path, season: str, gw_from: int, gw_to: int, zero_pad: bool) -> List[Path]:
    season_dir = root / str(season)
    a, b = _fmt_gw(gw_from, zero_pad), _fmt_gw(gw_to, zero_pad)
    return [season_dir / f"GW{a}_{b}.csv", season_dir / f"GW{a}_{b}.parquet"]

def _glob_fallback(root: Path, season: str, gw_from: int, gw_to: int) -> Optional[Path]:
    season_dir = root / str(season)
    if not season_dir.exists():
        return None
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

def _resolve_preds_path(explicit: Optional[str | Path],
                        root: Optional[str | Path],
                        season: Optional[str],
                        gw_from: Optional[int],
                        gw_to: Optional[int],
                        zero_pad: bool) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"[preds] file not found: {p}")
        return p
    if root is None or season is None or gw_from is None or gw_to is None:
        raise ValueError("[preds] need --preds-root, --future-season, and GW window to auto-resolve.")
    root = Path(root)
    for cand in _candidate_paths(root, season, gw_from, gw_to, zero_pad):
        if cand.exists():
            return cand
    fb = _glob_fallback(root, season, gw_from, gw_to)
    if fb:
        return fb
    season_dir = root / str(season)
    tried = "\n".join(f"  - {p}" for p in _candidate_paths(root, season, gw_from, gw_to, zero_pad))
    raise FileNotFoundError(
        f"[preds] Could not locate GW{gw_from}_{gw_to} under {season_dir}\n"
        f"Tried:\n{tried}\nAlso tried glob fallback."
    )

# ---------- writer helpers (GW-window + consolidated) ----------

def _out_paths(base_dir: Path, season: str, gw_from: int, gw_to: int, zero_pad: bool) -> List[Path]:
    a, b = _fmt_gw(gw_from, zero_pad), _fmt_gw(gw_to, zero_pad)
    sdir = base_dir / str(season)
    sdir.mkdir(parents=True, exist_ok=True)
    stem = sdir / f"GW{a}_{b}"
    return [Path(str(stem) + ".csv"), Path(str(stem) + ".parquet")]

def _write_dual(df: pd.DataFrame, paths: List[Path]) -> List[str]:
    written: List[str] = []
    for p in paths:
        if p.suffix.lower() == ".csv":
            tmp = df.copy()
            if "date_sched" in tmp.columns and pd.api.types.is_datetime64_any_dtype(tmp["date_sched"]):
                tmp["date_sched"] = pd.to_datetime(tmp["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp.to_csv(p, index=False)
            written.append(str(p))
        elif p.suffix.lower() in (".parquet", ".pq"):
            eng = _detect_parquet_engine()
            if eng:
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(p, index=False, engine=eng)
                written.append(str(p))
            else:
                # no parquet engine; silently skip parquet
                pass
        else:
            raise ValueError(f"Unsupported output extension: {p.suffix}")
    return written

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _update_consolidated(new_df: pd.DataFrame, out_dir: Path) -> List[str]:
    cons_csv = out_dir / "optimizer_input.csv"
    cons_parq = out_dir / "optimizer_input.parquet"

    if cons_csv.exists():
        old = pd.read_csv(cons_csv, low_memory=False)
    else:
        old = pd.DataFrame(columns=new_df.columns)

    # Drop banned columns from both sides (prevents resurrection)
    old = old.drop(columns=[c for c in BANNED_COLS_IN_STACK if c in old.columns], errors="ignore")
    new_df = new_df.drop(columns=[c for c in BANNED_COLS_IN_STACK if c in new_df.columns], errors="ignore")

    # Align columns (union), but prefer *new_df* column order
    union_cols = list(dict.fromkeys([*new_df.columns, *old.columns]))
    old = old.reindex(columns=union_cols)
    new = new_df.reindex(columns=union_cols)

    # Overwrite-by-key semantics
    key_df = new[KEY_FOR_STACK].drop_duplicates()
    old_mark = old.merge(key_df.assign(__hit__=1), on=KEY_FOR_STACK, how="left")
    old_kept = old_mark[old_mark["__hit__"].isna()].drop(columns="__hit__")
    merged = pd.concat([old_kept, new], ignore_index=True, sort=False)

    # Sort for readability
    sort_cols = [c for c in ["season", "gw", "team_id", "player_id"] if c in merged.columns]
    merged = merged.sort_values(sort_cols)

    # Reorder columns to match new_df first (legacy extras trail)
    trailing = [c for c in merged.columns if c not in new_df.columns]
    merged = merged[[*new_df.columns, *trailing]]

    # CSV (atomic)
    _atomic_write_csv(merged, cons_csv)

    # Parquet (best-effort)
    written = [str(cons_csv)]
    eng = _detect_parquet_engine()
    if eng:
        merged.to_parquet(cons_parq, index=False, engine=eng)
        written.append(str(cons_parq))
    return written

# ===============================
# Validation & Dtypes
# ===============================

def _normalize_pos(s: pd.Series) -> pd.Series:
    s_upper = s.astype("string").str.upper()
    mapped = s_upper.map(_POS_MAP).fillna(s_upper)
    bad = ~mapped.isin(list(POS_ALLOWED))
    if bad.any():
        raise ValueError(f"Unexpected pos values: {sorted(mapped[bad].dropna().unique().tolist())}")
    return mapped

def _coerce_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # strings
    for c in ["season", "player_id", "team_id", "pos", "team"]:
        df[c] = df[c].astype("string")
    if "player" in df.columns:
        df["player"] = df["player"].astype("string")
    for c in ["opponent","opponent_id","fbref_id","game_id"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    if "is_home" in df.columns:
        df["is_home"] = df["is_home"].astype("boolean")

    for c in ("date_sched","date_played","kickoff_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # ints / bools
    df["gw"] = pd.to_numeric(df["gw"], downcast="integer").astype("int64")
    df["is_dgw"] = df["is_dgw"].astype("bool")

    # floats
    for c in ["price", "sell_price", "p60", "xPts", "exp_pts_var", "cs_prob", "captain_uplift", "fdr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise", downcast="float")
    return df

def _validate_contract(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_OUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required output columns: {missing}")
    na_cols = [c for c in REQUIRED_OUT_COLS if df[c].isna().any()]
    if na_cols:
        raise ValueError(f"NaNs found in required columns: {na_cols}")
    for c in ["p60", "cs_prob"]:
        bad = df[(df[c] < 0) | (df[c] > 1)]
        if not bad.empty:
            raise ValueError(f"{c} out of [0,1]; first few:\n{bad[['player_id','gw',c]].head()}")
    if (df["price"] < 0).any() or (df["sell_price"] < 0).any():
        raise ValueError("Negative prices detected")
    if (df["exp_pts_var"] < 0).any():
        raise ValueError("Negative exp_pts_var detected")
    dupe = df.duplicated(subset=KEY_FOR_STACK, keep=False)
    if dupe.any():
        rows = df.loc[dupe, KEY_FOR_STACK].drop_duplicates()
        raise ValueError(f"Duplicate key(s):\n{rows}")

# ===============================
# Team state & Master prices / names
# ===============================

def _load_team_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _owned_sell_map(team_state: dict) -> Dict[str, float]:
    return {e["player_id"]: float(e["sell_price"]) for e in team_state.get("squad", [])}

def _load_master(master_path: Optional[str]) -> Optional[dict]:
    if not master_path:
        return None
    with open(master_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_short_season(s: str) -> str:
    s = s.strip()
    if len(s) == 9 and s[4] == "-":
        return f"{s[:4]}-{s[-2:]}"
    if "/" in s:
        a, b = s.split("/", 1)
        return f"{a}-{b}"
    return s

def _price_from_master(entry: dict, season_short: str, gw: int) -> Optional[float]:
    prices = entry.get("prices", {}).get(season_short, {})
    if not prices:
        return None
    gws = sorted(int(k) for k in prices.keys() if str(k).isdigit())
    cands = [g for g in gws if g <= gw]
    if not cands:
        return None
    return float(prices[str(max(cands))])

def _earliest_price_in_season(entry: dict) -> Optional[float]:
    all_seasons = entry.get("prices", {})
    best = None
    for s_short, blob in all_seasons.items():
        if not isinstance(blob, dict):
            continue
        gws = [int(k) for k in blob.keys() if str(k).isdigit()]
        if not gws:
            continue
        v = float(blob[str(min(gws))])
        best = v if best is None else min(best, v)
    return best

def _name_from_master(entry: dict) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    if entry.get("name"):
        return str(entry["name"])
    first = str(entry.get("first_name") or "").strip()
    second = str(entry.get("second_name") or "").strip()
    full = (first + " " + second).strip()
    return full or None

# ===============================
# Derivations & resolvers
# ===============================

def _derive_p60(df: pd.DataFrame, p60_col: Optional[str]) -> pd.Series:
    def _enforce(s: pd.Series) -> pd.Series:
        p1 = None
        if "p1" in df.columns:
            p1 = pd.to_numeric(df["p1"], errors="coerce").clip(0.0, 1.0)
        elif "p_play" in df.columns:
            p1 = pd.to_numeric(df["p_play"], errors="coerce").clip(0.0, 1.0)
        if p1 is not None:
            s = s.where(p1.isna(), np.minimum(s, p1))
        return s
    if p60_col and p60_col in df.columns:
        return _enforce(pd.to_numeric(df[p60_col], errors="coerce").clip(0.0, 1.0))
    if "p60" in df.columns:
        return _enforce(pd.to_numeric(df["p60"], errors="coerce").clip(0.0, 1.0))
    if "pred_minutes" in df.columns:
        return (pd.to_numeric(df["pred_minutes"], errors="coerce") / 60.0).clip(0.0, 1.0)
    if "xMins_mean" in df.columns:
        return (pd.to_numeric(df["xMins_mean"], errors="coerce") / 60.0).clip(0.0, 1.0)
    raise ValueError("Missing p60/pred_minutes/xMins_mean — cannot derive p60")

def _ensure_is_dgw(df: pd.DataFrame) -> pd.Series:
    if "is_dgw" in df.columns:
        return df["is_dgw"].astype("boolean").fillna(False).astype(bool)
    if "fixture_count" in df.columns:
        fc = pd.to_numeric(df["fixture_count"], errors="coerce").fillna(1)
        return (fc > 1).astype(bool)
    sizes = df.groupby(["season", "gw", "player_id"])["player_id"].transform("size")
    return (sizes > 1).astype(bool)

def _combine_prod_compl_one_minus(series: pd.Series) -> float:
    vals = series.dropna().astype(float)
    if len(vals) == 0:
        return 0.0
    prod = 1.0
    for v in vals:
        v = max(0.0, min(1.0, v))
        prod *= (1.0 - v)
    return 1.0 - prod

def _aggregate_player_gw(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "gw", "player_id"]
    def agg_fn(g: pd.DataFrame) -> pd.Series:
        out = {
            "team_id": g["team_id"].iloc[0],
            "pos": g["pos"].iloc[0],
            "xPts": g["xPts"].astype(float).sum(),
            "exp_pts_var": pd.to_numeric(g.get("exp_pts_var", pd.Series([0.0] * len(g))), errors="coerce").fillna(0.0).sum(),
            "p60": _combine_prod_compl_one_minus(g["p60"]),
            "cs_prob": _combine_prod_compl_one_minus(g["cs_prob"]),
            "is_dgw": (len(g) > 1),
        }
        # fdr: mean across legs if present
        if "fdr" in g.columns:
            out["fdr"] = pd.to_numeric(g["fdr"], errors="coerce").mean()
        for c in ["team_code","player",*LEGACY_META_COLS]:
            if c in g.columns and c != "fdr":  # avoid overwriting aggregated fdr
                out[c] = g[c].iloc[0]
        return pd.Series(out)
    agg = df.groupby(keys, as_index=False).apply(agg_fn)
    if isinstance(agg.columns, pd.MultiIndex):
        agg.columns = [c[0] if c[0] else c[1] for c in agg.columns]
    return agg

def _maybe_code_from_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = str(name).strip()
    if s.isupper() and s.replace("-", "").isalpha() and 2 <= len(s) <= 5 and " " not in s:
        return s
    tokens = [t for t in s.replace("&", " ").replace("-", " ").split() if t.isalpha()]
    if len(tokens) >= 2:
        letters = "".join(t[0] for t in tokens[:3]).upper()
        return letters[:3] if len(letters) >= 3 else letters
    return s[:3].upper()

def _load_team_code_map(path: Optional[str]) -> Dict[str, str]:
    by_id: Dict[str, str] = {}
    if not path:
        return by_id
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if all(isinstance(v, str) for v in obj.values()):
            sample_val = next(iter(obj.values()))
            if len(sample_val) <= 5 and sample_val.isupper():
                by_id = {str(k): str(v).upper() for k, v in obj.items()}
            else:
                inv = {str(v): str(k).upper() for k, v in obj.items()}
                by_id = inv
        else:
            for tid, payload in obj.items():
                if isinstance(payload, dict):
                    code = payload.get("code") or payload.get("fpl_code") or payload.get("short_name") or payload.get("name")
                    code = _maybe_code_from_name(code)
                    if code:
                        by_id[str(tid)] = code
    elif isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
        tid = item.get("team_id") or item.get("id")
        code = item.get("code") or item.get("fpl_code") or item.get("short_name") or item.get("name")
        code = _maybe_code_from_name(code)
        if tid and code:
            by_id[str(tid)] = code
    return {k: v.upper() for k, v in by_id.items()}

def _resolve_team(
    df: pd.DataFrame,
    team_code_map_path: Optional[str],
    strict: bool = False,
    missing_out: Optional[str] = None,
) -> pd.Series:
    by_id = _load_team_code_map(team_code_map_path)
    if "team_code" in df.columns and df["team_code"].notna().any():
        codes = df["team_code"].astype("string").str.upper()
        if by_id:
            expected = df["team_id"].astype("string").map(lambda tid: by_id.get(str(tid)))
            mismatch_mask = expected.notna() & codes.notna() & (codes != expected)
            if mismatch_mask.any():
                rows = df.loc[mismatch_mask, ["team_id"]].assign(pred_code=codes[mismatch_mask], map_code=expected[mismatch_mask])
                if missing_out:
                    if os.path.dirname(missing_out):
                        os.makedirs(os.path.dirname(missing_out), exist_ok=True)
                    rows.drop_duplicates().to_csv(missing_out, index=False)
                    print(f"[info] Wrote {rows.drop_duplicates().shape[0]} mismatched team_code rows to {missing_out}")
                if strict:
                    raise ValueError(f"team_code mismatch for some team_ids (strict). Examples:\n{rows.head(10)}")
        if by_id:
            fill = df["team_id"].astype("string").map(lambda tid: by_id.get(str(tid)))
            codes = codes.fillna(fill)
        if strict and codes.isna().any():
            bad = df.loc[codes.isna(), ["team_id"]].drop_duplicates().head(20)
            raise ValueError(f"Missing team code for some team_ids (strict). Examples:\n{bad}")
        return codes.fillna(df["team_id"].astype("string").str.upper().str[:3])
    mapped = df["team_id"].astype("string").map(lambda tid: by_id.get(str(tid))) if by_id else pd.Series([pd.NA]*len(df), dtype="string")
    if strict and mapped.isna().any():
        bad = df.loc[mapped.isna(), ["team_id"]].drop_duplicates()
        if missing_out:
            if os.path.dirname(missing_out):
                os.makedirs(os.path.dirname(missing_out), exist_ok=True)
            bad.to_csv(missing_out, index=False)
            print(f"[info] Wrote {bad.shape[0]} unmapped team_ids to {missing_out}")
        raise ValueError(f"Missing team code for some team_ids (strict). Examples:\n{bad.head(20)}")
    if mapped.isna().any() and missing_out:
        out = df.loc[mapped.isna(), ["team_id"]].drop_duplicates()
        if os.path.dirname(missing_out):
            os.makedirs(os.path.dirname(missing_out), exist_ok=True)
        out.to_csv(missing_out, index=False)
        print(f"[info] Wrote {out.shape[0]} unmapped team_ids to {missing_out}")
    return mapped.fillna(df["team_id"].astype("string").str.upper().str[:3])

# ---------- CS probability resolver (robust) ----------

def _resolve_cs_prob(df: pd.DataFrame, preferred: Optional[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    candidates = []
    if preferred and preferred in df.columns:
        candidates.append(preferred)
    for c in ["__p_cs__", "team_prob_cs", "cs_prob"]:
        if c not in candidates and c in df.columns:
            candidates.append(c)
    for c in candidates:
        vals = pd.to_numeric(df[c], errors="coerce").clip(0.0, 1.0)
        out = out.where(out.notna(), vals)
    if out.isna().any():
        lam = None
        for name in ["team_ga_lambda90", "__lambda90__"]:
            if name in df.columns:
                lam = pd.to_numeric(df[name], errors="coerce")
                break
        if lam is not None:
            mins = pd.to_numeric(df.get("pred_minutes", np.nan), errors="coerce")
            lam_on = lam * (np.clip(mins, 0.0, None) / 90.0) if mins.notna().any() else lam
            derived = np.exp(-np.clip(lam_on, 0.0, None))
            out = out.where(out.notna(), derived)
    return out.fillna(0.0).clip(0.0, 1.0)

# ===============================
# Names
# ===============================

def _ensure_player_names(df: pd.DataFrame, master: Optional[dict]) -> pd.DataFrame:
    if "player" in df.columns and df["player"].notna().any():
        return df
    if master is None:
        df["player"] = pd.NA
        return df
    def _get_name(pid: str) -> Optional[str]:
        entry = master.get(str(pid))
        if not entry:
            return None
        if entry.get("name"):
            return str(entry["name"])
        first = str(entry.get("first_name") or "").strip()
        second = str(entry.get("second_name") or "").strip()
        full = (first + " " + second).strip()
        return full or None
    df["player"] = df["player_id"].astype("string").map(_get_name)
    return df

# ===============================
# Builder
# ===============================

def build_optimizer_input(
    team_state_path: str,
    preds_path: str,
    out_dir: str,
    team_code_map_path: Optional[str] = None,
    captain_uplift_col: Optional[str] = None,
    gw_col: str = "gw",
    exp_mean_col: Optional[str] = None,
    cs_prob_col: Optional[str] = None,
    p60_col: Optional[str] = None,
    gw_fallback: Optional[int] = None,
    season_fallback: Optional[str] = None,
    master_path: Optional[str] = None,
    price_gw: Optional[int] = None,
    on_missing_price: str = "error",  # error|use_earliest_in_season|use_preds|drop
    missing_price_out: Optional[str] = None,
    drop_rows_missing_core: bool = False,
    strict_owned_sell: bool = True,
    validate: bool = True,
    preview: int = 0,
    debug_missing_prices: bool = False,
    no_aggregate: bool = False,
    strict_team_code: bool = False,
    missing_team_map_out: Optional[str] = None,
    # params sidecar
    clamp_captain_uplift: bool = True,
    captain_multiplier: float = 2.0,
    tc_multiplier: float = 3.0,
    params_out: Optional[str] = None,
    # for naming GW window files
    future_season: Optional[str] = None,
    zero_pad_filenames: bool = False,
) -> List[str]:
    out_dir = Path(out_dir)
    ts = _load_team_state(team_state_path)
    own_sell = _owned_sell_map(ts)
    df = _read_any(preds_path)

    # Trim columns
    extra = {gw_col, exp_mean_col, cs_prob_col, p60_col} - {None}
    df = df[[c for c in df.columns if c in ACCEPTED_IN_COLS or c in extra]].copy()

    # Season/GW
    if "season" not in df.columns:
        df["season"] = season_fallback or ts.get("season")
    if df["season"].isna().any():
        raise ValueError("Predictions missing 'season' and no fallback provided.")
    if gw_col not in df.columns:
        if gw_fallback is None:
            raise ValueError(f"Missing GW column '{gw_col}' and --gw-fallback not provided.")
        df["gw"] = gw_fallback
    else:
        df = df.rename(columns={gw_col: "gw"})

    # IDs / pos
    for c in ["player_id", "team_id", "pos"]:
        if c not in df.columns:
            raise ValueError(f"Predictions missing required column '{c}'")
    df["pos"] = _normalize_pos(df["pos"])

    # Means / variance / cs_prob / p60
    mean_col = exp_mean_col or ("xPts" if "xPts" in df.columns else "xPts")
    if mean_col not in df.columns:
        raise ValueError(f"Expected points mean column not found (looked for '{mean_col}'). Use --exp-mean-col.")
    df["xPts"] = pd.to_numeric(df[mean_col], errors="coerce")
    df["exp_pts_var"] = pd.to_numeric(
        df["exp_pts_var"] if "exp_pts_var" in df.columns else pd.Series(0.0, index=df.index),
        errors="coerce"
    ).fillna(0.0)

    df["cs_prob"] = _resolve_cs_prob(df, cs_prob_col)
    df["p60"] = _derive_p60(df, p60_col)

    # DGW + aggregate
    df["is_dgw"] = _ensure_is_dgw(df)
    if not no_aggregate and df.duplicated(subset=["season", "gw", "player_id"], keep=False).any():
        df = _aggregate_player_gw(df)

    # Captain uplift
    if captain_uplift_col and captain_uplift_col in df.columns:
        df["captain_uplift"] = pd.to_numeric(df[captain_uplift_col], errors="coerce")
    else:
        df["captain_uplift"] = df["xPts"]
    if clamp_captain_uplift:
        df["captain_uplift"] = df["captain_uplift"].clip(lower=0.0)

    # Team code, names, prices
    df["team"] = _resolve_team(df, team_code_map_path, strict=strict_team_code, missing_out=missing_team_map_out)
    if not master_path:
        raise ValueError("This configuration expects --master for prices (latest ≤ GW).")
    master = _load_master(master_path)
    df = _ensure_player_names(df, master)

    # prices from master
    season_short = _to_short_season(str(df["season"].iloc[0]))
    def _price_lookup(row) -> Optional[float]:
        m = master.get(str(row["player_id"]))
        if not m:
            return None
        gw = int(price_gw if price_gw is not None else row["gw"])
        return _price_from_master(m, season_short, gw)
    price_series = pd.to_numeric(df.apply(_price_lookup, axis=1), errors="coerce")
    missing_mask = price_series.isna()
    if missing_mask.any():
        if on_missing_price == "use_earliest_in_season":
            def _earliest(pid):
                m = master.get(str(pid))
                return _earliest_price_in_season(m) if m else None
            fills = df.loc[missing_mask, "player_id"].map(_earliest)
            price_series.loc[missing_mask] = pd.to_numeric(fills, errors="coerce")
        elif on_missing_price == "use_preds":
            preds_price = None
            for col in ["price","now_price","current_price","now_cost","cost"]:
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if col in {"now_cost","cost"}:
                        s = s / 10.0
                    if not s.isna().any():
                        preds_price = s
                        break
            if preds_price is None:
                raise ValueError("on-missing-price=use_preds, but no usable price column present in predictions.")
            price_series.loc[missing_mask] = preds_price.loc[missing_mask]
        elif on_missing_price in {"drop","error"}:
            pass
        else:
            raise ValueError("Unknown --on-missing-price policy")

        if missing_mask.any() and missing_price_out:
            review_cols = [c for c in ["season","gw","player_id","player","team_id","pos","xPts","cs_prob","p60"] if c in df.columns]
            review = df.loc[missing_mask, review_cols].copy()
            review["price_lookup_gw"] = int(price_gw) if price_gw is not None else df.loc[missing_mask, "gw"]
            out_dir.mkdir(parents=True, exist_ok=True)
            Path(missing_price_out).parent.mkdir(parents=True, exist_ok=True)
            review.to_csv(missing_price_out, index=False)
            print(f"[info] Wrote {len(review)} rows with missing master price to {missing_price_out}")

    df["price"] = price_series
    if on_missing_price == "drop" and df["price"].isna().any():
        before = len(df)
        df = df.loc[df["price"].notna()].copy()
        print(f"[info] Dropped {before - len(df)} rows due to missing master price")

    # sell_price
    if strict_owned_sell:
        df["sell_price"] = df.apply(lambda r: _owned_sell_map(ts).get(str(r["player_id"]), r["price"]), axis=1)
    else:
        df["sell_price"] = df["price"]

    # Final column order — legacy metadata block grouped on the left
    out_cols = [
        # identity + left metadata (like points_forecast)
        "season", "date_sched", "date_played", "kickoff_time",
        "gw",
        "player_id", "player",
        "team_id", "team",
        "opponent", "opponent_id", "is_home", "fbref_id", "game_id", "fdr",
        # core required for solver
        "pos", "price", "sell_price", "p60",
        "xPts", "exp_pts_var", "cs_prob", "is_dgw",
        "captain_uplift",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    output_df = df[out_cols].copy()

    # Types & contract
    output_df = _coerce_output_dtypes(output_df)
    if drop_rows_missing_core:
        before = len(output_df)
        output_df = output_df.dropna(subset=[c for c in REQUIRED_OUT_COLS if c in output_df.columns])
        after = len(output_df)
        if after < before:
            print(f"[info] Dropped {before - after} rows with missing core values")
    if validate:
        _validate_contract(output_df)

    if preview > 0:
        print(output_df.head(preview).to_string(index=False))

    # Sidecar params (unchanged)
    if params_out:
        params = {
            "captain_multiplier": float(captain_multiplier),
            "triple_captain_multiplier": float(tc_multiplier),
            "clamp_captain_uplift": bool(clamp_captain_uplift),
        }
        Path(params_out).parent.mkdir(parents=True, exist_ok=True) if os.path.dirname(params_out) else None
        with open(params_out, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print(f"OK (params): {params_out} -> {params}")

    # ----- Write GW-windowed + consolidated -----
    gw_series = pd.to_numeric(output_df["gw"], errors="coerce").dropna().astype(int)
    gw_from_eff = int(gw_series.min()) if len(gw_series) else 0
    gw_to_eff   = int(gw_series.max()) if len(gw_series) else 0
    season_for_paths = future_season or str(output_df["season"].iloc[0])

    window_paths = _out_paths(out_dir, season_for_paths, gw_from_eff, gw_to_eff, zero_pad_filenames)
    written = _write_dual(output_df, window_paths)

    stacked_written = _update_consolidated(output_df, out_dir)

    print(f"rows={len(output_df)} uniques={output_df[KEY_FOR_STACK].drop_duplicates().shape[0]}")
    return [*written, *stacked_written]

# ===============================
# CLI
# ===============================

def main():
    ap = argparse.ArgumentParser(description="Build strict optimizer_input from team_state + expected_points")

    # explicit preds OR auto-resolve
    ap.add_argument("--team-state", required=True)
    ap.add_argument("--preds")
    ap.add_argument("--preds-root", type=str, help="Root dir that contains <season>/GW<from>_<to>.csv|parquet")
    ap.add_argument("--future-season", type=str, help="Season for auto-resolve (e.g., 2025-2026)")
    ap.add_argument("--gw-from", type=int, help="GW window start")
    ap.add_argument("--gw-to", type=int, help="GW window end")
    ap.add_argument("--zero-pad-filenames", action="store_true", help="Use GW05_07 instead of GW5_7 in filenames")
    # output
    ap.add_argument("--out-dir", help="Directory for outputs (windowed and consolidated)")
    ap.add_argument("--out", help="Legacy: file path base; if given, we'll derive --out-dir from it")
    # mapping & columns
    ap.add_argument("--team-code-map", help="Path to master_teams.json")
    ap.add_argument("--strict-team-code", action="store_true")
    ap.add_argument("--missing-team-map-out")
    ap.add_argument("--captain-uplift-col")
    ap.add_argument("--gw-col", default="gw", help="GW column in preds (e.g., gw_orig)")
    ap.add_argument("--exp-mean-col")
    ap.add_argument("--cs-prob-col")
    ap.add_argument("--p60-col")
    ap.add_argument("--gw-fallback", type=int)
    ap.add_argument("--season-fallback")
    # prices
    ap.add_argument("--master", required=True)
    ap.add_argument("--price-gw", type=int)
    ap.add_argument("--on-missing-price", choices=["error","use_earliest_in_season","use_preds","drop"], default="error")
    ap.add_argument("--missing-price-out")
    # misc
    ap.add_argument("--drop-rows-missing-core", action="store_true")
    ap.add_argument("--simple-sell", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--debug-missing-prices", action="store_true")
    ap.add_argument("--no-aggregate", action="store_true")
    # sidecar params
    ap.add_argument("--no-clamp-captain-uplift", action="store_true")
    ap.add_argument("--captain-multiplier", type=float, default=2.0)
    ap.add_argument("--tc-multiplier", type=float, default=3.0)
    ap.add_argument("--params-out")

    args = ap.parse_args()

    # derive out_dir if only --out is provided
    out_dir = args.out_dir
    if not out_dir:
        if args.out:
            out_dir = os.path.dirname(args.out) or "."
        else:
            print("ERROR: provide --out-dir (preferred) or --out", file=sys.stderr)
            sys.exit(2)

    try:
        preds_path = _resolve_preds_path(
            explicit=args.preds,
            root=args.preds_root,
            season=args.future_season,
            gw_from=args.gw_from,
            gw_to=args.gw_to,
            zero_pad=args.zero_pad_filenames,
        ) if not args.preds else Path(args.preds)

        build_optimizer_input(
            team_state_path=args.team_state,
            preds_path=str(preds_path),
            out_dir=out_dir,
            team_code_map_path=args.team_code_map,
            captain_uplift_col=args.captain_uplift_col,
            gw_col=args.gw_col,
            exp_mean_col=args.exp_mean_col,
            cs_prob_col=args.cs_prob_col,
            p60_col=args.p60_col,
            gw_fallback=args.gw_fallback,
            season_fallback=args.season_fallback,
            master_path=args.master,
            price_gw=args.price_gw,
            on_missing_price=args.on_missing_price,
            missing_price_out=args.missing_price_out,
            drop_rows_missing_core=args.drop_rows_missing_core,
            strict_owned_sell=not args.simple_sell,
            validate=not args.no_validate,
            preview=args.preview,
            debug_missing_prices=args.debug_missing_prices,
            no_aggregate=args.no_aggregate,
            strict_team_code=args.strict_team_code,
            missing_team_map_out=args.missing_team_map_out,
            clamp_captain_uplift=not args.no_clamp_captain_uplift,
            captain_multiplier=args.captain_multiplier,
            tc_multiplier=args.tc_multiplier,
            params_out=args.params_out,
            future_season=args.future_season,
            zero_pad_filenames=args.zero_pad_filenames,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
