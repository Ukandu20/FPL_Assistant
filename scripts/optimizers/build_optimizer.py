#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, List, Tuple

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

# Final strict output columns (player name is optional; we add it when available)
REQUIRED_OUT_COLS = [
    "season", "gw", "player_id", "team_id", "pos", "price", "sell_price", "p60",
    "exp_pts_mean", "exp_pts_var", "cs_prob", "is_dgw", "team_quota_key", "captain_uplift",
]

# Upstream columns we accept (plus explicit flags we’ll map)
ACCEPTED_IN_COLS = {
    "season", "gw", "gw_orig",
    "player_id", "team_id", "team_code", "player", "pos",
    "p60", "pred_minutes", "xMins_mean",
    "xPts", "exp_pts_mean", "exp_pts_var",
    "__p_cs__", "team_prob_cs", "cs_prob",
    "is_dgw", "fixture_count", "date_sched",
    "price", "now_price", "current_price", "now_cost", "cost",
    "captain_uplift", "team_quota_key",
}

# ===============================
# IO Helpers
# ===============================

def _read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".json") or path.endswith(".jsonl"):
        return pd.read_json(path, lines=path.endswith(".jsonl"))
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
    for c in ["season", "player_id", "team_id", "pos", "team_quota_key"]:
        df[c] = df[c].astype("string")
    if "player" in df.columns:
        df["player"] = df["player"].astype("string")
    # ints / bools
    df["gw"] = pd.to_numeric(df["gw"], downcast="integer").astype("int64")
    df["is_dgw"] = df["is_dgw"].astype("bool")
    # floats
    for c in ["price", "sell_price", "p60", "exp_pts_mean", "exp_pts_var", "cs_prob", "captain_uplift"]:
        df[c] = pd.to_numeric(df[c], errors="raise", downcast="float")
    return df

def _validate_contract(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_OUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required output columns: {missing}")

    # Only enforce NaN-free on required columns
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

    dupe = df.duplicated(subset=["season", "gw", "player_id"], keep=False)
    if dupe.any():
        rows = df.loc[dupe, ["season", "gw", "player_id"]].drop_duplicates()
        raise ValueError(f"Duplicate (season,gw,player_id) keys:\n{rows}")

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
    if len(s) == 9 and s[4] == "-":  # 2025-2026
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

def _earliest_price_in_season(entry: dict, season_short: str) -> Optional[float]:
    prices = entry.get("prices", {}).get(season_short, {})
    if not prices:
        return None
    gws = sorted(int(k) for k in prices.keys() if str(k).isdigit())
    return float(prices[str(gws[0])]) if gws else None

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
# Derivations
# ===============================

def _derive_p60(df: pd.DataFrame, p60_col: Optional[str]) -> pd.Series:
    if p60_col and p60_col in df.columns:
        return pd.to_numeric(df[p60_col], errors="coerce").clip(0.0, 1.0)
    if "p60" in df.columns:
        return pd.to_numeric(df["p60"], errors="coerce").clip(0.0, 1.0)
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
            "exp_pts_mean": g["exp_pts_mean"].astype(float).sum(),
            "exp_pts_var": pd.to_numeric(g.get("exp_pts_var", pd.Series([0.0] * len(g))), errors="coerce").fillna(0.0).sum(),
            "p60": _combine_prod_compl_one_minus(g["p60"]),
            "cs_prob": _combine_prod_compl_one_minus(g["cs_prob"]),
            "is_dgw": (len(g) > 1),
        }
        if "team_code" in g.columns:
            out["team_code"] = g["team_code"].iloc[0]
        if "player" in g.columns:
            out["player"] = g["player"].iloc[0]
        return pd.Series(out)

    agg = df.groupby(keys, as_index=False).apply(agg_fn)
    if isinstance(agg.columns, pd.MultiIndex):
        agg.columns = [c[0] if c[0] else c[1] for c in agg.columns]
    return agg

# ===============================
# Team code mapping
# ===============================

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

def _resolve_team_quota_key(
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

# ===============================
# Price resolution (master)
# ===============================

def _find_preds_price(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ["price", "now_price", "current_price", "now_cost", "cost"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if col in {"now_cost", "cost"}:
                s = s / 10.0
            if not s.isna().any():
                return s
    return None

def _resolve_prices_from_master_per_row(
    df: pd.DataFrame,
    master: dict,
    season: str,
    price_gw: Optional[int],
    on_missing: str,
) -> Tuple[pd.Series, pd.Series]:
    season_short = _to_short_season(season)

    def _get_primary(row):
        gw = int(price_gw if price_gw is not None else row["gw"])
        m = master.get(str(row["player_id"]))
        if not m:
            return None
        return _price_from_master(m, season_short, gw)

    vals = df.apply(_get_primary, axis=1)
    s = pd.to_numeric(vals, errors="coerce")
    missing_mask = s.isna()
    if not missing_mask.any():
        return s, missing_mask

    if on_missing == "use_earliest_in_season":
        def _earliest(pid):
            m = master.get(str(pid))
            return _earliest_price_in_season(m, season_short) if m else None
        fills = df.loc[missing_mask, "player_id"].map(_earliest)
        s.loc[missing_mask] = pd.to_numeric(fills, errors="coerce")
    elif on_missing == "use_preds":
        preds_price = _find_preds_price(df)
        if preds_price is None:
            raise ValueError("on-missing-price=use_preds, but no usable price column present in predictions.")
        s.loc[missing_mask] = preds_price.loc[missing_mask]
    elif on_missing in {"drop", "error"}:
        pass
    else:
        raise ValueError("Unknown --on-missing-price policy")

    return s, missing_mask

# ===============================
# Names resolution
# ===============================

def _ensure_player_names(df: pd.DataFrame, master: Optional[dict]) -> pd.DataFrame:
    if "player" in df.columns and df["player"].notna().any():
        return df
    if master is None:
        df["player"] = pd.NA
        return df
    def _get_name(pid: str) -> Optional[str]:
        entry = master.get(str(pid))
        return _name_from_master(entry) if entry else None
    df["player"] = df["player_id"].astype("string").map(_get_name)
    return df

# ===============================
# Builder
# ===============================

def build_optimizer_input(
    team_state_path: str,
    preds_path: str,
    out_path: str,
    fmt: str = "csv",  # parquet|csv|both
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
    # NEW (v1 guards & params)
    clamp_captain_uplift: bool = True,
    captain_multiplier: float = 2.0,
    tc_multiplier: float = 3.0,
    params_out: Optional[str] = None,
) -> List[str]:
    ts = _load_team_state(team_state_path)
    own_sell = _owned_sell_map(ts)
    df = _read_any(preds_path)

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
    mean_col = exp_mean_col or ("xPts" if "xPts" in df.columns else "exp_pts_mean")
    if mean_col not in df.columns:
        raise ValueError(f"Expected points mean column not found (looked for '{mean_col}'). Use --exp-mean-col.")
    df["exp_pts_mean"] = pd.to_numeric(df[mean_col], errors="coerce")

    if "exp_pts_var" in df.columns:
        df["exp_pts_var"] = pd.to_numeric(df["exp_pts_var"], errors="coerce").fillna(0.0)
    else:
        df["exp_pts_var"] = 0.0

    if cs_prob_col and cs_prob_col in df.columns:
        df["cs_prob"] = pd.to_numeric(df[cs_prob_col], errors="coerce").clip(0.0, 1.0)
    elif "__p_cs__" in df.columns:
        df["cs_prob"] = pd.to_numeric(df["__p_cs__"], errors="coerce").clip(0.0, 1.0)
    elif "team_prob_cs" in df.columns:
        df["cs_prob"] = pd.to_numeric(df["team_prob_cs"], errors="coerce").clip(0.0, 1.0)
    elif "cs_prob" in df.columns:
        df["cs_prob"] = pd.to_numeric(df["cs_prob"], errors="coerce").clip(0.0, 1.0)
    else:
        raise ValueError("Clean sheet probability not found.")
    df["p60"] = _derive_p60(df, p60_col)

    # DGW and aggregation
    df["is_dgw"] = _ensure_is_dgw(df)
    if not no_aggregate and df.duplicated(subset=["season", "gw", "player_id"], keep=False).any():
        df = _aggregate_player_gw(df)

    # Captain uplift (default = mean), then clamp at 0 if requested
    if captain_uplift_col and captain_uplift_col in df.columns:
        df["captain_uplift"] = pd.to_numeric(df[captain_uplift_col], errors="coerce")
    else:
        df["captain_uplift"] = df["exp_pts_mean"]
    if clamp_captain_uplift:
        df["captain_uplift"] = df["captain_uplift"].clip(lower=0.0)

    # team_quota_key
    df["team_quota_key"] = _resolve_team_quota_key(
        df,
        team_code_map_path=team_code_map_path,
        strict=strict_team_code,
        missing_out=missing_team_map_out,
    )

    # Prices & Names from master
    if not master_path:
        raise ValueError("This configuration expects --master for prices (latest ≤ GW).")
    master = _load_master(master_path)

    # Names: ensure 'player' present
    df = _ensure_player_names(df, master)

    price_series, missing_mask = _resolve_prices_from_master_per_row(
        df, master, str(df["season"].iloc[0]), price_gw, on_missing_price
    )

    if debug_missing_prices and missing_mask.any():
        dbg = df.loc[missing_mask, ["season", "gw", "player_id", "team_id", "player"]].copy()
        print("[debug] Missing master prices (first 20):")
        print(dbg.head(20).to_string(index=False))

    if missing_mask.any() and missing_price_out:
        review_cols = [c for c in ["season","gw","player_id","player","team_id","pos","exp_pts_mean","cs_prob","p60"] if c in df.columns]
        review = df.loc[missing_mask, review_cols].copy()
        review["price_lookup_gw"] = int(price_gw) if price_gw is not None else df.loc[missing_mask, "gw"]
        out_dir = os.path.dirname(missing_price_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        review.to_csv(missing_price_out, index=False)
        print(f"[info] Wrote {len(review)} rows with missing master price to {missing_price_out}")

    df["price"] = price_series
    if on_missing_price == "drop" and missing_mask.any():
        before = len(df)
        df = df.loc[~missing_mask].copy()
        print(f"[info] Dropped {before - len(df)} rows due to missing master price")

    # sell_price
    if strict_owned_sell:
        df["sell_price"] = df.apply(lambda r: own_sell.get(str(r["player_id"]), r["price"]), axis=1)
    else:
        df["sell_price"] = df["price"]

    if drop_rows_missing_core:
        before = len(df)
        df = df.dropna(subset=[
            "season", "gw", "player_id", "team_id", "pos",
            "price", "sell_price", "p60", "exp_pts_mean", "exp_pts_var", "cs_prob"
        ])
        after = len(df)
        if after < before:
            print(f"[info] Dropped {before - after} rows with missing core values")

    # Final table — include 'player' if present (right after player_id)
    out_cols = REQUIRED_OUT_COLS.copy()
    if "player" in df.columns:
        idx = out_cols.index("player_id") + 1
        out_cols = out_cols[:idx] + ["player"] + out_cols[idx:]
    output_df = df[out_cols].copy()

    output_df = _coerce_output_dtypes(output_df)
    if validate:
        _validate_contract(output_df)

    if preview > 0:
        print(output_df.head(preview).to_string(index=False))

    # Optionally write a small params file the solver can read
    if params_out:
        params = {
            "captain_multiplier": float(captain_multiplier),
            "triple_captain_multiplier": float(tc_multiplier),
            "clamp_captain_uplift": bool(clamp_captain_uplift),
        }
        os.makedirs(os.path.dirname(params_out), exist_ok=True) if os.path.dirname(params_out) else None
        with open(params_out, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print(f"OK (params): {params_out} -> {params}")

    # Write artifacts
    written: List[str] = []
    base, ext = os.path.splitext(out_path)

    def _write_parquet(path: str):
        eng = _detect_parquet_engine()
        if eng:
            output_df.to_parquet(path, index=False, engine=eng)
            print(f"OK (parquet/{eng}): {path}")
            written.append(path)
        else:
            csv_fb = os.path.splitext(path)[0] + ".csv"
            output_df.to_csv(csv_fb, index=False)
            print(f"[warn] No parquet engine; wrote CSV fallback: {csv_fb}")
            written.append(csv_fb)

    def _write_csv(path: str):
        output_df.to_csv(path, index=False)
        print(f"OK (csv): {path}")
        written.append(path)

    if fmt == "both":
        root = base if ext == "" else os.path.splitext(out_path)[0]
        _write_parquet(root + ".parquet")
        _write_csv(root + ".csv")
    elif fmt == "parquet":
        target = out_path if out_path.endswith(".parquet") else base + ".parquet"
        _write_parquet(target)
    elif fmt == "csv":
        target = out_path if out_path.endswith(".csv") else base + ".csv"
        _write_csv(target)
    else:
        raise ValueError("--format must be one of: parquet, csv, both")

    print(f"rows={len(output_df)} uniques={output_df[['season','gw','player_id']].drop_duplicates().shape[0]}")
    return written

# ===============================
# CLI
# ===============================

def main():
    ap = argparse.ArgumentParser(description="Build strict optimizer_input from team_state + expected_points")
    ap.add_argument("--team-state", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True, help="Output path or base name")
    ap.add_argument("--format", choices=["parquet", "csv", "both"], default="csv")

    ap.add_argument("--team-code-map", help="Path to master_teams.json")
    ap.add_argument("--strict-team-code", action="store_true",
                    help="Fail if any team_id cannot be mapped to a real team code")
    ap.add_argument("--missing-team-map-out", help="CSV path to write unmapped/mismatched teams")

    ap.add_argument("--captain-uplift-col")

    ap.add_argument("--gw-col", default="gw", help="GW column in preds (e.g., gw_orig)")
    ap.add_argument("--exp-mean-col", help="Expected points mean column (e.g., xPts)")
    ap.add_argument("--cs-prob-col", help="Clean sheet probability column (e.g., __p_cs__ or team_prob_cs)")
    ap.add_argument("--p60-col", help="Minutes probability column (e.g., p60)")

    ap.add_argument("--gw-fallback", type=int)
    ap.add_argument("--season-fallback")

    ap.add_argument("--master", required=True, help="Path to master_fpl.json for prices & names")
    ap.add_argument("--price-gw", type=int, help="Override GW to look up prices (default: each row's gw)")
    ap.add_argument("--on-missing-price",
                    choices=["error", "use_earliest_in_season", "use_preds", "drop"],
                    default="error",
                    help="Fallback policy when master price is missing")
    ap.add_argument("--missing-price-out", help="CSV path to write rows with missing master prices")

    ap.add_argument("--drop-rows-missing-core", action="store_true")
    ap.add_argument("--simple-sell", action="store_true", help="sell_price = price for everyone (ignore team_state)")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--preview", type=int, default=0, help="print first N rows")
    ap.add_argument("--debug-missing-prices", action="store_true")
    ap.add_argument("--no-aggregate", action="store_true", help="skip DGW aggregation (keep per-fixture rows)")

    # NEW (v1 guards & params)
    ap.add_argument("--no-clamp-captain-uplift", action="store_true",
                    help="Do not clamp captain_uplift at zero")
    ap.add_argument("--captain-multiplier", type=float, default=2.0,
                    help="Captain multiplier m (default 2.0)")
    ap.add_argument("--tc-multiplier", type=float, default=3.0,
                    help="Triple Captain multiplier (default 3.0)")
    ap.add_argument("--params-out", help="Write a small JSON with multipliers and guards")

    args = ap.parse_args()
    try:
        build_optimizer_input(
            team_state_path=args.team_state,
            preds_path=args.preds,
            out_path=args.out,
            fmt=args.format,
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
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
