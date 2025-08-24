#!/usr/bin/env python3
"""
calendar_builder.py  –  Batch-capable builder for player_minutes_calendar.csv
(Optionally include price via --include-price, reading master/seasonal JSONs or merged_gws)

Creates one “skinny” file per season with:
    fbref_id, fpl_id, gw_orig, date_played,
    team_id, player_id, minutes, is_active, is_starter, starter_source,
    [price], xp, total_points, bonus, bps, clean_sheets
    + rich per-player stats (xg, shots, tackles, saves, etc.)
    + fixture context (team, opponent_id, venue, was_home, FDR)

FDR: attached from features/<team-version>/... (prefer views/<SEASON>/fixture_calendar_with_fdr__<team-version>.csv,
     else join team_form.csv). We do NOT mutate home/away ids.

Price (when --include-price):
- Priority: seasonal JSON > master_fpl JSON > merged_gws.csv
- Join key: (player_id, gw_orig). merged_gws fallback uses its 'event' column.

Usage:
  python -m scripts.fbref_pipeline.integrate.calendar_builder --season 2025-2026 ^
    --fixtures-root data/processed/registry/fixtures ^
    --fbref-root "data/processed/fbref/ENG-Premier League" ^
    --fpl-root data/processed/fpl ^
    --features-root data/processed/registry/features ^
    --team-version latest ^
    --include-price ^
    --price-master "data/processed/registry/fpl/master_fpl.json" ^
    --price-seasonal "data/processed/registry/fpl/2025-2026_prices.json" ^
    --force
"""
from __future__ import annotations
import argparse, logging, json
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

# ───────────────────────────── Canonical output schema (price is optional) ─────────────────────────
OUT_COLS = [
    # Fixture identity / join keys
    "fbref_id", "fpl_id", "gw_orig", "date_played",
    "team_id", "opponent_id", "team", "venue", "was_home", "fdr_home", "fdr_away",

    # Player identity & availability
    "player_id", "player", "pos",
    "minutes", "days_since_last", "is_active", "is_starter", "starter_source",

    # FPL enrichments (inputs & outcomes) — price is included only if --include-price
    "price", "xp", "total_points", "bonus", "bps", "clean_sheets",

    # Match result context
    "gf", "ga",

    # Attacking contribution
    "xg", "npxg", "xag", "shots", "sot", "gls", "ast", "pkatt", "pk_scored", "pk_won",

    # Defensive contribution
    "blocks", "tkl", "int", "clr", "recoveries",

    # Goalkeeper stats
    "saves", "sot_against", "save_pct",

    # Discipline
    "yellow_crd", "red_crd", "own_goals",
]

def write_empty_minutes_calendar(season_dir: Path, include_price: bool) -> None:
    out_fp = season_dir / "player_minutes_calendar.csv"
    season_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in OUT_COLS if include_price or c != "price"]
    pd.DataFrame(columns=cols).to_csv(out_fp, index=False)
    logging.info("%s • wrote EMPTY player_minutes_calendar.csv", season_dir.name)

# ───────────────────────────── Season key helpers ─────────────────────────────

def season_long_to_short(season: str) -> str:
    """'2025-2026' -> '2025-26'; if already short ('2023-24'), return as-is."""
    s = str(season)
    if len(s) == 7 and s[4] == "-":  # e.g. 2023-24
        return s
    if len(s) == 9 and s[4] == "-":  # 2025-2026
        y1 = s[:4]; y2 = s[-2:]
        return f"{y1}-{y2}"
    return s

# ───────────────────────────── FDR attachment helpers ─────────────────────────

def _read_calendar_base(season_dir: Path) -> pd.DataFrame:
    """Read the PURE fixtures calendar (expects gw_orig present)."""
    fp = season_dir / "fixture_calendar.csv"
    df = pd.read_csv(fp, low_memory=False, parse_dates=["date_played"])
    for c in ["team_id","home_id","away_id","opponent_id","fbref_id","fpl_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()
    return df

def _try_read_fdr_view(features_root: Path, season: str, team_version: str, views_subdir: str="views") -> Optional[pd.DataFrame]:
    view_fp = Path(features_root) / views_subdir / season / f"fixture_calendar_with_fdr__{team_version}.csv"
    if not view_fp.exists():
        return None
    view = pd.read_csv(view_fp, parse_dates=["date_played"], low_memory=False)
    for c in ["home_id","away_id"]:
        if c in view.columns:
            view[c] = view[c].astype("string").str.lower()
    cols = ["date_played","home_id","away_id","fdr_home","fdr_away"]
    if not set(cols).issubset(view.columns):
        return None
    return view[cols].drop_duplicates(["date_played","home_id","away_id"])

def _try_read_team_form(features_root: Path, season: str, team_version: str) -> Optional[pd.DataFrame]:
    tf_fp = Path(features_root) / team_version / season / "team_form.csv"
    if not tf_fp.exists():
        return None
    tf = pd.read_csv(tf_fp, parse_dates=["date_played"], low_memory=False)
    if {"date_played","home_id","away_id","fdr_home","fdr_away"}.issubset(tf.columns):
        for c in ["home_id","away_id"]:
            tf[c] = tf[c].astype("string").str.lower()
        return tf[["date_played","home_id","away_id","fdr_home","fdr_away"]].drop_duplicates(["date_played","home_id","away_id"])
    return tf

def _attach_fdr(cal: pd.DataFrame, features_root: Path, season: str, team_version: str, views_subdir: str="views") -> pd.DataFrame:
    cal = cal.copy()
    if all(c in cal.columns for c in ["fdr_home","fdr_away"]):
        return cal
    view = _try_read_fdr_view(features_root, season, team_version, views_subdir)
    if view is not None and {"home_id","away_id","date_played"}.issubset(cal.columns):
        before = len(cal)
        cal = cal.merge(view, on=["date_played","home_id","away_id"], how="left", validate="m:1")
        assert len(cal) == before, "Row count changed after FDR view merge"
        return cal
    tf = _try_read_team_form(features_root, season, team_version)
    if tf is None:
        return cal
    if {"date_played","home_id","away_id","fdr_home","fdr_away"}.issubset(tf.columns) and {"date_played","home_id","away_id"}.issubset(cal.columns):
        before = len(cal)
        cal = cal.merge(tf[["date_played","home_id","away_id","fdr_home","fdr_away"]],
                        on=["date_played","home_id","away_id"], how="left", validate="m:1")
        assert len(cal) == before, "Row count changed after FDR (fixture-level) merge"
        return cal
    if {"fpl_id","home_id","away_id"}.issubset(cal.columns) and {"fpl_id","team_id","fdr_home","fdr_away"}.issubset(tf.columns):
        before = len(cal)
        tmp = tf.copy()
        tmp["team_id"] = tmp["team_id"].astype("string").str.lower()
        home = tmp[["fpl_id","team_id","fdr_home"]].rename(columns={"team_id":"home_id"})
        home["fpl_id"] = home["fpl_id"].astype(str).str.lower()
        cal["fpl_id"] = cal["fpl_id"].astype(str).str.lower()
        cal = cal.merge(home, on=["fpl_id","home_id"], how="left", validate="m:1")
        away = tmp[["fpl_id","team_id","fdr_away"]].rename(columns={"team_id":"away_id"})
        cal = cal.merge(away, on=["fpl_id","away_id"], how="left", validate="m:1")
        assert len(cal) == before, "Calendar row count changed after FDR (per-team) merge"
        return cal
    return cal

def load_fixture_calendar(season_dir: Path, *, features_root: Path, team_version: str, views_subdir: str) -> pd.DataFrame:
    df = _read_calendar_base(season_dir)
    season_key = season_dir.name
    df = _attach_fdr(df, features_root, season_key, team_version, views_subdir)
    if "opponent_id" not in df.columns or df["opponent_id"].isna().any():
        if {"team_id","home_id","away_id"}.issubset(df.columns):
            mask = (df["team_id"].astype(str) == df["home_id"].astype(str))
            opp = np.where(mask, df["away_id"], df["home_id"])
            df["opponent_id"] = df.get("opponent_id", pd.Series(index=df.index, dtype="string"))
            df["opponent_id"] = df["opponent_id"].fillna(pd.Series(opp, index=df.index)).astype("string")
    if "date_played" in df.columns:
        df["date_played"] = pd.to_datetime(df["date_played"]).dt.tz_localize(None).dt.floor("D")
    return df

# ───────────────────────────── FBref loaders ─────────────────────────────

def load_minutes(season_dir: Path, fbref_root: Path) -> pd.DataFrame:
    season_key = season_dir.name
    summary_fp = fbref_root / season_key / "player_match" / "summary.csv"
    keeper_fp  = fbref_root / season_key / "player_match" / "keepers.csv"
    def_fp     = fbref_root / season_key / "player_match" / "defense.csv"
    misc_fp    = fbref_root / season_key / "player_match" / "misc.csv"

    df = pd.read_csv(
        summary_fp,
        usecols=[
            "game_id","player_id","player","min","team_id",
            "crdy","crdr","fpl_pos","gls","ast","xg","npxg","xag","pkatt","pk","sh","sot"
        ],
        low_memory=False
    ).rename(columns={
        "game_id":"fbref_id","min":"minutes","crdy":"yellow_crd","crdr":"red_crd",
        "fpl_pos":"pos","pk":"pk_scored","sh":"shots"
    })

    df_gk = pd.read_csv(
        keeper_fp, usecols=["game_id","player_id","team_id","sota","saves","save"],
        low_memory=False
    ).rename(columns={"game_id":"fbref_id","sota":"sot_against","save":"save_pct"})

    df_def = pd.read_csv(
        def_fp, usecols=["game_id","player_id","team_id","blocks","tklw","int","clr"],
        low_memory=False
    ).rename(columns={"game_id":"fbref_id","tklw":"tkl"})

    df_misc = pd.read_csv(
        misc_fp, usecols=["game_id","player_id","team_id","recov","pkwon","og"],
        low_memory=False
    ).rename(columns={"game_id":"fbref_id","recov":"recoveries","pkwon":"pk_won","og":"own_goals"})

    df = df.merge(df_gk,  on=["fbref_id","player_id","team_id"], how="left")
    df = df.merge(df_def, on=["fbref_id","player_id","team_id"], how="left")
    df = df.merge(df_misc,on=["fbref_id","player_id","team_id"], how="left")

    for col in ("sot_against","saves","save_pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for c in ["player_id","team_id","fbref_id"]:
        df[c] = df[c].astype(str).str.lower()

    return df

# ───────────────────────────── FPL loaders (xP, starts, points) ───────

def _coerce_bool_int8(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["1","true","t","y","yes"]).astype("Int8")

def _coerce_bool(series: pd.Series) -> pd.Series:
    s = series.copy()
    asnum = pd.to_numeric(s, errors="coerce")
    mask_num = asnum.notna()
    out = pd.Series(False, index=s.index, dtype="boolean")
    out.loc[mask_num] = asnum.loc[mask_num] > 0
    mask_str = ~mask_num
    str_true = s.loc[mask_str].astype(str).str.strip().str.lower().isin(["1","true","t","y","yes","started","start"])
    out.loc[mask_str] = str_true
    return out.fillna(False).astype(bool)

def _coerce_01(series: pd.Series) -> pd.Series:
    return _coerce_bool(series).astype("uint8")

def _prep_fixture_keys(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "kickoff_time" not in df.columns: return None
    df = df.copy()
    df["date_played"] = pd.to_datetime(df["kickoff_time"], utc=True).dt.tz_convert(None).dt.floor("D")
    if "was_home" in df.columns:
        df["was_home"] = _coerce_bool_int8(df["was_home"])
    elif "venue" in df.columns:
        df["was_home"] = (df["venue"].astype(str).str.strip().str.lower().eq("home")).astype("Int8")
    else:
        return None
    if "fbref_id" in df.columns:
        df["fbref_id"] = df["fbref_id"].astype(str).str.lower()
    df["player_id"] = df["player_id"].astype(str).str.lower()
    return df

def load_fpl_fixture_xp(fpl_root: Path, season: str) -> Optional[pd.DataFrame]:
    gws_fp = fpl_root / season / "gws" / "merged_gws.csv"
    if not gws_fp.exists():
        logging.warning("FPL file missing: %s", gws_fp); return None
    df = pd.read_csv(gws_fp, low_memory=False)
    df = _prep_fixture_keys(df)
    if df is None:
        logging.warning("%s: missing kickoff_time/venue for fixture keys", gws_fp); return None
    xp_col = next((c for c in ["xP","ep_this","expected_points","exp_points","xp_this"] if c in df.columns), None)
    if xp_col is None:
        logging.warning("%s: no xP/ep_this column; xp will be NaN", gws_fp); return None
    df["xp"] = pd.to_numeric(df[xp_col], errors="coerce")
    use = ["player_id","date_played","was_home","xp"]
    if "fbref_id" in df.columns: use.insert(1, "fbref_id")
    return (df[use].sort_values(["date_played"])
                 .drop_duplicates(subset=use[:-1], keep="last")
                 .reset_index(drop=True))

def load_fpl_fixture_starts(fpl_root: Path, season: str) -> Optional[pd.DataFrame]:
    gws_fp = fpl_root / season / "gws" / "merged_gws.csv"
    if not gws_fp.exists():
        logging.warning("FPL file missing: %s", gws_fp); return None
    df = pd.read_csv(gws_fp, low_memory=False)
    df = _prep_fixture_keys(df)
    if df is None:
        logging.warning("%s: missing kickoff_time/venue for fixture keys", gws_fp); return None
    starts_col = next((c for c in ["starts","was_starter","started"] if c in df.columns), None)
    if starts_col is None:
        return None
    df["is_starter"] = _coerce_01(df[starts_col])
    use = ["player_id","date_played","was_home","is_starter"]
    if "fbref_id" in df.columns: use.insert(1, "fbref_id")
    return (df[use].sort_values(["date_played"])
                 .drop_duplicates(subset=use[:-1], keep="last")
                 .reset_index(drop=True))

def load_fpl_fixture_points(fpl_root: Path, season: str) -> Optional[pd.DataFrame]:
    gws_fp = fpl_root / season / "gws" / "merged_gws.csv"
    if not gws_fp.exists():
        logging.warning("FPL file missing: %s", gws_fp); return None
    df = pd.read_csv(gws_fp, low_memory=False)
    df = _prep_fixture_keys(df)
    if df is None:
        logging.warning("%s: missing kickoff_time/venue for fixture keys", gws_fp); return None
    present = [c for c in ["total_points","bonus","bps","clean_sheets"] if c in df.columns]
    if not present:
        logging.warning("%s: no total_points/bonus/bps/clean_sheets", gws_fp); return None
    out = df[["player_id","date_played","was_home"] + present].copy()
    if "fbref_id" in df.columns:
        out["fbref_id"] = df["fbref_id"].astype(str).str.lower()
        out = out[["player_id","fbref_id","date_played","was_home"] + present]
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    key_len = out.shape[1] - len(present)
    subset = out.columns[:key_len].tolist()
    return (
        out.sort_values(["date_played"])
           .drop_duplicates(subset=subset, keep="last")
           .reset_index(drop=True))

# ───────────────────────────── Price loaders (by GW) ──────────────────────────

def _coerce_gw(v: Any) -> Optional[int]:
    try:
        return int(str(v).strip())
    except Exception:
        return None

def load_prices_master_by_gw(master_fp: Path, season_key: str) -> Optional[pd.DataFrame]:
    """master_fpl.json: {player_id: {'prices': {'2023-24': {'GW': price, ...}}}}"""
    if master_fp is None or not master_fp.exists():
        return None
    try:
        data = json.loads(master_fp.read_text("utf-8"))
    except Exception:
        logging.exception("Failed to read master_fpl JSON at %s", master_fp)
        return None

    target_keys = {season_key, season_long_to_short(season_key)}
    rows: List[Dict[str, Any]] = []
    for pid, pinfo in (data or {}).items():
        prices = (pinfo or {}).get("prices") or {}
        hit_key = next((k for k in prices.keys() if k in target_keys), None)
        if not hit_key:
            continue
        for gw, price in (prices.get(hit_key) or {}).items():
            gw_i = _coerce_gw(gw)
            if gw_i is None: continue
            try:
                val = float(price)
            except Exception:
                continue
            rows.append({"player_id": str(pid).lower(), "gw_orig": gw_i, "price": val})

    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=["player_id","gw_orig"])
    df["gw_orig"] = df["gw_orig"].astype(int)
    df["player_id"] = df["player_id"].astype(str).str.lower()
    return df.drop_duplicates(["player_id","gw_orig"], keep="last").reset_index(drop=True)

def load_prices_seasonal_by_gw(seasonal_fp: Path) -> Optional[pd.DataFrame]:
    """seasonal_prices.json: {player_id: {'GW': price, ...}} (assumed for the current season)"""
    if seasonal_fp is None or not seasonal_fp.exists():
        return None
    try:
        data = json.loads(seasonal_fp.read_text("utf-8"))
    except Exception:
        logging.exception("Failed to read seasonal price JSON at %s", seasonal_fp)
        return None

    rows: List[Dict[str, Any]] = []
    for pid, gw_map in (data or {}).items():
        if not isinstance(gw_map, dict):
            continue
        for gw, price in gw_map.items():
            gw_i = _coerce_gw(gw)
            if gw_i is None: continue
            try:
                val = float(price)
            except Exception:
                continue
            rows.append({"player_id": str(pid).lower(), "gw_orig": gw_i, "price": val})

    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=["player_id","gw_orig"])
    df["gw_orig"] = df["gw_orig"].astype(int)
    df["player_id"] = df["player_id"].astype(str).str.lower()
    return df.drop_duplicates(["player_id","gw_orig"], keep="last").reset_index(drop=True)

def load_prices_gws_fallback_by_gw(fpl_root: Path, season: str) -> Optional[pd.DataFrame]:
    """Fallback: merged_gws.csv -> (player_id, event -> gw_orig, now_cost/value/price)."""
    gws_fp = fpl_root / season / "gws" / "merged_gws.csv"
    if not gws_fp.exists():
        return None
    try:
        df = pd.read_csv(gws_fp, low_memory=False)
    except Exception:
        logging.exception("Failed reading %s", gws_fp); return None
    # try to find event column
    gw_col = next((c for c in ["event","gw","GW"] if c in df.columns), None)
    if gw_col is None:
        return None
    pcol = next((c for c in ["price","now_cost","value"] if c in df.columns), None)
    if pcol is None:
        return None
    out = df[["element" if "element" in df.columns else "player_id", gw_col, pcol]].copy()
    out.rename(columns={gw_col: "gw_orig", pcol: "price"}, inplace=True)
    if "element" in out.columns:
        out.rename(columns={"element": "player_id"}, inplace=True)
    out["player_id"] = out["player_id"].astype(str).str.lower()
    out["gw_orig"] = pd.to_numeric(out["gw_orig"], errors="coerce").astype("Int64")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    # normalize now_cost/value to real price if needed
    if pcol in ["now_cost","value"]:
        out["price"] = out["price"] / 10.0
    out = out.dropna(subset=["player_id","gw_orig"])
    out["gw_orig"] = out["gw_orig"].astype(int)
    return (out.sort_values(["player_id","gw_orig"])
               .drop_duplicates(["player_id","gw_orig"], keep="last")
               .reset_index(drop=True))

# ───────────────────────────── Main builder ─────────────────────────────

def build_minutes_calendar(
    season_dir: Path,
    fbref_root: Path,
    fpl_root: Path,
    features_root: Path,
    team_version: str,
    views_subdir: str,
    include_price: bool,
    price_master: Optional[Path],
    price_seasonal: Optional[Path],
    drop_missing_price: bool,
    force: bool = False,
    create_empty: bool = False,
) -> None:
    out_fp = season_dir / "player_minutes_calendar.csv"
    if out_fp.exists() and not force:
        logging.info("%s exists – skipping", out_fp.name)
        return

    season_key = season_dir.name
    # ---- Inputs presence gate (before reading heavy files)
    fix_fp = season_dir / "fixture_calendar.csv"
    summary_fp = fbref_root / season_key / "player_match" / "summary.csv"
    keeper_fp  = fbref_root / season_key / "player_match" / "keepers.csv"
    def_fp     = fbref_root / season_key / "player_match" / "defense.csv"
    misc_fp    = fbref_root / season_key / "player_match" / "misc.csv"

    missing = [str(p) for p in [fix_fp, summary_fp, keeper_fp, def_fp, misc_fp] if not p.exists()]
    if missing:
        msg = f"{season_key} • missing inputs → " + ", ".join(missing)
        if create_empty:
            logging.warning(msg + " — creating EMPTY player_minutes_calendar.csv")
            write_empty_minutes_calendar(season_dir, include_price)
            return
        else:
            logging.warning(msg + " — skipped (use --create-empty to write empty)")
            return

    # Load fixture calendar (+FDR if available)
    cal = load_fixture_calendar(season_dir, features_root=features_root, team_version=team_version, views_subdir=views_subdir)

    # Load player minutes directly
    minutes = load_minutes(season_dir, fbref_root)

    # Merge on both 'fbref_id' and 'team_id'
    merged = minutes.merge(cal, on=["fbref_id", "team_id"], how="left")

    # Add was_home (canonicalized)
    merged["was_home"] = (merged["venue"].astype(str).str.strip().str.title().eq("Home")).astype("Int8")

    # Integrity check: remove rows without fixture match (if any)
    missing_fixtures = merged["date_played"].isna().sum()
    if missing_fixtures:
        logging.warning(f"[{season_key}] {missing_fixtures} rows with missing fixture data dropped")
        merged.dropna(subset=["date_played"], inplace=True)

    # Flag active (minutes played > 0)
    merged["is_active"] = np.where(pd.to_numeric(merged["minutes"], errors="coerce").gt(0), 1, 0).astype("uint8")

    # Days since last match per player
    merged = merged.sort_values(["player_id", "date_played"])
    merged["days_since_last"] = (
        merged.groupby("player_id")["date_played"].diff().dt.days.fillna(0).astype(int)
    )

    # Normalize IDs for joins
    for c in ["player_id","fbref_id","team_id"]:
        if c in merged.columns:
            merged[c] = merged[c].astype(str).str.lower()
    merged["date_played"] = pd.to_datetime(merged["date_played"]).dt.floor("D")
    if "gw_orig" in merged.columns:
        merged["gw_orig"] = pd.to_numeric(merged["gw_orig"], errors="coerce").astype("Int64")

    # ── PRICE (optional) ────────────────────────────────────────────────────
    if include_price:
        pframes: List[pd.DataFrame] = []
        # 1) seasonal JSON (highest priority)
        ps = load_prices_seasonal_by_gw(price_seasonal) if price_seasonal else None
        if ps is not None:
            ps["_prio"] = 3
            pframes.append(ps)
        # 2) master JSON (mid)
        pm = load_prices_master_by_gw(price_master, season_key) if price_master else None
        if pm is not None:
            pm["_prio"] = 2
            pframes.append(pm)
        # 3) merged_gws fallback (lowest)
        pg = load_prices_gws_fallback_by_gw(fpl_root, season_key)
        if pg is not None:
            pg["_prio"] = 1
            pframes.append(pg)

        if pframes:
            prices_all = pd.concat(pframes, ignore_index=True)
            prices_all = (prices_all.sort_values(["player_id","gw_orig","_prio"], ascending=[True, True, False])
                                     .drop_duplicates(["player_id","gw_orig"], keep="first")
                                     .drop(columns=["_prio"]))
            before = len(merged)
            merged = merged.merge(prices_all, on=["player_id","gw_orig"], how="left", validate="m:1")
            assert len(merged) == before, "Row count changed after price merges"
            logging.info("[%s] Price coverage: %.3f", season_key, merged["price"].notna().mean())
        else:
            merged["price"] = np.nan
            logging.warning("[%s] No price sources available; 'price' set to NaN", season_key)
    else:
        if "price" in merged.columns:
            merged = merged.drop(columns=["price"], errors="ignore")

    # ── xP ──────────────────────────────────────────────────────────────────
    fpl_xp = load_fpl_fixture_xp(fpl_root, season_key)
    if fpl_xp is not None:
        xp_keys = ["player_id","date_played","was_home"]
        if "fbref_id" in fpl_xp.columns and "fbref_id" in merged.columns:
            xp_keys.insert(1, "fbref_id")
        before = len(merged)
        merged = merged.merge(fpl_xp, on=xp_keys, how="left", validate="m:1")
        assert len(merged) == before, "Row count changed after xp merge"
    else:
        merged["xp"] = np.nan

    # ── is_starter + source ────────────────────────────────────────────────
    starter_source = pd.Series("fallback", index=merged.index, dtype="string")
    fpl_starts = load_fpl_fixture_starts(fpl_root, season_key)
    if fpl_starts is not None:
        st_keys = ["player_id","date_played","was_home"]
        if "fbref_id" in fpl_starts.columns and "fbref_id" in merged.columns:
            st_keys.insert(1, "fbref_id")
        before = len(merged)
        merged = merged.merge(fpl_starts, on=st_keys, how="left", validate="m:1")
        assert len(merged) == before, "Row count changed after is_starter merge"
        has_fpl = merged["is_starter"].notna()
        starter_source.loc[has_fpl] = "fpl"
        merged["is_starter"] = merged["is_starter"].fillna(0).astype("uint8")
    else:
        merged["is_starter"] = 1
        merged["is_starter"] = merged["is_starter"].astype("uint8")
        starter_source[:] = "fallback"
        logging.warning("[%s] No starts column; 'is_starter' defaulted to 1 (fallback)", season_key)

    # GK backfill: if GK and minutes>0, force starter=1 and mark as imputed
    if "pos" in merged.columns:
        is_gk = merged["pos"].astype(str).str.upper().str.contains("GK", na=False)
        gk_mask = is_gk & (pd.to_numeric(merged["minutes"], errors="coerce") > 0) & (merged["is_starter"] == 0)
        if gk_mask.any():
            merged.loc[gk_mask, "is_starter"] = 1
            starter_source.loc[gk_mask] = "imputed"

    merged["starter_source"] = starter_source.astype("string")

    # ── total_points, bonus, bps, clean_sheets ──────────────────────────────
    fpl_pts = load_fpl_fixture_points(fpl_root, season_key)
    if fpl_pts is not None:
        pt_keys = ["player_id","date_played","was_home"]
        if "fbref_id" in fpl_pts.columns and "fbref_id" in merged.columns:
            pt_keys.insert(1, "fbref_id")
        before = len(merged)
        merged = merged.merge(fpl_pts, on=pt_keys, how="left", validate="m:1")
        assert len(merged) == before, "Row count changed after points merge"
        for c in ["total_points","bonus","bps","clean_sheets"]:
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")
    else:
        for c in ["total_points","bonus","bps","clean_sheets"]:
            merged[c] = np.nan

    # ── Drop rows missing price (if requested) ──────────────────────────────
    if include_price and drop_missing_price:
        if "price" not in merged.columns:
            logging.warning("[%s] --drop-missing-price set but price column absent.", season_key)
        else:
            n_before = len(merged)
            n_miss = int(merged["price"].isna().sum())
            if n_miss > 0:
                merged = merged[merged["price"].notna()].copy()
                logging.warning("[%s] Dropped %d rows with missing price (%.2f%%). Kept %d.",
                                season_key, n_miss, 100.0 * n_miss / max(n_before, 1), len(merged))

    # ── Output (enforce canonical order; keep only known cols) ──────────────
    out_cols = [c for c in OUT_COLS if include_price or c != "price"]
    cols = [c for c in out_cols if c in merged.columns]
    merged[cols].to_csv(out_fp, index=False)
    logging.info("✅ %s written (%d rows)  | include_price=%s", out_fp.name, len(merged), include_price)

# ───────────────────────────── Batch & CLI ─────────────────────────────

def run_batch(
    seasons: List[str],
    fixtures_root: Path,
    fbref_root: Path,
    fpl_root: Path,
    features_root: Path,
    team_version: str,
    views_subdir: str,
    include_price: bool,
    price_master: Optional[Path],
    price_seasonal: Optional[Path],
    drop_missing_price: bool,
    force: bool,
    create_empty: bool,
) -> None:
    for season in seasons:
        season_dir = fixtures_root / season
        if not season_dir.is_dir():
            if create_empty:
                logging.warning("%s • season directory missing — creating EMPTY file", season)
                season_dir.mkdir(parents=True, exist_ok=True)
                write_empty_minutes_calendar(season_dir, include_price)
                continue
            else:
                logging.warning("Season %s directory missing – skipped", season)
                continue
        try:
            build_minutes_calendar(
                season_dir, fbref_root, fpl_root,
                features_root, team_version, views_subdir,
                include_price, price_master, price_seasonal, drop_missing_price,
                force=force, create_empty=create_empty
            )
        except Exception:
            logging.exception("❌ %s failed", season)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/registry/fixtures"),
                    help="Root dir containing season subfolders")
    ap.add_argument("--fbref-root", type=Path, default=Path("data/processed/fbref/ENG-Premier League"),
                    help="FBref league root (for JSONs and player_match)")
    ap.add_argument("--fpl-root", type=Path, default=Path("data/processed/fpl"),
                    help="FPL processed root (contains <season>/gws/merged_gws.csv)")
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"),
                    help="Features root (contains versioned team_form and views/)")
    ap.add_argument("--team-version", default="latest",
                    help="Team features version to join FDR from (e.g., latest or v7)")
    ap.add_argument("--views-subdir", default="views",
                    help="Subdir under features/ where fixture+FDR views live")
    # Price options
    ap.add_argument("--include-price", action="store_true",
                    help="Include a 'price' column by merging from JSON/merged_gws.")
    ap.add_argument("--price-master", type=Path, default=None,
                    help="Path to master_fpl.json (player_id -> prices[season][gw]=price).")
    ap.add_argument("--price-seasonal", type=Path, default=None,
                    help="Path to seasonal prices JSON (player_id -> {gw: price}) for the season.")
    ap.add_argument("--drop-missing-price", action="store_true",
                    help="If set with --include-price, drop rows where price is NaN after merges.")
    ap.add_argument("--season", help="e.g., 2025-2026; omit to process all seasons under fixtures-root")
    ap.add_argument("--create-empty", action="store_true",
                    help="If inputs are missing, still write an empty player_minutes_calendar.csv")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    if args.season:
        seasons = [args.season]
    else:
        seasons = sorted(d.name for d in args.fixtures_root.iterdir() if d.is_dir())
        if not seasons:
            logging.error("No seasons found under %s", args.fixtures_root); return

    run_batch(
        seasons,
        args.fixtures_root,
        args.fbref_root,
        args.fpl_root,
        args.features_root,
        args.team_version,
        args.views_subdir,
        args.include_price,
        args.price_master,
        args.price_seasonal,
        args.drop_missing_price,
        args.force,
        create_empty=args.create_empty,
    )

if __name__ == "__main__":
    main()
