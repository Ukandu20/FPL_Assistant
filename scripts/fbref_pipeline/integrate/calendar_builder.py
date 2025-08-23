#!/usr/bin/env python3
"""
calendar_builder.py  –  Batch-capable builder for player_minutes_calendar.csv

Creates one “skinny” file per season with:
    fbref_id, fpl_id, gw_orig, date_played,
    team_id, player_id, minutes, is_active, is_starter, starter_source,
    price, xp, total_points, bonus, bps, clean_sheets

Notes:
- Any lingering rows with missing price are DROPPED before write.
- If FPL 'starts' column is missing for a season, we default is_starter=1 (fallback).
- Goalkeepers (pos contains 'GK') with minutes>0 are forced to is_starter=1 (imputed).

Usage:
  ▸ Single season
      python -m scripts.fbref_pipeline.integrate.calendar_builder --season 2024-2025
  ▸ All seasons under fixtures-root
      python -m scripts.fbref_pipeline.integrate.calendar_builder
"""
from __future__ import annotations
import argparse, logging
import json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

# ───────────────────────────── Canonical output schema ─────────────────────────
OUT_COLS = [
    # Fixture identity / join keys
    "fbref_id", "fpl_id", "gw_orig", "date_played",
    "team_id", "opponent_id", "team", "venue", "was_home", "fdr_home", "fdr_away",

    # Player identity & availability
    "player_id", "player", "pos",
    "minutes", "days_since_last", "is_active", "is_starter", "starter_source",

    # FPL enrichments (inputs & outcomes)
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

def write_empty_minutes_calendar(season_dir: Path) -> None:
    out_fp = season_dir / "player_minutes_calendar.csv"
    season_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OUT_COLS).to_csv(out_fp, index=False)
    logging.info("%s • wrote EMPTY player_minutes_calendar.csv", season_dir.name)

# ───────────────────────────── FBref loaders ─────────────────────────────

def load_fixture_calendar(season_dir: Path) -> pd.DataFrame:
    """Reads data/processed/fixtures/<season>/fixture_calendar.csv"""
    fp = season_dir / "fixture_calendar.csv"
    return pd.read_csv(
        fp,
        usecols=[
            "fbref_id",
            "fpl_id",
            "gw_orig",
            "date_played",
            "team_id",
            "team",
            "venue",
            "gf",
            "ga",
            "fdr_home",
            "fdr_away",
        ],
    )

def load_minutes(season_dir: Path, fbref_root: Path) -> pd.DataFrame:
    """Reads player_match summary/keepers/defense/misc; merges into one frame."""
    season_key = season_dir.name
    summary_fp = fbref_root / season_key / "player_match" / "summary.csv"
    keeper_fp  = fbref_root / season_key / "player_match" / "keepers.csv"
    def_fp     = fbref_root / season_key / "player_match" / "defense.csv"
    misc_fp    = fbref_root / season_key / "player_match" / "misc.csv"

    # Load outfield
    df = pd.read_csv(
        summary_fp,
        usecols=[
            "game_id","player_id","player","min","team_id",
            "crdy","crdr","fpl_pos","gls","ast","xg","npxg","xag","pkatt","pk","sh","sot"
        ]
    ).rename(columns={
        "game_id":"fbref_id","min":"minutes","crdy":"yellow_crd","crdr":"red_crd",
        "fpl_pos":"pos","pk":"pk_scored","sh":"shots"
    })

    # Keepers
    df_gk = pd.read_csv(
        keeper_fp, usecols=["game_id","player_id","team_id","sota","saves","save"]
    ).rename(columns={"game_id":"fbref_id","sota":"sot_against","save":"save_pct"})

    # Defense
    df_def = pd.read_csv(
        def_fp, usecols=["game_id","player_id","team_id","blocks","tklw","int","clr"]
    ).rename(columns={"game_id":"fbref_id","tklw":"tkl"})

    # Misc
    df_misc = pd.read_csv(
        misc_fp, usecols=["game_id","player_id","team_id","recov","pkwon","og"]
    ).rename(columns={"game_id":"fbref_id","recov":"recoveries","pkwon":"pk_won","og":"own_goals"})

    # Merge all
    df = df.merge(df_gk,  on=["fbref_id","player_id","team_id"], how="left")
    df = df.merge(df_def, on=["fbref_id","player_id","team_id"], how="left")
    df = df.merge(df_misc,on=["fbref_id","player_id","team_id"], how="left")

    # Fill GK stats with 0 for non-GKs
    for col in ("sot_against","saves","save_pct"):
        df[col] = df[col].fillna(0)
    return df

# ───────────────────────────── FPL loaders (price, xP, starts, points) ───────

def _coerce_bool_int8(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["1","true","t","y","yes"]).astype("Int8")

def _coerce_bool(series: pd.Series) -> pd.Series:
    s = series.copy()
    asnum = pd.to_numeric(s, errors="coerce")
    mask_num = asnum.notna()
    out = pd.Series(False, index=s.index, dtype="boolean")
    out.loc[mask_num] = asnum.loc[mask_num] > 0
    mask_str = ~mask_num
    str_true = s.loc[mask_str].astype(str).str.strip().str.lower().isin(
        ["1","true","t","y","yes","started","start"]
    )
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
        df["fbref_id"] = df["fbref_id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    return df

def load_fpl_fixture_prices(fpl_root: Path, season: str) -> Optional[pd.DataFrame]:
    gws_fp = fpl_root / season / "gws" / "merged_gws.csv"
    if not gws_fp.exists():
        logging.warning("FPL file missing: %s", gws_fp); return None
    df = pd.read_csv(gws_fp, low_memory=False)
    df = _prep_fixture_keys(df)
    if df is None:
        logging.warning("%s: missing kickoff_time/venue for fixture keys", gws_fp); return None
    if   "price"    in df.columns: df["price"] = df["price"]
    elif "value"    in df.columns: df["price"] = df["value"]
    elif "now_cost" in df.columns: df["price"] = df["now_cost"] / 10.0
    else:
        logging.warning("%s: no price/value/now_cost", gws_fp); return None
    use = ["player_id","date_played","was_home","price"]
    if "fbref_id" in df.columns: use.insert(1, "fbref_id")
    return (df[use].sort_values(["date_played"])
                 .drop_duplicates(subset=use[:-1], keep="last")
                 .reset_index(drop=True))

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
        out["fbref_id"] = df["fbref_id"].astype(str)
        out = out[["player_id","fbref_id","date_played","was_home"] + present]
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    key_len = out.shape[1] - len(present)
    subset = out.columns[:key_len].tolist()
    return (out.sort_values(["date_played"])
                .drop_duplicates(subset=subset, keep="last")
                .reset_index(drop=True))

# ───────────────────────────── Main builder ─────────────────────────────

def build_minutes_calendar(
    season_dir: Path,
    fbref_root: Path,
    fpl_root: Path,
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
            write_empty_minutes_calendar(season_dir)
            return
        else:
            logging.warning(msg + " — skipped (use --create-empty to write empty)")
            return

    # Load fixture calendar
    cal = load_fixture_calendar(season_dir)
    cal["date_played"] = pd.to_datetime(cal["date_played"]).dt.tz_localize(None).dt.floor("D")

    # Load player minutes directly
    minutes = load_minutes(season_dir, fbref_root)

    # Merge on both 'fbref_id' and 'team_id'
    merged = minutes.merge(cal, on=["fbref_id", "team_id"], how="left")

    # Add was_home (canonicalized)
    merged["was_home"] = (merged["venue"].astype(str).str.strip().str.title().eq("Home")).astype("Int8")

    # Integrity check: remove rows without fixture match (if any)
    missing_fixtures = merged["date_played"].isna().sum()
    if missing_fixtures:
        logging.warning(f"{missing_fixtures} rows with missing fixture data dropped")
        merged.dropna(subset=["date_played"], inplace=True)

    # Flag active (minutes played > 0)
    merged["is_active"] = np.where(merged["minutes"].gt(0), 1, 0).astype("uint8")

    # Days since last match per player
    merged = merged.sort_values(["player_id", "date_played"])
    merged["days_since_last"] = (
        merged.groupby("player_id")["date_played"].diff().dt.days.fillna(0).astype(int)
    )

    # Normalize types before FPL joins
    for c in ["player_id","fbref_id","team_id"]:
        if c in merged.columns:
            merged[c] = merged[c].astype(str)
    merged["date_played"] = pd.to_datetime(merged["date_played"]).dt.floor("D")
    merged["was_home"] = merged["was_home"].astype("Int8")

    # ── PRICE ────────────────────────────────────────────────────────────────
    fpl_prices = load_fpl_fixture_prices(fpl_root, season_key)
    if fpl_prices is not None:
        price_keys = ["player_id","date_played","was_home"]
        if "fbref_id" in fpl_prices.columns and "fbref_id" in merged.columns:
            price_keys.insert(1, "fbref_id")
        before = len(merged)
        merged = merged.merge(fpl_prices, on=price_keys, how="left", validate="m:1")
        assert len(merged) == before, "Row count changed after price merge"
        logging.info("[%s] Price coverage via merged_gws: %.3f", season_key, merged["price"].notna().mean())
    else:
        merged["price"] = np.nan
        logging.warning("[%s] No prices available; 'price' set to NaN", season_key)

    # ── xP ──────────────────────────────────────────────────────────────────
    fpl_xp = load_fpl_fixture_xp(fpl_root, season_key)
    if fpl_xp is not None:
        xp_keys = ["player_id","date_played","was_home"]
        if "fbref_id" in fpl_xp.columns and "fbref_id" in merged.columns:
            xp_keys.insert(1, "fbref_id")
        before = len(merged)
        merged = merged.merge(fpl_xp, on=xp_keys, how="left", validate="m:1")
        assert len(merged) == before, "Row count changed after xP merge"
    else:
        merged["xp"] = np.nan
        logging.warning("[%s] No xP column in merged_gws; 'xp' set to NaN", season_key)

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
        # FPL provided values override
        has_fpl = merged["is_starter"].notna()
        starter_source.loc[has_fpl] = "fpl"
        merged["is_starter"] = merged["is_starter"].fillna(0).astype("uint8")
    else:
        # Column absent → your preference: default to 1
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
        logging.warning("[%s] No points columns; total_points/bonus/bps/clean_sheets set to NaN", season_key)

    # ── Enforce price completeness: DROP rows with missing price ────────────
    if merged["price"].isna().any():
        before = len(merged)
        n_miss = int(merged["price"].isna().sum())
        merged = merged[merged["price"].notna()].copy()
        dropped = before - len(merged)
        logging.warning("[%s] Dropped %d rows with missing price (%.2f%% of rows). Kept %d.",
                        season_key, dropped, 100.0 * n_miss / max(before,1), len(merged))
        if len(merged) == 0:
            logging.error("[%s] All rows dropped due to missing price — output will be EMPTY.", season_key)

    # ── Output (enforce canonical order; keep only known cols) ──────────────
    cols = [c for c in OUT_COLS if c in merged.columns]
    merged[cols].to_csv(out_fp, index=False)
    logging.info("✅ %s written (%d rows)", out_fp.name, len(merged))

# ───────────────────────────── Batch & CLI ─────────────────────────────

def run_batch(
    seasons: List[str],
    fixtures_root: Path,
    fbref_root: Path,
    fpl_root: Path,
    force: bool,
    create_empty: bool,
) -> None:
    for season in seasons:
        season_dir = fixtures_root / season
        if not season_dir.is_dir():
            if create_empty:
                logging.warning("%s • season directory missing — creating EMPTY file", season)
                season_dir.mkdir(parents=True, exist_ok=True)
                write_empty_minutes_calendar(season_dir)
                continue
            else:
                logging.warning("Season %s directory missing – skipped", season)
                continue
        try:
            build_minutes_calendar(season_dir, fbref_root, fpl_root, force=force, create_empty=create_empty)
        except Exception:
            logging.exception("❌ %s failed", season)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/fixtures"),
                    help="Root dir containing season subfolders")
    ap.add_argument("--fbref-root", type=Path, default=Path("data/processed/fbref/ENG-Premier League"),
                    help="FBref league root (for JSONs and player_match)")
    ap.add_argument("--fpl-root", type=Path, default=Path("data/processed/fpl"),
                    help="FPL root containing <SEASON>/gws/merged_gws.csv")
    ap.add_argument("--season", help="e.g. 2024-2025 (omit to process all seasons)")
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
        args.force,
        create_empty=args.create_empty,
    )

if __name__ == "__main__":
    main()
