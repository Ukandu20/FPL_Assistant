#!/usr/bin/env python3
"""
prices_registry_builder.py

Build a per-season player price registry combining:
  • GW0 (preseason) prices from the current season's PMC (earliest observed per player)
  • If the current season PMC is missing/empty:
      - GW0 price = player's mean price from previous season's PMC
      - players absent in previous season -> initial_price = null
  • GW≥1 prices from player_minutes_calendar.csv (PMC):
      - pick the earliest observed fixture price in that GW per player
      - carry-forward last known price when the player has no PMC row in a GW

Outputs:
  • registry/prices/<SEASON>.json                 (nested, deterministic)
  • registry/prices_parquet/<SEASON>.parquet      (flat, fast joins)

Notes:
  • PMC prices are tied to MATCH TIME (not FPL deadline).
  • Per-GW entries carry a `source`: "pmc" (observed) or "carry" (forward-filled).
"""
from __future__ import annotations
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set

import numpy as np
import pandas as pd

# ------------------------- Helpers -------------------------

SEASON_RX = re.compile(r"^\d{4}-\d{4}$")

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

def _canon_date(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)
    return dt.dt.floor("D")

def _discover_seasons(fixtures_root: Path, fpl_root: Path) -> List[str]:
    cand: Set[str] = set()
    if fixtures_root.exists():
        cand |= {p.name for p in fixtures_root.iterdir() if p.is_dir()}
    if fpl_root.exists():
        cand |= {p.name for p in fpl_root.iterdir() if p.is_dir()}
    return sorted(s for s in cand if SEASON_RX.match(s))

def _prev_season(season: str) -> Optional[str]:
    m = re.match(r"^(\d{4})-(\d{4})$", season)
    if not m: return None
    a, b = int(m.group(1)), int(m.group(2))
    return f"{a-1}-{b-1}"

# ------------------------- Loaders -------------------------

def load_pmc(pmc_fp: Path) -> pd.DataFrame:
    """
    player_minutes_calendar.csv must include (at minimum):
        player_id, gw_orig, date_played, price
    """
    if not pmc_fp.exists():
        # Return empty with correct header to keep downstream sane
        cols = ["player_id","gw_orig","date_played","price","team_id","team","fpl_pos"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(pmc_fp, low_memory=False)
    req = {"player_id","gw_orig","date_played","price"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{pmc_fp} missing required columns: {missing}")

    df = df.copy()
    df["player_id"]   = df["player_id"].astype(str)
    df["gw_orig"]     = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")
    df["date_played"] = _canon_date(df["date_played"])
    df["price"]       = pd.to_numeric(df["price"], errors="coerce").round(1)

    # Standardize metadata columns
    if "team_id" not in df.columns:
        df["team_id"] = np.nan
    if "team" not in df.columns:
        df["team"] = np.nan
    if "fpl_pos" not in df.columns and "pos" in df.columns:
        df["fpl_pos"] = df["pos"]
    elif "fpl_pos" not in df.columns:
        df["fpl_pos"] = np.nan

    return df

# ------------------------- Core Builder -------------------------

def build_registry_from_pmc(
    season: str,
    fixtures_root: Path,
    fpl_root: Path,              # unused but kept for symmetry/CLI
    out_json: Path,
    out_parquet: Path,
    force: bool = False,
) -> Tuple[dict, pd.DataFrame] | Tuple[None, None]:
    """
    GW0 from current season PMC earliest observation.
    If current PMC missing/empty: GW0 from previous season PMC mean per player.
    GW≥1 from PMC (earliest per GW) + carry-forward.
    """
    # Skip logic
    if out_json.exists() and out_parquet.exists() and not force:
        logging.info("%s • outputs exist — skip (use --force to overwrite)", season)
        return None, None

    # ---- Paths
    season_dir = fixtures_root / season
    pmc_fp     = season_dir / "player_minutes_calendar.csv"

    # ---- Load current PMC
    pmc = load_pmc(pmc_fp)

    # ---- Derive GW list from PMC
    if not pmc.empty and pmc["gw_orig"].notna().any():
        gws = sorted(int(g) for g in pmc["gw_orig"].dropna().unique().tolist() if g > 0)
    else:
        gws = []

    # ---- Initial (GW0) price map
    # Case A: current PMC present → initial = earliest observed price in *this* season per player
    initial_price: Dict[str, float] = {}
    if not pmc.empty:
        earliest_any = (pmc.dropna(subset=["price"])
                          .sort_values(["player_id","date_played","gw_orig"])
                          .drop_duplicates("player_id", keep="first"))
        initial_price = dict(zip(earliest_any["player_id"], earliest_any["price"]))
        logging.info("[%s] Initial prices from current PMC for %d players", season, len(initial_price))
    else:
        # Case B: current PMC missing/empty → fallback to previous season mean per player
        prev = _prev_season(season)
        prev_pmc_fp = fixtures_root / prev / "player_minutes_calendar.csv" if prev else None
        if prev and prev_pmc_fp and prev_pmc_fp.exists():
            prev_df = load_pmc(prev_pmc_fp)
            prev_mean = (prev_df.dropna(subset=["price"])
                               .groupby("player_id", as_index=True)["price"]
                               .mean()
                               .round(1))
            initial_price = prev_mean.to_dict()
            logging.warning("[%s] PMC empty; GW0 seeded from previous season mean for %d players", season, len(initial_price))
        else:
            logging.warning("[%s] PMC empty and no previous season available — registry will be empty", season)
            initial_price = {}

    # ---- Per-GW price from PMC (earliest per GW)
    per_gw_df = pd.DataFrame(columns=["player_id","gw_orig","price","team_id","team","fpl_pos","source"])
    if not pmc.empty:
        pmc_valid = pmc.dropna(subset=["gw_orig","date_played","price"]).copy()
        if not pmc_valid.empty:
            earliest_gw = (pmc_valid.sort_values(["player_id","gw_orig","date_played"])
                                     .drop_duplicates(["player_id","gw_orig"], keep="first"))
            earliest_gw["source"] = "pmc"
            per_gw_df = earliest_gw[["player_id","gw_orig","price","team_id","team","fpl_pos","source"]].copy()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # If no GWs at all (season PMC empty): write initial-only from previous season means
    if not gws:
        nested: Dict[str, Any] = {"schema": "prices.v2", "season": season, "players": {}}
        for pid in sorted(initial_price.keys()):
            nested["players"][pid] = {"initial_price": float(initial_price[pid]), "gw": {}}
        _write_json(out_json, nested)
        pd.DataFrame(columns=["season","player_id","gw","price","source","team_hex","team","fpl_pos"]).to_parquet(out_parquet, index=False)
        logging.info("✅ %s • initial-only registry written (no PMC GWs)", season)
        return nested, pd.DataFrame()

    # ---------------- Grid build for GW≥1 ----------------

    # Players in this season = those who appear in current PMC at least once
    season_player_ids = sorted(pmc["player_id"].dropna().astype(str).unique().tolist())
    grid = pd.MultiIndex.from_product([season_player_ids, gws], names=["player_id","gw"])
    grid_df = pd.DataFrame(index=grid).reset_index()

    # Merge earliest PMC prices per GW
    if not per_gw_df.empty:
        tmp = per_gw_df.rename(columns={"gw_orig":"gw"})
        tmp["player_id"] = tmp["player_id"].astype(str)
        tmp["gw"]        = tmp["gw"].astype(int)
        grid_df = grid_df.merge(tmp, on=["player_id","gw"], how="left")

    # Seed metadata from each player's earliest observed row (across the season)
    earliest_meta = (pmc.sort_values(["player_id","date_played","gw_orig"])
                       .drop_duplicates("player_id", keep="first"))
    meta_keep = ["player_id"] + [c for c in ["team_id","team","fpl_pos"] if c in earliest_meta.columns]
    earliest_meta = earliest_meta[meta_keep].rename(columns={"team_id":"team_id_init","team":"team_init","fpl_pos":"fpl_pos_init"})
    grid_df = grid_df.merge(earliest_meta, on="player_id", how="left")

    # Seed initial price from current season earliest (if present for this player)
    grid_df["init_price"] = grid_df["player_id"].map(initial_price)

    # Vectorized forward-fill per player (no groupby.apply)
    grid_df = grid_df.sort_values(["player_id","gw"]).copy()
    grid_df["price_ffill"] = grid_df["price"].combine_first(grid_df["init_price"])
    grid_df["price_ffill"] = grid_df.groupby("player_id", group_keys=False)["price_ffill"].ffill()

    # Source attribution
    grid_df["source"] = np.where(grid_df["price"].notna(), "pmc",
                          np.where(grid_df["price_ffill"].notna(), "carry", np.nan))

    # Forward-fill metadata per player from *_init
    for m, m_init in [("team_id","team_id_init"), ("team","team_init"), ("fpl_pos","fpl_pos_init")]:
        if m_init in grid_df.columns:
            grid_df[m] = grid_df.get(m, pd.Series(index=grid_df.index, dtype="object")).combine_first(grid_df[m_init])
            grid_df[m] = grid_df.groupby("player_id", group_keys=False)[m].ffill()

    # Build nested JSON
    nested: Dict[str, Any] = {"schema": "prices.v2", "season": season, "players": {}}
    for pid, gpdf in grid_df.groupby("player_id"):
        # initial: earliest observed in THIS season; if missing (shouldn't for present players), fall back to first ffilled
        init_p = float(initial_price.get(pid)) if pid in initial_price else (
            float(gpdf["price_ffill"].iloc[0]) if len(gpdf) and pd.notna(gpdf["price_ffill"].iloc[0]) else float("nan")
        )
        gw_map = {}
        for _, r in gpdf.iterrows():
            gw_map[str(int(r["gw"]))] = {
                "price": float(r["price_ffill"]) if pd.notna(r["price_ffill"]) else None,
                "source": r.get("source", None),
                "team_hex": None if pd.isna(r.get("team_id", np.nan)) else str(r["team_id"]),
                "team":     None if pd.isna(r.get("team",    np.nan)) else str(r["team"]),
                "fpl_pos":  None if pd.isna(r.get("fpl_pos", np.nan)) else str(r["fpl_pos"]),
            }
        nested["players"][str(pid)] = {"initial_price": init_p, "gw": gw_map}

    _write_json(out_json, nested)

    # Flat Parquet mirror (use team_hex naming for consistency)
    flat = grid_df[["player_id","gw","price_ffill","source","team_id","team","fpl_pos"]].rename(
        columns={"price_ffill":"price", "team_id":"team_hex"}
    ).copy()
    flat["season"] = season
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    flat.to_parquet(out_parquet, index=False)

    miss_after = int(flat["price"].isna().sum())
    if miss_after:
        sample_nan = flat.loc[flat["price"].isna(), ["player_id","gw"]].head(12).to_dict("records")
        logging.warning("[%s] %d player-gw entries still NaN after carry-forward (sample: %s)",
                        season, miss_after, sample_nan)
    logging.info("✅ %s • wrote %s and %s", season, out_json, out_parquet)
    return nested, flat

# ------------------------- Batch Runner & CLI -------------------------

def run_batch(
    seasons: List[str],
    fixtures_root: Path,
    fpl_root: Path,
    out_dir_json: Path,
    out_dir_parquet: Path,
    force: bool,
) -> None:
    ok, fail = 0, 0
    for season in seasons:
        try:
            out_json    = out_dir_json / f"{season}.json"
            out_parquet = out_dir_parquet / f"{season}.parquet"
            build_registry_from_pmc(
                season=season,
                fixtures_root=fixtures_root,
                fpl_root=fpl_root,
                out_json=out_json,
                out_parquet=out_parquet,
                force=force,
            )
            ok += 1
        except Exception:
            logging.exception("%s • registry build failed", season)
            fail += 1
    logging.info("Batch summary: %d ok, %d failed", ok, fail)
    if fail:
        raise SystemExit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=str, help="Single season like 2024-2025")
    ap.add_argument("--seasons", type=str, help="Comma-separated list of seasons")
    ap.add_argument("--all", action="store_true", help="Process all seasons found under fixtures-root and/or fpl-root")
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/fixtures"))
    ap.add_argument("--fpl-root", type=Path, default=Path("data/processed/fpl"))
    ap.add_argument("--out-json-dir", type=Path, default=Path("data/registry/prices"))
    ap.add_argument("--out-parquet-dir", type=Path, default=Path("data/registry/prices_parquet"))
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    # Determine seasons to process
    if args.all:
        seasons = _discover_seasons(args.fixtures_root, args.fpl_root)
        if not seasons:
            logging.error("No seasons discovered under %s or %s", args.fixtures_root, args.fpl_root)
            raise SystemExit(1)
    elif args.seasons:
        seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    elif args.season:
        seasons = [args.season]
    else:
        # Default to 'all' if nothing is specified
        seasons = _discover_seasons(args.fixtures_root, args.fpl_root)
        if not seasons:
            logging.error("No seasons discovered; specify --season or --all")
            raise SystemExit(1)

    # Ensure output dirs exist
    args.out_json_dir.mkdir(parents=True, exist_ok=True)
    args.out_parquet_dir.mkdir(parents=True, exist_ok=True)

    run_batch(
        seasons=seasons,
        fixtures_root=args.fixtures_root,
        fpl_root=args.fpl_root,
        out_dir_json=args.out_json_dir,
        out_dir_parquet=args.out_parquet_dir,
        force=args.force,
    )

if __name__ == "__main__":
    main()
