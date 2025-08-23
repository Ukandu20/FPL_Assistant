#!/usr/bin/env python3
"""
scripts.fpl_pipeline.prices.prices_from_merged

Build per-season player price registry from processed FPL merged_gws.csv files
(after enrichment + team mapping + game_id assignment).

Outputs (per season under --out-json-dir / --out-parquet-dir):
  • <out-json-dir>/<SEASON>.json
      {
        "ef07a30f": {"1": 5.5, "2": 5.4, ...},
        "3f61b5e7": {"1": 6.0, ...},
        ...
      }
  • <out-parquet-dir>/<SEASON>.parquet
      columns: season, round, player_id, price, team_id, fpl_pos

Price detection:
  - Prefer columns (in order): 'price', 'now_cost', 'value', 'cost'
  - If the detected column looks like FPL tenths (>= 30 typical), divide by 10
  - Round to 1 decimal

Earliest-of-GW selection:
  - Use date_played (YYYY-MM-DD) and time (HH:MM) if available
  - Else fall back to game_date
  - Else use input ordering

Only rows with non-null player_id are considered.

CLI
---
py -m scripts.fpl_pipeline.prices.prices_from_merged ^
  --proc-root data/processed/fpl ^
  --out-json-dir data/processed/registry/prices ^
  --out-parquet-dir data/processed/registry/prices_parquet ^
  --season 2025-2026 ^
  --log-level INFO
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

PRICE_CANDIDATES = ["price", "now_cost", "value", "cost"]

def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

def write_parquet(p: Path, df: pd.DataFrame) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False)
    except Exception:
        # fallback to csv if parquet engine unavailable
        df.to_csv(str(p).replace(".parquet", ".csv"), index=False)

def detect_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for c in PRICE_CANDIDATES:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    return None

def normalise_price(raw: pd.Series) -> pd.Series:
    """Return price in £ (one decimal)."""
    s = pd.to_numeric(raw, errors="coerce")
    # Heuristic: if median >= 30, assume tenths
    med = s.median(skipna=True)
    if pd.notna(med) and med >= 30:
        s = s / 10.0
    return s.round(1)

def parse_dt(df: pd.DataFrame) -> pd.Series:
    """
    Build a timestamp for ordering within a GW:
      - combine date_played + time when available
      - else use game_date
    """
    if "date_played" in df.columns:
        date = pd.to_datetime(df["date_played"], errors="coerce")
        if "time" in df.columns:
            # 'time' as HH:MM
            tm = pd.to_datetime(df["time"], errors="coerce", format="%H:%M").dt.time
            # build datetime with date + time (naive)
            dt = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + df["time"].astype(str), errors="coerce")
        else:
            dt = date
    elif "game_date" in df.columns:
        dt = pd.to_datetime(df["game_date"], errors="coerce")
    else:
        # fallback: all NaT
        dt = pd.to_datetime(pd.Series([None] * len(df)))
    return dt

def earliest_price_per_gw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df expects: player_id, round, price_series, date/time columns optionally.
    Returns a frame with one row per (player_id, round): price at earliest dt in that GW.
    """
    # ensure round int
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    else:
        df["round"] = pd.Series([pd.NA]*len(df), dtype="Int64")

    # build dt order key
    dt = parse_dt(df)
    df["_ord_dt"] = dt

    # build price
    ps = detect_price_series(df)
    if ps is None:
        raise ValueError("No price-like column found (tried: %s)" % ", ".join(PRICE_CANDIDATES))
    df["_price"] = normalise_price(ps)

    # keep only rows with player_id and round
    df = df[df["player_id"].notna()].copy()
    # If still no round, try derive from 'gw' or 'gameweek'
    if df["round"].isna().all():
        for alt in ["gw","gameweek","Gameweek","GW"]:
            if alt in df.columns:
                df["round"] = pd.to_numeric(df[alt], errors="coerce").astype("Int64")
                break

    # determine earliest row per (player_id, round)
    # If all dt are NaT in a group, use first occurrence
    df["_row_ix"] = np.arange(len(df))
    def pick_idx(g: pd.DataFrame) -> int:
        if g["_ord_dt"].notna().any():
            return g.sort_values(["_ord_dt","_row_ix"]).index[0]
        return g.sort_values(["_row_ix"]).index[0]

    idxs: List[int] = []
    for (pid, rnd), g in df.groupby(["player_id","round"]):
        if pd.isna(pid) or pd.isna(rnd):
            continue
        idxs.append(pick_idx(g))

    subset = df.loc[idxs, ["player_id","round","_price","team_id","fpl_pos"]].copy()
    subset.rename(columns={"_price":"price"}, inplace=True)
    return subset

def build_registry_json(frame: pd.DataFrame) -> dict:
    """
    { player_id: { "<round>": price, ... }, ... }
    """
    reg = {}
    for pid, g in frame.groupby("player_id"):
        ent = {}
        for rnd, price in zip(g["round"].astype("Int64"), g["price"]):
            if pd.isna(rnd) or pd.isna(price):
                continue
            ent[str(int(rnd))] = float(price)
        if ent:
            reg[str(pid)] = ent
    return reg

def process_season(season_dir: Path, out_json_dir: Path, out_parquet_dir: Path) -> None:
    season = season_dir.name
    merged = season_dir / "gws" / "merged_gws.csv"
    if not merged.is_file():
        logging.warning("[%s] missing %s", season, merged)
        return
    df = read_csv(merged)
    try:
        ep = earliest_price_per_gw(df)
    except Exception as e:
        logging.error("[%s] %s", season, e)
        return

    # write JSON
    reg = build_registry_json(ep)
    write_json(out_json_dir / f"{season}.json", reg)

    # write parquet/csv
    ep2 = ep.copy()
    ep2["season"] = season
    ep2 = ep2[["season","round","player_id","price","team_id","fpl_pos"]]
    write_parquet(out_parquet_dir / f"{season}.parquet", ep2)

    logging.info("[%s] prices: players=%d rows=%d → %s ; %s",
                 season, ep["player_id"].nunique(), len(ep),
                 out_json_dir / f"{season}.json",
                 out_parquet_dir / f"{season}.parquet")

def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-season player price registry from processed merged_gws.csv files.")
    ap.add_argument("--proc-root", required=True, type=Path, help="Root with <season>/gws/merged_gws.csv")
    ap.add_argument("--out-json-dir", type=Path, default=Path("data/processed/registry/prices"))
    ap.add_argument("--out-parquet-dir", type=Path, default=Path("data/processed/registry/prices_parquet"))
    ap.add_argument("--season", help="Only process a single season (e.g., '2025-26' or '2025-2026').")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    seasons = [d for d in sorted(args.proc_root.iterdir()) if d.is_dir()]
    if args.season:
        # accept long or short
        s = args.season.strip()
        short = s if re.match(r"^\d{4}-\d{2}$", s) else f"{s[:4]}-{s[-2:]}" if re.match(r"^\d{4}-\d{4}$", s) else s
        seasons = [d for d in seasons if d.name in {s, short}]
    if not seasons:
        logging.warning("No seasons found under %s", args.proc_root)
        return

    for sdir in seasons:
        process_season(sdir, args.out_json_dir, args.out_parquet_dir)

if __name__ == "__main__":
    import re
    main()
