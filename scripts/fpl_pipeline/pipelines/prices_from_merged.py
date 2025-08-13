#!/usr/bin/env python3
"""
prices_from_merged.py  —  Build season price/state registry from historical merged_gws files
────────────────────────────────────────────────────────────────────────────────────────────
• Input: per-season merged_gws (CSV or Parquet) at any nested path you provide.
• Output:
    - registry/prices/<SEASON>.json        (nested, deterministic)
    - registry/prices_parquet/<SEASON>.parquet   (flat, join-friendly)

Key behavior
------------
• PRICE = value / 10.0 (explicit conversion from FPL tenths to £, rounded to 1dp).
• Derives 'active' per GW from --active-from status|minutes|always
  (default 'auto' → status if present, else minutes if present, else always True).
• Handles single-season OR batch discovery under --root, with flexible season dir names:
    YYYY-YYYY, YYYY-YY, YYYY_YYYY, YYYY_YY (e.g., 2023-2024, 2019-20, 2023_2024, 2019_20)
• Supports nested --filename like "gw/merged_gws.csv" or "gws/merged_gws.csv".
• If the merged file lacks a 'season' column, injects one from --season (or path regex).

CLI examples
------------
Single season:
  py -m scripts.fpl_pipeline.pipelines.prices_from_merged \
    --season 2023-2024 \
    --merged "data/processed/fpl/2023-2024/gw/merged_gws.csv" \
    --active-from minutes --force --log-level INFO

Batch (auto-discover seasons under root):
  py -m scripts.fpl_pipeline.pipelines.prices_from_merged \
    --root "data/processed/fpl" \
    --filename "gw/merged_gws.csv" \
    --active-from status --force --log-level INFO

Batch (explicit subset):
  py -m scripts.fpl_pipeline.pipelines.prices_from_merged \
    --root "data/processed/fpl" \
    --filename "gw/merged_gws.csv" \
    --seasons 2020-2021,2021-2022,2022-2023 \
    --active-from status --force
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd


# ───────────────────────────── IO helpers ─────────────────────────────

def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def _load_df(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


# ─────────────────────── normalization / features ─────────────────────

def _normalize_pos_value(x):
    """Map many variants to {GK, DEF, MID, FWD}; return np.nan if unknown/missing."""
    if x is None:
        return np.nan
    s = str(x).strip().upper()
    if s in {"", "NAN", "NONE", "NULL"}:
        return np.nan
    m = {
        "GKP": "GK", "GK": "GK", "GOALKEEPER": "GK",
        "DEF": "DEF", "DEFENDER": "DEF",
        "MID": "MID", "MIDFIELDER": "MID",
        "FWD": "FWD", "FORWARD": "FWD", "STR": "FWD"
    }
    if s in m:
        return m[s]
    if s in {"GK", "DEF", "MID", "FWD"}:
        return s
    return np.nan


def _coerce_pos(series: pd.Series) -> pd.Series:
    """Vectorized wrapper around _normalize_pos_value."""
    return series.map(_normalize_pos_value)


def _derive_active(df: pd.DataFrame, mode: str,
                   status_col: Optional[str], minutes_col: Optional[str]) -> pd.Series:
    mode = mode.lower()
    if mode == "status" and status_col and status_col in df.columns:
        return df[status_col].astype(str).str.lower().isin({"a", "d"})
    if mode == "minutes" and minutes_col and minutes_col in df.columns:
        return pd.to_numeric(df[minutes_col], errors="coerce").fillna(0) > 0
    if mode == "status" and (not status_col or status_col not in df.columns):
        logging.warning("active-from=status requested but status column missing; falling back to minutes/always.")
    if (minutes_col and minutes_col in df.columns):
        return pd.to_numeric(df[minutes_col], errors="coerce").fillna(0) > 0
    return pd.Series(True, index=df.index)


def _normalize_columns(df: pd.DataFrame,
                       season_col: str, gw_col: str, player_col: str, price_col: str,
                       pos_col: str, team_short_col: str, team_hex_col: str) -> Tuple[pd.DataFrame, str]:
    """Rename columns to canonical names; do NOT require 'season' (we may inject it)."""
    # Map common aliases for gw
    if gw_col not in df.columns:
        for alt in ["round", "gw_orig", "gw_played"]:
            if alt in df.columns:
                gw_col = alt
                break
    # Rename to canonical names if needed
    ren = {}
    if season_col in df.columns and season_col != "season": ren[season_col] = "season"
    if gw_col != "gw": ren[gw_col] = "gw"
    if player_col != "player_id": ren[player_col] = "player_id"
    if price_col != "value": ren[price_col] = "value"
    if pos_col != "fpl_pos": ren[pos_col] = "fpl_pos"
    if team_short_col != "team": ren[team_short_col] = "team"
    if team_hex_col != "team_id": ren[team_hex_col] = "team_id"
    df = df.rename(columns=ren)
    required = {"gw", "player_id", "value", "fpl_pos", "team", "team_id"}  # season not required here
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df, "gw"


# ─────────────────────── core single-season build ─────────────────────

def build_single_season(
    season: str,
    merged_path: Path,
    out_json: Path,
    out_parquet: Path,
    season_col: str = "season",
    gw_col: str = "gw",
    player_col: str = "player_id",
    price_col: str = "value",
    pos_col: str = "fpl_pos",
    team_short_col: str = "team",
    team_hex_col: str = "team_id",
    status_col: Optional[str] = "status",
    minutes_col: Optional[str] = "minutes",
    active_from: str = "auto",
) -> Tuple[dict, pd.DataFrame]:
    """Build registry artifacts for a single season from a merged_gws file."""
    df = _load_df(merged_path)

    # Inject season if missing
    if "season" not in df.columns:
        if season:
            df["season"] = str(season)
            logging.info("Injected 'season' column with value %s", season)
        else:
            m = re.search(r'(\d{4}[-_]\d{2,4})', str(merged_path))
            if not m:
                raise ValueError("Season column missing and cannot infer from path; pass --season")
            df["season"] = m.group(1).replace("_", "-")
            logging.info("Inferred 'season' as %s from path", df["season"].iloc[0])

    # Normalize columns (season not required here)
    df, gw_col = _normalize_columns(df, season_col, gw_col, player_col, price_col, pos_col, team_short_col, team_hex_col)

    # Filter season
    df["season"] = df["season"].astype(str)
    df = df[df["season"] == season].copy()
    if df.empty:
        raise ValueError(f"No rows for season {season} in {merged_path}")

    # Types
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["gw"]).copy()
    df["gw"] = df["gw"].astype(int)

    # PRICE: explicit tenths → pounds
    df["price"] = (pd.to_numeric(df["value"], errors="coerce").astype(float) / 10.0).round(1)

    # POS normalization & filling
    # 1) normalize fpl_pos; 2) fallback to 'position' column; 3) per-player modal position
    df["fpl_pos"] = _coerce_pos(df["fpl_pos"])
    if "position" in df.columns:
        df["position_norm"] = _coerce_pos(df["position"])
        df["fpl_pos"] = df["fpl_pos"].fillna(df["position_norm"])
    pos_mode_map = (
        df.loc[df["fpl_pos"].notna(), ["player_id", "fpl_pos"]]
          .groupby("player_id")["fpl_pos"]
          .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
          .to_dict()
    )
    df["fpl_pos"] = df["fpl_pos"].fillna(df["player_id"].map(pos_mode_map))
    missing_pos = df["fpl_pos"].isna().sum()
    if missing_pos:
        logging.warning("Dropping %d rows with unknown position after fallbacks.", missing_pos)
        df = df[df["fpl_pos"].notna()].copy()

    # ACTIVE
    if active_from == "auto":
        mode = "status" if (status_col and status_col in df.columns) else ("minutes" if (minutes_col and minutes_col in df.columns) else "always")
    else:
        mode = active_from
    df["active"] = _derive_active(df, mode, status_col, minutes_col)

    # Deduplicate: 1 row per (player_id, gw). Deterministic sort → take last.
    sort_cols = ["player_id", "gw"]
    for c in ["home", "away", "home_id", "away_id", "opp_id"]:
        if c in df.columns:
            sort_cols.append(c)
    df = df.sort_values(sort_cols)

    # Log duplicates collapsed
    dupes = df.duplicated(subset=["player_id", "gw"]).sum()
    if dupes:
        logging.info("Collapsing %d duplicate (player_id, gw) rows → last after deterministic sort.", dupes)

    agg = (
        df.groupby(["player_id", "gw"], as_index=False)
          .agg({
              "season": "first",
              "price": "last",
              "team_id": "last",
              "team": "last",
              "fpl_pos": "last",
              "active": "last"
          })
    )

    # Validation
    if (~agg["price"].between(3.0, 16.0)).any():
        bad = agg.loc[~agg["price"].between(3.0, 16.0)].head(10)
        logging.warning("Out-of-range prices (sample):\n%s", bad)
    if (~agg["fpl_pos"].isin({"GK", "DEF", "MID", "FWD"})).any():
        bad = agg.loc[~agg["fpl_pos"].isin({"GK", "DEF", "MID", "FWD"}), ["player_id", "fpl_pos"]].drop_duplicates().head(10)
        raise ValueError(f"Invalid fpl_pos detected, sample:\n{bad}")

    # initial_price from GW1 if present
    initial = (agg[agg["gw"] == 1].set_index("player_id")["price"].to_dict())

    # Build nested JSON
    nested: Dict[str, Any] = {
        "schema": "prices.v1",
        "deadline_source": "historical_merged",
        "season": season,
        "players": {}
    }

    for pid, pdf in agg.sort_values(["player_id", "gw"]).groupby("player_id"):
        gw_map = {}
        for _, r in pdf.iterrows():
            gw_map[str(int(r["gw"]))] = {
                "price": float(r["price"]),
                "team_hex": str(r["team_id"]),
                "team": str(r["team"]),
                "fpl_pos": str(r["fpl_pos"]),
                "active": bool(r["active"]),
                "deadline_id": int(r["gw"])   # GW acts as identifier for historical
            }
        nested["players"][str(pid)] = {
            "initial_price": float(initial.get(pid, list(gw_map.values())[0]["price"])),
            "gw": gw_map
        }

    # Write JSON and Parquet
    _write_json(out_json, nested)

    flat = agg[["season", "player_id", "gw", "price", "team_id", "team", "fpl_pos", "active"]].copy()
    flat.rename(columns={"team_id": "team_hex"}, inplace=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    flat.to_parquet(out_parquet, index=False)

    logging.info("✅ %s → %s | %s  (rows: %d)", season, out_json, out_parquet, len(flat))
    return nested, flat


# ─────────────────────── batch discovery utilities ────────────────────

SEASON_PATTERNS = [
    r"\d{4}-\d{4}",  # 2023-2024
    r"\d{4}-\d{2}",  # 2019-20
    r"\d{4}_\d{4}",  # 2023_2024
    r"\d{4}_\d{2}",  # 2019_20
]


def _is_season_dir(name: str) -> bool:
    return any(re.fullmatch(p, name) for p in SEASON_PATTERNS)


def discover_seasons(root: Path, filename: str) -> Dict[str, Path]:
    """
    Return {season: merged_path} for all subdirs under root that contain filename.
    Supports nested filename like "gw/merged_gws.csv".
    """
    out: Dict[str, Path] = {}
    for p in sorted(root.iterdir()):
        if p.is_dir() and _is_season_dir(p.name):
            candidate = p / filename
            if candidate.exists():
                out[p.name.replace("_", "-")] = candidate
    return out


# ───────────────────────────────── main ───────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    # Single-season mode
    ap.add_argument("--season", help="Single season like 2021-2022")
    ap.add_argument("--merged", type=Path, help="Path to merged_gws.csv or .parquet for the season")
    # Batch mode
    ap.add_argument("--root", type=Path, help="Root dir containing <season>/<filename>")
    ap.add_argument("--seasons", help="Comma-separated seasons to process under --root (optional; defaults to all discovered)")
    ap.add_argument("--filename", default="gw/merged_gws.csv", help="Relative path inside each <season> folder")
    # Output dirs
    ap.add_argument("--out-json-dir", default="data/registry/prices")
    ap.add_argument("--out-parquet-dir", default="data/registry/prices_parquet")
    # Column options
    ap.add_argument("--season-col", default="season")
    ap.add_argument("--gw-col", default="gw", help="Alternative names handled: round, gw_orig, gw_played")
    ap.add_argument("--player-col", default="player_id")
    ap.add_argument("--price-col", default="value")  # interpreted as tenths, divided by 10
    ap.add_argument("--pos-col", default="fpl_pos")
    ap.add_argument("--team-short-col", default="team")
    ap.add_argument("--team-hex-col", default="team_id")
    ap.add_argument("--status-col", default="status")
    ap.add_argument("--minutes-col", default="minutes")
    ap.add_argument("--active-from", default="auto", choices=["auto", "status", "minutes", "always"])
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    # Helpful discovery logs
    if args.root:
        try:
            dirs = [d.name for d in Path(args.root).iterdir() if d.is_dir()]
            logging.info("Found dirs under %s: %s", args.root, ", ".join(dirs[:50]) + (" ..." if len(dirs) > 50 else ""))
            logging.info("Looking for file: %s", args.filename)
        except Exception as e:
            logging.debug("Directory listing failed: %s", e)

    # Decide mode
    if args.root and not args.season and not args.merged:
        # Batch mode
        mapping = discover_seasons(args.root, args.filename)
        if args.seasons:
            wanted = set(s.strip().replace("_", "-") for s in args.seasons.split(",") if s.strip())
            mapping = {s: p for s, p in mapping.items() if s in wanted}
        if not mapping:
            raise SystemExit(f"No seasons discovered under {args.root} with {args.filename}")
        logging.info("Discovered seasons: %s", ", ".join(mapping.keys()))
        for season, merged_path in mapping.items():
            out_json = Path(args.out_json_dir) / f"{season}.json"
            out_parquet = Path(args.out_parquet_dir) / f"{season}.parquet"
            if out_json.exists() and out_parquet.exists() and not args.force:
                logging.info("Skipping %s (outputs exist; use --force to overwrite)", season)
                continue
            build_single_season(
                season=season,
                merged_path=merged_path,
                out_json=out_json,
                out_parquet=out_parquet,
                season_col=args.season_col,
                gw_col=args.gw_col,
                player_col=args.player_col,
                price_col=args.price_col,
                pos_col=args.pos_col,
                team_short_col=args.team_short_col,
                team_hex_col=args.team_hex_col,
                status_col=args.status_col if args.status_col else None,
                minutes_col=args.minutes_col if args.minutes_col else None,
                active_from=args.active_from,
            )
        logging.info("✅ Batch complete.")
    else:
        # Single season mode
        if not (args.season and args.merged):
            raise SystemExit("Provide --season and --merged for single run, or use --root for batch.")
        out_json = Path(args.out_json_dir) / f"{args.season}.json"
        out_parquet = Path(args.out_parquet_dir) / f"{args.season}.parquet"
        if (out_json.exists() or out_parquet.exists()) and not args.force:
            raise SystemExit(f"Output exists ({out_json} / {out_parquet}). Use --force to overwrite.")
        build_single_season(
            season=args.season,
            merged_path=args.merged,
            out_json=out_json,
            out_parquet=out_parquet,
            season_col=args.season_col,
            gw_col=args.gw_col,
            player_col=args.player_col,
            price_col=args.price_col,
            pos_col=args.pos_col,
            team_short_col=args.team_short_col,
            team_hex_col=args.team_hex_col,
            status_col=args.status_col if args.status_col else None,
            minutes_col=args.minutes_col if args.minutes_col else None,
            active_from=args.active_from,
        )


if __name__ == "__main__":
    main()
