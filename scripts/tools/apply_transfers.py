#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply transfers to a base team_state.json and write per-candidate team_state.json.

We infer team_id/pos (and player name if available) for IN players from your minutes file.

Usage (single candidate):
py -m scripts.tools.apply_transfers ^
  --base-team-state data/state/team_state.json ^
  --minutes data/predictions/minutes/2025-2026/GW4_6.csv ^
  --season 2025-2026 --gws 4,5,6 ^
  --candidate A1 ^
  --outs 4b2c14b1,9aa7f0d2 ^
  --ins  259f237e,7fb5d771 ^
  --captain 259f237e ^
  --vice 7fb5d771 ^
  --bench-order pidB1,pidB2,pidB3 ^
  --bench-gk pidGK ^
  --out-root data/state/candidates

Then build sim inputs for all candidates:
py -m scripts.optimizers.simulator ^
  --team-state-glob "data/state/candidates/*/team_state.json" ^
  --candidate-subdir hold ^
  --minutes ... --goals-assists ... --defense ... --saves ... --fixtures ... ^
  --season 2025-2026 --gws 4,5,6 ^
  --out-dir data/decisions/candidates ^
  --out-format parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

# Accept a few aliases in minutes to recover meta for IN players
ALIASES = {
    "season": ["season", "szn"],
    "gw": ["gw", "gw_orig", "GW", "round", "gameweek"],
    "player_id": ["player_id", "id", "element"],
    "team_id": ["team_id", "team", "team_code", "squad_id"],
    "pos": ["pos", "position", "element_type"],
    "player": ["player", "name"],
}

def pick_col(df: pd.DataFrame, logical: str) -> str:
    for cand in ALIASES.get(logical, []):
        if cand in df.columns:
            return cand
    raise KeyError(f"Missing column for '{logical}'. Aliases tried: {ALIASES.get(logical)} in {list(df.columns)}")

def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def parse_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def ensure_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    if "season" in df.columns:
        return df[df["season"] == season].copy()
    d = df.copy()
    d["season"] = season
    return d

def load_minutes_meta(minutes_path: Path, season: str, gws: List[int]) -> pd.DataFrame:
    # Read CSV or Parquet
    if minutes_path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(minutes_path)
    else:
        df = pd.read_parquet(minutes_path)
    df = ensure_season(df, season)

    # Narrow to horizon (helps de-dup; we’ll take the first row)
    gw_col = pick_col(df, "gw")
    df = df[df[gw_col].astype("Int64").isin(gws)]

    # Build a small meta view
    meta_cols = {
        "player_id": pick_col(df, "player_id"),
        "team_id": pick_col(df, "team_id"),
        "pos": pick_col(df, "pos"),
    }
    view = df[[meta_cols["player_id"], meta_cols["team_id"], meta_cols["pos"]]].copy()
    view = view.rename(columns={
        meta_cols["player_id"]: "player_id",
        meta_cols["team_id"]: "team_id",
        meta_cols["pos"]: "pos",
    })
    view["player_id"] = view["player_id"].astype(str)
    view["team_id"] = view["team_id"].astype(str)
    view["pos"] = view["pos"].astype(str).str.upper().str[:3]
    view = view.drop_duplicates(subset=["player_id"], keep="first")

    # Optional nice names if present
    try:
        name_col = pick_col(df, "player")
        names = df[[meta_cols["player_id"], name_col]].drop_duplicates(subset=[meta_cols["player_id"]]).rename(
            columns={meta_cols["player_id"]:"player_id", name_col:"player"}
        )
        names["player_id"] = names["player_id"].astype(str)
        view = view.merge(names, on="player_id", how="left")
    except Exception:
        view["player"] = np.nan

    return view

def apply_transfers(base: dict,
                    outs: List[str],
                    ins: List[str],
                    minutes_meta: pd.DataFrame,
                    captain: str|None,
                    vice: str|None,
                    bench_order: List[str],
                    bench_gk: str|None) -> dict:
    squad = base.get("squad") or base.get("players")
    if not isinstance(squad, list):
        raise ValueError("Invalid team_state: expected 'squad' or 'players' array.")
    # Normalize ids as str
    for p in squad:
        p["player_id"] = str(p["player_id"])

    # Remove OUTs
    if outs:
        keep = [p for p in squad if p["player_id"] not in set(outs)]
    else:
        keep = list(squad)

    # Add INs
    for pid in ins:
        pid = str(pid)
        if pid in {p["player_id"] for p in keep}:
            continue  # already present
        row = minutes_meta[minutes_meta["player_id"]==pid].head(1)
        if row.empty:
            raise ValueError(f"Cannot infer meta for IN player_id={pid} from minutes. Ensure the player appears in minutes for the chosen season/GWs.")
        rec = {
            "player_id": pid,
            "player": str(row["player"].iloc[0]) if "player" in row.columns and pd.notna(row["player"].iloc[0]) else pid,
            "pos": str(row["pos"].iloc[0]),
            "team_id": str(row["team_id"].iloc[0]),
            # Optional fields initialized empty; simulator can infer lineup later if needed
            "is_start_xi": None,
            "bench_order": None,
            "is_bench_gk": None,
        }
        keep.append(rec)

    if len(keep) != 15:
        raise ValueError(f"Resulting squad size is {len(keep)} (expected 15). Check your outs/ins.")

    # Reset bench/captain flags
    for p in keep:
        p["is_captain"] = False
        p["is_vice"] = False
        p["is_start_xi"] = p.get("is_start_xi", None)
        p["bench_order"] = p.get("bench_order", None)
        p["is_bench_gk"] = p.get("is_bench_gk", None)

    # Captain / Vice
    if captain:
        found = False
        for p in keep:
            if p["player_id"] == str(captain):
                p["is_captain"] = True
                found = True
                break
        if not found:
            raise ValueError(f"Captain id {captain} not in squad after transfers.")
    if vice:
        found = False
        for p in keep:
            if p["player_id"] == str(vice):
                p["is_vice"] = True
                found = True
                break
        if not found:
            raise ValueError(f"Vice id {vice} not in squad after transfers.")

    # Bench order (outfield)
    if bench_order:
        # clear all bench_order first
        for p in keep:
            if p.get("pos","").upper() != "GK":
                p["bench_order"] = None
        # assign 1..3 to provided ids
        if len(bench_order) != 3:
            raise ValueError("--bench-order expects exactly 3 outfield player_ids (bench1,bench2,bench3).")
        for idx, pid in enumerate(bench_order, 1):
            setok = False
            for p in keep:
                if p["player_id"] == str(pid):
                    if p["pos"].upper() == "GK":
                        raise ValueError(f"bench_order includes GK ({pid}); provide only outfield players.")
                    p["bench_order"] = idx
                    setok = True
                    break
            if not setok:
                raise ValueError(f"bench_order id {pid} not in squad after transfers.")

    # Bench GK
    if bench_gk:
        for p in keep:
            if p["pos"].upper() == "GK":
                p["is_bench_gk"] = (p["player_id"] == str(bench_gk))

    # Build new team_state
    out = dict(base)  # shallow copy base metadata
    # Use the same top-level key that existed
    if "squad" in base:
        out["squad"] = keep
    elif "players" in base:
        out["players"] = keep
    else:
        out["squad"] = keep
    return out

def main():
    ap = argparse.ArgumentParser(description="Apply transfers to base team_state and write per-candidate team_state.json")
    ap.add_argument("--base-team-state", required=True, help="Path to base team_state.json (current squad)")
    ap.add_argument("--minutes", required=True, help="Minutes file (CSV/Parquet) used to infer team_id/pos for IN players")
    ap.add_argument("--season", required=True)
    ap.add_argument("--gws", required=True, help="Comma-separated GWs, e.g., 4,5,6")

    ap.add_argument("--candidate", required=True, help="Candidate label, e.g., A1, A2, WC")
    ap.add_argument("--out-root", required=True, help="Root folder to write <candidate>/team_state.json")

    ap.add_argument("--outs", default="", help="Comma-separated player_ids to remove")
    ap.add_argument("--ins",  default="", help="Comma-separated player_ids to add (must match count of outs)")
    ap.add_argument("--captain", help="player_id to mark as captain", default=None)
    ap.add_argument("--vice", help="player_id to mark as vice", default=None)
    ap.add_argument("--bench-order", default="", help="Comma-separated outfield player_ids for bench1,bench2,bench3")
    ap.add_argument("--bench-gk", default=None, help="player_id to mark as bench GK")

    args = ap.parse_args()
    gws = [int(x.strip()) for x in args.gws.split(",") if x.strip()]
    outs = parse_list(args.outs)
    ins  = parse_list(args.ins)
    if len(outs) != len(ins):
        raise SystemExit(f"OUT count ({len(outs)}) must match IN count ({len(ins)}).")

    base = read_json(Path(args.base_team_state))
    minutes_meta = load_minutes_meta(Path(args.minutes), args.season, gws)

    updated = apply_transfers(
        base=base,
        outs=outs,
        ins=ins,
        minutes_meta=minutes_meta,
        captain=args.captain,
        vice=args.vice,
        bench_order=parse_list(args.bench_order),
        bench_gk=args.bench_gk,
    )

    out_path = Path(args.out_root) / args.candidate / "team_state.json"
    write_json(out_path, updated)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
