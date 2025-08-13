# prices_registry_builder.py
#!/usr/bin/env python3
"""
Builds per-season player price/state registry from FPL snapshots and deadlines.

Outputs:
  • registry/prices/<SEASON>.json  (nested, deterministic)
  • registry/prices_parquet/<SEASON>.parquet  (flat, fast joins)

Assumptions:
  • Input snapshots contain multiple timestamps ("asof") per GW so we can pick
    the last snapshot <= official deadline. If none exists (scrape lag), we
    pick the earliest snapshot > deadline and WARN.
  • Columns expected (rename via CLI if needed):
      season (str like '2024-2025')
      gw (int)
      player_id (your canonical ID; NOT FPL element id)
      now_cost (int in tenths)  OR price (float in £)
      status (e.g., 'a','d','i','u')
      team_hex (canonical)
      team (short code)
      fpl_pos in {'GK','DEF','MID','FWD'}
      asof (ISO8601 or pandas-parsable datetime in UTC)

Overrides:
  • set_active, set_price, set_team_hex, set_team
  • fpl_pos overrides are IGNORED by design (no reclass mid-season).

Author: you
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

def _read_json(p: Path) -> dict:
    return json.loads(p.read_text("utf-8"))

def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))

def _coerce_price(series: pd.Series) -> pd.Series:
    # If ints like 54 → £5.4; if already floats, pass through.
    if pd.api.types.is_integer_dtype(series):
        return (series.astype(float) / 10.0).round(1)
    return series.astype(float).round(1)

def _status_to_active(s: pd.Series) -> pd.Series:
    # Active if available or doubt at deadline.
    return s.astype(str).str.lower().isin({"a","d"})

def _pick_at_deadline(df: pd.DataFrame, deadline_ts: pd.Timestamp) -> pd.DataFrame:
    """Pick the last snapshot ≤ deadline; if none, first > deadline (WARN)."""
    before = df[df["asof"] <= deadline_ts]
    if not before.empty:
        return before.sort_values("asof").tail(1)
    after = df[df["asof"] > deadline_ts]
    if after.empty:
        # Totally missing; return empty and let caller handle.
        return after
    logging.warning("No snapshot ≤ deadline %s for group; using earliest after (%s)",
                    deadline_ts, after["asof"].min())
    return after.sort_values("asof").head(1)

def build_registry(
    season: str,
    snapshots_path: Path,
    deadlines_json: Path,
    overrides_json: Path | None,
    out_json: Path,
    out_parquet: Path,
    players_master: Path | None = None,
) -> Tuple[dict, pd.DataFrame]:
    dl = _read_json(deadlines_json)
    assert dl.get("season") == season, f"Deadlines season mismatch {dl.get('season')} != {season}"
    tz = dl.get("tz", "UTC")
    deadlines = {int(d["gw"]): pd.Timestamp(d["deadline_utc"]).tz_convert(None) if pd.Timestamp(d["deadline_utc"]).tzinfo else pd.Timestamp(d["deadline_utc"])
                 for d in dl["deadlines"]}

    ov = _read_json(overrides_json)["overrides"] if overrides_json and overrides_json.exists() else {}

    # Load snapshots (Parquet or CSV). Expect flat table with 'asof'.
    snaps = pd.read_parquet(snapshots_path) if snapshots_path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(snapshots_path)
    required = {"season","gw","player_id","status","team_hex","team","fpl_pos","asof"}
    if "price" not in snaps.columns and "now_cost" not in snaps.columns:
        raise ValueError("Snapshots must have 'price' (float £) or 'now_cost' (int tenths).")
    missing = required - set(snaps.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Coerce
    snaps = snaps.copy()
    snaps["season"] = snaps["season"].astype(str)
    snaps = snaps[snaps["season"] == season]
    snaps["gw"] = snaps["gw"].astype(int)
    snaps["asof"] = pd.to_datetime(snaps["asof"], utc=True).dt.tz_localize(None)
    if "price" not in snaps.columns:
        snaps["price"] = _coerce_price(snaps["now_cost"])
    else:
        snaps["price"] = snaps["price"].astype(float).round(1)
    snaps["active"] = _status_to_active(snaps["status"])
    snaps["fpl_pos"] = snaps["fpl_pos"].str.upper()

    # Select per (player_id, gw) the row at deadline.
    picks = []
    for gw, gdf in snaps.groupby("gw"):
        if gw not in deadlines:
            logging.warning("GW %s not present in deadlines; skipping", gw)
            continue
        dl_ts = deadlines[gw]
        def _pick(group: pd.DataFrame) -> pd.DataFrame:
            return _pick_at_deadline(group, dl_ts)
        gsel = gdf.groupby("player_id", group_keys=False).apply(_pick)
        gsel = gsel.assign(deadline_id=gw, deadline_ts=dl_ts)
        picks.append(gsel)
    if not picks:
        raise RuntimeError("No data selected at deadlines. Check snapshots and deadlines.json.")
    sel = pd.concat(picks, ignore_index=True)

    # Apply overrides (season-scoped, GW-level)
    def apply_ov(row):
        pid = str(row["player_id"])
        g = str(int(row["gw"]))
        o = ov.get(pid, {}).get("gw", {}).get(g, None)
        if not o: return row
        if "set_active" in o:
            row["active"] = bool(o["set_active"])
        if "set_price" in o:
            row["price"] = float(o["set_price"])
        if "set_team_hex" in o:
            row["team_hex"] = str(o["set_team_hex"])
        if "set_team" in o:
            row["team"] = str(o["set_team"])
        if "set_fpl_pos" in o:
            logging.warning("Ignoring set_fpl_pos override for %s GW%s (no reclass mid-season).", pid, g)
        return row
    sel = sel.apply(apply_ov, axis=1)

    # Validation
    if players_master and players_master.exists():
        pm = _read_json(players_master)
        bad = [pid for pid in sel["player_id"].astype(str).unique() if pid not in pm]
        if bad:
            logging.warning("Validation: %d player_ids in season not found in players_master.json (first 10 shown): %s", len(bad), bad[:10])
    if (~sel["price"].between(3.5, 15.5)).any():
        raise ValueError("Found price outside [3.5,15.5]. Inspect input/overrides.")
    if (~sel["fpl_pos"].isin({"GK","DEF","MID","FWD"})).any():
        bad = sel.loc[~sel["fpl_pos"].isin({"GK","DEF","MID","FWD"}), ["player_id","fpl_pos"]].drop_duplicates().head(10)
        raise ValueError(f"Invalid fpl_pos detected, sample:\n{bad}")

    # Build nested JSON
    sel["asof_iso"] = pd.to_datetime(sel["asof"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    initial = sel[sel["gw"] == 1].set_index("player_id")["price"].to_dict()
    nested: Dict[str, Any] = {"schema": "prices.v1", "deadline_source": "official_fpl", "season": season, "players": {}}
    for pid, pdf in sel.sort_values(["player_id","gw"]).groupby("player_id"):
        gw_map = {}
        for _, r in pdf.iterrows():
            gw_map[str(int(r["gw"]))] = {
                "price": float(r["price"]),
                "team_hex": r["team_hex"],
                "team": r["team"],
                "fpl_pos": r["fpl_pos"],
                "active": bool(r["active"]),
                "asof": r["asof_iso"],
                "deadline_id": int(r["deadline_id"])
            }
        nested["players"][str(pid)] = {
            "initial_price": float(initial.get(pid, list(gw_map.values())[0]["price"])),
            "gw": gw_map
        }

    # Write JSON deterministically
    _write_json(out_json, nested)

    # Parquet mirror (flat)
    flat = sel[["player_id","gw","price","team_hex","team","fpl_pos","active","asof","deadline_ts"]].copy()
    flat["season"] = season
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    flat.to_parquet(out_parquet, index=False)
    return nested, flat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--snapshots", required=True, type=Path, help="Parquet/CSV with multi-snapshot rows and 'asof'")
    ap.add_argument("--deadlines", required=True, type=Path)
    ap.add_argument("--overrides", type=Path, default=None)
    ap.add_argument("--players-master", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-parquet", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    season = args.season
    out_json = args.out_json or Path(f"registry/prices/{season}.json")
    out_parquet = args.out_parquet or Path(f"registry/prices_parquet/{season}.parquet")
    build_registry(season, args.snapshots, args.deadlines, args.overrides, out_json, out_parquet, args.players_master)
    logging.info("✅ Wrote %s and %s", out_json, out_parquet)

if __name__ == "__main__":
    main()
