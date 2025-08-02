#!/usr/bin/env python3
r"""fixtures_meta_builder.py – Batch-capable builder for **fixture_calendar.csv**
───────────────────────────────────────────────────────────────────────────────
Adds **home** and **away** (three‑letter short codes) to the final CSV, in
addition to the hex `home_hex` / `away_hex` columns.

Batch rules (unchanged)
• `--season` → single season; omit → loop over every folder in `--fpl-root`.
• `--force`  → overwrite existing outputs.

Output columns (order)
---------------------
  fpl_id, fbref_id, gw_orig, gw_played,
  date_sched, date_played,
  home, away, home_hex, away_hex,
  status, sched_missing
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd, numpy as np

# ───────────────────── helpers ──────────────────────────────────────────────

def load_json(p: Path) -> dict: return json.loads(p.read_text("utf-8"))

def canon(s: str) -> str: return " ".join(s.lower().split())

def build_maps(long2hex: Dict[str,str], long2code: Dict[str,str]):
    name2hex  = {canon(k): v.lower() for k, v in long2hex.items()}
    name2code = {canon(k): v.upper() for k, v in long2code.items()}
    code2hex  = {name2code[k]: v for k,v in name2hex.items() if k in name2code}
    return name2hex, name2code, code2hex

def normalise_date(series: pd.Series) -> pd.Series:
    if series.dt.tz is not None:
        series = series.dt.tz_convert(None)
    return series.dt.floor("D")

# ─────────────────── single‑season builder ─────────────────────────────────

def build_fixture_calendar(season: str, fpl_csv: Path, fb_csv: Path,
                           team_map_fp: Path, short_map_fp: Path,
                           out_dir: Path, teams_csv: Path|None=None,
                           force: bool=False) -> bool:
    dst_dir = out_dir/season; out_csv = dst_dir/"fixture_calendar.csv"
    if out_csv.exists() and not force:
        logging.info("%s • already done – skip (use --force)", season); return False
    if not (fpl_csv.is_file() and fb_csv.is_file()):
        logging.warning("%s • missing fixture or schedule csv – skipped", season); return False

    fpl = pd.read_csv(fpl_csv, parse_dates=["kickoff_time"])
    fb  = pd.read_csv(fb_csv,  parse_dates=["game_date"])
    name2hex,name2code,code2hex = build_maps(load_json(team_map_fp),load_json(short_map_fp))

    # FPL prep --------------------------------------------------------------
    fpl = fpl.rename(columns={"id":"fpl_id","event":"gw_orig",
                              "team_h":"home_id_fpl","team_a":"away_id_fpl"})
    fpl["status"] = np.where(fpl["finished"],"finished","scheduled")
    fpl["date_played"] = normalise_date(fpl["kickoff_time"])
    fpl["date_sched"]  = fpl["date_played"]; fpl["sched_missing"] = 1

    if teams_csv is None:
        teams_csv = fpl_csv.with_name("teams.csv")
        if not teams_csv.exists():
            logging.warning("%s • teams.csv missing – skipped",season); return False
    teams_df = pd.read_csv(teams_csv,usecols=["id","name"])
    id2name  = dict(zip(teams_df.id, teams_df.name.map(canon)))

    for side in ("home","away"):
        fpl[f"{side}_long"] = fpl[f"{side}_id_fpl"].map(id2name)
        fpl[f"{side}"]      = fpl[f"{side}_long"].map(name2code)   # short code col
        fpl[f"{side}_hex"]  = fpl[f"{side}"].map(code2hex)

    # FBref ---------------------------------------------------------------
    fb = fb[fb.get("is_home",1)==1].copy()
    fb["date_played"] = normalise_date(fb["game_date"])
    fb_match = fb[["game_id","home","away","date_played"]]

    cal = fpl.merge(fb_match,on=["home","away","date_played"],how="left")
    missing = cal[cal.game_id.isna()]

    cal["gw_played"] = cal["gw_orig"]
    cal = cal[["fpl_id","game_id","gw_orig","gw_played",
               "date_sched","date_played",
               "home","away","home_hex","away_hex",
               "status","sched_missing"]] \
             .rename(columns={"game_id":"fbref_id"})

    dst_dir.mkdir(parents=True,exist_ok=True)
    cal.to_csv(out_csv,index=False)
    logging.info("%s • fixture_calendar.csv (%d rows)",season,len(cal))
    if not missing.empty:
        missing.to_csv(dst_dir/"_manual_fbref_match.csv",index=False)
        logging.warning("%s • %d rows lack fbref_id",season,len(missing))
    return True

# ───────────────────── batch driver (unchanged) ─────────────────────────––

def run_batch(seasons: List[str], fpl_root: Path, fbref_league: Path,
              team_map: Path, short_map: Path, out_dir: Path, force: bool):
    for season in seasons:
        fpl_csv = fpl_root/season/"fixtures.csv"
        fb_csv  = fbref_league/season/"team_match"/"schedule.csv"
        teams_csv = fpl_root/season/"teams.csv"
        try:
            build_fixture_calendar(season,fpl_csv,fb_csv,
                                   team_map,short_map,out_dir,
                                   teams_csv=teams_csv,force=force)
        except Exception:
            logging.exception("%s • unhandled error",season)

# ───────────────────────────── CLI ─────────────────────────────────────––

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season")
    ap.add_argument("--fpl-root",type=Path,default=Path("data/raw/fpl"))
    ap.add_argument("--fbref-league-dir",type=Path,default=Path("data/processed/fbref/ENG-Premier League"))
    ap.add_argument("--team-map",required=True,type=Path)
    ap.add_argument("--short-map",required=True,type=Path)
    ap.add_argument("--out-dir",type=Path,default=Path("data/processed/fixtures"))
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--log-level",default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),format="%(levelname)s: %(message)s")
    seasons = [args.season] if args.season else sorted(d.name for d in args.fpl_root.iterdir() if d.is_dir())
    if not seasons:
        logging.error("No seasons found"); return
    run_batch(seasons, args.fpl_root, args.fbref_league_dir,
              args.team_map, args.short_map, args.out_dir, args.force)

if __name__ == "__main__":
    main()
