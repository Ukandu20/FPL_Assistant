#!/usr/bin/env python3
"""
update_fixtures_meta.py
──────────────────────────────────────────────────────────────
Incrementally (re)build per-team fixture metadata JSON.

Schema
──────
{
  "2024-2025": {
    "ARS": [
      {
        "game_id"       : "a7b9c3d1",
        "fpl_fixture_id": 12345,          # optional – None if not mapped
        "opponent"      : "MCI",
        "home"          : true,
        "fbref_date"    : "2025-02-12",
        "fpl_date"      : "2025-03-05",    # optional – "" if not mapped
        "gw"            : 28               # optional – null if not mapped
      },
      …
    ],
    …
  },
  "2025-2026": { … }
}

Usage
─────
python update_fixtures_meta.py \
    --season 2024-2025 \
    --fbref-root data/processed/fbref/ENG-Premier\ League \
    --fpl-root   data/processed/fpl \
    --out        data/processed/fixtures_meta.json
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, Tuple, List, Any

import pandas as pd

# ─────────────────────────────────────────────────────────────
def load_json(fp: Path) -> dict:
    if fp.exists():
        try:
            return json.loads(fp.read_text("utf-8"))
        except Exception as e:
            logging.warning("Could not parse %s – starting blank (%s)", fp, e)
    return {}

# ───────────────────────── FBref side ────────────────────────
def load_fbref_fixtures(fbref_root: Path, season: str) -> pd.DataFrame:
    """Concatenate all team_match CSVs for the season into one DataFrame."""
    season_dir = fbref_root / season / "team_match"
    files = list(season_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No team_match CSVs found in {season_dir}")
    frames = []
    cols_needed = ["game_id", "team_id", "opponent_id",
                   "game_date", "is_home"]
    for fp in files:
        df = pd.read_csv(fp, usecols=lambda c: c in cols_needed)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    df_all.rename(columns={"team_id": "team",
                           "opponent_id": "opponent"}, inplace=True)
    df_all["home"] = df_all["is_home"].astype(bool)
    return df_all[["game_id", "team", "opponent", "home", "game_date"]]

# ───────────────────────── FPL side (optional) ───────────────
def build_fpl_map(fpl_root: Path, season: str
                  ) -> Dict[Tuple[str, str, bool], Dict[str, Any]]:
    """
    Returns {(game_id, team, home): {fpl_fixture_id, fpl_date, gw}}.
    Expects a processed FPL fixtures CSV that already contains `game_id`.
    If not present, the mapping dict is empty – the pipeline still works.
    """
    fpl_csv = (Path(fpl_root) / season / "gws" / "merged_gws.csv")
    if not fpl_csv.exists():
        logging.warning("No FPL fixture CSV at %s – FPL fields will be blank",
                        fpl_csv)
        return {}

    cols = ["game_id", "team", "opponent", "kickoff_time",
            "event", "fixture"]
    df = pd.read_csv(fpl_csv, usecols=lambda c: c in cols)
    df["home"] = df["fixture"].str.contains(r"\(H\)", na=False)
    mapping = {}
    for _, r in df.iterrows():
        key = (str(r.game_id), r.team, bool(r.home))
        mapping[key] = {
            "fpl_fixture_id": int(r.fixture) if pd.notna(r.fixture) else None,
            "fpl_date": str(r.kickoff_time) if pd.notna(r.kickoff_time) else "",
            "gw": int(r.event) if pd.notna(r.event) else None,
        }
    return mapping

# ───────────────────────── record assembly ───────────────────
def assemble_records(fb: pd.DataFrame,
                     fpl_map: Dict[Tuple[str, str, bool], Dict[str, Any]]
                     ) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns {team: [fixture_dict,…]} with list sorted by fbref_date ASC.
    """
    recs: Dict[str, Dict[Tuple[str, bool], Dict[str, Any]]] = {}
    for _, r in fb.iterrows():
        key_inner = (r.game_id, r.is_home)
        rec = {
            "game_id":     str(r.game_id),
            "opponent":    str(r.opponent),
            "home":        str(r.is_home),
            "away":        str(r.is_away),
            "fbref_date":  str(r.game_date),
            # defaults – will be overwritten if FPL mapping exists
            "fpl_fixture_id": None,
            "fpl_date": "",
            "gw": None
        }
        rec.update(fpl_map.get((str(r.game_id), r.team, r.is_home, r.is_away), {}))
        recs.setdefault(r.team, {})[key_inner] = rec

    # tidy: convert inner dicts to chronologically-sorted lists
    out: Dict[str, List[Dict[str, Any]]] = {}
    for team, m in recs.items():
        out[team] = sorted(m.values(), key=lambda x: x["fbref_date"])
    return out

# ───────────────────────── incremental write ─────────────────
def incremental_write(out_fp: Path, season: str,
                      new_team_records: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Merges `new_team_records` into JSON at out_fp.
    Collision rule: (game_id, home) uniquely identifies a row;
    newer run wins.
    """
    data = load_json(out_fp)
    season_dict = data.setdefault(season, {})

    for team, new_list in new_team_records.items():
        existing = { (d["game_id"], d["home"]): d
                     for d in season_dict.get(team, []) }
        for rec in new_list:
            existing[(rec["game_id"], rec["home"])] = rec
        # back to list, sorted
        season_dict[team] = sorted(existing.values(),
                                   key=lambda x: x["fbref_date"])

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logging.info("✅ fixtures.json updated (%s)", out_fp)

# ──────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True,
                    help="Season string, e.g. 2024-2025")
    ap.add_argument("--fbref-root", type=Path, required=True,
                    help="Root of cleaned fbref directory (league folder)")
    ap.add_argument("--fpl-root", type=Path, required=True,
                    help="Root of processed FPL data")
    ap.add_argument("--out", type=Path, default=Path("data/config/fixtures.json"))
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s")

    logging.info("▶ loading FBref fixtures for %s …", a.season)
    fb = load_fbref_fixtures(a.fbref_root, a.season)

    logging.info("▶ building FPL mapping …")
    fpl_map = build_fpl_map(a.fpl_root, a.season)

    logging.info("▶ assembling per-team records …")
    team_records = assemble_records(fb, fpl_map)

    logging.info("▶ writing incremental JSON …")
    incremental_write(a.out, a.season, team_records)

if __name__ == "__main__":
    main()
