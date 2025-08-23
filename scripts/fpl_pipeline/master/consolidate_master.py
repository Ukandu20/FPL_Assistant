#!/usr/bin/env python3
"""
scripts.fpl_pipeline.master.consolidate_master

Build unified FPL master JSON by combining:
  • FBref master (source of truth for identity & seasons)
  • Enriched FPL season player lists (first/second names, team, element_type/fpl_pos hints)
  • Per-season price maps from prices/<season>.json

Output schema (per player_id):
{
  "first_name": "shkodran",
  "second_name": "mustafi",
  "name": "Shkodran Mustafi",           # FBref canonical
  "player_id": "ef07a30f",
  "nation": "GER",
  "born": 1992,
  "career": {
    "2019-20": { "team": "ARS", "position": "DEF", "fpl_pos": "DEF", "league": "ENG-Premier League" },
    ...
  },
  "prices": {                           # optional
    "2019-20": { "1": 5.5, "2": 5.4 },
    ...
  }
}

Rules:
- Names: FBref 'name' at top-level. first/second from FPL (lowercased most-common).
- Season keys normalized to short form 'YYYY-YY'; '2019-20' == '2019-2020'.
- For each season present in FBref career:
    * position/fpl_pos: prefer FBref season 'fpl_position'/'fpl_pos';
      else map FBref 'position' (GK/DF/MF/FW) → GKP/DEF/MID/FWD;
      else derive from FPL row (element_type or fpl_pos/position).
- If FPL shows a player-season missing in FBref career → written to manual_review/missing_career_<season>.csv.
- Prices attached per season if available.

CLI:
py -m scripts.fpl_pipeline.master.consolidate_master ^
  --fbref-master data/processed/registry/master_players.json ^
  --proc-root    data/processed/fpl ^
  --prices-dir   data/processed/registry/prices ^
  --out-json     data/processed/registry/master_fpl.json ^
  --league       "ENG-Premier League" ^
  --log-level    INFO
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from unidecode import unidecode

FPL_POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
FBREF_TO_FPL_POS = {"GK": "GKP", "DF": "DEF", "MF": "MID", "FW": "FWD"}

SEASON_RE_SHORT = re.compile(r"^\d{4}-\d{2}$")
SEASON_RE_LONG  = re.compile(r"^\d{4}-\d{4}$")
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")

def norm_dashes(s: str) -> str:
    return (s or "").translate(_DASHES)

def season_short(s: str) -> str:
    s = norm_dashes(str(s).strip())
    if SEASON_RE_SHORT.fullmatch(s):
        return s
    if SEASON_RE_LONG.fullmatch(s):
        return f"{s[:4]}-{s[-2:]}"
    return s

def season_long(s: str) -> str:
    s = norm_dashes(str(s).strip())
    if SEASON_RE_LONG.fullmatch(s):
        return s
    if SEASON_RE_SHORT.fullmatch(s):
        start = int(s[:4]); end = int(str(start)[:2] + s[-2:])
        return f"{start}-{end}"
    return s

def read_json_flex(p: Path):
    for enc in ("utf-8","utf-8-sig","cp1252","latin-1"):
        try:
            return json.loads(p.read_text(encoding=enc))
        except Exception:
            pass
    return json.loads(p.read_text())

def write_json_utf8(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

def canonical_lower(s: str) -> str:
    s = unidecode(str(s or "")).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def coerce_element_type(val) -> Optional[int]:
    """Accept ints (1..4), numeric strings, or role strings GK/GKP/DEF/DF/MID/MF/FWD/FW."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    # numeric fast-path
    try:
        iv = int(val)
        if iv in (1,2,3,4):
            return iv
    except Exception:
        pass
    s = str(val).strip().upper()
    if s.isdigit():
        iv = int(s)
        return iv if iv in (1,2,3,4) else None
    # string roles
    if s in {"GK","GKP"}: return 1
    if s in {"DF","DEF","D"}: return 2
    if s in {"MF","MID","M"}: return 3
    if s in {"FW","FWD","ST","F"}: return 4
    return None

def coerce_fpl_pos_from_row(row: pd.Series) -> Optional[str]:
    # direct fpl_pos
    if "fpl_pos" in row and pd.notna(row["fpl_pos"]):
        sp = str(row["fpl_pos"]).strip().upper()
        if sp in {"GKP","DEF","MID","FWD"}:
            return sp
        # map GK/DF/MF/FW too
        if sp in FBREF_TO_FPL_POS:
            return FBREF_TO_FPL_POS[sp]
    # element_type
    et = None
    if "element_type" in row:
        et = coerce_element_type(row["element_type"])
    if et in FPL_POS_MAP:
        return FPL_POS_MAP[et]
    # position column (sometimes carries FBref-like group)
    if "position" in row and pd.notna(row["position"]):
        sp = str(row["position"]).strip().upper()
        if sp in {"GKP","DEF","MID","FWD"}:
            return sp
        if sp in FBREF_TO_FPL_POS:
            return FBREF_TO_FPL_POS[sp]
    return None

def map_position(master_pos: Optional[str], master_fpl: Optional[str], row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (position, fpl_pos) in GKP/DEF/MID/FWD.
    Precedence: master_fpl > map(master_pos) > from-row (fpl_pos/element_type/position) > None
    """
    if master_fpl:
        mf = str(master_fpl).strip().upper()
        return mf, mf
    if master_pos:
        mp = FBREF_TO_FPL_POS.get(str(master_pos).strip().upper())
        if mp:
            return mp, mp
    row_pos = coerce_fpl_pos_from_row(row)
    if row_pos:
        return row_pos, row_pos
    return None, None

def load_prices_for_season(prices_dir: Path, season_short_str: str) -> Dict[str, Dict[str, float]]:
    """
    Returns mapping player_id -> {gw: price}
    If file missing, returns {}.
    """
    fp = prices_dir / f"{season_short_str}.json"
    if not fp.is_file():
        # also tolerate long form filename
        for alt in [f"{season_short_str[:4]}-20{season_short_str[-2:]}.json"]:
            fpa = prices_dir / alt
            if fpa.is_file():
                fp = fpa; break
    if fp.is_file():
        data = read_json_flex(fp)
        # Ensure keys are strings
        out = {}
        for pid, gmap in (data or {}).items():
            out[str(pid)] = {str(k): float(v) for k, v in gmap.items()}
        return out
    return {}

def main():
    ap = argparse.ArgumentParser(description="Consolidate FBref master + FPL season files + prices into master_fpl.json.")
    ap.add_argument("--fbref-master", type=Path, required=True)
    ap.add_argument("--proc-root",    type=Path, required=True, help="data/processed/fpl with <season>/season/cleaned_players.csv")
    ap.add_argument("--prices-dir",   type=Path, required=True, help="prices registry dir produced by prices_from_merged.py")
    ap.add_argument("--out-json",     type=Path, required=True, help="output master_fpl.json")
    ap.add_argument("--league",       default="ENG-Premier League")
    ap.add_argument("--log-level",    default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---------- Load FBref master (pid keyed) ----------
    raw_master = read_json_flex(args.fbref_master)
    if isinstance(raw_master, list):
        fb_players = {str(rec.get("player_id") or rec.get("id")): rec for rec in raw_master if rec.get("player_id") or rec.get("id")}
    else:
        fb_players = {str(pid): dict(rec, player_id=str(pid)) for pid, rec in raw_master.items()}

    # Normalize FBref career keys to SHORT season keys for output
    for pid, rec in fb_players.items():
        rec.setdefault("player_id", pid)
        rec.setdefault("name", None)
        rec.setdefault("nation", None)
        rec.setdefault("born", None)
        career = rec.get("career") or {}
        fixed = {}
        for k, v in career.items():
            fixed[season_short(k)] = dict(v) if isinstance(v, dict) else {"team": v}
        rec["career"] = fixed

    # ---------- Iterate seasons under proc-root ----------
    seasons_dirs = [d for d in sorted(args.proc_root.iterdir()) if d.is_dir()]
    missing_by_season: Dict[str, list] = defaultdict(list)

    # name-part tallies
    fn_candidates: Dict[str, Counter] = defaultdict(Counter)
    sn_candidates: Dict[str, Counter] = defaultdict(Counter)

    # season → price map
    season_prices: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sdir in seasons_dirs:
        season = sdir.name
        short = season_short(season)
        season_prices[short] = load_prices_for_season(args.prices_dir, short)

        season_csv = sdir / "season" / "cleaned_players.csv"
        if not season_csv.is_file():
            logging.warning("[%s] missing %s", season, season_csv)
            continue

        df = read_csv(season_csv)
        if df.empty:
            logging.warning("[%s] empty cleaned_players.csv", season)
            continue

        # normalize name parts
        df["first_name"]  = df["first_name"].astype(str).map(canonical_lower) if "first_name" in df.columns else ""
        df["second_name"] = df["second_name"].astype(str).map(canonical_lower) if "second_name" in df.columns else ""

        for _, row in df.iterrows():
            pid = str(row.get("player_id") or "").strip()
            if not pid:
                continue

            # collect name parts
            fn = row.get("first_name") or ""
            sn = row.get("second_name") or ""
            if fn: fn_candidates[pid][fn] += 1
            if sn: sn_candidates[pid][sn] += 1

            rec = fb_players.get(pid)
            if not rec:
                # not in FBref — log & continue
                missing_by_season[short].append({"player_id": pid, "name": row.get("name")})
                continue

            career = rec.setdefault("career", {})
            if short not in career:
                # FBref has no season entry — manual review
                missing_by_season[short].append({"player_id": pid, "name": rec.get("name") or row.get("name")})
                continue

            master_season = career.get(short) or {}
            master_pos     = master_season.get("position")
            master_fpl_pos = master_season.get("fpl_position") or master_season.get("fpl_pos")

            pos, fpl_pos = map_position(master_pos, master_fpl_pos, row)

            # team: prefer FBref season team, else FPL row
            team = master_season.get("team") or (row.get("team") if pd.notna(row.get("team")) else None)

            out_srec = {"team": team, "position": pos, "fpl_pos": fpl_pos}
            # carry league if present in FBref for that season
            if "league" in master_season:
                out_srec["league"] = master_season.get("league")
            career[short] = out_srec

    # ---------- Build output master with names & prices ----------
    out_master: Dict[str, dict] = {}
    for pid, rec in fb_players.items():
        fn = fn_candidates[pid].most_common(1)[0][0] if fn_candidates[pid] else None
        sn = sn_candidates[pid].most_common(1)[0][0] if sn_candidates[pid] else None

        entry = {
            "first_name": fn,
            "second_name": sn,
            "name": rec.get("name"),
            "player_id": pid,
            "nation": rec.get("nation"),
            "born": rec.get("born"),
            "career": rec.get("career") or {}
        }

        # attach prices block if any
        prices_block = {}
        for seas_short, pmap in season_prices.items():
            if pid in pmap and pmap[pid]:
                prices_block[seas_short] = pmap[pid]
        if prices_block:
            entry["prices"] = prices_block

        out_master[pid] = entry

    # ---------- Write output ----------
    write_json_utf8(args.out_json, out_master)
    logging.info("Wrote master FPL JSON: %s (players=%d)", args.out_json, len(out_master))

    # ---------- Manual review(s) ----------
    review_root = args.out_json.parent / "manual_review"
    if missing_by_season:
        review_root.mkdir(parents=True, exist_ok=True)
        total = 0
        for seas_short, rows in missing_by_season.items():
            if not rows:
                continue
            df = pd.DataFrame(rows).drop_duplicates(subset=["player_id"]).sort_values("player_id")
            fp = review_root / f"missing_career_{seas_short}.csv"
            df.to_csv(fp, index=False)
            logging.warning("[%s] missing FBref career entries: %d → %s", seas_short, len(df), fp)
            total += len(df)
        if not total:
            logging.info("No missing career entries to review.")
    else:
        logging.info("No missing career entries to review.")

if __name__ == "__main__":
    main()
