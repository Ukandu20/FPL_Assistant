#!/usr/bin/env python3
"""
consolidate_fpl_players.py – EPL-only master built from season-level
*FPL with-teampos* files, validated against master_players.json.

Revision 2025-07-28:
  • first_name, second_name, and name → always lower-case
  • born → coerced to int year (never float)
"""
from __future__ import annotations
import argparse, json, logging, re
from pathlib import Path
from typing import Dict, Any, Tuple, Set, List
import pandas as pd

# ------------ tiny JSON helpers -------------
def jload(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logging.error("Cannot read %s", p)
        return {}

def jdump(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_frame(fp: Path) -> pd.DataFrame:
    return pd.read_json(fp) if fp.suffix == ".json" else pd.read_csv(fp)

# ---------- season key equivalence ----------
SEASON_RE = re.compile(r"(\d{4})-(\d{2})")
def season_aliases(folder: str) -> list[str]:
    m = SEASON_RE.fullmatch(folder)
    if m:
        start, end2 = m.groups()
        return [folder, f"{start}-{int(start[:2])*100+int(end2)}"]
    return [folder]

# --------------- helpers --------------------
def _clean_str(val) -> str | None:
    """Lower-case & strip; return None if NaN/empty."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    return s or None

def _clean_born(val) -> int | None:
    """Convert 1997.0 → 1997; NaN/invalid → None."""
    if pd.isna(val):
        return None
    try:
        yr = int(float(val))
        return yr if yr > 0 else None
    except (ValueError, TypeError):
        return None

# --------------- core merge -----------------
def consolidate(fpl_root: Path, master_fb: dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for season_dir in sorted(fpl_root.iterdir()):
        if not season_dir.is_dir():
            continue

        fp = next(
            (
                season_dir / f
                for f in (
                    "players_lookup_enriched_with_teampos.csv",
                    "players_lookup_enriched_with_teampos.json",
                )
                if (season_dir / f).exists()
            ),
            None,
        )
        if not fp:
            continue

        df = load_frame(fp)
        aliases = season_aliases(season_dir.name)

        for _, row in df.iterrows():
            pid = str(row["player_id"])
            fbrec = master_fb.get(pid, {})
            career_fb = fbrec.get("career", {})

            # --- PREMIER-LEAGUE filter -----------------
            srec = next((career_fb[k] for k in aliases if k in career_fb), None)
            if not (srec and srec.get("league") == "ENG-Premier League"):
                continue

            # --- build / update master -----------------
            rec = out.setdefault(
                pid,
                {
                    "first_name": None,
                    "second_name": None,
                    "name": None,
                    "player_id": pid,
                    "nation": None,
                    "born": None,
                    "career": {},
                },
            )

            if rec["first_name"] is None:
                rec["first_name"] = _clean_str(row.get("first_name"))

            if rec["second_name"] is None:
                rec["second_name"] = _clean_str(row.get("second_name"))

            if rec["name"] is None:
                rec["name"] = _clean_str(row.get("name"))

            if rec["player_id"] is None:
                rec["player_id"] = pid

            if rec["nation"] is None and pd.notna(row.get("nation")):
                rec["nation"] = row["nation"]

            if rec["born"] is None:
                rec["born"] = _clean_born(row.get("born"))

            rec["career"][season_dir.name] = {
                "team": row["team"],
                "position": row["position"],
                "fpl_pos": row["fpl_pos"],
            }

    return out

# ------------------ CLI ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players-root", type=Path, required=True,
                    help="root containing <season>/players_lookup_enriched_with_teampos.*")
    ap.add_argument("--fbref-master", type=Path, required=True,
                    help="full master_players.json (has league info)")
    ap.add_argument("--outfile", type=Path, required=True,
                    help="destination master_fpl_players.json")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s")

    fb_master = jload(args.fbref_master)
    consolidated = consolidate(args.players_root, fb_master)
    jdump(args.outfile, consolidated)
    logging.info("Wrote %d EPL player records → %s",
                 len(consolidated), args.outfile)

if __name__ == "__main__":
    main()
