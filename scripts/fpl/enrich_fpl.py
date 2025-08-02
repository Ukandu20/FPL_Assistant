#!/usr/bin/env python3
"""
add_team_pos.py – enrich FPL look-up per season with
                  team, position, nation and born (year).

The file written becomes
<season>/players_lookup_enriched_with_teampos.csv|json

Rows without a season entry in master_players.json are skipped
and logged to  _manual_review/missing_teampos_<season>.json
"""
from __future__ import annotations
import argparse, json, logging, re, sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import datetime as datetime


# ─────────────────── IO helpers ────────────────────
def jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logging.error("Could not read %s", p)
        return {}

def jdump(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ───────────── season-key normaliser ───────────────
def normalise_season(dir_name: str) -> Tuple[str, str | None]:
    """
    Returns (exact_key, alt_key)
      exact_key = dir name itself
      alt_key   = YYYY-YYYY if dir is YYYY-YY
    """
    m = re.fullmatch(r"(\d{4})-(\d{2})", dir_name)
    if m:
        start, end_short = m.groups()
        end_full = str(int(start[:2]) * 100 + int(end_short))
        return dir_name, f"{start}-{end_full}"
    return dir_name, None


# ─────────── dataframe enrichment ─────────────
def enrich_dataframe(df: pd.DataFrame,
                     season_key: str,
                     alt_key: str | None,
                     master: Dict[str, dict]
                     ) -> Tuple[pd.DataFrame, list]:
    rows_keep, review = [], []

    for _, row in df.iterrows():
        pid = str(row.get("player_id"))
        rec = master.get(pid, {})

        # season entry ---------------------------
        career = rec.get("career", {})
        srec   = career.get(season_key) or (career.get(alt_key) if alt_key else None)

        if srec and srec.get("team") and srec.get("position"):
            row_out = row.copy()
            row_out["team"]     = srec["team"]
            row_out["position"] = srec["position"]
            row_out["fpl_pos"] = srec.get("fpl_pos", srec.get("position"))
            # top-level extras --------------------
            row_out["nation"]   = rec.get("nation")
            row_out["born"]     = rec.get("born")
            
            rows_keep.append(row_out)
        else:
            review.append({
                "player_id":   pid,
                "name":        row.get("name"),
                "first_name":  row.get("first_name"),
                "second_name": row.get("second_name"),
                "reason":      "no season/team/position in master"
            })

    return pd.DataFrame(rows_keep), review


# ─────────── per-season wrapper ─────────────
def process_season(season_dir: Path,
                   master: Dict[str, dict],
                   out_root: Path,
                   force_ext: str | None):
    in_file = next((season_dir / f for f in (
                       "players_lookup_enriched.csv",
                       "players_lookup_enriched.json")
                    if (season_dir / f).exists()), None)
    if not in_file:
        logging.warning("%s: enriched lookup file not found – skipped",
                        season_dir.name)
        return

    ext = force_ext or in_file.suffix.lower()
    out_file = (out_root / season_dir.name /
                f"players_lookup_enriched_with_teampos{ext}")
    review_file = (out_root / "_manual_review" /
                   f"missing_teampos_{season_dir.name}.json")

    df = pd.read_json(in_file) if in_file.suffix == ".json" else pd.read_csv(in_file)

    key1, key2 = normalise_season(season_dir.name)
    df_enriched, unmatched = enrich_dataframe(df, key1, key2, master)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    if "born" in df_enriched.columns:          # <<< new
        df_enriched["born"] = df_enriched["born"].astype("Int64")
    
    if out_file.suffix == ".json":
        jdump(out_file, df_enriched.to_dict(orient="records"))
    else:
        df_enriched.to_csv(out_file, index=False)

    if unmatched:
        jdump(review_file, unmatched)
        logging.info("✖︎ %d players missing team/pos → %s",
                     len(unmatched), review_file)
    else:
        review_file.unlink(missing_ok=True)
        logging.info("✔︎ all players enriched for %s", season_dir.name)


# ───────────────────── CLI ───────────────────────
def main():
    ap = argparse.ArgumentParser(description="Add team/position/nation/born")
    ap.add_argument("--players-root", type=Path, required=True)
    ap.add_argument("--master-file",  type=Path, required=True)
    ap.add_argument("--out-root",     type=Path,
                    help="default = players-root")
    ap.add_argument("--format",      choices=("csv","json"))
    ap.add_argument("--log-level",   default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    master   = jload(args.master_file)
    out_root = args.out_root or args.players_root

    processed = 0
    for season_dir in sorted(args.players_root.iterdir()):
        if not season_dir.is_dir():
            continue
        logging.info("Season %s …", season_dir.name)
        process_season(season_dir, master, out_root, args.format)
        processed += 1

    if not processed:
        logging.error("No seasons processed – check paths.")
        sys.exit(1)
    logging.info("Finished %d season(s) ✅", processed)


if __name__ == "__main__":
    main()
