#!/usr/bin/env python3
"""
extract_fpl_players.py  –  build a minimal player sheet for every season.

Keeps only:
  • first_name
  • second_name
  • element_type     (FPL numeric position 1–4)
  • position         (GK / DEF / MID / FWD) – derived

Outputs to <out-root>/<season>/min_players.{csv|json}

Example
-------
python extract_fpl_players.py \
       --in-root  data/raw/fpl \
       --out-root data/processed/fpl \
       --format json
"""
from __future__ import annotations
import argparse, logging, sys, json
from pathlib import Path
import pandas as pd

# ───────────────────────── constants ──────────────────────────
KEEP_COLS = {"first_name", "second_name"}


# ───────────────────────── utf-8 helpers ──────────────────────
def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logging.warning(f"Could not parse {path}; starting blank")
    return {}

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")

# ───────────────────────── core logic ─────────────────────────
def build_slim(in_file: Path) -> pd.DataFrame:
    df = pd.read_csv(in_file)
    missing = KEEP_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{in_file.name} missing {missing}")
    slim = df[list(KEEP_COLS)].copy()
    return slim

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract minimal FPL player files")
    ap.add_argument("--in-root",  type=Path, required=True,
                    help="root folder with season subdirs containing cleaned_players.csv")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="mirror root for slim outputs")
    ap.add_argument("--format", choices=("csv", "json"), default="csv",
                    help="output format (default csv)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out_ext = ".json" if args.format == "json" else ".csv"
    done = 0

    for season_dir in sorted(args.in_root.iterdir()):
        if not season_dir.is_dir():
            continue
        season = season_dir.name
        in_file = season_dir / "cleaned_players.csv"
        if not in_file.is_file():
            logging.warning("No cleaned_players.csv in %s – skipped", season)
            continue

        try:
            slim = build_slim(in_file)
        except Exception as exc:
            logging.error("✖︎ %s: %s", season, exc)
            continue

        out_dir  = args.out_root / season
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"players_lookup{out_ext}"

        if args.format == "json":
            save_json(out_file, slim.to_dict(orient="records"))
        else:
            slim.to_csv(out_file, index=False)

        logging.info("✔︎ %s → %s (%d rows)", season, out_file, len(slim))
        done += 1

    if done == 0:
        logging.error("No seasons processed – check input paths.")
        sys.exit(1)
    logging.info("Finished %d season(s) ✅", done)

if __name__ == "__main__":
    main()
