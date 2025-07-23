#!/usr/bin/env python3
"""
scripts/clean_fbref_csvs.py

Cleans FBref‐extracted CSVs by:
  • Dropping duplicate rows
  • Using the second row as true column headers
  • Flattening multi‐level headers and normalizing column names
  • Stripping or dropping 'Unnamed' fragments
  • Optionally removing numeric‐only segments from flattened names
  • Renaming output files via regex‐based rules
  • Writing cleaned versions under a target directory
Usage:
  clean_fbref_csvs.py --season 2024_25 \
    --raw-dir data/raw/players/fbref \
    --clean-dir data/processed/players/fbref \
    --rules config/player_rules.json \
    --workers 4
"""
import re
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from typing import Sequence, Tuple, Pattern

# —————————————————————————————————————————————————
# Constants & Types
# —————————————————————————————————————————————————
RenameRule = Tuple[Pattern, str]

# —————————————————————————————————————————————————
# Utility Functions
# —————————————————————————————————————————————————
def normalize_column(col: str) -> str:
    """
    - Strip leading 'Unnamed_<digits>_' (case‐insensitive)
    - Lowercase
    - Replace +,-,/,%, squad→club
    - Collapse non‐alphanumeric to '_'
    - Drop any segment that's pure digits
    """
    s = str(col).strip()
    # 1) drop leading Unnamed_n_
    s = re.sub(r'(?i)^unnamed(?:_\d+)?_', '', s)

    # 2) lowercase
    s = s.lower()

    # 3) custom replacements
    s = s.replace('+', '_plus_').replace('-', '_minus_')
    s = s.replace('/', '_')
    s = re.sub(r'\bsquad\b', 'club', s)
    s = s.replace('%', '_pct')

    # 4) collapse any other non-alphanumeric to underscore
    s = re.sub(r'[^a-z0-9_]+', '_', s)
    s = re.sub(r'__+', '_', s).strip('_')

    # 5) drop _pure_ numeric segments
    parts = [p for p in s.split('_') if not p.isdigit()]
    return '_'.join(parts)

def load_rename_rules(path: Path) -> Sequence[RenameRule]:
    """
    Load rename rules from a JSON file of the form:
      [
        {"pattern": "advanced_goalkeeping", "replacement": "adv_gk"},
        ...
      ]
    Compiles regex patterns with IGNORECASE.
    """
    data = json.loads(path.read_text())
    rules: Sequence[RenameRule] = []
    for entry in data:
        rules.append((re.compile(entry["pattern"], re.IGNORECASE), entry["replacement"]))
    return rules

def apply_rename_rules(stem: str, rules: Sequence[RenameRule]) -> str:
    """Return the first replacement whose pattern matches, else original stem."""
    for pattern, repl in rules:
        if pattern.search(stem):
            return repl
    return stem

# —————————————————————————————————————————————————
# Core Cleaning Logic
# —————————————————————————————————————————————————
def clean_csv(
    raw_csv: Path,
    clean_dir: Path,
    rename_rules: Sequence[RenameRule],
    remove_numeric_segments: bool = True
) -> None:
    """Read, clean, and write a single CSV file."""
    try:
        df = pd.read_csv(raw_csv, header=[0,1])
    except Exception as e:
        logging.error(f"Failed to read {raw_csv}: {e}")
        return

    flat_cols, keep_idx = [], []
    for idx, (cat, field) in enumerate(df.columns):
        cat_str = str(cat).strip()
        field_str = str(field).strip()
        


        cat_str   = re.sub(r'(?i)unnamed(?:_\d+)?_?', '', cat_str).strip()
        field_str = re.sub(r'(?i)unnamed(?:_\d+)?_?', '', field_str).strip()


        # drop columns with truly empty fields
        if not field_str:
            continue

        # drop pure auto-generated categories
        if re.fullmatch(r'Column\d+', cat_str):
            cat_str = ''

        # build flattened name
        if cat_str and cat_str.lower() != field_str.lower():
            flat = f"{cat_str}_{field_str}"
        else:
            flat = field_str

        # optionally drop _pure_ numeric segments only if segment is entire column name
        if remove_numeric_segments:
            parts = [p for p in flat.split('_') if not p.isdigit()]
            flat = '_'.join(parts)

        flat_cols.append(flat)
        keep_idx.append(idx)

    df = df.iloc[:, keep_idx]
    df.columns = [normalize_column(c) for c in flat_cols]
    df = df.drop_duplicates()

    # 1) Define helpers to clean “1,234”→1234
    def to_int(x):
        return int(x.replace(',', ''))

    # DROP matches
    if 'matches' in df.columns:
        df.drop(columns=['matches'], inplace=True)

    if 'notes' in df.columns:
        df.drop(columns=["notes"], inplace = True)

    # UPPERCASE nation
    if 'nation' in df.columns:
        df['nation'] = df['nation'].str[-3:].astype(str).str.upper()

    if "playing_time_min" in df.columns:
        s = df['playing_time_min'].astype(str).str.replace(r'[^\d\.]', '', regex=True)

        
        print(df["playing_time_min"].dtype)
        df['playing_time_min'] = (
        pd.to_numeric(s, errors='coerce')  # floats from strings
          .round(0)                         # drop any .0
          .astype('Int64')                  # nullable integer dtype
        )
        print(df["playing_time_min"].dtype)

        
    # SPLIT pos into primary and alternate poss
    if "pos" in df.columns:
        # uppercase and blank→NaN
        df["pos"] = (
            df["pos"]
            .astype(str)
            .str.upper()
            .str.strip()
            .replace("", np.nan)
        )
        # split on -, /, or ,
        df["pos_list"]    = df["pos"].str.split(r"[-/,]", regex=True)
        # first entry is primary
        df["pos_primary"] = df["pos_list"].str[0]
        # any remaining entries joined by |
        df["pos_alt"]     = df["pos_list"].apply(
            lambda lst: "|".join(lst[1:]) if len(lst) > 1 else ""
        )
    else:
        # ensure columns always exist
        df["pos_list"]    = [[]] * len(df)
        df["pos_primary"] = np.nan
        df["pos_alt"]     = ""

        # CONVERT 'born' FROM FLOAT TO INTEGER YEAR
    if 'born' in df.columns:
        # round off any .0 then cast to pandas nullable Int64
        df['born'] = df['born'].round(0).astype('Int64')

        # CONVERT 'born' FROM FLOAT TO INTEGER YEAR
    if 'age' in df.columns:
        # round off any .0 then cast to pandas nullable Int64
        df['age'] = df['age'].round(0).astype('Int64')

    if 'player' in df.columns:
        # ensure it’s a string and trimmed
        names = df['player'].astype(str).str.strip()
        # first token → first name
        df['first_name'] = names.str.split().str[0]
        # last token → last name
        df['last_name']  = names.str.split().str[-1]
    else:
        # no player column? still create the fields
        df['first_name'] = ''
        df['last_name']  = ''



    # 2) WEEKLY WAGES
    if 'weekly_wages' in df.columns:
        s = df['weekly_wages'].astype(str)
        # extract each currency’s numbers (if missing, leave NaN)
        df['weekly_gbp'] = (
            s.str.extract(r'£\s*([\d,]+)')[0]
            .dropna().apply(to_int)
        )
        df['weekly_eur'] = (
            s.str.extract(r'€\s*([\d,]+)')[0]
            .dropna().apply(to_int)
        )
        df['weekly_usd'] = (
            s.str.extract(r'\$\s*([\d,]+)')[0]
            .dropna().apply(to_int)
        )
        # optionally drop the original string column
        df.drop(columns=['weekly_wages'], inplace=True)

    # 3) ANNUAL WAGES
    if 'annual_wages' in df.columns:
        s = df['annual_wages'].astype(str)
        df['annual_gbp'] = (
            s.str.extract(r'£\s*([\d,]+)')[0]
            .dropna().apply(to_int)
        )
        df['annual_eur'] = (
            s.str.extract(r'€\s*([\d,]+)')[0]
            .dropna().apply(to_int)
        )
        df['annual_usd'] = (
            s.str.extract(r'\$\s*([\d,]+)')[0]
            .dropna().apply(to_int)
        )
        df.drop(columns=['annual_wages'], inplace=True)


    new_stem = apply_rename_rules(raw_csv.stem, rename_rules)
    out_path = clean_dir / f"{new_stem}{raw_csv.suffix}"

    try:
        df.to_csv(out_path, index=False)
        logging.info(f"Cleaned {raw_csv.name} → {out_path.name}")
    except Exception as e:
        logging.error(f"Failed to write {out_path}: {e}")

# —————————————————————————————————————————————————
# Main Entrypoint
# —————————————————————————————————————————————————
def main():
    parser = argparse.ArgumentParser(description="Clean FBref CSV exports.")
    parser.add_argument('--season',       required=True, help="Season identifier (e.g. 2024_25)")
    parser.add_argument('--raw-dir',      type=Path, required=True, help="Input directory for raw CSVs")
    parser.add_argument('--clean-dir',    type=Path, required=True, help="Output directory for cleaned CSVs")
    parser.add_argument('--rules',        type=Path, required=True, help="JSON file of rename rules")
    parser.add_argument('--workers',      type=int, default=1, help="Parallel workers (use >1 to speed up)")
    parser.add_argument('--no-num-drop',  action='store_true', help="Disable numeric-segment removal")
    parser.add_argument('--log-level',    default='INFO', help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), 'INFO'),
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    rename_rules = load_rename_rules(args.rules)
    args.clean_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.raw_dir.glob("*.csv"))
    if not csv_files:
        logging.warning(f"No CSV files found in {args.raw_dir}")
        return

    parallel = args.workers > 1
    if parallel:
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {
                exe.submit(
                    clean_csv, f, args.clean_dir, rename_rules, not args.no_num_drop
                ): f
                for f in csv_files
            }
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Cleaning CSVs"):
                pass
    else:
        for raw_csv in tqdm(csv_files, desc="Cleaning CSVs"):
            clean_csv(raw_csv, args.clean_dir, rename_rules, not args.no_num_drop)

    logging.info("🎉 All done!")

if __name__ == "__main__":
    main()
