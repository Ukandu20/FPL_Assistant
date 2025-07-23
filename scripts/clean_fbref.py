#!/usr/bin/env python3
"""
clean_fbref_csvs.py       (ID-less build)

• Handles FBref exports with either three-row headers **or** a single header
  row (player_match_schedule / team_match_schedule).
• Routes files into:
      <clean-dir>/<LEAGUE>/<SEASON>/<player|team>_<season|match>/<stat>.csv
  and strips the leading prefix from the filename once it’s in the folder.
• Adds `game_date`, `home`, `away` columns (keeps the original `game`).
• Converts `age` values like "24-234" → 24 (nullable Int64).
• Converts numeric-looking columns, logs any that stay `object`.
• Never coerces or drops id-like and text columns (`game`, `home`, `away`,
  `game_id`, `player_id`, `team_id`).
• Safe to parallelise with `--workers`.

Usage
-----
py scripts/clean_fbref_csvs.py ^
  --season 2024_25 ^
  --raw-dir "data/raw/fbref/ENG-Premier League/2024-2025" ^
  --clean-dir "data/processed" ^
  --workers 4
"""
# ───────────────── stdlib ─────────────────────
import argparse, json, logging, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Pattern, Sequence, Tuple

# ───────────────── 3-party ────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm

# ────────────── helpers / constants ───────────
RenameRule = Tuple[Pattern, str]

PREFIXES = (
    "player_season_", "player_match_",
    "team_season_",   "team_match_",
)

DROP_COLS = {'matches', 'notes'}     # columns to remove outright
ID_LIKE   = {'game_id', 'player_id', 'team_id', 'game', 'home', 'away'}

# ------------------------------------------------
def normalize(col: str, drop_numeric: bool = True) -> str:
    """Lowercase, squash punctuation, rename squad→club, drop numeric tokens."""
    s = re.sub(r'(?i)^unnamed(?:_\d+)?_', '', str(col)).strip().lower()
    s = (s.replace('+', '_plus_').replace('-', '_minus_')
           .replace('/', '_').replace('%', '_pct'))
    s = re.sub(r'\bsquad\b', 'club', s)
    s = re.sub(r'[^a-z0-9_]+', '_', s)
    s = re.sub(r'__+', '_', s).strip('_')
    if drop_numeric:
        s = '_'.join(p for p in s.split('_') if not p.isdigit())
    return s


def load_rules(path: Path | None) -> Sequence[RenameRule]:
    if not path:
        return []
    data = json.loads(path.read_text())
    return [(re.compile(d["pattern"], re.I), d["replacement"]) for d in data]


def apply_rules(stem: str, rules: Sequence[RenameRule]) -> str:
    for pat, repl in rules:
        if pat.search(stem):
            return repl
    return stem


def strip_prefix(stem: str) -> str:
    for p in PREFIXES:
        if stem.startswith(p):
            return stem[len(p):]
    return stem


def choose_subfolder(stem: str) -> str:
    if   stem.startswith("player_season_"): return "player_season"
    elif stem.startswith("player_match_"):  return "player_match"
    elif stem.startswith("team_season_"):   return "team_season"
    elif stem.startswith("team_match_"):    return "team_match"
    return "player_match"   # default for naked stat names


# ────────────── core cleaning fn ───────────────
def clean_csv(path: Path, season: str, clean_root: Path, rules,
              drop_numeric: bool):

    # 1 ─ read with appropriate header depth
    schedule_mode = path.stem.endswith('schedule')
    header_rows   = [0] if schedule_mode else [0, 1, 2]
    try:
        df = pd.read_csv(path, header=header_rows)
    except Exception as e:
        logging.error(f"[read] {path.name}: {e}")
        return

    # Short-circuit: write schedule table unchanged except prefix-strip + routing
    if schedule_mode:
        _write_out(df, path, season, clean_root, rules)
        return

    # 2 ─ flatten multi-index header (3-level)
    keep_idx, flat_cols, seen = [], [], set()
    for idx, col in enumerate(df.columns):
        lvl0, lvl1, lvl2 = [str(x).strip() for x in col]
        label = next((x for x in (lvl2, lvl1, lvl0)
                      if x and not re.match(r'(?i)unnamed', x)), '')
        if not label:
            continue
        flat = normalize(label, drop_numeric)
        if flat in seen:
            flat = normalize(f"{lvl1}_{label}", drop_numeric)
        if flat in seen:
            flat = normalize(f"{lvl0}_{flat}", drop_numeric)
        keep_idx.append(idx); flat_cols.append(flat); seen.add(flat)

    df = df.iloc[:, keep_idx]
    df.columns = flat_cols
    df.drop_duplicates(inplace=True)

    # 3 ─ housekeeping
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    if 'nation' in df.columns:
        df['nation'] = (df['nation'].astype(str)
                                  .str[-3:].str.upper()
                                  .replace({'': np.nan}))

    # playing_time_min / born first-pass numeric
    for col in ('playing_time_min', 'born'):
        if col in df:
            df[col] = (pd.to_numeric(
                df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce')
                .round(0).astype('Int64'))

    # age: keep years before hyphen
    # ── AGE: keep years before hyphen, coerce safely ─────────────
    if 'age' in df.columns:
        df['age'] = (
            pd.to_numeric(                               # ← handles blanks
                df['age']
                .astype(str)
                .str.split('-', n=1).str[0]            # left side of “24-234”
                .str.replace(r'[^\d]', '', regex=True),# strip stray chars
                errors='coerce'
            )
            .round(0)
            .astype('Int64')                             # nullable integer
        )

    # game_id to Int64
    if 'game_id' in df.columns:
        df['game_id'] = (pd.to_numeric(
            df['game_id'].astype(str).str.replace(r'[^\d]', '', regex=True),
            errors='coerce').astype('Int64'))

    # 4 ─ split `game` into date/home/away
    if 'game' in df.columns and df['game'].dtype == object:
        patt  = r'^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<home>.+?)-(?P<away>.+)$'
        parts = df['game'].str.extract(patt)
        insert = df.columns.get_loc('game') + 1
        df.insert(insert,     'game_date',
                  pd.to_datetime(parts['date'], errors='coerce'))
        df.insert(insert + 1, 'home', parts['home'].str.strip())
        df.insert(insert + 2, 'away', parts['away'].str.strip())
        ID_LIKE.update({'home', 'away'})  # ensure they’re excluded below

    # 5 ─ generic numeric coercion (& audit)
    suspects = []
    obj_cols = [c for c in df.select_dtypes(include='object').columns
                if c not in ID_LIKE]

    for c in obj_cols:
        s_orig  = df[c].astype(str)
        s_clean = s_orig.str.replace(r'[,\u202f]', '', regex=True).str.rstrip('%')
        numeric = pd.to_numeric(s_clean, errors='coerce')

        if numeric.notna().sum() > 0 and numeric.isna().sum() < numeric.size:
            df[c] = numeric.astype('float64')
        else:
            if s_orig.str.contains(r'\d').any():
                suspects.append(c)

    if suspects:
        logging.warning(f"{path.name}: non-numeric after coercion → {suspects}")

    # 6 ─ write out
    _write_out(df, path, season, clean_root, rules)


# helper to write cleaned df to disk
def _write_out(df: pd.DataFrame, path: Path, season: str,
               clean_root: Path, rules):

    stem   = apply_rules(path.stem, rules)
    league = path.parts[path.parts.index('fbref') + 1]  # e.g. ENG-Premier League
    sub    = choose_subfolder(stem)

    out_dir = clean_root / league / season / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{strip_prefix(stem)}.csv"
    try:
        df.to_csv(out_file, index=False)
        logging.info(f"✔ {path.name} → {out_file.relative_to(clean_root)}")
    except Exception as e:
        logging.error(f"[write] {out_file.name}: {e}")


# ──────────────── CLI wrapper ────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', required=True)
    ap.add_argument('--all-seasons', action='store_true',
                help='Walk every sub-folder of --raw-dir as a season')
    ap.add_argument('--raw-dir',   type=Path, required=True)
    ap.add_argument('--clean-dir', type=Path, required=True)
    ap.add_argument('--rules',     type=Path)
    ap.add_argument('--workers',   type=int, default=4)
    ap.add_argument('--no-num-drop', action='store_true')
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format='%(asctime)s %(levelname)s: %(message)s')

    rules = load_rules(args.rules)
    csvs  = sorted(args.raw_dir.rglob('*.csv'))
    if not csvs:
        logging.warning(f"No CSVs found under {args.raw_dir}")
        return

    drop_nums = not args.no_num_drop
    task = lambda f: clean_csv(f, args.season, args.clean_dir,
                               rules, drop_nums)

    if args.workers > 1:
        with ThreadPoolExecutor(args.workers) as ex:
            list(tqdm(ex.map(task, csvs), total=len(csvs), desc='Cleaning'))
    else:
        for f in tqdm(csvs, desc='Cleaning'):
            task(f)

    logging.info("🎉  All done!")


if __name__ == "__main__":
    main()
