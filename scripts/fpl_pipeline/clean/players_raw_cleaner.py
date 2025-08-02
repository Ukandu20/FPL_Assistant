#!/usr/bin/env python3
"""
clean_fpl.py – unify raw Fantasy-PL CSV dumps, attach FBref ids and
write per-season master sheets, while logging anything that still
doesn't resolve for manual review.

Usage
-----
python clean_fpl.py \
      data/raw/fpl \
      data/processed/fpl \
      --players-json  data/processed/fbref/players.json \
      --override-file data/processed/fpl/overrides.json \
      --threshold 80 \
      --skip-contains xp xP \
      --force
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
from unidecode import unidecode

##############################################################################
# ---------- helpers ---------------------------------------------------------
##############################################################################
LOG = logging.getLogger(__name__)
NAME_SUFFIX_RE = re.compile(r"_[0-9]+$")            # foo_123 → foo


def _asciifold(s: str) -> str:
    """Lower-case, accent-fold and squeeze whitespace."""
    s = unidecode(s)
    return " ".join(s.lower().split())


def canonical(name: str) -> str:
    """Convert any of the following to one canonical form:

    * underscores, mixed case, extra internal spaces
    * suffixes like '_534'
    * accented characters
    * HTML entities like  O&#039;Shea
    """
    name = html.unescape(name)                      # &amp; → &,  &#039; → '
    name = NAME_SUFFIX_RE.sub("", name)             # drop trailing _123
    name = name.replace("_", " ").replace("-", " ")
    name = _asciifold(name)

    # strip the lingering punctuation that often differs
    # e.g. O'Shea / OShea ; Luiz,Jr ; etc.
    name = re.sub(r"[^\w\s]", "", name)             # keep letters, digits, space
    return name


# ---------------------------------------------------------------------------


def load_global_players(path: Path) -> Dict[str, str]:
    """Return mapping canonical_name → player_id using the big players.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {canonical(raw): pid for raw, pid in data.items()}


def load_season_players(root_fbref: Path, season_tag: str) -> Dict[str, str]:
    """
    Build a name→id map from
    data/processed/fbref/ENG-Premier League/<season>/player_season/season_players.json
    """
    # season_tag comes in as '2024-25', '2019-20', …
    # fbref folder uses the *full* years: '2024-2025', '2019-2020', …
    start, end_two = season_tag.split("-")
    end_full = int(start[:2] + end_two)             # 2024 + 25 → 2025
    fbref_season = f"{start}-{end_full}"

    season_file = (
        root_fbref
        / "ENG-Premier League"
        / fbref_season
        / "player_season"
        / "season_players.json"
    )
    if not season_file.is_file():
        return {}

    try:
        data = json.loads(season_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOG.warning("Malformed season_players.json in %s – skipping", season_file)
        return {}

    mapping: Dict[str, str] = {}
    for pid, rec in data.items():
        mapping[canonical(rec["name"])] = pid
    return mapping


def load_overrides(path: Path) -> Dict[str, Dict[str, str] | str]:
    if not path.is_file():
        return {}
    ov = json.loads(path.read_text())
    fixed: Dict[str, Dict[str, str] | str] = {}
    for raw_key, val in ov.items():
        key = canonical(raw_key)
        # accept both {"key": "id"}  and  {"key": {"id": "...", "name": "..."}}
        if isinstance(val, str):
            fixed[key] = {"id": val}
        else:
            fixed[key] = {"id": val["id"], "name": val.get("name")}
    return fixed


def best_fuzzy_match(name: str,
                     candidates: Dict[str, str],
                     threshold: int) -> Tuple[str, int] | None:
    """Return (matched_key, score) or None."""
    best_key, best_score = None, 0
    for cand in candidates:
        score = fuzz.ratio(name, cand)
        if score > best_score:
            best_key, best_score = cand, score
    return (best_key, best_score) if best_score >= threshold else None

##############################################################################
# ---------- core ------------------------------------------------------------
##############################################################################


def process_season(season_dir: Path,
                   out_dir: Path,
                   global_map: Dict[str, str],
                   overrides: Dict[str, Dict[str, str]],
                   root_fbref: Path,
                   threshold: int,
                   skip_substrings: Set[str]) -> None:
    """Process one season folder – merge, clean, write master & unmatched log."""

    csv_files = sorted(
        p for p in season_dir.glob("**/*.csv")
        if not any(substr in p.as_posix() for substr in skip_substrings)
    )
    if not csv_files:
        LOG.warning("No input CSV files in %s", season_dir)
        return

    # -------- season-specific FBref mapping --------------------------------
    season_map = load_season_players(root_fbref, season_dir.name)
    mapping: Dict[str, str] = {**global_map, **season_map}   # season overrides global

    unmatched: Set[str] = set()
    dataframes: List[pd.DataFrame] = []

    # --------------------------------------------------------------------- #
    # Per-file handler (run in threads)                                     #
    # --------------------------------------------------------------------- #
    def handle_file(f: Path) -> pd.DataFrame | None:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            LOG.warning("Could not parse %s – %r; skipping", f, exc)
            return None
        if df.empty:
            LOG.warning("%s has zero rows; skipping.", f)
            return None

        # ---- column sanity ------------------------------------------------
        alias_map = {
            "player": "name",
            "player_name": "name",
            "Player Name": "name",
        }
        for old, new in alias_map.items():
            if old in df.columns and "name" not in df.columns:
                df = df.rename(columns={old: new})
                break
        if "name" not in df.columns:
            LOG.warning("%s has no 'name' column (%s); skipping.", f, list(df.columns))
            return None

        # ---- id resolution ------------------------------------------------
        def resolve(raw: str) -> str | None:
            key = canonical(raw)

            # 1️⃣  direct hits / manual overrides first
            if key in overrides:
                return overrides[key]["id"]
            if key in mapping:
                return mapping[key]

            # 2️⃣  generate smart aliases from the tokens
            tokens = key.split()
            n = len(tokens)
            if n >= 2:
                # every contiguous 2-gram and (n-1)-gram
                variants: set[str] = set()

                # • any two adjacent tokens (handles dropped middle names)
                if n > 2:
                    for i in range(n - 1):
                        variants.add(" ".join(tokens[i : i + 2]))

                # • first + last token (already helped for Cheick Oumar Doucoure)
                variants.add(f"{tokens[0]} {tokens[-1]}")

                # • all-but-first & all-but-last (already in previous version)
                variants.add(" ".join(tokens[1:]))
                variants.add(" ".join(tokens[:-1]))

                # • remove generational suffixes/jr/sr/iii etc.
                generational = {"jr", "sr", "ii", "iii", "iv"}
                if tokens[-1] in generational:
                    variants.add(" ".join(tokens[:-1]))

                # • comma-style “last, first” → “first last”
                if "," in raw:
                    parts = [canonical(p) for p in raw.replace(",", "").split()]
                    if len(parts) >= 2:
                        variants.add(" ".join(reversed(parts)))

                for v in variants:
                    # prefer an override hit if present
                    if v in overrides:
                        return overrides[v]["id"]
                    if v in mapping:
                        return mapping[v]

            # 3️⃣  fuzzy last resort (Token SET ratio is kinder to order/middles)
            match = best_fuzzy_match(key, mapping, threshold)
            return mapping[match[0]] if match else None

        # ── id column ────────────────────────────────────────────────
        df["player_id"] = df["name"].apply(resolve)

        # ── optional name rewrite from overrides ────────────────────
        def maybe_fix(raw):
            key = canonical(raw)
            info = overrides.get(key)
            return info["name"] if info and info.get("name") else raw

        df["name"] = df["name"].apply(maybe_fix)
        unmatched.update(df.loc[df["player_id"].isna(), "name"].tolist())
        return df

    # --------------------------------------------------------------------- #
    # Run handlers concurrently                                             #
    # --------------------------------------------------------------------- #
    with ThreadPoolExecutor() as ex:
        futures = {ex.submit(handle_file, f): f for f in csv_files}
        for fut in tqdm(as_completed(futures),
                        total=len(csv_files),
                        desc=season_dir.name):
            res = fut.result()
            if res is not None:
                dataframes.append(res)

    if not dataframes:
        LOG.warning("No data collected for %s; skipping write.", season_dir)
        return

    season_master = pd.concat(dataframes, ignore_index=True)

    # Put 'player_id' immediately after 'name' for readability
    cols = list(season_master.columns)
    if "player_id" in cols and "name" in cols:
        cols.insert(cols.index("name") + 1, cols.pop(cols.index("player_id")))
        season_master = season_master[cols]

    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = out_dir / "master.csv"
    season_master.to_csv(master_path, index=False)
    LOG.info("Wrote master: %s", master_path)

    # ---- unmatched names --------------------------------------------------
    # ---- unmatched names --------------------------------------------------
    review_dir  = out_dir.parent / "_manual_review"
    review_file = review_dir / f"unmatched_{season_dir.name}.txt"

    if unmatched:
        review_dir.mkdir(parents=True, exist_ok=True)
        review_file.write_text("\n".join(sorted(unmatched)), encoding="utf-8")
        LOG.info("%d unmatched names logged → %s", len(unmatched), review_file)
    else:
        # nothing left unresolved – remove any stale list from previous runs
        if review_file.exists():
            review_file.unlink()
            LOG.info("All players matched – removed %s", review_file)

##############################################################################
# ---------- cli -------------------------------------------------------------
##############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean FPL data dumps.")
    parser.add_argument("in_root", type=Path, help="raw season folders root")
    parser.add_argument("out_root", type=Path, help="processed season root")
    parser.add_argument("--players-json", type=Path, required=True)
    parser.add_argument("--override-file", type=Path, default=None)
    parser.add_argument("--threshold", type=int, default=80,
                        help="fuzzy match minimum ratio (0-100)")
    parser.add_argument("--skip-contains", nargs="*", default=[], metavar="SUBSTR",
                        help="ignore any CSV whose *path* contains these substrings")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing master files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    global_map = load_global_players(args.players_json)
    overrides = load_overrides(args.override_file) if args.override_file else {}

    root_fbref = args.players_json.parent     # …/fbref

    for season_dir in sorted(args.in_root.iterdir()):
        if not season_dir.is_dir():
            continue

        out_dir = args.out_root / season_dir.name
        if out_dir.exists() and not args.force:
            LOG.info("%s already processed – skipping (use --force to rebuild)",
                     season_dir.name)
            continue
        if out_dir.exists() and args.force:
            shutil.rmtree(out_dir)            # clean slate

        process_season(
            season_dir,
            out_dir,
            global_map,
            overrides,
            root_fbref,
            args.threshold,
            set(args.skip_contains),
        )


if __name__ == "__main__":
    main()
