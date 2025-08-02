#!/usr/bin/env python3
"""
fpl_clean_pipeline.py  –  STEP 7  (enrich with player metadata)
────────────────────────────────────────────────────────────────────────────
    • lower-case first/second names
    • rename now_cost → price
    • drop rows with element_type == 'am'
    • cast numeric columns (except *date* / *time*)
    • enrich each matched row with:
        player_id, name, nation, born, position
    • keep enriched rows in season/cleaned_players.csv
    • move still-unmatched rows to _unwanted/cleaned_players_unmatched.csv
"""

from __future__ import annotations
import argparse, json, logging, unicodedata, re
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd


# ───────────────────────── helper: canonical key ────────────────────────
def canonical(s: str) -> str:
    """accent-fold, strip punctuation, collapse spaces, lower-case."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.lower().split())


# ───────────────── master & override look-ups ───────────────────────────
def load_master(master_fp: Path) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Returns pid→record  and  canonical(first second)→pid."""
    raw = json.loads(master_fp.read_text("utf-8"))
    records = raw.values() if isinstance(raw, dict) else raw
    pid2rec, key2pid = {}, {}
    for rec in records:
        pid = rec.get("player_id") or rec.get("id")
        if not pid:
            continue
        pid2rec[pid] = rec
        key2pid[canonical(f"{rec.get('first_name','')} {rec.get('second_name','')}")] = pid
    logging.info("Loaded %d players from master", len(pid2rec))
    return pid2rec, key2pid


def load_overrides(fp: Path | None) -> Dict[str, str]:
    """
    Accepts flat file  { "first | second": "pid", ... }  or
            nested      { "alias": {"id": "pid"}, ... }
    Returns canonical(first second) → pid.
    """
    if not fp or not fp.is_file():
        return {}
    raw = json.loads(fp.read_text("utf-8"))
    out: Dict[str, str] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            pid = val.get("id")
            alias = val.get("name", key)
        else:
            pid = val
            alias = key
        if not pid:
            continue
        out[canonical(alias.replace("|", " "))] = pid
    logging.info("Loaded %d override aliases", len(out))
    return out


# ───────────────────────── helper: numeric cast ─────────────────────────
def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    mask_excl = df.columns.str.contains("date", case=False) | \
                df.columns.str.contains("time", case=False)
    for col in df.columns[~mask_excl]:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


# ───────────────────────── per-season handler ───────────────────────────
def handle_season(season_raw: Path,
                  season_out: Path,
                  pid2rec: Dict[str, dict],
                  master_key2pid: Dict[str, str],
                  override_key2pid: Dict[str, str]):
    src = season_raw / "cleaned_players.csv"
    if not src.exists():
        logging.warning("%s: cleaned_players.csv missing – skipped", season_raw.name)
        return

    df = pd.read_csv(src, engine="python")

    # 1️⃣ lower-case first/second names
    for col in ("first_name", "second_name"):
        if col in df.columns:
            df[col] = df[col].str.lower()

    # 2️⃣ rename now_cost → price
    if "now_cost" in df.columns:
        df = df.rename(columns={"now_cost": "price"})

    # 3️⃣ drop 'am' rows
    if "element_type" in df.columns:
        mask_am = df["element_type"].str.lower() == "am"
        df = df[~mask_am].reset_index(drop=True)

    # 4️⃣ numeric cast
    df = cast_numeric(df)

    # 5️⃣ enrich or mark unmatched
    enriched_rows, unmatched_rows = [], []
    n_master, n_override = 0, 0

    for _, row in df.iterrows():
        fn, sn = row.get("first_name", ""), row.get("second_name", "")
        key = canonical(f"{fn} {sn}")
        pid = None
        source = ""

        if key in master_key2pid:
            pid = master_key2pid[key]
            n_master += 1
            source = "master"
        elif key in override_key2pid:
            pid = override_key2pid[key]
            n_override += 1
            source = "override"

        if pid:
            rec = pid2rec.get(pid, {})
            new_row = row.copy()
            new_row["player_id"] = pid
            new_row["name"]      = rec.get("name", f"{fn} {sn}")
            new_row["nation"]    = rec.get("nation", "")
            new_row["born"]      = rec.get("born", "")
            # position may be season-specific; fall back to master record
            season_rec = rec.get("career", {}).get(season_raw.name)
            new_row["position"]  = season_rec["position"] if season_rec else rec.get("position", "")
            enriched_rows.append(new_row)
        else:
            unmatched_rows.append(row)

    # 6️⃣ write outputs
    season_dir = season_out / "season"
    season_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(enriched_rows).to_csv(season_dir / "cleaned_players.csv", index=False)

    if unmatched_rows:
        unwanted_dir = season_out / "_unwanted"
        unwanted_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unmatched_rows).to_csv(
            unwanted_dir / "cleaned_players_unmatched.csv", index=False
        )

    logging.info(
        "%s • matched(master=%d, override=%d) • unmatched=%d • final rows=%d",
        season_raw.name, n_master, n_override, len(unmatched_rows), len(enriched_rows)
    )


# ─────────────────────────────── main ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root",  type=Path, required=True, help="data/raw/fpl")
    ap.add_argument("--proc-root", type=Path, required=True, help="data/processed/fpl")
    ap.add_argument("--master",    type=Path, required=True, help="master_fpl_players.json")
    ap.add_argument("--overrides", type=Path, help="optional overrides.json")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s")

    pid2rec, master_key2pid = load_master(args.master)
    override_key2pid        = load_overrides(args.overrides)

    for season_dir in sorted(args.raw_root.iterdir()):
        if season_dir.is_dir():
            logging.info("Season %s …", season_dir.name)
            handle_season(
                season_dir,
                args.proc_root / season_dir.name,
                pid2rec,
                master_key2pid,
                override_key2pid
            )


if __name__ == "__main__":
    main()
