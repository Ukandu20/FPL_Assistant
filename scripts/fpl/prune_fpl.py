#!/usr/bin/env python3
"""
fpl_clean_pipeline.py – STEP 6  (2025-07-30)

• Adds/normalises columns: first_name, second_name, name, team, position, player_id
• ‘team’ and ‘position’ come from the *most-recent* season in the master career block
• ‘name’ now taken from master_fpl_players.json 🔸
• Unmatched rows →   _manual_review/missing_ids_<season>.json
                     _manual_review/unwanted_<season>.csv
"""

from __future__ import annotations
import argparse, json, logging, re
from pathlib import Path
from typing import Dict, Set, Tuple, List
import pandas as pd

# ───────────────────────── helpers ─────────────────────────
SEASON_KEY_RE = re.compile(r"(\d{4})[-_](\d{2,4})")

def _season_sort_key(key: str) -> int:
    m = SEASON_KEY_RE.fullmatch(key)
    if not m:
        return -1
    start, end = m.groups()
    if len(end) == 2:                      # '20'
        end = start[:2] + end             # '2020'
    return int(end)

def _latest_team_pos(career: Dict[str, Dict[str, str]]
                     ) -> Tuple[str | None, str | None, str | None]:
    if not career:
        return None, None, None
    latest = max(career.keys(), key=_season_sort_key)
    info   = career.get(latest, {})
    return ( info.get("team"), info.get("position"), info.get("fpl_pos"))

# ─────────── master & override look-ups ────────────
def build_master_maps(master: Dict[str, dict]
                      ) -> Tuple[Set[str], Dict[str, str]]:
    keys, key2pid = set(), {}
    for outer_pid, rec in master.items():
        fn, sn = rec.get("first_name"), rec.get("second_name")
        pid    = rec.get("player_id") or outer_pid
        if fn and sn and pid:
            key = f"{fn.lower()} | {sn.lower()}"
            keys.add(key)
            key2pid[key] = pid
    logging.info("Master JSON: %d distinct name pairs", len(keys))
    return keys, key2pid

def build_override_lookup(fp: Path | None) -> Dict[str, str]:
    if not fp or not fp.is_file():
        logging.info("No overrides supplied")
        return {}
    raw = json.loads(fp.read_text("utf-8"))
    out: Dict[str, str] = {}
    for k, v in raw.items():
        pid = v["id"] if isinstance(v, dict) else v
        if not pid:
            continue
        for alias in (k, v.get("name") if isinstance(v, dict) else None):
            if not alias:
                continue
            norm = " ".join(alias.strip().lower().replace("|", " | ").split())
            if "|" in norm:
                out[norm] = pid
    logging.info("Overrides JSON: %d aliases", len(out))
    return out

# ────────────── per-season handler ────────────────
def handle_season(
    season_raw   : Path,
    season_out   : Path,
    master_json  : Dict[str, dict],
    master_keys      : Set[str],
    master_key2pid   : Dict[str, str],
    override_map     : Dict[str, str],
):
    src = season_raw / "cleaned_players.csv"
    if not src.exists():
        logging.warning("%s: cleaned_players.csv missing – skipped", season_raw.name)
        return

    df = pd.read_csv(src, engine="python")

    # 1️⃣ lower-case first/second
    for col in ("first_name", "second_name"):
        if col in df.columns:
            df[col] = df[col].str.lower()

    # 2️⃣ rename now_cost → price
    if "now_cost" in df.columns:
        df = df.rename(columns={"now_cost": "price"})

    # 3️⃣ drop ‘am’
    if "element_type" in df.columns:
        df = df[df["element_type"].str.lower() != "am"].reset_index(drop=True)

    # 4️⃣ ensure mandatory cols
    for col in ("player_id", "name", "team", "position", "fpl_pos"):
        if col not in df.columns:
            df[col] = ""

    fixed = ["first_name", "second_name", "name", "team", "position", "player_id", "fpl_pos"]
    df = df[fixed + [c for c in df.columns if c not in fixed]]

    # 5️⃣ match loop
    unmatched_json: List[dict] = []
    unmatched_rows: List[dict] = []
    matched_master = matched_override = 0

    for idx, row in df.iterrows():
        fn, sn = row["first_name"], row["second_name"]
        if not fn or not sn:
            continue
        key = f"{fn} | {sn}"

        if key in master_keys:
            pid = master_key2pid[key]
            rec = master_json[pid]
            df.at[idx, "player_id"] = pid
            df.at[idx, "name"]      = rec.get("name", "").lower() or df.at[idx, "name"]   # 🔸
            team, pos, fpl = _latest_team_pos(rec.get("career", {}))
            if team: df.at[idx, "team"] = team
            if pos:  df.at[idx, "position"] = pos
            if fpl:  df.at[idx, "fpl_pos"]  = fpl
            matched_master += 1

        elif key in override_map:
            pid = override_map[key]
            rec = master_json.get(pid, {})
            df.at[idx, "player_id"] = pid
            df.at[idx, "name"]      = rec.get("name", "").lower() or df.at[idx, "name"]   # 🔸
            team, pos, fpl = _latest_team_pos(rec.get("career", {}))
            if team: df.at[idx, "team"] = team
            if pos:  df.at[idx, "position"] = pos
            if fpl:  df.at[idx, "fpl_pos"]  = fpl
            matched_override += 1

        else:
            unmatched_json.append(
                {
                    "first_name": fn,
                    "second_name": sn,
                    "team": row.get("team", ""),
                    "position": row.get("position", ""),
                    "reason": "no match",
                }
            )
            unmatched_rows.append(row.to_dict())

    # 6️⃣ outputs
    out_season = season_out / "season"
    out_season.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_season / "cleaned_players.csv", index=False)

    review_dir = season_out / "_manual_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    if unmatched_json:
        (review_dir / f"missing_ids_{season_raw.name}.json").write_text(
            json.dumps(unmatched_json, indent=2, ensure_ascii=False), "utf-8"
        )
    if unmatched_rows:
        pd.DataFrame(unmatched_rows).to_csv(
            review_dir / f"unwanted_{season_raw.name}.csv", index=False
        )

    logging.info(
        "%s • player_id added (master=%d, override=%d) • unmatched=%d • final rows=%d",
        season_raw.name, matched_master, matched_override, len(unmatched_rows), len(df)
    )

# ─────────────────── main ────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root",  type=Path, required=True)
    ap.add_argument("--proc-root", type=Path, required=True)
    ap.add_argument("--master",    type=Path, required=True)
    ap.add_argument("--overrides", type=Path)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s")

    master_json          = json.loads(args.master.read_text("utf-8"))
    master_keys, key2pid = build_master_maps(master_json)
    override_map         = build_override_lookup(args.overrides)

    for season_dir in sorted(args.raw_root.iterdir()):
        if season_dir.is_dir():
            logging.info("Season %s …", season_dir.name)
            handle_season(
                season_dir,
                args.proc_root / season_dir.name,
                master_json,
                master_keys,
                key2pid,
                override_map,
            )

if __name__ == "__main__":
    main()
