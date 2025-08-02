#!/usr/bin/env python3
"""
fpl_clean_pipeline.py  –  STEP 6  (override-aware matching, adds player_id)
Revision 2025-07-30:
  • write rows with empty player_id → unwanted_<season>.csv 🔸
  • ensure 'name' column exists & sits after second_name 🔸
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, Set, Tuple, List
import pandas as pd

# ────────────────────────── helpers to build look-ups ────────────────────
def build_master_maps(master_fp: Path) -> Tuple[Set[str], Dict[str, str]]:
    """
    Returns
    -------
    master_keys   : set  of "first | second"
    master_key2pid: dict of the same key → player_id
    """
    raw = json.loads(master_fp.read_text("utf-8"))
    records = raw.values() if isinstance(raw, dict) else raw
    keys, key2pid = set(), {}
    for rec_key, rec in enumerate(records):
        fn, sn = rec.get("first_name"), rec.get("second_name")
        pid    = rec.get("player_id") or (rec_key if isinstance(rec_key, str) else None)
        if fn and sn and pid:
            key = f"{fn.lower()} | {sn.lower()}"
            keys.add(key)
            key2pid[key] = pid
    logging.info("Master JSON: %d distinct name pairs", len(keys))
    return keys, key2pid

def build_override_lookup(fp: Path | None) -> Dict[str, str]:
    """Builds { 'first | second' : player_id } – matching master spacing."""
    if not fp or not fp.is_file():
        return {}
    raw = json.loads(fp.read_text("utf-8"))
    out: Dict[str, str] = {}
    for k, v in raw.items():
        pid = v["id"] if isinstance(v, dict) else v
        if not pid:
            continue
        # normalise any key or alias string we see
        for token in (k, v.get("name") if isinstance(v, dict) else None):
            if not token:
                continue
            norm = " ".join(token.strip().lower().replace("|", " | ").split())
            if "|" in norm:          # accept 2-token & 3-token names alike
                out[norm] = pid
    logging.info("Overrides JSON: %d aliases", len(out))
    return out

# ─────────────────────────── per-season processor ────────────────────────
def handle_season(
    season_raw : Path,
    season_out : Path,
    master_keys      : Set[str],
    master_key2pid   : Dict[str, str],
    override_map     : Dict[str, str],
):
    csv_fp = season_raw / "cleaned_players.csv"
    if not csv_fp.exists():
        logging.warning("%s: cleaned_players.csv missing – skipped", season_raw.name)
        return

    df = pd.read_csv(csv_fp, engine="python")

    # 1️⃣ lower-case the name fields
    for col in ("first_name", "second_name"):
        if col in df.columns:
            df[col] = df[col].str.lower()

    # 2️⃣ rename now_cost → price
    if "now_cost" in df.columns:
        df = df.rename(columns={"now_cost": "price"})

    # 3️⃣ drop rows where element_type == 'am'
    if "element_type" in df.columns:
        df = df[df["element_type"].str.lower() != "am"].reset_index(drop=True)

    # 4️⃣ guarantee player_id column
    if "player_id" not in df.columns:
        df["player_id"] = ""

    # 5️⃣ guarantee name column & set order: first | second | name | player_id
    if "name" not in df.columns:
        df["name"] = df["first_name"].fillna("") + " " + df["second_name"].fillna("")
    col_order = (
        ["first_name", "second_name", "name", "player_id"]
        + [c for c in df.columns if c not in {"first_name","second_name","name","player_id"}]
    )
    df = df[col_order]

    unmatched_json: List[dict] = []
    unmatched_rows : List[dict] = []
    matched_master = matched_override = 0

    # 6️⃣ match each row
    for idx, row in df.iterrows():
        fn, sn = row["first_name"], row["second_name"]
        if not fn or not sn:
            continue
        key = f"{fn} | {sn}"
        if key in master_keys:
            df.at[idx, "player_id"] = master_key2pid[key]
            matched_master += 1
        elif key in override_map:
            df.at[idx, "player_id"] = override_map[key]
            matched_override += 1
        else:
            unmatched_json.append(
                {"first_name": fn, "second_name": sn, "reason": "no match"}
            )
            unmatched_rows.append(row.to_dict())

    # 7️⃣ outputs ----------------------------------------------------------------
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
        )                                            # 🔸 new file

    logging.info(
        "%s • player_id added (master=%d, override=%d) • unmatched=%d • final rows=%d",
        season_raw.name, matched_master, matched_override, len(unmatched_rows), len(df)
    )

# ───────────────────────────── main entry-point ──────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root",  type=Path, required=True, help="data/raw/fpl")
    p.add_argument("--proc-root", type=Path, required=True, help="data/processed/fpl")
    p.add_argument("--master",    type=Path, required=True, help="master_fpl_players.json")
    p.add_argument("--overrides", type=Path, help="overrides.json")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s")

    master_keys, master_key2pid = build_master_maps(args.master)
    override_map = build_override_lookup(args.overrides)

    for season_dir in sorted(args.raw_root.iterdir()):
        if season_dir.is_dir():
            logging.info("Season %s …", season_dir.name)
            handle_season(
                season_dir,
                args.proc_root / season_dir.name,
                master_keys,
                master_key2pid,
                override_map
            )

if __name__ == "__main__":
    main()
