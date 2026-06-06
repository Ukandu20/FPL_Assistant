#!/usr/bin/env python3
"""
scripts.fpl_pipeline.utils.file_utils

intended use:


Adds player_id, name, nation, born(year), team & position to every
season-wide  players.csv  in the raw FPL dump, writing results to
<proc-root>/<season>/season/players_enriched.csv.

Misses (unmatched rows or malformed CSV lines) are written to
<proc-root>/<season>/_manual_review/missing_ids_<season>.json
"""

from __future__ import annotations
import argparse, json, logging, html, re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from unidecode import unidecode


# ─────────────────────────── nick-name table ────────────────────────────
NICK = {
    # canonical → list of common diminutives (all lower-case)
    "edward":   ["eddie", "ed"],
    "daniel":   ["dani", "danny", "dan"],
    "nicholas": ["nick", "nico", "niko"],
    "joshua":   ["josh"],
    "matthew":  ["matt"],
    "dominic":  ["dom"],
    "frederico": ["fred", "freddie", "freddy"],
    "maximillian": ["max"],
    "alexander": ["alex", "aleks", "sasha"],
    "sasa":      ["saša", "sasha"],
    "solomon":   ["solly"],
}


# ───────────────────────────── IO helpers ───────────────────────────────
def jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logging.error("Could not read %s", p)
        return {}


def jdump(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────── ultra-light string clean ───────────────────────
ZERO_WIDTH = dict.fromkeys([0x00A0, 0x200B, 0x2060, 0xFEFF], " ")  # NBSP, ZWSP, WJ, BOM
PUNCT_RE   = re.compile(r"[^\w\s]", re.UNICODE)                    # keep letters / digits / space
SUFF_RE    = re.compile(r"_[0-9]+$")                               # drop trailing _123 in FPL dumps


def _clean(s: str | None) -> str:
    """Trim + replace invisible spaces – nothing else."""
    if s is None:
        return ""
    return str(s).translate(ZERO_WIDTH).strip()


def canonical(s: str, *, fold: bool = False) -> str:
    """
    'Arnaut Danjuma-Groeneveld' →
      • canonical (fold=False):  'arnaut danjuma groeneveld'
      • folded    (fold=True) :  same but accents removed
    No nick-name expansion here.
    """
    s = html.unescape(SUFF_RE.sub("", s)).replace("-", " ")
    if fold:
        s = unidecode(s)
    s = PUNCT_RE.sub(" ", s).lower()
    return " ".join(s.split())


def expand_first_variants(token: str) -> List[str]:
    """
    Return canonical + folded + nick-name variants for the FIRST name token.
    """
    base_can = canonical(token)
    base_fold = canonical(token, fold=True)
    out = {base_can, base_fold}
    out.update(NICK.get(base_can, []))
    return list(out)


def all_variants(token: str) -> List[str]:
    """canonical + folded for *any* token (surname tokens)."""
    can = canonical(token)
    fold = canonical(token, fold=True)
    return [can] if can == fold else [can, fold]


# ─────────────────────────── look-up builders ───────────────────────────
def build_lookups(master_fp: Path, override_fp: Path | None):
    """
    Returns three dicts:
        id2rec   : pid -> master-record
        pair2id  : lower-cased  'first|second' -> pid
        override : lower-cased  'first|second' -> pid   (manual fixes)
    """
    master = jload(master_fp)
    if isinstance(master, list):  # allow list-style master JSON
        master = {d["player_id"]: d for d in master if d.get("player_id")}

    pair2id: Dict[str, str] = {}
    for pid, rec in master.items():
        key = f"{_clean(rec.get('first_name')).lower()}|" \
              f"{_clean(rec.get('second_name')).lower()}"
        pair2id[key] = pid

    override = {}
    if override_fp and override_fp.is_file():
        for raw_key, pid in jload(override_fp).items():
            override[raw_key.lower()] = pid

    return master, pair2id, override


# ───────────────────────── row enrichment logic ─────────────────────────
def enrich_row(row: pd.Series,
               season: str,
               id2rec: dict,
               pair2id: dict,
               override: dict,
               review_log: list) -> pd.Series | None:
    """
    Try in order:
       1. player_id already present in CSV
       2. exact first|second (case-insensitive)
       3. manual override
       4. nick-name / multi-token surname heuristic
    """

    # 0) basic clean
    fn_raw = _clean(row.get("first_name"))
    sn_raw = _clean(row.get("second_name"))
    fn_low = fn_raw.lower()
    sn_low = sn_raw.lower()

    # ------------------------------------------------------------------ #
    # 1️⃣  direct player_id
    pid = _clean(row.get("player_id"))
    rec = id2rec.get(pid)

    # 2️⃣  exact first|second
    if rec is None and fn_low and sn_low:
        pid = pair2id.get(f"{fn_low}|{sn_low}")
        rec = id2rec.get(pid)

    # 3️⃣  manual override
    if rec is None and fn_low and sn_low:
        pid = override.get(f"{fn_low}|{sn_low}")
        rec = id2rec.get(pid)

    # 4️⃣  heuristic (nick-names + all surname tokens)
    if rec is None:
        fn_variants = expand_first_variants(fn_raw)
        sn_tokens   = sn_raw.split()
        cand_ids = set()

        # candidate ids whose full name contains ALL surname tokens
        for variant in fn_variants:
            for master_key, master_pid in pair2id.items():
                if variant in master_key.split("|")[0]:
                    # check every surname token (canonical or folded) is present
                    if all(any(v in master_key.split("|")[1] for v in all_variants(tok))
                           for tok in sn_tokens):
                        cand_ids.add(master_pid)

        if len(cand_ids) == 1:
            pid = cand_ids.pop()
            rec = id2rec.get(pid)

    # ------------------------------------------------------------------ #
    if rec is None:
        review_log.append({
            "first_name": fn_raw,
            "second_name": sn_raw,
            "reason": "no player_id"
        })
        return None

    # season filter (EPL only)
    season_rec = rec.get("career", {}).get(season)
    if not season_rec:
        review_log.append({
            "name": rec.get("name"),
            "player_id": pid,
            "reason": "not in EPL for that season"
        })
        return None

    # good – build enriched row
    out = row.copy()
    out["player_id"] = pid
    out["name"]      = rec.get("name")
    out["nation"]    = rec.get("nation")
    born = rec.get("born")
    out["born"]      = int(born) if isinstance(born, (int, float)) else born
    out["team"]      = season_rec["team"]
    out["position"]  = season_rec["position"]
    return out


# ───────────────────────── per-season handler ───────────────────────────
def handle_season(season_raw: Path,
                  season_out: Path,
                  id2rec: dict,
                  pair2id: dict,
                  override: dict):
    players_fp = season_raw / "cleaned_players.csv"
    if not players_fp.exists():
        logging.warning("%s: players.csv missing – skipped", season_raw.name)
        return

    malformed: list[dict] = []

    def _capture_bad(line: list[str]):
        malformed.append({
            "raw": ",".join(line),
            "reason": "malformed CSV line"
        })

    df = pd.read_csv(players_fp,
                     engine="python",
                     on_bad_lines=_capture_bad)

    review: List[dict] = []
    enriched_rows = [
        enrich_row(r, season_raw.name, id2rec, pair2id, override, review)
        for _, r in df.iterrows()
    ]
    df_good = pd.DataFrame([r for r in enriched_rows if r is not None])

    season_dir_out = season_out / "season"
    season_dir_out.mkdir(parents=True, exist_ok=True)
    df_good.to_csv(season_dir_out / "players_enriched.csv", index=False)
    logging.info("%s players_enriched.csv (%d rows)", season_raw.name, len(df_good))

    review.extend(malformed)
    if review:
        jdump(season_out / "_manual_review" / f"missing_ids_{season_raw.name}.json",
              review)
        logging.info("✖︎ %d rows → _manual_review", len(review))


# ─────────────────────────────── main ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root",  type=Path, required=True,
                    help="data/raw/fpl")
    ap.add_argument("--proc-root", type=Path, required=True,
                    help="data/processed/fpl")
    ap.add_argument("--master",    type=Path, required=True,
                    help="master_fpl_players.json (EPL-filtered)")
    ap.add_argument("--overrides", type=Path,
                    help="optional overrides.json  {\"first|second\": \"player_id\"}")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s")

    id2rec, pair2id, override = build_lookups(args.master, args.overrides)

    for season_dir in sorted(args.raw_root.iterdir()):
        if not season_dir.is_dir():
            continue
        logging.info("Season %s …", season_dir.name)
        handle_season(
            season_dir,
            args.proc_root / season_dir.name,
            id2rec, pair2id, override
        )


if __name__ == "__main__":
    main()
