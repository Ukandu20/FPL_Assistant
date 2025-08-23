#clean_and_enrich.py
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from unidecode import unidecode

# Prefer rapidfuzz; fallback to fuzzywuzzy
try:
    from rapidfuzz import fuzz as rf_fuzz
    _USE_RAPIDFUZZ = True
except Exception:
    try:
        from fuzzywuzzy import fuzz as fw_fuzz  # type: ignore
        _USE_RAPIDFUZZ = False
    except Exception:
        fw_fuzz = None
        _USE_RAPIDFUZZ = False

# ───────────────────────── Canonicalisation & helpers ─────────────────────────

NAME_SUFFIX_RE = re.compile(r"_[0-9]+$")  # foo_123 → foo
FPL_POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
FBREF_TO_FPL_POS = {"GK": "GKP", "DF": "DEF", "MF": "MID", "FW": "FWD"}

def canonical(s: str) -> str:
    """Lowercase, accent-fold, treat separators as spaces, strip punctuation, squeeze."""
    s = s or ""
    s = (
        s.replace("|", " ")
         .replace("_", " ")
         .replace("-", " ")
    )
    s = NAME_SUFFIX_RE.sub("", s)
    s = unidecode(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)     # punctuation → space
    s = " ".join(s.split())            # squeeze
    return s

def season_longform(s: str) -> str:
    """'2019-20' → '2019-2020'; '19-20' → '2019-2020'; pass-through if already long."""
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}", s):
        start = int(s[:4])
        end   = int(str(start)[:2] + s[-2:])
        return f"{start}-{end}"
    if re.fullmatch(r"\d{2}-\d{2}", s):
        start = 2000 + int(s[:2])
        end   = 2000 + int(s[-2:])
        return f"{start}-{end}"
    return s

def season_shortform(long_s: str) -> str:
    """'2019-2020' → '2019-20'."""
    if re.fullmatch(r"\d{4}-\d{4}", long_s):
        start = long_s[:4]
        end   = long_s[-2:]
        return f"{start}-{end}"
    return long_s

def read_json_flex(p: Path) -> dict:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return json.loads(p.read_text(encoding=enc))
        except Exception:
            pass
    return json.loads(p.read_text())

def read_csv_flex(p: Path) -> pd.DataFrame:
    last: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception as e:
            last = e
    raise last or RuntimeError(f"Failed to read {p}")

def write_json_utf8(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

def write_csv_utf8(p: Path, df: pd.DataFrame) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")

# ───────────────────────── Master & overrides ─────────────────────────

def load_fbref_master(master_path: Path) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    Return:
      pid2rec: player_id -> full master record
      key2pid: canonical(full name) -> player_id
    """
    data = read_json_flex(master_path)
    if isinstance(data, list):
        records = data
    else:
        records = []
        for pid, rec in data.items():
            rec = dict(rec)
            rec["player_id"] = rec.get("player_id") or pid
            records.append(rec)

    pid2rec: Dict[str, dict] = {}
    key2pid: Dict[str, str] = {}
    for rec in records:
        pid = rec.get("player_id") or rec.get("id")
        if not pid:
            continue
        pid2rec[pid] = rec
        master_name = rec.get("name") or ""
        if master_name:
            key2pid[canonical(master_name)] = pid
    logging.info("FBref master: %d players, %d canonical keys", len(pid2rec), len(key2pid))
    return pid2rec, key2pid

def load_overrides(path: Optional[Path]) -> Dict[str, str]:
    """
    Overrides map: canonical("first | last") -> player_id
    Keys may include spaces around '|'; we normalise them away.
    """
    if not path or not path.is_file():
        return {}
    raw = read_json_flex(path)
    out: Dict[str, str] = {}
    for k, v in raw.items():
        key = canonical(k.replace(" | ", " ").replace("|", " "))
        if isinstance(v, str):
            out[key] = v
        elif isinstance(v, dict) and v.get("id"):
            out[key] = str(v["id"])
    logging.info("Overrides loaded: %d entries", len(out))
    return out

# ───────────────────────── Matching ─────────────────────────

def _token_variants(key: str) -> List[str]:
    toks = key.split()
    n = len(toks)
    vs: List[str] = []
    if n >= 2:
        if n > 2:
            for i in range(n - 1):
                vs.append(" ".join(toks[i:i+2]))      # sliding bigrams
        vs.append(f"{toks[0]} {toks[-1]}")            # first + last
        vs.append(" ".join(toks[1:]))                 # drop first
        vs.append(" ".join(toks[:-1]))                # drop last
    # "last, first" → "first last" handled at canonical() time if commas removed,
    # but if raw contained a comma, canonical would have removed it. Variants cover enough.
    return list(dict.fromkeys(vs))

def _fuzzy_best(name: str, keys: List[str], threshold: int) -> Optional[str]:
    if not keys:
        return None
    if _USE_RAPIDFUZZ:
        best_key, best_score = None, -1
        for k in keys:
            sc = rf_fuzz.token_set_ratio(name, k)
            if sc > best_score:
                best_key, best_score = k, sc
        return best_key if best_score >= threshold else None
    if fw_fuzz is None:
        return None
    best_key, best_score = None, -1
    for k in keys:
        sc = fw_fuzz.token_set_ratio(name, k)
        if sc > best_score:
            best_key, best_score = k, sc
    return best_key if best_score >= threshold else None

def resolve_player_id(raw_name: str,
                      key2pid: Dict[str, str],
                      overrides: Dict[str, str],
                      threshold: int = 85) -> Optional[str]:
    """Return player_id or None."""
    key = canonical(raw_name)

    # 1) overrides (strongest)
    if key in overrides:
        return overrides[key]

    # 2) exact canonical hit
    if key in key2pid:
        return key2pid[key]

    # 3) token variants
    for v in _token_variants(key):
        if v in overrides:
            return overrides[v]
        if v in key2pid:
            return key2pid[v]

    # 4) fuzzy last resort
    best = _fuzzy_best(key, list(key2pid.keys()), threshold)
    if best:
        return key2pid[best]
    return None

# ───────────────────────── Enrichment ─────────────────────────

def build_display_name(row: pd.Series) -> str:
    # Prefer concatenated first+second; fallback to web_name; else any existing 'name'
    fn = str(row.get("first_name") or "").strip()
    sn = str(row.get("second_name") or "").strip()
    if fn or sn:
        return f"{fn} {sn}".strip()
    wn = str(row.get("web_name") or "").strip()
    if wn:
        return wn
    for c in ["name", "player", "player_name", "Player Name"]:
        if c in row and str(row[c]).strip():
            return str(row[c]).strip()
    return ""

def get_career_season(rec: dict, season_long: str) -> Optional[dict]:
    """
    Try long form first (e.g., '2019-2020'); then short ('2019-20').
    """
    career = rec.get("career") or {}
    if season_long in career:
        return career[season_long]
    short = season_shortform(season_long)
    if short in career:
        return career[short]
    return None

def enrich_season(season_dir: Path,
                  out_root: Path,
                  pid2rec: Dict[str, dict],
                  key2pid: Dict[str, str],
                  overrides: Dict[str, str],
                  threshold: int,
                  fail_if_unmatched_pct: float) -> None:
    season = season_dir.name
    season_full = season_longform(season)

    in_csv  = season_dir / "season" / "cleaned_players.csv"
    if not in_csv.is_file():
        logging.warning("[%s] missing input: %s", season, in_csv)
        return

    df = read_csv_flex(in_csv)
    if df.empty:
        logging.warning("[%s] empty CSV: %s", season, in_csv)
        return

    # Normalise "name"
    if "name" not in df.columns:
        df["name"] = df.apply(build_display_name, axis=1)
    else:
        df["name"] = df["name"].astype(str)

    # Resolve player_id
    df["player_id"] = df["name"].apply(lambda nm: resolve_player_id(nm, key2pid, overrides, threshold))

    # Join FBref truth
    nations, borns, fb_positions, fb_teams, fplpos_from_master, master_names, has_career = [], [], [], [], [], [], []
    for pid in df["player_id"]:
        if not pid or pid not in pid2rec:
            nations.append(None); borns.append(None)
            fb_positions.append(None); fb_teams.append(None)
            fplpos_from_master.append(None); master_names.append(None); has_career.append(False)
            continue

        rec = pid2rec[pid]
        nations.append(rec.get("nation"))
        borns.append(rec.get("born"))
        master_names.append(rec.get("name") or None)

        srec = get_career_season(rec, season_full)
        if srec:
            has_career.append(True)
            fb_teams.append(srec.get("team") or srec.get("short") or srec.get("team_short"))
            fb_positions.append(srec.get("position") or srec.get("pos"))
            fplpos_from_master.append(srec.get("fpl_position") or srec.get("fpl_pos"))
        else:
            has_career.append(False)
            fb_teams.append(None)
            fb_positions.append(None)
            fplpos_from_master.append(None)

    df["nation"]   = nations
    df["born"]     = borns
    df["position"] = fb_positions
    df["team"]     = fb_teams

    # fpl_pos precedence: master > element_type > from FBref position
    if "element_type" in df.columns:
        fpl_pos_from_fpl = df["element_type"].map(FPL_POS_MAP).astype("string")
    else:
        fpl_pos_from_fpl = pd.Series([None] * len(df), dtype="string")
    fbref_to_fpl = df["position"].map(lambda p: FBREF_TO_FPL_POS.get(str(p).upper(), None) if p else None).astype("string")
    df["fpl_pos"] = pd.Series(fplpos_from_master, dtype="string").fillna(fpl_pos_from_fpl).fillna(fbref_to_fpl)

    # Names should be FBref canonical names where matched
    df["name"] = pd.Series(master_names, dtype="string").fillna(df["name"].astype("string"))

    # Review partitions
    unmatched_ids = df[df["player_id"].isna()].copy()
    # "no season entry" = player_id matched but FBref has no career record for this season
    no_season_mask = df["player_id"].notna() & (~pd.Series(has_career, index=df.index))
    no_season_rows = df[no_season_mask].copy()

    # Write enriched file (keep all rows; downstream can filter if needed)
    out_csv = out_root / season / "season" / "cleaned_players.csv"
    write_csv_utf8(out_csv, df)
    logging.info("[%s] wrote enriched players: %s (rows=%d)", season, out_csv, len(df))

    # Write unmatched IDs (CSV)
    if len(unmatched_ids):
        review_dir = out_root / season / "_manual_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        miss_csv = review_dir / f"missing_ids_{season}.csv"
        tmp = unmatched_ids[["name"]].copy()
        tmp["canonical"]   = tmp["name"].map(canonical)
        tmp["suggestions"] = tmp["canonical"].map(lambda c: " | ".join(_token_variants(c)[:4]))
        write_csv_utf8(miss_csv, tmp)
        logging.warning("[%s] unmatched rows=%d → %s", season, len(unmatched_ids), miss_csv)

    # Write missing season career entries (CSV)
    if len(no_season_rows):
        review_dir = out_root / season / "_manual_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        miss_season_csv = review_dir / f"missing_season_{season}.csv"
        cols = ["player_id", "name", "nation", "born"]
        write_csv_utf8(miss_season_csv, no_season_rows[cols])
        logging.warning("[%s] no-career-entry rows=%d → %s", season, len(no_season_rows), miss_season_csv)

    # Guardrail for unmatched percentage
    total = len(df)
    pct_unmatched = 100.0 * len(unmatched_ids) / max(total, 1)
    if pct_unmatched > fail_if_unmatched_pct:
        raise SystemExit(f"[{season}] Unmatched {pct_unmatched:.2f}% > {fail_if_unmatched_pct:.2f}% threshold")

# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich FPL season players with FBref metadata (names are FBref canonical).")
    ap.add_argument("--raw-root", type=Path, required=True, help="Processed FPL root with <season>/season/cleaned_players.csv")
    ap.add_argument("--proc-root", type=Path, required=True, help="Output root (usually same as --raw-root)")
    ap.add_argument("--fbref-master", type=Path, required=True, help="FBref master players JSON (source of truth)")
    ap.add_argument("--overrides", type=Path, default=None, help="Manual overrides JSON (e.g., 'first | last': 'pid')")
    ap.add_argument("--threshold", type=int, default=85, help="Fuzzy minimum (0-100)")
    ap.add_argument("--fail-if-unmatched", type=float, default=10.0, help="Fail run if unmatched percentage exceeds this value")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pid2rec, key2pid = load_fbref_master(args.fbref_master)
    overrides = load_overrides(args.overrides)

    seasons = [d for d in sorted(args.raw_root.iterdir()) if d.is_dir()]
    if not seasons:
        logging.warning("No seasons under %s", args.raw_root)
        return

    for season_dir in seasons:
        logging.info("Season %s …", season_dir.name)
        enrich_season(
            season_dir=season_dir,
            out_root=args.proc_root,
            pid2rec=pid2rec,
            key2pid=key2pid,
            overrides=overrides,
            threshold=args.threshold,
            fail_if_unmatched_pct=args.fail_if_unmatched
        )

if __name__ == "__main__":
    main()
