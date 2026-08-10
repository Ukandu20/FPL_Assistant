# clean_and_enrich.py
from __future__ import annotations

import argparse
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from unidecode import unidecode

from fpl_assistant.canonical.identity import stable_canonical_id

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
FPL_POS_ALIASES = {
    "1": "GKP", "GK": "GKP", "GKP": "GKP",
    "2": "DEF", "DF": "DEF", "DEF": "DEF",
    "3": "MID", "MF": "MID", "MID": "MID",
    "4": "FWD", "FW": "FWD", "FWD": "FWD",
}
FPL_TO_FBREF_POS = {"GKP": "GK", "DEF": "DF", "MID": "MF", "FWD": "FW"}

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


def normalise_fpl_position(value) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return FPL_POS_ALIASES.get(str(value).strip().upper())


def load_team_id_lookup(path: Optional[Path]) -> Dict[str, str]:
    if not path or not path.is_file():
        return {}
    raw = read_json_flex(path)
    return {
        str(code).strip().upper(): str(team_id)
        for code, team_id in raw.items()
        if team_id
    }


def attach_fpl_context(df: pd.DataFrame, season_dir: Path) -> pd.DataFrame:
    """Attach official element/team/position fields omitted by cleaned_players.csv."""
    players_path = season_dir / "players_raw.csv"
    teams_path = season_dir / "season" / "teams.csv"
    if not players_path.is_file():
        logging.warning("[%s] missing companion player file: %s", season_dir.name, players_path)
        return df

    players = read_csv_flex(players_path)
    required = {"first_name", "second_name", "id", "team", "element_type"}
    if not required.issubset(players.columns):
        logging.warning(
            "[%s] players_raw.csv lacks required context columns: %s",
            season_dir.name,
            sorted(required - set(players.columns)),
        )
        return df

    team_codes: Dict[int, str] = {}
    if teams_path.is_file():
        teams = read_csv_flex(teams_path)
        if {"id", "short_name"}.issubset(teams.columns):
            team_codes = {
                int(team_id): str(short_name).strip().upper()
                for team_id, short_name in zip(teams["id"], teams["short_name"])
                if pd.notna(team_id) and pd.notna(short_name)
            }

    optional = [
        c
        for c in (
            "web_name",
            "status",
            "can_select",
            "can_transact",
            "news",
            "code",
            "opta_code",
        )
        if c in players.columns
    ]
    context = players[
        ["first_name", "second_name", "id", "team", "element_type", *optional]
    ].copy()
    context["_fpl_join_key"] = context.apply(
        lambda row: canonical(f"{row['first_name']} {row['second_name']}"), axis=1
    )
    duplicate_keys = context["_fpl_join_key"].duplicated(keep=False)
    if duplicate_keys.any():
        logging.warning(
            "[%s] duplicate official player names=%d; keeping first context row",
            season_dir.name,
            int(duplicate_keys.sum()),
        )
        context = context.drop_duplicates("_fpl_join_key", keep="first")

    context = context.rename(
        columns={
            "id": "fpl_element_id",
            "team": "fpl_team_numeric_id",
            "element_type": "fpl_element_type",
            "code": "fpl_code",
            "opta_code": "fpl_opta_code",
        }
    )
    context["fpl_team"] = context["fpl_team_numeric_id"].map(team_codes)

    result = df.copy()
    result["_fpl_join_key"] = result.apply(
        lambda row: canonical(
            f"{row.get('first_name', '')} {row.get('second_name', '')}"
        ),
        axis=1,
    )
    result = result.merge(
        context.drop(columns=["first_name", "second_name"]),
        on="_fpl_join_key",
        how="left",
        validate="many_to_one",
    )
    return result.drop(columns=["_fpl_join_key"])

# ───────────────────────── Season selection helpers ─────────────────────────

def _season_sort_key(p: Path) -> tuple[int, str]:
    """
    Sort directories by starting year when they look like seasons.
    Accepts 'YYYY-YY' or 'YYYY-YYYY'. Unknown formats come first.
    """
    name = p.name
    longf = season_longform(name)
    m = re.match(r"^(\d{4})-(\d{4})$", longf)
    if m:
        return (int(m.group(1)), longf)
    return (-1, name)

def _looks_like_season(name: str) -> bool:
    # allow 2-2, 4-2, 4-4 patterns
    return re.fullmatch(r"\d{2,4}-\d{2,4}", name) is not None

def _match_season_dir(all_dirs: List[Path], sel: str) -> Optional[Path]:
    """
    Try to match 'sel' against directory names using both short/long forms.
    """
    target_long = season_longform(sel)
    target_short = season_shortform(target_long)
    # direct name match first
    for d in all_dirs:
        if d.name == target_long or d.name == target_short:
            return d
    # compare longform of dir names
    for d in all_dirs:
        if season_longform(d.name) == target_long:
            return d
    return None

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
        best_key, best_score = None, -1.0
        for k in keys:
            score = 100.0 * SequenceMatcher(None, name, k).ratio()
            if score > best_score:
                best_key, best_score = k, score
        return best_key if best_score >= threshold else None
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
                  team_ids: Dict[str, str],
                  generate_missing_ids: bool,
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

    df = attach_fpl_context(df, season_dir)

    # Normalise "name"
    if "name" not in df.columns:
        df["name"] = df.apply(build_display_name, axis=1)
    else:
        df["name"] = df["name"].astype(str)

    # Resolve player_id
    df["player_id"] = df["name"].apply(lambda nm: resolve_player_id(nm, key2pid, overrides, threshold))
    df["player_id_source"] = df["player_id"].map(
        lambda value: "master_or_override" if pd.notna(value) else None
    )

    if "web_name" in df.columns:
        for idx in df.index[df["player_id"].isna()]:
            web_name = df.at[idx, "web_name"]
            if pd.isna(web_name) or not str(web_name).strip():
                continue
            web_key = canonical(str(web_name))
            player_id = overrides.get(web_key) or key2pid.get(web_key)
            if player_id:
                df.at[idx, "player_id"] = player_id
                df.at[idx, "player_id_source"] = "master_or_override_web_name"

    generated_mask = pd.Series(False, index=df.index)
    if generate_missing_ids and "fpl_code" in df.columns:
        for idx in df.index[df["player_id"].isna()]:
            provider_code = df.at[idx, "fpl_code"]
            if pd.isna(provider_code) or not str(provider_code).strip():
                continue
            df.at[idx, "player_id"] = stable_canonical_id(
                "player", "fpl", str(provider_code), length=12
            )
            df.at[idx, "player_id_source"] = "generated_from_fpl_code"
            generated_mask.at[idx] = True

    duplicate_ids = df.loc[
        df["player_id"].notna() & df["player_id"].duplicated(keep=False),
        "player_id",
    ].unique()
    for duplicate_id in duplicate_ids:
        duplicate_rows = df.index[df["player_id"] == duplicate_id].tolist()

        def _identity_confidence(idx: int) -> tuple[int, int]:
            full_key = canonical(df.at[idx, "name"])
            original_key = canonical(
                f"{df.at[idx, 'first_name']} {df.at[idx, 'second_name']}"
            )
            web_key = (
                canonical(df.at[idx, "web_name"])
                if "web_name" in df.columns and pd.notna(df.at[idx, "web_name"])
                else ""
            )
            master_key = canonical(pid2rec.get(str(duplicate_id), {}).get("name", ""))
            exact_override = int(
                overrides.get(original_key) == duplicate_id
                or overrides.get(full_key) == duplicate_id
            )
            exact_master = int(bool(master_key) and master_key in {full_key, original_key})
            exact_web = int(bool(master_key) and master_key == web_key)
            return (3 * exact_override + 2 * exact_master + exact_web, -idx)

        keeper = max(duplicate_rows, key=_identity_confidence)
        for idx in duplicate_rows:
            if idx == keeper or not generate_missing_ids:
                continue
            provider_code = df.at[idx, "fpl_code"]
            df.at[idx, "player_id"] = stable_canonical_id(
                "player", "fpl", str(provider_code), length=12
            )
            df.at[idx, "player_id_source"] = "generated_from_fpl_code_duplicate"
            generated_mask.at[idx] = True

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

    df["nation"] = nations
    df["born"] = borns

    official_position_source = (
        df["fpl_element_type"]
        if "fpl_element_type" in df.columns
        else df.get("element_type", pd.Series([None] * len(df), index=df.index))
    )
    official_fpl_pos = official_position_source.map(normalise_fpl_position).astype("string")
    official_team = (
        df["fpl_team"].astype("string")
        if "fpl_team" in df.columns
        else pd.Series([None] * len(df), index=df.index, dtype="string")
    )

    df["team"] = official_team.fillna(pd.Series(fb_teams, dtype="string"))
    df["team_id"] = df["team"].map(
        lambda code: (
            team_ids.get(str(code).strip().upper())
            or stable_canonical_id(
                "team", "fpl", str(code).strip().upper(), length=12
            )
        )
        if pd.notna(code)
        else None
    )
    df["team_id_source"] = df["team"].map(
        lambda code: (
            "registry"
            if pd.notna(code) and str(code).strip().upper() in team_ids
            else "generated_from_fpl_code"
        )
        if pd.notna(code)
        else None
    )

    # Official FPL position is authoritative; master/FBref fill historical gaps.
    fbref_position = pd.Series(fb_positions, dtype="string")
    fbref_to_fpl = fbref_position.map(
        lambda p: FBREF_TO_FPL_POS.get(str(p).upper(), None)
        if pd.notna(p)
        else None
    ).astype("string")
    df["fpl_pos"] = official_fpl_pos.fillna(
        pd.Series(fplpos_from_master, dtype="string")
    ).fillna(fbref_to_fpl)
    df["position"] = fbref_position.fillna(df["fpl_pos"].map(FPL_TO_FBREF_POS))

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

    if generated_mask.any():
        review_dir = out_root / season / "_manual_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        generated_csv = review_dir / f"generated_ids_{season}.csv"
        generated_cols = [
            "player_id",
            "name",
            "fpl_element_id",
            "fpl_code",
            "fpl_opta_code",
            "team",
        ]
        write_csv_utf8(
            generated_csv,
            df.loc[generated_mask, [c for c in generated_cols if c in df.columns]],
        )
        logging.warning(
            "[%s] generated stable IDs=%d -> %s",
            season,
            int(generated_mask.sum()),
            generated_csv,
        )

    # Guardrail for unmatched percentage
    total = len(df)
    pct_unmatched = 100.0 * len(unmatched_ids) / max(total, 1)
    if pct_unmatched > fail_if_unmatched_pct:
        raise SystemExit(f"[{season}] Unmatched {pct_unmatched:.2f}% > {fail_if_unmatched_pct:.2f}% threshold")

# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enrich FPL season players with FBref metadata (names are FBref canonical)."
    )
    ap.add_argument("--raw-root", type=Path, required=True,
                    help="Processed FPL root with <season>/season/cleaned_players.csv")
    ap.add_argument("--proc-root", type=Path, required=True,
                    help="Output root (usually same as --raw-root)")
    ap.add_argument("--fbref-master", type=Path, required=True,
                    help="FBref master players JSON (source of truth)")
    ap.add_argument("--overrides", type=Path, default=None,
                    help="Manual overrides JSON (e.g., 'first | last': 'pid')")
    ap.add_argument("--team-map", type=Path, default=None,
                    help="Optional team-code to canonical team_id JSON")
    ap.add_argument("--generate-missing-ids", action="store_true",
                    help="Generate deterministic canonical IDs from stable FPL codes")
    ap.add_argument("--threshold", type=int, default=85,
                    help="Fuzzy minimum (0-100)")
    ap.add_argument("--fail-if-unmatched", type=float, default=10.0,
                    help="Fail run if unmatched percentage exceeds this value")
    ap.add_argument("--season", type=str, default="all",
                    help="Which season to process: 'all' (default), 'latest', or a specific season like '2025-26'/'2025-2026'")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pid2rec, key2pid = load_fbref_master(args.fbref_master)
    overrides = load_overrides(args.overrides)
    team_ids = load_team_id_lookup(args.team_map)

    # Discover available season directories under --raw-root
    if not args.raw_root.exists():
        raise SystemExit(f"--raw-root does not exist: {args.raw_root}")

    all_dirs = [d for d in args.raw_root.iterdir() if d.is_dir()]
    season_dirs = [d for d in all_dirs if _looks_like_season(d.name)]
    season_dirs = sorted(season_dirs, key=_season_sort_key)

    if not season_dirs:
        logging.warning("No season-like folders under %s", args.raw_root)
        return

    sel = (args.season or "all").strip().lower()
    if sel == "all":
        seasons = season_dirs
    elif sel == "latest":
        seasons = [season_dirs[-1]]
    else:
        match = _match_season_dir(season_dirs, args.season.strip())
        if not match:
            raise SystemExit(
                f"Requested season {args.season!r} not found under {args.raw_root}. "
                f"Available: {[d.name for d in season_dirs]}"
            )
        seasons = [match]

    logging.info("Processing season folder(s): %s", [d.name for d in seasons])

    for season_dir in seasons:
        logging.info("Season %s …", season_dir.name)
        enrich_season(
            season_dir=season_dir,
            out_root=args.proc_root,
            pid2rec=pid2rec,
            key2pid=key2pid,
            overrides=overrides,
            team_ids=team_ids,
            generate_missing_ids=args.generate_missing_ids,
            threshold=args.threshold,
            fail_if_unmatched_pct=args.fail_if_unmatched
        )

if __name__ == "__main__":
    from fpl_assistant.providers.fpl.pipelines.clean_and_enrich import main as package_main
    package_main()
