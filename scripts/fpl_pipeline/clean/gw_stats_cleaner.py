#!/usr/bin/env python3
"""
scripts.fpl_pipeline.clean.gw_stats_cleaner

Cleans FPL game-week CSVs into per-GW cleaned files + merged_gws.csv and
emits manual-review assets including a suggestions JSON you can paste into overrides.

New:
• manual_review/suggestions_<season>.json
  - alias → player_id entries (multiple variants per suggested player)
  - only emitted for unmatched rows whose top fuzzy match >= --suggest-threshold
"""

from __future__ import annotations

import argparse
import codecs
import html
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import pandas as pd
from unidecode import unidecode

# ---------- regex & constants ----------
SEASON_KEY_RE = re.compile(r"^(\d{4})(?:[-–_](\d{2}|\d{4}))?$")
GW_FILE_RE = re.compile(r"^gw(\d{1,2})\.csv$", re.IGNORECASE)

# normalize unicode dashes to ASCII hyphen
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")

def norm_dashes(s: str) -> str:
    return (s or "").translate(_DASHES)

def season_key(s: str) -> int:
    """Return YYYY end-key from '2019-20', '2019-2020', or '2019'."""
    s = norm_dashes(str(s).strip())
    m = SEASON_KEY_RE.fullmatch(s)
    if not m:
        return -1
    start, end = m.groups()
    if end is None:
        return int(start[-4:])
    if len(end) == 2:
        end = start[:2] + end
    return int(end)

def normalize_name(s: str) -> str:
    """HTML-unescape; strip accents; lowercase; collapse; keep '|' as a token barrier."""
    if s is None:
        return ""
    s = html.unescape(str(s))
    s = s.replace("|", " | ")
    s = unidecode(s)
    s = re.sub(r"[^\w\s\|]", " ", s, flags=re.UNICODE).lower()
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s

# ---------- IO utils ----------
def read_json(p: Path) -> dict:
    """
    Robust JSON reader: supports BOM and common encodings, avoids platform defaults.
    """
    data = p.read_bytes()
    # BOM fast-paths
    if data.startswith(codecs.BOM_UTF8):
        return json.loads(data.decode("utf-8-sig"))
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return json.loads(data.decode("utf-16"))
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"):
        try:
            return json.loads(data.decode(enc))
        except Exception:
            continue
    # last resort
    return json.loads(data.decode("utf-8", errors="replace"))

def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

# ---------- team maps (registry) ----------
def normalise_team_maps(team_lookup_json: dict, short_map_json: dict)\
        -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Accepts:
      team_lookup_json: usually CODE->hex id (e.g., {"mun":"767ff900"})
                        but we also tolerate NAME->hex.
      short_map_json:   NAME->CODE (many aliases), case-insensitive.

    Returns:
      name2hex:  normalize(long/alias name) -> hex team_id
      name2code: normalize(long/alias name) -> 3-letter code (UPPER)
      code2hex:  3-letter code (UPPER)      -> hex team_id
    """
    name2hex: Dict[str, str] = {}
    name2code: Dict[str, str] = {}
    code2hex: Dict[str, str] = {}

    # 1) Build code2hex from team_lookup_json if it's code->hex (your file is lower-case codes)
    for k, v in (team_lookup_json or {}).items():
        if isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{8}", v):
            if len(k.strip()) <= 5:  # heuristic for codes
                code2hex[k.strip().upper()] = v.lower()
                continue
        # Otherwise treat as name->hex
        if isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{8}", v):
            name2hex[normalize_name(k)] = v.lower()
        elif isinstance(v, dict):
            hx = v.get("id") or v.get("team_id") or v.get("hex")
            if isinstance(hx, str) and re.fullmatch(r"[0-9a-fA-F]{8}", hx):
                name2hex[normalize_name(k)] = hx.lower()

    # 2) Build name2code from short_map_json (NAME->CODE)
    for nm, val in (short_map_json or {}).items():
        if isinstance(val, str):
            code = val
        elif isinstance(val, dict):
            code = val.get("short") or val.get("code") or val.get("abbr")
        else:
            code = None
        if code:
            name2code[normalize_name(nm)] = code.strip().upper()

    # 3) Derive missing name2hex via name2code + code2hex
    for nm_norm, code in name2code.items():
        if nm_norm not in name2hex:
            hx = code2hex.get(code) or code2hex.get(code.upper()) or code2hex.get(code.lower())
            if hx:
                name2hex[nm_norm] = hx

    return name2hex, name2code, code2hex  # code2hex has UPPER keys

# ---------- build id-based resolvers from teams.csv ----------
def _pick_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None

def build_team_resolvers_from_teams_csv(teams_df: pd.DataFrame,
                                        name2code: Dict[str, str],
                                        name2hex: Dict[str, str],
                                        code2hex: Dict[str, str]) -> Tuple[Dict[int, str], Dict[int, Optional[str]], Dict[int, Optional[str]], Dict[str, str], Dict[str, str]]:
    """
    Return:
      id2name:    numeric fpl id -> long name (best effort)
      id2code3:   numeric fpl id -> 3-letter code (UPPER) or None
      id2hex:     numeric fpl id -> hex team_id or None
      csv_name2code: normalized name from teams.csv -> code (UPPER)
      csv_name2hex:  normalized name from teams.csv -> hex id
    """
    id_col    = _pick_col(teams_df, ["id", "team_id"])
    name_col  = _pick_col(teams_df, ["name", "team_name", "club_name", "team"])
    short_col = _pick_col(teams_df, ["short_name", "short", "abbr", "code3"])  # DO NOT use "code" (numeric)
    if not id_col:
        raise ValueError("teams.csv missing an 'id' column")

    ids    = teams_df[id_col].astype(int).tolist()
    names  = teams_df[name_col].astype(str).tolist() if name_col else [""] * len(ids)
    shorts = teams_df[short_col].astype(str).tolist() if short_col else [""] * len(ids)

    id2name: Dict[int, str] = {}
    id2code3: Dict[int, Optional[str]] = {}
    id2hex: Dict[int, Optional[str]] = {}
    csv_name2code: Dict[str, str] = {}
    csv_name2hex: Dict[str, str] = {}

    for i, nm, sc in zip(ids, names, shorts):
        id2name[i] = nm

        # 1) code: prefer teams.csv short; else map via registry names
        code = (sc or "").strip()
        if not code.isalpha():
            code = ""  # ignore numeric 'code'
        else:
            code = code.upper()
        if not code:
            code = name2code.get(normalize_name(nm), "")
        id2code3[i] = code or None

        # 2) hex: try name->hex; else code->hex
        hx = name2hex.get(normalize_name(nm))
        if not hx and code:
            hx = code2hex.get(code) or code2hex.get(code.upper()) or code2hex.get(code.lower())
        id2hex[i] = hx or None

        # CSV name maps (for string-team fallback)
        nm_norm = normalize_name(nm)
        if code:
            csv_name2code[nm_norm] = code
        if hx:
            csv_name2hex[nm_norm] = hx

    return id2name, id2code3, id2hex, csv_name2code, csv_name2hex

# ---------- master / override keys ----------
def load_master(master_path: Path) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    Returns:
      pid2rec: player_id -> record
      key2pid: normalize(full-name-variants) -> player_id
               (registers master['name'] and (first + second))
    """
    raw = read_json(master_path)
    if isinstance(raw, list):
        records = raw
    else:
        records = []
        for pid, rec in raw.items():
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
        nm = (rec.get("name") or "").strip()
        if nm:
            key2pid.setdefault(normalize_name(nm), pid)
        fn = (rec.get("first_name") or "").strip()
        sn = (rec.get("second_name") or "").strip()
        if fn or sn:
            key2pid.setdefault(normalize_name(f"{fn} | {sn}".strip()), pid)
            key2pid.setdefault(normalize_name(f"{fn} {sn}".strip()), pid)
    logging.info("Master: %d players; keys registered=%d", len(pid2rec), len(key2pid))
    return pid2rec, key2pid

def load_overrides(path: Optional[Path]) -> Dict[str, str]:
    """
    Overrides like:
      "ben | brereton diaz": "00e1d69d"
      "benoit | badiashile": "19c43f77"
    """
    if not path or not path.is_file():
        return {}
    raw = read_json(path)
    out: Dict[str, str] = {}
    for k, v in raw.items():
        key = normalize_name(k.replace(" | ", " ").replace("|", " | "))
        if isinstance(v, str):
            out[key] = v
        elif isinstance(v, dict) and v.get("id"):
            out[key] = str(v["id"])
    logging.info("Overrides: %d entries", len(out))
    return out

# ---------- matching helpers ----------
class OVRule(NamedTuple):
    left: Tuple[str, ...]
    right: Tuple[str, ...]

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

def resolve_player_id(first: str, second: str, display: str,
                      key2pid: Dict[str, str], overrides: Dict[str, str]) -> Optional[str]:
    """Deterministic resolver: overrides → exact master hits on 'first|second', 'first second', 'display'."""
    a = normalize_name(f"{first} | {second}")
    b = normalize_name(f"{first} {second}")
    c = normalize_name(display or "")

    for k in (a, b, c):
        if k in overrides:
            return overrides[k]
    for k in (a, b, c):
        if k in key2pid:
            return key2pid[k]
    return None

def fuzzy_topk(query_key: str, keyspace: List[str], k: int = 1) -> List[Tuple[str, int]]:
    """Return top-k (key, score) by token_set_ratio for a normalized query."""
    if not keyspace:
        return []
    scores: List[Tuple[str, int]] = []
    if _USE_RAPIDFUZZ:
        for key in keyspace:
            scores.append((key, int(rf_fuzz.token_set_ratio(query_key, key))))
    elif fw_fuzz:
        for key in keyspace:
            scores.append((key, int(fw_fuzz.token_set_ratio(query_key, key))))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores[:k]

# ---------- alias generation for suggestions ----------
def alias_variants(first_raw: str, second_raw: str) -> List[str]:
    """
    Build a small, high-yield set of alias strings in 'first | second' form.
    Includes:
      • full first | full second
      • full first | second suffixes (drop leading middle names)
      • (if multi-token first) first prefixes | full second
      • With and without diacritics (lowercased)
    """
    first = (first_raw or "").strip()
    second = (second_raw or "").strip()
    f_toks = [t for t in first.split() if t]
    s_toks = [t for t in second.split() if t]

    pipe_forms: List[str] = []
    if f_toks or s_toks:
        # full | full
        pipe_forms.append(f"{first} | {second}".strip())

        # second suffixes
        for k in range(len(s_toks), 0, -1):
            pipe_forms.append(f"{first} | {' '.join(s_toks[-k:])}".strip())

        # first prefixes
        for k in range(len(f_toks), 0, -1):
            pipe_forms.append(f"{' '.join(f_toks[:k])} | {second}".strip())

    # add ASCII-folded variants where they differ
    out: List[str] = []
    for form in dict.fromkeys(pipe_forms):  # dedupe, preserve order
        ascii_form = unidecode(form)
        base = form.lower()
        out.append(base)
        if ascii_form.lower() != base:
            out.append(ascii_form.lower())
    # final dedupe
    return list(dict.fromkeys(out))

# ---------- pandas helpers ----------
def ensure_numeric(df: pd.DataFrame, col: str, default=None, dtype="Int64") -> pd.Series:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series(np.nan, index=df.index, dtype="float64")
    if default is not None:
        s = s.fillna(default)
    return s.astype(dtype)

def valid_frames(frames: List[Optional[pd.DataFrame]]) -> List[pd.DataFrame]:
    good: List[pd.DataFrame] = []
    for fr in frames:
        if isinstance(fr, pd.DataFrame) and not fr.empty:
            if fr.dropna(axis=1, how="all").shape[1] > 0:
                good.append(fr)
    return good

# ---------- core cleaning ----------
def clean_gw_df(df: pd.DataFrame,
                season: str,
                pid2rec: Dict[str, dict], key2pid: Dict[str, str], overrides: Dict[str, str],
                id2name: Dict[int, str], id2code3: Dict[int, Optional[str]], id2hex: Dict[int, Optional[str]],
                name2code: Dict[str, str], name2hex: Dict[str, str], code2hex: Dict[str, str],
                gw: int) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    """
    Returns: (cleaned_df, unmatched_json_list, unmatched_rows_list)
    Works with 'team' as string (all seasons), and 'opponent_team' as numeric or string.
    """
    # --- build name columns if missing
    if "name" in df.columns:
        df["name"] = df["name"].astype(str)
    else:
        df["name"] = ""

    if "first_name" not in df.columns or "second_name" not in df.columns:
        parts = df["name"].str.extract(r"^(\S+)\s+(.*)$")
        df["first_name"] = parts[0].fillna("")
        df["second_name"] = parts[1].fillna("")

    # --- inputs
    team_str = df["team"].astype(str) if "team" in df.columns else pd.Series([""], index=df.index)
    opp_col = "opponent_team" if "opponent_team" in df.columns else None

    # --- attempt numeric coercion for opponent (many seasons use numeric id)
    df["_opp_id_num"] = ensure_numeric(df, "opponent_team", dtype="Int64") if opp_col else pd.Series([pd.NA]*len(df), dtype="Int64")

    # --- map TEAM by NAME (string across seasons)
    team_norm = team_str.map(normalize_name)
    team_code_from_name = team_norm.map(lambda x: name2code.get(x))
    team_hex_from_name  = team_norm.map(lambda x: name2hex.get(x))

    # fall back: if code known but hex missing, derive via code2hex
    team_hex_from_code = team_code_from_name.map(lambda c: code2hex.get(c) if isinstance(c, str) else None)
    team_hex = team_hex_from_name.fillna(team_hex_from_code)

    # --- map OPPONENT by NUMERIC first, then by NAME, then by CODE-like strings
    def _safe_map(series, mapping):
        return series.map(lambda i: mapping.get(int(i)) if pd.notna(i) and int(i) in mapping else None)

    opp_code_from_num = _safe_map(df["_opp_id_num"], id2code3)
    opp_hex_from_num  = _safe_map(df["_opp_id_num"], id2hex)

    if opp_col:
        opp_as_str = df["opponent_team"].astype(str)
        opp_norm = opp_as_str.map(normalize_name)

        # if opponent_team is a NAME
        opp_code_from_name = opp_norm.map(lambda x: name2code.get(x))
        opp_hex_from_name  = opp_norm.map(lambda x: name2hex.get(x))

        # if opponent_team is a short CODE string (e.g., "MUN")
        def looks_like_code(s: str) -> bool:
            s = (s or "").strip()
            return 2 <= len(s) <= 5 and s.isalpha()

        opp_code_from_code = opp_as_str.map(lambda s: s.upper() if looks_like_code(s) else None)
        opp_hex_from_code  = opp_code_from_code.map(lambda c: code2hex.get(c) if c else None)

        opp_code = opp_code_from_num.fillna(opp_code_from_name).fillna(opp_code_from_code)
        opp_hex  = opp_hex_from_num.fillna(opp_hex_from_name).fillna(opp_hex_from_code)
    else:
        opp_code = pd.Series([None]*len(df))
        opp_hex  = pd.Series([None]*len(df))

    # --- finalize team/opponent fields
    df["team_code"] = team_code_from_name
    df["team_id"]   = team_hex
    df["opp_code"]  = opp_code
    df["opp_id"]    = opp_hex

    # was_home coercion
    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].map(lambda x: str(x).strip().lower() in {"true","1","yes","y","t"})
    else:
        df["was_home"] = False

    # derive home/away and their ids
    df["home"]    = np.where(df["was_home"], df["team_code"], df["opp_code"])
    df["away"]    = np.where(df["was_home"], df["opp_code"], df["team_code"])
    df["home_id"] = np.where(df["was_home"], df["team_id"],  df["opp_id"])
    df["away_id"] = np.where(df["was_home"], df["opp_id"],   df["team_id"])

    # round coercion
    df["round"] = ensure_numeric(df, "round", dtype="Int64")

    # --- player-id matching
    unmatched_json: List[dict] = []
    unmatched_rows: List[dict] = []

    first = df["first_name"].astype(str)
    second = df["second_name"].astype(str)
    display = df["name"].astype(str)

    pids: List[Optional[str]] = []
    fb_names: List[Optional[str]] = []
    positions: List[Optional[str]] = []
    fpl_pos: List[Optional[str]] = []

    for idx in df.index:
        pid = resolve_player_id(first.iat[idx], second.iat[idx], display.iat[idx], key2pid, overrides)
        if pid:
            rec = pid2rec.get(pid, {})
            pids.append(pid)
            fb_names.append(rec.get("name") or f"{first.iat[idx]} {second.iat[idx]}".strip())
            career = rec.get("career") or {}
            if career:
                latest = max(career.keys(), key=season_key)
                srec = career.get(latest) or {}
                positions.append(srec.get("position"))
                fpl_pos.append(srec.get("fpl_position") or srec.get("fpl_pos"))
            else:
                positions.append(None); fpl_pos.append(None)
        else:
            pids.append(None); fb_names.append(None); positions.append(None); fpl_pos.append(None)
            unmatched_json.append({
                "first": first.iat[idx], "second": second.iat[idx], "display": display.iat[idx],
                "pipe_key": normalize_name(f"{first.iat[idx]} | {second.iat[idx]}"),
                "nopipe_key": normalize_name(f"{first.iat[idx]} {second.iat[idx]}"),
                "reason": "no player_id match"
            })
            unmatched_rows.append(df.loc[idx].to_dict())

    df["player_id"] = pids
    df["name"] = pd.Series(fb_names, index=df.index, dtype="string").fillna(df["name"].astype("string"))

    if "position" not in df.columns:
        df["position"] = positions
    if "fpl_pos" not in df.columns:
        df["fpl_pos"] = fpl_pos

    # make exported 'team' the 3-letter code (consistent with your other outputs)
    df["team"] = df["team_code"]

    lead = ["round","first_name","second_name","name","team","position","player_id","fpl_pos",
            "team_code","opp_code","team_id","opp_id","home","away","home_id","away_id","was_home"]
    existing_lead = [c for c in lead if c in df.columns]
    df = df[existing_lead + [c for c in df.columns if c not in existing_lead]]

    return df, unmatched_json, unmatched_rows

# ---------- per-GW wrapper ----------
def process_gw_file(fp: Path, out_dir: Path,
                    season: str,
                    pid2rec: Dict[str, dict], key2pid: Dict[str, str], overrides: Dict[str, str],
                    id2name: Dict[int, str], id2code3: Dict[int, Optional[str]], id2hex: Dict[int, Optional[str]],
                    name2code: Dict[str, str], name2hex: Dict[str, str], code2hex: Dict[str, str],
                    on_unmatched: str) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    gw = int(GW_FILE_RE.match(fp.name).group(1)) if GW_FILE_RE.match(fp.name) else -1
    df = pd.read_csv(fp)
    cleaned, uj, ur = clean_gw_df(
        df, season, pid2rec, key2pid, overrides,
        id2name, id2code3, id2hex,
        name2code, name2hex, code2hex,
        gw
    )

    if on_unmatched == "fail" and (uj or ur):
        raise RuntimeError(f"{fp} (gw={gw}): unmatched rows exist")

    if on_unmatched == "drop":
        before = len(cleaned)
        cleaned = cleaned[cleaned["player_id"].notna()].copy()
        dropped = before - len(cleaned)
        if dropped:
            logging.warning("GW %s: dropped %d unmatched rows", gw, dropped)

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_dir / fp.name.lower(), index=False)
    return cleaned, uj, ur

# ---------- suggestion builder (UNCHANGED PER YOUR REQUEST) ----------
def build_suggestions_from_unmatched(unmatched_rows: List[dict],
                                     key2pid: Dict[str, str],
                                     suggest_threshold: int,
                                     suggest_topk: int) -> Dict[str, str]:
    """
    Produce alias → player_id mapping for overrides.json.
    For each unmatched row:
      1) compute a normalized key from first/second or name
      2) fuzzy-match against master keys; take top-k above threshold
      3) for each suggested pid, emit multiple alias variants ('first | second', suffixes, ascii)
    """
    keyspace = list(key2pid.keys())
    suggestions: Dict[str, str] = {}

    for r in unmatched_rows:
        first = str(r.get("first_name") or "").strip()
        second = str(r.get("second_name") or "").strip()
        name = str(r.get("name") or "").strip()
        # prefer first|second when present; else split name
        if not first and not second and name:
            parts = name.split()
            if parts:
                first, second = parts[0], " ".join(parts[1:])

        q = normalize_name(f"{first} {second}".strip() or name)
        if not q:
            continue

        topk = fuzzy_topk(q, keyspace, k=max(1, suggest_topk))
        topk = [(k, sc) for (k, sc) in topk if sc >= suggest_threshold]
        if not topk:
            continue

        # we only need pid(s); we’ll generate many aliases per pid
        for best_key, _score in topk:
            pid = key2pid.get(best_key)
            if not pid:
                continue
            for alias in alias_variants(first, second):
                # don't overwrite an existing, differently-assigned alias
                if alias in suggestions and suggestions[alias] != pid:
                    continue
                suggestions[alias] = pid
    return suggestions

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root",   required=True, type=Path, help="data/raw/fpl")
    ap.add_argument("--proc-root",  required=True, type=Path, help="data/processed/fpl")
    ap.add_argument("--master",     required=True, type=Path, help="data/processed/registry/master_players.json")
    ap.add_argument("--overrides",  type=Path, help="data/processed/registry/overrides.json")
    ap.add_argument("--team-map",   required=True, type=Path, help="data/processed/registry/_id_lookup_teams.json")  # code->hex (lowercase keys ok)
    ap.add_argument("--short-map",  required=True, type=Path, help="data/config/teams.json")  # name->CODE (aliases)
    ap.add_argument("--season", help="Limit to a single season (accepts '2025-26' or '2025-2026').")
    ap.add_argument("--on-unmatched", choices=["keep","drop","fail"], default="drop",
                    help="What to do with rows without player_id (default: drop)")
    ap.add_argument("--suggest-threshold", type=int, default=90,
                    help="Min fuzzy score (0-100) to emit alias→pid suggestions")
    ap.add_argument("--suggest-topk", type=int, default=1,
                    help="How many top fuzzy candidates to include per unmatched row")
    ap.add_argument("--force", action="store_true", help="Skip seasons with missing gws/ or teams.csv instead of aborting")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pid2rec, key2pid = load_master(args.master)
    overrides = load_overrides(args.overrides)

    team_lookup_json = read_json(args.team_map)   # code->hex (lowercase keys in your file)
    short_map_json   = read_json(args.short_map)  # name->CODE (aliases)
    name2hex_reg, name2code_reg, code2hex = normalise_team_maps(team_lookup_json, short_map_json)

    # seasons to process
    seasons_dirs = [d for d in sorted(args.raw_root.iterdir()) if d.is_dir()]
    if args.season:
        target = norm_dashes(args.season.strip())
        seasons_dirs = [d for d in seasons_dirs if norm_dashes(d.name) in {target}]
    if not seasons_dirs:
        logging.warning("No seasons found under %s", args.raw_root)
        return

    for seas_dir in seasons_dirs:
        gws_dir = seas_dir / "gws"
        if not gws_dir.is_dir():
            msg = f"{seas_dir.name} has no gws/ under {seas_dir}"
            if args.force:
                logging.warning("%s — skipping", msg); continue
            logging.error("%s — aborting this season", msg); continue

        teams_csv = seas_dir / "teams.csv"
        if not teams_csv.is_file():
            msg = f"missing teams.csv in {seas_dir}"
            if args.force:
                logging.warning("%s — skipping", msg); continue
            logging.error("%s — aborting this season", msg); continue

        teams_df = pd.read_csv(teams_csv)  # read all columns; we’ll pick what we need
        id2name, id2code3, id2hex, csv_name2code, csv_name2hex = build_team_resolvers_from_teams_csv(
            teams_df, name2code=name2code_reg, name2hex=name2hex_reg, code2hex=code2hex
        )

        # Merge CSV-derived name maps with registry maps (CSV can override or fill)
        name2code = {**name2code_reg, **csv_name2code}
        name2hex  = {**name2hex_reg,  **csv_name2hex}
        # Ensure every name with a code also has hex
        for nm, code in name2code.items():
            if nm not in name2hex:
                hx = code2hex.get(code)
                if hx:
                    name2hex[nm] = hx

        missing_codes = sum(v is None for v in id2code3.values())
        missing_hex   = sum(v is None for v in id2hex.values())
        logging.info("Team resolver: %d teams, missing code=%d, missing hex=%d",
                     len(id2name), missing_codes, missing_hex)

        logging.info("Season %s …", seas_dir.name)
        out_root = args.proc_root / seas_dir.name
        all_clean, all_uj, all_ur = [], [], []

        for fp in sorted(gws_dir.glob("gw*.csv")):
            if not GW_FILE_RE.match(fp.name):
                continue
            cl, uj, ur = process_gw_file(
                fp, out_root / "gws", seas_dir.name,
                pid2rec, key2pid, overrides,
                id2name, id2code3, id2hex,
                name2code, name2hex, code2hex,
                args.on_unmatched
            )
            all_clean.append(cl); all_uj.extend(uj); all_ur.extend(ur)

        merged_raw = gws_dir / "merged_gws.csv"
        if merged_raw.is_file():
            cl, uj, ur = process_gw_file(
                merged_raw, out_root / "gws", seas_dir.name,
                pid2rec, key2pid, overrides,
                id2name, id2code3, id2hex,
                name2code, name2hex, code2hex,
                args.on_unmatched
            )
            all_clean.append(cl); all_uj.extend(uj); all_ur.extend(ur)

        # write merged_gws.csv (cleaned)
        valids = valid_frames(all_clean)
        (out_root / "gws").mkdir(parents=True, exist_ok=True)
        if valids:
            pd.concat(valids, ignore_index=True).to_csv(out_root / "gws" / "merged_gws.csv", index=False)
        else:
            pd.DataFrame(columns=[
                "round","first_name","second_name","name","team","position","player_id","fpl_pos",
                "team_code","opp_code","team_id","opp_id","home","away","home_id","away_id","was_home"
            ]).to_csv(out_root / "gws" / "merged_gws.csv", index=False)

        # ---- review artifacts
        rev = out_root / "manual_review"
        if all_uj or all_ur:
            rev.mkdir(parents=True, exist_ok=True)
        if all_uj:
            write_json(rev / f"missing_ids_{seas_dir.name}.json", all_uj)
        if all_ur:
            pd.DataFrame(all_ur).to_csv(rev / f"unmatched_rows_{seas_dir.name}.csv", index=False)

        # ---- suggestions JSON (alias → pid) for quick overrides (UNCHANGED)
        if all_ur:
            suggestions = build_suggestions_from_unmatched(
                unmatched_rows=all_ur,
                key2pid=key2pid,
                suggest_threshold=args.suggest_threshold,
                suggest_topk=args.suggest_topk,
            )
            if suggestions:
                write_json(rev / f"suggestions_{seas_dir.name}.json", suggestions)
                logging.warning("%s • suggestions emitted: %d aliases", seas_dir.name, len(suggestions))
            else:
                logging.info("%s • no suggestions (no candidate above threshold %d)", seas_dir.name, args.suggest_threshold)

        if not (all_uj or all_ur):
            logging.info("%s • all rows matched 🎉", seas_dir.name)

if __name__ == "__main__":
    main()
