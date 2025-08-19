#!/usr/bin/env python3
"""
fpl_clean_gw_pipeline.py – STEP 6F  (2025-08-02)

• Cleans FPL game-week CSVs into per-GW & merged outputs.
• Robust season-key parser (fixes ValueError on "3-24").
• Complete short-code → hex map (fixes blank team_id / home / away).
• Override matcher: exact (pipe → no‑pipe) first, then barrier‑aware scoring to handle split drift & accents.
• --force flag keeps going on missing artefacts.

Usage (example):
python fpl_clean_gw_pipeline.py ^
  --raw-root   data/raw/fpl ^
  --proc-root  data/processed/fpl ^
  --master     data/processed/fpl/master_fpl_players.json ^
  --overrides  data/processed/fpl/overrides.json ^
  --team-map   data/processed/_id_lookup_teams.json ^
  --short-map  data/config/teams.json ^
  --season     2024-2025 ^
  --log-level  INFO
"""
from __future__ import annotations
import argparse, json, logging, re, unicodedata
from pathlib import Path
from typing import Dict, Tuple, List, Set, Optional, NamedTuple
import pandas as pd, numpy as np
from dataclasses import dataclass

# ───────────────────────── config / toggles ─────────────────────────
# Try exact overrides/master first, then scored overrides (backward compatible)
EXACT_FIRST = True

# ───────────────────────── helpers ─────────────────────────
SEASON_KEY_RE = re.compile(r"(\d{4})(?:[-_](\d{2,4}))?$")          
GW_FILE_RE    = re.compile(r"gw(\d+)\.csv$", re.I)


def read_json(p: Path) -> dict:
    try:         return json.loads(p.read_text("utf-8"))
    except FileNotFoundError:
        logging.error("Missing %s", p); raise

def canon(s: str) -> str:
    # legacy canon (kept for minimal behavior change where still used)
    return re.sub(r"\s+", " ", s.lower().strip())


_WS = re.compile(r"\s+")
_PUNCT_KEEP_PIPE = re.compile(r"[^\w\s\|]+", flags=re.UNICODE)

def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s)

    # 1) normalize ALL unicode spaces to plain space
    s = re.sub(r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000]", " ", s)

    # 2) enforce spaces around the barrier
    s = s.replace("|", " | ")

    # 3) strip accents → ascii
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8", "ignore")

    # 4) lowercase
    s = s.lower()

    # 5) drop punctuation except pipe, then collapse spaces
    s = _PUNCT_KEEP_PIPE.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s



def season_key(s: str) -> int:
    """Return YYYY of season string like '2019-20', '2023-24', '2018'."""
    m = SEASON_KEY_RE.fullmatch(s)
    if not m: return -1
    start, end = m.groups()
    if end is None: return int(start[-4:])
    if len(end) == 2: end = start[:2] + end
    return int(end)

def build_maps(long2hex: dict, long2code: dict
               ) -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    # normalize using the robust normalizer (handles accents & punctuation)
    name2hex  = {normalize_name(k): str(v).lower()  for k, v in long2hex.items()}
    name2code = {normalize_name(k): str(v).upper()  for k, v in long2code.items()}

    # build code→hex even if the canonical names differ
    code2hex: Dict[str,str] = {}
    for nm, code in name2code.items():
        if nm in name2hex:               # ideal path
            code2hex[code] = name2hex[nm]
        else:                            # fallback: search long2hex for a name that maps to same code
            for ln, ln_code in name2code.items():
                if ln_code == code and ln in name2hex:
                    code2hex[code] = name2hex[ln]; break
    return name2hex, name2code, code2hex

# ── override barrier-aware matching ───────────────────────────────────

def _split_tokens_no_barrier(s: str) -> List[str]:
    s = normalize_name(s.replace("|", " "))
    return [t for t in s.split(" ") if t]

def _split_barrier(key: str) -> Tuple[List[str], List[str]]:
    """Split an override key into (first_tokens, second_tokens) using the first '|' as the barrier.
       If no pipe exists, treat everything as right side (surname block) for safer matching."""
    k = normalize_name(key)
    if "|" in k:
        left, right = k.split("|", 1)
        L = [t for t in normalize_name(left).split(" ") if t]
        R = [t for t in normalize_name(right).split(" ") if t]
        return L, R
    return [], [t for t in k.split(" ") if t]

@dataclass(frozen=True)
class OVRule:
    pid: str
    left: Tuple[str, ...]   # tokens intended on first_name side
    right: Tuple[str, ...]  # tokens intended on second_name side
    raw: str                # original key (debugging)

def build_override(fp: Path|None) -> Dict[str,str]:
    """Load overrides. Keep both pipe and no‑pipe aliases (normalized)."""
    if not fp or not fp.is_file(): return {}
    raw = read_json(fp)
    out: Dict[str,str] = {}
    for alias, v in raw.items():
        pid = v["id"] if isinstance(v, dict) else v
        if not pid: continue
        k = normalize_name(alias)
        out[k] = pid
        if "|" in k:
            no_pipe = normalize_name(k.replace("|"," "))
            out.setdefault(no_pipe, pid)
    return out

def build_override_index(ov_map: Dict[str,str]) -> List[OVRule]:
    """Convert the flat override map into barrier‑aware rules for scoring."""
    rules: List[OVRule] = []
    seen = set()
    for alias_norm, pid in ov_map.items():
        L, R = _split_barrier(alias_norm)
        key = (pid, tuple(sorted(set(L))), tuple(sorted(set(R))))
        if key in seen: continue
        seen.add(key)
        rules.append(OVRule(pid=pid, left=key[1], right=key[2], raw=alias_norm))
    return rules

class MatchResult(NamedTuple):
    score: int
    spill: int
    covered: int
    rule_idx: int
    rule: OVRule

def _score_match(rule: OVRule, first_tokens: List[str], second_tokens: List[str]) -> Optional[MatchResult]:
    F = set(first_tokens); S = set(second_tokens); ALL = F | S
    L = set(rule.left);    R = set(rule.right)
    if (L | R) - ALL:
        return None  # missing required token entirely

    l_in_f = len(L & F)
    r_in_s = len(R & S)
    l_sp   = len(L & S)
    r_sp   = len(R & F)

    covered = l_in_f + r_in_s + l_sp + r_sp
    spill   = l_sp + r_sp

    # scoring: reward intended‑side hits more than spillovers, penalize spill
    score = (2 * (l_in_f + r_in_s)) + (1 * (l_sp + r_sp)) - spill

    # small penalty if a side only matched via spillover
    if L and not l_in_f: score -= 1
    if R and not r_in_s: score -= 1

    return MatchResult(score=score, spill=spill, covered=covered, rule_idx=0, rule=rule)

def match_override_pid_scored(overrides: List[OVRule], first_name: str, second_name: str) -> Optional[OVRule]:
    F = _split_tokens_no_barrier(first_name)
    S = _split_tokens_no_barrier(second_name)
    best: Optional[MatchResult] = None
    for i, r in enumerate(overrides):
        mr = _score_match(r, F, S)
        if mr is None: continue
        cand = (mr.score, -mr.spill, mr.covered, len(r.right))
        if best is None:
            best = mr._replace(rule_idx=i)
        else:
            cur  = (best.score, -best.spill, best.covered, len(best.rule.right))
            if cand > cur:
                best = mr._replace(rule_idx=i)
    return None if best is None else best.rule

# ── master / overrides maps (normalized) ───────────────────────────────────
def build_master_maps(master: dict) -> Tuple[Set[str], Dict[str,str]]:
    keys, key2pid = set(), {}
    for pid, rec in master.items():
        fn, sn = rec.get("first_name"), rec.get("second_name")
        if fn and sn:
            k = f"{normalize_name(fn)} | {normalize_name(sn)}"
            keys.add(k); key2pid[k] = pid
    return keys, key2pid

def build_fullname_maps(master: dict, overrides: dict)->Tuple[Dict[str,str],Dict[str,str]]:
    full, ov = {}, {}
    for pid, rec in master.items():
        fn, sn = rec.get("first_name"), rec.get("second_name")
        if fn and sn:
            full.setdefault(normalize_name(f"{fn} | {sn}"), pid)
            full.setdefault(normalize_name(f"{fn} {sn}"), pid)
        if rec.get("name"):
            full.setdefault(normalize_name(rec["name"]), pid)
    # overrides already normalized in build_override()
    for a,p in overrides.items(): ov[normalize_name(a)] = p
    return full, ov

def latest_team_pos(career: dict)->Tuple[str|None,str|None,str|None]:
    if not career: return None,None,None
    latest = max(career.keys(), key=season_key)
    c = career.get(latest, {})
    return c.get("team"), c.get("position"), c.get("fpl_pos")

# ─────────────── cleaning one DF ───────────────────────────
def clean_df(df: pd.DataFrame,
             master_json, master_keys, key2pid,
             ov_map, full_map, ov_nopipe,
             ov_rules: List[OVRule],
             id2name: Dict[int,str],
             name2code: Dict[str,str],
             name2hex: Dict[str,str],
             code2hex: Dict[str,str],
             gw: int
             )->Tuple[pd.DataFrame,List[dict],List[dict]]:

    # Normalize names early (handle both 'name' and split columns)
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).map(normalize_name)
    else:
        df["name"] = ""

    if {"first_name","second_name"}-set(df.columns):
        # derive from 'name' if split cols absent
        parts = df["name"].str.extract(r"^(\S+)\s+(.*)$")
        df["first_name"]  = parts[0].fillna("")
        df["second_name"] = parts[1].fillna("")
    else:
        df["first_name"]  = df["first_name"].astype(str)
        df["second_name"] = df["second_name"].astype(str)

    df["first_name"]  = df["first_name"].map(normalize_name)
    df["second_name"] = df["second_name"].map(normalize_name)

    # Ensure displayable name populated
    df["name"] = np.where(df["name"].eq(""),
                          (df["first_name"] + " " + df["second_name"]).str.strip(),
                          df["name"])

    df["round"] = df.get("round", gw).fillna(gw).astype(int)

    if "now_cost" in df.columns: df = df.rename(columns={"now_cost":"price"})
    if "element_type" in df.columns:
        mask = df["element_type"].astype(str).str.lower() != "am"
        df = df.loc[mask].copy()
        df = df[df["element_type"].astype(str).str.lower()!="am"].reset_index(drop=True)

    if "kickoff_time" in df.columns:
        ts = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
        df["game_date"] = ts.dt.date.astype(str)
        df["time"]      = ts.dt.time.astype(str)

    df.drop(columns=[c for c in df.columns if c.startswith("mng_")],
            inplace=True, errors="ignore")

    # ── ID → name / code / hex ───────────────────────────────────
    df["team_name"] = df["team"]
    df["opp_name"]  = df["opponent_team"].map(id2name)

    df["team_code"] = df["team_name"].astype(str).map(lambda x: name2code.get(normalize_name(x)))
    df["opp_code"]  = df["opp_name"].astype(str).map(lambda x: name2code.get(normalize_name(x)))

    df["team_id"] = df["team"].astype(str).map(lambda x: name2hex.get(normalize_name(x)))
    df["opp_id"]  = df["opp_name"].astype(str).map(lambda x: name2hex.get(normalize_name(x)))

    miss = df["team_id"].isna()
    df.loc[miss, "team_id"] = df.loc[miss, "team_code"].map(code2hex)
    miss = df["opp_id"].isna()
    df.loc[miss, "opp_id"]  = df.loc[miss, "opp_code"].map(code2hex)

    # ── swap by was_home ─────────────────────────────────────────
    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(str).str.lower().isin(["true","1","yes"])
    else:
        df["was_home"] = False

    df["home"]    = np.where(df["was_home"], df["team_code"], df["opp_code"])
    df["away"]    = np.where(df["was_home"], df["opp_code"],  df["team_code"])
    df["home_id"] = np.where(df["was_home"], df["team_id"],   df["opp_id"])
    df["away_id"] = np.where(df["was_home"], df["opp_id"],    df["team_id"])

    df.drop(columns=["team_name","opp_name","team_code","opp_code"],
            inplace=True)

    # ── player-ID matching  ──────────────────────────────────────
    for col in ("player_id","team","position","fpl_pos"):
        if col not in df.columns: df[col] = ""

    unmatched_json, unmatched_rows = [], []

    # Precompute candidate keys
    df["_pipe_key"]   = (df["first_name"] + " | " + df["second_name"]).str.strip()
    df["_nopipe_key"] = (df["first_name"] + " "    + df["second_name"]).str.strip()

    

    for idx in df.index:
        pid: Optional[str] = None
        rec: dict = {}

        pipe_k   = df.at[idx, "_pipe_key"]
        nopipe_k = df.at[idx, "_nopipe_key"]
        name_k   = df.at[idx, "name"]

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            if (pipe_k not in ov_nopipe) and (nopipe_k not in ov_nopipe):
                cand = []
                # cheap near-neighbour check (space-insensitive)
                pipe_ns   = pipe_k.replace(" ", "")
                nopipe_ns = nopipe_k.replace(" ", "")
                for k in ov_nopipe.keys():
                    k_ns = k.replace(" ", "")
                    if k_ns == pipe_ns or k_ns == nopipe_ns:
                        cand.append(k)
                        if len(cand) >= 5:
                            break
                if cand:
                    logging.debug("Near override keys (space-insensitive) for [%s | %s]: %s",
                                pipe_k, nopipe_k, cand)


        if EXACT_FIRST:
            # 1) Exact override (pipe → no‑pipe → display)
            for k in (pipe_k, nopipe_k, name_k):
                if k in ov_nopipe:
                    pid = ov_nopipe[k]; rec = master_json.get(pid, {})
                    if pid: break

            # 2) Exact master (pipe → no‑pipe → display)
            if not pid:
                for k in (pipe_k, nopipe_k, name_k):
                    if k in full_map:
                        pid = full_map[k]; rec = master_json.get(pid, {})
                        if pid: break

            # 3) Scored override (barrier‑aware)
            if not pid:
                rule = match_override_pid_scored(ov_rules, df.at[idx,"first_name"], df.at[idx,"second_name"])
                if rule:
                    pid = rule.pid; rec = master_json.get(pid, {})
        else:
            # Prefer scored override first (more specific), then exact
            rule = match_override_pid_scored(ov_rules, df.at[idx,"first_name"], df.at[idx,"second_name"])
            if rule:
                pid = rule.pid; rec = master_json.get(pid, {})
            if not pid:
                for k in (pipe_k, nopipe_k, name_k):
                    if k in ov_nopipe:
                        pid = ov_nopipe[k]; rec = master_json.get(pid, {}); break
            if not pid:
                for k in (pipe_k, nopipe_k, name_k):
                    if k in full_map:
                        pid = full_map[k]; rec = master_json.get(pid, {}); break

        if rec:
            df.at[idx,"player_id"] = pid
            # prefer master display name if present (normalized)
            disp = rec.get("name")
            if disp:
                df.at[idx,"name"] = normalize_name(disp)
            t,p,fp = latest_team_pos(rec.get("career",{}))
            if t:  df.at[idx,"team"]     = t
            if p:  df.at[idx,"position"] = p
            if fp: df.at[idx,"fpl_pos"]  = fp
        else:
            unmatched_json.append({
                "first": df.at[idx,"first_name"],
                "second": df.at[idx,"second_name"],
                "pipe_key": pipe_k,
                "nopipe_key": nopipe_k,
                "reason": "no match"
            })
            unmatched_rows.append(df.loc[idx].to_dict())

    # finalize column order
    fixed = ["round","first_name","second_name","name","team","position","player_id","fpl_pos"]
    existing_fixed = [c for c in fixed if c in df.columns]
    df = df[existing_fixed + [c for c in df.columns if c not in existing_fixed + ["_pipe_key","_nopipe_key"]]]

    # cleanup temp
    df.drop(columns=["_pipe_key","_nopipe_key"], errors="ignore")

    return df, unmatched_json, unmatched_rows

# ───────────── per-GW wrapper ──────────────────────────────
def process_gw(fp: Path, out_dir: Path,
               master_json, master_keys, key2pid,
               ov_map, full_map, ov_nopipe, ov_rules,
               id2name, name2code, name2hex, code2hex, on_unmatched: str):

    gw = int(GW_FILE_RE.match(fp.name).group(1)) if GW_FILE_RE.match(fp.name) else -1
    df  = pd.read_csv(fp)
    cl, uj, ur = clean_df(
        df, master_json, master_keys, key2pid,
        ov_map, full_map, ov_nopipe, ov_rules,
        id2name, name2code, name2hex, code2hex, gw)

    if on_unmatched == "fail" and (uj or ur):
        raise RuntimeError(f"Unmatched rows in {fp} (gw={gw}): {len(ur)} rows")

    if on_unmatched == "drop":
        before = len(cl)
        cl = cl[cl["player_id"].astype(str) != ""].copy()
        dropped = before - len(cl)
        if dropped:
            logging.warning("GW %s: dropped %d unmatched rows from %s", gw, dropped, fp.name)

    out_dir.mkdir(parents=True, exist_ok=True)
    cl.to_csv(out_dir/f"{fp.name.lower()}", index=False)
    return cl, uj, ur

# ─────────────────── main ────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root",   required=True, type=Path)
    ap.add_argument("--proc-root",  required=True, type=Path)
    ap.add_argument("--master",     required=True, type=Path)
    ap.add_argument("--overrides",  type=Path)
    ap.add_argument("--team-map",   required=True, type=Path)
    ap.add_argument("--short-map",  required=True, type=Path)
    ap.add_argument("--season")
    ap.add_argument(
        "--on-unmatched",
        choices=["keep","drop","fail"],
        default="keep",
        help="What to do with rows that failed player-ID match: keep (default), drop, or fail the run."
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s: %(message)s")

    name2hex, name2code, code2hex = build_maps(
        read_json(args.team_map), read_json(args.short_map)
    )

    master_json = read_json(args.master)
    master_keys, key2pid      = build_master_maps(master_json)
    ov_map                    = build_override(args.overrides)
    full_map, ov_nopipe       = build_fullname_maps(master_json, ov_map)
    ov_rules                  = build_override_index(ov_nopipe)

    for seas_dir in sorted(args.raw_root.iterdir()):
        if not seas_dir.is_dir(): continue
        if args.season and seas_dir.name != args.season: continue

        teams_csv = seas_dir/"teams.csv"
        if not teams_csv.is_file():
            msg = f"⚠️  {teams_csv} missing"
            if args.force: logging.warning("%s – skipped", msg); continue
            else:          logging.error("%s – abort season", msg); continue

        teams_df = pd.read_csv(teams_csv, usecols=["id","name"])
        # normalize names for lookup consistency
        id2name  = {int(i): n for i, n in zip(teams_df["id"], teams_df["name"].astype(str))}

        gws_dir = seas_dir/"gws"
        if not gws_dir.is_dir():
            if args.force:
                logging.warning("%s has no gws/ – skipped", seas_dir.name); continue
            else:
                logging.error("%s has no gws/ – abort season", seas_dir.name); continue

        logging.info("Season %s …", seas_dir.name)
        out_root = args.proc_root/seas_dir.name
        all_cl, all_uj, all_ur = [], [], []

        for fp in sorted(gws_dir.glob("gw*.csv")):
            if not GW_FILE_RE.match(fp.name): continue
            cl, uj, ur = process_gw(
                fp, out_root/"gws",
                master_json, master_keys, key2pid,
                ov_map, full_map, ov_nopipe, ov_rules,
                id2name, name2code, name2hex, code2hex, args.on_unmatched)
            all_cl.append(cl); all_uj.extend(uj); all_ur.extend(ur)

        merged_raw = gws_dir/"merged_gws.csv"
        if merged_raw.is_file():
            cl, uj, ur = process_gw(
                merged_raw, out_root/"gws",
                master_json, master_keys, key2pid,
                ov_map, full_map, ov_nopipe, ov_rules,
                id2name, name2code, name2hex, code2hex, args.on_unmatched)
            all_cl.append(cl); all_uj.extend(uj); all_ur.extend(ur)

        if all_cl:
            pd.concat(all_cl, ignore_index=True).to_csv(
                out_root/"gws"/"merged_gws.csv", index=False)

        if all_uj or all_ur:
            rev = out_root/"manual_review"; rev.mkdir(parents=True, exist_ok=True)
            if all_uj:
                (rev/f"missing_ids_{seas_dir.name}.json").write_text(
                    json.dumps(all_uj, indent=2, ensure_ascii=False), "utf-8")
            if all_ur:
                pd.DataFrame(all_ur).to_csv(
                    rev/f"unmatched_rows_{seas_dir.name}.csv", index=False)
            logging.warning("%s • unmatched rows=%d", seas_dir.name, len(all_ur))
        else:
            logging.info("%s • all rows matched 🎉", seas_dir.name)

if __name__ == "__main__":
    main()
