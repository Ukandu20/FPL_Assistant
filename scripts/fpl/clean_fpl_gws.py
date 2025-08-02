#!/usr/bin/env python3
"""
fpl_clean_gw_pipeline.py – STEP 6F  (2025-08-02)

• Cleans FPL game-week CSVs into per-GW & merged outputs.
• Robust season-key parser (fixes ValueError on "3-24").
• Complete short-code → hex map (fixes blank team_id / home / away).
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
import argparse, json, logging, re
from pathlib import Path
from typing import Dict, Tuple, List, Set
import pandas as pd, numpy as np

# ───────────────────────── helpers ─────────────────────────
SEASON_KEY_RE = re.compile(r"(\d{4})(?:[-_](\d{2,4}))?$")          # 2019-20, 20192020, 2023
GW_FILE_RE    = re.compile(r"gw(\d+)\.csv$", re.I)

def read_json(p: Path) -> dict:
    try:         return json.loads(p.read_text("utf-8"))
    except FileNotFoundError:
        logging.error("Missing %s", p); raise

def canon(s: str) -> str: return re.sub(r"\s+", " ", s.lower().strip())

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
    name2hex  = {canon(k): v.lower()  for k, v in long2hex.items()}
    name2code = {canon(k): v.upper()  for k, v in long2code.items()}

    # build code→hex even if the canonical names differ
    code2hex: Dict[str,str] = {}
    for nm, code in name2code.items():
        if nm in name2hex:               # ideal path
            code2hex[code] = name2hex[nm]
        else:                            # fallback: search long2hex for a name that maps to same code
            # look for first long name with that code
            for ln, ln_code in name2code.items():
                if ln_code == code and ln in name2hex:
                    code2hex[code] = name2hex[ln]; break
    return name2hex, name2code, code2hex

# ── master / overrides (unchanged API) ───────────────────────────────────
def build_master_maps(master: dict) -> Tuple[Set[str], Dict[str,str]]:
    keys, key2pid = set(), {}
    for pid, rec in master.items():
        fn, sn = rec.get("first_name"), rec.get("second_name")
        if fn and sn:
            k = f"{fn.lower()} | {sn.lower()}"; keys.add(k); key2pid[k] = pid
    return keys, key2pid

def build_override(fp: Path|None) -> Dict[str,str]:
    if not fp or not fp.is_file(): return {}
    raw = read_json(fp)
    out = {}
    for alias, v in raw.items():
        pid = v["id"] if isinstance(v, dict) else v
        if pid:
            norm = " ".join(alias.lower().replace("|"," | ").split())
            if "|" in norm: out[norm] = pid
    return out

def build_fullname_maps(master: dict, overrides: dict)->Tuple[Dict[str,str],Dict[str,str]]:
    full, ov = {}, {}
    for pid, rec in master.items():
        fn, sn = rec.get("first_name"), rec.get("second_name")
        if fn and sn: full.setdefault(canon(f"{fn} {sn}"), pid)
        if rec.get("name"): full.setdefault(canon(rec["name"]), pid)
    for a,p in overrides.items(): ov[canon(a.replace("|"," "))]=p
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
             id2name: Dict[int,str],
             name2code: Dict[str,str],
             name2hex: Dict[str,str],
             code2hex: Dict[str,str],
             gw: int
             )->Tuple[pd.DataFrame,List[dict],List[dict]]:

    df["name"] = df["name"].astype(str).str.lower()

    if {"first_name","second_name"}-set(df.columns):
        parts = df["name"].str.extract(r"^(\S+)\s+(.*)$")
        df["first_name"]  = parts[0].fillna("")
        df["second_name"] = parts[1].fillna("")
    else:
        df["first_name"]  = df["first_name"].str.lower()
        df["second_name"] = df["second_name"].str.lower()

    df["round"] = df.get("round", gw).fillna(gw).astype(int)

    if "now_cost" in df.columns: df = df.rename(columns={"now_cost":"price"})
    if "element_type" in df.columns:
        df = df[df["element_type"].str.lower()!="am"].reset_index(drop=True)

    if "kickoff_time" in df.columns:
        ts = pd.to_datetime(df["kickoff_time"], utc=True)
        df["game_date"] = ts.dt.date.astype(str); df["time"] = ts.dt.time.astype(str)

    df.drop(columns=[c for c in df.columns if c.startswith("mng_")],
            inplace=True, errors="ignore")

    # ── ID → name / code / hex ───────────────────────────────────
    df["team_name"] = df["team"]
    df["opp_name"]  = df["opponent_team"].map(id2name)

    df["team_code"] = df["team_name"].str.lower().map(name2code)
    df["opp_code"]  = df["opp_name"].str.lower().map(name2code)

    df["team_id"] = df["team"].str.lower().map(name2hex)
    df["opp_id"]  = df["opp_name"].str.lower().map(name2hex)

    miss = df["team_id"].isna()
    df.loc[miss, "team_id"] = df.loc[miss, "team_code"].map(code2hex)
    miss = df["opp_id"].isna()
    df.loc[miss, "opp_id"]  = df.loc[miss, "opp_code"].map(code2hex)

    # ── swap by was_home ─────────────────────────────────────────
    df["was_home"] = df["was_home"].astype(str).str.lower() == "true"
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
    canon = df["name"].str.replace(r"\s+"," ",regex=True).str.strip()

    for idx, cname in canon.items():
        rec = pid = None
        if cname in full_map:
            pid = full_map[cname]; rec = master_json[pid]
        elif cname in ov_nopipe:
            pid = ov_nopipe[cname]; rec = master_json.get(pid,{})
        else:
            key = f"{df.at[idx,'first_name']} | {df.at[idx,'second_name']}"
            if key in master_keys:
                pid = key2pid[key]; rec = master_json[pid]
            elif key in ov_map:
                pid = ov_map[key];  rec = master_json.get(pid,{})

        if rec:
            df.at[idx,"player_id"] = pid
            df.at[idx,"name"]      = rec.get("name","").lower() or df.at[idx,"name"]
            t,p,fp = latest_team_pos(rec.get("career",{}))
            if t:  df.at[idx,"team"]     = t
            if p:  df.at[idx,"position"] = p
            if fp: df.at[idx,"fpl_pos"]  = fp
        else:
            unmatched_json.append({"name":df.at[idx,"name"],"reason":"no match"})
            unmatched_rows.append(df.loc[idx].to_dict())

    fixed = ["round","first_name","second_name",
             "name","team","position","player_id","fpl_pos"]
    return df[fixed+[c for c in df.columns if c not in fixed]], unmatched_json, unmatched_rows

# ───────────── per-GW wrapper ──────────────────────────────
def process_gw(fp: Path, out_dir: Path,
               master_json, master_keys, key2pid,
               ov_map, full_map, ov_nopipe,
               id2name, name2code, name2hex, code2hex):

    gw = int(GW_FILE_RE.match(fp.name).group(1)) if GW_FILE_RE.match(fp.name) else -1
    df  = pd.read_csv(fp)
    cl, uj, ur = clean_df(
        df, master_json, master_keys, key2pid,
        ov_map, full_map, ov_nopipe,
        id2name, name2code, name2hex, code2hex, gw)

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

    for seas_dir in sorted(args.raw_root.iterdir()):
        if not seas_dir.is_dir(): continue
        if args.season and seas_dir.name != args.season: continue

        teams_csv = seas_dir/"teams.csv"
        if not teams_csv.is_file():
            msg = f"⚠️  {teams_csv} missing"
            if args.force: logging.warning("%s – skipped", msg); continue
            else:          logging.error("%s – abort season", msg); continue

        teams_df = pd.read_csv(teams_csv, usecols=["id","name"])
        id2name  = dict(zip(teams_df["id"], teams_df["name"].str.strip()))

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
                ov_map, full_map, ov_nopipe,
                id2name, name2code, name2hex, code2hex)
            all_cl.append(cl); all_uj.extend(uj); all_ur.extend(ur)

        merged_raw = gws_dir/"merged_gws.csv"
        if merged_raw.is_file():
            cl, uj, ur = process_gw(
                merged_raw, out_root/"gws",
                master_json, master_keys, key2pid,
                ov_map, full_map, ov_nopipe,
                id2name, name2code, name2hex, code2hex)
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
