#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import secrets
import shutil
import unicodedata
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd

LOG = logging.getLogger("world_cup_cleaner")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def norm_text(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(v))).strip()


def norm_key(v: Any) -> str:
    return norm_text(v).lower()


def slugify(v: str) -> str:
    x = unicodedata.normalize("NFKD", norm_text(v)).encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"[^A-Za-z0-9]+", "-", x).strip("-").lower()
    x = re.sub(r"-{2,}", "-", x)
    return x or "tournament"


def mk_key(d: Any, h: Any, a: Any) -> str:
    d_s, h_s, a_s = norm_text(d), norm_key(h), norm_key(a)
    if not d_s or not h_s or not a_s:
        return ""
    return hashlib.blake2b(f"{d_s}|{h_s}|{a_s}".encode(), digest_size=8).hexdigest()


def norm_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "O":
            out[c] = out[c].map(norm_text)
    return out


def norm_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        d = pd.to_datetime(df[col], errors="coerce")
        df[col] = d.dt.strftime("%Y-%m-%d").fillna("")
    return df


class Registry:
    def __init__(self, path: Path, prefix: str, hex_len: int):
        self.path = path
        self.prefix = prefix
        self.hex_len = hex_len
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.m: dict[str, str] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.m = {norm_key(k): str(v) for k, v in raw.items() if norm_key(k)}

    def get(self, v: Any) -> str:
        k = norm_key(v)
        if not k:
            return ""
        if k not in self.m:
            nb = (self.hex_len + 1) // 2
            seen = set(self.m.values())
            while True:
                cand = f"{self.prefix}{secrets.token_hex(nb)[:self.hex_len]}"
                if cand not in seen:
                    self.m[k] = cand
                    break
        return self.m[k]

    def save(self) -> None:
        self.path.write_text(json.dumps({k: self.m[k] for k in sorted(self.m)}, indent=2, ensure_ascii=False), encoding="utf-8")

    def errors(self) -> list[str]:
        e: list[str] = []
        vals = list(self.m.values())
        if len(vals) != len(set(vals)):
            e.append(f"duplicate ids in {self.path}")
        if any(not v.startswith(self.prefix) for v in vals):
            e.append(f"wrong prefix in {self.path}")
        return e


def load_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {norm_key(k): norm_text(v) for k, v in raw.items() if norm_key(k)}


def former_idx(df: pd.DataFrame) -> dict[str, list[tuple[str, Any, Any]]]:
    idx: dict[str, list[tuple[str, Any, Any]]] = {}
    for _, r in df.iterrows():
        f, c = norm_text(r.get("former")), norm_text(r.get("current"))
        if not f or not c:
            continue
        s = pd.to_datetime(r.get("start_date"), errors="coerce")
        e = pd.to_datetime(r.get("end_date"), errors="coerce")
        idx.setdefault(norm_key(f), []).append((c, s, e))
    return idx


def canon_team(name: Any, d: Any, aliases: dict[str, str], idx: dict[str, list[tuple[str, Any, Any]]]) -> tuple[str, bool]:
    n = norm_text(name)
    if not n:
        return "", False
    out = aliases.get(norm_key(n), n)
    changed = norm_key(out) != norm_key(n)
    dt = pd.to_datetime(d, errors="coerce")
    if pd.isna(dt):
        return out, changed
    for k in (norm_key(out), norm_key(n)):
        for cur, s, e in idx.get(k, []):
            if (pd.isna(s) or dt >= s) and (pd.isna(e) or dt <= e):
                return cur, True
    return out, changed


def apply_canon(df: pd.DataFrame, date_col: str | None, cols: list[str], aliases: dict[str, str], idx: dict[str, list[tuple[str, Any, Any]]]) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce") if date_col and date_col in out.columns else pd.Series([pd.NaT] * len(out))
    for c in cols:
        if c not in out.columns:
            continue
        vals, rs = [], []
        for v, d in zip(out[c], dates):
            cv, r = canon_team(v, d, aliases, idx)
            vals.append(cv or norm_text(v))
            rs.append(r)
        out[f"{c}_canonical"] = vals
        out[f"{c}_canonical_resolved"] = rs
    return out


_GOAL_ENTRY_RE = re.compile(r"\s*([^,()]+?)\s*\(([^)]*)\)\s*(?:,|$)")


def parse_goal_events(v: Any) -> list[dict[str, Any]]:
    t = norm_text(v)
    if not t:
        return []
    events: list[dict[str, Any]] = []
    matches = list(_GOAL_ENTRY_RE.finditer(t))
    if not matches:
        raw_name = t.strip().strip(".;:")
        if raw_name:
            events.append(
                {
                    "scorer": raw_name,
                    "minute_label": "",
                    "minute": pd.NA,
                    "added_time": pd.NA,
                    "is_penalty": False,
                    "raw_token": "",
                }
            )
        return events

    for m in matches:
        scorer = norm_text(m.group(1)).strip(".;:")
        minute_blob = norm_text(m.group(2))
        tokens = [norm_text(x) for x in minute_blob.split(",") if norm_text(x)]
        if not tokens:
            tokens = [""]
        for token in tokens:
            tl = token.lower()
            is_penalty = "pen" in tl
            minute_label = token.replace("'", "").replace("’", "").strip()
            mm = re.search(r"(\d+)(?:\+(\d+))?", minute_label)
            minute = pd.NA
            added = pd.NA
            if mm:
                minute = int(mm.group(1))
                if mm.group(2):
                    added = int(mm.group(2))
            events.append(
                {
                    "scorer": scorer,
                    "minute_label": minute_label,
                    "minute": minute,
                    "added_time": added,
                    "is_penalty": bool(is_penalty),
                    "raw_token": token,
                }
            )
    return events


def parse_goal_names(v: Any) -> list[str]:
    return [e["scorer"] for e in parse_goal_events(v)]


def parse_referee(v: Any) -> tuple[str, str]:
    t = norm_text(v)
    if not t:
        return "", ""
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", t)
    if not m:
        return t, ""
    return norm_text(m.group(1)), norm_text(m.group(2))


def count_goal_events(v: Any) -> int:
    return len(parse_goal_events(v))


def penalty_label(v: Any) -> str:
    t = norm_key(v)
    if not t:
        return ""
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    return ""


def xlsx_profile(path: Path) -> dict[str, Any]:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "p": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    out: dict[str, Any] = {"file": path.name, "warning": "", "sheets": []}
    try:
        with zipfile.ZipFile(path) as z:
            wb = ET.fromstring(z.read("xl/workbook.xml"))
            rel = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
            relm = {r.attrib["Id"]: r.attrib["Target"] for r in rel.findall("p:Relationship", ns)}
            names = set(z.namelist())
            for s in wb.findall("a:sheets/a:sheet", ns):
                rid = s.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
                t = relm.get(rid, "").lstrip("/")
                p = t if t.startswith("xl/") else ("xl/" + t.replace("../", ""))
                if p not in names:
                    continue
                root = ET.fromstring(z.read(p))
                rows = root.findall(".//a:sheetData/a:row", ns)
                out["sheets"].append({"name": s.attrib.get("name", ""), "rows": len(rows), "xml_path": p})
    except Exception as exc:
        out["warning"] = str(exc)
    return out


def write_csv(df: pd.DataFrame, path: Path, dry: bool) -> None:
    if dry:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_json(obj: Any, path: Path, dry: bool) -> None:
    if dry:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def add_meta(df: pd.DataFrame, src: str, ts: str) -> pd.DataFrame:
    out = df.copy()
    out["source_file"] = src
    out["cleaned_at_utc"] = ts
    return out


def add_country_ids(df: pd.DataFrame, reg: Registry, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_id"] = out[c].map(reg.get)
    return out


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slug_map(results: pd.DataFrame, treg: Registry) -> pd.DataFrame:
    used: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for t in sorted(norm_text(x) for x in results["tournament"].dropna().unique().tolist()):
        base = slugify(t)
        s = base
        n = 2
        while s in used and used[s] != t:
            s = f"{base}-{n}"
            n += 1
        used[s] = t
        rows.append({"tournament": t, "tournament_id": treg.get(t), "tournament_slug": s})
    return pd.DataFrame(rows)


def reconcile_enh(enh: pd.DataFrame, rwc: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for _, r in enh.iterrows():
        d = r.get("date", "")
        h = norm_text(r.get("corrected_home_team_canonical"))
        a = norm_text(r.get("corrected_away_team_canonical"))
        m = rwc[(rwc["date"] == d) & (rwc["home_team_canonical"] == h) & (rwc["away_team_canonical"] == a)]
        st, ref = "unresolved", None
        if not m.empty:
            ref = m.iloc[0]
            if norm_key(r.get("corrected_home_team")) == norm_key(ref.get("home_team")) and norm_key(r.get("corrected_away_team")) == norm_key(ref.get("away_team")):
                st = "exact"
            else:
                st = "alias_fixed"
        else:
            sw = rwc[(rwc["date"] == d) & (rwc["home_team_canonical"] == a) & (rwc["away_team_canonical"] == h)]
            if not sw.empty:
                ref = sw.iloc[0]
                st = "swapped_home_away"
            else:
                rd = pd.to_datetime(d, errors="coerce")
                if not pd.isna(rd):
                    tset = {norm_key(h), norm_key(a)}
                    near = rwc.copy()
                    near["d2"] = pd.to_datetime(near["date"], errors="coerce")
                    near = near[near.apply(lambda x: {norm_key(x["home_team_canonical"]), norm_key(x["away_team_canonical"])} == tset, axis=1)]
                    if not near.empty:
                        near["gap"] = (near["d2"] - rd).abs()
                        near = near[near["gap"] <= timedelta(days=2)]
                        if not near.empty:
                            ref = near.sort_values("gap").iloc[0]
                            st = "date_shifted"
        out.append(
            {
                "match_id": r.get("match_id", ""),
                "results_match_status": st,
                "results_reference_date": "" if ref is None else ref.get("date", ""),
                "results_reference_home_team": "" if ref is None else ref.get("home_team", ""),
                "results_reference_away_team": "" if ref is None else ref.get("away_team", ""),
                "results_reference_match_key": "" if ref is None else ref.get("match_key_canonical", ""),
            }
        )
    return pd.DataFrame(out)


def legacy_player_ids() -> set[str]:
    out: set[str] = set()
    for p in (Path("data/processed/registry/master_players.json"), Path("data/processed/fbref/INT-World Cup/master_players.json")):
        if not p.exists():
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(j, dict):
            for k, v in j.items():
                if isinstance(k, str):
                    out.add(k)
                if isinstance(v, str):
                    out.add(v)
    return out


def build_enhanced_goal_events(enhanced: pd.DataFrame, preg: Registry) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in enhanced.iterrows():
        for side in ("home", "away"):
            detail_col = f"{side}_goals_detail"
            detail = r.get(detail_col, "")
            parsed = parse_goal_events(detail)
            team_raw = r.get("corrected_home_team") if side == "home" else r.get("corrected_away_team")
            team_canon = r.get("corrected_home_team_canonical") if side == "home" else r.get("corrected_away_team_canonical")
            team_id = r.get("corrected_home_team_canonical_id") if side == "home" else r.get("corrected_away_team_canonical_id")
            opp_raw = r.get("corrected_away_team") if side == "home" else r.get("corrected_home_team")
            opp_canon = r.get("corrected_away_team_canonical") if side == "home" else r.get("corrected_home_team_canonical")
            opp_id = r.get("corrected_away_team_canonical_id") if side == "home" else r.get("corrected_home_team_canonical_id")
            goals_for = int(r.get("home_goals", 0) if side == "home" else r.get("away_goals", 0))
            for i, ev in enumerate(parsed, start=1):
                pid = preg.get(ev["scorer"])
                rows.append(
                    {
                        "match_id": r.get("match_id"),
                        "date": r.get("date"),
                        "year": r.get("year"),
                        "stage": r.get("stage"),
                        "side": side,
                        "scoring_team": team_raw,
                        "scoring_team_canonical": team_canon,
                        "scoring_team_id": team_id,
                        "opponent_team": opp_raw,
                        "opponent_team_canonical": opp_canon,
                        "opponent_team_id": opp_id,
                        "scorer": ev["scorer"],
                        "scorer_player_id": pid,
                        "minute_label": ev["minute_label"],
                        "minute": ev["minute"],
                        "added_time": ev["added_time"],
                        "is_penalty": ev["is_penalty"],
                        "goal_index_for_team": i,
                        "goals_for_in_match": goals_for,
                        "detail_goal_count_for_side": len(parsed),
                        "detail_score_mismatch_for_side": len(parsed) != goals_for,
                        "detail_source_column": detail_col,
                        "detail_raw": detail,
                        "match_key_canonical": r.get("match_key_canonical"),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["goal_event_id"] = out.apply(
        lambda x: hashlib.blake2b(
            f"{x['match_id']}|{x['side']}|{x['scorer_player_id']}|{x['minute_label']}|{x['goal_index_for_team']}".encode(),
            digest_size=8,
        ).hexdigest(),
        axis=1,
    )
    return out


def build_enhanced_team_perspective(enhanced: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in enhanced.iterrows():
        for side in ("home", "away"):
            is_home = side == "home"
            team_raw = r.get("corrected_home_team") if is_home else r.get("corrected_away_team")
            team_canon = r.get("corrected_home_team_canonical") if is_home else r.get("corrected_away_team_canonical")
            team_id = r.get("corrected_home_team_canonical_id") if is_home else r.get("corrected_away_team_canonical_id")
            opp_raw = r.get("corrected_away_team") if is_home else r.get("corrected_home_team")
            opp_canon = r.get("corrected_away_team_canonical") if is_home else r.get("corrected_home_team_canonical")
            opp_id = r.get("corrected_away_team_canonical_id") if is_home else r.get("corrected_home_team_canonical_id")

            gf = int(r.get("home_goals", 0) if is_home else r.get("away_goals", 0))
            ga = int(r.get("away_goals", 0) if is_home else r.get("home_goals", 0))
            gd = gf - ga
            if gd > 0:
                result = "W"
                points = 3
            elif gd < 0:
                result = "L"
                points = 0
            else:
                result = "D"
                points = 1

            won_on_penalties = bool(r.get("went_to_penalties")) and r.get("winner_side") == side
            lost_on_penalties = bool(r.get("went_to_penalties")) and r.get("winner_side") in {"home", "away"} and r.get("winner_side") != side

            rows.append(
                {
                    "match_id": r.get("match_id"),
                    "date": r.get("date"),
                    "year": r.get("year"),
                    "stage": r.get("stage"),
                    "team_side": side,
                    "team": team_raw,
                    "team_canonical": team_canon,
                    "team_id": team_id,
                    "opponent": opp_raw,
                    "opponent_canonical": opp_canon,
                    "opponent_id": opp_id,
                    "goals_for": gf,
                    "goals_against": ga,
                    "goal_difference": gd,
                    "result": result,
                    "points": points,
                    "won_on_penalties": won_on_penalties,
                    "lost_on_penalties": lost_on_penalties,
                    "xg_for": r.get("home_xg") if is_home else r.get("away_xg"),
                    "xg_against": r.get("away_xg") if is_home else r.get("home_xg"),
                    "possession": r.get("possession_home") if is_home else r.get("possession_away"),
                    "shots_for": r.get("shots_home") if is_home else r.get("shots_away"),
                    "shots_against": r.get("shots_away") if is_home else r.get("shots_home"),
                    "shots_ontarget_for": r.get("shots_ontarget_home") if is_home else r.get("shots_ontarget_away"),
                    "shots_ontarget_against": r.get("shots_ontarget_away") if is_home else r.get("shots_ontarget_home"),
                    "corners_for": r.get("corners_home") if is_home else r.get("corners_away"),
                    "corners_against": r.get("corners_away") if is_home else r.get("corners_home"),
                    "fouls_for": r.get("fouls_home") if is_home else r.get("fouls_away"),
                    "fouls_against": r.get("fouls_away") if is_home else r.get("fouls_home"),
                    "yellow_cards_for": r.get("yellow_cards_home") if is_home else r.get("yellow_cards_away"),
                    "yellow_cards_against": r.get("yellow_cards_away") if is_home else r.get("yellow_cards_home"),
                    "red_cards_for": r.get("red_cards_home") if is_home else r.get("red_cards_away"),
                    "red_cards_against": r.get("red_cards_away") if is_home else r.get("red_cards_home"),
                    "attendance": r.get("attendance"),
                    "venue": r.get("venue"),
                    "city": r.get("city"),
                    "referee_name": r.get("referee_name"),
                    "referee_country": r.get("referee_country"),
                    "penalty_shootout_normalized": r.get("penalty_shootout_normalized"),
                    "went_to_extra_time": r.get("went_to_extra_time"),
                    "tournament_format": r.get("tournament_format"),
                    "host_nations": r.get("host_nations"),
                    "match_key_canonical": r.get("match_key_canonical"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["team_match_id"] = out.apply(
        lambda x: hashlib.blake2b(f"{x['match_id']}|{x['team_side']}".encode(), digest_size=8).hexdigest(),
        axis=1,
    )
    return out


def run(args: argparse.Namespace) -> int:
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    cleaned = out_dir / "cleaned"
    by_t = out_dir / "by_tournament"
    reg_dir = out_dir / "registry"
    audits = out_dir / "audits"
    ts = now_utc()

    csv_paths = sorted(in_dir.glob("*.csv"))
    xlsx_paths = sorted(in_dir.glob("*.xlsx"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files in {in_dir}")
    df_map = {p.name: pd.read_csv(p) for p in csv_paths}
    hashes = {p.name: hash_file(p) for p in csv_paths + xlsx_paths}

    required = {"results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv", "fifa_world_cup_enhanced_1974_2022.csv"}
    miss = sorted(required - set(df_map))
    if miss:
        raise FileNotFoundError(f"Missing required files: {miss}")

    aliases = load_aliases(Path("config/world_cup_team_aliases.json"))
    preg = Registry(reg_dir / "world_cup_player_ids.json", "wcp_", 12)
    creg = Registry(reg_dir / "world_cup_country_ids.json", "wct_", 12)
    treg = Registry(reg_dir / "world_cup_tournament_ids.json", "wctn_", 10)

    former = norm_df(df_map["former_names.csv"])
    former = norm_date(former, "start_date")
    former = norm_date(former, "end_date")
    fidx = former_idx(former)

    results = norm_df(df_map["results.csv"])
    results = norm_date(results, "date")
    results = apply_canon(results, "date", ["home_team", "away_team"], aliases, fidx)
    results["tournament_id"] = results["tournament"].map(treg.get)
    results["match_key_raw"] = results.apply(lambda r: mk_key(r.get("date"), r.get("home_team"), r.get("away_team")), axis=1)
    results["match_key_canonical"] = results.apply(lambda r: mk_key(r.get("date"), r.get("home_team_canonical"), r.get("away_team_canonical")), axis=1)
    results["is_world_cup_match"] = results["tournament"].eq("FIFA World Cup")
    results = add_country_ids(results, creg, ["home_team_canonical", "away_team_canonical"])
    results = add_meta(results, "results.csv", ts)

    enhanced = norm_df(df_map["fifa_world_cup_enhanced_1974_2022.csv"])
    enhanced = norm_date(enhanced, "date")
    for c in ("home_goals_detail", "away_goals_detail"):
        if c in enhanced.columns:
            for v in enhanced[c].tolist():
                for n in parse_goal_names(v):
                    preg.get(n)
    enhanced["corrected_home_team"] = enhanced["home_team"]
    enhanced["corrected_away_team"] = enhanced["away_team"]
    fix = enhanced["match_id"].eq("WC-1990-025")
    manual_fix = enhanced.loc[fix, ["match_id", "away_team"]].copy()
    if not manual_fix.empty:
        enhanced.loc[fix, "corrected_away_team"] = "England"
        manual_fix["field"] = "corrected_away_team"
        manual_fix["new_value"] = "England"
    e2 = apply_canon(
        enhanced.rename(columns={"corrected_home_team": "h2", "corrected_away_team": "a2"}),
        "date",
        ["h2", "a2", "winner"],
        aliases,
        fidx,
    )
    enhanced["corrected_home_team_canonical"] = e2["h2_canonical"]
    enhanced["corrected_away_team_canonical"] = e2["a2_canonical"]
    enhanced["winner_canonical"] = e2["winner_canonical"] if "winner_canonical" in e2.columns else enhanced["winner"]
    enhanced["match_key_canonical"] = enhanced.apply(
        lambda r: mk_key(r.get("date"), r.get("corrected_home_team_canonical"), r.get("corrected_away_team_canonical")),
        axis=1,
    )
    enhanced = add_country_ids(
        enhanced,
        creg,
        ["corrected_home_team_canonical", "corrected_away_team_canonical", "winner_canonical"],
    )
    # Better enhanced-table cleaning: normalized penalty markers, referee split, and strong integrity flags.
    enhanced["penalty_shootout_normalized"] = enhanced["penalty_shootout"].map(penalty_label)
    enhanced["went_to_penalties"] = (
        enhanced["penalty_shootout_normalized"].eq("yes")
        | (enhanced["penalty_home"].notna() & enhanced["penalty_away"].notna())
    )
    enhanced["went_to_extra_time"] = enhanced["penalty_shootout"].astype(str).str.contains("AET", case=False, na=False)
    enhanced["penalty_home_int"] = pd.to_numeric(enhanced["penalty_home"], errors="coerce").astype("Int64")
    enhanced["penalty_away_int"] = pd.to_numeric(enhanced["penalty_away"], errors="coerce").astype("Int64")

    ref = enhanced["referee"].map(parse_referee)
    enhanced["referee_name"] = ref.map(lambda x: x[0])
    enhanced["referee_country"] = ref.map(lambda x: x[1])

    enhanced["home_goal_detail_count"] = enhanced["home_goals_detail"].map(count_goal_events)
    enhanced["away_goal_detail_count"] = enhanced["away_goals_detail"].map(count_goal_events)
    enhanced["home_goal_detail_matches_score"] = enhanced["home_goal_detail_count"] == enhanced["home_goals"]
    enhanced["away_goal_detail_matches_score"] = enhanced["away_goal_detail_count"] == enhanced["away_goals"]

    enhanced["computed_goal_difference"] = enhanced["home_goals"] - enhanced["away_goals"]
    enhanced["computed_total_goals"] = enhanced["home_goals"] + enhanced["away_goals"]
    enhanced["goal_difference_consistent"] = enhanced["computed_goal_difference"] == enhanced["goal_difference"]
    enhanced["total_goals_consistent"] = enhanced["computed_total_goals"] == enhanced["total_goals"]
    enhanced["possession_sum"] = enhanced["possession_home"] + enhanced["possession_away"]
    enhanced["possession_consistent"] = enhanced["possession_sum"] == 100
    enhanced["shots_ontarget_consistent_home"] = enhanced["shots_ontarget_home"] <= enhanced["shots_home"]
    enhanced["shots_ontarget_consistent_away"] = enhanced["shots_ontarget_away"] <= enhanced["shots_away"]
    enhanced["shots_ontarget_consistent"] = (
        enhanced["shots_ontarget_consistent_home"] & enhanced["shots_ontarget_consistent_away"]
    )

    def _winner_side(row: pd.Series) -> str:
        w = norm_key(row.get("winner_canonical"))
        h = norm_key(row.get("corrected_home_team_canonical"))
        a = norm_key(row.get("corrected_away_team_canonical"))
        if not w or w == "draw":
            return "draw"
        if w == h:
            return "home"
        if w == a:
            return "away"
        return "unknown"

    enhanced["winner_side"] = enhanced.apply(_winner_side, axis=1)
    enhanced["winner_is_draw"] = enhanced["winner_side"].eq("draw")
    enhanced["winner_canonical_id"] = enhanced.apply(
        lambda r: ""
        if r["winner_side"] in {"draw", "unknown"}
        else (
            r["corrected_home_team_canonical_id"]
            if r["winner_side"] == "home"
            else r["corrected_away_team_canonical_id"]
        ),
        axis=1,
    )
    rwc = results[results["tournament"] == "FIFA World Cup"].copy()
    recon = reconcile_enh(enhanced, rwc)
    enhanced = enhanced.merge(recon, on="match_id", how="left")
    enhanced_goal_events = build_enhanced_goal_events(enhanced, preg)
    enhanced_team_perspective = build_enhanced_team_perspective(enhanced)
    enhanced = add_meta(enhanced, "fifa_world_cup_enhanced_1974_2022.csv", ts)
    enhanced_goal_events = add_meta(
        enhanced_goal_events, "fifa_world_cup_enhanced_1974_2022.csv", ts
    )
    enhanced_team_perspective = add_meta(
        enhanced_team_perspective, "fifa_world_cup_enhanced_1974_2022.csv", ts
    )

    goals = norm_df(df_map["goalscorers.csv"])
    goals = norm_date(goals, "date")
    goals = apply_canon(goals, "date", ["home_team", "away_team", "team"], aliases, fidx)
    goals["scorer_player_id"] = goals["scorer"].map(preg.get)
    goals = add_country_ids(goals, creg, ["home_team_canonical", "away_team_canonical", "team_canonical"])
    goals["match_key_canonical"] = goals.apply(lambda r: mk_key(r.get("date"), r.get("home_team_canonical"), r.get("away_team_canonical")), axis=1)
    exact = goals.duplicated(keep="first")
    goals_exact_removed = goals[exact].copy()
    goals = goals[~exact].copy()
    kcols = ["date", "home_team", "away_team", "team", "scorer", "minute"]
    dup = goals.groupby(kcols, dropna=False).size().reset_index(name="n")
    dup = dup[dup["n"] > 1].copy()
    if not dup.empty:
        dup["event_duplicate_group_id"] = range(1, len(dup) + 1)
        goals = goals.merge(dup[kcols + ["event_duplicate_group_id"]], on=kcols, how="left")
    else:
        goals["event_duplicate_group_id"] = pd.NA
    goals["event_key_duplicate_flag"] = goals["event_duplicate_group_id"].notna()
    goals["event_instance_rank"] = goals.groupby(kcols, dropna=False).cumcount() + 1
    goals_amb = goals[goals["event_key_duplicate_flag"]].copy()
    rkeys = set(results["match_key_canonical"].dropna().tolist())
    goals["match_found_in_results_flag"] = goals["match_key_canonical"].isin(rkeys)
    goals = add_meta(goals, "goalscorers.csv", ts)

    shoot = norm_df(df_map["shootouts.csv"])
    shoot = norm_date(shoot, "date")
    shoot = apply_canon(shoot, "date", ["home_team", "away_team", "winner"], aliases, fidx)
    shoot = add_country_ids(shoot, creg, ["home_team_canonical", "away_team_canonical", "winner_canonical"])
    shoot["match_key_canonical"] = shoot.apply(lambda r: mk_key(r.get("date"), r.get("home_team_canonical"), r.get("away_team_canonical")), axis=1)
    shoot["orphan_match_flag"] = ~shoot["match_key_canonical"].isin(rkeys)
    shoot["orphan_reason"] = shoot["orphan_match_flag"].map(lambda x: "missing_in_results" if x else "")
    shoot_orphans = shoot[shoot["orphan_match_flag"]].copy()
    shoot = add_meta(shoot, "shootouts.csv", ts)

    out_map: dict[str, pd.DataFrame] = {
        "former_names.csv": add_meta(former, "former_names.csv", ts),
        "results.csv": results,
        "goalscorers.csv": goals,
        "shootouts.csv": shoot,
        "fifa_world_cup_enhanced_1974_2022.csv": enhanced,
        "fifa_world_cup_enhanced_goal_events_1974_2022.csv": enhanced_goal_events,
        "fifa_world_cup_enhanced_team_perspective_1974_2022.csv": enhanced_team_perspective,
    }
    all_time = recent = champs = hosts = None
    summary_checks: dict[str, Any] = {}
    for name, df in df_map.items():
        if name in out_map:
            continue
        x = norm_df(df)
        for c in x.columns:
            if "date" in c.lower():
                x = norm_date(x, c)
        if "tournament" in x.columns:
            x["tournament_id"] = x["tournament"].map(treg.get)
        out_map[name] = add_meta(x, name, ts)
        if "Results_All_Time" in name:
            all_time = x
        elif "Recent" in name:
            recent = x
        elif "Champions" in name:
            champs = x
        elif "Hosts" in name:
            hosts = x

    if all_time is not None and champs is not None and "Total_Titles" in champs.columns:
        summary_checks["champions_total_titles"] = int(champs["Total_Titles"].sum())
        summary_checks["all_time_rows"] = int(len(all_time))
        summary_checks["titles_match_rows"] = summary_checks["champions_total_titles"] == summary_checks["all_time_rows"]
    if all_time is not None and recent is not None and "Year" in all_time.columns and "Year" in recent.columns:
        summary_checks["recent_subset_of_all_time"] = set(recent["Year"]).issubset(set(all_time["Year"]))

    smap = slug_map(results, treg)

    for name, df in out_map.items():
        write_csv(df, cleaned / name, args.dry_run)
    write_csv(smap, reg_dir / "tournament_slug_map.csv", args.dry_run)

    for _, r in smap.iterrows():
        t, tid, s = r["tournament"], r["tournament_id"], r["tournament_slug"]
        d = by_t / s
        if not args.dry_run:
            d.mkdir(parents=True, exist_ok=True)
        rsub = results[results["tournament"] == t].copy()
        keys = set(rsub["match_key_canonical"].dropna().tolist())
        gsub = goals[goals["match_key_canonical"].isin(keys)].copy()
        ssub = shoot[shoot["match_key_canonical"].isin(keys)].copy()
        write_csv(rsub, d / "results.csv", args.dry_run)
        write_csv(gsub, d / "goalscorers.csv", args.dry_run)
        write_csv(ssub, d / "shootouts.csv", args.dry_run)
        manifest = {
            "tournament": t,
            "tournament_id": tid,
            "tournament_slug": s,
            "build_utc": ts,
            "row_counts": {"results": int(len(rsub)), "goalscorers": int(len(gsub)), "shootouts": int(len(ssub))},
            "source_hashes": hashes,
        }
        write_json(manifest, d / "manifest.json", args.dry_run)

    write_csv(goals_exact_removed, audits / "goalscorers_exact_duplicates_removed.csv", args.dry_run)
    write_csv(goals_amb, audits / "goalscorers_ambiguous_duplicates_retained.csv", args.dry_run)
    write_csv(shoot_orphans, audits / "shootouts_orphans.csv", args.dry_run)
    write_csv(recon, audits / "enhanced_match_reconciliation.csv", args.dry_run)
    write_csv(manual_fix, audits / "manual_corrections_applied.csv", args.dry_run)

    profiles = [xlsx_profile(p) for p in xlsx_paths]
    write_json(profiles, audits / "xlsx_sheet_profile.json", args.dry_run)

    xlsx_warn = ""
    if args.write_xlsx and xlsx_paths and not args.dry_run:
        target = cleaned / "FIFA_World_Cup_Report_cleaned.xlsx"
        # If no excel engine is installed, keep a traceable copy and surface warning.
        try:
            import openpyxl  # noqa: F401
            engine_ok = True
        except Exception:
            engine_ok = False
        if engine_ok and all_time is not None:
            decade = all_time.groupby("Decade", dropna=False)["Year"].count().reset_index(name="Tournaments") if "Decade" in all_time.columns else pd.DataFrame()
            finals_cols = [c for c in ["Year", "Winner", "Runner_Up", "Final_Score", "Went_To_Extra_Time", "Decided_By_Penalties", "Venue"] if c in all_time.columns]
            finals = all_time[finals_cols].copy() if finals_cols else pd.DataFrame()
            summary = pd.DataFrame(
                [
                    {"Metric": "Total tournaments", "Value": int(len(all_time))},
                    {"Metric": "Recent tournaments", "Value": int(len(recent)) if recent is not None else 0},
                    {"Metric": "Champion title sum", "Value": int(champs["Total_Titles"].sum()) if champs is not None and "Total_Titles" in champs.columns else 0},
                ]
            )
            with pd.ExcelWriter(target, engine="openpyxl") as w:
                all_time.to_excel(w, sheet_name="All World Cups", index=False)
                (recent if recent is not None else pd.DataFrame()).to_excel(w, sheet_name="Recent (2000+)", index=False)
                (champs if champs is not None else pd.DataFrame()).to_excel(w, sheet_name="Champions", index=False)
                (hosts if hosts is not None else pd.DataFrame()).to_excel(w, sheet_name="Host Countries", index=False)
                decade.to_excel(w, sheet_name="By Decade", index=False)
                finals.to_excel(w, sheet_name="Finals Analysis", index=False)
                summary.to_excel(w, sheet_name="Summary Report", index=False)
        else:
            xlsx_warn = "openpyxl not installed; copied source report workbook without rebuild."
            shutil.copy2(xlsx_paths[0], target)

    preg_err, creg_err, treg_err = preg.errors(), creg.errors(), treg.errors()
    new_pids = set(preg.m.values())
    legacy = legacy_player_ids()
    inter = sorted(new_pids.intersection(legacy))
    bad_prefix = sum(1 for x in new_pids if not x.startswith("wcp_"))

    if not args.dry_run:
        preg.save()
        creg.save()
        treg.save()

    enhanced_checks = {
        "goal_difference_inconsistent": int((~enhanced["goal_difference_consistent"]).sum()),
        "total_goals_inconsistent": int((~enhanced["total_goals_consistent"]).sum()),
        "possession_inconsistent": int((~enhanced["possession_consistent"]).sum()),
        "shots_ontarget_inconsistent": int((~enhanced["shots_ontarget_consistent"]).sum()),
        "home_goal_detail_score_mismatch": int((~enhanced["home_goal_detail_matches_score"]).sum()),
        "away_goal_detail_score_mismatch": int((~enhanced["away_goal_detail_matches_score"]).sum()),
        "winner_unknown_side": int(enhanced["winner_side"].eq("unknown").sum()),
        "goal_events_rows": int(len(enhanced_goal_events)),
        "team_perspective_rows": int(len(enhanced_team_perspective)),
    }

    qa = {
        "build_utc": ts,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "rows": {
            "results": int(len(results)),
            "goalscorers": int(len(goals)),
            "shootouts": int(len(shoot)),
            "enhanced": int(len(enhanced)),
        },
        "duplicates": {
            "goalscorers_exact_removed": int(len(goals_exact_removed)),
            "goalscorers_ambiguous_retained": int(len(goals_amb)),
        },
        "orphans": {"shootouts": int(len(shoot_orphans))},
        "tournaments": {
            "unique": int(results["tournament"].nunique()),
            "slug_rows": int(len(smap)),
            "split_results_sum": int(sum(len(results[results["tournament"] == t]) for t in smap["tournament"].tolist())),
        },
        "id_checks": {
            "new_player_ids": int(len(new_pids)),
            "bad_prefix_count": int(bad_prefix),
            "legacy_intersection_count": int(len(inter)),
        },
        "enhanced_checks": enhanced_checks,
        "registry_errors": {"player": preg_err, "country": creg_err, "tournament": treg_err},
        "summary_checks": summary_checks,
        "xlsx_warning": xlsx_warn,
        "source_hashes": hashes,
    }
    if inter:
        qa["id_checks"]["legacy_intersection_samples"] = inter[:20]
    write_json(qa, audits / "qa_summary.json", args.dry_run)

    critical = preg_err + creg_err + treg_err
    if bad_prefix:
        critical.append("player ids outside wcp_ prefix")
    if inter:
        critical.append("player ids intersect with legacy ids")
    if int(results["tournament"].nunique()) != int(len(smap)):
        critical.append("tournament split mismatch")
    if args.strict and critical:
        for e in critical:
            LOG.error(e)
        return 2
    for e in critical:
        LOG.warning(e)
    LOG.info("completed world cup cleaning")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="World Cup cleaner with independent IDs and tournament splits.")
    p.add_argument("--input-dir", default="data/raw/fbref/INT-World Cup/world_cup")
    p.add_argument("--output-dir", default="data/processed/fbref/INT-World Cup/world_cup")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--write-xlsx", action="store_true", default=True)
    p.add_argument("--no-write-xlsx", action="store_false", dest="write_xlsx")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    logging.basicConfig(level=getattr(logging, str(a.log_level).upper(), logging.INFO), format="%(levelname)s | %(asctime)s | %(name)s | %(message)s")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
