# -*- coding: utf-8 -*-
"""
Unit tests for optimizer invariants across GW plans.

Run:
  pytest -q scripts/tests/optimizer_invariants.py

Environment overrides (optional):
- LEGAL_FORMATIONS: comma list; default FPL-legal set
- SQUAD_LIMIT_PER_TEAM: default 3
- PLAN_GLOBS: semi-colon separated custom globs to override defaults
"""

import json, os, glob, re
from collections import Counter
from typing import Dict, List, Any, Optional

import pytest

# ----- formations -----
DEFAULT_FORMATIONS = {
    "3-4-3","3-5-2","4-3-3","4-4-2","4-5-1","5-3-2","5-4-1"  # added 4-5-1
}
LEGAL_FORMATIONS = set(
    f.strip() for f in os.getenv("LEGAL_FORMATIONS", ",".join(DEFAULT_FORMATIONS)).split(",") if f.strip()
)
SQUAD_LIMIT_PER_TEAM = int(os.getenv("SQUAD_LIMIT_PER_TEAM", "3"))

# ---------- helpers for robust parsing

def _parse_int(x, default=None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default

def _pid(p: Dict[str, Any]) -> str:
    return str(p.get("id") or p.get("player_id") or p.get("pid") or "")

def _pos(p: Dict[str, Any]) -> str:
    return str(p.get("pos") or p.get("position","")).upper()

def _team(p: Dict[str, Any]) -> str:
    return str(p.get("team") or p.get("team_id") or p.get("team_code") or "")

def _bench_list_single(bench_root: Any) -> List[Dict[str, Any]]:
    """
    Single-GW bench is a dict with:
      {"order": [outfield...], "gk": {..}}  (sometimes "gk" is list)
    """
    if isinstance(bench_root, list):
        return bench_root
    if not isinstance(bench_root, dict):
        return []
    out: List[Dict[str, Any]] = []
    if isinstance(bench_root.get("order"), list):
        out.extend(bench_root["order"])
    gk = bench_root.get("gk")
    if isinstance(gk, dict):
        out.append(gk)
    elif isinstance(gk, list):
        out.extend(gk)
    # accept any additional lists/dicts
    for k, v in bench_root.items():
        if k in ("order","gk"):
            continue
        if isinstance(v, list): out.extend(v)
        elif isinstance(v, dict): out.append(v)
    return out

def _bench_list_multi(bench_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Multi-GW bench in new schema is shaped like:
      {"gw": 5, "bench": {"order":[...], "gk": {...}}}
    but we also accept legacy: {"gw":5, "bench":[...]}
    """
    if not isinstance(bench_entry, dict):
        return []
    v = bench_entry.get("bench")
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return _bench_list_single(v)
    return []

def _formation_from_xi(xi: List[Dict[str, Any]]) -> str:
    from collections import Counter as _C
    cnt = _C(_pos(p) for p in xi)
    return f"{cnt.get('DEF',0)}-{cnt.get('MID',0)}-{cnt.get('FWD',0)}"

def _norm_chip(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).upper()
    return {"NONE":"NONE","TC":"TC","BB":"BB","WC":"WC","FH":"FH"}.get(s, s)

def _looks_like_single(plan: Dict[str, Any]) -> bool:
    return isinstance(plan.get("xi"), list) or isinstance(plan.get("XI"), list) or isinstance(plan.get("starting"), list)

def _looks_like_multi(plan: Dict[str, Any]) -> bool:
    # must have per_gw metadata + xi list keyed by gw
    return isinstance(plan.get("objective", {}).get("per_gw"), list) and isinstance(plan.get("xi"), list)

def _skip_file(path: str) -> bool:
    nm = os.path.basename(path).lower()
    if "summary" in nm or "meta" in nm:
        return True
    return False

def _infer_gw_from_path(path: str) -> Optional[int]:
    m = re.search(r"[\\/]single[\\/](\d+)[\\/]", path)
    if m:
        return _parse_int(m.group(1))
    m2 = re.search(r"[gG][wW]\s*[_-]?(\d+)", path)
    if m2:
        return _parse_int(m2.group(1))
    return None

def _extract_single(plan: Dict[str, Any], path: str) -> List[Dict[str, Any]]:
    """Wrap single-GW root into a per-gw record with normalized keys."""
    gw = _parse_int(plan.get("meta", {}).get("gw"), None)
    if gw is None:
        gw = _infer_gw_from_path(path)
    xi = plan.get("xi") or plan.get("XI") or plan.get("starting") or []
    bench = _bench_list_single(plan.get("bench"))
    # captain sits under top-level "captain" object
    captain_id = None
    if isinstance(plan.get("captain"), dict):
        captain_id = _pid(plan["captain"])
    if captain_id is None:
        captain_id = str(plan.get("captain_id") or plan.get("cap") or "")
    formation = plan.get("meta", {}).get("formation")
    if not isinstance(formation, str):
        formation = _formation_from_xi(xi)
    rec = {
        "gw": gw,
        "xi": xi,
        "bench": bench,
        "captain_id": captain_id,
        "chip": _norm_chip(plan.get("chip")),
        "formation": formation,
        "bank_after": plan.get("meta", {}).get("bank_after"),
    }
    return [rec]

def _extract_multi(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Merge:
      - meta per GW: plan["objective"]["per_gw"][i] with fields {gw, formation, captain, chip, ev_team}
      - xi per GW:   top-level plan["xi"] = [ {"gw":5, "xi":[...]} , ... ]
      - bench per GW: top-level plan["bench"] = [ {"gw":5, "bench":{"order":[...], "gk":{...}}}, ... ]
    """
    per_gw = plan.get("objective", {}).get("per_gw") or []
    xi_index = {}
    if isinstance(plan.get("xi"), list):
        xi_index = { _parse_int(r.get("gw"), None): r.get("xi") or [] for r in plan["xi"] if isinstance(r, dict) }
    bench_index = {}
    if isinstance(plan.get("bench"), list):
        bench_index = { _parse_int(r.get("gw"), None): _bench_list_multi(r) for r in plan["bench"] if isinstance(r, dict) }

    out = []
    for meta in per_gw:
        gw = _parse_int(meta.get("gw"), None)
        if gw is None:
            continue
        xi = xi_index.get(gw, [])
        bench = bench_index.get(gw, [])
        formation = meta.get("formation") or _formation_from_xi(xi)
        # NOTE: read 'captain' (new schema), not 'captain_id'
        captain_id = str(meta.get("captain") or meta.get("captain_id") or "")
        rec = {
            "gw": gw,
            "xi": xi,
            "bench": bench,
            "captain_id": captain_id,
            "chip": _norm_chip(meta.get("chip")),
            "formation": formation,
        }
        out.append(rec)
    return out

def _extract_gws(plan: Dict[str, Any], path: str) -> List[Dict[str, Any]]:
    if _looks_like_multi(plan):
        return _extract_multi(plan)
    if _looks_like_single(plan):
        return _extract_single(plan, path)
    # legacy shape: objective.per_gw already per-GW with inline xi/bench
    if isinstance(plan.get("objective", {}).get("per_gw"), list):
        return plan["objective"]["per_gw"]
    raise KeyError("Unrecognized plan schema")

def _bank_after(obj: Dict[str, Any], plan_root: Dict[str, Any]) -> float:
    v = obj.get("bank_after")
    if v is not None:
        try: return float(v)
        except Exception: return 0.0
    try:
        return float(plan_root.get("meta", {}).get("bank_after", 0.0))
    except Exception:
        return 0.0

# ---------- pytest fixtures

@pytest.fixture(scope="session")
def plan_files():
    # seasonized directory layout
    DEFAULT_GLOBS = [
        # single-GW plans (all chips subfolders)
        "data/plans/*/single/*/*.json",
        "data/plans/*/single/*/chips/*/*.json",
        # multi-GW hold plans (only the concrete plan.json files)
        "data/plans/*/multi/hold/*/*/*/plan.json",
        "data/plans/*/multi/hold/*/chips/*/*/plan.json",
    ]
    custom = os.getenv("PLAN_GLOBS")
    globs_to_use = custom.split(";") if custom else DEFAULT_GLOBS
    files = []
    for patt in globs_to_use:
        files.extend(glob.glob(patt.replace("\\","/")))
    files = sorted(set(files))
    files = [f for f in files if not _skip_file(f)]
    if not files:
        pytest.skip("No plan files matched the expected directories.")
    return files

# ---------- the invariants

@pytest.mark.parametrize("enforce_bench_team_limit", [True, False], ids=["XI+Bench","XI-only"])
def test_invariants(plan_files, enforce_bench_team_limit):
    for pf in plan_files:
        with open(pf, "r", encoding="utf-8") as fh:
            plan = json.load(fh)

        try:
            gws = _extract_gws(plan, pf)
        except KeyError as e:
            pytest.fail(f"{pf}: {e}")

        assert gws, f"{pf}: no GW entries extracted"

        for g in gws:
            gw = g.get("gw")
            xi: List[Dict[str, Any]] = g.get("xi") or []
            bench: List[Dict[str, Any]] = g.get("bench") or []

            # XI size = 11
            assert len(xi) == 11, f"{pf} GW{gw}: XI must be 11; got {len(xi)}"

            # Formation legal + matches counts
            form = g.get("formation") or _formation_from_xi(xi)
            assert form in LEGAL_FORMATIONS, f"{pf} GW{gw}: illegal formation={form}"
            try:
                d, m, f = [int(x) for x in form.split("-")]
            except Exception:
                pytest.fail(f"{pf} GW{gw}: cannot parse formation={form}")
            from collections import Counter as _C
            cnt = _C(_pos(p) for p in xi)
            assert cnt.get("DEF",0)==d and cnt.get("MID",0)==m and cnt.get("FWD",0)==f, \
                f"{pf} GW{gw}: formation {form} vs counts {cnt}"

            # Captain: exactly one in XI; must not be GK
            captain_id = str(g.get("captain_id") or "")
            if not captain_id:
                pytest.fail(f"{pf} GW{gw}: captain_id missing")
            cap_seen = 0
            cap_pos = None
            for p in xi:
                if _pid(p) == captain_id:
                    cap_seen += 1
                    cap_pos = _pos(p)
            assert cap_seen == 1, f"{pf} GW{gw}: captain must be exactly one in XI; found {cap_seen}"
            assert cap_pos != "GK", f"{pf} GW{gw}: GK captain is not allowed"

            # ≤3 per team (XI or XI+bench)
            scope = xi + (bench if enforce_bench_team_limit else [])
            # guard against accidental duplicate entries in bench/order
            uniq_by_id: Dict[str, Dict[str, Any]] = {}
            for p in scope:
                pid = _pid(p)
                if pid and pid not in uniq_by_id:
                    uniq_by_id[pid] = p
            from collections import Counter as _C2
            team_counts = _C2(_team(p) for p in uniq_by_id.values() if _team(p))
            over = {t:c for t,c in team_counts.items() if c > SQUAD_LIMIT_PER_TEAM}
            assert not over, f"{pf} GW{gw}: >{SQUAD_LIMIT_PER_TEAM} from one team: {over}"

            # Bank >= 0
            bank = _bank_after(g, plan_root=plan)
            assert bank >= 0.0, f"{pf} GW{gw}: bank_after < 0: {bank}"
