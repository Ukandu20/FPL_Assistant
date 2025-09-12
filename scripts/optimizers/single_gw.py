#!/usr/bin/env python3
"""
Single-GW selector (MILP) — chip-aware, team validator, rich JSON output

Features
- Chips: WC/FH(no hit cost & no FT cap), TC(multiplier), BB(all 15 count).
- Team column: accepts 'team' (preferred) or legacy 'team_quota_key'.
- Validates team codes vs --teams-json mapping (code -> team_id).
- Human-readable output with UTF-8 names and xPts rounded to 1 dp.
- Budget breakdown with bank_before/bank_after.
- Formation string (DEF-MID-FWD), enriched transfers details.

Usage
python -m scripts.optimizers.single_gw \
  --team-state data/processed/registry/state/team_state.json \
  --optimizer-input data/aggregator/optimizer_input.parquet \
  --teams-json data/processed/registry/_id_lookup_teams.json \
  --out data/plans/gw04.json \
  --gw 4 --chip WC --allow-hits
"""
from __future__ import annotations

import argparse, json, os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    import pulp  # type: ignore
except Exception:
    raise SystemExit("pulp is required (pip install pulp).")

TEAM_COL_CANDIDATES = ("team", "team_quota_key")

# ---------------- helpers ----------------
def _read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet: {path}: {e}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("optimizer_input must be .parquet or .csv")

def _parse_topk(s: str) -> Dict[str, int]:
    out = {"GK": 5, "DEF": 15, "MID": 15, "FWD": 10}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.strip().split(":")
        out[k.strip().upper()] = int(v)
    return out

def _round1(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    val = float(x)
    r = round(val + (1e-12 if val >= 0 else -1e-12), 1)
    return 0.0 if r == 0.0 else r

def _get_team_col_name(df: pd.DataFrame) -> str:
    for c in TEAM_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("optimizer_input must include 'team' (preferred) or 'team_quota_key'.")

def _load_team_lookup_json(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path: return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {str(k).upper(): str(v) for k, v in d.items()}  # code -> team_id

def _validate_teams(df: pd.DataFrame, state: dict, teams_json: Optional[str]) -> None:
    team_col = _get_team_col_name(df)
    if "team_id" not in df.columns:
        raise ValueError("optimizer_input missing 'team_id'")
    chk = df.copy()
    chk[team_col] = chk[team_col].astype(str).str.strip().str.upper()
    chk["team_id"] = chk["team_id"].astype(str)

    # internal: single code per team_id in this file
    nunique = chk.groupby("team_id")[team_col].nunique()
    bad_ids = nunique[nunique > 1]
    if not bad_ids.empty:
        details = (chk[chk["team_id"].isin(bad_ids.index)]
                   .groupby(["team_id", team_col]).size()
                   .reset_index(name="rows").sort_values(["team_id","rows"], ascending=[True,False]))
        raise ValueError("Inconsistent team code per team_id:\n" + details.to_string(index=False))

    # canonical mapping check
    cmap = _load_team_lookup_json(teams_json)
    if cmap:
        unknown = chk[~chk[team_col].isin(cmap.keys())][[team_col]].drop_duplicates()
        if not unknown.empty:
            raise ValueError("Unknown team code(s) vs mapping:\n" + unknown.to_string(index=False))
        tmp = chk.copy()
        tmp["canon_team_id"] = tmp[team_col].map(cmap)
        mism = tmp[tmp["canon_team_id"].notna() & (tmp["team_id"] != tmp["canon_team_id"])]
        if not mism.empty:
            slim = (mism.groupby([team_col, "team_id", "canon_team_id"])
                    .size().reset_index(name="rows").sort_values("rows", ascending=False))
            raise ValueError("team_id mismatch vs mapping (code -> team_id):\n" + slim.to_string(index=False))

# --------------- core MILP ----------------
def solve_single_gw(
    team_state_path: str,
    optimizer_input_path: str,
    out_path: str,
    gw: Optional[int] = None,
    risk_lambda: float = 0.0,
    topk: Optional[Dict[str, int]] = None,
    allow_hits: bool = True,
    max_extra_transfers: int = 3,
    cap_cannot_equal_vice: bool = True,
    formation_bounds: Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]] = ((3,5),(2,5),(1,3)),
    chip: Optional[str] = None,         # None|WC|FH|TC|BB
    tc_multiplier: float = 3.0,
    teams_json: Optional[str] = None,   # path to _id_lookup_teams.json
    verbose: bool = False,
) -> dict:
    # State
    with open(team_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    bank = float(state.get("bank", 0.0))
    free_transfers = int(state.get("free_transfers", 1))
    season = str(state.get("season"))
    snapshot_gw = int(state.get("gw"))
    squad_owned: Set[str] = {str(p["player_id"]) for p in state.get("squad", [])}
    owned_sell_map: Dict[str, float] = {str(p["player_id"]): float(p["sell_price"]) for p in state.get("squad", [])}

    # Data
    df = _read_any(optimizer_input_path)
    required = ["season","gw","player_id","team_id","pos","price","sell_price",
                "p60","exp_pts_mean","exp_pts_var","cs_prob","is_dgw","captain_uplift"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"optimizer_input missing columns: {missing}")
    _ = _get_team_col_name(df)

    if gw is not None:
        df = df[df["gw"] == gw].copy()
        if df.empty: raise ValueError(f"No optimizer_input rows for gw={gw}")
    else:
        gws = sorted(df["gw"].unique().tolist())
        if len(gws) != 1: raise ValueError("optimizer_input has multiple GWs; specify --gw")
        gw = int(gws[0])

    _validate_teams(df, state=state, teams_json=teams_json)

    # Candidate pool: all owned + topK per position by EV
    team_col = _get_team_col_name(df)
    df["owned"] = df["player_id"].astype(str).isin(squad_owned)
    df[team_col] = df[team_col].astype(str).str.strip().str.upper()

    keep_rows: List[pd.DataFrame] = []
    kmap = topk or {"GK": 5, "DEF": 15, "MID": 15, "FWD": 10}
    for pos_name, g in df.groupby("pos", as_index=False):
        g = g.sort_values(["owned","exp_pts_mean"], ascending=[False, False])
        k = kmap.get(pos_name, 10)
        keep_rows.append(pd.concat([g[g["owned"]], g[~g["owned"]].head(k)], ignore_index=True))
    pool = pd.concat(keep_rows, ignore_index=True).drop_duplicates("player_id").reset_index(drop=True)

    # Arrays
    pid    = pool["player_id"].astype(str).tolist()
    pos    = pool["pos"].astype(str).tolist()
    teams  = pool[team_col].astype(str).tolist()
    ownedm = pool["owned"].astype(bool).to_numpy()
    price  = pool["price"].astype(float).to_numpy()
    ev     = pool["exp_pts_mean"].astype(float).to_numpy()
    var    = pool["exp_pts_var"].astype(float).clip(lower=0.0).to_numpy()
    capup  = pool["captain_uplift"].astype(float).clip(lower=0.0).to_numpy()

    names: List[Optional[str]] = [None]*len(pool)
    if "player" in pool.columns:
        names = [None if pd.isna(v) else str(v) for v in pool["player"].astype(object)]

    N = len(pool)
    m = pulp.LpProblem("single_gw_selector", pulp.LpMaximize)

    # Vars
    in_squad = pulp.LpVariable.dicts("in_squad", range(N), 0, 1, cat=pulp.LpBinary)
    buy      = pulp.LpVariable.dicts("buy",      range(N), 0, 1, cat=pulp.LpBinary)
    sell     = pulp.LpVariable.dicts("sell",     range(N), 0, 1, cat=pulp.LpBinary)
    start    = pulp.LpVariable.dicts("start",    range(N), 0, 1, cat=pulp.LpBinary)
    cap      = pulp.LpVariable.dicts("cap",      range(N), 0, 1, cat=pulp.LpBinary)
    vcap     = pulp.LpVariable.dicts("vcap",     range(N), 0, 1, cat=pulp.LpBinary)
    hits     = pulp.LpVariable("hits", lowBound=0, upBound=max_extra_transfers, cat=pulp.LpInteger)

    bench_ranks = [1,2,3]
    bench = {r: pulp.LpVariable.dicts(f"bench_r{r}", range(N), 0, 1, cat=pulp.LpBinary) for r in bench_ranks}

    # Ownership transitions
    for i in range(N):
        if ownedm[i]:
            m += in_squad[i] + sell[i] == 1, f"owned_transition_{i}"
            m += buy[i] == 0, f"cant_buy_owned_{i}"
        else:
            m += in_squad[i] == buy[i], f"new_transition_{i}"
            m += sell[i] == 0, f"cant_sell_not_owned_{i}"

    # Squad size & composition
    def _sum_pos(pname: str, vec): return pulp.lpSum(vec[i] for i in range(N) if pos[i] == pname)
    m += pulp.lpSum(in_squad[i] for i in range(N)) == 15, "squad_size_15"
    m += _sum_pos("GK",  in_squad) == 2, "comp_gk_2"
    m += _sum_pos("DEF", in_squad) == 5, "comp_def_5"
    m += _sum_pos("MID", in_squad) == 5, "comp_mid_5"
    m += _sum_pos("FWD", in_squad) == 3, "comp_fwd_3"

    # Team cap ≤3
    for t in sorted(set(teams)):
        m += pulp.lpSum(in_squad[i] for i in range(N) if teams[i] == t) <= 3, f"teamcap_{t}"

    # XI & formation
    m += pulp.lpSum(start[i] for i in range(N)) == 11, "xi_size_11"
    for i in range(N):
        m += start[i] <= in_squad[i], f"start_in_squad_{i}"
    (DEF_min,DEF_max),(MID_min,MID_max),(FWD_min,FWD_max) = formation_bounds
    m += _sum_pos("GK",  start) == 1, "xi_gk_1"
    m += _sum_pos("DEF", start) >= DEF_min; m += _sum_pos("DEF", start) <= DEF_max
    m += _sum_pos("MID", start) >= MID_min; m += _sum_pos("MID", start) <= MID_max
    m += _sum_pos("FWD", start) >= FWD_min; m += _sum_pos("FWD", start) <= FWD_max

    # C/VC
    m += pulp.lpSum(cap[i] for i in range(N)) == 1, "one_captain"
    m += pulp.lpSum(vcap[i] for i in range(N)) == 1, "one_vice"
    for i in range(N):
        m += cap[i]  <= start[i]; m += vcap[i] <= start[i]
        if cap_cannot_equal_vice: m += cap[i] + vcap[i] <= 1

    # Transfers & hits (chip-aware)
    transfers_cnt = pulp.lpSum(buy[i] for i in range(N))
    chip_norm = (chip or "").upper() or None
    if chip_norm in {"WC","FH"}:
        m += hits == 0, "hits_zero_chip"
    else:
        if not allow_hits:
            m += transfers_cnt <= free_transfers, "no_hits_allowed"
            m += hits == 0, "hits_zero"
        else:
            m += hits >= transfers_cnt - free_transfers, "hits_lb"
            m += hits <= max_extra_transfers, "hits_cap"

    # Budget
    proceeds_expr = pulp.lpSum(sell[i] * float( (owned_sell_map.get(pid[i], 0.0)) ) for i in range(N))
    cost_expr     = pulp.lpSum(buy[i]  * float(price[i]) for i in range(N))
    m += cost_expr <= bank + proceeds_expr, "budget"

    # Bench ordering (outfield only)
    for r in bench_ranks:
        m += pulp.lpSum(bench[r][i] for i in range(N) if pos[i] != "GK") == 1
    for i in range(N):
        if pos[i] == "GK":
            for r in bench_ranks: m += bench[r][i] == 0
        else:
            m += pulp.lpSum(bench[r][i] for r in bench_ranks) <= 1
            for r in bench_ranks: m += bench[r][i] <= in_squad[i] - start[i]

    # Objective (chip-aware)
    team_ev_term = pulp.lpSum(start[i] * ev[i] for i in range(N))
    cap_uplift   = pulp.lpSum(cap[i] * capup[i] for i in range(N))
    if chip_norm == "TC":
        factor = max(0.0, float(tc_multiplier) - 1.0)  # e.g., 3x => +2x uplift
        cap_uplift = pulp.lpSum(cap[i] * (factor * capup[i]) for i in range(N))
    if chip_norm == "BB":
        team_ev_term = pulp.lpSum(in_squad[i] * ev[i] for i in range(N))

    obj = team_ev_term + cap_uplift - float(risk_lambda) * pulp.lpSum(start[i] * var[i] for i in range(N))
    if chip_norm not in {"WC","FH"}:
        obj = obj - 4.0 * hits
    m += obj

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=bool(verbose))
    res = m.solve(solver)
    if pulp.LpStatus[res] != "Optimal":
        raise RuntimeError(f"MILP not optimal: status={pulp.LpStatus[res]}")

    # Extract
    def picks(mask): return [pid[i] for i in range(N) if pulp.value(mask[i]) > 0.5]
    xi_ids   = picks(start)
    cap_pid  = picks(cap)[0]
    vcap_pid = picks(vcap)[0]
    buy_ids  = picks(buy)
    sell_ids = picks(sell)

    # Bench order & bench GK
    outfield = [i for i in range(N) if pos[i] != "GK"]
    bench_order_ids: List[Optional[str]] = [None, None, None]
    for r in bench_ranks:
        for i in outfield:
            if pulp.value(bench[r][i]) > 0.5:
                bench_order_ids[r-1] = pid[i]; break
    gk_ids = [pid[i] for i in range(N) if pos[i] == "GK" and pulp.value(in_squad[i]) > 0.5]
    gk_start = [pid[i] for i in range(N) if pos[i] == "GK" and pulp.value(start[i]) > 0.5]
    gk_bench = [x for x in gk_ids if x not in gk_start]
    bench_gk_id = gk_bench[0] if gk_bench else None

    # Maps
    name_map = {pid[i]: (None if names[i] is None else names[i]) for i in range(N)}
    pos_map  = {pid[i]: pos[i] for i in range(N)}
    team_map = {pid[i]: teams[i] for i in range(N)}
    ev_map   = {pid[i]: float(ev[i]) for i in range(N)}
    price_map= {pid[i]: float(price[i]) for i in range(N)}
    sell_map = {pid[i]: float(owned_sell_map.get(pid[i], 0.0)) for i in range(N)}

    def _pobj(xid: Optional[str]) -> Optional[dict]:
        if xid is None: return None
        return {"id": xid, "name": name_map.get(xid), "pos": pos_map.get(xid),
                "team": team_map.get(xid), "xPts": _round1(ev_map.get(xid))}

    # Transfers enriched
    transfers_out: List[dict] = []
    remaining_buys = list(buy_ids)
    for out_id in sell_ids:
        in_id = remaining_buys.pop(0) if remaining_buys else None
        buy_price  = float(price_map.get(in_id, 0.0)) if in_id else None
        sell_value = float(sell_map.get(out_id, 0.0))
        pair_net   = (buy_price if buy_price is not None else 0.0) - sell_value
        transfers_out.append({
            "out": out_id, "out_name": name_map.get(out_id), "out_pos": pos_map.get(out_id),
            "out_team": team_map.get(out_id), "out_xPts": _round1(ev_map.get(out_id)),
            "in": in_id,  "in_name": name_map.get(in_id) if in_id else None,
            "in_pos": pos_map.get(in_id) if in_id else None,
            "in_team": team_map.get(in_id) if in_id else None,
            "in_xPts": _round1(ev_map.get(in_id)) if in_id else None,
            "sell_value": sell_value, "buy_price": buy_price,
            "pair_net": _round1(pair_net),
            "price_delta": float(0.0 - sell_value) if in_id else float(0.0 - sell_value),
        })
    for in_id in remaining_buys:
        buy_price = float(price_map.get(in_id, 0.0))
        transfers_out.append({
            "out": None, "out_name": None, "out_pos": None, "out_team": None, "out_xPts": None,
            "in": in_id, "in_name": name_map.get(in_id), "in_pos": pos_map.get(in_id),
            "in_team": team_map.get(in_id), "in_xPts": _round1(ev_map.get(in_id)),
            "sell_value": None, "buy_price": buy_price,
            "pair_net": _round1(buy_price), "price_delta": float(buy_price),
        })

    # Objective parts (chip-aware EV term for reporting)
    if chip_norm == "BB":
        ev_start = float(sum(ev[i] * pulp.value(in_squad[i]) for i in range(N)))
    else:
        ev_start = float(sum(ev[i] * pulp.value(start[i]) for i in range(N)))
    if chip_norm == "TC":
        factor = max(0.0, float(tc_multiplier) - 1.0)
        ev_cap = float(sum((factor * capup[i]) * pulp.value(cap[i]) for i in range(N)))
    else:
        ev_cap = float(sum(capup[i] * pulp.value(cap[i]) for i in range(N)))
    var_pen = float(risk_lambda * sum(var[i] * pulp.value(start[i]) for i in range(N)))
    hits_val = 0.0 if chip_norm in {"WC","FH"} else float(4.0 * pulp.value(hits))
    total = ev_start + ev_cap - var_pen - hits_val

    # Budget math
    buys_cost = sum(price_map[x] for x in buy_ids)
    sells_proceeds = sum(sell_map[x] for x in sell_ids)
    bank_after = bank + sells_proceeds - buys_cost

    # Formation string
    nDEF = int(sum(1 for i in range(N) if pos[i] == "DEF" and pulp.value(start[i]) > 0.5))
    nMID = int(sum(1 for i in range(N) if pos[i] == "MID" and pulp.value(start[i]) > 0.5))
    nFWD = int(sum(1 for i in range(N) if pos[i] == "FWD" and pulp.value(start[i]) > 0.5))
    formation_str = f"{nDEF}-{nMID}-{nFWD}"

    # Bindings (heuristic)
    bindings: List[str] = []
    if abs(sum(pulp.value(in_squad[i]) for i in range(N)) - 15) <= 1e-6: bindings.append("squad_size")
    if abs(sum(pulp.value(start[i]) for i in range(N)) - 11) <= 1e-6: bindings.append("xi_size")
    lhs_cost = sum(pulp.value(buy[i]) * price[i] for i in range(N))
    rhs_budget = bank + sum(pulp.value(sell[i]) * sell_map.get(pid[i], 0.0) for i in range(N))
    if abs(lhs_cost - rhs_budget) <= 1e-5 or lhs_cost > rhs_budget - 1e-5: bindings.append("budget")
    for t in sorted(set(teams)):
        team_count = sum(pulp.value(in_squad[i]) for i in range(N) if teams[i] == t)
        if team_count >= 3 - 1e-5: bindings.append("3-per-team"); break

    plan = {
        "objective": {
            "ev": _round1(ev_start), "hit_cost": _round1(hits_val),
            "risk_penalty": _round1(var_pen), "total": _round1(total),
        },
        "meta": {
            "season": season, "gw": gw, "free_transfers": free_transfers,
            "bank_before": _round1(bank), "bank_after": _round1(bank_after),
            "budget": {
                "buys_cost": _round1(buys_cost), "sells_proceeds": _round1(sells_proceeds),
                "net_spend": _round1(buys_cost - sells_proceeds),
            },
            "snapshot_gw": snapshot_gw, "formation": formation_str,
        },
        "chip": chip_norm or None,
        "transfers": transfers_out,
        "xi": [_pobj(x) for x in xi_ids],
        "bench": {"order": [_pobj(x) for x in bench_order_ids], "gk": _pobj(bench_gk_id)},
        "captain": _pobj(cap_pid), "vice": _pobj(vcap_pid),
        "explanations": {
            "binding_constraints": sorted(set(bindings)),
            "notes": [
                "Captain adds captain_uplift on top of XI EV (TC scales uplift).",
                "WC/FH: unlimited transfers with zero hit cost this GW; budget still enforced.",
                "BB: counts EV of all 15 players.",
                "Outfield bench ranks are 1..3; GK bench is implied.",
                "Team column validated vs --teams-json mapping.",
                "FH note: post-GW squad reverts; bank_after is transient."
            ],
        },
    }

    out_dir = os.path.dirname(out_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return plan

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Single-GW MILP selector (transfers + XI + C/VC)")
    ap.add_argument("--team-state", required=True)
    ap.add_argument("--optimizer-input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gw", type=int)
    ap.add_argument("--risk-lambda", type=float, default=0.0)
    ap.add_argument("--topk", default="GK:5,DEF:15,MID:15,FWD:10")
    ap.add_argument("--allow-hits", action="store_true")
    ap.add_argument("--max-extra-transfers", type=int, default=3)
    ap.add_argument("--no-cap-neq-vice", action="store_true",
                    help="allow same player as C and VC (not recommended)")
    ap.add_argument("--chip", choices=["WC","FH","TC","BB"], help="Apply chip logic")
    ap.add_argument("--tc-multiplier", type=float, default=3.0, help="Triple Captain multiplier (default 3x)")
    ap.add_argument("--teams-json", help="Path to _id_lookup_teams.json (code -> team_id) for strict validation")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    plan = solve_single_gw(
        team_state_path=args.team_state,
        optimizer_input_path=args.optimizer_input,
        out_path=args.out,
        gw=args.gw,
        risk_lambda=args.risk_lambda,
        topk=_parse_topk(args.topk),
        allow_hits=bool(args.allow_hits),
        max_extra_transfers=int(args.max_extra_transfers),
        cap_cannot_equal_vice=not args.no_cap_neq_vice,
        chip=args.chip,
        tc_multiplier=args.tc_multiplier,
        teams_json=args.teams_json,
        verbose=bool(args.verbose),
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
