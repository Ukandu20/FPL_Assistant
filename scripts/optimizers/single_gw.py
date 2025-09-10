#!/usr/bin/env python3
"""
Single-GW selector (MILP) — chip-aware, rich JSON output with xPts (1 dp) and bank_after

Chooses transfers, XI, bench order, captain/vice to maximize:
  team_EV (+ captain uplift) - 4*hits - λ*variance
Chips:
- WC/FH: unlimited transfers, no hit cost and no FT cap (budget still enforced)
- TC: captain gets `tc_multiplier` total (default 3x)
- BB: counts EV of all 15 players

Output:
- meta.bank_before = input bank (from team_state)
- meta.bank_after  = bank after the selected transfers (this GW)
- meta.budget      = buys_cost, sells_proceeds, net_spend
All xPts are rounded to 1 dp; objective breakdown rounded to 1 dp.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    import pulp  # type: ignore
except Exception:
    raise SystemExit("pulp is required (pip install pulp).")


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
    out: Dict[str, int] = {"GK": 5, "DEF": 15, "MID": 15, "FWD": 10}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        out[k.strip().upper()] = int(v)
    return out


def _round1(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    val = float(x)
    r = round(val + (1e-12 if val >= 0 else -1e-12), 1)
    return 0.0 if r == 0.0 else r


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
    chip: Optional[str] = None,       # None|"WC"|"FH"|"TC"|"BB"
    tc_multiplier: float = 3.0,       # total multiplier for TC (typically 3.0)
    verbose: bool = False,
) -> dict:
    # Load state
    with open(team_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    bank: float = float(state.get("bank", 0.0))
    free_transfers: int = int(state.get("free_transfers", 1))
    season: str = str(state.get("season"))
    snapshot_gw: int = int(state.get("gw"))
    squad_owned: Set[str] = {str(p["player_id"]) for p in state.get("squad", [])}
    owned_sell_map: Dict[str, float] = {str(p["player_id"]): float(p["sell_price"]) for p in state.get("squad", [])}

    # Load optimizer input
    df = _read_any(optimizer_input_path)
    required = [
        "season","gw","player_id","team_id","pos","price","sell_price",
        "p60","exp_pts_mean","exp_pts_var","cs_prob","is_dgw","team_quota_key","captain_uplift"
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"optimizer_input missing column: {c}")

    if gw is not None:
        df = df[df["gw"] == gw].copy()
        if df.empty:
            raise ValueError(f"No optimizer_input rows for gw={gw}")
    else:
        gws = sorted(df["gw"].unique().tolist())
        if len(gws) != 1:
            raise ValueError("optimizer_input contains multiple GWs; specify --gw")
        gw = int(gws[0])

    # Candidate pool: all owned + topK per position
    df["owned"] = df["player_id"].astype(str).isin(squad_owned)
    keep_rows = []
    kmap = topk or {"GK": 5, "DEF": 15, "MID": 15, "FWD": 10}
    for pos_name, g in df.groupby("pos", as_index=False):
        g = g.sort_values(["owned", "exp_pts_mean"], ascending=[False, False])
        k = kmap.get(pos_name, 10)
        owned_g = g[g["owned"]]
        cand_g  = g[~g["owned"]].head(k)
        keep_rows.append(pd.concat([owned_g, cand_g], ignore_index=True))
    pool = pd.concat(keep_rows, ignore_index=True).drop_duplicates("player_id").reset_index(drop=True)

    # Convenience arrays
    pid    = pool["player_id"].astype(str).tolist()
    pos    = pool["pos"].astype(str).tolist()
    teams  = pool["team_quota_key"].astype(str).tolist()
    owned_mask = pool["owned"].astype(bool).to_numpy()
    price  = pool["price"].astype(float).to_numpy()
    ev     = pool["exp_pts_mean"].astype(float).to_numpy()
    var    = pool["exp_pts_var"].astype(float).clip(lower=0.0).to_numpy()
    cap_up = pool["captain_uplift"].astype(float).clip(lower=0.0).to_numpy()

    if "player" in pool.columns:
        _series = pool["player"].astype(object)
        names: List[Optional[str]] = [None if pd.isna(v) else str(v) for v in _series]
    else:
        names = [None] * len(pool)

    N = len(pool)

    # MILP model
    m = pulp.LpProblem("single_gw_selector", pulp.LpMaximize)

    # Decision variables
    in_squad = pulp.LpVariable.dicts("in_squad", range(N), 0, 1, cat=pulp.LpBinary)
    buy      = pulp.LpVariable.dicts("buy",      range(N), 0, 1, cat=pulp.LpBinary)
    sell     = pulp.LpVariable.dicts("sell",     range(N), 0, 1, cat=pulp.LpBinary)
    start    = pulp.LpVariable.dicts("start",    range(N), 0, 1, cat=pulp.LpBinary)
    cap      = pulp.LpVariable.dicts("cap",      range(N), 0, 1, cat=pulp.LpBinary)
    vcap     = pulp.LpVariable.dicts("vcap",     range(N), 0, 1, cat=pulp.LpBinary)

    hits = pulp.LpVariable("hits", lowBound=0, upBound=max_extra_transfers, cat=pulp.LpInteger)

    # Bench assignment for outfield bench ranks r∈{1,2,3}
    bench_ranks = [1, 2, 3]
    bench = {r: pulp.LpVariable.dicts(f"bench_r{r}", range(N), 0, 1, cat=pulp.LpBinary) for r in bench_ranks}

    # Ownership transitions
    for i in range(N):
        if owned_mask[i]:
            m += in_squad[i] + sell[i] == 1, f"owned_transition_{i}"
            m += buy[i] == 0, f"cant_buy_owned_{i}"
        else:
            m += in_squad[i] == buy[i], f"new_transition_{i}"
            m += sell[i] == 0, f"cant_sell_not_owned_{i}"

    # Squad size & composition
    def _sum_pos(pname: str, vec): return pulp.lpSum(vec[i] for i in range(N) if pos[i] == pname)
    m += pulp.lpSum(in_squad[i] for i in range(N)) == 15, "squad_size_15"
    m += _sum_pos("GK", in_squad) == 2, "comp_gk_2"
    m += _sum_pos("DEF", in_squad) == 5, "comp_def_5"
    m += _sum_pos("MID", in_squad) == 5, "comp_mid_5"
    m += _sum_pos("FWD", in_squad) == 3, "comp_fwd_3"

    # Team ≤3
    for t in sorted(set(teams)):
        m += pulp.lpSum(in_squad[i] for i in range(N) if teams[i] == t) <= 3, f"teamcap_{t}"

    # XI + formation
    m += pulp.lpSum(start[i] for i in range(N)) == 11, "xi_size_11"
    for i in range(N):
        m += start[i] <= in_squad[i], f"start_in_squad_{i}"

    (DEF_min, DEF_max), (MID_min, MID_max), (FWD_min, FWD_max) = formation_bounds
    m += _sum_pos("GK", start) == 1, "xi_gk_1"
    m += _sum_pos("DEF", start) >= DEF_min, "xi_def_min"
    m += _sum_pos("DEF", start) <= DEF_max, "xi_def_max"
    m += _sum_pos("MID", start) >= MID_min, "xi_mid_min"
    m += _sum_pos("MID", start) <= MID_max, "xi_mid_max"
    m += _sum_pos("FWD", start) >= FWD_min, "xi_fwd_min"
    m += _sum_pos("FWD", start) <= FWD_max, "xi_fwd_max"

    # Captain/vice
    m += pulp.lpSum(cap[i] for i in range(N)) == 1, "one_captain"
    m += pulp.lpSum(vcap[i] for i in range(N)) == 1, "one_vice"
    for i in range(N):
        m += cap[i]  <= start[i], f"cap_starts_{i}"
        m += vcap[i] <= start[i], f"vcap_starts_{i}"
        if cap_cannot_equal_vice:
            m += cap[i] + vcap[i] <= 1, f"cap_neq_vcap_{i}"

    # Transfers & hits — chip-aware
    transfers_cnt = pulp.lpSum(buy[i] for i in range(N))
    chip_norm = (chip or "").upper() or None
    if chip_norm in {"WC", "FH"}:
        m += hits == 0, "hits_zero_chip"
    else:
        if not allow_hits:
            m += transfers_cnt <= free_transfers, "no_hits_allowed"
            m += hits == 0, "hits_zero"
        else:
            m += hits >= transfers_cnt - free_transfers, "hits_lb"
            m += hits >= 0, "hits_nonneg"
            m += hits <= max_extra_transfers, "hits_cap"

    # Budget
    proceeds_expr = pulp.lpSum(sell[i] * owned_sell_map.get(pid[i], 0.0) for i in range(N))
    cost_expr     = pulp.lpSum(buy[i]  *                 float(price[i]) for i in range(N))
    m += cost_expr <= bank + proceeds_expr, "budget"

    # Bench ordering (outfield only)
    for r in bench_ranks:
        m += (pulp.lpSum(bench[r][i] for i in range(N) if pos[i] != "GK") == 1), f"bench_rank_unique_{r}"
    for i in range(N):
        if pos[i] == "GK":
            for r in bench_ranks:
                m += bench[r][i] == 0, f"bench_gk_forbidden_{i}_{r}"
        else:
            m += pulp.lpSum(bench[r][i] for r in bench_ranks) <= 1, f"bench_one_rank_{i}"
            for r in bench_ranks:
                m += bench[r][i] <= in_squad[i] - start[i], f"bench_only_nxi_{i}_{r}"

    # Objective — chip-aware
    team_ev_term = pulp.lpSum(start[i] * ev[i] for i in range(N))
    cap_uplift_term = pulp.lpSum(cap[i] * cap_up[i] for i in range(N))
    if chip_norm == "TC":
        factor = max(0.0, float(tc_multiplier) - 1.0)  # TC 3x → extra 2x uplift
        cap_uplift_term = pulp.lpSum(cap[i] * (factor * cap_up[i]) for i in range(N))
    if chip_norm == "BB":
        team_ev_term = pulp.lpSum(in_squad[i] * ev[i] for i in range(N))

    obj = team_ev_term + cap_uplift_term - float(risk_lambda) * pulp.lpSum(start[i] * var[i] for i in range(N))
    if chip_norm not in {"WC", "FH"}:
        obj = obj - 4.0 * hits
    m += obj

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=bool(verbose))
    res = m.solve(solver)
    if pulp.LpStatus[res] != "Optimal":
        raise RuntimeError(f"MILP not optimal: status={pulp.LpStatus[res]}")

    # Extract solution
    def picks(mask): return [pid[i] for i in range(N) if pulp.value(mask[i]) > 0.5]
    xi       = picks(start)
    cap_pid  = picks(cap)[0]
    vcap_pid = picks(vcap)[0]
    buys     = picks(buy)
    sells    = picks(sell)

    # Bench order (outfield)
    outfield = [i for i in range(N) if pos[i] != "GK"]
    bench_order: List[Optional[str]] = [None, None, None]
    for r in bench_ranks:
        for i in outfield:
            if pulp.value(bench[r][i]) > 0.5:
                bench_order[r-1] = pid[i]
                break

    # Bench GK
    gk_ids   = [pid[i] for i in range(N) if pos[i] == "GK" and pulp.value(in_squad[i]) > 0.5]
    gk_start = [pid[i] for i in range(N) if pos[i] == "GK" and pulp.value(start[i]) > 0.5]
    gk_bench = [x for x in gk_ids if x not in gk_start]
    bench_gk = gk_bench[0] if gk_bench else None

    # Maps
    name_map = {pid[i]: (None if names[i] is None else names[i]) for i in range(N)}
    pos_map  = {pid[i]: pos[i] for i in range(N)}
    team_map = {pid[i]: teams[i] for i in range(N)}
    ev_map   = {pid[i]: float(ev[i]) for i in range(N)}
    price_map = {pid[i]: float(price[i]) for i in range(N)}
    sell_map  = {pid[i]: float(owned_sell_map.get(pid[i], 0.0)) for i in range(N)}

    def _pobj(pid_: Optional[str]) -> Optional[dict]:
        if pid_ is None:
            return None
        return {"id": pid_, "name": name_map.get(pid_), "pos": pos_map.get(pid_), "team": team_map.get(pid_), "xPts": _round1(ev_map.get(pid_))}

    # Transfers pretty print (do not mutate buys/sells lists)
    transfers_out: List[dict] = []
    remaining_buys = list(buys)
    for out_id in sells:
        in_id = remaining_buys.pop(0) if remaining_buys else None
        transfers_out.append({
            "out": out_id,
            "out_name": name_map.get(out_id),
            "out_pos":  pos_map.get(out_id),
            "out_team": team_map.get(out_id),
            "out_xPts": _round1(ev_map.get(out_id)),
            "in": in_id,
            "in_name": name_map.get(in_id) if in_id else None,
            "in_pos":  pos_map.get(in_id) if in_id else None,
            "in_team": team_map.get(in_id) if in_id else None,
            "in_xPts": _round1(ev_map.get(in_id)) if in_id else None,
            "price_delta": float(0.0 - sell_map.get(out_id, 0.0)),
        })
    for in_id in remaining_buys:
        transfers_out.append({
            "out": None, "out_name": None, "out_pos": None, "out_team": None, "out_xPts": None,
            "in": in_id, "in_name": name_map.get(in_id), "in_pos": pos_map.get(in_id), "in_team": team_map.get(in_id),
            "in_xPts": _round1(ev_map.get(in_id)),
            "price_delta": float(price_map.get(in_id, 0.0)),
        })

    # Budget and objective breakdowns (now 1 dp for objective)
    if chip_norm == "BB":
        ev_start = float(sum(ev[i] * pulp.value(in_squad[i]) for i in range(N)))
    else:
        ev_start = float(sum(ev[i] * pulp.value(start[i]) for i in range(N)))
    if chip_norm == "TC":
        factor = max(0.0, float(tc_multiplier) - 1.0)
        ev_cap = float(sum((factor * cap_up[i]) * pulp.value(cap[i]) for i in range(N)))
    else:
        ev_cap = float(sum(cap_up[i] * pulp.value(cap[i]) for i in range(N)))
    var_pen = float(risk_lambda * sum(var[i] * pulp.value(start[i]) for i in range(N)))
    hits_val = 0.0 if chip_norm in {"WC", "FH"} else float(4.0 * pulp.value(hits))
    total = ev_start + ev_cap - var_pen - hits_val

    # Compute realized GW budget usage from chosen buys/sells
    buys_cost = sum(price_map[x] for x in buys)
    sells_proceeds = sum(sell_map[x] for x in sells)
    bank_after = bank + sells_proceeds - buys_cost

    bindings: List[str] = []
    if abs(sum(pulp.value(in_squad[i]) for i in range(N)) - 15) <= 1e-6:
        bindings.append("squad_size")
    if abs(sum(pulp.value(start[i]) for i in range(N)) - 11) <= 1e-6:
        bindings.append("xi_size")
    lhs_cost = sum(pulp.value(buy[i]) * price[i] for i in range(N))
    rhs_budget = bank + sum(pulp.value(sell[i]) * owned_sell_map.get(pid[i], 0.0) for i in range(N))
    if abs(lhs_cost - rhs_budget) <= 1e-5 or lhs_cost > rhs_budget - 1e-5:
        bindings.append("budget")
    for t in sorted(set(teams)):
        team_count = sum(pulp.value(in_squad[i]) for i in range(N) if teams[i] == t)
        if team_count >= 3 - 1e-5:
            bindings.append("3-per-team")
            break

    plan = {
        "objective": {
            "ev": _round1(ev_start),
            "hit_cost": _round1(hits_val),
            "risk_penalty": _round1(var_pen),
            "total": _round1(total),
        },
        "meta": {
            "season": season,
            "gw": gw,
            "free_transfers": free_transfers,
            "bank_before": _round1(bank),
            "bank_after": _round1(bank_after),
            "budget": {
                "buys_cost": _round1(buys_cost),
                "sells_proceeds": _round1(sells_proceeds),
                "net_spend": _round1(buys_cost - sells_proceeds),
            },
            "snapshot_gw": snapshot_gw,
        },
        "chip": chip_norm or None,
        "transfers": transfers_out,
        "xi": [_pobj(x) for x in xi],
        "bench": {"order": [_pobj(x) for x in bench_order], "gk": _pobj(bench_gk)},
        "captain": _pobj(cap_pid),
        "vice": _pobj(vcap_pid),
        "explanations": {
            "binding_constraints": sorted(set(bindings)),
            "notes": [
                "Captain adds captain_uplift on top of XI EV (TC scales uplift).",
                "WC/FH: unlimited transfers with zero hit cost this GW; budget still enforced.",
                "Bench Boost: counts all 15 players' EV.",
                "Outfield bench ranks are 1..3; GK bench is implied.",
                "For FH, bank_after is transient; your squad reverts after the GW.",
            ],
        },
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    return plan


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
    ap.add_argument("--chip", choices=["WC","FH","TC","BB"],
                    help="Apply chip logic: WC/FH=no hit cost & no FT limit; TC=triple cap; BB=include bench EV")
    ap.add_argument("--tc-multiplier", type=float, default=3.0,
                    help="Triple Captain multiplier (default 3x)")
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
        verbose=bool(args.verbose),
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
