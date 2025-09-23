#!/usr/bin/env python3
"""
Multi-GW HOLD optimizer (MILP) — transfers only at first GW, squad held across the horizon.

Highlights
----------
• Exact FPL legality: squad 15 (2/5/5/3), XI=11 with valid formations.
• Optional --formation (e.g., 3-5-2) to force XI shape across all GWs.
• Chips:
  - WC: unlimited transfers at start GW, no hits that week.
  - FH: temporary 15+XI on that GW; persistent squad unchanged.
  - TC: triple-captain uplift on that GW only.
  - BB: EV term uses all 15 players on that GW.
• Sweep controls:
  - --sweep-free-transfers (K=1..FT), --sweep-include-hits (K=FT+1..FT+H).
• Output:
  - plan.json (per variant), transfers.json (for GW_start).
  - XI items include: id, name, pos, team, xPts, opp, is_home, venue, fdr.
• Chips not bounded by --max-extra-transfers:
  - WC has hits0==0; FH lifts hits upper bound to stack limit.
• Tie-break: tiny penalty on bench EV so, all else equal, higher-EV players start.

New in this version
-------------------
• Error out if --only-chip-gw includes any GW outside the horizon.
• GK captaincy is **forbidden** (both normal and TC).
"""

from __future__ import annotations

import argparse, json, os, re
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    import pulp  # type: ignore
except Exception:
    raise SystemExit("pulp is required (pip install pulp).")

# ---------- Constants ----------
TEAM_COL_CANDIDATES = ("team", "team_quota_key")
MAX_FREE_TRANSFERS_STACK = 5
HIT_COST = 4.0
EPS_BENCH = 1e-3  # tiny penalty on bench EV to break ties toward stronger XI

# Valid formations per FPL
VALID_FORMATIONS = {
    (3,4,3), (3,5,2),
    (4,3,3), (4,4,2), (4,5,1),
    (5,3,2), (5,4,1),
}

# ---------- Helpers ----------
def _read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet: {path}: {e}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("optimizer_input must be .parquet or .csv")

def _round1(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    val = float(x)
    r = round(val + (1e-12 if val >= 0 else -1e-12), 1)
    return 0.0 if r == 0.0 else r

def _get_team_col_name(df: pd.DataFrame) -> str:
    for c in TEAM_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("optimizer_input must include 'team' (preferred) or 'team_quota_key'.")

def _normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EV mean
    if "exp_pts_mean" not in df.columns:
        if "xPts" in df.columns:
            df["exp_pts_mean"] = pd.to_numeric(df["xPts"], errors="coerce").fillna(0.0)
        else:
            df["exp_pts_mean"] = 0.0
    # EV variance
    if "exp_pts_var" not in df.columns:
        if "exp_pts_std" in df.columns:
            s = pd.to_numeric(df["exp_pts_std"], errors="coerce").fillna(0.0)
            df["exp_pts_var"] = (s ** 2).astype(float)
        else:
            df["exp_pts_var"] = 0.0
    df["exp_pts_var"] = pd.to_numeric(df["exp_pts_var"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # captain uplift
    if "captain_uplift" not in df.columns:
        df["captain_uplift"] = 0.0
    df["captain_uplift"] = pd.to_numeric(df["captain_uplift"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # dgw
    if "is_dgw" in df.columns:
        df["is_dgw"] = (
            df["is_dgw"]
            .map(lambda v: 1 if str(v).strip().lower() in {"1","true","t","yes","y"} else 0)
            .astype(int)
        )
    else:
        df["is_dgw"] = 0
    # optional XI-context fields
    for c in ["opponent", "opponent_id", "is_home", "venue", "fdr"]:
        if c not in df.columns:
            df[c] = np.nan
    # normalize is_home to {0,1}
    df["is_home"] = df["is_home"].map(lambda v: 1 if str(v).strip().lower() in {"1","true","t","yes","y","h"} else 0).astype(int, errors="ignore")
    return df

def _validate_teams(df: pd.DataFrame) -> None:
    team_col = _get_team_col_name(df)
    if "team_id" not in df.columns:
        raise ValueError("optimizer_input missing 'team_id'")
    chk = df.copy()
    chk[team_col] = chk[team_col].astype(str).str.strip().str.upper()
    chk["team_id"] = chk["team_id"].astype(str)
    nunique = chk.groupby("team_id")[team_col].nunique()
    bad_ids = nunique[nunique > 1]
    if not bad_ids.empty:
        details = (chk[chk["team_id"].isin(bad_ids.index)]
                   .groupby(["team_id", team_col]).size()
                   .reset_index(name="rows").sort_values(["team_id", "rows"], ascending=[True, False]))
        raise ValueError("Inconsistent team code per team_id:\n" + details.to_string(index=False))

def _parse_topk(s: str) -> Dict[str, int]:
    out = {"GK": 10, "DEF": 25, "MID": 25, "FWD": 20}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.strip().split(":")
        out[k.strip().upper()] = int(v)
    return out

def _pool_union(df: pd.DataFrame, owned_ids: Set[str], gws: List[int], topk: Dict[str, int]) -> pd.DataFrame:
    keep: List[pd.DataFrame] = []
    for g in gws:
        gg = df[df["gw"] == g].copy()
        gg["owned"] = gg["player_id"].astype(str).isin(owned_ids)
        for pos_name, grp in gg.groupby("pos", as_index=False):
            grp = grp.sort_values(["owned", "exp_pts_mean"], ascending=[False, False])
            k = topk.get(pos_name, 10)
            keep.append(pd.concat([grp[grp["owned"]], grp[~grp["owned"]].head(k)], ignore_index=True))
    pool = (
        pd.concat(keep, ignore_index=True)
        .drop_duplicates(["player_id"])
        .reset_index(drop=True)
    )
    return pool

def _formation_from_cli(s: Optional[str]) -> Optional[Tuple[int,int,int]]:
    if not s:
        return None
    m = re.fullmatch(r"\s*([3-5])\s*-\s*([3-5])\s*-\s*([1-3])\s*", s)
    if not m:
        raise ValueError("--formation must look like '3-5-2', '4-4-2', etc.")
    d, m_, f = int(m.group(1)), int(m.group(2)), int(m.group(3))
    tpl = (d, m_, f)
    if tpl not in VALID_FORMATIONS:
        raise ValueError(f"--formation {s} is not an FPL-legal shape.")
    return tpl

def _order_by_pos_then_ev(pids: List[str], t: int, pid_index: Dict[str,int],
                          pos_arr: np.ndarray, ev: np.ndarray) -> List[str]:
    pos_order = {"GK":0, "DEF":1, "MID":2, "FWD":3}
    def key_func(pid: str):
        i = pid_index[pid]
        return (pos_order.get(str(pos_arr[i]), 9), -float(ev[t, i]))
    return sorted(pids, key=key_func)

# ---------- Core Solver (Hold Horizon) ----------
def solve_hold_horizon(
    team_state_path: str,
    optimizer_input_path: str,
    gw_start: int,
    next_k: int,
    risk_lambda: float,
    topk: Dict[str,int],
    allow_hits: bool,
    max_extra_transfers: int,
    exact_transfers: Optional[int],
    lock_ids: Set[str],
    ban_ids: Set[str],
    max_from_team: Optional[Dict[str,int]],
    forced_formation: Optional[Tuple[int,int,int]],
    chip: Optional[str],             # None | 'WC' | 'FH' | 'TC' | 'BB'
    chip_gw: Optional[int],          # GW to apply chip (WC forced to gw_start externally)
    time_limit: Optional[int],
    mip_gap: Optional[float],
    threads: Optional[int],
    verbose: bool,
) -> dict:
    # ---- Load state ----
    with open(team_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    bank0 = float(state.get("bank", 0.0))
    ft0   = int(state.get("free_transfers", 1))
    season = str(state.get("season"))
    snapshot_gw = int(state.get("gw"))
    owned0: List[dict] = list(state.get("squad", []))
    owned_ids0: Set[str] = {str(p["player_id"]) for p in owned0}
    sell_map0: Dict[str, float] = {
        str(p["player_id"]): float(p.get("sell_price", p.get("price", 0.0))) for p in owned0
    }

    # ---- Load data ----
    df_all = _read_any(optimizer_input_path)
    df_all = _normalize_input_columns(df_all)

    # Required columns
    core_required = [
        "season","gw","player_id","team_id","pos","price","sell_price",
        "p60","exp_pts_mean","exp_pts_var","cs_prob","is_dgw","captain_uplift"
    ]
    miss = [c for c in core_required if c not in df_all.columns]
    if miss:
        raise ValueError(f"optimizer_input missing columns: {miss}")

    _validate_teams(df_all)
    team_col = _get_team_col_name(df_all)

    # ---- Horizon ----
    gw_list = list(range(gw_start, gw_start + next_k))
    df = df_all[df_all["gw"].isin(gw_list)].copy()
    if df.empty:
        raise ValueError("optimizer_input has no rows for requested horizon")

    # ---- Candidate pool ----
    pool = _pool_union(df, owned_ids0, gw_list, topk)
    pid = pool["player_id"].astype(str).tolist()
    P = len(pid)
    pid_index = {pid[i]: i for i in range(P)}

    # Static attributes (from first appearance)
    rep = df[df["player_id"].astype(str).isin(pid)].copy().sort_values(["player_id","gw"]).drop_duplicates("player_id", keep="first")
    pos_arr = rep.set_index(rep["player_id"].astype(str))["pos"].reindex(pid).astype(str).to_numpy()
    name_arr = rep.set_index(rep["player_id"].astype(str)).get("player", rep["player_id"]).reindex(pid).astype(object).to_numpy()
    team_code = rep.set_index(rep["player_id"].astype(str))[team_col].reindex(pid).astype(str).str.upper().to_numpy()

    # Per-GW matrices (+ XI context)
    G = len(gw_list)
    gw_index = {gw_list[t]: t for t in range(G)}
    price = np.zeros((G,P), float)
    ev    = np.zeros((G,P), float)
    var   = np.zeros((G,P), float)
    capup = np.zeros((G,P), float)
    opp   = np.empty((G,P), dtype=object)
    ishm  = np.zeros((G,P), int)
    venue = np.empty((G,P), dtype=object)
    fdr   = np.zeros((G,P), float)

    opp[:] = None
    venue[:] = None
    fdr[:] = np.nan

    for g in gw_list:
        gg = df[df["gw"]==g]
        t = gw_index[g]
        for _,r in gg.iterrows():
            p = str(r["player_id"])
            if p not in pid_index: continue
            i = pid_index[p]
            price[t,i] = float(r["price"])
            ev[t,i]    = float(r["exp_pts_mean"])
            var[t,i]   = max(0.0, float(r["exp_pts_var"]))
            capup[t,i] = max(0.0, float(r["captain_uplift"]))
            # extras
            opp_val = r.get("opponent", None)
            opp[t,i] = None if pd.isna(opp_val) else str(opp_val).upper()
            ishome = int(r.get("is_home", 0)) if pd.notna(r.get("is_home", np.nan)) else 0
            ishm[t,i] = 1 if ishome==1 else 0
            venue[t,i] = "H" if ishm[t,i]==1 else "A"
            fdr_val = r.get("fdr", np.nan)
            try:
                fdr[t,i] = float(fdr_val) if pd.notna(fdr_val) else np.nan
            except Exception:
                fdr[t,i] = np.nan

    # ---- Model ----
    m = pulp.LpProblem("multi_gw_hold", pulp.LpMaximize)

    # hits upper bound logic (WC/FH not bound by --max-extra-transfers)
    hits_up_bound = 0 if chip == "WC" else (MAX_FREE_TRANSFERS_STACK if chip == "FH" else max_extra_transfers)

    # Decision vars
    in_squad = pulp.LpVariable.dicts("in", [i for i in range(P)], 0, 1, cat=pulp.LpBinary)
    start    = pulp.LpVariable.dicts("start", [(t,i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)

    bench = {r: pulp.LpVariable.dicts(f"bench_r{r}", [(t,i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
             for r in [1,2,3]}

    cap_n  = pulp.LpVariable.dicts("cap_n",  [(t,i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    cap_tc = pulp.LpVariable.dicts("cap_tc", [(t,i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)

    buy0  = pulp.LpVariable.dicts("buy0", [i for i in range(P)], 0, 1, cat=pulp.LpBinary)
    sell0 = pulp.LpVariable.dicts("sell0",[i for i in range(P)], 0, 1, cat=pulp.LpBinary)

    hits0 = pulp.LpVariable("hits0", lowBound=0, upBound=hits_up_bound, cat=pulp.LpInteger)
    ft_used0 = pulp.LpVariable("ft_used0", lowBound=0, upBound=MAX_FREE_TRANSFERS_STACK, cat=pulp.LpInteger)

    # FH temporary squad/XI
    tmp_in  = pulp.LpVariable.dicts("fh_in",  [(t,i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    tmp_sta = pulp.LpVariable.dicts("fh_sta", [(t,i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    tmp_capn= pulp.LpVariable.dicts("fh_cap_n",[(t,i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)
    tmp_capt= pulp.LpVariable.dicts("fh_cap_tc",[(t,i) for t in range(G) for i in range(P)], 0, 1, cat=pulp.LpBinary)

    # ----- Ownership transition only at GW0 -----
    owned_prev = np.array([1 if pid[i] in owned_ids0 else 0 for i in range(P)], dtype=int)
    for i in range(P):
        if owned_prev[i] == 1:
            m += buy0[i] == 0
            m += in_squad[i] + sell0[i] == 1
        else:
            m += sell0[i] == 0
            m += in_squad[i] == buy0[i]

    # ----- Budget & hits (WC special) -----
    transfers_cnt0 = pulp.lpSum(buy0[i] for i in range(P))
    if chip == "WC":
        m += hits0 == 0
        proceeds = pulp.lpSum(sell0[i] * float(sell_map0.get(pid[i], 0.0)) for i in range(P))
        cost     = pulp.lpSum(buy0[i] * price[0,i] for i in range(P))
        m += cost <= bank0 + proceeds
        if exact_transfers is not None:
            if exact_transfers < 0: raise ValueError("--exact-transfers must be >= 0")
            m += transfers_cnt0 == int(exact_transfers)
    else:
        if exact_transfers is not None:
            if exact_transfers < 0: raise ValueError("--exact-transfers must be >= 0")
            m += transfers_cnt0 == int(exact_transfers)
            if not allow_hits and exact_transfers > ft0:
                raise ValueError("exact_transfers > free_transfers but --allow-hits not set")
            if exact_transfers <= ft0 or not allow_hits:
                m += hits0 == 0
            else:
                m += hits0 == (int(exact_transfers) - ft0)
                m += hits0 <= hits_up_bound
        else:
            if not allow_hits:
                m += transfers_cnt0 <= ft0
                m += hits0 == 0
            else:
                m += hits0 >= transfers_cnt0 - ft0
                m += hits0 <= hits_up_bound
                m += transfers_cnt0 <= ft0 + hits_up_bound
        proceeds = pulp.lpSum(sell0[i] * float(sell_map0.get(pid[i], 0.0)) for i in range(P))
        cost     = pulp.lpSum(buy0[i] * price[0,i] for i in range(P))
        m += cost <= bank0 + proceeds
        m += ft_used0 <= transfers_cnt0
        m += ft_used0 <= ft0
        if exact_transfers is not None:
            m += ft_used0 == min(ft0, int(exact_transfers))

    # ----- Squad composition (persistent) -----
    def _sum_pos_on_squad(pos_name: str):
        return pulp.lpSum(in_squad[i] for i in range(P) if pos_arr[i] == pos_name)

    m += pulp.lpSum(in_squad[i] for i in range(P)) == 15
    m += _sum_pos_on_squad("GK") == 2
    m += _sum_pos_on_squad("DEF") == 5
    m += _sum_pos_on_squad("MID") == 5
    m += _sum_pos_on_squad("FWD") == 3

    # team cap ≤3 or overridden by --max-from-team
    team_caps = {str(code): 3 for code in set(team_code)}
    if max_from_team:
        for k,v in max_from_team.items():
            team_caps[str(k).upper()] = int(v)
    for T, cap in team_caps.items():
        m += pulp.lpSum(in_squad[i] for i in range(P) if team_code[i] == T) <= int(cap)

    # ----- XI/bench per GW -----
    if forced_formation:
        DEF_min, DEF_max = forced_formation[0], forced_formation[0]
        MID_min, MID_max = forced_formation[1], forced_formation[1]
        FWD_min, FWD_max = forced_formation[2], forced_formation[2]
    else:
        DEF_min, DEF_max = 3,5
        MID_min, MID_max = 3,5
        FWD_min, FWD_max = 1,3

    for t in range(G):
        for i in range(P):
            m += start[(t,i)] <= in_squad[i]

        m += pulp.lpSum(start[(t,i)] for i in range(P)) == 11
        m += pulp.lpSum(start[(t,i)] for i in range(P) if pos_arr[i] == "GK") == 1
        m += pulp.lpSum(start[(t,i)] for i in range(P) if pos_arr[i] == "DEF") >= DEF_min
        m += pulp.lpSum(start[(t,i)] for i in range(P) if pos_arr[i] == "DEF") <= DEF_max
        m += pulp.lpSum(start[(t,i)] for i in range(P) if pos_arr[i] == "MID") >= MID_min
        m += pulp.lpSum(start[(t,i)] for i in range(P) if pos_arr[i] == "MID") <= MID_max
        m += pulp.lpSum(start[(t,i)] for i in range(P) if pos_arr[i] == "FWD") >= FWD_min
        m += pulp.lpSum(start[(t,i)] for i in range(P) if pos_arr[i] == "FWD") <= FWD_max

        # Bench: exactly 1 per rank among non-GK outfielders; distinct; not starters
        for r in [1,2,3]:
            m += pulp.lpSum(bench[r][(t,i)] for i in range(P) if pos_arr[i] != "GK") == 1
            for i in range(P):
                if pos_arr[i] != "GK":
                    m += bench[r][(t,i)] <= in_squad[i]
                    m += bench[r][(t,i)] <= 1 - start[(t,i)]
        for i in range(P):
            if pos_arr[i] != "GK":
                m += pulp.lpSum(bench[r][(t,i)] for r in [1,2,3]) <= 1
                m += start[(t,i)] + pulp.lpSum(bench[r][(t,i)] for r in [1,2,3]) <= 1

        # Captains: exactly 1 overall; **no GK captaincy**
        m += pulp.lpSum(cap_tc[(t,i)] for i in range(P)) + pulp.lpSum(cap_n[(t,i)] for i in range(P)) == 1
        for i in range(P):
            m += cap_tc[(t,i)] <= start[(t,i)]
            m += cap_n[(t,i)]  <= start[(t,i)]
            if pos_arr[i] == "GK":
                m += cap_tc[(t,i)] == 0
                m += cap_n[(t,i)]  == 0

        # FH week support (temporary squad/XI)
        is_fh = (chip == "FH" and gw_list[t] == int(chip_gw or -1))
        if is_fh:
            m += pulp.lpSum(tmp_in[(t,i)] for i in range(P)) == 15
            m += pulp.lpSum(tmp_in[(t,i)] for i in range(P) if pos_arr[i] == "GK") == 2
            m += pulp.lpSum(tmp_in[(t,i)] for i in range(P) if pos_arr[i] == "DEF") == 5
            m += pulp.lpSum(tmp_in[(t,i)] for i in range(P) if pos_arr[i] == "MID") == 5
            m += pulp.lpSum(tmp_in[(t,i)] for i in range(P) if pos_arr[i] == "FWD") == 3
            for T, cap in team_caps.items():
                m += pulp.lpSum(tmp_in[(t,i)] for i in range(P) if team_code[i] == T) <= int(cap)

            m += pulp.lpSum(tmp_sta[(t,i)] for i in range(P)) == 11
            m += pulp.lpSum(tmp_sta[(t,i)] for i in range(P) if pos_arr[i] == "GK") == 1
            m += pulp.lpSum(tmp_sta[(t,i)] for i in range(P) if pos_arr[i] == "DEF") >= DEF_min
            m += pulp.lpSum(tmp_sta[(t,i)] for i in range(P) if pos_arr[i] == "DEF") <= DEF_max
            m += pulp.lpSum(tmp_sta[(t,i)] for i in range(P) if pos_arr[i] == "MID") >= MID_min
            m += pulp.lpSum(tmp_sta[(t,i)] for i in range(P) if pos_arr[i] == "MID") <= MID_max
            m += pulp.lpSum(tmp_sta[(t,i)] for i in range(P) if pos_arr[i] == "FWD") >= FWD_min
            m += pulp.lpSum(tmp_sta[(t,i)] for i in range(P) if pos_arr[i] == "FWD") <= FWD_max
            for i in range(P):
                m += tmp_sta[(t,i)] <= tmp_in[(t,i)]

            # FH captains: also forbid GK captaincy
            m += pulp.lpSum(tmp_capt[(t,i)] for i in range(P)) + pulp.lpSum(tmp_capn[(t,i)] for i in range(P)) == 1
            for i in range(P):
                m += tmp_capn[(t,i)] <= tmp_sta[(t,i)]
                m += tmp_capt[(t,i)] <= tmp_sta[(t,i)]
                if pos_arr[i] == "GK":
                    m += tmp_capt[(t,i)] == 0
                    m += tmp_capn[(t,i)] == 0

    # ----- Locks & bans -----
    for pid_locked in lock_ids:
        if pid_locked in pid_index:
            i = pid_index[pid_locked]
            m += in_squad[i] == 1
    for pid_banned in ban_ids:
        if pid_banned in pid_index:
            i = pid_index[pid_banned]
            m += in_squad[i] == 0
            for t in range(G):
                m += tmp_in[(t,i)] == 0

    # ----- Objective -----
    obj = 0
    for t in range(G):
        xi_ev   = pulp.lpSum(start[(t,i)]    * ev[t,i] for i in range(P))
        all15ev = pulp.lpSum(in_squad[i]     * ev[t,i] for i in range(P))
        xi_var  = pulp.lpSum(start[(t,i)]    * var[t,i] for i in range(P))
        upl_n   = pulp.lpSum(cap_n[(t,i)]    * capup[t,i] for i in range(P))
        upl_tc  = pulp.lpSum(cap_tc[(t,i)]   * (2.0 * capup[t,i]) for i in range(P))
        bench_ev = pulp.lpSum(bench[r][(t,i)] * ev[t,i] for r in [1,2,3] for i in range(P))

        is_fh = (chip == "FH" and gw_list[t] == int(chip_gw or -1))
        is_bb = (chip == "BB" and gw_list[t] == int(chip_gw or -1))
        is_tc = (chip == "TC" and gw_list[t] == int(chip_gw or -1))

        if is_fh:
            xi_ev_fh   = pulp.lpSum(tmp_sta[(t,i)]  * ev[t,i] for i in range(P))
            all15ev_fh = pulp.lpSum(tmp_in[(t,i)]   * ev[t,i] for i in range(P))
            xi_var_fh  = pulp.lpSum(tmp_sta[(t,i)]  * var[t,i] for i in range(P))
            upl_n_fh   = pulp.lpSum(tmp_capn[(t,i)] * capup[t,i] for i in range(P))
            upl_tc_fh  = pulp.lpSum(tmp_capt[(t,i)] * (2.0 * capup[t,i]) for i in range(P))
            ev_term = xi_ev_fh if not is_bb else all15ev_fh
            obj += ev_term + (upl_tc_fh if is_tc else upl_n_fh) - float(risk_lambda) * xi_var_fh - EPS_BENCH * bench_ev
        else:
            ev_term = all15ev if is_bb else xi_ev
            obj += ev_term + (upl_tc if is_tc else upl_n) - float(risk_lambda) * xi_var - EPS_BENCH * bench_ev

    if chip != "WC":
        obj -= HIT_COST * hits0

    m += obj

    # ----- Solve -----
    cbc_opts = []
    if mip_gap is not None: cbc_opts += ["-ratio", str(float(mip_gap))]
    if threads is not None: cbc_opts += ["-threads", str(int(threads))]
    solver = pulp.PULP_CBC_CMD(
        msg=bool(verbose),
        timeLimit=int(time_limit) if time_limit is not None else None,
        options=cbc_opts
    )
    res = m.solve(solver)
    status = pulp.LpStatus[res]
    if status not in {"Optimal","Not Solved","Infeasible","Unbounded","Undefined"}:
        raise RuntimeError(f"Unexpected CBC status={status}")
    if status != "Optimal" and pulp.value(m.objective) is None:
        raise RuntimeError(f"MILP not optimal: status={status} (no incumbent)")

    # ----- Extract solution & post-validate XI -----
    def _build_valid_xi_indices(t: int, use_fh: bool) -> List[int]:
        # raw mask
        raw = []
        if use_fh:
            raw = [i for i in range(P) if (pulp.value(tmp_sta[(t,i)]) or 0) > 0.5]
        else:
            raw = [i for i in range(P) if (pulp.value(start[(t,i)]) or 0) > 0.5]

        def _counts(lst):
            d = sum(1 for i in lst if pos_arr[i]=="DEF")
            m_ = sum(1 for i in lst if pos_arr[i]=="MID")
            f = sum(1 for i in lst if pos_arr[i]=="FWD")
            g = sum(1 for i in lst if pos_arr[i]=="GK")
            return g,d,m_,f

        if forced_formation:
            Dmin=Dmax=forced_formation[0]
            Mmin=Mmax=forced_formation[1]
            Fmin=Fmax=forced_formation[2]
        else:
            Dmin,Dmax=3,5; Mmin,Mmax=3,5; Fmin,Fmax=1,3

        gk_cnt,d_cnt,m_cnt,f_cnt = _counts(raw)
        if len(raw)==11 and gk_cnt==1 and Dmin<=d_cnt<=Dmax and Mmin<=m_cnt<=Mmax and Fmin<=f_cnt<=Fmax:
            return raw

        # repair: greedy EV by formation from available pool
        if use_fh:
            avail = [i for i in range(P) if (pulp.value(tmp_in[(t,i)]) or 0) > 0.5]
        else:
            avail = [i for i in range(P) if (pulp.value(in_squad[i]) or 0) > 0.5]
        gks = sorted([i for i in avail if pos_arr[i]=="GK"], key=lambda i: -ev[t,i])
        if not gks: return raw[:11] if len(raw)>=11 else raw
        base = [gks[0]]
        defs = sorted([i for i in avail if pos_arr[i]=="DEF" and i not in base], key=lambda i: -ev[t,i])
        mids = sorted([i for i in avail if pos_arr[i]=="MID" and i not in base], key=lambda i: -ev[t,i])
        fwds = sorted([i for i in avail if pos_arr[i]=="FWD" and i not in base], key=lambda i: -ev[t,i])

        shapes = [forced_formation] if forced_formation else sorted(list(VALID_FORMATIONS), reverse=True)
        best=None
        for D,M,F in shapes:
            if 1+D+M+F != 11: continue
            if len(defs)<D or len(mids)<M or len(fwds)<F: continue
            cand = base + defs[:D] + mids[:M] + fwds[:F]
            val = sum(float(ev[t,i]) for i in cand)
            if best is None or val > best[0]:
                best=(val,cand)
            if forced_formation: break
        return best[1] if best else (raw[:11] if len(raw)>=11 else raw)

    per_gw = []
    xi_blocks = []
    bench_blocks = []
    captain_ids = []

    for t,g in enumerate(gw_list):
        use_fh = (chip=="FH" and g == int(chip_gw or -1))
        xi_idx = _build_valid_xi_indices(t, use_fh)
        xi_pids = [pid[i] for i in xi_idx]
        xi_pids = _order_by_pos_then_ev(xi_pids, t, pid_index, pos_arr, ev)

        d = sum(1 for p in xi_pids if pos_arr[pid_index[p]]=="DEF")
        m_ = sum(1 for p in xi_pids if pos_arr[pid_index[p]]=="MID")
        f = sum(1 for p in xi_pids if pos_arr[pid_index[p]]=="FWD")
        formation = f"{d}-{m_}-{f}"

        if use_fh:
            c_tc = [i for i in range(P) if (pulp.value(tmp_capt[(t,i)]) or 0)>0.5]
            c_n  = [i for i in range(P) if (pulp.value(tmp_capn[(t,i)]) or 0)>0.5]
        else:
            c_tc = [i for i in range(P) if (pulp.value(cap_tc[(t,i)]) or 0)>0.5]
            c_n  = [i for i in range(P) if (pulp.value(cap_n[(t,i)]) or 0)>0.5]
        cap_idx = (c_tc or c_n)[0] if (c_tc or c_n) else None
        captain_ids.append(pid[cap_idx] if cap_idx is not None else None)

        # XI items with extras
        xi_items = []
        for p in xi_pids:
            i = pid_index[p]
            xi_items.append({
                "id": p,
                "name": None if pd.isna(name_arr[i]) else str(name_arr[i]),
                "pos": str(pos_arr[i]),
                "team": str(team_code[i]),
                "xPts": _round1(ev[t,i]),
                "opp": None if opp[t,i] is None else str(opp[t,i]),
                "is_home": bool(ishm[t,i] == 1),
                "venue": "H" if ishm[t,i] == 1 else "A",
                "fdr": (None if pd.isna(fdr[t,i]) else _round1(float(fdr[t,i])))
            })

        # Bench 1..3 + GK 4
        squad_gk = [i for i in range(P) if (pulp.value(in_squad[i]) or 0)>0.5 and pos_arr[i]=="GK"]
        bench_gk_idx = None
        if squad_gk:
            cand = [i for i in squad_gk if i not in xi_idx]
            bench_gk_idx = cand[0] if cand else squad_gk[0]

        bench_out = []
        for r in [1,2,3]:
            picked = None
            for i in range(P):
                if pos_arr[i] != "GK" and (pulp.value(bench[r][(t,i)]) or 0) > 0.5:
                    picked = i; break
            if picked is None:
                nonstar = [i for i in range(P) if pos_arr[i]!="GK" and (pulp.value(in_squad[i]) or 0)>0.5 and i not in xi_idx]
                nonstar = sorted(nonstar, key=lambda i: -ev[t,i])
                if nonstar:
                    picked = nonstar[min(r-1, len(nonstar)-1)]
            if picked is not None:
                bench_out.append((r, picked))

        bench_items = []
        for r,i in bench_out:
            bench_items.append({
                "id": pid[i],
                "name": None if pd.isna(name_arr[i]) else str(name_arr[i]),
                "pos": str(pos_arr[i]),
                "team": str(team_code[i]),
                "bench_order": r,
                "is_bench_gk": False,
                "xPts": _round1(ev[t,i]),
            })
        if bench_gk_idx is not None:
            bench_items.append({
                "id": pid[bench_gk_idx],
                "name": None if pd.isna(name_arr[bench_gk_idx]) else str(name_arr[bench_gk_idx]),
                "pos": "GK",
                "team": str(team_code[bench_gk_idx]),
                "bench_order": 4,
                "is_bench_gk": True,
                "xPts": _round1(ev[t,bench_gk_idx]),
            })

        xi_blocks.append({"gw": g, "xi": xi_items})
        bench_blocks.append({"gw": g, "bench": bench_items})

        chip_tag = None
        if chip in {"FH","TC","BB"} and g == int(chip_gw or -1):
            chip_tag = chip
        if chip == "WC" and g == gw_start:
            chip_tag = "WC"

        is_bb_week = (chip == "BB" and g == int(chip_gw or -1))
        if is_bb_week and not (chip=="FH" and g==int(chip_gw or -1)):
            ev_team = float(sum(ev[t,i] * (pulp.value(in_squad[i]) or 0.0) for i in range(P)))
        elif chip=="FH" and g==int(chip_gw or -1):
            ev_team = float(sum(ev[t,i] * (pulp.value(tmp_sta[(t,i)]) or 0.0) for i in range(P)))
        else:
            ev_team = float(sum(ev[t,i] * (1.0 if pid[i] in [x["id"] for x in xi_items] else 0.0) for i in range(P)))

        per_gw.append({
            "gw": g,
            "formation": formation,
            "xi_count": len(xi_items),
            "captain_id": captain_ids[-1],
            "chip": chip_tag,
            "ev_team": _round1(ev_team),
        })

    buy_ids  = [pid[i] for i in range(P) if (pulp.value(buy0[i]) or 0)>0.5]
    sell_ids = [pid[i] for i in range(P) if (pulp.value(sell0[i]) or 0)>0.5]
    transfers_out: List[dict] = []
    remaining = list(buy_ids)
    for out_id in sell_ids:
        in_id = remaining.pop(0) if remaining else None
        i_out = pid_index[out_id]
        buy_price = float(price[0, pid_index[in_id]]) if in_id else None
        sell_value = float(sell_map0.get(out_id, 0.0))
        pair_net = (buy_price if buy_price is not None else 0.0) - sell_value
        transfers_out.append({
            "gw": gw_start,
            "out": out_id,
            "out_name": None if pd.isna(name_arr[i_out]) else str(name_arr[i_out]),
            "out_pos": str(pos_arr[i_out]),
            "out_team": str(team_code[i_out]),
            "out_xPts": _round1(ev[0, i_out]),
            "in": in_id,
            "in_name": None if (in_id is None or pd.isna(name_arr[pid_index[in_id]])) else str(name_arr[pid_index[in_id]]),
            "in_pos": None if in_id is None else str(pos_arr[pid_index[in_id]]),
            "in_team": None if in_id is None else str(team_code[pid_index[in_id]]),
            "in_xPts": None if in_id is None else _round1(ev[0, pid_index[in_id]]),
            "sell_value": _round1(sell_value),
            "buy_price": _round1(buy_price),
            "pair_net": _round1(pair_net),
        })
    for in_id in remaining:
        i_in = pid_index[in_id]
        transfers_out.append({
            "gw": gw_start,
            "out": None,
            "out_name": None,
            "out_pos": None,
            "out_team": None,
            "out_xPts": None,
            "in": in_id,
            "in_name": None if pd.isna(name_arr[i_in]) else str(name_arr[i_in]),
            "in_pos": str(pos_arr[i_in]),
            "in_team": str(team_code[i_in]),
            "in_xPts": _round1(ev[0, i_in]),
            "sell_value": None,
            "buy_price": _round1(price[0, i_in]),
            "pair_net": _round1(price[0, i_in]),
        })

    total_ev = float(sum(item["ev_team"] or 0.0 for item in per_gw))
    hits_used = 0 if chip == "WC" else int(pulp.value(hits0))
    out = {
        "meta": {
            "season": season,
            "gw_start": gw_start,
            "next_k": next_k,
            "snapshot_gw": snapshot_gw,
            "locks": {"owned": sorted(list(lock_ids)), "ban": sorted(list(ban_ids))},
            "transfer_controls": {
                "free_transfers_before": ft0,
                "allow_hits": bool(allow_hits),
                "max_extra_transfers": int(max_extra_transfers),
                "exact_transfers": (None if exact_transfers is None else int(exact_transfers)),
            },
            "bank_before": _round1(bank0),
        },
        "objective": {
            "per_gw": per_gw,
            "total_ev": _round1(total_ev),
            "hit_cost": _round1(HIT_COST * hits_used),
            "total_minus_hits": _round1(total_ev - HIT_COST * hits_used),
        },
        "xi": xi_blocks,
        "bench": bench_blocks,
        "transfers_used": int(sum(1 for _ in buy_ids)),
        "hits_used": hits_used,
        "transfers": transfers_out,
        "notes": [
            "EV column: exp_pts_mean (xPts accepted upstream).",
            "Transfers only at first GW; squad held across horizon.",
            "XI is exactly 11 and obeys FPL formation bounds.",
            "Bench: outfield ranks 1..3 + bench GK.",
            "GK captaincy is forbidden by constraint.",
        ],
    }
    return out

# ---------- Sweep Runner & CLI ----------
def run_sweep_hold(
    team_state: str,
    optimizer_input: str,
    out_dir: str,
    gw_start: int,
    next_k: int,
    risk_lambda: float,
    topk: Dict[str,int],
    allow_hits: bool,
    max_extra_transfers: int,
    lock_ids: Set[str],
    ban_ids: Set[str],
    max_from_team: Optional[Dict[str,int]],
    formation: Optional[Tuple[int,int,int]],
    sweep_free_transfers: bool,
    sweep_include_hits: bool,
    time_limit: Optional[int],
    mip_gap: Optional[float],
    threads: Optional[int],
    verbose: bool,
    skip_chips: bool = False,
    only_chip: Optional[str] = None,             # "WC" | "FH" | "TC" | "BB"
    only_chip_gws: Optional[List[int]] = None    # applies to FH/TC/BB within horizon
) -> Dict[str,str]:
    with open(team_state, "r", encoding="utf-8") as f:
        state = json.load(f)
    ft0 = int(state.get("free_transfers", 1))

    # Validate --only-chip-gw against horizon BEFORE running
    horizon = list(range(gw_start, gw_start + next_k))
    if only_chip_gws:
        bad = sorted({int(x) for x in only_chip_gws if int(x) not in horizon})
        if bad:
            lo, hi = horizon[0], horizon[-1]
            raise ValueError(f"--only-chip-gw contains out-of-horizon GW(s): {bad}. "
                             f"Horizon is [{lo}..{hi}] (gw={gw_start}, next_k={next_k}).")

    ks: List[int] = []
    if sweep_free_transfers:
        ks.extend(range(1, max(1, ft0) + 1))
    else:
        ks.append(min(1, max(1, ft0)))
    if sweep_include_hits and allow_hits:
        ks.extend(range(ft0+1, ft0 + max(1, max_extra_transfers) + 1))

    base_root = os.path.join(out_dir, "hold", f"gw{gw_start}_k{next_k}", "base")
    chips_root = os.path.join(out_dir, "hold", f"gw{gw_start}_k{next_k}", "chips")
    os.makedirs(base_root, exist_ok=True)
    os.makedirs(chips_root, exist_ok=True)

    written: Dict[str,str] = {}

    # BASE (no chip)
    for K in ks:
        plan = solve_hold_horizon(
            team_state_path=team_state,
            optimizer_input_path=optimizer_input,
            gw_start=gw_start,
            next_k=next_k,
            risk_lambda=risk_lambda,
            topk=topk,
            allow_hits=allow_hits,
            max_extra_transfers=max_extra_transfers,
            exact_transfers=K,
            lock_ids=lock_ids,
            ban_ids=ban_ids,
            max_from_team=max_from_team,
            forced_formation=formation,
            chip=None,
            chip_gw=None,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            verbose=verbose,
        )
        k_dir = os.path.join(base_root, f"K{K}")
        os.makedirs(k_dir, exist_ok=True)
        with open(os.path.join(k_dir, "plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        with open(os.path.join(k_dir, "transfers.json"), "w", encoding="utf-8") as f:
            json.dump(plan["transfers"], f, indent=2, ensure_ascii=False)
        written[f"BASE_K{K}"] = os.path.join(k_dir, "plan.json")

    # CHIPS
    if not skip_chips:
        chips = ["WC","FH","TC","BB"] if not only_chip else [only_chip]
        for chip in chips:
            if chip == "WC":
                for K in ks:
                    plan = solve_hold_horizon(
                        team_state_path=team_state, optimizer_input_path=optimizer_input,
                        gw_start=gw_start, next_k=next_k,
                        risk_lambda=risk_lambda, topk=topk,
                        allow_hits=allow_hits, max_extra_transfers=max_extra_transfers,
                        exact_transfers=K, lock_ids=lock_ids, ban_ids=ban_ids,
                        max_from_team=max_from_team, forced_formation=formation,
                        chip="WC", chip_gw=gw_start,
                        time_limit=time_limit, mip_gap=mip_gap, threads=threads, verbose=verbose,
                    )
                    chip_dir = os.path.join(chips_root, "WC", f"gw{gw_start}", f"K{K}")
                    os.makedirs(chip_dir, exist_ok=True)
                    with open(os.path.join(chip_dir, "plan.json"), "w", encoding="utf-8") as f:
                        json.dump(plan, f, indent=2, ensure_ascii=False)
                    with open(os.path.join(chip_dir, "transfers.json"), "w", encoding="utf-8") as f:
                        json.dump(plan["transfers"], f, indent=2, ensure_ascii=False)
                    written[f"WC_K{K}"] = os.path.join(chip_dir, "plan.json")
            else:
                gw_range = horizon if not only_chip_gws else [g for g in horizon if g in set(only_chip_gws)]
                for g in gw_range:
                    for K in ks:
                        plan = solve_hold_horizon(
                            team_state_path=team_state, optimizer_input_path=optimizer_input,
                            gw_start=gw_start, next_k=next_k,
                            risk_lambda=risk_lambda, topk=topk,
                            allow_hits=allow_hits, max_extra_transfers=max_extra_transfers,
                            exact_transfers=K, lock_ids=lock_ids, ban_ids=ban_ids,
                            max_from_team=max_from_team, forced_formation=formation,
                            chip=chip, chip_gw=g,
                            time_limit=time_limit, mip_gap=mip_gap, threads=threads, verbose=verbose,
                        )
                        chip_dir = os.path.join(chips_root, chip, f"gw{g}", f"K{K}")
                        os.makedirs(chip_dir, exist_ok=True)
                        with open(os.path.join(chip_dir, "plan.json"), "w", encoding="utf-8") as f:
                            json.dump(plan, f, indent=2, ensure_ascii=False)
                        with open(os.path.join(chip_dir, "transfers.json"), "w", encoding="utf-8") as f:
                            json.dump(plan["transfers"], f, indent=2, ensure_ascii=False)
                        written[f"{chip}_GW{g}_K{K}"] = os.path.join(chip_dir, "plan.json")

    return written

def _parse_csv_ids(s: Optional[str]) -> Set[str]:
    if not s: return set()
    return {x.strip() for x in s.split(",") if x.strip()}

def _parse_team_caps(s: Optional[str]) -> Dict[str,int]:
    out: Dict[str,int] = {}
    if not s: return out
    for part in s.split(","):
        if not part.strip(): continue
        code, num = part.split(":",1)
        out[code.strip().upper()] = int(num.strip())
    return out

def main():
    ap = argparse.ArgumentParser(description="Multi-GW HOLD optimizer (single-GW transfers, hold thereafter)")
    ap.add_argument("--team-state", required=True)
    ap.add_argument("--optimizer-input", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--gw", type=int, required=True, help="Start GW (transfers happen here)")
    ap.add_argument("--next-k", type=int, required=True, help="Horizon length in GWs (including start GW)")

    ap.add_argument("--risk-lambda", type=float, default=0.0)
    ap.add_argument("--topk", default="GK:10,DEF:25,MID:25,FWD:20")

    ap.add_argument("--allow-hits", action="store_true")
    ap.add_argument("--max-extra-transfers", type=int, default=3)

    ap.add_argument("--lock", help="Comma-separated player_ids to force in final squad (persisted)")
    ap.add_argument("--ban", help="Comma-separated player_ids to forbid in persistent squad (also FH temp)")
    ap.add_argument("--max-from-team", dest="max_from_team", help='Per-team caps like "ARS:2,MCI:2" (defaults to 3 for others)')

    ap.add_argument("--formation", help="Exact XI formation to enforce each GW, e.g. 3-5-2")

    # Sweep flags
    ap.add_argument("--sweep-free-transfers", action="store_true", help="Produce K=1..FT")
    ap.add_argument("--sweep-include-hits", action="store_true", help="Also produce K=FT+1..FT+max_extra_transfers (requires --allow-hits)")

    # Chip control
    ap.add_argument("--skip-chips", action="store_true", help="BASE only")
    ap.add_argument("--only-chip", choices=["WC","FH","TC","BB"], help="Run only this chip variant")
    ap.add_argument("--only-chip-gw", help="Comma-separated GW(s) for FH/TC/BB (must be inside horizon)")

    # Runtime
    ap.add_argument("--time-limit", type=int, default=None)
    ap.add_argument("--mip-gap", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    formation = _formation_from_cli(args.formation) if args.formation else None
    only_chip_gws = None
    if args.only_chip_gw:
        only_chip_gws = [int(x.strip()) for x in args.only_chip_gw.split(",") if x.strip()]

    written = run_sweep_hold(
        team_state=args.team_state,
        optimizer_input=args.optimizer_input,
        out_dir=args.out_dir,
        gw_start=int(args.gw),
        next_k=int(args.next_k),
        risk_lambda=float(args.risk_lambda),
        topk=_parse_topk(args.topk),
        allow_hits=bool(args.allow_hits),
        max_extra_transfers=int(args.max_extra_transfers),
        lock_ids=_parse_csv_ids(args.lock),
        ban_ids=_parse_csv_ids(args.ban),
        max_from_team=_parse_team_caps(args.max_from_team),
        formation=formation,
        sweep_free_transfers=bool(args.sweep_free_transfers),
        sweep_include_hits=bool(args.sweep_include_hits),
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        verbose=bool(args.verbose),
        skip_chips=bool(args.skip_chips),
        only_chip=(args.only_chip or None),
        only_chip_gws=only_chip_gws,
    )
    print(json.dumps(written, indent=2))

if __name__ == "__main__":
    main()
