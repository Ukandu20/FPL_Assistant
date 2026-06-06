#!/usr/bin/env python3
"""
Multi-GW selector (MILP) — transfers + XI + C/VC + per-GW bank, FT carryover, and chip exclusivity.
Extended with:
- --max-extra-transfers: cap hits per GW by FT_g + max_extra_transfers (if hits allowed)
- --lock/--ban: force-include / force-exclude player_ids across the horizon
- --gw-col: choose GW column name in optimizer input
- --sweep-chips / --sweep-free-transfers / --sweep-include-hits: robust sweeps with infeasible-skip
- force_chip_at: allow forcing a chip to a specific GW

Supported chips:
- TC: captain uplift scales by (tc_multiplier - 1) on that GW only.
- BB: EV counts all 15 that GW (no double-count with starters).
- WC: unlimited permanent transfers that GW, hits=0, budget respected.
- FH: NOT modeled (temporary team), still rejected.

FT carryover:
- Modeled exactly for FPL’s {1,2}: roll (0 transfers) ⇒ next week 2; else ⇒ 1.

Author: You (cloned/adapted and extended)
"""
from __future__ import annotations

import argparse, json, os, sys
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd

try:
    import pulp  # type: ignore
except Exception:
    raise SystemExit("pulp is required (pip install pulp).")

# ---------------- config ----------------
TEAM_COL_CANDIDATES = ("team", "team_quota_key")
MAX_FT_CARRY = 2            # True FPL cap for free-transfers carryover
HIT_COST = 4.0
BENCH_RANKS = (1, 2, 3)
FDR_COLS = ("fdr",)

# ---------------- typed error ----------------
class MILPStatusError(RuntimeError):
    def __init__(self, status: str):
        self.status = status
        super().__init__(f"MILP not optimal: status={status}")

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
    out = {"GK": 20, "DEF": 60, "MID": 60, "FWD": 40}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.strip().split(":")
        out[k.strip().upper()] = int(v)
    return out

def _parse_id_list(arg: Optional[str]) -> Set[str]:
    if not arg:
        return set()
    parts = []
    for chunk in arg.replace(";", ",").split(","):
        s = chunk.strip()
        if s:
            parts.append(s)
    return set(parts)

def _parse_bool_list(arg: Optional[str]) -> List[bool]:
    if not arg:
        return [True]
    out = []
    for token in arg.replace(";", ",").split(","):
        s = token.strip().lower()
        if s in {"true","t","1","yes","y"}:
            out.append(True)
        elif s in {"false","f","0","no","n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean in list: {token}")
    return out

def _parse_int_list(arg: Optional[str]) -> List[int]:
    if not arg:
        return []
    out = []
    for token in arg.replace(";", ",").split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out

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

def _is_na_like(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)
    except Exception:
        return x is None

def _to_str(x) -> str:
    return "" if _is_na_like(x) else str(x)

def _first_nonempty_str(*vals) -> str:
    for v in vals:
        s = _to_str(v).strip()
        if s:
            return s
    return ""

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

def _ev_column(df: pd.DataFrame) -> str:
    for c in ["exp_pts_mean", "xPts", "xpts", "exp_points"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any() and (s.fillna(0).abs().sum() > 0):
                return c
    df["__ev__"] = 0.0
    return "__ev__"

def _prepare_pool_union(df: pd.DataFrame, state_owned_ids: Set[str], topk: Dict[str,int]) -> pd.DataFrame:
    team_col = _get_team_col_name(df)
    ev_col = _ev_column(df)
    df = df.copy()
    df["owned"] = df["player_id"].astype(str).isin(state_owned_ids)
    df[team_col] = df[team_col].astype(str).str.strip().str.upper()
    keep: List[pd.DataFrame] = []
    for pos_name, g in df.groupby("pos", as_index=False):
        g = g.sort_values(["owned", ev_col], ascending=[False, False])
        k = topk.get(pos_name, 50)
        keep.append(pd.concat([g[g["owned"]], g[~g["owned"]].head(k)], ignore_index=True))
    pool = (pd.concat(keep, ignore_index=True)
              .sort_values(["owned", ev_col], ascending=[False, False])
              .drop_duplicates("player_id")
              .reset_index(drop=True))
    return pool

def _to_bool_or_none(v):
    if _is_na_like(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "1", "h", "home"}:
            return True
        if s in {"false", "f", "0", "a", "away"}:
            return False
    return None

def _player_obj(idx: int, pid: List[str], names: List[Optional[str]], pos: List[str],
                teams: List[str], ev: np.ndarray,
                opp_map_t: Dict[int, Any], is_home_map_t: Dict[int, Any],
                venue_map_t: Dict[int, Any], fdr_map_t: Optional[Dict[int, Any]]) -> dict:
    out = {
        "id": pid[idx],
        "name": (None if names[idx] is None else names[idx]),
        "pos": pos[idx],
        "team": teams[idx],
        "xPts": _round1(float(ev[idx])),
        "opp": opp_map_t.get(idx),
        "is_home": is_home_map_t.get(idx),
        "venue": venue_map_t.get(idx),
    }
    if fdr_map_t is not None:
        out["fdr"] = fdr_map_t.get(idx)
    return out

# ---------------- core MILP ----------------
def solve_multi_gw(
    team_state_path: str,
    optimizer_input_path: str,
    out_path: str,
    gw_start: int,
    next_k: int,
    risk_lambda: float = 0.0,
    tc_multiplier: float = 3.0,
    topk: Optional[Dict[str, int]] = None,
    allow_hits: bool = True,
    cap_cannot_equal_vice: bool = True,
    formation_bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((3, 5), (2, 5), (1, 3)),
    forbid_fh: bool = True,
    verbose: bool = False,
    # new:
    max_extra_transfers: int = 2,
    lock_ids: Optional[Set[str]] = None,
    ban_ids: Optional[Set[str]] = None,
    gw_col: str = "gw",
    force_chip_at: Optional[Dict[str, Optional[int]]] = None,  # e.g. {"TC": 6, "BB": None, "WC": None}
    ft0_override: Optional[int] = None,
) -> dict:
    # ---- load state & data
    with open(team_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    season = str(state.get("season"))
    snapshot_gw = int(state.get("gw"))
    bank0 = float(state.get("bank", 0.0))
    ft0 = int(state.get("free_transfers", 1))
    ft0 = 1 if ft0 <= 0 else min(MAX_FT_CARRY, ft0)
    if ft0_override is not None:
        ft0 = int(ft0_override)
        ft0 = 1 if ft0 <= 0 else min(MAX_FT_CARRY, ft0)

    owned0: Set[str] = {str(p["player_id"]) for p in state.get("squad", [])}
    owned_sell0: Dict[str, float] = {
        str(p["player_id"]): float(p.get("sell_price", p.get("price", 0.0)))
        for p in state.get("squad", [])
    }

    df_all = _read_any(optimizer_input_path)
    if gw_col not in df_all.columns:
        raise ValueError(f"optimizer_input missing GW column '{gw_col}'")
    team_col = _get_team_col_name(df_all)
    _validate_teams(df_all)
    ev_col = _ev_column(df_all)

    horizon_gws = list(range(int(gw_start), int(gw_start)+int(next_k)))
    df = df_all[df_all[gw_col].isin(horizon_gws)].copy()
    if df.empty:
        raise ValueError(f"No optimizer_input rows for {gw_col} in {horizon_gws}")

    # pool union
    pool = _prepare_pool_union(df, owned0, topk or {"GK":20,"DEF":60,"MID":60,"FWD":40})
    pid = pool["player_id"].astype(str).tolist()
    pos = pool["pos"].astype(str).tolist()
    teams = pool[team_col].astype(str).tolist()
    N = len(pool)

    # static metadata per player
    names: List[Optional[str]] = [None] * N
    if "player" in pool.columns:
        names = [None if pd.isna(v) else str(v) for v in pool["player"].astype(object)]

    # build per-GW value maps (default 0 when missing)
    def _gw_map(col: str, fill: float = 0.0) -> Dict[int, Dict[int, float]]:
        gmap: Dict[int, Dict[int, float]] = {}
        src = df.merge(pool[["player_id"]], on="player_id", how="right")
        for g in horizon_gws:
            gg = src[src[gw_col] == g]
            arr = pd.to_numeric(gg.set_index("player_id")[col], errors="coerce").reindex(pid).fillna(fill).astype(float).to_numpy()
            gmap[g] = {i: float(arr[i]) for i in range(N)}
        return gmap

    EV = _gw_map(ev_col, 0.0)
    VAR = _gw_map("exp_pts_var", 0.0)
    CAPUP = _gw_map("captain_uplift", 0.0)
    PRICE = _gw_map("price", 0.0)
    SELLVAL = _gw_map("sell_price", 0.0)

    def _gw_plain(col: str) -> Dict[int, Dict[int, Any]]:
        out: Dict[int, Dict[int, Any]] = {}
        src = df.merge(pool[["player_id"]], on="player_id", how="right")
        for g in horizon_gws:
            gg = src[src[gw_col] == g].set_index("player_id")
            vals = gg[col] if col in gg.columns else pd.Series([None]*len(pid), index=pid)
            out[g] = {i: (None if pd.isna(vals.reindex(pid).iloc[i]) else vals.reindex(pid).iloc[i]) for i in range(N)}
        return out

    OPP = _gw_plain("opponent")
    IS_HOME = {g: {i: _to_bool_or_none(_gw_plain("is_home")[g][i]) for i in range(N)} for g in horizon_gws}
    VENUE = {g: {i: ("H" if IS_HOME[g][i] is True else ("A" if IS_HOME[g][i] is False else None)) for i in range(N)} for g in horizon_gws}

    fdr_map: Optional[Dict[int, Dict[int, Any]]] = None
    fdr_col = next((c for c in FDR_COLS if c in df.columns), None)
    if fdr_col:
        FDR = _gw_plain(fdr_col)
        fdr_map = FDR

    owned0_vec = np.array([pid[i] in owned0 for i in range(N)], dtype=int)

    lock_ids = set(lock_ids or set())
    ban_ids = set(ban_ids or set())
    inter = lock_ids & ban_ids
    if inter:
        raise ValueError(f"player_ids appear in both --lock and --ban: {sorted(inter)}")

    m = pulp.LpProblem("multi_gw_selector", pulp.LpMaximize)

    x = {(i, g): pulp.LpVariable(f"x_{i}_{g}", 0, 1, cat=pulp.LpBinary) for i in range(N) for g in horizon_gws}
    s = {(i, g): pulp.LpVariable(f"s_{i}_{g}", 0, 1, cat=pulp.LpBinary) for i in range(N) for g in horizon_gws}
    c = {(i, g): pulp.LpVariable(f"c_{i}_{g}", 0, 1, cat=pulp.LpBinary) for i in range(N) for g in horizon_gws}
    vc = {(i, g): pulp.LpVariable(f"vc_{i}_{g}", 0, 1, cat=pulp.LpBinary) for i in range(N) for g in horizon_gws}
    inb = {(i, g): pulp.LpVariable(f"in_{i}_{g}", 0, 1, cat=pulp.LpBinary) for i in range(N) for g in horizon_gws}
    outb = {(i, g): pulp.LpVariable(f"out_{i}_{g}", 0, 1, cat=pulp.LpBinary) for i in range(N) for g in horizon_gws}
    bench = {(r, i, g): pulp.LpVariable(f"bench_r{r}_{i}_{g}", 0, 1, cat=pulp.LpBinary)
             for r in BENCH_RANKS for i in range(N) for g in horizon_gws}
    u = {g: pulp.LpVariable(f"u_{g}", lowBound=0, upBound=15, cat=pulp.LpInteger) for g in horizon_gws}
    hits = {g: pulp.LpVariable(f"hits_{g}", lowBound=0, upBound=15, cat=pulp.LpInteger) for g in horizon_gws}
    bank = {g: pulp.LpVariable(f"bank_{g}", lowBound=0.0, cat=pulp.LpContinuous) for g in horizon_gws}
    ft = {g: pulp.LpVariable(f"ft_{g}", lowBound=1, upBound=MAX_FT_CARRY, cat=pulp.LpInteger) for g in horizon_gws}
    roll = {g: pulp.LpVariable(f"roll_{g}", 0, 1, cat=pulp.LpBinary) for g in horizon_gws}
    tc = {g: pulp.LpVariable(f"TC_{g}", 0, 1, cat=pulp.LpBinary) for g in horizon_gws}
    bb = {g: pulp.LpVariable(f"BB_{g}", 0, 1, cat=pulp.LpBinary) for g in horizon_gws}
    wc = {g: pulp.LpVariable(f"WC_{g}", 0, 1, cat=pulp.LpBinary) for g in horizon_gws}
    yx = {(i, g): pulp.LpVariable(f"yx_{i}_{g}", 0, 1, cat=pulp.LpBinary) for i in range(N) for g in horizon_gws}
    ys = {(i, g): pulp.LpVariable(f"ys_{i}_{g}", 0, 1, cat=pulp.LpBinary) for i in range(N) for g in horizon_gws}
    wtc = {(i, g): pulp.LpVariable(f"wtc_{i}_{g}", 0, 1, cat=pulp.LpBinary) for i in range(N) for g in horizon_gws}

    g0 = horizon_gws[0]
    m += ft[g0] == ft0, "ft_init"
    m += bank[g0] >= 0

    for i in range(N):
        m += x[i, g0] == owned0_vec[i] + inb[i, g0] - outb[i, g0]
        m += inb[i, g0] <= 1 - owned0_vec[i]
        m += outb[i, g0] <= owned0_vec[i]

    for g_prev, g in zip(horizon_gws[:-1], horizon_gws[1:]):
        for i in range(N):
            m += x[i, g] == x[i, g_prev] + inb[i, g] - outb[i, g]
            m += inb[i, g] <= 1 - x[i, g_prev]
            m += outb[i, g] <= x[i, g_prev]

    for g in horizon_gws:
        m += u[g] == pulp.lpSum(inb[i, g] for i in range(N)), f"u_count_{g}"

    id_to_idx = {pid[i]: i for i in range(N)}
    for lid in lock_ids:
        if lid in id_to_idx:
            i = id_to_idx[lid]
            for g in horizon_gws:
                m += x[i, g] == 1, f"lock_{lid}_{g}"
        else:
            raise ValueError(f"--lock player_id not in candidate pool: {lid}")
    for bid in ban_ids:
        if bid in id_to_idx:
            i = id_to_idx[bid]
            for g in horizon_gws:
                m += x[i, g] == 0, f"ban_{bid}_{g}"

    def _sum_pos(pname: str, vec, g):
        return pulp.lpSum(vec[i, g] for i in range(N) if pos[i] == pname)

    for g in horizon_gws:
        m += pulp.lpSum(x[i, g] for i in range(N)) == 15, f"squad15_{g}"
        m += _sum_pos("GK", x, g) == 2, f"comp_gk_2_{g}"
        m += _sum_pos("DEF", x, g) == 5, f"comp_def_5_{g}"
        m += _sum_pos("MID", x, g) == 5, f"comp_mid_5_{g}"
        m += _sum_pos("FWD", x, g) == 3, f"comp_fwd_3_{g}"
        for tcode in sorted(set(teams)):
            m += pulp.lpSum(x[i, g] for i in range(N) if teams[i] == tcode) <= 3, f"teamcap_{tcode}_{g}"

        m += pulp.lpSum(s[i, g] for i in range(N)) == 11, f"xi11_{g}"
        for i in range(N):
            m += s[i, g] <= x[i, g], f"start_in_squad_{i}_{g}"

        (DEF_min, DEF_max), (MID_min, MID_max), (FWD_min, FWD_max) = ((3,5),(2,5),(1,3))
        m += _sum_pos("GK", s, g) == 1, f"xi_gk1_{g}"
        m += _sum_pos("DEF", s, g) >= DEF_min
        m += _sum_pos("DEF", s, g) <= DEF_max
        m += _sum_pos("MID", s, g) >= MID_min
        m += _sum_pos("MID", s, g) <= MID_max
        m += _sum_pos("FWD", s, g) >= FWD_min
        m += _sum_pos("FWD", s, g) <= FWD_max

        m += pulp.lpSum(c[i, g] for i in range(N)) == 1, f"one_c_{g}"
        m += pulp.lpSum(vc[i, g] for i in range(N)) == 1, f"one_vc_{g}"
        for i in range(N):
            m += c[i, g] <= s[i, g]
            m += vc[i, g] <= s[i, g]
            if pos[i] == "GK":
                m += c[i, g] == 0, f"nogk_cap_{i}_{g}"
            m += c[i, g] + vc[i, g] <= 1

        for r in BENCH_RANKS:
            m += pulp.lpSum(bench[r, i, g] for i in range(N) if pos[i] != "GK") == 1, f"bench_r{r}_one_{g}"
        for i in range(N):
            if pos[i] == "GK":
                for r in BENCH_RANKS:
                    m += bench[r, i, g] == 0
            else:
                m += pulp.lpSum(bench[r, i, g] for r in BENCH_RANKS) <= 1
                for r in BENCH_RANKS:
                    m += bench[r, i, g] <= x[i, g] - s[i, g]

        m += tc[g] + bb[g] + wc[g] <= 1, f"chip_exclusive_{g}"

    m += pulp.lpSum(tc[g] for g in horizon_gws) <= 1, "tc_once"
    m += pulp.lpSum(bb[g] for g in horizon_gws) <= 1, "bb_once"
    m += pulp.lpSum(wc[g] for g in horizon_gws) <= 1, "wc_once"

    if force_chip_at:
        if force_chip_at.get("TC") is not None:
            g = int(force_chip_at["TC"])
            if g not in horizon_gws: raise ValueError("force_chip_at.TC not in horizon")
            for gg in horizon_gws:
                m += tc[gg] == (1 if gg == g else 0)
        if force_chip_at.get("BB") is not None:
            g = int(force_chip_at["BB"])
            if g not in horizon_gws: raise ValueError("force_chip_at.BB not in horizon")
            for gg in horizon_gws:
                m += bb[gg] == (1 if gg == g else 0)
        if force_chip_at.get("WC") is not None:
            g = int(force_chip_at["WC"])
            if g not in horizon_gws: raise ValueError("force_chip_at.WC not in horizon")
            for gg in horizon_gws:
                m += wc[gg] == (1 if gg == g else 0)

    BIG_M = 20

    def buys_cost(g): return pulp.lpSum(inb[i, g] * PRICE[g].get(i, 0.0) for i in range(N))
    def sells_proceeds(g): return pulp.lpSum(outb[i, g] * SELLVAL[g].get(i, 0.0) for i in range(N))

    m += buys_cost(g0) <= bank0 + sells_proceeds(g0), f"budget_{g0}"
    m += bank[g0] == bank0 + sells_proceeds(g0) - buys_cost(g0), f"bank_flow_{g0}"

    if allow_hits:
        m += hits[g0] >= u[g0] - ft[g0]
        if max_extra_transfers is not None:
            m += u[g0] <= ft[g0] + max(0, int(max_extra_transfers))
    else:
        m += u[g0] <= ft[g0]; m += hits[g0] == 0

    m += u[g0] <= BIG_M * (1 - roll[g0])
    m += u[g0] >= 1 - BIG_M * roll[g0]

    for g_prev, g in zip(horizon_gws[:-1], horizon_gws[1:]):
        m += ft[g] == 1 + roll[g_prev], f"ft_roll_{g_prev}_to_{g}"
        m += buys_cost(g) <= bank[g_prev] + sells_proceeds(g), f"budget_{g}"
        m += bank[g] == bank[g_prev] + sells_proceeds(g) - buys_cost(g), f"bank_flow_{g}"

        if allow_hits:
            m += hits[g] >= u[g] - ft[g]
            if max_extra_transfers is not None:
                m += u[g] <= ft[g] + max(0, int(max_extra_transfers))
        else:
            m += u[g] <= ft[g]; m += hits[g] == 0

        m += u[g] <= BIG_M * (1 - roll[g])
        m += u[g] >= 1 - BIG_M * roll[g]

    for g in horizon_gws:
        m += hits[g] <= BIG_M * (1 - wc[g]), f"wc_hits_zero_{g}"

    for g in horizon_gws:
        for i in range(N):
            m += yx[i, g] <= x[i, g]
            m += yx[i, g] <= bb[g]
            m += yx[i, g] >= x[i, g] + bb[g] - 1
            m += ys[i, g] <= s[i, g]
            m += ys[i, g] <= bb[g]
            m += ys[i, g] >= s[i, g] + bb[g] - 1
            m += wtc[i, g] <= c[i, g]
            m += wtc[i, g] <= tc[g]
            m += wtc[i, g] >= c[i, g] + tc[g] - 1

    def EV_XI(g): return pulp.lpSum(s[i, g] * EV[g].get(i, 0.0) for i in range(N))
    def EV_BB(g): return pulp.lpSum(yx[i, g] * EV[g].get(i, 0.0) for i in range(N)) - \
                         pulp.lpSum(ys[i, g] * EV[g].get(i, 0.0) for i in range(N))
    def UPLIFT(g): return pulp.lpSum(c[i, g] * CAPUP[g].get(i, 0.0) for i in range(N))
    extra_tc = max(0.0, float(tc_multiplier) - 1.0)
    def VAR_XI(g): return pulp.lpSum(s[i, g] * VAR[g].get(i, 0.0) for i in range(N))

    terms = []
    for g in horizon_gws:
        t_ev = EV_XI(g)
        t_bb = EV_BB(g)
        t_cap = UPLIFT(g)
        t_tc_extra = extra_tc * pulp.lpSum(wtc[i, g] * CAPUP[g].get(i, 0.0) for i in range(N))
        t_var = float(risk_lambda) * VAR_XI(g)
        t_hits = HIT_COST * hits[g]
        gw_total = t_ev + t_bb + t_cap + t_tc_extra - t_var - t_hits
        terms.append(gw_total)
    total_obj = pulp.lpSum(terms)
    m += total_obj

    solver = pulp.PULP_CBC_CMD(msg=bool(verbose))
    res = m.solve(solver)
    status = pulp.LpStatus[res]
    if status != "Optimal":
        raise MILPStatusError(status)

    def picks(mask, g):
        return [pid[i] for i in range(N) if pulp.value(mask[i, g]) > 0.5]

    def argmax_idx(mask, g):
        I = [i for i in range(N) if pulp.value(mask[i, g]) > 0.5]
        return I[0] if I else None

    per_gw = []
    chip_schedule = {"TC": None, "BB": None, "WC": None, "FH": None}
    for g in horizon_gws:
        xi_ids = picks(s, g)
        cap_idx = argmax_idx(c, g)
        vcap_idx = argmax_idx(vc, g)
        cap_pid = (pid[cap_idx] if cap_idx is not None else None)
        vcap_pid = (pid[vcap_idx] if vcap_idx is not None else None)

        outfield = [i for i in range(N) if pos[i] != "GK"]
        bench_order_ids: List[Optional[str]] = [None, None, None]
        for r in BENCH_RANKS:
            for i in outfield:
                if pulp.value(bench[r, i, g]) > 0.5:
                    bench_order_ids[r-1] = pid[i]; break

        gk_ids = [i for i in range(N) if pos[i] == "GK" and pulp.value(x[i, g]) > 0.5]
        gk_start = [i for i in range(N) if pos[i] == "GK" and pulp.value(s[i, g]) > 0.5]
        gk_bench_idx = next((i for i in gk_ids if i not in gk_start), None)

        buy_ids = picks(inb, g)
        sell_ids = picks(outb, g)
        remaining_buys = list(buy_ids)
        transfers_out: List[dict] = []
        for out_id in sell_ids:
            in_id = (remaining_buys.pop(0) if remaining_buys else None)
            out_idx = pid.index(out_id)
            in_idx = (pid.index(in_id) if in_id else None)
            buy_price = PRICE[g].get(in_idx, 0.0) if in_idx is not None else None
            sell_value = SELLVAL[g].get(out_idx, 0.0)
            pair_net = (0.0 if buy_price is None else buy_price) - sell_value
            payload = {
                "out": out_id, "out_name": names[out_idx], "out_pos": pos[out_idx],
                "out_team": teams[out_idx], "out_xPts": _round1(EV[g].get(out_idx, 0.0)),
                "out_opp": OPP[g].get(out_idx), "out_is_home": IS_HOME[g].get(out_idx),
                "out_venue": VENUE[g].get(out_idx),
                "in": (in_id if in_id else None),
                "in_name": (names[in_idx] if in_idx is not None else None),
                "in_pos": (pos[in_idx] if in_idx is not None else None),
                "in_team": (teams[in_idx] if in_idx is not None else None),
                "in_xPts": (_round1(EV[g].get(in_idx, 0.0)) if in_idx is not None else None),
                "in_opp": (OPP[g].get(in_idx) if in_idx is not None else None),
                "in_is_home": (IS_HOME[g].get(in_idx) if in_idx is not None else None),
                "in_venue": (VENUE[g].get(in_idx) if in_idx is not None else None),
                "sell_value": float(sell_value),
                "buy_price": (float(buy_price) if buy_price is not None else None),
                "pair_net": _round1(pair_net),
                "price_delta": float(0.0 - sell_value),
            }
            if fdr_map is not None:
                payload["out_fdr"] = fdr_map[g].get(out_idx)
                payload["in_fdr"] = (fdr_map[g].get(in_idx) if in_idx is not None else None)
            transfers_out.append(payload)
        for in_id in remaining_buys:
            in_idx = pid.index(in_id)
            buy_price = PRICE[g].get(in_idx, 0.0)
            payload = {
                "out": None, "out_name": None, "out_pos": None, "out_team": None, "out_xPts": None,
                "out_opp": None, "out_is_home": None, "out_venue": None,
                "in": in_id, "in_name": names[in_idx], "in_pos": pos[in_idx],
                "in_team": teams[in_idx], "in_xPts": _round1(EV[g].get(in_idx, 0.0)),
                "in_opp": OPP[g].get(in_idx), "in_is_home": IS_HOME[g].get(in_idx), "in_venue": VENUE[g].get(in_idx),
                "sell_value": None, "buy_price": float(buy_price),
                "pair_net": _round1(float(buy_price)), "price_delta": float(buy_price),
            }
            if fdr_map is not None:
                payload["out_fdr"] = None
                payload["in_fdr"] = fdr_map[g].get(in_idx)
            transfers_out.append(payload)

        ev_xi = float(sum(EV[g].get(i, 0.0) * pulp.value(s[i, g]) for i in range(N)))
        ev_bb_extra = float(sum(EV[g].get(i, 0.0) * pulp.value(yx[i, g]) for i in range(N))
                            - sum(EV[g].get(i, 0.0) * pulp.value(ys[i, g]) for i in range(N)))
        uplift = float(sum(CAPUP[g].get(i, 0.0) * pulp.value(c[i, g]) for i in range(N)))
        tc_extra = float(max(0.0, tc_multiplier - 1.0) * sum(CAPUP[g].get(i, 0.0) * pulp.value(wtc[i, g]) for i in range(N)))
        var_pen = float(risk_lambda * sum(VAR[g].get(i, 0.0) * pulp.value(s[i, g]) for i in range(N)))
        hits_val = float(HIT_COST * pulp.value(hits[g]))
        total = ev_xi + ev_bb_extra + uplift + tc_extra - var_pen - hits_val

        nDEF = int(sum(1 for i in range(N) if pos[i] == "DEF" and pulp.value(s[i, g]) > 0.5))
        nMID = int(sum(1 for i in range(N) if pos[i] == "MID" and pulp.value(s[i, g]) > 0.5))
        nFWD = int(sum(1 for i in range(N) if pos[i] == "FWD" and pulp.value(s[i, g]) > 0.5))
        formation_str = f"{nDEF}-{nMID}-{nFWD}"

        chip_here = ("TC" if pulp.value(tc[g]) > 0.5 else ("BB" if pulp.value(bb[g]) > 0.5 else ("WC" if pulp.value(wc[g]) > 0.5 else None)))
        if chip_here:
            chip_schedule[chip_here] = g

        per_gw.append({
            "gw": int(g),
            "formation": formation_str,
            "xi_count": 11,
            "captain_id": cap_pid,
            "vice_id": vcap_pid,
            "chip": chip_here,
            "ev_team": _round1(ev_xi + ev_bb_extra + uplift + tc_extra - var_pen),
            "hit_cost": _round1(hits_val),
            "ev_total": _round1(total),
            "bank_after": _round1(pulp.value(bank[g])),
            "free_transfers_start": int(pulp.value(ft[g])),
            "transfers_used": int(pulp.value(u[g])),
            "hits_charged": int(pulp.value(hits[g])),
            "transfers": transfers_out,
            "xi": [
                _player_obj(pid.index(xid), pid, names, pos, teams,
                            np.array([EV[g].get(pid.index(xid), 0.0)]*N),
                            OPP[g], IS_HOME[g], VENUE[g], (fdr_map[g] if fdr_map is not None else None))
                for xid in xi_ids
            ],
            "bench": {
                "order": [
                    (_player_obj(pid.index(xid), pid, names, pos, teams,
                                 np.array([EV[g].get(pid.index(xid), 0.0)]*N),
                                 OPP[g], IS_HOME[g], VENUE[g], (fdr_map[g] if fdr_map is not None else None))
                     if xid else None)
                    for xid in bench_order_ids
                ],
                "gk": (_player_obj(gk_bench_idx, pid, names, pos, teams,
                                   np.array([EV[g].get(gk_bench_idx, 0.0)]*N),
                                   OPP[g], IS_HOME[g], VENUE[g], (fdr_map[g] if fdr_map is not None else None))
                       if gk_bench_idx is not None else None)
            },
        })

    ev_sum = float(pulp.value(total_obj))
    hit_sum = float(sum(HIT_COST * pulp.value(hits[g]) for g in horizon_gws))
    var_sum = float(risk_lambda * sum(sum(VAR[g].get(i, 0.0) * pulp.value(s[i, g]) for i in range(N)) for g in horizon_gws))

    plan = {
        "meta": {
            "season": season,
            "gw_start": int(gw_start),
            "next_k": int(next_k),
            "snapshot_gw": snapshot_gw,
            "notes": [
                f"EV source column: {ev_col}",
                f"GW column: {gw_col}",
                "FT carryover: roll (0 transfers) => next=2, else next=1.",
                "WC week: hits forced to 0; permanent transfers, budget respected.",
                "BB counts EV of all 15; no double count of starters.",
                "FH is not modeled in this version.",
            ],
        },
        "chip_schedule": chip_schedule,
        "per_gw": per_gw,
        "objective": {
            "risk_lambda": float(risk_lambda),
            "hit_cost_unit": HIT_COST,
            "ev_sum": _round1(ev_sum),
            "hit_cost_sum": _round1(hit_sum),
            "risk_penalty_sum": _round1(var_sum),
            "total": _round1(ev_sum),
        },
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"INFO: Wrote plan JSON → {out_path}")

    md_path = os.path.splitext(out_path)[0] + ".md"
    lines = [f"# Multi-GW Plan (GW{gw_start} → GW{horizon_gws[-1]})",
             f"- Chip schedule: {chip_schedule}",
             f"- Total objective: {plan['objective']['total']} (λ={risk_lambda})",
             ""]
    for row in per_gw:
        lines.append(f"## GW{row['gw']}  [{row['chip'] or 'NONE'}]  XI EV={row['ev_team']}, hits={row['hit_cost']}, bank_after={row['bank_after']}")
        if row["transfers"]:
            for t in row["transfers"]:
                if t["in"]:
                    lines.append(f"- {t['out_name'] or '—'} → {t['in_name']}  (sell={t['sell_value']}, buy={t['buy_price']}, Δ={t['pair_net']})")
                else:
                    lines.append(f"- {t['in_name']}  (buy={t['buy_price']})")
        else:
            lines.append("- No transfers")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"INFO: Wrote markdown → {md_path}")

    return plan

# ---------------- Chip & option sweep ----------------
def sweep_chips_and_solve(
    team_state_path: str,
    optimizer_input_path: str,
    out_base_dir: str,
    gw_start: int,
    next_k: int,
    chips_csv: str,
    risk_lambda: float,
    tc_multiplier: float,
    topk: Dict[str, int],
    sweep_ft0: List[int],
    sweep_allow_hits: List[bool],
    max_extra_transfers: int,
    lock_ids: Set[str],
    ban_ids: Set[str],
    gw_col: str,
    verbose: bool,
):
    """
    Explore schedules:
      - For NONE and each chip in {TC,BB,WC}, optionally force the chip on each GW ∈ horizon.
      - Sweep over initial FT0 values and include-hits flag combos.
      - Skip infeasible combos; still select the best feasible.
    """
    chips = [c.strip().upper() for c in (chips_csv or "NONE").split(",") if c.strip()]
    chips = [c for c in chips if c in {"NONE","TC","BB","WC"}]
    gws = list(range(gw_start, gw_start + next_k))

    results = []

    def run_once(force_chip_at: Dict[str, Optional[int]], subdir: str, fname: str,
                 ft0_override: Optional[int], allow_hits: bool):
        out_dir = os.path.join(out_base_dir, "multi", f"gw{gw_start}_{next_k}w", subdir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        try:
            plan = solve_multi_gw(
                team_state_path=team_state_path,
                optimizer_input_path=optimizer_input_path,
                out_path=out_path,
                gw_start=gw_start,
                next_k=next_k,
                risk_lambda=risk_lambda,
                tc_multiplier=tc_multiplier,
                topk=topk,
                allow_hits=allow_hits,
                forbid_fh=True,
                verbose=verbose,
                max_extra_transfers=max_extra_transfers,
                lock_ids=lock_ids,
                ban_ids=ban_ids,
                gw_col=gw_col,
                force_chip_at=force_chip_at,
                ft0_override=ft0_override,
            )
            score = float(plan["objective"]["total"])
            return {"status": "ok", "path": out_path, "score": score}
        except MILPStatusError as e:
            if verbose:
                print(f"WARN: sweep combo infeasible (status={e.status}) → "
                      f"{force_chip_at}, ft0={ft0_override}, allow_hits={allow_hits}",
                      file=sys.stderr)
            return {"status": e.status or "Infeasible", "path": None, "score": float("-inf")}

    for ft0 in (sweep_ft0 or [None]):
        for ah in (sweep_allow_hits or [True]):
            subdir = os.path.join("chips", "NONE")
            r = run_once({"TC": None, "BB": None, "WC": None}, subdir, "plan.json", ft0, ah)
            results.append({"chip":"NONE","when":None,"path":r["path"],"total":r["score"],
                            "ft0":ft0,"allow_hits":ah,"status":r["status"]})
            for chip in [c for c in chips if c != "NONE"]:
                chip_root = os.path.join("chips", chip)
                best_g, best_path, best_score, best_status = None, None, float("-inf"), None
                for g in gws:
                    forced = {"TC": None, "BB": None, "WC": None}
                    forced[chip] = g
                    fname = f"{chip}_GW{g}.json"
                    r = run_once(forced, chip_root, fname, ft0, ah)
                    # update on strictly better score
                    if r["score"] > best_score:
                        best_g, best_path, best_score, best_status = g, r["path"], r["score"], r["status"]
                # if all attempts were infeasible, mark explicitly
                if best_status is None:
                    best_status = "Infeasible"
                results.append({"chip":chip,"when":best_g,"path":best_path,"total":best_score,
                                "ft0":ft0,"allow_hits":ah,"status":best_status})

    feasible = [r for r in results if r.get("status") == "ok"]
    infeasible_cnt = sum(1 for r in results if (r.get("status") or "") != "ok")

    if feasible:
        best_item = max(feasible, key=lambda r: r["total"])
        rationale = {"tested": results, "infeasible": infeasible_cnt, "selected": best_item}
        print(json.dumps(rationale, indent=2))
        return rationale
    else:
        # robust sorting: coerce status to string and total to number
        sample = sorted(
            results,
            key=lambda r: ((r.get("status") or "zzz"), - (r.get("total") if r.get("total") is not None else float("-inf")))
        )[:8]
        msg = {
            "error": "All sweep combinations were infeasible.",
            "tips": [
                "Increase --max-extra-transfers (currently capped).",
                "Allow hits for at least some runs (--sweep-include-hits true,false).",
                "Loosen --lock/--ban constraints if used.",
                "Widen candidate pool via --topk (or check data quality/coverage).",
                "Verify bank/prices and SELL/BUY columns are present and numeric."
            ],
            "samples": sample
        }
        print(json.dumps(msg, indent=2), file=sys.stderr)
        raise RuntimeError("All sweep combinations infeasible; see stderr for details.")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Multi-GW MILP selector (transfers + XI + C/VC + chips)")
    ap.add_argument("--team-state", required=True)
    ap.add_argument("--optimizer-input", required=True)
    ap.add_argument("--out-base", default="data/plans/multi/tranfers", help="root output dir")
    ap.add_argument("--out", help="path to write the single best plan (JSON)")

    ap.add_argument("--gw-start", type=int, required=True)
    ap.add_argument("--next-k", type=int, required=True)

    ap.add_argument("--risk-lambda", type=float, default=0.0)
    ap.add_argument("--tc-multiplier", type=float, default=3.0)
    ap.add_argument("--topk", default="GK:20,DEF:60,MID:60,FWD:40")
    ap.add_argument("--allow-hits", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--gw-col", default="gw")

    ap.add_argument("--max-extra-transfers", type=int, default=2,
                    help="Cap on extra transfers beyond FT_g each GW when hits are allowed (default: 2).")
    ap.add_argument("--lock", default="", help="Comma/space-separated player_ids to force-keep across horizon.")
    ap.add_argument("--ban", default="", help="Comma/space-separated player_ids to forbid across horizon.")

    ap.add_argument("--sweep-chips", default="NONE",
                    help="Comma list among NONE,TC,BB,WC (FH not supported).")
    ap.add_argument("--sweep-free-transfers", default="",
                    help="Comma list of initial FT0 values to test (e.g., '1,2').")
    ap.add_argument("--sweep-include-hits", default="",
                    help="Comma list of booleans for allowing hits (e.g., 'true,false').")

    args = ap.parse_args()

    topk = _parse_topk(args.topk)
    lock_ids = _parse_id_list(args.lock)
    ban_ids  = _parse_id_list(args.ban)

    out_dir = os.path.join(args.out_base, "multi", f"gw{args.gw_start}_{args.next_k}w")
    os.makedirs(out_dir, exist_ok=True)

    sweep_ft0 = _parse_int_list(args.sweep_free_transfers)
    sweep_hits = _parse_bool_list(args.sweep_include_hits) if args.sweep_include_hits else []

    if (args.sweep_chips and args.sweep_chips.upper() != "NONE") or sweep_ft0 or sweep_hits:
        _ = sweep_chips_and_solve(
            team_state_path=args.team_state,
            optimizer_input_path=args.optimizer_input,
            out_base_dir=args.out_base,
            gw_start=args.gw_start,
            next_k=args.next_k,
            chips_csv=args.sweep_chips,
            risk_lambda=args.risk_lambda,
            tc_multiplier=args.tc_multiplier,
            topk=topk,
            sweep_ft0=sweep_ft0,
            sweep_allow_hits=(sweep_hits if sweep_hits else [bool(args.allow_hits)]),
            max_extra_transfers=int(args.max_extra_transfers),
            lock_ids=lock_ids,
            ban_ids=ban_ids,
            gw_col=args.gw_col,
            verbose=bool(args.verbose),
        )
        return

    out_path = args.out or os.path.join(out_dir, "plan.json")
    plan = solve_multi_gw(
        team_state_path=args.team_state,
        optimizer_input_path=args.optimizer_input,
        out_path=out_path,
        gw_start=args.gw_start,
        next_k=args.next_k,
        risk_lambda=args.risk_lambda,
        tc_multiplier=args.tc_multiplier,
        topk=topk,
        allow_hits=bool(args.allow_hits),
        forbid_fh=True,
        verbose=bool(args.verbose),
        max_extra_transfers=int(args.max_extra_transfers),
        lock_ids=lock_ids,
        ban_ids=ban_ids,
        gw_col=args.gw_col,
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
