#!/usr/bin/env python3
"""
squad_optimizer.py — v2 (production)

Build a 15-man FPL squad for a single GW:
- 2 GK total; 1 GK starts in XI
- XI has at least 3 DEF, 2 MID, 1 FWD (total XI=11)
- Team constraint: max K from same club
- Budget constraint (in £m)

Objective: maximize expected GW points:
  sum( xp_i * start_i + bench_weight * xp_i * (pick_i - start_i) )

Where:
  pick_i ∈ {0,1} (in 15-man squad)
  start_i ∈ {0,1} (in XI), with start_i ≤ pick_i

Minutes filter:
  --min-exp-mins applies ONLY to the starting XI eligibility (bench can be < threshold).

Price units:
  Auto-detect if price looks like "tenths" (e.g., 55) and convert to millions (5.5).

Inputs
------
--xp-by-gw     CSV from total_points_combiner (xp_by_gw.csv)
--season       season to use (tolerates '2024/25', '2024–2025', '24-25')
--gw           integer GW
--budget       float budget in millions (e.g., 100.0)
--team-max     max players per club (default 3)
--min-exp-mins minimum expected minutes to be eligible for the XI (bench may be lower)
--bench-weight discount factor for bench points (default 0.1)
--out-dir      base output directory
--version      version subfolder; use --auto-version to pick next vN
--auto-version
--write-latest
--log-level

Outputs
-------
<out-dir>/<version>/squad.csv        # picked 15 with start flag and price/XP
<out-dir>/<version>/summary.json     # totals and diagnostics
"""

from __future__ import annotations
import argparse, json, logging, os, re, datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pulp as pl

SCHEMA_VERSION = "v2"

# ---------------- auto-version helpers ----------------
def _resolve_version(base_dir: Path, requested: str|None, auto: bool) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    if auto or not requested or requested.lower() == "auto":
        existing = [p.name for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        nxt = (max(int(s[1:]) for s in existing) + 1) if existing else 1
        ver = f"v{nxt}"
        logging.info("[info] auto-version -> %s", ver)
        return ver
    if not re.fullmatch(r"v\d+", requested):
        if requested.isdigit(): return f"v{requested}"
        raise ValueError("--version must be like v3 or pass --auto-version")
    return requested

def _write_latest_pointer(root: Path, version: str) -> None:
    latest = root / "latest"
    target = root / version
    try:
        if latest.exists() or latest.is_symlink():
            try: latest.unlink()
            except Exception: pass
        os.symlink(target.name, latest, target_is_directory=True)
        logging.info("Updated 'latest' symlink -> %s", version)
    except (OSError, NotImplementedError):
        (root / "LATEST_VERSION.txt").write_text(version, encoding="utf-8")
        logging.info("Wrote LATEST_VERSION.txt -> %s", version)

# ---------------- season normalization ----------------
_dash = re.compile(r"[–—/]")  # en/em dash or slash
def _canon_season(s: str) -> str:
    s = _dash.sub("-", str(s).strip())
    m = re.match(r"^\s*(\d{2,4})\s*-\s*(\d{2,4})\s*$", s)
    if not m:
        return s
    y1 = int(m.group(1)); y2 = m.group(2)
    if len(m.group(1)) == 2: y1 += 2000
    if len(y2) == 2: y2 = str(int(y2) + (y1 // 100) * 100)
    return f"{y1}-{int(y2)}"

# ---------------- I/O helpers ----------------
def _norm_xp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "date_played" in df.columns:
        df["date_played"] = pd.to_datetime(df["date_played"], errors="coerce")
    if "season" in df.columns:
        df["season"] = df["season"].astype(str).map(_canon_season)
    if "gw_orig" in df.columns:
        df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")
    for c in ["player_id","team_id","pos","player"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in ["exp_points_total","pred_exp_minutes","price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_xp_by_gw(fp: Path, season: str, gw: int) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False)
    df = _norm_xp(df)
    need = {"season","gw_orig","player_id","team_id","pos","player","exp_points_total","price"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"xp_by_gw missing columns: {miss}")
    df = df[(df["season"] == _canon_season(season)) & (df["gw_orig"] == int(gw))].copy()
    if df.empty:
        raise SystemExit(f"No rows for season={season}, gw={gw} in {fp}")
    return df

def _normalize_prices_to_millions(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Heuristic: if median price >= 25, treat as 'tenths' and divide by 10.
    Return df with 'price_m' and the scale used.
    """
    df["price"] = df["price"].fillna(999.0)

    med = float(df["price"].dropna().median())
    if not np.isfinite(med):
        raise ValueError("price has no finite values; price registry likely missing.")
    if med >= 25:  # e.g., 55 tenths -> 5.5m
        df["price_m"] = df["price"] / 10.0
        return df, 0.1
    else:
        df["price_m"] = df["price"].astype(float)
        return df, 1.0

def _canon_pos(s: pd.Series) -> pd.Series:
    p = s.str.upper().str[:3]
    return p.replace({"GKP":"GK","DEF":"DEF","MID":"MID","FWD":"FWD"})

# ---------------- solver ----------------
def optimize_squad(df: pd.DataFrame,
                   budget_m: float,
                   team_max: int,
                   min_exp_mins_xi: float,
                   bench_weight: float) -> Dict:
    # Canonical position
    df["pos"] = _canon_pos(df["pos"])
    # Price normalize
    df, scale = _normalize_prices_to_millions(df)

    # Diagnostics: counts per pos, XI-eligible
    df["xi_eligible"] = df["pred_exp_minutes"].fillna(0.0) >= float(min_exp_mins_xi)
    counts = df.groupby("pos")["player_id"].nunique().to_dict()
    xi_counts = df[df["xi_eligible"]].groupby("pos")["player_id"].nunique().to_dict()

    # Lower bound cost (ignoring team_max) to help detect unit/budget issues
    def _lb_cost():
        parts = []
        # 2 GK cheapest
        parts.append(df[df["pos"]=="GK"].nsmallest(2, "price_m")["price_m"].sum())
        # XI minima: 1 GK, 3 DEF, 2 MID, 1 FWD
        # (we already took 2 GK overall, but XI only needs 1; this LB is rough)
        parts.append(df[df["pos"]=="DEF"].nsmallest(3, "price_m")["price_m"].sum())
        parts.append(df[df["pos"]=="MID"].nsmallest(2, "price_m")["price_m"].sum())
        parts.append(df[df["pos"]=="FWD"].nsmallest(1, "price_m")["price_m"].sum())
        # Remaining players to reach 15: pick cheapest among all (excluding already counted indices)
        taken = set()
        for pos, k in [("GK",2), ("DEF",3), ("MID",2), ("FWD",1)]:
            taken |= set(df[df["pos"]==pos].nsmallest(k, "price_m").index)
        remain = df.drop(index=list(taken)).nsmallest(15 - (2+3+2+1), "price_m")["price_m"].sum()
        parts.append(remain)
        return float(np.sum(parts))
    lb_cost = _lb_cost()

    # Variables
    idx = list(df.index)
    pick = pl.LpVariable.dicts("pick", idx, lowBound=0, upBound=1, cat="Binary")
    start = pl.LpVariable.dicts("start", idx, lowBound=0, upBound=1, cat="Binary")

    # Problem
    prob = pl.LpProblem("FPL_Squad_Optimizer", pl.LpMaximize)

    xp = df["exp_points_total"].fillna(0.0).to_dict()
    price = df["price_m"].to_dict()
    pos = df["pos"].to_dict()
    team = df["team_id"].to_dict()
    xi_eligible = df["xi_eligible"].to_dict()

    # Objective
    prob += pl.lpSum( (xp[i] * start[i]) + (bench_weight * xp[i] * (pick[i] - start[i])) for i in idx )

    # Core constraints
    # total players
    prob += pl.lpSum(pick[i] for i in idx) == 15, "total15"
    # XI total
    prob += pl.lpSum(start[i] for i in idx) == 11, "xi11"
    # start implies picked
    for i in idx:
        prob += start[i] <= pick[i], f"start_implies_pick_{i}"
    # XI eligibility by minutes
    for i in idx:
        if not xi_eligible[i]:
            prob += start[i] == 0, f"no_start_lowmins_{i}"
    # Position constraints
    # 2 GK total
    prob += pl.lpSum(pick[i] for i in idx if pos[i]=="GK") == 2, "two_gk_total"
    # exactly 1 GK in XI
    prob += pl.lpSum(start[i] for i in idx if pos[i]=="GK") == 1, "one_gk_xi"
    # XI minima
    prob += pl.lpSum(start[i] for i in idx if pos[i]=="DEF") >= 3, "xi_min_def_3"
    prob += pl.lpSum(start[i] for i in idx if pos[i]=="MID") >= 2, "xi_min_mid_2"
    prob += pl.lpSum(start[i] for i in idx if pos[i]=="FWD") >= 1, "xi_min_fwd_1"
    # Budget
    prob += pl.lpSum(price[i] * pick[i] for i in idx) <= float(budget_m), "budget"
    # Team max
    for t, rows in df.groupby("team_id").groups.items():
        prob += pl.lpSum(pick[i] for i in rows) <= int(team_max), f"teamcap_{t}"

    # Solve
    status = prob.solve(pl.PULP_CBC_CMD(msg=False))  # use default CBC

    if pl.LpStatus[status] != "Optimal":
        # Build diagnostics
        diag = {
            "solver_status": pl.LpStatus[status],
            "budget_m": float(budget_m),
            "price_unit_scale_used": scale,  # 0.1 means tenths→millions
            "lower_bound_cost_m_ignoring_teammax": lb_cost,
            "counts_all": counts,
            "counts_xi_eligible": xi_counts
        }
        raise RuntimeError(json.dumps(diag, indent=2))

    # Extract solution
    df_out = df.copy()
    df_out["pick"]  = [int(round(pick[i].value()))  for i in idx]
    df_out["start"] = [int(round(start[i].value())) for i in idx]
    df_out = df_out[df_out["pick"] == 1].copy().sort_values(["start","pos","price_m"], ascending=[False, True, True])

    total_price = float((df_out["price_m"]).sum())
    xi_points   = float((df_out.loc[df_out["start"]==1, "exp_points_total"]).sum())
    bench_points= float((df_out.loc[df_out["start"]==0, "exp_points_total"]).sum())
    objective   = xi_points + bench_weight * bench_points

    # per-team counts
    team_counts = (df_out.groupby("team_id")["player_id"].count()
                      .sort_values(ascending=False).to_dict())
    pos_counts  = (df_out.groupby("pos")["player_id"].count()
                      .sort_values(ascending=False).to_dict())

    return {
        "squad_df": df_out,
        "objective": objective,
        "xi_points": xi_points,
        "bench_points": bench_points,
        "total_price_m": total_price,
        "team_counts": team_counts,
        "pos_counts": pos_counts
    }

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xp-by-gw", type=Path, required=True)
    ap.add_argument("--season", type=str, required=True)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--budget", type=float, default=100.0, help="£m, e.g. 100.0")
    ap.add_argument("--team-max", type=int, default=3)
    ap.add_argument("--min-exp-mins", type=float, default=45.0, help="XI eligibility threshold")
    ap.add_argument("--bench-weight", type=float, default=0.1)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--version", type=str, default=None)
    ap.add_argument("--auto-version", action="store_true")
    ap.add_argument("--write-latest", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    version = _resolve_version(args.out_dir, args.version, args.auto_version)
    out_dir = args.out_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_xp_by_gw(args.xp_by_gw, args.season, args.gw)

    # Basic cleaning
    # Keep one row per player (xp_by_gw should already be aggregated, but be safe)
    key = ["season","gw_orig","player_id","team_id","pos","player"]
    df = (df.groupby(key, as_index=False)
            .agg(exp_points_total=("exp_points_total","sum"),
                 pred_exp_minutes=("pred_exp_minutes","max"),
                 price=("price","max")))

    # Run optimization
    try:
        res = optimize_squad(df,
                             budget_m=args.budget,
                             team_max=args.team_max,
                             min_exp_mins_xi=args.min_exp_mins,
                             bench_weight=args.bench_weight)
    except RuntimeError as e:
        logging.error("Infeasible model. Diagnostics:\n%s", str(e))
        raise

    squad = res["squad_df"].copy()
    squad = squad[["player_id","player","team_id","pos","price_m","exp_points_total","pred_exp_minutes","start"]]
    squad = squad.rename(columns={"price_m":"price_millions","exp_points_total":"xp"})

    # Write outputs
    squad_fp = out_dir / "squad.csv"
    squad.to_csv(squad_fp, index=False)

    summary = {
        "schema": SCHEMA_VERSION,
        "build_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "season": _canon_season(args.season),
        "gw": int(args.gw),
        "budget_m": float(args.budget),
        "bench_weight": float(args.bench_weight),
        "team_max": int(args.team_max),
        "min_exp_mins_xi": float(args.min_exp_mins),
        "objective_points": res["objective"],
        "xi_points": res["xi_points"],
        "bench_points": res["bench_points"],
        "total_price_m": res["total_price_m"],
        "team_counts": res["team_counts"],
        "pos_counts": res["pos_counts"],
        "rows_written": int(len(squad))
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logging.info("Wrote squad to %s (price=%.1f m; XI=%.2f pts; bench=%.2f pts; obj=%.2f)",
                 squad_fp.resolve(), res["total_price_m"],
                 res["xi_points"], res["bench_points"], res["objective"])

    if args.write_latest:
        _write_latest_pointer(args.out_dir, version)

if __name__ == "__main__":
    main()
