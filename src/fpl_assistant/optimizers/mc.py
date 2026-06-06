#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte-Carlo simulator (distribution-aware) for a single roster over 1..N GWs.

Input: sim_input files emitted by simulator.py (one file per GW)
Output:
  out_dir/<prefix>_summary.json
  out_dir/<prefix>_player_samples.<parquet|csv|jsonl>   (optional via --export-player-samples)

Captaincy:
- We choose captain per-GW by rule:
    ev   : player with highest mean points (from the same sim run, no peeking per-path)
    risk : argmax mean - alpha * stdev

Bench realism (v0):
- GK autosub: if starting GK mins==0, and bench GK mins>0 -> swap
- Outfield autosubs: replace 0-min starters in bench order with bench players (mins>0),
  ignoring formation constraints.

Minutes model:
- With probability p60: mins ~ N(pred_minutes, sigma) clipped to [60,100]
- Else with probability cameo_prob: mins ~ Uniform[1, cameo_max]
- Else mins = 0

Team-goal model:
- For each team in your 15 per GW: Goals_for ~ Poisson(lambda_for),
  Goals_conc ~ Poisson(lambda_against)
- Allocate goals via Multinomial over players with weights ∝ xg_share * minutes_on_pitch
- Assists: per goal, draw assister via xa_share among teammates excluding the scorer

Saves:
- GK saves ~ Poisson(pred_saves_mean); points += floor(saves/3)

NOTE: If your upstream lambda_* are not per-fixture (e.g., season totals), points will be inflated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

POS_GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3

# ------------------------- Utils -------------------------

def parse_gws(gws_str: str) -> List[int]:
    parts = [p.strip() for p in gws_str.split(',') if p.strip()]
    gws = [int(p) for p in parts]
    if not gws:
        raise ValueError("No GWs parsed from --gws")
    return gws

def read_sim_input(sim_input_dir: Path, gw: int, in_format: str) -> pd.DataFrame:
    p = sim_input_dir / f"gw{gw}.{in_format}"
    if not p.exists():
        raise FileNotFoundError(f"Missing sim_input file: {p}")
    if in_format == "csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)

    # required columns
    need = ['gw','fixture_id','team_id','opp_team_id','player_id','player','pos',
            'xg_share','xa_share','lambda_goals_for','lambda_goals_against',
            'pred_minutes','p60','pred_saves_mean','is_start_xi','bench_order','is_bench_gk']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{p} missing columns: {missing}")

    # normalize types
    df['pos'] = df['pos'].astype(str).str.upper().str[:3]
    df['player_id'] = df['player_id'].astype(str)  # keep ids as strings (may be hex/hash)
    df['player'] = df['player'].astype(str)
    df['bench_order'] = df['bench_order'].fillna(0).astype(int)
    for c in ['is_start_xi','is_bench_gk']:
        df[c] = df[c].fillna(False).astype(bool)

    return df

def ensure_col(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default
    else:
        df[col] = df[col].fillna(default)
    return df

def truncnorm(mean, sigma, low, high, size):
    """Normal draw clipped into [low, high]."""
    x = np.random.normal(loc=mean, scale=sigma, size=size)
    return np.clip(x, low, high)

def minutes_sampler(pred_minutes, p60, minutes_sigma, cameo_prob, cameo_max):
    """
    Vectorized minutes draw for shape (n_players,).
    Returns minutes (float).
    """
    n = pred_minutes.shape[0]
    u = np.random.rand(n)
    mins = np.zeros(n, dtype=float)

    mask60 = u < p60
    if mask60.any():
        mins[mask60] = truncnorm(pred_minutes[mask60], minutes_sigma, 60.0, 100.0, mask60.sum())

    mask_no60 = ~mask60
    if mask_no60.any():
        cameo_mask = np.random.rand(mask_no60.sum()) < cameo_prob
        idx = np.where(mask_no60)[0]
        cameo_idx = idx[cameo_mask]
        if cameo_idx.size > 0:
            mins[cameo_idx] = np.random.randint(1, int(cameo_max)+1, size=cameo_idx.size)
        # remaining stay 0 (DNP)

    return mins

def allocate_assists_per_team(n_goals: int, scorer_draws: np.ndarray, w_assist: np.ndarray) -> np.ndarray:
    """
    For each goal, draw an assister among teammates (excluding the scorer for that goal).
    scorer_draws: indices in [0, n_players_in_team) for each goal.
    Returns assist counts per player (len = n_players_in_team).
    """
    if n_goals <= 0 or w_assist.sum() <= 0:
        return np.zeros_like(w_assist, dtype=int)

    counts = np.zeros_like(w_assist, dtype=int)
    for g in range(n_goals):
        scorer_k = scorer_draws[g]
        w = w_assist.copy()
        if w[scorer_k] > 0:
            w[scorer_k] = 0.0
        s = w.sum()
        if s <= 0:
            continue
        pick = np.random.choice(w.size, p=w / s)
        counts[pick] += 1
    return counts

def fpl_points_from_events(pos_arr, mins, goals, assists, cs_flag, gc, saves):
    """
    Compute FPL points arrays per player for a single GW (no captain).
    """
    n = len(pos_arr)
    pts = np.zeros(n, dtype=float)

    # Appearance
    pts += np.where(mins >= 60, 2.0, np.where(mins > 0, 1.0, 0.0))

    # Goals
    goal_pts = np.zeros(n)
    goal_pts += (pos_arr == "GK") * POS_GOAL_POINTS["GK"]
    goal_pts += (pos_arr == "DEF") * POS_GOAL_POINTS["DEF"]
    goal_pts += (pos_arr == "MID") * POS_GOAL_POINTS["MID"]
    goal_pts += (pos_arr == "FWD") * POS_GOAL_POINTS["FWD"]
    pts += goals * goal_pts

    # Assists
    pts += assists * ASSIST_POINTS

    # Clean sheet bonus
    pts += ((pos_arr == "GK") | (pos_arr == "DEF")) * (cs_flag.astype(int) * 4)
    pts += (pos_arr == "MID") * (cs_flag.astype(int) * 1)
    # FWD gets 0

    # Goals conceded penalty (GK/DEF): -1 per 2 conceded if they played at all
    pts += -1.0 * (((pos_arr == "GK") | (pos_arr == "DEF")) * (gc // 2) * (mins > 0))

    # Saves: GK only, 1pt per 3 saves
    pts += (pos_arr == "GK") * (saves // 3)

    return pts

def apply_bench_autosubs(is_start_xi, is_bench_gk, bench_order, mins, pos):
    """
    Return mask of players counted in final XI after autosubs.
    - GK autosub only GK<->GK.
    - Outfield subs by bench order; no formation enforcement (v0).
    """
    in_xi = is_start_xi.copy()

    # GK autosub
    gk_mask = (pos == "GK")
    start_gk_idx = np.where(in_xi & gk_mask)[0]
    bench_gk_idx = np.where(is_bench_gk & gk_mask)[0]
    if start_gk_idx.size == 1:
        sg = start_gk_idx[0]
        if mins[sg] <= 0 and bench_gk_idx.size == 1 and mins[bench_gk_idx[0]] > 0:
            in_xi[sg] = False
            in_xi[bench_gk_idx[0]] = True

    # Outfield autosubs
    zero_starters = np.where(in_xi & (mins <= 0) & (pos != "GK"))[0].tolist()
    if zero_starters:
        bench_out = np.where((~in_xi) & (~is_bench_gk) & (pos != "GK"))[0]
        bench_out = bench_out[np.argsort(bench_order[bench_out])]  # 1..3
        for b in bench_out:
            if mins[b] > 0 and zero_starters:
                s = zero_starters.pop(0)
                in_xi[s] = False
                in_xi[b] = True
            if not zero_starters:
                break

    # ensure at most 11
    if in_xi.sum() > 11:
        candidates = np.where(in_xi & (pos != "GK"))[0]
        if candidates.size > 0:
            order = candidates[np.argsort(mins[candidates])]
            for idx in order:
                if in_xi.sum() <= 11:
                    break
                in_xi[idx] = False
    return in_xi

# ------------------------- Core sim -------------------------

def simulate_gw(df_gw: pd.DataFrame, n_sims: int, minutes_sigma: float, cameo_prob: float, cameo_max: int) -> Dict[str, np.ndarray]:
    """
    Run n_sims for ONE GW and return dict of arrays:
    {
      'team_points': (n_sims,),
      'player_points': (n_sims, n_players),
      'player_ids': (n_players,),  # strings
      'player_names': (n_players,),
      'pos': (n_players,),
      'is_start_xi': (n_players,)
    }
    """
    df = df_gw.reset_index(drop=True).copy()
    n_players = len(df)

    pos = df['pos'].astype(str).str.upper().str[:3].values
    player_ids = df['player_id'].astype(str).values        # keep as strings
    player_names = df['player'].astype(str).values

    pred_minutes = df['pred_minutes'].astype(float).values
    p60 = df['p60'].astype(float).values

    is_start_xi = df['is_start_xi'].values.astype(bool)
    is_bench_gk = df['is_bench_gk'].values.astype(bool)
    bench_order = df['bench_order'].values.astype(int)

    teams = df['team_id'].values
    team_ids_unique = np.unique(teams)
    t2idx = {tid: i for i, tid in enumerate(team_ids_unique)}
    n_teams = len(team_ids_unique)

    lam_for = np.zeros(n_teams)
    lam_against = np.zeros(n_teams)
    for tid in team_ids_unique:
        r = df[df['team_id']==tid].iloc[0]
        lam_for[t2idx[tid]] = r['lambda_goals_for']
        lam_against[t2idx[tid]] = r['lambda_goals_against']

    xg_share = df['xg_share'].astype(float).values
    xa_share = df['xa_share'].astype(float).values
    pred_saves_mean = df['pred_saves_mean'].astype(float).values

    team_idx_per_player = np.array([t2idx[t] for t in teams], dtype=int)

    team_points = np.zeros(n_sims, dtype=float)
    player_points_all = np.zeros((n_sims, n_players), dtype=float)

    for s in range(n_sims):
        # minutes
        mins = minutes_sampler(pred_minutes, p60, minutes_sigma, cameo_prob, cameo_max)

        # team goals and conceded
        goals_for_team = np.random.poisson(lam_for)
        goals_conc_team = np.random.poisson(lam_against)

        goals = np.zeros(n_players, dtype=int)
        assists = np.zeros(n_players, dtype=int)

        # per-team allocation
        for t_i, tid in enumerate(team_ids_unique):
            pidx = np.where(team_idx_per_player == t_i)[0]
            if pidx.size == 0:
                continue
            w_goal = xg_share[pidx] * (mins[pidx] > 0).astype(float) * np.maximum(mins[pidx], 1.0)
            w_asst = xa_share[pidx] * (mins[pidx] > 0).astype(float) * np.maximum(mins[pidx], 1.0)

            ng = int(goals_for_team[t_i])
            if ng > 0 and w_goal.sum() > 0:
                p = w_goal / w_goal.sum()
                scorer_draws = np.random.choice(pidx.size, size=ng, p=p)
                goals[pidx] += np.bincount(scorer_draws, minlength=pidx.size)

                if w_asst.sum() > 0:
                    assists[pidx] += allocate_assists_per_team(ng, scorer_draws, w_asst)

        gc_players = goals_conc_team[team_idx_per_player]
        cs_flag = (gc_players == 0) & (mins >= 60.0)

        saves = np.zeros(n_players, dtype=int)
        gk_idx = np.where(pos == "GK")[0]
        if gk_idx.size > 0:
            lam_saves = np.maximum(pred_saves_mean[gk_idx], 0.0)
            saves[gk_idx] = np.where(mins[gk_idx] > 0, np.random.poisson(lam_saves), 0)

        pts = fpl_points_from_events(pos, mins, goals, assists, cs_flag, gc_players, saves)

        in_xi_mask = apply_bench_autosubs(is_start_xi, is_bench_gk, bench_order, mins, pos)
        team_pts = pts[in_xi_mask].sum()

        player_points_all[s, :] = pts
        team_points[s] = team_pts

    return {
        'team_points': team_points,
        'player_points': player_points_all,
        'player_ids': player_ids,
        'player_names': player_names,
        'pos': pos,
        'is_start_xi': is_start_xi,
    }

def choose_captain(player_points: np.ndarray, names: np.ndarray, rule: str, alpha: float = 0.5) -> Tuple[int, Dict]:
    means = player_points.mean(axis=0)
    ddof = 1 if player_points.shape[0] > 1 else 0
    sds = player_points.std(axis=0, ddof=ddof)
    score = means - alpha * sds if rule == "risk" else means
    cap_idx = int(np.argmax(score))
    order = np.argsort(score)[::-1]
    gap = score[order[0]] - (score[order[1]] if order.size > 1 else 0.0)
    info = {
        "captain": str(names[cap_idx]),
        "rule": rule,
        "cap_mean": float(means[cap_idx]),
        "cap_sd": float(sds[cap_idx]),
        "second_best": str(names[order[1]]) if order.size > 1 else None,
        "gap": float(gap)
    }
    return cap_idx, info

def summarize_distribution(samples: np.ndarray) -> Dict[str, float]:
    mean = float(samples.mean())
    sd = float(samples.std(ddof=1)) if samples.shape[0] > 1 else 0.0
    var10 = float(np.percentile(samples, 10))
    tail = samples[samples <= var10]
    cvar10 = float(tail.mean()) if tail.size > 0 else var10
    p60 = float((samples > 60).mean())
    p80 = float((samples > 80).mean())
    return {
        "mean": mean,
        "stdev": sd,
        "VaR10": var10,
        "CVaR10": cvar10,
        "P_gt_60": p60,
        "P_gt_80": p80
    }

def write_df(df: pd.DataFrame, path: Path, fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8")
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False, encoding="utf-8")
            print(f"Parquet write failed ({e}). Wrote CSV fallback at: {fallback}")

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Monte-Carlo simulator for team distributions")
    ap.add_argument("--sim-input-dir", required=True, help="Directory containing gw*.parquet/csv from simulator.py")
    ap.add_argument("--gws", required=True, help="Comma-separated GWs, e.g., 4,5,6")
    ap.add_argument("--in-format", choices=["parquet","csv"], default="parquet", help="Input format (matches simulator output)")
    ap.add_argument("--out-dir", required=True, help="Output directory for sims")
    ap.add_argument("--out-prefix", help="Prefix for output files; default gw{min}_{max} or gw{gw}")
    ap.add_argument("--n-sims", type=int, default=20000)
    ap.add_argument("--minutes-sigma", type=float, default=12.0)
    ap.add_argument("--cameo-prob", type=float, default=0.50, help="Prob of cameo (<60) when not 60+")
    ap.add_argument("--cameo-max", type=int, default=30)
    ap.add_argument("--captain-rule", choices=["ev","risk"], default="ev")
    ap.add_argument("--risk-alpha", type=float, default=0.5, help="Used for captain-rule=risk (mean - alpha*sd)")
    ap.add_argument("--export-player-samples", action="store_true")
    ap.add_argument("--out-format", choices=["parquet","csv","jsonl"], default="parquet",
                    help="Player samples format (summary is always JSON)")
    args = ap.parse_args()

    sim_input_dir = Path(args.sim_input_dir)
    gws = parse_gws(args.gws)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.out_prefix or (f"gw{gws[0]}" if len(gws) == 1 else f"gw{gws[0]}_{gws[-1]}")

    per_gw_team_points = []
    per_gw_player_points = []
    per_gw_player_meta = []

    # ---- per-GW sims ----
    summaries = []
    cap_infos = []
    for gw in gws:
        df_gw = read_sim_input(sim_input_dir, gw, args.in_format)
        sim = simulate_gw(df_gw, args.n_sims, args.minutes_sigma, args.cameo_prob, args.cameo_max)

        cap_idx, cap_info = choose_captain(sim['player_points'], sim['player_names'], args.captain_rule, args.risk_alpha)
        cap_infos.append({"gw": gw, **cap_info})

        # apply captain multiplier (approx: chosen cap is in XI; add extra copy)
        cap_points = sim['player_points'][:, cap_idx]
        team_points = sim['team_points'] + cap_points

        per_gw_team_points.append(team_points)
        per_gw_player_points.append(sim['player_points'])
        per_gw_player_meta.append({
            "player_ids": sim['player_ids'],     # strings
            "player_names": sim['player_names'],
            "pos": sim['pos']
        })

        summaries.append({
            "gw": gw,
            "summary": summarize_distribution(team_points),
            "captain": cap_info
        })

    # horizon aggregate
    total_points = np.sum(np.stack(per_gw_team_points, axis=0), axis=0)
    horizon = {
        "gws": gws,
        "summary": summarize_distribution(total_points)
    }

    # write summary JSON (keep UTF-8 names unescaped)
    summary = {
        "horizon": horizon,
        "per_gw": summaries,
    }
    out_summary = out_dir / f"{prefix}_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_summary}")

    # optional per-player samples
    if args.export_player_samples:
        rows = []

        # per-GW
        for i, gw in enumerate(gws):
            P = per_gw_player_points[i]     # (n_sims, n_players)
            meta = per_gw_player_meta[i]
            ddof = 1 if P.shape[0] > 1 else 0
            means = P.mean(axis=0)
            sds = P.std(axis=0, ddof=ddof)
            for j in range(P.shape[1]):
                rows.append({
                    "gw": gw,
                    "scope": "per_gw",
                    "player_id": str(meta["player_ids"][j]),     # keep as string
                    "player": str(meta["player_names"][j]),
                    "pos": str(meta["pos"][j]),
                    "mean": float(means[j]),
                    "stdev": float(sds[j]),
                })

        # horizon sum per player (ID-based, string-safe)
        all_ids: Dict[str, Dict[str,str]] = {}
        for meta in per_gw_player_meta:
            for pid, name, pos in zip(meta["player_ids"], meta["player_names"], meta["pos"]):
                pid_s = str(pid)
                if pid_s not in all_ids:
                    all_ids[pid_s] = {"name": str(name), "pos": str(pos)}

        for pid, info in all_ids.items():
            samples = None
            for i in range(len(gws)):
                meta = per_gw_player_meta[i]
                P = per_gw_player_points[i]
                # locate this pid
                indices = np.where(np.array(meta["player_ids"]) == pid)[0]
                if indices.size > 0:
                    vec = P[:, indices[0]]
                else:
                    vec = np.zeros(P.shape[0], dtype=float)
                samples = vec if samples is None else (samples + vec)
            ddof = 1 if samples.shape[0] > 1 else 0
            rows.append({
                "gw": None,
                "scope": "horizon_sum",
                "player_id": pid,
                "player": info["name"],
                "pos": info["pos"],
                "mean": float(samples.mean()),
                "stdev": float(samples.std(ddof=ddof)),
            })

        df_rows = pd.DataFrame(rows)
        out_players = out_dir / f"{prefix}_player_samples.{args.out_format}"
        write_df(df_rows, out_players, args.out_format)
        print(f"Wrote {out_players}")

if __name__ == "__main__":
    main()
