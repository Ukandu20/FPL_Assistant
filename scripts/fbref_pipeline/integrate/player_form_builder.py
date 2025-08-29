#!/usr/bin/env python3
r"""player_form_builder.py – schema v1.8
(per-90, causal, venue-aware, EWMA/rolling window, binomial save% using saves & sot_against,
position-weighted composite form scores)

Versioning:
• Writes under data/processed/registry/features/<version>/<SEASON>/players_form.csv
• --auto-version chooses next vN if not specified, e.g., v7
• --write-latest adds a 'latest' pointer (symlink if supported; else LATEST_VERSION.txt)

Outputs (per season):
  players_form.csv
  player_form.meta.json

What's new vs v1.7:
  • Added EWMA option (past-only; no leakage) mirroring team_form_builder semantics.
  • Introduced position-weighted composite form scores (overall/home/away) with configurable
    weights per group (ATT/CRE/DEF/GK) for GK/DEF/MID/FWD.
  • Single-pass z-score engine (GK vs OUT pools) + venue-conditional z-scores.
  • Reduced redundant z-score recomputation; fixed minor inefficiency.
  • CLI parity: --ewma, --halflife, --pos-weights "GK:gk=1;DEF:def=.6,att=.2,cre=.2;MID:att=.4,cre=.4,def=.2;FWD:att=.6,cre=.3,def=.1".
  • Meta JSON now records ewma params and pos-weights.
"""

from __future__ import annotations
import argparse, json, logging, datetime as dt, os, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

SCHEMA_VERSION = "v1.8"

# ───────────────────────── Canonical schema ─────────────────────────

OUTPUT_FILE = "players_form.csv"

# Canonical names & alias map (left = canonical)
ALIASES: Dict[str, List[str]] = {
    "player_id": ["player_id", "fbref_player_id", "fpl_player_id", "id_player"],
    "player": ["player", "player_name", "name"],
    "pos": ["pos", "position", "position_short", "fpl_pos"],
    "fbref_id": ["fbref_id", "fbref_url_id", "game_id", "fixture_id_fbref"],
    "fpl_id": ["fpl_id", "fpl_code", "fpl_fixture_id", "fixture_id_fpl"],
    "team_id": ["team_id", "team_code", "team_short_id", "squad_id", "team_hex"],
    "opponent_id": ["opponent_id", "opp_id"],
    "team": ["team", "team_name", "squad", "club"],
    "gw_orig": ["gw_orig", "gw", "gameweek", "round", "event"],
    "date_played": ["date_played", "kickoff_time", "match_date", "date"],
    "venue": ["venue", "was_home", "is_home", "home"],
    "minutes": ["minutes", "mins", "time_played"],
    "days_since_last": ["days_since_last", "days_rest", "rest_days"],
    "is_active": ["is_active", "active", "in_squad", "appearance", "played"],

    # outcomes/context
    "yellow_crd": ["yellow_crd", "yellow", "yellow_cards", "yc"],
    "red_crd": ["red_crd", "red", "red_cards", "rc"],
    "gf": ["gf", "goals_for", "team_goals_for", "goals_for_team"],
    "ga": ["ga", "goals_against", "team_goals_against", "conceded"],
    "fdr_home": ["fdr_home", "fdr_h", "fixture_difficulty_home"],
    "fdr_away": ["fdr_away", "fdr_a", "fixture_difficulty_away"],

    # attacking/creation
    "gls": ["gls", "goals"],
    "shots": ["shots", "sh", "shots_total"],
    "sot": ["sot", "shots_on_target", "shots_ot"],
    "ast": ["ast", "assists"],
    "xg": ["xg", "expected_goals"],
    "npxg": ["npxg", "np_xg", "non_pen_xg"],
    "xag": ["xag", "xa", "expected_assists"],
    "pkatt": ["pkatt", "pens_att", "pen_att", "penalties_att"],
    "pk_scored": ["pk_scored", "pens_scored", "pen_scored"],
    "pk_won": ["pk_won", "pens_won", "pen_won", "pkwon"],

    # defensive / misc
    "blocks": ["blocks", "blk"],
    "tkl": ["tkl", "tackles", "tklw"],
    "int": ["int", "interceptions"],
    "clr": ["clr", "clearances"],
    "own_goals": ["own_goals", "og"],
    "recoveries": ["recoveries", "rec"],

    # GK
    "saves": ["saves", "sv"],
    "sot_against": ["sot_against", "shots_on_target_against", "on_target_against", "sota"],
    "save_pct": ["save_pct", "save%", "sv_pct"],

    # optional FPL enrichments carried through
    "price": ["price", "now_cost", "value"],
    "xp": ["xp", "xP", "ep_this", "expected_points", "exp_points", "xp_this"],
    "total_points": ["total_points", "points"],
    "bonus": ["bonus"],
    "bps": ["bps"],
    "clean_sheets": ["clean_sheets", "cs", "cs_count"],
}

# Numeric canonicals we’ll coerce (and often fill). Enrichments handled separately.
NUMERIC_BASE = [
    "minutes","days_since_last","is_active","yellow_crd","red_crd",
    "gf","ga","shots","sot","fdr_home","fdr_away",
    "gls","ast","blocks","tkl","int","clr","xg","npxg","xag",
    "pkatt","pk_scored","pk_won",
    "saves","sot_against","save_pct",
    "own_goals","recoveries",
]

# Enrichment numerics we coerce but do NOT default-fill (leave NaN if absent)
NUMERIC_ENRICH = ["price","xp","total_points","bonus","bps","clean_sheets"]

# Metric families (same semantics)
# mkey -> config; raw stats are per-90'd before rolling.
METRICS = {
    "gls": {"raw": ("gls", "npxg", "shots", "sot"), "applies_to": "OUT", "flip_sign": False, "bayes_alpha": {}},
    "ast": {"raw": ("ast", "xag"), "applies_to": "OUT", "flip_sign": False, "bayes_alpha": {}},
    "def": {"raw": ("blocks", "tkl", "int", "own_goals", "recoveries", "clr"),
             "applies_to": "OUT", "flip_sign": True, "bayes_alpha": {"own_goals": 6}},
    "gk" : {"raw": ("saves", "sot_against"),
             "applies_to": "GK", "flip_sign": True, "bayes_alpha": {}},
    "pens":{"raw": ("pk_won",), "applies_to": "OUT", "flip_sign": False, "bayes_alpha": {"pk_won": 6}},
}

# Composite groups built from z-scores of the above metric families
# Each group consumes the z-scores of selected rolled series; internal weights are equal by default.
GROUP_DEFINITION: Dict[str, List[Tuple[str, str]]] = {
    # (mkey, raw)
    "ATT": [("gls","gls"), ("gls","npxg"), ("gls","sot")],
    "CRE": [("ast","ast"), ("ast","xag")],
    "DEF": [("def","tkl"), ("def","int"), ("def","blocks"), ("def","recoveries")],  # own_goals already sign-flipped, excluded here
    "GK":  [("gk","gk_save_pct_p90")],  # special case naming; see below
}

# Default position → group weights (sum to 1 per position). CLI can override.
DEFAULT_POS_WEIGHTS = {
    "GK":  {"GK": 1.00},
    "DEF": {"DEF": 0.60, "ATT": 0.20, "CRE": 0.20},
    "MID": {"ATT": 0.30, "CRE": 0.40, "DEF": 0.30},
    "FWD": {"ATT": 0.60, "CRE": 0.30, "DEF": 0.10},
}

# ───────────────────────── Helpers ─────────────────────────

def _resolve_version(base_dir: Path, requested: Optional[str], auto: bool) -> str:
    if auto or (not requested) or (requested.lower() == "auto"):
        existing = [p.name for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        nxt = (max(int(s[1:]) for s in existing) + 1) if existing else 1
        ver = f"v{nxt}"
        logging.info("Auto-version resolved to %s", ver)
        return ver
    if not re.fullmatch(r"v\d+", requested):
        if requested.isdigit(): return f"v{requested}"
        raise ValueError(f"--version must be like v3 or a number; got {requested}")
    return requested

def _write_latest_pointer(features_root: Path, version: str) -> None:
    latest = features_root / "latest"
    target = features_root / version
    try:
        if latest.exists() or latest.is_symlink():
            try: latest.unlink()
            except Exception: pass
        os.symlink(target.name, latest, target_is_directory=True)
        logging.info("Updated 'latest' symlink -> %s", version)
    except (OSError, NotImplementedError):
        (features_root / "LATEST_VERSION.txt").write_text(version, encoding="utf-8")
        logging.info("Wrote LATEST_VERSION.txt -> %s", version)

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _z_by_season_gw(df: pd.DataFrame, col: str) -> pd.Series:
    grp = df.groupby(["season","gw_orig"])[col]
    mu  = grp.transform("mean")
    sd  = grp.transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).fillna(0.0)

def _z_by_season_gw_venue(df: pd.DataFrame, col: str, venue_value: str) -> pd.Series:
    mask = df["venue"].eq(venue_value)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    grp = df.loc[mask].groupby(["season","gw_orig"]) [col]
    mu  = grp.transform("mean")
    sd  = grp.transform("std").replace(0, np.nan)
    z   = ((df.loc[mask, col] - mu) / sd).fillna(0.0)
    out.loc[mask] = z
    out.loc[~mask] = np.nan
    return out

def _bulk_add_zscores_by_pos_gw(feat: pd.DataFrame, metrics_cfg=METRICS) -> pd.DataFrame:
    """
    Compute all z-score columns in bulk, per (season, gw_orig), with position
    partitioning: GK metrics use GK-only pool; outfield metrics use OUT-only pool.
    Defensive metrics are sign-flipped per METRICS[*]["flip_sign"].
    Also computes venue-conditional z-scores (Home/Away) within those pools.
    """
    def flip_needed(col: str) -> bool:
        # col looks like: "{mkey}_{raw}_p90[_home|_away]_roll"
        mkey = col.split("_", 1)[0]
        cfg = metrics_cfg.get(mkey, None)
        return bool(cfg and cfg.get("flip_sign", False))

    def applies_to(col: str) -> str:
        mkey = col.split("_", 1)[0]
        cfg = metrics_cfg.get(mkey, None)
        return cfg.get("applies_to", "OUT") if cfg else "OUT"

    pos_series = feat["pos"].astype(str).str.upper()
    mask_gk  = pos_series.eq("GK")
    mask_out = ~mask_gk

    # Identify roll columns
    overall_cols = [c for c in feat.columns if c.endswith("_roll") and "_home_" not in c and "_away_" not in c]
    home_cols    = [c for c in feat.columns if c.endswith("_home_roll")]
    away_cols    = [c for c in feat.columns if c.endswith("_away_roll")]

    gk_overall  = [c for c in overall_cols if applies_to(c) == "GK"]
    out_overall = [c for c in overall_cols if applies_to(c) != "GK"]
    gk_home     = [c for c in home_cols if applies_to(c) == "GK"]
    out_home    = [c for c in home_cols if applies_to(c) != "GK"]
    gk_away     = [c for c in away_cols if applies_to(c) == "GK"]
    out_away    = [c for c in away_cols if applies_to(c) != "GK"]

    newcols = {}

    def compute_z(mask: pd.Series, cols: List[str]) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(index=feat.index)
        sub = feat.loc[mask, cols]
        # group by (season, gw_orig) using aligned index
        keys = [feat.loc[mask, "season"], feat.loc[mask, "gw_orig"]]
        def _tx(col: pd.Series) -> pd.Series:
            mu = col.groupby(keys).transform("mean")
            sd = col.groupby(keys).transform("std").replace(0, np.nan)
            return ((col - mu) / sd).fillna(0.0)
        z = sub.apply(_tx, axis=0)
        return z

    # Overall
    z_gk_overall  = compute_z(mask_gk, gk_overall)
    z_out_overall = compute_z(mask_out, out_overall)
    for c in gk_overall:
        s = pd.Series(np.nan, index=feat.index, dtype=float); s.loc[mask_gk] = z_gk_overall[c]
        newcols[c + "_z"] = -s if flip_needed(c) else s
    for c in out_overall:
        s = pd.Series(np.nan, index=feat.index, dtype=float); s.loc[mask_out] = z_out_overall[c]
        newcols[c + "_z"] = -s if flip_needed(c) else s

    # Home
    mask_home = feat["venue"].astype(str).eq("Home")
    z_gk_home  = compute_z(mask_gk & mask_home, gk_home)
    z_out_home = compute_z(mask_out & mask_home, out_home)
    for c in gk_home:
        s = pd.Series(np.nan, index=feat.index, dtype=float); s.loc[mask_gk & mask_home] = z_gk_home[c]
        newcols[c + "_z"] = -s if flip_needed(c) else s
    for c in out_home:
        s = pd.Series(np.nan, index=feat.index, dtype=float); s.loc[mask_out & mask_home] = z_out_home[c]
        newcols[c + "_z"] = -s if flip_needed(c) else s

    # Away
    mask_away = feat["venue"].astype(str).eq("Away")
    z_gk_away  = compute_z(mask_gk & mask_away, gk_away)
    z_out_away = compute_z(mask_out & mask_away, out_away)
    for c in gk_away:
        s = pd.Series(np.nan, index=feat.index, dtype=float); s.loc[mask_gk & mask_away] = z_gk_away[c]
        newcols[c + "_z"] = -s if flip_needed(c) else s
    for c in out_away:
        s = pd.Series(np.nan, index=feat.index, dtype=float); s.loc[mask_out & mask_away] = z_out_away[c]
        newcols[c + "_z"] = -s if flip_needed(c) else s

    if newcols:
        zdf = pd.DataFrame(newcols, index=feat.index)
        feat = pd.concat([feat, zdf], axis=1)
        feat = feat.copy()
    return feat


def _applicable_mask(pos_series: pd.Series, applies_to: str) -> pd.Series:
    return pos_series.str.upper().eq("GK") if applies_to == "GK" else ~pos_series.str.upper().eq("GK")

# Rolling mean with venue shrinkage + prior blending (hard window)
def _rolling_past_only_bayes_mean(
    vals: np.ndarray, venues: np.ndarray, window: int, tau: float,
    prior_val: Optional[float], prior_matches: int, bayes_alpha: float = 0.0
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    n = len(vals)
    roll  = np.full(n, np.nan); roll_h = np.full(n, np.nan); roll_a = np.full(n, np.nan)
    v = vals.copy(); is_nan = np.isnan(v); v[is_nan] = 0.0
    csum = np.r_[0.0, np.cumsum(v)]; cnum = np.r_[0, np.cumsum(~is_nan)]
    ven = venues.astype(str)
    for i in range(n):
        lo = max(0, i - window)
        cnt = int(cnum[i] - cnum[lo]); base = np.nan
        if cnt > 0: base = (csum[i] - csum[lo]) / cnt
        if prior_val is not None and prior_matches > 0 and i < prior_matches:
            w = 1.0 - (i / prior_matches)
            base = prior_val if np.isnan(base) else (1 - w) * base + w * prior_val
        if (bayes_alpha > 0) and (not np.isnan(base)) and (prior_val is not None):
            base = (base * cnt + prior_val * bayes_alpha) / (cnt + bayes_alpha)
        roll[i] = base
        mask = (ven[lo:i] == "Home")
        rec = vals[lo:i]
        rec_h = rec[mask]; nh = int(np.sum(~np.isnan(rec_h)))
        rec_a = rec[~mask]; na = int(np.sum(~np.isnan(rec_a)))
        mean_h = np.nanmean(rec_h) if nh else base
        mean_a = np.nanmean(rec_a) if na else base
        if (bayes_alpha > 0) and (prior_val is not None):
            if nh and not np.isnan(mean_h): mean_h = (mean_h*nh + prior_val*bayes_alpha)/(nh+bayes_alpha)
            if na and not np.isnan(mean_a): mean_a = (mean_a*na + prior_val*bayes_alpha)/(na+bayes_alpha)
        lam_h = nh/(nh+tau) if (nh+tau)>0 else 0.0; lam_a = na/(na+tau) if (na+tau)>0 else 0.0
        roll_h[i] = lam_h*mean_h + (1-lam_h)*base if not np.isnan(base) else np.nan
        roll_a[i] = lam_a*mean_a + (1-lam_a)*base if not np.isnan(base) else np.nan
    return roll, roll_h, roll_a

# EWMA variant (past-only, halflife)
def _ewma_past_only_bayes_mean(
    vals: np.ndarray, venues: np.ndarray, halflife: float, tau: float,
    prior_val: Optional[float], prior_matches: int, bayes_alpha: float = 0.0
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    n = len(vals)
    v = vals.copy(); v = np.where(np.isnan(v), 0.0, v)
    ven = venues.astype(str)
    alpha = 1 - 2 ** (-1.0 / max(halflife, 1e-9))
    m_all = prior_val if prior_val is not None else 0.0
    m_h = m_all; m_a = m_all
    cnt_all = 0; cnt_h = 0; cnt_a = 0
    roll  = np.full(n, np.nan); roll_h = np.full(n, np.nan); roll_a = np.full(n, np.nan)
    for i in range(n):
        lam_h = cnt_h/(cnt_h+tau) if (cnt_h+tau)>0 else 0.0
        lam_a = cnt_a/(cnt_a+tau) if (cnt_a+tau)>0 else 0.0
        roll[i]  = m_all
        roll_h[i]= lam_h*m_h + (1-lam_h)*m_all
        roll_a[i]= lam_a*m_a + (1-lam_a)*m_all
        # update state with current obs
        x = v[i]
        m_all = (1-alpha)*m_all + alpha*x; cnt_all += 1
        if ven[i] == "Home":
            m_h = (1-alpha)*m_h + alpha*x; cnt_h += 1
        else:
            m_a = (1-alpha)*m_a + alpha*x; cnt_a += 1
        if prior_val is not None and cnt_all <= prior_matches:
            w = 1.0 - (cnt_all / max(1, prior_matches))
            m_all = (1-w)*m_all + w*prior_val
            m_h   = (1-w)*m_h   + w*prior_val
            m_a   = (1-w)*m_a   + w*prior_val
        # optional extra shrink to prior each step (alpha-like); skipped here; bayes_alpha retained for symmetry
    return roll, roll_h, roll_a

# Beta-binomial posterior save% with venue shrinkage
def _rolling_past_only_binomial_savepct(
    saves: np.ndarray, shots: np.ndarray, venues: np.ndarray,
    window: int, tau: float, prior_p: float, prior_shots: float
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    n = len(saves)
    post  = np.full(n, np.nan); post_h = np.full(n, np.nan); post_a = np.full(n, np.nan)
    sv = np.nan_to_num(saves.astype(float), nan=0.0); st = np.nan_to_num(shots.astype(float), nan=0.0)
    csum_sv = np.r_[0.0, np.cumsum(sv)]; csum_st = np.r_[0.0, np.cumsum(st)]; ven = venues.astype(str)
    a0 = prior_p * prior_shots; b0 = (1.0 - prior_p) * prior_shots
    for i in range(n):
        lo = max(0, i - window)
        S = csum_sv[i] - csum_sv[lo]; N = csum_st[i] - csum_st[lo]
        post_overall = (S + a0) / (N + a0 + b0) if (N + a0 + b0) > 0 else prior_p
        post[i] = post_overall
        mask = (ven[lo:i] == "Home")
        S_h = np.nansum(saves[lo:i][mask]); N_h = np.nansum(shots[lo:i][mask])
        S_a = np.nansum(saves[lo:i][~mask]); N_a = np.nansum(shots[lo:i][~mask])
        post_home = (S_h + a0) / (N_h + a0 + b0) if (N_h + a0 + b0) > 0 else post_overall
        post_away = (S_a + a0) / (N_a + a0 + b0) if (N_a + a0 + b0) > 0 else post_overall
        nh = int(np.sum(~np.isnan(shots[lo:i][mask]))); na = int(np.sum(~np.isnan(shots[lo:i][~mask])))
        lam_h = nh/(nh+tau) if (nh+tau)>0 else 0.0; lam_a = na/(na+tau) if (na+tau)>0 else 0.0
        post_h[i] = lam_h*post_home + (1-lam_h)*post_overall
        post_a[i] = lam_a*post_away + (1-lam_a)*post_overall
    return post, post_h, post_a

# EWMA version of the above (approximate sequential beta posterior with EMA on rates)
def _ewma_binomial_savepct(
    saves: np.ndarray, shots: np.ndarray, venues: np.ndarray,
    halflife: float, tau: float, prior_p: float, prior_shots: float
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    n = len(saves)
    post  = np.full(n, np.nan); post_h = np.full(n, np.nan); post_a = np.full(n, np.nan)
    sv = np.nan_to_num(saves.astype(float), nan=0.0); st = np.nan_to_num(shots.astype(float), nan=0.0)
    alpha = 1 - 2 ** (-1.0 / max(halflife, 1e-9))
    ven = venues.astype(str)
    # Start with prior as pseudo-rate
    p_all = prior_p; p_h = prior_p; p_a = prior_p
    cnt_all = 0; cnt_h = 0; cnt_a = 0
    warmup_matches = int(prior_shots) if prior_shots is not None else 0
    for i in range(n):
        lam_h = cnt_h/(cnt_h+tau) if (cnt_h+tau)>0 else 0.0
        lam_a = cnt_a/(cnt_a+tau) if (cnt_a+tau)>0 else 0.0
        post[i]  = p_all
        post_h[i]= lam_h*p_h + (1-lam_h)*p_all
        post_a[i]= lam_a*p_a + (1-lam_a)*p_all
        # Update estimate using observed save rate for this match if shots>0, else decay
        x = sv[i]; N = st[i]
        obs = (x / N) if N > 0 else p_all
        p_all = (1-alpha)*p_all + alpha*obs; cnt_all += 1
        if ven[i] == "Home":
            p_h = (1-alpha)*p_h + alpha*obs; cnt_h += 1
        else:
            p_a = (1-alpha)*p_a + alpha*obs; cnt_a += 1
        if cnt_all <= warmup_matches:
                    # reuse prior_shots as a soft warm-up horizon to blend prior strongly
            w = 1.0 - (cnt_all / max(1, warmup_matches))
            p_all = (1-w)*p_all + w*prior_p
            p_h   = (1-w)*p_h   + w*prior_p
            p_a   = (1-w)*p_a   + w*prior_p
    return post, post_h, post_a


def _compute_last_season_priors(all_players: pd.DataFrame, last_season: Optional[str]) -> Dict[str, Dict[str, float]]:
    priors: Dict[str, Dict[str, float]] = {}
    if not last_season: return priors
    prev = all_players.loc[all_players["season"] == last_season].copy()
    if prev.empty: return priors
    prev["minutes"] = pd.to_numeric(prev["minutes"], errors="coerce")
    # per-90 baselines
    for cfg in METRICS.values():
        for raw in cfg["raw"]:
            col = f"{raw}_p90"
            prev[col] = np.where(prev["minutes"] > 0, prev[raw] * 90.0 / prev["minutes"], np.nan)
    # GK save% priors
    gk_prev = prev[prev["pos"] == "GK"].copy()
    gk_aggr = gk_prev.groupby("player_id").agg(saves_sum=("saves","sum"), sot_sum=("sot_against","sum"))
    gk_aggr["p0"] = np.where(gk_aggr["sot_sum"] > 0, gk_aggr["saves_sum"]/gk_aggr["sot_sum"], np.nan)
    gk_aggr["s0"] = gk_aggr["sot_sum"].clip(lower=20)

    # positional/global per-90 priors
    is_gk_prev = prev["pos"].eq("GK")
    pos_means: Dict[str, Dict[str, float]] = {"GK": {}, "OUT": {}}
    global_means: Dict[str, float] = {}
    for cfg in METRICS.values():
        for raw in cfg["raw"]:
            col = f"{raw}_p90"
            pos_means["GK"][col]  = prev.loc[is_gk_prev, col].mean()
            pos_means["OUT"][col] = prev.loc[~is_gk_prev, col].mean()
            global_means[col]     = prev[col].mean()

    p0_pos = (gk_aggr["p0"].mean() if not gk_aggr["p0"].dropna().empty else 0.70)
    s0_pos = (gk_aggr["s0"].mean() if not gk_aggr["s0"].dropna().empty else 80.0)

    per_player = prev.groupby("player_id").mean(numeric_only=True)
    for pid, row in per_player.iterrows():
        priors[str(pid)] = {}
        for cfg in METRICS.values():
            for raw in cfg["raw"]:
                priors[str(pid)][f"{raw}_p90"] = row.get(f"{raw}_p90", np.nan)

    for pid, r in gk_aggr.iterrows():
        d = priors.setdefault(str(pid), {})
        d["save_pct_p0"] = r["p0"]; d["save_pct_s0"] = r["s0"]

    priors["_POS_GK_"]  = pos_means["GK"]
    priors["_POS_OUT_"] = pos_means["OUT"]
    priors["_GLOBAL_"]  = global_means
    priors["_SAVE_PCT_"] = {"p0": p0_pos, "s0": s0_pos}
    return priors


def _get_prior_p90(priors: Dict[str, Dict[str, float]], pid: str, raw: str, is_gk: bool) -> Optional[float]:
    key = f"{raw}_p90"
    v = priors.get(pid, {}).get(key)
    if v is not None and not pd.isna(v):
        return float(v)
    pos_key = "_POS_GK_" if is_gk else "_POS_OUT_"
    v2 = priors.get(pos_key, {}).get(key)
    if v2 is not None and not pd.isna(v2):
        return float(v2)
    v3 = priors.get("_GLOBAL_", {}).get(key)
    if v3 is not None and not pd.isna(v3):
        return float(v3)
    return None


def _get_savepct_prior(priors: Dict[str, Dict[str, float]], pid: str) -> Tuple[float,float]:
    d = priors.get(pid, {})
    p0, s0 = d.get("save_pct_p0"), d.get("save_pct_s0")
    if (p0 is not None) and (not pd.isna(p0)) and (s0 is not None) and (s0 > 0): return float(p0), float(s0)
    grp = priors.get("_SAVE_PCT_", {})
    return float(grp.get("p0", 0.70)), float(grp.get("s0", 80.0))

# ───────────────────────── Schema coercion ─────────────────────────

def _normalize_pos_label(s: pd.Series) -> pd.Series:
    # Map various forms to {GK, DEF, MID, FWD}
    m = {
        "gk":"GK","goalkeeper":"GK",
        "d":"DEF","def":"DEF","defender":"DEF",
        "m":"MID","mid":"MID","midfielder":"MID",
        "f":"FWD","fw":"FWD","fwd":"FWD","forward":"FWD","striker":"FWD",
    }
    x = s.astype(str).str.strip().str.lower().map(m).fillna(s.astype(str).str.upper())
    # fallback: any non-GK and not one of DEF/MID/FWD → treat as OUT but preserve original
    x = x.where(x.isin(["GK","DEF","MID","FWD"]), s.astype(str).str.upper())
    return x


def _coerce_columns(df: pd.DataFrame, fill_missing_fdr: Optional[float]) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Map aliases → canonical, derive missing canonical fields, coerce types, and fill safe defaults.
    Returns the coerced DataFrame + a small stats dict for logging.
    """
    stats = {"filled_days_since_last": 0, "derived_save_pct": 0, "venue_from_flag": 0, "filled_fdr": 0}

    # 1) Rename aliases
    rename_map = {}
    for canon, alts in ALIASES.items():
        if canon in df.columns:
            continue
        for alt in alts:
            if alt in df.columns:
                rename_map[alt] = canon
                break
    df = df.rename(columns=rename_map).copy()

    # 2) Basic keys & types
    if "season" not in df.columns:
        raise KeyError("Input must have a 'season' column per row (builder adds it upstream; bug if missing).")

    # date
    if "date_played" in df.columns:
        df["date_played"] = pd.to_datetime(df["date_played"], errors="coerce")
    else:
        raise KeyError("Could not coerce a 'date_played' column from input (check aliases).")

    # venue normalization
    if "venue" not in df.columns:
        df["venue"] = pd.NA
    if df["venue"].dropna().isin([0,1,True,False]).any():
        mask = df["venue"].notna()
        home_like = df.loc[mask, "venue"].astype(str).isin(["1","True","true","TRUE"])
        df.loc[mask, "venue"] = np.where(home_like, "Home", "Away")
        stats["venue_from_flag"] += int(mask.sum())
    df.loc[~df["venue"].isin(["Home","Away"]), "venue"] = pd.NA

    # gw
    if "gw_orig" not in df.columns:
        raise KeyError("Missing 'gw_orig' (gameweek). Add an alias to ALIASES if your column is named differently.")
    df["gw_orig"] = pd.to_numeric(df["gw_orig"], errors="coerce").astype("Int64")

    # strings
    for c in ["player_id","team_id","opponent_id","fbref_id","fpl_id","player","team","pos"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "pos" in df.columns:
        df["pos"] = _normalize_pos_label(df["pos"])  # normalize to GK/DEF/MID/FWD if possible

    # 3) Numeric coercions
    _ensure_numeric(df, NUMERIC_BASE)
    _ensure_numeric(df, NUMERIC_ENRICH)

    # Derive save_pct if missing/NaN and ingredients present
    if "save_pct" in df.columns:
        need = df["save_pct"].isna()
    else:
        df["save_pct"] = np.nan
        need = pd.Series(True, index=df.index)
    can = need & df["saves"].notna() & df["sot_against"].notna() & (df["sot_against"] > 0)
    df.loc[can, "save_pct"] = (df.loc[can, "saves"] / df.loc[can, "sot_against"]).clip(0, 1.0)
    stats["derived_save_pct"] = int(can.sum())

    # days_since_last if missing
    if "days_since_last" not in df.columns or df["days_since_last"].isna().any():
        df = df.sort_values(["player_id","date_played","gw_orig"])
        grp = df.groupby("player_id", sort=False)["date_played"]
        dsl = grp.diff().dt.days
        df["days_since_last"] = dsl
        fill_mask = df["days_since_last"].isna()
        stats["filled_days_since_last"] = int(fill_mask.sum())
        df.loc[fill_mask, "days_since_last"] = 7.0  # neutral fallback
    df["days_since_last"] = pd.to_numeric(df["days_since_last"], errors="coerce")

    # is_active default 1 if missing
    if "is_active" not in df.columns:
        df["is_active"] = 1.0
    df["is_active"] = pd.to_numeric(df["is_active"], errors="coerce").fillna(1.0)

    # FDR fill for future fixtures (optional)
    if fill_missing_fdr is not None:
        for col in ["fdr_home","fdr_away"]:
            if col not in df.columns:
                df[col] = fill_missing_fdr
                stats["filled_fdr"] += len(df)
            else:
                miss = df[col].isna()
                if miss.any():
                    df.loc[miss, col] = fill_missing_fdr
                    stats["filled_fdr"] += int(miss.sum())

    # Defaults for very rare events / optional cols (safe zeros)
    safe_zero = ["pkatt","pk_scored","pk_won","own_goals","recoveries","blocks","tkl","int","clr",
                 "gls","ast","shots","sot","xg","npxg","xag","saves","sot_against","yellow_crd","red_crd","gf","ga"]
    for c in safe_zero:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Deduplicate (rare but can happen with joins)
    key_cols = [c for c in ["player_id","season","date_played","gw_orig"] if c in df.columns]
    if key_cols:
        before = len(df)
        df = df.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last")
        dropped = before - len(df)
        if dropped:
            logging.warning("Dropped %d duplicate rows on %s", dropped, key_cols)

    return df.reset_index(drop=True), stats

# ───────────────────────── Composite helpers ─────────────────────────

def _parse_pos_weights(s: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not s:
        return DEFAULT_POS_WEIGHTS
    # Format: "GK:gk=1;DEF:def=.6,att=.2,cre=.2;MID:att=.4,cre=.4,def=.2;FWD:att=.6,cre=.3,def=.1"
    out: Dict[str, Dict[str, float]] = {"GK":{},"DEF":{},"MID":{},"FWD":{}}
    for block in s.split(";"):
        block = block.strip()
        if not block: continue
        if ":" not in block: continue
        pos, rest = block.split(":", 1)
        pos = pos.strip().upper()
        if pos not in out: continue
        wmap: Dict[str,float] = {}
        for kv in rest.split(","):
            kv = kv.strip()
            if not kv or "=" not in kv: continue
            g, val = kv.split("=", 1)
            g = g.strip().upper()
            try:
                wmap[g] = float(val)
            except Exception:
                pass
        if wmap:
            # normalize to sum 1
            ssum = sum(wmap.values())
            if ssum > 0:
                for k in list(wmap.keys()): wmap[k] = wmap[k]/ssum
            out[pos] = wmap
    # fill missing with defaults
    for pos in out:
        if not out[pos]:
            out[pos] = DEFAULT_POS_WEIGHTS.get(pos, {})
    return out

def _parse_halflife_by_pos(s: Optional[str], default_hl: float) -> Dict[str, float]:
    """
    Parse --halflife-by-pos like 'GK:4,DEF:3.5,MID:3,FWD:2.5'.
    Missing/invalid entries fall back to default_hl.
    """
    out = {"GK": default_hl, "DEF": default_hl, "MID": default_hl, "FWD": default_hl}
    if not s:
        return out
    for kv in s.split(","):
        kv = kv.strip()
        if not kv or ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        k = k.strip().upper()
        if k in out:
            try:
                out[k] = float(v)
            except Exception:
                pass
    return out


def _composite_from_groups(feat: pd.DataFrame, pos_weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = feat.copy()

    def _avg_of_cols(cols: List[str]) -> pd.Series:
        if not cols: return pd.Series(np.nan, index=df.index)
        arr = df[cols].astype(float)
        return arr.mean(axis=1, skipna=True)

    def zcol(mkey: str, raw: str, suffix: str) -> str:
        # Normal rolled z columns are f"{mkey}_{raw}_p90{suffix}_roll_z" except the GK save% which we named 'gk_save_pct_p90'
        return f"{mkey}_{raw}_p90{suffix}_roll_z"

    # Build group-level z means (overall/home/away)
    group_cols = {}
    for grp, members in GROUP_DEFINITION.items():
        for suffix, tag in [("", grp.lower()), ("_home", grp.lower()+"_home"), ("_away", grp.lower()+"_away")]:
            cols = []
            for (mkey, raw) in members:
                # special-case GK save%
                if grp == "GK" and raw == "gk_save_pct_p90":
                    col = f"gk_save_pct_p90{suffix}_roll_z"  # produced by z-bulk on the rolled save% series
                else:
                    col = zcol(mkey, raw, suffix)
                if col in df.columns:
                    cols.append(col)
            group_cols[(grp, suffix)] = cols
            df[f"form_{tag}_z"] = _avg_of_cols(cols)

    # Composite per position using weights
    pos_norm = _normalize_pos_label(df["pos"]) if "pos" in df.columns else pd.Series("OUT", index=df.index)

    def _score_row(i: int, suffix: str) -> float:
        pos = pos_norm.iat[i]
        w = pos_weights.get(pos, DEFAULT_POS_WEIGHTS.get(pos, {}))
        total = 0.0; got = 0.0
        for grp, weight in w.items():
            col = f"form_{grp.lower()}{suffix}_z"
            val = df.iloc[i][col] if col in df.columns else np.nan
            if not pd.isna(val):
                total += weight * float(val)
                got += weight
        if got == 0: return np.nan
        return total  # weights already sum≈1

    for suffix, label in [("", "form_score_z"), ("_home", "form_score_home_z"), ("_away", "form_score_away_z")]:
        df[label] = [ _score_row(i, suffix) for i in range(len(df)) ]

    return df

# ───────────────────────── Core build ─────────────────────────

def build_player_form(
    seasons: List[str],
    fixtures_root: Path,
    version_dir: Path,
    window: int,
    tau: float,
    force: bool,
    prior_matches: int,
    fill_missing_fdr: Optional[float],
    use_ewma: bool,
    halflife: float,
    pos_weights: Dict[str, Dict[str, float]],
    halflife_by_pos: Dict[str, float],
) -> None:
    frames = []
    for s in seasons:
        fp = fixtures_root / s / "player_fixture_calendar.csv"
        if not fp.is_file():
            logging.warning("%s • missing player_fixture_calendar.csv – skipped", s); continue
        try:
            df = pd.read_csv(fp, parse_dates=["date_played"], low_memory=False)
        except Exception:
            df = pd.read_csv(fp, low_memory=False)
        df["season"] = s
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No seasons loaded from fixtures_root")

    raw_all = pd.concat(frames, ignore_index=True)
    # Coerce schema & derive critical fields
    all_players, stats = _coerce_columns(raw_all, fill_missing_fdr=fill_missing_fdr)

    unknown_venue = all_players["venue"].isna().sum()
    if unknown_venue:
        logging.warning("Rows with unknown venue: %d (venue-z unavailable for those rows)", unknown_venue)

    # Enforce GK/outfield raw NaNs to avoid leakage
    gk_only_raw   = ["saves","sot_against","save_pct"]
    out_only_raw  = ["gls","ast","shots","sot","xg","npxg","xag","pkatt","pk_scored","pk_won",
                     "blocks","tkl","int","own_goals","recoveries","clr"]
    is_gk_all = all_players["pos"].eq("GK")
    all_players.loc[~is_gk_all, [c for c in gk_only_raw if c in all_players.columns]] = np.nan
    all_players.loc[ is_gk_all, [c for c in out_only_raw if c in all_players.columns]] = np.nan

    # Season loop (priors use last season)
    seasons = sorted(seasons)
    for season in seasons:
        parts = season.split("-"); last_season = None
        if len(parts) == 2:
            y0 = int(parts[0]); last_season = f"{y0-1}-{y0}"

        priors = _compute_last_season_priors(all_players, last_season)
        cur = all_players[all_players["season"] == season].sort_values(
            ["player_id","date_played","gw_orig"]
        ).copy()

        # Per-90 base
        for cfg in METRICS.values():
            mask_app = _applicable_mask(cur["pos"], cfg["applies_to"])
            for raw in cfg["raw"]:
                col = f"{raw}_p90"
                cur[col] = np.where(cur["minutes"] > 0, cur[raw] * 90.0 / cur["minutes"], np.nan)
                cur.loc[~mask_app, col] = np.nan

        # Rolling features per player (past-only)
        out_rows: List[pd.DataFrame] = []
        for pid, g in cur.groupby("player_id", sort=False):
            g = g.sort_values(["date_played","gw_orig"]).copy()
            pos_label = _normalize_pos_label(g["pos"]).mode().iat[0] if len(g) else "MID"
            is_gk = (pos_label == "GK")
            # Halflife can vary by position; fallback to global --halflife
            hl = halflife_by_pos.get(pos_label, halflife)

            for mkey, cfg in METRICS.items():
                applies_mask = _applicable_mask(g["pos"], cfg["applies_to"]).to_numpy()
                for raw in cfg["raw"]:
                    col = f"{raw}_p90"; arr = np.where(applies_mask, g[col].to_numpy(), np.nan)
                    prior_val = _get_prior_p90(priors, str(pid), raw, is_gk)
                    alpha = float(cfg.get("bayes_alpha", {}).get(raw, 0.0))
                    if use_ewma:
                        roll, rh, ra = _ewma_past_only_bayes_mean(
                            vals=arr, venues=g["venue"].astype(str).to_numpy(),
                            halflife=hl, tau=tau, prior_val=prior_val, prior_matches=prior_matches, bayes_alpha=alpha
                        )
                    else:
                        roll, rh, ra = _rolling_past_only_bayes_mean(
                            vals=arr, venues=g["venue"].astype(str).to_numpy(),
                            window=window, tau=tau, prior_val=prior_val, prior_matches=prior_matches, bayes_alpha=alpha
                        )
                    base = f"{mkey}_{raw}_p90"
                    g[f"{base}_roll"] = roll; g[f"{base}_home_roll"] = rh; g[f"{base}_away_roll"] = ra

            # GK save% posterior rolling
            if is_gk:
                p0, s0 = _get_savepct_prior(priors, str(pid))
                if use_ewma:
                    post, post_h, post_a = _ewma_binomial_savepct(
                        saves=g["saves"].to_numpy(), shots=g["sot_against"].to_numpy(),
                        venues=g["venue"].astype(str).to_numpy(), halflife=hl, tau=tau,
                        prior_p=p0 if not np.isnan(p0) else 0.70, prior_shots=s0 if (s0 and not np.isnan(s0)) else 80.0
                    )
                else:
                    post, post_h, post_a = _rolling_past_only_binomial_savepct(
                        saves=g["saves"].to_numpy(), shots=g["sot_against"].to_numpy(),
                        venues=g["venue"].astype(str).to_numpy(), window=window, tau=tau,
                        prior_p=p0 if not np.isnan(p0) else 0.70, prior_shots=s0 if (s0 and not np.isnan(s0)) else 80.0
                    )
                g["gk_save_pct_p90_roll"] = post
                g["gk_save_pct_p90_home_roll"] = post_h
                g["gk_save_pct_p90_away_roll"] = post_a
            else:
                g["gk_save_pct_p90_roll"] = np.nan
                g["gk_save_pct_p90_home_roll"] = np.nan
                g["gk_save_pct_p90_away_roll"] = np.nan

            out_rows.append(g)

        feat = pd.concat(out_rows, ignore_index=True)

        # Z-scores (overall + venue-conditional), sign flips for defensive negatives — single bulk call
        feat = _bulk_add_zscores_by_pos_gw(feat)

        # Composite form (group z-averages + position-weighted score)
        feat = _composite_from_groups(feat, pos_weights=pos_weights)

        # Write
        out_dir_season = version_dir / season
        out_dir_season.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir_season / OUTPUT_FILE

        if "date_played" in feat.columns:
            feat["date_played"] = pd.to_datetime(feat["date_played"], errors="coerce")

        if out_fp.exists() and not force:
            logging.info("%s • %s exists – skip (use --force)", season, OUTPUT_FILE)
        else:
            feat.to_csv(out_fp, index=False, date_format="%Y-%m-%d")
            meta = {
                "schema": SCHEMA_VERSION,
                "build_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "version": version_dir.name,
                "window_matches": window,
                "tau": tau,
                "prior_matches": prior_matches,
                "per90": True,
                "ewma": use_ewma,
                "halflife": halflife if use_ewma else None,
                "halflife_by_pos": halflife_by_pos if use_ewma else None,
                "savepct_binomial": {"prior": "last_season (player) → GK_group → global",
                                     "prior_p_default": 0.70, "prior_s_default": 80.0,
                                     "mode": "ewma" if use_ewma else "window"},
                "rare_event_shrinkage_alpha": {"pk_won": 6, "own_goals": 6},
                "zscore_mode": {"overall": "season,gw_orig",
                                  "home": "season,gw_orig (Home only)",
                                  "away": "season,gw_orig (Away only)"},
                "groups": {
                    "definition": {k:[{"mkey":a,"raw":b} for (a,b) in v] for k,v in GROUP_DEFINITION.items()},
                    "pos_weights": pos_weights,
                },
                "features": sorted([c for c in feat.columns if c.endswith(("_roll","_roll_z"))] +
                                    [c for c in feat.columns if c.startswith("form_")]),
                "coercion_stats": stats,
            }
            (out_dir_season / "player_form.meta.json").write_text(json.dumps(meta, indent=2))
            logging.info("%s • %s (%d rows) written", season, OUTPUT_FILE, len(feat))

# ───────────────────────── Batch / CLI ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", help="e.g. 2025-2026; omit for batch over all folders under --fixtures-root")
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/registry/fixtures"),
                    help="Root containing <SEASON>/player_fixture_calendar.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/registry/features"),
                    help="Root for versioned features output")

    # Versioning
    ap.add_argument("--version", default=None, help="Version folder (e.g., v3). If omitted with --auto-version, next vN is used.")
    ap.add_argument("--auto-version", action="store_true", help="Pick the next vN under out-dir automatically.")
    ap.add_argument("--write-latest", action="store_true", help="Update features/latest to point to the resolved version.")

    # Rolling params
    ap.add_argument("--window", type=int, default=5, help="rolling window (matches, past-only) for classic mode")
    ap.add_argument("--tau", type=float, default=2.0, help="venue shrinkage strength")
    ap.add_argument("--prior-matches", type=int, default=6, help="first K matches blend prior → 0")
    ap.add_argument("--ewma", action="store_true", help="use EWMA past-only rolling instead of hard window")
    ap.add_argument("--halflife", type=float, default=3.0, help="EWMA halflife in matches (global fallback)")
    ap.add_argument("--halflife-by-pos", type=str, default=None,
                    help="Override EWMA halflife per position, e.g. 'GK:4,DEF:3.5,MID:3,FWD:2.5' (falls back to --halflife)")
    # Composite weights
    ap.add_argument("--pos-weights", type=str, default=None,
                    help="Per-position group weights; e.g. 'GK:gk=1;DEF:def=.6,att=.2,cre=.2;MID:att=.4,cre=.4,def=.2;FWD:att=.6,cre=.3,def=.1'")

    ap.add_argument("--fill-missing-fdr", type=float, default=None, help="If set, fill missing fdr_home/fdr_away with this value (e.g., 3.0 for neutral).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=args.log_level.upper())

    # seasons list
    if args.season:
        seasons = [args.season]
    else:
        seasons = sorted(d.name for d in args.fixtures_root.iterdir() if d.is_dir())
    if not seasons:
        logging.error("No season folders in %s", args.fixtures_root); return
    seasons = sorted(seasons)

    # Resolve version directory once
    features_root = args.out_dir
    features_root.mkdir(parents=True, exist_ok=True)
    version = _resolve_version(features_root, args.version, args.auto_version)
    version_dir = features_root / version
    version_dir.mkdir(parents=True, exist_ok=True)

    pos_weights = _parse_pos_weights(args.pos_weights)
    halflife_by_pos = _parse_halflife_by_pos(args.halflife_by_pos, args.halflife)

    logging.info("Processing seasons: %s", ", ".join(seasons))
    logging.info("Writing to version dir: %s", version_dir)

    build_player_form(
        seasons=seasons,
        fixtures_root=args.fixtures_root,
        version_dir=version_dir,
        window=args.window,
        tau=args.tau,
        force=args.force,
        prior_matches=args.prior_matches,
        fill_missing_fdr=args.fill_missing_fdr,
        use_ewma=args.ewma,
        halflife=args.halflife,
        pos_weights=pos_weights,
        halflife_by_pos=halflife_by_pos,
    )

    if args.write_latest:
        _write_latest_pointer(features_root, version)

if __name__ == "__main__":
    main()
