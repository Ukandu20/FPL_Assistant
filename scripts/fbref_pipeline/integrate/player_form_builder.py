#!/usr/bin/env python3
r"""player_form_builder.py – schema v1.5
(per-90, causal, venue-aware, binomial save% using saves & sot_against, venue-conditional z-scores)
Versioning:
• Writes under data/processed/features/<version>/<SEASON>/players_form.csv
• --auto-version chooses next vN if not specified, e.g., v7
• --write-latest adds a 'latest' pointer (symlink if supported; else LATEST_VERSION.txt)

Outputs (per season):
  players_form.csv
  player_form.meta.json
"""

from __future__ import annotations
import argparse, json, logging, datetime as dt, os, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

SCHEMA_VERSION = "v1.5"

# ───────────────────────── Configuration ─────────────────────────

METRICS = {
    "gls": {"raw": ("gls", "npxg", "shots", "sot"), "rate_like": {"gls": False, "npxg": False, "shots": False, "sot": False},
            "applies_to": "OUT", "flip_sign": False, "bayes_alpha": {}},
    "ast": {"raw": ("ast", "xag"), "rate_like": {"ast": False, "xag": False},
            "applies_to": "OUT", "flip_sign": False, "bayes_alpha": {}},
    "def": {"raw": ("blocks", "tkl", "int", "own_goals", "recoveries", "clr"),
            "rate_like": {"blocks": False, "tkl": False, "int": False, "own_goals": False, "recoveries": False, "clr": False},
            "applies_to": "OUT", "flip_sign": True, "bayes_alpha": {"own_goals": 6}},
    "gk":  {"raw": ("saves", "sot_against"),
            "rate_like": {"saves": False, "sot_against": False},
            "applies_to": "GK", "flip_sign": True, "bayes_alpha": {}},
    "pens":{"raw": ("pk_won",), "rate_like": {"pk_won": False},
            "applies_to": "OUT", "flip_sign": False, "bayes_alpha": {"pk_won": 6}},
    # GK save% handled via binomial (not per-90)
}

OUTPUT_FILE = "players_form.csv"

REQUIRED_BASE = {
    "player_id","player","pos","fbref_id","fpl_id",
    "gw_orig","date_played","team_id","team","minutes","days_since_last",
    "is_active","yellow_crd","red_crd","venue","gf","ga","fdr_home","fdr_away",
    "gls", "shots", "sot", "ast","blocks","tkl","int","clr","xg","npxg","xag",
    "pkatt","pk_scored","pk_won",
    "saves","sot_against","save_pct",
    "own_goals","recoveries",
}

NUMERIC_BASE = [
    "minutes","days_since_last","is_active","yellow_crd","red_crd",
    "gf","ga","shots", "sot","fdr_home","fdr_away",
    "gls","ast","blocks","tkl","int","clr","xg","npxg","xag",
    "pkatt","pk_scored","pk_won",
    "saves","sot_against","save_pct",
    "own_goals","recoveries",
]

# ───────────────────────── Helpers ─────────────────────────

def _resolve_version(base_dir: Path, requested: Optional[str], auto: bool) -> str:
    if auto or (not requested) or (requested.lower() == "auto"):
        existing = [p.name for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        if existing:
            nums = sorted(int(s[1:]) for s in existing)
            nxt = nums[-1] + 1
        else:
            nxt = 1
        ver = f"v{nxt}"
        logging.info("Auto-version resolved to %s", ver)
        return ver
    # normalize: ensure a leading 'v'
    if not re.fullmatch(r"v\d+", requested):
        if requested.isdigit():
            requested = f"v{requested}"
        else:
            raise ValueError(f"--version must be like v3 or a number; got {requested}")
    return requested

def _write_latest_pointer(features_root: Path, version: str) -> None:
    latest = features_root / "latest"
    target = features_root / version
    try:
        if latest.exists() or latest.is_symlink():
            if latest.is_dir() and not latest.is_symlink():
                # replace dir with symlink when possible
                for _ in (latest.iterdir() if latest.is_dir() else []): pass
            latest.unlink(missing_ok=True)
        os.symlink(target.name, latest, target_is_directory=True)
        logging.info("Updated 'latest' symlink -> %s", version)
    except (OSError, NotImplementedError):
        # Windows without admin can’t symlink by default; write a file instead
        (features_root / "LATEST_VERSION.txt").write_text(version, encoding="utf-8")
        logging.info("Wrote LATEST_VERSION.txt -> %s", version)

def load_json(p: Path) -> dict:
    return json.loads(p.read_text("utf-8")) if p.is_file() else {}

def save_json(obj: dict, p: Path) -> None:
    p.write_text(json.dumps(obj, indent=2))

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
    grp = df.loc[mask].groupby(["season","gw_orig"])[col]
    mu  = grp.transform("mean")
    sd  = grp.transform("std").replace(0, np.nan)
    z   = ((df.loc[mask, col] - mu) / sd).fillna(0.0)
    out.loc[mask] = z
    out.loc[~mask] = np.nan
    return out

def _applicable_mask(pos_series: pd.Series, applies_to: str) -> pd.Series:
    return pos_series.eq("GK") if applies_to == "GK" else ~pos_series.eq("GK")

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
        lo = max(0, i - window)  # exclude current
        cnt = int(cnum[i] - cnum[lo]); base = np.nan
        if cnt > 0: base = (csum[i] - csum[lo]) / cnt
        if prior_val is not None and prior_matches > 0 and i < prior_matches:
            w = 1.0 - (i / prior_matches)
            base = prior_val if np.isnan(base) else (1 - w) * base + w * prior_val
        if (bayes_alpha > 0) and (not np.isnan(base)) and (prior_val is not None):
            base = (base * cnt + prior_val * bayes_alpha) / (cnt + bayes_alpha)
        roll[i] = base
        mask = (ven[lo:i] == "Home"); rec = vals[lo:i]
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

def _rolling_past_only_binomial_savepct(
    saves: np.ndarray, shots: np.ndarray, venues: np.ndarray,
    window: int, tau: float, prior_p: Optional[float], prior_shots: Optional[float]
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    n = len(saves)
    post  = np.full(n, np.nan); post_h = np.full(n, np.nan); post_a = np.full(n, np.nan)
    sv = np.nan_to_num(saves.astype(float), nan=0.0); st = np.nan_to_num(shots.astype(float), nan=0.0)
    csum_sv = np.r_[0.0, np.cumsum(sv)]; csum_st = np.r_[0.0, np.cumsum(st)]; ven = venues.astype(str)
    if prior_p is None or prior_shots is None or prior_shots <= 0: prior_p, prior_shots = 0.70, 80.0
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

def _compute_last_season_priors(all_players: pd.DataFrame, last_season: Optional[str]) -> Dict[str, Dict[str, float]]:
    priors: Dict[str, Dict[str, float]] = {}
    if not last_season: return priors
    prev = all_players.loc[all_players["season"] == last_season].copy()
    if prev.empty: return priors
    prev["minutes"] = pd.to_numeric(prev["minutes"], errors="coerce")
    for cfg in METRICS.values():
        for raw in cfg["raw"]:
            col = f"{raw}_p90"
            prev[col] = np.where(prev["minutes"] > 0, prev[raw] * 90.0 / prev["minutes"], np.nan)
    gk_prev = prev[prev["pos"] == "GK"].copy()
    gk_aggr = gk_prev.groupby("player_id").agg(saves_sum=("saves","sum"), sot_sum=("sot_against","sum"))
    gk_aggr["p0"] = np.where(gk_aggr["sot_sum"] > 0, gk_aggr["saves_sum"]/gk_aggr["sot_sum"], np.nan)
    gk_aggr["s0"] = gk_aggr["sot_sum"].clip(lower=20)

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
    if not pd.isna(v): return float(v)
    pos_key = "_POS_GK_" if is_gk else "_POS_OUT_"
    v2 = priors.get(pos_key, {}).get(key)
    if not pd.isna(v2): return float(v2)
    v3 = priors.get("_GLOBAL_", {}).get(key)
    if not pd.isna(v3): return float(v3)
    return None

def _get_savepct_prior(priors: Dict[str, Dict[str, float]], pid: str) -> Tuple[float,float]:
    d = priors.get(pid, {})
    p0, s0 = d.get("save_pct_p0"), d.get("save_pct_s0")
    if (p0 is not None) and (not pd.isna(p0)) and (s0 is not None) and (s0 > 0): return float(p0), float(s0)
    grp = priors.get("_SAVE_PCT_", {})
    return float(grp.get("p0", 0.70)), float(grp.get("s0", 80.0))

# ───────────────────────── Core build ─────────────────────────

def build_player_form(
    seasons: List[str], fixtures_root: Path, version_dir: Path,  # version_dir resolved once
    window: int, tau: float, force: bool, prior_matches: int,
) -> None:
    frames = []
    for s in seasons:
        fp = fixtures_root / s / "player_minutes_calendar.csv"
        if not fp.is_file():
            logging.warning("%s • missing player_minutes_calendar.csv – skipped", s); continue
        df = pd.read_csv(fp, parse_dates=["date_played"]); df["season"] = s; frames.append(df)
    if not frames: raise FileNotFoundError("No seasons found under fixtures_root")

    all_players = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_BASE - set(all_players.columns)
    if missing: raise KeyError(f"Missing required columns: {missing}")

    all_players = all_players.sort_values(["player_id","season","date_played","gw_orig"]).reset_index(drop=True)
    for c in [c for c in NUMERIC_BASE if c in all_players.columns]:
        all_players[c] = pd.to_numeric(all_players[c], errors="coerce")
    for idcol in ["player_id","team_id","fbref_id","fpl_id"]:
        if idcol in all_players.columns: all_players[idcol] = all_players[idcol].astype(str)

    # ── NEW: Enforce position-consistent NaNs on raw columns (pre-derivations)
    gk_only_raw   = ["saves","sot_against","save_pct"]
    out_only_raw  = ["gls","ast","shots","sot","xg","npxg","xag",
                     "pkatt","pk_scored","pk_won","blocks","tkl","int","own_goals","recoveries"]
    # keep team outcome columns (gf, ga) for both
    is_gk_all = all_players["pos"].eq("GK")
    all_players.loc[~is_gk_all, [c for c in gk_only_raw if c in all_players.columns]] = np.nan
    all_players.loc[ is_gk_all, [c for c in out_only_raw if c in all_players.columns]] = np.nan
    # ─────────────────────────────────────────────────────────────

    seasons = sorted(seasons)
    for season in seasons:
        parts = season.split("-"); last_season = None
        if len(parts) == 2:
            y0 = int(parts[0]); last_season = f"{y0-1}-{y0}"

        priors = _compute_last_season_priors(all_players, last_season)
        cur = all_players[all_players["season"] == season].copy()

        # Per-90 build (already respects applies_to; we also pre-NaN'ed raw above)
        for cfg in METRICS.values():
            mask_app = _applicable_mask(cur["pos"], cfg["applies_to"])
            for raw in cfg["raw"]:
                col = f"{raw}_p90"
                cur[col] = np.where(cur["minutes"] > 0, cur[raw] * 90.0 / cur["minutes"], np.nan)
                cur.loc[~mask_app, col] = np.nan

        # Rolling per player
        out_rows: List[pd.DataFrame] = []
        for pid, g in cur.groupby("player_id", sort=False):
            g = g.sort_values(["date_played","gw_orig"]).copy()
            is_gk = bool((g["pos"] == "GK").iloc[0]) if len(g) else False

            for mkey, cfg in METRICS.items():
                applies_mask = _applicable_mask(g["pos"], cfg["applies_to"]).to_numpy()
                for raw in cfg["raw"]:
                    col = f"{raw}_p90"; arr = np.where(applies_mask, g[col].to_numpy(), np.nan)
                    prior_val = _get_prior_p90(priors, str(pid), raw, is_gk)
                    alpha = float(cfg.get("bayes_alpha", {}).get(raw, 0.0))
                    roll, rh, ra = _rolling_past_only_bayes_mean(
                        vals=arr, venues=g["venue"].astype(str).to_numpy(),
                        window=window, tau=tau, prior_val=prior_val, prior_matches=prior_matches, bayes_alpha=alpha
                    )
                    base = f"{mkey}_{raw}_p90"
                    g[f"{base}_roll"] = roll; g[f"{base}_home_roll"] = rh; g[f"{base}_away_roll"] = ra

            if is_gk:
                p0, s0 = _get_savepct_prior(priors, str(pid))
                post, post_h, post_a = _rolling_past_only_binomial_savepct(
                    saves=g["saves"].to_numpy(), shots=g["sot_against"].to_numpy(),
                    venues=g["venue"].astype(str).to_numpy(), window=window, tau=tau,
                    prior_p=p0, prior_shots=s0
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

        # Z-scores
        overall_cols = [c for c in feat.columns if c.endswith("_roll") and "_home_" not in c and "_away_" not in c]
        for c in overall_cols:
            z = _z_by_season_gw(feat, c)
            flip = any(c.startswith(f"{k}_") and METRICS[k]["flip_sign"] for k in METRICS)
            feat[c + "_z"] = (-z if flip else z)
        home_cols = [c for c in feat.columns if c.endswith("_home_roll")]
        for c in home_cols:
            z = _z_by_season_gw_venue(feat, c, "Home")
            flip = any(c.startswith(f"{k}_") and METRICS[k]["flip_sign"] for k in METRICS)
            feat[c + "_z"] = (-z if flip else z)
        away_cols = [c for c in feat.columns if c.endswith("_away_roll")]
        for c in away_cols:
            z = _z_by_season_gw_venue(feat, c, "Away")
            flip = any(c.startswith(f"{k}_") and METRICS[k]["flip_sign"] for k in METRICS)
            feat[c + "_z"] = (-z if flip else z)

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
                "savepct_binomial": {"prior": "last_season (player) → GK_group → global",
                                     "prior_p_default": 0.70, "prior_s_default": 80.0},
                "rare_event_shrinkage_alpha": {"pk_won": 6, "own_goals": 6},
                "zscore_mode": {"overall": "season,gw_orig",
                                "home": "season,gw_orig (Home only)",
                                "away": "season,gw_orig (Away only)"},
                "features": sorted([c for c in feat.columns if c.endswith(("_roll","_roll_z"))]),
            }
            (out_dir_season / "player_form.meta.json").write_text(json.dumps(meta, indent=2))
            logging.info("%s • %s (%d rows) written", season, OUTPUT_FILE, len(feat))

# ───────────────────────── Batch / CLI ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", help="e.g. 2024-2025; omit for batch")
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/fixtures"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--version", default=None, help="Version folder (e.g., v3). If omitted with --auto-version, next vN is used.")
    ap.add_argument("--auto-version", action="store_true", help="Pick the next vN under out-dir automatically.")
    ap.add_argument("--write-latest", action="store_true", help="Update features/latest to point to the resolved version.")
    ap.add_argument("--window", type=int, default=5, help="rolling window (matches, past-only)")
    ap.add_argument("--tau", type=float, default=2.0, help="venue shrinkage strength")
    ap.add_argument("--prior-matches", type=int, default=6, help="first K matches blend prior → 0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=args.log_level.upper())

    seasons = [args.season] if args.season else sorted(d.name for d in args.fixtures_root.iterdir() if d.is_dir())
    if not seasons:
        logging.error("No season folders in %s", args.fixtures_root); return
    seasons = sorted(seasons)

    # Resolve version directory once
    features_root = args.out_dir
    features_root.mkdir(parents=True, exist_ok=True)
    version = _resolve_version(features_root, args.version, args.auto_version)
    version_dir = features_root / version
    version_dir.mkdir(parents=True, exist_ok=True)

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
    )

    if args.write_latest:
        _write_latest_pointer(features_root, version)

if __name__ == "__main__":
    main()
