#!/usr/bin/env python3
r"""
saves_forecast.py — leak-free future scorer for GK saves using trained heads

What it does
------------
• Scores FUTURE fixtures only (no retraining).
• Uses your trained per-90 saves models from `data/models/saves/versions/<vN>/`.
• Pins feature order to the trainer’s `features_used.txt` (prevents LGBM shape errors).
• Builds features leak-free: past snapshots for rolling form; FUTURE venue & team Zs from team_form.
• Scales per-90 → per-match using `pred_minutes` from the minutes forecast.
• Robust venue resolution with multiple join strategies; handles doubles; never many-to-many.
• **Roster gate**: only score players on the target-season roster from your master teams JSON.
• **GK-only output**: rows for non-GK are dropped.

Outputs
-------
<out-dir>/<SEASON>/GW<from>_<to>.csv with columns:
  season, gw_orig, date_sched, player_id, team_id, player, pos, pred_minutes,
  team_def_z_venue, opp_att_z_venue,
  pred_saves_p90_mean, pred_saves_mean,
  pred_saves_p90_poisson, pred_saves_poisson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib


# ----------------------------- tiny utils ------------------------------------

def _pick_gw_col(cols: List[str]) -> Optional[str]:
    for k in ("gw_played", "gw_orig", "gw"):
        if k in cols:
            return k
    return None


def _coerce_ts(s: pd.Series, tz: Optional[str]) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if tz:
        if out.dt.tz is None:
            out = out.dt.tz_localize(tz)
        else:
            out = out.dt.tz_convert(tz)
    return out


# ----------------------------- loading registry ------------------------------

def _load_players_form(features_root: Path, form_version: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "players_form.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Missing players_form: {fp}")
        t = pd.read_csv(fp, parse_dates=["date_played"])  # tolerant
        t["season"] = s
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    need = {"season", "gw_orig", "date_played", "player_id", "team_id", "player", "pos", "venue", "minutes"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"players_form missing: {sorted(miss)}")
    return df


def _load_team_form(features_root: Path, form_version: str, seasons: List[str]) -> Optional[pd.DataFrame]:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "team_form.csv"
        if not fp.exists():
            continue
        t = pd.read_csv(fp, parse_dates=["date_played"])  # tolerant
        t["season"] = s
        frames.append(t)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _team_z_maps(team_form: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return unique (season, gw_orig, team_id) → team_def_z_venue, opp_att_z_venue.
    Falls back to split roll_z columns if combined fields are absent.
    """
    if team_form is None:
        return None

    if {"team_def_z_venue", "opp_att_z_venue"}.issubset(team_form.columns):
        t = (
            team_form[
                ["season", "gw_orig", "team_id", "team_def_z_venue", "opp_att_z_venue"]
            ]
            .drop_duplicates(subset=["season", "gw_orig", "team_id"], keep="last")
            .copy()
        )
        for c in ("team_def_z_venue", "opp_att_z_venue"):
            t[c] = pd.to_numeric(t[c], errors="coerce")
        return t

    need = {
        "season",
        "gw_orig",
        "team_id",
        "venue",
        "def_xga_home_roll_z",
        "def_xga_away_roll_z",
        "att_xg_home_roll_z",
        "att_xg_away_roll_z",
    }
    if not need.issubset(team_form.columns):
        return None

    t = team_form[list(need)].copy()
    t["team_def_z_venue"] = np.where(
        t["venue"].astype(str).str.lower().eq("home"), t["def_xga_home_roll_z"], t["def_xga_away_roll_z"]
    )
    t["opp_att_z_venue"] = np.where(
        t["venue"].astype(str).str.lower().eq("home"), t["att_xg_away_roll_z"], t["att_xg_home_roll_z"]
    )
    t = t.drop(
        columns=[
            "venue",
            "def_xga_home_roll_z",
            "def_xga_away_roll_z",
            "att_xg_home_roll_z",
            "att_xg_away_roll_z",
        ]
    )
    t = t.drop_duplicates(subset=["season", "gw_orig", "team_id"], keep="last")
    for c in ("team_def_z_venue", "opp_att_z_venue"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


# ----------------------------- minutes + venue -------------------------------

def _gw_for_selection(df: pd.DataFrame) -> pd.Series:
    def num(col: str) -> pd.Series:
        return pd.to_numeric(df.get(col), errors="coerce")

    gwp = num("gw_played")
    gwo = num("gw_orig")
    gwa = num("gw")
    return gwo.where(gwp.isna() | (gwp <= 0), gwp).where(lambda x: x.notna(), gwa)


def _derive_is_home_from_flags(df: pd.DataFrame) -> Optional[pd.Series]:
    if "is_home" in df.columns:
        return df["is_home"].astype(int)
    if "was_home" in df.columns:
        return df["was_home"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
    if "venue" in df.columns:
        return df["venue"].astype(str).str.lower().eq("home").astype(int)
    return None


def _safe_mode_is_home(s: pd.Series) -> float:
    """Aggregate potentially conflicting is_home flags within a group.
    Returns 1.0 if all ones, 0.0 if all zeros, NaN if mixed.
    """
    vals = pd.unique(pd.to_numeric(s, errors="coerce").dropna())
    if len(vals) == 1:
        return float(vals[0])
    return np.nan


def _resolve_venue(minutes: pd.DataFrame,
                   team_fix: Optional[pd.DataFrame],
                   future_season: str) -> pd.Series:
    """Return a 0/1 `venue_bin` Series aligned to `minutes` rows.
    Priority:
      1) Use explicit flags already present on minutes.
      2) Join via (season, game_id, team_id) if both sides have game_id.
      3) Join via (season, gw_key, team_id, opp_team_id) if both sides have opp ids.
      4) Join via (season, gw_key, team_id) after aggregating team fixtures to unique rows; mixed → NaN.
      5) Fallback 0 where still missing.
    """
    # 1) Directly present
    direct = _derive_is_home_from_flags(minutes)
    if direct is not None:
        return direct.clip(lower=0, upper=1)

    if team_fix is None or minutes.empty:
        return pd.Series(np.zeros(len(minutes), dtype=int), index=minutes.index)

    gw_key_m = _pick_gw_col(minutes.columns.tolist()) or "gw_orig"
    gw_key_t = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"

    # Harmonise season
    if "season" not in minutes.columns:
        minutes = minutes.copy()
        minutes["season"] = future_season

    # Prepare candidate right tables
    base_cols = [c for c in ["season", gw_key_t, "team_id", "is_home", "game_id", "home_id", "away_id"] if c in team_fix.columns]
    tf = team_fix[base_cols].dropna(subset=[gw_key_t, "team_id"]) if base_cols else pd.DataFrame()

    # 2) game_id join
    if {"game_id", "team_id"}.issubset(minutes.columns) and {"game_id", "team_id"}.issubset(tf.columns):
        v = minutes.merge(
            tf[["season", "game_id", "team_id", "is_home"]].drop_duplicates(),
            how="left",
            on=["season", "game_id", "team_id"],
            validate="many_to_one",
        )["is_home"]
        if v.notna().any():
            return v.fillna(0).astype(int)

    # 3) opp_team_id join
    if "opp_team_id" in minutes.columns and {"home_id", "away_id"}.issubset(tf.columns):
        tf2 = tf.copy()
        tf2["opp_team_id"] = np.where(
            tf2["team_id"].astype(str).str.lower().eq(tf2.get("home_id", "").astype(str).str.lower()),
            tf2.get("away_id"),
            tf2.get("home_id"),
        )
        right = tf2[["season", gw_key_t, "team_id", "opp_team_id", "is_home"]].drop_duplicates()
        v = minutes.merge(
            right.rename(columns={gw_key_t: gw_key_m}),
            how="left",
            on=["season", gw_key_m, "team_id", "opp_team_id"],
            validate="many_to_one",
        )["is_home"]
        if v.notna().any():
            return v.fillna(0).astype(int)

    # 4) aggregated (season, gw, team) map → unique
    if {gw_key_t, "team_id", "is_home"}.issubset(tf.columns):
        vmap = (
            tf.groupby(["season", gw_key_t, "team_id"], as_index=False)["is_home"].agg(_safe_mode_is_home)
        )
        v = minutes.merge(
            vmap.rename(columns={gw_key_t: gw_key_m}),
            how="left",
            on=["season", gw_key_m, "team_id"],
            validate="many_to_one",
        )["is_home"]
        return v.fillna(0).astype(int)

    # 5) fallback
    return pd.Series(np.zeros(len(minutes), dtype=int), index=minutes.index)


# ----------------------------- model loading ---------------------------------

def _load_feature_list(model_dir: Path) -> List[str]:
    # Preferred: features_used.txt (trainer wrote this)
    txt = model_dir / "artifacts" / "features_used.txt"
    if txt.exists():
        return [line.strip() for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Fallback: artifacts/features.json (in case you swapped formats)
    alt = model_dir / "artifacts" / "features.json"
    if alt.exists():
        try:
            return list(json.loads(alt.read_text(encoding="utf-8")))
        except Exception:
            pass
    raise FileNotFoundError(f"No feature list: {txt} or {alt}")


def _load_lgb_booster(model_dir: Path) -> lgb.Booster:
    p = model_dir / "models" / "lgbm_saves_p90.txt"
    if not p.exists():
        # also allow flat model in dir (older layout)
        p = model_dir / "lgbm_saves_p90.txt"
    return lgb.Booster(model_file=str(p))


def _load_poisson(model_dir: Path) -> Tuple[Optional[object], Optional[object]]:
    # returns (glm, imputer)
    for a, b in [
        (model_dir / "models" / "poisson_saves_p90.joblib", model_dir / "models" / "poisson_imputer.joblib"),
        (model_dir / "poisson_saves_p90.joblib", model_dir / "poisson_imputer.joblib"),
    ]:
        if a.exists() and b.exists():
            try:
                return joblib.load(a), joblib.load(b)
            except Exception:
                pass
    return None, None


# ----------------------------- roster gate -----------------------------------

def _load_roster_from_master(path: Path, season: str, league_filter: Optional[str]) -> pd.DataFrame:
    """
    Returns DataFrame with at least: player_id, team_id, season.
    JSON shape:
      { "<team_id>": { "name": "...", "career": { "<season>": {
           "league": "...", "players": [ {"id": "...", "name": "..."}, ... ]
      }}}}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for team_id, blob in (data or {}).items():
        career = (blob or {}).get("career", {})
        if season not in career:
            continue
        entry = career[season] or {}
        lg = entry.get("league")
        if league_filter and str(lg) != str(league_filter):
            continue
        for p in entry.get("players", []) or []:
            pid = str(p.get("id", "")).strip()
            if not pid:
                continue
            rows.append({"season": season, "team_id": str(team_id), "player_id": pid, "player_name_master": p.get("name")})
    if not rows:
        return pd.DataFrame(columns=["season", "team_id", "player_id", "player_name_master"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["season", "player_id"])
    for c in ("season", "team_id", "player_id"):
        df[c] = df[c].astype(str)
    return df


# ----------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # windows
    ap.add_argument("--history-seasons", required=True, help="Comma list of past seasons")
    ap.add_argument("--future-season", required=True)
    ap.add_argument("--as-of", default="now", help='ISO timestamp or "now"')
    ap.add_argument("--as-of-tz", default="Africa/Lagos")
    ap.add_argument("--as-of-gw", type=int, required=True, help="First unplayed GW at --as-of (next GW)")
    ap.add_argument("--n-future", type=int, default=3)
    ap.add_argument("--gw-from", type=int, default=None)
    ap.add_argument("--gw-to", type=int, default=None)
    ap.add_argument("--strict-n-future", action="store_true")

    # IO
    ap.add_argument("--features-root", type=Path, default=Path("data/processed/registry/features"))
    ap.add_argument("--form-version", required=True)
    ap.add_argument("--fix-root", type=Path, default=Path("data/processed/registry/fixtures"))
    ap.add_argument("--team-fixtures-filename", default="fixture_calendar.csv")
    ap.add_argument("--minutes-csv", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/predictions/saves"))
    ap.add_argument("--require-pred-minutes", action="store_true")

    # roster gating
    ap.add_argument("--teams-json", type=Path, help="Master teams JSON with per-season rosters")
    ap.add_argument("--league-filter", default="ENG-Premier League",
                    help="Only accept teams whose career[season].league matches this (set '' to disable)")
    ap.add_argument("--require-on-roster", action="store_true",
                    help="Error if any minutes rows are not on the season roster; otherwise silently drop them")

    ap.add_argument("--log-level", default="INFO")

    args = ap.parse_args()
    import logging

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    # --- seasons & as-of ---
    history = [s.strip() for s in args.history_seasons.split(",") if s.strip()]
    seasons_all = history + [args.future_season]

    gw_from_req = args.gw_from if args.gw_from is not None else args.as_of_gw
    gw_to_req = args.gw_to if args.gw_to is not None else (int(gw_from_req) + max(1, int(args.n_future)) - 1)

    as_of_ts = (
        pd.Timestamp.now(tz=args.as_of_tz)
        if str(args.as_of).lower() in ("now", "today", "auto")
        else pd.Timestamp(args.as_of, tz=args.as_of_tz)
    )

    # --- load artifacts & registry ---
    feat_cols = _load_feature_list(args.model_dir)
    lgb_booster = _load_lgb_booster(args.model_dir)
    pois_glm, pois_imputer = _load_poisson(args.model_dir)

    pf = _load_players_form(args.features_root, args.form_version, seasons_all)
    tf = _load_team_form(args.features_root, args.form_version, seasons_all)
    tz_map = _team_z_maps(tf)

    # Past-only rows for last snapshot
    du = _coerce_ts(pf["date_played"], args.as_of_tz)
    pf_hist = pf[(pf["season"].isin(history)) | ((pf["season"] == args.future_season) & (du < as_of_ts))].copy()

    # --- minutes target window ---
    minutes = pd.read_csv(args.minutes_csv, parse_dates=["date_sched"])
    if "season" not in minutes.columns:
        minutes["season"] = args.future_season

    # GK-only as early as possible
    if "pos" in minutes.columns:
        minutes = minutes[minutes["pos"].astype(str).str.upper().eq("GK")].copy()

    gw_sel = _gw_for_selection(minutes)
    avail_gws = sorted(pd.unique(gw_sel.dropna().astype(int))) if not gw_sel.dropna().empty else []
    target_gws = [int(g) for g in avail_gws if g >= int(gw_from_req)][: int(args.n_future)]
    if not target_gws:
        raise RuntimeError(f"No target GWs >= {gw_from_req}. Available: {avail_gws}")
    if args.strict_n_future and len(target_gws) < int(args.n_future):
        raise RuntimeError(f"Only {len(target_gws)} GW(s) available; wanted {args.n_future}. Available: {avail_gws}")

    gw_key_m = _pick_gw_col(minutes.columns.tolist()) or "gw_orig"
    minutes = minutes[gw_sel.isin(target_gws)].copy()
    if args.require_pred_minutes and "pred_minutes" not in minutes.columns:
        raise KeyError("--require-pred-minutes set but minutes CSV lacks pred_minutes")

    # --- roster gate (keep only players on target-season roster) ---
    if args.teams_json and args.teams_json.exists():
        roster = _load_roster_from_master(
            args.teams_json,
            season=args.future_season,
            league_filter=(args.league_filter if args.league_filter != "" else None),
        )
        if roster.empty:
            raise RuntimeError(
                f"No roster entries found in {args.teams_json} for season={args.future_season} "
                f"with league_filter={args.league_filter!r}."
            )
        before = len(minutes)
        minutes["player_id"] = minutes["player_id"].astype(str)
        minutes = minutes.merge(roster[["player_id"]].drop_duplicates(), on="player_id", how="inner", validate="many_to_one")
        after = len(minutes)
        dropped = before - after
        if dropped > 0:
            logging.info(
                "Roster gate dropped %d fixture rows not present on the %s roster.",
                dropped, args.future_season
            )
        if args.require_on_roster and dropped > 0:
            raise RuntimeError(f"--require-on-roster set: {dropped} minutes rows are not on the {args.future_season} roster.")

    if minutes.empty:
        raise RuntimeError("No minutes rows left after GK filter/roster gating/target GW selection.")

    # --- venue from team fixtures (robust; no many-to-many) ---
    try:
        team_fix = pd.read_csv(args.fix_root / args.future_season / args.team_fixtures_filename)
        for dc in ("date_sched", "date_played"):
            if dc in team_fix.columns:
                team_fix[dc] = pd.to_datetime(team_fix[dc], errors="coerce")
        if "is_home" not in team_fix.columns:
            if "was_home" in team_fix.columns:
                team_fix["is_home"] = team_fix["was_home"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
            elif "venue" in team_fix.columns:
                team_fix["is_home"] = team_fix["venue"].astype(str).str.lower().eq("home").astype(int)
        for c in ("gw_played", "gw_orig", "gw"):
            if c in team_fix.columns:
                team_fix[c] = pd.to_numeric(team_fix[c], errors="coerce")
        if "season" not in team_fix.columns:
            team_fix["season"] = args.future_season
    except FileNotFoundError:
        team_fix = None

    venue_bin = _resolve_venue(minutes, team_fix, args.future_season)

    # --- build FUTURE frame ---
    fut = minutes.copy()
    fut["venue_bin"] = venue_bin.astype(int)

    # attach team Zs
    if tz_map is not None and gw_key_m in fut.columns:
        fut = fut.merge(
            tz_map.rename(columns={"gw_orig": gw_key_m}),
            how="left",
            on=["season", gw_key_m, "team_id"],
            validate="many_to_one",
        )
    else:
        fut["team_def_z_venue"] = np.nan
        fut["opp_att_z_venue"] = np.nan

    # last snapshot per (season, player) strictly before as_of
    future_only = {"venue_bin", "team_def_z_venue", "opp_att_z_venue"}
    snap_cols = [c for c in feat_cols if c not in future_only]
    for c in snap_cols:
        if c not in pf_hist.columns:
            pf_hist[c] = np.nan

    sort_cols = ["player_id", "season", "date_played"]
    gw_key_hist = _pick_gw_col(pf_hist.columns.tolist())
    if gw_key_hist:
        sort_cols.append(gw_key_hist)
    pf_hist = pf_hist.sort_values(sort_cols)
    last = pf_hist.groupby(["season", "player_id"], as_index=False).tail(1)

    # Join last snapshot (per player) onto FUTURE fixtures
    keep_last = ["season", "player_id"] + snap_cols
    fut = fut.merge(last[keep_last], how="left", on=["season", "player_id"], validate="many_to_one")

    # Compose X in exact order expected by model
    X = pd.DataFrame(index=fut.index)
    for c in feat_cols:
        if c in fut.columns:
            X[c] = pd.to_numeric(fut[c], errors="coerce")
        else:
            X[c] = np.nan

    # Predict per-90 with LGBM
    Xn = X.select_dtypes(include=[np.number]).fillna(0.0)
    gk_p90_mean = np.clip(lgb_booster.predict(Xn), 0, None)

    # Optional Poisson head (Tweedie) on imputed matrix
    if pois_glm is not None and pois_imputer is not None:
        Xp = pois_imputer.transform(Xn)
        gk_p90_pois = np.clip(pois_glm.predict(Xp), 0, None)
    else:
        gk_p90_pois = np.full(len(Xn), np.nan, dtype=float)

    # Scale to per-match
    if "pred_minutes" not in fut.columns:
        fut["pred_minutes"] = fut.get("minutes", pd.Series(np.zeros(len(fut))))
    scale = pd.to_numeric(fut["pred_minutes"], errors="coerce").fillna(0.0).to_numpy() / 90.0
    pred_saves_mean = gk_p90_mean * scale
    pred_saves_pois = gk_p90_pois * scale if not np.isnan(gk_p90_pois).all() else gk_p90_pois

    # Assemble output (GK-only)
    cols: Dict[str, np.ndarray] = {
        "season": fut["season"].to_numpy(),
        "gw_orig": fut[gw_key_m].to_numpy() if gw_key_m in fut.columns else np.full(len(fut), np.nan),
        "date_sched": fut.get("date_sched", pd.Series([np.nan] * len(fut))).to_numpy(),
        "player_id": fut["player_id"].to_numpy(),
        "team_id": fut.get("team_id", pd.Series([np.nan] * len(fut))).to_numpy(),
        "player": fut.get("player", pd.Series([np.nan] * len(fut))).to_numpy(),
        "pos": (fut.get("pos", pd.Series(["GK"] * len(fut))).astype(str).str.upper()).to_numpy(),
        "pred_minutes": fut["pred_minutes"].to_numpy(),
        "team_def_z_venue": fut.get("team_def_z_venue", pd.Series([np.nan] * len(fut))).to_numpy(),
        "opp_att_z_venue": fut.get("opp_att_z_venue", pd.Series([np.nan] * len(fut))).to_numpy(),
        "pred_saves_p90_mean": gk_p90_mean,
        "pred_saves_mean": pred_saves_mean,
        "pred_saves_p90_poisson": gk_p90_pois,
        "pred_saves_poisson": pred_saves_pois,
    }
    out = pd.DataFrame(cols)

    # Hard-enforce GK-only rows in final output
    out = out[out["pos"].astype(str).str.upper().eq("GK")].reset_index(drop=True)

    # Sort & persist
    sort_keys = [k for k in ["gw_orig", "team_id", "player_id"] if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys).reset_index(drop=True)

    gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
    season_dir = args.out_dir / f"{args.future_season}"
    season_dir.mkdir(parents=True, exist_ok=True)
    out_path = season_dir / f"GW{gw_from_eff}_{gw_to_eff}.csv"
    out.to_csv(out_path, index=False)

    diag = {
        "rows": int(len(out)),
        "season": str(args.future_season),
        "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
        "available_team_gws": [int(x) for x in avail_gws],
        "scored_gws": [int(x) for x in target_gws],
        "as_of": str(as_of_ts),
        "out": str(out_path),
        "roster_gate": {
            "enabled": bool(args.teams_json),
            "league_filter": (args.league_filter if args.league_filter != "" else None)
        }
    }
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
