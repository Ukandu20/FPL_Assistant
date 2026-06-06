#!/usr/bin/env python3
r"""
saves_forecast.py — leak-free future scorer for GK saves using trained heads

• Auto-resolve minutes (CSV/Parquet), GK-only, roster gate
• Venue resolution + **FDR attach (INT, venue-consistent, DGW-safe)**
• Legacy fixture metadata (game_id/team/opponent_id/opponent/is_home)
• Output format control (csv/parquet/both)
"""

from __future__ import annotations

import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from scripts.utils.validate import validate_df

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

def _normalize_dates_inplace(df: pd.DataFrame, cols: Tuple[str, ...] = ("date_sched","date_played")) -> None:
    """Force datetime64[ns] and strip time to 00:00:00 for all listed date cols."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()

def _fmt_gw(n: int, zero_pad: bool) -> str:
    return f"{int(n):02d}" if zero_pad else f"{int(n)}"


# ----------------------------- loading registry ------------------------------

def _load_players_form(features_root: Path, form_version: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        fp = features_root / form_version / s / "players_form.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Missing players_form: {fp}")
        t = pd.read_csv(fp, parse_dates=["date_played"])
        t["season"] = s
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    need = {"season","gw_orig","date_played","player_id","team_id","player","pos","venue","minutes"}
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
        t = pd.read_csv(fp, parse_dates=["date_played"])
        t["season"] = s
        frames.append(t)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def _team_z_maps(team_form: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return unique (season, gw_orig, team_id) → team_def_z_venue, opp_att_z_venue."""
    if team_form is None:
        return None
    if {"team_def_z_venue","opp_att_z_venue"}.issubset(team_form.columns):
        t = (team_form[["season","gw_orig","team_id","team_def_z_venue","opp_att_z_venue"]]
             .drop_duplicates(subset=["season","gw_orig","team_id"], keep="last").copy())
        for c in ("team_def_z_venue","opp_att_z_venue"):
            t[c] = pd.to_numeric(t[c], errors="coerce")
        return t

    need = {"season","gw_orig","team_id","venue",
            "def_xga_home_roll_z","def_xga_away_roll_z",
            "att_xg_home_roll_z","att_xg_away_roll_z"}
    if not need.issubset(team_form.columns):
        return None

    t = team_form[list(need)].copy()
    vhome = t["venue"].astype(str).str.lower().eq("home")
    t["team_def_z_venue"] = np.where(vhome, t["def_xga_home_roll_z"], t["def_xga_away_roll_z"])
    t["opp_att_z_venue"]  = np.where(vhome, t["att_xg_away_roll_z"], t["att_xg_home_roll_z"])
    t = t.drop(columns=["venue","def_xga_home_roll_z","def_xga_away_roll_z","att_xg_home_roll_z","att_xg_away_roll_z"])
    t = t.drop_duplicates(subset=["season","gw_orig","team_id"], keep="last")
    for c in ("team_def_z_venue","opp_att_z_venue"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


# ----------------------------- minutes I/O & venue ---------------------------

def _gw_for_selection(df: pd.DataFrame) -> pd.Series:
    def num(col: str) -> pd.Series: return pd.to_numeric(df.get(col), errors="coerce")
    gwp = num("gw_played"); gwo = num("gw_orig"); gwa = num("gw")
    return gwo.where(gwp.isna() | (gwp <= 0), gwp).where(lambda x: x.notna(), gwa)

def _derive_is_home_from_flags(df: pd.DataFrame) -> Optional[pd.Series]:
    if "is_home" in df.columns:
        return pd.to_numeric(df["is_home"], errors="coerce").fillna(0).astype("Int8")
    if "was_home" in df.columns:
        return df["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype("Int8")
    if "venue" in df.columns:
        return df["venue"].astype(str).str.lower().eq("home").astype("Int8")
    return None

def _safe_mode_is_home(s: pd.Series) -> float:
    vals = pd.unique(pd.to_numeric(s, errors="coerce").dropna())
    if len(vals) == 1: return float(vals[0])
    return np.nan

def _resolve_venue(minutes: pd.DataFrame,
                   team_fix: Optional[pd.DataFrame],
                   future_season: str) -> pd.Series:
    """Return 0/1 venue_bin aligned to minutes with several fallback joins."""
    direct = _derive_is_home_from_flags(minutes)
    if direct is not None:
        return direct.clip(0,1)

    if team_fix is None or minutes.empty:
        return pd.Series(np.zeros(len(minutes), dtype=int), index=minutes.index)

    gw_key_m = _pick_gw_col(minutes.columns.tolist()) or "gw_orig"
    gw_key_t = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"

    if "season" not in minutes.columns:
        minutes = minutes.copy(); minutes["season"] = future_season

    base_cols = [c for c in ["season",gw_key_t,"team_id","is_home","game_id","home_id","away_id","fbref_id"]
                 if c in team_fix.columns]
    tf = team_fix[base_cols].dropna(subset=[gw_key_t,"team_id"]) if base_cols else pd.DataFrame()

    # game_id join
    for gid_col in ("game_id","fbref_id"):
        if {gid_col,"team_id"}.issubset(minutes.columns) and {gid_col,"team_id"}.issubset(tf.columns):
            v = minutes.merge(
                tf[["season",gid_col,"team_id","is_home"]].drop_duplicates(),
                how="left", on=["season",gid_col,"team_id"], validate="many_to_one"
            )["is_home"]
            if v.notna().any(): return v.fillna(0).astype("Int8")

    # opp join
    if "opp_team_id" in minutes.columns and {"home_id","away_id"}.issubset(tf.columns):
        tf2 = tf.copy()
        tf2["opp_team_id"] = np.where(
            tf2["team_id"].astype(str).str.lower().eq(tf2.get("home_id","").astype(str).str.lower()),
            tf2.get("away_id"), tf2.get("home_id"),
        )
        right = tf2[["season",gw_key_t,"team_id","opp_team_id","is_home"]].drop_duplicates()
        v = minutes.merge(
            right.rename(columns={gw_key_t: gw_key_m}),
            how="left", on=["season",gw_key_m,"team_id","opp_team_id"], validate="many_to_one"
        )["is_home"]
        if v.notna().any(): return v.fillna(0).astype("Int8")

    # aggregated (season, gw, team)
    if {gw_key_t,"team_id","is_home"}.issubset(tf.columns):
        vmap = tf.groupby(["season",gw_key_t,"team_id"], as_index=False)["is_home"].agg(_safe_mode_is_home)
        v = minutes.merge(
            vmap.rename(columns={gw_key_t: gw_key_m}),
            how="left", on=["season",gw_key_m,"team_id"], validate="many_to_one"
        )["is_home"]
        return v.fillna(0).astype("Int8")

    return pd.Series(np.zeros(len(minutes), dtype=int), index=minutes.index)


# ----------------------------- minutes resolver (auto) ------------------------

def _candidate_minutes_paths(minutes_root: Path, future_season: str, gw_from: int, gw_to: int) -> List[Path]:
    season_dir = minutes_root / str(future_season)
    cands: List[Path] = []
    for zp in (False, True):
        a = _fmt_gw(gw_from, zp); b = _fmt_gw(gw_to, zp)
        cands.append(season_dir / f"GW{a}_{b}.csv")
        cands.append(season_dir / f"GW{a}_{b}.parquet")
    return cands

def _glob_fallback(minutes_root: Path, future_season: str, gw_from: int, gw_to: int) -> Optional[Path]:
    season_dir = minutes_root / str(future_season)
    if not season_dir.exists(): return None
    patterns = [f"GW{gw_from}_*.csv", f"GW{gw_from}_*.parquet",
                f"GW{gw_from:02d}_*.csv", f"GW{gw_from:02d}_*.parquet"]
    for pat in patterns:
        for p in sorted(season_dir.glob(pat)):
            try:
                to_str = p.stem.split("_")[-1].replace("GW","")
                if int(to_str) == int(gw_to): return p
            except Exception:
                continue
    return None

def _resolve_minutes_path(args: argparse.Namespace, gw_from: int, gw_to: int) -> Path:
    if args.minutes_csv:
        p = Path(args.minutes_csv)
        if not p.exists(): raise FileNotFoundError(f"--minutes-csv not found: {p}")
        return p
    for cand in _candidate_minutes_paths(args.minutes_root, args.future_season, gw_from, gw_to):
        if cand.exists(): return cand
    fb = _glob_fallback(args.minutes_root, args.future_season, gw_from, gw_to)
    if fb: return fb
    season_dir = args.minutes_root / str(args.future_season)
    msg = [f"Minutes file not found for GW window {gw_from}-{gw_to}.",
           f"Looked under: {season_dir}", "Tried candidates:"]
    for c in _candidate_minutes_paths(args.minutes_root, args.future_season, gw_from, gw_to):
        msg.append(f"  - {c}")
    msg += ["Also tried glob fallback: GW{from}_*.{csv,parquet} (with and without zero-padding).",
            "Fix by either:",
            "  • generating that file, or",
            "  • pointing directly via --minutes-csv <path>, or",
            "  • adjusting --minutes-root / GW window."]
    raise FileNotFoundError("\n".join(msg))

def _load_minutes_dual(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        hdr = pd.read_csv(path, nrows=0)
        parse_cols = ["date_sched"] if "date_sched" in hdr.columns else None
        df = pd.read_csv(path, parse_dates=parse_cols)
        _normalize_dates_inplace(df, ("date_sched",))
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        if "date_sched" in df.columns:
            df["date_sched"] = pd.to_datetime(df["date_sched"], errors="coerce").dt.normalize()
    else:
        raise ValueError(f"Unsupported minutes file extension: {suffix}. Use .csv or .parquet")
    return df


# ----------------------------- FDR attach (consistent) ------------------------

def _ensure_is_home(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype("Int8")
        return out
    if "was_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["was_home"], errors="coerce").fillna(0).astype("Int8")
        return out
    if "venue" in out.columns:
        out["is_home"] = out["venue"].astype(str).str.lower().eq("home").astype("Int8")
        return out
    # if absent, we'll set from venue_bin later
    out["is_home"] = np.nan
    return out

def _find_fdr_cols(cols: set[str]) -> tuple[str, str]:
    home_aliases = ["fdr_home","team_fdr_home","def_fdr_home","fdrH"]
    away_aliases = ["fdr_away","team_fdr_away","def_fdr_away","fdrA"]
    home = next((c for c in home_aliases if c in cols), None)
    away = next((c for c in away_aliases if c in cols), None)
    if not home or not away:
        raise RuntimeError(
            f"FDR columns not found in team_form. Tried {home_aliases} and {away_aliases}. Got: {sorted(cols)}"
        )
    return home, away

def attach_fdr_consistent(df: pd.DataFrame,
                          seasons_all: List[str],
                          features_root: Path,
                          version: str,
                          team_form: Optional[pd.DataFrame] = None,
                          team_fix: Optional[pd.DataFrame] = None,
                          audit_dir: Optional[Path] = None,
                          audit_tag: Optional[str] = None,
                          drop_on_missing: bool = True) -> pd.DataFrame:
    """
    Attach integer FDR via (season, team_id, GW, is_home).
    Priority:
      1) team_form with separate home/away FDR columns (historical and, if available, forward-filled)
      2) team fixtures (fixture_calendar*) if they expose `fdr` (or `fdr_home`/`fdr_away`)
    DGW-safe: collapse dup keys by max.

    If rows remain without FDR after both sources:
      - If drop_on_missing=True (default): write them to an audit CSV and drop them, proceed with the rest.
      - Else: raise RuntimeError with diagnostics.
    """
    if df.empty:
        return df

    df = _ensure_is_home(df)
    df["team_id"] = df["team_id"].astype(str)
    for c in ("gw_played","gw_orig","gw"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    gw_df = _pick_gw_col(df.columns.tolist())
    if gw_df is None:
        raise RuntimeError("attach_fdr_consistent: no GW column among ['gw_played','gw_orig','gw'].")

    # ---------- Source 1: team_form (home/away columns) ----------
    tf_all = team_form.copy() if team_form is not None else _load_team_form(features_root, version, seasons_all)
    form_join_ok = False
    merged = df.copy()

    if tf_all is not None and not tf_all.empty:
        tf_all["team_id"] = tf_all.get("team_id", pd.Series(index=tf_all.index, dtype=object)).astype(str)
        for c in ("gw_played","gw_orig","gw"):
            if c in tf_all.columns:
                tf_all[c] = pd.to_numeric(tf_all[c], errors="coerce")

        try:
            home_col, away_col = _find_fdr_cols(set(tf_all.columns))
            gw_tf = _pick_gw_col(tf_all.columns.tolist())
            if gw_tf is None:
                raise RuntimeError("attach_fdr_consistent: no GW column in team_form.")

            base = tf_all[["season","team_id",gw_tf,home_col,away_col]].dropna(subset=["team_id",gw_tf])
            home_rows = base.rename(columns={home_col: "fdr_side"}).assign(is_home=1)[["season","team_id",gw_tf,"is_home","fdr_side"]]
            away_rows = base.rename(columns={away_col: "fdr_side"}).assign(is_home=0)[["season","team_id",gw_tf,"is_home","fdr_side"]]
            form_long = pd.concat([home_rows, away_rows], ignore_index=True)
            if gw_tf != gw_df:
                form_long = form_long.rename(columns={gw_tf: gw_df})

            form_long = (form_long.groupby(["season","team_id",gw_df,"is_home"], as_index=False)["fdr_side"].max())
            merged = merged.merge(form_long, how="left",
                                  on=["season","team_id",gw_df,"is_home"],
                                  validate="many_to_one", copy=False)
            form_join_ok = merged["fdr_side"].notna().any()
        except Exception:
            form_join_ok = False  # allow fallback

    # ---------- Source 2: team fixtures (fallback) ----------
    need_fixtures = (not form_join_ok) or merged["fdr_side"].isna().any()
    if need_fixtures and team_fix is not None and not team_fix.empty:
        tf = team_fix.copy()
        for c in ("team_id",):
            if c in tf.columns:
                tf[c] = tf[c].astype(str)
        for c in ("gw_played","gw_orig","gw"):
            if c in tf.columns:
                tf[c] = pd.to_numeric(tf[c], errors="coerce")
        if "season" not in tf.columns:
            tf["season"] = df["season"].iloc[0]

        gw_tf = _pick_gw_col(tf.columns.tolist()) or "gw_orig"

        fdr_cols = set(tf.columns)
        has_pair = ("fdr_home" in fdr_cols or "team_fdr_home" in fdr_cols) and ("fdr_away" in fdr_cols or "team_fdr_away" in fdr_cols)
        has_single = "fdr" in fdr_cols

        if has_pair:
            home_col = "fdr_home" if "fdr_home" in fdr_cols else "team_fdr_home"
            away_col = "fdr_away" if "fdr_away" in fdr_cols else "team_fdr_away"
            base = tf[["season","team_id",gw_tf,home_col,away_col]].dropna(subset=["team_id",gw_tf]).copy()
            home_rows = base.rename(columns={home_col: "fdr_side"}).assign(is_home=1)[["season","team_id",gw_tf,"is_home","fdr_side"]]
            away_rows = base.rename(columns={away_col: "fdr_side"}).assign(is_home=0)[["season","team_id",gw_tf,"is_home","fdr_side"]]
            fx_long = pd.concat([home_rows, away_rows], ignore_index=True)
        elif has_single:
            if "is_home" not in tf.columns:
                if "venue" in tf.columns:
                    tf["is_home"] = tf["venue"].astype(str).str.lower().eq("home").astype("Int8")
                else:
                    tf["is_home"] = 1
                    tf2 = tf.copy()
                    tf2["is_home"] = 0
                    tf = pd.concat([tf, tf2], ignore_index=True)
            fx_long = (tf[["season","team_id",gw_tf,"is_home","fdr"]]
                       .rename(columns={"fdr":"fdr_side"})
                       .dropna(subset=["team_id",gw_tf]))
        else:
            fx_long = pd.DataFrame(columns=["season","team_id",gw_tf,"is_home","fdr_side"])

        if not fx_long.empty:
            if gw_tf != gw_df:
                fx_long = fx_long.rename(columns={gw_tf: gw_df})
            fx_long["fdr_side"] = pd.to_numeric(fx_long["fdr_side"], errors="coerce")

            if "fdr_side" in merged.columns:
                miss_mask = merged["fdr_side"].isna()
                if miss_mask.any():
                    fill = merged.loc[miss_mask].merge(
                        fx_long.groupby(["season","team_id",gw_df,"is_home"], as_index=False)["fdr_side"].max(),
                        how="left", on=["season","team_id",gw_df,"is_home"], validate="many_to_one"
                    )
                    merged.loc[miss_mask, "fdr_side"] = fill["fdr_side"].values
            else:
                merged = merged.merge(
                    fx_long.groupby(["season","team_id",gw_df,"is_home"], as_index=False)["fdr_side"].max(),
                    how="left", on=["season","team_id",gw_df,"is_home"], validate="many_to_one"
                )

    # ---------- Finalize: audit + optionally drop missing ----------
    if merged["fdr_side"].isna().any():
        miss_keys = (merged.loc[merged["fdr_side"].isna(), ["season","team_id",gw_df,"is_home"]]
                     .drop_duplicates()
                     .sort_values(["season","team_id",gw_df,"is_home"]))
        miss_count = int(len(miss_keys))

        # Build richer audit payload (include a few helpful columns if present)
        audit_cols = [c for c in [
            "season", gw_df, "team_id", "team", "is_home", "opponent_id", "opponent",
            "player_id", "player", "pos", "date_sched"
        ] if c in merged.columns]
        audit_df = (merged.loc[merged["fdr_side"].isna(), audit_cols]
                    .drop_duplicates()
                    .sort_values([c for c in [gw_df, "team_id", "player_id"] if c in audit_cols]))

        if audit_dir is not None:
            audit_dir.mkdir(parents=True, exist_ok=True)
            tag = (audit_tag or "unknown").replace(" ", "_")
            audit_path = audit_dir / f"fdr_attach_missing__{tag}.csv"
            # Normalize date for readability
            if "date_sched" in audit_df.columns:
                audit_df = audit_df.copy()
                audit_df["date_sched"] = pd.to_datetime(audit_df["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
            audit_df.to_csv(audit_path, index=False)
            logging.warning("attach_fdr_consistent: wrote %d missing-FDR rows to audit file: %s",
                            miss_count, audit_path)
        else:
            logging.warning("attach_fdr_consistent: %d missing-FDR rows (no audit_dir provided).", miss_count)

        # Provide compact coverage context for logs
        debug_bits = []
        if tf_all is not None and not tf_all.empty:
            gw_tf_all = _pick_gw_col(tf_all.columns.tolist()) or "gw_orig"
            debug_bits.append({
                "source": "team_form",
                "rows": int(len(tf_all)),
                "gw_col": gw_tf_all,
                "seasons": sorted(map(str, pd.unique(tf_all["season"]))),
            })
        if team_fix is not None and not team_fix.empty:
            gw_fx = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"
            has_cols = [c for c in ["fdr","fdr_home","fdr_away","team_fdr_home","team_fdr_away"] if c in team_fix.columns]
            debug_bits.append({
                "source": "team_fix",
                "rows": int(len(team_fix)),
                "gw_col": gw_fx,
                "has_cols": has_cols,
            })
        logging.info("FDR coverage summary: %s", json.dumps(debug_bits))

        if not drop_on_missing:
            raise RuntimeError("attach_fdr_consistent: FDR merge produced NaNs after fallback and drop_on_missing=False.")

        # Drop missing and proceed
        merged = merged[merged["fdr_side"].notna()].copy()

    merged["fdr"] = pd.to_numeric(merged["fdr_side"], errors="raise").astype("Int8")
    merged.drop(columns=["fdr_side"], inplace=True, errors="ignore")
    return merged


# ----------------------------- model loading ---------------------------------

def _load_feature_list(model_dir: Path) -> List[str]:
    txt = model_dir / "artifacts" / "features_used.txt"
    if txt.exists():
        return [line.strip() for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]
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
        p = model_dir / "lgbm_saves_p90.txt"
    return lgb.Booster(model_file=str(p))

def _load_poisson(model_dir: Path) -> Tuple[Optional[object], Optional[object]]:
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
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for team_id, blob in (data or {}).items():
        career = (blob or {}).get("career", {})
        if season not in career: continue
        entry = career[season] or {}
        lg = entry.get("league")
        if league_filter and str(lg) != str(league_filter): continue
        for p in entry.get("players", []) or []:
            pid = str(p.get("id","")).strip()
            if not pid: continue
            rows.append({"season": season, "team_id": str(team_id), "player_id": pid, "player_name_master": p.get("name")})
    if not rows:
        return pd.DataFrame(columns=["season","team_id","player_id","player_name_master"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["season","player_id"])
    for c in ("season","team_id","player_id"):
        df[c] = df[c].astype(str)
    return df


# ----------------------------- output writers --------------------------------

def _saves_out_paths(base_dir: Path, season: str, gw_from: int, gw_to: int,
                     zero_pad: bool, out_format: str) -> List[Path]:
    a = _fmt_gw(gw_from, zero_pad); b = _fmt_gw(gw_to, zero_pad)
    season_dir = base_dir / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    stem = season_dir / f"GW{a}_{b}"
    if out_format == "csv":
        return [Path(str(stem) + ".csv")]
    if out_format == "parquet":
        return [Path(str(stem) + ".parquet")]
    return [Path(str(stem) + ".csv"), Path(str(stem) + ".parquet")]

def _write_saves(df: pd.DataFrame, paths: List[Path]) -> List[str]:
    # Ensure stable dtypes before any write
    df = df.copy()
    _normalize_dates_inplace(df, ("date_sched",))
    for c in ("gw_played","gw_orig"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("team_id","player_id","opponent_id","game_id","season"):
        if c in df.columns:
            df[c] = df[c].astype(str)

    written: List[str] = []
    for p in paths:
        if p.suffix.lower() == ".csv":
            tmp = df.copy()
            if "date_sched" in tmp.columns:
                tmp["date_sched"] = pd.to_datetime(tmp["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
            tmp.to_csv(p, index=False)
            written.append(str(p))
        elif p.suffix.lower() == ".parquet":
            df.to_parquet(p, index=False)
            written.append(str(p))
        else:
            raise ValueError(f"Unsupported output extension: {p.suffix}")
    return written

# ----------------------------- consolidated writer (NEW) ---------------------
def _consolidated_paths(base_dir: Path, season: str) -> List[Path]:
    season_dir = base_dir / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    # keep the existing filenames to avoid breaking downstream consumers
    return [season_dir / "expected_saves.parquet", season_dir / "expected_saves.csv"]

def _update_consolidated(
    new_df: pd.DataFrame,
    base_dir: Path,
    season: str,
    run_gw_from: int,
    run_gw_to: int,
) -> List[str]:
    """
    Merge window output into season-level consolidated file (parquet + csv).
    Latest-window-wins rule per (season, player_id, gw_orig):
      keep the row coming from the batch with the highest `_run_gw_to`.
    """
    if new_df.empty:
        return []

    season_dir = base_dir / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    # Prefer gw_orig as canonical GW key
    gw_key = "gw_orig" if "gw_orig" in new_df.columns else (
        "gw_played" if "gw_played" in new_df.columns else ("gw" if "gw" in new_df.columns else None)
    )
    if gw_key is None:
        return []

    # Tag the incoming batch with run window
    new_df = new_df.copy()
    new_df["_run_gw_from"] = int(run_gw_from)
    new_df["_run_gw_to"] = int(run_gw_to)

    # Load existing consolidated (union parquet+csv if both exist)
    cons_parq = season_dir / "expected_saves.parquet"
    cons_csv  = season_dir / "expected_saves.csv"
    old = pd.DataFrame()
    if cons_parq.exists():
        try:
            old = pd.read_parquet(cons_parq)
        except Exception:
            old = pd.DataFrame()
    if cons_csv.exists():
        try:
            # PARSE the date column on read to avoid object dtype
            old_csv = pd.read_csv(cons_csv, parse_dates=["date_sched"])
            _normalize_dates_inplace(old_csv, ("date_sched",))
            old = old_csv if old.empty else pd.concat([old, old_csv], ignore_index=True)
        except Exception:
            pass

    # Backfill batch-meta columns for older files
    if "_run_gw_from" not in old.columns:
        old["_run_gw_from"] = -1
    if "_run_gw_to" not in old.columns:
        old["_run_gw_to"] = -1

    # Normalize key dtypes BEFORE concat
    for c in ("season","team_id","player_id","opponent_id","game_id"):
        if c in new_df.columns: new_df[c] = new_df[c].astype(str)
        if c in old.columns:    old[c]    = old[c].astype(str)
    for c in ("gw_played","gw_orig","gw"):
        if c in new_df.columns: new_df[c] = pd.to_numeric(new_df[c], errors="coerce").astype("Int64")
        if c in old.columns:    old[c]    = pd.to_numeric(old[c], errors="coerce").astype("Int64")

    _normalize_dates_inplace(new_df, ("date_sched",))
    _normalize_dates_inplace(old, ("date_sched",))

    # Align schemas (outer union) then concat
    all_cols = list(dict.fromkeys([*old.columns.tolist(), *new_df.columns.tolist()]))
    old = old.reindex(columns=all_cols)
    new = new_df.reindex(columns=all_cols)
    merged = pd.concat([old, new], ignore_index=True, sort=False)

    # Temporary datetime for stable sorting (do not persist)
    if "date_sched" in merged.columns:
        merged["_date_sched_dt"] = pd.to_datetime(merged["date_sched"], errors="coerce")
    else:
        merged["_date_sched_dt"] = pd.NaT

    # Latest-window-wins de-dup key
    dedup_key = [c for c in ["season","player_id","gw_orig"] if c in merged.columns]
    if not dedup_key:
        dedup_key = [c for c in ["season","player_id",gw_key] if c in merged.columns]

    merged = merged.sort_values(
        dedup_key + ["_run_gw_to"] + (["_date_sched_dt"] if "_date_sched_dt" in merged.columns else []),
        kind="mergesort"
    ).drop_duplicates(subset=dedup_key, keep="last")

    # Final ordering for output and cleanup
    sort_keys = [k for k in [gw_key, "team_id", "player_id"] if k in merged.columns]
    if sort_keys:
        merged = merged.sort_values(sort_keys + ["_date_sched_dt"], kind="mergesort").reset_index(drop=True)
    merged = merged.drop(columns=["_date_sched_dt"], errors="ignore")

    # Ensure final dtypes are stable before writing Parquet
    _normalize_dates_inplace(merged, ("date_sched",))
    for c in ("gw_played","gw_orig","gw"):
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("Int64")
    for c in ("season","team_id","player_id","opponent_id","game_id"):
        if c in merged.columns:
            merged[c] = merged[c].astype(str)

    # Write parquet + csv (csv with date-only date_sched)
    parq_path, csv_path = _consolidated_paths(base_dir, season)
    merged.to_parquet(parq_path, index=False)

    csv_out = merged.copy()
    if "date_sched" in csv_out.columns:
        csv_out["date_sched"] = pd.to_datetime(csv_out["date_sched"], errors="coerce").dt.strftime("%Y-%m-%d")
    csv_out.to_csv(csv_path, index=False)
    return [str(parq_path), str(csv_path)]

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

    # Minutes input (auto-resolve + dual loader)
    ap.add_argument("--minutes-csv", type=Path, help="Explicit minutes file (CSV or Parquet). Overrides auto-resolution.")
    ap.add_argument("--minutes-root", type=Path, default=Path("data/predictions/minutes"),
                    help="Root containing <season>/GW<from>_<to>.csv|parquet")

    # Models & output
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/predictions/saves"))
    ap.add_argument("--out-format", choices=["csv","parquet","both"], default="csv",
                    help="Output format (default: csv)")
    ap.add_argument("--zero-pad-filenames", action="store_true",
                    help="Write filenames as GW05_06 instead of GW5_6")
    ap.add_argument("--require-pred-minutes", action="store_true")

    # roster gating
    ap.add_argument("--teams-json", type=Path, help="Master teams JSON with per-season rosters")
    ap.add_argument("--league-filter", default="ENG-Premier League",
                    help="Only accept teams whose career[season].league matches this (set '' to disable)")
    ap.add_argument("--require-on-roster", action="store_true",
                    help="Error if any minutes rows are not on the season roster; otherwise drop them")

    ap.add_argument("--log-level", default="INFO")

    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    # --- seasons & as-of ---
    history = [s.strip() for s in args.history_seasons.split(",") if s.strip()]
    seasons_all = history + [args.future_season]

    gw_from_req = args.gw_from if args.gw_from is not None else args.as_of_gw
    gw_to_req   = args.gw_to   if args.gw_to   is not None else (int(gw_from_req) + max(1, int(args.n_future)) - 1)

    as_of_ts = (pd.Timestamp.now(tz=args.as_of_tz)
                if str(args.as_of).lower() in ("now","today","auto")
                else pd.Timestamp(args.as_of, tz=args.as_of_tz))

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

    # --- minutes target window (auto-resolve + dual loader) ---
    minutes_path = _resolve_minutes_path(args, int(gw_from_req), int(gw_to_req))
    minutes = _load_minutes_dual(minutes_path)
    if "season" not in minutes.columns:
        minutes["season"] = args.future_season

    # GK-only early
    if "pos" in minutes.columns:
        minutes = minutes[minutes["pos"].astype(str).str.upper().eq("GK")].copy()

    gw_sel = _gw_for_selection(minutes)
    avail_gws = sorted(pd.unique(gw_sel.dropna().astype("Int8"))) if not gw_sel.dropna().empty else []
    target_gws = [int(g) for g in avail_gws if g >= int(gw_from_req)][: int(args.n_future)]
    if not target_gws:
        raise RuntimeError(f"No target GWs >= {gw_from_req}. Available: {avail_gws}")
    if args.strict_n_future and len(target_gws) < int(args.n_future):
        raise RuntimeError(f"Only {len(target_gws)} GW(s) available; wanted {args.n_future}. Available: {avail_gws}")

    gw_key_m = _pick_gw_col(minutes.columns.tolist()) or "gw_orig"
    minutes = minutes[gw_sel.isin(target_gws)].copy()
    if args.require_pred_minutes and "pred_minutes" not in minutes.columns:
        raise KeyError("--require-pred-minutes set but minutes file lacks pred_minutes")

    # --- roster gate ---
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
        minutes = minutes.merge(roster[["player_id"]].drop_duplicates(), on="player_id",
                                how="inner", validate="many_to_one")
        dropped = before - len(minutes)
        if dropped > 0:
            logging.info("Roster gate dropped %d fixture rows not on %s roster.", dropped, args.future_season)
        if args.require_on_roster and dropped > 0:
            raise RuntimeError(f"--require-on-roster set: {dropped} rows not on the {args.future_season} roster.")

    if minutes.empty:
        raise RuntimeError("No minutes rows left after GK filter/roster gating/target GW selection.")

    # --- team fixtures load (for venue + legacy meta) ---
    try:
        team_fix = pd.read_csv(args.fix_root / args.future_season / args.team_fixtures_filename)
        # normalize any dates
        if "date_sched" in team_fix.columns:
            team_fix["date_sched"] = pd.to_datetime(team_fix["date_sched"], errors="coerce").dt.normalize()
        if "date_played" in team_fix.columns:
            team_fix["date_played"] = pd.to_datetime(team_fix["date_played"], errors="coerce").dt.normalize()

        if "is_home" not in team_fix.columns:
            if "was_home" in team_fix.columns:
                team_fix["is_home"] = team_fix["was_home"].astype(str).str.lower().isin(["1","true","yes"]).astype("Int8")
            elif "venue" in team_fix.columns:
                team_fix["is_home"] = team_fix["venue"].astype(str).str.lower().eq("home").astype("Int8")
            else:
                team_fix["is_home"] = 0
        for c in ("gw_played","gw_orig","gw"):
            if c in team_fix.columns:
                team_fix[c] = pd.to_numeric(team_fix[c], errors="coerce")
        if "season" not in team_fix.columns:
            team_fix["season"] = args.future_season
        if "team_id" in team_fix.columns:
            team_fix["team_id"] = team_fix["team_id"].astype(str)
        if "fbref_id" in team_fix.columns and "game_id" not in team_fix.columns:
            team_fix["game_id"] = team_fix["fbref_id"]
    except FileNotFoundError:
        team_fix = None

    # --- venue_bin + is_home ---
    venue_bin = _resolve_venue(minutes, team_fix, args.future_season)
    fut = minutes.copy()
    fut["venue_bin"] = venue_bin.astype("Int8")
    fut["is_home"] = fut["venue_bin"].astype("Int8")  # canonicalize

    # Prepare audit directory + tag for any FDR attach misses
    audit_dir = (args.out_dir / str(args.future_season) / "artifacts")
    audit_tag = f"GW{int(gw_from_req)}_{int(gw_to_req)}"

    # --- FDR attach (INT, venue-consistent, DGW-safe) ---
    fut = attach_fdr_consistent(
        df=fut,
        seasons_all=seasons_all,
        features_root=args.features_root,
        version=args.form_version,
        team_form=tf,
        team_fix=team_fix,
        audit_dir=audit_dir,
        audit_tag=audit_tag,
        drop_on_missing=True,   # write audit + drop + continue
    )
    fut["fdr"] = pd.to_numeric(fut["fdr"], errors="raise").astype("Int8")

    # >>> Guard: if everything was dropped, write empty outputs and exit cleanly
    if fut.empty:
        logging.warning("No rows remain after FDR attach (all dropped or no coverage). Writing empty outputs and exiting.")
        gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
        empty_cols = [
            "season","gw_played","gw_orig","date_sched","game_id",
            "team_id","team","opponent_id","opponent","is_home","venue_bin",
            "player_id","player","pos","fdr",
            "pred_minutes","team_def_z_venue","opp_att_z_venue",
            "pred_saves_p90_mean","pred_saves_mean",
            "pred_saves_p90_poisson","pred_saves_poisson"
        ]
        out = pd.DataFrame(columns=empty_cols)
        out_paths = _saves_out_paths(
            base_dir=args.out_dir,
            season=args.future_season,
            gw_from=gw_from_eff,
            gw_to=gw_to_eff,
            zero_pad=args.zero_pad_filenames,
            out_format=args.out_format,
        )
        _ = _write_saves(out, out_paths)
        _ = _update_consolidated(
            new_df=out,
            base_dir=args.out_dir,
            season=args.future_season,
            run_gw_from=gw_from_eff,
            run_gw_to=gw_to_eff,
        )
        diag = {
            "rows": 0,
            "season": str(args.future_season),
            "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
            "available_team_gws": [int(x) for x in avail_gws],
            "scored_gws": [int(x) for x in target_gws],
            "as_of": str(as_of_ts),
            "minutes_in": str(minutes_path),
            "out": [str(p) for p in out_paths],
            "consolidated_out": [],
            "roster_gate": {
                "enabled": bool(args.teams_json),
                "league_filter": (args.league_filter if args.league_filter != "" else None)
            },
            "note": "All fixtures for this window lacked FDR coverage; empty outputs written. See audit CSV."
        }
        print(json.dumps(diag, indent=2))
        return
    # <<< End guard

    # --- attach team Zs ---
    if tz_map is not None and gw_key_m in fut.columns:
        fut = fut.merge(
            tz_map.rename(columns={"gw_orig": gw_key_m}),
            how="left", on=["season", gw_key_m, "team_id"], validate="many_to_one"
        )
    else:
        fut["team_def_z_venue"] = np.nan
        fut["opp_att_z_venue"] = np.nan

    # --- last snapshot per (season, player) strictly before as_of ---
    future_only = {"venue_bin","team_def_z_venue","opp_att_z_venue","fdr","is_home"}
    snap_cols = [c for c in feat_cols if c not in future_only]
    for c in snap_cols:
        if c not in pf_hist.columns:
            pf_hist[c] = np.nan

    sort_cols = ["player_id","season","date_played"]
    gw_key_hist = _pick_gw_col(pf_hist.columns.tolist())
    if gw_key_hist:
        sort_cols.append(gw_key_hist)
    pf_hist = pf_hist.sort_values(sort_cols)
    last = pf_hist.groupby(["season","player_id"], as_index=False).tail(1)

    fut = fut.merge(last[["season","player_id"] + snap_cols], how="left",
                    on=["season","player_id"], validate="many_to_one")

    # --- legacy metadata from fixtures (fill if missing) ---
    if team_fix is not None:
        gw_key_t = _pick_gw_col(team_fix.columns.tolist()) or "gw_orig"
        fix_keep = ["season","team_id",gw_key_t,"game_id","fbref_id","team","opponent_id","home","away","is_home"]
        fix_keep = [c for c in fix_keep if c in team_fix.columns]
        if fix_keep:
            fix_small = (team_fix[fix_keep]
                         .dropna(subset=[gw_key_t,"team_id"])
                         .drop_duplicates()
                         .rename(columns={gw_key_t: gw_key_m}))
            fut = fut.merge(fix_small, how="left",
                            on=["season","team_id",gw_key_m,"is_home"],
                            validate="many_to_one", suffixes=("", "_fix"))
            # derive opponent if needed
            if "opponent" not in fut.columns or fut["opponent"].isna().all():
                if {"home","away","is_home"}.issubset(fut.columns):
                    ih = pd.to_numeric(fut["is_home"], errors="coerce").fillna(0).astype("Int8")
                    fut["opponent"] = np.where(ih == 1, fut.get("away"), fut.get("home"))

    # ----------------------------- feature matrix -----------------------------
    X = pd.DataFrame(index=fut.index)
    for c in feat_cols:
        if c in fut.columns:
            X[c] = pd.to_numeric(fut[c], errors="coerce")
        else:
            X[c] = np.nan

    Xn = X.select_dtypes(include=[np.number]).fillna(0.0)
    gk_p90_mean = np.clip(lgb_booster.predict(Xn), 0, None)

    # Optional Poisson/Tweedie head
    if pois_glm is not None and pois_imputer is not None:
        Xp = pois_imputer.transform(Xn)
        gk_p90_pois = np.clip(pois_glm.predict(Xp), 0, None)
    else:
        gk_p90_pois = np.full(len(Xn), np.nan, dtype=float)

    # Scale to per-match
    if "pred_minutes" not in fut.columns:
        fut["pred_minutes"] = fut.get("minutes", pd.Series(np.zeros(len(fut))))
    scale = pd.to_numeric(fut["pred_minutes"], errors="coerce").fillna(0.0).to_numpy() / 90.0
    pred_saves_mean  = gk_p90_mean * scale
    pred_saves_pois  = gk_p90_pois * scale if not np.isnan(gk_p90_pois).all() else gk_p90_pois

    # ---------------------- legacy metadata helpers ----------------------
    def pick_col(df: pd.DataFrame, *names: str):
        for n in names:
            if n in df.columns:
                return df[n].values
        return np.full(len(df), np.nan, dtype=object)

    opponent_id_vals    = pick_col(fut, "opponent_id", "opp_team_id", "opp_id")
    opponent_name_vals  = pick_col(fut, "opponent", "opp_team", "opp_name")
    fbref_or_game_vals  = pick_col(fut, "game_id", "fbref_id", "fixture_id")
    team_name_vals      = pick_col(fut, "team", "team_name", "team_short", "team_long")

    gw_played_vals = pd.to_numeric(fut.get("gw_played", np.nan), errors="coerce").values
    gw_orig_vals   = pd.to_numeric(fut.get(gw_key_m, np.nan), errors="coerce").values

    # Assemble GK-only output
    out = pd.DataFrame({
        "season":      fut["season"].values,
        "gw_played":   gw_played_vals,
        "gw_orig":     gw_orig_vals,
        "date_sched":  pd.to_datetime(fut.get("date_sched"), errors="coerce").dt.normalize().values,
        "venue_bin":   pd.to_numeric(fut.get("venue_bin", np.nan), errors="coerce").fillna(0).astype(int).values,
        "game_id":     fbref_or_game_vals,
        "team_id":     fut.get("team_id", pd.Series([np.nan]*len(fut))).astype(str).values,
        "team":        team_name_vals,
        "opponent_id": opponent_id_vals,
        "opponent":    opponent_name_vals,
        "is_home":     pd.to_numeric(fut["is_home"], errors="coerce").fillna(0).astype("Int8").values,
        "player_id":   fut["player_id"].astype(str).values,
        "player":      fut.get("player", pd.Series([np.nan]*len(fut))).values,
        "pos":         (fut.get("pos", pd.Series(["GK"]*len(fut))).astype(str).str.upper()).values,
        "fdr":         pd.to_numeric(fut.get("fdr", np.nan), errors="raise").astype("Int8").values,
        "pred_minutes": pd.to_numeric(fut["pred_minutes"], errors="coerce").values,
        # context
        "team_def_z_venue": fut.get("team_def_z_venue", pd.Series([np.nan]*len(fut))).values,
        "opp_att_z_venue":  fut.get("opp_att_z_venue",  pd.Series([np.nan]*len(fut))).values,
        # predictions
        "pred_saves_p90_mean":    gk_p90_mean,
        "pred_saves_mean":        pred_saves_mean,
        "pred_saves_p90_poisson": gk_p90_pois,
        "pred_saves_poisson":     pred_saves_pois,
    })

    # GK-only hard filter (paranoia)
    out = out[out["pos"].astype(str).str.upper().eq("GK")].reset_index(drop=True)

    # ---- enforce integer dtypes (nullable) ----
    if "gw_played" in out.columns:
        out["gw_played"] = pd.to_numeric(out["gw_played"], errors="coerce").astype("Int16")
    if "gw_orig" in out.columns:
        out["gw_orig"] = pd.to_numeric(out["gw_orig"], errors="coerce").astype("Int16")
    if "fdr" in out.columns:
        out["fdr"] = pd.to_numeric(out["fdr"], errors="coerce").astype("Int8")
    if "is_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").astype("Int8")
    if "venue_bin" in out.columns:
        out["venue_bin"] = pd.to_numeric(out["venue_bin"], errors="coerce").astype("Int8")

    # Desired column order (metadata → player → context → preds)
    desired_order = [c for c in [
        "season","gw_played","gw_orig","date_sched","game_id",
        "team_id","team","opponent_id","opponent","is_home","venue_bin",
        "player_id","player","pos","fdr",
        "pred_minutes","team_def_z_venue","opp_att_z_venue",
        "pred_saves_p90_mean","pred_saves_mean",
        "pred_saves_p90_poisson","pred_saves_poisson"
    ] if c in out.columns]
    out = out[desired_order].copy()

    # Sort & persist
    sort_keys = [k for k in ["gw_orig","team_id","player_id"] if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys).reset_index(drop=True)

    # --- schema validation before write ---
    SAV_SCHEMA = {
    "required": ["season","gw_played","gw_orig","date_sched","game_id","team_id","team",
                "opponent_id","opponent","is_home","player_id","player","pos","fdr",
                "pred_minutes","team_def_z_venue","opp_att_z_venue",
                "pred_saves_p90_mean","pred_saves_mean",
                "pred_saves_p90_poisson","pred_saves_poisson"],
    "dtypes": {
        "season":"string","gw_played":"Int64","gw_orig":"Int64","date_sched":"datetime64[ns]",
        "game_id":"object","team_id":"string","team":"object","opponent_id":"object","opponent":"object",
        "is_home":"Int64","player_id":"string","player":"object","pos":"string","fdr":"Int64",
        "pred_minutes":"float","team_def_z_venue":"float","opp_att_z_venue":"float",
        "pred_saves_p90_mean":"float","pred_saves_mean":"float",
        "pred_saves_p90_poisson":"float","pred_saves_poisson":"float",
    },
    "na": {"gw_orig": False, "fdr": False, "is_home": False},
    "ranges": {
        "pred_minutes":{"min":0.0,"max":120.0},
        "pred_saves_p90_mean":{"min":0.0}, "pred_saves_mean":{"min":0.0}
    },
    "choices": {"pos":{"in":["GK"]}},
    "logic": [("is_home in {0,1}", ["is_home"])],
    "date_rules": {"normalize":["date_sched"]},
    "unique": ["season","gw_orig","team_id","player_id"]
    }
    validate_df(out, SAV_SCHEMA, name="saves_forecast")

    gw_from_eff, gw_to_eff = int(min(target_gws)), int(max(target_gws))
    out_paths = _saves_out_paths(
        base_dir=args.out_dir,
        season=args.future_season,
        gw_from=gw_from_eff,
        gw_to=gw_to_eff,
        zero_pad=args.zero_pad_filenames,
        out_format=args.out_format,
    )
    written_paths = _write_saves(out, out_paths)

    # ---- update consolidated (season-level) ----
    consolidated_paths = _update_consolidated(
        new_df=out,
        base_dir=args.out_dir,
        season=args.future_season,
        run_gw_from=gw_from_eff,
        run_gw_to=gw_to_eff,
    )

    diag = {
        "rows": int(len(out)),
        "season": str(args.future_season),
        "requested": {"gw_from": int(gw_from_req), "gw_to": int(gw_to_req), "n_future": int(args.n_future)},
        "available_team_gws": [int(x) for x in avail_gws],
        "scored_gws": [int(x) for x in target_gws],
        "as_of": str(as_of_ts),
        "minutes_in": str(minutes_path),
        "out": written_paths,
        "consolidated_out": consolidated_paths,
        "roster_gate": {
            "enabled": bool(args.teams_json),
            "league_filter": (args.league_filter if args.league_filter != "" else None)
        }
    }
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
