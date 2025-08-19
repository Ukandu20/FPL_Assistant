#!/usr/bin/env python3
"""
total_points_combiner.py — v1.2 (production)

Collapse per-fixture expected points into per-GW per-player totals (handles DGWs).
Always produces pred_exp_minutes in the output:
  • carry through if present in expected_points.csv
  • else pull from --minutes-csv (minutes_predictions.csv)
  • else derive from p1/p60 (mins ≈ 90*p60 + 30*max(p1-p60, 0))

Inputs
------
--xp-csv         Path to expected_points.csv (from expected_points_aggregator.py)
--minutes-csv    Optional minutes_predictions.csv (fallback source for pred_exp_minutes)
--season         e.g., 2024-2025 (tolerant to 2024–2025 / 24-25 forms)
--gws            Comma list of GWs to keep (e.g., 30,31,32). If omitted, keeps all GWs in file for --season.
--prices-json    Optional registry JSON to attach price per GW as price (tenths of £m)
--out-dir        Base output directory (e.g., data/models/expected_points_gw)
--version        Version subdir under out-dir (e.g., v1). Use --auto-version to pick next.
--auto-version   If set, picks the next vN under out-dir
--write-latest   If set, updates <out-dir>/latest pointer

Outputs
-------
<out-dir>/<version>/xp_by_gw.csv
<out-dir>/<version>/xp_by_gw.meta.json
"""

from __future__ import annotations
import argparse, json, logging, os, re, datetime as dt
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

SCHEMA_VERSION = "v1.2"

KEY_FIXTURE = ["season","gw_orig","date_played","player_id","team_id"]
KEY_GW      = ["season","gw_orig","player_id","team_id","pos","player"]

# ───────────────────────── Season canonicalization ─────────────────────────
_dash = re.compile(r"[–—/]")
def _canon_season(s: str) -> str:
    s = _dash.sub("-", str(s).strip())
    m = re.match(r"^\s*(\d{2,4})\s*-\s*(\d{2,4})\s*$", s)
    if not m:
        return s
    y1 = int(m.group(1)); y2 = m.group(2)
    if len(m.group(1)) == 2: y1 += 2000
    if len(y2) == 2: y2 = str(int(y2) + (y1 // 100) * 100)
    return f"{y1}-{int(y2)}"

# ───────────────────────── Auto-version helpers ─────────────────────────
def _resolve_version(base_dir: Path, requested: Optional[str], auto: bool) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    if auto or (not requested) or (requested.lower() == "auto"):
        existing = [p.name for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
        nxt = (max(int(s[1:]) for s in existing) + 1) if existing else 1
        ver = f"v{nxt}"
        logging.info("[info] auto-version -> %s", ver)
        return ver
    if not re.fullmatch(r"v\d+", requested):
        if requested and requested.isdigit():
            return f"v{requested}"
        raise ValueError("--version must look like v3, or pass --auto-version")
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

# ───────────────────────── Price helpers ─────────────────────────
def _pick_price_entry(gw_dict: dict, gw: int):
    sgw = str(gw)
    if sgw in gw_dict:
        return gw_dict[sgw]
    keys = sorted(int(k) for k in gw_dict.keys() if k.isdigit() and int(k) <= gw)
    return gw_dict[str(keys[-1])] if keys else None

def load_prices_json(fp: Path, season_gws: List[int]) -> pd.DataFrame:
    """
    Return per-GW prices: player_id, gw_orig, price (tenths).
    Auto-detect units:
      - If price >= 30 → treat as tenths (e.g., 106 -> 10.6m) → tenths = round(price)
      - Else → treat as £m (e.g., 10.6) → tenths = round(price * 10)
    Also accept keys named 'price', 'price', or 'price_tenths'.
    """
    reg = json.loads(Path(fp).read_text("utf-8"))
    players = reg.get("players", {})
    rows = []
    for pid, pdata in players.items():
        gwd = pdata.get("gw", {})
        for gw in season_gws:
            ent = _pick_price_entry(gwd, int(gw))
            if not ent:
                continue

            # Try multiple keys
            raw = ent.get("price", ent.get("price_tenths", ent.get("price")))
            if raw is None:
                continue

            val = float(raw)
            # unit detection
            if val >= 30:              # likely tenths already (e.g., 106 -> 10.6m)
                tenths = int(round(val))
            else:                      # likely £m (e.g., 10.6)
                tenths = int(round(val * 10))

            rows.append({
                "player_id": str(pid),
                "gw_orig": int(gw),
                "price": tenths,    # always tenths in outputs
            })
    return pd.DataFrame(rows)


# ───────────────────────── I/O helpers ─────────────────────────
def _norm(df: pd.DataFrame) -> pd.DataFrame:
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
    # helpful coercions
    for c in ["exp_points_total","pred_exp_minutes","p1","p60"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _read_xp(fp: Path) -> pd.DataFrame:
    head = pd.read_csv(fp, nrows=0)
    parse_dates = ["date_played"] if "date_played" in head.columns else None
    df = pd.read_csv(fp, parse_dates=parse_dates, low_memory=False)
    return _norm(df)

def _read_minutes(fp: Path) -> pd.DataFrame:
    head = pd.read_csv(fp, nrows=0)
    parse_dates = ["date_played"] if "date_played" in head.columns else None
    df = pd.read_csv(fp, parse_dates=parse_dates, low_memory=False)
    df = _norm(df)
    need = set(KEY_FIXTURE) | {"pred_exp_minutes"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"minutes file missing required columns: {miss}")
    return df

# ───────────────────────── Core ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xp-csv", type=Path, required=True)
    ap.add_argument("--minutes-csv", type=Path, default=None, help="Optional minutes_predictions.csv to source pred_exp_minutes")
    ap.add_argument("--season", type=str, required=True)
    ap.add_argument("--gws", type=str, default=None, help="Comma list (e.g., 30,31,32). If omitted: all GWs for --season.")
    ap.add_argument("--prices-json", type=Path, default=None)
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

    xp = _read_xp(args.xp_csv)
    need = {"season","gw_orig","player_id","team_id","pos","player","exp_points_total"}
    miss = need - set(xp.columns)
    if miss:
        raise ValueError(f"expected_points file missing columns: {miss}")

    season_canon = _canon_season(args.season)
    xp = xp[xp["season"] == season_canon].copy()
    if xp.empty:
        raise SystemExit(f"No rows for season={season_canon} in {args.xp_csv}")

    # GW filter
    if args.gws:
        keep_gws = sorted(int(x.strip()) for x in args.gws.split(",") if x.strip())
        xp = xp[xp["gw_orig"].isin(keep_gws)]
        if xp.empty:
            logging.warning("After GW filter %s there are no rows. Check season/GW values.", keep_gws)

    # Ensure we have a minutes column at fixture level (for later GW aggregation)
    minutes_col = None
    if "pred_exp_minutes" in xp.columns:
        minutes_col = "pred_exp_minutes"
    else:
        # Try minutes CSV
        if args.minutes_csv is not None:
            m = _read_minutes(args.minutes_csv)
            m = m[m["season"] == season_canon].copy()
            if args.gws:
                m = m[m["gw_orig"].isin(keep_gws)]
            # bring fixture-level minutes into xp (merge on KEY_FIXTURE)
            xp = xp.merge(m[KEY_FIXTURE + ["pred_exp_minutes"]], on=KEY_FIXTURE, how="left", validate="1:1")
            minutes_col = "pred_exp_minutes"
            logging.info("Filled pred_exp_minutes from minutes-csv for %d rows", xp["pred_exp_minutes"].notna().sum())
        # If still missing, derive from p1/p60
        if minutes_col is None:
            if {"p1","p60"}.issubset(xp.columns):
                xp["mins_approx"] = (
                    90.0 * xp["p60"].clip(0, 1).fillna(0.0) +
                    30.0 * (xp["p1"].clip(0, 1).fillna(0.0) - xp["p60"].clip(0, 1).fillna(0.0)).clip(lower=0.0)
                )
                minutes_col = "mins_approx"
                logging.warning("pred_exp_minutes not present; using mins_approx from p1/p60")
            else:
                xp["mins_approx"] = np.nan
                minutes_col = "mins_approx"
                logging.warning("No pred_exp_minutes nor p1/p60 available; pred_exp_minutes will be NaN")

    # Aggregate to per-player-GW
    comp_cols = [c for c in xp.columns if c.startswith("xp_")] + ["exp_points_total"]
    add_cols  = [minutes_col] if minutes_col in xp.columns else []

    grp = (xp.groupby(KEY_GW, as_index=False)[comp_cols + add_cols]
             .sum(min_count=1))

    # Rename mins_approx -> pred_exp_minutes post-aggregation
    if "mins_approx" in grp.columns and "pred_exp_minutes" not in grp.columns:
        grp = grp.rename(columns={"mins_approx": "pred_exp_minutes"})

    # If upstream already had price at fixture-level and it slipped through, consolidate by max within GW
    if "price" in xp.columns and "price" not in grp.columns:
        price_gw = xp.groupby(KEY_GW, as_index=False)["price"].max()
        grp = grp.merge(price_gw, on=KEY_GW, how="left")

    # Join prices registry (overrides any carried price)
    if args.prices_json:
        gws_present = sorted(grp["gw_orig"].dropna().astype(int).unique().tolist())
        price_df = load_prices_json(args.prices_json, gws_present)
        if not price_df.empty:
            grp = grp.drop(columns=["price"], errors="ignore")
            grp = grp.merge(price_df, on=["player_id","gw_orig"], how="left")
        else:
            grp["price"] = np.nan
    else:
        grp["price"] = grp.get("price", pd.Series(np.nan, index=grp.index))

    # Sort & write
    grp = grp.sort_values(["season","gw_orig","team_id","player_id"]).reset_index(drop=True)

    fp = out_dir / "xp_by_gw.csv"
    grp.to_csv(fp, index=False)
    uniq_gws = ",".join(map(str, sorted(grp['gw_orig'].dropna().unique().astype(int))))
    logging.info("Wrote per-GW xP to %s (rows=%d, gws=%s)",
                 fp.resolve(), len(grp), uniq_gws)

    meta = {
        "schema": SCHEMA_VERSION,
        "build_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_expected_points": str(args.xp_csv),
        "season": season_canon,
        "gws": (sorted(grp["gw_orig"].dropna().astype(int).unique().tolist())),
        "columns": grp.columns.tolist(),
        "version": version,
        "minutes_source": ("xp.pred_exp_minutes"
                           if "pred_exp_minutes" in xp.columns
                           else ("minutes_csv" if args.minutes_csv else "derived_from_p1_p60"))
    }
    (out_dir / "xp_by_gw.meta.json").write_text(json.dumps(meta, indent=2))

    if args.write_latest:
        _write_latest_pointer(args.out_dir, version)

if __name__ == "__main__":
    main()
