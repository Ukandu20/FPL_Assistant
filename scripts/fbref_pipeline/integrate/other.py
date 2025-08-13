#!/usr/bin/env python3
"""
calendar_builder.py – build player_minutes_calendar.csv and enrich with FPL price/xP

Adds:
  • price (value / 10)
  • xP (expected points)

Join key (strict, non-duplicating):
  minutes×fixtures:  (player_id, date_played.date(), venue→_was_home)
  merged_gws.csv:    (player_id, date_played, was_home)

Row count is validated to remain unchanged.
"""

from __future__ import annotations
import argparse, logging, json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np


# ───────────────────────── helpers ─────────────────────────

def load_fixture_calendar(season_dir: Path) -> pd.DataFrame:
    fp = season_dir / "fixture_calendar.csv"
    return pd.read_csv(
        fp,
        usecols=[
            "fbref_id","fpl_id","gw_orig","date_played",
            "team_id","team","venue","gf","ga","fdr_home","fdr_away",
        ],
    )

def load_roster_jsons(season_dir: Path, fbref_root: Path) -> pd.DataFrame:
    season_key = season_dir.name
    master_fp = fbref_root / "master_teams.json"
    players_fp = fbref_root / season_key / "player_season" / "season_players.json"

    with master_fp.open(encoding="utf-8") as f:
        teams: Dict = json.load(f)
    with players_fp.open(encoding="utf-8") as f:
        players: Dict = json.load(f)

    # map short-code → team_id via the fixture calendar
    fix = pd.read_csv(season_dir / "fixture_calendar.csv", usecols=["team","team_id"]).drop_duplicates()
    code2id = dict(zip(fix["team"], fix["team_id"]))

    rows: list[dict[str, str]] = []
    for team_id, rec in teams.items():
        for year, blob in rec.get("career", {}).items():
            if year == season_key:
                for p in blob["players"]:
                    rows.append({"player_id": p["id"], "team_id": team_id})
    roster_mt = pd.DataFrame(rows)

    extras: list[dict[str, str]] = []
    seen = set(roster_mt["player_id"].tolist())
    for p in players.values():
        pid, code = p["player_id"], p["team"]
        if pid in seen: 
            continue
        extras.append({"player_id": pid, "team_id": code2id.get(code)})
    roster_sp = pd.DataFrame(extras)

    roster = pd.concat([roster_mt, roster_sp], ignore_index=True).drop_duplicates()
    if roster["team_id"].isna().any():
        bad = roster.loc[roster.team_id.isna(), "player_id"].head(5).tolist()
        raise ValueError(f"{roster.team_id.isna().sum()} players missing team_id (e.g. {bad}).")
    return roster.reset_index(drop=True)

def load_minutes(season_dir: Path, fbref_root: Path) -> pd.DataFrame:
    season_key = season_dir.name
    summary_fp = fbref_root / season_key / "player_match" / "summary.csv"
    keeper_fp  = fbref_root / season_key / "player_match" / "keepers.csv"
    def_fp     = fbref_root / season_key / "player_match" / "defense.csv"
    misc_fp    = fbref_root / season_key / "player_match" / "misc.csv"

    df = pd.read_csv(
        summary_fp,
        usecols=[
            "game_id","player_id","player","min","team_id",
            "crdy","crdr","fpl_pos","gls","ast","xg","npxg","xag","pkatt","pk","sh","sot"
        ]
    ).rename(columns={
        "game_id":"fbref_id",
        "min":"minutes",
        "crdy":"yellow_crd",
        "crdr":"red_crd",
        "fpl_pos":"pos",
        "pk":"pk_scored",
        "sh":"shots",
    })

    df_gk = pd.read_csv(keeper_fp, usecols=["game_id","player_id","team_id","sota","saves","save"]) \
            .rename(columns={"game_id":"fbref_id","sota":"sot_against","save":"save_pct"})
    df_def = pd.read_csv(def_fp, usecols=["game_id","player_id","team_id","blocks","tklw","int","clr"]) \
             .rename(columns={"game_id":"fbref_id","tklw":"tkl"})
    df_misc = pd.read_csv(misc_fp, usecols=["game_id","player_id","team_id","recov","pkwon","og"]) \
              .rename(columns={"game_id":"fbref_id","recov":"recoveries","pkwon":"pk_won","og":"own_goals"})

    df = df.merge(df_gk,  on=["fbref_id","player_id","team_id"], how="left")
    df = df.merge(df_def, on=["fbref_id","player_id","team_id"], how="left")
    df = df.merge(df_misc,on=["fbref_id","player_id","team_id"], how="left")

    for col in ("sot_against","saves","save_pct"):
        df[col] = df[col].fillna(0)

    return df

def _find_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for n in names:
        if n in df.columns: 
            return n
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def load_fpl_price_xp_by_date(fpl_root: Path, season_key: str) -> pd.DataFrame:
    """
    Load data/processed/fpl/<season>/gws/merged_gws.csv and return UNIQUE rows on:
      (player_id, date_played, was_home)
    Columns returned: ['player_id','date_played','was_home','price','xP']
    """
    fp = fpl_root / season_key / "gws" / "merged_gws.csv"
    if not fp.exists():
        logging.info("No FPL merged_gws at %s – skipping price/xP enrichment", fp)
        return pd.DataFrame(columns=["player_id","game_date","price","xP"])

    gws = pd.read_csv(fp)
    gws = gws.rename(columns={"game_date": "date_played"})

    # player id
    pid_col = _find_col(gws, ["player_id"])
    if pid_col is None:
        logging.info("merged_gws lacks player_id/element – skipping enrichment")
        return pd.DataFrame(columns=["player_id","date_played","price","xP"])
    if pid_col != "player_id":
        gws = gws.rename(columns={pid_col:"player_id"})
    gws["player_id"] = pd.to_numeric(gws["player_id"], errors="coerce")

    # value/xP
    val_col = _find_col(gws, ["value","price"])
    xp_col  = _find_col(gws, ["xP","expected_points","xp","xpts"])
    if val_col and val_col != "value":
        gws = gws.rename(columns={val_col:"value"})
    if xp_col and xp_col != "xP":
        gws = gws.rename(columns={xp_col:"xP"})

    if "value" in gws.columns:
        gws["value"] = pd.to_numeric(gws["value"], errors="coerce")
        gws["price"] = (gws["value"] / 10.0).round(1)  # adjust if your 'value' is already 4.7 not 47
    else:
        gws["price"] = np.nan

    if "xP" in gws.columns:
        gws["xP"] = pd.to_numeric(gws["xP"], errors="coerce")
    else:
        gws["xP"] = np.nan

    gws = gws[["player_id","date_played","price","xP"]].dropna(subset=["player_id","date_played"])
    # one row per (player_id, date_played)
    gws = gws.sort_values(["player_id","date_played"]).drop_duplicates(
        subset=["player_id","date_played"], keep="last"
    )

    # assert uniqueness (defensive)
    dup_mask = gws.duplicated(["player_id","date_played"])
    if dup_mask.any():
        raise RuntimeError("merged_gws still has duplicate (player_id, date_played) after dedupe.")

    return gws.reset_index(drop=True)


# ───────────────────────── main pipeline ─────────────────────────

def build_minutes_calendar(season_dir: Path, fbref_root: Path, fpl_root: Path, force: bool = False) -> None:
    out_fp = season_dir / "player_minutes_calendar.csv"
    if out_fp.exists() and not force:
        logging.info("%s exists – skipping", out_fp.name)
        return

    season_key = season_dir.name

    cal = load_fixture_calendar(season_dir)
    cal["date_played"] = pd.to_datetime(cal["date_played"], errors="coerce")

    minutes = load_minutes(season_dir, fbref_root)

    # Base join defining row count
    merged = minutes.merge(cal, on=["fbref_id","team_id"], how="left")
    miss = merged["date_played"].isna().sum()
    if miss:
        logging.warning("%d rows with missing fixture data dropped", miss)
        merged = merged.dropna(subset=["date_played"])

    merged["is_active"] = (merged["minutes"] > 0).astype("uint8")
    merged = merged.sort_values(["player_id","date_played"])
    merged["days_since_last"] = (
        merged.groupby("player_id")["date_played"].diff().dt.days.fillna(0).astype(int)
    )

    # Derive join keys for FPL enrichment
    merged["date_played"] = merged["date_played"].dt.date
    merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce")

    before_rows = len(merged)
    fpl = load_fpl_price_xp_by_date(fpl_root, season_key)

    if not fpl.empty:
        # Validate right-side uniqueness (m:1 required)
        assert not fpl.duplicated(["player_id","date_played"]).any()

        merged = merged.merge(
            fpl,
            on = ["player_id","date_played"],
            how="left",
            validate="m:1",
        )

        after_rows = len(merged)
        if after_rows != before_rows:
            raise RuntimeError(f"Row count changed after FPL merge: {before_rows} → {after_rows}")

        # Coverage logging
        cov_price = merged["price"].notna().mean() * 100
        cov_xp    = merged["xP"].notna().mean() * 100
        logging.info("FPL enrichment coverage: price=%.1f%%, xP=%.1f%%", cov_price, cov_xp)
    else:
        merged["price"] = np.nan
        merged["xP"] = np.nan

    # Output
    out_cols = [
        "player_id","player","pos","fbref_id","fpl_id","gw_orig","date_played",
        "team_id","team","minutes","days_since_last","is_active",
        "yellow_crd","red_crd","venue","gf","ga","fdr_home","fdr_away",
        "gls","ast","shots","sot","blocks","tkl","int","clr",
        "xg","npxg","xag","pkatt","pk_scored","pk_won",
        "saves","sot_against","save_pct","own_goals","recoveries",
        "price","xP",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(out_fp, index=False)
    logging.info("✅ %s written (%d rows)", out_fp.name, len(merged))


def run_batch(seasons: List[str], fixtures_root: Path, fbref_root: Path, fpl_root: Path, force: bool) -> None:
    for season in seasons:
        season_dir = fixtures_root / season
        if not season_dir.is_dir():
            logging.warning("Season %s missing – skipped", season); continue
        try:
            build_minutes_calendar(season_dir, fbref_root, fpl_root, force=force)
        except Exception:
            logging.exception("❌ %s failed", season)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", type=Path, default=Path("data/processed/fixtures"))
    ap.add_argument("--fbref-root",    type=Path, default=Path("data/processed/fbref/ENG-Premier League"))
    ap.add_argument("--fpl-root",      type=Path, default=Path("data/processed/fpl"))
    ap.add_argument("--season")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    seasons = [args.season] if args.season else sorted(d.name for d in args.fixtures_root.iterdir() if d.is_dir())
    if not seasons:
        logging.error("No seasons found under %s", args.fixtures_root); return
    run_batch(seasons, args.fixtures_root, args.fbref_root, args.fpl_root, args.force)

if __name__ == "__main__":
    main()
