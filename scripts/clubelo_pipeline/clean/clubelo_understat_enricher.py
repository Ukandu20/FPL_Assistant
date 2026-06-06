#!/usr/bin/env python3
"""Clean ClubElo histories and enrich processed Understat match files."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

LOG = logging.getLogger("clubelo_understat")

DEFAULT_RAW_CLUBELO_DIR = Path("data/raw/clubelo/team_history")
DEFAULT_UNDERSTAT_ROOT = Path("data/processed/understat")
DEFAULT_OUT_ROOT = Path("data/processed/clubelo")
DEFAULT_TEAMS_CONFIG = Path("data/config/teams.json")
DEFAULT_ALIASES = Path("data/config/clubelo_team_aliases.json")

LEAGUE_TO_CLUBELO: Dict[str, str] = {
    "ENG-Premier League": "ENG_1",
    "ESP-La Liga": "ESP_1",
    "GER-Bundesliga": "GER_1",
    "ITA-Serie A": "ITA_1",
    "FRA-Ligue 1": "FRA_1",
    "RUS-Russian PL": "RUS_1",
}
CLUBELO_TO_LEAGUE: Dict[str, str] = {v: k for k, v in LEAGUE_TO_CLUBELO.items()}

MATCH_FILES = ("schedule.csv", "team_match.csv")
ELO_COLUMNS = [
    "team_start_elo",
    "team_end_elo",
    "opp_start_elo",
    "opp_end_elo",
    "elo_diff_start",
    "elo_diff_end",
]

DEFAULT_TEAM_ALIASES: Dict[str, str] = {
    "Alaves": "ALA",
    "Angers": "SCO",
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Atalanta": "ATA",
    "Atletico": "ATM",
    "Augsburg": "FCA",
    "Auxerre": "AJA",
    "Barcelona": "BAR",
    "Bayern": "FCB",
    "Betis": "BET",
    "Bilbao": "ATH",
    "Bologna": "BOL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brest": "SB29",
    "Brighton": "BHA",
    "Burnley": "BUR",
    "Cagliari": "CAG",
    "Celta": "CEL",
    "Chelsea": "CHE",
    "Como": "COM",
    "Cremonese": "CRE",
    "Crystal Palace": "CRY",
    "Dortmund": "BVB",
    "Elche": "ELC",
    "Espanyol": "ESP",
    "Everton": "EVE",
    "Fiorentina": "FIO",
    "Forest": "NFO",
    "Frankfurt": "SGE",
    "Freiburg": "SCF",
    "Fulham": "FUL",
    "Genoa": "GEN",
    "Getafe": "GET",
    "Girona": "GIR",
    "Gladbach": "BMG",
    "Hamburg": "HSV",
    "Heidenheim": "HDH",
    "Hoffenheim": "TSG",
    "Inter": "INT",
    "Ipswich": "IPS",
    "Juventus": "JUV",
    "Koeln": "KOE",
    "Las Palmas": "LPA",
    "Lazio": "LAZ",
    "Le Havre": "HAC",
    "Lecce": "LEC",
    "Leeds": "LEE",
    "Leganes": "LEG",
    "Leverkusen": "LEV",
    "Lens": "RCL",
    "Levante": "LEV",
    "Leicester": "LEI",
    "Lille": "LIL",
    "Liverpool": "LIV",
    "Lorient": "FCL",
    "Lyon": "OL",
    "Mainz": "M05",
    "Mallorca": "MLL",
    "Man City": "MCI",
    "Man United": "MUN",
    "Marseille": "OM",
    "Metz": "FCM",
    "Milan": "MIL",
    "Monaco": "ASM",
    "Nantes": "FCN",
    "Napoli": "NAP",
    "Newcastle": "NEW",
    "Nice": "OGC",
    "Osasuna": "OSA",
    "Oviedo": "OVI",
    "Paris FC": "PFC",
    "Paris SG": "PSG",
    "Parma": "PAR",
    "Pisa": "PIS",
    "Rayo Vallecano": "RAY",
    "RB Leipzig": "RBL",
    "Real Madrid": "RMA",
    "Rennes": "REN",
    "Roma": "ROM",
    "Sassuolo": "SAS",
    "Sevilla": "SEV",
    "Sociedad": "RSO",
    "Southampton": "SOU",
    "St Pauli": "STP",
    "Strasbourg": "RCS",
    "Stuttgart": "VFB",
    "Sunderland": "SUN",
    "Torino": "TOR",
    "Tottenham": "TOT",
    "Toulouse": "TFC",
    "Udinese": "UDI",
    "Union Berlin": "FCU",
    "Valencia": "VAL",
    "Valladolid": "VLL",
    "Verona": "VER",
    "Villarreal": "VIL",
    "Werder": "SVW",
    "West Ham": "WHU",
    "Wolfsburg": "WOB",
    "Wolves": "WOL",
}


@dataclass(frozen=True)
class SeasonInfo:
    league: str
    season: str
    path: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    @property
    def clubelo_league(self) -> str:
        return LEAGUE_TO_CLUBELO[self.league]


@dataclass(frozen=True)
class EloPair:
    start_elo: Optional[float]
    end_elo: Optional[float]
    reason: str = ""


class TeamResolver:
    def __init__(self, teams_config: Dict[str, str], aliases: Dict[str, str]) -> None:
        self.name_to_code: Dict[str, str] = {}
        self.code_set: set[str] = set()

        for raw_name, raw_code in teams_config.items():
            code = str(raw_code).strip().upper()
            if not code:
                continue
            self.code_set.add(code)
            key = normalize_team_name(raw_name)
            if key and key not in self.name_to_code:
                self.name_to_code[key] = code

        self.alias_to_code: Dict[str, str] = {}
        for raw_key, raw_target in aliases.items():
            key = normalize_team_name(raw_key)
            code = self._resolve_alias_target(raw_target)
            if key and code:
                self.alias_to_code[key] = code

    def _resolve_alias_target(self, raw_target: Any) -> str:
        target = str(raw_target).strip()
        if not target:
            return ""
        upper = target.upper()
        if upper in self.code_set:
            return upper
        by_name = self.name_to_code.get(normalize_team_name(target), "")
        if by_name:
            return by_name
        if re.fullmatch(r"[A-Z0-9]{2,5}", upper):
            return upper
        return ""

    def resolve(self, raw_name: Any) -> str:
        text = "" if pd.isna(raw_name) else str(raw_name).strip()
        if not text:
            return ""
        upper = text.upper()
        if upper in self.code_set:
            return upper
        key = normalize_team_name(text)
        return self.alias_to_code.get(key) or self.name_to_code.get(key, "")


def init_logger(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_team_name(value: Any) -> str:
    text = unescape("" if pd.isna(value) else str(value))
    text = text.replace("\u2019", "'").replace("`", "'").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("&", " and ")
    text = re.sub(r"\b(f\.?c\.?|a\.?f\.?c\.?|c\.?f\.?|club)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def standardize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", str(c).strip().lower())).strip("_")
        for c in out.columns
    ]
    return out


def _load_json_object(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        LOG.warning("Could not read JSON config %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return {str(k): str(v) for k, v in data.items()}


def load_team_resolver(teams_config_path: Path, aliases_path: Path) -> TeamResolver:
    aliases = dict(DEFAULT_TEAM_ALIASES)
    aliases.update(_load_json_object(aliases_path))
    return TeamResolver(_load_json_object(teams_config_path), aliases)


def _safe_filename(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]", "", text)
    return text or "unknown"


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "w", "d", "l"}


def _is_result_row(row: pd.Series) -> bool:
    if "is_result" in row.index:
        return _to_bool(row.get("is_result"))
    result = str(row.get("result", "")).strip().upper()
    if result in {"W", "D", "L"}:
        return True
    for col in ("team_goals", "opp_goals", "goals", "goals_against"):
        if col in row.index and pd.notna(row.get(col)) and str(row.get(col)).strip() != "":
            return True
    return False


def season_fallback_window(season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    match = re.fullmatch(r"(\d{4})-(\d{4})", str(season).strip())
    if not match:
        raise ValueError(f"Unsupported season format: {season!r}")
    return pd.Timestamp(f"{match.group(1)}-08-01"), pd.Timestamp(f"{match.group(2)}-06-30")


def _read_date_columns(path: Path) -> pd.Series:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=lambda c: c in {"game_date", "date"})
    except ValueError:
        return pd.Series(dtype="datetime64[ns]")
    if "game_date" in df.columns:
        raw = df["game_date"]
    elif "date" in df.columns:
        raw = df["date"].astype(str).str.split().str[0]
    else:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(raw, errors="coerce").dropna()


def discover_understat_seasons(
    understat_root: Path,
    *,
    leagues: Optional[Iterable[str]] = None,
    seasons: Optional[Iterable[str]] = None,
) -> list[SeasonInfo]:
    league_filter = set(leagues or [])
    season_filter = set(seasons or [])
    out: list[SeasonInfo] = []

    if not understat_root.exists():
        return out

    for league_dir in sorted(p for p in understat_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        league = league_dir.name
        if league not in LEAGUE_TO_CLUBELO:
            continue
        if league_filter and league not in league_filter:
            continue
        for season_dir in sorted(p for p in league_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            season = season_dir.name
            if season_filter and season not in season_filter:
                continue
            if not any((season_dir / name).exists() for name in MATCH_FILES):
                continue
            dates = [
                _read_date_columns(season_dir / name)
                for name in MATCH_FILES
                if (season_dir / name).exists()
            ]
            all_dates = pd.concat(dates, ignore_index=True).dropna() if dates else pd.Series(dtype="datetime64[ns]")
            if all_dates.empty:
                start_date, end_date = season_fallback_window(season)
            else:
                start_date = all_dates.min().normalize()
                end_date = all_dates.max().normalize()
            out.append(SeasonInfo(league, season, season_dir, start_date, end_date))
    return out


def clean_clubelo_history(df: pd.DataFrame, *, source_file: str, resolver: TeamResolver) -> pd.DataFrame:
    out = standardize_colnames(df).copy()
    if "club" in out.columns and "team" not in out.columns:
        out = out.rename(columns={"club": "team"})

    required = {"from", "team", "country", "level", "elo"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"ClubElo file {source_file} missing columns: {', '.join(sorted(missing))}")

    out["from"] = pd.to_datetime(out["from"], errors="coerce").dt.normalize()
    if "to" in out.columns:
        out["to"] = pd.to_datetime(out["to"], errors="coerce").dt.normalize()
    else:
        out["to"] = pd.NaT
    out["elo"] = pd.to_numeric(out["elo"], errors="coerce")
    if "rank" in out.columns:
        out["rank"] = pd.to_numeric(out["rank"], errors="coerce")

    out["country"] = out["country"].astype(str).str.strip().str.upper()
    out["level"] = out["level"].astype(str).str.strip()
    out["clubelo_league"] = out["country"] + "_" + out["level"]
    out["league"] = out["clubelo_league"].map(CLUBELO_TO_LEAGUE).fillna("")
    out["team"] = out["team"].astype(str).str.strip()
    out["team_code"] = out["team"].apply(resolver.resolve)
    out["clubelo_file"] = source_file

    out = out.dropna(subset=["from", "elo"]).sort_values(["clubelo_league", "team", "from"], kind="stable")
    preferred = [
        "league",
        "clubelo_league",
        "team_code",
        "team",
        "country",
        "level",
        "from",
        "to",
        "elo",
        "rank",
        "clubelo_file",
    ]
    return out[[c for c in preferred if c in out.columns] + [c for c in out.columns if c not in set(preferred)]]


def load_clean_clubelo_histories(
    raw_dir: Path,
    resolver: TeamResolver,
    *,
    clubelo_leagues: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    missing_parts: list[pd.DataFrame] = []
    league_set = set(clubelo_leagues)

    for path in sorted(raw_dir.glob("*.csv")):
        try:
            raw = pd.read_csv(path, dtype=str, keep_default_na=False)
            clean = clean_clubelo_history(raw, source_file=path.name, resolver=resolver)
        except Exception as exc:
            LOG.warning("Skipping ClubElo file %s: %s", path, exc)
            continue
        clean = clean.loc[clean["clubelo_league"].isin(league_set)].copy()
        if clean.empty:
            continue
        frames.append(clean)
        missing = clean.loc[clean["team_code"].eq(""), ["clubelo_league", "league", "team", "clubelo_file"]]
        if not missing.empty:
            missing_parts.append(missing)

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if missing_parts:
        missing_mappings = (
            pd.concat(missing_parts, ignore_index=True)
            .assign(normalized_team=lambda d: d["team"].apply(normalize_team_name))
            .groupby(["clubelo_league", "league", "team", "normalized_team", "clubelo_file"], dropna=False, as_index=False)
            .size()
            .rename(columns={"size": "rows"})
        )
    else:
        missing_mappings = pd.DataFrame(columns=["clubelo_league", "league", "team", "normalized_team", "clubelo_file", "rows"])
    return history, missing_mappings


def write_processed_clubelo_by_season(history: pd.DataFrame, seasons: list[SeasonInfo], out_root: Path) -> int:
    files_written = 0
    if history.empty:
        return files_written

    work = history.copy()
    open_to = pd.Timestamp.max.normalize()
    work["_to_for_filter"] = work["to"].fillna(open_to)

    for info in seasons:
        start = info.start_date - pd.Timedelta(days=2)
        end = info.end_date + pd.Timedelta(days=2)
        mask = (
            work["clubelo_league"].eq(info.clubelo_league)
            & (work["from"] <= end)
            & (work["_to_for_filter"] >= start)
        )
        sliced = work.loc[mask].drop(columns=["_to_for_filter"]).copy()
        if sliced.empty:
            continue
        if "season" in sliced.columns:
            sliced = sliced.drop(columns=["season"])
        sliced.insert(1, "season", info.season)
        sliced = sliced.sort_values(["team_code", "team", "from"], kind="stable")
        out_path = out_root / info.league / info.season / "team_history.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sliced.to_csv(out_path, index=False)
        files_written += 1
    return files_written


def build_elo_lookup(history: pd.DataFrame) -> Dict[tuple[str, str], pd.DataFrame]:
    lookup: Dict[tuple[str, str], pd.DataFrame] = {}
    if history.empty:
        return lookup
    eligible = history.loc[history["league"].ne("") & history["team_code"].ne("")].copy()
    for (league, team_code), group in eligible.groupby(["league", "team_code"], dropna=False):
        group = group.sort_values("from", kind="stable").reset_index(drop=True)
        lookup[(str(league), str(team_code).upper())] = group
    return lookup


def _active_start_elo(hist: pd.DataFrame, match_date: pd.Timestamp) -> Optional[float]:
    open_to = pd.Timestamp.max.normalize()
    active = hist.loc[
        (hist["from"] <= match_date)
        & (hist["to"].fillna(open_to) >= match_date)
    ]
    if active.empty:
        active = hist.loc[hist["from"] <= match_date]
    if active.empty:
        return None
    value = active.iloc[-1]["elo"]
    return None if pd.isna(value) else float(value)


def find_elo_pair(hist: Optional[pd.DataFrame], match_date: Any, *, is_result: bool) -> EloPair:
    date = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(date):
        return EloPair(None, None, "invalid_match_date")
    date = date.normalize()
    if hist is None or hist.empty:
        return EloPair(None, None, "missing_history")

    if is_result:
        candidates = hist.loc[(hist["from"] >= date) & (hist["from"] <= date + pd.Timedelta(days=1))]
        if not candidates.empty:
            post_idx = int(candidates.index[0])
            post_pos = hist.index.get_loc(post_idx)
            end_value = hist.loc[post_idx, "elo"]
            start_value = hist.iloc[post_pos - 1]["elo"] if post_pos > 0 else None
            return EloPair(
                None if pd.isna(start_value) else float(start_value),
                None if pd.isna(end_value) else float(end_value),
            )
        return EloPair(_active_start_elo(hist, date), None, "missing_post_match_elo")

    return EloPair(_active_start_elo(hist, date), None)


def _first_existing_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _format_float(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def enrich_understat_match_df(
    df: pd.DataFrame,
    elo_lookup: Dict[tuple[str, str], pd.DataFrame],
    *,
    source_path: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    for col in ELO_COLUMNS:
        if col in out.columns:
            out = out.drop(columns=[col])

    for col in ELO_COLUMNS:
        out[col] = ""

    date_col = _first_existing_col(out, ["game_date", "date"])
    required = {"league", "season", "team", "opp"}
    if date_col is None or not required <= set(out.columns):
        audit = pd.DataFrame(
            [
                {
                    "source_path": source_path,
                    "reason": "missing_required_columns",
                    "missing_columns": ",".join(sorted(required - set(out.columns))),
                }
            ]
        )
        return out, audit

    audit_rows: list[dict[str, Any]] = []
    for idx, row in out.iterrows():
        league = str(row.get("league", "")).strip()
        season = str(row.get("season", "")).strip()
        match_date = str(row.get(date_col, "")).split()[0]
        team = str(row.get("team", "")).strip().upper()
        opp = str(row.get("opp", "")).strip().upper()
        is_result = _is_result_row(row)

        team_pair = find_elo_pair(elo_lookup.get((league, team)), match_date, is_result=is_result)
        opp_pair = find_elo_pair(elo_lookup.get((league, opp)), match_date, is_result=is_result)

        values = {
            "team_start_elo": team_pair.start_elo,
            "team_end_elo": team_pair.end_elo,
            "opp_start_elo": opp_pair.start_elo,
            "opp_end_elo": opp_pair.end_elo,
            "elo_diff_start": (
                team_pair.start_elo - opp_pair.start_elo
                if team_pair.start_elo is not None and opp_pair.start_elo is not None
                else None
            ),
            "elo_diff_end": (
                team_pair.end_elo - opp_pair.end_elo
                if team_pair.end_elo is not None and opp_pair.end_elo is not None
                else None
            ),
        }
        for col, value in values.items():
            out.at[idx, col] = _format_float(value)

        for side, code, pair in (("team", team, team_pair), ("opp", opp, opp_pair)):
            if pair.reason:
                audit_rows.append(
                    {
                        "source_path": source_path,
                        "league": league,
                        "season": season,
                        "game_id": row.get("game_id", ""),
                        "game_date": match_date,
                        "team": team,
                        "opp": opp,
                        "side": side,
                        "side_code": code,
                        "is_result": str(bool(is_result)),
                        "reason": pair.reason,
                    }
                )

    audit = pd.DataFrame(audit_rows)
    return out, audit


def enrich_understat_files(
    seasons: list[SeasonInfo],
    elo_lookup: Dict[tuple[str, str], pd.DataFrame],
    *,
    overwrite: bool,
) -> tuple[int, pd.DataFrame]:
    files_written = 0
    audit_parts: list[pd.DataFrame] = []
    for info in seasons:
        for filename in MATCH_FILES:
            path = info.path / filename
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
            except Exception as exc:
                audit_parts.append(pd.DataFrame([{"source_path": str(path), "reason": f"read_failed:{exc}"}]))
                continue
            enriched, audit = enrich_understat_match_df(df, elo_lookup, source_path=str(path))
            if not audit.empty:
                audit_parts.append(audit)
            if overwrite:
                enriched.to_csv(path, index=False)
                files_written += 1

    audit_df = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    return files_written, audit_df


def write_audits(out_root: Path, *, missing_mappings: pd.DataFrame, missing_elo: pd.DataFrame, summary: Dict[str, Any]) -> None:
    audit_dir = out_root / "_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    missing_mappings.to_csv(audit_dir / "missing_team_mappings.csv", index=False)
    missing_elo.to_csv(audit_dir / "missing_understat_elo.csv", index=False)
    (audit_dir / "clubelo_understat_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_pipeline(
    *,
    raw_clubelo_dir: Path = DEFAULT_RAW_CLUBELO_DIR,
    understat_root: Path = DEFAULT_UNDERSTAT_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    teams_config_path: Path = DEFAULT_TEAMS_CONFIG,
    aliases_path: Path = DEFAULT_ALIASES,
    overwrite: bool = False,
    leagues: Optional[Iterable[str]] = None,
    seasons: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    resolver = load_team_resolver(teams_config_path, aliases_path)
    season_infos = discover_understat_seasons(understat_root, leagues=leagues, seasons=seasons)
    clubelo_leagues = sorted({info.clubelo_league for info in season_infos})

    history, missing_mappings = load_clean_clubelo_histories(
        raw_clubelo_dir,
        resolver,
        clubelo_leagues=clubelo_leagues,
    )
    processed_files = write_processed_clubelo_by_season(history, season_infos, out_root)
    elo_lookup = build_elo_lookup(history)
    enriched_files, missing_elo = enrich_understat_files(season_infos, elo_lookup, overwrite=overwrite)

    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "raw_clubelo_dir": str(raw_clubelo_dir),
        "understat_root": str(understat_root),
        "out_root": str(out_root),
        "overwrite": bool(overwrite),
        "seasons_discovered": len(season_infos),
        "clubelo_rows_cleaned": int(len(history)),
        "clubelo_processed_files_written": int(processed_files),
        "understat_match_files_written": int(enriched_files),
        "missing_team_mapping_groups": int(len(missing_mappings)),
        "missing_understat_elo_rows": int(len(missing_elo)),
    }
    write_audits(out_root, missing_mappings=missing_mappings, missing_elo=missing_elo, summary=summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Clean ClubElo histories and enrich processed Understat match files")
    parser.add_argument("--raw-clubelo-dir", type=Path, default=DEFAULT_RAW_CLUBELO_DIR)
    parser.add_argument("--understat-root", type=Path, default=DEFAULT_UNDERSTAT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--teams-config", type=Path, default=DEFAULT_TEAMS_CONFIG)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite processed Understat match CSVs in place.")
    parser.add_argument("--league", action="append", default=None, help="Optional processed league name filter. Can repeat.")
    parser.add_argument("--season", action="append", default=None, help="Optional season filter like 2025-2026. Can repeat.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    init_logger(args.verbose)
    summary = run_pipeline(
        raw_clubelo_dir=args.raw_clubelo_dir,
        understat_root=args.understat_root,
        out_root=args.out_root,
        teams_config_path=args.teams_config,
        aliases_path=args.aliases,
        overwrite=args.overwrite,
        leagues=args.league,
        seasons=args.season,
    )
    LOG.info("ClubElo Understat enrichment complete: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
