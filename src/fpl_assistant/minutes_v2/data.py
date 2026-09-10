from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import MinutesV2Config


TRUE_VALUES = {"1", "true", "yes", "y"}
OBSERVED_STATUSES = {"observed", "played", "did_not_play", "not_in_matchday_squad"}


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.lower().isin(TRUE_VALUES)


def as_nullable_bool(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.lower().str.strip()
    mapped = text.map({"1": True, "true": True, "yes": True, "y": True,
                       "0": False, "false": False, "no": False, "n": False})
    return mapped.astype("boolean")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class DatasetAudit:
    source_files: dict[str, dict[str, str | int]]
    season_rows_before_filter: dict[str, int]
    season_rows_after_filter: dict[str, int]
    starter_source_distribution: dict[str, dict[str, int]]
    eligibility_timestamp_safe_distribution: dict[str, dict[str, int]]
    excluded_seasons: dict[str, str]
    excluded_label_rows: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _normalize(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    df = raw.copy()
    df["season"] = season
    for column in ("date_sched", "date_played", "information_timestamp"):
        if column in df:
            df[column] = pd.to_datetime(
                df[column], format="mixed", utc=True, errors="coerce"
            )
    for column in ("minutes", "is_starter", "gw_orig", "gw_played"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    required_evidence = {"eligible_for_fixture", "confirmed_unavailable", "eligibility_timestamp_safe", "information_timestamp"}
    missing_evidence = required_evidence - set(df.columns)
    if missing_evidence:
        raise ValueError(f"Registry season {season} lacks required eligibility evidence: {sorted(missing_evidence)}")
    df["eligible_for_fixture"] = as_nullable_bool(df["eligible_for_fixture"])
    df["confirmed_unavailable"] = as_nullable_bool(df["confirmed_unavailable"])
    df["eligibility_timestamp_safe"] = as_nullable_bool(df["eligibility_timestamp_safe"])
    for column in ("starter_source", "eligibility_source", "observation_status"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    return df


def load_registry(
    config: MinutesV2Config,
    seasons: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, DatasetAudit]:
    chosen = tuple(seasons or (*config.canonical_label_seasons, config.forward_season))
    frames: list[pd.DataFrame] = []
    files: dict[str, dict[str, str | int]] = {}
    before: dict[str, int] = {}
    starter_dist: dict[str, dict[str, int]] = {}
    safe_dist: dict[str, dict[str, int]] = {}
    for season in chosen:
        path = config.registry_root / season / "player_fixture_calendar.csv"
        if not path.exists():
            raise FileNotFoundError(f"Configured season data is missing: {path}")
        raw = pd.read_csv(path, low_memory=False)
        df = _normalize(raw, season)
        before[season] = len(df)
        starter_dist[season] = {str(k): int(v) for k, v in df["starter_source"].value_counts(dropna=False).items()}
        safe_dist[season] = {str(k): int(v) for k, v in df["eligibility_timestamp_safe"].value_counts(dropna=False).items()}
        files[season] = {"path": str(path.resolve()), "sha256": file_sha256(path), "rows": len(df)}
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    audit = DatasetAudit(
        source_files=files,
        season_rows_before_filter=before,
        season_rows_after_filter={},
        starter_source_distribution=starter_dist,
        eligibility_timestamp_safe_distribution=safe_dist,
        excluded_seasons={s: "diagnostic_only: starter/substitute labels not audit-verified" for s in config.diagnostic_only_seasons},
        excluded_label_rows={},
    )
    return all_rows, audit


def canonical_training_rows(
    rows: pd.DataFrame,
    config: MinutesV2Config,
    prediction_cutoff: str | pd.Timestamp,
    audit: DatasetAudit | None = None,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(prediction_cutoff, utc=True)
    df = rows[rows["season"].isin(config.canonical_label_seasons)].copy()
    # No outcome at or after a forecast cutoff may influence any head.
    before_cutoff = df["date_sched"].notna() & (df["date_sched"] < cutoff)
    observed = df["minutes"].notna() & df["is_starter"].notna()
    eligible = df["eligible_for_fixture"].eq(True).fillna(False)
    # A confirmed-unavailable row is excluded only when its information existed
    # by the cutoff. Unknown/missing availability is not converted to false/zero.
    known_unavailable = (
        df["confirmed_unavailable"].eq(True).fillna(False)
        & df["information_timestamp"].notna()
        & (df["information_timestamp"] <= cutoff)
    )
    trusted = df["starter_source"].isin(config.trusted_starter_sources)
    selected = df[before_cutoff & observed & eligible & ~known_unavailable & trusted].copy()
    if audit is not None:
        for season in config.canonical_label_seasons:
            season_all = df[df["season"] == season]
            season_selected = selected[selected["season"] == season]
            audit.season_rows_after_filter[season] = len(season_selected)
            audit.excluded_label_rows[season] = len(season_all) - len(season_selected)
    if selected.empty:
        raise ValueError("No canonical fixture-eligible, reliable, pre-cutoff training rows")
    return selected


def inference_rows(
    rows: pd.DataFrame,
    season: str,
    prediction_cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(prediction_cutoff, utc=True)
    df = rows[rows["season"].eq(season)].copy()
    pending = df["date_sched"].notna() & (df["date_sched"] >= cutoff) & df["eligible_for_fixture"].eq(True).fillna(False)
    timestamp_safe = df["eligibility_timestamp_safe"].eq(True).fillna(False) & df["information_timestamp"].notna()
    known_by_cutoff = df["information_timestamp"] <= cutoff
    result = df[pending & timestamp_safe & known_by_cutoff].copy()
    if result.empty:
        raise ValueError("No timestamp-safe eligible inference roster rows known by prediction cutoff")
    return result


def current_season_history_rows(
    rows: pd.DataFrame,
    season: str,
    prediction_cutoff: str | pd.Timestamp,
    config: MinutesV2Config,
) -> pd.DataFrame:
    """Return current-season outcomes that were published before the cutoff."""
    cutoff = pd.to_datetime(prediction_cutoff, utc=True)
    df = rows[rows["season"].eq(season)].copy()
    observed_before_cutoff = (
        df["date_sched"].notna()
        & df["date_sched"].lt(cutoff)
        & df["minutes"].notna()
        & df["is_starter"].notna()
        & df["information_timestamp"].notna()
        & df["information_timestamp"].le(cutoff)
        & df["starter_source"].isin(config.trusted_starter_sources)
    )
    return df.loc[observed_before_cutoff].copy()
