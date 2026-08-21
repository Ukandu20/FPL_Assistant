from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .config import MinutesV2Config


@dataclass(frozen=True)
class ChronologicalFold:
    fold_id: str
    train_keys: tuple[str, ...]
    calibration_keys: tuple[str, ...]
    evaluation_keys: tuple[str, ...]
    is_final_holdout: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fixture_block_key(frame: pd.DataFrame) -> pd.Series:
    gw = pd.to_numeric(frame.get("gw_played", frame.get("gw_orig")), errors="coerce")
    return frame["season"].astype(str) + ":GW" + gw.astype("Int64").astype(str)


def generate_folds(rows: pd.DataFrame, config: MinutesV2Config) -> list[ChronologicalFold]:
    df = rows.copy()
    df["_key"] = fixture_block_key(df)
    ordered = (
        df[["_key", "date_sched"]].dropna().drop_duplicates()
        .groupby("_key", as_index=False)["date_sched"].min()
        .sort_values(["date_sched", "_key"], kind="mergesort")
    )
    keys = ordered["_key"].tolist()
    span = config.calibration_gws + config.evaluation_gws
    if len(keys) < span + config.final_holdout_gws + 1:
        raise ValueError("Insufficient chronological GW blocks for locked walk-forward protocol")
    final_keys = tuple(keys[-config.final_holdout_gws:])
    selectable = keys[:-config.final_holdout_gws]
    folds: list[ChronologicalFold] = []
    index = 1
    first_season = selectable[0].split(":GW", 1)[0]
    for season in dict.fromkeys(key.split(":GW", 1)[0] for key in selectable):
        # A selectable fold requires at least one complete prior season. This
        # implements the locked "completed prior seasons + current history"
        # minimum rather than selecting on a partial inaugural season.
        if season == first_season:
            continue
        season_keys = [key for key in selectable if key.startswith(f"{season}:GW")]
        eval_end = span
        while eval_end <= len(season_keys):
            cal_start = eval_end - span
            eval_start = eval_end - config.evaluation_gws
            calibration = tuple(season_keys[cal_start:eval_start])
            evaluation = tuple(season_keys[eval_start:eval_end])
            first_cal_index = keys.index(calibration[0])
            folds.append(ChronologicalFold(
                fold_id=f"wf_{index:02d}",
                train_keys=tuple(keys[:first_cal_index]),
                calibration_keys=calibration,
                evaluation_keys=evaluation,
            ))
            index += 1
            eval_end += config.step_gws
    holdout_cal_start = len(keys) - config.final_holdout_gws - config.calibration_gws
    folds.append(ChronologicalFold(
        fold_id="final_holdout",
        train_keys=tuple(keys[:holdout_cal_start]),
        calibration_keys=tuple(keys[holdout_cal_start:-config.final_holdout_gws]),
        evaluation_keys=final_keys,
        is_final_holdout=True,
    ))
    return folds


def assert_holdout_isolated(folds: list[ChronologicalFold]) -> None:
    holdouts = [fold for fold in folds if fold.is_final_holdout]
    if len(holdouts) != 1:
        raise ValueError("Exactly one untouched final holdout is required")
    protected = set(holdouts[0].evaluation_keys)
    for fold in folds:
        if fold.is_final_holdout:
            continue
        if protected & (set(fold.train_keys) | set(fold.calibration_keys) | set(fold.evaluation_keys)):
            raise ValueError("Final holdout leaked into a selection fold")
