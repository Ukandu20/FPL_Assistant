# predict_minutes.py
#!/usr/bin/env python3
"""
predict_minutes.py – train and evaluate continuous expected-minutes model for FPL players

Usage:
    python predict_minutes.py \
        --seasons 2020-2021,2021-2022,2022-2023,2023-2024,2024-2025 \
        --first-test-gw 26 \
        --fpl-root data/processed/fpl \
        --model-out models/expected_minutes
"""
import argparse
import json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error


def load_player_minutes(seasons, fix_root: Path) -> pd.DataFrame:
    """Load merged_gws.csv for each season and extract minutes per player per GW."""
    frames = []
    for season in seasons:
        path = fix_root / season / "player_minutes_calendar.csv"
        df = pd.read_csv(path, parse_dates=["date_played"])
        df = df.rename(columns={
            "player_id": "player_id",
            "player": "player",
            "gw_orig":   "gw_orig",
            "min": "minutes"
        })[["player_id", "player", "gw_orig", "minutes"]]
        df["season"] = season
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["minutes"] = data["minutes"].fillna(0)
    return data


def expand_grid(df: pd.DataFrame, seasons):
    """Ensure every (player_id, season, gw_orig in 1..38) exists; fill missing minutes=0."""
    players = df["player_id"].unique()
    all_gws = np.arange(1, 39)
    idx = pd.MultiIndex.from_product(
        [players, seasons, all_gws],
        names=["player_id", "season", "gw_orig"]
    )
    full = pd.DataFrame(index=idx).reset_index()
    df_full = full.merge(df, on=["player_id", "season", "gw_orig"], how="left")
    df_full["minutes"] = df_full["minutes"].fillna(0)
    return df_full


def compute_rolling_minutes(df: pd.DataFrame, window=5, decay=(0.8, 0.6, 0.4, 0.2)):
    """Compute decayed rolling mean of minutes for each player-season."""
    df = df.sort_values(["player_id", "season", "gw_orig"]).copy()
    weights = np.concatenate((np.array(decay, dtype = float), [0,0]))
    df["min_roll5"] = 0.0

    for (pid, season), group in df.groupby(["player_id", "season"]):
        vals = group["minutes"].to_numpy()
        roll = np.zeros_like(vals)
        for i in range(len(vals)):
            if i == 0:
                roll[i] = 0.0
            else:
                lo = max(0, i - window)
                window_vals = vals[lo:i]
                k = len(window_vals)
                w = weights[-k:]  # align most recent with highest weight
                roll[i] = np.dot(window_vals, w) / w.sum()
        df.loc[group.index, "min_roll5"] = roll
    return df


def train_and_evaluate(df: pd.DataFrame, first_test_gw: int, first_test_season: str):
    """Train LightGBM on training slice and evaluate on test slice."""
    # Split
    train = df.query(
        "(season < @first_test_season) or "
        "(season == @first_test_season and gw_orig < @first_test_gw)"
    )
    test = df.query(
        "season == @first_test_season and gw_orig >= @first_test_gw"
    )

    X_train = train[["min_roll5"]]
    y_train = train["minutes"]
    X_test = test[["min_roll5"]]
    y_test = test["minutes"]

    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        early_stopping_rounds=50,
        verbose=50
    )

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 90)
    mae = mean_absolute_error(y_test, preds)
    print(f"Test MAE: {mae:.3f} minutes")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        default="2020-2021,2021-2022,2022-2023,2023-2024,2024-2025"
    )
    parser.add_argument("--first-test-gw", type=int, default=26)
    parser.add_argument("--fix-root", default="data/processed/fixtures")
    parser.add_argument("--model-out", default="data/models/expected_minutes")
    parser.add_argument("--log-level",default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s: %(message)s")

    seasons = args.seasons.split(",")
    first_test_season = seasons[-1]

    data = load_player_minutes(seasons, Path(args.fix_root))
    data = expand_grid(data, seasons)
    data = compute_rolling_minutes(data)

    model = train_and_evaluate(data, args.first_test_gw, first_test_season)

    out_dir = Path(args.model_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(out_dir / "expected_minutes.txt")


if __name__ == "__main__":
    main()


