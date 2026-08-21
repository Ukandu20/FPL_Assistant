from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    CAMEO_FEATURES, CAMEO_MIN_FEATURES, P60_FEATURES, START_FEATURES,
    START_MIN_FEATURES, assert_canonical_features,
)


HEAD_FEATURES = {
    "start": START_FEATURES,
    "start_minutes": START_MIN_FEATURES,
    "cameo": CAMEO_FEATURES,
    "cameo_minutes": CAMEO_MIN_FEATURES,
    "p60": P60_FEATURES,
}
CLASSIFICATION_HEADS = {"start", "cameo", "p60"}


def model_family(position: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(position, dtype=str)
    return np.where(values == "GK", "GK", "OUTFIELD")


@dataclass
class ConstantModel:
    value: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


@dataclass
class HeadModel:
    head: str
    family: str
    features: list[str]
    estimator: Any
    sample_count: int
    positive_rate: float | None = None
    fallback_reason: str | None = None

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        frame = rows[self.features]
        if self.head in CLASSIFICATION_HEADS and hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(frame)[:, 1], dtype=float)
        return np.asarray(self.estimator.predict(frame), dtype=float)


@dataclass
class ModelBundle:
    heads: dict[str, dict[str, HeadModel]] = field(default_factory=dict)
    hyperparameters: dict[str, object] = field(default_factory=dict)
    seed: int = 0

    def predict_raw(self, head: str, rows: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
        result = np.zeros(len(rows), dtype=float)
        flags: list[list[str]] = [[] for _ in range(len(rows))]
        families = model_family(rows["pos"])
        for family in ("GK", "OUTFIELD"):
            mask = families == family
            if not mask.any():
                continue
            model = self.heads[head][family]
            result[mask] = model.predict(rows.loc[mask])
            if model.fallback_reason:
                for index in np.flatnonzero(mask):
                    flags[index].append(f"model_fallback:{head}:{family}:{model.fallback_reason}")
        return result, flags


def _lightgbm_estimator(head: str, features: list[str], seed: int, params: dict[str, object]) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("Training V2 requires the modeling extra: pip install -e .[modeling]") from exc
    common = {
        "n_estimators": 250, "learning_rate": 0.035, "num_leaves": 24,
        "min_child_samples": 30, "subsample": 1.0, "colsample_bytree": 1.0,
        "random_state": seed, "n_jobs": 1, "verbosity": -1,
    }
    common.update(params)
    if head in CLASSIFICATION_HEADS:
        return lgb.LGBMClassifier(objective="binary", **common)
    return lgb.LGBMRegressor(objective="regression_l2", **common)


def _population(rows: pd.DataFrame, head: str) -> tuple[pd.DataFrame, pd.Series]:
    eligible = rows[rows["eligible_for_fixture"]].copy()
    if head == "start":
        return eligible, eligible["is_starter"].astype(int)
    if head == "start_minutes":
        sample = eligible[eligible["is_starter"].eq(1)].copy()
        return sample, sample["minutes"].astype(float)
    if head == "cameo":
        sample = eligible[eligible["is_starter"].eq(0)].copy()
        return sample, sample["minutes"].gt(0).astype(int)
    if head == "cameo_minutes":
        sample = eligible[eligible["is_starter"].eq(0) & eligible["minutes"].gt(0)].copy()
        return sample, sample["minutes"].astype(float)
    if head == "p60":
        return eligible, eligible["minutes"].ge(60).astype(int)
    raise ValueError(f"Unknown head: {head}")


def train_bundle(
    rows: pd.DataFrame,
    seed: int,
    hyperparameters: dict[str, dict[str, object]] | None = None,
    feature_overrides: dict[str, list[str]] | None = None,
) -> ModelBundle:
    parameters = hyperparameters or {}
    bundle = ModelBundle(hyperparameters=parameters, seed=seed)
    families = model_family(rows["pos"])
    overrides = feature_overrides or {}
    for head, base_features in HEAD_FEATURES.items():
        features = list(dict.fromkeys(base_features + overrides.get(head, [])))
        if not overrides:
            assert_canonical_features(features)
        missing_features = set(features) - set(rows.columns)
        if missing_features:
            raise ValueError(f"Features missing for {head}: {sorted(missing_features)}")
        bundle.heads[head] = {}
        population, target = _population(rows, head)
        population_families = model_family(population["pos"])
        for family in ("GK", "OUTFIELD"):
            mask = population_families == family
            sample = population.loc[mask].copy()
            y = target.loc[sample.index]
            if sample.empty:
                if target.empty:
                    raise ValueError(f"No training population for {head} in any family")
                estimator = ConstantModel(float(target.mean()))
                bundle.heads[head][family] = HeadModel(
                    head=head, family=family, features=list(features), estimator=estimator,
                    sample_count=0,
                    positive_rate=float(target.mean()) if head in CLASSIFICATION_HEADS else None,
                    fallback_reason="no_family_events_global_constant",
                )
                continue
            fallback = None
            if head in CLASSIFICATION_HEADS and y.nunique() < 2:
                estimator: Any = ConstantModel(float(y.mean()))
                fallback = "degenerate_target"
            elif head not in CLASSIFICATION_HEADS and len(sample) < 30:
                estimator = ConstantModel(float(y.mean()))
                fallback = "sparse_duration_population"
            else:
                estimator = _lightgbm_estimator(head, features, seed, parameters.get(head, {}))
                fit_kwargs = {"categorical_feature": ["pos"]}
                estimator.fit(sample[features], y, **fit_kwargs)
            bundle.heads[head][family] = HeadModel(
                head=head, family=family, features=list(features), estimator=estimator,
                sample_count=len(sample),
                positive_rate=float(y.mean()) if head in CLASSIFICATION_HEADS else None,
                fallback_reason=fallback,
            )
    return bundle
