from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MinutesV2Config


EPS = 1e-9


@dataclass
class ProbabilityCalibrator:
    method: str = "raw"
    coef: float = 1.0
    intercept: float = 0.0
    x_thresholds: np.ndarray | None = None
    y_thresholds: np.ndarray | None = None

    def transform(self, raw: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw, dtype=float), EPS, 1.0 - EPS)
        if self.method == "raw":
            return p
        if self.method == "platt":
            logit = np.log(p / (1.0 - p))
            z = np.clip(self.intercept + self.coef * logit, -35, 35)
            return 1.0 / (1.0 + np.exp(-z))
        if self.method == "isotonic":
            if self.x_thresholds is None or self.y_thresholds is None:
                raise ValueError("Isotonic calibrator thresholds are missing")
            return np.interp(p, self.x_thresholds, self.y_thresholds)
        raise ValueError(f"Unknown calibration method: {self.method}")


def calibration_eligible(y: np.ndarray, config: MinutesV2Config) -> bool:
    target = np.asarray(y, dtype=int)
    return (
        len(target) >= config.calibration_min_rows
        and int(target.sum()) >= config.calibration_min_positive
        and int((1 - target).sum()) >= config.calibration_min_negative
    )


def fit_calibrator(method: str, raw: np.ndarray, y: np.ndarray) -> ProbabilityCalibrator:
    p = np.clip(np.asarray(raw, dtype=float), EPS, 1.0 - EPS)
    target = np.asarray(y, dtype=float)
    if method == "raw":
        return ProbabilityCalibrator()
    if len(np.unique(target)) < 2:
        return ProbabilityCalibrator()
    if method == "platt":
        x = np.log(p / (1.0 - p))
        design = np.column_stack([np.ones(len(x)), x])
        beta = np.zeros(2)
        for _ in range(100):
            fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -35, 35)))
            weights = np.clip(fitted * (1.0 - fitted), 1e-6, None)
            hessian = design.T @ (weights[:, None] * design) + np.eye(2) * 1e-6
            gradient = design.T @ (target - fitted)
            step = np.linalg.solve(hessian, gradient)
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break
        return ProbabilityCalibrator(method="platt", intercept=float(beta[0]), coef=float(beta[1]))
    if method == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            raise RuntimeError("Install fpl-assistant[modeling] for isotonic calibration") from exc
        fitted = IsotonicRegression(out_of_bounds="clip").fit(p, target)
        return ProbabilityCalibrator(
            method="isotonic",
            x_thresholds=np.asarray(fitted.X_thresholds_, dtype=float),
            y_thresholds=np.asarray(fitted.y_thresholds_, dtype=float),
        )
    raise ValueError(f"Unsupported calibrator: {method}")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


@dataclass(frozen=True)
class CalibrationFoldData:
    fold_id: str
    calibration_raw: np.ndarray
    calibration_y: np.ndarray
    evaluation_raw: np.ndarray
    evaluation_y: np.ndarray


@dataclass
class CalibrationSelection:
    method: str
    mean_brier_skill: float
    raw_mean_brier_skill: float
    fold_metrics: list[dict[str, float | str]] = field(default_factory=list)
    reason: str = ""


def select_calibration(folds: list[CalibrationFoldData], config: MinutesV2Config) -> CalibrationSelection:
    """Select raw/Platt/isotonic by deterministic chronological evidence."""
    candidates = ("raw", "platt", "isotonic")
    scored: dict[str, list[dict[str, float | str]]] = {method: [] for method in candidates}
    for fold in folds:
        prevalence = float(np.mean(fold.evaluation_y))
        prevalence_brier = max(prevalence * (1.0 - prevalence), EPS)
        for method in candidates:
            calibrator = fit_calibrator(method, fold.calibration_raw, fold.calibration_y)
            prediction = calibrator.transform(fold.evaluation_raw)
            score = brier(fold.evaluation_y, prediction)
            scored[method].append({
                "fold_id": fold.fold_id,
                "brier": score,
                "brier_skill": 1.0 - score / prevalence_brier,
            })
    raw_mean = float(np.mean([float(x["brier_skill"]) for x in scored["raw"]]))
    eligible: list[tuple[float, int, str]] = []
    for order, method in enumerate(candidates[1:], start=1):
        mean_skill = float(np.mean([float(x["brier_skill"]) for x in scored[method]]))
        deterioration = [
            float(candidate["brier"]) - float(raw["brier"])
            for candidate, raw in zip(scored[method], scored["raw"], strict=True)
        ]
        if mean_skill > raw_mean and max(deterioration, default=0.0) <= config.calibration_max_fold_brier_deterioration:
            # Lower order resolves exact ties: Platt before isotonic.
            eligible.append((mean_skill, -order, method))
    if not eligible:
        return CalibrationSelection("raw", raw_mean, raw_mean, scored["raw"], "no stable chronological improvement")
    mean_skill, _, method = max(eligible)
    return CalibrationSelection(method, mean_skill, raw_mean, scored[method], "stable mean Brier-skill improvement")


@dataclass
class HierarchicalCalibrators:
    method: str
    global_calibrator: ProbabilityCalibrator | None
    subgroup_calibrators: dict[str, ProbabilityCalibrator]

    def transform(self, raw: np.ndarray, families: np.ndarray) -> tuple[np.ndarray, list[list[str]]]:
        result = np.asarray(raw, dtype=float).copy()
        flags: list[list[str]] = [[] for _ in range(len(result))]
        for i, family in enumerate(np.asarray(families, dtype=str)):
            if family in self.subgroup_calibrators:
                result[i] = self.subgroup_calibrators[family].transform(np.array([result[i]]))[0]
            elif self.global_calibrator is not None:
                result[i] = self.global_calibrator.transform(np.array([result[i]]))[0]
                flags[i].append(f"calibration_global_fallback:{family}")
            else:
                flags[i].append(f"calibration_raw_fallback:{family}")
        return np.clip(result, 0.0, 1.0), flags


def fit_hierarchical_calibrators(
    method: str,
    raw: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    config: MinutesV2Config,
) -> HierarchicalCalibrators:
    if method == "raw":
        return HierarchicalCalibrators(method, None, {})
    target = np.asarray(y, dtype=int)
    global_cal = fit_calibrator(method, raw, target) if calibration_eligible(target, config) else None
    subgroup: dict[str, ProbabilityCalibrator] = {}
    family_values = np.asarray(families, dtype=str)
    for family in ("GK", "OUTFIELD"):
        mask = family_values == family
        if calibration_eligible(target[mask], config):
            subgroup[family] = fit_calibrator(method, np.asarray(raw)[mask], target[mask])
    return HierarchicalCalibrators(method, global_cal, subgroup)
