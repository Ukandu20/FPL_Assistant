"""Leakage-safe Expected-Minutes Model V2.0.

V1 remains in :mod:`fpl_assistant.models.minutes_model_builder`; this package
is intentionally separate so the benchmark and rollback path cannot drift.
"""

from .config import MinutesV2Config, load_config
from .schema import MODEL_VERSION, REQUIRED_OUTPUT_COLUMNS, validate_predictions

__all__ = [
    "MODEL_VERSION",
    "MinutesV2Config",
    "REQUIRED_OUTPUT_COLUMNS",
    "load_config",
    "validate_predictions",
]
