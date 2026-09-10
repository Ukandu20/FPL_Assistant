"""Lazy module exports keep optional workflows independent at import time."""
from importlib import import_module

__all__ = ['defense_forecast', 'goals_assists_forecast', 'minutes_forecast', 'points_forecast', 'saves_forecast']


def __getattr__(name: str):
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
