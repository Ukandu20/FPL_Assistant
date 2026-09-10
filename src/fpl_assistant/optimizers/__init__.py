"""Lazy module exports keep optional workflows independent at import time."""
from importlib import import_module

__all__ = ['multi_gw', 'multi_gw_hold', 'single_gw', 'strategy', 'team_state']


def __getattr__(name: str):
    if name == "team_state":
        return import_module("fpl_assistant.domain.team_state")
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
