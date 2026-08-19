"""Shareable navigation and persistent user state for the FPL app."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent


def query_value(name: str, default: str | None = None) -> str | None:
    """Return one normalized query-parameter value."""
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        value = value[-1] if value else default
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or default


def query_list(name: str) -> list[str]:
    """Return a comma-delimited query parameter as unique ordered values."""
    raw = query_value(name, "") or ""
    return list(dict.fromkeys(value.strip() for value in raw.split(",") if value.strip()))


def update_query(**values: object) -> None:
    """Update URL state while retaining unrelated query parameters."""
    for key, value in values.items():
        if value is None or value == "":
            st.query_params.pop(key, None)
        elif isinstance(value, (list, tuple, set)):
            st.query_params[key] = ",".join(str(item) for item in value)
        else:
            st.query_params[key] = str(value)


def shortlist() -> list[str]:
    return list(st.session_state.get("fpl_shortlist", []))


def set_shortlist(player_ids: Iterable[object]) -> list[str]:
    values = list(dict.fromkeys(str(value) for value in player_ids if str(value)))
    st.session_state["fpl_shortlist"] = values
    return values


def toggle_shortlist(player_id: object) -> bool:
    """Toggle a player and return whether they are now shortlisted."""
    value = str(player_id)
    values = shortlist()
    if value in values:
        values.remove(value)
        selected = False
    else:
        values.append(value)
        selected = True
    set_shortlist(values)
    return selected


def recent_players() -> list[str]:
    return list(st.session_state.get("fpl_recent_players", []))


def record_recent_player(player_id: object, *, limit: int = 8) -> list[str]:
    value = str(player_id)
    values = [item for item in recent_players() if item != value]
    values.insert(0, value)
    values = values[:limit]
    st.session_state["fpl_recent_players"] = values
    return values


def comparison() -> list[str]:
    queried = query_list("players")
    values = queried or st.session_state.get("fpl_comparison", [])
    return list(dict.fromkeys(str(value) for value in values if str(value)))[:4]


def set_comparison(player_ids: Iterable[object]) -> list[str]:
    values = list(dict.fromkeys(str(value) for value in player_ids if str(value)))[:4]
    st.session_state["fpl_comparison"] = values
    update_query(players=values)
    return values


def switch_page(filename: str, **query: object) -> None:
    """Navigate within the registered FPL application while preserving context."""
    update_query(**query)
    st.switch_page(APP_ROOT / "pages" / filename)
