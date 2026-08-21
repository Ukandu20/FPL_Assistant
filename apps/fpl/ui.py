"""Shared visual language and reusable Streamlit components for FPL."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


COLORS = {
    "purple": "#5B21B6",
    "purple_dark": "#37003C",
    "mint": "#008A5A",
    "magenta": "#C2185B",
    "amber": "#B45309",
    "risk": "#C2414A",
    "blue": "#2563EB",
    "slate": "#475569",
    "surface": "#F8FAFC",
}

POSITION_COLORS = {
    "GKP": "#2563EB",
    "GK": "#2563EB",
    "DEF": "#008A5A",
    "MID": "#7C3AED",
    "FWD": "#C2185B",
}

ARCHETYPE_FAMILY_BADGE_COLORS = {
    "Production Composite": "violet",
    "Production Style": "blue",
    "Production": "blue",
    "Usage": "green",
    "Return Shape": "orange",
    "Risk Badge": "red",
    "Value Historical": "gray",
    "Venue Behaviour": "gray",
}

AVAILABILITY_STATUS_STYLES = {
    "a": ("Available", "#16A34A", "#FFFFFF", "#DCFCE7", "#166534"),
    "d": ("Doubtful", "#EAB308", "#422006", "#FEF9C3", "#854D0E"),
    "i": ("Injured", "#DC2626", "#FFFFFF", "#FEE2E2", "#991B1B"),
    "s": ("Suspended", "#DC2626", "#FFFFFF", "#FEE2E2", "#991B1B"),
    "u": ("Unavailable", "#DC2626", "#FFFFFF", "#FEE2E2", "#991B1B"),
    "n": ("Unavailable", "#DC2626", "#FFFFFF", "#FEE2E2", "#991B1B"),
}

AVAILABILITY_STATUS_ICONS = {
    "a": "🟢",
    "d": "🟡",
    "i": "🔴",
    "s": "🔴",
    "u": "🔴",
    "n": "🔴",
}


def availability_status(value: object) -> tuple[str, str, str]:
    """Return an accessible label plus badge background and foreground colors."""
    code = str(value).strip().lower()
    label, background, foreground, _, _ = AVAILABILITY_STATUS_STYLES.get(
        code, ("Status unknown", "#64748B", "#FFFFFF", "#E2E8F0", "#334155")
    )
    return label, background, foreground


def availability_cell_style(value: object) -> str:
    """Style a status-table cell using both text and semantic color."""
    normalized = str(value).strip().lower()
    plain_label = normalized.lstrip("🟢🟡🔴⚪ ").strip()
    for code, style in AVAILABILITY_STATUS_STYLES.items():
        if normalized == code or plain_label == style[0].lower():
            _, _, _, background, foreground = style
            return f"background-color:{background};color:{foreground};font-weight:600"
    return "background-color:#E2E8F0;color:#334155;font-weight:600"


def style_availability_table(frame: pd.DataFrame) -> pd.DataFrame | pd.io.formats.style.Styler:
    """Render FPL status columns as labelled, color-coded table cells."""
    output = frame.copy()
    status_columns = [column for column in ("Status", "status") if column in output]
    if not status_columns:
        return output

    label_to_code = {
        style[0].lower(): code for code, style in AVAILABILITY_STATUS_STYLES.items()
    }

    def table_label(value: object) -> str:
        normalized = str(value).strip().lower()
        code = normalized if normalized in AVAILABILITY_STATUS_STYLES else label_to_code.get(normalized)
        if code is None:
            return "⚪ Status unknown"
        label = AVAILABILITY_STATUS_STYLES[code][0]
        return f"{AVAILABILITY_STATUS_ICONS[code]} {label}"

    for column in status_columns:
        output[column] = output[column].map(table_label)
    return output.style.map(availability_cell_style, subset=status_columns)


def inject_global_styles() -> None:
    """Apply small responsive refinements that Streamlit's theme cannot express."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1480px;}
        [data-testid="stMetric"] {background: var(--secondary-background-color);
            border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
            border-radius: .75rem; padding: .8rem 1rem; min-height: 6rem;}
        [data-testid="stMetricLabel"] {color: var(--text-color); opacity: .72;}
        [data-testid="stSidebar"] {border-right: 1px solid
            color-mix(in srgb, var(--text-color) 18%, transparent);}
        .fpl-eyebrow {font-size: .75rem; font-weight: 700; letter-spacing: .08em;
            text-transform: uppercase; color: var(--primary-color); margin-bottom: .2rem;}
        .fpl-page-subtitle {color: var(--text-color); opacity: .72;
            margin-top: -.35rem; margin-bottom: 1rem;}
        .fpl-freshness {display: inline-block; color: var(--text-color); opacity: .72;
            font-size: .8rem; border: 1px solid
            color-mix(in srgb, var(--text-color) 28%, transparent);
            border-radius: 999px; padding: .15rem .55rem;}
        @media (max-width: 800px) {
            .block-container {padding-left: 1rem; padding-right: 1rem;}
            [data-testid="stMetric"] {min-height: 5rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    subtitle: str,
    *,
    eyebrow: str | None = None,
    freshness: str | None = None,
) -> None:
    if eyebrow:
        st.markdown(f'<div class="fpl-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(
        f'<div class="fpl-page-subtitle">{escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )
    if freshness:
        st.markdown(
            f'<span class="fpl-freshness">Updated {escape(freshness)}</span>',
            unsafe_allow_html=True,
        )


def empty_state(title: str, message: str, *, icon: str = "ℹ️") -> None:
    with st.container(border=True):
        st.markdown(f"### {icon} {title}")
        st.caption(message)


def file_freshness(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "unavailable"
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    age = now - stamp
    if age.days == 0:
        hours = max(0, int(age.total_seconds() // 3600))
        return "just now" if hours == 0 else f"{hours}h ago"
    if age.days < 7:
        return f"{age.days}d ago"
    return stamp.strftime("%d %b %Y")


def format_number(value: object, digits: int = 1, fallback: str = "—") -> str:
    number = pd.to_numeric(value, errors="coerce")
    return fallback if pd.isna(number) else f"{float(number):.{digits}f}"


def badge_markdown(rows: pd.DataFrame) -> str:
    badges: list[str] = []
    for row in rows.to_dict("records"):
        label = str(row.get("display_name", "")).strip()
        if not label:
            continue
        label = label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        confidence = str(row.get("confidence_band", "")).strip().lower()
        color = "gray" if confidence == "low" else ARCHETYPE_FAMILY_BADGE_COLORS.get(
            str(row.get("family", "")), "gray"
        )
        badges.append(f":{color}-badge[{label}]")
    return " ".join(badges)


def render_badges(rows: pd.DataFrame) -> None:
    markup = badge_markdown(rows)
    if markup:
        st.markdown(markup)


def apply_chart_style(figure: go.Figure, *, height: int | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        font=dict(family="Arial, sans-serif", color="#172033"),
        title_font=dict(size=17, color="#172033"),
        colorway=[COLORS["purple"], COLORS["mint"], COLORS["magenta"], COLORS["blue"]],
        margin=dict(t=50, b=25, l=25, r=20),
        hoverlabel=dict(bgcolor="white", font_color="#172033"),
    )
    if height is not None:
        figure.update_layout(height=height)
    return figure
