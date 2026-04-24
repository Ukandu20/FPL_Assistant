from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="ClubElo Manager Eras", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
CLUBELO_ROOT = PROJECT_ROOT / "data" / "raw" / "clubelo" / "team_history"
MANIFEST_PATH = PROJECT_ROOT / "data" / "config" / "transfermarkt_premier_league_clubs.json"
DEFAULT_MAN_UNITED_START = pd.Timestamp("2013-07-01")
TOP6_CODES = ("MUN", "MCI", "ARS", "TOT", "CHE", "LIV")
LEAGUE_AVG_COLOR = "#666666"
TOP6_AVG_COLOR = "#111111"
STOCK_UP_COLOR = "#1f9d55"
STOCK_DOWN_COLOR = "#d64545"
STOCK_FLAT_COLOR = "#7a7a7a"


@st.cache_data(show_spinner=False)
def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else []


@st.cache_data(show_spinner=False)
def discover_clubs() -> list[dict[str, str]]:
    manifest = load_manifest()
    clubs: list[dict[str, str]] = []
    for item in manifest:
        stem = str(item.get("clubelo_stem", "")).strip()
        team_code = str(item.get("team_code", "")).strip().upper()
        club_name = str(item.get("club_name", "")).strip()
        if not stem or not team_code or not club_name:
            continue
        csv_path = CLUBELO_ROOT / f"{stem}.csv"
        if not csv_path.exists():
            continue
        try:
            cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        except Exception:
            continue
        if "manager" not in cols:
            continue
        clubs.append(
            {
                "label": f"{club_name} ({team_code})",
                "team_code": team_code,
                "club_name": club_name,
                "clubelo_stem": stem,
            }
        )
    return clubs


@st.cache_data(show_spinner=False)
def load_clubelo_history(stem: str) -> pd.DataFrame:
    path = CLUBELO_ROOT / f"{stem}.csv"
    df = pd.read_csv(path, dtype={"season": str}, low_memory=False)
    df["from"] = pd.to_datetime(df["from"], errors="coerce")
    df["elo"] = pd.to_numeric(df["elo"], errors="coerce")
    df["manager"] = df.get("manager", pd.Series(index=df.index, dtype=object)).fillna("Unknown")
    df["manager"] = df["manager"].replace("", "Unknown")
    df = df.dropna(subset=["from", "elo"]).sort_values("from").reset_index(drop=True)
    df["manager_change"] = df["manager"].ne(df["manager"].shift())
    df["manager_stint_id"] = df["manager_change"].cumsum()

    stint_lookup = df[["manager", "manager_stint_id"]].drop_duplicates().copy()
    stint_lookup["manager_stint_number"] = stint_lookup.groupby("manager").cumcount() + 1
    stint_lookup["manager_stint_count"] = (
        stint_lookup.groupby("manager")["manager_stint_id"].transform("size")
    )
    stint_lookup["manager_stint_label"] = np.where(
        stint_lookup["manager_stint_count"] > 1,
        stint_lookup["manager"]
        + " (Stint "
        + stint_lookup["manager_stint_number"].astype(str)
        + ")",
        stint_lookup["manager"],
    )
    df = df.merge(
        stint_lookup[["manager_stint_id", "manager_stint_label"]],
        on="manager_stint_id",
        how="left",
    )
    return df


@st.cache_data(show_spinner=False)
def load_benchmark_histories() -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for club in discover_clubs():
        club_df = load_clubelo_history(club["clubelo_stem"])
        histories[club["team_code"]] = club_df[["from", "elo"]].copy()
    return histories


def filter_history(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    filtered = df.loc[df["from"] >= start_date].copy()
    if end_date is not None:
        filtered = filtered.loc[filtered["from"] <= end_date].copy()
    return filtered


def get_stint_selector_options(df: pd.DataFrame) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    stint_meta = (
        df[["manager_stint_id", "manager_stint_label", "manager"]]
        .drop_duplicates()
        .sort_values("manager_stint_id")
    )
    for _, meta_row in stint_meta.iterrows():
        stint_id = int(meta_row["manager_stint_id"])
        stint_df = df.loc[df["manager_stint_id"] == stint_id].sort_values("from")
        if stint_df.empty:
            continue
        start_date = pd.Timestamp(stint_df["from"].iloc[0])
        end_date = pd.Timestamp(stint_df["from"].iloc[-1])
        options.append(
            {
                "manager_stint_id": stint_id,
                "manager": str(meta_row["manager"]),
                "manager_stint_label": str(meta_row["manager_stint_label"]),
                "start_date": start_date,
                "end_date": end_date,
                "label": (
                    f"{meta_row['manager_stint_label']}: "
                    f"{start_date.date().isoformat()} to {end_date.date().isoformat()}"
                ),
            }
        )
    return options


def compute_group_average(
    benchmark_histories: dict[str, pd.DataFrame],
    team_codes: list[str] | tuple[str, ...],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float | None:
    club_means: list[float] = []
    for team_code in team_codes:
        club_df = benchmark_histories.get(team_code)
        if club_df is None or club_df.empty:
            continue
        window = club_df.loc[
            (club_df["from"] >= start_date) & (club_df["from"] <= end_date),
            "elo",
        ]
        if window.empty:
            continue
        club_means.append(float(window.mean()))
    if not club_means:
        return None
    return float(np.mean(club_means))


def get_manager_colors(manager_order: list[str]) -> dict[str, str]:
    color_sequence = (
        st.session_state.get("_clubelo_manager_palette")
        or (
            [
                "#88CCEE",
                "#CC6677",
                "#44AA99",
                "#117733",
                "#332288",
                "#DDCC77",
                "#AA4499",
                "#882255",
                "#661100",
                "#999933",
                "#6699CC",
                "#888888",
            ]
        )
    )
    return {
        manager: color_sequence[idx % len(color_sequence)]
        for idx, manager in enumerate(manager_order)
    }


def get_stint_extrema(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for stint_label, stint_df in df.groupby("manager_stint_label", sort=False):
        stint_df = stint_df.sort_values(["from", "elo"]).reset_index(drop=True)
        manager_name = str(stint_df["manager"].iloc[0])
        high_idx = int(stint_df["elo"].idxmax())
        low_idx = int(stint_df["elo"].idxmin())
        high_row = stint_df.loc[high_idx]
        low_row = stint_df.loc[low_idx]
        if high_idx == low_idx:
            rows.append(
                {
                    "manager": manager_name,
                    "manager_stint_label": stint_label,
                    "from": high_row["from"],
                    "elo": high_row["elo"],
                    "point_type": "High/Low",
                    "label": f"High/Low {round(high_row['elo'])}",
                }
            )
        else:
            rows.extend(
                [
                    {
                        "manager": manager_name,
                        "manager_stint_label": stint_label,
                        "from": high_row["from"],
                        "elo": high_row["elo"],
                        "point_type": "High",
                        "label": f"High {round(high_row['elo'])}",
                    },
                    {
                        "manager": manager_name,
                        "manager_stint_label": stint_label,
                        "from": low_row["from"],
                        "elo": low_row["elo"],
                        "point_type": "Low",
                        "label": f"Low {round(low_row['elo'])}",
                    },
                ]
            )
    return pd.DataFrame(rows)


def get_stint_start_finish(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for stint_label, stint_df in df.groupby("manager_stint_label", sort=False):
        stint_df = stint_df.sort_values(["from", "elo"]).reset_index(drop=True)
        manager_name = str(stint_df["manager"].iloc[0])
        start_row = stint_df.iloc[0]
        finish_row = stint_df.iloc[-1]
        if start_row["from"] == finish_row["from"] and start_row["elo"] == finish_row["elo"]:
            rows.append(
                {
                    "manager": manager_name,
                    "manager_stint_label": stint_label,
                    "from": start_row["from"],
                    "elo": start_row["elo"],
                    "point_type": "Start/Finish",
                    "label": f"Start/Finish {round(start_row['elo'])}",
                }
            )
        else:
            rows.extend(
                [
                    {
                        "manager": manager_name,
                        "manager_stint_label": stint_label,
                        "from": start_row["from"],
                        "elo": start_row["elo"],
                        "point_type": "Start",
                        "label": f"Start {round(start_row['elo'])}",
                    },
                    {
                        "manager": manager_name,
                        "manager_stint_label": stint_label,
                        "from": finish_row["from"],
                        "elo": finish_row["elo"],
                        "point_type": "Finish",
                        "label": f"Finish {round(finish_row['elo'])}",
                    },
                ]
            )
    return pd.DataFrame(rows)


def get_stint_benchmarks(
    df: pd.DataFrame,
    benchmark_histories: dict[str, pd.DataFrame],
    league_team_codes: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for stint_label, stint_df in df.groupby("manager_stint_label", sort=False):
        stint_df = stint_df.sort_values("from").reset_index(drop=True)
        start_date = pd.Timestamp(stint_df["from"].iloc[0])
        end_date = pd.Timestamp(stint_df["from"].iloc[-1])
        manager_name = str(stint_df["manager"].iloc[0])

        league_avg = compute_group_average(
            benchmark_histories,
            league_team_codes,
            start_date,
            end_date,
        )
        top6_avg = compute_group_average(
            benchmark_histories,
            TOP6_CODES,
            start_date,
            end_date,
        )

        if league_avg is not None:
            rows.append(
                {
                    "manager": manager_name,
                    "manager_stint_label": stint_label,
                    "benchmark_type": "League Avg",
                    "value": league_avg,
                    "start_date": start_date,
                    "end_date": end_date,
                    "label": f"League Avg {round(league_avg)}",
                }
            )
        if top6_avg is not None:
            rows.append(
                {
                    "manager": manager_name,
                    "manager_stint_label": stint_label,
                    "benchmark_type": "T6 Avg",
                    "value": top6_avg,
                    "start_date": start_date,
                    "end_date": end_date,
                    "label": f"T6 Avg {round(top6_avg)}",
                }
            )
    return pd.DataFrame(rows)


def build_manager_chart(
    df: pd.DataFrame,
    title: str,
    manager_order: list[str],
    manager_colors: dict[str, str],
    benchmark_histories: dict[str, pd.DataFrame],
    league_team_codes: list[str],
    *,
    showlegend: bool = True,
    include_start_finish: bool = True,
    include_benchmarks: bool = True,
    stock_style: bool = False,
) -> go.Figure:
    fig = go.Figure()
    stint_meta = (
        df[["manager_stint_id", "manager", "manager_stint_label"]]
        .drop_duplicates()
        .sort_values("manager_stint_id")
    )
    shown_managers: set[str] = set()
    shown_benchmarks: set[str] = set()

    for _, meta_row in stint_meta.iterrows():
        stint_id = meta_row["manager_stint_id"]
        manager_name = str(meta_row["manager"])
        stint_label = str(meta_row["manager_stint_label"])
        stint_df = df.loc[df["manager_stint_id"] == stint_id].sort_values("from")
        if stint_df.empty:
            continue

        trace_showlegend = showlegend and manager_name not in shown_managers
        shown_managers.add(manager_name)

        if stock_style:
            stint_points = stint_df[["from", "elo"]].reset_index(drop=True)
            if len(stint_points) == 1:
                fig.add_trace(
                    go.Scatter(
                        x=stint_points["from"],
                        y=stint_points["elo"],
                        mode="markers",
                        name=manager_name,
                        legendgroup=manager_name,
                        showlegend=trace_showlegend,
                        marker=dict(color=STOCK_FLAT_COLOR, size=8),
                        customdata=np.column_stack([[stint_label] * len(stint_points)]),
                        hovertemplate="%{customdata[0]}<br>Date: %{x|%Y-%m-%d}<br>Elo: %{y:.1f}<extra></extra>",
                    )
                )
            else:
                for segment_idx in range(len(stint_points) - 1):
                    start_row = stint_points.iloc[segment_idx]
                    end_row = stint_points.iloc[segment_idx + 1]
                    if end_row["elo"] > start_row["elo"]:
                        line_color = STOCK_UP_COLOR
                    elif end_row["elo"] < start_row["elo"]:
                        line_color = STOCK_DOWN_COLOR
                    else:
                        line_color = STOCK_FLAT_COLOR
                    fig.add_trace(
                        go.Scatter(
                            x=[start_row["from"], end_row["from"]],
                            y=[start_row["elo"], end_row["elo"]],
                            mode="lines",
                            name=manager_name,
                            legendgroup=manager_name,
                            showlegend=trace_showlegend and segment_idx == 0,
                            line=dict(color=line_color, width=3),
                            customdata=np.array([[stint_label], [stint_label]]),
                            hovertemplate="%{customdata[0]}<br>Date: %{x|%Y-%m-%d}<br>Elo: %{y:.1f}<extra></extra>",
                        )
                    )
        else:
            fig.add_trace(
                go.Scatter(
                    x=stint_df["from"],
                    y=stint_df["elo"],
                    mode="lines",
                    name=manager_name,
                    legendgroup=manager_name,
                    showlegend=trace_showlegend,
                    line=dict(color=manager_colors[manager_name], width=3),
                    customdata=np.column_stack([[stint_label] * len(stint_df)]),
                    hovertemplate="%{customdata[0]}<br>Date: %{x|%Y-%m-%d}<br>Elo: %{y:.1f}<extra></extra>",
                )
            )

    if include_benchmarks:
        benchmark_df = get_stint_benchmarks(df, benchmark_histories, league_team_codes)
        benchmark_styles = {
            "League Avg": ("dash", LEAGUE_AVG_COLOR, "middle right"),
            "T6 Avg": ("dot", TOP6_AVG_COLOR, "top right"),
        }
        for _, benchmark_row in benchmark_df.iterrows():
            benchmark_type = str(benchmark_row["benchmark_type"])
            dash_style, color, textposition = benchmark_styles[benchmark_type]
            benchmark_showlegend = showlegend and benchmark_type not in shown_benchmarks
            shown_benchmarks.add(benchmark_type)
            fig.add_trace(
                go.Scatter(
                    x=[benchmark_row["start_date"], benchmark_row["end_date"]],
                    y=[benchmark_row["value"], benchmark_row["value"]],
                    mode="lines+text",
                    name=benchmark_type,
                    legendgroup=benchmark_type,
                    showlegend=benchmark_showlegend,
                    text=["", benchmark_row["label"]],
                    textposition=textposition,
                    line=dict(color=color, width=2, dash=dash_style),
                    opacity=0.8,
                    customdata=np.array([[benchmark_row["manager_stint_label"], benchmark_type]] * 2),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>Elo: %{y:.1f}<extra></extra>",
                )
            )

    marker_frames = [get_stint_extrema(df)]
    if include_start_finish:
        marker_frames.append(get_stint_start_finish(df))
    marker_df = pd.concat(marker_frames, ignore_index=True) if marker_frames else pd.DataFrame()

    point_styles = {
        "High": ("triangle-up", "top center"),
        "Low": ("triangle-down", "bottom center"),
        "High/Low": ("diamond", "top center"),
        "Start": ("circle", "bottom left"),
        "Finish": ("square", "bottom right"),
        "Start/Finish": ("diamond-open", "bottom center"),
    }

    for point_type, (symbol, textposition) in point_styles.items():
        point_df = marker_df.loc[marker_df["point_type"] == point_type]
        if point_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=point_df["from"],
                y=point_df["elo"],
                mode="markers+text",
                text=point_df["label"],
                textposition=textposition,
                textfont=dict(size=10),
                showlegend=False,
                customdata=np.column_stack(
                    [point_df["manager_stint_label"], point_df["point_type"]]
                ),
                hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>Date: %{x|%Y-%m-%d}<br>Elo: %{y:.1f}<extra></extra>",
                marker=dict(
                    size=12,
                    symbol=symbol,
                    color=[manager_colors[str(manager)] for manager in point_df["manager"]],
                    line=dict(color="white", width=1),
                ),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Elo rating",
        legend_title="Manager",
        height=650,
    )
    return fig


def build_stint_summary(df: pd.DataFrame) -> pd.DataFrame:
    benchmark_histories = load_benchmark_histories()
    league_team_codes = [club["team_code"] for club in discover_clubs()]
    rows: list[dict] = []
    for stint_label, stint_df in df.groupby("manager_stint_label", sort=False):
        stint_df = stint_df.sort_values("from").reset_index(drop=True)
        start_date = pd.Timestamp(stint_df["from"].iloc[0])
        end_date = pd.Timestamp(stint_df["from"].iloc[-1])
        league_avg = compute_group_average(
            benchmark_histories,
            league_team_codes,
            start_date,
            end_date,
        )
        top6_avg = compute_group_average(
            benchmark_histories,
            TOP6_CODES,
            start_date,
            end_date,
        )
        rows.append(
            {
                "manager_stint": stint_label,
                "manager": stint_df["manager"].iloc[0],
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "start_elo": round(float(stint_df["elo"].iloc[0])),
                "finish_elo": round(float(stint_df["elo"].iloc[-1])),
                "league_avg_elo": round(league_avg) if league_avg is not None else None,
                "t6_avg_elo": round(top6_avg) if top6_avg is not None else None,
                "high_elo": round(float(stint_df["elo"].max())),
                "low_elo": round(float(stint_df["elo"].min())),
            }
        )
    return pd.DataFrame(rows)


st.title("ClubElo Manager Eras")
st.caption("Combined and manager-level ClubElo charts using Transfermarkt manager timelines.")

clubs = discover_clubs()
if not clubs:
    st.error("No enriched ClubElo files with manager columns were found.")
    st.stop()

default_idx = next((i for i, item in enumerate(clubs) if item["team_code"] == "MUN"), 0)
selected = st.sidebar.selectbox("Club", options=clubs, index=default_idx, format_func=lambda x: x["label"])

club_df = load_clubelo_history(selected["clubelo_stem"])
min_date = club_df["from"].min().date()
max_date = club_df["from"].max().date()
default_start = DEFAULT_MAN_UNITED_START.date() if selected["team_code"] == "MUN" else min_date
if default_start < min_date:
    default_start = min_date
if default_start > max_date:
    default_start = min_date

range_mode = st.sidebar.radio(
    "Range mode",
    options=("Start date", "Date span", "Manager stint"),
)

filter_start = pd.Timestamp(default_start)
filter_end: pd.Timestamp | None = None
range_label = f"From {default_start.isoformat()}"

if range_mode == "Start date":
    start_date = st.sidebar.date_input(
        "Start date",
        value=default_start,
        min_value=min_date,
        max_value=max_date,
    )
    filter_start = pd.Timestamp(start_date)
    range_label = f"From {start_date.isoformat()}"
elif range_mode == "Date span":
    date_span = st.sidebar.date_input(
        "Date span",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_span, tuple):
        if len(date_span) == 2:
            start_date, end_date = date_span
        else:
            start_date = end_date = date_span[0]
    else:
        start_date = end_date = date_span
    filter_start = pd.Timestamp(start_date)
    filter_end = pd.Timestamp(end_date)
    range_label = f"{start_date.isoformat()} to {end_date.isoformat()}"
else:
    stint_options = get_stint_selector_options(club_df)
    default_stint_idx = 0
    for idx, option in enumerate(stint_options):
        if option["start_date"].date() >= default_start:
            default_stint_idx = idx
            break
    selected_stint = st.sidebar.selectbox(
        "Manager stint",
        options=stint_options,
        index=default_stint_idx,
        format_func=lambda x: str(x["label"]),
    )
    filter_start = pd.Timestamp(selected_stint["start_date"])
    filter_end = pd.Timestamp(selected_stint["end_date"])
    range_label = str(selected_stint["label"])

filtered = filter_history(club_df, filter_start, filter_end)
if filtered.empty:
    st.warning("No rows available for the selected club and range.")
    st.stop()

st.caption(f"Selected range: {range_label}")

manager_order = filtered["manager"].drop_duplicates().tolist()
manager_colors = get_manager_colors(manager_order)
benchmark_histories = load_benchmark_histories()
league_team_codes = [club["team_code"] for club in clubs]

st.markdown("### Combined Chart")
combined_fig = build_manager_chart(
    filtered,
    f"{selected['club_name']} Elo rating by manager",
    manager_order,
    manager_colors,
    benchmark_histories,
    league_team_codes,
    showlegend=True,
    include_start_finish=False,
    include_benchmarks=False,
    stock_style=False,
)
st.plotly_chart(combined_fig, use_container_width=True)

st.markdown("### Combined Chart (Stock Style)")
stock_combined_fig = build_manager_chart(
    filtered,
    f"{selected['club_name']} Elo rating by manager",
    manager_order,
    manager_colors,
    benchmark_histories,
    league_team_codes,
    showlegend=True,
    include_start_finish=False,
    include_benchmarks=False,
    stock_style=True,
)
st.plotly_chart(stock_combined_fig, use_container_width=True)

st.markdown("### Stint Summary")
st.dataframe(build_stint_summary(filtered), use_container_width=True, hide_index=True)

st.markdown("### Manager Charts")
tabs = st.tabs(manager_order)
for tab, manager in zip(tabs, manager_order):
    with tab:
        manager_df = filtered.loc[filtered["manager"] == manager].copy()
        manager_fig = build_manager_chart(
            manager_df,
            f"{manager}: {selected['club_name']} Elo rating",
            manager_order,
            manager_colors,
            benchmark_histories,
            league_team_codes,
            showlegend=False,
            include_start_finish=True,
            include_benchmarks=True,
            stock_style=False,
        )
        st.plotly_chart(manager_fig, use_container_width=True)

st.markdown("### Manager Charts (Stock Style)")
stock_tabs = st.tabs(manager_order)
for tab, manager in zip(stock_tabs, manager_order):
    with tab:
        manager_df = filtered.loc[filtered["manager"] == manager].copy()
        stock_manager_fig = build_manager_chart(
            manager_df,
            f"{manager}: {selected['club_name']} Elo rating",
            manager_order,
            manager_colors,
            benchmark_histories,
            league_team_codes,
            showlegend=False,
            include_start_finish=True,
            include_benchmarks=True,
            stock_style=True,
        )
        st.plotly_chart(stock_manager_fig, use_container_width=True)
