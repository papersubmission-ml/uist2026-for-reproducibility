"""
Plotly analysis of AI suggestion ratings from exported session JSON files.

This script loads each requested JSON export file separately, extracts either
`AI_Suggestions[*].ratings` or `AI_Suggestions[*].LLM_as_a_Judge` data for
`LLM-B`, `LLM-CF`, and `LLM-CT`, and
writes a timestamped output folder containing:

- `ratings_long.csv`
- `ratings_wide.csv`
- `distribution_counts.csv`
- `pairwise_correlations.csv`
- `analysis_summary.md`
- Plotly HTML figures for rating distributions and rating-pair correlations
- Matching PNG exports for each Plotly figure
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from packaging.version import Version
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.user_rating_analysis import (
    METRIC_DISPLAY,
    METRIC_ORDER,
    MODEL_COLORS,
    MODEL_DISPLAY,
    MODEL_ORDER,
    build_long_ratings,
    coerce_rating,
    compute_descriptives,
    load_unique_sessions,
)


RATING_LEVELS = [1, 2, 3, 4, 5]
RATING_PAIRS: List[Tuple[str, str]] = list(itertools.combinations(METRIC_ORDER, 2))
NOVELTY_FOCUS_PAIRS: List[Tuple[str, str]] = [
    ("Insight_Novelty", metric)
    for metric in METRIC_ORDER
    if metric != "Insight_Novelty"
]
SCATTER_JITTER = 0.14
SCATTER_SEED = 20260327
RATING_SOURCE_CONFIG = {
    "human": {
        "field": "ratings",
        "label": "Human Ratings",
    },
    "llm_judge": {
        "field": "LLM_as_a_Judge",
        "label": "LLM-as-a-Judge Ratings",
    },
}
CONDITION_LINE_STYLES = {
    "AIH": {
        "color": "#c44536",
        "dash": "solid",
    },
    "HAI": {
        "color": "#1f6f8b",
        "dash": "dash",
    },
}
SOURCE_LINE_STYLES = {
    "human": {
        "label": "Human",
        "dash": "solid",
    },
    "llm_judge": {
        "label": "LLM Judge",
        "dash": "dot",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more exported JSON files to analyze separately.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "plotly_ai_suggestion_outputs",
        help="Parent directory where the timestamped run folder will be created.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional explicit timestamp label. Defaults to current local time.",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Write interactive HTML figures only and skip PNG export.",
    )
    parser.add_argument(
        "--rating-source",
        choices=sorted(RATING_SOURCE_CONFIG.keys()),
        default="human",
        help="Which rating block to analyze from each AI suggestion.",
    )
    return parser.parse_args()


def infer_dataset_label(input_path: Path, metadata: pd.DataFrame) -> str:
    if not metadata.empty:
        conditions = [
            str(value).strip()
            for value in metadata["interaction_condition"].dropna().unique().tolist()
            if str(value).strip()
        ]
        if len(conditions) == 1:
            return conditions[0]

    match = re.search(r"(HAI|AIH)", input_path.stem.upper())
    if match:
        return match.group(1)
    return input_path.stem


def build_wide_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
        "session_uid",
        "export_id",
        "source_file",
        "session_key",
        "participant_id",
        "session_number",
        "interaction_condition",
        "scenario",
        "user_goal",
        "model",
    ]
    base = ratings.copy()
    base["model"] = base["model"].astype(str)
    base["metric"] = base["metric"].astype(str)
    wide = (
        base.set_index(index_cols + ["metric"])["rating"]
        .unstack("metric")
        .reset_index()
        .copy()
    )
    wide.columns.name = None
    for metric in METRIC_ORDER:
        if metric not in wide.columns:
            wide[metric] = np.nan
    return wide[index_cols + METRIC_ORDER]


def build_long_ratings_from_field(
    session_records: Iterable[dict],
    *,
    rating_field: str,
) -> pd.DataFrame:
    rows = []
    for record in session_records:
        session = record["session"]
        suggestions = session.get("AI_Suggestions") or {}
        for model in MODEL_ORDER:
            suggestion = suggestions.get(model) or {}
            suggestion_text = suggestion.get("item", "")
            ratings = suggestion.get(rating_field) or {}
            for metric in METRIC_ORDER:
                rating = coerce_rating(ratings.get(metric))
                if rating is None:
                    continue
                rows.append(
                    {
                        "session_uid": record["session_uid"],
                        "export_id": record["export_id"],
                        "source_file": record["source_file"],
                        "session_key": record["session_key"],
                        "participant_id": session.get("participandID", ""),
                        "session_number": session.get("session", ""),
                        "interaction_condition": record["interaction_condition"],
                        "scenario": session.get("scenario", ""),
                        "user_goal": session.get("user_goal", ""),
                        "model": model,
                        "metric": metric,
                        "rating": rating,
                        "suggestion_text": suggestion_text,
                    }
                )

    if not rows:
        raise ValueError(f"No ratings were found in the JSON exports for `{rating_field}`.")

    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["metric"] = pd.Categorical(df["metric"], categories=METRIC_ORDER, ordered=True)
    return df.sort_values(by=["session_uid", "metric", "model"]).reset_index(drop=True)


def compute_distribution_counts(ratings: pd.DataFrame) -> pd.DataFrame:
    with_levels = ratings.copy()
    with_levels["rating_level"] = with_levels["rating"].round().astype(int)
    full_index = pd.MultiIndex.from_product(
        [METRIC_ORDER, MODEL_ORDER, RATING_LEVELS],
        names=["metric", "model", "rating_level"],
    )
    counts = (
        with_levels.groupby(["metric", "model", "rating_level"], observed=True)
        .size()
        .reindex(full_index, fill_value=0)
        .rename("count")
        .reset_index()
    )
    counts["n_total"] = counts.groupby(["metric", "model"], observed=True)["count"].transform("sum")
    counts["percentage"] = np.where(
        counts["n_total"] > 0,
        counts["count"] / counts["n_total"] * 100.0,
        np.nan,
    )
    counts["metric_label"] = counts["metric"].map(METRIC_DISPLAY)
    counts["model_label"] = counts["model"].map(MODEL_DISPLAY)
    return counts


def compute_pairwise_correlations(wide_ratings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        model_df = wide_ratings[wide_ratings["model"] == model]
        for metric_x, metric_y in RATING_PAIRS:
            pair_df = model_df[[metric_x, metric_y]].dropna()
            pearson_r = np.nan
            spearman_rho = np.nan
            if len(pair_df) >= 2:
                pearson_r = pair_df[metric_x].corr(pair_df[metric_y], method="pearson")
                spearman_rho = pair_df[metric_x].corr(pair_df[metric_y], method="spearman")
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_DISPLAY[model],
                    "metric_x": metric_x,
                    "metric_x_label": METRIC_DISPLAY[metric_x],
                    "metric_y": metric_y,
                    "metric_y_label": METRIC_DISPLAY[metric_y],
                    "n_rows": len(pair_df),
                    "pearson_r": pearson_r,
                    "spearman_rho": spearman_rho,
                    "abs_spearman_rho": abs(spearman_rho) if pd.notna(spearman_rho) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_spearman_matrices(wide_ratings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    matrices: dict[str, pd.DataFrame] = {}
    for model in MODEL_ORDER:
        model_df = wide_ratings[wide_ratings["model"] == model]
        matrix = model_df[METRIC_ORDER].corr(method="spearman")
        matrices[model] = matrix.reindex(index=METRIC_ORDER, columns=METRIC_ORDER)
    return matrices


def format_corr(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2f}"


def write_png_figure(fig: go.Figure, png_path: Path) -> Path:
    try:
        import kaleido  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PNG export requires the `kaleido` package. Install it with "
            "`python -m pip install \"kaleido<1\"`, or rerun with "
            "`--html-only` to skip PNG generation."
        ) from exc

    plotly_version = Version(metadata.version("plotly"))
    kaleido_version = Version(metadata.version("kaleido"))
    if plotly_version < Version("6.1.1") and kaleido_version >= Version("1.0.0"):
        raise RuntimeError(
            f"Detected Plotly {plotly_version} with Kaleido {kaleido_version}, "
            "which cannot export static images together. Install a compatible "
            "Kaleido release with `python -m pip install \"kaleido<1\"`, "
            "upgrade Plotly to 6.1.1+, or rerun with `--html-only`."
        )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(png_path), scale=2)
    return png_path


def write_plotly_figure(
    fig: go.Figure,
    output_path: Path,
    *,
    png_dir: Path | None = None,
    write_png: bool = True,
) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="directory", full_html=True)
    if not write_png:
        return None
    if png_dir is None:
        png_path = output_path.with_suffix(".png")
    else:
        png_path = png_dir / f"{output_path.stem}.png"
    return write_png_figure(fig, png_path)


def make_distribution_figure(
    distribution_counts: pd.DataFrame,
    descriptives: pd.DataFrame,
    dataset_label: str,
    num_sessions: int,
    rating_label: str,
) -> go.Figure:
    subplot_titles = [METRIC_DISPLAY[metric] for metric in METRIC_ORDER]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.13,
    )

    max_percentage = max(5.0, float(distribution_counts["percentage"].max()) * 1.15)
    rating_labels = [str(level) for level in RATING_LEVELS]
    bar_width = 0.22
    bar_offsets = {
        "LLM-B": -bar_width,
        "LLM-CF": 0.0,
        "LLM-CT": bar_width,
    }

    for plot_index, metric in enumerate(METRIC_ORDER):
        row = plot_index // 2 + 1
        col = plot_index % 2 + 1
        metric_df = distribution_counts[distribution_counts["metric"] == metric]
        stat_lines = []

        for model in MODEL_ORDER:
            model_df = metric_df[metric_df["model"] == model].copy()
            customdata = np.column_stack(
                [
                    model_df["count"].to_numpy(),
                    model_df["percentage"].to_numpy(),
                    model_df["rating_level"].to_numpy(),
                ]
            )
            fig.add_trace(
                go.Bar(
                    x=model_df["rating_level"].to_numpy(dtype=float) + bar_offsets[model],
                    y=model_df["percentage"],
                    name=MODEL_DISPLAY[model],
                    legendgroup=model,
                    showlegend=plot_index == 0,
                    marker_color=MODEL_COLORS[model],
                    opacity=0.92,
                    width=bar_width * 0.92,
                    customdata=customdata,
                    hovertemplate=(
                        "Model: %{fullData.name}<br>"
                        "Rating: %{customdata[2]:.0f}<br>"
                        "Count: %{customdata[0]}<br>"
                        "Share: %{customdata[1]:.1f}%<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

            mean_row = descriptives[
                (descriptives["metric"] == metric) & (descriptives["model"] == model)
            ]
            if not mean_row.empty:
                mean_rating = float(mean_row.iloc[0]["mean_rating"])
                median_rating = float(mean_row.iloc[0]["median_rating"])
                fig.add_trace(
                    go.Scatter(
                        x=[mean_rating, mean_rating],
                        y=[0.0, max_percentage],
                        mode="lines",
                        showlegend=False,
                        legendgroup=model,
                        line=dict(color=MODEL_COLORS[model], width=2, dash="dash"),
                        hovertemplate=(
                            f"Model: {MODEL_DISPLAY[model]}<br>"
                            f"Mean rating: {mean_rating:.2f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[median_rating, median_rating],
                        y=[0.0, max_percentage],
                        mode="lines",
                        showlegend=False,
                        legendgroup=model,
                        line=dict(color=MODEL_COLORS[model], width=2, dash="dot"),
                        hovertemplate=(
                            f"Model: {MODEL_DISPLAY[model]}<br>"
                            f"Median rating: {median_rating:.2f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[mean_rating],
                        y=[max_percentage * 0.98],
                        mode="markers",
                        showlegend=False,
                        legendgroup=model,
                        marker=dict(color=MODEL_COLORS[model], size=9, symbol="diamond"),
                        hovertemplate=(
                            f"Model: {MODEL_DISPLAY[model]}<br>"
                            f"Mean rating: {mean_rating:.2f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[median_rating],
                        y=[max_percentage * 0.90],
                        mode="markers",
                        showlegend=False,
                        legendgroup=model,
                        marker=dict(color=MODEL_COLORS[model], size=8, symbol="circle"),
                        hovertemplate=(
                            f"Model: {MODEL_DISPLAY[model]}<br>"
                            f"Median rating: {median_rating:.2f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                stat_lines.append(
                    f"{MODEL_DISPLAY[model]}: mean {mean_rating:.2f}, median {median_rating:.2f}"
                )

        axis_index = plot_index + 1
        xref = "x domain" if axis_index == 1 else f"x{axis_index} domain"
        yref = "y domain" if axis_index == 1 else f"y{axis_index} domain"
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref=xref,
            yref=yref,
            text="<br>".join(stat_lines),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font=dict(size=10, color="#33415c"),
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(51,65,92,0.22)",
            borderwidth=1,
        )

        fig.update_xaxes(
            title_text="Rating",
            tickmode="array",
            tickvals=RATING_LEVELS,
            ticktext=rating_labels,
            range=[0.5, 5.5],
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="Share of ratings (%)",
            range=[0, max_percentage],
            row=row,
            col=col,
        )

    fig.update_layout(
        title=(
            f"{dataset_label}: {rating_label} Distributions"
            f"<br><sup>{num_sessions} sessions; dashed lines mark means, dotted lines mark medians</sup>"
        ),
        template="plotly_white",
        barmode="group",
        width=1200,
        height=900,
        legend_title_text="Model",
        margin=dict(t=90, r=40, b=60, l=60),
    )
    return fig


def make_pairwise_scatter_figure(
    wide_ratings: pd.DataFrame,
    dataset_label: str,
    model: str,
    rating_label: str,
) -> go.Figure:
    model_df = wide_ratings[wide_ratings["model"] == model].copy()
    subplot_titles = []
    for metric_x, metric_y in RATING_PAIRS:
        pair_df = model_df[[metric_x, metric_y]].dropna()
        pearson_r = pair_df[metric_x].corr(pair_df[metric_y], method="pearson") if len(pair_df) >= 2 else np.nan
        spearman_rho = pair_df[metric_x].corr(pair_df[metric_y], method="spearman") if len(pair_df) >= 2 else np.nan
        subplot_titles.append(
            (
                f"{METRIC_DISPLAY[metric_x]} vs {METRIC_DISPLAY[metric_y]}"
                f"<br><sup>Spearman {format_corr(spearman_rho)}, Pearson {format_corr(pearson_r)}, n={len(pair_df)}</sup>"
            )
        )

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )

    rng = np.random.default_rng(SCATTER_SEED)

    for plot_index, (metric_x, metric_y) in enumerate(RATING_PAIRS):
        row = plot_index // 3 + 1
        col = plot_index % 3 + 1
        pair_df = model_df.dropna(subset=[metric_x, metric_y]).copy()
        actual_x = pair_df[metric_x].to_numpy(dtype=float)
        actual_y = pair_df[metric_y].to_numpy(dtype=float)
        jittered_x = actual_x + rng.uniform(-SCATTER_JITTER, SCATTER_JITTER, size=len(pair_df))
        jittered_y = actual_y + rng.uniform(-SCATTER_JITTER, SCATTER_JITTER, size=len(pair_df))

        hover_data = np.column_stack(
            [
                actual_x,
                actual_y,
                pair_df["session_uid"].astype(str).to_numpy(),
                pair_df["participant_id"].astype(str).to_numpy(),
                pair_df["session_number"].astype(str).to_numpy(),
                pair_df["scenario"].astype(str).to_numpy(),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=jittered_x,
                y=jittered_y,
                mode="markers",
                showlegend=False,
                marker=dict(
                    color=MODEL_COLORS[model],
                    size=8,
                    opacity=0.58,
                    line=dict(color="rgba(17, 17, 17, 0.20)", width=0.5),
                ),
                customdata=hover_data,
                hovertemplate=(
                    f"{METRIC_DISPLAY[metric_x]}: %{{customdata[0]:.0f}}<br>"
                    f"{METRIC_DISPLAY[metric_y]}: %{{customdata[1]:.0f}}<br>"
                    "Session UID: %{customdata[2]}<br>"
                    "Participant ID: %{customdata[3]}<br>"
                    "Session: %{customdata[4]}<br>"
                    "Scenario: %{customdata[5]}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        if len(pair_df) >= 2 and np.ptp(actual_x) > 0:
            slope, intercept = np.polyfit(actual_x, actual_y, deg=1)
            line_x = np.array([1.0, 5.0], dtype=float)
            line_y = intercept + slope * line_x
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    showlegend=False,
                    line=dict(color="rgba(33, 33, 33, 0.8)", width=2),
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(
            title_text=METRIC_DISPLAY[metric_x],
            range=[0.7, 5.3],
            tickmode="array",
            tickvals=RATING_LEVELS,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text=METRIC_DISPLAY[metric_y],
            range=[0.7, 5.3],
            tickmode="array",
            tickvals=RATING_LEVELS,
            row=row,
            col=col,
        )

    fig.update_layout(
        title=(
            f"{dataset_label}: {rating_label} Pair Relationships for {MODEL_DISPLAY[model]}"
            "<br><sup>Jittered points reduce overlap; line is a least-squares fit on the original ratings</sup>"
        ),
        template="plotly_white",
        width=1650,
        height=900,
        margin=dict(t=110, r=30, b=50, l=70),
    )
    return fig


def make_novelty_focus_scatter_figure(
    wide_ratings: pd.DataFrame,
    dataset_label: str,
    rating_label: str,
) -> go.Figure:
    subplot_titles = []
    for model in MODEL_ORDER:
        model_df = wide_ratings[wide_ratings["model"] == model].copy()
        for metric_x, metric_y in NOVELTY_FOCUS_PAIRS:
            pair_df = model_df[[metric_x, metric_y]].dropna()
            pearson_r = pair_df[metric_x].corr(pair_df[metric_y], method="pearson") if len(pair_df) >= 2 else np.nan
            spearman_rho = pair_df[metric_x].corr(pair_df[metric_y], method="spearman") if len(pair_df) >= 2 else np.nan
            subplot_titles.append(
                (
                    f"{MODEL_DISPLAY[model]}: {METRIC_DISPLAY[metric_x]} vs {METRIC_DISPLAY[metric_y]}"
                    f"<br><sup>Spearman {format_corr(spearman_rho)}, Pearson {format_corr(pearson_r)}, n={len(pair_df)}</sup>"
                )
            )

    fig = make_subplots(
        rows=len(MODEL_ORDER),
        cols=len(NOVELTY_FOCUS_PAIRS),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    rng = np.random.default_rng(SCATTER_SEED)

    for row, model in enumerate(MODEL_ORDER, start=1):
        model_df = wide_ratings[wide_ratings["model"] == model].copy()
        for col, (metric_x, metric_y) in enumerate(NOVELTY_FOCUS_PAIRS, start=1):
            pair_df = model_df.dropna(subset=[metric_x, metric_y]).copy()
            actual_x = pair_df[metric_x].to_numpy(dtype=float)
            actual_y = pair_df[metric_y].to_numpy(dtype=float)
            jittered_x = actual_x + rng.uniform(-SCATTER_JITTER, SCATTER_JITTER, size=len(pair_df))
            jittered_y = actual_y + rng.uniform(-SCATTER_JITTER, SCATTER_JITTER, size=len(pair_df))

            hover_data = np.column_stack(
                [
                    actual_x,
                    actual_y,
                    pair_df["session_uid"].astype(str).to_numpy(),
                    pair_df["participant_id"].astype(str).to_numpy(),
                    pair_df["session_number"].astype(str).to_numpy(),
                    pair_df["scenario"].astype(str).to_numpy(),
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=jittered_x,
                    y=jittered_y,
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color=MODEL_COLORS[model],
                        size=7,
                        opacity=0.56,
                        line=dict(color="rgba(17, 17, 17, 0.20)", width=0.5),
                    ),
                    customdata=hover_data,
                    hovertemplate=(
                        f"{METRIC_DISPLAY[metric_x]}: %{{customdata[0]:.0f}}<br>"
                        f"{METRIC_DISPLAY[metric_y]}: %{{customdata[1]:.0f}}<br>"
                        "Session UID: %{customdata[2]}<br>"
                        "Participant ID: %{customdata[3]}<br>"
                        "Session: %{customdata[4]}<br>"
                        "Scenario: %{customdata[5]}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

            if len(pair_df) >= 2 and np.ptp(actual_x) > 0:
                slope, intercept = np.polyfit(actual_x, actual_y, deg=1)
                line_x = np.array([1.0, 5.0], dtype=float)
                line_y = intercept + slope * line_x
                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines",
                        showlegend=False,
                        line=dict(color="rgba(33, 33, 33, 0.8)", width=2),
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            fig.update_xaxes(
                title_text=METRIC_DISPLAY[metric_x],
                range=[0.7, 5.3],
                tickmode="array",
                tickvals=RATING_LEVELS,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text=METRIC_DISPLAY[metric_y],
                range=[0.7, 5.3],
                tickmode="array",
                tickvals=RATING_LEVELS,
                row=row,
                col=col,
            )

    fig.update_layout(
        title=(
            f"{dataset_label}: Novelty-Centered {rating_label} Relationships Across Models"
            "<br><sup>Insight / Novelty stays on the x-axis; each row is a model; line is a least-squares fit on the original ratings</sup>"
        ),
        template="plotly_white",
        width=1650,
        height=1280,
        margin=dict(t=120, r=30, b=60, l=70),
    )
    return fig


def make_novelty_focus_line_figure(
    wide_ratings: pd.DataFrame,
    dataset_label: str,
    rating_label: str,
) -> go.Figure:
    subplot_titles = []
    for model in MODEL_ORDER:
        model_df = wide_ratings[wide_ratings["model"] == model].copy()
        for metric_x, metric_y in NOVELTY_FOCUS_PAIRS:
            pair_df = model_df[[metric_x, metric_y]].dropna()
            pearson_r = pair_df[metric_x].corr(pair_df[metric_y], method="pearson") if len(pair_df) >= 2 else np.nan
            spearman_rho = pair_df[metric_x].corr(pair_df[metric_y], method="spearman") if len(pair_df) >= 2 else np.nan
            subplot_titles.append(
                (
                    f"{MODEL_DISPLAY[model]}: {METRIC_DISPLAY[metric_x]} vs {METRIC_DISPLAY[metric_y]}"
                    f"<br><sup>Spearman {format_corr(spearman_rho)}, Pearson {format_corr(pearson_r)}, n={len(pair_df)}</sup>"
                )
            )

    fig = make_subplots(
        rows=len(MODEL_ORDER),
        cols=len(NOVELTY_FOCUS_PAIRS),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for row, model in enumerate(MODEL_ORDER, start=1):
        model_df = wide_ratings[wide_ratings["model"] == model].copy()
        for col, (metric_x, metric_y) in enumerate(NOVELTY_FOCUS_PAIRS, start=1):
            pair_df = model_df.dropna(subset=[metric_x, metric_y]).copy()
            actual_x = pair_df[metric_x].to_numpy(dtype=float)
            actual_y = pair_df[metric_y].to_numpy(dtype=float)

            if len(pair_df) >= 2 and np.ptp(actual_x) > 0:
                slope, intercept = np.polyfit(actual_x, actual_y, deg=1)
                line_x = np.array([1.0, 5.0], dtype=float)
                line_y = intercept + slope * line_x
                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines",
                        showlegend=False,
                        line=dict(color=MODEL_COLORS[model], width=3),
                        hovertemplate=(
                            f"{MODEL_DISPLAY[model]}<br>"
                            f"{METRIC_DISPLAY[metric_x]}: %{{x:.2f}}<br>"
                            f"{METRIC_DISPLAY[metric_y]} fit: %{{y:.2f}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

            fig.update_xaxes(
                title_text=METRIC_DISPLAY[metric_x],
                range=[0.7, 5.3],
                tickmode="array",
                tickvals=RATING_LEVELS,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text=METRIC_DISPLAY[metric_y],
                range=[0.7, 5.3],
                tickmode="array",
                tickvals=RATING_LEVELS,
                row=row,
                col=col,
            )

    fig.update_layout(
        title=(
            f"{dataset_label}: Novelty-Centered {rating_label} Linear Trends Across Models"
            "<br><sup>Each line is a least-squares fit on the original ratings; points are omitted</sup>"
        ),
        template="plotly_white",
        width=1650,
        height=1280,
        margin=dict(t=120, r=30, b=60, l=70),
    )
    return fig


def make_condition_comparison_novelty_line_figure(
    wide_ratings: pd.DataFrame,
    rating_label: str,
) -> go.Figure:
    subplot_titles = [
        f"{MODEL_DISPLAY[model]}: {METRIC_DISPLAY[metric_y]} vs {METRIC_DISPLAY[metric_x]}"
        for model in MODEL_ORDER
        for metric_x, metric_y in NOVELTY_FOCUS_PAIRS
    ]
    fig = make_subplots(
        rows=len(MODEL_ORDER),
        cols=len(NOVELTY_FOCUS_PAIRS),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for row, model in enumerate(MODEL_ORDER, start=1):
        model_df = wide_ratings[wide_ratings["model"] == model].copy()
        for col, (metric_x, metric_y) in enumerate(NOVELTY_FOCUS_PAIRS, start=1):
            for condition in ["AIH", "HAI"]:
                condition_df = model_df[model_df["interaction_condition"] == condition].dropna(
                    subset=[metric_x, metric_y]
                )
                actual_x = condition_df[metric_x].to_numpy(dtype=float)
                actual_y = condition_df[metric_y].to_numpy(dtype=float)
                if len(condition_df) < 2 or np.ptp(actual_x) == 0:
                    continue

                slope, intercept = np.polyfit(actual_x, actual_y, deg=1)
                line_x = np.array([1.0, 5.0], dtype=float)
                line_y = intercept + slope * line_x
                style = CONDITION_LINE_STYLES[condition]
                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines",
                        name=condition,
                        legendgroup=condition,
                        showlegend=(row == 1 and col == 1),
                        line=dict(color=style["color"], width=3, dash=style["dash"]),
                        hovertemplate=(
                            f"Condition: {condition}<br>"
                            f"Model: {MODEL_DISPLAY[model]}<br>"
                            f"{METRIC_DISPLAY[metric_x]}: %{{x:.2f}}<br>"
                            f"{METRIC_DISPLAY[metric_y]} fit: %{{y:.2f}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

            fig.update_xaxes(
                title_text=METRIC_DISPLAY[metric_x],
                range=[0.7, 5.3],
                tickmode="array",
                tickvals=RATING_LEVELS,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text=METRIC_DISPLAY[metric_y],
                range=[0.7, 5.3],
                tickmode="array",
                tickvals=RATING_LEVELS,
                row=row,
                col=col,
            )

    fig.update_layout(
        title=(
            f"AIH vs HAI: Novelty-Centered {rating_label} Linear Trends Across Models"
            "<br><sup>Each line is a least-squares fit on the original ratings; AIH and HAI are overlaid in each subplot</sup>"
        ),
        template="plotly_white",
        width=1650,
        height=1280,
        margin=dict(t=120, r=30, b=60, l=70),
        legend_title_text="Condition",
    )
    return fig


def make_condition_source_comparison_novelty_line_figure(
    ratings_by_source: dict[str, pd.DataFrame],
) -> go.Figure:
    subplot_titles = [
        f"{MODEL_DISPLAY[model]}: {METRIC_DISPLAY[metric_y]} vs {METRIC_DISPLAY[metric_x]}"
        for model in MODEL_ORDER
        for metric_x, metric_y in NOVELTY_FOCUS_PAIRS
    ]
    fig = make_subplots(
        rows=len(MODEL_ORDER),
        cols=len(NOVELTY_FOCUS_PAIRS),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    wide_by_source = {
        source: build_wide_ratings(source_ratings)
        for source, source_ratings in ratings_by_source.items()
    }

    for row, model in enumerate(MODEL_ORDER, start=1):
        for col, (metric_x, metric_y) in enumerate(NOVELTY_FOCUS_PAIRS, start=1):
            for condition in ["AIH", "HAI"]:
                for source in ["human", "llm_judge"]:
                    source_wide = wide_by_source.get(source)
                    if source_wide is None:
                        continue
                    pair_df = source_wide[
                        (source_wide["model"] == model)
                        & (source_wide["interaction_condition"] == condition)
                    ].dropna(subset=[metric_x, metric_y])
                    actual_x = pair_df[metric_x].to_numpy(dtype=float)
                    actual_y = pair_df[metric_y].to_numpy(dtype=float)
                    if len(pair_df) < 2 or np.ptp(actual_x) == 0:
                        continue

                    slope, intercept = np.polyfit(actual_x, actual_y, deg=1)
                    line_x = np.array([1.0, 5.0], dtype=float)
                    line_y = intercept + slope * line_x
                    condition_style = CONDITION_LINE_STYLES[condition]
                    source_style = SOURCE_LINE_STYLES[source]
                    legend_name = f"{condition} {source_style['label']}"
                    fig.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode="lines",
                            name=legend_name,
                            legendgroup=legend_name,
                            showlegend=(row == 1 and col == 1),
                            line=dict(
                                color=condition_style["color"],
                                width=3,
                                dash=source_style["dash"],
                            ),
                            hovertemplate=(
                                f"Condition: {condition}<br>"
                                f"Source: {source_style['label']}<br>"
                                f"Model: {MODEL_DISPLAY[model]}<br>"
                                f"{METRIC_DISPLAY[metric_x]}: %{{x:.2f}}<br>"
                                f"{METRIC_DISPLAY[metric_y]} fit: %{{y:.2f}}<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )

            fig.update_xaxes(
                title_text=METRIC_DISPLAY[metric_x],
                range=[0.7, 5.3],
                tickmode="array",
                tickvals=RATING_LEVELS,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text=METRIC_DISPLAY[metric_y],
                range=[0.7, 5.3],
                tickmode="array",
                tickvals=RATING_LEVELS,
                row=row,
                col=col,
            )

    fig.update_layout(
        title=(
            "AIH vs HAI: Novelty-Centered Human and LLM-as-a-Judge Linear Trends Across Models"
            "<br><sup>Color encodes condition; line style encodes rating source; each line is a least-squares fit on the original ratings</sup>"
        ),
        template="plotly_white",
        width=1650,
        height=1280,
        margin=dict(t=120, r=30, b=60, l=70),
        legend_title_text="Condition x Source",
    )
    return fig


def compute_condition_source_descriptives(combined_ratings: pd.DataFrame) -> pd.DataFrame:
    summary = (
        combined_ratings.groupby(
            ["rating_source", "interaction_condition", "model", "metric"],
            observed=True,
        )["rating"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    summary["metric_label"] = summary["metric"].map(METRIC_DISPLAY)
    summary["model_label"] = summary["model"].map(MODEL_DISPLAY)
    summary["source_label"] = summary["rating_source"].map(
        lambda value: SOURCE_LINE_STYLES[value]["label"]
    )
    summary["condition_source_label"] = summary.apply(
        lambda row: f"{row['interaction_condition']} {SOURCE_LINE_STYLES[row['rating_source']]['label']}",
        axis=1,
    )
    return summary.rename(
        columns={
            "mean": "mean_rating",
            "std": "sd_rating",
            "median": "median_rating",
            "count": "n",
        }
    )


def make_condition_source_mean_profile_figure(
    summary: pd.DataFrame,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=len(MODEL_ORDER),
        subplot_titles=[MODEL_DISPLAY[model] for model in MODEL_ORDER],
        horizontal_spacing=0.08,
    )
    metric_positions = np.arange(1, len(METRIC_ORDER) + 1)
    metric_labels = [METRIC_DISPLAY[metric] for metric in METRIC_ORDER]

    for col, model in enumerate(MODEL_ORDER, start=1):
        model_summary = summary[summary["model"] == model].copy()
        for condition in ["AIH", "HAI"]:
            for source in ["human", "llm_judge"]:
                subset = (
                    model_summary[
                        (model_summary["interaction_condition"] == condition)
                        & (model_summary["rating_source"] == source)
                    ]
                    .set_index("metric")
                    .reindex(METRIC_ORDER)
                    .reset_index()
                )
                if subset["mean_rating"].isna().all():
                    continue
                style = CONDITION_LINE_STYLES[condition]
                source_style = SOURCE_LINE_STYLES[source]
                legend_name = f"{condition} {source_style['label']}"
                customdata = np.column_stack(
                    [
                        subset["metric"].map(METRIC_DISPLAY).to_numpy(),
                        subset["n"].fillna(0).to_numpy(),
                        subset["mean_rating"].to_numpy(),
                        subset["sd_rating"].to_numpy(),
                    ]
                )
                fig.add_trace(
                    go.Scatter(
                        x=metric_positions,
                        y=subset["mean_rating"],
                        mode="lines+markers",
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=col == 1,
                        line=dict(color=style["color"], width=3, dash=source_style["dash"]),
                        marker=dict(size=9),
                        customdata=customdata,
                        hovertemplate=(
                            f"Condition: {condition}<br>"
                            f"Source: {source_style['label']}<br>"
                            "Metric: %{customdata[0]}<br>"
                            "Mean: %{customdata[2]:.2f}<br>"
                            "SD: %{customdata[3]:.2f}<br>"
                            "n: %{customdata[1]:.0f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )

        fig.update_xaxes(
            title_text="Metric",
            tickmode="array",
            tickvals=metric_positions,
            ticktext=metric_labels,
            tickangle=25,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title_text="Mean Rating",
            range=[1.0, 5.1],
            tickmode="array",
            tickvals=RATING_LEVELS,
            row=1,
            col=col,
        )

    fig.update_layout(
        title=(
            "AIH vs HAI: Mean Rating Profiles Across Metrics"
            "<br><sup>Color encodes condition; line style encodes rating source</sup>"
        ),
        template="plotly_white",
        width=1600,
        height=520,
        margin=dict(t=100, r=30, b=80, l=70),
        legend_title_text="Condition x Source",
    )
    return fig


def make_condition_source_heatmap_figure(
    summary: pd.DataFrame,
) -> go.Figure:
    panel_order = [
        ("AIH", "human"),
        ("HAI", "human"),
        ("AIH", "llm_judge"),
        ("HAI", "llm_judge"),
    ]
    panel_titles = [
        f"{condition} {SOURCE_LINE_STYLES[source]['label']}"
        for condition, source in panel_order
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=panel_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )
    metric_labels = [METRIC_DISPLAY[metric] for metric in METRIC_ORDER]
    model_labels = [MODEL_DISPLAY[model] for model in MODEL_ORDER]

    for index, (condition, source) in enumerate(panel_order):
        row = index // 2 + 1
        col = index % 2 + 1
        panel = (
            summary[
                (summary["interaction_condition"] == condition)
                & (summary["rating_source"] == source)
            ][["model", "metric", "mean_rating"]]
            .pivot(index="model", columns="metric", values="mean_rating")
            .reindex(index=MODEL_ORDER, columns=METRIC_ORDER)
        )
        z_values = panel.to_numpy(dtype=float)
        text_values = [[f"{value:.2f}" if pd.notna(value) else "NA" for value in row_vals] for row_vals in z_values]
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=metric_labels,
                y=model_labels,
                zmin=1,
                zmax=5,
                colorscale="YlGnBu",
                text=text_values,
                texttemplate="%{text}",
                hovertemplate="Metric: %{x}<br>Model: %{y}<br>Mean rating: %{z:.2f}<extra></extra>",
                showscale=index == len(panel_order) - 1,
                colorbar=dict(title="Mean"),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(tickangle=25, row=row, col=col)

    fig.update_layout(
        title=(
            "AIH vs HAI: Mean Rating Heatmaps for Human and LLM-as-a-Judge"
            "<br><sup>Each panel shows mean ratings by model and metric</sup>"
        ),
        template="plotly_white",
        width=1350,
        height=900,
        margin=dict(t=95, r=30, b=60, l=80),
    )
    return fig


def make_spearman_heatmap_figure(
    wide_ratings: pd.DataFrame,
    dataset_label: str,
    rating_label: str,
) -> go.Figure:
    matrices = build_spearman_matrices(wide_ratings)
    metric_labels = [METRIC_DISPLAY[metric] for metric in METRIC_ORDER]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[MODEL_DISPLAY[model] for model in MODEL_ORDER],
        horizontal_spacing=0.08,
    )

    for plot_index, model in enumerate(MODEL_ORDER, start=1):
        matrix = matrices[model]
        z_values = matrix.to_numpy(dtype=float)
        text_values = [[format_corr(value) for value in row] for row in z_values]
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=metric_labels,
                y=metric_labels,
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale="RdBu",
                reversescale=True,
                text=text_values,
                texttemplate="%{text}",
                hovertemplate="x: %{x}<br>y: %{y}<br>Spearman rho: %{z:.2f}<extra></extra>",
                showscale=plot_index == len(MODEL_ORDER),
                colorbar=dict(title="rho"),
            ),
            row=1,
            col=plot_index,
        )
        fig.update_xaxes(tickangle=35, row=1, col=plot_index)

    fig.update_layout(
        title=f"{dataset_label}: Spearman Correlation Heatmaps Across {rating_label}",
        template="plotly_white",
        width=1450,
        height=520,
        margin=dict(t=80, r=30, b=60, l=60),
    )
    return fig


def build_summary_markdown(
    dataset_label: str,
    input_path: Path,
    output_dir: Path,
    metadata: pd.DataFrame,
    ratings: pd.DataFrame,
    descriptives: pd.DataFrame,
    correlations: pd.DataFrame,
    static_images_written: bool,
    html_only: bool,
    rating_label: str,
    rating_source: str,
) -> str:
    participant_ids = metadata["participant_id"].replace("", np.nan)
    num_participants = participant_ids.nunique(dropna=True)
    num_sessions = metadata["session_uid"].nunique()
    num_ratings = len(ratings)
    duplicate_count = metadata.attrs.get("duplicate_count", 0)

    lines = [
        f"# {dataset_label} {rating_label} Analysis",
        "",
        "## Dataset",
        f"- Source file: `{input_path}`",
        f"- Output directory: `{output_dir}`",
        f"- Rating source: `{rating_source}`",
        f"- Unique logical sessions: {num_sessions}",
        f"- Participants with non-empty IDs: {num_participants}",
        f"- Rating rows analyzed: {num_ratings}",
        f"- Duplicate exports removed: {duplicate_count}",
        "",
        "## Mean Rating Leaders",
    ]

    for metric in METRIC_ORDER:
        metric_df = descriptives[descriptives["metric"] == metric].sort_values(
            "mean_rating",
            ascending=False,
        )
        if metric_df.empty:
            continue
        leader = metric_df.iloc[0]
        lines.append(
            f"- {METRIC_DISPLAY[metric]}: {leader['model_label']} "
            f"(mean {leader['mean_rating']:.2f}, SD {leader['sd_rating']:.2f}, n={int(leader['n'])})"
        )

    lines.extend(["", "## Strongest Spearman Correlations"])
    for model in MODEL_ORDER:
        model_corr = (
            correlations[correlations["model"] == model]
            .sort_values("abs_spearman_rho", ascending=False)
            .head(3)
        )
        if model_corr.empty:
            lines.append(f"- {MODEL_DISPLAY[model]}: no complete rating pairs available.")
            continue
        parts = []
        for _, row in model_corr.iterrows():
            parts.append(
                f"{row['metric_x_label']} vs {row['metric_y_label']} "
                f"(rho={format_corr(row['spearman_rho'])}, n={int(row['n_rows'])})"
            )
        lines.append(f"- {MODEL_DISPLAY[model]}: " + "; ".join(parts))

    lines.extend(
        [
            "",
            "## Figures",
            "- `figures/rating_distribution.html`",
            "- `figures/spearman_correlation_heatmaps.html`",
            "- `figures/rating_pairs_LLM-B.html`",
            "- `figures/rating_pairs_LLM-CF.html`",
            "- `figures/rating_pairs_LLM-CT.html`",
            "- `figures/rating_pairs_novelty_x_all_models.html`",
            "- `figures/rating_pairs_novelty_x_lines_all_models.html`",
        ]
    )
    if static_images_written:
        lines.append("- Matching `.png` files were written to `figures_png/`.")
    elif html_only:
        lines.append("- PNG export was intentionally skipped because `--html-only` was used.")
    else:
        lines.append("- PNG export is expected for each figure. If files are missing, rerun after installing `kaleido`.")

    return "\n".join(lines) + "\n"


def analyze_one_file(
    input_path: Path,
    run_output_dir: Path,
    *,
    html_only: bool,
    rating_source: str,
) -> dict:
    session_records, metadata = load_unique_sessions(input_path)
    rating_config = RATING_SOURCE_CONFIG[rating_source]
    rating_label = rating_config["label"]
    if rating_source == "human":
        ratings = build_long_ratings(session_records)
    else:
        ratings = build_long_ratings_from_field(
            session_records,
            rating_field=rating_config["field"],
        )
    wide_ratings = build_wide_ratings(ratings)
    descriptives = compute_descriptives(ratings)
    distribution_counts = compute_distribution_counts(ratings)
    correlations = compute_pairwise_correlations(wide_ratings)

    dataset_label = infer_dataset_label(input_path, metadata)
    output_dir = run_output_dir / dataset_label
    figures_dir = output_dir / "figures"
    figures_png_dir = output_dir / "figures_png"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_dir / "session_metadata.csv", index=False)
    ratings.to_csv(output_dir / "ratings_long.csv", index=False)
    wide_ratings.to_csv(output_dir / "ratings_wide.csv", index=False)
    descriptives.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    distribution_counts.to_csv(output_dir / "distribution_counts.csv", index=False)
    correlations.to_csv(output_dir / "pairwise_correlations.csv", index=False)

    static_images_written = False

    distribution_png = write_plotly_figure(
        make_distribution_figure(
            distribution_counts,
            descriptives,
            dataset_label,
            metadata["session_uid"].nunique(),
            rating_label,
        ),
        figures_dir / "rating_distribution.html",
        png_dir=figures_png_dir,
        write_png=not html_only,
    )
    static_images_written = static_images_written or distribution_png is not None

    heatmap_png = write_plotly_figure(
        make_spearman_heatmap_figure(wide_ratings, dataset_label, rating_label),
        figures_dir / "spearman_correlation_heatmaps.html",
        png_dir=figures_png_dir,
        write_png=not html_only,
    )
    static_images_written = static_images_written or heatmap_png is not None

    for model in MODEL_ORDER:
        pair_png = write_plotly_figure(
            make_pairwise_scatter_figure(wide_ratings, dataset_label, model, rating_label),
            figures_dir / f"rating_pairs_{model}.html",
            png_dir=figures_png_dir,
            write_png=not html_only,
        )
        static_images_written = static_images_written or pair_png is not None

    novelty_pair_png = write_plotly_figure(
        make_novelty_focus_scatter_figure(wide_ratings, dataset_label, rating_label),
        figures_dir / "rating_pairs_novelty_x_all_models.html",
        png_dir=figures_png_dir,
        write_png=not html_only,
    )
    static_images_written = static_images_written or novelty_pair_png is not None

    novelty_line_png = write_plotly_figure(
        make_novelty_focus_line_figure(wide_ratings, dataset_label, rating_label),
        figures_dir / "rating_pairs_novelty_x_lines_all_models.html",
        png_dir=figures_png_dir,
        write_png=not html_only,
    )
    static_images_written = static_images_written or novelty_line_png is not None

    summary = build_summary_markdown(
        dataset_label=dataset_label,
        input_path=input_path,
        output_dir=output_dir,
        metadata=metadata,
        ratings=ratings,
        descriptives=descriptives,
        correlations=correlations,
        static_images_written=static_images_written,
        html_only=html_only,
        rating_label=rating_label,
        rating_source=rating_source,
    )
    (output_dir / "analysis_summary.md").write_text(summary)

    return {
        "dataset_label": dataset_label,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "num_sessions": int(metadata["session_uid"].nunique()),
        "num_ratings": int(len(ratings)),
        "num_participants": int(metadata["participant_id"].replace("", np.nan).nunique(dropna=True)),
        "static_images_written": static_images_written,
        "html_only": html_only,
        "rating_source": rating_source,
    }


def write_condition_comparison_outputs(
    input_paths: list[Path],
    run_output_dir: Path,
    *,
    html_only: bool,
    rating_source: str,
) -> dict | None:
    session_records = []
    metadata_frames = []
    for input_path in input_paths:
        file_records, file_metadata = load_unique_sessions(input_path)
        session_records.extend(file_records)
        metadata_frames.append(file_metadata)

    if not metadata_frames:
        return None

    metadata = pd.concat(metadata_frames, ignore_index=True)
    conditions = {
        str(value).strip()
        for value in metadata["interaction_condition"].dropna().tolist()
        if str(value).strip()
    }
    if not {"AIH", "HAI"}.issubset(conditions):
        return None

    rating_config = RATING_SOURCE_CONFIG[rating_source]
    rating_label = rating_config["label"]
    if rating_source == "human":
        ratings = build_long_ratings(session_records)
    else:
        ratings = build_long_ratings_from_field(
            session_records,
            rating_field=rating_config["field"],
        )
    wide_ratings = build_wide_ratings(ratings)

    output_dir = run_output_dir / "condition_comparison"
    figures_dir = output_dir / "figures"
    figures_png_dir = output_dir / "figures_png"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_dir / "session_metadata.csv", index=False)
    ratings.to_csv(output_dir / "ratings_long.csv", index=False)
    wide_ratings.to_csv(output_dir / "ratings_wide.csv", index=False)

    line_png = write_plotly_figure(
        make_condition_comparison_novelty_line_figure(wide_ratings, rating_label),
        figures_dir / "rating_pairs_novelty_x_lines_all_models.html",
        png_dir=figures_png_dir,
        write_png=not html_only,
    )

    return {
        "dataset_label": "condition_comparison",
        "output_dir": str(output_dir),
        "num_sessions": int(metadata["session_uid"].nunique()),
        "num_ratings": int(len(ratings)),
        "num_participants": int(metadata["participant_id"].replace("", np.nan).nunique(dropna=True)),
        "static_images_written": line_png is not None,
        "html_only": html_only,
        "rating_source": rating_source,
    }


def write_condition_source_comparison_outputs(
    input_paths: list[Path],
    run_output_dir: Path,
    *,
    html_only: bool,
) -> dict | None:
    session_records = []
    metadata_frames = []
    for input_path in input_paths:
        file_records, file_metadata = load_unique_sessions(input_path)
        session_records.extend(file_records)
        metadata_frames.append(file_metadata)

    if not metadata_frames:
        return None

    metadata = pd.concat(metadata_frames, ignore_index=True)
    conditions = {
        str(value).strip()
        for value in metadata["interaction_condition"].dropna().tolist()
        if str(value).strip()
    }
    if not {"AIH", "HAI"}.issubset(conditions):
        return None

    try:
        human_ratings = build_long_ratings(session_records)
        llm_judge_ratings = build_long_ratings_from_field(
            session_records,
            rating_field=RATING_SOURCE_CONFIG["llm_judge"]["field"],
        )
    except ValueError:
        return None

    output_dir = run_output_dir / "condition_source_comparison"
    figures_dir = output_dir / "figures"
    figures_png_dir = output_dir / "figures_png"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_dir / "session_metadata.csv", index=False)
    human_ratings.to_csv(output_dir / "ratings_long_human.csv", index=False)
    llm_judge_ratings.to_csv(output_dir / "ratings_long_llm_judge.csv", index=False)
    combined_ratings = pd.concat(
        [
            human_ratings.assign(rating_source="human"),
            llm_judge_ratings.assign(rating_source="llm_judge"),
        ],
        ignore_index=True,
    )
    combined_summary = compute_condition_source_descriptives(combined_ratings)
    combined_ratings.to_csv(output_dir / "ratings_long_human_vs_llm_judge.csv", index=False)
    combined_summary.to_csv(output_dir / "condition_source_descriptive_statistics.csv", index=False)

    combined_png = write_plotly_figure(
        make_condition_source_comparison_novelty_line_figure(
            {
                "human": human_ratings,
                "llm_judge": llm_judge_ratings,
            }
        ),
        figures_dir / "rating_pairs_novelty_x_lines_all_models_human_vs_llm_judge.html",
        png_dir=figures_png_dir,
        write_png=not html_only,
    )
    mean_profile_png = write_plotly_figure(
        make_condition_source_mean_profile_figure(combined_summary),
        figures_dir / "condition_source_mean_profiles.html",
        png_dir=figures_png_dir,
        write_png=not html_only,
    )
    heatmap_png = write_plotly_figure(
        make_condition_source_heatmap_figure(combined_summary),
        figures_dir / "condition_source_mean_heatmaps.html",
        png_dir=figures_png_dir,
        write_png=not html_only,
    )

    return {
        "dataset_label": "condition_source_comparison",
        "output_dir": str(output_dir),
        "num_sessions": int(metadata["session_uid"].nunique()),
        "num_ratings": int(len(human_ratings) + len(llm_judge_ratings)),
        "num_participants": int(metadata["participant_id"].replace("", np.nan).nunique(dropna=True)),
        "static_images_written": any(
            item is not None for item in [combined_png, mean_profile_png, heatmap_png]
        ),
        "html_only": html_only,
        "rating_source": "human_and_llm_judge",
    }


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = args.output_root / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        analyze_one_file(
            input_path,
            run_output_dir,
            html_only=args.html_only,
            rating_source=args.rating_source,
        )
        for input_path in args.inputs
    ]
    comparison_result = write_condition_comparison_outputs(
        args.inputs,
        run_output_dir,
        html_only=args.html_only,
        rating_source=args.rating_source,
    )
    if comparison_result is not None:
        results.append(comparison_result)
    condition_source_result = write_condition_source_comparison_outputs(
        args.inputs,
        run_output_dir,
        html_only=args.html_only,
    )
    if condition_source_result is not None:
        results.append(condition_source_result)

    manifest = {
        "timestamp": timestamp,
        "output_root": str(run_output_dir),
        "inputs": [str(path) for path in args.inputs],
        "rating_source": args.rating_source,
        "results": results,
    }
    (run_output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote Plotly AI suggestion analysis to {run_output_dir}")
    for result in results:
        print(
            f"- {result['dataset_label']}: "
            f"{result['num_sessions']} sessions, "
            f"{result['num_participants']} participants, "
            f"{result['num_ratings']} {result['rating_source']} rating rows"
        )


if __name__ == "__main__":
    main()
