from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.counterfactual_study_analysis import (
    LEGACY_MODEL_IDX,
    LEGACY_MODEL_ORDER,
    THRESH_MINILM,
    THRESH_TFIDF,
    prepare_counterfactual_dataframe,
)


MODEL_ORDER = ["LLM-B", "LLM-CF", "LLM-CT"]
MODEL_LABELS = {
    "LLM-B": "Baseline (B)",
    "LLM-CF": "CA-Within (CF)",
    "LLM-CT": "CA-Expand (CT)",
}
MODEL_SHORT = {"LLM-B": "B", "LLM-CF": "CF", "LLM-CT": "CT"}
QUAD_ORDER = [
    "Direct adoption (H/H)",
    "Surface mimicry (H/L)",
    "Semantic adoption (L/H)",
    "Independent (L/L)",
]
QUAD_COLORS = {
    "Direct adoption (H/H)": "#185FA5",
    "Surface mimicry (H/L)": "#BA7517",
    "Semantic adoption (L/H)": "#0F6E56",
    "Independent (L/L)": "#888780",
}
COND_COLORS = {"AIH": "#185FA5", "HAI": "#0F6E56"}
MODEL_COLORS = {"LLM-B": "#888780", "LLM-CF": "#0F6E56", "LLM-CT": "#993C1D"}
LAYOUT = dict(
    font=dict(family="Arial, sans-serif", size=13, color="#2C2C2A"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=40, t=90, b=60),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-only", action="store_true")
    return parser.parse_args()


def styled(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14)), **LAYOUT)
    fig.update_xaxes(showgrid=False, linecolor="#D3D1C7", linewidth=1)
    fig.update_yaxes(gridcolor="#EBEBEB", linecolor="#D3D1C7", linewidth=1)
    return fig


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_plotly_figure(fig: go.Figure, output_path: Path, html_only: bool) -> str | None:
    ensure_dir(output_path.parent)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    if html_only:
        return None
    png_path = output_path.with_suffix(".png")
    try:
        fig.write_image(str(png_path))
    except Exception:
        return None
    return str(png_path)


def sig_label(p_value: float) -> str:
    if pd.isna(p_value):
        return "na"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def safe_pearsonr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 2:
        return np.nan, np.nan
    if pair["x"].nunique(dropna=True) < 2 or pair["y"].nunique(dropna=True) < 2:
        return np.nan, np.nan
    r, p = stats.pearsonr(pair["x"], pair["y"])
    return float(r), float(p)


def safe_ttest_ind(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_clean = pd.Series(x).dropna()
    y_clean = pd.Series(y).dropna()
    if len(x_clean) < 2 or len(y_clean) < 2:
        return np.nan, np.nan
    if x_clean.nunique(dropna=True) < 2 or y_clean.nunique(dropna=True) < 2:
        return np.nan, np.nan
    stat, p_value = stats.ttest_ind(x_clean, y_clean, nan_policy="omit")
    return float(stat), float(p_value)


def safe_pointbiserialr(binary: pd.Series, values: pd.Series) -> tuple[float, float]:
    pair = pd.DataFrame({"binary": binary, "values": values}).dropna()
    if len(pair) < 2:
        return np.nan, np.nan
    if pair["binary"].nunique(dropna=True) < 2 or pair["values"].nunique(dropna=True) < 2:
        return np.nan, np.nan
    r, p = stats.pointbiserialr(pair["binary"].astype(int), pair["values"])
    return float(r), float(p)


def load_prepared_df(input_csv: Path) -> pd.DataFrame:
    return prepare_counterfactual_dataframe(pd.read_csv(input_csv))


def build_session_df(df: pd.DataFrame) -> pd.DataFrame:
    session_df = df.drop_duplicates(subset=["participant_id", "session", "interactionCondition"]).copy()
    for model in MODEL_ORDER:
        idx = LEGACY_MODEL_IDX[model]
        session_df[f"fd_tfidf_{model}"] = session_df["fd_score"].apply(
            lambda matrix: float(matrix[0, idx]) if matrix is not None and matrix.shape[0] >= 1 else np.nan
        )
        session_df[f"fd_minilm_{model}"] = session_df["fd_score"].apply(
            lambda matrix: float(matrix[1, idx]) if matrix is not None and matrix.shape[0] >= 2 else np.nan
        )
        session_df[f"fc_tfidf_max_{model}"] = session_df["fc_tfidf"].apply(
            lambda matrix: float(np.nanmax(matrix[:, idx])) if matrix is not None and matrix.size else np.nan
        )
        session_df[f"fc_minilm_max_{model}"] = session_df["fc_minilm"].apply(
            lambda matrix: float(np.nanmax(matrix[:, idx])) if matrix is not None and matrix.size else np.nan
        )

    session_df["n_candidates"] = session_df["fc_minilm"].apply(
        lambda matrix: int(matrix.shape[0]) if matrix is not None and matrix.ndim == 2 else 0
    )
    session_df["n_novel_minilm"] = session_df["fc_minilm"].apply(
        lambda matrix: int((np.nanmax(matrix, axis=1) < THRESH_MINILM).sum())
        if matrix is not None and matrix.size
        else 0
    )
    session_df["n_novel_tfidf"] = session_df["fc_tfidf"].apply(
        lambda matrix: int((np.nanmax(matrix, axis=1) < THRESH_TFIDF).sum())
        if matrix is not None and matrix.size
        else 0
    )
    session_df["n_novel_both"] = session_df.apply(
        lambda row: int(
            (
                (np.nanmax(row["fc_tfidf"], axis=1) < THRESH_TFIDF)
                & (np.nanmax(row["fc_minilm"], axis=1) < THRESH_MINILM)
            ).sum()
        )
        if row["fc_tfidf"] is not None and row["fc_minilm"] is not None and row["fc_tfidf"].size and row["fc_minilm"].size
        else 0,
        axis=1,
    )
    return session_df


def make_threshold_distribution_figure(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    t_vals = df["own_fc_tfidf"].dropna().to_numpy()
    m_vals = df["own_fc_minilm"].dropna().to_numpy()
    pct_table = pd.DataFrame(
        {
            "Percentile": [10, 25, 50, 75, 90, 95],
            "TF-IDF": [np.percentile(t_vals, p) for p in [10, 25, 50, 75, 90, 95]],
            "MiniLM": [np.percentile(m_vals, p) for p in [10, 25, 50, 75, 90, 95]],
        }
    )
    thresh_check = pd.DataFrame(
        [
            {
                "Threshold": threshold,
                "TF-IDF % above": round((t_vals >= threshold).mean() * 100, 1),
                "MiniLM % above": round((m_vals >= threshold).mean() * 100, 1),
            }
            for threshold in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        ]
    )
    thresh_check["Matched?"] = thresh_check.apply(
        lambda row: "*** MATCH ***"
        if abs(row["TF-IDF % above"] - row["MiniLM % above"]) < 1.0
        else "",
        axis=1,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "TF-IDF distribution (own-model, FC max)",
            "MiniLM distribution (own-model, FC max)",
        ],
    )
    for col_n, (vals, label, color, threshold) in enumerate(
        [
            (t_vals, "TF-IDF", "#BA7517", THRESH_TFIDF),
            (m_vals, "MiniLM", "#185FA5", THRESH_MINILM),
        ],
        1,
    ):
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=40, marker_color=color, opacity=0.75, name=label, showlegend=False),
            row=1,
            col=col_n,
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="#E24B4A", line_width=2, row=1, col=col_n)
        pct_above = (vals >= threshold).mean() * 100
        axis_suffix = "" if col_n == 1 else str(col_n)
        fig.add_annotation(
            x=0.78,
            y=0.88,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"<b>{pct_above:.1f}%</b> above {threshold}",
            showarrow=False,
            font=dict(size=12, color="#E24B4A"),
            bgcolor="rgba(255,255,255,0.85)",
        )
    fig.update_xaxes(title_text="Similarity score")
    fig.update_yaxes(title_text="Count", col=1)
    fig.update_layout(
        title=dict(
            text=(
                f"0b — TF-IDF vs MiniLM score distributions (thresholds: TF-IDF={THRESH_TFIDF}, MiniLM={THRESH_MINILM})<br>"
                f"<sup>Red dashed line = threshold. Above-threshold shares: "
                f"{(t_vals >= THRESH_TFIDF).mean() * 100:.1f}% / {(m_vals >= THRESH_MINILM).mean() * 100:.1f}%.</sup>"
            ),
            font=dict(size=14),
        ),
        font=dict(family="Arial, sans-serif", size=13, color="#2C2C2A"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=100, b=60),
    )
    fig.update_xaxes(showgrid=False, linecolor="#D3D1C7", linewidth=1)
    fig.update_yaxes(gridcolor="#EBEBEB", linecolor="#D3D1C7", linewidth=1)
    return fig, pct_table, thresh_check


def make_quadrant_distribution_table(df: pd.DataFrame, quadrant_col: str) -> pd.DataFrame:
    summary = df.groupby(["interactionCondition", quadrant_col]).size().reset_index(name="count")
    summary["pct"] = summary.groupby("interactionCondition")["count"].transform(lambda values: values / values.sum() * 100)
    return summary


def make_a1_figure(quad_fc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for quadrant in QUAD_ORDER:
        row = quad_fc[quad_fc["fc_quadrant"] == quadrant]
        fig.add_trace(
            go.Bar(
                name=quadrant,
                x=row["interactionCondition"],
                y=row["pct"],
                marker_color=QUAD_COLORS[quadrant],
                marker_line_width=0,
                text=row["pct"].round(1).astype(str) + "%",
                textposition="outside",
                textfont_size=10,
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="% of suggestions", range=[0, 60]),
        xaxis_title="Condition",
        legend=dict(title="Quadrant", orientation="h", y=1.28, x=0.5, xanchor="center"),
    )
    return styled(
        fig,
        "A1 — Quadrant distribution: Final Candidates (L1) by condition<br>"
        f"<sup>TF-IDF ≥ {THRESH_TFIDF} = lexical high | MiniLM ≥ {THRESH_MINILM} = semantic high</sup>",
    )


def make_a2_figure(quad_fd: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for quadrant in QUAD_ORDER:
        row = quad_fd[quad_fd["fd_quadrant"] == quadrant]
        fig.add_trace(
            go.Bar(
                name=quadrant,
                x=row["interactionCondition"],
                y=row["pct"],
                marker_color=QUAD_COLORS[quadrant],
                marker_line_width=0,
                text=row["pct"].round(1).astype(str) + "%",
                textposition="outside",
                textfont_size=10,
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="% of suggestions", range=[0, 85]),
        xaxis_title="Condition",
        legend=dict(title="Quadrant", orientation="h", y=1.28, x=0.5, xanchor="center"),
    )
    return styled(
        fig,
        "A2 — Quadrant distribution: Final Decision (L2) by condition<br>"
        "<sup>What type of AI influence reached the participant's final chosen plan?</sup>",
    )


def make_a3_figure(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    data = df.copy()
    data["cond_model"] = data["interactionCondition"] + "-" + data["model"].map(MODEL_SHORT)
    summary = data.groupby(["cond_model", "fc_quadrant"]).size().reset_index(name="count")
    summary["pct"] = summary.groupby("cond_model")["count"].transform(lambda values: values / values.sum() * 100)
    group_order = ["AIH-B", "AIH-CF", "AIH-CT", "HAI-B", "HAI-CF", "HAI-CT"]

    fig = go.Figure()
    for quadrant in QUAD_ORDER:
        subset = summary[summary["fc_quadrant"] == quadrant].set_index("cond_model").reindex(group_order)
        fig.add_trace(
            go.Bar(
                name=quadrant,
                x=group_order,
                y=subset["pct"].fillna(0),
                marker_color=QUAD_COLORS[quadrant],
                marker_line_width=0,
                text=subset["pct"].fillna(0).round(1).astype(str) + "%",
                textposition="inside",
                textfont=dict(color="white", size=10),
            )
        )
    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="% of suggestions", range=[0, 102]),
        xaxis_title="Condition × Model",
        legend=dict(title="Quadrant", orientation="h", y=1.28, x=0.5, xanchor="center"),
    )
    fig = styled(
        fig,
        "A3 — Final candidate quadrant breakdown by condition × model (stacked 100%)",
    )
    return fig, summary


def make_a4_figure(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    rows = []
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["AIH — Final Decision", "HAI — Final Decision"],
        shared_yaxes=True,
    )
    for cond, col_n in [("AIH", 1), ("HAI", 2)]:
        subset = df[df["interactionCondition"] == cond]
        r_value, p_value = safe_pearsonr(subset["own_fd_tfidf"], subset["own_fd_minilm"])
        rows.append({"interactionCondition": cond, "pearson_r": r_value, "p_value": p_value, "n": int(len(subset))})
        for quadrant in QUAD_ORDER:
            quadrant_subset = subset[subset["fd_quadrant"] == quadrant]
            fig.add_trace(
                go.Scatter(
                    x=quadrant_subset["own_fd_tfidf"],
                    y=quadrant_subset["own_fd_minilm"],
                    mode="markers",
                    name=quadrant,
                    marker=dict(color=QUAD_COLORS[quadrant], size=5, opacity=0.55, line=dict(width=0.3, color="white")),
                    showlegend=(col_n == 1),
                    legendgroup=quadrant,
                    hovertemplate=f"TF-IDF: %{{x:.3f}}<br>MiniLM: %{{y:.3f}}<br>{quadrant}",
                ),
                row=1,
                col=col_n,
            )
        fig.add_hline(y=THRESH_MINILM, line_dash="dash", line_color="gray", line_width=1, row=1, col=col_n)
        fig.add_vline(x=THRESH_TFIDF, line_dash="dash", line_color="gray", line_width=1, row=1, col=col_n)
        axis_suffix = "" if col_n == 1 else str(col_n)
        fig.add_annotation(
            x=0.9,
            y=0.04,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"r = {r_value:.3f}" if pd.notna(r_value) else "r = na",
            showarrow=False,
            font=dict(size=12, color="#444"),
            bgcolor="rgba(255,255,255,0.8)",
        )
    fig.update_xaxes(title_text="TF-IDF similarity (lexical)", range=[-0.2, 1.05])
    fig.update_yaxes(title_text="MiniLM similarity (semantic)", range=[-0.15, 1.05])
    fig.update_layout(
        legend=dict(title="Quadrant", orientation="h", y=1.28, x=0.5, xanchor="center"),
        **LAYOUT,
        title=dict(
            text="A4 — TF-IDF vs MiniLM similarity at Final Decision, colored by quadrant",
            font=dict(size=14),
        ),
    )
    return fig, pd.DataFrame(rows)


def make_heatmap_df(df_in: pd.DataFrame, quad_col: str) -> pd.DataFrame:
    pivot = df_in.groupby(["cond_model", quad_col]).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    return pivot.reindex(columns=QUAD_ORDER, fill_value=0)


def make_a5_figure(df: pd.DataFrame) -> go.Figure:
    heatmap_input = df.copy()
    heatmap_input["cond_model"] = heatmap_input["interactionCondition"] + "-" + heatmap_input["model"].map(MODEL_SHORT)
    heatmap_fc = make_heatmap_df(heatmap_input, "fc_quadrant")
    heatmap_fd = make_heatmap_df(heatmap_input, "fd_quadrant")
    group_order = ["AIH-B", "AIH-CF", "AIH-CT", "HAI-B", "HAI-CF", "HAI-CT"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Final Candidates (L1)", "Final Decision (L2)"],
        horizontal_spacing=0.14,
    )
    for col_n, heatmap in [(1, heatmap_fc), (2, heatmap_fd)]:
        z_values = heatmap.reindex(group_order).values
        text_values = [[f"{value:.1f}%" for value in row] for row in z_values]
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=[quadrant.replace(" ", "<br>") for quadrant in QUAD_ORDER],
                y=group_order,
                text=text_values,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=(col_n == 2),
                colorbar=dict(title="%", x=1.02, len=0.9),
                zmin=0,
                zmax=70,
            ),
            row=1,
            col=col_n,
        )
    fig.update_layout(
        title=dict(
            text="A5 — Quadrant heatmap by condition × model<br><sup>Cell values = % of suggestions in each quadrant</sup>",
            font=dict(size=14),
        ),
        **LAYOUT,
    )
    return fig


def make_b1_figure(session_df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    aih = session_df[session_df["interactionCondition"] == "AIH"]
    hai = session_df[session_df["interactionCondition"] == "HAI"]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["TF-IDF similarity (lexical)", "MiniLM similarity (semantic)"],
        shared_yaxes=False,
    )
    stats_rows = []
    panels = [
        ("TFIDF", {model: f"fd_tfidf_{model}" for model in MODEL_ORDER}, 1),
        ("MiniLM", {model: f"fd_minilm_{model}" for model in MODEL_ORDER}, 2),
    ]
    for metric_name, column_map, col_n in panels:
        for condition, subset, color in [("AIH", aih, COND_COLORS["AIH"]), ("HAI", hai, COND_COLORS["HAI"])]:
            means = [subset[column_map[model]].mean() for model in MODEL_ORDER]
            ses = [subset[column_map[model]].sem() for model in MODEL_ORDER]
            figure.add_trace(
                go.Bar(
                    name=condition,
                    x=[MODEL_LABELS[model] for model in MODEL_ORDER],
                    y=means,
                    error_y=dict(type="data", array=ses, visible=True, color="#444", thickness=1.5, width=4),
                    marker_color=color,
                    marker_line_width=0,
                    showlegend=(col_n == 1),
                    legendgroup=condition,
                ),
                row=1,
                col=col_n,
            )
        for model in MODEL_ORDER:
            stat, p_value = safe_ttest_ind(aih[column_map[model]], hai[column_map[model]])
            stats_rows.append(
                {
                    "metric": metric_name,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "AIH_mean": float(aih[column_map[model]].mean()),
                    "HAI_mean": float(hai[column_map[model]].mean()),
                    "t_stat": stat,
                    "p_value": p_value,
                    "sig": sig_label(p_value),
                }
            )
    figure.update_layout(
        barmode="group",
        legend=dict(title="Condition", orientation="h", y=1.12, x=0.5, xanchor="center"),
        **LAYOUT,
        title=dict(
            text="B1 — Final decision similarity: TF-IDF vs MiniLM by condition<br><sup>Error bars = ±1 SE</sup>",
            font=dict(size=14),
        ),
    )
    figure.update_yaxes(title_text="TF-IDF similarity", row=1, col=1)
    figure.update_yaxes(title_text="MiniLM similarity", row=1, col=2)
    return figure, pd.DataFrame(stats_rows)


def make_c1_figure(session_df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    aih = session_df[session_df["interactionCondition"] == "AIH"]
    hai = session_df[session_df["interactionCondition"] == "HAI"]
    composition = pd.DataFrame(
        {
            "Condition": ["AIH", "HAI"],
            "Novel by MiniLM only": [aih["n_novel_minilm"].mean(), hai["n_novel_minilm"].mean()],
            "Novel by TFIDF only": [aih["n_novel_tfidf"].mean(), hai["n_novel_tfidf"].mean()],
            "Novel by both": [aih["n_novel_both"].mean(), hai["n_novel_both"].mean()],
            "Total candidates": [aih["n_candidates"].mean(), hai["n_candidates"].mean()],
        }
    )
    tests = pd.DataFrame(
        [
            {
                "metric": "n_novel_both",
                "AIH_mean": float(aih["n_novel_both"].mean()),
                "HAI_mean": float(hai["n_novel_both"].mean()),
                "t_stat": safe_ttest_ind(aih["n_novel_both"], hai["n_novel_both"])[0],
                "p_value": safe_ttest_ind(aih["n_novel_both"], hai["n_novel_both"])[1],
            },
            {
                "metric": "n_novel_minilm",
                "AIH_mean": float(aih["n_novel_minilm"].mean()),
                "HAI_mean": float(hai["n_novel_minilm"].mean()),
                "t_stat": safe_ttest_ind(aih["n_novel_minilm"], hai["n_novel_minilm"])[0],
                "p_value": safe_ttest_ind(aih["n_novel_minilm"], hai["n_novel_minilm"])[1],
            },
            {
                "metric": "n_novel_tfidf",
                "AIH_mean": float(aih["n_novel_tfidf"].mean()),
                "HAI_mean": float(hai["n_novel_tfidf"].mean()),
                "t_stat": safe_ttest_ind(aih["n_novel_tfidf"], hai["n_novel_tfidf"])[0],
                "p_value": safe_ttest_ind(aih["n_novel_tfidf"], hai["n_novel_tfidf"])[1],
            },
        ]
    )
    tests["sig"] = tests["p_value"].apply(sig_label)

    figure = go.Figure()
    for column, color in zip(
        ["Novel by MiniLM only", "Novel by TFIDF only", "Novel by both"],
        ["#185FA5", "#BA7517", "#0F6E56"],
    ):
        figure.add_trace(
            go.Bar(
                name=column,
                x=composition["Condition"],
                y=composition[column],
                marker_color=color,
                marker_line_width=0,
                text=composition[column].round(2),
                textposition="inside",
                textfont=dict(color="white", size=11),
            )
        )
    for _, row in composition.iterrows():
        figure.add_annotation(
            x=row["Condition"],
            y=row["Total candidates"] + 0.08,
            text=f'total={row["Total candidates"]:.2f}',
            showarrow=False,
            font=dict(size=11, color="#444"),
        )
    figure.update_layout(
        barmode="overlay",
        yaxis=dict(title="Mean novel candidates per session"),
        xaxis_title="Condition",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    return styled(
        figure,
        'C1 — Novel candidates by metric: how many plans escape AI influence?<br><sup>"Novel by both" = low TF-IDF and low MiniLM</sup>',
    ), composition, tests


def make_d1_figure(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    rows = []
    for condition in ["AIH", "HAI"]:
        subset = df[df["interactionCondition"] == condition]
        for model in MODEL_ORDER:
            model_subset = subset[subset["model"] == model]
            rows.append(
                dict(
                    condition=condition,
                    model=model,
                    model_label=MODEL_LABELS[model],
                    rate_minilm=float(model_subset["L1_adopted_minilm"].mean()),
                    rate_tfidf=float(model_subset["L1_adopted_tfidf"].mean()),
                    rate_both=float(model_subset["L1_adopted_both"].mean()),
                    n=int(len(model_subset)),
                    adopted_minilm=int(model_subset["L1_adopted_minilm"].sum()),
                    adopted_tfidf=int(model_subset["L1_adopted_tfidf"].sum()),
                    adopted_both=int(model_subset["L1_adopted_both"].sum()),
                )
            )
    rates = pd.DataFrame(rows)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["MiniLM ≥ 0.6 (semantic)", "TF-IDF ≥ 0.6 (lexical)", "Both"],
        shared_yaxes=True,
    )
    for col_n, rate_col in enumerate(["rate_minilm", "rate_tfidf", "rate_both"], 1):
        for condition, color in [("AIH", COND_COLORS["AIH"]), ("HAI", COND_COLORS["HAI"])]:
            subset = rates[rates["condition"] == condition]
            figure.add_trace(
                go.Bar(
                    name=condition,
                    x=subset["model_label"],
                    y=subset[rate_col],
                    marker_color=color,
                    marker_line_width=0,
                    showlegend=(col_n == 1),
                    legendgroup=condition,
                    text=(subset[rate_col] * 100).round(1).astype(str) + "%",
                    textposition="outside",
                    textfont_size=9,
                ),
                row=1,
                col=col_n,
            )
    figure.update_yaxes(tickformat=".0%", title_text="L1 adoption rate", row=1, col=1)
    figure.update_layout(
        barmode="group",
        legend=dict(title="Condition", orientation="h", y=1.12, x=0.5, xanchor="center"),
        **LAYOUT,
        title=dict(
            text="D1 — L1 adoption rates: MiniLM vs TF-IDF vs both metrics<br><sup>\"Both\" = same words and same meaning</sup>",
            font=dict(size=14),
        ),
    )
    return figure, rates


def make_e1_figure(session_df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    hai_df = session_df[session_df["interactionCondition"] == "HAI"].copy()
    initial_rows = []
    for _, row in hai_df.iterrows():
        matrix_tfidf = row["ip_tfidf"]
        matrix_minilm = row["ip_minilm"]
        if matrix_tfidf is None or matrix_minilm is None:
            continue
        for plan_index in range(matrix_tfidf.shape[0]):
            for model, idx in LEGACY_MODEL_IDX.items():
                tfidf_value = float(matrix_tfidf[plan_index, idx])
                minilm_value = float(matrix_minilm[plan_index, idx])
                initial_rows.append(
                    {
                        "participant_id": row["participant_id"],
                        "session": row["session"],
                        "model": model,
                        "stage": "Initial (pre-AI)",
                        "tfidf": tfidf_value,
                        "minilm": minilm_value,
                        "quadrant": quadrant(tfidf_value, minilm_value),
                    }
                )
    candidate_rows = []
    for _, row in hai_df.iterrows():
        matrix_tfidf = row["fc_tfidf"]
        matrix_minilm = row["fc_minilm"]
        if matrix_tfidf is None or matrix_minilm is None:
            continue
        for candidate_index in range(matrix_tfidf.shape[0]):
            for model, idx in LEGACY_MODEL_IDX.items():
                tfidf_value = float(matrix_tfidf[candidate_index, idx])
                minilm_value = float(matrix_minilm[candidate_index, idx])
                candidate_rows.append(
                    {
                        "participant_id": row["participant_id"],
                        "session": row["session"],
                        "model": model,
                        "stage": "Final candidate (post-AI)",
                        "tfidf": tfidf_value,
                        "minilm": minilm_value,
                        "quadrant": quadrant(tfidf_value, minilm_value),
                    }
                )
    combined = pd.DataFrame(initial_rows + candidate_rows)
    stage_quad = combined.groupby(["stage", "quadrant"]).size().reset_index(name="count")
    stage_quad["pct"] = stage_quad.groupby("stage")["count"].transform(lambda values: values / values.sum() * 100)
    figure = go.Figure()
    for quadrant_name in QUAD_ORDER:
        subset = stage_quad[stage_quad["quadrant"] == quadrant_name]
        figure.add_trace(
            go.Bar(
                name=quadrant_name,
                x=subset["stage"],
                y=subset["pct"],
                marker_color=QUAD_COLORS[quadrant_name],
                marker_line_width=0,
                text=subset["pct"].round(1).astype(str) + "%",
                textposition="outside",
                textfont_size=10,
            )
        )
    figure.update_layout(
        barmode="group",
        yaxis=dict(title="% of plans/candidates", range=[0, 80]),
        xaxis_title="Stage",
        legend=dict(title="Quadrant", orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    return styled(
        figure,
        "E1 — HAI quadrant shift: initial plans (pre-AI) to final candidates (post-AI)",
    ), stage_quad


def quadrant(tfidf_score: float, minilm_score: float) -> str:
    if tfidf_score >= THRESH_TFIDF and minilm_score >= THRESH_MINILM:
        return "Direct adoption (H/H)"
    if tfidf_score >= THRESH_TFIDF and minilm_score < THRESH_MINILM:
        return "Surface mimicry (H/L)"
    if tfidf_score < THRESH_TFIDF and minilm_score >= THRESH_MINILM:
        return "Semantic adoption (L/H)"
    return "Independent (L/L)"


def make_f1_figure(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame, float]:
    tfidf_range = np.arange(0.2, 0.8, 0.05)
    minilm_range = np.arange(0.3, 0.85, 0.05)
    rows = []
    hai_df = df[df["interactionCondition"] == "HAI"]
    aih_df = df[df["interactionCondition"] == "AIH"]
    for tfidf_threshold in tfidf_range:
        for minilm_threshold in minilm_range:
            hai_rate = ((hai_df["own_fc_tfidf"] >= tfidf_threshold) & (hai_df["own_fc_minilm"] >= minilm_threshold)).mean()
            aih_rate = ((aih_df["own_fc_tfidf"] >= tfidf_threshold) & (aih_df["own_fc_minilm"] >= minilm_threshold)).mean()
            rows.append(
                {
                    "thresh_tfidf": round(tfidf_threshold, 2),
                    "thresh_minilm": round(minilm_threshold, 2),
                    "hai_direct_rate": float(hai_rate),
                    "aih_direct_rate": float(aih_rate),
                    "gap": float(hai_rate - aih_rate),
                }
            )
    sweep_df = pd.DataFrame(rows)
    pivot = sweep_df.pivot(index="thresh_tfidf", columns="thresh_minilm", values="gap") * 100
    figure = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[f"{value:.2f}" for value in pivot.columns],
            y=[f"{value:.2f}" for value in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}",
            colorbar=dict(title="HAI-AIH<br>gap (pp)"),
        )
    )
    figure.add_shape(
        type="rect",
        x0=str(round(THRESH_MINILM, 2)),
        x1=str(round(THRESH_MINILM + 0.05, 2)),
        y0=str(round(THRESH_TFIDF, 2)),
        y1=str(round(THRESH_TFIDF + 0.05, 2)),
        line=dict(color="black", width=2),
    )
    figure.update_layout(
        xaxis_title="MiniLM threshold (semantic)",
        yaxis_title="TF-IDF threshold (lexical)",
        **LAYOUT,
        title=dict(
            text="F1 — Sensitivity: HAI - AIH direct adoption gap across threshold combinations",
            font=dict(size=13),
        ),
    )
    positive_pct = float((pivot.values > 0).mean() * 100)
    return figure, sweep_df, positive_pct


def make_g1_figure(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    hai_df = df[df["interactionCondition"] == "HAI"].copy()
    rating_cols = ["clarity", "feasibility", "goal_alignment", "novelty"]
    rows = []
    for adopt_col, adopt_label in [
        ("L1_adopted_minilm", "MiniLM L1"),
        ("L1_adopted_tfidf", "TFIDF L1"),
        ("L1_adopted_both", "Both L1"),
    ]:
        for rating_col in rating_cols:
            r_value, p_value = safe_pointbiserialr(hai_df[adopt_col], hai_df[rating_col])
            valid = hai_df[[adopt_col, rating_col]].dropna()
            rows.append(
                {
                    "adopt": adopt_label,
                    "rating": rating_col.replace("_", " ").title(),
                    "r": r_value,
                    "p": p_value,
                    "sig": "*" if pd.notna(p_value) and p_value < 0.05 else "ns",
                    "n": int(len(valid)),
                }
            )
    corr_df = pd.DataFrame(rows)
    figure = px.bar(
        corr_df,
        x="rating",
        y="r",
        color="adopt",
        barmode="group",
        color_discrete_map={"MiniLM L1": "#185FA5", "TFIDF L1": "#BA7517", "Both L1": "#0F6E56"},
        labels={"r": "Point-biserial r", "rating": "Utility dimension", "adopt": "Adoption metric"},
        text=corr_df.apply(lambda row: f"r={row['r']:.3f} {row['sig']}" if pd.notna(row["r"]) else "r=na", axis=1),
    )
    figure.add_hline(y=0, line_dash="solid", line_color="#444", line_width=0.8)
    figure.update_traces(textposition="outside", textfont_size=9)
    figure.update_layout(
        yaxis=dict(title="Point-biserial r", range=[-0.12, 0.16]),
        legend=dict(title="Adoption metric", orientation="h", y=1.28, x=0.5, xanchor="center"),
    )
    return styled(figure, "G1 — Utility ratings vs L1 adoption (HAI)"), corr_df


def make_g2_figure(session_df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    rows = []
    for _, row in session_df[session_df["interactionCondition"] == "HAI"].iterrows():
        matrix_tfidf = row["ip_tfidf"]
        matrix_minilm = row["ip_minilm"]
        if matrix_tfidf is None or matrix_minilm is None:
            continue
        for index in range(matrix_tfidf.shape[0]):
            rows.append(
                {
                    "max_tfidf": float(np.nanmax(matrix_tfidf[index])),
                    "max_minilm": float(np.nanmax(matrix_minilm[index])),
                }
            )
    novelty_df = pd.DataFrame(rows)
    novel_minilm = float((novelty_df["max_minilm"] < THRESH_MINILM).mean() * 100) if not novelty_df.empty else np.nan
    novel_tfidf = float((novelty_df["max_tfidf"] < THRESH_TFIDF).mean() * 100) if not novelty_df.empty else np.nan
    novel_both = float(
        ((novelty_df["max_tfidf"] < THRESH_TFIDF) & (novelty_df["max_minilm"] < THRESH_MINILM)).mean() * 100
    ) if not novelty_df.empty else np.nan
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["TF-IDF similarity distribution", "MiniLM similarity distribution"],
    )
    for col_n, (column, threshold, label) in enumerate(
        [("max_tfidf", THRESH_TFIDF, "TF-IDF"), ("max_minilm", THRESH_MINILM, "MiniLM")],
        1,
    ):
        figure.add_trace(
            go.Histogram(
                x=novelty_df[column],
                nbinsx=30,
                marker_color=COND_COLORS["HAI"],
                name=label,
                showlegend=False,
            ),
            row=1,
            col=col_n,
        )
        figure.add_vline(x=threshold, line_dash="dash", line_color="#E24B4A", line_width=1.5, row=1, col=col_n)
    novel_values = {"TF-IDF": novel_tfidf, "MiniLM": novel_minilm}
    for col_n, label in [(1, "TF-IDF"), (2, "MiniLM")]:
        axis_suffix = "" if col_n == 1 else str(col_n)
        figure.add_annotation(
            x=0.15,
            y=0.9,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"{novel_values[label]:.1f}% novel" if pd.notna(novel_values[label]) else "na",
            showarrow=False,
            font=dict(size=12, color="#0F6E56"),
            bgcolor="rgba(255,255,255,0.85)",
        )
    figure.update_xaxes(title_text="Max similarity to any AI suggestion")
    figure.update_yaxes(title_text="Count of plans", col=1)
    figure.update_layout(
        **LAYOUT,
        title=dict(text="G2 — HAI initial plan novelty: TF-IDF vs MiniLM", font=dict(size=14)),
    )
    stats_df = pd.DataFrame(
        [
            {
                "metric": "MiniLM",
                "threshold": THRESH_MINILM,
                "pct_novel": novel_minilm,
                "n_plans": int(len(novelty_df)),
            },
            {
                "metric": "TF-IDF",
                "threshold": THRESH_TFIDF,
                "pct_novel": novel_tfidf,
                "n_plans": int(len(novelty_df)),
            },
            {
                "metric": "Both",
                "threshold": np.nan,
                "pct_novel": novel_both,
                "n_plans": int(len(novelty_df)),
            },
        ]
    )
    return figure, stats_df


def build_summary_table(
    b1_stats: pd.DataFrame,
    composition_tests: pd.DataFrame,
    adoption_rates: pd.DataFrame,
    utility_corr: pd.DataFrame,
    novelty_stats: pd.DataFrame,
    sensitivity_positive_pct: float,
) -> pd.DataFrame:
    best_fd = b1_stats.sort_values("p_value", na_position="last").iloc[0] if not b1_stats.empty else None
    best_both = composition_tests[composition_tests["metric"] == "n_novel_both"]
    ct_rate = adoption_rates[(adoption_rates["condition"] == "HAI") & (adoption_rates["model"] == "LLM-CT")]
    b_rate = adoption_rates[(adoption_rates["condition"] == "HAI") & (adoption_rates["model"] == "LLM-B")]
    novelty_both = novelty_stats[novelty_stats["metric"] == "Both"]
    max_corr = utility_corr.iloc[utility_corr["r"].abs().fillna(-1).idxmax()] if not utility_corr.empty else None
    rows = [
        {
            "finding": "Final decision similarity",
            "result": (
                f"{best_fd['metric']} {best_fd['model_label']}: HAI={best_fd['HAI_mean']:.3f}, "
                f"AIH={best_fd['AIH_mean']:.3f}, p={best_fd['p_value']:.4f}"
            )
            if best_fd is not None and pd.notna(best_fd["p_value"])
            else "",
        },
        {
            "finding": "Novel candidate composition",
            "result": (
                f"Novel by both: HAI={best_both.iloc[0]['HAI_mean']:.3f}, AIH={best_both.iloc[0]['AIH_mean']:.3f}, "
                f"p={best_both.iloc[0]['p_value']:.4f}"
            )
            if not best_both.empty and pd.notna(best_both.iloc[0]["p_value"])
            else "",
        },
        {
            "finding": "HAI CT adoption vs baseline",
            "result": (
                f"CT both-rate={ct_rate.iloc[0]['rate_both']:.3f}, "
                f"B both-rate={b_rate.iloc[0]['rate_both']:.3f}"
            )
            if not ct_rate.empty and not b_rate.empty
            else "",
        },
        {
            "finding": "Sensitivity",
            "result": f"HAI direct-adoption gap is positive for {sensitivity_positive_pct:.1f}% of tested thresholds",
        },
        {
            "finding": "Utility vs adoption",
            "result": (
                f"Max |r| = {max_corr['r']:.3f} for {max_corr['rating']} vs {max_corr['adopt']} (p={max_corr['p']:.4f})"
            )
            if max_corr is not None and pd.notna(max_corr["r"])
            else "",
        },
        {
            "finding": "Initial plan novelty",
            "result": (
                f"{novelty_both.iloc[0]['pct_novel']:.1f}% novel by both metrics"
            )
            if not novelty_both.empty and pd.notna(novelty_both.iloc[0]["pct_novel"])
            else "",
        },
    ]
    return pd.DataFrame(rows)


def write_markdown_summary(
    output_path: Path,
    df: pd.DataFrame,
    session_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figure_paths: list[str],
) -> None:
    lines = [
        "# Counterfactual Study Analysis",
        "",
        f"- Rows: {len(df)}",
        f"- Sessions: {len(session_df)}",
        f"- Conditions: {df['interactionCondition'].value_counts().to_dict()}",
        f"- Models: {df['model'].value_counts().to_dict()}",
        "",
        "## Key Results",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"- {row['finding']}: {row['result']}")
    lines.extend(["", "## Figures", ""])
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path}`")
    output_path.write_text("\n".join(lines) + "\n")


def write_run_manifest(output_dir: Path, figure_paths: list[str], table_paths: list[str], html_only: bool) -> None:
    manifest = {
        "output_dir": str(output_dir),
        "figures": figure_paths,
        "tables": table_paths,
        "html_only": html_only,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    df = load_prepared_df(args.input_csv)
    session_df = build_session_df(df)

    figure_paths: list[str] = []
    table_paths: list[str] = []

    threshold_fig, percentile_table, threshold_table = make_threshold_distribution_figure(df)
    write_plotly_figure(threshold_fig, figures_dir / "threshold_distributions.html", args.html_only)
    figure_paths.append("figures/threshold_distributions.html")
    percentile_table.to_csv(tables_dir / "threshold_percentiles.csv", index=False)
    threshold_table.to_csv(tables_dir / "threshold_checks.csv", index=False)
    table_paths.extend(["tables/threshold_percentiles.csv", "tables/threshold_checks.csv"])

    quad_fc = make_quadrant_distribution_table(df, "fc_quadrant")
    quad_fd = make_quadrant_distribution_table(df, "fd_quadrant")
    write_plotly_figure(make_a1_figure(quad_fc), figures_dir / "quadrant_l1_by_condition.html", args.html_only)
    write_plotly_figure(make_a2_figure(quad_fd), figures_dir / "quadrant_l2_by_condition.html", args.html_only)
    figure_paths.extend(["figures/quadrant_l1_by_condition.html", "figures/quadrant_l2_by_condition.html"])
    quad_fc.to_csv(tables_dir / "quadrant_l1_by_condition.csv", index=False)
    quad_fd.to_csv(tables_dir / "quadrant_l2_by_condition.csv", index=False)
    table_paths.extend(["tables/quadrant_l1_by_condition.csv", "tables/quadrant_l2_by_condition.csv"])

    a3_fig, cond_model_fc = make_a3_figure(df)
    write_plotly_figure(a3_fig, figures_dir / "quadrant_l1_condition_model.html", args.html_only)
    figure_paths.append("figures/quadrant_l1_condition_model.html")
    cond_model_fc.to_csv(tables_dir / "quadrant_l1_condition_model.csv", index=False)
    table_paths.append("tables/quadrant_l1_condition_model.csv")

    a4_fig, a4_stats = make_a4_figure(df)
    write_plotly_figure(a4_fig, figures_dir / "final_decision_scatter.html", args.html_only)
    figure_paths.append("figures/final_decision_scatter.html")
    a4_stats.to_csv(tables_dir / "final_decision_scatter_stats.csv", index=False)
    table_paths.append("tables/final_decision_scatter_stats.csv")

    a5_fig = make_a5_figure(df)
    write_plotly_figure(a5_fig, figures_dir / "quadrant_heatmap.html", args.html_only)
    figure_paths.append("figures/quadrant_heatmap.html")

    b1_fig, b1_stats = make_b1_figure(session_df)
    write_plotly_figure(b1_fig, figures_dir / "final_decision_similarity.html", args.html_only)
    figure_paths.append("figures/final_decision_similarity.html")
    b1_stats.to_csv(tables_dir / "final_decision_similarity_stats.csv", index=False)
    table_paths.append("tables/final_decision_similarity_stats.csv")

    c1_fig, composition_table, composition_tests = make_c1_figure(session_df)
    write_plotly_figure(c1_fig, figures_dir / "candidate_composition.html", args.html_only)
    figure_paths.append("figures/candidate_composition.html")
    composition_table.to_csv(tables_dir / "candidate_composition.csv", index=False)
    composition_tests.to_csv(tables_dir / "candidate_composition_tests.csv", index=False)
    table_paths.extend(["tables/candidate_composition.csv", "tables/candidate_composition_tests.csv"])

    d1_fig, adoption_rates = make_d1_figure(df)
    write_plotly_figure(d1_fig, figures_dir / "l1_adoption_rates.html", args.html_only)
    figure_paths.append("figures/l1_adoption_rates.html")
    adoption_rates.to_csv(tables_dir / "l1_adoption_rates.csv", index=False)
    table_paths.append("tables/l1_adoption_rates.csv")

    e1_fig, stage_quad = make_e1_figure(session_df)
    write_plotly_figure(e1_fig, figures_dir / "hai_stage_shift.html", args.html_only)
    figure_paths.append("figures/hai_stage_shift.html")
    stage_quad.to_csv(tables_dir / "hai_stage_shift.csv", index=False)
    table_paths.append("tables/hai_stage_shift.csv")

    f1_fig, sweep_df, positive_pct = make_f1_figure(df)
    write_plotly_figure(f1_fig, figures_dir / "sensitivity_heatmap.html", args.html_only)
    figure_paths.append("figures/sensitivity_heatmap.html")
    sweep_df.to_csv(tables_dir / "sensitivity_gap_grid.csv", index=False)
    table_paths.append("tables/sensitivity_gap_grid.csv")

    g1_fig, utility_corr = make_g1_figure(df)
    write_plotly_figure(g1_fig, figures_dir / "utility_vs_adoption.html", args.html_only)
    figure_paths.append("figures/utility_vs_adoption.html")
    utility_corr.to_csv(tables_dir / "utility_vs_adoption.csv", index=False)
    table_paths.append("tables/utility_vs_adoption.csv")

    g2_fig, novelty_stats = make_g2_figure(session_df)
    write_plotly_figure(g2_fig, figures_dir / "initial_plan_novelty.html", args.html_only)
    figure_paths.append("figures/initial_plan_novelty.html")
    novelty_stats.to_csv(tables_dir / "initial_plan_novelty.csv", index=False)
    table_paths.append("tables/initial_plan_novelty.csv")

    summary_df = build_summary_table(b1_stats, composition_tests, adoption_rates, utility_corr, novelty_stats, positive_pct)
    summary_df.to_csv(tables_dir / "summary_findings.csv", index=False)
    table_paths.append("tables/summary_findings.csv")

    write_markdown_summary(output_dir / "analysis_summary.md", df, session_df, summary_df, figure_paths)
    write_run_manifest(output_dir, figure_paths, table_paths, args.html_only)

    print(f"Wrote counterfactual plots to {output_dir}")
    print(f"- Rows: {len(df)}")
    print(f"- Sessions: {len(session_df)}")
    print(f"- Figures: {len(figure_paths)}")
    print(f"- Tables: {len(table_paths)}")


if __name__ == "__main__":
    main()
