"""
Visualize automatic-evaluation outputs for the reproducibility package.

Reads the CSVs produced by `automatic_evaluation.py` and writes summary tables
plus a small set of publication-style figures.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
for import_root in (PROJECT_DIR, REPO_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from utils.helpers import PHASE_LABELS


DISPLAY_NAME_MAP = {
    "human_alt_staged_json": "Human",
    "LLMB_alt_staged_json": "LLM-B",
    "LLMB_alt_staged_json_raw": "LLM-B (raw)",
    "LLMC_alt_staged_json": "LLM-C",
    "LLMC_alt_staged_json_raw": "LLM-C (raw)",
    "LLMCF_alt_staged_json": "LLM-CF",
    "LLMCT_alt_staged_json": "LLM-CT",
    "LLMCT_alt_staged_json_raw": "LLM-CT (raw)",
}

CONDITION_ORDER = [
    "Human",
    "LLM-B",
    "LLM-CF",
    "LLM-C",
    "LLM-CT",
    "LLM-B (raw)",
    "LLM-C (raw)",
    "LLM-CT (raw)",
]

COLORS = {
    "Human": "#ff6b6b",
    "LLM-B": "#ffe66d",
    "LLM-CF": "#4ecdc4",
    "LLM-C": "#4ecdc4",
    "LLM-CT": "#1a535c",
    "LLM-B (raw)": "#f6bd60",
    "LLM-C (raw)": "#84dcc6",
    "LLM-CT (raw)": "#3d5a80",
}

STAGE_LABELS = [label.replace(" ", "\n") for label in PHASE_LABELS]


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty frame if the file has no rows."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def condition_sort_key(name: str) -> tuple:
    """Stable sort order for model conditions."""
    try:
        return (0, CONDITION_ORDER.index(name))
    except ValueError:
        return (1, name)


def first_nonempty(series: pd.Series) -> str:
    """Return the first non-empty string in a series."""
    if series.empty:
        return ""
    for value in series.astype(str):
        value = value.strip()
        if value and value.lower() != "nan":
            return value
    return ""


def add_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable session identifier to a frame."""
    df = df.copy()
    if df.empty:
        df["session_id"] = pd.Series(dtype=str)
        return df

    parts = []
    for col in ["source_file", "userId", "sessionNumber"]:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))

    if not parts:
        df["session_id"] = df.index.astype(str)
        return df

    session_id = parts[0]
    for extra in parts[1:]:
        session_id = session_id + "::" + extra
    df["session_id"] = session_id
    return df


def discover_condition_outputs(input_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all condition-level metrics/diversity CSVs from an output directory."""
    conditions: Dict[str, Dict[str, pd.DataFrame]] = {}

    for metrics_path in sorted(input_dir.glob("metrics_*.csv")):
        key = metrics_path.stem.replace("metrics_", "", 1)
        diversity_path = input_dir / f"diversity_{key}.csv"

        metrics_df = add_session_id(safe_read_csv(metrics_path))
        diversity_df = add_session_id(safe_read_csv(diversity_path))

        display_name = ""
        if "display_name" in metrics_df.columns:
            display_name = first_nonempty(metrics_df["display_name"])
        if not display_name and "display_name" in diversity_df.columns:
            display_name = first_nonempty(diversity_df["display_name"])
        if not display_name:
            display_name = DISPLAY_NAME_MAP.get(key, key)

        conditions[display_name] = {
            "key": key,
            "metrics": metrics_df,
            "diversity": diversity_df,
        }

    return dict(sorted(conditions.items(), key=lambda item: condition_sort_key(item[0])))


def load_run_metadata(input_dir: Path) -> Dict[str, str]:
    """Load optional run metadata emitted by automatic_evaluation.py."""
    metadata_path = input_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def detect_quality_metric(
    conditions: Dict[str, Dict[str, pd.DataFrame]],
    run_metadata: Dict[str, str],
) -> Tuple[str, str, str]:
    """Infer which primary quality metric this directory contains."""
    metric_column = run_metadata.get("quality_metric_column", "").strip()
    metric_name = run_metadata.get("quality_metric_display_name", "").strip()

    if not metric_column:
        for payload in conditions.values():
            metrics_df = payload["metrics"]
            if "judge_score" in metrics_df.columns and metrics_df["judge_score"].notna().any():
                metric_column = "judge_score"
                metric_name = "LLM Judge Score"
                break

    if not metric_column:
        metric_column = "perplexity"
    if not metric_name:
        metric_name = "Perplexity" if metric_column == "perplexity" else "LLM Judge Score"

    summary_title = f"Mean {metric_name} {'↓' if metric_column == 'perplexity' else '↑'}"
    return metric_column, metric_name, summary_title


def compute_summary(
    conditions: Dict[str, Dict[str, pd.DataFrame]],
    quality_metric_column: str,
) -> pd.DataFrame:
    """Compute one summary row per condition."""
    rows = []

    for display_name, payload in conditions.items():
        metrics_df = payload["metrics"]
        diversity_df = payload["diversity"]
        valid_diversity = diversity_df[diversity_df["cf_count"] > 1] if "cf_count" in diversity_df.columns else diversity_df

        rows.append(
            {
                "condition": display_name,
                "condition_key": payload["key"],
                "n_counterfactuals": len(metrics_df),
                "n_sessions": diversity_df["session_id"].nunique() if "session_id" in diversity_df.columns else len(diversity_df),
                "mean_quality_metric": (
                    metrics_df[quality_metric_column].mean(skipna=True)
                    if quality_metric_column in metrics_df.columns
                    else float("nan")
                ),
                "mean_target_similarity": metrics_df["similarity_to_target"].mean(skipna=True) if "similarity_to_target" in metrics_df.columns else float("nan"),
                "mean_phase_similarity": metrics_df["similarity_to_phase"].mean(skipna=True) if "similarity_to_phase" in metrics_df.columns else float("nan"),
                "mean_diversity_valid": valid_diversity["diversity"].mean(skipna=True) if "diversity" in valid_diversity.columns and not valid_diversity.empty else float("nan"),
                "mean_cf_count": diversity_df["cf_count"].mean(skipna=True) if "cf_count" in diversity_df.columns else float("nan"),
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("condition", key=lambda s: s.map(lambda x: condition_sort_key(x)))
    return summary_df


def annotate_bars(ax, bars, fmt: str = "{:.2f}") -> None:
    """Add text labels to bar charts."""
    values = [bar.get_height() for bar in bars]
    finite_values = [value for value in values if np.isfinite(value)]
    scale = max(abs(value) for value in finite_values) if finite_values else 1.0

    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        offset = 0.02 * scale if scale else 0.02
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=9,
        )


def plot_summary_bars(summary_df: pd.DataFrame, output_path: Path, quality_metric_title: str) -> None:
    """Plot 4 summary bar charts, one per evaluation metric."""
    metrics = [
        ("mean_quality_metric", quality_metric_title),
        ("mean_target_similarity", "Mean Target Similarity ↑"),
        ("mean_phase_similarity", "Mean Phase Similarity ↑"),
        ("mean_diversity_valid", "Mean Diversity ↑"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    colors = [COLORS.get(name, "#9e9e9e") for name in summary_df["condition"]]

    for ax, (column, title) in zip(axes, metrics):
        plot_df = summary_df.dropna(subset=[column])
        bars = ax.bar(plot_df["condition"], plot_df[column], color=[COLORS.get(name, "#9e9e9e") for name in plot_df["condition"]])
        annotate_bars(ax, bars)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    fig.suptitle(
        "Automatic Evaluation Summary Across Conditions",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")


def build_session_distribution_df(
    conditions: Dict[str, Dict[str, pd.DataFrame]],
    quality_metric_column: str,
    quality_metric_name: str,
) -> pd.DataFrame:
    """Aggregate per-session metrics for boxplot-style distribution charts."""
    rows: List[Dict[str, object]] = []

    for display_name, payload in conditions.items():
        metrics_df = payload["metrics"]
        diversity_df = payload["diversity"]

        if not metrics_df.empty:
            aggregation_columns = [quality_metric_column, "similarity_to_target", "similarity_to_phase"]
            aggregation_columns = [col for col in aggregation_columns if col in metrics_df.columns]
            session_means = (
                metrics_df.groupby("session_id")[aggregation_columns]
                .mean(numeric_only=True)
                .reset_index()
            )
            for _, row in session_means.iterrows():
                if quality_metric_column in row.index:
                    rows.append(
                        {
                            "condition": display_name,
                            "metric": quality_metric_name,
                            "value": row[quality_metric_column],
                        }
                    )
                if "similarity_to_target" in row.index:
                    rows.append(
                        {
                            "condition": display_name,
                            "metric": "Target Similarity",
                            "value": row["similarity_to_target"],
                        }
                    )
                if "similarity_to_phase" in row.index:
                    rows.append(
                        {
                            "condition": display_name,
                            "metric": "Phase Similarity",
                            "value": row["similarity_to_phase"],
                        }
                    )

        if not diversity_df.empty and "cf_count" in diversity_df.columns:
            for value in diversity_df["cf_count"].dropna():
                rows.append(
                    {
                        "condition": display_name,
                        "metric": "Counterfactual Count",
                        "value": value,
                    }
                )

    dist_df = pd.DataFrame(rows)
    if not dist_df.empty:
        dist_df = dist_df.sort_values("condition", key=lambda s: s.map(lambda x: condition_sort_key(x)))
    return dist_df


def plot_session_boxplots(dist_df: pd.DataFrame, output_path: Path) -> None:
    """Plot per-session distributions for the main metrics."""
    quality_metric_name = ""
    if not dist_df.empty and "metric" in dist_df.columns:
        quality_metric_name = first_nonempty(
            dist_df.loc[
                ~dist_df["metric"].isin(["Target Similarity", "Phase Similarity", "Counterfactual Count"]),
                "metric",
            ]
        )
    metric_order = [
        quality_metric_name or "Perplexity",
        "Target Similarity",
        "Phase Similarity",
        "Counterfactual Count",
    ]
    conditions = list(dict.fromkeys(dist_df["condition"])) if not dist_df.empty else []

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=False)

    for ax, metric in zip(axes, metric_order):
        data = []
        for condition in conditions:
            values = (
                dist_df[
                    (dist_df["condition"] == condition) & (dist_df["metric"] == metric)
                ]["value"]
                .dropna()
                .values
            )
            data.append(values)

        bp = ax.boxplot(data, tick_labels=conditions, patch_artist=True, widths=0.6)
        for patch, condition in zip(bp["boxes"], conditions):
            patch.set_facecolor(COLORS.get(condition, "#9e9e9e"))
            patch.set_alpha(0.75)

        ax.set_title(metric, fontsize=12)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        if metric == "Target Similarity" or metric == "Phase Similarity":
            ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Per-Session Distribution of Automatic Evaluation Metrics",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")


def compute_phase_profiles(conditions: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Compute count and coverage percentages by inferred phase."""
    rows = []

    for display_name, payload in conditions.items():
        metrics_df = payload["metrics"]
        if metrics_df.empty or "phase" not in metrics_df.columns:
            continue

        metrics_df = metrics_df.copy()
        metrics_df["phase"] = metrics_df["phase"].astype(str)

        count_series = metrics_df["phase"].value_counts().reindex([str(idx) for idx in range(5)], fill_value=0)
        coverage_series = (
            metrics_df.groupby("session_id")["phase"]
            .unique()
            .apply(lambda phases: {str(phase) for phase in phases})
        )
        coverage_counts = np.zeros(5, dtype=int)
        for phases in coverage_series:
            for idx in range(5):
                if str(idx) in phases:
                    coverage_counts[idx] += 1

        count_total = count_series.sum()
        coverage_total = coverage_counts.sum()

        for idx in range(5):
            rows.append(
                {
                    "condition": display_name,
                    "phase": str(idx),
                    "phase_label": STAGE_LABELS[idx],
                    "count_pct": (count_series.iloc[idx] / count_total * 100.0) if count_total else 0.0,
                    "coverage_pct": (coverage_counts[idx] / coverage_total * 100.0) if coverage_total else 0.0,
                }
            )

    phase_df = pd.DataFrame(rows)
    if not phase_df.empty:
        phase_df = phase_df.sort_values("condition", key=lambda s: s.map(lambda x: condition_sort_key(x)))
    return phase_df


def plot_phase_profiles(phase_df: pd.DataFrame, output_path: Path) -> None:
    """Plot grouped bars for inferred phase count and coverage percentages."""
    conditions = list(dict.fromkeys(phase_df["condition"])) if not phase_df.empty else []
    width = 0.8 / max(len(conditions), 1)
    x = np.arange(5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False)

    for ax, metric, title in [
        (axes[0], "count_pct", "Counterfactual Share by Inferred Phase (%)"),
        (axes[1], "coverage_pct", "Session Coverage by Inferred Phase (%)"),
    ]:
        for idx, condition in enumerate(conditions):
            subset = phase_df[phase_df["condition"] == condition].sort_values("phase")
            ax.bar(
                x + idx * width,
                subset[metric].values,
                width,
                label=condition,
                color=COLORS.get(condition, "#9e9e9e"),
            )

        ax.set_xticks(x + width * (len(conditions) - 1) / 2)
        ax.set_xticklabels(STAGE_LABELS, rotation=25, ha="right")
        ax.set_ylabel("Percentage (%)")
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(conditions)), frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Phase-Level Profiles Across Conditions",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")


def plot_dataset_overview(normalized_df: pd.DataFrame, output_path: Path) -> None:
    """Plot simple dataset-overview counts for session JSON runs."""
    if normalized_df.empty:
        return

    interaction_counts = (
        normalized_df["interactionCondition"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().sort_index()
        if "interactionCondition" in normalized_df.columns
        else pd.Series(dtype=int)
    )
    plan_source_counts = (
        normalized_df["human_plan_source"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().sort_index()
        if "human_plan_source" in normalized_df.columns
        else pd.Series(dtype=int)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(interaction_counts.index, interaction_counts.values, color="#5c80bc")
    axes[0].set_title("Session Count by Interaction Condition", fontsize=12)
    axes[0].set_ylabel("Sessions")
    axes[0].grid(axis="y", alpha=0.25)
    annotate_bars(axes[0], axes[0].patches, fmt="{:.0f}")

    axes[1].bar(plan_source_counts.index, plan_source_counts.values, color="#84a59d")
    axes[1].set_title("Human Plan Source Used for Analysis", fontsize=12)
    axes[1].grid(axis="y", alpha=0.25)
    annotate_bars(axes[1], axes[1].patches, fmt="{:.0f}")

    fig.suptitle(
        "Dataset Overview for Session-JSON Analysis",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize reproducibility-package automatic-evaluation outputs")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "outputs" / "automatic_evaluation"),
        help="Directory containing metrics_*.csv and diversity_*.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to write figures and summary tables (defaults to <input>/figures)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve() if args.output else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = discover_condition_outputs(input_dir)
    if not conditions:
        raise ValueError(f"No metrics_*.csv files found in {input_dir}")

    run_metadata = load_run_metadata(input_dir)
    quality_metric_column, quality_metric_name, quality_metric_title = detect_quality_metric(
        conditions,
        run_metadata,
    )

    summary_df = compute_summary(conditions, quality_metric_column)
    summary_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved table -> {summary_path}")

    phase_df = compute_phase_profiles(conditions)
    phase_path = output_dir / "phase_profiles.csv"
    phase_df.to_csv(phase_path, index=False)
    print(f"Saved table -> {phase_path}")

    plot_summary_bars(summary_df, output_dir / "figure_autoeval_summary_bars.png", quality_metric_title)
    plot_session_boxplots(
        build_session_distribution_df(conditions, quality_metric_column, quality_metric_name),
        output_dir / "figure_autoeval_session_boxplots.png",
    )
    plot_phase_profiles(phase_df, output_dir / "figure_autoeval_phase_profiles.png")

    normalized_path = input_dir / "normalized_sessions.csv"
    if normalized_path.exists():
        normalized_df = safe_read_csv(normalized_path)
        plot_dataset_overview(normalized_df, output_dir / "figure_dataset_overview.png")


if __name__ == "__main__":
    main()
