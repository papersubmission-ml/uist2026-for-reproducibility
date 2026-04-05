"""
JSON-native user rating analysis from session exports.

This script analyzes participant ratings that are already stored in the
downloaded `HAI-UIST-DATA/*.json` files. It is intentionally separate from the
legacy automatic benchmark pipeline because it answers a different research
question: which AI suggestion type participants preferred on the in-product
rating axes.

Outputs:
- `ratings_long.csv`
- `session_metadata.csv`
- `descriptive_statistics.csv`
- `condition_descriptives.csv`
- `friedman_tests.csv`
- `pairwise_wilcoxon.csv`
- `analysis_summary.md`
- figures in `figures/`
- including a combined all-metrics boxplot
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.json_export_loader import (
    CONDITION_ORDER,
    SESSION_KEYS,
    iter_export_records,
    normalize_condition,
)


MODEL_ORDER = ["LLM-B", "LLM-CF", "LLM-CT"]
MODEL_DISPLAY = {
    "LLM-B": "LLM-B",
    "LLM-CF": "LLM-CF",
    "LLM-CT": "LLM-CT",
}
MODEL_COLORS = {
    "LLM-B": "#f4a261",
    "LLM-CF": "#2a9d8f",
    "LLM-CT": "#264653",
}
CONDITION_ALPHA = {
    "AIH": 0.55,
    "HAI": 0.85,
}

METRIC_ORDER = [
    "Action_Clarity",
    "Feasibility",
    "Goal_Alignment",
    "Insight_Novelty",
]
METRIC_DISPLAY = {
    "Action_Clarity": "Action Clarity",
    "Feasibility": "Feasibility",
    "Goal_Alignment": "Goal Alignment",
    "Insight_Novelty": "Insight / Novelty",
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing exported session JSON files.",
    )
    parser.add_argument(
        "--condition",
        choices=CONDITION_ORDER,
        default=None,
        help="Optional condition filter. Use this to analyze HAI and AIH separately.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where CSVs, figures, and the markdown summary will be written.",
    )
    return parser.parse_args()


def load_unique_sessions(input_path: Path, condition: str | None = None) -> Tuple[List[Dict], pd.DataFrame]:
    """Load and deduplicate logical sessions by export id + session key."""
    session_records: List[Dict] = []
    session_rows: List[Dict] = []
    seen_sessions: Dict[str, Dict] = {}
    duplicate_count = 0
    source_files = set()
    raw_export_count = 0

    for export_record in iter_export_records(input_path):
        payload = export_record["payload"]
        export_id = export_record["export_id"]
        source_files.add(export_record["source_file"])
        raw_export_count += 1

        for session_key in SESSION_KEYS:
            session = payload.get(session_key)
            if not isinstance(session, dict):
                continue
            session_condition = normalize_condition(session.get("interactionCondition"))
            if condition and session_condition != condition:
                continue

            session_uid = f"{export_id}:{session_key}"
            record = {
                "session_uid": session_uid,
                "export_id": export_id,
                "source_file": export_record["source_file"],
                "session_key": session_key,
                "interaction_condition": session_condition,
                "session": session,
            }

            if session_uid in seen_sessions:
                if seen_sessions[session_uid]["session"] != session:
                    raise ValueError(
                        f"Conflicting duplicate export for {session_uid}: "
                        f"{seen_sessions[session_uid]['source_file']} vs {export_record['source_file']}"
                    )
                duplicate_count += 1
                continue

            seen_sessions[session_uid] = record

    for record in seen_sessions.values():
        session = record["session"]
        session_records.append(record)
        session_rows.append(
            {
                "session_uid": record["session_uid"],
                "export_id": record["export_id"],
                "source_file": record["source_file"],
                "session_key": record["session_key"],
                "participant_id": session.get("participandID", ""),
                "participant_email": session.get("participantEmail", ""),
                "session_number": session.get("session", ""),
                "interaction_condition": record["interaction_condition"],
                "scenario": session.get("scenario", ""),
                "user_goal": session.get("user_goal", ""),
                "rating_submitted_at": session.get("ratingSubmittedAt", ""),
            }
        )

    metadata = pd.DataFrame(session_rows).sort_values(
        by=["participant_id", "session_number", "session_key", "source_file"]
    )
    metadata.attrs["num_source_files"] = len(source_files)
    metadata.attrs["num_raw_exports"] = raw_export_count
    metadata.attrs["duplicate_count"] = duplicate_count
    metadata.attrs["condition_filter"] = condition or ""
    return session_records, metadata


def coerce_rating(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def build_long_ratings(session_records: Iterable[Dict]) -> pd.DataFrame:
    rows = []
    for record in session_records:
        session = record["session"]
        suggestions = session.get("AI_Suggestions") or {}
        for model in MODEL_ORDER:
            suggestion = suggestions.get(model) or {}
            suggestion_text = suggestion.get("item", "")
            ratings = suggestion.get("ratings") or {}
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
        raise ValueError("No ratings were found in the JSON exports.")

    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["metric"] = pd.Categorical(df["metric"], categories=METRIC_ORDER, ordered=True)
    return df.sort_values(by=["session_uid", "metric", "model"]).reset_index(drop=True)


def compute_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["metric", "model"], observed=True)["rating"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    summary["metric_label"] = summary["metric"].map(METRIC_DISPLAY)
    summary["model_label"] = summary["model"].map(MODEL_DISPLAY)
    return summary.rename(
        columns={
            "mean": "mean_rating",
            "std": "sd_rating",
            "median": "median_rating",
            "count": "n",
        }
    )


def compute_condition_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["interaction_condition", "metric", "model"], observed=True)["rating"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    summary["metric_label"] = summary["metric"].map(METRIC_DISPLAY)
    summary["model_label"] = summary["model"].map(MODEL_DISPLAY)
    return summary.rename(
        columns={
            "mean": "mean_rating",
            "std": "sd_rating",
            "median": "median_rating",
            "count": "n",
        }
    )


def aggregate_ratings(df: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    """Average repeated sessions within the requested analysis unit."""
    keep_cols = [unit_col, "model", "metric"]
    if unit_col != "session_uid":
        keep_cols.insert(1, "interaction_condition")
    aggregated = (
        df.groupby(keep_cols, observed=True)["rating"]
        .mean()
        .reset_index()
    )
    return aggregated


def holm_correct(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni correction within a family of tests."""
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        candidate = (m - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def run_friedman_tests(df: pd.DataFrame, unit_col: str, analysis_level: str) -> pd.DataFrame:
    rows = []
    unit_df = aggregate_ratings(df, unit_col)
    for metric in METRIC_ORDER:
        pivot = (
            unit_df[unit_df["metric"] == metric]
            .pivot_table(index=unit_col, columns="model", values="rating", aggfunc="mean")
            .reindex(columns=MODEL_ORDER)
            .dropna()
        )
        if len(pivot) == 0:
            continue
        stat, p_value = friedmanchisquare(*[pivot[model].values for model in MODEL_ORDER])
        kendall_w = stat / (len(pivot) * (len(MODEL_ORDER) - 1))
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_DISPLAY[metric],
                "analysis_level": analysis_level,
                "unit_column": unit_col,
                "n_sessions": len(pivot),
                "chi_square": stat,
                "p_value": p_value,
                "kendalls_w": kendall_w,
            }
        )
    return pd.DataFrame(rows).rename(columns={"n_sessions": "n_units"})


def run_pairwise_tests(df: pd.DataFrame, unit_col: str, analysis_level: str) -> pd.DataFrame:
    rows = []
    unit_df = aggregate_ratings(df, unit_col)
    for metric in METRIC_ORDER:
        pivot = (
            unit_df[unit_df["metric"] == metric]
            .pivot_table(index=unit_col, columns="model", values="rating", aggfunc="mean")
            .reindex(columns=MODEL_ORDER)
            .dropna()
        )
        if len(pivot) == 0:
            continue

        metric_rows = []
        pairs = list(itertools.combinations(MODEL_ORDER, 2))
        raw_p_values = []
        for model_a, model_b in pairs:
            sample_a = pivot[model_a].to_numpy()
            sample_b = pivot[model_b].to_numpy()
            stat, p_value = wilcoxon(sample_a, sample_b, zero_method="wilcox")
            raw_p_values.append(p_value)
            metric_rows.append(
                {
                    "metric": metric,
                    "metric_label": METRIC_DISPLAY[metric],
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_label": MODEL_DISPLAY[model_a],
                    "model_b_label": MODEL_DISPLAY[model_b],
                    "analysis_level": analysis_level,
                    "unit_column": unit_col,
                    "n_units": len(pivot),
                    "wilcoxon_w": stat,
                    "p_value": p_value,
                    "mean_difference": float(np.mean(sample_a - sample_b)),
                    "median_difference": float(np.median(sample_a - sample_b)),
                }
            )

        adjusted = holm_correct(raw_p_values)
        for row, corrected_p in zip(metric_rows, adjusted):
            row["p_value_holm"] = corrected_p
            row["significant_holm_0_05"] = corrected_p < 0.05
            rows.append(row)

    return pd.DataFrame(rows)


def build_summary_markdown(
    metadata: pd.DataFrame,
    df: pd.DataFrame,
    descriptives: pd.DataFrame,
    friedman: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> str:
    num_participants = metadata["participant_id"].replace("", np.nan).nunique(dropna=True)
    num_sessions = metadata["session_uid"].nunique()
    duplicates_removed = metadata.attrs.get("duplicate_count", 0)
    condition_counts = metadata["interaction_condition"].value_counts().to_dict()
    participant_condition_counts = (
        metadata.groupby("interaction_condition")["participant_id"].nunique().to_dict()
    )
    sessions_per_participant = (
        metadata.groupby("participant_id")["session_uid"].nunique().sort_values()
    )
    primary_friedman = friedman[friedman["analysis_level"] == "participant"].copy()
    primary_pairwise = pairwise[pairwise["analysis_level"] == "participant"].copy()

    lines = [
        "# User Rating Analysis Summary",
        "",
        "## Dataset",
        f"- Source files: {metadata.attrs.get('num_source_files', 'unknown')}",
        f"- Raw exports scanned: {metadata.attrs.get('num_raw_exports', 'unknown')}",
        f"- Condition filter: {metadata.attrs.get('condition_filter', 'all') or 'all'}",
        f"- Unique logical sessions: {num_sessions}",
        f"- Duplicate exports removed: {duplicates_removed}",
        f"- Participants with non-empty IDs: {num_participants}",
        f"- Session counts by interaction condition: {condition_counts}",
        f"- Participant counts by interaction condition: {participant_condition_counts}",
        f"- Sessions per participant: min={sessions_per_participant.min()}, median={sessions_per_participant.median():.0f}, max={sessions_per_participant.max()}",
        f"- Rating rows analyzed: {len(df)}",
        "",
        "## Descriptive Highlights",
    ]

    for metric in METRIC_ORDER:
        sub = descriptives[descriptives["metric"] == metric].sort_values("mean_rating", ascending=False)
        if sub.empty:
            continue
        winner = sub.iloc[0]
        lines.append(
            f"- {METRIC_DISPLAY[metric]}: {winner['model_label']} had the highest mean "
            f"({winner['mean_rating']:.2f}, SD {winner['sd_rating']:.2f}, n={int(winner['n'])})."
        )

    lines.extend(["", "## Omnibus Tests"])
    if primary_friedman.empty:
        lines.append("- No complete repeated-measures rows were available for the Friedman tests.")
    else:
        for _, row in primary_friedman.iterrows():
            interpretation = "not statistically significant"
            if row["p_value"] < 0.05:
                interpretation = "statistically significant"
            lines.append(
                f"- {row['metric_label']}: Friedman chi-square={row['chi_square']:.2f}, "
                f"p={row['p_value']:.4f}, Kendall's W={row['kendalls_w']:.3f}, n={int(row['n_units'])} "
                f"({interpretation})."
            )

    lines.extend(["", "## Pairwise Tests"])
    if primary_pairwise.empty:
        lines.append("- No pairwise tests were computed.")
    else:
        significant = primary_pairwise[primary_pairwise["significant_holm_0_05"]]
        if significant.empty:
            lines.append("- No pairwise comparison survived Holm correction at alpha=0.05.")
        else:
            for _, row in significant.iterrows():
                direction = (
                    f"{row['model_a_label']} > {row['model_b_label']}"
                    if row["mean_difference"] > 0
                    else f"{row['model_b_label']} > {row['model_a_label']}"
                )
                lines.append(
                    f"- {row['metric_label']}: {direction}, Holm-adjusted p={row['p_value_holm']:.4f}, "
                    f"mean difference={row['mean_difference']:.2f}."
                )

    notes = [
        "",
        "## Interpretation Notes",
        "- These are participant ratings of AI-generated suggestions, not automatic text-quality scores.",
        "- Participant-level tests are the primary inferential result because each participant contributed three sessions.",
        "- Because the outcome is an ordinal 1-5 rating, Wilcoxon signed-rank tests are used for pairwise contrasts.",
    ]
    if metadata.attrs.get("condition_filter"):
        notes.insert(
            3,
            f"- This summary is restricted to the `{metadata.attrs['condition_filter']}` condition; compare it against the separate condition output rather than treating it as a pooled analysis.",
        )
    else:
        notes.insert(
            3,
            "- If HAI and AIH are imbalanced in a pooled run, interpret timing differences cautiously and prefer separate-condition summaries.",
        )
    lines.extend(notes)
    return "\n".join(lines) + "\n"


def make_bar_figure(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, METRIC_ORDER):
        sub = df[df["metric"] == metric]
        stats = (
            sub.groupby("model", observed=True)["rating"]
            .agg(["mean", "count", "std"])
            .reindex(MODEL_ORDER)
        )
        means = stats["mean"].to_numpy()
        errors = (stats["std"] / np.sqrt(stats["count"])).fillna(0.0).to_numpy()
        positions = np.arange(len(MODEL_ORDER))
        colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
        ax.bar(positions, means, yerr=errors, capsize=4, color=colors, alpha=0.9)
        ax.set_title(METRIC_DISPLAY[metric])
        ax.set_xticks(positions, [MODEL_DISPLAY[model] for model in MODEL_ORDER])
        ax.set_ylim(1.0, 5.0)
        ax.set_ylabel("Mean Rating")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Participant Ratings by Model", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_boxplot_figure(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, METRIC_ORDER):
        sub = df[df["metric"] == metric]
        series = [sub[sub["model"] == model]["rating"].to_list() for model in MODEL_ORDER]
        boxplot = ax.boxplot(
            series,
            tick_labels=[MODEL_DISPLAY[model] for model in MODEL_ORDER],
            patch_artist=True,
            widths=0.55,
        )
        for patch, model in zip(boxplot["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.85)
        for median in boxplot["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.8)
        ax.set_title(METRIC_DISPLAY[metric])
        ax.set_ylim(1.0, 5.0)
        ax.set_ylabel("Rating")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Participant Rating Distributions by Model", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_combined_boxplot_figure(df: pd.DataFrame, output_path: Path) -> None:
    conditions_present = [condition for condition in CONDITION_ORDER if condition in df["interaction_condition"].unique()]
    if not conditions_present:
        conditions_present = sorted(df["interaction_condition"].dropna().astype(str).unique().tolist())

    groups = []
    if len(conditions_present) <= 1:
        for model in MODEL_ORDER:
            groups.append(
                {
                    "label": MODEL_DISPLAY[model],
                    "model": model,
                    "condition": conditions_present[0] if conditions_present else None,
                    "color": MODEL_COLORS[model],
                    "alpha": 0.85,
                }
            )
    else:
        for condition in conditions_present:
            for model in MODEL_ORDER:
                groups.append(
                    {
                        "label": f"{MODEL_DISPLAY[model].replace('LLM-', '')}-{condition}",
                        "model": model,
                        "condition": condition,
                        "color": MODEL_COLORS[model],
                        "alpha": CONDITION_ALPHA.get(condition, 0.75),
                    }
                )

    metric_centers = np.arange(len(METRIC_ORDER), dtype=float) * 1.8
    box_width = min(0.22, 0.9 / max(len(groups), 1))
    offsets = (np.arange(len(groups), dtype=float) - (len(groups) - 1) / 2.0) * box_width

    fig, ax = plt.subplots(figsize=(14, 7))
    legend_handles = []

    for group_index, group in enumerate(groups):
        positions = metric_centers + offsets[group_index]
        series = []
        for metric in METRIC_ORDER:
            metric_df = df[(df["metric"] == metric) & (df["model"] == group["model"])]
            if group["condition"] is not None:
                metric_df = metric_df[metric_df["interaction_condition"] == group["condition"]]
            series.append(metric_df["rating"].to_list())

        boxplot = ax.boxplot(
            series,
            positions=positions,
            widths=box_width * 0.95,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(group["color"])
            patch.set_alpha(group["alpha"])
        for median in boxplot["medians"]:
            median.set_color("#444444")
            median.set_linewidth(1.6)
        for whisker in boxplot["whiskers"]:
            whisker.set_color("#666666")
        for cap in boxplot["caps"]:
            cap.set_color("#666666")
        for flier in boxplot["fliers"]:
            flier.set_markeredgecolor("#555555")
            flier.set_alpha(0.7)

        legend_handles.append(
            Patch(
                facecolor=group["color"],
                edgecolor="#444444",
                alpha=group["alpha"],
                label=group["label"],
            )
        )

    condition_suffix = ""
    if len(conditions_present) == 1:
        condition_suffix = f" ({conditions_present[0]})"
    ax.set_title(f"Model Comparison Across Rating Dimensions{condition_suffix}", fontsize=15)
    ax.set_xlabel("Rating Type", fontsize=12)
    ax.set_ylabel("Rating Value", fontsize=12)
    ax.set_xticks(metric_centers, [METRIC_DISPLAY[metric] for metric in METRIC_ORDER], rotation=18)
    ax.set_ylim(0.8, 5.2)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    legend_title = "Model" if len(conditions_present) <= 1 else "Condition x Model"
    ax.legend(handles=legend_handles, title=legend_title, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_condition_figure(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, METRIC_ORDER):
        sub = (
            df[df["metric"] == metric]
            .groupby(["interaction_condition", "model"], observed=True)["rating"]
            .mean()
            .reset_index()
        )
        positions = np.arange(len(CONDITION_ORDER))
        width = 0.24
        offsets = [-width, 0.0, width]

        for model, offset in zip(MODEL_ORDER, offsets):
            model_values = []
            for condition in CONDITION_ORDER:
                match = sub[
                    (sub["interaction_condition"] == condition) & (sub["model"] == model)
                ]["rating"]
                model_values.append(float(match.iloc[0]) if not match.empty else np.nan)
            ax.bar(
                positions + offset,
                model_values,
                width=width,
                color=MODEL_COLORS[model],
                label=MODEL_DISPLAY[model],
                alpha=0.9,
            )

        ax.set_title(METRIC_DISPLAY[metric])
        ax.set_xticks(positions, CONDITION_ORDER)
        ax.set_ylim(1.0, 5.0)
        ax.set_ylabel("Mean Rating")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Participant Ratings by Interaction Condition", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    metadata: pd.DataFrame,
    ratings: pd.DataFrame,
    participant_ratings: pd.DataFrame,
    descriptives: pd.DataFrame,
    condition_descriptives: pd.DataFrame,
    friedman: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_dir / "session_metadata.csv", index=False)
    ratings.to_csv(output_dir / "ratings_long.csv", index=False)
    participant_ratings.to_csv(output_dir / "participant_aggregated_ratings.csv", index=False)
    descriptives.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    condition_descriptives.to_csv(output_dir / "condition_descriptives.csv", index=False)
    friedman.to_csv(output_dir / "friedman_tests.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_wilcoxon.csv", index=False)

    summary = build_summary_markdown(metadata, ratings, descriptives, friedman, pairwise)
    (output_dir / "analysis_summary.md").write_text(summary)

    make_bar_figure(ratings, figures_dir / "figure_user_ratings_summary_bars.png")
    make_boxplot_figure(ratings, figures_dir / "figure_user_ratings_boxplots.png")
    make_combined_boxplot_figure(
        participant_ratings,
        figures_dir / "figure_user_ratings_all_in_one_boxplot.png",
    )
    make_condition_figure(ratings, figures_dir / "figure_user_ratings_by_condition.png")


def main() -> None:
    args = parse_args()
    session_records, metadata = load_unique_sessions(args.input, condition=args.condition)
    ratings = build_long_ratings(session_records)
    participant_ratings = aggregate_ratings(ratings, "participant_id")
    descriptives = compute_descriptives(ratings)
    condition_descriptives = compute_condition_descriptives(ratings)
    friedman = pd.concat(
        [
            run_friedman_tests(ratings, unit_col="session_uid", analysis_level="session"),
            run_friedman_tests(ratings, unit_col="participant_id", analysis_level="participant"),
        ],
        ignore_index=True,
    )
    pairwise = pd.concat(
        [
            run_pairwise_tests(ratings, unit_col="session_uid", analysis_level="session"),
            run_pairwise_tests(ratings, unit_col="participant_id", analysis_level="participant"),
        ],
        ignore_index=True,
    )
    write_outputs(
        args.output,
        metadata,
        ratings,
        participant_ratings,
        descriptives,
        condition_descriptives,
        friedman,
        pairwise,
    )

    print(f"Wrote user rating analysis to {args.output}")
    print(f"Unique logical sessions: {metadata['session_uid'].nunique()}")
    print(f"Rating rows: {len(ratings)}")


if __name__ == "__main__":
    main()
