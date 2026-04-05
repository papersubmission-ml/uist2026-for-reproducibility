"""
JSON-native final plan analysis from session exports.

This script reads `HAI-UIST-DATA/*.json` directly and analyzes the human-authored
initial plans, final plans, and the selected final plan. It does not use the
legacy automatic benchmark pipeline or the older staged CSV workflow.

Outputs:
- `final_plan_session_metrics.csv`
- `selected_final_vs_ai_similarity.csv`
- `participant_aggregated_final_plan_metrics.csv`
- `condition_descriptives.csv`
- `participant_delta_tests.csv`
- `closest_model_counts.csv`
- `analysis_summary.md`
- figures in `figures/`
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
SUMMARY_METRICS = [
    "plan_count_delta",
    "selected_final_goal_similarity",
    "selected_goal_similarity_delta",
    "selected_novelty_vs_initial",
    "max_ai_similarity",
]
SUMMARY_METRIC_LABELS = {
    "plan_count_delta": "Plan Count Delta",
    "selected_final_goal_similarity": "Selected Final Goal Similarity",
    "selected_goal_similarity_delta": "Goal Similarity Delta",
    "selected_novelty_vs_initial": "Selected Plan Novelty",
    "max_ai_similarity": "Selected-to-AI Similarity",
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


def normalize_text(value) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_list(values) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    seen = set()
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned


def format_summary_value(value: float) -> str:
    if pd.isna(value):
        return "undefined"
    return f"{value:.3f}"


def load_unique_sessions(input_path: Path, condition: str | None = None) -> Tuple[List[Dict], pd.DataFrame]:
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
        final_plans = clean_text_list(session.get("User_Plans_Final"))
        initial_plans = clean_text_list(session.get("User_Plans_Initial"))
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
                "scenario": normalize_text(session.get("scenario", "")),
                "user_goal": normalize_text(session.get("user_goal", "")),
                "n_initial_plans": len(initial_plans),
                "n_final_plans": len(final_plans),
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


def choose_selected_final_plan(final_plans: List[str], selected_index) -> Tuple[str, int | None, bool]:
    if isinstance(selected_index, int) and 0 <= selected_index < len(final_plans):
        return final_plans[selected_index], selected_index, True
    if final_plans:
        return final_plans[0], 0, False
    return "", None, False


def build_text_space(session_records: Iterable[Dict]) -> Tuple[TfidfVectorizer, Dict[str, int], object]:
    texts: List[str] = []
    for record in session_records:
        session = record["session"]
        texts.extend(clean_text_list(session.get("User_Plans_Initial")))
        texts.extend(clean_text_list(session.get("User_Plans_Final")))
        texts.extend(
            [
                normalize_text(session.get("user_goal", "")),
                normalize_text(session.get("scenario", "")),
            ]
        )
        suggestions = session.get("AI_Suggestions") or {}
        for model in MODEL_ORDER:
            suggestion = suggestions.get(model) or {}
            texts.append(normalize_text(suggestion.get("item", "")))

    unique_texts = []
    seen = set()
    for text in texts:
        if not text or text in seen:
            continue
        unique_texts.append(text)
        seen.add(text)

    if not unique_texts:
        raise ValueError("No analyzable text was found in the JSON exports.")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(unique_texts)
    lookup = {text: idx for idx, text in enumerate(unique_texts)}
    return vectorizer, lookup, matrix


def cosine_for_texts(text_a: str, text_b: str, lookup: Dict[str, int], matrix) -> float:
    text_a = normalize_text(text_a)
    text_b = normalize_text(text_b)
    if not text_a or not text_b:
        return float("nan")
    idx_a = lookup.get(text_a)
    idx_b = lookup.get(text_b)
    if idx_a is None or idx_b is None:
        return float("nan")
    return float(cosine_similarity(matrix[idx_a], matrix[idx_b])[0, 0])


def max_similarity(source_texts: Iterable[str], target_text: str, lookup: Dict[str, int], matrix) -> float:
    scores = [
        cosine_for_texts(source_text, target_text, lookup, matrix)
        for source_text in source_texts
        if normalize_text(source_text)
    ]
    scores = [score for score in scores if not math.isnan(score)]
    if not scores:
        return float("nan")
    return float(max(scores))


def safe_word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", normalize_text(text)))


def build_plan_metrics(
    session_records: Iterable[Dict],
    lookup: Dict[str, int],
    matrix,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    session_rows: List[Dict] = []
    similarity_rows: List[Dict] = []

    for record in session_records:
        session = record["session"]
        initial_plans = clean_text_list(session.get("User_Plans_Initial"))
        final_plans = clean_text_list(session.get("User_Plans_Final"))
        selected_final_plan, selected_index, selected_index_was_valid = choose_selected_final_plan(
            final_plans,
            session.get("Final_Selected_PlanNumber"),
        )
        user_goal = normalize_text(session.get("user_goal", ""))
        scenario = normalize_text(session.get("scenario", ""))
        suggestions = session.get("AI_Suggestions") or {}

        best_initial_goal_similarity = max_similarity(initial_plans, user_goal, lookup, matrix)
        best_final_goal_similarity = max_similarity(final_plans, user_goal, lookup, matrix)
        selected_final_goal_similarity = cosine_for_texts(selected_final_plan, user_goal, lookup, matrix)
        best_initial_scenario_similarity = max_similarity(initial_plans, scenario, lookup, matrix)
        best_final_scenario_similarity = max_similarity(final_plans, scenario, lookup, matrix)
        selected_final_scenario_similarity = cosine_for_texts(selected_final_plan, scenario, lookup, matrix)
        selected_vs_initial_similarity = max_similarity(initial_plans, selected_final_plan, lookup, matrix)
        selected_novelty_vs_initial = (
            float("nan")
            if math.isnan(selected_vs_initial_similarity)
            else float(max(0.0, 1.0 - selected_vs_initial_similarity))
        )

        model_similarities = {}
        closest_model = ""
        adopted_model = ""
        max_ai_similarity = float("nan")
        for model in MODEL_ORDER:
            suggestion = suggestions.get(model) or {}
            suggestion_text = normalize_text(suggestion.get("item", ""))
            similarity = cosine_for_texts(selected_final_plan, suggestion_text, lookup, matrix)
            model_similarities[model] = similarity
            similarity_rows.append(
                {
                    "session_uid": record["session_uid"],
                    "export_id": record["export_id"],
                    "source_file": record["source_file"],
                    "session_key": record["session_key"],
                    "participant_id": session.get("participandID", ""),
                    "session_number": session.get("session", ""),
                    "interaction_condition": record["interaction_condition"],
                    "model": model,
                    "model_label": MODEL_DISPLAY[model],
                    "selected_final_plan": selected_final_plan,
                    "ai_suggestion": suggestion_text,
                    "selected_to_ai_similarity": similarity,
                }
            )

        valid_model_scores = {
            model: score for model, score in model_similarities.items() if not math.isnan(score)
        }
        if valid_model_scores:
            closest_model = max(valid_model_scores, key=valid_model_scores.get)
            max_ai_similarity = float(valid_model_scores[closest_model])
            if max_ai_similarity > 0:
                adopted_model = closest_model

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
                "scenario": scenario,
                "user_goal": user_goal,
                "initial_plans_joined": " || ".join(initial_plans),
                "final_plans_joined": " || ".join(final_plans),
                "selected_final_plan": selected_final_plan,
                "selected_final_plan_index": selected_index,
                "selected_index_was_valid": selected_index_was_valid,
                "n_initial_plans": len(initial_plans),
                "n_final_plans": len(final_plans),
                "plan_count_delta": len(final_plans) - len(initial_plans),
                "selected_final_word_count": safe_word_count(selected_final_plan),
                "best_initial_goal_similarity": best_initial_goal_similarity,
                "best_final_goal_similarity": best_final_goal_similarity,
                "selected_final_goal_similarity": selected_final_goal_similarity,
                "selected_goal_similarity_delta": (
                    selected_final_goal_similarity - best_initial_goal_similarity
                    if not math.isnan(selected_final_goal_similarity)
                    and not math.isnan(best_initial_goal_similarity)
                    else float("nan")
                ),
                "best_initial_scenario_similarity": best_initial_scenario_similarity,
                "best_final_scenario_similarity": best_final_scenario_similarity,
                "selected_final_scenario_similarity": selected_final_scenario_similarity,
                "selected_scenario_similarity_delta": (
                    selected_final_scenario_similarity - best_initial_scenario_similarity
                    if not math.isnan(selected_final_scenario_similarity)
                    and not math.isnan(best_initial_scenario_similarity)
                    else float("nan")
                ),
                "selected_vs_initial_similarity": selected_vs_initial_similarity,
                "selected_novelty_vs_initial": selected_novelty_vs_initial,
                "selected_to_llm_b_similarity": model_similarities["LLM-B"],
                "selected_to_llm_cf_similarity": model_similarities["LLM-CF"],
                "selected_to_llm_ct_similarity": model_similarities["LLM-CT"],
                "closest_model": closest_model,
                "closest_model_label": MODEL_DISPLAY.get(closest_model, ""),
                "adopted_model": adopted_model,
                "adopted_model_label": MODEL_DISPLAY.get(adopted_model, ""),
                "max_ai_similarity": max_ai_similarity,
            }
        )

    session_df = pd.DataFrame(session_rows).sort_values(
        by=["participant_id", "session_number", "session_key"]
    )
    similarity_df = pd.DataFrame(similarity_rows).sort_values(
        by=["participant_id", "session_number", "model"]
    )
    return session_df.reset_index(drop=True), similarity_df.reset_index(drop=True)


def aggregate_participant_metrics(session_df: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        session_df.groupby(["participant_id", "interaction_condition"], observed=True)[SUMMARY_METRICS]
        .mean()
        .reset_index()
    )
    return aggregated


def compute_condition_descriptives(session_df: pd.DataFrame) -> pd.DataFrame:
    long_df = session_df.melt(
        id_vars=["interaction_condition", "session_uid"],
        value_vars=SUMMARY_METRICS,
        var_name="metric",
        value_name="value",
    )
    summary = (
        long_df.groupby(["interaction_condition", "metric"], observed=True)["value"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    summary["metric_label"] = summary["metric"].map(SUMMARY_METRIC_LABELS)
    return summary.rename(
        columns={
            "mean": "mean_value",
            "std": "sd_value",
            "median": "median_value",
            "count": "n_sessions",
        }
    )


def run_participant_delta_tests(participant_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["plan_count_delta", "selected_goal_similarity_delta", "selected_scenario_similarity_delta"]:
        if metric not in participant_df.columns:
            continue
        sample = participant_df[metric].dropna()
        if sample.empty:
            continue
        nonzero = sample[np.abs(sample.to_numpy()) > 1e-12]
        if nonzero.empty:
            stat = 0.0
            p_value = 1.0
        else:
            stat, p_value = wilcoxon(nonzero.to_numpy(), zero_method="wilcox")
        rows.append(
            {
                "metric": metric,
                "metric_label": SUMMARY_METRIC_LABELS.get(metric, metric),
                "n_participants": int(sample.shape[0]),
                "mean_value": float(sample.mean()),
                "median_value": float(sample.median()),
                "wilcoxon_w": float(stat),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def compute_closest_model_counts(session_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        session_df[session_df["closest_model"] != ""]
        .groupby(["interaction_condition", "closest_model"], observed=True)["session_uid"]
        .count()
        .reset_index(name="n_sessions")
    )
    if summary.empty:
        return summary
    summary["closest_model_label"] = summary["closest_model"].map(MODEL_DISPLAY)
    totals = summary.groupby("interaction_condition")["n_sessions"].transform("sum")
    summary["prevalence_within_condition"] = summary["n_sessions"] / totals
    return summary.sort_values(by=["interaction_condition", "closest_model"]).reset_index(drop=True)


def build_summary_markdown(
    metadata: pd.DataFrame,
    session_df: pd.DataFrame,
    participant_df: pd.DataFrame,
    delta_tests: pd.DataFrame,
    closest_counts: pd.DataFrame,
) -> str:
    num_participants = metadata["participant_id"].replace("", np.nan).nunique(dropna=True)
    num_sessions = metadata["session_uid"].nunique()
    duplicates_removed = metadata.attrs.get("duplicate_count", 0)
    condition_counts = metadata["interaction_condition"].value_counts().to_dict()
    participant_condition_counts = (
        metadata.groupby("interaction_condition")["participant_id"].nunique().to_dict()
    )
    selected_valid_rate = float(session_df["selected_index_was_valid"].mean()) if not session_df.empty else float("nan")

    lines = [
        "# Final Plan Analysis Summary",
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
        f"- Selected final plan index valid rate: {selected_valid_rate * 100:.1f}%",
        "",
        "## Core Descriptives",
        f"- Mean initial plan count: {session_df['n_initial_plans'].mean():.2f}",
        f"- Mean final plan count: {session_df['n_final_plans'].mean():.2f}",
        f"- Mean plan count delta: {session_df['plan_count_delta'].mean():.2f}",
        f"- Mean selected final goal similarity: {format_summary_value(session_df['selected_final_goal_similarity'].mean())}",
        f"- Mean selected goal similarity delta: {format_summary_value(session_df['selected_goal_similarity_delta'].mean())}",
        f"- Mean selected plan novelty vs initial plans: {format_summary_value(session_df['selected_novelty_vs_initial'].mean())}",
        f"- Mean selected-to-AI similarity: {format_summary_value(session_df['max_ai_similarity'].mean())}",
        "",
        "## Participant-Level Delta Tests",
    ]

    if float(session_df["n_initial_plans"].sum()) == 0.0:
        lines.append("- Initial human plans are empty in this condition, so initial-to-final delta metrics are undefined by design.")

    if delta_tests.empty:
        lines.append("- No participant-level delta tests were computed.")
    else:
        for _, row in delta_tests.iterrows():
            significance = "not statistically significant"
            if row["p_value"] < 0.05:
                significance = "statistically significant"
            lines.append(
                f"- {row['metric_label']}: mean={row['mean_value']:.3f}, median={row['median_value']:.3f}, "
                f"Wilcoxon W={row['wilcoxon_w']:.2f}, p={row['p_value']:.4f} ({significance})."
            )

    lines.extend(["", "## Model Overlap with Selected Final Plans"])
    nonzero_overlap_rate = float((session_df["max_ai_similarity"] > 0).mean()) if not session_df.empty else float("nan")
    lines.append(f"- Sessions with nonzero selected-to-AI overlap: {nonzero_overlap_rate * 100:.1f}%")
    if closest_counts.empty:
        lines.append("- No closest-model summary was available.")
    else:
        overall = (
            session_df[session_df["closest_model"] != ""]
            .groupby("closest_model")["session_uid"]
            .count()
            .reindex(MODEL_ORDER)
            .fillna(0)
        )
        for model, count in overall.items():
            lines.append(f"- {MODEL_DISPLAY[model]} was the closest AI suggestion for {int(count)} sessions.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- This analysis is JSON-native: it reads `User_Plans_Initial`, `User_Plans_Final`, `Final_Selected_PlanNumber`, and `AI_Suggestions` directly from the downloaded exports.",
            "- It does not use the legacy staged CSV workflow or the old automatic benchmark outputs.",
            "- Goal and scenario similarities are TF-IDF cosine proxies, not human judgments.",
            "- Closest-model counts reflect nearest textual overlap, not proof that a participant consciously adopted that model's suggestion.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_metric_boxplots(session_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    metrics = [
        "plan_count_delta",
        "selected_goal_similarity_delta",
        "selected_novelty_vs_initial",
        "max_ai_similarity",
    ]

    for ax, metric in zip(axes, metrics):
        values = session_df[metric].dropna().to_list()
        ax.boxplot(values, tick_labels=[SUMMARY_METRIC_LABELS[metric]], widths=0.45, patch_artist=True)
        ax.set_title(SUMMARY_METRIC_LABELS[metric])
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Selected Final Plan Metrics", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_plan_count_bar(session_df: pd.DataFrame, output_path: Path) -> None:
    means = [
        float(session_df["n_initial_plans"].mean()),
        float(session_df["n_final_plans"].mean()),
    ]
    labels = ["Initial Plans", "Final Plans"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, means, color=["#9c6644", "#2a9d8f"], alpha=0.9)
    ax.set_ylabel("Mean Count")
    ax.set_title("Initial vs Final Human Plan Counts")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_closest_model_bar(closest_counts: pd.DataFrame, output_path: Path) -> None:
    overall = (
        closest_counts.groupby("closest_model", observed=True)["n_sessions"]
        .sum()
        .reindex(MODEL_ORDER)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(MODEL_ORDER))
    ax.bar(
        positions,
        overall.to_numpy(),
        color=[MODEL_COLORS[model] for model in MODEL_ORDER],
        alpha=0.9,
    )
    ax.set_xticks(positions, [MODEL_DISPLAY[model] for model in MODEL_ORDER])
    ax.set_ylabel("Session Count")
    ax.set_title("Closest AI Suggestion to Selected Final Plan")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    metadata: pd.DataFrame,
    session_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    participant_df: pd.DataFrame,
    condition_descriptives: pd.DataFrame,
    delta_tests: pd.DataFrame,
    closest_counts: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_dir / "session_metadata.csv", index=False)
    session_df.to_csv(output_dir / "final_plan_session_metrics.csv", index=False)
    similarity_df.to_csv(output_dir / "selected_final_vs_ai_similarity.csv", index=False)
    participant_df.to_csv(output_dir / "participant_aggregated_final_plan_metrics.csv", index=False)
    condition_descriptives.to_csv(output_dir / "condition_descriptives.csv", index=False)
    delta_tests.to_csv(output_dir / "participant_delta_tests.csv", index=False)
    closest_counts.to_csv(output_dir / "closest_model_counts.csv", index=False)

    summary = build_summary_markdown(metadata, session_df, participant_df, delta_tests, closest_counts)
    (output_dir / "analysis_summary.md").write_text(summary)

    make_metric_boxplots(session_df, figures_dir / "figure_final_plan_metrics.png")
    make_plan_count_bar(session_df, figures_dir / "figure_final_plan_counts.png")
    make_closest_model_bar(closest_counts, figures_dir / "figure_final_plan_closest_model_counts.png")


def main() -> None:
    args = parse_args()
    session_records, metadata = load_unique_sessions(args.input, condition=args.condition)
    _, lookup, matrix = build_text_space(session_records)
    session_df, similarity_df = build_plan_metrics(session_records, lookup, matrix)
    participant_df = aggregate_participant_metrics(session_df)
    condition_descriptives = compute_condition_descriptives(session_df)
    delta_tests = run_participant_delta_tests(participant_df)
    closest_counts = compute_closest_model_counts(session_df)
    write_outputs(
        args.output,
        metadata,
        session_df,
        similarity_df,
        participant_df,
        condition_descriptives,
        delta_tests,
        closest_counts,
    )

    print(f"Wrote final plan analysis to {args.output}")
    print(f"Unique logical sessions: {metadata['session_uid'].nunique()}")
    print(f"Participant rows: {participant_df.shape[0]}")


if __name__ == "__main__":
    main()
