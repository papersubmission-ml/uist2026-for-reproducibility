from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.plan_source_analysis import (
    build_lexical_space,
    build_semantic_space,
    clean_text_list,
    cosine_similarity_for,
    format_score_matrix,
    normalize_text,
)
from shared.json_export_loader import CONDITION_ORDER, SESSION_KEYS, iter_export_records, normalize_condition


LEGACY_MODEL_ORDER = ["LLM-CT", "LLM-CF", "LLM-B"]
LEGACY_MODEL_IDX = {model: index for index, model in enumerate(LEGACY_MODEL_ORDER)}
HUMAN_RATING_COLUMNS = {
    "Action_Clarity": "clarity",
    "Feasibility": "feasibility",
    "Goal_Alignment": "goal_alignment",
    "Insight_Novelty": "novelty",
}
LLM_JUDGE_COLUMNS = {
    "Action_Clarity": "llm_judge_action_clarity",
    "Feasibility": "llm_judge_feasibility",
    "Goal_Alignment": "llm_judge_goal_alignment",
    "Insight_Novelty": "llm_judge_insight_novelty",
}
THRESH_TFIDF = 0.60
THRESH_MINILM = 0.60
CSV_COLUMNS = [
    "participant_id",
    "session",
    "interactionCondition",
    "user_goal",
    "scenario",
    "areas_initial",
    "plans_initial",
    "plans_final",
    "final_plan_id",
    "model_name",
    "interactionStartedAt",
    "initialStartedAt",
    "initialSubmittedAt",
    "outputsDisplayedAt",
    "ratingSubmittedAt",
    "finalStartedAt",
    "finalSubmittedAt",
    "interactionEndedAt",
    "feedback",
    "model",
    "suggestion",
    "clarity",
    "feasibility",
    "goal_alignment",
    "novelty",
    "llm_judge_action_clarity",
    "llm_judge_feasibility",
    "llm_judge_goal_alignment",
    "llm_judge_insight_novelty",
    "final_plan_text",
    "selected_index",
    "initial_plan_TFIDF_source",
    "initial_plan_MiniLM_source",
    "final_candidate_TFIDF_source",
    "final_candidate_MiniLM_source",
    "final_decision_score",
]
DEFAULT_JUDGE_OUTPUT_ROOT = (
    Path(__file__).resolve().parent.parent / "benchmark" / "outputs" / "llm_judge_augmented_outputs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more raw or augmented session-export JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to write the legacy cleaned CSV shape used by the counterfactual analysis notebook.",
    )
    parser.add_argument(
        "--judge-inputs",
        type=Path,
        nargs="+",
        default=None,
        help="Optional augmented JSON exports containing LLM_as_a_Judge. If omitted, matching files are auto-discovered under benchmark/outputs/llm_judge_augmented_outputs.",
    )
    parser.add_argument(
        "--semantic-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model used for the MiniLM similarity columns.",
    )
    return parser.parse_args()


def coerce_score(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def coerce_selected_index(value, final_plans: List[str]) -> int | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                value = int(text)
            except ValueError:
                value = None
    elif isinstance(value, float) and value.is_integer():
        value = int(value)

    if isinstance(value, int) and 0 <= value < len(final_plans):
        return value
    if final_plans:
        return 0
    return None


def iter_input_exports(input_paths: Iterable[Path]):
    for input_path in input_paths:
        yield from iter_export_records(input_path)


def weak_session_key(record: Dict) -> tuple[str, str, str]:
    return (
        str(record["participant_id"]).strip(),
        str(record["session_number"]).strip(),
        str(record["interaction_condition"]).strip(),
    )


def load_session_records(input_paths: Iterable[Path]) -> List[Dict]:
    seen_sessions: Dict[str, Dict] = {}
    for export_record in iter_input_exports(input_paths):
        payload = export_record["payload"]
        export_id = export_record["export_id"]
        for session_key in SESSION_KEYS:
            session = payload.get(session_key)
            if not isinstance(session, dict):
                continue
            interaction_condition = normalize_condition(session.get("interactionCondition"))
            if interaction_condition not in CONDITION_ORDER:
                continue
            session_uid = f"{export_id}:{session_key}"
            final_plans = clean_text_list(session.get("User_Plans_Final"))
            selected_index = coerce_selected_index(session.get("Final_Selected_PlanNumber"), final_plans)
            record = {
                "session_uid": session_uid,
                "export_id": export_id,
                "source_file": export_record["source_file"],
                "session_key": session_key,
                "participant_id": str(session.get("participandID", "")).strip(),
                "session_number": session.get("session", ""),
                "interaction_condition": interaction_condition,
                "user_goal": normalize_text(session.get("user_goal", "")),
                "scenario": normalize_text(session.get("scenario", "")),
                "areas_initial": session.get("Actionabl_Areas_Initial") if isinstance(session.get("Actionabl_Areas_Initial"), list) else [],
                "plans_initial_raw": session.get("User_Plans_Initial") if isinstance(session.get("User_Plans_Initial"), list) else [],
                "plans_final_raw": session.get("User_Plans_Final") if isinstance(session.get("User_Plans_Final"), list) else [],
                "initial_plans": clean_text_list(session.get("User_Plans_Initial")),
                "final_plans": final_plans,
                "selected_index": selected_index,
                "model_name": str(session.get("llmModel", "") or ""),
                "interactionStartedAt": session.get("interactionStartedAt", ""),
                "initialStartedAt": session.get("initialStartedAt", ""),
                "initialSubmittedAt": session.get("initialSubmittedAt", ""),
                "outputsDisplayedAt": session.get("outputsDisplayedAt", ""),
                "ratingSubmittedAt": session.get("ratingSubmittedAt", ""),
                "finalStartedAt": session.get("finalStartedAt", ""),
                "finalSubmittedAt": session.get("finalSubmittedAt", ""),
                "interactionEndedAt": session.get("interactionEndedAt", ""),
                "feedback": payload.get("feedback") if isinstance(payload.get("feedback"), dict) else {},
                "ai_suggestions": session.get("AI_Suggestions") if isinstance(session.get("AI_Suggestions"), dict) else {},
            }
            if session_uid in seen_sessions:
                if seen_sessions[session_uid] != record:
                    raise ValueError(f"Conflicting duplicate export for {session_uid}.")
                continue
            seen_sessions[session_uid] = record
    return sorted(
        seen_sessions.values(),
        key=lambda row: (row["interaction_condition"], row["participant_id"], str(row["session_number"])),
    )


def resolve_judge_inputs(raw_inputs: Iterable[Path], judge_inputs: Iterable[Path] | None) -> List[Path]:
    if judge_inputs:
        return [Path(path) for path in judge_inputs]

    discovered: List[Path] = []
    for raw_input in raw_inputs:
        if raw_input.is_file():
            pattern = raw_input.name
            matches = sorted(
                DEFAULT_JUDGE_OUTPUT_ROOT.glob(f"*/augmented_json/{pattern}"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if matches:
                discovered.append(matches[0])
            continue
        if raw_input.is_dir():
            for raw_file in sorted(raw_input.glob("*.json")):
                matches = sorted(
                    DEFAULT_JUDGE_OUTPUT_ROOT.glob(f"*/augmented_json/{raw_file.name}"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if matches:
                    discovered.append(matches[0])
    return sorted(dict.fromkeys(path.resolve() for path in discovered))


def merge_judge_fields(session_records: Iterable[Dict], judge_records: Iterable[Dict]) -> List[Dict]:
    merged_records = []
    judge_by_key: Dict[tuple[str, str, str], Dict] = {}
    for judge_record in judge_records:
        key = weak_session_key(judge_record)
        if key in judge_by_key and judge_by_key[key]["ai_suggestions"] != judge_record["ai_suggestions"]:
            raise ValueError(f"Conflicting augmented judge data for session key {key}.")
        judge_by_key[key] = judge_record

    for record in session_records:
        merged = dict(record)
        merged["ai_suggestions"] = dict(record["ai_suggestions"])
        judge_record = judge_by_key.get(weak_session_key(record))
        if judge_record:
            for model in LEGACY_MODEL_ORDER:
                base_suggestion = dict(merged["ai_suggestions"].get(model) or {})
                judge_suggestion = dict((judge_record["ai_suggestions"].get(model) or {}))
                if (
                    "LLM_as_a_Judge" not in base_suggestion
                    and isinstance(judge_suggestion.get("LLM_as_a_Judge"), dict)
                ):
                    base_suggestion["LLM_as_a_Judge"] = judge_suggestion["LLM_as_a_Judge"]
                if not base_suggestion.get("item") and judge_suggestion.get("item"):
                    base_suggestion["item"] = judge_suggestion["item"]
                if not base_suggestion.get("ratings") and judge_suggestion.get("ratings"):
                    base_suggestion["ratings"] = judge_suggestion["ratings"]
                merged["ai_suggestions"][model] = base_suggestion
        merged_records.append(merged)
    return merged_records


def collect_unique_texts(session_records: Iterable[Dict]) -> List[str]:
    texts: List[str] = []
    for record in session_records:
        texts.extend(record["initial_plans"])
        texts.extend(record["final_plans"])
        for model in LEGACY_MODEL_ORDER:
            texts.append(normalize_text(((record["ai_suggestions"].get(model) or {}).get("item", ""))))
    unique_texts: List[str] = []
    seen = set()
    for text in texts:
        if not text or text in seen:
            continue
        unique_texts.append(text)
        seen.add(text)
    if not unique_texts:
        raise ValueError("No plan text was found in the JSON exports.")
    return unique_texts


def ai_vector_for_text(text: str, record: Dict, lookup: Dict[str, int], embeddings: np.ndarray) -> List[float]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return [float("nan")] * len(LEGACY_MODEL_ORDER)
    return [
        cosine_similarity_for(
            normalized_text,
            normalize_text((record["ai_suggestions"].get(model_name) or {}).get("item", "")),
            lookup,
            embeddings,
        )
        for model_name in LEGACY_MODEL_ORDER
    ]


def ai_matrix(texts: Iterable[str], record: Dict, lookup: Dict[str, int], embeddings: np.ndarray) -> str:
    score_rows = [ai_vector_for_text(text, record, lookup, embeddings) for text in texts if normalize_text(text)]
    return format_score_matrix(score_rows)


def build_vector_lookup(session_records: Iterable[Dict], semantic_model: str):
    unique_texts = collect_unique_texts(session_records)
    lexical_lookup, lexical_embeddings = build_lexical_space(unique_texts)
    semantic_lookup, semantic_embeddings = build_semantic_space(unique_texts, semantic_model)
    return lexical_lookup, lexical_embeddings, semantic_lookup, semantic_embeddings


def build_legacy_cleaned_dataframe(session_records: Iterable[Dict], semantic_model: str) -> pd.DataFrame:
    records = list(session_records)
    lexical_lookup, lexical_embeddings, semantic_lookup, semantic_embeddings = build_vector_lookup(records, semantic_model)
    rows: List[Dict] = []

    for record in records:
        lexical_final_matrix = [
            ai_vector_for_text(text, record, lexical_lookup, lexical_embeddings)
            for text in record["final_plans"]
            if normalize_text(text)
        ]
        semantic_final_matrix = [
            ai_vector_for_text(text, record, semantic_lookup, semantic_embeddings)
            for text in record["final_plans"]
            if normalize_text(text)
        ]
        selected_index = record["selected_index"]
        selected_lexical_scores = (
            lexical_final_matrix[selected_index]
            if isinstance(selected_index, int) and 0 <= selected_index < len(lexical_final_matrix)
            else [float("nan")] * len(LEGACY_MODEL_ORDER)
        )
        selected_semantic_scores = (
            semantic_final_matrix[selected_index]
            if isinstance(selected_index, int) and 0 <= selected_index < len(semantic_final_matrix)
            else [float("nan")] * len(LEGACY_MODEL_ORDER)
        )
        final_plan_text = (
            record["final_plans"][selected_index]
            if isinstance(selected_index, int) and 0 <= selected_index < len(record["final_plans"])
            else ""
        )
        initial_tfidf_matrix = (
            ai_matrix(record["initial_plans"], record, lexical_lookup, lexical_embeddings)
            if record["interaction_condition"] == "HAI"
            else ""
        )
        initial_minilm_matrix = (
            ai_matrix(record["initial_plans"], record, semantic_lookup, semantic_embeddings)
            if record["interaction_condition"] == "HAI"
            else ""
        )

        for model in LEGACY_MODEL_ORDER:
            suggestion = record["ai_suggestions"].get(model) or {}
            ratings = suggestion.get("ratings") if isinstance(suggestion.get("ratings"), dict) else {}
            llm_judge = (
                suggestion.get("LLM_as_a_Judge")
                if isinstance(suggestion.get("LLM_as_a_Judge"), dict)
                else {}
            )
            row = {
                "participant_id": record["participant_id"],
                "session": record["session_number"],
                "interactionCondition": record["interaction_condition"],
                "user_goal": record["user_goal"],
                "scenario": record["scenario"],
                "areas_initial": record["areas_initial"],
                "plans_initial": record["plans_initial_raw"],
                "plans_final": record["plans_final_raw"],
                "final_plan_id": selected_index,
                "model_name": record["model_name"],
                "interactionStartedAt": record["interactionStartedAt"],
                "initialStartedAt": record["initialStartedAt"],
                "initialSubmittedAt": record["initialSubmittedAt"],
                "outputsDisplayedAt": record["outputsDisplayedAt"],
                "ratingSubmittedAt": record["ratingSubmittedAt"],
                "finalStartedAt": record["finalStartedAt"],
                "finalSubmittedAt": record["finalSubmittedAt"],
                "interactionEndedAt": record["interactionEndedAt"],
                "feedback": record["feedback"],
                "model": model,
                "suggestion": normalize_text(suggestion.get("item", "")),
                "final_plan_text": final_plan_text,
                "selected_index": selected_index,
                "initial_plan_TFIDF_source": initial_tfidf_matrix,
                "initial_plan_MiniLM_source": initial_minilm_matrix,
                "final_candidate_TFIDF_source": format_score_matrix(lexical_final_matrix),
                "final_candidate_MiniLM_source": format_score_matrix(semantic_final_matrix),
                "final_decision_score": format_score_matrix([selected_lexical_scores, selected_semantic_scores]),
            }
            for source_key, column_name in HUMAN_RATING_COLUMNS.items():
                row[column_name] = coerce_score(ratings.get(source_key))
            for source_key, column_name in LLM_JUDGE_COLUMNS.items():
                row[column_name] = coerce_score(llm_judge.get(source_key))
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df[CSV_COLUMNS].copy()
    df["model"] = pd.Categorical(df["model"], categories=LEGACY_MODEL_ORDER, ordered=True)
    return df.sort_values(
        by=["interactionCondition", "participant_id", "session", "model"],
        kind="stable",
    ).reset_index(drop=True)


def parse_matrix(value):
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=float)
    return np.asarray(value, dtype=float)


def quadrant(tfidf_score: float, minilm_score: float) -> str:
    if tfidf_score >= THRESH_TFIDF and minilm_score >= THRESH_MINILM:
        return "Direct adoption (H/H)"
    if tfidf_score >= THRESH_TFIDF and minilm_score < THRESH_MINILM:
        return "Surface mimicry (H/L)"
    if tfidf_score < THRESH_TFIDF and minilm_score >= THRESH_MINILM:
        return "Semantic adoption (L/H)"
    return "Independent (L/L)"


def prepare_counterfactual_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["fc_minilm"] = prepared["final_candidate_MiniLM_source"].apply(parse_matrix)
    prepared["fc_tfidf"] = prepared["final_candidate_TFIDF_source"].apply(parse_matrix)
    prepared["fd_score"] = prepared["final_decision_score"].apply(parse_matrix)
    prepared["ip_minilm"] = prepared["initial_plan_MiniLM_source"].apply(parse_matrix)
    prepared["ip_tfidf"] = prepared["initial_plan_TFIDF_source"].apply(parse_matrix)

    def own_fc_tfidf(row):
        matrix = row["fc_tfidf"]
        return float(np.nanmax(matrix[:, LEGACY_MODEL_IDX[row["model"]]])) if matrix is not None and matrix.size else np.nan

    def own_fc_minilm(row):
        matrix = row["fc_minilm"]
        return float(np.nanmax(matrix[:, LEGACY_MODEL_IDX[row["model"]]])) if matrix is not None and matrix.size else np.nan

    def own_fd_tfidf(row):
        matrix = row["fd_score"]
        return float(matrix[0, LEGACY_MODEL_IDX[row["model"]]]) if matrix is not None and matrix.shape[0] >= 1 else np.nan

    def own_fd_minilm(row):
        matrix = row["fd_score"]
        return float(matrix[1, LEGACY_MODEL_IDX[row["model"]]]) if matrix is not None and matrix.shape[0] >= 2 else np.nan

    prepared["own_fc_tfidf"] = prepared.apply(own_fc_tfidf, axis=1)
    prepared["own_fc_minilm"] = prepared.apply(own_fc_minilm, axis=1)
    prepared["own_fd_tfidf"] = prepared.apply(own_fd_tfidf, axis=1)
    prepared["own_fd_minilm"] = prepared.apply(own_fd_minilm, axis=1)
    prepared["fc_quadrant"] = prepared.apply(
        lambda row: quadrant(row["own_fc_tfidf"], row["own_fc_minilm"]),
        axis=1,
    )
    prepared["fd_quadrant"] = prepared.apply(
        lambda row: quadrant(row["own_fd_tfidf"], row["own_fd_minilm"]),
        axis=1,
    )
    prepared["L1_adopted_minilm"] = prepared["own_fc_minilm"] >= THRESH_MINILM
    prepared["L1_adopted_tfidf"] = prepared["own_fc_tfidf"] >= THRESH_TFIDF
    prepared["L1_adopted_both"] = prepared["L1_adopted_minilm"] & prepared["L1_adopted_tfidf"]
    prepared["L2_adopted_minilm"] = prepared["own_fd_minilm"] >= THRESH_MINILM
    prepared["L2_adopted_tfidf"] = prepared["own_fd_tfidf"] >= THRESH_TFIDF
    prepared["L2_adopted_both"] = prepared["L2_adopted_minilm"] & prepared["L2_adopted_tfidf"]
    return prepared


def main() -> None:
    args = parse_args()
    session_records = load_session_records(args.inputs)
    judge_inputs = resolve_judge_inputs(args.inputs, args.judge_inputs)
    if judge_inputs:
        judge_records = load_session_records(judge_inputs)
        session_records = merge_judge_fields(session_records, judge_records)
    df = build_legacy_cleaned_dataframe(session_records, args.semantic_model)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    prepared = prepare_counterfactual_dataframe(df)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {args.output_csv}")
    print(f"Conditions: {df['interactionCondition'].value_counts().to_dict()}")
    print(f"Models: {df['model'].value_counts().to_dict()}")
    print(
        "LLM judge non-null counts: "
        f"{df[list(LLM_JUDGE_COLUMNS.values())].notna().sum().to_dict()}"
    )
    print(f"Prepared FC quadrant counts: {prepared['fc_quadrant'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
