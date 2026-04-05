from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Keep transformers on the PyTorch path in environments with incompatible
# TensorFlow or Flax installs.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

# Transformers >=4.57 expects the public pytree registration API, but the
# current environment exposes only the older private helper on torch 2.1.x.
if (
    hasattr(torch.utils, "_pytree")
    and not hasattr(torch.utils._pytree, "register_pytree_node")
    and hasattr(torch.utils._pytree, "_register_pytree_node")
):
    def _compat_register_pytree_node(
        typ,
        flatten_fn,
        unflatten_fn,
        *,
        serialized_type_name=None,
        to_dumpable_context=None,
        from_dumpable_context=None,
    ):
        del serialized_type_name
        return torch.utils._pytree._register_pytree_node(
            typ,
            flatten_fn,
            unflatten_fn,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context,
        )

    torch.utils._pytree.register_pytree_node = _compat_register_pytree_node

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.json_export_loader import (
    CONDITION_ORDER,
    SESSION_KEYS,
    iter_export_records,
    normalize_condition,
)


MODEL_ORDER = ["LLM-B", "LLM-CF", "LLM-CT"]
DEFAULT_THRESHOLD = 0.75
DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"
SENSITIVITY_THRESHOLDS = [0.70, 0.75, 0.80]
WEAK_PLAN_SESSION_KEYS = {
    ("16015", "AIH", 3),
    ("22716", "HAI", 3),
    ("31629", "AIH", 2),
    ("31629", "AIH", 3),
    ("32393", "HAI", 1),
    ("35515", "AIH", 1),
    ("35515", "AIH", 2),
    ("35665", "AIH", 1),
    ("35665", "AIH", 3),
    ("46079", "HAI", 1),
    ("52175", "AIH", 2),
    ("52175", "AIH", 3),
    ("61579", "HAI", 2),
    ("94604", "HAI", 1),
    ("94604", "HAI", 2),
}


@dataclass(frozen=True)
class SimilaritySpace:
    name: str
    description: str
    threshold: float
    lookup: Dict[str, int]
    embeddings: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing exported session JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where analysis outputs will be written.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Default source-attribution similarity threshold for both metrics. Default: %(default)s.",
    )
    parser.add_argument(
        "--lexical-threshold",
        type=float,
        default=None,
        help="Optional lexical threshold override. Defaults to --match-threshold.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=None,
        help="Optional semantic threshold override. Defaults to --match-threshold.",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default=DEFAULT_SEMANTIC_MODEL,
        help="SentenceTransformer model name for semantic similarity. Default: %(default)s.",
    )
    return parser.parse_args()


def normalize_text(value) -> str:
    text = str(value or "")
    return re.sub(r"\s+", " ", text).strip()


def clean_text_list(values) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    seen = set()
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned


def load_unique_sessions(input_path: Path) -> List[Dict]:
    seen_sessions: Dict[str, Dict] = {}
    for export_record in iter_export_records(input_path):
        payload = export_record["payload"]
        export_id = export_record["export_id"]
        for session_key in SESSION_KEYS:
            session = payload.get(session_key)
            if not isinstance(session, dict):
                continue
            interaction_condition = normalize_condition(session.get("interactionCondition"))
            if interaction_condition not in CONDITION_ORDER:
                continue
            weak_plan_key = (
                str(session.get("participandID", "")).strip(),
                interaction_condition,
                session.get("session"),
            )
            if weak_plan_key in WEAK_PLAN_SESSION_KEYS:
                continue
            session_uid = f"{export_id}:{session_key}"
            record = {
                "session_uid": session_uid,
                "export_id": export_id,
                "source_file": export_record["source_file"],
                "session_key": session_key,
                "participant_id": str(session.get("participandID", "")).strip(),
                "session_number": session.get("session", ""),
                "interaction_condition": interaction_condition,
                "scenario": normalize_text(session.get("scenario", "")),
                "user_goal": normalize_text(session.get("user_goal", "")),
                "initial_plans": clean_text_list(session.get("User_Plans_Initial")),
                "final_plans": clean_text_list(session.get("User_Plans_Final")),
                "selected_index": session.get("Final_Selected_PlanNumber"),
                "ai_suggestions": {
                    model: normalize_text(((session.get("AI_Suggestions") or {}).get(model) or {}).get("item", ""))
                    for model in MODEL_ORDER
                },
            }
            if session_uid in seen_sessions:
                if seen_sessions[session_uid] != record:
                    raise ValueError(f"Conflicting duplicate session payload found for {session_uid}.")
                continue
            seen_sessions[session_uid] = record
    return sorted(
        seen_sessions.values(),
        key=lambda row: (row["interaction_condition"], row["participant_id"], str(row["session_number"])),
    )


def collect_unique_texts(session_records: Iterable[Dict]) -> List[str]:
    texts: List[str] = []
    for record in session_records:
        texts.extend(record["initial_plans"])
        texts.extend(record["final_plans"])
        texts.extend([text for text in record["ai_suggestions"].values() if text])

    unique_texts = list(dict.fromkeys([text for text in texts if text]))
    if not unique_texts:
        raise ValueError("No plan text was found in the JSON exports.")
    return unique_texts


def build_lexical_space(unique_texts: List[str]) -> Tuple[Dict[str, int], np.ndarray]:
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = tfidf.fit_transform(unique_texts)
    max_components = min(100, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    n_components = max(2, max_components)
    svd = TruncatedSVD(n_components=n_components, random_state=20260328)
    dense = svd.fit_transform(tfidf_matrix)
    embeddings = Normalizer(copy=False).fit_transform(dense)
    lookup = {text: idx for idx, text in enumerate(unique_texts)}
    return lookup, embeddings


def build_semantic_space(unique_texts: List[str], model_name: str) -> Tuple[Dict[str, int], np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency availability is environment-specific
        raise ModuleNotFoundError(
            "sentence-transformers is required for semantic similarity. "
            "Install it with `python -m pip install sentence-transformers`."
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        unique_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    lookup = {text: idx for idx, text in enumerate(unique_texts)}
    return lookup, np.asarray(embeddings, dtype=np.float32)


def cosine_similarity_for(text_a: str, text_b: str, lookup: Dict[str, int], embeddings: np.ndarray) -> float:
    idx_a = lookup.get(text_a)
    idx_b = lookup.get(text_b)
    if idx_a is None or idx_b is None:
        return float("nan")
    return float(np.dot(embeddings[idx_a], embeddings[idx_b]))


def compute_list_diversity(texts: List[str], lookup: Dict[str, int], embeddings: np.ndarray) -> float:
    if len(texts) < 2:
        return 0.0
    distances = []
    for idx_a in range(len(texts)):
        for idx_b in range(idx_a + 1, len(texts)):
            similarity = cosine_similarity_for(texts[idx_a], texts[idx_b], lookup, embeddings)
            if math.isnan(similarity):
                continue
            distances.append(max(0.0, 1.0 - similarity))
    if not distances:
        return 0.0
    return float(np.mean(distances))


def build_diversity_rows(
    session_records: Iterable[Dict],
    metric_name: str,
    lookup: Dict[str, int],
    embeddings: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for record in session_records:
        final_plans = record["final_plans"]
        initial_plans = record["initial_plans"]
        rows.append(
            {
                "session_uid": record["session_uid"],
                "participant_id": record["participant_id"],
                "session_number": record["session_number"],
                "interaction_condition": record["interaction_condition"],
                "similarity_metric": metric_name,
                "n_initial_plans": len(initial_plans),
                "n_final_plans": len(final_plans),
                "initial_plan_diversity": compute_list_diversity(initial_plans, lookup, embeddings),
                "final_plan_diversity": compute_list_diversity(final_plans, lookup, embeddings),
            }
        )
    return pd.DataFrame(rows)


def summarize_diversity(diversity_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_name, metric_df in diversity_df.groupby("similarity_metric", dropna=False):
        subsets = [
            ("final_plans_pooled", metric_df, "final_plan_diversity", "n_final_plans"),
            ("final_plans_HAI", metric_df[metric_df["interaction_condition"] == "HAI"], "final_plan_diversity", "n_final_plans"),
            ("final_plans_AIH", metric_df[metric_df["interaction_condition"] == "AIH"], "final_plan_diversity", "n_final_plans"),
            (
                "initial_plans_HAI",
                metric_df[metric_df["interaction_condition"] == "HAI"],
                "initial_plan_diversity",
                "n_initial_plans",
            ),
        ]
        for label, subset, score_col, count_col in subsets:
            if subset.empty:
                continue
            multi_item = subset[subset[count_col] >= 2]
            rows.append(
                {
                    "similarity_metric": metric_name,
                    "analysis_group": label,
                    "n_sessions": int(len(subset)),
                    "n_sessions_with_2plus_items": int(len(multi_item)),
                    "mean_diversity_all_sessions": float(subset[score_col].mean()),
                    "sd_diversity_all_sessions": float(subset[score_col].std(ddof=1)) if len(subset) > 1 else float("nan"),
                    "mean_diversity_2plus_only": float(multi_item[score_col].mean()) if not multi_item.empty else float("nan"),
                    "median_diversity_2plus_only": float(multi_item[score_col].median()) if not multi_item.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def source_priority(source_type: str, source_label: str) -> Tuple[int, int]:
    type_rank = 0 if source_type == "initial" else 1
    model_rank = MODEL_ORDER.index(source_label) if source_label in MODEL_ORDER else -1
    return type_rank, model_rank


def classify_candidate_source(
    candidate_text: str,
    record: Dict,
    lookup: Dict[str, int],
    embeddings: np.ndarray,
    threshold: float,
) -> Dict:
    candidate_text = normalize_text(candidate_text)
    candidates = []

    if record["interaction_condition"] == "HAI":
        for initial_index, source_text in enumerate(record["initial_plans"]):
            similarity = cosine_similarity_for(candidate_text, source_text, lookup, embeddings)
            if math.isnan(similarity):
                continue
            candidates.append(
                {
                    "source_type": "initial",
                    "source_label": "initial",
                    "source_index": initial_index,
                    "source_text": source_text,
                    "similarity": similarity,
                }
            )

    for model_name in MODEL_ORDER:
        source_text = record["ai_suggestions"].get(model_name, "")
        if not source_text:
            continue
        similarity = cosine_similarity_for(candidate_text, source_text, lookup, embeddings)
        if math.isnan(similarity):
            continue
        candidates.append(
            {
                "source_type": "ai",
                "source_label": model_name,
                "source_index": None,
                "source_text": source_text,
                "similarity": similarity,
            }
        )

    if not candidates:
        return {
            "best_match_similarity": float("nan"),
            "best_match_source_type": "new",
            "best_match_source_label": "new",
            "best_match_source_index": None,
            "best_match_source_text": "",
            "source_family": "new",
            "source_label": "new",
        }

    best_match = max(
        candidates,
        key=lambda item: (item["similarity"], -source_priority(item["source_type"], item["source_label"])[0], -source_priority(item["source_type"], item["source_label"])[1]),
    )

    if best_match["similarity"] < threshold:
        return {
            "best_match_similarity": float(best_match["similarity"]),
            "best_match_source_type": "new",
            "best_match_source_label": "new",
            "best_match_source_index": None,
            "best_match_source_text": "",
            "source_family": "new",
            "source_label": "new",
        }

    return {
        "best_match_similarity": float(best_match["similarity"]),
        "best_match_source_type": best_match["source_type"],
        "best_match_source_label": best_match["source_label"],
        "best_match_source_index": best_match["source_index"],
        "best_match_source_text": best_match["source_text"],
        "source_family": best_match["source_type"],
        "source_label": best_match["source_label"],
    }


def build_candidate_similarity_rows(
    session_records: Iterable[Dict],
    metric_name: str,
    lookup: Dict[str, int],
    embeddings: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for record in session_records:
        selected_index = record["selected_index"]
        sources = []
        if record["interaction_condition"] == "HAI":
            for initial_index, source_text in enumerate(record["initial_plans"]):
                sources.append(
                    {
                        "source_type": "initial",
                        "source_label": "initial",
                        "source_index": initial_index,
                        "source_text": source_text,
                    }
                )
        for model_name in MODEL_ORDER:
            source_text = record["ai_suggestions"].get(model_name, "")
            if not source_text:
                continue
            sources.append(
                {
                    "source_type": "ai",
                    "source_label": model_name,
                    "source_index": None,
                    "source_text": source_text,
                }
            )

        for candidate_index, candidate_text in enumerate(record["final_plans"]):
            for source in sources:
                similarity = cosine_similarity_for(candidate_text, source["source_text"], lookup, embeddings)
                rows.append(
                    {
                        "session_uid": record["session_uid"],
                        "participant_id": record["participant_id"],
                        "session_number": record["session_number"],
                        "interaction_condition": record["interaction_condition"],
                        "similarity_metric": metric_name,
                        "candidate_index": candidate_index,
                        "candidate_text": candidate_text,
                        "is_selected": bool(isinstance(selected_index, int) and selected_index == candidate_index),
                        "source_type": source["source_type"],
                        "source_label": source["source_label"],
                        "source_index": source["source_index"],
                        "source_text": source["source_text"],
                        "similarity": similarity,
                    }
                )
    return pd.DataFrame(rows)


def format_score_list(scores: List[float]) -> str:
    return json.dumps([None if math.isnan(score) else float(score) for score in scores])


def format_score_matrix(score_rows: List[List[float]]) -> str:
    return json.dumps(
        [
            [None if math.isnan(score) else float(score) for score in score_row]
            for score_row in score_rows
        ]
    )


def build_named_session_source_export(
    session_records: Iterable[Dict],
    spaces: List[SimilaritySpace],
) -> pd.DataFrame:
    space_by_name = {space.name: space for space in spaces}
    lexical_space = space_by_name["lexical"]
    semantic_space = space_by_name["semantic"]

    rows = []
    for record in session_records:
        selected_index = record["selected_index"]
        def ai_vector_for_text(text: str, space: SimilaritySpace) -> List[float]:
            normalized_text = normalize_text(text)
            if not normalized_text:
                return [float("nan")] * len(MODEL_ORDER)
            return [
                cosine_similarity_for(
                    normalized_text,
                    record["ai_suggestions"].get(model_name, ""),
                    space.lookup,
                    space.embeddings,
                )
                for model_name in MODEL_ORDER
            ]

        def ai_matrix(texts: Iterable[str], space: SimilaritySpace) -> str:
            score_rows = [ai_vector_for_text(text, space) for text in texts if normalize_text(text)]
            return format_score_matrix(score_rows)

        lexical_final_matrix = [
            ai_vector_for_text(text, lexical_space)
            for text in record["final_plans"]
            if normalize_text(text)
        ]
        semantic_final_matrix = [
            ai_vector_for_text(text, semantic_space)
            for text in record["final_plans"]
            if normalize_text(text)
        ]
        selected_lexical_scores = (
            lexical_final_matrix[selected_index]
            if isinstance(selected_index, int) and 0 <= selected_index < len(lexical_final_matrix)
            else [float("nan")] * len(MODEL_ORDER)
        )
        selected_semantic_scores = (
            semantic_final_matrix[selected_index]
            if isinstance(selected_index, int) and 0 <= selected_index < len(semantic_final_matrix)
            else [float("nan")] * len(MODEL_ORDER)
        )

        rows.append(
            {
                "session_uid": record["session_uid"],
                "participant_id": record["participant_id"],
                "session_number": record["session_number"],
                "interaction_condition": record["interaction_condition"],
                "selected_index": selected_index,
                "initial_plan_TFIDF_source": (
                    ai_matrix(record["initial_plans"], lexical_space)
                    if record["interaction_condition"] == "HAI"
                    else ""
                ),
                "initial_plan_MiniLM_source": (
                    ai_matrix(record["initial_plans"], semantic_space)
                    if record["interaction_condition"] == "HAI"
                    else ""
                ),
                "final_candidate_TFIDF_source": format_score_matrix(lexical_final_matrix),
                "final_candidate_MiniLM_source": format_score_matrix(semantic_final_matrix),
                "final_decision_score": format_score_matrix(
                    [selected_lexical_scores, selected_semantic_scores]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_candidate_attribution_rows(
    session_records: Iterable[Dict],
    metric_name: str,
    lookup: Dict[str, int],
    embeddings: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for record in session_records:
        selected_index = record["selected_index"]
        for candidate_index, candidate_text in enumerate(record["final_plans"]):
            attribution = classify_candidate_source(candidate_text, record, lookup, embeddings, threshold)
            rows.append(
                {
                    "session_uid": record["session_uid"],
                    "participant_id": record["participant_id"],
                    "session_number": record["session_number"],
                    "interaction_condition": record["interaction_condition"],
                    "similarity_metric": metric_name,
                    "candidate_index": candidate_index,
                    "candidate_text": candidate_text,
                    "is_selected": bool(isinstance(selected_index, int) and selected_index == candidate_index),
                    **attribution,
                }
            )
    return pd.DataFrame(rows)


def build_candidate_source_summary(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_name, metric_df in candidate_df.groupby("similarity_metric", dropna=False):
        for condition in ["ALL"] + CONDITION_ORDER:
            subset = metric_df if condition == "ALL" else metric_df[metric_df["interaction_condition"] == condition]
            if subset.empty:
                continue
            total = len(subset)
            ai_subset = subset[subset["source_family"] == "ai"]
            initial_subset = subset[subset["source_family"] == "initial"]
            new_subset = subset[subset["source_family"] == "new"]
            row = {
                "similarity_metric": metric_name,
                "interaction_condition": condition,
                "n_candidates": int(total),
                "n_from_ai": int(len(ai_subset)),
                "pct_from_ai": float(len(ai_subset) / total),
                "n_from_initial": int(len(initial_subset)),
                "pct_from_initial": float(len(initial_subset) / total),
                "n_new": int(len(new_subset)),
                "pct_new": float(len(new_subset) / total),
            }
            for model_name in MODEL_ORDER:
                model_count = int((subset["source_label"] == model_name).sum())
                row[f"n_{model_name.lower().replace('-', '_')}"] = model_count
                row[f"pct_{model_name.lower().replace('-', '_')}"] = float(model_count / total)
            rows.append(row)
    return pd.DataFrame(rows)


def build_selected_source_summary(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_name, metric_df in candidate_df.groupby("similarity_metric", dropna=False):
        selected_df = metric_df[metric_df["is_selected"]].copy()
        for condition in ["ALL"] + CONDITION_ORDER:
            subset = selected_df if condition == "ALL" else selected_df[selected_df["interaction_condition"] == condition]
            if subset.empty:
                continue
            total = len(subset)
            row = {
                "similarity_metric": metric_name,
                "interaction_condition": condition,
                "n_selected_plans": int(total),
                "n_selected_from_ai": int((subset["source_family"] == "ai").sum()),
                "pct_selected_from_ai": float((subset["source_family"] == "ai").mean()),
                "n_selected_from_initial": int((subset["source_family"] == "initial").sum()),
                "pct_selected_from_initial": float((subset["source_family"] == "initial").mean()),
                "n_selected_new": int((subset["source_family"] == "new").sum()),
                "pct_selected_new": float((subset["source_family"] == "new").mean()),
            }
            for model_name in MODEL_ORDER:
                mask = subset["source_label"] == model_name
                row[f"n_selected_{model_name.lower().replace('-', '_')}"] = int(mask.sum())
                row[f"pct_selected_{model_name.lower().replace('-', '_')}"] = float(mask.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_sensitivity(
    session_records: Iterable[Dict],
    spaces: List[SimilaritySpace],
    thresholds: List[float],
) -> pd.DataFrame:
    rows = []
    for space in spaces:
        for threshold in thresholds:
            candidate_df = build_candidate_attribution_rows(
                session_records,
                metric_name=space.name,
                lookup=space.lookup,
                embeddings=space.embeddings,
                threshold=threshold,
            )
            summary_df = build_candidate_source_summary(candidate_df)
            selected_df = build_selected_source_summary(candidate_df)
            for _, row in summary_df.iterrows():
                rows.append(
                    {
                        "threshold": threshold,
                        "summary_type": "candidate",
                        **row.to_dict(),
                    }
                )
            for _, row in selected_df.iterrows():
                rows.append(
                    {
                        "threshold": threshold,
                        "summary_type": "selected",
                        **row.to_dict(),
                    }
                )
    return pd.DataFrame(rows)


def fmt(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.3f}"


def write_markdown_summary(
    output_path: Path,
    diversity_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    spaces: List[SimilaritySpace],
) -> None:
    lines = [
        "# Plan Source Analysis",
        "",
        "- Similarity metrics are reported separately for lexical overlap and semantic alignment.",
        "- Lexical space: TF-IDF + truncated SVD (LSA), then cosine similarity",
        "- Semantic space: SentenceTransformer embeddings, then cosine similarity",
        "- Diversity score: mean pairwise distance within each plan list",
        "- `initial` source attribution is only available for `HAI`, because `AIH` has no substantive initial plan lists in the exports",
        "",
    ]
    for space in spaces:
        lines.extend(
            [
                f"## {space.name.title()} Similarity",
                "",
                f"- Description: {space.description}",
                f"- Match threshold: `{space.threshold:.2f}` cosine similarity",
                "",
                "### Diversity Summary",
                "",
            ]
        )
        for _, row in diversity_summary[diversity_summary["similarity_metric"] == space.name].iterrows():
            lines.extend(
                [
                    f"#### {row['analysis_group']}",
                    f"- Sessions: {fmt(row['n_sessions'])}",
                    f"- Sessions with 2+ plans: {fmt(row['n_sessions_with_2plus_items'])}",
                    f"- Mean diversity (all sessions): {fmt(row['mean_diversity_all_sessions'])}",
                    f"- Mean diversity (2+ plan sessions only): {fmt(row['mean_diversity_2plus_only'])}",
                    f"- Median diversity (2+ plan sessions only): {fmt(row['median_diversity_2plus_only'])}",
                    "",
                ]
            )

        lines.extend(["### Candidate Final Plan Sources", ""])
        for _, row in candidate_summary[candidate_summary["similarity_metric"] == space.name].iterrows():
            lines.extend(
                [
                    f"#### {row['interaction_condition']}",
                    f"- Candidates: {fmt(row['n_candidates'])}",
                    f"- From AI suggestions: {fmt(row['n_from_ai'])} ({row['pct_from_ai'] * 100:.1f}%)",
                    f"- From human initial plans: {fmt(row['n_from_initial'])} ({row['pct_from_initial'] * 100:.1f}%)",
                    f"- New: {fmt(row['n_new'])} ({row['pct_new'] * 100:.1f}%)",
                    f"- AI breakdown: LLM-B {row['pct_llm_b'] * 100:.1f}%, LLM-CF {row['pct_llm_cf'] * 100:.1f}%, LLM-CT {row['pct_llm_ct'] * 100:.1f}%",
                    "",
                ]
            )

        lines.extend(["### Selected Final Plan Sources", ""])
        for _, row in selected_summary[selected_summary["similarity_metric"] == space.name].iterrows():
            lines.extend(
                [
                    f"#### {row['interaction_condition']}",
                    f"- Selected plans: {fmt(row['n_selected_plans'])}",
                    f"- Selected from AI suggestions: {fmt(row['n_selected_from_ai'])} ({row['pct_selected_from_ai'] * 100:.1f}%)",
                    f"- Selected from human initial plans: {fmt(row['n_selected_from_initial'])} ({row['pct_selected_from_initial'] * 100:.1f}%)",
                    f"- Selected new: {fmt(row['n_selected_new'])} ({row['pct_selected_new'] * 100:.1f}%)",
                    f"- Selected AI breakdown: LLM-B {row['pct_selected_llm_b'] * 100:.1f}%, LLM-CF {row['pct_selected_llm_cf'] * 100:.1f}%, LLM-CT {row['pct_selected_llm_ct'] * 100:.1f}%",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    session_records = load_unique_sessions(args.input)
    unique_texts = collect_unique_texts(session_records)

    lexical_threshold = args.lexical_threshold if args.lexical_threshold is not None else args.match_threshold
    semantic_threshold = args.semantic_threshold if args.semantic_threshold is not None else args.match_threshold

    lexical_lookup, lexical_embeddings = build_lexical_space(unique_texts)
    semantic_lookup, semantic_embeddings = build_semantic_space(unique_texts, args.semantic_model)

    spaces = [
        SimilaritySpace(
            name="lexical",
            description="TF-IDF + truncated SVD (LSA), emphasizing shared terms and phrases.",
            threshold=lexical_threshold,
            lookup=lexical_lookup,
            embeddings=lexical_embeddings,
        ),
        SimilaritySpace(
            name="semantic",
            description=f"SentenceTransformer `{args.semantic_model}`, emphasizing paraphrase and action-level similarity.",
            threshold=semantic_threshold,
            lookup=semantic_lookup,
            embeddings=semantic_embeddings,
        ),
    ]

    diversity_frames = []
    similarity_frames = []
    candidate_frames = []
    for space in spaces:
        diversity_frames.append(build_diversity_rows(session_records, space.name, space.lookup, space.embeddings))
        similarity_frames.append(
            build_candidate_similarity_rows(
                session_records,
                metric_name=space.name,
                lookup=space.lookup,
                embeddings=space.embeddings,
            )
        )
        candidate_frames.append(
            build_candidate_attribution_rows(
                session_records,
                metric_name=space.name,
                lookup=space.lookup,
                embeddings=space.embeddings,
                threshold=space.threshold,
            )
        )

    diversity_df = pd.concat(diversity_frames, ignore_index=True)
    diversity_summary = summarize_diversity(diversity_df)
    candidate_similarity_df = pd.concat(similarity_frames, ignore_index=True)
    candidate_df = pd.concat(candidate_frames, ignore_index=True)
    named_session_source_df = build_named_session_source_export(session_records, spaces)
    candidate_summary = build_candidate_source_summary(candidate_df)
    selected_summary = build_selected_source_summary(candidate_df)
    sensitivity_df = build_threshold_sensitivity(session_records, spaces, SENSITIVITY_THRESHOLDS)

    session_metadata = pd.DataFrame(
        [
            {
                "session_uid": record["session_uid"],
                "participant_id": record["participant_id"],
                "session_number": record["session_number"],
                "interaction_condition": record["interaction_condition"],
                "n_initial_plans": len(record["initial_plans"]),
                "n_final_plans": len(record["final_plans"]),
                "selected_index": record["selected_index"],
            }
            for record in session_records
        ]
    )

    session_metadata.to_csv(output_dir / "session_metadata.csv", index=False)
    diversity_df.to_csv(output_dir / "diversity_session_scores.csv", index=False)
    diversity_summary.to_csv(output_dir / "diversity_summary.csv", index=False)
    candidate_similarity_df.to_csv(output_dir / "candidate_source_similarity_vectors.csv", index=False)
    named_session_source_df.to_csv(output_dir / "named_session_source_vectors.csv", index=False)
    candidate_df.to_csv(output_dir / "candidate_source_attribution.csv", index=False)
    candidate_summary.to_csv(output_dir / "candidate_source_summary.csv", index=False)
    selected_summary.to_csv(output_dir / "selected_source_summary.csv", index=False)
    sensitivity_df.to_csv(output_dir / "source_summary_threshold_sensitivity.csv", index=False)
    write_markdown_summary(
        output_dir / "analysis_summary.md",
        diversity_summary=diversity_summary,
        candidate_summary=candidate_summary,
        selected_summary=selected_summary,
        spaces=spaces,
    )


if __name__ == "__main__":
    main()
