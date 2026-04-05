from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.json_export_loader import (
    CONDITION_ORDER,
    SESSION_KEYS,
    iter_export_records,
    normalize_condition,
)


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "low"
OPENAI_MAX_RETRIES = 5
OPENAI_RETRY_BASE_SECONDS = 1.5
PROMPT_VERSION = "feedback-llm-judge-v1"

FEEDBACK_KEYS = ["1", "2", "3", "4"]
PROMPT_LABELS = {key: f"feedback_{key}" for key in FEEDBACK_KEYS}
RESPONSE_TYPE_ORDER = ["text", "url", "empty"]

PRIMARY_THEME_CHOICES = [
    "actionability_clarity",
    "new_perspectives",
    "repetition_derivative",
    "personalization_context_fit",
    "interactivity_iteration",
    "explanation_reasoning_support",
    "ui_presentation",
    "input_burden",
    "speed_adaptivity",
    "low_utility_or_skepticism",
    "other",
]
PRIMARY_THEME_LABELS = {
    "actionability_clarity": "Actionability / Clarity",
    "new_perspectives": "New Perspectives / Expansion",
    "repetition_derivative": "Repetition / Derivative Output",
    "personalization_context_fit": "Personalization / Context Fit",
    "interactivity_iteration": "Interactivity / Iteration",
    "explanation_reasoning_support": "Explanation / Reasoning Support",
    "ui_presentation": "UI / Presentation",
    "input_burden": "Input Burden / Friction",
    "speed_adaptivity": "Speed / Adaptivity",
    "low_utility_or_skepticism": "Low Utility / Skepticism",
    "other": "Other",
}

FACET_COLUMNS = [
    "requests_personalization",
    "requests_interactivity_or_iteration",
    "requests_explanation_or_reasoning_support",
    "requests_ui_or_presentation_changes",
    "requests_faster_or_realtime_support",
    "criticizes_repetition_or_derivative_output",
    "criticizes_input_burden",
    "expresses_low_utility_or_skepticism",
]
FACET_LABELS = {
    "requests_personalization": "Requests personalization",
    "requests_interactivity_or_iteration": "Requests interactivity / iteration",
    "requests_explanation_or_reasoning_support": "Requests explanation / reasoning support",
    "requests_ui_or_presentation_changes": "Requests UI / presentation changes",
    "requests_faster_or_realtime_support": "Requests faster / real-time support",
    "criticizes_repetition_or_derivative_output": "Criticizes repetition / derivative output",
    "criticizes_input_burden": "Criticizes input burden",
    "expresses_low_utility_or_skepticism": "Expresses low utility / skepticism",
}

ATTRIBUTE_COLUMNS = [
    "prompt_intent",
    "valence",
    "helpfulness_assessment",
    "novelty_expansion_assessment",
    "decision_influence_assessment",
]
ATTRIBUTE_LABELS = {
    "prompt_intent": "Prompt Intent",
    "valence": "Valence",
    "helpfulness_assessment": "Helpfulness",
    "novelty_expansion_assessment": "Novelty / Expansion",
    "decision_influence_assessment": "Decision Influence",
}

SYSTEM_PROMPT = """
You are an expert HCI qualitative analyst coding one participant feedback
response from a study about human-AI co-planning for counterfactual reflection
on past decisions.

Code ONLY what is explicit or strongly implied in the response text.
Do not assume missing survey wording. If uncertain, choose the conservative
option such as "unclear", "other", or false.

Codebook:

prompt_intent:
- general_reaction_or_usefulness: overall reaction, usefulness, likes/dislikes
- ai_influence_on_thinking_or_decisions: how AI suggestions changed thinking,
  options considered, or final decisions
- desired_future_system_features: desired features for an ideal future system
- other_or_unclear: none of the above or too ambiguous

primary_theme and secondary_themes:
- actionability_clarity: clear, practical, concrete, actionable, realistic
- new_perspectives: new ideas, different perspectives, bigger picture,
  expanded thinking
- repetition_derivative: repetitive, rephrased, same as user input,
  unoriginal, derivative
- personalization_context_fit: tailored, personalized, fits the user's specific
  situation or goals
- interactivity_iteration: dialogue, back-and-forth, refinement, asking
  questions, iteration, collaboration with the system
- explanation_reasoning_support: requests or values explanations, pros/cons,
  rationale, why a suggestion is good
- ui_presentation: formatting, interface, layout, bullet points, visual summary
- input_burden: too much writing, too much required input, too much effort
- speed_adaptivity: real-time, faster, adaptive, instant support
- low_utility_or_skepticism: not useful, not feasible, distrust, AI cannot help
- other: best-fit theme if none of the above fit well

Attribute scales:
- valence: positive, mixed, negative, neutral
- helpfulness_assessment: high, medium, low, unclear
- novelty_expansion_assessment: high, medium, low, unclear
- decision_influence_assessment: strong, some, none, unclear

Use short evidence spans copied from the response text. Keep the rationale brief.
"""


class FeedbackJudgment(BaseModel):
    prompt_intent: Literal[
        "general_reaction_or_usefulness",
        "ai_influence_on_thinking_or_decisions",
        "desired_future_system_features",
        "other_or_unclear",
    ]
    primary_theme: Literal[
        "actionability_clarity",
        "new_perspectives",
        "repetition_derivative",
        "personalization_context_fit",
        "interactivity_iteration",
        "explanation_reasoning_support",
        "ui_presentation",
        "input_burden",
        "speed_adaptivity",
        "low_utility_or_skepticism",
        "other",
    ]
    secondary_themes: List[
        Literal[
            "actionability_clarity",
            "new_perspectives",
            "repetition_derivative",
            "personalization_context_fit",
            "interactivity_iteration",
            "explanation_reasoning_support",
            "ui_presentation",
            "input_burden",
            "speed_adaptivity",
            "low_utility_or_skepticism",
            "other",
        ]
    ] = Field(default_factory=list, description="Zero to three additional themes.")
    valence: Literal["positive", "mixed", "negative", "neutral"]
    helpfulness_assessment: Literal["high", "medium", "low", "unclear"]
    novelty_expansion_assessment: Literal["high", "medium", "low", "unclear"]
    decision_influence_assessment: Literal["strong", "some", "none", "unclear"]
    requests_personalization: bool
    requests_interactivity_or_iteration: bool
    requests_explanation_or_reasoning_support: bool
    requests_ui_or_presentation_changes: bool
    requests_faster_or_realtime_support: bool
    criticizes_repetition_or_derivative_output: bool
    criticizes_input_burden: bool
    expresses_low_utility_or_skepticism: bool
    evidence_span: str = Field(description="Short verbatim span copied from the response, at most about 20 words.")
    brief_rationale: str = Field(description="Brief justification, under 30 words.")


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
        help="Directory where LLM-judge outputs will be written.",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="Optional OpenAI API key override. Falls back to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI judge model to use.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for the OpenAI judge.",
    )
    parser.add_argument(
        "--judge-prompts",
        nargs="+",
        default=["1", "2", "3"],
        choices=FEEDBACK_KEYS,
        help="Feedback prompt ids to judge. Prompt 4 is usually a URL, so it is excluded by default.",
    )
    parser.add_argument(
        "--condition",
        choices=CONDITION_ORDER,
        default=None,
        help="Optional condition filter. Use this to judge HAI and AIH separately.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of analyzable responses to judge.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rejudge responses even if cached JSONL judgments already exist.",
    )
    return parser.parse_args()


def load_openai_client(api_key: str) -> OpenAI:
    resolved_api_key = " ".join(str(api_key or os.environ.get("OPENAI_API_KEY", "")).split()).strip()
    if not resolved_api_key:
        raise ValueError(
            "feedback_llm_judge.py requires an OpenAI API key. Set OPENAI_API_KEY or pass --openai-api-key."
        )
    return OpenAI(api_key=resolved_api_key)


def sleep_before_retry(attempt: int, exc: Exception) -> None:
    wait_seconds = OPENAI_RETRY_BASE_SECONDS * (2 ** attempt)
    print(f"  Retry {attempt + 1}/{OPENAI_MAX_RETRIES}: {exc} (sleep {wait_seconds:.1f}s)")
    time.sleep(wait_seconds)


def first_session(payload: Dict) -> Dict | None:
    for session_key in SESSION_KEYS:
        session = payload.get(session_key)
        if isinstance(session, dict):
            return session
    return None


def load_unique_exports(input_dir: Path) -> Tuple[List[Dict], pd.DataFrame]:
    exports: List[Dict] = []
    export_rows: List[Dict] = []
    seen_exports: Dict[str, Dict] = {}
    duplicate_count = 0
    source_files = set()
    raw_export_count = 0

    for export_record in iter_export_records(input_dir):
        payload = export_record["payload"]
        export_id = export_record["export_id"]
        source_files.add(export_record["source_file"])
        raw_export_count += 1
        record = {
            "export_id": export_id,
            "source_file": export_record["source_file"],
            "payload": payload,
        }

        if export_id in seen_exports:
            if seen_exports[export_id]["payload"] != payload:
                raise ValueError(
                    f"Conflicting duplicate export for {export_id}: "
                    f"{seen_exports[export_id]['source_file']} vs {export_record['source_file']}"
                )
            duplicate_count += 1
            continue

        seen_exports[export_id] = record

    for record in seen_exports.values():
        payload = record["payload"]
        session = first_session(payload) or {}
        export_rows.append(
            {
                "export_id": record["export_id"],
                "source_file": record["source_file"],
                "participant_id": session.get("participandID", ""),
                "interaction_condition": normalize_condition(
                    payload.get("mode") or session.get("interactionCondition", "")
                ),
            }
        )
        exports.append(record)

    metadata = pd.DataFrame(export_rows).sort_values(by=["participant_id", "source_file"])
    metadata.attrs["num_source_files"] = len(source_files)
    metadata.attrs["num_raw_exports"] = raw_export_count
    metadata.attrs["duplicate_count"] = duplicate_count
    return exports, metadata


def classify_response(raw_value) -> Tuple[str, str, int, int]:
    text = str(raw_value or "").strip()
    if not text:
        return "empty", "", 0, 0
    if text.startswith("http://") or text.startswith("https://"):
        return "url", text, 0, len(text)
    word_count = len(re.findall(r"\b\w+\b", text))
    return "text", text, word_count, len(text)


def extract_feedback_long(exports: Iterable[Dict]) -> pd.DataFrame:
    rows = []
    for record in exports:
        payload = record["payload"]
        session = first_session(payload) or {}
        feedback = payload.get("feedback") or {}

        for prompt_id in FEEDBACK_KEYS:
            response_type, text, word_count, char_count = classify_response(feedback.get(prompt_id, ""))
            rows.append(
                {
                    "export_id": record["export_id"],
                    "source_file": record["source_file"],
                    "participant_id": session.get("participandID", ""),
                    "interaction_condition": payload.get("mode") or session.get("interactionCondition", ""),
                    "prompt_id": prompt_id,
                    "prompt_label": PROMPT_LABELS[prompt_id],
                    "response_type": response_type,
                    "text": text,
                    "word_count": word_count,
                    "char_count": char_count,
                }
            )

    df = pd.DataFrame(rows)
    df["prompt_id"] = pd.Categorical(df["prompt_id"], categories=FEEDBACK_KEYS, ordered=True)
    df["response_type"] = pd.Categorical(df["response_type"], categories=RESPONSE_TYPE_ORDER, ordered=True)
    return df.sort_values(by=["participant_id", "prompt_id"]).reset_index(drop=True)


def build_prompt_summary(feedback_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_id in FEEDBACK_KEYS:
        sub = feedback_df[feedback_df["prompt_id"] == prompt_id]
        type_counts = sub["response_type"].value_counts().to_dict()
        text_sub = sub[sub["response_type"] == "text"]
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_label": PROMPT_LABELS[prompt_id],
                "n_total": len(sub),
                "n_text": int(type_counts.get("text", 0)),
                "n_url": int(type_counts.get("url", 0)),
                "n_empty": int(type_counts.get("empty", 0)),
                "mean_word_count_text_only": float(text_sub["word_count"].mean()) if not text_sub.empty else 0.0,
                "median_word_count_text_only": float(text_sub["word_count"].median()) if not text_sub.empty else 0.0,
                "max_word_count_text_only": int(text_sub["word_count"].max()) if not text_sub.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def make_cache_key(row: pd.Series, model: str) -> str:
    text_hash = hashlib.sha1(str(row["text"]).encode("utf-8")).hexdigest()[:12]
    return f"{PROMPT_VERSION}:{model}:{row['export_id']}:{row['prompt_id']}:{text_hash}"


def load_existing_judgments(jsonl_path: Path) -> Dict[str, Dict]:
    cache: Dict[str, Dict] = {}
    if not jsonl_path.exists():
        return cache

    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[str(record["cache_key"])] = record
    return cache


def build_judge_input(row: pd.Series) -> str:
    payload = {
        "study_context": (
            "Participants used an AI system to reflect on a past event and plan "
            "alternative actions. The exact survey wording is unavailable."
        ),
        "prompt_id": str(row["prompt_id"]),
        "prompt_label": str(row["prompt_label"]),
        "response_text": str(row["text"]),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def judge_feedback_row(
    client: OpenAI,
    row: pd.Series,
    *,
    model: str,
    reasoning_effort: str,
) -> Dict:
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            response = client.responses.parse(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=build_judge_input(row),
                reasoning={"effort": reasoning_effort},
                text_format=FeedbackJudgment,
                max_output_tokens=700,
                store=False,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("The OpenAI response did not contain parsed structured output.")
            judgment = parsed.model_dump()
            return {
                "openai_response_id": response.id,
                "openai_model": model,
                "prompt_version": PROMPT_VERSION,
                **judgment,
            }
        except Exception as exc:  # pragma: no cover - network failures are external
            if attempt == OPENAI_MAX_RETRIES - 1:
                raise
            sleep_before_retry(attempt, exc)
    raise RuntimeError("Unexpected retry loop fallthrough.")


def run_llm_judge(
    feedback_df: pd.DataFrame,
    *,
    client: OpenAI,
    output_dir: Path,
    model: str,
    reasoning_effort: str,
    judge_prompts: List[str],
    limit: int,
    force: bool,
) -> pd.DataFrame:
    analyzable = feedback_df[
        (feedback_df["response_type"] == "text") & (feedback_df["prompt_id"].astype(str).isin(judge_prompts))
    ].copy()
    if limit > 0:
        analyzable = analyzable.head(limit).copy()

    jsonl_path = output_dir / "feedback_llm_judgments.jsonl"
    cache = {} if force else load_existing_judgments(jsonl_path)
    pending_records: List[Dict] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in analyzable.iterrows():
        cache_key = make_cache_key(row, model)
        if not force and cache_key in cache:
            continue

        judgment = judge_feedback_row(
            client,
            row,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        record = {
            "cache_key": cache_key,
            "export_id": row["export_id"],
            "participant_id": row["participant_id"],
            "interaction_condition": row["interaction_condition"],
            "prompt_id": str(row["prompt_id"]),
            "prompt_label": row["prompt_label"],
            "response_text": row["text"],
            **judgment,
        }
        pending_records.append(record)
        cache[cache_key] = record

        with jsonl_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    judged_rows = list(cache.values())
    judged_df = pd.DataFrame(judged_rows)
    if judged_df.empty:
        return judged_df

    judged_df["prompt_id"] = pd.Categorical(judged_df["prompt_id"], categories=judge_prompts, ordered=True)
    return judged_df.sort_values(by=["participant_id", "prompt_id"]).reset_index(drop=True)


def build_primary_theme_summary(judged_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_id in sorted(judged_df["prompt_id"].dropna().unique().tolist()):
        prompt_sub = judged_df[judged_df["prompt_id"] == prompt_id]
        denom = len(prompt_sub)
        for theme in PRIMARY_THEME_CHOICES:
            count = int((prompt_sub["primary_theme"] == theme).sum())
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_label": PROMPT_LABELS[str(prompt_id)],
                    "theme_name": theme,
                    "theme_label": PRIMARY_THEME_LABELS[theme],
                    "n_matches": count,
                    "prevalence": count / denom if denom else 0.0,
                    "n_prompt_responses": denom,
                }
            )
    return pd.DataFrame(rows)


def build_facet_summary(judged_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_id in sorted(judged_df["prompt_id"].dropna().unique().tolist()):
        prompt_sub = judged_df[judged_df["prompt_id"] == prompt_id]
        denom = len(prompt_sub)
        for column in FACET_COLUMNS:
            count = int(prompt_sub[column].sum())
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_label": PROMPT_LABELS[str(prompt_id)],
                    "facet_name": column,
                    "facet_label": FACET_LABELS[column],
                    "n_matches": count,
                    "prevalence": count / denom if denom else 0.0,
                    "n_prompt_responses": denom,
                }
            )
    return pd.DataFrame(rows)


def build_attribute_summary(judged_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_id in sorted(judged_df["prompt_id"].dropna().unique().tolist()):
        prompt_sub = judged_df[judged_df["prompt_id"] == prompt_id]
        denom = len(prompt_sub)
        for column in ATTRIBUTE_COLUMNS:
            counts = prompt_sub[column].value_counts().to_dict()
            for value, count in sorted(counts.items()):
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_label": PROMPT_LABELS[str(prompt_id)],
                        "attribute_name": column,
                        "attribute_label": ATTRIBUTE_LABELS[column],
                        "attribute_value": value,
                        "n_matches": int(count),
                        "prevalence": int(count) / denom if denom else 0.0,
                        "n_prompt_responses": denom,
                    }
                )
    return pd.DataFrame(rows)


def build_examples(judged_df: pd.DataFrame, max_examples: int = 2) -> pd.DataFrame:
    rows = []
    for prompt_id in sorted(judged_df["prompt_id"].dropna().unique().tolist()):
        prompt_sub = judged_df[judged_df["prompt_id"] == prompt_id]
        for theme in PRIMARY_THEME_CHOICES:
            matches = prompt_sub[prompt_sub["primary_theme"] == theme].head(max_examples)
            for _, row in matches.iterrows():
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_label": PROMPT_LABELS[str(prompt_id)],
                        "theme_name": theme,
                        "theme_label": PRIMARY_THEME_LABELS[theme],
                        "export_id": row["export_id"],
                        "participant_id": row["participant_id"],
                        "evidence_span": row["evidence_span"],
                        "brief_rationale": row["brief_rationale"],
                    }
                )
    return pd.DataFrame(rows)


def build_manual_coding_sheet(judged_df: pd.DataFrame) -> pd.DataFrame:
    coding = judged_df[
        [
            "export_id",
            "participant_id",
            "interaction_condition",
            "prompt_id",
            "prompt_label",
            "response_text",
            "primary_theme",
            "secondary_themes",
            "valence",
            "helpfulness_assessment",
            "novelty_expansion_assessment",
            "decision_influence_assessment",
            "evidence_span",
            "brief_rationale",
        ]
    ].copy()
    coding["secondary_themes"] = coding["secondary_themes"].apply(
        lambda value: "; ".join(value) if isinstance(value, list) else str(value)
    )
    coding["human_primary_code"] = ""
    coding["human_secondary_code"] = ""
    coding["human_valence"] = ""
    coding["quote_candidate"] = ""
    coding["analyst_memo"] = ""
    return coding


def build_summary_markdown(
    metadata: pd.DataFrame,
    feedback_df: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    judged_df: pd.DataFrame,
    primary_theme_summary: pd.DataFrame,
    facet_summary: pd.DataFrame,
    model: str,
) -> str:
    total_exports = metadata["export_id"].nunique()
    duplicates_removed = metadata.attrs.get("duplicate_count", 0)
    participant_counts = metadata["interaction_condition"].value_counts().to_dict()

    lines = [
        "# Feedback Analysis Summary",
        "",
        "## Dataset",
        f"- Source files scanned: {metadata.attrs.get('num_source_files', 'unknown')}",
        f"- Raw exports scanned: {metadata.attrs.get('num_raw_exports', 'unknown')}",
        f"- Condition filter: {metadata.attrs.get('condition_filter', 'all') or 'all'}",
        f"- Unique exports analyzed: {total_exports}",
        f"- Duplicate exports removed: {duplicates_removed}",
        f"- Participant counts by condition: {participant_counts}",
        f"- Feedback rows extracted: {len(feedback_df)}",
        f"- LLM-judged responses: {len(judged_df)}",
        f"- Judge model: {model}",
        "",
        "## Prompt Coverage",
    ]

    for _, row in prompt_summary.iterrows():
        lines.append(
            f"- {row['prompt_label']}: text={int(row['n_text'])}, url={int(row['n_url'])}, "
            f"empty={int(row['n_empty'])}, mean words={row['mean_word_count_text_only']:.1f}."
        )

    lines.extend(
        [
            "",
            "## LLM-Judge Highlights",
            "- These labels come from a structured GPT rubric and should still be checked by human coders before final paper claims.",
        ]
    )

    for prompt_id in sorted(judged_df["prompt_id"].dropna().unique().tolist()):
        theme_sub = primary_theme_summary[primary_theme_summary["prompt_id"] == prompt_id].sort_values(
            "prevalence", ascending=False
        )
        facet_sub = facet_summary[facet_summary["prompt_id"] == prompt_id].sort_values(
            "prevalence", ascending=False
        )
        top_themes = ", ".join(
            f"{row['theme_label']} ({row['prevalence'] * 100:.0f}%)"
            for _, row in theme_sub.head(3).iterrows()
            if row["n_matches"] > 0
        )
        top_facets = ", ".join(
            f"{row['facet_label']} ({row['prevalence'] * 100:.0f}%)"
            for _, row in facet_sub.head(3).iterrows()
            if row["n_matches"] > 0
        )
        lines.append(f"- {PROMPT_LABELS[str(prompt_id)]}: top primary themes = {top_themes or 'none'}; top facets = {top_facets or 'none'}.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- Prompt 4 is usually a URL field and is excluded from LLM judging by default.",
            "- Because the exact survey wording is not in the repo, the judge infers prompt intent from the response text itself.",
            "- Treat the LLM judgment as first-pass coding support, then validate with human coding for the paper.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_response_type_figure(prompt_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(FEEDBACK_KEYS))
    bottoms = np.zeros(len(FEEDBACK_KEYS))
    colors = {"text": "#2a9d8f", "url": "#e9c46a", "empty": "#bfc0c0"}

    for response_type in RESPONSE_TYPE_ORDER:
        counts = []
        for prompt_id in FEEDBACK_KEYS:
            row = prompt_summary[prompt_summary["prompt_id"] == prompt_id].iloc[0]
            counts.append(int(row[f"n_{response_type}"]))
        ax.bar(positions, counts, bottom=bottoms, color=colors[response_type], label=response_type)
        bottoms += np.array(counts)

    ax.set_xticks(positions, [PROMPT_LABELS[prompt_id] for prompt_id in FEEDBACK_KEYS])
    ax.set_ylabel("Count")
    ax.set_title("Feedback Response Types by Prompt")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_primary_theme_heatmap(primary_theme_summary: pd.DataFrame, output_path: Path) -> None:
    heatmap = (
        primary_theme_summary.pivot(index="prompt_label", columns="theme_label", values="prevalence")
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    vmax = max(0.35, float(heatmap.to_numpy().max())) if not heatmap.empty else 0.35
    image = ax.imshow(heatmap.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(heatmap.columns)), heatmap.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)), heatmap.index)
    ax.set_title("LLM-Judged Primary Theme Prevalence")

    for row_idx in range(len(heatmap.index)):
        for col_idx in range(len(heatmap.columns)):
            value = heatmap.iloc[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value * 100:.0f}%", ha="center", va="center", color="#111111", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.04)
    colorbar.set_label("Prevalence")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_facet_heatmap(facet_summary: pd.DataFrame, output_path: Path) -> None:
    heatmap = (
        facet_summary.pivot(index="prompt_label", columns="facet_label", values="prevalence")
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    vmax = max(0.5, float(heatmap.to_numpy().max())) if not heatmap.empty else 0.5
    image = ax.imshow(heatmap.to_numpy(), cmap="OrRd", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(heatmap.columns)), heatmap.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)), heatmap.index)
    ax.set_title("LLM-Judged Design/Critique Facet Prevalence")

    for row_idx in range(len(heatmap.index)):
        for col_idx in range(len(heatmap.columns)):
            value = heatmap.iloc[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value * 100:.0f}%", ha="center", va="center", color="#111111", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.04)
    colorbar.set_label("Prevalence")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    metadata: pd.DataFrame,
    feedback_df: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    judged_df: pd.DataFrame,
    primary_theme_summary: pd.DataFrame,
    facet_summary: pd.DataFrame,
    attribute_summary: pd.DataFrame,
    examples_df: pd.DataFrame,
    coding_sheet: pd.DataFrame,
    model: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_dir / "feedback_export_metadata.csv", index=False)
    feedback_df.to_csv(output_dir / "feedback_long.csv", index=False)
    prompt_summary.to_csv(output_dir / "feedback_prompt_summary.csv", index=False)
    judged_df.to_csv(output_dir / "feedback_llm_judgments.csv", index=False)
    primary_theme_summary.to_csv(output_dir / "feedback_llm_primary_theme_summary.csv", index=False)
    facet_summary.to_csv(output_dir / "feedback_llm_facet_summary.csv", index=False)
    attribute_summary.to_csv(output_dir / "feedback_llm_attribute_summary.csv", index=False)
    examples_df.to_csv(output_dir / "feedback_llm_examples.csv", index=False)
    coding_sheet.to_csv(output_dir / "feedback_manual_coding_sheet.csv", index=False)

    summary = build_summary_markdown(
        metadata,
        feedback_df,
        prompt_summary,
        judged_df,
        primary_theme_summary,
        facet_summary,
        model,
    )
    (output_dir / "feedback_analysis_summary.md").write_text(summary)

    make_response_type_figure(prompt_summary, figures_dir / "figure_feedback_response_types.png")
    make_primary_theme_heatmap(primary_theme_summary, figures_dir / "figure_feedback_llm_primary_themes.png")
    make_facet_heatmap(facet_summary, figures_dir / "figure_feedback_llm_facets.png")


def main() -> None:
    args = parse_args()
    exports, metadata = load_unique_exports(args.input)
    if args.condition:
        exports = [
            record
            for record in exports
            if normalize_condition(
                record["payload"].get("mode") or (first_session(record["payload"]) or {}).get("interactionCondition", "")
            )
            == args.condition
        ]
        metadata = metadata[metadata["interaction_condition"] == args.condition].copy()
        metadata.attrs["num_source_files"] = metadata.attrs.get("num_source_files", "unknown")
        metadata.attrs["num_raw_exports"] = metadata.attrs.get("num_raw_exports", "unknown")
        metadata.attrs["duplicate_count"] = metadata.attrs.get("duplicate_count", 0)
        metadata.attrs["condition_filter"] = args.condition
    else:
        metadata.attrs["condition_filter"] = ""
    feedback_df = extract_feedback_long(exports)
    prompt_summary = build_prompt_summary(feedback_df)

    client = load_openai_client(args.openai_api_key)
    judged_df = run_llm_judge(
        feedback_df,
        client=client,
        output_dir=args.output,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        judge_prompts=args.judge_prompts,
        limit=args.limit,
        force=args.force,
    )
    if judged_df.empty:
        raise ValueError("No LLM judgments were produced. Check your prompt selection, cache, and input data.")

    primary_theme_summary = build_primary_theme_summary(judged_df)
    facet_summary = build_facet_summary(judged_df)
    attribute_summary = build_attribute_summary(judged_df)
    examples_df = build_examples(judged_df)
    coding_sheet = build_manual_coding_sheet(judged_df)

    write_outputs(
        args.output,
        metadata,
        feedback_df,
        prompt_summary,
        judged_df,
        primary_theme_summary,
        facet_summary,
        attribute_summary,
        examples_df,
        coding_sheet,
        args.model,
    )

    print(f"Wrote LLM-judge feedback analysis to {args.output}")
    print(f"Unique exports: {metadata['export_id'].nunique()}")
    print(f"Feedback rows: {len(feedback_df)}")
    print(f"LLM judgments: {len(judged_df)}")


if __name__ == "__main__":
    main()
