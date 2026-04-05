"""
Augment session JSON exports with LLM-as-a-judge ratings.

This script reads exported session JSON files, filters out invalid top-level
owner ids, and asks an OpenAI model to rate each AI suggestion
(`LLM-B`, `LLM-CF`, `LLM-CT`).

It writes augmented JSON copies plus flat JSONL/CSV records for downstream
analysis. The original input files are left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.json_export_loader import (
    CONDITION_ORDER,
    SESSION_KEYS,
    INVALID_OWNER_IDS,
    is_valid_export_payload,
    normalize_condition,
)


DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_REASONING_EFFORT = "low"
OPENAI_MAX_RETRIES = 5
OPENAI_RETRY_BASE_SECONDS = 1.5
PROMPT_VERSION = "final-plan-and-ai-judge-v1"
MODEL_ORDER = ["LLM-B", "LLM-CF", "LLM-CT"]
JUDGE_FIELD_NAME = "LLM_as_a_Judge"
SUMMARY_TARGET_ORDER = [*MODEL_ORDER]


SYSTEM_PROMPT = """
You are a careful research judge evaluating one candidate counterfactual action
for a human-AI reflection study.

You will receive:
- the participant's overall goal
- the scenario describing what happened
- one candidate output, which is either the participant's selected final plan
  or one AI suggestion

Rate the candidate on a 1-5 integer scale using only the scenario, goal, and
candidate text.

Criteria:
1. Action_Clarity
   1 = vague, abstract, or not clearly actionable
   5 = concrete, specific, and clearly states what should be done differently

2. Feasibility
   1 = unrealistic or requires implausible changes
   5 = realistic and could plausibly be carried out with limited change

3. Goal_Alignment
   1 = unlikely to improve the outcome relative to the goal
   5 = strongly aligned with achieving a better outcome toward the goal

4. Insight_Novelty
   1 = merely restates obvious scenario details or offers little new insight
   5 = adds a meaningfully useful perspective or non-obvious alternative

Be conservative. Use the full 1-5 range when justified.
Return only the required JSON object.
"""


class JudgeRatings(BaseModel):
    Action_Clarity: int = Field(ge=1, le=5)
    Feasibility: int = Field(ge=1, le=5)
    Goal_Alignment: int = Field(ge=1, le=5)
    Insight_Novelty: int = Field(ge=1, le=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSON files or directories of JSON files to augment.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "llm_judge_augmented_outputs",
        help="Parent directory for the timestamped augmented-output folder.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional explicit timestamp label. Defaults to current local time.",
    )
    parser.add_argument(
        "--condition",
        choices=CONDITION_ORDER,
        default=None,
        help="Optional condition filter. Only matching sessions will be judged.",
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
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for the OpenAI judge.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omitted by default; gpt-5-nano currently rejects this parameter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of candidate texts to judge.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rejudge items even if cached judgments already exist in the output folder.",
    )
    return parser.parse_args()


def normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_text_list(values) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    seen = set()
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def choose_selected_final_plan(final_plans: List[str], selected_index) -> Tuple[str, int | None, bool]:
    if isinstance(selected_index, int) and 0 <= selected_index < len(final_plans):
        return final_plans[selected_index], selected_index, True
    if final_plans:
        return final_plans[0], 0, False
    return "", None, False


def resolve_input_paths(inputs: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            resolved.append(input_path)
            continue
        if input_path.is_dir():
            matches = sorted(input_path.glob("*.json"))
            if not matches:
                raise FileNotFoundError(f"No JSON files found in {input_path}")
            resolved.extend(matches)
            continue
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    unique = sorted(dict.fromkeys(path.resolve() for path in resolved))
    return [Path(path) for path in unique]


def unpack_exports(raw_payload, path: Path) -> List[Dict]:
    if isinstance(raw_payload, dict):
        return [raw_payload]
    if isinstance(raw_payload, list):
        if not all(isinstance(item, dict) for item in raw_payload):
            raise ValueError(f"Expected a list of JSON objects in {path}")
        return raw_payload
    raise ValueError(f"Unsupported JSON structure in {path}: expected an object or a list of objects.")


def load_openai_client(api_key: str) -> OpenAI:
    resolved_api_key = " ".join(str(api_key or os.environ.get("OPENAI_API_KEY", "")).split()).strip()
    if not resolved_api_key:
        for env_path in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
            if not env_path.exists():
                continue
            for line in env_path.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() != "OPENAI_API_KEY":
                    continue
                resolved_api_key = value.strip().strip("'").strip('"')
                break
            if resolved_api_key:
                break
    if not resolved_api_key:
        raise ValueError(
            "llm_judge_augment_json.py requires an OpenAI API key. "
            "Set OPENAI_API_KEY, add it to `.env`, or pass --openai-api-key."
        )
    client = OpenAI(api_key=resolved_api_key)
    if not hasattr(client, "responses"):
        raise RuntimeError(
            "The installed OpenAI Python SDK does not expose `client.responses`. "
            f"Current interpreter: {sys.executable}. "
            "Use a newer OpenAI SDK that supports the Responses API, or run this "
            "script with the same Python environment Codex is using."
        )
    return client


def sleep_before_retry(attempt: int, exc: Exception) -> None:
    wait_seconds = OPENAI_RETRY_BASE_SECONDS * (2 ** attempt)
    print(f"  Retry {attempt + 1}/{OPENAI_MAX_RETRIES}: {exc} (sleep {wait_seconds:.1f}s)")
    time.sleep(wait_seconds)


def make_cache_key(record: Dict, model: str) -> str:
    text_hash = hashlib.sha1(str(record["candidate_text"]).encode("utf-8")).hexdigest()[:12]
    return (
        f"{PROMPT_VERSION}:{model}:{record['export_id']}:{record['session_key']}:"
        f"{record['target_label']}:{text_hash}"
    )


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


def build_judge_input(record: Dict) -> str:
    payload = {
        "candidate_type": record["target_kind"],
        "candidate_label": record["target_label"],
        "overall_goal": record["user_goal"],
        "scenario": record["scenario"],
        "candidate_text": record["candidate_text"],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def judge_record(
    client: OpenAI,
    record: Dict,
    *,
    model: str,
    reasoning_effort: str,
    temperature: float | None,
) -> Dict:
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            request_kwargs = dict(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=build_judge_input(record),
                reasoning={"effort": reasoning_effort},
                text_format=JudgeRatings,
                max_output_tokens=400,
                store=False,
            )
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            response = client.responses.parse(**request_kwargs)
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("The OpenAI response did not contain parsed structured output.")
            return {
                "openai_response_id": response.id,
                "openai_model": model,
                "prompt_version": PROMPT_VERSION,
                **parsed.model_dump(),
            }
        except Exception as exc:  # pragma: no cover - network failures are external
            if attempt == OPENAI_MAX_RETRIES - 1:
                raise
            sleep_before_retry(attempt, exc)
    raise RuntimeError("Unexpected retry loop fallthrough.")


def insert_after_key(mapping: Dict, after_key: str, new_key: str, new_value) -> Dict:
    updated: Dict = {}
    inserted = False
    for key, value in mapping.items():
        updated[key] = value
        if key == after_key:
            updated[new_key] = new_value
            inserted = True
    if not inserted:
        updated[new_key] = new_value
    return updated


def iter_candidate_records(payload: Dict, source_file: str, *, condition_filter: str | None) -> Iterable[Dict]:
    export_id = payload.get("id") or ""
    owner_id = normalize_text(payload.get("owner_id", ""))

    for session_key in SESSION_KEYS:
        session = payload.get(session_key)
        if not isinstance(session, dict):
            continue

        interaction_condition = normalize_condition(session.get("interactionCondition") or payload.get("mode"))
        if condition_filter and interaction_condition != condition_filter:
            continue

        participant_id = normalize_text(session.get("participandID") or owner_id)
        user_goal = normalize_text(session.get("user_goal"))
        scenario = normalize_text(session.get("scenario"))
        suggestions = session.get("AI_Suggestions") or {}
        for model_name in MODEL_ORDER:
            suggestion = suggestions.get(model_name) or {}
            candidate_text = normalize_text(suggestion.get("item", ""))
            if not candidate_text:
                continue
            yield {
                "export_id": export_id,
                "source_file": source_file,
                "owner_id": owner_id,
                "participant_id": participant_id,
                "session_key": session_key,
                "session_number": normalize_text(session.get("session")),
                "interaction_condition": interaction_condition,
                "target_kind": "ai_suggestion",
                "target_label": model_name,
                "target_model": model_name,
                "user_goal": user_goal,
                "scenario": scenario,
                "candidate_text": candidate_text,
            }


def apply_judgment_to_payload(payload: Dict, judgment_rows: List[Dict]) -> Dict:
    updated_payload = payload
    rows_by_session: Dict[str, List[Dict]] = {}
    for row in judgment_rows:
        rows_by_session.setdefault(str(row["session_key"]), []).append(row)

    for session_key, session_rows in rows_by_session.items():
        session = updated_payload.get(session_key)
        if not isinstance(session, dict):
            continue

        session_copy = dict(session)
        suggestions_copy = dict(session.get("AI_Suggestions") or {})

        for row in session_rows:
            llm_ratings = {
                "Action_Clarity": int(row["Action_Clarity"]),
                "Feasibility": int(row["Feasibility"]),
                "Goal_Alignment": int(row["Goal_Alignment"]),
                "Insight_Novelty": int(row["Insight_Novelty"]),
            }

            if row["target_kind"] == "ai_suggestion":
                suggestion = dict(suggestions_copy.get(row["target_model"]) or {})
                suggestion = insert_after_key(suggestion, "ratings", JUDGE_FIELD_NAME, llm_ratings)
                suggestions_copy[row["target_model"]] = suggestion
        if suggestions_copy:
            session_copy["AI_Suggestions"] = suggestions_copy
        updated_payload[session_key] = session_copy

    return updated_payload


def build_summary(judgment_df: pd.DataFrame) -> pd.DataFrame:
    if judgment_df.empty:
        return pd.DataFrame()

    metric_columns = ["Action_Clarity", "Feasibility", "Goal_Alignment", "Insight_Novelty"]
    grouped = (
        judgment_df.groupby(["interaction_condition", "target_label"], observed=True)[metric_columns]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([str(part) for part in column if str(part)])
        for column in grouped.columns.to_flat_index()
    ]
    return grouped


def main() -> None:
    args = parse_args()
    client = load_openai_client(args.openai_api_key)
    effective_temperature = args.temperature
    if effective_temperature is not None and args.model == "gpt-5-nano":
        print("Warning: gpt-5-nano does not support `temperature`; omitting that parameter.")
        effective_temperature = None
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = args.output_root / timestamp
    augmented_dir = run_output_dir / "augmented_json"
    augmented_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = run_output_dir / "llm_judge_records.jsonl"
    cache = {} if args.force else load_existing_judgments(jsonl_path)
    all_rows: List[Dict] = []
    manifest_inputs: List[str] = []
    cache_hits = 0
    api_calls = 0

    input_paths = resolve_input_paths(args.inputs)
    for input_path in input_paths:
        manifest_inputs.append(str(input_path))
        with input_path.open() as f:
            raw_payload = json.load(f)

        exports = [payload for payload in unpack_exports(raw_payload, input_path) if is_valid_export_payload(payload)]
        updated_exports: List[Dict] = []

        for export_index, payload in enumerate(exports):
            payload = dict(payload)
            source_file = str(input_path)
            export_id = payload.get("id") or f"{input_path.stem}:{export_index}"
            judgment_rows_for_payload: List[Dict] = []
            candidate_records = list(
                iter_candidate_records(
                    payload,
                    source_file,
                    condition_filter=args.condition,
                )
            )

            if args.limit > 0:
                remaining = args.limit - len(all_rows)
                if remaining <= 0:
                    break
                candidate_records = candidate_records[:remaining]

            if candidate_records:
                print(
                    f"Judging {len(candidate_records)} records from "
                    f"{input_path.name} export {export_id}..."
                )

            for candidate_record in tqdm(
                candidate_records,
                desc=f"{input_path.name}:{export_id}",
                unit="record",
                leave=False,
            ):
                cache_key = make_cache_key(candidate_record, args.model)
                if not args.force and cache_key in cache:
                    record = cache[cache_key]
                    cache_hits += 1
                else:
                    judgment = judge_record(
                        client,
                        candidate_record,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        temperature=effective_temperature,
                    )
                    record = {
                        "cache_key": cache_key,
                        "export_id": export_id,
                        **candidate_record,
                        **judgment,
                    }
                    cache[cache_key] = record
                    api_calls += 1
                    with jsonl_path.open("a") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                all_rows.append(record)
                judgment_rows_for_payload.append(record)

            updated_payload = apply_judgment_to_payload(payload, judgment_rows_for_payload)
            updated_exports.append(updated_payload)

            if args.limit > 0 and len(all_rows) >= args.limit:
                break

        output_payload = updated_exports[0] if isinstance(raw_payload, dict) and updated_exports else updated_exports
        output_path = augmented_dir / input_path.name
        output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2))

        if args.limit > 0 and len(all_rows) >= args.limit:
            break

    judgment_df = pd.DataFrame(all_rows)
    if judgment_df.empty:
        raise ValueError(
            "No judgments were produced. Check the condition filter, the cache, the invalid owner-id filter, and the input files."
        )

    judgment_df["target_label"] = pd.Categorical(
        judgment_df["target_label"],
        categories=SUMMARY_TARGET_ORDER,
        ordered=True,
    )
    judgment_df.to_csv(run_output_dir / "llm_judge_records.csv", index=False)

    summary_df = build_summary(judgment_df)
    summary_df.to_csv(run_output_dir / "llm_judge_summary.csv", index=False)

    manifest = {
        "timestamp": timestamp,
        "inputs": manifest_inputs,
        "output_root": str(run_output_dir),
        "augmented_json_dir": str(augmented_dir),
        "invalid_owner_ids": sorted(INVALID_OWNER_IDS),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "temperature": effective_temperature,
        "condition_filter": args.condition or "",
        "limit": args.limit,
        "n_judgments": int(len(judgment_df)),
    }
    (run_output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote LLM-judge augmented outputs to {run_output_dir}")
    print(f"Augmented JSON directory: {augmented_dir}")
    print(f"Judgments written: {len(judgment_df)}")
    print(f"Fresh API calls: {api_calls}")
    print(f"Cached judgments reused: {cache_hits}")


if __name__ == "__main__":
    main()
