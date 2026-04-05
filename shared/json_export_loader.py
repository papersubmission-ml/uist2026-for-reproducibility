from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List


SESSION_KEYS = [f"session_{idx}" for idx in range(1, 6)]
CONDITION_ORDER = ["HAI", "AIH"]
INVALID_OWNER_IDS = {"Forgot to record", "survey_thpz6", "testtest"}


def normalize_condition(value) -> str:
    text = str(value or "").strip().upper()
    if text in CONDITION_ORDER:
        return text
    return ""


def infer_export_condition(payload: Dict) -> str:
    top_level = normalize_condition(payload.get("mode"))
    if top_level:
        return top_level

    session_conditions = {
        normalize_condition((payload.get(session_key) or {}).get("interactionCondition"))
        for session_key in SESSION_KEYS
        if isinstance(payload.get(session_key), dict)
    }
    session_conditions.discard("")
    if len(session_conditions) == 1:
        return next(iter(session_conditions))
    return ""


def normalize_owner_id(value) -> str:
    return str(value or "").strip()


def is_valid_export_payload(payload: Dict) -> bool:
    return normalize_owner_id(payload.get("owner_id")) not in INVALID_OWNER_IDS


def _resolve_input_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(input_path.glob("*.json"))
        if paths:
            return paths
        raise FileNotFoundError(f"No JSON files found in {input_path}")
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _unpack_exports(raw_payload, path: Path) -> List[Dict]:
    if isinstance(raw_payload, dict):
        return [raw_payload]
    if isinstance(raw_payload, list):
        if not all(isinstance(item, dict) for item in raw_payload):
            raise ValueError(f"Expected a list of JSON objects in {path}")
        return raw_payload
    raise ValueError(
        f"Unsupported JSON structure in {path}: expected an object or a list of objects."
    )


def iter_export_records(input_path: Path) -> Iterator[Dict]:
    for path in _resolve_input_paths(input_path):
        with path.open() as f:
            raw_payload = json.load(f)

        for export_index, payload in enumerate(_unpack_exports(raw_payload, path)):
            if not is_valid_export_payload(payload):
                continue
            export_id = payload.get("id") or f"{path.stem}:{export_index}"
            yield {
                "export_id": export_id,
                "export_index": export_index,
                "source_file": str(path),
                "payload": payload,
                "owner_id": normalize_owner_id(payload.get("owner_id")),
                "export_condition": infer_export_condition(payload),
            }
