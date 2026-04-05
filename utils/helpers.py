"""
Shared utility functions for the ICUI experiments codebase.
"""

import ast
import json
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

PHASE_KEYS = ["0", "1", "2", "3", "4"]

PHASE_LABELS = [
    "Situation Selection",
    "Situation Modification",
    "Attention Deployment",
    "Cognitive Change",
    "Response Modulation",
]

MIN_WORDS = 6  # Minimum word count to consider a counterfactual valid

SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
}


# ============================================================
# Text normalization
# ============================================================

def normalize_quotes(s: str) -> str:
    """Replace curly/smart quotes with straight quotes."""
    if not isinstance(s, str):
        return s
    for bad, good in SMART_QUOTES.items():
        s = s.replace(bad, good)
    return s


def word_count(s: str) -> int:
    """Count words in a string."""
    return len((s or "").strip().split())


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using basic heuristics."""
    sentence_enders = re.compile(
        r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bProf\.)"
        r"(?<!\bMs\.)(?<!\bJr\.)(?<![A-Z]\.)(?<=\.|\?|!)\s+"
    )
    sentences = sentence_enders.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# JSON / data parsing
# ============================================================

def parse_deep(obj: Any, default: Any = None) -> Any:
    """
    Robust parser for cells that may be: already a dict, JSON string,
    Python literal (single quotes), or double-encoded JSON.
    """
    if default is None:
        default = {k: "" for k in PHASE_KEYS}
    if isinstance(obj, dict):
        return obj
    if obj is None:
        return default

    s = str(obj).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return default

    s = normalize_quotes(s)

    for _ in range(3):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                s = parsed.strip()
                continue
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                s = parsed.strip()
                continue
        except Exception:
            pass
        break

    return default


def ensure_str_keys(d: Dict) -> Dict[str, Any]:
    """Convert any int keys to string keys."""
    return {str(k): v for k, v in (d or {}).items()}


def normalize_phases_map(raw_map: Dict) -> Dict[str, str]:
    """Normalize a phase map to have string keys '0'-'4' with string values."""
    m = ensure_str_keys(raw_map)
    return {k: str((m.get(k) or "")).strip() for k in PHASE_KEYS}


def normalize_cf_map(raw_map: Dict) -> Dict[str, List[str]]:
    """Normalize a counterfactual map to have string keys '0'-'4' with list[str] values."""
    m = ensure_str_keys(raw_map)
    out = {}
    for k in PHASE_KEYS:
        v = m.get(k, [])
        if isinstance(v, list):
            out[k] = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        elif isinstance(v, str):
            v = v.strip()
            out[k] = [v] if v else []
        else:
            out[k] = []
    return out


def flatten_counterfactuals(cf_map: Dict[str, List[str]]) -> List[str]:
    """Flatten a per-phase counterfactual map into a single list, filtering by MIN_WORDS."""
    all_cfs = []
    for k in PHASE_KEYS:
        vals = cf_map.get(k, [])
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, str) and word_count(v) >= MIN_WORDS:
                    all_cfs.append(v.strip())
    return all_cfs


# ============================================================
# Actionable / coverage arrays
# ============================================================

def is_nonempty(x: Any) -> bool:
    """True if x encodes at least one non-empty counterfactual."""
    if x is None:
        return False
    if isinstance(x, str):
        return x.strip() != ""
    if isinstance(x, (list, tuple)):
        return any(str(it).strip() != "" for it in x)
    if isinstance(x, dict):
        return any(is_nonempty(v) for v in x.values())
    return bool(x)


def compute_actionables(parsed_obj: Dict) -> List[int]:
    """Return [b0,b1,b2,b3,b4]: 1 if phase has any counterfactual, else 0."""
    out = []
    for i in range(5):
        val = parsed_obj.get(str(i), "")
        out.append(1 if is_nonempty(val) else 0)
    return out


def compute_count_array(cf_map: Dict) -> List[int]:
    """Return [c0,c1,c2,c3,c4]: count of counterfactuals per phase."""
    m = ensure_str_keys(cf_map)
    counts = []
    for k in PHASE_KEYS:
        v = m.get(k, [])
        if isinstance(v, list):
            counts.append(len([x for x in v if isinstance(x, str) and x.strip()]))
        elif isinstance(v, str) and v.strip():
            counts.append(1)
        else:
            counts.append(0)
    return counts


def to_percent(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to sum to 100%."""
    total = np.sum(arr)
    if total == 0:
        return np.zeros_like(arr, dtype=float)
    return arr / total * 100


# ============================================================
# Coercion helpers for data loading
# ============================================================

def coerce_transcript(val: Any) -> str:
    """Handle list-of-strings, stringified lists, or plain strings."""
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            maybe = ast.literal_eval(s)
            if isinstance(maybe, list):
                return " ".join(str(x) for x in maybe)
        except Exception:
            pass
    return s


def coerce_goal(val: Any) -> str:
    """Safely coerce a goal value to string."""
    return "" if pd.isna(val) else str(val).strip()
