import argparse
import dataclasses
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Keep transformers on the PyTorch path in environments with incompatible
# TensorFlow or Flax installs.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm

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

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
for import_root in (PROJECT_DIR, REPO_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

from shared.json_export_loader import iter_export_records
from utils.helpers import MIN_WORDS, PHASE_KEYS, normalize_phases_map, parse_deep, split_into_sentences, word_count


# ============================================================
# Device setup
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Configuration
# ============================================================

CSV_MODEL_COLS = {
    "human_alt_staged_json": "Human",
    "LLMB_alt_staged_json": "LLM-B (paraphrased)",
    "LLMB_alt_staged_json_raw": "LLM-B (raw)",
    "LLMC_alt_staged_json": "LLM-C (paraphrased)",
    "LLMC_alt_staged_json_raw": "LLM-C (raw)",
    "LLMCT_alt_staged_json": "LLM-CT (paraphrased)",
    "LLMCT_alt_staged_json_raw": "LLM-CT (raw)",
}

JSON_MODEL_COLS = {
    "human_alt_staged_json": "Human",
    "LLMB_alt_staged_json": "LLM-B",
    "LLMCF_alt_staged_json": "LLM-CF",
    "LLMCT_alt_staged_json": "LLM-CT",
}

JSON_SESSION_KEYS = [f"session_{idx}" for idx in range(1, 6)]

LOCAL_BACKEND = "local"
LLM_AS_JUDGE_BACKEND = "llm_as_judge"
DEFAULT_JUDGE_MODEL = "gpt-5.4"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_REASONING_EFFORT = "low"
OPENAI_MAX_RETRIES = 5
OPENAI_RETRY_BASE_SECONDS = 1.5

PHASE_PROTOTYPES = {
    "0": (
        "Situation Selection. Choosing, avoiding, approaching, or planning which "
        "situations, people, meetings, events, routes, or opportunities to enter."
    ),
    "1": (
        "Situation Modification. Changing the environment, format, schedule, agenda, "
        "setup, conversation, structure, or logistics of a situation."
    ),
    "2": (
        "Attentional Deployment. Redirecting attention, focusing, refocusing, "
        "ignoring distractions, listening, or concentrating on specific details."
    ),
    "3": (
        "Cognitive Change. Reframing, reinterpreting, reminding myself, accepting, "
        "thinking differently, or changing the meaning of the situation."
    ),
    "4": (
        "Response Modulation. Regulating emotional expression or physiology by "
        "breathing, calming down, pausing, walking away, relaxing, or managing reactions."
    ),
}

PHASE_KEYWORDS = {
    "0": [
        "beforehand",
        "before the meeting",
        "before the presentation",
        "ahead of time",
        "in advance",
        "seek out",
        "avoid",
        "choose",
        "chose",
        "join",
        "joined",
        "attend",
        "not go",
        "go with",
        "bring",
        "brought",
        "practice in front of",
        "supportive group",
        "prepared with",
        "come prepared",
    ],
    "1": [
        "change the environment",
        "change the format",
        "restructure",
        "redesign",
        "send",
        "shared agenda",
        "agenda",
        "schedule",
        "reschedule",
        "set up",
        "setup",
        "invite questions",
        "built in time",
        "leave enough time",
        "make space",
        "ask each person",
        "follow-up session",
        "move the meeting",
        "structured discussion",
        "brainstorming",
        "clarify roles",
    ],
    "2": [
        "focus",
        "refocus",
        "pay attention",
        "attention",
        "concentrate",
        "listen",
        "listen actively",
        "look away",
        "looked away",
        "ignored",
        "notice",
        "noticed",
        "watch",
        "watching",
        "stay on track",
    ],
    "3": [
        "remind myself",
        "told myself",
        "reframe",
        "reframed",
        "think differently",
        "thought differently",
        "accept",
        "accepted",
        "realize",
        "realized",
        "remember",
        "remembered",
        "it is okay",
        "okay not to be perfect",
        "kind to myself",
        "perspective",
    ],
    "4": [
        "breathe",
        "breathing",
        "deep breath",
        "pause",
        "paused",
        "calm",
        "calmer",
        "relax",
        "relaxed",
        "step away",
        "walk away",
        "walked away",
        "take a break",
        "cool down",
        "stay calm",
        "anxiety",
        "nervous",
        "yell",
        "yelled",
        "cry",
        "meditate",
    ],
}


# ============================================================
# Model loading
# ============================================================

def load_gpt2():
    """Load GPT-2 model and tokenizer for perplexity computation."""
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


def load_bert():
    """Load BERT model and tokenizer for embedding computation."""
    print("Loading BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    return model, tokenizer


# ============================================================
# Text helpers
# ============================================================

def clean_text(value: object) -> str:
    """Normalize whitespace and coerce a value to a clean string."""
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def load_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Load the OpenAI client for API-backed evaluation."""
    resolved_api_key = " ".join(str(api_key or os.environ.get("OPENAI_API_KEY", "")).split()).strip()
    if not resolved_api_key:
        raise ValueError(
            "LLM-as-Judge mode requires an OpenAI API key. Set OPENAI_API_KEY or pass --openai-api-key."
        )
    print("Loading OpenAI client...")
    return OpenAI(api_key=resolved_api_key)


def clean_text_list(values: object) -> List[str]:
    """Normalize a list-like text field into a list of non-empty strings."""
    if values is None:
        return []
    if isinstance(values, list):
        out = [clean_text(item) for item in values]
        return [item for item in out if item]
    text = clean_text(values)
    return [text] if text else []


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    """Remove exact duplicates while preserving order."""
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def expand_counterfactual_units(values: Sequence[str]) -> List[str]:
    """Split multi-sentence strings into separate units for phase assignment."""
    units: List[str] = []
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        sentences = [clean_text(s) for s in split_into_sentences(text)]
        if sentences:
            units.extend(sentences)
        else:
            units.append(text)
    return dedupe_preserve_order([unit for unit in units if unit])


def normalize_condition_cf_map(raw_map: Dict, split_sentence_values: bool = False) -> Dict[str, List[str]]:
    """Normalize a staged counterfactual map into {phase: list[str]}."""
    raw_map = {str(k): v for k, v in (raw_map or {}).items()}
    out: Dict[str, List[str]] = {}

    for phase in PHASE_KEYS:
        value = raw_map.get(phase, [])
        if isinstance(value, list):
            items = clean_text_list(value)
        elif isinstance(value, str):
            text = clean_text(value)
            if split_sentence_values and text:
                items = [clean_text(s) for s in split_into_sentences(text)] or [text]
            else:
                items = [text] if text else []
        else:
            items = []
        out[phase] = dedupe_preserve_order([item for item in items if item])

    return out


def filter_counterfactuals(cf_map: Dict[str, List[str]], min_words: int) -> Dict[str, List[str]]:
    """Filter staged counterfactuals by minimum word count."""
    out: Dict[str, List[str]] = {}
    for phase in PHASE_KEYS:
        out[phase] = [cf for cf in cf_map.get(phase, []) if word_count(cf) >= min_words]
    return out


def flatten_filtered_counterfactuals(cf_map: Dict[str, List[str]]) -> List[str]:
    """Flatten a filtered counterfactual map into a single list."""
    out: List[str] = []
    for phase in PHASE_KEYS:
        out.extend(cf_map.get(phase, []))
    return out


def extract_ai_item(session: Dict, key: str) -> List[str]:
    """Extract a single AI suggestion item as a one-element list."""
    payload = (session.get("AI_Suggestions") or {}).get(key) or {}
    if isinstance(payload, dict):
        item = clean_text(payload.get("item", ""))
        return [item] if item else []
    return []


def select_human_plans(session: Dict, human_source: str) -> Tuple[List[str], str]:
    """Select the human-authored plans to evaluate for session-JSON mode."""
    final_plans = clean_text_list(session.get("User_Plans_Final"))
    initial_plans = clean_text_list(session.get("User_Plans_Initial"))

    if human_source == "final":
        return final_plans, "final"
    if human_source == "initial":
        return initial_plans, "initial"
    if final_plans:
        return final_plans, "final"
    return initial_plans, "initial_fallback"


# ============================================================
# Metric computation
# ============================================================

@torch.inference_mode()
def compute_perplexity(sentence: str, gpt2_model, gpt2_tokenizer) -> float:
    """Compute GPT-2 perplexity for a sentence."""
    encodings = gpt2_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=gpt2_model.config.n_positions,
    )
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)

    if input_ids.shape[1] < 2:
        return float("nan")

    outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    token_count = target_mask.sum()
    if token_count.item() == 0:
        return float("nan")

    mean_nll = -(target_log_probs * target_mask).sum() / token_count
    perplexity = torch.exp(mean_nll)
    return perplexity.item()


@torch.inference_mode()
def embed_sentence_local(
    text: str,
    bert_model,
    bert_tokenizer,
    embedding_cache: Optional[Dict[str, torch.Tensor]] = None,
    max_len: int = 128,
) -> Optional[torch.Tensor]:
    """Compute a mean-pooled BERT embedding with simple caching."""
    text = clean_text(text)
    if not text:
        return None

    if embedding_cache is not None and text in embedding_cache:
        return embedding_cache[text]

    encoded = bert_tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_embeddings = (last_hidden * mask_expanded).sum(1)
    sum_mask = mask_expanded.sum(1).clamp_min(1.0)
    mean_pooled = (sum_embeddings / sum_mask).squeeze(0)

    if embedding_cache is not None:
        embedding_cache[text] = mean_pooled

    return mean_pooled


EmbeddingVector = Union[torch.Tensor, np.ndarray]


@dataclasses.dataclass
class EvaluationBackend:
    """Backend for local metrics or API-backed LLM-as-Judge metrics."""

    mode: str
    gpt2_model: Optional[GPT2LMHeadModel] = None
    gpt2_tokenizer: Optional[GPT2TokenizerFast] = None
    bert_model: Optional[BertModel] = None
    bert_tokenizer: Optional[BertTokenizer] = None
    openai_client: Optional[OpenAI] = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    embedding_cache: Optional[Dict[str, EmbeddingVector]] = None
    judge_cache: Optional[Dict[str, Dict[str, object]]] = None

    def __post_init__(self) -> None:
        if self.embedding_cache is None:
            self.embedding_cache = {}
        if self.judge_cache is None:
            self.judge_cache = {}

    def quality_metric_column(self) -> str:
        """Return the primary quality-metric column for this backend."""
        return "judge_score" if self.mode == LLM_AS_JUDGE_BACKEND else "perplexity"

    def quality_metric_display_name(self) -> str:
        """Return the human-readable label for the primary quality metric."""
        return "LLM Judge Score" if self.mode == LLM_AS_JUDGE_BACKEND else "Perplexity"

    def quality_metric_summary_label(self) -> str:
        """Return the summary-table label for the primary quality metric."""
        return "JudgeScore↑" if self.mode == LLM_AS_JUDGE_BACKEND else "Perplexity↓"

    def _sleep_before_retry(self, attempt: int, exc: Exception) -> None:
        """Sleep with exponential backoff after a transient API error."""
        wait_seconds = OPENAI_RETRY_BASE_SECONDS * (2 ** attempt)
        print(f"  Retry {attempt + 1}/{OPENAI_MAX_RETRIES}: {exc} (sleep {wait_seconds:.1f}s)")
        time.sleep(wait_seconds)

    def embed(self, text: str, max_len: int = 128) -> Optional[EmbeddingVector]:
        """Embed text using either local BERT or OpenAI embeddings."""
        text = clean_text(text)
        if not text:
            return None

        cache_key = f"{self.mode}:{self.embedding_model}:{max_len}:{text}"
        if self.embedding_cache is not None and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        if self.mode == LOCAL_BACKEND:
            embedding = embed_sentence_local(
                text,
                self.bert_model,
                self.bert_tokenizer,
                embedding_cache=None,
                max_len=max_len,
            )
        else:
            embedding = self._embed_openai(text)

        if embedding is not None and self.embedding_cache is not None:
            self.embedding_cache[cache_key] = embedding
        return embedding

    def _embed_openai(self, text: str) -> Optional[np.ndarray]:
        """Embed text with the OpenAI embeddings API."""
        if self.openai_client is None:
            raise ValueError("OpenAI client is required for LLM-as-Judge mode.")

        for attempt in range(OPENAI_MAX_RETRIES):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                )
                return np.asarray(response.data[0].embedding, dtype=np.float32)
            except Exception as exc:  # pragma: no cover - network failures are external
                if attempt == OPENAI_MAX_RETRIES - 1:
                    print(f"  OpenAI embedding failed: {exc}")
                    return None
                self._sleep_before_retry(attempt, exc)
        return None

    def cosine_similarity(
        self,
        emb1: Optional[EmbeddingVector],
        emb2: Optional[EmbeddingVector],
    ) -> float:
        """Compute cosine similarity for either torch or numpy vectors."""
        if emb1 is None or emb2 is None:
            return float("nan")

        if isinstance(emb1, torch.Tensor) and isinstance(emb2, torch.Tensor):
            return torch.cosine_similarity(emb1, emb2, dim=0).item()

        arr1 = np.asarray(emb1, dtype=np.float32)
        arr2 = np.asarray(emb2, dtype=np.float32)
        denom = np.linalg.norm(arr1) * np.linalg.norm(arr2)
        if denom == 0:
            return float("nan")
        return float(np.dot(arr1, arr2) / denom)

    def compute_diversity(self, sentences: List[str]) -> float:
        """Compute average pairwise cosine distance among a set of sentences."""
        if len(sentences) < 2:
            return 0.0

        embeddings = [self.embed(sentence) for sentence in sentences]
        total_distance = 0.0
        count = 0
        for idx in range(len(embeddings)):
            for jdx in range(idx + 1, len(embeddings)):
                sim = self.cosine_similarity(embeddings[idx], embeddings[jdx])
                if sim == sim:
                    total_distance += (1 - sim)
                    count += 1

        return total_distance / count if count > 0 else 0.0

    def score_counterfactual(
        self,
        sentence: str,
        *,
        target_text: str,
        phase: str,
        phase_text: str,
    ) -> Dict[str, object]:
        """Compute the primary quality metric for a candidate counterfactual."""
        if self.mode == LOCAL_BACKEND:
            try:
                perplexity = compute_perplexity(sentence, self.gpt2_model, self.gpt2_tokenizer)
            except Exception:
                perplexity = float("nan")
            return {
                "perplexity": perplexity,
                "judge_score": float("nan"),
                "judge_rationale": "",
            }

        return self._judge_with_openai(
            sentence,
            target_text=target_text,
            phase=phase,
            phase_text=phase_text,
        )

    def _judge_with_openai(
        self,
        sentence: str,
        *,
        target_text: str,
        phase: str,
        phase_text: str,
    ) -> Dict[str, object]:
        """Score a counterfactual with GPT-5.4 using a fixed rubric."""
        if self.openai_client is None:
            raise ValueError("OpenAI client is required for LLM-as-Judge mode.")

        cache_key = json.dumps(
            {
                "model": self.judge_model,
                "sentence": clean_text(sentence),
                "target_text": clean_text(target_text),
                "phase": str(phase),
                "phase_text": clean_text(phase_text),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if self.judge_cache is not None and cache_key in self.judge_cache:
            return self.judge_cache[cache_key]

        system_prompt = (
            "You are a strict evaluator for counterfactual quality in emotion-regulation research. "
            "Return only minified JSON with keys judge_score and judge_rationale. "
            "judge_score must be a number from 1.0 to 5.0, where 1 is poor and 5 is excellent. "
            "Score the candidate on overall usefulness, fluency, specificity, goal alignment, and fit to the stated Gross-model phase. "
            "Keep judge_rationale under 30 words."
        )
        user_prompt = (
            f"Gross-model phase: {phase}\n"
            f"Phase prototype: {PHASE_PROTOTYPES.get(str(phase), '')}\n"
            f"Scenario phase text: {clean_text(phase_text) or '(empty)'}\n"
            f"Target goal: {clean_text(target_text) or '(empty)'}\n"
            f"Candidate counterfactual: {clean_text(sentence)}"
        )

        payload = {
            "perplexity": float("nan"),
            "judge_score": float("nan"),
            "judge_rationale": "",
        }
        for attempt in range(OPENAI_MAX_RETRIES):
            try:
                response = self.openai_client.responses.create(
                    model=self.judge_model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    reasoning={"effort": self.reasoning_effort},
                    max_output_tokens=180,
                )
                parsed = json.loads(response.output_text)
                score = parsed.get("judge_score", float("nan"))
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = float("nan")
                if score == score:
                    score = min(5.0, max(1.0, score))
                payload = {
                    "perplexity": float("nan"),
                    "judge_score": score,
                    "judge_rationale": clean_text(parsed.get("judge_rationale", "")),
                }
                break
            except Exception as exc:  # pragma: no cover - network failures are external
                if attempt == OPENAI_MAX_RETRIES - 1:
                    print(f"  LLM judge failed: {exc}")
                    break
                self._sleep_before_retry(attempt, exc)

        if self.judge_cache is not None:
            self.judge_cache[cache_key] = payload
        return payload


def cosine_similarity(emb1: Optional[torch.Tensor], emb2: Optional[torch.Tensor]) -> float:
    """Compute cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None:
        return float("nan")
    return torch.cosine_similarity(emb1, emb2, dim=0).item()


# ============================================================
# Phase inference for session-JSON inputs
# ============================================================

def keyword_phase_scores(text: str) -> Dict[str, float]:
    """Assign lightweight keyword scores for each Gross-model phase."""
    text_lc = clean_text(text).lower()
    scores = {phase: 0.0 for phase in PHASE_KEYS}

    for phase, keywords in PHASE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lc:
                scores[phase] += 1.0 if " " in keyword else 0.75

    if any(token in text_lc for token in ["before ", "beforehand", "in advance", "ahead of time"]):
        scores["0"] += 0.5
    if any(token in text_lc for token in ["agenda", "schedule", "format", "environment", "discussion block"]):
        scores["1"] += 0.5
    if any(token in text_lc for token in ["focus", "refocus", "noticed", "attention", "listen"]):
        scores["2"] += 0.5
    if any(token in text_lc for token in ["remind myself", "realized", "perspective", "okay"]):
        scores["3"] += 0.5
    if any(token in text_lc for token in ["breath", "calm", "pause", "relax", "break", "walk away"]):
        scores["4"] += 0.5

    return scores


def build_phase_prototype_embeddings(
    backend: EvaluationBackend,
) -> Dict[str, EmbeddingVector]:
    """Embed the five phase descriptions once for phase inference."""
    return {
        phase: backend.embed(description)
        for phase, description in PHASE_PROTOTYPES.items()
    }


def infer_phase(
    text: str,
    backend: EvaluationBackend,
    prototype_embeddings: Dict[str, EmbeddingVector],
) -> str:
    """Infer the most likely Gross-model phase for a short text span."""
    text = clean_text(text)
    if not text:
        return "0"

    text_embedding = backend.embed(text)
    keyword_scores = keyword_phase_scores(text)

    combined_scores = {}
    for phase in PHASE_KEYS:
        proto_embedding = prototype_embeddings.get(phase)
        sim = backend.cosine_similarity(text_embedding, proto_embedding)
        sim = 0.0 if sim != sim else sim
        combined_scores[phase] = sim + 0.05 * keyword_scores[phase]

    return max(PHASE_KEYS, key=lambda phase: (combined_scores[phase], keyword_scores[phase]))


def build_scenario_phase_map(
    scenario: str,
    backend: EvaluationBackend,
    prototype_embeddings: Dict[str, EmbeddingVector],
) -> Dict[str, str]:
    """Group scenario sentences into the five Gross-model phases."""
    scenario = clean_text(scenario)
    if not scenario:
        return {phase: "" for phase in PHASE_KEYS}

    sentences = [clean_text(sentence) for sentence in split_into_sentences(scenario)]
    if not sentences:
        sentences = [scenario]

    grouped: Dict[str, List[str]] = {phase: [] for phase in PHASE_KEYS}
    for sentence in dedupe_preserve_order(sentences):
        phase = infer_phase(sentence, backend, prototype_embeddings)
        grouped[phase].append(sentence)

    return {phase: " ".join(grouped[phase]).strip() for phase in PHASE_KEYS}


def build_counterfactual_phase_map(
    counterfactuals: Sequence[str],
    backend: EvaluationBackend,
    prototype_embeddings: Dict[str, EmbeddingVector],
) -> Dict[str, List[str]]:
    """Assign counterfactual units to Gross-model phases."""
    grouped: Dict[str, List[str]] = {phase: [] for phase in PHASE_KEYS}

    for unit in expand_counterfactual_units(counterfactuals):
        phase = infer_phase(unit, backend, prototype_embeddings)
        grouped[phase].append(unit)

    return {phase: dedupe_preserve_order(grouped[phase]) for phase in PHASE_KEYS}


# ============================================================
# Input loading
# ============================================================

def load_csv_input(input_path: Path) -> Tuple[pd.DataFrame, Dict[str, str], str]:
    """Load the original experiment CSV input."""
    print(f"Loading CSV data from {input_path}...")
    df = pd.read_csv(input_path, dtype=str).fillna("")

    model_cols = {
        column: display_name
        for column, display_name in CSV_MODEL_COLS.items()
        if column in df.columns
    }

    phase_col = "transcript_paraphrased_staged_json"
    if phase_col not in df.columns:
        raise ValueError(
            "CSV input is missing 'transcript_paraphrased_staged_json', which is required "
            "for contextual coherence against staged scenario text."
        )

    return df, model_cols, phase_col


def load_json_directory(
    input_dir: Path,
    backend: EvaluationBackend,
    human_source: str,
    limit: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, str], str]:
    """Load and normalize HAI session JSON files into an evaluation dataframe."""
    export_records = list(iter_export_records(input_dir))
    if not export_records:
        raise ValueError(f"No JSON files found in {input_dir}")

    prototype_embeddings = build_phase_prototype_embeddings(backend)

    rows = []
    condition_counts = Counter()
    human_source_counts = Counter()
    source_files = {record["source_file"] for record in export_records}

    for export_record in tqdm(export_records, desc="Normalizing session JSONs"):
        payload = export_record["payload"]
        source_file = Path(export_record["source_file"]).name
        owner_id = export_record.get("owner_id", "")

        for session_key in JSON_SESSION_KEYS:
            session = payload.get(session_key)
            if not isinstance(session, dict):
                continue

            scenario = clean_text(session.get("scenario", ""))
            goal = clean_text(session.get("user_goal", ""))
            human_plans, human_plan_source = select_human_plans(session, human_source)

            scenario_phase_map = build_scenario_phase_map(
                scenario,
                backend,
                prototype_embeddings,
            )
            human_phase_map = build_counterfactual_phase_map(
                human_plans,
                backend,
                prototype_embeddings,
            )
            llmb_phase_map = build_counterfactual_phase_map(
                extract_ai_item(session, "LLM-B"),
                backend,
                prototype_embeddings,
            )
            llmcf_phase_map = build_counterfactual_phase_map(
                extract_ai_item(session, "LLM-CF"),
                backend,
                prototype_embeddings,
            )
            llmct_phase_map = build_counterfactual_phase_map(
                extract_ai_item(session, "LLM-CT"),
                backend,
                prototype_embeddings,
            )

            interaction_condition = clean_text(
                session.get("interactionCondition") or payload.get("mode", "")
            )
            condition_counts[interaction_condition or "UNKNOWN"] += 1
            human_source_counts[human_plan_source] += 1

            rows.append(
                {
                    "userId": clean_text(session.get("participandID") or owner_id),
                    "sessionNumber": str(session.get("session", session_key.split("_")[-1])),
                    "session_key": session_key,
                    "source_file": source_file,
                    "interactionCondition": interaction_condition,
                    "target_goal_paraphrased": goal,
                    "scenario": scenario,
                    "scenario_phase_json": json.dumps(scenario_phase_map, ensure_ascii=False),
                    "human_plan_source": human_plan_source,
                    "human_plan_count": len(human_plans),
                    "final_selected_plan_number": session.get("Final_Selected_PlanNumber", ""),
                    "human_alt_staged_json": json.dumps(human_phase_map, ensure_ascii=False),
                    "LLMB_alt_staged_json": json.dumps(llmb_phase_map, ensure_ascii=False),
                    "LLMCF_alt_staged_json": json.dumps(llmcf_phase_map, ensure_ascii=False),
                    "LLMCT_alt_staged_json": json.dumps(llmct_phase_map, ensure_ascii=False),
                }
            )

            if limit is not None and len(rows) >= limit:
                break

        if limit is not None and len(rows) >= limit:
            break

    df = pd.DataFrame(rows).fillna("")
    print(f"Loaded {len(df)} sessions from {len(source_files)} JSON files.")
    print(f"Interaction conditions: {dict(condition_counts)}")
    print(f"Human-plan source usage: {dict(human_source_counts)}")
    print(
        "Note: session JSONs store one AI suggestion per model, so per-scenario "
        "diversity for AI conditions will usually be 0."
    )
    return df, JSON_MODEL_COLS, "scenario_phase_json"


# ============================================================
# Per-condition evaluation
# ============================================================

def evaluate_condition(
    df: pd.DataFrame,
    condition_col: str,
    display_name: str,
    phase_col: str,
    backend: EvaluationBackend,
    min_words: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a single condition column. Returns per-counterfactual metrics and
    per-scenario diversity scores.
    """
    metric_rows = []
    diversity_rows = []
    split_human_strings = condition_col == "human_alt_staged_json"

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {display_name}", leave=False):
        user_id = clean_text(row.get("userId", ""))
        session = clean_text(row.get("sessionNumber", ""))
        target_text = clean_text(row.get("target_goal_paraphrased", ""))

        phase_json = parse_deep(row.get(phase_col, ""), default={})
        phases_map = normalize_phases_map(phase_json)

        condition_json = parse_deep(row.get(condition_col, ""), default={})
        cf_map = normalize_condition_cf_map(
            condition_json,
            split_sentence_values=split_human_strings,
        )
        cf_map = filter_counterfactuals(cf_map, min_words=min_words)
        all_cfs = flatten_filtered_counterfactuals(cf_map)

        diversity_rows.append(
            {
                "userId": user_id,
                "sessionNumber": session,
                "source_file": clean_text(row.get("source_file", "")),
                "interactionCondition": clean_text(row.get("interactionCondition", "")),
                "model_source": condition_col,
                "display_name": display_name,
                "analysis_backend": backend.mode,
                "cf_count": len(all_cfs),
                "diversity": backend.compute_diversity(all_cfs)
                if len(all_cfs) >= 2
                else 0.0,
            }
        )

        target_emb = (
            backend.embed(target_text)
            if target_text
            else None
        )

        phase_embeddings = {
            phase: backend.embed(phases_map.get(phase, ""))
            if clean_text(phases_map.get(phase, ""))
            else None
            for phase in PHASE_KEYS
        }

        for phase in PHASE_KEYS:
            for idx, cf_text in enumerate(cf_map.get(phase, [])):
                quality_scores = backend.score_counterfactual(
                    cf_text,
                    target_text=target_text,
                    phase=phase,
                    phase_text=phases_map.get(phase, ""),
                )
                cf_emb = backend.embed(cf_text)

                metric_rows.append(
                    {
                        "userId": user_id,
                        "sessionNumber": session,
                        "source_file": clean_text(row.get("source_file", "")),
                        "interactionCondition": clean_text(row.get("interactionCondition", "")),
                        "human_plan_source": clean_text(row.get("human_plan_source", "")),
                        "model_source": condition_col,
                        "display_name": display_name,
                        "analysis_backend": backend.mode,
                        "phase": phase,
                        "cf_index": idx,
                        "cf_text": cf_text,
                        "perplexity": quality_scores.get("perplexity", float("nan")),
                        "judge_score": quality_scores.get("judge_score", float("nan")),
                        "judge_rationale": quality_scores.get("judge_rationale", ""),
                        "similarity_to_target": backend.cosine_similarity(cf_emb, target_emb),
                        "similarity_to_phase": backend.cosine_similarity(
                            cf_emb,
                            phase_embeddings.get(phase),
                        ),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(diversity_rows)


# ============================================================
# Summary printing
# ============================================================

def print_summary(
    all_metrics: Dict[str, pd.DataFrame],
    all_diversity: Dict[str, pd.DataFrame],
    *,
    quality_metric_column: str,
    quality_metric_label: str,
):
    """Print average metrics matching the experiment summary format."""
    print("\n" + "=" * 70)
    print(f"{'Condition':<25} {quality_metric_label:>12} {'TargetSim↑':>12} {'PhaseSim↑':>12} {'Diversity↑':>12}")
    print("=" * 70)

    for display_name, metrics_df in all_metrics.items():
        if metrics_df.empty:
            print(f"{display_name:<25} {'n/a':>12} {'n/a':>12} {'n/a':>12} {'n/a':>12}")
            continue

        avg_quality = metrics_df[quality_metric_column].mean(skipna=True)
        avg_target = metrics_df["similarity_to_target"].mean(skipna=True)
        avg_phase = metrics_df["similarity_to_phase"].mean(skipna=True)

        div_df = all_diversity[display_name]
        valid_div = div_df[div_df["cf_count"] > 1]
        avg_div = valid_div["diversity"].mean(skipna=True) if not valid_div.empty else float("nan")

        print(f"{display_name:<25} {avg_quality:>12.2f} {avg_target:>12.2f} {avg_phase:>12.2f} {avg_div:>12.2f}")

    print("=" * 70)


def resolve_output_dir(base_output_dir: Path, analysis_backend: str) -> Path:
    """Place API-backed runs in a dedicated subdirectory to avoid collisions."""
    if analysis_backend == LLM_AS_JUDGE_BACKEND:
        return base_output_dir / "llm_as_judge"
    return base_output_dir


def build_backend(args: argparse.Namespace) -> EvaluationBackend:
    """Build either the local or API-backed evaluation backend."""
    if args.analysis_backend == LOCAL_BACKEND:
        gpt2_model, gpt2_tokenizer = load_gpt2()
        bert_model, bert_tokenizer = load_bert()
        return EvaluationBackend(
            mode=LOCAL_BACKEND,
            gpt2_model=gpt2_model,
            gpt2_tokenizer=gpt2_tokenizer,
            bert_model=bert_model,
            bert_tokenizer=bert_tokenizer,
        )

    client = load_openai_client(args.openai_api_key)
    return EvaluationBackend(
        mode=LLM_AS_JUDGE_BACKEND,
        openai_client=client,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        reasoning_effort=args.reasoning_effort,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 2-1: Automatic Evaluation")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original CSV or a directory of session JSON files",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "outputs" / "automatic_evaluation"),
        help="Output directory",
    )
    parser.add_argument(
        "--analysis-backend",
        choices=[LOCAL_BACKEND, LLM_AS_JUDGE_BACKEND],
        default=LOCAL_BACKEND,
        help="Choose local GPT-2/BERT scoring or OpenAI-backed LLM-as-Judge scoring",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional OpenAI API key override for LLM-as-Judge mode",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenAI model to use for judge scoring in LLM-as-Judge mode",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="OpenAI embedding model to use in LLM-as-Judge mode",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=DEFAULT_REASONING_EFFORT,
        help="Reasoning effort for GPT judge calls in LLM-as-Judge mode",
    )
    parser.add_argument(
        "--human-source",
        choices=["final", "initial", "final_or_initial"],
        default="final_or_initial",
        help="For session-JSON inputs, choose which human-authored plans to evaluate",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Minimum word count for a counterfactual to be evaluated",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of session rows to load (useful for smoke tests)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = resolve_output_dir(Path(args.output).expanduser().resolve(), args.analysis_backend)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_words = args.min_words
    if min_words is None:
        min_words = 1 if input_path.is_dir() else MIN_WORDS
    print(f"Using min_words={min_words}")

    backend = build_backend(args)
    print(f"Using analysis_backend={backend.mode}")
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "analysis_backend": backend.mode,
                "quality_metric_column": backend.quality_metric_column(),
                "quality_metric_display_name": backend.quality_metric_display_name(),
                "judge_model": backend.judge_model if backend.mode == LLM_AS_JUDGE_BACKEND else "",
                "embedding_model": (
                    backend.embedding_model if backend.mode == LLM_AS_JUDGE_BACKEND else "bert-base-uncased"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved run metadata -> {metadata_path}")

    if input_path.is_dir():
        df, model_cols, phase_col = load_json_directory(
            input_path,
            backend,
            human_source=args.human_source,
            limit=args.limit,
        )
        normalized_path = output_dir / "normalized_sessions.csv"
        df.to_csv(normalized_path, index=False)
        print(f"Saved normalized session table -> {normalized_path}")
    else:
        df, model_cols, phase_col = load_csv_input(input_path)
        if args.limit is not None:
            df = df.head(args.limit).copy()
            print(f"Limiting evaluation to the first {len(df)} CSV rows.")

    all_metrics: Dict[str, pd.DataFrame] = {}
    all_diversity: Dict[str, pd.DataFrame] = {}

    for condition_col, display_name in model_cols.items():
        print(f"\nEvaluating {display_name}...")
        metrics_df, diversity_df = evaluate_condition(
            df,
            condition_col,
            display_name,
            phase_col,
            backend,
            min_words=min_words,
        )

        all_metrics[display_name] = metrics_df
        all_diversity[display_name] = diversity_df

        metrics_path = output_dir / f"metrics_{condition_col}.csv"
        diversity_path = output_dir / f"diversity_{condition_col}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        diversity_df.to_csv(diversity_path, index=False)
        print(f"  Saved -> {metrics_path}")
        print(f"  Saved -> {diversity_path}")

    print_summary(
        all_metrics,
        all_diversity,
        quality_metric_column=backend.quality_metric_column(),
        quality_metric_label=backend.quality_metric_summary_label(),
    )


if __name__ == "__main__":
    main()
