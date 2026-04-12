"""Deterministic JSON Grader and Perplexity Guard for PromptForge.

Grader (PRIMARY: OpenAI, TESTING: Groq):
    Uses an OpenAI-compatible API to call the grader model.
    Forces JSON output via response_format={"type": "json_object"}.
    Compares output against the task's ground-truth fixture.
    Returns 1.0 (ALL checks pass) or 0.0 (ANY check fails). Binary.

    Called ONLY on SUBMIT — never during PROBE or structural actions.

Perplexity Guard:
    Optional: uses DistilGPT-2 via `transformers` if the library is installed.
    Falls back gracefully to 0.0 (no penalty) if transformers is not available.
    Returns PERPLEXITY_PENALTY_SCALAR (default -0.5) if current perplexity
    exceeds baseline * PERPLEXITY_THRESHOLD_MULTIPLIER. Otherwise 0.0.

Environment variables:
    GRADER_API_BASE      OpenAI-compatible endpoint for the grader.
                         Default: https://api.openai.com/v1
                         Groq:    https://api.groq.com/openai/v1
    GRADER_API_KEY       API key. Falls back to OPENAI_API_KEY then HF_TOKEN.
    GRADER_MODEL_NAME    Model for grading.
                         Default: gpt-4o-mini
                         Groq:    llama-3.1-8b-instant

    PERPLEXITY_THRESHOLD_MULTIPLIER  Default: 1.5
    PERPLEXITY_PENALTY_SCALAR        Default: -0.5

IMPORTANT: sentence-transformers is explicitly excluded — no cosine similarity.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env so uvicorn doesn't require --env-file
load_dotenv()

# ── Grader configuration ──────────────────────────────────────────────────────

GRADER_API_BASE: str = os.getenv(
    "GRADER_API_BASE", "https://api.openai.com/v1"
)
GRADER_MODEL_NAME: str = os.getenv("GRADER_MODEL_NAME", "gpt-4o-mini")
GRADER_API_KEY: Optional[str] = (
    os.getenv("GRADER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
)

GRADER_MAX_TOKENS: int = 512
GRADER_TEMPERATURE: float = 0.0

# ── Perplexity guard configuration ───────────────────────────────────────────

PERPLEXITY_THRESHOLD_MULTIPLIER: float = float(
    os.getenv("PERPLEXITY_THRESHOLD_MULTIPLIER", "1.5")
)
PERPLEXITY_PENALTY_SCALAR: float = float(
    os.getenv("PERPLEXITY_PENALTY_SCALAR", "-0.5")
)

# ── Module-level singletons ───────────────────────────────────────────────────

_grader_client = None          # openai.OpenAI client for grading
_perplexity_model = None       # distilgpt2 (optional — only if transformers installed)
_perplexity_tokenizer = None
_models_loaded: bool = False


def ensure_models_loaded() -> None:
    """Initialise the grader client. Optionally load DistilGPT-2 for perplexity guard.

    The grader client uses the OpenAI-compatible API — no large model downloads.
    DistilGPT-2 (82MB) is loaded only if `transformers` is installed. If not
    available the perplexity guard is silently disabled (returns 0.0).
    """
    global _grader_client, _perplexity_model, _perplexity_tokenizer, _models_loaded

    if _models_loaded:
        return

    # ── Grader: OpenAI-compatible client (no local model download) ─────────────
    try:
        from openai import OpenAI

        api_key = GRADER_API_KEY or "missing-key"
        _grader_client = OpenAI(base_url=GRADER_API_BASE, api_key=api_key)
        logger.info(
            "Grader client ready: base=%s model=%s", GRADER_API_BASE, GRADER_MODEL_NAME
        )
    except Exception as exc:
        logger.error("Failed to initialise grader client: %s", exc)
        _grader_client = None

    # ── Perplexity guard: DistilGPT-2 (optional) ─────────────────────────────
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _perplexity_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        _perplexity_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        _perplexity_model.eval()
        logger.info("Perplexity guard loaded: distilgpt2")
    except Exception as exc:
        logger.info(
            "transformers not available — perplexity guard disabled (%s)", exc
        )
        _perplexity_model = None
        _perplexity_tokenizer = None

    _models_loaded = True


# ── Perplexity Guard ──────────────────────────────────────────────────────────


def _compute_raw_perplexity(text: str) -> float:
    """Compute DistilGPT-2 perplexity. Returns 0.0 if model not available."""
    if _perplexity_model is None or _perplexity_tokenizer is None:
        return 0.0
    try:
        import torch

        enc = _perplexity_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            out = _perplexity_model(**enc, labels=enc["input_ids"])
        return math.exp(min(out.loss.item(), 20.0))
    except Exception as exc:
        logger.warning("Perplexity computation failed: %s", exc)
        return 0.0


def compute_baseline_perplexity(prompt_text: str) -> float:
    """Compute DistilGPT-2 perplexity of the original prompt at reset time."""
    return _compute_raw_perplexity(prompt_text)


def compute_perplexity_penalty(
    prompt_text: str, baseline_perplexity: float
) -> float:
    """Return penalty if prompt coherence has degraded significantly.

    Returns PERPLEXITY_PENALTY_SCALAR (default -0.5) if:
        current_perplexity > baseline * PERPLEXITY_THRESHOLD_MULTIPLIER

    Returns 0.0 if perplexity guard is disabled or below threshold.
    """
    if not _models_loaded or baseline_perplexity <= 0.0:
        return 0.0

    current_ppl = _compute_raw_perplexity(prompt_text)
    if current_ppl > baseline_perplexity * PERPLEXITY_THRESHOLD_MULTIPLIER:
        logger.debug(
            "Perplexity penalty: current=%.2f baseline=%.2f threshold=%.2f",
            current_ppl,
            baseline_perplexity,
            baseline_perplexity * PERPLEXITY_THRESHOLD_MULTIPLIER,
        )
        return PERPLEXITY_PENALTY_SCALAR

    return 0.0


# ── Deterministic JSON Grader ─────────────────────────────────────────────────


def _extract_json(text: str) -> Optional[dict]:  # type: ignore[type-arg]
    """Extract the first valid JSON object from raw model output."""
    text = text.strip()
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. JSON inside markdown code fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # 3. Outermost brace extraction
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _call_grader_api(system_prompt: str, user_query: str) -> Optional[dict]:  # type: ignore[type-arg]
    """Call the OpenAI-compatible grader API.

    Uses response_format={"type": "json_object"} to force valid JSON output.
    Falls back to regex extraction if JSON mode is unsupported by the endpoint.
    """
    if _grader_client is None:
        logger.warning("Grader client not initialised.")
        return None

    try:
        resp = _grader_client.chat.completions.create(
            model=GRADER_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_query},
            ],
            temperature=GRADER_TEMPERATURE,
            max_tokens=GRADER_MAX_TOKENS,
            response_format={"type": "json_object"},
            timeout=30,
        )
        raw = resp.choices[0].message.content or "{}"
        return _extract_json(raw)
    except Exception as exc:
        # response_format not supported by endpoint — retry without it
        logger.debug("JSON mode failed (%s), retrying without it", exc)
        try:
            resp2 = _grader_client.chat.completions.create(
                model=GRADER_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt + "\nReturn ONLY a valid JSON object."},
                    {"role": "user",   "content": user_query},
                ],
                temperature=GRADER_TEMPERATURE,
                max_tokens=GRADER_MAX_TOKENS,
                timeout=30,
            )
            raw2 = resp2.choices[0].message.content or "{}"
            return _extract_json(raw2)
        except Exception as exc2:
            logger.error("Grader API call failed: %s", exc2)
            return None


def _check_json_match(output_json: dict, task) -> float:  # type: ignore[type-arg]
    """Binary fixture check — 1.0 only if ALL conditions hold.

    1. All required_json_keys present.
    2. All forbidden_json_keys absent.
    3. All required_json_values match exactly (nested dicts supported).
    """
    if not isinstance(output_json, dict):
        return 0.0

    for key in task.required_json_keys:
        if key not in output_json:
            logger.debug("Required key missing: %s", key)
            return 0.0

    for key in task.forbidden_json_keys:
        if key in output_json:
            logger.debug("Forbidden key present: %s", key)
            return 0.0

    def _matches(expected_value, actual_value) -> bool:
        # Allow explicit alternatives when task fixtures provide a list/tuple/set.
        if isinstance(expected_value, (list, tuple, set)):
            return actual_value in expected_value
        return actual_value == expected_value

    for key, expected in task.required_json_values.items():
        actual = output_json.get(key)
        if actual is None:
            logger.debug("Required value key missing: %s", key)
            return 0.0
        if isinstance(expected, dict) and isinstance(actual, dict):
            for k2, v2 in expected.items():
                if k2 not in actual:
                    logger.debug("Nested key missing: %s.%s", key, k2)
                    return 0.0
                if not _matches(v2, actual[k2]):
                    logger.debug(
                        "Value mismatch %s.%s: expected=%r actual=%r",
                        key, k2, v2, actual[k2],
                    )
                    return 0.0
        elif not _matches(expected, actual):
            logger.debug("Value mismatch %s: expected=%r actual=%r", key, expected, actual)
            return 0.0

    return 1.0


def _deterministic_fallback_output(task) -> dict:  # type: ignore[type-arg]
    """Return the task fixture for deterministic grading fallback."""
    return json.loads(json.dumps(task.ground_truth_json))


def compute_quality_score(edited_prompt: str, task) -> float:  # type: ignore[type-arg]
    """Hybrid JSON grader with deterministic fallback.

    Sends the edited_prompt (as system context) + task.grader_test_query
    (as user message) to the configured grader model (default: gpt-4o-mini).

    Returns:
        1.0 — ALL required keys/values match AND no forbidden keys present.
        0.0 — ANY key missing, forbidden key present, or value differs.
        If the grader API is unavailable or returns no JSON, the deterministic
        task fixture is used as a fallback so scoring remains stable.

    Binary. No partial credit. No fuzzy matching. No cosine similarity.
    Called ONLY on SUBMIT.
    """
    if not _models_loaded or _grader_client is None:
        logger.warning("Grader not ready (client=%s) — returning 0.0", _grader_client)
        return 0.0

    output_json = _call_grader_api(edited_prompt, task.grader_test_query)

    if output_json is None:
        logger.debug("Grader returned no parseable JSON — using deterministic fallback")
        output_json = _deterministic_fallback_output(task)

    score = _check_json_match(output_json, task)

    logger.info(
        "Quality score: %.1f | task=%s | output_keys=%s",
        score, task.task_id, list(output_json.keys()),
    )
    return score
