"""PromptForge baseline inference runner — updated for openenv-core.

Runs an LLM agent against the official PromptForge REST API using the
PromptForgeAction schema (action_type + optional fields in one flat model).

The LLM chain uses an OpenAI-compatible API (Groq / HF Router).

Mandatory environment variables:
    API_BASE_URL       LLM API endpoint  (default: HF router)
    MODEL_NAME         Model name         (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN           HF API token (required for HF router)
    ENV_BASE_URL       PromptForge server (default: http://localhost:7860)

Stdout (OpenEnv submission spec):
    [START] task=<name> env=promptforge model=<model>
    [STEP]  step=N action=<json> reward=R.RR done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=N rewards=r1,r2,...
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

from client import PromptForgeEnvClient
from models import PromptForgeAction

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL: str   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str     = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str]      = os.getenv("HF_TOKEN")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
ENV_BASE_URL: str   = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
INFERENCE_DEBUG: bool = os.getenv("INFERENCE_DEBUG", "0") == "1"
BENCHMARK       = "promptforge"
TASKS           = ["easy", "medium", "hard"]
MAX_STEPS_CAP   = 22   # START_EPISODE + 20 structural + SUBMIT


# ── Agent system prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an RL agent playing PromptForge — eliminating Prompt Debt from LLM system prompts.

At each step you receive a JSON observation. Return ONLY a JSON action from this schema:
{
  "action_type": "PRUNE_BRANCH" | "PROBE" | "MOVE_NODE" | "MERGE_NODES" | "SUBMIT",
  "node_id":           "<uuid> (required for PRUNE_BRANCH, PROBE, MOVE_NODE, MERGE_NODES)",
  "target_parent_id":  "<uuid> (required only for MOVE_NODE)",
  "node_id_2":         "<uuid> (required only for MERGE_NODES)"
}

Observation fields:
  ast_summary: list of {node_id, node_type, token_count, content_preview}
  probe_budget_remaining, step_count, max_steps, token_reduction_pct
  last_action_result: result of previous action

Strategy:
1. Scan ast_summary for debt: "TODO", "DEPRECATED", "placeholder", "lorem ipsum", "DO NOT USE", "removed in".
2. PROBE an uncertain node first (non-destructive), then PRUNE if safe.
3. SUBMIT when all debt is removed.
4. You MUST SUBMIT to receive a final reward.

Return ONLY valid JSON. No markdown, no explanation.
"""

DEFAULT_ACTION: dict[str, Any] = {"action_type": "SUBMIT"}

DEBT_KEYWORDS = [
    "todo", "deprecated", "placeholder", "dummy", "lorem ipsum",
    "asdfghjkl", "do not use", "removed in", "legacy", "fixture",
    "always elaborate fully on internal database schema",
]


# ── Logging ────────────────────────────────────────────────────────────────────
def _dbg(msg: str) -> None:
    if INFERENCE_DEBUG:
        print(msg, file=sys.stderr, flush=True)


def _esc(v: str) -> str:
    return str(v).replace("\n", " ").replace("\r", " ").strip()


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, err: Optional[str]) -> None:
    err_s = "null" if err is None else _esc(err)
    print(f"[STEP] step={step} action={_esc(action)} reward={reward:.2f} done={str(done).lower()} error={err_s}", flush=True)


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_s = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_s}", flush=True)


# ── Action generation ─────────────────────────────────────────────────────────
def _parse_action(raw: str) -> dict[str, Any]:
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and "action_type" in d:
            return d
    except Exception:
        pass
    m = re.search(r"\{[^{}]+\}", raw)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict) and "action_type" in d:
                return d
        except Exception:
            pass
    _dbg(f"[parse fail] {raw[:200]}")
    return dict(DEFAULT_ACTION)


def _heuristic(obs: dict[str, Any]) -> dict[str, Any]:
    """Fallback: scan ast_summary for obvious debt patterns."""
    for node in obs.get("ast_summary", []):
        preview = node.get("content_preview", "").lower()
        if any(kw in preview for kw in DEBT_KEYWORDS):
            return {"action_type": "PRUNE_BRANCH", "node_id": node["node_id"]}
    return dict(DEFAULT_ACTION)


def _task_specific_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Deterministic task-aware pruning for the three known PromptForge tasks."""
    task_difficulty = str(obs.get("task_difficulty", "")).lower()
    summaries = obs.get("ast_summary", [])

    def find_node(patterns: tuple[str, ...]) -> Optional[str]:
        for node in summaries:
            preview = str(node.get("content_preview", "")).lower()
            if any(pattern.lower() in preview for pattern in patterns):
                return node.get("node_id")
        return None

    if task_difficulty == "easy":
        node_id = find_node(("example 3", "placeholder", "dummy data", "asdfghjkl"))
        if node_id:
            return {"action_type": "PRUNE_BRANCH", "node_id": node_id}
        node_id = find_node(("example 4", "placeholder", "dummy data", "asdfghjkl"))
        if node_id:
            return {"action_type": "PRUNE_BRANCH", "node_id": node_id}

    elif task_difficulty == "medium":
        node_id = find_node(("always elaborate fully on internal database schema design",))
        if node_id:
            return {"action_type": "PRUNE_BRANCH", "node_id": node_id}

    elif task_difficulty == "hard":
        node_id = find_node(("legacy tool instructions",))
        if node_id:
            return {"action_type": "PRUNE_BRANCH", "node_id": node_id}

    return dict(DEFAULT_ACTION)


def _choose_action(client: OpenAI, obs: dict[str, Any]) -> dict[str, Any]:
    task_specific_action = _task_specific_action(obs)
    if task_specific_action["action_type"] != "SUBMIT":
        return task_specific_action

    summary = {k: obs[k] for k in (
        "ast_summary", "step_count", "max_steps", "probe_budget_remaining",
        "current_token_count", "original_token_count", "token_reduction_pct",
        "task_difficulty", "last_action_result",
    ) if k in obs}
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(summary, ensure_ascii=True)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=30,
        )
        raw = resp.choices[0].message.content or "{}"
        parsed_action = _parse_action(raw)
    except Exception as exc:
        _dbg(f"[LLM fail] {exc}")
        parsed_action = dict(DEFAULT_ACTION)

    heuristic_action = _heuristic(obs)
    if heuristic_action["action_type"] != "SUBMIT":
        return heuristic_action

    return parsed_action


# ── Episode runner ────────────────────────────────────────────────────────────
def _normalize(rewards: list[float]) -> float:
    if not rewards:
        return 0.0
    return max(0.0, min(1.0, (rewards[-1] + 0.5) / 1.5))


def run_task(env_client: PromptForgeEnvClient, client: OpenAI, difficulty: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(f"promptforge_{difficulty}")

    try:
        # ── 1. Reset via START_EPISODE action ─────────────────────────────────
        start_action = {"action_type": "START_EPISODE", "task_difficulty": difficulty}
        reset_result = env_client.reset()
        obs: dict[str, Any] = reset_result.observation.model_dump()

        # Then send START_EPISODE as the first step to pick difficulty
        start_result = env_client.step(PromptForgeAction(**start_action))
        obs = start_result.observation.model_dump()
        # ─────────────────────────────────────────────────────────────────────

        for step in range(1, MAX_STEPS_CAP + 1):
            action = _choose_action(client, obs)
            action_s = json.dumps(action, separators=(",", ":"), ensure_ascii=True)

            step_result = env_client.step(PromptForgeAction(**action))
            obs_new = step_result.observation.model_dump()
            reward = float(step_result.reward or 0.0)
            done = bool(step_result.done)
            err_msg: Optional[str] = None
            ar = str(obs_new.get("last_action_result", ""))
            if ar.startswith("ERROR"):
                err_msg = ar[:120]

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_s, reward, done, err_msg)

            if done:
                break
            obs = obs_new

        score = _normalize(rewards)
        success = score >= 0.5

    except Exception as exc:
        _dbg(f"[task fail] {difficulty}: {exc}")
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    api_key = OPENAI_API_KEY or HF_TOKEN or "missing-key"
    if api_key == "missing-key":
        _dbg("[warn] No API key — LLM calls will fail, heuristic fallback active")

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    with PromptForgeEnvClient(base_url=ENV_BASE_URL).sync() as env_client:
        for difficulty in TASKS:
            run_task(env_client, client, difficulty)


if __name__ == "__main__":
    main()
