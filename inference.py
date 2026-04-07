"""OpenEnv baseline inference runner.

Mandatory environment variables for submission setups:
- API_BASE_URL   The API endpoint for the LLM.
- MODEL_NAME     The model identifier to use for inference.
- HF_TOKEN       Your Hugging Face / API key.
- LOCAL_IMAGE_NAME  The name of the local image to use for the environment
                    if you are using from_docker_image() method.

This script uses OpenAI Client for all LLM calls and prints only these line
types for each task episode:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ── Mandatory env vars (per submission spec) ─────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")  # docker-image launch

BENCHMARK: str = os.getenv("OPENENV_BENCHMARK", "github-issue-triage")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
TASKS: List[str] = ["task_easy", "task_medium", "task_hard"]
MAX_STEPS_FALLBACK = 20

SYSTEM_PROMPT = (
    "You are triaging GitHub issues. Return ONLY JSON with keys: "
    "labels, priority, is_duplicate, duplicate_of, needs_info, comment, is_security, close."
)

DEFAULT_ACTION = {
    "labels": ["bug"],
    "priority": "P2",
    "is_duplicate": False,
    "duplicate_of": None,
    "needs_info": False,
    "comment": None,
    "is_security": False,
    "close": False,
}


def _stderr(msg: str) -> None:
    print(msg, file=os.sys.stderr, flush=True)


def _escape_single_line(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").strip()


def log_start(task: str, env: str = BENCHMARK, model: str = MODEL_NAME) -> None:
    """Emit one [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action_str: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit one [STEP] line immediately after env.step() returns."""
    done_val = str(done).lower()
    error_val = "null" if error is None else _escape_single_line(error)
    action_val = _escape_single_line(action_str)
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit one [END] line always — even on exception."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def _safe_parse_action(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return dict(DEFAULT_ACTION)
        out = dict(DEFAULT_ACTION)
        out.update({k: v for k, v in data.items() if k in out})
        if not isinstance(out["labels"], list):
            out["labels"] = DEFAULT_ACTION["labels"]
        if out["priority"] not in {"P0", "P1", "P2", "P3"}:
            out["priority"] = DEFAULT_ACTION["priority"]
        return out
    except Exception:
        return dict(DEFAULT_ACTION)


def _heuristic_action(obs: dict[str, Any]) -> dict[str, Any]:
    issue = obs.get("current_issue", {})
    text = f"{issue.get('title', '')}\n{issue.get('body', '')}".lower()

    action = dict(DEFAULT_ACTION)

    labels = []
    if "feature" in text or "support" in text or "add " in text:
        labels.append("enhancement")
        action["priority"] = "P3"
    elif "doc" in text or "documentation" in text or "guide" in text:
        labels.append("documentation")
        action["priority"] = "P2"
    else:
        labels.append("bug")
        action["priority"] = "P2"

    if "security" in text or "xss" in text or "token" in text or "bypass" in text:
        action["is_security"] = True
        action["priority"] = "P0"
        if "security" not in labels:
            labels.append("security")

    if "not sure" in text or "please help" in text or "just says" in text:
        action["needs_info"] = True
        labels.append("needs-reproduction")
        action["comment"] = "Please share sdk version, os/python version, and full traceback."

    action["labels"] = sorted(set(labels))
    return action


def _choose_action(client: OpenAI, obs: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(obs, ensure_ascii=True)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        return _safe_parse_action(raw)
    except Exception as exc:
        _stderr(f"[DEBUG] model request failed: {exc}")
        return _heuristic_action(obs)


def _task_success(score: float) -> bool:
    return score >= 0.5


def _normalize_score(rewards: list[float]) -> float:
    if not rewards:
        return 0.0
    avg_reward = sum(rewards) / len(rewards)
    # Reward range is [-0.40, 1.00] per step -> map to [0, 1]
    normalized = (avg_reward + 0.40) / 1.40
    return max(0.0, min(1.0, normalized))


def run_task(client: OpenAI, task_id: str) -> None:
    """Run one full episode for task_id, emitting [START] / [STEP]* / [END]."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        max_steps = int(obs.get("max_steps", MAX_STEPS_FALLBACK))

        for step in range(1, max_steps + 1):
            action = _choose_action(client, obs)
            action_str = json.dumps(action, separators=(",", ":"), ensure_ascii=True)

            step_resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action,
                timeout=30,
            )
            if step_resp.status_code != 200:
                err = _escape_single_line(step_resp.text)
                log_step(step, action_str, 0.0, True, err)
                break

            result = step_resp.json()
            reward = float(result.get("reward", {}).get("total", 0.0))
            done = bool(result.get("done", False))
            info = result.get("info", {})
            err: Optional[str] = info.get("last_action_error") or None

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, err)

            if done:
                break
            obs = result["observation"]

        score = _normalize_score(rewards)
        success = _task_success(score)

    except Exception as exc:
        _stderr(f"[DEBUG] task failed: {exc}")

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY")
    if not api_key:
        _stderr("[DEBUG] missing HF_TOKEN/OPENAI_API_KEY; model calls will fail over to heuristic")
        api_key = "missing-key"

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
