"""Baseline inference agent using OpenAI API.

Usage:
    # Start the server first:
    #   uvicorn app:app --port 7860
    #
    # Then run this script:
    #   OPENAI_API_KEY=sk-... python inference.py
    #   python inference.py --base-url http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import sys

import requests
from openai import OpenAI

SYSTEM_PROMPT = """\
You are an experienced open source maintainer triaging GitHub issues.
You will receive a JSON object describing the current issue and the repo context.
You must respond with ONLY a valid JSON object — no markdown, no explanation.

The JSON must match this exact schema:
{
  "labels": ["label1", "label2"],
  "priority": "P0"|"P1"|"P2"|"P3",
  "is_duplicate": true|false,
  "duplicate_of": "existing-007"|null,
  "needs_info": true|false,
  "comment": "string or null",
  "is_security": true|false,
  "close": true|false
}

Priority guide: P0=security/data loss, P1=major bug, P2=minor bug, P3=enhancement.
Security issues: authentication bypass, data exposure, injection, privilege escalation.
Duplicate detection: check existing_issues carefully for same root cause.
needs_info: set true only if reproduction steps or environment info is missing.
If needs_info=true, comment must ask for the specific missing information.
"""

DEFAULT_ACTION = {
    "labels": [],
    "priority": "P3",
    "is_duplicate": False,
    "duplicate_of": None,
    "needs_info": False,
    "comment": None,
    "is_security": False,
    "close": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline triage agent")
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="Base URL of the triage environment server (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    return parser.parse_args()


def safe_parse_action(raw: str) -> dict:
    """Parse the model's JSON response, falling back to a safe default."""
    try:
        data = json.loads(raw)
        # Minimal validation
        if not isinstance(data, dict):
            return dict(DEFAULT_ACTION)
        # Ensure required keys exist with sane defaults
        action = dict(DEFAULT_ACTION)
        action.update({k: v for k, v in data.items() if k in DEFAULT_ACTION})
        return action
    except (json.JSONDecodeError, TypeError, KeyError):
        return dict(DEFAULT_ACTION)


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    client = OpenAI()  # reads OPENAI_API_KEY from env

    tasks = ["task_easy", "task_medium", "task_hard"]

    for task_id in tasks:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id}")
        print(f"{'='*60}")

        # Reset
        resp = requests.post(f"{base_url}/reset", json={"task_id": task_id})
        if resp.status_code != 200:
            print(f"ERROR resetting: {resp.text}", file=sys.stderr)
            continue
        obs = resp.json()

        total_reward = 0.0
        step = 0
        done = False

        while not done:
            # Call model
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(obs)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                raw_content = response.choices[0].message.content or "{}"
                action_json = safe_parse_action(raw_content)
            except Exception as exc:
                print(f"  Model error: {exc}", file=sys.stderr)
                action_json = dict(DEFAULT_ACTION)

            # Step
            resp = requests.post(f"{base_url}/step", json=action_json)
            if resp.status_code != 200:
                print(f"  Step error: {resp.text}", file=sys.stderr)
                break
            result = resp.json()

            reward = result["reward"]["total"]
            total_reward += reward
            done = result["done"]
            obs = result["observation"]
            step += 1

            print(
                json.dumps(
                    {
                        "task": task_id,
                        "step": step,
                        "reward": reward,
                        "done": done,
                        "label_score": result["reward"]["label_score"],
                        "security_score": result["reward"]["security_score"],
                    }
                )
            )

        print(
            json.dumps(
                {
                    "task": task_id,
                    "total_reward": round(total_reward, 3),
                    "steps": step,
                }
            )
        )


if __name__ == "__main__":
    main()
