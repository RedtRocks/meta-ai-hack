"""Core RL environment for GitHub Issue Triage."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from env.models import Action, Issue, Observation, Reward
from env.reward import compute_reward

_DATA_DIR = Path(__file__).resolve().parent / "data"


class GitHubTriageEnv:
    """OpenEnv-compliant environment for triaging GitHub issues.

    Lifecycle:
        env = GitHubTriageEnv()
        obs = env.reset("task_easy")
        obs, reward, done, info = env.step(action)
    """

    def __init__(self) -> None:
        with open(_DATA_DIR / "tasks.json", encoding="utf-8") as f:
            self.tasks: dict[str, Any] = json.load(f)
        with open(_DATA_DIR / "issues_pool.json", encoding="utf-8") as f:
            self.issue_pool: list[dict] = json.load(f)
        with open(_DATA_DIR / "label_schema.json", encoding="utf-8") as f:
            self.label_schema: dict = json.load(f)

        self._state: dict | None = None

    # ── Public API ───────────────────────────────────────────────────────

    def reset(self, task_id: str) -> Observation:
        """Start (or restart) an episode for the given task.

        Raises ``ValueError`` if *task_id* is not in the task catalogue.
        """
        if task_id not in self.tasks:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Available: {list(self.tasks.keys())}"
            )

        task = self.tasks[task_id]
        self._state = {
            "task_id": task_id,
            "step": 0,
            "done": False,
            "actions_taken": [],
            "rewards_history": [],
            "issue_queue": copy.deepcopy(task["current_issues"]),
            "gold_queue": copy.deepcopy(task["gold"]),
            "action_counts": {},
        }
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Apply an action to the current issue and advance the queue.

        Returns (observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        if self._state["done"]:
            raise RuntimeError("Episode is done. Call reset().")

        current_issue_id: str = self._state["issue_queue"][0]["id"]
        gold: dict = self._state["gold_queue"][0]

        # ── Loop guard ───────────────────────────────────────────────────
        count = self._state["action_counts"].get(current_issue_id, 0) + 1
        self._state["action_counts"][current_issue_id] = count
        if count >= 3:
            self._state["done"] = True
            zero_reward = Reward(
                total=0.0,
                label_score=0.0,
                duplicate_score=0.0,
                priority_score=0.0,
                comment_score=0.0,
                security_score=0.0,
                breakdown={"reason": "loop_guard"},
            )
            return (
                self._build_observation(),
                zero_reward,
                True,
                {"step": self._state["step"]},
            )

        # ── Compute reward ───────────────────────────────────────────────
        reward = compute_reward(action, gold)

        # ── Record ───────────────────────────────────────────────────────
        self._state["actions_taken"].append(action.model_dump())
        self._state["rewards_history"].append(reward.model_dump())
        self._state["step"] += 1

        # ── Advance queue ────────────────────────────────────────────────
        self._state["issue_queue"].pop(0)
        self._state["gold_queue"].pop(0)

        # ── Check done ───────────────────────────────────────────────────
        done = len(self._state["issue_queue"]) == 0
        task = self.tasks[self._state["task_id"]]
        if self._state["step"] >= task["max_steps"]:
            done = True
        self._state["done"] = done

        obs = (
            self._build_observation()
            if not done
            else self._build_final_observation()
        )
        return obs, reward, done, {
            "step": self._state["step"],
            "task_id": self._state["task_id"],
        }

    def state(self) -> dict:
        """Return a deep copy of the internal state (safe to inspect)."""
        return copy.deepcopy(self._state)

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        """Build an observation from the current state."""
        task = self.tasks[self._state["task_id"]]  # type: ignore[index]
        current_issue = Issue(**self._state["issue_queue"][0])  # type: ignore[index]
        return Observation(
            task_id=self._state["task_id"],  # type: ignore[index]
            repo_name=task["repo_name"],
            repo_description=task["repo_description"],
            label_schema=self.label_schema["labels"],
            current_issue=current_issue,
            existing_issues=[Issue(**i) for i in self.issue_pool],
            step_number=self._state["step"],  # type: ignore[index]
            max_steps=task["max_steps"],
            issues_remaining=len(self._state["issue_queue"]),  # type: ignore[index]
        )

    def _build_final_observation(self) -> Observation:
        """Build a terminal observation with a placeholder issue."""
        task = self.tasks[self._state["task_id"]]  # type: ignore[index]
        dummy_issue = Issue(
            id="done",
            title="Episode complete",
            body="",
            author="system",
            created_at="1970-01-01T00:00:00Z",
            tags=[],
        )
        return Observation(
            task_id=self._state["task_id"],  # type: ignore[index]
            repo_name=task["repo_name"],
            repo_description=task["repo_description"],
            label_schema=self.label_schema["labels"],
            current_issue=dummy_issue,
            existing_issues=[Issue(**i) for i in self.issue_pool],
            step_number=self._state["step"],  # type: ignore[index]
            max_steps=task["max_steps"],
            issues_remaining=0,
        )
