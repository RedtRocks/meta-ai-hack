"""Deterministic grader functions for each task difficulty level."""

from __future__ import annotations

from typing import List

from env.models import Action
from env.reward import compute_reward


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets.

    Returns 1.0 when both sets are empty, 0.0 when only one is empty.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def grade_task_easy(action: Action, gold: dict) -> float:
    """Grade a single-step easy task.

    Returns the total reward directly (0.0 – 1.0).
    """
    return compute_reward(action, gold).total


def grade_task_medium(actions: List[Action], golds: List[dict]) -> float:
    """Grade a 5-issue medium task.

    Returns the average total reward across all pairs, rounded to 3 dp.
    """
    if len(actions) != len(golds):
        raise ValueError(
            f"Expected {len(golds)} actions, got {len(actions)}"
        )
    rewards = [compute_reward(a, g).total for a, g in zip(actions, golds)]
    return round(sum(rewards) / len(rewards), 3)


def grade_task_hard(actions: List[Action], golds: List[dict]) -> float:
    """Grade a 10-issue hard task with an extra security penalty.

    Steps:
    1. Compute the average total reward across all 10 pairs.
    2. Find the gold entry with is_security=True and the matching action.
    3. If the agent failed to flag it (is_security=False), subtract 0.40.
    4. Clamp the result to [0.0, 1.0] and round to 3 dp.
    """
    if len(actions) != len(golds):
        raise ValueError(
            f"Expected {len(golds)} actions, got {len(actions)}"
        )

    rewards = [compute_reward(a, g).total for a, g in zip(actions, golds)]
    avg = sum(rewards) / len(rewards)

    # Additional security check
    for idx, g in enumerate(golds):
        if g.get("is_security", False):
            if not actions[idx].is_security:
                avg -= 0.40
            break  # only one security issue expected

    result = round(max(0.0, min(1.0, avg)), 3)
    return result
