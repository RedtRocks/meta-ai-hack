"""PromptForge Environment Client.

Provides a typed WebSocket client for interacting with the PromptForge
openenv-core server from training scripts and test runners.

Usage:
    from promptforge.client import PromptForgeEnvClient

    with PromptForgeEnvClient(base_url="http://localhost:7860") as client:
        result = client.reset()
        obs = result.observation

        result = client.step(PromptForgeAction(
            action_type="START_EPISODE", task_difficulty="hard"
        ))
        obs = result.observation
        print(obs.task_difficulty, obs.ast_summary)

        result = client.step(PromptForgeAction(
            action_type="PRUNE_BRANCH", node_id=obs.ast_summary[2]["node_id"]
        ))
        result = client.step(PromptForgeAction(action_type="SUBMIT"))
        print(f"done={result.done}, reward={result.reward}")
"""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import PromptForgeAction, PromptForgeObservation
except ImportError:
    import os as _os, sys as _sys

    _sys.path.insert(0, _os.path.dirname(__file__))
    from models import PromptForgeAction, PromptForgeObservation


class PromptForgeEnvClient(
    EnvClient[PromptForgeAction, PromptForgeObservation, State]
):
    """Typed WebSocket client for the PromptForge environment server.

    Each client instance gets its own isolated episode on the server
    (via SUPPORTS_CONCURRENT_SESSIONS = True).
    """

    def _step_payload(self, action: PromptForgeAction) -> Dict:  # type: ignore[type-arg]
        """Convert PromptForgeAction to the JSON payload for the step message."""
        payload = {"action_type": action.action_type}
        if action.node_id is not None:
            payload["node_id"] = action.node_id
        if action.node_id_2 is not None:
            payload["node_id_2"] = action.node_id_2
        if action.target_parent_id is not None:
            payload["target_parent_id"] = action.target_parent_id
        if action.task_difficulty is not None:
            payload["task_difficulty"] = action.task_difficulty
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[PromptForgeObservation]:  # type: ignore[type-arg]
        """Parse server response into StepResult[PromptForgeObservation]."""
        obs_data = payload.get("observation", {})
        observation = PromptForgeObservation(
            raw_prompt=obs_data.get("raw_prompt", ""),
            ast_summary=obs_data.get("ast_summary", []),
            current_token_count=obs_data.get("current_token_count", 0),
            original_token_count=obs_data.get("original_token_count", 0),
            token_reduction_pct=obs_data.get("token_reduction_pct", 0.0),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 20),
            probe_budget_remaining=obs_data.get("probe_budget_remaining", 5),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            last_action_result=obs_data.get("last_action_result"),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:  # type: ignore[type-arg]
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
