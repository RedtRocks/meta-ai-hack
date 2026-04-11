"""PromptForge Pydantic models using official openenv-core base classes.

PromptForgeAction holds all possible action fields in one flat model
(required by openenv-core's extra="forbid" constraint on Action).

PromptForgeObservation inherits `done`, `reward`, `metadata` from the
official base and adds all AST-level observation fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── Action ────────────────────────────────────────────────────────────────────

class PromptForgeAction(Action):
    """Flat action model for all five PromptForge action types.

    Uses a single model (required by openenv-core's extra="forbid" constraint).
    Fields not relevant to the selected action_type should be omitted / None.

    action_type semantics:
        START_EPISODE   — (Re)sets the environment; use before any edits.
        PRUNE_BRANCH    — Permanently removes a node and all descendants.
        MOVE_NODE       — Relocates a node to a different parent section.
        MERGE_NODES     — Combines two overlapping nodes into one.
        PROBE           — Non-destructive coherence test (AST restored after).
        SUBMIT          — Finalises episode; triggers Deterministic JSON Grader.
    """

    action_type: Literal[
        "START_EPISODE",
        "PRUNE_BRANCH",
        "MOVE_NODE",
        "MERGE_NODES",
        "PROBE",
        "SUBMIT",
    ] = Field(..., description="Which action to execute")

    # START_EPISODE
    task_difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        default=None,
        description="Task difficulty for START_EPISODE. Ignored for other actions.",
    )

    # PRUNE_BRANCH / PROBE / MOVE_NODE / MERGE_NODES
    node_id: Optional[str] = Field(
        default=None,
        description="node_id from ast_summary. Required for PRUNE_BRANCH, PROBE, MOVE_NODE, MERGE_NODES.",
    )

    # MOVE_NODE
    target_parent_id: Optional[str] = Field(
        default=None,
        description="Target parent node_id for MOVE_NODE.",
    )

    # MERGE_NODES
    node_id_2: Optional[str] = Field(
        default=None,
        description="Second node_id for MERGE_NODES (merged into node_id).",
    )


# ── Observation ───────────────────────────────────────────────────────────────

class NodeSummary(dict):
    """Lightweight JSON-serialisable node summary (typed alias for dict).

    Fields: node_id, node_type, token_count, content_preview (first 80 chars).
    """


class PromptForgeObservation(Observation):
    """Full observation returned by reset() and step().

    Inherits `done`, `reward`, `metadata` from openenv-core Observation base.
    All additional fields describe the current AST state.
    """

    raw_prompt: str = Field(
        default="",
        description="Current serialised prompt text (editable tree state).",
    )
    ast_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-node list: {node_id, node_type, token_count, content_preview}",
    )
    current_token_count: int = Field(default=0, description="Tokens in current (edited) prompt")
    original_token_count: int = Field(default=0, description="Tokens in bloated baseline prompt")
    token_reduction_pct: float = Field(default=0.0, description="Compression achieved so far (%)")
    step_count: int = Field(default=0, description="Steps taken this episode")
    max_steps: int = Field(default=20, description="Episode horizon")
    probe_budget_remaining: int = Field(default=5, description="Remaining PROBE actions")
    task_difficulty: str = Field(default="easy", description="easy | medium | hard")
    last_action_result: Optional[str] = Field(
        default=None, description="Human-readable outcome of the last action"
    )
