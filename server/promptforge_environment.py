"""PromptForge Environment — official openenv-core implementation.

Inherits from openenv.core.env_server.interfaces.Environment and satisfies:
    reset()          -> PromptForgeObservation
    step(action)     -> PromptForgeObservation
    @property state  -> State

Full episode semantics are documented in env/environment.py; this module
bridges the official openenv-core interface to the existing logic.

SUPPORTS_CONCURRENT_SESSIONS = True — each WebSocket client gets its own
PromptForgeEnvironment instance through the create_app factory pathway.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PromptForgeAction, PromptForgeObservation
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from models import PromptForgeAction, PromptForgeObservation

from .ast_parser import (
    PromptAST,
    _count_tokens,
    _update_token_counts,
    get_subtree_node_ids,
    parse_prompt,
    serialize_ast,
)
from .graders import (
    compute_baseline_perplexity,
    compute_perplexity_penalty,
    compute_quality_score,
    ensure_models_loaded,
)
from .reward import compute_reward
from .tasks import get_task

logger = logging.getLogger(__name__)

MAX_STEPS = 20
PROBE_BUDGET = 5


class PromptForgeEnvironment(Environment):
    """OpenEnv-compliant RL environment for Prompt Debt elimination.

    Compatible with openenv-core's create_app factory pattern:
        - SUPPORTS_CONCURRENT_SESSIONS = True
        - reset() and step() are synchronous
        - @property state returns openenv.core State object
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        ensure_models_loaded()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Optional[_EpisodeData] = None

    # ── OpenEnv-core required interface ───────────────────────────────────────

    def reset(self) -> PromptForgeObservation:
        """Reset to the EASY task (default). Use START_EPISODE action to pick difficulty."""
        return self._start_episode("easy")

    def step(self, action: PromptForgeAction) -> PromptForgeObservation:  # type: ignore[override]
        """Apply action and return updated observation.

        Special action: START_EPISODE resets the episode with a chosen difficulty.
        All other actions require an active episode (call START_EPISODE first).
        """
        if action.action_type == "START_EPISODE":
            difficulty = action.task_difficulty or "easy"
            return self._start_episode(difficulty)

        if self._episode is None:
            return self._error_obs("No active episode. Send START_EPISODE action first.")

        ep = self._episode
        result_msg = ""
        quality_score = 0.0
        perplexity_penalty = 0.0
        done = False

        # ── Dispatch ───────────────────────────────────────────────────────────
        if action.action_type == "SUBMIT":
            if ep.submitted:
                return PromptForgeObservation(
                    last_action_result="SUBMIT already called. Episode is over.",
                    done=True,
                    reward=0.0,
                )

            edited_prompt = serialize_ast(ep.current_ast)
            quality_score = compute_quality_score(edited_prompt, ep.task)
            perplexity_penalty = compute_perplexity_penalty(edited_prompt, ep.baseline_perplexity)
            ep.submitted = True
            done = True
            curr = ep.current_ast.node_registry[ep.current_ast.root_id].token_count
            pct = (ep.original_token_count - curr) / max(1, ep.original_token_count) * 100
            result_msg = (
                f"SUBMIT: quality={quality_score:.1f} | "
                f"perplexity_penalty={perplexity_penalty:.2f} | "
                f"token_reduction={pct:.1f}%"
            )
            ep.step_count += 1

        elif action.action_type == "PROBE":
            perplexity_penalty, result_msg = self._handle_probe(action, ep)

        elif action.action_type == "PRUNE_BRANCH":
            result_msg = self._handle_prune(action, ep)
            perplexity_penalty = self._check_perplexity(ep)
            ep.step_count += 1

        elif action.action_type == "MOVE_NODE":
            result_msg = self._handle_move(action, ep)
            perplexity_penalty = self._check_perplexity(ep)
            ep.step_count += 1

        elif action.action_type == "MERGE_NODES":
            result_msg = self._handle_merge(action, ep)
            perplexity_penalty = self._check_perplexity(ep)
            ep.step_count += 1

        else:
            result_msg = f"Unknown action_type: {action.action_type}"
            ep.step_count += 1

        # ── Truncation ────────────────────────────────────────────────────────
        if ep.step_count >= MAX_STEPS and not done:
            done = True
            result_msg += " [Truncated: max steps reached]"

        # ── Reward ────────────────────────────────────────────────────────────
        curr_tokens = ep.current_ast.node_registry[ep.current_ast.root_id].token_count
        reward = compute_reward(
            original_token_count=ep.original_token_count,
            current_token_count=curr_tokens,
            quality_score=quality_score,
            perplexity_penalty=perplexity_penalty,
            probe_steps_used=ep.probe_steps_used,
            is_submit=(action.action_type == "SUBMIT"),
        )

        # Update openenv-core State
        self._state.step_count = ep.step_count

        return self._build_obs(ep, result_msg, reward, done)

    @property
    def state(self) -> State:
        """Current openenv-core State object."""
        return self._state

    # ── Episode initialisation ─────────────────────────────────────────────────

    def _start_episode(self, difficulty: str) -> PromptForgeObservation:
        task = get_task(difficulty)  # type: ignore[arg-type]
        ast = parse_prompt(task.bloated_prompt)
        root = ast.node_registry[ast.root_id]
        baseline_ppl = compute_baseline_perplexity(task.bloated_prompt)
        task.baseline_token_count = root.token_count

        self._episode = _EpisodeData(
            task=task,
            current_ast=ast,
            baseline_perplexity=baseline_ppl,
            original_token_count=root.token_count,
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)

        logger.info(
            "START_EPISODE: difficulty=%s tokens=%d nodes=%d",
            difficulty, root.token_count, len(ast.node_registry),
        )
        return self._build_obs(
            self._episode,
            f"Episode started [{difficulty}]. Inspect ast_summary, then PRUNE/PROBE/SUBMIT.",
            reward=0.0,
            done=False,
        )

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_prune(self, action: PromptForgeAction, ep: "_EpisodeData") -> str:
        node_id = action.node_id
        if not node_id:
            return "ERROR: node_id is required for PRUNE_BRANCH."
        registry = ep.current_ast.node_registry
        if node_id not in registry:
            return f"ERROR: node_id '{node_id[:12]}' not found."
        if node_id == ep.current_ast.root_id:
            return "ERROR: Cannot prune the root DOCUMENT node."

        node = registry[node_id]
        preview = node.content[:50].replace("\n", " ")
        parent_id = node.parent_id
        if parent_id and parent_id in registry:
            parent = registry[parent_id]
            if node_id in parent.children:
                parent.children.remove(node_id)

        subtree = get_subtree_node_ids(node_id, registry)
        tokens_removed = sum(registry[n].token_count for n in subtree if n in registry)
        for nid in subtree:
            registry.pop(nid, None)

        _update_token_counts(ep.current_ast.root_id, registry)
        return (
            f"PRUNE_BRANCH: Removed {len(subtree)} nodes (~{tokens_removed} tokens). "
            f"Preview: '{preview}'"
        )

    def _handle_move(self, action: PromptForgeAction, ep: "_EpisodeData") -> str:
        node_id = action.node_id
        target = action.target_parent_id
        if not node_id:
            return "ERROR: node_id required for MOVE_NODE."
        if not target:
            return "ERROR: target_parent_id required for MOVE_NODE."
        registry = ep.current_ast.node_registry
        if node_id not in registry:
            return f"ERROR: node_id '{node_id[:12]}' not found."
        if target not in registry:
            return f"ERROR: target_parent_id '{target[:12]}' not found."

        node = registry[node_id]
        old_parent = node.parent_id
        if old_parent and old_parent in registry:
            p = registry[old_parent]
            if node_id in p.children:
                p.children.remove(node_id)

        registry[target].children.append(node_id)
        node.parent_id = target
        return f"MOVE_NODE: '{node_id[:8]}' -> parent '{target[:8]}' (was '{str(old_parent)[:8]}')"

    def _handle_merge(self, action: PromptForgeAction, ep: "_EpisodeData") -> str:
        nid1, nid2 = action.node_id, action.node_id_2
        if not nid1 or not nid2:
            return "ERROR: node_id and node_id_2 both required for MERGE_NODES."
        if nid1 == nid2:
            return "ERROR: node_id and node_id_2 must be different."
        registry = ep.current_ast.node_registry
        if nid1 not in registry:
            return f"ERROR: node_id '{nid1[:12]}' not found."
        if nid2 not in registry:
            return f"ERROR: node_id_2 '{nid2[:12]}' not found."

        node1 = registry[nid1]
        node2 = registry[nid2]
        node1.content = f"{node1.content}\n{node2.content}"
        node1.token_count = _count_tokens(node1.content)
        for child_id in node2.children:
            if child_id in registry:
                registry[child_id].parent_id = nid1
        node1.children.extend(node2.children)

        if node2.parent_id and node2.parent_id in registry:
            p = registry[node2.parent_id]
            if nid2 in p.children:
                p.children.remove(nid2)
        del registry[nid2]
        _update_token_counts(ep.current_ast.root_id, registry)
        return f"MERGE_NODES: '{nid2[:8]}' merged into '{nid1[:8]}'. Combined tokens: {node1.token_count}"

    def _handle_probe(
        self, action: PromptForgeAction, ep: "_EpisodeData"
    ) -> tuple[float, str]:
        if ep.probe_budget <= 0:
            return 0.0, "ERROR: PROBE budget exhausted."
        node_id = action.node_id
        if not node_id:
            return 0.0, "ERROR: node_id required for PROBE."
        registry = ep.current_ast.node_registry
        if node_id not in registry:
            return 0.0, f"ERROR: node_id '{node_id[:12]}' not found."

        ep.probe_budget -= 1
        ep.probe_steps_used += 1

        ast_snapshot = copy.deepcopy(ep.current_ast)
        prompt_with = serialize_ast(ep.current_ast)
        ppl_with = compute_baseline_perplexity(prompt_with)

        node = ep.current_ast.node_registry[node_id]
        parent_id = node.parent_id
        if parent_id and parent_id in ep.current_ast.node_registry:
            parent = ep.current_ast.node_registry[parent_id]
            if node_id in parent.children:
                parent.children.remove(node_id)

        prompt_without = serialize_ast(ep.current_ast)
        ppl_without = compute_baseline_perplexity(prompt_without)

        # RESTORE — non-destructive guarantee
        ep.current_ast = ast_snapshot

        delta = ppl_without - ppl_with
        significant = abs(delta) > (ppl_with * 0.10) if ppl_with > 0 else False
        direction = "INCREASES" if delta > 0 else "DECREASES"
        verdict = (
            "Safe to prune." if not significant
            else "CAUTION: high coherence contribution."
        )
        msg = (
            f"PROBE '{node_id[:8]}': removing it {direction} perplexity "
            f"by {abs(delta):.1f} ({'significant' if significant else 'minor'}). "
            f"{verdict} Budget left: {ep.probe_budget}."
        )
        return 0.0, msg

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _check_perplexity(self, ep: "_EpisodeData") -> float:
        if ep.baseline_perplexity <= 0:
            return 0.0
        return compute_perplexity_penalty(
            serialize_ast(ep.current_ast), ep.baseline_perplexity
        )

    def _build_obs(
        self,
        ep: "_EpisodeData",
        last_action_result: str,
        reward: float,
        done: bool,
    ) -> PromptForgeObservation:
        registry = ep.current_ast.node_registry
        root = registry[ep.current_ast.root_id]
        curr = root.token_count
        orig = ep.original_token_count
        reduction = round((orig - curr) / orig * 100, 2) if orig > 0 else 0.0

        nodes = []
        for sid in root.children:
            if sid not in registry:
                continue
            sec = registry[sid]
            nodes.append({
                "node_id": sec.node_id,
                "node_type": sec.node_type.value,
                "token_count": sec.token_count,
                "content_preview": sec.content[:80],
            })
            for rid in sec.children:
                if rid not in registry:
                    continue
                rule = registry[rid]
                nodes.append({
                    "node_id": rule.node_id,
                    "node_type": rule.node_type.value,
                    "token_count": rule.token_count,
                    "content_preview": rule.content[:80],
                })

        return PromptForgeObservation(
            raw_prompt=serialize_ast(ep.current_ast),
            ast_summary=nodes,
            current_token_count=curr,
            original_token_count=orig,
            token_reduction_pct=reduction,
            step_count=ep.step_count,
            max_steps=MAX_STEPS,
            probe_budget_remaining=ep.probe_budget,
            task_difficulty=ep.task.difficulty,
            last_action_result=last_action_result,
            done=done,
            reward=reward,
        )

    def _error_obs(self, msg: str) -> PromptForgeObservation:
        return PromptForgeObservation(
            last_action_result=msg,
            done=False,
            reward=0.0,
        )


# ── Episode state container ────────────────────────────────────────────────────

class _EpisodeData:
    """Mutable state for a single PromptForge episode."""

    def __init__(self, *, task, current_ast: PromptAST, baseline_perplexity: float, original_token_count: int):
        self.task = task
        self.current_ast = current_ast
        self.baseline_perplexity = baseline_perplexity
        self.original_token_count = original_token_count
        self.step_count: int = 0
        self.probe_budget: int = PROBE_BUDGET
        self.probe_steps_used: int = 0
        self.submitted: bool = False
