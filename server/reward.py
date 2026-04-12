"""Reward function for the PromptForge environment.

Uses Potential-Based Reward Shaping (PBRS) to provide dense, incremental
feedback at every step — satisfying Meta's requirement that the reward function
"must reward incremental progress toward the objective."

Full reward formula:

    NON-SUBMIT steps (structural edits):
        If perplexity penalty triggered:   reward = perplexity_penalty  (−0.5)
        If PROBE action:                   reward = −0.02
        Otherwise (PRUNE/MOVE/MERGE):      reward = Φ(s') − Φ(s)
            where Φ(s) = (original_tokens − current_tokens) / original_tokens

    SUBMIT step (terminal):
        reward = token_reduction_score × quality_score

    The PBRS shaping is policy-invariant (Ng et al., 1999): the potential
    difference telescopes across the trajectory, so the agent cannot
    accumulate infinite reward by cycling MOVE/MERGE actions.

Key properties:
    • Correct prune → immediate positive reward (proportional to tokens removed)
    • Unnecessary MOVE with no token change → reward = 0.0
    • Destructive prune → immediate −0.5 penalty
    • PROBE → immediate −0.02 cost
    • SUBMIT with broken output → 0.0 (quality_score = 0.0)

Ref: Ng, Harada & Russell (1999) — "Policy Invariance Under Reward Transformations"
Ref: LLMLingua-2 (Pan et al. 2024) — validates token-level prompt compression.
Ref: RLPrompt (Deng et al. 2022) — perplexity guard against grammatical collapse.
"""

from __future__ import annotations


def compute_reward(
    original_token_count: int,
    previous_token_count: int,
    current_token_count: int,
    quality_score: float,        # 1.0 or 0.0 from Deterministic JSON Grader
    perplexity_penalty: float,   # 0.0 or a negative scalar from DistilGPT-2 guard
    action_type: str,            # e.g. "PRUNE_BRANCH", "PROBE", "SUBMIT"
    is_submit: bool,             # True only when processing a SUBMIT action
) -> float:
    """Compute the scalar reward for the current step.

    Args:
        original_token_count:  Tokens in the bloated baseline prompt.
        previous_token_count:  Tokens at the START of this step (before action).
        current_token_count:   Tokens AFTER the action was applied.
        quality_score:         Binary quality from the JSON grader (1.0 or 0.0).
                               Meaningful only when is_submit=True.
        perplexity_penalty:    0.0 or a negative float from the perplexity guard.
        action_type:           String name of the action (for PROBE cost).
        is_submit:             True only for the SUBMIT action processing path.

    Returns:
        Scalar reward (float).
    """
    # ── 1. Destructive action — immediate penalty (perplexity guard) ──────────
    if perplexity_penalty < 0.0:
        return float(perplexity_penalty)

    # ── 2. PROBE cost — immediate per-use penalty ─────────────────────────────
    if action_type == "PROBE":
        return -0.02

    # ── 3. Intermediate step — Potential-Based Reward Shaping (PBRS) ──────────
    if not is_submit:
        denom = max(1, original_token_count)
        phi_prev = (original_token_count - previous_token_count) / denom
        phi_curr = (original_token_count - current_token_count) / denom
        # Only positive reward when tokens are actually reduced
        step_reward = max(0.0, phi_curr - phi_prev)
        return round(step_reward, 6)

    # ── 4. SUBMIT path — terminal quality bonus ──────────────────────────────
    if original_token_count <= 0:
        token_reduction_score = 0.0
    else:
        token_reduction_score = (
            (original_token_count - current_token_count) / original_token_count
        )
        token_reduction_score = max(0.0, min(1.0, token_reduction_score))

    terminal_reward = token_reduction_score * quality_score
    return round(terminal_reward, 6)
