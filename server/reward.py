"""Reward function for the PromptForge environment.

Full reward formula (on SUBMIT):
    reward = (token_reduction_score × quality_score)
             − probe_step_penalty
             + perplexity_penalty          # already negative or 0.0

Where:
    token_reduction_score  = (original_tokens − current_tokens) / original_tokens
                             clipped to [0.0, 1.0]
    quality_score          = 1.0 or 0.0 (from Deterministic JSON Grader)
    probe_step_penalty     = probe_steps_used × 0.02
    perplexity_penalty     = 0.0 or PERPLEXITY_PENALTY_SCALAR (−0.5 by default)

During non-SUBMIT steps:
    reward = perplexity_penalty   (coherence guard fires on structural changes)

Key properties this function enforces:
    • Agent that reduces tokens but breaks JSON schema:  0.0 × token_reduction = 0.0
    • Agent that preserves schema but makes zero compression: 0.0 × 1.0 = 0.0
    • Agent that produces incoherent output: hit by perplexity penalty regardless
    • Cosine similarity is completely absent — do NOT import sentence-transformers

Ref: LLMLingua-2 (Pan et al. 2024) — validates token-level prompt compression.
Ref: RLPrompt (Deng et al. 2022) — perplexity guard against grammatical collapse.
"""

from __future__ import annotations


def compute_reward(
    original_token_count: int,
    current_token_count: int,
    quality_score: float,        # 1.0 or 0.0 from Deterministic JSON Grader
    perplexity_penalty: float,   # 0.0 or a negative scalar from DistilGPT-2 guard
    probe_steps_used: int,       # Total PROBE actions consumed this episode
    is_submit: bool,             # True only when processing a SUBMIT action
) -> float:
    """Compute the scalar reward for the current step.

    Args:
        original_token_count:  Tokens in the bloated baseline prompt.
        current_token_count:   Tokens in the current (edited) prompt.
        quality_score:         Binary quality from the JSON grader (1.0 or 0.0).
                               Meaningful only when is_submit=True.
        perplexity_penalty:    0.0 or a negative float from the perplexity guard.
        probe_steps_used:      Total PROBE actions used (for small reward penalty).
        is_submit:             True only for the SUBMIT action processing path.

    Returns:
        Scalar reward (float).
    """
    if not is_submit:
        # Step-level reward: only the perplexity guard applies during exploration.
        # Structural actions (PRUNE, MOVE, MERGE) get no positive reward until SUBMIT.
        return float(perplexity_penalty)

    # ── SUBMIT path ───────────────────────────────────────────────────────────

    if original_token_count <= 0:
        token_reduction_score = 0.0
    else:
        token_reduction_score = (
            (original_token_count - current_token_count) / original_token_count
        )
        # Clip to [0.0, 1.0]: no reward for padding (negative compression)
        token_reduction_score = max(0.0, min(1.0, token_reduction_score))

    probe_step_penalty = probe_steps_used * 0.02   # 0.02 cost per PROBE used

    final_reward = (
        (token_reduction_score * quality_score)
        - probe_step_penalty
        + perplexity_penalty   # already ≤ 0.0
    )

    return round(final_reward, 6)
