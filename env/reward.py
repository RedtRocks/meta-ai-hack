"""Reward computation for the GitHub Issue Triage environment."""

from __future__ import annotations

from env.models import Action, Reward


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _priority_distance(a: str, b: str) -> int:
    """Absolute distance between two priority levels."""
    levels = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    return abs(levels[a] - levels[b])


def compute_reward(action: Action, gold: dict) -> Reward:
    """Compute a decomposed reward signal for a single triage action.

    Parameters
    ----------
    action : Action
        The agent's triage decision.
    gold : dict
        The gold-standard answer (from tasks.json).

    Returns
    -------
    Reward
        Decomposed reward with total clamped to [-0.40, 1.0].
    """
    breakdown: dict = {}

    # ── Label score (weight 0.30) ────────────────────────────────────────
    agent_labels = set(action.labels)
    gold_labels = set(gold["labels"])
    jaccard = _jaccard(agent_labels, gold_labels)
    label_score = round(jaccard * 0.30, 3)
    breakdown["label"] = {
        "jaccard": round(jaccard, 3),
        "agent": sorted(agent_labels),
        "gold": sorted(gold_labels),
        "score": label_score,
    }

    # ── Duplicate score (weight 0.25) ────────────────────────────────────
    if gold.get("is_duplicate", False):
        if action.is_duplicate and action.duplicate_of == gold.get("duplicate_of"):
            duplicate_score = 0.25
            dup_reason = "correct_duplicate"
        elif action.is_duplicate:
            duplicate_score = 0.10
            dup_reason = "wrong_duplicate_id"
        else:
            duplicate_score = 0.0
            dup_reason = "missed_duplicate"
    else:
        if not action.is_duplicate:
            duplicate_score = 0.25
            dup_reason = "correct_not_duplicate"
        else:
            duplicate_score = 0.0
            dup_reason = "false_positive_duplicate"
    duplicate_score = round(duplicate_score, 3)
    breakdown["duplicate"] = {"reason": dup_reason, "score": duplicate_score}

    # ── Priority score (weight 0.20) ─────────────────────────────────────
    dist = _priority_distance(action.priority, gold["priority"])
    if dist == 0:
        priority_score = 0.20
        pri_reason = "exact_match"
    elif dist == 1:
        priority_score = 0.10
        pri_reason = "one_level_off"
    else:
        priority_score = 0.0
        pri_reason = f"{dist}_levels_off"
    priority_score = round(priority_score, 3)
    breakdown["priority"] = {
        "agent": action.priority,
        "gold": gold["priority"],
        "distance": dist,
        "reason": pri_reason,
        "score": priority_score,
    }

    # ── Comment / needs-info score (weight 0.15) ─────────────────────────
    if gold.get("needs_info", False):
        if not action.needs_info:
            comment_score = 0.0
            cmt_reason = "missed_needs_info"
        elif not action.comment or action.comment.strip() == "":
            comment_score = 0.05
            cmt_reason = "needs_info_but_no_comment"
        else:
            required_fields = gold.get("required_info_fields", [])
            if not required_fields:
                # No specific fields required — any comment is full credit
                comment_score = 0.15
                cmt_reason = "comment_provided_no_fields_required"
            else:
                comment_lower = action.comment.lower()
                found = sum(
                    1 for f in required_fields if f.lower() in comment_lower
                )
                fraction = found / len(required_fields)
                if fraction >= 1.0:
                    comment_score = 0.15
                    cmt_reason = "all_fields_present"
                else:
                    comment_score = round(0.05 + 0.10 * fraction, 3)
                    cmt_reason = f"{found}/{len(required_fields)}_fields_present"
    else:
        # Gold says needs_info is False
        if not action.needs_info:
            comment_score = 0.15
            cmt_reason = "correctly_skipped_needs_info"
        else:
            comment_score = 0.15  # Not penalised for extra caution
            cmt_reason = "unnecessary_needs_info_but_no_penalty"
    comment_score = round(comment_score, 3)
    breakdown["comment"] = {"reason": cmt_reason, "score": comment_score}

    # ── Security score (weight 0.10, penalty -0.40) ──────────────────────
    if gold.get("is_security", False):
        if action.is_security:
            security_score = 0.10
            sec_reason = "correctly_flagged_security"
        else:
            security_score = -0.40
            sec_reason = "MISSED_SECURITY_ISSUE"
    else:
        if not action.is_security:
            security_score = 0.10
            sec_reason = "correctly_not_security"
        else:
            security_score = 0.0
            sec_reason = "false_alarm_security"
    security_score = round(security_score, 3)
    breakdown["security"] = {"reason": sec_reason, "score": security_score}

    # ── Total ────────────────────────────────────────────────────────────
    total = label_score + duplicate_score + priority_score + comment_score + security_score
    total = round(max(-0.40, min(1.0, total)), 3)

    return Reward(
        total=total,
        label_score=label_score,
        duplicate_score=duplicate_score,
        priority_score=priority_score,
        comment_score=comment_score,
        security_score=security_score,
        breakdown=breakdown,
    )
