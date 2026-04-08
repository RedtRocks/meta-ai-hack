"""Pydantic v2 models for the GitHub Issue Triage environment."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class Issue(BaseModel):
    """A single GitHub issue."""
    id: str                          # e.g. "issue-042"
    title: str
    body: str
    author: str
    created_at: str                  # ISO-8601 string
    tags: List[str]                  # author-applied tags (may be empty)


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    repo_name: str                   # e.g. "acme/payments-sdk"
    repo_description: str            # 1-2 sentence repo context
    label_schema: List[str]          # valid labels for this repo
    current_issue: Issue             # the issue the agent must triage NOW
    existing_issues: List[Issue]     # closed/open issue pool for duplicate lookup
    step_number: int
    max_steps: int
    issues_remaining: int


class Action(BaseModel):
    """The triage decision the agent must produce."""
    labels: List[str]                # must be subset of label_schema
    priority: Literal["P0", "P1", "P2", "P3"]
    is_duplicate: bool
    duplicate_of: Optional[str] = None    # issue id if is_duplicate is True
    needs_info: bool
    comment: Optional[str] = None         # required if needs_info is True
    is_security: bool
    close: bool                           # close as invalid / noise / off-topic


class Reward(BaseModel):
    """Decomposed reward signal returned after each step."""
    total: float                     # clamped to [-0.40, 1.0]
    label_score: float               # 0.0 – 0.30
    duplicate_score: float           # 0.0 – 0.25
    priority_score: float            # 0.0 – 0.20
    comment_score: float             # 0.0 – 0.15
    security_score: float            # 0.10 or -0.40 penalty
    breakdown: dict                  # human-readable explanation
