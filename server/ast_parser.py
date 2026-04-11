"""Custom prompt AST parser for PromptForge.

Converts flat prompt text into a structured tree:
    DOCUMENT → SECTION → RULE | EXAMPLE | CONSTRAINT

No external AST library required — fully self-contained recursive implementation.

Node types:
    DOCUMENT   — Root of the tree; spans the entire prompt.
    SECTION    — A heading-delimited block (## Heading, **Bold**, ALL CAPS, etc.)
    RULE       — An individual instruction or guideline within a section.
    EXAMPLE    — A few-shot example (detected by "Customer:", "Input:", etc.)
    CONSTRAINT — A hard constraint (starts with Never, Always, Must, etc.)

Ref: Arbiter (arXiv 2026) — AST-based interference detection in system prompts.
Ref: LLMLingua-2 (Pan et al. 2024) — token-level prompt compression as ML task.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Node types ────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    DOCUMENT   = "DOCUMENT"
    SECTION    = "SECTION"
    RULE       = "RULE"
    EXAMPLE    = "EXAMPLE"
    CONSTRAINT = "CONSTRAINT"


@dataclass
class Node:
    node_id: str
    node_type: NodeType
    content: str
    children: list[str] = field(default_factory=list)  # list of child node_ids
    token_count: int = 0
    parent_id: Optional[str] = None


@dataclass
class PromptAST:
    root_id: str
    node_registry: dict[str, Node]  # O(1) lookup by node_id


# ── Token counting ─────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Approximate token count (whitespace split; 1 word ≈ 1 token for TF models)."""
    return max(1, len(text.split()))


# ── Node factory ───────────────────────────────────────────────────────────────

def _make_node(
    node_type: NodeType,
    content: str,
    parent_id: Optional[str] = None,
) -> Node:
    node_id = str(uuid.uuid4())
    return Node(
        node_id=node_id,
        node_type=node_type,
        content=content.strip(),
        token_count=_count_tokens(content),
        parent_id=parent_id,
    )


# ── Content classification ─────────────────────────────────────────────────────

_EXAMPLE_SIGNALS = re.compile(
    r"(?:customer:|input:|output:|response:|example\s*\d|few[\-\s]?shot|"
    r"sample:|illustration:|query:|user:)",
    re.IGNORECASE,
)
_CONSTRAINT_SIGNAL = re.compile(
    r"^(?:never|always|must|do not|do\s+not|prohibited|forbidden|mandatory|"
    r"required|you\s+must|you\s+should\s+never)",
    re.IGNORECASE,
)
_DEPRECATED_SIGNAL = re.compile(
    r"(?:deprecated|do not use|removed in|no longer|legacy|obsolete)",
    re.IGNORECASE,
)


def _classify_content(text: str) -> NodeType:
    """Infer the most specific NodeType for a content block."""
    stripped = text.strip()
    if _EXAMPLE_SIGNALS.search(stripped):
        return NodeType.EXAMPLE
    if _CONSTRAINT_SIGNAL.match(stripped):
        return NodeType.CONSTRAINT
    return NodeType.RULE


# ── Section splitting ──────────────────────────────────────────────────────────

def _is_heading_line(line: str) -> tuple[bool, str]:
    """Return (is_heading, heading_text) for a single line."""
    stripped = line.strip()

    # Markdown heading: ## Heading
    if stripped.startswith("#") and len(stripped) > 2 and stripped[1:3].strip():
        return True, stripped.lstrip("#").strip()

    # Bold heading: **Heading**
    bold_match = re.match(r"^\*\*([^*]+)\*\*\s*$", stripped)
    if bold_match:
        return True, bold_match.group(1).strip()

    # ALL CAPS line (3+ words, no punctuation heavy)
    if (
        stripped.isupper()
        and len(stripped) > 4
        and " " in stripped
        and not stripped.endswith(":")
        and len(stripped) < 80
    ):
        return True, stripped

    # Numbered section prefix: "Section 3:", "1. Introduction"
    section_match = re.match(r"^(?:Section|SECTION|Chapter)\s+\d+", stripped)
    if section_match:
        return True, stripped

    return False, stripped


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split prompt text into (heading, body) pairs."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = "__preamble__"
    current_body: list[str] = []

    for line in lines:
        is_heading, heading_text = _is_heading_line(line)

        if is_heading and current_body:
            body = "\n".join(current_body).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = heading_text
            current_body = []
        elif is_heading:
            current_heading = heading_text
        else:
            current_body.append(line)

    if current_body:
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_heading, body))

    return [(h, b) for h, b in sections if b.strip()]


# ── Rule splitting within a section ───────────────────────────────────────────

def _split_into_rules(body: str) -> list[str]:
    """Split a section body into individual rule/example/constraint texts."""
    lines = body.split("\n")

    # Detect list-item format
    _list_re = re.compile(r"^[\-\*\•]\s+|^\d+[\.\)]\s+")
    has_list = any(_list_re.match(l.strip()) for l in lines if l.strip())

    if has_list:
        items: list[str] = []
        buf: list[str] = []
        for line in lines:
            stripped = line.strip()
            if _list_re.match(stripped):
                if buf:
                    items.append("\n".join(buf).strip())
                buf = [stripped]
            elif stripped and buf:
                buf.append(stripped)
            elif stripped:
                items.append(stripped)
        if buf:
            items.append("\n".join(buf).strip())
        return [i for i in items if i]

    # Sub-heading (### Example, Rule N:) splits
    sub_heading_re = re.compile(
        r"^(?:###|Rule\s+\d+:|Example\s+\d|#{2,3}\s+)",
        re.IGNORECASE,
    )
    blocks: list[str] = []
    buf2: list[str] = []
    for line in lines:
        stripped = line.strip()
        if sub_heading_re.match(stripped) and buf2:
            blocks.append("\n".join(buf2).strip())
            buf2 = [stripped]
        elif stripped:
            buf2.append(stripped)
        elif buf2:
            # Blank line = block boundary
            blocks.append("\n".join(buf2).strip())
            buf2 = []
    if buf2:
        blocks.append("\n".join(buf2).strip())

    return [b for b in blocks if b]


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_prompt(text: str) -> PromptAST:
    """Parse a flat prompt text into a PromptAST.

    Algorithm:
    1. Detect heading-delimited sections (##, **, ALL CAPS, Section N:).
    2. Within each section, split into individual Rule/Example/Constraint nodes
       by list-item boundaries or blank-line separators.
    3. Assign stable UUIDs to every node.
    4. Compute token counts bottom-up.

    Returns:
        PromptAST with a root DOCUMENT node and a flat node_registry for O(1)
        lookup by node_id.
    """
    registry: dict[str, Node] = {}

    # Root DOCUMENT node
    root = _make_node(NodeType.DOCUMENT, text)
    registry[root.node_id] = root

    sections = _split_into_sections(text)
    if not sections:
        sections = [("Main", text)]

    for sec_heading, sec_body in sections:
        sec_node = _make_node(NodeType.SECTION, sec_heading, parent_id=root.node_id)
        registry[sec_node.node_id] = sec_node
        root.children.append(sec_node.node_id)

        for rule_text in _split_into_rules(sec_body):
            ntype = _classify_content(rule_text)
            rule_node = _make_node(ntype, rule_text, parent_id=sec_node.node_id)
            registry[rule_node.node_id] = rule_node
            sec_node.children.append(rule_node.node_id)

    _update_token_counts(root.node_id, registry)
    return PromptAST(root_id=root.node_id, node_registry=registry)


def serialize_ast(ast: PromptAST) -> str:
    """Reconstruct flat prompt text from the current AST via depth-first traversal.

    Maintains section order and formatting so the serialised prompt is a valid
    system-prompt string that can be sent directly to an LLM.
    """
    registry = ast.node_registry
    root = registry[ast.root_id]
    parts: list[str] = []

    for section_id in root.children:
        if section_id not in registry:
            continue
        section = registry[section_id]
        heading = section.content.strip()

        # Format heading
        if heading.startswith("#") or heading.startswith("**"):
            parts.append(f"\n{heading}\n")
        else:
            parts.append(f"\n## {heading}\n")

        for rule_id in section.children:
            if rule_id not in registry:
                continue
            rule = registry[rule_id]
            content = rule.content.strip()
            parts.append(content)
            parts.append("")  # blank line between rules

    return "\n".join(parts).strip()


def ast_to_observation_dict(ast: PromptAST) -> dict:  # type: ignore[type-arg]
    """Return a JSON-serialisable summary of the AST for the agent's observation.

    Includes: total token count, section count, rule count, per-node summaries.
    """
    registry = ast.node_registry
    root = registry[ast.root_id]

    nodes: list[dict] = []  # type: ignore[type-arg]
    for section_id in root.children:
        if section_id not in registry:
            continue
        section = registry[section_id]
        nodes.append(
            {
                "node_id": section.node_id,
                "node_type": section.node_type.value,
                "token_count": section.token_count,
                "content_preview": section.content[:80],
            }
        )
        for rule_id in section.children:
            if rule_id not in registry:
                continue
            rule = registry[rule_id]
            nodes.append(
                {
                    "node_id": rule.node_id,
                    "node_type": rule.node_type.value,
                    "token_count": rule.token_count,
                    "content_preview": rule.content[:80],
                }
            )

    rule_count = sum(len(registry[sid].children) for sid in root.children if sid in registry)

    return {
        "total_token_count": root.token_count,
        "section_count": len(root.children),
        "rule_count": rule_count,
        "nodes": nodes,
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _update_token_counts(node_id: str, registry: dict[str, Node]) -> int:
    """Recursively compute and store token counts (sum of descendants)."""
    node = registry.get(node_id)
    if node is None:
        return 0
    if not node.children:
        node.token_count = _count_tokens(node.content)
        return node.token_count
    child_total = sum(
        _update_token_counts(cid, registry)
        for cid in node.children
        if cid in registry
    )
    node.token_count = _count_tokens(node.content) + child_total
    return node.token_count


def get_subtree_node_ids(node_id: str, registry: dict[str, Node]) -> list[str]:
    """Return all node_ids in the subtree rooted at node_id (inclusive)."""
    result = [node_id]
    node = registry.get(node_id)
    if node:
        for child_id in node.children:
            result.extend(get_subtree_node_ids(child_id, registry))
    return result
