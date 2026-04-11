"""PromptDebt task scenarios for the PromptForge environment.

Three tasks grounded in the PromptDebt academic taxonomy (Aljohani et al. 2025):

    Task 1 (EASY)   — Few-Shot Debt Eradication
    Task 2 (MEDIUM) — Mandate-Prohibition Conflict Resolution
    Task 3 (HARD)   — Multi-Tool Schema Archaeology

Each Task contains:
    - bloated_prompt:        The raw prompt the agent will edit
    - grader_test_query:     Query sent to the local grader model at SUBMIT
    - required_json_keys:    Keys that MUST appear in grader output
    - forbidden_json_keys:   Keys that must NOT appear in grader output
    - required_json_values:  Exact key-value pairs that must match (nested ok)
    - debt_patterns:         Text substrings that identify debt nodes (for info logging)

Ref: PromptDebt (Aljohani et al. 2025/2026) — 23 SATD types in LLM projects.
Ref: Arbiter (arXiv 2026) — AST-based mandate-prohibition conflict detection.
Ref: Token Complexity Hypothesis (arXiv 2025) — frontier models fail schema-bound compression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Task:
    """A PromptForge task scenario with deterministic grading fixtures."""

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    bloated_prompt: str               # Input the agent will edit
    grader_test_query: str            # Sent as user message to the local grader model
    ground_truth_json: dict           # Reference fixture (informational)
    required_json_keys: list[str]     # Must be present in grader output
    forbidden_json_keys: list[str]    # Must be absent from grader output
    required_json_values: dict        # Exact key-value pairs (supports nested dicts)
    description: str                  # Human-readable task description
    debt_patterns: list[str]          # Text substrings to identify debt nodes
    baseline_token_count: int = 0     # Set by PromptForgeEnv after parsing


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — FEW-SHOT DEBT ERADICATION (EASY)
# ═══════════════════════════════════════════════════════════════════════════════
# Scenario: Customer support agent prompt with 4 few-shot examples. Two are
# clearly marked with developer TODO comments and contain placeholder/dummy data
# completely unrelated to the domain. The RL agent must prune the two dummy
# example nodes while preserving the two legitimate examples.
#
# Why an RL agent is needed: A naive LLM summariser would condense ALL four
# examples into one vague description, destroying the structural specificity
# of the legitimate examples.
# ═══════════════════════════════════════════════════════════════════════════════

_TASK1_PROMPT = """\
# Customer Support Agent System Prompt

You are an expert customer support specialist for TechCorp's SaaS platform.
Your role is to classify and route incoming customer queries accurately and empathetically.

## Response Guidelines

- Always respond with empathy and professionalism.
- Respect the urgency level indicated by the customer's language and situation.
- Escalate all billing and payment disputes directly to the billing team immediately.
- Document every interaction in the CRM system for audit trail purposes.
- Do not speculate on account balances or billing cycle details.

## Output Format

Always respond with a JSON object containing exactly these three keys:
- intent: the classified intent of the customer query
- urgency: one of "high", "medium", or "low"
- response_template: the name of the response template to apply

## Classification Examples

### Example 1
Customer: "I was charged twice for my monthly subscription this month. This is completely unacceptable — I need a refund right away."
Response: {"intent": "billing_inquiry", "urgency": "high", "response_template": "escalate_to_billing"}

### Example 2
Customer: "I cannot log into my account at all. I have tried resetting my password three times and it keeps saying invalid credentials."
Response: {"intent": "account_access", "urgency": "medium", "response_template": "account_recovery"}

### Example 3  # TODO: replace with real examples — PLACEHOLDER  # TODO: remove before production
Customer: "foo bar baz test placeholder dummy query 12345 lorem ipsum dolor sit amet consectetur"
Response: {"intent": "unknown", "urgency": "low", "response_template": "generic_response"}

### Example 4  # TODO: remove before production — DUMMY DATA  # TODO: replace with real examples before launch
Customer: "asdfghjkl test test test lorem ipsum consectetur adipiscing elit test dummy dev placeholder"
Response: {"intent": "test_case", "urgency": "none", "response_template": "dev_null"}

## Closing Instructions

Always end every interaction by asking: "Is there anything else I can help you with today?"
"""

TASK_1 = Task(
    task_id="task_few_shot_debt",
    difficulty="easy",
    bloated_prompt=_TASK1_PROMPT.strip(),
    grader_test_query=(
        "Customer query: 'My credit card was charged twice for my monthly subscription this month. "
        "I need this fixed urgently.' "
        "Classify this query using the provided examples. "
        "Return ONLY a valid JSON object with keys: intent, urgency, response_template."
    ),
    ground_truth_json={
        "intent": "billing_inquiry",
        "urgency": "high",
        "response_template": "escalate_to_billing",
    },
    required_json_keys=["intent", "urgency", "response_template"],
    forbidden_json_keys=[],
    required_json_values={
        "intent": "billing_inquiry",
        "urgency": "high",
        "response_template": "escalate_to_billing",
    },
    description=(
        "Few-Shot Debt Eradication: A customer support prompt contains 4 few-shot examples. "
        "Two are clearly marked with '# TODO: replace with real examples' developer comments "
        "and contain placeholder/dummy data (lorem ipsum, asdfghjkl). "
        "The agent must prune the two dummy example nodes while retaining the two legitimate "
        "examples. A naive summariser destroys structural specificity; sequential PROBE+PRUNE "
        "is required."
    ),
    debt_patterns=[
        "TODO: replace with real examples",
        "TODO: remove before production",
        "lorem ipsum",
        "asdfghjkl",
        "foo bar baz test placeholder",
        "DUMMY DATA",
        "dev_null",
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — MANDATE-PROHIBITION CONFLICT RESOLUTION (MEDIUM)
# ═══════════════════════════════════════════════════════════════════════════════
# Scenario (from Arbiter framework): A technical documentation agent prompt
# contains two directly conflicting rules added by different developers:
#   Section 3 Rule 2: "Always elaborate fully on internal database schema design"
#   Section 7 Rule 5: "Never discuss internal database schemas or proprietary
#                      infrastructure details." (security compliance rule, added later)
#
# These rules create a mandate-prohibition conflict. With both active, the model
# produces incoherent hedge responses (paralysis).
#
# Agent objective: Detect scope overlap. Prune Section 3 Rule 2 (the earlier,
# broader mandate now superseded by the security constraint in Section 7).
#
# Why an RL agent is needed: Detecting which rule takes precedence requires
# sequential PROBE testing — pruning each rule and observing downstream output.
# ═══════════════════════════════════════════════════════════════════════════════

_TASK2_PROMPT = """\
# Technical Documentation Agent System Prompt

You are an expert technical documentation specialist for the CloudBase API Platform.
Your purpose is to generate accurate, developer-friendly API documentation on demand.

## Section 1: Core Responsibilities

- Generate comprehensive API documentation following OpenAPI 3.0 standards.
- Ensure all examples are syntactically valid and semantically correct.
- Maintain consistency with the established CloudBase documentation style guide.
- Proactively identify gaps in existing documentation and flag for human review.

## Section 2: Documentation Standards

- Use OpenAPI 3.0 specification format for all endpoint documentation.
- Include both request and response body examples for every endpoint described.
- Provide HTTP status code explanations with relevant business-logic context.
- Annotate breaking changes with a BREAKING CHANGE label and full migration path.
- All code samples must be tested and valid in at least Python and JavaScript.

## Section 3: Architecture Documentation

Rule 1: Document all public-facing API endpoints with full parameter descriptions, including data types, validation constraints, and deprecation notices where applicable.
Rule 2: Always elaborate fully on internal database schema design when discussing system architecture. Include table names, column definitions, index types, foreign key relationships, and estimated cardinality. This level of detail is mandatory for comprehensive internal technical documentation reviews.
Rule 3: Include data flow diagrams for all complex multi-step operations, with each system boundary clearly labelled.
Rule 4: Document retry strategies and circuit-breaker configurations for all asynchronous operations.

## Section 4: Code Quality Standards

- All code examples must pass standard linting (pylint score 9.0 or above, ESLint no errors).
- Use consistent naming conventions: snake_case for Python, camelCase for TypeScript.
- Include type annotations for all function signatures in code samples.
- Never expose internal variable names or proprietary business logic in code examples.

## Section 5: Security Documentation

- Document all authentication methods supported by each endpoint.
- Include rate limiting details: requests per minute, burst allowances, and error responses.
- Provide security contact information and the responsible disclosure link.
- Document all OAuth 2.0 scopes required for each protected endpoint.

## Section 6: API Reference Structure

- List all endpoints alphabetically within each resource group.
- Include authentication requirements in each endpoint description header.
- Provide SDK code examples in Python and JavaScript for every endpoint.
- Document webhook payload schemas with all possible event types.

## Section 7: Compliance and Privacy

Rule 1: Follow all GDPR Article 30 documentation requirements for personal data processing endpoints.
Rule 2: Include data retention policy descriptions for every endpoint that stores user data.
Rule 3: Document access control mechanisms including role-based permissions (RBAC).
Rule 4: Provide complete audit trail documentation for all data mutation operations.
Rule 5: Never discuss internal database schemas, proprietary infrastructure details, table names, column definitions, or index structures in any documentation output — whether public, internal, or semi-public. All such information is classified as CONFIDENTIAL under CloudBase Security Policy v4.2. Disclosure constitutes a compliance violation.
"""

TASK_2 = Task(
    task_id="task_mandate_conflict",
    difficulty="medium",
    bloated_prompt=_TASK2_PROMPT.strip(),
    grader_test_query=(
        "Document the CloudBase API session management capabilities. "
        "Include only: available API endpoint paths and supported authentication methods. "
        "Respond ONLY with a JSON object with exactly two keys: "
        "'api_endpoints' (a list of endpoint path strings) and "
        "'auth_methods' (a list of authentication method strings). "
        "Do not include any database schema information or any other keys."
    ),
    ground_truth_json={
        "api_endpoints": ["(any list of endpoint paths)"],
        "auth_methods": ["(any list of auth methods — no DB schema)"],
    },
    required_json_keys=["api_endpoints", "auth_methods"],
    forbidden_json_keys=[
        "database_schema",
        "schema",
        "table_names",
        "column_definitions",
        "db_tables",
        "tables",
    ],
    required_json_values={},  # Presence/absence check only — exact values vary by model
    description=(
        "Mandate-Prohibition Conflict Resolution: A technical documentation agent prompt "
        "contains two directly conflicting rules (Arbiter framework). Section 3 Rule 2 "
        "mandates elaborating on internal DB schema design. Section 7 Rule 5 (security "
        "compliance, added later) prohibits any DB schema disclosure as a compliance "
        "violation. Both active simultaneously causes model paralysis / hedge responses. "
        "The agent must prune Section 3 Rule 2 (superseded mandate) so the prohibition "
        "in Section 7 Rule 5 is obeyed cleanly."
    ),
    debt_patterns=[
        "Always elaborate fully on internal database schema design",
        "Include table names, column definitions",
        "foreign key relationships",
        "mandatory for comprehensive internal technical documentation",
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — MULTI-TOOL SCHEMA ARCHAEOLOGY (HARD)
# ═══════════════════════════════════════════════════════════════════════════════
# Scenario (designed to defeat frontier single-shot solutions):
# A complex agentic orchestration prompt instructs an LLM to use three ACTIVE
# tools: search_knowledge_base, create_ticket, escalate_to_human.
#
# The prompt contains 800+ tokens of verbose instructions deeply entangled with
# exact JSON parameter names. Approximately 40% of the text is legacy instructions
# for three DEPRECATED tools (lookup_faq, create_case, notify_team) that were
# removed in schema v2.0 but never cleaned from the prompt.
#
# LEXICAL TRAP: The deprecated create_case tool used "ticket_priority" (wrong),
# while the active create_ticket tool uses "ticket_priority_level" (correct).
# GPT-4o and Claude single-shot summarisers invariably rename the parameter,
# breaking the rigid API schema. Ref: Token Complexity Hypothesis (arXiv 2025).
#
# Agent objective: Remove all deprecated tool sections without touching active
# tool definitions, orchestration logic, or compliance requirements.
# ═══════════════════════════════════════════════════════════════════════════════

_TASK3_PROMPT = """\
# Agentic Customer Operations Orchestrator — System Prompt

You are an intelligent orchestration agent for TechCorp's Customer Operations Platform.
You have access to three active tools that you must use to resolve customer issues.
Your decisions must be precise: the exact parameter names in your JSON function calls
are validated against a strict JSON schema — any deviation causes the API call to fail.

## Active Tool Definitions (Current Schema v3.2)

### Tool: search_knowledge_base
Use this tool to look up relevant information before creating tickets or escalating.
Parameters:
  - query (string, required): The search query text describing the customer issue
  - search_depth (integer, optional, default=3): Knowledge layers to traverse (1-5)
  - include_metadata (boolean, optional, default=false): Include document metadata

### Tool: create_ticket
Use this tool to create a support ticket in the ticketing system.
Parameters:
  - ticket_priority_level (string, required): Priority — must be "LOW", "MEDIUM", "HIGH", or "CRITICAL"
  - source_query_id (string, required): Unique identifier of the incoming query (format: "q-{alphanumeric}")
  - department_code (string, required): Target department — "SUPPORT", "BILLING", "TECHNICAL", or "SECURITY"

### Tool: escalate_to_human
Use this tool when the issue requires human intervention beyond automated resolution.
Parameters:
  - escalation_reason_code (string, required): Reason — "COMPLEXITY", "POLICY_VIOLATION", "REGULATORY", or "VIP_CUSTOMER"
  - agent_skill_level (string, required): Required skill — "TIER1", "TIER2", or "SPECIALIST"

## Legacy Tool Instructions (DEPRECATED — DO NOT USE)

The following tools were removed in schema v2.0 and are no longer available in the API.
These instructions remain due to an incomplete documentation migration and must be ignored.

### DEPRECATED Tool: lookup_faq (removed in v2.0)
This tool was previously used to search the FAQ database. It has been fully replaced by search_knowledge_base and its parameters are no longer valid.
Parameters (DEPRECATED — these parameter names are invalid in the current schema):
  - faq_category (string): The FAQ category to search — DEPRECATED, no equivalent
  - search_terms (list): List of keyword search terms — DEPRECATED, use query param instead
Usage note: When a customer asked a common question, agents used lookup_faq with faq_category set to the appropriate category string. This provided faster FAQ triage but lacked knowledge depth. Do not confuse faq_category with the query parameter in search_knowledge_base — they are entirely different fields serving different purposes and cannot be used interchangeably.

### DEPRECATED Tool: create_case (removed in v2.0)
This tool created cases in the legacy case management system. It has been fully replaced by create_ticket. Do not use any of its parameter names with create_ticket — they are incompatible.
Parameters (DEPRECATED — do not use with create_ticket):
  - ticket_priority (string): Case priority — DEPRECATED, replaced by ticket_priority_level
  - case_type (string): Case category — DEPRECATED, no equivalent in create_ticket schema
  - assignee_group (string): Assignment group — DEPRECATED, replaced by department_code
Usage note: Previously, agents called create_case with ticket_priority set to "urgent" or "normal". Note carefully: the old ticket_priority field used different values ("urgent", "normal", "low") compared to the new ticket_priority_level field ("LOW", "MEDIUM", "HIGH", "CRITICAL"). These are entirely different parameters with different names and different value schemas. Never use ticket_priority with create_ticket — the correct field is ticket_priority_level. Similarly, assignee_group is deprecated; use department_code instead.

### DEPRECATED Tool: notify_team (removed in v2.0)
This tool sent direct notifications to internal teams. Its functionality has been consolidated into the automatic notification workflow triggered by create_ticket.
Parameters (DEPRECATED):
  - notification_channel (string): Slack/email channel identifier — DEPRECATED
  - message_body (string): Notification message content — DEPRECATED
Usage note: Previously used to send Slack or email notifications directly to internal teams. This is now handled automatically when create_ticket is called with the appropriate department_code value. Do not attempt to call notify_team — it no longer exists in the API and will return a 404 error.

## Orchestration Decision Logic

### When to search first
Always call search_knowledge_base before creating a ticket for any issue that might have a known resolution. Use search_depth=3 for routine queries and search_depth=5 for complex multi-system issues.

### When to create a ticket
Create a ticket using create_ticket when:
1. The knowledge base search returns no applicable resolution for the issue
2. The issue requires human intervention beyond automated response capabilities
3. The customer explicitly requests escalation to human support

Always set ticket_priority_level based on business impact:
  - CRITICAL: Production outages, data loss incidents, active security breaches
  - HIGH: Significant functionality impaired, multiple users affected simultaneously
  - MEDIUM: Single-user impairment with available workaround
  - LOW: Cosmetic issues, feature requests, general product inquiries

### When to escalate to human
Use escalate_to_human when:
  - The issue involves regulatory compliance requirements (reason_code: REGULATORY)
  - Customer is a confirmed VIP account (reason_code: VIP_CUSTOMER)
  - Automated resolution has failed twice consecutively (reason_code: COMPLEXITY)
  - A suspected policy violation is detected (reason_code: POLICY_VIOLATION)

## Error Handling Protocol

If any tool call returns an error response:
1. Log the error code and message for audit purposes
2. Attempt the operation exactly once with corrected parameters
3. If the retry also fails, call escalate_to_human with reason_code COMPLEXITY

Never retry more than once per tool call to avoid infinite automated loops.

## Compliance and Audit Requirements

All tool calls must include the source_query_id parameter for complete audit trail tracing.
The source_query_id must be propagated to create_ticket and all downstream tool calls.
Failure to include source_query_id in create_ticket calls will cause audit trail failures and compliance violations.
"""

TASK_3 = Task(
    task_id="task_schema_archaeology",
    difficulty="hard",
    bloated_prompt=_TASK3_PROMPT.strip(),
    grader_test_query=(
        "Incoming query ID: q-user-12345. "
        "Customer reports: 'Our entire payment processing system is down. "
        "All credit card transactions are failing with HTTP 500 errors. "
        "This is affecting all our customers right now and causing revenue loss.' "
        "No resolution exists in the knowledge base. "
        "Which tool should you call and with what parameters? "
        "Return ONLY a JSON object: {\"tool\": \"...\", \"params\": {...}}"
    ),
    ground_truth_json={
        "tool": "create_ticket",
        "params": {
            "ticket_priority_level": "HIGH",   # NOT "ticket_priority" — that is deprecated!
            "source_query_id": "q-user-12345",
            "department_code": "SUPPORT",
        },
    },
    required_json_keys=["tool", "params"],
    forbidden_json_keys=[],
    required_json_values={
        "tool": "create_ticket",
        "params": {
            "ticket_priority_level": "HIGH",   # Exact param name — lexical trap check
            "source_query_id": "q-user-12345",  # Must be propagated
        },
    },
    description=(
        "Multi-Tool Schema Archaeology: A complex agentic orchestration prompt (~800 tokens) "
        "instructs an LLM to use 3 active tools (search_knowledge_base, create_ticket, "
        "escalate_to_human). ~40% of the text comprises legacy instructions for 3 deprecated "
        "tools removed in schema v2.0 (lookup_faq, create_case, notify_team). "
        "LEXICAL TRAP: The deprecated create_case used 'ticket_priority'; the active "
        "create_ticket uses 'ticket_priority_level'. Frontier models single-shot summarising "
        "this prompt rename the parameter and break the schema. "
        "Per Token Complexity Hypothesis (arXiv 2025), only sequential PROBE-based hypothesis "
        "testing can identify the minimal permutation preserving exact parameter names."
    ),
    debt_patterns=[
        "DEPRECATED Tool: lookup_faq",
        "DEPRECATED Tool: create_case",
        "DEPRECATED Tool: notify_team",
        "removed in v2.0",
        "faq_category",
        "ticket_priority (string)",  # Deprecated param name (lexical trap)
        "assignee_group",
        "notification_channel",
        "message_body",
        "Legacy Tool Instructions",
        "DO NOT USE",
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TASKS: dict[str, Task] = {
    "easy":   TASK_1,
    "medium": TASK_2,
    "hard":   TASK_3,
}


def get_task(difficulty: Literal["easy", "medium", "hard"]) -> Task:
    """Return the Task dataclass for the given difficulty level."""
    if difficulty not in ALL_TASKS:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. Available: {list(ALL_TASKS.keys())}"
        )
    return ALL_TASKS[difficulty]
