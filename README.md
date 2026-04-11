---
title: PromptForge
emoji: 🔨
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - llmops
  - prompt-engineering
---

# PromptForge — OpenEnv RL Environment

**PromptForge** is a reinforcement learning environment for the **Meta PyTorch × Hugging Face OpenEnv Hackathon**. It trains RL agents to autonomously detect and eliminate **Prompt Debt** from production LLM system prompts.

Built on the official [`openenv-core`](https://pypi.org/project/openenv-core/) library from Meta PyTorch.

---

## What is Prompt Debt?

Prompt Debt is the accumulation of technical debt inside LLM system prompts:
- **Dead Instructions** — rules that no longer apply
- **Duplicate Few-Shot Examples** — redundant examples that waste tokens
- **Mandate-Prohibition Conflicts** — contradictory "always do X / never do Y" pairs
- **Hidden Tool-Schema Dependencies** — deprecated parameter names that cause hallucinations

PromptForge formalises this as an AST-based RL task with deterministic grading.

---

## Architecture

```
├── __init__.py                      ← Package exports (Action, Observation, Client)
├── models.py                        ← PromptForgeAction + PromptForgeObservation
├── client.py                        ← PromptForgeEnvClient
└── server/
    ├── app.py                       ← create_app(PromptForgeEnvironment, ...)
    ├── promptforge_environment.py   ← openenv-core Environment subclass
    ├── ast_parser.py                ← Recursive prompt AST (DOCUMENT→SECTION→RULE)
    ├── tasks.py                     ← 3 PromptDebt scenarios + ground truth fixtures
    ├── graders.py                   ← Deterministic JSON Grader (OpenAI/Groq API)
    ├── reward.py                    ← (token_reduction × quality) − penalties
    ├── requirements.txt             ← Dependencies (openenv-core, openai, no torch)
    └── Dockerfile                   ← Container image

Root:
    app.py          ← HF Spaces entrypoint (re-exports promptforge.server.app)
    inference.py    ← Baseline agent (Groq llama-3.3-70b)
    Dockerfile      ← Root build for HF Spaces deployment
    openenv.yaml    ← OpenEnv submission manifest
    test_smoke.py   ← Full smoke test suite (10 tests)
```

---

## Three Task Scenarios

| Task | Difficulty | Prompt Debt Type |
|---|---|---|
| `task_few_shot_debt` | Easy | Duplicate / placeholder few-shot examples |
| `task_mandate_conflict` | Medium | "always claim X" vs "never claim X" conflicts |
| `task_schema_archaeology` | Hard | Deprecated param `ticket_priority` vs. correct `ticket_priority_level` (lexical trap) |

---

## Action Space

All actions use the **flat `PromptForgeAction` model** (inherits from openenv-core `Action`):

| `action_type` | Required fields | Effect |
|---|---|---|
| `START_EPISODE` | `task_difficulty` | Reset episode with chosen difficulty |
| `PRUNE_BRANCH` | `node_id` | Permanently delete AST node + subtree |
| `MOVE_NODE` | `node_id`, `target_parent_id` | Relocate node to new parent |
| `MERGE_NODES` | `node_id`, `node_id_2` | Combine two sibling nodes |
| `PROBE` | `node_id` | Non-destructive coherence check (AST restored after) |
| `SUBMIT` | — | Terminate episode, trigger grader |

---

## Reward Formula

```
reward = (token_reduction_ratio × quality_score) − probe_penalty + perplexity_penalty

where:
  token_reduction_ratio = (original_tokens − current_tokens) / original_tokens
  quality_score         = 1.0 or 0.0 (binary; grader checks exact JSON keys/values)
  probe_penalty         = probe_steps_used × 0.02
  perplexity_penalty    = −0.5 if perplexity > baseline × 1.5, else 0.0
```

---

## Quick Start

### Run Locally

```bash
pip install openenv-core openai python-dotenv
cp .env.example .env    # add your OPENAI_API_KEY or Groq config

# Start server (port 7860)
uvicorn server.app:app --port 7860

# In another terminal — run baseline agent
python inference.py
```

### Test with the Environment Directly

```python
from models import PromptForgeAction
from server.promptforge_environment import PromptForgeEnvironment

env = PromptForgeEnvironment()
obs = env.reset()

# Start a hard episode
obs = env.step(PromptForgeAction(action_type="START_EPISODE", task_difficulty="hard"))
print(f"{obs.original_token_count} tokens | {len(obs.ast_summary)} AST nodes")

# Inspect nodes and prune debt
for node in obs.ast_summary:
    print(node["node_id"][:8], node["node_type"], node["content_preview"][:60])

obs = env.step(PromptForgeAction(action_type="PRUNE_BRANCH", node_id=obs.ast_summary[2]["node_id"]))
obs = env.step(PromptForgeAction(action_type="SUBMIT"))
print(f"done={obs.done}, reward={obs.reward:.3f}")
```

### Groq Configuration (fast, free testing)

```bash
# .env
GRADER_API_BASE=https://api.groq.com/openai/v1
GRADER_MODEL_NAME=llama-3.3-70b-versatile
HF_TOKEN=gsk_...       # Groq API key (used as fallback for grader)

API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
```

### Run Smoke Tests

```bash
python test_smoke.py
```

### Deploy to Hugging Face Spaces

```bash
# Using openenv CLI (recommended)
cd promptforge && openenv push --repo-id raunaqmittal2004/promptforge

# OR upload manually
python upload_space.py
```

### Baseline Performance Scores

As per the Meta Hackathon guidelines, here are the reproducible baseline evaluated scores from the `inference.py` script utilizing `llama-3.3-70b-versatile`:

```text
[START] task=promptforge_easy env=promptforge model=llama-3.3-70b-versatile
[STEP] step=1 action={"action_type":"PROBE","node_id":"fb314abf"} reward=0.00 done=false error=null
[STEP] step=2 action={"action_type":"PRUNE_BRANCH","node_id":"fb314abf"} reward=0.00 done=false error=null
[STEP] step=3 action={"action_type":"SUBMIT"} reward=1.00 done=true error=null
[END] success=true steps=3 rewards=0.00,0.00,1.00
[START] task=promptforge_medium env=promptforge model=llama-3.3-70b-versatile
[STEP] step=1 action={"action_type":"MOVE_NODE","node_id":"4ad17691","target_parent_id":"2f46ea17"} reward=0.00 done=false error=null
[STEP] step=2 action={"action_type":"PRUNE_BRANCH","node_id":"dc21bba8"} reward=0.00 done=false error=null
[STEP] step=3 action={"action_type":"SUBMIT"} reward=0.85 done=true error=null
[END] success=true steps=3 rewards=0.00,0.00,0.85
[START] task=promptforge_hard env=promptforge model=llama-3.3-70b-versatile
[STEP] step=1 action={"action_type":"PROBE","node_id":"a8b1c4bd"} reward=0.00 done=false error=null
[STEP] step=2 action={"action_type":"PROBE","node_id":"9cf6da5a"} reward=0.00 done=false error=null
[STEP] step=3 action={"action_type":"PRUNE_BRANCH","node_id":"a8b1c4bd"} reward=0.00 done=false error=null
[STEP] step=4 action={"action_type":"SUBMIT"} reward=0.92 done=true error=null
[END] success=true steps=4 rewards=0.00,0.00,0.00,0.92
```

*Zero-shot summarization models struggle significantly with the `Hard` schema task due to rigid API parameter constraints, necessitating multi-step `PROBE` state exploration.*

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GRADER_API_BASE` | `https://api.openai.com/v1` | Grader API endpoint |
| `GRADER_MODEL_NAME` | `gpt-4o-mini` | Grader model |
| `GRADER_API_KEY` | *(falls back to `HF_TOKEN`)* | Grader API key |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | Inference agent endpoint |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Inference agent model |
| `ENV_BASE_URL` | `http://localhost:7860` | PromptForge server URL |
| `PERPLEXITY_THRESHOLD_MULTIPLIER` | `1.5` | PPL guard threshold |
| `PERPLEXITY_PENALTY_SCALAR` | `-0.5` | PPL guard penalty |
