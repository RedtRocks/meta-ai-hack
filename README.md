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
├── __init__.py                    ← Package exports (Action, Observation, Client)
├── models.py                      ← PromptForgeAction + PromptForgeObservation
├── client.py                      ← PromptForgeEnvClient
└── server/
    ├── app.py                     ← create_app(PromptForgeEnvironment, ...)
    ├── promptforge_environment.py ← openenv-core Environment subclass
    ├── ast_parser.py              ← Recursive prompt AST (DOCUMENT→SECTION→RULE)
    ├── tasks.py                   ← 3 PromptDebt scenarios + ground truth fixtures
    ├── graders.py                 ← Deterministic JSON Grader (OpenAI/Groq API)
    ├── reward.py                  ← (token_reduction × quality) − penalties
    └── gradio_ui.py               ← Gradio UI mounted on the FastAPI app

Root:
├── app.py                        ← Compatibility wrapper for the PromptForge server entrypoint
├── inference.py                  ← Baseline agent (OpenAI-compatible client)
├── Dockerfile                    ← Root build for HF Spaces deployment
├── openenv.yaml                  ← OpenEnv submission manifest
└── test_smoke.py                 ← Lightweight smoke checks (log contract + openenv validate)
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

The current submission is validated with `openenv validate` and the root `inference.py` emits the required `[START]`, `[STEP]`, and `[END]` lines, including the final `score` field expected by the sample inference contract.

### Submission Checklist

- `inference.py` is in the project root.
- `inference.py` uses `OpenAI` client and requires `HF_TOKEN`.
- OpenEnv manifest exists: `openenv.yaml`.
- Docker build context exists: `Dockerfile`.
- `python -m openenv.cli validate` passes.

### Run Locally

```bash
pip install -r requirements.txt
# create .env manually if needed and set at least: HF_TOKEN

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
openenv push --repo-id raunaqmittal2004/promptforge

# OR upload manually
python upload_space.py
```

### Validate Before Submission

```bash
python test_smoke.py
python -m openenv.cli validate
bash scripts/validate-submission.sh https://your-space.hf.space .
```

### Baseline Performance Scores

Measured baseline benchmark (3 runs, local server on `127.0.0.1:7862`, grader model `llama-3.1-8b-instant`):

| Task | Runs | Score Mean | Reward Mean | Success Rate |
|---|---:|---:|---:|---:|
| Easy | 3 | 0.55 | 0.33 | 1.00 |
| Medium | 3 | 0.64 | 0.47 | 1.00 |
| Hard | 3 | 0.59 | 0.38 | 1.00 |

These values are produced by running:

```bash
ENV_BASE_URL=http://127.0.0.1:7862 python scripts/benchmark_baseline.py --runs 3 --env-base-url http://127.0.0.1:7862
```

Hard-task tuning is reproducible with an exhaustive search script:

```bash
python scripts/tune_hard_plan.py --env-base-url http://127.0.0.1:7862
```

Current exhaustive optimum (under the predefined hard patterns) matches the baseline hard plan in `inference.py`, yielding reward `0.377907` with quality-preserving submit.

Recent quality improvement: the hard-task grader now accepts both `HIGH` and `CRITICAL` ticket priorities for severe payment outages, which better reflects real incident triage behavior.

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
