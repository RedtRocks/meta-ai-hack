# 🏷️ GitHub Issue Triage — OpenEnv

An OpenEnv-compliant reinforcement learning environment that trains AI agents to
triage GitHub issues like an experienced open source maintainer.

Agents must label issues, detect duplicates, assign priorities, request missing
information, identify disguised security vulnerabilities, and close noise — all
within a realistic payments-SDK issue queue.

---

## Overview

Open source maintainers spend hours per day triaging incoming issues. This
environment simulates that workflow: the agent receives a queue of GitHub issues
and must make structured triage decisions for each one. A decomposed reward
function scores every dimension of a good triage — from label accuracy to
catching a critical security report hidden inside a routine feature request.

**Why it matters:** Automating issue triage lets maintainers focus on code.
Getting it wrong (especially missing a security report) has real consequences.

---

## Action Space

The agent must return an `Action` object for each issue:

| Field          | Type                             | Description                                                   |
| -------------- | -------------------------------- | ------------------------------------------------------------- |
| `labels`       | `List[str]`                      | Labels from the repo's `label_schema`                         |
| `priority`     | `Literal["P0","P1","P2","P3"]`   | P0 = security/data-loss … P3 = enhancement                    |
| `is_duplicate` | `bool`                           | Whether the issue duplicates an existing one                   |
| `duplicate_of` | `Optional[str]`                  | ID of the duplicate (e.g. `"existing-007"`)                   |
| `needs_info`   | `bool`                           | Whether the reporter needs to provide more info                |
| `comment`      | `Optional[str]`                  | Comment asking for the specific missing information            |
| `is_security`  | `bool`                           | Whether this is a security vulnerability                       |
| `close`        | `bool`                           | Close as invalid / noise / off-topic                           |

---

## Observation Space

At each step the agent receives an `Observation`:

| Field              | Type            | Description                                      |
| ------------------ | --------------- | ------------------------------------------------ |
| `task_id`          | `str`           | Current task identifier                          |
| `repo_name`        | `str`           | Repository name (`acme/payments-sdk`)            |
| `repo_description` | `str`           | Short description of the repository              |
| `label_schema`     | `List[str]`     | Valid labels for this repo                       |
| `current_issue`    | `Issue`         | The issue the agent must triage now              |
| `existing_issues`  | `List[Issue]`   | Pool of 25 existing issues for duplicate lookup  |
| `step_number`      | `int`           | Current step (0-indexed)                         |
| `max_steps`        | `int`           | Maximum steps in this task                       |
| `issues_remaining` | `int`           | Issues left in the queue                         |

---

## Reward Function

Total reward is clamped to **[-0.40, 1.00]**. Each component:

| Component        | Weight | Calculation                                                                                  |
| ---------------- | ------ | -------------------------------------------------------------------------------------------- |
| **Label**        | 0.30   | Jaccard similarity × 0.30                                                                    |
| **Duplicate**    | 0.25   | Correct flag + correct ID → 0.25; right flag, wrong ID → 0.10; miss → 0.0                   |
| **Priority**     | 0.20   | Exact match → 0.20; one level off → 0.10; two+ levels → 0.0                                 |
| **Comment**      | 0.15   | Correct `needs_info` + comment with all required fields → 0.15; partial → scaled             |
| **Security**     | 0.10   | Correct flag → 0.10; **missed security → −0.40 penalty**; false alarm → 0.0                 |

> ⚠️ **Missing a real security issue incurs −0.40**, which can make the total reward negative.

---

## Tasks

| Task ID       | Difficulty | Issues | Key Challenge                                           |
| ------------- | ---------- | ------ | ------------------------------------------------------- |
| `task_easy`   | Easy       | 1      | Single clean bug report                                 |
| `task_medium` | Medium     | 5      | Mixed queue: 2 bugs, 1 feature, 1 duplicate, 1 needs-info |
| `task_hard`   | Hard       | 10     | Full inbox including a **disguised security vulnerability** |

---

## Baseline Scores

| Task ID       | Score |
| ------------- | ----- |
| `task_easy`   | TBD   |
| `task_medium` | TBD   |
| `task_hard`   | TBD   |

*Run `inference.py` to fill in baseline scores.*

---

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn app:app --port 7860
```

### Docker

```bash
docker build -t github-triage .
docker run -p 7860:7860 github-triage
```

### Run Baseline Agent

```bash
OPENAI_API_KEY=sk-... python inference.py
# or with a custom server URL:
python inference.py --base-url http://localhost:7860
```

---

## API Endpoints

| Method | Path      | Description                    |
| ------ | --------- | ------------------------------ |
| GET    | `/`       | Discovery / metadata           |
| GET    | `/health` | Health check + task list       |
| POST   | `/reset`  | Start a new episode            |
| POST   | `/step`   | Submit a triage action         |
| GET    | `/state`  | Inspect current env state      |

---

## HuggingFace Space

> 🔗 **https://your-hf-space.hf.space** *(placeholder — update after deployment)*

---

## License

MIT
