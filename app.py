"""FastAPI + Gradio application for the GitHub Issue Triage environment.

Exposes OpenEnv REST endpoints AND a rich Gradio playground UI.
Port 7860 for HuggingFace Spaces compatibility.
"""

from __future__ import annotations

import json
from typing import Optional

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import GitHubTriageEnv
from env.models import Action

# ═══════════════════════════════════════════════════════════════════════════
# FastAPI backend
# ═══════════════════════════════════════════════════════════════════════════

api = FastAPI(
    title="GitHub Issue Triage — OpenEnv",
    version="1.0.0",
    description="OpenEnv-compliant RL environment for training AI agents to triage GitHub issues.",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = GitHubTriageEnv()


class ResetRequest(BaseModel):
    task_id: str | None = None


from fastapi.responses import RedirectResponse




@api.get("/health")
def health():
    return {"status": "ok", "tasks": list(env.tasks.keys())}


@api.post("/reset")
def reset(body: ResetRequest | None = None):
    task_id = body.task_id if body is not None else None
    try:
        obs = env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs.model_dump()


@api.post("/step")
def step(action: Action):
    if env._state is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if env._state.get("done", False):
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset.")
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@api.get("/state")
def get_state():
    if env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return env.state()


# ═══════════════════════════════════════════════════════════════════════════
# Gradio Playground UI
# ═══════════════════════════════════════════════════════════════════════════

TASK_CHOICES = [
    ("🟢 Easy — Single Bug Report (task_easy)", "task_easy"),
    ("🟡 Medium — Mixed Inbox, 5 Issues (task_medium)", "task_medium"),
    ("🔴 Hard — Full Inbox + Security Threat (task_hard)", "task_hard"),
    ("🚨 Medium — Release Blockers (task_release_blocker)", "task_release_blocker"),
    ("☠️ Hard — Community Inbox + PII Exposure (task_community)", "task_community"),
]
LABEL_CHOICES = env.label_schema["labels"]
PRIORITY_CHOICES = ["P0", "P1", "P2", "P3"]

# Custom CSS for a bold, premium dark theme
CUSTOM_CSS = """
/* ── Global overrides ─────────────────────────────────────────── */
body, .gradio-container {
    background-color: #333333 !important;
    color: #E2E8F0 !important;
}
.gradio-container * {
    border-color: #666666 !important;
}
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Poppins', 'SF Pro Display', -apple-system, sans-serif !important;
}

/* ── Hero header ──────────────────────────────────────────────── */
.hero-header {
    background: #333333;
    border: 1px solid #666666;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-header h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #4183C4 !important;
    margin: 0 0 8px 0 !important;
}
.hero-header p {
    color: #E2E8F0 !important;
    font-size: 14px !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}

/* ── Issue card styling ───────────────────────────────────────── */
.issue-card {
    background: #333333;
    border: 1px solid #666666;
    border-left: 4px solid #4183C4;
    border-radius: 12px;
    padding: 24px;
    font-family: 'Poppins', sans-serif;
}
.issue-card h3 {
    color: #E2E8F0 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin: 0 0 12px 0 !important;
}
.issue-card .meta {
    color: #E2E8F0;
    font-size: 12px;
    margin-bottom: 16px;
}
.issue-card .body-text {
    color: #E2E8F0;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
}

/* ── Reward display ───────────────────────────────────────────── */
.reward-panel {
    background: #333333;
    border: 1px solid #666666;
    border-radius: 12px;
    padding: 20px;
}
.reward-total {
    font-size: 42px !important;
    font-weight: 800 !important;
    text-align: center;
    padding: 12px;
    color: #4183C4 !important;
}

/* ── Action buttons ───────────────────────────────────────────── */
.primary-btn {
    background: #4183C4 !important;
    border: 1px solid #4183C4 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #333333 !important;
}
.primary-btn:hover {
    opacity: 0.9 !important;
}
.reset-btn {
    background: #666666 !important;
    border: 1px solid #666666 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
}

/* ── Step log ─────────────────────────────────────────────────── */
.step-log {
    background: #333333;
    border: 1px solid #666666;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    max-height: 400px;
    overflow-y: auto;
    color: #E2E8F0;
}

/* ── Code block in issue ──────────────────────────────────────── */
.issue-card pre, .issue-card code {
    background: #666666 !important;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #E2E8F0 !important;
}

/* ── Sidebar info panel ───────────────────────────────────────── */
.info-panel {
    background: #333333;
    border: 1px solid #666666;
    border-radius: 12px;
    padding: 20px;
}
.info-panel h4 {
    color: #4183C4 !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin: 0 0 12px 0 !important;
}
.info-panel code {
    background: #666666;
    border: 1px solid #E2E8F0;
    border-radius: 4px;
    padding: 2px 6px;
    color: #E2E8F0;
    font-size: 12px;
}
"""


def _format_issue_html(obs_dict: dict | None) -> str:
    """Render the current issue as a styled HTML card."""
    if obs_dict is None:
        return '<div class="issue-card"><h3>No active episode</h3><p class="body-text">Click <strong>Reset</strong> to start triaging issues.</p></div>'

    issue = obs_dict.get("current_issue", {})
    if issue.get("id") == "done":
        return '<div class="issue-card" style="border-left-color: #4183C4;"><h3>✅ Episode Complete</h3><p class="body-text">All issues have been triaged. Click <strong>Reset</strong> to start a new episode.</p></div>'

    title = issue.get("title", "")
    body = issue.get("body", "").replace("<", "&lt;").replace(">", "&gt;")
    author = issue.get("author", "unknown")
    created = issue.get("created_at", "")[:10]
    tags = issue.get("tags", [])
    tag_html = "".join(f'<span style="background:#666666;color:#E2E8F0;padding:2px 8px;border-radius:12px;font-size:11px;margin-right:4px;">{t}</span>' for t in tags)

    step = obs_dict.get("step_number", 0)
    remaining = obs_dict.get("issues_remaining", 0)
    max_steps = obs_dict.get("max_steps", 0)

    # Convert markdown code blocks to pre tags
    import re
    body = re.sub(r'```(\w*)\n(.*?)```', r'<pre style="background:#666666;border:1px solid #E2E8F0;border-radius:6px;padding:12px;overflow-x:auto;color:#E2E8F0;font-size:12px;">\2</pre>', body, flags=re.DOTALL)
    body = re.sub(r'`([^`]+)`', r'<code style="background:#666666;border:1px solid #E2E8F0;border-radius:4px;padding:1px 5px;color:#E2E8F0;font-size:12px;">\1</code>', body)
    body = body.replace("\n", "<br>")

    return f"""
    <div class="issue-card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="color:#4183C4;font-size:18px;">●</span>
                <span style="color:#E2E8F0;font-size:12px;">Issue #{issue.get('id', '')}</span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;">
                <span style="background:transparent;color:#4183C4;border:1px solid #4183C4;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;">Step {step + 1}/{max_steps}</span>
                <span style="background:transparent;color:#E2E8F0;border:1px solid #E2E8F0;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;">{remaining} remaining</span>
            </div>
        </div>
        <h3 style="margin:0 0 8px 0 !important;">{title}</h3>
        <div class="meta">
            <span style="margin-right:12px;">👤 <strong>{author}</strong></span>
            <span style="margin-right:12px;">📅 {created}</span>
            {tag_html}
        </div>
        <div class="body-text">{body}</div>
    </div>
    """


def _format_reward_html(reward_dict: dict | None) -> str:
    """Render a reward breakdown as styled HTML."""
    if reward_dict is None:
        return '<div class="reward-panel"><p style="text-align:center;color:#E2E8F0;">Submit an action to see the reward breakdown</p></div>'

    total = reward_dict.get("total", 0)
    if total >= 0.7:
        color_class = "reward-high"
    elif total >= 0.4:
        color_class = "reward-mid"
    else:
        color_class = "reward-low"

    components = [
        ("🏷️ Labels", reward_dict.get("label_score", 0), 0.30, "#4183C4"),
        ("🔗 Duplicate", reward_dict.get("duplicate_score", 0), 0.25, "#4183C4"),
        ("⚡ Priority", reward_dict.get("priority_score", 0), 0.20, "#4183C4"),
        ("💬 Comment", reward_dict.get("comment_score", 0), 0.15, "#4183C4"),
        ("🔒 Security", reward_dict.get("security_score", 0), 0.10, "#4183C4"),
    ]

    bars = ""
    for name, score, max_val, color in components:
        pct = max(0, min(100, (score / max_val * 100) if max_val > 0 else 0))
        sign = "+" if score >= 0 else ""
        bars += f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#E2E8F0;font-size:13px;">{name}</span>
                <span style="color:{color};font-weight:600;font-size:13px;">{sign}{score:.3f} / {max_val:.2f}</span>
            </div>
            <div style="background:#666666;border-radius:4px;height:6px;overflow:hidden;">
                <div style="background:{color};width:{pct}%;height:100%;border-radius:4px;transition:width 0.5s ease;"></div>
            </div>
        </div>
        """

    return f"""
    <div class="reward-panel">
        <div class="reward-total" style="color:#4183C4;">{total:.3f}</div>
        <p style="text-align:center;color:#E2E8F0;font-size:12px;margin:-8px 0 16px 0;">Total Reward</p>
        {bars}
    </div>
    """


def _format_step_log(history: list) -> str:
    """Format the step history as a monospace log."""
    if not history:
        return f'<div class="step-log">Episode started. Submit a triage action to see the step logs.</div>'

    lines = []
    for i, entry in enumerate(history):
        total = entry.get("total", 0)
        if total >= 0.7:
            color = "#4183C4"
        elif total >= 0.4:
            color = "#4183C4"
        else:
            color = "#4183C4"
        lines.append(f'<span style="color:#E2E8F0;">Step {i+1:2d}</span>  <span style="color:{color};font-weight:600;">reward={total:+.3f}</span>  <span style="color:#E2E8F0;">label={entry.get("label_score",0):.2f} dup={entry.get("duplicate_score",0):.2f} pri={entry.get("priority_score",0):.2f} cmt={entry.get("comment_score",0):.2f} sec={entry.get("security_score",0):.2f}</span>')

    return f'<div class="step-log">{"<br>".join(lines)}</div>'


# ── State ────────────────────────────────────────────────────────────────

_current_obs: dict | None = None
_step_history: list = []
_last_reward: dict | None = None


def do_reset(task_id: str):
    global _current_obs, _step_history, _last_reward
    _step_history = []
    _last_reward = None

    obs = env.reset(task_id)
    _current_obs = obs.model_dump()

    task = env.tasks[task_id]
    status = f"✅ Episode started — **{task.get('difficulty', '').upper()}** — {task['max_steps']} issues to triage"

    return (
        _format_issue_html(_current_obs),
        _format_reward_html(None),
        _format_step_log([]),
        status,
        json.dumps(_current_obs, indent=2),
    )


def do_step(labels_list, priority, is_duplicate, duplicate_of, needs_info, comment, is_security, close):
    global _current_obs, _step_history, _last_reward

    if env._state is None or env._state.get("done"):
        return (
            _format_issue_html(_current_obs),
            _format_reward_html(_last_reward),
            _format_step_log(_step_history),
            "⚠️ No active episode. Click **Reset** first.",
            "{}",
        )

    # labels_list comes directly from CheckboxGroup as a Python list
    labels = labels_list if labels_list else []

    action = Action(
        labels=labels,
        priority=priority,
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of if is_duplicate and duplicate_of else None,
        needs_info=needs_info,
        comment=comment if needs_info and comment else None,
        is_security=is_security,
        close=close,
    )

    obs, reward, done, info = env.step(action)
    _current_obs = obs.model_dump()
    _last_reward = reward.model_dump()
    _step_history.append(_last_reward)

    if done:
        total = sum(r.get("total", 0) for r in _step_history)
        avg = total / len(_step_history) if _step_history else 0
        status = f"🏁 Episode complete — **{len(_step_history)} steps** — avg reward: **{avg:.3f}**"
    else:
        status = f"Step {info['step']} complete — reward: **{reward.total:.3f}**"

    return (
        _format_issue_html(_current_obs),
        _format_reward_html(_last_reward),
        _format_step_log(_step_history),
        status,
        json.dumps(_current_obs, indent=2),
    )


def do_get_state():
    if env._state is None:
        return "{}"
    return json.dumps(env.state(), indent=2, default=str)


# ── Build Gradio Interface ───────────────────────────────────────────────

my_blue = gr.themes.Color(c50="#4183C4", c100="#4183C4", c200="#4183C4", c300="#4183C4", c400="#4183C4", c500="#4183C4", c600="#4183C4", c700="#4183C4", c800="#4183C4", c900="#4183C4", c950="#4183C4")
my_gray = gr.themes.Color(c50="#E2E8F0", c100="#E2E8F0", c200="#E2E8F0", c300="#E2E8F0", c400="#E2E8F0", c500="#666666", c600="#666666", c700="#666666", c800="#333333", c900="#333333", c950="#333333")

with gr.Blocks(
    css=CUSTOM_CSS,
    title="GitHub Issue Triage — OpenEnv Playground",
    theme=gr.themes.Base(
        primary_hue=my_blue,
        secondary_hue=my_gray,
        neutral_hue=my_gray,
        font=gr.themes.GoogleFont("Poppins"),
    ).set(
        background_fill_primary="#333333",
        background_fill_primary_dark="#333333",
        background_fill_secondary="#333333",
        background_fill_secondary_dark="#333333",
        body_background_fill="#333333",
        body_background_fill_dark="#333333",
        block_background_fill="#333333",
        block_background_fill_dark="#333333",
        block_border_color="#666666",
        block_border_color_dark="#666666",
        block_label_text_color="#E2E8F0",
        block_label_text_color_dark="#E2E8F0",
        block_title_text_color="#E2E8F0",
        block_title_text_color_dark="#E2E8F0",
        body_text_color="#E2E8F0",
        body_text_color_dark="#E2E8F0",
        body_text_color_subdued="#E2E8F0",
        body_text_color_subdued_dark="#E2E8F0",
        color_accent="#4183C4",
        color_accent_soft="#4183C4",
        input_background_fill="#666666",
        input_background_fill_dark="#666666",
        input_border_color="#E2E8F0",
        input_border_color_dark="#E2E8F0",
        button_primary_background_fill="#4183C4",
        button_primary_background_fill_dark="#4183C4",
        button_primary_background_fill_hover="#4183C4",
        button_primary_background_fill_hover_dark="#4183C4",
        button_primary_text_color="#333333",
        button_primary_text_color_dark="#333333",
        button_secondary_background_fill="#666666",
        button_secondary_background_fill_dark="#666666",
        button_secondary_text_color="#E2E8F0",
        button_secondary_text_color_dark="#E2E8F0",
        border_color_primary="#666666",
        border_color_primary_dark="#666666",
        shadow_drop="none",
        shadow_drop_lg="none",
        panel_background_fill="#333333",
        panel_background_fill_dark="#333333",
    ),
) as demo:

    # ── Header ───────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-header">
        <h1>🏷️ GitHub Issue Triage</h1>
        <p>OpenEnv-compliant RL Playground — Train agents to triage issues like an open source maintainer</p>
        <div style="display:flex;gap:8px;margin-top:12px;">
            <span style="background:transparent;color:#4183C4;padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600;border:1px solid #4183C4;">● OpenEnv v1.0</span>
            <span style="background:transparent;color:#E2E8F0;padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600;border:1px solid #E2E8F0;">5 Tasks</span>
            <span style="background:transparent;color:#E2E8F0;padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600;border:1px solid #E2E8F0;">acme/payments-sdk</span>
        </div>
    </div>
    """)

    with gr.Tabs() as tabs:

        # ═══════════════  TAB 1: PLAYGROUND  ═══════════════
        with gr.TabItem("🎮 Playground", id="playground"):
            status_md = gr.Markdown("Click **Reset** to start an episode.", elem_classes=["status-bar"])

            with gr.Row(equal_height=False):

                # ── Left: Issue + Action Form ────────────────────
                with gr.Column(scale=3):
                    issue_html = gr.HTML(
                        _format_issue_html(None),
                        label="Current Issue",
                    )

                    gr.Markdown("### ⚡ Triage Action", elem_classes=["section-title"])
                    with gr.Row():
                        task_dd = gr.Dropdown(
                            choices=TASK_CHOICES,
                            value="task_easy",
                            label="Task",
                            scale=2,
                        )
                        reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1, elem_classes=["reset-btn"])

                    gr.Markdown("**🏷️ Labels** — select all that apply:")
                    labels_input = gr.CheckboxGroup(
                        choices=LABEL_CHOICES,
                        value=[],
                        label="Labels",
                    )
                    with gr.Row():
                        priority_dd = gr.Dropdown(
                            choices=PRIORITY_CHOICES,
                            value="P2",
                            label="Priority",
                            scale=1,
                        )

                    with gr.Row():
                        is_duplicate_cb = gr.Checkbox(label="Is Duplicate", value=False)
                        duplicate_of_input = gr.Textbox(
                            label="Duplicate Of (issue ID)",
                            placeholder="existing-007",
                            scale=2,
                        )
                        is_security_cb = gr.Checkbox(label="Is Security", value=False)
                        close_cb = gr.Checkbox(label="Close Issue", value=False)

                    with gr.Row():
                        needs_info_cb = gr.Checkbox(label="Needs Info", value=False)
                        comment_input = gr.Textbox(
                            label="Comment (if needs info)",
                            placeholder="Please provide your SDK version, Python version, and full error traceback.",
                            scale=3,
                        )

                    submit_btn = gr.Button("▶️ Submit Action", variant="primary", size="lg", elem_classes=["primary-btn"])

                # ── Right: Reward + Log ──────────────────────────
                with gr.Column(scale=2):
                    reward_html = gr.HTML(
                        _format_reward_html(None),
                        label="Reward Breakdown",
                    )
                    gr.Markdown("### 📋 Episode Log")
                    step_log_html = gr.HTML(
                        _format_step_log([]),
                        label="Step History",
                    )

            # Hidden raw observation
            raw_obs_json = gr.Code(label="Raw Observation JSON", language="json", visible=False)

            # ── Wire events ──────────────────────────────────────
            reset_btn.click(
                fn=lambda task_id: (*do_reset(task_id), []),
                inputs=[task_dd],
                outputs=[issue_html, reward_html, step_log_html, status_md, raw_obs_json, labels_input],
            )

            submit_btn.click(
                fn=do_step,
                inputs=[
                    labels_input, priority_dd, is_duplicate_cb, duplicate_of_input,
                    needs_info_cb, comment_input, is_security_cb, close_cb,
                ],
                outputs=[issue_html, reward_html, step_log_html, status_md, raw_obs_json],
            )

        # ═══════════════  TAB 2: QUICK START  ═══════════════
        with gr.TabItem("📖 Quick Start", id="quickstart"):
            gr.HTML("""
            <div class="info-panel" style="margin-bottom:16px;">
                <h4>Connect to this environment</h4>
                <p style="color:#E2E8F0;font-size:13px;margin-bottom:12px;">Connect from Python using requests:</p>
                <pre style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:16px;color:#E2E8F0;font-size:13px;overflow-x:auto;font-family:'JetBrains Mono',monospace;">
import requests, json

BASE = "https://raunaqmittal2004-github-issue-env.hf.space"

# Reset environment
obs = requests.post(f"{BASE}/reset", json={"task_id": "task_easy"}).json()
print(f"Issue: {obs['current_issue']['title']}")

# Submit triage action
action = {
    "labels": ["bug", "needs-reproduction"],
    "priority": "P1",
    "is_duplicate": False,
    "duplicate_of": None,
    "needs_info": False,
    "comment": None,
    "is_security": False,
    "close": False
}
result = requests.post(f"{BASE}/step", json=action).json()
print(f"Reward: {result['reward']['total']}")
print(f"Done: {result['done']}")
                </pre>
            </div>

            <div class="info-panel" style="margin-bottom:16px;">
                <h4>Available Tasks</h4>
                <table style="width:100%;border-collapse:collapse;color:#E2E8F0;font-size:13px;">
                    <tr style="border-bottom:1px solid #666666;">
                        <th style="text-align:left;padding:8px;color:#4183C4;">Task ID</th>
                        <th style="text-align:left;padding:8px;color:#4183C4;">Difficulty</th>
                        <th style="text-align:center;padding:8px;color:#4183C4;">Issues</th>
                        <th style="text-align:left;padding:8px;color:#4183C4;">Challenge</th>
                    </tr>
                    <tr style="border-bottom:1px solid #666666;">
                        <td style="padding:8px;"><code style="color:#E2E8F0;">task_easy</code></td>
                        <td style="padding:8px;"><span style="color:#4183C4;">Easy</span></td>
                        <td style="text-align:center;padding:8px;">1</td>
                        <td style="padding:8px;">Single clean bug report</td>
                    </tr>
                    <tr style="border-bottom:1px solid #666666;">
                        <td style="padding:8px;"><code style="color:#E2E8F0;">task_medium</code></td>
                        <td style="padding:8px;"><span style="color:#4183C4;">Medium</span></td>
                        <td style="text-align:center;padding:8px;">5</td>
                        <td style="padding:8px;">Mixed: bugs, feature, duplicate, needs-info</td>
                    </tr>
                    <tr style="border-bottom:1px solid #666666;">
                        <td style="padding:8px;"><code style="color:#E2E8F0;">task_hard</code></td>
                        <td style="padding:8px;"><span style="color:#4183C4;">Hard</span></td>
                        <td style="text-align:center;padding:8px;">10</td>
                        <td style="padding:8px;">Full inbox with disguised security vulnerability</td>
                    </tr>
                    <tr style="border-bottom:1px solid #666666;">
                        <td style="padding:8px;"><code style="color:#E2E8F0;">task_release_blocker</code></td>
                        <td style="padding:8px;"><span style="color:#4183C4;">Medium</span></td>
                        <td style="text-align:center;padding:8px;">4</td>
                        <td style="padding:8px;">Release blockers: double-billing, Python 3.12 compat</td>
                    </tr>
                    <tr>
                        <td style="padding:8px;"><code style="color:#E2E8F0;">task_community</code></td>
                        <td style="padding:8px;"><span style="color:#4183C4;">Hard</span></td>
                        <td style="text-align:center;padding:8px;">6</td>
                        <td style="padding:8px;">Community inbox with disguised PII data exposure</td>
                    </tr>
                </table>
            </div>

            <div class="info-panel">
                <h4>API Endpoints</h4>
                <table style="width:100%;border-collapse:collapse;color:#E2E8F0;font-size:13px;">
                    <tr style="border-bottom:1px solid #666666;">
                        <th style="text-align:left;padding:8px;color:#4183C4;">Method</th>
                        <th style="text-align:left;padding:8px;color:#4183C4;">Path</th>
                        <th style="text-align:left;padding:8px;color:#4183C4;">Description</th>
                    </tr>
                    <tr style="border-bottom:1px solid #666666;">
                        <td style="padding:8px;"><span style="color:#4183C4;font-weight:600;">GET</span></td>
                        <td style="padding:8px;"><code style="color:#E2E8F0;">/</code></td>
                        <td style="padding:8px;">Discovery & metadata</td>
                    </tr>
                    <tr style="border-bottom:1px solid #666666;">
                        <td style="padding:8px;"><span style="color:#4183C4;font-weight:600;">GET</span></td>
                        <td style="padding:8px;"><code style="color:#E2E8F0;">/health</code></td>
                        <td style="padding:8px;">Health check + task list</td>
                    </tr>
                    <tr style="border-bottom:1px solid #666666;">
                        <td style="padding:8px;"><span style="color:#4183C4;font-weight:600;">POST</span></td>
                        <td style="padding:8px;"><code style="color:#E2E8F0;">/reset</code></td>
                        <td style="padding:8px;">Start a new episode</td>
                    </tr>
                    <tr style="border-bottom:1px solid #666666;">
                        <td style="padding:8px;"><span style="color:#4183C4;font-weight:600;">POST</span></td>
                        <td style="padding:8px;"><code style="color:#E2E8F0;">/step</code></td>
                        <td style="padding:8px;">Submit a triage action</td>
                    </tr>
                    <tr>
                        <td style="padding:8px;"><span style="color:#4183C4;font-weight:600;">GET</span></td>
                        <td style="padding:8px;"><code style="color:#E2E8F0;">/state</code></td>
                        <td style="padding:8px;">Inspect current env state</td>
                    </tr>
                </table>
            </div>
            """)

        # ═══════════════  TAB 3: RAW JSON  ═══════════════
        with gr.TabItem("🔧 Raw API", id="raw"):
            gr.Markdown("### Raw JSON Interface")
            with gr.Row():
                with gr.Column():
                    raw_task_dd = gr.Dropdown(choices=TASK_CHOICES, value="task_easy", label="Task")
                    raw_reset_btn = gr.Button("Reset", variant="secondary")
                    raw_state_btn = gr.Button("Get State", variant="secondary")
                with gr.Column(scale=2):
                    raw_output = gr.Code(label="Response", language="json")

            raw_reset_btn.click(
                fn=lambda t: json.dumps(env.reset(t).model_dump(), indent=2),
                inputs=[raw_task_dd],
                outputs=[raw_output],
            )
            raw_state_btn.click(
                fn=do_get_state,
                inputs=[],
                outputs=[raw_output],
            )


app = gr.mount_gradio_app(api, demo, path="/")
