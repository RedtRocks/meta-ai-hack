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


@api.get("/")
def root():
    return {
        "name": "github-issue-triage",
        "version": "1.0.0",
        "description": "Train agents to triage GitHub issues like an experienced open source maintainer.",
        "endpoints": [
            {"method": "POST", "path": "/reset", "description": "Start a new episode"},
            {"method": "POST", "path": "/step", "description": "Submit an action"},
            {"method": "GET", "path": "/state", "description": "Get current env state"},
            {"method": "GET", "path": "/health", "description": "Health check"},
        ],
    }


@api.get("/health")
def health():
    return {"status": "ok", "tasks": list(env.tasks.keys())}


@api.post("/reset")
def reset(body: ResetRequest):
    try:
        obs = env.reset(body.task_id)
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

TASK_CHOICES = list(env.tasks.keys())
LABEL_CHOICES = env.label_schema["labels"]
PRIORITY_CHOICES = ["P0", "P1", "P2", "P3"]

# Custom CSS for a bold, premium dark theme
CUSTOM_CSS = """
/* ── Global overrides ─────────────────────────────────────────── */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif !important;
}

/* ── Hero header ──────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a1e2e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(88, 166, 255, 0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(163, 113, 247, 0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-header h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #58a6ff, #a371f7, #f778ba);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0 !important;
}
.hero-header p {
    color: #8b949e !important;
    font-size: 14px !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}

/* ── Issue card styling ───────────────────────────────────────── */
.issue-card {
    background: linear-gradient(145deg, #0d1117, #161b22);
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 12px;
    padding: 24px;
    font-family: 'Inter', sans-serif;
}
.issue-card h3 {
    color: #f0f6fc !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin: 0 0 12px 0 !important;
}
.issue-card .meta {
    color: #8b949e;
    font-size: 12px;
    margin-bottom: 16px;
}
.issue-card .body-text {
    color: #c9d1d9;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
}

/* ── Reward display ───────────────────────────────────────────── */
.reward-panel {
    background: linear-gradient(145deg, #0d1117, #161b22);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
}
.reward-total {
    font-size: 42px !important;
    font-weight: 800 !important;
    text-align: center;
    padding: 12px;
}
.reward-high { color: #3fb950 !important; }
.reward-mid { color: #d29922 !important; }
.reward-low { color: #f85149 !important; }

/* ── Status badges ────────────────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-active {
    background: rgba(63, 185, 80, 0.15);
    color: #3fb950;
    border: 1px solid rgba(63, 185, 80, 0.3);
}
.badge-done {
    background: rgba(136, 136, 136, 0.15);
    color: #8b949e;
    border: 1px solid rgba(136, 136, 136, 0.3);
}

/* ── Action buttons ───────────────────────────────────────────── */
.primary-btn {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    border: 1px solid #2ea043 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.primary-btn:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    box-shadow: 0 0 20px rgba(46, 160, 67, 0.3) !important;
}
.reset-btn {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: 1px solid #388bfd !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Step log ─────────────────────────────────────────────────── */
.step-log {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    max-height: 400px;
    overflow-y: auto;
}

/* ── Code block in issue ──────────────────────────────────────── */
.issue-card pre, .issue-card code {
    background: #161b22 !important;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

/* ── Sidebar info panel ───────────────────────────────────────── */
.info-panel {
    background: linear-gradient(145deg, #0d1117, #161b22);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
}
.info-panel h4 {
    color: #58a6ff !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin: 0 0 12px 0 !important;
}
.info-panel code {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 2px 6px;
    color: #f0883e;
    font-size: 12px;
}
"""


def _format_issue_html(obs_dict: dict | None) -> str:
    """Render the current issue as a styled HTML card."""
    if obs_dict is None:
        return '<div class="issue-card"><h3>No active episode</h3><p class="body-text">Click <strong>Reset</strong> to start triaging issues.</p></div>'

    issue = obs_dict.get("current_issue", {})
    if issue.get("id") == "done":
        return '<div class="issue-card" style="border-left-color: #3fb950;"><h3>✅ Episode Complete</h3><p class="body-text">All issues have been triaged. Click <strong>Reset</strong> to start a new episode.</p></div>'

    title = issue.get("title", "")
    body = issue.get("body", "").replace("<", "&lt;").replace(">", "&gt;")
    author = issue.get("author", "unknown")
    created = issue.get("created_at", "")[:10]
    tags = issue.get("tags", [])
    tag_html = "".join(f'<span style="background:#21262d;color:#8b949e;padding:2px 8px;border-radius:12px;font-size:11px;margin-right:4px;">{t}</span>' for t in tags)

    step = obs_dict.get("step_number", 0)
    remaining = obs_dict.get("issues_remaining", 0)
    max_steps = obs_dict.get("max_steps", 0)

    # Convert markdown code blocks to pre tags
    import re
    body = re.sub(r'```(\w*)\n(.*?)```', r'<pre style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;overflow-x:auto;color:#c9d1d9;font-size:12px;">\2</pre>', body, flags=re.DOTALL)
    body = re.sub(r'`([^`]+)`', r'<code style="background:#161b22;border:1px solid #30363d;border-radius:4px;padding:1px 5px;color:#f0883e;font-size:12px;">\1</code>', body)
    body = body.replace("\n", "<br>")

    return f"""
    <div class="issue-card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="color:#3fb950;font-size:18px;">●</span>
                <span style="color:#8b949e;font-size:12px;">Issue #{issue.get('id', '')}</span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;">
                <span style="background:rgba(88,166,255,0.15);color:#58a6ff;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;">Step {step + 1}/{max_steps}</span>
                <span style="background:rgba(163,113,247,0.15);color:#a371f7;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;">{remaining} remaining</span>
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
        return '<div class="reward-panel"><p style="text-align:center;color:#8b949e;">Submit an action to see the reward breakdown</p></div>'

    total = reward_dict.get("total", 0)
    if total >= 0.7:
        color_class = "reward-high"
    elif total >= 0.4:
        color_class = "reward-mid"
    else:
        color_class = "reward-low"

    components = [
        ("🏷️ Labels", reward_dict.get("label_score", 0), 0.30, "#58a6ff"),
        ("🔗 Duplicate", reward_dict.get("duplicate_score", 0), 0.25, "#a371f7"),
        ("⚡ Priority", reward_dict.get("priority_score", 0), 0.20, "#d29922"),
        ("💬 Comment", reward_dict.get("comment_score", 0), 0.15, "#3fb950"),
        ("🔒 Security", reward_dict.get("security_score", 0), 0.10, "#f85149"),
    ]

    bars = ""
    for name, score, max_val, color in components:
        pct = max(0, min(100, (score / max_val * 100) if max_val > 0 else 0))
        sign = "+" if score >= 0 else ""
        bars += f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:#c9d1d9;font-size:13px;">{name}</span>
                <span style="color:{color};font-weight:600;font-size:13px;">{sign}{score:.3f} / {max_val:.2f}</span>
            </div>
            <div style="background:#21262d;border-radius:4px;height:6px;overflow:hidden;">
                <div style="background:{color};width:{pct}%;height:100%;border-radius:4px;transition:width 0.5s ease;"></div>
            </div>
        </div>
        """

    return f"""
    <div class="reward-panel">
        <div class="reward-total {color_class}">{total:.3f}</div>
        <p style="text-align:center;color:#8b949e;font-size:12px;margin:-8px 0 16px 0;">Total Reward</p>
        {bars}
    </div>
    """


def _format_step_log(history: list) -> str:
    """Format the step history as a monospace log."""
    if not history:
        return '<div class="step-log"><span style="color:#8b949e;">No steps taken yet. Reset the environment to start.</span></div>'

    lines = []
    for i, entry in enumerate(history):
        total = entry.get("total", 0)
        if total >= 0.7:
            color = "#3fb950"
        elif total >= 0.4:
            color = "#d29922"
        else:
            color = "#f85149"
        lines.append(f'<span style="color:#8b949e;">Step {i+1:2d}</span>  <span style="color:{color};font-weight:600;">reward={total:+.3f}</span>  <span style="color:#8b949e;">label={entry.get("label_score",0):.2f} dup={entry.get("duplicate_score",0):.2f} pri={entry.get("priority_score",0):.2f} cmt={entry.get("comment_score",0):.2f} sec={entry.get("security_score",0):.2f}</span>')

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


def do_step(labels_str, priority, is_duplicate, duplicate_of, needs_info, comment, is_security, close):
    global _current_obs, _step_history, _last_reward

    if env._state is None or env._state.get("done"):
        return (
            _format_issue_html(_current_obs),
            _format_reward_html(_last_reward),
            _format_step_log(_step_history),
            "⚠️ No active episode. Click **Reset** first.",
            "{}",
        )

    # Parse labels
    labels = [l.strip() for l in labels_str.split(",") if l.strip()] if labels_str else []

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

with gr.Blocks(
    css=CUSTOM_CSS,
    title="GitHub Issue Triage — OpenEnv Playground",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0d1117",
        body_background_fill_dark="#0d1117",
        block_background_fill="#161b22",
        block_background_fill_dark="#161b22",
        block_border_color="#30363d",
        block_border_color_dark="#30363d",
        block_label_text_color="#8b949e",
        block_label_text_color_dark="#8b949e",
        block_title_text_color="#f0f6fc",
        block_title_text_color_dark="#f0f6fc",
        body_text_color="#c9d1d9",
        body_text_color_dark="#c9d1d9",
        body_text_color_subdued="#8b949e",
        body_text_color_subdued_dark="#8b949e",
        input_background_fill="#0d1117",
        input_background_fill_dark="#0d1117",
        input_border_color="#30363d",
        input_border_color_dark="#30363d",
        button_primary_background_fill="#238636",
        button_primary_background_fill_dark="#238636",
        button_primary_background_fill_hover="#2ea043",
        button_primary_background_fill_hover_dark="#2ea043",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#21262d",
        button_secondary_background_fill_dark="#21262d",
        button_secondary_text_color="#c9d1d9",
        button_secondary_text_color_dark="#c9d1d9",
        border_color_primary="#30363d",
        border_color_primary_dark="#30363d",
        shadow_drop="0 4px 12px rgba(0,0,0,0.4)",
        shadow_drop_lg="0 8px 24px rgba(0,0,0,0.5)",
    ),
) as demo:

    # ── Header ───────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-header">
        <h1>🏷️ GitHub Issue Triage</h1>
        <p>OpenEnv-compliant RL Playground — Train agents to triage issues like an open source maintainer</p>
        <div style="display:flex;gap:8px;margin-top:12px;">
            <span style="background:rgba(35,134,54,0.2);color:#3fb950;padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600;border:1px solid rgba(35,134,54,0.3);">● OpenEnv v1.0</span>
            <span style="background:rgba(88,166,255,0.1);color:#58a6ff;padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600;border:1px solid rgba(88,166,255,0.2);">5 Tasks</span>
            <span style="background:rgba(163,113,247,0.1);color:#a371f7;padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600;border:1px solid rgba(163,113,247,0.2);">acme/payments-sdk</span>
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
                            value=TASK_CHOICES[0],
                            label="Task",
                            scale=2,
                        )
                        reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1, elem_classes=["reset-btn"])

                    with gr.Row():
                        labels_input = gr.Textbox(
                            label="Labels (comma-separated)",
                            placeholder="bug, needs-reproduction",
                            scale=3,
                        )
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
                fn=do_reset,
                inputs=[task_dd],
                outputs=[issue_html, reward_html, step_log_html, status_md, raw_obs_json],
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
                <p style="color:#c9d1d9;font-size:13px;margin-bottom:12px;">Connect from Python using requests:</p>
                <pre style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:16px;color:#c9d1d9;font-size:13px;overflow-x:auto;font-family:'JetBrains Mono',monospace;">
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
                <table style="width:100%;border-collapse:collapse;color:#c9d1d9;font-size:13px;">
                    <tr style="border-bottom:1px solid #30363d;">
                        <th style="text-align:left;padding:8px;color:#58a6ff;">Task ID</th>
                        <th style="text-align:left;padding:8px;color:#58a6ff;">Difficulty</th>
                        <th style="text-align:center;padding:8px;color:#58a6ff;">Issues</th>
                        <th style="text-align:left;padding:8px;color:#58a6ff;">Challenge</th>
                    </tr>
                    <tr style="border-bottom:1px solid #21262d;">
                        <td style="padding:8px;"><code style="color:#f0883e;">task_easy</code></td>
                        <td style="padding:8px;"><span style="color:#3fb950;">Easy</span></td>
                        <td style="text-align:center;padding:8px;">1</td>
                        <td style="padding:8px;">Single clean bug report</td>
                    </tr>
                    <tr style="border-bottom:1px solid #21262d;">
                        <td style="padding:8px;"><code style="color:#f0883e;">task_medium</code></td>
                        <td style="padding:8px;"><span style="color:#d29922;">Medium</span></td>
                        <td style="text-align:center;padding:8px;">5</td>
                        <td style="padding:8px;">Mixed: bugs, feature, duplicate, needs-info</td>
                    </tr>
                    <tr style="border-bottom:1px solid #21262d;">
                        <td style="padding:8px;"><code style="color:#f0883e;">task_hard</code></td>
                        <td style="padding:8px;"><span style="color:#f85149;">Hard</span></td>
                        <td style="text-align:center;padding:8px;">10</td>
                        <td style="padding:8px;">Full inbox with disguised security vulnerability</td>
                    </tr>
                    <tr style="border-bottom:1px solid #21262d;">
                        <td style="padding:8px;"><code style="color:#f0883e;">task_release_blocker</code></td>
                        <td style="padding:8px;"><span style="color:#d29922;">Medium</span></td>
                        <td style="text-align:center;padding:8px;">4</td>
                        <td style="padding:8px;">Release blockers: double-billing, Python 3.12 compat</td>
                    </tr>
                    <tr>
                        <td style="padding:8px;"><code style="color:#f0883e;">task_community</code></td>
                        <td style="padding:8px;"><span style="color:#f85149;">Hard</span></td>
                        <td style="text-align:center;padding:8px;">6</td>
                        <td style="padding:8px;">Community inbox with disguised PII data exposure</td>
                    </tr>
                </table>
            </div>

            <div class="info-panel">
                <h4>API Endpoints</h4>
                <table style="width:100%;border-collapse:collapse;color:#c9d1d9;font-size:13px;">
                    <tr style="border-bottom:1px solid #30363d;">
                        <th style="text-align:left;padding:8px;color:#58a6ff;">Method</th>
                        <th style="text-align:left;padding:8px;color:#58a6ff;">Path</th>
                        <th style="text-align:left;padding:8px;color:#58a6ff;">Description</th>
                    </tr>
                    <tr style="border-bottom:1px solid #21262d;">
                        <td style="padding:8px;"><span style="color:#3fb950;font-weight:600;">GET</span></td>
                        <td style="padding:8px;"><code style="color:#f0883e;">/</code></td>
                        <td style="padding:8px;">Discovery & metadata</td>
                    </tr>
                    <tr style="border-bottom:1px solid #21262d;">
                        <td style="padding:8px;"><span style="color:#3fb950;font-weight:600;">GET</span></td>
                        <td style="padding:8px;"><code style="color:#f0883e;">/health</code></td>
                        <td style="padding:8px;">Health check + task list</td>
                    </tr>
                    <tr style="border-bottom:1px solid #21262d;">
                        <td style="padding:8px;"><span style="color:#58a6ff;font-weight:600;">POST</span></td>
                        <td style="padding:8px;"><code style="color:#f0883e;">/reset</code></td>
                        <td style="padding:8px;">Start a new episode</td>
                    </tr>
                    <tr style="border-bottom:1px solid #21262d;">
                        <td style="padding:8px;"><span style="color:#58a6ff;font-weight:600;">POST</span></td>
                        <td style="padding:8px;"><code style="color:#f0883e;">/step</code></td>
                        <td style="padding:8px;">Submit a triage action</td>
                    </tr>
                    <tr>
                        <td style="padding:8px;"><span style="color:#3fb950;font-weight:600;">GET</span></td>
                        <td style="padding:8px;"><code style="color:#f0883e;">/state</code></td>
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
                    raw_task_dd = gr.Dropdown(choices=TASK_CHOICES, value=TASK_CHOICES[0], label="Task")
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


# ── Mount Gradio on FastAPI ──────────────────────────────────────────────

app = gr.mount_gradio_app(api, demo, path="/web")

# Make Gradio the default page
@api.get("/ui", include_in_schema=False)
def redirect_to_web():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web")
