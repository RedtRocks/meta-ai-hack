"""FastAPI application exposing the GitHub Issue Triage environment."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import GitHubTriageEnv
from env.models import Action

# ── App setup ────────────────────────────────────────────────────────────

app = FastAPI(
    title="GitHub Issue Triage — OpenEnv",
    version="1.0.0",
    description=(
        "OpenEnv-compliant RL environment for training AI agents to triage "
        "GitHub issues like an experienced open source maintainer."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = GitHubTriageEnv()


# ── Request schemas ──────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str | None = None


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Landing / discovery endpoint."""
    return {
        "name": "github-issue-triage",
        "version": "1.0.0",
        "description": (
            "Train agents to triage GitHub issues like an experienced "
            "open source maintainer. Covers labeling, duplicate detection, "
            "priority assignment, info requests, and security identification."
        ),
        "endpoints": [
            {"method": "POST", "path": "/reset", "description": "Start a new episode"},
            {"method": "POST", "path": "/step", "description": "Submit an action"},
            {"method": "GET", "path": "/state", "description": "Get current env state"},
            {"method": "GET", "path": "/health", "description": "Health check"},
        ],
    }


@app.get("/health")
def health():
    """Health-check endpoint."""
    return {
        "status": "ok",
        "tasks": list(env.tasks.keys()),
    }


@app.post("/reset")
def reset(body: ResetRequest):
    """Reset the environment with a specific task.

    Returns the first Observation.
    """
    try:
        obs = env.reset(body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    """Submit an action and advance the environment.

    Returns observation, reward, done flag, and info dict.
    """
    if env._state is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if env._state.get("done", False):
        raise HTTPException(
            status_code=400, detail="Episode is done. Call /reset."
        )

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


@app.get("/state")
def get_state():
    """Return a snapshot of the current environment state."""
    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )
    return env.state()
