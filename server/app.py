"""PromptForge FastAPI application — uses official openenv-core create_app.

Exposes:
    POST /reset          Reset the environment (starts easy task by default)
    POST /step           Apply a PromptForgeAction
    GET  /state          Inspect internal state
    GET  /schema         Action / observation JSON schemas
    WS   /ws             WebSocket endpoint for persistent sessions
    GET  /               Gradio web UI (built into openenv-core)

Port 7860 for HuggingFace Spaces compatibility.
"""

from __future__ import annotations

from openenv.core.env_server.http_server import create_fastapi_app

try:
    from models import PromptForgeAction, PromptForgeObservation
    from server.promptforge_environment import PromptForgeEnvironment
    from server.gradio_ui import build_ui
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import PromptForgeAction, PromptForgeObservation
    from server.promptforge_environment import PromptForgeEnvironment
    from server.gradio_ui import build_ui

app = create_fastapi_app(
    env=lambda: PromptForgeEnvironment(),
    action_cls=PromptForgeAction,
    observation_cls=PromptForgeObservation,
    max_concurrent_envs=4,
)

# Mount custom UI
app = build_ui(app)


def main() -> None:
    """Start the PromptForge server on the standard HF Spaces port."""
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
