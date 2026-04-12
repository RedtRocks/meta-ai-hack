import json
import gradio as gr
from models import PromptForgeAction
from server.promptforge_environment import PromptForgeEnvironment

def create_env():
    return PromptForgeEnvironment()

def ui_reset(env):
    if env is None:
        env = create_env()
    obs = env.reset()
    status = "Environment Reset OK"
    return env, obs.model_dump_json(indent=2), status

def ui_step(env, action_str):
    if env is None:
        env = create_env()
    try:
        data = json.loads(action_str)
        action = PromptForgeAction(**data)
    except Exception as e:
        return env, f"JSON Error: {e}", "Action Parse Failed"
        
    try:
        obs = env.step(action)
        status = f"Step executed | Reward: {obs.reward:.2f} | Done: {obs.done}"
        return env, obs.model_dump_json(indent=2), status
    except Exception as e:
        return env, f"Step Error: {e}", "Execution Failed"

def ui_state(env):
    if env is None:
        env = create_env()
    return env, env.state.model_dump_json(indent=2), "State refreshed"

def build_ui(fastapi_app):
    # Aggressive CSS overrides — Gradio ignores theme tokens in light mode,
    # so we must use !important on actual DOM classes to force dark colors.
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* ── Global font & size ─────────────────────────────────────── */
    *, body, .gradio-container, .gradio-container *,
    p, span, label, h1, h2, h3, h4, h5, h6, li, a, div {
        font-family: 'Poppins', sans-serif !important;
    }
    body, .gradio-container {
        font-size: 16px !important;
    }
    h1 { font-size: 28px !important; font-weight: 700 !important; }
    h2 { font-size: 22px !important; font-weight: 600 !important; }
    h3 { font-size: 18px !important; font-weight: 600 !important; }
    p, span, label, li, a, div.prose {
        font-size: 15px !important;
        line-height: 1.6 !important;
    }
    textarea, pre, code {
        font-family: 'Menlo', 'Consolas', monospace !important;
        font-size: 14px !important;
    }

    /* ── Dark background on everything ──────────────────────────── */
    body,
    .gradio-container,
    .gr-panel,
    .contain,
    footer {
        background: #111111 !important;
        background-image: none !important;
        color: #E5E5EA !important;
    }

    /* Block panels (each widget wrapper) */
    .block, .gr-block, .gr-box, .gr-padded, .gr-group,
    .panel, .form, .wrap {
        background: #1C1C1E !important;
        border: 1px solid #38383A !important;
        border-radius: 12px !important;
        color: #E5E5EA !important;
    }

    /* Labels above inputs */
    .label-wrap, .label-wrap span,
    .block label span,
    label {
        color: #A1A1A6 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    /* Text areas, inputs, code blocks */
    textarea, input[type="text"],
    .code-wrap, .cm-editor, .cm-content, .cm-line,
    .cm-gutters, .cm-gutter {
        background: #000000 !important;
        color: #E5E5EA !important;
        border-color: #38383A !important;
    }

    /* Primary button — Apple blue */
    button.primary, button[variant="primary"],
    .gr-button-primary, .gr-button.primary {
        background: #0A84FF !important;
        background-image: none !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    button.primary:hover, button[variant="primary"]:hover,
    .gr-button-primary:hover {
        background: #0070E0 !important;
    }

    /* Secondary buttons */
    button.secondary, button:not(.primary),
    .gr-button-secondary, .gr-button.secondary {
        background: #2C2C2E !important;
        background-image: none !important;
        color: #E5E5EA !important;
        border: 1px solid #48484A !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 15px !important;
    }
    button.secondary:hover, button:not(.primary):hover,
    .gr-button-secondary:hover {
        background: #3A3A3C !important;
    }

    /* Markdown text */
    .prose, .prose *, .markdown-text, .md {
        color: #E5E5EA !important;
    }
    .prose strong, .prose b {
        color: #FFFFFF !important;
    }
    .prose code {
        background: #2C2C2E !important;
        color: #FF9F0A !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* Status text box */
    .gr-textbox input, .gr-textbox textarea {
        background: #1C1C1E !important;
        color: #E5E5EA !important;
        border: 1px solid #38383A !important;
    }

    /* Vertical spacing between QuickStart sidebar and Playground */
    .gr-row > .gr-column:last-child {
        padding-top: 20px !important;
    }
    .gr-row > .gr-column:first-child {
        padding-right: 24px !important;
    }

    /* Hide Gradio footer */
    footer { display: none !important; }
    """

    # Base theme — we still set tokens for any Gradio element that
    # respects them, but CSS above is the real enforcement layer.
    theme = gr.themes.Base(
        font=[gr.themes.GoogleFont("Poppins"), "sans-serif"],
    )

    with gr.Blocks(title="PromptForge") as demo:
        # Create a session-specific environment container
        env_state = gr.State(None)

        with gr.Row():
            # LEFT SIDEBAR
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Quick Start\n**Connect to this environment**")
                gr.Markdown("Connect from Python using `PromptForgeEnvClient`:")
                gr.Code(
                    value='''from client import PromptForgeEnvClient

with PromptForgeEnvClient.from_env("promptforge") as env:
    obs = env.reset()
    action = {"action_type": "PROBE", "node_id": "..."}
    obs = env.step(action)''',
                    language="python",
                    interactive=False
                )
                gr.Markdown("<br>**Or connect directly to a running server:**")
                gr.Code(
                    value='''from client import PromptForgeEnvClient
env = PromptForgeEnvClient(base_url="http://localhost:7860")''',
                    language="python",
                    interactive=False
                )
                gr.Markdown("<br>**Contribute to this environment**")
                gr.Markdown("Submit improvements via pull request on the Hugging Face Hub.")
                gr.Code(
                    value="openenv clone raunaqmittal2004/promptforge\ncd promptforge\nopenenv push",
                    language="shell",
                    interactive=False
                )
                
            # RIGHT PLAYGROUND
            with gr.Column(scale=3):
                gr.Markdown("## Playground\nClick **Reset** to start a new episode.")
                
                with gr.Group():
                    action_input = gr.Code(
                        label="Code (Action JSON)", 
                        language="json",
                        value='{\n  "action_type": "START_EPISODE",\n  "task_difficulty": "easy"\n}',
                        interactive=True,
                        lines=5
                    )
                    
                    with gr.Row():
                        step_btn = gr.Button("Step", variant="primary")
                        reset_btn = gr.Button("Reset")
                        state_btn = gr.Button("Get state")
                    
                    status_text = gr.Textbox(label="Status", interactive=False)
                    json_output = gr.Code(label="Raw JSON response", language="json", interactive=False, lines=20)

        # Event listeners
        reset_btn.click(fn=ui_reset, inputs=[env_state], outputs=[env_state, json_output, status_text])
        step_btn.click(fn=ui_step, inputs=[env_state, action_input], outputs=[env_state, json_output, status_text])
        state_btn.click(fn=ui_state, inputs=[env_state], outputs=[env_state, json_output, status_text])
        
    return gr.mount_gradio_app(fastapi_app, demo, path="/", app_kwargs={"css": custom_css, "theme": theme})
