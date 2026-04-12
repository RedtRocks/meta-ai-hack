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
    # CSS to enforce Poppins font and sleek dark theme base
    custom_css = """
    * {
        font-family: 'Poppins', sans-serif !important;
        font-size: 15px;
    }
    .gradio-container {
        background: #111111 !important;
    }
    .dark .gradio-container {
        background: #111111 !important;
    }
    /* Make code editors slightly larger font */
    textarea {
        font-family: monospace !important;
        font-size: 14px !important;
    }
    /* Stylish subtle borders */
    .gr-box {
        border-radius: 8px !important;
        border: 1px solid #333333 !important;
    }
    """

    # Apple Dark Theme Colors 
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Poppins"), "sans-serif"],
        text_size="lg",  # Make general UI text larger
    ).set(
        body_background_fill="#111111",
        body_background_fill_dark="#111111",
        block_background_fill="#1C1C1E",
        block_background_fill_dark="#1C1C1E",
        block_label_background_fill="#2C2C2E",
        block_label_background_fill_dark="#2C2C2E",
        block_label_text_color="#EBEBF5",
        block_label_text_color_dark="#EBEBF5",
        button_primary_background_fill="#0A84FF",
        button_primary_background_fill_dark="#0A84FF",
        button_primary_background_fill_hover="#007AFF",
        button_secondary_background_fill="#2C2C2E",
        button_secondary_background_fill_hover="#3A3A3C",
        button_primary_text_color="white",
        button_secondary_text_color="#EBEBF5",
        border_color_primary="#38383A",
        border_color_primary_dark="#38383A",
        background_fill_primary="#000000",
        background_fill_primary_dark="#000000",
        background_fill_secondary="#1C1C1E",
        background_fill_secondary_dark="#1C1C1E",
        body_text_color="#EBEBF5",
        body_text_color_dark="#EBEBF5"
    )

    with gr.Blocks(title="PromptForge", css=custom_css, theme=theme) as demo:
        # Create a session-specific environment container
        env_state = gr.State(None)

        gr.Markdown("# 🔨 PromptForge Web UI\\nWelcome to the reinforcement learning environment for prompt debt elimination.")

        with gr.Row():
            # LEFT SIDEBAR - INSTRUCTIONS
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 📖 How to Test Manually")
                gr.Markdown(
                    "1. Click **Reset 🔄** on the right to start.\\n"
                    "2. Look at the **Raw JSON Response** area. Scroll down to `ast_summary` and find the `node_id` of the text you want to prune.\\n"
                    "3. Paste the following into the **Action JSON** box, replacing `<id>` entirely:\\n"
                )
                gr.Code(
                    value='{\\n  "action_type": "PRUNE_BRANCH",\\n  "node_id": "<id>"\\n}',
                    language="json",
                    interactive=False
                )
                gr.Markdown(
                    "4. Click **Step ▶**. Look at the `Status output` box to see your positive token-reduction reward!\\n"
                    "5. To finish, paste this and click Step:\\n"
                )
                gr.Code(
                    value='{\\n  "action_type": "SUBMIT"\\n}',
                    language="json",
                    interactive=False
                )
                
                gr.Markdown("---")
                gr.Markdown("### 💻 Connect via Code")
                gr.Code(
                    value='''from client import PromptForgeEnvClient
env = PromptForgeEnvClient(base_url="http://localhost:7860")
env.reset()''',
                    language="python",
                    interactive=False
                )
                
            # MIDDLE COLUMN - INPUTS & STATUS
            with gr.Column(scale=2, min_width=320):
                gr.Markdown("### 🕹️ Controls")
                
                action_input = gr.Code(
                    label="Code (Action JSON)", 
                    language="json",
                    value='{\\n  "action_type": "START_EPISODE",\\n  "task_difficulty": "easy"\\n}',
                    interactive=True,
                    lines=8
                )
                
                with gr.Row():
                    step_btn = gr.Button("Step ▶", variant="primary")
                    reset_btn = gr.Button("Reset 🔄")
                    state_btn = gr.Button("Get State")
                
                status_text = gr.Textbox(label="Status output", interactive=False)

            # RIGHT COLUMN - HUGE OUTPUT
            with gr.Column(scale=3, min_width=450):
                gr.Markdown("### 👁️ Observation Space")
                json_output = gr.Code(label="Raw JSON response", language="json", interactive=False, lines=25)

        # Event listeners
        reset_btn.click(fn=ui_reset, inputs=[env_state], outputs=[env_state, json_output, status_text])
        step_btn.click(fn=ui_step, inputs=[env_state, action_input], outputs=[env_state, json_output, status_text])
        state_btn.click(fn=ui_state, inputs=[env_state], outputs=[env_state, json_output, status_text])
        
    return gr.mount_gradio_app(fastapi_app, demo, path="/")

