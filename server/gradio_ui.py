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
    # CSS to enforce Poppins font and remove any pseudo-gradients
    custom_css = """
    * {
        font-family: 'Poppins', sans-serif !important;
        background-image: none !important; 
    }
    .gradio-container {
        background: #EBF4F6 !important;
    }
    .dark .gradio-container {
        background: #09637E !important;
    }
    textarea {
        font-family: monospace !important;
    }
    """

    # Flat, sleek theme using the requested color palette
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Poppins"), "sans-serif"]
    ).set(
        body_background_fill="#EBF4F6",
        body_background_fill_dark="#09637E",
        button_primary_background_fill="#088395",
        button_primary_background_fill_dark="#088395",
        button_primary_background_fill_hover="#09637E",
        button_secondary_background_fill="#7AB2B2",
        button_secondary_background_fill_hover="#088395",
        button_primary_text_color="white",
        button_secondary_text_color="white",
    )

    with gr.Blocks(title="PromptForge", css=custom_css, theme=theme) as demo:
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
        
    return gr.mount_gradio_app(fastapi_app, demo, path="/")
