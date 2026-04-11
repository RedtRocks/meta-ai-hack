"""upload_space.py — Upload PromptForge to Hugging Face Spaces.

Uses the openenv push command (preferred) or HfApi directly.

Usage:
    python upload_space.py
    # OR use the official CLI:
    cd promptforge && openenv push --repo-id raunaqmittal2004/promptforge
"""

from huggingface_hub import HfApi
import os

api = HfApi()

print("Uploading PromptForge to HuggingFace Space...")
api.upload_folder(
    folder_path=".",
    repo_id="raunaqmittal2004/promptforge",
    repo_type="space",
    allow_patterns=["*"],
    ignore_patterns=[
        ".git/*",
        ".venv/*",
        "venv/*",
        ".env",
        "__pycache__/*",
        "*.pyc",
        ".dockerignore",
        ".gitignore",
        ".agents/*",
        "upload_space.py",
        "docs/*",
        "scripts/*",
        "test_smoke.py",
    ]
)
print("Upload Complete!")
