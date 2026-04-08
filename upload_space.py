from huggingface_hub import HfApi
import os

api = HfApi()

print("Uploading files to space...")
api.upload_folder(
    folder_path=".",
    repo_id="raunaqmittal2004/github-issue-env",
    repo_type="space",
    allow_patterns=["*"],
    ignore_patterns=[
        ".git/*",
        ".venv/*",
        "venv/*",
        "env.bak/*",
        ".env",
        "docs/*",
        "scripts/*",
        "test_smoke.py",
        "__pycache__/*",
        "*.pyc",
        ".dockerignore",
        ".gitignore",
        ".agents/*",
        "upload_space.py"
    ]
)
print("Upload Complete!")
