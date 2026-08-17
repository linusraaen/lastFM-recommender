"""
Pulls the trained model artifacts from a Hugging Face Hub model repo into the
local paths src/config.py expects. Run this before starting the API in any
environment that doesn't already have data/ and models/ populated (e.g. a
freshly-built container for a Hugging Face Space) -- see start.sh.

    HF_MODEL_REPO=<your-username>/<repo-name> python -m src.serve.download_artifacts

Files already present locally are left alone, so this is safe to call even
when data/models were populated some other way (e.g. a local dev checkout).
"""

from __future__ import annotations

import os
import shutil

from huggingface_hub import hf_hub_download

from src.serve.artifacts import ARTIFACT_FILES


def download(repo_id: str | None = None) -> None:
    repo_id = repo_id or os.environ["HF_MODEL_REPO"]
    for filename, dest in ARTIFACT_FILES.items():
        if dest.exists():
            print(f"skip {filename}: {dest} already exists")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(cached_path, dest)
        print(f"downloaded {filename} -> {dest}")


if __name__ == "__main__":
    download()
