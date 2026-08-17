"""
Copies the trained model artifacts into artifact_dist/, flat and renamed to
match what src/serve/download_artifacts.py expects to find in the HF Hub
model repo. Run this locally after training, then upload the whole folder:

    python -m scripts.stage_artifacts
    huggingface-cli login
    huggingface-cli upload <your-username>/<repo-name> artifact_dist/ . --repo-type model
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.serve.artifacts import ARTIFACT_FILES

DIST_DIR = Path(__file__).resolve().parent.parent / "artifact_dist"


def stage() -> None:
    DIST_DIR.mkdir(exist_ok=True)
    missing = [str(src) for src in ARTIFACT_FILES.values() if not src.exists()]
    if missing:
        raise SystemExit(
            "missing artifacts, run the full pipeline first (make data features train index):\n"
            + "\n".join(missing)
        )

    for filename, src in ARTIFACT_FILES.items():
        shutil.copy2(src, DIST_DIR / filename)
        print(f"staged {filename}")

    print(f"\nstaged {len(ARTIFACT_FILES)} files to {DIST_DIR}/ -- upload with:")
    print("  huggingface-cli upload <your-username>/<repo-name> artifact_dist/ . --repo-type model")


if __name__ == "__main__":
    stage()
