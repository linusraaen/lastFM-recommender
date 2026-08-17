"""
The set of files needed to serve /recommend, mapping a flat filename (as
stored in the Hugging Face Hub model repo) to its local path (as src/config.py
expects). Single source of truth for both sides of the hosted-demo pipeline:

    scripts/stage_artifacts.py         -> uploads these, local path to filename
    src/serve/download_artifacts.py    -> downloads these, filename to local path
"""

from __future__ import annotations

from pathlib import Path

from src import config

ARTIFACT_FILES: dict[str, Path] = {
    "item_tower.pt": config.ITEM_TOWER_PATH,
    "user_tower.pt": config.USER_TOWER_PATH,
    "artist_embeddings.npy": config.ARTIST_EMBEDDINGS_PATH,
    "artists.faiss": config.FAISS_INDEX_PATH,
    "artist_features.parquet": config.ARTIST_FEATURES_PATH,
    "artist_ids.json": config.ARTIST_ID_MAP_PATH,
    "train.parquet": config.SPLIT_DIR / "train.parquet",
    "artists.jsonl": config.ARTISTS_PATH,
}
