"""
The set of files needed to serve /recommend, mapping a flat filename (as
stored in the Hugging Face Hub model repo) to its local path (as src/config.py
expects). Single source of truth for both sides of the hosted-demo pipeline:

    scripts/stage_artifacts.py         -> uploads these, local path to filename
    src/serve/download_artifacts.py    -> downloads these, filename to local path

Notably absent: item_tower.pt and artist_features.parquet. Both are only
needed for *training* -- src/model/recommend.py looks up context-artist
embeddings directly from the already-precomputed artist_embeddings.npy rather
than re-running the item tower over raw features per request, so serving
never touches either. artist_features.parquet in particular is the largest
artifact by far (hundreds of MB at 100K+ artists); skipping it here is what
keeps the deployed container's memory footprint sane.
"""

from __future__ import annotations

from pathlib import Path

from src import config

ARTIFACT_FILES: dict[str, Path] = {
    "user_tower.pt": config.USER_TOWER_PATH,
    "artist_embeddings.npy": config.ARTIST_EMBEDDINGS_PATH,
    "artists.faiss": config.FAISS_INDEX_PATH,
    "artist_ids.json": config.ARTIST_ID_MAP_PATH,
    "train.parquet": config.SPLIT_DIR / "train.parquet",
    "artists.jsonl": config.ARTISTS_PATH,
}
