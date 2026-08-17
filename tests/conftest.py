"""
Synthetic-data fixtures. Nothing here touches the real Last.fm API or the
real data/ directory -- everything is generated in a pytest tmp_path and
config module paths are monkey-patched to point at it for the session, so
the pipeline can be exercised end-to-end without a real crawl.
"""

from __future__ import annotations

import json
import random

import pytest

from src import config

ARTIST_NAMES = [f"Artist {i}" for i in range(40)]
TAG_POOL = ["shoegaze", "indie", "electronic", "jazz", "rock", "pop", "ambient", "metal"]


def _write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture(scope="session")
def synthetic_raw_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_project")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    rng = random.Random(0)

    users = [f"user{i}" for i in range(12)]
    interactions = []
    for u in users:
        n = rng.randint(8, 20)
        for a in rng.sample(ARTIST_NAMES, n):
            interactions.append({"user_id": u, "artist": a, "playcount": rng.randint(1, 500)})
    _write_jsonl(raw / "interactions.jsonl", interactions)

    # first half of users are the eval cohort: timestamped scrobbles -> enables a temporal split
    scrobbles = []
    base_ts = 1_700_000_000
    for u in users[:6]:
        user_artists = [row["artist"] for row in interactions if row["user_id"] == u]
        for i, a in enumerate(user_artists):
            for _ in range(rng.randint(1, 3)):
                ts = base_ts + i * 3600 + rng.randint(0, 3599)
                scrobbles.append({"user_id": u, "artist": a, "track": "t", "ts": ts})
    _write_jsonl(raw / "scrobbles.jsonl", scrobbles)

    # a few artists deliberately left without metadata, to exercise the missing-artist fallback
    artist_rows = []
    for name in ARTIST_NAMES[:-3]:
        artist_rows.append({
            "artist": name,
            "tags": rng.sample(TAG_POOL, 3),
            "listeners": rng.randint(100, 100_000),
            "bio": f"{name} is a band that plays {rng.choice(TAG_POOL)} music.",
        })
    _write_jsonl(raw / "artists.jsonl", artist_rows)

    return root


@pytest.fixture(scope="session")
def synthetic_project(synthetic_raw_dir):
    """Points src.config at the synthetic project dir and shrinks hyperparams for speed."""
    root = synthetic_raw_dir
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    models = root / "models"
    split = processed / "split"
    for d in (processed, models, split):
        d.mkdir(parents=True, exist_ok=True)

    overrides = {
        "RAW_DIR": raw,
        "PROCESSED_DIR": processed,
        "MODELS_DIR": models,
        "INTERACTIONS_PATH": raw / "interactions.jsonl",
        "SCROBBLES_PATH": raw / "scrobbles.jsonl",
        "ARTISTS_PATH": raw / "artists.jsonl",
        "MATRIX_PATH": processed / "matrix.parquet",
        "SPLIT_DIR": split,
        "ARTIST_ID_MAP_PATH": processed / "artist_ids.json",
        "ARTIST_FEATURES_PATH": processed / "artist_features.parquet",
        "ITEM_TOWER_PATH": models / "item_tower.pt",
        "USER_TOWER_PATH": models / "user_tower.pt",
        "ARTIST_EMBEDDINGS_PATH": models / "artist_embeddings.npy",
        "FAISS_INDEX_PATH": models / "artists.faiss",
        "EMBED_DIM": 16,
        "HIDDEN_DIM": 32,
    }
    originals = {k: getattr(config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(config, k, v)

    yield root

    for k, v in originals.items():
        setattr(config, k, v)


@pytest.fixture(scope="session")
def built_pipeline(synthetic_project):
    """Runs matrix building, feature extraction, training, and index building once per session."""
    from src.data import build_matrix
    from src.features import artist_features
    from src.model import train as train_module
    from src.serve import build_index

    build_matrix.build()
    artist_features.build(embed_bio=False)  # skip the sentence-transformers download in tests
    train_module.train(epochs=2, batch_size=8)
    build_index.build()
    return synthetic_project
