"""
Builds the artist (item) feature table: multi-hot genre tags + a bio text
embedding + a log-scaled listener-count popularity feature.

Output: data/processed/artist_features.parquet, one row per artist_id, with
columns [artist_id, artist, tag_multihot (list[float]), bio_embedding
(list[float]), log_listeners (float)]. Artists present in the matrix but
missing from artists.jsonl (getInfo/getTopTags failed or wasn't run yet) get
zero vectors -- the tower still runs, just on weaker features for those items.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config


def load_artist_meta() -> pd.DataFrame:
    if not config.ARTISTS_PATH.exists():
        return pd.DataFrame(columns=["artist", "tags", "listeners", "bio"])
    return pd.read_json(config.ARTISTS_PATH, lines=True)


def build_tag_vocab(
    meta: pd.DataFrame, max_tags: int = config.MAX_TAGS, vocab_size: int = config.TAG_VOCAB_SIZE
) -> list[str]:
    counts: dict[str, int] = {}
    for tags in meta["tags"]:
        for t in (tags or [])[:max_tags]:
            counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:vocab_size]]


def tag_multihot(tags: list[str], vocab: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tags or []:
        idx = vocab.get(t)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def embed_bios(bios: list[str]) -> np.ndarray:
    """Sentence-transformer embeddings for artist bios. Empty bios -> zero vector."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.BIO_EMBED_MODEL)
    non_empty_idx = [i for i, b in enumerate(bios) if b and b.strip()]
    out = np.zeros((len(bios), config.BIO_EMBED_DIM), dtype=np.float32)
    if non_empty_idx:
        embs = model.encode([bios[i] for i in non_empty_idx], show_progress_bar=False)
        out[non_empty_idx] = embs
    return out


def build(embed_bio: bool = True) -> pd.DataFrame:
    artist_ids: dict[str, int] = json.loads(config.ARTIST_ID_MAP_PATH.read_text())
    meta = load_artist_meta().set_index("artist")

    vocab_list = build_tag_vocab(meta.reset_index())
    vocab = {t: i for i, t in enumerate(vocab_list)}

    names = sorted(artist_ids, key=lambda n: artist_ids[n])
    tags_col, listeners_col, bios = [], [], []
    for name in names:
        if name in meta.index:
            row = meta.loc[name]
            tags_col.append(tag_multihot(row.get("tags", []), vocab))
            listeners_col.append(float(row.get("listeners", 0) or 0))
            bios.append(row.get("bio", "") or "")
        else:
            tags_col.append(np.zeros(len(vocab), dtype=np.float32))
            listeners_col.append(0.0)
            bios.append("")

    bio_embs = embed_bios(bios) if embed_bio else np.zeros((len(names), config.BIO_EMBED_DIM), dtype=np.float32)
    log_listeners = np.log1p(np.array(listeners_col, dtype=np.float32))

    df = pd.DataFrame({
        "artist_id": [artist_ids[n] for n in names],
        "artist": names,
        "tag_multihot": [v.tolist() for v in tags_col],
        "bio_embedding": [v.tolist() for v in bio_embs],
        "log_listeners": log_listeners.tolist(),
    }).sort_values("artist_id").reset_index(drop=True)

    df.to_parquet(config.ARTIST_FEATURES_PATH, index=False)
    (config.PROCESSED_DIR / "tag_vocab.json").write_text(json.dumps(vocab_list))
    print(f"artist features: {len(df)} artists, {len(vocab_list)} tags, bio_dim={config.BIO_EMBED_DIM}")
    return df


if __name__ == "__main__":
    build()
