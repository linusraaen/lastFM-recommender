"""Builds a FAISS ANN index over the trained artist embeddings.

Row i of the index is artist_id i (train.py writes artist_embeddings.npy in
that order, since the feature matrix it embeds is indexed by artist_id).
IndexFlatIP is exact search -- fine at a few-artist-thousand catalogue; swap
for IVF/HNSW if the catalogue grows into the millions.
"""

from __future__ import annotations

import faiss
import numpy as np

from src import config


def build() -> faiss.Index:
    embeddings = np.load(config.ARTIST_EMBEDDINGS_PATH).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    print(f"FAISS index: {index.ntotal} artists, dim={embeddings.shape[1]} -> {config.FAISS_INDEX_PATH}")
    return index


if __name__ == "__main__":
    build()
