"""
User representation is not a precomputed feature vector -- it's a
playcount-weighted pool of the user's artist embeddings, computed by the
user tower at train/serve time (src/model/towers.py:UserTower). This module
just builds the per-user (artist_id, confidence) history the tower pools
over, shared by the training Dataset and the live-serving path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config


def build_user_histories(train_matrix: pd.DataFrame) -> dict[str, list[tuple[int, float]]]:
    """user_id -> [(artist_id, confidence), ...] from the train split."""
    histories: dict[str, list[tuple[int, float]]] = {}
    for user_id, group in train_matrix.groupby("user_id"):
        histories[user_id] = list(zip(group["artist_id"].tolist(), group["confidence"].tolist()))
    return histories


def history_from_live_top_artists(
    top_artists: list[tuple[str, int]], artist_ids: dict[str, int]
) -> list[tuple[int, float]]:
    """Same confidence weighting as build_matrix._confidence, for a live /recommend call."""
    out = []
    for name, playcount in top_artists:
        aid = artist_ids.get(name)
        if aid is None:
            continue  # artist unseen at training time -> no embedding, can't use as context
        out.append((aid, 1.0 + config.CONFIDENCE_ALPHA * float(np.log1p(playcount))))
    return out


def load_artist_id_map() -> dict[str, int]:
    return json.loads(config.ARTIST_ID_MAP_PATH.read_text())
