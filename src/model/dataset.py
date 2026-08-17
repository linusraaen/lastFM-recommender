"""
Training examples: one per (user, positive artist) in the train split. The
target artist is leave-one-out'd from its own user's context so the model
can't just memorise "this artist is in its own context" -- it has to predict
the target from the *rest* of the user's history.

Users with fewer than 2 train interactions are dropped (no context left
after leave-one-out); they're still usable at serving time, just not for
training a leave-one-out pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config
from src.features.user_features import build_user_histories


def build_artist_feature_matrix(features_df: pd.DataFrame, num_artists: int) -> np.ndarray:
    """[num_artists, feature_dim], zero row for any artist_id missing from features_df."""
    sample = features_df.iloc[0]
    feature_dim = len(sample["tag_multihot"]) + len(sample["bio_embedding"]) + 1
    mat = np.zeros((num_artists, feature_dim), dtype=np.float32)
    for row in features_df.itertuples():
        vec = np.concatenate([row.tag_multihot, row.bio_embedding, [row.log_listeners]]).astype(np.float32)
        mat[row.artist_id] = vec
    return mat


def artist_popularity(train_matrix: pd.DataFrame, num_artists: int, eps: float = 1e-6) -> np.ndarray:
    """log Q(artist) over the train split, for the logQ sampled-softmax correction."""
    counts = np.full(num_artists, eps, dtype=np.float64)
    vc = train_matrix["artist_id"].value_counts()
    counts[vc.index.to_numpy()] += vc.to_numpy()
    probs = counts / counts.sum()
    return np.log(probs).astype(np.float32)


class TwoTowerDataset(Dataset):
    def __init__(self, train_matrix: pd.DataFrame):
        self.histories = build_user_histories(train_matrix)
        self.examples: list[tuple[str, int]] = [
            (user_id, i)
            for user_id, hist in self.histories.items()
            if len(hist) >= 2
            for i in range(len(hist))
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        user_id, i = self.examples[idx]
        hist = self.histories[user_id]
        target_id, target_conf = hist[i]
        context = hist[:i] + hist[i + 1:]
        if len(context) > config.MAX_CONTEXT_ITEMS:  # keep the most-played items, bounds batch memory
            context = sorted(context, key=lambda c: -c[1])[:config.MAX_CONTEXT_ITEMS]
        context_ids = np.array([c[0] for c in context], dtype=np.int64)
        context_weights = np.array([c[1] for c in context], dtype=np.float32)
        return context_ids, context_weights, target_id, target_conf


def collate(batch):
    max_h = max(len(b[0]) for b in batch)
    B = len(batch)
    context_ids = np.zeros((B, max_h), dtype=np.int64)
    context_weights = np.zeros((B, max_h), dtype=np.float32)
    mask = np.zeros((B, max_h), dtype=np.float32)
    target_ids = np.zeros(B, dtype=np.int64)
    target_conf = np.zeros(B, dtype=np.float32)

    for i, (cids, cw, tid, tconf) in enumerate(batch):
        h = len(cids)
        context_ids[i, :h] = cids
        context_weights[i, :h] = cw
        mask[i, :h] = 1.0
        target_ids[i] = tid
        target_conf[i] = tconf

    return (
        torch.from_numpy(context_ids),
        torch.from_numpy(context_weights),
        torch.from_numpy(mask),
        torch.from_numpy(target_ids),
        torch.from_numpy(target_conf),
    )
