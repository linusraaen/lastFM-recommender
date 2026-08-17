"""
Shared two-tower inference path: build a user query embedding from a
playcount-weighted history, then rank the artist catalog by dot product.
Used by both the eval harness (src/eval/evaluate.py, brute-force ranking is
plenty fast at this catalog size) and live serving (src/serve/api.py, which
swaps the brute-force step for a FAISS ANN lookup).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src import config
from src.model.dataset import build_artist_feature_matrix
from src.model.towers import ItemTower, UserTower


class TwoTowerRecommender:
    name = "two_tower"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.id_to_artist: dict[int, str] = {}
        self.train_artists_by_user: dict[str, set[int]] = {}
        self._histories: dict[str, list[tuple[int, float]]] = {}

    def load(self) -> "TwoTowerRecommender":
        artist_ids: dict[str, int] = json.loads(config.ARTIST_ID_MAP_PATH.read_text())
        self.id_to_artist = {v: k for k, v in artist_ids.items()}
        num_artists = len(artist_ids)

        features_df = pd.read_parquet(config.ARTIST_FEATURES_PATH)
        feature_matrix = build_artist_feature_matrix(features_df, num_artists)
        self.feature_matrix = torch.from_numpy(feature_matrix).to(self.device)

        self.item_tower = ItemTower(feature_matrix.shape[1], config.HIDDEN_DIM, config.EMBED_DIM).to(self.device)
        self.item_tower.load_state_dict(torch.load(config.ITEM_TOWER_PATH, map_location=self.device))
        self.item_tower.eval()

        self.user_tower = UserTower(config.EMBED_DIM, config.HIDDEN_DIM).to(self.device)
        self.user_tower.load_state_dict(torch.load(config.USER_TOWER_PATH, map_location=self.device))
        self.user_tower.eval()

        if config.ARTIST_EMBEDDINGS_PATH.exists():
            self.artist_embeddings = np.load(config.ARTIST_EMBEDDINGS_PATH)
        else:
            with torch.no_grad():
                self.artist_embeddings = self.item_tower(self.feature_matrix).cpu().numpy()
        return self

    def fit_eval_context(self, train_matrix: pd.DataFrame) -> "TwoTowerRecommender":
        """Wires up known-user histories so .recommend(user_id, k) works for the eval harness."""
        self.train_artists_by_user = {
            uid: set(g["artist_id"]) for uid, g in train_matrix.groupby("user_id")
        }
        self._histories = {
            uid: list(zip(g["artist_id"].tolist(), g["confidence"].tolist()))
            for uid, g in train_matrix.groupby("user_id")
        }
        return self

    @torch.no_grad()
    def embed_query(self, history: list[tuple[int, float]]) -> np.ndarray:
        if not history:
            return np.zeros(config.EMBED_DIM, dtype=np.float32)
        ids = torch.tensor([[a for a, _ in history]], dtype=torch.long, device=self.device)
        weights = torch.tensor([[w for _, w in history]], dtype=torch.float32, device=self.device)
        mask = torch.ones_like(weights)
        context_embeds = self.item_tower(self.feature_matrix[ids.view(-1)]).view(1, ids.shape[1], -1)
        query = self.user_tower(context_embeds, weights, mask)
        return query.cpu().numpy().ravel()

    def recommend_from_query(self, query: np.ndarray, k: int, exclude: set[int]) -> list[str]:
        scores = self.artist_embeddings @ query
        ranked = np.argsort(-scores)
        out = []
        for aid in ranked:
            if aid in exclude:
                continue
            out.append(self.id_to_artist[int(aid)])
            if len(out) >= k:
                break
        return out

    def recommend(self, user_id: str, k: int) -> list[str]:
        history = self._histories.get(user_id, [])
        query = self.embed_query(history)
        known = self.train_artists_by_user.get(user_id, set())
        return self.recommend_from_query(query, k, known)
