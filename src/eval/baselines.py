"""
Baselines the two-tower model has to beat: global popularity, tag-genre
similarity, item-item co-listen kNN, and implicit-feedback ALS.

Each baseline exposes .recommend(user_id, k) -> list[artist name], already
filtered to exclude artists the user has in their train history (the test
set is novel-discovery-only, see src/data/build_matrix.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


class _BaselineBase:
    id_to_artist: dict[int, str]
    train_artists_by_user: dict[str, set[int]]

    def _known_artists(self, user_id: str) -> set[int]:
        return self.train_artists_by_user.get(user_id, set())

    def _filter_and_take(self, user_id: str, ranked_artist_ids: np.ndarray, k: int) -> list[str]:
        known = self._known_artists(user_id)
        out = []
        for aid in ranked_artist_ids:
            if aid in known:
                continue
            out.append(self.id_to_artist[aid])
            if len(out) >= k:
                break
        return out


class PopularityBaseline(_BaselineBase):
    name = "popularity"

    def fit(self, train_matrix: pd.DataFrame, id_to_artist: dict[int, str]) -> "PopularityBaseline":
        num_artists = len(id_to_artist)
        self.id_to_artist = id_to_artist
        self.train_artists_by_user = _train_artists_by_user(train_matrix)
        pop = train_matrix.groupby("artist_id")["playcount"].sum()
        counts = np.zeros(num_artists, dtype=np.float64)
        counts[pop.index.to_numpy()] = pop.to_numpy()
        self.ranked_artist_ids = np.argsort(-counts)
        return self

    def recommend(self, user_id: str, k: int) -> list[str]:
        return self._filter_and_take(user_id, self.ranked_artist_ids, k)


class TagSimilarityBaseline(_BaselineBase):
    name = "tag_similarity"

    def fit(
        self, train_matrix: pd.DataFrame, artist_features: pd.DataFrame, id_to_artist: dict[int, str]
    ) -> "TagSimilarityBaseline":
        self.id_to_artist = id_to_artist
        self.train_artists_by_user = _train_artists_by_user(train_matrix)

        features = artist_features.sort_values("artist_id")
        self.artist_ids = features["artist_id"].to_numpy()
        tags = np.stack(features["tag_multihot"].to_numpy())
        norms = np.linalg.norm(tags, axis=1, keepdims=True)
        self.tag_matrix_normed = tags / np.clip(norms, 1e-8, None)

        self.user_confidence = {
            uid: dict(zip(g["artist_id"], g["confidence"])) for uid, g in train_matrix.groupby("user_id")
        }
        self.artist_id_to_row = {aid: i for i, aid in enumerate(self.artist_ids)}
        return self

    def _user_profile(self, user_id: str) -> np.ndarray:
        profile = np.zeros(self.tag_matrix_normed.shape[1], dtype=np.float32)
        for aid, conf in self.user_confidence.get(user_id, {}).items():
            row = self.artist_id_to_row.get(aid)
            if row is not None:
                profile += conf * self.tag_matrix_normed[row]
        n = np.linalg.norm(profile)
        return profile / n if n > 0 else profile

    def recommend(self, user_id: str, k: int) -> list[str]:
        profile = self._user_profile(user_id)
        scores = self.tag_matrix_normed @ profile
        ranked = self.artist_ids[np.argsort(-scores)]
        return self._filter_and_take(user_id, ranked, k)


class CoListenKNNBaseline(_BaselineBase):
    name = "co_listen_knn"

    def fit(self, train_matrix: pd.DataFrame, id_to_artist: dict[int, str]) -> "CoListenKNNBaseline":
        num_artists = len(id_to_artist)
        self.id_to_artist = id_to_artist
        self.train_artists_by_user = _train_artists_by_user(train_matrix)
        self.num_artists = num_artists

        user_ids = train_matrix["user_id"].astype("category")
        self.user_code_map = dict(zip(user_ids, user_ids.cat.codes))
        rows = user_ids.cat.codes.to_numpy()
        cols = train_matrix["artist_id"].to_numpy()
        vals = train_matrix["confidence"].to_numpy()
        ui = sp.csr_matrix((vals, (rows, cols)), shape=(user_ids.cat.categories.size, num_artists))

        # item-item cosine similarity via normalised co-occurrence: (A^T A) with column-normalised A
        col_norms = np.sqrt(ui.multiply(ui).sum(axis=0)).A1
        col_norms[col_norms == 0] = 1.0
        ui_normed = ui.multiply(1.0 / col_norms)
        self.item_sim = (ui_normed.T @ ui_normed).tocsr()  # [num_artists, num_artists]

        self.user_vectors = ui.tocsr()
        return self

    def recommend(self, user_id: str, k: int) -> list[str]:
        code = self.user_code_map.get(user_id)
        if code is None:
            return []
        user_row = self.user_vectors.getrow(code)  # [1, num_artists] confidence weights
        scores = (user_row @ self.item_sim).toarray().ravel()
        ranked = np.argsort(-scores)
        return self._filter_and_take(user_id, ranked, k)


class ALSBaseline(_BaselineBase):
    name = "als"

    def fit(
        self,
        train_matrix: pd.DataFrame,
        id_to_artist: dict[int, str],
        factors: int = 64,
        iterations: int = 15,
    ) -> "ALSBaseline":
        from implicit.als import AlternatingLeastSquares

        num_artists = len(id_to_artist)
        self.id_to_artist = id_to_artist
        self.train_artists_by_user = _train_artists_by_user(train_matrix)

        user_ids = train_matrix["user_id"].astype("category")
        self.user_code_map = dict(zip(user_ids, user_ids.cat.codes))
        rows = user_ids.cat.codes.to_numpy()
        cols = train_matrix["artist_id"].to_numpy()
        vals = train_matrix["confidence"].to_numpy()
        self.ui = sp.csr_matrix((vals, (rows, cols)), shape=(user_ids.cat.categories.size, num_artists))

        self.model = AlternatingLeastSquares(factors=factors, iterations=iterations, random_state=42)
        self.model.fit(self.ui)
        return self

    def recommend(self, user_id: str, k: int) -> list[str]:
        code = self.user_code_map.get(user_id)
        if code is None:
            return []
        known = self._known_artists(user_id)
        ids, _scores = self.model.recommend(
            code, self.ui.getrow(code), N=k + len(known), filter_already_liked_items=True
        )
        return [self.id_to_artist[i] for i in ids if i not in known][:k]


def _train_artists_by_user(train_matrix: pd.DataFrame) -> dict[str, set[int]]:
    return {uid: set(g["artist_id"]) for uid, g in train_matrix.groupby("user_id")}
