"""
Runs every baseline plus the trained two-tower model against the temporal
holdout and prints a comparison table. This is the harness that produces the
README results table.
"""

from __future__ import annotations

import json

import pandas as pd

from src import config
from src.eval.baselines import ALSBaseline, CoListenKNNBaseline, PopularityBaseline, TagSimilarityBaseline
from src.eval.metrics import evaluate_recommendations


def load_eval_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    train_matrix = pd.read_parquet(config.SPLIT_DIR / "train.parquet")
    test = pd.read_parquet(config.SPLIT_DIR / "test.parquet")
    artist_ids: dict[str, int] = json.loads(config.ARTIST_ID_MAP_PATH.read_text())
    id_to_artist = {v: k for k, v in artist_ids.items()}
    return train_matrix, test, id_to_artist


def ground_truth_from_test(test: pd.DataFrame) -> dict[str, set[str]]:
    return {uid: set(g["artist"]) for uid, g in test.groupby("user_id")}


def run_recommender(recommender, users: list[str], k: int) -> dict[str, list[str]]:
    return {u: recommender.recommend(u, k) for u in users}


def evaluate_all(k: int = config.EVAL_K, include_two_tower: bool = True) -> pd.DataFrame:
    train_matrix, test, id_to_artist = load_eval_inputs()
    ground_truth = ground_truth_from_test(test)
    eval_users = [u for u in ground_truth if ground_truth[u]]
    catalog_size = len(id_to_artist)

    models = [
        PopularityBaseline().fit(train_matrix, id_to_artist),
        TagSimilarityBaseline().fit(
            train_matrix, pd.read_parquet(config.ARTIST_FEATURES_PATH), id_to_artist
        ),
        CoListenKNNBaseline().fit(train_matrix, id_to_artist),
        ALSBaseline().fit(train_matrix, id_to_artist),
    ]

    if include_two_tower and config.ITEM_TOWER_PATH.exists():
        from src.model.recommend import TwoTowerRecommender

        models.append(TwoTowerRecommender().load().fit_eval_context(train_matrix))

    rows = []
    for model in models:
        recs = run_recommender(model, eval_users, k)
        metrics = evaluate_recommendations(recs, ground_truth, k, catalog_size)
        rows.append({"model": model.name, **metrics})
        print(f"{model.name:>16}  recall@{k}={metrics['recall@k']:.4f}  "
              f"map@{k}={metrics['map@k']:.4f}  ndcg@{k}={metrics['ndcg@k']:.4f}  "
              f"coverage={metrics['coverage']:.4f}  n_users={metrics['n_users']}")

    results = pd.DataFrame(rows)
    results.to_csv(config.PROCESSED_DIR / "eval_results.csv", index=False)
    return results


if __name__ == "__main__":
    evaluate_all()
