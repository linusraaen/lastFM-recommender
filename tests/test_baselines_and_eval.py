import pandas as pd

from src import config
from src.eval.baselines import ALSBaseline, CoListenKNNBaseline, PopularityBaseline, TagSimilarityBaseline
from src.eval.evaluate import evaluate_all


def _toy_train_matrix():
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "artist": ["A", "B", "A", "C", "B"],
        "artist_id": [0, 1, 0, 2, 1],
        "playcount": [100, 50, 10, 5, 20],
        "confidence": [5.0, 4.0, 3.0, 2.0, 3.5],
    })


def test_popularity_baseline_ranks_by_summed_playcount_and_excludes_known():
    train_matrix = _toy_train_matrix()
    id_to_artist = {0: "A", 1: "B", 2: "C"}
    model = PopularityBaseline().fit(train_matrix, id_to_artist)

    recs = model.recommend("u1", k=3)
    assert "A" not in recs and "B" not in recs  # u1 already knows these
    assert recs == ["C"]


def test_tag_similarity_recommends_artists_matching_user_tag_profile():
    train_matrix = pd.DataFrame({
        "user_id": ["u1"], "artist": ["A"], "artist_id": [0],
        "playcount": [10], "confidence": [3.0],
    })
    artist_features = pd.DataFrame({
        "artist_id": [0, 1, 2],
        "tag_multihot": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],  # artist 1 shares u1's tag, artist 2 doesn't
    })
    id_to_artist = {0: "A", 1: "B", 2: "C"}
    model = TagSimilarityBaseline().fit(train_matrix, artist_features, id_to_artist)

    recs = model.recommend("u1", k=2)
    assert recs[0] == "B"  # closer tag match than C
    assert "A" not in recs  # already known


def test_co_listen_knn_recommends_co_occurring_artists():
    train_matrix = _toy_train_matrix()
    id_to_artist = {0: "A", 1: "B", 2: "C"}
    model = CoListenKNNBaseline().fit(train_matrix, id_to_artist)
    recs = model.recommend("u3", k=2)  # u3 knows B; A and C both co-occur with B via u1/u2
    assert set(recs) <= {"A", "C"}


def test_als_baseline_returns_unseen_artists():
    train_matrix = _toy_train_matrix()
    id_to_artist = {0: "A", 1: "B", 2: "C"}
    model = ALSBaseline().fit(train_matrix, id_to_artist, factors=4, iterations=2)
    recs = model.recommend("u1", k=3)
    assert "A" not in recs and "B" not in recs


def test_evaluate_all_includes_two_tower_and_produces_bounded_metrics(built_pipeline):
    results = evaluate_all(k=5)
    assert "two_tower" in set(results["model"])
    assert set(results["model"]) >= {"popularity", "tag_similarity", "co_listen_knn", "als", "two_tower"}
    for col in ("recall@k", "map@k", "ndcg@k", "coverage"):
        assert results[col].between(0, 1).all()
    assert (config.PROCESSED_DIR / "eval_results.csv").exists()
