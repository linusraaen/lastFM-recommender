import random

import pandas as pd

from src.data.build_matrix import (
    _split_leave_n_out_user,
    _split_temporal_user,
    build,
    temporal_cutoff,
)


def test_temporal_cutoff_is_none_when_no_scrobbles():
    assert temporal_cutoff(pd.DataFrame(columns=["ts"]), 0.2) is None


def test_temporal_cutoff_is_a_quantile_of_timestamps():
    scrobbles = pd.DataFrame({"ts": list(range(100))})
    cutoff = temporal_cutoff(scrobbles, holdout_fraction=0.2)
    assert 75 <= cutoff <= 85  # 80th percentile of 0..99


def test_split_temporal_user_holds_out_only_novel_post_cutoff_artists():
    scrobbles = pd.DataFrame([
        {"artist": "A", "ts": 10},
        {"artist": "A", "ts": 20},  # A: entirely pre-cutoff
        {"artist": "B", "ts": 10},
        {"artist": "B", "ts": 30},  # B: seen both before and after -> not novel
        {"artist": "C", "ts": 30},  # C: entirely post-cutoff -> novel test positive
    ])
    train_counts, test_artists = _split_temporal_user(scrobbles, cutoff=25)
    assert train_counts == {"A": 2, "B": 1}
    assert test_artists == {"C"}


def test_split_leave_n_out_holds_out_up_to_n_and_rest_is_train():
    artists = [(f"a{i}", i) for i in range(10)]
    rng = random.Random(42)
    train, held = _split_leave_n_out_user(artists, n=3, rng=rng)
    assert len(held) == 2  # min(3, 10//5, 9) == 2
    assert set(train) | held == {a for a, _ in artists}
    assert set(train).isdisjoint(held)


def test_split_leave_n_out_tiny_history_holds_out_nothing():
    train, held = _split_leave_n_out_user([("a", 1)], n=5, rng=random.Random(0))
    assert train == {"a": 1}
    assert held == set()


def test_build_end_to_end_writes_disjoint_train_test_per_user(built_pipeline):
    from src import config

    train = pd.read_parquet(config.SPLIT_DIR / "train.parquet")
    test = pd.read_parquet(config.SPLIT_DIR / "test.parquet")

    assert len(train) > 0
    assert len(test) > 0
    assert train["confidence"].min() > 0

    for user_id in set(train["user_id"]) & set(test["user_id"]):
        train_artists = set(train.loc[train["user_id"] == user_id, "artist"])
        test_artists = set(test.loc[test["user_id"] == user_id, "artist"])
        assert train_artists.isdisjoint(test_artists)
