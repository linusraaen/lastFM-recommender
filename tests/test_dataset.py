import numpy as np
import pandas as pd

from src.model.dataset import (
    TwoTowerDataset,
    artist_popularity,
    build_artist_feature_matrix,
    build_user_histories,
    collate,
)


def _features_df():
    return pd.DataFrame({
        "artist_id": [0, 2],  # artist_id 1 is intentionally missing -> should default to zeros
        "tag_multihot": [[1.0, 0.0], [0.0, 1.0]],
        "bio_embedding": [[0.5, 0.5], [0.1, 0.1]],
        "log_listeners": [2.0, 3.0],
    })


def test_build_artist_feature_matrix_zero_fills_missing_artists():
    mat = build_artist_feature_matrix(_features_df(), num_artists=3)
    assert mat.shape == (3, 5)  # 2 tag dims + 2 bio dims + 1 listener dim
    assert np.array_equal(mat[1], np.zeros(5, dtype=np.float32))
    assert mat[0][-1] == 2.0
    assert mat[2][-1] == 3.0


def test_artist_popularity_ranks_frequent_artists_higher_logq():
    train_matrix = pd.DataFrame({"artist_id": [0, 0, 0, 1]})
    log_q = artist_popularity(train_matrix, num_artists=3)
    assert log_q[0] > log_q[1] > log_q[2]  # id 2 never appears -> lowest (epsilon-smoothed)


def test_build_user_histories_groups_by_user():
    train_matrix = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "artist_id": [0, 1, 2],
        "confidence": [1.0, 2.0, 3.0],
    })
    histories = build_user_histories(train_matrix)
    assert histories["u1"] == [(0, 1.0), (1, 2.0)]
    assert histories["u2"] == [(2, 3.0)]


def test_two_tower_dataset_drops_users_with_fewer_than_two_items():
    train_matrix = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "artist_id": [0, 1, 2],
        "confidence": [1.0, 2.0, 3.0],
    })
    ds = TwoTowerDataset(train_matrix)
    assert len(ds) == 2  # only u1's two examples; u2 has just 1 item -> no leave-one-out context


def test_two_tower_dataset_leaves_target_out_of_context():
    train_matrix = pd.DataFrame({
        "user_id": ["u1", "u1", "u1"],
        "artist_id": [0, 1, 2],
        "confidence": [1.0, 1.0, 1.0],
    })
    ds = TwoTowerDataset(train_matrix)
    for context_ids, _weights, target_id, _conf in ds:
        assert target_id not in context_ids


def test_collate_pads_and_masks_variable_length_context():
    batch = [
        (np.array([1, 2], dtype=np.int64), np.array([1.0, 1.0], dtype=np.float32), 0, 5.0),
        (np.array([3], dtype=np.int64), np.array([2.0], dtype=np.float32), 4, 1.0),
    ]
    context_ids, context_weights, mask, target_ids, target_conf = collate(batch)
    assert context_ids.shape == (2, 2)
    assert mask.tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert target_ids.tolist() == [0, 4]
    assert target_conf.tolist() == [5.0, 1.0]
