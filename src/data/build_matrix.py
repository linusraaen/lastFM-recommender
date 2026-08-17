"""
Turn the raw crawl (data/raw/*.jsonl) into a user-artist interaction matrix
with a leak-free train/test split.

Two split strategies, chosen per user:
  - Temporal (preferred): for the eval cohort that has timestamped scrobbles
    (collect.py's `--eval-every`), pick a single global cutoff T from the
    scrobble timestamp distribution. A user's train weights come only from
    their pre-T scrobble counts; the test set is the artists they listened to
    post-T that were *not* already in their pre-T history (novel discoveries
    only -- recommending something they already listen to isn't a win).
  - Leave-N-out (fallback): for users with no scrobbles, only an all-time
    top-artists list, so there's no timestamp to split on. Randomly hold out
    up to LEAVE_N_OUT of their artists as test positives.

Output (data/processed/):
  matrix.parquet        user_id, artist, artist_id, playcount, confidence, split, source
  split/train.parquet   matrix.parquet filtered to split == "train"
  split/test.parquet    user_id, artist, artist_id   (held-out positives)
  artist_ids.json       artist name -> contiguous int id
"""

from __future__ import annotations

import json
import random

import pandas as pd

from src import config


def load_interactions() -> pd.DataFrame:
    return pd.read_json(config.INTERACTIONS_PATH, lines=True)


def load_scrobbles() -> pd.DataFrame:
    if not config.SCROBBLES_PATH.exists():
        return pd.DataFrame(columns=["user_id", "artist", "track", "ts"])
    return pd.read_json(config.SCROBBLES_PATH, lines=True)


def build_artist_id_map(artists: "pd.Series[str]") -> dict[str, int]:
    names = sorted(artists.dropna().unique().tolist())
    mapping = {name: i for i, name in enumerate(names)}
    config.ARTIST_ID_MAP_PATH.write_text(json.dumps(mapping))
    return mapping


def temporal_cutoff(scrobbles: pd.DataFrame, holdout_fraction: float) -> int | None:
    """Global cutoff T: the (1 - holdout_fraction) quantile of all scrobble timestamps."""
    if scrobbles.empty:
        return None
    return int(scrobbles["ts"].quantile(1 - holdout_fraction))


def _confidence(playcount: pd.Series, alpha: float = config.CONFIDENCE_ALPHA) -> pd.Series:
    import numpy as np
    return 1 + alpha * np.log1p(playcount)


def _split_temporal_user(user_scrobbles: pd.DataFrame, cutoff: int) -> tuple[dict[str, int], set[str]]:
    """Returns (train artist -> pre-cutoff play count, test artist set of novel post-cutoff artists)."""
    pre = user_scrobbles[user_scrobbles["ts"] < cutoff]
    post = user_scrobbles[user_scrobbles["ts"] >= cutoff]
    train_counts: dict[str, int] = pre.groupby("artist").size().to_dict()
    test_artists = set(post["artist"].unique()) - set(train_counts.keys())
    return train_counts, test_artists


def _split_leave_n_out_user(
    artists: list[tuple[str, int]], n: int, rng: random.Random
) -> tuple[dict[str, int], set[str]]:
    """Holds out up to n artists (or 20% of the list, whichever is smaller) as test positives."""
    if len(artists) < 2:
        return {a: pc for a, pc in artists}, set()
    k = max(1, min(n, len(artists) // 5, len(artists) - 1))
    held = rng.sample(artists, k)
    held_names = {a for a, _ in held}
    train = {a: pc for a, pc in artists if a not in held_names}
    return train, held_names


def build() -> None:
    interactions = load_interactions()
    scrobbles = load_scrobbles()

    artist_ids = build_artist_id_map(interactions["artist"])
    cutoff = temporal_cutoff(scrobbles, config.HOLDOUT_FRACTION)
    eval_cohort = set(scrobbles["user_id"].unique()) if cutoff is not None else set()

    scrobbles_by_user: dict[str, pd.DataFrame] = (
        {u: df for u, df in scrobbles.groupby("user_id")} if cutoff is not None else {}
    )

    rows: list[dict] = []
    rng = random.Random(42)

    for user_id, group in interactions.groupby("user_id"):
        artists = list(zip(group["artist"], group["playcount"]))

        if user_id in eval_cohort:
            train_counts, test_artists = _split_temporal_user(scrobbles_by_user[user_id], cutoff)
            source = "temporal"
            if not train_counts:  # every scrobble landed post-cutoff -> fall back
                train_counts, test_artists = _split_leave_n_out_user(artists, config.LEAVE_N_OUT, rng)
                source = "leave_n_out"
        else:
            train_counts, test_artists = _split_leave_n_out_user(artists, config.LEAVE_N_OUT, rng)
            source = "leave_n_out"

        for artist, pc in train_counts.items():
            if artist not in artist_ids:  # scrobble-only artist, never a top-artist -> skip
                continue
            rows.append({
                "user_id": user_id, "artist": artist, "artist_id": artist_ids[artist],
                "playcount": pc, "split": "train", "source": source,
            })
        for artist in test_artists:
            if artist not in artist_ids:
                continue
            rows.append({
                "user_id": user_id, "artist": artist, "artist_id": artist_ids[artist],
                "playcount": 0, "split": "test", "source": source,
            })

    matrix = pd.DataFrame(rows)
    matrix.loc[matrix["split"] == "train", "confidence"] = _confidence(
        matrix.loc[matrix["split"] == "train", "playcount"]
    )
    matrix["confidence"] = matrix["confidence"].fillna(0.0)

    matrix.to_parquet(config.MATRIX_PATH, index=False)
    matrix[matrix["split"] == "train"].drop(columns=["split"]).to_parquet(
        config.SPLIT_DIR / "train.parquet", index=False
    )
    matrix[matrix["split"] == "test"][["user_id", "artist", "artist_id"]].to_parquet(
        config.SPLIT_DIR / "test.parquet", index=False
    )

    n_temporal = (matrix["source"] == "temporal").sum()
    n_lno = (matrix["source"] == "leave_n_out").sum()
    print(f"matrix: {len(matrix)} rows, {matrix['user_id'].nunique()} users, {len(artist_ids)} artists")
    print(f"split rows: temporal={n_temporal} leave_n_out={n_lno} (cutoff_ts={cutoff})")


if __name__ == "__main__":
    build()
