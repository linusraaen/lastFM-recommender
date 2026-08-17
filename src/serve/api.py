"""
FastAPI serving layer.

GET /recommend?user=<lastfm_username>&k=20
  1. Live-fetch the user's top artists from the Last.fm API (so this works
     for *any* username, not just ones in the crawl).
  2. Map to (artist_id, confidence) history against the training catalogue.
  3. Embed via the user tower -> ANN search the FAISS artist index.
  4. Cold start (unknown user, private profile, or every top artist is
     outside the training catalogue): fall back to the popularity baseline.

GET /health    liveness probe
GET /stats     rolling p50/p99 latency for /recommend
"""

from __future__ import annotations

import json
import time
from collections import deque
from contextlib import asynccontextmanager

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src import config
from src.collect import LastFmClient, top_artists
from src.eval.baselines import PopularityBaseline
from src.features.user_features import history_from_live_top_artists, load_artist_id_map
from src.model.recommend import TwoTowerRecommender

_LATENCY_WINDOW = deque(maxlen=500)
_state: dict = {}


class Recommendation(BaseModel):
    artist: str
    tags: list[str]


class RecommendResponse(BaseModel):
    user: str
    cold_start: bool
    recommendations: list[Recommendation]
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["artist_ids"] = load_artist_id_map()
    _state["recommender"] = TwoTowerRecommender().load()
    _state["faiss_index"] = (
        faiss.read_index(str(config.FAISS_INDEX_PATH)) if config.FAISS_INDEX_PATH.exists() else None
    )
    _state["lastfm_client"] = LastFmClient()

    train_matrix = pd.read_parquet(config.SPLIT_DIR / "train.parquet")
    id_to_artist = {v: k for k, v in _state["artist_ids"].items()}
    _state["popularity"] = PopularityBaseline().fit(train_matrix, id_to_artist)

    tags_by_artist: dict[str, list[str]] = {}
    if config.ARTISTS_PATH.exists():
        for line in config.ARTISTS_PATH.open():
            row = json.loads(line)
            tags_by_artist[row["artist"]] = (row.get("tags") or [])[:5]
    _state["tags_by_artist"] = tags_by_artist

    yield
    _state.clear()


app = FastAPI(title="two-tower-lastfm", lifespan=lifespan)


def _with_tags(artists: list[str]) -> list[Recommendation]:
    return [Recommendation(artist=a, tags=_state["tags_by_artist"].get(a, [])) for a in artists]


def _ann_search(query: np.ndarray, k: int, exclude_ids: set[int]) -> list[str]:
    recommender: TwoTowerRecommender = _state["recommender"]
    index = _state["faiss_index"]
    if index is None:
        return recommender.recommend_from_query(query, k, exclude_ids)

    search_k = k + len(exclude_ids)
    _scores, ids = index.search(query.reshape(1, -1).astype("float32"), search_k)
    id_to_artist = recommender.id_to_artist
    out = []
    for aid in ids[0]:
        if aid < 0 or aid in exclude_ids:
            continue
        out.append(id_to_artist[int(aid)])
        if len(out) >= k:
            break
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    if not _LATENCY_WINDOW:
        return {"p50_ms": None, "p99_ms": None, "n": 0}
    arr = np.array(_LATENCY_WINDOW)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "n": len(arr),
    }


@app.get("/recommend", response_model=RecommendResponse)
def recommend(user: str = Query(..., min_length=1), k: int = Query(20, ge=1, le=100)) -> RecommendResponse:
    start = time.perf_counter()

    live_top_artists = top_artists(_state["lastfm_client"], user, max_pages=2)
    if not live_top_artists:
        raise HTTPException(status_code=404, detail=f"no public listening history for user '{user}'")

    history = history_from_live_top_artists(live_top_artists, _state["artist_ids"])
    known_ids = {aid for aid, _ in history}

    if history:
        query = _state["recommender"].embed_query(history)
        recs = _ann_search(query, k, known_ids)
        cold_start = False
    else:
        recs = _state["popularity"].recommend(user_id="__unknown__", k=k)
        cold_start = True

    latency_ms = (time.perf_counter() - start) * 1000
    _LATENCY_WINDOW.append(latency_ms)

    return RecommendResponse(user=user, cold_start=cold_start, recommendations=_with_tags(recs), latency_ms=latency_ms)
