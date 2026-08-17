import json

import pytest
from fastapi.testclient import TestClient

from src import config


@pytest.fixture
def api_client(built_pipeline, monkeypatch):
    monkeypatch.setenv("LASTFM_API_KEY", "test-key")

    artist_ids = json.loads(config.ARTIST_ID_MAP_PATH.read_text())
    known_artists = list(artist_ids.keys())[:3]

    import src.serve.api as api_module

    def fake_top_artists(client, user, max_pages=2):
        if user == "known_user":
            return [(a, 50) for a in known_artists]
        if user == "cold_user":
            return [("Some Totally Unseen Artist", 10)]
        return []  # private/nonexistent profile

    monkeypatch.setattr(api_module, "top_artists", fake_top_artists)

    with TestClient(api_module.app) as client:
        yield client, known_artists


def test_health_check(api_client):
    client, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_recommend_known_user_excludes_their_own_history(api_client):
    client, known_artists = api_client
    resp = client.get("/recommend", params={"user": "known_user", "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cold_start"] is False
    recommended_names = {r["artist"] for r in body["recommendations"]}
    assert recommended_names.isdisjoint(known_artists)
    assert body["latency_ms"] >= 0


def test_recommend_cold_start_falls_back_to_popularity(api_client):
    client, _ = api_client
    resp = client.get("/recommend", params={"user": "cold_user", "k": 5})
    assert resp.status_code == 200
    assert resp.json()["cold_start"] is True


def test_recommend_unknown_user_returns_404(api_client):
    client, _ = api_client
    resp = client.get("/recommend", params={"user": "nonexistent_user"})
    assert resp.status_code == 404


def test_stats_reports_latency_after_requests(api_client):
    client, _ = api_client
    client.get("/recommend", params={"user": "known_user"})
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n"] >= 1
    assert body["p50_ms"] >= 0
