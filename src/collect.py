"""
Last.fm data collection for the two-tower music recommender.

Snowball-crawls the Last.fm friend graph from seed users, collecting:
  - per user:   top artists with playcounts (the interaction signal)
  - per user:   timestamped scrobbles for an eval subset (enables a temporal split)
  - per artist: genre tags + bio (item-tower features)

Engineering notes (this file IS the "I collected the data myself" story):
  - One HTTP endpoint; the `method` param selects the call.
  - Rate limited to stay under Last.fm's ~5 req/s/IP terms.
  - Resumable: the visited set + BFS frontier + collected rows are checkpointed
    to disk, so you can Ctrl-C and restart without losing progress.
  - Usernames are hashed in the output; raw usernames stay only in crawl state.
  - Don't redistribute the raw crawl — it's other people's listening history.

Usage:
    export LASTFM_API_KEY=...
    python src/collect.py --seeds your_username another_active_user --max-users 3000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import requests

API_URL = "https://ws.audioscrobbler.com/2.0/"
USER_AGENT = "music-reccomender-project/0.1 (portfolio project; contact: linus_raaen@hotmail.com)"

# Resolved relative to the repo root (this file lives in src/), so the crawler
# writes to the same data/raw/ that the rest of the pipeline (src/config.py) reads.
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
STATE_PATH = OUT / "state.json"
INTERACTIONS_PATH = OUT / "interactions.jsonl"  # user_id, artist, playcount
SCROBBLES_PATH = OUT / "scrobbles.jsonl"        # user_id, artist, track, ts
ARTISTS_PATH = OUT / "artists.jsonl"            # artist, tags, listeners, bio


def _require_api_key() -> str:
    key = os.environ.get("LASTFM_API_KEY")
    if not key:
        raise SystemExit("LASTFM_API_KEY is not set. export LASTFM_API_KEY=... first.")
    return key

MIN_INTERVAL = 0.25  # seconds between requests -> ~4 req/s, safely under the cap


def uid(username: str) -> str:
    """Stable, non-reversible id so the output dataset carries no real usernames."""
    return hashlib.sha256(username.lower().encode()).hexdigest()[:16]


class LastFmClient:
    """Thin client with throttling and exponential-backoff retries."""

    def __init__(self, api_key: str | None = None, min_interval: float = MIN_INTERVAL):
        api_key = api_key or _require_api_key()
        self.api_key = api_key
        self.min_interval = min_interval
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def call(self, method: str, **params: Any) -> dict:
        params.update(method=method, api_key=self.api_key, format="json")
        for attempt in range(6):
            self._throttle()
            try:
                r = self.session.get(API_URL, params=params, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    # Last.fm returns HTTP 200 with an {"error": N} body for some failures
                    if "error" in data:
                        if data["error"] == 29:  # rate limit exceeded -> back off and retry
                            time.sleep(2 ** attempt + random.random())
                            continue
                        return {}  # invalid/private user or artist -> skip
                    return data
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt + random.random())
                    continue
                return {}
            except (requests.RequestException, ValueError):
                # ValueError covers json.JSONDecodeError -- Last.fm occasionally returns
                # HTTP 200 with an empty/malformed body under sustained load; treat it as
                # transient and retry with backoff like any other flaky response.
                time.sleep(2 ** attempt + random.random())
                continue
        return {}


def paged(client: LastFmClient, method: str, container: str, item_key: str,
          max_pages: int | None = None, **params: Any) -> Iterable[dict]:
    """Yield items across pages of a paginated Last.fm response."""
    page, total = 1, 1
    while page <= total and (max_pages is None or page <= max_pages):
        data = client.call(method, page=page, **params)
        block = data.get(container, {})
        total = int(block.get("@attr", {}).get("totalPages", 1) or 1)
        items = block.get(item_key, [])
        if isinstance(items, dict):  # single-result responses aren't wrapped in a list
            items = [items]
        yield from items
        page += 1


# --- collectors ----------------------------------------------------------------

def top_artists(client: LastFmClient, user: str, max_pages: int = 2) -> list[tuple[str, int]]:
    return [
        (a["name"], int(a.get("playcount", 0)))
        for a in paged(client, "user.getTopArtists", "topartists", "artist",
                       max_pages=max_pages, user=user, period="overall", limit=200)
    ]


def friends(client: LastFmClient, user: str, cap: int = 100) -> list[str]:
    names: list[str] = []
    for f in paged(client, "user.getFriends", "friends", "user", user=user, limit=50):
        names.append(f["name"])
        if len(names) >= cap:
            break
    return names


def recent_tracks(client: LastFmClient, user: str, max_pages: int = 20) -> list[tuple[str, str, int]]:
    """Timestamped scrobbles -> enables a clean temporal train/test split."""
    rows: list[tuple[str, str, int]] = []
    for t in paged(client, "user.getRecentTracks", "recenttracks", "track",
                   max_pages=max_pages, user=user, limit=200):
        date = t.get("date")
        if not date:  # skip the 'now playing' track, which has no timestamp
            continue
        rows.append((t["artist"].get("#text", ""), t["name"], int(date["uts"])))
    return rows


def artist_meta(client: LastFmClient, name: str) -> dict:
    tags = client.call("artist.getTopTags", artist=name).get("toptags", {}).get("tag", [])
    tag_names = [t["name"] for t in tags[:10]] if isinstance(tags, list) else []
    info = client.call("artist.getInfo", artist=name).get("artist", {})
    return {
        "artist": name,
        "tags": tag_names,
        "listeners": int(info.get("stats", {}).get("listeners", 0) or 0),
        "bio": info.get("bio", {}).get("summary", ""),
    }


# --- checkpointing --------------------------------------------------------------

def load_state() -> tuple[set[str], deque[str], set[str]]:
    if STATE_PATH.exists():
        s = json.loads(STATE_PATH.read_text())
        return set(s["visited"]), deque(s["frontier"]), set(s["artists_seen"])
    return set(), deque(), set()


def save_state(visited: set[str], frontier: deque[str], artists_seen: set[str]) -> None:
    STATE_PATH.write_text(json.dumps({
        "visited": list(visited),
        "frontier": list(frontier),
        "artists_seen": list(artists_seen),
    }))


def append(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


# --- main crawl -----------------------------------------------------------------

def crawl(seeds: list[str], max_users: int, eval_every: int = 10) -> None:
    client = LastFmClient()
    visited, frontier, artists_seen = load_state()
    for s in seeds:
        if s not in visited:
            frontier.append(s)

    while frontier and len(visited) < max_users:
        user = frontier.popleft()
        if user in visited:
            continue

        arts = top_artists(client, user)
        if not arts:  # private or empty profile -> still mark visited so we don't retry
            visited.add(user)
            continue

        u = uid(user)
        for name, pc in arts:
            append(INTERACTIONS_PATH, {"user_id": u, "artist": name, "playcount": pc})
            artists_seen.add(name)

        # timestamped history for every Nth user -> a temporal-eval cohort
        if len(visited) % eval_every == 0:
            for artist, track, ts in recent_tracks(client, user):
                append(SCROBBLES_PATH,
                       {"user_id": u, "artist": artist, "track": track, "ts": ts})

        for fr in friends(client, user):
            if fr not in visited:
                frontier.append(fr)

        visited.add(user)
        if len(visited) % 25 == 0:
            save_state(visited, frontier, artists_seen)
            print(f"users={len(visited)} frontier={len(frontier)} artists={len(artists_seen)}")

    # item features once per artist we saw (resumable: skip ones already written)
    done = set()
    if ARTISTS_PATH.exists():
        done = {json.loads(line)["artist"] for line in ARTISTS_PATH.open()}
    for name in artists_seen - done:
        append(ARTISTS_PATH, artist_meta(client, name))

    save_state(visited, frontier, artists_seen)
    print(f"done: {len(visited)} users, {len(artists_seen)} artists")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", required=True, help="seed usernames")
    p.add_argument("--max-users", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=10,
                   help="collect timestamped scrobbles for 1 in N users")
    crawl(**vars(p.parse_args()))
