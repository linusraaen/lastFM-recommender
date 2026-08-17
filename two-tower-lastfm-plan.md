# Two-Tower Music Recommender (Last.fm) — 3-Week Build Plan

**Goal:** a finished, deployed retrieval recommender built on data you collected yourself from the Last.fm API — not a downloaded competition set. The self-collection is the differentiator: it adds real data engineering (a rate-limited, resumable crawler) that a CSV never shows, and it flips the story from *"I re-ran a 2022 competition"* to *"I built an end-to-end recommender from raw data I collected."*

**Framing for the write-up:** candidate generation / two-tower retrieval for music. Most on-target of any option for Spotify, since it's the same problem.

---

## What's new vs. a downloaded dataset

The pipeline now *starts* with collection. `collect.py` snowball-crawls the Last.fm friend graph from seed users and gathers, per user, their top artists with playcounts (the interaction signal) plus timestamped scrobbles for an eval subset, and per artist, genre tags + bio for features. It is rate-limited (~4 req/s, under Last.fm's ~5 req/s terms), resumable (visited set + BFS frontier + rows checkpointed to disk), and hashes usernames in the output. Everything downstream — towers, FAISS, serving — is unchanged.

**Ethics / terms (do this, it also reads as maturity):** the data is public and the API is sanctioned, so collecting is fine, but don't redistribute the raw crawl (the terms restrict it and it's other people's listening history). Hash usernames, and ship the repo with code + your trained embeddings, not the crawled data.

---

## Data & signal

- **Interaction:** `(user, artist, playcount)` from `user.getTopArtists`. A play is an implicit positive, and **playcount is a graded signal** — richer than a binary purchase/click. Weight positives by a log-scaled confidence `c = 1 + alpha * log(1 + playcount)` (the standard implicit-feedback trick).
- **Items = artists** to start: denser matrix, richer tag features, cleaner demo. Tracks are the stretch (bigger, sparser catalogue).
- **Item features:** genre tags (`artist.getTopTags`, multi-hot or embedded) + a text embedding of the bio (`artist.getInfo`) + a popularity feature (listeners). Tags are what make recommendations explainable ("because you listen to a lot of shoegaze").
- **User features:** Last.fm user metadata is thin/often private, so the user is represented mainly by their listening history — a playcount-weighted pool of their artists' embeddings, optionally plus a tag profile.
- **Scale:** ~3k users snowballed gives a solid collaborative signal and a catalogue of tens of thousands of artists — enough that a FAISS ANN index is justified (and it scales cleanly to track-level later).

---

## Evaluation (the part that separates this from clones)

- **Temporal split** using the scrobble timestamps: train on listens before date `T`, test on listens after. This is why the crawler collects `user.getRecentTracks` for an eval cohort — the `uts` timestamps make a leak-free split natural. If you skip scrobble collection and only have aggregate top-artists, fall back to **leave-N-out** (hold out a random subset of each user's artists, predict from the rest).
- **Metrics:** Recall@k, MAP@k, NDCG@k, and catalogue coverage (catches a model that recommends the same 20 megastar artists to everyone).
- **Baselines (mandatory):** global popularity, a tag/genre-similarity baseline, co-listen item-kNN, and implicit ALS (`implicit` library). Your two-tower must clearly beat popularity and be competitive with or better than ALS. Popularity is stubborn in music too — report honest margins.

---

## Model

Two MLP towers → L2-normalised embeddings in a shared space; affinity = dot product.

- **Item (artist) tower:** tag features + bio text embedding (sentence-transformers, precomputed) + listeners → MLP.
- **User tower:** playcount-weighted pool of the artist embeddings in the user's history (upgrade to a small attention/GRU encoder over recent scrobbles if time allows).
- **Negatives & loss:** in-batch negatives + sampled-softmax cross-entropy, with **logQ / popularity correction** so popular artists aren't over-penalised as negatives. Weight the loss by the playcount confidence.
- **Framework:** PyTorch (plugs straight into FAISS + sentence-transformers; the stronger employer signal).
- **Retrieval:** embed all artists → FAISS index (`IndexFlatIP`, or IVF/HNSW to show you understand the recall/latency trade-off) → user query vector → top-k.

---

## Serving

- **`GET /recommend?user=<lastfm_username>&k=20`** (FastAPI): live-fetch the user's top artists from the API, build their query embedding, ANN-search the artist index, return ranked artists with tags. Return p50/p99 latency.
- **Cold-start:** unknown user with few plays → fall back to popularity or a tag-profile embedding.
- **Streamlit demo:** a text box for *any Last.fm username* → recommended artists with their genre tags (and artist images from the API). A recruiter can type their own handle and watch it work on their real taste — worth more than any single metric.
- Dockerised; deployed on Hugging Face Spaces / Render / Fly.io.

---

## Repo & write-up polish

Front-load the README: demo link + results table up top, then the architecture diagram, then details. Include a "how to run" in ≤5 commands, a `Makefile`/`scripts/`, tests on the collection + eval code, and a short "what I'd do at production scale" section (feature store, distributed training, online user-embedding serving, retraining cadence). Put the repo link on your CV.

---

## Three-week timeline

**Week 1 — Collection + interaction matrix + eval harness.**
Run `collect.py` (seed with your own username + a few active users), let the crawl build to ~3k users, then assemble the `(user, artist, playcount)` matrix, the temporal split from scrobbles, and the evaluation harness (Recall@k / MAP@k / coverage) with all four baselines. *Deliverable:* a self-collected dataset + baseline numbers.

**Week 2 — The two-tower model.**
Build the towers and feature encoders, the in-batch-softmax training loop (logQ + playcount weighting), train, tune, and beat the baselines. Add tag + bio features. *Deliverable:* a trained two-tower beating popularity/ALS on the holdout, with a results table.

**Week 3 — Serving + polish.**
FAISS index, FastAPI service with the live username lookup, Docker, deployed Streamlit demo, README + architecture diagram + results, tests, latency numbers. *Deliverable:* a live demo + a clean repo.

**Stretch (only if ahead):** track-level items; a sequence encoder over recent scrobbles; a thin Bayesian layer (calibrated confidence, or Thompson-sampling exploration in serving) — your thesis edge, and a distinctive touch most portfolio recommenders lack.

---

## Definition of done

A public repo whose live demo takes a Last.fm username and returns artist recommendations, backed by a results table showing your two-tower beating popularity and ALS baselines on a temporal holdout, served behind a FastAPI/FAISS endpoint with measured latency — all on data you collected yourself.

**CV line (fill the brackets once you have numbers):**
> Built and deployed a two-tower music recommender on a self-collected Last.fm dataset (~[N] users, [M] artists via a rate-limited API crawler) — [Recall@k] vs. a popularity baseline — with tag/bio item features, a FAISS ANN index, and a Dockerised FastAPI service (live demo).

---

## Interview talking points

- Why two-tower: decoupled towers → precompute + index items → fast ANN retrieval at scale.
- Playcount as graded implicit feedback and the confidence weighting.
- Why the temporal split, and what leaks without it.
- The crawler: rate limiting, backoff, resumable BFS, and how you kept it within terms.
- Cold-start handling, and what changes at true production scale.
