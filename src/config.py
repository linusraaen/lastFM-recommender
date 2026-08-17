"""Central paths and hyperparameters shared across the pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

INTERACTIONS_PATH = RAW_DIR / "interactions.jsonl"
SCROBBLES_PATH = RAW_DIR / "scrobbles.jsonl"
ARTISTS_PATH = RAW_DIR / "artists.jsonl"

MATRIX_PATH = PROCESSED_DIR / "matrix.parquet"          # user_id, artist_id, playcount, confidence
SPLIT_DIR = PROCESSED_DIR / "split"                      # train.parquet / test.parquet
ARTIST_ID_MAP_PATH = PROCESSED_DIR / "artist_ids.json"   # artist name -> artist_id
ARTIST_FEATURES_PATH = PROCESSED_DIR / "artist_features.parquet"

ITEM_TOWER_PATH = MODELS_DIR / "item_tower.pt"
USER_TOWER_PATH = MODELS_DIR / "user_tower.pt"
TRAIN_CHECKPOINT_PATH = MODELS_DIR / "train_checkpoint.pt"  # per-epoch resume state
ARTIST_EMBEDDINGS_PATH = MODELS_DIR / "artist_embeddings.npy"
FAISS_INDEX_PATH = MODELS_DIR / "artists.faiss"

# --- interaction weighting -------------------------------------------------
CONFIDENCE_ALPHA = 1.0  # c = 1 + alpha * log(1 + playcount)

# --- temporal split ----------------------------------------------------------
# scrobbles are collected for only a subset of users (see collect.py eval-every);
# users without any scrobbles fall back to a leave-N-out split on their top artists.
HOLDOUT_FRACTION = 0.2
LEAVE_N_OUT = 5

# --- features ------------------------------------------------------------------
MAX_TAGS = 10
# Last.fm tags are a user-generated folksonomy -- thousands of near-unique tags
# in a real crawl. Cap the *vocabulary* (not just tags-per-artist) to the most
# frequent ones so the multi-hot feature stays a genre-level signal, not an
# artist-cardinality-sized one-hot.
TAG_VOCAB_SIZE = 300
BIO_EMBED_MODEL = "all-MiniLM-L6-v2"
BIO_EMBED_DIM = 384

# --- model -----------------------------------------------------------------
EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 512
# Cosine similarity of L2-normalised embeddings is bounded to [-1, 1], but the
# logQ correction (see train.py) can span 15-20+ log-units on a long-tailed,
# 100K+ item catalog. Unscaled, the correction swamps the actual similarity
# signal and the model has almost nothing real to learn from. Scale the
# similarity logits up so they're comparable in magnitude before logQ is
# subtracted -- standard practice in contrastive/retrieval training (e.g. CLIP).
LOGIT_SCALE = 20.0
# Defensive cap on a single training example's context length (keep the
# highest-confidence items). Bounds per-batch memory even for a user with a
# very long history, independent of feature width.
MAX_CONTEXT_ITEMS = 300
LEARNING_RATE = 3e-4
EPOCHS = 20
WEIGHT_DECAY = 1e-5

# --- eval --------------------------------------------------------------------
EVAL_K = 20

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, SPLIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
