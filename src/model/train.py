"""
Two-tower training: in-batch softmax with logQ popularity correction,
loss weighted by playcount confidence.

    logits[i, j] = LOGIT_SCALE * (user_embed[i] . item_embed[j]) - logQ(artist[j])
    loss[i]      = cross_entropy(logits[i], label=i)   # target j == i is the positive
    total loss   = weighted mean of loss[i] by target confidence[i]

logQ debiases the in-batch negatives: an artist that's merely popular (and so
turns up often as an accidental in-batch negative) shouldn't be penalised as
hard as one that's a genuine mismatch. LOGIT_SCALE matters more than it looks:
on a long-tailed 100K+ item catalog, logQ's dynamic range dwarfs the [-1, 1]
range of cosine similarity, so without scaling, the correction term swamps the
actual similarity signal and the model has almost nothing real to learn from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.model.dataset import TwoTowerDataset, artist_popularity, build_artist_feature_matrix, collate
from src.model.towers import ItemTower, UserTower


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train_matrix = pd.read_parquet(config.SPLIT_DIR / "train.parquet")
    features_df = pd.read_parquet(config.ARTIST_FEATURES_PATH)
    num_artists = int(features_df["artist_id"].max()) + 1
    return train_matrix, features_df, num_artists


def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    device: str | None = None,
    resume: bool = True,
) -> tuple[ItemTower, UserTower]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_matrix, features_df, num_artists = load_training_data()

    feature_matrix = torch.from_numpy(build_artist_feature_matrix(features_df, num_artists)).to(device)
    log_q = torch.from_numpy(artist_popularity(train_matrix, num_artists)).to(device)

    dataset = TwoTowerDataset(train_matrix)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    item_tower = ItemTower(feature_matrix.shape[1], config.HIDDEN_DIM, config.EMBED_DIM).to(device)
    user_tower = UserTower(config.EMBED_DIM, config.HIDDEN_DIM).to(device)
    opt = torch.optim.AdamW(
        list(item_tower.parameters()) + list(user_tower.parameters()),
        lr=lr, weight_decay=config.WEIGHT_DECAY,
    )

    start_epoch = 0
    if resume and config.TRAIN_CHECKPOINT_PATH.exists():
        # weights_only=False: this checkpoint carries optimizer state (not just tensors),
        # and it's always a file we wrote ourselves, never an untrusted third-party one.
        ckpt = torch.load(config.TRAIN_CHECKPOINT_PATH, map_location=device, weights_only=False)
        item_tower.load_state_dict(ckpt["item_tower"])
        user_tower.load_state_dict(ckpt["user_tower"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resuming from checkpoint: epoch {start_epoch}/{epochs}", flush=True)

    for epoch in range(start_epoch, epochs):
        item_tower.train()
        user_tower.train()
        total_loss, n_batches = 0.0, 0

        for context_ids, context_weights, mask, target_ids, target_conf in loader:
            context_ids, context_weights = context_ids.to(device), context_weights.to(device)
            mask, target_ids, target_conf = mask.to(device), target_ids.to(device), target_conf.to(device)

            B, H = context_ids.shape
            context_feats = feature_matrix[context_ids.reshape(-1)]
            context_embeds = item_tower(context_feats).view(B, H, -1)
            user_embeds = user_tower(context_embeds, context_weights, mask)

            target_embeds = item_tower(feature_matrix[target_ids])
            logits = (user_embeds @ target_embeds.T) * config.LOGIT_SCALE
            logits = logits - log_q[target_ids].unsqueeze(0)

            labels = torch.arange(B, device=device)
            loss_per_row = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = (loss_per_row * target_conf).sum() / target_conf.sum().clamp_min(1e-8)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch + 1}/{epochs}  loss={total_loss / max(n_batches, 1):.4f}", flush=True)

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "item_tower": item_tower.state_dict(),
            "user_tower": user_tower.state_dict(),
            "optimizer": opt.state_dict(),
        }, config.TRAIN_CHECKPOINT_PATH)

    torch.save(item_tower.state_dict(), config.ITEM_TOWER_PATH)
    torch.save(user_tower.state_dict(), config.USER_TOWER_PATH)

    item_tower.eval()
    with torch.no_grad():
        all_embeds = item_tower(feature_matrix).cpu().numpy()
    np.save(config.ARTIST_EMBEDDINGS_PATH, all_embeds)

    config.TRAIN_CHECKPOINT_PATH.unlink(missing_ok=True)  # training complete -> no longer needed
    print(f"saved towers to {config.MODELS_DIR}, artist embeddings shape={all_embeds.shape}", flush=True)
    return item_tower, user_tower


if __name__ == "__main__":
    train()
