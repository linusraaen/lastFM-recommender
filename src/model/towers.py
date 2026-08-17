"""
Two towers sharing an L2-normalised embedding space; affinity = dot product.

- ItemTower: artist content features (tags + bio embedding + log listeners)
  -> MLP -> embedding. A pure function of features, so it embeds *any*
  artist, including ones never seen in training (cold-start items).
- UserTower: confidence-weighted mean-pool of the user's history artist
  embeddings (computed by ItemTower, so gradients flow into both towers
  jointly) -> MLP -> embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [N, feature_dim] -> [N, embed_dim], L2-normalised."""
        return F.normalize(self.net(features), dim=-1)


class UserTower(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self, context_item_embeds: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        context_item_embeds: [B, H, embed_dim] item-tower embeddings of the user's history
        weights:             [B, H] playcount confidence per history item
        mask:                [B, H] 1 for real items, 0 for padding
        -> [B, embed_dim], L2-normalised
        """
        w = weights * mask
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (context_item_embeds * w.unsqueeze(-1)).sum(dim=1) / denom
        return F.normalize(self.net(pooled), dim=-1)
