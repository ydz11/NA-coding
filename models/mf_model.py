from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MfConfig:
    n_users: int
    n_items: int
    embedding_size: int


class MfModel(nn.Module):
    """
    Basic Matrix Factorization for explicit rating prediction.
    pred = dot(user_emb, item_emb) + user_bias + item_bias + global_bias
    """

    def __init__(self, cfg: MfConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.embedding_size

        self.user_emb = nn.Embedding(cfg.n_users + 1, d, padding_idx=0)
        self.item_emb = nn.Embedding(cfg.n_items + 1, d, padding_idx=0)
        self.user_bias = nn.Embedding(cfg.n_users + 1, 1, padding_idx=0)
        self.item_bias = nn.Embedding(cfg.n_items + 1, 1, padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_id.long())
        i = self.item_emb(item_id.long())
        dot = (u * i).sum(dim=-1)
        ub = self.user_bias(user_id.long()).squeeze(-1)
        ib = self.item_bias(item_id.long()).squeeze(-1)
        return dot + ub + ib + self.global_bias