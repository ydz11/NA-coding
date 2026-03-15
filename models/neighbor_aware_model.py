from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models.ncf_mlp import ncf_embedding_size, ncf_tower_layers




@dataclass(frozen=True)
class NeighborAwareConfig:
    n_users: int
    n_items: int
    factor: int  # same "factor" concept as NCF
    k: int = 5
    agg: str = "attention"
    attn_heads: int = 2
    attn_layers: int = 1
    dropout: float = 0.1
    freeze_pretrained: bool = True
    mlp_hidden: Optional[tuple[int, ...]] = None  # None = auto tower (same as NCF)

    @property
    def embedding_size(self) -> int:
        return ncf_embedding_size(self.factor)

    @property
    def effective_mlp_hidden(self) -> tuple[int, ...]:
        if self.mlp_hidden is not None:
            return self.mlp_hidden
        return ncf_tower_layers(self.factor)


class NeighborAwareModel(nn.Module):
    """
    Neighbor-aware model using concatenation (proposal version).

    target embedding + k neighbor embeddings
    """

    def __init__(
        self,
        cfg: NeighborAwareConfig,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_topk: np.ndarray,
        item_topk: np.ndarray,
    ):
        super().__init__()

        self.cfg = cfg
        self.k = cfg.k
        emb_size = cfg.embedding_size

        # ----- SASRec pretrained embeddings -----

        assert user_emb.shape[1] == emb_size
        assert item_emb.shape[1] == emb_size

        self.user_table = nn.Embedding.from_pretrained(
            user_emb,
            freeze=cfg.freeze_pretrained,
            padding_idx=0
        )

        self.item_table = nn.Embedding.from_pretrained(
            item_emb,
            freeze=cfg.freeze_pretrained,
            padding_idx=0
        )

        # ----- neighbor indices -----

        self.register_buffer(
            "user_topk",
            torch.from_numpy(user_topk.astype(np.int64)),
            persistent=False
        )

        self.register_buffer(
            "item_topk",
            torch.from_numpy(item_topk.astype(np.int64)),
            persistent=False
        )

        # ----- concatenation dimension -----

        user_dim = (self.k + 1) * emb_size
        item_dim = (self.k + 1) * emb_size

        mlp_hidden = cfg.effective_mlp_hidden
        mlp_in = user_dim + item_dim

        layers = []
        in_dim = mlp_in

        for h in mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def _gather_neighbor_ids(self, ids: torch.Tensor, topk_table: torch.Tensor) -> torch.Tensor:
        return topk_table.index_select(0, ids.long())

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:

        # ----- gather neighbors -----

        u_nei = self._gather_neighbor_ids(user_id, self.user_topk)
        i_nei = self._gather_neighbor_ids(item_id, self.item_topk)

        # target + neighbors
        u_ids = torch.cat([user_id.view(-1, 1), u_nei], dim=1)
        i_ids = torch.cat([item_id.view(-1, 1), i_nei], dim=1)

        # ----- embedding lookup -----

        u_tok = self.user_table(u_ids)
        i_tok = self.item_table(i_ids)

        # ----- concatenation -----

        u_ctx = u_tok.flatten(start_dim=1)
        i_ctx = i_tok.flatten(start_dim=1)

        x = torch.cat([u_ctx, i_ctx], dim=-1)

        return self.mlp(x).squeeze(-1)
