from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models.ncf_mlp import ncf_embedding_size, ncf_tower_layers


class MeanAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        if key_padding_mask is None:
            return x.mean(dim=1)
        mask = ~key_padding_mask
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


class SelfAttentionAggregator(nn.Module):
    """
    Self-attention over an unordered neighbor set.
    Token 0 is the target user/item embedding; output uses token 0 representation.
    """

    def __init__(self, d_model: int, n_heads: int = 2, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h0 = h[:, 0, :]
        return self.out_norm(h0)


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
    Proposed method for explicit rating prediction.

    Uses the same embedding_size and MLP structure as NCF baseline.
    The difference is HOW the input representation is constructed:
      - NCF: plain random-init embedding lookup
      - Ours: SASRec-pretrained embedding + cosine neighbor + self-attention

    Key dimensions (same as NCF for fair comparison):
      factor=8:  embedding=16, concat=32,  MLP hidden=[16, 8]  -> 1
      factor=16: embedding=32, concat=64,  MLP hidden=[32, 16] -> 1
      factor=32: embedding=64, concat=128, MLP hidden=[64, 32] -> 1
      factor=64: embedding=128,concat=256, MLP hidden=[128,64] -> 1
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

        # SASRec-pretrained embeddings (frozen) — proposal Step 2
        # SASRec must be pretrained with d_model = embedding_size
        assert user_emb.shape[1] == emb_size, \
            f"SASRec embedding dim {user_emb.shape[1]} != expected {emb_size} for factor={cfg.factor}"
        self.user_table = nn.Embedding.from_pretrained(user_emb, freeze=cfg.freeze_pretrained, padding_idx=0)
        self.item_table = nn.Embedding.from_pretrained(item_emb, freeze=cfg.freeze_pretrained, padding_idx=0)

        # Self-attention neighbor aggregation — proposal Step 3
        if cfg.agg == "attention":
            self.user_agg = SelfAttentionAggregator(emb_size, n_heads=cfg.attn_heads, n_layers=cfg.attn_layers,
                                                    dropout=cfg.dropout)
            self.item_agg = SelfAttentionAggregator(emb_size, n_heads=cfg.attn_heads, n_layers=cfg.attn_layers,
                                                    dropout=cfg.dropout)
        elif cfg.agg == "mean":
            self.user_agg = MeanAggregator()
            self.item_agg = MeanAggregator()
        else:
            raise ValueError(f"unknown agg: {cfg.agg}")

        # Offline neighbor indices — proposal Step 1
        self.register_buffer("user_topk", torch.from_numpy(user_topk.astype(np.int64)), persistent=False)
        self.register_buffer("item_topk", torch.from_numpy(item_topk.astype(np.int64)), persistent=False)

        # NCF-style tower MLP — proposal Step 4 (same structure as NCF baseline)
        mlp_hidden = cfg.effective_mlp_hidden
        mlp_in = 2 * emb_size
        layers: list[nn.Module] = []
        in_dim = mlp_in
        for h in mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # no sigmoid, rating regression
        self.mlp = nn.Sequential(*layers)

    def _gather_neighbor_ids(self, ids: torch.Tensor, topk_table: torch.Tensor) -> torch.Tensor:
        return topk_table.index_select(0, ids.long())

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        u_nei = self._gather_neighbor_ids(user_id, self.user_topk)
        i_nei = self._gather_neighbor_ids(item_id, self.item_topk)

        u_ids = torch.cat([user_id.view(-1, 1), u_nei], dim=1)
        i_ids = torch.cat([item_id.view(-1, 1), i_nei], dim=1)

        u_mask = u_ids.eq(0)
        i_mask = i_ids.eq(0)

        u_tok = self.user_table(u_ids)
        i_tok = self.item_table(i_ids)

        u_ctx = self.user_agg(u_tok, key_padding_mask=u_mask)
        i_ctx = self.item_agg(i_tok, key_padding_mask=i_mask)

        x = torch.cat([u_ctx, i_ctx], dim=-1)
        return self.mlp(x).squeeze(-1)