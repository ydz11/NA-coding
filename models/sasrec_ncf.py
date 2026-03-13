from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.ncf_mlp import ncf_embedding_size, ncf_tower_layers


@dataclass(frozen=True)
class SasRecNcfConfig:
    n_users: int
    n_items: int
    factor: int
    dropout: float = 0.1
    freeze_pretrained: bool = True
    mlp_hidden: Optional[tuple[int, ...]] = None

    @property
    def embedding_size(self) -> int:
        return ncf_embedding_size(self.factor)

    @property
    def effective_mlp_hidden(self) -> tuple[int, ...]:
        if self.mlp_hidden is not None:
            return self.mlp_hidden
        return ncf_tower_layers(self.factor)


class SasRecNcf(nn.Module):
    """
    Ablation model: SASRec-pretrained frozen embeddings + NCF tower MLP.
    NO neighbor information. This isolates the effect of SASRec pretraining.

    Compared to:
      - NCF-MLP:      random-init emb + MLP  (no pretraining, no neighbor)
      - SASRec-NCF:   pretrained emb  + MLP  (pretraining, no neighbor)  <-- THIS
      - NA-Attention:  pretrained emb  + neighbor + attn + MLP  (full method)
    """

    def __init__(self, cfg: SasRecNcfConfig, user_emb: torch.Tensor, item_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        emb_size = cfg.embedding_size

        assert user_emb.shape[1] == emb_size, \
            f"SASRec emb dim {user_emb.shape[1]} != expected {emb_size}"

        self.user_table = nn.Embedding.from_pretrained(user_emb, freeze=cfg.freeze_pretrained, padding_idx=0)
        self.item_table = nn.Embedding.from_pretrained(item_emb, freeze=cfg.freeze_pretrained, padding_idx=0)

        mlp_hidden = cfg.effective_mlp_hidden
        mlp_in = 2 * emb_size
        layers: list[nn.Module] = []
        in_dim = mlp_in
        for h in mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        u = self.user_table(user_id.long())
        i = self.item_table(item_id.long())
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x).squeeze(-1)