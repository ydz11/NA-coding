from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def ncf_embedding_size(factor: int) -> int:
    """
    NCF paper: embedding_size = layers[0] // 2, and layers[-1] = factor.
    Since layers halve each time, layers[0] = factor * (2 ^ num_hidden_layers).
    With 3 hidden layers (the paper default): layers[0] = factor * 2^?

    Actually the paper defines it more simply:
      layers = [factor*4, factor*2, factor*1]  (3 hidden layers)
      concat_dim = layers[0] = factor * 4
      embedding_size = concat_dim // 2 = factor * 2

    Examples from the paper (Table in Section 4.1):
      factor=8:  embedding=16, layers=[32,16,8]
      factor=16: embedding=32, layers=[64,32,16]
      factor=32: embedding=64, layers=[128,64,32]
      factor=64: embedding=128, layers=[256,128,64]
    """
    return factor * 2


def ncf_tower_layers(factor: int) -> tuple[int, ...]:
    """
    Generate NCF-style tower MLP hidden layers.

    The concat layer has size 2 * embedding_size = 4 * factor.
    Hidden layers halve from there down to factor:

      factor=8:  concat=32  -> hidden=(16, 8)       -> output 1
      factor=16: concat=64  -> hidden=(32, 16)      -> output 1
      factor=32: concat=128 -> hidden=(64, 32)      -> output 1
      factor=64: concat=256 -> hidden=(128, 64)     -> output 1

    Wait - the user specified:
      factor=8:  layers=[32,16,8]   -> 3 hidden layers
      factor=16: layers=[64,32,16]  -> 3 hidden layers
      factor=32: layers=[128,64,32] -> 3 hidden layers
      factor=64: layers=[256,128,64]-> 3 hidden layers

    Pattern: hidden = (factor*4, factor*2, factor)
    But the concat input is 2*embedding = 4*factor, and first hidden = 4*factor,
    so the first hidden layer doesn't reduce. Let me re-read...

    Actually NCF paper layers=[64,32,16,8] means:
      layers[0]=64 is the concat dimension (= 2*embedding_size)
      layers[1]=32, layers[2]=16, layers[3]=8 are the hidden Dense layers
      So hidden = (32, 16, 8) with 3 Dense layers

    For the user's specification:
      factor=8:  embedding=16, concat=32,  hidden=(16, 8)      [= layers=[32,16,8]]
      factor=16: embedding=32, concat=64,  hidden=(32, 16)     [= layers=[64,32,16]]
      factor=32: embedding=64, concat=128, hidden=(64, 32)     [= layers=[128,64,32]]
      factor=64: embedding=128,concat=256, hidden=(128, 64)    [= layers=[256,128,64]]

    Pattern: start from concat=4*factor, halve until factor.
    Hidden layers: (2*factor, factor)

    Hmm but user says factor=8 -> layers=[32,16,8] which is 3 layers.
    concat=32, Dense(16), Dense(8) -> that's hidden=(16,8), 2 hidden Dense layers.

    Let me just directly encode the user's specification.
    """
    # hidden layers: start at 2*factor, halve, down to factor
    # factor=8:  (16, 8)
    # factor=16: (32, 16)
    # factor=32: (64, 32)
    # factor=64: (128, 64)
    hidden = []
    h = factor * 2
    while h >= factor:
        hidden.append(h)
        h = h // 2
    return tuple(hidden)


@dataclass(frozen=True)
class NcfMlpConfig:
    n_users: int
    n_items: int
    factor: int  # predictive factor (NCF paper term)
    dropout: float = 0.1
    mlp_hidden: Optional[tuple[int, ...]] = None  # None = auto tower

    @property
    def embedding_size(self) -> int:
        return ncf_embedding_size(self.factor)

    @property
    def effective_mlp_hidden(self) -> tuple[int, ...]:
        if self.mlp_hidden is not None:
            return self.mlp_hidden
        return ncf_tower_layers(self.factor)


class NcfMlp(nn.Module):
    """
    NCF-MLP baseline for explicit rating prediction (He et al., 2017 adapted).

    Key dimensions (matching NCF paper exactly):
      factor=8:  embedding=16, concat=32,  MLP hidden=[16, 8]  -> 1
      factor=16: embedding=32, concat=64,  MLP hidden=[32, 16] -> 1
      factor=32: embedding=64, concat=128, MLP hidden=[64, 32] -> 1
      factor=64: embedding=128,concat=256, MLP hidden=[128,64] -> 1
    """

    def __init__(self, cfg: NcfMlpConfig):
        super().__init__()
        self.cfg = cfg
        emb_size = cfg.embedding_size

        # Random initialization (normal, std=0.01), same as NCF original
        self.user_table = nn.Embedding(cfg.n_users + 1, emb_size, padding_idx=0)
        self.item_table = nn.Embedding(cfg.n_items + 1, emb_size, padding_idx=0)
        nn.init.normal_(self.user_table.weight, std=0.01)
        nn.init.normal_(self.item_table.weight, std=0.01)
        self.user_table.weight.data[0].zero_()
        self.item_table.weight.data[0].zero_()

        # Tower MLP
        mlp_hidden = cfg.effective_mlp_hidden
        mlp_in = 2 * emb_size  # concat dimension
        layers: list[nn.Module] = []
        in_dim = mlp_in
        for h in mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # no sigmoid, rating regression
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        u = self.user_table(user_id.long())
        i = self.item_table(item_id.long())
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x).squeeze(-1)