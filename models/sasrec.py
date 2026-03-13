from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class SasRecConfig:
    num_items: int
    d_model: int = 64
    max_len: int = 50
    n_heads: int = 2
    n_layers: int = 2
    dropout: float = 0.1

class SasRec(nn.Module):
    """
    Minimal SASRec-style transformer for next-item prediction.
    - Input: padded item id sequence (B,L), 0 is padding.
    - Output: logits over items (B,num_items), for predicting the next item.
    """

    def __init__(self, cfg: SasRecConfig):
        super().__init__()
        self.cfg = cfg

        self.item_emb = nn.Embedding(cfg.num_items + 1, cfg.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, seq: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        seq: (B,L) item ids, 0 is pad
        target: (B,) item ids in 1..num_items (optional)
        """
        bsz, seqlen = seq.shape
        if seqlen > self.cfg.max_len:
            raise ValueError(f"seq length {seqlen} exceeds max_len {self.cfg.max_len}")

        pos = torch.arange(seqlen, device=seq.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.item_emb(seq) * math.sqrt(self.cfg.d_model) + self.pos_emb(pos)
        x = self.drop(self.norm(x))

        attn_mask = torch.triu(torch.ones(seqlen, seqlen, device=seq.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = seq.eq(0)

        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)

        h_last = h[:, -1, :]

        logits = h_last @ self.item_emb.weight[1:].t()
        out = {"logits": logits, "h_last": h_last}
        if target is not None:
            loss = F.cross_entropy(logits, target.long() - 1)
            out["loss"] = loss
        return out

    @torch.no_grad()
    def user_embedding_from_seq(self, seq: torch.Tensor) -> torch.Tensor:
        return self.forward(seq)["h_last"]


