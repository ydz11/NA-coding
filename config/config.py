from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Paths:
    ml1m_zip: Optional[Path] = Path("ml-1m.zip")
    data_dir: Path = Path("data") / "ml-1m"
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class SasRecPretrainConfig:
    # NOTE: d_model here = embedding_size = factor * 2
    # When factor_sweep=(8,16,32,64), d_model will be set to (16,32,64,128)
    d_model: int = 16          # factor=8 -> embedding=16, for quick test
    max_len: int = 50
    n_heads: int = 2
    n_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 20
    steps_per_epoch: int = 4000
    lr: float = 5e-4
    clip_grad: float = 1.0
    sdpa_backend: str = "math"


@dataclass(frozen=True)
class NeighborRetrievalConfig:
    k: int = 5
    block_size: int = 512


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    freeze_pretrained: bool = True

    neighbor_k: int = 5
    agg: str = "attention"
    attn_heads: int = 2
    attn_layers: int = 1
    dropout: float = 0.1

    mlp_hidden: Optional[tuple[int, ...]] = None  # None = auto tower


@dataclass(frozen=True)
class EvalConfig:
    ndcg_k: int = 10
    ndcg_ks: tuple[int, ...] = (5, 10, 20)
    num_neg_eval: int = 99       # NCF paper: 1 positive + 99 negatives
    seed: int = 42
    batch_size: int = 4096


@dataclass(frozen=True)
class Config:
    device: str = "auto"
    seed: int = 42
    factor_sweep: tuple[int, ...] = (8, 16, 32, 64)
    seed_sweep: tuple[int, ...] = (42,)
    paths: Paths = Paths()
    sasrec: SasRecPretrainConfig = SasRecPretrainConfig()
    neighbors: NeighborRetrievalConfig = NeighborRetrievalConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()


CFG = Config()