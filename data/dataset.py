from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

class SasRecRandomDataset(Dataset):
    """
    Randomly sample (prefix_sequence, next_item) pairs from train sequences.
    This follows the existing project approach and is sufficient for producing embeddings.
    """

    def __init__(self, user_seqs: list[list[int]], max_len: int, n_samples: int, n_users: int, seed: int):
        self.user_seqs = user_seqs
        self.max_len = max_len
        self.n_samples = n_samples
        self.n_users = n_users
        self.rng = np.random.default_rng(seed)
        self.eligible = np.array([u for u in range(1, n_users + 1) if len(user_seqs[u]) >= 2], dtype=np.int32)
        if len(self.eligible) == 0:
            raise ValueError("no eligible users with >=2 train interactions")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        _ = idx
        u = int(self.rng.choice(self.eligible))
        seq = self.user_seqs[u]
        t = int(self.rng.integers(1, len(seq)))
        target = int(seq[t])
        prefix = seq[:t]

        prefix = prefix[-self.max_len :]
        pad_len = self.max_len - len(prefix)
        x = prefix + ([0] * pad_len)
        return np.array(x, dtype=np.int64), np.int64(target)

@dataclass(frozen=True)
class BprBatch:
    u: torch.Tensor
    pos_i: torch.Tensor
    neg_i: torch.Tensor

class BprTrainDataset(Dataset):
    """
    Pairwise training dataset (u, pos_item, neg_item) for implicit feedback.
    Negatives are sampled from items the user has NOT interacted with in train.
    """

    def __init__(self, train_ui: np.ndarray, n_items: int, user_train_items: list[set[int]], num_neg: int, seed: int):
        if train_ui.ndim != 2 or train_ui.shape[1] != 2:
            raise ValueError("train_ui must be (N,2) => user_id,item_id")
        self.u = train_ui[:, 0].astype(np.int32)
        self.i = train_ui[:, 1].astype(np.int32)
        self.n_items = int(n_items)
        self.user_train_items = user_train_items
        self.num_neg = int(num_neg)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:

        return len(self.i) * self.num_neg

    def __getitem__(self, idx: int):
        pos_idx = idx // self.num_neg
        u = int(self.u[pos_idx])
        pos_i = int(self.i[pos_idx])

        seen = self.user_train_items[u]
        while True:
            neg_i = int(self.rng.integers(1, self.n_items + 1))
            if neg_i not in seen:
                break
        return np.int64(u), np.int64(pos_i), np.int64(neg_i)

class RatingTrainDataset(Dataset):
    """
    Pointwise training dataset (u, i, rating) for explicit rating prediction.
    """

    def __init__(self, train_uir: np.ndarray):
        """
        train_uir: (N, 3) => user_id, item_id, rating
        """
        if train_uir.ndim != 2 or train_uir.shape[1] != 3:
            raise ValueError("train_uir must be (N, 3) => user_id, item_id, rating")
        self.users = train_uir[:, 0].astype(np.int64)
        self.items = train_uir[:, 1].astype(np.int64)
        self.ratings = train_uir[:, 2].astype(np.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.ratings[idx]
