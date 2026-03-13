from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def ndcg_from_rank(rank: int, k: int) -> float:
    """
    Single-positive NDCG@k used by many NCF/SASRec implementations:
    - rank is 1-based position of the ground-truth item in the sorted list (descending score).
    """
    if rank <= 0:
        raise ValueError("rank must be >= 1")
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

@dataclass(frozen=True)
class EvalResult:
    ndcg: float
    hr: float
    n_users: int

class _EvalDataset(Dataset):
    """
    Holds per-user candidate item lists:
    - items: (n_users, 1 + num_neg) with items[:,0] as the positive
    """

    def __init__(self, users: np.ndarray, items: np.ndarray):
        self.users = users.astype(np.int64)
        self.items = items.astype(np.int64)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx]

def build_eval_candidates(
    pos_ui: np.ndarray,
    n_items: int,
    user_seen_items: list[set[int]],
    num_neg: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (users, candidates) following the common SASRec/NCF eval protocol:
    For each user: 1 positive item + num_neg negatives sampled from unseen items.
    """
    rng = np.random.default_rng(seed)
    users = pos_ui[:, 0].astype(np.int64)
    pos_items = pos_ui[:, 1].astype(np.int64)
    all_items = np.arange(1, n_items + 1, dtype=np.int64)

    candidates = np.zeros((len(users), 1 + num_neg), dtype=np.int64)
    candidates[:, 0] = pos_items

    for idx, (u, pos_i) in enumerate(zip(users.tolist(), pos_items.tolist())):
        excluded = set(user_seen_items[int(u)])
        excluded.add(int(pos_i))
        available = n_items - len(excluded)
        if available < num_neg:
            raise ValueError(f"not enough unique negatives for user={u}: available={available}, requested={num_neg}")
        if not num_neg:
            continue
        mask = np.ones(n_items, dtype=bool)
        excluded_idx = np.fromiter(excluded, dtype=np.int64, count=len(excluded)) - 1
        mask[excluded_idx] = False
        pool = all_items[mask]
        candidates[idx, 1:] = rng.choice(pool, size=num_neg, replace=False)

    return users, candidates

@torch.no_grad()
def evaluate_ndcg_hr(
    model: torch.nn.Module,
    users: np.ndarray,
    candidates: np.ndarray,
    k: int,
    device: torch.device,
    batch_size: int = 4096,
) -> EvalResult:
    """
    Evaluate NDCG@k and HR@k on pre-built candidate lists.
    This matches the typical "1 pos + 100 neg" evaluation used in NCF/SASRec codebases.
    """
    model.eval()
    ds = _EvalDataset(users, candidates)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ndcgs: list[float] = []
    hits: list[float] = []

    for u, items in dl:
        bsz, n_cand = items.shape
        u = u.to(device).view(-1, 1).expand(bsz, n_cand).reshape(-1)
        items = items.to(device).reshape(-1)
        scores = model(u, items).view(bsz, n_cand)

        pos_score = scores[:, 0].unsqueeze(1)
        rank = 1 + (scores[:, 1:] > pos_score).sum(dim=1)

        hits.extend((rank <= k).float().cpu().tolist())

        for r in rank.cpu().tolist():
            ndcgs.append(ndcg_from_rank(int(r), k))

    n = len(ndcgs)
    return EvalResult(ndcg=float(np.mean(ndcgs) if n else 0.0), hr=float(np.mean(hits) if n else 0.0), n_users=n)
