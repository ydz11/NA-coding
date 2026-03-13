from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

@dataclass(frozen=True)
class NeighborResult:
    topk: np.ndarray

def build_rating_csr(train_uir: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    u = train_uir[:, 0].astype(np.int32)
    i = train_uir[:, 1].astype(np.int32)
    ratings = train_uir[:, 2].astype(np.float32)
    rows = u - 1
    cols = i - 1
    return sp.csr_matrix((ratings, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

def _row_normalize(mat: sp.csr_matrix, eps: float = 1e-12) -> sp.csr_matrix:
    sq = mat.multiply(mat).sum(axis=1)
    norms = np.sqrt(np.asarray(sq).reshape(-1)).astype(np.float32)
    inv = 1.0 / np.maximum(norms, eps)
    dinv = sp.diags(inv)
    return dinv @ mat

def topk_cosine_neighbors(mat: sp.csr_matrix, k: int, block_size: int = 512) -> NeighborResult:
    """
    Compute Top-k cosine neighbors for each row vector in `mat`.
    Returns 1-indexed neighbor ids with shape (N+1,k). Index 0 is reserved as padding row.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    n = mat.shape[0]
    sq = mat.multiply(mat).sum(axis=1)
    norms = np.sqrt(np.asarray(sq).reshape(-1)).astype(np.float32)
    zero_row = norms <= 1e-12
    x = _row_normalize(mat)

    topk = np.zeros((n + 1, k), dtype=np.int32)

    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        s = (x[start:end] @ x.T).toarray().astype(np.float32)

        for bi, row in enumerate(range(start, end)):
            s[bi, row] = -np.inf
            if zero_row[row]:
                s[bi, :] = -np.inf

        idx = np.argpartition(s, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(s, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)

        block_topk = idx.astype(np.int32) + 1
        for bi, row in enumerate(range(start, end)):
            if zero_row[row]:
                block_topk[bi, :] = 0
        topk[start + 1 : end + 1] = block_topk

    return NeighborResult(topk=topk)


