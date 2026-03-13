from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Ml1mSplit:
    n_users: int
    n_items: int

    # Primary data: (N, 3) arrays with columns [user_id, item_id, rating]
    train_uir: np.ndarray
    val_uir: np.ndarray
    test_uir: np.ndarray

    user_train_seqs: list[list[int]]

    user_train_items: list[set[int]]
    user_train_val_items: list[set[int]]
    user_all_items: list[set[int]]

    # ---- Convenience accessors (return (N,2) int32 views) ----

    @property
    def train_ui(self) -> np.ndarray:
        return self.train_uir[:, :2].astype(np.int32)

    @property
    def val_ui(self) -> np.ndarray:
        return self.val_uir[:, :2].astype(np.int32)

    @property
    def test_ui(self) -> np.ndarray:
        return self.test_uir[:, :2].astype(np.int32)


def fingerprint_interactions(ui: np.ndarray) -> str:
    arr = np.ascontiguousarray(ui.astype(np.int32, copy=False))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def extract_ml1m(zip_path: str | Path, out_dir: str | Path) -> Path:
    """
    Extract ml-1m.zip into out_dir, returning the extracted ml-1m directory.
    """
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "ml-1m" / "ratings.dat"
    if target.exists():
        return out_dir / "ml-1m"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return out_dir / "ml-1m"


def read_ratings(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    ratings_path = data_dir / "ratings.dat"
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.dat not found under: {data_dir}")

    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "ts"],
        encoding="latin-1",
        dtype={"user_id": np.int32, "item_id": np.int32, "rating": np.float32, "ts": np.int64},
    )


def infer_counts(data_dir: str | Path, ratings_df: pd.DataFrame) -> tuple[int, int]:
    data_dir = Path(data_dir)
    movies_path = data_dir / "movies.dat"
    if movies_path.exists():
        movies = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            header=None,
            names=["item_id", "title", "genres"],
            usecols=[0],
            encoding="latin-1",
            dtype={"item_id": np.int32},
        )
        n_items = int(movies["item_id"].max())
    else:
        n_items = int(ratings_df["item_id"].max())
    n_users = int(ratings_df["user_id"].max())
    return n_users, n_items


def split_leave_one_out(ratings_df: pd.DataFrame, n_users: int, n_items: int) -> Ml1mSplit:
    """
    Leave-one-out split by timestamp per user, preserving explicit ratings:
    - last interaction => test
    - second last => val (if exists)
    - rest => train
    Each row is (user_id, item_id, rating).
    """
    df = ratings_df.copy()
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(
        ["user_id", "ts", "_row_id"], ascending=[True, True, True], kind="mergesort"
    ).reset_index(drop=True)

    train_rows: list[tuple[int, int, float]] = []
    val_rows: list[tuple[int, int, float]] = []
    test_rows: list[tuple[int, int, float]] = []
    user_train_seqs: list[list[int]] = [[] for _ in range(n_users + 1)]
    user_train_items: list[set[int]] = [set() for _ in range(n_users + 1)]
    user_train_val_items: list[set[int]] = [set() for _ in range(n_users + 1)]
    user_all_items: list[set[int]] = [set() for _ in range(n_users + 1)]

    for uid, g in df.groupby("user_id", sort=False):
        uid = int(uid)
        items = g["item_id"].to_numpy(dtype=np.int32)
        ratings = g["rating"].to_numpy(dtype=np.float32)

        user_all_items[uid].update(int(it) for it in items.tolist())

        if len(items) == 1:
            item = int(items[0])
            train_rows.append((uid, item, float(ratings[0])))
            user_train_seqs[uid].append(item)
            user_train_items[uid].add(item)
            user_train_val_items[uid].add(item)
            continue

        # test: last interaction
        test_rows.append((uid, int(items[-1]), float(ratings[-1])))

        if len(items) >= 3:
            val_item = int(items[-2])
            val_rows.append((uid, val_item, float(ratings[-2])))
            user_train_val_items[uid].add(val_item)
            train_items = items[:-2]
            train_rats = ratings[:-2]
        else:
            train_items = items[:-1]
            train_rats = ratings[:-1]

        for it, rt in zip(train_items.tolist(), train_rats.tolist()):
            item = int(it)
            train_rows.append((uid, item, float(rt)))
            user_train_seqs[uid].append(item)
            user_train_items[uid].add(item)
            user_train_val_items[uid].add(item)

    train_uir = np.array(train_rows, dtype=np.float32)
    val_uir = np.array(val_rows, dtype=np.float32) if val_rows else np.zeros((0, 3), dtype=np.float32)
    test_uir = np.array(test_rows, dtype=np.float32)

    return Ml1mSplit(
        n_users=n_users,
        n_items=n_items,
        train_uir=train_uir,
        val_uir=val_uir,
        test_uir=test_uir,
        user_train_seqs=user_train_seqs,
        user_train_items=user_train_items,
        user_train_val_items=user_train_val_items,
        user_all_items=user_all_items,
    )


def load_ml1m(data_dir: str | Path) -> Ml1mSplit:
    df = read_ratings(data_dir)
    n_users, n_items = infer_counts(data_dir, df)
    return split_leave_one_out(df, n_users=n_users, n_items=n_items)