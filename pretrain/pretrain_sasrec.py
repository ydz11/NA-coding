from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import SasRecPretrainConfig
from data.dataset import SasRecRandomDataset
from data.movielens import Ml1mSplit, fingerprint_interactions
from models.sasrec import SasRec, SasRecConfig
from utils.logger import ensure_dir, get_device, save_json

def _configure_sdpa(backend: str) -> None:
    if backend != "math":
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[sasrec] SDPA: force math (flash=off, mem_efficient=off, math=on)")
    except Exception as e:
        print(f"[sasrec][warn] SDPA backend config failed: {e} (continue with default)")

def pretrain_sasrec(data: Ml1mSplit, cfg: SasRecPretrainConfig, out_dir: str | Path, device: str = "auto", seed: int = 42) -> dict[str, Path]:
    """
    Pretrain SASRec to produce user/item embeddings.
    Outputs:
      - user_emb.pt (n_users+1, d_model)
      - item_emb.pt (n_items+1, d_model)
      - meta.json
    """
    device_t = get_device(device)
    _configure_sdpa(cfg.sdpa_backend)
    out_dir = ensure_dir(out_dir)

    model_cfg = SasRecConfig(
        num_items=data.n_items,
        d_model=cfg.d_model,
        max_len=cfg.max_len,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    model = SasRec(model_cfg).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ds = SasRecRandomDataset(
        user_seqs=data.user_train_seqs,
        max_len=model_cfg.max_len,
        n_samples=cfg.steps_per_epoch * cfg.batch_size,
        n_users=data.n_users,
        seed=seed,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=(device_t.type == "cuda"))

    model.train()
    for ep in range(1, cfg.epochs + 1):
        pbar = tqdm(dl, desc=f"sasrec ep{ep}/{cfg.epochs}", leave=False)
        losses = []
        for x, y in pbar:
            x = x.to(device_t)
            y = y.to(device_t)
            out = model(x, target=y)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"SASRec loss is not finite: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad)
            opt.step()
            losses.append(loss.item())
            if len(losses) % 50 == 0:
                pbar.set_postfix(loss=float(np.mean(losses[-50:])))
        print(f"[sasrec] epoch {ep} loss={float(np.mean(losses)):.4f}")

    model.eval()
    user_emb = torch.zeros((data.n_users + 1, model_cfg.d_model), dtype=torch.float32, device=device_t)
    for uid in range(1, data.n_users + 1):
        seq = data.user_train_seqs[uid]
        if len(seq) == 0:
            continue
        seq = seq[-model_cfg.max_len :]
        pad_len = model_cfg.max_len - len(seq)
        x = torch.tensor(([0] * pad_len) + seq, dtype=torch.int64, device=device_t).view(1, -1)
        user_emb[uid] = model.user_embedding_from_seq(x).squeeze(0)

    item_emb = model.item_emb.weight.detach().clone().to(device_t)

    if torch.isnan(user_emb).any() or torch.isinf(user_emb).any():
        raise RuntimeError(
            "user_emb contains NaN/Inf. Try lowering lr, increasing clip_grad, "
            "or reducing steps_per_epoch/epochs, then rerun pretraining."
        )

    user_path = out_dir / "user_emb.pt"
    item_path = out_dir / "item_emb.pt"
    torch.save(user_emb.cpu(), user_path)
    torch.save(item_emb.cpu(), item_path)
    save_json(
        out_dir / "meta.json",
        {
            "n_users": data.n_users,
            "n_items": data.n_items,
            "seed": seed,
            "train_ui_sha256": fingerprint_interactions(data.train_ui),
            **asdict(cfg),
            "device": str(device_t),
        },
    )
    return {"user_emb": user_path, "item_emb": item_path}

def main() -> None:
    from config.config import CFG
    from data.movielens import extract_ml1m, load_ml1m
    from utils.logger import set_seed

    set_seed(CFG.seed)
    if CFG.paths.ml1m_zip is not None:
        extract_ml1m(CFG.paths.ml1m_zip, Path("data"))
    data = load_ml1m(CFG.paths.data_dir)
    out_dir = CFG.paths.output_dir / "sasrec"
    pretrain_sasrec(data, CFG.sasrec, out_dir=out_dir, device=CFG.device, seed=CFG.seed)
    print(f"[ok] saved: {out_dir}")

if __name__ == "__main__":
    main()
