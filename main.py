from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import CFG, Config
from data.dataset import RatingTrainDataset
from data.movielens import Ml1mSplit, extract_ml1m, fingerprint_interactions, load_ml1m
from data.neighbor_retrieval import build_rating_csr, topk_cosine_neighbors
from models.mf_model import MfModel, MfConfig
from models.ncf_mlp import NcfMlp, NcfMlpConfig, ncf_embedding_size
from models.sasrec_ncf import SasRecNcf, SasRecNcfConfig
from models.neighbor_aware_model import NeighborAwareConfig, NeighborAwareModel
from pretrain.pretrain_sasrec import pretrain_sasrec
from utils.early_stopping import EarlyStopping
from utils.logger import ensure_dir, get_device, get_logger, save_json, set_seed
from utils.metrics import EvalResult, build_eval_candidates, evaluate_ndcg_hr


def _set_cfg(cfg: Config) -> None:
    global CFG
    CFG = cfg



@torch.no_grad()
def _load_embeddings(sasrec_dir: Path, device: torch.device):
    user_emb = torch.load(sasrec_dir / "user_emb.pt", map_location=device)
    item_emb = torch.load(sasrec_dir / "item_emb.pt", map_location=device)
    if torch.isnan(user_emb).any() or torch.isinf(user_emb).any():
        user_emb = torch.nan_to_num(user_emb, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.isnan(item_emb).any() or torch.isinf(item_emb).any():
        item_emb = torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0)
    return user_emb, item_emb


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(v):
    return json.loads(json.dumps(v, ensure_ascii=False))


def _meta_mismatch(meta, expected):
    if meta is None:
        return list(expected.keys())
    return [k for k, v in expected.items() if _jsonable(meta.get(k)) != _jsonable(v)]


def _artifact_ok(paths, meta, expected):
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        return False, f"missing {missing}"
    mm = _meta_mismatch(meta, expected)
    if mm:
        return False, f"meta mismatch: {', '.join(mm)}"
    return True, ""


def _split_meta(data):
    return {"n_users": data.n_users, "n_items": data.n_items,
            "train_ui_sha256": fingerprint_interactions(data.train_ui),
            "val_ui_sha256": fingerprint_interactions(data.val_ui)}


def _sasrec_meta(data):
    return {"n_users": data.n_users, "n_items": data.n_items, "seed": CFG.seed,
            "train_ui_sha256": fingerprint_interactions(data.train_ui), **asdict(CFG.sasrec)}


def _neighbor_meta(data):
    return {"k": CFG.neighbors.k, "block_size": CFG.neighbors.block_size,
            "n_users": data.n_users, "n_items": data.n_items,
            "train_ui_sha256": fingerprint_interactions(data.train_ui)}


def _train_meta():
    return {"batch_size": CFG.train.batch_size, "epochs": CFG.train.epochs,
            "lr": CFG.train.lr, "weight_decay": CFG.train.weight_decay,
            "loss": "mse", "seed": CFG.seed}


def _current_factor():
    return CFG.sasrec.d_model // 2


def _report_ks():
    ks, seen = [], set()
    for raw_k in [*CFG.eval.ndcg_ks, CFG.eval.ndcg_k]:
        k = int(raw_k)
        if k > 0 and k not in seen:
            seen.add(k); ks.append(k)
    return tuple(ks)



def _ensure_sasrec(data, outputs, log, allow_recompute):
    d = ensure_dir(outputs / "sasrec")
    paths = [d / "user_emb.pt", d / "item_emb.pt", d / "meta.json"]
    expected = _sasrec_meta(data)
    ok, reason = _artifact_ok(paths, _load_json(d / "meta.json"), expected)
    if not ok:
        if not allow_recompute:
            raise RuntimeError(f"SASRec stale ({reason})")
        log.info(f"pretrain sasrec ({reason}) ...")
        pretrain_sasrec(data, CFG.sasrec, out_dir=d, device=CFG.device, seed=CFG.seed)
    else:
        log.info(f"reuse sasrec: {d}")
    return d


def _ensure_neighbors(data, outputs, log, allow_recompute):
    d = ensure_dir(outputs / "neighbors")
    paths = [d / "user_topk.npy", d / "item_topk.npy", d / "meta.json"]
    expected = _neighbor_meta(data)
    ok, reason = _artifact_ok(paths, _load_json(d / "meta.json"), expected)
    if not ok:
        if not allow_recompute:
            raise RuntimeError(f"Neighbors stale ({reason})")
        log.info(f"build rating-based cosine neighbors ({reason}) ...")
        r = build_rating_csr(data.train_uir, n_users=data.n_users, n_items=data.n_items)
        user_res = topk_cosine_neighbors(r, k=CFG.neighbors.k, block_size=CFG.neighbors.block_size)
        item_res = topk_cosine_neighbors(r.T.tocsr(), k=CFG.neighbors.k, block_size=CFG.neighbors.block_size)
        np.save(d / "user_topk.npy", user_res.topk)
        np.save(d / "item_topk.npy", item_res.topk)
        save_json(d / "meta.json", expected)
    else:
        log.info(f"reuse neighbors: {d}")
    return d


# ---------------------------------------------------------------------------
#  Model builders
# ---------------------------------------------------------------------------

def _build_mf(data, factor, device):
    cfg = MfConfig(n_users=data.n_users, n_items=data.n_items,
                   embedding_size=ncf_embedding_size(factor))
    return MfModel(cfg).to(device), "MF"


def _build_ncf(data, factor, device):
    cfg = NcfMlpConfig(n_users=data.n_users, n_items=data.n_items,
                       factor=factor, dropout=CFG.train.dropout,
                       mlp_hidden=CFG.train.mlp_hidden)
    return NcfMlp(cfg).to(device), "NCF"


def _build_sasrec_ncf(data, factor, user_emb, item_emb, device):
    cfg = SasRecNcfConfig(n_users=data.n_users, n_items=data.n_items,
                          factor=factor, dropout=CFG.train.dropout,
                          freeze_pretrained=CFG.train.freeze_pretrained,
                          mlp_hidden=CFG.train.mlp_hidden)
    return SasRecNcf(cfg, user_emb=user_emb, item_emb=item_emb).to(device), "SASRec-NCF"


def _build_na(data, factor, user_emb, item_emb, neighbors_dir, agg, device):
    user_topk = np.load(neighbors_dir / "user_topk.npy")[:, :CFG.train.neighbor_k]
    item_topk = np.load(neighbors_dir / "item_topk.npy")[:, :CFG.train.neighbor_k]
    cfg = NeighborAwareConfig(
        n_users=data.n_users, n_items=data.n_items, factor=factor,
        k=CFG.train.neighbor_k, agg=agg,
        attn_heads=CFG.train.attn_heads, attn_layers=CFG.train.attn_layers,
        dropout=CFG.train.dropout, freeze_pretrained=CFG.train.freeze_pretrained,
        mlp_hidden=CFG.train.mlp_hidden)
    name = "NeighborAware" if agg == "attention" else "NA-Mean"
    return NeighborAwareModel(cfg, user_emb=user_emb, item_emb=item_emb,
                              user_topk=user_topk, item_topk=item_topk).to(device), name


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train_mse(name, model, train_dl, val_users, val_cands, device, out_dir,
              lr, weight_decay, epochs):
    log = get_logger(name)
    out_dir = ensure_dir(out_dir)
    best_path = out_dir / "best.pt"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_fn = torch.nn.MSELoss()
    es = EarlyStopping(patience=5, mode="max")
    best_val = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_dl, desc=f"{name} ep{ep}/{epochs}", leave=False)
        for u, i, r in pbar:
            u, i, r = u.to(device), i.to(device), r.to(device).float()
            loss = mse_fn(model(u, i), r)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if len(losses) % 100 == 0:
                pbar.set_postfix(loss=float(np.mean(losses[-100:])))

        val = evaluate_ndcg_hr(model=model, users=val_users, candidates=val_cands,
                               k=CFG.eval.ndcg_k, device=device, batch_size=CFG.eval.batch_size)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        log.info(f"ep{ep} mse={avg_loss:.4f} val_ndcg@{CFG.eval.ndcg_k}={val.ndcg:.5f} "
                 f"val_hr@{CFG.eval.ndcg_k}={val.hr:.5f}")

        if val.ndcg > best_val:
            best_val = val.ndcg
            torch.save(model.state_dict(), best_path)
        es.update(val.ndcg)
        if es.should_stop:
            log.info(f"early stop: best_val_ndcg={es.best:.5f}")
            break

    # reload best
    model.load_state_dict(torch.load(best_path, map_location=device))
    return best_val


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def _eval_multi_k(model, users, candidates, ks, device):
    return {k: evaluate_ndcg_hr(model=model, users=users, candidates=candidates,
                                k=k, device=device, batch_size=CFG.eval.batch_size) for k in ks}


def _metric_dict(results_by_k, ks):
    return {"ndcg": {str(k): float(results_by_k[k].ndcg) for k in ks},
            "hr": {str(k): float(results_by_k[k].hr) for k in ks}}


# ---------------------------------------------------------------------------
#  Plotting (multi-model)
# ---------------------------------------------------------------------------

COLORS = ["#4C5A92", "#2CA02C", "#FF7F0E", "#9467BD", "#C94B58"]
MARKERS = ["s", "^", "o", "v", "D"]

def plot_all(factors, results_by_model, k, out_path, dataset_name="MovieLens"):
    """
    results_by_model: dict[str, list[float]]  e.g. {"MF": [0.1, 0.12, ...], ...}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    multi = len(factors) > 1
    x_plot = np.arange(len(factors), dtype=np.float64) if multi else np.array(factors, dtype=np.float64)

    all_vals = np.concatenate([np.array(v) for v in results_by_model.values()])
    pad = max(0.01, float((all_vals.max() - all_vals.min()) * 0.5))
    y_min = max(0.0, float(np.floor((all_vals.min() - pad) * 100) / 100))
    y_max = min(1.0, float(np.ceil((all_vals.max() + pad) * 100) / 100))

    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]}):
        fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.2), dpi=200)
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")

        for idx, (name, vals) in enumerate(results_by_model.items()):
            c = COLORS[idx % len(COLORS)]
            m = MARKERS[idx % len(MARKERS)]
            ax.plot(x_plot, vals, color=c, marker=m, markersize=5, linewidth=1.3, label=name)

        ax.set_title(dataset_name, fontsize=10, fontweight="bold", pad=5)
        ax.set_xlabel("Factors", fontsize=9, fontweight="bold")
        ax.set_ylabel(f"NDCG@{k}", fontsize=9, fontweight="bold")
        if multi:
            ax.set_xticks(x_plot)
            ax.set_xticklabels([str(f) for f in factors], fontsize=8)
        else:
            ax.set_xticks(factors)
            ax.set_xticklabels([str(f) for f in factors], fontsize=8)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(labelsize=8, length=3)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="best", fontsize=7, frameon=True, fancybox=False)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig)


# ---------------------------------------------------------------------------
#  Single-factor run
# ---------------------------------------------------------------------------

def _run_single(stage):
    set_seed(CFG.seed)
    device = get_device(CFG.device)
    log = get_logger("main")
    outputs = ensure_dir(CFG.paths.output_dir)
    ks = _report_ks()
    factor = _current_factor()
    emb_size = ncf_embedding_size(factor)

    # Data
    if not CFG.paths.data_dir.exists():
        if CFG.paths.ml1m_zip is None:
            raise FileNotFoundError(f"data_dir not found: {CFG.paths.data_dir}")
        extract_ml1m(CFG.paths.ml1m_zip, CFG.paths.data_dir.parent)
    if stage == "extract":
        return None

    data = load_ml1m(CFG.paths.data_dir)
    log.info(f"data: users={data.n_users} items={data.n_items} train={len(data.train_uir)} | factor={factor} emb={emb_size}")

    if stage == "pretrain":
        _ensure_sasrec(data, outputs, log, True); return None
    if stage == "neighbors":
        _ensure_neighbors(data, outputs, log, True); return None

    recompute = stage in ("all", "train")
    sasrec_dir = _ensure_sasrec(data, outputs, log, recompute)
    neighbors_dir = _ensure_neighbors(data, outputs, log, recompute)

    val_users, val_cands = build_eval_candidates(data.val_ui, data.n_items, data.user_train_items,
                                                  CFG.eval.num_neg_eval, CFG.eval.seed)
    test_users, test_cands = build_eval_candidates(data.test_ui, data.n_items, data.user_train_val_items,
                                                    CFG.eval.num_neg_eval, CFG.eval.seed)
    user_emb, item_emb = _load_embeddings(sasrec_dir, device)

    # ----- Build all 5 models -----
    models = {}
    m, n = _build_mf(data, factor, device);                                    models[n] = m
    m, n = _build_ncf(data, factor, device);                                   models[n] = m
    m, n = _build_sasrec_ncf(data, factor, user_emb, item_emb, device);        models[n] = m
    m, n = _build_na(data, factor, user_emb, item_emb, neighbors_dir, "mean", device);      models[n] = m
    m, n = _build_na(data, factor, user_emb, item_emb, neighbors_dir, "attention", device);  models[n] = m

    for name in models:
        n_params = sum(p.numel() for p in models[name].parameters() if p.requires_grad)
        log.info(f"{name}: trainable_params={n_params:,}")

    # ----- Train -----
    train_ds = RatingTrainDataset(train_uir=data.train_uir)
    train_dl = DataLoader(train_ds, batch_size=CFG.train.batch_size, shuffle=True, num_workers=0)

    if stage in ("all", "train"):
        for name, model in models.items():
            model_dir = ensure_dir(outputs / name.lower().replace("-", "_"))
            train_mse(name=name, model=model, train_dl=train_dl,
                      val_users=val_users, val_cands=val_cands, device=device,
                      out_dir=model_dir, lr=CFG.train.lr,
                      weight_decay=CFG.train.weight_decay, epochs=CFG.train.epochs)
    elif stage == "eval":
        for name, model in models.items():
            p = outputs / name.lower().replace("-", "_") / "best.pt"
            if not p.exists():
                raise FileNotFoundError(f"{name}: {p}")
            model.load_state_dict(torch.load(p, map_location=device))

    # ----- Test -----
    all_results = {}
    for name, model in models.items():
        res = _eval_multi_k(model, test_users, test_cands, ks, device)
        all_results[name] = res
        for k in ks:
            log.info(f"[test] {name:15s} ndcg@{k}={res[k].ndcg:.5f}  hr@{k}={res[k].hr:.5f}")

    # ----- Save -----
    save_json(outputs / "results.json", {
        "factor": factor, "embedding_size": emb_size,
        "protocol": {"num_neg_eval": CFG.eval.num_neg_eval, "ks": list(ks)},
        **{name: _metric_dict(res, ks) for name, res in all_results.items()},
    })

    pk = CFG.eval.ndcg_k
    for k in ks:
        plot_all([factor], {name: [res[k].ndcg] for name, res in all_results.items()},
                 k, outputs / f"ndcg_compare_k{k}.png")

    return all_results


# ---------------------------------------------------------------------------
#  Factor sweep
# ---------------------------------------------------------------------------

def _make_factor_cfg(base, factor):
    emb = ncf_embedding_size(factor)
    return replace(base,
                   paths=replace(base.paths, output_dir=base.paths.output_dir / f"factor_{factor}"),
                   sasrec=replace(base.sasrec, d_model=emb))


def _make_seed_cfg(fcfg, seed):
    return replace(fcfg, seed=seed,
                   paths=replace(fcfg.paths, output_dir=fcfg.paths.output_dir / f"seed_{seed}"))


def run(stage):
    if stage != "sweep":
        _run_single(stage)
        return

    base_cfg = CFG
    base_out = ensure_dir(base_cfg.paths.output_dir)
    ks = _report_ks()
    pk = base_cfg.eval.ndcg_k
    model_names = ["MF", "NCF", "SASRec-NCF", "NA-Mean", "NeighborAware"]
    factor_summaries = []

    try:
        for factor in base_cfg.factor_sweep:
            fcfg = _make_factor_cfg(base_cfg, factor)
            seed_results = []
            for seed in base_cfg.seed_sweep:
                scfg = _make_seed_cfg(fcfg, seed)
                _set_cfg(scfg)
                res = _run_single("all")
                if res is None:
                    raise RuntimeError(f"sweep failed factor={factor} seed={seed}")
                seed_results.append({name: _metric_dict(res[name], ks) for name in res})

            # Average over seeds
            summary = {"factor": factor, "emb": ncf_embedding_size(factor)}
            for name in model_names:
                for metric in ("ndcg", "hr"):
                    for k in ks:
                        key = str(k)
                        vals = [sr[name][metric][key] for sr in seed_results if name in sr]
                        if vals:
                            summary.setdefault(name, {}).setdefault(f"{metric}_mean", {})[key] = float(np.mean(vals))
                            summary.setdefault(name, {}).setdefault(f"{metric}_std", {})[key] = float(np.std(vals))
            factor_summaries.append(summary)

            # Per-factor plot
            _set_cfg(fcfg)
            fout = ensure_dir(fcfg.paths.output_dir)
            for k in ks:
                key = str(k)
                plot_all([factor],
                         {n: [summary[n]["ndcg_mean"][key]] for n in model_names if n in summary},
                         k, fout / f"ndcg_compare_k{k}.png")
    finally:
        _set_cfg(base_cfg)

    # Sweep-level results
    save_json(base_out / "results.json", {"factors": list(base_cfg.factor_sweep), "per_factor": factor_summaries})
    factors = [s["factor"] for s in factor_summaries]
    for k in ks:
        key = str(k)
        by_model = {}
        for name in model_names:
            vals = [s[name]["ndcg_mean"][key] for s in factor_summaries if name in s]
            if vals:
                by_model[name] = vals
        plot_all(factors, by_model, k, base_out / f"ndcg_compare_k{k}.png")
    # Main comparison plot
    key = str(pk)
    by_model = {}
    for name in model_names:
        vals = [s[name]["ndcg_mean"][key] for s in factor_summaries if name in s]
        if vals:
            by_model[name] = vals
    plot_all(factors, by_model, pk, base_out / "ndcg_compare.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "extract", "pretrain", "neighbors", "train", "eval", "sweep"])
    args = ap.parse_args()
    run(args.stage)


if __name__ == "__main__":
    main()