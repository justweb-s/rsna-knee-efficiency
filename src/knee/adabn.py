"""Test-time recomputation of the BatchNorm statistics on the studies to predict (AdaBN), a
diagnostic that did not help and is not used by the final submission.

No labels and no gradients: before predicting, the BatchNorm layers of the encoder recompute mean
and variance on the test studies, then prediction proceeds as usual. It pays only if the test
images have a different distribution from training (scanners, protocols, sites); otherwise it is
a no-op.

Windows are sampled as in training: `k_train` windows drawn per batch with `randperm` among all
those of the study (absent slots included, as in `train._sample_windows`), batches of
`batch_size` studies, cumulative mean (`momentum=None`). So statistics recomputed on the training
studies reproduce, up to sampling noise, those accumulated during training: that is the control.

Two uses:
- `adapt_bn`: in `infer.run_submission(adabn=...)`, one pass over the test set before `predict`.
- `run_fold_diag`: the check on fold 0 of a five-fold run, from the cache. Three variants:
  `base` (the model as is, must reproduce `oof_fold<k>.csv`), `bn_train` (statistics on the
  fold's training studies: the cost of the recomputation itself) and `bn_val` (on the validation
  studies themselves: the AdaBN of the submission, on in-domain data). If `bn_val` collapses the
  code is broken.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .models.baseline import make_windows

VARIANTS = ("base", "bn_train", "bn_val")


@torch.no_grad()
def adapt_bn(models: list[nn.Module], loader, device, k: int, seed: int = 0,
             amp: bool = True, max_batches: int | None = None, log=None) -> int:
    """Recomputes the BatchNorm statistics of all `models` in a single pass over `loader`.

    The loader yields `(vol uint8, mask, ...)` (`KneeStudies` or `infer.DicomStudies`). One decoded
    batch serves every model: over the five folds of a run decoding is paid once. Same window draw
    for all models. Returns the number of batches.
    """
    bns = [m for model in models for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    for model in models:
        model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    g = torch.Generator().manual_seed(int(seed))
    on_gpu = device.type == "cuda"
    ref = models[0]
    n = 0
    try:
        for vol, *_ in loader:
            if max_batches is not None and n >= max_batches:
                break
            vol = vol.to(device, non_blocking=True)
            w = make_windows(vol.float().div_(255.0), ref.window_stride, ref.window_depth)
            pick = torch.randperm(w.shape[1], generator=g)[:k].to(device)
            w = w[:, pick]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and on_gpu):
                for model in models:
                    model.encode_windows(w)
            n += 1
            if log is not None and n % 25 == 0:
                log(f"  adabn: {n} batches")
    finally:
        for m in bns:
            m.eval()
    return n


def run_fold_diag(ckpt, cache_root, repo, fold: int = 0, recal_studies: int = 512,
                  seed: int = 0, workers: int = 4, device=None, oof_file=None,
                  delta_fold: float = 0.0030, log=print) -> tuple:
    """The check on one fold, from `fold<k>.pt` and the cache: (table, summary).

    `bn_train` uses `recal_studies` studies from the fold's training set, `bn_val` all the
    validation studies (as the submission uses the whole test set). `repo` is the root with
    `data/folds`, `data/labels` and `data/reports`.
    """
    import copy
    import time
    from pathlib import Path

    import pandas as pd
    from torch.utils.data import DataLoader

    from .aggregate import model_from_checkpoint
    from .constants import ID, TARGETS
    from .data.dataset import KneeStudies, open_cache
    from .models.baseline import window_slot_index
    from .screen import fold_auc, secondary_ruler, study_langs

    t0 = time.time()
    repo = Path(repo)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, cfg, spec = model_from_checkpoint(Path(ckpt), device)
    folds = pd.read_csv(repo / "data" / "folds" / "folds.csv").set_index(ID)["fold"]
    langs = study_langs(repo / "data" / "reports")
    labels = pd.read_csv(repo / "data" / "labels" / f"{cfg['data']['labels']}.csv", index_col=0)
    cache = open_cache(cache_root, spec)
    uids = [u for u in folds.index[folds == fold] if u in cache and u in labels.index]
    tr = [u for u in folds.index[folds != fold] if u in cache and u in labels.index]
    tr = list(np.random.default_rng(seed).permutation(tr)[:recal_studies])
    amp = bool(cfg["train"].get("amp", True))
    k = int(cfg["train"]["k_train"])
    bs = int(cfg["train"]["batch_size"])

    def loader(ids, batch_size):
        return DataLoader(KneeStudies(cache, ids), shuffle=False, batch_size=batch_size,
                          num_workers=workers, pin_memory=True)

    models = {"base": base.eval()}
    for name, ids in (("bn_train", tr), ("bn_val", uids)):
        m = copy.deepcopy(base)
        t = time.time()
        nb = adapt_bn([m], loader(ids, bs), device, k=k, seed=seed, amp=amp)
        log(f"{name}: BatchNorm recomputed on {len(ids)} studies ({nb} batches, k_train {k}) "
            f"in {time.time() - t:.0f} s")
        models[name] = m

    acc: dict[str, list[np.ndarray]] = {n: [] for n in models}
    with torch.no_grad():
        for vol, mask, *_ in loader(uids, int(cfg["train"]["eval_batch_size"])):
            vol, mask = vol.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            w = make_windows(vol.float().div_(255.0), base.window_stride, base.window_depth)
            slot = window_slot_index(vol.shape[1], vol.shape[2], device, base.window_stride,
                                     base.window_depth)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
                for n, m in models.items():
                    p = torch.sigmoid(m(None, slot, mask, encoded=m.encode_windows(w)).float())
                    acc[n].append(p.cpu().numpy())
    preds = {n: np.concatenate(v) for n, v in acc.items()}

    rows = []
    for name, p in preds.items():
        df = pd.DataFrame(p, index=uids, columns=TARGETS)
        m = fold_auc(df, labels, folds, fold)
        sec = secondary_ruler(df, labels, folds, fold, langs)
        rows.append({"variant": name, "macro_auc": m["macro_auc"],
                     "enes_macro_auc": sec["macro_auc"],
                     **{f"auc_{t}": v for t, v in m["auc_per_class"].items()}})
    table = pd.DataFrame(rows).set_index("variant")
    b0 = table.loc["base", "macro_auc"]
    table.insert(1, "delta_vs_base", table["macro_auc"] - b0)
    table.insert(2, "k_delta_fold", table["delta_vs_base"] / delta_fold)
    # Mean per-column Spearman against `base`: how much the recomputation reshuffles the ranking.
    rho = {n: float(np.mean([pd.Series(preds[n][:, j]).corr(pd.Series(preds["base"][:, j]),
                                                              method="spearman")
                             for j in range(len(TARGETS))]))
           for n in preds}
    table.insert(3, "spearman_vs_base", pd.Series(rho))
    summ = {"exp": cfg["name"], "fold": fold, "n_studies": len(uids),
            "recal_studies_train": len(tr), "k_train": k, "batch_size": bs, "seed": seed,
            "delta_fold": delta_fold,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "secs": round(time.time() - t0, 1),
            "macro_auc": table["macro_auc"].round(5).to_dict(),
            "delta_vs_base": table["delta_vs_base"].round(5).to_dict(),
            "k_delta_fold": table["k_delta_fold"].round(2).to_dict(),
            "spearman_vs_base": {n: round(v, 4) for n, v in rho.items()},
            "enes_macro_auc": table["enes_macro_auc"].round(5).to_dict()}
    if oof_file is not None and Path(oof_file).is_file():
        # The check of the check: `base` are the usual predictions.
        o = pd.read_csv(oof_file, index_col=0)[TARGETS]
        gap = float(np.abs(pd.DataFrame(preds["base"], index=uids, columns=TARGETS)
                           .loc[o.index].to_numpy() - o.to_numpy()).max())
        summ["reproduces_oof_max_gap"] = gap
        log(f"base vs {Path(oof_file).name}: max gap {gap:.2e}"
            + (" ⚠ does NOT match" if gap > 1e-3 else ""))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\n" + table[["macro_auc", "delta_vs_base", "k_delta_fold", "spearman_vs_base",
                          "enes_macro_auc"]].to_string())
    return table, summ
