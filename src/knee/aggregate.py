"""Window aggregators at inference: a diagnostic for models trained with a per-window loss.

With a per-window BCE (every window answers for the study target) the study can be read as the
mean of the window logits, with max, mean of probabilities or top-k. The question is answered
without training anything: the per-window logits of a checkpoint (`window_logits`) are saved once
and aggregated every way on the CPU.

Two parts. `dump_window_logits` runs on the GPU and produces `(N, W, T)` plus the mask `(N, W)`
of valid windows. `AGGREGATORS` and `aggregate_table` are pure numpy: they take those two arrays
and the teacher labels and return one row per aggregator with the macro AUC and the twelve
columns. `mean_logit` matches the forward of the model with `pooling: mean` to the decimal, so its
row must reproduce the screening number: it is the control of the diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import ID, TARGETS
from .metrics import evaluate
from .models.baseline import make_windows, window_slot_index


def _valid3(valid: np.ndarray) -> np.ndarray:
    return np.asarray(valid, dtype=bool)[:, :, None]


def _n_valid(valid: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(valid, dtype=bool).sum(1), 1)[:, None].astype(np.float64)


def agg_mean_logit(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean of the logits over the valid windows: the forward of `pooling: mean`."""
    v = _valid3(valid)
    return (np.where(v, z, 0.0)).sum(1) / _n_valid(valid)


def agg_mean_prob(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean of the probabilities (sigmoid before the mean)."""
    v = _valid3(valid)
    p = 1.0 / (1.0 + np.exp(-z))
    return (np.where(v, p, 0.0)).sum(1) / _n_valid(valid)


def agg_max(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Maximum over the valid windows: the most confident window decides."""
    return np.where(_valid3(valid), z, -np.inf).max(1)


def agg_topk(k: int):
    """Mean of the `k` highest logits among the valid windows (fewer than `k` valid: all of them)."""
    def f(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
        zz = np.where(_valid3(valid), z, -np.inf)
        zs = -np.sort(-zz, axis=1)                          # decreasing along W
        n = np.minimum(np.asarray(valid, dtype=bool).sum(1), k)   # (N,)
        n = np.maximum(n, 1)
        cs = np.cumsum(np.where(np.isfinite(zs), zs, 0.0), axis=1)   # (N, W, T)
        idx = (n - 1)[:, None, None]
        top = np.take_along_axis(cs, np.broadcast_to(idx, (z.shape[0], 1, z.shape[2])), axis=1)[:, 0]
        return top / n[:, None]
    f.__name__ = f"top{k}"
    return f


def agg_lse(tau: float):
    """Log-sum-exp at temperature `tau`: tau * log(mean_valid exp(z / tau)).
    At tau -> 0 it is the maximum, at tau -> inf the mean of the logits.
    """
    def f(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
        v = _valid3(valid)
        zz = np.where(v, z / tau, -np.inf)
        m = zz.max(1, keepdims=True)                         # (N, 1, T), finite if there is a valid window
        m = np.where(np.isfinite(m), m, 0.0)
        s = (np.where(v, np.exp(zz - m), 0.0)).sum(1)
        return tau * (m[:, 0] + np.log(np.maximum(s, 1e-300)) - np.log(_n_valid(valid)))
    f.__name__ = f"lse_tau{tau:g}"
    return f


AGGREGATORS = {
    "mean_logit": agg_mean_logit,
    "mean_prob": agg_mean_prob,
    "max": agg_max,
    "top3": agg_topk(3),
    "top5": agg_topk(5),
    "lse_tau1": agg_lse(1.0),
    "lse_tau0.3": agg_lse(0.3),
}


def teacher_targets(labels: pd.DataFrame, uids: list[str]) -> pd.DataFrame:
    """The teacher labels binarised as in `train.run` (> 0.5, NaN where missing)."""
    y = (labels.loc[uids, TARGETS] > 0.5).astype(float)
    y[labels.loc[uids, TARGETS].isna()] = np.nan
    return y


def aggregate_table(z: np.ndarray, valid: np.ndarray, uids: list[str],
                    labels: pd.DataFrame) -> pd.DataFrame:
    """One row per aggregator: `macro_auc` and the twelve columns, against the teacher."""
    z = np.asarray(z, dtype=np.float64)
    if z.shape[0] != len(uids) or z.shape[2] != len(TARGETS):
        raise ValueError(f"logits {z.shape} vs {len(uids)} studies and {len(TARGETS)} targets")
    y = teacher_targets(labels, uids)
    rows = {}
    for name, fn in AGGREGATORS.items():
        pred = pd.DataFrame(fn(z, valid), index=uids, columns=TARGETS)
        m = evaluate(y, pred)
        rows[name] = {"macro_auc": m["macro_auc"], **m["auc_per_class"]}
    return pd.DataFrame.from_dict(rows, orient="index")


@torch.no_grad()
def dump_window_logits(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-window logits `(N, W, T)` and the mask of valid windows `(N, W)`, over all windows of
    every study (as in `train.predict`).
    """
    model.eval()
    zs, vs = [], []
    for vol, mask, _, _ in loader:
        vol, mask = vol.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        w = make_windows(vol.float().div_(255.0), model.window_stride, model.window_depth)
        slot_of = window_slot_index(vol.shape[1], vol.shape[2], device, model.window_stride,
                                    model.window_depth)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            z, valid = model.window_logits(w, slot_of, mask)
        zs.append(z.float().cpu().numpy().astype(np.float32))
        vs.append(valid.cpu().numpy())
    return np.concatenate(zs), np.concatenate(vs)


def model_from_checkpoint(path, device) -> tuple:
    """Model, config and `CacheSpec` from a `train.run` checkpoint.

    The same constructor as `infer.load_ensemble`, without going through `infer`: that module
    imports the DICOM decoders (`pydicom`), which are not needed here. The config is read from the
    checkpoint: the cache geometry is part of the model.
    """
    from .data.cache import CacheSpec
    from .models.baseline import GROUP, KneeBaseline

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    spec = CacheSpec(**dict(cfg["cache"], band=tuple(cfg["cache"]["band"])))
    m = KneeBaseline(arch=cfg["model"]["arch"], n_slots=spec.shape[0], n_slices=spec.slices,
                     pretrained=False, drop_rate=0.0, img_size=spec.img,
                     slot_prior_strength=cfg["model"].get("slot_prior_strength", 0.0),
                     slot_prior_mask_at=cfg["model"].get("slot_prior_mask_at"),
                     window_stride=cfg["model"].get("window_stride", 1),
                     window_depth=cfg["model"].get("window_depth", GROUP),
                     context_layers=cfg["model"].get("context_layers", 0),
                     context_dim=cfg["model"].get("context_dim", 256),
                     context_heads=cfg["model"].get("context_heads", 4),
                     context_pos=cfg["model"].get("context_pos"),
                     pooling=cfg["model"].get("pooling", "attention"),
                     attn_hidden=cfg["model"].get("attn_hidden", 128),
                     spatial_attn=cfg["model"].get("spatial_attn"),
                     encoder_scope=cfg["model"].get("encoder_scope", "shared"),
                     spatial_pool=cfg["model"].get("spatial_pool", "avg"),
slice_attn_layers=cfg["model"].get("slice_attn_layers", 0),
                     slice_attn_dim=cfg["model"].get("slice_attn_dim", 256),
                     slice_attn_heads=cfg["model"].get("slice_attn_heads", 4),
                                          input_norm=cfg["model"].get("input_norm"),
                     compartment_route=cfg["model"].get("compartment_route"),
                     finding_layers=cfg["model"].get("finding_layers", 0),
                     finding_dim=cfg["model"].get("finding_dim", 256),
                     finding_heads=cfg["model"].get("finding_heads", 4),
                     vit_head=cfg["model"].get("vit_head"),
                     img_crop=cfg["model"].get("img_crop"))
    m.load_state_dict(ck["model"])
    return m.eval().to(device), cfg, spec


# ---------------------------------------------------------------- command line

def _dump(a) -> None:
    from torch.utils.data import DataLoader

    from .data.dataset import KneeStudies, open_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(Path(a.ckpt_dir).glob("fold*.pt"))
    if len(paths) != 1:
        raise SystemExit(f"{a.ckpt_dir}: expected a single checkpoint (fold0.pt), found {len(paths)}")
    model, cfg, spec = model_from_checkpoint(paths[0], device)
    if model.pooling != "mean":
        raise SystemExit(f"{cfg['name']}: per-window logits only make sense with pooling mean")
    cache = open_cache(a.cache_root, spec)
    labels = pd.read_csv(a.labels, index_col=0)[TARGETS]
    labels = labels.loc[[u for u in labels.index if u in cache]]
    folds = pd.read_csv(a.folds).set_index(ID)["fold"].to_dict()
    uids = [u for u in labels.index if folds.get(u) == a.fold]
    print(f"{cfg['name']}: fold {a.fold}, {len(uids)} studies, cache {spec.tag}, device {device}")
    dl = DataLoader(KneeStudies(cache, uids, labels), shuffle=False, batch_size=a.batch_size,
                    num_workers=a.workers, pin_memory=True)
    z, valid = dump_window_logits(model, dl, device, amp=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, logits=z, valid=valid, uids=np.array(uids), run=cfg["name"],
                        fold=a.fold, labels=cfg["data"]["labels"])
    ctrl = aggregate_table(z, valid, uids, labels).loc["mean_logit", "macro_auc"]
    print(f"saved {out}: logits {z.shape}, valid {int(valid.sum())}/{valid.size}; "
          f"mean_logit macro {ctrl:.4f} (must equal the screening number)")


def _table(a) -> None:
    labels = pd.read_csv(a.labels, index_col=0)[TARGETS]
    frames = []
    for spec in a.npz:
        name, _, path = spec.rpartition("=")
        d = np.load(path, allow_pickle=False)
        name = name or str(d["run"])
        t = aggregate_table(d["logits"], d["valid"], list(d["uids"]), labels)
        t.insert(0, "run", name)
        frames.append(t)
    table = pd.concat(frames).rename_axis("aggregator").reset_index()
    cols = ["run", "aggregator", "macro_auc"] + TARGETS
    table = table[cols]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))
    if a.json:
        best = table.loc[table.groupby("run")["macro_auc"].idxmax()]
        Path(a.json).write_text(json.dumps(
            {r.run: {"best": r.aggregator, "macro_auc": float(r.macro_auc)} for r in best.itertuples()},
            indent=2), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="window aggregators (diagnostic)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="GPU: save the per-window logits of a checkpoint on one fold")
    d.add_argument("--ckpt-dir", required=True, help="folder containing only fold0.pt")
    d.add_argument("--cache-root", required=True)
    d.add_argument("--labels", required=True)
    d.add_argument("--folds", required=True)
    d.add_argument("--fold", type=int, default=0)
    d.add_argument("--out", required=True, help="output .npz")
    d.add_argument("--batch-size", type=int, default=4)
    d.add_argument("--workers", type=int, default=2)
    d.set_defaults(fn=_dump)
    t = sub.add_parser("table", help="CPU: the aggregator table from one or more .npz files")
    t.add_argument("--npz", nargs="+", required=True, help="[name=]path.npz")
    t.add_argument("--labels", required=True)
    t.add_argument("--out", default=None, help="table .csv")
    t.add_argument("--json", default=None, help="the best one per run, for the decision rule")
    t.set_defaults(fn=_table)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
