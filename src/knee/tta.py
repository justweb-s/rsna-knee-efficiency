"""Test-time augmentation without flips: the variants, their evaluation on OOF, the combination.

No horizontal flip, by construction: medial and lateral are different columns (menisci, OA, MCL),
and mirroring the image swaps what the model must tell apart. The variants left are those that do
not touch laterality:

- **in-plane geometric**, deterministic: zoom ±5% around the centre, shifts of ±3% of the side.
  The same transforms as `augment.py` (`affine_grid` + `grid_sample`, zero border) with fixed
  instead of random parameters. The final submission uses a single zoom view, `s1.10` (scale
  1.10), registered by the submission script;
- **slice shift** (`off1`): windows start at slice 1 instead of 0. With stride 2 and depth 3 over
  24 slices there are again 11 windows per slot (slices 1-3, 3-5, ..., 21-23): same number of
  votes, a different triplet of slices for each;
- **stride 1** (`stride1`): all 22 windows per slot in a single pooling (176 instead of 88): more
  votes, which raised OOF but not the leaderboard; kept as a control. Not valid for models with
  context across windows (the position along the slices is wired to 11 per slot);
- **stride 3 and 4** (`stride3`, `stride4`): *fewer* votes, 8 and 6 windows per slot (64 and 48
  instead of 88) over the same 24 slices. An efficiency lever, since inference time is GPU-bound
  and linear in the windows: `stride4` cuts 45% of the encoder passes. Like `stride1`, not valid
  for context models;
- **photometric** (`g0.8`, `g1.25`): gamma on the input in [0, 1], x^0.8 brightens the mid-tones
  and x^1.25 darkens them. They keep 0 and 1 fixed, so they do not saturate fluid. The two gammas
  are symmetric on a log scale and touch neither geometry nor windows.

The combination is the **mean of the probabilities** of the variants, fixed before measuring (no
weights, no a-posteriori choice between mean of logits and of probabilities). `id` is the model as
is, and on OOF it must reproduce the recorded number: it is the control.

Two parts, as in `aggregate.py`: `predict_variants` runs on the GPU (one pass per variant, mean
over folds), the rest is numpy/pandas and is tested locally.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .constants import ID, TARGETS
from .models.baseline import make_windows, window_slot_index, windows_per_slot


@dataclass(frozen=True)
class Variant:
    """A variant: in-plane geometry (zoom `scale` around the centre, shift `tx`/`ty` as a fraction of
    the side), windows (`slice_offset`, `stride`; `stride` None = the model's) and intensity (`gamma`
    on the input in [0, 1]).
    """
    name: str
    scale: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    slice_offset: int = 0
    stride: int | None = None
    gamma: float = 1.0

    @property
    def geometric(self) -> bool:
        return self.scale != 1.0 or self.tx != 0.0 or self.ty != 0.0

    @property
    def photometric(self) -> bool:
        return self.gamma != 1.0

    @property
    def same_windows(self) -> bool:
        """Same number of windows as the model: what a compiled kernel requires."""
        return self.stride is None


VARIANTS: dict[str, Variant] = {
    "id": Variant("id"),
    "s0.95": Variant("s0.95", scale=0.95),
    "s1.05": Variant("s1.05", scale=1.05),
    "tx+": Variant("tx+", tx=0.03),
    "tx-": Variant("tx-", tx=-0.03),
    "ty+": Variant("ty+", ty=0.03),
    "ty-": Variant("ty-", ty=-0.03),
    "off1": Variant("off1", slice_offset=1),
    "stride1": Variant("stride1", stride=1),
    "stride3": Variant("stride3", stride=3),
    "stride4": Variant("stride4", stride=4),
    "g0.8": Variant("g0.8", gamma=0.8),
    "g1.25": Variant("g1.25", gamma=1.25),
}

# The combinations fixed before measuring. `geo3` is the candidate (3 passes); the others say
# whether adding passes costs or pays. `stride1` is the control. `geo3offg` adds the two gammas to
# the production TTA (6 passes); `pho3` reads the gammas alone around `id`.
COMBOS: dict[str, list[str]] = {
    "geo3": ["id", "s0.95", "s1.05"],
    "geo7": ["id", "s0.95", "s1.05", "tx+", "tx-", "ty+", "ty-"],
    "slice2": ["id", "off1"],
    "geo3off": ["id", "s0.95", "s1.05", "off1"],
    "stride1": ["stride1"],
    "pho3": ["id", "g0.8", "g1.25"],
    "geo3offg": ["id", "s0.95", "s1.05", "off1", "g0.8", "g1.25"],
}


def variant_passes(v: Variant, n_slices: int = 24, stride: int = 2, depth: int = 3) -> float:
    """The cost of the variant in encoder passes, relative to the model: the ratio between its windows
    and the model's (1 for the geometric variants and `off1`, 2 for `stride1` at 24 slices and stride
    2, 8/11 and 6/11 for `stride3` and `stride4`). It is what the efficiency track pays: inference
    time is linear in the windows.
    """
    if v.stride is None:
        return 1.0
    return windows_per_slot(n_slices, v.stride, depth) / windows_per_slot(n_slices, stride, depth)


def parse_tta(spec: str | None) -> list[Variant] | None:
    """`None`/"none" -> no TTA; the name of a combination or a comma-separated list of variants -> the
    variants, in the given order.
    """
    if spec is None or spec.strip().lower() in ("", "none"):
        return None
    names = COMBOS.get(spec) or [s.strip() for s in spec.split(",")]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown TTA variants: {unknown} (known: {sorted(VARIANTS)}, "
                         f"combinations: {sorted(COMBOS)})")
    return [VARIANTS[n] for n in names]


def affine_windows(w: torch.Tensor, scale: float = 1.0, tx: float = 0.0, ty: float = 0.0) -> torch.Tensor:
    """(B, K, D, H, W) -> same shape: zoom `scale` around the centre and shift (`tx`, `ty`) as a
    fraction of the side, the same for all windows. `affine_grid` maps output onto input: zoom in =
    tighter grid = 1/scale; the shift in normalised coordinates [-1, 1] is twice the fraction.
    Identity = `w` itself.
    """
    if scale == 1.0 and tx == 0.0 and ty == 0.0:
        return w
    b, k, d, h, wd = w.shape
    x = w.reshape(b * k, d, h, wd)
    theta = torch.tensor([[1.0 / scale, 0.0, -2.0 * tx], [0.0, 1.0 / scale, -2.0 * ty]],
                         device=x.device, dtype=x.dtype).unsqueeze(0)
    # The grid is the same for all windows: computed for one and expanded (stride-0 view). Computing it
    # for all was an fp16 GEMM per window, ~2% of the GPU of a pass with fewer windows per member; every
    # point is the same sum of three products.
    grid = F.affine_grid(theta, (1, d, h, wd), align_corners=False).expand(b * k, h, wd, 2)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return x.reshape(b, k, d, h, wd)


def variant_windows(vol01: torch.Tensor, v: Variant, stride: int, depth: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """The windows (B, W, depth, H, W_) and their slot (W,) for variant `v`, from a volume
    (B, S, K, H, W) already in [0, 1]. `stride`/`depth` are the model's.
    """
    st = stride if v.stride is None else v.stride
    if v.slice_offset:
        vol01 = vol01[:, :, v.slice_offset:]
    if v.photometric:
        # On the volume and not on the windows: 24 slices instead of 33 (the triplets overlap).
        # Under autocast `pow` returns float32 (`_probs_tta`): cast back to the input dtype.
        vol01 = vol01.pow(v.gamma).to(vol01.dtype)
    b, s, k = vol01.shape[:3]
    windows_per_slot(k, st, depth)
    w = make_windows(vol01, st, depth)
    if v.geometric:
        w = affine_windows(w, v.scale, v.tx, v.ty)
    slot_of = window_slot_index(s, k, vol01.device, st, depth)
    return w, slot_of


@torch.no_grad()
def predict_variants(models: list, loader, device, variants: list[Variant], amp: bool
                     ) -> dict[str, np.ndarray]:
    """For every variant, the probabilities (N, T) averaged over the `models` (the folds), over all
    windows of every study, in loader order (as in `train.predict`).
    """
    for m in models:
        m.eval()
    stride, depth = models[0].window_stride, models[0].window_depth
    out: dict[str, list[np.ndarray]] = {v.name: [] for v in variants}
    for vol, mask, _, _ in loader:
        vol, mask = vol.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        vol01 = vol.float().div_(255.0)
        for v in variants:
            w, slot_of = variant_windows(vol01, v, stride, depth)
            with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
                p = torch.stack([torch.sigmoid(m(w, slot_of, mask).float()) for m in models]).mean(0)
            out[v.name].append(p.cpu().numpy().astype(np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def combine(probs: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """The fixed combination: mean of the variants' probabilities."""
    return np.mean([probs[n] for n in names], axis=0)


def teacher_targets(labels: pd.DataFrame, uids: list[str]) -> pd.DataFrame:
    y = (labels.loc[uids, TARGETS] > 0.5).astype(float)
    y[labels.loc[uids, TARGETS].isna()] = np.nan
    return y


def tta_table(probs: dict[str, np.ndarray], uids: list[str], labels: pd.DataFrame,
              combos: dict[str, list[str]] | None = None, n_slices: int = 24,
              stride: int = 2, depth: int = 3) -> pd.DataFrame:
    """One row per variant and per combination: `macro_auc` and the twelve columns against the teacher,
    plus how many passes it costs (`variant_passes` on the model geometry: `n_slices`, `stride`,
    `depth`).
    """
    from .metrics import evaluate    # scikit-learn: 2.1 s cold import, only here
    combos = COMBOS if combos is None else combos
    y = teacher_targets(labels, uids)
    cost = {n: variant_passes(v, n_slices, stride, depth) for n, v in VARIANTS.items()}
    rows = {}
    for name, p in probs.items():
        m = evaluate(y, pd.DataFrame(p, index=uids, columns=TARGETS))
        rows[name] = {"kind": "variant", "passes": cost.get(name, 1.0),
                      "macro_auc": m["macro_auc"], **m["auc_per_class"]}
    for name, members in combos.items():
        if not all(mm in probs for mm in members):
            continue
        m = evaluate(y, pd.DataFrame(combine(probs, members), index=uids, columns=TARGETS))
        passes = sum(cost.get(mm, 1.0) for mm in members)
        rows[name] = {"kind": "combo", "passes": passes, "macro_auc": m["macro_auc"],
                      **m["auc_per_class"]}
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------- command line

def _oof(a) -> None:
    from torch.utils.data import DataLoader

    from .aggregate import model_from_checkpoint
    from .data.dataset import KneeStudies, open_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = parse_tta(a.variants)
    ckpts = sorted(Path(a.ckpt_dir).glob("fold*.pt"))
    if not ckpts:
        raise SystemExit(f"no fold*.pt in {a.ckpt_dir}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(a.labels, index_col=0)[TARGETS]
    folds = pd.read_csv(a.folds).set_index(ID)["fold"].to_dict()
    cache = spec = cfg = None
    parts: dict[str, list[np.ndarray]] = {v.name: [] for v in variants}
    uids_all: list[str] = []
    for p in ckpts:
        model, cfg, spec = model_from_checkpoint(p, device)
        k = int(p.stem.replace("fold", ""))
        if any(v.stride is not None for v in variants) and getattr(model, "context_layers", 0) > 0:
            raise SystemExit("variants with a different stride (stride1/3/4) are not valid for a "
                             "model with context across windows")
        if cache is None:
            cache = open_cache(a.cache_root, spec)
            labels = labels.loc[[u for u in labels.index if u in cache]]
        uids = [u for u in labels.index if folds.get(u) == k]
        print(f"{cfg['name']} fold {k}: {len(uids)} studies, {len(variants)} variants", flush=True)
        dl = DataLoader(KneeStudies(cache, uids, labels), shuffle=False, batch_size=a.batch_size,
                        num_workers=a.workers, pin_memory=True)
        pr = predict_variants([model], dl, device, variants, amp=True)
        for name, arr in pr.items():
            parts[name].append(arr)
        uids_all += uids
        del model
    probs = {name: np.concatenate(v) for name, v in parts.items()}
    for name, arr in probs.items():
        pd.DataFrame(arr, index=pd.Index(uids_all, name=ID), columns=TARGETS).to_csv(out / f"oof_{name}.csv")
    stride, depth = int(cfg["model"].get("window_stride", 1)), int(cfg["model"].get("window_depth", 3))
    table = tta_table(probs, uids_all, labels, n_slices=spec.slices, stride=stride, depth=depth)
    table.rename_axis("tta").reset_index().to_csv(out / "tta_table.csv", index=False)
    base = table.loc["id", "macro_auc"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(table[["kind", "passes", "macro_auc"]].assign(delta=table["macro_auc"] - base).to_string())
    (out / "summary.json").write_text(json.dumps({
        "run": cfg["name"], "n_studies": len(uids_all), "id_macro_auc": float(base),
        "n_slices": int(spec.slices), "window_stride": stride, "window_depth": depth,
        "rows": {n: {"macro_auc": float(r.macro_auc), "passes": float(r.passes), "kind": r.kind}
                 for n, r in table.iterrows()}}, indent=2), encoding="utf-8")
    print(f"written {out}: oof_<variant>.csv, tta_table.csv, summary.json")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="TTA without flips: evaluation on OOF")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof", help="GPU: every fold on its validation fold, per variant")
    o.add_argument("--ckpt-dir", required=True, help="folder with the run's fold*.pt")
    o.add_argument("--cache-root", required=True)
    o.add_argument("--labels", required=True)
    o.add_argument("--folds", required=True)
    o.add_argument("--variants", default=",".join(VARIANTS),
                   help="variants (comma-separated) or a combination; default all")
    o.add_argument("--out", required=True)
    o.add_argument("--batch-size", type=int, default=4)
    o.add_argument("--workers", type=int, default=2)
    o.set_defaults(fn=_oof)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
