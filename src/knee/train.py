"""Training of the image model, five-fold or on all data (`train.full_data`), with the validation
metrics computed by the code.

A run produces two artifacts that count: `oof.csv`, one prediction for every study made by a model
that did not see it (in-sample with `full_data`), and `metrics.json`.

On the 58 gold studies the model's `gold58_macro_auc` is compared with that of its *teacher*, i.e. the
label set it was trained on: the direction of that comparison says whether the next hour should go
into the image pipeline or into the labels.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .augment import AugmentSpec, MixupSpec, apply_augment, mixup_batch
from .constants import ID, TARGETS
from .data.cache import CacheSpec
from .data.dataset import KneeStudies, open_cache
from .data.extra import UNION, CombinedCache, extra_targets, has_extra, open_extras
from .data.windows import WindowBatchSampler, use_window_loader
from .dread_recipe import checkpoint_of, is_dread, open_corpus, train_fold_dread
from .ema import ModelEma, ema_decays
from .labels.gold import load_gold, score_label_set
from .metrics import evaluate
from .dino import dino_checkpoint, is_dino, train_fold_dino
from .qwen import is_qwen, qwen_checkpoint, train_fold_qwen
from .models.baseline import (GROUP, KneeBaseline, encode_chunk_for, make_windows,
                              window_slot_index, windows_per_slot)
from .models.video3d import channels_last_ok
from .reports import fit_report_targets, load_report_embeddings
from .utils.config import config_hash, git_sha
from .utils.device import check_device

# Studies in the test set, to project the inference runtime for the Efficiency track. A declared
# projection, not a measurement: DICOM decoding is not included and is measured by the submission
# kernel, where it is the largest item.
N_TEST_STUDIES = 1300


def masked_bce(logits: torch.Tensor, y: torch.Tensor,
               weight: torch.Tensor | None = None,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE on soft targets, ignoring NaN cells and optionally weighting per sample.

    `pos_weight` (one value per column, `train.pos_weight`) multiplies the positive term of the BCE,
    `y log sigma(z)`, as in `binary_cross_entropy_with_logits`. With None the call is the usual one, bit
    for bit.
    """
    m = torch.isfinite(y)
    if not m.any():
        return logits.sum() * 0.0
    pw = None if pos_weight is None else pos_weight.to(y.device, torch.float32).expand_as(y)[m]
    if weight is None:
        if pw is None:
            return nn.functional.binary_cross_entropy_with_logits(logits[m], y[m])
        return nn.functional.binary_cross_entropy_with_logits(logits[m], y[m], pos_weight=pw)
    if weight.ndim == 1:
        weight = weight.unsqueeze(1).expand_as(y)
    if pw is None:
        loss = nn.functional.binary_cross_entropy_with_logits(logits[m], y[m], reduction="none")
    else:
        loss = nn.functional.binary_cross_entropy_with_logits(logits[m], y[m], reduction="none",
                                                              pos_weight=pw)
    w = weight[m]
    return (loss * w).sum() / w.sum().clamp(min=1.0)


_MM, _LM = TARGETS.index("Medial Meniscus"), TARGETS.index("Lateral Meniscus")


def union_bce(z_a: torch.Tensor, z_b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """BCE of "at least one of the two": p = 1 - (1 - sigma(z_a)) (1 - sigma(z_b)), per element.

    It is MRNet's meniscus label without side: it says a meniscus is torn, not which one. In log space
    and without losing digits at any logit: log(1 - p) = logsigmoid(-z_a) + logsigmoid(-z_b), and since
    p = p_a + (1 - p_a) p_b, log p = logaddexp(logsigmoid(z_a), logsigmoid(-z_a) + logsigmoid(z_b)). The
    gradient pushes both until either is high, then fades: the column already convinced takes the
    credit, the other is not pulled up.
    """
    ls = nn.functional.logsigmoid
    z_a, z_b, t = z_a.float(), z_b.float(), t.float()
    log_p = torch.logaddexp(ls(z_a), ls(-z_a) + ls(z_b))
    log_q = ls(-z_a) + ls(-z_b)
    return -(t * log_p + (1.0 - t) * log_q)


def masked_bce_union(logits: torch.Tensor, y: torch.Tensor, y_union: torch.Tensor | None = None,
                     weight: torch.Tensor | None = None) -> torch.Tensor:
    """`masked_bce` plus the "meniscus without side" cells (`data.extra`, `UNION`).

    Every supervised cell counts once: the mean is over the finite cells of `y` plus the finite cells of
    `y_union`, with `masked_bce`'s per-sample or per-cell weights (the union cell takes the mean of the
    weights of the two meniscus columns). Without union cells in the batch it is `masked_bce` itself.
    """
    if y_union is None:
        return masked_bce(logits, y, weight)
    u = torch.isfinite(y_union)
    if not u.any():
        return masked_bce(logits, y, weight)
    m = torch.isfinite(y)
    if weight is None:
        w = torch.ones(y.shape, dtype=torch.float32, device=y.device)
    elif weight.ndim == 1:
        w = weight.float().unsqueeze(1).expand_as(y)
    else:
        w = weight.float()
    cell = nn.functional.binary_cross_entropy_with_logits(logits[m].float(), y[m].float(),
                                                          reduction="none")
    un = union_bce(logits[u, _MM], logits[u, _LM], y_union[u])
    w_u = w[:, [_MM, _LM]].mean(dim=1)[u]
    return ((cell * w[m]).sum() + (un * w_u).sum()) / (w[m].sum() + w_u.sum()).clamp(min=1.0)


def rank_loss(logits: torch.Tensor, y: torch.Tensor,
              weight: torch.Tensor | None = None) -> torch.Tensor:
    """Pairwise ranking loss on the study logit (`train.loss: rank`, an experiment).

    The metric is the macro ROC-AUC, i.e. the order within each column; BCE on soft targets optimises
    the absolute value instead, and the order comes as a side effect. Here, for every column t and every
    pair of studies (i, j) of the batch with different targets, the loss is the logistic on the logit
    difference in the teacher's direction, `softplus(-(z_i - z_j))` if `y_i > y_j`, weighted by
    `|y_i - y_j|`: a NO/full-YES pair weighs ~1, a NO/UNK pair ~0.15, a pair with equal targets zero. It
    is RankNet with the teacher's target as pair weight, and `w_ij = (w_i + w_j) / 2` on top for
    per-sample/per-cell weights, as in `masked_bce`.

    Per column the loss is the weighted mean over valid pairs (both targets finite and different); the
    total is the mean over the columns with at least one pair in the batch. A batch without pairs (all
    targets equal) returns zero with the graph attached. Scale: with all logits equal it is log 2 by
    construction, like BCE at p = 0.5.

    With `batch_size` 8 there are at most 28 pairs per column per step, and in the rare columns
    (Fracture 4.5% positives, MCL 5.3%) only a third of the batches contain a NO/YES pair; the `|dy|`
    weight still keeps a signal on NO/UNK and NO/mild pairs. No memory across batches: that would be a
    second variable.
    """
    z = logits.float()
    y = y.float()
    fin = torch.isfinite(y)
    if weight is None:
        w = torch.ones_like(y)
    elif weight.ndim == 1:
        w = weight.float().unsqueeze(1).expand_as(y)
    else:
        w = weight.float()
    y0 = torch.where(fin, y, torch.zeros_like(y))
    dz = z.unsqueeze(1) - z.unsqueeze(0)                    # (B, B, T): z_i - z_j
    dy = y0.unsqueeze(1) - y0.unsqueeze(0)                  # (B, B, T): y_i - y_j
    valid = fin.unsqueeze(1) & fin.unsqueeze(0) & (dy != 0)
    pair_w = dy.abs() * 0.5 * (w.unsqueeze(1) + w.unsqueeze(0)) * valid.to(z.dtype)
    per_pair = nn.functional.softplus(-torch.sign(dy) * dz)  # ~ -log P(teacher's order)
    num = (per_pair * pair_w).sum(dim=(0, 1))               # (T,)
    den = pair_w.sum(dim=(0, 1))                            # (T,)
    has = den > 0
    if not has.any():
        return logits.sum() * 0.0
    return (num[has] / den[has]).mean()


COMPARTMENT_PAIRS = (("Lateral OA", "Medial OA"), ("PF OA", "Medial OA"), ("Lateral OA", "PF OA"))


def compartment_pair_loss(logits: torch.Tensor, y: torch.Tensor, min_diff: float = 0.2) -> torch.Tensor:
    """Contrast between OA compartments (`train.compartment_pair`, an experiment ported from Matteo
    Vitali's v61m pipeline).

    The model separates medial from lateral on the menisci (0.989), but the Lateral OA logit follows
    osteoarthritis in general (lateral-only OA against medial-only: AUC 0.31, teacher 0.71). For every
    pair (A, B) of `COMPARTMENT_PAIRS` and every study with finite targets and |y_A - y_B| >=
    `min_diff`: BCE of sigmoid(z_A - z_B) against [y_A > y_B], weighted by |y_A - y_B|. It depends only
    on the DIFFERENCE of the logits: adding the same amount to both columns does not change it, so it
    does not push the common component ("arthritic knee") and trains the compartment one. The two
    directions are averaged separately ("lateral > medial" cases are ~1/8 of the others), then the mean
    over the pairs with at least one discordant study in the batch. A batch without discordant studies:
    zero with the graph attached. Same formula as v61m, with `isfinite(y)` instead of the cell weight.
    """
    z, y = logits.float(), y.float()
    tot = z.new_zeros(())
    npair = 0
    for a, b in COMPARTMENT_PAIRS:
        ia, ib = TARGETS.index(a), TARGETS.index(b)
        d = y[:, ia] - y[:, ib]
        ok = torch.isfinite(d) & (d.abs() >= float(min_diff))
        if not ok.any():
            continue
        d, dz = d[ok], z[ok, ia] - z[ok, ib]
        ce = nn.functional.binary_cross_entropy_with_logits(dz, (d > 0).float(), reduction="none")
        dirs = [ce[m].mul(d[m].abs()).sum() / d[m].abs().sum() for m in (d > 0, d < 0) if m.any()]
        tot = tot + torch.stack(dirs).mean()
        npair += 1
    return tot / max(npair, 1) + logits.sum() * 0.0


def compartment_pair_spec_of(cfg: dict) -> dict | None:
    """`train.compartment_pair: {weight, min_diff}`, validated, or None.

    The term is ADDED to the per-study BCE (`weight` x `compartment_pair_loss`), like the auxiliary
    heads: the BCE stays alone in the log, comparable with the plain model. Valid only with the
    per-study BCE without tails on `y`, mixup or positive weights: there the loss goes through its own
    branches, and the run would be a hybrid.
    """
    spec = cfg.get("train", {}).get("compartment_pair")
    if not spec:
        return None
    if not isinstance(spec, dict):
        raise ValueError("train.compartment_pair: expected {weight, min_diff}")
    unknown = set(spec) - {"weight", "min_diff"}
    if unknown:
        raise ValueError(f"train.compartment_pair: unknown keys {sorted(unknown)}")
    out = {"weight": float(spec.get("weight", 0.1)), "min_diff": float(spec.get("min_diff", 0.2))}
    if out["weight"] <= 0.0:
        raise ValueError("train.compartment_pair.weight must be > 0 (remove the block to "
                         "turn the term off)")
    if not 0.0 < out["min_diff"] <= 1.0:
        raise ValueError(f"train.compartment_pair.min_diff in (0, 1], got {out['min_diff']}")
    tr = cfg["train"]
    clash = [k for k, on in (("train.loss", tr.get("loss", "study") != "study"),
                             ("train.mixup", bool(tr.get("mixup"))),
                             ("train.report_distill", bool(tr.get("report_distill"))),
                             ("train.aux_findings", bool(tr.get("aux_findings"))),
                             ("train.site_bias", bool(tr.get("site_bias"))),
                             ("train.pos_weight", bool(tr.get("pos_weight"))),
                             ("data.extra", bool(cfg.get("data", {}).get("extra")))) if on]
    if clash:
        raise ValueError(f"train.compartment_pair applies only to the per-study BCE, without {clash}")
    return out


def window_bce(logits_w: torch.Tensor, valid: torch.Tensor, y: torch.Tensor,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    """Per-window BCE (`train.loss: window`, an experiment): every valid window answers for the study's
    soft target.

    `logits_w` (B, W, T) from `KneeBaseline.window_logits`, `valid` (B, W) bool, `y` (B, T) with NaN
    where the target is missing (ignored as in `masked_bce`). Every study weighs as in the per-study
    loss: the weights of its valid windows sum to 1, so a study with missing slots (fewer windows) does
    not count less than the others, and the loss is the mean over (study, finite target) of the mean BCE
    over the windows: the same mean `masked_bce` takes over the BCE of the mean logit. The per-sample
    `weight` is multiplied on top, as in `masked_bce`.

    Why it is a different loss and not a detail: on the mean of the logits the gradient reaching every
    window is (1/n)(sigma(z_mean) - y), the same for all, and one window can sit at +10 while the others
    sit at -10; per window it is (1/n)(sigma(z_w) - y), each with its own error: the "study label on
    every image" recipe.
    """
    b, w, t = logits_w.shape
    n_valid = valid.sum(dim=1).clamp(min=1).to(logits_w.dtype)        # (B,)
    per = valid.to(logits_w.dtype) / n_valid.unsqueeze(1)              # (B, W), sums to 1 per study
    if weight is not None:
        per = per * weight.to(per.dtype).unsqueeze(1)
    y_w = y.unsqueeze(1).expand(b, w, t)
    return masked_bce(logits_w.reshape(b * w, t), y_w.reshape(b * w, t), per.reshape(b * w))


def report_spec_of(cfg: dict) -> dict | None:
    """`train.report_distill` (an experiment), validated, or None.

    `emb` is the frozen artifact in `data/reports/`, `weight` the weight of the auxiliary loss next to
    the BCE, `dim` the components of the target, `drop_pcs` the nuisance directions removed before the
    residualisation (0 = none: the per-language means are already removed by the design of
    `knee.reports`).
    """
    spec = cfg.get("train", {}).get("report_distill")
    if not spec:
        return None
    if not isinstance(spec, dict) or "emb" not in spec:
        raise ValueError("train.report_distill: at least `emb` (artifact name) is required")
    out = {"emb": str(spec["emb"]), "weight": float(spec.get("weight", 0.3)),
           "dim": int(spec.get("dim", 64)), "drop_pcs": int(spec.get("drop_pcs", 0))}
    if out["weight"] <= 0.0:
        raise ValueError("train.report_distill.weight must be > 0 (remove the block "
                         "to turn the auxiliary head off)")
    if cfg["train"].get("loss", "study") != "study":
        raise ValueError("train.report_distill applies only to the per-study loss: the auxiliary "
                         "head is added to the BCE, not to `window` or `rank`")
    return out


def report_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine between the auxiliary head and the target. Both already have norm 1.

    With an untrained head it is ~1.0 (two random vectors in 64 dimensions are nearly orthogonal): the
    curve in the log is the marker of the run. If it does not go down, the report residual is not
    visible in the image, which is a conclusion, not a failed run.
    """
    return (1.0 - (pred.float() * target.float()).sum(-1)).mean()


def strip_aux(state: dict) -> dict:
    """The state_dict without the auxiliary heads: at inference the model does not have them.

    `infer.py` rebuilds `KneeBaseline` from the `model:` block of the cfg alone and loads strictly;
    leaving `report_proj.*` or `aux_head.*` inside the checkpoint would make the weights unusable for the
    submission without a dedicated branch.
    """
    return {k: v for k, v in state.items()
            if not k.startswith(("report_proj.", "aux_head."))}


def aux_findings_spec_of(cfg: dict) -> dict | None:
    """`train.aux_findings` (an experiment), validated, or None.

    `labels` is the CSV of the auxiliary targets in `data/labels/` (studies x K findings, values in
    [0, 1], NaN where the report does not mention them), `weight` the weight of the auxiliary loss next
    to the BCE of the 12 columns. Valid only with the per-study loss and without the other tails of `y`
    (report head, extra rows, site bias) or mixup: each slices the tail its own way, and two together
    would be ambiguous.
    """
    spec = cfg.get("train", {}).get("aux_findings")
    if not spec:
        return None
    if not isinstance(spec, dict) or "labels" not in spec:
        raise ValueError("train.aux_findings: at least `labels` (CSV name in data/labels) is required")
    unknown = set(spec) - {"labels", "weight"}
    if unknown:
        raise ValueError(f"train.aux_findings: unknown keys {sorted(unknown)}")
    out = {"labels": str(spec["labels"]), "weight": float(spec.get("weight", 0.5))}
    if out["weight"] <= 0.0:
        raise ValueError("train.aux_findings.weight must be > 0 (remove the block to "
                         "turn the auxiliary head off)")
    t = cfg["train"]
    if t.get("loss", "study") != "study" or t.get("mixup") or t.get("report_distill") \
            or t.get("site_bias") or has_extra(cfg):
        raise ValueError("train.aux_findings applies only to train.loss: study, without train.mixup, "
                         "train.report_distill, train.site_bias and data.extra: the tail of y carries "
                         "the auxiliary findings")
    return out


def aux_findings_for(cfg: dict, labels_dir, index) -> pd.DataFrame | None:
    """The auxiliary targets of `train.aux_findings` on the labelled studies, or None.

    Every labelled study must have a row (NaN allowed: "the report does not mention it"), values are in
    [0, 1], and no column is one of the 12: an auxiliary finding coinciding with a teacher column would
    be the BCE counted twice.
    """
    spec = aux_findings_spec_of(cfg)
    if spec is None:
        return None
    path = Path(labels_dir) / f"{spec['labels']}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"train.aux_findings: missing {path}")
    t = pd.read_csv(path, index_col=0)
    if t.shape[1] == 0:
        raise ValueError(f"train.aux_findings: {path.name} has no columns")
    clash = set(t.columns) & set(TARGETS)
    if clash:
        raise ValueError(f"train.aux_findings: {path.name} contains teacher columns {sorted(clash)}")
    if not t.columns.is_unique or not t.index.is_unique:
        raise ValueError(f"train.aux_findings: {path.name} has repeated columns or studies")
    missing = index.difference(t.index)
    if len(missing):
        raise ValueError(f"train.aux_findings: {path.name} does not cover {len(missing)} labelled "
                         f"studies, the first is {missing[0]}")
    t = t.loc[index].astype(np.float32)
    v = t.to_numpy()
    fin = np.isfinite(v)
    if not ((v[fin] >= 0) & (v[fin] <= 1)).all():
        raise ValueError(f"train.aux_findings: {path.name} must be in [0, 1] (NaN allowed)")
    if not fin.any(axis=0).all():
        empty = [c for c, ok in zip(t.columns, fin.any(axis=0)) if not ok]
        raise ValueError(f"train.aux_findings: all-NaN columns {empty}")
    return t


def aux_auc(pred: np.ndarray, target: pd.DataFrame, min_class: int = 5) -> dict:
    """The marker of the auxiliary findings: how visible they are in the image, out of sample.

    For every finding, the AUC of the auxiliary head on the validation studies against the target
    binarised at 0.5 (for cartilage, grade >= 2 of 4), NaN cells excluded; only findings with at least
    `min_class` studies per class. `mean` is the mean over those.
    """
    from sklearn.metrics import roc_auc_score
    per = {}
    for j, col in enumerate(target.columns):
        y = target[col].to_numpy()
        m = np.isfinite(y)
        yb = (y[m] >= 0.5).astype(int)
        if yb.sum() < min_class or (1 - yb).sum() < min_class:
            continue
        per[col] = round(float(roc_auc_score(yb, pred[m, j])), 4)
    return {"mean": round(float(np.mean(list(per.values()))), 4) if per else None,
            "n_fields": len(per), "per_field": per}


def loss_mode_of(cfg: dict, pooling: str) -> str:
    """`train.loss`: 'study' (default, the BCE on the study logit: the final model), 'window' (per-window
    BCE, only with `model.pooling: mean`) or 'rank' (pairwise ranking on the study logit, `rank_loss`,
    with any pooling).
    """
    mode = cfg["train"].get("loss", "study")
    if mode not in ("study", "window", "rank"):
        raise ValueError(f"train.loss: expected 'study', 'window' or 'rank', got {mode!r}")
    if mode == "window" and pooling != "mean":
        raise ValueError("train.loss: window requires model.pooling: mean (at inference "
                         "the output must be the mean of the per-window logits); "
                         f"here pooling is {pooling!r}")
    return mode


def mixup_spec_of(cfg: dict, loss_mode: str, rep_spec: dict | None) -> MixupSpec | None:
    """`train.mixup`, only with the per-study loss: the per-window and ranking losses read the target
    differently, and the report head has a target that does not mix with the same λ without one more
    choice.
    """
    spec = MixupSpec.from_config(cfg)
    if spec is not None and (loss_mode != "study" or rep_spec is not None):
        raise ValueError("train.mixup applies only to train.loss: study and without train.report_distill")
    return spec


def split_backbone_head(model, lr: float, lr_backbone: float) -> list[dict]:
    """Two parameter groups: the pre-trained encoder slow, the new head at `lr`.

    Needed **only** for transformers, and for a reason that is not a preference. With the CNNs (resnet34,
    convnext_nano, densenet121/169) a single `lr` of 3e-4 works. On a self-supervised pre-trained ViT it
    does not: public DINOv2 fine-tuning recipes use 1e-5 to 5e-5 on the encoder, and 3e-4 on all blocks
    erases the features that are the only reason to take DINOv2. But the head (`slot_emb`, `attn`,
    `cls_w`) starts from noise and at 1.5e-5 learns nothing in 20 epochs: a single low `lr` is just as
    wrong, the other way round. Hence the two groups.

    Note: enabling it changes **two** things with respect to the parent, `arch` and the optimisation
    recipe. The default stays `lr_backbone: null`, which builds no group and leaves the plain path
    untouched (the final DenseNet does not use it).
    """
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    return [{"params": enc, "lr": lr_backbone}, {"params": head, "lr": lr}]


def _vit_depth(name: str, n_blocks: int) -> int:
    """The depth of a ViT encoder parameter for layer-wise LR decay (Matteo Vitali's rule):
    `blocks.i.*` -> i, the final norm -> n_blocks, everything before the blocks (patch embedding,
    cls/registers, pos_embed) -> -1.
    """
    m = re.search(r"blocks\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    return n_blocks if "norm" in name else -1


def _intra_no_decay(name: str) -> bool:
    """The weights of the per-finding maps of the intra head (`models.intra.NO_DECAY`)."""
    from .models.intra import NO_DECAY
    return name.startswith("intra.") and name.split(".")[1] in NO_DECAY


def vit_param_groups(model, lr: float, lr_backbone: float | None, llrd: float,
                     weight_decay: float) -> list[dict]:
    """The AdamW groups of Matteo Vitali's ViT recipe.

    - encoder: one group per depth, `lr_backbone * llrd ** (top - depth)`, with `top` the final norm (at
      `lr_backbone`); the last block at `llrd` times that, the patch embedding at `llrd ** 13` on a
      12-block ViT. `lr_backbone=None` = encoder excluded (the LP phase, frozen encoder).
    - head at `lr`; the weights of the per-finding maps (`models.intra.NO_DECAY`) without weight decay,
      as in his recipe. Everything else with the usual weight decay on everything (biases and norms
      included).
    Every group carries `name`, for the log.
    """
    n_blocks = len(getattr(model.encoder, "blocks", []))
    enc, head, head_nd = {}, [], []
    for n, p in model.named_parameters():
        if n.startswith("encoder."):
            enc.setdefault(_vit_depth(n, n_blocks), []).append(p)
        elif _intra_no_decay(n):
            head_nd.append(p)
        else:
            head.append(p)
    groups = [{"params": head, "lr": lr, "weight_decay": weight_decay, "name": "testa"}]
    if head_nd:
        groups.append({"params": head_nd, "lr": lr, "weight_decay": 0.0, "name": "mappe"})
    if lr_backbone is not None:
        top = max(enc)
        for d in sorted(enc):
            groups.append({"params": enc[d], "lr": lr_backbone * float(llrd) ** (top - d),
                           "weight_decay": weight_decay, "name": f"enc{d}"})
    return groups


def lp_spec_of(cfg: dict) -> dict | None:
    """`train.lp`: before the `train.epochs` of fine-tuning, `epochs` epochs of head only with the encoder
    frozen (in eval: no drop_path), at `lr` with warm-up and cosine down to `min_frac` x `lr`. None
    without the key.
    """
    spec = cfg["train"].get("lp")
    if not spec:
        return None
    unknown = set(spec) - {"epochs", "lr", "min_frac"}
    if unknown:
        raise ValueError(f"train.lp: unknown keys {sorted(unknown)}")
    out = {"epochs": int(spec["epochs"]), "lr": float(spec["lr"]),
           "min_frac": float(spec.get("min_frac", 0.2))}
    if out["epochs"] < 1:
        raise ValueError(f"train.lp.epochs {out['epochs']}: at least 1")
    return out


def warmup_cosine(total: int, warmup_frac: float, min_frac: float):
    """The per-step LR factor: linear ramp for `warmup_frac` of the steps, then cosine from 1 to `min_frac`
    (the floor of the recipe: the last epochs, those SWA averages, are not at zero LR).
    """
    warm = max(1, int(total * float(warmup_frac)))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total - warm))
        return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t))
    return f


class SwaAverage:
    """The ARITHMETIC mean of the end-of-epoch weights from fraction `start_frac` of the fine-tuning epochs
    on (`train.swa`; 0.7 in the recipe, every epoch). Only the float tensors of the state_dict; the
    others stay those of the model. A ViT has no BatchNorm to recalibrate.
    """

    def __init__(self, start_frac: float):
        self.start_frac = float(start_frac)
        self.n = 0
        self.avg = None
        self.epochs: list[int] = []

    def starts_at(self, ft_epochs: int) -> int:
        """The index (from 0) of the first fine-tuning epoch collected."""
        return int(ft_epochs * self.start_frac)

    @torch.no_grad()
    def collect(self, model: nn.Module, epoch: int) -> None:
        sd = model.state_dict()
        if self.avg is None:
            self.avg = {k: v.detach().float().clone() for k, v in sd.items()
                        if v.is_floating_point()}
        else:
            for k, a in self.avg.items():
                a.mul_(self.n / (self.n + 1)).add_(sd[k].detach().float(), alpha=1.0 / (self.n + 1))
        self.n += 1
        self.epochs.append(epoch)

    @torch.no_grad()
    def load_into(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, a in self.avg.items():
            sd[k].copy_(a.to(sd[k].dtype))


def augment_window_extra(aug: AugmentSpec | None, has_extra: bool, windowed: bool,
                         n_slices: int, stride: int, depth: int) -> int:
    """The extra slices the loader reads per window (`train.augment.slice_offset`), after checking that the
    recipe makes sense for the run: `rows: extra` needs `data.extra`; the offset needs the window loader,
    a stride longer than the offset (with stride 2 an offset of 2 is the next window) and the last window
    of the slot still inside the stack.
    """
    if aug is None:
        return 0
    if aug.rows == "extra" and not has_extra:
        raise ValueError("train.augment.rows: extra without data.extra: there are no rows to perturb")
    o = aug.slice_offset
    if not o:
        return 0
    if not windowed:
        raise ValueError("train.augment.slice_offset needs the window loader (k_train smaller "
                         "than the windows per study)")
    if o >= stride:
        raise ValueError(f"train.augment.slice_offset {o} with window_stride {stride}: "
                         f"an offset of {stride} is already the next window")
    last = (windows_per_slot(n_slices, stride, depth) - 1) * stride
    if last + depth + o > n_slices:
        raise ValueError(f"train.augment.slice_offset {o}: the last window of the slot (slices "
                         f"{last}-{last + depth - 1}) would go beyond the stack of {n_slices}")
    return o


def _sample_windows(w: torch.Tensor, slot_of: torch.Tensor, k: int, generator):
    """k random windows per study: every epoch sees a different view of the same study."""
    if k <= 0 or k >= w.shape[1]:
        return w, slot_of
    pick = torch.randperm(w.shape[1], generator=generator)[:k].to(w.device)
    return w[:, pick], slot_of[pick]


@torch.no_grad()
def predict(model, loader, device, amp: bool, dtype: torch.dtype = torch.float16,
            aux: bool = False) -> np.ndarray:
    """Predictions on all windows of every study: no sampling at inference.

    `dtype` is the autocast one; the fp16 default is `torch.autocast`'s default on cuda, i.e. the path
    of every run without `train.speed.amp_dtype`. With `aux=True` the probabilities of the auxiliary
    findings instead of the 12 columns: only for the end-of-fold marker, validation stays the usual one.
    """
    model.eval()
    out = []
    for vol, mask, _, _ in loader:
        vol, mask = vol.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        w = make_windows(vol.float().div_(255.0), model.window_stride, model.window_depth)
        slot_of = window_slot_index(vol.shape[1], vol.shape[2], device, model.window_stride,
                                      model.window_depth)
        with torch.autocast("cuda", dtype=dtype, enabled=amp and device.type == "cuda"):
            logits = model(w, slot_of, mask, return_aux=True)[1] if aux else model(w, slot_of, mask)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def resolve_encoder_weights(cfg: dict, weights_dir) -> Path | None:
    """The path of the pre-trained encoder weights, or None if the config does not ask for them.

    The config carries only the **file name** (`model.encoder_weights`) and the Kaggle Dataset containing
    it (`model.encoder_weights_dataset`): where that file sits on disk depends on the machine and must not
    enter the config, otherwise `config_hash` would change from one machine to another for the same
    experiment. The directory is passed by whoever launches the run.
    """
    name = cfg["model"].get("encoder_weights")
    if not name:
        return None
    if weights_dir is None:
        raise ValueError(f"the config asks for model.encoder_weights={name!r} but no "
                         "weights_dir was passed to run()")
    path = Path(weights_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"encoder weights not found: {path}")
    return path


def cell_weights_for(cfg: dict, labels_dir, index) -> pd.DataFrame | None:
    """The per-cell weight matrix declared by `train.cell_weight`, or None.

    `train.cell_weight: {mask: <name>, weight: w}` reads `data/labels/<name>.csv`, a 0/1 matrix (studies
    x 12 targets), and assigns `w` to the cells at 1 and 1.0 to the others.
    """
    spec = cfg.get("train", {}).get("cell_weight")
    if not spec:
        return None
    w = float(spec["weight"])
    mask = pd.read_csv(Path(labels_dir) / f"{spec['mask']}.csv", index_col=0)[TARGETS]
    missing = index.difference(mask.index)
    if len(missing):
        raise ValueError(f"train.cell_weight: the mask {spec['mask']} does not cover "
                         f"{len(missing)} labelled studies, the first is {missing[0]}")
    mask = mask.loc[index]
    if not mask.isin([0, 1]).all().all():
        raise ValueError(f"train.cell_weight: {spec['mask']} must be a 0/1 matrix")
    return pd.DataFrame(np.where(mask.to_numpy() == 1, w, 1.0), index=index, columns=TARGETS)


def pos_weight_of(cfg: dict, targets: pd.DataFrame) -> np.ndarray | None:
    """The per-column weight of the positive BCE term, from `train.pos_weight`, or None.

    `train.pos_weight: {cols: [...], power: p}`: for every listed column `(n_neg / n_pos) ** p`, counted
    on `targets` (the fold's TRAINING targets: > 0.5 positive, <= 0.5 negative, NaN excluded); 1 for the
    others. With p = 0.5 it is the square root of the ratio: MCL ~4, Fracture ~4.5, Synovitis ~3. Valid
    only with the per-study BCE without tails on `y` (extra rows, report, auxiliary findings, site bias)
    and without mixup: there the loss goes through functions the weight does not touch, and the run would
    be a hybrid.
    """
    spec = cfg.get("train", {}).get("pos_weight")
    if not spec:
        return None
    cols = list(spec["cols"])
    power = float(spec.get("power", 0.5))
    unknown = [c for c in cols if c not in TARGETS]
    if unknown or not cols:
        raise ValueError(f"train.pos_weight: unknown or empty columns: {unknown or cols}")
    if not power > 0:
        raise ValueError(f"train.pos_weight: power must be > 0, got {power}")
    tr = cfg.get("train", {})
    clash = [k for k, on in (("train.loss", tr.get("loss", "study") != "study"),
                             ("train.mixup", bool(tr.get("mixup"))),
                             ("train.report_distill", bool(tr.get("report_distill"))),
                             ("train.aux_findings", bool(tr.get("aux_findings"))),
                             ("train.site_bias", bool(tr.get("site_bias"))),
                             ("data.extra", bool(cfg.get("data", {}).get("extra")))) if on]
    if clash:
        raise ValueError(f"train.pos_weight applies only to the per-study BCE, without {clash}")
    w = np.ones(len(TARGETS), dtype=np.float32)
    for c in cols:
        y = targets[c].dropna()
        n_pos, n_neg = int((y > 0.5).sum()), int((y <= 0.5).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"train.pos_weight: {c} has {n_pos} positives and {n_neg} negatives")
        w[list(TARGETS).index(c)] = (n_neg / n_pos) ** power
    return w


def pseudo_labels_for(cfg: dict, labels_dir, index) -> pd.DataFrame | None:
    """The transductive pseudo-labels declared by `train.pseudo`, or None.

    `train.pseudo: <name>` reads `data/labels/<name>.csv` (probabilities in [0, 1], studies x 12 targets)
    and in every fold adds **the validation studies** to training, with these labels instead of the
    teacher. It simulates on the fold what a submission would do on the test set: new images, labelled
    by models that have not seen them. The set must therefore be an OOF (every study predicted by models
    that did not have it in training): the code cannot check that, the script building it declares it.
    Validation stays against `data.labels`, which the pseudo-labels do not touch.
    """
    name = cfg.get("train", {}).get("pseudo")
    if not name:
        return None
    if is_full_data(cfg):
        raise ValueError("train.pseudo with train.full_data: there are no studies outside training")
    pseudo = pd.read_csv(Path(labels_dir) / f"{name}.csv", index_col=0)[TARGETS]
    missing = index.difference(pseudo.index)
    if len(missing):
        raise ValueError(f"train.pseudo: {name} does not cover {len(missing)} labelled studies, "
                         f"the first is {missing[0]}")
    pseudo = pseudo.loc[index]
    if pseudo.isna().any().any() or not ((pseudo >= 0) & (pseudo <= 1)).all().all():
        raise ValueError(f"train.pseudo: {name} must be complete and in [0, 1]")
    return pseudo


def teacher_labels_for(cfg: dict, labels_dir, index) -> dict[int, pd.DataFrame] | None:
    """The training targets declared by `train.teacher` (distillation), per fold, or None.

    `train.teacher: <name>` replaces the teacher **on the training studies** of the fold; validation and
    OOF stay against `data.labels` (same yardstick, `check_metro` 0 cells). Two forms, with different
    leakage:
    - `data/labels/<name>/fold<k>.csv`: one teacher per fold. The file of fold k must come from models
      trained WITHOUT fold k: no validation study touched the student's targets. Zero leakage; the
      script building it declares it.
    - `data/labels/<name>.csv`: the same table for all folds (e.g. an OOF). Fold k is trained on
      predictions of models that had seen fold k: the leakage is second-order and inflates ONLY the
      student's OOF; the test set is not involved, the leaderboard stays clean.
    The code does not know where the file comes from: it only tells the two forms apart.
    With `train.full_data` only the single table applies: every "fold" is a seed on all studies, so
    there is no fold k to keep out of the teacher. This is how the final models were trained: the
    teacher is the soft-label table `matteo-v9j-253-h1-342` built by Matteo Vitali.
    """
    name = cfg.get("train", {}).get("teacher")
    if not name:
        return None
    if cfg["train"].get("pseudo"):
        raise ValueError("train.teacher with train.pseudo: two variables on the training labels")
    folds = [int(k) for k in cfg["train"]["folds"]]
    per_fold = Path(labels_dir) / name
    if per_fold.is_dir():
        if is_full_data(cfg):
            raise ValueError("per-fold train.teacher with train.full_data: the fold is a seed, "
                             "there is no fold to keep out of the teacher")
        paths = {k: per_fold / f"fold{k}.csv" for k in folds}
    else:
        paths = {k: Path(labels_dir) / f"{name}.csv" for k in folds}
    out = {}
    for k, path in paths.items():
        if not path.is_file():
            raise ValueError(f"train.teacher: missing {path}")
        t = pd.read_csv(path, index_col=0)[TARGETS]
        missing = index.difference(t.index)
        if len(missing):
            raise ValueError(f"train.teacher: {path.name} does not cover {len(missing)} labelled studies, "
                             f"the first is {missing[0]}")
        t = t.loc[index]
        if t.isna().any().any() or not ((t >= 0) & (t <= 1)).all().all():
            raise ValueError(f"train.teacher: {path.name} must be complete and in [0, 1]")
        out[k] = t
    return out


def is_full_data(cfg: dict) -> bool:
    """`train.full_data: true`: the final polish, not an experiment.

    Every "fold" trains on ALL labelled studies and only the seed changes (study order, window draw,
    init); the `fold<k>.pt` files are seeds on 100% of the data instead of models on 80%. There is no
    OOF: validation stays as a probe on the studies of the fold, which are INSIDE training, and
    everything derived from it (oof_*.csv, val_auc, gold, metrics.json) is in-sample and marked as such.
    The only ruler is the leaderboard. Screening and `select: best` make no sense and are refused. The
    two final models are two such seeds (42 and 43).
    """
    return bool(cfg.get("train", {}).get("full_data", False))


def init_from_of(cfg: dict) -> dict | None:
    """`train.init_from`: the fold restarts from the whole checkpoint of a finished run (encoder, context,
    attention, classifier), instead of from ImageNet (an option, not used by the final model).

    It is the "final stage" of several RSNA writeups: first training with external data, then a few
    epochs at low lr on the competition data only. `{dataset, file, run}`: the Kaggle Dataset containing
    the weights (downloaded by whoever launches, like `model.encoder_weights`), the file name with
    `{fold}` in place of the fold, and the name of the run that produced it, which loading checks. Without
    the key None and the usual path.
    """
    spec = cfg.get("train", {}).get("init_from")
    if not spec:
        return None
    missing = [k for k in ("dataset", "file", "run") if not spec.get(k)]
    if missing:
        raise ValueError(f"train.init_from without {missing}: dataset, file and run are required")
    if cfg["model"].get("encoder_weights"):
        raise ValueError("train.init_from with model.encoder_weights: the encoder would come from two "
                         "places, and init_from overwrites all of it")
    return spec


def load_init_from(model: nn.Module, cfg: dict, fold: int, weights_dir, log=print) -> None:
    """Loads `train.init_from`'s checkpoint for this fold into `model` (strict).

    It refuses a checkpoint of another fold (its training would have seen this fold's validation: the OOF
    would no longer be out of sample), of another run, or whose model, cache or mode (full-data or fold)
    differ from the config: then the variable would no longer be a single one.
    """
    spec = init_from_of(cfg)
    if weights_dir is None:
        raise ValueError(f"train.init_from={spec['dataset']}/{spec['file']} but no "
                         "weights_dir was passed to run()")
    path = Path(weights_dir) / spec["file"].format(fold=fold)
    if not path.is_file():
        raise FileNotFoundError(f"train.init_from: checkpoint not found: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    src = ck.get("cfg") or {}
    problems = []
    if ck.get("fold") != fold:
        problems.append(f"checkpoint fold {ck.get('fold')} != {fold}")
    if src.get("name") != spec["run"]:
        problems.append(f"checkpoint run {src.get('name')!r} != {spec['run']!r}")
    if src.get("model") != cfg["model"]:
        problems.append(f"model differs: {src.get('model')} vs {cfg['model']}")
    if src.get("cache") != cfg["cache"]:
        problems.append(f"cache differs: {src.get('cache')} vs {cfg['cache']}")
    if is_full_data(src) != is_full_data(cfg):
        problems.append("full_data differs between checkpoint and config")
    if problems:
        raise ValueError(f"train.init_from {path.name}: " + "; ".join(problems))
    model.load_state_dict(ck["model"])
    log(f"  fold {fold}: INITIAL WEIGHTS from {spec['run']} ({spec['dataset']}/{path.name}, "
        f"fold {ck['fold']}): the final stage restarts from the whole model, not from ImageNet")


def extra_loss_ok(cfg: dict) -> None:
    """`data.extra` applies only to the per-study loss, without mixup or the report head: the tail of `y`
    carries the meniscus target without side, which only `masked_bce_union` reads.
    """
    t = cfg.get("train", {})
    if t.get("loss", "study") != "study" or t.get("mixup") or t.get("report_distill"):
        raise ValueError("data.extra applies only to train.loss: study, without train.mixup and without "
                         "train.report_distill")


SITE_BIAS_KEYS = {"sites", "unk"}


def site_bias_for(cfg: dict, labels_dir, index) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """`train.site_bias` (an experiment): the site of every study and the cells on which its report is
    silent, or None without the key.

    The mechanism: sites (here, the language of the report) are not silent in the same way. Synovitis is
    not mentioned in 77-96% of the reports of every language, and with a reported effusion the teacher
    gives ~0.22 everywhere; the student gives 0.37 on English studies and 0.24 on Cyrillic ones, at equal
    predicted effusion (0.77): it learned from the site how often that site *names* the finding, not how
    often it is present. On the test set there is a single criterion (MSK radiologists), and that
    offset between sites is noise.

    The variable: during training, on the silent cells only, a per-(site, finding) bias that starts at
    zero and is learned together with the network is added to the logit; it does not exist at
    validation and inference. So "this site never names it" is explained by the bias instead of by the
    image, and the cells where the finding is named (yes or no) stay as they are. The bias is not in the
    model: `state_dict`, checkpoints and submission kernel are unchanged.

    `train.site_bias: {sites: <name>, unk: <name>}` reads two files from `data/labels/`: `<sites>.csv`
    (StudyInstanceUID, site) and `<unk>.csv`, the 0/1 matrix of silent cells (1 = the report is silent).
    """
    spec = cfg.get("train", {}).get("site_bias")
    if not spec:
        return None
    unknown = set(spec) - SITE_BIAS_KEYS
    if unknown or not SITE_BIAS_KEYS <= set(spec):
        raise ValueError(f"train.site_bias: expected keys {sorted(SITE_BIAS_KEYS)}, got "
                         f"{sorted(spec)}")
    t = cfg["train"]
    if t.get("loss", "study") != "study" or t.get("mixup") or t.get("report_distill") \
            or has_extra(cfg):
        raise ValueError("train.site_bias applies only to train.loss: study, without train.mixup, "
                         "train.report_distill and data.extra: the tail of y carries site and silent cells")
    labels_dir = Path(labels_dir)
    sites = pd.read_csv(labels_dir / f"{spec['sites']}.csv", index_col=0)
    if "site" not in sites.columns:
        raise ValueError(f"train.site_bias: {spec['sites']} without the 'site' column")
    sites = sites["site"].astype(str)
    unk = pd.read_csv(labels_dir / f"{spec['unk']}.csv", index_col=0)[TARGETS]
    for name, table in ((spec["sites"], sites), (spec["unk"], unk)):
        missing = index.difference(table.index)
        if len(missing):
            raise ValueError(f"train.site_bias: {name} does not cover {len(missing)} labelled studies, "
                             f"the first is {missing[0]}")
    unk = unk.loc[index]
    if not unk.isin([0, 1]).all().all():
        raise ValueError(f"train.site_bias: {spec['unk']} must be a 0/1 matrix")
    names = sorted(sites.loc[index].unique())
    code = {n: i for i, n in enumerate(names)}
    idx = sites.loc[index].map(code).to_numpy(dtype=np.int64)
    return idx, unk.to_numpy(dtype=np.float32), names


def train_fold(cfg, cache, labels, folds, fold, device, log=print, weights_dir=None,
               gold_path=None, cell_weights: pd.DataFrame | None = None,
               pseudo: pd.DataFrame | None = None, teacher: pd.DataFrame | None = None,
               extra: pd.DataFrame | None = None,
               site_bias: tuple[np.ndarray, np.ndarray, list[str]] | None = None,
               aux_findings: pd.DataFrame | None = None):
    full = is_full_data(cfg)
    tr_uids = list(labels.index) if full else [u for u in labels.index if folds.get(u, -1) != fold]
    va_uids = [u for u in labels.index if folds.get(u, -1) == fold]
    labels_tr = labels
    if teacher is not None:
        # The training studies with the targets of the distilled teacher. The cells `data.labels` excludes (NaN)
        # stay excluded: the value changes, not the mask.
        labels_tr = labels.copy()
        t = teacher.loc[tr_uids].where(labels.loc[tr_uids].notna())
        labels_tr.loc[tr_uids] = t.to_numpy()
        log(f"  fold {fold}: DISTILLATION, the {len(tr_uids)} training studies with the targets of "
            f"train.teacher={cfg['train']['teacher']} (val against data.labels)")
    if pseudo is not None:
        # Transductive: the validation studies enter training with the pseudo-labels, not with the teacher.
        # `labels` stays intact: it is the yardstick of validation and OOF.
        labels_tr = labels.copy()
        labels_tr.loc[va_uids] = pseudo.loc[va_uids].to_numpy()
        tr_uids = tr_uids + va_uids
        log(f"  fold {fold}: TRANSDUCTIVE, the {len(va_uids)} validation studies in training with "
            f"the pseudo-labels of train.pseudo={cfg['train']['pseudo']} (val against data.labels)")
    if full:
        log(f"  fold {fold}: FULL-DATA, {len(tr_uids)} studies in training (all); the fold is the seed "
            f"({cfg['seed']} + {fold}); the {len(va_uids)} 'val' studies are INSIDE training: "
            "an in-sample probe, not an OOF")
    gold_weight = float(cfg.get("train", {}).get("gold_weight", 1.0))
    gold_uids = set(load_gold(gold_path).index) if gold_weight != 1.0 else set()
    n_gold_tr = len(set(tr_uids) & gold_uids)
    if gold_weight != 1.0:
        log(f"  fold {fold}: {len(tr_uids)} train ({n_gold_tr} gold, weight={gold_weight:g}) / {len(va_uids)} val")
    else:
        log(f"  fold {fold}: {len(tr_uids)} train / {len(va_uids)} val")
    if cell_weights is not None:
        cw = cell_weights.loc[tr_uids].to_numpy()
        spec = cfg["train"]["cell_weight"]
        log(f"  fold {fold}: per-cell weight {spec['weight']:g} on {int((cw != 1.0).sum())} cells "
            f"({(cw != 1.0).mean():.1%} of train, mask {spec['mask']})")
    weighted = gold_weight != 1.0 or cell_weights is not None
    # The per-column positive weight, counted on the training targets of this fold (after
    # `train.teacher`, before the extra rows, which do not go with the weight).
    pw = pos_weight_of(cfg, labels_tr.loc[tr_uids])
    pw_t = None if pw is None else torch.tensor(pw, dtype=torch.float32, device=device)
    if pw is not None:
        log(f"  fold {fold}: POSITIVE WEIGHT (train.pos_weight, power "
            f"{float(cfg['train']['pos_weight'].get('power', 0.5)):g}): "
            f"{ {c: round(float(v), 3) for c, v in zip(TARGETS, pw) if v != 1.0} }; the bce in the "
            "log is the unweighted one, comparable with the plain model")
    # The contrast between OA compartments, added to the BCE.
    cp_spec = compartment_pair_spec_of(cfg)
    if cp_spec is not None:
        log(f"  fold {fold}: COMPARTMENT CONTRAST (train.compartment_pair, weight "
            f"{cp_spec['weight']:g}, |dy| >= {cp_spec['min_diff']:g}) on "
            f"{[f'{a}/{b}' for a, b in COMPARTMENT_PAIRS]}; the bce in the log is the plain one, "
            "comparable with the plain model")
    # Gold-58 per epoch (read-out, never selection): the gold studies falling in the validation fold, 7-16
    # per fold. The per-fold number is noisy (in fold 0 MCL has 2 positives): what is read is the curve
    # over the merged folds, `gold58_curve` in metrics.json. The predictions of these studies at every
    # validation go into `gold_curve_fold<k>.csv`, because the curve over the 58 can only be rebuilt from
    # there.
    gold_va = None
    if gold_path is not None:
        gold_all = load_gold(gold_path)
        gold_va = gold_all.loc[gold_all.index.intersection(va_uids)]
        if gold_va.empty:
            gold_va = None

    # The shuffle generator is explicit, not the global one: so the study order does not depend on how
    # many other RNG consumers there are in the loop. Otherwise evaluating mid-training would change
    # training: a small effect, but free to avoid.
    g_tr = torch.Generator().manual_seed(cfg["seed"] * 1000 + fold)
    gen = torch.Generator().manual_seed(cfg["seed"] + fold)      # the window draw
    stride = cfg["model"].get("window_stride", 1)
    depth = cfg["model"].get("window_depth", GROUP)
    n_win = cache.spec.shape[0] * windows_per_slot(cache.spec.slices, stride, depth)
    k_train = cfg["train"]["k_train"]
    # Report distillation: the auxiliary target is estimated HERE, on this fold's `tr_uids` only (centre,
    # nuisance directions, teacher and language coefficients, PCA). The frozen artifact carries only what
    # depends on the single report: a PCA estimated on all 4,407 studies would bring this fold's validation
    # into training.
    rep_spec, rep_target = report_spec_of(cfg), None
    if rep_spec is not None:
        emb, langs, emb_meta = load_report_embeddings(rep_spec["emb"])
        fit = fit_report_targets(emb, labels, langs, tr_uids, dim=rep_spec["dim"],
                                 drop_pcs=rep_spec["drop_pcs"])
        rep_target = fit.transform(emb, labels, langs, tr_uids)
        log(f"  fold {fold}: report distillation ({emb_meta['model'].split('/')[-1]}, "
            f"{emb.shape[1]}d -> {rep_spec['dim']}d, weight {rep_spec['weight']:g}): "
            f"residual variance kept {fit.explained:.3f}, eta2 by language "
            f"{fit.language_eta2:.4f} (raw {emb_meta['language_eta2_raw']:.4f})")
    # `data.extra`: the rows of another cache, appended to the training studies and never in validation.
    # With their targets (NaN where the source is silent) and, at the end of `y`, two columns: the meniscus
    # without side (`UNION`, NaN outside MRNet) and the "extra row" flag, used only for the log: the
    # per-epoch `bce` stays that of the RSNA studies only, comparable with the plain model.
    aux = None
    if extra is not None:
        extra_loss_ok(cfg)
        clash = extra.index.intersection(labels.index)
        if len(clash):
            raise ValueError(f"data.extra: {len(clash)} IDs already among the RSNA studies, the first is "
                             f"{clash[0]}")
        n_main = len(tr_uids)
        labels_tr = pd.concat([labels_tr, extra[TARGETS]])
        tr_uids = tr_uids + list(extra.index)
        aux = np.full((len(tr_uids), 2), np.nan, dtype=np.float32)
        aux[:, 1] = 0.0
        aux[n_main:, 1] = 1.0
        if UNION in extra.columns:
            aux[n_main:, 0] = extra[UNION].to_numpy(dtype=np.float32)
        fin = np.isfinite(extra[TARGETS].to_numpy())
        log(f"  fold {fold}: EXTRA ROWS, {len(extra)} data.extra studies in training after the "
            f"{n_main} of the fold (finite cells {int(fin.sum())}, of which positive "
            f"{int((extra[TARGETS].to_numpy() == 1).sum())}; meniscus without side "
            f"{int(np.isfinite(aux[n_main:, 0]).sum())}); validation intact")
    # `train.site_bias`: at the end of `y` of the TRAINING studies the site and the 12 silent cells (same
    # form as `aux`, which is absent here: `site_bias_for` excludes it). The per-(site, finding) bias is a
    # parameter of the fold, not of the model; validation does not see it.
    sb_tail = sb_param = sb_names = None
    if site_bias is not None:
        if aux is not None or rep_target is not None:
            raise ValueError("train.site_bias with data.extra or train.report_distill: the tail of y "
                             "would be ambiguous")
        s_idx, s_unk, sb_names = site_bias
        pos = labels.index.get_indexer(tr_uids)
        if (pos < 0).any():
            raise ValueError("train.site_bias: training studies outside the site table")
        sb_tail = np.concatenate([s_idx[pos, None].astype(np.float32), s_unk[pos]], axis=1)
        sb_param = nn.Parameter(torch.zeros(len(sb_names), len(TARGETS), device=device))
        log(f"  fold {fold}: PER-SITE BIAS on the silent cells (train.site_bias), {len(sb_names)} sites "
            f"{sb_names}; silent cells {int(s_unk[pos].sum())} ({s_unk[pos].mean():.1%} of train); "
            "training only, validation and inference without it")
    # `train.aux_findings`: at the end of `y` of the TRAINING studies the K auxiliary findings (NaN where
    # the report does not mention them). Validation does not see them; the head predicting them does not
    # enter the checkpoint.
    find_spec, find_tail = aux_findings_spec_of(cfg), None
    if aux_findings is not None:
        if find_spec is None:
            raise ValueError("train_fold: aux_findings passed without train.aux_findings in the config")
        if aux is not None or rep_target is not None or sb_tail is not None:
            raise ValueError("train.aux_findings with another tail of y (data.extra, "
                             "report_distill, site_bias): it would be ambiguous")
        find_tail = aux_findings.loc[tr_uids].to_numpy(dtype=np.float32)
        fin = np.isfinite(find_tail)
        log(f"  fold {fold}: AUXILIARY FINDINGS (train.aux_findings={find_spec['labels']}, weight "
            f"{find_spec['weight']:g}): {find_tail.shape[1]} findings, cells mentioned {fin.mean():.1%}, "
            f"positive (>= 0.5) {int((find_tail[fin] >= 0.5).sum())}; training only, the head "
            "does not enter the checkpoint")
    # Window reading (`data/windows.py`): only the `k_train` sampled windows arrive from disk, not the whole
    # study. Same generators, same consumption order, same tensors as the plain path, which remains for
    # when no sampling happens.
    windowed = use_window_loader(k_train, n_win)
    # `train.augment` is read here, before the dataset: with `slice_offset` the loader reads
    # `depth + slice_offset` slices per window. Its generator is created further down, as always.
    aug = AugmentSpec.from_config(cfg)
    window_extra = augment_window_extra(aug, aux is not None, windowed, cache.spec.slices,
                                        stride, depth)
    ds_tr = KneeStudies(cache, tr_uids, labels_tr, gold_uids=gold_uids, gold_weight=gold_weight,
                        cell_weights=cell_weights, window_stride=stride, window_depth=depth,
                        report=rep_target, aux=aux if aux is not None else sb_tail,
                        window_extra=window_extra, findings=find_tail)
    # WHICH k windows per batch. Without the key, "uniform" = the usual `randperm`; "stratified" spreads
    # them across the slots and along the stack (`stratified_pick`), same `gen`, same study order.
    sampling = cfg["train"].get("window_sampling", "uniform")
    if sampling != "uniform" and not windowed:
        raise ValueError(f"train.window_sampling {sampling!r} without sampling (k_train {k_train} "
                         f"of {n_win} windows)")
    if sampling != "uniform":
        log(f"  fold {fold}: sampled windows {sampling} ({k_train} of {n_win})")
    if windowed:
        dl_tr = DataLoader(ds_tr, batch_sampler=WindowBatchSampler(
                               len(ds_tr), cfg["train"]["batch_size"], n_win, k_train, g_tr, gen,
                               sampling=sampling,
                               per=windows_per_slot(cache.spec.slices, stride, depth)),
                           generator=g_tr, num_workers=cfg["train"]["workers"], pin_memory=True)
    else:
        dl_tr = DataLoader(ds_tr, shuffle=True, drop_last=True,
                           batch_size=cfg["train"]["batch_size"], generator=g_tr,
                           num_workers=cfg["train"]["workers"], pin_memory=True)
    speed = speed_of(cfg)
    # `train.speed.val_rng: isolated`: the `_base_seed` of validation from its own generator instead of the
    # global RNG (see `speed_of`). Without the key, the usual path.
    g_va = ({"generator": torch.Generator().manual_seed(cfg["seed"] * 2000 + fold)}
            if speed["val_rng"] == "isolated" else {})
    dl_va = DataLoader(KneeStudies(cache, va_uids, labels), shuffle=False,
                       batch_size=cfg["train"]["eval_batch_size"],
                       num_workers=cfg["train"]["workers"], pin_memory=True, **g_va)

    model = KneeBaseline(arch=cfg["model"]["arch"], n_slots=cache.spec.shape[0],
                         n_slices=cache.spec.slices, pretrained=cfg["model"]["pretrained"],
                         drop_rate=cfg["model"]["drop_rate"],
                         img_size=cache.spec.img,
                         slot_prior_strength=cfg["model"].get("slot_prior_strength", 0.0),
                         slot_prior_mask_at=cfg["model"].get("slot_prior_mask_at"),
                         encoder_weights=resolve_encoder_weights(cfg, weights_dir),
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
                         report_dim=(rep_spec["dim"] if rep_spec else 0),
slice_attn_layers=cfg["model"].get("slice_attn_layers", 0),
                         slice_attn_dim=cfg["model"].get("slice_attn_dim", 256),
                         slice_attn_heads=cfg["model"].get("slice_attn_heads", 4),
                                                  input_norm=cfg["model"].get("input_norm"),
                         grad_checkpointing=bool(cfg["train"].get("grad_checkpointing", False)),
                         aux_targets=(find_tail.shape[1] if find_tail is not None else 0),
                         compartment_route=cfg["model"].get("compartment_route"),
                         finding_layers=cfg["model"].get("finding_layers", 0),
                         finding_dim=cfg["model"].get("finding_dim", 256),
                         finding_heads=cfg["model"].get("finding_heads", 4),
                         drop_path_rate=cfg["model"].get("drop_path_rate", 0.0),
                         vit_head=cfg["model"].get("vit_head"),
                         img_crop=cfg["model"].get("img_crop")
                         ).to(device)
    # `train.init_from`: the starting weights from the whole checkpoint of a finished run. Here, before
    # `channels_last` and the EMA shadows, which must start from the loaded weights.
    if init_from_of(cfg) is not None:
        load_init_from(model, cfg, fold, weights_dir, log)
    if getattr(model, "finding_layers", 0) > 0:
        log(f"  fold {fold}: ATTENTION ACROSS FINDINGS (model.finding_layers "
            f"{model.finding_layers}): the 12 per-finding vectors go through a transformer "
            "before the classifier, output zero at the first step")
    # The window geometry, in the log of every fold: a run that gets it wrong runs to the end without
    # saying so.
    assert (model.window_stride, model.window_depth) == (stride, depth)
    # `train.speed.channels_last` HERE, before the EMA shadows: `ModelEma` keeps the tensors of the
    # state_dict, and `.to(memory_format=...)` replaces `param.data` of the 4D weights. The model is
    # enough: with NHWC weights cudnn also brings the input of the first conv to NHWC.
    if speed["channels_last"]:
        if not channels_last_ok([model]):
            raise ValueError("train.speed.channels_last with a video encoder: NHWC is a "
                             "4D layout, 3D weights are 5D")
        model = model.to(memory_format=torch.channels_last)
    # `train.speed.fastdense`: DenseNet training without concatenations and with fused kernels
    # (`knee.fastdense`). It replaces only the forward of the encoder instance in training: same
    # state_dict, same EMA shadows, validation and inference with timm's forward.
    if speed["fastdense"]:
        from .fastdense import attach
        attach(model, log, conv=None if speed["fastdense"] is True else speed["fastdense"],
               graphs=speed["graphs"], fast_eval=speed["fast_eval"])
    # Validation in blocks of one study for the encoders that are not DenseNets: same numbers, memory peak
    # divided by the evaluation batch.
    model.encode_chunk = encode_chunk_for(cfg["model"]["arch"], n_win)
    if cfg["model"].get("encoder_weights"):
        log(f"  fold {fold}: encoder initialised from {cfg['model']['encoder_weights']} "
            f"({cfg['model'].get('encoder_weights_dataset')})")
    if model.input_norm or model.grad_checkpointing or model.encode_chunk:
        log(f"  fold {fold}: input_norm {model.input_norm}, grad_checkpointing "
            f"{model.grad_checkpointing}, validation in blocks of {model.encode_chunk} windows")
    # EMA shadows (`train.ema`): updated after every effective step, they never touch gradients or
    # generators, so the `last` weights stay bit-identical to a run without the key. Read at the end of the
    # fold, next to the last epoch (see `ema.py`).
    emas = [ModelEma(model, d) for d in ema_decays(cfg)]
    if emas:
        log(f"  fold {fold}: EMA of the weights, decays {[e.decay for e in emas]} "
            f"(read-out at the end of the fold, not selection)")
    # `train.speed.compile` after the shadows (which stay eager: they are evaluated once at the end of the
    # fold). In place, so the state_dict keeps its names. Shapes: the training batch is fixed
    # (`drop_last`, `k_train`), validation has two shapes (full batch and tail): three graphs, compiled
    # the first time they are seen and then taken from dynamo's cache.
    if speed["compile"]:
        for enc in {id(e): e for e in (model.encoder,
                                       *getattr(model, "plane_encoders", {}).values())}.values():
            enc.compile(mode="default", dynamic=False)
    if speed["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    if speed != SPEED_DEFAULTS:
        log(f"  fold {fold}: train.speed " + ", ".join(
            f"{k} {v}" for k, v in speed.items() if v != SPEED_DEFAULTS[k])
            + " (infrastructure: not bit-identical to the fp16 eager path)")
    log(f"fold {fold}: {cache.spec.shape[0]} slots x {cache.spec.slices} slices -> {n_win} windows "
        f"per study (depth {model.window_depth}, stride {model.window_stride})")
    mb = cache.spec.bytes_per_study / 1e6
    if windowed:
        mb_w = k_train * (depth + window_extra) * cache.spec.img ** 2 / 1e6
        log(f"fold {fold}: window reading, {k_train} per study ({mb_w:.1f} MB instead of {mb:.1f})"
            + (f", {window_extra} extra slices per window (train.augment.slice_offset)"
               if window_extra else ""))
    else:
        log(f"fold {fold}: whole-study reading ({mb:.1f} MB), all {n_win} windows")
    slot_all = window_slot_index(cache.spec.shape[0], cache.spec.slices, device, stride, depth)
    # The global index of every window, drawn together with them: `slot_all[idx]` is the usual slot, and
    # with `model.context_pos: window` the context derives from it the position along the slices instead of
    # the place in the draw. Same `gen`, same `randperm`: windows and slots bit for bit as in the plain
    # path.
    win_all = torch.arange(len(slot_all), device=device)
    if model.context_pos is not None:
        log(f"  fold {fold}: context with the window position in the slot "
            f"(model.context_pos: {model.context_pos}), not its place in the draw")
    if model.compartment_route is not None:
        log(f"  fold {fold}: COMPARTMENTS (model.compartment_route: {model.compartment_route}): "
            "Medial OA and Lateral OA read only their side (half of the coronal and "
            "axial images, side of the sagittal stack, context of the same side)")
    lr, lr_bb = cfg["train"]["lr"], cfg["train"].get("lr_backbone")
    fused = {"fused": True} if speed["fused_adam"] and device.type == "cuda" else {}
    # Matteo Vitali's ViT recipe: LLRD, LP phase, cosine with a floor, SWA. All new keys: without any of
    # them the usual path (OneCycle, plain groups) bit for bit.
    llrd = cfg["train"].get("llrd")
    lp = lp_spec_of(cfg)
    sched_kind = cfg["train"].get("schedule", "onecycle")
    swa_spec = cfg["train"].get("swa")
    if swa_spec and set(swa_spec) != {"start_frac"}:
        raise ValueError(f"train.swa: only the start_frac key is expected, got {sorted(swa_spec)}")
    vit_recipe = llrd is not None or lp is not None or sched_kind != "onecycle" or bool(swa_spec)
    if vit_recipe:
        if lr_bb is None or llrd is None:
            raise ValueError("train.lp/schedule/swa need the ViT recipe: lr_backbone and llrd")
        if sb_param is not None:
            raise ValueError("the ViT recipe with train.site_bias is not defined")
        if sched_kind not in ("onecycle", "cosine"):
            raise ValueError(f"train.schedule: expected 'onecycle' or 'cosine', got {sched_kind!r}")
        if (lp is not None or swa_spec) and sched_kind != "cosine":
            raise ValueError("train.lp and train.swa need train.schedule: cosine (the floor)")
    lp_epochs = lp["epochs"] if lp is not None else 0
    ft_epochs = cfg["train"]["epochs"]
    wd = cfg["train"]["weight_decay"]

    def vit_optimizer(stage: str):
        """AdamW and scheduler of one phase of the ViT recipe: "lp" (head only) or "ft"."""
        if stage == "lp":
            groups = vit_param_groups(model, lp["lr"], None, llrd, wd)
            steps, floor = lp_epochs * len(dl_tr), lp["min_frac"]
        else:
            groups = vit_param_groups(model, lr, lr_bb, llrd, wd)
            steps = ft_epochs * len(dl_tr)
            floor = float(cfg["train"].get("lr_min_frac", 0.15))
        o = torch.optim.AdamW(groups, lr=groups[0]["lr"], weight_decay=wd, **fused)
        if sched_kind == "cosine":
            s_ = torch.optim.lr_scheduler.LambdaLR(o, warmup_cosine(
                max(1, steps), float(cfg["train"].get("warmup_frac", 0.1)), floor))
        else:
            s_ = torch.optim.lr_scheduler.OneCycleLR(
                o, max_lr=[g["lr"] for g in groups], total_steps=max(1, steps), pct_start=0.1)
        # The peaks (`initial_lr`): the scheduler has already brought `lr` to the value of step 0.
        pk = [g.get("initial_lr", g["lr"]) for g in groups]
        enc_g = [(g, k) for g, k in zip(groups, pk) if g["name"].startswith("enc")]
        log(f"  fold {fold}: ViT recipe, phase {stage}: head {pk[0]:g}"
            + (f", maps without decay ({sum(p.numel() for p in groups[1]['params'])} parameters)"
               if len(groups) > 1 and groups[1]["name"] == "mappe" else "")
            + (f", encoder LLRD {llrd:g} from {enc_g[0][1]:.3g} ({enc_g[0][0]['name']}) to "
               f"{enc_g[-1][1]:.3g} ({enc_g[-1][0]['name']}), {len(enc_g)} groups (peaks)" if enc_g
               else ", frozen encoder")
            + f"; {sched_kind} over {steps} steps")
        return o, s_

    if vit_recipe:
        if lp is not None:
            for p_ in model.encoder.parameters():
                p_.requires_grad_(False)
        opt, sched = vit_optimizer("lp" if lp is not None else "ft")
    elif lr_bb is None:
        nd = [p_ for n_, p_ in model.named_parameters() if _intra_no_decay(n_)]
        if nd:
            # The per-finding maps of the intra head without weight decay (as in `vit_param_groups`). Without the
            # intra head no extra group: the usual AdamW.
            ids = {id(p_) for p_ in nd}
            groups = [{"params": [p_ for p_ in model.parameters() if id(p_) not in ids]},
                      {"params": nd, "weight_decay": 0.0}]
            log(f"  intra head: maps without decay ({sum(p_.numel() for p_ in nd)} parameters)")
        else:
            groups = model.parameters()
        opt = torch.optim.AdamW(groups, lr=lr,
                                weight_decay=cfg["train"]["weight_decay"], **fused)
        max_lr = lr
    else:
        if any(_intra_no_decay(n_) for n_, _ in model.named_parameters()):
            raise ValueError("intra head with train.lr_backbone outside the ViT recipe: not defined")
        groups = split_backbone_head(model, lr, lr_bb)
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=cfg["train"]["weight_decay"], **fused)
        max_lr = [g["lr"] for g in groups]
        log(f"  lr: encoder {lr_bb:g} ({len(groups[0]['params'])} tensors), "
            f"head {lr:g} ({len(groups[1]['params'])} tensors)")
    if sb_param is not None:
        # A group of its own, at the head's lr and without weight decay (the bias must be free to stay where
        # the data take it). AdamW updates every parameter independently: the model weights receive the same
        # steps as without this group, only the loss changes.
        opt.add_param_group({"params": [sb_param], "lr": lr, "weight_decay": 0.0})
        if isinstance(max_lr, list):
            max_lr = max_lr + [lr]
    epochs = lp_epochs + ft_epochs
    if not vit_recipe:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, total_steps=max(1, epochs * len(dl_tr)), pct_start=0.1)
    swa = SwaAverage(swa_spec["start_frac"]) if swa_spec else None
    if swa is not None:
        log(f"  fold {fold}: SWA of the weights from fine-tuning epoch {swa.starts_at(ft_epochs) + 1}/"
            f"{ft_epochs}, one snapshot per epoch; the fold prediction is that of the SWA weights")
    amp = cfg["train"]["amp"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if speed["amp_dtype"] == "bf16" else torch.float16
    if amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("train.speed.amp_dtype bf16: this GPU has no bf16 (a T4?); A100/L4 do")
    val_every = cfg["train"].get("val_every", 0)
    select = cfg["train"].get("select", "last")
    loss_mode = loss_mode_of(cfg, model.pooling)
    if loss_mode == "window":
        log(f"  fold {fold}: per-window loss (train.loss: window): every valid window "
            "answers for the study target, weight 1/n_valid per study; the bce in the log "
            "is per window and not comparable with the per-study one")
    if loss_mode == "rank":
        log(f"  fold {fold}: pairwise ranking loss (train.loss: rank): rank_loss is "
            "optimised; the bce in the log is the per-study BCE computed separately, without "
            "gradient, and is comparable with the plain model")
    best = {"auc": -1.0, "ep": 0, "state": None}
    # In bf16 the scaler is not needed (the exponent is fp32's) and stays off: the scale is fixed at 1.0,
    # no step is skipped and `sched.step()` advances at every batch.
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    # Augmentation has its OWN generator: if it consumed `gen`, a run with `train.augment` would see
    # different windows from one without, and the variable would no longer be a single one. Without
    # `train.augment` in the config `aug` is None and nothing is touched (`aug` is read above, before the
    # dataset).
    gen_aug = torch.Generator().manual_seed(cfg["seed"] * 7919 + fold) if aug else None
    if aug:
        log(f"  fold {fold}: per-window augment ({aug.describe()}), no flips")
    # Mixup also has its own generator, for the same reason: the sampled windows and the study order stay
    # those of the run without the key.
    mix = mixup_spec_of(cfg, loss_mode, rep_spec)
    rng_mix = np.random.default_rng(cfg["seed"] * 7927 + fold) if mix else None
    if mix:
        log(f"  fold {fold}: MIXUP between studies ({mix.describe()}); the bce in the log is on the mixed "
            "targets and not comparable with the plain model")

    history, gold_frames, val_frames = [], [], []
    last_val = None             # (epoch, predictions, seconds) of the last validation
    for ep in range(epochs):
        if lp is not None and ep == lp_epochs:
            # End of the LP phase: the encoder is unfrozen and the FT phase starts with a new AdamW.
            for p_ in model.encoder.parameters():
                p_.requires_grad_(True)
            opt, sched = vit_optimizer("ft")
        model.train()
        if lp is not None and ep < lp_epochs:
            model.encoder.eval()        # frozen = deterministic feature extractor (no drop_path)
        t0, tot, tot_bce, tot_rep, n = time.time(), 0.0, 0.0, 0.0, 0
        tot_aux = 0.0                                   # the loss of the auxiliary findings
        n_bce, tot_ex, n_ex = 0, 0.0, 0                 # `data.extra`: RSNA studies / extra rows
        for batch in dl_tr:
            if windowed:
                vol, mask, y, weight, pick = batch
                mask, y = mask.to(device, non_blocking=True), y.to(device, non_blocking=True)
                w = vol.to(device, non_blocking=True).float().div_(255.0)   # (B, k, depth, H, W)
                win_of = win_all[pick[0].to(device)]                        # the same pick for the whole batch
                slot_of = slot_all[win_of]
            else:
                vol, mask, y, weight = batch
                vol = vol.to(device, non_blocking=True)
                mask, y = mask.to(device, non_blocking=True), y.to(device, non_blocking=True)
                w = make_windows(vol.float().div_(255.0), model.window_stride, model.window_depth)
                w, win_of = _sample_windows(w, win_all, k_train, gen)
                slot_of = slot_all[win_of]
            # Report distillation: the auxiliary target travels at the end of `y` (dataset.KneeStudies).
            rep_t = None
            if rep_spec is not None:
                y, rep_t = y[:, :len(TARGETS)], y[:, len(TARGETS):]
            # The auxiliary findings, same tail.
            find_t = None
            if find_tail is not None:
                y, find_t = y[:, :len(TARGETS)], y[:, len(TARGETS):]
            # `data.extra`: at the end of `y` the meniscus without side and the "extra row" flag.
            y_union = is_extra = None
            if aux is not None:
                y, tail = y[:, :len(TARGETS)], y[:, len(TARGETS):]
                y_union, is_extra = tail[:, 0], tail[:, 1] > 0.5
            # `train.site_bias`: at the end of `y` the site and the silent cells of the study.
            sb_site = sb_unk = None
            if sb_param is not None:
                y, tail = y[:, :len(TARGETS)], y[:, len(TARGETS):]
                sb_site, sb_unk = tail[:, 0].long(), tail[:, 1:]
            # After the tail of `y` because `rows: extra` perturbs only the `data.extra` rows: the others come out
            # with the first `depth` slices of every window, bit for bit. With the plain recipe it is the usual
            # call (`augment_windows`).
            if aug:
                w = apply_augment(w, aug, gen_aug, depth,
                                  rows=is_extra if aug.rows == "extra" else None)
            w_loss = weight.to(device, non_blocking=True) if weighted else None
            if mix:
                w, mask, y, w_loss, _ = mixup_batch(w, mask, y, w_loss, mix, rng_mix)
            # Read only by `context_pos`, `slice_attn` (the position along the slices) and `compartment_route`
            # (the side of the sagittal stack): without any of the three, None keeps the usual call.
            wi = (win_of if model.context_pos is not None
                  or getattr(model, "slice_attn_layers", 0) > 0
                  or getattr(model, "compartment_route", None) is not None else None)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                if loss_mode == "window":
                    lw, valid = model.window_logits(w, slot_of, mask, window_index=wi)
                    loss = window_bce(lw, valid, y, w_loss)
                elif loss_mode == "rank":
                    logits = model(w, slot_of, mask, window_index=wi)
                    loss = rank_loss(logits, y, w_loss)
                    # The BCE stays in the log as a read-out (the marker of teacher memorisation): computed on the same
                    # logits, without gradient.
                    tot_bce += float(masked_bce(logits.detach().float(), y)) * len(y)
                elif rep_spec is not None:
                    logits, rep_p = model(w, slot_of, mask, return_report=True, window_index=wi)
                    loss = masked_bce(logits, y, w_loss)
                    # The auxiliary head is ADDED to the BCE: the variable of the run is the extra term, not one loss
                    # replacing another. `tot_bce` keeps the BCE alone, which is what compares with the plain model.
                    tot_bce += float(loss.detach()) * len(y)
                    rep_l = report_loss(rep_p, rep_t)
                    tot_rep += float(rep_l.detach()) * len(y)
                    loss = loss + rep_spec["weight"] * rep_l
                elif find_t is not None:
                    logits, aux_logits = model(w, slot_of, mask, return_aux=True, window_index=wi)
                    loss = masked_bce(logits, y, w_loss)
                    # As for the report head: the auxiliary-findings term is ADDED to the BCE of the 12 columns, and
                    # `tot_bce` keeps the BCE alone. NaN cells ("the report does not mention it") stay out of
                    # `masked_bce`.
                    tot_bce += float(loss.detach()) * len(y)
                    aux_l = masked_bce(aux_logits.float(), find_t)
                    tot_aux += float(aux_l.detach()) * len(y)
                    loss = loss + find_spec["weight"] * aux_l
                elif aux is not None:
                    logits = model(w, slot_of, mask, window_index=wi)
                    loss = masked_bce_union(logits, y, y_union, w_loss)
                    # Read-outs without gradient: the BCE of the RSNA studies only (the memorisation marker, comparable with
                    # the plain model) and the loss of the extra rows.
                    with torch.no_grad():
                        lf, main_rows = logits.detach().float(), ~is_extra
                        if main_rows.any():
                            k = int(main_rows.sum())
                            tot_bce += float(masked_bce(lf[main_rows], y[main_rows])) * k
                            n_bce += k
                        if is_extra.any():
                            k = int(is_extra.sum())
                            tot_ex += float(masked_bce_union(lf[is_extra], y[is_extra],
                                                             y_union[is_extra])) * k
                            n_ex += k
                elif sb_param is not None:
                    logits = model(w, slot_of, mask, window_index=wi)
                    # The site bias only where the report is silent; the BCE without bias stays in the log as a read-out
                    # (comparable with the plain model), the one with bias is optimised.
                    loss = masked_bce(logits.float() + sb_param[sb_site] * sb_unk, y, w_loss)
                    tot_bce += float(masked_bce(logits.detach().float(), y)) * len(y)
                elif pw_t is not None:
                    # The BCE with positive weights is optimised; the unweighted BCE stays in the log as a read-out,
                    # comparable with the plain model.
                    logits = model(w, slot_of, mask, window_index=wi)
                    loss = masked_bce(logits, y, w_loss, pos_weight=pw_t)
                    tot_bce += float(masked_bce(logits.detach().float(), y)) * len(y)
                elif cp_spec is not None:
                    # The contrast is ADDED to the BCE; `tot_bce` keeps the BCE alone.
                    logits = model(w, slot_of, mask, window_index=wi)
                    loss = masked_bce(logits, y, w_loss)
                    tot_bce += float(loss.detach()) * len(y)
                    cp_l = compartment_pair_loss(logits, y, cp_spec["min_diff"])
                    tot_aux += float(cp_l.detach()) * len(y)
                    loss = loss + cp_spec["weight"] * cp_l
                else:
                    loss = masked_bce(model(w, slot_of, mask, window_index=wi), y, w_loss)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scale_before <= scaler.get_scale():
                sched.step()
                for e in emas:
                    e.update()
            tot += float(loss.detach()) * len(y)
            n += len(y)
        if loss_mode == "rank":
            msg = (f"  fold {fold} ep {ep + 1}/{epochs}  rank {tot / max(n, 1):.4f}"
                   f"  bce {tot_bce / max(n, 1):.4f}")
            rec = {"epoch": ep + 1, "bce": tot_bce / max(n, 1), "rank": tot / max(n, 1)}
        elif rep_spec is not None:
            # `bce` is the BCE alone (comparable with the plain model), `rep` the cosine of the auxiliary head (1.0
            # = learned nothing), `loss` the sum actually optimised.
            msg = (f"  fold {fold} ep {ep + 1}/{epochs}  bce {tot_bce / max(n, 1):.4f}"
                   f"  rep {tot_rep / max(n, 1):.4f}  loss {tot / max(n, 1):.4f}")
            rec = {"epoch": ep + 1, "bce": tot_bce / max(n, 1),
                   "rep": tot_rep / max(n, 1), "loss": tot / max(n, 1)}
        elif find_tail is not None:
            # `bce` the 12 columns alone (comparable with the plain model), `aux` the BCE of the auxiliary findings
            # on the mentioned cells, `loss` the optimised sum.
            msg = (f"  fold {fold} ep {ep + 1}/{epochs}  bce {tot_bce / max(n, 1):.4f}"
                   f"  aux {tot_aux / max(n, 1):.4f}  loss {tot / max(n, 1):.4f}")
            rec = {"epoch": ep + 1, "bce": tot_bce / max(n, 1),
                   "aux": tot_aux / max(n, 1), "loss": tot / max(n, 1)}
        elif aux is not None:
            # `bce` on the RSNA studies only (comparable with the plain model), `bce_extra` on the `data.extra`
            # rows (cells and union), `loss` the one optimised on all of them.
            msg = (f"  fold {fold} ep {ep + 1}/{epochs}  bce {tot_bce / max(n_bce, 1):.4f}"
                   f"  extra {tot_ex / max(n_ex, 1):.4f}  loss {tot / max(n, 1):.4f}")
            rec = {"epoch": ep + 1, "bce": tot_bce / max(n_bce, 1),
                   "bce_extra": tot_ex / max(n_ex, 1), "loss": tot / max(n, 1)}
        elif sb_param is not None:
            # `bce` without the bias (comparable with the plain model), `loss` the optimised one, `|bias|` the mean
            # of the per-(site, finding) biases: how much the site explains the silence.
            b_abs = float(sb_param.detach().abs().mean())
            msg = (f"  fold {fold} ep {ep + 1}/{epochs}  bce {tot_bce / max(n, 1):.4f}"
                   f"  loss {tot / max(n, 1):.4f}  |bias| {b_abs:.3f}")
            rec = {"epoch": ep + 1, "bce": tot_bce / max(n, 1), "loss": tot / max(n, 1),
                   "site_bias_abs": b_abs}
        elif pw_t is not None:
            # `bce` unweighted (comparable with the plain model), `loss` the optimised one.
            msg = (f"  fold {fold} ep {ep + 1}/{epochs}  bce {tot_bce / max(n, 1):.4f}"
                   f"  loss {tot / max(n, 1):.4f}")
            rec = {"epoch": ep + 1, "bce": tot_bce / max(n, 1), "loss": tot / max(n, 1)}
        elif cp_spec is not None:
            # `bce` alone (comparable with the plain model), `pair` the compartment contrast (log 2 = 0.693 with
            # equal logits), `loss` the optimised sum.
            msg = (f"  fold {fold} ep {ep + 1}/{epochs}  bce {tot_bce / max(n, 1):.4f}"
                   f"  pair {tot_aux / max(n, 1):.4f}  loss {tot / max(n, 1):.4f}")
            rec = {"epoch": ep + 1, "bce": tot_bce / max(n, 1),
                   "pair": tot_aux / max(n, 1), "loss": tot / max(n, 1)}
        else:
            msg = f"  fold {fold} ep {ep + 1}/{epochs}  bce {tot / max(n, 1):.4f}"
            rec = {"epoch": ep + 1, "bce": tot / max(n, 1)}
        # Periodic validation. With `select: last` (default) it is diagnostics only: without this curve the only
        # way to know whether a longer schedule helps is to redo the whole run, and with OneCycle every length
        # is a different schedule (the curve inside a 30-epoch cycle does NOT say what a 12-epoch cycle would
        # be worth).
        #
        # With `select: best` the same curve becomes the weight selection criterion, and must then be read
        # knowing two things. (a) The choice is made on the fold that is then measured: the resulting OOF is
        # optimistic by an amount that cannot be estimated from inside the run. (b) Worse, that curve is
        # computed against **the teacher**, i.e. against the very ruler that can move without the gold labels
        # and the leaderboard following. Maximising it can buy exactly that kind of gain. The verdict of a run
        # with `select: best` is read on the gold studies and on the leaderboard, not here.
        t_train = time.time() - t0
        if val_every and va_uids and ((ep + 1) % val_every == 0 or ep + 1 == epochs):
            t_va = time.time()
            va = pd.DataFrame(predict(model, dl_va, device, amp, amp_dtype), index=va_uids,
                              columns=TARGETS)
            last_val = (ep + 1, va, time.time() - t_va)
            y = (labels.loc[va_uids] > 0.5).astype(float)
            y[labels.loc[va_uids].isna()] = np.nan
            auc = evaluate(y, va)["macro_auc"]
            msg += f"  val_auc {auc:.4f}"
            rec["val_auc"] = auc
            # The validation OOF at every validated epoch (`val_curve_fold<k>.csv.gz`): the per-column curve, which
            # `val_auc` (the macro mean) does not let you rebuild. Read-out, never selection: the same predictions
            # as `val_auc`, no extra pass.
            val_frames.append(va.assign(epoch=ep + 1))
            if gold_va is not None:
                # The same ruler as `gold58_macro_auc`, on this fold's gold studies only. `val_auc` measures agreement
                # with the teacher, this one with the radiologists: if the second peaks before the first, the extra
                # epochs buy fidelity to the teacher, not truth.
                gm = evaluate(gold_va, va.loc[gold_va.index])
                rec.update(gold_auc=gm["macro_auc"], gold_n=int(len(gold_va)),
                           gold_n_classes=gm["n_classes"])
                msg += f"  gold {gm['macro_auc']:.4f} (n={len(gold_va)}, {gm['n_classes']} col)"
                gold_frames.append(va.loc[gold_va.index].assign(epoch=ep + 1))
            if select == "best" and auc > best["auc"]:
                # CPU copy: keeping a second state_dict on the GPU for the whole fold costs memory the batch needs, and
                # here the cost is ~85 MB of RAM.
                best.update(auc=auc, ep=ep + 1,
                            state={k: v.detach().to("cpu", copy=True)
                                   for k, v in model.state_dict().items()})
                msg += " *"
            model.train()
        if getattr(model, "spatial_pool", "avg") == "gem":
            # GeM: the pooling power is the marker of the run (1 = mean, higher = selectivity); per epoch in the
            # history and in the log.
            rec["gem_p"] = round(float(model.gem.p.detach()), 4)
            msg += f"  gem_p {rec['gem_p']:.3f}"
        if lp is not None:
            rec["stage"] = "lp" if ep < lp_epochs else "ft"
        if swa is not None and ep >= lp_epochs + swa.starts_at(ft_epochs):
            swa.collect(model, ep + 1)
        rec["seconds"] = round(time.time() - t0, 1)
        # Training time alone, without validation: the number to compare between runs for training speed. Only
        # in the validated epochs, where it differs from `seconds` (the difference is the cost of validation);
        # in the others `seconds` is already that.
        if "val_auc" in rec:
            rec["train_seconds"] = round(t_train, 1)
        history.append(rec)
        log(msg + f"  {rec['seconds']:.0f}s")

    if select == "best" and best["state"] is not None:
        model.load_state_dict(best["state"])
        log(f"  fold {fold}: weights of epoch {best['ep']}/{epochs} "
            f"(val_auc {best['auc']:.4f}), not the last one")
    if swa is not None:
        swa.load_into(model)
        log(f"  fold {fold}: SWA weights, mean of {swa.n} epochs {swa.epochs}"
            + (f" (the last validation, raw weights: val_auc {history[-1]['val_auc']:.4f})"
               if history and "val_auc" in history[-1] else ""))

    # `train.final_predict: false` (multi-seed kernels): no final prediction for the fold. In full_data it
    # is an in-sample probe the kernel does not use (~4-5 min per seed on a T4), and from seed 5 on
    # (`fold` >= 5, no study with that fold) validation is empty.
    t0 = time.time()
    if (cfg["train"].get("final_predict", True) and va_uids and FINAL_FROM_LAST_VAL
            and select == "last" and swa is None and last_val is not None
            and last_val[0] == epochs):
        # With `val_every` the last epoch is always validated: the final prediction would redo the same
        # computation on the same model (eval, same order, no randomness), i.e. another pass over the ~880
        # studies of the fold. That one is reused. The DataLoader iterator is created anyway: it draws its
        # `_base_seed` from the global RNG, and without that draw the next fold (in sequence) would start from
        # another RNG state.
        del_it = iter(dl_va)
        del del_it
        out, infer_s = last_val[1], last_val[2]
    elif cfg["train"].get("final_predict", True) and va_uids:
        preds = predict(model, dl_va, device, amp, amp_dtype)
        out = pd.DataFrame(preds, index=va_uids, columns=TARGETS)
        infer_s = time.time() - t0
    else:
        out = pd.DataFrame(np.zeros((0, len(TARGETS)), dtype=np.float32), columns=TARGETS)
        infer_s = time.time() - t0

    # The marker that gives meaning to a NEUTRAL result: the auxiliary head on the validation studies,
    # against the findings of their reports (which the model has not seen). If they are not visible in the
    # image (mean AUC close to 0.5) a NEUTRAL measures nothing; if they are visible and the 12 columns do
    # not improve, the site written in the report does not help the student.
    aux_marker = None
    if find_tail is not None and len(out):
        t1 = time.time()
        pa = predict(model, dl_va, device, amp, amp_dtype, aux=True)
        aux_marker = aux_auc(pa, aux_findings.loc[va_uids])
        log(f"  fold {fold}: auxiliary findings on validation (out of sample): mean AUC "
            f"{aux_marker['mean']} over {aux_marker['n_fields']} findings ({time.time() - t1:.0f}s)")

    ema_out = {}
    if emas and len(out):
        # The same validation, once per shadow, and the comparison in the log: the number `train.ema` exists
        # for. The declared inference time stays that of `last`.
        y = (labels.loc[va_uids] > 0.5).astype(float)
        y[labels.loc[va_uids].isna()] = np.nan
        auc_last = evaluate(y, out)["macro_auc"]
        line = f"  fold {fold}: val_auc last {auc_last:.4f}"
        for e in emas:
            m = e.materialize()
            df = pd.DataFrame(predict(m, dl_va, device, amp, amp_dtype), index=va_uids, columns=TARGETS)
            auc_e = evaluate(y, df)["macro_auc"]
            line += f" | {e.tag} {auc_e:.4f} ({auc_e - auc_last:+.4f})"
            ema_out[e.tag] = {"preds": df, "decay": e.decay, "val_auc": auc_e,
                              "state": {k: v.detach().to("cpu", copy=True)
                                        for k, v in m.state_dict().items()}}
        log(line)
    extra = {"history": history,
             "gold_curve": pd.concat(gold_frames) if gold_frames else None,
             "val_curve": pd.concat(val_frames) if val_frames else None}
    if aux_marker is not None:
        extra["aux_auc"] = aux_marker
    if sb_param is not None:
        # The learned table (site x finding, in logits): how much every site is silent on every finding at
        # equal image. It goes into `fold<k>.json`, not into the checkpoint.
        table = pd.DataFrame(sb_param.detach().float().cpu().numpy(), index=sb_names,
                             columns=TARGETS)
        extra["site_bias"] = {s: {t: round(float(v), 4) for t, v in row.items()}
                              for s, row in table.iterrows()}
        log(f"  fold {fold}: learned per-site biases (logits, silent cells)\n"
            + table.round(2).to_string())
    return (out, infer_s, len(va_uids), model,
            best["ep"] if select == "best" else epochs, ema_out, extra)


# The final fold prediction taken from the validation of the last epoch, when available (`train_fold`):
# bit-identical predictions, one pass fewer over the fold's studies. False redoes the pass (for tests).
FINAL_FROM_LAST_VAL = True

SPEED_DEFAULTS = {"compile": False, "channels_last": False, "cudnn_benchmark": False,
                  "amp_dtype": "fp16", "val_rng": "global", "fastdense": False,
                  "fused_adam": False, "graphs": False, "fast_eval": False}


def speed_of(cfg: dict) -> dict:
    """The training speed knobs (`train.speed`), with their defaults.

    They are infrastructure, not model: none changes the recipe (data, network, loss, lr, batch,
    epochs). But none, except `val_rng`, is bit-identical to the fp16 eager path (inductor fusions,
    another convolution algorithm, another mantissa), so their effect on the numbers is measured with a
    replica before switching them on in a run that counts. Without the key everything stays as it was.

    - `compile`: in-place `torch.compile` of the encoders (`nn.Module.compile`), which keeps the
      state_dict names intact: checkpoints and EMA do not see `_orig_mod.`.
    - `channels_last`: the model in NHWC before the EMA shadows (which keep references to the state_dict
      tensors: converting afterwards would leave them on a dead copy).
    - `cudnn_benchmark`: convolution autotuning; the training shapes are fixed.
    - `amp_dtype`: `fp16` (with GradScaler, the default path) or `bf16` (no scaler: steps are never
      skipped, so OneCycle and the EMAs advance at every batch).
    - `val_rng`: `global` (default) or `isolated`. The validation DataLoader draws its `_base_seed` from
      the global RNG at every `iter()`: how many validations happen shifts the RNG the next fold's model
      is born from. With `isolated` it has its own generator and `val_every` stops being a hidden
      variable of folds 1-4. Nothing changes on fold 0: the model is born before the first validation.
    - `fastdense`: DenseNet training without concatenations, with shared-statistics BatchNorm
      (`knee.fastdense`): `true`/`cublas` GEMMs on cuBLAS and Triton elementwise, `cuda` the conv1 with the
      fused CUDA kernels (`knee.fastdense_cuda`, BN + ReLU prologue; used for the final models),
      `triton`/`triton_v`/`cudnn` fused Triton kernels (slow on the T4). The same function in exact
      arithmetic, a different summation order in fp16; not with `compile`. `true` = the default 3x3
      bottleneck conv of `TritonOps`; `triton`, `triton_v` or `cudnn` choose it (the same function, only
      the computation changes).
    - `fused_adam`: PyTorch's fused AdamW (`fused=True`: one kernel for all parameters, the fp16 scaler
      inside), same formula.
    - `graphs` (with `fastdense`): the encoder after the stem in two CUDA graphs, forward and backward
      (`fastdense.EncoderGraphs`, from the third step). On an A100 fastdense is CPU-bound (Python
      launches: GPU at 46-52%) and the graphs take it from 134 to 72 ms per step; same kernels in the
      same order, bit-identical losses and weights after 8 steps.
    - `fast_eval` (with `fastdense`): validation and final prediction with the fastdense kernels in eval
      (running averages, `fastdense.eval_forward`): 3.5 s against 8.9 for timm on 176 studies (A100),
      relative difference ~5e-4 from timm's eval. Training does not change.
    """
    spec = cfg.get("train", {}).get("speed") or {}
    unknown = set(spec) - set(SPEED_DEFAULTS)
    if unknown:
        raise ValueError(f"train.speed: unknown keys {sorted(unknown)}, "
                         f"allowed {sorted(SPEED_DEFAULTS)}")
    out = dict(SPEED_DEFAULTS, **spec)
    if out["amp_dtype"] not in ("fp16", "bf16"):
        raise ValueError(f"train.speed.amp_dtype: expected 'fp16' or 'bf16', got {out['amp_dtype']!r}")
    if out["val_rng"] not in ("global", "isolated"):
        raise ValueError(f"train.speed.val_rng: expected 'global' or 'isolated', got {out['val_rng']!r}")
    if out["fastdense"] not in (False, True, "cublas", "cuda", "triton", "triton_v", "cudnn"):
        raise ValueError(f"train.speed.fastdense: expected true/false or cublas/cuda/triton/triton_v/cudnn, "
                         f"got {out['fastdense']!r}")
    for k in ("compile", "channels_last", "cudnn_benchmark", "fused_adam", "graphs", "fast_eval"):
        if not isinstance(out[k], bool):
            raise ValueError(f"train.speed.{k}: expected true/false, got {out[k]!r}")
    if out["fastdense"] and out["compile"]:
        raise ValueError("train.speed: fastdense and compile together (the fastdense forward cannot "
                         "be compiled)")
    for k in ("graphs", "fast_eval"):
        if out[k] and not out["fastdense"]:
            raise ValueError(f"train.speed.{k} applies only with train.speed.fastdense")
    return out


def check_selection(cfg: dict) -> str:
    """Which epoch ends up in `oof.csv`: the last one, or the best on validation.

    Checked before mounting the cache because failing here costs two seconds, while finding out after
    the first fold costs forty minutes of GPU quota; and a `select: best` without validation would not
    fail at all: it would silently use the last epoch, producing a run that claims to have selected and
    has not.
    """
    select = cfg["train"].get("select", "last")
    if select not in ("last", "best"):
        raise ValueError(f"train.select: expected 'last' or 'best', got {select!r}")
    if select == "best" and not cfg["train"].get("val_every"):
        raise ValueError("train.select: 'best' requires val_every > 0: without validation "
                         "there is nothing to choose the epoch on")
    if select == "best" and is_full_data(cfg):
        raise ValueError("train.select: 'best' with train.full_data would choose the epoch on "
                         "studies that are in training: not a selection")
    return select


def run(cfg: dict, cache_root, labels_path, folds_path, out_dir, gold_path=None,
        log=print, folds_to_run: list | None = None, weights_dir=None,
        on_fold_done=None, extra_roots: dict | None = None,
        reseed_per_fold: bool = False, summary: bool = True) -> dict:
    """Trains the folds and writes OOF and metrics.

    `reseed_per_fold` (one fold per process, for parallel runs): before every fold
    `torch.manual_seed(seed + fold)`. In sequence the initialisation of fold k depends on the RNG
    consumed by the previous folds; a process starting at fold k does not have it, and with the usual
    seed the five folds (or the full-data seeds) would be born with the same head. Fold 0 stays bit for
    bit the sequential one (seed + 0).

    `summary=False` (the same children): only the fold fragments, no `oof.csv` / `metrics.json` of the
    run. Several children in the same folder writing and re-reading them together would read a file
    halfway through another's write; the parent recomposes them with `finalize`. ->
    {"folds_completed": [...]}.

    `folds_to_run` trains only a subset **without touching `cfg`**, which is the difference that matters:
    `complete` in `metrics.json` is always measured against the folds the config declares, so a partial
    session cannot declare itself complete. Every fold leaves its fragment and `finalize()` puts them back
    together.

    `on_fold_done(fold)` is called after the fold's fragment is on disk: the point at which to copy it
    elsewhere, because `out_dir` on a rented VM lives as long as the VM.

    `extra_roots` ({name: folder}): the caches of the `data.extra` sources (`knee.data.extra`), like
    `cache_root` for the main one.
    """
    check_selection(cfg)
    speed_of(cfg)        # a typo in `train.speed` costs two seconds here, not a fold
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Before mounting tens of GB of cache: can the assigned GPU run our kernels?
    log(f"device {check_device(device)} | config {config_hash(cfg)} | git {git_sha()}")

    # The Dread recipe reads its own corpus (80 slices, 336 px, five series) and has its own fold loop;
    # everything else in `run()` (fragments, mirroring, metrics) is the same.
    dread = is_dread(cfg)
    if dread:
        cache = open_corpus(cache_root, cfg)
    else:
        spec = dict(cfg["cache"], band=tuple(cfg["cache"]["band"]))
        # `cache_root` is a folder, or a list of folders if the cache is sharded and not merged (on Kaggle the
        # shards are outputs of other kernels, read-only).
        cache = open_cache(cache_root, CacheSpec(**spec))
    labels = pd.read_csv(labels_path, index_col=0)[TARGETS]
    labels = labels.loc[[u for u in labels.index if u in cache]]
    folds = pd.read_csv(folds_path).set_index(ID)["fold"].to_dict()
    log(f"{len(labels)} labelled studies | "
        + (cache.describe() if dread else f"cache {(len(cache),) + cache.spec.shape}"))
    cell_weights = cell_weights_for(cfg, Path(labels_path).parent, labels.index)
    pseudo = pseudo_labels_for(cfg, Path(labels_path).parent, labels.index)
    teachers = teacher_labels_for(cfg, Path(labels_path).parent, labels.index)
    site_bias = site_bias_for(cfg, Path(labels_path).parent, labels.index)
    if site_bias is not None and (dread or is_qwen(cfg) or is_dino(cfg)):
        raise ValueError("train.site_bias applies only to the train_fold loop (not Dread/Qwen/DINO)")
    aux_findings = aux_findings_for(cfg, Path(labels_path).parent, labels.index)
    if aux_findings is not None:
        if dread or is_qwen(cfg) or is_dino(cfg):
            raise ValueError("train.aux_findings applies only to the train_fold loop (not Dread/Qwen/DINO)")
        log(f"train.aux_findings: {aux_findings.shape[1]} auxiliary findings from "
            f"{cfg['train']['aux_findings']['labels']}, cells mentioned "
            f"{np.isfinite(aux_findings.to_numpy()).mean():.1%}")
    if init_from_of(cfg) is not None and (dread or is_qwen(cfg) or is_dino(cfg)):
        raise ValueError("train.init_from applies only to the train_fold loop (not Dread/Qwen/DINO)")
    # `data.extra`: the caches of the extra rows, opened with the run's spec, and their targets per fold.
    # From here `cache` sees them all (`CombinedCache`): validation asks only for RSNA studies, training
    # for those too.
    extras = None
    if has_extra(cfg):
        if dread or is_qwen(cfg) or is_dino(cfg):
            raise ValueError("data.extra applies only to the train_fold loop (not Dread/Qwen/DINO)")
        extra_loss_ok(cfg)
        caches = open_extras(cfg, extra_roots, cache.spec)
        run_folds = list(cfg["train"]["folds"] if folds_to_run is None else folds_to_run)
        extras = extra_targets(cfg, caches, Path(labels_path).parent, run_folds)
        cache = CombinedCache(cache, caches)
        log(f"data.extra: {cache.describe()}")

    oof, infer_s, n_infer, done, chosen = [], 0.0, 0, [], {}
    budget = cfg["train"].get("time_budget_s")
    t_start = time.time()
    for fold in (cfg["train"]["folds"] if folds_to_run is None else folds_to_run):
        if reseed_per_fold:
            torch.manual_seed(cfg["seed"] + fold)
            np.random.seed(cfg["seed"] + fold)
        if budget and time.time() - t_start > budget:
            # A run that runs out of time still writes what it has: half an OOF is useless for the metric but says
            # whether the pipeline works, and `complete` in metrics.json prevents confusing it with a real run.
            log(f"!! time budget exhausted before fold {fold}: partial run")
            break
        # Qwen 3.5 has its own fold loop (one study per step, LoRA); fragments, metrics and checkpoints go
        # through here as for the others. DINOv2 with its two-stage recipe also has its own loop.
        fit = (train_fold_dread if dread else train_fold_qwen if is_qwen(cfg) else
               train_fold_dino if is_dino(cfg) else train_fold)
        p, s, n, model, ep, ema_out, extra = fit(cfg, cache, labels, folds, fold, device, log,
                                                 weights_dir=weights_dir, gold_path=gold_path,
                                                 cell_weights=cell_weights, pseudo=pseudo,
                                                 teacher=teachers[fold] if teachers else None,
                                                 **({"extra": extras[fold]} if extras else {}),
                                                 **({"site_bias": site_bias} if site_bias else {}),
                                                 **({"aux_findings": aux_findings}
                                                    if aux_findings is not None else {}))
        oof.append(p)
        infer_s += s
        n_infer += n
        done.append(fold)
        chosen[str(fold)] = ep
        # This fold's fragment, for sessions that do not reach the end. It costs a 900-row CSV and makes a
        # ten-hour run restartable.
        p.rename_axis(ID).to_csv(out_dir / f"oof_fold{fold}.csv")
        # `history`: one row per epoch (training bce, `val_auc` and the fold's gold at the validation epochs):
        # the structured log of the curve.
        (out_dir / f"fold{fold}.json").write_text(json.dumps(
            {"fold": fold, "infer_s": s, "n_infer": n, "epoch": ep,
             "ema_val_auc": {tag: eo["val_auc"] for tag, eo in ema_out.items()},
             "history": extra["history"],
             **({"site_bias": extra["site_bias"]} if "site_bias" in extra else {}),
             **({"aux_auc": extra["aux_auc"]} if "aux_auc" in extra else {})}),
            encoding="utf-8")
        if extra["gold_curve"] is not None:
            extra["gold_curve"].rename_axis(ID).to_csv(out_dir / f"gold_curve_fold{fold}.csv")
        # Compressed and with five digits: ~900 studies x 10 validated epochs per fold. The trainers that do not
        # produce it (dino, dread, qwen) do not have the key.
        if extra.get("val_curve") is not None:
            extra["val_curve"].rename_axis(ID).to_csv(out_dir / f"val_curve_fold{fold}.csv.gz",
                                                     float_format="%.5g")
        if cfg["train"]["save_checkpoints"]:
            # The checkpoint of the Dread recipe is in the format of the public Raptor models ({model, arch, res,
            # lab}): `public_raptor.load_raptor` loads it without a new branch.
            torch.save(checkpoint_of(model, cfg, fold) if dread else
                       qwen_checkpoint(model, cfg, fold) if is_qwen(cfg) else
                       dino_checkpoint(model, cfg, fold) if is_dino(cfg) else
                       {"model": strip_aux(model.state_dict()), "cfg": cfg, "fold": fold},
                       out_dir / f"fold{fold}.pt")
        # The EMA shadows: `oof_ema<d>_fold<k>.csv` and `ema<d>_fold<k>.pt`. The names do NOT start with `fold`:
        # `infer.load_ensemble` mounts `fold*.pt` and must not see them. The `.pt` carries the same `cfg` (same
        # geometry) plus `ema: decay`, so a dataset of EMA weights renamed to `fold<k>.pt` declares by itself
        # what it contains.
        for tag, eo in ema_out.items():
            eo["preds"].rename_axis(ID).to_csv(out_dir / f"oof_{tag}_fold{fold}.csv")
            if cfg["train"]["save_checkpoints"]:
                torch.save({"model": strip_aux(eo["state"]), "cfg": dict(cfg, ema=eo["decay"]),
                            "fold": fold, "ema": eo["decay"]},
                           out_dir / f"{tag}_fold{fold}.pt")
        if on_fold_done is not None:
            # The caller can move the fragments off the VM before the next fold starts. It touches neither the
            # model nor the metrics.
            on_fold_done(fold)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not summary:
        return {"run": cfg["name"], "folds_completed": list(done)}
    oof_df = pd.concat(oof).rename_axis(ID)
    oof_df.to_csv(out_dir / "oof.csv")
    if oof_df.empty:
        # `train.final_predict: false`, or seeds beyond the fifth in full_data: nothing to measure.
        metrics = {"run": cfg["name"], "config_hash": config_hash(cfg), "git_sha": git_sha(),
                   "folds_completed": list(done),
                   "complete": sorted(done) == sorted(cfg["train"]["folds"]),
                   "note": "no final prediction: train.final_predict false or empty validation"}
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log(json.dumps(metrics, indent=2))
        return metrics
    metrics = compute_metrics(cfg, oof_df, labels, infer_s, n_infer, done, gold_path,
                              selected_epochs=chosen)
    metrics["ema"] = ema_metrics(cfg, out_dir, labels, done, gold_path)
    metrics["gold58_curve"] = gold58_curve(out_dir, done, gold_path)
    metrics = mark_in_sample(cfg, metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
    return metrics


IN_SAMPLE_KEYS = ("oof_macro_auc", "oof_bce", "oof_macro_auc_on_005", "auc_per_class",
                  "gold58_macro_auc", "gold58_auc_per_class", "gold58_n_studies",
                  "diagnosis_adr001_5", "diagnosis_delta", "diagnosis_threshold",
                  "diagnosis_threshold_source", "ema", "gold58_curve", "n_studies_oof")


def mark_in_sample(cfg: dict, metrics: dict) -> dict:
    """With `train.full_data` the metrics derived from 'validation' are in-sample: they leave
    `metrics.json` with the `in_sample_` prefix and a plain warning, so that no reader takes them for an
    OOF.
    """
    if not is_full_data(cfg):
        return metrics
    out = {"full_data": True,
           "warning": "train.full_data: every fold trained on all studies; the in_sample_* "
                      "keys are measured on studies that are in training and are NOT "
                      "an OOF. The only ruler is the leaderboard."}
    for k, v in metrics.items():
        out[f"in_sample_{k}" if k in IN_SAMPLE_KEYS else k] = v
    return out


def finalize(cfg: dict, out_dir, labels_path, gold_path=None, log=print) -> dict:
    """Puts the fold fragments back together into `oof.csv` and `metrics.json`.

    Needed when the folds ran in different sessions. It retrains nothing and computes nothing by hand:
    it reads the fragments `run()` left and calls the same `compute_metrics`, because the metrics have a
    single implementation.
    """
    out_dir = Path(out_dir)
    frags = sorted(out_dir.glob("oof_fold*.csv"))
    if not frags:
        raise FileNotFoundError(f"no oof_fold*.csv fragment in {out_dir}")
    oof_df = pd.concat([pd.read_csv(f, index_col=0) for f in frags]).rename_axis(ID)
    side = [json.loads((out_dir / f"fold{int(f.stem[8:])}.json").read_text(encoding="utf-8"))
            for f in frags]
    done = sorted(d["fold"] for d in side)
    log(f"folds found {done} | {len(oof_df)} studies")

    labels = pd.read_csv(labels_path, index_col=0)[TARGETS]
    labels = labels.loc[[u for u in labels.index if u in set(oof_df.index)]]
    metrics = compute_metrics(cfg, oof_df, labels,
                              sum(d["infer_s"] for d in side),
                              sum(d["n_infer"] for d in side),
                              done, gold_path,
                              selected_epochs={str(d["fold"]): d["epoch"] for d in side})
    oof_df.to_csv(out_dir / "oof.csv")
    metrics["ema"] = ema_metrics(cfg, out_dir, labels, done, gold_path)
    metrics["gold58_curve"] = gold58_curve(out_dir, done, gold_path)
    ema = metrics["ema"]
    metrics = mark_in_sample(cfg, metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
    for tag, m in ema.items():
        log(f"  {tag}: oof {m['oof_macro_auc']:.4f} ({m['oof_delta_vs_last']:+.4f} vs last) "
            f"| gold-58 {m['gold58_macro_auc']:.4f} ({m['gold58_delta_vs_last']:+.4f})"
            + (" [IN-SAMPLE: full_data]" if is_full_data(cfg) else ""))
    return metrics


def gold58_curve(out_dir, done_folds: list, gold_path=None) -> dict:
    """The gold-58 curve per epoch, rebuilt from the `gold_curve_fold<k>.csv` of the completed folds.

    It is the direct test of teacher memorisation. `val_auc` (in the `history` of `fold<k>.json`)
    measures agreement with the teacher and rises as long as the model copies it; this curve measures
    agreement with the radiologists, on the same weights at the same epochs. A gold peak before the last
    epoch, with `val_auc` still rising, says that the extra epochs buy fidelity to the teacher, not
    truth. Per fold there are 7-16 studies (noise ~0.03): the curve is read only here, on the merged
    folds, and covers all 58 with all five folds. Only the epochs ALL folds read have are taken, so every
    point compares the same studies. A run without `val_every` or without gold studies has no fragments
    and the curve is empty.
    """
    out_dir = Path(out_dir)
    frames = [pd.read_csv(out_dir / f"gold_curve_fold{f}.csv", index_col=0).assign(fold=f)
              for f in done_folds if (out_dir / f"gold_curve_fold{f}.csv").is_file()]
    if not frames:
        return {}
    df = pd.concat(frames)
    folds = sorted(int(f) for f in df["fold"].unique())
    epochs = sorted(set.intersection(*(set(df.loc[df["fold"] == f, "epoch"].astype(int))
                                       for f in folds)))
    if not epochs:
        return {}
    gold = load_gold(gold_path)
    val_hist: dict[int, list[float]] = {}
    for f in folds:
        p = out_dir / f"fold{f}.json"
        if p.is_file():
            for r in json.loads(p.read_text(encoding="utf-8")).get("history", []):
                if r.get("val_auc") is not None:
                    val_hist.setdefault(int(r["epoch"]), []).append(float(r["val_auc"]))
    by_epoch = {}
    for e in epochs:
        preds = df[df["epoch"] == e][TARGETS]
        common = gold.index.intersection(preds.index)
        m = evaluate(gold.loc[common], preds.loc[common])
        by_epoch[str(e)] = {"macro_auc": m["macro_auc"], "n_studies": m["n_studies"],
                            "n_classes": m["n_classes"], "auc_per_class": m["auc_per_class"],
                            "val_auc_mean": (float(np.mean(val_hist[e]))
                                             if val_hist.get(e) else None)}
    best = max(by_epoch, key=lambda k: by_epoch[k]["macro_auc"])
    last = str(epochs[-1])
    val_best = max((k for k in by_epoch if by_epoch[k]["val_auc_mean"] is not None),
                   key=lambda k: by_epoch[k]["val_auc_mean"], default=None)
    return {"folds": folds, "n_studies": by_epoch[last]["n_studies"],
            "epochs": epochs, "by_epoch": by_epoch,
            "best_epoch": int(best), "last_epoch": int(last),
            "last_minus_best": by_epoch[last]["macro_auc"] - by_epoch[best]["macro_auc"],
            "val_best_epoch": int(val_best) if val_best is not None else None}


def ema_metrics(cfg: dict, out_dir, labels: pd.DataFrame, done_folds: list,
                gold_path=None) -> dict:
    """The same metrics as `compute_metrics`, on the OOF of the EMA shadows, with the delta against `last`
    next to them. A shadow enters only if it has the fragment of EVERY completed fold: half an EMA OOF
    compared with a whole OOF would be a meaningless number.

    Everything goes through `compute_metrics`: the metrics have a single implementation.
    """
    out_dir = Path(out_dir)
    last = pd.read_csv(out_dir / "oof.csv", index_col=0)[TARGETS]
    ref = compute_metrics(cfg, last, labels.loc[last.index], 0.0, 1, done_folds, gold_path)
    res = {}
    tags = sorted({f.name[len("oof_"):].split("_fold")[0]
                   for f in out_dir.glob("oof_ema*_fold*.csv")})
    for tag in tags:
        frags = [out_dir / f"oof_{tag}_fold{k}.csv" for k in done_folds]
        if not all(f.is_file() for f in frags):
            continue
        oof = pd.concat([pd.read_csv(f, index_col=0)[TARGETS] for f in frags]).rename_axis(ID)
        oof = oof.loc[last.index]
        oof.to_csv(out_dir / f"oof_{tag}.csv")
        m = compute_metrics(cfg, oof, labels.loc[oof.index], 0.0, 1, done_folds, gold_path)
        res[tag] = {
            "decay": float(tag[len("ema"):]),
            "oof_macro_auc": m["oof_macro_auc"],
            "oof_delta_vs_last": m["oof_macro_auc"] - ref["oof_macro_auc"],
            "oof_auc_per_class": m["oof_auc_per_class"],
            "oof_bce": m["oof_bce"],
            "gold58_macro_auc": m["gold58_macro_auc"],
            "gold58_delta_vs_last": m["gold58_macro_auc"] - ref["gold58_macro_auc"],
            "gold58_auc_per_class": m["gold58_auc_per_class"],
            "diagnosis_delta": m["diagnosis_delta"],
        }
    return res


def compute_metrics(cfg, oof: pd.DataFrame, labels: pd.DataFrame, infer_s: float,
                    n_infer: int, done_folds: list, gold_path=None,
                    selected_epochs: dict | None = None) -> dict:
    """The required validation metrics, plus the teacher-versus-model diagnosis on the gold studies.

    `oof_macro_auc` is measured against the teacher labels **binarised**: AUC wants a binary truth and
    the teacher's values are probabilities. That number measures agreement with the teacher, not with
    reality: reality is `gold58_macro_auc`, which is why both are reported.

    **The 0.5 threshold is a delicate choice.** For teachers derived from discrete report buckets the
    values live on a few levels and the 0.35-0.55 band is (almost) empty. **Not for teachers built with
    continuous quantile mapping or model blends**: there more than 10% of the cells can fall in that
    band, and several percent of the cells change binary class between teachers. On those sets the 0.5
    cut is a parameter, not a formality, and whoever reads `oof_macro_auc` must know it.

    Corollary, which matters more than the threshold: **`oof_macro_auc` is not comparable across
    different teachers**, and when the teacher contains the OOF predictions of a previous model it is not
    comparable *even against that model*: a teacher carrying 25% of a model's rank made that model's
    existing OOF jump from 0.881 to 0.934 with the model unchanged. The verdict of a run on such a
    teacher is read on the gold studies and on the leaderboard.
    """
    y_teacher = (labels.loc[oof.index] > 0.5).astype(float)
    y_teacher[labels.loc[oof.index].isna()] = np.nan
    res = evaluate(y_teacher, oof)

    gold = load_gold(gold_path)
    covered = gold.index.intersection(oof.index)
    model_gold = evaluate(gold.loc[covered], oof.loc[covered])
    teacher_gold = score_label_set(labels.loc[covered], gold)

    eps = 1e-7
    p = oof.clip(eps, 1 - eps)
    t = labels.loc[oof.index]
    bce = float(np.nanmean(-(t * np.log(p) + (1 - t) * np.log(1 - p)).to_numpy()))

    delta = model_gold["macro_auc"] - teacher_gold["macro_auc"]
    # The threshold is δ, measured: below δ the difference is the seed, not the model. Before δ existed it
    # was 0.02, a placeholder of the size of the resolution limit of the 58 gold studies; the fallback stays
    # so that a run launched without `delta` in the config does not lie, and it is declared in metrics.json
    # so it shows which of the two decided.
    thr = float(cfg.get("delta", {}).get("gold58") or 0.02)
    diagnosis = ("model << teacher: it does not learn what it was taught "
                 "-> image pipeline" if delta < -thr else
                 "model >> teacher: label noise averages out, but the labels "
                 "hold it back -> labels" if delta > thr else
                 "model ~ teacher: it has saturated its labels -> labels")

    return {
        "run": cfg["name"],
        "parent_run": cfg.get("parent"),
        "seed": cfg["seed"],
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
        "folds_completed": list(done_folds),
        "complete": sorted(done_folds) == sorted(cfg["train"]["folds"]),
        "n_studies_oof": int(len(oof)),
        "oof_macro_auc": res["macro_auc"],
        "oof_auc_per_class": res["auc_per_class"],
        "oof_bce": bce,
        "gold58_macro_auc": model_gold["macro_auc"],
        "gold58_auc_per_class": model_gold["auc_per_class"],
        "gold58_n_studies": int(len(covered)),
        "teacher": {"labels": cfg["data"]["labels"],
                    "gold58_macro_auc": teacher_gold["macro_auc"],
                    # The training targets, if they are not `labels`. With an OOF as teacher the run's OOF has a
                    # second-order leak.
                    "train_teacher": cfg["train"].get("teacher")},
        "diagnosis_adr001_5": diagnosis,
        "diagnosis_threshold": thr,
        "diagnosis_threshold_source": "δ misurato" if cfg.get("delta", {}).get("gold58")
                                      else "placeholder 0.02, δ not in config",
        "diagnosis_delta": delta,
        # With `select: best` the chosen epoch is part of the result: two runs with the same config can stop at
        # different epochs, and without this field the difference would be invisible in `metrics.json`.
        "checkpoint_selection": cfg["train"].get("select", "last"),
        "selected_epochs": selected_epochs or {},
        "infer_runtime_s": round(infer_s / max(n_infer, 1) * N_TEST_STUDIES, 1),
        "infer_runtime_basis": (f"projection to {N_TEST_STUDIES} studies from {n_infer} validation "
                                f"studies in {infer_s:.1f}s; DICOM decoding excluded"),
    }
