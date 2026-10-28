"""DINOv2 as the student, with a ViT fine-tuning recipe rather than the CNN one (an experiment, not
used by the final DenseNet submission).

Earlier ViT students went through the ViT branch of `KneeBaseline` with the DenseNet recipe:
windows in [0, 1] without the weights' normalisation, a single OneCycle stage at 3e-4 on the head
and 1.5e-5 on the whole encoder, 12 random windows, 288 px with patch 16 (18x18 tokens of 6 mm,
which lost mostly on the menisci). Here the recipe is one that fine-tuned DINOv3-L well in another
competition ("dual pooling", two stages), adapted to the knee:

- **input**: a 384 px cache (12 slices, 110 mm), centre-cropped to 378 = 27x27 patches of 14
  (4 mm per token); ImageNet mean and std, the numbers DINOv2 was trained with. With `img` above the
  cache size it is resampled instead of cropped;
- **windows**: three contiguous slices in the three channels, stride 3: 4 windows per slot covering
  all 12 slices once, 16 per study, all used in training (no sampling);
- **encoder**: `vit_large_patch14_reg4_dinov2` (304M, 4 registers);
- **per-window features**: mean and max of the patch tokens (registers and CLS excluded),
  LayerNorm, half-width MLP with dropout 0.5;
- **windows to study**: per-finding attention with slot embedding, as in `KneeBaseline`;
- **stage 1**: frozen encoder, heads only, OneCycle at `lr_head`;
- **stage 2**: fine-tuning of the last `train_blocks` blocks and the final norm (patch embedding,
  tokens and early blocks stay frozen), encoder at `lr_head * backbone_lr_factor`, heads at
  `lr_head`, cosine down to 1e-6;
- AdamW with weight decay 0.1, gradient clipping at 1.0;
- **EMA** 0.995 of the weights at every step, EMA validation at every epoch in both stages, weights
  of the best epoch, early stopping after `patience` stage-2 epochs without a gain of at least
  `min_delta`;
- geometric and intensity **augmentation** per window (`augment.augment_windows`), never flips:
  they swap medial and lateral.

Note: the epoch is chosen on the fold being measured, so the resulting OOF is optimistic by a small
amount that cannot be estimated from inside the run.

The checkpoint carries only the trained weights (heads, last blocks, final norm): the rest of the
encoder is reloaded from the public weights (DINOv2, Apache-2.0).
"""
from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .augment import AugmentSpec, augment_windows
from .constants import TARGETS
from .models.baseline import GROUP, IMAGENET_MEAN, IMAGENET_STD, make_windows, window_slot_index

DEFAULT_ID = "vit_large_patch14_reg4_dinov2.lvd142m"


def is_dino(cfg: dict) -> bool:
    return str(cfg.get("model", {}).get("arch", "")) == "dinov2"


class DinoKnee(nn.Module):
    """DINOv2 ViT per window -> per-finding attention over the windows -> 12 logits."""

    def __init__(self, encoder: nn.Module, n_slots: int, n_slices: int, img: int,
                 window_stride: int = 3, head_dim: int | None = None, dropout: float = 0.5,
                 train_blocks: int = 8, n_classes: int = len(TARGETS)):
        super().__init__()
        self.encoder = encoder
        self.n_prefix = int(encoder.num_prefix_tokens)          # CLS + registers
        self.n_slots, self.n_slices = n_slots, n_slices
        self.img, self.window_stride = int(img), int(window_stride)
        patch = encoder.patch_embed.patch_size[0]
        if self.img % patch:
            raise ValueError(f"img {self.img} is not a multiple of the patch {patch}")
        n_blocks = len(encoder.blocks)
        if not 0 <= int(train_blocks) <= n_blocks:
            raise ValueError(f"train_blocks {train_blocks} outside [0, {n_blocks}]")
        self.train_blocks = int(train_blocks)
        d = encoder.num_features
        hd = int(head_dim or d)                                   # half of (avg, max)
        self.norm = nn.LayerNorm(2 * d)
        self.proj = nn.Sequential(nn.Linear(2 * d, hd), nn.GELU(), nn.Dropout(dropout))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, hd))
        self.attn = nn.Linear(hd, n_classes)
        self.cls_w = nn.Parameter(torch.zeros(n_classes, hd))
        self.cls_b = nn.Parameter(torch.zeros(n_classes))
        nn.init.trunc_normal_(self.slot_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_w, std=0.02)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("slot_of", window_slot_index(n_slots, n_slices, stride=self.window_stride),
                             persistent=False)
        self.encode_chunk = 0           # windows per encoder forward in eval (0 = all)

    # --- which weights are trained ------------------------------------------------------
    def head_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def tuned_encoder_names(self) -> list[str]:
        """The last `train_blocks` blocks and the final norm: what stage 2 unfreezes."""
        first = len(self.encoder.blocks) - self.train_blocks
        keep = [f"blocks.{i}." for i in range(first, len(self.encoder.blocks))]
        if self.train_blocks:
            keep.append("norm.")
        return [f"encoder.{n}" for n, _ in self.encoder.named_parameters()
                if any(n.startswith(k) for k in keep)]

    def set_stage(self, stage: int) -> None:
        """1 = heads only; 2 = heads + last encoder blocks."""
        tuned = set(self.tuned_encoder_names()) if stage == 2 else set()
        for n, p in self.named_parameters():
            p.requires_grad_(not n.startswith("encoder.") or n in tuned)

    def trained_state(self) -> dict:
        """The weights trained by either stage: the checkpoint (the rest is the public encoder)."""
        keep = set(self.tuned_encoder_names()) | {n for n, _ in self.named_parameters()
                                                  if not n.startswith("encoder.")}
        return {n: t.detach().to("cpu", copy=True) for n, t in self.state_dict().items() if n in keep}

    # --- forward -----------------------------------------------------------------------
    def windows01(self, vol: torch.Tensor) -> torch.Tensor:
        """`(B, S, K, P, P)` uint8 -> `(B, W, 3, img, img)` in [0, 1].

        Centre crop if `img <= P` (378 from 384: 3 px per side), otherwise bilinear resampling (more
        tokens on the same anatomy, no new detail).
        """
        p = vol.shape[-1]
        x = vol.float().div_(255.0)
        if self.img <= p:
            o = (p - self.img) // 2
            x = x[..., o:o + self.img, o:o + self.img]
        else:
            b, s, k = x.shape[:3]
            x = F.interpolate(x.flatten(0, 2).unsqueeze(1), size=(self.img, self.img),
                              mode="bilinear", align_corners=False).reshape(b, s, k, self.img, self.img)
        return make_windows(x, self.window_stride, GROUP)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 3, img, img)` normalised -> `(N, 2d)`: mean and max of the patch tokens."""
        t = self.encoder.forward_features(x)[:, self.n_prefix:]
        return torch.cat([t.mean(dim=1), t.amax(dim=1)], dim=-1).float()

    def forward(self, w: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """`w (B, W, 3, img, img)` in [0, 1], `slot_mask (B, S)` bool -> logits `(B, T)`."""
        b, nw = w.shape[:2]
        x = ((w - self.mean) / self.std).flatten(0, 1)
        n = self.encode_chunk
        if n and not self.training and x.shape[0] > n:
            z = torch.cat([self._encode(c) for c in x.split(n)])
        else:
            z = self._encode(x)
        f = self.proj(self.norm(z)).reshape(b, nw, -1)
        f = f + self.slot_emb[self.slot_of].unsqueeze(0)
        valid = slot_mask[:, self.slot_of].unsqueeze(-1)       # (B, W, 1)
        a = self.attn(f).float().masked_fill(~valid, float("-inf"))
        # A study without valid slots does not exist in the corpus; if one arrived, no NaN.
        a = torch.where(~valid.any(dim=1, keepdim=True), torch.zeros_like(a), a).softmax(dim=1)
        pooled = torch.einsum("bwt,bwd->btd", a, f.float())
        return (pooled * self.cls_w).sum(-1) + self.cls_b


# --- construction ---------------------------------------------------------------------------
def make_encoder(dino_id: str, img: int, pretrained: bool = True) -> nn.Module:
    """The ViT from timm at resolution `img` (timm resamples the pos_embed on loading).

    Offline (Kaggle) `KNEE_DINO_WEIGHTS` points to the weights file.
    """
    import timm
    kw = {}
    f = os.environ.get("KNEE_DINO_WEIGHTS")
    if pretrained and f:
        kw["pretrained_cfg_overlay"] = {"file": f}
    return timm.create_model(dino_id, pretrained=pretrained, num_classes=0, img_size=img, **kw)


def build_model(cfg: dict, n_slots: int, n_slices: int, pretrained: bool = True,
                encoder: nn.Module | None = None, log=print) -> DinoKnee:
    """Encoder + heads from the config, at stage 1. `encoder` replaces timm in the tests."""
    m, t = cfg["model"], cfg.get("train", {})
    img = int(m.get("img", 378))
    enc = encoder if encoder is not None else make_encoder(m.get("dino_id", DEFAULT_ID), img, pretrained)
    model = DinoKnee(enc, n_slots, n_slices, img, window_stride=int(m.get("window_stride", 3)),
                     head_dim=m.get("head_dim"), dropout=float(m.get("head_dropout", 0.5)),
                     train_blocks=int(t.get("train_blocks", 8)))
    if hasattr(enc, "set_grad_checkpointing"):
        enc.set_grad_checkpointing(bool(t.get("grad_checkpointing", True)))
    model.set_stage(1)
    n_all = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in model.head_parameters())
    named = dict(model.named_parameters())
    n_tuned = sum(named[n].numel() for n in model.tuned_encoder_names())
    grid = img // enc.patch_embed.patch_size[0]
    log(f"  DINOv2 {m.get('dino_id', DEFAULT_ID)} at {img} px ({grid}x{grid} patches + {model.n_prefix} "
        f"prefix tokens): {n_all / 1e6:.0f}M parameters; heads {n_head / 1e6:.2f}M, stage 2 unfreezes "
        f"{model.train_blocks}/{len(enc.blocks)} blocks ({n_tuned / 1e6:.1f}M); "
        f"{len(model.slot_of) // n_slots} windows per slot, {len(model.slot_of)} per study")
    return model


# --- EMA -----------------------------------------------------------------------------------
class ParamEma:
    """EMA of the weights trained by either stage.

    The shadow starts from the initial weights and is updated at every step for the whole fold,
    across both stages: in stage 1 the encoder blocks do not move and their shadow stays the public
    weight, in stage 2 it starts following them. Always-frozen weights are not needed (their mean is
    themselves). `applied()` puts the shadow into the model for an evaluation.
    """

    def __init__(self, named: dict[str, nn.Parameter], decay: float):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"ema: decay must be in (0, 1), got {decay!r}")
        self.decay = float(decay)
        self.names = list(named)
        self.params = [named[n] for n in self.names]
        self.shadow = [p.detach().float().clone() for p in self.params]

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_mul_(self.shadow, self.decay)
        torch._foreach_add_(self.shadow, [p.detach().float() for p in self.params], alpha=1.0 - self.decay)

    @contextmanager
    def applied(self):
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p, s in zip(self.params, self.shadow):
                p.copy_(s.to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)

    def state(self) -> dict[str, torch.Tensor]:
        return {n: s.to("cpu", copy=True) for n, s in zip(self.names, self.shadow)}


# --- training one fold -------------------------------------------------------------------
@torch.no_grad()
def predict(model: DinoKnee, loader, device, dtype=torch.bfloat16) -> np.ndarray:
    model.eval()
    out = []
    for vol, mask, *_ in loader:
        w = model.windows01(vol.to(device, non_blocking=True))
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logit = model(w, mask.to(device))
        out.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


@torch.no_grad()
def predict_study(model: DinoKnee, vol: torch.Tensor, mask: torch.Tensor,
                  dtype: torch.dtype = torch.float16) -> torch.Tensor | None:
    """One study `vol (S, K, P, P)` uint8, `mask (S,)` -> probabilities `(T,)` on the device.

    None if no slot is valid (decoding failed): downstream it becomes the neutral 0.5 row, not a
    prediction on black images. `dtype` float32 turns autocast off (the kernel's fallback if fp16
    gave non-finite values).
    """
    if not bool(mask.any()):
        return None
    w = model.windows01(vol.unsqueeze(0))
    amp = vol.device.type == "cuda" and dtype != torch.float32
    with torch.autocast(vol.device.type, dtype=dtype, enabled=amp):
        logit = model(w, mask.unsqueeze(0))
    return torch.sigmoid(logit.float())[0]


def _cosine(steps: int, floor: float):
    """Per-step cosine from the full value down to `floor` (fraction of the maximum)."""
    return lambda s: floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))


def train_fold_dino(cfg, cache, labels, folds, fold, device, log=print, gold_path=None, **_):
    """Same signature and outputs as `train.train_fold`: `run()` writes the fragments."""
    from torch.utils.data import DataLoader

    from .data.dataset import KneeStudies
    from .labels.gold import load_gold
    from .metrics import evaluate
    from .train import is_full_data, masked_bce

    full = is_full_data(cfg)
    tr_uids = list(labels.index) if full else [u for u in labels.index if folds.get(u, -1) != fold]
    va_uids = [u for u in labels.index if folds.get(u, -1) == fold]
    log(f"  fold {fold}: {len(tr_uids)} train / {len(va_uids)} val (DINOv2)")
    gold_va = None
    if gold_path is not None:
        g = load_gold(gold_path)
        gold_va = g.loc[g.index.intersection(va_uids)]
        gold_va = None if gold_va.empty else gold_va

    t = cfg["train"]
    torch.manual_seed(cfg["seed"] + fold)
    model = build_model(cfg, cache.spec.shape[0], cache.spec.slices, log=log).to(device)
    model.encode_chunk = int(t.get("encode_chunk", 64))
    workers = int(t.get("workers", 4))
    g_tr = torch.Generator().manual_seed(cfg["seed"] * 1000 + fold)
    dl_tr = DataLoader(KneeStudies(cache, tr_uids, labels), batch_size=int(t["batch_size"]),
                       shuffle=True, drop_last=True, generator=g_tr, num_workers=workers,
                       pin_memory=True, persistent_workers=workers > 0)
    dl_va = DataLoader(KneeStudies(cache, va_uids, labels), batch_size=int(t["eval_batch_size"]),
                       shuffle=False, num_workers=workers, pin_memory=True,
                       persistent_workers=workers > 0)
    aug = AugmentSpec.from_config(cfg)
    if aug and not aug.plain:
        raise ValueError("train.augment: rows and slice_offset exist only in the window loop "
                         "of train.py")
    gen_aug = torch.Generator().manual_seed(cfg["seed"] * 7919 + fold) if aug else None
    if aug:
        log(f"  fold {fold}: per-window augment ({aug.describe()}), no flips")
    dtype = torch.bfloat16 if (device.type != "cuda" or torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    clip = float(t.get("grad_clip", 1.0))
    wd = float(t.get("weight_decay", 0.1))
    lr_head = float(t["lr_head"])
    lr_bb = lr_head * float(t.get("backbone_lr_factor", 0.1))
    log_every = int(t.get("log_every", 100))
    e1, e2 = int(t.get("epochs_head", 3)), int(t["epochs"])
    patience, min_delta = int(t.get("patience", 0)), float(t.get("min_delta", 1e-4))
    y_va = (labels.loc[va_uids] > 0.5).astype(float)
    y_va[labels.loc[va_uids].isna()] = np.nan

    # The shadow covers everything either stage trains, from the start of the fold.
    named = dict(model.named_parameters())
    ema = ParamEma({n: named[n] for n in model.trained_state()}, float(t.get("ema", 0.995)))
    history, gold_frames = [], []
    best = {"auc": -1.0, "ep": 0, "state": None, "preds": None, "infer_s": 0.0}
    stale, ep_global = 0, 0
    for stage, n_ep in ((1, e1), (2, e2)):
        if n_ep <= 0:
            continue
        model.set_stage(stage)
        head = model.head_parameters()
        bb = [named[n] for n in model.tuned_encoder_names()] if stage == 2 else []
        groups = [g for g in ({"params": bb, "lr": lr_bb}, {"params": head, "lr": lr_head}) if g["params"]]
        opt = torch.optim.AdamW(groups, weight_decay=wd)
        steps = n_ep * len(dl_tr)
        if stage == 1:
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                        total_steps=max(1, steps))
        else:
            floor = float(t.get("lr_min", 1e-6)) / lr_head
            sched = torch.optim.lr_scheduler.LambdaLR(opt, _cosine(steps, floor))
        trainable = [p for g in groups for p in g["params"]]
        log(f"  fold {fold}: stage {stage} ({'heads, frozen encoder' if stage == 1 else f'last {model.train_blocks} blocks + heads'}), "
            f"{n_ep} epochs, {steps} steps, {sum(p.numel() for p in trainable) / 1e6:.1f}M weights, "
            f"lr {'encoder %g / ' % lr_bb if bb else ''}heads {lr_head:g}, EMA {ema.decay}")
        stop = False
        for _ep in range(n_ep):
            ep_global += 1
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for k, (vol, mask, y, _w) in enumerate(dl_tr):
                mask, y = mask.to(device, non_blocking=True), y.to(device, non_blocking=True)
                w = model.windows01(vol.to(device, non_blocking=True))
                if aug:
                    w = augment_windows(w, aug, gen_aug)
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                    logit = model(w, mask)
                loss = masked_bce(logit.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, clip)
                s0 = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if s0 <= scaler.get_scale():
                    sched.step()
                ema.update()
                tot += float(loss.detach()) * len(y)
                n += len(y)
                if (k + 1) % log_every == 0:
                    el = time.time() - t0
                    log(f"  fold {fold} ep {ep_global} step {k + 1}/{len(dl_tr)}  bce {tot / n:.4f}  "
                        f"{el / (k + 1):.2f} s/step  epoch ends in "
                        f"{el / (k + 1) * (len(dl_tr) - k - 1) / 60:.0f} min")
            t_train = time.time() - t0
            rec = {"epoch": ep_global, "stage": stage, "bce": tot / max(n, 1),
                   "lr": [g["lr"] for g in opt.param_groups]}
            t1 = time.time()
            with ema.applied():
                va = pd.DataFrame(predict(model, dl_va, device, dtype), index=va_uids, columns=TARGETS)
            infer_s = time.time() - t1
            rec["val_auc"] = evaluate(y_va, va)["macro_auc"]
            msg = f"  fold {fold} ep {ep_global} (stadio {stage})  bce {rec['bce']:.4f}  val_auc EMA {rec['val_auc']:.4f}"
            if gold_va is not None:
                gm = evaluate(gold_va, va.loc[gold_va.index])
                rec.update(gold_auc=gm["macro_auc"], gold_n=int(len(gold_va)), gold_n_classes=gm["n_classes"])
                msg += f"  gold {gm['macro_auc']:.4f} (n={len(gold_va)})"
                gold_frames.append(va.loc[gold_va.index].assign(epoch=ep_global))
            # The best over both stages; in stage 2 a gain of at least `min_delta` is required, and patience is
            # counted only there.
            gain = rec["val_auc"] - best["auc"]
            if gain > (min_delta if stage == 2 else 0.0):
                best.update(auc=rec["val_auc"], ep=ep_global, state=ema.state(), preds=va, infer_s=infer_s)
                msg += " *"
                stale = 0
            elif stage == 2:
                stale += 1
                stop = bool(patience) and stale >= patience
            rec["seconds"] = round(time.time() - t0, 1)
            rec["train_seconds"] = round(t_train, 1)
            history.append(rec)
            log(msg + f"  {t_train:.0f}s + {infer_s:.0f}s val")
            if stop:
                log(f"  fold {fold}: {patience} epochs without +{min_delta:g}, stopping at ep {ep_global}")
                break

    with torch.no_grad():
        for k, v in best["state"].items():
            named[k].copy_(v.to(named[k].dtype))
    log(f"  fold {fold}: EMA weights of epoch {best['ep']} (val_auc {best['auc']:.4f}, chosen on the fold)")
    extra = {"history": history, "gold_curve": pd.concat(gold_frames) if gold_frames else None}
    return best["preds"], best["infer_s"], len(va_uids), model, best["ep"], {}, extra


def dino_checkpoint(model: DinoKnee, cfg: dict, fold: int) -> dict:
    """The trained weights (heads + last blocks, ~100M for ViT-L): the rest is the public encoder."""
    return {"trained": model.trained_state(), "cfg": cfg, "fold": fold,
            "dino_id": cfg["model"].get("dino_id", DEFAULT_ID)}


def load_for_inference(ckpt: dict, n_slots: int, n_slices: int, device, encoder=None,
                       log=print) -> DinoKnee:
    """Public encoder + trained weights from the checkpoint."""
    model = build_model(ckpt["cfg"], n_slots, n_slices, encoder=encoder, log=log)
    missing = set(model.trained_state()) - set(ckpt["trained"])
    if missing:
        raise KeyError(f"checkpoint missing {len(missing)} trained weights, e.g. {sorted(missing)[:3]}")
    model.load_state_dict(ckpt["trained"], strict=False)
    return model.to(device).eval()
