"""Qwen 3.5 as the student (an experiment, not used by the final model).

Inspired by a public forum report of fine-tuning Qwen 3.5 2B with LoRA on the language model and the
visual encoder trained jointly, 384x384 inputs. Inference fits the time limit (4.8 s/study fp16 on
a T4 at 7.3k tokens) and fp16 = fp32.

The mechanism the 2.5D models lack: there every window is encoded separately and then aggregated
with per-finding attention; here the LM attends over the tokens of all slices of all sequences
together.

How a study enters. Every present slot (sequence x scale) is a video of `K` slices at `P` px, grey
replicated over three channels: the visual encoder splits it into pairs of slices
(`temporal_patch_size` 2), patch 16, 2x2 merge -> `(K/2) * (P/32)^2` tokens per slot, with the
textual timestamps the processor puts between pairs. All slots have the same shape, so the token
block of a slot is always the same: it is computed once from the processor (`Template`) and
repeated for the present slots, followed by a text tail ("Findings:"). The read-out is the hidden
state of the last token -> linear to 12.

Pixels do not go through the processor at every step (0.3-0.5 s of CPU per study): `patchify` does
the same transformation on the GPU (rescale, normalisation, patch order), and the preflight checks
that it matches the processor bit for bit on a real study.

What is trained: the whole visual encoder (including the merger), LoRA on the LM linears
(`LORA_TARGETS`), the head. The LM stays frozen in bf16. The checkpoint carries only this delta
(`delta_state`): the base is reloaded from the public weights (Apache-2.0).
"""
from __future__ import annotations

import collections
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .constants import TARGETS

DEFAULT_ID = "Qwen/Qwen3.5-2B"
# The LM linears with LoRA: full attention (q/k/v/o), linear attention (in_proj_qkv, in_proj_z,
# out_proj) and MLP. Left out: in_proj_a/in_proj_b (the Gated DeltaNet gates, few outputs) and
# lm_head (unused).
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                "out_proj", "gate_proj", "up_proj", "down_proj")
TAIL_TEXT = "Findings:"


def is_qwen(cfg: dict) -> bool:
    return str(cfg.get("model", {}).get("arch", "")).startswith("qwen")


# --- LoRA ----------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """`y = W x + (alpha/r) * B A x`, with `W` frozen and `B` zero at init (the model starts identical
    to the base). A and B in fp32: the only LM weights that are trained.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        z = self.drop(x).to(self.lora_a.dtype) @ self.lora_a.t() @ self.lora_b.t()
        return y + (self.scale * z).to(y.dtype)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        w = self.base.weight.float() + self.scale * (self.lora_b.float() @ self.lora_a.float())
        return w.to(self.base.weight.dtype)


def add_lora(module: nn.Module, r: int, alpha: float, dropout: float,
             targets=LORA_TARGETS) -> list[str]:
    """Replaces in place the `nn.Linear` modules whose name ends in `targets`. Returns the names."""
    done = []
    for name, child in list(module.named_modules()):
        for cname, sub in list(child.named_children()):
            if isinstance(sub, nn.Linear) and cname in targets:
                setattr(child, cname, LoRALinear(sub, r, alpha, dropout))
                done.append(f"{name}.{cname}" if name else cname)
    return done


def merge_lora(module: nn.Module) -> int:
    """Turns every `LoRALinear` back into an `nn.Linear` with the merged weight (inference without LoRA)."""
    n = 0
    for name, child in list(module.named_modules()):
        for cname, sub in list(child.named_children()):
            if isinstance(sub, LoRALinear):
                lin = sub.base
                lin.weight.data.copy_(sub.merged_weight())
                setattr(child, cname, lin)
                n += 1
    return n


# --- pixels and tokens -------------------------------------------------------------------------
def patchify(frames: torch.Tensor, patch: int, tps: int, merge: int) -> tuple[torch.Tensor, list[int]]:
    """`(T, C, H, W)` normalised -> `(t*h*w, C*tps*patch*patch)` in Qwen's processor order (contiguous
    2x2 merge groups), and the grid `[t, h, w]`. With odd `T` the last slice is repeated, as the
    processor does.
    """
    t_, c, h, w = frames.shape
    if t_ % tps:
        frames = torch.cat([frames, frames[-1:].expand(tps - t_ % tps, c, h, w)])
        t_ = frames.shape[0]
    gt, gh, gw = t_ // tps, h // patch, w // patch
    x = frames.reshape(gt, tps, c, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gt * gh * gw, c * tps * patch * patch), [gt, gh, gw]


@dataclass
class Template:
    """The token block of a slot and the text tail, taken once from the processor."""
    block_ids: list[int]
    block_types: list[int]
    tail_ids: list[int]
    tail_types: list[int]
    grid: list[int]                 # [t, h, w] of one slot
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def inputs(self, n_slots: int, device) -> dict:
        ids = self.block_ids * n_slots + self.tail_ids
        types = self.block_types * n_slots + self.tail_types
        out = {"input_ids": torch.tensor([ids], device=device),
               "attention_mask": torch.ones(1, len(ids), dtype=torch.long, device=device),
               "mm_token_type_ids": torch.tensor([types], device=device)}
        if n_slots:
            out["video_grid_thw"] = torch.tensor([self.grid] * n_slots, device=device)
        return out


def template_from_processor(proc, slices: int, img: int) -> tuple[Template, dict]:
    """Calls the processor on a fake video with the shape of a slot and cuts out its block.

    Also returns the raw output (for the bit-exact check of `patchify` in the preflight).
    """
    vs, ve = "<|vision_start|>", "<|vision_end|>"
    video_tok = getattr(proc, "video_token", "<|video_pad|>")
    rng = np.random.default_rng(0)
    # grey replicated over three channels, like our slots: the `patchify` check compares channel 0 with
    # the processor output
    frames = np.repeat(rng.integers(0, 256, (slices, img, img, 1), dtype=np.uint8), 3, axis=-1)
    enc = proc(text=[f"{vs}{video_tok}{ve}{TAIL_TEXT}"], videos=[frames], return_tensors="pt",
               do_resize=False, do_sample_frames=False)
    ids = enc["input_ids"][0].tolist()
    types = (enc["mm_token_type_ids"][0].tolist() if "mm_token_type_ids" in enc
             else [0] * len(ids))
    tail = proc.tokenizer(TAIL_TEXT, add_special_tokens=False)["input_ids"]
    if ids[-len(tail):] != tail:
        raise ValueError(f"the tail {TAIL_TEXT!r} does not end the processor sequence")
    n_block = len(ids) - len(tail)
    grid = enc["video_grid_thw"][0].tolist()
    ip = proc.video_processor if hasattr(proc, "video_processor") else proc.image_processor
    tpl = Template(block_ids=ids[:n_block], block_types=types[:n_block],
                   tail_ids=ids[n_block:], tail_types=types[n_block:], grid=grid,
                   mean=tuple(float(v) for v in ip.image_mean),
                   std=tuple(float(v) for v in ip.image_std))
    return tpl, {"frames": frames, "enc": enc}


# --- the model ----------------------------------------------------------------------------
class QwenKnee(nn.Module):
    """Qwen 3.5 without lm_head, with LoRA on the LM and a linear head on the last token."""

    def __init__(self, base: nn.Module, template: Template, n_classes: int = len(TARGETS),
                 lora: dict | None = None, train_visual: bool = True):
        super().__init__()
        self.qwen = base                                   # Qwen3_5Model (visual + language_model)
        self.template = template
        vc = base.config.vision_config
        self.patch, self.tps, self.merge = vc.patch_size, vc.temporal_patch_size, vc.spatial_merge_size
        for p in self.qwen.parameters():
            p.requires_grad_(False)
        self.lora_names = []
        if lora:
            self.lora_names = add_lora(self.qwen.language_model, int(lora["r"]),
                                       float(lora["alpha"]), float(lora.get("dropout", 0.0)))
        if train_visual:
            for p in self.qwen.visual.parameters():
                p.requires_grad_(True)
        hidden = base.config.text_config.hidden_size
        self.head = nn.Linear(hidden, n_classes)
        self.register_buffer("mean", torch.tensor(template.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(template.std).view(1, 3, 1, 1), persistent=False)

    def pixels(self, vol: torch.Tensor, mask: torch.Tensor, dtype) -> tuple[torch.Tensor | None, int]:
        """`vol (S, K, P, P)` uint8 -> patches of the present slots only, and how many there are."""
        parts = []
        for j in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            x = vol[j].float().div(255.0).unsqueeze(1).expand(-1, 3, -1, -1)   # (K, 3, P, P)
            x = (x - self.mean) / self.std
            p, _ = patchify(x, self.patch, self.tps, self.merge)
            parts.append(p)
        if not parts:
            return None, 0
        return torch.cat(parts).to(dtype), len(parts)

    def features(self, vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """One study -> the hidden state of the last token, what the head reads."""
        dev = vol.device
        pv, n = self.pixels(vol, mask, self.qwen.visual.patch_embed.proj.weight.dtype)
        b = self.template.inputs(n, dev)
        if pv is not None:
            b["pixel_values_videos"] = pv
        return self.qwen(**b).last_hidden_state[0, -1]

    def forward(self, vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """One study -> logits `(n_classes,)`."""
        return self.head(self.features(vol, mask).float())

    def delta_state(self) -> dict:
        """Only the trained weights (visual encoder, LoRA, head): the checkpoint."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: t.detach().to("cpu", copy=True)
                for n, t in self.state_dict().items() if n in names}


def load_base(model_dir: str | Path, dtype, device, attn: str = "sdpa"):
    """The `Qwen3_5Model` (without lm_head) from the public weights, in `dtype`."""
    from transformers import AutoModelForImageTextToText
    full = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=dtype,
                                                       attn_implementation=attn)
    base = full.model
    del full
    return base.to(device)


def build_model(cfg: dict, model_dir, device, dtype=torch.bfloat16, log=print):
    """Base + template + LoRA + head, from the config. `model_dir` = HF id or local folder."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(model_dir))
    c = cfg["cache"]
    tpl, raw = template_from_processor(proc, int(c["slices"]), int(c["img"]))
    base = load_base(model_dir, dtype, device)
    m = cfg["model"]
    model = QwenKnee(base, tpl, lora=m.get("lora"), train_visual=bool(m.get("train_visual", True)))
    # The trained weights in fp32 (master), the frozen LM stays in `dtype`.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.head.float()
    model.to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"  Qwen {model_dir}: {n_all / 1e6:.0f}M parameters, {n_tr / 1e6:.1f}M trained "
        f"({len(model.lora_names)} linears with LoRA); {len(tpl.block_ids)} tokens per slot, "
        f"grid {tpl.grid}")
    return model, proc, raw


def check_patchify(model: QwenKnee, raw: dict) -> float:
    """`patchify` against the processor on the same video: the maximum difference."""
    frames = torch.from_numpy(raw["frames"][..., 0].copy())            # (K, P, P) uint8
    vol = frames.unsqueeze(0).to(model.mean.device)
    pv, _ = model.pixels(vol, torch.tensor([True], device=vol.device), torch.float32)
    ref = raw["enc"]["pixel_values_videos"].float().to(pv.device)
    if pv.shape != ref.shape:
        raise ValueError(f"patchify {tuple(pv.shape)} vs processor {tuple(ref.shape)}")
    return float((pv - ref).abs().max())


# --- training one fold -------------------------------------------------------------------
class _Studies(torch.utils.data.Dataset):
    def __init__(self, cache, uids, labels: pd.DataFrame | None):
        self.cache, self.uids = cache, list(uids)
        self.y = None if labels is None else labels.loc[self.uids, TARGETS].to_numpy(np.float32)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        vol, mask = self.cache.get(self.uids[i])
        y = np.full(len(TARGETS), np.nan, np.float32) if self.y is None else self.y[i]
        return torch.from_numpy(np.ascontiguousarray(vol)), torch.from_numpy(mask.copy()), \
            torch.from_numpy(y), i


def _one(batch):
    return batch[0]


class CollapseError(RuntimeError):
    """The head read-out is the same for all studies: the run no longer measures anything."""


def rel_var(feats) -> float:
    """Variance across studies / mean squared norm of the read-outs: a collapsed model has ~1e-6 here
    against ~7e-3 for the base.
    """
    x = torch.stack([f.detach().float().cpu() for f in feats]).double()
    return float(x.var(0, unbiased=False).sum() / x.pow(2).sum(1).mean().clamp_min(1e-24))


def _loader(ds, shuffle: bool, workers: int, generator=None):
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=shuffle, collate_fn=_one,
                                       num_workers=workers, pin_memory=True,
                                       generator=generator, persistent_workers=workers > 0)


def constant_bce(y: np.ndarray) -> float:
    """The BCE of the best constant predictor (the soft prevalence per column) on `y` with NaN: the floor
    a model that learns something must go below.
    """
    losses, n = 0.0, 0
    for j in range(y.shape[1]):
        col = y[:, j][np.isfinite(y[:, j])]
        if not len(col):
            continue
        p = float(np.clip(col.mean(), 1e-6, 1 - 1e-6))
        losses += float(-(col * np.log(p) + (1 - col) * np.log(1 - p)).sum())
        n += len(col)
    return losses / max(n, 1)


@torch.no_grad()
def cached_features(model: QwenKnee, loader, device, dtype=torch.bfloat16) -> torch.Tensor:
    """The head read-out (last token) for every study of the loader, without gradient: (N, H)."""
    model.eval()
    out = None
    for vol, mask, _, i in loader:
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            h = model.features(vol.to(device, non_blocking=True), mask.to(device)).float()
        if out is None:
            out = torch.zeros(len(loader.dataset), h.shape[-1], device=device)
        out[i] = h
    return out


def fit_probe(head: nn.Linear, feats: torch.Tensor, y: torch.Tensor, steps: int, lr: float,
              weight_decay: float = 0.0, l2: float = 0.0) -> tuple[float, float]:
    """LP of LP-FT (Kumar et al., ICLR 2022): only the head on the frozen features, before fine-tuning.
    With a random head the first gradients flowing into the network are large noise and distort the
    pre-trained features; without it the student collapsed within the first ~75 steps.

    Optimised on standardised features (the huge-norm dimensions of the LM would make Adam
    ill-conditioned) and then mean and scale are folded exactly into the head weights, which stay
    linear on the raw features. Full batch, `masked_bce` on the soft targets. `l2` is sklearn's
    per-column penalty (`||w_c||^2 / (2 C N)`, i.e. `l2 = 1 / (2 C N)`): a probe without penalty on
    2,048 features memorises the training set and leaves near-zero gradients to the fine-tuning.
    Returns (initial BCE, final BCE, without penalty).
    """
    from .train import masked_bce

    mu = feats.mean(0)
    sd = feats.std(0).clamp_min(1e-6)
    z = (feats - mu) / sd
    w = torch.zeros(y.shape[1], feats.shape[1], device=feats.device, requires_grad=True)
    b = torch.zeros(y.shape[1], device=feats.device, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=weight_decay)
    first = last = float("nan")
    for s in range(int(steps)):
        bce = masked_bce(z @ w.t() + b, y)
        loss = bce + float(l2) * (w ** 2).sum() / w.shape[0]
        if s == 0:
            first = float(bce.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(bce.detach())
    with torch.no_grad():
        head.weight.copy_((w / sd).to(head.weight.dtype))
        head.bias.copy_((b - (w * (mu / sd)).sum(1)).to(head.bias.dtype))
    return first, last


@torch.no_grad()
def predict(model: QwenKnee, loader, device, dtype=torch.bfloat16) -> np.ndarray:
    model.eval()
    out = np.zeros((len(loader.dataset), len(TARGETS)), np.float32)
    for vol, mask, _, i in loader:
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            out[i] = torch.sigmoid(model(vol.to(device, non_blocking=True),
                                         mask.to(device))).float().cpu().numpy()
    return out


def resolve_model_dir(cfg: dict) -> str:
    """Local weights folder if present (`KNEE_QWEN_DIR`, Kaggle), otherwise the HF id."""
    d = os.environ.get("KNEE_QWEN_DIR")
    return d if d and Path(d, "config.json").is_file() else cfg["model"].get("qwen_id", DEFAULT_ID)


def train_fold_qwen(cfg, cache, labels, folds, fold, device, log=print, gold_path=None, **_):
    """Same signature and outputs as `train.train_fold`: `run()` writes the fragments."""
    from .labels.gold import load_gold
    from .metrics import evaluate
    from .train import is_full_data, masked_bce

    full = is_full_data(cfg)
    tr_uids = list(labels.index) if full else [u for u in labels.index if folds.get(u, -1) != fold]
    va_uids = [u for u in labels.index if folds.get(u, -1) == fold]
    log(f"  fold {fold}: {len(tr_uids)} train / {len(va_uids)} val (Qwen)")
    gold_va = None
    if gold_path is not None:
        g = load_gold(gold_path)
        gold_va = g.loc[g.index.intersection(va_uids)]
        gold_va = None if gold_va.empty else gold_va

    t = cfg["train"]
    torch.manual_seed(cfg["seed"] + fold)
    model, _, raw = build_model(cfg, resolve_model_dir(cfg), device, log=log)
    err = check_patchify(model, raw)
    log(f"  patchify vs processor: max difference {err:.2e}")
    if err > 1e-5:
        raise ValueError(f"patchify diverges from the processor ({err:.2e}): the pixels are not its own")
    model.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.qwen.config.use_cache = False

    groups = [
        {"params": [p for p in model.qwen.visual.parameters() if p.requires_grad],
         "lr": float(t["lr_visual"]), "weight_decay": float(t.get("weight_decay", 0.01))},
        {"params": [p for n, p in model.named_parameters() if "lora_" in n],
         "lr": float(t["lr_lora"]), "weight_decay": 0.0},
        {"params": list(model.head.parameters()), "lr": float(t["lr_head"]),
         "weight_decay": float(t.get("weight_decay", 0.01))},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    accum, epochs = int(t.get("accum", 8)), int(t["epochs"])
    steps = epochs * math.ceil(len(tr_uids) / accum)
    warm = max(1, int(float(t.get("warmup", 0.05)) * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    workers = int(t.get("workers", 4))
    g_tr = torch.Generator().manual_seed(cfg["seed"] * 1000 + fold)
    dl_tr = _loader(_Studies(cache, tr_uids, labels), True, workers, g_tr)
    dl_va = _loader(_Studies(cache, va_uids, labels), False, workers)
    dtype = torch.bfloat16
    clip = float(t.get("grad_clip", 1.0))
    log_every = int(t.get("log_every", 200))
    # A collapse guard: the spread across studies of the head read-out, over the last `guard_window`
    # training studies. Printed on every log line; from `guard_steps` optimizer steps on, below
    # `guard_min_rel_var` the run stops. It does not change training: it only interrupts it.
    guard_steps = t.get("guard_steps")
    guard_min = float(t.get("guard_min_rel_var", 1e-4))
    recent = collections.deque(maxlen=int(t.get("guard_window", 32)))
    # The second guard: the BCE of the last `guard_window` studies against that of the constant
    # predictor. `rel_var` can stay above threshold while the head learns nothing: geometry does not
    # measure usefulness. With `guard_bce: true`, from `guard_steps` on, a recent BCE not below the
    # constant one stops the run.
    guard_bce = bool(t.get("guard_bce", False))
    const_bce = constant_bce(labels.loc[tr_uids, TARGETS].to_numpy(np.float32))
    recent_bce = collections.deque(maxlen=int(t.get("guard_window", 32)))

    # LP-FT: first only the head on the frozen features of the base (LoRA with B=0, visual encoder as
    # is), then the usual fine-tuning with the head already in place.
    probe = t.get("probe")
    if probe:
        unknown = set(probe) - {"steps", "lr", "weight_decay", "l2", "validate"}
        if unknown:
            raise ValueError(f"train.probe: unknown keys {sorted(unknown)}")
        t0 = time.time()
        feats = cached_features(model, _loader(_Studies(cache, tr_uids, labels), False, workers),
                                device, dtype)
        y_tr = torch.from_numpy(labels.loc[tr_uids, TARGETS].to_numpy(np.float32)).to(device)
        first, last = fit_probe(model.head, feats, y_tr, int(probe.get("steps", 500)),
                                float(probe.get("lr", 1e-2)), float(probe.get("weight_decay", 0.0)),
                                float(probe.get("l2", 0.0)))
        msg = (f"  fold {fold}: LP, head only on frozen features ({len(tr_uids)} studies, "
               f"{time.time() - t0:.0f} s): BCE {first:.4f} -> {last:.4f}, constant {const_bce:.4f}, "
               f"rel_var {rel_var(list(feats)):.2e}")
        del feats, y_tr
        if probe.get("validate", True) and va_uids:
            va0 = pd.DataFrame(predict(model, dl_va, device, dtype), index=va_uids, columns=TARGETS)
            yb0 = (labels.loc[va_uids] > 0.5).astype(float)
            yb0[labels.loc[va_uids].isna()] = np.nan
            msg += f"; val_auc after LP {evaluate(yb0, va0)['macro_auc']:.4f}"
        log(msg)

    history, gold_frames, step = [], [], 0
    trainable = [p for g in groups for p in g["params"]]
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k, (vol, mask, y, _) in enumerate(dl_tr):
            vol, mask, y = vol.to(device, non_blocking=True), mask.to(device), y.to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                h = model.features(vol, mask)
                logit = model.head(h.float())
            recent.append(h.detach().float().cpu())
            loss = masked_bce(logit.float().unsqueeze(0), y.unsqueeze(0))
            if torch.isfinite(y).any():
                recent_bce.append(float(loss.detach()))
            (loss / accum).backward()
            tot += float(loss.detach())
            n += 1
            if (k + 1) % accum == 0 or k + 1 == len(dl_tr):
                torch.nn.utils.clip_grad_norm_(trainable, clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if (k + 1) % log_every == 0:
                el = time.time() - t0
                rv = rel_var(recent)
                rb = float(np.mean(recent_bce)) if recent_bce else float("nan")
                log(f"  fold {fold} ep {ep + 1} study {k + 1}/{len(dl_tr)}  bce {tot / n:.4f}  "
                    f"recent {rb:.4f} (constant {const_bce:.4f})  rel_var {rv:.2e}  "
                    f"{el / (k + 1):.2f} s/study  epoch ends in {el / (k + 1) * (len(dl_tr) - k - 1) / 60:.0f} min")
                if guard_steps is not None and step >= int(guard_steps) and rv < guard_min:
                    raise CollapseError(
                        f"COLLAPSE at step {step} (study {k + 1}, epoch {ep + 1}): rel_var {rv:.2e} "
                        f"< {guard_min:.0e} over the last {len(recent)} studies -> the head reads a fixed point")
                if guard_bce and guard_steps is not None and step >= int(guard_steps) \
                        and rb >= const_bce:
                    raise CollapseError(
                        f"NOT LEARNING at step {step} (study {k + 1}, epoch {ep + 1}): BCE of the "
                        f"last {len(recent_bce)} studies {rb:.4f} >= constant predictor {const_bce:.4f}")
        rec = {"epoch": ep + 1, "bce": tot / max(n, 1)}
        t_train = time.time() - t0
        va = pd.DataFrame(predict(model, dl_va, device, dtype), index=va_uids, columns=TARGETS)
        yb = (labels.loc[va_uids] > 0.5).astype(float)
        yb[labels.loc[va_uids].isna()] = np.nan
        rec["val_auc"] = evaluate(yb, va)["macro_auc"]
        msg = f"  fold {fold} ep {ep + 1}/{epochs}  bce {rec['bce']:.4f}  val_auc {rec['val_auc']:.4f}"
        if gold_va is not None:
            gm = evaluate(gold_va, va.loc[gold_va.index])
            rec.update(gold_auc=gm["macro_auc"], gold_n=int(len(gold_va)), gold_n_classes=gm["n_classes"])
            msg += f"  gold {gm['macro_auc']:.4f} (n={len(gold_va)})"
            gold_frames.append(va.loc[gold_va.index].assign(epoch=ep + 1))
        rec["seconds"] = round(time.time() - t0, 1)
        rec["train_seconds"] = round(t_train, 1)
        history.append(rec)
        log(msg + f"  {rec['seconds']:.0f}s")

    t0 = time.time()
    out = va if epochs else pd.DataFrame(predict(model, dl_va, device, dtype), index=va_uids,
                                         columns=TARGETS)
    infer_s = history[-1]["seconds"] - history[-1]["train_seconds"] if history else time.time() - t0
    extra = {"history": history, "gold_curve": pd.concat(gold_frames) if gold_frames else None}
    return out, infer_s, len(va_uids), model, epochs, {}, extra


def qwen_checkpoint(model: QwenKnee, cfg: dict, fold: int) -> dict:
    """The trained delta in bf16 (visual encoder ~0.66 GB), the base is reloaded separately."""
    return {"delta": {k: v.to(torch.bfloat16) for k, v in model.delta_state().items()},
            "cfg": cfg, "fold": fold, "qwen_id": cfg["model"].get("qwen_id", DEFAULT_ID),
            "lora_names": model.lora_names}


def load_for_inference(ckpt: dict, model_dir, device, dtype=torch.float16, log=print) -> QwenKnee:
    """Base in `dtype` + the checkpoint delta, LoRA merged into the weights: no LoRA at inference."""
    cfg = ckpt["cfg"]
    model, _, _ = build_model(cfg, model_dir, device, dtype=dtype, log=log)
    missing = set(model.delta_state()) - set(ckpt["delta"])
    if missing:
        raise KeyError(f"checkpoint missing {len(missing)} trained weights, e.g. {sorted(missing)[:3]}")
    model.load_state_dict(ckpt["delta"], strict=False)
    n = merge_lora(model.qwen.language_model)
    model.to(dtype)
    model.head.float()
    log(f"  LoRA merged into {n} linears; model in {dtype}")
    return model.eval()
