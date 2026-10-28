"""Exponential moving average of the weights (EMA): an extra read-out next to the last epoch.

It is not a training variable. The shadow is updated AFTER every optimizer step and never enters
the gradient, the scheduler or the random generators: the trajectory of the `last` weights is
bit-identical with or without `train.ema`. It is a second checkpoint of the same run, like
`select: best`, and is evaluated alongside it (`oof_ema<decay>.csv`, `ema` block in
`metrics.json`), not instead of it.

Why bias correction instead of timm's warm-up. In a ~9,000-step schedule a shadow starting from
the ImageNet weights with decay 0.9998 ends with e^(-8800/5000) = 17% of its weight still on the
initial weights: it would be an average with an untrained model. Here the shadow starts at zero
and is read divided by (1 - d^t): exactly the geometrically weighted mean of the iterates seen,
with no initial bias and no warm-up constant to tune. BatchNorm statistics follow the same
average (they are float tensors of the state_dict); `num_batches_tracked` is copied.
"""
from __future__ import annotations

import copy

import torch


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"train.ema: decay must be in (0, 1), got {decay!r}")
        self.decay = float(decay)
        self.steps = 0
        # The module that gets evaluated: same architecture, weights written by `materialize`.
        self.module = copy.deepcopy(model).eval()
        # The fast DenseNet training path puts into the encoder instance a `forward` that is a closure
        # over the ORIGINAL encoder, and deepcopy copies it by reference (functions are not copied):
        # the shadow would be evaluated with the live weights. It is removed from the shadow: timm forward.
        for m in self.module.modules():
            if "_fastdense" in m.__dict__:
                from .fastdense import uninstall
                uninstall(m)
        for p in self.module.parameters():
            p.requires_grad_(False)
        src = model.state_dict()
        dst = self.module.state_dict()
        if list(src) != list(dst):
            raise RuntimeError("shadow state_dict differs from the model's")
        self._src_f = [t for t in src.values() if t.dtype.is_floating_point]
        self._dst_f = [t for t in dst.values() if t.dtype.is_floating_point]
        self._src_i = [t for t in src.values() if not t.dtype.is_floating_point]
        self._dst_i = [t for t in dst.values() if not t.dtype.is_floating_point]
        # The actual accumulator: starts at zero, in float32 whatever the model dtype.
        self._acc = [torch.zeros_like(t, dtype=torch.float32) for t in self._src_f]

    @torch.no_grad()
    def update(self, model: torch.nn.Module | None = None) -> None:
        """Call after EVERY effective optimizer step (not after a step skipped by the GradScaler:
        that step did not move the weights).
        """
        d = self.decay
        torch._foreach_mul_(self._acc, d)
        torch._foreach_add_(self._acc, [t.detach().float() for t in self._src_f], alpha=1.0 - d)
        self.steps += 1

    @torch.no_grad()
    def materialize(self) -> torch.nn.Module:
        """Writes the bias-corrected mean, s_t / (1 - d^t), into the shadow and returns it."""
        if self.steps == 0:
            raise RuntimeError("EMA never updated: nothing to materialize")
        corr = 1.0 / (1.0 - self.decay ** self.steps)
        for dst, acc in zip(self._dst_f, self._acc):
            dst.copy_((acc * corr).to(dst.dtype))
        for dst, src in zip(self._dst_i, self._src_i):
            dst.copy_(src)
        return self.module

    @property
    def tag(self) -> str:
        """`ema0.9995`: the file name of this shadow (`ema0.9995_fold0.pt`)."""
        return f"ema{self.decay:g}"


def ema_decays(cfg: dict) -> list[float]:
    """The decays declared in `train.ema`, or [] if the key is missing or null.

    Accepts `train.ema: {decays: [0.999, 0.9995]}`, a list, or a single number.
    """
    spec = cfg.get("train", {}).get("ema")
    if spec is None:
        return []
    if isinstance(spec, dict):
        spec = spec.get("decays", [])
    if isinstance(spec, (int, float)):
        spec = [spec]
    decays = [float(d) for d in spec]
    if len(set(decays)) != len(decays):
        raise ValueError(f"train.ema: repeated decays in {decays}")
    for d in decays:
        if not 0.0 < d < 1.0:
            raise ValueError(f"train.ema: decay must be in (0, 1), got {d}")
    return decays
