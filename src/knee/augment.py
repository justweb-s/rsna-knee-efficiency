"""Window augmentation during training: on the GPU, after sampling, without flips.

Without augmentation the only variety between epochs is *which* windows the model sees
(`train._sample_windows`): every window, when it comes back, is identical to the pixel. With a
noisy teacher, what the model memorises includes the teacher's errors.

Three non-negotiable rules.

**No flips.** A horizontal flip swaps medial and lateral: four columns out of twelve (menisci and OA
of the two compartments) would change meaning. The cache already canonicalises the side
(`data/cache.py`, `normalise_laterality`), and that is the only reason the model can tell the two
compartments apart.

**One transform per window, the same on all its channels.** The channels of a window are
contiguous, registered slices: rotating or shifting them differently would misalign them, and
contiguity is the mechanism that pays.

**Zero stays zero.** A never-acquired slot is a black volume and its mask keeps it out of the
attention; augmentation must not invent anything inside it. Geometric padding is zeros and the
intensity jitter is multiplicative: 0 -> 0 by construction.

Without `train.augment` in the config nothing is built and training is unchanged: the run's random
generator is not touched either (augmentation has its own generator).

**The final recipe: mild perturbations only.** In this pipeline zoom, vertical shifts and gain
carry information: the box is in mm and centred on the joint line, and fluid sits at the top of
the percentile scale. The final recipe uses only the perturbations that single-view TTA showed to
be nearly free (horizontal shift of 3%: -0.0004/-0.0006 OOF; windows starting at slice 1:
+0.0001). The three rules still hold: the offset picks another triplet of real contiguous slices
(no interpolation), the shift is one per window over all its channels, a black slot stays black.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F


AUGMENT_ROWS = ("all", "extra")
_UNIT_AMPLITUDES = ("scale", "translate", "intensity", "gamma", "translate_x")


@dataclass(frozen=True)
class AugmentSpec:
    """The recipe: perturbation amplitudes. At zero, each one is off.

    - `scale`: zoom in [1 - scale, 1 + scale]. Zoom > 1 tightens the box (at 0.10 on 110 mm: 100 mm),
      zoom < 1 widens it with a black border (there is no new anatomy to show: the cache is already
      cropped).
    - `rotate_deg`: in-plane rotation in [-r, +r] degrees.
    - `translate`: shift in [-t, +t] of the side, per axis (at 0.05 on 110 mm: 5.5 mm).
    - `intensity`: multiplicative gain in [1 - i, 1 + i].
    - `gamma`: exponent in [1 - g, 1 + g] on values in [0, 1]: moves the contrast without moving the
      extremes. MRI intensities have no units: contrast changes between scanners and sequences anyway,
      and that is what the model must learn to ignore.
    - `translate_x`: horizontal-only shift in [-t, +t] of the side, on the W axis (the same `tx` as
      `tta.affine_windows`). Excludes `translate`, which contains it: in single-view TTA a vertical
      shift of 3% costs up to -0.0033 OOF (the height of the joint line is information), the
      horizontal one -0.0004/-0.0006.
    - `slice_offset`: the window starts `o` slices further, `o` uniform in {0, ..., slice_offset},
      one draw per window. With stride 2 the triplet 1-2-3 never occurs in training and is the `off1`
      TTA view. Needs the window loader reading `depth + slice_offset` slices
      (`KneeStudies(window_extra=...)`).
    - `rows`: "all", every study of the batch; "extra", only the `data.extra` rows. The other rows stay
      bit-identical to no augmentation.
    """
    scale: float = 0.0
    rotate_deg: float = 0.0
    translate: float = 0.0
    intensity: float = 0.0
    gamma: float = 0.0
    translate_x: float = 0.0
    slice_offset: int = 0
    rows: str = "all"

    def __post_init__(self):
        for name in _UNIT_AMPLITUDES:
            v = getattr(self, name)
            if not (0.0 <= float(v) < 1.0):
                raise ValueError(f"augment.{name} must be in [0, 1), got {v}")
        if not (0.0 <= self.rotate_deg < 180.0):
            raise ValueError(f"augment.rotate_deg must be in [0, 180), got {self.rotate_deg}")
        if self.translate and self.translate_x:
            raise ValueError("augment.translate and augment.translate_x together: the second is "
                             "only the horizontal component of the first")
        if isinstance(self.slice_offset, bool) or not isinstance(self.slice_offset, int) \
                or self.slice_offset < 0:
            raise ValueError(f"augment.slice_offset must be an integer >= 0, got "
                             f"{self.slice_offset!r}")
        if self.rows not in AUGMENT_ROWS:
            raise ValueError(f"augment.rows: expected one of {AUGMENT_ROWS}, got {self.rows!r}")

    @classmethod
    def from_config(cls, cfg: dict) -> "AugmentSpec | None":
        """`train.augment` from the config, or None if absent: absent = nothing, not zeros."""
        d = (cfg.get("train") or {}).get("augment")
        if not d:
            return None
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"train.augment: unknown keys {sorted(unknown)}")
        kw = {}
        for k, v in d.items():
            if k == "rows":
                kw[k] = str(v)
            elif k == "slice_offset":
                if isinstance(v, bool) or float(v) != int(v):
                    raise ValueError(f"train.augment.slice_offset must be an integer, got {v!r}")
                kw[k] = int(v)
            else:
                kw[k] = float(v)
        return cls(**kw)

    @property
    def geometric(self) -> bool:
        return bool(self.scale or self.rotate_deg or self.translate or self.translate_x)

    @property
    def photometric(self) -> bool:
        return bool(self.intensity or self.gamma)

    @property
    def plain(self) -> bool:
        """The original form: every row, windows already at the model's depth."""
        return self.rows == "all" and not self.slice_offset

    def describe(self) -> str:
        s = (f"scale ±{self.scale:g} · rotate ±{self.rotate_deg:g}° · translate "
             f"±{self.translate:g} · intensity ±{self.intensity:g} · gamma ±{self.gamma:g}")
        if self.translate_x:
            s += f" · translate_x ±{self.translate_x:g}"
        if self.slice_offset:
            s += f" · slice_offset 0..{self.slice_offset}"
        if self.rows != "all":
            s += f" · only rows {self.rows}"
        return s


def _affine_theta(n: int, spec: AugmentSpec, gen: torch.Generator) -> torch.Tensor:
    """(n, 2, 3): the matrix `affine_grid` uses to map output onto input.

    `affine_grid` works in normalised coordinates [-1, 1], so a shift of a fraction `t` of the side is
    `2t`; and a zoom `s` (bigger image) is obtained by sampling the input at `1/s`, because the matrix
    goes from output to input.
    """
    u = torch.rand(n, 4, generator=gen) * 2.0 - 1.0
    s = 1.0 + u[:, 0] * spec.scale
    a = u[:, 1] * spec.rotate_deg * math.pi / 180.0
    # `translate_x` uses the same draw as the horizontal component of `translate` (the two are mutually
    # exclusive): the same four numbers per window, whatever the recipe.
    tx = u[:, 2] * (spec.translate or spec.translate_x) * 2.0
    ty = u[:, 3] * spec.translate * 2.0
    c, si = torch.cos(a) / s, torch.sin(a) / s
    return torch.stack([torch.stack([c, -si, tx], 1),
                        torch.stack([si, c, ty], 1)], 1)


def augment_windows(w: torch.Tensor, spec: AugmentSpec, gen: torch.Generator) -> torch.Tensor:
    """(B, K, D, H, W) in [0, 1] -> same shape, every window perturbed its own way.

    The parameters are drawn from `gen` (CPU, seeded per fold) and moved to the windows' device: the
    run is reproducible for a fixed seed, on GPU as on CPU.
    """
    if not (spec.geometric or spec.photometric):
        return w
    b, k, d, h, wd = w.shape
    n = b * k
    x = w.reshape(n, d, h, wd)
    if spec.geometric:
        theta = _affine_theta(n, spec, gen).to(device=x.device, dtype=x.dtype)
        grid = F.affine_grid(theta, (n, d, h, wd), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if spec.photometric:
        v = torch.rand(n, 2, generator=gen) * 2.0 - 1.0
        gain = (1.0 + v[:, 0] * spec.intensity).view(n, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        gamma = (1.0 + v[:, 1] * spec.gamma).view(n, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        x = x.clamp(min=0.0).pow(gamma).mul(gain).clamp_(0.0, 1.0)
    return x.reshape(b, k, d, h, wd)


def _offset_slices(x: torch.Tensor, spec: AugmentSpec, gen: torch.Generator,
                   depth: int) -> torch.Tensor:
    """(n, K, depth + slice_offset, H, W) -> (n, K, depth, H, W): window j starts `o_j` slices further,
    `o_j` uniform in {0, ..., slice_offset}. Without an offset, `x` unchanged.
    """
    if not spec.slice_offset:
        return x
    n, k = x.shape[:2]
    o = torch.randint(0, spec.slice_offset + 1, (n, k), generator=gen).to(x.device)
    out = x[:, :, :depth].clone(memory_format=torch.contiguous_format)
    for v in range(1, spec.slice_offset + 1):
        sel = o == v
        out[sel] = x[:, :, v:v + depth][sel]
    return out


def apply_augment(w: torch.Tensor, spec: AugmentSpec, gen: torch.Generator, depth: int,
                  rows: torch.Tensor | None = None) -> torch.Tensor:
    """The training entry point: (B, K, depth + slice_offset, H, W) -> (B, K, depth, H, W).

    `rows` (bool (B,), the `data.extra` rows when `spec.rows` is "extra") restricts the perturbation to
    those rows; the others come out with the first `depth` slices of every window, i.e. bit for bit the
    window of the path without augmentation. The draws concern only the perturbed windows, in this
    order: slice offsets, geometry, intensity. With the plain recipe (`spec.plain`, `rows` None) it is
    just `augment_windows`: same draws, same numbers.
    """
    if w.shape[2] != depth + spec.slice_offset:
        raise ValueError(f"augment: windows of {w.shape[2]} slices, expected {depth} + "
                         f"slice_offset {spec.slice_offset}")
    if rows is None:
        return augment_windows(_offset_slices(w, spec, gen, depth), spec, gen)
    out = w[:, :, :depth].clone(memory_format=torch.contiguous_format)
    idx = rows.to(device=w.device, dtype=torch.bool).nonzero(as_tuple=True)[0]
    if idx.numel():
        out[idx] = augment_windows(_offset_slices(w[idx], spec, gen, depth), spec, gen)
    return out


# ---------------------------------------------------------------------------------------
# Mixup between studies (an option, not used by the final model)
#
# Not a per-window augmentation like the ones above: it mixes TWO studies, pixels and targets with
# the same weight λ ~ Beta(α, α) (Zhang et al. 2018). The three rules of the module still hold: no
# flips; the same λ on all channels of a window (contiguous slices stay registered); and zero stays
# zero where it is zero in both studies.
#
# Three choices, stated because each could have gone differently.
#
# **The partner is the study at the other end of the batch** (`flip(0)`, like `timm` in 'batch'
# mode): the batch is already shuffled by the sampler, and the reversal guarantees that no study is
# mixed with itself (batch of 8, even). No extra random numbers for the pairing.
#
# **Windows are mixed position by position.** The draw of the `k_train` windows is one per batch
# (`WindowBatchSampler`, `_sample_windows`): window j of study i and that of its partner are in the
# same slot at the same height, so two knees are summed in the same sequence and the same position,
# not a sagittal onto a coronal.
#
# **The slot mask is the union.** A slot present in only one of the two studies carries λ times the
# real image (or 1 - λ): a darker image, not a black volume, and attention must see it. Targets are
# mixed with the same λ; a NaN cell in either stays NaN (excluded by `masked_bce`).
#
# No CutMix: the findings are small and localised (menisci, ligaments), a patch can cover the lesion
# while the target stays positive: label noise exactly where the model is weak.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MixupSpec:
    """`train.mixup`: `alpha` of the symmetric Beta, `prob` the fraction of mixed batches."""
    alpha: float
    prob: float = 1.0

    def __post_init__(self):
        if not (float(self.alpha) > 0.0):
            raise ValueError(f"mixup.alpha must be > 0, got {self.alpha}")
        if not (0.0 < float(self.prob) <= 1.0):
            raise ValueError(f"mixup.prob must be in (0, 1], got {self.prob}")

    @classmethod
    def from_config(cls, cfg: dict) -> "MixupSpec | None":
        """`train.mixup` from the config, or None if absent: absent = no mixup."""
        d = (cfg.get("train") or {}).get("mixup")
        if not d:
            return None
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"train.mixup: unknown keys {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in d.items()})

    def describe(self) -> str:
        return f"λ ~ Beta({self.alpha:g}, {self.alpha:g}) on {self.prob:.0%} of the batches, partner = reversed batch"


def mixup_batch(w: torch.Tensor, mask: torch.Tensor, y: torch.Tensor,
                weight: torch.Tensor | None, spec: MixupSpec, rng):
    """Mixes every study of the batch with its partner (the reversed batch).

    w (B, K, D, H, W), mask (B, S) bool, y (B, T), weight (B,) or (B, T) or None. `rng` is its own
    `numpy.random.Generator`, seeded per fold: two draws per batch (whether to mix, and λ), always the
    same two, so the sequence does not depend on the outcomes. Returns (w, mask, y, weight, λ);
    λ = 1.0 when the batch is not mixed.
    """
    u, lam = float(rng.random()), float(rng.beta(spec.alpha, spec.alpha))
    if u >= spec.prob or w.shape[0] < 2:
        return w, mask, y, weight, 1.0
    w = w.mul(lam).add_(w.flip(0), alpha=1.0 - lam)
    mask = mask | mask.flip(0)
    y = y * lam + y.flip(0) * (1.0 - lam)
    if weight is not None:
        weight = weight * lam + weight.flip(0) * (1.0 - lam)
    return w, mask, y, weight, lam
