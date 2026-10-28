"""Pooling INSIDE a ViT window, from Matteo Vitali's v61m head (not used by the DenseNet).

Ported from his `IntraSlicePool` with its choices and v61m default values (k 24, 4 heads, 3x3
blur, rank-96 mix). The reason it is needed is his measurement with a frozen encoder: the mean of
the patch tokens (timm `global_pool: avg`) loses 0.019 mean AUC against the CLS token, and 0.091
on the lateral meniscus.

Two things come out of a window:

- `tok` (N, D), the shared vector: CLS + projection (zero at init) of the mean of the k strongest
  cells per channel. It is the vector that receives slot identity, context and attention over the
  windows, like the encoder output in the other models.
- `delta` (N, T, D), one per finding: H attention maps per finding over the cells (per-slot
  weights, `attn_delta`), each head reads only its block of D/H channels; minus the spatial mean,
  times `gamma`. The vector of finding t on the window is `tok + delta[:, t]`.

IDENTITY AT STEP 0. Map logits at zero -> uniform softmax -> weighted mean = spatial mean ->
`delta` exactly zero; `topk_proj` at zero -> `tok` = CLS. Thanks to the centring, `gamma` starts at
1 and the map receives full gradient from the first step: with a zero weight in front of it the
gradient of the maps would be null until that weight moves.

Deliberate differences from the original module: `delta` instead of `cls + delta` (the CLS goes
through `tok`, which also carries slot and context); the "plane" is the cache slot (8 = 4 sequences
x 2 scales), not his 5 planes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Parameters without weight decay: concentrating attention on one cell among hundreds needs logits
# with a large norm, and decay pulls towards the uniform map.
NO_DECAY = ("attn_base", "attn_delta", "gamma")


class IntraSlicePool(nn.Module):
    """patch (N, P, D) + cls (N, D) + plane (N,) -> tok (N, D), delta (N, T, D)."""

    def __init__(self, dim: int, n_targets: int, n_planes: int, k: int = 24, n_heads: int = 4,
                 blur: bool = True, blur_sigma: float = 1.0, head_mix_rank: int | None = 96):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"IntraSlicePool: dim {dim} not divisible by {n_heads} heads")
        self.dim, self.n_targets, self.n_planes = int(dim), int(n_targets), int(n_planes)
        self.n_heads, self.head_dim = int(n_heads), int(dim) // int(n_heads)
        self.k = int(k)
        # Shared branch: CLS + zero-initialised projection of the mean of the k strongest cells.
        self.topk_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.topk_proj.weight)
        # Per-finding branch: [T, H, D] the shared weights, [planes, T, H, D] the per-plane offset. The
        # logits read all D channels, the values are split across heads.
        self.attn_base = nn.Parameter(torch.zeros(n_targets, self.n_heads, dim))
        self.attn_delta = nn.Parameter(torch.zeros(n_planes, n_targets, self.n_heads, dim))
        self.head_mix = None
        if head_mix_rank:
            self.head_mix = nn.Sequential(nn.Linear(dim, int(head_mix_rank), bias=False),
                                          nn.Linear(int(head_mix_rank), dim, bias=False))
            nn.init.normal_(self.head_mix[0].weight, std=0.02)
            nn.init.zeros_(self.head_mix[1].weight)
        self.gamma = nn.Parameter(torch.ones(n_targets, dim))
        # A lesion occupies NEIGHBOURING cells: blurring the logits before the softmax (3x3, replicated
        # border) shares evidence between neighbours without losing resolution.
        self.blur = bool(blur)
        if self.blur:
            g = torch.tensor([math.exp(-(i * i) / (2 * blur_sigma ** 2)) for i in (-1, 0, 1)])
            g = g / g.sum()
            self.register_buffer("blur_k", (g[:, None] * g[None, :]).view(1, 1, 3, 3),
                                 persistent=False)

    def logits(self, patch: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
        """patch (N, P, D) -> (N, T, H, P), blurred if `blur`."""
        n, p, _ = patch.shape
        w = (self.attn_base.unsqueeze(0) + self.attn_delta[plane]).to(patch.dtype)  # (N, T, H, D)
        lg = torch.einsum("nthd,npd->nthp", w, patch)
        if self.blur:
            s = int(round(math.sqrt(p)))
            if s * s != p:
                raise ValueError(f"IntraSlicePool: non-square grid, {p} cells")
            x = lg.reshape(-1, 1, s, s)
            x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), self.blur_k.to(x.dtype))
            lg = x.reshape(n, self.n_targets, self.n_heads, p)
        return lg

    def forward(self, patch: torch.Tensor, cls: torch.Tensor, plane: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        n, p, _ = patch.shape
        kk = max(1, min(self.k, p))
        tok = cls + self.topk_proj(patch.topk(kk, dim=1).values.mean(1))           # (N, D)
        # Softmax in float32: over hundreds of nearly uniform cells bf16 loses resolution.
        a = self.logits(patch, plane).float().softmax(-1).to(patch.dtype)          # (N, T, H, P)
        pv = patch.view(n, p, self.n_heads, self.head_dim)
        per = torch.einsum("nthp,nphc->nthc", a, pv).reshape(n, self.n_targets, self.dim)
        per = per - patch.mean(dim=1, keepdim=True)                                # centring
        if self.head_mix is not None:
            per = per + self.head_mix(per)
        return tok, self.gamma.to(per.dtype).unsqueeze(0) * per                    # (N, T, D)
