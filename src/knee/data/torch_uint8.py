"""`cache._to_uint8` in torch, bit for bit, to run it on the GPU instead of in the workers.

Once the pixel reader, `channels_last`, BatchNorm folding and the second T4 are in place, inference
is bound by the CPU of the workers, and the percentiles and uint8 normalisation are about a
quarter of that CPU. Here they are redone on the GPU, which has headroom.

Bit-identical to NumPy 2 (linear `np.percentile` + the float64 normalisation of `_to_uint8`,
NEP 50), because every step is exact or rounded the same way:
- order statistics are selections (`torch.topk`), not estimates: the same float32 values;
- the interpolation is that of `numpy.lib._function_base_impl._lerp`, operation by operation:
  difference in float32, then product and sum in float64, and the `t >= 0.5` branch; `gamma` is
  computed by NumPy itself (`_get_gamma`);
- subtraction, multiplication and division in float64 are correctly rounded on both CPU and GPU,
  as long as no reciprocal is involved: `255 / (hi - lo)` is a tensor-tensor division (PyTorch
  computes `scalar / tensor` as a reciprocal times the scalar);
- `clamp` and the cast to uint8 (truncation) are exact.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

import torch

from .cache import CLIP_PCT, _np_fb


@lru_cache(maxsize=8)
def _ranks(n: int) -> tuple[int, int, np.ndarray]:
    """For `n` values: the ranks `k1`, `k2` of the two lower order statistics of the tails and the
    `gamma` (float64) of linear `np.percentile(..., CLIP_PCT)`.
    """
    q = np.true_divide(np.asanyarray(CLIP_PCT), 100)
    method = _np_fb._QuantileMethods["linear"]
    virtual = np.asanyarray(method["get_virtual_index"](n, q))
    prev = np.floor(virtual).astype(np.intp)
    if n < 4 or q.shape != (2,) or not (0 < q[0] < 0.5 < q[1] < 1) or prev[1] + 1 >= n:
        raise ValueError(f"percentiles {CLIP_PCT} over {n} values: unsupported case")
    gamma = _np_fb._get_gamma(virtual, prev.astype(virtual.dtype), method)
    return int(prev[0]), int(prev[1]), np.asarray(gamma, dtype=np.float64)


def to_uint8_torch(x: torch.Tensor) -> torch.Tensor:
    """(..., K, H, W) float32 -> uint8: `cache._to_uint8` applied to every stack (K, H, W)."""
    if x.dtype != torch.float32:
        raise TypeError(f"expected a float32 stack, got {x.dtype}")
    n = x.shape[-3] * x.shape[-2] * x.shape[-1]
    flat = x.reshape(-1, n)
    k1, k2, gamma = _ranks(n)
    r = n - k2                                  # values from rank k2 up
    low = torch.topk(flat, k1 + 2, dim=1, largest=False, sorted=True).values
    high = torch.topk(flat, r, dim=1, largest=True, sorted=True).values   # descending
    prev = torch.stack([low[:, k1], high[:, r - 1]], 1)      # sorted[k1], sorted[k2]
    nxt = torch.stack([low[:, k1 + 1], high[:, r - 2]], 1)   # sorted[k1+1], sorted[k2+1]
    return _normalize_rows(x, flat, prev, nxt, gamma)


_GAMMA: dict = {}


def _gamma_on(gamma: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    """`gamma` and `1 - gamma` on the device, copied once: a copy from non-pinned RAM for every stack
    would synchronise the stream (eight times per study at 512 px).
    """
    key = (gamma.tobytes(), str(device))
    if key not in _GAMMA:
        _GAMMA[key] = (torch.as_tensor(gamma, dtype=torch.float64, device=device),
                       torch.as_tensor(1 - gamma, dtype=torch.float64, device=device))
    return _GAMMA[key]


def _normalize_rows(x: torch.Tensor, flat: torch.Tensor, prev: torch.Tensor, nxt: torch.Tensor,
                    gamma: np.ndarray, sub: torch.Tensor | None = None,
                    out_buf: torch.Tensor | None = None) -> torch.Tensor:
    """From the order statistics (R, 2) to the percentiles (NumPy's `_lerp`) and the uint8 values."""
    t, one_minus_t = _gamma_on(gamma, x.device)
    # `_lerp`: diff_b_a = b - a (float32); a + diff_b_a * t; where t >= 0.5, b - diff_b_a * (1 - t).
    # Separate operations: no FMA.
    diff = (nxt - prev).double()
    lerp = prev.double() + diff * t
    lerp = torch.where(t >= 0.5, nxt.double() - diff * one_minus_t, lerp)
    lo, hi = lerp[:, 0:1], lerp[:, 1:2]
    use_fused = os.environ.get("KNEE_FUSED_UINT8", "off") == "on"
    chk = sub if (use_fused and sub is not None) else flat
    bad = (torch.isnan(chk).any(dim=1, keepdim=True) | ~torch.isfinite(lo)
           | ~torch.isfinite(hi) | (hi <= lo))
    scale = torch.full_like(hi, 255.0) / (hi - lo)
    if use_fused:
        # The same float64 computations in a single pass, without building and searching 255 thresholds.
        # The module checks the kernel on first use and keeps the usual fallback.
        from .fused_uint8 import normalize
        fused = normalize(flat, lo, scale, bad, out=out_buf)
        if fused is not None:
            return fused.reshape(x.shape)
    out = _levels_by_thresholds(flat, lo, scale, bad) if _THRESHOLDS[0] else None
    if out is None:
        y = flat.double()
        y.sub_(lo).mul_(scale).clamp_(0, 255)
        out = y.to(torch.uint8)
    out.masked_fill_(bad, 0)
    res = out.reshape(x.shape)
    if out_buf is not None:
        out_buf.copy_(res)
        return out_buf
    return res


# Threshold normalisation. `(x - lo) * scale` in float64, then clamp to [0, 255] and truncation, is a
# monotonic (non-decreasing) function of the float32 x: the uint8 of x is how many levels L = 1..255
# have f(x) >= L, i.e. how many thresholds t_L <= x, with t_L the smallest float32 for which
# f(t_L) >= L. The 255 thresholds per stack are found in float64 with the same arithmetic (estimate
# `lo + L / scale` and a few `nextafter` steps), then one `searchsorted` per stack in float32: one
# pass at 4 + 4 bytes per pixel instead of five in float64 (~64 bytes per pixel, 50 million pixels
# per study at 512 px). Same uint8 values bit for bit: the thresholds are checked (f(t) >= L and
# f(before t) < L) and, if one fails, it is redone in float64.
_THRESHOLDS = [True]
_THR_STEPS = 6
# Deferred check. The two `bool()` below are synchronous reads from the GPU: in the preparation thread
# (`infer._prep_raw_ahead`) they waited for the preparation stream, whose kernels queue behind the
# model's on the same SMs (40% of the thread time was spent on `isfinite(lo).all()`). With
# `_DEFER[0]` the check stays on the device (`_PENDING`) and the caller reads it once per batch
# (`take_pending`, after the synchronisation it does anyway); if it fails the batch is redone without
# thresholds. Rows with non-finite `lo`/`scale` are in `bad` and become zero in any case: they are
# not part of the check.
_DEFER = [False]
_PENDING: list = []


def take_pending(device) -> torch.Tensor | None:
    """The pending checks on `device` in a pinned bool tensor, copied without waiting on the current
    stream (read after synchronising it); None if there are none.
    """
    mine = [p for p in _PENDING if p.device == torch.device(device)]
    if not mine:
        return None
    _PENDING[:] = [p for p in _PENDING if p.device != torch.device(device)]
    host = torch.empty(len(mine), dtype=torch.bool, pin_memory=True)
    host.copy_(torch.stack(mine), non_blocking=True)
    return host


def _levels_by_thresholds(flat: torch.Tensor, lo: torch.Tensor, scale: torch.Tensor,
                          bad: torch.Tensor | None = None):
    """(R, n) float32 -> (R, n) uint8 like `clamp((x - lo) * scale, 0, 255)` truncated, or None if the
    thresholds cannot be found exactly (then the float64 path). `lo`, `scale`: (R, 1) float64.
    With `_DEFER[0]` (and `bad`, the rows the caller zeroes) there are no reads from the GPU: the check
    goes into `_PENDING`.
    """
    defer = _DEFER[0] and bad is not None
    if not defer and (not torch.isfinite(lo).all() or not torch.isfinite(scale).all()):
        return None
    dev = flat.device
    levels = torch.arange(1, 256, dtype=torch.float64, device=dev).unsqueeze(0)       # (1, 255)

    def f(t: torch.Tensor) -> torch.Tensor:
        return (t.double() - lo) * scale

    t = (lo + levels / scale).float()
    inf = torch.full((), float("inf"), dtype=torch.float32, device=dev)
    for _ in range(_THR_STEPS):                 # down while the previous value also exceeds L
        down = torch.nextafter(t, -inf)
        t = torch.where(f(down) >= levels, down, t)
    for _ in range(_THR_STEPS):                 # up while t does not exceed L
        t = torch.where(f(t) >= levels, t, torch.nextafter(t, inf))
    ok = (f(t) >= levels) & (f(torch.nextafter(t, -inf)) < levels)
    if defer:
        _PENDING.append((ok.all(dim=1, keepdim=True) | bad).all())
    elif not bool(ok.all()):
        return None
    return torch.searchsorted(t.contiguous(), flat.contiguous(), right=True,
                              out_int32=True).to(torch.uint8)


# --- histogram-based order statistics (`to_uint8_torch_hist`) -------------------------
# At 512 px a stack is 24 x 512 x 512 = 6.3 million values and the 1/99 percentile tails are 63
# thousand: `topk` would cost tens of seconds over 1,300 studies. Here rank k is found with a radix
# selection over the 32 bits of the float, mapped to an int32 with the same order (`_order_keys`):
# a 4,096-digit histogram on the 12 high bits, then two of 1,024 on the lower 10 + 10, restricted to
# the values whose digits were already found. It is a selection, not an estimate: the same float32
# as `np.partition`.
# The histograms are `torch.histc` with fixed bounds on the digits as float32 (exact integers, step
# 1): on CUDA `bincount` reads min and max on the CPU at every call (two GPU waits), `histc` with
# given bounds does not, and it ignores out-of-range values, so excluded digits are written as -1
# instead of all landing in one bin (atomics on the same address). Digits and remainders stay tensors:
# no synchronisation in the middle of a batch.
_L1, _L2 = 20, 10               # shifts: 12 high bits, then 10, then 10
_N1, _N2 = 1 << (32 - _L1), 1 << _L2


def _order_keys(flat: torch.Tensor) -> torch.Tensor:
    """float32 (n,) -> int32 with the same order as the floats (negatives flipped)."""
    b = flat.view(torch.int32)
    return torch.where(b < 0, b ^ 0x7FFFFFFF, b)


def _keys_to_float(k: torch.Tensor) -> torch.Tensor:
    return torch.where(k < 0, k ^ 0x7FFFFFFF, k).view(torch.float32)


def _digit_pick(counts: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The digit (0-d float) containing rank `r` (0-d float) given the counts per digit (float32, exact
    integers below 2^24), and the rank inside the digit.
    """
    cum = counts.cumsum(0)
    d = torch.searchsorted(cum, r.reshape(1), right=True).clamp_(max=cum.numel() - 1)
    return d[0].to(counts.dtype), r - (cum[d] - counts[d])[0]


def _kth_pair(key: torch.Tensor, d1, c1, d2, d3, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The keys of rank `k` and `k+1` (0-based) among `key`."""
    neg = torch.full((), -1.0, dtype=d1.dtype, device=key.device)
    r = torch.full((), float(k), dtype=d1.dtype, device=key.device)   # no copy: no waiting
    b1, r = _digit_pick(c1, r)
    m = d1 == b1
    b2, r = _digit_pick(torch.histc(torch.where(m, d2, neg), _N2, 0, _N2), r)
    m &= d2 == b2
    b3, _ = _digit_pick(torch.histc(torch.where(m, d3, neg), _N2, 0, _N2), r)
    v = (((b1.to(torch.int64) - _N1 // 2) << _L1) | (b2.to(torch.int64) << _L2)
         | b3.to(torch.int64)).to(torch.int32)
    n_le = (key <= v).sum()
    top = torch.full((), 2 ** 31 - 1, dtype=torch.int32, device=key.device)
    nxt = torch.where(n_le > k + 1, v, torch.where(key > v, key, top).amin())
    return v, nxt


def to_uint8_torch_hist(x: torch.Tensor) -> torch.Tensor:
    """`to_uint8_torch` (the same uint8 values bit for bit as `cache._to_uint8`) with histogram-based
    order statistics instead of `topk`: (..., K, H, W) float32 -> uint8.
    """
    if x.dtype != torch.float32:
        raise TypeError(f"expected a float32 stack, got {x.dtype}")
    n = x.shape[-3] * x.shape[-2] * x.shape[-1]
    if n >= 1 << 24:
        return to_uint8_torch(x)                # float32 counts no longer exact
    flat = x.reshape(-1, n)
    k1, k2, gamma = _ranks(n)
    prevs, nxts = [], []
    for row in flat:
        key = _order_keys(row.contiguous())
        d1 = ((key >> _L1) + _N1 // 2).float()
        c1 = torch.histc(d1, _N1, 0, _N1)
        d2 = ((key >> _L2) & (_N2 - 1)).float()
        d3 = (key & (_N2 - 1)).float()
        lo = _kth_pair(key, d1, c1, d2, d3, k1)
        hi = _kth_pair(key, d1, c1, d2, d3, k2)
        prevs.append(torch.stack([lo[0], hi[0]]))
        nxts.append(torch.stack([lo[1], hi[1]]))
    prev = _keys_to_float(torch.stack(prevs))
    nxt = _keys_to_float(torch.stack(nxts))
    return _normalize_rows(x, flat, prev, nxt, gamma)


# --- several stacks per launch (`to_uint8_torch_hist_batch`) -----------------------------------
# The single-stack version launches ~25 kernels per stack: at 512 px, eight stacks per study, that is
# tens of seconds of main-process CPU in launches (the same 4 vCPUs as decoding) plus GPU time. Here
# `_BATCH` stacks together: digits of 11 + 11 + 10 bits with the stack index inside the digits
# (`d + stack * bins`), one `histc` per level for all stacks. The bins are a power of two and match
# the width of the range, so the `histc` bin is the exact digit (multiplying and dividing by a power
# of two does not round); 4 stacks x 2,048 bins in float32 = 32 KB, the histogram stays in shared
# memory.
_B1, _B2, _B3 = 11, 11, 10
_BATCH = 4
# Percentiles on one pixel every `_PCT_STRIDE`: the exact selection is 84% of the GPU preparation
# (76 of 90 ms per study, 46 million pixels). 1 = exact (bit-identical to `cache._to_uint8`); >1 =
# order statistics of the subsample, all pixels normalised as before: not bit-identical, checked on
# the predictions (Spearman 0.99997 with stride 7). A prime stride, against aliasing with 512-pixel
# rows.
_PCT_STRIDE = [int(os.environ.get("KNEE_PCT_STRIDE", "1"))]


_HIST_CONST: dict = {}


def _hist_consts(R: int, k1: int, k2: int, device):
    key = (R, k1, k2, str(device))
    c = _HIST_CONST.get(key)
    if c is None:
        n1, n2, n3 = 1 << _B1, 1 << _B2, 1 << _B3
        ar = torch.arange(R, device=device, dtype=torch.int32).unsqueeze(1)
        off1_i, off2_i, off3_i = ar * n1, ar * n2, ar * n3
        neg = torch.full((), -1.0, device=device, dtype=torch.float32)
        top = torch.full((), 2 ** 31 - 1, dtype=torch.int32, device=device)
        r0 = torch.tensor([[float(k1), float(k2)]], device=device, dtype=torch.float32).expand(R, 2).contiguous()
        c = (off1_i, off2_i, off3_i, off1_i.float(), off2_i.float(), neg, top, r0)
        _HIST_CONST[key] = c
    return c


def _pick_rows(counts: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: the digit containing rank `r` (R, ...) and the rank inside the digit."""
    cum = counts.cumsum(1)
    d = torch.searchsorted(cum, r, right=True).clamp_(max=counts.shape[1] - 1)
    return d, r - (cum.gather(1, d) - counts.gather(1, d))


def _hist_order_stats(flat: torch.Tensor, k1: int, k2: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(R, n) float32 -> (prev, nxt) (R, 2): the order statistics k1, k2 and the next ones."""
    R = flat.shape[0]
    n1, n2, n3 = 1 << _B1, 1 << _B2, 1 << _B3
    off1_i, off2_i, off3_i, off1_f, off2_f, neg, top, r0 = _hist_consts(R, k1, k2, flat.device)
    key = _order_keys(flat)
    d1 = (((key >> (_B2 + _B3)) + (1 << (_B1 - 1))) + off1_i).float()
    d2 = (((key >> _B3) & (n2 - 1)) + off2_i).float()
    d3 = ((key & (n3 - 1)) + off3_i).float()
    c1 = torch.histc(d1, R * n1, 0, R * n1).view(R, n1)
    b1, r = _pick_rows(c1, r0)
    b1_f = b1.to(d1.dtype) + off1_f
    m0 = d1 == b1_f[:, 0:1]
    m1 = d1 == b1_f[:, 1:2]
    c2_0 = torch.histc(torch.where(m0, d2, neg), R * n2, 0, R * n2).view(R, n2)
    c2_1 = torch.histc(torch.where(m1, d2, neg), R * n2, 0, R * n2).view(R, n2)
    b2_all, r_all = _pick_rows(torch.cat([c2_0, c2_1], 0), r.T.contiguous().view(2 * R, 1))
    b2 = b2_all.view(2, R).T
    b2_f = b2.to(d2.dtype) + off2_f
    m0 &= d2 == b2_f[:, 0:1]
    m1 &= d2 == b2_f[:, 1:2]
    c3_0 = torch.histc(torch.where(m0, d3, neg), R * n3, 0, R * n3).view(R, n3)
    c3_1 = torch.histc(torch.where(m1, d3, neg), R * n3, 0, R * n3).view(R, n3)
    b3_all, _ = _pick_rows(torch.cat([c3_0, c3_1], 0), r_all)
    b3 = b3_all.view(2, R).T
    v = (((b1 - (1 << (_B1 - 1))) << (_B2 + _B3)) | (b2 << _B3) | b3).to(torch.int32)
    v0, v1 = v[:, 0:1], v[:, 1:2]
    nxt0 = torch.where((key <= v0).sum(1, keepdim=True) > k1 + 1, v0,
                       torch.where(key > v0, key, top).amin(1, keepdim=True))
    nxt1 = torch.where((key <= v1).sum(1, keepdim=True) > k2 + 1, v1,
                       torch.where(key > v1, key, top).amin(1, keepdim=True))
    return _keys_to_float(v), _keys_to_float(torch.cat([nxt0, nxt1], 1))


def to_uint8_torch_hist_batch(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """(R, K, H, W) float32 -> uint8: `cache._to_uint8` per stack (bit for bit), `_BATCH` stacks per
    launch.
    """
    if x.dtype != torch.float32:
        raise TypeError(f"expected a float32 stack, got {x.dtype}")
    n = x.shape[-3] * x.shape[-2] * x.shape[-1]
    if n >= 1 << 24:
        return to_uint8_torch(x)
    st = _PCT_STRIDE[0]
    k1, k2, gamma = _ranks(n if st <= 1 else len(range(0, n, st)))
    if x.is_cuda and os.environ.get("KNEE_FUSED_UINT8", "off") == "on":
        from .fused_uint8 import hist_to_uint8
        res = hist_to_uint8(x, k1, k2, gamma, out=out, sample_stride=max(1, st))
        if res is not None:
            return res
    if out is None:
        out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    for s in range(0, x.shape[0], _BATCH):
        xs = x[s:s + _BATCH]
        flat = xs.reshape(xs.shape[0], n)
        sub = flat if st <= 1 else flat[:, ::st].contiguous()
        prev, nxt = _hist_order_stats(sub, k1, k2)
        _normalize_rows(xs, flat, prev, nxt, gamma, sub=sub if st > 1 else None,
                        out_buf=out[s:s + _BATCH])
    return out
