"""Triton kernels for `knee.fastdense`: the `RefOps` primitives fused, for the T4 (sm_75, fp16).

Not used in practice: on sm_75 Triton does not emit mma instructions (see `CublasOps`), so the
fast DenseNet training uses cuBLAS/cuDNN for the matrix products and these kernels only for the
elementwise parts and reductions.

Every matrix is 2D P x C with rows = pixels (n, h, w) and contiguous channels (NHWC), possibly seen
with a row stride (`s*`) wider than the columns read: the block buffer is read in channel slices
without copying it. BatchNorm enters as (mean, rstd, gamma, beta) per column and becomes the affine
a = gamma * rstd, c = beta - mean * a inside the kernel. Column statistics come out as partial sums
per program (`part[pid, col]`), then summed with `torch.sum`: no atomics, deterministic result.

The GEMMs:
- `bnrelu_gemm_fwd`: U = relu(X a + c) W^T (conv1, BN+ReLU prologue), sums of U - shift;
- `trans_fwd`: the transition, 2x2 mean of the four pixels after BN+ReLU, then the GEMM;
- `gemm_bwd_fused`: dY = dU W recomputed per block of rows, G += a gm, BN reductions (sum of gm
  and of gm * xhat) and dW = dU^T relu(X a + c), in row slices (`SPLIT` programs per column). The
  BN corrections, affine in x per channel, are accumulated by `fastdense.EncoderFn` and applied by
  `dcols` when the gradient of a channel is complete: every layer reads its input ONCE in the
  backward pass;
- the same for the transition (`trans_bwd_fused`, plus `trans_wgrad`) and for norm5
  (`gap_bwd_fused`).
The rest is elementwise with per-column reductions (BN2 of the bottleneck, segment copy, final
global mean).

With TRITON_INTERPRET=1 they also run on CPU (slowly): so they are tested locally against `RefOps`.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fastdense import BOTTLENECK, GROWTH, conv3_cudnn_bwd, conv3_cudnn_fwd


@triton.jit
def _affine(mean_ptr, rstd_ptr, g_ptr, b_ptr, cols, cmask):
    mu = tl.load(mean_ptr + cols, mask=cmask, other=0.0)
    rs = tl.load(rstd_ptr + cols, mask=cmask, other=0.0)
    g = tl.load(g_ptr + cols, mask=cmask, other=0.0)
    b = tl.load(b_ptr + cols, mask=cmask, other=0.0)
    a = g * rs
    c = b - mu * a
    return a, c, mu, rs


@triton.jit
def _mm(a, b, acc, IEEE: tl.constexpr):
    """acc + a @ b; IEEE (only in the fp32 tests): a true fp32 product, no tf32."""
    if IEEE:
        return acc + tl.dot(a, b, input_precision="ieee")
    return acc + tl.dot(a, b)


# --------------------------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------------------------

@triton.jit
def _bnrelu_gemm_fwd(X, sx, M, K, mean, rstd, gam, bet, W, U, N, sh, part1, part2,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IEEE: tl.constexpr = False):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        kmask = rk < K
        a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rk, kmask)
        x = tl.load(X + rm[:, None] * sx + rk[None, :], mask=mmask[:, None] & kmask[None, :],
                    other=0.0)
        y = tl.maximum(x.to(tl.float32) * a[None, :] + c[None, :], 0.0).to(W.dtype.element_ty)
        w = tl.load(W + rn[None, :] * K + rk[:, None], mask=kmask[:, None] & nmask[None, :],
                    other=0.0)
        acc = _mm(y, w, acc, IEEE)
    u = acc.to(U.dtype.element_ty)
    full = mmask[:, None] & nmask[None, :]
    tl.store(U + rm[:, None] * N + rn[None, :], u, mask=full)
    shv = tl.load(sh + rn, mask=nmask, other=0.0)
    d = tl.where(full, u.to(tl.float32) - shv[None, :], 0.0)
    tl.store(part1 + pid_m * N + rn, tl.sum(d, axis=0), mask=nmask)
    tl.store(part2 + pid_m * N + rn, tl.sum(d * d, axis=0), mask=nmask)


@triton.jit
def _bnrelu_apply(U, V, M, N, mean, rstd, gam, bet, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rn, nmask)
    full = mmask[:, None] & nmask[None, :]
    u = tl.load(U + rm[:, None] * N + rn[None, :], mask=full, other=0.0)
    v = tl.maximum(u.to(tl.float32) * a[None, :] + c[None, :], 0.0)
    tl.store(V + rm[:, None] * N + rn[None, :], v.to(V.dtype.element_ty), mask=full)


@triton.jit
def _copy_cols_stats(S, ss, X, sx, c0, M, N, sh, part1, part2, COPY: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr):
    """X[:, c0:c0+N] = S (if COPY) and the per-column sums of (S - sh)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    full = mmask[:, None] & nmask[None, :]
    s = tl.load(S + rm[:, None] * ss + rn[None, :], mask=full, other=0.0)
    if COPY:
        tl.store(X + rm[:, None] * sx + c0 + rn[None, :], s, mask=full)
    shv = tl.load(sh + rn, mask=nmask, other=0.0)
    d = tl.where(full, s.to(tl.float32) - shv[None, :], 0.0)
    tl.store(part1 + pid_m * N + rn, tl.sum(d, axis=0), mask=nmask)
    tl.store(part2 + pid_m * N + rn, tl.sum(d * d, axis=0), mask=nmask)


@triton.jit
def _pool_rows(r, H, W):
    """Row of the top-left pixel of the 2x2 square of row `r` of the averaged output."""
    W2 = W // 2
    H2 = H // 2
    w2 = r % W2
    t = r // W2
    h2 = t % H2
    n = t // H2
    return n * H * W + (2 * h2) * W + 2 * w2


@triton.jit
def _trans_fwd(X, sx, H, W, M2, K, mean, rstd, gam, bet, Wt, O, so, N, sh, part1, part2,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IEEE: tl.constexpr = False):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    r = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rmask = r < M2
    nmask = rn < N
    p0 = _pool_rows(r, H, W)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        kmask = rk < K
        a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rk, kmask)
        lm = rmask[:, None] & kmask[None, :]
        y = tl.zeros((BM, BK), dtype=tl.float32)
        for q in tl.static_range(4):
            p = p0 + (q // 2) * W + (q % 2)
            x = tl.load(X + p[:, None] * sx + rk[None, :], mask=lm, other=0.0)
            y += tl.maximum(x.to(tl.float32) * a[None, :] + c[None, :], 0.0)
        y = (y * 0.25).to(Wt.dtype.element_ty)
        w = tl.load(Wt + rn[None, :] * K + rk[:, None], mask=kmask[:, None] & nmask[None, :],
                    other=0.0)
        acc = _mm(y, w, acc, IEEE)
    o = acc.to(O.dtype.element_ty)
    full = rmask[:, None] & nmask[None, :]
    tl.store(O + r[:, None] * so + rn[None, :], o, mask=full)
    shv = tl.load(sh + rn, mask=nmask, other=0.0)
    d = tl.where(full, o.to(tl.float32) - shv[None, :], 0.0)
    tl.store(part1 + pid_m * N + rn, tl.sum(d, axis=0), mask=nmask)
    tl.store(part2 + pid_m * N + rn, tl.sum(d * d, axis=0), mask=nmask)


@triton.jit
def _gap_fwd(X, sx, HW, C, mean, rstd, gam, bet, Fo, BH: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0)
    pid_c = tl.program_id(1)
    rc = pid_c * BC + tl.arange(0, BC)
    cmask = rc < C
    a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rc, cmask)
    acc = tl.zeros((BC,), dtype=tl.float32)
    for h0 in range(0, HW, BH):
        rh = h0 + tl.arange(0, BH)
        full = (rh < HW)[:, None] & cmask[None, :]
        x = tl.load(X + (n * HW + rh)[:, None] * sx + rc[None, :], mask=full, other=0.0)
        y = tl.where(full, tl.maximum(x.to(tl.float32) * a[None, :] + c[None, :], 0.0), 0.0)
        acc += tl.sum(y, axis=0)
    tl.store(Fo + n * C + rc, acc / HW, mask=cmask)


# --------------------------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------------------------

@triton.jit
def _bn_bwd_reduce(dV, U, M, N, mean, rstd, gam, bet, part1, part2,
                   BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    full = mmask[:, None] & nmask[None, :]
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rn, nmask)
    u = tl.load(U + rm[:, None] * N + rn[None, :], mask=full, other=0.0).to(tl.float32)
    dv = tl.load(dV + rm[:, None] * N + rn[None, :], mask=full, other=0.0).to(tl.float32)
    pos = ((u * a[None, :] + c[None, :]) > 0) & full
    gm = tl.where(pos, dv, 0.0)
    uh = (u - mu[None, :]) * rs[None, :]
    tl.store(part1 + pid_m * N + rn, tl.sum(gm, axis=0), mask=nmask)
    tl.store(part2 + pid_m * N + rn, tl.sum(gm * uh, axis=0), mask=nmask)


@triton.jit
def _bn_bwd_apply(dV, U, dU, M, N, mean, rstd, gam, bet, R1, R2,
                  BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    full = mmask[:, None] & nmask[None, :]
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rn, nmask)
    r1 = tl.load(R1 + rn, mask=nmask, other=0.0) / M
    r2 = tl.load(R2 + rn, mask=nmask, other=0.0) / M
    u = tl.load(U + rm[:, None] * N + rn[None, :], mask=full, other=0.0).to(tl.float32)
    dv = tl.load(dV + rm[:, None] * N + rn[None, :], mask=full, other=0.0).to(tl.float32)
    pos = (u * a[None, :] + c[None, :]) > 0
    gm = tl.where(pos, dv, 0.0)
    uh = (u - mu[None, :]) * rs[None, :]
    du = a[None, :] * (gm - r1[None, :] - uh * r2[None, :])
    tl.store(dU + rm[:, None] * N + rn[None, :], du.to(dU.dtype.element_ty), mask=full)


@triton.jit
def _gemm_bwd_fused(dU, M, X, sx, K, mean, rstd, gam, bet, W, G, sg, partR1, partR2, partW, SPLIT,
                    NO: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr, IEEE: tl.constexpr = False):
    """Program (column slice, row slice; one row every SPLIT blocks): dY = dU W recomputed, G += a gm,
    BN reductions (sum of gm and of gm xhat) and dW = dU^T relu(X a + c). The BN corrections (affine in
    x) are applied afterwards by the orchestrator (`fastdense.split_coeffs`).
    """
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    rc = pid_c * BC + tl.arange(0, BC)
    cmask = rc < K
    ro = tl.arange(0, NO)
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rc, cmask)
    w = tl.load(W + ro[:, None] * K + rc[None, :], mask=cmask[None, :], other=0.0)   # (NO, BC)
    r1 = tl.zeros((BC,), dtype=tl.float32)
    r2 = tl.zeros((BC,), dtype=tl.float32)
    dw = tl.zeros((NO, BC), dtype=tl.float32)
    for m0 in range(pid_s * BM, M, SPLIT * BM):
        rm = m0 + tl.arange(0, BM)
        mmask = rm < M
        full = mmask[:, None] & cmask[None, :]
        du = tl.load(dU + rm[:, None] * NO + ro[None, :], mask=mmask[:, None], other=0.0)
        x = tl.load(X + rm[:, None] * sx + rc[None, :], mask=full, other=0.0).to(tl.float32)
        dy = _mm(du, w, tl.zeros((BM, BC), dtype=tl.float32), IEEE)
        z = x * a[None, :] + c[None, :]
        pos = (z > 0) & full
        gm = tl.where(pos, dy, 0.0)
        xh = (x - mu[None, :]) * rs[None, :]
        r1 += tl.sum(gm, axis=0)
        r2 += tl.sum(gm * xh, axis=0)
        gp = G + rm[:, None] * sg + rc[None, :]
        g = tl.load(gp, mask=full, other=0.0).to(tl.float32)
        tl.store(gp, (g + a[None, :] * gm).to(G.dtype.element_ty), mask=full)
        y = tl.where(pos, z, 0.0).to(dU.dtype.element_ty)
        dw = _mm(tl.trans(du), y, dw, IEEE)
    tl.store(partR1 + pid_s * K + rc, r1, mask=cmask)
    tl.store(partR2 + pid_s * K + rc, r2, mask=cmask)
    tl.store(partW + pid_s * NO * K + ro[:, None] * K + rc[None, :], dw, mask=cmask[None, :])


@triton.jit
def _dcols(G, sg, X, sx, c0, M, N, A, B, O, BM: tl.constexpr, BN: tl.constexpr):
    """O (M x N, contiguous) = G[:, c0:c0+N] + A + B * X[:, c0:c0+N]: the complete gradient."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    nmask = rn < N
    full = (rm < M)[:, None] & nmask[None, :]
    g = tl.load(G + rm[:, None] * sg + c0 + rn[None, :], mask=full, other=0.0).to(tl.float32)
    x = tl.load(X + rm[:, None] * sx + c0 + rn[None, :], mask=full, other=0.0).to(tl.float32)
    av = tl.load(A + rn, mask=nmask, other=0.0)
    bv = tl.load(B + rn, mask=nmask, other=0.0)
    o = g + av[None, :] + bv[None, :] * x
    tl.store(O + rm[:, None] * N + rn[None, :], o.to(O.dtype.element_ty), mask=full)


@triton.jit
def _trans_dyp(dT, st, r, rmask, Wt, NO, K, rc, cmask, BM: tl.constexpr, BC: tl.constexpr,
               BO: tl.constexpr, IEEE: tl.constexpr = False):
    """Averaged dY = dT W (rows of the averaged output `r`, columns `rc`), product K = NO."""
    dyp = tl.zeros((BM, BC), dtype=tl.float32)
    for o0 in range(0, NO, BO):
        ro = o0 + tl.arange(0, BO)
        omask = ro < NO
        dt = tl.load(dT + r[:, None] * st + ro[None, :], mask=rmask[:, None] & omask[None, :],
                     other=0.0)
        w = tl.load(Wt + ro[:, None] * K + rc[None, :], mask=omask[:, None] & cmask[None, :],
                    other=0.0)
        dyp = _mm(dt, w, dyp, IEEE)
    return dyp


@triton.jit
def _trans_bwd_fused(dT, st, M2, H, W, X, sx, K, mean, rstd, gam, bet, Wt, NO, G, sg, partR1,
                     partR2, SPLIT, BM: tl.constexpr, BC: tl.constexpr, BO: tl.constexpr,
                     IEEE: tl.constexpr = False):
    """The transition backward: G (of the four pixels of every averaged row) = a gm, and the BN
    reductions. G is new (only the transition has written it so far).
    """
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    rc = pid_c * BC + tl.arange(0, BC)
    cmask = rc < K
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rc, cmask)
    r1 = tl.zeros((BC,), dtype=tl.float32)
    r2 = tl.zeros((BC,), dtype=tl.float32)
    for m0 in range(pid_s * BM, M2, SPLIT * BM):
        r = m0 + tl.arange(0, BM)
        rmask = r < M2
        full = rmask[:, None] & cmask[None, :]
        dyp = _trans_dyp(dT, st, r, rmask, Wt, NO, K, rc, cmask, BM, BC, BO, IEEE) * 0.25
        p0 = _pool_rows(r, H, W)
        for q in tl.static_range(4):
            p = p0 + (q // 2) * W + (q % 2)
            x = tl.load(X + p[:, None] * sx + rc[None, :], mask=full, other=0.0).to(tl.float32)
            pos = ((x * a[None, :] + c[None, :]) > 0) & full
            gm = tl.where(pos, dyp, 0.0)
            xh = (x - mu[None, :]) * rs[None, :]
            r1 += tl.sum(gm, axis=0)
            r2 += tl.sum(gm * xh, axis=0)
            tl.store(G + p[:, None] * sg + rc[None, :], (a[None, :] * gm).to(G.dtype.element_ty),
                     mask=full)
    tl.store(partR1 + pid_s * K + rc, r1, mask=cmask)
    tl.store(partR2 + pid_s * K + rc, r2, mask=cmask)


@triton.jit
def _trans_wgrad(dT, st, M2, H, W, X, sx, K, mean, rstd, gam, bet, NO, partW, SPLIT,
                 BM: tl.constexpr, BC: tl.constexpr, BO: tl.constexpr, IEEE: tl.constexpr = False):
    """dW[o, c] = sum over averaged rows of dT[r, o] * mean2x2(relu(X a + c))[r, c]."""
    pid_o = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)
    ro = pid_o * BO + tl.arange(0, BO)
    rc = pid_c * BC + tl.arange(0, BC)
    omask = ro < NO
    cmask = rc < K
    a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rc, cmask)
    dw = tl.zeros((BO, BC), dtype=tl.float32)
    for m0 in range(pid_s * BM, M2, SPLIT * BM):
        r = m0 + tl.arange(0, BM)
        rmask = r < M2
        full = rmask[:, None] & cmask[None, :]
        p0 = _pool_rows(r, H, W)
        y = tl.zeros((BM, BC), dtype=tl.float32)
        for q in tl.static_range(4):
            p = p0 + (q // 2) * W + (q % 2)
            x = tl.load(X + p[:, None] * sx + rc[None, :], mask=full, other=0.0).to(tl.float32)
            y += tl.maximum(x * a[None, :] + c[None, :], 0.0)
        y = tl.where(full, y * 0.25, 0.0).to(dT.dtype.element_ty)
        dt = tl.load(dT + r[:, None] * st + ro[None, :], mask=rmask[:, None] & omask[None, :],
                     other=0.0)
        dw = _mm(tl.trans(dt), y, dw, IEEE)
    tl.store(partW + pid_s * NO * K + ro[:, None] * K + rc[None, :], dw,
             mask=omask[:, None] & cmask[None, :])


@triton.jit
def _gap_bwd_fused(dF, HW, X, sx, C, mean, rstd, gam, bet, G, sg, part1, part2,
                   BH: tl.constexpr, BC: tl.constexpr):
    """norm5 + ReLU + global mean backward: G = a gm and the reductions (per image)."""
    n = tl.program_id(0)
    pid_c = tl.program_id(1)
    rc = pid_c * BC + tl.arange(0, BC)
    cmask = rc < C
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rc, cmask)
    dy = tl.load(dF + n * C + rc, mask=cmask, other=0.0).to(tl.float32) / HW
    r1 = tl.zeros((BC,), dtype=tl.float32)
    r2 = tl.zeros((BC,), dtype=tl.float32)
    for h0 in range(0, HW, BH):
        rh = h0 + tl.arange(0, BH)
        full = (rh < HW)[:, None] & cmask[None, :]
        rows = (n * HW + rh)[:, None]
        x = tl.load(X + rows * sx + rc[None, :], mask=full, other=0.0).to(tl.float32)
        pos = ((x * a[None, :] + c[None, :]) > 0) & full
        gm = tl.where(pos, dy[None, :], 0.0)
        xh = (x - mu[None, :]) * rs[None, :]
        r1 += tl.sum(gm, axis=0)
        r2 += tl.sum(gm * xh, axis=0)
        tl.store(G + rows * sg + rc[None, :], (a[None, :] * gm).to(G.dtype.element_ty), mask=full)
    tl.store(part1 + n * C + rc, r1, mask=cmask)
    tl.store(part2 + n * C + rc, r2, mask=cmask)


@triton.jit
def _conv3_fwd(U, M, H, W, mean, rstd, gam, bet, Wr, X, sx, c0, sh, part1, part2,
               CI: tl.constexpr, CO: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
               IEEE: tl.constexpr = False, PRO: tl.constexpr = True):
    """The 3x3 bottleneck conv (stride 1, padding 1) as an implicit GEMM on the NHWC rows: for each of the
    9 taps the row slice shifted by (dy W + dx), BN2 + ReLU in the prologue (zero outside the image: the
    padding comes AFTER the ReLU), weights Wr[t, ci, co]. The output goes straight into the block
    buffer (columns c0..c0+CO) with the sums for the statistics.
    """
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    mmask = rm < M
    ww = rm % W
    hh = (rm // W) % H
    rco = tl.arange(0, CO)
    acc = tl.zeros((BM, CO), dtype=tl.float32)
    for t in tl.static_range(9):
        dy = t // 3 - 1
        dx = t % 3 - 1
        valid = mmask & (hh + dy >= 0) & (hh + dy < H) & (ww + dx >= 0) & (ww + dx < W)
        src = rm + dy * W + dx
        for k0 in range(0, CI, BK):
            rk = k0 + tl.arange(0, BK)
            u = tl.load(U + src[:, None] * CI + rk[None, :], mask=valid[:, None], other=0.0)
            if PRO:
                a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rk, rk < CI)
                v = tl.maximum(u.to(tl.float32) * a[None, :] + c[None, :], 0.0)
                v = tl.where(valid[:, None], v, 0.0).to(Wr.dtype.element_ty)
            else:
                v = u.to(Wr.dtype.element_ty)          # V already materialised, zero outside
            wt = tl.load(Wr + t * CI * CO + rk[:, None] * CO + rco[None, :])
            acc = _mm(v, wt, acc, IEEE)
    s = acc.to(X.dtype.element_ty)
    tl.store(X + rm[:, None] * sx + c0 + rco[None, :], s, mask=mmask[:, None])
    shv = tl.load(sh + rco)
    d = tl.where(mmask[:, None], s.to(tl.float32) - shv[None, :], 0.0)
    tl.store(part1 + pid * CO + rco, tl.sum(d, axis=0))
    tl.store(part2 + pid * CO + rco, tl.sum(d * d, axis=0))


@triton.jit
def _conv3_bwd(dS, M, H, W, U, mean, rstd, gam, bet, Wr, R1, R2, dU, part1, part2,
               APPLY: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr, BM: tl.constexpr,
               IEEE: tl.constexpr = False):
    """dV = conv3x3^T(dS) for a block of rows (input pixels), then BN2 + ReLU backward. APPLY False: the
    reductions (sum of gm and of gm uhat) per program; APPLY True: complete dU with the reductions
    already summed (R1, R2). dV is recomputed in both passes (K = 9 x 32).
    """
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    mmask = rm < M
    ww = rm % W
    hh = (rm // W) % H
    rci = tl.arange(0, CI)
    rco = tl.arange(0, CO)
    acc = tl.zeros((BM, CI), dtype=tl.float32)
    for t in tl.static_range(9):
        dy = t // 3 - 1
        dx = t % 3 - 1
        valid = mmask & (hh - dy >= 0) & (hh - dy < H) & (ww - dx >= 0) & (ww - dx < W)
        q = rm - dy * W - dx
        ds = tl.load(dS + q[:, None] * CO + rco[None, :], mask=valid[:, None], other=0.0)
        wt = tl.load(Wr + t * CI * CO + rci[None, :] * CO + rco[:, None])          # (CO, CI)
        acc = _mm(ds, wt, acc, IEEE)
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rci, rci < CI)
    u = tl.load(U + rm[:, None] * CI + rci[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)
    pos = ((u * a[None, :] + c[None, :]) > 0) & mmask[:, None]
    gm = tl.where(pos, acc, 0.0)
    uh = (u - mu[None, :]) * rs[None, :]
    if APPLY:
        r1 = tl.load(R1 + rci) / M
        r2 = tl.load(R2 + rci) / M
        du = a[None, :] * (gm - r1[None, :] - uh * r2[None, :])
        tl.store(dU + rm[:, None] * CI + rci[None, :], du.to(dU.dtype.element_ty), mask=mmask[:, None])
    else:
        tl.store(part1 + pid * CI + rci, tl.sum(gm, axis=0))
        tl.store(part2 + pid * CI + rci, tl.sum(gm * uh, axis=0))


@triton.jit
def _conv3_wgrad(dS, M, H, W, U, mean, rstd, gam, bet, partW, SPLIT,
                 CI: tl.constexpr, CO: tl.constexpr, BM: tl.constexpr, IEEE: tl.constexpr = False,
                 PRO: tl.constexpr = True):
    """dW of tap t (grid axis 0) = sum over output pixels q of V[q + s_t]^T dS[q], with V =
    relu(bn2(U)) recomputed and zero outside the image. Partials per row slice.
    """
    t = tl.program_id(0)
    pid_s = tl.program_id(1)
    dy = t // 3 - 1
    dx = t % 3 - 1
    rci = tl.arange(0, CI)
    rco = tl.arange(0, CO)
    a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rci, rci < CI)
    acc = tl.zeros((CI, CO), dtype=tl.float32)
    for m0 in range(pid_s * BM, M, SPLIT * BM):
        rq = m0 + tl.arange(0, BM)
        qmask = rq < M
        ww = rq % W
        hh = (rq // W) % H
        valid = qmask & (hh + dy >= 0) & (hh + dy < H) & (ww + dx >= 0) & (ww + dx < W)
        src = rq + dy * W + dx
        u = tl.load(U + src[:, None] * CI + rci[None, :], mask=valid[:, None], other=0.0)
        if PRO:
            v = tl.maximum(u.to(tl.float32) * a[None, :] + c[None, :], 0.0)
            v = tl.where(valid[:, None], v, 0.0).to(dS.dtype.element_ty)
        else:
            v = u.to(dS.dtype.element_ty)              # V already materialised
        ds = tl.load(dS + rq[:, None] * CO + rco[None, :], mask=qmask[:, None], other=0.0)
        acc = _mm(tl.trans(v), ds, acc, IEEE)
    tl.store(partW + (pid_s * 9 + t) * CI * CO + rci[:, None] * CO + rco[None, :], acc)


# --------------------------------------------------------------------------------------------
# without dot: the elementwise and reduction parts of `CublasOps` (the GEMMs go to cuBLAS)
# --------------------------------------------------------------------------------------------

@triton.jit
def _bnrelu_rows(X, sx, V, sv, M, N, mean, rstd, gam, bet, BM: tl.constexpr, BN: tl.constexpr):
    """V[:, :N] = relu(X[:, :N] a + c), rows with stride sx (input) and sv (output)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    nmask = rn < N
    full = (rm < M)[:, None] & nmask[None, :]
    a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rn, nmask)
    x = tl.load(X + rm[:, None] * sx + rn[None, :], mask=full, other=0.0)
    v = tl.maximum(x.to(tl.float32) * a[None, :] + c[None, :], 0.0)
    tl.store(V + rm[:, None] * sv + rn[None, :], v.to(V.dtype.element_ty), mask=full)


@triton.jit
def _colstats_split(S, ss, X, sx, c0, M, N, sh, part1, part2, SPLIT, COPY: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr):
    """Per-column sums of (S - sh), partial per row slice (one every SPLIT blocks), and if COPY
    X[:, c0:c0+N] = S.
    """
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    nmask = rn < N
    shv = tl.load(sh + rn, mask=nmask, other=0.0)
    t1 = tl.zeros((BN,), dtype=tl.float32)
    t2 = tl.zeros((BN,), dtype=tl.float32)
    for m0 in range(pid_s * BM, M, SPLIT * BM):
        rm = m0 + tl.arange(0, BM)
        full = (rm < M)[:, None] & nmask[None, :]
        s = tl.load(S + rm[:, None] * ss + rn[None, :], mask=full, other=0.0)
        if COPY:
            tl.store(X + rm[:, None] * sx + c0 + rn[None, :], s, mask=full)
        d = tl.where(full, s.to(tl.float32) - shv[None, :], 0.0)
        t1 += tl.sum(d, axis=0)
        t2 += tl.sum(d * d, axis=0)
    tl.store(part1 + pid_s * N + rn, t1, mask=nmask)
    tl.store(part2 + pid_s * N + rn, t2, mask=nmask)


@triton.jit
def _bn_bwd_reduce_split(dV, U, M, N, mean, rstd, gam, bet, part1, part2, SPLIT,
                         BM: tl.constexpr, BN: tl.constexpr):
    """The BN + ReLU backward reductions (sum of gm and of gm uhat), in row slices."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    nmask = rn < N
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rn, nmask)
    t1 = tl.zeros((BN,), dtype=tl.float32)
    t2 = tl.zeros((BN,), dtype=tl.float32)
    for m0 in range(pid_s * BM, M, SPLIT * BM):
        rm = m0 + tl.arange(0, BM)
        full = (rm < M)[:, None] & nmask[None, :]
        u = tl.load(U + rm[:, None] * N + rn[None, :], mask=full, other=0.0).to(tl.float32)
        dv = tl.load(dV + rm[:, None] * N + rn[None, :], mask=full, other=0.0).to(tl.float32)
        pos = ((u * a[None, :] + c[None, :]) > 0) & full
        gm = tl.where(pos, dv, 0.0)
        t1 += tl.sum(gm, axis=0)
        t2 += tl.sum(gm * ((u - mu[None, :]) * rs[None, :]), axis=0)
    tl.store(part1 + pid_s * N + rn, t1, mask=nmask)
    tl.store(part2 + pid_s * N + rn, t2, mask=nmask)


@triton.jit
def _trans_pool(X, sx, H, W, M2, K, mean, rstd, gam, bet, Y, BM: tl.constexpr, BK: tl.constexpr):
    """Y (M2 x K, contiguous) = 2x2 mean of relu(X a + c): the input of the transition GEMM."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    r = pid_m * BM + tl.arange(0, BM)
    rk = pid_k * BK + tl.arange(0, BK)
    kmask = rk < K
    lm = (r < M2)[:, None] & kmask[None, :]
    a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rk, kmask)
    p0 = _pool_rows(r, H, W)
    y = tl.zeros((BM, BK), dtype=tl.float32)
    for q in tl.static_range(4):
        p = p0 + (q // 2) * W + (q % 2)
        x = tl.load(X + p[:, None] * sx + rk[None, :], mask=lm, other=0.0)
        y += tl.maximum(x.to(tl.float32) * a[None, :] + c[None, :], 0.0)
    tl.store(Y + r[:, None] * K + rk[None, :], (y * 0.25).to(Y.dtype.element_ty), mask=lm)


@triton.jit
def _trans_bwd_ew(dYp, M2, H, W, X, sx, K, mean, rstd, gam, bet, G, sg, part1, part2, SPLIT,
                  BM: tl.constexpr, BC: tl.constexpr):
    """The transition backward after dYp = dT W (cuBLAS, M2 x K contiguous): G of the four pixels of
    every averaged row = a gm, gm = dYp/4 [x a + c > 0], and the BN reductions.
    """
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    rc = pid_c * BC + tl.arange(0, BC)
    cmask = rc < K
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rc, cmask)
    r1 = tl.zeros((BC,), dtype=tl.float32)
    r2 = tl.zeros((BC,), dtype=tl.float32)
    for m0 in range(pid_s * BM, M2, SPLIT * BM):
        r = m0 + tl.arange(0, BM)
        full = (r < M2)[:, None] & cmask[None, :]
        dyp = tl.load(dYp + r[:, None] * K + rc[None, :], mask=full, other=0.0).to(tl.float32) * 0.25
        p0 = _pool_rows(r, H, W)
        for q in tl.static_range(4):
            p = p0 + (q // 2) * W + (q % 2)
            x = tl.load(X + p[:, None] * sx + rc[None, :], mask=full, other=0.0).to(tl.float32)
            pos = ((x * a[None, :] + c[None, :]) > 0) & full
            gm = tl.where(pos, dyp, 0.0)
            r1 += tl.sum(gm, axis=0)
            r2 += tl.sum(gm * ((x - mu[None, :]) * rs[None, :]), axis=0)
            tl.store(G + p[:, None] * sg + rc[None, :], (a[None, :] * gm).to(G.dtype.element_ty),
                     mask=full)
    tl.store(part1 + pid_s * K + rc, r1, mask=cmask)
    tl.store(part2 + pid_s * K + rc, r2, mask=cmask)


@triton.jit
def _gemm_bwd_ew(dY, M, X, sx, K, mean, rstd, gam, bet, G, sg, V, part1, part2, SPLIT,
                 BM: tl.constexpr, BC: tl.constexpr):
    """conv1 backward after dY = dU W (cuBLAS, M x K contiguous): gm = dY [x a + c > 0], G += a gm, the
    BN reductions (sum of gm and of gm xhat) and V = relu(x a + c) (M x K contiguous) for dW = dU^T V
    (cuBLAS): the input is read once.
    """
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    rc = pid_c * BC + tl.arange(0, BC)
    cmask = rc < K
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rc, cmask)
    r1 = tl.zeros((BC,), dtype=tl.float32)
    r2 = tl.zeros((BC,), dtype=tl.float32)
    for m0 in range(pid_s * BM, M, SPLIT * BM):
        rm = m0 + tl.arange(0, BM)
        full = (rm < M)[:, None] & cmask[None, :]
        x = tl.load(X + rm[:, None] * sx + rc[None, :], mask=full, other=0.0).to(tl.float32)
        dy = tl.load(dY + rm[:, None] * K + rc[None, :], mask=full, other=0.0).to(tl.float32)
        z = x * a[None, :] + c[None, :]
        pos = (z > 0) & full
        gm = tl.where(pos, dy, 0.0)
        r1 += tl.sum(gm, axis=0)
        r2 += tl.sum(gm * ((x - mu[None, :]) * rs[None, :]), axis=0)
        gp = G + rm[:, None] * sg + rc[None, :]
        g = tl.load(gp, mask=full, other=0.0).to(tl.float32)
        tl.store(gp, (g + a[None, :] * gm).to(G.dtype.element_ty), mask=full)
        tl.store(V + rm[:, None] * K + rc[None, :], tl.where(pos, z, 0.0).to(V.dtype.element_ty),
                 mask=full)
    tl.store(part1 + pid_s * K + rc, r1, mask=cmask)
    tl.store(part2 + pid_s * K + rc, r2, mask=cmask)


# --------------------------------------------------------------------------------------------
# per-channel BatchNorm statistics (vectors): one launch instead of ~14 torch operations
# --------------------------------------------------------------------------------------------

@triton.jit
def _bn_stats_k(S1, S2, SH, RM, RV, NBT, MEAN, RSTD, C, inv_n, n_corr, eps, mom,
                UPD_NBT: tl.constexpr, BLOCK: tl.constexpr):
    """`finalize_stats` + `update_running`: from the sums (shifted by SH) batch mean and rstd, and the
    running averages like `F.batch_norm` (variance corrected by n/(n-1), `num_batches_tracked`).
    """
    pid = tl.program_id(0)
    c = pid * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    s1 = tl.load(S1 + c, mask=m, other=0.0)
    s2 = tl.load(S2 + c, mask=m, other=0.0)
    sh = tl.load(SH + c, mask=m, other=0.0)
    m1 = s1 * inv_n
    var = tl.maximum(s2 * inv_n - m1 * m1, 0.0)
    mu = sh + m1
    rs = 1.0 / tl.sqrt(var + eps)
    tl.store(MEAN + c, mu, mask=m)
    tl.store(RSTD + c, rs, mask=m)
    rm = tl.load(RM + c, mask=m, other=0.0)
    rv = tl.load(RV + c, mask=m, other=0.0)
    tl.store(RM + c, rm * (1.0 - mom) + mu * mom, mask=m)
    tl.store(RV + c, rv * (1.0 - mom) + var * n_corr * mom, mask=m)
    if UPD_NBT:
        if pid == 0:
            nb = tl.load(NBT)
            tl.store(NBT, nb + 1)


@triton.jit
def _bn_coeffs_k(MEAN, RSTD, G, DG, DB, A, B, C, inv_p, BLOCK: tl.constexpr):
    """A += alpha, B += beta of the BN backward correction (`fastdense.split_coeffs`)."""
    pid = tl.program_id(0)
    c = pid * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    mu = tl.load(MEAN + c, mask=m, other=0.0)
    rs = tl.load(RSTD + c, mask=m, other=0.0)
    g = tl.load(G + c, mask=m, other=0.0).to(tl.float32)
    dg = tl.load(DG + c, mask=m, other=0.0)
    db = tl.load(DB + c, mask=m, other=0.0)
    a = g * rs
    beta = -a * rs * dg * inv_p
    alpha = -a * db * inv_p - beta * mu
    tl.store(A + c, tl.load(A + c, mask=m, other=0.0) + alpha, mask=m)
    tl.store(B + c, tl.load(B + c, mask=m, other=0.0) + beta, mask=m)


# --------------------------------------------------------------------------------------------
# the stem: norm0 (BN + ReLU) and 3x3 / 2 max-pool (padding 1) fused, without int64 indices
# --------------------------------------------------------------------------------------------

@triton.jit
def _stem_pool_fwd(Y, H, W, Ho, Wo, M2, mean, rstd, gam, bet, S, AM, YM, C: tl.constexpr,
                   BM: tl.constexpr, WITH_YM: tl.constexpr):
    """S[r] = max over the 9 taps of the window of relu(Y a + c) (= relu of the max of Y a + c), AM[r] =
    the tap of the maximum (0..8, the first in scan order), one byte per channel; with WITH_YM also
    YM[r] = Y at the tap of the maximum (the BN backward reductions of `CudaOps` are done on the
    windows).
    """
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    rc = tl.arange(0, C)
    rmask = r < M2
    wo = r % Wo
    t = r // Wo
    ho = t % Ho
    n = t // Ho
    a, c, _mu, _rs = _affine(mean, rstd, gam, bet, rc, rc < C)
    best = tl.full((BM, C), float("-inf"), dtype=tl.float32)
    arg = tl.zeros((BM, C), dtype=tl.int32)
    ybest = tl.zeros((BM, C), dtype=tl.float32)
    for k in tl.static_range(9):
        h = 2 * ho - 1 + k // 3
        w = 2 * wo - 1 + k % 3
        valid = rmask & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        pix = (n * H + h) * W + w
        y = tl.load(Y + pix[:, None] * C + rc[None, :], mask=valid[:, None], other=0.0)
        z = tl.where(valid[:, None], y.to(tl.float32) * a[None, :] + c[None, :], float("-inf"))
        up = z > best
        best = tl.where(up, z, best)
        arg = tl.where(up, k, arg)
        if WITH_YM:
            ybest = tl.where(up, y.to(tl.float32), ybest)
    out = tl.maximum(best, 0.0)
    tl.store(S + r[:, None] * C + rc[None, :], out.to(S.dtype.element_ty), mask=rmask[:, None])
    tl.store(AM + r[:, None] * C + rc[None, :], arg.to(tl.uint8), mask=rmask[:, None])
    if WITH_YM:
        tl.store(YM + r[:, None] * C + rc[None, :], ybest.to(YM.dtype.element_ty),
                 mask=rmask[:, None])


@triton.jit
def _stem_pool_bwd(Y, H, W, Ho, Wo, M, mean, rstd, gam, bet, dS, S, AM, dY, k1, k2, part1, part2,
                   SPLIT, APPLY: tl.constexpr, C: tl.constexpr, BM: tl.constexpr):
    """For every input pixel: gm = sum of dS over the windows (1, 2 or 4) of which it is the maximum with
    output > 0 (the ReLU), gathered (no atomics). APPLY False: the BN reductions (sum of gm and of gm
    xhat) in row slices; APPLY True: dY = a (gm - k1 - xhat k2).
    """
    pid_s = tl.program_id(0)
    rc = tl.arange(0, C)
    a, c, mu, rs = _affine(mean, rstd, gam, bet, rc, rc < C)
    t1 = tl.zeros((C,), dtype=tl.float32)
    t2 = tl.zeros((C,), dtype=tl.float32)
    kk1 = tl.load(k1 + rc)
    kk2 = tl.load(k2 + rc)
    for m0 in range(pid_s * BM, M, SPLIT * BM):
        pix = m0 + tl.arange(0, BM)
        pmask = pix < M
        w = pix % W
        t = pix // W
        h = t % H
        n = t // H
        gm = tl.zeros((BM, C), dtype=tl.float32)
        for i in tl.static_range(2):
            ho = (h + 1) // 2 - i
            dy = h - (2 * ho - 1)
            vh = (ho >= 0) & (ho < Ho) & (dy >= 0) & (dy <= 2)
            for j in tl.static_range(2):
                wo = (w + 1) // 2 - j
                dx = w - (2 * wo - 1)
                v = pmask & vh & (wo >= 0) & (wo < Wo) & (dx >= 0) & (dx <= 2)
                q = (n * Ho + ho) * Wo + wo
                off = q[:, None] * C + rc[None, :]
                am = tl.load(AM + off, mask=v[:, None], other=255).to(tl.int32)
                sv = tl.load(S + off, mask=v[:, None], other=0.0).to(tl.float32)
                ds = tl.load(dS + off, mask=v[:, None], other=0.0).to(tl.float32)
                hit = (am == (dy * 3 + dx)[:, None]) & (sv > 0.0) & v[:, None]
                gm += tl.where(hit, ds, 0.0)
        y = tl.load(Y + pix[:, None] * C + rc[None, :], mask=pmask[:, None], other=0.0).to(tl.float32)
        xh = (y - mu[None, :]) * rs[None, :]
        if APPLY:
            d = a[None, :] * (gm - kk1[None, :] - xh * kk2[None, :])
            tl.store(dY + pix[:, None] * C + rc[None, :], d.to(dY.dtype.element_ty),
                     mask=pmask[:, None])
        else:
            t1 += tl.sum(gm, axis=0)
            t2 += tl.sum(gm * xh, axis=0)
    if not APPLY:
        tl.store(part1 + pid_s * C + rc, t1)
        tl.store(part2 + pid_s * C + rc, t2)


# --------------------------------------------------------------------------------------------
# the primitives
# --------------------------------------------------------------------------------------------

def _f32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float32 and t.is_contiguous() else t.float().contiguous()


class TritonOps:
    """The `RefOps` primitives with the kernels above. fp16 activations, fp32 statistics."""

    name = "triton"

    def __init__(self, bm: int = 64, bk: int = 32, bm_bwd: int = 64, bc_bwd: int = 64,
                 split: int = 0, bm_ew: int = 64, ieee: bool = False, warps_bwd: int = 8,
                 conv: str = "triton", bm_conv: int = 64, bm_conv_bwd: int = 32,
                 warps_fwd: int = 4, warps_conv: int = 4):
        self.bm, self.bk, self.bm_bwd, self.bc_bwd, self.split = bm, bk, bm_bwd, bc_bwd, split
        self.bm_ew = bm_ew
        if conv not in ("triton", "triton_v", "cudnn"):
            raise ValueError(f"TritonOps: conv {conv!r}, expected 'triton', 'triton_v' or 'cudnn'")
        # the 3x3 bottleneck conv: in Triton with BN2 + ReLU in the prologue and no V ('triton': the
        # prologue is redone for each of the 9 taps), in Triton on V materialised once ('triton_v': V
        # written and re-read, no repeated prologues), or cuDNN on V ('cudnn')
        self.conv = conv
        self.bm_conv, self.bm_conv_bwd = bm_conv, bm_conv_bwd
        self.name = {"triton": "triton", "triton_v": "triton-v", "cudnn": "triton+cudnn"}[conv]
        self.warps_fwd, self.warps_conv = int(warps_fwd), int(warps_conv)

        # `ieee`: true fp32 dots (only to test the kernels in fp32 against the oracle)
        self.ieee = bool(ieee)
        self.warps_bwd = int(warps_bwd)

    def _split(self, M: int, BM: int, ncol: int) -> int:
        if self.split:
            return self.split
        tiles = triton.cdiv(M, BM)
        # enough programs to fill 40 SMs at least 4 times, no more than the row blocks
        want = max(1, (160 + ncol - 1) // ncol)
        return int(max(1, min(tiles, want)))

    def colstats(self, X, c0: int, c1: int, shift):
        M, N = X.shape[0], c1 - c0
        BM, BN = self.bm_ew, triton.next_power_of_2(min(N, 64))
        gm, gn = triton.cdiv(M, BM), triton.cdiv(N, BN)
        p1 = torch.empty(gm, N, dtype=torch.float32, device=X.device)
        p2 = torch.empty_like(p1)
        Xs = X[:, c0:]
        _copy_cols_stats[(gm, gn)](Xs, X.stride(0), Xs, X.stride(0), 0, M, N, _f32(shift), p1, p2,
                                   COPY=False, BM=BM, BN=BN)
        return p1.sum(0), p2.sum(0)

    def bnrelu_gemm(self, X, cin: int, mean, rstd, g, b, W, ushift):
        M, N = X.shape[0], W.shape[0]
        W = W.contiguous()
        BM, BK = self.bm, self.bk
        BN = min(128, triton.next_power_of_2(N))
        gm, gn = triton.cdiv(M, BM), triton.cdiv(N, BN)
        U = torch.empty(M, N, dtype=X.dtype, device=X.device)
        p1 = torch.empty(gm, N, dtype=torch.float32, device=X.device)
        p2 = torch.empty_like(p1)
        _bnrelu_gemm_fwd[(gm, gn)](X, X.stride(0), M, cin, _f32(mean), _f32(rstd), _f32(g),
                                   _f32(b), W, U, N, _f32(ushift), p1, p2, BM=BM, BN=BN, BK=BK, IEEE=self.ieee,
                                   num_warps=self.warps_fwd)
        return U, p1.sum(0), p2.sum(0)

    def bnrelu(self, U, mean, rstd, g, b):
        M, N = U.shape
        BM, BN = self.bm_ew, min(128, triton.next_power_of_2(N))
        V = torch.empty_like(U)
        _bnrelu_apply[(triton.cdiv(M, BM), triton.cdiv(N, BN))](U, V, M, N, _f32(mean), _f32(rstd),
                                                                _f32(g), _f32(b), BM=BM, BN=BN)
        return V

    def write_segment(self, X, c0: int, S, shift):
        M, N = S.shape
        BM, BN = self.bm_ew, triton.next_power_of_2(N)
        gm = triton.cdiv(M, BM)
        p1 = torch.empty(gm, N, dtype=torch.float32, device=X.device)
        p2 = torch.empty_like(p1)
        _copy_cols_stats[(gm, 1)](S, S.stride(0), X, X.stride(0), c0, M, N, _f32(shift), p1, p2,
                                  COPY=True, BM=BM, BN=BN)
        return p1.sum(0), p2.sum(0)

    def trans_fwd(self, X, C: int, mean, rstd, g, b, W, n: int, h: int, w: int, out, oshift):
        M2, N = n * (h // 2) * (w // 2), W.shape[0]
        W = W.contiguous()
        BM, BK = self.bm, self.bk
        BN = min(128, triton.next_power_of_2(N))
        gm, gn = triton.cdiv(M2, BM), triton.cdiv(N, BN)
        p1 = torch.empty(gm, N, dtype=torch.float32, device=X.device)
        p2 = torch.empty_like(p1)
        _trans_fwd[(gm, gn)](X, X.stride(0), h, w, M2, C, _f32(mean), _f32(rstd), _f32(g), _f32(b),
                             W, out, out.stride(0), N, _f32(oshift), p1, p2, BM=BM, BN=BN, BK=BK, IEEE=self.ieee,
                             num_warps=self.warps_fwd)
        return p1.sum(0), p2.sum(0)

    def gap_fwd(self, X, C: int, mean, rstd, g, b, n: int, hw: int):
        BC = 64
        BH = triton.next_power_of_2(min(hw, 32))
        Fo = torch.empty(n, C, dtype=torch.float32, device=X.device)
        _gap_fwd[(n, triton.cdiv(C, BC))](X, X.stride(0), hw, C, _f32(mean), _f32(rstd), _f32(g),
                                          _f32(b), Fo, BH=BH, BC=BC)
        return Fo.to(X.dtype)

    # --- backward ---------------------------------------------------------------------

    def gap_bwd(self, dfeat, X, C: int, mean, rstd, g, b, n: int, hw: int):
        BC = 64
        BH = triton.next_power_of_2(min(hw, 32))
        dF = dfeat.contiguous()
        p1 = torch.empty(n, C, dtype=torch.float32, device=X.device)
        p2 = torch.empty_like(p1)
        G = torch.empty(X.shape[0], C, dtype=X.dtype, device=X.device)
        _gap_bwd_fused[(n, triton.cdiv(C, BC))](dF, hw, X, X.stride(0), C, _f32(mean), _f32(rstd),
                                                _f32(g), _f32(b), G, G.stride(0), p1, p2,
                                                BH=BH, BC=BC)
        return G, p2.sum(0), p1.sum(0)

    def trans_bwd(self, dT, X, C: int, mean, rstd, g, b, W, n: int, h: int, w: int):
        M2, NO = n * (h // 2) * (w // 2), W.shape[0]
        W = W.contiguous()
        BM, BC, BO = self.bm_bwd, self.bc_bwd, 32
        nc = triton.cdiv(C, BC)
        split = self._split(M2, BM, nc)
        args = (_f32(mean), _f32(rstd), _f32(g), _f32(b))
        G = torch.empty(X.shape[0], C, dtype=X.dtype, device=X.device)
        p1 = torch.empty(split, C, dtype=torch.float32, device=X.device)
        p2 = torch.empty_like(p1)
        _trans_bwd_fused[(nc, split)](dT, dT.stride(0), M2, h, w, X, X.stride(0), C, *args, W, NO,
                                      G, G.stride(0), p1, p2, split, BM=BM, BC=BC, BO=BO,
                                      IEEE=self.ieee)
        BOW = 64
        no_t = triton.cdiv(NO, BOW)
        split_w = max(1, min(triton.cdiv(M2, BM), (160 + no_t * nc - 1) // (no_t * nc)))
        pw = torch.empty(split_w, NO, C, dtype=torch.float32, device=X.device)
        _trans_wgrad[(no_t, nc, split_w)](dT, dT.stride(0), M2, h, w, X, X.stride(0), C, *args, NO,
                                          pw, split_w, BM=BM, BC=BC, BO=BOW, IEEE=self.ieee)
        return G, pw.sum(0), p2.sum(0), p1.sum(0)

    def bnrelu_bwd(self, dV, U, mean, rstd, g, b):
        M, N = U.shape
        BM, BN = self.bm_ew, min(128, triton.next_power_of_2(N))
        gm, gn = triton.cdiv(M, BM), triton.cdiv(N, BN)
        dV = dV.contiguous()
        args = (_f32(mean), _f32(rstd), _f32(g), _f32(b))
        p1 = torch.empty(gm, N, dtype=torch.float32, device=U.device)
        p2 = torch.empty_like(p1)
        _bn_bwd_reduce[(gm, gn)](dV, U, M, N, *args, p1, p2, BM=BM, BN=BN)
        db, dg = p1.sum(0), p2.sum(0)
        dU = torch.empty_like(U)
        _bn_bwd_apply[(gm, gn)](dV, U, dU, M, N, *args, db, dg, BM=BM, BN=BN)
        return dU, dg, db

    def bnrelu_gemm_bwd(self, dU, W, X, cin: int, mean, rstd, g, b, G):
        M, NO = dU.shape
        if NO != BOTTLENECK:
            raise ValueError(f"gemm_bwd: bottleneck of {NO} channels, expected {BOTTLENECK}")
        W = W.contiguous()
        dU = dU.contiguous()
        BM, BC = self.bm_bwd, self.bc_bwd
        nc = triton.cdiv(cin, BC)
        split = self._split(M, BM, nc)
        pr1 = torch.empty(split, cin, dtype=torch.float32, device=X.device)
        pr2 = torch.empty_like(pr1)
        pw = torch.empty(split, NO, cin, dtype=torch.float32, device=X.device)
        _gemm_bwd_fused[(nc, split)](dU, M, X, X.stride(0), cin, _f32(mean), _f32(rstd), _f32(g),
                                     _f32(b), W, G, G.stride(0), pr1, pr2, pw, split,
                                     NO=NO, BM=BM, BC=BC, IEEE=self.ieee, num_warps=self.warps_bwd)
        return pw.sum(0), pr2.sum(0), pr1.sum(0)

    def conv3_fwd(self, U, mean2, rstd2, g2, b2, W2, X, c0: int, shift, n: int, h: int, w: int,
                  save: bool = True):
        if self.conv == "cudnn":
            return conv3_cudnn_fwd(self, U, mean2, rstd2, g2, b2, W2, X, c0, shift, n, h, w)
        M, CI = U.shape
        CO = W2.shape[0]
        Wr = W2.permute(2, 3, 1, 0).reshape(9, CI, CO).contiguous()
        V = self.bnrelu(U, mean2, rstd2, g2, b2) if self.conv == "triton_v" else None
        BM = self.bm_conv
        gm = triton.cdiv(M, BM)
        p1 = torch.empty(gm, CO, dtype=torch.float32, device=U.device)
        p2 = torch.empty_like(p1)
        _conv3_fwd[(gm,)](U if V is None else V, M, h, w, _f32(mean2), _f32(rstd2), _f32(g2),
                          _f32(b2), Wr, X, X.stride(0), c0, _f32(shift), p1, p2, CI=CI, CO=CO,
                          BM=BM, BK=32, IEEE=self.ieee, PRO=V is None, num_warps=self.warps_conv)
        return p1.sum(0), p2.sum(0), V

    def conv3_bwd(self, dS, U, saved, mean2, rstd2, g2, b2, W2, n: int, h: int, w: int):
        if self.conv == "cudnn":
            return conv3_cudnn_bwd(self, dS, U, saved, mean2, rstd2, g2, b2, W2, n, h, w)
        M, CI = U.shape
        CO = W2.shape[0]
        Wr = W2.permute(2, 3, 1, 0).reshape(9, CI, CO).contiguous()
        dS = dS.contiguous()
        args = (_f32(mean2), _f32(rstd2), _f32(g2), _f32(b2))
        BM = self.bm_conv_bwd
        gm = triton.cdiv(M, BM)
        p1 = torch.empty(gm, CI, dtype=torch.float32, device=U.device)
        p2 = torch.empty_like(p1)
        _conv3_bwd[(gm,)](dS, M, h, w, U, *args, Wr, p1, p2, U, p1, p2, APPLY=False,
                          CI=CI, CO=CO, BM=BM, IEEE=self.ieee)
        db, dg = p1.sum(0), p2.sum(0)
        dU = torch.empty_like(U)
        _conv3_bwd[(gm,)](dS, M, h, w, U, *args, Wr, db, dg, dU, p1, p2, APPLY=True,
                          CI=CI, CO=CO, BM=BM, IEEE=self.ieee)
        BMW = 64
        split = max(1, min(triton.cdiv(M, BMW), 18))
        pw = torch.empty(split, 9, CI, CO, dtype=torch.float32, device=U.device)
        _conv3_wgrad[(9, split)](dS, M, h, w, U if saved is None else saved, *args, pw, split,
                                 CI=CI, CO=CO, BM=BMW, IEEE=self.ieee, PRO=saved is None)
        dW2 = pw.sum(0).permute(2, 1, 0).reshape(CO, CI, 3, 3)
        return dU, dW2, dg, db

    # --- BatchNorm statistics in one launch (`fastdense._bn_stats`, `_bn_coeffs_add`) -------
    def bn_stats(self, s1, s2, shift, n: int, bn):
        C = s1.shape[0]
        mean = torch.empty(C, dtype=torch.float32, device=s1.device)
        rstd = torch.empty_like(mean)
        nbt = bn.num_batches_tracked
        BLOCK = 256
        _bn_stats_k[(triton.cdiv(C, BLOCK),)](
            s1.contiguous(), s2.contiguous(), _f32(shift), bn.running_mean, bn.running_var,
            nbt if nbt is not None else mean, mean, rstd, C, 1.0 / n, n / max(n - 1, 1),
            float(bn.eps), float(bn.momentum), UPD_NBT=nbt is not None, BLOCK=BLOCK)
        return mean, rstd

    def bn_coeffs_add(self, mean, rstd, g, dg, db, P: int, A, B):
        C = dg.shape[0]
        BLOCK = 256
        _bn_coeffs_k[(triton.cdiv(C, BLOCK),)](_f32(mean), _f32(rstd), g.detach(), _f32(dg), _f32(db),
                                              A, B, C, 1.0 / P, BLOCK=BLOCK)

    # --- the fused stem (`fastdense.StemFn`) ----------------------------------------------------
    stem_fused = True

    def stem_fwd(self, Y, n: int, H: int, W: int, mean, rstd, g, b):
        """Y (n*H*W x C NHWC rows): S = maxpool3x3/2(relu(bn(Y))) (n*Ho*Wo x C) and the tap of the maximum
        (uint8). -> (S, AM): `StemFn` saves everything that follows S and passes it back to `stem_bwd`.
        """
        return self._stem_fwd(Y, n, H, W, mean, rstd, g, b, with_ym=False)

    def _stem_fwd(self, Y, n: int, H: int, W: int, mean, rstd, g, b, with_ym: bool):
        C = Y.shape[1]
        Ho, Wo = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        M2 = n * Ho * Wo
        S = torch.empty(M2, C, dtype=Y.dtype, device=Y.device)
        AM = torch.empty(M2, C, dtype=torch.uint8, device=Y.device)
        YM = torch.empty_like(S) if with_ym else S
        BM = 32
        _stem_pool_fwd[(triton.cdiv(M2, BM),)](Y, H, W, Ho, Wo, M2, _f32(mean), _f32(rstd), _f32(g),
                                              _f32(b), S, AM, YM, C=C, BM=BM, WITH_YM=with_ym)
        return (S, AM, YM) if with_ym else (S, AM)

    def stem_bwd(self, Y, n: int, H: int, W: int, mean, rstd, g, b, dS, S, AM):
        """-> dY (like Y), dgamma, dbeta of norm0."""
        M, C = Y.shape
        Ho, Wo = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        args = (_f32(mean), _f32(rstd), _f32(g), _f32(b))
        BM = 32
        split = max(1, min(triton.cdiv(M, BM), 4 * 160))
        p1 = torch.empty(split, C, dtype=torch.float32, device=Y.device)
        p2 = torch.empty_like(p1)
        zero = torch.zeros(C, dtype=torch.float32, device=Y.device)
        dS = dS.contiguous()
        _stem_pool_bwd[(split,)](Y, H, W, Ho, Wo, M, *args, dS, S, AM, Y, zero, zero, p1, p2, split,
                                 APPLY=False, C=C, BM=BM)
        db, dg = p1.sum(0), p2.sum(0)
        dY = torch.empty_like(Y)
        n_t = triton.cdiv(M, BM)
        _stem_pool_bwd[(n_t,)](Y, H, W, Ho, Wo, M, *args, dS, S, AM, dY, (db / M).contiguous(),
                               (dg / M).contiguous(), p1, p2, n_t, APPLY=True, C=C, BM=BM)
        return dY, dg, db

    def dcols(self, G, X, c0: int, c1: int, A, B):
        M, N = G.shape[0], c1 - c0
        BM, BN = self.bm_ew, triton.next_power_of_2(min(N, 64))
        O = torch.empty(M, N, dtype=G.dtype, device=G.device)
        _dcols[(triton.cdiv(M, BM), triton.cdiv(N, BN))](G, G.stride(0), X, X.stride(0), c0, M, N,
                                                          _f32(A), _f32(B), O, BM=BM, BN=BN)
        return O


_MM_OUT_DTYPE: list = []          # is torch.mm(..., out_dtype=float32) available? (checked once)


def _mm_wgrad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b for the weight gradients (sums over all pixels): fp32 output if torch supports it
    (`torch.mm(..., out_dtype=)` on the GPU), otherwise fp16 like cuDNN in eager under autocast.
    """
    if not _MM_OUT_DTYPE or _MM_OUT_DTYPE[0]:
        try:
            out = torch.mm(a, b, out_dtype=torch.float32)
            if not _MM_OUT_DTYPE:
                _MM_OUT_DTYPE.append(True)
            return out
        except (TypeError, RuntimeError, NotImplementedError):
            if not _MM_OUT_DTYPE:
                _MM_OUT_DTYPE.append(False)
    return torch.mm(a, b)


class CublasOps(TritonOps):
    """The GEMMs on cuBLAS and the 3x3 conv on cuDNN (tensor cores), Triton only for the elementwise
    parts and the per-column reductions.

    Why: on the T4 (sm_75) Triton does not emit mma instructions, `tl.dot` becomes fp32 FMA (PTX for
    sm_75: 2,048 fma and no mma; for sm_80, 32 mma.sync). The fused kernels were correct but took
    16.8 s per step against 0.79 for eager. Here the prologue fusion is lost (relu(bn(x)) is written
    and re-read by the GEMM; in the backward dY and the V of dW too) and what remains is the per-block
    buffer without concatenations, statistics once per channel and the deferred BN corrections. The
    reductions run in row slices (`programs` programs in total), the partials are small.
    """

    name = "cublas"

    def __init__(self, bm_ew: int = 64, bm_red: int = 64, bc_red: int = 64, programs: int = 160,
                 **kw):
        kw.pop("conv", None)
        super().__init__(conv="cudnn", bm_ew=bm_ew, **kw)
        self.name = "cublas"
        self.bm_red, self.bc_red, self.programs = int(bm_red), int(bc_red), int(programs)

    def _nsplit(self, M: int, BM: int, ncol: int) -> int:
        want = max(1, (self.programs + ncol - 1) // ncol)
        return int(max(1, min(triton.cdiv(M, BM), want)))

    def _stats(self, S, N: int, shift, X=None, c0: int = 0):
        """Per-column sums of S[:, :N] - shift (and, with X, the copy into X[:, c0:c0+N])."""
        M = S.shape[0]
        BM, BN = self.bm_red, min(64, triton.next_power_of_2(N))
        nt = triton.cdiv(N, BN)
        split = self._nsplit(M, BM, nt)
        p1 = torch.empty(split, N, dtype=torch.float32, device=S.device)
        p2 = torch.empty_like(p1)
        Xd = S if X is None else X
        _colstats_split[(nt, split)](S, S.stride(0), Xd, Xd.stride(0), c0, M, N, _f32(shift), p1,
                                     p2, split, COPY=X is not None, BM=BM, BN=BN)
        return p1.sum(0), p2.sum(0)

    def colstats(self, X, c0: int, c1: int, shift):
        return self._stats(X[:, c0:], c1 - c0, shift)

    def write_segment(self, X, c0: int, S, shift):
        return self._stats(S, S.shape[1], shift, X=X, c0=c0)

    def _bnrelu_to(self, X, N: int, mean, rstd, g, b):
        """relu(X[:, :N] a + c) into a new contiguous M x N tensor."""
        M = X.shape[0]
        V = torch.empty(M, N, dtype=X.dtype, device=X.device)
        BM, BN = self.bm_ew, min(64, triton.next_power_of_2(N))
        _bnrelu_rows[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
            X, X.stride(0), V, N, M, N, _f32(mean), _f32(rstd), _f32(g), _f32(b), BM=BM, BN=BN)
        return V

    def bnrelu(self, U, mean, rstd, g, b):
        return self._bnrelu_to(U, U.shape[1], mean, rstd, g, b)

    def bnrelu_gemm(self, X, cin: int, mean, rstd, g, b, W, ushift):
        V1 = self._bnrelu_to(X, cin, mean, rstd, g, b)
        U = torch.mm(V1, W.t())
        s1, s2 = self._stats(U, U.shape[1], ushift)
        return U, s1, s2

    def _pool(self, X, C: int, mean, rstd, g, b, h: int, w: int, M2: int):
        Y = torch.empty(M2, C, dtype=X.dtype, device=X.device)
        BM, BK = self.bm_ew, 64
        _trans_pool[(triton.cdiv(M2, BM), triton.cdiv(C, BK))](
            X, X.stride(0), h, w, M2, C, _f32(mean), _f32(rstd), _f32(g), _f32(b), Y, BM=BM, BK=BK)
        return Y

    def trans_fwd(self, X, C: int, mean, rstd, g, b, W, n: int, h: int, w: int, out, oshift):
        M2 = n * (h // 2) * (w // 2)
        T = torch.mm(self._pool(X, C, mean, rstd, g, b, h, w, M2), W.t())
        return self.write_segment(out, 0, T, oshift)

    def trans_bwd(self, dT, X, C: int, mean, rstd, g, b, W, n: int, h: int, w: int):
        M2 = n * (h // 2) * (w // 2)
        dT = dT.contiguous()
        dYp = torch.mm(dT, W)                                          # (M2, C)
        odd = (h % 2) or (w % 2)                     # pixels outside the mean: zero gradient
        G = (torch.zeros if odd else torch.empty)(X.shape[0], C, dtype=X.dtype, device=X.device)
        BM, BC = self.bm_red, self.bc_red
        nc = triton.cdiv(C, BC)
        split = self._nsplit(M2, BM, nc)
        p1 = torch.empty(split, C, dtype=torch.float32, device=X.device)
        p2 = torch.empty_like(p1)
        _trans_bwd_ew[(nc, split)](dYp, M2, h, w, X, X.stride(0), C, _f32(mean), _f32(rstd),
                                   _f32(g), _f32(b), G, G.stride(0), p1, p2, split, BM=BM, BC=BC)
        dW = _mm_wgrad(dT.t(), self._pool(X, C, mean, rstd, g, b, h, w, M2))
        return G, dW, p2.sum(0), p1.sum(0)

    def _bn_bwd_sums(self, dV, U, mean, rstd, g, b):
        """The BN + ReLU backward reductions: (sum of gm, sum of gm uhat)."""
        M, N = U.shape
        args = (_f32(mean), _f32(rstd), _f32(g), _f32(b))
        BM, BN = self.bm_red, min(64, triton.next_power_of_2(N))
        nt = triton.cdiv(N, BN)
        split = self._nsplit(M, BM, nt)
        p1 = torch.empty(split, N, dtype=torch.float32, device=U.device)
        p2 = torch.empty_like(p1)
        _bn_bwd_reduce_split[(nt, split)](dV.contiguous(), U, M, N, *args, p1, p2, split, BM=BM,
                                          BN=BN)
        return p1.sum(0), p2.sum(0)

    def bnrelu_bwd(self, dV, U, mean, rstd, g, b):
        M, N = U.shape
        dV = dV.contiguous()
        args = (_f32(mean), _f32(rstd), _f32(g), _f32(b))
        db, dg = self._bn_bwd_sums(dV, U, mean, rstd, g, b)
        dU = torch.empty_like(U)
        BMa, BNa = self.bm_ew, min(128, triton.next_power_of_2(N))
        _bn_bwd_apply[(triton.cdiv(M, BMa), triton.cdiv(N, BNa))](dV, U, dU, M, N, *args, db, dg,
                                                                  BM=BMa, BN=BNa)
        return dU, dg, db

    def bnrelu_gemm_bwd(self, dU, W, X, cin: int, mean, rstd, g, b, G):
        M = X.shape[0]
        dU = dU.contiguous()
        dY = torch.mm(dU, W)                                           # (M, cin)
        V1 = torch.empty(M, cin, dtype=X.dtype, device=X.device)
        BM, BC = self.bm_red, self.bc_red
        nc = triton.cdiv(cin, BC)
        split = self._nsplit(M, BM, nc)
        p1 = torch.empty(split, cin, dtype=torch.float32, device=X.device)
        p2 = torch.empty_like(p1)
        _gemm_bwd_ew[(nc, split)](dY, M, X, X.stride(0), cin, _f32(mean), _f32(rstd), _f32(g),
                                  _f32(b), G, G.stride(0), V1, p1, p2, split, BM=BM, BC=BC)
        del dY
        dW = _mm_wgrad(dU.t(), V1)
        return dW, p2.sum(0), p1.sum(0)


__all__ = ["TritonOps", "CublasOps"]
