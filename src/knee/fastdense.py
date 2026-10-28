"""fastdense: the training step of timm's DenseNet without concatenations.

Without a local GPU every training seed costs hours of rented GPU, so the step has to be cheap.
The step of a DenseNet169 at 288-512 px is memory-bound, and almost all the memory traffic comes from
timm's DenseNet as written: every layer of a block concatenates ALL inputs again (`torch.cat`),
recomputes their BatchNorm statistics, writes their normalised and ReLU'd version, and in the
backward does the same plus the accumulation of the segment gradients. Traffic grows with the
square of the number of layers in the block.

Here the same function (in exact arithmetic) with a different computation scheme:
- one buffer per block, P x C_tot (P = N*H*W pixels, contiguous channels: NHWC layout), into which
  every layer writes its 32 channels ONCE;
- the batch statistics of a channel (sum and sum of squares, shifted by a constant for precision)
  are computed once, when the channel is born: every BatchNorm that reads it (norm1 of the later
  layers, transition, norm5) sees the same batch mean and variance, and only gamma and beta differ.
  The maths is BatchNorm in training, to the letter of the formulas: every consumer has its own
  norm, with the same statistics;
- BatchNorm + ReLU of the conv1 input (and of the transition, and of norm5) are applied while the
  GEMM loads (prologue), without writing the normalised tensor;
- in the backward, for every layer: dU (the gradient of the 128-channel bottleneck) and then two
  passes over its input, one for the BatchNorm reductions (sum of g and of g * xhat per channel,
  together with the conv1 weight gradient) and one writing dX by adding it into the block's
  gradient buffer. dY = dU W1 is recomputed in both passes instead of being written.

The transition (BN, ReLU, 1x1 conv, 2x2 mean) takes the mean BEFORE the conv (1x1 conv and mean are
linear and commute: same function, a quarter of the compute) and writes directly into the next
block's buffer, with its statistics. The 3x3 conv of every layer stays on cuDNN (NHWC, fp16) unless
the CUDA kernels of `knee.fastdense_cuda` are used.

Several engines for the same primitives: `RefOps` (torch operations: the oracle, also runs on CPU
in float64, checked against timm's autograd), the Triton kernels (`knee.fastdense_triton`) and the
CUDA kernels (`knee.fastdense_cuda`). The orchestration (`EncoderFn`) is one.

What does NOT change: `install(enc)` replaces only the instance's `forward`, so parameters, buffers
and names are those of the timm module (same state_dict: same checkpoints and inference), BatchNorm
running averages are updated as in `F.batch_norm` (variance corrected by n/(n-1), momentum 0.1,
`num_batches_tracked`), the stem (7x7 conv, norm0, max-pool) stays eager. In `eval` the forward is
timm's, unchanged. On the first fold of a 288 px DenseNet169 the fast step matched eager quality
(0.9082 against 0.9074) at 2.4x the speed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

GROWTH = 32
BOTTLENECK = 128


# ---------------------------------------------------------------------------------------------
# layout of timm's DenseNet
# ---------------------------------------------------------------------------------------------

@dataclass
class BlockSpec:
    layers: list            # timm DenseLayer (norm1, conv1, norm2, conv2)
    c0: int                 # channels of the block input
    trans: nn.Module | None  # DenseTransition after the block, or None (last one: norm5)

    @property
    def c_total(self) -> int:
        return self.c0 + GROWTH * len(self.layers)


@dataclass
class EncoderSpec:
    blocks: list[BlockSpec]
    norm5: nn.Module
    params: list[torch.Tensor] = field(default_factory=list)   # in `param_list` order
    # "gap": the global mean (N, C), the usual path. "map": the whole relu(norm5), (N, C, H, W), like
    # timm's `enc(x)` with `global_pool=""`.
    out: str = "gap"


def encoder_spec(enc: nn.Module) -> EncoderSpec:
    """Blocks, transitions and norm5 of a timm DenseNet (or `knee.models.densenet`)."""
    f = enc.features
    blocks = []
    for i in range(1, 10):
        blk = getattr(f, f"denseblock{i}", None)
        if blk is None:
            break
        layers = [m for _, m in blk.named_children()]
        for lay in layers:
            if lay.conv1.out_channels != BOTTLENECK or lay.conv2.out_channels != GROWTH:
                raise ValueError("fastdense: expected a DenseNet with bottleneck 128 and growth 32")
            if getattr(lay, "drop_rate", 0.0):
                raise ValueError("fastdense: drop_rate > 0 not supported")
        trans = getattr(f, f"transition{i}", None)
        blocks.append(BlockSpec(layers=layers, c0=layers[0].conv1.in_channels, trans=trans))
    return EncoderSpec(blocks=blocks, norm5=f.norm5)


def param_list(spec: EncoderSpec) -> list[nn.Parameter]:
    """The parameters in the order `EncoderFn` receives them (and returns their gradients)."""
    out = []
    for blk in spec.blocks:
        for lay in blk.layers:
            out += [lay.norm1.weight, lay.norm1.bias, lay.conv1.weight,
                    lay.norm2.weight, lay.norm2.bias, lay.conv2.weight]
        if blk.trans is not None:
            out += [blk.trans.norm.weight, blk.trans.norm.bias, blk.trans.conv.weight]
    out += [spec.norm5.weight, spec.norm5.bias]
    return out


# ---------------------------------------------------------------------------------------------
# statistics and running averages (as F.batch_norm in training)
# ---------------------------------------------------------------------------------------------

def finalize_stats(s1: torch.Tensor, s2: torch.Tensor, shift: torch.Tensor, n: int, eps: float):
    """From the sum and sum of squares of (x - shift) over n rows: mean, rstd, corrected variance.
    All in float32 (float64 if the sums are).
    """
    m1 = s1 / n
    var = (s2 / n - m1 * m1).clamp_min(0.0)
    mean = shift.to(s1.dtype) + m1
    rstd = torch.rsqrt(var + eps)
    return mean, rstd, var * (n / max(n - 1, 1))


@torch.no_grad()
def update_running(bn: nn.Module, mean: torch.Tensor, var_unbiased: torch.Tensor) -> None:
    """The running averages of `bn` as in `F.batch_norm` (training, fixed momentum)."""
    if not bn.track_running_stats or bn.running_mean is None:
        return
    if bn.num_batches_tracked is not None:
        bn.num_batches_tracked.add_(1)
    m = bn.momentum if bn.momentum is not None else 1.0 / float(bn.num_batches_tracked)
    c = bn.running_mean.shape[0]
    bn.running_mean.mul_(1.0 - m).add_(mean[:c].to(bn.running_mean.dtype), alpha=m)
    bn.running_var.mul_(1.0 - m).add_(var_unbiased[:c].to(bn.running_var.dtype), alpha=m)


# ---------------------------------------------------------------------------------------------
# primitives: the oracle in torch operations
# ---------------------------------------------------------------------------------------------

class RefOps:
    """The `EncoderFn` primitives in torch operations: clear, slow, exact. Matrices are 2D P x C (rows =
    pixels in n, h, w order; columns = channels), possibly seen with a row stride.
    """

    name = "ref"

    @staticmethod
    def _affine(mean, rstd, g, b):
        a = g.to(rstd.dtype) * rstd
        return a, b.to(rstd.dtype) - mean * a

    def colstats(self, X, c0: int, c1: int, shift):
        dt = torch.float64 if X.dtype == torch.float64 else torch.float32
        x = X[:, c0:c1].to(dt) - shift.to(dt)
        return x.sum(0), (x * x).sum(0)

    def bnrelu_gemm(self, X, cin: int, mean, rstd, g, b, W, ushift):
        """U = relu(bn(X[:, :cin])) @ W^T (W: cout x cin), and the sums of U for the next BN."""
        a, c = self._affine(mean[:cin], rstd[:cin], g, b)
        y = torch.relu(X[:, :cin].to(a.dtype) * a + c).to(X.dtype)
        U = y @ W.t().to(X.dtype)
        s1, s2 = self.colstats(U, 0, U.shape[1], ushift)
        return U, s1, s2

    def bnrelu(self, U, mean, rstd, g, b):
        a, c = self._affine(mean, rstd, g, b)
        return torch.relu(U.to(a.dtype) * a + c).to(U.dtype)

    def write_segment(self, X, c0: int, S, shift):
        X[:, c0:c0 + S.shape[1]] = S
        return self.colstats(X, c0, c0 + S.shape[1], shift)

    def trans_fwd(self, X, C: int, mean, rstd, g, b, W, n: int, h: int, w: int, out, oshift):
        """out[:, :cout] = avgpool2x2(relu(bn(X))) @ W^T; returns the sums of the output."""
        a, c = self._affine(mean[:C], rstd[:C], g, b)
        y = torch.relu(X[:, :C].to(a.dtype) * a + c)
        yp = F.avg_pool2d(y.reshape(n, h, w, C).permute(0, 3, 1, 2), 2)       # (n, C, h/2, w/2)
        yp = yp.permute(0, 2, 3, 1).reshape(-1, C).to(X.dtype)
        T = yp @ W.t().to(X.dtype)
        out[:, :T.shape[1]] = T
        return self.colstats(out, 0, T.shape[1], oshift)

    def gap_fwd(self, X, C: int, mean, rstd, g, b, n: int, hw: int):
        a, c = self._affine(mean[:C], rstd[:C], g, b)
        y = torch.relu(X[:, :C].to(a.dtype) * a + c)
        return y.reshape(n, hw, C).mean(1)

    # --- backward ---------------------------------------------------------------------
    #
    # BN (training) + ReLU backward, with gm = dY * [x a + c > 0] and xhat = (x - mean) rstd:
    #   dX = a gm - a R1/P - a xhat R2/P,   R1 = sum_p gm (= dbeta),  R2 = sum_p gm xhat (= dgamma).
    # The two correction terms are affine in x per channel: -a R1/P - a xhat R2/P = alpha + beta x,
    # beta = -a rstd R2/P, alpha = -a R1/P - beta mean (`split_coeffs`). Every consumer of a channel reads
    # the SAME values x, so the sum of the corrections of all consumers is (sum alpha) + (sum beta) x:
    # the primitives write only a gm into the gradient buffer G, and the orchestrator accumulates alpha
    # and beta per channel (A, B) and applies them once, when the gradient of the channel is complete
    # (`dcols`). This way every layer reads its input only once.

    def _bn_bwd_gm(self, dy, X, mean, rstd, g, b):
        """a gm, dgamma (R2), dbeta (R1) of BN + ReLU with input X and gradient dY."""
        a, c = self._affine(mean, rstd, g, b)
        x = X.to(a.dtype)
        mask = (x * a + c) > 0
        gm = torch.where(mask, dy.to(a.dtype), torch.zeros((), dtype=a.dtype, device=a.device))
        xhat = (x - mean) * rstd
        return a * gm, (gm * xhat).sum(0), gm.sum(0)

    def gap_bwd(self, dfeat, X, C: int, mean, rstd, g, b, n: int, hw: int):
        """-> G (P x C) = a gm, dgamma, dbeta of norm5."""
        dy = dfeat.to(rstd.dtype) / hw
        dy = dy[:, None, :].expand(n, hw, C).reshape(-1, C)
        agm, dg, db = self._bn_bwd_gm(dy, X[:, :C], mean[:C], rstd[:C], g, b)
        return agm.to(X.dtype), dg, db

    def trans_bwd(self, dT, X, C: int, mean, rstd, g, b, W, n: int, h: int, w: int):
        """dT: (P/4 x cout) gradient of the transition output. -> G (P x C) = a gm, dW, dg, db."""
        a, c = self._affine(mean[:C], rstd[:C], g, b)
        y = torch.relu(X[:, :C].to(a.dtype) * a + c)
        yp = F.avg_pool2d(y.reshape(n, h, w, C).permute(0, 3, 1, 2), 2)
        yp = yp.permute(0, 2, 3, 1).reshape(-1, C)
        dTf = dT.to(a.dtype)
        dW = dTf.t() @ yp
        dyp = dTf @ W.to(a.dtype)                                             # (P/4, C)
        dy = dyp.reshape(n, h // 2, w // 2, C).repeat_interleave(2, 1).repeat_interleave(2, 2)
        dy = dy.reshape(-1, C) / 4.0
        agm, dg, db = self._bn_bwd_gm(dy, X[:, :C], mean[:C], rstd[:C], g, b)
        return agm.to(X.dtype), dW, dg, db

    def bnrelu_bwd(self, dV, U, mean, rstd, g, b):
        """The bottleneck BN backward, complete (the conv1 GEMM needs the whole dU)."""
        agm, dg, db = self._bn_bwd_gm(dV, U, mean, rstd, g, b)
        al, be = split_coeffs(mean, rstd, g, dg, db, U.shape[0])
        return (agm + al + be * U.to(agm.dtype)).to(U.dtype), dg, db

    def bnrelu_gemm_bwd(self, dU, W, X, cin: int, mean, rstd, g, b, G):
        """From dU (P x cout): dW (cout x cin), dgamma, dbeta of norm1; G[:, :cin] += a gm (the corrections
        are accumulated by the caller, `split_coeffs`).
        """
        a, c = self._affine(mean[:cin], rstd[:cin], g, b)
        x = X[:, :cin].to(a.dtype)
        y = torch.relu(x * a + c)
        dUf = dU.to(a.dtype)
        dW = dUf.t() @ y
        dy = dUf @ W.to(a.dtype)
        agm, dg, db = self._bn_bwd_gm(dy, X[:, :cin], mean[:cin], rstd[:cin], g, b)
        G[:, :cin] += agm.to(G.dtype)
        return dW, dg, db

    def dcols(self, G, X, c0: int, c1: int, A, B):
        """The complete gradient of channels c0..c1: G + A + B x, contiguous (P x (c1 - c0))."""
        dt = A.dtype
        out = G[:, c0:c1].to(dt) + A.to(dt) + B.to(dt) * X[:, c0:c1].to(dt)
        return out.to(G.dtype).contiguous()

    def conv3_fwd(self, U, mean2, rstd2, g2, b2, W2, X, c0: int, shift, n: int, h: int, w: int,
                  save: bool = True):
        return conv3_cudnn_fwd(self, U, mean2, rstd2, g2, b2, W2, X, c0, shift, n, h, w)

    def conv3_bwd(self, dS, U, saved, mean2, rstd2, g2, b2, W2, n: int, h: int, w: int):
        return conv3_cudnn_bwd(self, dS, U, saved, mean2, rstd2, g2, b2, W2, n, h, w)


def _bn_stats(ops, s1, s2, shift, n: int, bn: nn.Module):
    """`finalize_stats` + `update_running` in one launch if the ops support it (`bn_stats` of the Triton
    kernels: ~14 tiny torch operations per BatchNorm, ~170 BatchNorms per step), otherwise in torch.
    -> mean, rstd. s1, s2 can be not-yet-reduced sums from the ops (`fastdense_cuda.Partial`): they go
    to the ops that produced them.
    """
    f = getattr(ops, "bn_stats", None)
    if not isinstance(s1, torch.Tensor):
        return f(s1, s2, shift, n, bn)
    if (f is not None and s1.is_cuda and s1.dtype == torch.float32 and bn.track_running_stats
            and bn.momentum is not None and bn.running_mean is not None
            and bn.running_mean.dtype == torch.float32 and bn.running_mean.shape[0] == s1.shape[0]):
        return f(s1, s2, shift, n, bn)
    mean, rstd, var_u = finalize_stats(s1, s2, shift, n, bn.eps)
    update_running(bn, mean, var_u)
    return mean, rstd


def _store_stats(ops, s1, s2, lo: int, a1, a2) -> None:
    """The sums of a new segment into those of the block buffer (s1, s2 from column lo); ops that return
    not-yet-reduced sums close them here (`store_stats`).
    """
    f = getattr(ops, "store_stats", None)
    if f is not None:
        return f(s1, s2, lo, a1, a2)
    s1[lo:lo + a1.shape[0]] = a1
    s2[lo:lo + a2.shape[0]] = a2


def _cw(ops, w: torch.Tensor, dt) -> torch.Tensor:
    """The weight of a conv in the GEMM dtype: from the ops if they prepare it (`conv_weight`: the
    `prep_weights` arena), otherwise `w.to(dt)`.
    """
    f = getattr(ops, "conv_weight", None)
    return f(w, dt) if f is not None else w.to(dt)


def _bn_coeffs_add(ops, mean, rstd, g, dg, db, P: int, A, B):
    """A += alpha, B += beta (`split_coeffs`) in one launch if the ops support it."""
    f = getattr(ops, "bn_coeffs_add", None)
    if f is not None and A.is_cuda and A.dtype == torch.float32:
        return f(mean, rstd, g, dg, db, P, A, B)
    al, be = split_coeffs(mean, rstd, g, dg, db, P)
    A += al
    B += be


def split_coeffs(mean, rstd, g, dg, db, P: int):
    """alpha, beta of the BN backward correction (see `RefOps`): dX = a gm + alpha + beta x."""
    C = dg.shape[0]
    a = g.to(rstd.dtype) * rstd[:C]
    beta = -a * rstd[:C] * dg / P
    alpha = -a * db / P - beta * mean[:C]
    return alpha, beta


# ---------------------------------------------------------------------------------------------
# orchestration: forward and backward of the encoder after the stem
# ---------------------------------------------------------------------------------------------

def _map_fwd(X, C: int, mean, rstd, g, b, n: int, h: int, w: int) -> torch.Tensor:
    """`out="map"`: relu(norm5(X)) as a map (n, C, h, w), a channels_last view of the NHWC rows. In torch:
    it is the last layer at 9x9, ~1% of the compute, the fused kernels are for the mean.
    """
    a, c = RefOps._affine(mean[:C], rstd[:C], g, b)
    y = torch.relu(X[:, :C].to(a.dtype) * a + c).to(X.dtype)
    return _as_nchw(y, n, h, w)


def _map_bwd(dmap, X, C: int, mean, rstd, g, b):
    """The backward of `_map_fwd` in training: G (P x C) = a gm, dgamma, dbeta of norm5, like `gap_bwd`
    but with the per-pixel gradient instead of the spread mean gradient.
    """
    agm, dg, db = RefOps()._bn_bwd_gm(_nhwc_rows(dmap), X[:, :C], mean[:C], rstd[:C], g, b)
    return agm.to(X.dtype).contiguous(), dg, db


def _nhwc_rows(t: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) in any format -> (N*H*W, C) contiguous (rows = pixels)."""
    n, c, h, w = t.shape
    return t.permute(0, 2, 3, 1).reshape(n * h * w, c).contiguous()


def _as_nchw(rows: torch.Tensor, n: int, h: int, w: int) -> torch.Tensor:
    """(N*H*W, C) with contiguous rows -> channels_last (N, C, H, W) view, without a copy."""
    c = rows.shape[1]
    return rows.reshape(n, h, w, c).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------------------------
# the 3x3 bottleneck conv with cuDNN (BN2 + ReLU materialised in V): oracle and fallback
# ---------------------------------------------------------------------------------------------

def conv3_cudnn_fwd(ops, U, mean2, rstd2, g2, b2, W2, X, c0: int, shift, n: int, h: int, w: int):
    """V = relu(bn2(U)); X[:, c0:c0+32] = conv3x3(V, W2); the segment sums. Saves V."""
    V = ops.bnrelu(U, mean2, rstd2, g2, b2)
    w2c = W2.contiguous(memory_format=torch.channels_last) if V.is_cuda else W2
    S = F.conv2d(_as_nchw(V, n, h, w), w2c, padding=1)
    s1, s2 = ops.write_segment(X, c0, _nhwc_rows(S), shift)
    return s1, s2, V


def conv3_cudnn_bwd(ops, dS, U, V, mean2, rstd2, g2, b2, W2, n: int, h: int, w: int):
    """From dS (P x 32, contiguous): dU, dW2, dgamma2, dbeta2."""
    w2c = W2.contiguous(memory_format=torch.channels_last) if V.is_cuda else W2
    dV4, dW2, _ = torch.ops.aten.convolution_backward(
        _as_nchw(dS, n, h, w), _as_nchw(V, n, h, w), w2c, None, [1, 1], [1, 1], [1, 1], False,
        [0, 0], 1, [True, True, False])
    dU, dg2, db2 = ops.bnrelu_bwd(_nhwc_rows(dV4), U, mean2, rstd2, g2, b2)
    return dU, dW2, dg2, db2


class EncoderFn(torch.autograd.Function):
    """The encoder after the stem: (N, C0, H, W) -> features (N, C_fin) = mean over H, W of
    relu(norm5(last block)); with `spec.out == "map"` the whole map (N, C_fin, h, w). Gradients come out
    for the input and for `param_list(spec)`.
    """

    @staticmethod
    def forward(ctx, ops, spec: EncoderSpec, conv_dtype, s, *params):
        act = s.dtype
        n, c0, h, w = s.shape
        dev = s.device
        stat_dt = torch.float64 if act == torch.float64 else torch.float32
        prep = getattr(ops, "prep_weights", None)
        if prep is not None and s.is_cuda:
            prep(spec, params, conv_dtype)     # once per step: weights in fp16, memory zeroed
        saved_blocks = []
        pi = 0
        X = None
        s1 = s2 = None
        for bi, blk in enumerate(spec.blocks):
            P = n * h * w
            if bi == 0:
                X = torch.empty(P, blk.c_total, dtype=act, device=dev)
                X[:, :c0] = _nhwc_rows(s)
                sh = _shift_of(blk, spec.norm5, dev, stat_dt)
                s1 = torch.zeros(blk.c_total, dtype=stat_dt, device=dev)
                s2 = torch.zeros(blk.c_total, dtype=stat_dt, device=dev)
                a1, a2 = ops.colstats(X, 0, c0, sh[:c0])
                s1[:c0], s2[:c0] = a1, a2
            else:
                sh = shn
            layers_saved = []
            for li, lay in enumerate(blk.layers):
                cin = blk.c0 + GROWTH * li
                g1, b1, w1, g2, b2, w2 = params[pi:pi + 6]
                pi += 6
                mean, rstd = _bn_stats(ops, s1[:cin], s2[:cin], sh[:cin], P, lay.norm1)
                ush = _ushift(ops, lay.norm2, stat_dt)
                U, u1, u2 = ops.bnrelu_gemm(X, cin, mean, rstd, g1, b1, _cw(ops, w1, conv_dtype).reshape(
                    BOTTLENECK, cin), ush)
                mean2, rstd2 = _bn_stats(ops, u1, u2, ush, P, lay.norm2)
                a1, a2, saved = ops.conv3_fwd(U, mean2, rstd2, g2, b2, _cw(ops, w2, conv_dtype), X, cin,
                                              sh[cin:cin + GROWTH], n, h, w)
                _store_stats(ops, s1, s2, cin, a1, a2)
                layers_saved.append((U, saved, mean, rstd, mean2, rstd2))
            C = blk.c_total
            if blk.trans is not None:
                gt, bt, wt = params[pi:pi + 3]
                pi += 3
                mean, rstd = _bn_stats(ops, s1, s2, sh, P, blk.trans.norm)
                nxt = spec.blocks[bi + 1]
                cout = wt.shape[0]
                if cout != nxt.c0:
                    raise ValueError(f"transition {bi}: {cout} channels, the next block wants {nxt.c0}")
                Xn = torch.empty(P // 4, nxt.c_total, dtype=act, device=dev)
                shn = _shift_of(nxt, spec.norm5, dev, stat_dt)
                n1 = torch.zeros(nxt.c_total, dtype=stat_dt, device=dev)
                n2 = torch.zeros(nxt.c_total, dtype=stat_dt, device=dev)
                a1, a2 = ops.trans_fwd(X, C, mean, rstd, gt, bt, _cw(ops, wt, conv_dtype).reshape(cout, C),
                                       n, h, w, Xn, shn[:cout])
                n1[:cout], n2[:cout] = a1, a2
                saved_blocks.append((X, layers_saved, (mean, rstd), (n, h, w)))
                X, s1, s2 = Xn, n1, n2
                h, w = h // 2, w // 2
            else:
                g5, b5 = params[pi:pi + 2]
                pi += 2
                mean, rstd = _bn_stats(ops, s1, s2, sh, P, spec.norm5)
                if spec.out == "map":
                    feats = _map_fwd(X, C, mean, rstd, g5, b5, n, h, w)
                else:
                    feats = ops.gap_fwd(X, C, mean, rstd, g5, b5, n, h * w)
                saved_blocks.append((X, layers_saved, (mean, rstd), (n, h, w)))
        if pi != len(params):
            raise ValueError(f"fastdense: used {pi} parameters out of {len(params)}")
        ctx.ops, ctx.spec, ctx.conv_dtype = ops, spec, conv_dtype
        ctx.saved_blocks = saved_blocks
        ctx.in_shape = (n, c0) + tuple(s.shape[2:])
        ctx.act = act
        ctx.save_for_backward(*params)
        return feats.to(act) if act != torch.float64 else feats

    @staticmethod
    def backward(ctx, dfeat):
        ops, spec, conv_dtype = ctx.ops, ctx.spec, ctx.conv_dtype
        params = ctx.saved_tensors
        grads = [None] * len(params)
        # the parameter indices per block
        starts, pi = [], 0
        for blk in spec.blocks:
            starts.append(pi)
            pi += 6 * len(blk.layers) + (3 if blk.trans is not None else 0)
        G = None
        dT = None
        for bi in range(len(spec.blocks) - 1, -1, -1):
            blk = spec.blocks[bi]
            X, layers_saved, (mean, rstd), (n, h, w) = ctx.saved_blocks[bi]
            C = blk.c_total
            P = X.shape[0]
            base = starts[bi]
            # G: the a gm parts of the buffer gradients; A, B: the BN corrections per channel, applied when the
            # gradient of a channel is complete (`RefOps`, `dcols`).
            if blk.trans is None:
                g5, b5 = params[pi:pi + 2]
                if spec.out == "map":
                    G, dg, db = _map_bwd(dfeat, X, C, mean, rstd, g5, b5)
                else:
                    G, dg, db = ops.gap_bwd(dfeat, X, C, mean, rstd, g5, b5, n, h * w)
                grads[pi], grads[pi + 1] = dg.to(g5.dtype), db.to(b5.dtype)
                A, B = split_coeffs(mean, rstd, g5, dg, db, P)
            else:
                ti = base + 6 * len(blk.layers)
                gt, bt, wt = params[ti:ti + 3]
                cout = wt.shape[0]
                G, dW, dg, db = ops.trans_bwd(dT, X, C, mean, rstd, gt, bt,
                                              _cw(ops, wt, conv_dtype).reshape(cout, C), n, h, w)
                grads[ti], grads[ti + 1] = dg.to(gt.dtype), db.to(bt.dtype)
                grads[ti + 2] = dW.reshape(wt.shape).to(wt.dtype)
                A, B = split_coeffs(mean, rstd, gt, dg, db, P)
            for li in range(len(blk.layers) - 1, -1, -1):
                cin = blk.c0 + GROWTH * li
                k = base + 6 * li
                g1, b1, w1, g2, b2, w2 = params[k:k + 6]
                U, saved, m1, r1, m2, r2 = layers_saved[li]
                dS = ops.dcols(G, X, cin, cin + GROWTH, A[cin:cin + GROWTH], B[cin:cin + GROWTH])
                dU, dW2, dg2, db2 = ops.conv3_bwd(dS, U, saved, m2, r2, g2, b2, _cw(ops, w2, conv_dtype),
                                                  n, h, w)
                dW1, dg1, db1 = ops.bnrelu_gemm_bwd(dU, _cw(ops, w1, conv_dtype).reshape(BOTTLENECK, cin),
                                                    X, cin, m1, r1, g1, b1, G)
                _bn_coeffs_add(ops, m1, r1, g1, dg1, db1, P, A[:cin], B[:cin])
                grads[k], grads[k + 1] = dg1.to(g1.dtype), db1.to(b1.dtype)
                grads[k + 2] = dW1.reshape(w1.shape).to(w1.dtype)
                grads[k + 3], grads[k + 4] = dg2.to(g2.dtype), db2.to(b2.dtype)
                grads[k + 5] = dW2.to(w2.dtype)
            dT = ops.dcols(G, X, 0, blk.c0, A[:blk.c0], B[:blk.c0])
            G = None
            pi = base
        n, c0, h, w = ctx.in_shape
        ds = _as_nchw(dT, n, h, w).to(ctx.act).contiguous(memory_format=torch.channels_last)
        ctx.saved_blocks = None
        return (None, None, None, ds, *grads)


def _ushift(ops, bn: nn.Module, dt) -> torch.Tensor:
    """The constant subtracted before the sums of the conv1 output (the norm2 statistics): the norm2
    running mean before this step. Ops that read the constant before updating the running averages in
    the same kernel (`shift_alias_ok`) use it in place, without a copy; the others want a copy.
    """
    rm = bn.running_mean.detach()
    if getattr(ops, "shift_alias_ok", False) and rm.dtype == dt:
        return rm
    return rm.to(dt).clone()


def _shift_of(blk: BlockSpec, norm5: nn.Module, dev, dt) -> torch.Tensor:
    """The constant subtracted before the sums of the channels of a block buffer: the running mean of the
    BN that reads them all (transition or norm5), copied before this step updates it. Precision only
    (E[x^2] - E[x]^2 in float32): the function does not change.
    """
    bn = blk.trans.norm if blk.trans is not None else norm5
    if bn is None or bn.running_mean is None or bn.running_mean.shape[0] != blk.c_total:
        return torch.zeros(blk.c_total, dtype=dt, device=dev)
    return bn.running_mean.detach().to(device=dev, dtype=dt).clone()


# ---------------------------------------------------------------------------------------------
# the module
# ---------------------------------------------------------------------------------------------

class StemFn(torch.autograd.Function):
    """norm0 (BN in training + ReLU) and pool0 (3x3 max-pool, stride 2, padding 1) of the stem in two
    kernels (`ops.stem_fwd` / `ops.stem_bwd`): in the forward the statistics in one pass, then BN +
    ReLU + max with the tap of the maximum in one byte (PyTorch saves int64 indices and its NHWC
    max-pool backward cost ~14 ms per step on the T4); in the backward gm gathered from the windows and
    the BN in two passes. Same function as timm's chain.
    """

    @staticmethod
    def forward(ctx, ops, norm0, y0, g0, b0):
        n, C, H, W = y0.shape
        Y = _nhwc_rows(y0)
        sh = norm0.running_mean.detach().to(torch.float32).clone()
        s1, s2 = ops.colstats(Y, 0, C, sh)
        mean, rstd = _bn_stats(ops, s1, s2, sh, Y.shape[0], norm0)
        # `stem_fwd` -> (S, *aux): the tap of the maximum and, for `CudaOps`, Y at the tap of the maximum
        S, *aux = ops.stem_fwd(Y, n, H, W, mean, rstd, g0, b0)
        Ho, Wo = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        ctx.save_for_backward(Y, S, mean, rstd, g0, b0, *aux)
        ctx.ops, ctx.dims = ops, (n, H, W, Ho, Wo)
        return _as_nchw(S, n, Ho, Wo)

    @staticmethod
    def backward(ctx, dS4):
        Y, S, mean, rstd, g0, b0, *aux = ctx.saved_tensors
        n, H, W, Ho, Wo = ctx.dims
        dS = _nhwc_rows(dS4.to(Y.dtype))
        dY, dg, db = ctx.ops.stem_bwd(Y, n, H, W, mean, rstd, g0, b0, dS, S, *aux)
        return None, None, _as_nchw(dY, n, H, W), dg.to(g0.dtype), db.to(b0.dtype)


def _stem_eager(f: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The eager stem: in timm norm0 is BatchNormAct2d (ReLU inside), in the hand-written DenseNet of
    `knee.models.densenet` (inference) there is a separate `relu0`.
    """
    s = f.norm0(f.conv0(x))
    relu0 = getattr(f, "relu0", None)
    return f.pool0(relu0(s) if relu0 is not None else s)


def _stem_fusable(f: nn.Module) -> bool:
    """The expected timm stem: norm0 = BatchNormAct2d with ReLU and no dropout, pool0 = 3x3 / 2 / 1
    max-pool, 64 channels.
    """
    n0, p0 = f.norm0, getattr(f, "pool0", None)
    return (type(n0).__name__ == "BatchNormAct2d" and isinstance(getattr(n0, "act", None), nn.ReLU)
            and isinstance(getattr(n0, "drop", nn.Identity()), nn.Identity)
            and n0.track_running_stats and n0.affine and n0.num_features == 64
            and isinstance(p0, nn.MaxPool2d) and p0.kernel_size in (3, (3, 3))
            and p0.stride in (2, (2, 2)) and p0.padding in (1, (1, 1))
            and p0.dilation in (1, (1, 1)) and not p0.ceil_mode)


class _Ctx:
    """The `EncoderFn` `ctx` outside autograd (forward and backward called by hand)."""

    def save_for_backward(self, *t):
        self.saved_tensors = t


class EncoderGraphs:
    """`EncoderFn` forward and backward captured in two CUDA graphs with the same memory pool (the
    activations saved by the forward stay at the same addresses for the backward).

    Why: per layer ~45 launches between statistics, coefficients, GEMMs and convs; in the small layers
    of blocks 3-4 Python launches more slowly than the GPU executes. The graph replays everything
    without Python.

    The first `warmup` steps are eager (Triton/NVRTC compilation, cuDNN algorithm selection,
    allocations), then the capture happens: fixed shapes (in training every step has batch x k_train
    windows); a different shape or a capture error -> eager from then on, with a message. The graph
    reads `s` from a fixed buffer (copy) and returns copies of outputs and gradients: no aliasing
    between one step and the next. One forward per backward (the training step).
    """

    def __init__(self, ops, spec: EncoderSpec, warmup: int = 2, log=print):
        self.ops, self.spec, self.warmup, self.log = ops, spec, int(warmup), log
        self.steps = 0
        self.state = "eager"            # eager | graph | off
        self.key = None
        self.pending = None             # the eager ctx of the step, or "graph"
        self.g_fwd = self.g_bwd = None

    def forward(self, s, params, cd):
        key = (tuple(s.shape), tuple(s.stride()), s.dtype, s.device, cd)
        if self.state == "graph" and key != self.key:
            self.log(f"fastdense: CUDA graph off, shape {key} differs from {self.key}")
            self.state = "off"
        if self.state == "graph":
            self.static_s.copy_(s)
            self.g_fwd.replay()
            self.pending = "graph"
            return self.static_out.clone()
        self.steps += 1
        if self.state == "eager" and self.steps > self.warmup:
            try:
                return self._capture_fwd(s, params, cd, key)
            except Exception as exc:                               # noqa: BLE001
                self.log(f"fastdense: forward capture failed ({type(exc).__name__}: "
                         f"{str(exc)[:300]}), staying eager")
                self.state, self.g_fwd = "off", None
                torch.cuda.synchronize()
        c = _Ctx()
        out = EncoderFn.forward(c, self.ops, self.spec, cd, s, *params)
        self.pending = c
        return out

    def _capture_fwd(self, s, params, cd, key):
        self.key = key
        self.static_s = s.detach().clone()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        c = _Ctx()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            out = EncoderFn.forward(c, self.ops, self.spec, cd, self.static_s, *params)
        self.g_fwd, self.c_fwd, self.static_out = g, c, out
        self.keep = c.saved_blocks          # lives on for the backward (even when the ctx clears it)
        self.state = "graph"
        g.replay()                          # capturing does not execute: this is the real step
        self.pending = "graph"
        self.log(f"fastdense: encoder in a CUDA graph (forward) from step {self.steps}")
        return out.clone()

    def backward(self, dfeat):
        if self.pending == "graph":
            self.pending = None
            if self.g_bwd is None:
                try:
                    self._capture_bwd(dfeat)
                except Exception as exc:                           # noqa: BLE001
                    self.log(f"fastdense: backward capture failed ({type(exc).__name__}: "
                             f"{str(exc)[:300]}), staying eager")
                    torch.cuda.synchronize()
                    self.state = "off"
                    self.c_fwd.saved_blocks = self.keep
                    res = EncoderFn.backward(self.c_fwd, dfeat)
                    return res[3], list(res[4:])
            self.static_dfeat.copy_(dfeat)
            self.g_bwd.replay()
            res = self.static_bwd
            return res[3].clone(), [t.clone() if t is not None else None for t in res[4:]]
        c, self.pending = self.pending, None
        res = EncoderFn.backward(c, dfeat)
        return res[3], list(res[4:])

    def _capture_bwd(self, dfeat):
        self.static_dfeat = dfeat.detach().clone()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.g_fwd.pool(), capture_error_mode="thread_local"):
            res = EncoderFn.backward(self.c_fwd, self.static_dfeat)
        self.c_fwd.saved_blocks = self.keep  # EncoderFn.backward clears it: the references remain
        self.g_bwd, self.static_bwd = g, res
        self.log("fastdense: encoder in a CUDA graph (backward)")


class GraphFn(torch.autograd.Function):
    """The encoder after the stem through `EncoderGraphs` (CUDA graphs, or eager in the first steps)."""

    @staticmethod
    def forward(ctx, eg, cd, s, *params):
        ctx.eg = eg
        return eg.forward(s, params, cd)

    @staticmethod
    def backward(ctx, dfeat):
        ds, grads = ctx.eg.backward(dfeat.contiguous())
        return (None, None, ds, *grads)


def fast_forward(enc: nn.Module, spec: EncoderSpec, ops, x: torch.Tensor,
                 conv_dtype: torch.dtype | None = None) -> torch.Tensor:
    """The training forward of the encoder: stem (conv0 eager under the caller's autocast; norm0 + pool0
    fused with `StemFn` if the ops support it, otherwise eager) + `EncoderFn`. Output (N, C_fin) like
    timm's `enc(x)` with `num_classes=0` (with `global_pool=""` the map, `spec.out`).
    """
    f = enc.features
    params = param_list(spec)
    fused_stem = getattr(ops, "stem_fused", False) and _stem_fusable(f)
    if fused_stem:
        s = f.conv0(x)
    else:
        s = _stem_eager(f, x)
    dt = s.dtype
    if s.is_cuda and torch.is_autocast_enabled():
        dt = (torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype")
              else torch.get_autocast_gpu_dtype())
        s = s.to(dt)
    if fused_stem:
        if s.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                s = StemFn.apply(ops, f.norm0, s, f.norm0.weight, f.norm0.bias)
        else:
            s = StemFn.apply(ops, f.norm0, s, f.norm0.weight, f.norm0.bias)
    cd = conv_dtype or dt
    if not s.is_cuda:
        return EncoderFn.apply(ops, spec, cd, s, *params)
    eg = getattr(enc, "_fastdense_graphs", None)
    with torch.autocast(device_type="cuda", enabled=False):
        if eg is not None and torch.is_grad_enabled():
            return GraphFn.apply(eg, cd, s, *params)
        return EncoderFn.apply(ops, spec, cd, s, *params)


@torch.no_grad()
def eval_forward(enc: nn.Module, spec: EncoderSpec, ops, x: torch.Tensor,
                 conv_dtype: torch.dtype | None = None) -> torch.Tensor:
    """The EVAL forward with the fastdense kernels: BatchNorm with the running averages (a = gamma rstd,
    c = beta - mean a, as `F.batch_norm` in eval), no batch statistics, no autograd. Same output as
    timm's `enc(x)` in eval.
    """
    f = enc.features
    s = _stem_eager(f, x)
    dt = s.dtype
    if s.is_cuda and torch.is_autocast_enabled():
        dt = (torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype")
              else torch.get_autocast_gpu_dtype())
        s = s.to(dt)
    cd = conv_dtype or dt
    if s.is_cuda:
        with torch.autocast(device_type="cuda", enabled=False):
            return _eval_blocks(ops, spec, cd, s)
    return _eval_blocks(ops, spec, cd, s)


def _running(bn: nn.Module, dt):
    mean = bn.running_mean.detach().to(dt)
    rstd = torch.rsqrt(bn.running_var.detach().to(dt) + bn.eps)
    return mean, rstd


# The eval constants cached (`_eval_consts`); False recomputes them at every call (for A/B probes).
EVAL_CACHE = True


def _eval_key(spec: EncoderSpec, conv_dtype, st, dev):
    """What determines the eval constants: pointer and version counter of every parameter and buffer
    involved (an optimizer step, `load_state_dict`, `.half()` or `.to()` change them), dtype and device.
    None (no caching) for inference tensors, which have no version counter.
    """
    ts = []

    def bn(m):
        ts.extend((m.weight, m.bias, m.running_mean, m.running_var))
    for blk in spec.blocks:
        for lay in blk.layers:
            bn(lay.norm1)
            bn(lay.norm2)
            ts.extend((lay.conv1.weight, lay.conv2.weight))
        if blk.trans is not None:
            bn(blk.trans.norm)
            ts.append(blk.trans.conv.weight)
    bn(spec.norm5)
    try:
        return (str(dev), conv_dtype, st, tuple((t.data_ptr(), t._version) for t in ts))
    except RuntimeError:
        return None


def _eval_consts(ops, spec: EncoderSpec, conv_dtype, st, dev) -> dict:
    """The eval constants: running averages and rstd in `st`, conv weights in `conv_dtype` (the 3x3 in the
    kernels' layout if the ops want it, `conv3_eval_weight`) and, with `CudaOps`, the a, c tables of the
    BatchNorms (`bn_table`, looked up by `_ac` instead of computed). Computed once per encoder and
    recomputed only if the key changes (`_eval_key`): otherwise every call launched ~20 tiny operations
    per layer around the two real kernels (~1,600 for the 169), CUDA's launch queue filled up and the
    thread serving both GPUs serialised them. Bit-identical values.
    """
    key = _eval_key(spec, conv_dtype, st, dev) if EVAL_CACHE else None
    hit = spec.__dict__.get("_eval")
    if key is not None and hit is not None and hit[0] == key:
        K = hit[1]
    else:
        tb = getattr(ops, "bn_table", None)
        w3 = getattr(ops, "conv3_eval_weight", None)
        tabs = []

        def bn(m):
            mean, rstd = _running(m, st)
            if tb is None:
                return mean, rstd
            g, b = m.weight.detach(), m.bias.detach()
            tab = tb(mean, rstd, g, b)
            tabs.append((tab, g, b))
            return tab[2], tab[3]
        blocks = []
        for blk in spec.blocks:
            layers = []
            for li, lay in enumerate(blk.layers):
                cin = blk.c0 + GROWTH * li
                m1, r1 = bn(lay.norm1)
                W1 = lay.conv1.weight.detach().to(conv_dtype).reshape(BOTTLENECK, cin)
                m2, r2 = bn(lay.norm2)
                W2 = lay.conv2.weight.detach().to(conv_dtype)
                layers.append((m1, r1, W1, m2, r2, w3(W2) if w3 is not None else W2))
            tr = None
            if blk.trans is not None:
                mt, rt = _running(blk.trans.norm, st)
                cout = blk.trans.conv.weight.shape[0]
                tr = (mt, rt, blk.trans.conv.weight.detach().to(conv_dtype).reshape(cout, blk.c_total))
            zeros = torch.zeros(max(blk.c_total, BOTTLENECK), dtype=st, device=dev)
            blocks.append((layers, tr, zeros))
        K = {"blocks": blocks, "norm5": _running(spec.norm5, st), "tabs": tabs}
        if key is not None:
            spec.__dict__["_eval"] = (key, K)
        else:
            spec.__dict__.pop("_eval", None)
    reg = getattr(ops, "remember_bn", None)
    if reg is not None:                 # CudaOps memory is zeroed at every training step
        for tab, g, b in K["tabs"]:
            reg(tab, g, b)
    return K


def _eval_blocks(ops, spec: EncoderSpec, conv_dtype, s: torch.Tensor) -> torch.Tensor:
    act = s.dtype
    n, c0, h, w = s.shape
    dev = s.device
    st = torch.float64 if act == torch.float64 else torch.float32
    K = _eval_consts(ops, spec, conv_dtype, st, dev)
    X = None
    for bi, blk in enumerate(spec.blocks):
        layers, trc, zeros = K["blocks"][bi]
        P = n * h * w
        if bi == 0:
            X = torch.empty(P, blk.c_total, dtype=act, device=dev)
            X[:, :c0] = _nhwc_rows(s)
        for li, lay in enumerate(blk.layers):
            cin = blk.c0 + GROWTH * li
            m1, r1, W1, m2, r2, W2 = layers[li]
            U, _, _ = ops.bnrelu_gemm(X, cin, m1, r1, lay.norm1.weight.detach(), lay.norm1.bias.detach(),
                                      W1, zeros[:BOTTLENECK])
            ops.conv3_fwd(U, m2, r2, lay.norm2.weight.detach(), lay.norm2.bias.detach(),
                          W2, X, cin, zeros[:GROWTH], n, h, w, save=False)
        C = blk.c_total
        if blk.trans is not None:
            tr = blk.trans
            mt, rt, Wt = trc
            nxt = spec.blocks[bi + 1]
            Xn = torch.empty(P // 4, nxt.c_total, dtype=act, device=dev)
            ops.trans_fwd(X, C, mt, rt, tr.norm.weight.detach(), tr.norm.bias.detach(), Wt, n, h, w,
                          Xn, zeros[:Wt.shape[0]])
            X = Xn
            h, w = h // 2, w // 2
        else:
            m5, r5 = K["norm5"]
            if spec.out == "map":
                out = _map_fwd(X, C, m5, r5, spec.norm5.weight.detach(), spec.norm5.bias.detach(),
                               n, h, w)
            else:
                out = ops.gap_fwd(X, C, m5, r5, spec.norm5.weight.detach(), spec.norm5.bias.detach(),
                                  n, h * w)
            return out.to(act) if act != torch.float64 else out
    raise ValueError("fastdense: encoder without blocks")


def _out_mode(enc: nn.Module) -> str:
    """What `enc(x)` returns: "gap" (global mean) or "map" (the map). timm says it through `global_pool`
    ("" = no pooling), `knee.models.densenet` through `spatial_pool`.
    """
    gp = getattr(enc, "global_pool", None)
    if gp is not None:
        kind = getattr(gp, "pool_type", "avg")
        if kind in ("avg", "fast"):
            return "gap"
        if kind == "":
            return "map"
        raise ValueError(f"fastdense: global pooling {kind}, expected avg or none")
    kind = getattr(enc, "spatial_pool", "avg")
    if kind not in ("avg", "map"):
        raise ValueError(f"fastdense: spatial_pool {kind}, expected avg or map")
    return "gap" if kind == "avg" else "map"


def install(enc: nn.Module, ops=None, conv_dtype: torch.dtype | None = None,
            fast_eval: bool = False, graphs: bool = False, log=print) -> nn.Module:
    """Replaces the `forward` OF THE INSTANCE `enc` (timm DenseNet, `num_classes=0`): in training
    `fast_forward`, in eval timm's forward (or, with `fast_eval`, `eval_forward`: the same kernels with
    the running averages). The module stays the same object, with the same parameters, buffers and
    names: state_dict, checkpoints, `.to()`, EMA shadows and inference do not change. Call it after any
    `deepcopy` (the copy would carry the closure over the original).
    """
    spec = encoder_spec(enc)
    if not isinstance(getattr(spec.norm5, "drop", nn.Identity()), nn.Identity):
        raise ValueError("fastdense: norm5 with dropout")
    spec.out = _out_mode(enc)
    if spec.out == "map" and graphs:
        raise ValueError("fastdense: CUDA graphs with the map output are untested")
    ops = ops or RefOps()
    orig = type(enc).forward

    def forward(x):
        if not enc.training:
            return eval_forward(enc, spec, ops, x, conv_dtype) if fast_eval else orig(enc, x)
        # A training step makes the cached eval constants (`_eval_consts`) stale even when the `_version`
        # counters do not move: the BatchNorm running averages are written by the CUDA kernels through
        # pointers, and fused AdamW updates the weights without touching them. Without this, with `fast_eval`
        # in validation, every epoch validated after the first (and the final prediction) used stale dense
        # blocks.
        spec.__dict__.pop("_eval", None)
        return fast_forward(enc, spec, ops, x, conv_dtype)

    enc.forward = forward
    enc._fastdense = {"ops": getattr(ops, "name", type(ops).__name__), "spec": spec,
                      "fast_eval": bool(fast_eval), "graphs": bool(graphs)}
    # `graphs`: in training the encoder after the stem in two CUDA graphs (`EncoderGraphs`)
    enc._fastdense_graphs = EncoderGraphs(ops, spec, log=log) if graphs else None
    return enc


def _attach_problems(model: nn.Module) -> list[str]:
    """What prevents `fastdense` on a `KneeBaseline` (empty = allowed)."""
    bad = []
    if not type(model.encoder).__name__.startswith("DenseNet"):
        bad.append(f"encoder {type(model.encoder).__name__}")
    if getattr(model, "encoder_scope", "shared") != "shared":
        bad.append(f"encoder_scope {model.encoder_scope}")
    if getattr(model, "spatial_pool", "avg") != "avg":
        bad.append(f"spatial_pool {model.spatial_pool}")
    if getattr(model, "spatial_attn", None) is not None:
        bad.append("spatial_attn")
    if getattr(model, "slice_attn_layers", 0):
        bad.append("slice_attn")
    if getattr(model, "grad_checkpointing", False):
        bad.append("grad_checkpointing")
    return bad


def attach_eval(model: nn.Module, log=print) -> bool:
    """The encoder of a `KneeBaseline` at INFERENCE with the fastdense kernels (`fast_eval`: running
    averages, `CudaOps` on the GPU). Micro-benchmark (DenseNet169, 96 windows at 288 px, T4): eager 297 ms,
    eager with `fold_bn` 294, fastdense 132, relative difference 5.5e-4. Apply after
    `.half()`/`.to(device)` and BEFORE `fold_bn` (which removes norm2). False if the model is not
    suitable (nothing installed).
    """
    bad = _attach_problems(model)
    if bad:
        log(f"fastdense in eval: not valid with {bad}, staying eager")
        return False
    if next(model.encoder.parameters()).is_cuda:
        from .fastdense_cuda import CudaOps
        ops = CudaOps()
    else:
        ops = RefOps()
    install(model.encoder, ops, fast_eval=True)
    return True


def attach(model: nn.Module, log=print, ops=None, conv: str | None = None,
           graphs: bool | None = None, fast_eval: bool = False) -> None:
    """`train.speed.fastdense`: the fast forward on the encoder of a `KneeBaseline`.

    Valid for the shared DenseNet encoder with the usual global mean, or with the whole map for the
    intra-window head: no per-plane encoders, GeM, spatial attention, slice attention inside the
    encoder or gradient checkpointing (they cut the encoder or read the map in ways that do not exist
    here). `graphs` (`train.speed.graphs`): the encoder in CUDA graphs from the third step; `fast_eval`
    (`train.speed.fast_eval`): in eval the fastdense kernels with the running averages.
    """
    bad = _attach_problems(model)
    if bad:
        raise ValueError(f"train.speed.fastdense is not valid with {bad}")
    if ops is None:
        if next(model.encoder.parameters()).is_cuda:
            # `cublas` (the default): GEMMs on cuBLAS, 3x3 conv on cuDNN, Triton only for elementwise parts. The
            # fused Triton kernels (`triton`, `triton_v`, `cudnn`) fall back to FMA on the T4 (sm_75: no mma in
            # Triton), 20 times slower than eager
            from .fastdense_triton import CublasOps, TritonOps
            if conv == "cuda":                  # conv1 with the fused CUDA kernels (`knee.fastdense_cuda`)
                from .fastdense_cuda import CudaOps
                ops = CudaOps()
            else:
                ops = CublasOps() if conv in (None, "cublas") else TritonOps(conv=conv)
        else:
            ops = RefOps()                  # on the CPU (tests): the oracle, same scheme
    # CUDA graphs off by default: -13 ms on one T4 and 0 on the other (at the power cap the removed gaps
    # lengthen the kernels: the energy of the step counts, not the gaps)
    graphs = bool(graphs)
    install(model.encoder, ops, graphs=graphs, fast_eval=bool(fast_eval), log=log)
    log(f"  fastdense: DenseNet encoder trained without concatenations ({getattr(ops, 'name', ops)}"
        f"{', CUDA graphs' if graphs else ''}), timm state_dict, validation "
        + ("with the fastdense kernels (fast_eval)" if fast_eval else "with timm"))


def uninstall(enc: nn.Module) -> nn.Module:
    """Removes `install`: the forward goes back to the class's."""
    if "forward" in enc.__dict__:
        del enc.__dict__["forward"]
    enc.__dict__.pop("_fastdense", None)
    enc.__dict__.pop("_fastdense_graphs", None)
    return enc


__all__ = ["install", "uninstall", "attach", "attach_eval", "fast_forward", "eval_forward", "EncoderFn", "StemFn", "EncoderGraphs", "GraphFn", "RefOps", "split_coeffs", "encoder_spec",
           "param_list", "finalize_stats", "update_running"]
