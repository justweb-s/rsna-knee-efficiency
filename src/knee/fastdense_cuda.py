"""`CudaOps`: the conv1 of every `knee.fastdense` layer with the CUDA kernels of
`fastdense_kernels.cu` (fp16 tensor cores, BatchNorm + ReLU in the prologue), the rest as
`CublasOps`. This is the fast DenseNet training path used for the final models.

Why: with `CublasOps` the relu(bn(x)) of the conv1 input is written and re-read by cuBLAS, and in
the backward dY and the V of dW do the same: ~20 bytes per input channel and pixel against ~8 for
the fused scheme. The kernels are compiled with NVRTC on first call (`knee.cudart`, like
`knee.dense8s`), for the GPU's architecture, and launched on PyTorch's current stream.

- forward (`c1f_<BM>`): U = relu(X a + c) W^T and the per-column sums of U (norm2), one partial
  per block of rows;
- backward (`c1b*`): in slices of 64 channels and of rows, dY = dU W, G += a gm, the BN
  reductions and dW += dU^T relu(X a + c): X read once, dY and y only in shared memory. Variants:
  `_p` (x and G of the next row block in flight during the current one), `_b` (bottleneck BN2
  backward fused: `bnrelu_bwd` only does the reductions and returns a `LazyDU`, the kernel
  computes dU from dV and U while loading it: no dU in global memory, no apply pass).
Shapes the kernels do not cover (fp32, cin not a multiple of 32, CPU) go to `CublasOps`.
"""
from __future__ import annotations

import ctypes
import threading
from pathlib import Path

import numpy as np
import torch
import triton

from . import cudart
from .fastdense_triton import CublasOps, _bn_bwd_apply, _f32

KERNELS = Path(__file__).with_name("fastdense_kernels.cu")
_U64, _I = ctypes.c_uint64, ctypes.c_int
_LOCK = threading.Lock()
_MODS: dict = {}
SMEM_C1F = {64: 24576, 128: 32768}
SMEM_C1B = 2 * 128 * 64 * 2 + 64 * 64 * 2 + 4 * 64 * 4


class C1F(ctypes.Structure):
    _fields_ = [("X", _U64), ("W", _U64), ("A", _U64), ("C", _U64), ("sh", _U64), ("U", _U64),
                ("part", _U64), ("rows", _I), ("cin", _I), ("sx", _I)]


class C1B(ctypes.Structure):
    _fields_ = [("dU", _U64), ("X", _U64), ("W", _U64), ("A", _U64), ("C", _U64), ("MU", _U64),
                ("RS", _U64), ("G", _U64), ("partR", _U64), ("partW", _U64), ("rows", _I),
                ("cin", _I), ("sx", _I), ("sg", _I), ("split", _I), ("dV", _U64), ("U2", _U64),
                ("bn2", _U64)]


SMEM_C1B_BN2 = SMEM_C1B + 6 * 128 * 4
SMEM_C1B2 = 128 * 64 * 2 + 32 * 128 * 2 + 32 * 64 * 2 + 4 * 64 * 4
SMEM_C1B3 = 128 * 64 * 2 + 64 * 128 * 2 + 3 * 64 * 64 * 2 + 4 * 64 * 4    # > 48 KB: must be enabled
C3_WMAX = 72                    # as in fastdense_kernels.cu: beyond this, the 3x3 conv stays on cuDNN


class C3F(ctypes.Structure):
    _fields_ = [("U", _U64), ("Wk", _U64), ("A", _U64), ("C", _U64), ("sh", _U64), ("V", _U64),
                ("X", _U64), ("part", _U64), ("rows", _I), ("H", _I), ("W", _I), ("sx", _I),
                ("c0", _I)]


class C3W(ctypes.Structure):
    _fields_ = [("dS", _U64), ("V", _U64), ("partW", _U64), ("rows", _I), ("H", _I), ("W", _I),
                ("split", _I)]


class STB(ctypes.Structure):
    _fields_ = [("Y", _U64), ("dS", _U64), ("S", _U64), ("YM", _U64), ("AM", _U64), ("tab", _U64),
                ("dY", _U64), ("part", _U64), ("M2", _I), ("H", _I), ("W", _I), ("Ho", _I),
                ("Wo", _I)]


class C3D(ctypes.Structure):
    _fields_ = [("dS", _U64), ("Wd", _U64), ("U", _U64), ("tab", _U64), ("G", _U64),
                ("part", _U64), ("rows", _I), ("H", _I), ("W", _I), ("split", _I)]


class BNA(ctypes.Structure):
    _fields_ = [("G", _U64), ("U", _U64), ("tab", _U64), ("dU", _U64), ("rows", _I)]


class WP(ctypes.Structure):
    _fields_ = [("jobs", _U64), ("total", ctypes.c_int64), ("njobs", _I)]


class CS(ctypes.Structure):
    _fields_ = [("part", _U64), ("part2", _U64), ("scr", _U64), ("cnt", _U64), ("sh", _U64), ("g", _U64),
                ("b", _U64), ("rm", _U64), ("rv", _U64), ("nbt", _U64), ("tab", _U64),
                ("out1", _U64), ("out2", _U64), ("ab", _U64), ("R", _I), ("ld", _I), ("lo1", _I),
                ("lo2", _I), ("C", _I), ("mode", _I), ("upd_nbt", _I), ("inv_n", ctypes.c_float),
                ("n_corr", ctypes.c_float), ("eps", ctypes.c_float), ("mom", ctypes.c_float)]


class WS(ctypes.Structure):
    _fields_ = [("partW", _U64), ("dW", _U64), ("S", _I)]


# the wprep job table (struct WJob of fastdense_kernels.cu, 72 bytes)
WJOB = np.dtype([("src", "<u8"), ("dst", "<u8"), ("s0", "<i8"), ("s1", "<i8"), ("s2", "<i8"),
                 ("s3", "<i8"), ("start", "<i8"), ("O", "<i4"), ("C", "<i4"), ("T", "<i4"),
                 ("layout", "<i4")])
assert WJOB.itemsize == 72


class Partial:
    """Column sums not yet reduced (`fuse_small`): part[:, lo:lo+n] summed over the rows. Produced by
    `bnrelu_gemm` and `conv3_fwd`; consumed by `bn_stats` and `store_stats`, which do the sum inside
    their kernel. `value()` for whoever wants the tensor.
    """

    __slots__ = ("part", "lo", "n")

    def __init__(self, part: torch.Tensor, lo: int, n: int):
        self.part, self.lo, self.n = part, int(lo), int(n)

    @property
    def shape(self):
        return (self.n,)

    def value(self) -> torch.Tensor:
        return self.part[:, self.lo:self.lo + self.n].sum(0)


def materialize(x):
    """The tensor of the sums, whether `x` is a `Partial` or already a tensor."""
    return x.value() if isinstance(x, Partial) else x


def smem_c3d(w: int) -> int:
    """Shared memory of c3d (blocks of 256 pixels): slice weights, dS halo + zeros, slice U, a c mu rs,
    masks (> 48 KB: must be enabled).
    """
    return 9 * 32 * 32 * 2 + (256 + 2 * w + 3) * 64 + 256 * 64 + 128 * 4 + 256 * 2


def smem_c3f(bm: int, w: int) -> int:
    """Shared memory of c3f_<bm>: halo + row of zeros, slice weights, a and c, masks."""
    return (bm + 2 * w + 3) * 64 + 9 * 32 * 32 * 2 + 256 * 4 + bm * 2


def smem_c3w(w: int) -> int:
    """Shared memory of c3w (blocks of 128 pixels): dS halo + zeros, V, masks."""
    return (128 + 2 * w + 3) * 64 + 128 * 64 + 128 * 2


class LazyDU:
    """dU of the bottleneck BN2 not yet applied: dV, U and the coefficients. `CudaOps` passes it from
    `bnrelu_bwd` to `bnrelu_gemm_bwd` (`EncoderFn` does not touch it in between); `materialize` writes
    it (shapes the kernel does not cover).
    """

    def __init__(self, dV, U, mean, rstd, g, b, r1, r2):
        self.dV, self.U = dV, U
        self.mean, self.rstd, self.g, self.b = _f32(mean), _f32(rstd), _f32(g), _f32(b)
        self.r1, self.r2 = r1, r2                  # sums (not divided by the rows)
        self.shape, self.dtype = U.shape, U.dtype

    def table(self) -> torch.Tensor:
        """[6][128]: a2, c2, mu2, rs2, k1 = R1/P, k2 = R2/P (the kernel's table)."""
        a, c = _ac(self.mean, self.rstd, self.g, self.b)
        P = float(self.U.shape[0])
        return torch.stack([a, c, self.mean, self.rstd, self.r1.float() / P,
                            self.r2.float() / P]).contiguous()

    def materialize(self, bm: int = 64) -> torch.Tensor:
        M, N = self.U.shape
        dU = torch.empty_like(self.U)
        BN = min(128, triton.next_power_of_2(N))
        _bn_bwd_apply[(triton.cdiv(M, bm), triton.cdiv(N, BN))](
            self.dV, self.U, dU, M, N, self.mean, self.rstd, self.g, self.b, self.r1, self.r2,
            BM=bm, BN=BN)
        return dU


def _function(dev: torch.device, name: str, smem: int = 0):
    """Function `name` of the module compiled for GPU `dev` (one compilation per GPU); with `smem`
    > 48 KB it enables dynamic shared memory (once per function).
    """
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    with _LOCK:
        mod = _MODS.get(idx)
        if mod is None:
            major, minor = torch.cuda.get_device_capability(idx)
            mod, _info = cudart.load_module(KERNELS, idx, arch=f"sm_{major}{minor}")
            _MODS[idx] = mod
    fn = mod.function(name)
    if smem > 48 * 1024 and getattr(fn, "_smem", 0) < smem:
        fn.set_max_shared(smem)
        fn._smem = smem
    return fn


def _ac(mean, rstd, g, b):
    a = (g.float() * rstd.float()).contiguous()
    return a, (b.float() - mean.float() * a).contiguous()


# Default grids of the kernels with a fixed number of programs, by number of SMs. Below 100 SMs
# (T4 40, L4 58) the values tuned on the T4; from 100 up (A100 108, H100 132) scaled with the SMs:
# on an A100 (step with CUDA graphs, where only the GPU counts) 80.9 ms with the T4 values, 68.3 with
# ctas_bwd 2 x SM, 66.1 with ctas_bwd 1 x SM. c1b3 and c3d use ~230 registers per thread: one program
# per SM, and 80 or 40 programs leave 28 or 68 of the A100's 108 SMs idle.
T4_GRID = {"ctas_bwd": 80, "ctas_c3w": 80, "ctas_c3d": 40, "programs": 160}


def grid_defaults(sms: int) -> dict:
    """Default programs of c1b* (`ctas_bwd`), c3w, c3d and of the Triton reductions (`programs`)."""
    if sms < 100:
        return dict(T4_GRID)
    return {"ctas_bwd": sms, "ctas_c3w": 2 * sms, "ctas_c3d": sms, "programs": 4 * sms}


def _sm_count() -> int:
    if not torch.cuda.is_available():
        return 40
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


class CudaOps(CublasOps):
    """`CublasOps` with the conv1 (forward and backward) on the fused CUDA kernels. `bm_fwd`: rows per
    forward block (64 or 128); `ctas_bwd`: backward programs (channel slices x row slices, one program
    per SM at a time), `split_bwd` fixes the row slices. `pf_bwd` and `fuse_bn2` are off by default:
    the fused BN2 repeats the prologue for every channel slice (up to 26 in block 4), and keeping the
    next block in flight does not help (the latency of x and G is not the limit). Defaults `c1b3` and
    `bm_fwd` 128: step -7% (c1b3) and -2% (forward at 128 rows).

    Further defaults, measured on two power-capped T4s with alternating arms: the 3x3 bottleneck conv
    with the c3f / c3d / c3w kernels (`conv3="c3d"`), the stem backward in CUDA (`stem="cuda"`), the
    tiny launches fused (`fuse_small`) and the persistent conv1 forward on the large blocks
    (`fwd="auto"`): step -15% (0.33-0.35 s against 0.40-0.45 on the same GPUs), as close to fp32 as
    before (median cosine of the full-scale gradients 0.9905 against 0.9902). `conv3="cudnn"`,
    `stem="triton"`, `fuse_small=False`, `fwd="c1f"` reproduce the earlier path.

    `ctas_bwd`, `ctas_c3w`, `ctas_c3d` and `programs` at 0 (default) take `grid_defaults` for the
    number of SMs of the current GPU: on the T4 the same values as before, on the A100 scaled.
    """

    name = "cuda"

    def __init__(self, bm_fwd: int = 128, ctas_bwd: int = 0, split_bwd: int = 0,
                 pf_bwd: bool = False, fuse_bn2: bool = False, bwd: str = "c1b3",
                 conv3: str = "c3d", stem: str = "cuda", c3f_bm: int = 0, ctas_c3w: int = 0,
                 ctas_c3d: int = 0, fuse_small: bool = True,
                 fwd: str = "auto", **kw):
        grid = grid_defaults(_sm_count())
        if not kw.get("programs"):
            kw["programs"] = grid["programs"]
        ctas_c3w, ctas_c3d = int(ctas_c3w) or grid["ctas_c3w"], int(ctas_c3d) or grid["ctas_c3d"]
        super().__init__(**kw)
        self.name = "cuda"
        # `conv3`: the 3x3 bottleneck conv on cuDNN over materialised V ("cudnn") or with the c3f kernels
        # (forward with BN2 + ReLU in the prologue, V written for the backward, segment statistics in the
        # epilogue) and c3w (weight gradient): "c3" with the input gradient on cuDNN and the BN2 backward in
        # Triton, "c3d" with c3d (dgrad + BN2 reductions in the epilogue) and bn2_apply. `stem`: the stem
        # backward in Triton ("triton") or in CUDA with the reductions over the windows ("cuda"). `c3f_bm`:
        # pixels per c3f block (0 = depending on the pixels).
        if conv3 not in ("cudnn", "c3", "c3d"):
            raise ValueError(f"CudaOps: conv3 {conv3!r}, expected 'cudnn', 'c3' or 'c3d'")
        if stem not in ("triton", "cuda"):
            raise ValueError(f"CudaOps: stem {stem!r}, expected 'triton' or 'cuda'")
        if c3f_bm not in (0, 64, 128, 256):
            raise ValueError(f"CudaOps: c3f_bm {c3f_bm}, expected 0, 64, 128, 256")
        self.conv3, self.stem = conv3, stem
        self.c3f_bm, self.ctas_c3w, self.ctas_c3d = int(c3f_bm), int(ctas_c3w), int(ctas_c3d)
        # `fuse_small`: the tiny launches fused. Conv weights in fp16 in the kernels' layouts once per step
        # (`prep_weights`, one wprep launch instead of ~500 casts and permutes), BN statistics with the a, c
        # coefficients in the same kernel (colsum or `_bn_tab`) and remembered for the step (`_ac`), the sums
        # of the c1f, c3f and c3d partials closed inside colsum, those of c3w in wsum
        self.fuse_small = bool(fuse_small)
        # `fwd`: the conv1 forward with one program per row block ("c1f") or persistent, one program per SM
        # taking blocks in turn with the first chunk of the next block in flight during the epilogue
        # ("c1fp_64", "c1fp_128": two programs per SM; "c1fp1_128": one). "auto": c1fp1_128 above 100,000
        # rows (blocks 1-2), c1f below. Micro-benchmark: c1fp1_128 -16% in block 1, -6% in block 2, +3% in
        # blocks 3-4; the two-program variants are worse (spills)
        if fwd not in ("c1f", "c1fp_64", "c1fp_128", "c1fp1_128", "auto"):
            raise ValueError(f"CudaOps: fwd {fwd!r}")
        self.fwd = fwd
        # colsum reads the shift constant of the sums before updating the running averages (same thread):
        # the norm2 constant can be its running mean, without a copy (`fastdense._ushift`)
        self.shift_alias_ok = self.fuse_small
        self._memo: dict = {}
        self._wmap: dict = {}
        self._wdmap: dict = {}
        self._wkey = None
        if conv3 != "cudnn":
            self.name += f"+{conv3}"
        if stem != "triton":
            self.name += f"+stem-{stem}"
        if bm_fwd not in SMEM_C1F:
            raise ValueError(f"CudaOps: bm_fwd {bm_fwd}, expected {sorted(SMEM_C1F)}")
        self.bm_fwd, self.ctas_bwd, self.split_bwd = int(bm_fwd), int(ctas_bwd), int(split_bwd)
        self.pf_bwd, self.fuse_bn2 = bool(pf_bwd), bool(fuse_bn2)
        # `bwd`: "c1b" (blocks of 64 rows, one program per SM), "c1b2" (32 rows, two per SM), "c1b3" (like
        # c1b, x and G in 16-byte chunks through shared memory)
        if bwd not in ("c1b", "c1b2", "c1b3") and not bwd.startswith("c1b3_a"):
            raise ValueError(f"CudaOps: bwd {bwd!r}, expected 'c1b', 'c1b2' or 'c1b3'")
        if bwd != "c1b" and (pf_bwd or fuse_bn2):
            raise ValueError(f"CudaOps: {bwd} without pf_bwd / fuse_bn2")
        self.bwd = bwd
        self.ctas_bwd = int(ctas_bwd) or (2 * grid["ctas_bwd"] if bwd == "c1b2" else grid["ctas_bwd"])

    # --- fuse_small: per-step memory, weights in one launch, fused statistics and sums ---------------
    def _ac(self, mean, rstd, g, b):
        """a, c of a BatchNorm: from the step's `bn_stats` table if present, otherwise computed."""
        e = self._memo.get((mean.data_ptr(), g.data_ptr())) if self._memo else None
        if e is not None and e[1] == rstd.data_ptr() and e[2] == b.data_ptr():
            return e[0][0], e[0][1]
        return _ac(mean, rstd, g, b)

    def remember_bn(self, tab, g, b) -> None:
        """The [4][C] table (a, c, mean, rstd) of a BatchNorm, which `_ac` / `_tab` find again from the
        pointers of tab[2] (the mean passed to the primitives) and of g. The memory keeps the table alive:
        its address is not reused by another tensor while the entry exists.
        """
        self._memo[(tab[2].data_ptr(), g.data_ptr())] = (tab, tab[3].data_ptr(), b.data_ptr())

    def bn_table(self, mean, rstd, g, b):
        """The constants of a BatchNorm in eval, once per encoder (`fastdense._eval_consts`): the `_tab`
        table, with a, c from `_ac` bit for bit.
        """
        a, c = _ac(mean, rstd, g, b)
        return torch.stack([a, c, _f32(mean), _f32(rstd)]).contiguous()

    def conv3_eval_weight(self, W2):
        """The 3x3 conv weights in the c3f layout ([t][o][c]) seen as (O, C, 3, 3): the permute of
        `conv3_fwd` becomes a view instead of a copy at every call (same values).
        """
        if tuple(W2.shape) != (32, 128, 3, 3):
            return W2
        return W2.permute(2, 3, 0, 1).reshape(9, 32, 128).contiguous().view(3, 3, 32, 128) \
            .permute(2, 3, 0, 1)

    def _tab(self, mean, rstd, g, b):
        """[4][C]: a, c, mean, rstd (the c3d table)."""
        e = self._memo.get((mean.data_ptr(), g.data_ptr())) if self._memo else None
        if e is not None and e[1] == rstd.data_ptr() and e[2] == b.data_ptr():
            return e[0]
        a, c = _ac(mean, rstd, g, b)
        return torch.stack([a, c, _f32(mean), _f32(rstd)]).contiguous()

    def _counters(self, dev, n: int) -> torch.Tensor:
        """colsum counters (zero between launches: the last block resets them)."""
        cnt = getattr(self, "_cnt", None)
        if cnt is None or cnt.device != dev or cnt.numel() < n:
            cnt = torch.zeros(max(n, 64), dtype=torch.int32, device=dev)
            self._cnt = cnt
        return cnt

    def _colsum(self, part, lo1: int, lo2: int, C: int, mode: int, **kw):
        """Launches colsum on the rows of `part` (see fastdense_kernels.cu)."""
        R, ld = part.shape[0], part.stride(0)
        dev = part.device
        gx = triton.cdiv(C, 128)
        gy = max(1, min(64, R))
        scr = torch.empty(gy, 2, C, dtype=torch.float32, device=dev)
        cnt = self._counters(dev, gx)
        z = 0
        ptr = (lambda t: t.data_ptr() if t is not None else z)
        arg = CS(part.data_ptr(), ptr(kw.get("part2")), scr.data_ptr(), cnt.data_ptr(), ptr(kw.get("sh")), ptr(kw.get("g")),
                 ptr(kw.get("b")), ptr(kw.get("rm")), ptr(kw.get("rv")), ptr(kw.get("nbt")),
                 ptr(kw.get("tab")), ptr(kw.get("out1")), ptr(kw.get("out2")), ptr(kw.get("ab")),
                 R, ld, lo1, lo2, C, mode, int(kw.get("upd_nbt", 0)), float(kw.get("inv_n", 0.0)),
                 float(kw.get("n_corr", 0.0)), float(kw.get("eps", 0.0)), float(kw.get("mom", 0.0)))
        _function(dev, "colsum").launch((gx, gy), (128,), [arg],
                                        torch.cuda.current_stream(dev).cuda_stream, 0)

    def bn_stats(self, s1, s2, shift, n: int, bn):
        """With `fuse_small`: mean, rstd and a, c in a [4][C] table (remembered for the step) and the running
        averages, in one launch; s1, s2 may also be `Partial` (the sums happen inside colsum).
        """
        if not self.fuse_small:
            return super().bn_stats(materialize(s1), materialize(s2), shift, n, bn)
        C = bn.running_mean.shape[0]
        dev = bn.running_mean.device
        tab = torch.empty(4, C, dtype=torch.float32, device=dev)
        nbt = bn.num_batches_tracked
        kw = dict(sh=_f32(shift), g=_f32(bn.weight.detach()), b=_f32(bn.bias.detach()),
                  rm=bn.running_mean, rv=bn.running_var, nbt=nbt, tab=tab,
                  upd_nbt=nbt is not None, inv_n=1.0 / n, n_corr=n / max(n - 1, 1),
                  eps=float(bn.eps), mom=float(bn.momentum))
        if isinstance(s1, Partial):
            if s2.part is not s1.part:
                raise ValueError("bn_stats: s1 and s2 from different partials")
            self._colsum(s1.part, s1.lo, s2.lo, C, 0, **kw)
        else:                                   # sums already reduced: one "row" per tensor
            self._colsum(_f32(s1).reshape(1, C), 0, 0, C, 0, part2=_f32(s2), **kw)
        self.remember_bn(tab, bn.weight, bn.bias)
        return tab[2], tab[3]

    def store_stats(self, s1, s2, lo: int, a1, a2):
        """s1[lo:lo+n], s2[lo:lo+n] = a1, a2 (with `Partial`: the sum inside colsum)."""
        if isinstance(a1, Partial):
            if a2.part is not a1.part:
                raise ValueError("store_stats: a1 and a2 from different partials")
            self._colsum(a1.part, a1.lo, a2.lo, a1.n, 1, out1=s1[lo:lo + a1.n], out2=s2[lo:lo + a1.n])
            return
        s1[lo:lo + a1.shape[0]] = a1
        s2[lo:lo + a2.shape[0]] = a2

    def prep_weights(self, spec, params, conv_dtype) -> None:
        """At the start of the `EncoderFn` forward, once per step: the statistics memory zeroed and (with
        `fuse_small`) the conv weights in fp16 in the arena, in one launch.
        """
        self._memo = {}
        if not self.fuse_small or conv_dtype != torch.float16:
            self._wmap, self._wdmap = {}, {}
            return
        convs = [p for p in params if p.dim() == 4 and p.is_cuda and p.dtype == torch.float32]
        if not convs:
            return
        key = tuple((p.data_ptr(), tuple(p.shape), tuple(p.stride())) for p in convs)
        if key != self._wkey:
            self._build_wprep(convs)
            self._wkey = key
        dev = convs[0].device
        _function(dev, "wprep").launch((triton.cdiv(self._wtotal, 256),), (256,), [self._wp],
                                       torch.cuda.current_stream(dev).cuda_stream, 0)

    def _build_wprep(self, convs) -> None:
        jobs, views, start = [], [], 0
        for w in convs:
            O, C, KH, KW = w.shape
            st = w.stride()
            if KH == KW == 1:
                lays = [0]
            elif KH == KW == 3:
                lays = [1, 2]
            else:
                continue
            for lay in lays:
                n = O * C * KH * KW
                jobs.append((w, lay, start, n, O, C, KH * KW, st))
                start += n
        dev = convs[0].device
        arena = torch.empty(start, dtype=torch.float16, device=dev)
        rows = np.zeros(len(jobs), dtype=WJOB)
        self._wmap, self._wdmap = {}, {}
        for i, (w, lay, st0, n, O, C, T, st) in enumerate(jobs):
            region = arena[st0:st0 + n]
            rows[i] = (w.data_ptr(), region.data_ptr(), st[0], st[1], st[2], st[3], st0, O, C, T, lay)
            if lay == 0:
                self._wmap[w.data_ptr()] = region.view(O, C, 1, 1)
            elif lay == 1:
                # the logical shape (O, C, 3, 3) over the [t][o][c] memory: c3f wants it this way
                v = region.as_strided((O, C, 3, 3), (C, 1, 3 * O * C, O * C))
                self._wmap[w.data_ptr()] = v
                wk_ptr = v.data_ptr()
            else:
                self._wdmap[wk_ptr] = region.view(9, C, O)
        self._arena = arena
        self._wjobs = torch.from_numpy(rows.view(np.uint8)).to(dev)
        self._wtotal = start
        self._wp = WP(self._wjobs.data_ptr(), start, len(jobs))

    def conv_weight(self, w, conv_dtype):
        """The weight of a conv in `conv_dtype`: the `prep_weights` arena view if present."""
        v = self._wmap.get(w.data_ptr()) if self._wmap else None
        if v is not None and conv_dtype == torch.float16:
            return v
        return w.to(conv_dtype)

    def bnrelu_bwd(self, dV, U, mean, rstd, g, b):
        """With `fuse_bn2` (on GPU, fp16, 128 channels): only the reductions, and a `LazyDU` the conv1 kernel
        applies while loading; otherwise as `CublasOps`.
        """
        if not (self.fuse_bn2 and U.is_cuda and U.dtype == torch.float16 and U.shape[1] == 128
                and U.is_contiguous()):
            return super().bnrelu_bwd(dV, U, mean, rstd, g, b)
        db, dg = self._bn_bwd_sums(dV, U, mean, rstd, g, b)
        return LazyDU(dV.contiguous(), U, mean, rstd, g, b, db, dg), dg, db

    @staticmethod
    def _ok(X, cin: int, *more) -> bool:
        return (X.is_cuda and X.dtype == torch.float16 and cin % 32 == 0 and X.stride(1) == 1
                and X.stride(0) % 8 == 0 and all(t.dtype == torch.float16 for t in more))

    def bnrelu_gemm(self, X, cin: int, mean, rstd, g, b, W, ushift):
        if not self._ok(X, cin, W) or W.shape[0] != 128:
            return super().bnrelu_gemm(X, cin, mean, rstd, g, b, W, ushift)
        M = X.shape[0]
        W = W.contiguous()
        a, c = self._ac(mean, rstd, g, b)
        sh = _f32(ushift)
        U = torch.empty(M, 128, dtype=X.dtype, device=X.device)
        fwd = self.fwd
        if fwd == "auto":
            fwd = "c1fp1_128" if M >= 100_000 else "c1f"
        if fwd == "c1f":
            BM, name = self.bm_fwd, f"c1f_{self.bm_fwd}"
            grid = triton.cdiv(M, BM)
        else:
            BM = int(fwd.split("_")[1])
            name = fwd
            per_sm = 1 if fwd.startswith("c1fp1") else 2
            sms = torch.cuda.get_device_properties(X.device).multi_processor_count
            grid = max(1, min(triton.cdiv(M, BM), per_sm * sms))
        part = torch.empty(grid, 256, dtype=torch.float32, device=X.device)
        arg = C1F(X.data_ptr(), W.data_ptr(), a.data_ptr(), c.data_ptr(), sh.data_ptr(),
                  U.data_ptr(), part.data_ptr(), M, cin, X.stride(0))
        _function(X.device, name).launch(
            (grid,), (256,), [arg], torch.cuda.current_stream(X.device).cuda_stream, SMEM_C1F[BM])
        if self.fuse_small:
            return U, Partial(part, 0, 128), Partial(part, 128, 128)
        s = part.sum(0)
        return U, s[:128], s[128:]

    def bnrelu_gemm_bwd(self, dU, W, X, cin: int, mean, rstd, g, b, G):
        lazy = isinstance(dU, LazyDU)
        if (not self._ok(X, cin, W, G) or W.shape[0] != 128 or G.stride(1) != 1
                or G.stride(0) % 2 or dU.dtype != torch.float16):
            return super().bnrelu_gemm_bwd(dU.materialize() if lazy else dU, W, X, cin, mean,
                                           rstd, g, b, G)
        M = X.shape[0]
        if lazy:
            dV, U2, tab = dU.dV, dU.U, dU.table()
            dU = None
        else:
            dU = dU.contiguous()
        W = W.contiguous()
        a, c = self._ac(mean, rstd, g, b)
        mu, rs = _f32(mean), _f32(rstd)
        nc = triton.cdiv(cin, 64)
        ntiles = triton.cdiv(M, 32 if self.bwd == "c1b2" else 64)
        # programs <= ctas_bwd (two full waves on 40 SMs with c1b): with round() many layers ended at 81-88
        # programs, a third wave of 1-8 and the layer ~1.5 times longer
        split = self.split_bwd or max(1, min(ntiles, self.ctas_bwd // nc))
        partR = torch.empty(split, 2, cin, dtype=torch.float32, device=X.device)
        partW = torch.empty(split, 128, cin, dtype=torch.float32, device=X.device)
        arg = C1B(0 if lazy else dU.data_ptr(), X.data_ptr(), W.data_ptr(), a.data_ptr(),
                  c.data_ptr(), mu.data_ptr(), rs.data_ptr(), G.data_ptr(), partR.data_ptr(),
                  partW.data_ptr(), M, cin, X.stride(0), G.stride(0), split,
                  dV.data_ptr() if lazy else 0, U2.data_ptr() if lazy else 0,
                  tab.data_ptr() if lazy else 0)
        if self.bwd == "c1b2" or self.bwd.startswith("c1b3"):
            name, smem = self.bwd, (SMEM_C1B2 if self.bwd == "c1b2" else SMEM_C1B3)
        else:
            name = "c1b" + ("_" if (self.pf_bwd or lazy) else "") + ("p" if self.pf_bwd else "") + \
                ("b" if lazy else "")
            smem = SMEM_C1B_BN2 if lazy else SMEM_C1B
        _function(X.device, name, smem).launch(
            (nc, split), (256,), [arg], torch.cuda.current_stream(X.device).cuda_stream, smem)
        r = partR.sum(0)
        return partW.sum(0), r[1], r[0]

    # --- the 3x3 bottleneck conv (c3f, c3w) ------------------------------------------------------
    def _c3_ok(self, U, W2, w: int) -> bool:
        return (self.conv3 in ("c3", "c3d") and U.is_cuda and U.dtype == torch.float16 and U.is_contiguous()
                and U.dim() == 2 and U.shape[1] == 128 and tuple(W2.shape) == (32, 128, 3, 3)
                and W2.dtype == torch.float16 and w <= C3_WMAX)

    def conv3_fwd(self, U, mean2, rstd2, g2, b2, W2, X, c0: int, shift, n: int, h: int, w: int,
                  save: bool = True):
        """c3f: X[:, c0:c0+32] = conv3x3(relu(U a + c), W2), the segment sums and (with `save`)
        V = relu(U a + c) for the backward, in one kernel.
        """
        if not (self._c3_ok(U, W2, w) and X.dtype == torch.float16 and X.stride(1) == 1
                and X.stride(0) % 8 == 0 and c0 % 8 == 0 and U.shape[0] == n * h * w):
            return super().conv3_fwd(U, mean2, rstd2, g2, b2, W2, X, c0, shift, n, h, w)
        P = U.shape[0]
        a, c = self._ac(mean2, rstd2, g2, b2)
        Wk = W2.permute(2, 3, 0, 1).reshape(9, 32, 128).contiguous()
        V = torch.empty_like(U) if save else None
        BM = self.c3f_bm or 256          # micro-benchmark: 256 the fastest in all 4 blocks
        abl = getattr(self, "c3f_abl", "")       # micro-benchmark only: "_a1", "_a2" (ablations)
        grid = triton.cdiv(P, BM)
        part = torch.empty(grid, 64, dtype=torch.float32, device=U.device)
        sh = _f32(shift)
        arg = C3F(U.data_ptr(), Wk.data_ptr(), a.data_ptr(), c.data_ptr(), sh.data_ptr(),
                  V.data_ptr() if V is not None else 0, X.data_ptr(), part.data_ptr(), P, h, w,
                  X.stride(0), c0)
        smem = smem_c3f(BM, w)
        _function(U.device, f"c3f_{BM}{abl}", smem).launch(
            (grid,), (256,), [arg], torch.cuda.current_stream(U.device).cuda_stream, smem)
        if self.fuse_small:
            return Partial(part, 0, 32), Partial(part, 32, 32), V
        s = part.sum(0)
        return s[:32], s[32:], V

    def conv3_wgrad(self, dS, V, h: int, w: int):
        """c3w: the weight gradient of the 3x3 conv (32, 128, 3, 3) in fp32 from dS (P x 32) and V.
        (Applying BN2 inside c3w was measured at +1..4% per step and removed.)
        """
        P = dS.shape[0]
        ntiles = triton.cdiv(P, 128)
        split = max(1, min(ntiles, self.ctas_c3w // 4))
        partW = torch.empty(split, 9, 32, 128, dtype=torch.float32, device=dS.device)
        arg = C3W(dS.data_ptr(), V.data_ptr(), partW.data_ptr(), P, h, w, split)
        smem = smem_c3w(w)
        stream = torch.cuda.current_stream(dS.device).cuda_stream
        _function(dS.device, "c3w", smem).launch(
            (4, split), (288,), [arg], stream, smem)
        if self.fuse_small:
            dW = torch.empty(32, 128, 3, 3, dtype=torch.float32, device=dS.device)
            _function(dS.device, "wsum").launch((triton.cdiv(32 * 128 * 9, 256),), (256,),
                                                [WS(partW.data_ptr(), dW.data_ptr(), split)], stream, 0)
            return dW
        return partW.sum(0).permute(1, 2, 0).reshape(32, 128, 3, 3)

    def conv3_dgrad_bn(self, dS, U, W2, mean2, rstd2, g2, b2, h: int, w: int, fin: bool = False):
        """c3d: G = a gm (P x 128, fp16; gm = dV [U a + c > 0], dV = dgrad of dS) and the norm2 backward sums
        R1 = sum of gm (dbeta), R2 = sum of gm uhat (dgamma). With `fin` also alpha, beta of BN2
        (`split_coeffs`, [2][128]) in the same closing step (colsum).
        """
        P = U.shape[0]
        tab = self._tab(mean2, rstd2, g2, b2)
        Wd = self._wdmap.get(W2.data_ptr()) if self._wdmap else None
        if Wd is None:
            Wd = W2.permute(2, 3, 1, 0).reshape(9, 128, 32).contiguous()
        G = torch.empty_like(U)
        ntiles = triton.cdiv(P, 256)
        split = max(1, min(ntiles, self.ctas_c3d // 4))
        part = torch.empty(split, 256, dtype=torch.float32, device=U.device)
        arg = C3D(dS.data_ptr(), Wd.data_ptr(), U.data_ptr(), tab.data_ptr(), G.data_ptr(),
                  part.data_ptr(), P, h, w, split)
        smem = smem_c3d(w)
        _function(U.device, "c3d", smem).launch(
            (4, split), (256,), [arg], torch.cuda.current_stream(U.device).cuda_stream, smem)
        if fin:
            db = torch.empty(128, dtype=torch.float32, device=U.device)
            dg = torch.empty_like(db)
            ab = torch.empty(2, 128, dtype=torch.float32, device=U.device)
            self._colsum(part, 0, 128, 128, 2, tab=tab, out1=db, out2=dg, ab=ab, inv_n=1.0 / P)
            return G, db, dg, ab
        r = part.sum(0)
        return G, r[:128], r[128:]

    def bn2_tab(self, U, mean2, rstd2, g2, dg2, db2):
        """[2][128]: alpha, beta of the norm2 corrections (`fastdense.split_coeffs`)."""
        tab = torch.zeros(2, 128, dtype=torch.float32, device=U.device)
        self.bn_coeffs_add(mean2, rstd2, g2, dg2, db2, U.shape[0], tab[0], tab[1])
        return tab

    def _bn2_apply_tab(self, G, U, tab, dU):
        n = U.shape[0] * 16
        arg = BNA(G.data_ptr(), U.data_ptr(), tab.data_ptr(), dU.data_ptr(), U.shape[0])
        _function(U.device, "bn2_apply").launch(
            (triton.cdiv(n, 256),), (256,), [arg], torch.cuda.current_stream(U.device).cuda_stream, 0)
        return dU

    def bn2_apply(self, G, U, mean2, rstd2, g2, dg2, db2):
        """dU = G + alpha + beta U (the norm2 corrections, `fastdense.split_coeffs`)."""
        tab = self.bn2_tab(U, mean2, rstd2, g2, dg2, db2)
        dU = torch.empty_like(U)
        n = U.shape[0] * 16
        arg = BNA(G.data_ptr(), U.data_ptr(), tab.data_ptr(), dU.data_ptr(), U.shape[0])
        _function(U.device, "bn2_apply").launch(
            (triton.cdiv(n, 256),), (256,), [arg], torch.cuda.current_stream(U.device).cuda_stream, 0)
        return dU

    def conv3_bwd(self, dS, U, saved, mean2, rstd2, g2, b2, W2, n: int, h: int, w: int):
        """`conv3` "c3": dV with cuDNN (dgrad) and the BN2 backward as in `CublasOps`; "c3d": c3d and
        bn2_apply. dW with c3w in both.
        """
        V = saved
        if not (self._c3_ok(U, W2, w) and V is not None and V.dtype == torch.float16
                and V.is_contiguous() and V.shape == U.shape and dS.is_contiguous()
                and dS.dtype == torch.float16 and dS.shape == (U.shape[0], 32)):
            return super().conv3_bwd(dS, U, saved, mean2, rstd2, g2, b2, W2, n, h, w)
        if self.conv3 == "c3d" and self.fuse_small:
            G, db2, dg2, ab = self.conv3_dgrad_bn(dS, U, W2, mean2, rstd2, g2, b2, h, w, fin=True)
            dU = torch.empty_like(U)
            self._bn2_apply_tab(G, U, ab, dU)
            return dU, self.conv3_wgrad(dS, V, h, w), dg2, db2
        if self.conv3 == "c3d":
            G, db2, dg2 = self.conv3_dgrad_bn(dS, U, W2, mean2, rstd2, g2, b2, h, w)
            dU = self.bn2_apply(G, U, mean2, rstd2, g2, dg2, db2)
        else:
            w2c = W2.contiguous(memory_format=torch.channels_last)
            dV4 = torch.ops.aten.convolution_backward(
                _as_nchw(dS, n, h, w), _as_nchw(V, n, h, w), w2c, None, [1, 1], [1, 1], [1, 1],
                False, [0, 0], 1, [True, False, False])[0]
            dU, dg2, db2 = self.bnrelu_bwd(_nhwc_rows(dV4), U, mean2, rstd2, g2, b2)
        return dU, self.conv3_wgrad(dS, V, h, w), dg2, db2

    # --- the stem backward in CUDA (stem_red, stem_bwd) ---------------------------------------
    def stem_fwd(self, Y, n: int, H: int, W: int, mean, rstd, g, b):
        """Like `TritonOps.stem_fwd`, plus YM = Y at the tap of the maximum (for stem_red)."""
        if self.stem != "cuda" or Y.dtype != torch.float16 or Y.shape[1] != 64:
            return super().stem_fwd(Y, n, H, W, mean, rstd, g, b)
        return self._stem_fwd(Y, n, H, W, mean, rstd, g, b, with_ym=True)

    def stem_bwd(self, Y, n: int, H: int, W: int, mean, rstd, g, b, dS, S, AM, YM=None):
        if YM is None:
            return super().stem_bwd(Y, n, H, W, mean, rstd, g, b, dS, S, AM)
        M = Y.shape[0]
        Ho, Wo = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        M2 = n * Ho * Wo
        dS = dS.contiguous()
        mu, rs = _f32(mean), _f32(rstd)
        tab = torch.empty(5, 64, dtype=torch.float32, device=Y.device)
        tab[0] = g.float() * rs
        tab[1] = mu
        tab[2] = rs
        nb = max(1, min(triton.cdiv(M2, 32), 320))
        part = torch.empty(nb, 128, dtype=torch.float32, device=Y.device)
        stream = torch.cuda.current_stream(Y.device).cuda_stream
        dY = torch.empty_like(Y)
        arg = STB(Y.data_ptr(), dS.data_ptr(), S.data_ptr(), YM.data_ptr(), AM.data_ptr(),
                  tab.data_ptr(), dY.data_ptr(), part.data_ptr(), M2, H, W, Ho, Wo)
        _function(Y.device, "stem_red").launch((nb,), (256,), [arg], stream, 0)
        r = part.sum(0)
        db, dg = r[:64], r[64:]
        tab[3] = db / M
        tab[4] = dg / M
        _function(Y.device, "stem_bwd").launch((triton.cdiv(W, 16), n * H), (128,), [arg], stream, 0)
        return dY, dg, db


def _as_nchw(rows: torch.Tensor, n: int, h: int, w: int) -> torch.Tensor:
    c = rows.shape[1]
    return rows.reshape(n, h, w, c).permute(0, 3, 1, 2)


def _nhwc_rows(t: torch.Tensor) -> torch.Tensor:
    n, c, h, w = t.shape
    return t.permute(0, 2, 3, 1).reshape(n * h * w, c).contiguous()


__all__ = ["CudaOps", "Partial", "materialize"]
