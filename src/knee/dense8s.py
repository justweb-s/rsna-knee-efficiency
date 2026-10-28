"""dense8s: the DenseNet encoder in INT8 with a per-block "store" and a hand-written conv1. This is the
encoder of the final efficiency submission.

The earlier engine (`knee.dense8`) took BatchNorm+ReLU+Quantize over the whole concatenation away
from TensorRT, but redid them by copying every segment, already quantized, into the input of EVERY
layer consuming it: in block 1 ~1,100 channels written per pixel instead of 256. At 49.9 ms per pass
of 88 images: fan kernels 15.6 ms, TensorRT conv1 ~13 (re-reading the copies), conv2 ~12.4, stem
4.1, transitions 2.4. Five seeds x four TTA views (20 passes per study) would take ~650 s on the two
T4s: ~18 ms per pass was needed.

Here every segment is written ONCE, in INT8 with a per-channel scale (the block "store":
`dense8s_kernels.cu`), and the conv1 of every layer is one of our tensor-core kernels that applies
its consumer's BatchNorm, ReLU and quantization while loading (prologue): no copies, no fan kernels.
The 3x3 conv2 can stay on TensorRT (INT8 in, INT8 out straight into the store: the per-channel scale
is folded into the weights, Q with scale 1) or run on our kernels (`conv2_q*`, the final choice); the
stem can stay fp16 in TensorRT with INT8 output (per-channel scale inside norm0). Transitions and
final features: our kernels (2x2 mean in the epilogue, from the integer accumulators).

Quantization, for every conv input:
- store: q = rint(x / sigma_c), sigma_c = max|x_c| / 127 per channel (calibration);
- conv1 and transition input: u8 [0, 255] = relu(bn(x)) / s, s = max / 255 per tensor (one bit more
  than dense8: no TensorRT, no symmetric int8);
- conv2 input: int8 [0, 127] per tensor (TensorRT), or u8 with our conv2;
- weights: int8 per output channel (`dense8._wq`).
Compared with dense8 there is one more rounding (the store) and a finer one (u8). Simulated on one
validation fold, post-training INT8 costs -0.0003 OOF AUC (Spearman 0.9986 with fp32).

Parts:
- `calibrate(encoder, batches)`: per-channel (store) and per-tensor (conv inputs) maxima;
- `make_plan(encoder, cal, img)` and `kernel_params(plan)`: weights, scales, kernel tables;
- `emulate(plan, x, quant)`: the same integer arithmetic in PyTorch (reference and tests);
- `build_engines(plan, n)`: stem and conv2 in TensorRT (needs the builder, on the GPU);
- `save_pack` / `load_pack`; `Dense8SEncoder(pack, dev, n_total)`: the runner, with the interface of
  `knee.trt.TrtEncoder` / `knee.dense8.Dense8Encoder`.
"""
from __future__ import annotations

import collections
import ctypes
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import dense8 as d8

QU = 255                        # u8 inputs of our kernels
QS = 127                        # store and TensorRT INT8 inputs
GROUP = d8.GROUP
GROWTH = d8.GROWTH
BOTTLENECK = d8.BOTTLENECK
KERNELS = Path(__file__).with_name("dense8s_kernels.cu")
PACK_FORMAT = 1
KIND = "dense8s"
THREADS = 256
STEM_TP = 9                     # side of the `stem_q` max-pool rectangle (ST_TP)
STEM_SMEM = 64 * 168 * 2 + 512 + 3 * (4 * STEM_TP + 7) * (2 * STEM_TP + 4) * 4 \
    + (2 * STEM_TP + 1) ** 2 * 68
# `stem_i8` / `stem_i8w` (SI_SMEM): int8 weights [64][176], k and c, two copies of the u8 rectangle
# (stride 5696), conv outputs [361][68]
STEM_I8_SMEM = -(-(64 * 176 + 512 + 5696 + 3 * 43 * 44) // 16) * 16 + 361 * 68
# `stem_q2` (ST2_SMEM): like stem_q with 80-byte conv rows
STEM2_SMEM = -(-(64 * 168 * 2 + 512 + 3 * (4 * STEM_TP + 7) * (2 * STEM_TP + 4) * 4) // 16) * 16 \
    + (2 * STEM_TP + 1) ** 2 * 80
# our stems: "ours" = `stem_q` (fp16), "q2" = `stem_q2` (bit-identical arithmetic), "i8" / "i8w" =
# `stem_i8` with 8 / 16 warps (INT8)
STEM_OURS = {"ours": ("stem_q", 256), "q2": ("stem_q2", 256), "i8": ("stem_i8", 256),
             "i8w": ("stem_i8w", 512)}


class KArg(ctypes.Structure):
    """The `struct KArg` of the GEMM kernels (`conv1_q*`, `trans_q128`), passed by value."""
    _fields_ = [("in0", ctypes.c_uint64), ("seg", ctypes.c_uint64), ("w", ctypes.c_uint64),
                ("ab", ctypes.c_uint64), ("gd", ctypes.c_uint64), ("out", ctypes.c_uint64),
                ("G0", ctypes.c_int), ("G", ctypes.c_int), ("N", ctypes.c_int),
                ("HW", ctypes.c_int), ("W", ctypes.c_int), ("rows", ctypes.c_int),
                ("cout", ctypes.c_int), ("G0n", ctypes.c_int)]


# warp-specialised conv1 (`conv1_ws_body`): type -> (tile rows, producer warps, MMA warps);
# 3-stage ring, one persistent block per SM
C1_WS = {"w4": (128, 4, 8), "w8": (128, 8, 8), "w4h": (64, 4, 8), "w2h": (64, 2, 8),
         "x4": (128, 4, 16), "x8": (128, 8, 16), "w4f": (128, 4, 8)}
C1_WS_STAGES = 3


def ws_shared(bm: int, groups: int) -> int:
    """Bytes of dynamic shared memory of `conv1_w*` (ring, epilogue, prologue)."""
    return C1_WS_STAGES * (bm * 32 + 128 * 32) + bm * 144 + groups * 32 * 8


# phased conv1 with more warps per SM (32 x 32 tile per warp): type -> (rows, threads)
C1_HI = {"u128x16": (128, 512), "u64x8": (64, 256), "h128": (128, 512), "h128h": (128, 512),
         "u128h": (128, 256), "u256h": (256, 512), "u128p": (128, 256), "u128hp": (128, 256),
         "u256hp": (256, 512), "u128hs": (128, 256), "u256hs": (256, 512),
         "u256hps": (256, 512), "p128": (128, 256), "u64hs": (64, 256), "u256hs_pl": (256, 512),
         # 64 x 64 warp tile, 8 warps (`conv1w_body`)
         "w64": (256, 256), "w64_pl": (256, 256)}
# conv1 with the prologue in fp16x2 (`pro4h`, table L*_abh): roundings as in `emulate(f16_layers)`
C1_F16 = {"h128h", "u128h", "u256h", "w4f", "u128hp", "u256hp", "u128hs", "u256hs", "u256hps",
          "p128", "u64hs", "u256hs_pl", "w64", "w64_pl"}
# Planar bottleneck: the "*_pl" conv1 kernels write the bottleneck as [4][N*HW][32] (MODE 3) and the
# "*_pl" conv2 kernels read it that way (in NHWC128 the conv2 reads 32 bytes out of every 128: ~2.3
# ms/GB at the power cap, against ~1.1 for contiguous reads). The two halves of a layer go in pairs.
PLANAR = "_pl"
# persistent conv1 (`conv1p_body`): grid of 2 blocks per SM, direct epilogue
C1_PERSIST = {"p128"}


def _c1kind(v):
    """The conv1 kernel of a layer: 64 / 128 / 256 (`conv1_u*`, block rows), a `C1_WS` type
    (warp-specialised) or a `C1_HI` type (phased, more warps per SM).
    """
    if isinstance(v, str) and (v in C1_WS or v in C1_HI):
        return v
    return int(v)


def gemm_shared(bm: int, mode: int, groups: int) -> int:
    """Bytes of dynamic shared memory of `gemm_q<bm, mode>` with `groups` groups read."""
    a_bytes, b_bytes = bm * 32, 128 * 32
    stage = (bm // 4 if mode == 1 else bm) * 144
    return max(2 * (a_bytes + b_bytes), stage) + groups * 32 * 8


class C2Arg(ctypes.Structure):
    """The `struct C2Arg` of `conv2_q*`, passed by value."""
    _fields_ = [("x", ctypes.c_uint64), ("w", ctypes.c_uint64), ("sc", ctypes.c_uint64),
                ("out", ctypes.c_uint64), ("N", ctypes.c_int), ("H", ctypes.c_int),
                ("W", ctypes.c_int), ("TH", ctypes.c_int), ("TW", ctypes.c_int),
                ("txn", ctypes.c_int)]


class LayerP(ctypes.Structure):
    """The `struct LayerP` of `dense_img*`: the parameters of one layer of the fused block."""
    _fields_ = [("w1", ctypes.c_uint64), ("abh", ctypes.c_uint64), ("gd", ctypes.c_uint64),
                ("w2", ctypes.c_uint64), ("sc", ctypes.c_uint64), ("G", ctypes.c_int),
                ("pad", ctypes.c_int)]


class TileArg(ctypes.Structure):
    """The `struct TileArg` of `dense_tile*` (one layer fused in rectangles), passed by value."""
    _fields_ = [("in0", ctypes.c_uint64), ("seg", ctypes.c_uint64), ("lp", ctypes.c_uint64),
                ("N", ctypes.c_int), ("H", ctypes.c_int), ("W", ctypes.c_int),
                ("G0", ctypes.c_int), ("l", ctypes.c_int), ("TH", ctypes.c_int),
                ("TW", ctypes.c_int), ("pad", ctypes.c_int)]


class ImgArg(ctypes.Structure):
    """The `struct ImgArg` of `dense_img*`, passed by value."""
    _fields_ = [("in0", ctypes.c_uint64), ("seg", ctypes.c_uint64), ("lp", ctypes.c_uint64),
                ("N", ctypes.c_int), ("H", ctypes.c_int), ("W", ctypes.c_int),
                ("G0", ctypes.c_int), ("L", ctypes.c_int), ("TH", ctypes.c_int)]


# Whole fused blocks (`dense_img_body`): type -> (kernel, images per thread block). The persistent
# block runs every layer of P images (conv1 with the fp16 prologue, bottleneck in shared memory,
# conv2): no per-layer launches and no bottleneck in DRAM. Only images with P * HW <= 255 (so not at
# 512 px, where they are unused).
# "band2": one image per thread block in two bands (block 2, 18 x 18: 10 x 18 conv1 rows per band,
# `dense_band_body`, 12 warps). type -> (kernel, images per block, bottleneck rows, threads)
# "slide": one image per thread block in sliding bands of TH = 144 / W rows (blocks 0 and 1),
# bottleneck in a ring of TH + 2 image rows (`dense_slide_body`, 10 warps).
# "slide2": like slide with the first step doing TH + 1 rows at once (block 2, 18 x 18, TH = 6, 8
# warps).
# "tile": one layer at a time (one launch per layer), conv1 + conv2 fused in 9 x 18 output rectangles
# of all images with the halo recomputed (11 x 20 = 220 conv1 rows out of 224, `dense_tile_body`, 14
# warps, persistent): the bottleneck does not go to DRAM (blocks 0 and 1).
FUSED = {"img3": ("dense_img3", 3, 256, 512), "band2": ("dense_band2", 1, 192, 384),
         "slide": ("dense_slide", 1, 160, 320), "slide2": ("dense_slide4", 1, 128, 256),
         "tile": ("dense_tile7", 1, 224, 448)}
TILE_HW = (9, 18)                   # output rectangle of "tile"


def fused_shared(groups_max: int, rows: int = 256) -> int:
    """Bytes of shared memory of `dense_img*` / `dense_band*`: bottleneck rows x 128, staging, fp16
    prologue.
    """
    return rows * 128 + max(2 * (rows * 32 + 128 * 32), 2 * 9216) + groups_max * 16 * 8


# Our conv2: kernel -> (8-row chunks per warp, warps); the rectangle shapes tried for every image
# side (TH, TW, kernel). The first one is the default.
# "q*": one thread block per rectangle (`conv2_q*`); "p*": persistent, one block per SM with resident
# weights (`conv2p_m*`, 8 warps). q4w4 16x8: 1.02 ms per layer in block 1.
# "q*u": like q* with the 9 taps unrolled (conv2 at ~39% of the IMMA peak).
C2_KERNELS = {"q3w8": (3, 8), "q4w8": (4, 8), "q6w8": (6, 8), "q3w4": (3, 4), "q4w4": (4, 4),
              "q5w4": (5, 4), "p2": (2, 8), "p3": (3, 8), "p4": (4, 8), "q4w4u": (4, 4),
              "q3w8u": (3, 8),
              # `conv2_k16_body` (blocks of 16 channels, more thread blocks per SM)
              "k4w9": (4, 9), "k4w6": (4, 6), "k3w9": (3, 9), "k3w7": (3, 7), "k3w6": (3, 6),
              "k3w4": (3, 4),
              # persistent with resident weights and unrolled taps (`conv2p_body<MW, true>`, 8 warps)
              "p2u": (2, 8), "p3u": (3, 8), "p4u": (4, 8),
              # two-stage prefetch: `conv2_body<.., PD = 2>`, `conv2p_body<.., 2>`
              "q4w4v": (4, 4), "q3w8v": (3, 8), "p3v": (3, 8), "p4v": (4, 8),
              # planar bottleneck
              "p3u_pl": (3, 8), "p4u_pl": (4, 8), "q4w4v_pl": (4, 4),
              # fixed shape, halo in two planes, constant addresses (`conv2t_body`): "t<TH>x<TW>", persistent like
              # the "p*" ones
              "t8x24": (3, 8), "t8x24_pl": (3, 8), "t18x12_pl": (4, 8), "t9x18_pl": (3, 8),
              "t16x16_pl": (4, 8), "t8x32_pl": (4, 8)}
# Per image side at 288 px: 72 16x8 q4w4 1.00 ms, 36 12x12 q5w4 0.257, 18 18x18 q6w8 0.0765, 9 9x9
# q3w4 0.0323: the first of every list is the default, the rest are alternatives.
C2_CANDIDATES = {
    72: [(16, 8, "q4w4"), (16, 8, "q4w4u"), (24, 8, "k4w6"), (24, 8, "p3u"), (8, 24, "p3u"),
         (16, 8, "q4w4v"), (8, 24, "p3v"), (24, 8, "q3w8v")],
    36: [(18, 12, "q4w8"), (9, 12, "q4w4u"), (18, 12, "k3w9"), (18, 12, "p4u"), (12, 16, "p3u"),
         (9, 12, "q4w4v"), (18, 12, "p4v"), (12, 12, "q3w8v")],
    18: [(6, 18, "q4w4u"), (18, 18, "q6w8"), (9, 18, "k3w7"), (9, 18, "p3u"), (6, 18, "p2u"),
         (6, 18, "q4w4v"), (9, 18, "q3w8v"), (9, 18, "p3v")],
    9: [(9, 9, "q4w4u"), (9, 9, "q3w4"), (9, 9, "k3w4")],
}


def trans_w_shared(bm: int, bn: int, groups: int) -> int:
    """Bytes of shared memory of `trans_w_body<bm, bn>`: double stage A + B (32 fp16 channels per row) or
    the output [bm][bn + 16], plus the fp16 prologue table.
    """
    return max(2 * (bm + bn) * 64, bm * (bn + 16)) + groups * 16 * 8


def c2_tshape(kern: str) -> tuple[int, int] | None:
    """(TH, TW) of the fixed-shape conv2 kernels ("t8x24", "t8x24_pl"), None for the others."""
    if not kern.startswith("t"):
        return None
    th, tw = kern[1:].split("_")[0].split("x")
    return int(th), int(tw)


def c2_valid(th: int, tw: int, kern: str) -> bool:
    mw, warps = C2_KERNELS[kern]
    if kern.startswith("t"):
        return (th, tw) == c2_tshape(kern) and c2_shared(th, tw, kern) <= 64 * 1024
    if kern.startswith("k"):
        nt = warps * 32
        hpc = -(-512 // nt)
        ok = th * tw <= warps * mw * 8 and (th + 2) * (tw + 2) <= min(hpc * nt, 512)
    else:
        ok = th * tw <= warps * mw * 8 and 2 * (th + 2) * (tw + 2) <= 4 * warps * 32
    return ok and c2_shared(th, tw, kern) <= 64 * 1024


def _halo_bytes(th: int, tw: int) -> int:
    return -(-(th + 2) * (tw + 2) * 32 // 128) * 128


def c2_shared(th: int, tw: int, kern: str = "q") -> int:
    if kern.startswith("t"):                    # `conv2t_body`: weights, 2 buffers x 2 planes (sO reuses buffer 1)
        plane = -(-(th + 2) * (tw + 2) * 16 // 128) * 128 + 64
        return 36864 + 4 * plane + 128
    if kern.startswith("p"):
        return 36864 + 2 * _halo_bytes(th, tw) + 128 + th * tw * 32
    if kern.startswith("k"):                    # `conv2_k16_body`: halo and weights at 16 channels
        hb = -(-(th + 2) * (tw + 2) * 16 // 128) * 128
        return max(2 * hb + 2 * 4608 + 128, th * tw * 32)
    return 2 * _halo_bytes(th, tw) + 2 * 9216 + 128


def c2_default(side: int) -> tuple[int, int, str]:
    if side in C2_CANDIDATES:
        return C2_CANDIDATES[side][0]
    return (min(side, 12), min(side, 8), "q3w4") if side * side > 96 else (side, side, "q3w4")


# --- calibration ----------------------------------------------------------------------------
@torch.no_grad()
def calibrate(encoder, batches) -> dict:
    """Maxima over `batches` (tensors (N, in, H, W) on the device of the float32 encoder): per channel
    |x| of every store segment (input of every block = output of pool0 or of the transition; output of
    every conv2) and per tensor the conv inputs (relu1 -> conv1, relu2 -> conv2, the transition relu).
    Keys: `seg0.{b}`, `conv2.{b}.{l}` (per-channel lists), `in1.{b}.{l}`, `in2.{b}.{l}`, `tin.{b}`
    (numbers).
    """
    f = encoder.features
    chan: dict[str, torch.Tensor] = {}
    tens: dict[str, float] = {}
    hooks = []

    def chan_max(key):
        def hook(_m, _i, y):
            v = y.detach().float().abs().amax(dim=(0, 2, 3))
            chan[key] = v if key not in chan else torch.maximum(chan[key], v)
        return hook

    def in_max(key):
        def hook(_m, inp):
            v = float(inp[0].detach().float().abs().max())
            tens[key] = max(tens.get(key, 0.0), v)
        return hook

    hooks.append(f.pool0.register_forward_hook(chan_max("seg0.0")))
    b = 0
    while hasattr(f, f"denseblock{b + 1}"):
        db = getattr(f, f"denseblock{b + 1}")
        for li, layer in enumerate(db.values()):
            hooks.append(layer.conv2.register_forward_hook(chan_max(f"conv2.{b}.{li}")))
            hooks.append(layer.conv1.register_forward_pre_hook(in_max(f"in1.{b}.{li}")))
            hooks.append(layer.conv2.register_forward_pre_hook(in_max(f"in2.{b}.{li}")))
        if hasattr(f, f"transition{b + 1}"):
            tr = getattr(f, f"transition{b + 1}")
            hooks.append(tr.conv.register_forward_pre_hook(in_max(f"tin.{b}")))
            hooks.append(tr.pool.register_forward_hook(chan_max(f"seg0.{b + 1}")))
        b += 1
    try:
        for x in batches:
            encoder(x)
    finally:
        for h in hooks:
            h.remove()
    out: dict = {k: v.cpu().numpy().astype(np.float64).tolist() for k, v in chan.items()}
    out.update(tens)
    return out


# --- plan -----------------------------------------------------------------------------------
@dataclass
class SLayer:
    c_in: int
    a1: np.ndarray              # norm1 as an affine [c_in] (float64)
    b1: np.ndarray
    s1: float                   # u8 scale of the conv1 input
    w1: np.ndarray              # int8 [128, c_in], sign of norm2 inside
    w1s: np.ndarray             # [128] > 0: weight scale times |a2|
    bias1: np.ndarray           # [128] = b2 of norm2
    s2: float                   # int8 scale of the conv2 input (TensorRT)
    w2: np.ndarray              # int8 [32, 128, 3, 3]
    w2s: np.ndarray             # [32]
    sig: np.ndarray             # [32] per-channel scale of the output in the store


@dataclass
class SBlock:
    c0: int
    h: int
    w: int
    sig0: np.ndarray            # [c0] per-channel scale of the block input
    layers: list[SLayer]
    ta: np.ndarray | None = None        # transition: norm as an affine [c_tot]
    tb: np.ndarray | None = None
    ts: float | None = None             # u8 scale of the conv input
    tw: np.ndarray | None = None        # int8 [c_tot // 2, c_tot]
    tws: np.ndarray | None = None       # [c_tot // 2]

    @property
    def c_tot(self) -> int:
        return self.c0 + GROWTH * len(self.layers)

    @property
    def sig_all(self) -> np.ndarray:
        return np.concatenate([self.sig0] + [L.sig for L in self.layers])


@dataclass
class SPlan:
    img: int
    in_ch: int
    w0: np.ndarray              # stem with norm0 folded in, float32 [64, in_ch, 7, 7]
    b0: np.ndarray
    blocks: list[SBlock] = field(default_factory=list)
    a5: np.ndarray | None = None
    b5: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return self.blocks[-1].c_tot


def _sig(v, floor: float = 1e-6) -> np.ndarray:
    return np.maximum(np.asarray(v, dtype=np.float64), floor) / QS


def make_plan(encoder, cal: dict, img: int) -> SPlan:
    """The plan of `encoder` (`knee.models.densenet.DenseNet` in eval, without `fold_bn`) with the
    thresholds `cal` from `calibrate`.
    """
    f = encoder.features
    a0, b0 = d8._affine(f.norm0)
    w0 = f.conv0.weight.detach().double().numpy() * a0[:, None, None, None]
    plan = SPlan(img=img, in_ch=int(f.conv0.weight.shape[1]), w0=w0.astype(np.float32),
                 b0=b0.astype(np.float32))
    h = w = img // 4
    c = int(f.conv0.weight.shape[0])
    k = 1
    sig0 = _sig(cal["seg0.0"])
    while hasattr(f, f"denseblock{k}"):
        db = getattr(f, f"denseblock{k}")
        bi = k - 1
        layers = []
        for li, layer in enumerate(db.values()):
            name = f"features.denseblock{k}.denselayer{li + 1}"
            if not hasattr(layer.norm2, "running_var"):
                raise ValueError(f"{name}: norm2 already folded (fold_bn): the whole model is needed")
            a1, b1 = d8._affine(layer.norm1)
            a2, b2 = d8._affine(layer.norm2)
            q1, s1w = d8._wq(layer.conv1.weight)
            sign = np.where(a2 < 0, -1, 1).astype(np.int8)
            mag = np.abs(a2)
            dead = mag < 1e-30
            q1 = (q1 * sign[:, None, None, None]).astype(np.int8)
            q1[dead] = 0
            w1s = np.where(dead, s1w, s1w * mag)
            q2, s2w = d8._wq(layer.conv2.weight)
            layers.append(SLayer(
                c_in=c + GROWTH * li, a1=a1, b1=b1,
                s1=max(float(cal[f"in1.{bi}.{li}"]), 1e-12) / QU,
                w1=q1.reshape(BOTTLENECK, -1), w1s=w1s.astype(np.float64),
                bias1=b2.astype(np.float64),
                s2=max(float(cal[f"in2.{bi}.{li}"]), 1e-12) / QS, w2=q2,
                w2s=s2w.astype(np.float64), sig=_sig(cal[f"conv2.{bi}.{li}"])))
        blk = SBlock(c0=c, h=h, w=w, sig0=sig0, layers=layers)
        c = blk.c_tot
        if hasattr(f, f"transition{k}"):
            tr = getattr(f, f"transition{k}")
            ta, tb = d8._affine(tr.norm)
            tq, tsw = d8._wq(tr.conv.weight)
            blk.ta, blk.tb = ta, tb
            blk.ts = max(float(cal[f"tin.{bi}"]), 1e-12) / QU
            blk.tw, blk.tws = tq.reshape(tq.shape[0], -1), tsw.astype(np.float64)
            sig0 = _sig(cal[f"seg0.{bi + 1}"])
            c //= 2
            h //= 2
            w //= 2
        plan.blocks.append(blk)
        k += 1
    a5, b5 = d8._affine(f.norm5)
    plan.a5, plan.b5 = a5, b5
    return plan


def _chunked(w: np.ndarray) -> np.ndarray:
    """[cout, cin] -> [cin // 32, cout, 32] (the weights of a group of contiguous channels)."""
    cout, cin = w.shape
    return np.ascontiguousarray(w.reshape(cout, cin // GROUP, GROUP).transpose(1, 0, 2))


def _abh(ab: np.ndarray) -> np.ndarray:
    """The prologue [c, 2] (a', b') in fp16 for `pro4h`: [c // 2, 4] int16 (fp16 bits) = (A_2p, A_2p+1,
    B_2p, B_2p+1), i.e. A and B of a pair of channels as two half2.
    """
    h = ab.astype(np.float16)
    return np.ascontiguousarray(np.stack([h[0::2, 0], h[1::2, 0], h[0::2, 1], h[1::2, 1]],
                                         1)).view(np.int16)


def _abh_ab(abh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) per channel, float64, from the `_abh` table."""
    h = np.ascontiguousarray(abh).view(np.float16).astype(np.float64)
    return h[:, :2].reshape(-1), h[:, 2:].reshape(-1)


def kernel_params(plan: SPlan) -> dict[str, np.ndarray]:
    """The tables of the kernels and engines, float32 / int8:
    - `L{b}_{l}_ab` [c_in, 2]: conv1 prologue (a sigma / (255 s1), b / (255 s1));
    - `L{b}_{l}_gd` [128, 2]: conv1 epilogue (w1s s1 / (127 s2), bias1 / (127 s2));
    - `L{b}_{l}_w1` [G, 128, 32] int8; `L{b}_{l}_c2` [32]: conv2 weight scale / sigma (in the engine,
      with a Q of scale 1 on the output);
    - `T{b}_ab` [c_tot, 2], `T{b}_gd` [c_tot // 2, 2] (k = ts tws / (4 sigma'), 0), `T{b}_w`;
    - `F_ab` [n_features, 2]: norm5 with sigma folded in (a5 sigma, b5);
    - `S_w0`, `S_b0`: the stem with 1 / sigma0 folded in (INT8 output from the engine, Q of scale 1);
      `S_w16`, `S_bias`, `S_inv`: for `stem_q` (fp16 weights [64, 168] without 1/sigma,
      `stem_weights16`; bias b0; 1/sigma0);
    - our conv2 (`conv2_q*`): `L{b}_{l}_gdu` [128, 2] (u8 bottleneck, scale s2u = amax / 255),
      `L{b}_{l}_w2k` [4, 9, 32, 32] int8 (block of 32 channels, tap, output, input),
      `L{b}_{l}_c2u` [32] = s2u w2s / sigma.
    """
    out: dict[str, np.ndarray] = {}
    for bi, blk in enumerate(plan.blocks):
        sig = blk.sig_all
        for li, L in enumerate(blk.layers):
            s = sig[:L.c_in]
            out[f"L{bi}_{li}_ab"] = np.stack([L.a1 * s / (QU * L.s1),
                                              L.b1 / (QU * L.s1)], 1).astype(np.float32)
            out[f"L{bi}_{li}_abh"] = _abh(out[f"L{bi}_{li}_ab"])
            out[f"L{bi}_{li}_gd"] = np.stack([L.w1s * L.s1 / (QS * L.s2),
                                              L.bias1 / (QS * L.s2)], 1).astype(np.float32)
            out[f"L{bi}_{li}_w1"] = _chunked(L.w1)
            out[f"L{bi}_{li}_c2"] = (L.w2s / L.sig).astype(np.float32)
            s2u = L.s2 * QS / QU
            out[f"L{bi}_{li}_gdu"] = np.stack([L.w1s * L.s1 / (QU * s2u),
                                               L.bias1 / (QU * s2u)], 1).astype(np.float32)
            w2 = L.w2.reshape(GROWTH, 4, GROUP, 9)                 # [o, c, i, tap]
            out[f"L{bi}_{li}_w2k"] = np.ascontiguousarray(w2.transpose(1, 3, 0, 2))
            out[f"L{bi}_{li}_c2u"] = (s2u * L.w2s / L.sig).astype(np.float32)
        if blk.ta is not None:
            nxt = plan.blocks[bi + 1].sig0
            out[f"T{bi}_ab"] = np.stack([blk.ta * sig / (QU * blk.ts),
                                         blk.tb / (QU * blk.ts)], 1).astype(np.float32)
            out[f"T{bi}_gd"] = np.stack([0.25 * blk.ts * blk.tws / nxt,
                                         np.zeros_like(nxt)], 1).astype(np.float32)
            out[f"T{bi}_w"] = _chunked(blk.tw)
            out[f"T{bi}_abh"] = _abh(out[f"T{bi}_ab"])
            # `trans_f16` (mean before the conv, fp16): fp16 weights and k for the mean
            out[f"T{bi}_wh"] = _chunked(blk.tw).astype(np.float16)
            out[f"T{bi}_gdf"] = np.stack([blk.ts * blk.tws / nxt, np.zeros_like(nxt)],
                                         1).astype(np.float32)
    last = plan.blocks[-1]
    out["F_ab"] = np.stack([plan.a5 * last.sig_all, plan.b5], 1).astype(np.float32)
    s0 = plan.blocks[0].sig0
    out["S_w0"] = (plan.w0.astype(np.float64) / s0[:, None, None, None]).astype(np.float32)
    out["S_b0"] = (plan.b0.astype(np.float64) / s0).astype(np.float32)
    # `stem_q`: fp16 weights WITHOUT 1/sigma (with small sigmas they would overflow), 1/sigma afterwards
    out["S_w16"] = stem_weights16(plan.w0)
    out["S_bias"] = plan.b0.astype(np.float32)
    out["S_inv"] = (1.0 / s0).astype(np.float32)
    # `stem_i8`: int8 weights per output channel, u8 pixels = rint(255 x):
    # q = rint(clamp(acc k + c, 0, 127)), k = sw / (255 sigma0), c = b0 / sigma0
    w0 = plan.w0.astype(np.float64)
    sw = np.maximum(np.abs(w0).reshape(w0.shape[0], -1).max(1), 1e-12) / QS
    w8 = np.clip(np.rint(w0 / sw[:, None, None, None]), -QS, QS)
    out["S_w8"] = stem_weights8(w8)
    out["S_w8f"] = w8.astype(np.float32)
    out["S_kc"] = np.concatenate([sw / (QU * s0), plan.b0 / s0]).astype(np.float32)
    return out


def stem_weights16(w: np.ndarray) -> np.ndarray:
    """The stem weights [64, 3, 7, 7] in the `stem_q` layout: [64, 168] fp16 with
    k = (channel * 7 + ky) * 8 + kx', kx' = kx + 1 (kx' = 0 is zero).
    """
    cout, cin, kh, kw = w.shape
    out = np.zeros((cout, cin, kh, kw + 1), np.float32)
    out[..., 1:] = w
    return np.ascontiguousarray(out.reshape(cout, -1).astype(np.float16))


def stem_weights8(w8: np.ndarray) -> np.ndarray:
    """The int8 stem weights [64, 3, 7, 7] in the `stem_i8` layout: [64, 176] with k as in
    `stem_weights16` (21 groups (channel, ky) of 8 kx', kx' = 0 null) and 8 trailing zeros.
    """
    cout, cin, kh, kw = w8.shape
    out = np.zeros((cout, cin, kh, kw + 1), np.int8)
    out[..., 1:] = w8
    flat = out.reshape(cout, -1)
    pad = np.zeros((cout, 176 - flat.shape[1]), np.int8)
    return np.ascontiguousarray(np.concatenate([flat, pad], 1))


# --- PyTorch reference ------------------------------------------------------------------
def _rq(t: torch.Tensor, lo: float, hi: float, quant: bool) -> torch.Tensor:
    return torch.round(t).clamp(lo, hi) if quant else t


@torch.no_grad()
def emulate(plan: SPlan, x: torch.Tensor, quant: bool = True, dtype=torch.float64,
            par: dict | None = None, trace: dict | None = None,
            conv2: str = "trt", stem: str = "ours",
            f16_layers: set | None = None) -> torch.Tensor:
    """The `Dense8SEncoder` pipeline in PyTorch (`dtype`, float64 = exact integer sums), on the device of
    `x` (N, in, H, W): features (N, n_features) float32.

    `quant=False`: the same tables and integer weights but without roundings and saturations (the
    ReLUs stay): it must give the encoder with only the weights quantized, and is the check of the scale
    formulas. `trace`: {(block, j): float store (integer values)} like `Dense8SEncoder.trace` (j = 0
    block input, j >= 1 output of layer j-1). `conv2`: "trt" (int8 bottleneck [0, 127], output
    [-128, 127]) or "ours" (`conv2_q*`: u8 bottleneck [0, 255], output [-127, 127]). `stem`: "i8" /
    "i8w" = `stem_i8` (u8 pixels, int8 weights, `stem_q_i8`), otherwise the floating-point stem.
    `f16_layers`: the layers (b, l) with the fp16 prologue (`conv1_h128h`, `pro4h`): v = fp16(A q + B)
    saturated, u = rint(255 v).
    """
    ours = conv2 == "ours"
    QB = QU if ours else QS
    par = kernel_params(plan) if par is None else par
    trace = {} if trace is None else trace
    dev = x.device
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)  # noqa: E731
    col = lambda a: T(a).view(1, -1, 1, 1)                                         # noqa: E731
    x = x.to(dtype)
    if stem in ("i8", "i8w"):
        q = stem_q_i8(x, par, quant).to(dtype)
    else:
        y = F.relu(F.conv2d(x, T(par["S_w0"]), T(par["S_b0"]), stride=2, padding=3))
        q = _rq(F.max_pool2d(y, 3, 2, 1), -QS, QS, quant)
    for bi, blk in enumerate(plan.blocks):
        store = [q]
        trace[(bi, 0)] = q
        for li, L in enumerate(blk.layers):
            X = torch.cat(store, 1)
            if quant and f16_layers and (bi, li) in f16_layers:
                a16, b16 = _abh_ab(par[f"L{bi}_{li}_abh"])
                t = col(a16) * X + col(b16)
                u = torch.round(t.to(torch.float16).to(dtype).clamp(0, 1) * QU)
            else:
                ab = par[f"L{bi}_{li}_ab"]
                t = col(ab[:, 0]) * X + col(ab[:, 1])
                u = torch.round(t.clamp(0, 1) * QU) if quant else t.clamp_min(0) * QU
            acc = F.conv2d(u, T(L.w1.astype(np.float64)).view(BOTTLENECK, -1, 1, 1))
            gd = par[f"L{bi}_{li}_gdu" if ours else f"L{bi}_{li}_gd"]
            t = col(gd[:, 0]) * acc + col(gd[:, 1])
            v = torch.round(t.clamp(0, 1) * QB) if quant else t.clamp_min(0) * QB
            acc2 = F.conv2d(v, T(L.w2.astype(np.float64)), padding=1)
            c2 = (torch.as_tensor(par[f"L{bi}_{li}_c2u"], dtype=torch.float32) if ours else
                  torch.as_tensor(L.s2 * par[f"L{bi}_{li}_c2"], dtype=torch.float32))
            seg = acc2 * c2.to(dtype).to(dev).view(1, -1, 1, 1)
            if quant:
                seg = torch.round(seg).clamp(-QS if ours else -128, QS)
            store.append(seg)
            trace[(bi, li + 1)] = seg
        X = torch.cat(store, 1)
        if blk.ta is not None:
            if quant and f16_layers and ("t", bi) in f16_layers:
                a16, b16 = _abh_ab(par[f"T{bi}_abh"])
                t = col(a16) * X + col(b16)
                u = torch.round(t.to(torch.float16).to(dtype).clamp(0, 1) * QU)
            else:
                ab = par[f"T{bi}_ab"]
                t = col(ab[:, 0]) * X + col(ab[:, 1])
                u = torch.round(t.clamp(0, 1) * QU) if quant else t.clamp_min(0) * QU
            acc = F.conv2d(u, T(blk.tw.astype(np.float64)).view(blk.tw.shape[0], -1, 1, 1))
            s4 = F.avg_pool2d(acc, 2) * 4
            k = col(par[f"T{bi}_gd"][:, 0])
            q = torch.round(k * s4).clamp(-QS, QS) if quant else k * s4
        else:
            ab = par["F_ab"]
            z = F.relu(col(ab[:, 0]) * X + col(ab[:, 1]))
            return z.mean(dim=(2, 3)).float()
    raise RuntimeError("plan without a final block")


def stem_q_i8(x: torch.Tensor, par: dict, quant: bool = True) -> torch.Tensor:
    """The output of `stem_i8` (in0 of block 1, integer values [0, 127]) as float64 on the device of x:
    u = rint(255 clamp(x, 0, 1)), acc = exact conv(u, w8), q = rint(clamp(fp32(acc k + c), 0, 127))
    (the kernel's fma rounds once to fp32), 3x3/2 max-pool. `quant=False`: no roundings (the ReLU and
    the int8 weights stay).
    """
    dev = x.device
    x = x.double().clamp(0, 1) * QU
    u = torch.round(x) if quant else x
    w8 = torch.as_tensor(par["S_w8f"], dtype=torch.float64, device=dev)
    acc = F.conv2d(u, w8, stride=2, padding=3)
    kc = torch.as_tensor(par["S_kc"], dtype=torch.float64, device=dev)
    n = w8.shape[0]
    t = acc * kc[:n].view(1, -1, 1, 1) + kc[n:].view(1, -1, 1, 1)
    if quant:
        t = torch.round(t.float().double().clamp(0, QS))
    else:
        t = t.clamp_min(0)
    return F.max_pool2d(t, 3, 2, 1)


@torch.no_grad()
def weight_quant_encoder(encoder):
    """A copy of the encoder with the conv weights (except the stem) rounded as in the plan (int8 per
    output channel): the reference for `emulate(quant=False)`.
    """
    import copy
    enc = copy.deepcopy(encoder).eval()
    for name, mod in enc.features.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and name != "conv0":
            q, s = d8._wq(mod.weight)
            mod.weight.copy_(torch.as_tensor(q.astype(np.float32) * s.reshape(-1, 1, 1, 1)))
    return enc


# --- TensorRT engines -------------------------------------------------------------------------
def _conv2_net(builder, L: SLayer, c2: np.ndarray, n: int, h: int, w: int,
               int8_out: bool) -> d8._Net:
    """3x3 conv2: INT8 kCHW32 input (norm2 + ReLU + Q of conv1, scale s2), weights with the per-channel
    scale divided by sigma; INT8 kCHW32 output after a Q of scale 1 (= the store's q), or fp16
    (fallback, then `quant_seg`).
    """
    nb = d8._Net(builder)
    trt = nb.trt
    x = nb.int8_input((n, BOTTLENECK, h, w))
    xd = nb.net.add_dequantize(x, nb.scalar(L.s2)).get_output(0)
    wsc = (L.w2s / L.sig).astype(np.float32) if int8_out else L.w2s.astype(np.float32)
    conv = nb.conv(xd, nb.qdq_weights(L.w2, wsc), GROWTH, 3, pad=1)
    if int8_out:
        y = nb.net.add_quantize(conv.get_output(0), nb.scalar(1.0)).get_output(0)
        y.name = "y"
        nb.net.mark_output(y)
        y.dtype = trt.int8
        y.allowed_formats = 1 << int(trt.TensorFormat.CHW32)
    else:
        nb.output(conv.get_output(0))
    return nb


def _stem_net(builder, par: dict, plan: SPlan, n: int, int8_out: bool) -> d8._Net:
    """fp16 stem (7x7 conv with norm0 and 1/sigma0 folded in, ReLU, max-pool) with INT8 kCHW32 output
    after a Q of scale 1 (= the store's q); `int8_out=False`: fp16 output before 1/sigma (fallback, then
    `quant_seg` with the max-pool).
    """
    nb = d8._Net(builder)
    trt = nb.trt
    x = nb.net.add_input("x", trt.float16, (n, plan.in_ch, plan.img, plan.img))
    x.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    w0 = par["S_w0"] if int8_out else plan.w0
    b0 = par["S_b0"] if int8_out else plan.b0
    c = nb.conv(x, np.ascontiguousarray(w0, np.float32), w0.shape[0], 7, stride=2, pad=3,
                bias=np.ascontiguousarray(b0, np.float32))
    r = nb.net.add_activation(c.get_output(0), trt.ActivationType.RELU)
    layers = [c, r]
    last = r
    if int8_out:
        last = nb.net.add_pooling_nd(r.get_output(0), trt.PoolingType.MAX, (3, 3))
        last.stride_nd = (2, 2)
        last.padding_nd = (1, 1)
        layers.append(last)
    for layer in layers:
        layer.precision = trt.float16
        layer.set_output_type(0, trt.float16)
    if int8_out:
        y = nb.net.add_quantize(last.get_output(0), nb.scalar(1.0)).get_output(0)
        y.name = "y"
        nb.net.mark_output(y)
        y.dtype = trt.int8
        y.allowed_formats = 1 << int(trt.TensorFormat.CHW32)
    else:
        nb.output(last.get_output(0))
    return nb


def engine_names(plan: SPlan, conv2: str = "trt", stem: str = "trt") -> list[str]:
    names = ["stem"] if stem == "trt" else []
    if conv2 == "trt":
        for bi, blk in enumerate(plan.blocks):
            names += [f"b{bi}l{li}" for li in range(len(blk.layers))]
    return names


def build_engines(plan: SPlan, n: int, timing_cache: bytes | None = None, log=print,
                  only: list[str] | None = None, int8_out: bool = True,
                  stem_int8: bool = True, conv2: str = "trt", stem: str = "trt"
                  ) -> tuple[dict[str, bytes], bytes, dict]:
    """The engines of `plan` with fixed batch `n`: {name: engine}, timing cache, info. With
    `conv2="ours"` only the stem (the conv2 layers are `conv2_q*`).
    """
    if not engine_names(plan, conv2, stem):
        return {}, bytes(timing_cache or b""), {"build_s": {}, "mb": {}, "total_s": 0.0}
    from .trt import logger
    trt = d8._trt()
    par = kernel_params(plan)
    builder = trt.Builder(logger())
    cfg0 = builder.create_builder_config()
    cache = cfg0.create_timing_cache(timing_cache or b"")
    out: dict[str, bytes] = {}
    info: dict = {"build_s": {}, "mb": {}}
    t_all = time.perf_counter()
    for name in engine_names(plan, conv2, stem):
        if only is not None and name not in only:
            continue
        if name == "stem":
            nb = _stem_net(builder, par, plan, n, stem_int8)
        else:
            bi, li = (int(v) for v in name[1:].split("l"))
            blk = plan.blocks[bi]
            nb = _conv2_net(builder, blk.layers[li], par[f"L{bi}_{li}_c2"], n, blk.h, blk.w,
                            int8_out)
        cfg = builder.create_builder_config()
        cfg.set_timing_cache(cache, True)
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        cfg.set_flag(trt.BuilderFlag.FP16)
        if name == "stem":
            cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            if stem_int8:
                cfg.set_flag(trt.BuilderFlag.INT8)
        else:
            cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if hasattr(cfg, "max_aux_streams"):
            cfg.max_aux_streams = 0
        t = time.perf_counter()
        blob = builder.build_serialized_network(nb.net, cfg)
        if blob is None:
            raise RuntimeError(f"dense8s: engine {name} not built")
        out[name] = bytes(blob)
        info["build_s"][name] = round(time.perf_counter() - t, 2)
        info["mb"][name] = round(len(out[name]) / 2**20, 2)
        del nb
    info["total_s"] = round(time.perf_counter() - t_all, 1)
    log(f"dense8s: {len(out)} engines in {info['total_s']} s, "
        f"{sum(info['mb'].values()) if out else 0:.1f} MiB")
    return out, bytes(cache.serialize()), info


# --- package -------------------------------------------------------------------------------
def kernels_sha() -> str:
    import hashlib
    return hashlib.sha1(KERNELS.read_bytes()).hexdigest()[:12]


def default_bm(rows: int) -> int:
    """Rows per thread block of the conv1: 64 where blocks of 128 would be too few for the 40 SMs of the
    T4 (block 4: 88 x 81 pixels = 56 blocks).
    """
    return 128 if rows >= 160 * 128 else 64


# wide transitions (`trans_w*`, 16 warps, fp16 prologue, same result as "f16h"): type -> (rows,
# output channels per thread block). "w1" only with cout a multiple of 256 (t1, t2).
TRANS_W = {"w0": (256, 128), "w1": (128, 256)}
TRANS_F16H = ("hh", "f16h", *TRANS_W)          # transitions with the fp16 prologue (table T*_abh)


def _tkind(v) -> str:
    """The transition kernel: "q128" / "q256" (`trans_q*`, conv at full resolution and sum of the 4
    accumulators), "f16" / "f16h" (`trans_f16*`, mean before the conv in fp16), "w0" / "w1"
    (`TRANS_W`). Also accepts the old integers 128 / 256.
    """
    v = str(v)
    return v if v in ("q128", "q256", "f16", "f16h", "h", "hh", *TRANS_W) else f"q{int(v)}"


def pack_meta(plan: SPlan, n: int, int8_out: bool = True, stem_int8: bool = True,
              bm: dict | None = None, conv2: str = "trt", c2cfg: dict | None = None,
              stem: str = "trt", tbm: dict | None = None, fused: dict | None = None) -> dict:
    """`conv2`: "trt" (engines) or "ours" (`conv2_q*`, rectangles `c2cfg` {side: [TH, TW, kernel]},
    default `c2_default`).
    """
    blocks = []
    for bi, b in enumerate(plan.blocks):
        rows = n * b.h * b.w
        cfg = list((c2cfg or {}).get(b.h) or (c2cfg or {}).get(str(b.h)) or c2_default(b.h))
        blocks.append({"c0": b.c0, "h": b.h, "w": b.w, "L": len(b.layers),
                       "trans": b.ta is not None,
                       "bm": [_c1kind((bm or {}).get(f"b{bi}l{li}", default_bm(rows)))
                              for li in range(len(b.layers))],
                       "c2": [int(cfg[0]), int(cfg[1]), str(cfg[2])],
                       "tbm": _tkind((tbm or {}).get(f"t{bi}", "q128")),
                       "fused": (fused or {}).get(f"b{bi}") or (fused or {}).get(bi)})
    return {"format": PACK_FORMAT, "kind": KIND, "n": int(n), "img": plan.img,
            "in_ch": plan.in_ch, "kernels_sha": kernels_sha(), "n_features": plan.n_features,
            "stem_c": int(plan.w0.shape[0]), "conv2_int8": bool(int8_out),
            "stem_int8": bool(stem_int8), "conv2": conv2, "stem": stem, "blocks": blocks}


def save_pack(path: Path, plan: SPlan, n: int, engines: dict[str, bytes], extra: dict | None = None,
              cubin: bytes | None = None, int8_out: bool = True, stem_int8: bool = True,
              bm: dict | None = None, conv2: str = "trt", c2cfg: dict | None = None,
              stem: str = "trt", tbm: dict | None = None, fused: dict | None = None) -> None:
    meta = pack_meta(plan, n, int8_out, stem_int8, bm, conv2, c2cfg, stem, tbm, fused) | {
        "engines": sorted(engines), **(extra or {})}
    arrays = {"meta": np.frombuffer(json.dumps(meta).encode(), np.uint8)}
    if cubin:
        arrays["cubin"] = np.frombuffer(cubin, np.uint8)
    arrays |= {f"eng_{k}": np.frombuffer(v, np.uint8) for k, v in engines.items()}
    par = pack_params(plan, conv2, int8_out, stem_int8)
    with open(path, "wb") as fh:
        np.savez(fh, **arrays, **{f"par_{k}": v for k, v in par.items()})


def pack_params(plan: SPlan, conv2: str = "trt", int8_out: bool = True,
                stem_int8: bool = True) -> dict[str, np.ndarray]:
    """The tables the runner uses (those of the other conv2 are left out)."""
    par = kernel_params(plan)
    drop = ("_gd", "_c2") if conv2 == "ours" else ("_gdu", "_w2k", "_c2u")
    par = {k: v for k, v in par.items() if not (k.startswith("L") and k.endswith(drop))}
    # S_inv is always present (kernel_params): needed by stem_q and by the fp16 + quant_seg fallback
    if not int8_out:
        for bi, blk in enumerate(plan.blocks):
            for li, L in enumerate(blk.layers):
                par[f"L{bi}_{li}_inv"] = (1.0 / L.sig).astype(np.float32)
    return par


def load_pack(path: Path) -> dict:
    return d8.load_pack(path)


def is_pack(pack: dict) -> bool:
    return isinstance(pack, dict) and pack.get("meta", {}).get("kind") == KIND


# --- runner ----------------------------------------------------------------------------------
_MOD: dict = {}


def kernel_cubin(arch: str) -> bytes:
    from . import cudart
    if arch not in _MOD:
        t = time.perf_counter()
        _MOD[arch] = cudart.compile_source(KERNELS.read_text(encoding="utf-8"), arch)
        _MOD["compile_s"] = round(time.perf_counter() - t, 3)
    return _MOD[arch]


def kernel_module(dev: int, cubin: bytes | None = None, sha: str | None = None):
    """The kernel module on GPU `dev`, once per GPU: from the package cubin if it comes from the same
    source (`sha`), otherwise compiled here with NVRTC.
    """
    if cubin and sha is not None and sha != kernels_sha():
        _MOD["pack_error"] = f"package cubin built from another source ({sha})"
        cubin = None
    from . import cudart
    key = ("mod", dev)
    if key in _MOD:
        return _MOD[key]
    major, minor = torch.cuda.get_device_capability(dev)
    with torch.cuda.device(dev):
        torch.cuda.current_stream(dev)
        mod = None
        if cubin:
            try:
                mod = cudart.Module(cubin, dev)
                _MOD["from"] = "pack"
            except cudart.CudaError as exc:
                _MOD["pack_error"] = str(exc)[:200]
        if mod is None:
            mod = cudart.Module(kernel_cubin(f"sm_{major}{minor}"), dev)
            _MOD["from"] = "nvrtc"
    _MOD[key] = mod
    return mod


# Graph inputs shared among the engines of the same GPU (the seeds of an ensemble): the windows of a
# TTA view go through every seed, and each one copied them into its own buffer (175 MB per call of 352
# images). Here they are copied once: whoever finds the same source in the buffer (same root tensor,
# same version, same view) skips the copy. {(gpu, shape, dtype): {"buf": tensor, "src": key, "root":
# root tensor kept alive}}.
_SHARED_IN: dict = {}


def _shared_in(dev: int, shape: tuple, dtype) -> dict:
    key = (dev, tuple(shape), dtype)
    s = _SHARED_IN.get(key)
    if s is None:
        s = {"buf": torch.zeros(shape, dtype=dtype, device=f"cuda:{dev}"), "src": None,
             "root": None}
        _SHARED_IN[key] = s
    return s


_NOVER = itertools.count()


def _src_key(x: torch.Tensor) -> tuple:
    """(root, source key). Tensors created under `torch.inference_mode` (infer.py) have no version counter
    (`x._version` raises): the key is always new, i.e. a copy at every call (~1 ms for 352 images against
    ~130 for the encoder).
    """
    root = x
    while root._base is not None:
        root = root._base
    ver = ("inf", next(_NOVER)) if x.is_inference() else x._version
    return root, (id(root), x.data_ptr(), ver, tuple(x.shape), tuple(x.stride()))


class Dense8SEncoder:
    """The dense8s engine on one GPU: `enc(x)` with x (N, in, img, img) fp16/float on GPU `dev`, N a
    multiple of the package's `n`; returns (N, n_features) float32 in its own buffer. Interface of
    `knee.dense8.Dense8Encoder` (own stream, `max_inflight`, `capture(n)`).
    """

    def __init__(self, pack: dict, dev: int, n_total: int, max_inflight: int | None = None) -> None:
        from .cudart import Ptr
        self.Ptr = Ptr
        meta = pack["meta"]
        if meta.get("kind") != KIND or meta.get("format") != PACK_FORMAT:
            raise ValueError(f"package {meta.get('kind')} format {meta.get('format')}, "
                             f"expected {KIND} {PACK_FORMAT}")
        self.meta, self.dev = meta, dev
        self.nb = int(meta["n"])
        # a block can be larger than the batch (graphs of 352 images): incomplete blocks are padded
        # (`_padded`), the extra rows are computed and discarded
        self.n_out = -(-n_total // self.nb) * self.nb
        self._pad: torch.Tensor | None = None
        self.max_inflight = max_inflight
        self.inflight: collections.deque = collections.deque()
        self.graphs: dict[int, tuple] = {}
        self.calls = {"graph": 0, "plain": 0}
        n, d = self.nb, f"cuda:{dev}"
        t0 = time.perf_counter()
        with torch.cuda.device(dev):
            self.stream = torch.cuda.Stream(dev)
            self.engines, self.ctxs, self.fmts = {}, {}, {}
            self.scratch, self.scratch_mb = None, 0.0
            self.stem_ours = meta.get("stem", "trt") in STEM_OURS
            if meta["engines"]:
                # TensorRT only for the engines present (TensorRT stem and/or conv2)
                from .trt import logger
                trt = d8._trt()
                self.rt = trt.Runtime(logger())
                for name in meta["engines"]:
                    eng = self.rt.deserialize_cuda_engine(pack["engines"][name])
                    if eng is None:
                        raise RuntimeError(f"dense8s: engine {name} cannot be deserialized")
                    self.engines[name] = eng
                self._contexts(trt)
                self._check_io(trt)
            elif set(engine_names_meta(meta)):
                raise RuntimeError(f"dense8s: engines required {engine_names_meta(meta)[:3]}...")
            self.deser_s = time.perf_counter() - t0
            self.mod = kernel_module(dev, pack.get("cubin"), meta.get("kernels_sha"))
            self.k = {name: self.mod.function(name)
                      for name in ("conv1_q128", "conv1_q64", "conv1_u128", "conv1_u64",
                                   "conv1_u256", "trans_q128", "trans_q256", "trans_f16",
                                   "trans_h", "trans_hh", "trans_f16h", "trans_w0", "trans_w1",
                                   "final_q", "quant_seg", "stem_q", "stem_q2", "stem_i8",
                                   "stem_i8w")}
            for name in ("conv1_u256", "trans_q256", "trans_f16", "trans_h", "trans_hh",
                         "trans_f16h", "trans_w0", "trans_w1"):
                self.k[name].set_max_shared(64 * 1024)
            for kind in (*C1_WS, *C1_HI):
                self.k[f"conv1_{kind}"] = self.mod.function(f"conv1_{kind}")
                self.k[f"conv1_{kind}"].set_max_shared(64 * 1024)
            for name, *_ in FUSED.values():
                self.k[name] = self.mod.function(name)
                self.k[name].set_max_shared(64 * 1024)
            self.k["stem_q"].set_max_shared(STEM_SMEM)
            self.k["stem_q2"].set_max_shared(STEM2_SMEM)
            for name in ("stem_i8", "stem_i8w"):
                self.k[name].set_max_shared(STEM_I8_SMEM)
            self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
            self.k.update({f"conv2_{k}": self.mod.function(
                f"conv2p_m{k[1:]}" if k.startswith("p") else f"conv2_{k}") for k in C2_KERNELS})
            for k in C2_KERNELS:
                if k.startswith(("p", "t")):
                    self.k[f"conv2_{k}"].set_max_shared(64 * 1024)
            self.ours = meta.get("conv2", "trt") == "ours"
            self._buffers(pack, n, d)
            self.out = torch.empty((self.n_out, meta["n_features"]), dtype=torch.float32,
                                   device=d)
            self.in_dtype = torch.float16
            self.in_shape = (n, meta["in_ch"], meta["img"], meta["img"])
        self.load_s = time.perf_counter() - t0

    # -- initialisation --
    def _contexts(self, trt) -> None:
        strat = getattr(trt, "ExecutionContextAllocationStrategy", None)
        sizes = {k: int(getattr(e, "device_memory_size_v2", 0) or e.device_memory_size)
                 for k, e in self.engines.items()}
        self.scratch = None
        if strat is not None:
            self.scratch = torch.empty(max(1, max(sizes.values())), dtype=torch.uint8,
                                       device=f"cuda:{self.dev}")
        for k, e in self.engines.items():
            if self.scratch is not None:
                ctx = e.create_execution_context(strat.USER_MANAGED)
                if hasattr(ctx, "set_device_memory"):
                    ctx.set_device_memory(self.scratch.data_ptr(), sizes[k])
                else:
                    ctx.device_memory = self.scratch.data_ptr()
            else:
                ctx = e.create_execution_context()
            if ctx is None:
                raise RuntimeError(f"dense8s: context of engine {k} not created")
            self.ctxs[k] = ctx
        self.scratch_mb = round(max(sizes.values()) / 2**20, 1)

    def _check_io(self, trt) -> None:
        meta = self.meta
        blocks = meta["blocks"]
        self.fmts: dict[str, d8.Fmt] = {}
        want_names = set(engine_names_meta(meta))
        if set(self.engines) != want_names:
            raise RuntimeError(f"dense8s: engines {sorted(self.engines)[:4]}..., expected "
                               f"{sorted(want_names)[:4]}...")
        for k, e in self.engines.items():
            fx = e.get_tensor_format("x")
            want = trt.TensorFormat.LINEAR if k == "stem" else trt.TensorFormat.CHW32
            if fx != want:
                raise RuntimeError(f"dense8s: engine {k} with input {fx}")
            int8 = meta["stem_int8"] if k == "stem" else meta["conv2_int8"]
            fy, ty = e.get_tensor_format("y"), e.get_tensor_dtype("y")
            if int8:
                if fy != trt.TensorFormat.CHW32 or ty != trt.int8:
                    raise RuntimeError(f"dense8s: engine {k} with output {fy} {ty}, expected "
                                       f"INT8 kCHW32")
                continue
            if k == "stem":
                c, h, w = meta["stem_c"], meta["img"] // 2, meta["img"] // 2
            else:
                b = blocks[int(k[1:].split("l")[0])]
                c, h, w = GROWTH, b["h"], b["w"]
            self.fmts[k] = d8.tensor_fmt(trt, e, "y", c, h, w)

    def _buffers(self, pack: dict, n: int, d: str) -> None:
        meta, par = self.meta, pack["par"]
        blocks = meta["blocks"]
        dev = torch.device(d)
        # Two alternating arenas: block b reads its own and the transition writes into the other.
        sizes = [0, 0]
        lay = []
        for bi, b in enumerate(blocks):
            hw = b["h"] * b["w"]
            g0 = b["c0"] // GROUP
            in0 = n * g0 * hw * GROUP
            seg = b["L"] * n * hw * GROUP
            lay.append((bi % 2, in0, seg))
            sizes[bi % 2] = max(sizes[bi % 2], in0 + seg)
        self.arenas = [torch.empty(max(s, 16), dtype=torch.int8, device=d) for s in sizes]
        self.arena_mb = round(sum(sizes) / 2**20, 1)
        self.in0 = [self.arenas[a].data_ptr() for a, _, _ in lay]
        self.seg = [self.arenas[a].data_ptr() + in0 for a, in0, _ in lay]
        hw_max = max(b["h"] * b["w"] for b in blocks)
        self.bott = torch.empty(n * BOTTLENECK * hw_max, dtype=torch.int8, device=d)
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
        self.par = {k: T(v) for k, v in par.items() if k not in ("S_w0", "S_b0", "S_w8f")}
        self.ybuf = self.s0 = None
        if not self.stem_ours and not meta["stem_int8"]:
            f = self.fmts["stem"]
            self.s0 = torch.empty(f.elems(n) * f.itemsize(), dtype=torch.uint8, device=d)
        if not meta["conv2_int8"] and not self.ours:
            self.ybuf = torch.empty(max(self.fmts[k].elems(n) * self.fmts[k].itemsize()
                                        for k in self.fmts if k != "stem"),
                                    dtype=torch.uint8, device=d)
        # Precomputed launches: (kernel, grid, block, arguments, shared).
        self.plan_l: list = []
        self.fused: dict = {}          # block -> launch of the fused block (instead of the layers)
        self._lp: dict = {}            # LayerP tables on the GPU for the fused blocks
        for bi, b in enumerate(blocks):
            hw = b["h"] * b["w"]
            g0 = b["c0"] // GROUP
            rows = n * hw
            layers = []
            th, tw, kern = tuple(b.get("c2", c2_default(b["h"])))
            bms = list(b["bm"])
            if (self.ours and (th, tw, kern) in ((16, 8, "p4u_pl"), (16, 8, "q4w4u"))
                    and b["h"] % 16 == 0 and b["w"] % 16 == 0):
                th, tw, kern = 16, 16, "t16x16_pl"
                bms = ["w64_pl" if str(k) == "u128hs" else k for k in bms]
            if (bi == 0 and self.ours and (th, tw, kern) == (16, 16, "t16x16_pl")
                    and b["h"] % 8 == 0 and b["w"] % 32 == 0):
                th, tw, kern = 8, 32, "t8x32_pl"
            pl1 = {str(k).endswith(PLANAR) for k in bms}
            if not b.get("fused") and (len(pl1) > 1 or pl1 != {kern.endswith(PLANAR)}):
                raise ValueError(f"block {bi}: planar bottleneck only with conv1 and conv2 both "
                                 f"'{PLANAR}' ({sorted(map(str, bms))[:2]}, {kern})")
            for li in range(b["L"]):
                slot = self.seg[bi] + li * n * hw * GROUP
                c1 = self._c1_item(bi, li, bms[li])
                c2 = None
                if self.ours:
                    c2 = self._c2_item(bi, li, th, tw, kern)
                else:
                    ctx = self.ctxs[f"b{bi}l{li}"]
                    ctx.set_tensor_address("x", self.bott.data_ptr())
                    ctx.set_tensor_address("y", slot if meta["conv2_int8"]
                                           else self.ybuf.data_ptr())
                layers.append((c1, c2))
            trans = None
            if b["trans"]:
                G = g0 + b["L"]
                cout = b["c0"] + GROWTH * b["L"]
                cout //= 2
                arg = KArg(self.in0[bi], self.seg[bi], self.par[f"T{bi}_w"].data_ptr(),
                           self.par[f"T{bi}_ab"].data_ptr(), self.par[f"T{bi}_gd"].data_ptr(),
                           self.in0[bi + 1], g0, G, n, hw, b["w"], rows, cout,
                           cout // GROUP)
                trans = self._trans_item(bi, _tkind(b.get("tbm", "q128")))
            self.plan_l.append((layers, trans))
            if b.get("fused"):
                self.fused[bi] = self._fused_item(bi, b["fused"])
        if not self.stem_ours:
            ctx = self.ctxs["stem"]
            ctx.set_tensor_address("y", self.in0[0] if meta["stem_int8"] else self.s0.data_ptr())

    def _fused_item(self, bi: int, kind: str):
        """The launch of the fused block `kind` (see `FUSED`) for block bi: table of the layers on the GPU
        (w1, fp16 prologue, u8 epilogue, conv2 weights and scale, groups read).
        """
        name, per, rows, nth = FUSED[kind]
        n = self.nb
        b = self.meta["blocks"][bi]
        if kind == "tile":
            th, tw = TILE_HW
            ok = b["h"] % th == 0 and b["w"] % tw == 0 and (th + 2) * (tw + 2) <= rows
        elif kind == "slide":
            th = 144 // b["w"]
            ok = b["h"] % th == 0 and th * b["w"] <= rows
        elif kind == "slide2":
            th = rows // b["w"] - 1
            ok = th > 0 and b["h"] % th == 0 and (th + 1) * b["w"] <= rows
        elif kind.startswith("img"):
            th = 0
            ok = per * b["h"] * b["w"] < rows
        else:
            th = b["h"] // 2
            ok = (th + 1) * b["w"] < rows
        if not self.ours or not ok:
            raise ValueError(f"fused block {kind}: requires our conv2 and a geometry that "
                             f"fits in {rows} rows")
        g0 = b["c0"] // GROUP
        lps = (LayerP * b["L"])()
        for li in range(b["L"]):
            P = lambda k: self.par[f"L{bi}_{li}_{k}"].data_ptr()  # noqa: E731
            lps[li] = LayerP(P("w1"), P("abh"), P("gdu"), P("w2k"), P("c2u"), g0 + li, 0)
        raw = bytes(lps)
        lp = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(f"cuda:{self.dev}")
        self._lp[bi] = lp
        if kind == "tile":
            # one launch per layer: the table of layer li, rectangles of all images
            tiles = n * (b["h"] // th) * (b["w"] // tw)
            shared = fused_shared(g0 + b["L"] - 1, rows)
            return [(self.k[name], (min(tiles, self.sms), 1), (nth,),
                     TileArg(self.in0[bi], self.seg[bi], lp.data_ptr() + li * ctypes.sizeof(LayerP),
                             n, b["h"], b["w"], g0, li, th, tw, 0), shared)
                    for li in range(b["L"])]
        arg = ImgArg(self.in0[bi], self.seg[bi], lp.data_ptr(), n, b["h"], b["w"], g0, b["L"], th)
        groups = -(-n // per)
        if kind.startswith("slide"):
            # ring (TH + 2) * W rows + the zero row, staging, fp16 prologue
            shared = ((th + 2) * b["w"] + 1) * 128 + max(2 * (rows * 32 + 128 * 32), 2 * 9216) \
                + (g0 + b["L"] - 1) * 16 * 8
        else:
            shared = fused_shared(g0 + b["L"] - 1, rows)
        return (self.k[name], (min(groups, self.sms), 1), (nth,), arg, shared)

    def _c1_item(self, bi: int, li: int, kind):
        """The launch of the conv1 of layer (bi, li) with kernel `kind` (see `_c1kind`)."""
        n = self.nb
        b = self.meta["blocks"][bi]
        hw = b["h"] * b["w"]
        g0 = b["c0"] // GROUP
        G, rows = g0 + li, n * hw
        gd = self.par[f"L{bi}_{li}_gdu" if self.ours else f"L{bi}_{li}_gd"]
        arg = KArg(self.in0[bi], self.seg[bi], self.par[f"L{bi}_{li}_w1"].data_ptr(),
                   self.par[f"L{bi}_{li}_ab"].data_ptr(), gd.data_ptr(),
                   self.bott.data_ptr(), g0, G, n, hw, b["w"], rows, BOTTLENECK, 0)
        kind = _c1kind(kind)
        if kind in C1_HI:
            if not self.ours:
                raise ValueError(f"conv1 {kind}: only with the u8 bottleneck of our conv2")
            bm, nth = C1_HI[kind]
            if kind in C1_F16:
                arg.ab = self.par[f"L{bi}_{li}_abh"].data_ptr()
            if kind in C1_PERSIST:
                # persistent (`conv1p_body`): 2 thread blocks per SM loop over the row blocks; no output in shared
                # memory
                return (self.k[f"conv1_{kind}"], (min(-(-rows // bm), 2 * self.sms), 1), (nth,),
                        arg, 2 * (bm * 32 + 128 * 32) + G * 16 * 8)
            return (self.k[f"conv1_{kind}"], (-(-rows // bm), 1), (nth,), arg,
                    gemm_shared(bm, 2, G))
        if isinstance(kind, str):
            if not self.ours:
                raise ValueError(f"conv1 {kind}: only with the u8 bottleneck of our conv2")
            bm, np_, nmma = C1_WS[kind]
            if kind in C1_F16:
                arg.ab = self.par[f"L{bi}_{li}_abh"].data_ptr()
            return (self.k[f"conv1_{kind}"], (min(self.sms, -(-rows // bm)), 1),
                    (32 * (nmma + np_),), arg, ws_shared(bm, G))
        kname = ("conv1_u" if self.ours else "conv1_q") + str(kind)
        return (self.k[kname], (-(-rows // kind), 1), (512 if kind == 256 else THREADS,), arg,
                gemm_shared(kind, 2 if self.ours else 0, G))

    def _trans_item(self, bi: int, kind: str):
        """The launch of the transition of block bi with kernel `kind` ("q128", "q256", "f16")."""
        n = self.nb
        b = self.meta["blocks"][bi]
        hw = b["h"] * b["w"]
        g0 = b["c0"] // GROUP
        G = g0 + b["L"]
        cout = (b["c0"] + GROWTH * b["L"]) // 2
        rows = n * hw
        f16 = kind in ("f16", "f16h", "h", "hh", *TRANS_W)
        arg = KArg(self.in0[bi], self.seg[bi],
                   self.par[f"T{bi}_wh" if f16 else f"T{bi}_w"].data_ptr(),
                   self.par[f"T{bi}_abh" if kind in TRANS_F16H else f"T{bi}_ab"].data_ptr(),
                   self.par[f"T{bi}_gdf" if f16 else f"T{bi}_gd"].data_ptr(),
                   self.in0[bi + 1], g0, G, n, hw, b["w"], rows, cout, cout // GROUP)
        if kind in TRANS_W:
            bm, bn = TRANS_W[kind]
            if cout % bn:
                raise ValueError(f"transition {kind}: cout {cout} not a multiple of {bn}")
            return (self.k[f"trans_{kind}"], (cout // bn, -(-(rows // 4) // bm)), (512,), arg,
                    trans_w_shared(bm, bn, G))
        if kind in ("h", "hh"):
            # `trans_hi_body`: 512 threads, 2 blocks per SM, stages of 16 channels
            return (self.k[f"trans_{kind}"], (cout // 128, -(-(rows // 4) // 128)), (512,), arg,
                    max(2 * (128 * 32 + 128 * 32), 128 * 144) + G * 32 * 8)
        if f16:
            return (self.k[f"trans_{kind}"], (cout // 128, -(-(rows // 4) // 128)), (THREADS,),
                    arg, 2 * (128 * 64 + 128 * 64) + G * 32 * 8)
        tbm = int(kind[1:])
        return (self.k[f"trans_q{tbm}"], (cout // 128, -(-rows // tbm)),
                (512 if tbm == 256 else THREADS,), arg, gemm_shared(tbm, 1, G))

    def _c2_item(self, bi: int, li: int, th: int, tw: int, kern: str):
        """The launch of `conv2_<kern>` for layer (bi, li) with th x tw rectangles."""
        n = self.nb
        b = self.meta["blocks"][bi]
        h, w = b["h"], b["w"]
        if not c2_valid(th, tw, kern) or (kern.startswith("t") and (h % th or w % tw)):
            raise ValueError(f"conv2 {kern} with {th}x{tw} rectangles is not valid")
        txn, tyn = -(-w // tw), -(-h // th)
        slot = self.seg[bi] + li * n * h * w * GROUP
        arg = C2Arg(self.bott.data_ptr(), self.par[f"L{bi}_{li}_w2k"].data_ptr(),
                    self.par[f"L{bi}_{li}_c2u"].data_ptr(), slot, n, h, w, th, tw, txn)
        warps = C2_KERNELS[kern][1]
        if kern.startswith(("p", "t")):
            grid = (min(self.sms, txn * tyn * n), 1)
        else:
            grid = (txn * tyn, n)
        return (self.k[f"conv2_{kern}"], grid, (warps * 32,), arg, c2_shared(th, tw, kern))

    # -- one pass --
    def _launch(self, item, st: int) -> None:
        if isinstance(item, list):                 # fused block layer by layer ("tile")
            for it in item:
                self._launch(it, st)
            return
        k, grid, block, arg, shared = item
        k.launch(grid, block, [arg], st, shared=shared)

    def _conv2(self, bi: int, li: int, c2, st: int) -> None:
        if c2 is not None:
            self._launch(c2, st)
            return
        self.ctxs[f"b{bi}l{li}"].execute_async_v3(st)
        if not self.meta["conv2_int8"]:
            self._conv2_fp16(bi, li, st)

    def _conv2_fp16(self, bi: int, li: int, st: int) -> None:
        """Fallback: fp16 conv2 output -> the store's q (`quant_seg`)."""
        P, n = self.Ptr, self.nb
        b = self.meta["blocks"][bi]
        hw = b["h"] * b["w"]
        slot = self.seg[bi] + li * n * hw * GROUP
        self.k["quant_seg"].launch(((2 * n * hw + 127) // 128, 1), (128,),
                                   [P(self.ybuf.data_ptr()), self.fmts[f"b{bi}l{li}"], 0, b["h"],
                                    b["w"], n, b["h"], b["w"], 0, 1,
                                    self.par[f"L{bi}_{li}_inv"], P(slot)], st)

    def _stem_fp16(self, st: int) -> None:
        P, n = self.Ptr, self.nb
        b = self.meta["blocks"][0]
        side = self.meta["img"] // 2
        self.k["quant_seg"].launch(((2 * n * b["h"] * b["w"] + 127) // 128, b["c0"] // GROUP),
                                   (128,),
                                   [P(self.s0.data_ptr()), self.fmts["stem"], 2, side, side, n,
                                    b["h"], b["w"], 0, b["c0"] // GROUP, self.par["S_inv"],
                                    P(self.in0[0])], st)

    def _stem(self, x_ptr: int, st: int) -> None:
        """The stem: our `stem_q` kernel (persistent, one block per SM) or the TensorRT engine (INT8 output,
        or fp16 + `quant_seg` with the max-pool). The final 512 px submission uses the TensorRT stem: our
        stem works on 9-pixel rectangles and 128 is not divisible by 9.
        """
        if self.stem_ours:
            P, n, img = self.Ptr, self.nb, self.meta["img"]
            tiles = n * (img // 4 // STEM_TP) ** 2
            name, nth = STEM_OURS[self.meta["stem"]]
            if name in ("stem_q", "stem_q2"):
                args, shared = [P(x_ptr), self.par["S_w16"], self.par["S_bias"],
                                self.par["S_inv"], P(self.in0[0]), n, img, img], \
                    STEM_SMEM if name == "stem_q" else STEM2_SMEM
            else:
                args, shared = [P(x_ptr), self.par["S_w8"], self.par["S_kc"], P(self.in0[0]),
                                n, img, img], STEM_I8_SMEM
            self.k[name].launch((min(self.sms, tiles),), (nth,), args, st, shared=shared)
            return
        ctx = self.ctxs["stem"]
        ctx.set_tensor_address("x", x_ptr)
        ctx.execute_async_v3(st)
        if not self.meta["stem_int8"]:
            self._stem_fp16(st)

    def _pass(self, x_ptr: int, f_ptr: int, events: list | None = None) -> None:
        st = self.stream.cuda_stream
        mark = (lambda tag: events.append((tag, _record(self.stream)))) if events is not None \
            else (lambda tag: None)
        meta = self.meta
        blocks = meta["blocks"]
        mark("start")
        self._stem(x_ptr, st)
        mark("stem")
        n = self.nb
        for bi, b in enumerate(blocks):
            layers, trans = self.plan_l[bi]
            if bi in self.fused:
                self._launch(self.fused[bi], st)
                mark(f"fused_{bi}")
            else:
                for li, (c1, c2) in enumerate(layers):
                    self._launch(c1, st)
                    mark(f"conv1_{bi}")
                    self._conv2(bi, li, c2, st)
                    mark(f"conv2_{bi}")
            if trans is not None:
                self._launch(trans, st)
                mark(f"trans{bi}")
            else:
                P = self.Ptr
                hw = b["h"] * b["w"]
                g0 = b["c0"] // GROUP
                self.k["final_q"].launch((n, g0 + b["L"]), (128,),
                                         [P(self.in0[bi]), g0, P(self.seg[bi]), n, hw,
                                          self.par["F_ab"], P(f_ptr), meta["n_features"]], st)
                mark("final")

    def _padded(self, chunk: torch.Tensor) -> torch.Tensor:
        """One block of `nb` images: `chunk` itself if it is full, otherwise copied to the head of an engine
        buffer (the rest holds old images: outputs computed and discarded).
        """
        if chunk.shape[0] == self.nb:
            return chunk
        if self._pad is None:
            self._pad = torch.zeros((self.nb, *chunk.shape[1:]), dtype=self.in_dtype,
                                    device=f"cuda:{self.dev}")
        self._pad[:chunk.shape[0]].copy_(chunk)
        return self._pad

    def _enqueue(self, xf: torch.Tensor, out_off: int = 0) -> None:
        for i, chunk in enumerate(xf.split(self.nb)):
            if self.max_inflight:
                while len(self.inflight) >= self.max_inflight:
                    self.inflight.popleft().synchronize()
            self._pass(self._padded(chunk).data_ptr(), self.out[out_off + i * self.nb].data_ptr())
            if self.max_inflight:
                ev = torch.cuda.Event(blocking=True)
                ev.record(self.stream)
                self.inflight.append(ev)

    def capture(self, n: int) -> float:
        """One CUDA graph for every block of `nb` images of a batch of `n` (as in dense8)."""
        n = -(-n // self.nb) * self.nb
        if n > self.out.shape[0]:
            raise ValueError(f"graph for {n} images: the engine does {self.nb} per block, "
                             f"the output holds {self.out.shape[0]}")
        t = time.perf_counter()
        with torch.cuda.device(self.dev):
            shared = _shared_in(self.dev, (n, *self.in_shape[1:]), self.in_dtype)
            static_in = shared["buf"]
            shared["src"] = shared["root"] = None
            with torch.cuda.stream(self.stream):
                self._enqueue(static_in[:self.nb])
            self.stream.synchronize()
            self.inflight.clear()
            graphs = []
            for i, chunk in enumerate(static_in.split(self.nb)):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=self.stream, capture_error_mode="thread_local"):
                    self._pass(chunk.data_ptr(), self.out[i * self.nb].data_ptr())
                graphs.append(g)
            self.stream.synchronize()
        self.graphs[n] = (graphs, shared)
        return time.perf_counter() - t

    def graph_in_buf(self, n: int) -> torch.Tensor | None:
        """The static input buffer `[:n]` of the graph captured for `n` images, or None."""
        gs = self.graphs.get(n)
        if gs is None:
            k = -(-n // self.nb)
            for m_, (g_, s_) in sorted(self.graphs.items()):
                if m_ >= k * self.nb:
                    gs = (g_[:k], s_)
                    break
        return None if gs is None else gs[1]["buf"][:n]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cur = torch.cuda.current_stream(self.dev)
        self.stream.wait_stream(cur)
        n = x.shape[0]
        with torch.cuda.stream(self.stream):
            gs = self.graphs.get(n)
            if gs is None:
                # fewer images than the captured batch (short final batch, windows skipped by
                # `infer._model_logits`): the first ceil(n / nb) graphs, which read static_in[:n] and beyond (old
                # rows, outputs discarded)
                k = -(-n // self.nb)
                for m_, (g_, s_) in sorted(self.graphs.items()):
                    if m_ >= k * self.nb:
                        gs = (g_[:k], s_)
                        break
            if gs is not None:
                graphs, shared = gs
                buf = shared["buf"][:n]
                if (x.data_ptr() == buf.data_ptr() and x.shape == buf.shape
                        and x.stride() == buf.stride()):
                    shared["src"] = shared["root"] = None
                else:
                    root, key = _src_key(x)
                    key = key + (n,)
                    if shared["src"] != key:
                        buf.copy_(x)
                        shared["src"], shared["root"] = key, root
                        self.calls["copy"] = self.calls.get("copy", 0) + 1
                for g in graphs:
                    if self.max_inflight:
                        while len(self.inflight) >= self.max_inflight:
                            self.inflight.popleft().synchronize()
                    g.replay()
                    if self.max_inflight:
                        ev = torch.cuda.Event(blocking=True)
                        ev.record(self.stream)
                        self.inflight.append(ev)
                self.calls["graph"] += 1
            else:
                self._enqueue(x.to(self.in_dtype, memory_format=torch.contiguous_format))
                self.calls["plain"] += 1
        cur.wait_stream(self.stream)
        return self.out[:n]

    # -- diagnostics --
    def store(self, bi: int, j: int) -> torch.Tensor:
        """Store segment (bi, j) as a float tensor (N, c, h, w) of integers (j = 0 block input, j >= 1
        output of layer j-1).
        """
        b = self.meta["blocks"][bi]
        n, h, w = self.nb, b["h"], b["w"]
        hw = h * w
        g0 = b["c0"] // GROUP
        if j == 0:
            a = self.arenas[bi % 2]
            off = self.in0[bi] - a.data_ptr()
            t = a[off:off + n * g0 * hw * GROUP].view(n, g0, hw, GROUP)
            return t.permute(0, 1, 3, 2).reshape(n, g0 * GROUP, h, w).float()
        a = self.arenas[bi % 2]
        off = self.seg[bi] - a.data_ptr() + (j - 1) * n * hw * GROUP
        t = a[off:off + n * hw * GROUP].view(n, hw, GROUP)
        return t.permute(0, 2, 1).reshape(n, GROUP, h, w).float()

    @torch.no_grad()
    def trace(self, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """One pass over the first `n` images stopping after every layer: the store {(block, j): tensor} like
        `emulate(trace=...)`, and the features. For debugging.
        """
        xf = self._padded(x[:self.nb].to(self.in_dtype).contiguous())
        out: dict = {}
        st = self.stream.cuda_stream
        meta = self.meta
        with torch.cuda.device(self.dev), torch.cuda.stream(self.stream):
            f = torch.zeros((self.nb, meta["n_features"]), dtype=torch.float32,
                            device=f"cuda:{self.dev}")
            self._stem(xf.data_ptr(), st)
            for bi, b in enumerate(meta["blocks"]):
                self.stream.synchronize()
                out[(bi, 0)] = self.store(bi, 0).clone()
                layers, trans = self.plan_l[bi]
                if bi in self.fused:
                    self._launch(self.fused[bi], st)
                    self.stream.synchronize()
                    for li in range(len(layers)):
                        out[(bi, li + 1)] = self.store(bi, li + 1).clone()
                for li, (c1, c2) in enumerate(layers if bi not in self.fused else []):
                    self._launch(c1, st)
                    self._conv2(bi, li, c2, st)
                    self.stream.synchronize()
                    out[(bi, li + 1)] = self.store(bi, li + 1).clone()
                if trans is not None:
                    self._launch(trans, st)
                else:
                    P = self.Ptr
                    g0 = b["c0"] // GROUP
                    self.k["final_q"].launch((self.nb, g0 + b["L"]), (128,),
                                             [P(self.in0[bi]), g0, P(self.seg[bi]), self.nb,
                                              b["h"] * b["w"], self.par["F_ab"], f,
                                              meta["n_features"]], st)
            self.stream.synchronize()
        return out, f

    def profile(self, x: torch.Tensor, reps: int = 3) -> dict:
        """GPU milliseconds per item (stem, conv1/conv2 per block, transitions, final) of a pass of `n`
        images, from CUDA events between launches.
        """
        xf = self._padded(x[:self.nb].to(self.in_dtype).contiguous())
        tot: dict[str, float] = collections.defaultdict(float)
        with torch.cuda.device(self.dev), torch.cuda.stream(self.stream):
            for r in range(reps + 1):
                ev: list = []
                self._pass(xf.data_ptr(), self.out.data_ptr(), ev)
                self.stream.synchronize()
                if r == 0:
                    continue
                for (_, a), (tag, b) in zip(ev, ev[1:]):
                    tot[tag] += a.elapsed_time(b)
        return {k: round(v / reps, 3) for k, v in sorted(tot.items())}

    def _time(self, fn, reps: int = 20) -> float:
        """ms per call of `fn` (which launches on the runner's stream): a CUDA graph with `reps` calls,
        replayed once as warm-up and once under events (no CPU in between).
        """
        with torch.cuda.device(self.dev), torch.cuda.stream(self.stream):
            fn()
            self.stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=self.stream, capture_error_mode="thread_local"):
                for _ in range(reps):
                    fn()
            g.replay()
            e0 = _record(self.stream)
            g.replay()
            e1 = _record(self.stream)
            self.stream.synchronize()
        return e0.elapsed_time(e1) / reps

    def time_layers(self, reps: int = 20) -> dict:
        """ms of every conv1 (our kernel), conv2 (engine) and transition, on the current store."""
        st = self.stream.cuda_stream
        out: dict = {}
        for bi, b in enumerate(self.meta["blocks"]):
            layers, trans = self.plan_l[bi]
            for li, (c1, c2) in enumerate(layers):
                out[f"c1 b{bi}l{li}"] = round(self._time(lambda it=c1: self._launch(it, st),
                                                         reps), 4)
                out[f"c2 b{bi}l{li}"] = round(self._time(
                    lambda i=li, c=c2, bb=bi: self._conv2(bb, i, c, st), reps), 4)
            if trans is not None:
                out[f"t{bi}"] = round(self._time(lambda it=trans: self._launch(it, st), reps), 4)
        return out

    def tune_bm(self, reps: int = 20) -> dict:
        """For every conv1 the ms of every candidate kernel (`conv1_u*` with 64 / 128 / 256 rows and, with our
        conv2, the warp-specialised ones `C1_WS`): {name: {type: ms}}.
        """
        st = self.stream.cuda_stream
        # w8, w4h, w2h, x8 never better than w4 / x4, left out of the tuning; conv1 with 32 warps per SM
        # (h128, u128x16, u64x8) 40-70% worse; 64, w4, x4 always behind the fp16 prologue (u128h, u256h).
        # Then: L2 prefetch (*p), steady state without gaps (*hs, *hps), the persistent one (p128), 64 rows
        # with 3 blocks per SM (u64hs)
        kinds = ((128, "u128h", "u256h", "w4f", "u128hp", "u256hp", "u128hs", "u256hs",
                  "u256hps", "p128", "u64hs") if self.ours else (64, 128))
        out: dict = {}
        for bi, b in enumerate(self.meta["blocks"]):
            for li in range(b["L"]):
                res = {}
                for kind in kinds:
                    it = self._c1_item(bi, li, kind)
                    res[kind] = round(self._time(lambda it=it: self._launch(it, st), reps), 4)
                out[f"b{bi}l{li}"] = res
        return out

    def tune_trans(self, reps: int = 20) -> dict:
        """For every transition the ms of the "q128", "q256", "f16" kernels: {"t<b>": {type: ms}}."""
        st = self.stream.cuda_stream
        out: dict = {}
        for bi, b in enumerate(self.meta["blocks"]):
            if self.plan_l[bi][1] is None:
                continue
            res = {}
            cout = (b["c0"] + GROWTH * b["L"]) // 2
            kinds = ["q128", "f16", "f16h"]               # h/hh slower
            kinds += [k for k, (_, bn) in TRANS_W.items() if cout % bn == 0]
            for kind in kinds:
                it = self._trans_item(bi, kind)
                res[kind] = round(self._time(lambda it=it: self._launch(it, st), reps), 4)
            out[f"t{bi}"] = res
        return out

    def tune_c2(self, reps: int = 20) -> dict:
        """For every block the ms of our conv2 (layer 0) with every candidate rectangle:
        {side: {"THxTW kern": ms}}.
        """
        st = self.stream.cuda_stream
        out: dict = {}
        for bi, b in enumerate(self.meta["blocks"]):
            res = {}
            for th, tw, kern in C2_CANDIDATES.get(b["h"], [c2_default(b["h"])]):
                if not c2_valid(th, tw, kern):
                    continue
                it = self._c2_item(bi, 0, th, tw, kern)
                res[f"{th}x{tw} {kern}"] = round(self._time(
                    lambda it=it: self._launch(it, st), reps), 4)
            out[b["h"]] = res
        return out


def f16_layers_meta(meta: dict) -> set:
    """The layers (b, l) of the package with the fp16 prologue (`C1_F16`)."""
    out = {(bi, li) for bi, b in enumerate(meta["blocks"]) for li, k in enumerate(b["bm"])
           if str(k) in C1_F16 or b.get("fused")}
    return out | {("t", bi) for bi, b in enumerate(meta["blocks"])
                  if b.get("tbm") in TRANS_F16H}


def engine_names_meta(meta: dict) -> list[str]:
    names = ["stem"] if meta.get("stem", "trt") == "trt" else []
    if meta.get("conv2", "trt") == "trt":
        for bi, b in enumerate(meta["blocks"]):
            names += [f"b{bi}l{li}" for li in range(b["L"])]
    return names


def _record(stream) -> torch.cuda.Event:
    e = torch.cuda.Event(enable_timing=True)
    e.record(stream)
    return e
