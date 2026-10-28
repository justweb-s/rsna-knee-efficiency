"""dense8: the DenseNet encoder in INT8, with TensorRT only for the convolutions and the rest by hand.
The first INT8 engine, superseded by `knee.dense8s` (still imported for comparisons).

In the chained TensorRT engine (`knee.quant.ChainDenseNet`) the INT8 convolutions take 41 ms out of
126 per study; 72 ms are BatchNorm+ReLU (39, IScaleLayer) and Quantize (33) in front of every
conv1, over the whole concatenation of the block: two memory passes per layer, a cost that grows
with the square of the number of layers, and TensorRT does not fuse them.

Here the concatenation disappears. Every feature segment (the input of a block or the 32 outputs
of a layer) is read once, right after it is produced, and one of our kernels (`dense8_kernels.cu`,
`fan_*`) writes it already normalised, ReLU'd and quantized into the INT8 input (NC/32HW32) of
EVERY layer that will use it, of the transition and of the final features (norm5, ReLU, mean).
TensorRT does only what it does well: one small engine per layer (1x1 conv1 with norm2 folded in
and ReLU, Q, 3x3 conv2; INT8 kCHW32 input, fp16 output), one per transition and one for the stem
(fp16: 7x7 conv, norm0, ReLU, max-pool). The traffic of the pointwise parts drops from ~8 to ~1
byte per element.

Quantization: the same as the engines before (`knee.quant`), from the same thresholds (`amax` of
`Int8Sim`, `max` scaling): weights per output channel, activations per tensor at the input of every
conv. A deliberate difference: the stem stays fp16 (it sees the pixels). The transition has two
forms (`Plan.trans_mode`): "full" is the encoder's order (INT8 at full resolution, conv, 2x2 mean in
the engine); "pool" takes the 2x2 mean before the 1x1 conv (they are linear and commute: 4x fewer
MACs and bytes) but quantizes the averaged value, and the error grows slightly (the mean of four
roundings is lost; `pool_amax` recalibrates the threshold and recovers part of it).

Parts:
- `make_plan(encoder, amax, img)`: weights, scales and affines from the model;
- `emulate(plan, x, quant)`: the same pipeline in PyTorch (reference and tests);
- `build_engines(plan, n)`: the TensorRT engines (needs the builder, on the GPU);
- `save_pack`/`load_pack`: engines + tables in an .npz;
- `Dense8Encoder(pack, dev, n_total)`: the runner, with the interface of `knee.trt.TrtEncoder`.
"""
from __future__ import annotations

import collections
import ctypes
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

QMAX = 127                      # as in knee.quant: symmetric, scale = amax / 127
GROUP = 32                      # channels per group in NC/32HW32; = the DenseNet growth rate
GROWTH = 32
BOTTLENECK = 128
KERNELS = Path(__file__).with_name("dense8_kernels.cu")
PACK_FORMAT = 1
FAN_THREADS = 128               # threads per block of the fan_dense/fan_trans kernels
# fp16 formats allowed for the engine outputs (the fan kernels read them all, `Fmt`).
OUT_FORMATS = ("LINEAR", "CHW2", "HWC8", "CHW4", "CHW16", "CHW32", "HWC16")


class Fmt(ctypes.Structure):
    """Where element (n, c, s) of a TensorRT output lives: n*sN + (c // V)*sG + s*sS + (c % V)*sC, in
    elements; `f32` 1 if fp32. It is the kernels' `struct Fmt`, passed by value.
    """
    _fields_ = [("sN", ctypes.c_longlong), ("sG", ctypes.c_longlong),
                ("sS", ctypes.c_longlong), ("sC", ctypes.c_longlong),
                ("V", ctypes.c_int), ("f32", ctypes.c_int)]

    def elems(self, n: int) -> int:
        return n * int(self.sN)

    def itemsize(self) -> int:
        return 4 if self.f32 else 2


class Src(ctypes.Structure):
    """The pooling the fan kernels apply to the segment source: `mode` 0 none, 1 2x2 mean ("fan"
    transition), 2 3x3 max with stride 2 and padding 1 (stem with `stem_pool`); `sh` x `sw` the source
    pixels. It is the kernels' `struct Src`.
    """
    _fields_ = [("mode", ctypes.c_int), ("sh", ctypes.c_int), ("sw", ctypes.c_int)]


def tensor_fmt(trt, eng, name: str, c: int, h: int, w: int) -> Fmt:
    """The `Fmt` of engine tensor `name` (c channels, h x w pixels), from the format TensorRT chose: NCHW,
    NC/xHWx (channels vectorised in groups of x), HWCx (channels last, padded to multiples of x).
    """
    fmt = eng.get_tensor_format(name)
    f32 = 1 if eng.get_tensor_dtype(name) == trt.float32 else 0
    hw = h * w
    name_f = str(fmt).split(".")[-1]
    if name_f == "LINEAR":
        return Fmt(c * hw, 0, 1, hw, c, f32)
    k = int(eng.get_tensor_components_per_element(name))
    if name_f.startswith("CHW"):
        groups = -(-c // k)
        return Fmt(groups * hw * k, hw * k, k, 1, k, f32)
    if name_f.startswith("HWC"):
        cpad = -(-c // max(k, 1)) * max(k, 1)
        return Fmt(hw * cpad, 0, cpad, 1, cpad, f32)
    raise RuntimeError(f"dense8: unsupported format {fmt} for output {name}")


def _index(fmt: Fmt, n: int, c: int, h: int, w: int, device) -> torch.Tensor:
    ni = torch.arange(n, device=device).view(n, 1, 1) * int(fmt.sN)
    ci = torch.arange(c, device=device).view(1, c, 1)
    si = torch.arange(h * w, device=device).view(1, 1, h * w) * int(fmt.sS)
    return ni + (ci // fmt.V) * int(fmt.sG) + (ci % fmt.V) * int(fmt.sC) + si


def unpack(buf: torch.Tensor, fmt: Fmt, n: int, c: int, h: int, w: int) -> torch.Tensor:
    """The float tensor (n, c, h, w) from a byte buffer in format `fmt` (debug, tests)."""
    flat = buf.view(torch.float32 if fmt.f32 else torch.float16)
    return flat[_index(fmt, n, c, h, w, buf.device)].float().view(n, c, h, w)


def pack_into(y: torch.Tensor, fmt: Fmt, buf: torch.Tensor) -> None:
    """Writes y (n, c, h, w) into the byte buffer `buf` with format `fmt` (tests)."""
    n, c, h, w = y.shape
    flat = buf.view(torch.float32 if fmt.f32 else torch.float16)
    flat[_index(fmt, n, c, h, w, buf.device)] = y.reshape(n, c, h * w).to(flat.dtype)


# --- plan ---------------------------------------------------------------------------------
@dataclass
class Layer:
    c_in: int
    a1: np.ndarray              # norm1 as an affine (float32, [c_in])
    b1: np.ndarray
    s1: float                   # per-tensor scale of the conv1 input
    w1: np.ndarray              # conv1 with norm2 folded in: int8 [128, c_in, 1, 1]
    w1s: np.ndarray             # per-channel scale (> 0) [128]
    bias1: np.ndarray           # [128]
    s2: float                   # scale of the conv2 input
    w2: np.ndarray              # int8 [32, 128, 3, 3]
    w2s: np.ndarray             # [32]


@dataclass
class Block:
    c0: int
    h: int
    w: int
    layers: list[Layer]
    ta: np.ndarray | None = None        # transition: norm as an affine [c_tot]
    tb: np.ndarray | None = None
    ts: float | None = None             # scale of the conv input
    tw: np.ndarray | None = None        # int8 [c_tot // 2, c_tot, 1, 1]
    tws: np.ndarray | None = None

    @property
    def c_tot(self) -> int:
        return self.c0 + GROWTH * len(self.layers)


@dataclass
class Plan:
    img: int
    in_ch: int
    w0: np.ndarray              # stem with norm0 folded in, float32 [64, in_ch, 7, 7]
    b0: np.ndarray
    blocks: list[Block] = field(default_factory=list)
    a5: np.ndarray | None = None
    b5: np.ndarray | None = None
    # "pool": the transition quantizes after the 2x2 mean and the conv runs at half resolution (4x fewer
    # MACs and bytes); "full": the encoder's order (quantization at full resolution, conv, mean in the
    # engine), closer to the simulation (the mean of the 4 roundings); "fan": like "full" but the 2x2
    # mean is done by the fan kernels while reading the conv output (in the engine the mean cost ~1.6 ms
    # between pooling and reformatting).
    trans_mode: str = "pool"
    # The stem max-pool in the fan kernels (the stem stops at the ReLU): same value, max does not round
    # (1.2 ms of pooling in the engine).
    stem_pool: bool = False

    @property
    def n_features(self) -> int:
        return self.blocks[-1].c_tot


def _affine(bn) -> tuple[np.ndarray, np.ndarray]:
    s = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
    t = bn.bias.detach().double() - bn.running_mean.detach().double() * s
    return s.numpy(), t.numpy()


def _wq(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """INT8 per output channel like `knee.quant.fake_quant_weight`: scale = max|w| / 127."""
    w = w.detach().float()
    scale = w.abs().flatten(1).amax(1).clamp_min(1e-12) / QMAX
    inv = (1.0 / scale).view(-1, *[1] * (w.dim() - 1))
    q = torch.round(w * inv).clamp(-128, 127)
    return q.to(torch.int8).numpy(), scale.numpy().astype(np.float32)


@torch.no_grad()
def pool_amax(encoder, batches) -> dict[str, float]:
    """For every transition the maximum of the 2x2 mean of relu(norm(x)) over `batches`: the threshold of
    its conv input when the mean comes first (here). With the threshold of the non-averaged value
    (`amax` of `Int8Sim`) the quantization step stays the full one while the value shrinks, and the
    mean of the four rounding errors of the original engine is lost (error against float 0.0021
    instead of 0.0017).
    """
    out: dict[str, float] = {}
    hooks = []
    for name, mod in encoder.features.named_children():
        if name.startswith("transition"):
            key = f"features.{name}.conv"

            def hook(_m, _inp, y, key=key):
                v = float(F.avg_pool2d(y.float(), 2).abs().max())
                out[key] = max(out.get(key, 0.0), v)
            hooks.append(mod.relu.register_forward_hook(hook))
    try:
        for x in batches:
            encoder(x)
    finally:
        for h in hooks:
            h.remove()
    return out


def make_plan(encoder, amax: dict[str, float], img: int,
              trans_amax: dict[str, float] | None = None, trans_mode: str = "pool",
              stem_pool: bool = False) -> Plan:
    """The plan of `encoder` (`knee.models.densenet.DenseNet`, in eval, without `fold_bn`) with the
    thresholds `amax` {conv name: max|x|} of `Int8Sim` (`amax_fold<k>.json`). `trans_amax`: the
    thresholds of the transitions after the mean (`pool_amax`) instead of those in `amax`.
    """
    if trans_mode not in ("pool", "full", "fan"):
        raise ValueError(f"trans_mode: 'pool', 'full' or 'fan', not {trans_mode!r}")
    amax = {**amax, **(trans_amax or {})}
    f = encoder.features
    a0, b0 = _affine(f.norm0)
    w0 = f.conv0.weight.detach().double().numpy() * a0[:, None, None, None]
    plan = Plan(img=img, in_ch=int(f.conv0.weight.shape[1]), w0=w0.astype(np.float32),
                b0=b0.astype(np.float32), trans_mode=trans_mode, stem_pool=stem_pool)
    h = w = img // 4
    c = int(f.conv0.weight.shape[0])
    k = 1
    while hasattr(f, f"denseblock{k}"):
        db = getattr(f, f"denseblock{k}")
        layers = []
        for li, layer in enumerate(db.values()):
            name = f"features.denseblock{k}.denselayer{li + 1}"
            if not hasattr(layer.norm2, "running_var"):
                raise ValueError(f"{name}: norm2 already folded (fold_bn): the whole model is needed")
            a1, b1 = _affine(layer.norm1)
            a2, b2 = _affine(layer.norm2)
            q1, s1w = _wq(layer.conv1.weight)
            # norm2 inside conv1: out_c = (sum q x) * s_c * a2_c + b2_c. TensorRT's per-channel scale must be
            # positive: the sign of a2 goes into the weights (|q| <= 127, no -128).
            sign = np.where(a2 < 0, -1, 1).astype(np.int8)
            mag = np.abs(a2)
            dead = mag < 1e-30
            q1 = (q1 * sign[:, None, None, None]).astype(np.int8)
            q1[dead] = 0
            w1s = np.where(dead, s1w, s1w * mag).astype(np.float32)
            q2, s2w = _wq(layer.conv2.weight)
            layers.append(Layer(
                c_in=c + GROWTH * li, a1=a1.astype(np.float32), b1=b1.astype(np.float32),
                s1=max(float(amax[f"{name}.conv1"]), 1e-12) / QMAX, w1=q1, w1s=w1s,
                bias1=b2.astype(np.float32),
                s2=max(float(amax[f"{name}.conv2"]), 1e-12) / QMAX, w2=q2, w2s=s2w))
        blk = Block(c0=c, h=h, w=w, layers=layers)
        c = blk.c_tot
        if hasattr(f, f"transition{k}"):
            tr = getattr(f, f"transition{k}")
            ta, tb = _affine(tr.norm)
            tq, tsw = _wq(tr.conv.weight)
            blk.ta, blk.tb = ta.astype(np.float32), tb.astype(np.float32)
            blk.ts = max(float(amax[f"features.transition{k}.conv"]), 1e-12) / QMAX
            blk.tw, blk.tws = tq, tsw
            c //= 2
            h //= 2
            w //= 2
        plan.blocks.append(blk)
        k += 1
    a5, b5 = _affine(f.norm5)
    plan.a5, plan.b5 = a5.astype(np.float32), b5.astype(np.float32)
    return plan


def fan_params(plan: Plan) -> dict[str, np.ndarray]:
    """The consumers' affines with the scale folded in: `L{b}_{l}` [2, c_in] = (a1, b1) / s1,
    `T{b}` [2, c_tot] = (a, b) / s of the transition, `F` [2, n_features] = norm5 (unscaled). They
    are the tables of the `fan_*` kernels (and of `emulate`).
    """
    out = {}
    for bi, blk in enumerate(plan.blocks):
        for li, L in enumerate(blk.layers):
            inv = 1.0 / L.s1
            out[f"L{bi}_{li}"] = np.stack([L.a1.astype(np.float64) * inv,
                                           L.b1.astype(np.float64) * inv]).astype(np.float32)
        if blk.ta is not None:
            inv = 1.0 / blk.ts
            out[f"T{bi}"] = np.stack([blk.ta.astype(np.float64) * inv,
                                      blk.tb.astype(np.float64) * inv]).astype(np.float32)
    out["F"] = np.stack([plan.a5, plan.b5]).astype(np.float32)
    return out


# --- PyTorch reference --------------------------------------------------------------
@torch.no_grad()
def emulate(plan: Plan, x: torch.Tensor, quant: bool = True, half_io: bool | None = None,
            trace: dict | None = None) -> torch.Tensor:
    """The `Dense8Encoder` pipeline in PyTorch float32, on the device of `x` (N, in, H, W): features
    (N, n_features). `quant=False`: without INT8, the same function as the encoder (to check the
    rewrite: folding, pooling before the conv, segments); `half_io` (default = `quant`): engine
    outputs rounded to fp16 as in TensorRT. `trace`: receives the segments {(block, j): tensor}, like
    `Dense8Encoder.trace`.
    """
    half_io = quant if half_io is None else half_io
    if trace is None:
        trace = {}
    dev = x.device
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    rnd = (lambda t: t.half().float()) if half_io else (lambda t: t)  # noqa: E731
    fp = fan_params(plan) if quant else None
    x = x.float()
    y = F.relu(F.conv2d(x, T(plan.w0), T(plan.b0), stride=2, padding=3))
    seg = F.max_pool2d(rnd(y), 3, 2, 1)                  # max does not round: = rnd(max)
    for bi, blk in enumerate(plan.blocks):
        segs = [seg]
        trace[(bi, 0)] = seg
        for li, L in enumerate(blk.layers):
            xin = torch.cat(segs, 1)
            if quant:
                p = T(fp[f"L{bi}_{li}"])
                z = torch.round(p[0].view(1, -1, 1, 1) * xin + p[1].view(1, -1, 1, 1)).clamp(0, QMAX)
                hq = F.conv2d(z, T(L.w1.astype(np.float32)))
                hq = hq * (L.s1 * T(L.w1s)).view(1, -1, 1, 1) + T(L.bias1).view(1, -1, 1, 1)
                z2 = torch.round(hq * (1.0 / L.s2)).clamp(0, QMAX)
                yy = F.conv2d(z2, T(L.w2.astype(np.float32)), padding=1)
                yy = yy * (L.s2 * T(L.w2s)).view(1, -1, 1, 1)
            else:
                z = F.relu(T(L.a1).view(1, -1, 1, 1) * xin + T(L.b1).view(1, -1, 1, 1))
                w1 = T(L.w1.astype(np.float32)) * T(L.w1s).view(-1, 1, 1, 1)
                hq = F.relu(F.conv2d(z, w1, T(L.bias1)))
                w2 = T(L.w2.astype(np.float32)) * T(L.w2s).view(-1, 1, 1, 1)
                yy = F.conv2d(hq, w2, padding=1)
            segs.append(rnd(yy))
            trace[(bi, li + 1)] = segs[-1]
        xin = torch.cat(segs, 1)
        if blk.ta is not None:
            if quant:
                p = T(fp[f"T{bi}"])
                z = F.relu(p[0].view(1, -1, 1, 1) * xin + p[1].view(1, -1, 1, 1))
                sc = (blk.ts * T(blk.tws)).view(1, -1, 1, 1)
                if plan.trans_mode == "pool":
                    z = torch.round(F.avg_pool2d(z, 2)).clamp(0, QMAX)
                    t = F.conv2d(z, T(blk.tw.astype(np.float32))) * sc
                elif plan.trans_mode == "fan":
                    # the conv leaves the engine in fp16, the kernel computes the mean in float32
                    z = torch.round(z).clamp(0, QMAX)
                    t = F.avg_pool2d(rnd(F.conv2d(z, T(blk.tw.astype(np.float32))) * sc), 2)
                    trace[(bi + 1, 0)] = seg = t
                    continue
                else:
                    z = torch.round(z).clamp(0, QMAX)
                    t = F.avg_pool2d(F.conv2d(z, T(blk.tw.astype(np.float32))) * sc, 2)
            else:
                z = F.relu(T(blk.ta).view(1, -1, 1, 1) * xin + T(blk.tb).view(1, -1, 1, 1))
                tw = T(blk.tw.astype(np.float32)) * T(blk.tws).view(-1, 1, 1, 1)
                t = F.conv2d(F.avg_pool2d(z, 2), tw)
            seg = rnd(t)
        else:
            z = F.relu(T(plan.a5).view(1, -1, 1, 1) * xin + T(plan.b5).view(1, -1, 1, 1))
            return z.mean(dim=(2, 3))
    raise RuntimeError("plan without a final block")


# --- TensorRT engines ------------------------------------------------------------------------
def _trt():
    from .trt import trt_module
    return trt_module()


class _Net:
    """Building a TensorRT network with the network API; keeps the weight arrays alive (`trt.Weights`
    does not copy) until the build is done.
    """

    def __init__(self, builder) -> None:
        self.trt = _trt()
        self.net = builder.create_network(0)
        self.keep: list[np.ndarray] = []

    def const(self, arr: np.ndarray, shape=None):
        trt = self.trt
        arr = np.ascontiguousarray(arr)
        self.keep.append(arr)
        return self.net.add_constant(tuple(arr.shape) if shape is None else shape,
                                     trt.Weights(arr)).get_output(0)

    def scalar(self, v: float):
        a = np.array([v], dtype=np.float32)
        try:
            return self.const(a, ())
        except Exception:                                          # noqa: BLE001
            return self.const(a, (1,))

    def qdq_weights(self, q: np.ndarray, scale: np.ndarray):
        """Per-channel INT8 weights in the graph as a float constant -> Q -> DQ (axis 0): the form TensorRT
        recognises and turns into INT8 weights with those scales.
        """
        shape = (-1,) + (1,) * (q.ndim - 1)
        wf = (q.astype(np.float32) * scale.reshape(shape)).astype(np.float32)
        wt, sc = self.const(wf), self.const(scale.astype(np.float32))
        ql = self.net.add_quantize(wt, sc)
        ql.axis = 0
        dq = self.net.add_dequantize(ql.get_output(0), sc)
        dq.axis = 0
        return dq.get_output(0)

    def qdq(self, t, s: float):
        sc = self.scalar(s)
        q = self.net.add_quantize(t, sc)
        return self.net.add_dequantize(q.get_output(0), sc).get_output(0)

    def conv(self, inp, w, cout: int, k: int, stride: int = 1, pad: int = 0,
             bias: np.ndarray | None = None):
        """`w`: a tensor (from the Q/DQ nodes) or a float32 array [cout, cin, k, k] as constant weights."""
        trt = self.trt
        b = trt.Weights()
        if bias is not None:
            bias = np.ascontiguousarray(bias, dtype=np.float32)
            self.keep.append(bias)
            b = trt.Weights(bias)
        if isinstance(w, np.ndarray):
            w = np.ascontiguousarray(w, dtype=np.float32)
            self.keep.append(w)
            layer = self.net.add_convolution_nd(inp, cout, (k, k), trt.Weights(w), b)
        else:
            layer = self.net.add_convolution_nd(inp, cout, (k, k), trt.Weights(), b)
            layer.set_input(1, w)
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (pad, pad)
        return layer

    def int8_input(self, shape):
        trt = self.trt
        x = self.net.add_input("x", trt.int8, tuple(shape))
        x.allowed_formats = 1 << int(trt.TensorFormat.CHW32)
        return x

    def output(self, t) -> None:
        """The output "y" in fp16, in the format TensorRT prefers among those the fan kernels can read
        (`Fmt`): forcing NCHW cost 4.5 ms of Reformat per pass.
        """
        trt = self.trt
        t.name = "y"
        self.net.mark_output(t)
        t.dtype = trt.float16
        mask = 0
        for name in OUT_FORMATS:
            if hasattr(trt.TensorFormat, name):
                mask |= 1 << int(getattr(trt.TensorFormat, name))
        t.allowed_formats = mask


def _layer_net(builder, L: Layer, n: int, h: int, w: int) -> _Net:
    nb = _Net(builder)
    trt = nb.trt
    x = nb.int8_input((n, L.c_in, h, w))
    xd = nb.net.add_dequantize(x, nb.scalar(L.s1)).get_output(0)
    c1 = nb.conv(xd, nb.qdq_weights(L.w1, L.w1s), BOTTLENECK, 1, bias=L.bias1)
    r = nb.net.add_activation(c1.get_output(0), trt.ActivationType.RELU).get_output(0)
    c2 = nb.conv(nb.qdq(r, L.s2), nb.qdq_weights(L.w2, L.w2s), GROWTH, 3, pad=1)
    nb.output(c2.get_output(0))
    return nb


def _trans_net(builder, blk: Block, n: int, mode: str) -> _Net:
    nb = _Net(builder)
    trt = nb.trt
    div = 2 if mode == "pool" else 1
    x = nb.int8_input((n, blk.c_tot, blk.h // div, blk.w // div))
    xd = nb.net.add_dequantize(x, nb.scalar(blk.ts)).get_output(0)
    c = nb.conv(xd, nb.qdq_weights(blk.tw, blk.tws), blk.c_tot // 2, 1)
    y = c.get_output(0)
    if mode == "full":                   # "fan": the mean is done by the fan kernels
        p = nb.net.add_pooling_nd(y, trt.PoolingType.AVERAGE, (2, 2))
        p.stride_nd = (2, 2)
        y = p.get_output(0)
    nb.output(y)
    return nb


def _stem_net(builder, plan: Plan, n: int) -> _Net:
    nb = _Net(builder)
    trt = nb.trt
    x = nb.net.add_input("x", trt.float16, (n, plan.in_ch, plan.img, plan.img))
    x.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    c = nb.conv(x, plan.w0, plan.w0.shape[0], 7, stride=2, pad=3, bias=plan.b0)
    r = nb.net.add_activation(c.get_output(0), trt.ActivationType.RELU)
    layers = [c, r]
    last = r
    if not plan.stem_pool:
        last = nb.net.add_pooling_nd(r.get_output(0), trt.PoolingType.MAX, (3, 3))
        last.stride_nd = (2, 2)
        last.padding_nd = (1, 1)
        layers.append(last)
    for layer in layers:
        layer.precision = trt.float16
        layer.set_output_type(0, trt.float16)
    nb.output(last.get_output(0))
    return nb


def engine_names(plan: Plan) -> list[str]:
    names = ["stem"]
    for bi, blk in enumerate(plan.blocks):
        names += [f"b{bi}l{li}" for li in range(len(blk.layers))]
        if blk.ta is not None:
            names.append(f"t{bi}")
    return names


def build_engines(plan: Plan, n: int, timing_cache: bytes | None = None, log=print,
                  only: list[str] | None = None, opt_level: int | None = None
                  ) -> tuple[dict[str, bytes], bytes, dict]:
    """The engines of `plan` with fixed batch `n` (images): {name: engine}, timing cache, info. `only`:
    only these names (probes).
    """
    from .trt import logger
    trt = _trt()
    builder = trt.Builder(logger())
    cfg0 = builder.create_builder_config()
    cache = cfg0.create_timing_cache(timing_cache or b"")
    out: dict[str, bytes] = {}
    info: dict = {"build_s": {}, "mb": {}}
    t_all = time.perf_counter()
    for name in engine_names(plan):
        if only is not None and name not in only:
            continue
        if name == "stem":
            nb = _stem_net(builder, plan, n)
        elif name.startswith("t"):
            nb = _trans_net(builder, plan.blocks[int(name[1:])], n, plan.trans_mode)
        else:
            bi, li = (int(v) for v in name[1:].split("l"))
            blk = plan.blocks[bi]
            nb = _layer_net(builder, blk.layers[li], n, blk.h, blk.w)
        cfg = builder.create_builder_config()
        cfg.set_timing_cache(cache, True)
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        cfg.set_flag(trt.BuilderFlag.FP16)
        if name == "stem":
            # No Q/DQ in the stem: with the INT8 flag TensorRT would switch to implicit INT8 and want a
            # calibrator ("Calibration failure ... no scaling factors").
            cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        else:
            cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if hasattr(cfg, "max_aux_streams"):
            cfg.max_aux_streams = 0
        if opt_level is not None:
            cfg.builder_optimization_level = opt_level
        t = time.perf_counter()
        blob = builder.build_serialized_network(nb.net, cfg)
        if blob is None:
            raise RuntimeError(f"dense8: engine {name} not built (build_serialized_network None)")
        out[name] = bytes(blob)
        info["build_s"][name] = round(time.perf_counter() - t, 2)
        info["mb"][name] = round(len(out[name]) / 2**20, 2)
        del nb
    info["total_s"] = round(time.perf_counter() - t_all, 1)
    log(f"dense8: {len(out)} engines in {info['total_s']} s, "
        f"{sum(info['mb'].values()):.1f} MiB")
    return out, bytes(cache.serialize()), info


# --- package ------------------------------------------------------------------------------
def kernels_sha() -> str:
    import hashlib
    return hashlib.sha1(KERNELS.read_bytes()).hexdigest()[:12]


def pack_meta(plan: Plan, n: int) -> dict:
    return {"format": PACK_FORMAT, "n": int(n), "img": plan.img, "in_ch": plan.in_ch,
            "kernels_sha": kernels_sha(),
            "n_features": plan.n_features, "stem_c": int(plan.w0.shape[0]),
            "trans_mode": plan.trans_mode, "stem_pool": plan.stem_pool,
            "blocks": [{"c0": b.c0, "h": b.h, "w": b.w, "L": len(b.layers),
                        "trans": b.ta is not None} for b in plan.blocks]}


def save_pack(path: Path, plan: Plan, n: int, engines: dict[str, bytes], extra: dict | None = None,
              cubin: bytes | None = None) -> None:
    """`cubin`: the `fan_*` kernels already compiled for the inference GPU (loaded without NVRTC)."""
    meta = pack_meta(plan, n) | {"engines": sorted(engines), **(extra or {})}
    arrays = {"meta": np.frombuffer(json.dumps(meta).encode(), np.uint8)}
    if cubin:
        arrays["cubin"] = np.frombuffer(cubin, np.uint8)
    arrays |= {f"eng_{k}": np.frombuffer(v, np.uint8) for k, v in engines.items()}
    arrays |= {f"par_{k}": v for k, v in fan_params(plan).items()}
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)


def load_pack(path: Path) -> dict:
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        engines = {k[4:]: z[k].tobytes() for k in z.files if k.startswith("eng_")}
        par = {k[4:]: np.array(z[k]) for k in z.files if k.startswith("par_")}
        cubin = z["cubin"].tobytes() if "cubin" in z.files else None
    return {"meta": meta, "engines": engines, "par": par, "cubin": cubin}


# --- runner ----------------------------------------------------------------------------------
_CUBIN: dict = {}


def kernel_cubin(arch: str) -> bytes:
    """The `fan_*` kernels compiled with NVRTC for `arch` (e.g. "sm_75"), once per process."""
    from . import cudart
    if arch not in _CUBIN:
        t = time.perf_counter()
        _CUBIN[arch] = cudart.compile_source(KERNELS.read_text(encoding="utf-8"), arch)
        _CUBIN["compile_s"] = round(time.perf_counter() - t, 3)
    return _CUBIN[arch]


def kernel_module(dev: int, cubin: bytes | None = None, sha: str | None = None):
    """The module of the `fan_*` kernels on GPU `dev`, once per GPU: from the package `cubin` if it loads
    and comes from the same source (`sha`; the rerun has no internet and possibly no NVRTC), otherwise
    compiled here for the GPU's architecture.
    """
    if cubin and sha is not None and sha != kernels_sha():
        _CUBIN["pack_error"] = f"package cubin built from another source ({sha})"
        cubin = None
    from . import cudart
    key = ("mod", dev)
    if key in _CUBIN:
        return _CUBIN[key]
    major, minor = torch.cuda.get_device_capability(dev)
    with torch.cuda.device(dev):
        torch.cuda.current_stream(dev)              # PyTorch's current primary context
        mod = None
        if cubin:
            try:
                mod = cudart.Module(cubin, dev)
                _CUBIN["from"] = "pack"
            except cudart.CudaError as exc:
                _CUBIN["pack_error"] = str(exc)[:200]
        if mod is None:
            mod = cudart.Module(kernel_cubin(f"sm_{major}{minor}"), dev)
            _CUBIN["from"] = "nvrtc"
    _CUBIN[key] = mod
    return mod


class _Seg:
    """A segment and its consumers, with the tables already on the GPU."""
    __slots__ = ("b", "j", "c", "g0", "h", "w", "ncons", "dst", "groups", "par", "tail", "tpar",
                 "tbuf", "tgroups", "off", "fmt", "src", "kch", "gz")


class Dense8Encoder:
    """The dense8 engine on one GPU: `enc(x)` with x (N, in, img, img) fp16/float on GPU `dev`, N a
    multiple of the package's `n`; returns (N, n_features) float32 in its own buffer. Same interface as
    `knee.trt.TrtEncoder` (own stream, `max_inflight`, `capture(n)` for graphs).
    """

    def __init__(self, pack: dict, dev: int, n_total: int, max_inflight: int | None = None) -> None:
        from .cudart import Ptr
        from .trt import logger
        trt = _trt()
        self.Ptr = Ptr
        meta = pack["meta"]
        if meta.get("format") != PACK_FORMAT:
            raise ValueError(f"dense8 package format {meta.get('format')}, expected {PACK_FORMAT}")
        self.meta, self.dev = meta, dev
        self.nb = int(meta["n"])
        if n_total % self.nb:
            raise ValueError(f"{n_total} images do not split into blocks of {self.nb}")
        self.max_inflight = max_inflight
        self.inflight: collections.deque = collections.deque()
        self.graphs: dict[int, tuple] = {}
        self.calls = {"graph": 0, "plain": 0}
        n, d = self.nb, f"cuda:{dev}"
        t0 = time.perf_counter()
        with torch.cuda.device(dev):
            self.stream = torch.cuda.Stream(dev)
            self.rt = trt.Runtime(logger())
            self.engines, self.ctxs = {}, {}
            for name in meta["engines"]:
                eng = self.rt.deserialize_cuda_engine(pack["engines"][name])
                if eng is None:
                    raise RuntimeError(f"dense8: engine {name} cannot be deserialized")
                self.engines[name] = eng
            self.deser_s = time.perf_counter() - t0
            self._contexts(trt)
            self._check_io(trt)
            self.mod = kernel_module(dev, pack.get("cubin"), meta.get("kernels_sha"))
            self.k_dense = self.mod.function("fan_dense")
            self.k_trans = self.mod.function("fan_trans")
            self.k_final = self.mod.function("fan_final")
            self._buffers(pack, n, d)
            self.out = torch.empty((n_total, meta["n_features"]), dtype=torch.float32, device=d)
            self.in_dtype = torch.float16
            self.in_shape = (n, meta["in_ch"], meta["img"], meta["img"])
        self.load_s = time.perf_counter() - t0

    # -- initialisation --
    def _contexts(self, trt) -> None:
        strat = getattr(trt, "ExecutionContextAllocationStrategy", None)
        sizes = {}
        for k, e in self.engines.items():
            sizes[k] = int(getattr(e, "device_memory_size_v2", 0) or e.device_memory_size)
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
                raise RuntimeError(f"dense8: context of engine {k} not created")
            self.ctxs[k] = ctx
        self.scratch_mb = round(max(sizes.values()) / 2**20, 1)

    def _check_io(self, trt) -> None:
        """Inputs in the format we write; outputs in the one TensorRT chose (`Fmt` for every engine, in
        `self.fmts`).
        """
        meta = self.meta
        blocks = meta["blocks"]
        self.fmts: dict[str, Fmt] = {}
        for k, e in self.engines.items():
            fx = e.get_tensor_format("x")
            want = trt.TensorFormat.LINEAR if k == "stem" else trt.TensorFormat.CHW32
            if fx != want:
                raise RuntimeError(f"dense8: engine {k} with input {fx}")
            if k == "stem":
                c, h, w = meta["stem_c"], blocks[0]["h"], blocks[0]["w"]
                if meta.get("stem_pool"):
                    h, w = meta["img"] // 2, meta["img"] // 2
            elif k.startswith("t"):
                nb = blocks[int(k[1:]) + 1]
                c, h, w = nb["c0"], nb["h"], nb["w"]
                if meta.get("trans_mode") == "fan":
                    h, w = 2 * h, 2 * w
            else:
                b = blocks[int(k[1:].split("l")[0])]
                c, h, w = GROWTH, b["h"], b["w"]
            self.fmts[k] = tensor_fmt(trt, e, "y", c, h, w)

    def _buffers(self, pack: dict, n: int, d: str) -> None:
        meta, par = self.meta, pack["par"]
        blocks = meta["blocks"]
        self.full = meta.get("trans_mode", "pool") in ("full", "fan")
        dev = torch.device(d)
        # Byte buffers for the engine outputs, as large as the format chosen by TensorRT.
        nbytes = lambda keys: max([self.fmts[k].elems(n) * self.fmts[k].itemsize()  # noqa: E731
                                   for k in keys] or [2])
        self.s0 = torch.empty(nbytes(["stem"]), dtype=torch.uint8, device=d)
        self.ybuf = torch.empty(nbytes([k for k in self.fmts if k.startswith("b")]),
                                dtype=torch.uint8, device=d)
        self.segbuf = torch.empty(nbytes([k for k in self.fmts if k.startswith("t")]),
                                  dtype=torch.uint8, device=d)
        # INT8 arena per block: the inputs of all layers and of the transition.
        lay, arena = [], 0
        for b in blocks:
            hw, off, xs = b["h"] * b["w"], 0, []
            for li in range(b["L"]):
                g = (b["c0"] + GROWTH * li) // GROUP
                xs.append((off, g))
                off += n * g * hw * GROUP
            tr = None
            if b["trans"]:
                gt = (b["c0"] + GROWTH * b["L"]) // GROUP
                tr = (off, gt)
                thw = hw if self.full else (b["h"] // 2) * (b["w"] // 2)
                off += n * gt * thw * GROUP
            lay.append((xs, tr))
            arena = max(arena, off)
        self.arena = torch.empty(arena, dtype=torch.int8, device=d)
        self.arena_mb = round(arena / 2**20, 1)
        base = self.arena.data_ptr()
        self.x_ptr = [[base + o for o, _ in xs] for xs, _ in lay]
        self.t_ptr = [None if tr is None else base + tr[0] for _, tr in lay]
        # Segments: (block, j) with j = 0 the block input, j >= 1 the output of layer j-1.
        self.segs: dict[tuple[int, int], _Seg] = {}
        for bi, b in enumerate(blocks):
            xs, tr = lay[bi]
            for j in range(b["L"] + 1):
                s = _Seg()
                s.b, s.j, s.h, s.w = bi, j, b["h"], b["w"]
                s.c = b["c0"] if j == 0 else GROWTH
                s.off = 0 if j == 0 else b["c0"] + GROWTH * (j - 1)
                s.fmt = (self.fmts["stem"] if (bi, j) == (0, 0)
                         else self.fmts[f"t{bi - 1}"] if j == 0
                         else self.fmts[f"b{bi}l{j - 1}"])
                if (bi, j) == (0, 0) and meta.get("stem_pool"):
                    s.src = Src(2, meta["img"] // 2, meta["img"] // 2)
                elif j == 0 and bi > 0 and meta.get("trans_mode") == "fan":
                    s.src = Src(1, 2 * s.h, 2 * s.w)
                else:
                    s.src = Src(0, s.h, s.w)
                s.g0 = s.off // GROUP
                cons = list(range(j, b["L"]))              # layers reading the segment
                ptrs = [self.x_ptr[bi][li] for li in cons]
                grps = [xs[li][1] for li in cons]
                pars = [par[f"L{bi}_{li}"][:, s.off:s.off + s.c] for li in cons]
                if b["trans"] and self.full:
                    # transition in the encoder's order: a consumer like the layers
                    ptrs.append(self.t_ptr[bi])
                    grps.append(tr[1])
                    pars.append(par[f"T{bi}"][:, s.off:s.off + s.c])
                s.ncons = len(ptrs)
                # Few thread blocks (block 4: 88 x 81 pixels) -> consumers split over blockIdx.z, so the T4 (40 SMs)
                # is filled.
                ctas = -(-2 * n * s.h * s.w // FAN_THREADS) * (s.c // GROUP)
                nz = min(max(s.ncons, 1), max(1, -(-320 // ctas)))
                s.kch = max(1, -(-s.ncons // nz))
                s.gz = max(1, -(-s.ncons // s.kch))
                if ptrs:
                    s.dst = torch.tensor(ptrs, dtype=torch.int64, device=dev)
                    s.groups = torch.tensor(grps, dtype=torch.int32, device=dev)
                    s.par = torch.from_numpy(np.ascontiguousarray(np.stack(pars),
                                                                  np.float32)).to(dev)
                s.tail = None
                if b["trans"] and self.full:
                    pass
                elif b["trans"]:
                    s.tail = "trans"
                    s.tpar = torch.from_numpy(np.ascontiguousarray(
                        par[f"T{bi}"][:, s.off:s.off + s.c], np.float32)).to(dev)
                    s.tbuf, s.tgroups = self.t_ptr[bi], tr[1]
                else:
                    s.tail = "final"
                    s.tpar = torch.from_numpy(np.ascontiguousarray(
                        par["F"][:, s.off:s.off + s.c], np.float32)).to(dev)
                self.segs[(bi, j)] = s
        # Fixed engine addresses: the layers read their buffer and write into ybuf.
        for bi, b in enumerate(blocks):
            for li in range(b["L"]):
                ctx = self.ctxs[f"b{bi}l{li}"]
                ctx.set_tensor_address("x", self.x_ptr[bi][li])
                ctx.set_tensor_address("y", self.ybuf.data_ptr())
            if b["trans"]:
                ctx = self.ctxs[f"t{bi}"]
                ctx.set_tensor_address("x", self.t_ptr[bi])
                ctx.set_tensor_address("y", self.segbuf.data_ptr())
        self.ctxs["stem"].set_tensor_address("y", self.s0.data_ptr())

    # -- one pass --
    def _fan(self, s: _Seg, y_ptr: int, f_ptr: int, st: int) -> None:
        P, n = self.Ptr, self.nb
        hw = s.h * s.w
        groups = s.c // GROUP
        if s.ncons:
            self.k_dense.launch(((2 * n * hw + FAN_THREADS - 1) // FAN_THREADS, groups, s.gz),
                                (FAN_THREADS,),
                                [P(y_ptr), s.fmt, s.src, n * hw, s.h, s.w, s.c, s.g0, s.ncons,
                                 s.kch, s.dst, s.groups, s.par], st, shared=s.kch * 64 * 4)
        if s.tail is None:
            return
        if s.tail == "trans":
            hw2 = (s.h // 2) * (s.w // 2)
            self.k_trans.launch(((2 * n * hw2 + FAN_THREADS - 1) // FAN_THREADS, groups),
                                (FAN_THREADS,),
                                [P(y_ptr), s.fmt, s.src, n, s.h, s.w, s.c, s.g0, s.tgroups,
                                 P(s.tbuf), s.tpar], st)
        else:
            self.k_final.launch((n, groups), (128,),
                                [P(y_ptr), s.fmt, s.src, s.h, s.w, s.c, s.off,
                                 self.meta["n_features"], P(f_ptr), s.tpar], st)

    def _pass(self, x_ptr: int, f_ptr: int, events: list | None = None) -> None:
        st = self.stream.cuda_stream
        mark = (lambda tag: events.append((tag, _record(self.stream)))) if events is not None \
            else (lambda tag: None)
        blocks = self.meta["blocks"]
        mark("start")
        ctx = self.ctxs["stem"]
        ctx.set_tensor_address("x", x_ptr)
        ctx.execute_async_v3(st)
        mark("stem")
        self._fan(self.segs[(0, 0)], self.s0.data_ptr(), f_ptr, st)
        mark("fan0")
        yp = self.ybuf.data_ptr()
        for bi, b in enumerate(blocks):
            for li in range(b["L"]):
                self.ctxs[f"b{bi}l{li}"].execute_async_v3(st)
                mark(f"conv{bi}")
                self._fan(self.segs[(bi, li + 1)], yp, f_ptr, st)
                mark(f"fan{bi}")
            if b["trans"]:
                self.ctxs[f"t{bi}"].execute_async_v3(st)
                mark(f"trans{bi}")
                self._fan(self.segs[(bi + 1, 0)], self.segbuf.data_ptr(), f_ptr, st)
                mark(f"fan{bi + 1}")

    def _enqueue(self, xf: torch.Tensor, out_off: int = 0) -> None:
        for i, chunk in enumerate(xf.split(self.nb)):
            if self.max_inflight:
                while len(self.inflight) >= self.max_inflight:
                    self.inflight.popleft().synchronize()
            self._pass(chunk.data_ptr(), self.out[out_off + i * self.nb].data_ptr())
            if self.max_inflight:
                ev = torch.cuda.Event(blocking=True)
                ev.record(self.stream)
                self.inflight.append(ev)

    def capture(self, n: int) -> float:
        """One CUDA graph for every block of `nb` images of a batch of `n` (~400 kernels each), from a static
        input. Not one for the whole batch: its nodes would all go through CUDA's launch queue (~1,000) at
        once and the thread would spin; this way `max_inflight` applies between blocks.
        """
        if n % self.nb or n > self.out.shape[0]:
            raise ValueError(f"graph for {n} images: the engine does {self.nb} per block, "
                             f"the output holds {self.out.shape[0]}")
        t = time.perf_counter()
        with torch.cuda.device(self.dev):
            static_in = torch.zeros((n, *self.in_shape[1:]), dtype=self.in_dtype,
                                    device=f"cuda:{self.dev}")
            with torch.cuda.stream(self.stream):
                self._enqueue(static_in[:self.nb])      # one warm-up pass per context is enough for TensorRT
            self.stream.synchronize()
            self.inflight.clear()
            graphs = []
            for i, chunk in enumerate(static_in.split(self.nb)):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=self.stream, capture_error_mode="thread_local"):
                    self._pass(chunk.data_ptr(), self.out[i * self.nb].data_ptr())
                graphs.append(g)
            self.stream.synchronize()
        self.graphs[n] = (graphs, static_in)
        return time.perf_counter() - t

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cur = torch.cuda.current_stream(self.dev)
        self.stream.wait_stream(cur)
        n = x.shape[0]
        with torch.cuda.stream(self.stream):
            gs = self.graphs.get(n)
            if gs is None and n % self.nb == 0:
                # fewer blocks than the captured batch (windows of empty slots skipped, `infer._model_logits`): the
                # first n / nb graphs, which read static_in[:n]
                for m_, (g_, s_) in sorted(self.graphs.items()):
                    if m_ >= n:
                        gs = (g_[:n // self.nb], s_)
                        break
            if gs is not None:
                graphs, static_in = gs
                static_in[:n].copy_(x)
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

    @torch.no_grad()
    def trace(self, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """One pass (the first `n` images) stopping after every engine: the segments {(block, j): float
        tensor} like `emulate(trace=...)`, and the features. For debugging.
        """
        xf = x[:self.nb].to(self.in_dtype).contiguous()
        out: dict = {}
        st = self.stream.cuda_stream
        blocks = self.meta["blocks"]
        n = self.nb
        with torch.cuda.device(self.dev), torch.cuda.stream(self.stream):
            f = torch.zeros((n, self.meta["n_features"]), dtype=torch.float32,
                            device=f"cuda:{self.dev}")
            fp = f.data_ptr()
            ctx = self.ctxs["stem"]
            ctx.set_tensor_address("x", xf.data_ptr())
            ctx.execute_async_v3(st)
            b0 = blocks[0]
            if self.meta.get("stem_pool"):
                side = self.meta["img"] // 2
                out[(0, 0)] = F.max_pool2d(unpack(self.s0, self.fmts["stem"], n,
                                                  self.meta["stem_c"], side, side), 3, 2, 1)
            else:
                out[(0, 0)] = unpack(self.s0, self.fmts["stem"], n, self.meta["stem_c"],
                                     b0["h"], b0["w"])
            self._fan(self.segs[(0, 0)], self.s0.data_ptr(), fp, st)
            yp = self.ybuf.data_ptr()
            for bi, b in enumerate(blocks):
                h, w = b["h"], b["w"]
                for li in range(b["L"]):
                    self.ctxs[f"b{bi}l{li}"].execute_async_v3(st)
                    out[(bi, li + 1)] = unpack(self.ybuf, self.fmts[f"b{bi}l{li}"], n, GROWTH,
                                               h, w)
                    self._fan(self.segs[(bi, li + 1)], yp, fp, st)
                if b["trans"]:
                    self.ctxs[f"t{bi}"].execute_async_v3(st)
                    nb = blocks[bi + 1]
                    if self.meta.get("trans_mode") == "fan":
                        out[(bi + 1, 0)] = F.avg_pool2d(unpack(
                            self.segbuf, self.fmts[f"t{bi}"], n, nb["c0"], 2 * nb["h"],
                            2 * nb["w"]), 2)
                    else:
                        out[(bi + 1, 0)] = unpack(self.segbuf, self.fmts[f"t{bi}"], n,
                                                  nb["c0"], nb["h"], nb["w"])
                    self._fan(self.segs[(bi + 1, 0)], self.segbuf.data_ptr(), fp, st)
            self.stream.synchronize()
        return out, f

    def profile(self, x: torch.Tensor, reps: int = 3) -> dict:
        """GPU milliseconds per item (stem, conv per block, transitions, fan) of a pass of `n` images, from
        CUDA events between launches.
        """
        xf = x[:self.nb].to(self.in_dtype).contiguous()
        tot: dict[str, float] = collections.defaultdict(float)
        with torch.cuda.device(self.dev), torch.cuda.stream(self.stream):
            for _ in range(reps + 1):
                ev: list = []
                self._pass(xf.data_ptr(), self.out.data_ptr(), ev)
                self.stream.synchronize()
                if _ == 0:
                    continue                          # warm-up pass
                for (_, a), (tag, b) in zip(ev, ev[1:]):
                    tot[tag] += a.elapsed_time(b)
        return {k: round(v / reps, 3) for k, v in sorted(tot.items())}


def _record(stream) -> torch.cuda.Event:
    e = torch.cuda.Event(enable_timing=True)
    e.record(stream)
    return e
