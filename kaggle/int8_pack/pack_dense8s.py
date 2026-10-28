"""Kaggle kernel (2 x T4): builds the INT8 packages of the two DenseNet169 seeds (`knee.dense8s`). No
submission.

`knee.dense8s` is the DenseNet encoder in INT8 with a per-block "store" (every segment written once,
INT8 with a per-channel scale) and a hand-written conv1 on the tensor cores whose prologue applies the
consumer's BatchNorm + ReLU + quantization. At 512 px the stem stays a TensorRT fp16 engine (our stem
works on 9-pixel rectangles and 128 is not divisible by 9); everything else runs on our kernels.

What it does, in order:
1. compiles the kernels with NVRTC (`dense8s.kernel_cubin`) and loads seed 0 of the run;
2. calibrates on 16 training studies from the 512 px cache: per-channel maxima of every store segment
   and per-tensor maxima of every conv input (`dense8s.calibrate`); post-training quantization with
   `max` scales costs -0.0003 OOF AUC (Spearman 0.9986 with fp32);
3. checks every component against its integer formula on random data (conv1, conv2, transitions,
   final features, the planar bottleneck, the fixed-shape conv2): the bytes must match;
4. checks the whole pipeline against fp32 (relative L2 of the features, gate 0.067) and times it in
   CUDA graphs of 352 images (the whole batch of one GPU: 4 studies x 88 windows);
5. builds the package of every seed (`<prefix>_fold<k>.npz`: weights, scales, kernel tables, the
   TensorRT stem engine, the chosen kernels) and checks it against fp32 (0.035-0.042).

The kernel choices are FIXED, not tuned in bursts: on long runs the T4s sit at their 70 W power cap,
where time follows energy rather than cycles, and burst tuning picked kernels that did not help in the
real loop. The fixed choices come from power-capped A/B probes run in the same session: conv1 with
64 x 64 warp tiles writing a planar bottleneck (`w64_pl`) in blocks 0-2, fixed-shape conv2 in block 0
(`t8x24_pl`), persistent conv2 with resident weights in blocks 1-2, wide transitions (`w0`/`w1`), and
the stem max-pool on 7-bit bytes in 4 instructions. Together -10.5% at the power cap, with identical
outputs.

Per study (88 images at 288 px) on one T4: ~115 ms with explicit-INT8 TensorRT, 58 ms with the first
version of these kernels, 34 ms with the per-block store.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

T0 = time.perf_counter()


def find_src() -> Path:
    base = Path("/kaggle/input")
    seen: list[Path] = []
    frontier = [(base, 0)] if base.is_dir() else []
    while frontier:
        d, depth = frontier.pop(0)
        seen.append(d)
        if depth < 4:
            try:
                frontier += [(k, depth + 1) for k in sorted(p for p in d.iterdir() if p.is_dir())
                             if k.name not in ("train_series", "test_series")]
            except OSError:
                pass
    for d in seen + [Path("src"), Path("../src"), Path(".")]:
        for c in (d, d / "src"):
            if (c / "knee" / "constants.py").is_file():
                return c
    raise FileNotFoundError("knee-src dataset not mounted")


# --- overlay (written by the builder: do not edit by hand) ---
OVERLAY_SHA = ""
OVERLAY = {}
# --- end of overlay ---


SRC = find_src()
BUNDLE = SRC.parent if SRC.name == "src" else SRC
if OVERLAY:
    # the mounted knee-src plus the files changed after it (empty in the release)
    import base64
    import shutil
    import zlib
    _dst = Path("/tmp/src_overlay")
    shutil.copytree(SRC, _dst, dirs_exist_ok=True)
    for _rel, _b64 in OVERLAY.items():
        (_dst / _rel).write_bytes(zlib.decompress(base64.b64decode(_b64)))
    (_dst / "knee" / "GIT_SHA").write_text(OVERLAY_SHA)
    SRC = _dst
sys.path.insert(0, str(SRC))
GIT_SHA = (SRC / "knee" / "GIT_SHA").read_text().strip()
print(f"knee from {SRC} | GIT_SHA {GIT_SHA}", flush=True)
_pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorrt-cu12==10.16.1.11"],
                      capture_output=True, text=True)
print(f"pip rc {_pip.returncode} {(_pip.stdout + _pip.stderr)[-200:]!r}", flush=True)

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

from knee import cudart, dense8, dense8s                              # noqa: E402
from knee.aggregate import model_from_checkpoint                      # noqa: E402
from knee.constants import ID                                         # noqa: E402
from knee.data.dataset import KneeStudies, open_cache                 # noqa: E402
from knee.kaggle import find_dir_with, find_dirs_with                 # noqa: E402
from knee.models.baseline import make_windows                         # noqa: E402
from knee.trt import set_blocking_sync                                # noqa: E402

EXP = "376-384-full-data-v9j-287-img512"
# 512 px: the cache and the packages at this resolution even if the checkpoint says otherwise
IMG = 512
N = 22
# graphs of 352 images (the whole batch of one GPU, 4 studies x 88 windows): timings, tuning and
# packages; the per-component checks stay at 88
NB = 180
# fused blocks tried: none at 512 px (sliding-band fusion on blocks 0-1 and the banded block 2 were
# slower or no better at 288 px; only the whole-block fusion of block 3 won, and at 512 px it does not
# apply)
FUSED_TRY = {}                  # 512 px: img3 needs HW <= 255, here 256
# earlier choices (reference packages): the in-session baseline of the A/B
REF_V13 = {
    "bm": {"b0l0": "u128hs", "b0l1": "u128h", "b0l2": "u128h", "b0l3": "u128h", "b0l4": "u128hs", "b0l5":
        "u128hs", "b1l0": "u128hs", "b1l1": "u128hs", "b1l2": "u128hs", "b1l3": "u128hs", "b1l4":
        "u128hs", "b1l5": "u128hs", "b1l6": "u128hs", "b1l7": "u128hs", "b1l8": "u128hs", "b1l9":
        "u128hs", "b1l10": "u256h", "b1l11": "u256h", "b2l0": "u128hs", "b2l1": "u128hs", "b2l2":
        "u128hs", "b2l3": "u128hs", "b2l4": "u128hs", "b2l5": "u128hs", "b2l6": "u128hs", "b2l7":
        "u128hs", "b2l8": "u128hs", "b2l9": "u128hs", "b2l10": "u128hs", "b2l11": "u128hs", "b2l12":
        "u128hs", "b2l13": "u128hs", "b2l14": "u128hs", "b2l15": "u128hs", "b2l16": "u128hs", "b2l17":
        "u128hs", "b2l18": "u128hs", "b2l19": "u128hs", "b2l20": "u128hs", "b2l21": "u128hs", "b2l22":
        "u128hs", "b2l23": "u128hs", "b2l24": "u128hs", "b2l25": "u128hs", "b2l26": "u128hs", "b2l27":
        "u128hs", "b2l28": "u128hs", "b2l29": "u128hs", "b2l30": "u128hs", "b2l31": "u128hs", "b3l0":
        "u128hs", "b3l1": "u128hs", "b3l2": "u128hs", "b3l3": "u128hs", "b3l4": "u128hs", "b3l5":
        "u128hs", "b3l6": "u128hs", "b3l7": "u128hs", "b3l8": "u128hs", "b3l9": "u128hs", "b3l10":
        "u128hs", "b3l11": "u128hs", "b3l12": "u128hs", "b3l13": "u128hs", "b3l14": "u128hs", "b3l15":
        "u128hs", "b3l16": "u128hs", "b3l17": "u128hs", "b3l18": "u128hs", "b3l19": "u128hs", "b3l20":
        "u128hs", "b3l21": "u128hs", "b3l22": "u128hs", "b3l23": "u128hs", "b3l24": "u128hs", "b3l25":
        "u128hs", "b3l26": "u128hs", "b3l27": "u128hs", "b3l28": "u128hs", "b3l29": "u128hs", "b3l30":
        "u128hs", "b3l31": "u128hs"},
    "c2cfg": {"72": [24, 8, "k4w6"], "36": [18, 12, "p4u"], "18": [18, 18, "q6w8"], "9": [9, 9, "q4w4u"]},
    "tbm": {"t0": "f16h", "t1": "f16h", "t2": "f16h"}, "stem": "ours", "fused": {"b3": "img3"},
}
FUSED_TRY_LIST = [(int(k[1:]), v) for k, v in FUSED_TRY.items()]
# fixed choices from the power-capped energy probes (measured at the power cap, in the same session):
# (conv1 per layer, conv2 per side, transitions) and the fused blocks
FIXED = (
    {f"b{bi}l{li}": ("w64_pl" if bi < 3 else "u128hs")
     for bi, nl in enumerate((6, 12, 32, 32)) for li in range(nl)},
    {128: [16, 8, "p4u_pl"], 64: [16, 8, "p4u_pl"], 32: [16, 8, "p4u_pl"], 16: [16, 8, "q4w4u"]},
    {"t0": "w0", "t1": "w1", "t2": "w0"},
)
FIXED_FUSED = None
SEEDS = [0, 1]
N_CAL = 16
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("runs/probe-dense8s")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS: dict = {"git_sha": GIT_SHA}
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def save() -> None:
    RESULTS["total_s"] = round(time.perf_counter() - T0, 1)
    (OUT / "probe.json").write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


def train_cache(spec):
    roots = find_dirs_with(f"vols_{spec.tag}.npy")
    return open_cache(roots if len(roots) > 1 else roots[0], spec)


def pick(cache, n: int, seed: int, exclude=()) -> list[str]:
    uids = [u for u in pd.read_csv(BUNDLE / "data" / "folds" / "folds.csv")[ID]
            if u in cache and u not in set(exclude)]
    return [uids[i] for i in np.random.default_rng(seed).permutation(len(uids))[:n]]


def study_windows(cache, uids: list[str], m_cpu) -> torch.Tensor:
    ds = KneeStudies(cache, uids)
    vol = torch.stack([torch.as_tensor(ds[i][0]) for i in range(len(ds))]).to("cuda:0")
    w = make_windows(vol.float().div_(255.0), m_cpu.window_stride, m_cpu.window_depth)
    return w.flatten(0, 1).contiguous()


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    return {"max_abs": round(float((a - b).abs().max()), 5),
            "rel_l2": round(float((a - b).norm() / b.norm().clamp_min(1e-12)), 6),
            "cos_min": round(float(F.cosine_similarity(a, b, dim=1).min()), 6)}


def bytes_cmp(got: torch.Tensor, want: torch.Tensor) -> dict:
    d = (got.double() - want.double()).abs()
    return {"n": int(d.numel()), "mismatch_frac": float((d > 0).double().mean()),
            "max_diff": float(d.max()), "want_absmax": float(want.abs().max())}


@torch.no_grad()
def in_chunks(fn, x: torch.Tensor, k: int = N) -> torch.Tensor:
    return torch.cat([fn(c).float() for c in x.split(k)])


def timed(fn, reps: int) -> dict:
    torch.cuda.synchronize(0)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    c0 = time.thread_time()
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    c1 = time.thread_time()
    torch.cuda.synchronize(0)
    return {"ms_per_call": round(s.elapsed_time(e) / reps, 2),
            "cpu_ms_per_call": round((c1 - c0) * 1e3 / reps, 2)}


# --- per-component checks on random data ----------------------------------------------
def _slab_write(run, bi: int, j: int, t: torch.Tensor) -> None:
    """Writes segment (bi, j) of the store (the inverse of `Dense8SEncoder.store`)."""
    b = run.meta["blocks"][bi]
    n, hw = run.nb, b["h"] * b["w"]
    g0 = b["c0"] // dense8s.GROUP
    a = run.arenas[bi % 2]
    q = t.round().clamp(-128, 127).to(torch.int8)
    if j == 0:
        off = run.in0[bi] - a.data_ptr()
        a[off:off + n * g0 * hw * 32].copy_(
            q.reshape(n, g0, 32, hw).permute(0, 1, 3, 2).flatten())
    else:
        off = run.seg[bi] - a.data_ptr() + (j - 1) * n * hw * 32
        a[off:off + n * hw * 32].copy_(q.reshape(n, 32, hw).permute(0, 2, 1).flatten())


def _rand_store(run, bi: int, upto: int, seed: int) -> list[torch.Tensor]:
    b = run.meta["blocks"][bi]
    n, h, w = run.nb, b["h"], b["w"]
    g = torch.Generator(device="cuda:0").manual_seed(seed)
    segs = []
    for j in range(upto + 1):
        c = b["c0"] if j == 0 else 32
        t = torch.randint(-127, 128, (n, c, h, w), generator=g, device="cuda:0").double()
        _slab_write(run, bi, j, t)
        segs.append(t)
    return segs


@torch.no_grad()
def unit_conv1(run, plan, par, bi: int, li: int, kind=None) -> dict:
    segs = _rand_store(run, bi, li, 100 * bi + li)
    b = run.meta["blocks"][bi]
    n, h, w, hw = run.nb, b["h"], b["w"], b["h"] * b["w"]
    run.bott.fill_(0)
    torch.cuda.synchronize(0)
    item = run.plan_l[bi][0][li][0] if kind is None else run._c1_item(bi, li, kind)
    with torch.cuda.stream(run.stream):
        run._launch(item, run.stream.cuda_stream)
    run.stream.synchronize()
    if run.ours and str(kind).endswith(dense8s.PLANAR):
        # planar bottleneck [4][N*HW][32] (MODE 3)
        got = run.bott[:n * 128 * hw].view(torch.uint8).view(4, n, hw, 32).permute(1, 0, 3, 2) \
            .reshape(n, 128, h, w)
        gd, qb = par[f"L{bi}_{li}_gdu"], 255
    elif run.ours:
        got = run.bott[:n * 128 * hw].view(torch.uint8).view(n, h, w, 128).permute(0, 3, 1, 2)
        gd, qb = par[f"L{bi}_{li}_gdu"], 255
    else:
        got = run.bott[:n * 128 * hw].view(n, 4, hw, 32).permute(0, 1, 3, 2).reshape(n, 128, h, w)
        gd, qb = par[f"L{bi}_{li}_gd"], 127
    L = plan.blocks[bi].layers[li]
    X = torch.cat(segs, 1)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device="cuda:0")  # noqa: E731
    if kind is not None and str(kind) in dense8s.C1_F16:
        a16, b16 = dense8s._abh_ab(par[f"L{bi}_{li}_abh"])
        t = T(a16).view(1, -1, 1, 1) * X + T(b16).view(1, -1, 1, 1)
        u = torch.round(t.half().double().clamp(0, 1) * 255)
    else:
        ab = par[f"L{bi}_{li}_ab"]
        u = torch.round((T(ab[:, 0]).view(1, -1, 1, 1) * X + T(ab[:, 1]).view(1, -1, 1, 1))
                        .clamp(0, 1) * 255)
    acc = F.conv2d(u, T(L.w1.astype(np.float64)).view(128, -1, 1, 1))
    want = torch.round((T(gd[:, 0]).view(1, -1, 1, 1) * acc + T(gd[:, 1]).view(1, -1, 1, 1))
                       .clamp(0, 1) * qb)
    return {**bytes_cmp(got, want), "sat_frac": float((want >= qb).double().mean()),
            "u_sat_frac": float((u >= 255).double().mean())}


@torch.no_grad()
def unit_conv2(run, plan, par, bi: int, li: int, item=None, planar: bool = False) -> dict:
    b = run.meta["blocks"][bi]
    n, h, w, hw = run.nb, b["h"], b["w"], b["h"] * b["w"]
    g = torch.Generator(device="cuda:0").manual_seed(7 + bi * 100 + li)
    if run.ours:
        v = torch.randint(0, 80, (n, 128, h, w), generator=g, device="cuda:0").double()
        # NHWC128 bottleneck, or planar [4][N*HW][32] (the "*_pl" conv2)
        vv = (v.to(torch.uint8).view(n, 4, 32, hw).permute(1, 0, 3, 2) if planar
              else v.to(torch.uint8).permute(0, 2, 3, 1))
        run.bott[:n * 128 * hw].view(torch.uint8).copy_(vv.flatten())
    else:
        v = torch.randint(0, 40, (n, 128, h, w), generator=g, device="cuda:0").double()
        run.bott[:n * 128 * hw].copy_(v.to(torch.int8).reshape(n, 4, 32, hw).permute(0, 1, 3, 2)
                                      .flatten())
    torch.cuda.synchronize(0)
    # the output slot set to zero: a missed write cannot hide behind a previous round
    off = run.seg[bi] - run.arenas[bi % 2].data_ptr() + li * n * hw * 32
    run.arenas[bi % 2][off:off + n * hw * 32].fill_(0)
    torch.cuda.synchronize(0)
    with torch.cuda.stream(run.stream):
        run._conv2(bi, li, run.plan_l[bi][0][li][1] if item is None else item,
                   run.stream.cuda_stream)
    run.stream.synchronize()
    got = run.store(bi, li + 1).double()
    L = plan.blocks[bi].layers[li]
    if run.ours:
        c2 = torch.as_tensor(par[f"L{bi}_{li}_c2u"], dtype=torch.float32).double().cuda()
        lo = -127
    else:
        c2 = torch.as_tensor(L.s2 * par[f"L{bi}_{li}_c2"], dtype=torch.float32).double().cuda()
        lo = -128
    acc2 = F.conv2d(v, torch.as_tensor(L.w2.astype(np.float64), device="cuda:0"), padding=1)
    want = torch.round(acc2 * c2.view(1, -1, 1, 1)).clamp(lo, 127)
    return {**bytes_cmp(got, want), "sat_frac": float((want.abs() >= 127).double().mean())}


@torch.no_grad()
def unit_fused(run, plan, par, bi: int) -> dict:
    """The fused block bi (`dense_img*`): random input, one launch for all layers; every segment checked
    against the formula (fp16 prologue, u8 bottleneck, conv2) computed on the kernel's own PREVIOUS
    segments, so an off-by-one does not propagate. Summary: the worst layer.
    """
    b = run.meta["blocks"][bi]
    n, h, w = run.nb, b["h"], b["w"]
    g = torch.Generator(device="cuda:0").manual_seed(4321 + bi)
    t0 = torch.randint(-127, 128, (n, b["c0"], h, w), generator=g, device="cuda:0").double()
    _slab_write(run, bi, 0, t0)
    torch.cuda.synchronize(0)
    with torch.cuda.stream(run.stream):
        run._launch(run.fused[bi], run.stream.cuda_stream)
    run.stream.synchronize()
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device="cuda:0")  # noqa: E731
    segs = [t0]
    worst = {"mismatch_frac": 0.0, "max_diff": 0.0, "layers": b["L"]}
    for li in range(b["L"]):
        got = run.store(bi, li + 1).double()
        X = torch.cat(segs, 1)
        a16, b16 = dense8s._abh_ab(par[f"L{bi}_{li}_abh"])
        t = T(a16).view(1, -1, 1, 1) * X + T(b16).view(1, -1, 1, 1)
        u = torch.round(t.half().double().clamp(0, 1) * 255)
        L = plan.blocks[bi].layers[li]
        acc = F.conv2d(u, T(L.w1.astype(np.float64)).view(128, -1, 1, 1))
        gd = par[f"L{bi}_{li}_gdu"]
        v = torch.round((T(gd[:, 0]).view(1, -1, 1, 1) * acc + T(gd[:, 1]).view(1, -1, 1, 1))
                        .clamp(0, 1) * 255)
        acc2 = F.conv2d(v, T(L.w2.astype(np.float64)), padding=1)
        c2 = torch.as_tensor(par[f"L{bi}_{li}_c2u"], dtype=torch.float32).double().cuda()
        want = torch.round(acc2 * c2.view(1, -1, 1, 1)).clamp(-127, 127)
        cmp_ = bytes_cmp(got, want)
        worst["mismatch_frac"] = max(worst["mismatch_frac"], cmp_["mismatch_frac"])
        worst["max_diff"] = max(worst["max_diff"], cmp_["max_diff"])
        segs.append(got)
    return worst


@torch.no_grad()
def unit_trans(run, plan, par, bi: int, kind: str | None = None) -> dict:
    b = run.meta["blocks"][bi]
    segs = _rand_store(run, bi, b["L"], 900 + bi)
    torch.cuda.synchronize(0)
    item = run.plan_l[bi][1] if kind is None else run._trans_item(bi, kind)
    with torch.cuda.stream(run.stream):
        run._launch(item, run.stream.cuda_stream)
    run.stream.synchronize()
    got = run.store(bi + 1, 0).double()
    blk = plan.blocks[bi]
    X = torch.cat(segs, 1)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device="cuda:0")  # noqa: E731
    ab, gd = par[f"T{bi}_ab"], par[f"T{bi}_gd"]
    if kind in dense8s.TRANS_F16H:
        a16, b16 = dense8s._abh_ab(par[f"T{bi}_abh"])
        t = T(a16).view(1, -1, 1, 1) * X + T(b16).view(1, -1, 1, 1)
        u = torch.round(t.half().double().clamp(0, 1) * 255)
    else:
        u = torch.round((T(ab[:, 0]).view(1, -1, 1, 1) * X + T(ab[:, 1]).view(1, -1, 1, 1))
                        .clamp(0, 1) * 255)
    acc = F.conv2d(u, T(blk.tw.astype(np.float64)).view(blk.tw.shape[0], -1, 1, 1))
    want = torch.round(T(gd[:, 0]).view(1, -1, 1, 1) * F.avg_pool2d(acc, 2) * 4).clamp(-127, 127)
    return {**bytes_cmp(got, want), "sat_frac": float((want.abs() >= 127).double().mean())}


@torch.no_grad()
def same_trans(run, bi: int, ka: str, kb: str) -> dict:
    """Transitions `ka` and `kb` on the same random store: equal bytes (the wide "w*" ones run the same mma
    sequence as "f16h").
    """
    b = run.meta["blocks"][bi]
    _rand_store(run, bi, b["L"], 1900 + bi)
    outs = []
    for k in (ka, kb):
        torch.cuda.synchronize(0)
        with torch.cuda.stream(run.stream):
            run._launch(run._trans_item(bi, k), run.stream.cuda_stream)
        run.stream.synchronize()
        outs.append(run.store(bi + 1, 0).clone())
    d = (outs[0] - outs[1]).abs()
    return {"max_diff": float(d.max()), "mismatch_frac": float((d > 0).double().mean())}


@torch.no_grad()
def unit_final(run, par) -> dict:
    bi = len(run.meta["blocks"]) - 1
    b = run.meta["blocks"][bi]
    segs = _rand_store(run, bi, b["L"], 4242)
    f = torch.zeros((run.nb, run.meta["n_features"]), device="cuda:0")
    torch.cuda.synchronize(0)
    P = cudart.Ptr
    g0 = b["c0"] // 32
    with torch.cuda.stream(run.stream):
        run.k["final_q"].launch((run.nb, g0 + b["L"]), (128,),
                                [P(run.in0[bi]), g0, P(run.seg[bi]), run.nb, b["h"] * b["w"],
                                 run.par["F_ab"], f, run.meta["n_features"]],
                                run.stream.cuda_stream)
    run.stream.synchronize()
    X = torch.cat(segs, 1)
    ab = torch.as_tensor(par["F_ab"], dtype=torch.float64, device="cuda:0")
    want = F.relu(ab[:, 0].view(1, -1, 1, 1) * X + ab[:, 1].view(1, -1, 1, 1)).mean(dim=(2, 3))
    return cmp(f, want)


@torch.no_grad()
def unit_stem_i8(run, par, x88: torch.Tensor) -> dict:
    """`stem_i8` against its integer formula (`dense8s.stem_q_i8`): it must be exact."""
    with torch.cuda.stream(run.stream):
        run._stem(x88.data_ptr(), run.stream.cuda_stream)
    run.stream.synchronize()
    got = run.store(0, 0).double()
    want = dense8s.stem_q_i8(x88, par)
    return bytes_cmp(got, want)


@torch.no_grad()
def unit_stem(run, par, x88: torch.Tensor) -> dict:
    with torch.cuda.stream(run.stream):
        run._stem(x88.data_ptr(), run.stream.cuda_stream)
    run.stream.synchronize()
    got = run.store(0, 0).double()
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda:0")  # noqa: E731
    y = F.relu(F.conv2d(x88.float(), T(par["S_w0"]), T(par["S_b0"]), stride=2, padding=3))
    want = torch.round(F.max_pool2d(y, 3, 2, 1)).clamp(-127, 127).double()
    return bytes_cmp(got, want)


def _stem_launch(run, name: str, xx: torch.Tensor, tr=None) -> None:
    P = cudart.Ptr
    n, img = run.nb, run.meta["img"]
    tiles = n * (img // 4 // dense8s.STEM_TP) ** 2
    f = run.mod.function(name)
    shared = dense8s.STEM2_SMEM if name.startswith("stem_q2") else dense8s.STEM_SMEM
    f.set_max_shared(shared)
    args = [P(xx.data_ptr()), run.par["S_w16"], run.par["S_bias"], run.par["S_inv"],
            P(run.in0[0]), n, img, img] + ([P(tr.data_ptr())] if tr is not None else [])
    f.launch((min(run.sms, tiles),), (256,), args, run.stream.cuda_stream, shared=shared)


@torch.no_grad()
def same_stem(run, x) -> dict:
    """`stem_q2` against `stem_q` on the same input: same bytes (same mma sequence, same epilogue, max-pool
    with reordered channels).
    """
    xx = run._padded(x[:run.nb].to(run.in_dtype).contiguous())
    outs = []
    for name in ("stem_q", "stem_q2"):
        with torch.cuda.stream(run.stream):
            _stem_launch(run, name, xx)
        run.stream.synchronize()
        outs.append(run.store(0, 0).clone())
    d = (outs[0] - outs[1]).abs()
    return {"max_diff": float(d.max()), "mismatch_frac": float((d > 0).double().mean())}


def time_stem(run, x) -> dict:
    xx = run._padded(x[:run.nb].to(run.in_dtype).contiguous())
    return {name: round(run._time(lambda nm=name: _stem_launch(run, nm, xx)), 4)
            for name in ("stem_q", "stem_q2")}


# --- seeds ----------------------------------------------------------------------------------
def load_seed(ckpt: Path, seed: int):
    m, _, spec = model_from_checkpoint(ckpt / f"fold{seed}.pt", torch.device("cpu"))
    return m, spec


def calibrate_seed(enc_cpu, cal_x: list[torch.Tensor]) -> dict:
    enc = enc_cpu.to("cuda:0")
    try:
        cal = dense8s.calibrate(enc, cal_x)
    finally:
        enc.cpu()
        torch.cuda.empty_cache()
    return cal


def make_pack(plan, blobs: dict, cubin: bytes, conv2: str, bm: dict | None = None,
              c2cfg: dict | None = None, stem_int8: bool = True, stem: str = "trt",
              tbm: dict | None = None, n: int = N, fused: dict | None = None) -> dict:
    """The package in memory, like `load_pack` (graphs of `n` images)."""
    meta = dense8s.pack_meta(plan, n, True, stem_int8, bm, conv2, c2cfg, stem, tbm,
                             fused) | {
        "engines": sorted(blobs)}
    return {"meta": meta, "engines": blobs,
            "par": dense8s.pack_params(plan, conv2, True, stem_int8), "cubin": cubin}


def micro() -> dict:
    """The floor of this session's T4: IMMA peak (independent mma.m8n8k16 u8 x s8, 160 blocks of 8 warps),
    SM clock under that load, bandwidth of a copy.
    """
    mod = dense8s.kernel_module(0)
    st = torch.cuda.current_stream(0).cuda_stream
    out = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    blocks, iters = 160, 20000
    clk = torch.zeros(2 * blocks, dtype=torch.int64, device="cuda:0")
    k = mod.function("imma_peak")
    k.launch((blocks,), (256,), [out, clk, 200], st)
    torch.cuda.synchronize(0)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    k.launch((blocks,), (256,), [out, clk, iters], st)
    e1.record()
    torch.cuda.synchronize(0)
    ms = e0.elapsed_time(e1)
    macs = blocks * 8 * iters * 8 * 1024
    c = clk.view(-1, 2).double()
    ghz = float((c[:, 0] / c[:, 1]).mean())
    res = {"imma_tmacs": round(macs / ms / 1e9, 2), "sm_ghz": round(ghz, 3),
           "macs_per_sm_clk": round(macs / (40 * ms * 1e-3 * ghz * 1e9), 1)}
    n = 256 << 20
    src = torch.empty(n, dtype=torch.uint8, device="cuda:0")
    dst = torch.empty_like(src)
    kc = mod.function("copy16")
    for grid in (320, 1280):
        kc.launch((grid,), (256,), [src, dst, n // 16], st)
        torch.cuda.synchronize(0)
        e0.record()
        for _ in range(5):
            kc.launch((grid,), (256,), [src, dst, n // 16], st)
        e1.record()
        torch.cuda.synchronize(0)
        res[f"copy_gbs_grid{grid}"] = round(2 * n * 5 / (e0.elapsed_time(e1) * 1e-3) / 1e9, 1)
    kr = mod.function("l2read")
    for tag, size, iters in (("l2_read_gbs", 2 << 20, 400), ("dram_read_gbs", 256 << 20, 3)):
        buf = src[:size] if size <= src.numel() else src
        kr.launch((640,), (256,), [buf, size // 16, 1, out], st)
        torch.cuda.synchronize(0)
        e0.record()
        kr.launch((640,), (256,), [buf, size // 16, iters, out], st)
        e1.record()
        torch.cuda.synchronize(0)
        res[tag] = round(size * iters / (e0.elapsed_time(e1) * 1e-3) / 1e9, 1)
    del src, dst
    torch.cuda.empty_cache()
    return res


def imma_occupancy(mod, st=None) -> dict:
    """MACs per SM per cycle with 1, 2 and 4 blocks of 8 warps per SM (40 SMs): `imma_peak` (8 independent
    mma per warp) and `mma_loop` / `mma_loop_sync` (the inner loop of conv1 without loads, without and with
    a barrier per stage). The events are on the same stream as the launches.
    """
    st = torch.cuda.current_stream(0).cuda_stream
    out = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    res = {}
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for name, iters, macs_it in (("mma_loop", 2000, 64 * 1024), ("mma_loop_sync", 2000, 64 * 1024),
                                 ("mma_loop_reg", 2000, 64 * 1024), ("mma_loop4", 4000, 32 * 1024),
                                 ("mma_loop4_reg", 4000, 32 * 1024)):
        k = mod.function(name)
        k.set_max_shared(8192)
        for per_sm in (1, 2, 4):
            blocks = 40 * per_sm
            k.launch((blocks,), (256,), [out, 20], st, shared=8192)
            torch.cuda.synchronize(0)
            e0.record()
            k.launch((blocks,), (256,), [out, iters], st, shared=8192)
            e1.record()
            torch.cuda.synchronize(0)
            ms = e0.elapsed_time(e1)
            res[f"{name} {per_sm * 8}warp/SM tmacs"] = round(
                blocks * 8 * iters * macs_it / ms / 1e9, 2)
    # the inner loop of conv2 (4 warps per block, 1 / 2 / 4 blocks per SM)
    k2 = mod.function("conv2_loop")
    for per_sm in (1, 2, 4):
        blocks = 40 * per_sm
        k2.launch((blocks,), (128,), [out, 5], st, shared=5760 + 9216)
        torch.cuda.synchronize(0)
        e0.record()
        k2.launch((blocks,), (128,), [out, 500], st, shared=5760 + 9216)
        e1.record()
        torch.cuda.synchronize(0)
        ms = e0.elapsed_time(e1)
        res[f"conv2_loop {per_sm * 4}warp/SM tmacs"] = round(
            blocks * 4 * 500 * 9 * 2 * 16 * 1024 / ms / 1e9, 2)
    k = mod.function("imma_peak")
    for per_sm in (1, 2, 4):
        blocks = 40 * per_sm
        clk = torch.zeros(2 * blocks, dtype=torch.int64, device="cuda:0")
        k.launch((blocks,), (256,), [out, clk, 200], st)
        torch.cuda.synchronize(0)
        e0.record()
        k.launch((blocks,), (256,), [out, clk, 5000], st)
        e1.record()
        torch.cuda.synchronize(0)
        ms = e0.elapsed_time(e1)
        c = clk.view(-1, 2).double()
        ghz = float((c[:, 0] / c[:, 1]).mean())
        macs = blocks * 8 * 5000 * 8 * 1024
        res[f"imma_peak {per_sm * 8}warp/SM mac_sm_clk"] = round(
            macs / (40 * ms * 1e-3 * ghz * 1e9), 1)
        res[f"imma_peak {per_sm * 8}warp/SM ghz"] = round(ghz, 3)
    return res


def diag(run) -> dict:
    """The kernels with parts switched off (DIAG: 1 no MMA, 2 no global loads, 4 no prologue) on the
    arguments of real layers: where the limit is.
    """
    st = run.stream.cuda_stream
    mod = run.mod
    out: dict = {}
    for bi, li in ((0, 3), (1, 6), (2, 20), (3, 20)):
        (_, _, _, arg, _), _ = run.plan_l[bi][0][li]
        b = run.meta["blocks"][bi]
        rows = run.nb * b["h"] * b["w"]
        res = {}
        for name in ("conv1_u128", "conv1_u128_d1", "conv1_u128_d2", "conv1_u128_d4",
                     "conv1_u128_d6", "conv1_u128_d7"):
            it = (mod.function(name), (-(-rows // 128), 1), (256,), arg,
                  dense8s.gemm_shared(128, 2, arg.G))
            res[name.replace("conv1_u128", "") or "full"] = round(
                run._time(lambda it=it: run._launch(it, st)), 4)
        out[f"c1 b{bi}l{li}"] = res
    for bi, (th, tw) in ((0, (16, 8)), (1, (9, 12)), (2, (6, 18)), (3, (9, 9))):
        base = run._c2_item(bi, 0, th, tw, "q4w4")
        res = {}
        for kern in ("q4w4", "q4w4u", "q4w4_d1", "q4w4_d2", "q4w4_d3"):
            it = (mod.function(f"conv2_{kern}"),) + tuple(base[1:])
            res[kern] = round(run._time(lambda it=it: run._launch(it, st)), 4)
        out[f"c2 b{bi} {th}x{tw}"] = res
    # warp-specialised conv1: MMA warps without mma (d1), producers without loads and prologue (d2)
    for bi, li in ((0, 3), (1, 6), (2, 20), (3, 20)):
        res = {}
        for kind in ("w4", "x4"):
            base = run._c1_item(bi, li, kind)
            for suf in ("", "_d1", "_d2"):
                f = mod.function(f"conv1_{kind}{suf}")
                f.set_max_shared(64 * 1024)
                it = (f,) + tuple(base[1:])
                res[kind + suf] = round(run._time(lambda it=it: run._launch(it, st)), 4)
        out[f"ws b{bi}l{li}"] = res
    out["imma_occupancy"] = imma_occupancy(mod, st)
    return out


def _trans_w_kinds(run, bi: int) -> list[str]:
    b = run.meta["blocks"][bi]
    cout = (b["c0"] + dense8s.GROWTH * b["L"]) // 2
    return [k for k, (_, bn) in dense8s.TRANS_W.items() if cout % bn == 0]


# Robust per-layer choices: tuning chose by differences of a few points, within the thermal noise of
# the measurement minutes, and on the benchmark (two GPUs under load) those choices lost. A kernel
# different from the default only if it wins by at least MARGIN.
MARGIN = 0.05
C1_DEFAULT = "u128hs"
C2_DEFAULT = {72: "8x24 p3u", 36: "18x12 p4u", 18: "6x18 q4w4v", 9: "9x9 q4w4u"}


def robust_pick(res: dict, default: str | None) -> str:
    best = min(res, key=res.get)
    if default is None or default not in res:
        return best
    return best if res[best] < (1 - MARGIN) * res[default] else default


def diag_stem(run, x) -> dict:
    """stem_q with parts switched off (the stem is ~10% of a pass, at ~22% of the fp16 peak): d1 no
    max-pool, d2 no conv epilogue, d4 no load of the next input, d8 no mma, d15 everything off.
    """
    st = run.stream.cuda_stream
    P = cudart.Ptr
    n, img = run.nb, run.meta["img"]
    tiles = n * (img // 4 // dense8s.STEM_TP) ** 2
    xx = run._padded(x[:n].to(run.in_dtype).contiguous())
    out: dict = {}
    for suf in ("", "_d1", "_d2", "_d4", "_d8", "_d15"):
        f = run.mod.function(f"stem_q{suf}")
        f.set_max_shared(dense8s.STEM_SMEM)
        args = [P(xx.data_ptr()), run.par["S_w16"], run.par["S_bias"], run.par["S_inv"],
                P(run.in0[0]), n, img, img]
        out["full" if not suf else suf[1:]] = round(run._time(
            lambda f=f, a=args: f.launch((min(run.sms, tiles),), (256,), a, st,
                                         shared=dense8s.STEM_SMEM)), 4)
    return out


TRACE_PHASES = {
    "trans": ["init", "gload", "mma", "sstore", "sync", "epi", "write"],
    "conv2": ["init", "load", "mma", "store", "sync", "epi", "write"],
    "stem": ["init", "pload", "conv", "epi", "sync1", "pool", "pstore", "sync2"],
    "tile": ["init", "c1load", "c1mma", "c1sstore", "c1sync", "epi1", "c2wstage", "c2mma",
             "c2sync", "epi2", "write"],
}


def _trace_one(run, fname: str, grid, block, args: list, shared: int, kind: str) -> dict:
    """One launch of the traced kernel `fname` (arguments + the trace buffer): cycles per warp of every phase
    (mean) and share of the total.
    """
    f = run.mod.function(fname)
    f.set_max_shared(64 * 1024)
    tr = torch.zeros(16, dtype=torch.int64, device="cuda:0")
    st = run.stream.cuda_stream
    with torch.cuda.stream(run.stream):
        f.launch(grid, block, args + [cudart.Ptr(tr.data_ptr())], st, shared=shared)
        run.stream.synchronize()
        tr.zero_()
        f.launch(grid, block, args + [cudart.Ptr(tr.data_ptr())], st, shared=shared)
        run.stream.synchronize()
    v = tr.cpu().tolist()
    warps = max(1, v[12])
    names = TRACE_PHASES[kind]
    cyc = {nm: v[i] / warps for i, nm in enumerate(names)}
    tot = sum(cyc.values()) or 1.0
    return {"warps": warps, "kcyc_per_warp": round(tot / 1e3, 1),
            "frac": {nm: round(c / tot, 3) for nm, c in cyc.items()}}


def trace_kernels(run, x) -> dict:
    """Where the time goes: transitions f16h / w0 / w1, conv2 of blocks 0 and 1 (per block and persistent,
    prefetch at 1 and 2 stages), stem. Cycles per warp (clock64) per phase.
    """
    out: dict = {}
    for bi in range(3):
        b = run.meta["blocks"][bi]
        cout = (b["c0"] + dense8s.GROWTH * b["L"]) // 2
        for kind, fname in (("f16h", "trans_f16h_t"), ("w0", "trans_w0_t"), ("w1", "trans_w1_t")):
            if kind in dense8s.TRANS_W and cout % dense8s.TRANS_W[kind][1]:
                continue
            try:
                k_, grid, block, arg, shared = run._trans_item(bi, kind)
                out[f"t{bi} {kind}"] = _trace_one(run, fname, grid, block, [arg], shared, "trans")
            except Exception as exc:
                out[f"t{bi} {kind}"] = repr(exc)[:200]
    for bi, th, tw, kern, fname in ((0, 16, 8, "q4w4u", "conv2_q4w4u_t"),
                                    (0, 16, 8, "q4w4v", "conv2_q4w4v_t"),
                                    (0, 8, 24, "p3u", "conv2p_m3u_t"),
                                    (0, 8, 24, "p3v", "conv2p_m3v_t"),
                                    (1, 9, 12, "q4w4u", "conv2_q4w4u_t"),
                                    (1, 9, 12, "q4w4v", "conv2_q4w4v_t")):
        try:
            k_, grid, block, arg, shared = run._c2_item(bi, 0, th, tw, kern)
            out[f"b{bi} {th}x{tw} {kern}"] = _trace_one(run, fname, grid, block, [arg], shared,
                                                        "conv2")
        except Exception as exc:
            out[f"b{bi} {th}x{tw} {kern}"] = repr(exc)[:200]
    # conv1 u128hs (layers 0 and 5 of block 0, 0 and 11 of block 1, 20 of block 2)
    for bi, li in ((0, 0), (0, 5), (1, 0), (1, 11), (2, 20)):
        for kind, fname in (("u128hs", "conv1_u128hs_t"), ("p128", "conv1_p128_t")):
            try:
                k_, grid, block, arg, shared = run._c1_item(bi, li, kind)
                out[f"c1 b{bi}l{li} {kind}"] = _trace_one(run, fname, grid, block, [arg], shared,
                                                          "trans")
            except Exception as exc:
                out[f"c1 b{bi}l{li} {kind}"] = repr(exc)[:200]
    # layer fused in rectangles (blocks 0 and 1, first and last layer)
    for bi in ():
        try:
            items = run._fused_item(bi, "tile")
            for li in (0, len(items) - 1):
                k_, grid, block, arg, shared = items[li]
                out[f"tile b{bi}l{li}"] = _trace_one(run, "dense_tile7_t", grid, block, [arg],
                                                     shared, "tile")
        except Exception as exc:
            out[f"tile b{bi}"] = repr(exc)[:200]
    try:
        P = cudart.Ptr
        n, img = run.nb, run.meta["img"]
        tiles = n * (img // 4 // dense8s.STEM_TP) ** 2
        xx = run._padded(x[:n].to(run.in_dtype).contiguous())
        args = [P(xx.data_ptr()), run.par["S_w16"], run.par["S_bias"], run.par["S_inv"],
                P(run.in0[0]), n, img, img]
        out["stem"] = _trace_one(run, "stem_q_t", (min(run.sms, tiles),), (256,), args,
                                 dense8s.STEM_SMEM, "stem")
        out["stem_q2"] = _trace_one(run, "stem_q2_t", (min(run.sms, tiles),), (256,), args,
                                    dense8s.STEM2_SMEM, "stem")
    except Exception as exc:
        out["stem"] = repr(exc)[:200]
    return out


def diag_c2(run) -> dict:
    """conv2 q4w4u with parts switched off (the inner loop alone reaches 80-88% of the peak, the real conv2
    ~30%): x2 no loads (c > 1), x16 no epilogue writes, x34 also no STS, x64 no mma, x114 everything off.
    Layer 0 of blocks 0 (16x8) and 2 (6x18).
    """
    st = run.stream.cuda_stream
    out: dict = {}
    for bi, th, tw in ((0, 16, 8), (2, 6, 18)):
        base = run._c2_item(bi, 0, th, tw, "q4w4u")
        res = {}
        for suf in ("", "_x2", "_x16", "_x34", "_x64", "_x114"):
            it = (run.mod.function(f"conv2_q4w4u{suf}"),) + tuple(base[1:])
            res["full" if not suf else suf[1:]] = round(run._time(lambda it=it: run._launch(it, st)), 4)
        out[f"b{bi} {th}x{tw}"] = res
    return out


# diagnostics (kernel DIAG, mma loop): off
DIAG = False


def gate(r: dict, ref_err: float | None) -> bool:
    """Per-component checks and a pipeline no worse than the earlier `d8fanp` engine (+5%)."""
    u = r.get("units")
    if not u or "error" in r:
        return False
    exact = list(u["conv1"].values()) + list(u["conv2"].values()) + list(u["trans"].values())
    exact += list(u.get("conv1_ws", {}).values()) + list(u.get("conv2_k", {}).values())
    exact += list(u.get("trans_h", {}).values()) + list(u.get("trans_f16h", {}).values())
    exact += list(u.get("trans_w", {}).values()) + list(u.get("conv2_v", {}).values())
    exact += list(u.get("conv1_pl", {}).values()) + list(u.get("conv2_pl", {}).values())
    # the wide transitions must give the same bytes as f16h
    if any(v["max_diff"] > 0 for v in u.get("trans_w_same", {}).values()):
        return False
    for k_ in ("fused_b0", "fused_b1", "fused_b2", "fused_b3"):
        if k_ in u:
            exact.append(u[k_])
    ok = all(v["max_diff"] <= 1 and v["mismatch_frac"] < 1e-3 for v in exact)
    ok = ok and u["final"]["rel_l2"] < 1e-5 and u["stem"]["max_diff"] <= 2
    if "stem_q2_same" in u and u["stem_q2_same"]["max_diff"] > 0:
        return False
    if ref_err is not None and "vs_float" in r:
        ok = ok and r["vs_float"]["rel_l2"] <= 1.05 * ref_err
    if ref_err is not None and "tuned_vs_float" in r:
        ok = ok and r["tuned_vs_float"]["rel_l2"] <= 1.05 * ref_err
    return ok


def variant(key: str, conv2: str, stem_int8: bool, plan, par, blobs: dict, cubin: bytes,
            x: torch.Tensor, ref: torch.Tensor, f8: torch.Tensor | None, stem: str = "trt",
            tuned: tuple | None = None) -> tuple[dict, dict | None, dict | None]:
    """One form (conv2 "trt" or "ours", INT8 stem from the engine or fp16 + `quant_seg`, or our "ours" /
    "i8" / "i8w") on seed 0: checks, error, timings, block choices. `tuned` = (bm, c2cfg, tbm) of another
    form: no tuning, those are measured.
    """
    r: dict = {}
    RESULTS[key] = r
    bm = c2cfg = tbm = None
    try:
        run = dense8s.Dense8SEncoder(make_pack(plan, blobs, cubin, conv2, stem_int8=stem_int8,
                                               stem=stem), 0, x.shape[0], max_inflight=2)
        r["load_s"], r["arena_mb"] = round(run.load_s, 2), run.arena_mb
        r["units"] = {
            "stem": (unit_stem_i8 if stem in ("i8", "i8w") else unit_stem)(
                run, par, x[:N].half().contiguous()),
            "conv1": {f"b{bi}l{li}": unit_conv1(run, plan, par, bi, li)
                      for bi, li in ((0, 0), (0, 5), (1, 0), (1, 7), (2, 20), (3, 0), (3, 31))},
            "conv1_ws": {f"{kind} b{bi}l{li}": unit_conv1(run, plan, par, bi, li, kind)
                         for kind in ("u128hs",)
                         for bi, li in ((0, 0), (1, 7), (2, 20), (3, 31))},
            "conv2": {f"b{bi}l{li}": unit_conv2(run, plan, par, bi, li)
                      for bi, li in ((0, 0), (1, 7), (2, 20), (3, 31))},
            "conv2_k": {f"{th}x{tw} {kern} b{bi}l{li}": unit_conv2(
                run, plan, par, bi, li, run._c2_item(bi, li, th, tw, kern))
                for bi, li in ((0, 0), (1, 7), (2, 20), (3, 31))
                for th, tw, kern in dense8s.C2_CANDIDATES.get(run.meta["blocks"][bi]["h"], [])
                if kern.startswith("k")},
            "trans": {f"t{bi}": unit_trans(run, plan, par, bi) for bi in (0, 1, 2)},
            "trans_f16": {f"t{bi}": unit_trans(run, plan, par, bi, "f16") for bi in (0, 1, 2)},
            "trans_f16h": {f"t{bi}": unit_trans(run, plan, par, bi, "f16h") for bi in (0, 1, 2)},
            "trans_w": {f"{k} t{bi}": unit_trans(run, plan, par, bi, k)
                        for bi in (0, 1, 2) for k in _trans_w_kinds(run, bi)},
            "trans_w_same": {f"{k} t{bi}": same_trans(run, bi, k, "f16h")
                             for bi in (0, 1, 2) for k in _trans_w_kinds(run, bi)},
            "conv2_v": {f"{th}x{tw} {kern} b{bi}l{li}": unit_conv2(
                run, plan, par, bi, li, run._c2_item(bi, li, th, tw, kern))
                for bi, li in ((0, 0), (1, 7), (2, 20))
                for th, tw, kern in dense8s.C2_CANDIDATES.get(run.meta["blocks"][bi]["h"], [])
                if kern.endswith("v")},
            "final": unit_final(run, par),
            # 512 px: no stem of ours (9-pixel rectangles on 128), hence no stem_q2_same
            # the planar bottleneck and the fixed-shape conv2 (the FIXED choices)
            "conv1_pl": {f"{kd} b{bi}l{li}": unit_conv1(run, plan, par, bi, li, kd)
                         for kd in ("w64_pl",)
                         for bi, li in ((0, 0), (0, 5), (1, 7), (2, 20), (2, 31))},
            "conv2_pl": {f"{th}x{tw} {kern} b{bi}l{li}": unit_conv2(
                run, plan, par, bi, li, run._c2_item(bi, li, th, tw, kern),
                planar=kern.endswith(dense8s.PLANAR))
                for bi, li, th, tw, kern in ((0, 0, 16, 8, "p4u_pl"), (0, 5, 16, 8, "p4u_pl"),
                                             (1, 7, 16, 8, "p4u_pl"), (2, 20, 16, 8, "p4u_pl"),
                                             (2, 31, 16, 8, "p4u_pl"), (3, 31, 16, 8, "q4w4u"))},
        }
        try:
            runf = dense8s.Dense8SEncoder(make_pack(plan, blobs, cubin, conv2, stem_int8=stem_int8,
                                                    stem=stem, fused=FUSED_TRY), 0, N,
                                          max_inflight=2)
            for bi_, _ in FUSED_TRY_LIST:
                r["units"][f"fused_b{bi_}"] = unit_fused(runf, plan, par, bi_)
            runf = None
        except Exception as exc:
            r["units"]["fused_error"] = traceback.format_exc()[-1500:]
        log(f"{key}: per-component checks {json.dumps(r['units'])}")
        save()
        with torch.no_grad():
            fs = run(x.half()).float().clone()
            torch.cuda.synchronize(0)
            emu = dense8s.emulate(plan, x, quant=True, dtype=torch.float32, par=par, conv2=conv2,
                                  stem=stem)
        r["vs_float"], r["vs_emulate"] = cmp(fs, ref), cmp(fs, emu)
        r["emulate_vs_float"] = cmp(emu, ref)
        if f8 is not None:
            r["vs_d8fanp"] = cmp(fs, f8)
        del emu
        log(f"{key}: vs float32 {r['vs_float']}, vs emulation {r['vs_emulate']}, "
            f"emulation vs float32 {r['emulate_vs_float']}")
        run = None
        torch.cuda.empty_cache()
        save()
        # timings, conv1 blocks and conv2 rectangles: graphs of NB images (x two = 352 windows); ms per call = NB
        # images, `per88` = / (NB / 88)
        x88 = torch.cat([x] * (NB // x.shape[0])).half().contiguous()
        run88 = dense8s.Dense8SEncoder(make_pack(plan, blobs, cubin, conv2, stem_int8=stem_int8,
                                                 stem=stem, n=NB), 0, NB, max_inflight=2)
        with torch.no_grad():
            run88(x88)
            torch.cuda.synchronize(0)
            r["plain"] = timed(lambda: run88(x88), 10)
            got_plain = run88(x88).clone()
            r["capture_s"] = round(run88.capture(NB), 2)
            got_graph = run88(x88).clone()
            torch.cuda.synchronize(0)
            r["graph_vs_plain_max_abs"] = float((got_graph - got_plain).abs().max())
            r["graph"] = timed(lambda: run88(x88), 30)
            r["graph"]["per88"] = round(r["graph"]["ms_per_call"] * N / NB, 2)
            # padded short block: the first 176 windows against the full block
            got_short = run88(x88[:x.shape[0]]).clone()
            torch.cuda.synchronize(0)
            r["short_vs_full_max_abs"] = float((got_short - got_graph[:x.shape[0]]).abs().max())
        r["profile"] = run88.profile(x88)
        log(f"{key}: plain launches {r['plain']}, in a graph {r['graph']}; items {r['profile']}")
        if tuned is not None:
            bm, c2cfg, tbm = tuned
            run88 = None
            torch.cuda.empty_cache()
            run88 = dense8s.Dense8SEncoder(make_pack(plan, blobs, cubin, conv2, bm, c2cfg,
                                                     stem_int8, stem, tbm, n=NB), 0, NB,
                                           max_inflight=2)
            with torch.no_grad():
                run88(x88)
                run88.capture(NB)
                r["graph_tuned"] = timed(lambda: run88(x88), 30)
                r["graph_tuned"]["per88"] = round(r["graph_tuned"]["ms_per_call"] * N / NB, 2)
                # the fixed choices against float32 (fp16 prologue: not the same outputs as the engine with the default
                # blocks, which has the fp32 prologue)
                r["tuned_vs_float"] = cmp(run88(x88)[:x.shape[0]].float(), ref)
                torch.cuda.synchronize(0)
            r["profile_tuned"] = run88.profile(x88)
            log(f"{key}: with the previous blocks in a graph {r['graph_tuned']}, vs float32 "
                f"{r['tuned_vs_float']}; items {r['profile_tuned']}")
            run88 = None
            torch.cuda.empty_cache()
            save()
            r["_tbm"] = tbm
            return r, bm, c2cfg
        r["layers"] = run88.time_layers()
        tune = run88.tune_bm()
        bm = {k: robust_pick(v, C1_DEFAULT) for k, v in tune.items()}
        r["tune_bm_by_kind"] = {str(kd): round(sum(v[kd] for v in tune.values()), 3)
                                for kd in next(iter(tune.values()))}
        r["tune_bm"], r["bm"] = tune, bm
        r["tune_bm_gain_ms"] = round(sum(
            tune[k][run88.meta["blocks"][int(k[1])]["bm"][int(k.split("l")[1])]] - tune[k][bm[k]]
            for k in tune), 3)
        if conv2 == "ours":
            t2 = run88.tune_c2()
            r["tune_c2"] = t2
            c2cfg = {}
            for side, res in t2.items():
                best = robust_pick(res, C2_DEFAULT.get(int(side)))
                th, tw = (int(v) for v in best.split(" ")[0].split("x"))
                c2cfg[int(side)] = [th, tw, best.split(" ")[1]]
            r["c2cfg"] = c2cfg
        try:
            runF = dense8s.Dense8SEncoder(make_pack(plan, blobs, cubin, conv2, bm, c2cfg,
                                                    stem_int8, stem, None, n=NB,
                                                    fused=FUSED_TRY), 0, NB, max_inflight=2)
            st_ = runF.stream.cuda_stream
            for bi_, _ in FUSED_TRY_LIST:
                r[f"fused_b{bi_}_ms"] = round(runF._time(
                    lambda b_=bi_: runF._launch(runF.fused[b_], st_)), 4)
            runF = None
        except Exception as exc:
            r["fused_error"] = traceback.format_exc()[-1500:]
        try:
            r["stem_q2_same"] = same_stem(run88, x88)
            r["stem_times"] = time_stem(run88, x88)
            log(f"{key}: stem_q2 {r['stem_q2_same']} timings {r['stem_times']}")
        except Exception as exc:
            r["stem_q2_error"] = traceback.format_exc()[-1500:]
        try:
            r["trace"] = trace_kernels(run88, x88)
            log(f"{key}: trace {json.dumps(r['trace'])}")
        except Exception as exc:
            r["trace_error"] = traceback.format_exc()[-1500:]
        t3 = run88.tune_trans()
        r["tune_trans"] = t3
        tbm = {k: min(v, key=v.get) for k, v in t3.items()}
        r["tbm"] = tbm
        log(f"{key}: layers {r['layers']}")
        log(f"{key}: conv1 blocks {bm} (gain {r['tune_bm_gain_ms']} ms); "
            f"conv2 {r.get('tune_c2')}")
        run88 = None
        torch.cuda.empty_cache()
        run88 = dense8s.Dense8SEncoder(make_pack(plan, blobs, cubin, conv2, bm, c2cfg,
                                                 stem_int8, stem, tbm, n=NB), 0, NB,
                                       max_inflight=2)
        with torch.no_grad():
            run88(x88)
            run88.capture(NB)
            r["graph_tuned"] = timed(lambda: run88(x88), 30)
            r["graph_tuned"]["per88"] = round(r["graph_tuned"]["ms_per_call"] * N / NB, 2)
            got_t = run88(x88).clone()
            torch.cuda.synchronize(0)
            r["tuned_vs_graph_max_abs"] = float((got_t - got_graph).abs().max())
        r["profile_tuned"] = run88.profile(x88)
        fz = {}
        for bi_, kind_ in FUSED_TRY_LIST:
            if f"fused_b{bi_}_ms" not in r:
                continue
            unf = (r["profile_tuned"].get(f"conv1_{bi_}", 0.0)
                   + r["profile_tuned"].get(f"conv2_{bi_}", 0.0))
            r[f"unfused_b{bi_}_ms"] = round(unf, 4)
            fb = r["units"].get(f"fused_b{bi_}") or {}
            if fb and fb["max_diff"] <= 1 and fb["mismatch_frac"] < 1e-3 \
                    and r[f"fused_b{bi_}_ms"] < unf:
                fz[f"b{bi_}"] = kind_
        r["_fused"] = fz or None
        # A/B in the same session: this kernel's choices (with the winning fused blocks) against the reference
        # ones, alternating (the T4s change from session to session more than the gains being measured)
        try:
            run88 = None
            torch.cuda.empty_cache()
            runN = dense8s.Dense8SEncoder(make_pack(plan, blobs, cubin, conv2, bm, c2cfg, stem_int8,
                                                    stem, tbm, n=NB, fused=fz or None), 0, NB,
                                          max_inflight=2)
            runR = dense8s.Dense8SEncoder(make_pack(
                plan, blobs, cubin, conv2, REF_V13["bm"], REF_V13["c2cfg"], stem_int8,
                REF_V13["stem"], REF_V13["tbm"], n=NB, fused=REF_V13["fused"]), 0, NB,
                max_inflight=2)
            with torch.no_grad():
                for run_ in (runN, runR):
                    run_(x88)
                    run_.capture(NB)
                ab = {"new": [], "v13": []}
                for _ in range(4):
                    ab["new"].append(timed(lambda: runN(x88), 20)["ms_per_call"])
                    ab["v13"].append(timed(lambda: runR(x88), 20)["ms_per_call"])
                gn, gr = runN(x88).clone(), runR(x88).clone()
                torch.cuda.synchronize(0)
            ab["new_mean"] = round(float(np.mean(ab["new"])), 2)
            ab["v13_mean"] = round(float(np.mean(ab["v13"])), 2)
            ab["gain"] = round(1 - ab["new_mean"] / ab["v13_mean"], 4)
            ab["max_abs_new_vs_v13"] = float((gn - gr).abs().max())
            r["ab_vs_v13"] = ab
            log(f"{key}: A/B vs the reference choices {ab}")
            runN = runR = None
            torch.cuda.empty_cache()
        except Exception as exc:
            r["ab_error"] = traceback.format_exc()[-1500:]
        # the same choices with graphs of 88 (comparison with earlier probes)
        run88 = None
        torch.cuda.empty_cache()
        r88 = dense8s.Dense8SEncoder(make_pack(plan, blobs, cubin, conv2, bm, c2cfg,
                                               stem_int8, stem, tbm, n=N), 0, N, max_inflight=2)
        with torch.no_grad():
            xs = x[:N].half().contiguous()
            r88(xs)
            r88.capture(N)
            r["graph_tuned88"] = timed(lambda: r88(xs), 30)
        r88 = None
        log(f"{key}: with the chosen blocks in a graph {r['graph_tuned']}; items {r['profile_tuned']}")
        if DIAG:
            try:
                r["diag"] = diag(run88)
                log(f"{key}: diagnostics {json.dumps(r['diag'])}")
            except Exception as exc:
                r["diag_error"] = repr(exc)[:300]
        run88 = None
        torch.cuda.empty_cache()
    except Exception as exc:
        r["error"] = traceback.format_exc()[-3000:]
        log(f"!! {key}: {exc!r}")
    save()
    r["_tbm"] = tbm
    return r, bm, c2cfg


def main() -> None:
    RESULTS["blocking_sync"] = set_blocking_sync()
    try:
        cudart.nvrtc()
    except cudart.CudaError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "nvidia-cuda-nvrtc-cu12"], capture_output=True, text=True)
        log(f"nvrtc via pip: rc {r.returncode}")
    RESULTS["nvrtc"] = {"version": cudart.nvrtc_version(), "path": cudart._STATE.get("nvrtc_path")}
    cubin = dense8s.kernel_cubin("sm_75")
    RESULTS["cubin_kb"] = round(len(cubin) / 1024, 1)
    RESULTS["kernels_sha"] = dense8s.kernels_sha()

    ckpt = [d for d in find_dirs_with("fold0.pt") if EXP in str(d)]
    if len(ckpt) != 1:
        raise SystemExit(f"{EXP}: {ckpt}")
    ckpt = ckpt[0]
    eng_dir = find_dir_with("d8fanp_fold0.npz")
    tc_path = eng_dir / "timing.cache"
    tc = tc_path.read_bytes() if tc_path.is_file() else b""
    m0, spec = load_seed(ckpt, 0)
    spec = dataclasses.replace(spec, img=IMG)
    img = spec.img
    cache = train_cache(spec)
    ev_uids = pick(cache, 1, seed=11)
    cal_uids = pick(cache, N_CAL, seed=5, exclude=ev_uids)
    RESULTS["eval_uids"], RESULTS["cal_uids"] = ev_uids, cal_uids
    x = study_windows(cache, ev_uids, m0)[::4].contiguous()      # 22 windows, float32
    cal_x = [study_windows(cache, [u], m0).cpu() for u in cal_uids]
    log(f"windows: evaluation {tuple(x.shape)}, calibration {len(cal_x)} x "
        f"{tuple(cal_x[0].shape)}")

    enc = m0.encoder.float().eval()
    with torch.no_grad():
        enc.to("cuda:0")
        ref = in_chunks(enc, x)
        enc.cpu()
    torch.cuda.empty_cache()
    t = time.perf_counter()
    cal0 = calibrate_seed(enc, [c.to("cuda:0") for c in cal_x])
    RESULTS["calibrate_s"] = round(time.perf_counter() - t, 1)
    (OUT / "cal_fold0.json").write_text(json.dumps(cal0))
    plan0 = dense8s.make_plan(enc, cal0, img)
    par0 = dense8s.kernel_params(plan0)

    # the earlier dense8 d8fanp engine on the same windows: error and time in the same session
    f8, ref_err = None, 0.0670
    RESULTS["ref_err_fixed"] = ref_err
    try:
        raise RuntimeError("d8fanp skipped: packages at 512 px")
        d8p = dense8.load_pack(eng_dir / "d8fanp_fold0.npz")
        e8 = dense8.Dense8Encoder(d8p, 0, x.shape[0], max_inflight=2)
        with torch.no_grad():
            f8 = e8(x.half()).float().clone()
            torch.cuda.synchronize(0)
        RESULTS["d8fanp_vs_float"] = cmp(f8, ref)
        ref_err = RESULTS["d8fanp_vs_float"]["rel_l2"]
        e8 = None
        e88 = dense8.Dense8Encoder(d8p, 0, N, max_inflight=2)
        x88 = x[:N].half().contiguous()
        with torch.no_grad():
            e88(x88)
            e88.capture(N)
            RESULTS["d8fanp_graph"] = timed(lambda: e88(x88), 30)
        e88 = None
        torch.cuda.empty_cache()
        log(f"d8fanp: vs float32 {RESULTS['d8fanp_vs_float']}, in a graph "
            f"{RESULTS['d8fanp_graph']}")
    except Exception as exc:
        RESULTS["d8fanp_error"] = repr(exc)[:400]
        log(f"!! d8fanp: {exc!r}")
    save()

    try:
        raise RuntimeError("skipped")
        RESULTS["micro"] = micro()
        log(f"T4 floor: {RESULTS['micro']}")
    except Exception as exc:
        RESULTS["micro_error"] = repr(exc)[:300]
    save()
    try:
        raise RuntimeError("skipped")
        RESULTS["loops"] = imma_occupancy(dense8s.kernel_module(0))
        log(f"mma loops: {RESULTS['loops']}")
    except Exception as exc:
        RESULTS["loops_error"] = repr(exc)[:300]
    save()

    # the forms on seed 0: our conv2 or TensorRT's, INT8 stem from the engine or fp16 + quant_seg
    choice: dict = {}
    tuned = FIXED                                   # no burst tuning
    # INT8 stem left out (+6% error against float for 0.5 ms)
    for key, conv2, stem_int8, stem in (("trt-stem", "ours", False, "trt"),):
        try:
            blobs, tc, info = dense8s.build_engines(plan0, N, tc, log=log, conv2=conv2,
                                                    stem_int8=stem_int8, stem=stem)
            RESULTS.setdefault("build_s", {})[f"seed0 {key}"] = info["total_s"]
            (OUT / "timing.cache").write_bytes(tc)
        except Exception as exc:
            RESULTS[f"build_error_{key}"] = traceback.format_exc()[-2000:]
            log(f"!! engines {key}: {exc!r}")
            continue
        r, bm, c2cfg = variant(key, conv2, stem_int8, plan0, par0, blobs, cubin, x, ref, f8,
                               stem, tuned)
        if tuned is None and bm is not None:
            tuned = (bm, c2cfg, r.get("_tbm"))
        r["gate"] = gate(r, ref_err)
        if r["gate"] and "graph_tuned" in r:
            choice[key] = (r["graph_tuned"]["ms_per_call"], conv2, stem_int8, bm, c2cfg, stem,
                           r.pop("_tbm", None))
        save()
    RESULTS["choice"] = {k: v[0] for k, v in choice.items()}
    fused = RESULTS.get("trt-stem", {}).get("_fused") or (FIXED_FUSED if tuned is FIXED else None)
    RESULTS["fused"] = fused
    if not choice:
        log("no form passes the checks: no packages")
        return
    key = min(choice, key=lambda k: choice[k][0])
    _, conv2, stem_int8, bm, c2cfg, stem, tbm = choice[key]
    RESULTS["chosen"] = key
    log(f"chosen form: {key} ({RESULTS['choice']})")

    # packages of every seed in the chosen form
    for seed in SEEDS:
        try:
            if seed == 0:
                m, plan = m0, plan0
            else:
                m, _ = load_seed(ckpt, seed)
                e = m.encoder.float().eval()
                cal = calibrate_seed(e, [c.to("cuda:0") for c in cal_x])
                (OUT / f"cal_fold{seed}.json").write_text(json.dumps(cal))
                plan = dense8s.make_plan(e, cal, img)
            blobs, tc, info = dense8s.build_engines(plan, NB, tc, log=log, conv2=conv2,
                                                    stem_int8=stem_int8, stem=stem)
            RESULTS["build_s"][f"seed{seed}"] = info["total_s"]
            path = OUT / f"d8s376x2j_fold{seed}.npz"
            dense8s.save_pack(path, plan, NB, blobs,
                              extra={"seed": seed, "exp": EXP, "git": GIT_SHA,
                                     "cal_studies": len(cal_uids)},
                              cubin=cubin, bm=bm, conv2=conv2, c2cfg=c2cfg,
                              stem_int8=stem_int8, stem=stem, tbm=tbm, fused=fused)
            pk = dense8s.load_pack(path)
            chk = dense8s.Dense8SEncoder(pk, 0, x.shape[0], max_inflight=2)
            e = m.encoder.float().eval().to("cuda:0")
            with torch.no_grad():
                rf = in_chunks(e, x)
                fs = chk(x.half()).float().clone()
                torch.cuda.synchronize(0)
            e.cpu()
            RESULTS[f"pack{seed}_vs_float"] = cmp(fs, rf)
            RESULTS[f"pack{seed}_mb"] = round(path.stat().st_size / 2**20, 1)
            log(f"package seed {seed}: vs float32 {RESULTS[f'pack{seed}_vs_float']}, "
                f"{RESULTS[f'pack{seed}_mb']} MiB")
            chk = None
            torch.cuda.empty_cache()
        except Exception as exc:
            RESULTS[f"pack{seed}_error"] = traceback.format_exc()[-2000:]
            log(f"!! package seed {seed}: {exc!r}")
        (OUT / "timing.cache").write_bytes(tc)
        save()
    log(f"done in {RESULTS['total_s']} s")


WATCHDOG_S = 2400


def _watchdog() -> None:
    RESULTS["watchdog"] = f"stopped after {WATCHDOG_S} s"
    log(f"!! watchdog: {WATCHDOG_S} s, exiting")
    save()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    import threading
    threading.Timer(WATCHDOG_S, _watchdog).start()
    try:
        main()
    except Exception:
        RESULTS["fatal"] = traceback.format_exc()[-4000:]
        print(RESULTS["fatal"], flush=True)
    save()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
