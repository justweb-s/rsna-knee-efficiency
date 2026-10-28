"""Kaggle kernel: the efficiency submission. GPU (2 x T4), **no internet**.

Reads the checkpoints of the two DenseNet169 seeds and their INT8 packages, decodes the test DICOMs
and writes into /kaggle/working:

    submission.csv   one row per study of test.csv, twelve columns, in the order of
                     `constants.TARGETS`
    runtime.json     wall time, decoding and GPU separately, and how many rows came out neutral.

The constraint that shapes this file is the absence of internet: the plugins for compressed transfer
syntaxes cannot be installed from PyPI and come from the `knee-wheels` dataset, and the ImageNet
weights cannot be downloaded, so the encoder is built with `pretrained=False` and the real weights
come from the checkpoint.

Everything that can start before `import torch` does: reading our own datasets, warming the library
pages, the decoding workers (`knee.data.early`), the C library. The source of the `knee` package and
the precompiled binaries are embedded in this file by `tools/build_kernels.py`.
"""
from __future__ import annotations

import os
os.environ["KNEE_HEAD_ONLY"] = "on"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import base64 as _cb64, zlib as _czlib
from pathlib import Path as _CPath
_croot = _CPath("/kaggle/working/uint8-runtime"); _croot.mkdir(exist_ok=True)
for _name, _data in {}.items():
    (_croot / _name).write_bytes(_czlib.decompress(_cb64.b64decode(_data)))
os.environ["KNEE_UINT8_CUBIN_DIR"] = str(_croot)
os.environ["KNEE_LEAN"] = "on"
os.environ["KNEE_LEAN_WORKERS"] = "8"
os.environ["KNEE_LEAN_DIRECT"] = "1"
os.environ["KNEE_LEAN_RING_MB"] = "10240"   # a 10 GB ring per worker pool (16 GB made the first lap of the ring slower)
os.environ["KNEE_REGISTER_VIA"] = "off"   # rings not registered with CUDA, copies through pin_memory
os.environ["KNEE_FUSED_UINT8"] = "on"
os.environ["KNEE_READ_SLICES"] = "0:23"   # slices read (the 24th slice is not used by any window)
os.environ["KNEE_TTA_U8_CHECK"] = "0"   # self-check of the uint8 TTA (benchmark only)
import time as _tm; _T0 = _tm.time()
def _tp(m): print(f'[t {_tm.time() - _T0:6.2f}] {m}', flush=True)
_tp('start')
PREFETCH_GROUPS = [('datasets', 'knee-wheels', '', ('-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl', '-py3-none-any.whl', 'WHEELS')), ('datasets', 'knee-runs-376-384-full-data-v9j-287-img512', '', ('.pt', '.json')), ('notebooks', 'knee-repack-dense8s-t8x32-376x2j', 'dense8s-runtime', ('.npz',)), ('notebooks', 'knee-trt-runtime-cp313', 'trt_runtime', None)]
# --- start-up: the files of our datasets read in parallel from t=0, one group at a time ------
# The first access to our datasets has a latency that changes from run to run (from ~0.8 s to ~20 s to
# copy the source tree, checkpoints at 15-35 MB/s), and sequential reads pay all of it. Here 16 threads
# read in 8 MB chunks; a group starts when the previous one is done. Whoever reads afterwards finds the
# page cache. No competition data here: the workers read those.
def _prefetch_inputs(groups, threads=16, chunk=8 << 20):
    import os as _os, queue as _qu, threading as _th, time as _t
    jobs, stats, t0 = _qu.Queue(), {"files": 0, "bytes": 0, "errors": 0, "groups": []}, _t.time()
    def producer():
        for kind, slug, sub, suffixes in groups:
            root = _os.path.join("/kaggle/input", kind, "marcoledda29", slug, sub)
            if not _os.path.isdir(root):   # old mount layout: /kaggle/input/<slug>
                root = _os.path.join("/kaggle/input", slug, sub)
            for d, dirs, files in _os.walk(root):
                dirs.sort()
                for f in sorted(files):
                    if suffixes and not f.endswith(suffixes):
                        continue
                    p = _os.path.join(d, f)
                    try:
                        n = _os.path.getsize(p)
                    except OSError:
                        continue
                    stats["files"] += 1
                    for off in range(0, max(n, 1), chunk):
                        jobs.put((p, off, min(chunk, n - off)))
            jobs.join()
            stats["groups"].append((slug, round(_t.time() - t0, 1)))
        for _ in range(threads):
            jobs.put(None)
    def reader():
        buf = bytearray(1 << 22)
        while True:
            job = jobs.get()
            if job is None:
                jobs.task_done()
                return
            p, off, n = job
            try:
                fd = _os.open(p, _os.O_RDONLY)
                try:
                    while n > 0:
                        k = _os.preadv(fd, [memoryview(buf)[:min(n, len(buf))]], off)
                        if k <= 0:
                            break
                        off, n = off + k, n - k
                        stats["bytes"] += k
                finally:
                    _os.close(fd)
            except OSError:
                stats["errors"] += 1
            finally:
                jobs.task_done()
    ths = [_th.Thread(target=reader, daemon=True, name=f"prefetch-{i}") for i in range(threads)]
    for th in ths:
        th.start()
    def report():
        producer()
        for th in ths:
            th.join()
        _tp(f"read-ahead: {stats['files']} files, {stats['bytes'] / 2**20:.0f} MB, "
            f"{stats['errors']} errors, in {_t.time() - t0:.1f} s; groups {stats['groups']}")
    _th.Thread(target=report, daemon=True, name="prefetch-producer").start()
# two streams, datasets and kernel outputs together (cold datasets read at ~10 MB/s)
_prefetch_inputs([g for g in PREFETCH_GROUPS if g[0] == "datasets"], threads=8)
_prefetch_inputs([g for g in PREFETCH_GROUPS if g[0] == "notebooks"], threads=8)
# --- start-up: the library pages into the page cache from a separate process ------------------
# Cold, `import knee.infer` costs 10.9 s, warm 3.2: the rest is disk. Here torch cannot be imported
# before the workers are forked; a separate process can: it reads the same pages and exits. No CUDA
# initialisation: the torch import is lazy on the device.
def _warm_pages():
    import subprocess as _sp, sys as _sy, threading as _th
    code = ("import os\n"
            "try:\n    os.nice(5)\nexcept (AttributeError, OSError):\n    pass\n"
            "import numpy, pandas\n"
            "import torch, torch.nn.functional\n"
            "import pydicom\n")
    try:
        p = _sp.Popen([_sy.executable, "-c", code], stdin=_sp.DEVNULL, stdout=_sp.DEVNULL,
                      stderr=_sp.DEVNULL, close_fds=True)
    except Exception as exc:                                        # noqa: BLE001
        _tp(f"page warm-up not started: {exc!r}")
        return None
    def wait():
        _tp(f"page warm-up done (rc {p.wait()})")
    _th.Thread(target=wait, daemon=True, name="warm-pages").start()
    return p
_WARM = _warm_pages()
# --- start-up sampler: the stacks of every Python thread, every few ms, for 60 s ------------
def _start_startup_sampler(every=0.01, until=120.0, out="/kaggle/working/startup_stacks.json"):
    import collections as _co, json as _js, sys as _sy, threading as _th, time as _t
    main_id = _th.main_thread().ident
    def frames(f):
        out = []
        while f is not None and len(out) < 4:
            c = f.f_code
            if not c.co_filename.startswith("<frozen"):
                out.append(f"{'/'.join(c.co_filename.rsplit('/', 2)[-2:])}:{f.f_lineno}:{c.co_name}")
            f = f.f_back
        return " < ".join(out)
    def run():
        me, names, buckets = _th.get_ident(), {}, _co.defaultdict(_co.Counter)
        while True:
            now = _t.time() - _T0
            if now > until:
                break
            for th in _th.enumerate():
                names[th.ident] = th.name
            for ident, f in _sy._current_frames().items():
                if ident != me:
                    name = "main" if ident == main_id else names.get(ident, str(ident))
                    buckets[(name, int(now))][frames(f)] += 1
            _t.sleep(every)
        res = {}
        for (name, sec), c in sorted(buckets.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            res.setdefault(name, {})[str(sec)] = c.most_common(4)
        try:
            with open(out, "w") as fh:
                fh.write(_js.dumps({"every_s": every, "threads": res}))
        except OSError:
            pass
    _th.Thread(target=run, daemon=True, name="startup-sampler").start()
# --- GPU sampler: clock, power, temperature and utilisation every second -------------------
def _start_gpu_sampler(path="/kaggle/working/gpu_smi.csv"):
    import subprocess as _sp
    try:
        return _sp.Popen(["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,"
                          "temperature.gpu,utilization.gpu,utilization.memory", "--format=csv,noheader",
                          "-lms", "1000", "-f", path], stdin=_sp.DEVNULL, stdout=_sp.DEVNULL,
                         stderr=_sp.DEVNULL, close_fds=True)
    except Exception as exc:                                        # noqa: BLE001
        _tp(f"GPU sampler not started: {exc!r}")
        return None
# --- sampler: machine CPU and network bytes received every 2 s ---------
def _start_sampler(path="/kaggle/working/sampler.csv", every=2.0):
    import threading as _th, time as _t
    def cpu():
        v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(v), v[3] + v[4], v[4]
    def rx():
        n = 0
        for line in open("/proc/net/dev").readlines()[2:]:
            name, rest = line.split(":", 1)
            if name.strip() != "lo":
                n += int(rest.split()[0])
        return n
    def run():
        t0 = _t.time(); c0, r0 = cpu(), rx()
        with open(path, "w") as fh:
            fh.write("t,cpu_busy,iowait,rx_mb_s\n")
            while True:
                _t.sleep(every)
                c1, r1 = cpu(), rx()
                tot = c1[0] - c0[0] or 1
                fh.write(f"{_t.time() - t0:.1f},{1 - (c1[1] - c0[1]) / tot:.3f},"
                         f"{(c1[2] - c0[2]) / tot:.3f},{(r1 - r0) / every / 2**20:.1f}\n")
                fh.flush(); c0, r0 = c1, r1
    _th.Thread(target=run, daemon=True, name="sampler").start()
os.environ.setdefault("KNEE_PCT_STRIDE", "7")   # percentiles on a subsample (one pixel in 7): before importing knee
import sys
from pathlib import Path


# --- bootstrap: putting `knee` on sys.path before it can be imported ------------------
# It cannot use knee.kaggle, which is exactly what it is looking for. Kaggle does not guarantee the
# depth of the mount: datasets may be under /kaggle/input/datasets/<owner>/<slug>/, two levels deeper
# than usual.
def find_src() -> Path:
    base = Path("/kaggle/input")
    # first the source dataset from the kernel metadata, without walking /kaggle/input
    for d in (base / "datasets" / "marcoledda29" / "knee-src-label-ablation-196-197", base / "knee-src-label-ablation-196-197"):
        for c in (d, d / "src"):
            if (c / "knee" / "constants.py").is_file():
                return c
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
    print("the `knee` package is not mounted anywhere.", flush=True)
    print("content of /kaggle/input:", flush=True)
    for d in seen[:60]:
        print(f"  {d}", flush=True)
    raise FileNotFoundError("knee-src dataset not mounted: see the listing above")


# The `knee` sources embedded in this script by the builder (`tools/build_kernels.py`): BASE is the
# package, OVERLAY can override single files (empty in the release).
# --- overlay (written by the builder: do not edit by hand) ---
OVERLAY_SHA = "659ef18d-dirty+gpu-resize+tta-u8"
OVERLAY = {}
# --- end of overlay ---
# --- embedded package (written by the builder) ---
BASE = {}


# --- the whole source from this script (BASE = the package, OVERLAY = newer files) ---------
# With cold datasets, finding and copying the mounted source cost 8 s before anything else.
SRC = Path("/tmp/src_overlay")
import base64
import zlib
for _rel, _b64 in {**BASE, **OVERLAY}.items():
    (SRC / _rel).parent.mkdir(parents=True, exist_ok=True)
    (SRC / _rel).write_bytes(zlib.decompress(base64.b64decode(_b64)))
(SRC / "knee" / "GIT_SHA").write_text(OVERLAY_SHA)
sys.path.insert(0, str(SRC))
print(f"knee from {SRC}" + (f" (overlay {OVERLAY_SHA})" if OVERLAY else ""), flush=True)
# --- the precompiled C library of the workers (its file name carries the hash of source and flags). It
# goes where `cfast._build` looks for it only if the source written here has that hash and if it loads
# and answers; otherwise it is compiled as usual.
CFAST_LIB = {"tag": 'e129ddc8593e', "flags": ['-O2', '-ffp-contract=off', '-fno-fast-math', '-shared'], "so": ""}
def _cfast_prebuilt():
    import ctypes as _ct, hashlib as _hl, tempfile as _tf
    d = SRC / "knee" / "data"
    srcs = [d / "fastdecode.c"] + ([d / "leanio.c"] if (d / "leanio.c").is_file() else [])
    tag = _hl.sha1(b"".join(p.read_bytes() for p in srcs) + " ".join(CFAST_LIB["flags"]).encode()).hexdigest()[:12]
    if tag != CFAST_LIB["tag"]:
        _tp(f"precompiled C library for another source ({tag} instead of {CFAST_LIB['tag']}): compiling")
        return
    lib = Path(_tf.gettempdir()) / f"knee_fastdecode_{tag}.so"
    tmp = lib.with_name(lib.name + ".pre")
    try:
        tmp.write_bytes(zlib.decompress(base64.b64decode(CFAST_LIB["so"])))
        h = _ct.CDLL(str(tmp))
        h.knee_abi.restype = h.lio_abi.restype = _ct.c_int64
        if h.knee_abi() != 1 or h.lio_abi() != 1:
            raise RuntimeError("ABI mismatch")
        os.replace(tmp, lib)
    except Exception as exc:                                         # noqa: BLE001
        _tp(f"precompiled C library rejected ({exc!r}): compiling")
        try:
            tmp.unlink()
        except OSError:
            pass
_cfast_prebuilt()
# --- the C library of the workers (gcc, 3.5 s) is compiled from here, in a thread; `cfast.load` runs once
# per process under a lock, and `early.start_for_kernel` waits for this outcome instead of starting over.
def _cfast_early():
    import threading as _th
    def run():
        try:
            from knee.data import cfast as _cf
            _cf.load()
        except Exception as exc:                                     # noqa: BLE001
            _tp(f"early C compilation failed: {exc!r}")
    _th.Thread(target=run, daemon=True, name="cfast-early").start()
_cfast_early()
# --- the walk of /kaggle/input in a thread, the folders of every level in parallel
# (`knee.kaggle.prewalk`): the searches after the fork find it done or wait for it, instead of redoing
# it sequentially on a cold mount (~5 s).
def _prewalk_early():
    import threading as _th
    def run():
        t = _tm.time()
        try:
            from knee import kaggle as _kg
            n = len(_kg.prewalk())
        except Exception as exc:                                     # noqa: BLE001
            _tp(f"early walk failed: {exc!r}")
            return
        _tp(f"walk of /kaggle/input: {n} folders in {_tm.time() - t:.1f} s")
    _th.Thread(target=run, daemon=True, name="prewalk").start()
_prewalk_early()

# The encoder engines (`knee.trt`, `knee.dense8s`): `no-trt` = off; a prefix (e.g. `d8s376x2j`) = the
# packages `<prefix>_fold<k>.npz` (dense8s INT8 packages, with the TensorRT stem) found among the inputs,
# with TensorRT from the `knee-trt-runtime-cp313` kernel output (unpacked, without the builder).
TRT = os.environ.get("KNEE_TRT", "d8s376x2j")


def prefetch_trt_runtime():
    """The TensorRT libraries (~700 MiB, `libnvinfer.so` 639) read in a thread while torch and the rest are
    imported: cold from the mount they would cost seconds on the first import.
    """
    import threading

    from knee.kaggle import find_dir_with
    rt = find_dir_with("TRT_RUNTIME")
    if rt is None:
        return None
    libs = sorted(rt.glob("tensorrt_libs/*.so*")) + sorted(rt.glob("tensorrt_bindings/*.so"))

    def run() -> None:
        for lib in libs:
            try:
                with open(lib, "rb", buffering=0) as fh:
                    while fh.read(16 << 20):
                        pass
            except OSError:
                pass

    th = threading.Thread(target=run, name="trt-prefetch", daemon=True)
    th.start()
    return rt


def pack_needs_trt(prefix: str) -> bool:
    """Whether the prefix names TensorRT engines (`.plan`) or packages with engines; a dense8s package that
    is entirely ours (`meta["engines"]` empty) does not need TensorRT: no runtime to read from the mount
    or import.
    """
    import json as _json

    import numpy as _np

    from knee.kaggle import find_dir_with
    d = find_dir_with(f"{prefix}_fold0.npz")
    if d is None:
        return True
    try:
        with _np.load(d / f"{prefix}_fold0.npz") as z:
            meta = _json.loads(bytes(z["meta"]).decode())
    except Exception:                                           # noqa: BLE001
        return True
    return bool(meta.get("engines"))



# Early decoding (`knee.data.early`): the raw-crop workers start here, before `import torch` (which
# with `knee.infer` costs ~12 s cold). `KNEE_EARLY=off` = the plain DataLoader.
if os.environ.get("KNEE_EARLY", "on") == "on" and os.environ.get("KNEE_GPU_RESIZE", "gpu-resize") != "cpu-resize":
    try:
        from knee.data import early as _early
        from knee.data.cache import CacheSpec as _CS
        _early.start_for_kernel(
                                workers=int(os.environ.get("KNEE_WORKERS_D8", "6")),
                                pack=TRT if TRT != 'no-trt' else None,
                                # the competition root and the checkpoint spec written here (`take` compares them again with those of
                                # run_submission), decoders after the fork: none of our datasets is read before the workers start
                                root=Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                                spec=_CS(img=512, slices=24, crop_mm=110.0, band=(0.15, 0.85), crop_mm2=150.0),
                                decoders_after=True)
    except Exception as _exc:                                          # noqa: BLE001
        print(f"!! early pool not started: {_exc!r}: staying on the DataLoader", flush=True)
_tp('early pool started')
# --- is TensorRT needed, and where is the runtime? (walk and package metadata) in a thread after the
# fork; `main` waits for it. Done sequentially before the workers it cost ~5 s cold.
NEEDS_TRT, TRT_RUNTIME = True, None
def _trt_lookup():
    global NEEDS_TRT, TRT_RUNTIME
    try:
        NEEDS_TRT = TRT != 'no-trt' and pack_needs_trt(TRT)
        TRT_RUNTIME = prefetch_trt_runtime() if NEEDS_TRT else None
    except Exception as exc:                                         # noqa: BLE001
        _tp(f"TensorRT lookup failed: {exc!r}")
import threading as _th_trt
_TRT_LOOKUP = _th_trt.Thread(target=_trt_lookup, daemon=True, name="trt-lookup")
_TRT_LOOKUP.start()
from knee.data.dicom import REQUIRED_DECODERS, ensure_decoders       # noqa: E402
from knee.infer import run_submission                                # noqa: E402
_tp('import knee.infer')
# --- `torch._check_with(err, True, msg)` returns immediately. The original first imports
# torch.fx.experimental.symbolic_shapes (sympy, mpmath, ~490 modules) for `expect_true`, which with a
# Python bool returns the bool itself: the first MultiheadAttention paid it while attaching the engines
# (4.8 s per GPU). Every condition that is not `True` goes to the original.
def _fast_check_with():
    import torch as _torch
    _orig = _torch._check_with
    def _check_with(error_type, cond, message):
        if cond is True:
            return
        return _orig(error_type, cond, message)
    _check_with.__wrapped__ = _orig
    _torch._check_with = _check_with
_fast_check_with()
from knee.tta import parse_tta                                       # noqa: E402
import knee.tta as _tta_mod                                          # noqa: E402
_tta_mod.VARIANTS["s1.10"] = _tta_mod.Variant("s1.10", scale=1.10)
from knee.kaggle import describe_inputs, find_dir_with, find_test_root  # noqa: E402

OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("runs/submit")
WORKERS = int(os.environ.get("KNEE_WORKERS", "4"))    # 4 vCPUs on the Kaggle GPU machine
# With the INT8 packages the GPU is no longer the limit and decoding also waits for the mount's I/O:
# more workers in flight (two seeds + TTA 0.1318 s/study with 6 against 0.1433 with 4).
WORKERS_D8 = int(os.environ.get("KNEE_WORKERS_D8", "6"))
BATCH = int(os.environ.get("KNEE_BATCH", "8"))        # Tensor Core saturation on the T4
BUDGET = float(os.environ.get("KNEE_BUDGET_S", "25200"))   # 7 h: below the kernel's limit
# Which folds (seeds) of the run enter the mean: `all` = every `fold*.pt` found; a subset (e.g. `0`)
# uses only those.
FOLDS = os.environ.get("KNEE_FOLDS", "all")
# Flip-free TTA (`knee.tta`): `none` = the model as is; a combination (e.g. `geo3`) or comma-separated
# variants = the mean of the probabilities over those passes. The final submission uses `s1.10`, a
# single zoom-1.10 view instead of the original image.
TTA = os.environ.get("KNEE_TTA", "s1.10")
# The inference window stride (`knee.infer.apply_stride`): `model` = the checkpoint's; an integer (e.g.
# `4`: 48 windows instead of 88 with 24 slices) = fewer encoder passes.
STRIDE = os.environ.get("KNEE_STRIDE", "model")
# The encoder at inference: `auto` = AOTI packages if present, otherwise `torch.compile`, and eager if
# that fails; `eager` = no compilation. With the INT8 packages the encoder is the package anyway.
ENCODER = os.environ.get("KNEE_ENCODER", "eager")
# AdaBN (`knee.adabn`): before predicting, the BatchNorm layers recompute their statistics on the test
# studies. `off` = off; `all` = the whole test set; an integer = the first N studies.
ADABN = os.environ.get("KNEE_ADABN", "off")
# Fewer windows per member (`knee.window_budget`): `all-windows` = all; a design (e.g. `parx`) = every
# seed x TTA view pair sees only the windows of its design. Only with the dense8s packages. The final
# submission uses `parx`: each seed sees half of the windows, complementary.
WINDOW_BUDGET = os.environ.get("KNEE_WINDOW_BUDGET", "parx")
# Studies per GPU per batch with a design: one member = at most 352 images = one graph of the package
# (`mod3`: 29 or 30 windows per member, 11 studies = 319-330 images).
BUDGET_BATCH = {"par": 8, "mod3": 6, "mod4": 16, "parx": 8}
# Per-site test-time correction (`knee.site_correction`): `no-site` = off; `mute` = after the
# predictions, the per-site offset learned from the teacher is removed from the silent columns, with the
# site predicted from the series geometry.
SITE = os.environ.get("KNEE_SITE", "no-site")

# Raw crops (`knee.data.raw_crop`): the workers ship the 150 mm box as uint16 and resize, percentiles and
# uint8 happen on the GPU (`run_submission(gpu_resize=True)`). `cpu-resize` = the CPU path.
GPU_RESIZE = os.environ.get("KNEE_GPU_RESIZE", "gpu-resize")


def main() -> None:
    _TRT_LOOKUP.join()                  # the TensorRT lookup started after the fork
    wheels = find_dir_with("WHEELS")
    if wheels is None:
        print(describe_inputs(max_depth=3), flush=True)
        raise FileNotFoundError(
            "knee-wheels dataset not mounted. Without internet the decoding plugins cannot be "
            "installed from PyPI: build the wheels dataset and add it to the "
            "dataset_sources of this kernel.")
    have = ensure_decoders(wheels_dir=wheels, allow_network=False)
    print(f"decoders from {wheels}: {have}", flush=True)
    if not all(have[m] for m in REQUIRED_DECODERS):
        raise SystemExit(f"missing decoders: {[m for m in REQUIRED_DECODERS if not have[m]]}. "
                         "Without them every compressed series becomes a neutral row.")

    ckpt = find_dir_with("fold0.pt", extra=[Path("runs"), Path(".")])
    if ckpt is None:
        print(describe_inputs(max_depth=3), flush=True)
        raise FileNotFoundError(
            "no fold0.pt among the inputs. Is the weights dataset among the dataset_sources "
            "of this kernel?")

    root = find_test_root()
    # The torch.compile artifacts of the run (an optional dataset): without them compilation is cold, ~75 s
    # more on the T4 and nothing else different. Unused with the INT8 packages.
    cache_dir = find_dir_with("COMPILE_CACHE")
    print(f"competition mount: {root} | checkpoints from {ckpt} | compilation cache: "
          f"{cache_dir}", flush=True)
    folds = None if FOLDS == 'all' else [int(f) for f in FOLDS.split(",")]
    tta = parse_tta(TTA)
    if tta:
        print(f"flip-free TTA: {[v.name for v in tta]} ({len(tta)} passes per study)", flush=True)
    stride = None if STRIDE == 'model' else int(STRIDE)
    compile = ENCODER != 'eager'
    print(f"encoder: {'compiled if possible' if compile else 'eager, compilation off'}",
          flush=True)
    adabn = None if ADABN == 'off' else (-1 if ADABN == 'all' else int(ADABN))
    if adabn:
        print(f"adabn: BatchNorm recomputed on {'the whole test set' if adabn < 0 else f'{adabn} studies'}",
              flush=True)
    trt_dir, pack = None, None
    if TRT != 'no-trt':
        # The engines with the name `run_submission` looks for (`trt8q_fold<k>.plan`), outside /kaggle/working
        # (they are not outputs). Without runtime or engines it stays eager: the submission still comes out,
        # slower. A prefix can also name `knee.dense8`/`knee.dense8s` packages (`<prefix>_fold<k>.npz`), which
        # `run_submission` looks for as `dense8_fold<k>.npz`; they are linked (hundreds of MiB), not copied.
        import shutil
        import tempfile
        eng = find_dir_with(f"{TRT}_fold0.plan")
        pack = find_dir_with(f"{TRT}_fold0.npz") if eng is None else None
        if (NEEDS_TRT and TRT_RUNTIME is None) or (eng is None and pack is None) or compile:
            print(f"!! TensorRT requested ({TRT}) but runtime {TRT_RUNTIME}, engines {eng or pack}, "
                  f"compile {compile}: staying eager", flush=True)
        else:
            if TRT_RUNTIME is not None:
                sys.path.insert(0, str(TRT_RUNTIME))
            trt_dir = Path(tempfile.mkdtemp(prefix="knee-trt-"))
            # All the folds `run_submission` will load (with `all`: every `fold*.pt` of the run): with only fold 0
            # linked the engines of the others would be missing and it would stay eager.
            ids = folds if folds is not None else sorted(
                int(q.stem.removeprefix("fold")) for q in Path(ckpt).glob("fold*.pt"))
            for f in ids:
                if eng is not None:
                    shutil.copyfile(eng / f"{TRT}_fold{f}.plan", trt_dir / f"trt8q_fold{f}.plan")
                else:
                    os.symlink(pack / f"{TRT}_fold{f}.npz", trt_dir / f"dense8_fold{f}.npz")
            print(f"engines {TRT} from {eng or pack}, TensorRT runtime {TRT_RUNTIME} "
                  f"({'needed' if NEEDS_TRT else 'not needed: dense8s packages entirely ours'})",
                  flush=True)
    budget = None if WINDOW_BUDGET == 'all-windows' else WINDOW_BUDGET
    if budget is not None:
        if pack is None:
            raise SystemExit(f"window budget {budget}: only with the dense8s packages (TRT prefix)")
        print(f"windows per member: design {budget}, {BUDGET_BATCH.get(budget, BATCH)} studies per "
              f"GPU per batch", flush=True)
    run_submission(root, ckpt, OUT,
                   batch_size=BATCH if budget is None else BUDGET_BATCH.get(budget, BATCH),
                   workers=WORKERS if pack is None else max(WORKERS, WORKERS_D8),
                   time_budget_s=BUDGET,
                   compile=compile, compile_cache_dir=cache_dir, folds=folds, tta=tta,
                   stride=stride, adabn=adabn, trt_dir=trt_dir,
                   trt_inflight=2 if trt_dir is not None else None,
                   trt_graphs=trt_dir is not None and pack is not None,
                   early_workers=trt_dir is not None and pack is not None,
                   # windows of empty slots skipped (bit-exact)
                   skip_empty=trt_dir is not None and pack is not None and budget is None,
                   budget=budget,
                   gpu_resize=GPU_RESIZE != 'cpu-resize')
    if SITE != 'no-site':
        # No fallback to the raw version: if the correction fails the kernel fails, so an "equal" leaderboard
        # score really means a correction without effect.
        from knee.site_correction import correct_submission
        print(f"per-site correction ({SITE}): {correct_submission(OUT / 'submission.csv', root, OUT)}",
              flush=True)


if __name__ == "__main__":
    main()
    # the C library compiled in this run, for a kernel that wants to carry it prebuilt
    try:
        import glob as _gl
        import shutil as _sh
        for _lib in _gl.glob("/tmp/knee_fastdecode_*.so"):
            (OUT / "cfast-lib").mkdir(exist_ok=True)
            _sh.copy2(_lib, OUT / "cfast-lib")
    except Exception:                                                # noqa: BLE001
        pass
    # Exit without tearing down the interpreter. With persistent DataLoader workers, torch.compile and
    # inductor caches already unloaded, the process once died of a segfault at shutdown, with
    # submission.csv and runtime.json already written. Kaggle runs `python3 -u script.py` and reads its exit
    # code: a segfault there is a failed submission with the work done.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
