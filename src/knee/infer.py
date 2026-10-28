"""From the test DICOMs to the submission, in a single pass.

Why the training cache is not reused. The test set is not preprocessed once and for all: the
submission kernel runs on hidden data, once, under a time limit. Writing study volumes to disk to
read them back thirty seconds later is an I/O round trip nobody needs. Here a study is decoded, goes
through the models and is thrown away.

**Decoding is the bottleneck, not the GPU**, and it is pure CPU: it lives in the DataLoader workers
(or in the early pool of `knee.data.early`), so while the GPU consumes a batch the workers prepare
the next one. `runtime.json` reports the two items separately, because a single total does not say
which of the two is worth optimising.

**An error does not sink the submission.** A study that fails to decode cannot become a missing
row: a missing row is not a low score, it is a rejected submission. It gets 0.5 on all twelve
columns, and the count of how many times it happened goes into `runtime.json`: a silent failure here
would look like a mediocre model.
"""
from __future__ import annotations

import contextlib
import copy
import gc
import json
import lzma
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import ID, TARGETS
from .artifacts import (aoti_dir_name, aoti_study_dir_name, compile_cache_name,  # noqa: F401
                        stride_suffix)                                          # (re-exported)
from .data.cache import CacheSpec, build_study, count_files, pick_slots
from .data import cache as cache_mod
from .data.dicom import DecodeErrors, read_slice
from .data.torch_uint8 import to_uint8_torch
from .data import raw_crop
from .tta import Variant, variant_windows
from .window_budget import member_windows
from .models.baseline import (GROUP, KneeBaseline, encode_chunk_for, make_encoder, make_windows,
                              window_slot_index, windows_per_slot)
from .fastdense import attach_eval
from .models.densenet import fold_bn as fold_densenet_bn, fold_bn_status
from .models.video3d import channels_last_ok
from .utils.device import check_device

NEUTRAL = 0.5

# Start-up trace (diagnostics only, off if the variable is unset): every worker writes per study index,
# pid, start instant (epoch) and duration into `$KNEE_TRACE_DIR/w<pid>.csv`, and `predict` puts the
# batch instants and the machine CPU into `stats["trace"]`. Used to break down the seconds the pipeline
# loses while filling up.
TRACE_ENV = "KNEE_TRACE_DIR"


def _trace_item(i: int, t0: float, dt: float) -> None:
    d = os.environ.get(TRACE_ENV)
    if not d:
        return
    try:
        with open(os.path.join(d, f"w{os.getpid()}.csv"), "a", encoding="ascii") as fh:
            fh.write(f"{i},{os.getpid()},{t0:.4f},{dt:.4f}\n")
    except OSError:
        pass


class _CpuSampler:
    """The busy CPU fraction of the whole machine every `period` s, from /proc/stat."""

    def __init__(self, period: float = 0.25):
        import threading
        self.period, self.rows, self._stop = period, [], threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read() -> tuple[int, int] | None:
        try:
            with open("/proc/stat", encoding="ascii") as fh:
                v = [int(x) for x in fh.readline().split()[1:]]
        except (OSError, ValueError):
            return None
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0)

    def _run(self) -> None:
        prev = self._read()
        while prev is not None and not self._stop.wait(self.period):
            cur = self._read()
            if cur is None:
                return
            tot, idle = cur[0] - prev[0], cur[1] - prev[1]
            self.rows.append((round(time.time(), 3), round(1 - idle / tot, 3) if tot else None))
            prev = cur

    def start(self) -> "_CpuSampler":
        self._th.start()
        return self

    def stop(self) -> list:
        self._stop.set()
        self._th.join(timeout=2)
        return self.rows


def _thread_cpu() -> dict[int, tuple[str, float]]:
    """{tid: (name, CPU in s)} of the threads of this process, from /proc/self/task (Linux).

    The benchmark's `main_cpu_s` (RUSAGE_SELF) sums all threads of the main process: the Python thread,
    the DataLoader `pin_memory` thread, torch's OpenMP threads and those of the CUDA driver. Here they
    are separated, to know who takes CPU from the workers.
    """
    out: dict[int, tuple[str, float]] = {}
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    try:
        tids = os.listdir("/proc/self/task")
    except OSError:
        return out
    for t in tids:
        try:
            raw = Path(f"/proc/self/task/{t}/stat").read_text(encoding="ascii", errors="replace")
        except OSError:
            continue
        name = raw[raw.find("(") + 1:raw.rfind(")")]
        f = raw[raw.rfind(")") + 2:].split()
        out[int(t)] = (name, (int(f[11]) + int(f[12])) / tick)     # utime + stime
    return out


def _thread_cpu_delta(a: dict, b: dict, named: dict[int, str]) -> dict:
    """The CPU per thread group between two `_thread_cpu()` calls: `named` = {tid: group} for the known
    threads (main, pin_memory, sampler), the others by thread name.
    """
    groups: dict[str, float] = {}
    for tid, (name, cpu) in b.items():
        key = named.get(tid, f"other:{name}")
        groups[key] = groups.get(key, 0.0) + cpu - a.get(tid, (name, 0.0))[1]
    return {k: round(v, 2) for k, v in sorted(groups.items(), key=lambda kv: -kv[1])}


class DicomStudies(Dataset):
    """One study per item, decoded on the fly inside the worker.

    Also returns how much it cost: decoding time is half the kernel's budget and cannot be measured from
    outside, because from outside only the wall time of the loop is visible, which is already the sum of
    the two items.
    """

    def __init__(self, root: Path, uids: list[str], slots: dict[str, dict[int, str]],
                 spec: CacheSpec, series_dir: str = "test_series", to_uint8: bool = True,
                 raw: bool = False):
        self.root, self.uids, self.slots, self.spec = Path(root), list(uids), slots, spec
        self.series_dir = series_dir
        # False: the worker delivers the float32 stacks and `predict` computes percentiles + uint8 on the
        # device (`to_uint8_torch`, the same uint8 values bit for bit).
        self.to_uint8 = to_uint8
        # True: the worker delivers the raw crops (`raw_crop.build_study_raw`, one payload per study) and
        # `_forward_split` does resize, percentiles and uint8 on the device.
        self.raw = raw

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        study = self.uids[i]
        errors = DecodeErrors()
        t0 = time.perf_counter()
        if self.raw:
            try:
                payload, mask, _ = raw_crop.build_study_raw(
                    str(self.root / self.series_dir / study), self.slots.get(study, {}),
                    self.spec, errors)
            except Exception:
                payload = raw_crop.empty_payload()
                mask = np.zeros(self.spec.shape[0], dtype=bool)
            dt = time.perf_counter() - t0
            _trace_item(i, time.time() - dt, dt)
            return (payload, torch.from_numpy(mask), i, dt, len(errors))
        try:
            vol, mask, _ = build_study(str(self.root / self.series_dir / study),
                                       self.slots.get(study, {}), self.spec, errors,
                                       to_uint8=self.to_uint8)
        except Exception:
            # Anything: missing folder, unreadable header, empty series. The row comes out anyway, with the mask at
            # zero, and downstream it becomes a declared 0.5.
            vol = np.zeros(self.spec.shape, dtype=np.uint8 if self.to_uint8 else np.float32)
            mask = np.zeros(self.spec.shape[0], dtype=bool)
        dt = time.perf_counter() - t0
        _trace_item(i, time.time() - dt, dt)
        return (torch.from_numpy(vol), torch.from_numpy(mask), i, dt, len(errors))


def _spec_of(cfg: dict) -> CacheSpec:
    return CacheSpec(**dict(cfg["cache"], band=tuple(cfg["cache"]["band"])))


class _EncoderPayload:
    """Immutable CPU weights shared by the replicas; they are not registered parameters of the model."""

    def __init__(self, state, cfg):
        self.state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        self.arch = cfg["arch"]
        self.depth = int(cfg.get("window_depth", GROUP))

    def __deepcopy__(self, memo):
        return self


class _DeferredEncoder(torch.nn.Module):
    """No DenseNet as long as only the package features are needed; full eager model on the first fallback."""

    def __init__(self, state, cfg):
        super().__init__()
        self.num_features = int(state["slot_emb"].shape[1])
        self.payload = _EncoderPayload(state, cfg)
        self.register_buffer("_anchor", torch.empty(0), persistent=False)
        self._eager = None

    def _materialize(self):
        if self._eager is None:
            eager = make_encoder(self.payload.arch, False, 0.0,
                                 in_chans=self.payload.depth, spatial_pool="avg")
            eager.load_state_dict(self.payload.state, strict=True)
            eager = eager.to(device=self._anchor.device, dtype=self._anchor.dtype)
            eager.train(self.training)
            self._eager = eager
        return self._eager

    def forward(self, x):
        return self._materialize()(x)


def _head_only_supported(cfg):
    model = cfg["model"]
    return (model["arch"].startswith("densenet")
            and model.get("encoder_scope", "shared") == "shared"
            and model.get("spatial_pool", "avg") == "avg"
            and model.get("vit_head") is None and model.get("spatial_attn") is None
            and not model.get("slice_attn_layers", 0)
            and model.get("compartment_route") is None)


def _log_head_only(models, log, stage):
    deferred = [m.encoder for m in models if isinstance(m.encoder, _DeferredEncoder)]
    status = {"deferred": len(deferred), "models": len(models),
              "encoder_parameters": sum(p.numel() for m in models for p in m.encoder.parameters()),
              "cpu_payloads": len({id(e.payload) for e in deferred})}
    log(f"head-only: {stage} deferred={status['deferred']}/{status['models']} "
        f"encoder_parameters={status['encoder_parameters']} cpu_payloads={status['cpu_payloads']}")
    return status


def _prepare_deferred_encoders(models, trt, eligible, log):
    # If the package did not reach predict, the eager model must exist BEFORE fold_bn/layout/copies:
    # otherwise the empty module would skip transformations the usual fallback applies.
    if trt is not None and eligible:
        return 0
    restored = 0
    for model in models:
        if isinstance(model.encoder, _DeferredEncoder):
            model.encoder = model.encoder._materialize()
            restored += 1
    if restored:
        log(f"head-only: eager-fallback materialized={restored} before-layout")
    return restored


def load_ensemble(ckpt_dir: Path, device, log=print, on_cfg=None,
                  folds: list[int] | None = None, head_only: bool = False) -> tuple[list, dict]:
    """The fold checkpoints, plus the config they were trained with.

    The config is read **from the checkpoint**, not from the code bundle: the cache geometry (resolution,
    slices, crop) is as much part of the model as the weights, and taking it from a file that may have
    changed in the meantime would feed the model images at a different scale from the one it learned
    on, with nothing reporting it.
    """
    paths = sorted(Path(ckpt_dir).glob("fold*.pt"))
    if not paths:
        raise FileNotFoundError(f"no fold*.pt checkpoint in {ckpt_dir}")
    if folds is not None:
        # A subset of the folds of a 5-fold run (or of the seeds of full-data models). A requested fold that
        # is missing is an error, not a silently smaller ensemble.
        want = [Path(ckpt_dir) / f"fold{k}.pt" for k in folds]
        missing = [p.name for p in want if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"requested folds missing in {ckpt_dir}: {missing} "
                                    f"(present: {[p.name for p in paths]})")
        paths = want
    models, cfg = [], None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if cfg is None and on_cfg is not None:
            on_cfg(ck["cfg"])       # whoever waits for the run name can start now
        cfg = ck["cfg"]
        spec = _spec_of(cfg)
        deferred = (_DeferredEncoder(ck["model"], cfg["model"])
                    if head_only and _head_only_supported(cfg) else None)
        # `pretrained=False` is not a detail: with True timm would fetch the ImageNet weights from the
        # network, which the submission kernel does not have.
        m = KneeBaseline(arch=cfg["model"]["arch"], n_slots=spec.shape[0],
                         n_slices=spec.slices, pretrained=False, drop_rate=0.0,
                         img_size=spec.img,
                         slot_prior_strength=cfg["model"].get("slot_prior_strength", 0.0),
                         slot_prior_mask_at=cfg["model"].get("slot_prior_mask_at"),
                         window_stride=cfg["model"].get("window_stride", 1),
                         window_depth=cfg["model"].get("window_depth", GROUP),
                         context_layers=cfg["model"].get("context_layers", 0),
                         context_dim=cfg["model"].get("context_dim", 256),
                         context_heads=cfg["model"].get("context_heads", 4),
                         context_pos=cfg["model"].get("context_pos"),
                         pooling=cfg["model"].get("pooling", "attention"),
                         attn_hidden=cfg["model"].get("attn_hidden", 128),
                         spatial_attn=cfg["model"].get("spatial_attn"),
                         encoder_scope=cfg["model"].get("encoder_scope", "shared"),
                         spatial_pool=cfg["model"].get("spatial_pool", "avg"),
slice_attn_layers=cfg["model"].get("slice_attn_layers", 0),
                         slice_attn_dim=cfg["model"].get("slice_attn_dim", 256),
                         slice_attn_heads=cfg["model"].get("slice_attn_heads", 4),
                                                  input_norm=cfg["model"].get("input_norm"),
                         compartment_route=cfg["model"].get("compartment_route"),
                         finding_layers=cfg["model"].get("finding_layers", 0),
                         finding_dim=cfg["model"].get("finding_dim", 256),
                         finding_heads=cfg["model"].get("finding_heads", 4),
                         vit_head=cfg["model"].get("vit_head"),
                         img_crop=cfg["model"].get("img_crop"),
                         img_resize=cfg["model"].get("img_resize"),
                         encoder_override=deferred)
        state = (ck["model"] if deferred is None else
                 {k: v for k, v in ck["model"].items() if not k.startswith("encoder.")})
        m.load_state_dict(state, strict=True)
        # In blocks of one study for the encoders that do not fit a T4 at 704 windows. DenseNets stay at 0.
        m.encode_chunk = encode_chunk_for(
            cfg["model"]["arch"],
            spec.shape[0] * windows_per_slot(spec.slices, m.window_stride, m.window_depth))
        models.append(m.eval().to(device))
    if head_only:
        _log_head_only(models, log, "load")
    log(f"{len(models)} folds from {ckpt_dir}: {', '.join(p.name for p in paths)}")
    return models, cfg


_infer_ctx = getattr(torch, "inference_mode", torch.no_grad)


def skip_cpu_isa_check() -> None:
    """Tells inductor that CPU vectorisation is fine, without checking it.

    The first `aoti_load_package` cost 10.3 s, all in `cpu_vec_isa.check_build`: it compiles a test
    .cpp for AVX512 and AVX2 and loads it in a subprocess that re-imports torch. It only matters for
    inductor's CPU kernels, which our CUDA graphs do not have. With `vec_isa_ok = True` loading drops to
    0.04 s. The result does not change: it is a check, not an optimisation.
    """
    try:
        import torch._inductor.config as icfg
        icfg.cpp.vec_isa_ok = True
    except Exception:
        pass


def compile_encoders(models: list, log=print) -> bool:
    """`torch.compile` on the encoder of each model. Returns whether it worked.

    DenseNet121 fp16 goes from 700 to 1,057 img/s (1.51x) with `mode="default"`, for 78 s of compilation
    over 5 folds; `max-autotune` costs 178 s for 980 img/s and `reduce-overhead` gives less. Not
    bit-exact (max |delta logit| 4e-3 from fp16 fusions). `dynamic=False` because shapes are fixed by
    construction (`_pad_batch` pads the last batch) and a recompilation would cost as much as the first.
    Any error here leaves the models eager: a slow submission is worth more than a missing one. Not used
    by the final submission.
    """
    if not hasattr(torch, "compile"):
        return False
    skip_cpu_isa_check()
    try:
        done: dict[int, torch.nn.Module] = {}
        for m in models:
            # Shared encoder (`share_encoder`): a single graph, compiled once.
            if id(m.encoder) not in done:
                done[id(m.encoder)] = torch.compile(m.encoder, mode="default", dynamic=False)
            m.encoder = done[id(m.encoder)]
        return True
    except Exception as exc:
        log(f"torch.compile not available, staying eager: {type(exc).__name__}: {exc}")
        return False


class _PackagedEncoder(torch.nn.Module):
    """The encoder from an AOTInductor package: an already compiled library, no dynamo or triton at use
    time. `_orig_mod` is the eager encoder, for the fallback.
    """

    def __init__(self, runner, orig: torch.nn.Module) -> None:
        super().__init__()
        self.runner = runner
        self._orig_mod = orig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The runner uses the current device's stream: on `cuda:1` outside its context it gave "invalid
        # resource handle". Shape and strides are those of the export: the windows of `make_windows` can be
        # views.
        x = x.contiguous()
        if x.is_cuda:
            with torch.cuda.device(x.device):
                return self.runner(x)
        return self.runner(x)


def study_aoti_meta(n_windows: int, depth: int, spec_img: int) -> dict:
    """The "single-study" AOTI packages: the encoder on (n_windows, depth, img, img) fp16, i.e. the windows
    of one study per call (`encode_chunk`), one package per fold. They do not depend on the batch or on
    the number of folds loaded: they serve any subset of folds and any number of GPUs.
    """
    return {"input_shape": [int(n_windows), int(depth), int(spec_img), int(spec_img)],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


# The AOTI container, at every call, waits for the previous call on the same runner to finish
# (`cudaEventSynchronize`, spinning). In blocks of one study that is 8 calls per batch per GPU: the two
# T4s ran in series and the main thread spun (92 s of CPU per pass against 12). `run_single_threaded`
# skips that wait: the calls queue on the stream as in eager. The main process calls from a single
# thread, so it is the right mode.
AOTI_SINGLE_THREADED = True


def load_study_packages(aoti_dir: Path, fold_ids: list[int], devices: list, meta_expected: dict,
                        log=print) -> list[list] | None:
    """The single-study AOTI runners: `[device][fold]`, or None if the folder is missing, does not match
    (shape, torch, GPU) or a load fails. One runner per device (`device_index`): the package weights live
    on the GPU where it is loaded.
    """
    aoti_dir = Path(aoti_dir)
    meta_file = aoti_dir / "meta.json"
    if not meta_file.is_file():
        log(f"single-study AOTI packages missing in {aoti_dir}")
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        bad = {k: (meta.get(k), v) for k, v in meta_expected.items() if meta.get(k) != v}
        if bad:
            log(f"AOTI packages in {aoti_dir} do not match: {bad}")
            return None
        import importlib
        importlib.import_module("torch._inductor.codecache")   # aoti_load_package assumes them
        importlib.import_module("torch._inductor.package")
        skip_cpu_isa_check()
        out = []
        for d in devices:
            idx = d.index if d.index is not None else torch.cuda.current_device()
            with torch.cuda.device(idx):
                out.append([torch._inductor.aoti_load_package(
                    str(aoti_dir / f"encoder_fold{k}.pt2"),
                    run_single_threaded=AOTI_SINGLE_THREADED, device_index=idx)
                    for k in fold_ids])
        return out
    except Exception as exc:
        log(f"single-study AOTI packages not loaded from {aoti_dir}: {type(exc).__name__}: "
            f"{str(exc)[:300]}")
        return None


def apply_stride(models: list, stride: int | None, log=print) -> int | None:
    """Changes the inference window stride on all folds (fewer windows over the same slices, an efficiency
    lever). `None` = the checkpoint's, touches nothing. Only for models without context across windows
    (the position along the slices is wired to the training stride). Returns the stride in effect.
    """
    if stride is None:
        return models[0].window_stride
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if any(getattr(m, "context_layers", 0) > 0 for m in models):
        raise ValueError("the inference stride cannot be changed on a model with context across "
                         "windows: the position along the slices is wired to the training stride")
    was = models[0].window_stride
    for m in models:
        m.window_stride = stride
    log(f"inference window stride: {was} -> {stride} (from the checkpoint: {was})")
    return stride


def aoti_meta(models: list, batch_size: int, n_windows: int, spec_img: int) -> dict:
    """What must match between whoever built the packages and whoever uses them: the input shape (fixed:
    the batch is always padded), torch, the GPU.
    """
    return {"n_folds": len(models), "batch_size": batch_size,
            "input_shape": [batch_size * n_windows, models[0].window_depth, spec_img, spec_img],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def load_aoti_encoders(aoti_dir: Path, models: list, meta_expected: dict, log=print) -> bool:
    """Replaces the encoders with the `.pt2` packages of `aoti_dir`, if they match.

    5 packages load in 4.8 s, first forward 2.6 s, against ~19 s for `torch.compile` with the artifact
    cache; steady state 1.94 s/batch against 2.2-2.5. Same max |delta logit| from eager (4e-3). Any
    mismatch (shape, torch, GPU) or error leaves the encoders as they are: `torch.compile` is tried next,
    then eager.
    """
    aoti_dir = Path(aoti_dir)
    meta_file = aoti_dir / "meta.json"
    if not meta_file.is_file():
        return False
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        same = {k: meta.get(k) == v for k, v in meta_expected.items()}
        if not all(same.values()):
            log(f"AOTI packages in {aoti_dir} do not match: "
                f"{ {k: (meta.get(k), meta_expected[k]) for k, ok in same.items() if not ok} }")
            return False
        import importlib
        importlib.import_module("torch._inductor.codecache")   # aoti_load_package assumes them
        importlib.import_module("torch._inductor.package")
        skip_cpu_isa_check()
        runners = [torch._inductor.aoti_load_package(str(aoti_dir / f"encoder_fold{k}.pt2"))
                   for k in range(len(models))]
    except Exception as exc:
        log(f"AOTI packages not loaded from {aoti_dir}: {type(exc).__name__}: {str(exc)[:300]}")
        return False
    for m, r in zip(models, runners):
        m.encoder = _PackagedEncoder(r, m.encoder)
    log(f"encoders from {len(runners)} AOTI packages in {aoti_dir}")
    return True


def build_aoti_encoders(models: list, out_dir: Path, meta: dict, log=print) -> Path:
    """Exports and compiles with AOTInductor the encoder of each fold (`torch.export` +
    `aoti_compile_and_package`) on the fixed shape of the padded batch. Runs in a benchmark kernel, on
    the same image and the same T4 as the rerun: ~105 s for the first fold, ~40 for the next ones. The
    models must already be in fp16 and in eval.

    On the Kaggle image the linker does not find `-lcuda` (there is `libcuda.so.1`, not `libcuda.so`): a
    link in a folder of ours passed to gcc via LIBRARY_PATH.
    """
    import glob
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = [c for pat in ("/usr/local/nvidia/lib64/libcuda.so*", "/usr/lib/x86_64-linux-gnu/libcuda.so*")
             for c in glob.glob(pat)]
    real = next((c for c in cands if c.endswith(".so.1")), None)
    if real is not None:
        stub = Path("/tmp/cudastub")
        stub.mkdir(exist_ok=True)
        if not (stub / "libcuda.so").exists():
            (stub / "libcuda.so").symlink_to(real)
        os.environ["LIBRARY_PATH"] = ":".join(p for p in (str(stub), os.environ.get("LIBRARY_PATH")) if p)
    x = torch.zeros(meta["input_shape"], dtype=torch.float16, device="cuda")
    for k, m in enumerate(models):
        t = time.perf_counter()
        enc = getattr(m.encoder, "_orig_mod", m.encoder).eval()
        with torch.no_grad():
            ep = torch.export.export(enc, (x,), strict=False)
            path = torch._inductor.aoti_compile_and_package(
                ep, package_path=str(out_dir / f"encoder_fold{k}.pt2"))
        log(f"AOTI fold{k}: {Path(path).name} {Path(path).stat().st_size / 2**20:.1f} MiB "
            f"in {time.perf_counter() - t:.0f}s")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def _free_cuda() -> None:
    """Gives back to the GPU what a failed forward left behind: first the garbage collector (the frames of
    the traceback just released), then the allocator cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def uncompile_encoders(models: list) -> None:
    """Restores the eager encoders (`OptimizedModule._orig_mod`): the fallback when compilation fails on the
    first forward, which is where it actually happens (`torch.compile` is lazy, and on a GPU triton does
    not support, or with an inductor error, the exception arrives there). Better slow eager than a
    missing submission.
    """
    for m in models:
        m.encoder = getattr(m.encoder, "_orig_mod", m.encoder)


def share_encoder(models: list) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """A single encoder for all folds: the weights of each in a bank, copied in before the forward with
    `load_fold_weights`.

    Call it **before** `compile_encoders`: so dynamo traces and inductor compiles a single graph instead
    of five identical ones. The weight copy (one `_foreach_copy_` of ~16 MB in fp16) costs microseconds
    against ~0.4 s for a fold's forward. Predictions are the same: the graph reads the weights it finds
    at call time. Floating-point tensors only: `num_batches_tracked` does not matter in eval.
    """
    enc = models[0].encoder

    def bank(e):
        return [t.detach() for t in list(e.parameters()) + list(e.buffers())
                if t.is_floating_point()]

    live = bank(enc)
    banks = []
    for m in models:
        banks.append([t.clone() for t in bank(m.encoder)])
        m.encoder = enc
    return live, banks


def load_fold_weights(live: list[torch.Tensor], bank: list[torch.Tensor]) -> None:
    torch._foreach_copy_(live, bank)


def _make_loader(ds, batch_size: int, workers: int, device,
                 prefetch: int | None = None) -> DataLoader:
    """The DataLoader delivers micro-batches of `batch_size // workers` studies, which `_regroup` puts back
    together into batches of `batch_size` for the GPU.

    A DataLoader assigns every whole batch to one worker: with batches of 8 the first one arrived after
    ~10 s, the time for one worker to decode 8 studies alone while the other three prepared the next
    batches. With micro-batches of 2 the four workers decode the first 8 studies together and the GPU
    starts after ~3 s. The prefetch is scaled so that the queued studies stay the same (2 full batches
    per worker).
    """
    micro = max(1, batch_size // max(workers, 1)) if workers > 0 else batch_size
    kwargs = {
        "batch_size": micro,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": (device.type == "cuda"),
        "worker_init_fn": _nice_worker,
    }
    if getattr(ds, "raw", False):
        kwargs["collate_fn"] = raw_crop.collate_raw
    if workers > 0:
        kwargs["persistent_workers"] = True
        # `prefetch`: micro-batches queued per worker instead of two full batches (float32 stacks weigh 4
        # times the uint8 ones).
        kwargs["prefetch_factor"] = prefetch or max(2, 2 * batch_size // micro)
    return DataLoader(ds, **kwargs)


def _regroup(loader, batch_size: int, cat_vol: bool = True):
    """Consecutive micro-batches regrouped into batches of exactly `batch_size` studies (the last may be
    shorter): a micro-batch straddling the boundary is split and the rest opens the next batch. With 32
    studies per batch and 6 workers the micro-batches have 5 studies, and without the split the batch
    reached 35, one part more than there are GPUs. `cat_vol=False`: the volumes stay the list of
    micro-batches (in pinned memory, the slices are views), which `_forward_split` copies to the GPUs and
    concatenates there: concatenating here cost a RAM copy of the whole batch, 1 GB with float32 stacks.
    """
    buf: list[tuple] = []
    n = 0

    def out(buf):
        parts = list(zip(*buf))
        if isinstance(parts[0][0], list):           # raw payloads (`DicomStudies(raw=True)`)
            vol = [p for ps in parts[0] for p in ps] if cat_vol else list(parts[0])
        else:
            vol = torch.cat(parts[0]) if cat_vol else list(parts[0])
        return (vol, *(torch.cat(p) for p in parts[1:]))

    for item in loader:
        buf.append(item)
        n += _nrows(item[0])
        while n >= batch_size:
            head, tail, m = [], [], 0
            for it in buf:
                k = _nrows(it[0])
                if m + k <= batch_size:
                    head.append(it)
                elif m < batch_size:
                    cut = batch_size - m
                    head.append(tuple(x[:cut] for x in it))
                    tail.append(tuple(x[cut:] for x in it))
                else:
                    tail.append(it)
                m += k
            yield out(head)
            buf, n = tail, n - batch_size
    if buf:
        yield out(buf)


def _nrows(x) -> int:
    """Studies in a micro-batch: a tensor (B, ...) or a list of raw payloads."""
    return len(x) if isinstance(x, list) else x.shape[0]


def _rows(parts: list, s: int, e: int) -> list:
    """Rows [s, e) of a list of micro-batches, as views (they stay pinned)."""
    out, off = [], 0
    for p in parts:
        a, b = max(s - off, 0), min(e - off, _nrows(p))
        if a < b:
            out.append(p[a:b])
        off += _nrows(p)
    return out


def _raw_on_device(payloads: list, spec: CacheSpec, device, check: dict, log=print) -> torch.Tensor:
    """The raw payloads of a batch piece -> (B, S, K, img, img) uint8 on the device
    (`raw_crop.payload_to_uint8_torch`).

    Self-check: while `check["left"] > 0` the first study is also redone with the CPU path
    (`payload_to_uint8_cpu`, that of `build_study`) and the difference is added into `check`. It is not
    bit-identical (the resize), but if the difference exceeds the thresholds (`check["max_frac"]` of
    differing pixels, `check["max_abs"]` levels) the path is switched off and from then on the volumes
    are built on the main process CPU: slower, but the usual ones.
    """
    if not check["ok"]:
        return torch.from_numpy(np.stack([raw_crop.payload_to_uint8_cpu(p, spec)
                                          for p in payloads])).to(device)
    out = raw_crop.payloads_to_uint8_torch(payloads, spec, device)
    check["cpu_series"] += sum(int((p["meta"][:, raw_crop.M_MODE] == raw_crop.MODE_CPU).sum())
                               for p in payloads)
    check["series"] += sum(int(p["meta"].shape[0]) for p in payloads)
    if check["left"] > 0 and payloads:
        check["left"] -= 1
        ref = raw_crop.payload_to_uint8_cpu(payloads[0], spec)
        c = raw_crop.compare_uint8(ref, out[0].cpu().numpy())
        check["checked"] += 1
        check["pixels"] += c["pixels"]
        check["diff"] += c["diff"]
        check["worst_abs"] = max(check["worst_abs"], c["max_abs"])
        if c["diff"] > check["max_frac"] * c["pixels"] or c["max_abs"] > check["max_abs"]:
            check["ok"] = False
            log(f"!! GPU volume too different from the CPU one ({c}): CPU from now on")
            return _raw_on_device(payloads, spec, device, check, log)
    return out


class _PrepFailed:
    """An error of the preparation thread, handed over to the main loop."""

    def __init__(self, exc: BaseException):
        self.exc = exc


def _prep_raw_ahead(batches, spec: CacheSpec, devices: list, per_gpu: int, check: dict,
                    log=print, depth: int = 1):
    """The batches of raw payloads (`DicomStudies(raw=True)`) already turned into uint8 volumes on the
    devices, one batch ahead of the model: a dedicated thread runs `_raw_on_device` for every piece of
    `per_gpu` studies (the same split as `_forward_split`), on the piece's device and on its own CUDA
    stream, while the GPUs compute the previous batch. Doing it in the main process with the GPUs idle
    cost +63 s of GPU and 41 s of CPU in series with the model.

    Returns `(pieces, mask, idx, dt, errs)` with `pieces` = list of uint8 tensors (one per GPU), which
    `_forward_split` takes as they are (`_rows` cuts them at the piece boundaries, `.to(d)` does not
    copy). The volumes are complete when they arrive (synchronisation of the preparation stream) and
    marked for the default stream (`record_stream`): the allocator does not reuse their memory until the
    model has read them. An error in the thread is re-raised in the main loop.
    """
    import queue
    import threading
    q: queue.Queue = queue.Queue(maxsize=depth)
    streams = [torch.cuda.Stream(d) if d.type == "cuda" else None for d in devices]
    done = object()

    def one_piece_async(args):
        j, sub = args
        d, st = devices[j], streams[j]
        ctx = (torch.cuda.stream(st) if st is not None else contextlib.nullcontext())
        with torch.cuda.device(d) if st is not None else contextlib.nullcontext(), ctx:
            if st is not None and check.get("left", 0) == 0 and check.get("ok", True):
                v = raw_crop.payloads_to_uint8_torch(sub, spec, d)
                c_ser = sum(int((p["meta"][:, raw_crop.M_MODE] == raw_crop.MODE_CPU).sum())
                            for p in sub)
                n_ser = sum(int(p["meta"].shape[0]) for p in sub)
                check["cpu_series"] += c_ser
                check["series"] += n_ser
            else:
                v = _raw_on_device(sub, spec, d, check, log)
        return v

    def work() -> None:
        try:
            for vol, mask, idx, dt, errs in batches:
                t0 = time.perf_counter()
                pays = ([p for part in vol for p in part]
                        if vol and isinstance(vol[0], list) else vol)
                items = [(j, pays[s:s + per_gpu])
                         for j, s in enumerate(range(0, len(pays), per_gpu))]
                pieces = [one_piece_async(it) for it in items]
                for j, p in enumerate(pieces):
                    st = streams[j]
                    if st is not None:
                        p.record_stream(torch.cuda.default_stream(devices[j]))
                        st.synchronize()
                check["prep_s"] = check.get("prep_s", 0.0) + time.perf_counter() - t0
                q.put((pieces, mask, idx, dt, errs))
            q.put(done)
        except BaseException as exc:                            # noqa: BLE001
            q.put(_PrepFailed(exc))

    th = threading.Thread(target=work, name="raw-prep", daemon=True)
    th.start()
    while True:
        it = q.get()
        if it is done:
            return
        if isinstance(it, _PrepFailed):
            raise it.exc
        yield it


def _uint8_on_device(v: torch.Tensor, check: dict, first_cpu=None, log=print) -> torch.Tensor:
    """(B, S, K, H, W) float32 on the device -> uint8, study by study (the float64 of one whole stack per
    study: 128 MB instead of 1 GB per batch).

    Self-check: while `check["left"] > 0`, the first study (`first_cpu`, the same stacks in RAM) is
    compared with NumPy's `cache._to_uint8`. A disagreement switches the path off (`check["ok"] = False`)
    and from then on normalisation goes back to the CPU, in the main process: slower, but the usual one.
    """
    if not check["ok"]:
        cpu = v.cpu().numpy()
        return torch.from_numpy(np.stack([np.stack([cache_mod._to_uint8(s) for s in study])
                                          for study in cpu])).to(v.device)
    out = torch.empty(v.shape, dtype=torch.uint8, device=v.device)
    for b in range(v.shape[0]):
        out[b] = to_uint8_torch(v[b])
    if check["left"] > 0 and first_cpu is not None:
        check["left"] -= 1
        ref = np.stack([cache_mod._to_uint8(s) for s in first_cpu.numpy()])
        got = out[0].cpu().numpy()
        check["checked"] += 1
        if not np.array_equal(ref, got):
            check["ok"] = False
            log(f"!! uint8 on the GPU differs from NumPy ({int((ref != got).sum())} pixels): "
                f"normalising on CPU from now on")
            return _uint8_on_device(v, check)
    return out


# Memory of the decoding workers. Every study allocates hundreds of MB of new arrays (file bytes,
# float32 slices, crops, resizes, stacks, the float64 of `_to_uint8`), and with the default thresholds
# glibc returns the large blocks to the system as soon as they are freed: for the next study the same
# pages come back as page faults to be zeroed. 48 MB more per study cost the workers ~4 ms/MB, the
# price of new pages rather than of a copy. "keep": `mallopt` sets the mmap threshold to the maximum
# (32 MB, above the arrays of a study: 8 MB stacks, 16 MB float64) and disables heap trimming, so freed
# blocks are reused. Only who supplies the memory changes, not the arithmetic. In an A/B on the T4
# benchmark: worker page faults -81%, worker system CPU -44%, worker CPU -12.5%, steady state -18.7%.
_WORKER_MALLOC = ["keep"]
_M_TRIM_THRESHOLD, _M_MMAP_THRESHOLD = -1, -3          # glibc's malloc.h


def _keep_freed_memory() -> bool:
    """glibc's `mallopt`: blocks up to 32 MB from the heap and no trimming. False outside glibc (Windows,
    musl) or if glibc rejects the values.
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        return (libc.mallopt(_M_MMAP_THRESHOLD, 32 * 1024 * 1024) == 1
                and libc.mallopt(_M_TRIM_THRESHOLD, 2 ** 31 - 1) == 1)
    except (OSError, AttributeError):
        return False


def _nice_worker(_worker_id: int) -> None:
    try:
        os.nice(10)
    except (AttributeError, OSError):    # Windows, or no permission: carry on anyway
        pass
    if _WORKER_MALLOC[0] == "keep":
        _keep_freed_memory()


def _skip_ok(m) -> bool:
    """Whether the model tolerates skipping the windows of empty slots (`_model_logits`): shared encoder, no
    attention across windows inside the encoder (`slice_attn`) and no intra-window head (`vit_head`),
    which reduces the tokens inside `forward` and does not accept `encoded`, so the encoder runs on all
    windows (slots are ~97% full).
    """
    return (getattr(m, "encoder_scope", "shared") == "shared"
            and int(getattr(m, "slice_attn_layers", 0)) == 0 and hasattr(m, "_encode")
            and getattr(m, "vit_head", None) is None)


def _encoder_chunk(m) -> int:
    """The multiple the encoder wants for the number of images: the block of the TensorRT/dense8 engines
    (CUDA graphs of `nb` images), 1 for the eager encoder.
    """
    r = getattr(getattr(m, "encoder", None), "runner", None)
    return int(getattr(r, "nb", 1) or 1)


def _skip_selection(mask_cpu: torch.Tensor, slot_cpu: torch.Tensor, chunk: int, device):
    """The windows of the present slots of a batch piece: (indices with padding rows up to a multiple of
    `chunk`, real indices, how many), on the device. None if there is nothing to skip (no valid window,
    or padding that saves nothing).
    """
    valid = mask_cpu[:, slot_cpu].reshape(-1)
    idx = valid.nonzero().reshape(-1)
    nv = int(idx.numel())
    npad = -(-nv // chunk) * chunk
    if nv == 0 or npad >= valid.numel():
        return None
    pad = torch.cat([idx, idx[-1:].expand(npad - nv)]) if npad > nv else idx
    return (pad.to(device, non_blocking=True), idx.to(device, non_blocking=True), nv)


def _model_logits(m, w: torch.Tensor, slot_of, mask, sel) -> torch.Tensor:
    """The model logits on windows (B, W, ...) computing the encoder only on those of the present slots
    (`sel` from `_skip_selection`); the others enter as null features. Bit-exact: the windows of absent
    slots are masked keys in the context and -inf weights in the pooling (`KneeBaseline._features_full`),
    their features touch no valid output. With `sel` None it is `m(w, slot_of, mask)`.
    """
    if sel is None:
        return m(w, slot_of, mask)
    pad, idx, nv = sel
    b, nw = w.shape[:2]
    z = m._encode(m.encoder, w.flatten(0, 1).index_select(0, pad))
    f = z.new_zeros((b * nw, z.shape[1]))
    f.index_copy_(0, idx, z[:nv])
    return m(w, slot_of, mask, encoded=f.view(b, nw, -1))


def _model_logits_budget(m, w: torch.Tensor, slot_of, mask, idx: torch.Tensor) -> torch.Tensor:
    """The model logits on windows `idx` only (the member's design, `knee.window_budget`): the encoder sees
    B * len(idx) images, the head (context and per-finding attention) only the member's windows, as in
    training with `k_train` windows.
    """
    b = w.shape[0]
    z = m._encode(m.encoder, w.index_select(1, idx).flatten(0, 1))
    kw = {"window_index": idx} if getattr(m, "context_pos", None) is not None else {}
    return m(None, slot_of.index_select(0, idx), mask, encoded=z.view(b, idx.numel(), -1), **kw)


def _budget_slice_index(idx: torch.Tensor, per: int, n_slices: int, stride: int,
                        depth: int, offset: int = 0) -> torch.Tensor:
    """The slice indices in `v.view(B, S * K, H, W)` of windows `idx` (`make_windows` followed by
    `index_select(1, idx).flatten(0, 1)`). `offset` = the view's `slice_offset` (`off1`: the windows of
    `make_windows(vol[:, :, 1:])`, same number per slot with 24 slices).
    """
    s_id = torch.div(idx, per, rounding_mode="floor")
    wi = idx - s_id * per
    base = (s_id * n_slices + wi * stride + offset).unsqueeze(1)
    offs = torch.arange(depth, device=idx.device, dtype=idx.dtype).unsqueeze(0)
    return (base + offs).reshape(-1)


_GEO_TMP: dict = {}
# Self-check of the TTA on the uint8 volume (`KNEE_TTA_U8_CHECK` = how many calls to check): on the
# first two studies of the batch the previous path (`variant_windows` on the whole fp16 volume +
# `_model_logits_budget`) must give the same probabilities bit for bit.
_TTA_U8_CHECK = {"left": int(os.environ.get("KNEE_TTA_U8_CHECK", "0")), "checked": 0, "max_abs": 0.0,
                 "views": {}, "errors": 0}


def _tta_u8_check(m, v_u8: torch.Tensor, slot_of, mask, idx: torch.Tensor, var: Variant,
                  stride: int, depth: int, p: torch.Tensor, log=print, n: int = 2) -> None:
    _TTA_U8_CHECK["left"] -= 1
    try:
        with torch.autocast("cuda", enabled=False):
            vol01 = v_u8[:n].half().mul_(1.0 / 255.0)
            w, _ = variant_windows(vol01, var, stride, depth)
        with torch.autocast("cuda", enabled=v_u8.is_cuda, dtype=torch.float16):
            ref = torch.sigmoid(_model_logits_budget(m, w, slot_of, mask[:n], idx).float())
        d = float(torch.nan_to_num(ref - p[:n], nan=0.0).abs().max())
        _TTA_U8_CHECK["checked"] += 1
        _TTA_U8_CHECK["max_abs"] = max(_TTA_U8_CHECK["max_abs"], d)
        _TTA_U8_CHECK["views"][var.name] = max(_TTA_U8_CHECK["views"].get(var.name, 0.0), d)
        log(f"tta-u8 self-check {var.name} on {v_u8.device}: max|d| {d:.3g}")
    except Exception as exc:                                         # noqa: BLE001
        _TTA_U8_CHECK["errors"] += 1
        log(f"!! tta-u8 self-check failed: {type(exc).__name__}: {str(exc)[:300]}")


def _geo_scratch(n: int, depth: int, h: int, w: int, device, dtype=torch.float16) -> torch.Tensor:
    """An fp16 buffer (n, depth, h, w) reused per device: the windows of a geometric view before the zoom
    (the encoder graph reads its own buffer, which receives the output of `affine_windows`).
    """
    key = (str(device), depth, h, w, dtype)
    t = _GEO_TMP.get(key)
    if t is None or t.shape[0] < n:
        t = _GEO_TMP[key] = torch.empty((n, depth, h, w), dtype=dtype, device=device)
    return t[:n]


def _model_logits_budget_u8(m, v_u8: torch.Tensor, slot_of, mask, idx: torch.Tensor,
                            slice_idx: torch.Tensor, depth: int,
                            var: Variant | None = None) -> torch.Tensor:
    """Like `_model_logits_budget`, starting from the uint8 volume `(B, S, K, H, W)` without building and
    converting to fp16 the windows the member does not use.

    `var` (TTA without flips): the member's view. The slice shift is already in `slice_idx`; zoom and
    shift are applied only to the member's windows with `tta.affine_windows` in fp16 and outside
    autocast, like `variant_windows` on the whole volume (`affine_windows` works window by window with
    the same grid: bit-identical result). The final submission uses this path with the single view
    `s1.10` (zoom 1.10).
    """
    b, s, k, h, w = v_u8.shape
    nw = int(idx.numel())
    n = b * nw
    runner = getattr(getattr(m, "encoder", None), "runner", None)
    buf = runner.graph_in_buf(n) if hasattr(runner, "graph_in_buf") else None
    geo = var is not None and var.geometric
    dst = _geo_scratch(n, depth, h, w, v_u8.device, buf.dtype) if (geo and buf is not None) else buf
    done_gather = False
    if dst is not None and v_u8.is_cuda:
        from .data.fused_uint8 import gather_u8_f16
        done_gather = gather_u8_f16(v_u8, slice_idx, dst)
    if done_gather:
        x = dst
    elif dst is not None:
        u8_sel = v_u8.view(b, s * k, h, w).index_select(1, slice_idx).view(n, depth, h, w)
        dst.copy_(u8_sel).mul_(1.0 / 255.0)
        x = dst
    else:
        u8_sel = v_u8.view(b, s * k, h, w).index_select(1, slice_idx).view(n, depth, h, w)
        x = (u8_sel.half() if v_u8.is_cuda else u8_sel.float()).mul_(1.0 / 255.0)
    if geo:
        from .tta import affine_windows
        with torch.autocast("cuda", enabled=False):
            y = affine_windows(x.view(n, 1, depth, h, w), var.scale, var.tx, var.ty).view(n, depth, h, w)
        if buf is not None:
            buf.copy_(y)
            x = buf
        else:
            x = y
    z = m._encode(m.encoder, x)
    kw = {"window_index": idx} if getattr(m, "context_pos", None) is not None else {}
    return m(None, slot_of.index_select(0, idx), mask, encoded=z.view(b, nw, -1), **kw)


# `KNEE_TTA_BATCH_VIEWS=1`: with TTA, a single encoder call per seed with the windows of all views in
# a row in the graph buffer (`_seed_logits_views_u8`): the package's blocks of `nb` images fill up
# better (704 images in 720 instead of four calls of 154-184 in 900).
_BATCH_VIEWS = [os.environ.get("KNEE_TTA_BATCH_VIEWS", "0") == "1"]


def _seed_logits_views_u8(m, v_u8: torch.Tensor, slot_of, mask, idxs: list, sidxs: list, depth: int,
                          variants: list) -> list:
    """The logits of a seed for every view (`_model_logits_budget_u8` view by view) with the encoder called
    once: the windows of view i occupy segment i of the graph buffer, in the same (study, window) order
    as the per-view call; every image is computed on its own, so the features are the same. Without a
    graph buffer or off the GPU it falls back to per-view calls.
    """
    b, s, k, h, w = v_u8.shape
    ns = [int(i.numel()) for i in idxs]
    n = b * sum(ns)
    runner = getattr(getattr(m, "encoder", None), "runner", None)
    buf = runner.graph_in_buf(n) if hasattr(runner, "graph_in_buf") else None
    if buf is None:
        return [_model_logits_budget_u8(m, v_u8, slot_of, mask, i, si, depth, var=var)
                for i, si, var in zip(idxs, sidxs, variants)]
    from .data.fused_uint8 import gather_u8_f16
    from .tta import affine_windows
    off = 0
    for si, var, nv in zip(sidxs, variants, ns):
        seg = buf[off:off + b * nv]
        geo = var is not None and var.geometric
        dst = _geo_scratch(b * nv, depth, h, w, v_u8.device, buf.dtype) if geo else seg
        if not gather_u8_f16(v_u8, si, dst):
            u8_sel = v_u8.view(b, s * k, h, w).index_select(1, si).view(b * nv, depth, h, w)
            dst.copy_(u8_sel).mul_(1.0 / 255.0)
        if geo:
            with torch.autocast("cuda", enabled=False):
                y = affine_windows(dst.view(b * nv, 1, depth, h, w), var.scale, var.tx, var.ty)
            seg.copy_(y.view(b * nv, depth, h, w))
        off += b * nv
    z = m._encode(m.encoder, buf)
    out, off = [], 0
    for idx, nv in zip(idxs, ns):
        kw = {"window_index": idx} if getattr(m, "context_pos", None) is not None else {}
        out.append(m(None, slot_of.index_select(0, idx), mask, encoded=z[off:off + b * nv].view(b, nv, -1), **kw))
        off += b * nv
    return out


def _forward_split(dev_models: list, devices: list, slot_ofs: list, vol, mask,
                   per_gpu: int, stride: int, depth: int, tta, block: bool,
                   u8: dict | None = None, log=print, pool=None,
                   skip_empty: bool = False, skip_stats: dict | None = None,
                   budget: str | None = None, raw_spec: CacheSpec | None = None,
                   rawchk: dict | None = None) -> np.ndarray:
    """A batch split across the GPUs: `per_gpu` studies each (`dev_models[j]` are the folds copied to
    `devices[j]`). All parts are queued and only then waited for: launches are asynchronous, so the GPUs
    work together. Studies without slots come out as NaN, as in the single-GPU path. `slot_ofs[j]` is
    computed on the first call if missing.

    `vol` is a tensor or the list of micro-batches (`_regroup(cat_vol=False)`); if it is float32 (stacks
    before normalisation, `DicomStudies(to_uint8=False)`) percentiles and uint8 are computed on the
    device (`_uint8_on_device`, self-check state in `u8`).

    **Launches alternate between GPUs fold by fold**, as in `_forward_split_arms`: queueing all folds on
    GPU 0 first fills CUDA's launch queue and serialises the two GPUs (-57% on the steady state with the
    alternation). The mean over folds and variants is the same, summed in another order.

    `pool` (a `ThreadPoolExecutor` with one thread per GPU): every GPU computes windows, folds and
    variants in its own thread. Needed when the encoder call blocks the thread (TensorRT engines,
    `predict(trt=...)`): from a single thread the GPUs would run in series. Same arithmetic, same order
    within each GPU: identical predictions.

    `skip_empty`: the encoder only on the windows of present slots (`_model_logits`, bit-exact), the
    indices from the mask on the CPU (no GPU wait). With an 8-slot spec on average 3.86 slots out of 8
    are full: half of the windows are masked black images. `skip_stats` accumulates valid and computed
    windows.

    `budget`: the per-member window design (`knee.window_budget`: fold `f` on variant `v` sees
    `member_windows(budget, v, seed=f)`): every member passes only its windows to the encoder and the
    head. Not exact: it changes what every member sees (the AUC cost was measured on OOF). Excludes
    `skip_empty`. The final submission uses `parx`: each seed sees half of the windows, complementary.
    """
    if u8 is None:
        u8 = {"left": 0, "ok": True, "checked": 0}
    variants = tta or [None]
    for var in variants:
        if var is not None and not var.same_windows:
            raise ValueError(f"TTA {var.name}: changes the number of windows, not valid in a submission")
    # Also with TTA (geometric views and `off1`) the member's windows are gathered from the uint8 volume
    # (`_model_logits_budget_u8(var=...)`): converting the whole volume to fp16 and zooming all 88 windows
    # of every view to keep a third of them was wasted work.
    fast_u8 = budget is not None and all(
        v is None or (v.same_windows and not v.photometric) for v in variants)
    offs_v = [0 if v is None else int(v.slice_offset) for v in variants]
    starts = list(range(0, mask.shape[0], per_gpu))
    v01s, mks = [], []
    for j, s in enumerate(starts):
        d = devices[j]
        on_gpu = d.type == "cuda"
        with torch.cuda.device(d) if on_gpu else contextlib.nullcontext():
            pieces = _rows(vol, s, s + per_gpu) if isinstance(vol, list) else [vol[s:s + per_gpu]]
            if isinstance(pieces[0], list):
                # Raw payloads: resize, percentiles and uint8 here (`DicomStudies(raw=True)`).
                pieces = [_raw_on_device([q for p in pieces for q in p], raw_spec, d, rawchk,
                                         log=log)]
            elif pieces[0].dtype == torch.float32:
                # Piece by piece: the uint8 values are concatenated, not the float32 ones (4x the bytes).
                pieces = [_uint8_on_device(p.to(d, non_blocking=True), u8,
                                           first_cpu=p[0] if k == 0 else None, log=log)
                          for k, p in enumerate(pieces)]
            else:
                pieces = [p.to(d, non_blocking=True) for p in pieces]
            v = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
            mks.append(mask[s:s + per_gpu].to(d, non_blocking=True))
            if slot_ofs[j] is None:
                slot_ofs[j] = window_slot_index(v.shape[1], v.shape[2], d, stride, depth)
            if fast_u8 and on_gpu and v.dtype == torch.uint8:
                v01s.append(v)
            else:
                v01s.append((v.half() if on_gpu else v.float()).mul_(1.0 / 255.0))
    acc = [None] * len(starts)
    sels = [None] * len(starts)
    bidx: list[dict] = [{} for _ in starts]
    sidx: list[dict] = [{} for _ in starts]
    if budget is not None:
        if not all(_skip_ok(m) for ms in dev_models for m in ms):
            raise ValueError("budget: requires the shared encoder without slice_attn")
        n_slots = int(v01s[0].shape[1])
        n_slices = int(v01s[0].shape[2])
        per = windows_per_slot(n_slices, stride, depth)
        nv = len(variants)
        for j in range(len(starts)):
            for f in range(len(dev_models[j])):
                for vi in range(nv):
                    key = ("budget", budget, f, vi, n_slots, per, str(devices[j]))
                    if key not in _SKIP_CACHE:
                        _SKIP_CACHE[key] = member_windows(budget, vi, n_slots, per,
                                                          seed=f).to(devices[j])
                    bidx[j][(f, vi)] = _SKIP_CACHE[key]
                    if fast_u8:
                        skey = ("budget_slices", budget, f, vi, n_slots, n_slices, per,
                                stride, depth, offs_v[vi], str(devices[j]))
                        if skey not in _SKIP_CACHE:
                            _SKIP_CACHE[skey] = _budget_slice_index(
                                bidx[j][(f, vi)], per, n_slices, stride, depth, offs_v[vi])
                        sidx[j][(f, vi)] = _SKIP_CACHE[skey]
        if skip_stats is not None:
            b_all = int(mask.shape[0])
            skip_stats["windows"] = skip_stats.get("windows", 0) + (
                b_all * n_slots * per * nv * len(dev_models[0]))
            skip_stats["encoded"] = skip_stats.get("encoded", 0) + b_all * sum(
                int(i.numel()) for i in bidx[0].values())
    elif skip_empty and all(_skip_ok(m) for ms in dev_models for m in ms):
        chunk = _encoder_chunk(dev_models[0][0])
        # the slot of every window on the CPU, from the shapes (known without asking the GPU: a .cpu() of
        # slot_ofs would wait for all the queued work)
        key = ("slot_cpu", int(v01s[0].shape[1]), int(v01s[0].shape[2]), stride, depth)
        if key not in _SKIP_CACHE:
            _SKIP_CACHE[key] = window_slot_index(key[1], key[2], "cpu", stride, depth)
        slot_cpu = _SKIP_CACHE[key]
        for j, s in enumerate(starts):
            sels[j] = _skip_selection(mask[s:s + per_gpu], slot_cpu, chunk, devices[j])
            if skip_stats is not None:
                nall = int(mask[s:s + per_gpu].shape[0] * slot_cpu.numel())
                skip_stats["windows"] = skip_stats.get("windows", 0) + nall
                skip_stats["encoded"] = skip_stats.get("encoded", 0) + (
                    nall if sels[j] is None else int(sels[j][0].numel()))

    def member(j: int, f: int, vi: int, w: torch.Tensor) -> torch.Tensor:
        if budget is not None:
            return _model_logits_budget(dev_models[j][f], w, slot_ofs[j], mks[j],
                                        bidx[j][(f, vi)])
        return _model_logits(dev_models[j][f], w, slot_ofs[j], mks[j], sels[j])

    def one_gpu(j: int):
        # In the thread: inference_mode and autocast are per thread, so they are reopened here.
        d = devices[j]
        on_gpu = d.type == "cuda"
        a = None
        with torch.inference_mode(), \
                torch.cuda.device(d) if on_gpu else contextlib.nullcontext():
            if v01s[j].dtype == torch.uint8 and _BATCH_VIEWS[0] and len(variants) > 1:
                ps = {}
                for f in range(len(dev_models[j])):
                    with torch.autocast("cuda", enabled=on_gpu, dtype=torch.float16):
                        lgs = _seed_logits_views_u8(
                            dev_models[j][f], v01s[j], slot_ofs[j], mks[j],
                            [bidx[j][(f, vi)] for vi in range(len(variants))],
                            [sidx[j][(f, vi)] for vi in range(len(variants))], depth, variants)
                        for vi, lg in enumerate(lgs):
                            ps[(f, vi)] = torch.sigmoid(lg.float())
                    for vi in range(len(variants)):
                        if _TTA_U8_CHECK["left"] > 0 and variants[vi] is not None:
                            _tta_u8_check(dev_models[j][f], v01s[j], slot_ofs[j], mks[j],
                                          bidx[j][(f, vi)], variants[vi], stride, depth, ps[(f, vi)], log)
                for vi in range(len(variants)):          # the sum in the order of the per-view call
                    for f in range(len(dev_models[j])):
                        a = ps[(f, vi)] if a is None else a + ps[(f, vi)]
            elif v01s[j].dtype == torch.uint8:
                for vi in range(len(variants)):
                    for f in range(len(dev_models[j])):
                        with torch.autocast("cuda", enabled=on_gpu, dtype=torch.float16):
                            p = torch.sigmoid(_model_logits_budget_u8(
                                dev_models[j][f], v01s[j], slot_ofs[j], mks[j],
                                bidx[j][(f, vi)], sidx[j][(f, vi)], depth,
                                var=variants[vi]).float())
                        if _TTA_U8_CHECK["left"] > 0 and variants[vi] is not None:
                            _tta_u8_check(dev_models[j][f], v01s[j], slot_ofs[j], mks[j],
                                          bidx[j][(f, vi)], variants[vi], stride, depth, p, log)
                        a = p if a is None else a + p
            else:
                for vi, var in enumerate(variants):
                    w = (make_windows(v01s[j], stride, depth) if var is None
                         else variant_windows(v01s[j], var, stride, depth)[0])
                    for f in range(len(dev_models[j])):
                        with torch.autocast("cuda", enabled=on_gpu, dtype=torch.float16):
                            p = torch.sigmoid(member(j, f, vi, w).float())
                        a = p if a is None else a + p
        return a

    if pool is not None and len(starts) > 1:
        acc = list(pool.map(one_gpu, range(len(starts))))
    else:
        for vi, var in enumerate(variants):
            ws = []
            for j in range(len(starts)):
                d = devices[j]
                with torch.cuda.device(d) if d.type == "cuda" else contextlib.nullcontext():
                    ws.append(None if v01s[j].dtype == torch.uint8
                              else (make_windows(v01s[j], stride, depth) if var is None
                                    else variant_windows(v01s[j], var, stride, depth)[0]))
            for f in range(len(dev_models[0])):
                for j in range(len(starts)):
                    d = devices[j]
                    on_gpu = d.type == "cuda"
                    with torch.cuda.device(d) if on_gpu else contextlib.nullcontext(), \
                            torch.autocast("cuda", enabled=on_gpu, dtype=torch.float16):
                        if ws[j] is None:
                            logits = _model_logits_budget_u8(
                                dev_models[j][f], v01s[j], slot_ofs[j], mks[j],
                                bidx[j][(f, vi)], sidx[j][(f, vi)], depth, var=var)
                        else:
                            logits = member(j, f, vi, ws[j])
                        p = torch.sigmoid(logits.float())
                    acc[j] = p if acc[j] is None else acc[j] + p
    evs = []
    for j in range(len(starts)):
        d = devices[j]
        with torch.cuda.device(d) if d.type == "cuda" else contextlib.nullcontext():
            acc[j] = acc[j] / (len(dev_models[j]) * len(variants))
            ev = None
            if block and d.type == "cuda":
                ev = torch.cuda.Event(blocking=True)
                ev.record()
        evs.append(ev)
    out = []
    for pm, ev in zip(acc, evs):
        if ev is not None:
            ev.synchronize()        # sleeps instead of spinning (see `predict(gpu_wait)`)
        out.append(pm.cpu())
    batch = torch.cat(out).numpy()
    batch[(~mask.any(dim=1)).numpy()] = np.nan
    return batch


_SKIP_CACHE: dict = {}


def _forward_split_arms(dev_arms: list, devices: list, slot_ofs: dict, arms: list, vol, mask,
                        per_gpu: int, tta, block: bool) -> list[np.ndarray]:
    """`_forward_split` for several arms: the batch split across GPUs, `per_gpu` studies each, and on every
    GPU all arms (`dev_arms[j][i]` are the folds of arm `i` copied to `devices[j]`) on the same stack
    transferred once. Returns one (B, T) array per arm, in the order of `arms`; studies without slots come
    out as NaN as in the single-GPU path. `slot_ofs[(j, name)]` is computed on the first call if missing.

    **Launches alternate between GPUs model by model** (fold `f` of arm `i` on GPU 0, then on GPU 1, then
    the next fold). Queueing all 15 models of a three-arm ensemble on GPU 0 first and then those of GPU 1
    serialised them: CUDA's launch queue has a limit (~1,000 pending kernels), the main process stopped
    there until GPU 0 drained, and GPU 1 started at the end of the batch (16 studies on two T4s took 1.7
    times as long as 8 on one). The arithmetic is the same: the mean over folds and TTA variants of
    `_probs_tta` and `fold_probs`, summed in another order.
    """
    starts = list(range(0, mask.shape[0], per_gpu))
    v01s, mks = [], []
    for j, s in enumerate(starts):
        d = devices[j]
        with torch.cuda.device(d) if d.type == "cuda" else contextlib.nullcontext():
            pieces = _rows(vol, s, s + per_gpu) if isinstance(vol, list) else [vol[s:s + per_gpu]]
            pieces = [p.to(d, non_blocking=True) for p in pieces]
            v = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
            mks.append(mask[s:s + per_gpu].to(d, non_blocking=True))
            v01s.append((v.half() if d.type == "cuda" else v.float()).mul_(1.0 / 255.0))
    variants = tta or [None]
    for var in variants:
        if var is not None and not var.same_windows:
            raise ValueError(f"TTA {var.name}: changes the number of windows, not valid in a submission")
    acc = [[None] * len(arms) for _ in starts]
    for i, a in enumerate(arms):
        k, stride, depth = a["n_slots"], a["stride"], a["depth"]
        for var in variants:
            ws = []
            for j in range(len(starts)):
                d = devices[j]
                with torch.cuda.device(d) if d.type == "cuda" else contextlib.nullcontext():
                    if slot_ofs.get((j, a["name"])) is None:
                        slot_ofs[(j, a["name"])] = window_slot_index(k, v01s[j].shape[2], d, stride, depth)
                    va = v01s[j][:, :k]
                    ws.append(make_windows(va, stride, depth) if var is None
                              else variant_windows(va, var, stride, depth)[0])
            for f in range(len(dev_arms[0][i])):
                for j in range(len(starts)):
                    d = devices[j]
                    on_gpu = d.type == "cuda"
                    with torch.cuda.device(d) if on_gpu else contextlib.nullcontext(), \
                            torch.autocast("cuda", enabled=on_gpu, dtype=torch.float16):
                        p = torch.sigmoid(dev_arms[j][i][f](ws[j], slot_ofs[(j, a["name"])],
                                                            mks[j][:, :k]).float())
                    acc[j][i] = p if acc[j][i] is None else acc[j][i] + p
    evs = []
    for j in range(len(starts)):
        d = devices[j]
        with torch.cuda.device(d) if d.type == "cuda" else contextlib.nullcontext():
            for i in range(len(arms)):
                acc[j][i] = acc[j][i] / (len(dev_arms[j][i]) * len(variants))
            ev = None
            if block and d.type == "cuda":
                ev = torch.cuda.Event(blocking=True)
                ev.record()
        evs.append(ev)
    out: list[list[torch.Tensor]] = [[] for _ in arms]
    for j, ev in enumerate(evs):
        if ev is not None:
            ev.synchronize()        # sleeps instead of spinning (see `predict(gpu_wait)`)
        for i in range(len(arms)):
            out[i].append(acc[j][i].cpu())
    empty = (~mask.any(dim=1)).numpy()
    res = []
    for chunks in out:
        b = torch.cat(chunks).numpy()
        b[empty] = np.nan
        res.append(b)
    return res


def fold_probs(models: list, w, slot_of, mask, weights=None) -> torch.Tensor:
    """(F, B, T) sigmoids of the folds. With `weights=(live, banks)` (from `share_encoder`) the weights of
    the fold are copied into the shared encoder before each forward.
    """
    out = []
    for k, m in enumerate(models):
        if weights is not None:
            load_fold_weights(weights[0], weights[1][k])
        out.append(torch.sigmoid(m(w, slot_of, mask).float()))
    return torch.stack(out)


# The artifacts are saved compressed: 52 MB -> 2.2 MB with lzma (0.24 s to decompress), and reading
# 52 MB from the Kaggle mount cost 5 s.
_XZ_MAGIC = b"\xfd7zXZ\x00"


def load_compile_cache(path: Path, log=print, data: bytes | None = None) -> bool:
    """Loads inductor's artifacts (`torch.compiler.load_cache_artifacts`).

    Compiling 5 DenseNet121 folds on a T4 costs ~85 s cold and ~9 s when inductor's disk cache is already
    populated. The submission kernel always starts from an empty machine: the cache travels as a file,
    produced by a kernel with the same image and the same GPU. If it does not match (another torch/triton
    version, another GPU) inductor simply does not use it and recompiles: slower, not different.
    """
    try:
        data = Path(path).read_bytes() if data is None else data
        if data[:6] == _XZ_MAGIC:
            data = lzma.decompress(data)
        info = torch.compiler.load_cache_artifacts(data)
    except Exception as exc:
        log(f"compilation cache not loaded from {path}: {type(exc).__name__}: {exc}")
        return False
    keys = {k: len(v) for k, v in getattr(info, "artifacts", {}).items()} if info else None
    log(f"compilation cache: {len(data) / 2**20:.1f} MiB from {path} | entries {keys}")
    return info is not None


def save_compile_cache(path: Path, log=print) -> int:
    """Writes the compilation artifacts of this process. Returns the bytes written."""
    try:
        res = torch.compiler.save_cache_artifacts()
    except Exception as exc:
        log(f"compilation cache not saved: {type(exc).__name__}: {exc}")
        return 0
    if res is None:
        log("compilation cache: nothing to save")
        return 0
    data, info = res
    packed = lzma.compress(data, preset=6)
    Path(path).write_bytes(packed)
    keys = {k: len(v) for k, v in getattr(info, "artifacts", {}).items()}
    log(f"compilation cache: {len(data) / 2**20:.1f} MiB -> {len(packed) / 2**20:.1f} MiB "
        f"in {path} | entries {keys}")
    return len(packed)


def _pad_batch(vol: torch.Tensor, mask: torch.Tensor, batch_size: int):
    """The last batch padded to `batch_size` with empty studies: fixed shapes for the encoder.

    The added studies have an all-false mask, so they come out like the neutral rows and are discarded by
    the caller anyway, which cuts the output to `n`.
    """
    n = vol.shape[0]
    if n == batch_size:
        return vol, mask, n
    pad = batch_size - n
    vol = torch.cat([vol, vol.new_zeros((pad, *vol.shape[1:]))])
    mask = torch.cat([mask, mask.new_zeros((pad, *mask.shape[1:]))])
    return vol, mask, n


def _probs_tta(models: list, vol01, mask, slot_of, stride: int, depth: int, weights,
               tta: list[Variant]) -> torch.Tensor:
    """(F, B, T): the mean over the TTA variants of the fold sigmoids. Every variant is one more pass over the
    same transformed windows (`knee.tta`); the mean of the probabilities is the fixed combination.
    `slot_of` is the model's and holds for every variant with the same number of windows (the only ones
    allowed here: a compiled kernel has fixed shapes).
    """
    acc = None
    for v in tta:
        if not v.same_windows:
            raise ValueError(f"TTA {v.name}: changes the number of windows, not valid in a submission")
        w, _ = variant_windows(vol01, v, stride, depth)
        p = fold_probs(models, w, slot_of, mask, weights)
        acc = p if acc is None else acc + p
    return acc / len(tta)


AOTI_CHECK_TOL = 0.05     # maximum |feature| difference between package and fp16 eager at load time


def _attach_study_packages(dev_models: list, devices: list, runners: list, n_win: int,
                           depth: int, img: int, log=print) -> tuple[bool, float | None]:
    """Puts the AOTI packages in place of the encoders, device by device, and compares them with eager on a
    random input with the shape of one study. At the first difference above `AOTI_CHECK_TOL`, or error,
    it restores all eager encoders and returns False.
    """
    worst = 0.0
    try:
        for ms, d, rs in zip(dev_models, devices, runners):
            for m, r in zip(ms, rs):
                eager = m.encoder
                m.encoder = _PackagedEncoder(r, eager)
                m.encode_chunk = int(n_win)
            with torch.cuda.device(d), torch.inference_mode():
                g = torch.Generator(device=d).manual_seed(0)
                x = torch.rand((n_win, depth, img, img), generator=g, device=d,
                               dtype=torch.float16)
                for m in ms:
                    with torch.autocast("cuda", dtype=torch.float16):
                        ref = m.encoder._orig_mod(x).float()
                    got = m.encoder(x).float()
                    worst = max(worst, float((got - ref).abs().max()))
            torch.cuda.synchronize(d)
        if not worst <= AOTI_CHECK_TOL:
            raise RuntimeError(f"difference {worst:.4f} above {AOTI_CHECK_TOL}")
    except Exception as exc:
        log(f"!! single-study AOTI packages discarded, staying eager: {type(exc).__name__}: "
            f"{str(exc)[:300]}")
        for ms in dev_models:
            for m in ms:
                if isinstance(m.encoder, _PackagedEncoder):
                    m.encoder = m.encoder._orig_mod
                m.encode_chunk = 0
        return False, round(worst, 5)
    log(f"encoders from single-study AOTI packages on {len(devices)} GPUs "
        f"({sum(len(ms) for ms in dev_models)} models), check difference {worst:.4f}")
    return True, round(worst, 5)


class _TrtEncoderModule(torch.nn.Module):
    """The encoder from a TensorRT engine (`knee.trt.TrtEncoder`) on one GPU; `_orig_mod` is the eager
    encoder. The features come out in fp16, in a new tensor: the engine buffer is reused on the next call.
    """

    def __init__(self, runner, orig: torch.nn.Module) -> None:
        super().__init__()
        self.runner = runner
        self._orig_mod = orig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.device(x.device):
            return self.runner(x).to(torch.float16, copy=True)


def _attach_trt(dev_models: list, devices: list, blobs: dict, fold_ids: list[int],
                n_total: int, n_win: int, depth: int, img: int, log=print,
                graphs: bool = False, max_inflight: int | None = None) -> dict:
    """Puts the TensorRT engines (one per fold, `blobs[fold]`) in place of the encoders on every GPU, one
    GPU per thread. Load-time check: shape and finite values on a random study (INT8 does not match eager;
    the difference is measured on the predictions in the benchmark). An error restores all eager encoders.

    `graphs`: for every engine the CUDA graphs of the full batch (`n_total` images, `capture`), checked
    against plain launches on the first block of `n_win` images: the difference must stay within that
    between two rounds of plain launches (a tactic with atomic sums may not repeat bit for bit) or 1e-3,
    otherwise the engine stays without a graph. The short final batch uses plain launches. Checking on the
    whole batch, repeated four times per engine with the GPUs in series, made attaching cost ~9 s per run.

    `max_inflight`: queued blocks per engine (`TrtEncoder`), then the thread sleeps.

    `blobs[fold]` can also be a `knee.dense8` / `knee.dense8s` package (dict from `load_pack`): then the
    encoder is a `Dense8Encoder` / `Dense8SEncoder` (our INT8 kernels, TensorRT only where needed), same
    interface. This is the path of the final submission.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .trt import TrtEncoder
    t = time.perf_counter()

    def _wait(d) -> None:
        # Wait for the GPU with `synchronize` (releases the GIL), then `bool`/`float` are immediate. An
        # `.item()` with a full queue waits holding the GIL and blocks the other GPU's thread (~4 s of the GPU 1
        # thread stuck during attaching).
        torch.cuda.current_stream(d).synchronize()

    def one(ms: list, d) -> dict:
        info = {"dense8_load_s": [], "captured": 0, "failed": 0, "capture_s": 0.0,
                "max_abs": 0.0, "max_abs_plain": 0.0, "errors": [], "phases": {}}
        di = d.index if d.index is not None else torch.cuda.current_device()
        ph, tq = info["phases"], [time.perf_counter()]

        def lap(name: str) -> None:
            now = time.perf_counter()
            ph[name] = round(ph.get(name, 0.0) + now - tq[0], 3)
            tq[0] = now

        with torch.cuda.device(d), torch.inference_mode():
            for m, f in zip(ms, fold_ids):
                if isinstance(blobs[f], dict):
                    # dense8 or dense8s package (the per-block store, `meta["kind"]`)
                    if blobs[f].get("meta", {}).get("kind") == "dense8s":
                        from .dense8s import Dense8SEncoder as Enc
                    else:
                        from .dense8 import Dense8Encoder as Enc
                    runner = Enc(blobs[f], di, n_total, max_inflight=max_inflight)
                    info["dense8_load_s"].append(round(runner.load_s, 2))
                else:
                    runner = TrtEncoder(blobs[f], di, n_total, max_inflight=max_inflight)
                m.encoder = _TrtEncoderModule(runner, m.encoder)
                m.encode_chunk = 0
            lap("load")
            x = torch.rand((n_win, depth, img, img), device=d, dtype=torch.float16)
            slot_w = torch.zeros(n_win, device=d, dtype=torch.long)
            mask_w = torch.ones((1, 4), device=d, dtype=torch.bool)
            idx_w = torch.arange(n_win, device=d, dtype=torch.long)
            for m in ms:
                y = m.encoder(x)
                fin = torch.isfinite(y).all()
                _wait(d)
                lap("first")
                if y.shape[0] != n_win or not bool(fin):
                    raise RuntimeError(f"output {tuple(y.shape)} or not finite on {d}")
                try:
                    with torch.autocast("cuda", dtype=torch.float16):
                        kw = {"window_index": idx_w} if getattr(m, "context_pos", None) is not None else {}
                        _ = m(None, slot_w, mask_w, encoded=y.view(1, n_win, -1), **kw)
                except Exception:
                    pass
                _wait(d)
                lap("heads")
            if os.environ.get("KNEE_FUSED_UINT8", "off") == "on":
                from .data.fused_uint8 import warmup as _f_warmup
                info["uint8_warmup"] = _f_warmup(d)
            torch.cuda.synchronize(d)
            lap("uint8")
            if not graphs:
                return info
            xb = torch.rand((n_total, depth, img, img), device=d, dtype=torch.float16)
            for m in ms:
                r = m.encoder.runner
                try:
                    ref = r(xb[:n_win]).clone()
                    dp = (r(xb[:n_win]) - ref).abs().max()
                    _wait(d)
                    plain = float(dp)
                    lap("eager")
                    info["capture_s"] += r.capture(n_total)
                    lap("capture")
                    # Only the first graph is replayed (`r` with fewer images than the captured batch uses the first
                    # ceil(n / nb) graphs, here one): same check, 180 images instead of 720. The other graphs are the same
                    # capture on other addresses.
                    dg = (r(xb[:n_win]) - ref).abs().max()
                    _wait(d)
                    got = float(dg)
                    lap("replay")
                    info["max_abs"] = max(info["max_abs"], got)
                    info["max_abs_plain"] = max(info["max_abs_plain"], plain)
                    if not got <= max(1e-3, 2 * plain):
                        raise RuntimeError(f"graph differs from plain launches: max|d| "
                                           f"{got:.3g} (between two plain rounds {plain:.3g})")
                    info["captured"] += 1
                except Exception as exc:
                    r.graphs.clear()
                    info["failed"] += 1
                    info["errors"].append(repr(exc)[:300])
                    log(f"!! CUDA graph of the engine on {d} not recorded, plain launches: "
                        f"{type(exc).__name__}: {str(exc)[:300]}")
            del xb
            torch.cuda.synchronize(d)
        return info

    try:
        with ThreadPoolExecutor(max_workers=max(1, len(devices))) as ex:
            parts = list(ex.map(lambda a: one(*a), zip(dev_models, devices)))
    except Exception as exc:
        log(f"!! TensorRT engines discarded, staying eager: {type(exc).__name__}: "
            f"{str(exc)[:300]}")
        for ms in dev_models:
            for m in ms:
                if isinstance(m.encoder, _TrtEncoderModule):
                    m.encoder = m.encoder._orig_mod
        return {"ok": False, "error": repr(exc)[:300]}
    info: dict = {}
    if any("uint8_warmup" in p for p in parts):
        info["uint8_warmup"] = [p["uint8_warmup"] for p in parts if "uint8_warmup" in p]
    loads = [v for p in parts for v in p["dense8_load_s"]]
    if loads:
        info["dense8_load_s"] = loads
    info["phases"] = [p.get("phases", {}) for p in parts]
    if graphs:
        info["graphs"] = {"captured": sum(p["captured"] for p in parts),
                          "failed": sum(p["failed"] for p in parts),
                          "capture_s": round(sum(p["capture_s"] for p in parts), 2),
                          "max_abs": max(p["max_abs"] for p in parts),
                          "max_abs_plain": max(p["max_abs_plain"] for p in parts)}
        errs = [e for p in parts for e in p["errors"]]
        if errs:
            info["graphs"]["error"] = errs[0]
    log(f"encoders from TensorRT engines on {len(devices)} GPUs "
        f"({sum(len(ms) for ms in dev_models)} models) in {time.perf_counter() - t:.2f}s"
        + (f", graphs {info['graphs']}" if graphs else "") + f", phases {info['phases']}")
    return {"ok": True, "load_s": round(time.perf_counter() - t, 2), **info}


@_infer_ctx()
def predict(models: list, ds: DicomStudies, device, batch_size: int = 4, workers: int = 4,
            amp: bool = True, time_budget_s: float | None = None, log=print,
            log_every: int = 100, compile: bool = True,
            packaged: bool = False,
            tta: list[Variant] | None = None,
            gpu_wait: str = "block", cudnn_benchmark: bool = False,
            channels_last: bool | None = None,
            fold_bn: bool | None = None, gpus: int | None = None,
            main_threads: int | None = None, aoti: Path | None = None,
            fold_ids: list[int] | None = None, trt: dict | None = None,
            gpu_threads: bool | None = None,
            trt_graphs: bool = False,
            trt_inflight: int | None = None,
            early_workers: bool = False, skip_empty: bool = False,
            budget: str | None = None, fastdense: bool = False) -> tuple[np.ndarray, dict]:
    """Probabilities averaged over the folds. Rows not computed stay NaN, on purpose.

    `fastdense`: the DenseNet encoder with the `knee.fastdense` kernels in eval (`attach_eval`,
    `CudaOps`): ~2.2x eager on the encoder, outputs at the fp16 rounding level. Eager only (no compile,
    AOTI packages, TensorRT); turns off `fold_bn` (which folds norm2 into conv1, and fastdense needs it).

    `packaged=True`: the encoders are already AOTI packages (`load_aoti_encoders`), which want
    fixed-shape batches like the compiled ones; nothing to compile.

    `gpu_wait`: how the main process waits for the end of every batch on the GPU. `spin` is the plain
    `.cpu()`, which busy-waits: `main_cpu_s` 427 s against `gpu_s` 432 s, a full core at nice 0 taken from
    the decoding workers at nice 10, which cost +17% to decoding. `block` waits on a
    `torch.cuda.Event(blocking=True)`, which puts the thread to sleep until the GPU is done; the `.cpu()`
    afterwards finds the result ready. Only how the wait happens changes, not what is computed. Default
    `block`: -6.4% on the steady state in an A/B, identical predictions.

    `cudnn_benchmark`: in eager, whether cudnn tries the algorithms of every convolution on the first
    batch of every shape (`True`) or uses its heuristics (`False`). Autotuning was most of the ~39 s of
    the first batch, and repeats on the last, partial batch; in steady state the GPU has ~16% headroom
    over decoding. Off anyway with the compiled encoder. Predictions can change at the level of cudnn
    noise between two runs. Default `False`: cold first batch 13.8 s against ~39 s, unchanged steady
    state.

    `channels_last`: 4D weights in NHWC (`train.speed.channels_last` in training). With NHWC weights
    cudnn also brings the input of the first convolution to NHWC, and from there the activations stay
    NHWC (the DenseNet `cat`s too); in fp16 on the T4 the tensor-core kernels are native NHWC, in NCHW
    cudnn transposes inside. It changes the layout, not the network: predictions move at the level of
    cudnn noise. Before `share_encoder` and compilation (the weight banks clone the layout); not with
    AOTI packages, which have their own. `None` (default) = on in eager (`compile=False`), off when
    compiling: -12.2% on the steady state and -7.7% on the GPU; with inductor NHWC is already chosen by
    inductor itself. The eager fallback of a failed compiled encoder stays NCHW.

    `fold_bn`: eager only (`compile=False`, no AOTI packages), the BatchNorm between the two convolutions
    of every layer of the hand-written DenseNet folded into the first, and convolution + bias + ReLU in a
    single cudnn operator (`models.densenet.fold_bn`): 82 fewer BatchNorm and 82 fewer ReLU passes per
    image for the 169. Predictions move at the level of the fp16 rounding of the folded weights. If the
    fused operator is missing or does not match on the first batch, the three operators stay
    (`fold_bn_status`). `None` (default) = on in eager: -6.3% on the steady state and -9.6% on the GPU,
    Spearman >= 0.999995.

    `gpus`: how many GPUs to use, eager fp16 only. Kaggle's `NvidiaTeslaT4` machine has two T4s. With
    `gpus=2` the models are copied to `cuda:1` and every iteration takes `batch_size * gpus` studies:
    `batch_size` per GPU, the same batch shape as a single GPU, all parts queued before waiting
    (`_forward_split`). Predictions are those of a single GPU. `None` (default) = all visible GPUs in
    eager fp16 on CUDA, otherwise one: GPU 0.167 against 0.298 s/study, steady state -13.1% (-54 s over
    1,300 studies).

    `main_threads`: torch's OpenMP threads in the main process during the pass (`torch.set_num_threads`,
    restored at the end); `None` = torch's own, i.e. one per vCPU. In the main process the CPU only does
    copies (pin_memory, masks): with 4 vCPUs already busy with workers, a thread pool spinning after every
    copy takes CPU from decoding. DataLoader workers already have a single one (torch enforces it).

    `aoti`: the folder of the single-study AOTI packages (`study_aoti_meta`; `fold_ids` = the fold of
    each model, default 0..n-1). If they match, after the replicas every model on every GPU gets the
    package of its fold loaded on that device and the encoder sees one study per call (`encode_chunk`);
    `fold_bn` and `channels_last` are not needed (the encoder is the package). Load-time self-check
    against eager on every device: a difference above the threshold or an error restores eager. 1.30x
    the `fold_bn` + NHWC eager, max|dp| 1.5e-4, loading 0.05 s.

    `trt`: the TensorRT engines of the encoder, {fold: bytes} (`knee.trt`, explicit INT8 quantization from
    `knee.quant.quantized_copy`, one study), or `knee.dense8`/`knee.dense8s` packages, in place of the eager
    encoders on every GPU (`_attach_trt`); eager fp16 only. The engine call blocks the thread for almost
    all of the GPU time, so from a single thread the two T4s run in series (1.09x); with one thread per
    GPU 1.99x.
    `gpu_threads`: every GPU computes its part of the batch in its own thread (`_forward_split(pool=...)`);
    `None` = on with `trt`, off otherwise.
    `trt_graphs`: the engines run from CUDA graphs recorded at load time on the full batch
    (`_attach_trt(graphs=True)`): one launch per batch instead of hundreds of kernels, which filled the
    launch queue and made the thread spin (296 s of CPU over 1,300 studies, even with blocking sync). Not
    enough on its own: the graph launch goes through the same queue.
    `trt_inflight`: at most this many blocks queued per engine, then the thread sleeps on an event
    (`TrtEncoder(max_inflight=...)`). With engines from the chained export (`knee.quant.ChainDenseNet`, 433
    kernels per call instead of 1,589) and 2 blocks, two seeds on the two T4s cost 0.008 s/study of CPU
    instead of 0.29.
    `budget`: the per-member window design (`_forward_split`), two GPUs only; the engines record the
    graphs on the member's batch (studies per GPU x windows of the largest member).
    """
    if gpu_wait not in ("spin", "block"):
        raise ValueError(f"gpu_wait={gpu_wait!r}: 'spin' or 'block'")
    eligible = device.type == "cuda" and amp and not compile and not packaged
    _prepare_deferred_encoders(models, trt, eligible, log)
    if gpus is None:
        gpus = max(torch.cuda.device_count(), 1) if eligible else 1
    if gpus > 1 and not eligible:
        log(f"!! gpus={gpus} applies only to eager fp16 on CUDA: staying on one GPU")
        gpus = 1
    if gpus > 1 and torch.cuda.device_count() < gpus:
        log(f"!! gpus={gpus} but {torch.cuda.device_count()} GPUs visible: using those")
        gpus = max(torch.cuda.device_count(), 1)
    stride, depth = models[0].window_stride, models[0].window_depth
    n_win_study = (ds.spec.shape[0] * windows_per_slot(ds.spec.slices, stride, depth)
                   if hasattr(ds, "spec") else None)
    aoti_runners, aoti_info = None, {"requested": aoti is not None}
    if aoti is not None and eligible and n_win_study:
        t_a = time.perf_counter()
        ids = list(fold_ids) if fold_ids is not None else list(range(len(models)))
        devs = [device] + [torch.device("cuda", j) for j in range(1, gpus)]
        aoti_runners = load_study_packages(
            Path(aoti), ids, devs, study_aoti_meta(n_win_study, depth, ds.spec.img), log)
        aoti_info.update(fold_ids=ids, load_s=round(time.perf_counter() - t_a, 2))
    if aoti_runners is not None or (trt is not None and eligible):
        fold_bn, channels_last = False, False
    # The copies for the other GPUs before folding, changing layout and dtype: outside inference_mode they
    # are born with normal parameters, then receive the same transformations.
    replicas = []
    if gpus > 1:
        with torch.inference_mode(False):
            replicas = [[copy.deepcopy(m) for m in models] for _ in range(gpus - 1)]
    compiled, weights = False, None
    if fastdense and (compile or packaged or aoti_runners is not None or trt is not None):
        log("!! fastdense in eval applies only to eager: off")
        fastdense = False
    if fastdense:
        fold_bn = False
    if fold_bn is None:
        fold_bn = not compile and not packaged
    fold_bn = bool(fold_bn and not compile and not packaged)
    n_folded = sum(fold_densenet_bn(m) for m in models) if fold_bn else 0
    if channels_last is None:
        channels_last = not compile and not packaged
    # 3D encoders: NHWC is a 4D layout, the weights of a video encoder are 5D.
    channels_last = bool(channels_last and channels_last_ok(models))
    if channels_last and not packaged:
        models = [m.to(memory_format=torch.channels_last) for m in models]
    if device.type == "cuda":
        if amp:
            models = [m.half() for m in models]
        if fastdense:
            n_fast = sum(attach_eval(m, log) for m in models)
            log(f"fastdense in eval: {n_fast}/{len(models)} encoders")
        if packaged:
            compiled = True
        elif compile:
            if len(models) > 1 and len({type(m.encoder) for m in models}) == 1:
                weights = share_encoder(models)
            compiled = compile_encoders(models, log)
        # cudnn autotuning over the ~120 distinct convs costs ~7 s on the first batch and, with the compiled
        # encoder, gains nothing in steady state (2.43 s/batch without against 2.50 with). In eager it stays on:
        # there it is the only kernel choice.
        torch.backends.cudnn.benchmark = (not compiled) and cudnn_benchmark
    devices = [device] + [torch.device("cuda", j) for j in range(1, gpus)]
    dev_models = [models]
    for d, rep in zip(devices[1:], replicas):
        if fold_bn:
            for m in rep:
                fold_densenet_bn(m)
        rep = [m.to(d) for m in rep]
        if channels_last:
            rep = [m.to(memory_format=torch.channels_last) for m in rep]
        rep = [m.half() for m in rep]
        if fastdense:
            for m in rep:
                attach_eval(m, log)
        dev_models.append(rep)
    if any(isinstance(m.encoder, _DeferredEncoder) for ms in dev_models for m in ms):
        _log_head_only([m for ms in dev_models for m in ms], log, "before-attach")
    trt_info = {"requested": trt is not None}
    # `early_workers`: the DataLoader workers start before the engines are attached, so the first studies
    # are decoded while the GPU loads the kernels and records the graphs (~9 s per run of attaching, in
    # series with decoding otherwise).
    float_input = not getattr(ds, "to_uint8", True)
    raw_input = bool(getattr(ds, "raw", False))
    if raw_input and gpus < 2:
        raise ValueError("raw crops (gpu_resize): only on two GPUs (`_forward_split`)")
    # Self-check on the first study of `KNEE_RAW_CHECKS` pieces (1 by default: every check redoes the study
    # with the CPU path in the main process, ~12 s at start-up with the CPU saturated).
    _def_chk = "0" if os.environ.get("KNEE_LEAN", "off") == "on" else "1"
    rawchk = {"left": int(os.environ.get("KNEE_RAW_CHECKS", _def_chk)), "ok": True, "checked": 0, "pixels": 0,
              "diff": 0, "worst_abs": 0,
              "max_frac": 1e-3 if os.environ.get("KNEE_PCT_STRIDE", "1") == "1" else 1.0, "max_abs": 3, "series": 0, "cpu_series": 0, "prep_s": 0.0}
    # Read-ahead of the DICOMs into the page cache (`knee.data.readahead`): the NFS mount is bound by
    # per-file latency, 32 threads ahead of the workers. `KNEE_READAHEAD` = threads, 0 = off.
    _ra_prog, _ra_stats = [0], None
    if int(os.environ.get("KNEE_READAHEAD", "0")) > 0 and hasattr(ds, "slots"):
        from .data import readahead as _readahead
        _ra_stats = _readahead.start(ds.root, ds.series_dir, ds.uids, ds.slots, _ra_prog,
                                     threads=int(os.environ["KNEE_READAHEAD"]),
                                     ahead=int(os.environ.get("KNEE_READAHEAD_AHEAD", "48")), log=log)
    loader = batches = None
    # The decoding pool started by the kernel before torch (`knee.data.early`) for the same studies: its
    # batches replace the DataLoader (no workers to spawn, no pin_memory).
    early_pool = None
    if raw_input:
        from .data import early as _early_mod
        early_pool = _early_mod.take(ds)
    if early_pool is not None:
        batches = early_pool.batches(batch_size * gpus)
        log(f"early: {len(early_pool.procs)} workers started {time.time() - early_pool.t_start:.1f} s ago")
    elif early_workers and trt is not None and not compiled:
        loader = _make_loader(ds, batch_size * gpus, workers, device,
                              prefetch=2 if float_input else None)
        batches = iter(loader)
    if budget is not None and gpus < 2:
        raise ValueError("budget: only on two GPUs (`_forward_split`)")
    n_member = n_win_study
    if budget is not None and n_win_study:
        per_slot = windows_per_slot(ds.spec.slices, stride, depth)
        nv = len(tta or [None])
        sizes = {(f, v): int(member_windows(budget, v, ds.spec.shape[0], per_slot, seed=f).numel())
                 for f in range(len(models)) for v in range(nv)}
        # with `KNEE_TTA_BATCH_VIEWS` the graph receives the windows of all views of a seed in one call
        n_member = (max(sum(sizes[(f, v)] for v in range(nv)) for f in range(len(models)))
                    if _BATCH_VIEWS[0] and nv > 1 else max(sizes.values()))
    if trt is not None and eligible and n_win_study:
        ids = list(fold_ids) if fold_ids is not None else list(range(len(models)))
        trt_info.update(fold_ids=ids, **_attach_trt(dev_models, devices, trt, ids,
                                                    batch_size * n_member, n_win_study,
                                                    depth, ds.spec.img, log,
                                                    graphs=trt_graphs,
                                                    max_inflight=trt_inflight))
    if gpu_threads is None:
        gpu_threads = bool(trt_info.get("ok"))
    gpu_pool = None
    if gpu_threads and gpus > 1:
        from concurrent.futures import ThreadPoolExecutor
        gpu_pool = ThreadPoolExecutor(max_workers=gpus)
    if aoti_runners is not None:
        aoti_info["ok"], aoti_info["check_max_abs"] = _attach_study_packages(
            dev_models, devices, aoti_runners, n_win_study, depth, ds.spec.img, log)
    if gpus > 1:
        log(f"{gpus} GPUs: {[torch.cuda.get_device_name(d) for d in devices]} | "
            f"{batch_size} studies per GPU per batch")

    # float32 stacks from the worker (`DicomStudies(to_uint8=False)`): percentiles and uint8 here.
    u8 = {"left": 2, "ok": True, "checked": 0}
    if loader is None and early_pool is None:
        loader = _make_loader(ds, batch_size * gpus, workers, device,
                              prefetch=2 if float_input else None)
        batches = loader
    wait_ev = (torch.cuda.Event(blocking=True)
               if device.type == "cuda" and gpu_wait == "block" else None)

    probs = np.full((len(ds), len(TARGETS)), np.nan, dtype=np.float32)
    t0, decode_s, gpu_s, n_empty, n_err, done = time.time(), 0.0, 0.0, 0, 0, 0
    first_batches: list[float] = []     # wall time of the first batches: compilation and start-up
    skip_stats: dict = {}                # valid and computed windows (`skip_empty`)

    # The folds of a run share the cfg, hence also the window stride (computed at the top, before the AOTI
    # packages that depend on it).
    # Static tensor computed outside the batch loop if the dataset exposes the spec
    slot_of = (window_slot_index(ds.spec.shape[0], ds.spec.shape[1], device, stride, depth)
               if hasattr(ds, "spec") else None)
    slot_ofs = [slot_of] + [None] * (gpus - 1)

    tracing = bool(os.environ.get(TRACE_ENV))
    trace = {"batches": []} if tracing else None
    sampler = _CpuSampler().start() if tracing else None
    b0 = time.perf_counter()
    if compiled and device.type == "cuda" and amp and hasattr(ds, "spec"):
        # Compile on a batch of zeros **before** starting the workers: compilation is pure Python in the main
        # process, and with the 4 workers decoding on the same 4 vCPUs it cost 24.7 s instead of 14.5 (`nice`
        # was not enough). The first studies then arrive in ~3 s thanks to the micro-batches of `_make_loader`.
        # Same shapes and dtypes as the real batch, so the compiled graph is the right one. The output (NaN:
        # all-false mask) is discarded.
        z = torch.zeros((batch_size, *ds.spec.shape), dtype=torch.uint8, device=device)
        zm = torch.zeros((batch_size, ds.spec.shape[0]), dtype=torch.bool, device=device)
        laps = {}
        try:
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                w = make_windows(z.half().mul_(1.0 / 255.0), stride, depth)
                # First the GPU alone (context, cudnn, torch's kernels): a start-up cost that exists in eager too, and
                # must be kept apart from compilation.
                t = time.perf_counter()
                enc = getattr(models[0].encoder, "_orig_mod", models[0].encoder)
                enc(w.flatten(0, 1)[:8])
                torch.cuda.synchronize()
                laps["gpu_init"] = round(time.perf_counter() - t, 1)
                t = time.perf_counter()
                fold_probs(models, w, slot_of, zm, weights)
                torch.cuda.synchronize()
                laps["compile"] = round(time.perf_counter() - t, 1)
                t = time.perf_counter()
                fold_probs(models, w, slot_of, zm, weights)
                torch.cuda.synchronize()
                laps["forward"] = round(time.perf_counter() - t, 2)
        except Exception as exc:
            log(f"!! compiled encoder failed during warm-up, staying eager: "
                f"{type(exc).__name__}: {str(exc)[:300]}")
            uncompile_encoders(models)
            compiled = False
            torch.backends.cudnn.benchmark = cudnn_benchmark
        del z, zm
        laps["total"] = round(time.perf_counter() - b0, 1)
        log(f"encoder warm-up: {laps}")
    if trace is not None:
        trace["t_iter"] = round(time.time(), 3)     # the workers start here
    prev_threads = torch.get_num_threads()
    if main_threads:
        torch.set_num_threads(int(main_threads))
    th0 = _thread_cpu()
    stream_in = _regroup(batches, batch_size * gpus, cat_vol=gpus == 1)
    if raw_input:
        # Raw payloads: the volumes are prepared one batch ahead, overlapping the model.
        stream_in = _prep_raw_ahead(stream_in, ds.spec, devices, batch_size, rawchk, log=log)
    for vol, mask, idx, dt, errs in stream_in:
        _ra_prog[0] = max(_ra_prog[0], int(idx.max()) + 1)
        if trace is not None:
            # [batch arrival, studies already done, studies in the batch, end on the GPU]
            trace["batches"].append([round(time.time(), 3), done, len(idx)])
        decode_s += float(dt.sum())
        n_err += int(errs.sum())
        n_empty += int((~mask.any(dim=1)).sum())
        g0 = time.perf_counter()
        if gpus > 1:
            probs[idx.numpy()] = _forward_split(dev_models, devices, slot_ofs, vol, mask,
                                                batch_size, stride, depth, tta,
                                                block=gpu_wait == "block", u8=u8, log=log,
                                                pool=gpu_pool, skip_empty=skip_empty,
                                                skip_stats=skip_stats, budget=budget,
                                                raw_spec=ds.spec if raw_input else None,
                                                rawchk=rawchk)
            gpu_s += time.perf_counter() - g0
            done += len(idx)
            if trace is not None:
                trace["batches"][-1].append(round(time.time(), 3))
            if len(first_batches) < 4:
                first_batches.append(round(time.perf_counter() - b0, 1))
                b0 = time.perf_counter()
            if done % log_every < len(idx):
                rate = done / max(time.time() - t0, 1e-6)
                log(f"  {done}/{len(ds)} | {rate:.2f} studies/s | ~"
                    f"{(len(ds) - done) / rate / 60:.0f} min left")
            if time_budget_s and time.time() - t0 > time_budget_s:
                log(f"!! budget exhausted at {done}/{len(ds)} studies: the rest stays neutral")
                break
            continue
        if compiled:
            vol, mask, n_real = _pad_batch(vol, mask, batch_size)
        else:
            n_real = vol.shape[0]
        first_cpu = vol[0] if vol.dtype == torch.float32 else None
        vol = vol.to(device, non_blocking=True)
        if vol.dtype == torch.float32:
            vol = _uint8_on_device(vol, u8, first_cpu=first_cpu, log=log)
        mask = mask.to(device, non_blocking=True)
        if slot_of is None:
            slot_of = window_slot_index(vol.shape[1], vol.shape[2], device, stride, depth)
        if device.type == "cuda" and amp:
            vol01 = vol.half().mul_(1.0 / 255.0)
            w = make_windows(vol01, stride, depth)
            p = None
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                try:
                    p = (fold_probs(models, w, slot_of, mask, weights) if not tta
                         else _probs_tta(models, vol01, mask, slot_of, stride, depth, weights, tta))
                except Exception as exc:
                    if not compiled:
                        raise
                    log(f"!! compiled encoder failed on the first forward, staying eager: "
                        f"{type(exc).__name__}: {str(exc)[:300]}")
                    uncompile_encoders(models)
                    compiled = False
                    torch.backends.cudnn.benchmark = cudnn_benchmark
            if p is None:
                # Retry outside the `except`, as in `predict_arms`: see the comment there.
                _free_cuda()
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    p = (fold_probs(models, w, slot_of, mask, weights) if not tta
                         else _probs_tta(models, vol01, mask, slot_of, stride, depth, weights, tta))
        else:
            vol01 = vol.float().div_(255.0)
            w = make_windows(vol01, stride, depth)
            p = (fold_probs(models, w, slot_of, mask, weights) if not tta
                 else _probs_tta(models, vol01, mask, slot_of, stride, depth, weights, tta))
        pm = p.mean(0)[:n_real]
        if wait_ev is not None:
            wait_ev.record()
            wait_ev.synchronize()       # sleeps instead of spinning (see `gpu_wait`)
        batch = pm.cpu().numpy()
        mask = mask[:n_real]
        # A study without a single valid slot has no images: the model still returns a number (the head
        # bias), which looks like a prediction and is not. It becomes NaN here, so it comes out as 0.5 like
        # the rows not computed and enters the same count: fake rows must be counted, not plausible.
        batch[(~mask.any(dim=1)).cpu().numpy()] = np.nan
        probs[idx.numpy()] = batch
        gpu_s += time.perf_counter() - g0
        done += len(idx)
        if trace is not None:
            trace["batches"][-1].append(round(time.time(), 3))
        if len(first_batches) < 4:
            first_batches.append(round(time.perf_counter() - b0, 1))
            b0 = time.perf_counter()
        if done % log_every < len(idx):
            rate = done / max(time.time() - t0, 1e-6)
            log(f"  {done}/{len(ds)} | {rate:.2f} studies/s | ~"
                f"{(len(ds) - done) / rate / 60:.0f} min left")
        if time_budget_s and time.time() - t0 > time_budget_s:
            # A valid submission with a neutral tail is better than a kernel killed by the time limit, which
            # produces nothing. How many rows ended that way is reported in `runtime.json`: a long tail invalidates
            # the result instead of hiding it.
            log(f"!! budget exhausted at {done}/{len(ds)} studies: the rest stays neutral")
            break

    wall = time.time() - t0
    th1 = _thread_cpu()
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(prev_threads)
    named = {os.getpid(): "main"}
    pin = getattr(getattr(loader, "_iterator", None), "_pin_memory_thread", None)
    if getattr(pin, "native_id", None):
        named[pin.native_id] = "pin_memory"
    if trace is not None:
        trace["t_end"] = round(time.time(), 3)
        trace["cpu"] = sampler.stop()
        if getattr(sampler._th, "native_id", None):
            named[sampler._th.native_id] = "trace_sampler"
    stats = {
        "n_studies": len(ds), "n_predicted": done,
        "wall_s": round(wall, 1),
        "decode_cpu_s": round(decode_s, 1),
        "gpu_s": round(gpu_s, 1),
        "s_per_study": round(wall / max(done, 1), 3),
        "decode_share_of_cpu_work": round(decode_s / max(decode_s + gpu_s, 1e-9), 3),
        "studies_without_any_slot": n_empty,
        "slice_decode_failures": n_err,
        "compiled": compiled, "shared_encoder": weights is not None, "packaged": packaged,
        "first_batches_s": first_batches, "gpu_wait": gpu_wait,
        "skip_empty": {"on": bool(skip_empty), **skip_stats},
        "budget": budget,
        "cudnn_benchmark": cudnn_benchmark, "channels_last": channels_last,
        "fold_bn": fold_bn, "fold_bn_layers": n_folded,
        "fold_bn_status": (fold_bn_status([m for ms in dev_models for m in ms])
                           if fold_bn else None),
        "gpus": gpus, "gpu_uint8": float_input,
        "gpu_uint8_checked": u8["checked"] if float_input else None,
        "gpu_uint8_ok": u8["ok"] if float_input else None,
        "gpu_resize": raw_input,
        "gpu_resize_check": ({k: round(rawchk[k], 1) if k == "prep_s" else rawchk[k]
                              for k in ("ok", "checked", "pixels", "diff", "worst_abs", "series",
                                        "cpu_series", "prep_s")}
                             if raw_input else None),
        "main_threads": main_threads, "torch_threads": torch_threads,
        "aoti_study": aoti_info, "trt": trt_info,
        "gpu_threads": gpu_pool is not None,
        "thread_cpu_s": _thread_cpu_delta(th0, th1, named) if th1 else None,
    }
    if early_pool is not None:
        stats["early_pool"] = early_pool.stats
    from .data.fused_uint8 import status as _fused_status
    stats["uint8_fused"] = _fused_status()
    if _TTA_U8_CHECK["checked"] or _TTA_U8_CHECK["errors"]:
        stats["tta_u8_check"] = dict(_TTA_U8_CHECK)
    if trace is not None:
        stats["trace"] = trace
    if gpu_pool is not None:
        gpu_pool.shutdown(wait=True)
    return probs, stats


def write_submission(uids: list[str], probs: np.ndarray, out_path: Path) -> int:
    """The file, with exactly the rows of `test.csv` and in their order."""
    df = pd.DataFrame(probs, index=pd.Index(uids, name=ID), columns=TARGETS)
    n_missing = int(df.isna().any(axis=1).sum())
    df.fillna(NEUTRAL).to_csv(out_path)
    return n_missing


def preflight(dirs: dict[str, Path], n: int = 4, seed: int = 0) -> float:
    """Decodes one slice for each of n random series.

    A light preflight (4 series): answers right away whether pixels and decoders work, without wasting
    precious seconds of I/O on the test set.
    """
    rng = np.random.default_rng(seed)
    keys = list(dirs)
    pick = [keys[i] for i in rng.permutation(len(keys))[:min(n, len(keys))]]
    errs, tried = DecodeErrors(), 0
    for uid in pick:
        try:
            files = os.listdir(dirs[uid])
        except OSError:
            continue
        if not files:
            continue
        tried += 1
        read_slice(str(Path(dirs[uid]) / files[len(files) // 2]), errs)
    return len(errs) / max(tried, 1)


def _decode_jit(log=print) -> dict:
    """The fast decoding kernels loaded here, in the main process, before the workers are born (with fork
    they inherit them): the C library (`knee.data.cfast`, compiled with the system cc in ~0.5 s) and,
    only if that is missing, numba (`knee.data.jit`, 3.5 s of compilation on the T4). Without either,
    NumPy and Python are used (same pixels). With `cache._JIT` off nothing is loaded.
    """
    from .data import cache, cfast, jit
    if not cache._JIT[0]:
        return {"ok": False, "error": "off (cache._JIT)", "compile_s": 0.0}
    st = {"c": cfast.load()}
    if not st["c"]["ok"]:
        st["numba"] = jit.warmup()
    st["ok"] = bool(st["c"]["ok"] or st.get("numba", {}).get("ok"))
    log(f"fast decoding kernels: {st}")
    return st


def run_submission(root: Path, ckpt_dir: Path, out_dir: Path, batch_size: int = 8,
                   workers: int = 4, time_budget_s: float | None = None,
                   log=print, split: str = "test", limit: int | None = None,
                   offset: int = 0, compile: bool = True,
                   compile_cache_dir: Path | None = None,
                   folds: list[int] | None = None,
                   tta: list[Variant] | None = None,
                   stride: int | None = None, gpu_wait: str = "block",
                   cudnn_benchmark: bool = False, channels_last: bool | None = None,
                   fold_bn: bool | None = None, gpus: int | None = None,
                   gpu_uint8: bool = False, main_threads: int | None = None,
                   aoti_dir: Path | None = None, adabn: int | None = None,
                   uids: list[str] | None = None, amp: bool = True,
                   trt_dir: Path | None = None, gpu_threads: bool | None = None,
                   blocking_sync: bool | None = None, trt_graphs: bool = False,
                   trt_inflight: int | None = None, early_workers: bool = False,
                   skip_empty: bool = False, budget: str | None = None,
                   fastdense: bool = False, gpu_resize: bool = False) -> dict:
    """The complete path: `test.csv` -> `submission.csv` + `runtime.json`.

    `budget`: the per-member window design (`predict`, `knee.window_budget`); None = all.

    `skip_empty`: the encoder only on the windows of present slots (`_model_logits`, bit-exact; on two
    GPUs, `_forward_split`).

    `uids`: the studies to predict instead of the whole `<split>.csv` (and `limit`/`offset`), in the
    given order; a study missing from the CSV is an error. Used to compare the kernel with the OOF
    predictions of a validation fold. `amp`: `False` keeps the weights in fp32 and removes the autocast
    of `predict` (and with it the second GPU); the caller can put it back from outside (`torch.autocast`)
    to get training precision. `None`/default = unchanged.

    `adabn`: before predicting, the encoder BatchNorm layers recompute their statistics on the first
    `adabn` test studies (`knee.adabn.adapt_bn`, `-1` = all), one more decoding pass. Eager only
    (`compile=False`): AOTI packages and compiled encoders carry the statistics inside. `None`/0 = off.

    `gpu_wait`: how the GPU is awaited at every batch (`predict`): `spin` or `block`.
    `cudnn_benchmark`: cudnn autotuning in eager (`predict`).
    `channels_last`: encoder weights in NHWC (`predict`); `None` = eager only.
    `fold_bn`: folded BatchNorm and fused convolution + ReLU (`predict`); `None` = in eager.
    `gpus`: how many GPUs to split every batch across (`predict`); `None` = all, in eager.
    `gpu_uint8`: the workers deliver the float32 stacks and percentiles + uint8 are computed on the
    device (`DicomStudies(to_uint8=False)`, `torch_uint8.to_uint8_torch`). NEUTRAL on the T4 at 288 px:
    bit-identical, but steady state only -1.5%, because the float32 bytes (4x the uint8 ones) cost almost
    as much in shared memory, pinning and copies as the workers save. Off. (At 512 px the raw-crop path,
    `gpu_resize`, replaces it.)
    `main_threads`: torch's OpenMP threads in the main process (`predict`); `None` = torch's own.
    `aoti_dir`: the folder of the run's single-study AOTI packages (`predict(aoti=...)`), on all GPUs; if
    they do not match, eager.

    `trt_dir`: the folder of the run's TensorRT engines, `trt8q_fold<k>.plan` for every fold used
    (`predict(trt=...)`); a fold without an engine or an import error -> eager. If `dense8_fold<k>.npz`
    packages (`knee.dense8`) are present, those are used.
    `gpu_threads`: one thread per GPU (`predict`); `None` = on with the engines.
    `blocking_sync`: the CUDA context of every GPU in blocking sync (`knee.trt.set_blocking_sync`, before
    any other use of the GPU): the TensorRT engine call waits for the GPU, and two threads busy-waiting
    take CPU from decoding (main thread 335 s against 30, +78 s). `None` = on with `trt_dir`.
    `trt_graphs`: the engines from CUDA graphs recorded on the full batch (`predict(trt_graphs=...)`).
    `trt_inflight`: queued blocks per engine before the thread sleeps (`predict`).
    `early_workers`: the decoding workers start before the engines are attached (`predict`).

    `tta` (from `knee.tta.parse_tta`): the flip-free variants to average at inference, one pass each over
    the same windows; `None` = the model as is.

    `stride`: the inference window stride instead of the checkpoint's (`apply_stride`): with 24 slices
    and depth 3, stride 4 = 48 windows instead of 88, 45% fewer encoder passes. Compilation artifacts
    and AOTI packages with a different stride have their own name (`-stride4`): those of the run do not
    match in shape and would be rejected anyway.

    `split="train"` with `limit`/`offset` is the test bench of the Efficiency track: the public test has
    three studies and only measures start-up, while the hidden rerun (~1,300 studies) shows no logs. The
    same training DICOMs, on the same code path, are the only way to see where the time goes. With the
    defaults the function is unchanged.
    """
    if not amp and compile:
        raise ValueError("amp=False only with the eager encoder (compile=False): compiled encoders "
                         "and AOTI packages are in fp16")
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_flags = None
    if blocking_sync if blocking_sync is not None else trt_dir is not None:
        from .trt import set_blocking_sync
        sync_flags = set_blocking_sync()
        log(f"CUDA context in blocking sync: {sync_flags}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # First of all: if the assigned GPU cannot run this PyTorch build, the kernel must die now and say
    # so, not halfway through the test set with a message that does not name the cause. Here a crash does
    # not cost an hour of quota, it costs the submission.
    log(f"device: {check_device(device)}")
    if device.type == "cuda":
        from .data import early as _early_init
        _ep = _early_init.active()
        if _ep is not None and hasattr(_ep, "register_cuda"):
            _ep.register_cuda()
    decode_jit = _decode_jit(log)
    if compile and device.type == "cuda":
        # `torch._inductor` and `torch._dynamo` cost ~5 s of disk on a cold page cache: they are imported in a
        # thread while checkpoints and CSV are read here. The GIL is not a problem, the time is I/O; if the
        # import fails, `compile_encoders` will find out and stay eager.
        import threading

        def _warm_imports() -> None:
            try:
                import torch._dynamo  # noqa: F401
                import torch._inductor.codecache  # noqa: F401
                import torch._inductor.compile_fx  # noqa: F401
            except Exception:
                pass

        threading.Thread(target=_warm_imports, daemon=True).start()
    series_dir = f"{split}_series"

    all_uids = pd.read_csv(root / f"{split}.csv", usecols=[ID])[ID].tolist()
    series = pd.read_csv(root / f"{series_dir}.csv")
    if uids is not None:
        missing = sorted(set(uids) - set(all_uids))
        if missing:
            raise ValueError(f"{len(missing)} requested studies are not in {split}.csv, "
                             f"e.g. {missing[:3]}")
        uids = list(uids)
        series = series[series[ID].isin(set(uids))].reset_index(drop=True)
    elif limit is not None:
        uids = all_uids[offset:offset + limit]
        series = series[series[ID].isin(set(uids))].reset_index(drop=True)
    else:
        uids = all_uids
    log(f"{len(uids)} studies, {len(series)} series")

    # The ~50 MB of compilation artifacts are read from the mount in a thread while the other checkpoints
    # load: both are I/O, and on a cold page cache the read alone cost 6 s. The file name is the run's,
    # known only from the first checkpoint: hence the callback.
    from concurrent.futures import ThreadPoolExecutor
    pool, fut, cache_file = ThreadPoolExecutor(max_workers=1), None, None

    def start_cache_read(first_cfg: dict) -> None:
        nonlocal fut, cache_file
        if compile and compile_cache_dir is not None:
            cache_file = Path(compile_cache_dir) / compile_cache_name(first_cfg.get("name", ""), stride)
            if cache_file.is_file():
                fut = pool.submit(cache_file.read_bytes)

    t = time.perf_counter()
    head_only = (os.environ.get("KNEE_HEAD_ONLY", "off") == "on"
                 and trt_dir is not None and not compile and not adabn and device.type == "cuda")
    models, cfg = load_ensemble(ckpt_dir, device, log, on_cfg=start_cache_read, folds=folds,
                                head_only=head_only)
    load_s = time.perf_counter() - t
    spec = _spec_of(cfg)
    apply_stride(models, stride, log)
    n_win = spec.shape[0] * windows_per_slot(spec.slices, models[0].window_stride,
                                             models[0].window_depth)
    log(f"run {cfg.get('name')} | spec {spec.tag} | {n_win} windows per study "
        f"(stride {models[0].window_stride}) | checkpoints in {load_s:.1f}s")
    cache_loaded, cache_s, packaged = False, 0.0, False
    if compile and compile_cache_dir is not None:
        t = time.perf_counter()
        # First the AOTI packages (nothing to compile), then the inductor cache.
        if device.type == "cuda":
            models = [m.half() for m in models]
            # The window stride travels in the checkpoint (or in `stride`): the shape of the encoder input, and
            # therefore the AOTI package that serves it, depends on it.
            packaged = load_aoti_encoders(
                Path(compile_cache_dir) / aoti_dir_name(cfg.get("name", ""), stride), models,
                aoti_meta(models, batch_size, n_win, spec.img), log)
        if not packaged:
            if fut is not None:
                cache_loaded = load_compile_cache(cache_file, log, data=fut.result())
            else:
                log(f"compilation cache missing: {cache_file}")
        cache_s = time.perf_counter() - t
    pool.shutdown(wait=False)

    dirs = {r.SeriesInstanceUID: root / series_dir / getattr(r, ID) / r.SeriesInstanceUID
            for r in series.itertuples(index=False)}
    t = time.perf_counter()
    rate = preflight(dirs, n=4)
    preflight_s = time.perf_counter() - t
    log(f"preflight: {rate:.1%} decoding failures in {preflight_s:.1f}s")
    t = time.perf_counter()
    from .data import early as _early_slots
    _ep = _early_slots.active()
    if (_ep is not None and _ep.uids == list(uids) and _ep.spec == spec
            and _ep.series_dir == series_dir and _ep.root == str(root)):
        slots = _ep.slots
    else:
        slots = pick_slots(series, series_dirs=dirs, slots=spec.active_slots,
                           fallback=spec.slot_fallback)
    slots_s = time.perf_counter() - t
    filled = np.array([len(slots.get(s, {})) for s in uids])
    log(f"filled slots: mean {filled.mean():.2f}/{spec.shape[0]} | "
        + " ".join(f"{k}:{int((filled == k).sum())}" for k in range(spec.shape[0] + 1))
        + f" | in {slots_s:.1f}s")

    ds = DicomStudies(root, uids, slots, spec, series_dir=series_dir, to_uint8=not gpu_uint8,
                      raw=gpu_resize)
    adabn_n, adabn_batches, adabn_s = 0, 0, 0.0
    if adabn:
        if compile:
            raise ValueError("adabn only with the eager encoder (compile=False)")
        from .adabn import adapt_bn
        t = time.perf_counter()
        adabn_n = len(uids) if adabn < 0 else min(int(adabn), len(uids))
        ds_bn = DicomStudies(root, uids[:adabn_n], slots, spec, series_dir=series_dir)
        dl_bn = DataLoader(ds_bn, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                           num_workers=workers, pin_memory=device.type == "cuda")
        adabn_batches = adapt_bn(models, dl_bn, device, k=int(cfg["train"]["k_train"]),
                                 seed=0, amp=bool(cfg["train"].get("amp", True)), log=log)
        del dl_bn
        adabn_s = time.perf_counter() - t
        log(f"adabn: BatchNorm recomputed on {adabn_n} {split} studies ({adabn_batches} "
            f"batches, k_train {cfg['train']['k_train']}) in {adabn_s:.1f}s")
    trt_blobs, trt_load = None, {}
    if trt_dir is not None:
        t = time.perf_counter()
        ids = list(folds) if folds is not None else list(range(len(models)))
        try:
            if all((Path(trt_dir) / f"dense8_fold{f}.npz").is_file() for f in ids):
                from .dense8 import load_pack
                trt_blobs = {f: load_pack(Path(trt_dir) / f"dense8_fold{f}.npz") for f in ids}
                trt_load["kind"] = trt_blobs[ids[0]]["meta"].get("kind", "dense8")
                # TensorRT only if a package has engines (a dense8s package that is entirely ours: none)
                if any(b["meta"].get("engines") for b in trt_blobs.values()):
                    from .trt import trt_module
                    trt_module()
                    trt_load["import_s"] = round(time.perf_counter() - t, 2)
                else:
                    trt_load["import_s"] = 0.0
            else:
                from .trt import trt_module
                trt_module()
                trt_load["import_s"] = round(time.perf_counter() - t, 2)
                trt_blobs = {f: (Path(trt_dir) / f"trt8q_fold{f}.plan").read_bytes()
                             for f in ids}
                trt_load["kind"] = "trt"
        except Exception as exc:
            log(f"!! TensorRT engines not loaded, staying eager: {type(exc).__name__}: "
                f"{str(exc)[:200]}")
            trt_blobs = None
        trt_load["read_s"] = round(time.perf_counter() - t, 2)
    probs, stats = predict(models, ds, device, batch_size=batch_size, workers=workers,
                           time_budget_s=time_budget_s, log=log, compile=compile,
                           packaged=packaged, tta=tta, gpu_wait=gpu_wait,
                           cudnn_benchmark=cudnn_benchmark, channels_last=channels_last,
                           fold_bn=fold_bn, gpus=gpus, main_threads=main_threads,
                           aoti=aoti_dir, fold_ids=folds, amp=amp, trt=trt_blobs,
                           gpu_threads=gpu_threads, trt_graphs=trt_graphs,
                           trt_inflight=trt_inflight, early_workers=early_workers,
                           skip_empty=skip_empty, budget=budget, fastdense=fastdense)
    if trt_dir is not None:
        stats["trt_load"] = trt_load
    stats["blocking_sync"] = sync_flags
    stats["decode_jit"] = decode_jit
    n_missing = write_submission(uids, probs, out_dir / "submission.csv")

    stats.update({
        "run": cfg.get("name"), "n_folds": len(models), "cache_spec": spec.tag,
        "folds": folds,
        "window_stride": models[0].window_stride, "n_windows": n_win,
        "stride_override": stride,
        "device": device.type, "preflight_failure_rate": round(rate, 4),
        "rows_filled_neutral": n_missing,
        "degraded": bool(n_missing),
        "split": split, "batch_size": batch_size, "workers": workers,
        "load_ckpt_s": round(load_s, 1), "preflight_s": round(preflight_s, 1),
        "pick_slots_s": round(slots_s, 1),
        "compile_cache_loaded": cache_loaded, "compile_cache_s": round(cache_s, 1),
        "adabn_studies": adabn_n, "adabn_batches": adabn_batches,
        "adabn_s": round(adabn_s, 1), "amp": amp,
    })
    (out_dir / "runtime.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(json.dumps(stats, indent=2))
    if n_missing:
        log(f"!! {n_missing} rows out of {len(uids)} came out neutral (0.5): "
            f"the submission is valid but partial")
    return stats


# --- ensembles of several runs ------------------------------------------------------------
# The combination is the same one measured on OOF predictions: equal-weight rank averaging of the
# admitted arms. Combining differently here would make the measured number differ from the leaderboard
# one. Not used by the efficiency submission.

def is_slot_prefix(narrow: CacheSpec, wide: CacheSpec) -> bool:
    """Does `narrow` produce the same pixels as the first slots of `wide`?

    In `build_study` slot `j + k*n_seq` is sequence `j` at scale `crops[k]`, and `build_series` changes
    **only** `crop_box` between scales: same series, same slice order, same band. So the first
    `len(active_slots)` slots of a two-scale spec are, byte for byte, those of a one-scale spec on
    `crops[0]`. This is what allows decoding the DICOMs once for all arms, and since decoding dominates
    wall time, it is the difference between an ensemble that costs the sum of its arms and one that
    costs little more than the widest.
    """
    return (narrow.img == wide.img and narrow.slices == wide.slices
            and narrow.band == wide.band and narrow.center == wide.center
            and narrow.side_tie == wide.side_tie and narrow.sag_band_lm == wide.sag_band_lm
            and narrow.slot_fallback == wide.slot_fallback
            and narrow.active_slots == wide.active_slots
            and len(narrow.crops) <= len(wide.crops)
            and narrow.crops == wide.crops[:len(narrow.crops)])


def load_arms(ckpt_dirs: dict, device, log=print) -> tuple[list, CacheSpec]:
    """One arm per directory. Also returns the widest spec, the one to decode.

    The arm's identity is read from the checkpoint (`cfg["name"]`), not from the folder name: the Kaggle
    mount can rename, the checkpoint cannot. If one weights dataset is mounted in place of another, this
    says so instead of producing a plausible submission made with the wrong models.
    """
    arms = []
    for want, d in ckpt_dirs.items():
        models, cfg = load_ensemble(Path(d), device, log)
        got = cfg.get("name")
        if got != want:
            raise ValueError(f"{d} contains run `{got}`, expected `{want}`: "
                             f"the weights datasets are mounted swapped")
        arms.append({"name": got, "models": models, "spec": _spec_of(cfg)})

    wide = max((a["spec"] for a in arms), key=lambda s: s.shape[0])
    for a in arms:
        if not is_slot_prefix(a["spec"], wide):
            raise ValueError(
                f"the spec of `{a['name']}` ({a['spec'].tag}) is not a slot prefix of "
                f"{wide.tag}: two decodings would be needed and this path does not do them.")
        a["n_slots"] = a["spec"].shape[0]
        a["stride"] = a["models"][0].window_stride
        a["depth"] = a["models"][0].window_depth
    log("arms: " + " | ".join(f"{a['name']} {a['n_slots']}sl x{len(a['models'])}f"
                                for a in arms) + f" || decoding on {wide.tag}")
    return arms, wide


@_infer_ctx()
def predict_arms(arms: list, ds: DicomStudies, device, batch_size: int = 4, workers: int = 4,
                 amp: bool = True, time_budget_s: float | None = None, log=print,
                 log_every: int = 100, compile: bool = True,
                 tta: list[Variant] | None = None,
                 gpu_wait: str = "block", cudnn_benchmark: bool = False,
                 channels_last: bool | None = None,
                 fold_bn: bool | None = None, gpus: int | None = None) -> tuple[dict, dict]:
    """One decoding, N arms. Rows not computed stay NaN, as in `predict`.

    `tta` (from `knee.tta.parse_tta`) applies to every arm the same way: the same flip-free variants as
    `predict`, averaged in probability within the arm before the rank average across arms.

    `gpu_wait`, `cudnn_benchmark`, `channels_last`, `fold_bn`, `gpus`: the same levers as `predict`, with
    the same defaults and the same conditions (the last three eager only, `compile=False`). With
    `gpus=2` every GPU receives `batch_size` studies and runs **all** arms on them
    (`_forward_split_arms`): studies are split, not models, as in `knee.multigpu`, but in a single
    process. The prediction of a study does not depend on what else is in the batch (BatchNorm in eval),
    so the probabilities are those of a single GPU.
    """
    if gpu_wait not in ("spin", "block"):
        raise ValueError(f"gpu_wait={gpu_wait!r}: 'spin' or 'block'")
    eligible = device.type == "cuda" and amp and not compile
    if gpus is None:
        gpus = max(torch.cuda.device_count(), 1) if eligible else 1
    if gpus > 1 and not eligible:
        log(f"!! gpus={gpus} applies only to eager fp16 on CUDA: staying on one GPU")
        gpus = 1
    if gpus > 1 and torch.cuda.device_count() < gpus:
        log(f"!! gpus={gpus} but {torch.cuda.device_count()} GPUs visible: using those")
        gpus = max(torch.cuda.device_count(), 1)
    # The copies for the other GPUs before folding and changing layout (see `predict`).
    replicas = []
    if gpus > 1:
        with torch.inference_mode(False):
            replicas = [[[copy.deepcopy(m) for m in a["models"]] for a in arms]
                        for _ in range(gpus - 1)]
    if fold_bn is None:
        fold_bn = not compile
    fold_bn = bool(fold_bn and not compile)
    n_folded = sum(fold_densenet_bn(m) for a in arms for m in a["models"]) if fold_bn else 0
    if channels_last is None:
        channels_last = not compile
    channels_last = bool(channels_last and all(channels_last_ok(a["models"]) for a in arms))
    if channels_last:
        for a in arms:
            a["models"] = [m.to(memory_format=torch.channels_last) for m in a["models"]]
    compiled = False
    if device.type == "cuda":
        if amp:
            for a in arms:
                a["models"] = [m.half() for m in a["models"]]
        if compile:
            for a in arms:
                ms = a["models"]
                a["weights"] = (share_encoder(ms) if len(ms) > 1
                                and len({type(m.encoder) for m in ms}) == 1 else None)
            compiled = all(compile_encoders(a["models"], log) for a in arms)
        torch.backends.cudnn.benchmark = (not compiled) and cudnn_benchmark    # see `predict`
    devices = [device] + [torch.device("cuda", j) for j in range(1, gpus)]
    dev_arms = [[a["models"] for a in arms]]
    for d, rep in zip(devices[1:], replicas):
        on_d = []
        for ms in rep:
            if fold_bn:
                for m in ms:
                    fold_densenet_bn(m)
            ms = [m.to(d) for m in ms]
            if channels_last:
                ms = [m.to(memory_format=torch.channels_last) for m in ms]
            on_d.append([m.half() for m in ms])
        dev_arms.append(on_d)
    if gpus > 1:
        log(f"{gpus} GPUs: {[torch.cuda.get_device_name(d) for d in devices]} | "
            f"{batch_size} studies per GPU per batch, all arms on every GPU")

    loader = _make_loader(ds, batch_size * gpus, workers, device)
    wait_ev = (torch.cuda.Event(blocking=True)
               if device.type == "cuda" and gpu_wait == "block" else None)

    probs = {a["name"]: np.full((len(ds), len(TARGETS)), np.nan, dtype=np.float32)
             for a in arms}
    t0, decode_s, gpu_s, n_empty, n_err, done = time.time(), 0.0, 0.0, 0, 0, 0
    first_batches: list[float] = []     # wall time of the first batches: start-up and autotuning
    b0 = time.perf_counter()

    # Static tensors computed once outside the loop for each arm if available
    slot_ofs = ({a["name"]: window_slot_index(a["n_slots"], ds.spec.shape[1], device, a["stride"], a["depth"])
                 for a in arms} if hasattr(ds, "spec") else {})
    split_slot_ofs = {(0, name): so for name, so in slot_ofs.items()}

    for vol, mask, idx, dt, errs in _regroup(loader, batch_size * gpus, cat_vol=gpus == 1):
        decode_s += float(dt.sum())
        n_err += int(errs.sum())
        empty = (~mask.any(dim=1)).numpy()
        n_empty += int(empty.sum())
        g0 = time.perf_counter()
        if gpus > 1:
            outs = _forward_split_arms(dev_arms, devices, split_slot_ofs, arms, vol, mask,
                                       batch_size, tta, block=gpu_wait == "block")
            for a, batch in zip(arms, outs):
                probs[a["name"]][idx.numpy()] = batch
        else:
            if compiled:
                vol, mask, n_real = _pad_batch(vol, mask, batch_size)
            else:
                n_real = vol.shape[0]
            vol = vol.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pms = []
            if device.type == "cuda" and amp:
                volf = vol.half().mul_(1.0 / 255.0)
                for a in arms:
                    k = a["n_slots"]
                    v, mk = volf[:, :k], mask[:, :k]
                    w = make_windows(v, a["stride"], a["depth"])
                    if a["name"] not in slot_ofs:
                        slot_ofs[a["name"]] = window_slot_index(k, v.shape[2], device, a["stride"], a["depth"])
                    slot_of = slot_ofs[a["name"]]
                    p = None
                    with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                        try:
                            p = (fold_probs(a["models"], w, slot_of, mk, a.get("weights")) if not tta
                                 else _probs_tta(a["models"], v, mk, slot_of, a["stride"], a["depth"],
                                                 a.get("weights"), tta))
                        except Exception as exc:
                            if not compiled:
                                raise
                            log(f"!! compiled encoder failed on the first forward, staying "
                                f"eager: {type(exc).__name__}: {str(exc)[:300]}")
                            for b_ in arms:
                                uncompile_encoders(b_["models"])
                            compiled = False
                            torch.backends.cudnn.benchmark = cudnn_benchmark
                    if p is None:
                        # The eager retry is OUTSIDE the `except`: while the exception is alive its traceback holds the frames
                        # of the failed forward, and with them the activations (11 GiB on a T4 for densenet169 at 288 px): a
                        # retry inside the `except` ran out of memory in turn.
                        _free_cuda()
                        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                            p = (fold_probs(a["models"], w, slot_of, mk, a.get("weights")) if not tta
                                 else _probs_tta(a["models"], v, mk, slot_of, a["stride"], a["depth"],
                                                 a.get("weights"), tta))
                    pms.append(p.mean(0)[:n_real])
            else:
                volf = vol.float().div_(255.0)
                for a in arms:
                    k = a["n_slots"]
                    v, mk = volf[:, :k], mask[:, :k]
                    w = make_windows(v, a["stride"], a["depth"])
                    if a["name"] not in slot_ofs:
                        slot_ofs[a["name"]] = window_slot_index(k, v.shape[2], device, a["stride"], a["depth"])
                    slot_of = slot_ofs[a["name"]]
                    p = (fold_probs(a["models"], w, slot_of, mk, a.get("weights")) if not tta
                         else _probs_tta(a["models"], v, mk, slot_of, a["stride"], a["depth"],
                                         a.get("weights"), tta))
                    pms.append(p.mean(0)[:n_real])
            # All arms queued, then a single wait (see `gpu_wait` in `predict`).
            if wait_ev is not None:
                wait_ev.record()
                wait_ev.synchronize()
            for a, pm in zip(arms, pms):
                batch = pm.cpu().numpy()
                batch[empty] = np.nan
                probs[a["name"]][idx.numpy()] = batch
        gpu_s += time.perf_counter() - g0
        done += len(idx)
        if len(first_batches) < 4:
            first_batches.append(round(time.perf_counter() - b0, 1))
            b0 = time.perf_counter()
        if done % log_every < len(idx):
            rate = done / max(time.time() - t0, 1e-6)
            log(f"  {done}/{len(ds)} | {rate:.2f} studies/s | ~"
                f"{(len(ds) - done) / rate / 60:.0f} min left")
        if time_budget_s and time.time() - t0 > time_budget_s:
            log(f"!! budget exhausted at {done}/{len(ds)} studies: the rest stays neutral")
            break

    wall = time.time() - t0
    stats = {
        "n_studies": len(ds), "n_predicted": done,
        "wall_s": round(wall, 1), "decode_cpu_s": round(decode_s, 1), "gpu_s": round(gpu_s, 1),
        "s_per_study": round(wall / max(done, 1), 3),
        "decode_share_of_cpu_work": round(decode_s / max(decode_s + gpu_s, 1e-9), 3),
        "studies_without_any_slot": n_empty, "slice_decode_failures": n_err,
        "compiled": compiled, "first_batches_s": first_batches, "gpu_wait": gpu_wait,
        "cudnn_benchmark": cudnn_benchmark, "channels_last": channels_last,
        "fold_bn": fold_bn, "fold_bn_layers": n_folded,
        "fold_bn_status": (fold_bn_status([m for dev in dev_arms for ms in dev for m in ms])
                           if fold_bn else None),
        "gpus": gpus,
    }
    return probs, stats


def rank_average(probs: dict) -> np.ndarray:
    """Mean of the percentile ranks per column.

    NaN rows stay NaN (`na_option="keep"`): they become 0.5 in `write_submission` and are counted, not
    hidden. The metric is rank-based, so ranking on the whole test set and then splitting public and
    private changes nothing: the rank within a subset is preserved.
    """
    frames = [pd.DataFrame(p).rank(pct=True, na_option="keep") for p in probs.values()]
    return (sum(frames) / len(frames)).to_numpy(dtype=np.float32)


def run_submission_ensemble(root: Path, ckpt_dirs: dict, out_dir: Path,
                            batch_size: int = 8, workers: int = 4,
                            time_budget_s: float | None = None, log=print,
                            split: str = "test", limit: int | None = None,
                            offset: int = 0, tta: list[Variant] | None = None,
                            compile: bool = True, predict_kw: dict | None = None) -> dict:
    """Like `run_submission`, but with several arms and a single decoding.

    `predict_kw`: levers of `predict_arms` passed as they are (`gpus`, `gpu_wait`, ...); empty = the
    defaults.

    `tta`: the flip-free variants (`knee.tta.parse_tta`), applied to every arm and recorded in
    `runtime.json` as in `run_submission`.

    `compile=False`: eager encoders with the levers of `predict_arms` (the two T4s, blocking wait, NHWC,
    folded BatchNorm).

    `split="train"` with `limit`/`offset` is the ensemble's rerun test bench, as for `run_submission`.
    With the defaults it is unchanged.
    """
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {check_device(device)}")
    decode_jit = _decode_jit(log)

    series_dir = f"{split}_series"
    uids = pd.read_csv(root / f"{split}.csv", usecols=[ID])[ID].tolist()
    series = pd.read_csv(root / f"{series_dir}.csv")
    if limit is not None:
        uids = uids[offset:offset + limit]
        series = series[series[ID].isin(set(uids))].reset_index(drop=True)
    log(f"{len(uids)} studies, {len(series)} series")

    arms, spec = load_arms(ckpt_dirs, device, log)

    dirs = {r.SeriesInstanceUID: root / series_dir / getattr(r, ID) / r.SeriesInstanceUID
            for r in series.itertuples(index=False)}
    rate = preflight(dirs, n=4)
    log(f"preflight: {rate:.1%} decoding failures")
    slots = pick_slots(series, series_dirs=dirs, slots=spec.active_slots,
                       fallback=spec.slot_fallback)
    filled = np.array([len(slots.get(s, {})) for s in uids])
    log(f"filled slots: mean {filled.mean():.2f}/{spec.shape[0]}")

    ds = DicomStudies(root, uids, slots, spec, series_dir=series_dir)
    if tta:
        log(f"flip-free TTA on every arm: {[v.name for v in tta]} "
            f"({len(tta)} passes per window)")
    probs, stats = predict_arms(arms, ds, device, batch_size=batch_size, workers=workers,
                                time_budget_s=time_budget_s, log=log, tta=tta,
                                compile=compile, **(predict_kw or {}))
    blended = rank_average(probs)
    n_missing = write_submission(uids, blended, out_dir / "submission.csv")

    stats.update({
        "run": "ensemble", "arms": [a["name"] for a in arms],
        "n_folds_per_arm": {a["name"]: len(a["models"]) for a in arms},
        "combination": "equal-weight rank average",
        "tta": [v.name for v in tta] if tta else None,
        "split": split, "batch_size": batch_size, "workers": workers,
        "cache_spec": spec.tag, "device": device.type,
        "preflight_failure_rate": round(rate, 4),
        "rows_filled_neutral": n_missing, "degraded": bool(n_missing),
        "decode_jit": decode_jit,
    })
    (out_dir / "runtime.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(json.dumps(stats, indent=2))
    if n_missing:
        log(f"!! {n_missing} rows out of {len(uids)} came out neutral (0.5)")
    return stats
