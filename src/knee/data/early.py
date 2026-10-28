"""Decoding that starts before `torch`: the raw-crop workers a few seconds after kernel start.

In a plain submission kernel the DataLoader workers are born inside `predict`, after
`import knee.infer` (~12 s on a cold page cache: torch 5.2, scikit-learn via `knee.tta` 2.1, pandas
1.6), checkpoint loading, slot selection and engine setup: ~23 s during which nobody reads a DICOM.
With decoding as the bottleneck, every second of start-up is a second added to the total.

Here the kernel, before importing torch, reads the spec from the checkpoint without torch
(`spec_from_checkpoint`: an unpickler with placeholders for the tensors), computes the slots as
`run_submission` does and starts `workers` processes (fork) that take studies in order from a
shared counter, at most `ahead` studies ahead of those consumed, and build them with
`raw_crop.build_study_raw_np` (the same bytes as `DicomStudies(raw=True)`). Every payload goes into
a file in `/dev/shm`; the main process maps it, deletes it immediately (the memory lives as long as
the arrays do) and turns it into tensors without copies: no `pin_memory`. `predict` takes the
batches from `take(ds)` if the pool was started for the same studies, otherwise it uses the plain
DataLoader. If a worker dies, the main process decodes the missing studies.
"""
from __future__ import annotations

import mmap
import os
import pickle
import queue as _queue
import time
import zipfile

import numpy as np

_ACTIVE: list = [None]
_KEYS = ("raw", "u8", "resc", "meta")
_SHM = "/dev/shm"


# --- the spec from the checkpoint, without torch ------------------------------------------------------------
class _Stub:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return None

    def __setstate__(self, state):
        pass


class _NoTorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch") or module.startswith("numpy.core.multiarray") and name == "_reconstruct":
            return _Stub
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub

    def persistent_load(self, pid):
        return None


def checkpoint_cfg(path) -> dict:
    """The `cfg` of a `torch.save` checkpoint (zip format) without importing torch."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        with z.open(name) as fh:
            ck = _NoTorchUnpickler(fh).load()
    return ck["cfg"]


def spec_from_checkpoint(path):
    from .cache import CacheSpec
    cfg = checkpoint_cfg(path)
    return CacheSpec(**dict(cfg["cache"], band=tuple(cfg["cache"]["band"])))


# --- the workers -----------------------------------------------------------------------------------------
def _keep_freed_memory() -> None:
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 32 * 1024 * 1024)      # M_MMAP_THRESHOLD
        libc.mallopt(-1, 2 ** 31 - 1)           # M_TRIM_THRESHOLD
    except (OSError, AttributeError):
        pass


def _write_shm(name: str, payload: dict) -> list:
    layout, off = [], 0
    for k in _KEYS:
        a = np.ascontiguousarray(payload[k])
        layout.append((k, a.dtype.str, a.shape, off))
        off += (a.nbytes + 63) // 64 * 64
    size = max(off, 64)
    fd = os.open(os.path.join(_SHM, name), os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size)
    finally:
        os.close(fd)
    for (k, dt, shape, o), a in zip(layout, (payload[k] for k in _KEYS)):
        a = np.ascontiguousarray(a)
        if a.nbytes:
            mm[o:o + a.nbytes] = a.reshape(-1).view(np.uint8).data
    mm.close()
    return layout


def _trace_item(d: str, i: int, t0: float, dt: float, wait: float) -> None:
    """Like `infer._trace_item` (`w<pid>.csv`: study, pid, start, duration) plus the wait on the semaphore
    before the study: the time the worker spent waiting for the GPU to consume.
    """
    try:
        with open(os.path.join(d, f"w{os.getpid()}.csv"), "a", encoding="ascii") as fh:
            fh.write(f"{i},{os.getpid()},{t0:.4f},{dt:.4f},{wait:.4f}\n")
    except OSError:
        pass


def _worker(k: int, tag: str, root: str, series_dir: str, uids: list, slots: dict, spec,
            counter, lock, sem, res_q) -> None:
    from .dicom import DecodeErrors
    from .raw_crop import build_study_raw_np, empty_payload_np
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    _keep_freed_memory()
    n = len(uids)
    trace = os.environ.get("KNEE_TRACE_DIR")
    while True:
        tw = time.time()
        sem.acquire()
        with lock:
            i = counter.value
            counter.value = i + 1
        if i >= n:
            sem.release()
            break
        errors = DecodeErrors()
        t0 = time.perf_counter()
        try:
            payload, mask, _ = build_study_raw_np(os.path.join(root, series_dir, uids[i]),
                                                  slots.get(uids[i], {}), spec, errors)
        except Exception:
            payload, mask = empty_payload_np(), np.zeros(spec.shape[0], dtype=bool)
        dt = time.perf_counter() - t0
        if trace:
            _trace_item(trace, i, time.time() - dt, dt, time.time() - dt - tw)
        name = f"knee-early-{tag}-{i}"
        try:
            layout = _write_shm(name, payload)
            res_q.put((i, name, layout, np.asarray(mask, dtype=bool).tobytes(), dt, len(errors)))
        except Exception as exc:                 # /dev/shm full or similar: the parent redoes the study
            res_q.put((i, None, repr(exc), None, dt, len(errors)))


# --- the pool ------------------------------------------------------------------------------------------
class EarlyPool:
    def __init__(self, root, series_dir: str, uids: list, slots: dict, spec, workers: int = 6,
                 ahead: int = 96, log=print):
        import multiprocessing as mp
        self.root, self.series_dir = str(root), series_dir
        self.uids, self.slots, self.spec = list(uids), slots, spec
        self.log = log
        self.t_start = time.time()
        self.tag = f"{os.getpid()}-{int(self.t_start * 1000) % 10**8}"
        # The C decoding library loaded before the fork, like `infer._decode_jit` before the DataLoader: the
        # workers inherit it (without it, `_c_walk` falls back to the Python walks).
        from . import cfast
        self.decode_jit = cfast.load()
        ctx = mp.get_context("fork")
        self.counter, self.lock = ctx.Value("i", 0, lock=False), ctx.Lock()
        self.sem = ctx.Semaphore(ahead)
        self.res_q = ctx.Queue()
        self.procs = [ctx.Process(target=_worker, name=f"knee-early-{k}", daemon=True,
                                  args=(k, self.tag, self.root, series_dir, self.uids, slots, spec,
                                        self.counter, self.lock, self.sem, self.res_q))
                      for k in range(workers)]
        for p in self.procs:
            p.start()
        self.stats = {"workers": workers, "ahead": ahead, "parent_decoded": 0, "shm_failed": 0}

    def _attach(self, name: str, layout: list) -> dict:
        path = os.path.join(_SHM, name)
        fd = os.open(path, os.O_RDWR)
        try:
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
            os.unlink(path)                       # the memory stays alive as long as the arrays do
        out = {}
        for k, dt, shape, off in layout:
            cnt = int(np.prod(shape)) if len(shape) else 1
            out[k] = np.frombuffer(mm, dtype=np.dtype(dt), count=cnt, offset=off).reshape(shape)
        return out

    def _decode_here(self, i: int):
        from .dicom import DecodeErrors
        from .raw_crop import build_study_raw_np, empty_payload_np
        errors = DecodeErrors()
        t0 = time.perf_counter()
        try:
            payload, mask, _ = build_study_raw_np(os.path.join(self.root, self.series_dir, self.uids[i]),
                                                  self.slots.get(self.uids[i], {}), self.spec, errors)
        except Exception:
            payload, mask = empty_payload_np(), np.zeros(self.spec.shape[0], dtype=bool)
        self.stats["parent_decoded"] += 1
        return payload, np.asarray(mask, dtype=bool), time.perf_counter() - t0, len(errors)

    def studies(self):
        """(i, numpy payload, mask, dt, errs) in index order."""
        n, nxt, got = len(self.uids), 0, {}
        trace = [] if os.environ.get("KNEE_TRACE_DIR") else None
        while nxt < n:
            while nxt not in got:
                try:
                    i, name, layout, mask_b, dt, ne = self.res_q.get(timeout=5.0)
                except _queue.Empty:
                    if all(not p.is_alive() for p in self.procs) or any(
                            p.exitcode not in (None, 0) for p in self.procs):
                        self.log(f"!! early: workers died {[p.exitcode for p in self.procs]}: "
                                 f"study {nxt} is decoded by the main process")
                        got[nxt] = self._decode_here(nxt)
                    continue
                if name is None:
                    self.stats["shm_failed"] += 1
                    got[i] = self._decode_here(i)
                else:
                    got[i] = (self._attach(name, layout), np.frombuffer(mask_b, dtype=bool).copy(), dt, ne)
            payload, mask, dt, ne = got.pop(nxt)
            if trace is not None:               # [study, time it is delivered, number already ready after it]
                trace.append(f"{nxt},{time.time():.4f},{len(got)}")
            self.sem.release()
            yield nxt, payload, mask, dt, ne
            nxt += 1
        if trace is not None:
            try:
                with open(os.path.join(os.environ["KNEE_TRACE_DIR"], "early_parent.csv"), "w",
                          encoding="ascii") as fh:
                    fh.write("\n".join(trace) + "\n")
            except OSError:
                pass

    def batches(self, batch_size: int):
        """The batches in the format of `raw_crop.collate_raw` (payloads as CPU tensors)."""
        import torch
        from .raw_crop import payload_to_torch
        buf = []
        for i, payload, mask, dt, ne in self.studies():
            buf.append((payload_to_torch(payload), torch.from_numpy(mask), i, dt, ne))
            if len(buf) == batch_size:
                yield _collate(buf)
                buf = []
        if buf:
            yield _collate(buf)
        self.close()

    def close(self) -> None:
        for p in self.procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        try:
            for f in os.listdir(_SHM):
                if f.startswith(f"knee-early-{self.tag}-"):
                    os.unlink(os.path.join(_SHM, f))
        except OSError:
            pass


# --- the lean pool: payloads written into a shared buffer, I/O in C (`knee.data.lean`) -----------
# Every worker has its own ring in the same shared anonymous mapping (created before the fork): it
# allocates in order, the parent frees in order (a worker's studies arrive in index order and are
# consumed in order). No files in /dev/shm, no copies: the parent reads views on the mapping and,
# after `register_cuda`, copies them to the GPU as pinned memory. `ctrl` (shared int64): [0, W) the
# tails freed by the parent (total bytes), _NXT the index the parent is waiting for (a worker stuck on
# a full ring for exactly that study falls back to /dev/shm: no deadlock). This is the pool of the
# final submission (8 workers).
_NXT = 63
_PAGE = 4096


class _RingWriter:
    """FIFO allocations of one worker; the parent's counter protects the views still in use."""

    def __init__(self, ring, ctrl, worker: int):
        self.ring, self.ctrl, self.worker = ring, ctrl, worker
        self.pos = self.head = 0
        self.current = -1
        self.pending = None
        self.wait_s = 0.0

    def alloc(self, nbytes: int):
        size = len(self.ring)
        nbytes = (int(nbytes) + _PAGE - 1) // _PAGE * _PAGE
        if nbytes > size:
            return None, None
        pos, head = self.pos, self.head
        if pos + nbytes > size:
            head += size - pos
            pos = 0
        end = head + nbytes
        t0 = time.perf_counter()
        while end - int(self.ctrl[self.worker]) > size:
            if self.current == int(self.ctrl[_NXT]) and time.perf_counter() - t0 > 0.5:
                self.wait_s += time.perf_counter() - t0
                return None, None
            time.sleep(0.001)
        self.wait_s += time.perf_counter() - t0
        tok = (pos, end, self.pos, self.head)
        self.pos, self.head = pos + nbytes, end
        self.pending = tok
        return self.ring[pos:pos + nbytes], tok

    def rollback(self, tok=None) -> None:
        tok = self.pending if tok is None else tok
        if tok is not None:
            self.pos, self.head = tok[2], tok[3]
        self.pending = None


def _ring_worker(k: int, tag: str, root: str, series_dir: str, uids: list, slots: dict, spec, counter, lock,
                 res_q, mm, ring_size: int, ctrl_mm, check: int) -> None:
    from . import lean
    from .dicom import DecodeErrors
    from .raw_crop import build_study_raw_np, empty_payload_np
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    _keep_freed_memory()
    ring = np.frombuffer(mm, np.uint8, count=ring_size, offset=k * ring_size)
    ctrl = np.frombuffer(ctrl_mm, np.int64, count=64)
    writer = _RingWriter(ring, ctrl, k)
    stats = {"lean": 0, "old": 0, "shm": 0, "checked": 0, "mismatch": 0, "errors": 0, "wait_s": 0.0}
    alloc, rollback = writer.alloc, writer.rollback

    def to_ring(payload: dict):
        """A numpy payload copied into the ring: (token, layout) or (None, None)."""
        layout, off = [], 0
        for key in _KEYS:
            a = np.ascontiguousarray(payload[key])
            layout.append((key, a.dtype.str, a.shape, off))
            off += (a.nbytes + 63) // 64 * 64
        mem, tok = alloc(max(off, 64))
        if mem is None:
            return None, None
        for (key, _dt, _shape, o), a in zip(layout, (payload[x] for x in _KEYS)):
            a = np.ascontiguousarray(a)
            if a.nbytes:
                mem[o:o + a.nbytes] = a.reshape(-1).view(np.uint8)
        return tok, layout

    n = len(uids)
    use_lean = lean.usable(spec)
    trace = os.environ.get("KNEE_TRACE_DIR")
    while True:
        with lock:
            i = counter.value
            counter.value = i + 1
        if i >= n:
            break
        writer.current = i
        writer.pending = None
        study = os.path.join(root, series_dir, uids[i])
        sl = slots.get(uids[i], {})
        errors = DecodeErrors()
        t0 = time.perf_counter()
        tok = layout = None
        mask = None
        if use_lean:
            got = None
            try:
                got = lean.build_study_lean(study, sl, spec, errors, alloc=alloc)
            except Exception as exc:                 # a failure of the lean path: the study goes the old way
                stats["errors"] += 1
                if stats["errors"] <= 3:
                    res_q.put(("warn", k, f"lean path failed on {uids[i]}: {exc!r}"))
            if got is not None:
                payload, mask, info = got
                tok, layout = info["alloc"]
                if check > 0:
                    check -= 1
                    stats["checked"] += 1
                    ref_p, ref_m, _ = build_study_raw_np(study, sl, spec, DecodeErrors())
                    if not (lean.same_payload(payload, ref_p) and np.array_equal(mask, ref_m)):
                        stats["mismatch"] += 1
                        use_lean = False
                        res_q.put(("warn", k, f"lean path differs from build_study_raw_np on {uids[i]}: "
                                              "falling back to the old one"))
                        rollback(tok)
                        tok, layout, mask = None, None, None
                if tok is not None:
                    stats["lean"] += 1
            else:
                errors = DecodeErrors()
        if tok is None:
            # The builder can fail after reserving memory: that reservation never reaches the parent, so it must
            # be cancelled before rewriting with the usual path.
            rollback()
            try:
                payload, mask, _ = build_study_raw_np(study, sl, spec, errors)
            except Exception:
                payload, mask = empty_payload_np(), np.zeros(spec.shape[0], dtype=bool)
            tok, layout = to_ring(payload)
            stats["old"] += 1
            if tok is None:                          # ring full for the study the parent is waiting for, or a huge study
                stats["shm"] += 1
                dt = time.perf_counter() - t0
                name = f"knee-early-{tag}-{i}"
                try:
                    lay = _write_shm(name, payload)
                    res_q.put(("shm", i, name, lay, np.asarray(mask, dtype=bool).tobytes(), dt, len(errors)))
                except Exception as exc:
                    res_q.put(("shm", i, None, repr(exc), None, dt, len(errors)))
                continue
        dt = time.perf_counter() - t0
        if trace:
            _trace_item(trace, i, time.time() - dt, dt, 0.0)
        res_q.put(("ring", i, k, tok[0], layout, np.asarray(mask, dtype=bool).tobytes(), dt, len(errors), tok[1]))
    stats["wait_s"] = writer.wait_s
    res_q.put(("stats", k, stats))


# Driver flags for `LeanPool.register_cuda_early` (cuda.h): blocking sync as in
# `trt.set_blocking_sync`, registered memory portable like `cudaHostRegisterPortable`.
_CU_CTX_SCHED_BLOCKING_SYNC = 0x04
_CU_MEMHOSTREGISTER_PORTABLE = 0x01


class LeanPool:
    """`EarlyPool` with the payloads in a shared buffer and I/O in C (`knee.data.lean`): same interface
    (`studies`, `batches`, `close`, `stats`), bit-identical payloads (the first `check` studies of every
    worker are compared with `build_study_raw_np`).
    """

    def __init__(self, root, series_dir: str, uids: list, slots: dict, spec, workers: int = 5,
                 ring_mb: int = 4096, check: int | None = None, log=print):
        import mmap
        import multiprocessing as mp
        if check is None:
            check = int(os.environ.get("KNEE_LEAN_CHECKS", "0"))
        if not 1 <= workers < _NXT:
            raise ValueError(f"workers must be between 1 and {_NXT - 1}")
        if ring_mb <= 0:
            raise ValueError("ring_mb must be positive")
        self.root, self.series_dir = str(root), series_dir
        self.uids, self.slots, self.spec = list(uids), slots, spec
        self.log = log
        self.t_start = time.time()
        self.tag = f"{os.getpid()}-{int(self.t_start * 1000) % 10**8}"
        from . import cfast
        self.decode_jit = cfast.load()
        self.ring_size = max(64 << 20, (int(ring_mb) << 20) // workers // (2 << 20) * (2 << 20))
        self.mm = mmap.mmap(-1, self.ring_size * workers, flags=getattr(mmap, "MAP_SHARED", 1))
        self.ctrl_mm = mmap.mmap(-1, _PAGE, flags=getattr(mmap, "MAP_SHARED", 1))
        self.ctrl = np.frombuffer(self.ctrl_mm, np.int64, count=64)
        self.ctrl[:] = 0
        self.ctrl[_NXT] = -1
        self.base = np.frombuffer(self.mm, np.uint8)
        ctx = mp.get_context("fork")
        self.counter, self.lock = ctx.Value("i", 0, lock=False), ctx.Lock()
        self.res_q = ctx.Queue()
        self.procs = [ctx.Process(target=_ring_worker, name=f"knee-lean-{k}", daemon=True,
                                  args=(k, self.tag, self.root, series_dir, self.uids, slots, spec, self.counter,
                                        self.lock, self.res_q, self.mm, self.ring_size, self.ctrl_mm, check))
                      for k in range(workers)]
        for p in self.procs:
            p.start()
        self.stats = {"workers": workers, "ring_mb": self.ring_size * workers >> 20, "parent_decoded": 0,
                      "shm_failed": 0, "registered": False, "worker_stats": {}}
        self._reg = None

    # --- pinned memory -------------------------------------------------------------------------------
    def register_cuda(self) -> None:
        """The rings registered as CUDA pinned memory (portable: all GPUs), in a thread: from then on copies
        to the GPU start without going through a torch pinned buffer. Must be called with CUDA initialised;
        until it finishes `raw_crop._pinned` copies as before.

        Through the driver with ctypes (`_register_driver`, GIL released) if `KNEE_REGISTER_VIA` is
        `driver` (the default), then `torch.cuda.cudart()` if the driver path fails.
        `torch.cuda.cudart().cudaHostRegister` is pybind and holds the GIL: ~0.6 s per ring with the main
        process blocked (while loading the checkpoints). It is called after importing torch (from
        `run_submission`), so it overlaps with attaching the engines, which waits for the GPU with the CPU
        free. The final submission sets `KNEE_REGISTER_VIA=off` and copies through `pin_memory`.
        """
        import threading
        if self._reg is not None:
            return
        if os.environ.get("KNEE_REGISTER_VIA", "driver") == "off":
            # measured: ~8 s anyway (CPU during the import, or the driver lock afterwards): copies go through
            # `pin_memory`
            self.stats["register_via"] = "off"
            return
        via_driver = os.environ.get("KNEE_REGISTER_VIA", "driver") == "driver"

        def run() -> None:
            try:                                  # pinning costs kernel CPU: behind everything else
                os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
            except (AttributeError, OSError):
                pass
            if via_driver and self._register_driver(blocking_flags=False):
                return
            try:
                import torch
                if not torch.cuda.is_available():
                    return
                rt = torch.cuda.cudart()
                t, ok = time.time(), 0
                for k in range(len(self.procs)):
                    ptr = self.base.ctypes.data + k * self.ring_size
                    r = rt.cudaHostRegister(ptr, self.ring_size, 1)      # cudaHostRegisterPortable
                    ok += int(int(getattr(r, "value", r)) == 0)
                self.stats["registered"] = ok == len(self.procs)
                self.stats["register_s"] = round(time.time() - t, 2)
                self.stats["register_via"] = "torch"
            except Exception as exc:                                     # noqa: BLE001
                self.stats["register_error"] = repr(exc)[:200]

        self._reg = threading.Thread(target=run, name="knee-ring-register", daemon=True)
        self._reg.start()

    def _register_driver(self, blocking_flags: bool) -> bool:
        """The rings registered through the driver with ctypes (`cuMemHostRegister`, portable flag like
        `cudaHostRegisterPortable`) in the primary context of GPU 0; ctypes releases the GIL for the whole
        call. `blocking_flags`: first the blocking sync of `trt.set_blocking_sync` on every GPU (needed if
        the context is created here, before torch). True on success; otherwise the error stays in `stats`.
        """
        try:
            import ctypes

            from ..cudart import _check, driver, primary_context
            t = time.time()
            cu = driver()
            n = ctypes.c_int()
            _check(cu, cu.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
            if n.value < 1:
                raise RuntimeError("no GPU")
            if blocking_flags:
                set_flags = getattr(cu, "cuDevicePrimaryCtxSetFlags_v2", None) or cu.cuDevicePrimaryCtxSetFlags
                for i in range(n.value):
                    dev = ctypes.c_int()
                    _check(cu, cu.cuDeviceGet(ctypes.byref(dev), i), "cuDeviceGet")
                    _check(cu, set_flags(dev, ctypes.c_uint(_CU_CTX_SCHED_BLOCKING_SYNC)),
                           "cuDevicePrimaryCtxSetFlags")
            primary_context(0)
            reg = cu.cuMemHostRegister_v2
            reg.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            reg.restype = ctypes.c_int
            self.stats["context_s"] = round(time.time() - t, 2)
            t, ok = time.time(), 0
            for k in range(len(self.procs)):
                ptr = self.base.ctypes.data + k * self.ring_size
                ok += int(reg(ctypes.c_void_p(ptr), ctypes.c_size_t(self.ring_size),
                              _CU_MEMHOSTREGISTER_PORTABLE) == 0)
            self.stats["registered"] = ok == len(self.procs)
            self.stats["register_s"] = round(time.time() - t, 2)
            self.stats["register_via"] = "driver"
            return True
        except Exception as exc:                                         # noqa: BLE001
            self.stats["register_driver_error"] = repr(exc)[:200]
            return False

    def register_cuda_early(self) -> None:
        """`register_cuda` through the driver right after the fork and before `import torch`
        (`KNEE_EARLY_REGISTER=on`): the primary context of GPU 0 is created here, with blocking sync on
        all GPUs (torch then picks it up, `set_blocking_sync` sets the same flags again). It works, but
        pinning a ring not yet touched costs ~8-9 s of kernel CPU right during the torch import; the
        default is `register_cuda` after the import. If the driver path fails, `_reg` returns None and
        `register_cuda` (torch) works as before.
        """
        import threading
        if self._reg is not None:
            return

        def run() -> None:
            try:
                os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
            except (AttributeError, OSError):
                pass
            if not self._register_driver(blocking_flags=True):
                self._reg = None

        self._reg = threading.Thread(target=run, name="knee-ring-register", daemon=True)
        self._reg.start()

    # --- consumption -----------------------------------------------------------------------------------
    def _view(self, k: int, start: int, layout: list) -> dict:
        out = {}
        for key, dt, shape, off in layout:
            cnt = int(np.prod(shape)) if len(shape) else 1
            out[key] = np.frombuffer(self.mm, dtype=np.dtype(dt), count=cnt,
                                     offset=k * self.ring_size + start + off).reshape(shape)
        return out

    _attach = EarlyPool._attach
    _decode_here = EarlyPool._decode_here

    def release(self, toks: list) -> None:
        """The ring tails advanced up to the given studies (in index order: per worker, in arrival order)."""
        for t in toks:
            if t is not None:
                k, end = t
                if end > int(self.ctrl[k]):
                    self.ctrl[k] = end

    def _studies_tok(self):
        n, nxt, got = len(self.uids), 0, {}
        while nxt < n:
            self.ctrl[_NXT] = nxt
            while nxt not in got:
                try:
                    msg = self.res_q.get(timeout=5.0)
                except _queue.Empty:
                    if all(not p.is_alive() for p in self.procs) or any(
                            p.exitcode not in (None, 0) for p in self.procs):
                        self.log(f"!! lean: workers died {[p.exitcode for p in self.procs]}: "
                                 f"study {nxt} is decoded by the main process")
                        p_, m_, dt_, ne_ = self._decode_here(nxt)
                        got[nxt] = (p_, m_, dt_, ne_, None)
                    continue
                kind = msg[0]
                if kind == "ring":
                    _, i, k, start, layout, mask_b, dt, ne, end = msg
                    got[i] = (self._view(k, start, layout), np.frombuffer(mask_b, dtype=bool).copy(), dt, ne,
                              (k, end))
                elif kind == "shm":
                    _, i, name, layout, mask_b, dt, ne = msg
                    if name is None:
                        self.stats["shm_failed"] += 1
                        p_, m_, dt_, ne_ = self._decode_here(i)
                        got[i] = (p_, m_, dt_, ne_, None)
                    else:
                        got[i] = (self._attach(name, layout), np.frombuffer(mask_b, dtype=bool).copy(), dt, ne,
                                  None)
                elif kind == "warn":
                    self.log(f"!! lean worker {msg[1]}: {msg[2]}")
                elif kind == "stats":
                    self.stats["worker_stats"][msg[1]] = msg[2]
            payload, mask, dt, ne, tok = got.pop(nxt)
            yield nxt, payload, mask, dt, ne, tok
            nxt += 1
        self.ctrl[_NXT] = -1

    def studies(self):
        """(i, numpy payload, mask, dt, errs) in index order; the memory of a study is released when the next
        one is requested (whoever keeps it longer uses `batches`).
        """
        for i, payload, mask, dt, ne, tok in self._studies_tok():
            try:
                yield i, payload, mask, dt, ne
            finally:
                self.release([tok])

    def batches(self, batch_size: int):
        """The batches of `raw_crop.collate_raw`; the memory of a batch is released when the next one is
        requested (the preparation thread requests it after copying and synchronising the previous batch).
        """
        import torch
        from .raw_crop import payload_to_torch
        self.register_cuda()
        reg = self._reg                       # the read-ahead thread may set it back to None if it fails
        if reg is not None:
            reg.join(timeout=5.0)
        buf, toks = [], []
        for i, payload, mask, dt, ne, tok in self._studies_tok():
            buf.append((payload_to_torch(payload), torch.from_numpy(mask), i, dt, ne))
            toks.append(tok)
            if len(buf) == batch_size:
                yield _collate(buf)
                self.release(toks)
                buf, toks = [], []
        if buf:
            yield _collate(buf)
            self.release(toks)
        self.close()

    def close(self) -> None:
        deadline = time.time() + 2.0
        while time.time() < deadline and len(self.stats["worker_stats"]) < len(self.procs):
            try:
                msg = self.res_q.get(timeout=0.1)
            except _queue.Empty:
                if all(not p.is_alive() for p in self.procs):
                    break
                continue
            if msg[0] == "stats":
                self.stats["worker_stats"][msg[1]] = msg[2]
        ws = list(self.stats["worker_stats"].values())
        for key in ("lean", "old", "shm", "checked", "mismatch", "errors"):
            self.stats[key] = sum(int(w.get(key, 0)) for w in ws)
        self.stats["wait_s"] = round(sum(float(w.get("wait_s", 0.0)) for w in ws), 1)
        EarlyPool.close(self)


def _collate(items: list) -> tuple:
    import torch
    payloads, masks, idx, dts, errs = zip(*items)
    return (list(payloads), torch.stack(masks), torch.tensor(idx, dtype=torch.int64),
            torch.tensor(dts, dtype=torch.float64), torch.tensor(errs, dtype=torch.int64))


def _mem_available_mb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def shm_ahead(want: int, per_study_mb: float = 40.0, frac: float = 0.75, keep_free_gb: float = 12.0) -> int:
    """Studies in flight that fit in `frac` of the free space of /dev/shm (~37 MB per study at 512 px),
    leaving `keep_free_gb` of RAM available (/dev/shm pages are RAM) to the main process and the
    workers. With 96 the queue filled at ~21 s; with 149 (half of /dev/shm at 48 MB) the workers sat
    idle from ~20 to ~40 s waiting for the GPU (~70 studies of decoding lost, with decoding as the
    bottleneck).
    """
    try:
        st = os.statvfs(_SHM)
        free_mb = st.f_bavail * st.f_frsize / 2**20
    except (OSError, AttributeError):
        return min(want, 96)
    n = int(free_mb * frac / per_study_mb)
    avail = _mem_available_mb()
    if avail is not None:
        n = min(n, int((avail - keep_free_gb * 1024) / per_study_mb))
    return max(16, min(want, n))


def start(root, series_dir: str, uids: list, slots: dict, spec, workers: int = 6, ahead: int = 96,
          log=print, lean: bool = False, ring_mb: int = 4096):
    if "torch" in __import__("sys").modules:
        log("early: torch is already imported, forking is still allowed but start-up gains nothing")
    if lean:
        pool = LeanPool(root, series_dir, uids, slots, spec, workers=workers, ring_mb=ring_mb, log=log)
    else:
        pool = EarlyPool(root, series_dir, uids, slots, spec, workers=workers, ahead=ahead, log=log)
    _ACTIVE[0] = pool
    return pool


def ring_budget_mb(cap_mb: int = 4096, keep_free_gb: float = 10.0) -> int:
    """The memory of the `LeanPool` rings: `cap_mb`, but leaving `keep_free_gb` of RAM available to the
    main process (torch, CUDA, engines) and the workers.
    """
    avail = _mem_available_mb()
    if avail is None:
        return min(cap_mb, 4096)
    return int(max(1024, min(cap_mb, avail - keep_free_gb * 1024)))


def prefetch_files(paths: list, log=print):
    """Files read in a thread (GIL released during `os.read`) so that they are in the page cache when
    needed: checkpoints (2 x 57 MB) and engine packages on the Dataset mount, which the kernel would
    otherwise read cold at ~22 s and ~29 s with the CPU almost idle.
    """
    import threading
    paths = [str(x) for x in paths]

    def run() -> None:
        t, n = time.time(), 0
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                while True:
                    b = os.read(fd, 16 << 20)
                    if not b:
                        break
                    n += len(b)
            except OSError:
                pass
            finally:
                os.close(fd)
        log(f"early: {len(paths)} files read ahead ({n / 2**20:.0f} MB) in {time.time() - t:.1f} s")

    th = threading.Thread(target=run, name="knee-prefetch", daemon=True)
    th.start()
    return th


def _decoders_dir_on_path() -> None:
    """The (empty) folder of `dicom.unpack_wheels` at the head of `sys.path` before the fork, where the
    extraction would put it: the workers inherit it and, should a plugin ever be needed, they find it
    once `_decoders_late` has extracted it (the FileFinder re-reads a folder when it changes).
    """
    import sys
    import tempfile
    from pathlib import Path
    target = Path(tempfile.gettempdir()) / "knee-pkgs"
    target.mkdir(parents=True, exist_ok=True)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))


def _decoders_late(log) -> None:
    from ..kaggle import find_dir_with
    from .dicom import ensure_decoders
    t = time.time()
    try:
        wheels = find_dir_with("WHEELS")
        have = ensure_decoders(wheels_dir=wheels, allow_network=False) if wheels is not None else None
    except Exception as exc:                                         # noqa: BLE001
        log(f"!! early: decoders after the fork failed: {exc!r}")
        return
    log(f"early: decoders after the fork from {wheels} in {time.time() - t:.1f} s: {have}")


def start_for_kernel(split: str = "test", limit: int | None = None, offset: int = 0,
                     workers: int = 6, ahead: int = 320, pack: str | None = None,
                     log=print, lean: bool | None = None, root=None, spec=None,
                     decoders_after: bool = False):
    """From the kernel, before `import torch`: the same studies, series and slots that
    `run_submission(split=..., limit=..., offset=...)` will compute, the spec from the checkpoint, then
    the pool. `pack`: the prefix of the engine packages (`<pack>_fold<k>.npz`), read ahead together with
    the checkpoints.

    Cold start: with `root` (the competition folder) and `spec` (the checkpoint's `CacheSpec`, written
    into the kernel) before the fork, /kaggle/input is not walked and the checkpoint is not read
    (~10 s on cold datasets before the workers); with `decoders_after` the decoding plugins (never
    used: the DICOMs are all Explicit VR Little Endian) are extracted in a thread after the fork.
    `take` still compares uid, slots, spec and root with those of `run_submission`: if they do not
    match the pool is not used.
    """
    import threading
    from pathlib import Path

    from . import cfast
    # The C library compiles (gcc, ~1.5-2 s) while pandas is imported and the slots are computed here;
    # `EarlyPool` loads it before the fork, after the join.
    build = threading.Thread(target=cfast.load, name="knee-cfast-build", daemon=True)
    build.start()

    import pandas as pd

    from ..constants import ID
    from ..kaggle import find_dir_with, find_test_root
    from .cache import pick_slots
    from .dicom import ensure_decoders
    t = time.time()
    if decoders_after:
        _decoders_dir_on_path()
    else:
        wheels = find_dir_with("WHEELS")
        if wheels is not None:
            ensure_decoders(wheels_dir=wheels, allow_network=False)   # the workers inherit them
    if root is not None and not (Path(root) / f"{split}.csv").is_file():
        log(f"!! early: no {split}.csv in {root}: searching for the competition")
        root = None
    root = find_test_root() if root is None else Path(root)
    if spec is None:
        ckpt = find_dir_with("fold0.pt", extra=[Path("runs"), Path(".")])
        if ckpt is None:
            build.join()
            log("!! early: no fold0.pt, staying on the DataLoader")
            return None
        ahead_files = sorted(ckpt.glob("fold*.pt"))
        if pack:
            pdir = find_dir_with(f"{pack}_fold0.npz")
            if pdir is not None:
                ahead_files += sorted(pdir.glob(f"{pack}_fold*.npz"))
        prefetch_files(ahead_files, log=log)
        spec = spec_from_checkpoint(ckpt / "fold0.pt")
    series_dir = f"{split}_series"
    uids = pd.read_csv(root / f"{split}.csv", usecols=[ID])[ID].tolist()
    series = pd.read_csv(root / f"{series_dir}.csv")
    if limit is not None:
        uids = uids[offset:offset + limit]
        series = series[series[ID].isin(set(uids))].reset_index(drop=True)
    dirs = {r.SeriesInstanceUID: root / series_dir / getattr(r, ID) / r.SeriesInstanceUID
            for r in series.itertuples(index=False)}
    # 64 threads to count the files of the series under contention (~2 s with 16, before the fork)
    slots = pick_slots(series, series_dirs=dirs, slots=spec.active_slots, threads=64)
    build.join()
    if lean is None:
        lean = os.environ.get("KNEE_LEAN", "off") == "on" and cfast.lio_ok()
    if lean:
        workers = int(os.environ.get("KNEE_LEAN_WORKERS", "5"))
        pool = start(root, series_dir, uids, slots, spec, workers=workers, log=log, lean=True,
                     ring_mb=ring_budget_mb(int(os.environ.get("KNEE_LEAN_RING_MB", "4096"))))
        if os.environ.get("KNEE_EARLY_REGISTER", "off") == "on":
            pool.register_cuda_early()       # rings pinned through the driver while torch is imported
        log(f"early: {workers} lean workers on {len(uids)} studies (rings {pool.stats['ring_mb']} MB), started "
            f"in {time.time() - t:.1f} s (spec {spec.tag}, C {pool.decode_jit.get('ok')})")
    else:
        ahead = shm_ahead(ahead)
        pool = start(root, series_dir, uids, slots, spec, workers=workers, ahead=ahead, log=log)
        log(f"early: {workers} workers on {len(uids)} studies ({ahead} in flight), started in {time.time() - t:.1f} s "
            f"(spec {spec.tag}, C {pool.decode_jit.get('ok')})")
    if decoders_after:
        threading.Thread(target=_decoders_late, args=(log,), name="knee-decoders", daemon=True).start()
    return pool


def take(ds):
    """The pool started for the same studies as `ds` (same uids, slots and spec), removed from the
    registry; or None.
    """
    pool = _ACTIVE[0]
    if pool is None:
        return None
    _ACTIVE[0] = None
    same = (list(ds.uids) == pool.uids and ds.slots == pool.slots and ds.spec == pool.spec
            and str(ds.root) == pool.root and ds.series_dir == pool.series_dir)
    if not same:
        pool.log("!! early: the pool does not match the predict dataset: using the DataLoader")
        pool.close()
        return None
    return pool


def active():
    return _ACTIVE[0]
