"""The C decoding library (`fastdecode.c`), loaded with ctypes.

Profiling the CPU of the inference workers: `_to_uint8` is 17%, parsing the headers in Python
(two walks over the DICOM elements per file) ~9%, converting the pixels to float32 with the bit
mask ~3%. Here the same computations in C, bit-identical: compiled without FMA and without
fast-math; callers compare the first calls of every process with the previous path and fall back
at the first disagreement.

It is compiled on the first `load()` with the system C compiler (`$CC`, `gcc`, `cc`) into a
temporary folder named after the hash of the source, or with `python -m ziglang cc` where
available (Windows, development). Half a second, no JIT: it is called in the main process and the
forked workers inherit the library. Without a compiler `ok()` is False and everything stays as it
was.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).with_name("fastdecode.c")
# Lean worker I/O (`leanio.c`) in the same library, where the source is present.
LEANIO = Path(__file__).with_name("leanio.c")
FLAGS = ["-O2", "-ffp-contract=off", "-fno-fast-math", "-shared"]
_STATE: dict = {"tried": False, "lib": None, "error": None, "build_s": None, "how": None}
_LOCK = threading.Lock()

_i64 = ctypes.c_int64
_p = ctypes.c_void_p


def _compilers() -> list[list[str]]:
    out = []
    for cc in (os.environ.get("CC"), "gcc", "cc"):
        if cc and shutil.which(cc):
            out.append([cc])
    try:
        import ziglang  # noqa: F401
        out.append([sys.executable, "-m", "ziglang", "cc"])
    except ImportError:
        pass
    return out


def _sources() -> list[Path]:
    return [SRC] + ([LEANIO] if LEANIO.is_file() else [])


def _build() -> Path:
    srcs = _sources()
    code = b"".join(p.read_bytes() for p in srcs)
    tag = hashlib.sha1(code + " ".join(FLAGS).encode()).hexdigest()[:12]
    ext = ".dll" if sys.platform == "win32" else ".so"
    lib = Path(tempfile.gettempdir()) / f"knee_fastdecode_{tag}{ext}"
    if lib.is_file():
        _STATE["how"] = "already compiled"
        return lib
    errs = []
    for cc in _compilers():
        tmp = lib.with_suffix(f".{os.getpid()}{ext}")
        cmd = [*cc, *FLAGS, *([] if sys.platform == "win32" else ["-fPIC", "-pthread"]), "-o", str(tmp),
               *map(str, srcs)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and tmp.is_file():
            os.replace(tmp, lib)
            _STATE["how"] = " ".join(cc)
            return lib
        errs.append(f"{' '.join(cc)}: {(r.stderr or r.stdout)[-300:]}")
    raise RuntimeError("no C compiler: " + " | ".join(errs) if errs
                       else "no C compiler found")


def load() -> dict:
    """Compiles (if needed) and loads the library, once per process; returns the state. Thread-safe:
    whoever arrives while another thread is compiling (`early.start_for_kernel`) waits for its outcome.
    """
    with _LOCK:
        return _load()


def _load() -> dict:
    if _STATE["tried"]:
        return {k: _STATE[k] for k in ("error", "build_s", "how")} | {"ok": ok()}
    _STATE["tried"] = True
    t = time.perf_counter()
    try:
        lib = ctypes.CDLL(str(_build()))
        lib.knee_abi.restype = _i64
        if lib.knee_abi() != 1:
            raise RuntimeError("library ABI mismatch")
        lib.tails_f32.argtypes = [_p, _i64, ctypes.c_float, ctypes.c_float, _p, _i64, _p, _i64, _p]
        lib.tails_f32.restype = None
        lib.normalize_u8.argtypes = [_p, _i64, _i64, _i64, ctypes.c_double, ctypes.c_double, _p,
                                     _i64, _i64, _i64]
        lib.normalize_u8.restype = None
        lib.pix_to_f32.argtypes = [_p, _i64, _i64, _i64, _i64, _p]
        lib.pix_to_f32.restype = None
        lib.walk_order.argtypes = [ctypes.c_char_p, _i64, _p]
        lib.walk_order.restype = _i64
        lib.walk_pixels.argtypes = [ctypes.c_char_p, _i64, _p]
        lib.walk_pixels.restype = _i64
        lib.resize_area_f32.argtypes = [_p, _i64, _i64, _i64, _p, _i64, _i64, _i64]
        lib.resize_area_f32.restype = _i64
        try:
            lib.scan_prefix.argtypes = [ctypes.c_char_p, _p, _i64, _i64, _p, _p, _p, _i64, _p, _p]
            lib.scan_prefix.restype = _i64
            _STATE["scan"] = True
        except AttributeError:
            _STATE["scan"] = False
        lib.knee_set_simd.argtypes = [_i64]
        lib.knee_set_simd.restype = None
        try:
            lib.lio_abi.restype = _i64
            lib.lio_task_fields.restype = _i64
            lib.lio_listdir.argtypes = [ctypes.c_char_p, _p, _i64, _p, _i64, _p, _i64, _p, _i64]
            lib.lio_listdir.restype = _i64
            lib.lio_scan.argtypes = [ctypes.c_char_p, _p, _i64, _i64, _p, _p, _p, _i64, _p, _p, _p, _p, _i64]
            lib.lio_scan.restype = _i64
            lib.lio_read.argtypes = [_p, _i64, _i64, _p, _p, _p, _i64]
            lib.lio_read.restype = _i64
            lib.lio_close.argtypes = [_p, _i64]
            lib.lio_close.restype = None
            _STATE["lio"] = lib.lio_abi() == 1 and lib.lio_task_fields() == LIO_T_N
        except AttributeError:
            _STATE["lio"] = False
        _STATE["lib"] = lib
    except Exception as exc:
        _STATE["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    _STATE["build_s"] = round(time.perf_counter() - t, 2)
    return {k: _STATE[k] for k in ("error", "build_s", "how")} | {"ok": ok()}


def ok() -> bool:
    return _STATE["lib"] is not None


def tails(flat: np.ndarray, thr_lo, thr_hi, low: np.ndarray, high: np.ndarray) -> tuple[int, int]:
    """`flat` contiguous float32; `low`, `high` contiguous float32 (capacity = their size)."""
    counts = np.zeros(2, np.int64)
    _STATE["lib"].tails_f32(flat.ctypes.data, flat.size, float(thr_lo), float(thr_hi),
                            low.ctypes.data, low.size, high.ctypes.data, high.size,
                            counts.ctypes.data)
    return int(counts[0]), int(counts[1])


def normalize(stack: np.ndarray, lo, scale, out: np.ndarray) -> None:
    """`stack` contiguous float32 (K, H, W); `out` uint8 of the same shape, any strides."""
    k, h, w = stack.shape
    s0, s1, s2 = (s // out.itemsize for s in out.strides)
    _STATE["lib"].normalize_u8(stack.ctypes.data, k, h, w, float(lo), float(scale),
                               out.ctypes.data, s0, s1, s2)


def pix_to_f32(raw: bytes, offset: int, count: int, bits: int, rep: int, shift: int) -> np.ndarray:
    """The `count` pixels at `offset` of `raw` (8 or 16 bit, little endian) as float32."""
    out = np.empty(count, np.float32)
    buf = np.frombuffer(raw, dtype=np.uint8, count=count * (bits // 8), offset=offset)
    _STATE["lib"].pix_to_f32(buf.ctypes.data, count, bits, rep, shift, out.ctypes.data)
    return out


def set_simd(on: bool) -> None:
    """AVX/AVX2 vectors on (default) or scalar code only: the tests compare the two."""
    _STATE["lib"].knee_set_simd(1 if on else 0)


def resize_area(a: np.ndarray, dh: int, dw: int, out: np.ndarray) -> int:
    """`cv2.resize(a, (dw, dh), interpolation=cv2.INTER_AREA)` bit for bit, written into `out`
    (float32 (dh, dw), contiguous rows). `a` 2D float32 with contiguous rows, possibly a cropped
    view. 0 = done; 1 (upscaling) or 2 (integer scale): not OpenCV's generic path, the caller uses
    cv2; -2 = unsupported shapes or strides.
    """
    if (a.dtype != np.float32 or a.ndim != 2 or a.strides[1] != 4 or a.strides[0] % 4
            or out.dtype != np.float32 or out.shape != (dh, dw) or out.strides[1] != 4
            or out.strides[0] % 4):
        return -2
    return int(_STATE["lib"].resize_area_f32(a.ctypes.data, a.shape[0], a.shape[1],
                                             a.strides[0] // 4, out.ctypes.data, dh, dw,
                                             out.strides[0] // 4))


def walk_order(buf: bytes) -> list[int] | None:
    """[ipp_off, ipp_len, iop_off, iop_len, inst_off, inst_len] (-1 = absent), or None where the C walk
    stops (the caller redoes it in Python).
    """
    out = np.empty(6, np.int64)
    if _STATE["lib"].walk_order(buf, len(buf), out.ctypes.data) != 0:
        return None
    return out.tolist()


def walk_pixels(buf: bytes) -> list[int] | None:
    """[pix_off, pix_len, (kind, a, b) x 10 tags], or None (the caller redoes it in Python)."""
    out = np.empty(32, np.int64)
    if _STATE["lib"].walk_pixels(buf, len(buf), out.ctypes.data) != 0:
        return None
    return out.tolist()


def scan_prefix(paths: list[str], cap: int, keep: bool, threads: int = 8):
    """`fastdecode.scan_prefix` on chunks of `paths` in parallel (ctypes releases the GIL):
    (heads (n, cap) uint8, head_len (n,), fds (n,), offs (n, 6), walk_rc (n,)) or None.
    """
    if _STATE["lib"] is None or not _STATE.get("scan"):
        return None
    n = len(paths)
    heads = np.empty((n, cap), np.uint8)
    head_len = np.empty(n, np.int64)
    # -1 and not `np.empty`: if a chunk does not run (the Windows stub returns -2 without touching
    # anything), below we close only the descriptors that were actually opened, not random numbers of
    # the process.
    fds = np.full(n, -1, np.int64)
    offs = np.empty((n, 6), np.int64)
    rc = np.empty(n, np.int64)
    enc = [os.fsencode(p) + b"\x00" for p in paths]
    off = np.zeros(n, np.int64)
    if n > 1:
        off[1:] = np.cumsum([len(e) for e in enc[:-1]])
    blob = b"".join(enc)
    k = max(1, min(int(threads), n))
    bounds = [(n * j // k, n * (j + 1) // k) for j in range(k)]

    def run(b):
        a, z = b
        if z <= a:
            return 0
        return _STATE["lib"].scan_prefix(blob, off[a:].ctypes.data, z - a, cap, heads[a:].ctypes.data,
                                         head_len[a:].ctypes.data, fds[a:].ctypes.data, int(bool(keep)),
                                         offs[a:].ctypes.data, rc[a:].ctypes.data)
    if k == 1:
        r = [run(bounds[0])]
    else:
        from .pools import thread_pool
        with thread_pool("order", k) as ex:
            r = list(ex.map(run, bounds))
    if any(v != 0 for v in r):
        for f in fds:
            if f >= 0:
                os.close(int(f))
        return None
    return heads, head_len, fds, offs, rc


# --- lean worker I/O (`leanio.c`) -------------------------------------------------------------------
# Fields of a `lio_read` task (T_* enum of leanio.c) and status codes.
(LIO_FD, LIO_PIX_OFF, LIO_PIX_LEN, LIO_RESUME, LIO_ROWS, LIO_COLS, LIO_BITS, LIO_REP, LIO_SHIFT,
 LIO_Y0, LIO_Y1, LIO_X0, LIO_X1, LIO_DEST, LIO_DIRECT, LIO_T_N) = range(16)
LIO_OK, LIO_VALUES, LIO_IO, LIO_SHORT, LIO_HEADER, LIO_PARAM = range(6)
_LIO_BUF: dict = {}


def lio_ok() -> bool:
    return _STATE["lib"] is not None and bool(_STATE.get("lio"))


def _lio_array(name: str, shape: tuple, dtype) -> np.ndarray:
    """A per-process buffer reused across calls (no page faults on every study): the returned view is
    valid until the next call with the same name.
    """
    need = int(np.prod(shape))
    cur = _LIO_BUF.get(name)
    if cur is None or cur.size < need or cur.dtype != np.dtype(dtype):
        cur = np.empty(max(need, 1) * (2 if cur is not None else 1), dtype)
        _LIO_BUF[name] = cur
    return cur[:need].reshape(shape)


def _blob(strings: list[str]) -> tuple[bytes, np.ndarray]:
    enc = [os.fsencode(s) + b"\x00" for s in strings]
    off = np.zeros(len(enc), np.int64)
    if len(enc) > 1:
        off[1:] = np.cumsum([len(e) for e in enc[:-1]])
    return b"".join(enc), off


def lio_listdir(dirs: list[str], threads: int = 8) -> list[list[str] | None]:
    """Per folder: the names as `os.listdir` (unordered), `[]` if the folder does not exist or is not a
    folder (like `os.path.isdir` false), None for any other error (the caller falls back to the
    usual path).
    """
    blob, off = _blob(dirs)
    n = len(dirs)
    counts = _lio_array("ls_counts", (n,), np.int64)
    cap, mx = 1 << 16, 1 << 10
    while True:
        names = _lio_array("ls_names", (cap,), np.uint8)
        name_off = _lio_array("ls_offsets", (mx,), np.int64)
        r = _STATE["lib"].lio_listdir(blob, off.ctypes.data, n, names.ctypes.data, cap, name_off.ctypes.data,
                                      mx, counts.ctypes.data, int(threads))
        if r >= 0:
            break
        if r != -1:
            raise RuntimeError(f"lio_listdir failed: {r}")
        cap, mx = cap * 4, mx * 4
    # A copy of the names actually used only.
    if r:
        end = int(name_off[r - 1])
        while names[end] != 0:
            end += 1
        raw = names[:end + 1].tobytes()
    else:
        raw = b""
    out: list[list[str] | None] = []
    k = 0
    for d in range(n):
        c = int(counts[d])
        if c == -1:
            out.append([])
        elif c < 0:
            out.append(None)
        else:
            lst = []
            for _ in range(c):
                a = int(name_off[k])
                lst.append(os.fsdecode(raw[a:raw.index(b"\x00", a)]))
                k += 1
            out.append(lst)
    return out


def lio_scan(paths: list[str], cap: int, keep: bool, threads: int = 16):
    """`scan_prefix` on `threads` threads plus the walk of the pixel tags (`walk_header`): (heads (n, cap),
    head_len, fds, offs (n, 6), walk_rc, hdr (n, 34), hdr_rc). Views on reused buffers: valid until the
    next call.
    """
    cap = int(cap)
    if cap <= 0:
        raise ValueError("cap must be positive")
    n = len(paths)
    heads = _lio_array("heads", (n, cap), np.uint8)
    head_len = _lio_array("head_len", (n,), np.int64)
    fds = _lio_array("fds", (n,), np.int64)
    fds[:] = -1
    offs = _lio_array("offs", (n, 6), np.int64)
    rc = _lio_array("rc", (n,), np.int64)
    hdr = _lio_array("hdr", (n, 34), np.int64)
    hdr_rc = _lio_array("hdr_rc", (n,), np.int64)
    blob, off = _blob(paths)
    status = _STATE["lib"].lio_scan(blob, off.ctypes.data, n, cap, heads.ctypes.data, head_len.ctypes.data,
                                   fds.ctypes.data, int(bool(keep)), offs.ctypes.data, rc.ctypes.data,
                                   hdr.ctypes.data, hdr_rc.ctypes.data, int(threads))
    if status != 0:
        lio_close(fds)
        raise RuntimeError(f"lio_scan failed: {status}")
    return heads, head_len, fds, offs, rc, hdr, hdr_rc


def lio_read(tasks: np.ndarray, threads: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`tasks` (n, LIO_T_N) contiguous int64 -> (status, pix_off found, bytes read) per task."""
    tasks = np.ascontiguousarray(tasks, dtype=np.int64)
    if tasks.ndim != 2 or tasks.shape[1] != LIO_T_N:
        raise ValueError(f"tasks must have shape (n, {LIO_T_N})")
    n = tasks.shape[0]
    status = np.empty(n, np.int64)
    found = np.empty(n, np.int64)
    nread = np.empty(n, np.int64)
    if n:
        rc = _STATE["lib"].lio_read(tasks.ctypes.data, n, tasks.shape[1], status.ctypes.data,
                                   found.ctypes.data, nread.ctypes.data, int(threads))
        if rc != 0:
            raise RuntimeError(f"lio_read failed: {rc}")
    return status, found, nread


def lio_close(fds: np.ndarray) -> None:
    fds = np.ascontiguousarray(fds, dtype=np.int64)
    if fds.size:
        _STATE["lib"].lio_close(fds.ctypes.data, fds.size)
