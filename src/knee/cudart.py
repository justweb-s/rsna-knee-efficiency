"""Hand-written CUDA kernels without a toolkit: NVRTC to compile, the driver API to launch.

Used for the kernels TensorRT cannot fuse. No nvcc, no PyTorch C++ extensions (ABI tied to the
version), no cupy: only `libcuda` (present wherever there is a GPU) and `libnvrtc` (shipped with
torch, pip package `nvidia-cuda-nvrtc-cu12`), called through ctypes.

- `compile_source(src, arch)`: CUDA C++ source -> cubin (SASS for `sm_XX`) or PTX
  (`compute_XX`). The source includes no headers: vector and fp16 types are written by hand.
- `Module(image, dev)`: the cubin/PTX loaded into the primary context of GPU `dev` (the one
  PyTorch uses), `module.function(name)`.
- `Function.launch(grid, block, args, stream, shared)`: `args` are torch tensors (their address
  is passed), `Ptr`, `int` (-> int32), `float` (-> float32) or `ctypes.Structure` (a kernel
  struct passed by value); `stream` is the `cuda_stream` of a torch stream. The launch is
  asynchronous like PyTorch's and is recorded into CUDA graphs when the stream is capturing.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
import threading
from pathlib import Path

_LOCK = threading.Lock()
_STATE: dict = {}


class CudaError(RuntimeError):
    pass


# --- driver ------------------------------------------------------------------------------
def driver() -> ctypes.CDLL:
    with _LOCK:
        if "cu" not in _STATE:
            name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
            cu = ctypes.CDLL(name)
            vp, u, i, sz = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_size_t
            cu.cuInit.argtypes = [u]
            cu.cuDeviceGet.argtypes = [ctypes.POINTER(i), i]
            cu.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(vp), i]
            cu.cuCtxGetCurrent.argtypes = [ctypes.POINTER(vp)]
            cu.cuCtxSetCurrent.argtypes = [vp]
            cu.cuModuleLoadData.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p]
            cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
            cu.cuLaunchKernel.argtypes = [vp, u, u, u, u, u, u, u, vp,
                                          ctypes.POINTER(vp), ctypes.POINTER(vp)]
            cu.cuFuncSetAttribute.argtypes = [vp, i, i]
            cu.cuGetErrorString.argtypes = [i, ctypes.POINTER(ctypes.c_char_p)]
            _check(cu, cu.cuInit(0), "cuInit")
            _STATE["cu"] = cu
            _STATE["sz"] = sz
        return _STATE["cu"]


def _check(cu, rc: int, what: str) -> None:
    if rc != 0:
        msg = ctypes.c_char_p()
        try:
            cu.cuGetErrorString(rc, ctypes.byref(msg))
            text = msg.value.decode() if msg.value else ""
        except Exception:                                     # noqa: BLE001
            text = ""
        raise CudaError(f"{what}: CUresult {rc} {text}")


def primary_context(dev: int) -> int:
    """The primary context of GPU `dev` (PyTorch's), made current on the calling thread."""
    cu = driver()
    d = ctypes.c_int()
    _check(cu, cu.cuDeviceGet(ctypes.byref(d), dev), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    _check(cu, cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), d), "cuDevicePrimaryCtxRetain")
    _check(cu, cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    return int(ctx.value)


class Ptr:
    """A GPU memory address passed as is (not a tensor)."""
    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = int(value)


class Module:
    def __init__(self, image: bytes, dev: int) -> None:
        cu = driver()
        self.dev = dev
        self.ctx = primary_context(dev)
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(image, len(image))
        _check(cu, cu.cuModuleLoadData(ctypes.byref(mod), buf), "cuModuleLoadData")
        self.handle = mod.value
        self._funcs: dict[str, Function] = {}

    def function(self, name: str) -> "Function":
        if name not in self._funcs:
            cu = driver()
            f = ctypes.c_void_p()
            _check(cu, cu.cuModuleGetFunction(ctypes.byref(f), ctypes.c_void_p(self.handle),
                                              name.encode()), f"cuModuleGetFunction {name}")
            self._funcs[name] = Function(f.value, name, self.ctx)
        return self._funcs[name]


CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


class Function:
    def __init__(self, handle: int, name: str, ctx: int) -> None:
        self.handle, self.name, self.ctx = handle, name, ctx

    def set_max_shared(self, nbytes: int) -> None:
        cu = driver()
        _check(cu, cu.cuFuncSetAttribute(ctypes.c_void_p(self.handle),
                                         CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                         int(nbytes)), f"cuFuncSetAttribute {self.name}")

    @staticmethod
    def _arg(a):
        if isinstance(a, ctypes.Structure):
            return a                          # passed by value (its bytes)
        if isinstance(a, (ctypes.c_void_p, ctypes.c_double, ctypes.c_float)):
            return a                          # explicit CUDA type, not converted to float32
        if isinstance(a, Ptr):
            return ctypes.c_void_p(a.value)
        if hasattr(a, "data_ptr"):
            return ctypes.c_void_p(a.data_ptr())
        if isinstance(a, bool):
            return ctypes.c_int(int(a))
        if isinstance(a, int):
            return ctypes.c_int(a)
        if isinstance(a, float):
            return ctypes.c_float(a)
        raise TypeError(f"unsupported kernel argument: {type(a).__name__}")

    def launch(self, grid, block, args, stream: int, shared: int = 0) -> None:
        cu = driver()
        vals = [self._arg(a) for a in args]
        params = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        rc = cu.cuLaunchKernel(ctypes.c_void_p(self.handle), g[0], g[1], g[2], b[0], b[1], b[2],
                               int(shared), ctypes.c_void_p(int(stream)), params, None)
        if rc != 0:
            # The thread may have no current context (a new thread): set it and retry.
            cu.cuCtxSetCurrent(ctypes.c_void_p(self.ctx))
            rc = cu.cuLaunchKernel(ctypes.c_void_p(self.handle), g[0], g[1], g[2], b[0], b[1],
                                   b[2], int(shared), ctypes.c_void_p(int(stream)), params, None)
        _check(cu, rc, f"cuLaunchKernel {self.name}")


# --- NVRTC -------------------------------------------------------------------------------
def _nvrtc_candidates() -> list[str]:
    pats = []
    for base in sys.path:
        if not base or not os.path.isdir(base):
            continue
        if sys.platform == "win32":
            pats.append(os.path.join(base, "nvidia", "cuda_nvrtc", "bin", "nvrtc64_*.dll"))
        else:
            pats.append(os.path.join(base, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*"))
            pats.append(os.path.join(base, "torch", "lib", "libnvrtc*.so*"))
    if sys.platform != "win32":
        pats += ["/usr/local/cuda/lib64/libnvrtc.so*", "/usr/local/cuda*/lib64/libnvrtc.so*",
                 "/usr/lib/x86_64-linux-gnu/libnvrtc.so*"]
    out = []
    for p in pats:
        for f in sorted(glob.glob(p)):
            if "builtins" in os.path.basename(f) or "alt" in os.path.basename(f):
                continue
            out.append(f)
    return out


def nvrtc() -> ctypes.CDLL:
    with _LOCK:
        if "nvrtc" in _STATE:
            return _STATE["nvrtc"]
        errs = []
        for path in _nvrtc_candidates():
            try:
                # The builtins library sits next to it: loaded first, globally, because nvrtc opens it by name.
                d = os.path.dirname(path)
                for b in sorted(glob.glob(os.path.join(d, "libnvrtc-builtins*.so*"))
                                + glob.glob(os.path.join(d, "nvrtc-builtins64_*.dll"))):
                    try:
                        ctypes.CDLL(b, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                    except OSError:
                        pass
                lib = ctypes.CDLL(path)
                vp, i, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
                lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p,
                                                   ctypes.c_char_p, i, vp, vp]
                lib.nvrtcCompileProgram.argtypes = [vp, i, ctypes.POINTER(ctypes.c_char_p)]
                lib.nvrtcGetProgramLogSize.argtypes = [vp, ctypes.POINTER(sz)]
                lib.nvrtcGetProgramLog.argtypes = [vp, ctypes.c_char_p]
                lib.nvrtcGetCUBINSize.argtypes = [vp, ctypes.POINTER(sz)]
                lib.nvrtcGetCUBIN.argtypes = [vp, ctypes.c_char_p]
                lib.nvrtcGetPTXSize.argtypes = [vp, ctypes.POINTER(sz)]
                lib.nvrtcGetPTX.argtypes = [vp, ctypes.c_char_p]
                lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(vp)]
                lib.nvrtcVersion.argtypes = [ctypes.POINTER(i), ctypes.POINTER(i)]
                lib.nvrtcGetErrorString.argtypes = [i]
                lib.nvrtcGetErrorString.restype = ctypes.c_char_p
                _STATE["nvrtc"] = lib
                _STATE["nvrtc_path"] = path
                return lib
            except (OSError, AttributeError) as exc:
                errs.append(f"{path}: {exc}")
        raise CudaError("libnvrtc not found" + (f" ({errs[:3]})" if errs else ""))


def nvrtc_version() -> tuple[int, int]:
    lib = nvrtc()
    a, b = ctypes.c_int(), ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(a), ctypes.byref(b))
    return a.value, b.value


def compile_source(src: str, arch: str = "sm_75", name: str = "knee.cu",
                   options: tuple[str, ...] = ()) -> bytes:
    """`arch` "sm_75" -> cubin (SASS), "compute_75" -> PTX. Raises with the compiler log."""
    lib = nvrtc()
    prog = ctypes.c_void_p()

    def ok(rc: int, what: str) -> None:
        if rc != 0:
            raise CudaError(f"{what}: {lib.nvrtcGetErrorString(rc).decode()}")

    ok(lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), name.encode(), 0, None, None),
       "nvrtcCreateProgram")
    try:
        opts = [f"--gpu-architecture={arch}", "--std=c++14", *options]
        arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
        n = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(max(1, n.value))
        lib.nvrtcGetProgramLog(prog, log)
        if rc != 0:
            raise CudaError(f"nvrtc: compilation failed\n{log.value.decode(errors='replace')}")
        if arch.startswith("sm_"):
            lib.nvrtcGetCUBINSize(prog, ctypes.byref(n))
            out = ctypes.create_string_buffer(n.value)
            ok(lib.nvrtcGetCUBIN(prog, out), "nvrtcGetCUBIN")
        else:
            lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
            out = ctypes.create_string_buffer(n.value)
            ok(lib.nvrtcGetPTX(prog, out), "nvrtcGetPTX")
        return bytes(out.raw[:n.value])
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))


def load_module(src_path: Path | str, dev: int, arch: str = "sm_75",
                cubin: bytes | None = None) -> tuple[Module, dict]:
    """The kernel module of `src_path` on GPU `dev`: from the precompiled `cubin` if given and
    loadable, otherwise compiled here with NVRTC. Returns (module, info).
    """
    import time
    info: dict = {}
    if cubin is not None:
        try:
            m = Module(cubin, dev)
            info["from"] = "cubin"
            return m, info
        except CudaError as exc:
            info["cubin_error"] = str(exc)[:200]
    t = time.perf_counter()
    image = compile_source(Path(src_path).read_text(encoding="utf-8"), arch)
    info.update({"from": "nvrtc", "compile_s": round(time.perf_counter() - t, 3),
                 "nvrtc": _STATE.get("nvrtc_path")})
    return Module(image, dev), info
