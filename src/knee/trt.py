"""TensorRT for the encoder: engine building and a runner on one GPU.

TensorRT fp16 gives 1.20x over eager per T4, INT8 2.79x; post-training quantization with `max`
scales costs -0.0003 AUC on a validation fold. The INT8 engine is built from the explicitly
quantized ONNX of `knee.quant.quantized_copy` (Q/DQ on input and weights of every conv), not with
TensorRT's implicit calibrator (entropy calibration moved the predictions: Spearman 0.980).

Lessons learned the hard way:
- building is slow (46 min for 352 fp16 images, 89 min implicit INT8 at 88): build once on the
  same T4 and TensorRT as the rerun and load it (0.02 s);
- the engine is for one study (88 images); a batch of several studies is passed in blocks;
- never on the default stream (TensorRT synchronises at every enqueue and the two T4s run in
  sequence);
- the workspace memory belongs to the context (STATIC strategy): resetting it at every call made
  TensorRT scale 1.18x on two GPUs against 1.97x for eager.
The `tensorrt` import is lazy: the rest of `knee` does not depend on it. In the final submission
TensorRT runs only the stem (fp16); the dense blocks run on `knee.dense8s`.
"""
from __future__ import annotations

import collections
import threading
import time
from pathlib import Path

import torch

_STATE: dict = {}


def trt_module():
    import tensorrt as trt
    return trt


def logger():
    trt = trt_module()
    if "logger" not in _STATE:
        _STATE["logger"] = trt.Logger(trt.Logger.WARNING)
    return _STATE["logger"]


def minmax_calibrator(batches: list[torch.Tensor], dev: int = 0):
    """Implicit INT8 calibrator with max scaling (`IInt8MinMaxCalibrator`), on all tensors.
    `batches`: uint8 windows on CPU, one shape per batch (the engine's); converted to float [0, 1] on
    the GPU one at a time.
    """
    trt = trt_module()

    class Calib(trt.IInt8MinMaxCalibrator):
        def __init__(self) -> None:
            super().__init__()
            self.i = 0
            self.buf = torch.empty(batches[0].shape, dtype=torch.float32, device=f"cuda:{dev}")

        def get_batch_size(self) -> int:
            return int(batches[0].shape[0])

        def get_batch(self, names):                                   # noqa: ANN001
            if self.i >= len(batches):
                return None
            self.buf.copy_(batches[self.i].to(self.buf.device).float().div_(255.0))
            torch.cuda.synchronize(dev)
            self.i += 1
            return [int(self.buf.data_ptr())]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache) -> None:             # noqa: ANN001
            self.cache = bytes(cache)

    return Calib()


def build_engine(onnx_path: Path, n: int, depth: int, img: int, int8: bool = True,
                 timing_cache: bytes | None = None, log=print,
                 workspace_gb: int = 4, calibrator=None,
                 detailed: bool = False, half: bool = False) -> tuple[bytes, bytes, dict]:
    """Fixed-shape engine (n, depth, img, img), fp16 plus INT8: where the ONNX has Q/DQ nodes
    (explicit) or, with `calibrator`, on all tensors (implicit, calibration at `n`).
    Returns (engine, updated timing cache, info). One log line every 5 minutes.
    `detailed`: the engine also stores type, formats and tactic of every layer (`layer_times`,
    inspector); metadata only, the chosen kernels do not change.
    `half`: fp16 input and, outside the Q/DQ nodes, every layer preferably in fp16 ("prefer"
    precision constraints). Without it the chained engine keeps the stem, the max-pool and the whole
    concatenation of the first block in float32.
    """
    trt = trt_module()
    builder = trt.Builder(logger())
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger())
    if not parser.parse_from_file(str(onnx_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX not parsed by TensorRT: {errs[:3]}")
    cfg = builder.create_builder_config()
    if detailed:
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache = cfg.create_timing_cache(timing_cache or b"")
    cfg.set_timing_cache(cache, True)   # cache from another machine: used anyway (timings, not correctness)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
    if half:
        net.get_input(0).dtype = trt.float16
        skip = (trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE, trt.LayerType.CONSTANT,
                trt.LayerType.SHAPE)
        for i in range(net.num_layers):
            layer = net.get_layer(i)
            if layer.type in skip:
                continue
            outs = [layer.get_output(k) for k in range(layer.num_outputs)]
            if all(o.dtype == trt.float32 for o in outs):
                layer.precision = trt.float16
                for k in range(layer.num_outputs):
                    layer.set_output_type(k, trt.float16)
        cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    shape = (n, depth, img, img)
    prof = builder.create_optimization_profile()
    prof.set_shape(net.get_input(0).name, shape, shape, shape)
    cfg.add_optimization_profile(prof)
    if calibrator is not None:
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = calibrator
        cfg.set_calibration_profile(prof)
    t = time.perf_counter()
    done = threading.Event()

    def beat() -> None:
        while not done.wait(300):
            log(f"  building for {time.perf_counter() - t:.0f} s")

    threading.Thread(target=beat, daemon=True).start()
    kind = ("implicit INT8 (calibrator) + fp16" if calibrator is not None
            else "explicit INT8 + fp16" if int8 else "fp16") + (" (tutto fp16)" if half else "")
    log(f"  build started at {time.strftime('%H:%M:%S', time.gmtime())} UTC, shape "
        f"{shape}, {kind}")
    try:
        blob = builder.build_serialized_network(net, cfg)
    finally:
        done.set()
    if blob is None:
        raise RuntimeError("build_serialized_network returned None")
    blob = bytes(blob)
    return blob, bytes(cache.serialize()), {"build_s": round(time.perf_counter() - t, 1),
                                           "engine_mb": round(len(blob) / 2**20, 1)}


class TrtEncoder:
    """One engine on one GPU: `enc(x)` with x (N, depth, H, W) on GPU `dev`, N a multiple of the engine
    shape; returns the features (N, C) float32 in its own buffer, reused on the next call. It runs on
    its own stream, ordered with the current stream on entry and exit (whoever reads the output on the
    current stream finds it ready).

    CUDA graphs (`capture(n)`): the blocks of a batch of `n` images recorded once in a graph, then
    replayed with a single call from a static input; bit-identical output. Calling the engine launches
    its kernels, CUDA's launch queue (~1,000 pending) fills up and the thread spins in the driver
    until the GPU drains some (blocking-sync contexts do not help): 296 s of CPU over 1,300 studies,
    taken from decoding. **A graph is not enough**: its launch goes through the same queue and spins
    the same way. The cause is the number of kernels: the engine exported from the "list" DenseNet has
    1,589 per call, ~1,150 of them copies for the concatenations (`knee.quant.ChainDenseNet`). With
    fewer kernels per call and `max_inflight` the thread sleeps instead of spinning.
    """

    def __init__(self, blob: bytes, dev: int, n_total: int,
                 max_inflight: int | None = None) -> None:
        """`max_inflight`: the maximum number of blocks queued on the GPU; before queueing another one the
        thread waits for the oldest to finish on a blocking event (it sleeps). Needed when the kernels of
        a few blocks fill CUDA's launch queue (~1,000): beyond that, launching spins in the driver.
        `None` = no limit.
        """
        trt = trt_module()
        self.dev = dev
        self.max_inflight = max_inflight
        self.inflight: collections.deque = collections.deque()
        with torch.cuda.device(dev):
            t = time.perf_counter()
            self.rt = trt.Runtime(logger())
            self.eng = self.rt.deserialize_cuda_engine(blob)
            if self.eng is None:
                raise RuntimeError("engine cannot be deserialized (different TensorRT or GPU?)")
            self.ctx = self.eng.create_execution_context()
            self.load_s = time.perf_counter() - t
            names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
            self.inp = next(k for k in names
                            if self.eng.get_tensor_mode(k) == trt.TensorIOMode.INPUT)
            self.out_name = next(k for k in names if k != self.inp)
            shape = tuple(self.eng.get_tensor_shape(self.inp))
            self.ctx.set_input_shape(self.inp, shape)
            self.nb = int(shape[0])
            self.in_shape = shape
            self.in_dtype = (torch.float16 if self.eng.get_tensor_dtype(self.inp) == trt.float16
                             else torch.float32)
            if n_total % self.nb:
                raise ValueError(f"{n_total} images do not split into blocks of {self.nb}")
            c = int(self.ctx.get_tensor_shape(self.out_name)[1])
            out_dt = (torch.float16 if self.eng.get_tensor_dtype(self.out_name) == trt.float16
                      else torch.float32)
            self.out = torch.empty((n_total, c), dtype=out_dt, device=f"cuda:{dev}")
            self.stream = torch.cuda.Stream(dev)
        self.graphs: dict[int, tuple] = {}      # n images -> (graph, static input)
        self.calls = {"graph": 0, "plain": 0}

    def _enqueue(self, xf: torch.Tensor) -> None:
        """The blocks of `xf` (contiguous, engine dtype) on the engine stream, with at most `max_inflight`
        blocks queued.
        """
        for i, chunk in enumerate(xf.split(self.nb)):
            if self.max_inflight:
                while len(self.inflight) >= self.max_inflight:
                    self.inflight.popleft().synchronize()
            self.ctx.set_tensor_address(self.inp, chunk.data_ptr())
            self.ctx.set_tensor_address(self.out_name, self.out[i * self.nb].data_ptr())
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            if self.max_inflight:
                ev = torch.cuda.Event(blocking=True)
                ev.record(self.stream)
                self.inflight.append(ev)

    def capture(self, n: int) -> float:
        """Records the graph for batches of `n` images (a multiple of the engine, <= `n_total`): one normal
        pass on the static addresses (TensorRT prepares the state of the blocks), then the capture.
        Returns the seconds spent. An error leaves the engine without a graph for `n`.
        """
        if n % self.nb or n > self.out.shape[0]:
            raise ValueError(f"graph for {n} images: the engine does {self.nb} per block, "
                             f"the output holds {self.out.shape[0]}")
        t = time.perf_counter()
        with torch.cuda.device(self.dev):
            static_in = torch.zeros((n, *self.in_shape[1:]), dtype=self.in_dtype,
                                    device=f"cuda:{self.dev}")
            with torch.cuda.stream(self.stream):
                self._enqueue(static_in)
            self.stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=self.stream, capture_error_mode="thread_local"):
                self._enqueue(static_in)
            self.stream.synchronize()
        self.graphs[n] = (g, static_in)
        return time.perf_counter() - t

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cur = torch.cuda.current_stream(self.dev)
        self.stream.wait_stream(cur)            # the input and the previous output are ready
        n = x.shape[0]
        with torch.cuda.stream(self.stream):
            gs = self.graphs.get(n)
            if gs is not None:
                gs[1].copy_(x)                  # in the engine dtype
                gs[0].replay()
                self.calls["graph"] += 1
            else:
                self._enqueue(x.to(self.in_dtype, memory_format=torch.contiguous_format))
                self.calls["plain"] += 1
        cur.wait_stream(self.stream)            # whoever reads the output waits for the engine
        return self.out[:n]


def layer_times(blob: bytes, dev: int = 0, reps: int = 3) -> list[tuple[str, float]]:
    """Milliseconds per engine layer for one call (mean over `reps`, after a warm-up pass), from
    TensorRT's `IProfiler` on synchronous `execute_v2`, random input. To see where the time goes:
    convolutions, reformatting, pointwise operations.
    """
    trt = trt_module()

    class Prof(trt.IProfiler):
        def __init__(self) -> None:
            trt.IProfiler.__init__(self)
            self.ms: dict[str, float] = {}
            self.on = False

        def report_layer_time(self, name, ms) -> None:              # noqa: ANN001
            if self.on:
                self.ms[name] = self.ms.get(name, 0.0) + float(ms)

    with torch.cuda.device(dev):
        eng = trt.Runtime(logger()).deserialize_cuda_engine(blob)
        ctx = eng.create_execution_context()
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        bufs = []
        for k in names:
            shape = tuple(eng.get_tensor_shape(k))
            if eng.get_tensor_mode(k) == trt.TensorIOMode.INPUT:
                ctx.set_input_shape(k, shape)
            dt = torch.float16 if eng.get_tensor_dtype(k) == trt.float16 else torch.float32
            bufs.append(torch.rand(shape, dtype=dt, device=f"cuda:{dev}"))
        prof = Prof()
        ctx.profiler = prof
        ctx.execute_v2([b.data_ptr() for b in bufs])
        prof.on = True
        for _ in range(reps):
            ctx.execute_v2([b.data_ptr() for b in bufs])
        torch.cuda.synchronize(dev)
    return [(k, v / reps) for k, v in prof.ms.items()]


CU_CTX_SCHED_BLOCKING_SYNC = 0x04


def set_blocking_sync() -> dict:
    """Puts the primary context of every GPU into blocking sync (`CU_CTX_SCHED_BLOCKING_SYNC`): whoever
    waits for the GPU sleeps instead of spinning. A TensorRT call waits for the GPU by spinning; two
    such threads on the 4 vCPUs of the T4 machine took CPU from decoding (main thread 335 s against
    30, +78 s over 1,300 studies). The primary-context flags are set through the driver
    (`cuDevicePrimaryCtxSetFlags_v2`, since CUDA 11 also on an already active context) and read back:
    returns {gpu: flags read} or {"error": ...}.
    """
    import ctypes
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return {"error": f"libcuda: {exc}"}
    if cu.cuInit(0) != 0:
        return {"error": "cuInit"}
    n = ctypes.c_int()
    cu.cuDeviceGetCount(ctypes.byref(n))
    set_flags = getattr(cu, "cuDevicePrimaryCtxSetFlags_v2", None) or cu.cuDevicePrimaryCtxSetFlags
    out = {}
    for i in range(n.value):
        dev = ctypes.c_int()
        cu.cuDeviceGet(ctypes.byref(dev), i)
        rc = set_flags(dev, ctypes.c_uint(CU_CTX_SCHED_BLOCKING_SYNC))
        flags, active = ctypes.c_uint(), ctypes.c_int()
        cu.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(active))
        out[i] = {"rc": int(rc), "flags": int(flags.value), "active": int(active.value),
                  "blocking": bool(flags.value & CU_CTX_SCHED_BLOCKING_SYNC)}
    return out
