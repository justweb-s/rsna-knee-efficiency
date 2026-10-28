"""Kaggle kernel (CPU, internet): TensorRT 10.16.1.11 for offline inference, unpacked, wheels for
Python 3.13.

The submission runs without internet: the TensorRT runtime must come from a mounted input.
`pip install --target` of tensorrt-cu12 (+ libs and bindings, without dependencies: the CUDA runtime
already comes with torch) into /kaggle/working/trt_runtime, site-packages style, so that
`sys.path.insert(0, ...)` and `import tensorrt` are enough, with no cold pip. The libraries needed
only to build engines (the builder resources, ~GiB) are removed: the submission loads engines already
built on the T4. Prints files, sizes and the `__init__` files that load the libraries, and tests the
import from the folder. No submission.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/trt_runtime") if Path("/kaggle/working").is_dir() else Path("runs/trt_runtime")
VERSION = "10.16.1.11"
PKGS = [f"tensorrt-cu12=={VERSION}", f"tensorrt-cu12-libs=={VERSION}",
        f"tensorrt-cu12-bindings=={VERSION}"]
DROP = ("libnvinfer_builder_resource",)


def main() -> None:
    t = time.perf_counter()
    # cp313: a kernel stays on the Docker image it was created with; this one must be created on the same
    # image as the submission kernel (Python 3.13), so pip picks the cp313 wheels by itself. Forcing
    # --python-version/--only-binary does not work: `tensorrt-cu12` is a source-only package
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target",
                        str(OUT), *PKGS], capture_output=True, text=True)
    print(f"pip rc {r.returncode} in {time.perf_counter() - t:.0f} s {(r.stdout + r.stderr)[-500:]!r}",
          flush=True)
    files = sorted((p for p in OUT.rglob("*") if p.is_file()), key=lambda p: -p.stat().st_size)
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {total / 2**20:.0f} MiB; the largest:", flush=True)
    for p in files[:25]:
        print(f"  {p.stat().st_size / 2**20:8.1f} MiB  {p.relative_to(OUT)}", flush=True)
    for init in list(OUT.glob("tensorrt*/__init__.py")):
        print(f"\n--- {init.relative_to(OUT)} ---\n{init.read_text()[:3000]}", flush=True)
    dropped = []
    for p in files:
        if p.name.startswith(DROP):
            dropped.append((str(p.relative_to(OUT)), round(p.stat().st_size / 2**20, 1)))
            p.unlink()
    left = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\nremoved {dropped}: {left / 2**20:.0f} MiB left", flush=True)
    sys.path.insert(0, str(OUT))
    t = time.perf_counter()
    try:
        import tensorrt as trt
        info = {"import_s": round(time.perf_counter() - t, 2), "version": trt.__version__,
                "file": trt.__file__}
    except Exception as exc:
        info = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    print(f"import from the folder: {info}", flush=True)
    (OUT / "TRT_RUNTIME").write_text(json.dumps({"version": VERSION, "dropped": dropped,
                                                 "mib": round(left / 2**20), **info}, indent=1))
    cache = Path.home() / ".cache" / "pip"
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
    # Without a GPU a TensorRT Runtime cannot be created, and the normal interpreter exit after the import
    # segfaults: the kernel would end in error with the work done.
    sys.stdout.flush()
    os._exit(0)
