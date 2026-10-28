"""Updates only the CUBIN and kernels_sha of the dense8s packages, without calibration or training.

    python repack_dense8s_runtime.py build --cpu --runtime <kernel-manifest.json> --output <dir>
    # CPU kernel, offline: CUBIN already compiled and embedded, no NVRTC on Kaggle.

    python repack_dense8s_runtime.py run --packs <pack>_fold0.npz <pack>_fold1.npz --output <dir>

run() can also be called by the benchmark before loading the encoders. It compiles once with NVRTC,
exports dense8s_sm75.cubin and kernel-manifest.json, then copies the packages. --runtime reuses an
already exported CUBIN: source and binary hashes are checked. The eng_*/par_* members (and every other
member not replaced) are copied as NPY bytes, without deserialising the weights; the manifest checks
their hashes before and after. The only metadata field that changes is kernels_sha. The accuracy of
the predictions is still validated on the benchmark: a compilation and a module load are not a
numerical check.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import time
import zipfile
import zlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "src" / "knee" / "dense8s_kernels.cu"
DEFAULT_CUDART = REPO / "src" / "knee" / "cudart.py"
DEFAULT_BUILD = REPO / "build" / "repack-dense8s"
OWNER = "marcoledda29"
PACK_SOURCE = f"{OWNER}/knee-pack-dense8s-314x2"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(reader) -> str:
    h = hashlib.sha256()
    while chunk := reader.read(1 << 20):
        h.update(chunk)
    return h.hexdigest()


def _file_sha(path: Path) -> str:
    with path.open("rb") as reader:
        return _digest(reader)


def _source_info(source: Path) -> dict:
    return _source_bytes_info(source.read_bytes())


def _source_bytes_info(raw: bytes) -> dict:
    return {"kernels_sha": hashlib.sha1(raw).hexdigest()[:12], "source_sha256": _sha(raw)}


def _npy_bytes(data: bytes) -> bytes:
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, np.frombuffer(data, dtype=np.uint8), allow_pickle=False)
    return buf.getvalue()


def _meta(data: bytes) -> dict:
    import numpy as np
    array = np.load(io.BytesIO(data), allow_pickle=False)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError("meta.npy must be a uint8 vector")
    meta = json.loads(array.tobytes().decode("utf-8"))
    if meta.get("kind") != "dense8s" or meta.get("format") != 1:
        raise ValueError(f"not a dense8s format-1 package: {meta.get('kind')}/{meta.get('format')}")
    return meta


def _load_cudart(path: Path):
    spec = importlib.util.spec_from_file_location("_dense8s_repack_cudart", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_runtime(cubin: bytes, manifest: dict, source: Path | bytes) -> None:
    if not re.fullmatch(r"sm_\d+[a-z]?", manifest.get("arch", "")):
        raise ValueError("a CUBIN for an sm_XX architecture is required, not PTX")
    if not cubin.startswith(b"\x7fELF") or manifest.get("cubin_sha256") != _sha(cubin):
        raise ValueError("CUBIN not ELF or binary hash differs from the manifest")
    expected = _source_bytes_info(source) if isinstance(source, bytes) else _source_info(source)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("the CUBIN manifest does not match the current source")


def compile_runtime(source: Path, cudart_path: Path, output: Path, arch: str = "sm_75",
                    verify_load: bool = True) -> tuple[bytes, dict]:
    """One compilation, with the same arguments as dense8s.kernel_cubin; no shared _MOD state."""
    if not re.fullmatch(r"sm_\d+[a-z]?", arch):
        raise ValueError("arch must be sm_XX")
    output.mkdir(parents=True, exist_ok=True)
    image_name = f"dense8s_{arch.replace('_', '')}.cubin"
    if (output / image_name).exists() or (output / "kernel-manifest.json").exists():
        raise FileExistsError(f"runtime already present in {output}")
    before = _source_info(source)
    cudart = _load_cudart(cudart_path)
    started = time.perf_counter()
    cubin = cudart.compile_source(source.read_text(encoding="utf-8"), arch)
    compile_s = round(time.perf_counter() - started, 4)
    if before != _source_info(source):
        raise RuntimeError("source modified during compilation")
    manifest = {"format": 1, **before, "arch": arch, "cubin_file": image_name,
                "cubin_sha256": _sha(cubin), "cubin_bytes": len(cubin), "compile_s": compile_s,
                "compile_options": [f"--gpu-architecture={arch}", "--std=c++14"],
                "nvrtc_version": list(cudart.nvrtc_version()), "cudart_sha256": _file_sha(cudart_path),
                "module_load_verified": False, "numerically_validated": False}
    _validate_runtime(cubin, manifest, source)
    if verify_load:
        cudart.Module(cubin, 0)
        manifest["module_load_verified"] = True
    (output / image_name).write_bytes(cubin)
    (output / "kernel-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return cubin, manifest


def reuse_runtime(manifest_path: Path, source: Path | bytes, arch: str = "sm_75") -> tuple[bytes, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = manifest.get("cubin_file", "")
    if not name or Path(name).name != name:
        raise ValueError("cubin_file must be a relative name without directories")
    cubin = (manifest_path.parent / name).read_bytes()
    _validate_runtime(cubin, manifest, source)
    if manifest["arch"] != arch:
        raise ValueError(f"architecture {manifest['arch']} differs from {arch}")
    return cubin, manifest


def repack_pack(original: Path, output: Path, cubin: bytes, runtime: dict, source: Path) -> dict:
    """Copies the ZIP members streaming; changes only meta.kernels_sha and cubin.npy."""
    _validate_runtime(cubin, runtime, source)
    if original.resolve() == output.resolve() or output.exists():
        raise FileExistsError(f"refusing to overwrite package {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    untouched = {}
    input_sha = _file_sha(original)
    try:
        with zipfile.ZipFile(original) as old:
            infos = old.infolist()
            if len(infos) != len({v.filename for v in infos}) or "meta.npy" not in old.namelist():
                raise ValueError("package with duplicate names or missing meta.npy")
            previous = _meta(old.read("meta.npy"))
            current = dict(previous, kernels_sha=runtime["kernels_sha"])
            replacements = {"meta.npy": _npy_bytes(json.dumps(current).encode("utf-8")),
                            "cubin.npy": _npy_bytes(cubin)}
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as new:
                for info in infos:
                    if info.filename in replacements:
                        new.writestr(copy.copy(info), replacements.pop(info.filename))
                        continue
                    with old.open(info) as src, new.open(copy.copy(info), "w", force_zip64=True) as dst:
                        h = hashlib.sha256()
                        while chunk := src.read(1 << 20):
                            h.update(chunk)
                            dst.write(chunk)
                        untouched[info.filename] = h.hexdigest()
                for name, data in replacements.items():
                    new.writestr(zipfile.ZipInfo(name), data)
        with zipfile.ZipFile(temporary) as new:
            if _meta(new.read("meta.npy")) != current:
                raise RuntimeError("metadata not preserved")
            for name, expected in untouched.items():
                with new.open(name) as reader:
                    if _digest(reader) != expected:
                        raise RuntimeError(f"member modified during the copy: {name}")
        if _file_sha(original) != input_sha:
            raise RuntimeError("original package modified during the copy")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"input": str(original), "output": output.name, "input_sha256": input_sha,
            "output_sha256": _file_sha(output), "old_kernels_sha": previous.get("kernels_sha"),
            "kernels_sha": runtime["kernels_sha"], "members_unchanged": untouched,
            "parameters_and_engines_unchanged": True, "numerically_validated": False}


def discover_packs(base: Path, prefix: str = "d8s314x2", folds: tuple = (0, 1)) -> list[Path]:
    found = []
    for fold in folds:
        hits = sorted(base.rglob(f"{prefix}_fold{fold}.npz"))
        if len(hits) != 1:
            raise FileNotFoundError(f"expected exactly one {prefix}_fold{fold}.npz in {base}, found {len(hits)}")
        found.append(hits[0])
    return found


def run(packs: list[Path], output: Path, source: Path = DEFAULT_SOURCE,
        cudart_path: Path = DEFAULT_CUDART, arch: str = "sm_75", runtime: Path | None = None,
        verify_load: bool = True) -> dict:
    """Exports runtime and packages before inference; --runtime reuses a verified export."""
    packs, output, source, cudart_path = list(map(Path, packs)), Path(output), Path(source), Path(cudart_path)
    if not packs or len({p.name for p in packs}) != len(packs):
        raise ValueError("packages with distinct names are required")
    if any(not p.is_file() for p in packs):
        raise FileNotFoundError("one of the packages does not exist")
    if any((output / p.name).exists() for p in packs):
        raise FileExistsError("the output already contains a requested package")
    output.mkdir(parents=True, exist_ok=True)
    if runtime is None:
        cubin, compiled = compile_runtime(source, cudart_path, output, arch, verify_load)
    else:
        cubin, compiled = reuse_runtime(Path(runtime), source, arch)
        for name in (compiled["cubin_file"], "kernel-manifest.json"):
            target = output / name
            origin = Path(runtime) if name == "kernel-manifest.json" else Path(runtime).parent / name
            if origin.resolve() != target.resolve():
                if target.exists():
                    raise FileExistsError(target)
                shutil.copyfile(origin, target)
    report = {"runtime": compiled, "packs": [repack_pack(p, output / p.name, cubin, compiled, source) for p in packs]}
    (output / "repack-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build(output: Path = DEFAULT_BUILD, source: Path = DEFAULT_SOURCE,
          cudart_path: Path = DEFAULT_CUDART, owner: str = OWNER,
          pack_source: str = PACK_SOURCE, prefix: str = "d8s314x2",
          kernel_id: str = "knee-repack-dense8s-314x2-v6", runtime: Path | None = None,
          cpu: bool = False) -> dict:
    """Prepares the private offline Kaggle kernel; embeds only the helper, the NVRTC wrapper and the .cu."""
    output, source, cudart_path = Path(output), Path(source), Path(cudart_path)
    # The benchmark overlay does read_text + write_text on Linux: the effective .cu is LF even when the
    # Windows working copy is CRLF. Hashes and payloads here are those of the deployed bytes.
    sources = {"repack_dense8s_runtime.py": Path(__file__).read_text(encoding="utf-8").encode("utf-8"),
               "cudart.py": cudart_path.read_text(encoding="utf-8").encode("utf-8"),
               "dense8s_kernels.cu": source.read_text(encoding="utf-8").encode("utf-8")}
    if cpu and runtime is None:
        raise ValueError("--cpu requires --runtime: the CPU kernel copies an already compiled CUBIN")
    compiled = None
    if runtime is not None:
        runtime = Path(runtime)
        cubin, compiled = reuse_runtime(runtime, sources["dense8s_kernels.cu"])
        sources["kernel-manifest.json"] = runtime.read_bytes()
        sources[compiled["cubin_file"]] = cubin
    hashes = {name: _sha(raw) for name, raw in sources.items()}
    payloads = {name: base64.b64encode(zlib.compress(raw, 9)).decode("ascii") for name, raw in sources.items()}
    code = '''"""dense8s CUBIN update, offline: no training, calibration or TensorRT rebuild."""
import base64
import hashlib
import importlib.util
import json
import zlib
from pathlib import Path

PAYLOADS = __PAYLOADS__
HASHES = __HASHES__
root = Path("/kaggle/working/repack-runtime-src")
root.mkdir(parents=True, exist_ok=True)
for name, encoded in PAYLOADS.items():
    raw = zlib.decompress(base64.b64decode(encoded))
    if hashlib.sha256(raw).hexdigest() != HASHES[name]:
        raise RuntimeError(f"hash of the embedded source differs: {name}")
    (root / name).write_bytes(raw)
spec = importlib.util.spec_from_file_location("dense8s_repack", root / "repack_dense8s_runtime.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
packs = helper.discover_packs(Path("/kaggle/input"), prefix=__PREFIX__)
runtime = root / "kernel-manifest.json" if __HAS_RUNTIME__ else None
result = helper.run(packs, Path("/kaggle/working/dense8s-runtime"),
                    source=root / "dense8s_kernels.cu", cudart_path=root / "cudart.py",
                    runtime=runtime, verify_load=__VERIFY_LOAD__)
print(json.dumps(result, indent=2), flush=True)
'''.replace("__PAYLOADS__", repr(payloads)).replace("__HASHES__", repr(hashes)).replace("__PREFIX__", repr(prefix)) \
       .replace("__HAS_RUNTIME__", repr(runtime is not None)).replace("__VERIFY_LOAD__", repr(not cpu))
    compile(code, "repack.py", "exec")
    meta = {"id": f"{owner}/{kernel_id}", "title": "knee repack dense8s 314x2 v6",
            "code_file": "repack.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": not cpu, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [f"{owner}/knee-src-label-ablation-196-197"],
            "kernel_sources": [pack_source], "competition_sources": [], "model_sources": []}
    if not cpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    report = {"sources": hashes, "kernels_sha": hashlib.sha1(sources["dense8s_kernels.cu"]).hexdigest()[:12],
              "source_sha256": hashes["dense8s_kernels.cu"], "pack_source": pack_source,
              "prefix": prefix, "arch": "sm_75", "numerically_validated": False}
    if runtime is not None:
        report.update(cpu=cpu, runtime=compiled)
    output.mkdir(parents=True, exist_ok=True)
    (output / "repack.py").write_text(code, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (output / "repack-build-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="action", required=True)
    for name in ("build", "run"):
        p = sub.add_parser(name)
        p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
        p.add_argument("--cudart", type=Path, default=DEFAULT_CUDART)
        p.add_argument("--output", type=Path, default=DEFAULT_BUILD if name == "build" else Path("/kaggle/working/dense8s-runtime"))
        p.add_argument("--runtime", type=Path, help="kernel-manifest.json of an already exported CUBIN")
        if name == "run":
            p.add_argument("--packs", nargs="+", type=Path, required=True)
            p.add_argument("--arch", default="sm_75")
            p.add_argument("--no-verify-load", action="store_true", help="NVRTC cross-compilation only, without loading on the GPU")
        else:
            p.add_argument("--cpu", action="store_true", help="offline CPU kernel: requires --runtime")
    args = cli.parse_args()
    if args.action == "build":
        report = build(args.output, args.source, args.cudart, runtime=args.runtime, cpu=args.cpu)
    else:
        report = run(args.packs, args.output, args.source, args.cudart, args.arch, args.runtime, not args.no_verify_load)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
