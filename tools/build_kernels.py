"""Build the Kaggle kernels and datasets of the pipeline from this repository.

Every Kaggle kernel of the pipeline is a single script. The submission kernel carries the `knee` package and two
precompiled binaries embedded in it (base64 + zlib), so that nothing but the competition data, the weights, the
INT8 packages, the TensorRT runtime and the decoder wheels has to be read from the network mount at start-up.
This tool writes those scripts, with their `kernel-metadata.json`, into `build/`:

    python tools/build_kernels.py src-dataset          # build/knee-src: the source dataset (src, configs, data)
    python tools/build_kernels.py cache                # build/cache_build-s{k}of14: the 14 cache shards
    python tools/build_kernels.py weights --seed42 runs/376 --seed43 runs/384
                                                       # build/knee-runs-...: the weights dataset
    python tools/build_kernels.py int8-pack            # build/int8_pack: the INT8 packages of the two seeds
    python tools/build_kernels.py int8-repack          # build/int8_repack: the CUBIN of the kernels, offline
    python tools/build_kernels.py trt-runtime          # build/trt_runtime: TensorRT for offline inference
    python tools/build_kernels.py submission           # build/submission: the efficiency submission

Push each folder with `kaggle kernels push -p <folder>` (`kaggle datasets create -p` for the datasets) in the
order above; README.md lists the inputs of every kernel. `--owner` replaces the Kaggle user name in the
metadata; the slugs are those of the original runs.

`int8-repack` compiles `src/knee/dense8s_kernels.cu` with NVRTC on this machine (pip package
`nvidia-cuda-nvrtc-cu12`, ~2 min): the CUBIN is identical to the one the submission used, and its hash names
the source it was built from. `submission` embeds `prebuilt/knee_fastdecode_linux_x86_64.so` (the C library of
the decoding workers, built by gcc on the Kaggle image from `src/knee/data/fastdecode.c` and `leanio.c` with
the flags below) and `prebuilt/fused.cubin` (the uint8 normalisation kernels of `knee.data.fused_uint8`): the
kernel checks both against the source hashes and recompiles if they do not match.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import lzma
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
KAGGLE = ROOT / "kaggle"
PREBUILT = ROOT / "prebuilt"
OUT = ROOT / "build"
ORIG_OWNER = "marcoledda29"
CFAST_FLAGS = ["-O2", "-ffp-contract=off", "-fno-fast-math", "-shared"]
SRC_DATASET = "knee-src-label-ablation-196-197"      # the slug of the source dataset in the original runs
WEIGHTS_DATASET = "knee-runs-376-384-full-data-v9j-287-img512"
CACHE_SHARDS = 14


def b64z(data: bytes) -> str:
    return base64.b64encode(zlib.compress(data, 9)).decode("ascii")


def package() -> dict[str, bytes]:
    """The `knee` package as {relative path: bytes}, the way the kernels embed it."""
    out = {}
    for p in sorted((SRC / "knee").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            out[p.relative_to(SRC).as_posix()] = p.read_bytes()
    return out


def _metadata(src: Path, dst: Path, owner: str, **changes) -> None:
    meta = json.loads(src.read_text(encoding="utf-8"))
    meta.update(changes)
    for key in ("id", "dataset_sources", "kernel_sources"):
        v = meta.get(key)
        if isinstance(v, str):
            meta[key] = v.replace(f"{ORIG_OWNER}/", f"{owner}/")
        elif isinstance(v, list):
            meta[key] = [s.replace(f"{ORIG_OWNER}/", f"{owner}/") for s in v]
    dst.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _spans(src: str):
    """Character offsets of an AST node in `src` (the AST columns are UTF-8 bytes)."""
    lines = src.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))

    def pos(ln, bcol):
        return offs[ln - 1] + len(lines[ln - 1].encode("utf-8")[:bcol].decode("utf-8", errors="ignore"))

    return lambda n: (pos(n.lineno, n.col_offset), pos(n.end_lineno, n.end_col_offset))


def _fill(src: str, reps: list[tuple[int, int, str]]) -> str:
    for s, e, text in sorted(reps, reverse=True):
        src = src[:s] + text + src[e:]
    return src


# --------------------------------------------------------------------------- datasets
def src_dataset(owner: str) -> Path:
    """`src/`, `configs/` and `data/` as the dataset the cache and INT8 kernels mount (they look for
    `knee/constants.py` under /kaggle/input and read `configs/exp` and `data/folds` next to it)."""
    out = OUT / "knee-src"
    if out.exists():
        shutil.rmtree(out)
    for d in ("src", "configs", "data"):
        shutil.copytree(ROOT / d, out / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "knee src", "id": f"{owner}/{SRC_DATASET}", "licenses": [{"name": "other"}]},
        indent=2) + "\n", encoding="utf-8")
    return out


def weights(owner: str, seed42: Path, seed43: Path) -> Path:
    """The two seeds as the weights dataset of the submission: `fold0.pt` (seed 42) and `fold1.pt` (seed 43).
    In a full-data run every "fold" is a seed, and the submission averages every `fold*.pt` it finds."""
    out = OUT / WEIGHTS_DATASET
    out.mkdir(parents=True, exist_ok=True)
    prov = {"note": "two full-data seeds of the final DenseNet169 recipe at 512 px; the folds are seeds"}
    for name, run in (("fold0", seed42), ("fold1", seed43)):
        ck = Path(run) / "fold0.pt"
        shutil.copyfile(ck, out / f"{name}.pt")
        if (Path(run) / "fold0.json").is_file():
            shutil.copyfile(Path(run) / "fold0.json", out / f"{name}.json")
        prov[f"{name}.pt"] = {"run": Path(run).name, "md5": hashlib.md5(ck.read_bytes()).hexdigest()}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "knee runs 376 384 full data v9j 287 img512", "id": f"{owner}/{WEIGHTS_DATASET}",
         "licenses": [{"name": "other"}]}, indent=2) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- kernels
def cache(owner: str) -> list[Path]:
    """The 14 shard kernels of the 512 px cache: the same script with a different default `KNEE_SHARD`."""
    code = (KAGGLE / "cache_build" / "cache_build.py").read_text(encoding="utf-8")
    line = f'SHARD = os.environ.get("KNEE_SHARD", "0/{CACHE_SHARDS}")'
    if code.count(line) != 1:
        raise SystemExit("cache_build.py: the KNEE_SHARD line was not found exactly once")
    outs = []
    for k in range(CACHE_SHARDS):
        out = OUT / f"cache_build-s{k}of{CACHE_SHARDS}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "cache_build.py").write_text(code.replace(line, line.replace(f'"0/{CACHE_SHARDS}"',
                                                                            f'"{k}/{CACHE_SHARDS}"')),
                                            encoding="utf-8")
        _metadata(KAGGLE / "cache_build" / "kernel-metadata.json", out / "kernel-metadata.json", owner,
                  id=f"{owner}/knee-cache-build-313-matteo-split-img512-s{k}of{CACHE_SHARDS}",
                  title=f"knee cache build 313-matteo-split-img512 s{k}of{CACHE_SHARDS}")
        outs.append(out)
    return outs


def copy_kernel(name: str, script: str, owner: str) -> Path:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(KAGGLE / name / script, out / script)
    _metadata(KAGGLE / name / "kernel-metadata.json", out / "kernel-metadata.json", owner)
    return out


def _repack_helper():
    path = KAGGLE / "int8_repack" / "repack_dense8s_runtime.py"
    spec = importlib.util.spec_from_file_location("repack_dense8s_runtime", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def int8_repack(owner: str, nvrtc_site: Path | None) -> Path:
    """The offline CPU kernel that puts the compiled CUBIN into the INT8 packages of the two seeds.

    Same steps as the original: one NVRTC compilation here (`compile_runtime`), the kernel built by the
    helper's own `build` with the CUBIN embedded, the CUBIN payload compressed with LZMA (a Kaggle script
    must stay under 1 MB), and `dense8s.py` / `dense8s_kernels.cu` written next to the packages."""
    if nvrtc_site:
        sys.path.insert(0, str(nvrtc_site))
    helper = _repack_helper()
    out = OUT / "int8_repack"
    runtime = OUT / "int8_repack_runtime"
    if runtime.exists():
        shutil.rmtree(runtime)
    source, cudart = SRC / "knee" / "dense8s_kernels.cu", SRC / "knee" / "cudart.py"
    _, compiled = helper.compile_runtime(source, cudart, runtime, verify_load=False)
    compiled["compiler_platform"] = sys.platform
    manifest = runtime / "kernel-manifest.json"
    manifest.write_text(json.dumps(compiled, indent=2) + "\n", encoding="utf-8")
    meta = json.loads((KAGGLE / "int8_repack" / "kernel-metadata.json").read_text(encoding="utf-8"))
    pack_source = meta["kernel_sources"][0].replace(f"{ORIG_OWNER}/", f"{owner}/")
    report = helper.build(out, source=source, cudart_path=cudart, owner=owner, pack_source=pack_source,
                          prefix="d8s376x2j", kernel_id=meta["id"].split("/", 1)[1], runtime=manifest, cpu=True)
    code = (out / "repack.py").read_text(encoding="utf-8")
    cubin_name = compiled["cubin_file"]
    cubin = (runtime / cubin_name).read_bytes()
    tree = ast.parse(code)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PAYLOADS")
    payloads = ast.literal_eval(node.value)
    payloads[cubin_name] = base64.b64encode(lzma.compress(cubin, preset=9)).decode("ascii")
    s, e = _spans(code)(node.value)
    code = code[:s] + repr(payloads) + code[e:]
    code = code.replace("import zlib\n", "import lzma\nimport zlib\n", 1)
    code = code.replace("    raw = zlib.decompress(base64.b64decode(encoded))\n",
                        f"    raw = (lzma.decompress(base64.b64decode(encoded)) if name == {cubin_name!r}\n"
                        "           else zlib.decompress(base64.b64decode(encoded)))\n", 1)
    extras = {name: b64z((SRC / "knee" / name).read_bytes()) for name in ("dense8s.py", "dense8s_kernels.cu")}
    code += f"\nTILE_SOURCES = {extras!r}\n"
    code += ('for name, encoded in TILE_SOURCES.items():\n'
             '    (Path("/kaggle/working/dense8s-runtime") / name).write_bytes('
             'zlib.decompress(base64.b64decode(encoded)))\n')
    compile(code, "repack.py", "exec")
    if len(code.encode("utf-8")) >= 1_000_000:
        raise SystemExit("repack.py is over 1 MB: Kaggle would refuse it")
    (out / "repack.py").write_text(code, encoding="utf-8")
    _metadata(KAGGLE / "int8_repack" / "kernel-metadata.json", out / "kernel-metadata.json", owner)
    report["cubin_sha256"] = hashlib.sha256(cubin).hexdigest()
    (out / "repack-build-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out


def submission(owner: str, tag: str | None = None) -> Path:
    """The submission kernel: the template with the package and the binaries embedded."""
    code = (KAGGLE / "submission" / "submit.py").read_text(encoding="utf-8")
    tree = ast.parse(code)
    span = _spans(code)
    files = package()
    tag = tag or files["knee/GIT_SHA"].decode().strip()
    srcs = [files["knee/data/fastdecode.c"], files["knee/data/leanio.c"]]
    cfast_tag = hashlib.sha1(b"".join(srcs) + " ".join(CFAST_FLAGS).encode()).hexdigest()[:12]
    cubins = {"fused.cubin": b64z((PREBUILT / "fused.cubin").read_bytes()),
              "fused-compileinfo.json": b64z((PREBUILT / "fused-compileinfo.json").read_bytes())}
    reps = []
    for n in tree.body:
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Attribute) \
                and isinstance(n.iter.func.value, ast.Dict) and not n.iter.func.value.keys:
            reps.append((*span(n.iter.func.value), repr(cubins)))
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name == "BASE":
                reps.append((*span(n.value), repr({k: b64z(v) for k, v in files.items()})))
            elif name == "OVERLAY_SHA":
                reps.append((*span(n.value), repr(tag)))
            elif name == "CFAST_LIB":
                for k, v in zip(n.value.keys, n.value.values):
                    key = ast.literal_eval(k)
                    if key == "tag":
                        reps.append((*span(v), repr(cfast_tag)))
                    elif key == "so":
                        reps.append((*span(v), repr(b64z((PREBUILT / "knee_fastdecode_linux_x86_64.so").read_bytes()))))
                    elif key == "flags" and ast.literal_eval(v) != CFAST_FLAGS:
                        raise SystemExit("submit.py: CFAST_LIB flags differ from the prebuilt library's")
    if len(reps) != 5:
        raise SystemExit(f"submit.py: expected 5 placeholders, found {len(reps)}")
    code = _fill(code, reps)
    compile(code, "submit.py", "exec")
    out = OUT / "submission"
    out.mkdir(parents=True, exist_ok=True)
    (out / "submit.py").write_text(code, encoding="utf-8")
    _metadata(KAGGLE / "submission" / "kernel-metadata.json", out / "kernel-metadata.json", owner)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["src-dataset", "cache", "weights", "int8-pack", "int8-repack",
                                       "trt-runtime", "submission"])
    ap.add_argument("--owner", default=ORIG_OWNER, help="Kaggle user name for the metadata")
    ap.add_argument("--seed42", type=Path, help="weights: output folder of the seed-42 run")
    ap.add_argument("--seed43", type=Path, help="weights: output folder of the seed-43 run")
    ap.add_argument("--nvrtc-site", type=Path, help="int8-repack: folder with the nvidia-cuda-nvrtc wheel unpacked")
    a = ap.parse_args()
    if a.target == "src-dataset":
        out = src_dataset(a.owner)
    elif a.target == "cache":
        out = cache(a.owner)
    elif a.target == "weights":
        if not (a.seed42 and a.seed43):
            ap.error("weights needs --seed42 and --seed43")
        out = weights(a.owner, a.seed42, a.seed43)
    elif a.target == "int8-pack":
        out = copy_kernel("int8_pack", "pack_dense8s.py", a.owner)
    elif a.target == "int8-repack":
        out = int8_repack(a.owner, a.nvrtc_site)
    elif a.target == "trt-runtime":
        out = copy_kernel("trt_runtime", "trt_runtime_cp313.py", a.owner)
    else:
        out = submission(a.owner)
    print(out)


if __name__ == "__main__":
    main()
