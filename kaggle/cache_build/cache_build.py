"""Kaggle kernel: builds the training cache from the DICOMs. CPU.

Why it exists. The raw DICOMs are ~819,000 files on a network mount. This kernel reads them once and
writes a uint8 array per study: from then on every training run starts from ready pixels and the GPU
does not wait for the disk.

What it writes into /kaggle/working:

    vols_<tag>.npy    uint8 (N, S, K, P, P)   N studies x S sequence slots
    mask_<tag>.npy    bool  (N, S)            which slots were actually acquired
    index_<tag>.csv   one row per study: id, row in the array, side and where it comes from,
                      ordering method per slot, decoded slices
    build_<tag>.json  summary: coverage, decoding failures, timings

The tag names resolution, slices, crop and band: two different configurations cannot end up in the
same file (see `CacheSpec` in src/knee/data/cache.py).

Which spec is built is given by KNEE_EXP, from the `cache:` block of that experiment's config: the
same source from which training derives the tag it then looks for. For the final models the cache is
512 px, 24 slices, crops of 110 and 150 mm, band 0.15-0.85, built in 14 shards (KNEE_SHARD).

Training set only. At submission time the test studies are decoded on the fly by the submission
kernel: the test set is hidden and re-scored.

Internet on, needed only for the decoding plugins: the offline constraint applies to the submission
kernel, not to a preprocessing kernel.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --- bootstrap: putting `knee` on sys.path before it can be imported ------------------
# This part cannot use knee.kaggle, which is exactly what it is looking for. Kaggle does not guarantee
# the depth of the mount: datasets may be under /kaggle/input/datasets/<owner>/<slug>/, two levels deeper
# than usual.
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
    print("the `knee` package is not mounted anywhere.", flush=True)
    print("content of /kaggle/input:", flush=True)
    for d in seen[:60]:
        print(f"  {d}", flush=True)
    raise FileNotFoundError("knee-src dataset not mounted: see the listing above")


SRC = find_src()
sys.path.insert(0, str(SRC))
BUNDLE = SRC.parent if SRC.name == "src" else SRC
log(f"knee from {SRC} | bundle {BUNDLE}")

from knee.constants import ID, SEQUENCE_SLOTS                       # noqa: E402
from knee.data.cache import CacheSpec, build_study, crop_fit, pick_slots  # noqa: E402
from knee.data.dicom import DecodeErrors                            # noqa: E402
from knee.kaggle import describe_inputs, find_competition_root       # noqa: E402
from knee.utils.config import load_config                           # noqa: E402

EXP = os.environ.get("KNEE_EXP", "313-matteo-split-img512")
CFG = load_config(BUNDLE / "configs" / "exp" / f"{EXP}.yaml")
SPEC = CacheSpec(**dict(CFG["cache"], band=tuple(CFG["cache"]["band"])))
# A cache above ~20 GB does not fit in a kernel's output: it is built in shards, one per kernel, over
# contiguous blocks of studies (`row` stays global). Training reads the shards as they are, or merges
# them where there is disk. Zero of one = everything in a single kernel. Kaggle passes no environment
# variables: the default string below is rewritten for every shard by `tools/build_kernels.py`.
SHARD = os.environ.get("KNEE_SHARD", "0/14")
SHARD_K, SHARD_N = (int(x) for x in SHARD.split("/"))
if not 0 <= SHARD_K < SHARD_N:
    raise SystemExit(f"KNEE_SHARD={SHARD!r} makes no sense: k/n with 0 <= k < n is required")
WORKERS = 16                # the mount is latency-bound: more threads than cores is fine
LIMIT = int(os.environ.get("KNEE_LIMIT", "0"))   # >0 for a short smoke test
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
log(f"experiment {CFG['name']} | cache {SPEC.tag} | shard {SHARD_K + 1}/{SHARD_N}")


def shard_bounds(n: int, k: int, parts: int) -> tuple[int, int]:
    """Rows [lo, hi) of shard `k` over `parts` contiguous blocks of `n` studies."""
    edges = np.linspace(0, n, parts + 1).round().astype(int)
    return int(edges[k]), int(edges[k + 1])


def ensure_decoders() -> dict:
    """The plugins for compressed transfer syntaxes.

    The corpus may mix uncompressed Explicit VR LE, JPEG Lossless, JPEG 2000 and Implicit VR LE. Without
    the plugins, `pixel_array` raises on everything compressed, and since decoding failures are counted
    here instead of becoming black images, the run would not go wrong: it would waste hours producing a
    half-empty cache.
    """
    have = {}
    for mod in ("pylibjpeg", "libjpeg", "openjpeg", "gdcm"):
        try:
            __import__(mod)
            have[mod] = True
        except ImportError:
            have[mod] = False
    if not (have["pylibjpeg"] and have["libjpeg"] and have["openjpeg"]):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylibjpeg",
                        "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
        for mod in ("pylibjpeg", "libjpeg", "openjpeg"):
            try:
                __import__(mod)
                have[mod] = True
            except ImportError:
                have[mod] = False
    return have


def count_files(series_dirs: dict[str, Path]) -> dict[str, int]:
    """Number of slices per series. Used to break ties within a slot.

    One `listdir` per series, ~24,000 calls: in parallel because it is pure latency.
    """
    def one(item):
        uid, p = item
        try:
            return uid, len(os.listdir(p))
        except OSError:
            return uid, 0
    with ThreadPoolExecutor(WORKERS) as ex:
        return dict(ex.map(one, series_dirs.items()))


def preflight(root: Path, series: pd.DataFrame, n: int = 40) -> float:
    """Tries to decode one slice for each of n random series.

    It costs about ten seconds and answers the one question that can ruin the run: can the pixels be
    read? A high failure rate means decoders are missing, and that must be found out now, not at the end
    of an hour of work.
    """
    from knee.data.dicom import DecodeErrors as DE, read_slice
    pick = series.sample(min(n, len(series)), random_state=0)
    errs, tried = DE(), 0
    for r in pick.itertuples(index=False):
        d = root / "train_series" / r.StudyInstanceUID / r.SeriesInstanceUID
        try:
            files = os.listdir(d)
        except OSError:
            continue
        if not files:
            continue
        tried += 1
        read_slice(str(d / files[len(files) // 2]), errs)
    rate = len(errs) / max(tried, 1)
    log(f"preflight: {tried} series tried, {len(errs)} failed ({rate:.1%})")
    for _, msg in errs.items[:3]:
        log(f"  {msg}")
    return rate


def main() -> None:
    log(f"decoders: {ensure_decoders()}")
    try:
        root = find_competition_root()
    except FileNotFoundError:
        print(describe_inputs(max_depth=3), flush=True)
        raise
    log(f"competition mount: {root}")

    series = pd.read_csv(root / "train_series.csv")
    studies = pd.read_csv(root / "train.csv", usecols=[ID])[ID].tolist()
    if LIMIT:
        studies = studies[:LIMIT]
        series = series[series[ID].isin(studies)]
    log(f"{len(studies)} studies, {len(series)} series")

    if preflight(root, series) > 0.05:
        raise SystemExit("preflight failed: the pixels do not decode, the plugins for compressed "
                         "transfer syntaxes are missing. Run stopped before "
                         "spending hours on an empty cache.")

    dirs = {r.SeriesInstanceUID: root / "train_series" / r.StudyInstanceUID / r.SeriesInstanceUID
            for r in series.itertuples(index=False)}
    n_files = count_files(dirs)
    log(f"slices counted: {sum(n_files.values())} files in {len(n_files)} series")

    # The spec's slots, not the default ones: a 5-slot cache built with the default 4 would always have the
    # fifth empty, and no log would say so.
    n_seq = len(SPEC.active_slots)
    slots = pick_slots(series, n_files, slots=SPEC.active_slots)
    filled = np.array([len(slots.get(s, {})) for s in studies])
    log(f"filled slots: mean {filled.mean():.2f}/{n_seq} | "
        + " ".join(f"{k}:{int((filled == k).sum())}" for k in range(n_seq + 1)))

    n_all = len(studies)
    lo, hi = shard_bounds(n_all, SHARD_K, SHARD_N)
    studies = studies[lo:hi]
    n = len(studies)
    if SHARD_N > 1:
        log(f"shard {SHARD_K + 1}/{SHARD_N}: rows [{lo}, {hi}) of {n_all}, "
            f"{n * SPEC.bytes_per_study / 1e9:.1f} GB")
    vols = np.lib.format.open_memmap(OUT / f"vols_{SPEC.tag}.npy", mode="w+",
                                     dtype=np.uint8, shape=(n,) + SPEC.shape)
    mask = np.zeros((n, SPEC.shape[0]), dtype=bool)
    errors = DecodeErrors()
    rows: list[dict] = []

    def one(i_study):
        i, study = i_study
        d = root / "train_series" / study
        vol, m, meta = build_study(str(d), slots.get(study, {}), SPEC, errors)
        return i, study, vol, m, meta

    fovs: list[float] = []
    spacings: list[float] = []
    offsets: list[float] = []
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, study, vol, m, meta in ex.map(one, enumerate(studies)):
            vols[i] = vol
            mask[i] = m
            fov = {k: v for k, v in meta["fov_mm"].items() if v is not None}
            mmpx = {k: v for k, v in meta["mm_px"].items() if v is not None}
            offsets += [v for v in meta.get("offset_mm", {}).values() if v is not None]
            # The two lists must be aligned series by series: `crop_fit` pairs them to compute the acquired pixels
            # inside the box.
            for k in sorted(fov):
                if k in mmpx:
                    fovs.append(fov[k])
                    spacings.append(mmpx[k])
            rows.append({ID: study, "row": lo + i, "side": meta["side"],
                         "side_source": meta["side_source"], "n_slices_read": meta["n_read"],
                         "n_slots": int(m.sum()),
                         "orders": "|".join(f"{k}:{v}" for k, v in sorted(meta["orders"].items())),
                         "fov_mm": "|".join(f"{k}:{v:.0f}" for k, v in sorted(fov.items())),
                         "mm_px": "|".join(f"{k}:{v:.3f}" for k, v in sorted(mmpx.items())),
                         "sag_band": "|".join(f"{k}:{v}" for k, v in
                                              sorted(meta.get("sag_band", {}).items()))})
            done += 1
            if done % 250 == 0:
                rate = done / (time.time() - T0)
                log(f"{done}/{n} | {rate:.1f} studies/s | ~{(n - done) / rate / 60:.0f} min left "
                    f"| decode failures {len(errors)}")
    vols.flush()

    idx = pd.DataFrame(rows).sort_values("row")
    idx.to_csv(OUT / f"index_{SPEC.tag}.csv", index=False)
    np.save(OUT / f"mask_{SPEC.tag}.npy", mask)

    # The summary is the part that gets read afterwards: if one of these measurements is off, the cache is
    # silently wrong and every downstream number is meaningless.
    order_counts = idx["orders"].str.split("|").explode().str.split(":").str[1].value_counts()
    summary = {
        "tag": SPEC.tag, "spec": SPEC.__dict__, "exp": CFG["name"], "n_studies": n,
        "shard": {"k": SHARD_K, "n": SHARD_N, "rows": [lo, hi], "n_studies_total": n_all},
        "crop_fit": crop_fit(fovs, SPEC.crop_mm, spacings, SPEC.img),
        # How much centring moved the box. Empty with `center: image`, i.e. when the box is in the middle of the
        # frame (the final configuration).
        "center_offset_mm": ({"n_series": len(offsets),
                              "pct": {str(q): round(float(np.percentile(offsets, q)), 1)
                                      for q in (5, 25, 50, 75, 95, 99)},
                              "over_10mm": round(float(np.mean(np.array(offsets) > 10)), 4),
                              "over_20mm": round(float(np.mean(np.array(offsets) > 20)), 4)}
                             if offsets else {"n_series": 0}),
        "slots_per_study": {str(k): int((filled == k).sum()) for k in range(5)},
        # With two scales there are eight slots: coverage is reported per scale, otherwise the second half of
        # the mask would appear nowhere.
        "slot_coverage": {f"{p}/F{f}/S{s}" + ("" if k == 0 else f"@{int(c)}mm"):
                          float(mask[:, j + k * len(SEQUENCE_SLOTS)].mean())
                          for k, c in enumerate(SPEC.crops)
                          for j, (p, f, s) in enumerate(SEQUENCE_SLOTS)},
        "side": idx["side_source"].value_counts().to_dict(),
        "slice_order_method": order_counts.to_dict(),
        # With `sag_band_lm`: where the medial side ended up in every sagittal series ("high"/"low") or why the
        # symmetric band was used. Empty without the key.
        "sag_band": (idx["sag_band"].fillna("").astype(str).str.split("|").explode()
                     .loc[lambda s: s != ""].str.split(":").str[1].value_counts().to_dict()),
        "decode_failures": len(errors),
        "decode_failures_sample": errors.items[:20],
        "seconds": round(time.time() - T0, 1),
    }
    (OUT / f"build_{SPEC.tag}.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({k: v for k, v in summary.items() if k != "decode_failures_sample"}, indent=2))


if __name__ == "__main__":
    main()
