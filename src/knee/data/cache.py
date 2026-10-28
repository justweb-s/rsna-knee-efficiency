"""From the DICOM folder of a study to the array the model reads.

One study -> `(slot, slice, y, x)` uint8, plus a mask saying which slots were actually acquired. The
slots are the four plane x fluid x fat-sat combinations with >= 94% coverage in the training set
(`constants.SEQUENCE_SLOTS`).

Two choices worth stating, because they are not obvious from the code.

**An empty slot stays empty.** If a series with the right weighting is missing, it is not replaced
by the most similar one: it is set to zero and its mask bit is cleared. Filling a T1 slot with a PD
would mean the mask claims an acquisition that never happened, and downstream pooling would give
that series the weight of two.

**Intensity normalisation is per series, not per slice.** MRI intensities are arbitrary across
series but comparable within a series: normalising slice by slice would flatten exactly the
differences between neighbouring slices, which tell an effusion from normal marrow.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..constants import ID, SEQUENCE_SLOTS

# Threads for the slices of a series: ~10 files of ~150 KB each on the Kaggle mount.
SLICE_THREADS = 10
from .pools import thread_pool
from .geometry import (band_indices, crop_box, foreground_center, fov_mm,
                       normalise_laterality, side_from_centers)

# `.dicom` (and pydicom with it) is imported **inside** the functions that build the cache, not here.
# Whoever reads an already built cache never touches a DICOM: `CacheSpec` and `StudyCache` serve
# training, and training also runs where pydicom is not installed. `DecodeErrors` appears only in
# annotations, which `from __future__ import annotations` keeps as strings: not needed at runtime.

SERIES_ID = "SeriesInstanceUID"
# Percentiles for the intensity window. Robust to single bright-voxel outliers, which in MRI are the
# rule rather than the exception.
CLIP_PCT = (1.0, 99.0)


@dataclass(frozen=True)
class CacheSpec:
    """Everything that determines the pixels. The tag is its name on disk.

    The tag names every parameter, not just the dimensions: two configurations agreeing on resolution
    but not on crop produce arrays of the same shape and different content, and a tag built on
    dimensions alone would let the second attach to the first one's file with nothing reporting the
    difference.
    """
    img: int = 224
    slices: int = 10
    crop_mm: float = 130.0
    band: tuple[float, float] = (0.15, 0.85)
    center: str = "image"          # "image" = centre of the frame, "mass" = of the tissue
    # Second, optional scale. When present, every series is cropped twice and the study has 8 slots
    # instead of 4: the first four at `crop_mm`, the next four at `crop_mm2`. No branch in the model is
    # needed: `KneeBaseline` takes `n_slots` from `shape[0]`, already has a per-slot embedding and
    # per-finding attention, so "which scale to look at for which finding" is something it can already
    # learn. A scale is just one more sequence. The final model uses 110 and 150 mm.
    crop_mm2: float | None = None
    # Sequences used for the slots. If None, SEQUENCE_SLOTS (the 4 default slots with coverage >= 94%).
    # When declared explicitly (e.g. 5 slots with Coronal T1), the tag reflects the number of slots.
    slots: tuple[tuple[str, int, int], ...] | None = None
    # How a tie of the `Laterality` tags among the series of a study is resolved (2 R and 2 L). None =
    # the default rule, `max(set(...))`: on a tie the first element of the set wins, and the order of a
    # set of strings changes in every process (randomised hash), so the side is random at every run.
    # "geometry" = on a tie the geometry decides (`side_from_centers`), as for studies without tags.
    # Outside ties the two rules coincide.
    side_tie: str | None = None
    # Band of the SAGITTAL series measured from the lateral (0) to the medial (1) side, instead of `band`
    # (which stays for axial and coronal). None = the default rule, `band` symmetric on the stack indices.
    # The 0.15-0.85 band is centred on the knee, but on the medial side the model sees up to 30 mm from
    # the centre against 37 on the lateral side. The medial side is derived from the geometry
    # (`sagittal_band`), not from the slice order. An option, not used by the final model.
    sag_band_lm: tuple[float, float] | None = None
    # The side of AMBIGUOUS studies (no tag and geometry in the dead zone, or majority tag against the
    # geometry) is decided by the anatomy: comparison with the training template, flip if the vote says
    # "mirrored" (`knee.data.side_anat`). False = the default rule. Bit-identical elsewhere. An option,
    # not used by the final model.
    side_anat: bool = False

    def __post_init__(self) -> None:
        if self.side_tie not in (None, "geometry"):
            raise ValueError(f"side_tie: expected None or 'geometry', got {self.side_tie!r}")
        if self.sag_band_lm is not None:
            lo, hi = (float(b) for b in self.sag_band_lm)
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError(f"sag_band_lm: expected 0 <= lat < med <= 1, got {self.sag_band_lm!r}")
            object.__setattr__(self, "sag_band_lm", (lo, hi))
        # From YAML `slots` and `band` arrive as lists: `pick_slots` uses the triples as dictionary keys and
        # `tag` needs them stable, so here they become tuples once and for all, with integer flags (YAML
        # writes `1`, but `True` would also pass and break equality with `train_series.csv`).
        object.__setattr__(self, "band", tuple(float(b) for b in self.band))
        if self.slots is not None:
            object.__setattr__(self, "slots", tuple(
                (str(plane), int(fluid), int(fatsat)) for plane, fluid, fatsat in self.slots))

    @property
    def active_slots(self) -> list[tuple[str, int, int]]:
        return list(self.slots) if self.slots is not None else list(SEQUENCE_SLOTS)

    @property
    def tag(self) -> str:
        # The suffixes appear only when the value is not the default one: so the tags of already built caches
        # stay identical byte for byte, and an old run still finds its file.
        suffix = "" if self.center == "image" else f"_{self.center}"
        if self.crop_mm2 is not None:
            suffix += f"_x{int(self.crop_mm2)}mm"
        if self.slots is not None:
            suffix += f"_{len(self.slots)}slots"
        if self.side_tie is not None:
            suffix += "_tiegeo"
        if self.sag_band_lm is not None:
            suffix += f"_sag{self.sag_band_lm[0]:.2f}-{self.sag_band_lm[1]:.2f}"
        if self.side_anat:
            suffix += "_sideanat"
        return (f"{self.img}px_{self.slices}sl_{int(self.crop_mm)}mm_"
                f"{self.band[0]:.2f}-{self.band[1]:.2f}{suffix}")

    @property
    def n_scales(self) -> int:
        return 1 if self.crop_mm2 is None else 2

    @property
    def crops(self) -> tuple[float, ...]:
        """The scales in slot order: slot `j + 4*s` is sequence `j` at scale `s`."""
        return (self.crop_mm,) if self.crop_mm2 is None else (self.crop_mm, self.crop_mm2)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self.active_slots) * self.n_scales, self.slices, self.img, self.img)

    @property
    def bytes_per_study(self) -> int:
        return int(np.prod(self.shape))


def _count_files(p) -> int:
    try:
        return len(os.listdir(p)) if os.path.isdir(p) else 0
    except OSError:
        return 0


def _drop_slots_env() -> tuple[int, ...]:
    v = os.environ.get("KNEE_DROP_SLOTS", "").strip()
    return tuple(int(x) for x in v.split(",")) if v else ()


# Slots of `active_slots` the kernel does not use: `pick_slots` assigns them no series, which are
# therefore never listed or read, and the slot stays absent as in a study without that sequence.
# `KNEE_DROP_SLOTS=1`: without the non-fat-sat sagittal, ~27% fewer bytes (rejected: -0.0009 OOF).
DROP_SLOTS = [_drop_slots_env()]


def pick_slots(series: pd.DataFrame, n_files: dict[str, int] | None = None,
               slots: list[tuple[str, int, int]] | None = None,
               series_dirs: dict[str, str | Path] | None = None,
               threads: int = 16) -> dict[str, dict[int, str]]:
    """One series per slot per study -> {study: {slot_index: series_uid}}.

    On a tie within a slot the series with more slices wins: a denser stack samples the joint better
    and leaves room for band sampling. On equal slices the first in CSV order wins.
    With `series_dirs` and `n_files=None` the file count is lazy: the disk is scanned only for the
    series competing for the same slot, in parallel (I/O on the Kaggle mount, where every `listdir`
    costs milliseconds), with `threads` threads: the result does not depend on how many.

    Vectorised: the per-study version (one pandas filter per slot per study) cost ~6 s over the 1,300
    test studies; this one ~0.1 s, same result checked on the 4,407 training studies, ties and key
    order included.
    """
    active = slots if slots is not None else SEQUENCE_SLOTS
    key = {t: j for j, t in enumerate(active)}
    trip = zip(series["Anatomical_Plane"].tolist(), series["Fluid_Sensitive"].tolist(),
               series["Fat_Suppression"].tolist())
    df = pd.DataFrame({"study": series[ID].to_numpy(), "uid": series[SERIES_ID].to_numpy(),
                       "slot": [key.get(t, -1) for t in trip]})
    df["pos"] = np.arange(len(df))
    df = df[df["slot"] >= 0]
    if DROP_SLOTS[0]:
        df = df[~df["slot"].isin(DROP_SLOTS[0])]      # slots dropped by the kernel: their series are not read
    if n_files is not None:
        n = df["uid"].map(n_files)
    else:
        contested = df.loc[df.duplicated(["study", "slot"], keep=False), "uid"].tolist()
        counts: dict[str, int] = {}
        if contested and series_dirs:
            todo = [u for u in contested if u in series_dirs]
            with ThreadPoolExecutor(max_workers=threads) as ex:
                counts = dict(zip(todo, ex.map(lambda u: _count_files(series_dirs[u]), todo)))
        n = df["uid"].map(counts)
    df = df.assign(n=n.fillna(0).to_numpy())
    df = df.sort_values(["n", "pos"], ascending=[False, True], kind="stable")
    df = (df.drop_duplicates(["study", "slot"], keep="first")
            .sort_values(["slot", "pos"], kind="stable"))
    out: dict[str, dict[int, str]] = {s: {} for s in pd.unique(series[ID])}
    for s, j, u in zip(df["study"].tolist(), df["slot"].tolist(), df["uid"].tolist()):
        out[s][j] = u
    return out


# Percentiles without sorting the whole stack. `_to_uint8` was 36% of the inference workers' CPU
# (64 ms per call, two calls per series), half of it in the percentiles. The two order statistics
# needed lie in the lowest 1% and the highest 1%: sorting the tails is enough. The interpolation is
# NumPy's own (`_lerp`, `_get_gamma`), so the result is `np.percentile` bit for bit; where that cannot
# be guaranteed (NaN, tails too short, NumPy without those functions) `np.percentile` is used, and the
# first calls of every process compare the two paths.
try:                                            # NumPy 2
    from numpy.lib import _function_base_impl as _np_fb
except ImportError:                             # NumPy 1.22+
    from numpy.lib import function_base as _np_fb
_PCT_SAMPLE = 16384     # sample values from which the tail thresholds are estimated
_PCT_MARGIN = 0.01      # the threshold is one percentage point beyond the target quantile
_PCT_CHECKS = [3]       # comparisons with np.percentile left in this process
_PCT_FAST = [True]      # the fast path is still valid in this process
# The fast kernels (one pass for the tails, one for normalisation): the C library of `cfast` if
# `cfast.load()` loaded it in this process, otherwise the numba ones of `jit` if `jit.warmup()` compiled
# them; the first calls are compared bit for bit with NumPy (`_JIT_CHECKS`) and a disagreement turns
# them off. With NumPy `_to_uint8` takes ~7 passes over the stack; the fast version gave 24 studies out
# of 24 bit-identical, -19.7% CPU per study.
_JIT = [True]
_JIT_CHECKS = [3]
_TLS = threading.local()                # tail buffers, one per thread (series in parallel)


def _tail_buffers(n: int) -> tuple[np.ndarray, np.ndarray]:
    bufs = getattr(_TLS, "tails", None)
    if bufs is None or bufs[0].size < n:
        bufs = (np.empty(n, np.float32), np.empty(n, np.float32))
        _TLS.tails = bufs
    return bufs


def _use_jit():
    """The module of the fast kernels (`tails`, `normalize`), or None: the C library (`knee.data.cfast`,
    loaded by `cfast.load()`) if present, otherwise numba (`knee.data.jit`, compiled by `jit.warmup()`),
    otherwise the NumPy path.
    """
    if not _JIT[0]:
        return None
    from . import cfast, jit
    if cfast.ok():
        return cfast
    return jit if jit.ok() else None


def _tail_percentiles(flat: np.ndarray, pct=CLIP_PCT) -> np.ndarray | None:
    """`np.percentile(flat, pct)` (linear, two quantiles on either side of the median) sorting only the
    tails, or `None` if the same result cannot be guaranteed.
    """
    n = flat.size
    q = np.true_divide(np.asanyarray(pct), 100)
    if n < 4 or q.shape != (2,) or not (0 < q[0] < 0.5 < q[1] < 1):
        return None
    methods = getattr(_np_fb, "_QuantileMethods", None)
    if methods is None or not hasattr(_np_fb, "_lerp") or not hasattr(_np_fb, "_get_gamma"):
        return None
    method = methods["linear"]
    virtual = np.asanyarray(method["get_virtual_index"](n, q))
    prev_idx = np.floor(virtual).astype(np.intp)
    if prev_idx[1] + 1 >= n:
        return None
    k1, k2 = int(prev_idx[0]), int(prev_idx[1])
    fast = _use_jit() if flat.dtype == np.float32 and flat.flags.c_contiguous else None
    if np.isnan(flat.max()):                    # a NaN makes np.percentile return NaN
        return None
    sample = flat[::max(1, n // _PCT_SAMPLE)].copy()
    m = sample.size
    r_lo = min(m - 1, int(m * (q[0] + _PCT_MARGIN)))
    r_hi = max(0, int(m * (q[1] - _PCT_MARGIN)))
    sample.partition((r_lo, r_hi))
    low = high = None
    if fast is not None:
        # One pass: the same values as the two masks below, in the same order.
        b_lo, b_hi = _tail_buffers(n)
        n_lo, n_hi = fast.tails(flat, sample[r_lo], sample[r_hi], b_lo, b_hi)
        if n_lo <= b_lo.size and n_hi <= b_hi.size:
            low, high = b_lo[:n_lo], b_hi[:n_hi]
    if low is None:
        low = flat[flat <= sample[r_lo]]        # all the values up to the low threshold
        high = flat[flat >= sample[r_hi]]       # all the values from the high threshold up
    off = n - high.size
    if low.size < k1 + 2 or off > k2:           # the target ranks are not in the tails
        return None
    low.partition((k1, k1 + 1))
    high.partition((k2 - off, k2 + 1 - off))
    previous = np.array([low[k1], high[k2 - off]], dtype=flat.dtype)
    nxt = np.array([low[k1 + 1], high[k2 + 1 - off]], dtype=flat.dtype)
    gamma = _np_fb._get_gamma(virtual, prev_idx.astype(virtual.dtype), method)
    return _np_fb._lerp(previous, nxt, gamma)


def _clip_percentiles(stack: np.ndarray) -> np.ndarray:
    if _PCT_FAST[0]:
        try:
            fast = _tail_percentiles(stack.reshape(-1))
        except Exception:
            fast = None
        if fast is not None:
            if _PCT_CHECKS[0] <= 0:
                return fast
            _PCT_CHECKS[0] -= 1
            ref = np.percentile(stack, CLIP_PCT)
            if np.array_equal(ref, fast) and ref.dtype == fast.dtype:
                return fast
            _PCT_FAST[0] = False                # a disagreement: np.percentile only from now on
            return ref
    return np.percentile(stack, CLIP_PCT)


def _to_uint8(stack: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """The stack (K, H, W) to uint8 between the `CLIP_PCT` percentiles. With `out` (uint8, same shape,
    possibly a flipped view) it writes there one slice at a time: same operations in the same type, but
    the float64 of one slice (0.6 MB at 288 px) stays in cache instead of crossing RAM four times for
    16 MB.
    """
    lo, hi = _clip_percentiles(stack)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        if out is None:
            return np.zeros(stack.shape, dtype=np.uint8)
        out[...] = 0
        return out
    # `np.clip((stack - lo) * (255.0 / (hi - lo)), 0, 255)` with the same operations in the same type (the
    # one promotion picks for `stack - lo`: float64 with NumPy 2, float32 with NumPy 1), but in a single
    # buffer instead of three temporaries.
    if out is None:
        y = np.subtract(stack, lo)
        np.multiply(y, 255.0 / (hi - lo), out=y)
        np.clip(y, 0, 255, out=y)
        return y.astype(np.uint8)
    scale = 255.0 / (hi - lo)
    dtype = np.subtract(stack.reshape(-1)[:1], lo).dtype
    fast = (_use_jit() if dtype == np.float64 and stack.dtype == np.float32 and stack.ndim == 3
            and stack.flags.c_contiguous and out.dtype == np.uint8 else None)
    if fast is not None:
        if _JIT_CHECKS[0] <= 0:
            fast.normalize(stack, lo, scale, out)
            return out
        _JIT_CHECKS[0] -= 1
        ref = _normalize_numpy(stack, lo, scale, np.empty(stack.shape, np.uint8), dtype)
        fast.normalize(stack, lo, scale, out)
        if np.array_equal(ref, out):
            return out
        _JIT[0] = False                         # a disagreement: NumPy only from now on
        out[...] = ref
        return out
    return _normalize_numpy(stack, lo, scale, out, dtype)


def _normalize_numpy(stack: np.ndarray, lo, scale, out: np.ndarray, dtype) -> np.ndarray:
    buf = np.empty(stack.shape[1:], dtype=dtype)
    for i in range(stack.shape[0]):
        np.subtract(stack[i], lo, out=buf)
        np.multiply(buf, scale, out=buf)
        np.clip(buf, 0, 255, out=buf)
        np.copyto(out[i], buf, casting="unsafe")          # like `astype(np.uint8)`
    return out


# Downscaling (INTER_AREA) with the C library (`cfast.resize_area`: OpenCV's generic path redone bit
# for bit, 8 rows at a time with AVX; the resize was 31% of the worker CPU). The first calls of every
# process are compared bit for bit with cv2; a disagreement turns it off in that process. Upscaling
# (INTER_LINEAR) stays on cv2: for float32 it goes through IPP, which cannot be reproduced bit for bit.
_CRESIZE = [True]
_CRESIZE_CHECKS = [8]


def _area_c(a: np.ndarray, img: int, dst: np.ndarray) -> bool:
    if not _CRESIZE[0] or a.dtype != np.float32:
        return False
    from . import cfast
    if not cfast.ok() or cfast.resize_area(a, img, img, dst) != 0:
        return False
    if _CRESIZE_CHECKS[0] > 0:
        _CRESIZE_CHECKS[0] -= 1
        ref = cv2.resize(a, (img, img), interpolation=cv2.INTER_AREA)
        if not np.array_equal(ref, dst):
            _CRESIZE[0] = False
            dst[...] = ref
    return True


def _resize(a: np.ndarray, img: int) -> np.ndarray:
    interp = cv2.INTER_AREA if a.shape[0] > img else cv2.INTER_LINEAR
    if interp == cv2.INTER_AREA:
        out = np.empty((img, img), np.float32)
        if _area_c(a, img, out):
            return out
    return cv2.resize(a, (img, img), interpolation=interp)


def _resize_into(a: np.ndarray, img: int, dst: np.ndarray) -> None:
    """`_resize` written into `dst` (float32 (img, img), contiguous) instead of a new array. If cv2 could
    not use `dst` and allocated elsewhere, it is copied.
    """
    interp = cv2.INTER_AREA if a.shape[0] > img else cv2.INTER_LINEAR
    if interp == cv2.INTER_AREA and _area_c(a, img, dst):
        return
    r = cv2.resize(a, (img, img), dst=dst, interpolation=interp)
    if r is not dst and r.ctypes.data != dst.ctypes.data:
        dst[...] = r


# Fewer copies in the decoding workers (`build_series`), with an identical result. After `mallopt`
# (page faults -81%, steady state -18.7%) the remaining cost is the memory traversed: per series the
# crop copied to float32 even if it already is (~36 MB), `np.stack` of the 24 resized slices (16 MB),
# the float64 of `_to_uint8` (~200 MB of traffic), the uint8, the flipped copy and the copy into the
# study volume. "lean": the crop goes to cv2 as a view, the resize writes into the stack, `_to_uint8`
# works one slice at a time and writes straight into the volume (already flipped). The first series
# of every process also redo the previous path on the same pixels and compare bit for bit; a
# disagreement turns "lean" off in that process. Eight series = the first two whole studies of a
# worker (the four series of a study run in parallel). "lean" is the default: 60 real studies out of
# 60 bit-identical, worker CPU -6.1%, steady state -7.8%.
_COPIES = ["lean"]
_COPIES_CHECKS = [8]
_COPIES_OK = [True]
# The repeated slices of the band (short series) read, decoded and resized once (`build_series`):
# 3,433 distinct slices out of 3,792 requested.
_DEDUP = [True]


# Below this distance between the first and last slice along x the stack does not say where the medial
# side is (a real sagittal covers ~100 mm): fall back to the symmetric band.
SAG_MIN_SPAN_MM = 10.0


def sagittal_band(x_first: float | None, x_last: float | None, side: str | None,
                  spec: CacheSpec) -> tuple[tuple[float, float], str]:
    """The band (as fractions of the indices of the ordered stack) of a sagittal series.

    `x_first`/`x_last`: the patient x (LPS, increasing towards the patient's left) of the first and last
    slice in `order_series_files` order. The medial side of a left knee is towards the patient's right
    (smaller x), that of a right knee towards the left (larger x). Also returns where the medial side
    ended up ("high" = high indices, "low" = low) or why the symmetric band was used.
    """
    if spec.sag_band_lm is None:
        return spec.band, ""
    if side not in ("L", "R"):
        return spec.band, "fallback_side"
    if x_first is None or x_last is None or abs(x_last - x_first) < SAG_MIN_SPAN_MM:
        return spec.band, "fallback_geometry"
    medial_high = (x_last < x_first) if side == "L" else (x_last > x_first)
    lo, hi = spec.sag_band_lm
    if medial_high:
        return (lo, hi), "high"
    return (round(1.0 - hi, 9), round(1.0 - lo, 9)), "low"


def _stack_ends_x(series_dir: str, files: list[str]) -> tuple[float | None, float | None]:
    """The patient x of the first and last slice: two headers, read as for the ordering."""
    from .dicom import _as_floats, _read_header, fast_order_tags

    out: list[float | None] = []
    for f in (files[0], files[-1]):
        path = os.path.join(series_dir, f)
        ipp = None
        try:
            fast = fast_order_tags(path)
            ipp = _as_floats(fast[0], 3) if fast is not None else None
            if ipp is None:
                ipp = _as_floats(getattr(_read_header(path), "ImagePositionPatient", None), 3)
        except Exception:
            ipp = None
        out.append(float(ipp[0]) if ipp is not None else None)
    return out[0], out[1]


def build_series(series_dir: str, plane: str, side: str | None, spec: CacheSpec,
                 errors: DecodeErrors | None = None, hdr=None,
                 crops: tuple[float, ...] | None = None,
                 to_uint8: bool = True,
                 out: list[np.ndarray] | None = None) -> tuple[list[np.ndarray] | None, dict]:
    """One series -> a list of `(slices, img, img)` uint8, one per crop scale.

    `out`: the uint8 destinations (the slices of the study volume), used with `_COPIES = "lean"`; in
    that case the returned list is `out` itself and `info["in_out"]` is True.

    `to_uint8=False`: the float32 stacks before `_to_uint8` (already flipped: percentiles and
    normalisation do not see the pixel order), to normalise them on the GPU with
    `torch_uint8.to_uint8_torch`, which gives the same uint8 values bit for bit.

    The scales are cropped **within the same read**: decoding the DICOMs is 71% of the cache work, and
    calling this function twice for two boxes would double exactly the slow part. From here on only
    `crop_box` changes, so the two scales share file order, band, decoded slices and lateralisation,
    which is why they are comparable with each other rather than two different experiments.
    """
    from . import dicom as dicom_mod
    from .dicom import order_series_files, probe_series, read_slice

    crops = (spec.crop_mm,) if crops is None else crops
    order = order_series_files(series_dir, keep_fds=dicom_mod._FD_KEEP[0])
    try:
        return _build_series_ordered(series_dir, plane, side, spec, errors, hdr, crops,
                                     to_uint8, out, order, probe_series, read_slice)
    finally:
        dicom_mod.close_kept(series_dir, order.files)


def _build_series_ordered(series_dir, plane, side, spec, errors, hdr, crops, to_uint8, out,
                          order, probe_series, read_slice):
    """The body of `build_series` after the ordering (files still open are closed on exit anyway, see
    `dicom._FD_CACHE`).
    """
    if not order.files:
        return None, {"order": "empty", "n_read": 0}
    raw, sag = _read_band(series_dir, plane, side, spec, errors, order, read_slice)
    n_read = sum(a is not None for a in raw)
    if n_read == 0:
        return None, {"order": order.method, "n_read": 0}

    # A slice that fails to decode takes the first one that decoded, so the shape of the stack stays the
    # declared one: a hole would change the meaning of the slice index, which downstream is an anatomical
    # position.
    fill = next(a for a in raw if a is not None)
    raw = [fill if a is None else a for a in raw]

    ps = (hdr if hdr is not None else probe_series(series_dir)).pixel_spacing
    return _build_series_from_raw(raw, ps, plane, side, spec, crops, to_uint8, out, order, n_read,
                                  sag)


def _read_band(series_dir, plane, side, spec, errors, order, read_slice):
    """The `spec.slices` slices of the band of an ordered series, decoded (None where they fail), and which
    side the sagittal band ended on (`sagittal_band`).
    """
    band, sag = spec.band, ""
    if spec.sag_band_lm is not None and plane == "Sagittal":
        # Only with the physical order are the first and last files the ends of the stack.
        ends = (_stack_ends_x(series_dir, order.files) if order.method == "projection"
                else (None, None))
        band, sag = sagittal_band(ends[0], ends[1], side, spec)
    idx = band_indices(len(order.files), spec.slices, band)
    # Slices in parallel: the time is mostly waiting for the mount (opening and reading a ~150 KB file),
    # not CPU. The output order is that of `idx`. On a short series `band_indices` repeats indices (9.5% of
    # the band slices): with `_DEDUP` every file is read and decoded once and the repetitions are the same
    # array (then also the same resize, below). Same pixels.
    if _DEDUP[0]:
        uniq = list(dict.fromkeys(int(i) for i in idx))
        paths = [os.path.join(series_dir, order.files[i]) for i in uniq]
        with thread_pool("slices", min(len(paths), SLICE_THREADS)) as ex:
            got = dict(zip(uniq, ex.map(lambda q: read_slice(q, errors), paths)))
        raw = [got[int(i)] for i in idx]
    else:
        paths = [os.path.join(series_dir, order.files[int(i)]) for i in idx]
        with thread_pool("slices", min(len(paths), SLICE_THREADS)) as ex:
            raw = list(ex.map(lambda q: read_slice(q, errors), paths))
    return raw, sag


def _build_series_from_raw(raw, ps, plane, side, spec, crops, to_uint8, out, order, n_read, sag):
    """The rest of `build_series` from the decoded slices (without holes) on."""
    # The centre is computed once per series, on the middle slice, and holds for all slices: computing it
    # slice by slice would make the box wobble along the volume, and the third dimension would stop being
    # an anatomical position.
    center = foreground_center(raw[len(raw) // 2]) if spec.center == "mass" else None
    offset = None
    fov, mm_px = None, None
    lean = (to_uint8 and out is not None and _COPIES[0] == "lean" and _COPIES_OK[0]
            and len(out) == len(crops)
            and all(o.shape == (len(raw), spec.img, spec.img) and o.dtype == np.uint8
                    for o in out))
    outs: list[list[np.ndarray]] = [[] for _ in crops]
    views: list[list[np.ndarray]] = [[] for _ in crops]
    stacks = [np.empty((len(raw), spec.img, spec.img), np.float32) for _ in crops] if lean else None
    # A repeated slice (same array: repeated index or fallback) is resized once.
    done_at: dict[int, int] = {}
    for i, a in enumerate(raw):
        j = done_at.get(id(a)) if _DEDUP[0] else None
        if j is None:
            done_at[id(a)] = i
        if ps is not None:
            fov = min(fov_mm(a.shape[0], a.shape[1], ps))
            mm_px = min(float(ps[0]), float(ps[1]))
            if center is not None and offset is None:
                # How far the centroid is from the centre of the frame, in millimetres. This is the measurement that
                # says whether the experiment could change anything: a median offset of two millimetres answers it
                # before training, as `crop_fit` does for the crop.
                offset = float(np.hypot((center[0] - (a.shape[0] - 1) / 2.0) * float(ps[0]),
                                        (center[1] - (a.shape[1] - 1) / 2.0) * float(ps[1])))
        for k, crop in enumerate(crops):
            # `a` is never reassigned: with several scales the second box would crop what the first already
            # cropped.
            cropped = a
            if ps is not None:
                y0, y1, x0, x1 = crop_box(a.shape[0], a.shape[1], ps, crop, center)
                cropped = a[y0:y1, x0:x1]
            if lean:
                views[k].append(cropped)
                if j is not None:
                    stacks[k][i] = stacks[k][j]
                else:
                    _resize_into(cropped.astype(np.float32, copy=False), spec.img, stacks[k][i])
            else:
                outs[k].append(outs[k][j] if j is not None
                               else _resize(cropped.astype(np.float32), spec.img))
    # The intensity window is computed per scale. They are the same pixels seen through two boxes: a wide
    # box includes more background, and imposing on it the percentiles of the tight one would shift the
    # contrast of one scale for a non-anatomical reason.
    info = {"order": order.method, "n_read": n_read, "fov_mm": fov, "mm_px": mm_px,
            "offset_mm": offset, "in_out": lean, "sag_band": sag}
    if lean:
        for k in range(len(crops)):
            _to_uint8(stacks[k], out=normalise_laterality(out[k], plane, side))
        if _COPIES_CHECKS[0] > 0:
            _COPIES_CHECKS[0] -= 1
            ref = [normalise_laterality(
                _to_uint8(np.stack([_resize(c.astype(np.float32), spec.img) for c in v])),
                plane, side) for v in views]
            if not all(np.array_equal(r, o) for r, o in zip(ref, out)):
                _COPIES_OK[0] = False
                for r, o in zip(ref, out):
                    o[...] = r                          # the previous path applies
        return out, info
    vols = [normalise_laterality(_to_uint8(np.stack(o)) if to_uint8 else np.stack(o),
                                 plane, side) for o in outs]
    # `fov` is the short side of the field of view in millimetres. Below `crop_mm` the crop cut nothing
    # and this series kept its own physical scale: the only way to know whether a spec says "150 mm" or
    # "150 mm where it fitted".
    #
    # `mm_px` is the *acquired* pixel spacing. It answers the second question, which the field of view
    # alone does not touch: `crop_mm / mm_px` are the real pixels inside the box, and if they are more
    # than `img` the resize throws some away. A tighter crop buys resolution only while that ratio stays
    # above `img`; below it, it just magnifies.
    return [np.ascontiguousarray(v) for v in vols], info


def study_side(headers, tie: str | None = None) -> tuple[str | None, str]:
    """Knee side: the `Laterality` tag if present, otherwise the geometry.

    Also returns where it came from, because the coverage of the two sources is a measurement we want
    in the cache index, not an internal detail.

    `tie` (`CacheSpec.side_tie`) resolves tag ties. With None the default rule applies, which is not
    deterministic on a tie (see `CacheSpec.side_tie`); with "geometry" the tie goes to the geometry and
    the source is written `tie-geometry` (or `tie-unknown` if the geometry does not decide: the study
    is not flipped, like the `unknown` ones).
    """
    lats = [h.laterality for h in headers if h.laterality in ("L", "R")]
    centers = [h.center for h in headers if h.center is not None]
    if lats:
        n_l, n_r = lats.count("L"), lats.count("R")
        if n_l != n_r or tie is None:
            return max(set(lats), key=lats.count), "tag"
        side = side_from_centers(centers)
        return side, ("tie-geometry" if side else "tie-unknown")
    side = side_from_centers(centers)
    return side, ("geometry" if side else "unknown")


def build_study(study_dir: str, slots: dict[int, str], spec: CacheSpec,
                errors: DecodeErrors | None = None,
                to_uint8: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """One study -> `(vol (S,K,P,P) uint8, mask (S,) bool, meta)`; with `to_uint8=False` `vol` is float32,
    the stacks before normalisation (`build_series`).
    """
    from .dicom import probe_series

    vol = np.zeros(spec.shape, dtype=np.uint8 if to_uint8 else np.float32)
    mask = np.zeros(spec.shape[0], dtype=bool)
    items = list(slots.items())
    # The series of the study in parallel, and so the probes: every series is mostly waiting for the
    # mount (0.94 s/study in one process, two thirds of it in `join` of reader threads), and the four do
    # not depend on each other. The result does not change: every slot writes its own part of `vol`.
    # One probe per series: the header serves both the study side and the crop, and reading it twice
    # would double the slowest part, the accesses to the mount.
    with thread_pool("study", len(items)) as ex:
        hdrs = dict(zip([j for j, _ in items],
                        ex.map(lambda ju: probe_series(os.path.join(study_dir, ju[1]), ju[1]),
                               items)))
        side, side_src = study_side(list(hdrs.values()), tie=spec.side_tie)
        n_seq = len(spec.active_slots)
        # With `_COPIES = "lean"` every series writes straight into its slices of `vol`.
        dsts = ({j: [vol[j + k * n_seq] for k in range(spec.n_scales)] for j, _ in items}
                if to_uint8 and _COPIES[0] == "lean" else {})
        built = list(ex.map(
            lambda ju: build_series(os.path.join(study_dir, ju[1]), spec.active_slots[ju[0]][0],
                                    side, spec, errors, hdr=hdrs[ju[0]], crops=spec.crops,
                                    to_uint8=to_uint8, out=dsts.get(ju[0])),
            items))
    meta = {"side": side or "", "side_source": side_src, "n_read": 0, "orders": {},
            "fov_mm": {}, "mm_px": {}, "offset_mm": {}, "sag_band": {}}
    for (j, uid), (vs, info) in zip(items, built):
        meta["orders"][j] = info["order"]
        if info.get("sag_band"):
            meta["sag_band"][j] = info["sag_band"]
        meta["fov_mm"][j] = info.get("fov_mm")
        meta["mm_px"][j] = info.get("mm_px")
        meta["offset_mm"][j] = info.get("offset_mm")
        meta["n_read"] += info["n_read"]
        if vs is not None:
            # Slot `j + n_seq*k` is sequence `j` at scale `k`: the order is that of `spec.crops`, which is what
            # makes `slot_emb` interpretable afterwards.
            for k, v in enumerate(vs):
                if not info.get("in_out"):
                    vol[j + k * n_seq] = v
                mask[j + k * n_seq] = True
    if spec.side_anat:
        from .side_anat import resolve
        side, side_src, meta["side_anat"] = resolve(vol, mask, spec, list(hdrs.values()),
                                                    side, side_src)
        meta["side"], meta["side_source"] = side or "", side_src
    return vol, mask, meta


# The crop thresholds the grid reports on. They cover both directions in which `crop_mm` can move:
# towards coverage and towards resolution. The bottom of the scale goes down to 80 because the tight
# direction is the one that paid, and a grid that stops where the next run wants to go does not help
# decide.
CROP_GRID = (80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200)


def crop_fit(fovs: list[float], crop_mm: float, spacings: list[float] | None = None,
             img: int | None = None) -> dict:
    """The two measurements without which a crop experiment cannot be interpreted.

    **Coverage.** `crop_box` does not raise when the requested crop is larger than the field of view: it
    returns the whole image, and that series stays at a physical scale different from all the cropped
    ones. A silent failure that grows with `crop_mm`, so an experiment that widens the crop must report
    how much it grew; otherwise the variable that changed is not "the crop" but "the crop plus an
    unknown fraction of mixed-scale series".

    **Resolution.** The other half, which the field of view does not touch: `crop_mm / mm_px` are the
    *acquired* pixels inside the box. If they are more than `img` the resize throws them away and
    tightening the crop recovers real detail; if fewer, the resize interpolates and tightening the crop
    only magnifies. Two opposite readings of the same result, told apart only by this number.

    The grid answers for every threshold in a single build: the cache is built once and says what would
    have happened at the sizes not tried as well.
    """
    if not fovs:
        return {"n_series": 0}
    a = np.asarray(fovs, dtype=np.float64)
    out = {
        "n_series": int(a.size),
        "fits_at_crop_mm": round(float((a >= crop_mm).mean()), 4),
        "fits_by_threshold": {str(t): round(float((a >= t).mean()), 4) for t in CROP_GRID},
        "fov_mm_pct": {p: round(float(np.percentile(a, p)), 1)
                       for p in (1, 5, 25, 50, 75, 95, 99)},
    }
    if spacings and img:
        s = np.asarray([v for v in spacings if v], dtype=np.float64)
        px = np.minimum(crop_mm, a[:len(s)]) / s          # acquired pixels in the box
        out["resample"] = {
            "mm_px_pct": {p: round(float(np.percentile(s, p)), 3)
                          for p in (5, 25, 50, 75, 95)},
            "crop_px_pct": {p: round(float(np.percentile(px, p)), 1)
                            for p in (5, 25, 50, 75, 95)},
            # Above `img` the resize discards acquired detail: that is where tightening the crop buys real
            # resolution instead of magnifying.
            "downsampled_at_crop_mm": round(float((px > img).mean()), 4),
            "downsampled_by_threshold": {
                str(t): round(float((np.minimum(t, a[:len(s)]) / s > img).mean()), 4)
                for t in CROP_GRID},
        }
    return out


def count_files(series_dirs: dict[str, str], workers: int = 16) -> dict[str, int]:
    """Number of slices per series, to break ties within a slot.

    One `listdir` per series, thousands of them: in parallel because it is pure network latency, not
    compute, which is why there are more threads than cores.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(item):
        uid, p = item
        try:
            return uid, len(os.listdir(p))
        except OSError:
            return uid, 0
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, series_dirs.items()))


def duplicate_cache_slots(work_dir: str | Path, spec_base: CacheSpec, spec_out: CacheSpec) -> None:
    """Duplicates the 4 slots of `spec_base` to create the 8 slots of `spec_out` (a spec with
    crop_mm2 == crop_mm).

    Bit-identical to extracting the cache from the raw DICOMs with two identical scales: with equal
    crops, the two scales get the same crop_box, resize and normalisation. Instead of ~60 min of CPU, it
    generates the duplicated array in ~30 seconds.
    """
    import json
    import shutil

    work = Path(work_dir)
    tag_base = spec_base.tag
    tag_out = spec_out.tag

    v_base = work / f"vols_{tag_base}.npy"
    m_base = work / f"mask_{tag_base}.npy"
    i_base = work / f"index_{tag_base}.csv"

    if not v_base.is_file():
        raise FileNotFoundError(f"missing {v_base} for the duplication")
    if not m_base.is_file():
        raise FileNotFoundError(f"missing {m_base} for the duplication")

    vols_in = np.load(v_base, mmap_mode="r")
    n = len(vols_in)
    stage = work / ".staging_dup"
    stage.mkdir(parents=True, exist_ok=True)

    vols_out = np.lib.format.open_memmap(stage / f"vols_{tag_out}.npy", mode="w+",
                                         dtype=np.uint8, shape=(n,) + spec_out.shape)
    chunk = 500
    n_seq = len(spec_base.active_slots)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        blk = vols_in[start:end]
        vols_out[start:end, :n_seq] = blk
        vols_out[start:end, n_seq:] = blk
    vols_out.flush()
    del vols_out

    mask_in = np.load(m_base)
    mask_out = np.concatenate([mask_in, mask_in], axis=1)
    np.save(stage / f"mask_{tag_out}.npy", mask_out)

    if i_base.is_file():
        shutil.copy2(i_base, stage / f"index_{tag_out}.csv")

    b_base = work / f"build_{tag_base}.json"
    if b_base.is_file():
        summary = json.loads(b_base.read_text(encoding="utf-8"))
        summary["tag"] = tag_out
        summary["spec"] = spec_out.__dict__
        if "slot_coverage" in summary:
            crops = spec_out.crops
            summary["slot_coverage"] = {
                f"{p}/F{f}/S{s}" + ("" if k == 0 else f"@{int(c)}mm"):
                float(mask_out[:, j + k * n_seq].mean())
                for k, c in enumerate(crops)
                for j, (p, f, s) in enumerate(spec_base.active_slots)
            }
        (stage / f"build_{tag_out}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for f in stage.iterdir():
        f.replace(work / f.name)
    shutil.rmtree(stage, ignore_errors=True)

