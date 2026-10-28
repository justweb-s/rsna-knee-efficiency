"""DICOM geometry: physical slice order, knee side, crop in millimetres.

Three problems that look like details and are not.

**Slice order.** The name of a DICOM file here is a SOP Instance UID, an arbitrary number: sorting
by name gives a sequence unrelated to anatomy. Stacking three "neighbouring" slices into three
channels without ordering them first does not give the model the context above and below, it
gives it three random views. The physical order is recoverable exactly: the normal to the plane is
the cross product of the row and column direction cosines, and the projection of the position on
that normal is a monotonic coordinate along the stack.

**The side.** `Laterality` (0020,0060) is Type 2C and is missing in about half of the studies.
Treating a knee without the tag as a left knee leaves half the corpus unnormalised exactly on the
medial-lateral axis, i.e. on the five targets that have a side (both menisci, both tibiofemoral
compartments, the MCL). The patient coordinate system solves it without the tag: +x is the
patient's left, so the centre of a right knee lies at negative x. The *centre* of the image is
used, not `ImagePositionPatient`, which is the corner and is shifted by half a field of view.

**The crop.** A central crop larger than the field of view silently does nothing and leaves that
series at a physical scale different from all the others. The crop is expressed in millimetres and
applied only when it fits.

The resize brings everything to the same *shape*, not the same scale: a series with a 140 mm field
of view kept whole ends at 0.625 mm per pixel at 224 px, one cut at 150 mm at 0.670. At 130 mm
the fallback affects 7 series out of 17,071; at 150 mm, 1,532 (9%). The invariant "the crop is in
millimetres" holds as long as `crop_mm` is below the field of view of practically everything, and
stops holding silently. `fov_mm` exists to make that boundary visible; `crop_fit` in `cache.py`
counts it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# Beyond this distance from the midline the sign of x decides the side; below it, the study stays
# unresolved instead of being guessed. The dead zone is wide because a knee near the midline carries
# no side information, and a wrong normalisation is worse than none: it flips the medial-lateral
# axis.
MIDLINE_DEAD_ZONE_MM = 20.0


def slice_normal(iop: np.ndarray) -> np.ndarray:
    """Normal to the acquisition plane, from the six direction cosines of ImageOrientationPatient."""
    iop = np.asarray(iop, dtype=np.float64).reshape(6)
    return _normal_of(tuple(iop.tolist())).copy()


# The normal depends only on the six cosines, the same for every slice of a series: `np.cross` and
# `np.linalg.norm` per file were ~17% of the worker CPU (NumPy overhead on 3-element arrays). Same
# formula, computed once per tuple.
@lru_cache(maxsize=4096)
def _normal_of(iop: tuple) -> np.ndarray:
    v = np.asarray(iop, dtype=np.float64)
    n = np.cross(v[:3], v[3:])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else n


def through_plane_key(ipp: np.ndarray, iop: np.ndarray) -> float:
    """Signed coordinate along the stack: the projection of the position on the normal.

    It is signed in the patient system, so two series of the same plane acquired in opposite
    directions end up in the same order, which is exactly what side normalisation needs to mean
    anything.
    """
    return float(np.dot(np.asarray(ipp, dtype=np.float64).reshape(3), slice_normal(iop)))


def image_center_xyz(ipp, iop, pixel_spacing, rows: int, cols: int) -> np.ndarray:
    """Centre of the image in patient coordinates.

    `ImagePositionPatient` is the centre of the *first pixel*, i.e. a corner. The centre is obtained
    by moving half an image along both in-plane axes, each scaled by its own pixel spacing (which in
    DICOM is [row, column] = [dy, dx]).
    """
    ipp = np.asarray(ipp, dtype=np.float64).reshape(3)
    iop = np.asarray(iop, dtype=np.float64).reshape(6)
    dy, dx = (float(pixel_spacing[0]), float(pixel_spacing[1]))
    return ipp + iop[3:] * (dy * (rows - 1) / 2.0) + iop[:3] * (dx * (cols - 1) / 2.0)


def side_from_centers(centers: list[np.ndarray]) -> str | None:
    """'L' / 'R' / None from the sign of the median x of the series centres of a study.

    The median over the series is used instead of a single centre: an arbitrary sagittal slice can be
    anywhere across the joint.
    """
    xs = [float(c[0]) for c in centers if c is not None and np.isfinite(c).all()]
    if not xs:
        return None
    x = float(np.median(xs))
    if abs(x) < MIDLINE_DEAD_ZONE_MM:
        return None
    return "L" if x > 0 else "R"


def normalise_laterality(vol: np.ndarray, plane: str, side: str | None) -> np.ndarray:
    """Brings every knee to the left-knee convention. `vol` is (slice, y, x).

    Coronal and axial images are mirror images: the horizontal axis is flipped. Sagittal ones are not
    (one is not the mirror image of the other, it is the same view traversed from medial to lateral in
    the opposite direction), so the slice order is reversed, not the pixels.
    """
    if side != "R":
        return vol
    return vol[:, :, ::-1] if plane in ("Coronal", "Axial") else vol[::-1]


def crop_box(rows: int, cols: int, pixel_spacing, crop_mm: float,
             center: tuple[float, float] | None = None) -> tuple[int, int, int, int]:
    """Box of `crop_mm` millimetres per side -> (y0, y1, x0, x1).

    Without `center` the box is centred on the image: the knee is assumed to be in the middle of the
    frame. With `center` it is centred where requested (see `foreground_center`) and the box is
    shifted, not shrunk, if it would fall outside.

    If the field of view is smaller than the requested crop the whole image is returned: the same as
    a crop larger than the image would do, but made explicit. The downstream resize brings everything
    to the same shape, **not** to the same physical scale: that series stays at
    `min(fov, crop_mm)` millimetres over the output pixels. Whoever raises `crop_mm` must look at
    `crop_fit` before the metrics, because the share of series in this state grows and the variable
    stops being a single one.
    """
    dy, dx = (float(pixel_spacing[0]), float(pixel_spacing[1]))
    h = min(rows, max(1, int(round(crop_mm / dy)))) if dy > 0 else rows
    w = min(cols, max(1, int(round(crop_mm / dx)))) if dx > 0 else cols
    cy = (rows - 1) / 2.0 if center is None else float(center[0])
    cx = (cols - 1) / 2.0 if center is None else float(center[1])
    # The box stays inside the image: a centre near the border shifts it instead of cutting a smaller
    # one, because the size in millimetres is the invariant and the position is not.
    y0 = int(min(max(round(cy - (h - 1) / 2.0), 0), rows - h))
    x0 = int(min(max(round(cx - (w - 1) / 2.0), 0), cols - w))
    return y0, y0 + h, x0, x0 + w


def fov_mm(rows: int, cols: int, pixel_spacing) -> tuple[float, float]:
    """Field of view of the series in millimetres -> (height, width).

    Needed to know *when* `crop_box` stopped cropping. A crop larger than the field of view raises
    nothing: it returns the whole image and that series stays at a physical scale different from all
    those the crop actually cut. Measuring the field of view is the only way to tell "crop at N mm"
    from "crop at N mm where it fitted".
    """
    dy, dx = (float(pixel_spacing[0]), float(pixel_spacing[1]))
    return (rows * dy, cols * dx)


def band_indices(n: int, k: int, band: tuple[float, float]) -> np.ndarray:
    """`k` indices evenly spread within a central band of a stack of `n`.

    The band exists because the first and last slices of a series are often outside the joint. With a
    small `n` the indices repeat instead of leaving the band: a short stack is sampled with
    repetition, not widened.
    """
    lo = int(np.floor(band[0] * n))
    hi = int(np.ceil(band[1] * n))
    lo = max(0, min(lo, n - 1))
    hi = max(lo + 1, min(hi, n))
    return np.clip(np.round(np.linspace(lo, hi - 1, k)).astype(int), 0, n - 1)


# Percentile separating background from tissue for `foreground_center`. In MRI air is not exactly
# zero (background noise, artefacts), so a threshold at zero would include all the air; the 20th
# percentile of the slice is below tissue in every series of the corpus and above noise in all but
# the nearly empty ones, where falling back to the geometric centre is the right answer anyway.
BACKGROUND_PCT = 20.0


def foreground_center(a) -> tuple[float, float]:
    """Tissue centroid of a slice -> (row, column), in pixels.

    `crop_box` crops the centre of the **image**, which is right only if the knee is centred in the
    frame. When it is not, a tight box cuts on one side and leaves air on the other.

    The centroid is weighted by the intensity above the background, not by the binary mask: so a rim of
    noise does not count as much as the femur. If the slice is flat or empty the geometric centre is
    returned.
    """
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"a 2D slice is required, not {getattr(a, 'shape', None)}")
    w = a - float(np.percentile(a, BACKGROUND_PCT))
    np.clip(w, 0, None, out=w)
    total = float(w.sum())
    if total <= 0:
        return ((a.shape[0] - 1) / 2.0, (a.shape[1] - 1) / 2.0)
    ys = np.arange(a.shape[0], dtype=np.float32)
    xs = np.arange(a.shape[1], dtype=np.float32)
    return (float(w.sum(axis=1) @ ys / total), float(w.sum(axis=0) @ xs / total))
