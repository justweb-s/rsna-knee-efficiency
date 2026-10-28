"""The anatomical side: where tags and geometry are not enough, the image decides (optional, not
used by the final configuration).

After laterality normalisation all knees share the same orientation, so a study looks more like
the mean of the others (the template) than like the mirrored mean. In the training set 56 studies
reached the network mirrored: 36 of the 75 without a side (`unknown`, never flipped) and 20 of
the 25 whose tag disagrees with the geometry. On the `unknown` ones the test agrees with the
report in 62 cases out of 64.

Only on studies with an AMBIGUOUS side (unknown side, or majority tag against the geometry) the
volume is built with the side of the default rule, compared with the template and, if the vote
says "mirrored" by at least `MIN_VOTE` slots, flipped. Everywhere else nothing changes, not even
one pixel.

The flip after the fact is exact: in `build_series` the side enters only `normalise_laterality`,
applied last on the already cropped box (the side-dependent sagittal band, `sag_band_lm`, is
explicitly excluded).

The template (`side_anat_template.npz`, next to this file) is the mean of the training studies
whose side comes from tags or geometry, minus those a first pass flagged as mirrored; float16,
(slot, slices, 72, 72).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry import side_from_centers

TEMPLATE = Path(__file__).with_name("side_anat_template.npz")
MIN_VOTE = 4          # slot margin needed to flip (fixed before use: vote <= -4)
_PLANE_AXIS = {"Axial": 2, "Coronal": 2, "Sagittal": 0}   # mirror axis on (slices, y, x)


def ambiguity(headers, side_src: str) -> str | None:
    """Why the side of a study is ambiguous, or None if it is not.

    `unknown`: no tag and geometry in the dead zone (also `tie-unknown`). `conflict`: the majority
    tag says one side and the geometry the other. `tie`: a tie of the tags decided by the default
    rule (`side_tie` None, source "tag"), i.e. by the process hash. A tie decided by the geometry
    (`tie-geometry`) is not ambiguous.
    """
    if side_src in ("unknown", "tie-unknown"):
        return "unknown"
    lats = [h.laterality for h in headers if h.laterality in ("L", "R")]
    n_l, n_r = lats.count("L"), lats.count("R")
    if n_l == n_r:
        return "tie" if n_l > 0 and side_src == "tag" else None
    tag = "L" if n_l > n_r else "R"
    geo = side_from_centers([h.center for h in headers if h.center is not None])
    return "conflict" if geo is not None and geo != tag else None


@lru_cache(maxsize=4)
def load_template(path: str = str(TEMPLATE)) -> tuple[np.ndarray, int, str, dict]:
    z = np.load(path)
    axis = {int(k): int(v) for k, v in json.loads(str(z["mirror_axis"])).items()}
    return z["template"].astype(np.float32), int(z["block"]), str(z["spec"]), axis


def _reduce(v: np.ndarray, block: int) -> np.ndarray:
    k, p, _ = v.shape
    q = p // block
    return v[:, :q * block, :q * block].reshape(k, q, block, q, block).mean((2, 4))


def scores(vol: np.ndarray, mask: np.ndarray, spec, template: np.ndarray | None = None,
           block: int | None = None) -> np.ndarray:
    """Per slot: correlation with the template minus correlation of the mirror with the template
    (NaN if absent).

    `vol` (S, K, P, P) as returned by `build_study`.
    """
    if template is None:
        template, block, tag, _ = load_template()
        if not spec.tag.startswith(tag):
            raise ValueError(f"anatomical-side template built on {tag}, spec {spec.tag}")
    n_seq = len(spec.active_slots)
    d = np.full(len(vol), np.nan, dtype=np.float32)
    for k in range(len(vol)):
        if not mask[k]:
            continue
        z = _reduce(vol[k].astype(np.float32), block)
        z = np.round(z)                                    # the kernel reduces it in uint8
        z = (z - z.mean()) / (z.std() + 1e-6)
        axis = _PLANE_AXIS[spec.active_slots[k % n_seq][0]]
        t = template[k]
        d[k] = ((z * t).sum() - (np.flip(z, axis=axis) * t).sum()) / z.size
    return d


def mirror_study(vol: np.ndarray, mask: np.ndarray, spec) -> None:
    """Flips every present slot in place, like `normalise_laterality` for a right knee."""
    n_seq = len(spec.active_slots)
    for k in range(len(vol)):
        if mask[k]:
            axis = _PLANE_AXIS[spec.active_slots[k % n_seq][0]]
            vol[k] = np.flip(vol[k], axis=axis).copy()


def resolve(vol: np.ndarray, mask: np.ndarray, spec, headers, side: str | None,
            side_src: str) -> tuple[str | None, str, dict]:
    """The rule: `(side, side_src, info)`; flips `vol` in place if needed."""
    if getattr(spec, "sag_band_lm", None) is not None:
        raise NotImplementedError("side_anat with sag_band_lm: the sagittal band depends on the side")
    why = ambiguity(headers, side_src)
    if why is None:
        return side, side_src, {}
    d = scores(vol, mask, spec)
    ok = ~np.isnan(d)
    vote = int(np.sign(d[ok]).sum())
    info = {"why": why, "vote": vote, "mean": float(d[ok].mean()) if ok.any() else None,
            "flipped": False}
    if vote <= -MIN_VOTE:
        mirror_study(vol, mask, spec)
        side = "L" if side == "R" else "R"
        info["flipped"] = True
        return side, f"anat-{why}", info
    return side, side_src, info
