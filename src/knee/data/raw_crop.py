"""The raw crop of a study instead of the uint8 volume: resize, percentiles and uint8 on the GPU.

At 512 px inference is bound by the CPU of the workers: upscaling the crops to 512 x 512 and
normalising them costs 67% more CPU than at 288. Here the worker stops earlier: for every series it
ships the slices of the band cropped to the box that contains every scale (with 110 + 150 mm, the
150 mm one), as exact uint16, and the device does the rest (`payload_to_uint8_torch`): crop of
every scale, bilinear resize, exact 1/99 percentiles (`torch_uint8.to_uint8_torch_hist`) and uint8,
side flip.

**It is not bit-identical.** cv2 upscaling (INTER_LINEAR in float32, via IPP) and torch's
(`interpolate` bilinear, `align_corners=False`) do the same arithmetic with the same weights but
round in a different order: differences of ~1 ulp, which after the intensity window move a few
pixels in ~10^4 by one level. Percentiles and normalisation are exact (the same as on CPU on the
resized pixels). The size of the difference is measured by `compare_uint8`.

Downscaling (box taller than `img`: cv2 uses INTER_AREA) is done on the GPU with cv2's tables
(`_area_taps`, `computeResizeAreaTab` row by row) in two separable passes, with products and sums
in the same order: bit-identical to `cv2.resize` (at 512 px the 150 mm box exceeds 512 px in 34% of
the series).

DICOM Rescale: the worker ships the stored (integer) values and, for series with non-trivial
RescaleSlope/Intercept (Philips "normalized", 17% of the corpus), slope and intercept per slice;
the GPU redoes `a * slope + intercept` in float32, product then sum, as NumPy: bit for bit.

The series the GPU would not redo with the same formula stay on the CPU, as before (final uint8 in
the payload): a crop taller than `img` but narrower (cv2 switches to a mixed INTER_AREA), slices of
different shapes in the same series, stored values outside uint16, MONOCHROME1.

The payload of a study (`build_study_raw`) is a dictionary of tensors:
- `raw`: 1-D int16, the uint16 pixels of all GPU series back to back (reinterpreted: torch does
  not compute in uint16); a series = (K, Hr, Wr) starting at `raw_off`;
- `u8`: 1-D uint8, the final uint8 values of the CPU series (already flipped), K x img x img per
  scale;
- `resc`: float64 (m, 2), slope and intercept per slice of the series with non-trivial Rescale, K
  rows starting at `M_RESC` (-1 = no Rescale);
- `meta`: int64 (n, `N_META`), one row per present series (columns `M_*`).
"""
from __future__ import annotations

import os

import numpy as np

from .cache import CacheSpec, _read_band, _resize, _to_uint8
from .geometry import crop_box, foreground_center, normalise_laterality

# Columns of `meta`.
M_SLOT, M_MODE, M_K, M_HR, M_WR, M_RAW_OFF, M_FLIP, M_U8_OFF = range(8)
M_BOX = 8                       # 4 integers (y0, y1, x0, x1) per scale, relative to the box
MAX_SCALES = 2
M_RESC = M_BOX + 4 * MAX_SCALES  # first row of `resc` for the series, -1 = no Rescale
N_META = M_RESC + 1
MODE_GPU, MODE_CPU = 1, 2
FLIP_NONE, FLIP_X, FLIP_SLICES = 0, 1, 2


def _flip_code(plane: str, side: str | None) -> int:
    """`normalise_laterality` as a code: nothing, columns flipped, slices in reverse order."""
    if side != "R":
        return FLIP_NONE
    return FLIP_X if plane in ("Coronal", "Axial") else FLIP_SLICES


def _read_slices_env() -> tuple[int, int] | None:
    v = os.environ.get("KNEE_READ_SLICES", "").strip()
    if not v:
        return None
    a, z = (int(x) for x in v.split(":"))
    return a, z


# The slices of every stack, in final order (after the side), that the readers read and on which
# percentiles and uint8 are computed: None = all, the usual path. `KNEE_READ_SLICES=a:z`: a kernel
# that uses only the windows of the range [a, z) (`window_budget` `edgeA-Bx`) does not read the other
# slices and leaves them at zero. The final submission uses `0:23`: no window touches the 24th slice.
READ_SLICES = [_read_slices_env()]


def read_range(k_sl: int, flip: int) -> tuple[int, int] | None:
    """The positions [a, z) of the payload to read and normalise in a series of `k_sl` slices with side
    code `flip`, or None (all). `FLIP_SLICES` reverses the slices after normalisation: final slice i is
    payload position k_sl - 1 - i.
    """
    r = READ_SLICES[0]
    if r is None:
        return None
    a, z = r
    if not 0 <= a < z <= k_sl:
        raise ValueError(f"KNEE_READ_SLICES {a}:{z} outside 0..{k_sl}")
    return (k_sl - z, k_sl - a) if flip == FLIP_SLICES else (a, z)


def _uint8_ranges(stacks, ranges: list):
    """`to_uint8_torch_hist_batch` stack by stack on the positions of its range only (`read_range`, None =
    all), zeros elsewhere: the stacks with the same range in a single launch.
    """
    import torch
    from .torch_uint8 import to_uint8_torch_hist_batch
    out = torch.zeros(stacks.shape, dtype=torch.uint8, device=stacks.device)
    groups: dict = {}
    for i, r in enumerate(ranges):
        groups.setdefault(r, []).append(i)
    for r, idx in groups.items():
        sel = torch.tensor(idx, device=stacks.device)
        if r is None:
            out[sel] = to_uint8_torch_hist_batch(stacks[sel].contiguous())
        else:
            out[sel, r[0]:r[1]] = to_uint8_torch_hist_batch(stacks[sel, r[0]:r[1]].contiguous())
    return out


def _scales_uint8_cpu(raw: list[np.ndarray], boxes: list[tuple[int, int, int, int]], img: int,
                      plane: str, side: str | None, k_range: tuple[int, int] | None = None) -> list[np.ndarray]:
    """The uint8 values of every scale with the CPU path (`build_series` "old", which "lean" reproduces
    bit for bit): crop, `_resize`, `_to_uint8` on the stack, side. With `k_range` = (a, z)
    (`read_range`) percentiles and uint8 on the positions [a, z) of the payload only, zeros elsewhere.
    """
    out = []
    for (y0, y1, x0, x1) in boxes:
        done: dict[int, np.ndarray] = {}
        rs = []
        for k, a in enumerate(raw):
            if k_range is not None and not k_range[0] <= k < k_range[1]:
                rs.append(None)
                continue
            r = done.get(id(a))
            if r is None:
                r = done[id(a)] = _resize(a[y0:y1, x0:x1].astype(np.float32), img)
            rs.append(r)
        if k_range is None:
            u = _to_uint8(np.stack(rs))
        else:
            a0, z0 = k_range
            kept = np.stack(rs[a0:z0])
            u = np.zeros((len(rs), *kept.shape[1:]), np.uint8)
            u[a0:z0] = _to_uint8(kept)
        out.append(np.ascontiguousarray(normalise_laterality(u, plane, side)))
    return out


def _parts(x) -> tuple:
    """A slice as (stored values, slope, intercept, MONOCHROME1): the parts of
    `dicom.read_slice_parts`, or an already final array (trivial Rescale).
    """
    return x if isinstance(x, tuple) else (x, 1.0, 0.0, False)


def _fits_uint16(a: np.ndarray) -> bool:
    """True if `a.astype(uint16)` preserves the values: for integers a range check (instead of
    `np.array_equal` after the cast, which was ~7% of the worker CPU).
    """
    if a.dtype == np.uint16 or a.dtype == np.uint8:
        return True
    if np.issubdtype(a.dtype, np.integer):
        return a.size == 0 or (int(a.min()) >= 0 and int(a.max()) <= 0xFFFF)
    return bool(np.array_equal(a.astype(np.uint16), a))


def _series_payload(raw: list, ps, plane: str, side: str | None, spec: CacheSpec,
                    crops: tuple[float, ...]) -> tuple[str, dict]:
    """A decoded series (no holes; slices as `read_slice_parts` parts or final arrays, repeated ones are
    the same object) -> ("gpu", {region, boxes, resc}) or ("cpu", {u8, why}).
    """
    from .dicom import _apply_rescale
    parts = [_parts(x) for x in raw]
    finals_by_id: dict[int, np.ndarray] = {}

    def final(i: int) -> np.ndarray:            # the pixels of `read_slice`, repeated slices shared
        f = finals_by_id.get(id(raw[i]))
        if f is None:
            f = finals_by_id[id(raw[i])] = _apply_rescale(*parts[i])
        return f

    # For the series going to the GPU only the stored values are needed (and the middle slice for the
    # centre): the final pixels of all slices are computed only if the series stays on the CPU.
    stored = [pt[0] for pt in parts]
    center = foreground_center(final(len(raw) // 2)) if spec.center == "mass" else None
    shapes = {a.shape for a in stored}
    why = None
    if any(pt[3] for pt in parts):
        why = "mono1"
    if len(shapes) != 1:
        why = "shapes"
    rows, cols = stored[0].shape
    full = [(0, rows, 0, cols) for _ in crops]
    boxes = ([crop_box(rows, cols, ps, c, center) for c in crops] if ps is not None else full)
    if why is None and any(y1 - y0 > spec.img and x1 - x0 < spec.img for (y0, y1, x0, x1) in boxes):
        why = "area_mixed"                      # INTER_AREA upscaling in x: a different cv2 formula
    if why is None:
        ry0, ry1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
        rx0, rx1 = min(b[2] for b in boxes), max(b[3] for b in boxes)
        region = np.empty((len(raw), ry1 - ry0, rx1 - rx0), np.uint16)
        done: dict[int, int] = {}
        for i, (x, a) in enumerate(zip(raw, stored)):
            j = done.get(id(x))
            if j is not None:
                region[i] = region[j]
                continue
            done[id(x)] = i
            sub = a[ry0:ry1, rx0:rx1]
            if not _fits_uint16(sub):           # the cast to uint16 must be exact
                why = "values"
                break
            region[i] = sub
        if why is None:
            rel = [(y0 - ry0, y1 - ry0, x0 - rx0, x1 - rx0) for (y0, y1, x0, x1) in boxes]
            resc = np.array([[pt[1], pt[2]] for pt in parts], np.float64)
            plain = bool(((resc[:, 0] == 1.0) & (resc[:, 1] == 0.0)).all())
            return "gpu", {"region": region, "boxes": rel, "resc": None if plain else resc}
    finals = [final(i) for i in range(len(raw))]
    if why == "shapes":
        # Every slice with its own box, as in `build_series`: no common box.
        u8 = []
        for c in crops:
            per = []
            for a in finals:
                b = crop_box(a.shape[0], a.shape[1], ps, c, center) if ps is not None \
                    else (0, a.shape[0], 0, a.shape[1])
                per.append(_resize(a[b[0]:b[1], b[2]:b[3]].astype(np.float32), spec.img))
            u8.append(np.ascontiguousarray(normalise_laterality(_to_uint8(np.stack(per)),
                                                                plane, side)))
        return "cpu", {"u8": u8, "why": why}
    return "cpu", {"u8": _scales_uint8_cpu(finals, boxes, spec.img, plane, side), "why": why}


def build_series_raw(series_dir: str, plane: str, side: str | None, spec: CacheSpec,
                     errors=None, hdr=None, order=None) -> tuple[tuple[str, dict] | None, dict]:
    """`build_series` up to the crop: the series as a payload ("gpu" or "cpu", see `_series_payload`),
    or None if no slice decodes.
    """
    from . import dicom as dicom_mod
    from .dicom import order_series_files, probe_series, read_slice_parts

    own_order = order is None
    if own_order:
        order = order_series_files(series_dir, keep_fds=dicom_mod._FD_KEEP[0])
    try:
        if not order.files:
            return None, {"order": "empty", "n_read": 0}
        raw, sag = _read_band(series_dir, plane, side, spec, errors, order, read_slice_parts)
        n_read = sum(a is not None for a in raw)
        if n_read == 0:
            return None, {"order": order.method, "n_read": 0}
        fill = next(a for a in raw if a is not None)
        raw = [fill if a is None else a for a in raw]
        ps = (hdr if hdr is not None else probe_series(series_dir)).pixel_spacing
        return (_series_payload(raw, ps, plane, side, spec, spec.crops),
                {"order": order.method, "n_read": n_read, "sag_band": sag})
    finally:
        if own_order:
            dicom_mod.close_kept(series_dir, order.files)


def build_study_raw_np(study_dir: str, slots: dict[int, str], spec: CacheSpec,
                       errors=None) -> tuple[dict, np.ndarray, dict]:
    """`cache.build_study` with the series as raw crops -> (payload, mask (S,), meta).

    Same side, same series and same slices as `build_study`; `side_anat` is not supported (it needs
    the uint8 volume in the worker).
    """
    from .cache import study_side
    from .dicom import probe_series
    from .pools import thread_pool

    if spec.side_anat:
        raise ValueError("build_study_raw: side_anat needs the uint8 volume in the worker")
    if spec.n_scales > MAX_SCALES:
        raise ValueError(f"build_study_raw: at most {MAX_SCALES} scales")
    mask = np.zeros(spec.shape[0], dtype=bool)
    items = list(slots.items())
    with thread_pool("study", max(len(items), 1)) as ex:
        hdrs = dict(zip([j for j, _ in items],
                        ex.map(lambda ju: probe_series(os.path.join(study_dir, ju[1]), ju[1]),
                               items)))
        side, side_src = study_side(list(hdrs.values()), tie=spec.side_tie)
        built = list(ex.map(
            lambda ju: build_series_raw(os.path.join(study_dir, ju[1]),
                                        spec.active_slots[ju[0]][0], side, spec, errors,
                                        hdr=hdrs[ju[0]]),
            items))
    n_seq = len(spec.active_slots)
    rows, raws, u8s, rescs = [], [], [], []
    raw_off = u8_off = resc_off = 0
    meta = {"side": side or "", "side_source": side_src, "n_read": 0, "cpu_series": {}}
    for (j, _uid), (res, info) in zip(items, built):
        meta["n_read"] += info["n_read"]
        if res is None:
            continue
        kind, d = res
        plane = spec.active_slots[j][0]
        row = np.zeros(N_META, np.int64)
        row[M_SLOT], row[M_K], row[M_FLIP] = j, spec.slices, _flip_code(plane, side)
        if kind == "gpu":
            reg = d["region"]
            row[M_MODE], row[M_HR], row[M_WR], row[M_RAW_OFF] = MODE_GPU, reg.shape[1], reg.shape[2], raw_off
            row[M_RESC] = -1
            if d.get("resc") is not None:
                row[M_RESC] = resc_off
                rescs.append(d["resc"])
                resc_off += d["resc"].shape[0]
            for k, b in enumerate(d["boxes"]):
                row[M_BOX + 4 * k: M_BOX + 4 * k + 4] = b
            raws.append(reg.reshape(-1))
            raw_off += reg.size
        else:
            row[M_MODE], row[M_U8_OFF] = MODE_CPU, u8_off
            meta["cpu_series"][d["why"]] = meta["cpu_series"].get(d["why"], 0) + 1
            for v in d["u8"]:
                u8s.append(v.reshape(-1))
                u8_off += v.size
        rows.append(row)
        for k in range(spec.n_scales):
            mask[j + k * n_seq] = True
    payload = {
        "raw": np.concatenate(raws).view(np.int16) if raws else np.zeros(0, np.int16),
        "u8": np.concatenate(u8s) if u8s else np.zeros(0, np.uint8),
        "resc": np.concatenate(rescs) if rescs else np.zeros((0, 2), np.float64),
        "meta": np.stack(rows) if rows else np.zeros((0, N_META), np.int64),
    }
    return payload, mask, meta


def build_study_raw(study_dir: str, slots: dict[int, str], spec: CacheSpec,
                    errors=None) -> tuple[dict, np.ndarray, dict]:
    """`build_study_raw_np` with the payload as (CPU) tensors, as `DicomStudies` wants it."""
    payload, mask, meta = build_study_raw_np(study_dir, slots, spec, errors)
    return payload_to_torch(payload), mask, meta


def payload_to_torch(payload: dict) -> dict:
    import torch
    return {k: torch.from_numpy(v) for k, v in payload.items()}


def empty_payload_np() -> dict:
    return {"raw": np.zeros(0, np.int16), "u8": np.zeros(0, np.uint8),
            "resc": np.zeros((0, 2), np.float64), "meta": np.zeros((0, N_META), np.int64)}


def empty_payload() -> dict:
    return payload_to_torch(empty_payload_np())


_TAPS: dict = {}


def _area_taps(ssize: int, dsize: int) -> tuple[np.ndarray, np.ndarray]:
    """cv2's INTER_AREA tables (`computeResizeAreaTab`) for a downscaling axis: (dsize, T) source indices
    and float32 weights, zeros where a row has fewer than T terms.
    """
    key = (ssize, dsize)
    if key not in _TAPS:
        import math
        scale = 1.0 / (dsize / ssize)           # as cv2: inv_scale = dsize / ssize, scale = 1 / inv
        rows = []
        for dx in range(dsize):
            fsx1 = dx * scale
            fsx2 = fsx1 + scale
            cw = min(scale, ssize - fsx1)
            sx1, sx2 = math.ceil(fsx1), math.floor(fsx2)
            sx2 = min(sx2, ssize - 1)
            sx1 = min(sx1, sx2)
            t = []
            if sx1 - fsx1 > 1e-3:
                t.append((sx1 - 1, np.float32((sx1 - fsx1) / cw)))
            for sx in range(sx1, sx2):
                t.append((sx, np.float32(1.0 / cw)))
            if fsx2 - sx2 > 1e-3:
                t.append((sx2, np.float32(min(min(fsx2 - sx2, 1.0), cw) / cw)))
            rows.append(t)
        n_t = max(len(t) for t in rows)
        idx = np.zeros((dsize, n_t), np.int64)
        w = np.zeros((dsize, n_t), np.float32)
        for i, t in enumerate(rows):
            for j, (src, a) in enumerate(t):
                idx[i, j], w[i, j] = src, a
        _TAPS[key] = (idx, w)
    return _TAPS[key]


_TAPS_DEV: dict = {}


def _taps_on(ssize: int, dsize: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """`_area_taps` on the device, copied once per shape (a copy from RAM for every stack would go
    through non-pinned memory).
    """
    import torch
    key = (ssize, dsize, str(device))
    if key not in _TAPS_DEV:
        ix, wx = _area_taps(ssize, dsize)
        _TAPS_DEV[key] = (torch.from_numpy(ix).to(device), torch.from_numpy(wx).to(device))
    return _TAPS_DEV[key]


def area_resize_torch(sub: torch.Tensor, img: int) -> torch.Tensor:
    """(K, h, w) float32 -> (K, img, img) like `cv2.resize(..., INTER_AREA)` for h, w >= img: columns
    first, then rows, every term multiplied and added separately (no FMA), in table order. An axis
    already at `img` passes through unchanged.
    """
    import torch
    k, h, w = sub.shape
    out = sub
    if w != img:
        idx, wt = _taps_on(w, img, sub.device)
        acc = torch.zeros((k, h, img), dtype=torch.float32, device=sub.device)
        for j in range(idx.shape[1]):
            acc.add_(out.index_select(2, idx[:, j]).mul_(wt[:, j]))
        out = acc
    if h != img:
        idx, wt = _taps_on(h, img, sub.device)
        acc = torch.zeros((k, img, out.shape[2]), dtype=torch.float32, device=sub.device)
        for j in range(idx.shape[1]):
            acc.add_(out.index_select(1, idx[:, j]).mul_(wt[:, j].unsqueeze(1)))
        out = acc
    return out


def _resize_torch(sub: torch.Tensor, img: int) -> torch.Tensor:
    """The resize of `cache._resize` on the stack (K, h, w): INTER_AREA if the box is taller than `img`
    (and at least `img` wide), otherwise bilinear (INTER_LINEAR: half-step centres, no antialiasing,
    also where it shrinks in x); unchanged at `img` x `img`.
    """
    import torch
    import torch.nn.functional as F
    if sub.shape[1] == img and sub.shape[2] == img:
        return sub.contiguous()
    if sub.shape[1] > img:
        return area_resize_torch(sub, img)
    return F.interpolate(sub.unsqueeze(1), size=(img, img), mode="bilinear",
                         align_corners=False, antialias=False).squeeze(1)


def _region(raw: torch.Tensor, row) -> torch.Tensor:
    k, hr, wr, off = int(row[M_K]), int(row[M_HR]), int(row[M_WR]), int(row[M_RAW_OFF])
    return raw[off: off + k * hr * wr].view(k, hr, wr)


def payload_to_uint8_cpu(payload: dict, spec: CacheSpec) -> np.ndarray:
    """The volume (S, K, img, img) uint8 from the payload with the CPU path: for GPU series the same
    uint8 values bit for bit as `build_study` (the uint16 pixels are exact). Reference for the
    self-check and fallback if the GPU is wrong.
    """
    import torch
    img, n_seq = spec.img, len(spec.active_slots)
    vol = np.zeros(spec.shape, np.uint8)
    raw = payload["raw"].numpy().view(np.uint16)
    u8 = payload["u8"].numpy()
    plane_of = [p for p, _, _ in spec.active_slots]
    side_of = {FLIP_NONE: None, FLIP_X: "R", FLIP_SLICES: "R"}
    for row in payload["meta"].numpy():
        j, k_sl = int(row[M_SLOT]), int(row[M_K])
        if row[M_MODE] == MODE_GPU:
            hr, wr, off = int(row[M_HR]), int(row[M_WR]), int(row[M_RAW_OFF])
            region = raw[off: off + k_sl * hr * wr].reshape(k_sl, hr, wr).astype(np.float32)
            raw_list = [region[i] for i in range(k_sl)]
            if row[M_RESC] >= 0:                # as in `read_slice`: float32 * slope + intercept
                rs = payload["resc"].numpy()[int(row[M_RESC]): int(row[M_RESC]) + k_sl]
                raw_list = [a * float(sl) + float(ic) if (sl != 1.0 or ic != 0.0) else a
                            for a, (sl, ic) in zip(raw_list, rs)]
            boxes = [tuple(int(v) for v in row[M_BOX + 4 * k: M_BOX + 4 * k + 4])
                     for k in range(spec.n_scales)]
            # Repeated slices (same pixels) are resized separately: same result.
            vs = _scales_uint8_cpu(raw_list, boxes, img, plane_of[j], side_of[int(row[M_FLIP])],
                                   k_range=read_range(k_sl, int(row[M_FLIP])))
        else:
            off, n = int(row[M_U8_OFF]), k_sl * img * img
            vs = [u8[off + k * n: off + (k + 1) * n].reshape(k_sl, img, img)
                  for k in range(spec.n_scales)]
        for k, v in enumerate(vs):
            vol[j + k * n_seq] = v
    return vol


def _pinned(payload: dict, device) -> dict:
    """`raw`, `u8`, `resc` in pinned memory (torch's pinned allocator, which does not reuse a block until
    its copy is done) if they go to a GPU from normal RAM. From pageable memory (mapped `/dev/shm`,
    `knee.data.early`) CUDA synchronises the stream before the copy: in the preparation thread that
    meant waiting for the model kernels queued ahead (18% of the thread).
    """
    import torch
    if torch.device(device).type != "cuda":
        return payload
    out = dict(payload)
    for k in ("raw", "u8", "resc"):
        v = payload[k]
        if v.device.type == "cpu" and v.numel() and not v.is_pinned():
            out[k] = v.pin_memory()
    return out


def payloads_to_uint8_torch(payloads: list, spec: CacheSpec, device) -> torch.Tensor:
    """The volumes (B, S, K, img, img) uint8 on the device: for every GPU series the box in float32, the
    crop of every scale and the resize of `cache._resize` (`_resize_torch`), then exact percentiles and
    uint8 with several stacks per launch (`to_uint8_torch_hist_batch`, one study at a time: eight 25 MB
    float32 stacks), side. Payloads can be in (pinned) RAM or on the device.
    """
    import torch
    from .torch_uint8 import to_uint8_torch_hist_batch

    img, n_seq = spec.img, len(spec.active_slots)
    vol = torch.zeros((len(payloads), *spec.shape), dtype=torch.uint8, device=device)
    for b, payload in enumerate(payloads):
        meta = payload["meta"]
        if meta.numel() == 0:
            continue
        meta = meta.numpy() if meta.device.type == "cpu" else meta.cpu().numpy()
        src = _pinned(payload, device)
        raw = src["raw"].to(device, non_blocking=True)
        u8 = src["u8"].to(device, non_blocking=True)
        resc = (src["resc"].to(device, non_blocking=True).float()
                if src["resc"].numel() else None)
        gpu_count = sum(spec.n_scales for row in meta if row[M_MODE] == MODE_GPU)
        k_sl_ref = int(meta[0][M_K]) if len(meta) else spec.slices
        stacks_buf = torch.empty((gpu_count, k_sl_ref, img, img), dtype=torch.float32, device=device) if gpu_count else None
        dests, ranges = [], []
        s_idx = 0
        for row in meta:
            j, k_sl = int(row[M_SLOT]), int(row[M_K])
            if row[M_MODE] == MODE_GPU:
                region = _region(raw, row).view(torch.uint16).float()
                if row[M_RESC] >= 0:            # Rescale as in NumPy: slope and intercept in float32
                    rs = resc[int(row[M_RESC]): int(row[M_RESC]) + k_sl]
                    region.mul_(rs[:, 0].view(-1, 1, 1)).add_(rs[:, 1].view(-1, 1, 1))
                rr = read_range(k_sl, int(row[M_FLIP]))
                for k in range(spec.n_scales):
                    y0, y1, x0, x1 = (int(v) for v in row[M_BOX + 4 * k: M_BOX + 4 * k + 4])
                    if rr is None:
                        stacks_buf[s_idx].copy_(_resize_torch(region[:, y0:y1, x0:x1], img))
                    else:                       # the slices outside `KNEE_READ_SLICES` were not read
                        stacks_buf[s_idx, rr[0]:rr[1]].copy_(
                            _resize_torch(region[rr[0]:rr[1], y0:y1, x0:x1], img))
                    dests.append((j + k * n_seq, int(row[M_FLIP])))
                    ranges.append(rr)
                    s_idx += 1
            else:
                off, n = int(row[M_U8_OFF]), k_sl * img * img
                for k in range(spec.n_scales):
                    vol[b, j + k * n_seq] = u8[off + k * n: off + (k + 1) * n].view(k_sl, img, img)
        if stacks_buf is not None and s_idx > 0:
            u = (to_uint8_torch_hist_batch(stacks_buf) if all(r is None for r in ranges)
                 else _uint8_ranges(stacks_buf, ranges))
            for (slot, flip), v in zip(dests, u):
                if flip == FLIP_X:
                    v = v.flip(2)
                elif flip == FLIP_SLICES:
                    v = v.flip(0)
                vol[b, slot] = v
    return vol


def payload_to_uint8_torch(payload: dict, spec: CacheSpec, device) -> torch.Tensor:
    """One volume (S, K, img, img): `payloads_to_uint8_torch` on a single payload."""
    return payloads_to_uint8_torch([payload], spec, device)[0]


def compare_uint8(ref: np.ndarray, got: np.ndarray) -> dict:
    """How much two uint8 volumes differ: differing pixels (fraction) and maximum difference."""
    d = np.abs(ref.astype(np.int16) - got.astype(np.int16))
    return {"pixels": int(d.size), "diff": int((d > 0).sum()), "max_abs": int(d.max()) if d.size else 0}


def collate_raw(items: list) -> tuple:
    """The DataLoader `collate` for `DicomStudies(pixels="raw")`: the payloads stay a list (different
    shapes per study), the rest as the default collate.
    """
    import torch
    payloads, masks, idx, dts, errs = zip(*items)
    return (list(payloads), torch.stack(masks), torch.tensor(idx, dtype=torch.int64),
            torch.tensor(dts, dtype=torch.float64), torch.tensor(errs, dtype=torch.int64))
