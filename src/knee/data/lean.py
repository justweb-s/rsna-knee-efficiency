"""The raw payload of a study with I/O in C: the same decisions as `raw_crop.build_study_raw_np`,
bit for bit.

Measured on a T4: the Python workers cost ~0.47 vCPU-s per study (8.1 studies/s), half in
Python/NumPy and half in the kernel (16 thousand page faults per study, /dev/shm, the page cache
filling and emptying), while the same read pattern done in C costs ~0.10 vCPU-s and the NFS mount
sustains ~730 MB/s, ~15 studies/s. Here I/O and pixel copies go to C (`cfast.lio_*`, `leanio.c`)
and Python keeps the decisions, with the usual functions (same bytes, same functions, same
results):

- `lio_listdir` + `sorted`: the files of every series (`os.listdir` + `sorted`);
- `lio_scan`: the first 16 KB of all files of the study on a thread group, descriptors kept open;
  from the prefixes the ordering keys (`dicom._order_tags` + `_key_from_fast` + `order_from_keys`,
  as in `_scan_keys`), the series header (`dicom._probe_values_fast` on the middle file, with the
  same comparisons with pydicom), the side (`cache.study_side`), the band
  (`geometry.band_indices`), the crop boxes (`geometry.crop_box`), the Rescale (`dicom._ds_single`)
  and the conditions of `raw_crop._series_payload`;
- `lio_read`: only the rows of the crop box of every file of the band (~8% fewer bytes than the
  whole file), from the kept descriptors, as uint16 straight into the destination (the workers'
  shared buffer, `early.LeanPool`): no `bytes`, no intermediate arrays, no concatenations.

A series outside the simple case (MONOCHROME1, mixed shapes, mixed INTER_AREA, headers the fast
path rejects, a prefix that does not reach the group 0028 tags) goes through
`raw_crop.build_series_raw` as is; an I/O error or a value outside uint16 makes it return `None`
and the caller redoes the study with `build_study_raw_np`. `early.LeanPool` compares the two paths
bit for bit on the first studies of every worker.
"""
from __future__ import annotations

import os

import numpy as np

from . import cfast
from . import dicom as D
from .cache import CacheSpec, study_side
from .geometry import band_indices, crop_box, image_center_xyz
from .raw_crop import (M_BOX, M_FLIP, M_HR, M_K, M_MODE, M_RAW_OFF, M_RESC, M_SLOT, M_U8_OFF, M_WR, MAX_SCALES,
                       MODE_CPU, MODE_GPU, N_META, _flip_code, read_range)

THREADS = [int(os.environ.get("KNEE_LEAN_THREADS", "16"))]
DIRECT = [os.environ.get("KNEE_LEAN_DIRECT", "1") == "1"]
_PIX_NAMES = list(D._PIX_TAGS.values())
_ALIGN = 64


def usable(spec: CacheSpec) -> bool:
    """Whether the lean path covers this spec and this process (otherwise `build_study_raw_np`)."""
    from . import cache
    return (cfast.lio_ok() and not spec.side_anat and spec.n_scales <= MAX_SCALES and spec.center == "image"
            and spec.sag_band_lm is None and cache._DEDUP[0] and D._HEADER_READ[0] == "exact"
            and D._FAST_PIX_OK[0])


def _probe_values_buf(buf: bytes, path: str) -> tuple:
    """`dicom._probe_values(path)` from the prefix bytes already read (the same as `_read_prefix`)."""
    if D._PROBE_FAST[0]:
        try:
            got = D._probe_values_fast(buf)
        except Exception:
            got = None
        if got is not None:
            if D._PROBE_CHECKS[0] > 0:
                D._PROBE_CHECKS[0] -= 1
                ref = D._probe_values_pydicom(path)
                if ref != got:
                    D._PROBE_FAST[0] = False
                    return ref
            return got
    return D._probe_values_pydicom(path)


def _probe(series_dir: str, uid: str, files: list[str], heads, head_len, a: int) -> D.SeriesHeader:
    """`dicom.probe_series` from the scanned prefixes."""
    h = D.SeriesHeader(uid=uid, path=series_dir, n_files=len(files))
    if not files:
        return h
    m = a + len(files) // 2
    try:
        vals = _probe_values_buf(heads[m, :int(head_len[m])].tobytes(),
                                 os.path.join(series_dir, files[len(files) // 2]))
    except Exception:
        return h
    h.rows, h.cols, h.pixel_spacing, h.laterality, ipp, iop = vals
    if ipp is not None and iop is not None and h.rows and h.cols and h.pixel_spacing:
        h.center = image_center_xyz(np.asarray(ipp), np.asarray(iop), h.pixel_spacing, h.rows, h.cols)
    return h


def _order(series_dir: str, files: list[str], heads, head_len, offs, rc, a: int) -> D.OrderResult:
    """`dicom.order_series_files` from the scanned prefixes (keys as in `_scan_keys`)."""
    if len(files) <= 1:
        return D.OrderResult(files, "name", 0)
    keys = []
    for k, f in enumerate(files):
        i = a + k
        buf = heads[i, :int(head_len[i])].tobytes()
        try:
            fast = D._order_tags(buf, offs[i].tolist() if rc[i] == 0 else None)
        except Exception:
            fast = None
        keys.append(D._key_from_fast(f, os.path.join(series_dir, f), fast))
    return D.order_from_keys(files, keys)


def _tags(h: np.ndarray, buf_row: np.ndarray) -> dict:
    """The group 0028 tags from the `walk_header` walk (like `dicom._pixel_tags`)."""
    tags = {}
    for t, name in enumerate(_PIX_NAMES):
        kind, a, b = int(h[2 + 3 * t]), int(h[3 + 3 * t]), int(h[4 + 3 * t])
        if kind == 1:
            tags[name] = a
        elif kind == 2:
            tags[name] = buf_row[a:a + b].tobytes()
    return tags


def _pixel_meta(tags: dict, pix_len: int | None):
    """The decisions of `dicom._native_pixel_parts(..., as_int=True)` without the pixels: (rows, cols,
    bits, rep, shift, slope, intercept, mono1), or None where the fast path would stop (or raise).
    """
    try:
        rows, cols = tags.get("Rows"), tags.get("Columns")
        bits, stored = tags.get("BitsAllocated"), tags.get("BitsStored")
        rep = tags.get("PixelRepresentation")
        if not all(isinstance(v, int) for v in (rows, cols, bits, stored, rep)):
            return None
        if not rows or not cols or bits not in (8, 16) or rep not in (0, 1):
            return None
        if not 0 < stored <= bits or tags.get("SamplesPerPixel", 1) != 1:
            return None
        frames = tags.get("NumberOfFrames")
        if frames is not None:
            s = frames.decode("ascii").strip("\x00 ")
            if s and int(s) != 1:
                return None
        if pix_len is not None and pix_len < rows * cols * bits // 8:
            return None
        shift = bits - stored if (D._PYDICOM_SHIFTS_UNUSED_BITS and stored < bits) else 0
        slope_v = D._ds_single(tags["RescaleSlope"]) if "RescaleSlope" in tags else 1
        inter_v = D._ds_single(tags["RescaleIntercept"]) if "RescaleIntercept" in tags else 0
        slope, intercept = float(slope_v or 1), float(inter_v or 0)
        photometric = tags.get("PhotometricInterpretation", b"")
        mono1 = photometric.decode("ascii").rstrip("\x00 ") == "MONOCHROME1"
    except Exception:
        return None
    return rows, cols, bits, rep, shift, slope, intercept, mono1


def _plan_series(spec: CacheSpec, order: D.OrderResult, index_of: dict, hdr_s: D.SeriesHeader,
                 heads, hdr, hdr_rc) -> dict | None:
    """The plan of a series for the GPU (the choices of `raw_crop._series_payload` on the "gpu" branch),
    or None if the series must go through `build_series_raw` (anything outside the simple case).
    """
    idx = band_indices(len(order.files), spec.slices, spec.band)
    pos_file = [index_of[order.files[int(i)]] for i in idx]          # study files by position
    uniq = list(dict.fromkeys(pos_file))
    metas = {}
    for i in uniq:
        rc = int(hdr_rc[i])
        if rc == 0:
            pix_len = int(hdr[i, 1])
        elif rc == 1 and hdr[i, 33] == 1:
            pix_len = None                                           # PixelData beyond the prefix
        else:
            return None
        m = _pixel_meta(_tags(hdr[i], heads[i]), pix_len)
        if m is None:
            return None
        metas[i] = m
    if any(m[7] for m in metas.values()):
        return None                                                  # "mono1": CPU path
    if len({(m[0], m[1]) for m in metas.values()}) != 1:
        return None                                                  # "shapes"
    rows, cols = metas[pos_file[0]][0], metas[pos_file[0]][1]
    ps = hdr_s.pixel_spacing
    full = [(0, rows, 0, cols) for _ in spec.crops]
    boxes = [crop_box(rows, cols, ps, c, None) for c in spec.crops] if ps is not None else full
    why = ("area_mixed"
           if any(y1 - y0 > spec.img and x1 - x0 < spec.img for (y0, y1, x0, x1) in boxes)
           else None)
    ry0, ry1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
    rx0, rx1 = min(b[2] for b in boxes), max(b[3] for b in boxes)
    rel = [(y0 - ry0, y1 - ry0, x0 - rx0, x1 - rx0) for (y0, y1, x0, x1) in boxes]
    resc = np.array([[metas[i][5], metas[i][6]] for i in pos_file], np.float64)
    plain = bool(((resc[:, 0] == 1.0) & (resc[:, 1] == 0.0)).all())
    signed = any(m[3] == 1 for m in metas.values())
    return {"pos_file": pos_file, "uniq": uniq, "metas": metas, "box": (ry0, ry1, rx0, rx1), "rel": rel,
            "resc": None if plain else resc, "hr": ry1 - ry0, "wr": rx1 - rx0,
            "why": why, "signed": signed}


def _resolve_plan(p: dict, plane: str, side: str | None, spec: CacheSpec, fds, hdr, hdr_rc):
    """Reads in C the crop box of an `area_mixed` series or of one with signed representation
    (`rep == 1`): if it is not `area_mixed` and the pixels in the box are >= 0 it stays
    `("gpu", ...)`, otherwise it computes the uint8 values in memory with `_scales_uint8_cpu`
    without re-reading the files in Python.
    """
    from .raw_crop import _scales_uint8_cpu

    k_sl, hr, wr = spec.slices, p["hr"], p["wr"]
    reg = np.empty((k_sl, hr, wr), np.uint16)
    base = reg.ctypes.data
    reg_bytes = 2 * hr * wr
    first, tasks, dups = {}, [], []
    for pos, i in enumerate(p["pos_file"]):
        if i in first:
            dups.append((pos, first[i]))
            continue
        first[i] = pos
        rows_, cols_, bits, rep, shift = p["metas"][i][:5]
        t = np.zeros(cfast.LIO_T_N, np.int64)
        t[cfast.LIO_FD] = fds[i]
        if hdr_rc[i] == 0:
            t[cfast.LIO_PIX_OFF], t[cfast.LIO_PIX_LEN], t[cfast.LIO_RESUME] = hdr[i, 0], hdr[i, 1], -1
        else:
            t[cfast.LIO_PIX_OFF], t[cfast.LIO_PIX_LEN], t[cfast.LIO_RESUME] = -1, -1, hdr[i, 32]
        t[cfast.LIO_ROWS], t[cfast.LIO_COLS], t[cfast.LIO_BITS] = rows_, cols_, bits
        t[cfast.LIO_REP], t[cfast.LIO_SHIFT] = rep, shift
        t[cfast.LIO_Y0], t[cfast.LIO_Y1], t[cfast.LIO_X0], t[cfast.LIO_X1] = p["box"]
        t[cfast.LIO_DEST] = base + pos * reg_bytes
        t[cfast.LIO_DIRECT] = 1 if DIRECT[0] else 0
        tasks.append(t)
    if tasks:
        status, _found, _nread = cfast.lio_read(np.stack(tasks), threads=THREADS[0])
        if ((status != cfast.LIO_OK) & (status != cfast.LIO_VALUES)).any():
            return None
    else:
        status = np.zeros(0, np.int64)
    if p["why"] is None and (status == cfast.LIO_OK).all():
        for dst_pos, src_pos in dups:
            reg[dst_pos] = reg[src_pos]
        return ("gpu", {"region": reg, "boxes": p["rel"], "resc": p["resc"]}), len(tasks), 1
    by_file: dict[int, np.ndarray] = {}
    finals = []
    for pos, i in enumerate(p["pos_file"]):
        f = by_file.get(i)
        if f is None:
            _, _, _, rep, _, slope, intercept, _ = p["metas"][i]
            arr = reg[pos].view(np.int16) if rep else reg[pos]
            f = by_file[i] = D._apply_rescale(arr, slope, intercept, False)
        finals.append(f)
    why = p["why"] or "values"
    return (("cpu", {"u8": _scales_uint8_cpu(finals, p["rel"], spec.img, plane, side), "why": why}),
            len(tasks), 0)


def _aligned(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def build_study_lean(study_dir: str, slots: dict[int, str], spec: CacheSpec, errors=None, alloc=None):
    """`raw_crop.build_study_raw_np(study_dir, slots, spec, errors)` with I/O in C -> identical (payload,
    mask, meta), or None (the caller redoes the study with `build_study_raw_np`). Without `alloc` the
    payload lives in new arrays; with `alloc(nbytes) -> (writable uint8 array, token)` it lives in that
    memory (the shared buffer), and `meta["alloc"]` = (token, layout) with layout = [(key, dtype,
    shape, offset)]. `alloc` is called at most once: if afterwards this returns None or raises, the
    caller cancels the reservation.
    """
    from .raw_crop import build_series_raw

    if not usable(spec):
        return None
    # An abandoned attempt will be redone in full: the errors of its fallbacks are published only on
    # success, otherwise the caller would count them twice.
    pending_errors = D.DecodeErrors() if errors is not None else None
    items = list(slots.items())
    sdirs = [os.path.join(study_dir, uid) for _, uid in items]
    listing = cfast.lio_listdir(sdirs, threads=max(1, len(sdirs)))
    if any(x is None for x in listing):
        return None
    files_by = [sorted(x) for x in listing]
    paths, starts = [], []
    for d, files in zip(sdirs, files_by):
        starts.append(len(paths))
        paths += [os.path.join(d, f) for f in files]
    heads, head_len, fds, offs, rc, hdr, hdr_rc = cfast.lio_scan(paths, D._HEADER_BYTES, True,
                                                                 threads=THREADS[0])
    fds_all = fds.copy()
    try:
        if (head_len < 0).any():
            return None                                              # open or read failed: usual path
        hdrs = {j: _probe(sdirs[s], uid, files_by[s], heads, head_len, starts[s])
                for s, (j, uid) in enumerate(items)}
        side, side_src = study_side(list(hdrs.values()), tie=spec.side_tie)
        n_seq = len(spec.active_slots)
        built: list = []                                             # per series: None, ("lean", plan), ("old", res)
        fallback_reads = pre_lean_reads = pre_lean_series = 0
        for s, (j, uid) in enumerate(items):
            files = files_by[s]
            order = _order(sdirs[s], files, heads, head_len, offs, rc, starts[s])
            if not order.files:
                built.append(None)
                continue
            index_of = {f: starts[s] + k for k, f in enumerate(files)}
            plan = _plan_series(spec, order, index_of, hdrs[j], heads, hdr, hdr_rc)
            if plan is not None:
                if plan["why"] is None and not plan["signed"]:
                    built.append(("lean", plan))
                else:
                    resolved = _resolve_plan(plan, spec.active_slots[j][0], side, spec, fds, hdr, hdr_rc)
                    if resolved is None:
                        return None
                    res, n_lread, n_lser = resolved
                    fallback_reads += spec.slices
                    pre_lean_reads += n_lread
                    pre_lean_series += n_lser
                    built.append(("old", res))
            else:
                res, old_info = build_series_raw(sdirs[s], spec.active_slots[j][0], side, spec, pending_errors,
                                                 hdr=hdrs[j], order=order)
                fallback_reads += old_info["n_read"]
                built.append(None if res is None else ("old", res))
        # layout: raw (int16), u8, resc (float64), meta (int64), as in build_study_raw_np
        k_sl = spec.slices
        raw_n = u8_n = resc_n = n_rows = 0
        for b in built:
            if b is None:
                continue
            n_rows += 1
            if b[0] == "lean":
                raw_n += k_sl * b[1]["hr"] * b[1]["wr"]
                resc_n += 0 if b[1]["resc"] is None else k_sl
            elif b[1][0] == "gpu":
                raw_n += b[1][1]["region"].size
                resc_n += 0 if b[1][1].get("resc") is None else b[1][1]["resc"].shape[0]
            else:
                u8_n += sum(v.size for v in b[1][1]["u8"])
        o_raw = 0
        o_u8 = _aligned(o_raw + 2 * raw_n)
        o_resc = _aligned(o_u8 + u8_n)
        o_meta = _aligned(o_resc + 16 * resc_n)
        total = _aligned(o_meta + 8 * N_META * n_rows)
        token = None
        if alloc is None:
            mem = np.empty(max(total, _ALIGN), np.uint8)
        else:
            mem, token = alloc(max(total, _ALIGN))
            if mem is None:
                return None
        raw = mem[o_raw:o_raw + 2 * raw_n].view(np.int16)
        u8 = mem[o_u8:o_u8 + u8_n]
        resc = mem[o_resc:o_resc + 16 * resc_n].view(np.float64).reshape(resc_n, 2)
        meta_a = mem[o_meta:o_meta + 8 * N_META * n_rows].view(np.int64).reshape(n_rows, N_META)
        base = mem.ctypes.data
        tasks, dups = [], []
        mask = np.zeros(spec.shape[0], dtype=bool)
        info = {"side": side or "", "side_source": side_src, "n_read": fallback_reads,
                "cpu_series": {}, "lean_series": pre_lean_series, "lean_reads": pre_lean_reads}
        raw_off = u8_off = resc_off = r = 0
        for (j, _uid), b in zip(items, built):
            if b is None:
                continue
            plane = spec.active_slots[j][0]
            row = meta_a[r]
            row[:] = 0
            row[M_SLOT], row[M_K], row[M_FLIP] = j, k_sl, _flip_code(plane, side)
            if b[0] == "lean" or b[1][0] == "gpu":
                if b[0] == "lean":
                    p = b[1]
                    hr, wr, boxes, rs = p["hr"], p["wr"], p["rel"], p["resc"]
                    reg_bytes = 2 * hr * wr
                    first = {}
                    rr = read_range(k_sl, int(row[M_FLIP]))
                    for pos, i in enumerate(p["pos_file"]):
                        if rr is not None and not rr[0] <= pos < rr[1]:
                            continue                   # outside `KNEE_READ_SLICES`: not read
                        if i in first:
                            dups.append((raw_off + pos * hr * wr, raw_off + first[i] * hr * wr, hr * wr))
                            continue
                        first[i] = pos
                        rows_, cols_, bits, rep, shift = p["metas"][i][:5]
                        t = np.zeros(cfast.LIO_T_N, np.int64)
                        t[cfast.LIO_FD] = fds[i]
                        if hdr_rc[i] == 0:
                            t[cfast.LIO_PIX_OFF], t[cfast.LIO_PIX_LEN], t[cfast.LIO_RESUME] = hdr[i, 0], hdr[i, 1], -1
                        else:
                            t[cfast.LIO_PIX_OFF], t[cfast.LIO_PIX_LEN], t[cfast.LIO_RESUME] = -1, -1, hdr[i, 32]
                        t[cfast.LIO_ROWS], t[cfast.LIO_COLS], t[cfast.LIO_BITS] = rows_, cols_, bits
                        t[cfast.LIO_REP], t[cfast.LIO_SHIFT] = rep, shift
                        t[cfast.LIO_Y0], t[cfast.LIO_Y1], t[cfast.LIO_X0], t[cfast.LIO_X1] = p["box"]
                        t[cfast.LIO_DEST] = base + o_raw + 2 * raw_off + pos * reg_bytes
                        t[cfast.LIO_DIRECT] = 1 if DIRECT[0] else 0
                        tasks.append(t)
                    info["n_read"] += k_sl                 # valid positions of the band, as in raw_crop
                    info["lean_reads"] += len(first)       # files actually read, after deduplication
                    info["lean_series"] += 1
                    size = k_sl * hr * wr
                else:
                    d = b[1][1]
                    reg = d["region"]
                    hr, wr, boxes, rs = reg.shape[1], reg.shape[2], d["boxes"], d.get("resc")
                    raw[raw_off:raw_off + reg.size] = reg.reshape(-1).view(np.int16)
                    size = reg.size
                row[M_MODE], row[M_HR], row[M_WR], row[M_RAW_OFF] = MODE_GPU, hr, wr, raw_off
                row[M_RESC] = -1
                if rs is not None:
                    row[M_RESC] = resc_off
                    resc[resc_off:resc_off + rs.shape[0]] = rs
                    resc_off += rs.shape[0]
                for k, bx in enumerate(boxes):
                    row[M_BOX + 4 * k: M_BOX + 4 * k + 4] = bx
                raw_off += size
            else:
                d = b[1][1]
                row[M_MODE], row[M_U8_OFF] = MODE_CPU, u8_off
                info["cpu_series"][d["why"]] = info["cpu_series"].get(d["why"], 0) + 1
                for v in d["u8"]:
                    u8[u8_off:u8_off + v.size] = v.reshape(-1)
                    u8_off += v.size
            r += 1
            for k in range(spec.n_scales):
                mask[j + k * n_seq] = True
        if tasks:
            status, _found, _nread = cfast.lio_read(np.stack(tasks), threads=THREADS[0])
            if (status != cfast.LIO_OK).any():
                return None                                          # I/O or values: usual path
        for dst, src, n in dups:                                     # the repeated slices of the band
            raw[dst:dst + n] = raw[src:src + n]
        payload = {"raw": raw, "u8": u8, "resc": resc, "meta": meta_a}
        if alloc is not None:
            info["alloc"] = (token, [("raw", "<i2", (raw_n,), o_raw), ("u8", "|u1", (u8_n,), o_u8),
                                     ("resc", "<f8", (resc_n, 2), o_resc), ("meta", "<i8", (n_rows, N_META), o_meta)])
        if pending_errors is not None:
            errors.items.extend(pending_errors.items)
        return payload, mask, info
    finally:
        cfast.lio_close(fds_all)


def same_payload(a: dict, b: dict) -> bool:
    """Two bit-identical payloads (shape, dtype and values of every array)."""
    for k in ("raw", "u8", "resc", "meta"):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if not np.array_equal(np.ascontiguousarray(x).view(np.uint8),
                              np.ascontiguousarray(y).view(np.uint8)):
            return False
    return True
