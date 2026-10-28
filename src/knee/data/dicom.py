"""DICOM reading: headers, slice order, pixels.

Decoding, not the GPU, is the runtime bottleneck, so as little as possible is read here: headers
with `stop_before_pixels`, and pixels only for the slices that actually end up in the cache.

A decoding error is never swallowed. A slice that fails to decode and becomes black is
indistinguishable from a black knee, and it would be a silent defect among 819,000 files: here it
is counted and reported.
"""
from __future__ import annotations

import io
import os
import re
import threading
from dataclasses import dataclass, field

import numpy as np
import pydicom

from .geometry import image_center_xyz, through_plane_key

GEOMETRY_TAGS = [
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
    "Rows", "Columns", "Laterality", "InstanceNumber",
]

# Minimum fraction of slices with usable geometry for the physical ordering to be considered
# reliable. Below this threshold the fallback is used: a physical order rebuilt on a third of the
# stack is worse than a consistent InstanceNumber.
GEOMETRY_QUORUM = 0.8


@dataclass
class SeriesHeader:
    """What needs to be known about a series without opening its pixels."""
    uid: str
    path: str
    n_files: int
    rows: int | None = None
    cols: int | None = None
    pixel_spacing: tuple[float, float] | None = None
    center: np.ndarray | None = None
    laterality: str | None = None


@dataclass
class OrderResult:
    files: list[str]
    method: str  # "projection" | "instance" | "name"
    n_geometry: int = 0


class DecodeErrors:
    """Log of decoding failures, shared by the reader threads.

    `list.append` is atomic under the GIL, so no lock is needed: that is why it is a list and not an
    incremented counter.
    """

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def record(self, path: str, exc: Exception) -> None:
        self.items.append((path, f"{type(exc).__name__}: {exc}"))

    def __len__(self) -> int:
        return len(self.items)


def _natural_key(name: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(name)))


def _read_header(path: str):
    return pydicom.dcmread(path, force=True, stop_before_pixels=True,
                           specific_tags=GEOMETRY_TAGS)


def _as_floats(value, n: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        v = np.asarray([float(x) for x in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return v if v.size >= n and np.isfinite(v).all() else None


def probe_series(series_dir: str, uid: str = "") -> SeriesHeader:
    """Reads the header of *one* slice of the series: that is where the geometry comes from.

    In-plane geometry (spacing, size, orientation) is a property of the series, not of the single
    slice, so one slice is enough. Position is not (it changes from slice to slice), but here it is
    needed only for the centre, and for the side the median over the series of the study absorbs the
    fact that the sampled slice is arbitrary.
    """
    files = sorted(os.listdir(series_dir)) if os.path.isdir(series_dir) else []
    h = SeriesHeader(uid=uid, path=series_dir, n_files=len(files))
    if not files:
        return h
    try:
        vals = _probe_values(os.path.join(series_dir, files[len(files) // 2]))
    except Exception:
        return h
    h.rows, h.cols, h.pixel_spacing, h.laterality, ipp, iop = vals
    if ipp is not None and iop is not None and h.rows and h.cols and h.pixel_spacing:
        h.center = image_center_xyz(np.asarray(ipp), np.asarray(iop), h.pixel_spacing, h.rows, h.cols)
    return h


def _probe_values_pydicom(path: str) -> tuple:
    ds = _read_header(path)
    rows = int(getattr(ds, "Rows", 0)) or None
    cols = int(getattr(ds, "Columns", 0)) or None
    ps = _as_floats(getattr(ds, "PixelSpacing", None), 2)
    ps = (float(ps[0]), float(ps[1])) if ps is not None else None
    lat = getattr(ds, "Laterality", None)
    lat = str(lat).strip().upper()[:1] if lat else None
    ipp = _as_floats(getattr(ds, "ImagePositionPatient", None), 3)
    iop = _as_floats(getattr(ds, "ImageOrientationPatient", None), 6)
    return (rows, cols, ps, lat, None if ipp is None else tuple(ipp.tolist()),
            None if iop is None else tuple(iop.tolist()))


# The series header without pydicom: `probe_series` reads one slice per series and pydicom took
# ~5.5 ms, ~11% of the worker CPU on raw crops. Here the same six tags come from the 16 KB prefix
# with the element walk of `_order_tags_py`; anything unexpected -> None and pydicom. The first
# `_PROBE_CHECKS` series of every process are read both ways: a disagreement turns the fast path off.
_PROBE_FAST = [True]
_PROBE_CHECKS = [8]


def _probe_values(path: str) -> tuple:
    if _PROBE_FAST[0]:
        try:
            got = _probe_values_fast(_read_prefix(path, _HEADER_BYTES))
        except Exception:
            got = None
        if got is not None:
            if _PROBE_CHECKS[0] > 0:
                _PROBE_CHECKS[0] -= 1
                ref = _probe_values_pydicom(path)
                if ref != got:
                    _PROBE_FAST[0] = False
                    return ref
            return got
    return _probe_values_pydicom(path)


def _probe_values_fast(buf: bytes):
    import struct

    n = len(buf)
    if n < 140 or buf[128:132] != b"DICM":
        return None
    pos, ts, raw = 132, None, {}
    want = {(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0060),
            (0x0028, 0x0010), (0x0028, 0x0011), (0x0028, 0x0030)}
    while pos + 8 <= n:
        group, elem = struct.unpack_from("<HH", buf, pos)
        if group != 0x0002:
            if pos == 132 or ts is None or ts == _TS_IMPLICIT_LE or ts == _TS_BIG_ENDIAN:
                return None                   # Explicit VR Little Endian only
            if group > 0x0028 or (group == 0x0028 and elem > 0x0030):
                break
        if (group, elem) in want or (group == 0x0002 and elem == 0x0010):
            vr = buf[pos + 4:pos + 6]
            if vr in _VR_LONG:
                return None
            length = struct.unpack_from("<H", buf, pos + 6)[0]
            val, end = pos + 8, pos + 8 + length
            if end > n:
                return None
            if group == 0x0002:
                ts = buf[val:end].rstrip(b"\x00 ")
            else:
                raw[(group, elem)] = (vr, buf[val:end])
            pos = end
        else:
            pos = _skip_element(buf, pos, n)
    else:
        return None                           # buffer ended before group 0028 beyond 0030

    def us(key):
        v = raw.get(key)
        if v is None:
            return None
        if v[0] != b"US" or len(v[1]) != 2:
            raise ValueError("unexpected US")
        return int(struct.unpack("<H", v[1])[0]) or None

    def ds(key, k):
        v = raw.get(key)
        if v is None:
            return None
        if v[0] != b"DS":
            raise ValueError("unexpected DS")
        f = _ds_floats(v[1])
        if f is None:
            raise ValueError("unreadable DS")       # pydicom might read it differently
        a = np.asarray(f, dtype=np.float64)
        return tuple(a.tolist()) if a.size >= k and np.isfinite(a).all() else None

    rows, cols = us((0x0028, 0x0010)), us((0x0028, 0x0011))
    ps = ds((0x0028, 0x0030), 2)
    ps = (float(ps[0]), float(ps[1])) if ps is not None else None
    lat = raw.get((0x0020, 0x0060))
    if lat is not None:
        if lat[0] != b"CS" or b"\\" in lat[1]:
            raise ValueError("unexpected Laterality")
        txt = lat[1].decode("ascii").strip("\x00 ")
        lat = txt.upper()[:1] if txt else None
    return rows, cols, ps, lat, ds((0x0020, 0x0032), 3), ds((0x0020, 0x0037), 6)


# --- fast read of the three ordering tags --------------------------------------
# Profiling decoding on the Kaggle mount: 59% of the time per study is `order_series_files`: pydicom
# takes ~4.8 ms per header, and reads one per file of the series (~38 per series, ~150 per study) to
# extract three tags. Here the elements of the raw file are walked and the walk stops at group 0020:
# ~0.1 ms. Anything unexpected (no preamble, big endian, undefined-length sequence before group 0020,
# header longer than the buffer) returns None and falls back to pydicom, so the result is the same
# as before by construction: only who reads the bytes changes.
_HEADER_BYTES = 16384
_VR_LONG = {b"OB", b"OW", b"OF", b"OD", b"OL", b"OV", b"SQ", b"SV", b"UC", b"UN", b"UR",
            b"UT", b"UV"}
_TS_IMPLICIT_LE = b"1.2.840.10008.1.2"
_TS_BIG_ENDIAN = b"1.2.840.10008.1.2.2"


# How many bytes reading those 16 KB costs. `open(path, "rb").read(16384)` goes through Python's
# buffer, of size `st_blksize`: on NFS that is the mount's read size (512 KB on /kaggle/input), so the
# request read 512 KB and the kernel read-ahead the rest of the file (~1.4x the bytes per study). Of
# the ~130 files of a study whose header is read, ~34 are not read again for pixels.
# "exact": unbuffered read of only `_HEADER_BYTES`, with POSIX_FADV_RANDOM turning off read-ahead on
# that descriptor; "buffered": as before. The bytes read are the same from the start of the file, so
# the parser result does not change.
# "exact" is the default: in an A/B on the T4 benchmark it read -37% bytes from the NFS server (54
# against 84 MB per study) and was -11.7% in steady state; worker CPU dropped only 2.6%: the mount was
# a hidden limit, not the CPU.
_HEADER_READ = ["exact"]

# Files opened for the header during ordering stay open with their first bytes
# (`order_series_files(keep_fds=True)`, from `cache.build_series`): `read_slice` reads only the rest
# from the same descriptor, and `fast_order_tags` re-reads the headers from memory. On the NFS mount
# an open costs ~0.2 ms of CPU and every slice of the band did two, plus re-reading the first 16 KB.
# Same bytes. `close_kept` closes the ones not read.
_FD_KEEP = [True]
_FD_CACHE: dict[str, tuple[int, bytes]] = {}


def _read_prefix(path: str, n: int, keep: bool = False) -> bytes:
    """The first `n` bytes of the file (fewer if it ends first), as `_HEADER_READ` says. `keep`: the
    descriptor stays open in `_FD_CACHE` for `read_slice`.
    """
    cached = _FD_CACHE.get(path)
    if cached is not None and n <= _HEADER_BYTES:
        return cached[1][:n]
    if _HEADER_READ[0] != "exact":
        with open(path, "rb") as fh:
            return fh.read(n)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    kept = False
    try:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            except OSError:
                pass
        chunks, got = [], 0
        while got < n:
            b = os.read(fd, n - got)
            if not b:
                break
            chunks.append(b)
            got += len(b)
        data = b"".join(chunks)
        if keep and n == _HEADER_BYTES:
            _FD_CACHE[path] = (fd, data)
            kept = True
        return data
    finally:
        if not kept:
            os.close(fd)


def _read_rest(fd: int, head: bytes) -> bytes:
    """The whole file from a descriptor already read for `len(head)` bytes (and closes it)."""
    try:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_NORMAL)
            except OSError:
                pass
        chunks = [head]
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks)
    finally:
        os.close(fd)


def close_kept(series_dir: str, files: list[str]) -> None:
    """Closes the descriptors kept for the files of the series that nobody read."""
    for f in files:
        ent = _FD_CACHE.pop(os.path.join(series_dir, f), None)
        if ent is not None:
            try:
                os.close(ent[0])
            except OSError:
                pass


def _ds_floats(raw: bytes) -> list[float] | None:
    try:
        return [float(x) for x in raw.decode("ascii").strip("\x00 ").split("\\")]
    except (UnicodeDecodeError, ValueError):
        return None


# The element walks in C (`cfast.walk_order`, `cfast.walk_pixels`) where the library is loaded in this
# process: same logic, same bytes; where C stops, the Python walk is redone, and the first
# `_C_ORDER_CHECKS` headers of every process are read both ways (a disagreement turns C off). The two
# Python walks (~64 elements per file, 138 headers and 95 whole files per study) were ~9% of the
# decoding CPU.
_C_WALK = [True]
_C_ORDER_CHECKS = [16]


def _c_walk():
    if not _C_WALK[0]:
        return None
    from . import cfast
    return cfast if cfast.ok() else None


def fast_order_tags(path: str, keep: bool = False
                    ) -> tuple[list[float] | None, list[float] | None, float | None] | None:
    """(ImagePositionPatient, ImageOrientationPatient, InstanceNumber) from the file bytes.

    `None` = unsupported format: the caller falls back to pydicom. `keep`: the file stays open for
    `read_slice` (`_FD_CACHE`).
    """
    return _order_tags(_read_prefix(path, _HEADER_BYTES, keep))


def _order_tags(buf: bytes, offs: list[int] | None | bool = False):
    """`offs`: the offsets of `walk_order` already computed for this `buf` (`None` = C stopped), or
    `False` = to be computed here.
    """
    c = _c_walk()
    if c is not None:
        if offs is False:
            offs = c.walk_order(buf)
        if offs is not None:
            got = _order_values(buf, offs)
            if _C_ORDER_CHECKS[0] > 0:
                _C_ORDER_CHECKS[0] -= 1
                try:
                    ref = _order_tags_py(buf)
                except Exception:
                    ref = "error"
                if ref != got:
                    _C_WALK[0] = False              # a disagreement: Python only from now on
                    return _order_tags_py(buf)
            return got
    return _order_tags_py(buf)


def _order_values(buf: bytes, offs: list[int]):
    """The values of `_order_tags_py` from the offsets of the C walk."""
    ipp = _ds_floats(buf[offs[0]:offs[0] + offs[1]]) if offs[0] >= 0 else None
    iop = _ds_floats(buf[offs[2]:offs[2] + offs[3]]) if offs[2] >= 0 else None
    inst = None
    if offs[4] >= 0:
        try:
            inst = float(buf[offs[4]:offs[4] + offs[5]].decode("ascii").strip("\x00 "))
        except (UnicodeDecodeError, ValueError):
            inst = None
    return ipp, iop, inst


def _order_tags_py(buf: bytes):
    import struct

    n = len(buf)
    if n < 140 or buf[128:132] != b"DICM":
        return None
    pos, explicit, ts = 132, True, None
    ipp = iop = inst = None
    while pos + 8 <= n:
        group, elem = struct.unpack_from("<HH", buf, pos)
        if group != 0x0002 and ts is not None:
            # End of the meta header: from here the syntax is decided by the transfer syntax.
            if ts == _TS_BIG_ENDIAN:
                return None
            explicit, ts = ts != _TS_IMPLICIT_LE, None
        elif group != 0x0002 and pos == 132:
            return None                       # no meta header: leave it to pydicom
        if group > 0x0020 or (group == 0x0020 and elem > 0x0037):
            break
        if group == 0x0002 or explicit:
            vr = buf[pos + 4:pos + 6]
            if vr in _VR_LONG:
                length, hdr = struct.unpack_from("<I", buf, pos + 8)[0], 12
            else:
                length, hdr = struct.unpack_from("<H", buf, pos + 6)[0], 8
        else:
            length, hdr = struct.unpack_from("<I", buf, pos + 4)[0], 8
        if length == 0xFFFFFFFF:
            return None                       # undefined length: do not skip blindly
        val = pos + hdr
        end = val + length
        if end > n:
            return None                       # header beyond the buffer
        if group == 0x0002 and elem == 0x0010:
            ts = buf[val:end].rstrip(b"\x00 ")
        elif group == 0x0020:
            if elem == 0x0013:
                try:
                    inst = float(buf[val:end].decode("ascii").strip("\x00 "))
                except (UnicodeDecodeError, ValueError):
                    inst = None
            elif elem == 0x0032:
                ipp = _ds_floats(buf[val:end])
            elif elem == 0x0037:
                iop = _ds_floats(buf[val:end])
        pos = end
    else:
        return None                           # buffer ended before group 0020
    if ts is not None:
        return None                           # only the meta header in the buffer
    return ipp, iop, inst


def _read_file_key(args: tuple) -> tuple[str, float | None, float | None]:
    series_dir, f = args[0], args[1]
    keep = len(args) > 2 and bool(args[2])
    path = os.path.join(series_dir, f)
    try:
        fast = fast_order_tags(path, keep)
    except Exception:
        fast = None
    return _key_from_fast(f, path, fast)


def _key_from_fast(f: str, path: str, fast) -> tuple[str, float | None, float | None]:
    """The tail of `_read_file_key`: the key from the hand-read tags, or from pydicom if `fast` is None."""
    k = inst = None
    if fast is not None:
        ipp, iop, inst = _as_floats(fast[0], 3), _as_floats(fast[1], 6), fast[2]
        if ipp is not None and iop is not None:
            k = through_plane_key(ipp, iop)
        return f, k, inst
    try:
        ds = _read_header(path)
        ipp = _as_floats(getattr(ds, "ImagePositionPatient", None), 3)
        iop = _as_floats(getattr(ds, "ImageOrientationPatient", None), 6)
        if ipp is not None and iop is not None:
            k = through_plane_key(ipp, iop)
        n = getattr(ds, "InstanceNumber", None)
        if n is not None:
            inst = float(n)
    except Exception:
        pass
    return f, k, inst


# The series headers read in C (`cfast.scan_prefix`: open, fadvise, read of the prefix and
# `walk_order` per block of files, without a Python task per file): same bytes and same keys as
# `_read_file_key`, which remains for the files C does not open and where the library is missing
# (Windows).
_SCAN = [True]


def _scan_keys(series_dir: str, files: list[str], keep: bool):
    if not _SCAN[0] or _HEADER_READ[0] != "exact" or _c_walk() is None:
        return None
    paths = [os.path.join(series_dir, f) for f in files]
    if any(p in _FD_CACHE for p in paths):
        return None
    from . import cfast
    try:
        got = cfast.scan_prefix(paths, _HEADER_BYTES, keep)
    except Exception:
        got = None
    if got is None:
        return None
    heads, head_len, fds, offs, rc = got
    keys = []
    for i, (f, path) in enumerate(zip(files, paths)):
        n = int(head_len[i])
        if n < 0:                                 # open failed: as before, with the Python walk
            keys.append(_read_file_key((series_dir, f, keep)))
            continue
        buf = heads[i, :n].tobytes()
        if fds[i] >= 0:
            _FD_CACHE[path] = (int(fds[i]), buf)
        try:
            fast = _order_tags(buf, offs[i].tolist() if rc[i] == 0 else None)
        except Exception:
            fast = None
        keys.append(_key_from_fast(f, path, fast))
    return keys


def order_series_files(series_dir: str, keep_fds: bool = False) -> OrderResult:
    """The files of the series ordered along the axis perpendicular to the plane.

    Three levels, in order of trust: the projection on the normal (physical, signed), `InstanceNumber`
    (usually follows the projection up to the sign, but interleaved or multi-echo acquisitions do not
    guarantee it), the file name (which here means nothing, being a UID, and only makes the desperate
    case deterministic). `keep_fds`: the files stay open for `read_slice` (the caller then calls
    `close_kept`).
    """
    files = sorted(os.listdir(series_dir)) if os.path.isdir(series_dir) else []
    if len(files) <= 1:
        return OrderResult(files, "name", 0)

    if len(files) > 4:
        keys = _scan_keys(series_dir, files, keep_fds)
        if keys is None:
            from .pools import thread_pool
            with thread_pool("order", min(16, len(files))) as ex:
                keys = list(ex.map(_read_file_key, [(series_dir, f, keep_fds) for f in files]))
    else:
        keys = [_read_file_key((series_dir, f, keep_fds)) for f in files]
    return order_from_keys(files, keys)


def order_from_keys(files: list[str], keys: list[tuple[str, float | None, float | None]]) -> OrderResult:
    """The tail of `order_series_files` from the keys `(file, projection, InstanceNumber)` of the files (in
    name order, more than one): also used by `lean` with keys from the headers read in C.
    """
    n_geom = sum(k is not None for _, k, _ in keys)
    quorum = max(2, int(GEOMETRY_QUORUM * len(files)))
    if n_geom >= quorum:
        spare = float(np.median([k for _, k, _ in keys if k is not None]))
        ordered = sorted(keys, key=lambda r: (r[1] if r[1] is not None else spare,
                                              r[2] if r[2] is not None else np.inf))
        return OrderResult([f for f, _, _ in ordered], "projection", n_geom)

    n_inst = sum(i is not None for _, _, i in keys)
    if n_inst >= quorum:
        ordered = sorted(keys, key=lambda r: (r[2] if r[2] is not None else np.inf,
                                              _natural_key(r[0])))
        return OrderResult([f for f, _, _ in ordered], "instance", n_geom)

    return OrderResult(sorted(files, key=_natural_key), "name", n_geom)


# --- reader of uncompressed pixels ----------------------------------------------------
# The competition DICOMs are all Explicit VR Little Endian (checked on 9,336 slices out of 9,336):
# the pixels are a raw array at the end of the file. Parsing with pydicom (0.8 ms) + `pixel_array`
# (1.1 ms) took ~1.9 ms per slice, ~93 slices per study. Here the file elements are walked up to
# PixelData, collecting the few group 0028 tags needed, and the array is taken with `np.frombuffer`.
# Anything that is not the simple case (no preamble, another transfer syntax, several frames or
# samples, unusual bits, multi-valued DS, undefined-length UN) returns None and falls back to pydicom.
# The first calls of every process compare the two paths bit for bit; a disagreement turns the fast
# one off.
_TS_EXPLICIT_LE = b"1.2.840.10008.1.2.1"
_PIX_TAGS = {0x0002: "SamplesPerPixel", 0x0004: "PhotometricInterpretation",
             0x0008: "NumberOfFrames", 0x0010: "Rows", 0x0011: "Columns",
             0x0100: "BitsAllocated", 0x0101: "BitsStored", 0x0103: "PixelRepresentation",
             0x1052: "RescaleIntercept", 0x1053: "RescaleSlope"}
# pydicom 3 clears the unused high bits (or sign-extends) of uncompressed pixels when Bits Stored <
# Bits Allocated (`correct_unused_bits`); pydicom 2 does not.
_PYDICOM_SHIFTS_UNUSED_BITS = int(pydicom.__version__.split(".")[0]) >= 3
_FAST_PIX_CHECKS = [3]      # comparisons with pydicom left in this process
_FAST_PIX_OK = [True]       # the fast path is still valid in this process


def _skip_undefined_item(buf: bytes, pos: int, n: int) -> int:
    """Skips the elements of an undefined-length item up to (FFFE,E00D)."""
    import struct
    while pos + 8 <= n:
        group, elem = struct.unpack_from("<HH", buf, pos)
        if group == 0xFFFE:
            if elem != 0xE00D:
                raise ValueError("item not closed")
            return pos + 8
        pos = _skip_element(buf, pos, n)
    raise ValueError("item beyond the file")


def _skip_undefined_sequence(buf: bytes, pos: int, n: int) -> int:
    """Skips the items of an undefined-length sequence up to (FFFE,E0DD)."""
    import struct
    while pos + 8 <= n:
        group, elem, length = struct.unpack_from("<HHI", buf, pos)
        if group != 0xFFFE:
            raise ValueError("expected an item")
        pos += 8
        if elem == 0xE0DD:
            return pos
        if elem != 0xE000:
            raise ValueError("unexpected delimiter")
        pos = _skip_undefined_item(buf, pos, n) if length == 0xFFFFFFFF else pos + length
    raise ValueError("sequence beyond the file")


def _skip_element(buf: bytes, pos: int, n: int) -> int:
    """The position after the Explicit VR Little Endian element starting at `pos`."""
    import struct
    vr = buf[pos + 4:pos + 6]
    if vr in _VR_LONG:
        length, hdr = struct.unpack_from("<I", buf, pos + 8)[0], 12
    else:
        length, hdr = struct.unpack_from("<H", buf, pos + 6)[0], 8
    if length == 0xFFFFFFFF:
        if vr != b"SQ":
            raise ValueError(f"{vr!r} with undefined length")
        return _skip_undefined_sequence(buf, pos + hdr, n)
    end = pos + hdr + length
    if end > n:
        raise ValueError("element beyond the file")
    return end


def _ds_single(raw: bytes) -> float | None:
    """A DS as pydicom reads it: empty = absent; several values = unsupported."""
    s = raw.decode("ascii").strip("\x00 ")
    if not s:
        return None
    if "\\" in s:
        raise ValueError("multi-valued DS")
    return float(s)


def _read_native_pixels(raw: bytes) -> np.ndarray | None:
    """The slice as `read_slice` would return it going through pydicom, or None if the file is not the
    simple case: Explicit VR Little Endian, one frame, one sample, 8 or 16 bits.
    """
    walked = _pixel_tags(raw)
    if walked is None:
        return None
    tags, pix = walked
    return _native_pixels(raw, tags, pix)


def _pixel_tags(raw: bytes):
    """(group 0028 tags, (offset, length) of PixelData) or None, from the C walk if available (the
    comparison with pydicom in the first `read_slice` calls covers this too).
    """
    c = _c_walk()
    if c is not None:
        r = c.walk_pixels(raw)
        if r is not None:
            tags = {}
            for t, name in enumerate(_PIX_TAGS.values()):
                kind, a, b = r[2 + 3 * t:5 + 3 * t]
                if kind == 1:
                    tags[name] = a
                elif kind == 2:
                    tags[name] = raw[a:a + b]
            return tags, (r[0], r[1])
    return _pixel_tags_py(raw)


def _pixel_tags_py(raw: bytes):
    import struct
    n = len(raw)
    if n < 140 or raw[128:132] != b"DICM":
        return None
    pos, ts, tags, pix = 132, None, {}, None
    while pos + 8 <= n:
        group, elem = struct.unpack_from("<HH", raw, pos)
        if group != 0x0002 and ts != _TS_EXPLICIT_LE:
            return None                       # another transfer syntax, or missing meta header
        if group == 0x7FE0 and elem == 0x0010:
            vr = raw[pos + 4:pos + 6]
            if vr not in (b"OW", b"OB"):
                return None
            length = struct.unpack_from("<I", raw, pos + 8)[0]
            if length == 0xFFFFFFFF:
                return None                   # encapsulated: compressed
            pix = (pos + 12, length)
            break
        if group == 0x0002 and elem == 0x0010:
            length = struct.unpack_from("<H", raw, pos + 6)[0]
            ts = raw[pos + 8:pos + 8 + length].rstrip(b"\x00 ")
        elif group == 0x0028 and elem in _PIX_TAGS:
            vr = raw[pos + 4:pos + 6]
            if vr in _VR_LONG:
                return None
            length = struct.unpack_from("<H", raw, pos + 6)[0]
            val = raw[pos + 8:pos + 8 + length]
            if vr == b"US":
                if length != 2:
                    return None
                tags[_PIX_TAGS[elem]] = struct.unpack_from("<H", val)[0]
            else:
                tags[_PIX_TAGS[elem]] = val
        pos = _skip_element(raw, pos, n)
    if pix is None:
        return None
    return tags, pix


def _native_pixels(raw: bytes, tags: dict, pix: tuple[int, int]) -> np.ndarray | None:
    parts = _native_pixel_parts(raw, tags, pix)
    if parts is None:
        return None
    return _apply_rescale(*parts)


def _apply_rescale(a: np.ndarray, slope: float, intercept: float, mono1: bool) -> np.ndarray:
    """From the stored values (float32, or the native integers of `read_slice_parts`) to the pixels of
    `read_slice`: float32 (exact from 16-bit integers), Rescale, then MONOCHROME1.
    """
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if slope != 1.0 or intercept != 0.0:
        a = a * slope + intercept
    if mono1:
        a = a.max() - a
    return a


def _native_pixel_parts(raw: bytes, tags: dict, pix: tuple[int, int], as_int: bool = False):
    """(stored values as float32, slope, intercept, MONOCHROME1) from the fast path, or None. `as_int`:
    with no bits to clear, the values stay the file's integers (a view on the bytes, no float32):
    `read_slice_parts` wants them for the raw crops, which ship them as uint16 (uint16 -> float32 ->
    uint16 for every slice, and Rescale on all of them, was wasted work).
    """
    rows, cols = tags.get("Rows"), tags.get("Columns")
    bits, stored = tags.get("BitsAllocated"), tags.get("BitsStored")
    rep = tags.get("PixelRepresentation")
    if not all(isinstance(v, int) for v in (rows, cols, bits, stored, rep)):
        return None                           # a missing tag, or one with an unexpected VR
    if not rows or not cols or bits not in (8, 16) or rep not in (0, 1):
        return None
    if not 0 < stored <= bits or tags.get("SamplesPerPixel", 1) != 1:
        return None
    frames = tags.get("NumberOfFrames")
    if frames is not None:
        s = frames.decode("ascii").strip("\x00 ")
        if s and int(s) != 1:
            return None
    count = rows * cols
    if pix[1] < count * bits // 8:
        return None
    shift = bits - stored if (_PYDICOM_SHIFTS_UNUSED_BITS and stored < bits) else 0
    c = _c_walk()
    if as_int and shift == 0:
        dtype = np.dtype(("<i" if rep else "<u") + str(bits // 8))
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=pix[0]).reshape(rows, cols)
    elif c is not None:
        # Masking (or sign extension) and float32 in one pass, same values.
        a = c.pix_to_f32(raw, pix[0], count, bits, rep, shift).reshape(rows, cols)
    else:
        dtype = np.dtype(("<i" if rep else "<u") + str(bits // 8))
        arr = np.frombuffer(raw, dtype=dtype, count=count, offset=pix[0]).reshape(rows, cols)
        if shift:
            if rep:                           # signed: the shifts extend the sign
                arr = arr.copy()
                np.left_shift(arr, shift, out=arr)
                np.right_shift(arr, shift, out=arr)
            else:                             # unsigned: the same bits, one pass
                arr = np.bitwise_and(arr, dtype.type((1 << stored) - 1))
        a = arr.astype(np.float32)
    # As in `read_slice`: `float(getattr(ds, "RescaleSlope", 1) or 1)` and friends.
    slope_v = _ds_single(tags["RescaleSlope"]) if "RescaleSlope" in tags else 1
    inter_v = _ds_single(tags["RescaleIntercept"]) if "RescaleIntercept" in tags else 0
    slope, intercept = float(slope_v or 1), float(inter_v or 0)
    photometric = tags.get("PhotometricInterpretation", b"")
    # pydicom strips only the trailing padding of CS values.
    return a, slope, intercept, photometric.decode("ascii").rstrip("\x00 ") == "MONOCHROME1"


def read_slice(path: str, errors: DecodeErrors | None = None) -> np.ndarray | None:
    """Pixels of a slice as float32, with Rescale applied. `None` if it does not decode.

    The file is read whole with a single `read` and parsed from memory: pydicom reads element by
    element and on the Kaggle mount every `read` costs ~40 us of latency (203,000 `read` calls for
    1,441 files, a quarter of decoding). Uncompressed files go through `_read_native_pixels`, and
    pydicom stays for everything else (and for the comparison of the first calls).
    """
    ent = _FD_CACHE.pop(path, None)
    try:
        if ent is not None:
            raw = _read_rest(*ent)
        else:
            with open(path, "rb") as f:
                raw = f.read()
    except Exception as exc:
        if errors is not None:
            errors.record(path, exc)
        return None
    if _FAST_PIX_OK[0]:
        try:
            fast = _read_native_pixels(raw)
        except Exception:
            fast = None
        if fast is not None:
            if _FAST_PIX_CHECKS[0] <= 0:
                return fast
            _FAST_PIX_CHECKS[0] -= 1
            ref = _read_slice_pydicom(raw, path, errors)
            if ref is not None and ref.dtype == fast.dtype and np.array_equal(ref, fast):
                return fast
            _FAST_PIX_OK[0] = False             # a disagreement: pydicom only from now on
            return ref
    return _read_slice_pydicom(raw, path, errors)


def read_slice_parts(path: str, errors: DecodeErrors | None = None):
    """`read_slice` before the Rescale: (stored values as float32, slope, intercept, MONOCHROME1), with
    `_apply_rescale(*parts)` == `read_slice(path)` bit for bit; None if it does not decode. Used by the
    raw crops (`raw_crop`): Philips series with "normalized" RescaleSlope (17% of the corpus) have
    non-integer pixels after the Rescale, but integer ones before, and the Rescale (float32, product
    then sum) is redone identically on the GPU. Same read and same self-checks as `read_slice`.
    """
    ent = _FD_CACHE.pop(path, None)
    try:
        if ent is not None:
            raw = _read_rest(*ent)
        else:
            with open(path, "rb") as f:
                raw = f.read()
    except Exception as exc:
        if errors is not None:
            errors.record(path, exc)
        return None
    if _FAST_PIX_OK[0]:
        try:
            walked = _pixel_tags(raw)
            fast = _native_pixel_parts(raw, *walked, as_int=True) if walked is not None else None
        except Exception:
            fast = None
        if fast is not None:
            if _FAST_PIX_CHECKS[0] <= 0:
                return fast
            _FAST_PIX_CHECKS[0] -= 1
            ref = _read_slice_pydicom(raw, path, errors)
            got = _apply_rescale(*fast)
            if ref is not None and ref.dtype == got.dtype and np.array_equal(ref, got):
                return fast
            _FAST_PIX_OK[0] = False             # a disagreement: pydicom only from now on
    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=True)
        a = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        if errors is not None:
            errors.record(path, exc)
        return None
    return (a, float(getattr(ds, "RescaleSlope", 1) or 1),
            float(getattr(ds, "RescaleIntercept", 0) or 0),
            getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1")


def _read_slice_pydicom(raw: bytes, path: str, errors: DecodeErrors | None) -> np.ndarray | None:
    """The original pydicom-based `read_slice`, from the bytes already read."""
    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=True)
        a = ds.pixel_array.astype(np.float32)
    except Exception as exc:  # exotic transfer syntax, truncated file, missing tags
        if errors is not None:
            errors.record(path, exc)
        return None
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    if slope != 1.0 or intercept != 0.0:
        a = a * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    return a


# --- decoder provisioning --------------------------------------------------------
# The plugins for compressed transfer syntaxes are not in the Kaggle image: without them every
# `pixel_array` on a compressed file raises. In the preprocessing kernel they are installed from the
# internet; the submission kernel has no internet, and the same wheels come from a dataset. The two
# paths are the same function because the failure mode is identical: discovering a missing decoder
# hours into the run.
DECODERS = ("pylibjpeg", "libjpeg", "openjpeg", "gdcm")
REQUIRED_DECODERS = ("pylibjpeg", "libjpeg", "openjpeg")


def _importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def unpack_wheels(wheels_dir, target_dir=None) -> Path | None:
    """Unpacks the .whl files directly into a local folder and puts it on sys.path.

    Takes ~0.3 s against ~26 s for a `pip install` subprocess.
    """
    import importlib
    import sys
    import tempfile
    import zipfile
    from pathlib import Path

    if wheels_dir is None or not Path(wheels_dir).is_dir():
        return None
    target = Path(target_dir) if target_dir else Path(tempfile.gettempdir()) / "knee-pkgs"
    target.mkdir(parents=True, exist_ok=True)
    whls = sorted(Path(wheels_dir).glob("*.whl"))
    if not whls:
        return None
    # The dataset carries wheels for several ABIs (cp311/cp312/cp313): only those of this interpreter and
    # the pure-Python ones are extracted. The others would be wasted I/O, and if for some reason none
    # matches everything is extracted, as before.
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    mine = [w for w in whls if abi in w.name or "none-any" in w.name]
    whls = mine or whls
    for w in whls:
        try:
            with zipfile.ZipFile(w, "r") as zf:
                zf.extractall(target)
        except Exception:
            pass
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    importlib.invalidate_caches()
    return target


_DECODERS_LOCK = threading.Lock()


def ensure_decoders(wheels_dir=None, allow_network: bool = True) -> dict:
    """Installs the decoding plugins if missing. Returns which are available after the attempt.

    With `wheels_dir` it first tries extracting the wheels directly into `/tmp/knee-pkgs` via `zipfile`
    (~0.3 s against ~26 s for pip). If that is not enough, it falls back to offline pip. Under a lock:
    in the kernel the first call can run in a thread after the workers are forked
    (`early.start_for_kernel(decoders_after=True)`) while `main` repeats it, and two extractions at
    once would write the same files.
    """
    with _DECODERS_LOCK:
        return _ensure_decoders(wheels_dir, allow_network)


def _ensure_decoders(wheels_dir, allow_network: bool) -> dict:
    import importlib
    import subprocess
    import sys

    have = {m: _importable(m) for m in DECODERS}
    if all(have[m] for m in REQUIRED_DECODERS):
        return have

    if wheels_dir is not None:
        unpack_wheels(wheels_dir)
        have = {m: _importable(m) for m in DECODERS}
        if all(have[m] for m in REQUIRED_DECODERS):
            return have

    pkgs = ["pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"]
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location"]
    if wheels_dir is not None:
        cmd += ["--no-index", "--no-deps", "--find-links", str(wheels_dir)]
        pkgs += ["pylibjpeg-rle", "python-gdcm"]
    elif not allow_network:
        return have
    subprocess.run(cmd + pkgs, check=False)

    importlib.invalidate_caches()
    return {m: _importable(m) for m in DECODERS}
