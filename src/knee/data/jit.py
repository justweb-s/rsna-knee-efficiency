"""Numba kernels for decoding, bit-identical to the NumPy path they replace.

Profiling 40 studies in a row on one process: `_to_uint8` is 17% of the decoding CPU (7.9 calls
per study, ~10 ms each). The NumPy path traverses the stack (24 x 288 x 288 float32) ~7 times:
the maximum (NaN check), two masked selections for the percentile tails, then per slice
subtraction, multiplication, clipping and conversion. Here there are two passes: `tails` (the
values of the two tails, in their original order; the NaN check stays NumPy's `flat.max()`) and
`normalize` (the same float64 arithmetic as NumPy 2, element by element, no FMA).

Lazy import and explicit compilation (`warmup`), to be done in the main process before creating
the workers: with fork the workers inherit the compiled functions. If numba is missing or does
not compile, `ok()` is False and callers stay on NumPy. Callers compare the first calls of every
process with NumPy (`cache._to_uint8`).
"""
from __future__ import annotations

import time

import numpy as np

_STATE: dict = {"tried": False, "ok": False, "error": None, "compile_s": None,
                "tails": None, "normalize": None}


def _build() -> None:
    import numba as nb

    @nb.njit(nogil=True, cache=False)
    def tails(flat, thr_lo, thr_hi, low, high):
        """One pass over `flat` (float32, 1D): the values <= `thr_lo` and >= `thr_hi` copied into
        `low` and `high` in the order they appear, like `flat[flat <= thr_lo]` and
        `flat[flat >= thr_hi]`; returns how many (even beyond the capacity: the caller checks).
        NaNs are excluded beforehand by the caller (`flat.max()`), as in the NumPy path.
        """
        n_lo = 0
        n_hi = 0
        cap_lo = low.size
        cap_hi = high.size
        for i in range(flat.size):
            v = flat[i]
            if v <= thr_lo:
                if n_lo < cap_lo:
                    low[n_lo] = v
                n_lo += 1
            if v >= thr_hi:
                if n_hi < cap_hi:
                    high[n_hi] = v
                n_hi += 1
        return n_lo, n_hi

    @nb.njit(nogil=True, cache=False)
    def normalize(stack, lo, scale, out):
        """out[k, i, j] = uint8(clip((float64(stack) - lo) * scale, 0, 255)), like
        `np.subtract` + `np.multiply` + `np.clip` + `copyto(casting="unsafe")` in float64.
        """
        kk, hh, ww = stack.shape
        for k in range(kk):
            for i in range(hh):
                for j in range(ww):
                    v = (np.float64(stack[k, i, j]) - lo) * scale
                    if v < 0.0:
                        v = 0.0
                    elif v > 255.0:
                        v = 255.0
                    out[k, i, j] = np.uint8(v)

    _STATE["tails"] = tails
    _STATE["normalize"] = normalize


def warmup() -> dict:
    """Imports numba and compiles the kernels for the shapes needed (contiguous stack, contiguous
    or flipped output). Once per process; returns the state.
    """
    if _STATE["tried"]:
        return {k: _STATE[k] for k in ("ok", "error", "compile_s")}
    _STATE["tried"] = True
    t = time.perf_counter()
    try:
        _build()
        s = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        buf = np.empty(8, np.float32)
        _STATE["tails"](s.reshape(-1), np.float32(1.0), np.float32(20.0), buf, buf.copy())
        o = np.empty(s.shape, np.uint8)
        _STATE["normalize"](s, np.float64(1.0), 2.0, o)
        _STATE["normalize"](s, np.float64(1.0), 2.0, o[:, :, ::-1])
        _STATE["normalize"](s, np.float64(1.0), 2.0, o[::-1])
        _STATE["ok"] = True
    except Exception as exc:                  # numba missing or not compiling
        _STATE["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        _STATE["ok"] = False
    _STATE["compile_s"] = round(time.perf_counter() - t, 2)
    return {k: _STATE[k] for k in ("ok", "error", "compile_s")}


def ok() -> bool:
    """Whether the kernels are compiled (without compiling: `warmup` is called from outside)."""
    return bool(_STATE["ok"])


def tails(flat, thr_lo, thr_hi, low, high):
    return _STATE["tails"](flat, thr_lo, thr_hi, low, high)


def normalize(stack, lo, scale, out) -> None:
    _STATE["normalize"](stack, lo, scale, out)
