"""Finding things inside /kaggle/input without guessing the path.

Kaggle does not guarantee how deep an input is mounted. It may mount
`/kaggle/input/competitions/<slug>/` and `/kaggle/input/datasets/<owner>/<slug>/`, two levels
below the `/kaggle/input/<slug>/` seen in most public notebooks.

So no path is ever written: it is searched for. With one rule, which is what makes the search
practical: **never descend into the DICOM folders**. `train_series` holds ~819,000 files and
walking it to look for a `.npy` would cost more than the whole preprocessing.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

INPUT = Path("/kaggle/input")
# The DICOM folders: recognised by name and never walked.
DICOM_DIRS = {"train_series", "test_series"}
MAX_DEPTH = 4

# Under /kaggle/input nothing changes during a run (read-only mount): the tree is walked once and
# the content of every folder is remembered. Otherwise every `find_*` would redo the whole walk with
# one `stat` per child, about ten times at kernel start; with slow NFS metadata that was tens of
# seconds before the workers start. Other roots (test, temporary folders) are not cached: they may
# change between calls.
CACHED_ROOTS = {INPUT}
_LISTINGS: dict[str, dict[str, tuple[bool, bool]] | None] = {}
_WALKS: dict[tuple[str, int], list[Path]] = {}
# Walks in progress in a thread (`prewalk`): whoever asks for the same walk waits for it instead of
# redoing it.
_PREWALKS: dict[tuple[str, int], threading.Event] = {}


def _cached(d: Path) -> bool:
    d = Path(d)
    return any(d == r or r in d.parents for r in CACHED_ROOTS)


def _listing(d: Path) -> dict[str, tuple[bool, bool]] | None:
    """{name: (is_file, is_dir)} of the children of `d` (links followed, like `Path.is_file`), from a
    single read of the folder; None if it cannot be read.
    """
    key = str(d)
    if key not in _LISTINGS:
        try:
            with os.scandir(d) as it:
                _LISTINGS[key] = {e.name: (e.is_file(), e.is_dir()) for e in it}
        except OSError:
            _LISTINGS[key] = None
    return _LISTINGS[key]


def _kind(d: Path, name: str) -> tuple[bool, bool]:
    """(is_file, is_dir) of `d / name`: from the cached folder if `d` is under a cached root and `name`
    is a plain name, otherwise with `stat` as usual.
    """
    if "/" not in name and _cached(d):
        lst = _listing(d)
        if lst is not None:
            return lst.get(name, (False, False))
    p = d / name
    return p.is_file(), p.is_dir()


def _walk_cached(root: Path, max_depth: int) -> list[Path]:
    key = (str(root), max_depth)
    ev = _PREWALKS.get(key)
    if ev is not None:
        ev.wait()
    if key not in _WALKS:
        out: list[Path] = []
        if root.is_dir():
            frontier = [(root, 0)]
            while frontier:
                d, depth = frontier.pop(0)
                out.append(d)
                if depth >= max_depth:
                    continue
                lst = _listing(d)
                if lst is None:
                    continue
                kids = sorted(d / n for n, (_f, is_dir) in lst.items() if is_dir)
                frontier += [(k, depth + 1) for k in kids if k.name not in DICOM_DIRS]
        _WALKS[key] = out
    return _WALKS[key]


def prewalk(root: Path = INPUT, max_depth: int = MAX_DEPTH, threads: int = 16) -> list[Path]:
    """The walk of `_walk_cached(root, max_depth)` with the folders of each level read in parallel, to be
    started in a thread when the kernel starts: same order (breadth-first, sorted children) and same
    folders. The folders of the last level are read too, which `find_dir_with` would otherwise read
    one by one.

    On cold Kaggle datasets every folder can cost up to ~1.4 s (~5 s of sequential walking before
    the workers); in parallel the latencies overlap. Whoever asks for the same walk while this one is
    running waits for it (`_walk_cached`).
    """
    from concurrent.futures import ThreadPoolExecutor

    key = (str(root), max_depth)
    if key in _WALKS or key in _PREWALKS:
        return _walk_cached(Path(root), max_depth)
    ev = _PREWALKS[key] = threading.Event()
    try:
        root = Path(root)
        out: list[Path] = []
        if root.is_dir():
            level, depth = [root], 0
            with ThreadPoolExecutor(threads, thread_name_prefix="knee-prewalk") as ex:
                while level:
                    out += level
                    lists = list(ex.map(_listing, level))
                    if depth >= max_depth:
                        break
                    nxt: list[Path] = []
                    for d, lst in zip(level, lists):
                        if lst is None:
                            continue
                        kids = sorted(d / n for n, (_f, is_dir) in lst.items() if is_dir)
                        nxt += [k for k in kids if k.name not in DICOM_DIRS]
                    level, depth = nxt, depth + 1
        _WALKS[key] = out
    finally:
        ev.set()
    return _WALKS[key]


def walk_inputs(root: Path = INPUT, max_depth: int = MAX_DEPTH):
    """The directories under `root`, breadth-first, skipping the DICOM mounts."""
    if _cached(root):
        yield from _walk_cached(Path(root), max_depth)
        return
    if not root.is_dir():
        return
    frontier = [(root, 0)]
    while frontier:
        d, depth = frontier.pop(0)
        yield d
        if depth >= max_depth:
            continue
        try:
            kids = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            continue
        frontier += [(k, depth + 1) for k in kids if k.name not in DICOM_DIRS]


def find_dir_with(name: str, root: Path = INPUT, extra: list[Path] | None = None) -> Path | None:
    """The first directory containing `name` (a relative path, possibly nested)."""
    for d in list(walk_inputs(root)) + list(extra or []):
        if any(_kind(d, name)):
            return d
    return None


def find_dirs_with(name: str, root: Path = INPUT, extra: list[Path] | None = None) -> list[Path]:
    """All directories containing `name`: the shards of a cache are N mounts."""
    return [d for d in list(walk_inputs(root)) + list(extra or [])
            if any(_kind(d, name))]


def find_competition_root(root: Path = INPUT, extra: list[Path] | None = None) -> Path:
    """The folder with `train.csv` and `train_series/`.

    Both are required: `train.csv` alone can also be found in a derived dataset, and taking that one
    would mean looking for the DICOMs where they are not.
    """
    if extra is None:
        extra = [Path("data/raw"), Path(".")]
    for d in list(walk_inputs(root)) + list(extra):
        if _kind(d, "train.csv")[0] and _kind(d, "train_series")[1]:
            return d
    raise FileNotFoundError("competition mount not found: "
                            f"no folder with train.csv and train_series/ under {root}")


def find_train_csv(root: Path = INPUT, extra: list[Path] | None = None) -> Path | None:
    """Only the CSV, for the 58 gold studies: no DICOMs needed here."""
    d = find_dir_with("train.csv", root=root,
                      extra=[Path("data/raw")] if extra is None else extra)
    return d / "train.csv" if d else None


def describe_inputs(root: Path = INPUT, max_depth: int = 2) -> str:
    """What is actually mounted. Printed when a search fails: an error that does not show where it looked
    forces exactly the trial-and-error this module exists to avoid.
    """
    if not root.is_dir():
        return f"  ({root} does not exist)"
    return "\n".join(f"  {d}" for d in walk_inputs(root, max_depth)) or f"  ({root} vuoto)"


def find_test_root(root: Path = INPUT, extra: list[Path] | None = None) -> Path:
    """The folder with `test.csv` and `test_series/`, for the submission kernel.

    Same rule as `find_competition_root` and for the same reason: both are required, because
    `test.csv` alone can also be found in a derived dataset.
    """
    if extra is None:
        extra = [Path("data/raw"), Path(".")]
    for d in list(walk_inputs(root)) + list(extra):
        if _kind(d, "test.csv")[0] and _kind(d, "test_series")[1]:
            return d
    raise FileNotFoundError("competition mount not found: "
                            f"no folder with test.csv and test_series/ under {root}")
