"""Read-ahead of the DICOM files into the page cache, with many threads, ahead of the workers.

The competition data mount on Kaggle is NFS, and reading is bound by per-file latency rather
than bandwidth (cold page cache): ~61 MB/s with one thread, ~250 with 4, ~400 with 8, ~630 with
32, ~730 with 64. The page cache, on the other hand, works: re-read, the same files come at
~470 MB/s with one thread and ~1.8 GB/s with 4.

Here one thread drives a pool of `threads` that reads in full (and discards) the files of the
selected series of the studies the workers will open next, at most `ahead` studies past the last
one consumed: the workers then find the bytes in memory. It changes neither what is read nor the
pixels, only when they arrive from the mount. `progress[0]` = studies consumed (updated by the
consumer).
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_CHUNK = 4 << 20


def _drain(path: str, local: threading.local) -> int:
    buf = getattr(local, "buf", None)
    if buf is None:
        buf = local.buf = bytearray(_CHUNK)
    n = 0
    try:
        with open(path, "rb", buffering=0) as fh:
            while True:
                k = fh.readinto(buf)
                if not k:
                    break
                n += k
    except OSError:
        pass
    return n


def start(root: Path, series_dir: str, uids: list[str], slots: dict, progress: list,
          threads: int = 32, ahead: int = 48, log=print) -> dict:
    """Starts in a daemon thread and returns the statistics (updated live)."""
    stats = {"threads": threads, "ahead": ahead, "studies": 0, "files": 0, "mb": 0.0,
             "wait_s": 0.0, "read_s": 0.0}
    local = threading.local()

    def run() -> None:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(threads, thread_name_prefix="knee-readahead") as ex:
            pending: list = []
            for i, study in enumerate(uids):
                w = time.perf_counter()
                while i >= progress[0] + ahead:
                    time.sleep(0.02)
                stats["wait_s"] += time.perf_counter() - w
                files = []
                for uid in slots.get(study, {}).values():
                    d = os.path.join(str(root), series_dir, study, uid)
                    try:
                        files += [os.path.join(d, f) for f in os.listdir(d)]
                    except OSError:
                        pass
                pending.append([ex.submit(_drain, f, local) for f in files])
                stats["files"] += len(files)
                # no more than ~4 studies in flight in the pool: the order stays the workers' order
                while len(pending) > 4:
                    stats["mb"] += sum(f.result() for f in pending.pop(0)) / 2**20
                stats["studies"] = i + 1
            for fs in pending:
                stats["mb"] += sum(f.result() for f in fs) / 2**20
        stats["read_s"] = time.perf_counter() - t0
        log(f"readahead: {stats['studies']} studies, {stats['files']} files, {stats['mb']:.0f} MB "
            f"in {stats['read_s']:.1f} s (waiting {stats['wait_s']:.1f} s)")

    threading.Thread(target=run, name="knee-readahead", daemon=True).start()
    return stats
