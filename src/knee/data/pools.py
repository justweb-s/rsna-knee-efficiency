"""Thread pools for decoding: one per call, or shared within the process.

In the inference workers every study opens a pool for its series (`cache.build_study`), and every
series one for the headers used to order the slices (`dicom.order_series_files`, up to 16
threads) and one for the slices (`cache.build_series`, up to `SLICE_THREADS`): ~100 threads
created and destroyed per study. `MODE = "persistent"` keeps them alive per process, one per name,
with the same maximum concurrency as before when the four series of a study run together:
4 for the series, 16 x 4 for the headers, 10 x 4 for the slices. Tasks and result order (`map`)
do not change, so neither do the pixels.

Shared pools are created on first use *in the process that uses them*: after a `fork` (the
DataLoader workers) the child starts from scratch, because the parent's threads do not exist
in the child.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

# "persistent" by default: in an A/B on the T4 benchmark (with a warm-up pass, alternating
# order) steady state -5.7%, worker CPU -3%, system CPU -15%, identical predictions.
MODE = ["persistent"]
CAPACITY = {"study": 4, "order": 16 * 4, "slices": 10 * 4}
_LOCK = [threading.Lock()]
_SHARED: dict[str, ThreadPoolExecutor] = {}


def _reset_after_fork() -> None:
    _LOCK[0] = threading.Lock()
    _SHARED.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


@contextmanager
def thread_pool(name: str, n: int):
    """A pool for `n` tasks: new and closed on exit (`per_call`), or the process-wide shared one
    with capacity `CAPACITY[name]` (`persistent`).
    """
    if MODE[0] != "persistent":
        with ThreadPoolExecutor(max_workers=max(1, n)) as ex:
            yield ex
        return
    with _LOCK[0]:
        ex = _SHARED.get(name)
        if ex is None:
            ex = _SHARED[name] = ThreadPoolExecutor(max_workers=CAPACITY[name],
                                                   thread_name_prefix=f"knee-{name}")
    yield ex
