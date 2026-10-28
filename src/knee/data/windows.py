"""Read from disk only the windows training uses, not the whole study.

The training loop samples `k_train` windows per study (e.g. 24 of the 88 with 24 slices and
stride 2), but a naive DataLoader would read the WHOLE study from the memmap and crop the windows
on the GPU: most of the bytes read would be thrown away.

Here sampling moves BEFORE reading: the batch sampler decides the windows, the dataset reads only
those (a window is a contiguous block `vol[slot, s:s+depth]`).

What it does NOT buy: time. An epoch costs as much as the encoder compute on the windows, which
scales exactly with the pixels; training is GPU-bound, not disk-bound. The loader stays because it
is correct, bit-identical and removes most of the I/O.

The point that matters: **training stays bit-identical**. In the plain loop the shuffle uses
`g_tr` (`seed*1000 + fold`, explicit in the DataLoader) and the window draw uses `gen`
(`seed + fold`), consumed ONCE per step with `randperm(n_windows)` and equal for all the studies of
the batch. The sampler consumes the same two generators in the same order: the same per-epoch
`randperm` of the studies (what `RandomSampler` does) and the same per-batch `randperm` of the
windows. The DataLoader still receives `g_tr` as `generator`, because it draws the worker seed
from it when the iterator is created (before the first `randperm`), exactly as before. Same
studies, same windows, same order, same values (uint8 -> float32 / 255 is exact); the plain path
stays in `train.py` for the case where no sampling happens (`k_train` >= windows per study).
"""
from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from ..models.baseline import GROUP, windows_per_slot


def window_span(i: int, n_slices: int, stride: int = 1, depth: int = GROUP) -> tuple[int, int]:
    """Window `i` (slot-major order of `make_windows`) -> (slot, first slice).

    `make_windows` is `unfold(2, depth, stride)`: window `j` of a slot covers slices
    `j*stride .. j*stride+depth-1`, and the global index is `slot * per + j`.
    """
    per = windows_per_slot(n_slices, stride, depth)
    return i // per, (i % per) * stride


def use_window_loader(k: int, n_windows: int) -> bool:
    """Windows are read one by one only when `_sample_windows` would actually sample.

    With `k <= 0` or `k >= n_windows` the plain loop does not consume `gen` and uses all windows:
    reading them one by one would cost more than the whole study (at stride 2 they overlap), so the
    old path stays.
    """
    return 0 < k < n_windows


WINDOW_SAMPLING = ("uniform", "stratified")


def stratified_pick(n_slots: int, per: int, k: int, g: torch.Generator) -> tuple[int, ...]:
    """Stratified sampling (an experiment): k windows spread across the slots and, inside a slot, along
    the stack.

    Every slot gets `k // n_slots` windows, and `k % n_slots` randomly drawn slots get one more
    (12 on 8 slots: all at least one, four get two). The `c` windows of a slot lie in `c` contiguous
    bands of equal width over the slot's `per` windows, one per band, at random inside the band. The
    final order is random as in the uniform draw, so the positional rule of the context stays the
    same: the only thing that changes is WHICH windows are seen. The uniform draw (`randperm(n)[:k]`)
    leaves out a given slot in 17.9% of the batches and a whole sequence (both scales) in 2.4%, at 12
    windows out of 8 x 11.
    """
    if not 0 < k <= n_slots * per:
        raise ValueError(f"k={k} over {n_slots} slots x {per} windows")
    base, extra = divmod(k, n_slots)
    counts = [base] * n_slots
    for s in torch.randperm(n_slots, generator=g)[:extra].tolist():
        counts[s] += 1
    if max(counts) > per:
        raise ValueError(f"k={k}: more than {per} windows in one slot")
    picks = []
    for s, c in enumerate(counts):
        edges = [round(i * per / c) for i in range(c + 1)] if c else []
        for lo, hi in zip(edges, edges[1:]):
            picks.append(s * per + lo + int(torch.randint(hi - lo, (1,), generator=g)))
    order = torch.randperm(len(picks), generator=g).tolist()
    return tuple(picks[i] for i in order)


class WindowBatchSampler(Sampler[list[tuple[int, tuple[int, ...]]]]):
    """Batches of `(study index, windows)`, with the same draws as the plain loop.

    Per epoch: `randperm(n, generator=g_shuffle)` over the studies (like `RandomSampler`), in blocks
    of `batch_size`, without the last incomplete block (`drop_last=True`). Per batch:
    `randperm(n_windows, generator=g_pick)[:k]`, the same `pick` for all studies of the batch (like
    `_sample_windows`). Both draws happen inside `__iter__`, i.e. at the first `next` and not at
    creation: the DataLoader draws the worker seed from `g_shuffle` first, and the consumption order
    is unchanged.
    """

    def __init__(self, n: int, batch_size: int, n_windows: int, k: int,
                 g_shuffle: torch.Generator, g_pick: torch.Generator,
                 sampling: str = "uniform", per: int | None = None):
        if sampling not in WINDOW_SAMPLING:
            raise ValueError(f"window_sampling: expected one of {WINDOW_SAMPLING}, got {sampling!r}")
        if sampling == "stratified" and (not per or n_windows % per):
            raise ValueError(f"stratified: {n_windows} windows do not split into slots of {per}")
        self.sampling, self.per = sampling, per
        if not use_window_loader(k, n_windows):
            raise ValueError(f"k={k} of {n_windows} windows: no sampling, use the plain DataLoader")
        if batch_size < 1 or n < batch_size:
            raise ValueError(f"batch_size {batch_size} over {n} studies")
        self.n, self.batch_size, self.n_windows, self.k = n, batch_size, n_windows, k
        self.g_shuffle, self.g_pick = g_shuffle, g_pick

    def __len__(self) -> int:
        return self.n // self.batch_size

    def __iter__(self) -> Iterator[list[tuple[int, tuple[int, ...]]]]:
        order = torch.randperm(self.n, generator=self.g_shuffle).tolist()
        for b in range(len(self)):
            studies = order[b * self.batch_size:(b + 1) * self.batch_size]
            if self.sampling == "stratified":
                pick = stratified_pick(self.n_windows // self.per, self.per, self.k, self.g_pick)
            else:
                pick = tuple(torch.randperm(self.n_windows, generator=self.g_pick)[:self.k].tolist())
            yield [(s, pick) for s in studies]
        # `RandomSampler.__iter__` (torch 2.x) ends every epoch with a second `randperm(n)[:num_samples % n]`,
        # empty but drawn anyway, and `BatchSampler` with `drop_last` always reaches it because it pulls
        # until StopIteration. Without this line the second epoch would have a different order from the
        # plain loop.
        torch.randperm(self.n, generator=self.g_shuffle)
