"""Access to the cache built on Kaggle and a PyTorch Dataset over the studies.

The cache is a single `.npy` (tens of GB at 512 px): it is opened as a memmap and never loaded
whole. Training does not read whole studies but only the sampled windows (`windows.py`,
`KneeStudies._windows`): same tensors, a fraction of the bytes.

A single `.npy` does not fit in the output of a Kaggle kernel (~20 GB), so the cache is built in
**shards** (`cache_shards: N` in the config, one kernel per shard). On a training machine with
enough disk the shards can be merged into one file; otherwise `StudyCache` also accepts N folders
and reads them as they are: one memmap per shard, addressed by global row. The bytes are the same
in both cases (merging is a copy), so a run reading the shards is comparable with one reading the
merged file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..constants import ID, TARGETS
from .cache import CacheSpec


class ShardedVols:
    """`vols` of a sharded cache: one memmap per shard, read by global row.

    It exposes what the rest of the code uses of a memmap (`shape`, `len`, `[row]`) and nothing else:
    slicing across shards would cost a silent copy, and nobody needs it. The shards arrive already
    sorted by row and contiguous (checked by `StudyCache`), so the shard of a row is found with a
    binary search on the starts.
    """

    def __init__(self, parts: list[np.ndarray], starts: list[int]):
        self.parts, self.starts = parts, np.asarray(starts)
        n = int(self.starts[-1]) + len(parts[-1])
        self.shape = (n,) + tuple(parts[0].shape[1:])

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, r: int) -> np.ndarray:
        r = int(r)
        if not 0 <= r < len(self):
            raise IndexError(f"row {r} outside 0..{len(self) - 1}")
        i = int(np.searchsorted(self.starts, r, side="right") - 1)
        return self.parts[i][r - int(self.starts[i])]


class StudyCache:
    """The three files written by the cache-building kernel, kept together and aligned by row.

    `root` is a folder (cache in a single file) or a list of folders (one per shard, in any order).
    The shards must cover rows 0..N-1 with no gaps or duplicates: a cache with rows in another order
    would silently train on the pixels of the wrong study.
    """

    def __init__(self, root: str | Path | list | tuple, spec: CacheSpec | None = None):
        roots = [Path(r) for r in root] if isinstance(root, (list, tuple)) else [Path(root)]
        if not roots:
            raise ValueError("no cache folder")
        self.root, self.roots = roots[0], roots
        self.spec = spec or CacheSpec()
        tag = self.spec.tag
        if len(roots) == 1:
            self.vols = np.load(self.root / f"vols_{tag}.npy", mmap_mode="r")
            self.mask = np.load(self.root / f"mask_{tag}.npy")
            self.index = pd.read_csv(self.root / f"index_{tag}.csv")
        else:
            self.vols, self.mask, self.index = self._open_shards(roots, tag)
        self.row_of = dict(zip(self.index[ID], self.index["row"]))
        if tuple(self.vols.shape[1:]) != tuple(self.spec.shape):
            raise ValueError(f"cache {self.vols.shape[1:]} does not match the spec {self.spec.shape}")
        if len(self.vols) != len(self.index):
            raise ValueError(f"{len(self.vols)} volumes but {len(self.index)} index rows")

    @staticmethod
    def _open_shards(roots: list[Path], tag: str):
        parts = [np.load(r / f"vols_{tag}.npy", mmap_mode="r") for r in roots]
        idx = [pd.read_csv(r / f"index_{tag}.csv") for r in roots]
        for r, v, df in zip(roots, parts, idx):
            if len(v) != len(df):
                raise ValueError(f"shard {r}: index of {len(df)} rows, array of {len(v)}")
            if len(df) == 0 or not (np.diff(df["row"].to_numpy()) == 1).all():
                raise ValueError(f"shard {r}: the rows are not a contiguous block")
            if v.shape[1:] != parts[0].shape[1:]:
                raise ValueError(f"shard {r}: shape {v.shape[1:]} differs from {parts[0].shape[1:]}")
        order = sorted(range(len(roots)), key=lambda i: int(idx[i]["row"].iloc[0]))
        rows = np.concatenate([idx[i]["row"].to_numpy() for i in order])
        if not np.array_equal(rows, np.arange(len(rows))):
            raise ValueError(f"the {len(roots)} shards do not cover 0..{len(rows) - 1} without gaps or duplicates")
        starts = [int(idx[i]["row"].iloc[0]) for i in order]
        vols = ShardedVols([parts[i] for i in order], starts)
        mask = np.concatenate([np.load(roots[i] / f"mask_{tag}.npy") for i in order])
        index = pd.concat([idx[i] for i in order], ignore_index=True)
        return vols, mask, index

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, uid: str) -> bool:
        return uid in self.row_of

    def get(self, uid: str) -> tuple[np.ndarray, np.ndarray]:
        r = self.row_of[uid]
        return self.vols[r], self.mask[r]


class ShardedStudyCache:
    """A cache built in N shards (`cache_shards: N`), read without merging.

    Every shard is a valid cache on its own (same spec, fewer studies), and in its `index_*.csv` the
    `row` column is **global** (the row the study would have in the whole cache) while the array is
    local to the shard. On Kaggle the shards arrive as outputs of other kernels, mounted read-only,
    and they do not fit in `/kaggle/working`. Here one memmap per shard is kept and
    `uid -> (shard, local row)` is translated; training does not notice, because `KneeStudies` only
    needs `spec`, `__contains__` and `get`.

    The shards are checked to cover 0..N-1 with no gaps or duplicates, as merging does: a missing
    shard would give a training run on 2/3 of the corpus that goes all the way to the end.
    """

    def __init__(self, roots: list[str | Path], spec: CacheSpec | None = None):
        if not roots:
            raise ValueError("no cache shard")
        self.spec = spec or CacheSpec()
        self.parts = [StudyCache(r, self.spec) for r in roots]
        self.parts.sort(key=lambda c: int(c.index["row"].min()))
        self.where: dict[str, tuple[int, int]] = {}
        rows = []
        for i, c in enumerate(self.parts):
            r = c.index["row"].to_numpy()
            if len(r) > 1 and not (np.diff(r) == 1).all():
                raise ValueError(f"shard {c.root}: the rows are not contiguous")
            lo = int(r[0])
            for uid, row in zip(c.index[ID], r):
                self.where[uid] = (i, int(row) - lo)
            rows.append(r)
        rows = np.concatenate(rows)
        if not np.array_equal(rows, np.arange(len(rows))):
            raise ValueError(f"the {len(self.parts)} shards do not cover 0..{len(rows) - 1} "
                             "without gaps or duplicates")
        self.mask = np.concatenate([c.mask for c in self.parts])

    def __len__(self) -> int:
        return len(self.where)

    def __contains__(self, uid: str) -> bool:
        return uid in self.where

    def get(self, uid: str) -> tuple[np.ndarray, np.ndarray]:
        i, r = self.where[uid]
        c = self.parts[i]
        return c.vols[r], c.mask[r]


def open_cache(root, spec: CacheSpec | None = None):
    """`StudyCache` from one folder, `ShardedStudyCache` from a list of folders."""
    if isinstance(root, (list, tuple)):
        return ShardedStudyCache(list(root), spec) if len(root) > 1 else StudyCache(root[0], spec)
    return StudyCache(root, spec)


class KneeStudies(Dataset):
    """One study per item: uint8 volume, slot mask, soft targets.

    Targets stay **soft**: the teacher labels are probabilities, not zeros and ones, and rounding them
    would throw away exactly the part that tells a confident report from a hesitant one. Studies
    without labels (none in train, but the class is also used at inference) get NaN, which the loss
    ignores.

    Two ways to request an item. With an integer, the whole study (uint8 volume `(S, K, H, W)`): the
    validation and inference path, where all windows are needed. With a pair `(index, windows)`, as
    produced by `windows.WindowBatchSampler`, only those windows, read one by one from the memmap
    (`(k, depth, H, W)` uint8) plus their indices: the training path, which samples `k_train` windows
    and has no reason to read the others. The second needs `window_stride` and `window_depth`, the
    geometry of `make_windows`.
    """

    def __init__(self, cache: StudyCache, uids: list[str], labels: pd.DataFrame | None = None,
                 gold_uids: set[str] | None = None, gold_weight: float = 1.0,
                 cell_weights: pd.DataFrame | None = None,
                 window_stride: int | None = None, window_depth: int | None = None,
                 report: np.ndarray | None = None, aux: np.ndarray | None = None,
                 window_extra: int = 0, findings: np.ndarray | None = None):
        missing = [u for u in uids if u not in cache]
        if missing:
            raise KeyError(f"{len(missing)} studies are not in the cache, the first is {missing[0]}")
        self.cache, self.uids = cache, list(uids)
        self.window_stride, self.window_depth = window_stride, window_depth
        # `train.augment.slice_offset`: every window is read with `window_extra` extra slices at the end;
        # `augment.apply_augment` picks the triplet on the GPU. At 0 the read is the usual one.
        self.window_extra = int(window_extra)
        if self.window_extra < 0:
            raise ValueError(f"window_extra must be >= 0, got {window_extra}")
        self.y = (labels.reindex(self.uids)[TARGETS].to_numpy(dtype=np.float32)
                  if labels is not None else
                  np.full((len(self.uids), len(TARGETS)), np.nan, dtype=np.float32))
        # Report distillation (an experiment): the auxiliary target (residual of the report embedding,
        # `knee.reports`) travels **at the end of `y`** instead of as a separate tensor. A choice of form, not
        # substance: the arity of what `__getitem__` returns stays the same, so `predict` and both branches
        # of `train_fold` do not change, and the consumer slices `y[:, :12]` for the BCE and `y[:, 12:]` for
        # the cosine. It is populated only in the TRAINING dataset: validation does not need the auxiliary
        # target and must not see it.
        self.n_report = 0
        if report is not None:
            if len(report) != len(self.uids):
                raise ValueError(f"report: {len(report)} rows for {len(self.uids)} studies")
            self.n_report = int(report.shape[1])
            self.y = np.concatenate([self.y, report.astype(np.float32)], axis=1)
        # `data.extra`: same form, other columns at the end (the meniscus target without side and the
        # "extra row" flag used by `train_fold`). Never together with the report: whoever slices the tail
        # would not know where one ends and the other starts.
        self.n_aux = 0
        if aux is not None:
            if report is not None:
                raise ValueError("KneeStudies: aux and report together, the tail of y would be ambiguous")
            if len(aux) != len(self.uids):
                raise ValueError(f"aux: {len(aux)} rows for {len(self.uids)} studies")
            self.n_aux = int(aux.shape[1])
            self.y = np.concatenate([self.y, aux.astype(np.float32)], axis=1)
        # Auxiliary findings from the report (NaN where the report is silent), same form. Also alone at the
        # end, for the same reason.
        self.n_findings = 0
        if findings is not None:
            if report is not None or aux is not None:
                raise ValueError("KneeStudies: findings with report or aux, the tail of y would be ambiguous")
            if len(findings) != len(self.uids):
                raise ValueError(f"findings: {len(findings)} rows for {len(self.uids)} studies")
            self.n_findings = int(findings.shape[1])
            self.y = np.concatenate([self.y, findings.astype(np.float32)], axis=1)
        if gold_uids and gold_weight != 1.0:
            gw = float(gold_weight)
            self.weights = np.array([gw if u in gold_uids else 1.0 for u in self.uids], dtype=np.float32)
        else:
            self.weights = np.ones(len(self.uids), dtype=np.float32)
        if cell_weights is not None:
            # Weight per CELL (studies x 12), not per study: cells where the report is silent can weigh less
            # than an explicit "NO". It multiplies the study weight, so the two axes compose, and `masked_bce`
            # receives it already two-dimensional. A study without a row in the matrix weighs 1 everywhere: the
            # mask says where to remove, not where to add.
            cw = cell_weights.reindex(self.uids)[TARGETS].to_numpy(dtype=np.float32)
            cw = np.where(np.isfinite(cw), cw, 1.0).astype(np.float32)
            self.weights = cw * self.weights[:, None]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i):
        if isinstance(i, tuple):
            return self._windows(*i)
        vol, mask = self.cache.get(self.uids[i])
        # `np.array` copies: the memmap is read-only and `torch.from_numpy` on a non-writable array warns
        # at every batch.
        return (torch.from_numpy(np.array(vol)),
                torch.from_numpy(np.array(mask)),
                torch.from_numpy(self.y[i]),
                torch.tensor(self.weights[i], dtype=torch.float32))

    def _windows(self, i: int, pick):
        """The windows `pick` of study `i`, read from the memmap one contiguous block each.

        They are the same bytes `make_windows` would crop from the whole study (`unfold` over the slices,
        slot-major order): `vol[slot, s:s+depth]` for every window, in `pick` order. With
        `window_extra` > 0 the block is `vol[slot, s:s+depth+window_extra]`, and the first `depth`
        slices are unchanged.
        """
        if self.window_stride is None or self.window_depth is None:
            raise ValueError("window reading without window_stride/window_depth: "
                             "KneeStudies needs the geometry of make_windows")
        from .windows import window_span
        vol, mask = self.cache.get(self.uids[i])
        n_slices = vol.shape[1]
        span = self.window_depth + self.window_extra
        out = np.empty((len(pick), span) + tuple(vol.shape[2:]), dtype=vol.dtype)
        for j, w in enumerate(pick):
            slot, s = window_span(int(w), n_slices, self.window_stride, self.window_depth)
            if s + span > n_slices:
                raise ValueError(f"window {int(w)}: slices {s}-{s + span - 1} beyond the stack of "
                                 f"{n_slices} (window_extra {self.window_extra})")
            out[j] = vol[slot, s:s + span]
        return (torch.from_numpy(out),
                torch.from_numpy(np.array(mask)),
                torch.from_numpy(self.y[i]),
                torch.tensor(self.weights[i], dtype=torch.float32),
                torch.tensor(pick, dtype=torch.int64))
