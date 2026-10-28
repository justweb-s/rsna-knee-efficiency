"""Training rows from another cache (`data.extra`): always in training, never in validation (an
experimental option, not used by the final model).

A source is a cache with the SAME shape as the run's (same `CacheSpec`, same tag, only the folder
changes) whose study IDs carry the prefix `<name>.`, plus its targets, in one of two forms:

- `targets: <rule>`: computed from the columns of the cache index with a rule registered here
  (`RULES`: `mrnet-v1`). MRNet labels are not in the repository: their licence does not allow
  redistribution;
- `labels: <name>`: `data/labels/<name>.csv` (one table for all folds) or
  `data/labels/<name>/fold<k>.csv` (one per fold, like `train.teacher`), probabilities in [0, 1]
  or NaN (cell excluded).

`n_studies` (optional) is how many exams the source cache must have: the run stops if it finds a
different number. Without it, a partial or oversized copy would silently train on another corpus.

    data:
      extra:
        - name: mrnet
          targets: mrnet-v1

The caller passes the folders, `run(..., extra_roots={"mrnet": <dir>})`, as for the main cache:
where a cache lives depends on the machine and is not part of the config, hence not of the
`config_hash` either.

Targets have the 12 `TARGETS` columns and, if the source has it, `UNION`: MRNet's meniscus label
without side, which `train.masked_bce_union` reads as "at least one of the two menisci".
Validation and OOF see none of this: extra rows have no fold, and `train_fold` adds them only to
the training studies.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import ID, TARGETS
from .dataset import open_cache

UNION = "Meniscus (any)"
# The rules computing targets from the cache index: name -> "module:function", imported only when
# needed.
RULES = {"mrnet-v1": "knee.data.mrnet_rows:rule_mrnet_v1"}


def extra_specs(cfg: dict) -> list[dict]:
    """`data.extra` validated: a list of sources with `name` and `targets` or `labels`."""
    specs = (cfg.get("data") or {}).get("extra") or []
    if not isinstance(specs, list):
        raise ValueError("data.extra: expected a list of sources")
    seen = set()
    for s in specs:
        if not isinstance(s, dict) or not isinstance(s.get("name"), str) or not s["name"]:
            raise ValueError(f"data.extra: every source needs a `name`, got {s!r}")
        if "." in s["name"] or s["name"] in seen:
            raise ValueError(f"data.extra: name {s['name']!r} contains a dot or is repeated")
        seen.add(s["name"])
        unknown = set(s) - {"name", "targets", "labels", "n_studies"}
        if unknown:
            raise ValueError(f"data.extra.{s['name']}: unknown keys {sorted(unknown)}")
        if ("targets" in s) == ("labels" in s):
            raise ValueError(f"data.extra.{s['name']}: needs `targets` (rule) or `labels` "
                             "(file), exactly one")
        if "n_studies" in s and (not isinstance(s["n_studies"], int) or isinstance(s["n_studies"], bool)
                                 or s["n_studies"] <= 0):
            raise ValueError(f"data.extra.{s['name']}: n_studies must be an integer > 0")
        if "targets" in s and s["targets"] not in RULES:
            raise ValueError(f"data.extra.{s['name']}: unknown rule {s['targets']!r}, "
                             f"allowed {sorted(RULES)}")
    return specs


def has_extra(cfg: dict) -> bool:
    return bool(extra_specs(cfg))


def _ids(cache) -> list[str]:
    if hasattr(cache, "index"):
        return list(cache.index[ID])
    return list(cache.where)                       # ShardedStudyCache


def _index(cache) -> pd.DataFrame:
    if hasattr(cache, "index"):
        return cache.index
    return pd.concat([c.index for c in cache.parts], ignore_index=True)


def open_extras(cfg: dict, roots: dict | None, spec) -> dict:
    """One cache per source, with the run's spec; every ID must have the prefix `<name>.`."""
    out = {}
    for s in extra_specs(cfg):
        name = s["name"]
        root = (roots or {}).get(name)
        if root is None:
            raise ValueError(f"data.extra: missing folder for source {name!r} "
                             f"(run(extra_roots=...), --extra {name}=<dir>)")
        cache = open_cache(root, spec)
        n = len(_ids(cache))
        if "n_studies" in s and n != s["n_studies"]:
            raise ValueError(f"data.extra.{name}: the cache has {n} exams, the config declares "
                             f"{s['n_studies']} (n_studies)")
        bad = [u for u in _ids(cache) if not str(u).startswith(f"{name}.")]
        if bad:
            raise ValueError(f"data.extra.{name}: {len(bad)} IDs without the prefix '{name}.', "
                             f"the first is {bad[0]}")
        out[name] = cache
    return out


def _rule(name: str):
    mod, fn = RULES[name].split(":")
    return getattr(importlib.import_module(mod), fn)


def _label_tables(labels_dir, name: str, folds: list[int]) -> dict[int, pd.DataFrame]:
    per_fold = Path(labels_dir) / name
    paths = ({k: per_fold / f"fold{k}.csv" for k in folds} if per_fold.is_dir()
             else {k: Path(labels_dir) / f"{name}.csv" for k in folds})
    out = {}
    for k, p in paths.items():
        if not p.is_file():
            raise ValueError(f"data.extra: missing {p}")
        out[k] = pd.read_csv(p, index_col=0)
    return out


def extra_targets(cfg: dict, caches: dict, labels_dir, folds: list[int]) -> dict[int, pd.DataFrame]:
    """The targets of the extra rows per fold: `TARGETS` (+ `UNION` if present), indexed by ID.

    Every ID of the cache must have a target and at least one finite cell (a row with nothing to
    learn would only cost time); values are in [0, 1].
    """
    parts: dict[int, list[pd.DataFrame]] = {k: [] for k in folds}
    for s in extra_specs(cfg):
        name = s["name"]
        cache = caches[name]
        ids = _ids(cache)
        if "targets" in s:
            t = _rule(s["targets"])(_index(cache))
            tables = {k: t for k in folds}
        else:
            tables = _label_tables(labels_dir, s["labels"], folds)
        for k, t in tables.items():
            cols = TARGETS + ([UNION] if UNION in t.columns else [])
            missing = pd.Index(ids).difference(t.index)
            if len(missing):
                raise ValueError(f"data.extra.{name}: {len(missing)} studies without targets "
                                 f"(fold {k}), the first is {missing[0]}")
            t = t.loc[ids, cols].astype(np.float32)
            v = t.to_numpy()
            fin = np.isfinite(v)
            if ((v[fin] < 0) | (v[fin] > 1)).any():
                raise ValueError(f"data.extra.{name}: targets outside [0, 1] (fold {k})")
            if not fin.any(axis=1).all():
                raise ValueError(f"data.extra.{name}: {int((~fin.any(axis=1)).sum())} studies "
                                 f"without any target (fold {k})")
            parts[k].append(t)
    return {k: pd.concat(v) for k, v in parts.items()}


class CombinedCache:
    """The run's cache plus those of the extra rows, seen as one.

    `KneeStudies` only needs `spec`, `in` and `get`; an ID lives in exactly one cache (extra rows carry
    the source prefix, and `train_fold` rejects collisions).
    """

    def __init__(self, main, extras: dict):
        self.main, self.extras = main, dict(extras)
        self.spec = main.spec
        for name, c in self.extras.items():
            if tuple(c.spec.shape) != tuple(main.spec.shape):
                raise ValueError(f"data.extra.{name}: shape {c.spec.shape}, the run wants "
                                 f"{main.spec.shape}")

    def __len__(self) -> int:
        return len(self.main) + sum(len(c) for c in self.extras.values())

    def __contains__(self, uid: str) -> bool:
        return uid in self.main or any(uid in c for c in self.extras.values())

    def get(self, uid: str):
        if uid in self.main:
            return self.main.get(uid)
        for c in self.extras.values():
            if uid in c:
                return c.get(uid)
        raise KeyError(uid)

    def describe(self) -> str:
        return ", ".join([f"cache {(len(self.main),) + tuple(self.spec.shape)}"]
                         + [f"extra {n} {len(c)} studies" for n, c in self.extras.items()])
