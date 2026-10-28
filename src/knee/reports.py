"""The auxiliary target of the report-distillation experiment (not used by the final model): what
of the report does not fit into the 12 columns.

The median report has 129 words and we extract 12 numbers from it. This module takes a frozen
sentence embedding of the report and derives the vector the model must learn to predict from the
image, removing two things that would make the target useless or misleading:

1. **Language identity.** Measured on the embeddings: eta squared by language **0.2445**, a quarter
   of the variance. An MRI carries the signature of scanner and centre, so a model can guess the
   language without knowing anything about knees: if that direction stays in the target, the
   auxiliary head learns it first.
2. **What the 12 columns already say.** The part **linearly** explained by the teacher (the 12
   soft values plus the 12 missing-cell indicators) is projected out. What remains is orthogonal
   to the teacher by construction.

**Everything is estimated on the fold's training uids only.** The frozen embeddings contain only
what depends on the individual report; means, components and coefficients do not. A PCA estimated
on all 4,407 studies would leak the fold's validation studies into its training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ID, REPO, TARGETS


def load_report_embeddings(name: str, root: Path | str | None = None
                           ) -> tuple[pd.DataFrame, pd.Series, dict]:
    """The frozen embeddings: (studies x dim), the language code per study, the metadata."""
    base = Path(root) if root is not None else REPO / "data" / "reports"
    path = base / (name if name.endswith(".npz") else f"emb-{name}.npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"report embeddings not found: {path}. They are built with "
            "`python scripts/build_report_embeddings.py`.")
    with np.load(path, allow_pickle=True) as z:
        uids = [str(u) for u in z["uids"]]
        emb = np.asarray(z["emb"], dtype=np.float32)
        lang = [str(v) for v in z["lang"]]
        meta = json.loads(str(z["meta"]))
    idx = pd.Index(uids, name=ID)
    df = pd.DataFrame(emb, index=idx)
    if df.index.has_duplicates:
        raise ValueError(f"{path}: duplicate uids in the embeddings")
    return df, pd.Series(lang, index=idx, name="lang"), meta


def _design(labels: pd.DataFrame, uids: list[str], langs: pd.Series,
            lang_levels: list[str]) -> np.ndarray:
    """[1 | soft y (12) | missing cell (12) | language one-hot] -- what gets removed.

    The first two parts are what training already receives (the teacher and *where* the report is
    silent); the third is the measured confounder: eta squared by language 0.2445 on the raw
    embedding, which the first principal components do not concentrate (0.136 after removing four).
    A per-language mean removed by regression zeroes it by construction, for the part a mean can
    explain. Differences **between centres of the same language** remain: there is no centre id.
    """
    y = labels.reindex(uids)[TARGETS].to_numpy(dtype=np.float64)
    miss = ~np.isfinite(y)
    g = langs.reindex(uids).to_numpy()
    onehot = np.stack([(g == lev).astype(np.float64) for lev in lang_levels[1:]], axis=1)         if len(lang_levels) > 1 else np.zeros((len(uids), 0))
    return np.concatenate([np.ones((len(uids), 1)), np.where(miss, 0.0, y),
                           miss.astype(np.float64), onehot], axis=1)


@dataclass
class ReportTargets:
    """The transformation estimated on the training set of ONE fold. `transform` only applies it."""

    mean: np.ndarray          # (D,)  training centre
    nuisance: np.ndarray      # (D, drop_pcs)  the removed directions
    coef: np.ndarray          # (P, D)  the part explained by teacher + language
    lang_levels: list[str]    # the one-hot levels, in the order estimated on training
    proj: np.ndarray          # (D, dim)  the final PCA
    dim: int
    explained: float          # share of the residual variance kept by the `dim` components
    language_eta2: float      # eta squared by language AFTER the removal, if measurable

    def transform(self, emb: pd.DataFrame, labels: pd.DataFrame, langs: pd.Series,
                  uids: list[str]) -> np.ndarray:
        x = emb.reindex(uids).to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            missing = [u for u in uids if u not in emb.index]
            raise KeyError(f"{len(missing)} studies without report embedding, the first is "
                           f"{missing[0] if missing else '?'}")
        x = x - self.mean
        x = x - (x @ self.nuisance) @ self.nuisance.T
        x = x - _design(labels, uids, langs, self.lang_levels) @ self.coef
        z = x @ self.proj
        n = np.linalg.norm(z, axis=1, keepdims=True)
        # A null residual does not occur in the corpus, but a zero-norm target would give a NaN cosine
        # that would poison the gradient of the whole batch.
        return (z / np.maximum(n, 1e-6)).astype(np.float32)


def fit_report_targets(emb: pd.DataFrame, labels: pd.DataFrame, langs: pd.Series,
                       train_uids: list[str], dim: int = 64, drop_pcs: int = 4) -> ReportTargets:
    """Estimates centre, nuisance directions, coefficients (teacher + language) and PCA **on training**."""
    x = emb.reindex(train_uids).to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise KeyError("missing embeddings for some training studies")
    mean = x.mean(axis=0)
    xc = x - mean

    if drop_pcs > 0:
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        nuisance = np.ascontiguousarray(vt[:drop_pcs].T)      # (D, drop_pcs)
        xc = xc - (xc @ nuisance) @ nuisance.T
    else:
        nuisance = np.zeros((x.shape[1], 0))

    lang_levels = sorted(pd.unique(langs.reindex(train_uids).to_numpy()).tolist())
    d = _design(labels, train_uids, langs, lang_levels)
    coef, *_ = np.linalg.lstsq(d, xc, rcond=None)             # (25, D)
    resid = xc - d @ coef

    u, s, vt = np.linalg.svd(resid, full_matrices=False)
    if dim > len(s):
        raise ValueError(f"dim {dim} above the available rank ({len(s)})")
    proj = np.ascontiguousarray(vt[:dim].T)                   # (D, dim)
    explained = float((s[:dim] ** 2).sum() / max(float((s ** 2).sum()), 1e-12))

    z = resid @ proj
    zl = langs.reindex(train_uids).to_numpy()
    total = float(((z - z.mean(0)) ** 2).sum())
    between = sum(float((zl == g).sum()) * float(((z[zl == g].mean(0) - z.mean(0)) ** 2).sum())
                  for g in np.unique(zl))
    eta2 = between / max(total, 1e-12)

    return ReportTargets(mean=mean, nuisance=nuisance, coef=coef, lang_levels=lang_levels,
                         proj=proj, dim=int(dim), explained=explained, language_eta2=eta2)
