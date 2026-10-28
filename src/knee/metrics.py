"""Metrics. The official metric is the macro AUC over the 12 targets.

NaN cells are ignored per column. A column without both classes has no defined AUC: it is
returned as NaN and excluded from the mean, and the number of columns actually evaluated is
reported in `n_classes`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import TARGETS


def auc_per_class(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    """AUC for each target, aligning the rows on the index."""
    y_true, y_pred = y_true.align(y_pred, join="inner", axis=0)
    out: dict[str, float] = {}
    for t in TARGETS:
        yt, yp = y_true[t], y_pred[t]
        m = yt.notna() & yp.notna()
        yt, yp = yt[m], yp[m]
        out[t] = float(roc_auc_score(yt, yp)) if yt.nunique() == 2 else float("nan")
    return out



def auc_within_groups(y_true: pd.DataFrame, y_pred: pd.DataFrame,
                      groups: pd.Series) -> dict[str, float]:
    """AUC per target counting only the positive/negative pairs within the SAME group.

    Per column: sum over the groups of the Mann-Whitney statistic (ties count 1/2) divided by the
    sum of n_pos * n_neg. A constant shift per group does not change it: it separates the ranking
    within a site from the per-site prior. Groups without both classes do not contribute; NaN if
    no group has them.
    """
    from scipy.stats import rankdata

    y_true, y_pred = y_true.align(y_pred, join="inner", axis=0)
    g = groups.reindex(y_true.index)
    out: dict[str, float] = {}
    for t in TARGETS:
        m = y_true[t].notna() & y_pred[t].notna() & g.notna()
        yt, yp, gg = y_true[t][m].values, y_pred[t][m].values, g[m].values
        num = den = 0.0
        for gv in np.unique(gg):
            sel = gg == gv
            y, s = yt[sel], yp[sel]
            npos = int((y == 1).sum())
            nneg = len(y) - npos
            if npos == 0 or nneg == 0:
                continue
            r = rankdata(s)
            num += r[y == 1].sum() - npos * (npos + 1) / 2
            den += npos * nneg
        out[t] = float(num / den) if den else float("nan")
    return out

def macro_auc(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """The official metric: mean of the per-class AUCs."""
    return float(np.nanmean(list(auc_per_class(y_true, y_pred).values())))


def evaluate(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    """Full report, in the form written to metrics.json."""
    per = auc_per_class(y_true, y_pred)
    vals = np.array(list(per.values()), dtype=float)
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner", axis=0)
    return {
        "macro_auc": float(np.nanmean(vals)),
        "auc_per_class": per,
        "n_classes": int(np.isfinite(vals).sum()),
        "n_studies": int(len(aligned_true)),
        "worst_class": min(
            (k for k in per if np.isfinite(per[k])), key=lambda k: per[k], default=None
        ),
    }
