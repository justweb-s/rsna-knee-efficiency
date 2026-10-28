"""The 58 gold labels: the only yardstick independent of the labels derived from the reports.

58 studies are a short ruler: macro differences below ~0.02 are not measurable on it. Use it
to rank very different candidates, not to choose between close variants.
"""
from __future__ import annotations

import pandas as pd

from ..constants import ID, RAW, TARGETS
from ..metrics import evaluate


def load_gold(path=None) -> pd.DataFrame:
    """The training studies with all 12 labels annotated by radiologists. Indexed by StudyInstanceUID."""
    tr = pd.read_csv(path or RAW / "train.csv").set_index(ID)
    return tr[TARGETS][tr[TARGETS].notna().all(axis=1)].astype(int)


def load_reports(path=None) -> pd.Series:
    tr = pd.read_csv(path or RAW / "train.csv").set_index(ID)
    return tr["Report"].fillna("")


def score_label_set(labels: pd.DataFrame, gold: pd.DataFrame | None = None) -> dict:
    """How well a set of report-derived labels reproduces the gold labels."""
    gold = load_gold() if gold is None else gold
    common = gold.index.intersection(labels.index)
    res = evaluate(gold.loc[common], labels.loc[common][TARGETS])
    res["coverage"] = float(len(common) / len(gold))
    return res
