"""Single-fold screening: one fold to explore, five to confirm.

A full five-fold run costs hours of GPU; fold 0 alone costs a fifth, and over the history of the
project its delta against the parent tracks the five-fold OOF delta with a residual sd of 0.0026:
large losses are already visible there, half-sigma ones are not. The screening fold does not
decide a WIN (that stays with the five folds and the leaderboard); it only decides whether the
remaining folds are worth their cost.

Contents: the AUC of one fold against the run's teacher (same binarisation as
`compute_metrics`: > 0.5, NaN excluded), the verdict against the ruler pre-registered in the config
(`screen:`), and the reading of `screen.json` for whoever decides whether to continue. The parent's
ruler is computed locally and written into the config BEFORE the run: the training script does
not have the parent's OOF predictions and must not have them.

Next to the verdict there is a secondary ruler that decides nothing: the same fold restricted to
the English and Spanish reports, where the teacher is closest to the gold labels. It is recorded
to check prospectively whether it tracks the leaderboard better than the full OOF.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ID, TARGETS
from .metrics import auc_per_class
from .reports import load_report_embeddings
from .utils.config import load_config

SCREEN_FOLD = 0      # the screening fold, fixed: the first one `run()` trains
STOP_K = 2.0         # STOP at delta <= -STOP_K * delta_fold
SCREEN_FILE = "screen.json"
SECONDARY_LANGS = ("en", "es")   # secondary ruler
LANG_ARTIFACT = "minilm-l12-v2"  # the frozen report embeddings also carry the language


def binarize_teacher(labels: pd.DataFrame) -> pd.DataFrame:
    """Same binarisation as `compute_metrics`: > 0.5 is positive, NaN stays NaN."""
    y = (labels > 0.5).astype(float)
    y[labels.isna()] = np.nan
    return y


def fold_auc(oof: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series, fold: int) -> dict:
    """Macro AUC and per-class AUC on the studies of ONE fold, against the binarised teacher.

    `folds` is the `fold` Series indexed by study (`data/folds/folds.csv`). Studies of the fold that
    are not in `oof` (incomplete cache) are left out, as in `run()`.
    """
    ids = folds.index[folds == fold].intersection(oof.index)
    if len(ids) == 0:
        raise ValueError(f"no study of fold {fold} in these OOF predictions")
    y = binarize_teacher(labels.loc[ids, TARGETS])
    per = auc_per_class(y, oof.loc[ids, TARGETS])
    return {"fold": int(fold), "n_studies": int(len(ids)),
            "macro_auc": float(np.nanmean(list(per.values()))), "auc_per_class": per}


def screen_verdict(auc: float, parent_auc: float, delta_fold: float,
                   stop_k: float = STOP_K) -> dict:
    """STOP if the fold is `stop_k` δ_fold or more below the parent, otherwise PASS.

    The threshold is inclusive and single: screening has no NEUTRAL band and no WIN, because it is
    not a verdict on the axis, only the decision to spend the other folds.
    """
    if delta_fold <= 0:
        raise ValueError(f"delta_fold must be > 0, not {delta_fold}")
    d = float(auc - parent_auc)
    stop_below = float(parent_auc - stop_k * delta_fold)
    return {"delta": d, "k": d / delta_fold, "stop_below": stop_below,
            "verdict": "STOP" if auc <= stop_below else "PASS"}


def study_langs(root=None) -> pd.Series:
    """Report language per study, from the report-embeddings artifact.

    `root` is the `data/reports` folder; by default the repository's.
    """
    _, langs, _ = load_report_embeddings(LANG_ARTIFACT, root)
    return langs


def secondary_ruler(oof: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series, fold: int,
                    langs: pd.Series, parent_auc: float | None = None,
                    delta_fold: float | None = None, n_full: int | None = None) -> dict:
    """The screening fold on the en+es reports only: a secondary ruler that decides nothing.

    Same `fold_auc`, same fold studies, restricted to `SECONDARY_LANGS`. With `parent_auc` (the
    parent's fold on the same subset and against the same labels) it also gives the delta. δ_fold is
    rescaled to the size, δ_fold · √(n_full / n_en+es), because a subset of ~55% of the fold is
    noisier: a declared approximation.
    """
    keep = langs.index[langs.isin(SECONDARY_LANGS)]
    res = fold_auc(oof.loc[oof.index.intersection(keep)], labels, folds, fold)
    out = {"langs": list(SECONDARY_LANGS), "n_studies": res["n_studies"],
           "macro_auc": res["macro_auc"], "auc_per_class": res["auc_per_class"]}
    if parent_auc is not None:
        d = float(res["macro_auc"] - parent_auc)
        out.update({"parent_fold_auc": float(parent_auc), "delta": d})
        if delta_fold and n_full:
            scaled = float(delta_fold * np.sqrt(n_full / res["n_studies"]))
            out.update({"delta_fold_scaled": scaled, "k": d / scaled})
    return out


def metro_mismatch(child: pd.DataFrame, parent: pd.DataFrame) -> int:
    """How many cells change side between two label sets, binarised as in `fold_auc`.

    Zero = **the same yardstick**, and the parent's ruler is valid even if the teacher has another
    name. Non-zero = two yardsticks, and an AUC against one is not comparable with an AUC against the
    other.
    """
    ids = child.index.intersection(parent.index)
    a = binarize_teacher(child.loc[ids, TARGETS])
    b = binarize_teacher(parent.loc[ids, TARGETS])
    return int(((a != b) & ~(a.isna() & b.isna())).sum().sum())


def ruler_labels(cfg: dict, configs_dir=None) -> str | None:
    """Which label set `screen.parent_fold_auc` was measured on.

    `screen.parent_labels` if present, otherwise the parent's teacher, read from its config. `None` =
    cannot be determined (parent without a config at hand): the yardstick check is skipped and
    reported.
    """
    sc = cfg.get("screen", {})
    if sc.get("parent_labels"):
        return str(sc["parent_labels"])
    par = cfg.get("parent")
    if not par or configs_dir is None:
        return None
    p = Path(configs_dir) / f"{par}.yaml"
    if not p.is_file():
        return None
    return load_config(p).get("data", {}).get("labels")


def check_metro(cfg: dict, labels_path, configs_dir=None, log=print) -> dict:
    """The ruler and the screening fold must use the same yardstick.

    `screen_run` measures the fold against the run's `data.labels`. If `parent_fold_auc` was measured
    on another label set the comparison is meaningless (a run once inherited the parent's ruler on one
    teacher while being measured on another, a PASS by construction). Here it fails before writing
    `screen.json`, not after.
    """
    name = ruler_labels(cfg, configs_dir)
    metro = cfg["data"]["labels"]
    if name is None:
        log(f"screening: yardstick of the ruler cannot be determined (parent "
            f"{cfg.get('parent')}), check skipped")
        return {"parent_labels": None, "metro_cells_diff": None}
    if name == metro:
        return {"parent_labels": name, "metro_cells_diff": 0}
    d = Path(labels_path).parent
    diff = metro_mismatch(pd.read_csv(d / f"{metro}.csv", index_col=0),
                          pd.read_csv(d / f"{name}.csv", index_col=0))
    if diff:
        raise ValueError(
            f"{cfg.get('name')}: `screen.parent_fold_auc` was measured on `{name}` but the "
            f"fold is measured on `{metro}`, and the two label sets differ in {diff} cells "
            f"(the ruler must be the parent's fold ON THE SAME teacher). "
            f"Recompute it with `python scripts/screen_fold0.py {cfg.get('parent')} "
            f"--labels {metro}` and write it into the config with `screen.parent_labels: {metro}`.")
    log(f"screening: ruler on `{name}`, fold on `{metro}`, identical binarisation "
        f"(0 differing cells): same yardstick")
    return {"parent_labels": name, "metro_cells_diff": 0}


def screen_fold(cfg: dict) -> int:
    return int(cfg.get("screen", {}).get("fold", SCREEN_FOLD))


def _secondary(cfg: dict, oof: pd.DataFrame, labels: pd.DataFrame, folds: pd.Series,
               fold: int, n_full: int, langs: pd.Series | None, labels_path, log) -> dict | None:
    """The en+es ruler for `screen_run`. Any problem skips it and says so: the secondary ruler must
    never prevent the primary verdict.
    """
    sc = cfg["screen"]
    try:
        if langs is None:
            langs = study_langs(Path(labels_path).resolve().parents[1] / "reports")
        sec = secondary_ruler(oof, labels, folds, fold, langs,
                              parent_auc=sc.get("parent_fold_auc_enes"),
                              delta_fold=float(sc["delta_fold"]), n_full=n_full)
    except (FileNotFoundError, ValueError, KeyError) as e:
        log(f"screening: secondary en+es ruler skipped ({e})")
        return None
    head = (f"screening, secondary en+es ruler (does not decide): "
            f"{sec['macro_auc']:.4f} on {sec['n_studies']} studies")
    if "delta" not in sec:
        log(f"{head}; `screen.parent_fold_auc_enes` missing from the config, no delta")
    elif "k" in sec:
        log(f"{head} vs parent {sec['parent_fold_auc']:.4f} = {sec['delta']:+.4f} = "
            f"{sec['k']:+.1f} rescaled δ_fold ({sec['delta_fold_scaled']:.4f})")
    else:
        log(f"{head} vs parent {sec['parent_fold_auc']:.4f} = {sec['delta']:+.4f}")
    return sec


def screen_run(cfg: dict, out_dir, labels_path, folds_path, log=print,
               configs_dir=None, langs: pd.Series | None = None) -> dict:
    """Reads `oof_fold<k>.csv` in `out_dir`, applies the config's ruler, writes screen.json.

    The `screen:` block of the config carries `parent_fold_auc` (the parent on the same fold and the
    same teacher), `delta_fold` and, optionally, `fold`, `stop_k`, `parent_labels` and
    `parent_fold_auc_enes`. Without the block there is no ruler and no screening: it is an error, not
    a silent default.

    With `configs_dir` it also checks that ruler and fold use the same yardstick (`check_metro`):
    without that check a run that changes `data.labels` produces a meaningless PASS.

    The secondary en+es ruler (`secondary_ruler`) goes into `screen.json` under `enes` with its sign
    compared to the primary one (`same_sign`); the verdict is the primary's alone. `langs` (language
    per study) is read by default from `data/reports` next to the labels.
    """
    sc = cfg.get("screen")
    if not sc or "parent_fold_auc" not in sc or "delta_fold" not in sc:
        raise KeyError(f"{cfg.get('name')}: the config has no `screen:` block with "
                       "parent_fold_auc and delta_fold")
    out_dir = Path(out_dir)
    fold = screen_fold(cfg)
    frag = out_dir / f"oof_fold{fold}.csv"
    if not frag.is_file():
        raise FileNotFoundError(f"missing {frag}: the screening fold was not trained")
    metro = check_metro(cfg, labels_path, configs_dir, log=log)
    oof = pd.read_csv(frag, index_col=0)
    labels = pd.read_csv(labels_path, index_col=0)[TARGETS]
    folds = pd.read_csv(folds_path).set_index(ID)["fold"]
    res = fold_auc(oof, labels, folds, fold)
    ver = screen_verdict(res["macro_auc"], float(sc["parent_fold_auc"]),
                         float(sc["delta_fold"]), float(sc.get("stop_k", STOP_K)))
    sec = _secondary(cfg, oof, labels, folds, fold, res["n_studies"], langs, labels_path, log)
    if sec is not None and "delta" in sec:
        sec["same_sign"] = bool(np.sign(sec["delta"]) == np.sign(ver["delta"]))
    out = {"run": cfg["name"], "parent_run": cfg.get("parent"),
           "labels": cfg["data"]["labels"], **res,
           "parent_fold_auc": float(sc["parent_fold_auc"]),
           "delta_fold": float(sc["delta_fold"]), "stop_k": float(sc.get("stop_k", STOP_K)),
           **metro, **ver, "enes": sec}
    (out_dir / SCREEN_FILE).write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"screening fold {fold} (labels {out['labels']}): {res['macro_auc']:.4f} vs "
        f"parent {out['parent_fold_auc']:.4f} = {ver['delta']:+.4f} = {ver['k']:+.1f} δ_fold "
        f"(STOP at <= {ver['stop_below']:.4f}) -> **{ver['verdict']}**")
    return out


def read_screen(out_dir) -> dict | None:
    p = Path(out_dir) / SCREEN_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def screen_stopped(out_dir) -> bool:
    """True if `out_dir` contains a screen.json with a STOP verdict."""
    s = read_screen(out_dir)
    return bool(s) and s.get("verdict") == "STOP"
