"""Per-site test-time correction (not used by the final submission): on the "silent" columns it
removes the per-site offset the student learned from the teacher.

The teacher comes from the reports of ~10 sites, recognisable by their language. A site that
never mentions a finding (synovitis in the Cyrillic reports) makes it "negative", and the
student, which recognises the site from the image, learns "this site -> less synovitis". The
test set is read by radiologists with a single criterion: there, that offset is noise.

At test time there is no report, so the language is predicted from the series geometry: one
header per series (matrix, spacing, thickness, number of files, plane), 93% in CV on train. The
correction is the expected offset, sum_l P(l | geometry) * off[l, c], subtracted from the logit
of the silent columns only. The offsets come from five-fold OOF predictions. The classifier is
trained inside the kernel from the training table (`site_geom_train.csv.gz`): no pickles across
scikit-learn versions.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

MUTE = ("Synovitis", "Baker's", "Contusion", "Fracture")
GEOM_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns",
             "SliceThickness"]
HERE = Path(__file__).resolve().parent
TABLE = HERE / "site_geom_train.csv.gz"
OFFSETS = HERE / "site_offsets.json"
FEATURES = ["n_series", "n_files"] + [f"{ax}_{k}" for ax in "xyz" for k in
                                      ("n", "rows", "cols", "ps0", "thick", "n_files", "rows_max",
                                       "ps_min", "thick_min")]


def _floats(v, n):
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return out if len(out) == n else None


def series_features(series_dir) -> dict | None:
    """The scalars of a series from the first file (in name order) with a valid position and
    orientation.
    """
    import pydicom
    try:
        files = sorted(os.listdir(series_dir))
    except OSError:
        return None
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, f), stop_before_pixels=True,
                                 specific_tags=GEOM_TAGS)
        except Exception:                                       # noqa: BLE001
            continue
        o = _floats(getattr(ds, "ImageOrientationPatient", None), 6)
        if _floats(getattr(ds, "ImagePositionPatient", None), 3) is None or o is None:
            continue
        ps = _floats(getattr(ds, "PixelSpacing", None), 2)
        normal = np.cross(np.asarray(o[:3]), np.asarray(o[3:]))
        return {"rows": int(getattr(ds, "Rows", 0) or 0), "cols": int(getattr(ds, "Columns", 0) or 0),
                "ps0": ps[0] if ps else None, "thick": float(getattr(ds, "SliceThickness", 0) or 0),
                "n_files": len(files), "normal_axis": "xyz"[int(np.argmax(np.abs(normal)))]}
    return None


def study_features(series: list[dict]) -> dict:
    """One study -> the classifier features. `series` in series-UID order: on equal file counts the
    first one wins, in training as in the kernel.
    """
    f = {"n_series": len(series), "n_files": sum(s["n_files"] for s in series)}
    for ax in "xyz":
        e = [s for s in series if s["normal_axis"] == ax]
        f[f"{ax}_n"] = len(e)
        if not e:
            continue
        b = max(e, key=lambda s: s["n_files"])
        for k in ("rows", "cols", "ps0", "thick", "n_files"):
            f[f"{ax}_{k}"] = b[k]
        f[f"{ax}_rows_max"] = max(s["rows"] for s in e)
        ps = [s["ps0"] for s in e if s["ps0"] is not None]
        f[f"{ax}_ps_min"] = min(ps) if ps else None
        f[f"{ax}_thick_min"] = min(s["thick"] for s in e)
    return f


def frame(rows: dict):
    """{study: features} -> DataFrame with the FEATURES columns (NaN where a plane is missing)."""
    import pandas as pd
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=FEATURES).astype(float)


def fit_classifier(table):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
    return clf.fit(table[FEATURES].astype(float), table["lang"].astype(str))


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def correct(pred, proba, offsets: dict):
    """`pred`: probabilities (studies x TARGETS); `proba`: P(language | geometry) (studies x languages,
    missing rows = study without geometry, left unchanged). Returns a copy corrected on the
    columns of `offsets["columns"]` only.
    """
    out = pred.copy()
    off = offsets["offsets"]
    p = proba.reindex(pred.index)
    have = p.notna().all(axis=1).values
    for c in offsets["columns"]:
        shift = np.zeros(len(pred))
        for lang in p.columns:
            shift += p[lang].fillna(0).values * off.get(lang, {}).get(c, 0.0)
        z = logit(pred[c].values) - shift
        # only the studies with geometry: the others keep their value bit for bit
        out[c] = np.where(have, 1 / (1 + np.exp(-z)), pred[c].values)
    return out


def correct_submission(sub_path: Path, test_root: Path, out_dir: Path, workers: int = 8) -> dict:
    """In the kernel, after `run_submission`: reads `submission.csv`, predicts the site of every test
    study from the geometry, writes the corrected version in its place and the raw one next to it
    (`submission_raw.csv`). Returns a summary, also written to `site_correction.json`.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    from .constants import ID
    t0 = time.time()
    sub = pd.read_csv(sub_path).set_index(ID)
    offsets = json.loads(OFFSETS.read_text(encoding="utf-8"))
    table = pd.read_csv(TABLE)
    clf = fit_classifier(table)
    base = Path(test_root) / "test_series"

    def one(study: str):
        d = base / study
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return study, None
        series = [s for s in (series_features(d / n) for n in names) if s is not None]
        return study, (study_features(series) if series else None)

    with ThreadPoolExecutor(workers) as ex:
        feats = dict(ex.map(one, list(sub.index)))
    rows = {u: f for u, f in feats.items() if f is not None}
    proba = pd.DataFrame(index=pd.Index([], name=ID), columns=list(clf.classes_), dtype=float)
    if rows:
        X = frame(rows)
        proba = pd.DataFrame(clf.predict_proba(X), index=X.index, columns=list(clf.classes_))
    fixed = correct(sub, proba, offsets)
    sub.to_csv(out_dir / "submission_raw.csv")
    fixed.to_csv(sub_path)
    summary = {"studies": len(sub), "with_geometry": len(rows), "secs": round(time.time() - t0, 1),
               "predicted": proba.idxmax(axis=1).value_counts().to_dict() if len(proba) else {},
               "mean_abs_logit_shift": {c: round(float(np.mean(np.abs(logit(fixed[c]) - logit(sub[c])))), 4)
                                        for c in offsets["columns"]},
               "offsets_source": offsets.get("source")}
    (out_dir / "site_correction.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
