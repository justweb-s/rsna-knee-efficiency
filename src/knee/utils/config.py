"""YAML configs with inheritance, and the hash written to metrics.json.

A run is identified by its config, so the hash is computed on the *resolved* config
(base + experiment), not on the experiment file: two runs inheriting from different bases
must not be able to share a hash.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ..constants import REPO


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load_config((path.parent / parent).resolve()), cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def git_sha() -> str:
    """Commit SHA. There is no git on Kaggle: the bundling script writes it to a file."""
    stamp = Path(__file__).resolve().parents[1] / "GIT_SHA"
    if stamp.is_file():
        return stamp.read_text(encoding="utf-8").strip()
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"
