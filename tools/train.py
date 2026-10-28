"""Train one of the final DenseNet169 seeds (or any config in configs/exp) on a prepared cache.

This is the same call the training driver made for the two final seeds: `knee.train.run` with the
resolved config, the cache shards built by `kaggle/cache_build`, the report-derived labels (`data.labels`,
used for the validation probe and for the mask of excluded cells), the frozen split and the competition
`train.csv` (for the 58 radiologist-labelled studies). The teacher named by `train.teacher` is read from
the same folder as `data.labels`.

    python tools/train.py --config configs/exp/376-full-data-v9j-287-img512.yaml \\
        --cache /path/to/cache_shard_0 /path/to/cache_shard_1 ... \\
        --train-csv /path/to/rsna-knee-abnormality-detection/train.csv --out runs/376

The checkpoint `fold0.pt` in `--out` is the model used by the submission (seed 42 for the 376 config,
seed 43 for the 384 one). The final runs used a single A100/H100-class GPU (~65 min on an H100, ~4 h on a
40 GB A100) with Python 3.11, torch 2.8.0 and timm 1.0.15 (see README).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knee.train import run  # noqa: E402
from knee.utils.config import config_hash, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cache", type=Path, nargs="+", required=True,
                    help="the cache folder, or the folders of its shards (in any order)")
    ap.add_argument("--train-csv", type=Path, required=True, help="the competition train.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="folder with labels/ and folds/")
    a = ap.parse_args()

    cfg = load_config(a.config)
    print(f"{cfg['name']} | config_hash {config_hash(cfg)}", flush=True)
    cache_root = a.cache[0] if len(a.cache) == 1 else list(a.cache)
    a.out.mkdir(parents=True, exist_ok=True)
    run(cfg, cache_root=cache_root,
        labels_path=a.data / "labels" / f"{cfg['data']['labels']}.csv",
        folds_path=a.data / "folds" / cfg["data"]["folds"],
        gold_path=a.train_csv, out_dir=a.out)


if __name__ == "__main__":
    main()
