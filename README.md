# RSNA 2026 Knee Abnormality Detection: efficiency submission

Code of the efficiency submission of team *Effusion confusion*: two DenseNet169 2.5D seeds at 512 px, trained on
all 4,407 training studies with soft labels from a teacher, run in INT8 on custom CUDA kernels on 2 × T4, without
internet.

| | Public LB | Private LB | Scoring time (2 × T4) |
|---|---|---|---|
| 2 × DenseNet169 2.5D, 512 px, INT8 | 0.960 | 0.94853 | 146 s (official run); 112-125 s in our benchmarks |

Efficiency leaderboard on the private data: 4th of 6,213 (efficiency −2.0540). No external data was used: only the
competition's training images and reports, and the public ImageNet weights of DenseNet169 (timm).

## Layout

```
src/knee/                the Python package (data, model, training, inference, INT8 encoder)
  data/early.py          decoding workers started before `import torch`
  data/lean.py           the reader of the submission: only the rows of the crop and the slices that are used
  data/cfast.py          ctypes bindings of the C decoder (data/fastdecode.c, data/leanio.c)
  data/raw_crop.py       crop and resize on the GPU
  data/fused_uint8.py    uint8 normalisation and TTA kernels
  dense8s.py             the DenseNet encoder in INT8 (kernels in dense8s_kernels.cu)
  window_budget.py       which windows each seed reads (half of them each)
  infer.py               the inference loop
  train.py, fastdense*.py  training, with the fast DenseNet training kernels
configs/exp/             the cache spec (313) and the two final seeds (376: seed 42, 384: seed 43)
data/folds/folds.csv     the 5-fold split used for every out-of-fold measurement
data/labels/             report-derived labels and the teacher's soft labels (see "Labels")
kaggle/                  the Kaggle kernels of the pipeline, one folder each, with kernel-metadata.json
prebuilt/                the binaries the submission embeds: C decoder library and uint8 CUBIN
tools/build_kernels.py   writes the ready-to-push kernels and datasets into build/
tools/train.py           trains one seed from a config and the cache
VERIFICATION.json        the check of this release against the code that ran (see below)
```

## Pipeline

Every stage except training runs on Kaggle. `tools/build_kernels.py <target>` writes the folder to push;
`--owner <kaggle user>` replaces the user name in the metadata. Keep the slugs: the submission finds its inputs by
slug.

| # | Stage | Where | Inputs | Output |
|---|---|---|---|---|
| 1 | `src-dataset`: `src/`, `configs/`, `data/` as a dataset | local | this repository | dataset `knee-src-label-ablation-196-197` |
| 2 | decoder wheels | local | PyPI | dataset `knee-wheels` |
| 3 | `cache`: 14 shard kernels, 512 px, 24 slices, crops of 110 and 150 mm | Kaggle, CPU, internet on | competition, 1 | uint8 volumes of the training studies |
| 4 | training of the two seeds (`tools/train.py`) | any single GPU with ≥40 GB | 3, `data/` | `fold0.pt` per seed |
| 5 | `weights`: the two checkpoints as one dataset | local | 4 | dataset `knee-runs-376-384-full-data-v9j-287-img512` |
| 6 | `int8-pack`: calibration and INT8 packages, TensorRT fp16 stem | Kaggle, 2 × T4, internet on | 1, 5, shard 0 of 3 | `knee-pack-dense8s-376x2v9j-b` |
| 7 | `int8-repack`: the compiled CUBIN into the packages | Kaggle, CPU, offline | 1, 6 | `knee-repack-dense8s-t8x32-376x2j` |
| 8 | `trt-runtime`: TensorRT 10.16.1.11 for Python 3.13, without the builder | Kaggle, CPU, internet on | none | `knee-trt-runtime-cp313` |
| 9 | `submission` | Kaggle, 2 × T4, internet off | competition, 2, 5, 7, 8 | `submission.csv` |

### Commands

```bash
python tools/build_kernels.py src-dataset
kaggle datasets create -p build/knee-src --dir-mode zip
```

The decoder wheels (the versions the submission used):

```bash
mkdir -p build/knee-wheels
pip download --no-deps --only-binary=:all: -d build/knee-wheels pylibjpeg==2.1.0
for v in 11 12 13; do pip download --no-deps --only-binary=:all: --platform manylinux2014_x86_64 --python-version 3.$v --implementation cp --abi cp3$v -d build/knee-wheels pylibjpeg-libjpeg==2.4.0 pylibjpeg-openjpeg==2.5.0 pylibjpeg-rle==2.2.0 python-gdcm==3.0.26; done
echo "DICOM decoders for the offline submission kernel" > build/knee-wheels/WHEELS
```

`build/knee-wheels` then needs a `dataset-metadata.json` with the id `<owner>/knee-wheels` before
`kaggle datasets create -p build/knee-wheels`. The `WHEELS` file is how the kernel finds the folder.

The cache, 14 kernels with the competition data attached; download their outputs to the training machine:

```bash
python tools/build_kernels.py cache
for k in $(seq 0 13); do kaggle kernels push -p build/cache_build-s${k}of14; done
```

Training, once per seed (the 384 config differs only in the seed):

```bash
python tools/train.py --config configs/exp/376-full-data-v9j-287-img512.yaml --cache <shard folders> --train-csv <competition>/train.csv --out runs/376
```

```bash
python tools/train.py --config configs/exp/384-full-data-v9j-287-img512-s43.yaml --cache <shard folders> --train-csv <competition>/train.csv --out runs/384
```

Weights, INT8, TensorRT runtime and submission, in this order, each pushed after the previous one has finished:

```bash
python tools/build_kernels.py weights --seed42 runs/376 --seed43 runs/384
python tools/build_kernels.py int8-pack
python tools/build_kernels.py int8-repack --nvrtc-site <folder with the nvidia-cuda-nvrtc-cu12 wheel unpacked>
python tools/build_kernels.py trt-runtime
python tools/build_kernels.py submission
```

`kaggle datasets create -p` for the weights folder, `kaggle kernels push -p` for the others. `int8-repack` compiles
`src/knee/dense8s_kernels.cu` with NVRTC on the local machine (~2 min); the resulting CUBIN is identical to the one the
submission used. The `trt-runtime` kernel must be created on the same Kaggle image as the submission kernel
(Python 3.13), so that pip selects the cp313 wheels. The submission is then submitted from its Kaggle page; it reads
the hidden test set only when submitted.

## Labels

- `data/labels/012-host-criteria-v2-t2-lieve.csv`: twelve findings per training study, extracted from the radiology
  reports by an LLM with the host's criteria. Training uses them for the validation probe and for the mask of cells
  excluded from the loss.
- `data/labels/matteo-v9j-253-h1-342.csv`: the soft labels of the final teacher (`train.teacher`), the training
  target of both seeds. They were built by Matteo Vitali (@matteovitali1004) from the reports and from out-of-fold
  predictions of image models; that pipeline is not in this repository and is described in his write-up.

Both files are derived from the competition's training data and are subject to the competition rules.

## Relation to the code that ran

Each stage ran from its own snapshot of the package. This release merges them into one package, with comments,
docstrings and messages translated into English. `VERIFICATION.json` records the check
(`scripts/opensource/verify_release.py` in the development repository):

- every Python file has the same AST as the one that ran, apart from docstrings and messages; C and CUDA files have
  the same tokens without comments;
- the submission kernel rebuilt by `tools/build_kernels.py submission` has the same AST as the one that was scored,
  apart from docstrings, messages and embedded blobs, and embeds the same files, except four modules
  (`aggregate.py`, `fastdense.py`, `metrics.py`, `train.py`) that come from the newer training snapshot of seed 42:
  the differences add options and functions, and the defaults keep the previous behaviour;
- the configs resolve to the same `config_hash` as the runs (376: `217892da98`, 384: `a4ce849603`);
- `prebuilt/knee_fastdecode_linux_x86_64.so` is byte-identical to the library of the scored kernel.

The kernel metadata differ from the original in three points: the INT8 package kernel is named after its `-b` copy
(the one the repack read), `code_file` matches the script name, and the kernels that mount source code all mount the
dataset of stage 1 (the cache kernel originally mounted a snapshot identical for the files it uses; the package
kernel also mounted a dataset it never read, now removed).

The package also contains the modules of experiments that are not part of the final model (for example
`public_raptor.py`, `dread_recipe.py`, `qwen.py`, `models/intra.py`), because the final kernels import from the same
package.

## Third-party code and weights

- DenseNet169 ImageNet weights from timm, loaded at training time.
- `public_raptor.py` and `dread_recipe.py` follow the public notebooks of Dread Development
  (`dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`,
  `dreaddevelopment/knee-mri-training-the-twelve-finding-model`); neither is used by the final model.
- `models/intra.py` and the compartment contrast in `train.py` are ports of code by Matteo Vitali; not used by the
  final model.
- At inference: TensorRT 10.16.1.11 (NVIDIA) for the stem, pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg,
  pylibjpeg-rle and python-gdcm for compressed DICOMs, each under its own licence.

## License

MIT, see `LICENSE`. The licence covers the code in this repository, not the competition data or the labels derived
from it.
