"""The public "Raptor" models (Dread Development, CC0-1.0 weights) inside our kernel: an experiment,
not used by the final submission.

Rank-averaging our own models had been measured only within one family (DenseNet, same recipe,
rho 0.93-0.95). A model from another family (CoAtNet at 384 px, labels, crop and slice sampling
from another team) is the one variable that measurement did not cover. The Raptor weights are
declared CC0-1.0 on their dataset pages, so they do not affect the winners' licence obligations.

What is copied and what is not. Model, windows and preprocessing are those of the author's
notebook (`dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`, with the geometry of
every checkpoint from its dataset page): the geometry is as much part of the model as the weights,
and a volume different from the training one does not return the declared score. Only the how
changes, not the what:

- decoding runs in the workers of a DataLoader (the notebook decoded study by study in the main
  process), and the two checkpoints read the same series once;
- file order is `order_series_files` (projection on the normal, like the notebook, but reading
  only the first bytes of each file);
- windows are built on the GPU (same bilinear interpolation, same ImageNet normalisation), not on
  CPU window by window.

No per-checkpoint weights tuned on the public leaderboard: the models are combined with equal
weights, like ours.
"""
from __future__ import annotations

import gc
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .constants import ID, TARGETS

# `data.dicom` (and pydicom with it) is imported where decoding happens, not here: `dread_recipe`
# takes only model, geometries and windows from this module, and the training environment has no
# pydicom.

# Slot (plane, fluid-sensitive preference, number of slices). Preference 1/0 picks the
# fluid-sensitive series or not if present, otherwise the first of the plane; -1 = just the first.
SLOTS_44 = (("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
            ("Coronal", 0, 6), ("Axial", -1, 8))
SLOTS_64 = (("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
            ("Coronal", 0, 8), ("Axial", -1, 12))
SLOTS_80 = (("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15),
            ("Coronal", 0, 10), ("Axial", -1, 15))

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class RaptorGeometry:
    """How the corpus a checkpoint was trained on was built."""
    name: str
    file: str                       # the checkpoint file inside the dataset
    dataset: str                    # the Kaggle slug of the weights dataset
    img: int                        # resolution at which the slice is stored (then -> res)
    slots: tuple
    span: tuple[float, float]       # fraction of the series covered by the slices
    k_eval: int                     # windows evaluated per study
    crop_mm: float = 140.0
    public_lb: float | None = None  # declared by the author, at this k_eval

    @property
    def n_slices(self) -> int:
        return sum(s[2] for s in self.slots)


# From the dataset pages (CC0-1.0 licence on all four). `k_eval` = every window position the volume
# contains, as in the declared measurement: the same checkpoint with fewer windows scores less
# (0.927 -> 0.924 from 62 to 42 windows, same page).
GEOMETRIES: dict[str, RaptorGeometry] = {
    "finespacing": RaptorGeometry("finespacing", "raptor_ft_coatnet_v9_full.pt",
                                  "dreaddevelopment/raptor-knee-finespacing", 336, SLOTS_80,
                                  (0.02, 0.98), 78, public_lb=0.932),
    "maxspan": RaptorGeometry("maxspan", "raptor_ft_coatnet_v5_full_swa.pt",
                              "dreaddevelopment/raptor-knee-maxspan", 336, SLOTS_64,
                              (0.02, 0.98), 62, public_lb=0.928),
    "native384dense": RaptorGeometry("native384dense", "raptor_ft_coatnet_v10_full.pt",
                                     "dreaddevelopment/raptor-knee-native384dense", 384,
                                     SLOTS_64, (0.02, 0.98), 62),
    "native384": RaptorGeometry("native384", "raptor_ft_coatnet_v8_full_swa.pt",
                                "dreaddevelopment/raptor-knee-native384", 384, SLOTS_44,
                                (0.06, 0.94), 42, public_lb=0.926),
}
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"


# --- model: identical to the author's `finetune_raptor.py` -------------------------------
def build_backbone(arch: str) -> nn.Module:
    """Always `pretrained=False`: the kernel has no network, the weights come from the checkpoint."""
    import timm
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=False, num_classes=0, in_chans=3)
    kw.update(dict(global_pool="token", dynamic_img_size=True) if is_vit else dict(global_pool="avg"))
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    """Backbone per window, then attention with separate weights for each of the 12 findings."""

    def __init__(self, backbone: nn.Module, F_dim: int = 768, n: int = 12, drop: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, K = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).view(B, K, -1)

    def head(self, feats: torch.Tensor) -> torch.Tensor:
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def load_raptor(path: Path, device, log=print) -> tuple[RaptorClassifier, int, list[int]]:
    """The checkpoint is a dict {model, arch, res, lab}. Also returns the permutation that brings the 12
    outputs into `TARGETS` order: read from the checkpoint, not assumed.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", ARCH)
    res = int(ck.get("res", 384))
    lab = list(ck.get("lab", TARGETS))
    if sorted(lab) != sorted(TARGETS):
        raise ValueError(f"{path}: checkpoint findings {lab} differ from {TARGETS}")
    bb = build_backbone(arch)
    model = RaptorClassifier(bb, F_dim=bb.num_features, n=len(lab))
    model.load_state_dict(ck["model"], strict=True)
    del ck
    gc.collect()
    log(f"raptor {Path(path).name}: {arch} at {res} px")
    return model.eval().to(device), res, [lab.index(t) for t in TARGETS]


# --- windows: the notebook's `_eval_centers`, pixels on the GPU ----------------------------
def eval_centers(mask: np.ndarray, k: int) -> np.ndarray:
    """The `k` central slices of the windows, between the first and last non-empty slice.

    Verbatim from the notebook (`StudyWindows`, train=False): windows may cross an empty slot between
    two full ones, and that is how the model was trained.
    """
    D = len(mask)
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return np.clip(np.asarray([cs[i] for i in idx], dtype=np.int64), 1, D - 2)


def make_windows(vol: torch.Tensor, centers: np.ndarray, res: int) -> torch.Tensor:
    """(K, 3, res, res) ImageNet-normalised from a uint8 volume (D, img, img).

    Every plane is interpolated once and then the triplets are gathered: the interpolation is per
    channel, so it is the same operation as the per-window version of the author's notebook.
    """
    trip = torch.as_tensor(centers, device=vol.device)[:, None] + torch.tensor([-1, 0, 1], device=vol.device)
    uniq, inv = torch.unique(trip.reshape(-1), return_inverse=True)
    planes = vol[uniq].float().div_(255.0)
    if planes.shape[-1] != res or planes.shape[-2] != res:
        planes = F.interpolate(planes[:, None], size=(res, res), mode="bilinear",
                               align_corners=False)[:, 0]
    x = planes[inv.reshape(-1, 3)]
    mean = torch.tensor(_MEAN, device=vol.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=vol.device).view(1, 3, 1, 1)
    return (x - mean) / std


# --- preprocessing: the notebook's `build_study`, several checkpoints on the same read ------
def _flag(value) -> int | None:
    """An unknown protocol flag is not false."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return int(v) if v in (0.0, 1.0) else None


def pick_series(rows: list[dict], plane: str, fluid: int, used: set) -> dict | None:
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if _flag(r.get("Fluid_Sensitive")) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _read_pixels(path: str) -> tuple[np.ndarray, float | None]:
    """float32 pixels with the modality LUT (as the notebook) and the slice's PixelSpacing."""
    import pydicom
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:                      # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    with open(path, "rb") as f:
        d = pydicom.dcmread(io.BytesIO(f.read()))
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    ps = getattr(d, "PixelSpacing", None)
    return a, (float(ps[0]) if ps is not None else None)


def _crop_resize(a: np.ndarray, ps: float, crop_mm: float, img: int) -> np.ndarray:
    import cv2
    h, w = a.shape
    cpx = min(int(round(crop_mm / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (img, img), interpolation=cv2.INTER_AREA)


def build_raptor_volumes(study_dir: str, rows: list[dict], geoms: list[RaptorGeometry],
                         errors: "DecodeErrors | None" = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One study -> {geometry: (vol (D, img, img) uint8, mask (D,) uint8)}.

    For every geometry it is the notebook's `build_study`: series per slot, slices taken with
    `linspace` over the band, intensity window at the 2-98 percentiles **of the slices taken for that
    geometry**, 140 mm crop from the slice's PixelSpacing, INTER_AREA resize. Geometries share file
    order and decoded pixels, nothing else.
    """
    from .data.dicom import order_series_files

    cache_order: dict[str, list[str]] = {}
    cache_px: dict[str, tuple[np.ndarray, float | None] | None] = {}

    def files_of(series_uid: str) -> list[str]:
        if series_uid not in cache_order:
            sdir = os.path.join(study_dir, series_uid)
            cache_order[series_uid] = [os.path.join(sdir, f)
                                       for f in order_series_files(sdir).files]
        return cache_order[series_uid]

    def pixels(path: str):
        if path not in cache_px:
            try:
                cache_px[path] = _read_pixels(path)
            except Exception as exc:
                if errors is not None:
                    errors.record(path, exc)
                cache_px[path] = None
        return cache_px[path]

    out = {}
    for g in geoms:
        vol = np.zeros((g.n_slices, g.img, g.img), np.uint8)
        idx, used = 0, set()
        for plane, fluid, k in g.slots:
            r = pick_series(rows, plane, fluid, used)
            if r is None:
                idx += k
                continue
            used.add(r["SeriesInstanceUID"])
            files = files_of(str(r["SeriesInstanceUID"]))
            n = len(files)
            if not n:
                idx += k
                continue
            lo, hi = int(n * g.span[0]), int(n * g.span[1]) - 1
            hi = max(hi, lo)
            picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
            got = [pixels(files[min(p, n - 1)]) for p in picks]
            valid = [x[0] for x in got if x is not None]
            # As in the notebook: a missing PixelSpacing counts as 0.5 mm, and the series median is used only
            # for a non-positive spacing.
            spacings = [x[1] if x[1] is not None else 0.5 for x in got if x is not None]
            med_ps = float(np.median(spacings)) if spacings else 0.5
            if valid:
                loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
            else:
                loq, hiq = 0.0, 1.0
            for x in got:
                if idx >= g.n_slices:
                    break
                if x is None:
                    idx += 1
                    continue
                a, ps = x
                ps = ps if ps is not None else 0.5
                aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                aw = _crop_resize(aw, ps if ps > 0 else med_ps, g.crop_mm, g.img)
                vol[idx] = (aw * 255).astype(np.uint8)
                idx += 1
            if idx >= g.n_slices:
                break
        mask = (vol.reshape(g.n_slices, -1).sum(1) > 0).astype(np.uint8)
        out[g.name] = (vol, mask)
    return out


class RaptorStudies(Dataset):
    """One study per item, all geometries from the same read, in the worker."""

    def __init__(self, root: Path, uids: list[str], series: pd.DataFrame,
                 geoms: list[RaptorGeometry], series_dir: str = "test_series"):
        self.root, self.uids, self.geoms = Path(root), list(uids), list(geoms)
        self.series_dir = series_dir
        s = series.copy()
        s[ID] = s[ID].astype(str)
        s["SeriesInstanceUID"] = s["SeriesInstanceUID"].astype(str)
        self.rows = {k: v.to_dict("records") for k, v in s.groupby(ID)}

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        study = self.uids[i]
        from .data.dicom import DecodeErrors

        errors = DecodeErrors()
        t0 = time.perf_counter()
        try:
            vols = build_raptor_volumes(str(self.root / self.series_dir / study),
                                        self.rows.get(study, []), self.geoms, errors)
        except Exception:
            # Like `DicomStudies`: the study comes out anyway, empty, and becomes NaN downstream.
            vols = {g.name: (np.zeros((g.n_slices, g.img, g.img), np.uint8),
                             np.zeros(g.n_slices, np.uint8)) for g in self.geoms}
        flat = []
        for g in self.geoms:
            v, m = vols[g.name]
            flat += [torch.from_numpy(v), torch.from_numpy(m)]
        return (*flat, i, time.perf_counter() - t0, len(errors))


def _worker_init(_worker_id: int) -> None:
    try:
        import cv2
        cv2.setNumThreads(1)        # the workers already provide the parallelism
    except Exception:
        pass
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass


@torch.no_grad()
def predict_raptor(models: dict, ds: RaptorStudies, device, workers: int = 2,
                   time_budget_s: float | None = None, log=print,
                   log_every: int = 100) -> tuple[dict[str, np.ndarray], dict]:
    """{name: (N, 12) probabilities in TARGETS order}. Rows not computed = NaN.

    `models`: {geometry name: (model, res, permutation)}. One study per forward, as in the notebook:
    K = 62-78 windows at 384 px are already a full batch for a T4.
    """
    names = [g.name for g in ds.geoms]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                        pin_memory=(device.type == "cuda"), worker_init_fn=_worker_init,
                        persistent_workers=workers > 0,
                        prefetch_factor=(4 if workers > 0 else None))
    probs = {n: np.full((len(ds), len(TARGETS)), np.nan, np.float32) for n in names}
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True    # fixed shapes per geometry
    t0 = time.time()
    decode_s, gpu_s, done, n_err, n_empty, n_fp32 = 0.0, 0.0, 0, 0, 0, 0
    for batch in loader:
        *flat, idx, dt, errs = batch
        decode_s += float(dt.sum())
        n_err += int(errs.sum())
        i = int(idx[0])
        g0 = time.perf_counter()
        for j, name in enumerate(names):
            vol, mask = flat[2 * j][0], flat[2 * j + 1][0].numpy()
            if not mask.any():
                n_empty += name == names[0]
                continue
            model, res, perm = models[name]
            k = next(g.k_eval for g in ds.geoms if g.name == name)
            x = make_windows(vol.to(device, non_blocking=True), eval_centers(mask, k), res)
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    p = torch.sigmoid(model(x[None]).float())[0]
            except RuntimeError:
                # The notebook does the same: fp32 always has a conv engine on Turing.
                n_fp32 += 1
                if use_amp:
                    torch.cuda.empty_cache()
                p = torch.sigmoid(model(x[None].float()).float())[0]
            probs[name][i] = p.cpu().numpy()[perm]
        gpu_s += time.perf_counter() - g0
        done += 1
        if done % log_every == 0:
            rate = done / max(time.time() - t0, 1e-6)
            log(f"  raptor {done}/{len(ds)} | {rate:.2f} studies/s | ~"
                f"{(len(ds) - done) / rate / 60:.0f} min left")
        if time_budget_s and time.time() - t0 > time_budget_s:
            log(f"!! raptor: budget exhausted at {done}/{len(ds)} studies, the rest stays NaN")
            break
    wall = time.time() - t0
    return probs, {"n_studies": len(ds), "n_predicted": done, "wall_s": round(wall, 1),
                   "decode_cpu_s": round(decode_s, 1), "gpu_s": round(gpu_s, 1),
                   "s_per_study": round(wall / max(done, 1), 3),
                   "studies_without_any_slot": n_empty, "slice_decode_failures": n_err,
                   "fp32_retries": n_fp32, "geometries": names}
