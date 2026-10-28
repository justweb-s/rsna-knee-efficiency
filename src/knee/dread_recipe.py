"""Replication of a public training recipe (Dread's "Raptor" models, CC0) with our labels and our
split: an experiment, not used by the final model.

The public recipe in full (a corpus of 80 slices over five series, 2-98% band, 140 mm, 336 px read
at 384; windows of 3 consecutive slices between the first and last full slice of the volume;
separate attention per finding; BCE with pos_weight on soft labels; AdamW 3e-5 / 1e-3, wd 0.02,
OneCycle at 15%, bf16, 16 epochs) and, instead of the author's LLM labels, which are not public,
ours. The question is how much of the gap between those models and ours lies in the recipe.

What is copied and what is not (public notebook `dreaddevelopment/knee-mri-training-the-twelve-
finding-model`):

- the volume is the corpus of `knee.raptor_corpus` (the author's `build_study`, byte for byte);
- window centres: in training `k` random centres from the repeated pool of valid positions (like
  `_centers`), in evaluation `public_raptor.eval_centers` (verbatim);
- the model is `public_raptor.RaptorClassifier`: a checkpoint of this run loads with
  `public_raptor.load_raptor`;
- bilinear resize, ±10% intensity jitter per study, ImageNet normalisation: on the GPU instead of
  per window on CPU, same arithmetic (`prep_windows` is checked against
  `public_raptor.make_windows`);
- NOT copied: the split. The author trains on all of train minus the 58 gold studies and picks the
  epoch (and the mean of the best three) on the gold studies; here training uses the training
  folds of `folds.csv` with the gold studies inside, like every other run, and keeps the last epoch
  (`select: last`): choosing the epoch on the validation fold would make the OOF optimistic.
- randomness: global `random` in the notebook, here one generator per (seed, fold, epoch, study),
  so the run does not depend on how many workers read the disk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .constants import TARGETS
from .labels.gold import load_gold
from .metrics import evaluate
from .public_raptor import GEOMETRIES, RaptorClassifier, _MEAN, _STD, eval_centers

RECIPE = "dread"


def is_dread(cfg: dict) -> bool:
    return (cfg.get("recipe") or {}).get("name") == RECIPE


def recipe_of(cfg: dict) -> dict:
    """The `recipe:` block with the defaults of the author's recipe."""
    r = dict(cfg.get("recipe") or {})
    if r.get("name") != RECIPE:
        raise ValueError(f"{cfg.get('name')}: recipe.name != {RECIPE!r}")
    r.setdefault("geometry", "finespacing")
    g = GEOMETRIES[r["geometry"]]
    r.setdefault("corpus_shards", 3)
    r.setdefault("res", 384)
    r.setdefault("k_eval", g.k_eval)
    r.setdefault("jitter", 0.20)
    r.setdefault("prev_clip", [0.03, 0.7])
    r.setdefault("pos_weight_clip", [1.0, 10.0])
    r.setdefault("pct_start", 0.15)
    r.setdefault("grad_clip", 3.0)
    r.setdefault("head_drop", 0.2)
    return r


# --- the corpus: contiguous shards, read as memmaps ------------------------------------------

def shard_prefix(geometry: str, k: int, n: int) -> str:
    """The same name as `raptor_corpus.corpus_prefix`."""
    return f"raptor_{geometry}_s{k}of{n}"


def check_shard(d: Path, prefix: str, geometry: str) -> tuple[bool, str]:
    """Is the shard in `d` complete? Checks the array (its shape), not the mere existence."""
    g = GEOMETRIES[geometry]
    d = Path(d)
    names = [f"{prefix}_{s}" for s in ("vols.npy", "masks.npy", "ids.npy")]
    for n in names:
        if not (d / n).is_file():
            return False, f"missing {n}"
    try:
        v = np.load(d / names[0], mmap_mode="r")
        m = np.load(d / names[1], mmap_mode="r")
        ids = np.load(d / names[2], allow_pickle=True)
    except Exception as e:  # noqa: BLE001 - il messaggio dice quale file
        return False, f"illeggibile ({type(e).__name__}: {e})"
    want = (g.n_slices, g.img, g.img)
    if tuple(v.shape[1:]) != want:
        return False, f"shape {tuple(v.shape[1:])}, the geometry wants {want}"
    if not (len(v) == len(m) == len(ids)):
        return False, f"rows differ: vols {len(v)}, masks {len(m)}, ids {len(ids)}"
    b = d / f"{prefix}_build.json"
    if b.is_file():
        n_b = json.loads(b.read_text(encoding="utf-8")).get("n_studies")
        if n_b is not None and n_b != len(v):
            return False, f"{len(v)} studies against {n_b} declared by {b.name}"
    return True, f"{v.nbytes / 1e9:.1f} GB, {len(v)} studies"


class Corpus:
    """The shards as a single corpus, in shard order (`s0`, `s1`, ...).

    The constructor opens nothing beyond ids and masks: a memmap passed to the workers of a
    DataLoader is serialised by copying the data, so every process opens its own (`vol`).
    """

    def __init__(self, dirs, geometry: str = "finespacing"):
        dirs = [Path(d) for d in (dirs if isinstance(dirs, (list, tuple)) else [dirs])]
        self.geometry = geometry
        self.g = GEOMETRIES[geometry]
        found = []
        for d in dirs:
            for p in sorted(d.glob(f"raptor_{geometry}_s*of*_vols.npy")):
                pre = p.name[: -len("_vols.npy")]
                k, n = pre.rsplit("_s", 1)[1].split("of")
                found.append((int(k), int(n), p.parent, pre))
        if not found:
            raise FileNotFoundError(f"no raptor_{geometry}_s*of* shard in {dirs}")
        ns = {n for _, n, _, _ in found}
        ks = sorted(k for k, _, _, _ in found)
        if len(ns) != 1 or ks != list(range(ns.pop())):
            raise ValueError(f"incomplete or duplicate shards: {[(k, n) for k, n, _, _ in found]}")
        found.sort()
        self.parts = []
        ids, masks, lo = [], [], 0
        for k, n, d, pre in found:
            ok, why = check_shard(d, pre, geometry)
            if not ok:
                raise ValueError(f"shard {pre} in {d}: {why}")
            i = np.load(d / f"{pre}_ids.npy", allow_pickle=True).astype(str)
            self.parts.append((d / f"{pre}_vols.npy", lo, lo + len(i)))
            ids.append(i)
            masks.append(np.load(d / f"{pre}_masks.npy"))
            lo += len(i)
        self.ids = np.concatenate(ids)
        self.masks = np.concatenate(masks)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("StudyInstanceUID repeated across shards")
        self.row = {u: r for r, u in enumerate(self.ids)}
        self._vols = None

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, uid) -> bool:
        return str(uid) in self.row

    def describe(self) -> str:
        g = self.g
        return (f"corpus {self.geometry} {len(self)} studies x {g.n_slices} slices x {g.img} px, "
                f"{len(self.parts)} shards, filled slots {self.masks.sum(1).mean():.1f}/{g.n_slices}")

    def vol(self, row: int) -> np.ndarray:
        if self._vols is None:
            self._vols = [np.load(p, mmap_mode="r") for p, _, _ in self.parts]
        for (_, lo, hi), v in zip(self.parts, self._vols):
            if lo <= row < hi:
                return v[row - lo]
        raise IndexError(row)

    def __getstate__(self):
        s = dict(self.__dict__)
        s["_vols"] = None
        return s


def open_corpus(dirs, cfg: dict) -> Corpus:
    return Corpus(dirs, recipe_of(cfg)["geometry"])


# --- windows --------------------------------------------------------------------------

def valid_centers(mask: np.ndarray) -> list[int]:
    """The possible centre positions: between the first and last full slice (`_centers`)."""
    D = len(mask)
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    return cs


def train_centers(mask: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """`k` random centres: the pool of positions repeated as needed, shuffled, the first `k` (like
    `_centers` in training: no repetitions as long as there are enough positions).
    """
    cs = valid_centers(mask)
    pool = np.asarray(cs * (k // len(cs) + 1), dtype=np.int64)
    rng.shuffle(pool)
    return np.clip(pool[:k], 1, len(mask) - 2)


class DreadWindows(Dataset):
    """One study -> (K, 3, img, img) uint8 (triplets of consecutive slices) and the target.

    Resize to `res` and normalisation happen on the GPU (`prep_windows`). In training the centres are
    drawn with one generator per (seed, epoch, study): `epoch` is set before every epoch, and
    non-persistent workers see it.
    """

    def __init__(self, corpus: Corpus, uids: list[str], targets: np.ndarray, k: int,
                 train: bool, seed: int = 0):
        self.corpus, self.uids = corpus, [str(u) for u in uids]
        self.rows = [corpus.row[u] for u in self.uids]
        self.targets = np.asarray(targets, dtype=np.float32)
        self.k, self.train, self.seed, self.epoch = int(k), bool(train), int(seed), 0

    def __len__(self) -> int:
        return len(self.uids)

    def centers(self, i: int) -> np.ndarray:
        m = self.corpus.masks[self.rows[i]]
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            return train_centers(m, self.k, rng)
        return eval_centers(m, self.k)

    def __getitem__(self, i: int):
        c = self.centers(i)
        vol = self.corpus.vol(self.rows[i])
        trip = (c[:, None] + np.array([-1, 0, 1])).reshape(-1)
        x = np.asarray(vol[trip]).reshape(len(c), 3, *vol.shape[1:])
        return torch.from_numpy(x), torch.from_numpy(self.targets[i])


def prep_windows(x: torch.Tensor, res: int, norm: str | None = "imagenet",
                 jitter: float = 0.0, gen: torch.Generator | None = None) -> torch.Tensor:
    """(B, K, 3, H, W) uint8 -> (B, K, 3, res, res) float, like `StudyWindows.__getitem__`.

    The author's order: [0, 1], bilinear resize per channel, one intensity gain per study in
    [1 - jitter/2, 1 + jitter/2] with clamp to [0, 1], then ImageNet mean/std.
    """
    B, K, C, H, W = x.shape
    t = x.float().div_(255.0).reshape(B * K, C, H, W)
    if H != res or W != res:
        t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
    t = t.reshape(B, K, C, res, res)
    if jitter:
        g = 1.0 + (torch.rand(B, generator=gen) - 0.5) * jitter
        t = (t * g.to(t.device).view(B, 1, 1, 1, 1)).clamp_(0, 1)
    if norm == "imagenet":
        mean = torch.tensor(_MEAN, device=t.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(_STD, device=t.device).view(1, 1, 3, 1, 1)
        t = (t - mean) / std
    elif norm not in (None, "none"):
        raise ValueError(f"norm {norm!r}")
    return t


# --- model ---------------------------------------------------------------------------

def build_model(cfg: dict) -> RaptorClassifier:
    import timm
    arch = cfg["model"]["arch"]
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=bool(cfg["model"].get("pretrained", True)), num_classes=0, in_chans=3)
    kw.update(dict(global_pool="token", dynamic_img_size=True) if is_vit else dict(global_pool="avg"))
    bb = timm.create_model(arch, **kw)
    if cfg["train"].get("grad_checkpointing"):
        bb.set_grad_checkpointing(True)
    return RaptorClassifier(bb, F_dim=bb.num_features, n=len(TARGETS),
                            drop=float(recipe_of(cfg)["head_drop"]))


def pos_weight_of(y: np.ndarray, prev_clip, pw_clip) -> np.ndarray:
    """`(1 - p) / p` on the soft training prevalence, as the author does."""
    prev = np.clip(np.nanmean(y, axis=0), *prev_clip)
    return np.clip((1 - prev) / prev, *pw_clip).astype(np.float32)


def checkpoint_of(model: nn.Module, cfg: dict, fold: int) -> dict:
    """The `public_raptor.load_raptor` format ({model, arch, res, lab}) plus cfg and fold."""
    return {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "arch": cfg["model"]["arch"], "res": int(recipe_of(cfg)["res"]),
            "lab": list(TARGETS), "cfg": cfg, "fold": fold}


@torch.no_grad()
def predict_dread(model, loader, device, res: int, norm, dtype) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        x = prep_windows(x.to(device, non_blocking=True), res, norm)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    return np.concatenate(out)


def train_fold_dread(cfg, corpus: Corpus, labels: pd.DataFrame, folds: dict, fold: int, device,
                     log=print, weights_dir=None, gold_path=None, cell_weights=None,
                     pseudo=None, teacher=None):
    """Same contract as `train.train_fold`: (oof, infer_s, n_val, model, epoch, {}, extra)."""
    for name, v in (("encoder_weights", weights_dir), ("cell_weight", cell_weights),
                    ("pseudo", pseudo), ("teacher", teacher)):
        if v is not None:
            raise ValueError(f"the Dread recipe does not support {name}")
    if cfg["train"].get("select", "last") != "last" or cfg["train"].get("full_data"):
        raise ValueError("the Dread recipe runs here with select: last and without full_data")
    r = recipe_of(cfg)
    tr_uids = [u for u in labels.index if folds.get(u, -1) != fold]
    va_uids = [u for u in labels.index if folds.get(u, -1) == fold]
    log(f"  fold {fold}: {len(tr_uids)} train / {len(va_uids)} val")
    gold_va = None
    if gold_path is not None:
        gold_all = load_gold(gold_path)
        gold_va = gold_all.loc[gold_all.index.intersection(va_uids)]
        gold_va = None if gold_va.empty else gold_va

    seed = int(cfg["seed"])
    k, k_eval, res = int(cfg["train"]["k_train"]), int(r["k_eval"]), int(r["res"])
    norm = cfg["model"].get("input_norm", "imagenet")
    ds_tr = DreadWindows(corpus, tr_uids, labels.loc[tr_uids].to_numpy(), k, True,
                         seed=seed * 1000 + fold)
    ds_va = DreadWindows(corpus, va_uids, labels.loc[va_uids].to_numpy(), k_eval, False)
    workers = int(cfg["train"]["workers"])
    g_tr = torch.Generator().manual_seed(seed * 1000 + fold)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                       drop_last=True, generator=g_tr, num_workers=workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["eval_batch_size"], shuffle=False,
                       num_workers=workers, pin_memory=True)
    gen_aug = torch.Generator().manual_seed(seed * 7919 + fold)

    model = build_model(cfg).to(device)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    lr, lr_bb = float(cfg["train"]["lr"]), float(cfg["train"]["lr_backbone"])
    opt = torch.optim.AdamW([{"params": bb, "lr": lr_bb}, {"params": head, "lr": lr}],
                            weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_bb, lr],
                                                total_steps=max(1, epochs * len(dl_tr)),
                                                pct_start=float(r["pct_start"]))
    pw = pos_weight_of(labels.loc[tr_uids].to_numpy(), r["prev_clip"], r["pos_weight_clip"])
    pw_t = torch.tensor(pw, device=device)
    dtype = torch.bfloat16 if cfg["train"].get("amp_dtype", "bf16") == "bf16" else torch.float16
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    log(f"fold {fold}: Dread recipe, {r['geometry']} at {res} px, k {k} in training / "
        f"{k_eval} in evaluation, {cfg['model']['arch']} (pretrained "
        f"{cfg['model'].get('pretrained', True)}), input_norm {norm}, jitter {r['jitter']}")
    log(f"  lr: encoder {lr_bb:g} ({len(bb)} tensors), head {lr:g} ({len(head)} tensors), "
        f"wd {cfg['train']['weight_decay']}, OneCycle pct_start {r['pct_start']}, "
        f"{'bf16' if dtype == torch.bfloat16 else 'fp16'}, grad_clip {r['grad_clip']}")
    log("  pos_weight " + " ".join(f"{t.split()[0][:6]} {w:.2f}" for t, w in zip(TARGETS, pw)))

    val_every = int(cfg["train"].get("val_every", 0))
    history, gold_frames, last_va, infer_s = [], [], None, 0.0
    for ep in range(epochs):
        model.train()
        ds_tr.epoch = ep
        t0, tot, tot_bce, n = time.time(), 0.0, 0.0, 0
        for x, y in dl_tr:
            y = y.to(device, non_blocking=True)
            x = prep_windows(x.to(device, non_blocking=True), res, norm, float(r["jitter"]), gen_aug)
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(r["grad_clip"]))
            opt.step()
            sched.step()
            # The BCE without pos_weight: the number comparable with the `bce` of our other runs.
            tot_bce += float(F.binary_cross_entropy_with_logits(logits.detach(), y)) * len(y)
            tot += float(loss.detach()) * len(y)
            n += len(y)
        rec = {"epoch": ep + 1, "bce": tot_bce / max(n, 1), "loss": tot / max(n, 1)}
        msg = (f"  fold {fold} ep {ep + 1}/{epochs}  bce {rec['bce']:.4f}  "
               f"loss {rec['loss']:.4f}")
        if val_every and ((ep + 1) % val_every == 0 or ep + 1 == epochs):
            t_va = time.time()
            va = pd.DataFrame(predict_dread(model, dl_va, device, res, norm, dtype),
                              index=va_uids, columns=TARGETS)
            infer_s, last_va = time.time() - t_va, (ep + 1, va)
            yb = (labels.loc[va_uids] > 0.5).astype(float)
            yb[labels.loc[va_uids].isna()] = np.nan
            rec["val_auc"] = evaluate(yb, va)["macro_auc"]
            msg += f"  val_auc {rec['val_auc']:.4f}"
            if gold_va is not None:
                gm = evaluate(gold_va, va.loc[gold_va.index])
                rec.update(gold_auc=gm["macro_auc"], gold_n=int(len(gold_va)),
                           gold_n_classes=gm["n_classes"])
                msg += f"  gold {gm['macro_auc']:.4f} (n={len(gold_va)}, {gm['n_classes']} col)"
                gold_frames.append(va.loc[gold_va.index].assign(epoch=ep + 1))
        rec["seconds"] = round(time.time() - t0, 1)
        history.append(rec)
        log(msg + f"  {rec['seconds']:.0f}s")

    if last_va is not None and last_va[0] == epochs:
        out = last_va[1]                     # the last epoch has already been evaluated
    else:
        t_va = time.time()
        out = pd.DataFrame(predict_dread(model, dl_va, device, res, norm, dtype),
                           index=va_uids, columns=TARGETS)
        infer_s = time.time() - t_va
    extra = {"history": history,
             "gold_curve": pd.concat(gold_frames) if gold_frames else None}
    return out, infer_s, len(va_uids), model, epochs, {}, extra
