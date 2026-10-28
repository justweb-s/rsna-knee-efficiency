"""Fewer windows per ensemble member: each (seed, TTA view) pair sees its own subset of windows.

The cost of a submission is linear in the number of windows that go through the encoder: five
seeds x four TTA views x 88 windows = 1,760 images per study, and on two power-capped T4s the GPU
does not go below ~0.35 s/study, against ~0.20 s for a single seed. The model, however, is trained
on `k_train` windows drawn at random among the 88, and averaging the probabilities over subsets
recovers the prediction on all 88 (on one validation fold: 44 windows x 4 draws -0.0002 AUC,
12 x 16 -0.0002, 12 x 32 +0.0009). Here the subsets become the ensemble itself: member (seed `f`,
view `v`) sees only the windows of its design, and the average over the members covers them all.

Designs (window = slot `s`, position `p` along the slices of the slot, 0..per-1; `m` = the view,
`f` = the seed, `o_f(s)` = a fixed offset per seed and slot, `o_0(s) = s`):

- `full`: all windows;
- `par`: `(p + o_f(s) + m) % 2 == 0`, ~44 windows (6 or 5 per slot): half the cost;
- `mod3`: `(p + o_f(s) + m) % 3 == 0`, ~29 (4 or 3 per slot);
- `mod4`: `(p + o_f(s) + m) % 4 == 0`, ~22 (3, 3, 3 or 2 per slot): a quarter of the cost; the
  four views of a seed cover every window once;
- `modQxR`: `(p + o_f(s) + m) % Q < R`, R/Q of the windows (`mod3x2` ~59, `mod4x3` 66): the steps
  between `par` and `full`;
- `scale`: the four sequences at only one of the two crop scales, `(s // (S/2) + m + f) % 2`: 44;
- `rndK`: K random windows per member (the same draw for all studies of a batch), the baseline
  for the grid designs.

In the grid designs every slot has windows in every member: a study with a single acquired slot
is still predicted by all members, and inside each slot the kept windows are evenly spaced along
the slices (one every Q). With the per-seed, per-slot offset every seed covers every window the
same number of times across its views, while different seeds drop different windows: the subset
error is averaged over the seeds too, instead of repeating identically in all of them (with the
same grid for every seed there would be only Q distinct subsets). Seed 0 uses `o_0(s) = s`.
"""
from __future__ import annotations

import numpy as np
import torch

from .models.baseline import window_slot_index
from .tta import Variant, variant_windows

GRID = ("full", "par", "parx", "mod3", "mod4", "scale")


def seed_offsets(seed: int, n_slots: int, q: int) -> torch.Tensor:
    """`o_f(s)`: the offset of the grid of step `q` in each slot for seed `seed`:
    `o_0(s) = s % q`, the other seeds use a fixed permutation of the same values across slots.
    Every value appears the same number of times for every seed, so with `n_slots` a multiple of
    `q` every member has the same number of windows (44 for `par`, 22 for `mod4` at 8 x 11): the
    encoder graphs stay full.
    """
    base = torch.arange(n_slots) % q
    if seed == 0:
        return base
    g = torch.Generator().manual_seed(7919 * int(seed))
    return base[torch.randperm(n_slots, generator=g)]


def member_windows(design: str, member: int, n_slots: int, per: int,
                   gen: torch.Generator | None = None, seed: int = 0) -> torch.Tensor:
    """The (sorted) indices of the windows of member `member` (the view) of seed `seed` in design
    `design`, among the `n_slots * per` windows of the study in slot-major order
    (`make_windows`).
    """
    n = n_slots * per
    if "-d" in design:
        # `<design>-d<k>`: the design without the windows of sequence k (slots k and k + n_slots/2), which the
        # kernel drops with `KNEE_DROP_SLOTS=k`: like `parxdrop<k>` for any design (e.g. `mod3-d1`).
        base, k = design.rsplit("-d", 1)
        k = int(k)
        if not 0 <= k < n_slots // 2:
            raise ValueError(f"{design}: sequence outside 0..{n_slots // 2 - 1}")
        idx = member_windows(base, member, n_slots, per, gen=gen, seed=seed)
        return idx[(idx // per) % (n_slots // 2) != k]
    if design.startswith("rnd"):
        k = int(design[3:])
        if not 0 < k <= n:
            raise ValueError(f"{design}: K outside 1..{n}")
        return torch.randperm(n, generator=gen)[:k].sort().values
    s = torch.arange(n_slots).repeat_interleave(per)
    p = torch.arange(per).repeat(n_slots)
    if design == "full":
        keep = torch.ones(n, dtype=torch.bool)
    elif design == "par" or design.startswith("mod"):
        q, r = (2, 1) if design == "par" else (
            (int(design[3:].split("x")[0]), int(design.split("x")[1]) if "x" in design else 1))
        if not 0 < r <= q:
            raise ValueError(f"{design}: requires 0 < R <= Q")
        keep = (p + seed_offsets(seed, n_slots, q)[s] + member) % q < r
    elif design[:1] == "q" and "f" in design and design[1:].replace("f", "", 1).isdigit():
        # `qQfF`: `(p + s + F * seed + member) % Q == 0`, a linear offset like `parx`. With Q = seeds x views
        # and F = views (e.g. `q8f4`: two seeds x four views) every window goes through the encoder exactly once,
        # in a different (seed, view) member depending on slot and position: TTA and a second seed at the cost of `parx`.
        q, f = (int(v) for v in design[1:].split("f"))
        if q < 1:
            raise ValueError(f"{design}: requires Q >= 1")
        keep = (p + s + f * seed + member) % q == 0
    elif design == "parx":
        # Half of the windows per seed, complementary across seeds: seed 1 keeps, in every slot, the
        # positions that seed 0 drops, so two seeds without TTA cover every window exactly once
        # (88 per study instead of 176). With `par` the per-seed offsets are a permutation across
        # slots and two seeds can keep the same half of a slot. This is the design of the final submission.
        keep = (p + s + seed + member) % 2 == 0
    elif design.startswith("edge") and design.endswith("x"):
        # Windows A..B of every slot split between seeds as in `parx`: with `KNEE_READ_SLICES` set to
        # the slices they touch, the kernel does not read the others (fewer bytes from the input mount).
        lo, hi = (int(v) for v in design[4:-1].split("-"))
        if not 0 <= lo <= hi < per:
            raise ValueError(f"{design}: requires 0 <= A <= B < {per}")
        keep = (p >= lo) & (p <= hi) & ((p + s + seed + member) % 2 == 0)
    elif design.startswith("parxdrop"):
        # `parx` without the windows of sequence k (slots k and k + n_slots/2), which the kernel drops with
        # `KNEE_DROP_SLOTS`: the GPU does not compute windows of empty slots.
        k = int(design[8:])
        if not 0 <= k < n_slots // 2:
            raise ValueError(f"{design}: sequence outside 0..{n_slots // 2 - 1}")
        keep = ((p + s + seed + member) % 2 == 0) & (s % (n_slots // 2) != k)
    elif design == "scale":
        if n_slots % 2:
            raise ValueError("scale: requires both crop scales (even number of slots)")
        keep = (s // (n_slots // 2) + member + seed) % 2 == 0
    else:
        raise ValueError(f"unknown design {design!r} (known: {GRID}, modQxR and rndK)")
    return keep.nonzero().flatten()


def budget_windows(design: str, n_views: int, n_slots: int, per: int, n_seeds: int = 1) -> int:
    """Windows per study computed by the encoder with `n_seeds` seeds x `n_views` views (grid
    designs).
    """
    return sum(int(member_windows(design, v, n_slots, per, seed=f).numel())
               for f in range(n_seeds) for v in range(n_views))


@torch.no_grad()
def predict_budget(model, loader, device, variants: list[Variant], designs: list[str],
                   seed: int = 0, amp: bool = True,
                   dtype: torch.dtype = torch.float16) -> dict[str, np.ndarray]:
    """Probabilities (N, T) per design, with one model and the views `variants` as members
    (`m` = index of the view): the mean over the views of the head on the member's windows.
    Also `view_<name>` = the view alone with all windows (`view_id` is the plain prediction:
    the control). The encoder runs once per view on all windows (in `eval` every window is
    independent); the head is repeated for each design.
    """
    model.eval()
    stride, depth = model.window_stride, model.window_depth
    use_idx = getattr(model, "context_pos", None) is not None
    g = torch.Generator().manual_seed(int(seed))
    names = [f"view_{v.name}" for v in variants] + list(designs)
    out: dict[str, list[np.ndarray]] = {n: [] for n in names}
    on_gpu = device.type == "cuda"
    for vol, mask, *_ in loader:
        vol, mask = vol.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        vol01 = vol.float().div_(255.0)
        acc: dict[str, torch.Tensor] = {}
        for m, v in enumerate(variants):
            w, _ = variant_windows(vol01, v, stride, depth)
            n_slots, n_win = vol.shape[1], w.shape[1]
            per = n_win // n_slots
            slot_all = window_slot_index(n_slots, vol.shape[2], device, stride, depth)
            with torch.autocast("cuda", dtype=dtype, enabled=amp and on_gpu):
                enc = model.encode_windows(w)
                full = torch.sigmoid(model(None, slot_all, mask, encoded=enc).float())
                out[f"view_{v.name}"].append(full.cpu().numpy())
                for d in designs:
                    if d == "full":
                        p = full
                    else:
                        pick = member_windows(d, m, n_slots, per, g).to(device)
                        kw = {"window_index": pick} if use_idx else {}
                        p = torch.sigmoid(model(None, slot_all[pick], mask,
                                                encoded=enc[:, pick], **kw).float())
                    acc[d] = p if d not in acc else acc[d] + p
        for d in designs:
            out[d].append((acc[d] / len(variants)).cpu().numpy())
    return {n: np.concatenate(v) for n, v in out.items()}
