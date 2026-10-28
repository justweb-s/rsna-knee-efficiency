"""The model: a 2D encoder over windows + attention pooling, one per finding.

Three decisions, in order of how much they matter.

**Separate attention per finding.** A study arrives as a bag of windows (sequence slots x the
slices of each) and must be reduced to twelve numbers. With a single shared pooling weight the
twelve findings compete for the same slices: an ACL tear is visible on two images, osteoarthritis is
spread over many, and a single pooling forces them to choose. Giving every finding its own attention
costs one (d x 12) matrix and removes the constraint. Replacing it with the mean of the windows
cost 0.004 on the public leaderboard and 0.007 on the private one.

**Three adjacent slices in the three channels.** The model sees some of what lies above and below
the central slice: much of the benefit of a 3D model at the cost of a 2D one. It makes sense *only*
because the cache orders the slices physically: with a file-name order the three channels would be
three unrelated views (see `data/geometry.py`).

**Absent slots are masked, not zeroed.** A never-acquired slot is zero in the cache; if it entered the
attention softmax with a finite logit, the model would spread weight over a black image. Here its
windows are excluded before the softmax.

**`pooling: mean` is the control of the first decision.** With a linear head, the mean of the
features followed by the classifier is identical to the mean of the per-window logits: exactly
"predict every window and average". It is obtained from the same path as the attention with all
logits at zero (uniform softmax over the valid windows), so masks, absent slots and the anatomical
prior stay the same bit for bit; with `pooling: mean` the `attn` matrix does not exist.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import TARGETS
from .intra import IntraSlicePool

GROUP = 3  # slices per window, stacked into the encoder's three channels

# Pure ViTs carry the resolution inside the architecture: the positional embedding has one row per
# patch, so `img_size` is not a detail of the data loader but a constructor parameter.
# `vit_small_patch14_dinov2` is born at 518 px and with 224 px windows dies on an `assert` before
# seeing a gradient. Passing it makes timm resample the positional embedding when loading the weights.
#
# CNNs instead **reject** the kwarg with a TypeError (resnet34, densenet121, convnext_nano), so it
# cannot always be passed: only to those that accept it. The prefix is the discriminant timm itself
# uses to separate the families.
_VIT_PREFIXES = ("vit_", "deit", "eva", "beit", "samvit", "flexivit")
# timm's MaxxVit hybrids (CoAtNet, MaxViT) are on the ViT side: the attention of stages 3-4 has a table
# of relative positions built at the constructor's resolution. `coatnet_rmlp_2_rw_384` without
# `img_size` is born at 384 (24x24 tokens at stage 3) and cannot handle 288 px windows; with
# `img_size=288` it is born at 18x18, and the 384 state_dict still loads strictly, because the tables
# are non-persistent buffers and the positions are computed by an MLP.
_HYBRID_PREFIXES = ("coatnet", "coatnext", "maxvit", "maxxvit")


def _wants_img_size(arch: str) -> bool:
    """If the encoder is a ViT or a MaxxVit hybrid, `img_size` goes into the constructor; on CNNs it raises
    TypeError.
    """
    return arch.startswith(_VIT_PREFIXES + _HYBRID_PREFIXES)


# `input_norm: imagenet`: the per-channel mean and std the ImageNet weights were trained with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def encode_chunk_for(arch: str, n_windows: int) -> int:
    """How many windows per encoder forward in evaluation and inference (0 = all).

    A submission batch of 8 studies x 88 windows = 704 images at 288 px fits a T4 with a DenseNet, but
    not with larger encoders. In blocks of one study (`n_windows`) the peak drops 8 times and the numbers
    do not change: in `eval` BatchNorm uses the accumulated statistics and LayerNorm is per sample, so
    splitting the batch touches no output. DenseNets stay at 0, the usual path.
    """
    return 0 if arch.startswith("densenet") else int(n_windows)


def make_encoder(arch: str, pretrained: bool, drop_rate: float, in_chans: int = GROUP,
                 spatial_pool: str = "avg", **extra) -> nn.Module:
    """The encoder: from timm, except DenseNet121/169 at inference (`densenet.BY_ARCH`).

    In the submission kernel every import on a cold page cache is paid for (`import timm` ~9.5 s, ~6.8 s
    of which is torchvision, pulled in by timm). `DenseNet121` imports only torch and is the same network
    as `timm.create_model("densenet121", num_classes=0)`: same state_dict keys, bit-identical output; the
    same holds for `DenseNet169`. With `pretrained=True` (training) or `drop_rate`, timm's ImageNet
    weights and dropout are needed, so timm is used; the import is lazy for the same reason.
    """
    if arch in ("densenet121", "densenet169") and not pretrained and not drop_rate and not extra:
        from .densenet import BY_ARCH
        return BY_ARCH[arch](in_chans=in_chans, spatial_pool=spatial_pool)
    from .video3d import VideoEncoder, is_video_arch
    if is_video_arch(arch):
        # 3D experiment: the window is a volume, `in_chans` are its slices along z.
        if spatial_pool != "avg" or extra:
            raise ValueError(f"{arch}: video encoder only with spatial_pool 'avg' and no "
                             f"extra arguments, got {spatial_pool!r} {extra}")
        return VideoEncoder(arch, pretrained, in_chans=in_chans, drop_rate=drop_rate)
    import timm
    if spatial_pool in ("gem", "map"):
        # GeM: the encoder returns the map (B, C, H, W) and the generalised mean is done by `KneeBaseline.gem`.
        # With `global_pool=""` timm skips pooling and flatten; `num_features` stays the number of channels.
        # "map" is the same map, for per-finding spatial attention.
        if arch.startswith(_VIT_PREFIXES):
            raise ValueError(f"spatial_pool={spatial_pool!r} applies to CNNs: a ViT has no map")
        extra = {**extra, "global_pool": ""}
    # With `in_chans != 3` timm adapts the ImageNet first conv by itself (repeats the three channels and
    # rescales): what a 5-slice window needs.
    return timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=in_chans,
                             drop_rate=drop_rate, **extra)


class GeM(nn.Module):
    """Generalised mean over the feature map (Radenovic et al., 2018).

    `(mean_{h,w} x^p)^{1/p}` per channel, with learnable `p` (3.0 at init, the paper's reference
    choice). At `p = 1` it is the plain mean of timm and of `DenseNet121`, at `p -> inf` the maximum: it
    sits in between and lets the few strong positions the mean dilutes emerge (a fracture line or a
    meniscal lesion occupy few points of a 9x9 map at 288 px). The clamp to `eps` is for the fractional
    power: the map is post-ReLU (>= 0) both in timm and in the hand-written DenseNet, so nothing
    informative is cut.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), float(p)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"GeM wants a map (B, C, H, W), got {tuple(x.shape)}")
        # In float32 even under autocast: x^p with p ~3 on values ~10 overflows half precision.
        x = x.float().clamp(min=self.eps)
        return x.pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


def windows_per_slot(n_slices: int, stride: int = 1, depth: int = GROUP) -> int:
    """How many windows of `depth` slices with stride `stride` fit in a slot of `n_slices`.

    It is the `unfold` formula: with stride 1 there are `n_slices - depth + 1` (8 for 10 slices, 14 for
    16), with stride 2 over 16 slices 7 ([0-2] [2-4] ... [12-14], covering slices 0..14 with an overlap
    of one). The stride separates "a window sees three contiguous slices" (the third dimension) from "the
    attention has more windows to vote over". The depth is how many contiguous slices a window sees, i.e.
    the encoder channels. The final model uses 24 slices, stride 2, depth 3: 11 windows per slot.
    """
    if stride < 1:
        raise ValueError(f"window_stride must be >= 1, got {stride}")
    if depth < 1:
        raise ValueError(f"window_depth must be >= 1, got {depth}")
    if n_slices < depth:
        raise ValueError(f"at least {depth} slices per slot are needed, got {n_slices}")
    return (n_slices - depth) // stride + 1


def make_windows(vol: torch.Tensor, stride: int = 1, depth: int = GROUP) -> torch.Tensor:
    """(B, S, K, H, W) -> (B, S*n_win, depth, H, W), sliding windows over the slices.

    `n_win = windows_per_slot(K, stride, depth)`; the order is slot-major, so window `i` comes from slot
    `i // n_win`, as `window_slot_index` says.
    """
    b, s, k, h, w = vol.shape
    windows_per_slot(k, stride, depth)         # validates k, stride and depth
    win = vol.unfold(2, depth, stride)         # (B, S, n_win, H, W, depth)
    win = win.permute(0, 1, 2, 5, 3, 4)        # (B, S, n_win, depth, H, W)
    return win.reshape(b, s * win.shape[2], depth, h, w)


def window_slot_index(n_slots: int, n_slices: int, device=None, stride: int = 1,
                      depth: int = GROUP) -> torch.Tensor:
    """Which slot each window comes from. Needed for masking and for the slot embedding."""
    per = windows_per_slot(n_slices, stride, depth)
    return torch.arange(n_slots, device=device).repeat_interleave(per)


# Table of clinical anatomical priors per sequence (4 base slots), an option not used by the final model:
# Rows: 0: Axial FS, 1: Sagittal T1, 2: Coronal FS, 3: Sagittal FS
# Columns: the 12 targets in TARGETS ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
#           "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
#           "Contusion", "Fracture")
BASE_4_SLOT_PRIOR: list[list[float]] = [
    # 0: Axial FS (Fluid_Sensitive=1, Fat_Suppression=1)
    # ACL   MCL  M.Men  L.Men   M.OA   L.OA  PF.OA    Eff    Syn   Bake   Cont   Frac
    [-1.0,  0.0, -2.0, -2.0,  -1.0,  -1.0,   1.5,   1.0,   1.0,   1.5,   0.5,   0.5],
    # 1: Sagittal T1 (Fluid_Sensitive=0, Fat_Suppression=0)
    [ 1.0, -2.0,  1.0,  1.0,   0.5,   0.5,   0.5,  -1.0,  -1.0,  -2.0,   0.0,   0.5],
    # 2: Coronal FS (Fluid_Sensitive=1, Fat_Suppression=1)
    [ 0.5,  1.5,  1.0,  1.0,   1.0,   1.0,  -2.0,   0.5,   0.5,  -2.0,   1.0,   1.0],
    # 3: Sagittal FS (Fluid_Sensitive=1, Fat_Suppression=1)
    [ 1.0, -2.0,  1.0,  1.0,   0.5,   0.5,   0.5,   1.0,   1.0,   1.5,   1.0,   0.5],
]


def build_slot_prior(n_slots: int) -> torch.Tensor:
    """Builds the anatomical prior matrix (n_slots, 12).

    With 4 slots: clinical prior on plane and contrast.
    With 8 slots (two scales: 0..3 at 110 mm, 4..7 at 150 mm):
      - 110 mm: favours structures with micro-tears (menisci).
      - 150 mm: favours structures with peripheral extent (coronal MCL, axial/sagittal Baker's cyst).
    """
    p4 = torch.tensor(BASE_4_SLOT_PRIOR, dtype=torch.float32)
    if n_slots == 4:
        return p4
    if n_slots == 8:
        p_110 = p4.clone()
        p_110[:, 2] += 0.5  # Medial Meniscus: fine resolution
        p_110[:, 3] += 0.5  # Lateral Meniscus: fine resolution

        p_150 = p4.clone()
        p_150[2, 1] += 0.5  # MCL on Coronal FS at 150 mm: medial/proximal extent
        p_150[0, 9] += 0.5  # Baker's on Axial FS at 150 mm: distended popliteal bursa
        p_150[3, 9] += 0.5  # Baker's on Sagittal FS at 150 mm: distal extension into the calf
        return torch.cat([p_110, p_150], dim=0)

    # Neutral fallback
    return torch.zeros((n_slots, len(TARGETS)), dtype=torch.float32)


def load_encoder_weights(encoder: nn.Module, arch: str, path) -> None:
    """Overwrites the ImageNet encoder weights with those of our own pre-training (an option, not used by
    the final model).

    `pretrained=True` brings ImageNet, this brings ImageNet + a pass over external knee MRIs. The file is
    a `.pt` with `{"arch": ..., "encoder": state_dict}` written by the pre-training notebook, with the
    encoder already at `in_chans=GROUP` and `num_classes=0`: the same keys as `timm.create_model` above,
    so loading is `strict`: a file of another architecture or with the head inside must stop the run,
    not half-load. The head (attention, `slot_emb`, `cls_w`) is not touched: it stays initialised from
    scratch.

    The second accepted format is Dread Development's **Raptor classifier** (CC0-1.0 weights):
    `{"model", "arch", "res", "lab"}`, with the encoder under the `backbone.` prefix and its attention
    head next to it (`norm`, `att`, `clsW`, `clsb`), which is discarded here: our head has another shape
    and starts from scratch as always. `arch` in the checkpoint carries the tag of the original timm
    weights (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`): the name before the dot counts.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(encoder_state_from_checkpoint(ckpt, arch, path), strict=True)


RAPTOR_PREFIX = "backbone."


def encoder_state_from_checkpoint(ckpt, arch: str, path="checkpoint") -> dict:
    """The encoder state_dict from a weights file, in the two formats of `load_encoder_weights`.

    Separate from loading so that a preflight can report what is about to enter the encoder (how many
    tensors, from which epoch) before the run.
    """
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected a dict, got {type(ckpt).__name__}")
    if "encoder" in ckpt:
        if ckpt.get("arch") != arch:
            raise ValueError(f"{path}: weights of '{ckpt.get('arch')}', the config asks for '{arch}'")
        return ckpt["encoder"]
    model = ckpt.get("model")
    if isinstance(model, dict) and any(k.startswith(RAPTOR_PREFIX) for k in model):
        # Without `arch` in the file (the author's inference treats it as optional) the check is left to the
        # strict load: another architecture does not have the same keys.
        src = str(ckpt.get("arch") or "")
        if src and src.split(".")[0] != arch:
            raise ValueError(f"{path}: Raptor checkpoint of '{src}', the config asks for "
                             f"'{arch}'")
        return {k[len(RAPTOR_PREFIX):]: v for k, v in model.items() if k.startswith(RAPTOR_PREFIX)}
    raise ValueError(f"{path}: expected {{'arch', 'encoder'}} or a Raptor classifier "
                     f"({{'model': {{'{RAPTOR_PREFIX}...'}}, 'arch'}}), got keys "
                     f"{list(ckpt)[:5]}")


def sincos_2d(h: int, w: int, dim: int, device=None) -> torch.Tensor:
    """Fixed 2D position (h*w, dim): half of the channels for the row, half for the column, sine and cosine
    at geometric frequencies. No parameters, valid at any resolution.
    """
    if dim % 4:
        raise ValueError(f"sincos_2d: dim {dim} not divisible by 4")
    q = dim // 4
    freq = 1.0 / (10000 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    ay, ax = ys.reshape(-1, 1) * freq, xs.reshape(-1, 1) * freq
    return torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], dim=1)


SLICE_ATTN_AT = "denseblock4"


class WindowMapAttention(nn.Module):
    """Attention across the windows of a study on the intermediate encoder maps (an experiment, not used by
    the final model).

    The map of every window before `denseblock4` (9x9, 640 channels for the 169 at 288 px) is split into
    tokens, one per position; all tokens of all windows of the study go through a transformer
    (bottleneck `dim`, pre-norm, dropout 0) and the result returns to the map as a residual, before the
    last convolutional layers. Every token knows which slot it comes from (`slot`), its height in the
    stack (`pos`, the window inside the slot) and its point on the map (2D sine/cosine). The output
    projection starts at ZERO: at the first step the model is the parent bit for bit. Unlike the window
    context (one vector per window, after the encoder), here attention is between image points and the
    DenseNet keeps working on the result.
    """

    def __init__(self, channels: int, dim: int, heads: int, layers: int, n_slots: int, per: int):
        super().__init__()
        self.dim = int(dim)
        self.inp = nn.Linear(channels, self.dim)
        self.slot = nn.Parameter(torch.zeros(n_slots, self.dim))
        self.pos = nn.Parameter(torch.zeros(per, self.dim))
        nn.init.trunc_normal_(self.slot, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(self.dim, int(heads), dim_feedforward=2 * self.dim,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, int(layers), enable_nested_tensor=False)
        self.out = nn.Linear(self.dim, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, m: torch.Tensor, b: int, slot_of_window: torch.Tensor,
                pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """m (b*w, C, H, W) -> same shape; `valid` (b, w): windows of absent slots are not keys (a study without
        valid slots lets everything through).
        """
        n, c, h, wd = m.shape
        w, hw = n // b, h * wd
        t = m.flatten(2).transpose(1, 2).reshape(b, w, hw, c)
        emb = (self.slot[slot_of_window] + self.pos[pos]).unsqueeze(1) \
            + sincos_2d(h, wd, self.dim, m.device).unsqueeze(0)            # (w, hw, dim)
        x = (self.inp(t) + emb.unsqueeze(0).to(t.dtype)).reshape(b, w * hw, self.dim)
        pad = (~valid).unsqueeze(-1).expand(b, w, hw).reshape(b, w * hw)
        pad = pad & ~pad.all(dim=1, keepdim=True)
        y = self.out(self.tf(x, src_key_padding_mask=pad))                 # (b, w*hw, C)
        return m + y.reshape(n, hw, c).transpose(1, 2).reshape(n, c, h, wd).to(m.dtype)


class FindingContext(nn.Module):
    """Attention across the 12 findings, after per-finding pooling and before the classifier (an
    experiment, not used by the final model).

    Up to here every finding has its own vector (B, T, d), produced by its attention over the windows,
    and its own product with `cls_w`: the Synovitis logit does not know what the Effusion attention
    found. Here the T vectors go through a small transformer (bottleneck `dim`, one embedding per
    finding, pre-norm, dropout 0) and the result returns as a residual, so rare columns can take evidence
    from those they co-occur with (Effusion and Synovitis, ACL and Contusion, Medial Meniscus and Medial
    OA). The output projection starts at ZERO: at the first step the model is the parent bit for bit.
    """

    def __init__(self, d: int, k: int, dim: int, heads: int, layers: int):
        super().__init__()
        self.inp = nn.Linear(d, int(dim))
        self.emb = nn.Parameter(torch.zeros(k, int(dim)))
        nn.init.trunc_normal_(self.emb, std=0.02)
        layer = nn.TransformerEncoderLayer(int(dim), int(heads), dim_feedforward=2 * int(dim),
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, int(layers), enable_nested_tensor=False)
        self.out = nn.Linear(int(dim), d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled (B, T, d) -> same shape."""
        h = self.inp(pooled) + self.emb.unsqueeze(0).to(pooled.dtype)
        return pooled + self.out(self.tf(h)).to(pooled.dtype)


class AuxHead(nn.Module):
    """The head of the auxiliary findings (an experiment): the same form as the per-study output of the 12
    columns (one score per window and per finding, softmax over the valid windows, weighted mean of the
    features, product with one vector per finding), with its own parameters. So every sub-finding learns
    WHERE to look.
    """

    def __init__(self, d: int, k: int):
        super().__init__()
        self.attn = nn.Linear(d, k)
        self.cls_w = nn.Parameter(torch.zeros(k, d))
        self.cls_b = nn.Parameter(torch.zeros(k))
        nn.init.trunc_normal_(self.cls_w, std=0.02)

    def forward(self, f: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """f (B, W, d) the per-window features, valid (B, W) the windows of the present slots."""
        a = self.attn(f)                                      # (B, W, K)
        v = valid.unsqueeze(-1)
        a = a.masked_fill(~v, float("-inf"))
        # A study without valid slots would give NaN in the softmax (as for the main output).
        dead = ~v.any(dim=1, keepdim=True).expand_as(a)
        a = torch.where(dead, torch.zeros_like(a), a).softmax(dim=1)
        pooled = torch.einsum("bwk,bwd->bkd", a, f)          # (B, K, d)
        return (pooled * self.cls_w).sum(-1) + self.cls_b    # (B, K)


# Compartment routing (an experiment): the columns that read a single tibiofemoral compartment. After
# `normalise_laterality` every knee is a left knee: in coronal and axial images the medial side is in
# the LEFT half of the image (the row runs towards patient +x in every coronal and axial series of the
# training set), in sagittal ones towards the HIGH stack indices (normal towards -x in every sagittal
# series).
ROUTE_SIDES = (("Medial OA", "medial"), ("Lateral OA", "lateral"))


class KneeBaseline(nn.Module):
    def __init__(self, arch: str = "resnet34", n_slots: int = 4, n_slices: int = 10,
                 pretrained: bool = True, drop_rate: float = 0.0,
                 img_size: int | None = None,
                 slot_prior_strength: float = 0.0,
                 slot_prior_mask_at: float | None = None,
                 encoder_weights: str | None = None,
                 window_stride: int = 1,
                 window_depth: int = GROUP,
                 context_layers: int = 0,
                 context_dim: int = 256,
                 context_heads: int = 4,
                 context_pos: str | None = None,
                 pooling: str = "attention",
                 attn_hidden: int = 128,
                 spatial_attn: str | None = None,
                 encoder_scope: str = "shared",
                 spatial_pool: str = "avg",
                 report_dim: int = 0,
                 input_norm: str | None = None,
                 grad_checkpointing: bool = False,
                 slice_attn_layers: int = 0,
                 slice_attn_dim: int = 256,
                 slice_attn_heads: int = 4,
                 aux_targets: int = 0,
                 compartment_route: str | None = None,
                 finding_layers: int = 0,
                 finding_dim: int = 256,
                 finding_heads: int = 4,
                 drop_path_rate: float = 0.0,
                 vit_head: str | None = None,
                 img_crop: int | None = None,
                 img_resize: int | None = None,
                 encoder_override: nn.Module | None = None):
        super().__init__()
        # Upscaling experiment: the cache windows enlarged on the GPU to `img_resize` px (bilinear, pixel
        # centres as in `cv2.INTER_LINEAR`) before the encoder. At 512 px the 110 mm crop is already finer than
        # the native resolution in 90% of the series: beyond that the cache brings no new detail and only the
        # network's input scale changes (640 px did not help). DenseNet with the whole encoder only.
        self.img_resize = int(img_resize) if img_resize else None
        if self.img_resize is not None:
            if not arch.startswith("densenet"):
                raise ValueError(f"img_resize applies to DenseNets, got {arch}")
            if img_crop or vit_head is not None or compartment_route is not None \
                    or int(slice_attn_layers) > 0:
                raise ValueError("img_resize with img_crop, vit_head, compartment_route or "
                                 "slice_attn: untested paths")
        # Centre crop of the windows before the encoder (a 288 px cache is not a multiple of patch 14:
        # 280 = 20 x 14). ViTs only, which carry the resolution in the constructor; the removed border is 4 px
        # per side (~1.5 mm over 110 mm).
        self.img_crop = int(img_crop) if img_crop else None
        if self.img_crop is not None:
            if not arch.startswith(_VIT_PREFIXES):
                raise ValueError(f"img_crop applies to ViTs, got {arch}")
            if not img_size or self.img_crop > int(img_size):
                raise ValueError(f"img_crop {self.img_crop} beyond the cache image {img_size}")
            if compartment_route is not None:
                raise ValueError("img_crop with compartment_route: the halves are cut on the whole image")
            img_size = self.img_crop
        if vit_head not in (None, "intra"):
            raise ValueError(f"vit_head: expected None or 'intra', got {vit_head!r}")
        # `vit_head` was named after the ViT; it also applies to the DenseNet, where the "patches" are the
        # cells of the final map (9x9 at 288 px) and the "CLS" is their mean.
        is_vit = arch.startswith(_VIT_PREFIXES)
        if vit_head is not None:
            if not (is_vit or arch.startswith("densenet")):
                raise ValueError(f"vit_head {vit_head!r} needs a ViT (patch tokens) or a "
                                 f"DenseNet (map cells), got {arch}")
            if (pooling != "attention" or spatial_attn is not None or spatial_pool != "avg"
                    or encoder_scope != "shared" or int(slice_attn_layers) > 0
                    or compartment_route is not None):
                raise ValueError("vit_head needs 'attention' pooling on a shared encoder, "
                                 "without spatial_attn, spatial_pool, slice_attn or compartment_route")
        if int(finding_layers) > 0 and compartment_route is not None:
            raise ValueError("finding_layers with compartment_route: Medial/Lateral OA would come out "
                             "of `_route_compartments` without going through the attention across findings")
        if pooling not in ("attention", "mean", "gated"):
            raise ValueError(f"pooling: expected 'attention', 'mean' or 'gated', got {pooling!r}")
        if compartment_route not in (None, "oa"):
            raise ValueError(f"compartment_route: expected None or 'oa', got {compartment_route!r}")
        if compartment_route is not None:
            if (pooling != "attention" or spatial_attn is not None or spatial_pool != "avg"
                    or encoder_scope != "shared" or int(slice_attn_layers) > 0
                    or int(aux_targets) > 0 or int(report_dim) > 0):
                raise ValueError("compartment_route needs 'attention' pooling on a shared encoder "
                                 "with the usual mean, without spatial_attn, "
                                 "slice_attn, report_dim or aux_targets")
            if n_slots not in (4, 8):
                raise ValueError("compartment_route needs the 4 anatomical slots (or the two scales, 8): "
                                 f"the plane is read from slot % 4, got {n_slots} slots")
        if spatial_attn not in (None, "target"):
            raise ValueError(f"spatial_attn: expected None or 'target', got {spatial_attn!r}")
        if spatial_attn is not None and (spatial_pool != "avg" or encoder_scope != "shared"):
            raise ValueError("spatial_attn needs spatial_pool 'avg' and a shared encoder: "
                             "it reduces the map itself")
        if context_pos not in (None, "window"):
            raise ValueError(f"context_pos: expected None or 'window', got {context_pos!r}")
        if context_pos is not None and int(context_layers) == 0:
            raise ValueError("context_pos without context (context_layers 0): there is no "
                             "position to assign")
        if input_norm not in (None, "imagenet"):
            raise ValueError(f"input_norm: expected None or 'imagenet', got {input_norm!r}")
        if input_norm is not None and window_depth != 3:
            raise ValueError("input_norm='imagenet' has three channels (R, G, B): with window_depth "
                             f"{window_depth} there is no per-channel mean to use")
        if spatial_pool not in ("avg", "gem"):
            raise ValueError(f"spatial_pool: expected 'avg' or 'gem', got {spatial_pool!r}")
        if encoder_scope not in ("shared", "plane"):
            raise ValueError("encoder_scope: expected 'shared' or 'plane', "
                             f"got {encoder_scope!r}")
        if encoder_scope == "plane" and n_slots not in (4, 8):
            raise ValueError("encoder_scope='plane' requires the 4 default anatomical slots "
                             f"(or the two scales, 8 slots), got {n_slots}")
        if encoder_scope == "plane" and encoder_weights is not None:
            raise ValueError("encoder_weights is not defined for three per-plane encoders")
        self.pooling = pooling
        self.encoder_scope = encoder_scope
        # How the encoder reduces the map to one vector per window. With "avg" it is the usual global mean,
        # inside the encoder, and no extra module exists (state_dict identical to the plain model). With "gem"
        # the encoder returns the map and `self.gem` reduces it, a single parameter `gem.p`.
        self.spatial_pool = spatial_pool
        self.n_slots, self.n_slices = n_slots, n_slices
        # Window stride and depth live on the model, not in the data loader: whoever builds the windows
        # (train, validation, submission) reads them from here, and since the `cfg` travels in the checkpoint,
        # weights trained with stride 2 or 5 channels cannot be queried with stride 1 or 3 channels (the depth
        # is also the shape of the first conv).
        windows_per_slot(n_slices, window_stride, window_depth)   # validates the triple right away
        self.window_stride = int(window_stride)
        self.window_depth = int(window_depth)
        self.n_targets = len(TARGETS)
        self.slot_prior_strength = float(slot_prior_strength)
        # `slot_prior_mask_at` is the NON-absorbable alternative to `slot_prior_strength`. An additive prior
        # can be cancelled by the model through `slot_emb` (the only resistance is `weight_decay`); a -inf mask
        # before the softmax cannot be cancelled by any parameter.
        self.slot_prior_mask_at = (None if slot_prior_mask_at is None
                                   else float(slot_prior_mask_at))
        extra = {"img_size": img_size} if (img_size and _wants_img_size(arch)) else {}
        # ViT experiment: timm's stochastic depth (linear ramp from 0 on the first block to `drop_path_rate`
        # on the last). Training only; the state_dict does not change.
        if float(drop_path_rate) > 0.0:
            extra["drop_path_rate"] = float(drop_path_rate)
        # Intra-window head: the encoder returns all tokens (prefixes + patches) through `forward` (timm with
        # `global_pool=""`: the same tensor as `forward_features`, which `train.speed.compile` compiles);
        # `self.intra` reduces them. Same state_dict keys. On the DenseNet the same thing is the map
        # (N, C, H, W), `spatial_pool="map"` (timm `global_pool=""`, or `densenet.DenseNet`).
        self.vit_head = vit_head
        self.intra_on_map = vit_head is not None and not is_vit
        if vit_head is not None and is_vit:
            extra["global_pool"] = ""
        # Per-finding spatial attention: the encoder returns the map and `_encode_once` derives from it both
        # the usual global mean and the local vector per finding.
        self.spatial_attn = spatial_attn
        self.encoder = (encoder_override if encoder_override is not None else
                        make_encoder(arch, pretrained, drop_rate, in_chans=self.window_depth,
                                     spatial_pool="map" if (spatial_attn or self.intra_on_map)
                                     else spatial_pool, **extra))
        if spatial_pool == "gem":
            self.gem = GeM()
        self.n_prefix = int(getattr(self.encoder, "num_prefix_tokens", 0)) if vit_head else 0
        if encoder_weights is not None:
            load_encoder_weights(self.encoder, arch, encoder_weights)
        # The input normalisation the weights expect. By default windows enter the encoder in [0, 1] without
        # mean or std: for CNNs with BatchNorm right after the stem it is a shift the first BN absorbs, for an
        # encoder trained on normalised inputs it is another distribution. With `None` no buffer exists and
        # the output is bit-identical; the buffers are non-persistent, so the state_dict does not change
        # shape.
        self.input_norm = input_norm
        if input_norm == "imagenet":
            self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
                                 persistent=False)
            self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
                                 persistent=False)
        # Windows per encoder forward when the model is in `eval` (0 = all together): not a hyperparameter and
        # not in the cfg, set by whoever evaluates or infers with `encode_chunk_for`. Never split in training:
        # BatchNorm would see different batch statistics.
        self.encode_chunk = 0
        d = self.encoder.num_features
        # Per-plane encoders (an experiment): the axial one stays in `encoder` to keep the compilation and
        # inference paths compatible; sagittal and coronal have independent copies. Every window goes through
        # ONE DenseNet only: resident parameters triple, the number of images processed does not. Attention and
        # classifier stay shared, so the only variable is encoder sharing.
        self.plane_encoders = nn.ModuleDict()
        if self.encoder_scope == "plane":
            # Building two more modules consumes RNG even when the ImageNet weights overwrite them. The state is
            # restored: slot_emb, shared attention and head start from the same numbers as the parent; only
            # encoder sharing changes.
            rng = torch.random.get_rng_state()
            try:
                self.plane_encoders["sagittal"] = make_encoder(
                    arch, pretrained, drop_rate, in_chans=self.window_depth,
                    spatial_pool=spatial_pool, **extra)
                self.plane_encoders["coronal"] = make_encoder(
                    arch, pretrained, drop_rate, in_chans=self.window_depth,
                    spatial_pool=spatial_pool, **extra)
            finally:
                torch.random.set_rng_state(rng)
            dims = {d, *(m.num_features for m in self.plane_encoders.values())}
            if len(dims) != 1:
                raise ValueError(f"the per-plane encoders have different sizes: {dims}")
        # Gradient checkpointing of the encoder (`train.grad_checkpointing`): the stages recompute their
        # activations in the backward pass instead of keeping them, same function and same gradients, ~30% more
        # compute. It is what fits CoAtNet-2 at 96 windows per step on a 40 GB A100. It does not touch the
        # state_dict. Note: CoAtNet-RMLP has its own dropout, 0.125 inside the relative-position MLP (timm
        # `RelPosMlp`), active in training with or without this option: the recomputation reproduces it exactly
        # because `checkpoint_seq` preserves the RNG state (`preserve_rng_state`), so function and gradients
        # stay the same.
        self.grad_checkpointing = bool(grad_checkpointing)
        if self.grad_checkpointing:
            for enc in (self.encoder, *self.plane_encoders.values()):
                if not hasattr(enc, "set_grad_checkpointing"):
                    raise ValueError(f"grad_checkpointing: encoder {type(enc).__name__} "
                                     "has no set_grad_checkpointing")
                enc.set_grad_checkpointing(True)
        # Slot identity is added to the features: the same image acquired as sagittal T1 or as coronal STIR
        # does not mean the same thing, and without this the attention cannot tell them apart.
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, d))
        # With `pooling: mean` the matrix is not created: the state_dict says so, and a checkpoint trained with
        # the mean cannot be read as if it had an attention.
        if self.pooling == "attention":
            self.attn = nn.Linear(d, self.n_targets)
        # Gated attention (an experiment): the score of a window per finding goes through a gated hidden layer
        # (Ilse et al. 2018, attention-based deep MIL): `w · [tanh(V f) ⊙ sigmoid(U f)]`, instead of the plain
        # linear projection `attn`. The RNG state is restored and consumed like the linear `attn`: `cls_w`,
        # `slot_emb` and the context start from the same numbers as the parent, only the scoring head changes.
        if self.pooling == "gated":
            h = int(attn_hidden)
            rng = torch.random.get_rng_state()
            try:
                self.attn_V = nn.Linear(d, h)
                self.attn_U = nn.Linear(d, h)
                self.attn_w = nn.Linear(h, self.n_targets)
            finally:
                torch.random.set_rng_state(rng)
            nn.Linear(d, self.n_targets)        # consumes the RNG like `attn`, then is discarded
        # Per-finding spatial attention (an experiment): inside every window, twelve attention maps (one per
        # finding, 1x1 conv on the encoder map) reduce the map to a local vector per finding, added to the
        # global mean with a per-finding weight that starts at ZERO: at the first step the model computes the
        # parent's function. Modules built with the restored RNG, like `report_proj`.
        if self.spatial_attn is not None:
            rng = torch.random.get_rng_state()
            try:
                self.sp_attn = nn.Conv2d(d, self.n_targets, kernel_size=1)
            finally:
                torch.random.set_rng_state(rng)
            self.sp_scale = nn.Parameter(torch.zeros(self.n_targets))
        # Report distillation (an experiment): the auxiliary head that predicts the residual of the report
        # embedding (`knee.reports`) from the mean of the windows. It exists ONLY in training and only if the
        # config asks for it: with `report_dim = 0` there is no module and the state_dict is identical to the
        # plain model. `train.py` removes its keys from the checkpoint before saving anyway, because at
        # inference the model is rebuilt from the `model:` block of the cfg alone and `load_state_dict` is
        # strict. The RNG state is saved and restored, as for `plane_encoders`: building an extra module
        # consumes random numbers and would shift the initialisation of `cls_w`/`slot_emb`/`attn`.
        self.report_dim = int(report_dim)
        if self.report_dim > 0:
            rng = torch.random.get_rng_state()
            try:
                self.report_proj = nn.Linear(d, self.report_dim)
            finally:
                torch.random.set_rng_state(rng)
        # Auxiliary findings (an experiment): K more findings (site and grade from the report,
        # `knee.labels.aux_findings`), each with ITS OWN attention over the windows, on the same features as
        # the 12 columns. It exists only in training and only if the config asks for it: with
        # `aux_targets = 0` no module, state_dict and output of the plain model. `train.strip_aux` removes
        # `aux_head.*` from the checkpoint. Built with the restored RNG, like `report_proj`.
        self.aux_targets = int(aux_targets)
        if self.aux_targets > 0:
            if self.pooling != "attention" or self.spatial_attn is not None:
                raise ValueError("aux_targets needs 'attention' pooling without spatial_attn: "
                                 "the auxiliary head is a second per-finding attention")
            rng = torch.random.get_rng_state()
            try:
                self.aux_head = AuxHead(d, self.aux_targets)
            finally:
                torch.random.set_rng_state(rng)
        # Intra-window pooling from Matteo Vitali's v61m head (`models.intra`). The "plane" of the per-finding
        # maps is the cache slot. Built with the restored RNG, like `report_proj`. On the DenseNet map two of
        # his defaults are brought back to the grid: his is 24x24 at ~5.4 mm per patch (336 px, 130 mm), ours
        # 9x9 at 12-17 mm per cell. top-k 24 out of 576 (4%) -> 4 out of 81 (5%); the 3x3 blur of the logits,
        # which shares evidence between neighbouring patches, on cells 2-3 times larger (with a receptive
        # field that already covers them) would smear the lesion over the compartment: off.
        if self.vit_head == "intra":
            rng = torch.random.get_rng_state()
            try:
                self.intra = (IntraSlicePool(d, self.n_targets, n_slots, k=4, blur=False)
                              if self.intra_on_map else IntraSlicePool(d, self.n_targets, n_slots))
            finally:
                torch.random.set_rng_state(rng)
        self.cls_w = nn.Parameter(torch.zeros(self.n_targets, d))
        self.cls_b = nn.Parameter(torch.zeros(self.n_targets))
        nn.init.trunc_normal_(self.cls_w, std=0.02)
        nn.init.trunc_normal_(self.slot_emb, std=0.02)

        # Context across windows. Without it every window leaves the encoder as an independent vector: none
        # knows what is in the neighbouring window, none knows its height in the stack, and the per-finding
        # attention weighs vectors that never talked to each other. With `context_layers > 0` the windows of
        # the study go through a small transformer (bottleneck `context_dim`, position embedding along the
        # slices within the slot, key mask for the windows of absent slots) and the result returns to the
        # features as a residual. The output projection starts at ZERO: at the first step the model computes
        # exactly the parent's function, and the context is learned from there. With `context_layers = 0` no
        # module exists and the output is bit-identical. The final model uses one layer of 256 dimensions and
        # 4 heads.
        self.context_layers = int(context_layers)
        # Where the position along the slices comes from. None = the default rule, `arange(W) % per`: the real
        # position only if ALL windows arrive in order (inference, validation). In training `k_train` sampled
        # windows arrive in random order, and the default rule gives each one its place in the draw, which says
        # nothing about its position. "window" = the position is `window_index % per`, with `window_index` the
        # global index of the window passed by the caller; without it, only all windows in order are valid,
        # where the two rules coincide. No extra parameters: the state_dict and the inference output are the
        # usual ones.
        self.context_pos = context_pos
        if self.context_layers > 0:
            cd = int(context_dim)
            per = windows_per_slot(n_slices, self.window_stride, self.window_depth)
            self.ctx_in = nn.Linear(d, cd)
            self.ctx_pos = nn.Parameter(torch.zeros(per, cd))
            nn.init.trunc_normal_(self.ctx_pos, std=0.02)
            layer = nn.TransformerEncoderLayer(cd, int(context_heads), dim_feedforward=2 * cd,
                                               dropout=0.0, batch_first=True, norm_first=True)
            # `enable_nested_tensor=False`: the nested-tensor path rewrites the masked rows and changes the numbers
            # between train and eval; here the same computation is wanted in both modes.
            self.ctx = nn.TransformerEncoder(layer, self.context_layers,
                                             enable_nested_tensor=False)
            self.ctx_out = nn.Linear(cd, d)
            nn.init.zeros_(self.ctx_out.weight)
            nn.init.zeros_(self.ctx_out.bias)

        # Attention across windows on the maps before `denseblock4` (`WindowMapAttention`). With 0 layers no
        # module: state_dict and output of the plain model. DenseNet only (it needs the cut point) with a shared
        # encoder and the map reduced by the usual mean. Built with the restored RNG, like `report_proj`.
        self.slice_attn_layers = int(slice_attn_layers)
        if self.slice_attn_layers > 0:
            if not arch.startswith("densenet"):
                raise ValueError(f"slice_attn: DenseNet only (cut at {SLICE_ATTN_AT}), got {arch}")
            if encoder_scope != "shared" or spatial_pool != "avg" or spatial_attn is not None \
                    or grad_checkpointing:
                raise ValueError("slice_attn needs a shared encoder, spatial_pool 'avg', "
                                 "no spatial_attn and no grad_checkpointing")
            names = [n for n, _ in self.encoder.features.named_children()]
            k = names.index(SLICE_ATTN_AT)
            ch = self.encoder.features[k].denselayer1.norm1.num_features
            rng = torch.random.get_rng_state()
            try:
                self.slice_attn = WindowMapAttention(
                    ch, int(slice_attn_dim), int(slice_attn_heads), self.slice_attn_layers,
                    n_slots, windows_per_slot(n_slices, self.window_stride, self.window_depth))
            finally:
                torch.random.set_rng_state(rng)

        # Attention across findings on the per-finding vectors (`FindingContext`). With 0 layers no module:
        # state_dict and output of the plain model. Built with the restored RNG, like `report_proj`, and with
        # the output projection at zero the first step is the parent bit for bit.
        self.finding_layers = int(finding_layers)
        if self.finding_layers > 0:
            rng = torch.random.get_rng_state()
            try:
                self.finding_ctx = FindingContext(d, self.n_targets, int(finding_dim),
                                                  int(finding_heads), self.finding_layers)
            finally:
                torch.random.set_rng_state(rng)

        prior = build_slot_prior(n_slots)
        self.register_buffer("slot_prior", prior, persistent=False)
        # The clinically allowed slots for each finding. The -2.0 of the table is the value "this sequence is
        # wrong for this finding" (Baker's on Sag T1, PF OA on Coronal, MCL on the sagittals): masking at -2.0
        # uses only those clear-cut cells and leaves the graded values alone. Note: with `n_slots` outside
        # (4, 8) `build_slot_prior` returns zeros, so the mask becomes a no-op instead of switching everything
        # off.
        allowed = (prior > self.slot_prior_mask_at if self.slot_prior_mask_at is not None
                   else torch.ones_like(prior, dtype=torch.bool))
        self.register_buffer("slot_allowed", allowed, persistent=False)

        # Compartment routing: Medial OA and Lateral OA are computed on their own bags of windows, from their
        # compartment only (`_route_compartments`). No extra parameters: same attention and classifier rows,
        # same context. Precisely for this the path would leave no trace in the state_dict, and a model built
        # without the key would silently load the weights while computing another function. The PERSISTENT
        # buffer `route_oa` prevents that: a strict `load_state_dict` fails both ways (unexpected or missing
        # key). Float, so EMA shadows average it and stay at 1.
        self.compartment_route = compartment_route
        if compartment_route is not None:
            self.register_buffer("route_oa", torch.ones(1))

    def _encode(self, enc: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """One vector per window: the encoder, plus GeM if the map comes out whole.

        With `input_norm` the windows in [0, 1] are first brought to the weights' distribution; with
        `encode_chunk` and the model in `eval` the encoder sees `encode_chunk` windows at a time (see
        `encode_chunk_for`).
        """
        if self.img_crop is not None:
            o0 = (x.shape[-2] - self.img_crop) // 2
            o1 = (x.shape[-1] - self.img_crop) // 2
            x = x[..., o0:o0 + self.img_crop, o1:o1 + self.img_crop]
        if self.img_resize is not None and x.shape[-2:] != (self.img_resize, self.img_resize):
            x = F.interpolate(x, size=(self.img_resize, self.img_resize), mode="bilinear",
                              align_corners=False)
        if self.input_norm is not None:
            x = (x - self.in_mean.to(x.dtype)) / self.in_std.to(x.dtype)
        n = self.encode_chunk
        if n and not self.training and x.shape[0] > n:
            return torch.cat([self._encode_once(enc, c) for c in x.split(n)])
        return self._encode_once(enc, x)

    def _encode_sliced(self, windows: torch.Tensor, slot_of_window: torch.Tensor,
                       valid: torch.Tensor, window_index: torch.Tensor | None) -> torch.Tensor:
        """The encoder cut before `denseblock4`, with `slice_attn` in between: (B*W, d).

        The encoder is always taken in its eager form (`_orig_mod`): the compiled paths and the AOTI packages
        wrap the whole encoder and have no cut. The position along the slices is the real one:
        `window_index` if present (training, sampled windows), otherwise all windows in order.
        """
        b, w = windows.shape[:2]
        x = windows.flatten(0, 1)
        if self.input_norm is not None:
            x = (x - self.in_mean.to(x.dtype)) / self.in_std.to(x.dtype)
        enc = getattr(self.encoder, "_orig_mod", self.encoder)
        names = [n for n, _ in enc.features.named_children()]
        k = names.index(SLICE_ATTN_AT)
        per = self.slice_attn.pos.shape[0]
        if window_index is not None:
            pos = window_index.to(x.device).remainder(per)
        else:
            if w != self.n_slots * per:
                raise ValueError(f"slice_attn: {w} windows out of {self.n_slots * per} without "
                                 "window_index, the position along the slices cannot be derived")
            pos = torch.arange(w, device=x.device) % per
        z = enc.features[:k](x)
        z = self.slice_attn(z, b, slot_of_window.to(x.device), pos, valid)
        z = enc.features[k:](z)
        if hasattr(enc, "forward_head"):            # timm: `norm5` is BatchNormAct2d
            return enc.forward_head(z)
        return torch.flatten(F.adaptive_avg_pool2d(F.relu(z), (1, 1)), 1)

    def _encode_once(self, enc: nn.Module, x: torch.Tensor) -> torch.Tensor:
        z = enc(x)
        if self.spatial_pool == "gem":
            z = self.gem(z).to(z.dtype)
        elif self.spatial_attn is not None:
            z = self._target_local(z)
        return z

    def _target_local(self, m: torch.Tensor) -> torch.Tensor:
        """Per-finding spatial attention: from the map (N, C, H, W) the global mean (N, C) and, per finding, the
        mean weighted by its attention map (N, T, C). They come out concatenated as (N, C + T*C), so the
        chunking and `reshape` of `_features` stay the same; `_features_full` separates them. Softmax over
        positions in float32.
        """
        a = self.sp_attn(m).flatten(2).float().softmax(-1).to(m.dtype)   # (N, T, HW)
        g = torch.einsum("nth,nch->ntc", a, m.flatten(2))                 # (N, T, C)
        return torch.cat([m.mean((2, 3)), g.flatten(1)], dim=1)

    def _features(self, windows: torch.Tensor, slot_of_window: torch.Tensor,
                  slot_mask: torch.Tensor, window_index: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-window features (B, W, d) and the mask (B, W): `_features_full` without the local
        per-finding vector.
        """
        f, valid, _ = self._features_full(windows, slot_of_window, slot_mask, window_index)
        return f, valid

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """The encoder output per window, (B, W, e), without slot or context: what `_features_full` accepts as
        `encoded`. Shared encoder only. In `eval` every window goes through the encoder alone (BatchNorm with
        the accumulated statistics), so the features of a subset are rows of those computed on all windows:
        a diagnostic can compute them once and repeat only the head.
        """
        if self.encoder_scope != "shared":
            raise ValueError("encode_windows: only with encoder_scope 'shared'")
        if self.vit_head is not None:
            raise ValueError("encode_windows: with vit_head the encoder outputs tokens, not vectors")
        if self.slice_attn_layers > 0:
            raise ValueError("encode_windows: with slice_attn the windows talk to each other inside "
                             "the encoder, the features of a subset are not rows of the whole")
        b, w = windows.shape[:2]
        return self._encode(self.encoder, windows.flatten(0, 1)).reshape(b, w, -1)

    def _features_full(self, windows: torch.Tensor | None, slot_of_window: torch.Tensor,
                       slot_mask: torch.Tensor, window_index: torch.Tensor | None = None,
                       encoded: torch.Tensor | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The per-window features (B, W, d), with slot identity and optional context, the mask (B, W) of the
        windows of present slots and, with `spatial_attn`, the local per-finding vector (B, W, T, d);
        otherwise None.

        `window_index` (W,) is the global index of each window (that of `window_slot_index`); only the
        context with `context_pos: window` reads it, the others ignore it.

        It is the part of the forward that precedes pooling, shared between the per-study output (`forward`)
        and the per-window one (`window_logits`): the two cannot diverge.

        The encoder runs on all windows, including those of empty slots (black images, masked to -inf
        afterwards). Skipping them would be bit-exact but does not pay off: slots are 97% full and the T4 is
        compute-bound.
        """
        g = None
        if encoded is not None and self.vit_head is not None:
            raise ValueError("_features_full: `encoded` is not defined with vit_head")
        if encoded is not None:
            b, w = encoded.shape[:2]
            f = encoded
        elif self.vit_head is not None:
            b, w = windows.shape[:2]
            f, g = self._intra(windows, slot_of_window)
        elif self.slice_attn_layers > 0:
            b, w = windows.shape[:2]
            f = self._encode_sliced(windows, slot_of_window, slot_mask[:, slot_of_window],
                                    window_index).reshape(b, w, -1)
        elif self.encoder_scope == "shared":
            b, w = windows.shape[:2]
            f = self._encode(self.encoder, windows.flatten(0, 1)).reshape(b, w, -1)
        else:
            b, w = windows.shape[:2]
            # Slots 0/1/2/3 are Axial/Sagittal/Coronal/Sagittal; with two scales 4..7 repeat the same order.
            # `slot_of_window` can be a random, unordered subset during training, so routing is done from the
            # index rather than assuming contiguous blocks.
            plane = slot_of_window.remainder(4)
            routes = ((self.encoder, plane == 0),
                      (self.plane_encoders["sagittal"], (plane == 1) | (plane == 3)),
                      (self.plane_encoders["coronal"], plane == 2))
            parts = []
            for enc, take in routes:
                idx = take.nonzero(as_tuple=False).flatten()
                if idx.numel():
                    parts.append((idx, self._encode(enc, windows[:, idx].flatten(0, 1))
                                  .reshape(b, idx.numel(), -1)))
            # The buffer takes the encoder's dtype, not the windows': under autocast the features are half and
            # `index_copy` refuses a copy between different dtypes.
            f = parts[0][1].new_zeros((b, w, self.encoder.num_features))
            for idx, z in parts:
                f = f.index_copy(1, idx, z)
        valid = slot_mask[:, slot_of_window]                # (B, W)
        if self.spatial_attn is not None:
            c = self.encoder.num_features
            f, g = f[..., :c], f[..., c:].reshape(b, w, self.n_targets, c)
        f = f + self.slot_emb[slot_of_window].unsqueeze(0)

        if self.context_layers > 0:
            f = self._context(f, valid, window_index)
        return f, valid, g

    def _intra(self, windows: torch.Tensor, slot_of_window: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """From the ViT tokens the shared vector (B, W, d) and the per-finding offset (B, W, T, d). The windows
        are flattened study by study, so the slot of window j of study i is `slot_of_window[j]`.

        On the DenseNet the patches are the cells of the map (rows in h, w order: the grid the blur expects)
        and instead of the CLS there is their mean, i.e. the vector of the usual global mean: at step 0 the
        model is the parent.
        """
        b, w = windows.shape[:2]
        t = self._encode(self.encoder, windows.flatten(0, 1))           # (b*w, prefixes + P, d)
        plane = slot_of_window.to(t.device).repeat(b)
        if self.intra_on_map:
            n, c = t.shape[:2]
            patch = t.permute(0, 2, 3, 1).reshape(n, -1, c)             # (b*w, H*W, C)
            tok, delta = self.intra(patch, patch.mean(1), plane)
        else:
            tok, delta = self.intra(t[:, self.n_prefix:], t[:, 0], plane)
        return tok.reshape(b, w, -1), delta.reshape(b, w, self.n_targets, -1)

    def _context(self, f: torch.Tensor, valid: torch.Tensor,
                 window_index: torch.Tensor | None) -> torch.Tensor:
        """The context across windows on the features (B, W, d): `valid` (B, W) says which windows act as keys.
        Used by the usual output and, with `compartment_route`, by the single-compartment bags, where the
        windows of the other side are not keys.
        """
        w = f.shape[1]
        per = self.ctx_pos.shape[0]
        if self.context_pos == "window" and window_index is not None:
            pos = window_index.to(f.device).remainder(per)
        else:
            if self.context_pos == "window" and w != self.slot_emb.shape[0] * per:
                raise ValueError(f"context_pos 'window': {w} windows out of "
                                 f"{self.slot_emb.shape[0] * per} without window_index, "
                                 "the position along the slices cannot be derived")
            pos = torch.arange(w, device=f.device) % per    # position along the slices
        h = self.ctx_in(f) + self.ctx_pos[pos].unsqueeze(0)
        # Windows of absent slots are not keys: a never-acquired slot must not influence the output. A study
        # with no valid slot would give an all-masked row (NaN in the inner softmax): everything is let
        # through and the row is zeroed further down as usual.
        pad = ~valid
        pad = pad & ~pad.all(dim=1, keepdim=True)
        return f + self.ctx_out(self.ctx(h, src_key_padding_mask=pad))

    def _route_compartments(self, logits: torch.Tensor, windows: torch.Tensor,
                            encoded: torch.Tensor, slot_of_window: torch.Tensor,
                            slot_mask: torch.Tensor, window_index: torch.Tensor | None
                            ) -> torch.Tensor:
        """Compartment routing: Medial OA and Lateral OA recomputed on their compartment only.

        For each of the two columns its own bag of windows, aligned with the study's windows: in coronal and
        axial images the half of the image on its side (288 x 144 at 288 px), through the same encoder; in
        sagittal ones the whole windows on its side of the stack, the central one on neither (stride 2 over
        24 slices: 5 lateral, window 5 in the centre, 5 medial). The cut is on the image and not on the
        encoder map: on the 9x9 map the medial half moves by 29% when only the lateral half of the input
        changes. The knee sits in the centre of the box: median -2 mm, 90% within half a map column.
        Then slot identity and context as on the other features, but the context sees only windows of the
        same side: nothing arrives from the other compartment. Attention and classifier are the column's
        usual rows. If in a training step the window draw leaves nothing on one side, that study keeps the
        usual logit. The other ten columns are untouched.

        `encoded` (B, W, d): the encoder output on the whole windows, i.e. the features before slot and
        context (also those inference computes skipping empty slots).
        """
        b, w, d = encoded.shape
        dev = encoded.device
        per = windows_per_slot(self.n_slices, self.window_stride, self.window_depth)
        sow = slot_of_window.to(dev)
        if window_index is not None:
            pos = window_index.to(dev).remainder(per)
        else:
            if w != self.n_slots * per:
                raise ValueError(f"compartment_route: {w} windows out of {self.n_slots * per} without "
                                 "window_index, the side of the sagittal windows cannot be derived")
            pos = torch.arange(w, device=dev) % per
        plane = sow.remainder(4)                       # 0 axial, 1 sag T1, 2 coronal, 3 sag FS
        inplane = (plane == 0) | (plane == 2)          # both compartments are in the image
        towards = {"medial": 2 * pos > per - 1, "lateral": 2 * pos < per - 1}
        width = windows.shape[-1]
        half = width // 2
        cols = {"medial": slice(0, half), "lateral": slice(width - half, width)}
        ip = inplane.nonzero(as_tuple=False).flatten()
        valid = slot_mask.to(dev)[:, sow]              # (B, W)
        out, live = [], []
        for name, side in ROUTE_SIDES:
            t = TARGETS.index(name)
            sag = ~inplane & towards[side]
            tok = encoded.new_zeros((b, w, d))
            if ip.numel():
                x = windows[:, ip.to(windows.device)][..., cols[side]]
                z = self._encode(self.encoder, x.flatten(0, 1).contiguous())
                tok = tok.index_copy(1, ip, z.reshape(b, ip.numel(), -1).to(tok.dtype))
            sg = sag.nonzero(as_tuple=False).flatten()
            if sg.numel():
                tok = tok.index_copy(1, sg, encoded[:, sg])
            keep = valid & (inplane | sag).unsqueeze(0)  # (B, W)
            tok = tok + self.slot_emb[sow].unsqueeze(0)
            if self.context_layers > 0:
                tok = self._context(tok, keep, window_index)
            a = F.linear(tok, self.attn.weight[t:t + 1], self.attn.bias[t:t + 1])[..., 0]
            if self.slot_prior_strength > 0.0:
                a = a + self.slot_prior_strength * self.slot_prior[sow, t].unsqueeze(0)
            allowed = keep
            if self.slot_prior_mask_at is not None:
                allowed = keep & self.slot_allowed[sow, t].unsqueeze(0)
                allowed = torch.where(~allowed.any(dim=1, keepdim=True), keep, allowed)
            ok = allowed.any(dim=1)                     # (B,) the side has at least one window
            a = a.masked_fill(~allowed, float("-inf"))
            a = torch.where(ok.unsqueeze(1), a, torch.zeros_like(a)).softmax(dim=1)
            pooled = torch.einsum("bw,bwd->bd", a, tok)
            out.append((pooled * self.cls_w[t]).sum(-1) + self.cls_b[t])
            live.append(ok)
        idx = torch.tensor([TARGETS.index(n) for n, _ in ROUTE_SIDES], device=logits.device)
        routed = torch.stack(out, dim=1).to(logits.dtype)
        routed = torch.where(torch.stack(live, dim=1), routed, logits.index_select(1, idx))
        return logits.index_copy(1, idx, routed)

    def window_logits(self, windows: torch.Tensor, slot_of_window: torch.Tensor,
                      slot_mask: torch.Tensor, window_index: torch.Tensor | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """One logit per window and per finding, (B, W, T), plus the mask (B, W) of valid windows.

        It is the linear head applied to every window instead of to their mean. With `pooling: mean` the
        mean of these logits over the valid windows equals `forward` to the decimal (the head is linear), so
        nothing changes at inference; only WHERE the error is measured in training changes (`train.loss:
        window`): every valid window answers for the study target, instead of all answering together through
        the mean. With attention it makes no sense, because at inference the windows do not weigh 1/n and a
        per-window head would not be the model's head: it is refused. Windows of absent slots come out with a
        logit (black image) the caller must ignore through the mask.
        """
        if self.pooling != "mean":
            raise ValueError("window_logits: only makes sense with pooling 'mean' (at inference "
                             "the output is the mean of the per-window logits); here pooling "
                             f"is {self.pooling!r}")
        if self.slot_prior_mask_at is not None:
            raise ValueError("window_logits: the per-finding anatomical mask "
                             "(`slot_prior_mask_at`) is not defined per window")
        if self.spatial_attn is not None:
            raise ValueError("window_logits: the local per-finding vector (`spatial_attn`) "
                             "is not defined per window")
        if self.finding_layers > 0:
            raise ValueError("window_logits: the attention across findings (`finding_layers`) works "
                             "on per-study vectors, it is not defined per window")
        if self.vit_head is not None:
            raise ValueError("window_logits: with vit_head every finding has its own vector per "
                             "window, the per-window head is not defined")
        f, valid = self._features(windows, slot_of_window, slot_mask, window_index)
        return f @ self.cls_w.t() + self.cls_b, valid       # (B, W, T), (B, W)

    def forward(self, windows: torch.Tensor, slot_of_window: torch.Tensor,
                slot_mask: torch.Tensor, return_report: bool = False,
                window_index: torch.Tensor | None = None,
                encoded: torch.Tensor | None = None, return_aux: bool = False):
        """windows (B, W, depth, H, W_) float, slot_of_window (W,), slot_mask (B, S) bool.
        One logit per study and per finding, (B, T): the pooling over the valid windows.

        With `return_report=True` it also returns the vector of the report-distillation head,
        (B, report_dim) normalised: the plain mean of the valid windows (not the per-finding pooling),
        projected and normalised to 1, so that the cosine with the target is a dot product. With
        `return_report=False` the computation is bit-identical to the usual one.

        With `return_aux=True` it also returns the logits of the auxiliary findings, (B, aux_targets), from
        their attention over the same features. The 12 logits are bit for bit those of `return_aux=False`:
        the extra head reads `f` and does not touch it.

        With `compartment_route` Medial OA and Lateral OA come out of `_route_compartments`; with sampled
        windows (training) `window_index` is needed, to know which side of the stack every sagittal window
        is on.
        """
        if return_aux and return_report:
            raise ValueError("return_aux and return_report together: two auxiliary heads, "
                             "no run asks for both")
        if return_aux and self.aux_targets <= 0:
            raise ValueError("return_aux=True on a model without auxiliary findings "
                             "(aux_targets = 0)")
        if self.compartment_route is not None:
            # The encoder output on the whole windows is needed twice: for the usual ten columns (passed as
            # `encoded`, which skips the encoder inside `_features_full`) and for the per-compartment bags, which
            # reuse the sagittal ones. The coronal and axial halves are cropped from the windows, which are
            # therefore needed even with `encoded`.
            if windows is None:
                raise ValueError("compartment_route: the windows are needed, the coronal and "
                                 "axial halves are cropped from them")
            if encoded is None:
                encoded = self._encode(self.encoder, windows.flatten(0, 1)).reshape(
                    windows.shape[0], windows.shape[1], -1)
        f, valid, g = self._features_full(windows, slot_of_window, slot_mask, window_index,
                                          encoded)
        b, w = f.shape[:2]

        h = None
        if self.vit_head is not None:
            # Every finding has its own per-window vector, `f` + its offset; from it come both the window's score
            # and what gets summed. At step 0 the offset is zero and the computation is that of
            # `pooling: attention` on `f`.
            h = f.unsqueeze(2) + g.to(f.dtype)              # (B, W, T, d)
            a = torch.einsum("bwtd,td->bwt", h, self.attn.weight) + self.attn.bias
        elif self.pooling == "attention":
            a = self.attn(f)                               # (B, W, T)
        elif self.pooling == "gated":
            a = self.attn_w(torch.tanh(self.attn_V(f)) * torch.sigmoid(self.attn_U(f)))
        else:
            # Plain mean: all logits equal, the softmax below becomes 1/n over the valid windows. Same path of masks
            # and priors as the attention.
            a = f.new_zeros(b, w, self.n_targets)
        if self.slot_prior_strength > 0.0:
            a = a + self.slot_prior_strength * self.slot_prior[slot_of_window].unsqueeze(0)

        valid = valid.unsqueeze(-1)                         # (B, W, 1)
        if self.slot_prior_mask_at is None:
            a = a.masked_fill(~valid, float("-inf"))
        else:
            # (1, W, T): which windows are anatomically allowed, per finding.
            anat = self.slot_allowed[slot_of_window].unsqueeze(0)
            allowed = valid & anat                          # (B, W, T)
            # If for a study and a finding anatomy and acquisition leave no window, **acquisition wins**: fall back
            # to availability alone instead of producing an all -inf row. On the 8-slot cache this never triggers
            # (0 studies out of 4,407 and 0 out of 12 columns, because the axial slot has 100% coverage and is
            # allowed for every finding except the menisci, which have the other three sequences), but the test set
            # has no such guarantee, and a column vanishing at inference would cost the submission.
            empty = ~allowed.any(dim=1, keepdim=True)       # (B, 1, T)
            allowed = torch.where(empty, valid.expand_as(allowed), allowed)
            a = a.masked_fill(~allowed, float("-inf"))
        # A study with no valid slot does not exist in the corpus (the axial slot has 100% coverage), but a
        # softmax over an all -inf row would give NaNs propagating into the gradient of the whole batch: it is
        # neutralised here.
        dead = ~valid.any(dim=1, keepdim=True).expand_as(a)
        a = torch.where(dead, torch.zeros_like(a), a)
        a = a.softmax(dim=1)

        if h is not None:
            pooled = torch.einsum("bwt,bwtd->btd", a, h)    # (B, T, d)
        else:
            pooled = torch.einsum("bwt,bwd->btd", a, f)    # (B, T, d)
        if g is not None and h is None:
            # The local vector of finding t, weighted by the same attention over the windows.
            pooled = pooled + self.sp_scale.to(pooled.dtype).unsqueeze(-1) * \
                torch.einsum("bwt,bwtd->btd", a.to(g.dtype), g)
        if self.finding_layers > 0:
            pooled = self.finding_ctx(pooled)               # the findings talk to each other
        logits = (pooled * self.cls_w).sum(-1) + self.cls_b  # (B, T)
        if self.compartment_route is not None:
            logits = self._route_compartments(logits, windows, encoded, slot_of_window,
                                              slot_mask, window_index)
        if return_aux:
            return logits, self.aux_head(f, valid[..., 0])
        if not return_report:
            return logits
        if self.report_dim <= 0:
            raise ValueError("return_report=True on a model without the auxiliary head "
                             "(report_dim = 0)")
        # Mean over the present windows only: a study with missing slots must not dilute its vector with the
        # black images.
        keep = valid.to(f.dtype)                            # (B, W, 1)
        mean = (f * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        r = self.report_proj(mean.to(self.report_proj.weight.dtype))
        return logits, r / r.norm(dim=-1, keepdim=True).clamp(min=1e-6)
