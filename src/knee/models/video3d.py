"""3D encoder: a whole slot (24 slices x 288 x 288) enters as one volume (an experiment, not used
by the final model).

The main model is 2.5D: windows of three contiguous slices in the three channels of a 2D encoder,
then per-finding attention over the windows. Depth is mixed LATE, after the encoder. Here it is
mixed from the first layer: the "window" is the whole slot (`window_depth` = slices of the slot,
equal `window_stride`, one window per slot) and the encoder is a video network pre-trained on
Kinetics-400, which also convolves along z. Everything else (slot identity, context, per-finding
attention, head) is unchanged: with 8 slots the "windows" of a study are 8.

Why S3D and not torchvision's 3D ResNets (measured with `torch.utils.flop_counter` on a
24x288x288 volume):
- `s3d` 89 GFLOP, 8.3 M parameters: 8 slots = ~0.7x the encoder of the 2.5D DenseNet169 at
  288 px. Separable convolutions (1,k,k) + (k,1,1), which respect the anisotropy of clinical MRI
  (thick slices, fine pixels).
- `r3d_18` 807, `mc3_18` 860, `r2plus1d_18` 804 GFLOP: 9x S3D, over budget both in training and
  in the submission time limit.

Three choices, part of the experiment:
- **Grey in one channel.** Kinetics weights have three RGB channels; MRI has one. The first conv
  is reduced to one channel by summing the weights over the three inputs: EXACTLY the RGB conv
  applied to the grey image repeated three times, at a third of the cost. No Kinetics
  mean/std: windows enter in [0, 1] as in every other model, and the BatchNorm right after the
  first conv absorbs the shift.
- **BatchNorm momentum 0.1.** torchvision uses 0.001 (the value from the TensorFlow port): the
  running statistics would move by a thousandth per step and after an epoch (~440 steps) would
  still be 64% Kinetics. 0.1 is the value of timm's DenseNet BatchNorm.
- **Global mean over (z, y, x)** instead of torchvision's fixed `AvgPool3d(2, 7, 7)`, which only
  fits 16x224x224: at 24x288x288 the final map is 3x9x9.

Gradient checkpointing (`train.grad_checkpointing`) splits `features` into segments recomputed in
the backward pass: at 24x288x288 one volume holds ~1.7 GB of activations in fp32 (~0.9 in fp16),
and 8 studies x 4 slots per step do not fit a 40 GB A100 without it. Note: during recomputation
BatchNorm in training updates its running statistics a second time with the same batch (like
timm's `checkpoint_seq`): the moving average runs slightly faster, gradients and training output
do not change.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

VIDEO_ARCHS = ("s3d",)
MIN_DEPTH = 8        # S3D halves z three times (stem, pool 3, pool 4): below 8 slices nothing is left
BN_MOMENTUM = 0.1
CKPT_SEGMENTS = 4


def is_video_arch(arch: str) -> bool:
    return arch in VIDEO_ARCHS


def _gray_stem(conv: nn.Conv3d) -> nn.Conv3d:
    """The first RGB conv reduced to one channel: weights summed over the three inputs."""
    out = nn.Conv3d(1, conv.out_channels, conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, dilation=conv.dilation, groups=1,
                    bias=conv.bias is not None)
    with torch.no_grad():
        out.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            out.bias.copy_(conv.bias)
    return out


class VideoEncoder(nn.Module):
    """(N, T, H, W) in [0, 1] -> (N, num_features): one volume per row.

    `in_chans` is the window depth (`window_depth`), i.e. the slices of the volume: for
    `KneeBaseline` a window (N, depth, H, W) has always been "depth channels", here it is "depth
    slices along z". Same interface as timm encoders (`num_features`, `set_grad_checkpointing`), so
    `KneeBaseline` has no separate path.
    """

    def __init__(self, arch: str, pretrained: bool, in_chans: int, drop_rate: float = 0.0):
        super().__init__()
        if not is_video_arch(arch):
            raise ValueError(f"VideoEncoder: arch {arch!r} is not in {VIDEO_ARCHS}")
        if drop_rate:
            raise ValueError("VideoEncoder: drop_rate is not defined (the recipe uses 0)")
        if in_chans < MIN_DEPTH:
            raise ValueError(f"VideoEncoder: {in_chans} slices per volume, at least "
                             f"{MIN_DEPTH} are needed (S3D halves z three times)")
        from torchvision.models.video import S3D_Weights, s3d
        net = s3d(weights=S3D_Weights.KINETICS400_V1 if pretrained else None)
        feats = net.features
        stem = feats[0][0]                       # Conv3dNormActivation (1, 7, 7) with stride (1, 2, 2)
        stem[0] = _gray_stem(stem[0])
        for m in feats.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.momentum = BN_MOMENTUM
        self.features = feats
        self.depth = int(in_chans)
        self.num_features = 1024
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = bool(enable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.depth:
            raise ValueError(f"VideoEncoder: expected (N, {self.depth}, H, W), got {tuple(x.shape)}")
        x = x.unsqueeze(1)                       # (N, 1, T, H, W)
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            z = checkpoint_sequential(self.features, CKPT_SEGMENTS, x, use_reentrant=False)
        else:
            z = self.features(x)
        return z.mean(dim=(2, 3, 4))


def channels_last_ok(models) -> bool:
    """`channels_last` is a 4D layout: on 5D weights `Module.to` raises "required rank 4 tensor".
    False if at least one model has a video encoder.
    """
    return not any(isinstance(getattr(m, "encoder", None), VideoEncoder) for m in models)
