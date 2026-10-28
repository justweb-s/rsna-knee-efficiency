"""DenseNet121 and DenseNet169 written by hand, for the submission kernel.

They are the networks of `timm.create_model("densenet121" | "densenet169", num_classes=0)`, which
in turn are torchvision's: same modules, same state_dict names
(`features.denseblock1.denselayer1.norm1.weight`, ...), same sequence of operations, bit-identical
output. Only the number of layers per block changes. It exists because in the submission kernel
every import on a cold page cache is paid for: `import timm` costs ~9.5 s and
`import torchvision` alone ~6.8 s (it loads `torchvision._C`, all models, `io`). This file
imports only torch. Training still uses timm, which brings the ImageNet weights and dropout:
here `pretrained=False` and the weights come from the checkpoint.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

GROWTH_RATE = 32
BN_SIZE = 4
INIT_FEATURES = 64
BLOCK_CONFIG = (6, 12, 24, 16)          # densenet121
BLOCK_CONFIGS = {"densenet121": BLOCK_CONFIG, "densenet169": (6, 12, 32, 32)}


class _DenseLayer(nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(n_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_in, BN_SIZE * GROWTH_RATE, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(BN_SIZE * GROWTH_RATE)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(BN_SIZE * GROWTH_RATE, GROWTH_RATE, kernel_size=3, stride=1,
                               padding=1, bias=False)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(inputs, 1)
        x = self.conv1(self.relu1(self.norm1(x)))
        return self.conv2(self.relu2(self.norm2(x)))


class _DenseBlock(nn.ModuleDict):
    def __init__(self, n_layers: int, n_in: int) -> None:
        super().__init__()
        for i in range(n_layers):
            self.add_module(f"denselayer{i + 1}", _DenseLayer(n_in + i * GROWTH_RATE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.values():
            feats.append(layer(feats))
        return torch.cat(feats, 1)


class _Transition(nn.Sequential):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__(OrderedDict([
            ("norm", nn.BatchNorm2d(n_in)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv", nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, bias=False)),
            ("pool", nn.AvgPool2d(kernel_size=2, stride=2)),
        ]))


class DenseNet(nn.Module):
    """`forward` returns the global features (B, C): final BN, ReLU, spatial mean; C = 1024 for the
    121, 1664 for the 169.

    With `spatial_pool="gem"` the mean is not taken here: `forward` returns the post-ReLU map
    (B, 1024, H, W) and the generalised pooling lives in `KneeBaseline` (`baseline.GeM`), as for the
    timm encoder built with `global_pool=""`. The state_dict keys do not change.
    """

    def __init__(self, in_chans: int = 3, spatial_pool: str = "avg",
                 block_config: tuple[int, ...] = BLOCK_CONFIG) -> None:
        super().__init__()
        if spatial_pool not in ("avg", "gem", "map"):
            raise ValueError(f"spatial_pool: expected 'avg', 'gem' or 'map', got {spatial_pool!r}")
        self.spatial_pool = spatial_pool
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(in_chans, INIT_FEATURES, kernel_size=7, stride=2, padding=3,
                                bias=False)),
            ("norm0", nn.BatchNorm2d(INIT_FEATURES)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        n = INIT_FEATURES
        for i, n_layers in enumerate(block_config):
            self.features.add_module(f"denseblock{i + 1}", _DenseBlock(n_layers, n))
            n += n_layers * GROWTH_RATE
            if i != len(block_config) - 1:
                self.features.add_module(f"transition{i + 1}", _Transition(n, n // 2))
                n //= 2
        self.features.add_module("norm5", nn.BatchNorm2d(n))
        self.num_features = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = F.relu(self.features(x), inplace=True)
        if self.spatial_pool != "avg":
            # "gem" and "map": return the map, the pooling happens outside.
            return f                                  # (B, C, H, W)
        return torch.flatten(F.adaptive_avg_pool2d(f, (1, 1)), 1)


class DenseNet121(DenseNet):
    def __init__(self, in_chans: int = 3, spatial_pool: str = "avg") -> None:
        super().__init__(in_chans, spatial_pool, BLOCK_CONFIGS["densenet121"])


class DenseNet169(DenseNet):
    def __init__(self, in_chans: int = 3, spatial_pool: str = "avg") -> None:
        super().__init__(in_chans, spatial_pool, BLOCK_CONFIGS["densenet169"])


BY_ARCH = {"densenet121": DenseNet121, "densenet169": DenseNet169}


def _try_fused(x: torch.Tensor) -> bool:
    """The fused operator exists only on GPU (the tests fake it here)."""
    return x.is_cuda


class _ConvBiasReLU(nn.Module):
    """`conv1 -> norm2 -> relu2` of a `_DenseLayer` in eval, as a 1x1 convolution with bias followed
    by ReLU: BatchNorm with frozen statistics is a per-channel affine and folds into the weights of
    the preceding convolution (`fold_bn`).

    On GPU it runs as a single cudnn operator (`torch.cudnn_convolution_relu`): the memory pass of
    `norm2` and the one of `relu2` over the 128 intermediate channels disappear. Folding alone would
    not pay off, because in PyTorch the bias of a convolution is added by a separate kernel that
    takes the place of the BatchNorm. On the first GPU call the fused operator is compared with the
    three separate operators on the same weights; if it is missing, fails or does not match, the
    three are kept (`fused` = False).
    """

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        super().__init__()
        assert conv.kernel_size == (1, 1) and conv.stride == (1, 1) and conv.bias is None
        scale = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
        w = conv.weight.double() * scale[:, None, None, None]
        b = bn.bias.double() - bn.running_mean.double() * scale
        self.weight = nn.Parameter(w.to(conv.weight.dtype), requires_grad=False)
        self.bias = nn.Parameter(b.to(conv.weight.dtype), requires_grad=False)
        self.fused: bool | None = None          # None = not yet tested on GPU

    def _unfused(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(F.conv2d(x, self.weight, self.bias), inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _try_fused(x) or self.fused is False:
            return self._unfused(x)
        try:
            y = torch.cudnn_convolution_relu(x, self.weight, self.bias, (1, 1), (0, 0),
                                             (1, 1), 1)
        except RuntimeError:
            self.fused = False
            return self._unfused(x)
        if self.fused is None:
            # The comparison on a few images: the convolution is per image, and the whole batch in
            # float32 would cost GiB right on the first batch.
            k = min(4, x.shape[0])
            ref, got = self._unfused(x[:k]).float(), y[:k].float()
            tol = 1e-2 * float(ref.abs().max()) + 1e-3
            self.fused = bool(y.shape[1:] == ref.shape[1:] and bool(torch.isfinite(got).all())
                              and float((got - ref).abs().max()) <= tol)
            if not self.fused:
                return self._unfused(x)
        return y


def fold_bn(model: nn.Module) -> int:
    """Folds `norm2` into `conv1` in every `_DenseLayer` of `model` (in eval: the BatchNorm statistics
    must be the frozen ones). `norm2` and `relu2` become `nn.Identity`, so `_DenseLayer.forward`
    does not change. Returns how many layers.
    """
    n = 0
    for layer in model.modules():
        if isinstance(layer, _DenseLayer) and isinstance(layer.norm2, nn.BatchNorm2d):
            layer.conv1 = _ConvBiasReLU(layer.conv1, layer.norm2)
            layer.norm2, layer.relu2 = nn.Identity(), nn.Identity()
            n += 1
    return n


def fold_bn_status(models: list) -> dict:
    """How many folded layers run fused on GPU, how many on the three operators, how many never
    tested (not tested on CPU).
    """
    out = {"fused": 0, "unfused": 0, "untried": 0}
    for m in models:
        for mod in m.modules():
            if isinstance(mod, _ConvBiasReLU):
                out[{True: "fused", False: "unfused", None: "untried"}[mod.fused]] += 1
    return out
