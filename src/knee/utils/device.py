"""Can the GPU Kaggle gave us run this PyTorch build?

Lives here and not in `train` because two different kernels need it: training and submission.
The second is the one that matters, since a crash there costs the score, not an hour of quota.
"""
from __future__ import annotations

import torch


def check_device(device: torch.device) -> str:
    """Can the assigned GPU run the kernels compiled into this PyTorch build?

    Kaggle assigns whatever machine it has: a P100 (sm_60) with a PyTorch built for sm_70 and up
    fails with `no kernel image is available` on the first tensor moved to the device, a message
    that does not name the cause.

    Binary compatibility goes by major version: an sm_80 cubin runs on sm_86, not the other way
    round. So the same major with a minor not above the device's is accepted.
    """
    if device.type != "cuda":
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    arches = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    ok = any(int(a[3:]) // 10 == major and int(a[3:]) % 10 <= minor for a in arches)
    if not ok:
        raise RuntimeError(
            f"{name} is sm_{major}{minor}, and this PyTorch only contains {arches}. "
            f"No CUDA kernel would run. On Kaggle the machine is chosen with "
            f"`machine_shape` in kernel-metadata.json: 'NvidiaTeslaT4' is sm_75.")
    return f"{name} (sm_{major}{minor})"
