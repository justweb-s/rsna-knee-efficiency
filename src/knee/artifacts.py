"""Names of the per-run inference artifacts: `torch.compile` caches and AOTI packages.

Kept in a module without torch or cv2 so that the packaging scripts can compute them with the
system interpreter, where OpenCV is not installed.
"""
from __future__ import annotations


def stride_suffix(stride: int | None) -> str:
    """The marker of artifacts built with a different stride (`-stride4`); empty = the run's own."""
    return f"-stride{int(stride)}" if stride is not None else ""


def aoti_dir_name(run: str, stride: int | None = None) -> str:
    return f"{run}{stride_suffix(stride)}-aoti"


def aoti_study_dir_name(run: str, stride: int | None = None) -> str:
    """The single-study AOTI packages (`infer.study_aoti_meta`): one per fold, valid for any subset
    of folds and any number of GPUs.
    """
    return f"{run}{stride_suffix(stride)}-aoti-study"


def compile_cache_name(run: str, stride: int | None = None) -> str:
    """The compilation-artifact file of a run: `<run>.torchcache`; with a different stride
    `<run>-stride4.torchcache` (the shapes differ, so the cache is not the same).
    """
    return f"{run}{stride_suffix(stride)}.torchcache"
