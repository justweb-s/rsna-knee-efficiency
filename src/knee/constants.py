"""Shared constants. The 12 targets, in the exact order of the submission."""
from __future__ import annotations
import pathlib

TARGETS: list[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

ID = "StudyInstanceUID"

# The sequence slots used as input channels, ordered by coverage in the training set.
# (plane, fluid_sensitive, fat_suppression) -> coverage measured on the 4,407 training studies
SEQUENCE_SLOTS: list[tuple[str, int, int]] = [
    ("Axial", 1, 1),      # 100%
    ("Sagittal", 0, 0),   #  97%
    ("Coronal", 1, 1),    #  96%
    ("Sagittal", 1, 1),   #  94%
]

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
EXTERNAL = REPO / "data" / "external"
FOLDS = REPO / "data" / "folds"
EXPERIMENTS = REPO / "experiments"
