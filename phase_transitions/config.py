"""Explicit configuration for the three phase transitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


COLS_20 = np.setdiff1d(
    np.arange(30),
    [2, 3, 5, 7, 12, 18, 21, 24, 27, 28],
)
COLS_12 = np.array([0, 2, 4, 5, 6, 11, 12, 15, 17, 23, 24, 29])


def feature_sets() -> dict[str, np.ndarray]:
    """Return copies of the canonical feature sets."""

    return {
        "30": np.arange(30),
        "20": COLS_20.copy(),
        "12": COLS_12.copy(),
    }


@dataclass(frozen=True)
class TransitionConfig:
    name: str
    parameter_name: str
    phase_low_value: float
    phase_high_value: float
    phase_low_name: str
    phase_high_name: str
    data_dir: Path
    train_stride: int = 20
    full_stride: int = 10
    unsupervised_train_stride: int = 1
    unsupervised_full_stride: int = 1
    meanshift_fit_stride: int = 10
    calibration_cv: int = 5
    validation_fraction: float = 0.2
    validation_threshold: float = 0.999

    @property
    def train_path(self) -> Path:
        return self.data_dir / "TRAIN_30.npz"

    @property
    def full_path(self) -> Path:
        return self.data_dir / "FULL_30.npz"


TRANSITIONS = {
    "AB": TransitionConfig(
        name="AB",
        parameter_name="Delta",
        phase_low_value=-0.128,
        phase_high_value=-0.108,
        phase_low_name="B",
        phase_high_name="A",
        data_dir=ROOT / "a_b" / "data" / "processed",
        unsupervised_train_stride=1,
        unsupervised_full_stride=1,
    ),
    "BCB": TransitionConfig(
        name="BCB",
        parameter_name="Delta",
        phase_low_value=0.036,
        phase_high_value=0.041,
        phase_low_name="Cb",
        phase_high_name="B",
        data_dir=ROOT / "b_cb" / "data" / "processed",
        unsupervised_train_stride=1,
        unsupervised_full_stride=1,
    ),
    "CA": TransitionConfig(
        name="CA",
        parameter_name="K0",
        phase_low_value=4.67,
        phase_high_value=4.93,
        phase_low_name="C",
        phase_high_name="A",
        data_dir=ROOT / "c_a" / "data" / "processed",
        unsupervised_train_stride=10,
        unsupervised_full_stride=10,
    ),
}
