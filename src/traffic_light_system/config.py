"""Shared project configuration and reproducibility helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMO_HOME_ENV = "SUMO_HOME"


@dataclass(frozen=True)
class SimulationConfig:
    """Settings shared by the SUMO environment and training scripts."""

    seed: int = 42
    simulation_step_seconds: float = 1.0
    episode_seconds: int = 3600
    yellow_duration_seconds: int = 3


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy and expose the seed to compatible libraries."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

