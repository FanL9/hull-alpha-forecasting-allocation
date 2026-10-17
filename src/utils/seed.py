"""Reproducibility controls for implemented Python and NumPy operations."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed RNGs; later model implementations must also set their own seeds."""
    random.seed(seed)
    np.random.seed(seed)
