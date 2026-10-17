"""Artificial fixtures for software checks only; never research data."""

import pandas as pd
import pytest

from src.utils.paths import load_config, project_root


@pytest.fixture
def config():
    return load_config(project_root() / "configs" / "base.yaml")


@pytest.fixture
def artificial_frames():
    """Tiny deliberately artificial observations confined to unit tests."""
    train = pd.DataFrame({
        "date_id": [0, 1, 2, 3], "M1": [1.0, None, 2.0, 3.0],
        "MOM1": [None, None, 1.0, 2.0], "D1": [0, 1, 0, 1],
        "forward_returns": [0.01, -0.01, 0.0, 0.02],
        "risk_free_rate": [0.0001] * 4,
        "market_forward_excess_returns": [0.009, -0.011, -0.001, 0.019],
    })
    test = pd.DataFrame({"date_id": [4, 5], "M1": [3.0, None], "MOM1": [2.0, 1.0], "D1": [1, 0]})
    return train, test
