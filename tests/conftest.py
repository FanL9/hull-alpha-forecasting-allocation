"""Artificial fixtures for software checks only; never research data."""

import pandas as pd
import pytest

from src.utils.paths import load_config, project_root


@pytest.fixture(scope="session", autouse=True)
def forbid_real_project_data():
    """The complete suite uses artificial fixtures, even if real data exists."""
    import builtins
    import io
    import os
    from pathlib import Path

    data_root = (project_root() / "data").resolve()
    original_open, original_io_open = builtins.open, io.open

    def guarded(original):
        def open_file(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(file)).resolve()
                if path.is_relative_to(data_root):
                    raise AssertionError("Tests must never access real project data; use synthetic tmp_path inputs.")
            return original(file, *args, **kwargs)
        return open_file

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", guarded(original_open))
        patch.setattr(io, "open", guarded(original_io_open))
        yield


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
