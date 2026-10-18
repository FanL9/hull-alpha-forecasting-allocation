"""Reproducible artificial inputs for pipeline checks, never market evidence."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.alpha_pipeline import load_alpha_design


def make_synthetic_history(root: Path, seed: int = 314159) -> pd.DataFrame:
    """Full registered schema and chronology, with a small active feature subset.

    The remaining artificial candidates stay unavailable to exercise the fixed
    multiplicity denominator. No CSV or existing market artifacts are read.
    """
    design = load_alpha_design(root)
    rng = np.random.default_rng(seed)
    n = design.validation.research_end + 1
    y = rng.normal(size=n)
    values = {c: np.full(n, np.nan) for c in design.source_columns}
    values.update(date_id=np.arange(n), market_forward_excess_returns=y,
                  forward_returns=rng.normal(size=n), risk_free_rate=rng.normal(size=n))
    for name, sign in [("M1", 1), ("M2", -1), ("E1", 1), ("E2", 1)]:
        values[name] = sign * y + rng.normal(scale=1.5, size=n)
    values["D1"] = (y + rng.normal(size=n) > 0).astype(float)
    values["D2"] = (y + rng.normal(size=n) < 0).astype(float)
    frame = pd.DataFrame(values, columns=design.source_columns)
    frame.attrs["phase3_data_kind"] = "synthetic"
    return frame
