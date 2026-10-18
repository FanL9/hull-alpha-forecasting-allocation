"""HAC and paired moving-block uncertainty tools, tested only synthetically.

Gaussian HAC p-values and stationary-block bootstrap intervals are approximate;
neither guarantees inference under nonstationarity or after researcher search.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
import pandas as pd


def newey_west_mean_se(contributions, lags: int = 5) -> float:
    """Bartlett-kernel HAC SE of an observed-value mean on its original grid.

    NaNs represent unavailable timestamps. Their centered contributions are
    zero while the denominator uses observed count: gaps are never compressed
    into adjacent observations. This is a mean/sandwich tool, not OLS fitting.
    """
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 1 or np.isinf(values).any() or type(lags) is not int or lags < 0:
        raise ValueError("Use a one-dimensional finite-or-NaN series and nonnegative integer lag.")
    valid = np.isfinite(values)
    n = int(valid.sum())
    if n < 3:
        return float("nan")
    residual = np.where(valid, values - values[valid].mean(), 0.0)
    effective = min(lags, len(values) - 1)
    variance_sum = float(residual @ residual)
    for lag in range(1, effective + 1):
        weight = 1 - lag / (effective + 1)
        variance_sum += 2 * weight * float(residual[lag:] @ residual[:-lag])
    return sqrt(max(0, variance_sum) / n ** 2)


@dataclass(frozen=True)
class CorrelationInference:
    correlation: float
    standard_error: float
    pvalue: float
    pairs: int
    method: str


def correlation_hac(x, y, lags: int = 5, method: str = "pearson") -> CorrelationInference:
    """Correlation influence-function HAC; no regression or model is fitted.

    Rank-IC uses average ranks of complete pairs. The same original timestamp
    grid is retained for both forms. Constant/insufficient data are undefined,
    not significant. A zero estimated SE also yields an undefined p-value.
    """
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("Use aligned one-dimensional finite-or-NaN arrays.")
    if method not in {"pearson", "spearman"}:
        raise ValueError("Correlation method must be pearson or spearman.")
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    undefined = CorrelationInference(float("nan"), float("nan"), float("nan"), n, method)
    if n < 3:
        return undefined
    av, bv = a[mask], b[mask]
    if method == "spearman":
        av, bv = pd.Series(av).rank(method="average").to_numpy(), pd.Series(bv).rank(method="average").to_numpy()
    if av.std() == 0 or bv.std() == 0:
        return undefined
    u, v = (av - av.mean()) / av.std(ddof=0), (bv - bv.mean()) / bv.std(ddof=0)
    rho = float(np.clip(np.mean(u * v), -1, 1))
    influence = np.full(len(a), np.nan)
    influence[mask] = u * v - .5 * rho * (u ** 2 + v ** 2)
    se = newey_west_mean_se(influence, lags)
    pvalue = erfc(abs(rho / se) / sqrt(2)) if np.isfinite(se) and se > 1e-15 else float("nan")
    return CorrelationInference(rho, se, pvalue, n, method)


def moving_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Resample contiguous joint-data blocks without permuting X and Y apart."""
    if type(length) is not int or type(block_length) is not int or not 1 <= block_length <= length:
        raise ValueError("Block length must be an integer in [1, series length].")
    starts = rng.integers(0, length - block_length + 1, size=(length + block_length - 1) // block_length)
    return (starts[:, None] + np.arange(block_length)).ravel()[:length]


def moving_block_correlation_interval(x, y, block_length: int = 20, repetitions: int = 2000,
                                      confidence: float = .95, seed: int = 42,
                                      method: str = "pearson") -> dict:
    """Percentile interval on paired blocks; no bootstrap p-value is implied."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or np.isinf(a).any() or np.isinf(b).any()
            or type(repetitions) is not int or repetitions < 2 or not 0 < confidence < 1):
        raise ValueError("Invalid aligned arrays or bootstrap settings.")
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(repetitions):
        indices = moving_block_indices(len(a), block_length, rng)
        result = correlation_hac(a[indices], b[indices], lags=0, method=method)
        if np.isfinite(result.correlation):
            estimates.append(result.correlation)
    valid = len(estimates)
    lower, upper = (np.quantile(estimates, [(1 - confidence) / 2, (1 + confidence) / 2])
                    if valid >= 2 else [np.nan, np.nan])
    return {"lower": float(lower), "upper": float(upper), "confidence": confidence,
            "valid_repetitions": valid, "requested_repetitions": repetitions,
            "block_length": block_length, "seed": seed, "method": method}
