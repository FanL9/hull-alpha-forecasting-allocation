"""BH FDR and dependence-robust BY sensitivity, with explicit hypotheses."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FDRResult:
    adjusted_pvalues: np.ndarray
    rejected: np.ndarray
    alpha: float
    method: str


def adjust_fdr(pvalues, alpha: float = .05, method: str = "bh") -> FDRResult:
    """Step-up adjusted p-values in original order, with stable tie handling.

    Supply p=1 for prespecified but unavailable hypotheses. NaNs are rejected
    rather than silently reducing the multiplicity denominator. This tool does
    not decide which hypotheses to include or inspect any feature/target data.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a finite one-dimensional array in [0,1].")
    if not 0 < alpha < 1 or method not in {"bh", "by"}:
        raise ValueError("Use alpha in (0,1) and method bh or by.")
    if not len(p):
        return FDRResult(p.copy(), np.zeros(0, dtype=bool), alpha, method)
    order = np.argsort(p, kind="stable")
    m = len(p)
    multiplier = np.sum(1 / np.arange(1, m + 1)) if method == "by" else 1.0
    scaled = p[order] * m * multiplier / np.arange(1, m + 1)
    adjusted_sorted = np.minimum(1, np.minimum.accumulate(scaled[::-1])[::-1])
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return FDRResult(adjusted, adjusted <= alpha, alpha, method)
