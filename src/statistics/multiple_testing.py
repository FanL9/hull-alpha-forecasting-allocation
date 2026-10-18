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


def family_pvalues(pvalues: dict, hypotheses) -> np.ndarray:
    """Order p-values by a prespecified family; absent or untestable entries get p=1.

    Unknown names are rejected so a family can never grow or shrink silently.
    """
    names = list(hypotheses)
    if len(set(names)) != len(names) or set(pvalues) - set(names):
        raise ValueError("p-values must belong to one prespecified family of distinct hypotheses.")
    values = [pvalues.get(name) for name in names]
    return np.array([1. if v is None or not np.isfinite(v) else float(v) for v in values])


def protocol_multiplicity(single_pvalues: dict, single_hypotheses, composite_pvalues: dict,
                          composite_hypotheses, alpha: float = .05) -> dict:
    """Family-specific BH (primary), plus combined-BH and BY sensitivity analyses.

    The combined family pools both raw primary p-value vectors. It is reported
    as sensitivity only and never replaces the two family-specific results.
    """
    single = family_pvalues(single_pvalues, single_hypotheses)
    composite = family_pvalues(composite_pvalues, composite_hypotheses)
    combined = np.r_[single, composite]
    return {"single_feature_bh": adjust_fdr(single, alpha, "bh"),
            "family_composite_bh": adjust_fdr(composite, alpha, "bh"),
            "global_bh_sensitivity": adjust_fdr(combined, alpha, "bh"),
            "single_feature_by_sensitivity": adjust_fdr(single, alpha, "by"),
            "family_composite_by_sensitivity": adjust_fdr(composite, alpha, "by"),
            "global_by_sensitivity": adjust_fdr(combined, alpha, "by")}
