"""Official Hull metric arithmetic, independently expressed from the source.

Source: https://www.kaggle.com/code/metric/hull-competition-sharpe
Source hashes and units: configs/official_metric.yaml. No real data is read.
The compatibility score preserves pandas alignment/NaN behavior; the project
adapter adds finite-input checks. Neither silently clips allocations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2
TRADING_DAYS = 252
VOLATILITY_RATIO_THRESHOLD = 1.20


class ParticipantVisibleError(Exception):
    """Official-style invalid-prediction or zero-volatility error."""


@dataclass(frozen=True)
class MetricComponents:
    strategy_geometric_excess_mean: float
    market_geometric_excess_mean: float
    strategy_std: float
    market_std: float
    sharpe: float
    volatility_ratio: float
    volatility_penalty: float
    annualized_return_gap_percentage_points: float
    return_penalty: float
    score: float


def metric_components(solution: pd.DataFrame, submission: pd.DataFrame) -> MetricComponents:
    """Match official formulas, including geometric excess means and ddof=1.

    Inputs must already represent the scored rows. is_scored and row IDs are
    not used by the official formula. Inputs are not modified (unlike source).
    """
    allocation = submission["prediction"]
    if not is_numeric_dtype(allocation):
        raise ParticipantVisibleError("Predictions must be numeric")
    position = allocation.reindex(solution.index)
    if position.max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f"Position exceeds maximum of {MAX_INVESTMENT}")
    if position.min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f"Position below minimum of {MIN_INVESTMENT}")
    market, cash = solution["forward_returns"], solution["risk_free_rate"]
    strategy = cash * (1 - position) + position * market
    n = len(solution)
    # Keep the publisher's product/power, not log compounding or arithmetic mean.
    strategy_excess = strategy - cash
    market_excess = market - cash
    strategy_mean = (1 + strategy_excess).prod() ** (1 / n) - 1
    market_mean = (1 + market_excess).prod() ** (1 / n) - 1
    strategy_std, market_std = strategy.std(), market.std()
    if strategy_std == 0:
        raise ParticipantVisibleError("Division by zero, strategy std is zero")
    raw_sharpe = strategy_mean / strategy_std * np.sqrt(TRADING_DAYS)
    annual_strategy_vol = float(strategy_std * np.sqrt(TRADING_DAYS) * 100)
    annual_market_vol = float(market_std * np.sqrt(TRADING_DAYS) * 100)
    if annual_market_vol == 0:
        raise ParticipantVisibleError("Division by zero, market std is zero")
    ratio = annual_strategy_vol / annual_market_vol
    vol_penalty = 1 + (max(0, ratio - VOLATILITY_RATIO_THRESHOLD) if annual_market_vol > 0 else 0)
    gap = max(0, (market_mean - strategy_mean) * 100 * TRADING_DAYS)
    return_penalty = 1 + gap ** 2 / 100
    result = min(float(raw_sharpe / (vol_penalty * return_penalty)), 1_000_000)
    return MetricComponents(float(strategy_mean), float(market_mean), float(strategy_std),
        float(market_std), float(raw_sharpe), float(ratio), float(vol_penalty),
        float(gap), float(return_penalty), result)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Kaggle-compatible entry point; the official row_id argument is unused."""
    return metric_components(solution, submission).score


def evaluate_allocations(forward_returns, risk_free_rate, allocations) -> MetricComponents:
    """Stricter local adapter for already-selected rows; no costs or optimization.

    The original scorer does not consistently reject NaNs or mismatched rows.
    This adapter explicitly rejects those cases rather than manufacturing scores.
    """
    arrays = [np.asarray(a) for a in [forward_returns, risk_free_rate, allocations]]
    if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1 or len(arrays[0]) < 2:
        raise ValueError("At least two equal-length one-dimensional numeric arrays are required.")
    if any(not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Metric inputs must be finite and numeric.")
    solution = pd.DataFrame({"forward_returns": arrays[0], "risk_free_rate": arrays[1]})
    submission = pd.DataFrame({"prediction": arrays[2]})
    result = metric_components(solution, submission)
    if not np.isfinite(result.score):
        raise ValueError("Official arithmetic is undefined for these inputs.")
    return result
