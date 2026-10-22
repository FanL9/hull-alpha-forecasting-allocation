"""Protocol v4 forecast-to-allocation: standardization, mappings, scores, selection.

Pure functions over caller-supplied arrays. No loader, no file access. The
official score is the unmodified Hull metric; the cost-adjusted variant is a
separate sensitivity that equals the official arithmetic at zero cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.kaggle_metric import (MAX_INVESTMENT, MIN_INVESTMENT, TRADING_DAYS,
                                          VOLATILITY_RATIO_THRESHOLD, evaluate_allocations)

K_VALUES = (0.10, 0.20, 0.35, 0.50, 0.75)
BASELINE = 'constant_1'
CANDIDATES = tuple([f'linear_k{k:.2f}' for k in K_VALUES] + [f'volscaled_k{k:.2f}' for k in K_VALUES])


def standardization(predictions) -> dict:
    """Location/scale of research OOS forecasts only (ddof=1); no target enters."""
    values = np.asarray(predictions, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError('At least two finite research forecasts are required.')
    sigma = float(values.std(ddof=1))
    if not sigma > 0:
        raise ValueError('Research forecasts have no variation.')
    return {'mu': float(values.mean()), 'sigma': sigma, 'rows': int(len(values))}


def standardize(predictions: pd.Series, state: dict) -> pd.Series:
    return (predictions.astype(float) - state['mu']) / state['sigma']


def volatility_ratio(lagged_returns: pd.Series, config: dict) -> pd.Series:
    """c_t = clip(sigma_long/sigma_short); trailing windows include t (returns up to t-1)."""
    if not lagged_returns.index.is_monotonic_increasing or not lagged_returns.index.is_unique:
        raise ValueError('Lagged returns must be on an ordered unique date grid.')
    short = lagged_returns.rolling(config['short_window_rows'], min_periods=config['min_periods']['short']).std(ddof=1)
    long = lagged_returns.rolling(config['long_window_rows'], min_periods=config['min_periods']['long']).std(ddof=1)
    ratio = (long / short.where(short > 0)).clip(*config['ratio_clip'])
    return ratio.where(np.isfinite(ratio), config['undefined_ratio'])


def positions(candidate: str, z: pd.Series, c: pd.Series | None = None) -> pd.Series:
    """Registered mapping; a missing forecast holds 1.0 (protocol v4 section 3)."""
    if candidate == BASELINE:
        return pd.Series(1.0, index=z.index)
    family, k = candidate.split('_k')
    k = float(k)
    if family == 'linear':
        raw = 1 + k*z
    elif family == 'volscaled':
        if c is None or not c.index.equals(z.index):
            raise ValueError('The volatility state must align with the forecast.')
        raw = c*(1 + k*z)
    else:
        raise ValueError(f'Unregistered candidate: {candidate}')
    if f'{family}_k{k:.2f}' not in CANDIDATES:
        raise ValueError(f'Unregistered k: {candidate}')
    return raw.where(z.notna(), 1.0).clip(MIN_INVESTMENT, MAX_INVESTMENT)


def official_score(forward_returns, risk_free_rate, allocation) -> float:
    """Unmodified official metric; undefined cases become NaN, never a made-up score."""
    try:
        return float(evaluate_allocations(forward_returns, risk_free_rate, allocation).score)
    except Exception:  # official zero-volatility errors and nonfinite inputs
        return float('nan')


def cost_adjusted_score(forward_returns, risk_free_rate, allocation, one_way_bps: float) -> float:
    """Official arithmetic with strategy returns reduced by bps x |change in exposure|."""
    r, f, w = (np.asarray(a, dtype=float) for a in (forward_returns, risk_free_rate, allocation))
    if len(r) < 2 or not (np.isfinite(r).all() and np.isfinite(f).all() and np.isfinite(w).all()):
        return float('nan')
    cost = one_way_bps/1e4*np.abs(np.diff(w, prepend=w[0]))
    strategy = f*(1 - w) + w*r - cost
    n = len(r)
    strategy_mean = np.prod(1 + strategy - f)**(1/n) - 1
    market_mean = np.prod(1 + r - f)**(1/n) - 1
    strategy_std, market_std = strategy.std(ddof=1), r.std(ddof=1)
    if strategy_std == 0 or market_std == 0:
        return float('nan')
    sharpe = strategy_mean/strategy_std*np.sqrt(TRADING_DAYS)
    ratio = strategy_std/market_std
    vol_penalty = 1 + max(0, ratio - VOLATILITY_RATIO_THRESHOLD)
    gap = max(0, (market_mean - strategy_mean)*100*TRADING_DAYS)
    return float(min(sharpe/(vol_penalty*(1 + gap**2/100)), 1_000_000))


def candidate_positions(frame: pd.DataFrame) -> pd.DataFrame:
    """frame has z and c columns on the scoring date grid."""
    return pd.DataFrame({cid: positions(cid, frame['z'], frame['c']) for cid in (BASELINE, *CANDIDATES)},
                        index=frame.index)


def score_table(frame: pd.DataFrame, positions_frame: pd.DataFrame, cost_bps=(0, 1, 5)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overall and per-block official scores, turnover and cost sensitivity."""
    overall, blocks = [], []
    r, f = frame['forward_returns'], frame['risk_free_rate']
    for cid in positions_frame:
        w = positions_frame[cid]
        row = {'candidate': cid, 'official_score': official_score(r, f, w),
               'mean_position': float(w.mean()), 'min_position': float(w.min()), 'max_position': float(w.max()),
               'turnover': float(np.abs(np.diff(w.to_numpy(), prepend=w.iloc[0])).sum())}
        for bps in cost_bps:
            row[f'score_cost_{bps}bp'] = cost_adjusted_score(r, f, w, bps)
        overall.append(row)
        for block, rows in frame.groupby('block', sort=True):
            blocks.append({'candidate': cid, 'block': int(block),
                           'official_score': official_score(r.loc[rows.index], f.loc[rows.index], w.loc[rows.index])})
    return pd.DataFrame(overall), pd.DataFrame(blocks)


def select(overall: pd.DataFrame, blocks: pd.DataFrame, rules: dict) -> dict:
    """Admissible -> near-best (strict margin) -> first in simplicity order; else constant_1."""
    o = overall.set_index('candidate')['official_score']
    base = o[BASELINE]
    b = blocks.pivot(index='block', columns='candidate', values='official_score')
    wins = {cid: int((b[cid] > b[BASELINE]).sum()) for cid in CANDIDATES}
    admissible = [cid for cid in CANDIDATES if np.isfinite(o[cid]) and np.isfinite(base) and o[cid] > base
                  and wins[cid] >= rules['admissible']['min_blocks_beating_constant_1']]
    record = {'baseline_score': float(base), 'block_wins': wins, 'admissible': admissible,
              'margin': rules['near_best_margin']}
    if not admissible:
        return {**record, 'best_score': None, 'near_best': [], 'selected': BASELINE,
                'reason': 'no_admissible_candidate'}
    best = max(o[cid] for cid in admissible)
    near = [cid for cid in admissible if best - o[cid] < rules['near_best_margin']]
    selected = next(cid for cid in CANDIDATES if cid in near)
    return {**record, 'best_score': float(best), 'near_best': near, 'selected': selected,
            'reason': 'first_near_best_in_simplicity_order'}


def bootstrap_intervals(frame: pd.DataFrame, positions_frame: pd.DataFrame, block_length=20,
                        repetitions=2000, seed=42, confidence=.95) -> pd.DataFrame:
    """Descriptive moving-block intervals of each official score; never a selection input."""
    n = len(frame)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block_length + 1, size=(repetitions, -(-n//block_length)))
    index = (starts[:, :, None] + np.arange(block_length)).reshape(repetitions, -1)[:, :n]
    r, f = frame['forward_returns'].to_numpy(float), frame['risk_free_rate'].to_numpy(float)
    rows = []
    for cid in positions_frame:
        w = positions_frame[cid].to_numpy(float)
        draws = np.array([cost_adjusted_score(r[i], f[i], w[i], 0) for i in index])
        valid = draws[np.isfinite(draws)]
        lo, hi = (np.quantile(valid, [(1-confidence)/2, (1+confidence)/2]) if len(valid) >= 2 else (np.nan, np.nan))
        rows.append({'candidate': cid, 'ci_lower': float(lo), 'ci_upper': float(hi),
                     'valid_repetitions': int(len(valid)), 'block_length': block_length, 'seed': seed})
    return pd.DataFrame(rows)


def build_selection(predictions: pd.DataFrame, history: pd.DataFrame, config: dict) -> dict:
    """predictions: A rows (date_id, fold_id, prediction) on 5088-7859; history: research prefix."""
    lo, hi = config['data']['calibration_range']
    a = predictions.loc[predictions['pipeline'].eq('A')].sort_values('date_id')
    if a['date_id'].tolist() != list(range(lo, hi+1)):
        raise ValueError('A research OOS forecasts must cover exactly the calibration range.')
    if history['date_id'].iloc[-1] != hi or history['date_id'].tolist() != list(range(0, hi+1)):
        raise ValueError('Scoring history must be the consecutive research prefix ending at 7859.')
    h = history.set_index('date_id')
    lagged = h['forward_returns'].shift(1)  # official lagged_forward_returns, offset -1
    state = standardization(a['prediction'])
    frame = pd.DataFrame({'block': a['fold_id'].to_numpy(),
                          'z': standardize(a.set_index('date_id')['prediction'], state).to_numpy()},
                         index=pd.Index(a['date_id'].to_numpy(), name='date_id'))
    frame['c'] = volatility_ratio(lagged, config['volatility_state']).loc[frame.index].to_numpy()
    frame['forward_returns'] = h.loc[frame.index, 'forward_returns'].to_numpy(float)
    frame['risk_free_rate'] = h.loc[frame.index, 'risk_free_rate'].to_numpy(float)
    pos = candidate_positions(frame)
    overall, blocks = score_table(frame, pos, tuple(config['costs']['sensitivity_one_way_bps']))
    choice = select(overall, blocks, config['selection'])
    intervals = bootstrap_intervals(frame, pos)
    return {'standardization': state, 'frame': frame, 'positions': pos, 'overall': overall.merge(intervals, on='candidate'),
            'blocks': blocks, 'selection': choice}
