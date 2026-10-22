"""Phase 6B: one frozen Internal Holdout evaluation under Protocol v4/v4.1.

All learned state is fitted on date_id 0-7858 (purge 7859) and frozen through
7860-8867. The holdout target is physically removed from every fitting series
and attached only after all forecasts are sealed. Final rows 8868+ are never
parsed: the bounded loader reads exactly 8868 rows.
"""
from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd

from src.features.deep_inputs import windows
from src.features.engineered_alpha import CandidateProvider
from src.research import phase6_allocation as alloc
from src.research.alpha_pipeline import validate_research_history
from src.research.phase4_engine import NestedSelector
from src.research.phase5_engine import Engine, combined
from src.research.phase5_tasks import hash_object
from src.statistics.robust_inference import blockwise_spearman_hac, correlation_hac
from src.validation.protocol_v3 import inner_splits, stopping_split
from src.validation.time_split import read_header
from src.data.download import sha256

FIT_END = 7858
PURGE = 7859
START, STOP = 7860, 8868          # holdout rows 7860..8867 (1008)
BLOCK_ROWS = 252
PIPELINES = ('A', 'B', 'C', 'D', 'Ridge', 'GRU', 'TCN')
PRIMARY_MAPPING = 'volscaled_k0.10'
VOLATILITY_ONLY = 'volatility_only'


def load_holdout_history(train_path, split_plan: dict) -> pd.DataFrame:
    """Bounded holdout loader: exactly rows 0-8867; the final 180 rows are never parsed."""
    if (split_plan['competition_simulation_locked'] is not True or split_plan['competition_simulation_start'] != STOP
            or sha256(train_path) != split_plan['source_train_sha256']
            or read_header(train_path) != split_plan['train_columns']):
        raise ValueError('Holdout source/schema or final reservation differs.')
    frame = pd.read_csv(train_path, nrows=STOP)
    if not np.array_equal(frame['date_id'], np.arange(STOP)):
        raise ValueError('Holdout history must be the consecutive rows 0-8867.')
    return frame


def holdout_windows(raw: pd.DataFrame, state, endpoints) -> np.ndarray:
    """Same causal 63-row windows as Phase 5, permitted only for holdout endpoints."""
    dates = np.asarray(endpoints, dtype=int)
    if dates.ndim != 1 or dates.min() < START or dates.max() >= STOP or len(raw) > STOP:
        raise ValueError('Holdout windows must lie in 7860-8867 and never reach the final rows.')
    x = state.transform(raw.iloc[:int(dates.max())+1])
    return x[dates[:, None]+np.arange(-62, 1)[None, :]]


class HoldoutEngine(Engine):
    """Phase 5 frozen models refitted once at 7858; inner fits stay inside research."""

    def __init__(self, history, config, design, store, test_budget=None):
        if len(history) != STOP or not np.array_equal(history['date_id'], np.arange(STOP)):
            raise ValueError('The holdout engine accepts exactly rows 0-8867.')
        validate_research_history(history.iloc[:START].reset_index(drop=True), design)
        if test_budget not in (None, 1, 2) or (test_budget is not None and store.identity['run_type'] != 'synthetic'):
            raise ValueError('Small epoch budget is restricted to explicit synthetic software tests.')
        self.history, self.config, self.design, self.store = history, config, design, store
        self.test_budget = test_budget
        self.raw = CandidateProvider(history, config, 'A').raw   # X and lagged (t-1) outcomes only
        y = history[config['target']].to_numpy(float).copy()
        y[PURGE:] = np.nan                                       # purge and holdout labels physically removed
        self.y = y
        self.states, self.scores, self.neural, self.ridges = {}, {}, {}, {}
        self.folds = ()
        self.outer = (FIT_END,)
        # D needs C's inner forecasts at the two inner references; no deeper layer is required.
        self.references = tuple(sorted({FIT_END} | {s.training_end for s in inner_splits(FIT_END, config)}))

    def horizon(self, end):
        return (START, STOP) if end == FIT_END else (end+2, end+254)

    def forecast_windows(self, state, end):
        start, stop = self.horizon(end)
        dates = np.arange(start, stop)
        return holdout_windows(self.raw, state, dates) if end == FIT_END else windows(self.raw, state, dates)

    def ab(self, name, end):
        if end != FIT_END:
            return super().ab(name, end)
        key = name, end
        if key not in self.scores:
            task = {'kind': 'selector_forecast', 'model': name, 'reference_end': end, 'seed': None,
                    'training_range': [0, end], 'validation_range': [START, STOP-1], 'holdout': True}

            def compute():
                provider = CandidateProvider(self.history.copy(), self.config, name)
                provider.y = provider.y.iloc[:end+1].copy()     # fitting labels end at 7858
                state = NestedSelector(provider).fit(end)
                return {'prediction': state.predict(provider.frame(end, START, STOP)), 'selector': state.audit(),
                        'status': 'active' if len(state.selected) >= 2 else 'no_alpha'}
            self.scores[key] = self.store.execute(task, compute)
        return self.scores[key]['prediction']

    def job(self, end, name, seed):
        split = stopping_split(end, self.config)
        early, full = self.input_state(split.training_end), self.input_state(end)
        from src.features.deep_inputs import training_arrays
        a, b = training_arrays(self.raw, self.y, early), training_arrays(self.raw, self.y, full)
        dates = np.arange(split.validation_start, split.validation_end+1)
        dates = dates[np.isfinite(self.y[dates])]
        valid = a is not None and b is not None and len(dates) > 0
        start, stop = self.horizon(end)
        return {'directory': str(self.store.directory), 'identity': self.store.identity,
                'task': {'kind': 'neural', 'reference_end': end, 'model': name, 'seed': seed,
                         'outer_consumers': [{'scenario': 'holdout' if end == FIT_END else 'holdout_D_inner'}],
                         'model_config_hash': hash_object({'model': self.config['models'][name.lower()],
                                                           'training': self.config['training'], 'deep': self.config['deep']})},
                'stop_x': a[0] if valid else None, 'stop_y': a[1] if valid else None,
                'validation_x': windows(self.raw, early, dates) if valid else None,
                'validation_y': (self.y[dates]-early.target_mean)/early.target_scale if valid else None,
                'refit_x': b[0] if valid else None, 'refit_y': b[1] if valid else None,
                'forecast_x': self.forecast_windows(full, end),
                'stop_range': [0, split.training_end], 'stopping_range': [split.validation_start, split.validation_end],
                'refit_range': [0, end], 'forecast_range': [start, stop-1],
                'stop_feature_hash': hash_object(early.audit()), 'refit_feature_hash': hash_object(full.audit()),
                'target_mean': full.target_mean, 'target_scale': full.target_scale, 'test_budget': self.test_budget}

    def deep(self, name, end):
        if name == 'C':
            return .5*self.deep('GRU', end)+.5*self.deep('TCN', end)
        rows = [self.neural[end, name, seed] for seed in self.config['deep']['seeds']]
        start, stop = self.horizon(end)
        if any(row['status'] != 'active' or len(row['prediction']) != stop-start for row in rows):
            return np.full(stop-start, np.nan)
        return np.mean([row['prediction'] for row in rows], axis=0)

    def fixed_ridge(self, end):
        if end != FIT_END:
            return super().fixed_ridge(end)
        from src.features.deep_inputs import training_arrays
        from src.models.deep_alpha import ridge_predict, ridge_solutions
        task = {'kind': 'ridge_svd', 'reference_end': end, 'model': 'Ridge', 'seed': None, 'training_range': [0, end],
                'validation_range': [START, STOP-1], 'input_feature_hash': hash_object(self.input_state(end).audit())}

        def compute():
            state = self.input_state(end)
            arrays = training_arrays(self.raw, self.y, state)
            alphas = self.config['models']['ridge']['alphas']
            if arrays is None:
                return {str(a): {'prediction': np.full(STOP-START, np.nan), 'status': 'unavailable'} for a in alphas}
            fits = ridge_solutions(arrays[0], arrays[1], alphas)
            forecast = self.forecast_windows(state, end)
            return {str(a): {'prediction': ridge_predict(fits[a], forecast)*state.target_scale+state.target_mean,
                             'state': fits[a], 'status': 'active'} for a in alphas}
        return self.store.execute(task, compute)

    def holdout_forecasts(self) -> tuple[pd.DataFrame, dict]:
        """Every forecast and map is fixed here; no target argument exists."""
        outer = {n: self.forecast(n, FIT_END) for n in ('A', 'B', 'Ridge', 'GRU', 'TCN', 'C')}
        outer['D'], dstate = combined([self.inner(n, FIT_END) for n in ('A', 'B', 'C')],
                                      [outer[n] for n in ('A', 'B', 'C')])
        dates = np.arange(START, STOP)
        frame = pd.DataFrame({'date_id': np.tile(dates, len(PIPELINES)), 'pipeline': np.repeat(PIPELINES, len(dates)),
                              'prediction': np.concatenate([np.asarray(outer[n], float) for n in PIPELINES])})
        audit = {'fit_end': FIT_END, 'purge': PURGE, 'holdout': [START, STOP-1],
                 'A_selector': self.scores['A', FIT_END]['selector'], 'B_selector': self.scores['B', FIT_END]['selector'],
                 'inputs': self.input_state(FIT_END).audit(), 'D_normalization': dstate,
                 'ridge': {k: v for k, v in self.ridges[FIT_END].items() if k != 'prediction'}}
        return frame, audit


# ---------------------------------------------------------------- evaluation

def holdout_blocks(dates) -> np.ndarray:
    return (np.asarray(dates) - START)//BLOCK_ROWS + 1


def forecast_metrics(predictions: pd.DataFrame, target: pd.Series, lags=5) -> pd.DataFrame:
    """Block-wise Rank IC with Newey-West inference; the A row is the v4.1 confirmation test."""
    rows = []
    for name, g in predictions.groupby('pipeline', sort=False):
        g = g.sort_values('date_id')
        x, y = g['prediction'].to_numpy(float), target.loc[g['date_id']].to_numpy(float)
        blocks = holdout_blocks(g['date_id'])
        est = blockwise_spearman_hac(x, y, blocks, lags=lags)
        z = est.correlation/est.standard_error if np.isfinite(est.standard_error) and est.standard_error > 0 else np.nan
        row = {'pipeline': name, 'blockwise_rank_ic': est.correlation, 'pairs': est.pairs, 'hac_se': est.standard_error,
               'one_sided_p': .5*erfc(z/sqrt(2)) if np.isfinite(z) else np.nan,
               'ci95_lower': est.correlation-1.959963984540054*est.standard_error,
               'ci95_upper': est.correlation+1.959963984540054*est.standard_error}
        for b in range(1, 5):
            m = blocks == b
            row[f'block_{b}_ic'] = correlation_hac(x[m], y[m], lags=0, method='spearman').correlation
        if name == 'A':  # the single pre-registered v4.1 confirmation test
            row['confirmed_at_0.05'] = bool(np.isfinite(row['one_sided_p']) and row['one_sided_p'] < .05)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference_interval(r, f, w1, w2, block_length=20, repetitions=2000, seed=42, confidence=.95) -> dict:
    """Same resampled days for both positions; descriptive only."""
    r, f, w1, w2 = (np.asarray(a, float) for a in (r, f, w1, w2))
    n = len(r)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n-block_length+1, size=(repetitions, -(-n//block_length)))
    index = (starts[:, :, None]+np.arange(block_length)).reshape(repetitions, -1)[:, :n]
    draws = np.array([alloc.cost_adjusted_score(r[i], f[i], w1[i], 0)-alloc.cost_adjusted_score(r[i], f[i], w2[i], 0)
                      for i in index])
    valid = draws[np.isfinite(draws)]
    lo, hi = np.quantile(valid, [(1-confidence)/2, (1+confidence)/2]) if len(valid) >= 2 else (np.nan, np.nan)
    return {'estimate': alloc.official_score(r, f, w1)-alloc.official_score(r, f, w2), 'ci_lower': float(lo),
            'ci_upper': float(hi), 'valid_repetitions': int(len(valid)), 'block_length': block_length,
            'repetitions': repetitions, 'seed': seed}


def research_standardization(research_predictions: pd.DataFrame) -> dict:
    """Each pipeline's own research-OOS mean/std (5088-7859); forecasts only, no target."""
    out = {}
    for name in PIPELINES:
        p = research_predictions.loc[(research_predictions['pipeline'] == name)
                                     & research_predictions['date_id'].between(5088, 7859), 'prediction']
        out[name] = alloc.standardization(p) if len(p) == 2772 else None
    return out


def allocation_evaluation(predictions, history, v4, v41, research_predictions) -> dict:
    h = history.set_index('date_id')
    dates = np.arange(START, STOP)
    lagged = h['forward_returns'].shift(1)                        # official lagged field, offset -1
    c = alloc.volatility_ratio(lagged, v4['volatility_state']).loc[dates]
    a = predictions.loc[predictions['pipeline'] == 'A'].set_index('date_id')['prediction'].loc[dates]
    sel = v41['phase6_selection']
    state = {'mu': sel['mu_A'], 'sigma': sel['sigma_A']}           # frozen; never refitted on holdout
    frame = pd.DataFrame({'block': holdout_blocks(dates), 'z': alloc.standardize(a, state).to_numpy(), 'c': c.to_numpy(),
                          'forward_returns': h.loc[dates, 'forward_returns'].to_numpy(float),
                          'risk_free_rate': h.loc[dates, 'risk_free_rate'].to_numpy(float)},
                         index=pd.Index(dates, name='date_id'))
    positions = alloc.candidate_positions(frame)
    positions[VOLATILITY_ONLY] = frame['c'].clip(0, 2)
    overall, blocks = alloc.score_table(frame, positions, tuple(v4['costs']['sensitivity_one_way_bps']))
    overall = overall.merge(alloc.bootstrap_intervals(frame, positions), on='candidate')
    overall['role'] = np.where(overall['candidate'] == PRIMARY_MAPPING, 'primary', 'secondary')
    r, f = frame['forward_returns'], frame['risk_free_rate']
    paired = {f'{PRIMARY_MAPPING} - {other}': paired_difference_interval(r, f, positions[PRIMARY_MAPPING], positions[other])
              for other in (VOLATILITY_ONLY, alloc.BASELINE)}
    secondary, standards = [], research_standardization(research_predictions)
    for name in PIPELINES[1:]:
        s = predictions.loc[predictions['pipeline'] == name].set_index('date_id')['prediction'].loc[dates]
        if standards[name] is None or not np.isfinite(s).any():
            secondary.append({'pipeline': name, 'official_score': np.nan, 'status': 'unavailable'})
            continue
        w = alloc.positions(PRIMARY_MAPPING, alloc.standardize(s, standards[name]), frame['c'])
        secondary.append({'pipeline': name, 'official_score': alloc.official_score(r, f, w),
                          'mean_position': float(w.mean()), 'research_mu': standards[name]['mu'],
                          'research_sigma': standards[name]['sigma'], 'status': 'active'})
    return {'frame': frame, 'positions': positions, 'overall': overall, 'blocks': blocks, 'paired': paired,
            'secondary_forecast_allocations': pd.DataFrame(secondary)}
