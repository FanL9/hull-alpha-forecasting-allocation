"""Protocol v5 frozen refit engine: one reference end, one frozen forecast span.

Generalizes the Phase 6B HoldoutEngine (which is frozen and hard-wired to
7858 -> 7860-8867) to any (fit_end, start, stop) with start = fit_end + 2.
The Phase 3-5 helpers guard date_id 7859; the copies here differ only in
taking that bound from the engine. Labels from the purge row on are removed
from every fitting series. With the holdout boundaries the engine reproduces
HoldoutEngine exactly (tested on artificial data).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.features.deep_inputs import InputState, average_clusters
from src.features.engineered_alpha import CandidateProvider, eligibility
from src.models.dynamic_ridge import DynRidge
from src.research.alpha_pipeline import validate_research_history
from src.research.phase4_engine import NestedSelector
from src.research.phase5_engine import Engine, combined
from src.research.phase5_tasks import hash_object
from src.validation.protocol_v3 import inner_splits, stopping_split

FINAL_FIT_END, FINAL_START, FINAL_STOP = 8866, 8868, 9048


def fit_inputs_until(history: pd.DataFrame, end: int, config: dict, last: int) -> InputState:
    """deep_inputs.fit_inputs with the 7859 guard replaced by the engine bound."""
    if not 62 <= end < len(history) or end > last:
        raise ValueError('Invalid input fitting reference.')
    prefix = history.iloc[:end+1].copy()
    provider = CandidateProvider(prefix, config, 'A')
    screen = NestedSelector(provider).fit(end).candidates.set_index('candidate_id')
    available = eligibility(provider.raw, config['availability'])
    names = available.loc[available.eligible, 'feature'].tolist()
    clusters = average_clusters(provider.raw[names])
    qualified = screen.loc[screen.eligible_all_references & screen.same_sign]
    scores = {cid.split('/')[1]: float(row.ranking_score) for cid, row in qualified.iterrows()}
    reps = []
    for cluster in clusters:
        pool = [n for n in cluster if n in scores]
        if pool:
            reps.append(min(pool, key=lambda n: (-scores[n], n)))
    reps = tuple(sorted(reps, key=lambda n: (-scores[n], n))[:16])
    a = provider.raw[list(reps)].to_numpy(float)
    means, scales = np.nanmean(a, axis=0), np.nanstd(a, axis=0, ddof=0)
    y = prefix[config['target']].iloc[62:].to_numpy(float)
    y = y[np.isfinite(y)]
    mean, scale = (float(y.mean()), float(y.std(ddof=0))) if len(y) else (np.nan, np.nan)
    return InputState(end, reps, means, scales, mean, scale, tuple(map(tuple, clusters)), {n: scores[n] for n in reps})


def causal_windows(raw: pd.DataFrame, state, endpoints, last: int) -> np.ndarray:
    """deep_inputs.windows with the endpoint bound taken from the engine."""
    dates = np.asarray(endpoints, dtype=int)
    if dates.ndim != 1 or np.any(dates < 62) or np.any(dates >= len(raw)) or np.any(dates > last):
        raise ValueError('Windows require 63 real rows and endpoints inside the engine span.')
    x = state.transform(raw.iloc[:int(dates.max())+1])
    return x[dates[:, None]+np.arange(-62, 1)[None, :]]


def causal_training_arrays(raw, y, state, last: int):
    dates = np.arange(62, state.end+1)
    dates = dates[np.isfinite(np.asarray(y)[dates])]
    if not state.names or not np.isfinite(state.target_scale) or state.target_scale <= 1e-12:
        return None
    return causal_windows(raw, state, dates, last), (np.asarray(y)[dates]-state.target_mean)/state.target_scale, dates


def bounded_config(config: dict, last_reference: int) -> dict:
    """Same rules; only the reference guard moves to the purge row of this engine."""
    out = copy.deepcopy(config)
    out['research']['end'] = last_reference
    return out


class RefitEngine(Engine):
    """Phase 5 frozen models refitted once at fit_end and frozen through [start, stop)."""

    def __init__(self, history, config, design, store, fit_end, start, stop, test_budget=None):
        if start != fit_end+2 or len(history) != stop or not np.array_equal(history['date_id'], np.arange(stop)):
            raise ValueError('The refit engine accepts exactly rows 0..stop-1 with one purge row.')
        validate_research_history(history.iloc[:7860].reset_index(drop=True), design)
        if test_budget not in (None, 1, 2) or (test_budget is not None and store.identity['run_type'] != 'synthetic'):
            raise ValueError('Small epoch budget is restricted to explicit synthetic software tests.')
        self.fit_end, self.start, self.stop, self.purge = fit_end, start, stop, fit_end+1
        self.history, self.design, self.store = history, design, store
        self.config = bounded_config(config, self.purge)
        self.test_budget = test_budget
        self.raw = CandidateProvider(history, self.config, 'A').raw   # X and lagged (t-1) outcomes only
        y = history[self.config['target']].to_numpy(float).copy()
        y[self.purge:] = np.nan                                       # purge and later labels physically removed
        self.y = y
        self.states, self.scores, self.neural, self.ridges = {}, {}, {}, {}
        self.folds = ()
        self.outer = (fit_end,)
        self.references = tuple(sorted({fit_end} | {s.training_end for s in inner_splits(fit_end, self.config)}))
        self.dynridge = DynRidge(self.raw, self.y, self.config['availability'])

    def horizon(self, end):
        return (self.start, self.stop) if end == self.fit_end else (end+2, end+254)

    def input_state(self, end):
        if end not in self.states:
            from dataclasses import asdict
            task = {'kind': 'inputs', 'reference_end': end, 'model': 'shared_Ridge_GRU_TCN',
                    'training_range': [0, end], 'validation_range': None, 'seed': None}
            saved = self.store.execute(task, lambda: asdict(fit_inputs_until(self.history, end, self.config, self.fit_end)))
            saved['names'] = tuple(saved['names']); saved['clusters'] = tuple(map(tuple, saved['clusters']))
            self.states[end] = InputState(**saved)
        return self.states[end]

    def windows(self, state, dates):
        return causal_windows(self.raw, state, dates, self.stop-1)

    def ab(self, name, end):
        key = name, end
        if key not in self.scores:
            start, stop = self.horizon(end)
            task = {'kind': 'selector_forecast', 'model': name, 'reference_end': end, 'seed': None,
                    'training_range': [0, end], 'validation_range': [start, stop-1]}
            if end == self.fit_end:
                task['holdout'] = True     # same task identity as the frozen HoldoutEngine

            def compute():
                provider = CandidateProvider(self.history.copy(), self.config, name)
                provider.y = provider.y.iloc[:end+1].copy()     # fitting labels end at the reference
                state = NestedSelector(provider).fit(end)
                return {'prediction': state.predict(provider.frame(end, start, stop)), 'selector': state.audit(),
                        'status': 'active' if len(state.selected) >= 2 else 'no_alpha'}
            self.scores[key] = self.store.execute(task, compute)
        return self.scores[key]['prediction']

    def job(self, end, name, seed):
        split = stopping_split(end, self.config)
        early, full = self.input_state(split.training_end), self.input_state(end)
        a = causal_training_arrays(self.raw, self.y, early, self.fit_end)
        b = causal_training_arrays(self.raw, self.y, full, self.fit_end)
        dates = np.arange(split.validation_start, split.validation_end+1)
        dates = dates[np.isfinite(self.y[dates])]
        valid = a is not None and b is not None and len(dates) > 0
        start, stop = self.horizon(end)
        scenario = 'holdout' if end == self.fit_end else 'holdout_D_inner'
        return {'directory': str(self.store.directory), 'identity': self.store.identity,
                'task': {'kind': 'neural', 'reference_end': end, 'model': name, 'seed': seed,
                         'outer_consumers': [{'scenario': scenario}],
                         'model_config_hash': hash_object({'model': self.config['models'][name.lower()],
                                                           'training': self.config['training'], 'deep': self.config['deep']})},
                'stop_x': a[0] if valid else None, 'stop_y': a[1] if valid else None,
                'validation_x': self.windows(early, dates) if valid else None,
                'validation_y': (self.y[dates]-early.target_mean)/early.target_scale if valid else None,
                'refit_x': b[0] if valid else None, 'refit_y': b[1] if valid else None,
                'forecast_x': self.windows(full, np.arange(start, stop)),
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
        from src.models.deep_alpha import ridge_predict, ridge_solutions
        start, stop = self.horizon(end)
        task = {'kind': 'ridge_svd', 'reference_end': end, 'model': 'Ridge', 'seed': None, 'training_range': [0, end],
                'validation_range': [start, stop-1], 'input_feature_hash': hash_object(self.input_state(end).audit())}

        def compute():
            state = self.input_state(end)
            arrays = causal_training_arrays(self.raw, self.y, state, self.fit_end)
            alphas = self.config['models']['ridge']['alphas']
            if arrays is None:
                return {str(a): {'prediction': np.full(stop-start, np.nan), 'status': 'unavailable'} for a in alphas}
            fits = ridge_solutions(arrays[0], arrays[1], alphas)
            forecast = self.windows(state, np.arange(start, stop))
            return {str(a): {'prediction': ridge_predict(fits[a], forecast)*state.target_scale+state.target_mean,
                             'state': fits[a], 'status': 'active'} for a in alphas}
        return self.store.execute(task, compute)

    def forecast(self, name, end):
        if name == 'DynRidge':
            return self.dynridge.forecast(end, *self.horizon(end))
        return super().forecast(name, end)

    def refit_forecasts(self, names) -> tuple[pd.DataFrame, dict]:
        """Forecasts of the requested pipelines over [start, stop); no target argument exists."""
        outer = {}
        for name in names:
            if name == 'D':
                outer['D'], dstate = combined([self.inner(n, self.fit_end) for n in ('A', 'B', 'C')],
                                              [self.forecast(n, self.fit_end) for n in ('A', 'B', 'C')])
            else:
                outer[name] = self.forecast(name, self.fit_end)
        dates = np.arange(self.start, self.stop)
        frame = pd.DataFrame({'date_id': np.tile(dates, len(names)), 'pipeline': np.repeat(list(names), len(dates)),
                              'prediction': np.concatenate([np.asarray(outer[n], float) for n in names])})
        audit = {'fit_end': self.fit_end, 'purge': self.purge, 'forecast_rows': [self.start, self.stop-1]}
        for name in names:
            if name in ('A', 'B'):
                audit[name+'_selector'] = self.scores[name, self.fit_end]['selector']
            if name == 'Ridge':
                audit['ridge'] = {k: v for k, v in self.ridges[self.fit_end].items() if k != 'prediction'}
            if name == 'DynRidge':
                audit['dynridge'] = self.dynridge.audit(self.fit_end)
            if name == 'D':
                audit['D_normalization'] = dstate
            if name in ('Ridge', 'C', 'D'):
                audit['inputs'] = self.input_state(self.fit_end).audit()
        return frame, audit


def neural_needed(names) -> bool:
    return any(n in ('C', 'D', 'GRU', 'TCN') for n in names)
