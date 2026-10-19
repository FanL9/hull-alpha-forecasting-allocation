"""Protocol v3.1 causal representations and reference-fitted candidate providers.

No loader lives here. Fitting accepts explicit historical prefixes; applying a
frozen map to later X never changes a fitted direction, threshold or ECDF.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import ceil

import numpy as np
import pandas as pd

from src.features.availability import available_lagged_outcomes
from src.features.family_composites import family_composite
from src.validation.protocol_v3 import FAMILIES, candidate_ids, raw_names


def representations(x: pd.Series) -> pd.DataFrame:
    """Row-grid deltas/Z; rank on the last 63 OBSERVED values, including t."""
    roll = x.rolling(252, min_periods=126)
    scale = roll.std(ddof=0).replace(0, np.nan)
    observed = x.dropna()
    ranks = observed.rolling(63, min_periods=32).rank(method='average')
    counts = observed.rolling(63, min_periods=32).count()
    rank = ((ranks - .5) / counts).reindex(x.index)
    return pd.DataFrame({'Level': x, 'Delta5': x-x.shift(5),
        'Delta21': x-x.shift(21), 'Z252': (x-roll.mean())/scale, 'Rank63': rank})


def eligibility(x: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """The frozen availability rules, generalized to namespaced candidates."""
    if x.empty or x.columns.duplicated().any() or not x.index.is_unique or not x.index.is_monotonic_increasing:
        raise ValueError('Nonempty ordered unique candidate frame required.')
    a = x.to_numpy(dtype=float)
    if np.isinf(a).any():
        raise ValueError('Infinite candidates are forbidden.')
    present = np.isfinite(a)
    count = present.sum(axis=0)
    first = present.argmax(axis=0)
    coverage = count / (len(x)-first)
    recent = 1-present[-policy['recent_window']:].mean(axis=0)
    # Pandas counts exact observed values, including ties in D/interaction fields.
    distinct = x.nunique().to_numpy()
    conditions = (count >= policy['min_observed'], coverage >= policy['min_post_launch_coverage'],
        recent <= policy['max_recent_missing'], distinct >= policy['min_distinct'])
    reasons = ('insufficient_observations', 'insufficient_post_launch_coverage',
        'excess_recent_missingness', 'insufficient_variation')
    return pd.DataFrame({'feature': x.columns, 'observed': count,
        'first_observed_position': np.where(count, first, np.nan),
        'post_launch_coverage': coverage, 'recent_missing_ratio': recent,
        'distinct_observed': distinct, 'eligible': np.logical_and.reduce(conditions),
        'reasons': [';'.join(r for r,c in zip(reasons,conditions) if not c[j]) for j in range(len(x.columns))]})


def array_hash(a) -> str:
    return hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest()


def mid_ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(reference[np.isfinite(reference)])
    if not len(ref):
        raise ValueError('Empty training map.')
    lo = np.searchsorted(ref, values, side='left')
    hi = np.searchsorted(ref, values, side='right')
    return np.where(np.isfinite(values), (lo+hi)/(2*len(ref)), np.nan)


@dataclass
class VRegime:
    members: tuple[str, ...]
    references: dict[str, np.ndarray]
    required: int
    threshold: float
    available: bool
    training_end: int

    def proxy(self, x: pd.DataFrame) -> np.ndarray:
        if len(self.members) < 2:
            return np.full(len(x), np.nan)
        mapped = np.column_stack([mid_ecdf(self.references[c], x[c].to_numpy(float)) for c in self.members])
        counts = np.isfinite(mapped).sum(axis=1)
        return np.divide(np.nansum(mapped, axis=1), counts,
            out=np.full(len(x), np.nan), where=counts >= self.required)

    def high(self, x: pd.DataFrame) -> np.ndarray:
        p = self.proxy(x)
        return np.where(np.isfinite(p) & self.available, (p > self.threshold).astype(float), np.nan)

    def audit(self) -> dict:
        return {'training_end': self.training_end, 'members': list(self.members), 'directions': '+1 without y',
            'required_per_row': self.required, 'threshold': self.threshold, 'available': self.available,
            'map_hashes': {c: array_hash(v) for c,v in self.references.items()}}


def fit_v_regime(training_x: pd.DataFrame, policy: dict) -> VRegime:
    """X-only API: neither training nor validation y is accepted."""
    columns = [c for c in training_x if c.startswith('V') and c[1:].isdigit()]
    eligible = eligibility(training_x[columns], policy)
    members = tuple(eligible.loc[eligible.eligible, 'feature'])
    refs = {c: np.sort(training_x[c].dropna().to_numpy(float)) for c in members}
    state = VRegime(members, refs, max(2, ceil(.5*len(members))), np.nan, False, int(training_x.index[-1]))
    p = state.proxy(training_x)
    finite = p[np.isfinite(p)]
    if len(finite) >= 252 and np.ptp(finite) > 0:
        state.threshold = float(np.median(finite))
        state.available = bool((finite > state.threshold).sum() >= 126 and (finite <= state.threshold).sum() >= 126)
    return state


class CandidateProvider:
    """A/B share this interface. A never calls the transformation/family engine."""
    def __init__(self, history: pd.DataFrame, config: dict, pipeline: str):
        if pipeline not in ('A', 'B'):
            raise ValueError('Only A Raw-Control and B Engineered are implemented.')
        self.config, self.pipeline = config, pipeline
        self.y = history[config['target']].reset_index(drop=True)
        self.raw = pd.concat([history[list(raw_names(config)[:94])].reset_index(drop=True),
            available_lagged_outcomes(history).reset_index(drop=True)], axis=1)[list(raw_names(config))]
        if pipeline == 'A':
            self.base = self.raw.copy()
            self.base.columns = [f'raw/{c}/Level' for c in self.raw]
        else:
            self.base = pd.concat({c: representations(self.raw[c]) for c in self.raw}, axis=1)
            self.base.columns = [f'raw/{c}/{t}' for c,t in self.base.columns]
        self.ids = tuple(self.base.columns) if pipeline == 'A' else candidate_ids(config)
        self.contexts: dict[int, tuple[pd.DataFrame, dict]] = {}

    def context(self, end: int) -> tuple[pd.DataFrame, dict]:
        if end not in self.contexts:
            meta = {'reference_end': end, 'pipeline': self.pipeline}
            extra = pd.DataFrame(index=self.raw.index)
            if self.pipeline == 'B':
                train = self.raw.iloc[:end+1]
                available = eligibility(train, self.config['availability'])
                regime = fit_v_regime(train, self.config['availability'])
                high = regime.high(self.raw)
                meta['regime'] = regime.audit()
                meta['families'] = {}
                for f in FAMILIES:
                    score, family_meta = family_composite(f, train, self.y.iloc[:end+1], self.raw, available)
                    family_meta['map_hashes'] = {c:array_hash(np.sort(train[c].dropna().to_numpy(float)))
                        for c in family_meta['qualified_members']}
                    extra[f'family/{f}/Level'] = score
                    extra[f'interaction/{f}/HighV'] = score.to_numpy()*high
                    meta['families'][f] = family_meta
                # Stable registry order independent of loop insertion order.
                extra = extra[[*map(lambda f: f'family/{f}/Level', FAMILIES),
                    *map(lambda f: f'interaction/{f}/HighV', FAMILIES)]]
            self.contexts[end] = extra, meta
        return self.contexts[end]

    def frame(self, end: int, start: int = 0, stop: int | None = None) -> pd.DataFrame:
        extra, _ = self.context(end)
        return pd.concat([self.base.iloc[start:stop], extra.iloc[start:stop]], axis=1)

    def registry(self) -> pd.DataFrame:
        rows = []
        for cid in self.ids:
            kind, source, transform = cid.split('/')
            rows.append({'pipeline': self.pipeline, 'candidate_id': cid, 'source': source,
                'transformation': transform, 'family': source[0] if kind == 'raw' and not source.startswith('lagged') else source if kind != 'raw' else None,
                'interaction_type': 'family_times_high_V' if kind == 'interaction' else None})
        return pd.DataFrame(rows)
