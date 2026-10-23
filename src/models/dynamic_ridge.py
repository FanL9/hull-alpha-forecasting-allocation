"""Protocol v5 DynRidge: recency-weighted ridge on the 97 raw levels.

Pure numpy over caller-supplied prefixes. A fit at reference end e reads only
raw rows 0..e and labels 0..e; a forecast at t reads only raw row t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.engineered_alpha import eligibility
from src.research.phase4_engine import corr

ALPHAS = (0.1, 1.0, 10.0, 100.0)
HALF_LIFE = 504
CLIP = 5.0
BLOCK_ROWS, BLOCKS = 252, 2


@dataclass(frozen=True)
class DynRidgeState:
    end: int
    alpha: float
    names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    target_mean: float
    target_scale: float
    intercept: float
    coef: np.ndarray

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        return (self.intercept + scaled(raw, self)@self.coef)*self.target_scale + self.target_mean

    def audit(self) -> dict:
        return {'reference_end': self.end, 'alpha': self.alpha, 'inputs': list(self.names),
                'target_mean': self.target_mean, 'target_scale': self.target_scale, 'intercept': self.intercept,
                'coef': self.coef.tolist()}


def scaled(raw: pd.DataFrame, state) -> np.ndarray:
    x = raw[list(state.names)].to_numpy(float)
    z = (x - state.means)/state.scales
    return np.clip(np.where(np.isfinite(z), z, 0.0), -CLIP, CLIP)


def inner_blocks(end: int, rows: int = BLOCK_ROWS, blocks: int = BLOCKS) -> tuple[tuple[int, int, int], ...]:
    """(training_end, validation_start, validation_end) in the last 504 rows, purge 1."""
    first = end - blocks*rows + 1
    if first < 2:
        raise ValueError('Insufficient history for the inner blocks.')
    return tuple((start-2, start, start+rows-1) for start in range(first, end+1, rows))


def fit(raw: pd.DataFrame, y, end: int, availability: dict, alphas=ALPHAS, half_life=HALF_LIFE) -> dict | None:
    """One state per alpha at reference end; None when no input or label is usable."""
    train = raw.iloc[:end+1]
    labels = np.asarray(y, float)[:end+1]
    usable = eligibility(train, availability)
    names = [n for n in usable.loc[usable.eligible, 'feature'] if np.nanstd(train[n].to_numpy(float)) > 1e-12]
    rows = np.flatnonzero(np.isfinite(labels))
    if not names or len(rows) < BLOCK_ROWS:
        return None
    x = train[names].to_numpy(float)
    means, scales = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    target_mean, target_scale = float(labels[rows].mean()), float(labels[rows].std())
    if not target_scale > 1e-12:
        return None
    probe = DynRidgeState(end, np.nan, tuple(names), means, scales, target_mean, target_scale, 0.0, np.zeros(len(names)))
    z = scaled(train.iloc[rows], probe)
    t = (labels[rows] - target_mean)/target_scale
    w = 0.5**((end - rows)/half_life)
    w = w/w.sum()
    zbar, tbar = w@z, float(w@t)
    zc, tc = z - zbar, t - tbar
    gram, rhs = (zc*w[:, None]).T@zc, (zc*w[:, None]).T@tc
    out = {}
    for alpha in alphas:
        coef = np.linalg.solve(gram + alpha*np.eye(len(names)), rhs)
        out[alpha] = DynRidgeState(end, float(alpha), tuple(names), means, scales, target_mean, target_scale,
                                   float(tbar - zbar@coef), coef)
    return out


class DynRidge:
    """Reference-end fits with alpha chosen on two inner blocks; caches by reference."""

    def __init__(self, raw: pd.DataFrame, y, availability: dict):
        self.raw, self.y, self.availability = raw, np.asarray(y, float), availability
        self.fits: dict[int, dict | None] = {}
        self.choices: dict[int, dict] = {}

    def fitted(self, end: int):
        if end not in self.fits:
            self.fits[end] = fit(self.raw, self.y, end, self.availability)
        return self.fits[end]

    def choose(self, end: int) -> dict:
        if end not in self.choices:
            scores = []
            for alpha in ALPHAS:
                ics = []
                for training_end, start, stop in inner_blocks(end):
                    states = self.fitted(training_end)
                    pred = states[alpha].predict(self.raw.iloc[start:stop+1]) if states else np.full(stop-start+1, np.nan)
                    ics.append(corr(pred, self.y[start:stop+1], True))
                scores.append((alpha, float(np.mean(ics)) if np.isfinite(ics).all() else np.nan))
            valid = [s for s in scores if np.isfinite(s[1])]
            alpha = None
            if valid and self.fitted(end) is not None:
                best = max(s for _, s in valid)
                alpha = max(a for a, s in valid if best - s <= 1e-12)
            self.choices[end] = {'alpha': alpha, 'inner_ics': scores}
        return self.choices[end]

    def forecast(self, end: int, start: int, stop: int) -> np.ndarray:
        if not end < start <= stop-1 < len(self.raw):
            raise ValueError('DynRidge forecasts must follow the reference end.')
        alpha = self.choose(end)['alpha']
        if alpha is None:
            return np.full(stop-start, np.nan)
        return self.fitted(end)[alpha].predict(self.raw.iloc[start:stop])

    def audit(self, end: int) -> dict:
        choice = self.choose(end)
        state = self.fitted(end)[choice['alpha']] if choice['alpha'] is not None else None
        return {**choice, 'state': state.audit() if state else None}
