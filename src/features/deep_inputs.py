"""Frozen v3.1 shared representatives, prefix scalers and exact row-grid windows."""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
import pandas as pd

from src.features.engineered_alpha import CandidateProvider, eligibility, array_hash
from src.research.phase4_engine import NestedSelector, PairRanks


def average_clusters(x: pd.DataFrame, cut: float = .10) -> list[list[str]]:
    """Average linkage with explicit ascending current-node tie handling.

    Initial nodes follow the protocol raw-name order; newly merged nodes are
    assigned consecutive IDs. Distances are size-weighted averages of all
    original pair distances, never single linkage or correlation-of-centroids.
    """
    names = list(x)
    n = len(names)
    if not n:
        return []
    pairs = PairRanks(x)
    members = {i: [names[i]] for i in range(n)}
    sizes = dict.fromkeys(members, 1)
    distances, heap = {}, []
    for i in range(n):
        for j in range(i+1, n):
            rho, _ = pairs.pair(i, j)
            d = 1-abs(rho) if np.isfinite(rho) else 1.
            distances[i,j] = d
            heapq.heappush(heap, (d, i, j))
    next_id = n
    while heap:
        d, i, j = heapq.heappop(heap)
        if i not in members or j not in members:
            continue
        if d > cut:
            break
        others = sorted(set(members)-{i,j})
        members[next_id] = members[i]+members[j]
        sizes[next_id] = sizes[i]+sizes[j]
        for k in others:
            a = distances[tuple(sorted((i,k)))]
            b = distances[tuple(sorted((j,k)))]
            value = (sizes[i]*a+sizes[j]*b)/sizes[next_id]
            distances[k,next_id] = value
            heapq.heappush(heap, (value,k,next_id))
        del members[i], members[j]
        next_id += 1
    return [members[i] for i in sorted(members)]


@dataclass(frozen=True)
class InputState:
    end: int
    names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    target_mean: float
    target_scale: float
    clusters: tuple[tuple[str, ...], ...]
    scores: dict[str, float]

    def audit(self):
        return {'reference_end': self.end, 'representatives': list(self.names),
                'clusters': self.clusters, 'scores': self.scores,
                'means': self.means.tolist(), 'scales': self.scales.tolist(),
                'target_mean': self.target_mean, 'target_scale': self.target_scale,
                'input_features_sha256': array_hash(np.frombuffer('|'.join(self.names).encode(), dtype=np.uint8)),
                'target_fit_endpoints': [62,self.end], 'scaler_fit_rows': [0,self.end]}

    def transform(self, raw: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(raw),16), dtype=np.float64)
        for j, name in enumerate(self.names):
            if not np.isfinite(self.scales[j]) or self.scales[j] <= 1e-12:
                continue
            values = raw[name].to_numpy(float)
            result[:,j] = np.clip(np.where(np.isfinite(values), (values-self.means[j])/self.scales[j], 0), -5,5)
        return result


def fit_inputs(history: pd.DataFrame, end: int, config: dict) -> InputState:
    """Only the physically truncated fitting prefix is passed to selectors."""
    if not 62 <= end < len(history) or end > 7859:
        raise ValueError('Invalid input fitting reference.')
    prefix = history.iloc[:end+1].copy()
    provider = CandidateProvider(prefix, config, 'A')
    screen = NestedSelector(provider).fit(end).candidates.set_index('candidate_id')
    available = eligibility(provider.raw, config['availability'])
    names = available.loc[available.eligible, 'feature'].tolist()
    clusters = average_clusters(provider.raw[names])
    qualified = screen.loc[screen.eligible_all_references & screen.same_sign]
    scores = {cid.split('/')[1]: float(row.ranking_score) for cid,row in qualified.iterrows()}
    reps = []
    for cluster in clusters:
        pool = [n for n in cluster if n in scores]
        if pool:
            reps.append(min(pool, key=lambda n: (-scores[n],n)))
    reps = tuple(sorted(reps,key=lambda n: (-scores[n],n))[:16])
    a = provider.raw[list(reps)].to_numpy(float)
    means, scales = np.nanmean(a,axis=0), np.nanstd(a,axis=0,ddof=0)
    y = prefix[config['target']].iloc[62:].to_numpy(float)
    y = y[np.isfinite(y)]
    mean, scale = (float(y.mean()),float(y.std(ddof=0))) if len(y) else (np.nan,np.nan)
    return InputState(end,reps,means,scales,mean,scale,tuple(map(tuple,clusters)),
                      {n:scores[n] for n in reps})


def windows(raw: pd.DataFrame, state: InputState, endpoints) -> np.ndarray:
    dates = np.asarray(endpoints,dtype=int)
    if dates.ndim != 1 or np.any(dates < 62) or np.any(dates >= len(raw)) or np.any(dates > 7859):
        raise ValueError('Windows require 63 real rows and research-only endpoints.')
    # Transform only through the last legal source date, never t+1.
    x = state.transform(raw.iloc[:int(dates.max())+1])
    return x[dates[:,None]+np.arange(-62,1)[None,:]]


def training_arrays(raw, y, state):
    dates = np.arange(62,state.end+1)
    dates = dates[np.isfinite(np.asarray(y)[dates])]
    if not state.names or not np.isfinite(state.target_scale) or state.target_scale <= 1e-12:
        return None
    return (windows(raw,state,dates),
            (np.asarray(y)[dates]-state.target_mean)/state.target_scale, dates)
