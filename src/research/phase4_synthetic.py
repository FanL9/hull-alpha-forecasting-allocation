"""Prospectively fixed v3.1 artificial generators; never reads market data."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.research.alpha_pipeline import load_alpha_design
from src.validation.protocol_v3_1 import load_config


def ar_factor(rng, n, coefficient):
    innovations = rng.normal(size=n)
    values = np.empty(n); values[0] = innovations[0]
    scale = np.sqrt(1-coefficient**2)
    for t in range(1,n):
        values[t] = coefficient*values[t-1]+scale*innovations[t]
    return values


def make_history(root: Path, seed: int, kind: str = 'noise') -> pd.DataFrame:
    c = load_config(root)
    if kind not in c['synthetic_calibration']['cases']:
        raise ValueError('Unregistered synthetic case.')
    design = load_alpha_design(root)
    n = c['research']['end']+1
    rng = np.random.default_rng(seed)
    values = {'date_id': np.arange(n)}
    for family, count in c['input_universe']['anonymous_counts'].items():
        factor = ar_factor(rng,n,.8)
        for i in range(1,count+1):
            values[f'{family}{i}'] = (rng.integers(0,2,n).astype(float) if family=='D'
                else np.sqrt(.5)*factor+np.sqrt(.5)*rng.normal(size=n))
    noise = rng.normal(size=n)
    settings = c['synthetic_calibration']
    if kind == 'noise':
        y = noise
    elif kind == 'raw':
        spec = settings['raw_plant']
        y = sum(b*values[f] for f,b in zip(spec['features'],spec['coefficients']))+spec['target_noise_std']*noise
    elif kind == 'transformed':
        spec = settings['transformed_plant']
        eps = rng.normal(scale=spec['innovation_std'],size=n)
        values[spec['feature']] = spec['trend_per_row']*np.arange(n)+eps
        change = pd.Series(eps).diff(21).to_numpy()/np.sqrt(2)
        y = spec['coefficient']*change+spec['target_noise_std']*noise
    else:
        spec = settings['regime_plant']
        latent = ar_factor(rng,n,spec['volatility_latent_ar'])
        common = rng.normal(size=n)
        for f in [f'V{i}' for i in range(1,14)]:
            values[f] = spec['V_loading']*latent+rng.normal(scale=spec['V_noise_std'],size=n)
        for f in [f'P{i}' for i in range(1,14)]:
            values[f] = common+rng.normal(scale=spec['P_common_noise_std'],size=n)
        y = spec['coefficient']*common*(latent>0)+spec['target_noise_std']*noise
    values[c['target']] = y
    values['forward_returns'] = rng.normal(size=n)
    values['risk_free_rate'] = rng.normal(size=n)
    result = pd.DataFrame(values,columns=design.source_columns)
    result.attrs.update(phase4_data_kind='synthetic',synthetic_case=kind,synthetic_seed=seed)
    return result
