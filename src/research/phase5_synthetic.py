"""Prospectively registered Phase 5 panels, never a market-data loader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.alpha_pipeline import load_alpha_design
from src.utils.paths import project_root


def make_history(case,seed,rows=7860):
    if case not in ('noise','linear','nonlinear') or rows not in (7860,9048):
        raise ValueError('Only registered artificial panel definitions are allowed.')
    rng = np.random.default_rng(seed)
    design = load_alpha_design(project_root())
    values = {'date_id':np.arange(rows)}
    for name in design.single_features[:94]:
        values[name] = rng.integers(0,2,rows).astype(float) if name.startswith('D') else rng.normal(size=rows)
    noise = rng.normal(size=rows)
    if case=='noise':
        y = noise
    elif case=='linear':
        m = np.empty(rows); m[0]=values['M1'][0]
        for t in range(1,rows):
            m[t]=.98*m[t-1]+np.sqrt(1-.98**2)*values['M1'][t]
        values['M1']=m
        y = pd.Series(m).rolling(63,min_periods=63).mean().to_numpy()+.1*noise
    else:
        m,e = values['M1'],values['E1']
        y = m+e+2*m*np.roll(e,1)+.2*noise
        y[0]=np.nan
    values['forward_returns'] = rng.normal(size=rows)
    values['risk_free_rate'] = rng.normal(size=rows)
    values['market_forward_excess_returns']=y
    return pd.DataFrame(values)[list(design.source_columns)]


def synthetic_project(directory,case='noise',seed=101):
    """Production-shaped split policy plus poison outside the research prefix."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    train = directory/'data/raw/train.csv'
    train.parent.mkdir(parents=True)
    frame = make_history(case,seed,9048).astype(object)
    frame.loc[7860:8867,frame.columns!='date_id']='INTERNAL_HOLDOUT_MUST_NOT_BE_PARSED'
    frame.loc[8868:9047,frame.columns!='date_id']='FINAL_180_MUST_NOT_BE_PARSED'
    frame.to_csv(train,index=False)
    policy = json.loads((project_root()/'configs/split_policy.json').read_text(encoding='utf-8'))
    policy['source_train_sha256']=hashlib.sha256(train.read_bytes()).hexdigest()
    policy['data_sha256']['data/raw/train.csv']=policy['source_train_sha256']
    (directory/'configs').mkdir()
    (directory/'configs/split_policy.json').write_text(json.dumps(policy,indent=2)+'\n',encoding='utf-8')
    return directory
