"""Artificial-only causal transformation, availability and regime contracts."""
from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import pytest

from src.features import engineered_alpha as ea
from src.features.availability import feature_availability
from src.research.phase4_synthetic import make_history
from src.validation.protocol_v3_1 import load_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def config():
    return load_config(ROOT)


@pytest.mark.parametrize('transform',['Level','Delta5','Delta21','Z252','Rank63'])
def test_future_mutation_is_causal(transform):
    rng=np.random.default_rng(8); x=pd.Series(rng.normal(size=600))
    x.iloc[::17]=np.nan
    changed=x.copy(); changed.iloc[300:]=rng.normal(size=300)*1e9
    pd.testing.assert_series_equal(ea.representations(x)[transform].iloc[:300],
        ea.representations(changed)[transform].iloc[:300])


@pytest.mark.parametrize('window',[5,21])
def test_delta_uses_original_row_grid(window):
    x=pd.Series(np.arange(300,dtype=float)); x.iloc[12]=np.nan
    result=ea.representations(x)[f'Delta{window}']
    pd.testing.assert_series_equal(result,(x-x.shift(window)).rename(result.name))
    assert np.isnan(result.iloc[12+window])


def test_zscore_current_included_ddof_zero():
    x=pd.Series(np.arange(400,dtype=float))
    z=ea.representations(x).Z252
    assert z.iloc[:125].isna().all()
    assert z.iloc[125]==pytest.approx((125-np.mean(np.arange(126)))/np.std(np.arange(126)))
    assert z.iloc[300]==pytest.approx((300-np.mean(np.arange(49,301)))/np.std(np.arange(49,301)))
    assert ea.representations(pd.Series(np.ones(400))).Z252.isna().all()


def test_rank_observed_queue_ties_and_missing_current():
    x=pd.Series(np.tile([0.,1.,1.,2.,np.nan],60))
    rank=ea.representations(x).Rank63
    assert rank[x.isna()].isna().all()
    for t in [38,42,98,172,298]:
        observed=x.iloc[:t+1].dropna().iloc[-63:]
        expected=((observed<x.iloc[t]).sum()+.5*(observed==x.iloc[t]).sum())/len(observed)
        assert rank.iloc[t]==pytest.approx(expected)
    assert rank.iloc[:38].isna().all()  # fewer than32 observations


@pytest.mark.parametrize('case',['leading','intermittent','late','recent','constant','absent'])
def test_generalized_availability_matches_frozen_raw_rule(config,case):
    x=pd.Series(np.arange(1000,dtype=float))
    if case=='leading':x.iloc[:200]=np.nan
    elif case=='intermittent':x.iloc[::3]=np.nan
    elif case=='late':x.iloc[:800]=np.nan
    elif case=='recent':x.iloc[-200:]=np.nan
    elif case=='constant':x[:]=1
    else:x[:]=np.nan
    policy=config['availability']
    old={'min_nonmissing_observations':policy['min_observed'],
        'min_historical_coverage_since_first_observation':policy['min_post_launch_coverage'],
        'recent_window':policy['recent_window'],'max_recent_missing_ratio':policy['max_recent_missing'],
        'min_distinct_values':policy['min_distinct']}
    a=feature_availability(pd.DataFrame({'M1':x}),old).iloc[0]
    b=ea.eligibility(pd.DataFrame({'raw/M1/Z252':x}),policy).iloc[0]
    for field in ['observed','post_launch_coverage','recent_missing_ratio','distinct_observed','eligible','reasons']:
        if isinstance(a[field],float): assert b[field]==pytest.approx(a[field])
        else: assert a[field]==b[field]


def test_transform_rechecks_eligibility(config):
    x=pd.Series(np.arange(370,dtype=float)); r=ea.representations(x)
    result=ea.eligibility(r,config['availability']).set_index('feature')
    assert result.loc['Level','eligible']
    assert not result.loc['Z252','eligible']  #245 observed, raw has370
    assert result.loc['Z252','observed']==245


def test_registry_exact_499_and_raw97_no_engineering(config,monkeypatch):
    h=make_history(ROOT,101)
    b=ea.CandidateProvider(h,config,'B')
    reg=b.registry()
    assert len(reg)==499 and reg.candidate_id.is_unique
    assert reg.candidate_id.str.startswith('raw/').sum()==485
    assert reg.candidate_id.str.startswith('interaction/').sum()==7
    def forbidden(*a,**k):raise AssertionError('Raw A must not invoke engineering')
    monkeypatch.setattr(ea,'representations',forbidden)
    monkeypatch.setattr(ea,'family_composite',forbidden)
    a=ea.CandidateProvider(h,config,'A'); frame=a.frame(1000)
    assert len(a.ids)==97 and frame.shape==(7860,97)
    assert all(c.endswith('/Level') for c in a.ids)


def test_regime_has_no_target_argument_and_training_only_maps(config):
    assert 'y' not in inspect.signature(ea.fit_v_regime).parameters
    h=make_history(ROOT,11,'regime')
    x=h[[f'V{i}' for i in range(1,14)]]
    state=ea.fit_v_regime(x.iloc[:4000],config['availability'])
    assert state.available
    changed=x.copy(); changed.iloc[4000:]=1e8
    other=ea.fit_v_regime(changed.iloc[:4000],config['availability'])
    assert state.audit()==other.audit()
    assert state.threshold==pytest.approx(np.nanmedian(state.proxy(x.iloc[:4000])))
    assert np.nanmean(state.high(changed.iloc[4000:]))==1
    # Applying future extremes cannot change the previously fitted threshold.
    assert state.audit()==other.audit()


def test_unavailable_regime_disables_all_interactions(config):
    h=make_history(ROOT,101)
    h[[f'V{i}' for i in range(1,14)]]=1.
    p=ea.CandidateProvider(h,config,'B')
    extra,meta=p.context(3000)
    assert not meta['regime']['available']
    assert extra.filter(like='interaction/').isna().all().all()


def test_regime_invariant_even_to_training_y_mutation(config):
    h=make_history(ROOT,11,'regime')
    other=h.copy();other[config['target']]*=-1000
    p=ea.CandidateProvider(h,config,'B');q=ea.CandidateProvider(other,config,'B')
    assert p.context(2000)[1]['regime']==q.context(2000)[1]['regime']


def test_lagged_information_uses_previous_row(config):
    h=make_history(ROOT,101); p=ea.CandidateProvider(h,config,'A')
    assert np.isnan(p.raw['lagged_market_forward_excess_returns'].iloc[0])
    np.testing.assert_array_equal(p.raw['lagged_market_forward_excess_returns'].iloc[1:],h[config['target']].iloc[:-1])
