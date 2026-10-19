"""Selector isolation and numerical contracts using artificial inputs only."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.features.engineered_alpha import CandidateProvider, eligibility
from src.research.phase4_engine import (NestedSelector,FittedSelector,PairRanks,
    corr,column_ics,greedy_select,evaluate)
from src.research.phase4_synthetic import make_history
from src.validation.protocol_v3 import inner_splits
from src.validation.protocol_v3_1 import load_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def config():return load_config(ROOT)


@pytest.fixture(scope='module')
def histories():return {kind:make_history(ROOT,11,kind) for kind in ['noise','raw','transformed','regime']}


def test_inner_boundaries(config):
    splits=inner_splits(5086,config)
    assert [(s.training_end,s.purge_id,s.validation_start,s.validation_end) for s in splits]==[
        (4581,4582,4583,4834),(4833,4834,4835,5086)]


@pytest.mark.parametrize('pipeline',['A','B'])
def test_outer_y_isolation_and_determinism(config,histories,pipeline):
    h=histories['raw']; changed=h.copy();changed.loc[5088:,config['target']]*=-1e6
    first=NestedSelector(CandidateProvider(h,config,pipeline)).fit(5086)
    second=NestedSelector(CandidateProvider(changed,config,pipeline)).fit(5086)
    assert first.audit()==second.audit()
    pd.testing.assert_frame_equal(first.candidates,second.candidates)
    p=CandidateProvider(h,config,pipeline)
    third=NestedSelector(p).fit(5086)
    np.testing.assert_array_equal(first.predict(p.frame(5086,5088,5340)),third.predict(p.frame(5086,5088,5340)))
    assert len(first.selected)<=16


def test_future_x_cannot_change_selection_or_maps(config,histories):
    h=histories['raw']; changed=h.copy()
    changed.loc[5088:,[c for c in h if c[0] in 'MEIPVSD' and c[1:].isdigit()]]=1e9
    p=CandidateProvider(h,config,'B'); q=CandidateProvider(changed,config,'B')
    a=NestedSelector(p).fit(5086); b=NestedSelector(q).fit(5086)
    assert a.audit()==b.audit()
    assert p.context(5086)[1]==q.context(5086)[1]
    pd.testing.assert_frame_equal(p.frame(5086,0,5087),q.frame(5086,0,5087))


def test_identical_selector_for_identical_candidate_universe(config,histories):
    # B's provider restricted to the SAME97 Level IDs must reproduce A exactly.
    a=CandidateProvider(histories['raw'],config,'A')
    b=CandidateProvider(histories['raw'],config,'B')
    original=b.frame
    b.frame=lambda end,start=0,stop=None:original(end,start,stop)[list(a.ids)]
    b.ids=a.ids
    left,right=NestedSelector(a).fit(5086),NestedSelector(b).fit(5086)
    assert left.audit()==right.audit()
    pd.testing.assert_frame_equal(left.candidates,right.candidates)


def test_raw_signal_signs(config,histories):
    s=NestedSelector(CandidateProvider(histories['raw'],config,'A')).fit(5086)
    assert s.signs['raw/M1/Level']==1
    assert s.signs['raw/E1/Level']==-1
    assert s.signs['raw/P1/Level']==1


def test_exact_pairwise_deletion_before_ranking():
    rng=np.random.default_rng(7);a=rng.normal(size=600);b=a+rng.normal(size=600)
    a[::5]=np.nan;b[::7]=np.nan
    rho,n=PairRanks(pd.DataFrame({'a':a,'b':b})).pair(0,1)
    mask=np.isfinite(a)&np.isfinite(b)
    assert n==mask.sum() and rho==pytest.approx(spearmanr(a[mask],b[mask]).statistic)


def test_redundancy_skip_and_continue():
    rng=np.random.default_rng(1);a=rng.normal(size=1000)
    x=pd.DataFrame({'first':a,'duplicate':-a,'independent':rng.normal(size=1000)})
    selected,audit=greedy_select(list(x),x)
    assert selected==['first','independent']
    assert audit[0]['candidate_id']=='duplicate' and audit[0]['rejected_with']=='first'
    assert audit[0]['rho']==pytest.approx(-1)


def test_unknown_redundancy_rejects_lower():
    x=pd.DataFrame({'a':np.r_[np.arange(300.),np.full(400,np.nan)],
        'b':np.r_[np.full(400,np.nan),np.arange(300.)]})
    selected,audit=greedy_select(list(x),x)
    assert selected==['a'] and audit[0]['reason']=='unknown_pair'


def test_top16_and_deterministic_tie_rule(config,histories):
    h=histories['noise'].copy();h['M2']=h['M1']
    s=NestedSelector(CandidateProvider(h,config,'A')).fit(5086)
    assert len(s.selected)<=16
    audit=s.candidates.set_index('candidate_id')
    if audit.loc['raw/M1/Level','same_sign']:
        assert audit.loc['raw/M1/Level','ranking_position']<audit.loc['raw/M2/Level','ranking_position']
    assert not {'raw/M1/Level','raw/M2/Level'}.issubset(s.selected)


def test_fixed_weights_missing_contribution_and_no_alpha():
    names=('a','b','c','d')
    s=FittedSelector(9,names,dict.fromkeys(names,1),dict.fromkeys(names,np.array([0.,1.])),pd.DataFrame(),[])
    x=pd.DataFrame({'a':[1.,1.],'b':[1.,np.nan],'c':[np.nan,np.nan],'d':[np.nan,np.nan]})
    assert s.predict(x)[0]==pytest.approx(.25) #2*0.75-1=.5, two terms /4
    assert np.isnan(s.predict(x)[1])
    s.selected=('a',)
    np.testing.assert_array_equal(s.predict(x),np.zeros(2))


def test_column_ic_nan_ties_matches_scipy():
    rng=np.random.default_rng(3);x=pd.DataFrame(rng.integers(0,4,(252,4)).astype(float));y=rng.normal(size=252)
    x.iloc[::4,1]=np.nan;x.iloc[:150,2]=np.nan;x[3]=1.
    rho,n=column_ics(x,y)
    for j in [0,1]:
        m=x[j].notna();assert rho[j]==pytest.approx(spearmanr(x.loc[m,j],y[m]).statistic)
    assert np.isnan(rho[2]) and np.isnan(rho[3])


def test_pearson_uses_earlier_complete_selectors(config,histories):
    engine=NestedSelector(CandidateProvider(histories['raw'],config,'A'))
    bounds,sources=engine.inner_oof_bounds(5086)
    assert [(s['fit_end'],s['start'],s['end']) for s in sources]==[(4581,4583,4834),(4833,4835,5086)]
    assert all(s['selector']['reference_end']==s['fit_end'] for s in sources)
    scores=np.concatenate([engine.forecast(s['fit_end'],s['start'],s['end']+1) for s in sources])
    np.testing.assert_allclose(bounds,np.quantile(scores[np.isfinite(scores)],[.01,.99]))
    assert 5086 not in engine.cache  # no outer-fitted selector substituted


def test_paired_evaluation_excludes_missing_dates_and_constant_pipeline():
    data=[];rng=np.random.default_rng(6)
    for fold in [1,2]:
        y=rng.normal(size=252)
        for pipeline in ['A','B']:
            score=y+rng.normal(size=252)
            if pipeline=='B':score[:10]=np.nan
            data.append(pd.DataFrame({'pipeline':pipeline,'fold_id':fold,'date_id':np.arange(252)+252*(fold-1),
                'prediction':score,'target':y,'status':'active','clip_lower':-2.,'clip_upper':2.}))
    folds,summary=evaluate(pd.concat(data))
    assert (folds.paired_count==242).all() and (summary.finite_folds==2).all()
    data[1]['prediction']=0.
    folds,_=evaluate(pd.concat(data))
    assert not folds.loc[folds.pipeline=='B','structurally_available_all_folds'].any()
    assert folds.loc[folds.pipeline=='B','outer_rank_ic'].isna().all()


@pytest.mark.parametrize('case',['noise','raw','transformed','regime'])
def test_generators_deterministic_exact_schema_and_boundaries(case):
    a=make_history(ROOT,11,case);b=make_history(ROOT,11,case)
    pd.testing.assert_frame_equal(a,b)
    assert a.shape==(7860,98) and a.date_id.iloc[-1]==7859
    assert a.attrs['phase4_data_kind']=='synthetic'
