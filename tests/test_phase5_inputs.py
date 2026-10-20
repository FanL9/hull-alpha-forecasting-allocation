"""Artificial-only nested input and temporal boundary contracts."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from src.features.deep_inputs import average_clusters, fit_inputs, windows
from src.features.engineered_alpha import CandidateProvider
from src.research.phase5_synthetic import make_history
from src.utils.paths import project_root
from src.validation.protocol_v3_1 import load_config
from src.validation.protocol_v3 import inner_splits, stopping_split

@pytest.fixture(scope='module')
def panel():
    c=load_config(project_root()); h=make_history('nonlinear',307)
    return h,c,fit_inputs(h,5086,c)

def test_representatives_find_both_marginal_inputs(panel):
    _,_,state=panel
    assert {'M1','E1'}<=set(state.names)
    assert 1<=len(state.names)<=16

def test_representatives_scalers_outer_future_mutation(panel):
    h,c,state=panel; modified=h.copy()
    modified.loc[5087:,modified.columns!='date_id']=1e8
    other=fit_inputs(modified,5086,c)
    assert state.audit()==other.audit()

def test_stop_scaler_excludes_stopping_x_y(panel):
    h,c,_=panel; split=stopping_split(5086,c)
    first=fit_inputs(h,split.training_end,c)
    h=h.copy(); h.loc[split.purge_id:,h.columns!='date_id']=-1e9
    assert first.audit()==fit_inputs(h,split.training_end,c).audit()

def test_scaler_observed_training_only_ddof0(panel):
    h,c,s=panel; raw=CandidateProvider(h,c,'A').raw.iloc[:s.end+1]
    np.testing.assert_array_equal(s.means,np.nanmean(raw[list(s.names)],axis=0))
    np.testing.assert_array_equal(s.scales,np.nanstd(raw[list(s.names)],axis=0,ddof=0))
    y=h[c['target']].iloc[62:s.end+1].dropna().to_numpy()
    assert s.target_mean==y.mean() and s.target_scale==y.std(ddof=0)

def test_windows_exact_endpoints_no_tplus1(panel):
    h,c,state=panel; raw=CandidateProvider(h,c,'A').raw
    x=windows(raw,state,[62,5088])
    np.testing.assert_array_equal(x[0],state.transform(raw.iloc[:63]))
    np.testing.assert_array_equal(x[1],state.transform(raw.iloc[:5089])[-63:])
    raw=raw.copy(); raw.iloc[5089:]=1e12
    np.testing.assert_array_equal(x,windows(raw,state,[62,5088]))

@pytest.mark.parametrize('date',[-1,0,61,7860,8868,9047])
def test_windows_reject_pre_history_and_reserved_dates(panel,date):
    h,c,state=panel
    with pytest.raises(ValueError):
        windows(CandidateProvider(h,c,'A').raw,state,[date])

def test_missing_values_impute_zero_and_clip(panel):
    h,c,state=panel; raw=CandidateProvider(h,c,'A').raw
    raw=raw.copy(); raw.loc[0,state.names[0]]=np.nan; raw.loc[1,state.names[0]]=1e99
    x=state.transform(raw)
    assert x[0,0]==0 and x[1,0]==5
    assert np.isfinite(x).all() and abs(x).max()<=5
    empty=replace(state,names=state.names[:2],means=state.means[:2],scales=state.scales[:2])
    assert (empty.transform(raw)[:,2:]==0).all()

def test_constant_channel_unavailable(panel):
    h,c,state=panel
    state=replace(state,scales=np.zeros_like(state.scales))
    assert (state.transform(CandidateProvider(h,c,'A').raw)==0).all()

def test_average_linkage_absolute_and_ties():
    x=np.arange(300,dtype=float)
    frame=pd.DataFrame({'a':x,'b':-x,'c':np.roll(x,150),'d':np.nan})
    assert sorted(map(sorted,average_clusters(frame)))==[['a','b'],['c'],['d']]
    assert average_clusters(frame)==average_clusters(frame)

def test_every_inner_and_stopping_purge(panel):
    _,c,_=panel
    for end in range(5086,7607,252):
        for s in (*inner_splits(end,c),stopping_split(end,c)):
            assert s.training_end+1==s.purge_id==s.validation_start-1
            assert s.validation_end<=end and s.validation_end-s.validation_start+1==252
        assert stopping_split(end,c).training_end==end-253
