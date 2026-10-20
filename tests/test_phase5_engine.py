"""Frozen OOF normalization, model comparison and fold-local sealing."""
import numpy as np
import pandas as pd
import pytest
from src.research.phase5_engine import Engine, combined, residualize, normalization, primary_selection
from src.research.phase5_tasks import TaskStore, encode
from src.research.phase5_synthetic import make_history
from src.research.alpha_pipeline import load_alpha_design
from src.validation.protocol_v3_1 import load_config
from src.utils.paths import project_root

def make_engine(tmp_path,h=None):
    root=project_root()
    return Engine(make_history('nonlinear',307) if h is None else h,load_config(root),load_alpha_design(root),
                  TaskStore(tmp_path,{'run_type':'synthetic'}),test_budget=1)

def test_reference_dag_complete_three_nested_layers(tmp_path):
    e=make_engine(tmp_path)
    assert len(e.references)==36 and min(e.references)==4076 and max(e.references)==7606
    assert len(e.outer)==11

def test_worker_capsule_has_only_legal_labels_and_deterministic_plan(tmp_path):
    e=make_engine(tmp_path/'a'); job=e.job(5086,'GRU',11)
    assert job['stop_range']==[0,4833] and job['stopping_range']==[4835,5086]
    assert job['refit_range']==[0,5086] and job['forecast_range']==[5088,5339]
    assert job['forecast_x'].shape==(252,63,16)
    h=e.history.copy(); h.loc[5087:,h.columns!='date_id']=1e8
    b=make_engine(tmp_path/'b',h).job(5086,'GRU',11)
    for k in ('stop_x','stop_y','validation_x','validation_y','refit_x','refit_y','task',
              'stop_feature_hash','refit_feature_hash','target_mean','target_scale'):
        assert encode(job[k])==encode(b[k])

def test_same_fold_y_excluded_but_past_lagged_y_legal(tmp_path):
    e=make_engine(tmp_path/'a'); job=e.job(5086,'TCN',22)
    h=e.history.copy(); h.loc[5088:5339,'market_forward_excess_returns']=-999.
    other=make_engine(tmp_path/'b',h).job(5086,'TCN',22)
    # Online official lagged outcomes can change later X after realization;
    # fitting state and the first validation prediction's window cannot.
    assert encode(job['refit_y'])==encode(other['refit_y'])
    assert job['refit_feature_hash']==other['refit_feature_hash']
    np.testing.assert_array_equal(job['forecast_x'][0],other['forecast_x'][0])

def test_ab_fit_target_series_is_physically_prefix_bounded(tmp_path,monkeypatch):
    from src.research import phase5_engine as mod
    original=mod.NestedSelector
    seen=[]
    class Audited(original):
        def fit(self,end):
            seen.append((end,len(self.provider.y)))
            return super().fit(end)
    monkeypatch.setattr(mod,'NestedSelector',Audited)
    e=make_engine(tmp_path); e.ab('B',5086)
    assert seen and all(n==5087 for _,n in seen)

def test_normalizers_no_outer_fit_and_common_rows_only():
    x=np.linspace(-1,1,504); inner=[x,2*x+3,np.sin(x)]
    inner[0]=inner[0].copy(); inner[0][:5]=np.nan
    outer=[np.arange(252)*10.,np.ones(252)*100,np.arange(252)*-4.]
    _,a=combined(inner,outer)
    _,b=combined(inner,[v*1000 for v in outer])
    assert encode(a)==encode(b) and a['pairs']==499
    np.testing.assert_array_equal(a['means'],np.column_stack(inner)[5:].mean(axis=0))

def test_residualizer_no_y_regressand_and_frozen_coefficients():
    rng=np.random.default_rng(19); a,b=rng.normal(size=(2,504)); deep=2*a-3*b+.1*rng.normal(size=504)
    residual,state=residualize([a,b,deep],[a[:252],b[:252],deep[:252]])
    assert state['pairs']==504 and len(state['beta'])==3
    z=(np.column_stack([a,b,deep])-state['means'])/state['scales']
    expected=z[:252,2]-np.column_stack([np.ones(252),z[:252,:2]])@state['beta']
    np.testing.assert_array_equal(residual,expected)

@pytest.mark.parametrize('kind',['too_few','zero_scale','missing'])
def test_no_normalization_or_combination_fallback(kind):
    n=251 if kind=='too_few' else 504
    a=np.arange(n,dtype=float); b=a.copy(); c=a.copy()
    if kind=='zero_scale': c[:]=0
    if kind=='missing': c[:]=np.nan
    out,state=combined([a,b,c],[a[:100],b[:100],c[:100]])
    assert state is None and np.isnan(out).all()

def test_failed_seed_disables_entire_architecture_and_c(tmp_path):
    e=make_engine(tmp_path)
    for name in ('GRU','TCN'):
        for seed in (11,22,33,44,55):
            e.neural[5086,name,seed]={'status':'active','prediction':np.ones(252)*seed}
    assert np.all(e.deep('GRU',5086)==33)
    e.neural[5086,'GRU',22]['status']='unavailable'
    assert np.isnan(e.deep('GRU',5086)).all() and np.isnan(e.deep('C',5086)).all()

def table(means):
    return pd.DataFrame([{'pipeline':name,'fold_id':i,'outer_rank_ic':value} for name,value in means.items() for i in range(11)])

def test_primary_near_best_global_anchor_strict_margin():
    result=primary_selection(table({'A':.02,'B':.025,'C':.04,'D':.045}))
    assert result['selected']=='C' and result['holdout_unlocked'] is False
    assert primary_selection(table({'A':.01,'B':.02,'C':.04,'D':.03}))['selected']=='C'

def test_primary_no_admissible_stops():
    result=primary_selection(table({'A':-.01,'B':np.nan,'C':0,'D':-.03}))
    assert result['selected'] is None and result['status']=='no_admissible_primary'

def test_real_mode_rejects_test_compute_budget(tmp_path):
    root=project_root()
    with pytest.raises(ValueError,match='synthetic'):
        Engine(make_history('noise',101),load_config(root),load_alpha_design(root),TaskStore(tmp_path,{'run_type':'real'}),test_budget=2)
