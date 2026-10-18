"""Phase 2.5-v2 rank, winsorization, composite and multiplicity rules on synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from src.features.availability import LAGGED_OUTCOME_COLUMNS, feature_availability
from src.features.family_composites import (COMPOSITE_FAMILIES, family_composite, family_composites,
                                            training_direction, training_percentile)
from src.statistics.multiple_testing import adjust_fdr, family_pvalues, protocol_multiplicity
from src.statistics.robust_inference import (apply_winsor_bounds, blockwise_rank, blockwise_rank_pairs,
                                            blockwise_spearman_hac, blockwise_spearman_interval,
                                            correlation_hac, fit_winsor_bounds)


# Block-wise ranking

def test_blockwise_rank_is_within_block_with_average_ties_and_nan_positions():
    values=[3.,1.,np.nan,1.,100.,300.,200.,np.nan]
    blocks=[1,1,1,1,2,2,2,2]
    ranks=blockwise_rank(values,blocks)
    np.testing.assert_allclose(ranks[:4],[3/4,1.5/4,np.nan,1.5/4])
    np.testing.assert_allclose(ranks[4:],[1/4,3/4,2/4,np.nan])
    assert np.isnan(ranks[[2,7]]).all()


def test_blockwise_differs_from_global_ranking_when_levels_shift():
    rng=np.random.default_rng(0)
    x=np.r_[rng.normal(0,1,100),rng.normal(50,1,100)]
    y=np.r_[rng.normal(0,1,100),rng.normal(50,1,100)]
    blocks=np.repeat([1,2],100)
    global_rho=correlation_hac(x,y,method='spearman').correlation
    blockwise=blockwise_spearman_hac(x,y,blocks)
    assert global_rho>.7
    assert abs(blockwise.correlation)<.2 and blockwise.method=='blockwise_spearman'
    rx,ry=blockwise_rank_pairs(x,y,blocks)
    assert blockwise.correlation==pytest.approx(np.corrcoef(rx,ry)[0,1])


def test_blockwise_rank_ignores_other_blocks_and_never_reranks():
    blocks=np.repeat([1,2,3],5)
    base=np.arange(15,dtype=float)
    changed=base.copy()
    changed[10:]=[-1e6,5e6,0,3,2]
    np.testing.assert_allclose(blockwise_rank(base,blocks)[:10],blockwise_rank(changed,blocks)[:10])
    assert np.allclose(blockwise_rank(base,blocks)[:5],blockwise_rank(base,blocks)[5:10])


def test_blockwise_rank_pairs_use_complete_pairs_and_keep_time_positions():
    x=np.array([1.,2.,np.nan,4.,5.,6.])
    y=np.array([1.,np.nan,3.,4.,6.,5.])
    rx,ry=blockwise_rank_pairs(x,y,[1,1,1,2,2,2])
    assert np.isnan(rx[[1,2]]).all() and np.isnan(ry[[1,2]]).all()
    np.testing.assert_allclose(rx[[0,3,4,5]],[.5,.25,.5,.75])
    np.testing.assert_allclose(ry[[3,4,5]],[.25,.75,.5])
    result=blockwise_spearman_hac(x,y,[1,1,1,2,2,2],lags=1)
    assert result.pairs==4


def test_noncontiguous_blocks_rejected():
    with pytest.raises(ValueError,match='contiguous'):
        blockwise_rank([1.,2.,3.],[1,2,1])


def test_bootstrap_interval_runs_on_blockwise_ranked_series():
    rng=np.random.default_rng(3)
    x=rng.normal(size=120)
    y=x+rng.normal(size=120)
    blocks=np.repeat([1,2,3],40)
    interval=blockwise_spearman_interval(x,y,blocks,block_length=10,repetitions=200,seed=42)
    rx,ry=blockwise_rank_pairs(x,y,blocks)
    from src.statistics.robust_inference import moving_block_correlation_interval
    direct=moving_block_correlation_interval(rx,ry,10,200,.95,42)
    assert (interval['lower'],interval['upper'])==(direct['lower'],direct['upper'])
    assert interval['method']=='blockwise_spearman' and 'pvalue' not in interval


# Training-window winsorization

def test_winsor_bounds_come_from_training_only_and_clip_validation():
    train=pd.DataFrame({'E8':np.r_[np.arange(1.,101.),np.nan]})
    validation=pd.DataFrame({'E8':[-1e6,50.,np.nan,1e6]})
    bounds=fit_winsor_bounds(train)
    expected=train['E8'].quantile([.01,.99]).to_numpy()
    np.testing.assert_allclose(bounds.loc['E8',['lower','upper']],expected)
    clipped=apply_winsor_bounds(validation,bounds)
    np.testing.assert_allclose(clipped['E8'],[expected[0],50.,np.nan,expected[1]])
    other=pd.DataFrame({'E8':[1e9,-1e9]})
    pd.testing.assert_frame_equal(fit_winsor_bounds(train),bounds)
    assert apply_winsor_bounds(other,bounds)['E8'].between(*expected).all()
    np.testing.assert_allclose(apply_winsor_bounds(train,bounds)['E8'].dropna().agg(['min','max']),expected)


def test_winsorization_never_touches_target():
    with pytest.raises(ValueError,match='target'):
        fit_winsor_bounds(pd.DataFrame({'M1':[1.,2.],'market_forward_excess_returns':[.1,.2]}))
    bounds=fit_winsor_bounds(pd.DataFrame({'M1':[1.,2.,3.]}))
    with pytest.raises(ValueError,match='target'):
        apply_winsor_bounds(pd.DataFrame({'market_forward_excess_returns':[.1]}),bounds)
    y=pd.Series([-.04,.0,.04],name='market_forward_excess_returns')
    before=y.copy()
    apply_winsor_bounds(pd.DataFrame({'M1':[0.,9.,2.]}),bounds)
    pd.testing.assert_series_equal(y,before)


def test_undefined_bounds_are_not_silently_applied():
    bounds=fit_winsor_bounds(pd.DataFrame({'M1':[np.nan,np.nan]}))
    with pytest.raises(ValueError,match='not testable'):
        apply_winsor_bounds(pd.DataFrame({'M1':[1.]}),bounds)


# Family composites

POLICY={'min_nonmissing_observations':5,'min_historical_coverage_since_first_observation':.8,
        'recent_window':10,'max_recent_missing_ratio':.25,'min_distinct_values':2}


def synthetic_fold(n=40,seed=1):
    rng=np.random.default_rng(seed)
    y=pd.Series(rng.normal(size=n))
    x=pd.DataFrame({'M1':y+rng.normal(scale=.1,size=n),'M2':-y+rng.normal(scale=.1,size=n),
                    'M3':y+rng.normal(scale=.5,size=n),'M4':np.r_[np.full(n-5,np.nan),np.arange(5.)],
                    'D1':(y>0).astype(float),'D2':(y<0).astype(float),'D3':(y>.5).astype(float),
                    'lagged_forward_returns':y.shift(1)})
    return x,y


def test_direction_comes_from_training_spearman_sign_only():
    x,y=synthetic_fold()
    assert training_direction(x['M1'],y)==1 and training_direction(x['M2'],y)==-1
    assert training_direction(pd.Series([1.,2.,3.,4.]),pd.Series([1.,2.,2.,1.]))==0
    assert training_direction(pd.Series([1.,1.,1.]),pd.Series([1.,2.,3.]))==0


def test_training_percentile_is_mid_rank_ecdf_of_training_values():
    train=pd.Series([1.,2.,2.,4.])
    np.testing.assert_allclose(training_percentile(train,train),[.125,.5,.5,.875])
    np.testing.assert_allclose(training_percentile(train,pd.Series([0.,3.,9.,np.nan])),[0.,.75,1.,np.nan])


def test_composite_is_equal_weight_mean_of_oriented_training_percentiles():
    x,y=synthetic_fold()
    train,validation=x.iloc[:30],x.iloc[30:]
    availability=feature_availability(train,POLICY)
    composite,meta=family_composite('M',train,y.iloc[:30],validation,availability)
    assert meta['qualified_members']==['M1','M2','M3'] and 'M4' in meta['excluded_unavailable']
    assert meta['directions']=={'M1':1,'M2':-1,'M3':1} and meta['required_per_row']==2
    expected=pd.concat([training_percentile(train['M1'],validation['M1']),
                        -training_percentile(train['M2'],validation['M2']),
                        training_percentile(train['M3'],validation['M3'])],axis=1).mean(axis=1)
    np.testing.assert_allclose(composite,expected)


def test_validation_values_never_change_fitted_composite_rules():
    x,y=synthetic_fold()
    train,validation=x.iloc[:30],x.iloc[30:]
    availability=feature_availability(train,POLICY)
    first,meta=family_composite('M',train,y.iloc[:30],validation,availability)
    altered=validation.copy()
    altered.iloc[1:]=altered.iloc[1:]*1000
    second,meta2=family_composite('M',train,y.iloc[:30],altered,availability)
    assert meta==meta2 and first.iloc[0]==second.iloc[0]


def test_row_minimum_members_and_fraction_rule():
    rng=np.random.default_rng(7)
    y=pd.Series(rng.normal(size=60))
    train=pd.DataFrame({f'M{i}':y+rng.normal(scale=.2,size=60) for i in range(1,9)})
    availability=feature_availability(train,POLICY)
    validation=pd.DataFrame([[1.]*8,[1.]*4+[np.nan]*4,[1.]*3+[np.nan]*5],columns=train.columns)
    composite,meta=family_composite('M',train,y,validation,availability)
    assert len(meta['qualified_members'])==8 and meta['required_per_row']==4
    assert composite.notna().tolist()==[True,True,False]
    small=train[['M1','M2']]
    composite2,meta2=family_composite('M',small,y,pd.DataFrame({'M1':[1.,1.],'M2':[np.nan,1.]}),
                                      feature_availability(small,POLICY))
    assert meta2['required_per_row']==2 and composite2.isna().tolist()==[True,False]


def test_fewer_than_two_qualified_members_gives_nan_composite():
    x,y=synthetic_fold()
    train=x[['M1','M4']].iloc[:30]
    composite,meta=family_composite('M',train,y.iloc[:30],x[['M1','M4']].iloc[30:],feature_availability(train,POLICY))
    assert meta['qualified_members']==['M1'] and composite.isna().all()


def test_zero_direction_member_is_excluded():
    y=pd.Series([1.,2.,3.,4.,5.,6.,7.,8.])
    train=pd.DataFrame({'V1':y,'V2':y*2,'V3':[1.,2.,3.,4.,4.,3.,2.,1.]})
    composite,meta=family_composite('V',train,y,train,feature_availability(train,POLICY))
    assert meta['excluded_zero_direction']==['V3'] and meta['qualified_members']==['V1','V2']


def test_seven_families_lagged_fields_excluded_and_d_family_constructed():
    x,y=synthetic_fold()
    train,validation=x.iloc[:30],x.iloc[30:]
    composites,meta=family_composites(train,y.iloc[:30],validation,feature_availability(train,POLICY))
    assert tuple(composites.columns)==COMPOSITE_FAMILIES==('M','E','I','P','V','S','D')
    members=[m for info in meta.values() for m in info['qualified_members']+info['excluded_unavailable']]
    assert not set(LAGGED_OUTCOME_COLUMNS)&set(members)
    assert meta['D']['qualified_members']==['D1','D2','D3'] and composites['D'].notna().any()
    assert composites['E'].isna().all()


def test_composite_rejects_target_in_x_or_misaligned_target():
    x,y=synthetic_fold()
    availability=feature_availability(x[['M1']],POLICY)
    with pytest.raises(ValueError):
        family_composite('M',x[['M1']].assign(forward_returns=1.),y,x[['M1']],availability)
    with pytest.raises(ValueError):
        family_composite('M',x[['M1']].iloc[:10],y.iloc[:20],x[['M1']],availability)


# Multiplicity

SINGLE=[f'F{i}' for i in range(97)]
COMPOSITES=list('MEIPVSD')


def test_family_specific_bh_and_global_104_sensitivity():
    single={'F0':.0001,'F1':.0004,'F2':.03}
    composite={'M':.004,'E':.01,'V':.2,'D':None}
    result=protocol_multiplicity(single,SINGLE,composite,COMPOSITES)
    assert len(result['single_feature_bh'].adjusted_pvalues)==97
    assert len(result['family_composite_bh'].adjusted_pvalues)==7
    assert len(result['global_bh_sensitivity'].adjusted_pvalues)==104
    expected_single=adjust_fdr(family_pvalues(single,SINGLE))
    np.testing.assert_allclose(result['single_feature_bh'].adjusted_pvalues,expected_single.adjusted_pvalues)
    assert result['family_composite_bh'].rejected[:2].tolist()==[True,True]
    combined=np.r_[family_pvalues(single,SINGLE),family_pvalues(composite,COMPOSITES)]
    np.testing.assert_allclose(result['global_bh_sensitivity'].adjusted_pvalues,adjust_fdr(combined).adjusted_pvalues)
    for key in ['single_feature_by_sensitivity','family_composite_by_sensitivity','global_by_sensitivity']:
        assert result[key].method=='by'
    assert np.all(result['global_by_sensitivity'].adjusted_pvalues>=result['global_bh_sensitivity'].adjusted_pvalues)


def test_not_testable_hypotheses_keep_fixed_denominators():
    p=family_pvalues({'F0':.001,'F1':float('nan'),'F2':None},SINGLE)
    assert len(p)==97 and p[0]==.001 and (p[1:]==1).all()
    assert family_pvalues({},COMPOSITES).tolist()==[1.]*7
    with pytest.raises(ValueError):
        family_pvalues({'NOT_REGISTERED':.01},SINGLE)
