"""Generic uncertainty tests on constructed mathematical sequences only."""
import math

import numpy as np
import pytest

from src.statistics.robust_inference import (correlation_hac, moving_block_correlation_interval,
                                            moving_block_indices, newey_west_mean_se)


def test_newey_west_known_lag_zero_and_one():
    x=np.array([1.,2.,3.,4.])
    residual=x-x.mean()
    expected0=np.sqrt((residual@residual)/16)
    expected1=np.sqrt(((residual@residual)+(residual[1:]@residual[:-1]))/16)
    assert newey_west_mean_se(x,0)==pytest.approx(expected0)
    assert newey_west_mean_se(x,1)==pytest.approx(expected1)


def test_gaps_are_not_compressed_for_hac():
    assert newey_west_mean_se([1.,np.nan,2.,3.,4.],1)!=pytest.approx(newey_west_mean_se([1.,2.,3.,4.],1))


def test_correlation_and_rank_ties_match_numpy_and_average_ranks():
    x=np.array([1.,2.,2.,4.,5.,6.])
    y=np.array([2.,1.,4.,3.,8.,5.])
    result=correlation_hac(x,y,lags=1)
    assert result.correlation==pytest.approx(np.corrcoef(x,y)[0,1])
    assert result.standard_error>0 and 0<=result.pvalue<=1 and result.pairs==6
    ranks=np.array([1.,2.5,2.5,4.,5.,6.])
    yranks=np.array([2.,1.,4.,3.,6.,5.])
    assert correlation_hac(x,y,method='spearman').correlation==pytest.approx(np.corrcoef(ranks,yranks)[0,1])


def test_undefined_constant_or_insufficient_data_are_not_significant():
    result=correlation_hac([1.]*5,np.arange(5))
    assert math.isnan(result.correlation) and math.isnan(result.pvalue)
    assert math.isnan(correlation_hac([1.,np.nan],[2.,3.]).pvalue)


def test_bootstrap_samples_joint_contiguous_blocks_and_is_reproducible():
    first=moving_block_indices(23,5,np.random.default_rng(42))
    second=moving_block_indices(23,5,np.random.default_rng(42))
    np.testing.assert_array_equal(first,second)
    assert len(first)==23 and first.min()>=0 and first.max()<23
    for start in range(0,23,5): assert np.all(np.diff(first[start:start+5])==1)
    x=np.arange(40,dtype=float)
    y=np.sin(x/4)+x*.03
    a=moving_block_correlation_interval(x,y,block_length=5,repetitions=100,seed=4)
    b=moving_block_correlation_interval(x,y,block_length=5,repetitions=100,seed=4)
    assert a==b and -1<=a['lower']<=a['upper']<=1 and a['valid_repetitions']==100


def test_invalid_bootstrap_and_hac_inputs_rejected():
    with pytest.raises(ValueError): moving_block_indices(5,6,np.random.default_rng(1))
    with pytest.raises(ValueError): newey_west_mean_se([1.,2.,np.inf])
    with pytest.raises(ValueError): correlation_hac([1.,2.],[1.])
    with pytest.raises(ValueError): moving_block_correlation_interval([1.,2.],[1.,2.],repetitions=0)
