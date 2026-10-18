"""Known p-value examples, without financial observations."""
import numpy as np
import pytest

from src.statistics.multiple_testing import adjust_fdr


def test_bh_known_example():
    result=adjust_fdr([.01,.04,.03,.002,.5],alpha=.05)
    np.testing.assert_allclose(result.adjusted_pvalues,[.025,.05,.05,.01,.5])
    assert result.rejected.tolist()==[True,True,True,True,False]


def test_bh_unsorted_ties_and_boundaries():
    p=np.array([1.,.02,0.,.02,.8])
    result=adjust_fdr(p)
    assert result.adjusted_pvalues[0]==1 and result.adjusted_pvalues[2]==0
    assert result.adjusted_pvalues[1]==result.adjusted_pvalues[3]
    np.testing.assert_allclose(p,[1.,.02,0.,.02,.8])


def test_by_harmonic_factor_is_more_conservative():
    p=[.01,.04,.03,.002,.5]
    bh,by=adjust_fdr(p),adjust_fdr(p,method='by')
    harmonic=sum(1/i for i in range(1,6))
    np.testing.assert_allclose(by.adjusted_pvalues,np.minimum(1,bh.adjusted_pvalues*harmonic))
    assert np.all(by.adjusted_pvalues>=bh.adjusted_pvalues)


def test_prespecified_unavailable_hypotheses_stay_in_denominator():
    result=adjust_fdr([.001]+[1.]*96)
    assert result.adjusted_pvalues[0]==pytest.approx(.097)
    assert not result.rejected.any()


@pytest.mark.parametrize('bad',[[np.nan],[np.inf],[-.1],[1.1],[[.1,.2]]])
def test_invalid_pvalues_rejected(bad):
    with pytest.raises(ValueError): adjust_fdr(bad)


def test_empty_family_and_invalid_parameters():
    assert len(adjust_fdr([]).adjusted_pvalues)==0
    with pytest.raises(ValueError): adjust_fdr([.1],alpha=1)
    with pytest.raises(ValueError): adjust_fdr([.1],method='unknown')
