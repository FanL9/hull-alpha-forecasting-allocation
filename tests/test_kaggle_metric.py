"""Synthetic numerical parity with the retrieved official scorer, never a backtest."""
import json
from math import prod, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.kaggle_metric import (ParticipantVisibleError, evaluate_allocations,
                                         metric_components, score)

REFERENCE = json.loads((Path(__file__).parent/'fixtures/kaggle_metric_synthetic_cases.json').read_text())


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c:c['name'])
def test_against_official_synthetic_reference(case):
    solution = pd.DataFrame({'forward_returns':case['forward_returns'],'risk_free_rate':case['risk_free_rate']},
                            index=case['solution_index'])
    submission = pd.DataFrame({'prediction':case['allocation']},index=case['submission_index'])
    if 'expected_error' in case:
        with pytest.raises(ParticipantVisibleError):
            score(solution,submission,'date_id')
    else:
        assert score(solution,submission,'date_id') == pytest.approx(case['expected_score'],rel=2e-13,abs=2e-13)


def data():
    return np.array([.012,-.004,.018,.003,-.008,.014,.005,-.002]), np.full(8,.0001)


def test_one_hundred_percent_is_market_baseline():
    r,rf=data()
    result=evaluate_allocations(r,rf,np.ones(len(r)))
    assert result.volatility_ratio == pytest.approx(1)
    assert result.volatility_penalty == result.return_penalty == 1
    assert result.strategy_geometric_excess_mean == result.market_geometric_excess_mean


def test_zero_exposure_has_cash_return_and_zero_excess_score_when_cash_varies():
    r,rf=data()
    rf=rf*np.linspace(.5,1.5,len(r))
    result=evaluate_allocations(r,rf,np.zeros(len(r)))
    assert result.strategy_std == pytest.approx(rf.std(ddof=1))
    assert result.strategy_geometric_excess_mean == 0 and result.score == 0


def test_zero_exposure_constant_cash_is_undefined_in_official_metric():
    r,rf=data()
    with pytest.raises(ParticipantVisibleError,match='strategy std is zero'):
        evaluate_allocations(r,rf,np.zeros(len(r)))


def test_volatility_threshold_is_soft_and_uses_total_return_std():
    r,rf=data()
    at=evaluate_allocations(r,rf,np.full(len(r),1.2))
    above=evaluate_allocations(r,rf,np.full(len(r),1.5))
    assert at.volatility_penalty == pytest.approx(1)
    assert above.volatility_ratio == pytest.approx(1.5)
    assert above.volatility_penalty == pytest.approx(1.3)
    assert above.score == pytest.approx(above.sharpe/(above.volatility_penalty*above.return_penalty))


def test_underperformance_penalty_units_and_square():
    r,rf=data()
    result=evaluate_allocations(r,rf,np.full(len(r),.5))
    gap=(result.market_geometric_excess_mean-result.strategy_geometric_excess_mean)*100*252
    assert gap > 0
    assert result.annualized_return_gap_percentage_points == pytest.approx(gap)
    assert result.return_penalty == pytest.approx(1+gap**2/100)


def test_known_example_without_using_forecasting_target():
    r,rf,a=[.01,-.02,.03],[.0001]*3,[1.,0.,2.]
    portfolio=[cash*(1-w)+w*market for market,cash,w in zip(r,rf,a)]
    excess=[p-c for p,c in zip(portfolio,rf)]
    g=prod(1+e for e in excess)**(1/3)-1
    std=sqrt(sum((p-sum(portfolio)/3)**2 for p in portfolio)/2)
    result=evaluate_allocations(r,rf,a)
    assert result.strategy_geometric_excess_mean == pytest.approx(g)
    assert result.sharpe == pytest.approx(g/std*sqrt(252))


def test_does_not_mutate_inputs_or_use_adjusted_target():
    r,rf=data()
    solution=pd.DataFrame({'forward_returns':r,'risk_free_rate':rf,'market_forward_excess_returns':999.})
    submission=pd.DataFrame({'prediction':np.ones(len(r))})
    original=solution.copy()
    first=score(solution,submission,'unused')
    solution['market_forward_excess_returns']=-999.
    assert score(solution,submission,'unused') == first
    pd.testing.assert_frame_equal(solution.drop(columns='market_forward_excess_returns'),original.drop(columns='market_forward_excess_returns'))
    assert 'strategy_returns' not in solution


@pytest.mark.parametrize('allocation',[[-.01]*3,[2.01]*3])
def test_allocation_bounds_reject_without_clipping(allocation):
    with pytest.raises(ParticipantVisibleError):
        evaluate_allocations([.01,-.02,.03],[.0001]*3,allocation)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_strict_adapter_rejects_nonfinite_values(bad):
    with pytest.raises(ValueError,match='finite'):
        evaluate_allocations([.01,bad,.03],[.0001]*3,[1.]*3)


def test_strict_adapter_rejects_misaligned_lengths():
    with pytest.raises(ValueError,match='equal-length'):
        evaluate_allocations([.01,.02],[.0001],[1.,1.])
