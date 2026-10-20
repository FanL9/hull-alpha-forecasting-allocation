import numpy as np
import pytest
import torch
from src.models.deep_alpha import GRU, TCN, Stopper, deterministic, train, ridge_solutions, ridge_predict
from src.research.phase5_tasks import encode

@pytest.mark.parametrize('cls,count',[(GRU,633),(TCN,1137)])
def test_actual_parameter_count(cls,count):
    assert sum(p.numel() for p in cls().parameters() if p.requires_grad)==count<10000

def test_ensemble_budget():
    assert 5*(sum(p.numel() for p in GRU().parameters())+sum(p.numel() for p in TCN().parameters()))==8850<10000

def test_tcn_causal_blocks_no_future_positions():
    deterministic(11); net=TCN(); x=torch.randn(2,63,16)
    a=net.blocks(x.transpose(1,2)); x[:,31:]+=100
    b=net.blocks(x.transpose(1,2))
    assert torch.equal(a[:,:,:31],b[:,:,:31])

@pytest.mark.parametrize('name',['GRU','TCN'])
def test_same_seed_stage_byte_determinism_and_fresh_refit(name):
    rng=np.random.default_rng(8); x=rng.normal(size=(70,63,16)); y=rng.normal(size=70)
    a,_,_=train(name,11,x,y,2)
    b,_,_=train(name,11,x,y,2)
    assert encode({k:v.detach().numpy() for k,v in a.state_dict().items()})==encode({k:v.detach().numpy() for k,v in b.state_dict().items()})

def test_early_stopping_minimum_patience_tie():
    s=Stopper()
    for epoch in range(1,20):
        assert not s.update(epoch,1.)
    assert s.update(20,1.)
    assert s.epoch==10

def test_early_stopping_min_delta_strict_and_maximum():
    s=Stopper(minimum=1,maximum=4)
    assert not s.update(1,1.)
    assert not s.update(2,.99995)
    assert s.epoch==1
    assert not s.update(3,.9998)
    assert s.epoch==3
    assert s.update(4,.9998)

def test_early_stopping_nonfinite():
    with pytest.raises(FloatingPointError):
        Stopper().update(1,np.nan)

def test_ridge_mse_penalty_and_unpenalized_intercept():
    rng=np.random.default_rng(1); x=rng.normal(size=(200,3)); y=4+x@np.array([1.,-2.,.4])
    fits=ridge_solutions(x,y,[.1,1,10])
    z=x-x.mean(axis=0)
    for alpha,state in fits.items():
        expected=np.linalg.solve(z.T@z+len(x)*alpha*np.eye(3),z.T@(y-y.mean()))
        np.testing.assert_allclose(state['coef'],expected,atol=1e-14)
        assert abs(ridge_predict(state,x).mean()-y.mean())<1e-14
