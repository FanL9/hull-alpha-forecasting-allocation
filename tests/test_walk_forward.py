"""Boundary and training-only availability tests; never read real X or Y."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.download import sha256
from src.features.availability import (available_lagged_outcomes, feature_availability,
                                       research_candidate_columns)
from src.features.feature_groups import column_role
from src.utils.paths import project_root
from src.validation.walk_forward import (ValidationProtocol, holdout_training_ids,
                                         load_protocol, load_research_prefix, make_folds)


def test_frozen_protocol_boundaries_and_expansion():
    protocol=load_protocol(project_root()/'configs/validation.yaml')
    folds=make_folds(np.arange(8868),protocol)
    assert len(folds)==11 and protocol.initial_train_size==5087
    assert protocol.internal_holdout_start==7860 and protocol.internal_holdout_end==8867
    assert len(range(7860,8868))==1008
    assert (folds[0].train_ids[0],folds[0].train_ids[-1])==(0,5086)
    assert (folds[0].validation_ids[0],folds[0].validation_ids[-1])==(5088,5339)
    assert (folds[-1].validation_ids[0],folds[-1].validation_ids[-1])==(7608,7859)
    validations=[]
    for i,fold in enumerate(folds):
        assert set(fold.train_ids).isdisjoint(fold.validation_ids)
        assert set(fold.purged_ids).isdisjoint(fold.train_ids)
        assert max(fold.train_ids)+1 < min(fold.validation_ids)  # last training outcome ends before boundary
        assert len(fold.validation_ids)==252 and len(fold.purged_ids)==1 and not fold.embargo_ids
        assert max(fold.train_ids+fold.validation_ids)<7860<8868
        if i:
            assert set(folds[i-1].train_ids) < set(fold.train_ids)
        validations.extend(fold.validation_ids)
    assert validations==list(range(5088,7860))
    assert holdout_training_ids(protocol)==tuple(range(7859))


@pytest.mark.parametrize('kind',['reverse','duplicate','float','full_train'])
def test_bad_identifiers_are_rejected_not_reordered(kind):
    protocol=load_protocol(project_root()/'configs/validation.yaml')
    dates=np.arange(8868)
    if kind=='reverse': dates=dates[::-1]
    elif kind=='duplicate': dates[3]=2
    elif kind=='float': dates=dates.astype(float)
    else: dates=np.arange(9048)
    with pytest.raises(ValueError,match='ordered'):
        make_folds(dates,protocol)


def toy(embargo=2,purge=1):
    return ValidationProtocol.from_mapping({'method':'expanding_window','shuffle':False,
        'development_start':0,'development_end':39,'research_start':0,'research_end':27,
        'initial_train_size':10,'validation_window':4,'step_size':6,'purge_size':purge,'embargo_size':embargo,
        'fold_count':3,'internal_holdout_start':28,'internal_holdout_end':39,
        'final_competition_start':40,'final_competition_end':45})


def test_nonzero_embargo_has_real_effect_on_later_train_and_validation():
    protocol=toy()
    folds=make_folds(np.arange(40),protocol)
    assert folds[0].validation_ids==(11,12,13,14)
    assert folds[0].embargo_ids==(15,16)
    for later in folds[1:]:
        assert set((15,16)).isdisjoint(later.train_ids+later.validation_ids)
    assert set((15,16)).isdisjoint(holdout_training_ids(protocol))
    assert folds[1].purged_ids==(16,)


def test_nonzero_embargo_cannot_be_silently_ignored():
    mapping=yaml.safe_load((project_root()/'configs/validation.yaml').read_text())
    mapping['embargo_size']=2
    with pytest.raises(ValueError,match='embargo'):
        ValidationProtocol.from_mapping(mapping)


def test_research_loader_excludes_both_holdouts(tmp_path,monkeypatch):
    protocol=load_protocol(project_root()/'configs/validation.yaml')
    path=tmp_path/'train.csv'
    frame=pd.DataFrame({'date_id':np.arange(9048),'M1':[1.]*7860+['LOCKED_VALUE']*(9048-7860),
                        'market_forward_excess_returns':[.001]*7860+['LOCKED_TARGET']*(9048-7860)})
    frame.to_csv(path,index=False)
    plan={'competition_simulation_locked':True,'source_train_sha256':sha256(path),'train_columns':list(frame)}
    original=pd.read_csv
    def guarded(path,**kwargs):
        assert kwargs['nrows']==7860
        return original(path,**kwargs)
    monkeypatch.setattr(pd,'read_csv',guarded)
    prefix=load_research_prefix(path,plan,protocol)
    assert prefix.shape==(7860,3) and prefix.date_id.max()==7859
    assert pd.api.types.is_numeric_dtype(prefix.M1)


def policy():
    return yaml.safe_load((project_root()/'configs/research.yaml').read_text())['protocol']['feature_availability']


def test_leading_missing_feature_enters_only_after_training_coverage():
    p=policy()
    X=pd.DataFrame({'E7':[np.nan]*700+list(np.linspace(0,1,700)),'M1':np.arange(1400)})
    early=feature_availability(X.iloc[:800],p).set_index('feature')
    later=feature_availability(X.iloc[:1300],p).set_index('feature')
    assert not early.loc['E7','eligible']
    assert later.loc['E7','eligible']
    assert early.loc['M1','eligible']
    changed=X.copy()
    changed.loc[800:,'E7']=999.
    pd.testing.assert_frame_equal(feature_availability(changed.iloc[:800],p).reset_index(drop=True),
                                  early.reset_index().reset_index(drop=True))


def test_stale_missing_and_constant_features_are_unavailable():
    X=pd.DataFrame({'M1':np.r_[np.arange(600),np.full(300,np.nan)],'D1':np.ones(900)})
    result=feature_availability(X,policy()).set_index('feature')
    assert not result.eligible.any()
    assert 'excess_recent_missingness' in result.loc['M1','reasons']
    assert 'insufficient_variation' in result.loc['D1','reasons']


def test_availability_never_accepts_current_outcome():
    with pytest.raises(ValueError,match='candidate X only'):
        feature_availability(pd.DataFrame({'M1':[1.],'forward_returns':[.01]}),policy())


def test_lagged_outcomes_are_available_candidates_with_correct_timing():
    history=pd.DataFrame({'date_id':[0,1,2,3],'M1':[1.,2.,3.,4.],
        'forward_returns':[.01,.02,.03,.04],'risk_free_rate':[.001]*4,
        'market_forward_excess_returns':[.009,.019,.029,.039]})
    lagged=available_lagged_outcomes(history)
    assert lagged.lagged_forward_returns.iloc[1]==history.forward_returns.iloc[0]
    assert lagged.iloc[0].isna().all()
    altered=history.copy()
    altered.loc[2:,'forward_returns']=99.
    pd.testing.assert_frame_equal(available_lagged_outcomes(altered).iloc[:3],lagged.iloc[:3])
    assert research_candidate_columns([*history,*lagged])==['M1',*lagged]
    assert 'competition-available' in column_role('lagged_forward_returns')
    with pytest.raises(ValueError,match='consecutive'):
        available_lagged_outcomes(history.iloc[[0,2,3]])
