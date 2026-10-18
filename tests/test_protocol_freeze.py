"""Freeze and tamper checks without official data or research execution."""
import json
import shutil

import pytest
import yaml

from src.utils.paths import project_root
from src.validation.protocol_freeze import NORMATIVE_FILES, freeze_protocol, verify_protocol_lock


def copied_protocol(tmp_path):
    for relative in NORMATIVE_FILES:
        destination=tmp_path/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project_root()/relative,destination)
    return tmp_path


def test_protocol_freeze_roundtrip_without_market_data(tmp_path):
    root=copied_protocol(tmp_path)
    first=freeze_protocol(root)
    assert verify_protocol_lock(root)==freeze_protocol(root)==first
    assert first['competition_simulation']=={'start':8868,'end':9047,'rows':180,'locked':True}
    assert first['internal_holdout']['rows']==1008
    assert first['execution']['real_xy_tests']==first['execution']['models_fitted']==0
    assert not (root/'data').exists()


def test_protocol_rejects_silent_config_change(tmp_path):
    root=copied_protocol(tmp_path)
    freeze_protocol(root)
    path=root/'configs/research.yaml'
    path.write_text(path.read_text()+'\n# silent edit\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Frozen protocol changed'):
        verify_protocol_lock(root)


def test_protocol_rejects_conflicting_duplicate_settings(tmp_path):
    root=copied_protocol(tmp_path)
    path=root/'configs/base.yaml'
    config=yaml.safe_load(path.read_text())
    config['validation']['purge_size']=0
    path.write_text(yaml.safe_dump(config),encoding='utf-8')
    with pytest.raises(ValueError,match='Duplicated'):
        freeze_protocol(root)


def test_forecasting_and_portfolio_baselines_are_fixed():
    config=yaml.safe_load((project_root()/'configs/research.yaml').read_text())['protocol']
    assert config['baselines']=={'forecast':'historical_training_mean','diagnostic_forecast':'zero',
                                'portfolio_allocation':1.,'sanity_allocation':0.}
    assert config['multiple_testing']['fdr_alpha']==.05
    assert config['multiple_testing']['primary_family_size']==97
    assert config['inference']['newey_west_lags']==5
