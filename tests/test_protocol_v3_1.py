from copy import deepcopy
import pytest
from src.validation import protocol_v3,protocol_v3_1 as v31
from src.utils.paths import project_root

ROOT=project_root()

def test_only_authorized_regime_and_calibration_differ():
    c=v31.load_config(ROOT);old=protocol_v3.load_config(ROOT)
    for k in old:
        if k not in {'protocol_version','freeze_tag','parent_protocol','regime'}:assert c[k]==old[k]
    assert c['regime']['training_y_used'] is False
    assert c['regime']['validation_y_used'] is False
    assert c['regime']['member_directions']=='all_positive_without_target'
    assert len(c['synthetic_calibration']['noise_seeds'])==20

def test_registry_and_selector_remain_identical():
    c=v31.load_config(ROOT);old=protocol_v3.load_config(ROOT)
    assert protocol_v3.candidate_ids(c)==protocol_v3.candidate_ids(old)
    assert len(protocol_v3.raw_names(c))==97
    assert len(protocol_v3.candidate_ids(c))==499
    assert c['selection']==old['selection']
    assert c['selection']['top_k']==16
    assert c['selection']['redundancy']['absolute_threshold']==.9

@pytest.mark.parametrize('key',['transformations','selection','deep','models','primary_selection','holdout','final_simulation'])
def test_unamended_settings_are_parent_identical(key):
    assert v31.load_config(ROOT)[key]==protocol_v3.load_config(ROOT)[key]

def test_prior_freeze_valid_and_no_data_needed(monkeypatch):
    import pandas as pd
    import src.validation.walk_forward as walk
    def fail(*a,**k):pytest.fail('Protocol checks may not load data')
    monkeypatch.setattr(pd,'read_csv',fail);monkeypatch.setattr(walk,'load_research_prefix',fail)
    assert protocol_v3.verify_tag(ROOT)=='c3e35d62cfaa2e8ccff231511ebe803b7656e34b'
    assert v31.load_config(ROOT)['regime']['source'].startswith('unsigned_training')

def test_calibration_is_prospective_finite_and_not_a_feature_gate():
    cal=v31.load_config(ROOT)['synthetic_calibration']
    assert tuple(cal['noise_seeds'])==v31.NOISE_SEEDS
    assert len(set(cal['noise_seeds']))==20
    assert cal['planted_seeds']==[11,22,33]
    assert cal['no_parameter_retuning_from_calibration']
    assert 'not_market_significance_or_feature_screen' in cal['alarm_role']

def test_no_implicit_execution():
    with pytest.raises(SystemExit):v31.main([])
    with pytest.raises(SystemExit):v31.main(['--run-real-phase4'])
