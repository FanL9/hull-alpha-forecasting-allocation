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
    assert first['lock_version']==2
    assert all(not p.startswith(('docs/','reports/','notebooks/')) for p in first['sha256'])
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


def test_explanatory_documents_can_change_without_invalidating_lock(tmp_path):
    root=copied_protocol(tmp_path)
    frozen=freeze_protocol(root)
    for relative in ['README.md','docs/research_protocol.md','docs/methodology.md','docs/data_dictionary.md',
                     'docs/project_flow.md','docs/official_metric_notes.md','reports/data_audit.md']:
        path=root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('Editable explanatory text.\n',encoding='utf-8')
    assert verify_protocol_lock(root)==frozen
    policy=json.loads((root/'configs/split_policy.json').read_text(encoding='utf-8'))
    assert not any(p.startswith(('docs/','reports/')) for p in policy['preserved_phase1_sha256'])


def test_statistical_or_metric_code_edits_still_invalidate_lock(tmp_path):
    root=copied_protocol(tmp_path)
    freeze_protocol(root)
    path=root/'src/evaluation/kaggle_metric.py'
    path.write_text(path.read_text(encoding='utf-8')+'\n# executable change\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Frozen protocol changed'):
        verify_protocol_lock(root)


def test_frozen_metadata_cannot_disagree_with_config(tmp_path):
    root=copied_protocol(tmp_path)
    freeze_protocol(root)
    path=root/'configs/protocol_lock.json'
    lock=json.loads(path.read_text(encoding='utf-8'))
    lock['fold_metadata'][0]['validation_start']+=1
    path.write_text(json.dumps(lock),encoding='utf-8')
    with pytest.raises(ValueError,match='metadata differs'):
        verify_protocol_lock(root)


def test_raw_data_hashes_are_verified_without_parsing_market_values(tmp_path):
    import hashlib
    root=copied_protocol(tmp_path)
    data=root/'data/raw/train.csv'
    data.parent.mkdir(parents=True,exist_ok=True)
    data.write_bytes(b'artificial raw bytes; not a research dataset\n')
    digest=hashlib.sha256(data.read_bytes()).hexdigest()
    path=root/'configs/split_policy.json'
    policy=json.loads(path.read_text(encoding='utf-8'))
    policy['source_train_sha256']=digest
    policy['data_sha256']={'data/raw/train.csv':digest}
    path.write_text(json.dumps(policy),encoding='utf-8')
    freeze_protocol(root)
    verify_protocol_lock(root,verify_data=True)
    data.write_bytes(b'artificial tampered bytes\n')
    with pytest.raises(ValueError,match='raw data differs'):
        verify_protocol_lock(root,verify_data=True)


def test_text_newline_conversion_does_not_change_executable_hash(tmp_path):
    root=copied_protocol(tmp_path)
    freeze_protocol(root)
    path=root/'configs/research.yaml'
    path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
    verify_protocol_lock(root)


def test_v1_upgrade_is_explicit_and_does_not_change_scientific_rules(tmp_path):
    root=copied_protocol(tmp_path)
    current=freeze_protocol(root)
    path=root/'configs/protocol_lock.json'
    previous=dict(current)
    previous.pop('lock_version')
    path.write_text(json.dumps(previous),encoding='utf-8')
    with pytest.raises(ValueError,match='explicit lock-format upgrade'):
        freeze_protocol(root)
    migrated=freeze_protocol(root,upgrade_lock=True)
    assert migrated['lock_version']==2 and migrated['protocol_version']==current['protocol_version']
    assert migrated['fold_metadata']==current['fold_metadata']
