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
    assert config['primary_hypothesis_family']['fdr_alpha']==.05
    assert config['primary_hypothesis_family']['size']==97
    assert config['inference']['newey_west_lags']==5


def test_current_lock_is_v2_and_v1_locks_are_archived():
    root=project_root()
    current=verify_protocol_lock(root)
    assert current['lock_version']==2 and current['protocol_version']=='phase2.5-v2'
    assert current['defined_before_real_xy_research'] is True
    archives=['configs/protocol_lock_phase2.5-v1.json','configs/protocol_lock_phase2.5-v1_lockformat2.json']
    assert current['supersedes']=={'protocol_version':'phase2.5-v1','archived_locks':archives}
    for relative in archives:
        archived=json.loads((root/relative).read_text())
        assert archived['protocol_version']=='phase2.5-v1' and relative in current['sha256']
        assert archived['fold_metadata']==current['fold_metadata']
    assert current['primary_metric']=='spearman_rank_ic' and current['secondary_metric']=='winsorized_pearson_ic'
    assert current['hypothesis_families']=={'single_features':97,'family_composites':7,'global_sensitivity':104}


def test_v2_rules_are_machine_readable_and_frozen_rules_unchanged():
    config=yaml.safe_load((project_root()/'configs/research.yaml').read_text())['protocol']
    validation=yaml.safe_load((project_root()/'configs/validation.yaml').read_text())
    assert config['version']==validation['protocol_version']=='phase2.5-v2'
    assert config['primary_metric']['ranking_scope']=='validation_block'
    assert config['primary_metric']['rerank_after_concatenation'] is False
    secondary=config['secondary_metric']
    assert (secondary['x_winsor_lower'],secondary['x_winsor_upper'])==(.01,.99)
    assert secondary['thresholds_fit_on']=='training_window'
    assert secondary['winsorize_target'] is False and secondary['confirmatory'] is False
    assert config['inference']['pvalue_source']=='newey_west_hac'
    assert config['inference']['bootstrap_role']=='confidence_interval'
    family=config['secondary_hypothesis_family']
    assert family['families']==list('MEIPVSD') and family['size']==7 and family['lagged_fields_included'] is False
    assert (family['normalization'],family['direction'],family['weighting'])==(
        'training_empirical_rank_percentile','training_spearman_sign','equal')
    assert (family['min_available_members'],family['min_available_fraction'])==(2,.5)
    assert config['global_sensitivity']=={'combined_hypotheses':104,'method':'benjamini_hochberg',
                                          'alpha':.05,'replaces_family_results':False}
    assert config['by_sensitivity']['enabled'] is True
    assert config['feature_availability']=={'min_nonmissing_observations':252,
        'min_historical_coverage_since_first_observation':.8,'recent_window':504,
        'max_recent_missing_ratio':.25,'min_distinct_values':2}
    assert (validation['initial_train_size'],validation['validation_window'],validation['step_size'],
            validation['fold_count'],validation['purge_size'],validation['embargo_size'])==(5087,252,252,11,1,0)
    assert (validation['internal_holdout_start'],validation['internal_holdout_end'])==(7860,8867)
    assert (validation['final_competition_start'],validation['final_competition_end'])==(8868,9047)


def test_freeze_rejects_stale_protocol_version(tmp_path):
    root=copied_protocol(tmp_path)
    path=root/'configs/validation.yaml'
    path.write_text(path.read_text().replace('phase2.5-v2','phase2.5-v1'),encoding='utf-8')
    with pytest.raises(ValueError,match='version'):
        freeze_protocol(root)


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
