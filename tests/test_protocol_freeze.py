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
    assert config['primary_hypothesis_family']['fdr_alpha']==.05
    assert config['primary_hypothesis_family']['size']==97
    assert config['inference']['newey_west_lags']==5


def test_current_lock_is_v2_and_v1_lock_is_archived():
    root=project_root()
    current=verify_protocol_lock(root)
    assert current['protocol_version']=='phase2.5-v2' and current['defined_before_real_xy_research'] is True
    assert current['supersedes']=={'protocol_version':'phase2.5-v1','archived_lock':'configs/protocol_lock_phase2.5-v1.json'}
    archived=json.loads((root/'configs/protocol_lock_phase2.5-v1.json').read_text())
    assert archived['protocol_version']=='phase2.5-v1'
    assert current['fold_metadata']==archived['fold_metadata']
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
