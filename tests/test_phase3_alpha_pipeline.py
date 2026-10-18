"""Phase 3 synthetic integration, statistical wiring and training-only fitting."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.cli import main
from src.features.availability import LAGGED_OUTCOME_COLUMNS
from src.features.family_composites import family_composite, training_direction, training_percentile
from src.research import alpha_pipeline, phase3
from src.research.alpha_pipeline import build_alpha_tables, load_alpha_design, summarize_hypothesis
from src.research.alpha_results import apply_multiple_testing, write_phase3_outputs
from src.research.synthetic_alpha import make_synthetic_history
from src.statistics.multiple_testing import adjust_fdr
from src.statistics.robust_inference import (apply_winsor_bounds, blockwise_rank_pairs,
                                            blockwise_spearman_hac, correlation_hac, fit_winsor_bounds)
from src.utils.paths import project_root
from src.validation.protocol_freeze import verify_protocol_lock

ROOT = project_root()


@pytest.fixture(scope="module")
def end_to_end():
    history = make_synthetic_history(ROOT)
    return history, phase3.run_synthetic_phase3(ROOT, history)


def test_registered_universe_and_frozen_eleven_folds(end_to_end):
    history, result = end_to_end
    design = load_alpha_design(ROOT)
    assert history.shape == (7860, 98)
    assert len(design.single_features) == 97 and "MOM1" not in design.single_features
    assert tuple(design.single_features[-3:]) == LAGGED_OUTCOME_COLUMNS
    manifest = result.manifest
    assert (manifest['single_hypothesis_count'], manifest['family_hypothesis_count'], manifest['global_hypothesis_count']) == (97, 7, 104)
    folds = manifest['folds']
    assert len(folds) == 11
    assert (folds[0]['train_end'], folds[0]['purged_ids'], folds[0]['validation_start'], folds[0]['validation_end']) == (5086, [5087], 5088, 5339)
    assert (folds[-1]['train_end'], folds[-1]['purged_ids'], folds[-1]['validation_start'], folds[-1]['validation_end']) == (7606, [7607], 7608, 7859)
    assert len(result.tables.fold_level_ic) == 104 * 11
    assert len(result.tables.feature_availability) == 97 * 11


def test_positive_and_negative_alpha_two_sided_and_noise(end_to_end):
    _, result = end_to_end
    table = result.tables.single_features.set_index('feature')
    assert table.loc['M1', 'oos_rank_ic'] > .4
    assert table.loc['M2', 'oos_rank_ic'] < -.4
    assert table.loc['M1', 'nw_stat'] > 0 and table.loc['M2', 'nw_stat'] < 0
    assert table.loc['M1', 'raw_p_value'] < .001 and table.loc['M2', 'raw_p_value'] < .001
    assert table.loc['M1', 'formal_conclusion'] == table.loc['M2', 'formal_conclusion'] == 'PASS'
    # A fixed independent noise stream should have a small descriptive IC;
    # no assertion that random noise must fail BH.
    assert abs(table.loc['lagged_forward_returns', 'oos_rank_ic']) < .1


def test_fixed_fdr_denominators_and_global_sensitivity(end_to_end):
    _, result = end_to_end
    single, composite = result.tables.single_features, result.tables.family_composites
    assert len(single) == 97 and len(composite) == 7
    assert tuple(composite['feature']) == ('M', 'E', 'I', 'P', 'V', 'S', 'D')
    unavailable = single[single['status'].eq('not_testable')]
    assert len(unavailable) > 0 and unavailable['raw_p_value'].eq(1).all()
    assert unavailable['formal_conclusion'].eq('NOT_TESTABLE').all()
    assert set(single['formal_conclusion']) <= {'PASS', 'FAIL', 'NOT_TESTABLE'}
    assert result.multiple_testing_summary['hypotheses'].tolist() == [97, 7, 104, 97, 7, 104]
    np.testing.assert_allclose(single['bh_q_value'], adjust_fdr(single['raw_p_value']).adjusted_pvalues)
    np.testing.assert_allclose(composite['bh_q_value'], adjust_fdr(composite['raw_p_value']).adjusted_pvalues)
    raw = np.r_[single['raw_p_value'], composite['raw_p_value']]
    np.testing.assert_allclose(np.r_[single['global_bh_q_value'], composite['global_bh_q_value']], adjust_fdr(raw).adjusted_pvalues)
    np.testing.assert_allclose(np.r_[single['global_by_q_value'], composite['global_by_q_value']], adjust_fdr(raw, method='by').adjusted_pvalues)


def test_global_or_diagnostics_never_override_family_bh(end_to_end):
    _, result = end_to_end
    tables = deepcopy(result.tables)
    tables.single_features['raw_p_value'] = 1.
    tables.single_features['status'] = 'testable'
    tables.single_features['direction_consistency'] = 0.
    tables.single_features['bootstrap_ci_lower'] = -1.
    tables.single_features['bootstrap_ci_upper'] = 1.
    tables.family_composites['raw_p_value'] = 1.
    tables.family_composites.loc[0, ['raw_p_value', 'status']] = [.004, 'testable']
    apply_multiple_testing(tables, load_alpha_design(ROOT))
    row = tables.family_composites.iloc[0]
    assert row['bh_pass'] and not row['global_bh_pass_sensitivity']
    assert row['formal_conclusion'] == 'PASS'
    # Composites are still tested even with no passing single features.
    assert not tables.single_features['bh_pass'].any()


def test_synthetic_manifest_and_all_output_files(end_to_end, tmp_path):
    history, original = end_to_end
    result = deepcopy(original)
    output = tmp_path / 'synthetic_outputs'
    write_phase3_outputs(result, output)
    expected = {'single_feature_results.csv', 'family_composite_results.csv', 'fold_level_ic.csv',
                'feature_availability.csv', 'multiple_testing_summary.csv', 'phase3_run_manifest.json',
                'fold_transform_audit.json'}
    assert {p.name for p in output.iterdir()} == expected
    manifest = json.loads((output / 'phase3_run_manifest.json').read_text())
    assert manifest['protocol_version'] == 'phase2.5-v2'
    assert manifest['git_commit'] and len(manifest['protocol_lock_hash']) == 64
    assert manifest['run_mode'] == 'synthetic' and not manifest['real_alpha_research_executed']
    assert not manifest['internal_holdout_used'] and not manifest['final_competition_used']
    assert manifest['verified_dataset_identity']['kind'] == 'synthetic'
    assert not manifest['verified_dataset_identity']['raw_data_hashes_verified']
    assert manifest['research_date_id_start'] == 0 and manifest['research_date_id_end'] == 7859
    assert manifest['hac_lag'] == 5 and manifest['bootstrap_config']['bootstrap_repetitions'] == 2000
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    loaded = pd.read_csv(output / 'single_feature_results.csv')
    pd.testing.assert_series_equal(loaded['oos_rank_ic'], result.tables.single_features['oos_rank_ic'], check_names=True)
    with pytest.raises(FileExistsError):
        write_phase3_outputs(result, output)
    pd.testing.assert_frame_equal(history, make_synthetic_history(ROOT))


@pytest.mark.parametrize('change', ['holdout', 'final', 'shuffle', 'missing_date', 'fractional_date', 'unknown_feature'])
def test_engine_rejects_nonresearch_or_bad_inputs_before_statistics(change, monkeypatch):
    history = make_synthetic_history(ROOT)
    if change in {'holdout', 'final'}:
        extra = history.iloc[[-1]].copy()
        extra['date_id'] = 7860 if change == 'holdout' else 8868
        history = pd.concat([history, extra], ignore_index=True)
    elif change == 'shuffle':
        history = history.iloc[::-1]
    elif change == 'missing_date':
        history = history.iloc[:-1]
    elif change == 'fractional_date':
        history['date_id'] = history['date_id'].astype(float)
    else:
        history = history.rename(columns={'M1': 'M999'})
    def forbidden(*a, **k):
        pytest.fail('Bad input reached statistical execution')
    monkeypatch.setattr(alpha_pipeline, 'feature_availability', forbidden)
    with pytest.raises(ValueError):
        build_alpha_tables(history, load_alpha_design(ROOT))


def test_synthetic_entry_never_calls_real_loader_and_requires_marker(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('Synthetic entry attempted the real loader')
    monkeypatch.setattr(phase3, 'load_research_prefix', forbidden)
    monkeypatch.setattr(phase3, '_execute', lambda *a, **k: k)
    frame = make_synthetic_history(ROOT)
    assert phase3.run_synthetic_phase3(ROOT, frame)['synthetic'] is True
    frame.attrs.clear()
    with pytest.raises(ValueError, match='synthetic'):
        phase3.run_synthetic_phase3(ROOT, frame)


def test_real_entry_requires_opt_in_before_any_verification(monkeypatch):
    monkeypatch.setattr(phase3, 'verify_protocol_lock', lambda *a, **k: pytest.fail('No opt-in'))
    with pytest.raises(PermissionError):
        phase3.run_real_phase3(ROOT)


def test_failed_real_verification_never_loads_data(monkeypatch):
    events = []
    def verify(root, *, verify_data):
        assert verify_data is True
        events.append('verify')
        raise ValueError('synthetic verification failure')
    monkeypatch.setattr(phase3, 'verify_protocol_lock', verify)
    monkeypatch.setattr(phase3, 'load_research_prefix', lambda *a, **k: pytest.fail('verification failed'))
    with pytest.raises(ValueError, match='verification failure'):
        phase3.run_real_phase3(ROOT, run_real_phase3=True)
    assert events == ['verify']


def test_real_entry_call_order_with_only_test_doubles(monkeypatch):
    events = []
    frame = make_synthetic_history(ROOT)
    def verify(root, *, verify_data):
        assert verify_data is True
        events.append('verify_data')
        return {'test_double': True}
    def header(path):
        assert events == ['verify_data']
        events.append('header')
        return list(frame.columns)
    def load(path, policy, validation):
        assert events == ['verify_data', 'header']
        assert validation.research_end == 7859 and policy['competition_simulation_locked']
        events.append('safe_loader')
        return frame
    def execute(root, history, lock, **kwargs):
        assert history is frame and lock == {'test_double': True} and not kwargs['synthetic']
        events.append('engine')
    monkeypatch.setattr(phase3, 'verify_protocol_lock', verify)
    monkeypatch.setattr(phase3, 'read_header', header)
    monkeypatch.setattr(phase3, 'load_research_prefix', load)
    monkeypatch.setattr(phase3, '_execute', execute)
    phase3.run_real_phase3(ROOT, run_real_phase3=True)
    assert events == ['verify_data', 'header', 'safe_loader', 'engine']


def test_cli_no_implicit_real_execution_or_mode_mix(monkeypatch):
    monkeypatch.setattr(phase3, 'run_real_phase3', lambda *a, **k: pytest.fail('Implicit real execution'))
    for argv in [['phase3'], ['phase3', '--synthetic', '--run-real-phase3'],
                 ['phase3', '--synthetic', '--config', 'other.yaml'], ['phase2', '--run-real-phase3']]:
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2


def test_rank_ic_uses_blockwise_complete_pairs_not_global_ranks(end_to_end):
    history, result = end_to_end
    raw = history.loc[5088:, 'M1'].to_numpy()
    y = history.loc[5088:, 'market_forward_excess_returns'].to_numpy()
    blocks = np.repeat(np.arange(1, 12), 252)
    expected = blockwise_spearman_hac(raw, y, blocks, lags=5)
    row = result.tables.single_features.set_index('feature').loc['M1']
    assert row['oos_rank_ic'] == pytest.approx(expected.correlation)
    assert row['nw_se'] == pytest.approx(expected.standard_error)
    assert row['raw_p_value'] == pytest.approx(expected.pvalue)
    # Changing levels by block has no effect on within-block ranks.
    drift_x, drift_y = raw + blocks * 1e4, y + blocks * 1e4
    drift = blockwise_spearman_hac(drift_x, drift_y, blocks, lags=5)
    assert drift.correlation == pytest.approx(expected.correlation)
    rx, ry = blockwise_rank_pairs(drift_x, drift_y, blocks)
    assert drift.correlation == pytest.approx(correlation_hac(rx, ry).correlation)
    assert correlation_hac(drift_x, drift_y, method='spearman').correlation > .98


def test_heavy_tail_winsor_bounds_and_target_untouched(end_to_end):
    history, result = end_to_end
    design = load_alpha_design(ROOT)
    bounds = fit_winsor_bounds(history.loc[:5086, ['M1']])
    table = result.tables.feature_availability
    saved = table.loc[(table['fold'] == 1) & (table['feature'] == 'M1')].iloc[0]
    assert saved['winsor_lower'] == pytest.approx(bounds.loc['M1', 'lower'])
    assert saved['winsor_upper'] == pytest.approx(bounds.loc['M1', 'upper'])
    validation = history.loc[5088:5339, ['M1']].copy()
    validation.iloc[0, 0], validation.iloc[-1, 0] = -1e20, 1e20
    clipped = apply_winsor_bounds(validation, bounds)
    assert clipped.iloc[0, 0] == bounds.loc['M1', 'lower']
    assert clipped.iloc[-1, 0] == bounds.loc['M1', 'upper']
    assert design.rules['secondary_metric']['winsorize_target'] is False


def test_full_suite_guard_blocks_existing_real_data():
    # The guard refuses the path before checking whether a real file exists.
    with pytest.raises(AssertionError, match='real project data'):
        (ROOT / 'data/raw/train.csv').open('rb')


def _skip_bootstrap(monkeypatch):
    monkeypatch.setattr(alpha_pipeline, 'blockwise_spearman_interval',
                        lambda *a, **k: {'lower': np.nan, 'upper': np.nan, 'valid_repetitions': 0})


def test_late_feature_and_all_missing_keep_not_testable_p_one(monkeypatch):
    _skip_bootstrap(monkeypatch)
    history = make_synthetic_history(ROOT)
    history.loc[:7100, 'E7'] = np.nan
    history.loc[7101:, 'E7'] = np.sin(np.arange(759))
    tables = build_alpha_tables(history, load_alpha_design(ROOT))
    late = tables.single_features.set_index('feature').loc['E7']
    assert 0 < late['eligible_fold_count'] < 3
    assert late['status'] == 'not_testable' and late['raw_p_value'] == 1.
    assert tables.single_features.set_index('feature').loc['V1', 'raw_p_value'] == 1.
    assert tables.feature_availability.query("feature == 'V1'")['reasons'].str.contains('insufficient_observations').all()


def test_training_decisions_ignore_changed_validation_and_future(monkeypatch):
    _skip_bootstrap(monkeypatch)
    history = make_synthetic_history(ROOT)
    changed = history.copy(deep=True)
    changed.loc[5088:, 'M1'] = np.nan
    changed.loc[5088:, 'M2'] *= -1e8
    changed.loc[5088:, 'market_forward_excess_returns'] *= -100
    a = build_alpha_tables(history, load_alpha_design(ROOT))
    b = build_alpha_tables(changed, load_alpha_design(ROOT))
    pd.testing.assert_frame_equal(a.feature_availability.query('fold == 1').reset_index(drop=True),
                                  b.feature_availability.query('fold == 1').reset_index(drop=True))
    assert a.fold_transforms[0] == b.fold_transforms[0]
    assert a.fold_transforms[0]['family_composites']['M']['directions'] == {'M1': 1, 'M2': -1}
    # A later training window may legitimately absorb earlier validation rows.
    assert a.fold_transforms[-1] != b.fold_transforms[-1]


def test_family_missingness_weak_members_and_discrete_d(end_to_end):
    history, result = end_to_end
    first = result.tables.fold_transforms[0]['family_composites']
    assert first['M']['qualified_members'] == ['M1', 'M2']
    assert first['M']['required_per_row'] == 2
    assert first['E']['qualified_members'] == ['E1', 'E2']
    assert first['D']['qualified_members'] == ['D1', 'D2']
    assert first['V']['qualified_members'] == []
    for family, metadata in first.items():
        assert not set(metadata['qualified_members']) & set(LAGGED_OUTCOME_COLUMNS)
    d = result.tables.family_composites.set_index('feature').loc['D']
    assert d['eligible_fold_count'] == 11 and np.isfinite(d['oos_rank_ic'])


def test_no_available_features_still_returns_both_full_families(monkeypatch):
    _skip_bootstrap(monkeypatch)
    history = make_synthetic_history(ROOT)
    history.loc[:, history.columns != 'date_id'] = np.nan
    design = load_alpha_design(ROOT)
    tables = build_alpha_tables(history, design)
    apply_multiple_testing(tables, design)
    for table, count in [(tables.single_features, 97), (tables.family_composites, 7)]:
        assert len(table) == count and table['raw_p_value'].eq(1).all()
        assert table['formal_conclusion'].eq('NOT_TESTABLE').all()


def test_undefined_hac_uncertainty_cannot_reject(monkeypatch):
    _skip_bootstrap(monkeypatch)
    x = np.tile(np.arange(252, dtype=float), 3)
    blocks = np.repeat([1, 2, 3], 252)
    diagnostics = [{'fold': i, 'eligible': True, 'training_eligible': True, 'fold_ic': 1.} for i in [1, 2, 3]]
    row = summarize_hypothesis('M1', 'M', x, x, x, blocks, diagnostics, load_alpha_design(ROOT).rules)
    assert row['status'] == 'not_testable' and row['raw_p_value'] == 1.
    assert row['not_testable_reason'] == 'undefined_hac_uncertainty'


def test_protocol_unchanged_after_synthetic_execution(end_to_end):
    assert verify_protocol_lock(ROOT)['protocol_version'] == 'phase2.5-v2'


def test_minimum_validation_pairs_and_half_coverage_are_enforced(monkeypatch):
    _skip_bootstrap(monkeypatch)
    history = make_synthetic_history(ROOT)
    # Every training window still has ample observations and <=25% recent
    # missingness; availability cannot inspect the first validation block.
    history.loc[5088:5213, 'M1'] = np.nan  # 126 pairs: exactly half the block.
    history.loc[5088:5214, 'M2'] = np.nan  # 125 pairs: below both cutoffs.
    tables = build_alpha_tables(history, load_alpha_design(ROOT))
    fold = tables.fold_level_ic.query('fold == 1').set_index('hypothesis')
    assert fold.loc['M1', 'training_eligible'] and fold.loc['M2', 'training_eligible']
    assert fold.loc['M1', 'pairs'] == 126 and fold.loc['M1', 'eligible']
    assert fold.loc['M2', 'pairs'] == 125 and not fold.loc['M2', 'eligible']
    assert 'insufficient_validation_pairs' in fold.loc['M2', 'reasons']


def test_weak_mixed_family_equal_weights_and_row_missingness():
    rng = np.random.default_rng(982)
    y = pd.Series(rng.normal(size=5087))
    train = pd.DataFrame({f'M{i}': sign * .08 * y + rng.normal(size=len(y))
                          for i, sign in enumerate([1, -1, 1, -1, 1], 1)})
    validation = pd.DataFrame(rng.normal(size=(4, 5)), columns=train.columns)
    validation.iloc[1, 3:] = np.nan  # 3/5 meets the 50% rule.
    validation.iloc[2, 2:] = np.nan  # 2/5 fails the 50% rule.
    validation.iloc[3, 1:] = np.nan  # Fewer than two fails as well.
    availability = alpha_pipeline.feature_availability(train, load_alpha_design(ROOT).rules['feature_availability'])
    composite, metadata = family_composite('M', train, y, validation, availability)
    directions = {c: training_direction(train[c], y) for c in train}
    assert set(directions.values()) == {-1, 1}
    assert metadata['directions'] == directions and metadata['required_per_row'] == 3
    expected = pd.DataFrame({c: directions[c] * training_percentile(train[c], validation[c]) for c in train})
    np.testing.assert_allclose(composite.iloc[:2], expected.iloc[:2].mean(axis=1))
    assert composite.iloc[2:].isna().all()
    # This test intentionally makes no significance claim about weak members.


def test_bootstrap_is_descriptive_and_no_rerank_during_inference(end_to_end, monkeypatch):
    history, result = end_to_end
    row = result.tables.single_features.set_index('feature').loc['M1']
    assert row['bootstrap_valid_repetitions'] == 2000
    assert np.isfinite(row['bootstrap_ci_lower']) and np.isfinite(row['bootstrap_ci_upper'])
    assert not any('bootstrap_p' in c for c in result.tables.single_features.columns)
    # Integration already checks the NW p-value against the frozen primitive.
    # Replace only the CI, proving it cannot change formal decisions.
    changed = deepcopy(result.tables)
    changed.single_features['bootstrap_ci_lower'] = -1.
    changed.single_features['bootstrap_ci_upper'] = 1.
    apply_multiple_testing(changed, load_alpha_design(ROOT))
    pd.testing.assert_series_equal(changed.single_features['formal_conclusion'], result.tables.single_features['formal_conclusion'])


def test_real_output_preflight_precedes_safe_loader(tmp_path, monkeypatch):
    (tmp_path / 'artifacts/phase3').mkdir(parents=True)
    (tmp_path / 'artifacts/phase3/existing.json').write_text('{}')
    monkeypatch.setattr(phase3, 'verify_protocol_lock', lambda *a, **k: {})
    monkeypatch.setattr(phase3, 'load_research_prefix', lambda *a, **k: pytest.fail('Output must be checked before loading'))
    with pytest.raises(FileExistsError):
        phase3.run_real_phase3(tmp_path, run_real_phase3=True)
