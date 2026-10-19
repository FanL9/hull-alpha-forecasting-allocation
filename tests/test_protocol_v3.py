"""Protocol contracts over metadata/IDs only; no Phase 4/5 implementation."""
from copy import deepcopy
import json

import pytest

from src.utils.paths import project_root
from src.validation import protocol_v3 as v3
from src.validation.protocol_freeze import NORMATIVE_FILES, verify_protocol_lock

ROOT = project_root()


@pytest.fixture
def config():
    return v3.load_config(ROOT)


@pytest.fixture
def metadata_clone(tmp_path):
    """Explicit metadata inventory: never copy data/ or ignored artifacts/."""
    manifest = json.loads((ROOT / 'reports/phase3_real_run/phase3_run_manifest.json').read_text())
    names = {*NORMATIVE_FILES, *v3.NORMATIVE, 'configs/protocol_lock.json',
             *manifest['implementation_sha256'], *v3.protected_report_paths(ROOT)}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    return tmp_path


def test_closed_universe_is_enumerable_without_market_values(config):
    raw = v3.raw_names(config)
    names = v3.candidate_ids(config)
    assert len(raw) == len(set(raw)) == 97
    assert len(names) == len(set(names)) == 499
    assert sum(n.startswith('raw/') for n in names) == 485
    assert sum(n.startswith('interaction/') for n in names) == 7
    assert not any('MOM' in n or 'Delta1/' in n or 'LowV' in n for n in names)
    assert 'raw/lagged_risk_free_rate/Rank63' in names


@pytest.mark.parametrize(('path', 'value'), [
    (('research', 'end'), 7860),
    (('input_universe', 'phase3_pass_global_prefilter'), True),
    (('transformations', 'centered'), True),
    (('transformations', 'allowed_max_source_offset'), 1),
    (('transformations', 'impute_before_transform'), True),
    (('transformations', 'delta_windows'), [1, 5, 21]),
    (('transformations', 'zscore', 'min_periods'), 1),
    (('transformations', 'rank', 'window_unit'), 'trading_rows'),
    (('interactions', 'additional_pairwise'), True),
    (('selection', 'fit_on'), 'outer_validation'),
    (('selection', 'top_k'), 32),
    (('selection', 'fixed_ic_threshold'), .04),
    (('selection', 'redundancy', 'fit_on'), 'outer_validation'),
    (('regime', 'fit_on'), 'outer_validation'),
    (('regime', 'validation_y_used'), True),
    (('regime', 'quantile'), .8),
    (('deep', 'scaling', 'fit_on'), 'full_history'),
    (('deep', 'phase3_posthoc_clusters_reused'), True),
    (('deep', 'seeds'), [11]),
    (('models', 'gru', 'layers'), 2),
    (('models', 'tcn', 'padding'), 'centered_same'),
    (('models', 'tcn', 'trainable_parameters'), 999),
    (('training', 'early_stopping', 'outer_validation_used'), True),
    (('training', 'early_stopping', 'selector_scaler_fit_on'), 'stopping_block'),
    (('incremental', 'outer_y_used_to_fit'), True),
    (('combined', 'normalization_fit_on'), 'outer_validation'),
    (('primary_selection', 'minimum_positive_folds'), 6),
    (('primary_selection', 'simplicity_margin'), .005),
    (('holdout', 'locked'), False),
    (('final_simulation', 'start'), 8867),
])
def test_config_rejects_leakage_and_post_result_rule_changes(config, path, value):
    bad = deepcopy(config)
    container = bad
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    with pytest.raises(ValueError):
        v3.validate_config(bad)


def test_inner_and_stopping_splits_stay_before_every_outer_validation(config):
    # Original ID-only fold formula, including both one-row purges.
    for fold in range(11):
        start = 5088 + 252 * fold
        outer = v3.TemporalSplit(0, start - 2, start - 1, start, start + 251)
        splits = v3.inner_splits(outer.training_end, config)
        assert [s.validation_end - s.validation_start + 1 for s in splits] == [252, 252]
        assert splits[0].validation_end + 1 == splits[1].validation_start
        assert splits[-1].validation_end == outer.training_end
        for s in splits:
            assert s.training_end + 1 == s.purge_id == s.validation_start - 1
            assert s.validation_end < outer.purge_id
            stopping = v3.stopping_split(s.training_end, config)
            assert stopping.validation_end < s.validation_start
            assert stopping.validation_end - stopping.validation_start + 1 == 252
            # Selection inside a model fit has its own earlier reference.
            for earlier in v3.inner_splits(stopping.training_end, config):
                assert earlier.validation_end <= stopping.training_end < s.validation_start
            for role in v3.FIT_ROLES:
                v3.assert_fit_scope(role, 0, s.training_end, s, config)
                with pytest.raises(ValueError, match='purged training'):
                    v3.assert_fit_scope(role, 0, s.validation_start, s, config)


def test_first_inner_dates_and_last_outer_boundary_are_exact(config):
    assert v3.inner_splits(5086, config) == (
        v3.TemporalSplit(0, 4581, 4582, 4583, 4834),
        v3.TemporalSplit(0, 4833, 4834, 4835, 5086))
    assert v3.stopping_split(4581, config) == v3.TemporalSplit(0, 4328, 4329, 4330, 4581)
    assert 5088 + 10 * 252 + 251 == 7859


@pytest.mark.parametrize('end', [0, 503, 7860, 8867, 8868, 9047])
def test_no_shortened_or_heldout_reference(config, end):
    with pytest.raises(ValueError):
        v3.inner_splits(end, config)


@pytest.mark.parametrize(('source', 'decision'), [(0, 0), (0, 7859), (7607, 7608), (7858, 7859)])
def test_current_and_past_source_metadata_allowed(config, source, decision):
    v3.assert_source_date(source, decision, config)


@pytest.mark.parametrize(('source', 'decision'), [(1, 0), (7860, 7859), (7860, 7860), (9047, 9047), (-1, 0)])
def test_future_pre_history_holdout_and_final_sources_rejected(config, source, decision):
    with pytest.raises(ValueError):
        v3.assert_source_date(source, decision, config)


def test_parameter_budget_includes_seed_ensemble_and_fair_ridge(config):
    counts = v3.parameter_counts(config)
    assert counts['ridge'] == 63 * 16 + 1
    assert counts['gru'] == 633 and counts['tcn'] == 1137
    assert counts['nonlinear_ensemble'] == 5 * (633 + 1137) <= 10000
    assert config['models']['tcn']['pooling'] == 'mean_all_63_causal_positions'


def test_document_discloses_prior_observations_and_unimplemented_contracts():
    text = (ROOT / v3.DOCUMENT).read_text(encoding='utf-8')
    for sentence in ['Delta5 and Delta21 are **not** first-ever unseen transformations',
                     'Phase 2 descriptive EDA covered Development 0–8867',
                     'A is an explicitly new **v3 raw-only benchmark pipeline**',
                     'A config test is a design contract', 'Top-K=16', 'M−mean < 0.01']:
        assert sentence in text


def test_freeze_and_verification_cannot_parse_real_data(metadata_clone, monkeypatch):
    import pandas as pd
    import src.validation.walk_forward as walk
    def forbidden(*args, **kwargs):
        pytest.fail('Protocol-only checks must never load X/Y')
    monkeypatch.setattr(pd, 'read_csv', forbidden)
    monkeypatch.setattr(walk, 'load_research_prefix', forbidden)
    before = (ROOT / 'configs/protocol_lock.json').read_bytes()
    lock = v3.freeze(metadata_clone)
    assert v3.verify_lock(metadata_clone) == lock
    assert len(lock['protected_report_sha256']) == 43
    assert not lock['phase4_phase5_results_seen'] and not lock['raw_data_opened_this_task']
    assert (ROOT / 'configs/protocol_lock.json').read_bytes() == before
    assert not (metadata_clone / 'data').exists()


def test_lock_rejects_changed_document_without_rewriting(metadata_clone):
    v3.freeze(metadata_clone)
    saved = (metadata_clone / v3.LOCK).read_bytes()
    path = metadata_clone / v3.DOCUMENT
    path.write_text(path.read_text(encoding='utf-8') + '\nUnregistered rule.\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Frozen v3 file changed'):
        v3.freeze(metadata_clone)
    assert (metadata_clone / v3.LOCK).read_bytes() == saved


def test_lock_preserves_prior_diagnostics_and_parent_protocol(metadata_clone):
    lock = v3.freeze(metadata_clone)
    path = metadata_clone / 'reports/phase3_posthoc_extended.md'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Inherited sealed evidence changed'):
        v3.verify_lock(metadata_clone)
    assert lock['phase2_5_lock_sha256'] == v3.digest(ROOT / 'configs/protocol_lock.json')


def test_annotated_tag_binds_all_frozen_paths(metadata_clone, monkeypatch):
    v3.freeze(metadata_clone)
    commit = 'a' * 40
    def git(root, *args):
        if args[0] == 'cat-file':
            return b'tag\n'
        if args[0] in ['rev-parse', 'merge-base']:
            return commit.encode() + b'\n'
        if args[0] == 'show':
            return (root / args[1].split(':', 1)[1]).read_bytes()
        pytest.fail('Unexpected git operation')
    monkeypatch.setattr(v3, 'git', git)
    assert v3.verify_tag(metadata_clone) == commit
    monkeypatch.setattr(v3, 'git', lambda *args: b'commit\n')
    with pytest.raises(ValueError, match='annotated'):
        v3.verify_tag(metadata_clone)


def test_metadata_cli_requires_explicit_operation_and_exposes_no_real_runner(monkeypatch):
    monkeypatch.setattr(v3, 'freeze', lambda *a: pytest.fail('Implicit freeze'))
    with pytest.raises(SystemExit) as error:
        v3.main([])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        v3.main(['--run-real-phase4'])


def test_v2_and_phase3_identity_remain_unchanged():
    lock = verify_protocol_lock(ROOT, verify_data=False)
    identity = v3.inherited_identity(ROOT)
    assert lock['protocol_version'] == 'phase2.5-v2'
    assert identity['phase3_pipeline_code_hash'] == '987ac1a342c2a551155eb855997aaa842d528f52ed7d9495971406bab9f4ddba'
    assert len(identity['protected_report_sha256']) == 43
