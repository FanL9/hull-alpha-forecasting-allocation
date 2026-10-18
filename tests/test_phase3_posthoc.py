"""Artificial-only tests for separate post-hoc diagnostics and immutable inputs."""
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.features.availability import available_lagged_outcomes
from src.features.family_composites import family_composite
from src.research.alpha_calibration import make_calibration_history
from src.research import posthoc_diagnostics as diagnostic
from src.research.posthoc_diagnostics import (EXPECTED_COMPOSITES, EXPECTED_SINGLES, SealedPhase3,
    diagnostic_tables, load_sealed_phase3, protected_hashes, reconstruct_composites)
from src.research.posthoc_statistics import (average_linkage, correlation_structure, decile_analysis,
    divergence_summary, lag_decay, pearson, rank_ic, reverse_return_ic)
from src.utils.paths import project_root
from src.validation.protocol_freeze import verify_protocol_lock

ROOT = project_root()


def grid(n=12000, start=2000):
    index = pd.Index(np.arange(n), name='date_id')
    dates = index[start:]
    blocks = pd.Series(np.repeat(np.arange(10), (n - start) // 10), index=dates)
    masks = pd.DataFrame(True, index=dates, columns=['M1'])
    return index, blocks, masks


def test_persistent_signal_decays_gradually_on_original_grid():
    rng = np.random.default_rng(81)
    index, blocks, masks = grid()
    values = np.zeros(len(index))
    for i in range(1, len(values)):
        values[i] = .9 * values[i - 1] + rng.normal()
    x = pd.DataFrame({'M1': values}, index=index)
    y = pd.Series(values + rng.normal(scale=.5, size=len(values)), index=index)
    long, summary = lag_decay(x, y, blocks, masks)
    assert len(long) == 6 and np.all(np.diff(long['rank_ic']) < 0)
    assert 0 < summary.iloc[0]['relative_drop_k0_to_k1'] < .3
    assert long['n_obs'].eq(len(blocks)).all()


def test_same_row_synthetic_relationship_has_a_lag_cliff():
    rng = np.random.default_rng(82)
    index, blocks, masks = grid()
    y = pd.Series(rng.normal(size=len(index)), index=index)
    x = pd.DataFrame({'M1': y + rng.normal(scale=.01, size=len(index))}, index=index)
    long, summary = lag_decay(x, y, blocks, masks)
    assert long.iloc[0]['rank_ic'] > .99 and abs(long.iloc[1]['rank_ic']) < .04
    assert summary.iloc[0]['relative_drop_k0_to_k1'] > .95


def test_negative_signal_drop_uses_magnitude_not_signed_difference():
    rng = np.random.default_rng(83)
    index, blocks, masks = grid()
    y = pd.Series(rng.normal(size=len(index)), index=index)
    x = pd.DataFrame({'M1': -y + rng.normal(scale=.01, size=len(index))}, index=index)
    _, summary = lag_decay(x, y, blocks, masks)
    row = summary.iloc[0]
    assert row['IC_k0'] < -.99 and row['drop_k0_to_k1'] > .95
    assert row['drop_k0_to_k1'] == pytest.approx(abs(row['IC_k0']) - abs(row['IC_k1']))


def test_reverse_test_recovers_previous_outcome_without_calling_it_leakage():
    rng = np.random.default_rng(84)
    index, blocks, masks = grid()
    y = pd.Series(rng.normal(size=len(index)), index=index)
    x = pd.DataFrame({'M1': y.shift(1)}, index=index)
    reverse = reverse_return_ic(x, y, blocks, masks)
    assert reverse.iloc[0]['rank_ic'] == pytest.approx(1.)
    assert reverse.iloc[1:]['rank_ic'].abs().max() < .04


def test_shifts_precede_oos_slicing_and_nan_positions_are_not_compressed():
    rng = np.random.default_rng(85)
    dates = pd.Index(np.arange(120), name='date_id')
    x = pd.DataFrame({'M1': rng.normal(size=120)}, index=dates)
    y = pd.Series(rng.normal(size=120), index=dates)
    blocks = pd.Series(np.repeat([1, 2], 30), index=dates[60:])
    mask = pd.DataFrame(True, index=blocks.index, columns=x.columns)
    long, _ = lag_decay(x, y, blocks, mask)
    assert long.iloc[1]['n_obs'] == 60  # x_59 is used for y_60.
    x.loc[75, 'M1'] = np.nan
    y.loc[90] = np.nan
    changed, _ = lag_decay(x, y, blocks, mask)
    assert changed.iloc[0]['n_obs'] == 58 and changed.iloc[1]['n_obs'] == 58
    expected, count = rank_ic(x['M1'].shift(1).loc[blocks.index], y.loc[blocks.index], blocks)
    assert changed.iloc[1]['rank_ic'] == pytest.approx(expected) and count == 58
    mask.loc[90:, 'M1'] = False
    restricted, _ = lag_decay(x, y, blocks, mask)
    assert restricted.iloc[0]['n_obs'] == 29


def test_monotone_heavy_tail_rank_pearson_divergence_and_deciles():
    rng = np.random.default_rng(86)
    latent = rng.normal(size=3000)
    x = pd.DataFrame({'M9': np.exp(5 * latent)})
    y = pd.Series(latent + rng.normal(scale=.05, size=3000))
    rank, _ = rank_ic(x['M9'], y)
    linear, _ = pearson(x['M9'], y)
    assert rank > .99 and abs(linear) < .4
    deciles = decile_analysis(x, y, 'synthetic')
    assert len(deciles) == 10 and deciles['n_obs'].sum() == 3000
    assert (np.diff(deciles['mean_y']) > 0).all() and (np.diff(deciles['median_y']) > 0).all()
    clipped = x.clip(lower=x.quantile(.01), upper=x.quantile(.99), axis=1)
    blocks = pd.Series(np.repeat([1, 2, 3], 1000))
    frozen_rank, _ = rank_ic(x['M9'], y, blocks)
    winsor, _ = pearson(clipped['M9'], y)
    original = pd.DataFrame([{'feature': 'M9', 'oos_rank_ic': frozen_rank, 'winsorized_pearson_ic': winsor}])
    result = divergence_summary(x, clipped, y, blocks, original).iloc[0]
    assert result['x_core_winsor_corr_contribution'] + result['x_tail_winsor_corr_contribution'] == pytest.approx(winsor)
    assert result['y_core_winsor_corr_contribution'] + result['y_extreme_winsor_corr_contribution'] == pytest.approx(winsor)


def test_tied_deciles_keep_identical_values_together():
    x = pd.DataFrame({'D1': np.repeat([0., 1.], 50)})
    result = decile_analysis(x, pd.Series(np.arange(100.)), 'synthetic')
    assert result['n_bins'].max() < 10
    assert result['n_obs'].sum() == 100 and set(result['mean_x']) == {0., 1.}


def test_average_linkage_recovers_two_correlated_groups_and_negative_orientation():
    rng = np.random.default_rng(87)
    a, b = rng.normal(size=(2, 2000))
    x = pd.DataFrame({'M1': a, 'M3': -a + rng.normal(scale=.03, size=len(a)),
                      'P2': b, 'P3': b + rng.normal(scale=.03, size=len(b))})
    blocks = pd.Series(np.repeat([1, 2], 1000))
    _, rank, _, pairs, long = correlation_structure(x, blocks)
    hierarchy, groups = average_linkage(rank, .2)
    assert set(groups['members']) == {'M1;M3', 'P2;P3'}
    assert groups['member_count'].tolist() == [2, 2]
    assert (groups['mean_abs_within_cluster_correlation'] > .99).all()
    assert np.all(np.diff(hierarchy['distance']) >= -1e-12)
    assert len(long) == 6 and pairs.eq(2000).all().all()


def test_singleton_clusters_do_not_claim_perfect_within_correlation():
    matrix = pd.DataFrame([[1., np.nan], [np.nan, 1.]], index=['M1', 'P2'], columns=['M1', 'P2'])
    hierarchy, groups = average_linkage(matrix, .2)
    assert len(groups) == 2 and groups['mean_abs_within_cluster_correlation'].isna().all()
    assert hierarchy.iloc[0]['distance'] == 1.


@pytest.fixture(scope='module')
def artificial_archive():
    history = make_calibration_history(ROOT, 707)
    frame = history.set_index('date_id')
    x = pd.concat([frame[[c for c in frame if c[0] in 'MEIPVSD']],
                   available_lagged_outcomes(history).set_axis(frame.index)], axis=1)
    target = frame['market_forward_excess_returns']
    actual = load_sealed_phase3(ROOT)
    manifest = deepcopy(actual.manifest)
    metadata = manifest['folds']
    blocks = pd.Series(np.repeat(np.arange(1, 12), 252), index=np.arange(5088, 7860))
    folds, availability, recipes = [], [], []
    composite = pd.DataFrame(np.nan, index=frame.index, columns=EXPECTED_COMPOSITES)
    members = {'M': ['M1', 'M3'], 'P': ['P2', 'P3'], 'V': ['V7', 'V9'], 'S': ['S2', 'S3']}
    for fold in metadata:
        train = x.loc[fold['train_start']:fold['train_end']]
        validation = x.loc[fold['validation_start']:fold['validation_end']]
        audit = {'fold': fold['fold'], 'train_start': fold['train_start'], 'train_end': fold['train_end'], 'family_composites': {}}
        for family, names in members.items():
            available = pd.DataFrame({'feature': names, 'eligible': True})
            signal, recipe = family_composite(family, train[names], target.loc[train.index], validation[names], available)
            recipe['percentile_reference_sha256'] = {name: hashlib.sha256(np.sort(train[name].dropna().to_numpy(dtype='<f8')).tobytes()).hexdigest() for name in recipe['qualified_members']}
            audit['family_composites'][family] = recipe
            composite.loc[validation.index, family] = signal
        recipes.append(audit)
        for name in EXPECTED_SINGLES + EXPECTED_COMPOSITES:
            source = x if name in EXPECTED_SINGLES else composite
            ic, n = rank_ic(source.loc[validation.index, name], target.loc[validation.index])
            folds.append({'hypothesis': name, 'fold': fold['fold'], 'eligible': True, 'fold_ic': ic,
                          'pairs': n, 'winsorized_pearson_ic': np.nan})
        for name in EXPECTED_SINGLES:
            bounds = train[name].quantile([.01, .99])
            availability.append({'feature': name, 'fold': fold['fold'], 'eligible': True,
                                 'winsor_lower': bounds.iloc[0], 'winsor_upper': bounds.iloc[1]})
    clipped = x.loc[blocks.index, list(EXPECTED_SINGLES)].copy()
    for row in availability:
        ids = blocks.index[blocks.eq(row['fold'])]
        clipped.loc[ids, row['feature']] = clipped.loc[ids, row['feature']].clip(row['winsor_lower'], row['winsor_upper'])
    single_rows, composite_rows = [], []
    for names, values, destination in [(EXPECTED_SINGLES, x, single_rows), (EXPECTED_COMPOSITES, composite, composite_rows)]:
        for name in names:
            ic, n = rank_ic(values.loc[blocks.index, name], target.loc[blocks.index], blocks)
            secondary = pearson(clipped[name], target.loc[blocks.index])[0] if name in EXPECTED_SINGLES else np.nan
            destination.append({'feature': name, 'oos_rank_ic': ic, 'oos_pairs': n,
                                'winsorized_pearson_ic': secondary, 'formal_conclusion': 'PASS'})
    sealed = SealedPhase3(manifest, pd.DataFrame(single_rows), pd.DataFrame(composite_rows),
        pd.DataFrame(folds), pd.DataFrame(availability), recipes, {})
    return history, sealed


def test_end_to_end_diagnostics_never_calls_confirmatory_tools(artificial_archive, monkeypatch):
    import src.research.alpha_pipeline as alpha
    import src.research.phase3 as phase3
    import src.statistics.multiple_testing as testing
    import src.statistics.robust_inference as inference
    def forbidden(*a, **k):
        pytest.fail('Post-hoc diagnostics invoked confirmatory research')
    monkeypatch.setattr(alpha, 'build_alpha_tables', forbidden)
    monkeypatch.setattr(phase3, 'run_real_phase3', forbidden)
    monkeypatch.setattr(testing, 'adjust_fdr', forbidden)
    monkeypatch.setattr(testing, 'protocol_multiplicity', forbidden)
    monkeypatch.setattr(inference, 'blockwise_spearman_hac', forbidden)
    monkeypatch.setattr(inference, 'blockwise_spearman_interval', forbidden)
    history, sealed = artificial_archive
    tables = diagnostic_tables(history, ROOT, sealed)
    assert len(tables['lag_decay']) == 24 * 6 and len(tables['reverse_return_ic']) == 24 * 5
    assert len(tables['composite_lag_decay']) == 4 * 6
    assert len(tables['decile_analysis']) == 24 * 10 * 2
    assert len(tables['pass_feature_correlations']) == 24 * 23 // 2
    assert len(tables['k0_reconciliation']) == 28
    assert tables['k0_reconciliation']['difference'].abs().max() < 1e-12
    assert tables['lagged_field_timing_audit']['exact_previous_row_match'].all()
    assert all('diagnostic_type' in table for table in tables.values())


def test_composite_shift_uses_historical_fold_value_and_no_invented_prefix(artificial_archive):
    history, sealed = artificial_archive
    frame = history.set_index('date_id')
    candidates = pd.concat([frame[[c for c in frame if c[0] in 'MEIPVSD']],
                           available_lagged_outcomes(history).set_axis(frame.index)], axis=1)
    composite = reconstruct_composites(candidates, sealed)
    assert composite.loc[:5087].isna().all().all()
    shifted = composite.shift(1)
    pd.testing.assert_series_equal(shifted.loc[5340], composite.loc[5339], check_names=False)
    assert shifted.loc[5088].isna().all()
    altered = deepcopy(sealed)
    altered.transforms[0]['family_composites']['M']['percentile_reference_sha256']['M1'] = '0' * 64
    with pytest.raises(ValueError, match='percentile reference'):
        reconstruct_composites(candidates, altered)


def test_synthetic_report_and_figures_are_explicitly_posthoc(artificial_archive, tmp_path):
    from src.research.posthoc_reporting import write_diagnostic_report
    history, sealed = artificial_archive
    tables = diagnostic_tables(history, ROOT, sealed)
    output = tmp_path / 'diagnostics'
    output.mkdir()
    report = write_diagnostic_report(tables, sealed.manifest, output, tmp_path / 'report.md')
    assert 'This analysis is post-hoc and does not alter' in report
    assert 'exploratory' in report.lower() and 'No Internal Holdout or Final-180 data were used.' in report
    assert len(list((output / 'figures').glob('*.png'))) == 5
    assert all(p.stat().st_size > 1000 for p in (output / 'figures').glob('*.png'))


@pytest.mark.parametrize('outside', [7860, 8868])
def test_diagnostics_rejects_heldout_rows_before_calculations(artificial_archive, outside):
    history, sealed = artificial_archive
    extra = history.iloc[[-1]].copy()
    extra['date_id'] = outside
    wider = pd.concat([history, extra], ignore_index=True)
    with pytest.raises(ValueError, match='0–7859'):
        diagnostic_tables(wider, ROOT, sealed)


def test_safe_loader_order_and_preservation_checks_with_doubles(tmp_path, artificial_archive, monkeypatch):
    history, sealed = artificial_archive
    events = []
    def verify(root, *, verify_data):
        events.append(('verify', verify_data))
        if verify_data:
            raise ValueError('artificial data identity failure')
        return {}
    monkeypatch.setattr(diagnostic, 'verify_protocol_lock', verify)
    monkeypatch.setattr(diagnostic, 'load_sealed_phase3', lambda root: sealed)
    monkeypatch.setattr(diagnostic, 'assert_preserved', lambda *a: events.append(('preserved', True)))
    monkeypatch.setattr(diagnostic, 'load_research_prefix', lambda *a: pytest.fail('Verification failed'))
    monkeypatch.setattr(diagnostic, 'git', lambda *a: b'synthetic')
    with pytest.raises(ValueError, match='identity failure'):
        diagnostic.run_posthoc(tmp_path)
    assert events == [('verify', False), ('verify', True), ('preserved', True)]


def test_completed_diagnostic_outputs_cannot_be_overwritten(tmp_path):
    (tmp_path / 'reports/phase3_posthoc_diagnostics').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        diagnostic.run_posthoc(tmp_path)


def test_cli_never_defaults_to_a_real_or_confirmatory_run(monkeypatch):
    monkeypatch.setattr(diagnostic, 'run_posthoc', lambda *a: pytest.fail('Missing opt-in'))
    with pytest.raises(SystemExit) as error:
        diagnostic.main([])
    assert error.value.code == 2


def test_formal_snapshot_and_protocol_remain_unchanged_after_unit_tests():
    before = protected_hashes(ROOT)
    sealed = load_sealed_phase3(ROOT)
    diagnostic.assert_preserved(ROOT, sealed.protected_hashes)
    assert protected_hashes(ROOT) == before
    assert verify_protocol_lock(ROOT, verify_data=False)['protocol_version'] == 'phase2.5-v2'
