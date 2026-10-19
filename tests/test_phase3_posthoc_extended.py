"""Artificial-only tests for the long-lag and ΔX post-hoc extension."""
import numpy as np
import pandas as pd
import pytest

from src.research import posthoc_extended as extended
from src.research.posthoc_diagnostics import EXPECTED_COMPOSITES, EXPECTED_SINGLES, protected_hashes
from src.research.posthoc_extended import (LONG_LAGS, change_signal_ic, extended_tables, long_lag_decay,
                                           summarize_long_lag, write_extended_report)
from src.utils.paths import project_root
from src.validation.protocol_freeze import verify_protocol_lock
from tests.test_phase3_posthoc import artificial_archive  # noqa: F401  (shared synthetic fixture)

ROOT = project_root()


def grid(n=9000, start=3000, blocks=6):
    index = pd.Index(np.arange(n), name='date_id')
    dates = index[start:]
    labels = pd.Series(np.repeat(np.arange(1, blocks + 1), (n - start) // blocks), index=dates)
    return index, labels, pd.DataFrame(True, index=dates, columns=['M1'])


def by_lag(table, column='common_sample_rank_ic'):
    return table.set_index('lag_k')[column]


def test_slow_level_effect_persists_to_long_lags():
    rng = np.random.default_rng(11)
    index, blocks, masks = grid()
    level = np.zeros(len(index))
    for i in range(1, len(level)):  # AR(1) with a ~200-row half-life, varying inside each block
        level[i] = .9965 * level[i - 1] + rng.normal()
    persistent = pd.DataFrame({'M1': level}, index=index)
    y = pd.Series(.3 * level / level.std() + rng.normal(size=len(index)), index=index)
    ic = by_lag(long_lag_decay(persistent, y, blocks, masks))
    assert ic[0] > .1
    assert ic[21] / ic[0] > .7 and ic[63] / ic[0] > .5


def test_centered_smoothing_leak_collapses_after_its_half_window():
    rng = np.random.default_rng(12)
    index, blocks, masks = grid()
    y = pd.Series(rng.normal(size=len(index)), index=index)
    leak = y.rolling(61, center=True, min_periods=61).mean()  # uses y_(t−30)…y_(t+30)
    ic = by_lag(long_lag_decay(pd.DataFrame({'M1': leak}, index=index), y, blocks, masks))
    assert ic[0] > .08 and ic[21] > .04
    assert abs(ic[63]) < .03 and abs(ic[126]) < .03


def test_shifts_only_use_earlier_rows_and_common_sample_is_fixed():
    rng = np.random.default_rng(13)
    index, blocks, masks = grid(n=1200, start=600, blocks=2)
    x = pd.DataFrame({'M1': rng.normal(size=len(index))}, index=index)
    y = pd.Series(rng.normal(size=len(index)), index=index)
    x.loc[600 - 252 + 10, 'M1'] = np.nan  # removes one common-sample row through k=252
    table = long_lag_decay(x, y, blocks, masks)
    assert table['common_sample_n_obs'].nunique() == 1
    assert table['common_sample_n_obs'].iloc[0] == 600 - 1
    future = x.copy()
    future.loc[1200 - 1, 'M1'] = 1e6  # the last row can only appear at k=0
    changed = long_lag_decay(future, y, blocks, masks)
    pd.testing.assert_series_equal(changed.loc[changed.lag_k > 0, 'rank_ic'].reset_index(drop=True),
                                   table.loc[table.lag_k > 0, 'rank_ic'].reset_index(drop=True))


def test_change_signal_separates_latest_update_from_slow_level():
    rng = np.random.default_rng(14)
    index, blocks, masks = grid()
    news = rng.normal(size=len(index))
    y = pd.Series(.3 * news + rng.normal(size=len(index)), index=index)
    accumulating = pd.DataFrame({'M1': np.cumsum(news)}, index=index)  # today's update carries the signal
    result = change_signal_ic(accumulating, y, blocks, masks).set_index('horizon_h')
    assert result.loc[1, 'change_rank_ic'] > .2
    assert abs(result.loc[1, 'level_rank_ic']) < result.loc[1, 'change_rank_ic'] / 2
    stepwise = pd.DataFrame({'M1': np.repeat(rng.normal(size=10), 900)}, index=index)
    flat = change_signal_ic(stepwise, y, blocks, masks).set_index('horizon_h')
    assert flat.loc[1, 'nonzero_change_fraction'] < .01


def test_summary_retained_share_keeps_sign():
    table = pd.DataFrame({'feature': ['M1'] * 3, 'lag_k': [0, 1, 5],
                          'common_sample_rank_ic': [-.1, -.05, .02], 'common_sample_n_obs': [100] * 3})
    summary = summarize_long_lag(table).iloc[0]
    assert summary['retained_k1'] == pytest.approx(.5) and summary['retained_k5'] == pytest.approx(-.2)


def test_extended_tables_reconcile_k0_and_report_is_posthoc(artificial_archive, tmp_path):  # noqa: F811
    history, sealed = artificial_archive
    tables = extended_tables(history, ROOT, sealed)
    assert len(tables['long_lag_decay']) == len(EXPECTED_SINGLES) * len(LONG_LAGS)
    assert len(tables['composite_long_lag_decay']) == len(EXPECTED_COMPOSITES) * len(LONG_LAGS)
    assert len(tables['change_signal_ic']) == len(EXPECTED_SINGLES) * 3
    output = tmp_path / 'extended'
    output.mkdir()
    report = write_extended_report(tables, output, tmp_path / 'report.md')
    assert 'does not alter the pre-registered Phase 3 conclusions' in report
    assert 'No Internal Holdout or Final-180 data were used.' in report and 'p-value' in report


@pytest.mark.parametrize('outside', [7860, 8868])
def test_extended_rejects_heldout_rows(artificial_archive, outside):  # noqa: F811
    history, sealed = artificial_archive
    extra = history.iloc[[-1]].copy()
    extra['date_id'] = outside
    with pytest.raises(ValueError, match='0–7859'):
        extended_tables(pd.concat([history, extra], ignore_index=True), ROOT, sealed)


def test_extended_never_calls_confirmatory_tools(artificial_archive, monkeypatch):  # noqa: F811
    import src.research.alpha_pipeline as alpha
    import src.statistics.multiple_testing as testing
    import src.statistics.robust_inference as inference
    for module, name in [(alpha, 'build_alpha_tables'), (testing, 'adjust_fdr'),
                         (inference, 'blockwise_spearman_hac'), (inference, 'blockwise_spearman_interval')]:
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail('confirmatory tool invoked'))
    history, sealed = artificial_archive
    extended_tables(history, ROOT, sealed)


def test_failed_data_verification_never_loads_prefix(tmp_path, artificial_archive, monkeypatch):  # noqa: F811
    _, sealed = artificial_archive
    events = []
    def verify(root, *, verify_data):
        events.append(verify_data)
        if verify_data:
            raise ValueError('artificial data identity failure')
        return {}
    monkeypatch.setattr(extended, 'verify_protocol_lock', verify)
    monkeypatch.setattr(extended, 'load_sealed_phase3', lambda root: sealed)
    monkeypatch.setattr(extended, 'assert_preserved', lambda *a: None)
    monkeypatch.setattr(extended, 'load_research_prefix', lambda *a: pytest.fail('verification failed'))
    monkeypatch.setattr(extended, 'git', lambda *a: b'synthetic')
    with pytest.raises(ValueError, match='identity failure'):
        extended.run_extended(tmp_path)
    assert events == [False, True]


def test_existing_outputs_are_never_overwritten(tmp_path):
    (tmp_path / extended.OUTPUT).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        extended.run_extended(tmp_path)


def test_cli_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(extended, 'run_extended', lambda *a: pytest.fail('missing opt-in'))
    with pytest.raises(SystemExit) as error:
        extended.main([])
    assert error.value.code == 2


def test_sealed_results_and_protocol_unchanged():
    before = protected_hashes(ROOT)
    assert verify_protocol_lock(ROOT, verify_data=False)['protocol_version'] == 'phase2.5-v2'
    assert protected_hashes(ROOT) == before
