"""Phase 6B holdout pipeline on artificial data only; no real holdout row."""
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.research import phase6_allocation as alloc
from src.research import phase6_holdout as ho
from src.research import phase6_holdout_run as run
from src.research.alpha_pipeline import load_alpha_design
from src.research.phase5_synthetic import make_history
from src.research.phase5_tasks import TaskStore
from src.utils.paths import project_root
from src.validation import protocol_v3_1, protocol_v4, protocol_v4_1

ROOT = project_root()
DESIGN = load_alpha_design(ROOT)
V31 = protocol_v3_1.load_config(ROOT)


def artificial_history(seed=211):
    h = make_history('linear', seed, rows=9048)
    h['forward_returns'] = 0.0003 + 0.01*h['forward_returns']
    h['risk_free_rate'] = 0.0001 + 0.00001*h['risk_free_rate'].abs()
    return h


def write_project(root, history):
    """Production-shaped artificial train.csv (with final-row sentinels) and split policy."""
    frame = history.astype(object)
    frame.loc[8868:, [c for c in frame if c != 'date_id']] = 'FINAL_SENTINEL'
    train = root/'data/raw/train.csv'
    train.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(train, index=False)
    policy = json.loads((ROOT/'configs/split_policy.json').read_text(encoding='utf-8'))
    policy['source_train_sha256'] = hashlib.sha256(train.read_bytes()).hexdigest()
    (root/'configs').mkdir(exist_ok=True)
    (root/'configs/split_policy.json').write_text(json.dumps(policy), encoding='utf-8')
    return train, policy


def test_loader_reads_exactly_rows_0_to_8867(tmp_path, monkeypatch):
    from src.validation.time_split import read_header
    train, policy = write_project(tmp_path, artificial_history())
    calls, original = [], pd.read_csv
    monkeypatch.setattr(pd, 'read_csv', lambda p, *a, **k: calls.append(k.get('nrows')) or original(p, *a, **k))
    frame = ho.load_holdout_history(train, {**policy, 'train_columns': read_header(train)})
    assert calls == [8868] and len(frame) == 8868 and not frame.isin(['FINAL_SENTINEL']).any().any()
    with pytest.raises(ValueError, match='final reservation'):
        ho.load_holdout_history(train, {**policy, 'train_columns': read_header(train), 'source_train_sha256': '0'*64})


def test_holdout_windows_bounds():
    class State:
        def transform(self, raw):
            return np.zeros((len(raw), 16))
    raw = pd.DataFrame(np.zeros((8868, 3)))
    assert ho.holdout_windows(raw, State(), np.arange(7860, 8868)).shape == (1008, 63, 16)
    for bad in ([7859], [8868]):
        with pytest.raises(ValueError):
            ho.holdout_windows(raw, State(), bad)


@pytest.fixture(scope='module')
def engine_parts(tmp_path_factory):
    history = artificial_history().iloc[:8868].reset_index(drop=True)
    store = TaskStore(tmp_path_factory.mktemp('store'), {'run_type': 'synthetic'})
    return history, ho.HoldoutEngine(history, V31, DESIGN, store, test_budget=1)


def test_engine_removes_purge_and_holdout_labels(engine_parts):
    history, engine = engine_parts
    assert np.isnan(engine.y[7859:]).all() and np.isfinite(engine.y[62:7859]).all()
    assert engine.references == (7353, 7605, 7858)


def test_neural_capsule_has_no_holdout_label(engine_parts):
    _, engine = engine_parts
    job = engine.job(7858, 'GRU', 11)
    assert job['forecast_x'].shape == (1008, 63, 16) and job['forecast_range'] == [7860, 8867]
    assert job['refit_range'] == [0, 7858] and len(job['refit_y']) == 7858-62+1
    inner = engine.job(7605, 'TCN', 11)
    assert inner['forecast_range'] == [7607, 7858] and inner['forecast_x'].shape == (252, 63, 16)


def fitted(history, store_dir):
    e = ho.HoldoutEngine(history, V31, DESIGN, TaskStore(store_dir, {'run_type': 'synthetic'}), test_budget=1)
    preds = (e.forecast('A', 7858), e.forecast('B', 7858), e.ridge(7858))
    ridge = e.ridges[7858]
    return preds, (e.scores['A', 7858]['selector'], e.scores['B', 7858]['selector'], ridge['alpha'],
                   ho.hash_object(e.fixed_ridge(7858)[str(ridge['alpha'])]['state']))


def test_fitted_state_ignores_holdout_targets(tmp_path):
    """Lagged targets are legal online X, so predictions may move; fitted state may not."""
    base = artificial_history().iloc[:8868].reset_index(drop=True)
    changed = base.copy()
    changed.loc[7859:, 'market_forward_excess_returns'] = 99.0
    (_, state_a), (_, state_b) = fitted(base, tmp_path/'a'), fitted(changed, tmp_path/'b')
    assert state_a == state_b


def test_last_holdout_target_never_reaches_any_forecast(tmp_path):
    base = artificial_history().iloc[:8868].reset_index(drop=True)
    changed = base.copy()
    changed.loc[8867, ['market_forward_excess_returns', 'forward_returns', 'risk_free_rate']] = 99.0
    (a, _), (b, _) = fitted(base, tmp_path/'a'), fitted(changed, tmp_path/'b')
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert len(a[0]) == 1008


def test_forecast_metrics_confirmation_row():
    rng = np.random.default_rng(1)
    dates = np.arange(7860, 8868)
    y = pd.Series(rng.normal(size=len(dates)), index=dates)
    pred = pd.DataFrame({'date_id': np.tile(dates, 2), 'pipeline': np.repeat(['A', 'B'], len(dates)),
                         'prediction': np.r_[y.to_numpy()+rng.normal(scale=2, size=len(dates)), rng.normal(size=len(dates))]})
    m = ho.forecast_metrics(pred, y).set_index('pipeline')
    assert m.loc['A', 'blockwise_rank_ic'] > .2 and m.loc['A', 'one_sided_p'] < .001 and m.loc['A', 'confirmed_at_0.05']
    assert m.loc['A', 'pairs'] == 1008 and all(f'block_{b}_ic' in m for b in range(1, 5))
    assert pd.isna(m.loc['B', 'confirmed_at_0.05'])


def test_paired_difference_identical_positions_is_zero():
    rng = np.random.default_rng(2)
    r, f, w = rng.normal(.0003, .01, 1008), np.full(1008, .0001), np.clip(1+rng.normal(scale=.2, size=1008), 0, 2)
    d = ho.paired_difference_interval(r, f, w, w)
    assert d['estimate'] == 0 and d['ci_lower'] == 0 and d['ci_upper'] == 0 and d['valid_repetitions'] == 2000


def test_allocation_uses_frozen_standardization_and_adds_volatility_only():
    history = artificial_history().iloc[:8868].reset_index(drop=True)
    dates = np.arange(7860, 8868)
    rng = np.random.default_rng(3)
    preds = pd.DataFrame({'date_id': np.tile(dates, 7), 'pipeline': np.repeat(ho.PIPELINES, len(dates)),
                          'prediction': rng.normal(scale=.2, size=7*len(dates))})
    research = pd.read_csv(ROOT/run.RESEARCH_PREDICTIONS)
    v4, v41 = protocol_v4.load_config(ROOT), protocol_v4_1.load_config(ROOT)
    result = ho.allocation_evaluation(preds, history, v4, v41, research)
    o = result['overall'].set_index('candidate')
    assert 'volatility_only' in o.index and o.loc['volscaled_k0.10', 'role'] == 'primary'
    z = (preds.query("pipeline == 'A'").prediction.to_numpy() - v41['phase6_selection']['mu_A'])/v41['phase6_selection']['sigma_A']
    np.testing.assert_allclose(result['frame']['z'], z)
    np.testing.assert_allclose(result['positions']['volatility_only'], result['frame']['c'].clip(0, 2))
    assert set(result['paired']) == {'volscaled_k0.10 - volatility_only', 'volscaled_k0.10 - constant_1'}
    assert set(result['secondary_forecast_allocations'].pipeline) == {'B', 'C', 'D', 'Ridge', 'GRU', 'TCN'}


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    ident = {'run_type': 'holdout', 'git_commit': 'abc', 'pipeline_tag': run.TAG}
    monkeypatch.setattr(run, 'identity', lambda root: dict(ident))
    answers = {('status', '--porcelain', '--untracked-files=all'): '', ('cat-file', '-t', f'refs/tags/{run.TAG}'): 'tag',
               ('rev-parse', f'{run.TAG}^{{commit}}'): 'abc'}
    monkeypatch.setattr(run, 'git', lambda root, *a: answers[a])
    return tmp_path, answers, ident


@pytest.mark.parametrize('key,value,match', [
    (('status', '--porcelain', '--untracked-files=all'), ' M x', 'Dirty'),
    (('cat-file', '-t', f'refs/tags/{run.TAG}'), 'commit', 'annotated'),
    (('rev-parse', f'{run.TAG}^{{commit}}'), 'other', 'differs')])
def test_preflight_refusals(guarded, key, value, match):
    root, answers, _ = guarded
    answers[key] = value
    with pytest.raises(RuntimeError, match=match):
        run.preflight(root)


def test_completed_or_foreign_state_refuses(guarded):
    root, _, ident = guarded
    run.write_state(root, ident, 'completed') if (root/'artifacts').mkdir() is None else None
    with pytest.raises(RuntimeError, match='already been evaluated'):
        run.preflight(root)
    run.write_state(root, {**ident, 'git_commit': 'zzz'}, 'forecasting')
    with pytest.raises(RuntimeError, match='different identity'):
        run.preflight(root)


def end_to_end_project(root, monkeypatch):
    history = artificial_history()
    write_project(root, history)
    (root/'reports/phase5_real_run').mkdir(parents=True)
    shutil.copyfile(ROOT/run.RESEARCH_PREDICTIONS, root/run.RESEARCH_PREDICTIONS)
    import src.validation.protocol_freeze as freeze
    import src.research.alpha_pipeline as alpha
    monkeypatch.setattr(freeze, 'verify_protocol_lock', lambda *a, **k: {})
    monkeypatch.setattr(alpha, 'load_alpha_design', lambda r: DESIGN)
    monkeypatch.setattr(protocol_v3_1, 'load_config', lambda r: V31)
    v4, v41 = protocol_v4.load_config(ROOT), protocol_v4_1.load_config(ROOT)
    monkeypatch.setattr(protocol_v4, 'load_config', lambda r: v4)
    monkeypatch.setattr(protocol_v4_1, 'load_config', lambda r: v41)
    return history


def test_end_to_end_seals_forecasts_before_targets_and_never_reads_final(guarded, monkeypatch):
    root, _, _ = guarded
    end_to_end_project(root, monkeypatch)
    calls, original = [], pd.read_csv
    monkeypatch.setattr(pd, 'read_csv', lambda p, *a, **k: calls.append((str(p), k.get('nrows'))) or original(p, *a, **k))
    attach = []
    real_metrics = ho.forecast_metrics
    def spy(forecasts, target, **k):
        attach.append(json.loads((root/run.STATE).read_text())['state'])
        return real_metrics(forecasts, target, **k)
    monkeypatch.setattr(ho, 'forecast_metrics', spy)
    result = run.run_holdout(root, workers=1, test_budget=1, store_identity={'run_type': 'synthetic'})
    assert attach == ['forecasts_sealed']
    assert [n for p, n in calls if p.endswith('train.csv')] == [8868]
    state = json.loads((root/run.STATE).read_text())
    assert state['state'] == 'completed'
    assert set(result['metrics'].pipeline) == set(ho.PIPELINES)
    files = set(result['manifest']['output_sha256'])
    assert {'holdout_forecasts_sealed.csv', 'holdout_forecast_metrics.csv', 'holdout_allocation_scores.csv',
            'holdout_summary.json'} <= files
    with pytest.raises(RuntimeError, match='already been evaluated'):
        run.run_holdout(root, workers=1, test_budget=1, store_identity={'run_type': 'synthetic'})


def test_post_seal_failure_replays_evaluation_without_refitting(guarded, monkeypatch):
    root, _, ident = guarded
    end_to_end_project(root, monkeypatch)
    real_eval = ho.allocation_evaluation
    monkeypatch.setattr(ho, 'allocation_evaluation', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('injected')))
    with pytest.raises(RuntimeError, match='injected'):
        run.run_holdout(root, workers=1, test_budget=1, store_identity={'run_type': 'synthetic'})
    assert json.loads((root/run.STATE).read_text())['state'] == 'failed_post_forecast'
    monkeypatch.setattr(ho, 'allocation_evaluation', real_eval)
    monkeypatch.setattr(ho, 'HoldoutEngine', lambda *a, **k: pytest.fail('must not refit after sealing'))
    run.run_holdout(root, workers=1, test_budget=1, store_identity={'run_type': 'synthetic'})
    assert json.loads((root/run.STATE).read_text())['state'] == 'completed'
    sealed = root/run.SEALED
    assert hashlib.sha256(sealed.read_bytes()).hexdigest() == json.loads((root/run.STATE).read_text())['sealed_sha256']
