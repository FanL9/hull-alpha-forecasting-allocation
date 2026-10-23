"""Protocol v5 refit engine and Final Simulation run on artificial data only."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.features import deep_inputs
from src.research import phase6_holdout as ho
from src.research import phase7_final as fin
from src.research import phase7_refit as refit
from src.research import phase7_selection as sel
from src.research.alpha_pipeline import load_alpha_design
from src.research.phase5_tasks import TaskStore
from src.utils.paths import project_root
from src.validation import protocol_v3_1, protocol_v4, protocol_v4_1
from tests.test_phase6_holdout import artificial_history, write_project

ROOT = project_root()
DESIGN = load_alpha_design(ROOT)
V31 = protocol_v3_1.load_config(ROOT)


def store(path):
    return TaskStore(path, {'run_type': 'synthetic'})


def test_copied_helpers_equal_frozen_helpers_inside_research():
    history = artificial_history().iloc[:7860].reset_index(drop=True)
    frozen = deep_inputs.fit_inputs(history, 6000, V31)
    copied = refit.fit_inputs_until(history, 6000, V31, 7859)
    assert frozen.names == copied.names and frozen.clusters == copied.clusters
    np.testing.assert_array_equal(frozen.means, copied.means)
    raw = refit.CandidateProvider(history, V31, 'A').raw
    dates = np.arange(6002, 6254)
    np.testing.assert_array_equal(deep_inputs.windows(raw, frozen, dates), refit.causal_windows(raw, frozen, dates, 7859))
    with pytest.raises(ValueError):
        refit.fit_inputs_until(history, 6000, V31, 5999)


def test_bounded_config_moves_only_the_reference_guard():
    moved = refit.bounded_config(V31, 8867)
    assert moved['research']['end'] == 8867 and V31['research']['end'] == 7859
    assert {k: v for k, v in moved.items() if k != 'research'} == {k: v for k, v in V31.items() if k != 'research'}


def test_refit_engine_reproduces_frozen_holdout_engine(tmp_path):
    history = artificial_history().iloc[:8868].reset_index(drop=True)
    old = ho.HoldoutEngine(history, V31, DESIGN, store(tmp_path/'old'), test_budget=1)
    new = refit.RefitEngine(history, V31, DESIGN, store(tmp_path/'new'), 7858, 7860, 8868, test_budget=1)
    assert new.references == old.references and new.purge == 7859
    np.testing.assert_array_equal(new.y, old.y)
    for name in ('A', 'B', 'Ridge'):
        np.testing.assert_array_equal(new.forecast(name, 7858), old.forecast(name, 7858))
    np.testing.assert_array_equal(new.inner('B', 7858), old.inner('B', 7858))
    a, b = new.job(7858, 'GRU', 11), old.job(7858, 'GRU', 11)
    for key in ('forecast_x', 'refit_x', 'refit_y', 'stop_x', 'validation_x'):
        np.testing.assert_array_equal(a[key], b[key])
    assert a['task'] == b['task'] and a['forecast_range'] == b['forecast_range']


@pytest.fixture(scope='module')
def final_history():
    return artificial_history()


def final_engine(history, path):
    return refit.RefitEngine(history, V31, DESIGN, store(path), refit.FINAL_FIT_END, refit.FINAL_START,
                             refit.FINAL_STOP, test_budget=1)


def test_final_engine_boundaries(final_history, tmp_path):
    e = final_engine(final_history, tmp_path)
    assert np.isnan(e.y[8867:]).all() and np.isfinite(e.y[7859:8867]).all()
    assert e.references == (8361, 8613, 8866)
    job = e.job(8866, 'TCN', 11)
    assert job['forecast_range'] == [8868, 9047] and job['forecast_x'].shape == (180, 63, 16)
    assert job['refit_range'] == [0, 8866] and len(job['refit_y']) == 8866-62+1


def test_final_forecasts_are_causal_and_fits_ignore_final_outcomes(final_history, tmp_path):
    names = ['A', 'Ridge', 'DynRidge']
    base, audit = final_engine(final_history, tmp_path/'a').refit_forecasts(names)
    assert len(base) == 540 and base.groupby('pipeline').size().tolist() == [180]*3
    later = final_history.copy()
    columns = [c for c in later if c != 'date_id']
    later.loc[8958:, columns] = later.loc[8958:, columns]*3 + 1      # every later X and outcome
    moved, _ = final_engine(later, tmp_path/'b').refit_forecasts(names)
    early = base['date_id'] < 8958
    np.testing.assert_array_equal(base.loc[early, 'prediction'], moved.loc[early, 'prediction'])
    assert not np.array_equal(base.loc[~early, 'prediction'], moved.loc[~early, 'prediction'])
    outcomes = final_history.copy()
    outcomes.loc[8867:, ['market_forward_excess_returns', 'forward_returns', 'risk_free_rate']] += 5.0
    _, other = final_engine(outcomes, tmp_path/'c').refit_forecasts(names)
    # Lagged outcomes are legal online X, so forecasts may move; no fitted state may.
    assert json.dumps(audit, sort_keys=True, default=str) == json.dumps(other, sort_keys=True, default=str)


def chosen(selected='volscaled_k0.10/DynRidge'):
    return {'selection': {'selected': selected},
            'standardization': {'DynRidge': {'mu': 0.0, 'sigma': 0.002}, 'C': {'mu': 0.0, 'sigma': 0.001}}}


def forecasts_frame(names, seed=4):
    rng = np.random.default_rng(seed)
    dates = np.arange(8868, 9048)
    return pd.DataFrame({'date_id': np.tile(dates, len(names)), 'pipeline': np.repeat(names, len(dates)),
                         'prediction': rng.normal(scale=.002, size=len(dates)*len(names))})


def test_evaluation_tracks_and_frozen_standardization(final_history):
    v4, v41 = protocol_v4.load_config(ROOT), protocol_v4_1.load_config(ROOT)
    f = forecasts_frame(['A', 'DynRidge'])
    out = fin.evaluation(f, final_history, 'market_forward_excess_returns', v4, v41, chosen())
    assert list(out['positions']) == ['constant_1', 'volatility_only', 'research_track', 'competition_track']
    a = f.query("pipeline == 'A'").prediction.to_numpy()
    z = (a - v41['phase6_selection']['mu_A'])/v41['phase6_selection']['sigma_A']
    np.testing.assert_allclose(out['positions']['research_track'], np.clip(out['frame']['c']*(1 + .1*z), 0, 2))
    assert out['tracks'] == {'research': 'volscaled_k0.10/A', 'competition': 'volscaled_k0.10/DynRidge'}
    free = fin.evaluation(forecasts_frame(['A']), final_history, 'market_forward_excess_returns', v4, v41,
                          chosen('volatility_only'))
    np.testing.assert_array_equal(free['positions']['competition_track'], free['positions']['volatility_only'])
    assert fin.track_forecasters('volatility_only') == ['A'] and fin.track_forecasters('volscaled_k0.20/D') == ['A', 'D']


# ---------------------------------------------------------------- guarded run

def publish_selection(root, selected='volscaled_k0.10/DynRidge'):
    out = root/sel.PUBLISHED
    out.mkdir(parents=True)
    body = json.dumps({'selection': {'selected': selected}, 'standardization': chosen(selected)['standardization']})
    (out/'v5_selection.json').write_text(body, encoding='utf-8')
    (out/'v5_run_manifest.json').write_text('{}', encoding='utf-8')
    (out/'phase7_selection_state.json').write_text(json.dumps({'state': 'completed', 'output_sha256': {
        'v5_selection.json': hashlib.sha256(body.encode()).hexdigest()}}), encoding='utf-8')


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    ident = {'run_type': 'final', 'git_commit': 'abc'}
    monkeypatch.setattr(fin, 'identity', lambda root: dict(ident))
    answers = {('status', '--porcelain', '--untracked-files=all'): '', ('cat-file', '-t', f'refs/tags/{sel.TAG}'): 'tag',
               ('rev-parse', f'{sel.TAG}^{{commit}}'): 'abc', ('merge-base', '--is-ancestor', 'abc', 'HEAD'): '',
               ('diff', '--name-only', 'abc', 'HEAD'): 'reports/phase7_selection/v5_selection.json'}
    monkeypatch.setattr(fin, 'git', lambda root, *a: answers[a])
    return tmp_path, answers, ident


@pytest.mark.parametrize('key,value,match', [
    (('status', '--porcelain', '--untracked-files=all'), ' M x', 'Dirty'),
    (('cat-file', '-t', f'refs/tags/{sel.TAG}'), 'commit', 'annotated'),
    (('diff', '--name-only', 'abc', 'HEAD'), 'reports/phase7_selection/a.json\nsrc/research/phase7_final.py', 'Only')])
def test_preflight_refusals(guarded, key, value, match):
    root, answers, _ = guarded
    answers[key] = value
    with pytest.raises(RuntimeError, match=match):
        fin.preflight(root)


def test_selection_tampering_is_refused(tmp_path):
    publish_selection(tmp_path)
    fin.sealed_selection(tmp_path)
    (tmp_path/sel.PUBLISHED/'v5_selection.json').write_text('{"selection": {"selected": "constant_1"}}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='differs'):
        fin.sealed_selection(tmp_path)


def test_end_to_end_seals_before_outcomes_and_refuses_rerun(guarded, monkeypatch):
    root, _, _ = guarded
    write_project(root, artificial_history())
    publish_selection(root)
    import src.validation.protocol_freeze as freeze
    import src.research.alpha_pipeline as alpha
    monkeypatch.setattr(freeze, 'verify_protocol_lock', lambda *a, **k: {})
    monkeypatch.setattr(alpha, 'load_alpha_design', lambda r: DESIGN)
    monkeypatch.setattr(protocol_v3_1, 'load_config', lambda r: V31)
    v4, v41 = protocol_v4.load_config(ROOT), protocol_v4_1.load_config(ROOT)
    monkeypatch.setattr(protocol_v4, 'load_config', lambda r: v4)
    monkeypatch.setattr(protocol_v4_1, 'load_config', lambda r: v41)
    # write_project fills final rows with sentinels; the final run must read real artificial values instead.
    history = artificial_history()
    history.to_csv(root/'data/raw/train.csv', index=False)
    policy = json.loads((root/'configs/split_policy.json').read_text())
    policy['source_train_sha256'] = hashlib.sha256((root/'data/raw/train.csv').read_bytes()).hexdigest()
    (root/'configs/split_policy.json').write_text(json.dumps(policy))
    attach, real = [], fin.evaluation
    monkeypatch.setattr(fin, 'evaluation', lambda *a, **k: attach.append(json.loads((root/fin.STATE).read_text())['state'])
                        or real(*a, **k))
    result = fin.run_final(root, workers=1, test_budget=1, store_identity={'run_type': 'synthetic'})
    assert attach == ['forecasts_sealed']
    sealed = pd.read_csv(root/fin.SEALED)
    assert set(sealed.pipeline) == {'A', 'DynRidge'} and sealed.date_id.min() == 8868 and len(sealed) == 360
    assert set(result['overall'].candidate) == {'constant_1', 'volatility_only', 'research_track', 'competition_track'}
    assert json.loads((root/fin.STATE).read_text())['state'] == 'completed'
    with pytest.raises(RuntimeError, match='already been evaluated'):
        fin.run_final(root, workers=1, test_budget=1, store_identity={'run_type': 'synthetic'})
