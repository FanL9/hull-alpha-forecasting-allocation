"""Phase 6A allocation pipeline on artificial data only."""
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.kaggle_metric import evaluate_allocations
from src.research import phase6
from src.research import phase6_allocation as alloc
from src.utils.paths import project_root
from src.validation import protocol_v4

ROOT = project_root()
CONFIG = protocol_v4.load_config(ROOT)


def test_registered_inventory_matches_protocol():
    assert alloc.CANDIDATES == tuple(CONFIG['candidates']['simplicity_order'])
    assert len(alloc.CANDIDATES) == 10 and alloc.BASELINE == 'constant_1'


def test_standardization_uses_forecasts_only_and_ddof1():
    p = pd.Series([0.1, -0.2, 0.3, np.nan, 0.0])
    state = alloc.standardization(p)
    assert state['mu'] == pytest.approx(0.05) and state['sigma'] == pytest.approx(np.std([0.1, -0.2, 0.3, 0.0], ddof=1))
    with pytest.raises(ValueError):
        alloc.standardization([1.0, 1.0])


def test_positions_mapping_clip_and_missing_forecast():
    z = pd.Series([0.0, 2.0, -2.0, 20.0, -20.0, np.nan])
    c = pd.Series([1.5, 0.5, 1.0, 1.0, 1.0, 1.2])
    lin = alloc.positions('linear_k0.20', z)
    np.testing.assert_allclose(lin, [1.0, 1.4, 0.6, 2.0, 0.0, 1.0])
    vol = alloc.positions('volscaled_k0.20', z, c)
    np.testing.assert_allclose(vol, [1.5, 0.7, 0.6, 2.0, 0.0, 1.0])
    assert (alloc.positions('constant_1', z) == 1.0).all()
    with pytest.raises(ValueError):
        alloc.positions('linear_k0.30', z)


def test_volatility_ratio_is_causal_and_clipped():
    rng = np.random.default_rng(1)
    lagged = pd.Series(rng.normal(scale=.01, size=1000))
    ratio = alloc.volatility_ratio(lagged, CONFIG['volatility_state'])
    changed = lagged.copy()
    changed.iloc[600:] *= 50
    np.testing.assert_allclose(alloc.volatility_ratio(changed, CONFIG['volatility_state']).iloc[:600], ratio.iloc[:600])
    assert ratio.between(.5, 1.5).all() and (ratio.iloc[:14] == 1.0).all()


def test_zero_cost_score_equals_official_metric():
    rng = np.random.default_rng(2)
    r, f = rng.normal(.0003, .01, 500), np.full(500, .0001)
    w = np.clip(1 + .3*rng.normal(size=500), 0, 2)
    assert alloc.cost_adjusted_score(r, f, w, 0) == pytest.approx(evaluate_allocations(r, f, w).score, rel=1e-12)
    assert alloc.cost_adjusted_score(r, f, w, 5) < alloc.cost_adjusted_score(r, f, w, 0)
    assert alloc.cost_adjusted_score(r, f, np.ones(500), 5) == pytest.approx(alloc.official_score(r, f, np.ones(500)))


def table(scores, blocks_beating):
    """Artificial score tables: candidate overall scores and per-block win counts."""
    overall = pd.DataFrame({'candidate': ['constant_1', *alloc.CANDIDATES],
                            'official_score': [0.40, *scores]})
    rows = []
    for b in range(1, 12):
        rows.append({'candidate': 'constant_1', 'block': b, 'official_score': 0.0})
        for cid, wins in zip(alloc.CANDIDATES, blocks_beating):
            rows.append({'candidate': cid, 'block': b, 'official_score': 1.0 if b <= wins else -1.0})
    return overall, pd.DataFrame(rows)


def test_selection_near_best_prefers_simplest():
    scores = [0.42, 0.465, 0.50, 0.51, 0.52, 0.48, 0.49, 0.50, 0.40, 0.60]
    wins = [8, 8, 8, 8, 8, 8, 8, 8, 8, 5]
    choice = alloc.select(*table(scores, wins), CONFIG['selection'])
    # volscaled_k0.75 (0.60) fails 6/11, volscaled_k0.50 only ties the baseline; best admissible 0.52
    assert choice['best_score'] == pytest.approx(0.52)
    assert choice['near_best'] == ['linear_k0.35', 'linear_k0.50', 'linear_k0.75',
                                   'volscaled_k0.10', 'volscaled_k0.20', 'volscaled_k0.35']
    assert choice['selected'] == 'linear_k0.35'
    assert 'volscaled_k0.75' not in choice['admissible'] and 'volscaled_k0.50' not in choice['admissible']


def test_selection_margin_is_strict_and_fallback_is_constant():
    outside = alloc.select(*table([0.449, 0.50] + [0.30]*8, [8]*10), CONFIG['selection'])
    assert outside['selected'] == 'linear_k0.20'  # 0.051 behind the best: outside the margin
    inside = alloc.select(*table([0.451, 0.50] + [0.30]*8, [8]*10), CONFIG['selection'])
    assert inside['selected'] == 'linear_k0.10'  # 0.049 behind: simplest near-best wins
    none = alloc.select(*table([0.39]*10, [11]*10), CONFIG['selection'])
    assert none['selected'] == 'constant_1' and none['admissible'] == []


def test_synthetic_signal_beats_constant_and_noise_is_not_inflated(tmp_path):
    predictions, history = phase6.synthetic_inputs(seed=3, strength=0.25)
    strong = alloc.build_selection(predictions, history, CONFIG)
    assert strong['selection']['selected'] != 'constant_1'
    assert len(strong['positions']) == 2772 and strong['positions'].stack().between(0, 2).all()
    assert abs(strong['frame']['z'].mean()) < 1e-12 and strong['frame']['z'].std() == pytest.approx(1)
    noise = alloc.build_selection(*phase6.synthetic_inputs(seed=3, strength=0.0), CONFIG)
    o = noise['overall'].set_index('candidate')['official_score']
    assert (o.drop('constant_1') - o['constant_1']).max() < 0.25


def test_build_selection_rejects_wrong_ranges():
    predictions, history = phase6.synthetic_inputs()
    with pytest.raises(ValueError, match='calibration range'):
        alloc.build_selection(predictions.iloc[1:], history, CONFIG)
    longer = pd.concat([history, history.tail(1).assign(date_id=7860)], ignore_index=True)
    with pytest.raises(ValueError, match='7859'):
        alloc.build_selection(predictions, longer, CONFIG)


def test_selection_ignores_holdout_and_final_values():
    predictions, history = phase6.synthetic_inputs(seed=4)
    first = alloc.build_selection(predictions, history, CONFIG)
    altered = predictions.copy()
    # Forecast rows outside the calibration range are filtered out before use.
    extra = pd.DataFrame({'date_id': [7860], 'fold_id': [12], 'pipeline': ['B'], 'prediction': [99.0]})
    second = alloc.build_selection(pd.concat([altered, extra], ignore_index=True), history, CONFIG)
    assert first['selection'] == second['selection']


def test_synthetic_outputs_and_manifest(tmp_path):
    result = phase6.run_synthetic(tmp_path/'out', seed=5, strength=0.2)
    files = sorted(p.name for p in (tmp_path/'out').iterdir())
    assert files == ['phase6_block_scores.csv', 'phase6_candidate_scores.csv', 'phase6_daily_positions.csv',
                     'phase6_run_manifest.json', 'phase6_selection.json']
    manifest = json.loads((tmp_path/'out/phase6_run_manifest.json').read_text())
    assert manifest['run_type'] == 'synthetic' and len(manifest['output_sha256']) == 4
    with pytest.raises(FileExistsError):
        phase6.run_synthetic(tmp_path/'out', seed=5, strength=0.2)
    with pytest.raises(ValueError, match='outside'):
        phase6.run_synthetic(ROOT/'artifacts/phase6_selection', seed=5, strength=0.2)
    assert result['selection']['selected'] in ('constant_1', *alloc.CANDIDATES)


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    monkeypatch.setattr(phase6.protocol_v4, 'verify_lock', lambda root: {'sealed_phase5_sha256': {}})
    monkeypatch.setattr(phase6.protocol_v4, 'text_digest', lambda path: 'x')
    answers = {('rev-parse', 'HEAD'): 'abc', ('status', '--porcelain', '--untracked-files=all'): '',
               ('cat-file', '-t', f'refs/tags/{phase6.TAG}'): 'tag', ('rev-parse', f'{phase6.TAG}^{{commit}}'): 'abc'}
    monkeypatch.setattr(phase6, 'git', lambda root, *args: answers[args])
    return tmp_path, answers


def test_preflight_passes_without_writes(guarded):
    root, _ = guarded
    checked = phase6.preflight(root)
    assert checked['preflight'] == 'passed' and not checked['internal_holdout_read']
    assert list(root.iterdir()) == []


@pytest.mark.parametrize('key,value,match', [
    (('status', '--porcelain', '--untracked-files=all'), ' M x.py', 'Dirty'),
    (('cat-file', '-t', f'refs/tags/{phase6.TAG}'), 'commit', 'annotated'),
    (('rev-parse', f'{phase6.TAG}^{{commit}}'), 'other', 'differs')])
def test_preflight_refusals(guarded, key, value, match):
    root, answers = guarded
    answers[key] = value
    with pytest.raises(RuntimeError, match=match):
        phase6.preflight(root)


def test_existing_state_blocks_rerun(guarded):
    root, _ = guarded
    (root/'artifacts').mkdir()
    (root/phase6.STATE).write_text('{}')
    with pytest.raises(RuntimeError, match='rerun'):
        phase6.preflight(root)


def test_failed_run_consumes_reservation_before_any_loading(guarded, monkeypatch):
    root, _ = guarded
    import src.validation.protocol_freeze as freeze
    def fail(*a, **k):
        raise RuntimeError('injected data-identity failure')
    monkeypatch.setattr(freeze, 'verify_protocol_lock', fail)
    import src.research.phase4_execution as p4
    monkeypatch.setattr(p4, 'load_research_history', lambda *a: pytest.fail('must not load'))
    with pytest.raises(RuntimeError, match='injected'):
        phase6.run_selection(root)
    state = json.loads((root/phase6.STATE).read_text())
    assert state['state'] == 'failed_or_interrupted' and state['error_type'] == 'RuntimeError'
    with pytest.raises(RuntimeError, match='rerun'):
        phase6.preflight(root)


def test_real_entry_loads_research_prefix_only(guarded, monkeypatch):
    """Integration: unpatched bounded loader on a production-shaped artificial file."""
    from tests.test_phase4_execution import research_shaped_project
    root, _ = guarded
    design, _, _ = research_shaped_project(root)
    import src.validation.protocol_freeze as freeze
    monkeypatch.setattr(freeze, 'verify_protocol_lock', lambda *a, **k: {})
    import src.research.alpha_pipeline as alpha
    monkeypatch.setattr(alpha, 'load_alpha_design', lambda r: design)
    monkeypatch.setattr(phase6.protocol_v4, 'load_config', lambda r: CONFIG)
    predictions, _ = phase6.synthetic_inputs(seed=6)
    (root/'reports/phase5_real_run').mkdir(parents=True)
    predictions.to_csv(root/phase6.PREDICTIONS, index=False)
    calls, original = [], pd.read_csv
    def spy(path, *a, **k):
        calls.append((str(path).endswith('train.csv'), k.get('nrows')))
        return original(path, *a, **k)
    monkeypatch.setattr(pd, 'read_csv', spy)
    result = phase6.run_selection(root)
    assert (True, 7860) in calls and all(n == 7860 for is_train, n in calls if is_train)
    state = json.loads((root/phase6.STATE).read_text())
    assert state['state'] == 'completed' and state['selected'] == result['selection']['selected']
    assert (root/phase6.OUTPUT/'phase6_run_manifest.json').exists()
