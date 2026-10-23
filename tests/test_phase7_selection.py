"""Protocol v5 DynRidge and competition-track selection on artificial data only."""
import json

import numpy as np
import pandas as pd
import pytest

from src.models import dynamic_ridge as dr
from src.research import phase7_selection as sel
from src.research.alpha_pipeline import load_alpha_design
from src.utils.paths import project_root
from src.validation import protocol_v3_1, protocol_v4, protocol_v5
from src.validation.protocol_v3 import inner_splits

ROOT = project_root()
V31 = protocol_v3_1.load_config(ROOT)
V4, V5 = protocol_v4.load_config(ROOT), protocol_v5.load_config(ROOT)
ORDER = protocol_v5.candidate_order()
POLICY = V31['availability']


def panel(seed=1, rows=3000, features=6, beta=0.3):
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.normal(size=(rows, features)), columns=[f'X{j}' for j in range(features)])
    raw.iloc[:400, 0] = np.nan                              # late launch
    y = beta*raw['X1'].to_numpy() - beta*raw['X2'].to_numpy() + rng.normal(size=rows)
    return raw, y


def test_registered_constants_match_protocol():
    d = V5['dynridge']
    assert list(dr.ALPHAS) == d['alphas'] and dr.HALF_LIFE == d['sample_weight']['half_life_rows']
    assert [-dr.CLIP, dr.CLIP] == d['scaler']['clip']


def test_inner_blocks_match_protocol_inner_splits():
    for end in (5086, 6346, 7858):
        frozen = [(s.training_end, s.validation_start, s.validation_end) for s in inner_splits(end, V31)]
        assert list(dr.inner_blocks(end)) == frozen
    assert dr.inner_blocks(8866) == ((8361, 8363, 8614), (8613, 8615, 8866))


def test_weighted_solution_matches_augmented_least_squares():
    raw, y = panel()
    end = 2500
    state = dr.fit(raw, y, end, POLICY)[10.0]
    rows = np.flatnonzero(np.isfinite(y[:end+1]))
    z = dr.scaled(raw.iloc[rows], state)
    t = (y[rows] - state.target_mean)/state.target_scale
    w = 0.5**((end - rows)/504); w = w/w.sum()
    a = np.column_stack([np.ones(len(rows)), z])*np.sqrt(w)[:, None]
    a = np.vstack([a, np.column_stack([np.zeros(z.shape[1]), np.sqrt(10.0)*np.eye(z.shape[1])])])
    b = np.r_[t*np.sqrt(w), np.zeros(z.shape[1])]
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose([state.intercept, *state.coef], solution, atol=1e-10)
    assert state.coef[1] > 0 > state.coef[2]


def test_fit_reads_only_its_prefix_and_forecast_only_row_t():
    raw, y = panel()
    end = 2000
    a = dr.fit(raw, y, end, POLICY)
    changed, cy = raw.copy(), y.copy()
    changed.iloc[end+1:] = 99.0
    cy[end+1:] = -99.0
    b = dr.fit(changed, cy, end, POLICY)
    for alpha in dr.ALPHAS:
        assert a[alpha].names == b[alpha].names
        np.testing.assert_array_equal(a[alpha].coef, b[alpha].coef)
    model, moved = dr.DynRidge(raw, y, POLICY), raw.copy()
    moved.iloc[2101:] = 50.0
    other = dr.DynRidge(moved, y, POLICY)
    np.testing.assert_array_equal(model.forecast(end, 2002, 2101), other.forecast(end, 2002, 2101))


def test_alpha_choice_and_signal_recovery():
    raw, y = panel(beta=0.5)
    model = dr.DynRidge(raw, y, POLICY)
    choice = model.choose(2600)
    assert choice['alpha'] in dr.ALPHAS and len(choice['inner_ics']) == 4
    pred = model.forecast(2600, 2602, 2854)
    assert np.corrcoef(pred, y[2602:2854])[0, 1] > .3
    noise = dr.DynRidge(raw, np.full(len(y), 1.0), POLICY)  # constant target: no fit, no forecast
    assert np.isnan(noise.forecast(2600, 2602, 2700)).all()


def test_alpha_ties_choose_larger(monkeypatch):
    raw, y = panel()
    model = dr.DynRidge(raw, y, POLICY)
    monkeypatch.setattr(dr, 'corr', lambda a, b, rank: 0.1)
    assert model.choose(2600)['alpha'] == 100.0


# ---------------------------------------------------------------- selection rule

def table(scores: dict, wins_vs_const: dict, wins_vs_vol: dict | None = None):
    """Artificial overall and 15-block scores; constant_1 scores 0 in every block, vol 0.5."""
    overall = pd.DataFrame({'candidate': list(scores), 'official_score': list(scores.values())})
    rows = []
    for b in range(1, 16):
        rows.append({'candidate': 'constant_1', 'block': b, 'official_score': 0.0})
        rows.append({'candidate': 'volatility_only', 'block': b,
                     'official_score': 0.5 if b <= wins_vs_const.get('volatility_only', 0) else -0.5})
        for cid in ORDER:
            n = (wins_vs_vol or {}).get(cid, wins_vs_const.get(cid, 0))
            rows.append({'candidate': cid, 'block': b, 'official_score': 1.0 if b <= n else -1.0})
    return overall, pd.DataFrame(rows)


def scores(base=0.4, vol=0.45, **named):
    out = {'constant_1': base, 'volatility_only': vol, **{c: 0.30 for c in ORDER}}
    out.update({c.replace('__', '/'): v for c, v in named.items()})
    return out


def test_default_is_volatility_only_only_when_it_clears_both_bars():
    pick = sel.select(*table(scores(), {'volatility_only': 9}), V5['selection'], ORDER)
    assert pick['default'] == 'volatility_only' and pick['selected'] == 'volatility_only'
    pick = sel.select(*table(scores(), {'volatility_only': 7}), V5['selection'], ORDER)
    assert pick['default'] == 'constant_1' and pick['selected'] == 'constant_1'
    pick = sel.select(*table(scores(vol=0.39), {'volatility_only': 12}), V5['selection'], ORDER)
    assert pick['default'] == 'constant_1'


def test_forecast_candidate_must_beat_default_and_simplest_near_best_wins():
    s = scores(**{'volscaled_k0.20__A': 0.60, 'volscaled_k0.10__C': 0.58, 'volscaled_k0.10__D': 0.70})
    wins = {'volatility_only': 9, 'volscaled_k0.20/A': 10, 'volscaled_k0.10/C': 10, 'volscaled_k0.10/D': 7}
    pick = sel.select(*table(s, wins), V5['selection'], ORDER)
    # D has the best score but only 7/15 block wins; 0.58 and 0.60 are within 0.05 of the best admissible 0.60
    assert pick['admissible'] == ['volscaled_k0.10/C', 'volscaled_k0.20/A']
    assert pick['selected'] == 'volscaled_k0.10/C'


def test_margin_is_strict():
    wins = {'volatility_only': 9, 'volscaled_k0.10/A': 9, 'volscaled_k0.10/Ridge': 9}
    out = sel.select(*table(scores(**{'volscaled_k0.10__A': 0.549, 'volscaled_k0.10__Ridge': 0.60}), wins),
                     V5['selection'], ORDER)
    assert out['selected'] == 'volscaled_k0.10/Ridge'
    inside = sel.select(*table(scores(**{'volscaled_k0.10__A': 0.551, 'volscaled_k0.10__Ridge': 0.60}), wins),
                        V5['selection'], ORDER)
    assert inside['selected'] == 'volscaled_k0.10/A'


# ---------------------------------------------------------------- assembly and outputs

def test_assemble_joins_research_and_holdout_without_overlap():
    research = pd.DataFrame({'date_id': np.arange(5088, 7860), 'pipeline': 'A', 'prediction': 1.0})
    holdout = pd.DataFrame({'date_id': np.arange(7860, 8868), 'pipeline': 'A', 'prediction': 2.0})
    dyn = pd.DataFrame({'date_id': np.arange(5088, 8868), 'prediction': 3.0})
    wide = sel.assemble(research, holdout, dyn, ['A', 'DynRidge'])
    assert wide.loc[7859, 'A'] == 1.0 and wide.loc[7860, 'A'] == 2.0 and len(wide) == 3780
    with pytest.raises(ValueError, match='exactly once'):
        sel.assemble(research, holdout.iloc[1:], dyn, ['A'])


def test_blocks_are_research_then_holdout():
    assert sel.oos_blocks([5088, 7859, 7860, 8867]).tolist() == [1, 11, 12, 15]


def test_synthetic_signal_is_selected_and_noise_is_not_inflated():
    strong = sel.build_selection(*sel.synthetic_inputs(seed=3, strength=0.25), V4, V5)
    assert '/' in strong['selection']['selected']
    assert strong['positions'].stack().between(0, 2).all() and len(strong['positions']) == 3780
    noise = sel.build_selection(*sel.synthetic_inputs(seed=3, strength=0.0), V4, V5)
    o = noise['overall'].set_index('candidate')['official_score']
    assert (o.drop(['constant_1', 'volatility_only']) - o['constant_1']).max() < 0.3


def test_synthetic_outputs_contain_no_daily_returns(tmp_path):
    sel.run_synthetic(tmp_path/'out', seed=5, strength=0.2)
    names = sorted(p.name for p in (tmp_path/'out').iterdir())
    assert names == ['v5_block_scores.csv', 'v5_candidate_scores.csv', 'v5_daily_positions.csv', 'v5_dynridge_audit.json',
                     'v5_dynridge_forecasts.csv', 'v5_run_manifest.json', 'v5_selection.json']
    daily = pd.read_csv(tmp_path/'out/v5_daily_positions.csv')
    assert not {'forward_returns', 'risk_free_rate', 'market_forward_excess_returns'} & set(daily)
    with pytest.raises(ValueError, match='outside'):
        sel.run_synthetic(ROOT/'reports/phase7_selection_synthetic', seed=5, strength=0.2)


def test_dynridge_oos_schedule_and_holdout_label_isolation():
    from tests.test_phase6_holdout import artificial_history
    design = load_alpha_design(ROOT)
    base = artificial_history().iloc[:8868].reset_index(drop=True)
    frame, audit = sel.dynridge_oos(base, V31, design)
    assert frame['date_id'].tolist() == list(range(5088, 8868)) and len(audit) == 12
    assert frame.groupby('reference_end').size().tolist() == [252]*11 + [1008]
    assert frame.loc[frame.date_id == 7860, 'reference_end'].item() == 7858
    changed = base.copy()
    changed.loc[7859:, 'market_forward_excess_returns'] = 99.0    # holdout labels; lagged X moves legally
    _, other = sel.dynridge_oos(changed, V31, design)
    assert json.dumps(audit, default=float) == json.dumps(other, default=float)


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    monkeypatch.setattr(sel.protocol_v5, 'verify_lock', lambda root: {'evidence_sha256': {}})
    monkeypatch.setattr(sel.protocol_v4, 'text_digest', lambda path: 'x')
    answers = {('rev-parse', 'HEAD'): 'abc', ('status', '--porcelain', '--untracked-files=all'): '',
               ('cat-file', '-t', f'refs/tags/{sel.TAG}'): 'tag', ('rev-parse', f'{sel.TAG}^{{commit}}'): 'abc'}
    monkeypatch.setattr(sel, 'git', lambda root, *args: answers[args])
    return tmp_path, answers


@pytest.mark.parametrize('key,value,match', [
    (('status', '--porcelain', '--untracked-files=all'), ' M x.py', 'Dirty'),
    (('cat-file', '-t', f'refs/tags/{sel.TAG}'), 'commit', 'annotated'),
    (('rev-parse', f'{sel.TAG}^{{commit}}'), 'other', 'differs')])
def test_preflight_refusals(guarded, key, value, match):
    root, answers = guarded
    answers[key] = value
    with pytest.raises(RuntimeError, match=match):
        sel.preflight(root)


def test_failed_run_consumes_reservation(guarded, monkeypatch):
    root, _ = guarded
    assert sel.preflight(root)['final_simulation_read'] is False
    import src.validation.protocol_freeze as freeze
    monkeypatch.setattr(freeze, 'verify_protocol_lock', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('injected')))
    with pytest.raises(RuntimeError, match='injected'):
        sel.run_selection(root)
    assert json.loads((root/sel.STATE).read_text())['state'] == 'failed_or_interrupted'
    with pytest.raises(RuntimeError, match='rerun'):
        sel.preflight(root)


def test_real_entry_on_committed_evidence_never_reads_final_rows(guarded, monkeypatch):
    """Integration: unpatched bounded loader, the committed forecast evidence, artificial market rows."""
    import shutil
    from tests.test_phase6_holdout import artificial_history, write_project
    root, _ = guarded
    write_project(root, artificial_history())
    for name in protocol_v5.EVIDENCE:
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, root/name)
    import src.validation.protocol_freeze as freeze
    import src.research.alpha_pipeline as alpha
    monkeypatch.setattr(freeze, 'verify_protocol_lock', lambda *a, **k: {})
    monkeypatch.setattr(alpha, 'load_alpha_design', lambda r: load_alpha_design(ROOT))
    monkeypatch.setattr(protocol_v3_1, 'load_config', lambda r: V31)
    monkeypatch.setattr(protocol_v4, 'load_config', lambda r: V4)
    monkeypatch.setattr(protocol_v5, 'load_config', lambda r: V5)
    calls, original = [], pd.read_csv
    monkeypatch.setattr(pd, 'read_csv', lambda p, *a, **k: calls.append((str(p), k.get('nrows'))) or original(p, *a, **k))
    result = sel.run_selection(root)
    assert [n for p, n in calls if p.endswith('train.csv')] == [8868]
    state = json.loads((root/sel.STATE).read_text())
    assert state['state'] == 'completed' and state['selected'] == result['selection']['selected']
    assert set(result['standardization']) == set(V5['forecasters'])
    assert (root/sel.OUTPUT/'v5_run_manifest.json').exists()
