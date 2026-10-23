"""Protocol v5 competition-track selection on all 15 OOS blocks (5088-8867).

python -m src.research.phase7_selection --check                    # metadata only
python -m src.research.phase7_selection --synthetic --output DIR   # artificial data
python -m src.research.phase7_selection --run-selection            # one real run

The real run needs a clean checkout at the annotated pipeline tag, a verified
Protocol v5 lock and an unused canonical state. It parses exactly rows 0-8867;
the Final Simulation rows are never read. No output contains a daily return.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.dynamic_ridge import DynRidge
from src.research import phase6_allocation as alloc
from src.utils.paths import project_root, utc_timestamp
from src.validation import protocol_v4, protocol_v5

TAG = 'phase7-pipeline-v1'
STATE = 'artifacts/phase7_selection_state.json'
OUTPUT = 'artifacts/phase7_selection'
PUBLISHED = 'reports/phase7_selection'
OOS_START, OOS_END = 5088, 8867
HOLDOUT_FIT_END = 7858
FORECAST_FREE = ('constant_1', 'volatility_only')
CODE = ('src/models/dynamic_ridge.py', 'src/research/phase7_selection.py', 'src/research/phase7_refit.py',
        'src/research/phase7_final.py', 'src/research/phase6_allocation.py',
        'tests/test_phase7_selection.py', 'tests/test_phase7_final.py')


def oos_blocks(dates) -> np.ndarray:
    """Blocks 1-11 are the research outer folds, 12-15 the holdout blocks."""
    return (np.asarray(dates) - OOS_START)//252 + 1


# ---------------------------------------------------------------- forecasts

def dynridge_oos(history: pd.DataFrame, config: dict, design) -> tuple[pd.DataFrame, dict]:
    """DynRidge on the frozen schedule: each research fold end, then 7858 for the holdout."""
    from src.features.engineered_alpha import CandidateProvider
    from src.validation.walk_forward import make_folds
    if len(history) != OOS_END+1 or not np.array_equal(history['date_id'], np.arange(OOS_END+1)):
        raise ValueError('DynRidge OOS forecasts need exactly rows 0-8867.')
    raw = CandidateProvider(history, config, 'A').raw
    y = history[config['target']].to_numpy(float).copy()
    y[HOLDOUT_FIT_END+1:] = np.nan                     # holdout labels never enter any fit
    model = DynRidge(raw, y, config['availability'])
    spans = [(int(f.train_ids[-1]), int(f.validation_ids[0]), int(f.validation_ids[-1])+1)
             for f in make_folds(np.arange(design.validation.development_end+1), design.validation)]
    spans.append((HOLDOUT_FIT_END, HOLDOUT_FIT_END+2, OOS_END+1))
    rows, audit = [], {}
    for end, start, stop in spans:
        rows.append(pd.DataFrame({'date_id': np.arange(start, stop), 'reference_end': end,
                                  'prediction': model.forecast(end, start, stop)}))
        audit[str(end)] = model.audit(end)
    frame = pd.concat(rows, ignore_index=True)
    if frame['date_id'].tolist() != list(range(OOS_START, OOS_END+1)):
        raise ValueError('DynRidge forecasts must cover 5088-8867 exactly once.')
    return frame, audit


def assemble(research: pd.DataFrame, holdout: pd.DataFrame, dynridge: pd.DataFrame, forecasters) -> pd.DataFrame:
    """Wide OOS forecast table on 5088-8867; research and holdout pieces never overlap."""
    dates = np.arange(OOS_START, OOS_END+1)
    out = pd.DataFrame(index=pd.Index(dates, name='date_id'))
    for name in forecasters:
        if name == 'DynRidge':
            s = dynridge.set_index('date_id')['prediction']
        else:
            r = research.loc[research['pipeline'].eq(name) & research['date_id'].between(OOS_START, 7859)]
            h = holdout.loc[holdout['pipeline'].eq(name) & holdout['date_id'].between(7860, OOS_END)]
            s = pd.concat([r.set_index('date_id')['prediction'], h.set_index('date_id')['prediction']])
        if not s.index.is_unique or s.index.tolist() != dates.tolist():
            raise ValueError(f'{name} forecasts must cover 5088-8867 exactly once.')
        out[name] = s.astype(float)
    return out


# ---------------------------------------------------------------- selection

def positions(frame: pd.DataFrame, forecasts: pd.DataFrame, standards: dict, order) -> pd.DataFrame:
    pos = pd.DataFrame({'constant_1': 1.0, 'volatility_only': frame['c'].clip(0, 2)}, index=frame.index)
    for cid in order:
        mapping, name = cid.split('/')
        pos[cid] = alloc.positions(mapping, alloc.standardize(forecasts[name], standards[name]), frame['c'])
    return pos


def select(overall: pd.DataFrame, blocks: pd.DataFrame, rules: dict, order) -> dict:
    """Default (forecast-free) first, then admissible -> near-best -> simplicity order."""
    o = overall.set_index('candidate')['official_score']
    b = blocks.pivot(index='block', columns='candidate', values='official_score')
    need = rules['min_blocks']

    def wins(cid, ref):
        return int((b[cid] > b[ref]).sum())

    def beats(cid, ref):
        return bool(np.isfinite(o[cid]) and np.isfinite(o[ref]) and o[cid] > o[ref] and wins(cid, ref) >= need)

    default = 'volatility_only' if beats('volatility_only', 'constant_1') else 'constant_1'
    record = {'default': default, 'default_reason': {'volatility_only_score': float(o['volatility_only']),
              'constant_1_score': float(o['constant_1']), 'volatility_only_block_wins': wins('volatility_only', 'constant_1')},
              'block_wins_vs_default': {c: wins(c, default) for c in order}, 'margin': rules['near_best_margin'],
              'min_blocks': need}
    admissible = [c for c in order if beats(c, default)]
    if not admissible:
        return {**record, 'admissible': [], 'near_best': [], 'best_score': None, 'selected': default,
                'reason': 'no_admissible_forecast_candidate'}
    best = max(o[c] for c in admissible)
    near = [c for c in admissible if best - o[c] < rules['near_best_margin']]
    return {**record, 'admissible': admissible, 'near_best': near, 'best_score': float(best),
            'selected': next(c for c in order if c in near), 'reason': 'first_near_best_in_simplicity_order'}


def build_selection(forecasts: pd.DataFrame, history: pd.DataFrame, v4: dict, v5: dict) -> dict:
    order = protocol_v5.candidate_order()
    dates = np.arange(OOS_START, OOS_END+1)
    if not forecasts.index.equals(pd.Index(dates, name='date_id')) or list(forecasts) != v5['forecasters']:
        raise ValueError('Forecast table must be the registered forecasters on 5088-8867.')
    if history['date_id'].tolist() != list(range(OOS_END+1)):
        raise ValueError('Scoring history must be the consecutive rows 0-8867.')
    h = history.set_index('date_id')
    lagged = h['forward_returns'].shift(1)            # official lagged_forward_returns, offset -1
    standards = {n: alloc.standardization(forecasts[n]) for n in v5['forecasters']}
    frame = pd.DataFrame({'block': oos_blocks(dates),
                          'c': alloc.volatility_ratio(lagged, v4['volatility_state']).loc[dates].to_numpy(),
                          'forward_returns': h.loc[dates, 'forward_returns'].to_numpy(float),
                          'risk_free_rate': h.loc[dates, 'risk_free_rate'].to_numpy(float)},
                         index=pd.Index(dates, name='date_id'))
    pos = positions(frame, forecasts, standards, order)
    overall, blocks = alloc.score_table(frame, pos, tuple(v5['selection']['cost_sensitivity_one_way_bps']))
    choice = select(overall, blocks, v5['selection'], order)
    overall = overall.merge(alloc.bootstrap_intervals(frame, pos), on='candidate')
    return {'standardization': standards, 'frame': frame, 'positions': pos, 'overall': overall, 'blocks': blocks,
            'selection': choice}


# ---------------------------------------------------------------- run

def git(root: Path, *args: str) -> str:
    import subprocess
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def code_identity(root: Path) -> dict:
    files = {n: protocol_v4.text_digest(root/n) for n in CODE}
    return {'code_sha256': files, 'code_hash': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def preflight(root: Path | None = None) -> dict:
    """Metadata only: lock, clean tagged checkout, frozen code, unused state."""
    root = (root or project_root()).resolve()
    lock = protocol_v5.verify_lock(root)
    commit = git(root, 'rev-parse', 'HEAD')
    if git(root, 'status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Dirty worktree: the v5 selection needs a clean checkout.')
    if git(root, 'cat-file', '-t', f'refs/tags/{TAG}') != 'tag':
        raise RuntimeError(f'An annotated {TAG} tag is required.')
    if git(root, 'rev-parse', f'{TAG}^{{commit}}') != commit:
        raise RuntimeError(f'Current commit differs from {TAG}.')
    if (root/STATE).exists() or (root/OUTPUT).exists():
        raise RuntimeError('The v5 selection was already reserved or run. Refusing to rerun.')
    return {'run_type': 'real', 'git_commit': commit, 'pipeline_tag': TAG, 'protocol_tag': protocol_v5.TAG,
            'protocol_v5_lock_sha256': protocol_v4.text_digest(root/protocol_v5.LOCK),
            'evidence_sha256': lock['evidence_sha256'], **code_identity(root), 'oos_range': [OOS_START, OOS_END],
            'final_simulation_read': False, 'preflight': 'passed'}


def write_outputs(result: dict, dynridge: pd.DataFrame, dyn_audit: dict, destination: Path, manifest: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    result['overall'].to_csv(destination/'v5_candidate_scores.csv', index=False, lineterminator='\n')
    result['blocks'].to_csv(destination/'v5_block_scores.csv', index=False, lineterminator='\n')
    daily = result['frame'][['block', 'c']].join(result['positions'])     # no daily return column
    daily.to_csv(destination/'v5_daily_positions.csv', lineterminator='\n')
    dynridge.to_csv(destination/'v5_dynridge_forecasts.csv', index=False, lineterminator='\n')
    (destination/'v5_dynridge_audit.json').write_text(json.dumps(dyn_audit, indent=2, sort_keys=True, default=float)+'\n',
                                                     encoding='utf-8', newline='\n')
    (destination/'v5_selection.json').write_text(json.dumps(
        {'selection': result['selection'], 'standardization': result['standardization']}, indent=2, sort_keys=True)+'\n',
        encoding='utf-8', newline='\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir())}
    manifest = {**manifest, 'output_sha256': hashes}
    (destination/'v5_run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n',
                                                    encoding='utf-8', newline='\n')
    return manifest


def run_selection(root: Path | None = None) -> dict:
    from src.research.alpha_pipeline import load_alpha_design
    from src.research.phase6_holdout import load_holdout_history
    from src.validation import protocol_v3_1
    from src.validation.protocol_freeze import verify_protocol_lock
    from src.validation.time_split import read_header
    root = (root or project_root()).resolve()
    identity = preflight(root)
    path = root/STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # single atomic claim
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump({**identity, 'state': 'reserved', 'started_at_utc': utc_timestamp(), 'real_xy_read': False}, handle, indent=2)

    def update(state: str, **extra):
        path.write_text(json.dumps({**identity, 'state': state, **extra}, indent=2)+'\n', encoding='utf-8')
    try:
        verify_protocol_lock(root, verify_data=True)
        design, v31 = load_alpha_design(root), protocol_v3_1.load_config(root)
        policy = json.loads((root/'configs/split_policy.json').read_text(encoding='utf-8'))
        train = root/'data/raw/train.csv'
        history = load_holdout_history(train, {**policy, 'train_columns': read_header(train)})
        update('running', real_xy_read=True)
        v5 = protocol_v5.load_config(root)
        dynridge, dyn_audit = dynridge_oos(history, v31, design)
        forecasts = assemble(pd.read_csv(root/v5['evidence']['research_forecasts']),
                             pd.read_csv(root/v5['evidence']['holdout_forecasts']), dynridge, v5['forecasters'])
        result = build_selection(forecasts, history, protocol_v4.load_config(root), v5)
        manifest = write_outputs(result, dynridge, dyn_audit, root/OUTPUT,
                                 {**identity, 'state': 'completed', 'real_xy_read': True, 'completed_at_utc': utc_timestamp()})
        update('completed', real_xy_read=True, selected=result['selection']['selected'],
               output_sha256=manifest['output_sha256'])
    except BaseException as exc:
        update('failed_or_interrupted', error_type=type(exc).__name__)
        raise
    return result


# ---------------------------------------------------------------- synthetic

def synthetic_inputs(seed: int = 7, strength: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Artificial rows 0-8867 and a wide forecast table; strength=0 is pure noise."""
    rng = np.random.default_rng(seed)
    n = OOS_END+1
    signal = rng.normal(size=n)
    vol = 0.01*np.exp(0.3*np.sin(np.arange(n)/400))
    market = 0.0003 + vol*(strength*signal + np.sqrt(1 - strength**2)*rng.normal(size=n))
    history = pd.DataFrame({'date_id': np.arange(n), 'forward_returns': market, 'risk_free_rate': np.full(n, 0.0001)})
    dates = np.arange(OOS_START, n)
    forecasts = pd.DataFrame({name: scale*(signal[dates] + rng.normal(size=len(dates))*noise) + shift
                              for name, scale, noise, shift in (('A', .2, 2, -.04), ('Ridge', .001, 3, 0),
                                                                ('DynRidge', .002, 1.5, .0001), ('C', .001, 4, 0),
                                                                ('D', 1, 2.5, .1))},
                             index=pd.Index(dates, name='date_id'))
    return forecasts, history


def run_synthetic(destination: Path, seed: int, strength: float) -> dict:
    root = project_root()
    for forbidden in ('artifacts', 'data', 'reports'):
        if destination.resolve().is_relative_to((root/forbidden).resolve()):
            raise ValueError('Synthetic output must stay outside artifacts, data and reports.')
    forecasts, history = synthetic_inputs(seed, strength)
    result = build_selection(forecasts, history, protocol_v4.load_config(root), protocol_v5.load_config(root))
    dyn = forecasts['DynRidge'].rename('prediction').reset_index()
    write_outputs(result, dyn, {}, destination, {'run_type': 'synthetic', 'seed': seed, 'strength': strength,
                                                 **code_identity(root)})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('check', 'synthetic', 'run-selection'):
        mode.add_argument('--'+name, action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--strength', type=float, default=0.15)
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(preflight(), indent=2)); return 0
    if args.run_selection:
        print(json.dumps(run_selection()['selection'], indent=2)); return 0
    if args.output is None:
        parser.error('--synthetic requires --output outside artifacts/, data/ and reports/.')
    print(json.dumps(run_synthetic(args.output, args.seed, args.strength)['selection'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
