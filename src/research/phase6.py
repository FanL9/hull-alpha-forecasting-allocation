"""Phase 6A: select the allocation mapping on A's research OOS forecasts (5088-7859).

python -m src.research.phase6 --check                      # metadata only
python -m src.research.phase6 --synthetic --output DIR     # artificial data
python -m src.research.phase6 --run-selection              # one real run

The real run needs a clean checkout at the annotated pipeline tag, a verified
Protocol v4 lock and an unused canonical state. It parses only the research
prefix through the bounded loader; the holdout and final rows are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research import phase6_allocation as alloc
from src.utils.paths import project_root, utc_timestamp
from src.validation import protocol_v4

TAG = 'phase6-allocation-pipeline-v1'
STATE = 'artifacts/phase6_selection_state.json'
OUTPUT = 'artifacts/phase6_selection'
PREDICTIONS = 'reports/phase5_real_run/phase5_predictions.csv'
CODE = ('src/research/phase6_allocation.py', 'src/research/phase6.py', 'tests/test_phase6_allocation.py')


def git(root: Path, *args: str) -> str:
    import subprocess
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def code_identity(root: Path) -> dict:
    files = {n: protocol_v4.text_digest(root/n) for n in CODE}
    return {'phase6_code_sha256': files,
            'phase6_code_hash': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def assert_unused(root: Path) -> None:
    if (root/STATE).exists():
        raise RuntimeError('Phase 6 selection was already reserved or run. Refusing to rerun.')
    if (root/OUTPUT).exists():
        raise RuntimeError('Canonical Phase 6 selection output exists; refusing to overwrite.')


def preflight(root: Path | None = None) -> dict:
    """Metadata only: lock, clean tagged checkout, frozen code, unused state."""
    root = (root or project_root()).resolve()
    lock = protocol_v4.verify_lock(root)
    commit = git(root, 'rev-parse', 'HEAD')
    if git(root, 'status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Dirty worktree: the real Phase 6 selection needs a clean checkout.')
    if git(root, 'cat-file', '-t', f'refs/tags/{TAG}') != 'tag':
        raise RuntimeError(f'An annotated {TAG} tag is required.')
    if git(root, 'rev-parse', f'{TAG}^{{commit}}') != commit:
        raise RuntimeError(f'Current commit differs from {TAG}.')
    assert_unused(root)
    return {'run_type': 'real', 'git_commit': commit, 'pipeline_tag': TAG, 'protocol_tag': protocol_v4.TAG,
            'protocol_v4_lock_sha256': protocol_v4.text_digest(root/protocol_v4.LOCK),
            'sealed_phase5_sha256': lock['sealed_phase5_sha256'], **code_identity(root),
            'calibration_range': [5088, 7859], 'internal_holdout_read': False, 'final_simulation_read': False,
            'preflight': 'passed'}


def write_outputs(result: dict, destination: Path, manifest: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    result['overall'].to_csv(destination/'phase6_candidate_scores.csv', index=False, lineterminator='\n')
    result['blocks'].to_csv(destination/'phase6_block_scores.csv', index=False, lineterminator='\n')
    daily = result['frame'].join(result['positions'])
    daily.to_csv(destination/'phase6_daily_positions.csv', lineterminator='\n')
    (destination/'phase6_selection.json').write_text(json.dumps(
        {'selection': result['selection'], 'standardization': result['standardization']}, indent=2, sort_keys=True)+'\n',
        encoding='utf-8', newline='\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir())}
    manifest = {**manifest, 'output_sha256': hashes}
    (destination/'phase6_run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n',
                                                         encoding='utf-8', newline='\n')
    return manifest


def run_selection(root: Path | None = None) -> dict:
    from src.research.alpha_pipeline import load_alpha_design
    from src.research.phase4_execution import load_research_history
    from src.validation.protocol_freeze import verify_protocol_lock
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
        config = protocol_v4.load_config(root)
        history = load_research_history(root, load_alpha_design(root))
        update('running', real_xy_read=True)
        predictions = pd.read_csv(root/PREDICTIONS)
        result = alloc.build_selection(predictions, history, config)
        manifest = write_outputs(result, root/OUTPUT, {**identity, 'state': 'completed', 'real_xy_read': True,
                                                       'completed_at_utc': utc_timestamp()})
        update('completed', real_xy_read=True, selected=result['selection']['selected'],
               output_sha256=manifest['output_sha256'])
    except BaseException as exc:
        update('failed_or_interrupted', error_type=type(exc).__name__)
        raise
    return result


def synthetic_inputs(seed: int = 7, strength: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Artificial research prefix and A-shaped OOS forecasts; strength=0 is pure noise."""
    rng = np.random.default_rng(seed)
    n = 7860
    signal = rng.normal(size=n)
    vol = 0.01*np.exp(0.3*np.sin(np.arange(n)/400))
    market = 0.0003 + vol*(strength*signal + np.sqrt(1 - strength**2)*rng.normal(size=n))
    history = pd.DataFrame({'date_id': np.arange(n), 'forward_returns': market,
                            'risk_free_rate': np.full(n, 0.0001)})
    dates = np.arange(5088, 7860)
    forecast = 0.2*(signal[dates] + rng.normal(size=len(dates))*2) - 0.04
    predictions = pd.DataFrame({'date_id': dates, 'fold_id': (dates - 5088)//252 + 1, 'pipeline': 'A',
                                'prediction': forecast})
    predictions.attrs['phase6_data_kind'] = 'synthetic'
    return predictions, history


def run_synthetic(destination: Path, seed: int, strength: float) -> dict:
    root = project_root()
    if destination.resolve().is_relative_to((root/'artifacts').resolve()) or \
            destination.resolve().is_relative_to((root/'data').resolve()):
        raise ValueError('Synthetic output must stay outside canonical artifacts and data.')
    predictions, history = synthetic_inputs(seed, strength)
    result = alloc.build_selection(predictions, history, protocol_v4.load_config(root))
    write_outputs(result, destination, {'run_type': 'synthetic', 'seed': seed, 'strength': strength,
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
        result = run_selection()
        print(json.dumps(result['selection'], indent=2)); return 0
    if args.output is None:
        parser.error('--synthetic requires --output outside artifacts/ and data/.')
    print(json.dumps(run_synthetic(args.output, args.seed, args.strength)['selection'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
