"""Phase 6B single Internal Holdout run: guards, resume before evaluation, outputs.

python -m src.research.phase6_holdout_run --check
python -m src.research.phase6_holdout_run --run-holdout --workers 4

States: reserved -> forecasting -> forecasts_sealed -> completed. A crash before
forecasts are sealed may resume with the same identity (holdout targets are not
attached before sealing). After sealing, only the deterministic evaluation of the
sealed forecasts may be replayed; no model is ever refitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src.research import phase6_holdout as ho
from src.research.phase5_tasks import TaskStore
from src.utils.paths import project_root, utc_timestamp
from src.validation import protocol_v4, protocol_v4_1

TAG = 'phase6-holdout-pipeline-v1'
STATE = 'artifacts/phase6_holdout_state.json'
OUTPUT = 'artifacts/phase6_holdout'
SEALED = OUTPUT + '/holdout_forecasts_sealed.csv'
RESEARCH_PREDICTIONS = 'reports/phase5_real_run/phase5_predictions.csv'
CODE = ('src/research/phase6_holdout.py', 'src/research/phase6_holdout_run.py', 'src/research/phase6_allocation.py',
        'tests/test_phase6_holdout.py')


def git(root: Path, *args: str) -> str:
    import subprocess
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def identity(root: Path) -> dict:
    lock = protocol_v4_1.verify_lock(root)
    files = {n: protocol_v4.text_digest(root/n) for n in CODE}
    return {'run_type': 'holdout', 'pipeline_tag': TAG, 'git_commit': git(root, 'rev-parse', 'HEAD'),
            'protocol_tags': [protocol_v4.TAG, protocol_v4_1.TAG],
            'protocol_v4_1_lock_sha256': protocol_v4.text_digest(root/protocol_v4_1.LOCK),
            'sealed_phase6_selection_sha256': lock['sealed_phase6_selection_sha256'],
            'code_sha256': files, 'code_hash': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'fit_rows': [0, ho.FIT_END], 'purge_row': ho.PURGE, 'holdout_rows': [ho.START, ho.STOP-1],
            'final_simulation_read': False}


def read_state(root: Path) -> dict | None:
    path = root/STATE
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def preflight(root: Path | None = None) -> dict:
    root = (root or project_root()).resolve()
    ident = identity(root)
    if git(root, 'status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Dirty worktree: the holdout run needs a clean checkout.')
    if git(root, 'cat-file', '-t', f'refs/tags/{TAG}') != 'tag':
        raise RuntimeError(f'An annotated {TAG} tag is required.')
    if git(root, 'rev-parse', f'{TAG}^{{commit}}') != ident['git_commit']:
        raise RuntimeError(f'Current commit differs from {TAG}.')
    state = read_state(root)
    if state is not None:
        if state.get('identity') != ident:
            raise RuntimeError('Holdout state exists with a different identity; refusing to run.')
        if state['state'] == 'completed':
            raise RuntimeError('The Internal Holdout has already been evaluated. Refusing to rerun.')
    return {**ident, 'preflight': 'passed', 'existing_state': None if state is None else state['state']}


def write_state(root: Path, ident: dict, state: str, **extra) -> None:
    path = root/STATE
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'identity': ident, 'state': state, 'updated_at_utc': utc_timestamp(), **extra},
                              indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    tmp.replace(path)


def reserve(root: Path, ident: dict) -> str:
    """Atomic first claim; an existing same-identity state can only continue forward."""
    path = root/STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return read_state(root)['state']
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump({'identity': ident, 'state': 'reserved', 'updated_at_utc': utc_timestamp()}, handle, indent=2, sort_keys=True)
    return 'reserved'


def write_outputs(root: Path, ident: dict, forecasts: pd.DataFrame, audit: dict, metrics: pd.DataFrame, allocation: dict) -> dict:
    out = root/OUTPUT
    metrics.to_csv(out/'holdout_forecast_metrics.csv', index=False, lineterminator='\n')
    allocation['overall'].to_csv(out/'holdout_allocation_scores.csv', index=False, lineterminator='\n')
    allocation['blocks'].to_csv(out/'holdout_allocation_block_scores.csv', index=False, lineterminator='\n')
    allocation['frame'].join(allocation['positions']).to_csv(out/'holdout_daily_positions.csv', lineterminator='\n')
    allocation['secondary_forecast_allocations'].to_csv(out/'holdout_secondary_forecast_allocations.csv', index=False,
                                                       lineterminator='\n')
    a = metrics.set_index('pipeline').loc['A']
    summary = {'forecast_confirmation_A': {k: (None if pd.isna(v) else v) for k, v in a.to_dict().items()},
               'paired_allocation_differences': allocation['paired'],
               'primary_allocation': ho.PRIMARY_MAPPING, 'interpretation': 'per Protocol v4.1 section 4'}
    (out/'holdout_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, default=float)+'\n',
                                            encoding='utf-8', newline='\n')
    (out/'holdout_fit_audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True,
        default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))+'\n', encoding='utf-8', newline='\n')
    files = sorted(p for p in out.iterdir() if p.is_file())
    manifest = {**ident, 'state': 'completed', 'completed_at_utc': utc_timestamp(),
                'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out/'holdout_run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n',
                                                 encoding='utf-8', newline='\n')
    return manifest


def run_holdout(root: Path | None = None, workers: int = 4, *, test_budget=None, store_identity=None) -> dict:
    from src.research.alpha_pipeline import load_alpha_design
    from src.validation import protocol_v3_1
    from src.validation.protocol_freeze import verify_protocol_lock
    root = (root or project_root()).resolve()
    checked = preflight(root)
    ident = {k: v for k, v in checked.items() if k not in ('preflight', 'existing_state')}
    state = reserve(root, ident)
    try:
        verify_protocol_lock(root, verify_data=True)
        design = load_alpha_design(root)
        policy = json.loads((root/'configs/split_policy.json').read_text(encoding='utf-8'))
        from src.validation.time_split import read_header
        train = root/'data/raw/train.csv'
        history = ho.load_holdout_history(train, {**policy, 'train_columns': read_header(train)})
        sealed = root/SEALED
        if state in ('reserved', 'forecasting', 'failed_pre_forecast'):
            write_state(root, ident, 'forecasting')
            store = TaskStore(root/OUTPUT/'tasks', store_identity or {**ident, 'run_type': 'holdout'})
            engine = ho.HoldoutEngine(history, protocol_v3_1.load_config(root), design, store, test_budget=test_budget)
            engine.run_neural(workers)
            forecasts, audit = engine.holdout_forecasts()
            forecasts.to_csv(sealed, index=False, lineterminator='\n')
            (root/OUTPUT/'holdout_fit_audit_pending.json').write_text(json.dumps(audit, sort_keys=True,
                default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)), encoding='utf-8')
            write_state(root, ident, 'forecasts_sealed', sealed_sha256=hashlib.sha256(sealed.read_bytes()).hexdigest())
        saved = read_state(root)
        if saved['state'] not in ('forecasts_sealed', 'failed_post_forecast') or \
                hashlib.sha256(sealed.read_bytes()).hexdigest() != saved['sealed_sha256']:
            raise RuntimeError('Sealed holdout forecasts are missing or changed; refusing evaluation.')
        forecasts = pd.read_csv(sealed)
        audit = json.loads((root/OUTPUT/'holdout_fit_audit_pending.json').read_text(encoding='utf-8'))
        # The only place the holdout target is attached: after forecasts are sealed.
        target = history.set_index('date_id')[design.rules['primary_target']]
        metrics = ho.forecast_metrics(forecasts, target)
        allocation = ho.allocation_evaluation(forecasts, history, protocol_v4.load_config(root),
                                              protocol_v4_1.load_config(root), pd.read_csv(root/RESEARCH_PREDICTIONS))
        manifest = write_outputs(root, ident, forecasts, audit, metrics, allocation)
        write_state(root, ident, 'completed', sealed_sha256=saved['sealed_sha256'], output_sha256=manifest['output_sha256'])
        return {'metrics': metrics, 'allocation': allocation, 'manifest': manifest}
    except BaseException as exc:
        current = read_state(root) or {}
        after = current.get('state') in ('forecasts_sealed', 'failed_post_forecast')
        write_state(root, ident, 'failed_post_forecast' if after else 'failed_pre_forecast',
                    error_type=type(exc).__name__, **({'sealed_sha256': current['sealed_sha256']} if after else {}))
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run-holdout', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(preflight(), indent=2)); return 0
    result = run_holdout(workers=args.workers)
    print(json.dumps({'state': 'completed', 'output_sha256': result['manifest']['output_sha256']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
