"""Protocol v5 Final Simulation (8868-9047): one run, research and competition tracks.

python -m src.research.phase7_final --check
python -m src.research.phase7_final --run-final --workers 4

Requires the v5 selection to be completed and its outputs committed under
reports/phase7_selection. The checkout must descend from the pipeline tag with
no change outside that directory. States: reserved -> forecasting ->
forecasts_sealed -> completed. Before sealing, a crash may resume with the same
identity (no Final outcome is attached before sealing); after sealing only the
deterministic evaluation of the sealed forecasts may be replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import sha256
from src.research import phase6_allocation as alloc
from src.research import phase7_refit as refit
from src.research import phase7_selection as sel
from src.research.phase4_engine import corr
from src.utils.paths import project_root, utc_timestamp
from src.validation import protocol_v4, protocol_v4_1, protocol_v5
from src.validation.time_split import read_header

START, STOP, FIT_END = refit.FINAL_START, refit.FINAL_STOP, refit.FINAL_FIT_END
STATE = 'artifacts/phase7_final_state.json'
OUTPUT = 'artifacts/phase7_final'
SEALED = OUTPUT + '/final_forecasts_sealed.csv'
SELECTION_FILES = ('v5_selection.json', 'v5_run_manifest.json', 'phase7_selection_state.json')
RESEARCH_MAPPING = 'volscaled_k0.10'


def git(root: Path, *args: str) -> str:
    import subprocess
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def load_final_history(train_path, split_plan: dict) -> pd.DataFrame:
    """All 9048 rows; only this run may parse the Final Simulation rows."""
    if (split_plan['competition_simulation_start'] != START or sha256(train_path) != split_plan['source_train_sha256']
            or read_header(train_path) != split_plan['train_columns']):
        raise ValueError('Final source/schema differs from the split policy.')
    frame = pd.read_csv(train_path)
    if not np.array_equal(frame['date_id'], np.arange(STOP)):
        raise ValueError('Final history must be the consecutive rows 0-9047.')
    return frame


def sealed_selection(root: Path) -> dict:
    """The committed v5 selection, bound to the local completed state by hash."""
    published = root/sel.PUBLISHED
    state = json.loads((published/'phase7_selection_state.json').read_text(encoding='utf-8'))
    if state.get('state') != 'completed':
        raise RuntimeError('The v5 selection is not completed.')
    digest = hashlib.sha256((published/'v5_selection.json').read_bytes()).hexdigest()
    if digest != state['output_sha256'].get('v5_selection.json'):
        raise RuntimeError('Committed v5_selection.json differs from the sealed selection.')
    choice = json.loads((published/'v5_selection.json').read_text(encoding='utf-8'))
    return {'selection': choice['selection'], 'standardization': choice['standardization'],
            'sha256': {n: hashlib.sha256((published/n).read_bytes()).hexdigest() for n in SELECTION_FILES}}


def track_forecasters(selected: str) -> list[str]:
    """A for the research track plus the competition forecaster, if any."""
    names = ['A']
    if '/' in selected and selected.split('/')[1] not in names:
        names.append(selected.split('/')[1])
    return names


def identity(root: Path) -> dict:
    lock = protocol_v5.verify_lock(root)
    chosen = sealed_selection(root)
    files = {n: protocol_v4.text_digest(root/n) for n in sel.CODE}
    return {'run_type': 'final', 'pipeline_tag': sel.TAG, 'git_commit': git(root, 'rev-parse', 'HEAD'),
            'protocol_tags': [protocol_v4.TAG, protocol_v4_1.TAG, protocol_v5.TAG],
            'protocol_v5_lock_sha256': protocol_v4.text_digest(root/protocol_v5.LOCK),
            'evidence_sha256': lock['evidence_sha256'], 'v5_selection_sha256': chosen['sha256'],
            'competition_selected': chosen['selection']['selected'],
            'code_sha256': files, 'code_hash': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'fit_rows': [0, FIT_END], 'purge_row': FIT_END+1, 'final_rows': [START, STOP-1]}


def read_state(root: Path) -> dict | None:
    path = root/STATE
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def preflight(root: Path | None = None) -> dict:
    root = (root or project_root()).resolve()
    if git(root, 'status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Dirty worktree: the Final Simulation needs a clean checkout.')
    if git(root, 'cat-file', '-t', f'refs/tags/{sel.TAG}') != 'tag':
        raise RuntimeError(f'An annotated {sel.TAG} tag is required.')
    tagged = git(root, 'rev-parse', f'{sel.TAG}^{{commit}}')
    try:
        git(root, 'merge-base', '--is-ancestor', tagged, 'HEAD')
    except Exception as exc:
        raise RuntimeError(f'HEAD does not descend from {sel.TAG}.') from exc
    changed = [p for p in git(root, 'diff', '--name-only', tagged, 'HEAD').splitlines() if p]
    if any(not p.startswith(sel.PUBLISHED+'/') for p in changed):
        raise RuntimeError(f'Only {sel.PUBLISHED}/ may change after {sel.TAG}: {changed}')
    ident = identity(root)
    state = read_state(root)
    if state is not None:
        if state.get('identity') != ident:
            raise RuntimeError('Final state exists with a different identity; refusing to run.')
        if state['state'] == 'completed':
            raise RuntimeError('The Final Simulation has already been evaluated. Refusing to rerun.')
    return {**ident, 'preflight': 'passed', 'existing_state': None if state is None else state['state']}


def write_state(root: Path, ident: dict, state: str, **extra) -> None:
    path = root/STATE
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'identity': ident, 'state': state, 'updated_at_utc': utc_timestamp(), **extra},
                              indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    tmp.replace(path)


def reserve(root: Path, ident: dict) -> str:
    path = root/STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return read_state(root)['state']
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump({'identity': ident, 'state': 'reserved', 'updated_at_utc': utc_timestamp()}, handle, indent=2, sort_keys=True)
    return 'reserved'


# ---------------------------------------------------------------- evaluation

def evaluation(forecasts: pd.DataFrame, history: pd.DataFrame, target_name: str, v4: dict, v41: dict, chosen: dict) -> dict:
    """Both tracks and the two forecast-free comparators; descriptive only, no winner."""
    h = history.set_index('date_id')
    dates = np.arange(START, STOP)
    lagged = h['forward_returns'].shift(1)                        # official lagged field, offset -1
    frame = pd.DataFrame({'block': 1, 'c': alloc.volatility_ratio(lagged, v4['volatility_state']).loc[dates].to_numpy(),
                          'forward_returns': h.loc[dates, 'forward_returns'].to_numpy(float),
                          'risk_free_rate': h.loc[dates, 'risk_free_rate'].to_numpy(float)},
                         index=pd.Index(dates, name='date_id'))
    wide = forecasts.pivot(index='date_id', columns='pipeline', values='prediction').loc[dates]
    research_state = {'mu': v41['phase6_selection']['mu_A'], 'sigma': v41['phase6_selection']['sigma_A']}
    pos = pd.DataFrame({'constant_1': 1.0, 'volatility_only': frame['c'].clip(0, 2)}, index=frame.index)
    pos['research_track'] = alloc.positions(RESEARCH_MAPPING, alloc.standardize(wide['A'], research_state), frame['c'])
    selected = chosen['selection']['selected']
    if '/' in selected:
        mapping, name = selected.split('/')
        pos['competition_track'] = alloc.positions(mapping, alloc.standardize(wide[name], chosen['standardization'][name]),
                                                   frame['c'])
    else:
        pos['competition_track'] = pos[selected]
    overall, _ = alloc.score_table(frame, pos, (0, 1, 5))
    target = h.loc[dates, target_name].to_numpy(float)
    ics = [{'pipeline': n, 'rank_ic_180': corr(wide[n].to_numpy(float), target, True), 'role': 'descriptive_only'}
           for n in wide]
    return {'frame': frame, 'positions': pos, 'overall': overall, 'rank_ic': pd.DataFrame(ics),
            'tracks': {'research': f'{RESEARCH_MAPPING}/A', 'competition': selected}}


def write_outputs(root: Path, ident: dict, audit: dict, result: dict) -> dict:
    out = root/OUTPUT
    result['overall'].to_csv(out/'final_scores.csv', index=False, lineterminator='\n')
    result['rank_ic'].to_csv(out/'final_rank_ic.csv', index=False, lineterminator='\n')
    result['frame'][['c']].join(result['positions']).to_csv(out/'final_daily_positions.csv', lineterminator='\n')
    scores = result['overall'].set_index('candidate')['official_score']
    summary = {'tracks': result['tracks'], 'official_score': {k: float(v) for k, v in scores.items()},
               'rows': STOP-START, 'interpretation': 'Protocol v5 section 6: completes the process; no track is ranked'}
    (out/'final_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    (out/'final_fit_audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True,
        default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))+'\n', encoding='utf-8', newline='\n')
    files = sorted(p for p in out.iterdir() if p.is_file())
    manifest = {**ident, 'state': 'completed', 'completed_at_utc': utc_timestamp(),
                'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out/'final_run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n',
                                               encoding='utf-8', newline='\n')
    return manifest


def run_final(root: Path | None = None, workers: int = 4, *, test_budget=None, store_identity=None) -> dict:
    from src.research.alpha_pipeline import load_alpha_design
    from src.research.phase5_tasks import TaskStore
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
        train = root/'data/raw/train.csv'
        history = load_final_history(train, {**policy, 'train_columns': read_header(train)})
        chosen = sealed_selection(root)
        names = track_forecasters(chosen['selection']['selected'])
        sealed = root/SEALED
        if state in ('reserved', 'forecasting', 'failed_pre_forecast'):
            write_state(root, ident, 'forecasting')
            store = TaskStore(root/OUTPUT/'tasks', store_identity or {**ident, 'run_type': 'final'})
            engine = refit.RefitEngine(history, protocol_v3_1.load_config(root), design, store, FIT_END, START, STOP,
                                       test_budget=test_budget)
            if refit.neural_needed(names):
                if 'D' not in names:
                    engine.references = (FIT_END,)       # C alone needs no inner references
                engine.run_neural(workers)
            forecasts, audit = engine.refit_forecasts(names)
            forecasts.to_csv(sealed, index=False, lineterminator='\n')
            (root/OUTPUT/'final_fit_audit_pending.json').write_text(json.dumps(audit, sort_keys=True,
                default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)), encoding='utf-8')
            write_state(root, ident, 'forecasts_sealed', sealed_sha256=hashlib.sha256(sealed.read_bytes()).hexdigest())
        saved = read_state(root)
        if saved['state'] not in ('forecasts_sealed', 'failed_post_forecast') or \
                hashlib.sha256(sealed.read_bytes()).hexdigest() != saved['sealed_sha256']:
            raise RuntimeError('Sealed Final forecasts are missing or changed; refusing evaluation.')
        forecasts = pd.read_csv(sealed)
        audit = json.loads((root/OUTPUT/'final_fit_audit_pending.json').read_text(encoding='utf-8'))
        # The only place Final outcomes are attached: after forecasts are sealed.
        result = evaluation(forecasts, history, design.rules['primary_target'], protocol_v4.load_config(root),
                            protocol_v4_1.load_config(root), chosen)
        manifest = write_outputs(root, ident, audit, result)
        write_state(root, ident, 'completed', sealed_sha256=saved['sealed_sha256'], output_sha256=manifest['output_sha256'])
        return {**result, 'manifest': manifest}
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
    mode.add_argument('--run-final', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(preflight(), indent=2)); return 0
    result = run_final(workers=args.workers)
    print(json.dumps({'state': 'completed', 'output_sha256': result['manifest']['output_sha256']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
