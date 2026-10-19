"""Phase 4 execution identity, read-only preflight and canonical reservation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy

from src.utils.paths import project_root, utc_timestamp
from src.validation import protocol_v3_1 as protocol
from src.validation.protocol_freeze import verify_protocol_lock

TAG = 'phase4-engineered-pipeline-v1'
STATE = 'artifacts/phase4_execution_state.json'
REAL_OUTPUT = 'artifacts/phase4'
MANIFEST = REAL_OUTPUT+'/phase4_run_manifest.json'
CODE = ('src/features/engineered_alpha.py','src/research/phase4_engine.py',
    'src/research/phase4_synthetic.py','src/research/phase4_calibration.py',
    'src/research/phase4_execution.py','src/research/phase4.py','requirements.txt')


def git(root: Path, *args: str) -> str:
    return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()


def code_identity(root: Path) -> dict:
    files = {n:protocol.parent.digest(root/n) for n in CODE}
    h = hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'phase4_pipeline_code_hash':h,'implementation_sha256':files}


def execution_identity(root: Path, run_type: str) -> dict:
    if run_type not in ('real','synthetic'):
        raise ValueError('Explicit real or synthetic run_type required.')
    lock = protocol.verify_lock(root)
    return {'run_type':run_type, 'created_at_utc':utc_timestamp(), 'git_commit':git(root,'rev-parse','HEAD'),
        'git_tag':git(root,'tag','--points-at','HEAD').splitlines(),
        'git_worktree_dirty':bool(git(root,'status','--porcelain','--untracked-files=all')),
        'protocol_tag':protocol.TAG, 'protocol_commit':protocol.verify_tag(root),
        'protocol_lock_sha256':protocol.parent.digest(root/protocol.LOCK),
        'phase3_pipeline_code_hash':lock['phase3_pipeline_code_hash'], **code_identity(root),
        'research_boundary':[0,7859], 'internal_holdout':[7860,8867], 'final_simulation':[8868,9047],
        'data_hash':verify_protocol_lock(root)['data_sha256']['data/raw/train.csv'],
        'data_hash_source':'frozen metadata; real execution revalidates bytes before loading',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
        'real_xy_read':run_type=='real','internal_holdout_read':False,'final_simulation_read':False}


def assert_unused(root: Path):
    # Canonical project state is independent of the requested destination.
    if (root/STATE).exists():
        raise RuntimeError('Real Phase 4 reservation already exists. Refusing to rerun.')
    if (root/MANIFEST).exists():
        saved = json.loads((root/MANIFEST).read_text(encoding='utf-8'))
        if saved.get('run_type')=='real':
            raise RuntimeError('Real Phase 4 has already been executed. Refusing to rerun.')
        raise RuntimeError('Canonical real output is occupied. Refusing to overwrite.')
    if (root/REAL_OUTPUT).exists():
        raise RuntimeError('Canonical real output already exists; refusing to overwrite.')


def preflight(root: Path | None = None) -> dict:
    root = (root or project_root()).resolve()
    if root != project_root().resolve() or Path(git(root,'rev-parse','--show-toplevel')).resolve()!=root:
        raise ValueError('Real entry must use the canonical executing project root.')
    identity = execution_identity(root,'real')
    if identity['git_worktree_dirty']:
        raise RuntimeError('Dirty worktree: real Phase 4 requires a clean worktree.')
    if git(root,'cat-file','-t',f'refs/tags/{TAG}')!='tag':
        raise RuntimeError('An annotated Phase 4 pipeline tag is required.')
    if git(root,'rev-parse',f'{TAG}^{{commit}}')!=identity['git_commit']:
        raise RuntimeError('Current commit differs from the frozen Phase 4 pipeline tag.')
    for n,h in identity['implementation_sha256'].items():
        blob = subprocess.run(['git','-C',str(root),'show',f'{TAG}:{n}'],check=True,capture_output=True).stdout
        if hashlib.sha256(blob.replace(b'\r\n',b'\n')).hexdigest()!=h:
            raise RuntimeError(f'Frozen Phase 4 code differs: {n}')
    c = protocol.load_config(root)
    if (c['research']['start'],c['research']['end'],c['holdout']['start'],c['final_simulation']['start'])!=(0,7859,7860,8868):
        raise ValueError('Research/holdout/final boundaries changed.')
    assert_unused(root)
    return {**identity,'preflight':'passed','real_xy_read':False,'data_bytes_verified':False,
        'real_run_state':'unused','pipeline_tag':TAG,'read_only':True}


def reserve(root: Path, identity: dict):
    assert_unused(root)
    path = root/STATE
    path.parent.mkdir(parents=True,exist_ok=True)
    # O_EXCL is the cross-process authority. Any failed/interrupted claim remains consumed.
    fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as f:
        json.dump({**identity,'run_type':'real','state':'reserved','reservation_consumed':True},f,indent=2)
        f.flush(); os.fsync(f.fileno())


def update_state(root: Path, identity: dict, state: str, **extra):
    temp = (root/STATE).with_suffix('.tmp')
    temp.write_text(json.dumps({**identity,'state':state,'reservation_consumed':True,**extra},indent=2)+'\n',encoding='utf-8')
    temp.replace(root/STATE)


def write_outputs(tables, destination: Path, manifest: dict):
    # No overwrites, including synthetic runs.
    destination.mkdir(parents=True,exist_ok=False)
    for pipeline,name in [('A','raw_control'),('B','engineered')]:
        tables.folds.loc[tables.folds.pipeline==pipeline].to_csv(destination/f'phase4_{name}_fold_results.csv',index=False)
        tables.selection.loc[tables.selection.pipeline==pipeline].to_csv(destination/f'phase4_{name}_selection.csv',index=False)
    tables.candidates.to_csv(destination/'phase4_candidate_audit.csv',index=False)
    tables.redundancy.to_csv(destination/'phase4_redundancy_audit.csv',index=False)
    tables.summary.to_csv(destination/'phase4_multiple_fold_summary.csv',index=False)
    tables.predictions.to_csv(destination/'phase4_predictions.csv',index=False)
    (destination/'phase4_transform_audit.json').write_text(json.dumps(tables.transforms,indent=2)+'\n',encoding='utf-8')
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}
    (destination/'phase4_run_manifest.json').write_text(json.dumps({**manifest,'output_sha256':hashes},indent=2)+'\n',encoding='utf-8')


def run_real(root: Path | None = None, output: Path | None = None):
    """Future explicit entry. This function is NOT called during Phase 4A."""
    from src.research.alpha_pipeline import load_alpha_design
    from src.research.phase4_engine import build_phase4_tables
    from src.validation.walk_forward import load_research_prefix
    root = (root or project_root()).resolve()
    # Reject an alternative destination before any read or reservation.
    if output is not None and output.resolve()!=(root/REAL_OUTPUT).resolve():
        raise ValueError('Real output is canonical; changing directories cannot bypass the reservation.')
    checked = preflight(root)
    identity = {**checked,'run_type':'real','real_xy_read':False,'read_only':False}
    reserve(root,identity)
    try:
        verify_protocol_lock(root,verify_data=True)
        split = json.loads((root/'configs/split_policy.json').read_text(encoding='utf-8'))
        design = load_alpha_design(root)
        history = load_research_prefix(root/'data/raw/train.csv',split,design.validation)
        identity.update(real_xy_read=True,data_bytes_verified=True)
        update_state(root,identity,'running')
        tables = build_phase4_tables(history,protocol.load_config(root),design)
        write_outputs(tables,root/REAL_OUTPUT,{**identity,'run_state':'completed'})
        update_state(root,identity,'completed')
    except BaseException as exc:
        update_state(root,identity,'failed_or_interrupted',error_type=type(exc).__name__)
        raise
    return tables
