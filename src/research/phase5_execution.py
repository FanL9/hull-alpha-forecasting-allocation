"""Canonical Phase 5 identity, read-only checks, one resumable run and outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess

from filelock import FileLock
import numpy as np
import pandas as pd
import scipy
import torch
from threadpoolctl import threadpool_limits

from src.research.alpha_pipeline import load_alpha_design
from src.research.phase4_execution import load_research_history, code_identity as phase4_code
from src.research.phase5_tasks import TaskStore, atomic, encode, hash_object, read, replace_atomic
from src.utils.paths import project_root
from src.validation import protocol_v3_1 as protocol
from src.validation.protocol_freeze import verify_protocol_lock

TAG='phase5-deep-alpha-pipeline-v1'
STATE='artifacts/phase5_execution_state.json'
ANCHOR='artifacts/phase5_reservation.json'
OUTPUT='artifacts/phase5'
CODE=('src/features/deep_inputs.py','src/models/deep_alpha.py',
      'src/research/phase5_tasks.py','src/research/phase5_engine.py',
      'src/research/phase5_execution.py','src/research/phase5_synthetic.py',
      'src/research/phase5_calibration.py','src/research/phase5_benchmark.py',
      'src/research/phase5.py','requirements-phase5.txt','configs/phase5_environment.json',
      'configs/phase5_calibration.json','configs/phase5_protected.json','docs/phase5_pipeline.md')


def git(root,*args):
    return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()


def environment():
    return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'pandas':pd.__version__,'scipy':scipy.__version__,'platform':platform.system(),'machine':platform.machine()}


def code_identity(root):
    files={n:protocol.parent.digest(root/n) for n in CODE}
    return {'phase5_pipeline_code_hash':hash_object(files),'implementation_sha256':files}


def protected(root):
    expected=json.loads((root/'configs/phase5_protected.json').read_text(encoding='utf-8'))
    for name,digest in expected['file_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Protected Phase 3/4 artifact changed: '+name)
    if phase4_code(root)['phase4_pipeline_code_hash']!=expected['phase4_pipeline_code_hash']:
        raise RuntimeError('Frozen Phase 4 implementation changed.')
    return expected


def identity(root,run_type,data_root=None,test_budget=None):
    if run_type not in ('real','synthetic') or (test_budget is not None and run_type!='synthetic'):
        raise ValueError('Invalid execution mode/budget.')
    config=protocol.load_config(root)
    source=data_root or root
    split=json.loads((source/'configs/split_policy.json').read_text(encoding='utf-8'))
    return {'run_type':run_type,'git_commit':git(root,'rev-parse','HEAD'),**code_identity(root),
            'protocol_commit':protocol.verify_tag(root),'protocol_tag':protocol.TAG,
            'protocol_hash':protocol.parent.digest(root/protocol.LOCK),
            'model_config_hash':hash_object({k:config[k] for k in ('deep','models','training','nonlinear_pipeline','incremental','combined')}),
            'data_hash':split['source_train_sha256'],
            'split_policy_hash':hashlib.sha256((source/'configs/split_policy.json').read_bytes()).hexdigest(),
            'seeds':config['deep']['seeds'],'environment':environment(),'test_budget':test_budget,
            'research_range':[0,7859],'internal_holdout':[7860,8867],'final_simulation':[8868,9047]}


def canonical_state(root,ident):
    anchor,state,output=root/ANCHOR,root/STATE,root/OUTPUT
    if not anchor.exists():
        if state.exists() or output.exists() or (root/'artifacts/phase5_task_receipts.json').exists():
            raise RuntimeError('Real run evidence exists without canonical reservation; refusing a new attempt.')
        return 'UNUSED'
    if not state.exists():
        raise RuntimeError('Canonical state missing with consumed reservation; refusing rerun.')
    if read(anchor)['identity']!=ident or read(state)['identity']!=ident:
        raise RuntimeError('Run identity changed; resume requires the exact frozen run.')
    status=read(state)['state']
    if status not in ('PREPARED','TRAINING','TRAINING_COMPLETE','EVALUATING','COMPLETE','FAILED_PRE_EVAL','FAILED_POST_EVAL'):
        raise RuntimeError('Unknown Phase 5 execution state.')
    store=TaskStore(output/'cache',ident)
    if status!='PREPARED' and not store.journal.exists():
        raise RuntimeError('Canonical completion journal missing; refusing retraining.')
    inventory=store.inventory(recover=False)
    completed=read(state).get('completed_tasks',{})
    if any(inventory.get(k)!=v for k,v in completed.items()):
        raise RuntimeError('Completed task inventory differs from sealed run state.')
    return status


def preflight(root=None):
    root=(root or project_root()).resolve()
    if root!=project_root().resolve() or Path(git(root,'rev-parse','--show-toplevel')).resolve()!=root:
        raise ValueError('Real entry requires the canonical project root.')
    if git(root,'status','--porcelain','--untracked-files=all'):
        raise RuntimeError('Dirty worktree: real Phase 5 requires a clean worktree.')
    ident=identity(root,'real')
    if git(root,'cat-file','-t',f'refs/tags/{TAG}')!='tag' or git(root,'rev-parse',f'{TAG}^{{commit}}')!=ident['git_commit']:
        raise RuntimeError('Current commit must be the annotated frozen Phase 5 pipeline tag.')
    for name,digest in ident['implementation_sha256'].items():
        blob=subprocess.run(['git','-C',str(root),'show',f'{TAG}:{name}'],capture_output=True,check=True).stdout
        if hashlib.sha256(blob.replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise RuntimeError('Frozen Phase 5 code differs: '+name)
    if environment()!=json.loads((root/'configs/phase5_environment.json').read_text(encoding='utf-8')):
        raise RuntimeError('Execution environment differs from the Phase 5 freeze.')
    lock=verify_protocol_lock(root,verify_data=False)
    if ident['data_hash']!=lock['data_sha256']['data/raw/train.csv']:
        raise RuntimeError('Source data metadata differs.')
    seal=protected(root)
    if (root/'artifacts/phase4_execution_state_v1.1.json').read_bytes()!=(root/'reports/phase4_real_run/phase4_execution_state_v1.1.json').read_bytes():
        raise RuntimeError('Phase 4 completion state differs from its archive.')
    old=read(root/'artifacts/phase4_execution_state_v1.1.json')
    if old['state']!='completed' or old['run_type']!='real':
        raise RuntimeError('Phase 4 must be completed first.')
    config=protocol.load_config(root)
    if not config['holdout']['locked'] or not config['final_simulation']['locked']:
        raise RuntimeError('Held-out reservations must stay locked.')
    status=canonical_state(root,ident)
    return {'identity':ident,'pipeline_tag':TAG,'git_tag':git(root,'tag','--points-at','HEAD').splitlines(),
            'git_worktree_dirty':False,'read_only':True,'real_xy_read':False,'data_bytes_verified':False,
            'real_run_state':status,'protected_phase4_seal':hash_object(seal),'preflight':'passed'}


def publish(path,value):
    """Immutable deterministic final output; an identical resume is a no-op."""
    path=Path(path)
    data=value if isinstance(value,bytes) else encode(value)
    if path.exists():
        if path.read_bytes()!=data:
            raise RuntimeError('Existing result output differs; refusing replacement: '+path.name)
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        import os
        f.write(data); f.flush(); os.fsync(f.fileno())
    replace_atomic(tmp,path)


def write_outputs(tables,audit,destination,ident):
    outputs={}
    for key in ('predictions','folds','summary','incremental'):
        path=destination/f'phase5_{key}.csv'
        publish(path,tables[key].to_csv(index=False,lineterminator='\n').encode())
        outputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for key,value in [('audit',audit),('differences',tables['differences']),('primary_selection',tables['primary_selection'])]:
        path=destination/f'phase5_{key}.json'; publish(path,value)
        outputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'identity':ident,'run_type':ident['run_type'],'git_commit':ident['git_commit'],'git_tag':[TAG] if ident['run_type']=='real' else [],
              'git_worktree_dirty':False if ident['run_type']=='real' else None,
              'phase5_pipeline_code_hash':ident['phase5_pipeline_code_hash'],'state':'COMPLETE','output_sha256':outputs,
              'real_xy_read':ident['run_type']=='real','internal_holdout_read':False,'final_simulation_read':False}
    publish(destination/'phase5_run_manifest.json',manifest)
    return manifest


def archived_ab_check(predictions,root):
    old=pd.read_csv(root/'reports/phase4_real_run/phase4_predictions.csv',float_precision='round_trip')
    cols=['fold_id','pipeline','date_id']
    a=old.sort_values(cols)
    b=predictions[predictions.pipeline.isin(('A','B'))].sort_values(cols)
    if not np.array_equal(a[cols].to_numpy(),b[cols].to_numpy()) or not np.array_equal(a.prediction,b.prediction,equal_nan=True):
        raise RuntimeError('Phase 4 A/B benchmark continuity failed; stop before Phase 5 evaluation.')


def execute(root,data_root,destination,ident,workers,*,real=False):
    """Shared production entry used by synthetic CSV E2E and future real run."""
    from src.research.phase5_engine import Engine
    destination=Path(destination)
    state_path=root/STATE if real else destination/'execution_state.json'
    anchor_path=root/ANCHOR if real else destination/'reservation.json'
    lock_path=root/'artifacts/phase5_coordinator.lock' if real else destination/'coordinator.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(lock_path),timeout=0):
        if anchor_path.exists():
            if not state_path.exists() or read(anchor_path)['identity']!=ident or read(state_path)['identity']!=ident:
                raise RuntimeError('Reservation/state identity mismatch; refusing resume.')
            state=read(state_path)['state']
            if state=='COMPLETE':
                raise RuntimeError('Phase 5 already completed. Refusing a second statistical run.')
            if state not in ('PREPARED','TRAINING','TRAINING_COMPLETE','EVALUATING','FAILED_PRE_EVAL','FAILED_POST_EVAL'):
                raise RuntimeError('Unknown Phase 5 execution state; refusing resume.')
        else:
            if state_path.exists() or (destination/'phase5_run_manifest.json').exists() or (destination/'cache').exists():
                raise RuntimeError('Run evidence exists without canonical reservation.')
            atomic(anchor_path,{'identity':ident,'reservation_consumed':True})
            state='PREPARED'; atomic(state_path,{'identity':ident,'state':state})
        store=TaskStore(destination/'cache',ident)
        if state=='PREPARED':
            store.initialize()
        elif not store.journal.exists():
            raise RuntimeError('Completion journal missing after training began; refusing resume.')
        inventory=store.inventory()
        if any(inventory.get(k)!=v for k,v in read(state_path).get('completed_tasks',{}).items()):
            raise RuntimeError('Completed task inventory differs from sealed run state.')
        prediction_path=destination/'sealed_predictions.json'
        post=state in ('TRAINING_COMPLETE','EVALUATING','FAILED_POST_EVAL')
        try:
            if post:
                saved=read(state_path)
                if not prediction_path.exists() or hashlib.sha256(prediction_path.read_bytes()).hexdigest()!=saved.get('sealed_prediction_hash'):
                    raise RuntimeError('Sealed predictions missing/corrupt after evaluation boundary; no retraining permitted.')
            if real:
                # This branch is never called in Phase 5A. Byte hashes precede
                # bounded CSV parsing; no value at/after 7860 is parsed.
                verify_protocol_lock(root,verify_data=True)
            history=load_research_history(data_root,load_alpha_design(root))
            engine=Engine(history,protocol.load_config(root),load_alpha_design(root),store,ident['test_budget'])
            if not post:
                atomic(state_path,{'identity':ident,'state':'TRAINING'})
                with threadpool_limits(limits=1):
                    timing=engine.run_neural(workers)
                    predictions,audit=engine.predictions()
                if real:
                    archived_ab_check(predictions,root)
                payload={'predictions':predictions.to_dict(orient='list'),'audit':audit}
                publish(prediction_path,payload)
                sealed=hashlib.sha256(prediction_path.read_bytes()).hexdigest()
                atomic(destination/'runtime_log.json',{'workers':workers,'tasks':timing})
                atomic(state_path,{'identity':ident,'state':'TRAINING_COMPLETE','sealed_prediction_hash':sealed,
                                   'completed_tasks':store.inventory()})
                post=True
            saved=read(state_path)
            payload=read(prediction_path)
            predictions=pd.DataFrame(payload['predictions'])
            atomic(state_path,{**saved,'state':'EVALUATING'})
            tables=engine.evaluate(predictions)
            manifest=write_outputs(tables,payload['audit'],destination,ident)
            atomic(state_path,{**saved,'state':'COMPLETE','manifest_hash':hash_object(manifest)})
            return tables
        except BaseException as exc:
            current=read(state_path)
            atomic(state_path,{**current,'state':'FAILED_POST_EVAL' if post else 'FAILED_PRE_EVAL',
                               'error_type':type(exc).__name__})
            raise


def run_real(output=None,workers=4):
    root=project_root().resolve()
    if output is not None and Path(output).resolve()!=(root/OUTPUT).resolve():
        raise ValueError('Real output is canonical; an alternate directory cannot bypass run state.')
    checked=preflight(root)
    return execute(root,root,root/OUTPUT,checked['identity'],workers,real=True)


def run_synthetic(data_root,output,workers=4,test_budget=None):
    from src.research.alpha_calibration import artificial_data_only
    root=project_root().resolve(); data_root=Path(data_root).resolve(); output=Path(output).resolve()
    for p in (data_root,output):
        if p.is_relative_to(root/'data') or p.is_relative_to(root/'artifacts') or p==root:
            raise ValueError('Synthetic source/output must be separate from canonical real data/artifacts.')
    with artificial_data_only(root):
        ident=identity(root,'synthetic',data_root,test_budget)
        return execute(root,data_root,output,ident,workers)


def synthetic_destination(path):
    root=project_root().resolve(); path=Path(path).resolve()
    if path==root or path.is_relative_to(root/'data') or path.is_relative_to(root/'artifacts'):
        raise ValueError('Synthetic workspace must stay outside canonical data/artifacts.')
    return path


def verified_completed_synthetic(data_root,destination):
    """Read-only evidence reuse, with the SAME identity checks as execution.

    This is not a second run: it cannot train, replace outputs or recover tasks.
    Code changes must never silently reuse an older calibration summary.
    """
    from src.research.alpha_calibration import artificial_data_only
    root=project_root().resolve()
    source=synthetic_destination(data_root); destination=synthetic_destination(destination)
    with artificial_data_only(root):
        ident=identity(root,'synthetic',source)
        state=read(destination/'execution_state.json')
        anchor=read(destination/'reservation.json')
        manifest=read(destination/'phase5_run_manifest.json')
        if (state.get('state')!='COMPLETE' or manifest.get('state')!='COMPLETE'
                or any(saved.get('identity')!=ident for saved in (state,anchor,manifest))):
            raise RuntimeError('Completed calibration identity/state differs; refusing evidence reuse.')
        if hashlib.sha256((source/'data/raw/train.csv').read_bytes()).hexdigest()!=ident['data_hash']:
            raise RuntimeError('Completed calibration source bytes changed.')
        if hash_object(manifest)!=state.get('manifest_hash'):
            raise RuntimeError('Completed calibration manifest changed.')
        seal=destination/'sealed_predictions.json'
        if hashlib.sha256(seal.read_bytes()).hexdigest()!=state.get('sealed_prediction_hash'):
            raise RuntimeError('Completed calibration prediction seal changed.')
        inventory=TaskStore(destination/'cache',ident).inventory(recover=False)
        if inventory!=state.get('completed_tasks'):
            raise RuntimeError('Completed calibration task inventory changed.')
        for name,digest in manifest['output_sha256'].items():
            if Path(name).name!=name or hashlib.sha256((destination/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Completed calibration output changed: '+name)
        return pd.read_csv(destination/'phase5_summary.csv')
