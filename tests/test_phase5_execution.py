from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research import phase5_execution as ex
from src.research.phase5_tasks import atomic, read, hash_object
from src.research.phase5_synthetic import synthetic_project
from src.utils.paths import project_root

def test_protected_old_results_and_protocol_locks():
    ex.protected(project_root())
    from src.validation.protocol_freeze import verify_protocol_lock
    verify_protocol_lock(project_root(),verify_data=False)
    ex.protocol.verify_tag(project_root())

def test_real_alternative_output_cannot_bypass(tmp_path):
    with pytest.raises(ValueError,match='canonical'):
        ex.run_real(tmp_path)

def test_dirty_worktree_guard_before_any_data_read(monkeypatch):
    monkeypatch.setattr(ex,'git',lambda root,*args: str(root) if args==('rev-parse','--show-toplevel') else ' M file')
    with pytest.raises(RuntimeError,match='Dirty worktree'):
        ex.preflight()

def test_tag_guard_before_data_access(monkeypatch):
    root=project_root()
    def fake_git(r,*args):
        if args==('rev-parse','--show-toplevel'): return str(root)
        if args[0]=='status': return ''
        if args[0]=='cat-file': return 'tag'
        return 'different'
    monkeypatch.setattr(ex,'git',fake_git)
    monkeypatch.setattr(ex,'identity',lambda *a:{'git_commit':'frozen'})
    with pytest.raises(RuntimeError,match='annotated frozen'):
        ex.preflight()

def test_canonical_reservation_cannot_be_reset_by_deleting_state(tmp_path):
    atomic(tmp_path/ex.ANCHOR,{'identity':{'code':'one'}})
    with pytest.raises(RuntimeError,match='missing'):
        ex.canonical_state(tmp_path,{'code':'one'})

def test_canonical_state_without_anchor_refused(tmp_path):
    atomic(tmp_path/ex.STATE,{'identity':{},'state':'FAILED_PRE_EVAL'})
    with pytest.raises(RuntimeError,match='without canonical'):
        ex.canonical_state(tmp_path,{})

def test_orphan_completion_journal_alone_blocks_new_real_attempt(tmp_path):
    atomic(tmp_path/'artifacts/phase5_task_receipts.json',{'completed':{}})
    with pytest.raises(RuntimeError,match='without canonical'):
        ex.canonical_state(tmp_path,{})

@pytest.mark.parametrize('state',['PREPARED','TRAINING','TRAINING_COMPLETE','EVALUATING','COMPLETE','FAILED_PRE_EVAL','FAILED_POST_EVAL'])
def test_same_run_states_and_identity_guard(tmp_path,state):
    ident={'code':'one','data':'one'}
    atomic(tmp_path/ex.ANCHOR,{'identity':ident})
    atomic(tmp_path/ex.STATE,{'identity':ident,'state':state})
    from src.research.phase5_tasks import TaskStore
    TaskStore(tmp_path/ex.OUTPUT/'cache',ident).initialize()
    assert ex.canonical_state(tmp_path,ident)==state
    with pytest.raises(RuntimeError,match='identity'):
        ex.canonical_state(tmp_path,{**ident,'code':'two'})

def test_immutable_output_no_replacement(tmp_path):
    p=tmp_path/'output.json'; ex.publish(p,{'x':1}); original=p.read_bytes()
    ex.publish(p,{'x':1}); assert p.read_bytes()==original
    with pytest.raises(RuntimeError,match='replacement'):
        ex.publish(p,{'x':2})

def test_missing_completion_journal_cannot_regain_run(tmp_path):
    ident={'code':'one'}
    atomic(tmp_path/ex.ANCHOR,{'identity':ident})
    atomic(tmp_path/ex.STATE,{'identity':ident,'state':'TRAINING'})
    with pytest.raises(RuntimeError,match='journal missing'):
        ex.canonical_state(tmp_path,ident)

def test_cli_no_real_test_budget():
    from src.research.phase5 import main
    with pytest.raises(SystemExit):
        main(['--run-real-phase5','--synthetic-test-epochs','1'])

@pytest.fixture(scope='module')
def e2e_run(tmp_path_factory):
    from src.research.phase5 import main
    directory=tmp_path_factory.mktemp('phase5_e2e')
    source=synthetic_project(directory/'source','nonlinear',307)
    output=directory/'run'
    reads=[]
    code=pd.read_csv.__code__
    def observe(frame,event,arg):
        if frame.f_code is code and event=='call':
            path=frame.f_locals.get('filepath_or_buffer')
            if path is not None and Path(path).resolve()==(source/'data/raw/train.csv').resolve():
                reads.append({'nrows':frame.f_locals.get('nrows')})
        if frame.f_code is code and event=='return' and reads:
            reads[-1]['parsed_rows']=len(arg)
            reads[-1]['last_date']=int(arg.date_id.iloc[-1])
            reads[-1]['all_numeric']=all(pd.api.types.is_numeric_dtype(t) for t in arg.dtypes)
            sys.setprofile(None)
    sys.setprofile(observe)
    try:
        main(['--synthetic','--data-root',str(source),'--output',str(output),'--workers','4','--synthetic-test-epochs','1'])
    finally:
        sys.setprofile(None)
    return source,output,reads

def test_production_entry_e2e_real_reader_with_sentinels(e2e_run):
    _,output,reads=e2e_run
    assert reads==[{'nrows':7860,'parsed_rows':7860,'last_date':7859,'all_numeric':True}]
    assert read(output/'execution_state.json')['state']=='COMPLETE'
    manifest=read(output/'phase5_run_manifest.json')
    assert manifest['run_type']=='synthetic' and not manifest['real_xy_read']
    assert not manifest['internal_holdout_read'] and not manifest['final_simulation_read']
    preds=pd.read_csv(output/'phase5_predictions.csv')
    assert len(preds)==11*252*11 and preds.date_id.min()==5088 and preds.date_id.max()==7859
    assert set(preds.pipeline)=={'A','B','Ridge','GRU','TCN','C','D','residual_GRU','residual_TCN','historical_mean','zero_forecast'}
    assert preds.groupby(['fold_id','pipeline']).size().eq(252).all()
    assert {'target','clip_lower','clip_upper','status'}<=set(preds)

def test_e2e_completed_state_rejects_second_run(e2e_run):
    source,output,_=e2e_run
    with pytest.raises(RuntimeError,match='already completed'):
        ex.run_synthetic(source,output,workers=1,test_budget=1)

def test_e2e_post_eval_replay_never_trains(e2e_run,monkeypatch):
    source,output,_=e2e_run
    from src.research.phase5_engine import Engine
    saved=read(output/'execution_state.json')
    atomic(output/'execution_state.json',{**saved,'state':'FAILED_POST_EVAL'})
    monkeypatch.setattr(Engine,'run_neural',lambda *a:pytest.fail('Post-eval resume retrained'))
    original=(output/'phase5_predictions.csv').read_bytes()
    ex.run_synthetic(source,output,workers=1,test_budget=1)
    assert (output/'phase5_predictions.csv').read_bytes()==original

def test_e2e_post_eval_missing_seal_never_retrains(e2e_run,monkeypatch):
    source,output,_=e2e_run
    saved=read(output/'execution_state.json')
    atomic(output/'execution_state.json',{**saved,'state':'FAILED_POST_EVAL','sealed_prediction_hash':'bad'})
    with pytest.raises(RuntimeError,match='Sealed predictions'):
        ex.run_synthetic(source,output,workers=1,test_budget=1)
    atomic(output/'execution_state.json',saved)

def test_complete_run_worker_1_vs_parallel_exact_outputs(e2e_run):
    source,output,_=e2e_run
    serial=output.with_name('serial_repeat')
    ex.run_synthetic(source,serial,workers=1,test_budget=1)
    manifest=read(output/'phase5_run_manifest.json')
    repeated=read(serial/'phase5_run_manifest.json')
    assert manifest==repeated
    for name in manifest['output_sha256']:
        assert (output/name).read_bytes()==(serial/name).read_bytes()
    assert (output/'sealed_predictions.json').read_bytes()==(serial/'sealed_predictions.json').read_bytes()
    a={p.name:p.read_bytes() for p in (output/'cache/tasks').glob('*.json')}
    b={p.name:p.read_bytes() for p in (serial/'cache/tasks').glob('*.json')}
    assert a==b

def test_same_completed_tasks_pre_eval_resume_without_training(e2e_run,monkeypatch):
    import shutil
    import src.models.deep_alpha as models
    source,output,_=e2e_run
    destination=output.with_name('pre_eval_resume')
    shutil.copytree(output,destination)
    state=read(destination/'execution_state.json')
    atomic(destination/'execution_state.json',{**state,'state':'FAILED_PRE_EVAL'})
    monkeypatch.setattr(models,'train',lambda *a,**k:pytest.fail('Completed task retrained'))
    ex.run_synthetic(source,destination,workers=1,test_budget=1)
    assert (destination/'phase5_predictions.csv').read_bytes()==(output/'phase5_predictions.csv').read_bytes()

def test_completed_calibration_reuse_checks_full_identity(e2e_run,monkeypatch):
    source,output,_=e2e_run
    # E2E has a one-epoch test budget, so it must NEVER count as production
    # calibration evidence even when every output/task checksum is valid.
    with pytest.raises(RuntimeError,match='identity/state differs'):
        ex.verified_completed_synthetic(source,output)
    old_identity=ex.identity
    monkeypatch.setattr(ex,'identity',lambda root,run_type,data_root=None,test_budget=None:
                        old_identity(root,run_type,data_root,1))
    assert len(ex.verified_completed_synthetic(source,output))==9

def test_completed_evidence_refuses_changed_outputs_and_source(e2e_run,monkeypatch):
    source,output,_=e2e_run
    old_identity=ex.identity
    monkeypatch.setattr(ex,'identity',lambda root,run_type,data_root=None,test_budget=None:
                        old_identity(root,run_type,data_root,1))
    path=output/'phase5_summary.csv'; original=path.read_bytes()
    path.write_bytes(original+b'changed')
    try:
        with pytest.raises(RuntimeError,match='output changed'):
            ex.verified_completed_synthetic(source,output)
    finally:
        path.write_bytes(original)
    path=source/'data/raw/train.csv'; original=path.read_bytes()
    path.write_bytes(original+b'\n')
    try:
        with pytest.raises(RuntimeError,match='source bytes changed'):
            ex.verified_completed_synthetic(source,output)
    finally:
        path.write_bytes(original)
