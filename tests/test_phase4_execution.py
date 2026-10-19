"""Canonical execution guards tested in temporary projects; no real run."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research import phase4_execution as ex
from src.research.phase4_calibration import synthetic_run,summarize_null,recovery
from src.research.phase4 import main
from src.research.phase4_engine import Phase4Tables

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_project(tmp_path,monkeypatch):
    identity={'run_type':'real','git_commit':'abc','git_worktree_dirty':False,
        'implementation_sha256':{},'research_boundary':[0,7859]}
    monkeypatch.setattr(ex,'project_root',lambda:tmp_path)
    monkeypatch.setattr(ex,'execution_identity',lambda root,kind:identity.copy())
    def fake_git(root,*args):
        if args==('rev-parse','--show-toplevel'):return str(tmp_path)
        if args[:2]==('cat-file','-t'):return 'tag'
        if args[0]=='rev-parse':return 'abc'
        raise AssertionError(args)
    monkeypatch.setattr(ex,'git',fake_git)
    monkeypatch.setattr(ex.protocol,'load_config',lambda root:{'research':{'start':0,'end':7859},
        'holdout':{'start':7860},'final_simulation':{'start':8868}})
    return tmp_path,identity


def test_preflight_readonly_no_writes(fake_project):
    root,_=fake_project
    before=list(root.rglob('*'))
    checked=ex.preflight(root)
    assert checked['preflight']=='passed' and not checked['real_xy_read']
    assert checked['read_only'] and not checked['data_bytes_verified']
    assert list(root.rglob('*'))==before


def test_dirty_worktree_refused(fake_project):
    root,identity=fake_project;identity['git_worktree_dirty']=True
    with pytest.raises(RuntimeError,match='Dirty worktree'):ex.preflight(root)


def test_wrong_pipeline_commit_refused(fake_project):
    root,identity=fake_project;identity['git_commit']='different'
    with pytest.raises(RuntimeError,match='differs'):ex.preflight(root)


def test_lightweight_tag_refused(fake_project,monkeypatch):
    root,_=fake_project;original=ex.git
    monkeypatch.setattr(ex,'git',lambda r,*a:'commit' if a[0]=='cat-file' else original(r,*a))
    with pytest.raises(RuntimeError,match='annotated'):ex.preflight(root)


def test_code_hash_refused(fake_project,monkeypatch):
    root,identity=fake_project;identity['implementation_sha256']={'source.py':hashlib.sha256(b'expected').hexdigest()}
    monkeypatch.setattr(ex.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=b'changed'))
    with pytest.raises(RuntimeError,match='code differs'):ex.preflight(root)


def test_alternative_root_refused(fake_project):
    root,_=fake_project
    with pytest.raises(ValueError,match='canonical'):ex.preflight(root/'other')


@pytest.mark.parametrize('state',['reserved','running','completed','failed_or_interrupted'])
def test_any_existing_reservation_blocks(tmp_path,state):
    p=tmp_path/ex.STATE;p.parent.mkdir();p.write_text(json.dumps({'state':state,'run_type':'real'}))
    with pytest.raises(RuntimeError,match='reservation'):ex.assert_unused(tmp_path)


def test_canonical_real_manifest_blocks_output_bypass(tmp_path):
    p=tmp_path/ex.MANIFEST;p.parent.mkdir(parents=True);p.write_text('{"run_type":"real"}')
    with pytest.raises(RuntimeError,match='already been executed'):ex.assert_unused(tmp_path)
    with pytest.raises(ValueError,match='canonical'):ex.run_real(tmp_path,tmp_path/'new-output')


def test_canonical_nonreal_manifest_not_overwritten(tmp_path):
    p=tmp_path/ex.MANIFEST;p.parent.mkdir(parents=True);p.write_text('{"run_type":"synthetic"}')
    with pytest.raises(RuntimeError,match='occupied'):ex.assert_unused(tmp_path)


def test_existing_empty_output_refused_before_loading(tmp_path):
    (tmp_path/ex.REAL_OUTPUT).mkdir(parents=True)
    with pytest.raises(RuntimeError,match='already exists'):ex.assert_unused(tmp_path)


def test_atomic_reservation_has_one_winner(tmp_path):
    def claim():
        try:ex.reserve(tmp_path,{'git_commit':'abc'});return True
        except (RuntimeError,FileExistsError):return False
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:claim(),range(2)))
    assert sum(results)==1
    saved=json.loads((tmp_path/ex.STATE).read_text())
    assert saved['reservation_consumed'] and saved['state']=='reserved' and saved['run_type']=='real'


def test_failed_attempt_preserves_reservation(fake_project,monkeypatch):
    root,_=fake_project
    def no_data(*a,**k):raise RuntimeError('synthetic injected failure before data access')
    monkeypatch.setattr(ex,'verify_protocol_lock',no_data)
    with pytest.raises(RuntimeError,match='injected failure'):ex.run_real(root)
    state=json.loads((root/ex.STATE).read_text())
    assert state['state']=='failed_or_interrupted' and not state['real_xy_read']
    with pytest.raises(RuntimeError,match='reservation'):ex.preflight(root)


def test_preflight_before_any_data_verification(fake_project,monkeypatch):
    root,identity=fake_project;identity['git_worktree_dirty']=True
    def forbidden(*a,**k):raise AssertionError('Data verification must follow guards')
    monkeypatch.setattr(ex,'verify_protocol_lock',forbidden)
    with pytest.raises(RuntimeError,match='Dirty'):ex.run_real(root)
    assert not (root/ex.STATE).exists()


def test_code_identity_changes_with_source(tmp_path):
    for name in ex.CODE:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source\n')
    first=ex.code_identity(tmp_path)
    (tmp_path/ex.CODE[0]).write_text('changed\n')
    assert first['phase4_pipeline_code_hash']!=ex.code_identity(tmp_path)['phase4_pipeline_code_hash']


def test_synthetic_manifest_identity_has_all_required_fields():
    identity=ex.execution_identity(ROOT,'synthetic')
    for key in ['run_type','git_commit','git_tag','git_worktree_dirty','phase4_pipeline_code_hash',
        'protocol_tag','protocol_commit','data_hash','research_boundary','environment']:
        assert key in identity
    assert identity['run_type']=='synthetic' and not identity['real_xy_read']
    assert not identity['internal_holdout_read'] and not identity['final_simulation_read']


def test_synthetic_rejects_unmarked_frame(tmp_path):
    with pytest.raises(ValueError,match='artificial'):synthetic_run(ROOT,pd.DataFrame(),tmp_path/'out')


def test_synthetic_api_forbids_canonical_output():
    h=pd.DataFrame();h.attrs['phase4_data_kind']='synthetic'
    with pytest.raises(ValueError,match='separate'):synthetic_run(ROOT,h,ROOT/'artifacts/phase4')


def test_cli_requires_explicit_mode():
    with pytest.raises(SystemExit):main([])


@pytest.mark.parametrize('path',['artifacts/phase4','data/raw/synthetic'])
def test_cli_synthetic_forbids_canonical_paths(path):
    with pytest.raises(SystemExit):main(['--synthetic','--output',str(ROOT/path)])


def test_null_summary_fixed20_not_all_seeds_must_zero():
    rows=[]
    for pipeline in ['A','B']:
        for seed,value in enumerate(np.linspace(-.03,.03,20)):
            rows.append({'case':'noise','pipeline':pipeline,'mean_fold_ic':value,'finite_folds':11})
    result=summarize_null(pd.DataFrame(rows))
    assert not result.engineering_alarm.any() and (result.seeds==20).all()
    assert (result['min']<0).all() and (result['max']>0).all()
    rows[0]['finite_folds']=10
    with pytest.raises(RuntimeError,match='Incomplete'):summarize_null(pd.DataFrame(rows))


def test_systematic_null_bias_raises_alarm_without_retuning():
    rows=[{'case':'noise','pipeline':p,'mean_fold_ic':.2,'finite_folds':11} for p in ['A','B'] for s in range(20)]
    result=summarize_null(pd.DataFrame(rows))
    assert result.engineering_alarm.all() and (result.alarm_threshold==.02).all()


def test_output_inventory_and_no_overwrite(tmp_path):
    folds=pd.DataFrame({'pipeline':['A','B'],'fold_id':[1,1]})
    selection=pd.DataFrame({'pipeline':['A','B'],'candidate_id':['raw/M1/Level','raw/M1/Level']})
    tables=Phase4Tables(folds,selection,pd.DataFrame({'id':[1]}),pd.DataFrame({'rho':[.99]}),
        {'test':'synthetic'},pd.DataFrame({'pipeline':['A','B']}),pd.DataFrame({'prediction':[0.]}))
    out=tmp_path/'synthetic';ex.write_outputs(tables,out,{'run_type':'synthetic'})
    assert len(list(out.iterdir()))==10
    manifest=json.loads((out/'phase4_run_manifest.json').read_text())
    assert manifest['run_type']=='synthetic' and len(manifest['output_sha256'])==9
    with pytest.raises(FileExistsError):ex.write_outputs(tables,out,{'run_type':'synthetic'})
