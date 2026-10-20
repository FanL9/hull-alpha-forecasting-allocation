from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pytest
from src.research.phase5_tasks import TaskStore, atomic, encode, read

@pytest.fixture
def store(tmp_path):
    return TaskStore(tmp_path,{'run_type':'synthetic','code':'one','data':'one'})

def test_serialization_exact_arrays():
    x=np.arange(24,dtype=np.float32).reshape(2,3,4)
    import json
    from src.research.phase5_tasks import unpack
    y=unpack(json.loads(encode({'x':x,'nan':np.nan})))
    assert y['x'].dtype==x.dtype and y['x'].tobytes()==x.tobytes() and np.isnan(y['nan'])

def test_completed_task_reused_without_retraining(store):
    a=store.execute({'seed':11},lambda:{'prediction':np.arange(3)})
    b=store.execute({'seed':11},lambda:pytest.fail('Completed task retrained'))
    assert encode(a)==encode(b)

def test_atomic_incomplete_task_restarts(store):
    task={'seed':11}; key,path,_=store.paths(task)
    path.parent.mkdir(parents=True); path.with_suffix('.tmp').write_text('partial')
    assert store.execute(task,lambda:{'epoch':2})=={'epoch':2}

def test_completed_artifact_missing_rejected(store):
    task={'seed':11}; store.execute(task,lambda:{'epoch':2})
    store.paths(task)[1].unlink()
    with pytest.raises(RuntimeError,match='missing'):
        store.execute(task,lambda:pytest.fail('must not retrain'))

def test_completed_corruption_rejected(store):
    task={'seed':11}; store.execute(task,lambda:{'epoch':2})
    path=store.paths(task)[1]; saved=read(path); saved['result']['epoch']=3; atomic(path,saved)
    with pytest.raises(RuntimeError,match='Corrupt'):
        store.get(task)

def test_identity_change_refuses_resume(store):
    task={'seed':11}; store.execute(task,lambda:{'epoch':2})
    other=TaskStore(store.directory,{'run_type':'synthetic','code':'two','data':'one'})
    with pytest.raises(RuntimeError,match='identity'):
        other.get(task)

def test_receipt_change_rejected(store):
    task={'seed':11}; store.execute(task,lambda:{'epoch':2})
    atomic(store.paths(task)[2],{'fake':'receipt'})
    with pytest.raises(RuntimeError,match='receipt'):
        store.get(task)

def test_orphan_atomic_complete_recovers_receipt(store):
    task={'seed':11}; store.execute(task,lambda:{'epoch':2})
    store.paths(task)[2].unlink()
    assert store.get(task)=={'epoch':2} and store.paths(task)[2].exists()

def test_concurrent_task_exactly_once(store):
    calls=[]
    def run(_):
        return store.execute({'seed':11},lambda:(calls.append(1) or {'epoch':2}))
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(run,range(2)))==[{'epoch':2},{'epoch':2}]
    assert len(calls)==1

def test_inventory_rejects_deleted_complete(store):
    store.execute({'seed':11},lambda:{'epoch':2})
    store.paths({'seed':11})[1].unlink()
    with pytest.raises(RuntimeError,match='deleted'):
        store.inventory()

def test_deleting_whole_cache_cannot_regain_completed_tasks(store):
    import shutil
    task={'seed':11}; store.execute(task,lambda:{'epoch':2})
    shutil.rmtree(store.directory)
    assert store.journal.exists()
    with pytest.raises(RuntimeError,match='deleted'):
        store.inventory()
    with pytest.raises(RuntimeError,match='missing'):
        store.execute(task,lambda:pytest.fail('Deleted completed task retrained'))

def test_readonly_inventory_does_not_recover_receipt(store):
    task={'seed':11}; store.execute(task,lambda:{'epoch':2})
    receipt=store.paths(task)[2]; receipt.unlink()
    before=store.journal.read_bytes()
    assert len(store.inventory(recover=False))==1
    assert not receipt.exists() and store.journal.read_bytes()==before

def test_unfinished_compute_can_resume_from_same_initial_state(store):
    calls=[]
    def interrupted():
        calls.append(1); raise RuntimeError('simulated power interruption')
    with pytest.raises(RuntimeError): store.execute({'seed':11},interrupted)
    assert store.execute({'seed':11},lambda:{'epoch':2})=={'epoch':2}
    assert calls==[1]

def test_transient_windows_sharing_denial_retries_same_bytes(tmp_path,monkeypatch):
    from src.research import phase5_tasks as tasks
    original=tasks.os.replace; attempts=[]
    def transient(source,destination):
        attempts.append(Path(source).read_bytes())
        if len(attempts)<4:
            raise PermissionError('simulated Windows read handle denies rename')
        original(source,destination)
    from pathlib import Path
    monkeypatch.setattr(tasks.os,'replace',transient)
    tasks.atomic(tmp_path/'result.json',{'value':np.arange(5)})
    assert len(attempts)==4 and all(v==attempts[0] for v in attempts)
    np.testing.assert_array_equal(tasks.read(tmp_path/'result.json')['value'],np.arange(5))

def test_persistent_windows_denial_remains_fatal(tmp_path,monkeypatch):
    from src.research import phase5_tasks as tasks
    attempts=[]
    def denied(*args):
        attempts.append(1); raise PermissionError('persistent denial')
    monkeypatch.setattr(tasks.os,'replace',denied)
    monkeypatch.setattr(tasks.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError): tasks.atomic(tmp_path/'result.json',{'value':1})
    assert len(attempts)==101 and not (tmp_path/'result.json').exists()

def test_transient_windows_read_denial_retries_without_rewriting(tmp_path,monkeypatch):
    from src.research import phase5_tasks as tasks
    path=tmp_path/'result.json'; tasks.atomic(path,{'value':3})
    original=Path.read_bytes; calls=[]
    def transient(p):
        calls.append(1)
        if len(calls)<4: raise PermissionError('temporary rename/open sharing race')
        return original(p)
    monkeypatch.setattr(Path,'read_bytes',transient)
    assert tasks.read(path)=={'value':3} and len(calls)==4

def _concurrent_journal_worker(spec):
    directory,worker=spec
    store=TaskStore(directory,{'run_type':'synthetic','source':'concurrency_stress'})
    for i in range(30):
        task={'worker':worker,'index':i}
        assert store.execute(task,lambda:{'value':i})=={'value':i}
        assert store.get(task)=={'value':i}
    return 30

def test_multiprocess_journal_read_rename_stress(tmp_path):
    specs=[(str(tmp_path/'cache'),worker) for worker in range(4)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(_concurrent_journal_worker,specs))==[30]*4
    store=TaskStore(tmp_path/'cache',{'run_type':'synthetic','source':'concurrency_stress'})
    assert len(store.inventory())==120
