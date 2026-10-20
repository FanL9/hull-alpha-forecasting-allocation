"""Content-addressed deterministic tasks with immutable completion receipts.

The OS lock is released even on process death. No partial checkpoint is ever
loaded. A completed artifact/receipt mismatch is an error, not a cache miss.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
import uuid

from filelock import FileLock
import numpy as np


def pack(value):
    if isinstance(value,np.ndarray):
        a = np.ascontiguousarray(value)
        return {'__array__':base64.b64encode(a.tobytes()).decode(), 'dtype':a.dtype.str,'shape':list(a.shape)}
    if isinstance(value,np.generic):
        return pack(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return {'__float__':str(value)}
    if isinstance(value,dict):
        return {str(k):pack(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [pack(v) for v in value]
    return value


def unpack(value):
    if isinstance(value,dict):
        if '__array__' in value:
            return np.frombuffer(base64.b64decode(value['__array__']),dtype=value['dtype']).reshape(value['shape']).copy()
        if '__float__' in value:
            return float(value['__float__'])
        return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list):
        return [unpack(v) for v in value]
    return value


def encode(value) -> bytes:
    return (json.dumps(pack(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def hash_object(value):
    return hashlib.sha256(encode(value)).hexdigest()


def read(path):
    # Windows may deny opening a destination while another process atomically
    # replaces it. Retry ONLY access denial, never malformed/corrupt JSON.
    for attempt in range(101):
        try:
            return unpack(json.loads(Path(path).read_bytes()))
        except PermissionError:
            if attempt==100:
                raise
            time.sleep(.05)


def replace_atomic(source,destination):
    """Windows readers/virus scanners can briefly deny FILE_SHARE_DELETE.

    Retry the SAME already-fsynced bytes, never recompute a completed task.
    Other errors and persistent access denial remain fatal after five seconds.
    """
    for attempt in range(101):
        try:
            os.replace(source,destination)
            return
        except PermissionError:
            if attempt==100:
                raise
            time.sleep(.05)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Short sibling names avoid Windows MAX_PATH failures in deep task stores.
    # The final content-addressed name and its bytes remain unchanged.
    tmp = path.parent/('.tmp-'+uuid.uuid4().hex)
    with tmp.open('xb') as stream:
        stream.write(encode(value)); stream.flush(); os.fsync(stream.fileno())
    replace_atomic(tmp,path)


class TaskStore:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.identity = identity

    @property
    def journal(self):
        # Real completion evidence is outside the removable output/cache tree.
        if self.identity.get('run_type')=='real':
            return self.directory.parent.parent/'phase5_task_receipts.json'
        return self.directory.with_name(self.directory.name+'_completion_journal.json')

    def initialize(self):
        self.journal.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(self.journal)+'.lock',timeout=60):
            if self.journal.exists():
                if read(self.journal)['identity']!=self.identity:
                    raise RuntimeError('Completion journal run identity changed.')
            else:
                if any((self.directory/'tasks').glob('*.json')) or any((self.directory/'receipts').glob('*.json')):
                    raise RuntimeError('Completion journal missing with existing tasks.')
                atomic(self.journal,{'identity':self.identity,'completed':{}})

    def commit_receipt(self,key,receipt):
        with FileLock(str(self.journal)+'.lock',timeout=60):
            journal=read(self.journal)
            if journal['identity']!=self.identity:
                raise RuntimeError('Completion journal identity mismatch.')
            existing=journal['completed'].get(key)
            if existing is not None and existing!=receipt:
                raise RuntimeError('Immutable completion journal mismatch.')
            if existing is None:
                journal['completed'][key]=receipt
                atomic(self.journal,journal)

    def paths(self, task):
        key = hash_object(task)
        return key,self.directory/'tasks'/(key+'.json'),self.directory/'receipts'/(key+'.json')

    def get(self, task, *, recover=True):
        key,path,receipt = self.paths(task)
        if not path.exists():
            recorded=self.journal.exists() and key in read(self.journal)['completed']
            if receipt.exists() or recorded:
                raise RuntimeError('Completed task artifact is missing; refusing retraining.')
            return None
        saved = read(path)
        if saved.get('status')!='completed' or saved.get('identity')!=self.identity or saved.get('task')!=task:
            raise RuntimeError('Task identity/completion mismatch; refusing cache reuse.')
        digest = hash_object(saved['result'])
        if digest!=saved.get('output_hash'):
            raise RuntimeError('Corrupt completed task output.')
        expected = {'task_id':key,'artifact_hash':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'run_identity_hash':hash_object(self.identity),'output_hash':digest,'status':'completed'}
        if receipt.exists():
            if read(receipt)!=expected:
                raise RuntimeError('Corrupt completed task receipt.')
        elif recover:
            # A crash between atomic artifact rename and receipt creation is
            # recoverable only after verifying the complete envelope above.
            atomic(receipt,expected)
        if self.journal.exists():
            journal=read(self.journal)
            if journal['identity']!=self.identity or (key in journal['completed'] and journal['completed'][key]!=expected):
                raise RuntimeError('Completion journal identity/artifact mismatch.')
        if recover:
            self.commit_receipt(key,expected)
        return saved['result']

    def execute(self, task, compute):
        self.initialize()
        key,_,_ = self.paths(task)
        lock = self.directory/'locks'/(key+'.lock')
        lock.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(lock),timeout=60):
            result = self.get(task)
            if result is not None:
                return result
            result = compute()
            _,path,_ = self.paths(task)
            atomic(path,{'identity':self.identity,'task':task,'status':'completed',
                         'output_hash':hash_object(result),'result':result})
            return self.get(task)

    def inventory(self, *, recover=True):
        recorded={}
        if self.journal.exists():
            journal=read(self.journal)
            if journal['identity']!=self.identity:
                raise RuntimeError('Completion journal run identity mismatch.')
            recorded=journal['completed']
        elif any((self.directory/'tasks').glob('*.json')) or any((self.directory/'receipts').glob('*.json')):
            raise RuntimeError('Completion journal missing with task evidence.')
        for key,receipt in recorded.items():
            artifact=self.directory/'tasks'/(key+'.json')
            if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest()!=receipt['artifact_hash']:
                raise RuntimeError('Completed task deleted or changed; refusing resume.')
        for path in sorted((self.directory/'tasks').glob('*.json')):
            self.get(read(path)['task'],recover=recover)
        receipts = self.directory/'receipts'
        rows = {}
        for path in sorted(receipts.glob('*.json')):
            receipt = read(path)
            artifact = self.directory/'tasks'/path.name
            if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest()!=receipt['artifact_hash']:
                raise RuntimeError('Completed task deleted or changed; refusing resume.')
            saved = read(artifact)
            if saved['identity']!=self.identity or hash_object(saved['result'])!=receipt['output_hash']:
                raise RuntimeError('Task run identity/output hash mismatch.')
            rows[path.stem] = receipt
        return {**recorded,**rows}
