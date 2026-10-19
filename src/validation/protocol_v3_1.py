"""Additive metadata-only amendment; old v3 remains immutable."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import yaml
from src.utils.paths import project_root, utc_timestamp
from src.validation import protocol_v3 as parent

CONFIG='configs/protocol_v3_1.yaml'
DOCUMENT='docs/protocol_v3_1_engineered_alpha_fair_comparison.md'
LOCK='configs/protocol_v3_1_lock.json'
TAG='protocol-v3.1'
NORMATIVE=(CONFIG,DOCUMENT,'src/validation/protocol_v3_1.py','tests/test_protocol_v3_1.py')
NOISE_SEEDS=(101,211,307,401,503,601,701,809,907,1009,1103,1201,1301,1409,1511,1601,1709,1801,1901,2003)

def load_config(root: Path) -> dict:
    old=parent.load_config(root)
    c=yaml.safe_load((root/CONFIG).read_text(encoding='utf-8'))
    parent.require((c['protocol_version'],c['parent_protocol'],c['freeze_tag'])==('v3.1','v3',TAG),'Invalid v3.1 identity.')
    normalized=deepcopy(c)
    normalized.update(protocol_version=old['protocol_version'],parent_protocol=old['parent_protocol'],freeze_tag=old['freeze_tag'])
    normalized['regime']=old['regime']
    calibration=normalized.pop('synthetic_calibration')
    parent.require(normalized==old,'Only the authorized regime and prospective calibration amendment is allowed.')
    expected=deepcopy(old['regime'])
    expected.update(source='unsigned_training_V_family_mid_ecdf_summary',training_y_used=False,
        member_input='eligible_raw_V_levels_only',member_directions='all_positive_without_target',normalization='training_mid_ecdf',
        minimum_members=2,minimum_available_fraction=.5)
    parent.require(c['regime']==expected,'Regime must be the specified X-only V summary.')
    parent.require(tuple(calibration['noise_seeds'])==NOISE_SEEDS and calibration['planted_seeds']==[11,22,33]
        and calibration['cases']==['noise','raw','transformed','regime'] and calibration['full_outer_folds']==11
        and calibration['no_parameter_retuning_from_calibration'],'Calibration panel changed.')
    return c

def identity(root: Path) -> dict:
    old=parent.verify_lock(root)
    return {'parent_lock_sha256':parent.digest(root/parent.LOCK),'parent_normative_sha256':old['normative_sha256'],
        'phase3_pipeline_code_hash':old['phase3_pipeline_code_hash'],'protected_report_sha256':old['protected_report_sha256']}

def build_lock(root: Path) -> dict:
    load_config(root)
    return {'protocol_version':'v3.1','freeze_tag':TAG,'created_at_utc':utc_timestamp(),
        'real_phase4_phase5_executed':False,'real_data_read':False,
        'normative_sha256':{n:parent.digest(root/n) for n in NORMATIVE},**identity(root)}

def verify_lock(root: Path) -> dict:
    load_config(root)
    saved=json.loads((root/LOCK).read_text(encoding='utf-8'))
    parent.require(saved['protocol_version']=='v3.1' and saved['freeze_tag']==TAG,'Unknown v3.1 lock.')
    parent.require(set(saved['normative_sha256'])==set(NORMATIVE),'Incomplete v3.1 inventory.')
    for n,h in saved['normative_sha256'].items():parent.require(parent.digest(root/n)==h,f'Frozen v3.1 changed: {n}')
    for k,v in identity(root).items():parent.require(saved[k]==v,f'Inherited evidence changed: {k}')
    return saved

def freeze(root: Path) -> dict:
    if (root/LOCK).exists():return verify_lock(root)
    lock=build_lock(root)
    with (root/LOCK).open('x',encoding='utf-8',newline='\n') as f:json.dump(lock,f,indent=2);f.write('\n')
    return lock

def verify_tag(root: Path) -> str:
    lock=verify_lock(root)
    parent.require(parent.git(root,'cat-file','-t',f'refs/tags/{TAG}').strip()==b'tag','v3.1 tag must be annotated.')
    commit=parent.git(root,'rev-parse',f'{TAG}^{{commit}}').decode().strip()
    parent.require(parent.git(root,'merge-base',commit,'HEAD').decode().strip()==commit,'HEAD must descend from v3.1.')
    for n in (*NORMATIVE,LOCK):
        blob=parent.git(root,'show',f'{commit}:{n}').replace(b'\r\n',b'\n')
        parent.require(hashlib.sha256(blob).hexdigest()==lock['normative_sha256'].get(n,parent.digest(root/LOCK)),'v3.1 tag/hash mismatch.')
    parent.verify_tag(root)
    return commit

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=project_root())
    g=p.add_mutually_exclusive_group(required=True)
    for n in ['check-config','freeze','check','check-tag']:g.add_argument('--'+n,action='store_true')
    a=p.parse_args(argv);r=a.root.resolve()
    if a.check_config:load_config(r);value='config_verified'
    elif a.freeze:freeze(r);value='lock_created_or_verified'
    elif a.check_tag:value=verify_tag(r)
    else:verify_lock(r);value='lock_verified'
    print(json.dumps({'protocol':'v3.1','result':value,'real_data_read':False,'models_trained':False},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
