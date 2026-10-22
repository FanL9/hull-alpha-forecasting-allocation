"""Protocol v4 (Phase 6 allocation) lock: metadata only, never market rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from src.utils.paths import project_root, utc_timestamp
from src.validation import protocol_v3_1

CONFIG = 'configs/protocol_v4.yaml'
DOCUMENT = 'docs/protocol_v4_phase6_allocation.md'
LOCK = 'configs/protocol_v4_lock.json'
TAG = 'protocol-v4'
NORMATIVE = (CONFIG, DOCUMENT, 'src/validation/protocol_v4.py', 'tests/test_protocol_v4.py',
    'src/evaluation/kaggle_metric.py', 'configs/official_metric.yaml')
# Sealed Phase 5 evidence that defines A and its research OOS forecasts (byte hashes).
SEALED = ('reports/phase5_real_run/phase5_primary_selection.json', 'reports/phase5_real_run/phase5_predictions.csv',
    'reports/phase5_real_run/snapshot_seal.json')
K_VALUES = [0.10, 0.20, 0.35, 0.50, 0.75]


def text_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def byte_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_config(root: Path) -> dict:
    c = yaml.safe_load((root/CONFIG).read_text(encoding='utf-8'))
    require(c['protocol_version'] == 'v4' and c['freeze_tag'] == TAG, 'Invalid v4 identity.')
    require(c['primary_forecast_pipeline'] == 'A', 'Phase 6 maps the frozen primary A only.')
    require(c['data']['calibration_range'] == [5088, 7859] and c['data']['internal_holdout'] == [7860, 8867]
        and c['data']['final_simulation'] == [8868, 9047], 'Phase 6 boundaries changed.')
    cand = c['candidates']
    require(cand['linear']['k'] == K_VALUES and cand['vol_scaled']['k'] == K_VALUES, 'Registered k grid changed.')
    order = [f'linear_k{k:.2f}' for k in K_VALUES] + [f'volscaled_k{k:.2f}' for k in K_VALUES]
    require(cand['simplicity_order'] == order and cand['candidate_count_excluding_baseline'] == 10, 'Candidate inventory changed.')
    v = c['volatility_state']
    require((v['short_window_rows'], v['long_window_rows'], v['ratio_clip']) == (21, 252, [0.5, 1.5]), 'Volatility state changed.')
    s = c['selection']
    require((s['admissible']['min_blocks_beating_constant_1'], s['near_best_margin'], s['if_none_admissible'])
        == (6, 0.05, 'constant_1'), 'Selection rule changed.')
    require(c['costs']['costs_change_selection'] is False and c['holdout']['runs'] == 1
        and c['final_simulation']['runs'] == 1, 'Cost/held-out rules changed.')
    return c


def build_lock(root: Path) -> dict:
    load_config(root)
    parent = protocol_v3_1.verify_lock(root)
    return {'protocol_version': 'v4', 'freeze_tag': TAG, 'created_at_utc': utc_timestamp(),
        'phase6_scores_computed': False, 'real_data_read': False,
        'normative_sha256': {n: text_digest(root/n) for n in NORMATIVE},
        'sealed_phase5_sha256': {n: byte_digest(root/n) for n in SEALED},
        'parent_v3_1_lock_sha256': text_digest(root/protocol_v3_1.LOCK),
        'parent_v3_1_normative_sha256': parent['normative_sha256']}


def verify_lock(root: Path) -> dict:
    load_config(root)
    saved = json.loads((root/LOCK).read_text(encoding='utf-8'))
    require(saved['protocol_version'] == 'v4' and saved['freeze_tag'] == TAG, 'Unknown v4 lock.')
    require(set(saved['normative_sha256']) == set(NORMATIVE) and set(saved['sealed_phase5_sha256']) == set(SEALED),
        'Incomplete v4 inventory.')
    for n, h in saved['normative_sha256'].items():
        require(text_digest(root/n) == h, f'Frozen v4 changed: {n}')
    for n, h in saved['sealed_phase5_sha256'].items():
        require(byte_digest(root/n) == h, f'Sealed Phase 5 evidence changed: {n}')
    protocol_v3_1.verify_lock(root)
    require(text_digest(root/protocol_v3_1.LOCK) == saved['parent_v3_1_lock_sha256'], 'Parent v3.1 lock changed.')
    return saved


def freeze(root: Path) -> dict:
    if (root/LOCK).exists():
        return verify_lock(root)
    lock = build_lock(root)
    (root/LOCK).write_text(json.dumps(lock, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return verify_lock(root)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    args = parser.parse_args(argv)
    root = project_root()
    saved = freeze(root) if args.freeze else verify_lock(root)
    print(json.dumps({'protocol': 'v4', 'result': 'lock_verified', 'freeze_tag': saved['freeze_tag'],
        'real_data_read': False, 'phase6_scores_computed': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
