"""Protocol v4.1 holdout amendment lock: additive to v4, metadata only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.utils.paths import project_root, utc_timestamp
from src.validation import protocol_v4
from src.validation.protocol_v4 import byte_digest, require, text_digest

CONFIG = 'configs/protocol_v4_1.yaml'
DOCUMENT = 'docs/protocol_v4_1_holdout_amendment.md'
LOCK = 'configs/protocol_v4_1_lock.json'
TAG = 'protocol-v4.1'
NORMATIVE = (CONFIG, DOCUMENT, 'src/validation/protocol_v4_1.py', 'tests/test_protocol_v4_1.py')
SELECTION = 'reports/phase6_selection'
SEALED_SELECTION = tuple(f'{SELECTION}/{n}' for n in (
    'phase6_selection.json', 'phase6_candidate_scores.csv', 'phase6_block_scores.csv',
    'phase6_daily_positions.csv', 'phase6_run_manifest.json', 'phase6_selection_state.json', 'snapshot_seal.json'))


def load_config(root: Path) -> dict:
    c = yaml.safe_load((root/CONFIG).read_text(encoding='utf-8'))
    require((c['protocol_version'], c['parent_protocol'], c['freeze_tag']) == ('v4.1', 'v4', TAG), 'Invalid v4.1 identity.')
    parent = protocol_v4.load_config(root)
    selection = json.loads((root/SELECTION/'phase6_selection.json').read_text(encoding='utf-8'))
    s = c['phase6_selection']
    require(s['selected_mapping'] == selection['selection']['selected'] == c['holdout']['primary_allocation'],
            'Holdout primary allocation must be the sealed Phase 6A selection.')
    require(s['mu_A'] == selection['standardization']['mu'] and s['sigma_A'] == selection['standardization']['sigma'],
            'Frozen standardization differs from the sealed selection.')
    h = c['holdout']
    require(h['rows'] == parent['holdout']['evaluate_rows'] and h['fit_rows'] == parent['holdout']['fit_rows']
            and h['purge_row'] == parent['holdout']['purge_row'] and h['blocks'] == {'count': 4, 'rows': 252},
            'Holdout boundaries must inherit v4.')
    require(h['added_secondary'] == {'id': 'volatility_only', 'position': 'clip(c_t, 0, 2)',
                                     'c_t': 'protocol_v4.volatility_state.ratio'}, 'Only the volatility_only comparator is added.')
    f = c['forecast_confirmation']
    require((f['hypothesis'], f['alpha']) == ('one_sided_rank_ic_greater_than_zero', 0.05), 'Confirmation test changed.')
    b = c['paired_allocation_differences']['bootstrap']
    require((b['block_length'], b['repetitions'], b['seed']) == (20, 2000, 42), 'Bootstrap settings changed.')
    require(c['interpretation']['secondaries_never_replace_primary'] is True, 'Primary replacement forbidden.')
    return c


def build_lock(root: Path) -> dict:
    load_config(root)
    parent = protocol_v4.verify_lock(root)
    return {'protocol_version': 'v4.1', 'freeze_tag': TAG, 'created_at_utc': utc_timestamp(),
            'holdout_rows_read': False, 'normative_sha256': {n: text_digest(root/n) for n in NORMATIVE},
            'sealed_phase6_selection_sha256': {n: byte_digest(root/n) for n in SEALED_SELECTION},
            'parent_v4_lock_sha256': text_digest(root/protocol_v4.LOCK),
            'parent_v4_normative_sha256': parent['normative_sha256']}


def verify_lock(root: Path) -> dict:
    load_config(root)
    saved = json.loads((root/LOCK).read_text(encoding='utf-8'))
    require(saved['protocol_version'] == 'v4.1' and saved['freeze_tag'] == TAG, 'Unknown v4.1 lock.')
    require(set(saved['normative_sha256']) == set(NORMATIVE)
            and set(saved['sealed_phase6_selection_sha256']) == set(SEALED_SELECTION), 'Incomplete v4.1 inventory.')
    for n, h in saved['normative_sha256'].items():
        require(text_digest(root/n) == h, f'Frozen v4.1 changed: {n}')
    for n, h in saved['sealed_phase6_selection_sha256'].items():
        require(byte_digest(root/n) == h, f'Sealed Phase 6A selection changed: {n}')
    protocol_v4.verify_lock(root)
    require(text_digest(root/protocol_v4.LOCK) == saved['parent_v4_lock_sha256'], 'Parent v4 lock changed.')
    return saved


def freeze(root: Path) -> dict:
    if (root/LOCK).exists():
        return verify_lock(root)
    (root/LOCK).write_text(json.dumps(build_lock(root), indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return verify_lock(root)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    args = parser.parse_args(argv)
    saved = freeze(project_root()) if args.freeze else verify_lock(project_root())
    print(json.dumps({'protocol': 'v4.1', 'result': 'lock_verified', 'freeze_tag': saved['freeze_tag'],
                      'holdout_rows_read': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
