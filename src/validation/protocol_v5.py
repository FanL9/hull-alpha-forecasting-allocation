"""Protocol v5 (competition track) lock: metadata only, never market rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.utils.paths import project_root, utc_timestamp
from src.validation import protocol_v4, protocol_v4_1
from src.validation.protocol_v4 import byte_digest, require, text_digest

CONFIG = 'configs/protocol_v5.yaml'
DOCUMENT = 'docs/protocol_v5_competition_track.md'
LOCK = 'configs/protocol_v5_lock.json'
TAG = 'protocol-v5'
NORMATIVE = (CONFIG, DOCUMENT, 'src/validation/protocol_v5.py', 'tests/test_protocol_v5.py')
# Sealed forecast evidence the competition track is chosen from (byte hashes).
EVIDENCE = ('reports/phase5_real_run/phase5_predictions.csv', 'reports/phase6_holdout_public/holdout_forecasts_sealed.csv',
            'reports/phase6_holdout_public/phase6_holdout_state.json')
FORECASTERS = ['A', 'Ridge', 'DynRidge', 'C', 'D']
K_VALUES = [0.10, 0.20]


def candidate_order() -> list[str]:
    return [f'volscaled_k{k:.2f}/{f}' for k in K_VALUES for f in FORECASTERS]


def load_config(root: Path) -> dict:
    c = yaml.safe_load((root/CONFIG).read_text(encoding='utf-8'))
    require((c['protocol_version'], c['freeze_tag']) == ('v5', TAG) and c['research_track_unchanged'] is True,
            'Invalid v5 identity.')
    e = c['evidence']
    require(e['oos_range'] == [5088, 8867] and e['oos_rows'] == 3780 and e['loader_rows'] == 8868
            and e['blocks'] == {'count': 15, 'rows': 252, 'research': 11, 'holdout': 4} and e['final_rows_read'] is False,
            'Evidence boundaries changed.')
    require(c['forecasters'] == FORECASTERS, 'Forecaster set changed.')
    d = c['dynridge']
    require((d['alphas'], d['sample_weight']['half_life_rows'], d['scaler']['clip'], d['alpha_ties'])
            == ([0.1, 1.0, 10.0, 100.0], 504, [-5, 5], 'larger_alpha'), 'DynRidge specification changed.')
    cand = c['candidates']
    require(cand['forecast_free'] == ['constant_1', 'volatility_only'] and cand['forecast_based']['k'] == K_VALUES,
            'Candidate inventory changed.')
    s = c['selection']
    require((s['min_blocks'], s['near_best_margin'], s['costs_change_selection']) == (8, 0.05, False),
            'Selection rule changed.')
    f = c['final_simulation']
    require((f['rows'], f['fit_rows'], f['purge_row'], f['runs']) == ([8868, 9047], [0, 8866], 8867, 1)
            and f['ranks_tracks'] is False and f['feeds_back'] is False, 'Final Simulation rule changed.')
    research = protocol_v4_1.load_config(root)['phase6_selection']
    require(f['tracks']['research']['mapping'] == research['selected_mapping'] == 'volscaled_k0.10',
            'The research track must stay the sealed v4.1 primary.')
    return c


def build_lock(root: Path) -> dict:
    c = load_config(root)
    protocol_v4_1.verify_lock(root)
    evidence = {n: byte_digest(root/n) for n in EVIDENCE}
    require(evidence[EVIDENCE[1]] == c['evidence']['holdout_forecasts_sha256'], 'Holdout forecasts differ from the seal.')
    return {'protocol_version': 'v5', 'freeze_tag': TAG, 'created_at_utc': utc_timestamp(),
            'v5_scores_computed': False, 'final_rows_read': False,
            'normative_sha256': {n: text_digest(root/n) for n in NORMATIVE}, 'evidence_sha256': evidence,
            'parent_v4_1_lock_sha256': text_digest(root/protocol_v4_1.LOCK),
            'parent_v4_lock_sha256': text_digest(root/protocol_v4.LOCK)}


def verify_lock(root: Path) -> dict:
    load_config(root)
    saved = json.loads((root/LOCK).read_text(encoding='utf-8'))
    require(saved['protocol_version'] == 'v5' and saved['freeze_tag'] == TAG, 'Unknown v5 lock.')
    require(set(saved['normative_sha256']) == set(NORMATIVE) and set(saved['evidence_sha256']) == set(EVIDENCE),
            'Incomplete v5 inventory.')
    for n, h in saved['normative_sha256'].items():
        require(text_digest(root/n) == h, f'Frozen v5 changed: {n}')
    for n, h in saved['evidence_sha256'].items():
        require(byte_digest(root/n) == h, f'Sealed v5 evidence changed: {n}')
    protocol_v4_1.verify_lock(root)
    require(text_digest(root/protocol_v4_1.LOCK) == saved['parent_v4_1_lock_sha256'], 'Parent v4.1 lock changed.')
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
    print(json.dumps({'protocol': 'v5', 'result': 'lock_verified', 'freeze_tag': saved['freeze_tag'],
                      'final_rows_read': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
