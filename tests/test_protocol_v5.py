"""Protocol v5 metadata contract; no market rows."""
import shutil

import pytest

from src.utils.paths import project_root
from src.validation import protocol_v4_1, protocol_v5 as v5

ROOT = project_root()


def test_lock_verifies_before_any_v5_score():
    saved = v5.verify_lock(ROOT)
    assert saved['v5_scores_computed'] is False and saved['final_rows_read'] is False


def test_registered_inventory():
    c = v5.load_config(ROOT)
    assert c['forecasters'] == ['A', 'Ridge', 'DynRidge', 'C', 'D']
    assert v5.candidate_order()[:2] == ['volscaled_k0.10/A', 'volscaled_k0.10/Ridge']
    assert len(v5.candidate_order()) == 10 and c['final_simulation']['ranks_tracks'] is False


def copied(tmp_path):
    names = [*v5.NORMATIVE, *v5.EVIDENCE, v5.LOCK, *protocol_v4_1.NORMATIVE, *protocol_v4_1.SEALED_SELECTION,
             protocol_v4_1.LOCK, 'configs/protocol_v4.yaml', 'configs/protocol_v4_lock.json']
    for name in names:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    return tmp_path


@pytest.mark.parametrize('old,new', [('min_blocks: 8', 'min_blocks: 7'),
                                     ('half_life_rows: 504', 'half_life_rows: 252'),
                                     ('forecasters: [A, Ridge, DynRidge, C, D]', 'forecasters: [A, Ridge, DynRidge, C, D, GRU]'),
                                     ('ranks_tracks: false', 'ranks_tracks: true')])
def test_changed_protocol_rejected(tmp_path, old, new):
    root = copied(tmp_path)
    path = root/v5.CONFIG
    text = path.read_text(encoding='utf-8')
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
    with pytest.raises(ValueError):
        v5.load_config(root)


def test_changed_evidence_breaks_lock(tmp_path, monkeypatch):
    root = copied(tmp_path)
    monkeypatch.setattr(v5.protocol_v4_1, 'verify_lock', lambda r: {})
    v5.verify_lock(root)
    with (root/v5.EVIDENCE[1]).open('a') as handle:
        handle.write('9999,A,0.0\n')
    with pytest.raises(ValueError, match='Sealed v5 evidence changed'):
        v5.verify_lock(root)
