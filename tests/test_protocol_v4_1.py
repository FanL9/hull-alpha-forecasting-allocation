"""Protocol v4.1 metadata contract; no holdout rows."""
import shutil

import pytest

from src.utils.paths import project_root
from src.validation import protocol_v4_1 as v41

ROOT = project_root()


def test_lock_verifies_before_holdout():
    saved = v41.verify_lock(ROOT)
    assert saved['holdout_rows_read'] is False


def test_amendment_binds_sealed_selection_and_adds_only_volatility_only():
    c = v41.load_config(ROOT)
    assert c['holdout']['primary_allocation'] == 'volscaled_k0.10'
    assert c['holdout']['added_secondary']['id'] == 'volatility_only'
    assert c['forecast_confirmation']['alpha'] == 0.05
    assert c['interpretation']['secondaries_never_replace_primary'] is True


def copied(tmp_path):
    names = [*v41.NORMATIVE, *v41.SEALED_SELECTION, v41.LOCK, 'configs/protocol_v4.yaml', 'configs/protocol_v4_lock.json']
    for name in names:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    return tmp_path


@pytest.mark.parametrize('old,new', [('selected_mapping: volscaled_k0.10', 'selected_mapping: volscaled_k0.50'),
                                     ('alpha: 0.05', 'alpha: 0.10'),
                                     ('id: volatility_only', 'id: volatility_plus')])
def test_changed_amendment_rejected(tmp_path, old, new):
    root = copied(tmp_path)
    path = root/v41.CONFIG
    text = path.read_text(encoding='utf-8')
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
    with pytest.raises(ValueError):
        v41.load_config(root)


def test_changed_sealed_selection_breaks_lock(tmp_path, monkeypatch):
    root = copied(tmp_path)
    monkeypatch.setattr(v41.protocol_v4, 'verify_lock', lambda r: {})
    original = v41.protocol_v4.load_config
    monkeypatch.setattr(v41.protocol_v4, 'load_config', lambda r: original(ROOT))
    v41.verify_lock(root)
    with (root/v41.SELECTION/'phase6_daily_positions.csv').open('a') as handle:
        handle.write('x\n')
    with pytest.raises(ValueError, match='Sealed Phase 6A selection changed'):
        v41.verify_lock(root)
