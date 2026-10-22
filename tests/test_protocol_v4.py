"""Protocol v4 metadata contract; no market data or allocation score."""
import json
import shutil

import pytest
import yaml

from src.utils.paths import project_root
from src.validation import protocol_v4 as v4

ROOT = project_root()


def test_lock_verifies_and_records_no_scores():
    saved = v4.verify_lock(ROOT)
    assert saved['phase6_scores_computed'] is False and saved['real_data_read'] is False


def test_registered_rules_are_exact():
    c = v4.load_config(ROOT)
    assert c['candidates']['linear']['k'] == [0.10, 0.20, 0.35, 0.50, 0.75]
    assert c['selection']['near_best_margin'] == 0.05
    assert c['selection']['admissible']['min_blocks_beating_constant_1'] == 6
    assert c['selection']['if_none_admissible'] == 'constant_1'
    assert c['signal_standardization']['fit_rows'] == [5088, 7859]
    assert c['holdout']['fit_rows'] == [0, 7858] and c['holdout']['purge_row'] == 7859


@pytest.mark.parametrize('path,old,new', [
    ('configs/protocol_v4.yaml', 'near_best_margin: 0.05', 'near_best_margin: 0.10'),
    ('configs/protocol_v4.yaml', 'k: [0.10, 0.20, 0.35, 0.50, 0.75]', 'k: [0.10, 0.20, 0.35, 0.50, 1.00]'),
])
def test_changed_rules_are_rejected(tmp_path, monkeypatch, path, old, new):
    text = (ROOT/path).read_text(encoding='utf-8')
    assert old in text
    copy = tmp_path/path
    copy.parent.mkdir(parents=True)
    copy.write_text(text.replace(old, new, 1), encoding='utf-8')
    with pytest.raises(ValueError):
        v4.load_config(tmp_path)


def copied_root(tmp_path):
    for name in [*v4.NORMATIVE, *v4.SEALED, v4.LOCK, v4.protocol_v3_1.LOCK]:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    return tmp_path


@pytest.mark.parametrize('name', [v4.DOCUMENT, 'src/evaluation/kaggle_metric.py', 'reports/phase5_real_run/phase5_predictions.csv'])
def test_frozen_or_sealed_edit_breaks_lock(tmp_path, monkeypatch, name):
    root = copied_root(tmp_path)
    monkeypatch.setattr(v4.protocol_v3_1, 'verify_lock', lambda r: {})
    v4.verify_lock(root)
    with (root/name).open('a', encoding='utf-8') as handle:
        handle.write('edited\n')
    with pytest.raises(ValueError, match='changed'):
        v4.verify_lock(root)
