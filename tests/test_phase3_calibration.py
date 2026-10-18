"""Broad, fixed-seed software calibration checks, not estimates of market Alpha.

Acceptance bounds are intentionally loose finite-panel regression checks. Five
null runs cannot certify nominal FDR. No seed must have zero BH discoveries,
and no weak signal is required to pass BH. Bootstrap is fully executed.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.research.alpha_calibration import (NOISE_SEEDS, POWER_SEEDS, POWER_STRENGTHS,
                                           make_calibration_history, run_calibrations)
from src.utils.paths import project_root


@pytest.fixture(scope='module')
def calibrations(tmp_path_factory):
    output = tmp_path_factory.mktemp('phase3_calibration')
    return run_calibrations(project_root(), output, workers=3), output


def test_noise_panel_all_97_hypotheses_and_full_bootstrap(calibrations):
    result, _ = calibrations
    noise = result.noise
    assert noise['seed'].tolist() == list(NOISE_SEEDS)
    assert noise['single_hypothesis_count'].eq(97).all()
    assert noise['testable_count'].eq(97).all()
    assert noise['family_hypothesis_count'].eq(7).all() and noise['global_hypothesis_count'].eq(104).all()
    assert noise['bootstrap_valid_repetitions_min'].eq(2000).all()
    assert noise['mean_abs_ic'].between(0, .04).all()
    assert abs(noise['mean_signed_ic'].mean()) < .015
    # Permit occasional discoveries; reject widespread, systematic inflation.
    assert (noise['bh_pass_count'] <= 2).sum() >= 4
    assert noise['bh_pass_count'].max() <= 8 and noise['bh_pass_count'].sum() <= 15
    assert (noise['by_pass_count'] <= noise['bh_pass_count']).all()


def test_weak_population_rank_ic_estimates_and_both_signs(calibrations):
    result, _ = calibrations
    power = result.power
    assert set(power['seed']) == set(POWER_SEEDS)
    assert len(power) == len(POWER_SEEDS) * len(POWER_STRENGTHS)
    assert power['single_hypothesis_count'].eq(97).all() and power['testable_count'].eq(97).all()
    assert power['bootstrap_valid_repetitions_min'].eq(2000).all()
    means = power.groupby('injected_strength')['estimated_rank_ic'].mean()
    assert (np.abs(means.index.to_numpy() - means.to_numpy()) < .04).all()
    assert abs(means.loc[0.]) < .04
    assert (means.loc[[.03, .05, .08]] > 0).all()
    assert (means.loc[[-.03, -.05, -.08]] < 0).all()
    assert set(power['bh_pass']) <= {True, False}  # weak FAIL is allowed


def test_nw_pvalues_decrease_overall_with_absolute_signal(calibrations):
    power = calibrations[0].power.copy()
    power['absolute_strength'] = power['injected_strength'].abs()
    medians = power.groupby('absolute_strength')['nw_p_value'].median().reindex([0., .03, .05, .08])
    assert np.isfinite(medians).all()
    assert medians.iloc[-1] < medians.iloc[0]
    assert (np.diff(medians.to_numpy()) < 0).sum() >= 2


def test_calibration_summary_and_provenance_are_saved(calibrations):
    result, output = calibrations
    manifest = json.loads((output / 'calibration_manifest.json').read_text())
    assert manifest['run_type'] == 'synthetic' and manifest['market_data_read'] is False
    assert manifest['bootstrap_repetitions'] == 2000
    assert manifest['protocol_version'] == 'phase2.5-v2'
    assert manifest['csv_line_ending'] == 'LF'
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
        assert b'\r\n' not in (output / name).read_bytes()
    assert {'seed', 'mean_abs_ic', 'bh_pass_count', 'by_pass_count'} <= set(result.noise.columns)
    assert {'injected_strength', 'estimated_rank_ic', 'nw_p_value', 'bh_pass'} <= set(result.power.columns)
    assert set(pd.read_csv(output / 'weak_signal_summary.csv')['injected_strength']) == set(POWER_STRENGTHS.values())


def test_artificial_null_generator_is_complete_and_deterministic():
    root = project_root()
    first = make_calibration_history(root, NOISE_SEEDS[0])
    second = make_calibration_history(root, NOISE_SEEDS[0])
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (7860, 98) and not first.isna().any().any()
    assert set(first['D1']) == {0., 1.} and first.attrs['phase3_data_kind'] == 'synthetic'
