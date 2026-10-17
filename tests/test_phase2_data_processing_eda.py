"""Artificial fixtures test the firewall and causality, never market results."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.cli import main
from src.data.download import sha256
from src.data.preprocess import expanding_prior_median, missing_indicators, retain_missing
from src.data.validate_data import run_audit
from src.research.eda import (feature_eda, missing_intervals, target_eda, temporal_blocks,
                              trailing_statistics, xx_structure)
from src.research.pipeline import load_eda_settings, run_phase2
from src.utils.paths import project_root, write_json
from src.validation.time_split import (DEVELOPMENT_ROWS, load_development, prepare_split,
                                        validate_development)

SENTINEL = "SIMULATION_VALUES_MUST_NOT_ENTER_EDA_7f38"


@pytest.fixture(scope="module")
def settings():
    return load_eda_settings(project_root() / "configs/research.yaml")


@pytest.fixture
def reserved_fixture(tmp_path, config):
    """9048 synthetic rows with unparseable feature/target values after 8867."""
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    # Generate a synthetic 94-column schema: unit tests need no Kaggle files.
    features = [f"{family}{i}" for family, count in
                {"M": 18, "E": 20, "I": 9, "P": 13, "V": 13, "S": 12, "D": 9}.items()
                for i in range(1, count + 1)]
    columns = ["date_id", *features, *config["data"]["outcome_columns"]]
    outcome = set(config["data"]["outcome_columns"])
    rows = np.arange(9048)
    frame = pd.DataFrame({c: rows if c == "date_id" else
        (np.sin(rows / (i + 3)) * .01 if c in outcome else np.cos(rows / (i + 3)))
        for i, c in enumerate(columns)})
    frame.loc[:200, "M1"] = np.nan
    frame.loc[400:410, "M1"] = np.nan
    frame.loc[:, "D6"] = np.where(rows % 2, -1, 0)
    # Text sentinels make accidental full-file numeric analysis fail immediately.
    for column in columns:
        if column != "date_id":
            frame[column] = frame[column].astype(object)
            frame.loc[8868:, column] = SENTINEL
    frame.to_csv(raw / "train.csv", index=False)
    test_columns = ["date_id", *features, "is_scored", *[f"lagged_{c}" for c in config["data"]["outcome_columns"]]]
    pd.DataFrame({c: np.arange(8980, 8990) if c == "date_id" else [SENTINEL] * 10
                  for c in test_columns}).to_csv(raw / "test.csv", index=False)
    manifest = {"competition": config["data"]["competition"], "source": "official_kaggle_competition_cli",
        "files": [{"path": name, "sha256": sha256(raw / name)} for name in ["train.csv", "test.csv"]]}
    write_json(raw / "download_manifest.json", manifest)  # test double, not real provenance
    preserved = {}
    for relative in ["reports/data_audit.md", "docs/data_dictionary.md", "artifacts/data_summary.json", "artifacts/phase1_status.json"]:
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("Artificial Phase 1 preservation fixture\n", encoding="utf-8")
        preserved[relative] = sha256(p)
    policy = {"development_start": 0, "development_end": 8867, "development_rows": 8868,
        "competition_simulation_start": 8868, "competition_simulation_end": 9047,
        "competition_simulation_rows": 180, "competition_simulation_locked": True,
        "source_train_sha256": sha256(raw / "train.csv"), "preserved_phase1_sha256": preserved}
    write_json(tmp_path / "configs/split_policy.json", policy)
    (tmp_path / "configs/base.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "configs/research.yaml").write_bytes((project_root() / "configs/research.yaml").read_bytes())
    return tmp_path, raw, preserved


def test_exact_partition_identifiers_and_schema(reserved_fixture):
    root, raw, _ = reserved_fixture
    plan = prepare_split(root, raw / "train.csv", raw / "test.csv")
    development = range(plan["development_start"], plan["development_end"] + 1)
    simulation = range(plan["competition_simulation_start"], plan["competition_simulation_end"] + 1)
    assert len(development) == plan["development_rows"] == 8868
    assert len(simulation) == plan["competition_simulation_rows"] == 180
    assert max(development) == 8867 and min(simulation) == 8868 and max(simulation) == 9047
    assert set(development).isdisjoint(simulation)
    assert list(development) + list(simulation) == list(range(9048))
    assert plan["competition_simulation_locked"] is True
    assert len(plan["feature_columns"]) == 94
    assert len(plan["outcome_columns"]) == 3
    assert plan["official_test"]["role"] == "inference_schema_only"
    assert plan["official_test"]["used_as_model_test_set"] is False
    assert plan["official_test"]["value_statistics_computed"] is False
    frame = load_development(raw / "train.csv", plan)
    assert frame.shape == (8868, 98)
    assert list(frame.columns) == plan["train_columns"]
    assert frame.date_id.is_unique and frame.date_id.is_monotonic_increasing
    assert frame.M1.isna().sum() == 212
    pd.testing.assert_frame_equal(retain_missing(frame), frame)


def guarded_csv(monkeypatch, raw):
    original = pd.read_csv
    calls = []
    def guard(path, *args, **kwargs):
        path = Path(path)
        if path.parent == raw:
            calls.append((path.name, kwargs.copy()))
            identifiers_only = kwargs.get("usecols") == ["date_id"]
            bounded_values = path.name == "train.csv" and kwargs.get("nrows") == DEVELOPMENT_ROWS
            assert identifiers_only or bounded_values, "Forbidden raw value read"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guard)
    return calls


def test_pipeline_firewall_no_simulation_targets_in_any_analysis_artifact(reserved_fixture, config, monkeypatch):
    root, raw, preserved = reserved_fixture
    calls = guarded_csv(monkeypatch, raw)
    summary = run_phase2(root, config, root / "configs/research.yaml")
    assert summary["analysis_scope"]["rows"] == 8868
    assert summary["analysis_scope"]["end_date_id"] == 8867
    assert summary["analysis_scope"]["simulation_values_parsed"] is False
    assert summary["analysis_scope"]["feature_target_analysis_performed"] is False
    assert summary["execution"]["models_fitted"] == summary["execution"]["folds_executed"] == 0
    assert summary["target_only"]["rows"] == 8868 and summary["target_only"]["analysis_end"] == 8867
    for path in (root / "artifacts").glob("*"):
        if path.suffix in {".csv", ".json"}:
            text = path.read_text(encoding="utf-8")
            assert SENTINEL not in text
            if path.suffix == ".csv":
                data = pd.read_csv(path)
                for column in data:
                    if column.endswith("date_id"):
                        assert data[column].max() <= 8867
    assert all(sha256(root / p) == digest for p, digest in preserved.items())
    assert calls and all(c[1].get("usecols") == ["date_id"] for c in calls if c[0] == "test.csv")
    processed = pd.read_csv(root / "data/processed/development.csv")
    assert processed.shape == (8868, 98) and processed.date_id.max() == 8867
    assert processed.M1.isna().sum() == 212


def test_protected_phase1_preserves_history_and_never_downloads(reserved_fixture, config, monkeypatch, capsys):
    root, raw, preserved = reserved_fixture
    calls = guarded_csv(monkeypatch, raw)
    from src import cli
    monkeypatch.setattr(cli, "download_official", lambda *a, **k: pytest.fail("Unexpected download"))
    assert main(["phase1", "--root", str(root)]) == 0
    capsys.readouterr()
    assert all(sha256(root / p) == digest for p, digest in preserved.items())
    assert all(kwargs.get("usecols") == ["date_id"] for _, kwargs in calls)
    # Both notebook 01 and CLI consumers still have header-derived group counts.
    summary = run_audit(root, config)
    assert sum(map(len, summary["test"]["feature_groups"].values())) == 94


def test_protected_phase1_failure_does_not_overwrite_old_audit(reserved_fixture, monkeypatch, capsys):
    root, raw, preserved = reserved_fixture
    from src import cli
    monkeypatch.setattr(cli, "download_official", lambda *a, **k: pytest.fail("Unexpected download"))
    with (raw / "train.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered")
    assert main(["phase1", "--root", str(root)]) == 2
    capsys.readouterr()
    assert all(sha256(root / p) == digest for p, digest in preserved.items())


def test_split_cannot_change_silently(reserved_fixture):
    root, raw, _ = reserved_fixture
    before = prepare_split(root, raw / "train.csv", raw / "test.csv")
    assert prepare_split(root, raw / "train.csv", raw / "test.csv") == before
    bad = copy.deepcopy(before)
    bad["competition_simulation_start"] -= 1
    write_json(root / "artifacts/split_definition.json", bad)
    with pytest.raises(ValueError, match="refusing"):
        prepare_split(root, raw / "train.csv", raw / "test.csv")
    with pytest.raises(ValueError, match="stale"):
        load_development(raw / "train.csv", {**before, "source_train_sha256": "invalid"})


def test_tracked_policy_rejects_moved_boundary(reserved_fixture):
    root, raw, _ = reserved_fixture
    path = root / "configs/split_policy.json"
    policy = json.loads(path.read_text())
    policy["development_end"] = 8866
    write_json(path, policy)
    with pytest.raises(ValueError, match="policy differs"):
        prepare_split(root, raw / "train.csv", raw / "test.csv")


@pytest.mark.parametrize("change", ["shuffle", "duplicate", "fractional", "missing"])
def test_development_chronology_is_not_repaired(change):
    dates = np.arange(8868, dtype=float)
    if change == "shuffle":
        dates[[2, 3]] = dates[[3, 2]]
    elif change == "duplicate":
        dates[3] = dates[2]
    elif change == "fractional":
        dates[3] = 3.5
    else:
        dates[3] = np.nan
    frame = pd.DataFrame({"date_id": dates})
    with pytest.raises(ValueError, match="ordered"):
        validate_development(frame, {"train_columns": ["date_id"]})


def test_default_preserves_all_values_outcomes_and_nan():
    frame = pd.DataFrame({"M1": [None, 1., 3.], "forward_returns": [.1, -.1, .0],
                          "risk_free_rate": [.01] * 3, "market_forward_excess_returns": [.2, .0, None]})
    result = retain_missing(frame)
    pd.testing.assert_frame_equal(result, frame)
    result.loc[1, "M1"] = 99
    assert frame.loc[1, "M1"] == 1


def test_prior_median_has_no_future_or_initial_fallback():
    X = pd.DataFrame({"M1": [np.nan, 1., 100., np.nan, 5., np.nan], "E1": [np.nan] * 6})
    result = expanding_prior_median(X)
    assert pd.isna(result.M1.iloc[0]) and result.M1.iloc[3] == 50.5
    assert result.M1.iloc[5] == 5 and result.E1.isna().all()
    changed = X.copy()
    changed.loc[4:, "M1"] = -999
    pd.testing.assert_frame_equal(expanding_prior_median(changed).iloc[:4], result.iloc[:4])
    pd.testing.assert_frame_equal(missing_indicators(X), X.isna().astype("int8").add_suffix("_missing"))


@pytest.mark.parametrize("function", [expanding_prior_median, missing_indicators])
def test_candidate_preprocessing_rejects_outcomes(function):
    with pytest.raises(ValueError, match="X columns only"):
        function(pd.DataFrame({"M1": [1.], "market_forward_excess_returns": [.1]}))


def test_eda_rejects_any_simulation_identifier(settings):
    X = pd.DataFrame({"M1": [1., 2.]})
    dates = pd.Series([8867, 8868])
    with pytest.raises(ValueError, match="development"):
        feature_eda(X, dates, settings)
    with pytest.raises(ValueError, match="development"):
        target_eda(pd.Series([.1, .2], name="market_forward_excess_returns"), dates, settings)


def test_eda_entry_points_keep_x_and_y_separate(settings):
    dates = pd.Series(np.arange(300))
    X = pd.DataFrame({"M1": np.sin(dates), "E1": dates * .01})
    with pytest.raises(ValueError, match="X only"):
        feature_eda(X.assign(market_forward_excess_returns=.1), dates, settings)
    with pytest.raises(ValueError, match="target only"):
        target_eda(X.M1, dates, settings)
    structure = xx_structure(X, dates, 20, .95)
    assert list(structure["xx_correlation"].columns) == ["M1", "E1"]
    assert structure["xx_pair_counts"].loc["M1", "E1"] == 300


def test_future_perturbations_leave_earlier_rolling_and_drift_unchanged(settings):
    settings = {**settings, "time_block_rows": 100, "rolling_window_rows": 50,
                "rolling_min_observations": 20, "rolling_output_stride": 1}
    dates = pd.Series(np.arange(400))
    X = pd.DataFrame({"M1": np.sin(dates / 7), "E1": np.cos(dates / 3)})
    altered = X.copy()
    altered.loc[200:] = 1e9
    first = trailing_statistics(X, dates, settings)
    second = trailing_statistics(altered, dates, settings)
    pd.testing.assert_frame_equal(first[first.date_id < 200], second[second.date_id < 200])
    blocks_a, drift_a = temporal_blocks(X, dates, 100)
    blocks_b, drift_b = temporal_blocks(altered, dates, 100)
    pd.testing.assert_frame_equal(blocks_a[blocks_a.end_date_id < 200], blocks_b[blocks_b.end_date_id < 200])
    pd.testing.assert_frame_equal(drift_a[drift_a.end_date_id < 200], drift_b[drift_b.end_date_id < 200])
    y = pd.Series(np.sin(dates / 10) * .01, name="market_forward_excess_returns")
    changed = y.copy()
    changed.iloc[200:] = 10
    trace_a = target_eda(y, dates, settings)["tables"]["target_rolling_statistics"]
    trace_b = target_eda(changed, dates, settings)["tables"]["target_rolling_statistics"]
    pd.testing.assert_frame_equal(trace_a[trace_a.date_id < 200], trace_b[trace_b.date_id < 200])


def test_missing_intervals_are_contiguous_and_inclusive():
    X = pd.DataFrame({"M1": [None, None, 1., None, 2., None]})
    runs = missing_intervals(X, pd.Series(np.arange(6)))
    assert runs[["start_date_id", "end_date_id", "rows"]].values.tolist() == [[0, 1, 2], [3, 3, 1], [5, 5, 1]]
    assert runs.leading.tolist() == [True, False, False]
    assert runs.trailing.tolist() == [False, False, True]


def test_correlation_counts_and_undefined_constant_fields():
    X = pd.DataFrame({"M1": [1., 2., 3., None], "M2": [2., 4., None, 8.], "D1": [0.] * 4})
    result = xx_structure(X, pd.Series(np.arange(4)), 2, .95)
    assert result["xx_pair_counts"].loc["M1", "M2"] == 2
    assert len(result["high_xx_pairs"]) == 1
    assert result["xx_correlation"].D1.isna().all()


def test_future_design_choices_must_remain_undecided(tmp_path, settings):
    changed = copy.deepcopy(settings)
    changed["future_validation"]["fold_count"] = 7
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="undecided"):
        load_eda_settings(path)
