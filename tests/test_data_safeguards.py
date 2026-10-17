"""Test meaningful acquisition and audit failures using artificial fixtures."""

import json
import zipfile

import numpy as np
import pytest

from src.cli import main
from src.data import download
from src.data.download import AcquisitionError, COMPETITION, safe_extract, sha256, verify_manifest
from src.data.load_data import DataUnavailableError, load_csv
from src.data.preprocess import eligible_features
from src.data.validate_data import build_summary, inspect_frame, run_audit
from src.features.feature_groups import feature_family, group_features
from src.utils.paths import inside_root, write_json


def test_momentum_and_metadata_are_not_misclassified():
    groups = group_features(["M1", "MOM1", "D1", "date_id", "market_forward_excess_returns"])
    assert groups["M"] == ["M1"]
    assert groups["MOM"] == ["MOM1"]
    assert groups["D"] == ["D1"]
    assert feature_family("Market1") is None


def test_predictor_filter_excludes_outcomes_and_unverified_fields():
    columns = ["M1", "E2", "forward_returns", "risk_free_rate", "market_forward_excess_returns", "lagged_forward_returns", "date_id", "is_scored", "new_unknown"]
    assert eligible_features(columns, ["E2"]) == ["M1"]


def test_missing_input_never_creates_a_replacement(tmp_path):
    with pytest.raises(DataUnavailableError, match="missing"):
        load_csv(tmp_path / "train.csv")
    assert not (tmp_path / "train.csv").exists()


def test_duplicate_csv_header_is_not_silently_renamed(tmp_path):
    path = tmp_path / "artificial.csv"
    path.write_text("date_id,M1,M1\n0,1,2\n", encoding="utf-8")
    with pytest.raises(DataUnavailableError, match="Duplicate"):
        load_csv(path)


def test_empty_csv_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("date_id,M1\n", encoding="utf-8")
    with pytest.raises(DataUnavailableError, match="no data rows"):
        load_csv(path)


def test_missingness_uses_observed_counts_and_chronological_blocks(artificial_frames):
    train, _ = artificial_frames
    audit = inspect_frame(train, "date_id", 2)
    assert audit["column_details"]["M1"]["missing_count"] == 1
    assert audit["column_details"]["M1"]["missing_percentage"] == 25.0
    assert [part["missing_percentage"] for part in audit["family_missingness"]["MOM"]["time_blocks"]] == [100.0, 0.0]
    assert audit["feature_groups"]["MOM"] == ["MOM1"]


def test_bad_time_order_is_reported_without_reordering(artificial_frames, config):
    train, test = artificial_frames
    train = train.iloc[[0, 2, 1, 3]].copy()
    before = train["date_id"].tolist()
    summary = build_summary(train, test, config)
    assert summary["status"] == "validation_failed"
    assert train["date_id"].tolist() == before
    assert not summary["train"]["time"]["monotonic_increasing"]


@pytest.mark.parametrize("dates", [[0, 1, 1, 3], [0, 1, None, 3], [0, 1, "unknown", 3], [0, 1, np.inf, 3], [0, 1, 2.5, 3]])
def test_invalid_time_identifiers_are_blocking(dates, artificial_frames, config):
    train, test = artificial_frames
    train["date_id"] = dates
    assert build_summary(train, test, config)["status"] == "validation_failed"


def test_official_test_overlap_does_not_become_a_holdout_claim(artificial_frames, config):
    train, test = artificial_frames
    test["date_id"] = [2, 3]
    summary = build_summary(train, test, config)
    assert summary["time_relationship"]["overlapping_date_ids"] == 2
    assert summary["time_relationship"]["test_strictly_after_train"] is False
    assert any("independent final holdout" in message for message in summary["warnings"])


def test_missing_expected_target_is_an_error_not_a_constructed_label(artificial_frames, config):
    train, test = artificial_frames
    train = train.drop(columns="market_forward_excess_returns")
    summary = build_summary(train, test, config)
    assert summary["status"] == "validation_failed"
    assert "market_forward_excess_returns" not in train


def test_nonbinary_dummy_and_infinity_are_visible(artificial_frames, config):
    train, test = artificial_frames
    train.loc[0, "D1"] = 2
    train.loc[1, "M1"] = np.inf
    summary = build_summary(train, test, config)
    assert summary["status"] == "validation_failed"
    assert summary["train"]["nonbinary_dummy_columns"] == ["D1"]
    assert summary["train"]["column_details"]["M1"]["infinite_count"] == 1


def test_two_state_negative_encoding_is_binary_and_preserved(artificial_frames, config):
    train, test = artificial_frames
    train["D6"] = [-1, 0, -1, 0]
    summary = build_summary(train, test, config)
    assert summary["train"]["dummy_value_sets"]["D6"] == [-1, 0]
    assert "D6" in summary["train"]["non_01_dummy_columns"]
    assert "D6" not in summary["train"]["nonbinary_dummy_columns"]
    assert train["D6"].tolist() == [-1, 0, -1, 0]


@pytest.mark.parametrize("unsafe", ["../escape.txt", "nested/../../escape.txt", "C:/escape.txt", "..\\escape.txt", "/escape.txt"])
def test_zip_traversal_fails_before_any_extraction(tmp_path, unsafe):
    archive = tmp_path / "artificial.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("safe.txt", "artificial fixture")
        bundle.writestr(unsafe, "must not be written")
    destination = tmp_path / "raw"
    with pytest.raises(AcquisitionError, match="Unsafe"):
        safe_extract(archive, destination)
    assert not (destination / "safe.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_zip_symlink_rejected(tmp_path):
    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        entry = zipfile.ZipInfo("link")
        entry.create_system = 3
        entry.external_attr = 0o120777 << 16
        bundle.writestr(entry, "../escape")
    with pytest.raises(AcquisitionError, match="Unsafe"):
        safe_extract(archive, tmp_path / "raw")


def test_extraction_never_overwrites_changed_existing_file(tmp_path):
    archive = tmp_path / "artificial.zip"
    destination = tmp_path / "raw"
    destination.mkdir()
    (destination / "train.csv").write_text("existing", encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("train.csv", "different")
    with pytest.raises(AcquisitionError, match="differs"):
        safe_extract(archive, destination)
    assert (destination / "train.csv").read_text(encoding="utf-8") == "existing"


def test_unverified_local_data_cannot_pass_provenance(tmp_path):
    (tmp_path / "train.csv").write_text("artificial", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="manifest missing"):
        verify_manifest(tmp_path, "download_manifest.json", ["train.csv"])


def test_download_uses_official_cli_and_preserves_evaluation_files(tmp_path, monkeypatch):
    # Mock the server response only in this temporary test; never contact Kaggle.
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    archive = raw / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("train.csv", "date_id,M1\n0,1\n")
        bundle.writestr("test.csv", "date_id,M1\n1,2\n")
        bundle.writestr("kaggle_evaluation/test_api.py", "# artificial API fixture\n")
    executable = tmp_path / "kaggle.exe"
    monkeypatch.setattr(download, "kaggle_executable", lambda: executable)
    monkeypatch.setattr(download, "credential_status", lambda: {"configured": True})
    calls = []

    def fake_run(command, **kwargs):
        from subprocess import CompletedProcess
        calls.append(command)
        assert kwargs["capture_output"]
        assert kwargs["timeout"] == 30
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    manifest = download.download_official(raw, timeout=30)
    assert calls == [[str(executable), "competitions", "download", "-c", COMPETITION, "-p", str(raw)]]
    assert (raw / "kaggle_evaluation" / "test_api.py").is_file()
    assert len(manifest["files"]) == 3
    verify_manifest(raw, "download_manifest.json", ["train.csv", "test.csv"])
    (raw / "kaggle_evaluation" / "test_api.py").write_text("changed", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="hash differs"):
        verify_manifest(raw, "download_manifest.json", ["train.csv", "test.csv"])


def test_audit_end_to_end_and_hash_tamper_detection(tmp_path, artificial_frames, config):
    # This manifest is a test double in a temporary directory, never real-data provenance.
    train, test = artificial_frames
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    train.to_csv(raw / "train.csv", index=False)
    test.to_csv(raw / "test.csv", index=False)
    manifest = {"competition": COMPETITION, "source": "official_kaggle_competition_cli",
                "files": [{"path": name, "sha256": sha256(raw / name)} for name in ["train.csv", "test.csv"]]}
    write_json(raw / "download_manifest.json", manifest)
    summary = run_audit(tmp_path, config)
    assert summary["status"] == "complete"
    saved = json.loads((tmp_path / "artifacts" / "data_summary.json").read_text(encoding="utf-8"))
    assert saved["train"]["rows"] == 4
    assert saved["test"]["columns_count"] == 4
    assert "anonymous feature" in (tmp_path / "docs" / "data_dictionary.md").read_text(encoding="utf-8")
    assert "25.0000" in (tmp_path / "reports" / "data_audit.md").read_text(encoding="utf-8")
    with (raw / "train.csv").open("a", encoding="utf-8") as handle:
        handle.write("changed")
    with pytest.raises(AcquisitionError, match="hash differs"):
        run_audit(tmp_path, config)


def test_blocked_phase1_keeps_shapes_unknown(tmp_path, config, monkeypatch, capsys):
    (tmp_path / "configs").mkdir()
    import yaml
    (tmp_path / "configs" / "base.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(download, "kaggle_executable", lambda: tmp_path / "kaggle.exe")
    monkeypatch.setattr(download, "credential_status", lambda: {"configured": False})
    assert main(["phase1", "--root", str(tmp_path)]) == 2
    status = json.loads(capsys.readouterr().out)
    assert status["train_shape"] is None
    assert status["feature_group_counts"] is None
    saved = json.loads((tmp_path / "artifacts" / "data_summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "blocked"
    assert saved["train"] is None
    assert not (tmp_path / "data" / "raw" / "train.csv").exists()


def test_credential_values_never_appear_in_status(tmp_path, monkeypatch):
    monkeypatch.setattr(download.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "artificial-secret-do-not-log")
    status = download.credential_status()
    assert status["configured"]
    assert "artificial-secret-do-not-log" not in json.dumps(status)


def test_config_cannot_write_outside_project(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        inside_root(tmp_path, "../escaped.json")


def test_artifact_writer_rejects_nonstandard_json_numbers(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "invalid.json", {"statistic": float("nan")})
