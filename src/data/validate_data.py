"""Descriptive schema, missingness and time-order auditing of actual inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.download import sha256, verify_manifest
from src.data.load_data import DataUnavailableError, load_official_pair
from src.features.feature_groups import FAMILY_LABELS, column_role, feature_family, group_features
from src.utils.paths import inside_root, utc_timestamp, write_json


def _scalar(value: Any) -> Any:
    """Convert pandas/NumPy scalars into strict JSON-compatible values."""
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def inspect_frame(frame: pd.DataFrame, date_column: str, blocks: int = 4) -> dict[str, Any]:
    """Inspect observed values without changing their order or fitting anything."""
    if blocks < 1:
        raise ValueError("missingness_blocks must be positive.")
    if frame.empty:
        raise DataUnavailableError("Cannot audit an empty dataframe.")
    groups = group_features(frame.columns)
    numeric = frame.select_dtypes(include="number")
    infinite = {column: int(np.isinf(numeric[column].to_numpy(dtype=float)).sum()) for column in numeric}
    columns: dict[str, Any] = {}
    for column in frame:
        missing = frame[column].isna()
        columns[column] = {
            "dtype": str(frame[column].dtype), "role": column_role(column),
            "family": feature_family(column), "missing_count": int(missing.sum()),
            "missing_percentage": float(missing.mean() * 100),
            "unique_nonmissing": int(frame[column].nunique(dropna=True)),
            "infinite_count": infinite.get(column, 0),
        }
    boundaries = np.linspace(0, len(frame), min(blocks, len(frame)) + 1, dtype=int)
    family_missing: dict[str, Any] = {}
    for family, names in groups.items():
        if not names:
            family_missing[family] = {"columns": 0, "missing_percentage": None, "time_blocks": []}
            continue
        values = frame[names].isna()
        family_missing[family] = {
            "columns": len(names), "missing_count": int(values.sum().sum()),
            "missing_percentage": float(values.to_numpy().mean() * 100),
            "rows_with_any_missing": int(values.any(axis=1).sum()),
            "time_blocks": [{"start_row": int(start), "end_row_exclusive": int(end),
                             "missing_percentage": float(values.iloc[start:end].to_numpy().mean() * 100)}
                            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)],
        }
    time: dict[str, Any] = {"present": date_column in frame}
    if date_column in frame:
        dates = frame[date_column]
        converted = pd.to_numeric(dates, errors="coerce")
        finite = converted.notna() & np.isfinite(converted.to_numpy(dtype=float, na_value=np.nan))
        clean = converted[finite]
        time.update({
            "missing_count": int(dates.isna().sum()),
            "nonnumeric_or_nonfinite_count": int((~finite & dates.notna()).sum()),
            "duplicate_count": int(dates.duplicated().sum()),
            "monotonic_increasing": bool(converted.is_monotonic_increasing),
            "integer_like": bool(finite.all() and (clean % 1 == 0).all()),
            "minimum": _scalar(clean.min()) if len(clean) else None,
            "maximum": _scalar(clean.max()) if len(clean) else None,
            "non_unit_step_count": int((clean.diff().dropna() != 1).sum()),
        })
    return {"rows": len(frame), "columns_count": len(frame.columns), "columns": list(frame.columns),
            "total_missing_count": int(frame.isna().sum().sum()),
            "columns_with_missing_values": int(frame.isna().any().sum()),
            "column_details": columns, "feature_groups": groups, "family_missingness": family_missing,
            "time": time, "all_missing_columns": [c for c in frame if frame[c].isna().all()],
            "constant_columns": [c for c in frame if frame[c].nunique(dropna=True) == 1],
            "nonnumeric_features": [c for names in groups.values() for c in names if c not in numeric],
            "dummy_value_sets": {c: [_scalar(value) for value in sorted(frame[c].dropna().unique(), key=str)]
                                 if frame[c].nunique(dropna=True) <= 20 else None for c in groups["D"]},
            "non_01_dummy_columns": [c for c in groups["D"] if not frame[c].dropna().isin([0, 1]).all()],
            "nonbinary_dummy_columns": [c for c in groups["D"] if frame[c].nunique(dropna=True) > 2]}


def build_summary(train: pd.DataFrame, test: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Describe real frames and classify blocking errors separately from warnings."""
    date_column = config["data"]["date_column"]
    blocks = config["audit"]["missingness_blocks"]
    train_audit = inspect_frame(train, date_column, blocks)
    test_audit = inspect_frame(test, date_column, blocks)
    errors: list[str] = []
    warnings: list[str] = []
    for name, details in (("train", train_audit), ("test", test_audit)):
        time = details["time"]
        if not time["present"]:
            errors.append(f"{name}: required date_id column missing.")
        elif (time["missing_count"] or time["duplicate_count"] or not time["monotonic_increasing"]
              or time["nonnumeric_or_nonfinite_count"] or not time["integer_like"]):
            errors.append(f"{name}: date_id must be numeric, finite, integer-like, unique and monotonically increasing; input has not been reordered.")
        if details["nonnumeric_features"]:
            errors.append(f"{name}: nonnumeric anonymous features require investigation: {details['nonnumeric_features']}")
        if any(item["infinite_count"] for item in details["column_details"].values()):
            errors.append(f"{name}: infinite numeric values detected.")
        if details["all_missing_columns"]:
            warnings.append(f"{name}: all-missing columns: {details['all_missing_columns']}")
        if details["constant_columns"]:
            warnings.append(f"{name}: constant nonmissing columns: {details['constant_columns']}")
        if details["non_01_dummy_columns"]:
            observed = {c: details["dummy_value_sets"][c] for c in details["non_01_dummy_columns"]}
            warnings.append(f"{name}: D-family encoding is not 0/1: {observed}; preserve observed values. Two distinct values still constitute a binary encoding.")
        if details["nonbinary_dummy_columns"]:
            warnings.append(f"{name}: D-family fields have more than two observed values: {details['nonbinary_dummy_columns']}")
        unknown = [c for c in details["columns"] if column_role(c).startswith("unclassified")]
        if unknown:
            warnings.append(f"{name}: unclassified fields excluded from predictors: {unknown}")
    for column in config["data"]["outcome_columns"]:
        if column not in train:
            errors.append(f"train: expected outcome column absent: {column}; official schema must take precedence.")
        elif not pd.api.types.is_numeric_dtype(train[column]):
            errors.append(f"train: outcome column is not numeric: {column}")
        elif train[column].isna().any():
            warnings.append(f"train: outcome has missing labels: {column}; do not impute returns.")
    train_groups = {c for values in train_audit["feature_groups"].values() for c in values}
    test_groups = {c for values in test_audit["feature_groups"].values() for c in values}
    absent = sorted(train_groups - test_groups)
    if absent:
        warnings.append(f"test: training feature columns absent: {absent}")
    if not train_audit["feature_groups"]["MOM"]:
        warnings.append("No MOM-prefixed features are present in the actual training CSV; do not invent a momentum family or infer momentum meaning from another anonymous field.")
    relationship: dict[str, Any] = {"overlapping_date_ids": None, "test_strictly_after_train": None}
    if date_column in train and date_column in test:
        relationship["overlapping_date_ids"] = int(len(set(train[date_column].dropna()) & set(test[date_column].dropna())))
        if train_audit["time"]["maximum"] is not None and test_audit["time"]["minimum"] is not None:
            relationship["test_strictly_after_train"] = bool(test_audit["time"]["minimum"] > train_audit["time"]["maximum"])
        if relationship["test_strictly_after_train"] is False:
            warnings.append("Official test rows are not strictly after training rows; do not treat the supplied test file as an independent final holdout.")
    warnings.append("Official test.csv is an API/demo input until official evaluation files establish its meaning; final research holdout must come from the chronological tail of labelled training history.")
    return {"schema_version": 1, "status": "validation_failed" if errors else "complete",
            "generated_at_utc": utc_timestamp(), "competition": config["data"]["competition"],
            "train": train_audit, "test": test_audit, "time_relationship": relationship,
            "train_only_columns": sorted(set(train) - set(test)), "test_only_columns": sorted(set(test) - set(train)),
            "errors": errors, "warnings": warnings, "final_holdout": "not_created_or_evaluated"}


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(cell(value) for value in row) + " |" for row in rows]


def render_report(summary: dict[str, Any]) -> str:
    """Produce a reviewable report with all columns and actual observed counts."""
    lines = ["> Historical snapshot. For current research rules and validation settings, see [research_protocol.md](../docs/research_protocol.md).", "",
             "# Official data audit", "", f"Status: **{summary['status']}**", "",
             f"Generated (UTC): {summary['generated_at_utc']}", "",
             "Source: [Kaggle — Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data).", ""]
    if summary["status"] == "blocked":
        lines += [summary["reason"], "", "No real CSV was read. Dataset shape, columns, feature groups, missingness and time relationships are **unknown**, not zero.", "",
                  "No market data or audit statistics were generated from examples or third-party sources.", "",
                  "Configure Kaggle authentication, confirm competition access, then run `python -m src.cli phase1`.", ""]
        return "\n".join(lines)
    lines += ["## Dataset shape", ""] + _table(["File", "Rows", "Columns"], [
        [name, summary[name]["rows"], summary[name]["columns_count"]] for name in ("train", "test")]) + [""]
    lines += ["## Missingness overview", ""] + _table(["File", "Columns with missing values", "Missing cells"], [
        [name, summary[name]["columns_with_missing_values"], summary[name]["total_missing_count"]] for name in ("train", "test")]) + [""]
    for name in ("train", "test"):
        audit = summary[name]
        lines += [f"## {name}: feature families", "", "Family labels follow the project taxonomy; individual fields remain anonymous features.", ""]
        lines += _table(["Prefix", "Family", "Count", "Actual columns"], [
            [family, label, len(audit["feature_groups"][family]), ", ".join(audit["feature_groups"][family]) or "none"]
            for family, label in FAMILY_LABELS.items()]) + [""]
        lines += [f"## {name}: full column and missingness audit", ""]
        lines += _table(["Column", "dtype", "Role", "Missing", "Missing %", "Inf"], [
            [column, item["dtype"], item["role"], item["missing_count"], f"{item['missing_percentage']:.4f}", item["infinite_count"]]
            for column, item in audit["column_details"].items()]) + [""]
        lines += [f"## {name}: missingness by family and input-order block", ""]
        lines += _table(["Family", "Missing %", "Block missing percentages (chronological input order)"], [
            [family, values["missing_percentage"], ", ".join(f"{part['start_row']}:{part['end_row_exclusive']} = {part['missing_percentage']:.2f}%" for part in values["time_blocks"])]
            for family, values in audit["family_missingness"].items()]) + [""]
        lines += [f"## {name}: time structure", ""] + _table(["Check", "Observed"], [[key, value] for key, value in audit["time"].items()]) + [""]
        lines += [f"## {name}: observed D-family encodings", "", "These are observed value sets, not inferred economic meanings. A two-state -1/0 field is binary and is not recoded to match a 0/1 assumption.", ""]
        lines += _table(["Column", "Observed values (up to 20)", "Unique nonmissing"], [
            [column, values, audit["column_details"][column]["unique_nonmissing"]] for column, values in audit["dummy_value_sets"].items()]) + [""]
    lines += ["## Train/test relationship", ""] + _table(["Check", "Observed"], [[key, value] for key, value in summary["time_relationship"].items()]) + [""]
    lines += _table(["Schema difference", "Observed columns"], [
        ["Train only", ", ".join(summary["train_only_columns"]) or "none"],
        ["Test only", ", ".join(summary["test_only_columns"]) or "none"]]) + [""]
    lines += ["## Official evaluation/API files", "", "These files are inventoried, not executed. File presence does not establish a scoring formula or label availability; see docs/official_metric_notes.md for the separate source review.", ""]
    lines += [f"- `{path}`" for path in summary.get("evaluation_files", [])] or ["No matching evaluation/API files found in the official manifest."]
    lines += ["", "## Errors", ""] + ([f"- {item}" for item in summary["errors"]] or ["None."])
    lines += ["", "## Warnings and research safeguards", ""] + [f"- {item}" for item in summary["warnings"]]
    lines += ["", "A `date_id` is an ordered identifier; no calendar mapping or daily continuity is inferred. Missing features are not filled during the audit. Outcome fields are never predictors. No split, target-performance analysis or final-holdout evaluation has run.", ""]
    return "\n".join(lines)


def render_dictionary(summary: dict[str, Any]) -> str:
    """Render observed schema with verified definitions, without fitting data."""
    lines = ["# Data dictionary", "", "Source: [official competition data](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data).", "",
             "Individual numbered predictors are anonymous features; broad official family labels do not identify their exact economic meanings.", ""]
    if summary["status"] == "blocked":
        lines += ["Status: **pending official data acquisition**. No actual column list, feature count or dtype has been established.", "",
                  "Expected fields to verify: `date_id`, `forward_returns`, `risk_free_rate`, `market_forward_excess_returns`. These are expectations from the brief, not observations.", ""]
    else:
        lines += _table(["Prefix", "Family taxonomy", "Observed count"], [
            [key, value, len(summary["train"]["feature_groups"][key])]
            for key, value in FAMILY_LABELS.items() if summary["train"]["feature_groups"][key]]) + [""]
        columns = list(dict.fromkeys(summary["train"]["columns"] + summary["test"]["columns"]))
        definitions = {
            "date_id": "Ordered trading-day identifier",
            "forward_returns": "S&P 500 return from buying on the identified day and selling one trading day later",
            "risk_free_rate": "Supplied federal funds rate used directly in cash accounting",
            "market_forward_excess_returns": "Forward return minus rolling five-year mean, then MAD winsorization with criterion 4",
            "is_scored": "Whether the inference row enters competition scoring",
            "lagged_forward_returns": "One-trading-day-lagged forward return; inference-available candidate",
            "lagged_market_forward_excess_returns": "One-trading-day-lagged adjusted target; inference-available candidate",
            "lagged_risk_free_rate": "One-trading-day-lagged cash rate; inference-available candidate",
        }
        lines += [f"Observed shape: train {summary['train']['rows']} × {summary['train']['columns_count']}; test {summary['test']['rows']} × {summary['test']['columns_count']}.", ""]
        lines += _table(["Observed column", "Train dtype", "Test dtype", "Role", "Definition"], [
            [column, summary["train"]["column_details"].get(column, {}).get("dtype", "absent"),
             summary["test"]["column_details"].get(column, {}).get("dtype", "absent"), column_role(column),
             "anonymous feature; " + FAMILY_LABELS[feature_family(column)] if feature_family(column) else definitions.get(column, "Unclassified field")]
            for column in columns]) + [""]
        lines += ["## Observed schema qualifications", "", "Only observed feature families are listed; no missing family is reconstructed.", ""]
        lines += _table(["D field", "Observed training values", "Unique nonmissing"], [
            [column, values, summary["train"]["column_details"][column]["unique_nonmissing"]]
            for column, values in summary["train"]["dummy_value_sets"].items()]) + [""]
    lines += ["Keep the supplied target on its original row; it is not forward_returns minus risk_free_rate. Use supplied cash-rate units unchanged.", "",
              "Current outcomes are not predictors; the three named one-day lagged outcomes are competition-available candidates under [Research Protocol](research_protocol.md).", "",
              "See [Official Metric](official_metric_notes.md) for the verified scoring implementation and penalty rules.", ""]
    return "\n".join(lines)


def save_audit(root: Path, config: dict[str, Any], summary: dict[str, Any]) -> None:
    """Persist report and strict JSON from the same audit result."""
    paths = config["audit"]
    write_json(inside_root(root, paths["summary_path"]), summary)
    for key, content in (("report_path", render_report(summary)), ("dictionary_path", render_dictionary(summary))):
        path = inside_root(root, paths[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def save_blocked_audit(root: Path, config: dict[str, Any], reason: str) -> dict[str, Any]:
    """Record unavailable inputs explicitly, without inventing dataset statistics."""
    summary = {"schema_version": 1, "status": "blocked", "generated_at_utc": utc_timestamp(),
               "competition": config["data"]["competition"], "reason": reason,
               "train": None, "test": None, "feature_groups": None,
               "evaluation_files": None, "final_holdout": "not_created_or_evaluated"}
    save_audit(root, config, summary)
    return summary


def run_audit(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Audit only inputs matching the official-download manifest."""
    if firewall_enabled(root):
        return run_locked_integrity(root, config)
    settings = config["data"]
    raw = inside_root(root, settings["raw_dir"])
    manifest = verify_manifest(raw, settings["manifest_file"], [settings["train_file"], settings["test_file"]])
    train, test = load_official_pair(raw, settings["train_file"], settings["test_file"])
    summary = build_summary(train, test, config)
    summary["provenance"] = {"source": manifest["source"], "downloaded_at_utc": manifest.get("downloaded_at_utc"),
                             "required_file_hashes_verified": True, "all_manifest_file_hashes_verified": True,
                             "official_file_count": len(manifest["files"]), "archive": manifest.get("archive")}
    summary["evaluation_files"] = [item["path"] for item in manifest["files"]
                                   if any(marker in item["path"].lower() for marker in settings["evaluation_markers"])]
    if not summary["evaluation_files"]:
        summary["warnings"].append("No evaluation/API file matched the configured markers in the official archive; do not recreate an unverified competition metric.")
    save_audit(root, config, summary)
    return summary


def firewall_enabled(root: Path) -> bool:
    """Tracked policy keeps protection active even if ignored artifacts vanish."""
    return (root / "configs/split_policy.json").exists() or (root / "artifacts/split_definition.json").exists()


def run_locked_integrity(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Reverify Phase 1 provenance without repeating pre-lock value statistics.

    Machine-readable Phase 1 artifacts remain protected. Narrative reports and
    current dictionaries are editable; the receipt contains no market values.
    """
    from src.validation.time_split import prepare_split

    settings = config["data"]
    raw = inside_root(root, settings["raw_dir"])
    manifest = verify_manifest(raw, settings["manifest_file"], [settings["train_file"], settings["test_file"]])
    policy_path = root / "configs/split_policy.json"
    if not policy_path.is_file():
        raise ValueError("Tracked split policy is required for locked Phase 1 integrity verification.")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    for relative, expected in policy["preserved_phase1_sha256"].items():
        path = inside_root(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Preserved Phase 1 file differs: {relative}")
    if sha256(raw / settings["train_file"]) != policy["source_train_sha256"]:
        raise ValueError("Raw train differs from the fixed split policy.")
    plan = prepare_split(root, raw / settings["train_file"], raw / settings["test_file"])
    summary = {"status": "complete", "mode": "locked_integrity_only",
        "generated_at_utc": utc_timestamp(), "preserved_phase1_machine_artifacts_unchanged": True,
        "simulation_values_parsed": False, "official_test_values_parsed": False,
        "train": {"rows": plan["full_train_rows"], "columns_count": len(plan["train_columns"]),
                  "feature_groups": group_features(plan["feature_columns"])},
        "test": {"rows": plan["official_test"]["rows"], "columns_count": len(plan["official_test"]["columns"]),
                 "feature_groups": group_features(plan["feature_columns"])},
        "provenance": {"source": manifest["source"], "downloaded_at_utc": manifest.get("downloaded_at_utc"),
            "all_manifest_file_hashes_verified": True, "official_file_count": len(manifest["files"]),
            "archive": manifest.get("archive")},
        "evaluation_files": [item["path"] for item in manifest["files"]
            if any(marker in item["path"].lower() for marker in settings["evaluation_markers"])],
        "errors": [], "warnings": ["Historical Phase 1 value audit preserved; no locked-period value statistics recomputed."]}
    write_json(root / "artifacts/phase1_integrity_check.json", summary)
    return summary
