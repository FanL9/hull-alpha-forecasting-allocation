"""Official acquisition, protected Phase 1 integrity and development-only EDA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from src.data.download import AcquisitionError, credential_status, download_official, verify_manifest
from src.data.load_data import DataUnavailableError
from src.data.validate_data import firewall_enabled, run_audit, save_blocked_audit
from src.utils.paths import inside_root, load_config, project_root, utc_timestamp, write_json
from src.utils.seed import set_seed


def environment_status() -> dict:
    """Report dependency versions and credential presence, never their values."""
    packages = {}
    for name in ("kaggle", "pandas", "numpy", "PyYAML", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages, "authentication": credential_status()}


def main(argv: list[str] | None = None) -> int:
    """Return 0 on success, 1 on schema failure, 2 on unavailable official data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-env", "download", "audit", "phase1", "phase2"])
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--eda-config", "--research-config", dest="eda_config", default="configs/research.yaml")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = load_config(inside_root(root, args.config))
    set_seed(config["project"]["seed"])
    if args.command == "phase2":
        from src.research.pipeline import run_phase2
        result = run_phase2(root, config, inside_root(root, args.eda_config))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    environment = environment_status()
    if args.command == "check-env":
        print(json.dumps(environment, indent=2, ensure_ascii=False))
        return 0
    settings = config["data"]
    raw = inside_root(root, settings["raw_dir"])
    status = {"generated_at_utc": utc_timestamp(), "command": args.command, "environment": environment}
    protected = firewall_enabled(root)
    try:
        if args.command == "download":
            manifest = download_official(raw, timeout=settings["timeout_seconds"])
            status.update({"status": "downloaded", "file_count": len(manifest["files"]), "audit": "not_run"})
            status["acquisition"] = {"source": manifest["source"], "downloaded_at_utc": manifest["downloaded_at_utc"],
                                     "server_access_evidence": "successful official CLI download at the recorded acquisition time"}
        else:
            if args.command == "phase1" and not protected:
                try:
                    verify_manifest(raw, settings["manifest_file"], [settings["train_file"], settings["test_file"]])
                except AcquisitionError:
                    download_official(raw, timeout=settings["timeout_seconds"])
            summary = run_audit(root, config)
            status.update({"status": summary["status"], "train_shape": [summary["train"]["rows"], summary["train"]["columns_count"]],
                           "test_shape": [summary["test"]["rows"], summary["test"]["columns_count"]],
                           "feature_group_counts": {name: len(columns) for name, columns in summary["train"]["feature_groups"].items()},
                           "errors": summary["errors"], "warnings": summary["warnings"]})
            status["acquisition"] = {"source": summary["provenance"]["source"],
                                     "downloaded_at_utc": summary["provenance"]["downloaded_at_utc"],
                                     "server_access_evidence": "successful official CLI download at the recorded acquisition time; subsequent audits verify local hashes",
                                     "all_manifest_file_hashes_verified": True,
                                     "official_file_count": summary["provenance"]["official_file_count"],
                                     "evaluation_file_count": len(summary["evaluation_files"])}
    except (AcquisitionError, DataUnavailableError, OSError, ValueError) as exc:
        # Exceptions raised by our acquisition layer are sanitized; no raw API logs.
        reason = str(exc) if isinstance(exc, (AcquisitionError, DataUnavailableError, ValueError)) else f"Local file I/O failed: {type(exc).__name__}"
        if not protected:
            save_blocked_audit(root, config, reason)
        status.update({"status": "blocked", "reason": reason, "train_shape": None,
                       "test_shape": None, "feature_group_counts": None})
    destination = "artifacts/phase1_integrity_status.json" if protected else config["audit"]["status_path"]
    write_json(inside_root(root, destination), status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 2 if status["status"] == "blocked" else 1 if status["status"] == "validation_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
