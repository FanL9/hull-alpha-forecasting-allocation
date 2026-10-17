"""Acquire only official competition files and retain a hash manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from src.utils.paths import utc_timestamp, write_json

COMPETITION = "hull-tactical-market-prediction"


class AcquisitionError(RuntimeError):
    """An actionable official-data acquisition failure, containing no secrets."""


def credential_status() -> dict[str, Any]:
    """Inspect standard credential locations without exposing credential values."""
    default = Path.home() / ".kaggle"
    configured_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    config_dir = Path(configured_dir) if configured_dir else default
    if not configured_dir and sys.platform.startswith("linux") and not default.exists():
        config_dir = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "kaggle"
    sources: list[str] = []
    invalid: list[str] = []
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        sources.append("KAGGLE_API_TOKEN")
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        sources.append("KAGGLE_USERNAME + KAGGLE_KEY")
    for directory in [config_dir]:
        legacy = directory / "kaggle.json"
        if legacy.is_file():
            try:
                credentials = json.loads(legacy.read_text(encoding="utf-8-sig"))
                if isinstance(credentials, dict) and credentials.get("username") and credentials.get("key"):
                    sources.append("configured legacy credential file")
                else:
                    invalid.append("legacy credential file lacks username/key")
            except (OSError, ValueError):
                invalid.append("legacy credential file is unreadable or invalid JSON")
    # The current SDK looks for token files under ~/.kaggle independently of
    # KAGGLE_CONFIG_DIR. OAuth cache presence is not proof of server validity.
    for token in [default / "access_token", default / "access_token.txt"]:
        if token.is_file():
            try:
                if token.read_text(encoding="utf-8").strip():
                    sources.append("configured API token file")
            except OSError:
                invalid.append("API token file is unreadable")
    oauth = default / "credentials.json"
    if oauth.is_file():
        try:
            credentials = json.loads(oauth.read_text(encoding="utf-8-sig"))
            if isinstance(credentials, dict) and credentials.get("refresh_token"):
                sources.append("configured OAuth credential cache")
            else:
                invalid.append("OAuth cache lacks refresh token")
        except (OSError, ValueError):
            invalid.append("OAuth cache is unreadable or invalid JSON")
    return {"configured": bool(sources), "sources": sources, "invalid": invalid,
            "rules_acceptance": "not_verified", "server_authentication": "not_verified"}


def sha256(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> list[Path]:
    """Validate the full ZIP before extracting; reject traversal and symlinks."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        targets: list[Path] = []
        for member in members:
            normalized = member.filename.replace("\\", "/")
            parts = PurePosixPath(normalized).parts
            target = (destination / normalized).resolve()
            if (not target.is_relative_to(destination) or normalized.startswith("/")
                    or ".." in parts or any(":" in part for part in parts)
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise AcquisitionError("Unsafe member in official ZIP; extraction refused.")
            targets.append(target)
        if len(targets) != len(set(targets)):
            raise AcquisitionError("Duplicate paths in ZIP; extraction refused.")
        bad = bundle.testzip()
        if bad is not None:
            raise AcquisitionError("ZIP integrity check failed; download again.")
        for member, target in zip(members, targets, strict=True):
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                with bundle.open(member) as handle:
                    archived_hash = hashlib.file_digest(handle, "sha256").hexdigest()
                if sha256(target) != archived_hash:
                    raise AcquisitionError(f"Existing file differs from archive: {target.name}; move it before retrying.")
        bundle.extractall(destination)
    return [target for member, target in zip(members, targets, strict=True) if not member.is_dir()]


def kaggle_executable() -> Path:
    """Use the CLI installed in the current Python environment."""
    script_dir = Path(sys.executable).parent
    executable = script_dir / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if not executable.is_file():
        raise AcquisitionError("Kaggle CLI missing in this Python environment. Install requirements.txt.")
    return executable


def classify_cli_error(output: str) -> str:
    """Translate failure types without persisting potentially secret output."""
    lowered = output.lower()
    if "403" in lowered or "forbidden" in lowered:
        return "Kaggle denied access (403). Check competition rules acceptance and API permissions."
    if any(word in lowered for word in ("401", "unauthorized", "authenticate", "authentication required", "credentials")):
        return "Kaggle authentication failed. Configure a valid Kaggle credential."
    if any(word in lowered for word in ("getaddrinfo", "connection", "timed out", "name resolution")):
        return "Kaggle network request failed. Check DNS/network access; no alternate data source was used."
    return "Official Kaggle CLI download failed. Check CLI diagnostics locally and Kaggle access permissions."


def download_official(raw_dir: Path, *, timeout: int = 300) -> dict[str, Any]:
    """Download the complete official ZIP, verify it, and inventory its files.

    No third-party fallback, browser login, rules acceptance or submission occurs.
    Presence of a local credential does not establish server-side validity.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    executable = kaggle_executable()
    auth = credential_status()
    if not auth["configured"]:
        raise AcquisitionError("Kaggle credentials missing: no valid kaggle.json, access_token, OAuth cache, KAGGLE_API_TOKEN, or username/key environment pair.")
    command = [str(executable), "competitions", "download", "-c", COMPETITION, "-p", str(raw_dir)]
    cli_environment = os.environ.copy()
    if not cli_environment.get("KAGGLE_CONFIG_DIR") and not (Path.home() / ".kaggle").exists():
        # The CLI creates its configuration directory even for env-var tokens.
        # Keep this incidental write in the project rather than in the home dir.
        local_config = raw_dir.resolve().parents[1] / "work" / "kaggle-config"
        local_config.mkdir(parents=True, exist_ok=True)
        cli_environment["KAGGLE_CONFIG_DIR"] = str(local_config)
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout, check=False, env=cli_environment)
    except subprocess.TimeoutExpired as exc:
        raise AcquisitionError("Official Kaggle download timed out; no fallback data was used.") from exc
    if result.returncode:
        raise AcquisitionError(classify_cli_error(result.stdout + result.stderr))
    archive = raw_dir / f"{COMPETITION}.zip"
    if not archive.is_file():
        raise AcquisitionError("CLI returned success but the expected competition ZIP is absent.")
    try:
        files = safe_extract(archive, raw_dir)
    except AcquisitionError:
        raise
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        raise AcquisitionError(f"Official ZIP could not be extracted: {type(exc).__name__}.") from exc
    required = {"train.csv", "test.csv"}
    if not all((raw_dir / filename).is_file() for filename in required):
        raise AcquisitionError("The official archive lacks a root-level train.csv or test.csv; inspect its structure without inventing substitutes.")
    manifest = {
        "status": "downloaded", "competition": COMPETITION, "source": "official_kaggle_competition_cli",
        "source_url": f"https://www.kaggle.com/competitions/{COMPETITION}/data",
        "downloaded_at_utc": utc_timestamp(), "kaggle_version": importlib.metadata.version("kaggle"),
        "archive": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive)},
        "files": [{"path": path.relative_to(raw_dir).as_posix(), "bytes": path.stat().st_size,
                   "sha256": sha256(path)} for path in sorted(files)],
    }
    write_json(raw_dir / "download_manifest.json", manifest)
    return manifest


def verify_manifest(raw_dir: Path, manifest_name: str, required_files: list[str]) -> dict[str, Any]:
    """Require matching official acquisition provenance before running the audit."""
    manifest_path = raw_dir / manifest_name
    if not manifest_path.is_file():
        raise AcquisitionError("Official download manifest missing. Run the official download command first.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("competition") != COMPETITION or manifest.get("source") != "official_kaggle_competition_cli":
            raise AcquisitionError("Data provenance does not identify the official Hull competition CLI.")
        entries = {entry["path"]: entry for entry in manifest["files"]}
        for filename in required_files:
            if filename not in entries or not (raw_dir / filename).is_file():
                raise AcquisitionError(f"Official input absent from manifest or disk: {filename}")
        for filename, entry in entries.items():
            path = (raw_dir / filename).resolve()
            if not path.is_relative_to(raw_dir.resolve()):
                raise AcquisitionError("Manifest path escapes raw data directory.")
            if not path.is_file():
                raise AcquisitionError(f"Official manifest file is missing on disk: {filename}")
            if sha256(path) != entry["sha256"]:
                raise AcquisitionError(f"Input hash differs from official download manifest: {filename}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise AcquisitionError("Download manifest is unreadable or invalid.") from exc
    return manifest
