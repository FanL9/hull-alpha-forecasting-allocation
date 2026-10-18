"""Phase 3 execution freeze and canonical, fail-closed real-run reservation.

This is an execution policy, independent of the Phase 2.5 statistical lock.
Preflight reads Git and execution metadata only, never market observations.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess

from src.utils.paths import inside_root, utc_timestamp, write_json

PIPELINE_TAG = "phase3-pipeline-v1"
REAL_OUTPUT = "artifacts/phase3"
REAL_MANIFEST = f"{REAL_OUTPUT}/phase3_run_manifest.json"
REAL_STATE = "artifacts/phase3_execution_state.json"
PIPELINE_FILES = (
    "src/cli.py", "src/research/alpha_pipeline.py", "src/research/alpha_results.py",
    "src/research/phase3.py", "src/research/synthetic_alpha.py",
    "src/research/phase3_execution.py", "src/research/alpha_calibration.py",
)


def _git(root: Path, *args: str) -> bytes:
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if process.returncode:
        raise RuntimeError(f"Git identity query failed: {' '.join(args[:2])}")
    return process.stdout


def git_identity(root: Path) -> dict:
    """Unknown Git state stays unknown rather than being treated as clean."""
    try:
        commit = _git(root, "rev-parse", "HEAD").decode().strip()
        dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=all").strip())
    except (OSError, RuntimeError):
        return {"git_commit": None, "git_tag": None, "git_tag_commit": None,
                "git_tag_annotated": False, "git_worktree_dirty": None}
    try:
        tag_commit = _git(root, "rev-parse", f"refs/tags/{PIPELINE_TAG}^{{commit}}").decode().strip()
        annotated = _git(root, "cat-file", "-t", f"refs/tags/{PIPELINE_TAG}").strip() == b"tag"
    except (OSError, RuntimeError):
        tag_commit, annotated = None, False
    return {"git_commit": commit, "git_tag": PIPELINE_TAG if tag_commit == commit else None,
            "git_tag_commit": tag_commit, "git_tag_annotated": annotated, "git_worktree_dirty": dirty}


def pipeline_code_identity() -> dict:
    """Hash executing source bytes, including calibration and execution guards."""
    source_root = Path(__file__).resolve().parents[2]
    hashes = {name: hashlib.sha256((source_root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for name in PIPELINE_FILES}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"implementation_sha256": hashes, "phase3_pipeline_code_hash": digest}


def validate_frozen_checkout(root: Path) -> dict:
    """Require a clean checkout at the annotated freeze tag with matching code."""
    identity = git_identity(root)
    if identity["git_worktree_dirty"] is not False:
        raise PermissionError("Real Phase 3 requires a clean worktree; Git state is dirty or unavailable.")
    if identity["git_tag"] != PIPELINE_TAG or not identity["git_tag_annotated"]:
        raise PermissionError(f"Real Phase 3 requires HEAD at annotated tag {PIPELINE_TAG}.")
    code = pipeline_code_identity()
    for name, digest in code["implementation_sha256"].items():
        committed = _git(root, "show", f"HEAD:{name}").replace(b"\r\n", b"\n")
        if hashlib.sha256(committed).hexdigest() != digest:
            raise PermissionError("Executing Phase 3 implementation differs from the frozen checkout.")
    return {**identity, **code}


def assert_real_run_available(root: Path) -> None:
    """Canonical evidence blocks reruns regardless of a supplied output path."""
    manifest = inside_root(root, REAL_MANIFEST)
    if manifest.exists():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            real = (saved.get("run_type") == "real" or saved.get("run_mode") == "real"
                    or saved.get("real_alpha_research_executed") is True)
        except (OSError, ValueError, AttributeError) as exc:
            raise PermissionError("Canonical real-run manifest is unreadable; refusing real execution.") from exc
        if real:
            raise PermissionError("Real Phase 3 has already been executed. Refusing to rerun.")
        # Even a non-real manifest occupies the canonical output destination.
        raise PermissionError("Canonical Phase 3 manifest already exists; refusing to overwrite execution evidence.")
    if inside_root(root, REAL_STATE).exists():
        raise PermissionError("Real Phase 3 is already reserved, completed or interrupted. Refusing to rerun.")


def real_execution_preflight(root: Path, output_dir: str = REAL_OUTPUT) -> dict:
    assert_real_run_available(root)
    if inside_root(root, output_dir) != inside_root(root, REAL_OUTPUT):
        raise PermissionError("Real Phase 3 must use the canonical artifacts/phase3/ directory.")
    return validate_frozen_checkout(root)


@contextmanager
def reserve_real_run(root: Path, identity: dict):
    """Atomic single-run claim before prefix parsing; interrupted runs stay spent.

    The reservation is outside the output directory so the completed artifact
    writer still requires an empty destination. There is no force-rerun switch.
    """
    assert_real_run_available(root)
    path = inside_root(root, REAL_STATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"run_type": "real", "status": "started", "started_at_utc": utc_timestamp(),
             "git_commit": identity["git_commit"], "git_tag": identity["git_tag"],
             "phase3_pipeline_code_hash": identity["phase3_pipeline_code_hash"]}
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise PermissionError("Real Phase 3 has already been reserved. Refusing concurrent execution.") from exc
    try:
        yield
    except BaseException:
        write_json(path, {**state, "status": "interrupted_or_failed", "finished_at_utc": utc_timestamp()})
        raise
    else:
        write_json(path, {**state, "status": "completed", "finished_at_utc": utc_timestamp()})
