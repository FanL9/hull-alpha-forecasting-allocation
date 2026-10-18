"""Execution policy tests use metadata doubles and temporary artificial state."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.cli import main
from src.research import alpha_results, phase3, phase3_execution as execution
from src.research.alpha_pipeline import load_alpha_design
from src.research.synthetic_alpha import make_synthetic_history
from src.utils.paths import project_root, write_json
from src.validation.protocol_freeze import verify_protocol_lock

ROOT = project_root()


def identity():
    return {'git_commit': 'a' * 40, 'git_tag': execution.PIPELINE_TAG, 'git_tag_commit': 'a' * 40,
            'git_tag_annotated': True, 'git_worktree_dirty': False, **execution.pipeline_code_identity()}


@pytest.mark.parametrize('dirty', [True, None])
def test_dirty_or_unknown_git_refused(dirty, monkeypatch):
    state = identity()
    state['git_worktree_dirty'] = dirty
    monkeypatch.setattr(execution, 'git_identity', lambda root: state)
    with pytest.raises(PermissionError, match='clean worktree'):
        execution.validate_frozen_checkout(ROOT)


@pytest.mark.parametrize('tag,annotated', [(None, False), (None, True), (execution.PIPELINE_TAG, False)])
def test_missing_wrong_commit_or_lightweight_tag_refused(tag, annotated, monkeypatch):
    state = identity()
    state.update(git_tag=tag, git_tag_annotated=annotated)
    monkeypatch.setattr(execution, 'git_identity', lambda root: state)
    with pytest.raises(PermissionError, match='annotated tag'):
        execution.validate_frozen_checkout(ROOT)


def test_matching_clean_annotated_commit_and_executing_code_accepted(monkeypatch):
    state = identity()
    monkeypatch.setattr(execution, 'git_identity', lambda root: state)
    def git(root, *args):
        assert args[0] == 'show' and args[1].startswith('HEAD:')
        return (ROOT / args[1][5:]).read_bytes()
    monkeypatch.setattr(execution, '_git', git)
    accepted = execution.validate_frozen_checkout(ROOT)
    assert accepted == state


def test_checkout_different_from_executing_code_refused(monkeypatch):
    monkeypatch.setattr(execution, 'git_identity', lambda root: identity())
    monkeypatch.setattr(execution, '_git', lambda *a: b'other implementation')
    with pytest.raises(PermissionError, match='differs'):
        execution.validate_frozen_checkout(ROOT)


@pytest.mark.parametrize('tag_commit,kind', [('a' * 40, b'tag\n'), ('b' * 40, b'tag\n'), ('a' * 40, b'commit\n')])
def test_git_snapshot_resolves_tag_commit_and_annotation(tag_commit, kind, monkeypatch):
    def git(root, *args):
        if args == ('rev-parse', 'HEAD'):
            return ('a' * 40).encode()
        if args[0] == 'status':
            return b''
        if args[0] == 'rev-parse':
            assert args[1] == 'refs/tags/phase3-pipeline-v1^{commit}'
            return tag_commit.encode()
        if args[0] == 'cat-file':
            return kind
        pytest.fail(str(args))
    monkeypatch.setattr(execution, '_git', git)
    state = execution.git_identity(ROOT)
    assert state['git_tag'] == (execution.PIPELINE_TAG if tag_commit == 'a' * 40 else None)
    assert state['git_tag_annotated'] == (kind == b'tag\n')


def test_git_query_failure_is_not_misreported_as_clean(monkeypatch):
    def fail(*a):
        raise RuntimeError('artificial git failure')
    monkeypatch.setattr(execution, '_git', fail)
    assert execution.git_identity(ROOT)['git_worktree_dirty'] is None


@pytest.mark.parametrize('fields', [{'run_type': 'real'}, {'run_mode': 'real'}, {'real_alpha_research_executed': True}])
def test_canonical_real_manifest_blocks_alternate_output(tmp_path, monkeypatch, fields):
    write_json(tmp_path / execution.REAL_MANIFEST, fields)
    monkeypatch.setattr(execution, 'validate_frozen_checkout', lambda *a: pytest.fail('Canonical evidence must be checked first'))
    with pytest.raises(PermissionError, match='already been executed'):
        execution.real_execution_preflight(tmp_path, 'artifacts/another_output')


@pytest.mark.parametrize('content', ['{not json', '[]', '{"run_type":"synthetic"}'])
def test_canonical_manifest_never_silently_overwritten(tmp_path, content):
    path = tmp_path / execution.REAL_MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(content)
    with pytest.raises(PermissionError):
        execution.assert_real_run_available(tmp_path)


def test_no_manifest_still_cannot_redirect_real_run(tmp_path):
    with pytest.raises(PermissionError, match='canonical'):
        execution.real_execution_preflight(tmp_path, 'artifacts/another_output')


@pytest.mark.parametrize('failed', [False, True])
def test_atomic_claim_blocks_concurrent_and_later_runs_even_after_failure(tmp_path, failed):
    state = identity()
    def attempt():
        with execution.reserve_real_run(tmp_path, state):
            started = json.loads((tmp_path / execution.REAL_STATE).read_text())
            assert started['status'] == 'started'
            with pytest.raises(PermissionError):
                with execution.reserve_real_run(tmp_path, state):
                    pytest.fail('Concurrent real execution entered')
            if failed:
                raise RuntimeError('artificial execution interruption')
    if failed:
        with pytest.raises(RuntimeError):
            attempt()
    else:
        attempt()
    saved = json.loads((tmp_path / execution.REAL_STATE).read_text())
    assert saved['status'] == ('interrupted_or_failed' if failed else 'completed')
    with pytest.raises(PermissionError, match='reserved'):
        execution.assert_real_run_available(tmp_path)


def test_rejected_execution_never_reaches_data_verification_or_loader(monkeypatch):
    def reject(*a):
        raise PermissionError('artificial preflight refusal')
    monkeypatch.setattr(phase3, 'real_execution_preflight', reject)
    monkeypatch.setattr(phase3, 'verify_protocol_lock', lambda *a, **k: pytest.fail('Refused run touched data verification'))
    monkeypatch.setattr(phase3, 'load_research_prefix', lambda *a, **k: pytest.fail('Refused run touched research data'))
    with pytest.raises(PermissionError, match='preflight refusal'):
        phase3.run_real_phase3(ROOT, run_real_phase3=True)


def test_manifest_synthetic_identity_and_real_clean_tag_requirement(monkeypatch):
    history = make_synthetic_history(ROOT)
    design = load_alpha_design(ROOT)
    lock = verify_protocol_lock(ROOT)
    synthetic = alpha_results.build_manifest(ROOT, history, design, lock, synthetic=True)
    assert synthetic['run_type'] == 'synthetic'
    assert len(synthetic['phase3_pipeline_code_hash']) == 64 and 'git_tag' in synthetic
    state = identity()
    monkeypatch.setattr(alpha_results, 'validate_frozen_checkout', lambda root: deepcopy(state))
    real = alpha_results.build_manifest(ROOT, history, design, lock, synthetic=False, execution_identity=state)
    assert real['run_type'] == 'real' and real['git_worktree_dirty'] is False
    assert real['git_tag'] == execution.PIPELINE_TAG and real['git_commit'] == 'a' * 40
    changed = {**state, 'git_commit': 'b' * 40}
    with pytest.raises(PermissionError, match='identity changed'):
        alpha_results.build_manifest(ROOT, history, design, lock, synthetic=False, execution_identity=changed)


def test_aggregate_code_hash_binds_all_pipeline_file_hashes():
    state = execution.pipeline_code_identity()
    expected = hashlib.sha256(json.dumps(state['implementation_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert state['phase3_pipeline_code_hash'] == expected
    assert {'src/research/phase3_execution.py', 'src/research/alpha_pipeline.py'} <= set(state['implementation_sha256'])


def test_metadata_only_preflight_cli_never_verifies_raw_data(monkeypatch, capsys):
    import src.validation.protocol_freeze as freeze
    def verify(root, *, verify_data):
        assert verify_data is False
        return {}
    monkeypatch.setattr(freeze, 'verify_protocol_lock', verify)
    monkeypatch.setattr(execution, 'real_execution_preflight', lambda root: identity())
    assert main(['phase3-check']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['preflight'] == 'passed' and output['market_data_read'] is False
