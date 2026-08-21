from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eval_factory.agent_system.supervisor import (
    AgentLeaseV2,
    AgentWorkerRef,
)
from eval_factory.agent_system.workspace import (
    AgentWorkspaceError,
    AgentWorkspaceManager,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentWorkspaceStateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64
NOW = datetime(2026, 8, 6, tzinfo=UTC)


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="agent-workspace-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="agent-workspace-v1",
                sha256=HASH,
            ),
        ),
    )


def _lease(
    *,
    task: str = "task-a",
    attempt: int = 1,
    fence: int = 1,
) -> AgentLeaseV2:
    return AgentLeaseV2(
        agent_task_ref=_ref("agent-task", task),
        agent_definition_ref=_ref("agent-definition"),
        lease_id=f"agent-lease://{task}/{attempt}/{fence}",
        worker=AgentWorkerRef(worker_id="worker://workspace"),
        attempt=attempt,
        fencing_token=fence,
        acquired_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )


def test_workspace_namespace_is_isolated_and_replays_after_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-workspaces"
    manager = AgentWorkspaceManager(root)
    first = manager.open(
        run_ref=_ref("factory-run"),
        lease=_lease(task="task-a"),
        audit=_audit(),
    )
    second = manager.open(
        run_ref=_ref("factory-run"),
        lease=_lease(task="task-b"),
        audit=_audit(),
    )

    assert first.path != second.path
    assert first.path.is_dir() and second.path.is_dir()
    assert first.path.is_relative_to(root)
    assert "factory-run://example" not in str(first.path)
    replay = AgentWorkspaceManager(root).open(
        run_ref=_ref("factory-run"),
        lease=_lease(task="task-a"),
        audit=_audit(),
    )
    assert replay.receipt == first.receipt
    assert replay.path == first.path


def test_workspace_commit_is_content_bound_and_path_free(
    tmp_path: Path,
) -> None:
    manager = AgentWorkspaceManager(tmp_path / "private-workspaces")
    handle = manager.open(
        run_ref=_ref("factory-run"),
        lease=_lease(),
        audit=_audit(),
    )
    (handle.path / "input.txt").write_text(
        "safe input state",
        encoding="utf-8",
    )
    committed = manager.commit(handle, audit=_audit())
    replay = manager.commit(handle, audit=_audit())

    assert committed == replay
    assert committed.state is AgentWorkspaceStateV2.COMMITTED
    assert committed.inventory_sha256 is not None
    assert committed.predecessor_workspace_receipt_ref == (handle.receipt.to_ref())
    assert manager.resolve(committed.to_ref()) == committed
    assert "path" not in committed.model_dump(mode="json")


def test_workspace_symlink_and_quarantine_fail_closed(
    tmp_path: Path,
) -> None:
    manager = AgentWorkspaceManager(tmp_path / "private-workspaces")
    handle = manager.open(
        run_ref=_ref("factory-run"),
        lease=_lease(),
        audit=_audit(),
    )
    (handle.path / "outside-link").symlink_to(tmp_path)

    with pytest.raises(AgentWorkspaceError, match="symlink"):
        manager.commit(handle, audit=_audit())
    quarantined = manager.quarantine(
        handle,
        reason_code="WORKSPACE_UNSAFE",
        audit=_audit(),
    )
    assert quarantined.state is AgentWorkspaceStateV2.QUARANTINED
    assert not handle.path.exists()

    unsafe_root = tmp_path / "unsafe-root"
    unsafe_root.symlink_to(tmp_path / "private-workspaces")
    with pytest.raises(AgentWorkspaceError, match="symlink"):
        AgentWorkspaceManager(unsafe_root)


def test_workspace_terminal_states_are_replayable_and_exclusive(
    tmp_path: Path,
) -> None:
    manager = AgentWorkspaceManager(tmp_path / "private-workspaces")
    committed_handle = manager.open(
        run_ref=_ref("factory-run"),
        lease=_lease(task="committed-task"),
        audit=_audit(),
    )
    nested = committed_handle.path / "nested"
    nested.mkdir()
    (nested / "input.txt").write_text(
        "nested input",
        encoding="utf-8",
    )
    manager.commit(committed_handle, audit=_audit())
    with pytest.raises(AgentWorkspaceError, match="terminal"):
        manager.open(
            run_ref=_ref("factory-run"),
            lease=_lease(task="committed-task"),
            audit=_audit(),
        )
    with pytest.raises(AgentWorkspaceError, match="committed"):
        manager.quarantine(
            committed_handle,
            reason_code="TOO_LATE",
            audit=_audit(),
        )

    quarantined_handle = manager.open(
        run_ref=_ref("factory-run"),
        lease=_lease(task="quarantined-task"),
        audit=_audit(),
    )
    quarantined = manager.quarantine(
        quarantined_handle,
        reason_code="WORKER_CANCELLED",
        audit=_audit(),
    )
    assert (
        manager.quarantine(
            quarantined_handle,
            reason_code="WORKER_CANCELLED",
            audit=_audit(),
        )
        == quarantined
    )
    with pytest.raises(AgentWorkspaceError, match="quarantined"):
        manager.commit(quarantined_handle, audit=_audit())


def test_workspace_invalid_root_and_receipt_ref_are_rejected(
    tmp_path: Path,
) -> None:
    invalid_root = tmp_path / "not-a-directory"
    invalid_root.write_text("file", encoding="utf-8")
    with pytest.raises(AgentWorkspaceError, match="directory"):
        AgentWorkspaceManager(invalid_root)

    manager = AgentWorkspaceManager(tmp_path / "private-workspaces")
    with pytest.raises(AgentWorkspaceError, match="ref type"):
        manager.resolve(_ref("not-a-workspace-receipt"))


def test_workspace_recovers_exact_committed_namespace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-workspaces"
    manager = AgentWorkspaceManager(root)
    lease = _lease(task="recover-committed")
    handle = manager.open(
        run_ref=_ref("factory-run"),
        lease=lease,
        audit=_audit(),
    )
    (handle.path / "result.json").write_text(
        "{}",
        encoding="utf-8",
    )
    committed = manager.commit(handle, audit=_audit())

    recovered = AgentWorkspaceManager(root).recover(
        run_ref=_ref("factory-run"),
        lease=lease,
        audit=_audit(),
    )

    assert recovered.path == handle.path
    assert recovered.receipt == committed


def test_workspace_recovery_rejects_terminal_and_inventory_drift(
    tmp_path: Path,
) -> None:
    quarantined_manager = AgentWorkspaceManager(tmp_path / "quarantined")
    quarantined_lease = _lease(task="recover-quarantined")
    quarantined = quarantined_manager.open(
        run_ref=_ref("factory-run"),
        lease=quarantined_lease,
        audit=_audit(),
    )
    quarantined_manager.quarantine(
        quarantined,
        reason_code="CANCELLED",
        audit=_audit(),
    )
    with pytest.raises(
        AgentWorkspaceError,
        match="terminal authority",
    ):
        quarantined_manager.recover(
            run_ref=_ref("factory-run"),
            lease=quarantined_lease,
            audit=_audit(),
        )
    with pytest.raises(
        AgentWorkspaceError,
        match="terminal authority",
    ):
        quarantined_manager.quarantine_lease(
            run_ref=_ref("factory-run"),
            lease=quarantined_lease,
            reason_code="DIFFERENT_REASON",
            audit=_audit(),
        )

    committed_manager = AgentWorkspaceManager(tmp_path / "committed")
    committed_lease = _lease(task="recover-drift")
    committed_handle = committed_manager.open(
        run_ref=_ref("factory-run"),
        lease=committed_lease,
        audit=_audit(),
    )
    result_path = committed_handle.path / "result.json"
    result_path.write_text("first", encoding="utf-8")
    committed_manager.commit(
        committed_handle,
        audit=_audit(),
    )
    result_path.write_text("changed", encoding="utf-8")

    with pytest.raises(
        AgentWorkspaceError,
        match="inventory drifted",
    ):
        committed_manager.recover(
            run_ref=_ref("factory-run"),
            lease=committed_lease,
            audit=_audit(),
        )


def test_workspace_recovery_rejects_marker_and_path_drift(
    tmp_path: Path,
) -> None:
    marker_manager = AgentWorkspaceManager(tmp_path / "marker")
    lease = _lease(task="recover-marker")
    handle = marker_manager.open(
        run_ref=_ref("factory-run"),
        lease=lease,
        audit=_audit(),
    )
    (handle.path / "result.json").write_text(
        "{}",
        encoding="utf-8",
    )
    marker_manager.commit(handle, audit=_audit())
    other = marker_manager.open(
        run_ref=_ref("factory-run"),
        lease=_lease(task="other-marker"),
        audit=_audit(),
    )
    handle.path.joinpath(".agent-workspace-authority.json").write_bytes(other.receipt.canonical_json())

    with pytest.raises(
        AgentWorkspaceError,
        match="marker drifted",
    ):
        marker_manager.recover(
            run_ref=_ref("factory-run"),
            lease=lease,
            audit=_audit(),
        )

    missing_manager = AgentWorkspaceManager(tmp_path / "missing")
    missing_lease = _lease(task="recover-missing")
    missing = missing_manager.open(
        run_ref=_ref("factory-run"),
        lease=missing_lease,
        audit=_audit(),
    )
    missing_manager.commit(missing, audit=_audit())
    missing.path.rename(missing.path.with_name("removed-workspace"))
    with pytest.raises(
        AgentWorkspaceError,
        match="namespace is missing",
    ):
        missing_manager.recover(
            run_ref=_ref("factory-run"),
            lease=missing_lease,
            audit=_audit(),
        )
