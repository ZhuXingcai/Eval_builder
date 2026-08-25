from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from env_mock_agent.facade import (
    FacadeObjectRef,
    RuntimeEventKindV2,
    RuntimeFailureCodeV2,
    UnifiedRuntimeEventV2,
    UnifiedRuntimeUsageV2,
)
from env_mock_agent.facade.telemetry_v2 import ReportedCostV2
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness import (
    HarnessSessionConflictError,
    HarnessSessionIntegrityError,
    HarnessSessionStore,
    SessionRuntimePayloadV1,
)

NOW = datetime(2026, 8, 23, tzinfo=UTC)
HASH = "a" * 64


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v1",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="runtime-event-store-test",
        governing_versions=(
            VersionBinding(
                component="stage5-runtime-events",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _runtime_event(
    sequence: int,
    kind: RuntimeEventKindV2,
    *,
    run_id: str = "run-shell-001",
    work_id: str = "work-shell-001",
    source_label: str | None = None,
) -> UnifiedRuntimeEventV2:
    digest = hashlib.sha256(
        (source_label or f"source-{sequence}").encode(),
    ).hexdigest()
    return UnifiedRuntimeEventV2.create(
        run_id=run_id,
        work_id=work_id,
        runtime_id="pi_rpc",
        runtime_version="0.80.10",
        sequence=sequence,
        kind=kind,
        occurred_at=NOW,
        source_refs=(
            FacadeObjectRef(
                object_type="runtime-event-content",
                object_id=f"runtime-event-content://sha256/{digest}",
                object_version="private-v1",
                object_sha256=digest,
            ),
        ),
        usage=(
            UnifiedRuntimeUsageV2(
                model_requests=1,
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                tool_calls=0,
                turns=1,
                duration_ms=20,
                reported_cost=ReportedCostV2.unavailable(),
            )
            if kind is RuntimeEventKindV2.USAGE_REPORTED
            else None
        ),
        failure_code=(
            RuntimeFailureCodeV2.PROVIDER_FAILED if kind is RuntimeEventKindV2.RUN_FAILED else None
        ),
    )


def _created(
    store: HarnessSessionStore,
    *,
    session_id: str = "session-runtime-events",
):
    return store.create_session(
        session_id=session_id,
        incarnation_id=f"{session_id}-incarnation",
        composition_ref=_ref("harness-composition"),
        created_by="session-user",
        idempotency_key=f"create-{session_id}",
        audit=_audit(),
    )


def test_runtime_event_append_replay_and_reopen_are_exact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "harness.sqlite3"
    store = HarnessSessionStore(path, clock=lambda: NOW)
    created = _created(store)
    runtime_event = _runtime_event(1, RuntimeEventKindV2.RUN_STARTED)

    session_event = store.append_runtime_event(
        session_ref=created.session.session_ref,
        runtime_event=runtime_event,
        audit=_audit(),
        idempotency_key="runtime-event-one",
    )
    replay = store.append_runtime_event(
        session_ref=created.session.session_ref,
        runtime_event=runtime_event,
        audit=_audit(),
        idempotency_key="runtime-event-one",
    )
    second_key_replay = store.append_runtime_event(
        session_ref=created.session.session_ref,
        runtime_event=runtime_event,
        audit=_audit(),
        idempotency_key="runtime-event-one-second-key",
    )

    assert replay == session_event
    assert second_key_replay == session_event
    assert isinstance(session_event.payload, SessionRuntimePayloadV1)
    assert session_event.payload.event_kind is RuntimeEventKindV2.RUN_STARTED
    assert store.get_runtime_event(session_event.payload.runtime_event_ref) == runtime_event
    reopened = HarnessSessionStore(path, clock=lambda: NOW)
    assert reopened.get_runtime_event(session_event.payload.runtime_event_ref) == runtime_event
    assert reopened.get_projection("session-runtime-events").session.last_event_sequence == 2


def test_runtime_event_sequence_and_terminal_are_fail_closed(
    tmp_path: Path,
) -> None:
    store = HarnessSessionStore(tmp_path / "harness.sqlite3", clock=lambda: NOW)
    created = _created(store)
    with pytest.raises(HarnessSessionConflictError, match="contiguous"):
        store.append_runtime_event(
            session_ref=created.session.session_ref,
            runtime_event=_runtime_event(2, RuntimeEventKindV2.RUN_STARTED),
            audit=_audit(),
            idempotency_key="runtime-gap",
        )
    with pytest.raises(HarnessSessionConflictError, match="start with"):
        store.append_runtime_event(
            session_ref=created.session.session_ref,
            runtime_event=_runtime_event(1, RuntimeEventKindV2.WARNING),
            audit=_audit(),
            idempotency_key="runtime-wrong-start",
        )

    store.append_runtime_event(
        session_ref=created.session.session_ref,
        runtime_event=_runtime_event(1, RuntimeEventKindV2.RUN_STARTED),
        audit=_audit(),
        idempotency_key="runtime-start",
    )
    with pytest.raises(HarnessSessionConflictError, match="cannot restart"):
        store.append_runtime_event(
            session_ref=created.session.session_ref,
            runtime_event=_runtime_event(
                2,
                RuntimeEventKindV2.RUN_STARTED,
                source_label="second-start",
            ),
            audit=_audit(),
            idempotency_key="runtime-second-start",
        )
    with pytest.raises(HarnessSessionConflictError, match="prior usage"):
        store.append_runtime_event(
            session_ref=created.session.session_ref,
            runtime_event=_runtime_event(2, RuntimeEventKindV2.RUN_COMPLETED),
            audit=_audit(),
            idempotency_key="runtime-terminal-without-usage",
        )
    store.append_runtime_event(
        session_ref=created.session.session_ref,
        runtime_event=_runtime_event(2, RuntimeEventKindV2.USAGE_REPORTED),
        audit=_audit(),
        idempotency_key="runtime-usage",
    )
    store.append_runtime_event(
        session_ref=created.session.session_ref,
        runtime_event=_runtime_event(3, RuntimeEventKindV2.RUN_COMPLETED),
        audit=_audit(),
        idempotency_key="runtime-terminal",
    )
    with pytest.raises(HarnessSessionConflictError, match="terminal"):
        store.append_runtime_event(
            session_ref=created.session.session_ref,
            runtime_event=_runtime_event(4, RuntimeEventKindV2.WARNING),
            audit=_audit(),
            idempotency_key="runtime-after-terminal",
        )


def test_runtime_event_changed_key_cross_session_and_drift_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "harness.sqlite3"
    store = HarnessSessionStore(path, clock=lambda: NOW)
    first = _created(store)
    second = _created(store, session_id="session-runtime-events-other")
    runtime_event = _runtime_event(1, RuntimeEventKindV2.RUN_STARTED)
    stored = store.append_runtime_event(
        session_ref=first.session.session_ref,
        runtime_event=runtime_event,
        audit=_audit(),
        idempotency_key="runtime-shared-key",
    )

    with pytest.raises(HarnessSessionConflictError, match="idempotency key"):
        store.append_runtime_event(
            session_ref=first.session.session_ref,
            runtime_event=_runtime_event(2, RuntimeEventKindV2.WARNING),
            audit=_audit(),
            idempotency_key="runtime-shared-key",
        )
    with pytest.raises(HarnessSessionConflictError, match="another session"):
        store.append_runtime_event(
            session_ref=second.session.session_ref,
            runtime_event=runtime_event,
            audit=_audit(),
            idempotency_key="runtime-cross-session",
        )

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            UPDATE harness_runtime_events
            SET event_kind = 'WARNING'
            WHERE runtime_event_object_id = ?
            """,
            (runtime_event.runtime_event_id,),
        )
        connection.commit()
    finally:
        connection.close()
    assert isinstance(stored.payload, SessionRuntimePayloadV1)
    with pytest.raises(HarnessSessionIntegrityError, match="columns drifted"):
        store.get_runtime_event(stored.payload.runtime_event_ref)


def test_runtime_event_projection_rebuild_preserves_public_event(
    tmp_path: Path,
) -> None:
    store = HarnessSessionStore(tmp_path / "harness.sqlite3", clock=lambda: NOW)
    created = _created(store)
    session_event = store.append_runtime_event(
        session_ref=created.session.session_ref,
        runtime_event=_runtime_event(1, RuntimeEventKindV2.RUN_STARTED),
        audit=_audit(),
        idempotency_key="runtime-before-rebuild",
    )

    rebuilt = store.rebuild_projection("session-runtime-events")

    assert rebuilt.session.last_event_sequence == 2
    assert rebuilt.event_refs[-1] == session_event.to_ref()


def test_runtime_event_transaction_fault_rolls_back_every_surface(
    tmp_path: Path,
) -> None:
    path = tmp_path / "harness.sqlite3"
    store = HarnessSessionStore(path, clock=lambda: NOW)
    created = _created(store)
    faulted = HarnessSessionStore(
        path,
        clock=lambda: NOW,
        fault_injector=lambda point: (
            (_ for _ in ()).throw(RuntimeError("injected"))
            if point == "before_runtime_event_commit"
            else None
        ),
    )

    with pytest.raises(RuntimeError, match="injected"):
        faulted.append_runtime_event(
            session_ref=created.session.session_ref,
            runtime_event=_runtime_event(1, RuntimeEventKindV2.RUN_STARTED),
            audit=_audit(),
            idempotency_key="runtime-fault",
        )

    projection = store.get_projection("session-runtime-events")
    assert projection.session.last_event_sequence == 1
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM harness_runtime_events",
        ).fetchone() == (0,)
    finally:
        connection.close()
