from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from tests.eval_factory.integration.test_harness_agent_loop import (
    _audit,
    _components,
    _ref,
)

from env_mock_agent.facade import (
    FacadeObjectRef,
    RuntimeEventKindV2,
    UnifiedRuntimeEventV2,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.console_api import (
    AgentShellCreateSessionCommandV1,
    AgentShellService,
    create_agent_app,
)
from eval_factory.console_api.sse import stream_session_events
from eval_factory.harness import (
    HarnessSessionStore,
    RequirementInterpretationProposalV1,
)

NOW = datetime(2026, 8, 23, tzinfo=UTC)


def _shell(root: Path) -> tuple[AgentShellService, HarnessSessionStore]:
    proposal = RequirementInterpretationProposalV1(
        outcome="CLARIFICATION_REQUIRED",
        assistant_message="请补充数据来源。",
        missing_field_codes=("SOURCE_EXPECTATION_MISSING",),
        clarification_questions=("数据来源是什么?",),
    )
    sessions, store, _, _ = _components(root, proposal)
    shell = AgentShellService(
        sessions=sessions,
        composition_ref=_ref("harness-composition", version="v1"),
        principal_resolver=lambda _: _ref("principal", version="v1"),
        governing_versions=_audit().governing_versions,
        clock=lambda: NOW,
    )
    return shell, store


class _RequestState:
    def __init__(self, *, disconnected: bool = False) -> None:
        self.disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _request(*, disconnected: bool = False) -> Request:
    return cast(Request, _RequestState(disconnected=disconnected))


def _create_session(shell: AgentShellService, session_id: str):
    return shell.create_session(
        AgentShellCreateSessionCommandV1(
            session_id=session_id,
            incarnation_id=f"{session_id}-incarnation",
            idempotency_key=f"create-{session_id}",
        ),
        principal="user://agent-shell",
    )


def _runtime_event() -> UnifiedRuntimeEventV2:
    digest = hashlib.sha256(b"runtime-source").hexdigest()
    return UnifiedRuntimeEventV2.create(
        run_id="run-shell-api",
        work_id="work-shell-api",
        runtime_id="pi_rpc",
        runtime_version="0.80.10",
        sequence=1,
        kind=RuntimeEventKindV2.RUN_STARTED,
        occurred_at=NOW,
        source_refs=(
            FacadeObjectRef(
                object_type="runtime-event-content",
                object_id=f"runtime-event-content://sha256/{digest}",
                object_version="private-v1",
                object_sha256=digest,
            ),
        ),
    )


def test_agent_shell_http_projection_and_sse_reconnect(
    tmp_path: Path,
) -> None:
    shell, store = _shell(tmp_path)
    app = create_agent_app(
        PlanReviewService.__new__(PlanReviewService),
        shell,
    )
    client = TestClient(app)
    principal = "user://agent-shell"
    headers = {"X-Eval-Factory-Principal": principal}

    created = client.post(
        "/api/harness/sessions",
        headers=headers,
        json={
            "session_id": "session-shell-api",
            "incarnation_id": "session-shell-api-incarnation",
            "idempotency_key": "create-session-shell-api",
        },
    )
    assert created.status_code == 200, created.json()
    posted = client.post(
        "/api/harness/sessions/session-shell-api/messages",
        headers=headers,
        json={
            "expected_session_version": 1,
            "content": "请生产评测数据。",
            "artifact_envelope_refs": [],
            "idempotency_key": "message-session-shell-api",
        },
    )
    assert posted.status_code == 200, posted.json()
    assert posted.json()["session"]["title"] == "请生产评测数据。"

    projection = store.get_projection("session-shell-api")
    store.append_runtime_event(
        session_ref=projection.session.session_ref,
        runtime_event=_runtime_event(),
        audit=_audit(),
        idempotency_key="runtime-session-shell-api",
    )

    events = client.get(
        "/api/harness/sessions/session-shell-api/events",
        params={"after_sequence": 0, "limit": 100},
    )
    assert events.status_code == 200
    assert events.json()["events"][-1]["family"] == "RUNTIME"
    assert events.json()["events"][-1]["runtime_id"] == "pi_rpc"

    stream = client.get(
        "/api/harness/sessions/session-shell-api/stream",
        params={"after_sequence": 0, "follow": "false"},
    )
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert "event: session-event" in stream.text
    assert "id: 1" in stream.text
    assert "runtime-event-content" not in stream.text

    watermark = store.get_projection(
        "session-shell-api",
    ).session.last_event_sequence
    exact_reconnect = client.get(
        "/api/harness/sessions/session-shell-api/stream",
        headers={"Last-Event-ID": str(watermark - 1)},
        params={"follow": "false"},
    )
    assert exact_reconnect.status_code == 200
    assert f"id: {watermark}\n" in exact_reconnect.text

    reconnect = client.get(
        "/api/harness/sessions/session-shell-api/stream",
        headers={"Last-Event-ID": "1"},
        params={"after_sequence": 2, "follow": "false"},
    )
    assert reconnect.status_code == 409
    assert "INVALID_SSE_CURSOR" in reconnect.text

    malformed = client.get(
        "/api/harness/sessions/session-shell-api/stream",
        headers={"Last-Event-ID": "01"},
        params={"follow": "false"},
    )
    assert malformed.status_code == 409
    non_decimal = client.get(
        "/api/harness/sessions/session-shell-api/stream",
        headers={"Last-Event-ID": "latest"},
        params={"follow": "false"},
    )
    assert non_decimal.status_code == 409
    ahead = client.get(
        "/api/harness/sessions/session-shell-api/stream",
        headers={"Last-Event-ID": str(watermark + 1)},
        params={"follow": "false"},
    )
    assert ahead.status_code == 409

    sessions = client.get("/api/harness/sessions")
    assert sessions.status_code == 200
    assert sessions.json()["sessions"][0]["session_id"] == "session-shell-api"
    assert client.get("/api/harness/contract").status_code == 200
    assert client.get("/api/plan-reviews/contract").status_code == 200


def test_agent_shell_http_close_is_idempotent_and_auditable(
    tmp_path: Path,
) -> None:
    shell, store = _shell(tmp_path)
    client = TestClient(
        create_agent_app(
            PlanReviewService.__new__(PlanReviewService),
            shell,
        )
    )
    headers = {"X-Eval-Factory-Principal": "user://agent-shell"}
    created = client.post(
        "/api/harness/sessions",
        headers=headers,
        json={
            "session_id": "session-shell-close-api",
            "incarnation_id": "session-shell-close-api-incarnation",
            "idempotency_key": "create-session-shell-close-api",
        },
    )
    assert created.status_code == 200
    command = {
        "expected_session_version": 1,
        "idempotency_key": "close-session-shell-close-api",
    }

    closed = client.request(
        "DELETE",
        "/api/harness/sessions/session-shell-close-api",
        headers=headers,
        json=command,
    )
    replay = client.request(
        "DELETE",
        "/api/harness/sessions/session-shell-close-api",
        headers=headers,
        json=command,
    )

    assert closed.status_code == 200, closed.text
    assert closed.json()["session"]["status"] == "CLOSED"
    assert replay.json() == closed.json()
    assert client.get("/api/harness/sessions").json()["total"] == 0
    retained = client.get(
        "/api/harness/sessions/session-shell-close-api",
    )
    assert retained.status_code == 200
    assert retained.json()["session"]["status"] == "CLOSED"
    assert (
        store.list_events(
            "session-shell-close-api",
            after_sequence=0,
            limit=100,
        ).events[-1].payload.event_kind.value
        == "SESSION_CLOSED"
    )

    conflict = client.request(
        "DELETE",
        "/api/harness/sessions/session-shell-close-api",
        headers=headers,
        json={
            "expected_session_version": 1,
            "idempotency_key": "close-session-shell-close-api-again",
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["error_code"] == "HARNESS_SESSION_CONFLICT"


def test_agent_shell_public_contract_rejects_private_fields(
    tmp_path: Path,
) -> None:
    shell, _ = _shell(tmp_path)
    client = TestClient(
        create_agent_app(
            PlanReviewService.__new__(PlanReviewService),
            shell,
        )
    )
    response = client.post(
        "/api/harness/sessions",
        headers={"X-Eval-Factory-Principal": "user://agent-shell"},
        json={
            "session_id": "session-shell-private",
            "incarnation_id": "session-shell-private-incarnation",
            "idempotency_key": "create-session-shell-private",
            "provider_response_body": "private",
        },
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_COMMAND"


def test_agent_shell_maps_session_errors_to_closed_http_responses(
    tmp_path: Path,
) -> None:
    shell, store = _shell(tmp_path)
    client = TestClient(
        create_agent_app(
            PlanReviewService.__new__(PlanReviewService),
            shell,
        )
    )
    headers = {"X-Eval-Factory-Principal": "user://agent-shell"}
    assert (
        client.get("/api/harness/sessions/missing-session").json()["error_code"]
        == "HARNESS_SESSION_NOT_FOUND"
    )
    created = client.post(
        "/api/harness/sessions",
        headers=headers,
        json={
            "session_id": "session-shell-errors",
            "incarnation_id": "session-shell-errors-incarnation",
            "idempotency_key": "create-session-shell-errors",
        },
    )
    assert created.status_code == 200
    assert client.get("/api/harness/sessions/session-shell-errors").status_code == 200
    stale = client.post(
        "/api/harness/sessions/session-shell-errors/messages",
        headers=headers,
        json={
            "expected_session_version": 2,
            "content": "stale command",
            "idempotency_key": "stale-session-shell-errors",
        },
    )
    assert stale.status_code == 409
    assert stale.json()["error_code"] == "HARNESS_SESSION_STALE"
    ahead = client.get(
        "/api/harness/sessions/session-shell-errors/events",
        params={"after_sequence": 2},
    )
    assert ahead.status_code == 409
    assert ahead.json()["error_code"] == "HARNESS_SESSION_CONFLICT"

    connection = sqlite3.connect(store.path)
    try:
        connection.execute(
            """
            UPDATE harness_session_projections
            SET projection_json = '{}'
            WHERE session_id = 'session-shell-errors'
            """
        )
        connection.commit()
    finally:
        connection.close()
    corrupt = client.get("/api/harness/sessions/session-shell-errors")
    assert corrupt.status_code == 500
    assert corrupt.json()["error_code"] == "HARNESS_SESSION_INTEGRITY"


@pytest.mark.asyncio
async def test_follow_stream_observes_new_commits_and_heartbeat_is_ephemeral(
    tmp_path: Path,
) -> None:
    shell, store = _shell(tmp_path)
    created = _create_session(shell, "session-shell-follow")
    cursor = created.reconnect_cursor
    stream = stream_session_events(
        service=shell,
        session_id=created.session.session_id,
        request=_request(),
        after_sequence=cursor,
        follow=True,
        poll_interval_seconds=0.001,
        heartbeat_seconds=10,
    )
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.01)
    store.append_runtime_event(
        session_ref=store.get_projection(
            created.session.session_id,
        ).session.session_ref,
        runtime_event=_runtime_event(),
        audit=_audit(),
        idempotency_key="runtime-session-shell-follow",
    )

    live_event = await asyncio.wait_for(pending, timeout=1)
    assert b"event: session-event" in live_event
    assert b'"family":"RUNTIME"' in live_event
    await stream.aclose()

    heartbeat_stream = stream_session_events(
        service=shell,
        session_id=created.session.session_id,
        request=_request(),
        after_sequence=store.get_projection(
            created.session.session_id,
        ).session.last_event_sequence,
        follow=True,
        poll_interval_seconds=0.001,
        heartbeat_seconds=0.001,
    )
    heartbeat = await asyncio.wait_for(anext(heartbeat_stream), timeout=1)
    assert heartbeat.startswith(b"event: heartbeat\n")
    assert b"\nid:" not in heartbeat
    assert (
        store.get_projection(
            created.session.session_id,
        ).session.last_event_sequence
        == cursor + 1
    )
    await heartbeat_stream.aclose()


@pytest.mark.asyncio
async def test_disconnect_stops_observer_and_slow_consumer_does_not_block_commit(
    tmp_path: Path,
) -> None:
    shell, store = _shell(tmp_path)
    created = _create_session(shell, "session-shell-disconnect")
    disconnected = stream_session_events(
        service=shell,
        session_id=created.session.session_id,
        request=_request(disconnected=True),
        after_sequence=created.reconnect_cursor,
        follow=True,
        poll_interval_seconds=0.001,
        heartbeat_seconds=10,
    )
    with pytest.raises(StopAsyncIteration):
        await anext(disconnected)

    slow = stream_session_events(
        service=shell,
        session_id=created.session.session_id,
        request=_request(),
        after_sequence=0,
        follow=True,
        poll_interval_seconds=0.001,
        heartbeat_seconds=10,
    )
    first = await anext(slow)
    assert first.startswith(b"id: 1\n")
    store.append_runtime_event(
        session_ref=store.get_projection(
            created.session.session_id,
        ).session.session_ref,
        runtime_event=_runtime_event(),
        audit=_audit(),
        idempotency_key="runtime-session-shell-disconnect",
    )
    assert (
        store.get_projection(
            created.session.session_id,
        ).session.last_event_sequence
        == created.reconnect_cursor + 1
    )
    await slow.aclose()

    invalid_timing = stream_session_events(
        service=shell,
        session_id=created.session.session_id,
        request=_request(),
        after_sequence=0,
        follow=True,
        poll_interval_seconds=0,
        heartbeat_seconds=10,
    )
    with pytest.raises(ValueError, match="positive"):
        await anext(invalid_timing)
