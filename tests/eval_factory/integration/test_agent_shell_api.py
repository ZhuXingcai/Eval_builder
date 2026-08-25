from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.eval_factory.integration.test_agent_shell_graph_flow import (
    RAW_ROOT,
    _host,
    _real_trace_root,
)
from tests.eval_factory.integration.test_agent_shell_sse import (
    _shell as _unconfigured_shell,
)

from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.console_api import (
    AgentShellSourceAdmissionCommandV1,
    AgentShellSourceFileClaimV1,
    create_agent_app,
)

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "manifest.csv").is_file(),
    reason="private 91-trace corpus is not installed",
)


def test_composed_agent_shell_http_routes_are_owner_backed_and_private(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    client = TestClient(
        create_agent_app(
            host.reviews,
            host.shell,
            host.sources,
        )
    )
    headers = {
        "X-Eval-Factory-Principal": "user://agent-shell",
    }
    created = client.post(
        "/api/harness/sessions",
        headers=headers,
        json={
            "session_id": "session-shell-graph",
            "incarnation_id": "session-shell-graph-incarnation",
            "idempotency_key": "create-session-shell-graph",
        },
    )
    assert created.status_code == 200, created.text
    (tmp_path / "source-fixture").mkdir()
    manifest_path, _manifest_sha256 = _real_trace_root(
        tmp_path / "source-fixture",
        instance_ids=("LH_077",),
    )
    manifest = manifest_path.read_bytes()
    trace_path = next(manifest_path.parent.glob("*.jsonl"))
    trace = trace_path.read_bytes()
    source_command = AgentShellSourceAdmissionCommandV1(
        expected_session_version=created.json()["session"]["session_version"],
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        manifest_size_bytes=len(manifest),
        trace_files=(
            AgentShellSourceFileClaimV1(
                relative_name=trace_path.name,
                expected_sha256=hashlib.sha256(trace).hexdigest(),
                expected_size_bytes=len(trace),
            ),
        ),
        idempotency_key="admit-session-shell-graph",
    )
    admitted = client.post(
        "/api/harness/sessions/session-shell-graph/sources",
        headers=headers,
        data={"command": source_command.model_dump_json()},
        files=(
            ("manifest", ("manifest.csv", manifest, "text/csv")),
            (
                "traces",
                (
                    trace_path.name,
                    trace,
                    "application/x-ndjson",
                ),
            ),
        ),
    )
    assert admitted.status_code == 200, admitted.text
    started = client.post(
        "/api/harness/sessions/session-shell-graph/messages",
        headers=headers,
        json={
            "expected_session_version": (created.json()["session"]["session_version"]),
            "content": "Build the admitted generic Agent evaluation dataset.",
            "artifact_envelope_refs": admitted.json()["artifact_envelope_refs"],
            "idempotency_key": "start-session-shell-graph",
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["execution_phase"] == "WAITING_REVIEW"

    team = client.get(
        "/api/harness/sessions/session-shell-graph/team",
    )
    assert team.status_code == 200
    member_id = team.json()["members"][0]["member_id"]
    member = client.get(
        f"/api/harness/sessions/session-shell-graph/members/{member_id}",
    )
    workspace = client.get(
        "/api/harness/sessions/session-shell-graph/workspaces/TEAM",
    )
    delivery = client.get(
        "/api/harness/sessions/session-shell-graph/delivery",
    )
    assert member.status_code == 200
    assert workspace.status_code == 200
    assert workspace.json()["owner_refs"] == [team.json()["team_ref"]]
    assert delivery.status_code == 404
    assert delivery.json()["error_code"] == ("AGENT_SHELL_COMPOSITION_NOT_FOUND")

    reconcile_command = {
        "expected_session_version": body["session"]["session_version"],
        "idempotency_key": "reconcile-session-shell-graph",
    }
    reconciled = client.post(
        "/api/harness/sessions/session-shell-graph/reconcile",
        headers=headers,
        json=reconcile_command,
    )
    replayed = client.post(
        "/api/harness/sessions/session-shell-graph/reconcile",
        headers=headers,
        json=reconcile_command,
    )
    changed = client.post(
        "/api/harness/sessions/session-shell-graph/reconcile",
        headers={
            "X-Eval-Factory-Principal": "user://another-agent-shell",
        },
        json=reconcile_command,
    )
    assert reconciled.status_code == 200, reconciled.text
    assert replayed.json() == reconciled.json()
    assert changed.status_code == 409
    assert changed.json()["error_code"] == "HARNESS_SESSION_CONFLICT"

    public = "\n".join(
        (
            started.text,
            team.text,
            member.text,
            workspace.text,
            json.dumps(client.app.openapi(), sort_keys=True),
        )
    )
    for denied in (
        str(tmp_path),
        "manifest_content_ref",
        "message_body_ref",
        "provider_response_body",
        "physical_path",
        "private-v1",
    ):
        assert denied not in public


def test_unconfigured_composition_routes_fail_closed(
    tmp_path: Path,
) -> None:
    shell, _store = _unconfigured_shell(
        tmp_path / "unconfigured",
    )
    client = TestClient(
        create_agent_app(
            PlanReviewService.__new__(PlanReviewService),
            shell,
        )
    )
    created = client.post(
        "/api/harness/sessions",
        headers={
            "X-Eval-Factory-Principal": "user://agent-shell",
        },
        json={
            "session_id": "session-shell-unconfigured",
            "incarnation_id": "session-shell-unconfigured-incarnation",
            "idempotency_key": "create-session-shell-unconfigured",
        },
    )
    assert created.status_code == 200

    for method, path, payload in (
        (
            client.get,
            "/api/harness/sessions/session-shell-unconfigured/team",
            None,
        ),
        (
            client.get,
            ("/api/harness/sessions/session-shell-unconfigured/workspaces/TEAM"),
            None,
        ),
        (
            client.get,
            "/api/harness/sessions/session-shell-unconfigured/delivery",
            None,
        ),
        (
            client.post,
            "/api/harness/sessions/session-shell-unconfigured/reconcile",
            {
                "expected_session_version": (created.json()["session"]["session_version"]),
                "idempotency_key": "reconcile-unconfigured",
            },
        ),
    ):
        response = (
            method(path)
            if payload is None
            else method(
                path,
                headers={
                    "X-Eval-Factory-Principal": "user://agent-shell",
                },
                json=payload,
            )
        )
        assert response.status_code == 503
        assert response.json()["error_code"] == ("AGENT_SHELL_COMPOSITION_NOT_CONFIGURED")
