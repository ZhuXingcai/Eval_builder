from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from tests.eval_factory.integration.test_agent_shell_sse import _shell
from tests.eval_factory.unit.test_harness_source_admission import (
    _manifest,
    _setup,
    _trace,
)

from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.console_api import (
    AgentShellSourceAdmissionCommandV1,
    AgentShellSourceFileClaimV1,
    create_agent_app,
)


def _command(
    manifest: bytes,
    trace: bytes,
    *,
    expected_session_version: int = 1,
) -> AgentShellSourceAdmissionCommandV1:
    return AgentShellSourceAdmissionCommandV1(
        expected_session_version=expected_session_version,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        manifest_size_bytes=len(manifest),
        trace_files=(
            AgentShellSourceFileClaimV1(
                relative_name="LH_001_sid-001.jsonl",
                expected_sha256=hashlib.sha256(trace).hexdigest(),
                expected_size_bytes=len(trace),
            ),
        ),
        idempotency_key="source-admission-api",
    )


def _post(
    client: TestClient,
    command: AgentShellSourceAdmissionCommandV1,
    manifest: bytes,
    trace: bytes,
):
    return client.post(
        "/api/harness/sessions/session-source-admission/sources",
        headers={"X-Eval-Factory-Principal": "user://source-admission-test"},
        data={"command": command.model_dump_json()},
        files=(
            ("manifest", ("manifest.csv", manifest, "text/csv")),
            (
                "traces",
                (
                    "LH_001_sid-001.jsonl",
                    trace,
                    "application/x-ndjson",
                ),
            ),
        ),
    )


def test_source_admission_multipart_api_and_safe_projection(
    tmp_path: Path,
) -> None:
    source_service, _ = _setup(tmp_path)
    shell, _ = _shell(tmp_path / "unused-shell")
    app = create_agent_app(
        PlanReviewService.__new__(PlanReviewService),
        shell,
        source_service,
    )
    client = TestClient(app)
    manifest = _manifest()
    trace = _trace()

    contract = client.get("/api/harness/source-admission/contract")
    admitted = _post(client, _command(manifest, trace), manifest, trace)
    loaded = client.get(
        "/api/harness/sessions/session-source-admission/sources",
    )

    assert contract.status_code == 200
    assert contract.json()["manifest_name"] == "manifest.csv"
    assert admitted.status_code == 200, admitted.text
    assert loaded.status_code == 200
    assert loaded.json() == admitted.json()
    assert admitted.json()["source_count"] == 1
    serialized = admitted.text
    assert str(tmp_path) not in serialized
    assert '"request"' not in serialized
    assert "harness-source-content" not in serialized
    openapi = json.dumps(app.openapi(), sort_keys=True)
    assert "manifest_content_ref" not in openapi
    assert "content_ref" not in openapi
    assert "physical_path" not in openapi


def test_source_admission_api_returns_closed_errors(
    tmp_path: Path,
) -> None:
    shell, _ = _shell(tmp_path / "shell")
    unavailable = TestClient(
        create_agent_app(
            PlanReviewService.__new__(PlanReviewService),
            shell,
        )
    )
    assert (
        unavailable.get(
            "/api/harness/source-admission/contract",
        ).json()["error_code"]
        == "SOURCE_ADMISSION_NOT_CONFIGURED"
    )

    source_service, _ = _setup(tmp_path / "configured")
    configured_shell, _ = _shell(tmp_path / "other-shell")
    client = TestClient(
        create_agent_app(
            PlanReviewService.__new__(PlanReviewService),
            configured_shell,
            source_service,
        )
    )
    manifest = _manifest()
    trace = _trace()
    missing = client.get(
        "/api/harness/sessions/session-source-admission/sources",
    )
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "SOURCE_ADMISSION_NOT_FOUND"

    stale = _post(
        client,
        _command(
            manifest,
            trace,
            expected_session_version=2,
        ),
        manifest,
        trace,
    )
    assert stale.status_code == 409
    assert stale.json()["error_code"] == "HARNESS_SESSION_STALE"

    wrong_hash = _command(manifest, trace).model_copy(
        update={
            "trace_files": (
                AgentShellSourceFileClaimV1(
                    relative_name="LH_001_sid-001.jsonl",
                    expected_sha256="f" * 64,
                    expected_size_bytes=len(trace),
                ),
            ),
        },
    )
    rejected = _post(client, wrong_hash, manifest, trace)
    assert rejected.status_code == 422
    assert rejected.json()["error_code"] == ("SOURCE_ADMISSION_HASH_MISMATCH")

    malformed = client.post(
        "/api/harness/sessions/session-source-admission/sources",
        headers={
            "X-Eval-Factory-Principal": "user://source-admission-test",
        },
        data={
            "command": _command(manifest, trace).model_dump_json(),
            "unexpected": "field",
        },
        files=(
            ("manifest", ("manifest.csv", manifest, "text/csv")),
            (
                "traces",
                (
                    "LH_001_sid-001.jsonl",
                    trace,
                    "application/x-ndjson",
                ),
            ),
        ),
    )
    assert malformed.status_code == 422
    assert malformed.json()["error_code"] == ("SOURCE_ADMISSION_MULTIPART_INVALID")

    parser_rejected = client.post(
        "/api/harness/sessions/session-source-admission/sources",
        headers={
            "X-Eval-Factory-Principal": "user://source-admission-test",
        },
        data={"command": "x" * ((512 * 1024) + 1)},
        files=(
            ("manifest", ("manifest.csv", manifest, "text/csv")),
            (
                "traces",
                (
                    "LH_001_sid-001.jsonl",
                    trace,
                    "application/x-ndjson",
                ),
            ),
        ),
    )
    assert parser_rejected.status_code == 422
    assert parser_rejected.json()["error_code"] == ("SOURCE_ADMISSION_MULTIPART_INVALID")

    oversized = client.post(
        "/api/harness/sessions/session-source-admission/sources",
        headers={
            "X-Eval-Factory-Principal": "user://source-admission-test",
            "Content-Length": str(source_service.limits.max_request_bytes + 1),
        },
        data={"command": _command(manifest, trace).model_dump_json()},
        files=(
            ("manifest", ("manifest.csv", manifest, "text/csv")),
            (
                "traces",
                (
                    "LH_001_sid-001.jsonl",
                    trace,
                    "application/x-ndjson",
                ),
            ),
        ),
    )
    assert oversized.status_code == 413
    assert oversized.json()["error_code"] == ("SOURCE_ADMISSION_LIMIT_EXCEEDED")
