from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import RuntimeEventKindV2
from eval_factory.console_api.agent_contracts import (
    AgentShellCloseSessionCommandV1,
    AgentShellCreateSessionCommandV1,
    AgentShellEventFamilyV1,
    AgentShellEventV1,
    AgentShellGraphSummaryV1,
    AgentShellInteractionCardV1,
    AgentShellInteractionKindV1,
    AgentShellInteractionStateV1,
    AgentShellPostMessageCommandV1,
    AgentShellSourceAdmissionCommandV1,
    AgentShellSourceAdmissionContractV1,
    AgentShellSourceFileClaimV1,
    AgentShellSourceFileKindV1,
    AgentShellSourceFileV1,
    AgentShellWorkspaceDescriptorV1,
    AgentShellWorkspaceKindV1,
    AgentShellWorkspaceStatusV1,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.harness.session_models import SessionMessageEventKindV1

NOW = datetime(2026, 8, 23, tzinfo=UTC)


def test_agent_shell_commands_are_strict_and_bounded() -> None:
    create = AgentShellCreateSessionCommandV1(
        session_id="session-shell-001",
        incarnation_id="session-shell-incarnation-001",
        idempotency_key="create-shell-001",
    )
    assert create.session_id == "session-shell-001"

    close = AgentShellCloseSessionCommandV1(
        expected_session_version=1,
        idempotency_key="close-shell-001",
    )
    assert close.expected_session_version == 1

    message = AgentShellPostMessageCommandV1(
        expected_session_version=1,
        content="构建一个可审计的 Agent 评测集。",
        idempotency_key="message-shell-001",
    )
    assert message.artifact_envelope_refs == ()

    with pytest.raises(ValidationError):
        AgentShellCreateSessionCommandV1.model_validate(
            {
                **create.model_dump(mode="json"),
                "physical_path": "/private/tmp/session.sqlite3",
            }
        )
    with pytest.raises(ValidationError):
        AgentShellPostMessageCommandV1.model_validate(
            {
                **message.model_dump(mode="json"),
                "provider_response_body": "private",
            }
        )
    with pytest.raises(ValidationError):
        AgentShellPostMessageCommandV1(
            expected_session_version=1,
            content="x" * 32_769,
            idempotency_key="message-shell-too-large",
        )
    with pytest.raises(ValidationError):
        AgentShellCloseSessionCommandV1.model_validate(
            {
                **close.model_dump(mode="json"),
                "physical_delete": True,
            }
        )


def test_public_event_rejects_runtime_fields_on_non_runtime_family() -> None:
    with pytest.raises(ValidationError, match="event kind"):
        AgentShellEventV1(
            event_id="session-event://shell/family-mismatch",
            sequence=1,
            authority_version=1,
            family=AgentShellEventFamilyV1.MESSAGE,
            event_kind=RuntimeEventKindV2.RUN_STARTED,
            occurred_at=NOW,
        )

    with pytest.raises(ValidationError, match="runtime fields"):
        AgentShellEventV1(
            event_id="session-event://shell/1",
            sequence=1,
            authority_version=1,
            family=AgentShellEventFamilyV1.MESSAGE,
            event_kind=SessionMessageEventKindV1.USER_MESSAGE,
            occurred_at=NOW,
            runtime_id="pi_rpc",
        )

    with pytest.raises(ValidationError, match="runtime_id"):
        AgentShellEventV1(
            event_id="session-event://shell/2",
            sequence=2,
            authority_version=2,
            family=AgentShellEventFamilyV1.RUNTIME,
            event_kind=RuntimeEventKindV2.RUN_STARTED,
            occurred_at=NOW,
        )


def test_source_admission_command_is_strict_sorted_and_path_free() -> None:
    first = AgentShellSourceFileClaimV1(
        relative_name="LH_001_sid-001.jsonl",
        expected_sha256="a" * 64,
        expected_size_bytes=100,
    )
    second = AgentShellSourceFileClaimV1(
        relative_name="LH_002_sid-002.jsonl",
        expected_sha256="b" * 64,
        expected_size_bytes=200,
    )
    command = AgentShellSourceAdmissionCommandV1(
        expected_session_version=1,
        manifest_sha256="c" * 64,
        manifest_size_bytes=300,
        trace_files=(first, second),
        idempotency_key="source-admission-001",
    )
    assert command.trace_files == (first, second)

    with pytest.raises(ValidationError, match="sorted"):
        AgentShellSourceAdmissionCommandV1.model_validate(
            {
                **command.model_dump(mode="python"),
                "trace_files": (second, first),
            }
        )
    with pytest.raises(ValidationError):
        AgentShellSourceFileClaimV1(
            relative_name="../private.jsonl",
            expected_sha256="a" * 64,
            expected_size_bytes=100,
        )
    with pytest.raises(ValidationError, match="unsafe"):
        AgentShellSourceFileClaimV1(
            relative_name="trace..private.jsonl",
            expected_sha256="a" * 64,
            expected_size_bytes=100,
        )
    with pytest.raises(ValidationError, match="extension"):
        AgentShellSourceFileClaimV1(
            relative_name="trace.JSONL",
            expected_sha256="a" * 64,
            expected_size_bytes=100,
        )
    alias = AgentShellSourceFileClaimV1(
        relative_name="lh_001_sid-001.jsonl",
        expected_sha256="d" * 64,
        expected_size_bytes=100,
    )
    with pytest.raises(ValidationError, match="aliased"):
        AgentShellSourceAdmissionCommandV1(
            expected_session_version=1,
            manifest_sha256="c" * 64,
            manifest_size_bytes=300,
            trace_files=(first, alias),
            idempotency_key="source-admission-alias",
        )
    with pytest.raises(ValidationError):
        AgentShellSourceAdmissionCommandV1.model_validate(
            {
                **command.model_dump(mode="json"),
                "staging_path": "/private/tmp/upload",
            }
        )

    with pytest.raises(ValidationError, match="invalid"):
        AgentShellSourceFileV1(
            kind=AgentShellSourceFileKindV1.TRACE,
            relative_name="LH_001_sid-001.jsonl",
            media_type="application/x-ndjson",
            size_bytes=100,
            sha256="a" * 64,
            source_ref=ObjectRef(
                object_type="trace-source",
                object_id="source-trace://agent-upload/source-001",
                object_version="v2",
                object_sha256="b" * 64,
            ),
            artifact_envelope_ref=ObjectRef(
                object_type="artifact-envelope",
                object_id="artifact-envelope://source-001",
                object_version="v1",
                object_sha256="c" * 64,
            ),
        )
    with pytest.raises(ValidationError, match="inconsistent"):
        AgentShellSourceAdmissionContractV1(
            max_source_files=10,
            max_manifest_bytes=100,
            max_source_bytes=1_000,
            max_total_source_bytes=500,
            max_request_bytes=10_000,
        )


def test_composition_contracts_reject_incomplete_public_state() -> None:
    binding_ref = ObjectRef(
        object_type="graph-execution-binding",
        object_id="graph-execution-binding://shell",
        object_version="v1",
        object_sha256="a" * 64,
    )
    with pytest.raises(ValidationError, match="transition state"):
        AgentShellGraphSummaryV1(
            binding_ref=binding_ref,
            transition_number=1,
        )
    with pytest.raises(ValidationError, match="permission"):
        AgentShellInteractionCardV1(
            interaction_id="permission-shell",
            kind=AgentShellInteractionKindV1.PERMISSION,
            state=AgentShellInteractionStateV1.BLOCKED,
            subject_ref=ObjectRef(
                object_type="requirement-interpretation",
                object_id="requirement-interpretation://shell",
                object_version="v1",
                object_sha256="b" * 64,
            ),
        )
    with pytest.raises(ValidationError, match="empty workspace"):
        AgentShellWorkspaceDescriptorV1(
            kind=AgentShellWorkspaceKindV1.TEAM,
            status=AgentShellWorkspaceStatusV1.EMPTY,
            owner_refs=(binding_ref,),
            item_count=1,
        )


def test_agent_shell_contract_root_remains_a_cold_import() -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import sys; import eval_factory.console_api; "
                "blocked = [name for name in sys.modules "
                "if name.endswith('.agent_system.candidate_output') "
                "or name == 'claude_agent_sdk']; "
                "print('\\n'.join(blocked))"
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
