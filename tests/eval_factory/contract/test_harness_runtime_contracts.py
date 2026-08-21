from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.ai_gateway_v2 import GatewayInvocationStatusV2, GatewayUsageV2
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness import (
    GatewayJournalStateV1,
    HarnessGatewayJournalV1,
    HarnessMessageRoleV1,
    HarnessMessageV1,
    HarnessSessionIdentityV1,
    HarnessSessionRefV1,
    HarnessSessionStateV1,
    HarnessSessionStatusV1,
    ProviderEvidenceClassV1,
    RequirementInterpretationProposalV1,
    SessionEventV1,
    SessionLifecycleEventKindV1,
    SessionLifecyclePayloadV1,
    validate_session_event_log,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _ref(object_type: str, *, version: str = "v1") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://example/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="harness-runtime-contract-test",
        governing_versions=(VersionBinding(component="evaluation-agent-harness", version="v1", sha256=HASH),),
    )


def test_session_state_and_message_are_strict_and_hash_bound() -> None:
    identity = HarnessSessionIdentityV1.create(
        session_id="session-runtime-001",
        incarnation_id="session-runtime-incarnation-001",
        composition_ref=_ref("harness-composition"),
        created_by="runtime-user",
        created_at=NOW,
        audit=_audit(),
    )
    state = HarnessSessionStateV1.create(
        identity=identity,
        session_version=1,
        status=HarnessSessionStatusV1.ACTIVE,
        last_event_sequence=1,
        current_interpretation_ref=None,
        current_requirement_policy_ref=None,
        updated_at=NOW,
    )
    message = HarnessMessageV1.create(
        message_id="harness-message-runtime-001",
        session_ref=identity.to_ref(),
        role=HarnessMessageRoleV1.USER,
        content="生成通用 Agent 评测数据",
        artifact_envelope_refs=(),
        chunk_index=None,
        final=True,
        created_at=NOW,
        audit=_audit(),
    )
    assert state.to_session_ref().session_version == 1
    assert message.content_sha256 != HASH

    stale = state.model_dump(mode="python")
    stale["state_sha256"] = HASH
    with pytest.raises(ValidationError, match="state hash"):
        HarnessSessionStateV1.model_validate(stale)
    with pytest.raises(ValidationError, match="assistant chunks"):
        HarnessMessageV1.create(
            message_id="harness-message-runtime-invalid",
            session_ref=identity.to_ref(),
            role=HarnessMessageRoleV1.USER,
            content="invalid",
            artifact_envelope_refs=(),
            chunk_index=0,
            final=False,
            created_at=NOW,
            audit=_audit(),
        )


def test_harness_message_normalizes_boundary_whitespace_before_hashing() -> None:
    message = HarnessMessageV1.create(
        message_id="harness-message-runtime-whitespace",
        session_ref=_ref("harness-session"),
        role=HarnessMessageRoleV1.USER,
        content="  first line\nsecond line  \n",
        artifact_envelope_refs=(),
        chunk_index=None,
        final=True,
        created_at=NOW,
        audit=_audit(),
    )

    assert message.content == "first line\nsecond line"
    assert (
        message.content_sha256
        == hashlib.sha256(
            message.content.encode(),
        ).hexdigest()
    )


def test_requirement_proposal_enforces_clarification_and_ready_completeness() -> None:
    clarification = RequirementInterpretationProposalV1(
        outcome="CLARIFICATION_REQUIRED",
        assistant_message="请补充数据来源。",
        missing_field_codes=("SOURCE_EXPECTATION_MISSING",),
        clarification_questions=("数据来源是什么?",),
    )
    assert clarification.goals == ()

    ready = RequirementInterpretationProposalV1(
        outcome="READY",
        assistant_message="需求已具备执行条件。",
        goals=("构建评测集",),
        source_expectations=("用户提供轨迹",),
        target_capabilities=("agent-evaluation",),
        quality_intent="可审计且无答案泄漏",
        delivery_intent="候选数据包",
        max_model_requests=10,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
    )
    assert ready.outcome == "READY"

    with pytest.raises(ValidationError, match="incomplete"):
        RequirementInterpretationProposalV1(
            outcome="READY",
            assistant_message="错误地假装完整。",
            goals=("构建评测集",),
        )


def test_gateway_journal_state_matrix_is_closed() -> None:
    usage = GatewayUsageV2(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=15,
        reported_cost_micro_usd=100,
    )
    common = {
        "session_ref": _ref("harness-session"),
        "command_ref": _ref("session-command"),
        "evidence_class": ProviderEvidenceClassV1.MECHANISM_FIXTURE,
        "route_ref": _ref("model-route-decision", version="v2"),
        "invocation_request_ref": _ref("gateway-invocation-request", version="v2"),
        "prompt_template_ref": _ref("prompt-template", version="v2"),
        "model_profile_ref": _ref("model-capability-profile", version="v2"),
        "audit": _audit(),
    }
    prepared = HarnessGatewayJournalV1.create(
        journal_id="harness-gateway-journal-prepared",
        state=GatewayJournalStateV1.PREPARED,
        predecessor_journal_ref=None,
        receipt_ref=None,
        invocation_result_ref=None,
        output_ref=None,
        gateway_status=None,
        usage=None,
        failure_code=None,
        **common,  # type: ignore[arg-type]
    )
    committed = HarnessGatewayJournalV1.create(
        journal_id="harness-gateway-journal-committed",
        state=GatewayJournalStateV1.COMMITTED,
        predecessor_journal_ref=prepared.to_ref(),
        receipt_ref=_ref("gateway-receipt", version="v2"),
        invocation_result_ref=_ref("gateway-invocation-result", version="v2"),
        output_ref=_ref("requirement-proposal", version="v2"),
        gateway_status=GatewayInvocationStatusV2.SUCCEEDED,
        usage=usage,
        failure_code=None,
        **common,  # type: ignore[arg-type]
    )
    assert committed.predecessor_journal_ref == prepared.to_ref()

    with pytest.raises(ValidationError, match="prepared"):
        HarnessGatewayJournalV1.create(
            journal_id="harness-gateway-journal-invalid",
            state=GatewayJournalStateV1.PREPARED,
            predecessor_journal_ref=None,
            receipt_ref=_ref("gateway-receipt", version="v2"),
            invocation_result_ref=None,
            output_ref=None,
            gateway_status=None,
            usage=None,
            failure_code=None,
            **common,  # type: ignore[arg-type]
        )


def test_event_log_allows_session_versions_to_advance_monotonically() -> None:
    events = tuple(
        SessionEventV1.create(
            event_id=f"session-runtime-event-{sequence}",
            session=HarnessSessionRefV1(
                session_ref=_ref("harness-session"),
                incarnation_id="runtime-incarnation-001",
                session_version=sequence,
                composition_ref=_ref("harness-composition"),
            ),
            sequence=sequence,
            authority_version=sequence,
            turn_id=None,
            step_id=None,
            command_ref=None,
            payload=SessionLifecyclePayloadV1(
                event_kind=SessionLifecycleEventKindV1.SESSION_OPENED,
            ),
            occurred_at=NOW,
            audit=_audit(),
        )
        for sequence in (1, 2)
    )
    validate_session_event_log(events)
