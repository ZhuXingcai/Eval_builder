from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    ArtifactEnvelopeV1,
    ArtifactHeadV1,
    ArtifactModalityV1,
    CapabilityCallV1,
    CapabilityConsumerBindingV1,
    CapabilityConsumerKindV1,
    CapabilityInvocationContextV1,
    CapabilityInvocationOutcomeV1,
    CapabilityResultV1,
    HarnessSessionRefV1,
    SessionCommandKindV1,
    SessionCommandV1,
    SessionEventV1,
    SessionLifecycleEventKindV1,
    SessionLifecyclePayloadV1,
    SessionMessageEventKindV1,
    SessionMessagePayloadV1,
    StaticPackRegistrationV1,
    validate_session_commands,
    validate_session_event_log,
)
from eval_factory.packs import build_generic_agent_trace_pack

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit(
    *,
    created_at: datetime | None = None,
    actor: str = "harness-contract-test",
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at or datetime(2026, 8, 17, tzinfo=UTC),
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage0",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _registration() -> StaticPackRegistrationV1:
    return build_generic_agent_trace_pack(audit=_audit())


def _session() -> HarnessSessionRefV1:
    registration = _registration()
    return HarnessSessionRefV1(
        session_ref=_ref("harness-session"),
        incarnation_id="session-incarnation-001",
        session_version=1,
        composition_ref=registration.composition.to_ref(),
    )


def test_harness_objects_are_strict_frozen_and_audit_independent() -> None:
    first = build_generic_agent_trace_pack(
        audit=_audit(created_at=datetime(2026, 8, 17, tzinfo=UTC)),
    )
    second = build_generic_agent_trace_pack(
        audit=_audit(
            created_at=datetime(2027, 1, 1, tzinfo=UTC),
            actor="another-contract-test",
        ),
    )

    assert first.composition.to_ref() == second.composition.to_ref()
    assert first.manifest.to_ref() == second.manifest.to_ref()
    assert len(first.capability_definitions) == 10
    assert any(value.capability_id == "capability.batch-quality" for value in first.capability_definitions)

    definition = first.capability_definitions[0]
    payload = definition.model_dump(mode="python")
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        type(definition).model_validate(payload)

    payload.pop("unknown")
    payload["schema_version"] = "eval-harness/capability-definition/v0"
    with pytest.raises(ValidationError):
        type(definition).model_validate(payload)

    with pytest.raises(ValidationError):
        definition.capability_id = "changed"


def test_session_event_union_and_contiguous_order_are_closed() -> None:
    session = _session()
    first = SessionEventV1.create(
        event_id="session-event-001",
        session=session,
        sequence=1,
        authority_version=1,
        turn_id=None,
        step_id=None,
        command_ref=None,
        payload=SessionLifecyclePayloadV1(
            event_kind=SessionLifecycleEventKindV1.SESSION_OPENED,
        ),
        occurred_at=datetime(2026, 8, 17, tzinfo=UTC),
        audit=_audit(),
    )
    second = SessionEventV1.create(
        event_id="session-event-002",
        session=session,
        sequence=2,
        authority_version=1,
        turn_id="turn-001",
        step_id=None,
        command_ref=None,
        payload=SessionLifecyclePayloadV1(
            event_kind=SessionLifecycleEventKindV1.TURN_STARTED,
        ),
        occurred_at=datetime(2026, 8, 17, 0, 0, 1, tzinfo=UTC),
        audit=_audit(),
    )

    validate_session_event_log((first, second))

    with pytest.raises(ValueError, match="contiguous"):
        validate_session_event_log(
            (
                first,
                second.model_copy(update={"sequence": 3}),
            )
        )
    duplicate_id = SessionEventV1.create(
        event_id=first.event_id,
        session=session,
        sequence=2,
        authority_version=1,
        turn_id=None,
        step_id=None,
        command_ref=None,
        payload=SessionLifecyclePayloadV1(
            event_kind=SessionLifecycleEventKindV1.SESSION_CLOSED,
        ),
        occurred_at=datetime(2026, 8, 17, 0, 0, 2, tzinfo=UTC),
        audit=_audit(),
    )
    with pytest.raises(ValueError, match="unique"):
        validate_session_event_log((first, duplicate_id))

    payload = second.model_dump(mode="python")
    payload["payload"] = {
        "schema_version": "eval-harness/session-message-payload/v1",
        "family": "MESSAGE",
        "event_kind": SessionMessageEventKindV1.USER_MESSAGE,
        "message_ref": _ref("harness-message"),
        "model_visible_artifact_refs": (_ref("raw-trace"),),
    }
    with pytest.raises(ValidationError, match="artifact envelopes"):
        SessionEventV1.model_validate(payload)


def test_model_visible_message_context_uses_artifact_envelopes() -> None:
    payload = SessionMessagePayloadV1(
        event_kind=SessionMessageEventKindV1.USER_MESSAGE,
        message_ref=_ref("harness-message"),
        model_visible_artifact_refs=(_ref("artifact-envelope"),),
    )
    assert payload.model_visible_artifact_refs[0].object_type == "artifact-envelope"

    with pytest.raises(ValidationError, match="artifact envelopes"):
        SessionMessagePayloadV1(
            event_kind=SessionMessageEventKindV1.USER_MESSAGE,
            message_ref=_ref("harness-message"),
            model_visible_artifact_refs=(_ref("raw-trace"),),
        )


def test_session_command_idempotency_binds_exact_authority_and_command() -> None:
    session = _session()
    authority_ref = _ref("execution-authority")
    principal_ref = _ref("principal")
    subject_refs = (_ref("harness-message"),)
    first = SessionCommandV1.create(
        command_id="session-command-001",
        session=session,
        command_kind=SessionCommandKindV1.USER_MESSAGE,
        expected_event_sequence=0,
        authority_ref=authority_ref,
        principal_ref=principal_ref,
        subject_refs=subject_refs,
        idempotency_key="session-command-key-001",
        audit=_audit(),
    )
    replay = SessionCommandV1.model_validate_json(first.canonical_json())
    validate_session_commands((first, replay))

    changed = SessionCommandV1.create(
        command_id="session-command-002",
        session=session,
        command_kind=SessionCommandKindV1.USER_MESSAGE,
        expected_event_sequence=0,
        authority_ref=authority_ref,
        principal_ref=principal_ref,
        subject_refs=subject_refs,
        idempotency_key="session-command-key-001",
        audit=_audit(),
    )
    with pytest.raises(ValueError, match="different commands"):
        validate_session_commands((first, changed))


def test_artifact_envelope_carries_metadata_not_payload() -> None:
    capability_ref = _registration().capability_definitions[0].to_ref()
    envelope = ArtifactEnvelopeV1.create(
        artifact_id="artifact-001",
        subject_ref=_ref("evaluation-requirement-spec", version="v2"),
        schema_ref=_ref("json-schema", version="v2"),
        content_ref=_ref("content-object"),
        media_type="application/json",
        modality=ArtifactModalityV1.DOCUMENT,
        domain_tags=("agent-evaluation", "generic"),
        semantic_role="evaluation-requirement",
        purpose="evaluation-data-production",
        classification="INTERNAL",
        lineage_refs=(),
        producer_capability_ref=capability_ref,
        producer_task_ref=None,
        validation_refs=(),
        revision=1,
        predecessor_envelope_ref=None,
        audit=_audit(),
    )
    assert envelope.revision == 1
    assert "payload" not in type(envelope).model_fields

    unknown = envelope.model_dump(mode="python")
    unknown["payload"] = {"unsafe": "body"}
    with pytest.raises(ValidationError):
        ArtifactEnvelopeV1.model_validate(unknown)

    with pytest.raises(ValidationError, match="successor requires"):
        ArtifactEnvelopeV1.create(
            artifact_id="artifact-001",
            subject_ref=envelope.subject_ref,
            schema_ref=envelope.schema_ref,
            content_ref=envelope.content_ref,
            media_type=envelope.media_type,
            modality=envelope.modality,
            domain_tags=envelope.domain_tags,
            semantic_role=envelope.semantic_role,
            purpose=envelope.purpose,
            classification=envelope.classification,
            lineage_refs=(envelope.to_ref(),),
            producer_capability_ref=capability_ref,
            producer_task_ref=None,
            validation_refs=(),
            revision=2,
            predecessor_envelope_ref=None,
            audit=_audit(),
        )


def test_artifact_envelope_and_head_successors_are_explicit() -> None:
    capability_ref = _registration().capability_definitions[0].to_ref()
    first = ArtifactEnvelopeV1.create(
        artifact_id="artifact-successor",
        subject_ref=_ref("evaluation-requirement-spec", version="v2"),
        schema_ref=_ref("json-schema", version="v2"),
        content_ref=_ref("content-object"),
        media_type="application/json",
        modality=ArtifactModalityV1.DOCUMENT,
        domain_tags=(),
        semantic_role="evaluation-requirement",
        purpose="evaluation-data-production",
        classification="INTERNAL",
        lineage_refs=(),
        producer_capability_ref=capability_ref,
        producer_task_ref=_ref("team-task"),
        validation_refs=(),
        revision=1,
        predecessor_envelope_ref=None,
        audit=_audit(),
    )
    successor = ArtifactEnvelopeV1.create(
        artifact_id=first.artifact_id,
        subject_ref=first.subject_ref,
        schema_ref=first.schema_ref,
        content_ref=_ref("content-object", "successor"),
        media_type=first.media_type,
        modality=first.modality,
        domain_tags=first.domain_tags,
        semantic_role=first.semantic_role,
        purpose=first.purpose,
        classification=first.classification,
        lineage_refs=(first.to_ref(),),
        producer_capability_ref=capability_ref,
        producer_task_ref=_ref("agent-task"),
        validation_refs=(),
        revision=2,
        predecessor_envelope_ref=first.to_ref(),
        audit=_audit(),
    )
    first_head = ArtifactHeadV1.create(
        head_id="artifact-head-requirement",
        team_ref=_ref("agent-team"),
        semantic_role="evaluation-requirement",
        revision=1,
        envelope_ref=first.to_ref(),
        predecessor_head_ref=None,
        audit=_audit(),
    )
    successor_head = ArtifactHeadV1.create(
        head_id=first_head.head_id,
        team_ref=first_head.team_ref,
        semantic_role=first_head.semantic_role,
        revision=2,
        envelope_ref=successor.to_ref(),
        predecessor_head_ref=first_head.to_ref(),
        audit=_audit(),
    )
    assert successor_head.predecessor_head_ref == first_head.to_ref()

    with pytest.raises(ValidationError, match="Agent or Team task"):
        ArtifactEnvelopeV1.create(
            artifact_id="artifact-invalid-producer",
            subject_ref=first.subject_ref,
            schema_ref=first.schema_ref,
            content_ref=first.content_ref,
            media_type=first.media_type,
            modality=first.modality,
            domain_tags=(),
            semantic_role=first.semantic_role,
            purpose=first.purpose,
            classification=first.classification,
            lineage_refs=(),
            producer_capability_ref=capability_ref,
            producer_task_ref=_ref("interaction-decision"),
            validation_refs=(),
            revision=1,
            predecessor_envelope_ref=None,
            audit=_audit(),
        )
    with pytest.raises(ValidationError, match="first artifact revision"):
        ArtifactEnvelopeV1.create(
            artifact_id="artifact-invalid-predecessor",
            subject_ref=first.subject_ref,
            schema_ref=first.schema_ref,
            content_ref=first.content_ref,
            media_type=first.media_type,
            modality=first.modality,
            domain_tags=(),
            semantic_role=first.semantic_role,
            purpose=first.purpose,
            classification=first.classification,
            lineage_refs=(),
            producer_capability_ref=capability_ref,
            producer_task_ref=None,
            validation_refs=(),
            revision=1,
            predecessor_envelope_ref=first.to_ref(),
            audit=_audit(),
        )
    with pytest.raises(ValidationError, match="head successor requires"):
        ArtifactHeadV1.create(
            head_id="artifact-head-invalid",
            team_ref=_ref("agent-team"),
            semantic_role="evaluation-requirement",
            revision=2,
            envelope_ref=successor.to_ref(),
            predecessor_head_ref=None,
            audit=_audit(),
        )


def test_capability_consumer_context_and_call_use_one_typed_seam() -> None:
    registration = _registration()
    definition = registration.capability_definitions[0]
    provider = next(
        value
        for value in registration.provider_bindings
        if value.capability_definition_ref == definition.to_ref()
    )
    projection = registration.projections[0]
    consumer = CapabilityConsumerBindingV1.create(
        consumer_id="consumer.requirement-agent-tool",
        consumer_version="1.0.0",
        consumer_kind=CapabilityConsumerKindV1.AGENT_TOOL,
        capability_definition_ref=definition.to_ref(),
        provider_binding_ref=provider.to_ref(),
        projection_ref=projection.to_ref(),
        audit=_audit(),
    )
    context = CapabilityInvocationContextV1(
        session_ref=_ref("harness-session"),
        team_ref=_ref("agent-team"),
        member_ref=_ref("team-member"),
        task_ref=_ref("team-task"),
        authority_ref=_ref("execution-authority"),
        principal_ref=_ref("principal"),
        data_purpose="evaluation-data-production",
        data_classification="INTERNAL",
        idempotency_key="capability-call-key-001",
    )
    call = CapabilityCallV1.create(
        call_id="capability-call-001",
        capability_definition_ref=definition.to_ref(),
        provider_binding_ref=provider.to_ref(),
        context=context,
        request_ref=_ref("dataset-build-plan", version="v2"),
        input_artifact_refs=(_ref("artifact-envelope"),),
        audit=_audit(),
    )
    assert consumer.provider_binding_ref == provider.to_ref()
    assert call.context.task_ref == _ref("team-task")

    with pytest.raises(ValidationError, match="requires a member ref"):
        CapabilityInvocationContextV1(
            session_ref=_ref("harness-session"),
            team_ref=_ref("agent-team"),
            member_ref=None,
            task_ref=None,
            authority_ref=_ref("execution-authority"),
            principal_ref=_ref("principal"),
            data_purpose="evaluation-data-production",
            data_classification="INTERNAL",
            idempotency_key="capability-call-key-002",
        )
    with pytest.raises(ValidationError, match="Team invocation"):
        CapabilityInvocationContextV1(
            session_ref=_ref("harness-session"),
            team_ref=None,
            member_ref=_ref("team-member"),
            task_ref=None,
            authority_ref=_ref("execution-authority"),
            principal_ref=_ref("principal"),
            data_purpose="evaluation-data-production",
            data_classification="INTERNAL",
            idempotency_key="capability-call-key-003",
        )
    with pytest.raises(ValidationError, match="artifact envelopes"):
        CapabilityCallV1.create(
            call_id="capability-call-invalid",
            capability_definition_ref=definition.to_ref(),
            provider_binding_ref=provider.to_ref(),
            context=context,
            request_ref=_ref("dataset-build-plan", version="v2"),
            input_artifact_refs=(_ref("raw-trace"),),
            audit=_audit(),
        )


def test_capability_result_never_publishes_partial_canonical_output() -> None:
    result = CapabilityResultV1.create(
        result_id="capability-result-001",
        call_ref=_ref("harness-capability-call"),
        outcome=CapabilityInvocationOutcomeV1.SUCCEEDED,
        canonical_result_ref=_ref("compiled-dataset-build-plan", version="v2"),
        output_artifact_refs=(_ref("artifact-envelope"),),
        validation_refs=(_ref("validator-result"),),
        failure_code=None,
        audit=_audit(),
    )
    assert result.outcome is CapabilityInvocationOutcomeV1.SUCCEEDED

    with pytest.raises(ValidationError, match="cannot publish canonical output"):
        CapabilityResultV1.create(
            result_id="capability-result-002",
            call_ref=_ref("harness-capability-call"),
            outcome=CapabilityInvocationOutcomeV1.BLOCKED_POLICY,
            canonical_result_ref=_ref("compiled-dataset-build-plan", version="v2"),
            output_artifact_refs=(),
            validation_refs=(_ref("validator-result"),),
            failure_code="POLICY_BLOCKED",
            audit=_audit(),
        )


@pytest.mark.parametrize(
    "model_type",
    (
        ArtifactEnvelopeV1,
        CapabilityResultV1,
        HarnessSessionRefV1,
        SessionCommandV1,
        SessionEventV1,
    ),
)
def test_harness_json_schemas_forbid_additional_properties(
    model_type: type[BaseModel],
) -> None:
    assert model_type.model_json_schema()["additionalProperties"] is False
