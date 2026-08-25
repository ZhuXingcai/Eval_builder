from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from env_mock_agent.facade import RuntimeEventKindV2
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique_refs,
    sorted_refs,
)


class SessionLifecycleEventKindV1(StrEnum):
    SESSION_OPENED = "SESSION_OPENED"
    SESSION_CLOSED = "SESSION_CLOSED"
    TURN_STARTED = "TURN_STARTED"
    TURN_COMPLETED = "TURN_COMPLETED"
    STEP_STARTED = "STEP_STARTED"
    STEP_COMPLETED = "STEP_COMPLETED"


class SessionMessageEventKindV1(StrEnum):
    USER_MESSAGE = "USER_MESSAGE"
    ASSISTANT_MESSAGE = "ASSISTANT_MESSAGE"
    ASSISTANT_CHUNK = "ASSISTANT_CHUNK"


class SessionCapabilityEventKindV1(StrEnum):
    CAPABILITY_CALLED = "CAPABILITY_CALLED"
    CAPABILITY_RETURNED = "CAPABILITY_RETURNED"


class SessionInteractionEventKindV1(StrEnum):
    INTERACTION_REQUESTED = "INTERACTION_REQUESTED"
    INTERACTION_DECIDED = "INTERACTION_DECIDED"


class SessionCheckpointEventKindV1(StrEnum):
    GRAPH_CHECKPOINTED = "GRAPH_CHECKPOINTED"
    TEAM_CHECKPOINTED = "TEAM_CHECKPOINTED"


class SessionTeamEventKindV1(StrEnum):
    TEAM_CHANGED = "TEAM_CHANGED"
    TEAM_MESSAGE_POSTED = "TEAM_MESSAGE_POSTED"
    ARTIFACT_HEAD_CHANGED = "ARTIFACT_HEAD_CHANGED"
    CONFLICT_OPENED = "CONFLICT_OPENED"
    CONFLICT_RESOLVED = "CONFLICT_RESOLVED"


class SessionGatewayEventKindV1(StrEnum):
    GATEWAY_PREPARED = "GATEWAY_PREPARED"
    GATEWAY_COMPLETED = "GATEWAY_COMPLETED"
    GATEWAY_FAILED = "GATEWAY_FAILED"
    GATEWAY_UNKNOWN = "GATEWAY_UNKNOWN"


class SessionRequirementEventKindV1(StrEnum):
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    REQUIREMENT_READY = "REQUIREMENT_READY"
    REQUIREMENT_BLOCKED = "REQUIREMENT_BLOCKED"


class SessionCommandKindV1(StrEnum):
    USER_MESSAGE = "USER_MESSAGE"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    INTERRUPT = "INTERRUPT"
    RESUME = "RESUME"
    CANCEL = "CANCEL"


class HarnessSessionRefV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-ref/v1"] = "eval-harness/session-ref/v1"
    session_ref: ObjectRef
    incarnation_id: Identifier
    session_version: int = Field(ge=0, le=1_000_000_000)
    composition_ref: ObjectRef

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        require_ref(self.session_ref, "harness-session", "session_ref")
        require_ref(
            self.composition_ref,
            "harness-composition",
            "composition_ref",
        )
        return self


class SessionLifecyclePayloadV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-lifecycle-payload/v1"] = (
        "eval-harness/session-lifecycle-payload/v1"
    )
    family: Literal["LIFECYCLE"] = "LIFECYCLE"
    event_kind: SessionLifecycleEventKindV1


class SessionMessagePayloadV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-message-payload/v1"] = (
        "eval-harness/session-message-payload/v1"
    )
    family: Literal["MESSAGE"] = "MESSAGE"
    event_kind: SessionMessageEventKindV1
    message_ref: ObjectRef
    model_visible_artifact_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )

    @model_validator(mode="after")
    def validate_message(self) -> Self:
        require_ref(self.message_ref, "harness-message", "message_ref")
        require_sorted_unique_refs(
            self.model_visible_artifact_refs,
            "model_visible_artifact_refs",
        )
        if any(ref.object_type != "artifact-envelope" for ref in self.model_visible_artifact_refs):
            raise ValueError("model-visible artifacts must use artifact envelopes")
        return self


class SessionCapabilityPayloadV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-capability-payload/v1"] = (
        "eval-harness/session-capability-payload/v1"
    )
    family: Literal["CAPABILITY"] = "CAPABILITY"
    event_kind: SessionCapabilityEventKindV1
    record_ref: ObjectRef

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        expected_type = (
            "harness-capability-call"
            if self.event_kind is SessionCapabilityEventKindV1.CAPABILITY_CALLED
            else "harness-capability-result"
        )
        require_ref(self.record_ref, expected_type, "record_ref")
        return self


class SessionInteractionPayloadV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-interaction-payload/v1"] = (
        "eval-harness/session-interaction-payload/v1"
    )
    family: Literal["INTERACTION"] = "INTERACTION"
    event_kind: SessionInteractionEventKindV1
    record_ref: ObjectRef

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        expected_type = (
            "interaction-request"
            if self.event_kind is SessionInteractionEventKindV1.INTERACTION_REQUESTED
            else "interaction-decision"
        )
        require_ref(self.record_ref, expected_type, "record_ref")
        return self


class SessionCheckpointPayloadV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-checkpoint-payload/v1"] = (
        "eval-harness/session-checkpoint-payload/v1"
    )
    family: Literal["CHECKPOINT"] = "CHECKPOINT"
    event_kind: SessionCheckpointEventKindV1
    checkpoint_ref: ObjectRef

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        expected_type = (
            "graph-checkpoint"
            if self.event_kind is SessionCheckpointEventKindV1.GRAPH_CHECKPOINTED
            else "team-checkpoint"
        )
        require_ref(
            self.checkpoint_ref,
            expected_type,
            "checkpoint_ref",
            object_version=self.checkpoint_ref.object_version,
        )
        return self


class SessionTeamPayloadV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-team-payload/v1"] = "eval-harness/session-team-payload/v1"
    family: Literal["TEAM"] = "TEAM"
    event_kind: SessionTeamEventKindV1
    record_ref: ObjectRef

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        allowed = {
            SessionTeamEventKindV1.TEAM_CHANGED: {"agent-team", "team-roster", "team-task-graph"},
            SessionTeamEventKindV1.TEAM_MESSAGE_POSTED: {"team-message"},
            SessionTeamEventKindV1.ARTIFACT_HEAD_CHANGED: {"artifact-head"},
            SessionTeamEventKindV1.CONFLICT_OPENED: {"team-conflict"},
            SessionTeamEventKindV1.CONFLICT_RESOLVED: {"team-conflict"},
        }
        if self.record_ref.object_type not in allowed[self.event_kind]:
            raise ValueError("Team event record type does not match event kind")
        return self


class SessionGatewayPayloadV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-gateway-payload/v1"] = (
        "eval-harness/session-gateway-payload/v1"
    )
    family: Literal["GATEWAY"] = "GATEWAY"
    event_kind: SessionGatewayEventKindV1
    journal_ref: ObjectRef

    @model_validator(mode="after")
    def validate_journal(self) -> Self:
        require_ref(
            self.journal_ref,
            "harness-gateway-journal",
            "journal_ref",
        )
        return self


class SessionRequirementPayloadV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-requirement-payload/v1"] = (
        "eval-harness/session-requirement-payload/v1"
    )
    family: Literal["REQUIREMENT"] = "REQUIREMENT"
    event_kind: SessionRequirementEventKindV1
    interpretation_ref: ObjectRef

    @model_validator(mode="after")
    def validate_interpretation(self) -> Self:
        require_ref(
            self.interpretation_ref,
            "requirement-interpretation",
            "interpretation_ref",
        )
        return self


class SessionRuntimePayloadV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-runtime-payload/v1"] = (
        "eval-harness/session-runtime-payload/v1"
    )
    family: Literal["RUNTIME"] = "RUNTIME"
    event_kind: RuntimeEventKindV2
    runtime_id: Identifier
    runtime_event_ref: ObjectRef

    @model_validator(mode="after")
    def validate_runtime_event(self) -> Self:
        require_ref(
            self.runtime_event_ref,
            "runtime-event",
            "runtime_event_ref",
            object_version="v2",
        )
        return self


SessionEventPayloadV1 = Annotated[
    SessionLifecyclePayloadV1
    | SessionMessagePayloadV1
    | SessionCapabilityPayloadV1
    | SessionInteractionPayloadV1
    | SessionCheckpointPayloadV1
    | SessionTeamPayloadV1
    | SessionGatewayPayloadV1
    | SessionRequirementPayloadV1
    | SessionRuntimePayloadV1,
    Field(discriminator="family"),
]


class SessionCommandV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/session-command/v1"] = "eval-harness/session-command/v1"
    OBJECT_TYPE: ClassVar[str] = "session-command"

    command_id: Identifier
    session: HarnessSessionRefV1
    command_kind: SessionCommandKindV1
    expected_event_sequence: int = Field(ge=0, le=1_000_000_000)
    authority_ref: ObjectRef
    principal_ref: ObjectRef
    subject_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    idempotency_key: Identifier

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        command_id: str,
        session: HarnessSessionRefV1,
        command_kind: SessionCommandKindV1,
        expected_event_sequence: int,
        authority_ref: ObjectRef,
        principal_ref: ObjectRef,
        subject_refs: tuple[ObjectRef, ...],
        idempotency_key: str,
        audit: ContractAudit,
    ) -> SessionCommandV1:
        return super().create(
            audit=audit,
            command_id=command_id,
            session=session,
            command_kind=command_kind,
            expected_event_sequence=expected_event_sequence,
            authority_ref=authority_ref,
            principal_ref=principal_ref,
            subject_refs=sorted_refs(subject_refs),
            idempotency_key=idempotency_key,
        )

    @model_validator(mode="after")
    def validate_command(self) -> Self:
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        require_sorted_unique_refs(self.subject_refs, "subject_refs")
        return self


class SessionEventV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/session-event/v1"] = "eval-harness/session-event/v1"
    OBJECT_TYPE: ClassVar[str] = "session-event"

    event_id: Identifier
    session: HarnessSessionRefV1
    sequence: int = Field(ge=1, le=1_000_000_000)
    authority_version: int = Field(ge=0, le=1_000_000_000)
    turn_id: Identifier | None = None
    step_id: Identifier | None = None
    command_ref: ObjectRef | None = None
    payload: SessionEventPayloadV1
    occurred_at: datetime

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.step_id is not None and self.turn_id is None:
            raise ValueError("step event requires a turn ID")
        if self.command_ref is not None:
            require_ref(self.command_ref, "session-command", "command_ref")
        return self


def validate_session_event_log(events: tuple[SessionEventV1, ...]) -> None:
    if not events:
        return
    first_session = events[0].session
    session_identity = (
        first_session.session_ref,
        first_session.incarnation_id,
        first_session.composition_ref,
    )
    expected_sequence = 1
    seen_ids: set[str] = set()
    prior_authority_version = 0
    prior_session_version = 0
    for event in events:
        observed_identity = (
            event.session.session_ref,
            event.session.incarnation_id,
            event.session.composition_ref,
        )
        if observed_identity != session_identity:
            raise ValueError("session event log cannot mix session incarnations")
        if event.sequence != expected_sequence:
            raise ValueError("session event sequence must be contiguous from one")
        if event.event_id in seen_ids:
            raise ValueError("session event IDs must be unique")
        if event.authority_version < prior_authority_version:
            raise ValueError("session authority version cannot move backward")
        if event.session.session_version < prior_session_version:
            raise ValueError("session version cannot move backward")
        seen_ids.add(event.event_id)
        prior_authority_version = event.authority_version
        prior_session_version = event.session.session_version
        expected_sequence += 1


def validate_session_commands(commands: tuple[SessionCommandV1, ...]) -> None:
    authorities: dict[str, ObjectRef] = {}
    command_refs: dict[str, ObjectRef] = {}
    for command in commands:
        prior_ref = command_refs.get(command.idempotency_key)
        if prior_ref is not None and prior_ref != command.to_ref():
            raise ValueError("idempotency key cannot identify different commands")
        prior_authority = authorities.get(command.idempotency_key)
        if prior_authority is not None and prior_authority != command.authority_ref:
            raise ValueError("idempotent command cannot change authority version")
        command_refs[command.idempotency_key] = command.to_ref()
        authorities[command.idempotency_key] = command.authority_ref


__all__ = [
    "HarnessSessionRefV1",
    "SessionCapabilityEventKindV1",
    "SessionCapabilityPayloadV1",
    "SessionCheckpointEventKindV1",
    "SessionCheckpointPayloadV1",
    "SessionCommandKindV1",
    "SessionCommandV1",
    "SessionEventPayloadV1",
    "SessionEventV1",
    "SessionGatewayEventKindV1",
    "SessionGatewayPayloadV1",
    "SessionInteractionEventKindV1",
    "SessionInteractionPayloadV1",
    "SessionLifecycleEventKindV1",
    "SessionLifecyclePayloadV1",
    "SessionMessageEventKindV1",
    "SessionMessagePayloadV1",
    "SessionRequirementEventKindV1",
    "SessionRequirementPayloadV1",
    "SessionRuntimePayloadV1",
    "SessionTeamEventKindV1",
    "SessionTeamPayloadV1",
    "validate_session_commands",
    "validate_session_event_log",
]
