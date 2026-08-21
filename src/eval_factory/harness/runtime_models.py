from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationStatusV2,
    GatewayUsageV2,
)
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique,
    require_sorted_unique_refs,
    sorted_refs,
)
from eval_factory.harness.session_models import HarnessSessionRefV1, SessionEventV1


class HarnessSessionStatusV1(StrEnum):
    ACTIVE = "ACTIVE"
    BLOCKED = "BLOCKED"
    CLOSED = "CLOSED"


class HarnessMessageRoleV1(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class RequirementInterpretationOutcomeV1(StrEnum):
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    READY = "READY"
    ABSTAINED = "ABSTAINED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"


class GatewayJournalStateV1(StrEnum):
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"


class ProviderEvidenceClassV1(StrEnum):
    REAL_SEMANTIC = "REAL_SEMANTIC"
    MECHANISM_FIXTURE = "MECHANISM_FIXTURE"


class HarnessTurnOutcomeV1(StrEnum):
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    READY = "READY"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"


class HarnessSessionIdentityV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/session-identity/v1"] = "eval-harness/session-identity/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-session"

    session_id: Identifier
    incarnation_id: Identifier
    composition_ref: ObjectRef
    created_by: Identifier
    created_at: datetime

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        require_ref(
            self.composition_ref,
            "harness-composition",
            "composition_ref",
        )
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("session creation time must be timezone-aware")
        return self


class HarnessSessionStateV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-state/v1"] = "eval-harness/session-state/v1"
    session_ref: ObjectRef
    session_id: Identifier
    incarnation_id: Identifier
    composition_ref: ObjectRef
    session_version: int = Field(ge=1, le=1_000_000_000)
    status: HarnessSessionStatusV1
    last_event_sequence: int = Field(ge=1, le=1_000_000_000)
    current_interpretation_ref: ObjectRef | None = None
    current_requirement_policy_ref: ObjectRef | None = None
    created_at: datetime
    updated_at: datetime
    state_sha256: Sha256

    @classmethod
    def create(
        cls,
        *,
        identity: HarnessSessionIdentityV1,
        session_version: int,
        status: HarnessSessionStatusV1,
        last_event_sequence: int,
        current_interpretation_ref: ObjectRef | None,
        current_requirement_policy_ref: ObjectRef | None,
        updated_at: datetime,
    ) -> HarnessSessionStateV1:
        values = {
            "session_ref": identity.to_ref(),
            "session_id": identity.session_id,
            "incarnation_id": identity.incarnation_id,
            "composition_ref": identity.composition_ref,
            "session_version": session_version,
            "status": status,
            "last_event_sequence": last_event_sequence,
            "current_interpretation_ref": current_interpretation_ref,
            "current_requirement_policy_ref": current_requirement_policy_ref,
            "created_at": identity.created_at,
            "updated_at": updated_at,
        }
        return cls(
            session_ref=identity.to_ref(),
            session_id=identity.session_id,
            incarnation_id=identity.incarnation_id,
            composition_ref=identity.composition_ref,
            session_version=session_version,
            status=status,
            last_event_sequence=last_event_sequence,
            current_interpretation_ref=current_interpretation_ref,
            current_requirement_policy_ref=current_requirement_policy_ref,
            created_at=identity.created_at,
            updated_at=updated_at,
            state_sha256=_payload_sha256(values),
        )

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        require_ref(self.session_ref, "harness-session", "session_ref")
        require_ref(
            self.composition_ref,
            "harness-composition",
            "composition_ref",
        )
        if self.current_interpretation_ref is not None:
            require_ref(
                self.current_interpretation_ref,
                "requirement-interpretation",
                "current_interpretation_ref",
            )
        if self.current_requirement_policy_ref is not None:
            require_ref(
                self.current_requirement_policy_ref,
                "harness-requirement-policy",
                "current_requirement_policy_ref",
            )
        if self.updated_at < self.created_at:
            raise ValueError("session update cannot precede creation")
        values = self.model_dump(
            mode="python",
            exclude={"schema_version", "state_sha256"},
        )
        if self.state_sha256 != _payload_sha256(values):
            raise ValueError("session state hash is stale")
        return self

    def to_session_ref(self) -> HarnessSessionRefV1:
        return HarnessSessionRefV1(
            session_ref=self.session_ref,
            incarnation_id=self.incarnation_id,
            session_version=self.session_version,
            composition_ref=self.composition_ref,
        )


class HarnessMessageV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/message/v1"] = "eval-harness/message/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-message"

    message_id: Identifier
    session_ref: ObjectRef
    role: HarnessMessageRoleV1
    content: str = Field(min_length=1, max_length=1_000_000)
    content_sha256: Sha256
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    chunk_index: int | None = Field(default=None, ge=0, le=1_000_000)
    final: bool
    created_at: datetime

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        message_id: str,
        session_ref: ObjectRef,
        role: HarnessMessageRoleV1,
        content: str,
        artifact_envelope_refs: tuple[ObjectRef, ...],
        chunk_index: int | None,
        final: bool,
        created_at: datetime,
        audit: ContractAudit,
    ) -> HarnessMessageV1:
        normalized_content = content.strip()
        return super().create(
            audit=audit,
            message_id=message_id,
            session_ref=session_ref,
            role=role,
            content=normalized_content,
            content_sha256=hashlib.sha256(normalized_content.encode()).hexdigest(),
            artifact_envelope_refs=sorted_refs(artifact_envelope_refs),
            chunk_index=chunk_index,
            final=final,
            created_at=created_at,
        )

    @model_validator(mode="after")
    def validate_message(self) -> Self:
        require_ref(self.session_ref, "harness-session", "session_ref")
        require_sorted_unique_refs(
            self.artifact_envelope_refs,
            "artifact_envelope_refs",
        )
        if any(ref.object_type != "artifact-envelope" for ref in self.artifact_envelope_refs):
            raise ValueError("message artifacts must reference envelopes")
        if hashlib.sha256(self.content.encode()).hexdigest() != self.content_sha256:
            raise ValueError("message content hash is stale")
        if self.role is HarnessMessageRoleV1.USER and (self.chunk_index is not None or not self.final):
            raise ValueError("user messages cannot be assistant chunks")
        if self.final == (self.chunk_index is not None):
            raise ValueError("assistant chunk index must appear exactly for non-final messages")
        return self


class HarnessMessageCommandV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/message-command/v1"] = "eval-harness/message-command/v1"
    OBJECT_TYPE: ClassVar[str] = "session-command"

    command_id: Identifier
    session_ref: ObjectRef
    expected_session_version: int = Field(ge=1, le=1_000_000_000)
    principal_ref: ObjectRef
    content_sha256: Sha256
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    idempotency_key: Identifier

    @model_validator(mode="after")
    def validate_command(self) -> Self:
        require_ref(self.session_ref, "harness-session", "session_ref")
        require_sorted_unique_refs(
            self.artifact_envelope_refs,
            "artifact_envelope_refs",
        )
        return self


class RequirementIntakeV1(ContractModelV2):
    schema_version: Literal["eval-harness/private-requirement-intake/v1"] = (
        "eval-harness/private-requirement-intake/v1"
    )
    session_ref: ObjectRef
    command_ref: ObjectRef
    user_message_ref: ObjectRef
    user_text: str = Field(min_length=1, max_length=1_000_000)
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )


class RequirementInterpretationProposalV1(ContractModelV2):
    schema_version: Literal["eval-harness/private-requirement-proposal/v1"] = (
        "eval-harness/private-requirement-proposal/v1"
    )
    outcome: Literal["CLARIFICATION_REQUIRED", "READY", "ABSTAINED"]
    assistant_message: str = Field(min_length=1, max_length=32_768)
    goals: tuple[str, ...] = Field(default=(), max_length=128)
    constraints: tuple[str, ...] = Field(default=(), max_length=256)
    assumptions: tuple[str, ...] = Field(default=(), max_length=256)
    source_expectations: tuple[str, ...] = Field(default=(), max_length=256)
    target_capabilities: tuple[Identifier, ...] = Field(default=(), max_length=256)
    quality_intent: str | None = Field(default=None, max_length=4_000)
    delivery_intent: str | None = Field(default=None, max_length=4_000)
    max_model_requests: int | None = Field(default=None, ge=0, le=10_000_000)
    max_model_tokens: int | None = Field(default=None, ge=0, le=10_000_000_000)
    max_cost_micro_usd: int | None = Field(
        default=None,
        ge=0,
        le=10_000_000_000_000,
    )
    missing_field_codes: tuple[Identifier, ...] = Field(default=(), max_length=64)
    clarification_questions: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        for values, label in (
            (self.goals, "goals"),
            (self.constraints, "constraints"),
            (self.assumptions, "assumptions"),
            (self.source_expectations, "source_expectations"),
            (self.target_capabilities, "target_capabilities"),
            (self.missing_field_codes, "missing_field_codes"),
            (self.clarification_questions, "clarification_questions"),
        ):
            require_sorted_unique(values, label)
        if self.outcome == "READY":
            required = (
                self.goals,
                self.source_expectations,
                self.target_capabilities,
                self.quality_intent,
                self.delivery_intent,
                self.max_model_requests,
                self.max_model_tokens,
                self.max_cost_micro_usd,
            )
            if any(value is None or value == () for value in required):
                raise ValueError("ready requirement proposal is incomplete")
            if self.missing_field_codes or self.clarification_questions:
                raise ValueError("ready requirement cannot retain clarification")
        elif self.outcome == "CLARIFICATION_REQUIRED":
            if not self.missing_field_codes or not self.clarification_questions:
                raise ValueError("clarification requires missing fields and questions")
        elif self.missing_field_codes or self.clarification_questions:
            raise ValueError("abstained proposal cannot fabricate clarification")
        return self


class RequirementInterpretationV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/requirement-interpretation/v1"] = (
        "eval-harness/requirement-interpretation/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "requirement-interpretation"

    interpretation_id: Identifier
    session_ref: ObjectRef
    command_ref: ObjectRef
    user_message_ref: ObjectRef
    assistant_message_ref: ObjectRef
    proposal_ref: ObjectRef | None
    gateway_result_ref: ObjectRef | None
    outcome: RequirementInterpretationOutcomeV1
    evidence_class: ProviderEvidenceClassV1
    goals: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    source_expectations: tuple[str, ...] = ()
    target_capabilities: tuple[Identifier, ...] = ()
    quality_intent: str | None = None
    delivery_intent: str | None = None
    missing_field_codes: tuple[Identifier, ...] = ()
    clarification_questions: tuple[str, ...] = ()
    reason_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_interpretation(self) -> Self:
        require_ref(self.session_ref, "harness-session", "session_ref")
        require_ref(self.command_ref, "session-command", "command_ref")
        require_ref(self.user_message_ref, "harness-message", "user_message_ref")
        require_ref(
            self.assistant_message_ref,
            "harness-message",
            "assistant_message_ref",
        )
        for values, label in (
            (self.goals, "goals"),
            (self.constraints, "constraints"),
            (self.assumptions, "assumptions"),
            (self.source_expectations, "source_expectations"),
            (self.target_capabilities, "target_capabilities"),
            (self.missing_field_codes, "missing_field_codes"),
            (self.clarification_questions, "clarification_questions"),
            (self.reason_codes, "reason_codes"),
        ):
            require_sorted_unique(values, label)
        gateway_backed = self.outcome in {
            RequirementInterpretationOutcomeV1.READY,
            RequirementInterpretationOutcomeV1.CLARIFICATION_REQUIRED,
            RequirementInterpretationOutcomeV1.ABSTAINED,
        }
        if gateway_backed != (self.proposal_ref is not None and self.gateway_result_ref is not None):
            raise ValueError("semantic interpretation Gateway authority is incomplete")
        if self.outcome is RequirementInterpretationOutcomeV1.READY:
            if (
                not self.goals
                or not self.source_expectations
                or not self.target_capabilities
                or self.quality_intent is None
                or self.delivery_intent is None
                or self.missing_field_codes
                or self.clarification_questions
                or self.reason_codes
            ):
                raise ValueError("ready interpretation is incomplete")
        elif self.outcome is RequirementInterpretationOutcomeV1.CLARIFICATION_REQUIRED:
            if not self.missing_field_codes or not self.clarification_questions or self.reason_codes:
                raise ValueError("clarification interpretation is incomplete")
        elif not self.reason_codes:
            raise ValueError("non-ready interpretation requires reason codes")
        return self


class HarnessRequirementPolicyV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/requirement-policy/v1"] = "eval-harness/requirement-policy/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-requirement-policy"

    policy_id: Identifier
    interpretation_ref: ObjectRef
    data_classification: Identifier
    residency: Identifier
    max_model_requests: int = Field(ge=0, le=10_000_000)
    max_model_tokens: int = Field(ge=0, le=10_000_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)
    source_admission_requires_approval: Literal[True] = True
    external_execution_requires_approval: Literal[True] = True
    release_requires_approval: Literal[True] = True
    production_release_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        require_ref(
            self.interpretation_ref,
            "requirement-interpretation",
            "interpretation_ref",
        )
        return self


class HarnessGatewayJournalV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/gateway-journal/v1"] = "eval-harness/gateway-journal/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-gateway-journal"

    journal_id: Identifier
    session_ref: ObjectRef
    command_ref: ObjectRef
    state: GatewayJournalStateV1
    evidence_class: ProviderEvidenceClassV1
    route_ref: ObjectRef
    invocation_request_ref: ObjectRef
    prompt_template_ref: ObjectRef
    model_profile_ref: ObjectRef
    predecessor_journal_ref: ObjectRef | None = None
    receipt_ref: ObjectRef | None = None
    invocation_result_ref: ObjectRef | None = None
    output_ref: ObjectRef | None = None
    gateway_status: GatewayInvocationStatusV2 | None = None
    usage: GatewayUsageV2 | None = None
    failure_code: Identifier | None = None

    @model_validator(mode="after")
    def validate_journal(self) -> Self:
        require_ref(self.session_ref, "harness-session", "session_ref")
        require_ref(self.command_ref, "session-command", "command_ref")
        require_ref(self.route_ref, "model-route-decision", "route_ref", object_version="v2")
        require_ref(
            self.invocation_request_ref,
            "gateway-invocation-request",
            "invocation_request_ref",
            object_version="v2",
        )
        require_ref(
            self.prompt_template_ref,
            "prompt-template",
            "prompt_template_ref",
            object_version="v2",
        )
        require_ref(
            self.model_profile_ref,
            "model-capability-profile",
            "model_profile_ref",
            object_version="v2",
        )
        if self.predecessor_journal_ref is not None:
            require_ref(
                self.predecessor_journal_ref,
                "harness-gateway-journal",
                "predecessor_journal_ref",
            )
        closure = (
            self.receipt_ref,
            self.invocation_result_ref,
            self.gateway_status,
            self.usage,
        )
        if self.state is GatewayJournalStateV1.PREPARED:
            if any(value is not None for value in (*closure, self.output_ref, self.failure_code)):
                raise ValueError("prepared Gateway journal cannot contain result authority")
        elif self.state is GatewayJournalStateV1.COMMITTED:
            if any(value is None for value in closure):
                raise ValueError("committed Gateway journal requires receipt/result/usage")
            if self.predecessor_journal_ref is None:
                raise ValueError("committed Gateway journal requires prepared predecessor")
            if self.gateway_status is GatewayInvocationStatusV2.SUCCEEDED:
                if self.output_ref is None or self.failure_code is not None:
                    raise ValueError("successful Gateway journal requires output only")
            elif self.output_ref is not None or self.failure_code is None:
                raise ValueError("failed Gateway journal requires failure code only")
        else:
            if self.predecessor_journal_ref is None or self.failure_code is None:
                raise ValueError("unknown Gateway outcome requires predecessor and reason")
            if any(value is not None for value in closure):
                raise ValueError("unknown Gateway outcome cannot claim committed result")
        return self


class HarnessTurnResultV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/turn-result/v1"] = "eval-harness/turn-result/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-turn-result"

    turn_id: Identifier
    session_ref: ObjectRef
    command_ref: ObjectRef
    user_message_ref: ObjectRef
    assistant_message_ref: ObjectRef | None
    interpretation_ref: ObjectRef | None
    requirement_policy_ref: ObjectRef | None
    gateway_journal_ref: ObjectRef | None
    event_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=10_000)
    outcome: HarnessTurnOutcomeV1
    reason_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_turn(self) -> Self:
        require_ref(self.session_ref, "harness-session", "session_ref")
        require_ref(self.command_ref, "session-command", "command_ref")
        require_ref(self.user_message_ref, "harness-message", "user_message_ref")
        require_sorted_unique_refs(self.event_refs, "event_refs")
        if any(ref.object_type != "session-event" for ref in self.event_refs):
            raise ValueError("turn event refs contain an invalid type")
        if self.outcome in {
            HarnessTurnOutcomeV1.READY,
            HarnessTurnOutcomeV1.CLARIFICATION_REQUIRED,
        }:
            if (
                self.assistant_message_ref is None
                or self.interpretation_ref is None
                or self.gateway_journal_ref is None
                or self.reason_codes
            ):
                raise ValueError("successful semantic turn authority is incomplete")
            if self.outcome is HarnessTurnOutcomeV1.READY and self.requirement_policy_ref is None:
                raise ValueError("ready turn requires requirement policy")
        elif not self.reason_codes:
            raise ValueError("blocked turn requires reason codes")
        return self


class HarnessTranscriptMessageV1(ContractModelV2):
    schema_version: Literal["eval-harness/transcript-message/v1"] = "eval-harness/transcript-message/v1"
    message_ref: ObjectRef
    role: HarnessMessageRoleV1
    content: str
    artifact_envelope_refs: tuple[ObjectRef, ...]
    created_at: datetime


class HarnessGatewaySummaryV1(ContractModelV2):
    schema_version: Literal["eval-harness/gateway-summary/v1"] = "eval-harness/gateway-summary/v1"
    journal_ref: ObjectRef
    state: GatewayJournalStateV1
    evidence_class: ProviderEvidenceClassV1
    route_ref: ObjectRef
    invocation_result_ref: ObjectRef | None
    gateway_status: GatewayInvocationStatusV2 | None
    model_profile_ref: ObjectRef
    prompt_template_ref: ObjectRef
    usage: GatewayUsageV2 | None
    failure_code: Identifier | None


class HarnessSessionProjectionV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-projection/v1"] = "eval-harness/session-projection/v1"
    session: HarnessSessionStateV1
    transcript: tuple[HarnessTranscriptMessageV1, ...]
    event_refs: tuple[ObjectRef, ...]
    current_interpretation_ref: ObjectRef | None
    current_requirement_policy_ref: ObjectRef | None
    latest_gateway: HarnessGatewaySummaryV1 | None
    pending_clarification_questions: tuple[str, ...]
    source_fingerprint: Sha256


class HarnessEventPageV1(ContractModelV2):
    schema_version: Literal["eval-harness/event-page/v1"] = "eval-harness/event-page/v1"
    session_ref: ObjectRef
    after_sequence: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    events: tuple[SessionEventV1, ...]
    next_sequence: int | None = Field(default=None, ge=1)


class HarnessSessionPageV1(ContractModelV2):
    schema_version: Literal["eval-harness/session-page/v1"] = "eval-harness/session-page/v1"
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    total: int = Field(ge=0)
    sessions: tuple[HarnessSessionStateV1, ...]


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "GatewayJournalStateV1",
    "HarnessEventPageV1",
    "HarnessGatewayJournalV1",
    "HarnessGatewaySummaryV1",
    "HarnessMessageCommandV1",
    "HarnessMessageRoleV1",
    "HarnessMessageV1",
    "HarnessRequirementPolicyV1",
    "HarnessSessionIdentityV1",
    "HarnessSessionPageV1",
    "HarnessSessionProjectionV1",
    "HarnessSessionStateV1",
    "HarnessSessionStatusV1",
    "HarnessTranscriptMessageV1",
    "HarnessTurnOutcomeV1",
    "HarnessTurnResultV1",
    "ProviderEvidenceClassV1",
    "RequirementIntakeV1",
    "RequirementInterpretationOutcomeV1",
    "RequirementInterpretationProposalV1",
    "RequirementInterpretationV1",
]
