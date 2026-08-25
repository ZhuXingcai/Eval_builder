from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from env_mock_agent.facade import (
    RuntimeEventKindV2,
    RuntimeFailureCodeV2,
    ToolFamilyV2,
    UnifiedRuntimeUsageV2,
)
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunStatusV2,
    PlanKindV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.core import Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetNextActionV2,
)
from eval_factory.harness.graph_models import (
    GraphCheckpointOutcomeV1,
    GraphCheckpointPhaseV1,
)
from eval_factory.harness.runtime_models import (
    HarnessMessageRoleV1,
    HarnessSessionStatusV1,
    ProviderEvidenceClassV1,
    RequirementInterpretationOutcomeV1,
)
from eval_factory.harness.session_models import (
    SessionCapabilityEventKindV1,
    SessionCheckpointEventKindV1,
    SessionGatewayEventKindV1,
    SessionInteractionEventKindV1,
    SessionLifecycleEventKindV1,
    SessionMessageEventKindV1,
    SessionRequirementEventKindV1,
    SessionTeamEventKindV1,
)
from eval_factory.harness.source_admission import (
    SOURCE_MANIFEST_COLUMNS,
)
from eval_factory.team.models import (
    TeamLifecycleV1,
    TeamMemberStatusV1,
    TeamMessageAudienceV1,
    TeamMessageKindV1,
    TeamTaskStatusV1,
)


class AgentShellEventFamilyV1(StrEnum):
    LIFECYCLE = "LIFECYCLE"
    MESSAGE = "MESSAGE"
    CAPABILITY = "CAPABILITY"
    INTERACTION = "INTERACTION"
    CHECKPOINT = "CHECKPOINT"
    TEAM = "TEAM"
    GATEWAY = "GATEWAY"
    REQUIREMENT = "REQUIREMENT"
    RUNTIME = "RUNTIME"


AgentShellEventKindV1 = (
    SessionLifecycleEventKindV1
    | SessionMessageEventKindV1
    | SessionCapabilityEventKindV1
    | SessionInteractionEventKindV1
    | SessionCheckpointEventKindV1
    | SessionTeamEventKindV1
    | SessionGatewayEventKindV1
    | SessionRequirementEventKindV1
    | RuntimeEventKindV2
)

_EVENT_KIND_TYPES: dict[AgentShellEventFamilyV1, type[StrEnum]] = {
    AgentShellEventFamilyV1.LIFECYCLE: SessionLifecycleEventKindV1,
    AgentShellEventFamilyV1.MESSAGE: SessionMessageEventKindV1,
    AgentShellEventFamilyV1.CAPABILITY: SessionCapabilityEventKindV1,
    AgentShellEventFamilyV1.INTERACTION: SessionInteractionEventKindV1,
    AgentShellEventFamilyV1.CHECKPOINT: SessionCheckpointEventKindV1,
    AgentShellEventFamilyV1.TEAM: SessionTeamEventKindV1,
    AgentShellEventFamilyV1.GATEWAY: SessionGatewayEventKindV1,
    AgentShellEventFamilyV1.REQUIREMENT: SessionRequirementEventKindV1,
    AgentShellEventFamilyV1.RUNTIME: RuntimeEventKindV2,
}


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


SourceUploadNameV1 = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$",
        max_length=255,
    ),
]


class AgentShellSourceFileKindV1(StrEnum):
    MANIFEST = "MANIFEST"
    TRACE = "TRACE"


class AgentShellSourceAdmissionErrorCodeV1(StrEnum):
    CONFLICT = "SOURCE_ADMISSION_CONFLICT"
    HASH_MISMATCH = "SOURCE_ADMISSION_HASH_MISMATCH"
    INVALID = "SOURCE_ADMISSION_INVALID"
    INTEGRITY = "SOURCE_ADMISSION_INTEGRITY"
    INVENTORY_MISMATCH = "SOURCE_ADMISSION_INVENTORY_MISMATCH"
    LIMIT_EXCEEDED = "SOURCE_ADMISSION_LIMIT_EXCEEDED"
    MANIFEST_INVALID = "SOURCE_ADMISSION_MANIFEST_INVALID"
    MULTIPART_INVALID = "SOURCE_ADMISSION_MULTIPART_INVALID"
    NOT_CONFIGURED = "SOURCE_ADMISSION_NOT_CONFIGURED"
    NOT_FOUND = "SOURCE_ADMISSION_NOT_FOUND"
    SOURCE_INVALID = "SOURCE_ADMISSION_SOURCE_INVALID"
    SOURCE_METADATA_MISMATCH = "SOURCE_ADMISSION_SOURCE_METADATA_MISMATCH"
    UNSAFE_NAME = "SOURCE_ADMISSION_UNSAFE_NAME"


class AgentShellExecutionPhaseV1(StrEnum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    WAITING_REQUIREMENT = "WAITING_REQUIREMENT"
    WAITING_SOURCE = "WAITING_SOURCE"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING_REVIEW = "WAITING_REVIEW"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"


class AgentShellInteractionKindV1(StrEnum):
    CLARIFICATION = "CLARIFICATION"
    PERMISSION = "PERMISSION"
    PLAN_REVIEW = "PLAN_REVIEW"
    VERIFICATION = "VERIFICATION"


class AgentShellInteractionStateV1(StrEnum):
    PENDING = "PENDING"
    BLOCKED = "BLOCKED"


class AgentShellWorkspaceKindV1(StrEnum):
    CONVERSATION = "CONVERSATION"
    PLAN_REVIEW = "PLAN_REVIEW"
    TRACE = "TRACE"
    TASK = "TASK"
    ATTACHMENT = "ATTACHMENT"
    RUBRIC = "RUBRIC"
    GRADING = "GRADING"
    QUALITY = "QUALITY"
    TEAM = "TEAM"
    ACTIVITY = "ACTIVITY"
    DELIVERY = "DELIVERY"


class AgentShellWorkspaceStatusV1(StrEnum):
    AVAILABLE = "AVAILABLE"
    EMPTY = "EMPTY"
    BLOCKED = "BLOCKED"
    COMPLETE = "COMPLETE"


class AgentShellSessionSummaryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-session-summary/v1"] = (
        "eval-factory/agent-shell-session-summary/v1"
    )
    session_id: Identifier
    title: str | None = Field(default=None, max_length=160)
    status: HarnessSessionStatusV1
    session_version: int = Field(ge=1, le=1_000_000_000)
    last_event_sequence: int = Field(ge=1, le=1_000_000_000)
    created_at: datetime
    updated_at: datetime


class AgentShellTranscriptEntryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-transcript-entry/v1"] = (
        "eval-factory/agent-shell-transcript-entry/v1"
    )
    message_id: Identifier
    role: HarnessMessageRoleV1
    content: str = Field(min_length=1, max_length=32_768)
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=256,
    )
    created_at: datetime


class AgentShellEventV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-event/v1"] = "eval-factory/agent-shell-event/v1"
    event_id: Identifier
    sequence: int = Field(ge=1, le=1_000_000_000)
    authority_version: int = Field(ge=0, le=1_000_000_000)
    family: AgentShellEventFamilyV1
    event_kind: AgentShellEventKindV1
    occurred_at: datetime
    turn_id: Identifier | None = None
    step_id: Identifier | None = None
    runtime_id: Identifier | None = None
    tool_family: ToolFamilyV2 | None = None
    usage: UnifiedRuntimeUsageV2 | None = None
    failure_code: RuntimeFailureCodeV2 | None = None

    @model_validator(mode="after")
    def validate_runtime_fields(self) -> AgentShellEventV1:
        if not isinstance(self.event_kind, _EVENT_KIND_TYPES[self.family]):
            raise ValueError("event kind does not match event family")
        runtime = self.family is AgentShellEventFamilyV1.RUNTIME
        has_runtime_fields = any(
            value is not None
            for value in (
                self.runtime_id,
                self.tool_family,
                self.usage,
                self.failure_code,
            )
        )
        if runtime and self.runtime_id is None:
            raise ValueError("runtime event requires runtime_id")
        if not runtime and has_runtime_fields:
            raise ValueError("runtime fields are allowed only for runtime events")
        return self


class AgentShellEventPageV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-event-page/v1"] = (
        "eval-factory/agent-shell-event-page/v1"
    )
    session_id: Identifier
    after_sequence: int = Field(ge=0, le=1_000_000_000)
    watermark: int = Field(ge=1, le=1_000_000_000)
    events: tuple[AgentShellEventV1, ...] = Field(default=(), max_length=500)
    next_sequence: int | None = Field(
        default=None,
        ge=1,
        le=1_000_000_001,
    )


class AgentShellFactorySummaryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-factory-summary/v1"] = (
        "eval-factory/agent-shell-factory-summary/v1"
    )
    run_ref: ObjectRef
    run_version: int = Field(ge=0, le=100_000_000)
    status: FactoryRunStatusV2
    next_action: FactoryDatasetNextActionV2
    pending_review_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=500,
    )
    candidate_count: int = Field(ge=0, le=1_000_000)
    rejected_count: int = Field(ge=0, le=1_000_000)
    blocked_count: int = Field(ge=0, le=1_000_000)
    incomplete_count: int = Field(ge=0, le=1_000_000)
    aggregate_result_ref: ObjectRef | None = None
    delivery_manifest_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_factory(self) -> AgentShellFactorySummaryV1:
        if self.run_ref.object_type != "factory-run" or self.run_ref.object_version != "v2":
            raise ValueError("factory summary run_ref is invalid")
        refs = tuple(sorted(self.pending_review_refs, key=_ref_key))
        if refs != self.pending_review_refs or len(refs) != len(set(refs)):
            raise ValueError("factory pending review refs must be sorted and unique")
        if any(value.object_type != "plan-review-request" or value.object_version != "v2" for value in refs):
            raise ValueError("factory pending review refs are invalid")
        if self.aggregate_result_ref is not None and (
            self.aggregate_result_ref.object_type != "factory-dataset-aggregate-result"
            or self.aggregate_result_ref.object_version != "v2"
        ):
            raise ValueError("factory aggregate result ref is invalid")
        if self.delivery_manifest_ref is not None and (
            self.delivery_manifest_ref.object_type != "candidate-dataset-delivery-manifest"
            or self.delivery_manifest_ref.object_version != "v2"
        ):
            raise ValueError("factory delivery manifest ref is invalid")
        return self


class AgentShellGraphSummaryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-graph-summary/v1"] = (
        "eval-factory/agent-shell-graph-summary/v1"
    )
    binding_ref: ObjectRef
    checkpoint_ref: ObjectRef | None = None
    checkpoint_phase: GraphCheckpointPhaseV1 | None = None
    checkpoint_outcome: GraphCheckpointOutcomeV1 | None = None
    node: Identifier | None = None
    transition_number: int = Field(default=0, ge=0, le=100_000)
    reason_codes: tuple[Identifier, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_graph(self) -> AgentShellGraphSummaryV1:
        if (
            self.binding_ref.object_type != "graph-execution-binding"
            or self.binding_ref.object_version != "v1"
        ):
            raise ValueError("graph binding ref is invalid")
        checkpoint_values = (
            self.checkpoint_ref,
            self.checkpoint_phase,
            self.checkpoint_outcome,
            self.node,
        )
        if any(value is None for value in checkpoint_values) != all(
            value is None for value in checkpoint_values
        ):
            raise ValueError("graph checkpoint summary is incomplete")
        if self.checkpoint_ref is not None and (
            self.checkpoint_ref.object_type != "graph-checkpoint"
            or self.checkpoint_ref.object_version != "v1"
        ):
            raise ValueError("graph checkpoint ref is invalid")
        if self.checkpoint_ref is None and (self.transition_number != 0 or self.reason_codes):
            raise ValueError(
                "Graph summary without a checkpoint cannot claim transition state",
            )
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("graph reason codes must be sorted and unique")
        return self


class AgentShellTeamMemberSummaryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-team-member-summary/v1"] = (
        "eval-factory/agent-shell-team-member-summary/v1"
    )
    member_ref: ObjectRef
    member_id: Identifier
    role: Identifier
    status: TeamMemberStatusV1
    independent_session_ref: ObjectRef
    capability_definition_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    is_coordinator: bool

    @model_validator(mode="after")
    def validate_member(self) -> AgentShellTeamMemberSummaryV1:
        if self.member_ref.object_type != "team-member":
            raise ValueError("Team member ref is invalid")
        if self.independent_session_ref.object_type != "harness-session":
            raise ValueError("Team member session ref is invalid")
        refs = tuple(sorted(self.capability_definition_refs, key=_ref_key))
        if (
            refs != self.capability_definition_refs
            or len(refs) != len(set(refs))
            or any(value.object_type != "harness-capability-definition" for value in refs)
        ):
            raise ValueError("Team member capability refs are invalid")
        return self


class AgentShellTeamTaskSummaryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-team-task-summary/v1"] = (
        "eval-factory/agent-shell-team-task-summary/v1"
    )
    task_ref: ObjectRef
    task_id: Identifier
    task_kind: Identifier
    status: TeamTaskStatusV1
    assigned_member_id: Identifier | None = None
    dependency_task_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    capability_definition_ref: ObjectRef
    attempt: int = Field(ge=0, le=100)
    event_version: int = Field(ge=0, le=1_000_000_000)
    claim_ref: ObjectRef | None = None
    lease_ref: ObjectRef | None = None
    latest_event_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_task(self) -> AgentShellTeamTaskSummaryV1:
        if self.task_ref.object_type != "team-task":
            raise ValueError("Team task ref is invalid")
        if self.capability_definition_ref.object_type != "harness-capability-definition":
            raise ValueError("Team task capability ref is invalid")
        if tuple(sorted(set(self.dependency_task_ids))) != self.dependency_task_ids:
            raise ValueError("Team task dependencies must be sorted and unique")
        for value, object_type in (
            (self.claim_ref, "team-task-claim"),
            (self.lease_ref, "team-task-lease"),
            (self.latest_event_ref, "team-task-event"),
        ):
            if value is not None and value.object_type != object_type:
                raise ValueError("Team task runtime ref is invalid")
        return self


class AgentShellTeamSummaryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-team-summary/v1"] = (
        "eval-factory/agent-shell-team-summary/v1"
    )
    team_ref: ObjectRef
    team_version: int = Field(ge=1, le=1_000_000_000)
    lifecycle: TeamLifecycleV1
    roster_ref: ObjectRef
    task_graph_ref: ObjectRef
    authority_ref: ObjectRef
    checkpoint_ref: ObjectRef | None = None
    members: tuple[AgentShellTeamMemberSummaryV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    tasks: tuple[AgentShellTeamTaskSummaryV1, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    artifact_head_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=100_000,
    )
    open_conflict_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )

    @model_validator(mode="after")
    def validate_team(self) -> AgentShellTeamSummaryV1:
        expectations = (
            (self.team_ref, "agent-team"),
            (self.roster_ref, "team-roster"),
            (self.task_graph_ref, "team-task-graph"),
            (self.authority_ref, "execution-authority"),
        )
        if any(value.object_type != object_type for value, object_type in expectations):
            raise ValueError("Team authority refs are invalid")
        if self.checkpoint_ref is not None and self.checkpoint_ref.object_type != "team-checkpoint":
            raise ValueError("Team checkpoint ref is invalid")
        member_ids = tuple(value.member_id for value in self.members)
        task_ids = tuple(value.task_id for value in self.tasks)
        if member_ids != tuple(sorted(set(member_ids))):
            raise ValueError("Team members must be sorted and unique")
        if task_ids != tuple(sorted(set(task_ids))):
            raise ValueError("Team tasks must be sorted and unique")
        for refs, object_type in (
            (self.artifact_head_refs, "artifact-head"),
            (self.open_conflict_refs, "team-conflict"),
        ):
            if (
                tuple(sorted(refs, key=_ref_key)) != refs
                or len(refs) != len(set(refs))
                or any(value.object_type != object_type for value in refs)
            ):
                raise ValueError("Team projection refs are invalid")
        return self


class AgentShellPlanReviewSummaryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-plan-review-summary/v1"] = (
        "eval-factory/agent-shell-plan-review-summary/v1"
    )
    run_id: Identifier
    request_ref: ObjectRef
    result_ref: ObjectRef
    plan_ref: ObjectRef
    plan_kind: PlanKindV2
    plan_version: int = Field(ge=1, le=1_000_000)
    state: PlanReviewStateV2
    decision_ref: ObjectRef | None = None
    revision_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_review(self) -> AgentShellPlanReviewSummaryV1:
        if (
            self.request_ref.object_type != "plan-review-request"
            or self.result_ref.object_type != "plan-review-result"
        ):
            raise ValueError("PlanReview refs are invalid")
        if self.decision_ref is not None and self.decision_ref.object_type != "plan-decision":
            raise ValueError("PlanReview decision ref is invalid")
        if self.revision_ref is not None and self.revision_ref.object_type != "plan-revision":
            raise ValueError("PlanReview revision ref is invalid")
        return self


class AgentShellInteractionCardV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-interaction-card/v1"] = (
        "eval-factory/agent-shell-interaction-card/v1"
    )
    interaction_id: Identifier
    kind: AgentShellInteractionKindV1
    state: AgentShellInteractionStateV1
    subject_ref: ObjectRef
    questions: tuple[str, ...] = Field(default=(), max_length=32)
    reason_codes: tuple[Identifier, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_interaction(self) -> AgentShellInteractionCardV1:
        if tuple(sorted(set(self.questions))) != self.questions:
            raise ValueError("interaction questions must be sorted and unique")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("interaction reasons must be sorted and unique")
        if self.kind is AgentShellInteractionKindV1.CLARIFICATION:
            if self.subject_ref.object_type != "requirement-interpretation" or not self.questions:
                raise ValueError("clarification interaction is incomplete")
        elif self.kind is AgentShellInteractionKindV1.PLAN_REVIEW:
            if self.subject_ref.object_type != "plan-review-request":
                raise ValueError("PlanReview interaction subject is invalid")
        elif self.kind is AgentShellInteractionKindV1.PERMISSION:
            if self.subject_ref.object_type != "requirement-interpretation" or not self.reason_codes:
                raise ValueError("permission interaction is incomplete")
        elif self.kind is AgentShellInteractionKindV1.VERIFICATION and (
            self.subject_ref.object_type != "graph-checkpoint" or not self.reason_codes
        ):
            raise ValueError("verification interaction is incomplete")
        return self


class AgentShellWorkspaceDescriptorV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-workspace/v1"] = "eval-factory/agent-shell-workspace/v1"
    kind: AgentShellWorkspaceKindV1
    status: AgentShellWorkspaceStatusV1
    owner_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    item_count: int = Field(ge=0, le=1_000_000)
    reason_codes: tuple[Identifier, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def validate_workspace(self) -> AgentShellWorkspaceDescriptorV1:
        if tuple(sorted(self.owner_refs, key=_ref_key)) != self.owner_refs or len(self.owner_refs) != len(
            set(self.owner_refs)
        ):
            raise ValueError("workspace owner refs must be sorted and unique")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("workspace reasons must be sorted and unique")
        if self.status is AgentShellWorkspaceStatusV1.EMPTY and (self.owner_refs or self.item_count):
            raise ValueError("empty workspace cannot claim owner items")
        return self


class AgentShellDeliverySummaryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-delivery-summary/v1"] = (
        "eval-factory/agent-shell-delivery-summary/v1"
    )
    completion_ref: ObjectRef
    manifest_ref: ObjectRef
    inventory_ref: ObjectRef
    item_count: int = Field(ge=0, le=1_000_000)
    file_count: int = Field(ge=1, le=10_000_000)
    total_bytes: int = Field(ge=0, le=10_000_000_000)
    bundle_sha256: Sha256

    @model_validator(mode="after")
    def validate_delivery(self) -> AgentShellDeliverySummaryV1:
        expectations = (
            (self.completion_ref, "factory-run-completion"),
            (self.manifest_ref, "candidate-dataset-delivery-manifest"),
            (self.inventory_ref, "candidate-dataset-inventory"),
        )
        if any(value.object_type != object_type for value, object_type in expectations):
            raise ValueError("delivery refs are invalid")
        return self


class AgentShellTeamMessageSummaryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-team-message-summary/v1"] = (
        "eval-factory/agent-shell-team-message-summary/v1"
    )
    message_ref: ObjectRef
    sender_member_ref: ObjectRef
    audience: TeamMessageAudienceV1
    message_kind: TeamMessageKindV1
    recipient_member_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000,
    )
    task_ref: ObjectRef | None = None
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    reply_to_message_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_message(self) -> AgentShellTeamMessageSummaryV1:
        if (
            self.message_ref.object_type != "team-message"
            or self.sender_member_ref.object_type != "team-member"
        ):
            raise ValueError("Team message refs are invalid")
        for refs, object_type in (
            (self.recipient_member_refs, "team-member"),
            (self.artifact_envelope_refs, "artifact-envelope"),
        ):
            if (
                tuple(sorted(refs, key=_ref_key)) != refs
                or len(refs) != len(set(refs))
                or any(value.object_type != object_type for value in refs)
            ):
                raise ValueError("Team message projection refs are invalid")
        if self.task_ref is not None and self.task_ref.object_type != "team-task":
            raise ValueError("Team message task ref is invalid")
        if self.reply_to_message_ref is not None and self.reply_to_message_ref.object_type != "team-message":
            raise ValueError("Team reply ref is invalid")
        return self


class AgentShellMemberProjectionV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-member-projection/v1"] = (
        "eval-factory/agent-shell-member-projection/v1"
    )
    team_ref: ObjectRef
    member: AgentShellTeamMemberSummaryV1
    task_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    messages: tuple[AgentShellTeamMessageSummaryV1, ...] = Field(
        default=(),
        max_length=10_000,
    )
    subscription_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000,
    )
    source_fingerprint: Sha256

    @model_validator(mode="after")
    def validate_projection(self) -> AgentShellMemberProjectionV1:
        if self.team_ref.object_type != "agent-team":
            raise ValueError("member projection Team ref is invalid")
        for refs, object_type in (
            (self.task_refs, "team-task"),
            (self.artifact_envelope_refs, "artifact-envelope"),
            (self.subscription_refs, "artifact-subscription"),
        ):
            if (
                tuple(sorted(refs, key=_ref_key)) != refs
                or len(refs) != len(set(refs))
                or any(value.object_type != object_type for value in refs)
            ):
                raise ValueError("member projection refs are invalid")
        if tuple(sorted(self.acceptance_check_refs, key=_ref_key)) != self.acceptance_check_refs or len(
            self.acceptance_check_refs
        ) != len(set(self.acceptance_check_refs)):
            raise ValueError("member acceptance refs are invalid")
        return self


class AgentShellProjectionV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-projection/v1"] = (
        "eval-factory/agent-shell-projection/v1"
    )
    session: AgentShellSessionSummaryV1
    transcript: tuple[AgentShellTranscriptEntryV1, ...] = Field(
        default=(),
        max_length=10_000,
    )
    activity: tuple[AgentShellEventV1, ...] = Field(
        default=(),
        max_length=500,
    )
    requirement_outcome: RequirementInterpretationOutcomeV1 | None = None
    evidence_class: ProviderEvidenceClassV1 | None = None
    source_admission_ref: ObjectRef | None = None
    execution_phase: AgentShellExecutionPhaseV1 = AgentShellExecutionPhaseV1.NOT_CONFIGURED
    factory: AgentShellFactorySummaryV1 | None = None
    graph: AgentShellGraphSummaryV1 | None = None
    team: AgentShellTeamSummaryV1 | None = None
    plan_reviews: tuple[AgentShellPlanReviewSummaryV1, ...] = Field(
        default=(),
        max_length=500,
    )
    pending_interactions: tuple[AgentShellInteractionCardV1, ...] = Field(
        default=(),
        max_length=500,
    )
    workspaces: tuple[AgentShellWorkspaceDescriptorV1, ...] = Field(
        default=(),
        max_length=32,
    )
    delivery: AgentShellDeliverySummaryV1 | None = None
    reconnect_cursor: int = Field(ge=1, le=1_000_000_000)
    source_fingerprint: Sha256

    @model_validator(mode="after")
    def validate_projection(self) -> AgentShellProjectionV1:
        if self.source_admission_ref is not None and (
            self.source_admission_ref.object_type != "harness-source-admission"
            or self.source_admission_ref.object_version != "v1"
        ):
            raise ValueError("source admission ref is invalid")
        if (self.factory is None) != (self.graph is None):
            raise ValueError("Factory and Graph summaries must appear together")
        if self.team is not None and self.graph is None:
            raise ValueError("Team summary requires Graph authority")
        review_refs = tuple(value.request_ref for value in self.plan_reviews)
        if tuple(sorted(review_refs, key=_ref_key)) != review_refs or len(review_refs) != len(
            set(review_refs)
        ):
            raise ValueError("PlanReview summaries must be sorted and unique")
        interaction_ids = tuple(value.interaction_id for value in self.pending_interactions)
        if interaction_ids != tuple(sorted(set(interaction_ids))):
            raise ValueError("pending interactions must be sorted and unique")
        workspace_kinds = tuple(value.kind for value in self.workspaces)
        if self.workspaces and workspace_kinds != tuple(AgentShellWorkspaceKindV1):
            raise ValueError("Agent Shell workspaces must be complete and ordered")
        if self.execution_phase is AgentShellExecutionPhaseV1.COMPLETED and self.delivery is None:
            raise ValueError("completed Agent Shell projection requires delivery")
        return self


class AgentShellSessionPageV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-session-page/v1"] = (
        "eval-factory/agent-shell-session-page/v1"
    )
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    total: int = Field(ge=0)
    sessions: tuple[AgentShellSessionSummaryV1, ...] = Field(
        default=(),
        max_length=500,
    )


class AgentShellCreateSessionCommandV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-create-session-command/v1"] = (
        "eval-factory/agent-shell-create-session-command/v1"
    )
    session_id: Identifier
    incarnation_id: Identifier
    idempotency_key: Identifier


class AgentShellCloseSessionCommandV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-close-session-command/v1"] = (
        "eval-factory/agent-shell-close-session-command/v1"
    )
    expected_session_version: int = Field(ge=1, le=1_000_000_000)
    idempotency_key: Identifier


class AgentShellPostMessageCommandV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-post-message-command/v1"] = (
        "eval-factory/agent-shell-post-message-command/v1"
    )
    expected_session_version: int = Field(ge=1, le=1_000_000_000)
    content: str = Field(min_length=1, max_length=32_768)
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=256,
    )
    idempotency_key: Identifier


class AgentShellReconcileCommandV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-reconcile-command/v1"] = (
        "eval-factory/agent-shell-reconcile-command/v1"
    )
    expected_session_version: int = Field(ge=1, le=1_000_000_000)
    idempotency_key: Identifier


class AgentShellSourceFileClaimV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-source-file-claim/v1"] = (
        "eval-factory/agent-shell-source-file-claim/v1"
    )
    relative_name: SourceUploadNameV1
    expected_sha256: Sha256
    expected_size_bytes: int = Field(ge=1, le=1_000_000_000)

    @model_validator(mode="after")
    def validate_trace_name(self) -> AgentShellSourceFileClaimV1:
        if ".." in self.relative_name:
            raise ValueError("trace upload claim name is unsafe")
        if not self.relative_name.endswith(".jsonl"):
            raise ValueError("trace upload claim must use the .jsonl extension")
        return self


class AgentShellSourceAdmissionCommandV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-source-admission-command/v1"] = (
        "eval-factory/agent-shell-source-admission-command/v1"
    )
    expected_session_version: int = Field(ge=1, le=1_000_000_000)
    manifest_sha256: Sha256
    manifest_size_bytes: int = Field(ge=1, le=16_777_216)
    trace_files: tuple[AgentShellSourceFileClaimV1, ...] = Field(
        min_length=1,
        max_length=500,
    )
    idempotency_key: Identifier

    @model_validator(mode="after")
    def validate_claims(self) -> AgentShellSourceAdmissionCommandV1:
        names = tuple(item.relative_name for item in self.trace_files)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("trace upload claims must be sorted and unique")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("trace upload claims cannot contain aliased names")
        return self


class AgentShellSourceFileV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-source-file/v1"] = (
        "eval-factory/agent-shell-source-file/v1"
    )
    kind: AgentShellSourceFileKindV1
    relative_name: SourceUploadNameV1
    media_type: Literal["text/csv", "application/x-ndjson"]
    size_bytes: int = Field(ge=1, le=1_000_000_000)
    sha256: Sha256
    source_ref: ObjectRef | None = None
    artifact_envelope_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_file(self) -> AgentShellSourceFileV1:
        if ".." in self.relative_name:
            raise ValueError("source file name is unsafe")
        trace = self.kind is AgentShellSourceFileKindV1.TRACE
        if trace != (self.source_ref is not None):
            raise ValueError("source_ref must exactly match a trace file")
        if trace != (self.artifact_envelope_ref is not None):
            raise ValueError(
                "artifact_envelope_ref must exactly match a trace file",
            )
        if trace:
            assert self.source_ref is not None
            assert self.artifact_envelope_ref is not None
            if (
                self.media_type != "application/x-ndjson"
                or not self.relative_name.endswith(".jsonl")
                or self.source_ref.object_type != "trace-source"
                or self.source_ref.object_version != "v2"
                or self.source_ref.object_sha256 != self.sha256
                or self.artifact_envelope_ref.object_type != "artifact-envelope"
                or self.artifact_envelope_ref.object_version != "v1"
            ):
                raise ValueError("trace file projection is invalid")
        elif self.relative_name != "manifest.csv" or self.media_type != "text/csv":
            raise ValueError("manifest file projection is invalid")
        return self


class AgentShellSourceAdmissionV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-source-admission/v1"] = (
        "eval-factory/agent-shell-source-admission/v1"
    )
    status: Literal["ADMITTED"] = "ADMITTED"
    admission_ref: ObjectRef
    session_id: Identifier
    session_version: int = Field(ge=1, le=1_000_000_000)
    manifest_sha256: Sha256
    source_count: int = Field(ge=1, le=500)
    total_source_bytes: int = Field(ge=1, le=1_000_000_000)
    files: tuple[AgentShellSourceFileV1, ...] = Field(
        min_length=2,
        max_length=501,
    )
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=500,
    )
    created_at: datetime

    @model_validator(mode="after")
    def validate_admission(self) -> AgentShellSourceAdmissionV1:
        if (
            self.admission_ref.object_type != "harness-source-admission"
            or self.admission_ref.object_version != "v1"
        ):
            raise ValueError("admission_ref is invalid")
        if (
            self.files[0].kind is not AgentShellSourceFileKindV1.MANIFEST
            or self.files[0].sha256 != self.manifest_sha256
        ):
            raise ValueError("source admission must list manifest first")
        traces = self.files[1:]
        if (
            any(item.kind is not AgentShellSourceFileKindV1.TRACE for item in traces)
            or len(traces) != self.source_count
            or sum(item.size_bytes for item in traces) != self.total_source_bytes
        ):
            raise ValueError("source admission trace summary is inconsistent")
        trace_names = tuple(item.relative_name for item in traces)
        source_refs = tuple(item.source_ref for item in traces if item.source_ref is not None)
        if (
            trace_names != tuple(sorted(trace_names))
            or len(trace_names) != len(set(trace_names))
            or len({name.casefold() for name in trace_names}) != len(trace_names)
            or len(source_refs) != len(set(source_refs))
            or len({item.sha256 for item in traces}) != len(traces)
        ):
            raise ValueError("source admission trace files are not unique")
        refs = tuple(item.artifact_envelope_ref for item in traces if item.artifact_envelope_ref is not None)
        ordered_refs = tuple(sorted(refs, key=_ref_key))
        if (
            len(refs) != len(set(refs))
            or self.artifact_envelope_refs
            != tuple(
                sorted(
                    self.artifact_envelope_refs,
                    key=_ref_key,
                )
            )
            or ordered_refs != self.artifact_envelope_refs
        ):
            raise ValueError("source admission envelope refs are inconsistent")
        return self


class AgentShellSourceAdmissionContractV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-source-admission-contract/v1"] = (
        "eval-factory/agent-shell-source-admission-contract/v1"
    )
    manifest_name: Literal["manifest.csv"] = "manifest.csv"
    manifest_columns: tuple[str, ...] = SOURCE_MANIFEST_COLUMNS
    max_source_files: int = Field(ge=1, le=500)
    max_manifest_bytes: int = Field(ge=1, le=16_777_216)
    max_source_bytes: int = Field(ge=1, le=1_000_000_000)
    max_total_source_bytes: int = Field(ge=1, le=1_000_000_000)
    max_request_bytes: int = Field(ge=1, le=2_000_000_000)
    error_codes: tuple[AgentShellSourceAdmissionErrorCodeV1, ...] = tuple(
        AgentShellSourceAdmissionErrorCodeV1
    )

    @model_validator(mode="after")
    def validate_contract(self) -> AgentShellSourceAdmissionContractV1:
        if (
            self.manifest_columns != SOURCE_MANIFEST_COLUMNS
            or self.error_codes != tuple(AgentShellSourceAdmissionErrorCodeV1)
            or self.max_total_source_bytes < self.max_source_bytes
            or self.max_request_bytes < self.max_manifest_bytes + self.max_total_source_bytes
        ):
            raise ValueError("source admission contract is inconsistent")
        return self


class AgentShellApiContractV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-api-contract/v1"] = (
        "eval-factory/agent-shell-api-contract/v1"
    )
    principal_header: Literal["X-Eval-Factory-Principal"] = "X-Eval-Factory-Principal"
    event_stream_media_type: Literal["text/event-stream"] = "text/event-stream"
    event_families: tuple[AgentShellEventFamilyV1, ...] = tuple(AgentShellEventFamilyV1)
    max_event_page_size: Literal[500] = 500


def agent_shell_api_contract() -> AgentShellApiContractV1:
    return AgentShellApiContractV1()


__all__ = [
    "AgentShellApiContractV1",
    "AgentShellCloseSessionCommandV1",
    "AgentShellCreateSessionCommandV1",
    "AgentShellDeliverySummaryV1",
    "AgentShellEventFamilyV1",
    "AgentShellEventKindV1",
    "AgentShellEventPageV1",
    "AgentShellEventV1",
    "AgentShellExecutionPhaseV1",
    "AgentShellFactorySummaryV1",
    "AgentShellGraphSummaryV1",
    "AgentShellInteractionCardV1",
    "AgentShellInteractionKindV1",
    "AgentShellInteractionStateV1",
    "AgentShellMemberProjectionV1",
    "AgentShellPlanReviewSummaryV1",
    "AgentShellPostMessageCommandV1",
    "AgentShellProjectionV1",
    "AgentShellReconcileCommandV1",
    "AgentShellSessionPageV1",
    "AgentShellSessionSummaryV1",
    "AgentShellSourceAdmissionCommandV1",
    "AgentShellSourceAdmissionContractV1",
    "AgentShellSourceAdmissionErrorCodeV1",
    "AgentShellSourceAdmissionV1",
    "AgentShellSourceFileClaimV1",
    "AgentShellSourceFileKindV1",
    "AgentShellSourceFileV1",
    "AgentShellTeamMemberSummaryV1",
    "AgentShellTeamMessageSummaryV1",
    "AgentShellTeamSummaryV1",
    "AgentShellTeamTaskSummaryV1",
    "AgentShellTranscriptEntryV1",
    "AgentShellWorkspaceDescriptorV1",
    "AgentShellWorkspaceKindV1",
    "AgentShellWorkspaceStatusV1",
    "SourceUploadNameV1",
    "agent_shell_api_contract",
]
