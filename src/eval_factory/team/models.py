from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique,
    require_sorted_unique_refs,
    sorted_refs,
)


class TeamLifecycleV1(StrEnum):
    PROPOSED = "PROPOSED"
    FORMING = "FORMING"
    ACTIVE = "ACTIVE"
    CONVERGING = "CONVERGING"
    WAITING_USER = "WAITING_USER"
    WAITING_REVIEW = "WAITING_REVIEW"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TeamMemberStatusV1(StrEnum):
    PROPOSED = "PROPOSED"
    READY = "READY"
    ACTIVE = "ACTIVE"
    WAITING = "WAITING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TeamTaskStatusV1(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    CLAIMED = "CLAIMED"
    ACTIVE = "ACTIVE"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TeamMessageKindV1(StrEnum):
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    PROPOSAL = "PROPOSAL"
    CHALLENGE = "CHALLENGE"
    ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
    BLOCKED = "BLOCKED"
    HANDOFF = "HANDOFF"
    REVIEW_REQUEST = "REVIEW_REQUEST"
    REVIEW_RESPONSE = "REVIEW_RESPONSE"
    SYSTEM = "SYSTEM"


class TeamMessageAudienceV1(StrEnum):
    DIRECT = "DIRECT"
    BROADCAST = "BROADCAST"
    SYSTEM = "SYSTEM"


class TeamConflictStatusV1(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    SUPERSEDED = "SUPERSEDED"


class TeamTaskEventKindV1(StrEnum):
    CLAIMED = "CLAIMED"
    LEASE_ACQUIRED = "LEASE_ACQUIRED"
    HEARTBEAT = "HEARTBEAT"
    EXPIRED = "EXPIRED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    EXHAUSTED = "EXHAUSTED"


class TeamConvergenceOutcomeV1(StrEnum):
    CONTINUE = "CONTINUE"
    REWORK = "REWORK"
    WAITING_REVIEW = "WAITING_REVIEW"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"


class TeamOutboxKindV1(StrEnum):
    TEAM_CHANGED = "TEAM_CHANGED"
    TEAM_MESSAGE_POSTED = "TEAM_MESSAGE_POSTED"
    ARTIFACT_HEAD_CHANGED = "ARTIFACT_HEAD_CHANGED"
    CONFLICT_OPENED = "CONFLICT_OPENED"
    CONFLICT_RESOLVED = "CONFLICT_RESOLVED"
    TEAM_CHECKPOINTED = "TEAM_CHECKPOINTED"


class TeamMemberV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-member/v1"] = "eval-harness/team-member/v1"
    OBJECT_TYPE: ClassVar[str] = "team-member"

    member_id: Identifier
    member_incarnation_id: Identifier
    role: Identifier
    agent_profile_ref: ObjectRef
    independent_session_ref: ObjectRef
    capability_definition_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    status: TeamMemberStatusV1
    is_coordinator: bool = False

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        require_ref(
            self.agent_profile_ref,
            "pack-agent-profile",
            "agent_profile_ref",
        )
        require_ref(
            self.independent_session_ref,
            "harness-session",
            "independent_session_ref",
        )
        require_sorted_unique_refs(
            self.capability_definition_refs,
            "capability_definition_refs",
        )
        if any(ref.object_type != "harness-capability-definition" for ref in self.capability_definition_refs):
            raise ValueError("member capabilities must reference Harness definitions")
        return self


class TeamRosterV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-roster/v1"] = "eval-harness/team-roster/v1"
    OBJECT_TYPE: ClassVar[str] = "team-roster"

    roster_id: Identifier
    team_id: Identifier
    revision: int = Field(ge=1, le=1_000_000_000)
    predecessor_roster_ref: ObjectRef | None = None
    members: tuple[TeamMemberV1, ...] = Field(min_length=1, max_length=1_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        roster_id: str,
        team_id: str,
        revision: int,
        predecessor_roster_ref: ObjectRef | None,
        members: tuple[TeamMemberV1, ...],
        audit: ContractAudit,
    ) -> TeamRosterV1:
        return super().create(
            audit=audit,
            roster_id=roster_id,
            team_id=team_id,
            revision=revision,
            predecessor_roster_ref=predecessor_roster_ref,
            members=tuple(sorted(members, key=lambda value: value.member_id)),
        )

    @model_validator(mode="after")
    def validate_roster(self) -> Self:
        member_ids = tuple(member.member_id for member in self.members)
        require_sorted_unique(member_ids, "member IDs")
        incarnation_ids = tuple(sorted(member.member_incarnation_id for member in self.members))
        require_sorted_unique(incarnation_ids, "member incarnation IDs")
        if sum(member.is_coordinator for member in self.members) != 1:
            raise ValueError("Team roster requires exactly one coordinator")
        if self.revision == 1:
            if self.predecessor_roster_ref is not None:
                raise ValueError("first roster cannot have a predecessor")
        elif self.predecessor_roster_ref is None:
            raise ValueError("roster successor requires its predecessor")
        else:
            require_ref(
                self.predecessor_roster_ref,
                "team-roster",
                "predecessor_roster_ref",
            )
        return self


class TeamTaskV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-task/v1"] = "eval-harness/team-task/v1"
    OBJECT_TYPE: ClassVar[str] = "team-task"

    task_id: Identifier
    task_kind: Identifier
    dependency_task_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    assigned_member_id: Identifier | None = None
    capability_definition_ref: ObjectRef
    input_artifact_head_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    output_artifact_head_ids: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    status: TeamTaskStatusV1
    max_attempts: int = Field(ge=1, le=100)
    max_model_requests: int = Field(ge=0, le=1_000_000)
    max_model_tokens: int = Field(ge=0, le=1_000_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=1_000_000_000_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        task_id: str,
        task_kind: str,
        dependency_task_ids: tuple[str, ...],
        assigned_member_id: str | None,
        capability_definition_ref: ObjectRef,
        input_artifact_head_ids: tuple[str, ...],
        output_artifact_head_ids: tuple[str, ...],
        acceptance_check_refs: tuple[ObjectRef, ...],
        status: TeamTaskStatusV1,
        max_attempts: int,
        max_model_requests: int,
        max_model_tokens: int,
        max_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> TeamTaskV1:
        return super().create(
            audit=audit,
            task_id=task_id,
            task_kind=task_kind,
            dependency_task_ids=tuple(sorted(dependency_task_ids)),
            assigned_member_id=assigned_member_id,
            capability_definition_ref=capability_definition_ref,
            input_artifact_head_ids=tuple(sorted(input_artifact_head_ids)),
            output_artifact_head_ids=tuple(sorted(output_artifact_head_ids)),
            acceptance_check_refs=sorted_refs(acceptance_check_refs),
            status=status,
            max_attempts=max_attempts,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
        )

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        require_sorted_unique(self.dependency_task_ids, "dependency_task_ids")
        if self.task_id in self.dependency_task_ids:
            raise ValueError("Team task cannot depend on itself")
        require_ref(
            self.capability_definition_ref,
            "harness-capability-definition",
            "capability_definition_ref",
        )
        require_sorted_unique(
            self.input_artifact_head_ids,
            "input_artifact_head_ids",
        )
        require_sorted_unique(
            self.output_artifact_head_ids,
            "output_artifact_head_ids",
        )
        require_sorted_unique_refs(
            self.acceptance_check_refs,
            "acceptance_check_refs",
        )
        return self


class TeamTaskGraphV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-task-graph/v1"] = "eval-harness/team-task-graph/v1"
    OBJECT_TYPE: ClassVar[str] = "team-task-graph"

    graph_id: Identifier
    team_id: Identifier
    revision: int = Field(ge=1, le=1_000_000_000)
    predecessor_graph_ref: ObjectRef | None = None
    roster_ref: ObjectRef
    member_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1_000)
    tasks: tuple[TeamTaskV1, ...] = Field(min_length=1, max_length=100_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        graph_id: str,
        team_id: str,
        revision: int,
        predecessor_graph_ref: ObjectRef | None,
        roster_ref: ObjectRef,
        member_ids: tuple[str, ...],
        tasks: tuple[TeamTaskV1, ...],
        audit: ContractAudit,
    ) -> TeamTaskGraphV1:
        return super().create(
            audit=audit,
            graph_id=graph_id,
            team_id=team_id,
            revision=revision,
            predecessor_graph_ref=predecessor_graph_ref,
            roster_ref=roster_ref,
            member_ids=tuple(sorted(member_ids)),
            tasks=tuple(sorted(tasks, key=lambda value: value.task_id)),
        )

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        require_ref(self.roster_ref, "team-roster", "roster_ref")
        require_sorted_unique(self.member_ids, "member_ids")
        task_ids = tuple(task.task_id for task in self.tasks)
        require_sorted_unique(task_ids, "task IDs")
        known_tasks = set(task_ids)
        known_members = set(self.member_ids)
        output_owners: dict[str, str] = {}
        for task in self.tasks:
            if not set(task.dependency_task_ids).issubset(known_tasks):
                raise ValueError("Team task depends on an unknown task")
            if task.assigned_member_id is not None and task.assigned_member_id not in known_members:
                raise ValueError("Team task is assigned to an unknown member")
            for head_id in task.output_artifact_head_ids:
                prior = output_owners.get(head_id)
                if prior is not None and prior != task.task_id:
                    raise ValueError("artifact head has overlapping task write ownership")
                output_owners[head_id] = task.task_id
        _require_acyclic(self.tasks)
        if self.revision == 1:
            if self.predecessor_graph_ref is not None:
                raise ValueError("first task graph cannot have a predecessor")
        elif self.predecessor_graph_ref is None:
            raise ValueError("task graph successor requires its predecessor")
        else:
            require_ref(
                self.predecessor_graph_ref,
                "team-task-graph",
                "predecessor_graph_ref",
            )
        return self


class TeamV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/agent-team/v1"] = "eval-harness/agent-team/v1"
    OBJECT_TYPE: ClassVar[str] = "agent-team"

    team_id: Identifier
    team_incarnation_id: Identifier
    team_version: int = Field(ge=1, le=1_000_000_000)
    goal_ref: ObjectRef
    composition_ref: ObjectRef
    roster_ref: ObjectRef
    task_graph_ref: ObjectRef
    authority_ref: ObjectRef
    coordinator_member_id: Identifier
    lifecycle: TeamLifecycleV1

    @model_validator(mode="after")
    def validate_team(self) -> Self:
        require_ref(
            self.composition_ref,
            "harness-composition",
            "composition_ref",
        )
        require_ref(self.roster_ref, "team-roster", "roster_ref")
        require_ref(self.task_graph_ref, "team-task-graph", "task_graph_ref")
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        return self


class TeamTaskClaimV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-task-claim/v1"] = "eval-harness/team-task-claim/v1"
    OBJECT_TYPE: ClassVar[str] = "team-task-claim"

    claim_id: Identifier
    team_ref: ObjectRef
    task_ref: ObjectRef
    member_ref: ObjectRef
    graph_revision: int = Field(ge=1, le=1_000_000_000)
    claimed_at: datetime

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.task_ref, "team-task", "task_ref")
        require_ref(self.member_ref, "team-member", "member_ref")
        return self


class TeamTaskLeaseV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-task-lease/v1"] = "eval-harness/team-task-lease/v1"
    OBJECT_TYPE: ClassVar[str] = "team-task-lease"

    lease_id: Identifier
    claim_ref: ObjectRef
    fencing_token: int = Field(ge=1, le=1_000_000_000)
    acquired_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def validate_lease(self) -> Self:
        require_ref(self.claim_ref, "team-task-claim", "claim_ref")
        if self.expires_at <= self.acquired_at:
            raise ValueError("Team task lease must expire after acquisition")
        return self


class TeamMessageV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-message/v1"] = "eval-harness/team-message/v1"
    OBJECT_TYPE: ClassVar[str] = "team-message"

    message_id: Identifier
    team_ref: ObjectRef
    sender_member_ref: ObjectRef
    audience: TeamMessageAudienceV1
    recipient_member_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000,
    )
    message_kind: TeamMessageKindV1
    task_ref: ObjectRef | None = None
    message_body_ref: ObjectRef
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    reply_to_message_ref: ObjectRef | None = None

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        message_id: str,
        team_ref: ObjectRef,
        sender_member_ref: ObjectRef,
        audience: TeamMessageAudienceV1,
        recipient_member_refs: tuple[ObjectRef, ...],
        message_kind: TeamMessageKindV1,
        task_ref: ObjectRef | None,
        message_body_ref: ObjectRef,
        artifact_envelope_refs: tuple[ObjectRef, ...],
        reply_to_message_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> TeamMessageV1:
        return super().create(
            audit=audit,
            message_id=message_id,
            team_ref=team_ref,
            sender_member_ref=sender_member_ref,
            audience=audience,
            recipient_member_refs=sorted_refs(recipient_member_refs),
            message_kind=message_kind,
            task_ref=task_ref,
            message_body_ref=message_body_ref,
            artifact_envelope_refs=sorted_refs(artifact_envelope_refs),
            reply_to_message_ref=reply_to_message_ref,
        )

    @model_validator(mode="after")
    def validate_message(self) -> Self:
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.sender_member_ref, "team-member", "sender_member_ref")
        require_ref(
            self.message_body_ref,
            "team-message-body",
            "message_body_ref",
        )
        require_sorted_unique_refs(
            self.recipient_member_refs,
            "recipient_member_refs",
        )
        if any(ref.object_type != "team-member" for ref in self.recipient_member_refs):
            raise ValueError("message recipients must reference Team members")
        if self.audience is TeamMessageAudienceV1.DIRECT:
            if not self.recipient_member_refs:
                raise ValueError("direct message requires at least one recipient")
        elif self.recipient_member_refs:
            raise ValueError("broadcast/system message cannot name recipients")
        if self.task_ref is not None:
            require_ref(self.task_ref, "team-task", "task_ref")
        require_sorted_unique_refs(
            self.artifact_envelope_refs,
            "artifact_envelope_refs",
        )
        if any(ref.object_type != "artifact-envelope" for ref in self.artifact_envelope_refs):
            raise ValueError("Team messages may exchange artifact envelopes only")
        if self.reply_to_message_ref is not None:
            require_ref(
                self.reply_to_message_ref,
                "team-message",
                "reply_to_message_ref",
            )
        return self


class ArtifactSubscriptionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/artifact-subscription/v1"] = "eval-harness/artifact-subscription/v1"
    OBJECT_TYPE: ClassVar[str] = "artifact-subscription"

    subscription_id: Identifier
    team_ref: ObjectRef
    member_ref: ObjectRef
    task_ids: tuple[Identifier, ...] = Field(default=(), max_length=10_000)
    artifact_head_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    semantic_roles: tuple[Identifier, ...] = Field(default=(), max_length=256)
    cursor: int = Field(ge=0, le=1_000_000_000)

    @model_validator(mode="after")
    def validate_subscription(self) -> Self:
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.member_ref, "team-member", "member_ref")
        for values, label in (
            (self.task_ids, "task_ids"),
            (self.artifact_head_ids, "artifact_head_ids"),
            (self.semantic_roles, "semantic_roles"),
        ):
            require_sorted_unique(values, label)
        if not self.task_ids and not self.artifact_head_ids and not self.semantic_roles:
            raise ValueError("artifact subscription requires a bounded selector")
        return self


class TeamConflictV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-conflict/v1"] = "eval-harness/team-conflict/v1"
    OBJECT_TYPE: ClassVar[str] = "team-conflict"

    conflict_id: Identifier
    team_ref: ObjectRef
    task_ref: ObjectRef
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        min_length=2,
        max_length=100,
    )
    status: TeamConflictStatusV1
    resolution_artifact_ref: ObjectRef | None = None
    resolved_by_member_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_conflict(self) -> Self:
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.task_ref, "team-task", "task_ref")
        require_sorted_unique_refs(
            self.artifact_envelope_refs,
            "artifact_envelope_refs",
        )
        if any(ref.object_type != "artifact-envelope" for ref in self.artifact_envelope_refs):
            raise ValueError("conflict subjects must be artifact envelopes")
        closed = self.status is not TeamConflictStatusV1.OPEN
        has_resolution = self.resolution_artifact_ref is not None and self.resolved_by_member_ref is not None
        if closed != has_resolution:
            raise ValueError(
                "closed conflict requires resolution artifact and resolver",
            )
        if self.resolution_artifact_ref is not None:
            require_ref(
                self.resolution_artifact_ref,
                "artifact-envelope",
                "resolution_artifact_ref",
            )
        if self.resolved_by_member_ref is not None:
            require_ref(
                self.resolved_by_member_ref,
                "team-member",
                "resolved_by_member_ref",
            )
        return self


class MemberContextProjectionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/member-context-projection/v1"] = (
        "eval-harness/member-context-projection/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "member-context-projection"

    projection_id: Identifier
    team_ref: ObjectRef
    member_ref: ObjectRef
    authority_ref: ObjectRef
    task_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    message_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    subscription_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    acceptance_check_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000,
    )

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.member_ref, "team-member", "member_ref")
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        expectations = (
            (self.task_refs, "task_refs", "team-task"),
            (
                self.artifact_envelope_refs,
                "artifact_envelope_refs",
                "artifact-envelope",
            ),
            (self.message_refs, "message_refs", "team-message"),
            (
                self.subscription_refs,
                "subscription_refs",
                "artifact-subscription",
            ),
        )
        for values, label, object_type in expectations:
            require_sorted_unique_refs(values, label)
            if any(ref.object_type != object_type for ref in values):
                raise ValueError(f"{label} contains an invalid object type")
        require_sorted_unique_refs(
            self.acceptance_check_refs,
            "acceptance_check_refs",
        )
        return self


class TeamCheckpointV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-checkpoint/v1"] = "eval-harness/team-checkpoint/v1"
    OBJECT_TYPE: ClassVar[str] = "team-checkpoint"

    checkpoint_id: Identifier
    team_ref: ObjectRef
    roster_ref: ObjectRef
    task_graph_ref: ObjectRef
    authority_ref: ObjectRef
    artifact_head_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=100_000,
    )
    subscription_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=100_000,
    )
    member_session_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    message_cursor: int = Field(ge=0, le=1_000_000_000)

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.roster_ref, "team-roster", "roster_ref")
        require_ref(self.task_graph_ref, "team-task-graph", "task_graph_ref")
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        expectations = (
            (self.artifact_head_refs, "artifact_head_refs", "artifact-head"),
            (
                self.subscription_refs,
                "subscription_refs",
                "artifact-subscription",
            ),
            (
                self.member_session_refs,
                "member_session_refs",
                "harness-session",
            ),
        )
        for values, label, object_type in expectations:
            require_sorted_unique_refs(values, label)
            if any(ref.object_type != object_type for ref in values):
                raise ValueError(f"{label} contains an invalid object type")
        return self


class TeamTaskEventV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-task-event/v1"] = "eval-harness/team-task-event/v1"
    OBJECT_TYPE: ClassVar[str] = "team-task-event"

    event_key: Identifier
    team_ref: ObjectRef
    task_ref: ObjectRef
    graph_ref: ObjectRef
    graph_revision: int = Field(ge=1, le=1_000_000_000)
    task_event_version: int = Field(ge=1, le=1_000_000_000)
    attempt: int = Field(ge=1, le=100)
    event_kind: TeamTaskEventKindV1
    status_after: TeamTaskStatusV1
    claim_ref: ObjectRef | None = None
    lease_ref: ObjectRef | None = None
    fencing_token: int | None = Field(default=None, ge=1, le=1_000_000_000)
    effective_expires_at: datetime | None = None
    capability_call_ref: ObjectRef | None = None
    capability_result_ref: ObjectRef | None = None
    reason_code: Identifier | None = None
    occurred_at: datetime

    @model_validator(mode="after")
    def validate_task_event(self) -> Self:
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.task_ref, "team-task", "task_ref")
        require_ref(self.graph_ref, "team-task-graph", "graph_ref")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("Team task event time must be timezone-aware")
        if self.claim_ref is not None:
            require_ref(self.claim_ref, "team-task-claim", "claim_ref")
        if self.lease_ref is not None:
            require_ref(self.lease_ref, "team-task-lease", "lease_ref")
        lease_bound = self.event_kind in {
            TeamTaskEventKindV1.LEASE_ACQUIRED,
            TeamTaskEventKindV1.HEARTBEAT,
            TeamTaskEventKindV1.EXPIRED,
            TeamTaskEventKindV1.CANCEL_REQUESTED,
            TeamTaskEventKindV1.CANCELLED,
            TeamTaskEventKindV1.COMPLETED,
            TeamTaskEventKindV1.EXHAUSTED,
        }
        if lease_bound != (
            self.claim_ref is not None and self.lease_ref is not None and self.fencing_token is not None
        ):
            raise ValueError("lease-bound task event requires claim, lease, and fence")
        if self.event_kind is TeamTaskEventKindV1.CLAIMED and (
            self.claim_ref is None or self.status_after is not TeamTaskStatusV1.CLAIMED
        ):
            raise ValueError("claimed event requires claim authority")
        if (
            self.event_kind
            in {
                TeamTaskEventKindV1.LEASE_ACQUIRED,
                TeamTaskEventKindV1.HEARTBEAT,
                TeamTaskEventKindV1.CANCEL_REQUESTED,
            }
            and self.status_after is not TeamTaskStatusV1.ACTIVE
        ):
            raise ValueError("active lease event must retain ACTIVE task status")
        if (
            self.event_kind
            in {
                TeamTaskEventKindV1.LEASE_ACQUIRED,
                TeamTaskEventKindV1.HEARTBEAT,
            }
            and self.effective_expires_at is None
        ):
            raise ValueError("active lease event requires effective expiry")
        if self.effective_expires_at is not None and (
            self.effective_expires_at.tzinfo is None or self.effective_expires_at.utcoffset() is None
        ):
            raise ValueError("effective expiry must be timezone-aware")
        completion_bound = self.event_kind is TeamTaskEventKindV1.COMPLETED
        if completion_bound != (
            self.capability_call_ref is not None and self.capability_result_ref is not None
        ):
            raise ValueError("completed event requires exact Capability call and result")
        if self.capability_call_ref is not None:
            require_ref(
                self.capability_call_ref,
                "harness-capability-call",
                "capability_call_ref",
            )
        if self.capability_result_ref is not None:
            require_ref(
                self.capability_result_ref,
                "harness-capability-result",
                "capability_result_ref",
            )
        expected_statuses = {
            TeamTaskEventKindV1.EXPIRED: {TeamTaskStatusV1.BLOCKED},
            TeamTaskEventKindV1.RETRY_SCHEDULED: {
                TeamTaskStatusV1.PENDING,
                TeamTaskStatusV1.READY,
            },
            TeamTaskEventKindV1.CANCELLED: {TeamTaskStatusV1.CANCELLED},
            TeamTaskEventKindV1.COMPLETED: {
                TeamTaskStatusV1.COMPLETED,
                TeamTaskStatusV1.FAILED,
            },
            TeamTaskEventKindV1.EXHAUSTED: {TeamTaskStatusV1.FAILED},
        }
        allowed = expected_statuses.get(self.event_kind)
        if allowed is not None and self.status_after not in allowed:
            raise ValueError("task event kind and resulting status disagree")
        reason_required = self.event_kind in {
            TeamTaskEventKindV1.EXPIRED,
            TeamTaskEventKindV1.CANCELLED,
            TeamTaskEventKindV1.EXHAUSTED,
        } or (
            self.event_kind is TeamTaskEventKindV1.COMPLETED and self.status_after is TeamTaskStatusV1.FAILED
        )
        reason_allowed = reason_required or self.event_kind is TeamTaskEventKindV1.RETRY_SCHEDULED
        if (reason_required and self.reason_code is None) or (
            not reason_allowed and self.reason_code is not None
        ):
            raise ValueError("terminal or failed task event requires one closed reason")
        return self


class TeamConvergenceDecisionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-convergence-decision/v1"] = (
        "eval-harness/team-convergence-decision/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "team-convergence-decision"

    decision_key: Identifier
    team_ref: ObjectRef
    graph_ref: ObjectRef
    authority_ref: ObjectRef
    checkpoint_ref: ObjectRef
    outcome: TeamConvergenceOutcomeV1
    blocking_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    successor_team_ref: ObjectRef | None = None
    successor_graph_ref: ObjectRef | None = None
    successor_authority_ref: ObjectRef | None = None
    reason_codes: tuple[Identifier, ...] = Field(default=(), max_length=64)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        decision_key: str,
        team_ref: ObjectRef,
        graph_ref: ObjectRef,
        authority_ref: ObjectRef,
        checkpoint_ref: ObjectRef,
        outcome: TeamConvergenceOutcomeV1,
        blocking_refs: tuple[ObjectRef, ...],
        successor_team_ref: ObjectRef | None,
        successor_graph_ref: ObjectRef | None,
        successor_authority_ref: ObjectRef | None,
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
    ) -> TeamConvergenceDecisionV1:
        return super().create(
            audit=audit,
            decision_key=decision_key,
            team_ref=team_ref,
            graph_ref=graph_ref,
            authority_ref=authority_ref,
            checkpoint_ref=checkpoint_ref,
            outcome=outcome,
            blocking_refs=sorted_refs(blocking_refs),
            successor_team_ref=successor_team_ref,
            successor_graph_ref=successor_graph_ref,
            successor_authority_ref=successor_authority_ref,
            reason_codes=tuple(sorted(reason_codes)),
        )

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.graph_ref, "team-task-graph", "graph_ref")
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        require_ref(self.checkpoint_ref, "team-checkpoint", "checkpoint_ref")
        require_sorted_unique_refs(self.blocking_refs, "blocking_refs")
        require_sorted_unique(self.reason_codes, "reason_codes")
        successors = (
            self.successor_team_ref,
            self.successor_graph_ref,
            self.successor_authority_ref,
        )
        if self.outcome is TeamConvergenceOutcomeV1.REWORK:
            if any(value is None for value in successors):
                raise ValueError("rework decision requires all successor authorities")
            assert self.successor_team_ref is not None
            assert self.successor_graph_ref is not None
            assert self.successor_authority_ref is not None
            require_ref(self.successor_team_ref, "agent-team", "successor_team_ref")
            require_ref(
                self.successor_graph_ref,
                "team-task-graph",
                "successor_graph_ref",
            )
            require_ref(
                self.successor_authority_ref,
                "execution-authority",
                "successor_authority_ref",
            )
        elif any(value is not None for value in successors):
            raise ValueError("only rework may bind successor Team authority")
        if self.outcome is TeamConvergenceOutcomeV1.COMPLETE and self.blocking_refs:
            raise ValueError("complete convergence cannot retain blockers")
        if self.outcome is not TeamConvergenceOutcomeV1.COMPLETE and not self.reason_codes:
            raise ValueError("non-complete convergence requires a closed reason")
        return self


class TeamOutboxRecordV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/team-outbox-record/v1"] = "eval-harness/team-outbox-record/v1"
    OBJECT_TYPE: ClassVar[str] = "team-outbox-record"

    outbox_key: Identifier
    team_id: Identifier
    sequence: int = Field(ge=1, le=1_000_000_000)
    event_kind: TeamOutboxKindV1
    aggregate_ref: ObjectRef
    record_ref: ObjectRef
    intended_member_session_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=1_000,
    )
    occurred_at: datetime

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        outbox_key: str,
        team_id: str,
        sequence: int,
        event_kind: TeamOutboxKindV1,
        aggregate_ref: ObjectRef,
        record_ref: ObjectRef,
        intended_member_session_refs: tuple[ObjectRef, ...],
        occurred_at: datetime,
        audit: ContractAudit,
    ) -> TeamOutboxRecordV1:
        return super().create(
            audit=audit,
            outbox_key=outbox_key,
            team_id=team_id,
            sequence=sequence,
            event_kind=event_kind,
            aggregate_ref=aggregate_ref,
            record_ref=record_ref,
            intended_member_session_refs=sorted_refs(
                intended_member_session_refs,
            ),
            occurred_at=occurred_at,
        )

    @model_validator(mode="after")
    def validate_outbox(self) -> Self:
        require_ref(self.aggregate_ref, "agent-team", "aggregate_ref")
        expected_types = {
            TeamOutboxKindV1.TEAM_CHANGED: {"agent-team"},
            TeamOutboxKindV1.TEAM_MESSAGE_POSTED: {"team-message"},
            TeamOutboxKindV1.ARTIFACT_HEAD_CHANGED: {"artifact-head"},
            TeamOutboxKindV1.CONFLICT_OPENED: {"team-conflict"},
            TeamOutboxKindV1.CONFLICT_RESOLVED: {"team-conflict"},
            TeamOutboxKindV1.TEAM_CHECKPOINTED: {"team-checkpoint"},
        }
        if self.record_ref.object_type not in expected_types[self.event_kind]:
            raise ValueError("Team outbox record type does not match event kind")
        require_sorted_unique_refs(
            self.intended_member_session_refs,
            "intended_member_session_refs",
        )
        if any(ref.object_type != "harness-session" for ref in self.intended_member_session_refs):
            raise ValueError("Team outbox recipients must be Harness sessions")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("Team outbox time must be timezone-aware")
        return self


def validate_team_task_event_log(
    events: tuple[TeamTaskEventV1, ...],
) -> None:
    if not events:
        return
    task_id = events[0].task_ref.object_id
    expected_version = 1
    prior_status = TeamTaskStatusV1.READY
    prior_graph_revision = 0
    allowed: dict[
        TeamTaskEventKindV1,
        frozenset[TeamTaskStatusV1],
    ] = {
        TeamTaskEventKindV1.CLAIMED: frozenset(
            {
                TeamTaskStatusV1.PENDING,
                TeamTaskStatusV1.READY,
            },
        ),
        TeamTaskEventKindV1.LEASE_ACQUIRED: frozenset({TeamTaskStatusV1.CLAIMED}),
        TeamTaskEventKindV1.HEARTBEAT: frozenset({TeamTaskStatusV1.ACTIVE}),
        TeamTaskEventKindV1.EXPIRED: frozenset({TeamTaskStatusV1.ACTIVE}),
        TeamTaskEventKindV1.RETRY_SCHEDULED: frozenset(
            {
                TeamTaskStatusV1.BLOCKED,
                TeamTaskStatusV1.COMPLETED,
                TeamTaskStatusV1.FAILED,
            },
        ),
        TeamTaskEventKindV1.CANCEL_REQUESTED: frozenset(
            {TeamTaskStatusV1.ACTIVE},
        ),
        TeamTaskEventKindV1.CANCELLED: frozenset({TeamTaskStatusV1.ACTIVE}),
        TeamTaskEventKindV1.COMPLETED: frozenset({TeamTaskStatusV1.ACTIVE}),
        TeamTaskEventKindV1.EXHAUSTED: frozenset(
            {TeamTaskStatusV1.BLOCKED, TeamTaskStatusV1.FAILED},
        ),
    }
    for event in events:
        if event.task_ref.object_id != task_id:
            raise ValueError("task event log cannot mix task authority")
        if event.task_event_version != expected_version:
            raise ValueError("task event version must be contiguous from one")
        if event.graph_revision < prior_graph_revision:
            raise ValueError("task event graph revision cannot move backward")
        if event.event_kind is TeamTaskEventKindV1.RETRY_SCHEDULED:
            if event.graph_revision > prior_graph_revision:
                if prior_status not in {
                    TeamTaskStatusV1.COMPLETED,
                    TeamTaskStatusV1.FAILED,
                    TeamTaskStatusV1.CANCELLED,
                }:
                    raise ValueError("graph rework requires terminal prior task state")
            elif prior_status not in allowed[event.event_kind]:
                raise ValueError("illegal task event transition")
        elif prior_status not in allowed[event.event_kind]:
            raise ValueError("illegal task event transition")
        prior_status = event.status_after
        prior_graph_revision = event.graph_revision
        expected_version += 1


def _require_acyclic(tasks: tuple[TeamTaskV1, ...]) -> None:
    dependencies = {task.task_id: set(task.dependency_task_ids) for task in tasks}
    ready = sorted(task_id for task_id, dependency_ids in dependencies.items() if not dependency_ids)
    visited: list[str] = []
    while ready:
        task_id = ready.pop(0)
        visited.append(task_id)
        for candidate_id in sorted(dependencies):
            if task_id in dependencies[candidate_id]:
                dependencies[candidate_id].remove(task_id)
                if not dependencies[candidate_id] and candidate_id not in visited:
                    ready.append(candidate_id)
                    ready.sort()
    if len(visited) != len(tasks):
        raise ValueError("Team task graph must be acyclic")


__all__ = [
    "ArtifactSubscriptionV1",
    "MemberContextProjectionV1",
    "TeamCheckpointV1",
    "TeamConflictStatusV1",
    "TeamConflictV1",
    "TeamConvergenceDecisionV1",
    "TeamConvergenceOutcomeV1",
    "TeamLifecycleV1",
    "TeamMemberStatusV1",
    "TeamMemberV1",
    "TeamMessageAudienceV1",
    "TeamMessageKindV1",
    "TeamMessageV1",
    "TeamOutboxKindV1",
    "TeamOutboxRecordV1",
    "TeamRosterV1",
    "TeamTaskClaimV1",
    "TeamTaskEventKindV1",
    "TeamTaskEventV1",
    "TeamTaskGraphV1",
    "TeamTaskLeaseV1",
    "TeamTaskStatusV1",
    "TeamTaskV1",
    "TeamV1",
    "validate_team_task_event_log",
]
