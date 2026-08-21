from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    CapabilitySideEffectV1,
    ExecutionAuthorityV1,
    MemberExecutionGrantV1,
    StaticPackRegistrationV1,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs import build_generic_agent_trace_pack
from eval_factory.team import (
    ArtifactSubscriptionV1,
    MemberContextProjectionV1,
    TeamCheckpointV1,
    TeamConflictStatusV1,
    TeamConflictV1,
    TeamLifecycleV1,
    TeamMemberStatusV1,
    TeamMemberV1,
    TeamMessageAudienceV1,
    TeamMessageKindV1,
    TeamMessageV1,
    TeamRosterV1,
    TeamTaskClaimV1,
    TeamTaskGraphV1,
    TeamTaskLeaseV1,
    TeamTaskStatusV1,
    TeamTaskV1,
    TeamV1,
)

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


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 17, tzinfo=UTC),
        created_by="team-contract-test",
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


def _members() -> tuple[TeamMemberV1, TeamMemberV1]:
    registration = _registration()
    profiles = {profile.role: profile for profile in registration.agent_profiles}
    coordinator_profile = profiles["coordinator"]
    task_profile = profiles["task"]
    coordinator = TeamMemberV1.create(
        member_id="member-coordinator",
        member_incarnation_id="member-coordinator-incarnation-001",
        role="coordinator",
        agent_profile_ref=coordinator_profile.to_ref(),
        independent_session_ref=_ref("harness-session", "coordinator"),
        capability_definition_refs=coordinator_profile.capability_definition_refs,
        status=TeamMemberStatusV1.READY,
        is_coordinator=True,
        audit=_audit(),
    )
    task = TeamMemberV1.create(
        member_id="member-task",
        member_incarnation_id="member-task-incarnation-001",
        role="task",
        agent_profile_ref=task_profile.to_ref(),
        independent_session_ref=_ref("harness-session", "task"),
        capability_definition_refs=task_profile.capability_definition_refs,
        status=TeamMemberStatusV1.READY,
        is_coordinator=False,
        audit=_audit(),
    )
    return coordinator, task


def _roster() -> TeamRosterV1:
    return TeamRosterV1.create(
        roster_id="team-roster.generic-agent-eval",
        team_id="team-generic-agent-eval",
        revision=1,
        predecessor_roster_ref=None,
        members=_members(),
        audit=_audit(),
    )


def _tasks() -> tuple[TeamTaskV1, TeamTaskV1]:
    registration = _registration()
    capabilities = {value.capability_id: value for value in registration.capability_definitions}
    requirement = TeamTaskV1.create(
        task_id="task-requirement",
        task_kind="requirement",
        dependency_task_ids=(),
        assigned_member_id="member-coordinator",
        capability_definition_ref=capabilities["capability.requirement-planning"].to_ref(),
        input_artifact_head_ids=(),
        output_artifact_head_ids=("head-requirement",),
        acceptance_check_refs=(_ref("acceptance-check", "requirement"),),
        status=TeamTaskStatusV1.READY,
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=20_000,
        max_cost_micro_usd=2_000_000,
        audit=_audit(),
    )
    task = TeamTaskV1.create(
        task_id="task-authoring",
        task_kind="task-authoring",
        dependency_task_ids=("task-requirement",),
        assigned_member_id="member-task",
        capability_definition_ref=capabilities["capability.task-authoring"].to_ref(),
        input_artifact_head_ids=("head-requirement",),
        output_artifact_head_ids=("head-task",),
        acceptance_check_refs=(_ref("acceptance-check", "task"),),
        status=TeamTaskStatusV1.PENDING,
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=20_000,
        max_cost_micro_usd=2_000_000,
        audit=_audit(),
    )
    return requirement, task


def _replace_task(
    task: TeamTaskV1,
    *,
    dependency_task_ids: tuple[str, ...] | None = None,
    assigned_member_id: str | None = None,
    replace_assignee: bool = False,
    output_artifact_head_ids: tuple[str, ...] | None = None,
) -> TeamTaskV1:
    return TeamTaskV1.create(
        task_id=task.task_id,
        task_kind=task.task_kind,
        dependency_task_ids=(
            task.dependency_task_ids if dependency_task_ids is None else dependency_task_ids
        ),
        assigned_member_id=(assigned_member_id if replace_assignee else task.assigned_member_id),
        capability_definition_ref=task.capability_definition_ref,
        input_artifact_head_ids=task.input_artifact_head_ids,
        output_artifact_head_ids=(
            task.output_artifact_head_ids if output_artifact_head_ids is None else output_artifact_head_ids
        ),
        acceptance_check_refs=task.acceptance_check_refs,
        status=task.status,
        max_attempts=task.max_attempts,
        max_model_requests=task.max_model_requests,
        max_model_tokens=task.max_model_tokens,
        max_cost_micro_usd=task.max_cost_micro_usd,
        audit=_audit(),
    )


def _graph(tasks: tuple[TeamTaskV1, ...] | None = None) -> TeamTaskGraphV1:
    roster = _roster()
    return TeamTaskGraphV1.create(
        graph_id="team-task-graph.generic-agent-eval",
        team_id=roster.team_id,
        revision=1,
        predecessor_graph_ref=None,
        roster_ref=roster.to_ref(),
        member_ids=tuple(member.member_id for member in roster.members),
        tasks=tasks or _tasks(),
        audit=_audit(),
    )


def _authority(
    *,
    roster: TeamRosterV1,
    graph: TeamTaskGraphV1,
) -> ExecutionAuthorityV1:
    registration = _registration()
    policy = registration.permission_policies[0]
    providers = {provider.capability_definition_ref: provider for provider in registration.provider_bindings}
    grants = tuple(
        MemberExecutionGrantV1(
            member_id=member.member_id,
            principal_ref=_ref("principal", member.member_id),
            capability_definition_refs=member.capability_definition_refs,
            provider_binding_refs=sorted_refs(
                providers[capability_ref].to_ref() for capability_ref in member.capability_definition_refs
            ),
            task_ids=tuple(
                sorted(task.task_id for task in graph.tasks if task.assigned_member_id == member.member_id)
            ),
            data_scope_refs=(),
            data_purposes=("evaluation-data-production",),
            data_classifications=("INTERNAL",),
            allowed_side_effects=tuple(sorted(CapabilitySideEffectV1, key=lambda value: value.value)),
        )
        for member in roster.members
    )
    return ExecutionAuthorityV1.create(
        authority_id="execution-authority.generic-agent-eval",
        authority_version=1,
        predecessor_authority_ref=None,
        team_id=roster.team_id,
        team_incarnation_id="team-incarnation-001",
        roster_ref=roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        permission_policy_ref=policy.to_ref(),
        grants=grants,
        max_model_requests=20,
        max_model_tokens=200_000,
        max_cost_micro_usd=20_000_000,
        used_model_requests=0,
        used_model_tokens=0,
        used_cost_micro_usd=0,
        audit=_audit(),
    )


def _team() -> tuple[
    TeamV1,
    TeamRosterV1,
    TeamTaskGraphV1,
    ExecutionAuthorityV1,
]:
    registration = _registration()
    roster = _roster()
    graph = TeamTaskGraphV1.create(
        graph_id="team-task-graph.generic-agent-eval",
        team_id=roster.team_id,
        revision=1,
        predecessor_graph_ref=None,
        roster_ref=roster.to_ref(),
        member_ids=tuple(member.member_id for member in roster.members),
        tasks=_tasks(),
        audit=_audit(),
    )
    authority = _authority(roster=roster, graph=graph)
    team = TeamV1.create(
        team_id=roster.team_id,
        team_incarnation_id=authority.team_incarnation_id,
        team_version=1,
        goal_ref=_ref("evaluation-requirement-spec", version="v2"),
        composition_ref=registration.composition.to_ref(),
        roster_ref=roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        authority_ref=authority.to_ref(),
        coordinator_member_id="member-coordinator",
        lifecycle=TeamLifecycleV1.ACTIVE,
        audit=_audit(),
    )
    return team, roster, graph, authority


def test_team_roster_requires_one_coordinator_and_unique_incarnations() -> None:
    coordinator, task = _members()
    roster = _roster()
    assert roster.members == (coordinator, task)

    non_coordinator = TeamMemberV1.create(
        member_id=coordinator.member_id,
        member_incarnation_id=coordinator.member_incarnation_id,
        role=coordinator.role,
        agent_profile_ref=coordinator.agent_profile_ref,
        independent_session_ref=coordinator.independent_session_ref,
        capability_definition_refs=coordinator.capability_definition_refs,
        status=coordinator.status,
        is_coordinator=False,
        audit=_audit(),
    )
    with pytest.raises(ValidationError, match="exactly one coordinator"):
        TeamRosterV1.create(
            roster_id="team-roster.invalid",
            team_id="team-invalid",
            revision=1,
            predecessor_roster_ref=None,
            members=(
                non_coordinator,
                task,
            ),
            audit=_audit(),
        )


def test_team_task_graph_enforces_dag_and_unique_write_ownership() -> None:
    graph = _graph()
    assert tuple(task.task_id for task in graph.tasks) == (
        "task-authoring",
        "task-requirement",
    )

    requirement, task = _tasks()
    cyclic_requirement = _replace_task(
        requirement,
        dependency_task_ids=("task-authoring",),
    )
    with pytest.raises(ValidationError, match="acyclic"):
        _graph((cyclic_requirement, task))

    overlapping = _replace_task(
        task,
        output_artifact_head_ids=("head-requirement",),
    )
    with pytest.raises(ValidationError, match="overlapping"):
        _graph((requirement, overlapping))

    unknown_member = _replace_task(
        task,
        assigned_member_id="member-unknown",
        replace_assignee=True,
    )
    with pytest.raises(ValidationError, match="unknown member"):
        _graph((requirement, unknown_member))


def test_team_message_separates_coordination_body_and_artifact_authority() -> None:
    team, roster, graph, _ = _team()
    coordinator, task_member = roster.members
    task = next(value for value in graph.tasks if value.task_id == "task-authoring")
    message = TeamMessageV1.create(
        message_id="team-message-001",
        team_ref=team.to_ref(),
        sender_member_ref=coordinator.to_ref(),
        audience=TeamMessageAudienceV1.DIRECT,
        recipient_member_refs=(task_member.to_ref(),),
        message_kind=TeamMessageKindV1.REVIEW_REQUEST,
        task_ref=task.to_ref(),
        message_body_ref=_ref("team-message-body", "review-request"),
        artifact_envelope_refs=(_ref("artifact-envelope", "task-candidate"),),
        reply_to_message_ref=None,
        audit=_audit(),
    )
    assert message.artifact_envelope_refs[0].object_type == "artifact-envelope"
    assert "authority_ref" not in type(message).model_fields

    with pytest.raises(ValidationError, match="artifact envelopes"):
        TeamMessageV1.create(
            message_id="team-message-invalid",
            team_ref=team.to_ref(),
            sender_member_ref=coordinator.to_ref(),
            audience=TeamMessageAudienceV1.DIRECT,
            recipient_member_refs=(task_member.to_ref(),),
            message_kind=TeamMessageKindV1.REVIEW_REQUEST,
            task_ref=task.to_ref(),
            message_body_ref=_ref("team-message-body", "review-request"),
            artifact_envelope_refs=(_ref("interaction-decision", "forged"),),
            reply_to_message_ref=None,
            audit=_audit(),
        )

    unknown = message.model_dump(mode="python")
    unknown["grant_refs"] = (_ref("execution-authority"),)
    with pytest.raises(ValidationError):
        TeamMessageV1.model_validate(unknown)


def test_claim_lease_subscription_conflict_projection_and_checkpoint_are_typed() -> None:
    team, roster, graph, authority = _team()
    coordinator, task_member = roster.members
    task = next(value for value in graph.tasks if value.task_id == "task-authoring")
    claim = TeamTaskClaimV1.create(
        claim_id="team-task-claim-001",
        team_ref=team.to_ref(),
        task_ref=task.to_ref(),
        member_ref=task_member.to_ref(),
        graph_revision=graph.revision,
        claimed_at=datetime(2026, 8, 17, tzinfo=UTC),
        audit=_audit(),
    )
    lease = TeamTaskLeaseV1.create(
        lease_id="team-task-lease-001",
        claim_ref=claim.to_ref(),
        fencing_token=1,
        acquired_at=datetime(2026, 8, 17, tzinfo=UTC),
        expires_at=datetime(2026, 8, 17, tzinfo=UTC) + timedelta(minutes=5),
        audit=_audit(),
    )
    subscription = ArtifactSubscriptionV1.create(
        subscription_id="artifact-subscription-001",
        team_ref=team.to_ref(),
        member_ref=task_member.to_ref(),
        task_ids=(task.task_id,),
        artifact_head_ids=("head-task",),
        semantic_roles=("task-candidate",),
        cursor=0,
        audit=_audit(),
    )
    conflict = TeamConflictV1.create(
        conflict_id="team-conflict-001",
        team_ref=team.to_ref(),
        task_ref=task.to_ref(),
        artifact_envelope_refs=(
            _ref("artifact-envelope", "candidate-a"),
            _ref("artifact-envelope", "candidate-b"),
        ),
        status=TeamConflictStatusV1.OPEN,
        resolution_artifact_ref=None,
        resolved_by_member_ref=None,
        audit=_audit(),
    )
    projection = MemberContextProjectionV1.create(
        projection_id="member-context-projection-001",
        team_ref=team.to_ref(),
        member_ref=task_member.to_ref(),
        authority_ref=authority.to_ref(),
        task_refs=(task.to_ref(),),
        artifact_envelope_refs=(_ref("artifact-envelope", "task-candidate"),),
        message_refs=(),
        subscription_refs=(subscription.to_ref(),),
        acceptance_check_refs=task.acceptance_check_refs,
        audit=_audit(),
    )
    checkpoint = TeamCheckpointV1.create(
        checkpoint_id="team-checkpoint-001",
        team_ref=team.to_ref(),
        roster_ref=roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        authority_ref=authority.to_ref(),
        artifact_head_refs=(),
        subscription_refs=(subscription.to_ref(),),
        member_session_refs=tuple(member.independent_session_ref for member in roster.members),
        message_cursor=0,
        audit=_audit(),
    )

    assert lease.claim_ref == claim.to_ref()
    assert conflict.status is TeamConflictStatusV1.OPEN
    assert projection.authority_ref == authority.to_ref()
    assert checkpoint.member_session_refs == sorted_refs(checkpoint.member_session_refs)
    assert coordinator.is_coordinator is True

    with pytest.raises(ValidationError, match="expire after"):
        TeamTaskLeaseV1.create(
            lease_id="team-task-lease-invalid",
            claim_ref=claim.to_ref(),
            fencing_token=2,
            acquired_at=datetime(2026, 8, 17, tzinfo=UTC),
            expires_at=datetime(2026, 8, 17, tzinfo=UTC),
            audit=_audit(),
        )
