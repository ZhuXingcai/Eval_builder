from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness import (
    ArtifactEnvelopeV1,
    ArtifactModalityV1,
    CapabilityCallV1,
    CapabilityInvocationOutcomeV1,
    CapabilityProviderExecution,
    CapabilityRuntimeRegistry,
    CapabilitySideEffectV1,
    ExecutionAuthorityV1,
    HarnessCapabilityRuntime,
    MemberExecutionGrantV1,
    StaticPackRegistrationV1,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs import build_generic_agent_trace_pack
from eval_factory.packs.generic_agent_trace.capabilities import (
    AttachmentQualityCapabilityProvider,
    AttachmentReconstructionCapabilityProvider,
    BatchQualityCapabilityProvider,
    CriteriaRubricCapabilityProvider,
    DeliveryCapabilityProvider,
    GradingDesignCapabilityProvider,
    PlanReviewCapabilityProvider,
    RequirementPlanningCapabilityProvider,
    TaskAuthoringCapabilityProvider,
    TraceIngestionCapabilityProvider,
)
from eval_factory.team import (
    TeamLifecycleV1,
    TeamMemberStatusV1,
    TeamMemberV1,
    TeamRosterV1,
    TeamTaskGraphV1,
    TeamTaskStatusV1,
    TeamTaskV1,
    TeamV1,
)

HASH = "a" * 64
PROVIDERS = (
    RequirementPlanningCapabilityProvider,
    TraceIngestionCapabilityProvider,
    TaskAuthoringCapabilityProvider,
    AttachmentReconstructionCapabilityProvider,
    CriteriaRubricCapabilityProvider,
    GradingDesignCapabilityProvider,
    AttachmentQualityCapabilityProvider,
    BatchQualityCapabilityProvider,
    PlanReviewCapabilityProvider,
    DeliveryCapabilityProvider,
)


def audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
        created_by="team-runtime-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage3",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


@dataclass(frozen=True, slots=True)
class TeamFixture:
    registration: StaticPackRegistrationV1
    team: TeamV1
    roster: TeamRosterV1
    graph: TeamTaskGraphV1
    authority: ExecutionAuthorityV1


@dataclass
class StaticCapabilityProvider:
    provider_binding_ref: ObjectRef
    implementation_id: str
    request_model: type[BaseModel]
    request_schema_ref: ObjectRef
    result_model: type[BaseModel]
    result_schema_ref: ObjectRef
    capability_id: str
    calls: int = 0
    results: dict[str, CapabilityProviderExecution] | None = None

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        del request, audit
        if self.results is None:
            self.results = {}
        operation_key = call.context.idempotency_key
        prior = self.results.get(operation_key)
        if prior is not None:
            return prior
        self.calls += 1
        result_ref = ref(
            "capability-owner-result",
            f"{self.capability_id}.call-{self.calls}",
            version="v2",
        )
        result = CapabilityProviderExecution(
            outcome=CapabilityInvocationOutcomeV1.SUCCEEDED,
            canonical_result_ref=result_ref,
            content_ref=result_ref,
        )
        self.results[operation_key] = result
        return result


def team_fixture(
    session_refs: dict[str, ObjectRef] | None = None,
) -> TeamFixture:
    registration = build_generic_agent_trace_pack(audit=audit())
    profiles = {profile.role: profile for profile in registration.agent_profiles}
    roles = ("coordinator", "requirement", "trace", "task", "quality")
    members = tuple(
        TeamMemberV1.create(
            member_id=f"member-{role}",
            member_incarnation_id=f"member-{role}-incarnation-001",
            role=role,
            agent_profile_ref=profiles[role].to_ref(),
            independent_session_ref=(
                session_refs[role] if session_refs is not None else ref("harness-session", role)
            ),
            capability_definition_refs=profiles[role].capability_definition_refs,
            status=TeamMemberStatusV1.READY,
            is_coordinator=role == "coordinator",
            audit=audit(),
        )
        for role in roles
    )
    roster = TeamRosterV1.create(
        roster_id="roster-stage3",
        team_id="team-stage3",
        revision=1,
        predecessor_roster_ref=None,
        members=members,
        audit=audit(),
    )
    capabilities = {value.capability_id: value for value in registration.capability_definitions}

    def task(
        task_id: str,
        role: str,
        capability_id: str,
        dependencies: tuple[str, ...],
        inputs: tuple[str, ...],
        output: str,
        status: TeamTaskStatusV1,
    ) -> TeamTaskV1:
        return TeamTaskV1.create(
            task_id=task_id,
            task_kind=role,
            dependency_task_ids=dependencies,
            assigned_member_id=f"member-{role}",
            capability_definition_ref=capabilities[capability_id].to_ref(),
            input_artifact_head_ids=inputs,
            output_artifact_head_ids=(output,),
            acceptance_check_refs=(ref("acceptance-check", task_id),),
            status=status,
            max_attempts=2,
            max_model_requests=2,
            max_model_tokens=20_000,
            max_cost_micro_usd=2_000_000,
            audit=audit(),
        )

    tasks = (
        task(
            "task-coordinate",
            "coordinator",
            "capability.plan-review",
            (),
            ("head-plan",),
            "head-control",
            TeamTaskStatusV1.COMPLETED,
        ),
        task(
            "task-requirement",
            "requirement",
            "capability.requirement-planning",
            (),
            ("head-goal",),
            "head-plan",
            TeamTaskStatusV1.READY,
        ),
        task(
            "task-trace",
            "trace",
            "capability.trace-ingestion",
            ("task-requirement",),
            ("head-plan",),
            "head-trace",
            TeamTaskStatusV1.PENDING,
        ),
        task(
            "task-author",
            "task",
            "capability.task-authoring",
            ("task-requirement", "task-trace"),
            ("head-trace",),
            "head-task",
            TeamTaskStatusV1.PENDING,
        ),
        task(
            "task-quality",
            "quality",
            "capability.quality-review",
            ("task-author",),
            ("head-attachment", "head-task"),
            "head-quality",
            TeamTaskStatusV1.PENDING,
        ),
    )
    graph = TeamTaskGraphV1.create(
        graph_id="graph-stage3",
        team_id=roster.team_id,
        revision=1,
        predecessor_graph_ref=None,
        roster_ref=roster.to_ref(),
        member_ids=tuple(member.member_id for member in members),
        tasks=tasks,
        audit=audit(),
    )
    providers = {value.capability_definition_ref: value for value in registration.provider_bindings}
    grants = tuple(
        MemberExecutionGrantV1(
            member_id=member.member_id,
            principal_ref=ref("principal", member.member_id),
            capability_definition_refs=member.capability_definition_refs,
            provider_binding_refs=sorted_refs(
                providers[capability].to_ref() for capability in member.capability_definition_refs
            ),
            task_ids=tuple(
                sorted(task.task_id for task in tasks if task.assigned_member_id == member.member_id),
            ),
            data_scope_refs=(),
            data_purposes=("evaluation-data-production",),
            data_classifications=("INTERNAL",),
            allowed_side_effects=tuple(
                sorted(CapabilitySideEffectV1, key=lambda value: value.value),
            ),
        )
        for member in members
    )
    authority = ExecutionAuthorityV1.create(
        authority_id="authority-stage3",
        authority_version=1,
        predecessor_authority_ref=None,
        team_id=roster.team_id,
        team_incarnation_id="team-stage3-incarnation-001",
        roster_ref=roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        permission_policy_ref=registration.permission_policies[0].to_ref(),
        grants=grants,
        max_model_requests=20,
        max_model_tokens=200_000,
        max_cost_micro_usd=20_000_000,
        used_model_requests=0,
        used_model_tokens=0,
        used_cost_micro_usd=0,
        audit=audit(),
    )
    team = TeamV1.create(
        team_id=roster.team_id,
        team_incarnation_id=authority.team_incarnation_id,
        team_version=1,
        goal_ref=ref("evaluation-requirement-spec", "goal", version="v2"),
        composition_ref=registration.composition.to_ref(),
        roster_ref=roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        authority_ref=authority.to_ref(),
        coordinator_member_id="member-coordinator",
        lifecycle=TeamLifecycleV1.ACTIVE,
        audit=audit(),
    )
    return TeamFixture(registration, team, roster, graph, authority)


def capability_runtime(
    fixture: TeamFixture,
) -> tuple[
    CapabilityRuntimeRegistry,
    HarnessCapabilityRuntime,
    dict[str, StaticCapabilityProvider],
]:
    registration = fixture.registration
    definitions = {value.capability_id: value for value in registration.capability_definitions}
    bindings = {value.capability_definition_ref: value for value in registration.provider_bindings}
    providers = {}
    for provider_type in PROVIDERS:
        definition = definitions[provider_type.CAPABILITY_ID]
        binding = bindings[definition.to_ref()]
        providers[provider_type.CAPABILITY_ID] = StaticCapabilityProvider(
            provider_binding_ref=binding.to_ref(),
            implementation_id=binding.implementation_id,
            request_model=provider_type.REQUEST_MODEL,
            request_schema_ref=definition.request_schema_ref,
            result_model=provider_type.RESULT_MODEL,
            result_schema_ref=definition.result_schema_ref,
            capability_id=provider_type.CAPABILITY_ID,
        )
    registry = CapabilityRuntimeRegistry(
        registration=registration,
        providers=tuple(providers.values()),
    )
    return registry, HarnessCapabilityRuntime(registry), providers


def seed_envelope(
    fixture: TeamFixture,
    *,
    artifact_id: str,
    semantic_role: str,
    producer_capability_id: str,
) -> ArtifactEnvelopeV1:
    definition = next(
        value
        for value in fixture.registration.capability_definitions
        if value.capability_id == producer_capability_id
    )
    subject = ref("team-input", artifact_id, version="v2")
    return ArtifactEnvelopeV1.create(
        artifact_id=artifact_id,
        subject_ref=subject,
        schema_ref=definition.request_schema_ref,
        content_ref=subject,
        media_type="application/json",
        modality=ArtifactModalityV1.DOCUMENT,
        domain_tags=("mechanism-fixture",),
        semantic_role=semantic_role,
        purpose="evaluation-data-production",
        classification="INTERNAL",
        lineage_refs=(),
        producer_capability_ref=definition.to_ref(),
        producer_task_ref=None,
        validation_refs=(),
        revision=1,
        predecessor_envelope_ref=None,
        audit=audit(),
    )
