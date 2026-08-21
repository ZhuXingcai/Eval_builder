from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from eval_factory.blueprints import EvaluationBlueprintV1
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness import (
    CapabilitySideEffectV1,
    ExecutionAuthorityV1,
    MemberExecutionGrantV1,
    StaticPackRegistrationV1,
)
from eval_factory.harness.contracts import (
    sorted_refs,
    static_object_ref,
)
from eval_factory.packs.generic_agent_trace.graph_narrowing_rules import (
    generic_agent_authority_is_narrowing_successor,
    generic_agent_graph_is_narrowing_successor,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION,
    GENERIC_AGENT_TRACE_PACK_ID,
    GENERIC_AGENT_TRACE_PACK_VERSION,
)
from eval_factory.team import (
    TeamCapabilityRunner,
    TeamLifecycleV1,
    TeamMemberStatusV1,
    TeamMemberV1,
    TeamRosterV1,
    TeamSnapshot,
    TeamStore,
    TeamTaskCompletion,
    TeamTaskGraphV1,
    TeamTaskStatusV1,
    TeamTaskV1,
    TeamTaskWork,
    TeamV1,
)


class GenericAgentFactoryWorkflowError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GenericAgentTeamTaskSpec:
    capability_id: str
    dependency_capability_ids: tuple[str, ...]
    input_schema_refs: tuple[ObjectRef, ...]
    output_schema_refs: tuple[ObjectRef, ...]


@dataclass(frozen=True, slots=True)
class GenericAgentDispatchFacts:
    attempted_task_refs: tuple[ObjectRef, ...]
    completed_result_refs: tuple[ObjectRef, ...]
    published_head_refs: tuple[ObjectRef, ...]


@dataclass(frozen=True, slots=True)
class GenericAgentTeamAuthority:
    team: TeamV1
    roster: TeamRosterV1
    graph: TeamTaskGraphV1
    authority: ExecutionAuthorityV1

    def snapshot(self) -> TeamSnapshot:
        return TeamSnapshot(
            team=self.team,
            roster=self.roster,
            graph=self.graph,
            authority=self.authority,
        )


class GenericAgentCapabilityRequestSource(Protocol):
    def build(
        self,
        *,
        snapshot: TeamSnapshot,
        work: TeamTaskWork,
        spec: GenericAgentTeamTaskSpec,
    ) -> BaseModel: ...


class GenericAgentCapabilityRequestPreparation(Protocol):
    def prepare(
        self,
        *,
        snapshot: TeamSnapshot,
        work: TeamTaskWork,
        spec: GenericAgentTeamTaskSpec,
        audit: ContractAudit,
    ) -> None: ...


class GenericAgentTaskGraphRefinement(Protocol):
    def refine(
        self,
        team_id: str,
        *,
        audit: ContractAudit,
    ) -> object | None: ...


class GenericAgentFactoryWorkflow:
    _CAPABILITY_IDS = frozenset(
        {
            "capability.requirement-planning",
            "capability.trace-ingestion",
            "capability.task-authoring",
            "capability.attachment-reconstruction",
            "capability.criteria-rubric",
            "capability.grading-design",
            "capability.quality-review",
            "capability.batch-quality",
            "capability.plan-review",
            "capability.delivery",
        }
    )

    def __init__(
        self,
        *,
        blueprint: EvaluationBlueprintV1,
        store: TeamStore,
        runner: TeamCapabilityRunner,
        request_source: GenericAgentCapabilityRequestSource,
        request_preparation: (GenericAgentCapabilityRequestPreparation | None) = None,
        graph_refinement: GenericAgentTaskGraphRefinement | None = None,
    ) -> None:
        self.blueprint = blueprint
        self.store = store
        self.runner = runner
        self.request_source = request_source
        self.request_preparation = request_preparation
        self.graph_refinement = graph_refinement
        registration = runner.registry.registration
        if (
            registration.manifest.pack_id != GENERIC_AGENT_TRACE_PACK_ID
            or registration.manifest.pack_version
            not in {
                GENERIC_AGENT_TRACE_PACK_VERSION,
                GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION,
            }
            or registration.manifest.to_ref() not in blueprint.pack_manifest_refs
            or registration.composition.to_ref() != blueprint.composition_ref
        ):
            raise GenericAgentFactoryWorkflowError(
                "factory workflow Pack authority differs from Blueprint",
            )
        self.task_specs = compile_generic_agent_team_task_specs(
            blueprint,
        )

    def validate_team(
        self,
        team_id: str,
    ) -> TeamSnapshot:
        snapshot = self.store.get_snapshot(team_id)
        blueprint_snapshot = self.store.get_snapshot_at(
            self.blueprint.team_ref,
        )
        if (
            blueprint_snapshot.team.to_ref() != self.blueprint.team_ref
            or blueprint_snapshot.roster.to_ref() != self.blueprint.roster_ref
            or blueprint_snapshot.graph.to_ref() != self.blueprint.task_graph_ref
            or blueprint_snapshot.authority.to_ref() != self.blueprint.authority_ref
            or snapshot.team.team_id != blueprint_snapshot.team.team_id
            or snapshot.team.team_incarnation_id != blueprint_snapshot.team.team_incarnation_id
            or snapshot.roster.to_ref() != blueprint_snapshot.roster.to_ref()
            or not generic_agent_graph_is_narrowing_successor(
                source=blueprint_snapshot.graph,
                current=snapshot.graph,
            )
            or not generic_agent_authority_is_narrowing_successor(
                source=blueprint_snapshot.authority,
                current=snapshot.authority,
            )
            or snapshot.authority.task_graph_ref != snapshot.graph.to_ref()
            or snapshot.team.composition_ref != self.blueprint.composition_ref
        ):
            raise GenericAgentFactoryWorkflowError(
                "Team authority differs from fixed Blueprint",
            )
        registration = self.runner.registry.registration
        definitions = {value.to_ref(): value for value in registration.capability_definitions}
        works = self.store.list_task_work(team_id)
        capability_by_task = {}
        for work in works:
            definition = definitions.get(work.task.capability_definition_ref)
            if definition is None:
                raise GenericAgentFactoryWorkflowError(
                    "Team task capability is not installed",
                )
            capability_by_task[work.task.task_id] = definition.capability_id
        installed_capabilities = set(capability_by_task.values())
        if not {
            "capability.requirement-planning",
            "capability.trace-ingestion",
            "capability.batch-quality",
            "capability.plan-review",
            "capability.delivery",
        }.issubset(installed_capabilities) or not installed_capabilities.issubset(
            self._CAPABILITY_IDS,
        ):
            raise GenericAgentFactoryWorkflowError(
                "Team DAG does not contain the exact ten capabilities",
            )
        blueprint_edges = {
            (
                edge.producer_capability_id,
                edge.consumer_capability_id,
            )
            for edge in self.blueprint.data_flow_edges
        }
        specs = {value.capability_id: value for value in self.task_specs}
        ancestors = _capability_ancestors(
            works,
            capability_by_task=capability_by_task,
        )
        output_roles = {
            head_id: definitions[work.task.capability_definition_ref].output_artifact_roles[0]
            for work in works
            for head_id in work.task.output_artifact_head_ids
        }
        for work in works:
            capability_id = capability_by_task[work.task.task_id]
            direct_capabilities = {capability_by_task[task_id] for task_id in work.task.dependency_task_ids}
            if any(
                not _legal_task_instance_edge(
                    producer=producer,
                    consumer=capability_id,
                    blueprint_edges=blueprint_edges,
                )
                for producer in direct_capabilities
            ):
                raise GenericAgentFactoryWorkflowError(
                    "Team task dependency is not a legal Blueprint or review edge",
                )
            required = (
                set(
                    specs[capability_id].dependency_capability_ids,
                )
                & installed_capabilities
            )
            if not required.issubset(ancestors[work.task.task_id]):
                raise GenericAgentFactoryWorkflowError(
                    "Team task lacks a required Blueprint producer ancestor",
                )
            observed_roles = {
                output_roles.get(head_id, "evaluation-requirement")
                for head_id in work.task.input_artifact_head_ids
            }
            definition = definitions[work.task.capability_definition_ref]
            if observed_roles != set(definition.input_artifact_roles):
                raise GenericAgentFactoryWorkflowError(
                    "Team task input roles differ from the capability definition",
                )
            if any(
                head_id not in output_roles and capability_id != "capability.requirement-planning"
                for head_id in work.task.input_artifact_head_ids
            ):
                raise GenericAgentFactoryWorkflowError(
                    "non-root Team task reads an unowned Artifact Head",
                )
        return snapshot

    async def dispatch_ready(
        self,
        team_id: str,
        *,
        audit: ContractAudit,
        limit: int = 10,
        capability_ids: frozenset[str] | None = None,
    ) -> GenericAgentDispatchFacts:
        if not 1 <= limit <= 10:
            raise ValueError(
                "generic Agent dispatch limit must be between 1 and 10",
            )
        if self.graph_refinement is not None:
            self.graph_refinement.refine(
                team_id,
                audit=audit,
            )
        self.validate_team(team_id)
        definitions = {
            value.to_ref(): value.capability_id
            for value in self.runner.registry.registration.capability_definitions
        }
        specs = {value.capability_id: value for value in self.task_specs}
        selected = capability_ids or self._CAPABILITY_IDS
        if not selected or not selected.issubset(self._CAPABILITY_IDS):
            raise ValueError(
                "generic Agent dispatch capabilities must be installed",
            )
        attempted: list[ObjectRef] = []
        completed: list[ObjectRef] = []
        heads: list[ObjectRef] = []
        ready = tuple(
            value
            for value in sorted(
                self.store.ready_tasks(team_id),
                key=lambda item: item.task.task_id,
            )
            if definitions.get(value.task.capability_definition_ref) in selected
        )[:limit]
        for work in ready:
            if work.status is not TeamTaskStatusV1.READY:
                raise GenericAgentFactoryWorkflowError(
                    "dispatch frontier contains non-ready work",
                )
            capability_id = definitions.get(
                work.task.capability_definition_ref,
            )
            if capability_id is None:
                raise GenericAgentFactoryWorkflowError(
                    "ready Team capability is not installed",
                )
            current_snapshot = self.validate_team(team_id)
            self._prepare_request(
                snapshot=current_snapshot,
                work=work,
                spec=specs[capability_id],
                audit=audit,
            )
            request = self.request_source.build(
                snapshot=current_snapshot,
                work=work,
                spec=specs[capability_id],
            )
            completion = await self.runner.execute(
                team_id=team_id,
                task_id=work.task.task_id,
                request=request,
                audit=audit,
            )
            self._validate_completion(
                work=work,
                completion=completion,
            )
            attempted.append(work.task.to_ref())
            completed.append(completion.result.to_ref())
            heads.extend(value.to_ref() for value in completion.artifact_heads)
        if attempted:
            checkpoint_key = hashlib.sha256(
                "|".join(
                    value.object_sha256
                    for value in sorted_refs(
                        (*attempted, *completed, *heads),
                    )
                ).encode()
            ).hexdigest()
            self.store.create_checkpoint(
                team_id,
                audit=audit,
                idempotency_key=(f"generic-agent-dispatch.{checkpoint_key}"),
            )
        return GenericAgentDispatchFacts(
            attempted_task_refs=sorted_refs(attempted),
            completed_result_refs=sorted_refs(completed),
            published_head_refs=sorted_refs(heads),
        )

    async def dispatch_capability(
        self,
        team_id: str,
        capability_id: str,
        *,
        audit: ContractAudit,
    ) -> GenericAgentDispatchFacts:
        if capability_id not in self._CAPABILITY_IDS:
            raise ValueError(
                "generic Agent capability is not installed",
            )
        snapshot = self.validate_team(team_id)
        definitions = {
            value.to_ref(): value.capability_id
            for value in self.runner.registry.registration.capability_definitions
        }
        matches = tuple(
            work
            for work in self.store.list_task_work(team_id)
            if definitions.get(work.task.capability_definition_ref) == capability_id
        )
        if len(matches) != 1:
            raise GenericAgentFactoryWorkflowError(
                "Team capability task is not uniquely installed",
            )
        return await self._dispatch_work(
            team_id=team_id,
            snapshot=snapshot,
            work=matches[0],
            capability_id=capability_id,
            audit=audit,
        )

    async def dispatch_task(
        self,
        team_id: str,
        task_id: str,
        *,
        audit: ContractAudit,
    ) -> GenericAgentDispatchFacts:
        snapshot = self.validate_team(team_id)
        work = self.store.get_task_work(team_id, task_id)
        definition = next(
            (
                value
                for value in self.runner.registry.registration.capability_definitions
                if value.to_ref() == work.task.capability_definition_ref
            ),
            None,
        )
        if definition is None:
            raise GenericAgentFactoryWorkflowError(
                "Team task capability is not installed",
            )
        return await self._dispatch_work(
            team_id=team_id,
            snapshot=snapshot,
            work=work,
            capability_id=definition.capability_id,
            audit=audit,
        )

    async def _dispatch_work(
        self,
        *,
        team_id: str,
        snapshot: TeamSnapshot,
        work: TeamTaskWork,
        capability_id: str,
        audit: ContractAudit,
    ) -> GenericAgentDispatchFacts:
        if work.status is TeamTaskStatusV1.COMPLETED:
            result_ref = work.latest_event.capability_result_ref if work.latest_event is not None else None
            if result_ref is None:
                raise GenericAgentFactoryWorkflowError(
                    "completed Team capability lacks result authority",
                )
            heads_by_id = {value.head_id: value for value in self.store.current_artifact_heads(team_id)}
            try:
                head_refs = tuple(
                    heads_by_id[head_id].to_ref() for head_id in work.task.output_artifact_head_ids
                )
            except KeyError as exc:
                raise GenericAgentFactoryWorkflowError(
                    "completed Team capability lacks Artifact Head authority",
                ) from exc
            return GenericAgentDispatchFacts(
                attempted_task_refs=(),
                completed_result_refs=(result_ref,),
                published_head_refs=sorted_refs(head_refs),
            )
        if work.status not in {
            TeamTaskStatusV1.READY,
            TeamTaskStatusV1.ACTIVE,
        }:
            raise GenericAgentFactoryWorkflowError(
                "Team capability is not ready for terminal dispatch",
            )
        spec = next(value for value in self.task_specs if value.capability_id == capability_id)
        self._prepare_request(
            snapshot=snapshot,
            work=work,
            spec=spec,
            audit=audit,
        )
        request = self.request_source.build(
            snapshot=snapshot,
            work=work,
            spec=spec,
        )
        completion = await self.runner.execute(
            team_id=team_id,
            task_id=work.task.task_id,
            request=request,
            audit=audit,
        )
        self._validate_completion(
            work=work,
            completion=completion,
        )
        checkpoint_key = hashlib.sha256(
            "|".join(
                value.object_sha256
                for value in sorted_refs(
                    (
                        work.task.to_ref(),
                        completion.result.to_ref(),
                        *(value.to_ref() for value in completion.artifact_heads),
                    ),
                )
            ).encode()
        ).hexdigest()
        self.store.create_checkpoint(
            team_id,
            audit=audit,
            idempotency_key=(f"generic-agent-dispatch.{checkpoint_key}"),
        )
        return GenericAgentDispatchFacts(
            attempted_task_refs=(work.task.to_ref(),),
            completed_result_refs=(completion.result.to_ref(),),
            published_head_refs=sorted_refs(value.to_ref() for value in completion.artifact_heads),
        )

    def _prepare_request(
        self,
        *,
        snapshot: TeamSnapshot,
        work: TeamTaskWork,
        spec: GenericAgentTeamTaskSpec,
        audit: ContractAudit,
    ) -> None:
        if self.request_preparation is not None:
            self.request_preparation.prepare(
                snapshot=snapshot,
                work=work,
                spec=spec,
                audit=audit,
            )

    @staticmethod
    def _validate_completion(
        *,
        work: TeamTaskWork,
        completion: TeamTaskCompletion,
    ) -> None:
        if completion.event.task_ref != work.task.to_ref():
            raise GenericAgentFactoryWorkflowError(
                "Team completion belongs to another task",
            )
        canonical_ref = completion.result.canonical_result_ref
        if canonical_ref is None:
            if completion.artifact_heads:
                raise GenericAgentFactoryWorkflowError(
                    "non-success completion published Artifact Heads",
                )
            return
        if not completion.artifact_heads:
            raise GenericAgentFactoryWorkflowError(
                "successful completion published no Artifact Head",
            )


def compile_generic_agent_team_task_specs(
    blueprint: EvaluationBlueprintV1,
) -> tuple[GenericAgentTeamTaskSpec, ...]:
    bindings = {value.capability_id: value for value in blueprint.capability_bindings}
    if set(bindings) != GenericAgentFactoryWorkflow._CAPABILITY_IDS:
        raise GenericAgentFactoryWorkflowError(
            "Blueprint does not contain the exact ten capabilities",
        )
    dependencies: dict[str, set[str]] = {capability_id: set() for capability_id in bindings}
    for edge in blueprint.data_flow_edges:
        if edge.producer_capability_id not in bindings or edge.consumer_capability_id not in bindings:
            raise GenericAgentFactoryWorkflowError(
                "Blueprint flow references an unknown capability",
            )
        dependencies[edge.consumer_capability_id].add(
            edge.producer_capability_id,
        )
    return tuple(
        GenericAgentTeamTaskSpec(
            capability_id=capability_id,
            dependency_capability_ids=tuple(sorted(dependencies[capability_id])),
            input_schema_refs=bindings[capability_id].input_schema_refs,
            output_schema_refs=bindings[capability_id].output_schema_refs,
        )
        for capability_id in sorted(bindings)
    )


def materialize_generic_agent_team(
    *,
    registration: StaticPackRegistrationV1,
    requirement_ref: ObjectRef,
    independent_session_refs: dict[str, ObjectRef],
    team_id: str,
    team_incarnation_id: str,
    max_model_requests: int,
    max_model_tokens: int,
    max_cost_micro_usd: int,
    audit: ContractAudit,
) -> GenericAgentTeamAuthority:
    expected_roles = {
        "control",
        "coordinator",
        "quality",
        "requirement",
        "task",
        "trace",
    }
    if set(independent_session_refs) != expected_roles:
        raise GenericAgentFactoryWorkflowError(
            "generic Agent Team requires exact independent sessions",
        )
    profiles = {value.role: value for value in registration.agent_profiles}
    if set(profiles) != expected_roles - {"control"}:
        raise GenericAgentFactoryWorkflowError(
            "generic Agent Pack profiles are incomplete",
        )
    members = tuple(
        TeamMemberV1.create(
            member_id=f"{team_id}.member-{role}",
            member_incarnation_id=(f"{team_incarnation_id}.member-{role}.1"),
            role=role,
            agent_profile_ref=profiles[("coordinator" if role == "control" else role)].to_ref(),
            independent_session_ref=(independent_session_refs[role]),
            capability_definition_refs=(
                profiles[("coordinator" if role == "control" else role)].capability_definition_refs
            ),
            status=TeamMemberStatusV1.READY,
            is_coordinator=role == "coordinator",
            audit=audit,
        )
        for role in sorted(expected_roles)
    )
    roster = TeamRosterV1.create(
        roster_id=f"{team_id}.roster",
        team_id=team_id,
        revision=1,
        predecessor_roster_ref=None,
        members=members,
        audit=audit,
    )
    definitions = {value.capability_id: value for value in registration.capability_definitions}
    if set(definitions) != GenericAgentFactoryWorkflow._CAPABILITY_IDS:
        raise GenericAgentFactoryWorkflowError(
            "generic Agent Team requires exact ten capabilities",
        )
    dependencies = _generic_agent_dependencies()
    task_ids = {
        capability_id: (f"{team_id}.task-{capability_id.removeprefix('capability.')}")
        for capability_id in definitions
    }
    output_heads = {
        capability_id: (f"{team_id}.head-{capability_id.removeprefix('capability.')}")
        for capability_id in definitions
    }
    role_by_capability = {
        "capability.requirement-planning": "requirement",
        "capability.trace-ingestion": "trace",
        "capability.task-authoring": "task",
        "capability.attachment-reconstruction": "task",
        "capability.criteria-rubric": "task",
        "capability.grading-design": "task",
        "capability.quality-review": "quality",
        "capability.batch-quality": "quality",
        "capability.plan-review": "control",
        "capability.delivery": "control",
    }
    tasks = tuple(
        TeamTaskV1.create(
            task_id=task_ids[capability_id],
            task_kind=capability_id.removeprefix("capability."),
            dependency_task_ids=tuple(task_ids[value] for value in dependencies[capability_id]),
            assigned_member_id=(f"{team_id}.member-{role_by_capability[capability_id]}"),
            capability_definition_ref=definitions[capability_id].to_ref(),
            input_artifact_head_ids=(
                (f"{team_id}.head-evaluation-requirement",)
                if capability_id == "capability.requirement-planning"
                else tuple(output_heads[value] for value in dependencies[capability_id])
            ),
            output_artifact_head_ids=(output_heads[capability_id],),
            acceptance_check_refs=(
                static_object_ref(
                    object_type="acceptance-check",
                    object_id=(f"acceptance-check://generic-agent/{capability_id}"),
                    object_version="v1",
                    payload={
                        "capability_id": capability_id,
                        "pack_version": (registration.manifest.pack_version),
                    },
                ),
            ),
            status=(TeamTaskStatusV1.READY if not dependencies[capability_id] else TeamTaskStatusV1.PENDING),
            max_attempts=3,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            audit=audit,
        )
        for capability_id in sorted(definitions)
    )
    graph = TeamTaskGraphV1.create(
        graph_id=f"{team_id}.graph",
        team_id=team_id,
        revision=1,
        predecessor_graph_ref=None,
        roster_ref=roster.to_ref(),
        member_ids=tuple(value.member_id for value in members),
        tasks=tasks,
        audit=audit,
    )
    providers = {value.capability_definition_ref: value for value in registration.provider_bindings}
    grants = tuple(
        MemberExecutionGrantV1(
            member_id=member.member_id,
            principal_ref=static_object_ref(
                object_type="principal",
                object_id=(f"principal://generic-agent/{member.member_id}"),
                object_version="v1",
                payload={"member_id": member.member_id},
            ),
            capability_definition_refs=(member.capability_definition_refs),
            provider_binding_refs=sorted_refs(
                providers[reference].to_ref() for reference in (member.capability_definition_refs)
            ),
            task_ids=tuple(
                sorted(task.task_id for task in tasks if task.assigned_member_id == member.member_id)
            ),
            data_scope_refs=(),
            data_purposes=("evaluation-data-production",),
            data_classifications=(
                "INTERNAL",
                "RESTRICTED",
            ),
            allowed_side_effects=tuple(
                sorted(
                    CapabilitySideEffectV1,
                    key=lambda value: value.value,
                )
            ),
        )
        for member in members
    )
    authority = ExecutionAuthorityV1.create(
        authority_id=f"{team_id}.authority",
        authority_version=1,
        predecessor_authority_ref=None,
        team_id=team_id,
        team_incarnation_id=team_incarnation_id,
        roster_ref=roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        permission_policy_ref=(registration.permission_policies[0].to_ref()),
        grants=grants,
        max_model_requests=max_model_requests,
        max_model_tokens=max_model_tokens,
        max_cost_micro_usd=max_cost_micro_usd,
        used_model_requests=0,
        used_model_tokens=0,
        used_cost_micro_usd=0,
        audit=audit,
    )
    team = TeamV1.create(
        team_id=team_id,
        team_incarnation_id=team_incarnation_id,
        team_version=1,
        goal_ref=requirement_ref,
        composition_ref=registration.composition.to_ref(),
        roster_ref=roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        authority_ref=authority.to_ref(),
        coordinator_member_id=f"{team_id}.member-coordinator",
        lifecycle=TeamLifecycleV1.ACTIVE,
        audit=audit,
    )
    return GenericAgentTeamAuthority(
        team=team,
        roster=roster,
        graph=graph,
        authority=authority,
    )


def _generic_agent_dependencies() -> dict[str, tuple[str, ...]]:
    return {
        "capability.requirement-planning": (),
        "capability.trace-ingestion": ("capability.requirement-planning",),
        "capability.task-authoring": ("capability.trace-ingestion",),
        "capability.attachment-reconstruction": ("capability.task-authoring",),
        "capability.criteria-rubric": (
            "capability.attachment-reconstruction",
            "capability.task-authoring",
        ),
        "capability.grading-design": ("capability.criteria-rubric",),
        "capability.quality-review": (
            "capability.attachment-reconstruction",
            "capability.task-authoring",
        ),
        "capability.batch-quality": (
            "capability.criteria-rubric",
            "capability.grading-design",
            "capability.quality-review",
            "capability.task-authoring",
        ),
        "capability.plan-review": ("capability.requirement-planning",),
        "capability.delivery": (
            "capability.batch-quality",
            "capability.criteria-rubric",
            "capability.grading-design",
            "capability.plan-review",
            "capability.quality-review",
            "capability.task-authoring",
        ),
    }


def _capability_ancestors(
    works: tuple[TeamTaskWork, ...],
    *,
    capability_by_task: dict[str, str],
) -> dict[str, set[str]]:
    dependencies = {work.task.task_id: work.task.dependency_task_ids for work in works}
    memo: dict[str, set[str]] = {}

    def visit(task_id: str) -> set[str]:
        current = memo.get(task_id)
        if current is not None:
            return current
        values: set[str] = set()
        for dependency_id in dependencies[task_id]:
            values.add(capability_by_task[dependency_id])
            values.update(visit(dependency_id))
        memo[task_id] = values
        return values

    return {task_id: visit(task_id) for task_id in sorted(dependencies)}


def _legal_task_instance_edge(
    *,
    producer: str,
    consumer: str,
    blueprint_edges: set[tuple[str, str]],
) -> bool:
    if (producer, consumer) in blueprint_edges:
        return True
    if producer == "capability.plan-review":
        return consumer != "capability.requirement-planning"
    if consumer == "capability.plan-review":
        return producer in {
            "capability.requirement-planning",
            "capability.task-authoring",
            "capability.quality-review",
            "capability.criteria-rubric",
            "capability.batch-quality",
        }
    return False


def _authority_is_successor(
    *,
    source: ExecutionAuthorityV1,
    current: ExecutionAuthorityV1,
) -> bool:
    if (
        current.authority_version < source.authority_version
        or current.authority_id != source.authority_id
        or current.team_id != source.team_id
        or current.team_incarnation_id != source.team_incarnation_id
        or current.roster_ref != source.roster_ref
        or current.task_graph_ref != source.task_graph_ref
        or current.permission_policy_ref != source.permission_policy_ref
        or current.max_model_requests != source.max_model_requests
        or current.max_model_tokens != source.max_model_tokens
        or current.max_cost_micro_usd != source.max_cost_micro_usd
        or current.used_model_requests < source.used_model_requests
        or current.used_model_tokens < source.used_model_tokens
        or current.used_cost_micro_usd < source.used_cost_micro_usd
    ):
        return False
    source_grants = {value.member_id: value for value in source.grants}
    current_grants = {value.member_id: value for value in current.grants}
    if set(source_grants) != set(current_grants):
        return False
    for member_id, source_grant in source_grants.items():
        current_grant = current_grants[member_id]
        if (
            current_grant.principal_ref != source_grant.principal_ref
            or current_grant.capability_definition_refs != source_grant.capability_definition_refs
            or current_grant.provider_binding_refs != source_grant.provider_binding_refs
            or current_grant.task_ids != source_grant.task_ids
            or current_grant.data_purposes != source_grant.data_purposes
            or current_grant.data_classifications != source_grant.data_classifications
            or current_grant.allowed_side_effects != source_grant.allowed_side_effects
            or not set(
                source_grant.data_scope_refs,
            ).issubset(current_grant.data_scope_refs)
        ):
            return False
    return True


__all__ = [
    "GenericAgentCapabilityRequestPreparation",
    "GenericAgentCapabilityRequestSource",
    "GenericAgentDispatchFacts",
    "GenericAgentFactoryWorkflow",
    "GenericAgentFactoryWorkflowError",
    "GenericAgentTeamAuthority",
    "GenericAgentTeamTaskSpec",
    "compile_generic_agent_team_task_specs",
    "materialize_generic_agent_team",
]
