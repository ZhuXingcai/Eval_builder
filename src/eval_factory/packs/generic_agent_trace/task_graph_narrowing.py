from __future__ import annotations

from dataclasses import dataclass

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness import (
    ExecutionAuthorityV1,
    MemberExecutionGrantV1,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterialization,
    GenericAgentTaskInstance,
    GenericAgentTaskScopeV1,
)
from eval_factory.team import (
    TeamSnapshot,
    TeamTaskGraphV1,
    TeamTaskV1,
    TeamV1,
)


@dataclass(frozen=True, slots=True)
class GenericAgentTaskGraphNarrowing:
    snapshot: TeamSnapshot
    task_instances: tuple[GenericAgentTaskInstance, ...]
    candidate_source_refs: tuple[ObjectRef, ...]
    removed_task_ids: tuple[str, ...]
    changed: bool


class GenericAgentTaskGraphNarrowingError(RuntimeError):
    pass


class GenericAgentTaskGraphNarrower:
    """Narrows the superset graph after every Trace disposition is durable."""

    _CANDIDATE_FAN_IN_CAPABILITIES = frozenset(
        {
            "capability.task-authoring",
            "capability.quality-review",
            "capability.criteria-rubric",
            "capability.grading-design",
        }
    )

    def narrow(
        self,
        *,
        snapshot: TeamSnapshot,
        materialization: GenericAgentTaskGraphMaterialization,
        candidate_source_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> GenericAgentTaskGraphNarrowing:
        candidates = sorted_refs(candidate_source_refs)
        all_sources = sorted_refs(
            instance.scope_ref
            for instance in materialization.task_instances
            if (instance.capability_id == "capability.trace-ingestion" and instance.scope_ref is not None)
        )
        if candidates != candidate_source_refs or not set(candidates).issubset(all_sources):
            raise GenericAgentTaskGraphNarrowingError(
                "candidate source partition is not a sorted subset",
            )
        baseline = materialization.authority
        if (
            snapshot.team.team_id != baseline.team.team_id
            or snapshot.team.team_incarnation_id != baseline.team.team_incarnation_id
            or snapshot.roster.to_ref() != baseline.roster.to_ref()
            or snapshot.team.composition_ref != baseline.team.composition_ref
        ):
            raise GenericAgentTaskGraphNarrowingError(
                "Team narrowing source differs from materialized authority",
            )
        candidate_set = set(candidates)
        retained_instances = tuple(
            instance
            for instance in materialization.task_instances
            if (
                instance.scope is GenericAgentTaskScopeV1.DATASET
                or instance.capability_id == "capability.trace-ingestion"
                or instance.scope_ref in candidate_set
            )
        )
        retained_ids = {instance.task_id for instance in retained_instances}
        current_by_id = {task.task_id: task for task in snapshot.graph.tasks}
        baseline_by_id = {task.task_id: task for task in baseline.graph.tasks}
        if not set(current_by_id).issubset(baseline_by_id):
            raise GenericAgentTaskGraphNarrowingError(
                "current Team graph contains tasks outside the superset",
            )
        trace_task_ids = tuple(
            sorted(
                instance.task_id
                for instance in retained_instances
                if instance.capability_id == "capability.trace-ingestion"
            )
        )
        fan_in_task_ids = tuple(
            sorted(
                instance.task_id
                for instance in retained_instances
                if instance.capability_id in self._CANDIDATE_FAN_IN_CAPABILITIES
            )
        )
        target_tasks = []
        for task_id in sorted(retained_ids):
            source_task = current_by_id.get(
                task_id,
                baseline_by_id[task_id],
            )
            instance = next(value for value in retained_instances if value.task_id == task_id)
            if instance.capability_id == "capability.batch-quality":
                source_task = self._with_dependencies(
                    source_task,
                    dependency_task_ids=(
                        *trace_task_ids,
                        *fan_in_task_ids,
                    ),
                    audit=audit,
                )
            target_tasks.append(source_task)
        target_task_tuple = tuple(target_tasks)
        removed_task_ids = tuple(sorted(set(baseline_by_id) - retained_ids))
        if tuple(snapshot.graph.tasks) == target_task_tuple and all(
            set(grant.task_ids).issubset(retained_ids) for grant in snapshot.authority.grants
        ):
            return GenericAgentTaskGraphNarrowing(
                snapshot=snapshot,
                task_instances=retained_instances,
                candidate_source_refs=candidates,
                removed_task_ids=removed_task_ids,
                changed=False,
            )
        graph = TeamTaskGraphV1.create(
            graph_id=snapshot.graph.graph_id,
            team_id=snapshot.team.team_id,
            revision=snapshot.graph.revision + 1,
            predecessor_graph_ref=snapshot.graph.to_ref(),
            roster_ref=snapshot.roster.to_ref(),
            member_ids=snapshot.graph.member_ids,
            tasks=target_task_tuple,
            audit=audit,
        )
        grants = tuple(
            MemberExecutionGrantV1(
                member_id=grant.member_id,
                principal_ref=grant.principal_ref,
                capability_definition_refs=(grant.capability_definition_refs),
                provider_binding_refs=grant.provider_binding_refs,
                task_ids=tuple(task_id for task_id in grant.task_ids if task_id in retained_ids),
                data_scope_refs=grant.data_scope_refs,
                data_purposes=grant.data_purposes,
                data_classifications=grant.data_classifications,
                allowed_side_effects=grant.allowed_side_effects,
            )
            for grant in snapshot.authority.grants
        )
        authority = ExecutionAuthorityV1.create(
            authority_id=snapshot.authority.authority_id,
            authority_version=(snapshot.authority.authority_version + 1),
            predecessor_authority_ref=snapshot.authority.to_ref(),
            team_id=snapshot.team.team_id,
            team_incarnation_id=snapshot.team.team_incarnation_id,
            roster_ref=snapshot.roster.to_ref(),
            task_graph_ref=graph.to_ref(),
            permission_policy_ref=(snapshot.authority.permission_policy_ref),
            grants=grants,
            max_model_requests=snapshot.authority.max_model_requests,
            max_model_tokens=snapshot.authority.max_model_tokens,
            max_cost_micro_usd=(snapshot.authority.max_cost_micro_usd),
            used_model_requests=snapshot.authority.used_model_requests,
            used_model_tokens=snapshot.authority.used_model_tokens,
            used_cost_micro_usd=(snapshot.authority.used_cost_micro_usd),
            audit=audit,
        )
        team = TeamV1.create(
            team_id=snapshot.team.team_id,
            team_incarnation_id=snapshot.team.team_incarnation_id,
            team_version=snapshot.team.team_version + 1,
            goal_ref=snapshot.team.goal_ref,
            composition_ref=snapshot.team.composition_ref,
            roster_ref=snapshot.roster.to_ref(),
            task_graph_ref=graph.to_ref(),
            authority_ref=authority.to_ref(),
            coordinator_member_id=(snapshot.team.coordinator_member_id),
            lifecycle=snapshot.team.lifecycle,
            audit=audit,
        )
        return GenericAgentTaskGraphNarrowing(
            snapshot=TeamSnapshot(
                team=team,
                roster=snapshot.roster,
                graph=graph,
                authority=authority,
            ),
            task_instances=retained_instances,
            candidate_source_refs=candidates,
            removed_task_ids=removed_task_ids,
            changed=True,
        )

    @staticmethod
    def _with_dependencies(
        task: TeamTaskV1,
        *,
        dependency_task_ids: tuple[str, ...],
        audit: ContractAudit,
    ) -> TeamTaskV1:
        return TeamTaskV1.create(
            task_id=task.task_id,
            task_kind=task.task_kind,
            dependency_task_ids=dependency_task_ids,
            assigned_member_id=task.assigned_member_id,
            capability_definition_ref=(task.capability_definition_ref),
            input_artifact_head_ids=task.input_artifact_head_ids,
            output_artifact_head_ids=task.output_artifact_head_ids,
            acceptance_check_refs=task.acceptance_check_refs,
            status=task.status,
            max_attempts=task.max_attempts,
            max_model_requests=task.max_model_requests,
            max_model_tokens=task.max_model_tokens,
            max_cost_micro_usd=task.max_cost_micro_usd,
            audit=audit,
        )


def generic_agent_graph_is_narrowing_successor(
    *,
    source: TeamTaskGraphV1,
    current: TeamTaskGraphV1,
) -> bool:
    if (
        current.graph_id != source.graph_id
        or current.team_id != source.team_id
        or current.revision < source.revision
        or current.roster_ref != source.roster_ref
        or current.member_ids != source.member_ids
    ):
        return False
    source_tasks = {task.task_id: task for task in source.tasks}
    current_tasks = {task.task_id: task for task in current.tasks}
    if not current_tasks or not set(current_tasks).issubset(
        source_tasks,
    ):
        return False
    for task_id, task in current_tasks.items():
        baseline = source_tasks[task_id]
        if (
            task.task_kind != baseline.task_kind
            or task.assigned_member_id != baseline.assigned_member_id
            or task.capability_definition_ref != baseline.capability_definition_ref
            or not set(task.dependency_task_ids).issubset(
                baseline.dependency_task_ids,
            )
            or not set(task.input_artifact_head_ids).issubset(
                baseline.input_artifact_head_ids,
            )
            or task.output_artifact_head_ids != baseline.output_artifact_head_ids
            or task.acceptance_check_refs != baseline.acceptance_check_refs
            or task.status != baseline.status
            or task.max_attempts != baseline.max_attempts
            or task.max_model_requests != baseline.max_model_requests
            or task.max_model_tokens != baseline.max_model_tokens
            or task.max_cost_micro_usd != baseline.max_cost_micro_usd
        ):
            return False
    return True


def generic_agent_authority_is_narrowing_successor(
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
            or not set(current_grant.task_ids).issubset(
                source_grant.task_ids,
            )
            or current_grant.data_purposes != source_grant.data_purposes
            or current_grant.data_classifications != source_grant.data_classifications
            or current_grant.allowed_side_effects != source_grant.allowed_side_effects
            or not set(source_grant.data_scope_refs).issubset(
                current_grant.data_scope_refs,
            )
        ):
            return False
    return True


__all__ = [
    "GenericAgentTaskGraphNarrower",
    "GenericAgentTaskGraphNarrowing",
    "GenericAgentTaskGraphNarrowingError",
    "generic_agent_authority_is_narrowing_successor",
    "generic_agent_graph_is_narrowing_successor",
]
