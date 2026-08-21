from __future__ import annotations

from eval_factory.harness import ExecutionAuthorityV1
from eval_factory.team import TeamTaskGraphV1


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
    "generic_agent_authority_is_narrowing_successor",
    "generic_agent_graph_is_narrowing_successor",
]
