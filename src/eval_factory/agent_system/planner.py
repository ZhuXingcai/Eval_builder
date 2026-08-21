from __future__ import annotations

from eval_factory.agent_system.registry import AgentRegistry, AgentRegistryError
from eval_factory.contracts.agent_system_v2 import (
    CompiledDatasetBuildPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit


class DatasetBuildPlanCompilerError(RuntimeError):
    pass


class DatasetBuildPlanCompiler:
    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def compile(
        self,
        *,
        plan: DatasetBuildPlanV2,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledDatasetBuildPlanV2:
        canonical_plan = DatasetBuildPlanV2.model_validate_json(plan.canonical_json())
        canonical_policy = FactoryRunPolicyV2.model_validate_json(policy.canonical_json())
        if not {task.task_kind for task in canonical_plan.tasks}.issubset(
            canonical_policy.allowed_task_kinds
        ):
            raise DatasetBuildPlanCompilerError("plan contains a task kind outside run policy")
        if (
            canonical_plan.total_model_requests > canonical_policy.max_model_requests
            or canonical_plan.total_model_tokens > canonical_policy.max_model_tokens
            or canonical_plan.total_cost_micro_usd > canonical_policy.max_cost_micro_usd
        ):
            raise DatasetBuildPlanCompilerError("plan exceeds Factory run budget")
        if any(task.max_attempts > canonical_policy.max_agent_attempts for task in canonical_plan.tasks):
            raise DatasetBuildPlanCompilerError("plan task exceeds attempt policy")
        by_key = {task.task_key: task for task in canonical_plan.tasks}
        for task in canonical_plan.tasks:
            self._validate_task(task)
            for dependency_key in task.dependency_task_keys:
                dependency = by_key[dependency_key]
                if not set(dependency.output_object_types) & set(task.input_object_types):
                    raise DatasetBuildPlanCompilerError(
                        "plan dependency output and input schemas do not connect"
                    )
        topological = _topological_order(canonical_plan.tasks)
        return CompiledDatasetBuildPlanV2.create(
            compiled_plan_id=(
                "compiled-dataset-build-plan://"
                f"{canonical_plan.plan_id.rsplit('://', 1)[-1]}/v{canonical_plan.plan_version}"
            ),
            source_plan_ref=canonical_plan.to_ref(),
            policy_ref=canonical_policy.to_ref(),
            tasks=canonical_plan.tasks,
            topological_task_keys=topological,
            audit=audit,
        )

    def _validate_task(self, task: DatasetBuildPlanTaskV2) -> None:
        try:
            definition = self.registry.resolve(task.agent_role, task.task_kind)
            capabilities = self.registry.capabilities_for(definition)
        except AgentRegistryError as exc:
            raise DatasetBuildPlanCompilerError("plan Agent resolution failed") from exc
        capability_ids = {capability.capability_id for capability in capabilities}
        if not set(task.required_capability_ids).issubset(capability_ids):
            raise DatasetBuildPlanCompilerError("plan requires an unsupported Agent capability")
        supported_inputs = {value for capability in capabilities for value in capability.input_object_types}
        supported_outputs = {value for capability in capabilities for value in capability.output_object_types}
        if not set(task.input_object_types).issubset(supported_inputs):
            raise DatasetBuildPlanCompilerError("plan Agent input schema is incompatible")
        if not set(task.output_object_types).issubset(supported_outputs):
            raise DatasetBuildPlanCompilerError("plan Agent output schema is incompatible")
        if (
            task.max_attempts > definition.max_attempts
            or task.max_model_requests > definition.max_model_requests
            or task.max_model_tokens > definition.max_model_tokens
            or task.max_cost_micro_usd > definition.max_cost_micro_usd
        ):
            raise DatasetBuildPlanCompilerError("plan task exceeds Agent definition budget")


def _topological_order(
    tasks: tuple[DatasetBuildPlanTaskV2, ...],
) -> tuple[str, ...]:
    remaining = {task.task_key: set(task.dependency_task_keys) for task in tasks}
    result: list[str] = []
    while remaining:
        ready = sorted(key for key, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise DatasetBuildPlanCompilerError("plan task graph contains a cycle")
        for key in ready:
            result.append(key)
            del remaining[key]
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return tuple(result)


__all__ = ["DatasetBuildPlanCompiler", "DatasetBuildPlanCompilerError"]
