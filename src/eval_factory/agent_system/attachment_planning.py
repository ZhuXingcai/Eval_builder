from __future__ import annotations

from eval_factory.agent_system.registry import AgentRegistry, AgentRegistryError
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
    AttachmentWorkAssignmentV2,
    CompiledAttachmentGenerationPlanV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef

_ATTACHMENT_MOCK_TASK_KIND = "attachment-mock"


class AttachmentGenerationPlanCompilerError(RuntimeError):
    pass


class AttachmentGenerationPlanCompiler:
    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def compile(
        self,
        *,
        plan: AttachmentGenerationPlanV2,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
        max_parallel_groups: int | None = None,
    ) -> CompiledAttachmentGenerationPlanV2:
        canonical_plan = AttachmentGenerationPlanV2.model_validate_json(plan.canonical_json())
        canonical_policy = FactoryRunPolicyV2.model_validate_json(policy.canonical_json())
        self._validate_policy(
            canonical_plan,
            canonical_policy,
            max_parallel_groups=max_parallel_groups,
        )
        assignments = tuple(self._compile_assignment(work) for work in canonical_plan.works)
        topological = _topological_order(canonical_plan.works)
        return CompiledAttachmentGenerationPlanV2.create(
            compiled_plan_id=(
                "compiled-attachment-generation-plan://"
                f"{canonical_plan.plan_id.rsplit('://', 1)[-1]}"
                f"/v{canonical_plan.plan_version}"
            ),
            source_plan_ref=canonical_plan.to_ref(),
            policy_ref=canonical_policy.to_ref(),
            works=canonical_plan.works,
            assignments=assignments,
            topological_work_keys=topological,
            max_parallel_groups=canonical_plan.max_parallel_groups,
            audit=audit,
        )

    def _validate_policy(
        self,
        plan: AttachmentGenerationPlanV2,
        policy: FactoryRunPolicyV2,
        *,
        max_parallel_groups: int | None,
    ) -> None:
        if _ATTACHMENT_MOCK_TASK_KIND not in policy.allowed_task_kinds:
            raise AttachmentGenerationPlanCompilerError("attachment-mock task kind is outside run policy")
        if plan.plan_version > policy.max_plan_revisions + 1:
            raise AttachmentGenerationPlanCompilerError("attachment plan exceeds revision policy")
        if (
            plan.total_model_requests > policy.max_model_requests
            or plan.total_model_tokens > policy.max_model_tokens
            or plan.total_cost_micro_usd > policy.max_cost_micro_usd
        ):
            raise AttachmentGenerationPlanCompilerError("attachment plan exceeds Factory run budget")
        if any(work.max_attempts > policy.max_agent_attempts for work in plan.works):
            raise AttachmentGenerationPlanCompilerError("attachment work exceeds attempt policy")
        if max_parallel_groups is not None and plan.max_parallel_groups > max_parallel_groups:
            raise AttachmentGenerationPlanCompilerError("attachment plan exceeds parallel group policy")

    def _compile_assignment(
        self,
        work: AttachmentMockWorkV2,
    ) -> AttachmentWorkAssignmentV2:
        try:
            definition = self.registry.resolve(
                work.agent_role,
                _ATTACHMENT_MOCK_TASK_KIND,
            )
            capabilities = self.registry.capabilities_for(definition)
        except AgentRegistryError as exc:
            raise AttachmentGenerationPlanCompilerError("attachment Agent resolution failed") from exc

        capabilities_by_id = {capability.capability_id: capability for capability in capabilities}
        if not set(work.required_capability_ids).issubset(capabilities_by_id):
            raise AttachmentGenerationPlanCompilerError(
                "attachment work requires an unsupported Agent capability"
            )
        supported_inputs = {
            object_type for capability in capabilities for object_type in capability.input_object_types
        }
        supported_outputs = {
            object_type for capability in capabilities for object_type in capability.output_object_types
        }
        if not set(work.input_object_types).issubset(supported_inputs):
            raise AttachmentGenerationPlanCompilerError("attachment Agent input schema is incompatible")
        if not set(work.output_object_types).issubset(supported_outputs):
            raise AttachmentGenerationPlanCompilerError("attachment Agent output schema is incompatible")
        if not set(work.allowed_tool_ids).issubset(definition.tool_ids):
            raise AttachmentGenerationPlanCompilerError("attachment work widens Agent tool scope")
        if not set(work.data_purposes).issubset({definition.data_purpose}):
            raise AttachmentGenerationPlanCompilerError("attachment work widens Agent data purpose")
        if not set(work.data_classifications).issubset(definition.allowed_data_classifications):
            raise AttachmentGenerationPlanCompilerError("attachment work widens Agent data classification")
        if definition.workspace_isolated is not True:
            raise AttachmentGenerationPlanCompilerError("attachment Agent requires an isolated workspace")
        if (
            work.max_attempts > definition.max_attempts
            or work.max_model_requests > definition.max_model_requests
            or work.max_model_tokens > definition.max_model_tokens
            or work.max_cost_micro_usd > definition.max_cost_micro_usd
        ):
            raise AttachmentGenerationPlanCompilerError("attachment work exceeds Agent definition budget")

        capability_refs = tuple(
            sorted(
                (
                    capabilities_by_id[capability_id].to_ref()
                    for capability_id in work.required_capability_ids
                ),
                key=_ref_key,
            )
        )
        return AttachmentWorkAssignmentV2(
            work_key=work.work_key,
            agent_definition_ref=definition.to_ref(),
            capability_refs=capability_refs,
        )


def _topological_order(
    works: tuple[AttachmentMockWorkV2, ...],
) -> tuple[str, ...]:
    remaining = {work.work_key: set(work.dependency_work_keys) for work in works}
    result: list[str] = []
    while remaining:
        ready = sorted(key for key, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise AttachmentGenerationPlanCompilerError("attachment work graph contains a cycle")
        for key in ready:
            result.append(key)
            del remaining[key]
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return tuple(result)


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "AttachmentGenerationPlanCompiler",
    "AttachmentGenerationPlanCompilerError",
]
