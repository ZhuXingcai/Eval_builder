from __future__ import annotations

from eval_factory.agent_system.registry import AgentRegistry, AgentRegistryError
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    CompiledCriteriaRubricPlanV2,
    CriteriaRubricPlanV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef

_CRITERIA_RUBRIC_TASK_KIND = "criteria-rubric"
_REQUIRED_INPUT_TYPES = {
    "task-draft",
    "attachment-quality-assessment",
    "solvability-assessment",
}
_REQUIRED_OUTPUT_TYPE = "criteria-rubric-result"


class CriteriaRubricPlanCompilerError(RuntimeError):
    pass


class CriteriaRubricPlanCompiler:
    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def compile(
        self,
        *,
        plan: CriteriaRubricPlanV2,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledCriteriaRubricPlanV2:
        canonical_plan = CriteriaRubricPlanV2.model_validate_json(plan.canonical_json())
        canonical_policy = FactoryRunPolicyV2.model_validate_json(policy.canonical_json())
        self._validate_run_policy(canonical_plan, canonical_policy)
        definition, capabilities = self._resolve_agent(canonical_plan)
        capabilities_by_id = {capability.capability_id: capability for capability in capabilities}
        if not set(canonical_plan.required_capability_ids).issubset(capabilities_by_id):
            raise CriteriaRubricPlanCompilerError(
                "criteria/rubric plan requires an unsupported Agent capability"
            )
        input_types = {
            object_type for capability in capabilities for object_type in capability.input_object_types
        }
        output_types = {
            object_type for capability in capabilities for object_type in capability.output_object_types
        }
        if not _REQUIRED_INPUT_TYPES.issubset(input_types):
            raise CriteriaRubricPlanCompilerError("criteria/rubric Agent input schema is incomplete")
        if _REQUIRED_OUTPUT_TYPE not in output_types:
            raise CriteriaRubricPlanCompilerError("criteria/rubric Agent output schema is incompatible")
        if not set(canonical_plan.specialist_tool_ids).issubset(definition.tool_ids):
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan widens Agent tool scope")
        if canonical_plan.data_purpose != definition.data_purpose:
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan widens Agent data purpose")
        if not set(canonical_plan.data_classifications).issubset(definition.allowed_data_classifications):
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan widens Agent data classification")
        if canonical_plan.prompt_template_ref != definition.prompt_template_ref:
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan changes the Agent prompt template")
        if canonical_plan.model_policy_ref != definition.model_policy_ref:
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan changes the Agent model policy")
        if not set(canonical_plan.acceptance_check_refs).issubset(definition.validator_refs):
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan uses an unregistered validator")
        if definition.workspace_isolated is not True:
            raise CriteriaRubricPlanCompilerError("criteria/rubric Agent requires an isolated workspace")
        if (
            canonical_plan.max_attempts > definition.max_attempts
            or canonical_plan.max_model_requests > definition.max_model_requests
            or canonical_plan.max_model_tokens > definition.max_model_tokens
            or canonical_plan.max_cost_micro_usd > definition.max_cost_micro_usd
        ):
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan exceeds Agent definition budget")
        capability_refs = tuple(
            sorted(
                (
                    capabilities_by_id[capability_id].to_ref()
                    for capability_id in canonical_plan.required_capability_ids
                ),
                key=_ref_key,
            )
        )
        return CompiledCriteriaRubricPlanV2.create(
            compiled_plan_id=(
                "compiled-criteria-rubric-plan://"
                f"{canonical_plan.plan_id.rsplit('://', 1)[-1]}"
                f"/v{canonical_plan.plan_version}"
            ),
            source_plan_ref=canonical_plan.to_ref(),
            policy_ref=canonical_policy.to_ref(),
            agent_definition_ref=definition.to_ref(),
            capability_refs=capability_refs,
            task_draft_ref=canonical_plan.task_draft_ref,
            attachment_quality_ref=(canonical_plan.attachment_quality_ref),
            solvability_ref=canonical_plan.solvability_ref,
            tool_catalog_ref=canonical_plan.tool_catalog_ref,
            prompt_template_ref=canonical_plan.prompt_template_ref,
            model_policy_ref=canonical_plan.model_policy_ref,
            acceptance_check_refs=(canonical_plan.acceptance_check_refs),
            audit=audit,
        )

    @staticmethod
    def _validate_run_policy(
        plan: CriteriaRubricPlanV2,
        policy: FactoryRunPolicyV2,
    ) -> None:
        if _CRITERIA_RUBRIC_TASK_KIND not in policy.allowed_task_kinds:
            raise CriteriaRubricPlanCompilerError("criteria-rubric task kind is outside run policy")
        if plan.plan_version > policy.max_plan_revisions + 1:
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan exceeds revision policy")
        if plan.max_attempts > policy.max_agent_attempts:
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan exceeds attempt policy")
        if (
            plan.max_model_requests > policy.max_model_requests
            or plan.max_model_tokens > policy.max_model_tokens
            or plan.max_cost_micro_usd > policy.max_cost_micro_usd
        ):
            raise CriteriaRubricPlanCompilerError("criteria/rubric plan exceeds Factory run budget")

    def _resolve_agent(
        self,
        plan: CriteriaRubricPlanV2,
    ) -> tuple[
        AgentDefinitionV2,
        tuple[AgentCapabilityV2, ...],
    ]:
        try:
            definition = self.registry.resolve(
                plan.agent_role,
                _CRITERIA_RUBRIC_TASK_KIND,
            )
            capabilities = self.registry.capabilities_for(definition)
        except AgentRegistryError as exc:
            raise CriteriaRubricPlanCompilerError("criteria/rubric Agent resolution failed") from exc
        return definition, capabilities


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "CriteriaRubricPlanCompiler",
    "CriteriaRubricPlanCompilerError",
]
