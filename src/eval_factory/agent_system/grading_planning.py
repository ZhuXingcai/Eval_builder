from __future__ import annotations

from eval_factory.agent_system.registry import (
    AgentRegistry,
    AgentRegistryError,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    CompiledGradingDesignPlanV2,
    FactoryRunPolicyV2,
    GradingDesignPlanV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef

_TASK_KIND = "grading-design"
_REQUIRED_INPUT_TYPES = {
    "criteria-rubric-result",
    "rubric-set",
    "evaluator-spec",
    "reference-policy",
    "tool-policy",
}
_REQUIRED_OUTPUT_TYPE = "grading-design-result"


class GradingDesignPlanCompilerError(RuntimeError):
    pass


class GradingDesignPlanCompiler:
    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def compile(
        self,
        *,
        plan: GradingDesignPlanV2,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledGradingDesignPlanV2:
        canonical_plan = GradingDesignPlanV2.model_validate_json(plan.canonical_json())
        canonical_policy = FactoryRunPolicyV2.model_validate_json(policy.canonical_json())
        self._validate_run_policy(
            canonical_plan,
            canonical_policy,
        )
        definition, capabilities = self._resolve_agent(canonical_plan)
        capabilities_by_id = {capability.capability_id: capability for capability in capabilities}
        if not set(canonical_plan.required_capability_ids).issubset(capabilities_by_id):
            raise GradingDesignPlanCompilerError("grading plan requires an unsupported Agent capability")
        input_types = {
            object_type for capability in capabilities for object_type in capability.input_object_types
        }
        output_types = {
            object_type for capability in capabilities for object_type in capability.output_object_types
        }
        if not _REQUIRED_INPUT_TYPES.issubset(input_types):
            raise GradingDesignPlanCompilerError("grading Agent input schema is incomplete")
        if _REQUIRED_OUTPUT_TYPE not in output_types:
            raise GradingDesignPlanCompilerError("grading Agent output schema is incompatible")
        if not set(canonical_plan.specialist_tool_ids).issubset(definition.tool_ids):
            raise GradingDesignPlanCompilerError("grading plan widens Agent tool scope")
        if canonical_plan.data_purpose != definition.data_purpose:
            raise GradingDesignPlanCompilerError("grading plan widens Agent data purpose")
        if not set(canonical_plan.data_classifications).issubset(definition.allowed_data_classifications):
            raise GradingDesignPlanCompilerError("grading plan widens Agent data classification")
        if canonical_plan.judge_prompt_template_ref != definition.prompt_template_ref:
            raise GradingDesignPlanCompilerError("grading plan changes the Agent prompt template")
        if canonical_plan.model_policy_ref != definition.model_policy_ref:
            raise GradingDesignPlanCompilerError("grading plan changes the Agent model policy")
        if not set(canonical_plan.acceptance_check_refs).issubset(definition.validator_refs):
            raise GradingDesignPlanCompilerError("grading plan uses an unregistered validator")
        if definition.workspace_isolated is not True:
            raise GradingDesignPlanCompilerError("grading Agent requires an isolated workspace")
        if (
            canonical_plan.max_attempts > definition.max_attempts
            or canonical_plan.max_model_requests > definition.max_model_requests
            or canonical_plan.max_model_tokens > definition.max_model_tokens
            or canonical_plan.max_cost_micro_usd > definition.max_cost_micro_usd
        ):
            raise GradingDesignPlanCompilerError("grading plan exceeds Agent definition budget")
        capability_refs = tuple(
            sorted(
                (
                    capabilities_by_id[capability_id].to_ref()
                    for capability_id in (canonical_plan.required_capability_ids)
                ),
                key=_ref_key,
            )
        )
        return CompiledGradingDesignPlanV2.create(
            compiled_plan_id=(
                "compiled-grading-design-plan://"
                f"{canonical_plan.plan_id.rsplit('://', 1)[-1]}"
                f"/v{canonical_plan.plan_version}"
            ),
            source_plan_ref=canonical_plan.to_ref(),
            policy_ref=canonical_policy.to_ref(),
            agent_definition_ref=definition.to_ref(),
            capability_refs=capability_refs,
            criteria_rubric_result_ref=(canonical_plan.criteria_rubric_result_ref),
            rubric_set_ref=canonical_plan.rubric_set_ref,
            evaluator_spec_ref=canonical_plan.evaluator_spec_ref,
            reference_policy_ref=(canonical_plan.reference_policy_ref),
            tool_policy_ref=canonical_plan.tool_policy_ref,
            generator_model_profile_ref=(canonical_plan.generator_model_profile_ref),
            judge_prompt_template_ref=(canonical_plan.judge_prompt_template_ref),
            model_policy_ref=canonical_plan.model_policy_ref,
            acceptance_check_refs=(canonical_plan.acceptance_check_refs),
            audit=audit,
        )

    @staticmethod
    def _validate_run_policy(
        plan: GradingDesignPlanV2,
        policy: FactoryRunPolicyV2,
    ) -> None:
        if _TASK_KIND not in policy.allowed_task_kinds:
            raise GradingDesignPlanCompilerError("grading-design task kind is outside run policy")
        if plan.plan_version > policy.max_plan_revisions + 1:
            raise GradingDesignPlanCompilerError("grading plan exceeds revision policy")
        if plan.max_attempts > policy.max_agent_attempts:
            raise GradingDesignPlanCompilerError("grading plan exceeds attempt policy")
        if (
            plan.max_model_requests > policy.max_model_requests
            or plan.max_model_tokens > policy.max_model_tokens
            or plan.max_cost_micro_usd > policy.max_cost_micro_usd
        ):
            raise GradingDesignPlanCompilerError("grading plan exceeds Factory run budget")

    def _resolve_agent(
        self,
        plan: GradingDesignPlanV2,
    ) -> tuple[
        AgentDefinitionV2,
        tuple[AgentCapabilityV2, ...],
    ]:
        try:
            definition = self.registry.resolve(
                plan.agent_role,
                _TASK_KIND,
            )
            capabilities = self.registry.capabilities_for(definition)
        except AgentRegistryError as exc:
            raise GradingDesignPlanCompilerError("grading Agent resolution failed") from exc
        return definition, capabilities


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "GradingDesignPlanCompiler",
    "GradingDesignPlanCompilerError",
]
