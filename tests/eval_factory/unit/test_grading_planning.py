from __future__ import annotations

import pytest
from grading_fixtures import audit, plan, policy, ref, registry

from eval_factory.agent_system.grading_planning import (
    GradingDesignPlanCompiler,
    GradingDesignPlanCompilerError,
)
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
)


def test_grading_plan_compiler_resolves_least_privilege_agent() -> None:
    source = plan()
    compiled = GradingDesignPlanCompiler(registry()).compile(
        plan=source,
        policy=policy(),
        audit=audit(),
    )

    assert compiled.source_plan_ref == source.to_ref()
    assert compiled.criteria_rubric_result_ref == (source.criteria_rubric_result_ref)
    assert compiled.generator_model_profile_ref == (source.generator_model_profile_ref)
    assert compiled.production_release_allowed is False


@pytest.mark.parametrize(
    ("plan_kwargs", "message"),
    (
        (
            {"required_capability_ids": ("agent-capability://missing",)},
            "capability",
        ),
        (
            {"specialist_tool_ids": ("unknown-tool",)},
            "tool",
        ),
        (
            {"data_classifications": ("SECRET",)},
            "classification",
        ),
        (
            {
                "prompt_template_ref": ref(
                    "prompt-template",
                    "other",
                )
            },
            "prompt",
        ),
        (
            {
                "model_policy_ref": ref(
                    "model-routing-policy",
                    "other",
                )
            },
            "model policy",
        ),
        (
            {"acceptance_check_refs": (ref("validator", "other"),)},
            "validator",
        ),
        (
            {"max_model_tokens": 16_001},
            "budget",
        ),
    ),
)
def test_grading_plan_compiler_rejects_scope_widening(
    plan_kwargs: dict[str, object],
    message: str,
) -> None:
    source = plan(**plan_kwargs)

    with pytest.raises(
        GradingDesignPlanCompilerError,
        match=message,
    ):
        GradingDesignPlanCompiler(registry()).compile(
            plan=source,
            policy=policy(),
            audit=audit(),
        )


def test_grading_plan_compiler_rejects_run_policy() -> None:
    source = plan()
    base = policy()
    denied = type(base).create(
        policy_id=base.policy_id,
        allowed_task_kinds=("other-task",),
        max_transitions=base.max_transitions,
        max_plan_revisions=base.max_plan_revisions,
        max_agent_attempts=base.max_agent_attempts,
        max_model_requests=base.max_model_requests,
        max_model_tokens=base.max_model_tokens,
        max_cost_micro_usd=base.max_cost_micro_usd,
        audit=audit(),
    )

    with pytest.raises(
        GradingDesignPlanCompilerError,
        match="task kind",
    ):
        GradingDesignPlanCompiler(registry()).compile(
            plan=source,
            policy=denied,
            audit=audit(),
        )


def test_grading_plan_compiler_rejects_incomplete_agent_schema() -> None:
    base = registry()
    definition = base.definitions[0]
    capability = AgentCapabilityV2.create(
        capability_id="agent-capability://grading-design",
        task_kinds=("grading-design",),
        input_object_types=("criteria-rubric-result",),
        output_object_types=("grading-design-result",),
        model_capabilities=("reasoning",),
        tool_ids=("reference-grant-read",),
        data_purposes=("grading-design-authoring",),
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
        audit=audit(),
    )
    incomplete_definition = AgentDefinitionV2.create(
        agent_definition_id=definition.agent_definition_id,
        agent_role=definition.agent_role,
        agent_version=definition.agent_version,
        capability_refs=(capability.to_ref(),),
        prompt_template_ref=definition.prompt_template_ref,
        model_policy_ref=definition.model_policy_ref,
        tool_ids=definition.tool_ids,
        data_purpose=definition.data_purpose,
        allowed_data_classifications=(definition.allowed_data_classifications),
        validator_refs=definition.validator_refs,
        max_attempts=definition.max_attempts,
        max_model_requests=definition.max_model_requests,
        max_model_tokens=definition.max_model_tokens,
        max_cost_micro_usd=definition.max_cost_micro_usd,
        workspace_isolated=True,
        network_allowed=False,
        audit=audit(),
    )
    incomplete = AgentRegistry(
        capabilities=(capability,),
        definitions=(incomplete_definition,),
    )
    source = plan(
        prompt_template_ref=(incomplete_definition.prompt_template_ref),
        model_policy_ref=incomplete_definition.model_policy_ref,
        acceptance_check_refs=(incomplete_definition.validator_refs),
    )

    with pytest.raises(
        GradingDesignPlanCompilerError,
        match="input schema",
    ):
        GradingDesignPlanCompiler(incomplete).compile(
            plan=source,
            policy=policy(),
            audit=audit(),
        )
