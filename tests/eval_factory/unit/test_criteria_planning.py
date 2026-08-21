from __future__ import annotations

from datetime import UTC, datetime

import pytest

from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
    CriteriaRubricPlanCompilerError,
)
from eval_factory.agent_system.criteria_registry import (
    CriteriaRubricAgentRegistryConfig,
    build_criteria_rubric_agent_registry,
)
from eval_factory.contracts.agent_system_v2 import (
    CriteriaRubricGoalV2,
    CriteriaRubricPlanV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.task import ReferenceMode
from eval_factory.contracts.task_v2 import RubricJudgedObjectKindV2

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "current",
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 6, tzinfo=UTC),
        created_by="criteria-planning-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="criteria-rubric-v1",
                sha256=HASH,
            ),
        ),
    )


def _registry():
    return build_criteria_rubric_agent_registry(
        config=CriteriaRubricAgentRegistryConfig(
            prompt_ref=_ref("prompt-template", "criteria-rubric"),
            model_policy_ref=_ref(
                "model-routing-policy",
                "criteria-rubric",
            ),
        ),
        audit=_audit(),
    )


def _plan(
    *,
    updates: dict[str, object] | None = None,
) -> CriteriaRubricPlanV2:
    registry = _registry()
    definition = registry.resolve(
        "criteria-rubric-agent",
        "criteria-rubric",
    )
    values: dict[str, object] = {
        "plan_id": "criteria-rubric-plan://planning",
        "run_ref": _ref("factory-run"),
        "plan_version": 1,
        "predecessor_plan_ref": None,
        "task_draft_ref": _ref("task-draft"),
        "attachment_quality_ref": _ref("attachment-quality-assessment"),
        "solvability_ref": _ref("solvability-assessment"),
        "allowed_prompt_requirement_ids": ("requirement://visible",),
        "required_prompt_requirement_ids": ("requirement://visible",),
        "allowed_attachment_dependency_ids": (),
        "required_attachment_dependency_ids": (),
        "allowed_task_tool_ids": (),
        "required_task_tool_ids": (),
        "criterion_goals": (
            CriteriaRubricGoalV2(
                goal_id="criteria-goal://response",
                goal_summary="Judge the visible response.",
                judged_object_kind=(RubricJudgedObjectKindV2.CONTESTANT_RESPONSE),
                prompt_requirement_ids=("requirement://visible",),
                evaluator_binding_id="evaluator-binding://default",
                weight_basis_points=10_000,
            ),
        ),
        "allowed_evaluator_binding_ids": ("evaluator-binding://default",),
        "allowed_reference_modes": (ReferenceMode.NONE,),
        "selected_reference_mode": ReferenceMode.NONE,
        "tool_catalog_ref": _ref(
            "tool-capability-catalog",
            version="tool-capability-catalog/r4-07-v1",
        ),
        "agent_role": definition.agent_role,
        "required_capability_ids": ("agent-capability://criteria-rubric",),
        "specialist_tool_ids": ("rubric-candidate-read",),
        "data_classifications": (
            "RESTRICTED_EVALUATOR_CONTROL",
            "RESTRICTED_TRACE_DERIVED",
        ),
        "prompt_template_ref": definition.prompt_template_ref,
        "model_policy_ref": definition.model_policy_ref,
        "acceptance_check_refs": definition.validator_refs,
        "max_attempts": 2,
        "max_model_requests": 1,
        "max_model_tokens": 16_000,
        "max_cost_micro_usd": 500_000,
        "audit": _audit(),
    }
    if updates:
        values.update(updates)
    return CriteriaRubricPlanV2.create(**values)


def _policy(
    *,
    allowed_task_kinds: tuple[str, ...] = ("criteria-rubric",),
    max_model_requests: int = 10,
) -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://criteria-rubric",
        allowed_task_kinds=allowed_task_kinds,
        max_transitions=128,
        max_plan_revisions=8,
        max_agent_attempts=2,
        max_model_requests=max_model_requests,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )


def test_compiler_resolves_least_privilege_agent_authority() -> None:
    plan = _plan()
    compiled = CriteriaRubricPlanCompiler(_registry()).compile(
        plan=plan,
        policy=_policy(),
        audit=_audit(),
    )

    assert compiled.source_plan_ref == plan.to_ref()
    assert compiled.agent_definition_ref.object_type == "agent-definition"
    assert compiled.capability_refs[0].object_id.startswith("agent-capability://")
    assert compiled.task_draft_ref == plan.task_draft_ref
    assert compiled.attachment_quality_ref == plan.attachment_quality_ref
    assert compiled.solvability_ref == plan.solvability_ref
    assert compiled.production_release_allowed is False


@pytest.mark.parametrize(
    ("plan_updates", "policy", "message"),
    (
        (
            {},
            _policy(allowed_task_kinds=("attachment-mock",)),
            "task kind",
        ),
        (
            {"max_model_requests": 2},
            _policy(max_model_requests=1),
            "Factory run budget",
        ),
        (
            {"agent_role": "missing-agent"},
            _policy(),
            "resolution",
        ),
        (
            {"required_capability_ids": ("agent-capability://unsupported",)},
            _policy(),
            "capability",
        ),
        (
            {"specialist_tool_ids": ("shell",)},
            _policy(),
            "tool",
        ),
        (
            {"data_classifications": ("PUBLIC",)},
            _policy(),
            "classification",
        ),
        (
            {
                "prompt_template_ref": _ref(
                    "prompt-template",
                    "other",
                )
            },
            _policy(),
            "prompt",
        ),
        (
            {
                "model_policy_ref": _ref(
                    "model-routing-policy",
                    "other",
                )
            },
            _policy(),
            "model policy",
        ),
        (
            {"acceptance_check_refs": (_ref("validator", "unregistered"),)},
            _policy(),
            "validator",
        ),
        (
            {"max_model_tokens": 20_000},
            _policy(),
            "definition budget",
        ),
    ),
)
def test_compiler_rejects_scope_or_budget_widening(
    plan_updates: dict[str, object],
    policy: FactoryRunPolicyV2,
    message: str,
) -> None:
    with pytest.raises(
        CriteriaRubricPlanCompilerError,
        match=message,
    ):
        CriteriaRubricPlanCompiler(_registry()).compile(
            plan=_plan(updates=plan_updates),
            policy=policy,
            audit=_audit(),
        )
