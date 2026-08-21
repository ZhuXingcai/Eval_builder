from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.agent_system_v2 import (
    CompiledCriteriaRubricPlanV2,
    CriteriaRubricGoalV2,
    CriteriaRubricOutcomeV2,
    CriteriaRubricPlanV2,
    CriteriaRubricResultV2,
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
        created_by="criteria-rubric-contract-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="criteria-rubric-v1",
                sha256=HASH,
            ),
        ),
    )


def _goals() -> tuple[CriteriaRubricGoalV2, ...]:
    return (
        CriteriaRubricGoalV2(
            goal_id="criteria-goal://response",
            goal_summary="Judge the visible response requirement.",
            judged_object_kind=(RubricJudgedObjectKindV2.CONTESTANT_RESPONSE),
            prompt_requirement_ids=("requirement://response",),
            attachment_dependency_ids=(),
            allowed_tool_ids=(),
            evaluator_binding_id="evaluator-binding://default",
            weight_basis_points=5000,
        ),
        CriteriaRubricGoalV2(
            goal_id="criteria-goal://workspace-tool",
            goal_summary="Judge workspace state and allowed tool use.",
            judged_object_kind=RubricJudgedObjectKindV2.WORKSPACE_STATE,
            prompt_requirement_ids=("requirement://workspace",),
            attachment_dependency_ids=("attachment-dependency://workspace",),
            allowed_tool_ids=("file-read",),
            evaluator_binding_id="evaluator-binding://default",
            weight_basis_points=5000,
        ),
    )


def _plan(
    *,
    version: int = 1,
    predecessor: ObjectRef | None = None,
    goals: tuple[CriteriaRubricGoalV2, ...] | None = None,
    attachment_quality_ref: ObjectRef | None = None,
) -> CriteriaRubricPlanV2:
    return CriteriaRubricPlanV2.create(
        plan_id="criteria-rubric-plan://example",
        run_ref=_ref("factory-run"),
        plan_version=version,
        predecessor_plan_ref=predecessor,
        task_draft_ref=_ref("task-draft"),
        attachment_quality_ref=(attachment_quality_ref or _ref("attachment-quality-assessment")),
        solvability_ref=_ref("solvability-assessment"),
        allowed_prompt_requirement_ids=(
            "requirement://workspace",
            "requirement://response",
        ),
        required_prompt_requirement_ids=(
            "requirement://workspace",
            "requirement://response",
        ),
        allowed_attachment_dependency_ids=("attachment-dependency://workspace",),
        required_attachment_dependency_ids=("attachment-dependency://workspace",),
        allowed_task_tool_ids=("file-read",),
        required_task_tool_ids=("file-read",),
        criterion_goals=goals or _goals(),
        allowed_evaluator_binding_ids=("evaluator-binding://default",),
        allowed_reference_modes=(
            ReferenceMode.NONE,
            ReferenceMode.STRUCTURED_EXPECTATIONS,
        ),
        selected_reference_mode=ReferenceMode.NONE,
        tool_catalog_ref=_ref(
            "tool-capability-catalog",
            version="tool-capability-catalog/r4-07-v1",
        ),
        agent_role="criteria-rubric-agent",
        required_capability_ids=("agent-capability://criteria-rubric",),
        specialist_tool_ids=("rubric-candidate-read",),
        data_classifications=(
            "RESTRICTED_EVALUATOR_CONTROL",
            "RESTRICTED_TRACE_DERIVED",
        ),
        prompt_template_ref=_ref("prompt-template"),
        model_policy_ref=_ref("model-routing-policy"),
        acceptance_check_refs=(_ref("validator"),),
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=16_000,
        max_cost_micro_usd=500_000,
        audit=_audit(),
    )


def test_plan_is_strict_frozen_canonical_and_audit_independent() -> None:
    plan = _plan(goals=tuple(reversed(_goals())))

    assert tuple(goal.goal_id for goal in plan.criterion_goals) == (
        "criteria-goal://response",
        "criteria-goal://workspace-tool",
    )
    assert plan.object_sha256 == plan.to_ref().object_sha256
    with pytest.raises(ValidationError):
        plan.plan_version = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        CriteriaRubricPlanV2.model_validate(
            {
                **plan.model_dump(mode="python"),
                "private_reference_body": "forbidden",
            }
        )


def test_plan_rejects_scope_coverage_weight_and_sensitive_refs() -> None:
    goals = _goals()
    with pytest.raises(ValidationError, match="10000"):
        _plan(
            goals=(
                goals[0].model_copy(update={"weight_basis_points": 4000}),
                goals[1],
            )
        )
    with pytest.raises(ValidationError, match="cover required"):
        _plan(goals=(goals[0].model_copy(update={"weight_basis_points": 10_000}),))
    with pytest.raises(ValidationError, match="sensitive"):
        _plan(
            attachment_quality_ref=_ref(
                "attachment-quality-assessment",
                "private-reference",
            )
        )


def test_plan_requires_contiguous_predecessor_shape() -> None:
    with pytest.raises(ValidationError, match="predecessor"):
        _plan(version=2)
    first = _plan()
    successor = _plan(
        version=2,
        predecessor=first.to_ref(),
    )
    assert successor.predecessor_plan_ref == first.to_ref()


def test_compiled_plan_requires_exact_safe_authority_refs() -> None:
    plan = _plan()
    compiled = CompiledCriteriaRubricPlanV2.create(
        compiled_plan_id="compiled-criteria-rubric-plan://example",
        source_plan_ref=plan.to_ref(),
        policy_ref=_ref("factory-run-policy"),
        agent_definition_ref=_ref("agent-definition"),
        capability_refs=(_ref("agent-capability"),),
        task_draft_ref=plan.task_draft_ref,
        attachment_quality_ref=plan.attachment_quality_ref,
        solvability_ref=plan.solvability_ref,
        tool_catalog_ref=plan.tool_catalog_ref,
        prompt_template_ref=plan.prompt_template_ref,
        model_policy_ref=plan.model_policy_ref,
        acceptance_check_refs=plan.acceptance_check_refs,
        audit=_audit(),
    )

    assert compiled.production_release_allowed is False
    assert compiled.source_plan_ref == plan.to_ref()


def test_result_requires_all_or_none_r4_chain() -> None:
    plan = _plan()
    common = {
        "result_id": "criteria-rubric-result://example",
        "plan_ref": plan.to_ref(),
        "task_draft_ref": plan.task_draft_ref,
        "attachment_quality_ref": plan.attachment_quality_ref,
        "solvability_ref": plan.solvability_ref,
        "route_decision_ref": _ref("model-route-decision"),
        "gateway_receipt_ref": _ref("gateway-receipt"),
        "gateway_invocation_result_ref": _ref(
            "gateway-invocation-result",
        ),
        "audit": _audit(),
    }
    succeeded = CriteriaRubricResultV2.create(
        **common,
        proposal_ref=_ref("criteria-rubric-proposal"),
        rubric_set_ref=_ref("rubric-set"),
        evaluator_spec_ref=_ref("evaluator-spec"),
        reference_policy_ref=_ref("reference-policy"),
        tool_policy_ref=_ref("tool-policy"),
        contestant_tool_policy_ref=_ref("contestant-tool-policy"),
        outcome=CriteriaRubricOutcomeV2.SUCCEEDED,
        reason_codes=(),
    )
    blocked = CriteriaRubricResultV2.create(
        **common,
        proposal_ref=None,
        rubric_set_ref=None,
        evaluator_spec_ref=None,
        reference_policy_ref=None,
        tool_policy_ref=None,
        contestant_tool_policy_ref=None,
        outcome=CriteriaRubricOutcomeV2.BLOCKED_CAPABILITY,
        reason_codes=("MODEL_UNAVAILABLE",),
    )

    assert succeeded.rubric_set_ref is not None
    assert blocked.rubric_set_ref is None
    with pytest.raises(ValidationError, match="partial chain"):
        CriteriaRubricResultV2.create(
            **common,
            proposal_ref=_ref("criteria-rubric-proposal"),
            rubric_set_ref=_ref("rubric-set"),
            evaluator_spec_ref=None,
            reference_policy_ref=None,
            tool_policy_ref=None,
            contestant_tool_policy_ref=None,
            outcome=CriteriaRubricOutcomeV2.BLOCKED_BINDING,
            reason_codes=("MISSING_BINDING",),
        )
