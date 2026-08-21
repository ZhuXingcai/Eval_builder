from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from attachment_gateway_fixtures import (
    audit,
    profile,
    ref,
)
from test_attachment_supervised_execution import (
    _prepared as prepare_attachment,
)
from test_criteria_agent import (
    _binding_definitions,
    _catalog,
)
from test_criteria_agent import (
    _candidate_agent as criteria_candidate_agent,
)
from test_criteria_planning import (
    _registry as criteria_registry,
)
from test_deterministic_validation import (
    _FakeValidationFacade,
    _reference_set,
)
from test_domain_plan_review import (
    RUN_ID,
    USER,
)
from test_grading_agent import (
    _candidate_agent as grading_candidate_agent,
)
from test_grading_agent import (
    _CriteriaAuthority,
)
from test_grading_agent import (
    _plan as grading_plan,
)
from test_grading_agent import (
    _prompts as grading_prompts,
)
from test_grading_agent import (
    _proposal as grading_proposal,
)
from test_item_quality import _task_contract_set
from test_rubric_authoring import EVALUATOR_BINDING
from test_semantic_review import (
    _AcceptingBackend,
    _DynamicResolver,
    _NoRepairBackend,
)
from test_semantic_review import (
    _policy as review_policy,
)
from test_semantic_review import (
    _sources as review_sources,
)
from test_semantic_review import (
    _store as quality_store,
)
from test_solvability_agent import (
    _agent as solvability_agent,
)
from typer.testing import CliRunner

from env_mock_agent.facade.semantic_review_adapter import (
    RegistryAttachmentSemanticReviewFacade,
)
from eval_factory.agent_system.attachment_finalization import (
    AttachmentR5FinalizationPipeline,
)
from eval_factory.agent_system.attachment_subgraph import (
    SupervisedAttachmentR5Runner,
)
from eval_factory.agent_system.criteria_agent import (
    CriteriaRubricProposalV1,
)
from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
)
from eval_factory.agent_system.criteria_subgraph import (
    SupervisedCriteriaRubricRunner,
)
from eval_factory.agent_system.grading_planning import (
    GradingDesignPlanCompiler,
)
from eval_factory.agent_system.grading_registry import (
    GradingDesignAgentRegistryConfig,
    build_grading_design_agent_registry,
)
from eval_factory.agent_system.grading_subgraph import (
    SupervisedGradingDesignRunner,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.supervisor import ExecutionSupervisor
from eval_factory.agent_system.workspace import AgentWorkspaceManager
from eval_factory.cli import app
from eval_factory.contracts.agent_system_v2 import (
    CriteriaRubricGoalV2,
    CriteriaRubricOutcomeV2,
    CriteriaRubricPlanV2,
    GradingDesignOutcomeV2,
    GradingDesignPlanV2,
    JudgeDesignSpecV2,
    PlanDecisionKindV2,
    PlanKindV2,
)
from eval_factory.contracts.task import ReferenceMode, RubricVisibility
from eval_factory.contracts.task_v2 import (
    EvaluatorSpecV2,
    ReferencePolicyV2,
    RubricJudgedObjectKindV2,
    RubricSetV2,
    ToolPolicyV2,
    producer_task_view_ref,
    task_draft_ref,
)
from eval_factory.task_authoring import (
    RubricAuthoringOutcome,
    RubricCriterionSelection,
    tool_capability_catalog_ref,
)


@pytest.mark.asyncio
async def test_reviewed_attachment_criteria_grading_chain_replays(
    tmp_path: Path,
) -> None:
    attachment_kwargs, attachment_facade, attachment_supervisor, producer_view = await prepare_attachment(
        tmp_path / "attachment"
    )
    attachment_runner = SupervisedAttachmentR5Runner(
        supervisor=attachment_supervisor,
    )
    attachment = await attachment_runner.run(**attachment_kwargs)
    attachment_calls = attachment_facade.calls.copy()
    assert await attachment_runner.run(**attachment_kwargs) == attachment
    assert attachment_facade.calls == attachment_calls

    policy = review_policy()
    backend = _AcceptingBackend()
    semantic_facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: backend for role in policy.roles_in_order},
        repair_backend=_NoRepairBackend(),
    )
    job_store, job_id, item_id = quality_store(
        tmp_path / "quality",
    )
    pipeline = AttachmentR5FinalizationPipeline()
    finalization = await pipeline.finalize(
        supervised_execution=attachment,
        producer_task_view=producer_view,
        leakage_reference_set=_reference_set(),
        configured_pii_rules=(),
        validation_facade=_FakeValidationFacade(),
        task_contract_set=_task_contract_set(
            producer_task_view_ref(producer_view),
        ),
        review_policy=policy,
        context_sources=review_sources(),
        review_facade=semantic_facade,
        job_store=job_store,
        job_id=job_id,
        item_id=item_id,
        audit=audit(),
    )
    evidence_refs = (finalization.attachment_quality.validator_result_refs[0],)
    solvability, _, solvability_provider, generator, _ = solvability_agent(
        tmp_path / "solvability",
        proposal_evidence_refs=evidence_refs,
    )
    solvability_result = await pipeline.assess_solvability(
        supervised_execution=attachment,
        finalization=finalization,
        agent=solvability,
        task_ref=ref(
            "agent-task",
            "specialist-chain-solvability",
        ),
        evidence_refs=evidence_refs,
        audit=audit(),
    )
    assert len(solvability_provider.calls) == 1

    store = attachment_supervisor.store
    assert (
        store.get_domain_result(
            RUN_ID,
            PlanKindV2.ATTACHMENT_GENERATION,
        ).result_ref
        == attachment.subgraph_result.to_ref()
    )

    task_draft = attachment_kwargs["preparation"].task_draft
    dependency_ids = tuple(
        sorted(dependency.dependency_id for dependency in task_draft.attachment_dependencies)
    )
    prompt_requirement_ids = tuple(sorted(task_draft.prompt_requirement_ids))
    tool_ids = tuple(sorted(task_draft.allowed_tools))
    catalog = _catalog()
    criterion_goal = CriteriaRubricGoalV2(
        goal_id="criteria-goal://specialist-chain",
        goal_summary="Judge the visible workspace-grounded response.",
        judged_object_kind=RubricJudgedObjectKindV2.WORKSPACE_STATE,
        prompt_requirement_ids=prompt_requirement_ids,
        attachment_dependency_ids=dependency_ids,
        allowed_tool_ids=tool_ids,
        evaluator_binding_id=EVALUATOR_BINDING,
        weight_basis_points=10_000,
    )
    criteria_definition = criteria_registry().resolve(
        "criteria-rubric-agent",
        "criteria-rubric",
    )
    criteria_plan = CriteriaRubricPlanV2.create(
        plan_id="criteria-rubric-plan://specialist-chain",
        run_ref=store.get_run(RUN_ID).to_ref(),
        plan_version=1,
        predecessor_plan_ref=None,
        task_draft_ref=task_draft_ref(task_draft),
        attachment_quality_ref=finalization.attachment_quality.to_ref(),
        solvability_ref=solvability_result.assessment.to_ref(),
        allowed_prompt_requirement_ids=prompt_requirement_ids,
        required_prompt_requirement_ids=prompt_requirement_ids,
        allowed_attachment_dependency_ids=dependency_ids,
        required_attachment_dependency_ids=dependency_ids,
        allowed_task_tool_ids=tool_ids,
        required_task_tool_ids=tool_ids,
        criterion_goals=(criterion_goal,),
        allowed_evaluator_binding_ids=(EVALUATOR_BINDING,),
        allowed_reference_modes=(ReferenceMode.NONE,),
        selected_reference_mode=ReferenceMode.NONE,
        tool_catalog_ref=tool_capability_catalog_ref(catalog),
        agent_role=criteria_definition.agent_role,
        required_capability_ids=("agent-capability://criteria-rubric",),
        specialist_tool_ids=("rubric-candidate-read",),
        data_classifications=(
            "RESTRICTED_EVALUATOR_CONTROL",
            "RESTRICTED_TRACE_DERIVED",
        ),
        prompt_template_ref=criteria_definition.prompt_template_ref,
        model_policy_ref=criteria_definition.model_policy_ref,
        acceptance_check_refs=criteria_definition.validator_refs,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=16_000,
        max_cost_micro_usd=500_000,
        audit=audit(),
    )
    criteria_proposal = CriteriaRubricProposalV1(
        outcome=RubricAuthoringOutcome.COMPILED,
        criteria=(
            RubricCriterionSelection(
                selection_id=criterion_goal.goal_id,
                judged_object_id="judged-object://specialist-chain",
                judged_object_kind=criterion_goal.judged_object_kind,
                judged_object_description=criterion_goal.goal_summary,
                description="Judge only visible workspace-grounded behavior.",
                weight=1.0,
                prompt_requirement_ids=prompt_requirement_ids,
                attachment_dependency_ids=dependency_ids,
                allowed_tool_ids=tool_ids,
                visibility=RubricVisibility.EVALUATOR_ONLY,
                evaluator_binding=EVALUATOR_BINDING,
            ),
        ),
        unresolved_reasons=frozenset(),
        confidence_basis_points=9200,
    )
    (
        criteria_agent,
        criteria_material_store,
        criteria_provider,
        criteria_generator,
        _,
    ) = criteria_candidate_agent(
        tmp_path / "criteria",
        plan=criteria_plan,
        proposal=criteria_proposal,
        control_store=store,
    )
    criteria_review = _approve_and_resume(
        PlanReviewService(store),
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        plan=criteria_plan,
        suffix="criteria",
    )
    criteria_compiled = CriteriaRubricPlanCompiler(
        criteria_registry(),
    ).compile(
        plan=criteria_plan,
        policy=store.get_policy(
            store.get_run(RUN_ID).policy_ref.object_id,
        ),
        audit=audit(),
    )
    criteria_supervisor = ExecutionSupervisor(
        store,
        criteria_registry(),
        workspace_manager=AgentWorkspaceManager(
            tmp_path / "criteria-workspace",
        ),
    )
    criteria_runner = SupervisedCriteriaRubricRunner(
        supervisor=criteria_supervisor,
        agent=criteria_agent,
    )
    criteria = await criteria_runner.run(
        run_id=RUN_ID,
        plan=criteria_plan,
        compiled_plan=criteria_compiled,
        task_draft=task_draft,
        attachment_quality=finalization.attachment_quality,
        solvability=solvability_result.assessment,
        binding_definitions=_binding_definitions(),
        tool_catalog=catalog,
        audit=audit(),
    )
    criteria_calls = criteria_provider.calls.copy()
    assert (
        await criteria_runner.run(
            run_id=RUN_ID,
            plan=criteria_plan,
            compiled_plan=criteria_compiled,
            task_draft=task_draft,
            attachment_quality=finalization.attachment_quality,
            solvability=solvability_result.assessment,
            binding_definitions=_binding_definitions(),
            tool_catalog=catalog,
            audit=audit(),
        )
        == criteria
    )
    assert criteria_provider.calls == criteria_calls
    assert criteria.result.outcome is CriteriaRubricOutcomeV2.SUCCEEDED
    assert criteria.result.rubric_set_ref is not None
    assert criteria.result.evaluator_spec_ref is not None
    assert criteria.result.reference_policy_ref is not None
    assert criteria.result.tool_policy_ref is not None
    rubric_set = criteria_material_store.get_model(
        criteria.result.rubric_set_ref,
        RubricSetV2,
    )
    evaluator_spec = criteria_material_store.get_model(
        criteria.result.evaluator_spec_ref,
        EvaluatorSpecV2,
    )
    reference_policy = criteria_material_store.get_model(
        criteria.result.reference_policy_ref,
        ReferencePolicyV2,
    )
    tool_policy = criteria_material_store.get_model(
        criteria.result.tool_policy_ref,
        ToolPolicyV2,
    )
    criteria_route = criteria_agent.gateway.get_route(
        criteria.result.route_decision_ref,
    )

    authority = _CriteriaAuthority(
        store=store,
        run_ref=store.get_run(RUN_ID).to_ref(),
        result=criteria.result,
        route=criteria_route,
        generator_profile=criteria_generator,
        rubric_set=rubric_set,
        evaluator_spec=evaluator_spec,
        reference_policy=reference_policy,
        tool_policy=tool_policy,
    )
    _, judge_prompt = grading_prompts()
    judge = profile(
        "specialist-chain-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    grading = grading_plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
    )
    grading_registry = build_grading_design_agent_registry(
        config=GradingDesignAgentRegistryConfig(
            prompt_ref=judge_prompt.to_ref(),
            model_policy_ref=grading.model_policy_ref,
        ),
        audit=audit(),
    )
    grading_definition = grading_registry.resolve(
        grading.agent_role,
        "grading-design",
    )
    (
        grading_agent,
        _,
        grading_material_store,
        grading_provider,
    ) = grading_candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=grading,
        proposal=grading_proposal(grading),
        judge_profile=judge,
        agent_definition_ref=grading_definition.to_ref(),
    )
    grading_compiled = GradingDesignPlanCompiler(
        grading_registry,
    ).compile(
        plan=grading,
        policy=store.get_policy(
            store.get_run(RUN_ID).policy_ref.object_id,
        ),
        audit=audit(),
    )
    store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=store.get_run(RUN_ID).run_version,
        plan_kind=PlanKindV2.GRADING_DESIGN,
        plan=grading,
        compiled_plan=grading_compiled,
        audit=audit(),
        idempotency_key="commit-specialist-chain-grading-plan",
    )
    grading_review = _approve_and_resume(
        PlanReviewService(store),
        plan_kind=PlanKindV2.GRADING_DESIGN,
        plan=grading,
        suffix="grading",
    )
    grading_supervisor = ExecutionSupervisor(
        store,
        grading_registry,
        workspace_manager=AgentWorkspaceManager(
            tmp_path / "grading-workspace",
        ),
    )
    grading_runner = SupervisedGradingDesignRunner(
        supervisor=grading_supervisor,
        agent=grading_agent,
    )
    grading_result = await grading_runner.run(
        run_id=RUN_ID,
        plan=grading,
        compiled_plan=grading_compiled,
        criteria_result=criteria.result,
        criteria_route=criteria_route,
        rubric_set=rubric_set,
        evaluator_spec=evaluator_spec,
        reference_policy=reference_policy,
        tool_policy=tool_policy,
        model_authorizations=(),
        evaluated_at=datetime(2026, 8, 7, tzinfo=UTC),
        audit=audit(),
    )
    grading_calls = grading_provider.calls.copy()
    assert (
        await grading_runner.run(
            run_id=RUN_ID,
            plan=grading,
            compiled_plan=grading_compiled,
            criteria_result=criteria.result,
            criteria_route=criteria_route,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
            model_authorizations=(),
            evaluated_at=datetime(2026, 8, 7, tzinfo=UTC),
            audit=audit(),
        )
        == grading_result
    )
    assert grading_provider.calls == grading_calls
    assert grading_result.result.outcome is GradingDesignOutcomeV2.SUCCEEDED
    assert grading_result.result.judge_design_spec_ref is not None
    grading_route = grading_agent.gateway.get_route(
        grading_result.result.route_decision_ref,
    )
    assert grading_route.selected_model_profile_ref == judge.to_ref()
    assert grading_route.selected_model_profile_ref != generator.to_ref()
    assert grading_route.selected_model_profile_ref != criteria_generator.to_ref()
    grading_material_store.get_model(
        grading_result.result.judge_design_spec_ref,
        JudgeDesignSpecV2,
    )

    expected_heads = {
        PlanKindV2.ATTACHMENT_GENERATION: (attachment.subgraph_result.to_ref()),
        PlanKindV2.CRITERIA_RUBRIC: criteria.result.to_ref(),
        PlanKindV2.GRADING_DESIGN: grading_result.result.to_ref(),
    }
    assert {
        kind: store.get_domain_result(
            RUN_ID,
            kind,
        ).result_ref
        for kind in expected_heads
    } == expected_heads
    rebuilt = store.rebuild_current_heads()
    cli = CliRunner()
    parity_service = PlanReviewService(store)
    for expected in (
        criteria_review,
        grading_review,
    ):
        current = parity_service.show(
            expected.request.review_request_id,
        )
        for command in ("show", "export"):
            result = cli.invoke(
                app,
                [
                    "agent",
                    "plan",
                    command,
                    "--factory-store",
                    str(store.path),
                    "--review",
                    expected.request.review_request_id,
                ],
            )
            assert result.exit_code == 0, result.stderr
            assert json.loads(result.stdout) == current.model_dump(
                mode="json",
            )

    assert rebuilt.domain_result_head_count == 3
    assert rebuilt.domain_result_head_count == 3
    assert {
        kind: store.get_domain_result(
            RUN_ID,
            kind,
        ).result_ref
        for kind in expected_heads
    } == expected_heads

    public_values = json.dumps(
        {
            "attachment": attachment.subgraph_result.model_dump(
                mode="json",
            ),
            "criteria": criteria.result.model_dump(mode="json"),
            "grading": grading_result.result.model_dump(mode="json"),
        },
        sort_keys=True,
    ).casefold()
    for marker in (
        "credential",
        "final-answer",
        "grader-rule",
        "hidden-condition",
        "private-reference",
        "raw-trace",
    ):
        assert marker not in public_values


def _approve_and_resume(
    service: PlanReviewService,
    *,
    plan_kind: PlanKindV2,
    plan: CriteriaRubricPlanV2 | GradingDesignPlanV2,
    suffix: str,
):
    opened = service.open_plan(
        run_id=RUN_ID,
        plan_kind=plan_kind,
        requested_by=USER,
        idempotency_key=f"open-specialist-chain-{suffix}",
        audit=audit(),
    )
    service.decide(
        opened.request.review_request_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=plan.plan_version,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by=USER,
            reason_code=f"{suffix.upper()}_PLAN_APPROVED",
            idempotency_key=f"approve-specialist-chain-{suffix}",
        ),
        audit=audit(),
    )
    return service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=plan.plan_version,
            resumed_by=USER,
            idempotency_key=f"resume-specialist-chain-{suffix}",
        ),
        audit=audit(),
    )
