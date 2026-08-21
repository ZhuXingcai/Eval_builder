from __future__ import annotations

from pathlib import Path

import pytest
from attachment_gateway_fixtures import (
    DeterministicAttachmentProvider,
    audit,
    build_gateway,
    profile,
    prompt,
    ref,
)
from test_criteria_planning import (
    _policy,
    _registry,
)
from test_domain_plan_review import (
    RUN_ID,
)
from test_domain_plan_review import (
    _setup as _domain_setup,
)
from test_evaluator_reference_policy import (
    _binding_definition,
)
from test_rubric_authoring import (
    DEPENDENCY_ID,
    EVALUATOR_BINDING,
    PROMPT_REQUIREMENT_ID,
    SECOND_REQUIREMENT_ID,
    _draft,
)
from test_tool_policy import (
    _catalog,
)

from eval_factory.agent_system.criteria_agent import (
    CriteriaRubricAgentConfig,
    CriteriaRubricAgentError,
    CriteriaRubricPlanningAgentConfig,
    CriteriaRubricProposalV1,
    GatewayCriteriaRubricAgent,
    GatewayCriteriaRubricPlanningAgent,
)
from eval_factory.agent_system.criteria_material_store import (
    CriteriaRubricMaterialStore,
)
from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityAssessmentV2,
    AttachmentQualityOutcomeV2,
    AttachmentSubgraphOutcomeV2,
    AttachmentSubgraphResultV2,
    CriteriaRubricGoalV2,
    CriteriaRubricOutcomeV2,
    CriteriaRubricPlanV2,
    PlanKindV2,
    SolvabilityAssessmentV2,
    SolvabilityOutcomeV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.task import ReferenceMode, RubricVisibility
from eval_factory.contracts.task_v2 import (
    RubricJudgedObjectKindV2,
    rubric_set_ref,
    task_draft_ref,
)
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    RubricAuthoringOutcome,
    RubricAuthoringReason,
    RubricCriterionSelection,
    tool_capability_catalog_ref,
)


def _attachment_result(
    plan_ref: ObjectRef | None = None,
) -> AttachmentSubgraphResultV2:
    work_result_ref = ref(
        "agent-result-envelope",
        "criteria",
    )
    return AttachmentSubgraphResultV2.create(
        result_id="attachment-subgraph-result://criteria",
        plan_ref=(
            plan_ref
            or ref(
                "attachment-generation-plan",
                "criteria",
            )
        ),
        work_result_refs=(work_result_ref,),
        succeeded_work_result_refs=(work_result_ref,),
        retryable_work_result_refs=(),
        blocked_work_result_refs=(),
        cancelled_work_result_refs=(),
        outcome=AttachmentSubgraphOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=audit(),
    )


def _quality(
    attachment_result: AttachmentSubgraphResultV2 | None = None,
) -> AttachmentQualityAssessmentV2:
    attachment_result = attachment_result or _attachment_result()
    return AttachmentQualityAssessmentV2.create(
        assessment_id="attachment-quality-assessment://criteria",
        attachment_subgraph_result_ref=attachment_result.to_ref(),
        item_quality_result_ref=ref(
            "item-quality-compilation-result",
            "criteria",
        ),
        validator_result_refs=(
            ref(
                "deterministic-item-validation-result",
                "criteria",
            ),
        ),
        outcome=AttachmentQualityOutcomeV2.PASSED,
        nonwaivable=False,
        reason_codes=(),
        audit=audit(),
    )


def _solvability(
    quality: AttachmentQualityAssessmentV2,
) -> SolvabilityAssessmentV2:
    return SolvabilityAssessmentV2.create(
        assessment_id="solvability-assessment://criteria",
        attachment_subgraph_result_ref=(quality.attachment_subgraph_result_ref),
        quality_assessment_ref=quality.to_ref(),
        route_decision_ref=ref(
            "model-route-decision",
            "solvability",
        ),
        gateway_receipt_ref=ref(
            "gateway-receipt",
            "solvability",
        ),
        evidence_refs=quality.validator_result_refs,
        outcome=SolvabilityOutcomeV2.SOLVABLE,
        reason_codes=(),
        audit=audit(),
    )


def _goals() -> tuple[CriteriaRubricGoalV2, ...]:
    return (
        CriteriaRubricGoalV2(
            goal_id="criteria-goal://response",
            goal_summary="Judge the visible response.",
            judged_object_kind=(RubricJudgedObjectKindV2.CONTESTANT_RESPONSE),
            prompt_requirement_ids=(PROMPT_REQUIREMENT_ID,),
            evaluator_binding_id=EVALUATOR_BINDING,
            weight_basis_points=4000,
        ),
        CriteriaRubricGoalV2(
            goal_id="criteria-goal://tool",
            goal_summary="Judge the allowed tool behavior.",
            judged_object_kind=RubricJudgedObjectKindV2.TOOL_BEHAVIOR,
            prompt_requirement_ids=(PROMPT_REQUIREMENT_ID,),
            allowed_tool_ids=("file-read",),
            evaluator_binding_id=EVALUATOR_BINDING,
            weight_basis_points=3000,
        ),
        CriteriaRubricGoalV2(
            goal_id="criteria-goal://workspace",
            goal_summary="Judge the input workspace state.",
            judged_object_kind=RubricJudgedObjectKindV2.WORKSPACE_STATE,
            prompt_requirement_ids=(SECOND_REQUIREMENT_ID,),
            attachment_dependency_ids=(DEPENDENCY_ID,),
            evaluator_binding_id=EVALUATOR_BINDING,
            weight_basis_points=3000,
        ),
    )


def _plan(
    *,
    run_ref_value=None,
    task_draft=None,
    quality=None,
    solvability=None,
    catalog=None,
    allowed_reference_modes: tuple[ReferenceMode, ...] = (ReferenceMode.NONE,),
    selected_reference_mode: ReferenceMode = ReferenceMode.NONE,
) -> CriteriaRubricPlanV2:
    task_draft = task_draft or _draft()
    quality = quality or _quality()
    solvability = solvability or _solvability(quality)
    catalog = catalog or _catalog()
    registry = _registry()
    definition = registry.resolve(
        "criteria-rubric-agent",
        "criteria-rubric",
    )
    return CriteriaRubricPlanV2.create(
        plan_id="criteria-rubric-plan://gateway",
        run_ref=run_ref_value or ref("factory-run", "criteria"),
        plan_version=1,
        predecessor_plan_ref=None,
        task_draft_ref=task_draft_ref(task_draft),
        attachment_quality_ref=quality.to_ref(),
        solvability_ref=solvability.to_ref(),
        allowed_prompt_requirement_ids=(task_draft.prompt_requirement_ids),
        required_prompt_requirement_ids=(task_draft.prompt_requirement_ids),
        allowed_attachment_dependency_ids=tuple(
            value.dependency_id for value in task_draft.attachment_dependencies
        ),
        required_attachment_dependency_ids=(DEPENDENCY_ID,),
        allowed_task_tool_ids=task_draft.allowed_tools,
        required_task_tool_ids=("file-read",),
        criterion_goals=_goals(),
        allowed_evaluator_binding_ids=(EVALUATOR_BINDING,),
        allowed_reference_modes=allowed_reference_modes,
        selected_reference_mode=selected_reference_mode,
        tool_catalog_ref=tool_capability_catalog_ref(catalog),
        agent_role=definition.agent_role,
        required_capability_ids=("agent-capability://criteria-rubric",),
        specialist_tool_ids=("rubric-candidate-read",),
        data_classifications=(
            "RESTRICTED_EVALUATOR_CONTROL",
            "RESTRICTED_TRACE_DERIVED",
        ),
        prompt_template_ref=definition.prompt_template_ref,
        model_policy_ref=definition.model_policy_ref,
        acceptance_check_refs=definition.validator_refs,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=16_000,
        max_cost_micro_usd=500_000,
        audit=audit(),
    )


def _proposal(
    *,
    outcome: RubricAuthoringOutcome = RubricAuthoringOutcome.COMPILED,
    reasons: frozenset[RubricAuthoringReason] = frozenset(),
) -> CriteriaRubricProposalV1:
    criteria = ()
    if outcome is RubricAuthoringOutcome.COMPILED:
        criteria = tuple(
            RubricCriterionSelection(
                selection_id=goal.goal_id,
                judged_object_id=(f"judged-object://{goal.goal_id.rsplit('://', 1)[-1]}"),
                judged_object_kind=goal.judged_object_kind,
                judged_object_description=goal.goal_summary,
                description=f"Criterion for {goal.goal_summary}",
                weight=goal.weight_basis_points / 10_000,
                prompt_requirement_ids=(goal.prompt_requirement_ids),
                attachment_dependency_ids=(goal.attachment_dependency_ids),
                allowed_tool_ids=goal.allowed_tool_ids,
                visibility=RubricVisibility.EVALUATOR_ONLY,
                evaluator_binding=goal.evaluator_binding_id,
            )
            for goal in _goals()
        )
    return CriteriaRubricProposalV1(
        outcome=outcome,
        criteria=criteria,
        unresolved_reasons=reasons,
        confidence_basis_points=9200 if criteria else 0,
    )


def _prompts():
    return (
        prompt(
            task_kind="criteria-rubric-planning",
            agent_role="criteria-rubric-planning-agent",
            input_type="criteria-rubric-planning-input",
            output_type="criteria-rubric-plan",
        ),
        prompt(
            task_kind="criteria-rubric",
            agent_role="criteria-rubric-agent",
            input_type="criteria-rubric-agent-input",
            output_type="criteria-rubric-proposal",
        ),
    )


def _binding_definitions() -> tuple[EvaluatorBindingDefinition, ...]:
    return (
        _binding_definition(
            binding_id=EVALUATOR_BINDING,
            principal_id="principal://evaluator/criteria-rubric",
        ),
    )


def _candidate_agent(
    tmp_path: Path,
    *,
    plan: CriteriaRubricPlanV2,
    proposal: CriteriaRubricProposalV1,
    failed: bool = False,
    malformed: bool = False,
    control_store: FactoryControlStore | None = None,
):
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    material_store = CriteriaRubricMaterialStore(
        tmp_path / "materials",
    )
    _, candidate_prompt = _prompts()
    proposal_ref = private_store.put_model(
        object_type="criteria-rubric-proposal",
        value=(plan if malformed else proposal),
    )
    provider = DeterministicAttachmentProvider(
        {_ref_key(candidate_prompt.to_ref()): proposal_ref},
        failed_prompt_refs=((candidate_prompt.to_ref(),) if failed else ()),
    )
    model = profile(
        "criteria-rubric",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    definition_ref = ref(
        "agent-definition",
        "criteria-rubric",
    )
    gateway = build_gateway(
        tmp_path,
        prompts=(candidate_prompt,),
        profiles=(model,),
        agent_definition_refs=(definition_ref,),
        provider=provider,
    )
    if control_store is None:
        _, store, _ = _domain_setup(tmp_path / "control")
    else:
        store = control_store
    compiler = CriteriaRubricPlanCompiler(_registry())
    compiled = compiler.compile(
        plan=plan,
        policy=store.get_policy(store.get_run(RUN_ID).policy_ref.object_id),
        audit=audit(),
    )
    store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=store.get_run(RUN_ID).run_version,
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        plan=plan,
        compiled_plan=compiled,
        audit=audit(),
        idempotency_key="commit-criteria-plan-agent",
    )
    agent = GatewayCriteriaRubricAgent(
        gateway=gateway,
        private_store=private_store,
        material_store=material_store,
        config=CriteriaRubricAgentConfig(
            prompt=candidate_prompt,
            agent_definition_ref=definition_ref,
            allowed_model_profile_refs=(model.to_ref(),),
            budget_reservation_ref=ref(
                "work-model-reservation",
                "criteria-rubric",
            ),
        ),
    )
    return agent, material_store, provider, model, store


def _ref_key(value) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


@pytest.mark.asyncio
async def test_planning_agent_routes_compiles_and_replays(
    tmp_path: Path,
) -> None:
    plan = _plan()
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    planning_prompt, _ = _prompts()
    output_ref = private_store.put_model(
        object_type="criteria-rubric-plan",
        value=plan,
    )
    provider = DeterministicAttachmentProvider({_ref_key(planning_prompt.to_ref()): output_ref})
    model = profile(
        "criteria-rubric-planning",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    definition_ref = ref(
        "agent-definition",
        "criteria-rubric-planning",
    )
    agent = GatewayCriteriaRubricPlanningAgent(
        gateway=build_gateway(
            tmp_path,
            prompts=(planning_prompt,),
            profiles=(model,),
            agent_definition_refs=(definition_ref,),
            provider=provider,
        ),
        private_store=private_store,
        compiler=CriteriaRubricPlanCompiler(_registry()),
        config=CriteriaRubricPlanningAgentConfig(
            prompt=planning_prompt,
            agent_definition_ref=definition_ref,
            allowed_model_profile_refs=(model.to_ref(),),
            budget_reservation_ref=ref(
                "work-model-reservation",
                "criteria-rubric-planning",
            ),
        ),
    )

    first = await agent.propose(
        task_ref=ref("agent-task", "criteria-planning"),
        plan_template=plan,
        policy=_policy(),
        audit=audit(),
    )
    replay = await agent.propose(
        task_ref=ref("agent-task", "criteria-planning"),
        plan_template=plan,
        policy=_policy(),
        audit=audit(),
    )

    assert first.plan == plan
    assert first.compiled_plan.source_plan_ref == plan.to_ref()
    assert replay == first
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_candidate_agent_composes_complete_r4_chain_and_replays(
    tmp_path: Path,
) -> None:
    task_draft = _draft()
    quality = _quality()
    solvability = _solvability(quality)
    catalog = _catalog()
    _, store, _ = _domain_setup(tmp_path / "source-control")
    plan = _plan(
        run_ref_value=store.get_run(RUN_ID).to_ref(),
        task_draft=task_draft,
        quality=quality,
        solvability=solvability,
        catalog=catalog,
    )
    agent, material_store, provider, model, _ = _candidate_agent(
        tmp_path / "agent",
        plan=plan,
        proposal=_proposal(),
    )

    first = await agent.author(
        task_ref=ref("agent-task", "criteria-rubric"),
        plan=plan,
        task_draft=task_draft,
        attachment_quality=quality,
        solvability=solvability,
        binding_definitions=_binding_definitions(),
        tool_catalog=catalog,
        audit=audit(),
    )
    replay = await agent.author(
        task_ref=ref("agent-task", "criteria-rubric"),
        plan=plan,
        task_draft=task_draft,
        attachment_quality=quality,
        solvability=solvability,
        binding_definitions=_binding_definitions(),
        tool_catalog=catalog,
        audit=audit(),
    )

    assert first.result.outcome is CriteriaRubricOutcomeV2.SUCCEEDED
    assert first.rubric_set is not None
    assert first.evaluator_spec is not None
    assert first.reference_policy is not None
    assert first.tool_policy is not None
    assert first.contestant_tool_policy is not None
    assert first.route.selected_model_profile_ref == model.to_ref()
    assert replay == first
    assert len(provider.calls) == 1
    assert (
        material_store.get_model(
            rubric_set_ref(first.rubric_set),
            type(first.rubric_set),
        )
        == first.rubric_set
    )
    serialized = first.result.model_dump_json()
    assert "Criterion for" not in serialized
    assert "private-reference" not in serialized
    contestant = first.contestant_tool_policy.model_dump_json()
    assert "evaluator" not in contestant
    assert "private-reference" not in contestant


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("proposal", "failed", "outcome"),
    (
        (
            _proposal(
                outcome=RubricAuthoringOutcome.ABSTAIN,
                reasons=frozenset({RubricAuthoringReason.AMBIGUOUS_RUBRIC}),
            ),
            False,
            CriteriaRubricOutcomeV2.ABSTAINED,
        ),
        (
            _proposal(),
            True,
            CriteriaRubricOutcomeV2.BLOCKED_CAPABILITY,
        ),
    ),
)
async def test_candidate_agent_returns_typed_non_success_without_fallback(
    tmp_path: Path,
    proposal: CriteriaRubricProposalV1,
    failed: bool,
    outcome: CriteriaRubricOutcomeV2,
) -> None:
    task_draft = _draft()
    quality = _quality()
    solvability = _solvability(quality)
    catalog = _catalog()
    _, source_store, _ = _domain_setup(tmp_path / "source-control")
    plan = _plan(
        run_ref_value=source_store.get_run(RUN_ID).to_ref(),
        task_draft=task_draft,
        quality=quality,
        solvability=solvability,
        catalog=catalog,
    )
    agent, _, provider, _, _ = _candidate_agent(
        tmp_path / "agent",
        plan=plan,
        proposal=proposal,
        failed=failed,
    )

    execution = await agent.author(
        task_ref=ref("agent-task", "criteria-rubric"),
        plan=plan,
        task_draft=task_draft,
        attachment_quality=quality,
        solvability=solvability,
        binding_definitions=_binding_definitions(),
        tool_catalog=catalog,
        audit=audit(),
    )

    assert execution.result.outcome is outcome
    assert execution.rubric_set is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_candidate_agent_rejects_stale_upstream_before_provider(
    tmp_path: Path,
) -> None:
    task_draft = _draft()
    quality = _quality()
    solvability = _solvability(quality)
    catalog = _catalog()
    _, source_store, _ = _domain_setup(tmp_path / "source-control")
    plan = _plan(
        run_ref_value=source_store.get_run(RUN_ID).to_ref(),
        task_draft=task_draft,
        quality=quality,
        solvability=solvability,
        catalog=catalog,
    )
    agent, _, provider, _, _ = _candidate_agent(
        tmp_path / "agent",
        plan=plan,
        proposal=_proposal(),
    )
    stale = task_draft.model_copy(update={"visible_prompt": task_draft.visible_prompt + " changed"})

    with pytest.raises(
        CriteriaRubricAgentError,
        match="stale",
    ):
        await agent.author(
            task_ref=ref("agent-task", "criteria-rubric"),
            plan=plan,
            task_draft=stale,
            attachment_quality=quality,
            solvability=solvability,
            binding_definitions=_binding_definitions(),
            tool_catalog=catalog,
            audit=audit(),
        )

    assert provider.calls == []


@pytest.mark.asyncio
async def test_candidate_agent_rejects_proposal_that_widens_reviewed_goal(
    tmp_path: Path,
) -> None:
    task_draft = _draft()
    quality = _quality()
    solvability = _solvability(quality)
    catalog = _catalog()
    _, source_store, _ = _domain_setup(tmp_path / "source-control")
    plan = _plan(
        run_ref_value=source_store.get_run(RUN_ID).to_ref(),
        task_draft=task_draft,
        quality=quality,
        solvability=solvability,
        catalog=catalog,
    )
    proposal = _proposal()
    changed = proposal.criteria[0].model_copy(update={"weight": 0.5})
    widened = proposal.model_copy(update={"criteria": (changed, *proposal.criteria[1:])})
    agent, _, provider, _, _ = _candidate_agent(
        tmp_path / "agent",
        plan=plan,
        proposal=widened,
    )

    with pytest.raises(
        CriteriaRubricAgentError,
        match="widens",
    ):
        await agent.author(
            task_ref=ref("agent-task", "criteria-rubric"),
            plan=plan,
            task_draft=task_draft,
            attachment_quality=quality,
            solvability=solvability,
            binding_definitions=_binding_definitions(),
            tool_catalog=catalog,
            audit=audit(),
        )

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_candidate_agent_rejects_malformed_provider_output(
    tmp_path: Path,
) -> None:
    task_draft = _draft()
    quality = _quality()
    solvability = _solvability(quality)
    catalog = _catalog()
    _, source_store, _ = _domain_setup(tmp_path / "source-control")
    plan = _plan(
        run_ref_value=source_store.get_run(RUN_ID).to_ref(),
        task_draft=task_draft,
        quality=quality,
        solvability=solvability,
        catalog=catalog,
    )
    agent, _, provider, _, _ = _candidate_agent(
        tmp_path / "agent",
        plan=plan,
        proposal=_proposal(),
        malformed=True,
    )

    with pytest.raises(
        CriteriaRubricAgentError,
        match="malformed",
    ):
        await agent.author(
            task_ref=ref("agent-task", "criteria-rubric"),
            plan=plan,
            task_draft=task_draft,
            attachment_quality=quality,
            solvability=solvability,
            binding_definitions=_binding_definitions(),
            tool_catalog=catalog,
            audit=audit(),
        )

    assert len(provider.calls) == 1
