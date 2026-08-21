from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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
from grading_fixtures import (
    plan as grading_plan,
)
from grading_fixtures import (
    policy as grading_policy,
)
from test_criteria_agent import (
    _binding_definitions,
    _catalog,
    _draft,
    _quality,
    _solvability,
)
from test_criteria_agent import (
    _candidate_agent as criteria_candidate_agent,
)
from test_criteria_agent import (
    _plan as criteria_plan,
)
from test_criteria_agent import (
    _proposal as criteria_proposal,
)
from test_domain_plan_review import RUN_ID
from test_domain_plan_review import _setup as domain_setup
from test_evaluator_reference_policy import (
    _authorization,
    _binding_definition,
)
from test_rubric_authoring import EVALUATOR_BINDING

from eval_factory.agent_system.grading_agent import (
    GatewayGradingDesignAgent,
    GatewayGradingDesignPlanningAgent,
    GradingDesignAgentConfig,
    GradingDesignAgentError,
    GradingDesignAgentInputV1,
    GradingDesignPlanningAgentConfig,
)
from eval_factory.agent_system.grading_design import (
    JudgeDesignProposalOutcomeV1,
    JudgeDesignProposalV1,
    JudgeInstructionSelectionV1,
)
from eval_factory.agent_system.grading_material_store import (
    GradingDesignMaterialStore,
)
from eval_factory.agent_system.grading_planning import (
    GradingDesignPlanCompiler,
)
from eval_factory.agent_system.grading_registry import (
    GradingDesignAgentRegistryConfig,
    build_grading_design_agent_registry,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    CriteriaRubricResultV2,
    GradingDesignOutcomeV2,
    GradingDesignPlanV2,
    JudgeDesignSpecV2,
    JudgeDesignValidationOutcomeV2,
    JudgeDesignValidationV2,
    JudgeTaskMappingV2,
    PlanKindV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    ModelCapabilityProfileV2,
    ModelRouteDecisionV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.task import ReferenceMode
from eval_factory.contracts.task_v2 import (
    EvaluatorExecutionModeV2,
    EvaluatorModelDomainV2,
    EvaluatorReferenceDataClassV2,
    EvaluatorSpecV2,
    ReferencePolicyV2,
    RubricSetV2,
    ToolPolicyV2,
    evaluator_reference_grant_ref,
    evaluator_spec_ref,
    reference_policy_ref,
    rubric_set_ref,
    tool_policy_ref,
)
from eval_factory.task_authoring import EvaluatorBindingDefinition


@dataclass(frozen=True, slots=True)
class _CriteriaAuthority:
    store: FactoryControlStore
    run_ref: ObjectRef
    result: CriteriaRubricResultV2
    route: ModelRouteDecisionV2
    generator_profile: ModelCapabilityProfileV2
    rubric_set: RubricSetV2
    evaluator_spec: EvaluatorSpecV2
    reference_policy: ReferencePolicyV2
    tool_policy: ToolPolicyV2


async def _criteria_authority(
    tmp_path: Path,
    *,
    reference_mode: ReferenceMode = ReferenceMode.NONE,
    binding_definitions: tuple[EvaluatorBindingDefinition, ...] | None = None,
) -> _CriteriaAuthority:
    task_draft = _draft()
    quality = _quality()
    solvability = _solvability(quality)
    catalog = _catalog()
    _, source_store, _ = domain_setup(tmp_path / "source-control")
    run_ref = source_store.get_run(RUN_ID).to_ref()
    source_plan = criteria_plan(
        run_ref_value=run_ref,
        task_draft=task_draft,
        quality=quality,
        solvability=solvability,
        catalog=catalog,
        allowed_reference_modes=(reference_mode,),
        selected_reference_mode=reference_mode,
    )
    agent, _, _, generator_profile, store = criteria_candidate_agent(
        tmp_path / "criteria-agent",
        plan=source_plan,
        proposal=criteria_proposal(),
    )
    execution = await agent.author(
        task_ref=ref("agent-task", "criteria-rubric"),
        plan=source_plan,
        task_draft=task_draft,
        attachment_quality=quality,
        solvability=solvability,
        binding_definitions=(
            binding_definitions if binding_definitions is not None else _binding_definitions()
        ),
        tool_catalog=catalog,
        audit=audit(),
    )
    assert execution.rubric_set is not None
    assert execution.evaluator_spec is not None
    assert execution.reference_policy is not None
    assert execution.tool_policy is not None
    store.commit_domain_result(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        result=execution.result,
        idempotency_key=("commit-criteria-result-for-grading"),
    )
    return _CriteriaAuthority(
        store=store,
        run_ref=store.get_run(RUN_ID).to_ref(),
        result=execution.result,
        route=execution.route,
        generator_profile=generator_profile,
        rubric_set=execution.rubric_set,
        evaluator_spec=execution.evaluator_spec,
        reference_policy=execution.reference_policy,
        tool_policy=execution.tool_policy,
    )


def _prompts() -> tuple[PromptTemplateV2, PromptTemplateV2]:
    return (
        prompt(
            task_kind="grading-design-planning",
            agent_role="grading-design-planning-agent",
            input_type="grading-design-planning-input",
            output_type="grading-design-plan",
        ),
        prompt(
            task_kind="grading-design",
            agent_role="grading-design-agent",
            input_type="grading-design-agent-input",
            output_type="judge-design-proposal",
        ),
    )


def _plan(
    authority: _CriteriaAuthority,
    *,
    judge_prompt: PromptTemplateV2,
    allowed_judge_model_profile_refs: tuple[ObjectRef, ...],
    minimum_confidence_basis_points: int = 8000,
    escalate_on_reference_unavailable: bool = True,
    criteria_rubric_result_ref: ObjectRef | None = None,
) -> GradingDesignPlanV2:
    grouped: dict[str, list[str]] = {}
    for criterion in authority.rubric_set.criteria:
        grouped.setdefault(
            criterion.evaluator_binding,
            [],
        ).append(criterion.criterion_id)
    assert len(grouped) == 1
    binding_id, criterion_ids = next(iter(grouped.items()))
    judge_tasks = (
        JudgeTaskMappingV2(
            task_key="judge-main",
            criterion_ids=tuple(sorted(criterion_ids)),
            evaluator_binding_id=binding_id,
            score_weight_basis_points=10_000,
        ),
    )
    return grading_plan(
        run_ref=authority.run_ref,
        criteria_rubric_result_ref=(criteria_rubric_result_ref or authority.result.to_ref()),
        rubric_set_ref=rubric_set_ref(authority.rubric_set),
        evaluator_spec_ref=evaluator_spec_ref(authority.evaluator_spec),
        reference_policy_ref=reference_policy_ref(authority.reference_policy),
        tool_policy_ref=tool_policy_ref(authority.tool_policy),
        generator_model_profile_ref=(authority.generator_profile.to_ref()),
        judge_tasks=judge_tasks,
        allowed_judge_model_profile_refs=(allowed_judge_model_profile_refs),
        minimum_confidence_basis_points=(minimum_confidence_basis_points),
        escalate_on_reference_unavailable=(escalate_on_reference_unavailable),
        prompt_template_ref=judge_prompt.to_ref(),
    )


def _proposal(
    source_plan: GradingDesignPlanV2,
    *,
    outcome: JudgeDesignProposalOutcomeV1 = (JudgeDesignProposalOutcomeV1.PROPOSED),
    confidence_basis_points: int = 9200,
) -> JudgeDesignProposalV1:
    if outcome is not JudgeDesignProposalOutcomeV1.PROPOSED:
        return JudgeDesignProposalV1(
            outcome=outcome,
            confidence_basis_points=confidence_basis_points,
            reason_codes=(
                "REFERENCE_REQUIRED"
                if outcome is JudgeDesignProposalOutcomeV1.ESCALATE_REFERENCE_REQUIRED
                else "UNJUDGEABLE",
            ),
        )
    return JudgeDesignProposalV1(
        outcome=outcome,
        instructions=tuple(
            JudgeInstructionSelectionV1(
                task_key=task.task_key,
                criterion_ids=task.criterion_ids,
                evaluator_binding_id=task.evaluator_binding_id,
                instruction=("Evaluate only the bound criteria and return the reviewed structured fields."),
            )
            for task in source_plan.judge_tasks
        ),
        output_fields=source_plan.required_output_fields,
        confidence_basis_points=confidence_basis_points,
    )


def _candidate_agent(
    tmp_path: Path,
    *,
    authority: _CriteriaAuthority,
    source_plan: GradingDesignPlanV2,
    proposal: JudgeDesignProposalV1,
    judge_profile: ModelCapabilityProfileV2,
    agent_definition_ref: ObjectRef | None = None,
    failed: bool = False,
    malformed: bool = False,
) -> tuple[
    GatewayGradingDesignAgent,
    FactoryPrivateObjectStore,
    GradingDesignMaterialStore,
    DeterministicAttachmentProvider,
]:
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    material_store = GradingDesignMaterialStore(tmp_path / "materials")
    _, judge_prompt = _prompts()
    output_ref = private_store.put_model(
        object_type="judge-design-proposal",
        value=(source_plan if malformed else proposal),
    )
    provider = DeterministicAttachmentProvider(
        {_ref_key(judge_prompt.to_ref()): output_ref},
        failed_prompt_refs=((judge_prompt.to_ref(),) if failed else ()),
    )
    definition_ref = agent_definition_ref or ref("agent-definition", "grading-design")
    gateway = build_gateway(
        tmp_path,
        prompts=(judge_prompt,),
        profiles=(
            authority.generator_profile,
            judge_profile,
        ),
        agent_definition_refs=(definition_ref,),
        provider=provider,
    )
    return (
        GatewayGradingDesignAgent(
            gateway=gateway,
            private_store=private_store,
            material_store=material_store,
            config=GradingDesignAgentConfig(
                prompt=judge_prompt,
                agent_definition_ref=definition_ref,
                allowed_model_profile_refs=(source_plan.allowed_judge_model_profile_refs),
                budget_reservation_ref=ref(
                    "work-model-reservation",
                    "grading-design",
                ),
            ),
        ),
        private_store,
        material_store,
        provider,
    )


def _author_kwargs(
    authority: _CriteriaAuthority,
    source_plan: GradingDesignPlanV2,
) -> dict[str, object]:
    return {
        "task_ref": ref("agent-task", "grading-design"),
        "plan": source_plan,
        "criteria_result": authority.result,
        "criteria_route": authority.route,
        "rubric_set": authority.rubric_set,
        "evaluator_spec": authority.evaluator_spec,
        "reference_policy": authority.reference_policy,
        "tool_policy": authority.tool_policy,
        "model_authorizations": (),
        "evaluated_at": datetime(2026, 8, 6, tzinfo=UTC),
        "audit": audit(),
    }


@pytest.mark.asyncio
async def test_grading_planning_agent_compiles_and_replays(
    tmp_path: Path,
) -> None:
    _, judge_prompt = _prompts()
    generator = profile(
        "grading-generator",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    judge = profile(
        "grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    planning_profile = profile(
        "grading-planning",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = grading_plan(
        generator_model_profile_ref=generator.to_ref(),
        allowed_judge_model_profile_refs=(
            generator.to_ref(),
            judge.to_ref(),
        ),
        prompt_template_ref=judge_prompt.to_ref(),
    )
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    planning_prompt, _ = _prompts()
    output_ref = private_store.put_model(
        object_type="grading-design-plan",
        value=source_plan,
    )
    provider = DeterministicAttachmentProvider({_ref_key(planning_prompt.to_ref()): output_ref})
    definition_ref = ref(
        "agent-definition",
        "grading-design-planning",
    )
    registry = build_grading_design_agent_registry(
        config=GradingDesignAgentRegistryConfig(
            prompt_ref=judge_prompt.to_ref(),
            model_policy_ref=source_plan.model_policy_ref,
        ),
        audit=audit(),
    )
    agent = GatewayGradingDesignPlanningAgent(
        gateway=build_gateway(
            tmp_path,
            prompts=(planning_prompt,),
            profiles=(planning_profile,),
            agent_definition_refs=(definition_ref,),
            provider=provider,
        ),
        private_store=private_store,
        compiler=GradingDesignPlanCompiler(registry),
        config=GradingDesignPlanningAgentConfig(
            prompt=planning_prompt,
            agent_definition_ref=definition_ref,
            allowed_model_profile_refs=(planning_profile.to_ref(),),
            budget_reservation_ref=ref(
                "work-model-reservation",
                "grading-design-planning",
            ),
        ),
    )

    first = await agent.propose(
        task_ref=ref("agent-task", "grading-design-planning"),
        plan_template=source_plan,
        policy=grading_policy(),
        audit=audit(),
    )
    replay = await agent.propose(
        task_ref=ref("agent-task", "grading-design-planning"),
        plan_template=source_plan,
        policy=grading_policy(),
        audit=audit(),
    )

    assert first.plan == source_plan
    assert first.compiled_plan.source_plan_ref == source_plan.to_ref()
    assert replay == first
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_grading_agent_uses_independent_judge_and_replays(
    tmp_path: Path,
) -> None:
    authority = await _criteria_authority(tmp_path / "criteria")
    _, judge_prompt = _prompts()
    judge = profile(
        "grading-design-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(
            authority.generator_profile.to_ref(),
            judge.to_ref(),
        ),
    )
    agent, private_store, material_store, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
    )

    first = await agent.author(**_author_kwargs(authority, source_plan))
    replay = await agent.author(**_author_kwargs(authority, source_plan))

    assert first.result.outcome is GradingDesignOutcomeV2.SUCCEEDED
    assert first.route is not None
    assert first.route.selected_model_profile_ref == judge.to_ref()
    assert first.design_spec is not None
    assert first.validation.outcome is (JudgeDesignValidationOutcomeV2.VALID)
    generator_candidate = next(
        candidate
        for candidate in first.route.candidates
        if candidate.model_profile_ref == authority.generator_profile.to_ref()
    )
    assert tuple(code.value for code in generator_candidate.rejection_codes) == ("GENERATOR_JUDGE_COLLISION",)
    assert replay == first
    assert len(provider.calls) == 1
    assert (
        material_store.get_model(
            first.design_spec.to_ref(),
            JudgeDesignSpecV2,
        )
        == first.design_spec
    )
    assert (
        material_store.get_model(
            first.validation.to_ref(),
            JudgeDesignValidationV2,
        )
        == first.validation
    )
    rendering = private_store.get_model(
        provider.calls[0][0].prompt_rendering_ref,
        GradingDesignAgentInputV1,
    )
    assert rendering.plan_ref == source_plan.to_ref()
    assert rendering.criteria_result_ref == authority.result.to_ref()
    public_result = first.result.model_dump_json()
    assert "Evaluate only" not in public_result
    assert "private-reference" not in public_result
    assert "grader-rule" not in public_result


@pytest.mark.asyncio
async def test_grading_agent_blocks_generator_collision_before_provider(
    tmp_path: Path,
) -> None:
    authority = await _criteria_authority(tmp_path / "criteria")
    _, judge_prompt = _prompts()
    judge = profile(
        "unused-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(authority.generator_profile.to_ref(),),
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
    )

    execution = await agent.author(**_author_kwargs(authority, source_plan))

    assert execution.result.outcome is (GradingDesignOutcomeV2.BLOCKED_POLICY)
    assert execution.result.reason_codes == ("GENERATOR_JUDGE_COLLISION",)
    assert execution.route is None
    assert execution.design_spec is None
    assert provider.calls == []


@pytest.mark.asyncio
async def test_grading_agent_blocks_missing_judge_capability_before_provider(
    tmp_path: Path,
) -> None:
    authority = await _criteria_authority(tmp_path / "criteria")
    _, judge_prompt = _prompts()
    judge = profile(
        "incapable-grading-judge",
        capabilities=("structured-output",),
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
    )

    execution = await agent.author(**_author_kwargs(authority, source_plan))

    assert execution.result.outcome is (GradingDesignOutcomeV2.BLOCKED_CAPABILITY)
    assert execution.result.reason_codes == ("NO_ELIGIBLE_JUDGE_MODEL",)
    assert execution.route is None
    assert execution.design_spec is None
    assert provider.calls == []


@pytest.mark.asyncio
async def test_grading_agent_persists_failed_receipt_without_fallback(
    tmp_path: Path,
) -> None:
    authority = await _criteria_authority(tmp_path / "criteria")
    _, judge_prompt = _prompts()
    judge = profile(
        "failed-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
        failed=True,
    )

    first = await agent.author(**_author_kwargs(authority, source_plan))
    replay = await agent.author(**_author_kwargs(authority, source_plan))

    assert first.result.outcome is (GradingDesignOutcomeV2.BLOCKED_CAPABILITY)
    assert first.result.gateway_receipt_ref is not None
    assert first.design_spec is None
    assert replay == first
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("proposal_outcome", "result_outcome"),
    (
        (
            JudgeDesignProposalOutcomeV1.ABSTAIN,
            GradingDesignOutcomeV2.ABSTAINED,
        ),
        (
            JudgeDesignProposalOutcomeV1.ESCALATE_REFERENCE_REQUIRED,
            GradingDesignOutcomeV2.ESCALATED,
        ),
    ),
)
async def test_grading_agent_returns_typed_semantic_non_success(
    tmp_path: Path,
    proposal_outcome: JudgeDesignProposalOutcomeV1,
    result_outcome: GradingDesignOutcomeV2,
) -> None:
    authority = await _criteria_authority(tmp_path / "criteria")
    _, judge_prompt = _prompts()
    judge = profile(
        f"grading-{proposal_outcome.value.casefold()}",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(
            source_plan,
            outcome=proposal_outcome,
        ),
        judge_profile=judge,
    )

    execution = await agent.author(**_author_kwargs(authority, source_plan))

    assert execution.result.outcome is result_outcome
    assert execution.design_spec is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_grading_agent_persists_only_opaque_reference_grant(
    tmp_path: Path,
) -> None:
    reference = ref(
        "structured-expectation",
        "grading-reference",
    )
    authority = await _criteria_authority(
        tmp_path / "criteria",
        reference_mode=ReferenceMode.STRUCTURED_EXPECTATIONS,
        binding_definitions=(
            _binding_definition(
                binding_id=EVALUATOR_BINDING,
                principal_id="principal://evaluator/grading",
                reference_refs=(reference,),
            ),
        ),
    )
    _, judge_prompt = _prompts()
    judge = profile(
        "reference-grant-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
    )
    agent, _, material_store, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
    )

    execution = await agent.author(**_author_kwargs(authority, source_plan))

    assert execution.result.outcome is (GradingDesignOutcomeV2.SUCCEEDED)
    assert execution.design_spec is not None
    assert len(execution.reference_grants) == 1
    grant = execution.reference_grants[0]
    grant_ref = evaluator_reference_grant_ref(grant)
    assert grant.reference_refs == (reference,)
    assert execution.design_spec.reference_grant_refs == (grant_ref,)
    assert (
        material_store.get_model(
            grant_ref,
            type(grant),
        )
        == grant
    )
    public_result = execution.result.model_dump_json()
    assert "structured-expectation" not in public_result
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_grading_agent_abstains_for_human_only_evaluator(
    tmp_path: Path,
) -> None:
    authority = await _criteria_authority(
        tmp_path / "criteria",
        reference_mode=ReferenceMode.HUMAN_ONLY,
        binding_definitions=(
            _binding_definition(
                binding_id=EVALUATOR_BINDING,
                execution_mode=(EvaluatorExecutionModeV2.HUMAN_ONLY),
                reference_refs=(
                    ref(
                        "human-only-reference",
                        "grading-human-only",
                    ),
                ),
            ),
        ),
    )
    _, judge_prompt = _prompts()
    judge = profile(
        "human-only-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
    )

    execution = await agent.author(**_author_kwargs(authority, source_plan))

    assert execution.result.outcome is (GradingDesignOutcomeV2.ABSTAINED)
    assert execution.result.reason_codes == ("HUMAN_ONLY_EVALUATOR",)
    assert execution.design_spec is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("escalate", "expected_outcome", "reason"),
    (
        (
            True,
            GradingDesignOutcomeV2.ESCALATED,
            "REFERENCE_ACCESS_REQUIRED",
        ),
        (
            False,
            GradingDesignOutcomeV2.BLOCKED_POLICY,
            "REFERENCE_ACCESS_DENIED",
        ),
    ),
)
async def test_grading_agent_closes_missing_model_authorization(
    tmp_path: Path,
    escalate: bool,
    expected_outcome: GradingDesignOutcomeV2,
    reason: str,
) -> None:
    reference = ref(
        "structured-expectation",
        "model-grading-reference",
    )
    authority = await _criteria_authority(
        tmp_path / "criteria",
        reference_mode=ReferenceMode.STRUCTURED_EXPECTATIONS,
        binding_definitions=(
            _binding_definition(
                binding_id=EVALUATOR_BINDING,
                execution_mode=EvaluatorExecutionModeV2.MODEL,
                principal_id="principal://evaluator/model-grading",
                model_profile_ref=ref(
                    "model-profile",
                    "r4-evaluator",
                ),
                reference_refs=(reference,),
            ),
        ),
    )
    _, judge_prompt = _prompts()
    judge = profile(
        f"authorization-{escalate}-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
        escalate_on_reference_unavailable=escalate,
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
    )

    execution = await agent.author(**_author_kwargs(authority, source_plan))

    assert execution.result.outcome is expected_outcome
    assert execution.result.reason_codes == (reason,)
    assert execution.design_spec is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_grading_agent_requires_authorized_evaluator_model(
    tmp_path: Path,
) -> None:
    judge = profile(
        "authorized-model-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    reference = ref(
        "structured-expectation",
        "authorized-model-reference",
    )
    evaluator_model_ref = ref(
        "model-profile",
        "authorized-r4-evaluator",
    )
    authority = await _criteria_authority(
        tmp_path / "criteria",
        reference_mode=ReferenceMode.STRUCTURED_EXPECTATIONS,
        binding_definitions=(
            _binding_definition(
                binding_id=EVALUATOR_BINDING,
                execution_mode=EvaluatorExecutionModeV2.MODEL,
                principal_id=("principal://evaluator/authorized-model"),
                model_profile_ref=evaluator_model_ref,
                reference_refs=(reference,),
            ),
        ),
    )
    _, judge_prompt = _prompts()
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
    )
    authorization = _authorization(
        model_profile_ref=evaluator_model_ref,
        domain=EvaluatorModelDomainV2.INTERNAL_APPROVED,
        data_class=(EvaluatorReferenceDataClassV2.RESTRICTED_EVAL_CONTROL),
        reference_refs=(reference,),
        valid_from=datetime(2026, 8, 5, tzinfo=UTC),
        expires_at=datetime(2026, 8, 7, tzinfo=UTC),
    )
    kwargs = _author_kwargs(authority, source_plan)
    kwargs["model_authorizations"] = (authorization,)

    execution = await agent.author(**kwargs)

    assert execution.result.outcome is (GradingDesignOutcomeV2.SUCCEEDED)
    assert execution.design_spec is not None
    assert len(execution.reference_grants) == 1
    assert execution.reference_grants[0].model_profile_ref == evaluator_model_ref
    assert execution.design_spec.selected_judge_model_profile_ref == judge.to_ref()
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_grading_agent_abstains_on_low_confidence(
    tmp_path: Path,
) -> None:
    authority = await _criteria_authority(tmp_path / "criteria")
    _, judge_prompt = _prompts()
    judge = profile(
        "low-confidence-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
        minimum_confidence_basis_points=9000,
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(
            source_plan,
            confidence_basis_points=8999,
        ),
        judge_profile=judge,
    )

    execution = await agent.author(**_author_kwargs(authority, source_plan))

    assert execution.result.outcome is (GradingDesignOutcomeV2.ABSTAINED)
    assert execution.result.reason_codes == ("JUDGE_CONFIDENCE_INSUFFICIENT",)
    assert execution.design_spec is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_grading_agent_rejects_stale_sources_before_provider(
    tmp_path: Path,
) -> None:
    authority = await _criteria_authority(tmp_path / "criteria")
    _, judge_prompt = _prompts()
    judge = profile(
        "stale-source-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
        criteria_rubric_result_ref=ref(
            "criteria-rubric-result",
            "stale",
        ),
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
    )

    with pytest.raises(
        GradingDesignAgentError,
        match="stale",
    ):
        await agent.author(**_author_kwargs(authority, source_plan))

    assert provider.calls == []


@pytest.mark.asyncio
async def test_grading_agent_rejects_stale_prompt_before_provider(
    tmp_path: Path,
) -> None:
    authority = await _criteria_authority(tmp_path / "criteria")
    stale_prompt = prompt(
        task_kind="grading-design-stale",
        agent_role="grading-design-agent",
        input_type="grading-design-agent-input",
        output_type="judge-design-proposal",
    )
    judge = profile(
        "stale-prompt-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=stale_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
    )

    with pytest.raises(
        GradingDesignAgentError,
        match="configuration is stale",
    ):
        await agent.author(**_author_kwargs(authority, source_plan))

    assert provider.calls == []


@pytest.mark.asyncio
async def test_grading_agent_rejects_malformed_provider_output(
    tmp_path: Path,
) -> None:
    authority = await _criteria_authority(tmp_path / "criteria")
    _, judge_prompt = _prompts()
    judge = profile(
        "malformed-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    source_plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=(judge.to_ref(),),
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "grading",
        authority=authority,
        source_plan=source_plan,
        proposal=_proposal(source_plan),
        judge_profile=judge,
        malformed=True,
    )

    with pytest.raises(
        GradingDesignAgentError,
        match="malformed",
    ):
        await agent.author(**_author_kwargs(authority, source_plan))

    assert len(provider.calls) == 1


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )
