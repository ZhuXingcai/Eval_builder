from __future__ import annotations

from pathlib import Path

import pytest
from attachment_gateway_fixtures import (
    DeterministicAttachmentProvider,
    build_gateway,
    profile,
    prompt,
    ref,
)
from attachment_gateway_fixtures import (
    audit as attachment_audit,
)
from test_attachment_supervised_execution import (
    _prepared as prepare_attachment,
)
from test_criteria_agent import (
    _binding_definitions,
    _catalog,
    _draft,
)
from test_criteria_planning import _registry as criteria_registry
from test_deterministic_validation import (
    _FakeValidationFacade,
    _reference_set,
)
from test_factory_dataset_store import (
    _binding,
    _create_run,
)
from test_item_quality import _task_contract_set
from test_semantic_review import (
    _AcceptingBackend,
    _DynamicResolver,
    _NoRepairBackend,
)
from test_semantic_review import (
    _policy as semantic_review_policy,
)
from test_semantic_review import (
    _sources as semantic_review_sources,
)
from test_semantic_review import (
    _store as quality_store,
)

from env_mock_agent.facade.semantic_review_adapter import (
    RegistryAttachmentSemanticReviewFacade,
)
from eval_factory.agent_system.attachment_item_runtime import (
    FactoryAttachmentItemState,
    FactoryAttachmentItemView,
)
from eval_factory.agent_system.attachment_quality_material import (
    FactoryAttachmentQualityMaterialError,
    FactoryAttachmentQualityMaterialStore,
)
from eval_factory.agent_system.attachment_quality_runtime import (
    FactoryAttachmentQualityContext,
    FactoryAttachmentQualityRuntime,
)
from eval_factory.agent_system.attachment_subgraph import (
    SupervisedAttachmentR5Runner,
)
from eval_factory.agent_system.criteria_agent import (
    CriteriaRubricAgentConfig,
    CriteriaRubricAgentInputV1,
    CriteriaRubricPlanningAgentConfig,
    CriteriaRubricPlanningInputV1,
    CriteriaRubricProposalV1,
    GatewayCriteriaRubricAgent,
    GatewayCriteriaRubricPlanningAgent,
)
from eval_factory.agent_system.criteria_authority_material import (
    FactoryCriteriaAuthorityMaterialError,
    FactoryCriteriaAuthorityMaterialStore,
)
from eval_factory.agent_system.criteria_item_runtime import (
    FactoryCriteriaItemRuntime,
    FactoryCriteriaItemState,
    FactoryCriteriaPlanTemplateBuilder,
)
from eval_factory.agent_system.criteria_material_store import (
    CriteriaRubricMaterialStore,
)
from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
)
from eval_factory.agent_system.criteria_subgraph import (
    SupervisedCriteriaRubricRunner,
)
from eval_factory.agent_system.grading_agent import (
    GatewayGradingDesignAgent,
    GatewayGradingDesignPlanningAgent,
    GradingDesignAgentConfig,
    GradingDesignAgentInputV1,
    GradingDesignPlanningAgentConfig,
    GradingDesignPlanningInputV1,
)
from eval_factory.agent_system.grading_authority_material import (
    FactoryGradingAuthorityMaterialError,
    FactoryGradingAuthorityMaterialStore,
)
from eval_factory.agent_system.grading_design import (
    JudgeDesignProposalOutcomeV1,
    JudgeDesignProposalV1,
    JudgeInstructionSelectionV1,
)
from eval_factory.agent_system.grading_item_runtime import (
    FactoryGradingItemRuntime,
    FactoryGradingItemState,
    FactoryGradingPlanTemplateBuilder,
    FactoryGradingPlanTemplateConfig,
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
from eval_factory.agent_system.grading_subgraph import (
    SupervisedGradingDesignRunner,
)
from eval_factory.agent_system.item_specialist_runtime import (
    FactoryItemSpecialistContext,
    FactoryItemSpecialistInput,
    FactoryItemSpecialistRuntime,
    FactoryItemSpecialistState,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.solvability import (
    GatewaySolvabilityAgent,
    SolvabilityAgentConfig,
    SolvabilityProposalV1,
    SolvabilitySafeViewV1,
)
from eval_factory.agent_system.store import (
    FactoryControlInjectedCrash,
    FactoryControlNotFoundError,
    FactoryControlStore,
    FactoryControlStoreFaultPoint,
    StaticFactoryControlStoreFaultInjector,
)
from eval_factory.agent_system.supervisor import ExecutionSupervisor
from eval_factory.agent_system.workspace import AgentWorkspaceManager
from eval_factory.ai_gateway.invocation import (
    ProviderInvocationResult,
)
from eval_factory.contracts.agent_system_v2 import (
    GradingDesignPlanV2,
    PlanDecisionKindV2,
    SolvabilityOutcomeV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.task import RubricVisibility
from eval_factory.contracts.task_v2 import (
    producer_task_view_ref,
    r4_task_contract_set_ref,
    task_draft_ref,
)
from eval_factory.task_authoring import (
    RubricAuthoringOutcome,
    RubricCriterionSelection,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_point",
    (
        FactoryControlStoreFaultPoint.AFTER_ITEM_STAGE,
        FactoryControlStoreFaultPoint.AFTER_ITEM_STAGE_HEAD,
        FactoryControlStoreFaultPoint.AFTER_OUTBOX,
        FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
    ),
)
async def test_attachment_quality_item_head_fault_reuses_semantic_and_solvability_results(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    (
        kwargs,
        _execution_facade,
        supervisor,
        producer_view,
    ) = await prepare_attachment(tmp_path / "attachment")
    supervised = await SupervisedAttachmentR5Runner(supervisor=supervisor).run(**kwargs)
    store = supervisor.store
    job_store, job_id, item_id = quality_store(tmp_path / "quality")
    parent_ref = _create_run(
        store,
        "factory-run://dataset/quality-parent",
    )
    binding = _binding(
        parent_ref,
        kwargs["plan"].run_ref,
        item_id=item_id,
    )
    store.commit_item_binding(
        binding,
        idempotency_key="bind-factory-quality-item",
    )
    task_contract_set = _task_contract_set(producer_task_view_ref(producer_view))
    task_head = FactoryItemStageHeadV2.create(
        item_binding_ref=binding.to_ref(),
        item_run_ref=binding.item_run_ref,
        stage=FactoryItemStageV2.TASK_AUTHORING,
        stage_version=1,
        predecessor_head_ref=None,
        dependency_result_refs=(binding.rewrite_candidate_ref,),
        result_ref=r4_task_contract_set_ref(task_contract_set),
        outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=kwargs["audit"],
    )
    store.commit_item_stage_head(
        task_head,
        idempotency_key="commit-factory-quality-r4-head",
    )
    attachment_head = FactoryItemStageHeadV2.create(
        item_binding_ref=binding.to_ref(),
        item_run_ref=binding.item_run_ref,
        stage=FactoryItemStageV2.ATTACHMENT,
        stage_version=1,
        predecessor_head_ref=None,
        dependency_result_refs=(task_head.result_ref,),
        result_ref=supervised.subgraph_result.to_ref(),
        outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=kwargs["audit"],
    )
    store.commit_item_stage_head(
        attachment_head,
        idempotency_key=("commit-factory-quality-attachment-head"),
    )
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    materials = FactoryAttachmentQualityMaterialStore(private_store)
    solvability, solvability_provider = _solvability_agent(
        tmp_path / "solvability",
        private_store,
    )
    policy = semantic_review_policy()
    review_backend = _AcceptingBackend()
    context = FactoryAttachmentQualityContext(
        job_store=job_store,
        job_id=job_id,
        item_id=item_id,
        configured_pii_rules=(),
        validation_facade=_FakeValidationFacade(),
        review_policy=policy,
        context_sources=semantic_review_sources(),
        review_facade=RegistryAttachmentSemanticReviewFacade(
            resolver=_DynamicResolver(),
            review_backends={role: review_backend for role in policy.roles_in_order},
            repair_backend=_NoRepairBackend(),
        ),
        solvability_agent=solvability,
    )
    faulting_store = FactoryControlStore(
        store.path,
        fault_injector=(
            StaticFactoryControlStoreFaultInjector(
                crash_points=frozenset({fault_point}),
            )
        ),
    )
    faulting_runtime = FactoryAttachmentQualityRuntime(
        store=faulting_store,
        materials=materials,
    )

    with pytest.raises(
        FactoryControlInjectedCrash,
        match=fault_point.value,
    ):
        await faulting_runtime.advance(
            binding=binding,
            producer_task_view=producer_view,
            leakage_reference_set=_reference_set(),
            task_contract_set=task_contract_set,
            supervised_execution=supervised,
            context=context,
            audit=kwargs["audit"],
        )

    with pytest.raises(
        FactoryControlNotFoundError,
        match="not found",
    ):
        store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.ITEM_QUALITY,
        )
    semantic_calls = tuple(review_backend.context_ids)
    solvability_calls = len(solvability_provider.calls)
    recovered = await FactoryAttachmentQualityRuntime(
        store=store,
        materials=materials,
    ).advance(
        binding=binding,
        producer_task_view=producer_view,
        leakage_reference_set=_reference_set(),
        task_contract_set=task_contract_set,
        supervised_execution=supervised,
        context=context,
        audit=kwargs["audit"],
    )

    assert len(semantic_calls) == 3
    assert tuple(review_backend.context_ids) == semantic_calls
    assert solvability_calls == 1
    assert len(solvability_provider.calls) == 1
    assert (
        store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.ITEM_QUALITY,
        ).to_ref()
        == recovered.item_quality_head_ref
    )


class _SolvabilityProvider(DeterministicAttachmentProvider):
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
    ) -> None:
        super().__init__({})
        self.private_store = private_store

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        self.calls.append((request, model_profile))
        rendering = self.private_store.get_model(
            request.prompt_rendering_ref,
            SolvabilitySafeViewV1,
        )
        proposal_ref = self.private_store.put_model(
            object_type="solvability-proposal",
            value=SolvabilityProposalV1(
                outcome=SolvabilityOutcomeV2.SOLVABLE,
                evidence_refs=rendering.evidence_refs,
                reason_codes=(),
            ),
        )
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=self.private_store.put_text(
                object_type="model-response-content",
                text=proposal_ref.object_id,
            ),
            output_ref=proposal_ref,
            usage=GatewayUsageV2(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                charged_tokens=150,
                reported_cost_micro_usd=100,
            ),
            failure_code=None,
        )


def _solvability_agent(
    tmp_path: Path,
    private_store: FactoryPrivateObjectStore,
) -> tuple[GatewaySolvabilityAgent, _SolvabilityProvider]:
    solvability_prompt = prompt(
        task_kind="attachment-solvability",
        agent_role="attachment-solvability-agent",
        input_type="solvability-safe-view",
        output_type="solvability-assessment",
    )
    generator = profile("factory-quality-generator")
    judge = profile("factory-quality-solvability")
    definition_ref = ref(
        "agent-definition",
        "factory-quality-solvability",
    )
    provider = _SolvabilityProvider(private_store)
    gateway = build_gateway(
        tmp_path,
        prompts=(solvability_prompt,),
        profiles=(generator, judge),
        agent_definition_refs=(definition_ref,),
        provider=provider,
    )
    return (
        GatewaySolvabilityAgent(
            gateway=gateway,
            private_store=private_store,
            config=SolvabilityAgentConfig(
                prompt=solvability_prompt,
                agent_definition_ref=definition_ref,
                allowed_model_profile_refs=tuple(
                    sorted(
                        (
                            generator.to_ref(),
                            judge.to_ref(),
                        ),
                        key=lambda value: (
                            value.object_type,
                            value.object_id,
                            value.object_version,
                            value.object_sha256,
                        ),
                    )
                ),
                budget_reservation_ref=ref(
                    "work-model-reservation",
                    "factory-quality-solvability",
                ),
                generator_model_profile_ref=generator.to_ref(),
            ),
        ),
        provider,
    )


class _CriteriaProvider(DeterministicAttachmentProvider):
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
        *,
        planning_prompt_ref,
    ) -> None:
        super().__init__({})
        self.private_store = private_store
        self.planning_prompt_ref = planning_prompt_ref

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        self.calls.append((request, model_profile))
        if request.prompt_template_ref == self.planning_prompt_ref:
            planning_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                CriteriaRubricPlanningInputV1,
            )
            output_ref = self.private_store.put_model(
                object_type="criteria-rubric-plan",
                value=planning_input.plan_template,
            )
        else:
            agent_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                CriteriaRubricAgentInputV1,
            )
            criteria = tuple(
                RubricCriterionSelection(
                    selection_id=goal.goal_id,
                    judged_object_id=(f"judged-object://{goal.goal_id.rsplit('://', 1)[-1]}"),
                    judged_object_kind=goal.judged_object_kind,
                    judged_object_description=(goal.goal_summary),
                    description=(f"Criterion for {goal.goal_summary}"),
                    weight=(goal.weight_basis_points / 10_000),
                    prompt_requirement_ids=(goal.prompt_requirement_ids),
                    attachment_dependency_ids=(goal.attachment_dependency_ids),
                    allowed_tool_ids=goal.allowed_tool_ids,
                    visibility=RubricVisibility.EVALUATOR_ONLY,
                    evaluator_binding=(goal.evaluator_binding_id),
                )
                for goal in agent_input.criterion_goals
            )
            output_ref = self.private_store.put_model(
                object_type="criteria-rubric-proposal",
                value=CriteriaRubricProposalV1(
                    outcome=RubricAuthoringOutcome.COMPILED,
                    criteria=criteria,
                    unresolved_reasons=frozenset(),
                    confidence_basis_points=9_200,
                ),
            )
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=self.private_store.put_text(
                object_type="model-response-content",
                text=output_ref.object_id,
            ),
            output_ref=output_ref,
            usage=GatewayUsageV2(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                charged_tokens=150,
                reported_cost_micro_usd=100,
            ),
            failure_code=None,
        )


def _criteria_agents(
    tmp_path: Path,
    *,
    private_store: FactoryPrivateObjectStore,
    material_store: CriteriaRubricMaterialStore,
    store,
) -> tuple[
    GatewayCriteriaRubricPlanningAgent,
    SupervisedCriteriaRubricRunner,
    _CriteriaProvider,
    ModelCapabilityProfileV2,
]:
    planning_prompt = prompt(
        task_kind="criteria-rubric-planning",
        agent_role="criteria-rubric-planning-agent",
        input_type="criteria-rubric-planning-input",
        output_type="criteria-rubric-plan",
    )
    candidate_prompt = prompt(
        task_kind="criteria-rubric",
        agent_role="criteria-rubric-agent",
        input_type="criteria-rubric-agent-input",
        output_type="criteria-rubric-proposal",
    )
    planning_profile = profile(
        "factory-criteria-planning",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    candidate_profile = profile(
        "factory-criteria-authoring",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    planning_definition_ref = ref(
        "agent-definition",
        "factory-criteria-planning",
    )
    candidate_definition_ref = ref(
        "agent-definition",
        "factory-criteria-authoring",
    )
    provider = _CriteriaProvider(
        private_store,
        planning_prompt_ref=planning_prompt.to_ref(),
    )
    gateway = build_gateway(
        tmp_path,
        prompts=(planning_prompt, candidate_prompt),
        profiles=(planning_profile, candidate_profile),
        agent_definition_refs=(
            planning_definition_ref,
            candidate_definition_ref,
        ),
        provider=provider,
    )
    registry = criteria_registry()
    planning = GatewayCriteriaRubricPlanningAgent(
        gateway=gateway,
        private_store=private_store,
        compiler=CriteriaRubricPlanCompiler(registry),
        config=CriteriaRubricPlanningAgentConfig(
            prompt=planning_prompt,
            agent_definition_ref=planning_definition_ref,
            allowed_model_profile_refs=(planning_profile.to_ref(),),
            budget_reservation_ref=ref(
                "work-model-reservation",
                "factory-criteria-planning",
            ),
        ),
    )
    candidate = GatewayCriteriaRubricAgent(
        gateway=gateway,
        private_store=private_store,
        material_store=material_store,
        config=CriteriaRubricAgentConfig(
            prompt=candidate_prompt,
            agent_definition_ref=candidate_definition_ref,
            allowed_model_profile_refs=(candidate_profile.to_ref(),),
            budget_reservation_ref=ref(
                "work-model-reservation",
                "factory-criteria-authoring",
            ),
        ),
    )
    runner = SupervisedCriteriaRubricRunner(
        supervisor=ExecutionSupervisor(
            store,
            registry,
            workspace_manager=AgentWorkspaceManager(tmp_path / "workspaces"),
        ),
        agent=candidate,
        review_service=PlanReviewService(store),
    )
    return planning, runner, provider, candidate_profile


class _GradingProvider(DeterministicAttachmentProvider):
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
        *,
        planning_prompt_ref,
    ) -> None:
        super().__init__({})
        self.private_store = private_store
        self.planning_prompt_ref = planning_prompt_ref
        self.source_plan: GradingDesignPlanV2 | None = None

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        self.calls.append((request, model_profile))
        if request.prompt_template_ref == self.planning_prompt_ref:
            planning_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                GradingDesignPlanningInputV1,
            )
            self.source_plan = planning_input.plan_template
            output_ref = self.private_store.put_model(
                object_type="grading-design-plan",
                value=planning_input.plan_template,
            )
        else:
            self.private_store.get_model(
                request.prompt_rendering_ref,
                GradingDesignAgentInputV1,
            )
            if self.source_plan is None:
                raise AssertionError("grading candidate requires planning authority")
            proposal = JudgeDesignProposalV1(
                outcome=(JudgeDesignProposalOutcomeV1.PROPOSED),
                instructions=tuple(
                    JudgeInstructionSelectionV1(
                        task_key=task.task_key,
                        criterion_ids=task.criterion_ids,
                        evaluator_binding_id=(task.evaluator_binding_id),
                        instruction=(
                            "Evaluate only the bound criteria and return the reviewed structured fields."
                        ),
                    )
                    for task in self.source_plan.judge_tasks
                ),
                output_fields=(self.source_plan.required_output_fields),
                confidence_basis_points=9_200,
            )
            output_ref = self.private_store.put_model(
                object_type="judge-design-proposal",
                value=proposal,
            )
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=self.private_store.put_text(
                object_type="model-response-content",
                text=output_ref.object_id,
            ),
            output_ref=output_ref,
            usage=GatewayUsageV2(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                charged_tokens=150,
                reported_cost_micro_usd=100,
            ),
            failure_code=None,
        )


def _grading_agents(
    tmp_path: Path,
    *,
    private_store: FactoryPrivateObjectStore,
    material_store: GradingDesignMaterialStore,
    store,
    generator_profile: ModelCapabilityProfileV2,
) -> tuple[
    GatewayGradingDesignPlanningAgent,
    SupervisedGradingDesignRunner,
    _GradingProvider,
    ModelCapabilityProfileV2,
    AgentRegistry,
]:
    planning_prompt = prompt(
        task_kind="grading-design-planning",
        agent_role="grading-design-planning-agent",
        input_type="grading-design-planning-input",
        output_type="grading-design-plan",
    )
    judge_prompt = prompt(
        task_kind="grading-design",
        agent_role="grading-design-agent",
        input_type="grading-design-agent-input",
        output_type="judge-design-proposal",
    )
    planning_profile = profile(
        "factory-grading-planning",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    judge_profile = profile(
        "factory-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    planning_definition_ref = ref(
        "agent-definition",
        "factory-grading-planning",
    )
    model_policy_ref = ref(
        "model-routing-policy",
        "factory-grading",
    )
    registry = build_grading_design_agent_registry(
        config=GradingDesignAgentRegistryConfig(
            prompt_ref=judge_prompt.to_ref(),
            model_policy_ref=model_policy_ref,
        ),
        audit=attachment_audit(),
    )
    definition = registry.resolve(
        "grading-design-agent",
        "grading-design",
    )
    provider = _GradingProvider(
        private_store,
        planning_prompt_ref=planning_prompt.to_ref(),
    )
    gateway = build_gateway(
        tmp_path,
        prompts=(planning_prompt, judge_prompt),
        profiles=(
            planning_profile,
            generator_profile,
            judge_profile,
        ),
        agent_definition_refs=(
            planning_definition_ref,
            definition.to_ref(),
        ),
        provider=provider,
    )
    planning = GatewayGradingDesignPlanningAgent(
        gateway=gateway,
        private_store=private_store,
        compiler=GradingDesignPlanCompiler(registry),
        config=GradingDesignPlanningAgentConfig(
            prompt=planning_prompt,
            agent_definition_ref=planning_definition_ref,
            allowed_model_profile_refs=(planning_profile.to_ref(),),
            budget_reservation_ref=ref(
                "work-model-reservation",
                "factory-grading-planning",
            ),
        ),
    )
    candidate = GatewayGradingDesignAgent(
        gateway=gateway,
        private_store=private_store,
        material_store=material_store,
        config=GradingDesignAgentConfig(
            prompt=judge_prompt,
            agent_definition_ref=definition.to_ref(),
            allowed_model_profile_refs=tuple(
                sorted(
                    (
                        generator_profile.to_ref(),
                        judge_profile.to_ref(),
                    ),
                    key=lambda value: (
                        value.object_type,
                        value.object_id,
                        value.object_version,
                        value.object_sha256,
                    ),
                )
            ),
            budget_reservation_ref=ref(
                "work-model-reservation",
                "factory-grading-authoring",
            ),
        ),
    )
    runner = SupervisedGradingDesignRunner(
        supervisor=ExecutionSupervisor(
            store,
            registry,
            workspace_manager=AgentWorkspaceManager(tmp_path / "workspaces"),
        ),
        agent=candidate,
        review_service=PlanReviewService(store),
    )
    return (
        planning,
        runner,
        provider,
        judge_profile,
        registry,
    )


@pytest.mark.asyncio
async def test_attachment_quality_runtime_finalizes_and_replays(
    tmp_path: Path,
) -> None:
    kwargs, execution_facade, supervisor, producer_view = await prepare_attachment(tmp_path / "attachment")
    supervised = await SupervisedAttachmentR5Runner(supervisor=supervisor).run(**kwargs)
    execution_calls = execution_facade.calls.copy()
    store = supervisor.store
    job_store, job_id, item_id = quality_store(tmp_path / "quality")
    parent_ref = _create_run(
        store,
        "factory-run://dataset/quality-parent",
    )
    binding = _binding(
        parent_ref,
        kwargs["plan"].run_ref,
        item_id=item_id,
    )
    store.commit_item_binding(
        binding,
        idempotency_key="bind-factory-quality-item",
    )
    task_draft = _draft()
    task_contract_set = _task_contract_set(
        producer_task_view_ref(producer_view),
        task_draft_ref_value=task_draft_ref(task_draft),
    )
    task_head = FactoryItemStageHeadV2.create(
        item_binding_ref=binding.to_ref(),
        item_run_ref=binding.item_run_ref,
        stage=FactoryItemStageV2.TASK_AUTHORING,
        stage_version=1,
        predecessor_head_ref=None,
        dependency_result_refs=(binding.rewrite_candidate_ref,),
        result_ref=r4_task_contract_set_ref(task_contract_set),
        outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=kwargs["audit"],
    )
    store.commit_item_stage_head(
        task_head,
        idempotency_key="commit-factory-quality-r4-head",
    )
    attachment_head = FactoryItemStageHeadV2.create(
        item_binding_ref=binding.to_ref(),
        item_run_ref=binding.item_run_ref,
        stage=FactoryItemStageV2.ATTACHMENT,
        stage_version=1,
        predecessor_head_ref=None,
        dependency_result_refs=(task_head.result_ref,),
        result_ref=supervised.subgraph_result.to_ref(),
        outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=kwargs["audit"],
    )
    store.commit_item_stage_head(
        attachment_head,
        idempotency_key="commit-factory-quality-attachment-head",
    )
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    materials = FactoryAttachmentQualityMaterialStore(private_store)
    solvability, solvability_provider = _solvability_agent(
        tmp_path / "solvability",
        private_store,
    )
    policy = semantic_review_policy()
    review_backend = _AcceptingBackend()
    validation_facade = _FakeValidationFacade()
    context = FactoryAttachmentQualityContext(
        job_store=job_store,
        job_id=job_id,
        item_id=item_id,
        configured_pii_rules=(),
        validation_facade=validation_facade,
        review_policy=policy,
        context_sources=semantic_review_sources(),
        review_facade=RegistryAttachmentSemanticReviewFacade(
            resolver=_DynamicResolver(),
            review_backends={role: review_backend for role in policy.roles_in_order},
            repair_backend=_NoRepairBackend(),
        ),
        solvability_agent=solvability,
    )
    runtime = FactoryAttachmentQualityRuntime(
        store=store,
        materials=materials,
    )

    first = await runtime.advance(
        binding=binding,
        producer_task_view=producer_view,
        leakage_reference_set=_reference_set(),
        task_contract_set=task_contract_set,
        supervised_execution=supervised,
        context=context,
        audit=kwargs["audit"],
    )
    replay = await runtime.advance(
        binding=binding,
        producer_task_view=producer_view,
        leakage_reference_set=_reference_set(),
        task_contract_set=task_contract_set,
        supervised_execution=supervised,
        context=context,
        audit=kwargs["audit"],
    )

    assert replay == first
    assert first.result.finalization.item_quality.quality_report.approvable
    assert first.result.solvability.assessment.outcome is SolvabilityOutcomeV2.SOLVABLE
    assert execution_facade.calls == execution_calls
    assert len(validation_facade.calls) == 2
    assert len(review_backend.context_ids) == 3
    assert len(solvability_provider.calls) == 1
    quality_head = store.get_item_stage_head(
        item_id,
        FactoryItemStageV2.ITEM_QUALITY,
    )
    assert quality_head.outcome is FactoryItemStageOutcomeV2.SUCCEEDED
    assert quality_head.to_ref() == first.item_quality_head_ref
    assert store.get_item_stage_material_ref(quality_head.to_ref()) == first.material_ref
    assert materials.get(first.material_ref) == first.result
    with pytest.raises(
        FactoryAttachmentQualityMaterialError,
        match="reference type",
    ):
        materials.get(first.material_ref.model_copy(update={"object_type": "wrong-material"}))

    criteria_private = FactoryPrivateObjectStore(tmp_path / "criteria-private")
    criteria_materials = CriteriaRubricMaterialStore(tmp_path / "criteria-behavior")
    (
        criteria_planning,
        criteria_runner,
        criteria_provider,
        criteria_generator_profile,
    ) = _criteria_agents(
        tmp_path / "criteria",
        private_store=criteria_private,
        material_store=criteria_materials,
        store=store,
    )
    criteria_authority_materials = FactoryCriteriaAuthorityMaterialStore(criteria_private)
    criteria_reviews = PlanReviewService(store)
    criteria_runtime = FactoryCriteriaItemRuntime(
        store=store,
        plan_reviews=criteria_reviews,
        planning_agent=criteria_planning,
        runner=criteria_runner,
        template_builder=FactoryCriteriaPlanTemplateBuilder(criteria_registry()),
        criteria_materials=criteria_materials,
        authority_materials=criteria_authority_materials,
        requested_by="user://factory-quality-owner",
    )
    bindings = _binding_definitions()
    catalog = _catalog()
    child_policy = store.get_policy(store.get_item_run(binding).policy_ref.object_id)
    attachment_view = FactoryAttachmentItemView(
        item_id=binding.item_id,
        item_run_ref=binding.item_run_ref,
        plan_ref=kwargs["plan"].to_ref(),
        compiled_plan_ref=kwargs["compiled_plan"].to_ref(),
        review_ref=ref(
            "plan-review-request",
            "factory-quality-attachment",
        ),
        stage_head_ref=attachment_head.to_ref(),
        supervised_execution=supervised,
        state=FactoryAttachmentItemState.EXECUTED,
    )
    specialist_source = FactoryItemSpecialistInput(
        producer_task_view=producer_view,
        leakage_reference_set=_reference_set(),
        task_contract_set=task_contract_set,
        task_draft=task_draft,
    )
    specialist_context = FactoryItemSpecialistContext(
        quality=context,
        evaluator_bindings=bindings,
        tool_catalog=catalog,
        model_authorizations=(),
        evaluated_at=kwargs["audit"].created_at,
    )
    criteria_specialist = FactoryItemSpecialistRuntime(
        store=store,
        quality_runtime=runtime,
        criteria_runtime=criteria_runtime,
    )
    specialist_waiting = await criteria_specialist.advance(
        binding=binding,
        source=specialist_source,
        attachment=attachment_view,
        policy=child_policy,
        context=specialist_context,
        audit=kwargs["audit"],
    )
    specialist_waiting_replay = await criteria_specialist.advance(
        binding=binding,
        source=specialist_source,
        attachment=attachment_view,
        policy=child_policy,
        context=specialist_context,
        audit=kwargs["audit"],
    )
    waiting = specialist_waiting.criteria
    waiting_replay = specialist_waiting_replay.criteria
    assert waiting is not None
    assert waiting_replay == waiting
    assert specialist_waiting.state is FactoryItemSpecialistState.WAITING_CRITERIA_REVIEW
    assert waiting.state is FactoryCriteriaItemState.WAITING_REVIEW
    assert len(criteria_provider.calls) == 1

    review_id = waiting.review_ref.object_id
    criteria_reviews.decide(
        review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by="user://factory-quality-owner",
            reason_code="CRITERIA_PLAN_APPROVED",
            idempotency_key="approve-factory-criteria",
        ),
        audit=kwargs["audit"],
    )
    criteria_reviews.resume(
        review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by="user://factory-quality-owner",
            idempotency_key="resume-factory-criteria",
        ),
        audit=kwargs["audit"],
    )
    criteria_ready_to_execute = await criteria_runtime.prepare_review(
        binding=binding,
        policy=child_policy,
        task_draft=specialist_source.task_draft,
        task_contract_set=specialist_source.task_contract_set,
        quality=first,
        quality_context=context,
        binding_definitions=bindings,
        tool_catalog=catalog,
        audit=kwargs["audit"],
    )
    assert criteria_ready_to_execute.state is FactoryCriteriaItemState.READY_TO_EXECUTE
    assert len(criteria_provider.calls) == 1
    with pytest.raises(
        FactoryControlNotFoundError,
        match="not found",
    ):
        store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.CRITERIA_RUBRIC,
        )
    criteria_runtime.store = FactoryControlStore(
        store.path,
        fault_injector=(
            StaticFactoryControlStoreFaultInjector(
                crash_points=frozenset({FactoryControlStoreFaultPoint.AFTER_ITEM_STAGE}),
            )
        ),
    )
    with pytest.raises(
        FactoryControlInjectedCrash,
        match="after_item_stage",
    ):
        await criteria_specialist.advance(
            binding=binding,
            source=specialist_source,
            attachment=attachment_view,
            policy=child_policy,
            context=specialist_context,
            audit=kwargs["audit"],
        )
    with pytest.raises(
        FactoryControlNotFoundError,
        match="not found",
    ):
        store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.CRITERIA_RUBRIC,
        )
    assert len(criteria_provider.calls) == 2
    criteria_runtime.store = store
    specialist_ready = await criteria_specialist.advance(
        binding=binding,
        source=specialist_source,
        attachment=attachment_view,
        policy=child_policy,
        context=specialist_context,
        audit=kwargs["audit"],
    )
    specialist_ready_replay = await criteria_specialist.advance(
        binding=binding,
        source=specialist_source,
        attachment=attachment_view,
        policy=child_policy,
        context=specialist_context,
        audit=kwargs["audit"],
    )
    criteria = specialist_ready.criteria
    criteria_replay = specialist_ready_replay.criteria

    assert criteria is not None
    assert criteria_replay == criteria
    assert specialist_ready.state is FactoryItemSpecialistState.READY_FOR_GRADING
    assert criteria.state is FactoryCriteriaItemState.EXECUTED
    assert criteria.authority is not None
    assert criteria.authority.execution.rubric_set is not None
    assert criteria.authority.execution.evaluator_spec is not None
    assert criteria.authority.execution.reference_policy is not None
    assert criteria.authority.execution.tool_policy is not None
    assert criteria.authority.execution.contestant_tool_policy is not None
    assert len(criteria_provider.calls) == 2
    criteria_head = store.get_item_stage_head(
        item_id,
        FactoryItemStageV2.CRITERIA_RUBRIC,
    )
    assert criteria_head.outcome is FactoryItemStageOutcomeV2.SUCCEEDED
    assert criteria_head.to_ref() == criteria.stage_head_ref
    assert criteria.material_ref is not None
    assert criteria_authority_materials.get(criteria.material_ref) == criteria.authority
    with pytest.raises(
        FactoryCriteriaAuthorityMaterialError,
        match="reference type",
    ):
        criteria_authority_materials.get(
            criteria.material_ref.model_copy(update={"object_type": "wrong-material"})
        )

    grading_private = FactoryPrivateObjectStore(tmp_path / "grading-private")
    grading_materials = GradingDesignMaterialStore(tmp_path / "grading-behavior")
    (
        grading_planning,
        grading_runner,
        grading_provider,
        judge_profile,
        grading_registry,
    ) = _grading_agents(
        tmp_path / "grading",
        private_store=grading_private,
        material_store=grading_materials,
        store=store,
        generator_profile=criteria_generator_profile,
    )
    grading_authority_materials = FactoryGradingAuthorityMaterialStore(grading_private)
    grading_reviews = PlanReviewService(store)
    grading_runtime = FactoryGradingItemRuntime(
        store=store,
        plan_reviews=grading_reviews,
        planning_agent=grading_planning,
        runner=grading_runner,
        template_builder=FactoryGradingPlanTemplateBuilder(
            grading_registry,
            FactoryGradingPlanTemplateConfig(
                allowed_judge_model_profile_refs=tuple(
                    sorted(
                        (
                            criteria_generator_profile.to_ref(),
                            judge_profile.to_ref(),
                        ),
                        key=lambda value: (
                            value.object_type,
                            value.object_id,
                            value.object_version,
                            value.object_sha256,
                        ),
                    )
                ),
                judge_input_schema_ref=ref(
                    "json-schema",
                    "factory-grading-input",
                ),
                judge_output_schema_ref=ref(
                    "json-schema",
                    "factory-grading-output",
                ),
            ),
        ),
        grading_materials=grading_materials,
        authority_materials=grading_authority_materials,
        requested_by="user://factory-quality-owner",
    )
    full_specialist = FactoryItemSpecialistRuntime(
        store=store,
        quality_runtime=runtime,
        criteria_runtime=criteria_runtime,
        grading_runtime=grading_runtime,
    )
    specialist_grading_waiting = await full_specialist.advance(
        binding=binding,
        source=specialist_source,
        attachment=attachment_view,
        policy=child_policy,
        context=specialist_context,
        audit=kwargs["audit"],
    )
    specialist_grading_waiting_replay = await full_specialist.advance(
        binding=binding,
        source=specialist_source,
        attachment=attachment_view,
        policy=child_policy,
        context=specialist_context,
        audit=kwargs["audit"],
    )
    grading_waiting = specialist_grading_waiting.grading
    grading_waiting_replay = specialist_grading_waiting_replay.grading
    assert grading_waiting is not None
    assert grading_waiting_replay == grading_waiting
    assert specialist_grading_waiting.state is FactoryItemSpecialistState.WAITING_GRADING_REVIEW
    assert grading_waiting.state is FactoryGradingItemState.WAITING_REVIEW
    assert len(grading_provider.calls) == 1

    grading_review_id = grading_waiting.review_ref.object_id
    grading_reviews.decide(
        grading_review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by="user://factory-quality-owner",
            reason_code="GRADING_PLAN_APPROVED",
            idempotency_key="approve-factory-grading",
        ),
        audit=kwargs["audit"],
    )
    grading_reviews.resume(
        grading_review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by="user://factory-quality-owner",
            idempotency_key="resume-factory-grading",
        ),
        audit=kwargs["audit"],
    )
    grading_ready_to_execute = await grading_runtime.prepare_review(
        binding=binding,
        policy=child_policy,
        criteria=criteria,
        model_authorizations=specialist_context.model_authorizations,
        evaluated_at=specialist_context.evaluated_at,
        audit=kwargs["audit"],
    )
    assert grading_ready_to_execute.state is FactoryGradingItemState.READY_TO_EXECUTE
    assert len(grading_provider.calls) == 1
    with pytest.raises(
        FactoryControlNotFoundError,
        match="not found",
    ):
        store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.GRADING_DESIGN,
        )
    grading_runtime.store = FactoryControlStore(
        store.path,
        fault_injector=(
            StaticFactoryControlStoreFaultInjector(
                crash_points=frozenset({FactoryControlStoreFaultPoint.AFTER_ITEM_STAGE}),
            )
        ),
    )
    with pytest.raises(
        FactoryControlInjectedCrash,
        match="after_item_stage",
    ):
        await full_specialist.advance(
            binding=binding,
            source=specialist_source,
            attachment=attachment_view,
            policy=child_policy,
            context=specialist_context,
            audit=kwargs["audit"],
        )
    with pytest.raises(
        FactoryControlNotFoundError,
        match="not found",
    ):
        store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.GRADING_DESIGN,
        )
    assert len(grading_provider.calls) == 2
    grading_runtime.store = store
    specialist_executed = await full_specialist.advance(
        binding=binding,
        source=specialist_source,
        attachment=attachment_view,
        policy=child_policy,
        context=specialist_context,
        audit=kwargs["audit"],
    )
    specialist_executed_replay = await full_specialist.advance(
        binding=binding,
        source=specialist_source,
        attachment=attachment_view,
        policy=child_policy,
        context=specialist_context,
        audit=kwargs["audit"],
    )
    grading = specialist_executed.grading
    grading_replay = specialist_executed_replay.grading

    assert grading is not None
    assert grading_replay == grading
    assert specialist_executed.state is FactoryItemSpecialistState.EXECUTED
    assert grading.state is FactoryGradingItemState.EXECUTED
    assert grading.authority is not None
    assert grading.authority.execution.design_spec is not None
    assert grading.authority.execution.route is not None
    assert grading.authority.execution.route.selected_model_profile_ref == judge_profile.to_ref()
    assert grading.authority.execution.route.selected_model_profile_ref != criteria_generator_profile.to_ref()
    assert len(grading_provider.calls) == 2
    grading_head = store.get_item_stage_head(
        item_id,
        FactoryItemStageV2.GRADING_DESIGN,
    )
    assert grading_head.outcome is FactoryItemStageOutcomeV2.SUCCEEDED
    assert grading_head.to_ref() == grading.stage_head_ref
    assert grading.material_ref is not None
    assert grading_authority_materials.get(grading.material_ref) == grading.authority
    with pytest.raises(
        FactoryGradingAuthorityMaterialError,
        match="reference type",
    ):
        grading_authority_materials.get(
            grading.material_ref.model_copy(update={"object_type": "wrong-material"})
        )
