from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from env_mock_agent.facade import (
    AttachmentInventoryMemberTypeV2,
    AttachmentInventoryMemberV2,
    AttachmentValidationRequestV2,
    AttachmentValidationResultV2,
    AttachmentValidationStatusV2,
    FacadeObjectRef,
    attachment_inventory_member_carried_sha256,
    attachment_validation_request_ref,
    attachment_validation_result_carried_sha256,
    validate_attachment_validation_request_identity,
)
from env_mock_agent.facade.semantic_review_adapter import (
    ProviderAttachmentRepairMaterial,
    ProviderSemanticReviewContext,
    ProviderSemanticReviewDecision,
    ProviderSemanticReviewMaterial,
    RegistryAttachmentSemanticReviewFacade,
)
from env_mock_agent.facade.semantic_review_v2 import (
    AttachmentRepairRequestV2,
    AttachmentSemanticReviewerRoleV2,
    AttachmentSemanticReviewRequestV2,
)
from eval_factory.agent_system.attachment_quality_material import (
    FactoryAttachmentQualityMaterialStore,
)
from eval_factory.agent_system.attachment_quality_runtime import (
    FactoryAttachmentQualityContext,
    FactoryAttachmentQualityRuntime,
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
    FactoryCriteriaAuthorityMaterialStore,
)
from eval_factory.agent_system.criteria_item_runtime import (
    FactoryCriteriaItemRuntime,
    FactoryCriteriaPlanTemplateBuilder,
)
from eval_factory.agent_system.criteria_material_store import (
    CriteriaRubricMaterialStore,
)
from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
)
from eval_factory.agent_system.criteria_registry import (
    CriteriaRubricAgentRegistryConfig,
    build_criteria_rubric_agent_registry,
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
    FactoryGradingAuthorityMaterialStore,
)
from eval_factory.agent_system.grading_design import (
    JudgeDesignProposalOutcomeV1,
    JudgeDesignProposalV1,
    JudgeInstructionSelectionV1,
)
from eval_factory.agent_system.grading_item_runtime import (
    FactoryGradingItemRuntime,
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
    FactoryItemSpecialistRuntime,
)
from eval_factory.agent_system.job_store_bridge import (
    FactoryJobStoreAuthority,
    FactoryJobStoreBridge,
    FactoryJobStoreTemplate,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.solvability import (
    GatewaySolvabilityAgent,
    SolvabilityAgentConfig,
    SolvabilityProposalV1,
    SolvabilitySafeViewV1,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.supervisor import ExecutionSupervisor
from eval_factory.agent_system.workspace import AgentWorkspaceManager
from eval_factory.ai_gateway.invocation import ProviderInvocationResult
from eval_factory.ai_gateway.protocols import AIGateway
from eval_factory.attachment_planning import SemanticReviewContextSources
from eval_factory.contracts.agent_system_v2 import SolvabilityOutcomeV2
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
    PromptTemplateV2,
)
from eval_factory.contracts.approval import UserApprovalPolicy
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemRunBindingV2,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
)
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.contracts.review_v2 import SemanticReviewPolicyV2
from eval_factory.contracts.task import RubricVisibility
from eval_factory.orchestration.job_store import JobStore
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    RubricAuthoringOutcome,
    RubricCriterionSelection,
    ToolCapabilityCatalog,
)


class FactoryDatasetJobStoreRuntimeConfigV1(ContractModelV2):
    schema_version: Literal["eval-factory/factory-dataset-job-store-runtime-config/v1"] = (
        "eval-factory/factory-dataset-job-store-runtime-config/v1"
    )
    requested_stages: tuple[StageNameV2, ...] = Field(
        min_length=1,
        max_length=32,
    )
    privacy_profile: str = Field(min_length=1, max_length=128)
    model_profiles: tuple[str, ...] = Field(default=(), max_length=256)
    budget: ResourceBudget
    concurrency: ConcurrencyLimit
    approval_policy: UserApprovalPolicy
    export_target: ExportTarget
    adapter_name: str = Field(
        default="raw-traj-v1",
        min_length=1,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        self.to_template()
        if self.model_profiles != tuple(sorted(set(self.model_profiles))):
            raise ValueError("JobStore model profiles must be canonical")
        return self

    def to_template(self) -> FactoryJobStoreTemplate:
        return FactoryJobStoreTemplate(
            requested_stages=self.requested_stages,
            privacy_profile=self.privacy_profile,
            model_profiles=self.model_profiles,
            budget=self.budget,
            concurrency=self.concurrency,
            approval_policy=self.approval_policy,
            export_target=self.export_target,
            adapter_name=self.adapter_name,
        )


class FactoryDatasetSpecialistRuntimeConfigV1(ContractModelV2):
    schema_version: Literal["eval-factory/factory-dataset-specialist-runtime-config/v1"] = (
        "eval-factory/factory-dataset-specialist-runtime-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    job_store: FactoryDatasetJobStoreRuntimeConfigV1
    review_policy: SemanticReviewPolicyV2
    context_sources: SemanticReviewContextSources
    evaluator_bindings: tuple[
        EvaluatorBindingDefinition,
        ...,
    ] = Field(min_length=1, max_length=256)
    tool_catalog: ToolCapabilityCatalog
    evaluated_at: datetime
    solvability_prompt: PromptTemplateV2
    solvability_agent_definition_ref: ObjectRef
    solvability_allowed_model_profile_refs: tuple[
        ObjectRef,
        ...,
    ] = Field(min_length=2, max_length=256)
    solvability_budget_reservation_ref: ObjectRef
    generator_model_profile_ref: ObjectRef
    criteria_planning_prompt: PromptTemplateV2
    criteria_prompt: PromptTemplateV2
    criteria_planning_agent_definition_ref: ObjectRef
    criteria_agent_definition_ref: ObjectRef
    criteria_planning_allowed_model_profile_refs: tuple[
        ObjectRef,
        ...,
    ] = Field(min_length=1, max_length=256)
    criteria_allowed_model_profile_refs: tuple[
        ObjectRef,
        ...,
    ] = Field(min_length=1, max_length=256)
    criteria_planning_budget_reservation_ref: ObjectRef
    criteria_budget_reservation_ref: ObjectRef
    criteria_model_policy_ref: ObjectRef
    grading_planning_prompt: PromptTemplateV2
    grading_prompt: PromptTemplateV2
    grading_planning_agent_definition_ref: ObjectRef
    grading_agent_definition_ref: ObjectRef
    grading_planning_allowed_model_profile_refs: tuple[
        ObjectRef,
        ...,
    ] = Field(min_length=1, max_length=256)
    grading_allowed_model_profile_refs: tuple[
        ObjectRef,
        ...,
    ] = Field(min_length=2, max_length=256)
    grading_planning_budget_reservation_ref: ObjectRef
    grading_budget_reservation_ref: ObjectRef
    grading_model_policy_ref: ObjectRef
    judge_input_schema_ref: ObjectRef
    judge_output_schema_ref: ObjectRef

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if tuple(
            prompt.task_kind
            for prompt in (
                self.solvability_prompt,
                self.criteria_planning_prompt,
                self.criteria_prompt,
                self.grading_planning_prompt,
                self.grading_prompt,
            )
        ) != (
            "attachment-solvability",
            "criteria-rubric-planning",
            "criteria-rubric",
            "grading-design-planning",
            "grading-design",
        ):
            raise ValueError("specialist runtime prompts use the wrong task kinds")
        for reference in (
            self.solvability_agent_definition_ref,
            self.criteria_planning_agent_definition_ref,
            self.criteria_agent_definition_ref,
            self.grading_planning_agent_definition_ref,
            self.grading_agent_definition_ref,
        ):
            _require_ref(
                reference,
                object_type="agent-definition",
                label="specialist Agent definition",
            )
        for reference in (
            self.solvability_budget_reservation_ref,
            self.criteria_planning_budget_reservation_ref,
            self.criteria_budget_reservation_ref,
            self.grading_planning_budget_reservation_ref,
            self.grading_budget_reservation_ref,
        ):
            _require_ref(
                reference,
                object_type="work-model-reservation",
                label="specialist budget reservation",
            )
        for reference in (
            self.criteria_model_policy_ref,
            self.grading_model_policy_ref,
        ):
            _require_ref(
                reference,
                object_type="model-routing-policy",
                label="specialist model policy",
            )
        for reference in (
            self.judge_input_schema_ref,
            self.judge_output_schema_ref,
        ):
            _require_ref(
                reference,
                object_type="json-schema",
                label="judge schema",
            )
        profile_sets = (
            self.solvability_allowed_model_profile_refs,
            self.criteria_planning_allowed_model_profile_refs,
            self.criteria_allowed_model_profile_refs,
            self.grading_planning_allowed_model_profile_refs,
            self.grading_allowed_model_profile_refs,
        )
        for values in profile_sets:
            if values != tuple(sorted(set(values), key=_ref_key)):
                raise ValueError("specialist model profile refs must be canonical")
        if (
            self.generator_model_profile_ref not in self.solvability_allowed_model_profile_refs
            or len(set(self.solvability_allowed_model_profile_refs) - {self.generator_model_profile_ref}) < 1
        ):
            raise ValueError("solvability requires generator and independent judge models")
        criteria_generators = set(self.criteria_allowed_model_profile_refs)
        if not criteria_generators.intersection(self.grading_allowed_model_profile_refs) or not (
            set(self.grading_allowed_model_profile_refs) - criteria_generators
        ):
            raise ValueError("grading requires criteria generator and independent judge models")
        binding_ids = tuple(value.evaluator_binding_id for value in self.evaluator_bindings)
        if binding_ids != tuple(sorted(set(binding_ids))):
            raise ValueError("evaluator bindings must be canonical")
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("specialist evaluated_at must be timezone-aware")
        criteria_registry = build_criteria_rubric_agent_registry(
            config=CriteriaRubricAgentRegistryConfig(
                prompt_ref=self.criteria_prompt.to_ref(),
                model_policy_ref=self.criteria_model_policy_ref,
            ),
            audit=self.criteria_prompt.audit,
        )
        if (
            criteria_registry.resolve(
                "criteria-rubric-agent",
                "criteria-rubric",
            ).to_ref()
            != self.criteria_agent_definition_ref
        ):
            raise ValueError("criteria Agent definition differs from registry")
        grading_registry = build_grading_design_agent_registry(
            config=GradingDesignAgentRegistryConfig(
                prompt_ref=self.grading_prompt.to_ref(),
                model_policy_ref=self.grading_model_policy_ref,
            ),
            audit=self.grading_prompt.audit,
        )
        if (
            grading_registry.resolve(
                "grading-design-agent",
                "grading-design",
            ).to_ref()
            != self.grading_agent_definition_ref
        ):
            raise ValueError("grading Agent definition differs from registry")
        return self

    def prompt_templates(self) -> tuple[PromptTemplateV2, ...]:
        return (
            self.solvability_prompt,
            self.criteria_planning_prompt,
            self.criteria_prompt,
            self.grading_planning_prompt,
            self.grading_prompt,
        )


class FixtureSpecialistGatewayProvider:
    def __init__(
        self,
        *,
        private_store: FactoryPrivateObjectStore,
        config: FactoryDatasetSpecialistRuntimeConfigV1,
    ) -> None:
        self.private_store = private_store
        self.config = config
        self._prompt_refs = frozenset(prompt.to_ref() for prompt in config.prompt_templates())

    def supports(self, prompt_ref: ObjectRef) -> bool:
        return prompt_ref in self._prompt_refs

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        del model_profile
        prompt_ref = request.prompt_template_ref
        if prompt_ref == self.config.solvability_prompt.to_ref():
            safe_view = self.private_store.get_model(
                request.prompt_rendering_ref,
                SolvabilitySafeViewV1,
            )
            output_ref = self.private_store.put_model(
                object_type="solvability-proposal",
                value=SolvabilityProposalV1(
                    outcome=SolvabilityOutcomeV2.SOLVABLE,
                    evidence_refs=safe_view.evidence_refs,
                    reason_codes=(),
                ),
            )
        elif prompt_ref == self.config.criteria_planning_prompt.to_ref():
            criteria_planning = self.private_store.get_model(
                request.prompt_rendering_ref,
                CriteriaRubricPlanningInputV1,
            )
            output_ref = self.private_store.put_model(
                object_type="criteria-rubric-plan",
                value=criteria_planning.plan_template,
            )
        elif prompt_ref == self.config.criteria_prompt.to_ref():
            criteria_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                CriteriaRubricAgentInputV1,
            )
            output_ref = self.private_store.put_model(
                object_type="criteria-rubric-proposal",
                value=CriteriaRubricProposalV1(
                    outcome=RubricAuthoringOutcome.COMPILED,
                    criteria=tuple(
                        RubricCriterionSelection(
                            selection_id=goal.goal_id,
                            judged_object_id=(f"judged-object://{goal.goal_id.rsplit('://', 1)[-1]}"),
                            judged_object_kind=goal.judged_object_kind,
                            judged_object_description=goal.goal_summary,
                            description=(f"Criterion for {goal.goal_summary}"),
                            weight=(goal.weight_basis_points / 10_000),
                            prompt_requirement_ids=(goal.prompt_requirement_ids),
                            attachment_dependency_ids=(goal.attachment_dependency_ids),
                            allowed_tool_ids=goal.allowed_tool_ids,
                            visibility=(RubricVisibility.EVALUATOR_ONLY),
                            evaluator_binding=(goal.evaluator_binding_id),
                        )
                        for goal in criteria_input.criterion_goals
                    ),
                    unresolved_reasons=frozenset(),
                    confidence_basis_points=9_200,
                ),
            )
        elif prompt_ref == self.config.grading_planning_prompt.to_ref():
            grading_planning = self.private_store.get_model(
                request.prompt_rendering_ref,
                GradingDesignPlanningInputV1,
            )
            output_ref = self.private_store.put_model(
                object_type="grading-design-plan",
                value=grading_planning.plan_template,
            )
        elif prompt_ref == self.config.grading_prompt.to_ref():
            grading_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                GradingDesignAgentInputV1,
            )
            groups: dict[str, list[str]] = {}
            for criterion in grading_input.rubric_set.criteria:
                groups.setdefault(
                    criterion.evaluator_binding,
                    [],
                ).append(criterion.criterion_id)
            output_ref = self.private_store.put_model(
                object_type="judge-design-proposal",
                value=JudgeDesignProposalV1(
                    outcome=(JudgeDesignProposalOutcomeV1.PROPOSED),
                    instructions=tuple(
                        JudgeInstructionSelectionV1(
                            task_key=(
                                f"judge-task://sha256/{hashlib.sha256(binding_id.encode()).hexdigest()}"
                            ),
                            criterion_ids=tuple(sorted(criterion_ids)),
                            evaluator_binding_id=binding_id,
                            instruction=(
                                "Evaluate only the bound criteria and return the reviewed structured fields."
                            ),
                        )
                        for binding_id, criterion_ids in sorted(groups.items())
                    ),
                    output_fields=(
                        "ABSTAIN",
                        "EVIDENCE_IDS",
                        "FAILURE_CLASS",
                        "SCORE_BASIS_POINTS",
                    ),
                    confidence_basis_points=9_200,
                ),
            )
        else:
            raise ValueError("fixture specialist provider capability is unavailable")
        return _success(
            private_store=self.private_store,
            output_ref=output_ref,
        )


@dataclass(frozen=True, slots=True)
class FactoryFixtureSpecialistComponents:
    job_store_bridge: FactoryJobStoreBridge
    runtime: FactoryItemSpecialistRuntime
    context_factory: FactoryFixtureSpecialistContextFactory


class FactoryFixtureSpecialistContextFactory:
    def __init__(
        self,
        *,
        job_store: JobStore,
        config: FactoryDatasetSpecialistRuntimeConfigV1,
        solvability_agent: GatewaySolvabilityAgent,
    ) -> None:
        self.job_store = job_store
        self.config = config
        self.solvability_agent = solvability_agent
        self._contexts: dict[str, FactoryItemSpecialistContext] = {}

    def build(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        job_authority: FactoryJobStoreAuthority,
    ) -> FactoryItemSpecialistContext:
        if (
            job_authority.job_spec.job_id != self.job_store.get_job(job_authority.job_spec.job_id).job_id
            or binding.item_id not in job_authority.graph.item_ids
        ):
            raise ValueError("specialist context requires current JobStore authority")
        current = self._contexts.get(binding.item_id)
        if current is not None:
            return current
        resolver = _FixtureSemanticMaterialResolver()
        backend = _FixtureAcceptingSemanticBackend()
        context = FactoryItemSpecialistContext(
            quality=FactoryAttachmentQualityContext(
                job_store=self.job_store,
                job_id=job_authority.job_spec.job_id,
                item_id=binding.item_id,
                configured_pii_rules=(),
                validation_facade=(_FixturePassingValidationFacade()),
                review_policy=self.config.review_policy,
                context_sources=self.config.context_sources,
                review_facade=RegistryAttachmentSemanticReviewFacade(
                    resolver=resolver,
                    review_backends={
                        AttachmentSemanticReviewerRoleV2(role.value): backend
                        for role in (self.config.review_policy.roles_in_order)
                    },
                    repair_backend=None,
                    clock=lambda: self.config.evaluated_at,
                ),
                solvability_agent=self.solvability_agent,
            ),
            evaluator_bindings=self.config.evaluator_bindings,
            tool_catalog=self.config.tool_catalog,
            model_authorizations=(),
            evaluated_at=self.config.evaluated_at,
        )
        self._contexts[binding.item_id] = context
        return context


def build_fixture_specialist_components(
    *,
    config: FactoryDatasetSpecialistRuntimeConfigV1,
    gateway: AIGateway,
    private_store: FactoryPrivateObjectStore,
    factory_store: FactoryControlStore,
    plan_reviews: PlanReviewService,
    job_store_path: Path,
    workspace_path: Path,
    requested_by: str,
) -> FactoryFixtureSpecialistComponents:
    job_store = JobStore(job_store_path)
    job_store_bridge = FactoryJobStoreBridge(
        job_store=job_store,
        template=config.job_store.to_template(),
    )
    criteria_registry = build_criteria_rubric_agent_registry(
        config=CriteriaRubricAgentRegistryConfig(
            prompt_ref=config.criteria_prompt.to_ref(),
            model_policy_ref=config.criteria_model_policy_ref,
        ),
        audit=config.criteria_prompt.audit,
    )
    grading_registry = build_grading_design_agent_registry(
        config=GradingDesignAgentRegistryConfig(
            prompt_ref=config.grading_prompt.to_ref(),
            model_policy_ref=config.grading_model_policy_ref,
        ),
        audit=config.grading_prompt.audit,
    )
    criteria_materials = CriteriaRubricMaterialStore(workspace_path / "criteria-materials")
    grading_materials = GradingDesignMaterialStore(workspace_path / "grading-materials")
    criteria_agent = GatewayCriteriaRubricAgent(
        gateway=gateway,
        private_store=private_store,
        material_store=criteria_materials,
        config=CriteriaRubricAgentConfig(
            prompt=config.criteria_prompt,
            agent_definition_ref=(config.criteria_agent_definition_ref),
            allowed_model_profile_refs=(config.criteria_allowed_model_profile_refs),
            budget_reservation_ref=(config.criteria_budget_reservation_ref),
        ),
    )
    grading_agent = GatewayGradingDesignAgent(
        gateway=gateway,
        private_store=private_store,
        material_store=grading_materials,
        config=GradingDesignAgentConfig(
            prompt=config.grading_prompt,
            agent_definition_ref=(config.grading_agent_definition_ref),
            allowed_model_profile_refs=(config.grading_allowed_model_profile_refs),
            budget_reservation_ref=(config.grading_budget_reservation_ref),
        ),
    )
    quality_runtime = FactoryAttachmentQualityRuntime(
        store=factory_store,
        materials=FactoryAttachmentQualityMaterialStore(private_store),
    )
    criteria_runtime = FactoryCriteriaItemRuntime(
        store=factory_store,
        plan_reviews=plan_reviews,
        planning_agent=GatewayCriteriaRubricPlanningAgent(
            gateway=gateway,
            private_store=private_store,
            compiler=CriteriaRubricPlanCompiler(criteria_registry),
            config=CriteriaRubricPlanningAgentConfig(
                prompt=config.criteria_planning_prompt,
                agent_definition_ref=(config.criteria_planning_agent_definition_ref),
                allowed_model_profile_refs=(config.criteria_planning_allowed_model_profile_refs),
                budget_reservation_ref=(config.criteria_planning_budget_reservation_ref),
            ),
        ),
        runner=SupervisedCriteriaRubricRunner(
            supervisor=ExecutionSupervisor(
                factory_store,
                criteria_registry,
                workspace_manager=AgentWorkspaceManager(workspace_path / "criteria-workspaces"),
            ),
            agent=criteria_agent,
            review_service=plan_reviews,
        ),
        template_builder=FactoryCriteriaPlanTemplateBuilder(criteria_registry),
        criteria_materials=criteria_materials,
        authority_materials=(FactoryCriteriaAuthorityMaterialStore(private_store)),
        requested_by=requested_by,
    )
    grading_runtime = FactoryGradingItemRuntime(
        store=factory_store,
        plan_reviews=plan_reviews,
        planning_agent=GatewayGradingDesignPlanningAgent(
            gateway=gateway,
            private_store=private_store,
            compiler=GradingDesignPlanCompiler(grading_registry),
            config=GradingDesignPlanningAgentConfig(
                prompt=config.grading_planning_prompt,
                agent_definition_ref=(config.grading_planning_agent_definition_ref),
                allowed_model_profile_refs=(config.grading_planning_allowed_model_profile_refs),
                budget_reservation_ref=(config.grading_planning_budget_reservation_ref),
            ),
        ),
        runner=SupervisedGradingDesignRunner(
            supervisor=ExecutionSupervisor(
                factory_store,
                grading_registry,
                workspace_manager=AgentWorkspaceManager(workspace_path / "grading-workspaces"),
            ),
            agent=grading_agent,
            review_service=plan_reviews,
        ),
        template_builder=FactoryGradingPlanTemplateBuilder(
            grading_registry,
            FactoryGradingPlanTemplateConfig(
                allowed_judge_model_profile_refs=(config.grading_allowed_model_profile_refs),
                judge_input_schema_ref=(config.judge_input_schema_ref),
                judge_output_schema_ref=(config.judge_output_schema_ref),
            ),
        ),
        grading_materials=grading_materials,
        authority_materials=(FactoryGradingAuthorityMaterialStore(private_store)),
        requested_by=requested_by,
    )
    solvability_agent = GatewaySolvabilityAgent(
        gateway=gateway,
        private_store=private_store,
        config=SolvabilityAgentConfig(
            prompt=config.solvability_prompt,
            agent_definition_ref=(config.solvability_agent_definition_ref),
            allowed_model_profile_refs=(config.solvability_allowed_model_profile_refs),
            budget_reservation_ref=(config.solvability_budget_reservation_ref),
            generator_model_profile_ref=(config.generator_model_profile_ref),
        ),
    )
    context_factory = FactoryFixtureSpecialistContextFactory(
        job_store=job_store,
        config=config,
        solvability_agent=solvability_agent,
    )
    return FactoryFixtureSpecialistComponents(
        job_store_bridge=job_store_bridge,
        runtime=FactoryItemSpecialistRuntime(
            store=factory_store,
            quality_runtime=quality_runtime,
            criteria_runtime=criteria_runtime,
            grading_runtime=grading_runtime,
        ),
        context_factory=context_factory,
    )


class _FixturePassingValidationFacade:
    async def validate(
        self,
        request: AttachmentValidationRequestV2,
    ) -> AttachmentValidationResultV2:
        validate_attachment_validation_request_identity(request)
        required = (
            "common",
            "configured-pii",
            "metadata",
            "package-inventory",
            "restricted-fingerprint",
            "secrets",
            "text",
        )
        member = AttachmentInventoryMemberV2(
            inventory_member_id=("attachment-inventory-member://pending"),
            artifact_id=request.artifact_id,
            output_ref=request.output_ref,
            normalized_path=request.logical_path,
            member_type=AttachmentInventoryMemberTypeV2.FILE,
            media_type=request.media_type,
            size_bytes=32,
            content_sha256=request.output_sha256,
            container_ref=None,
            inventory_member_sha256="0" * 64,
        )
        member_digest = attachment_inventory_member_carried_sha256(member)
        member = member.model_copy(
            update={
                "inventory_member_id": (f"attachment-inventory-member://sha256/{member_digest}"),
                "inventory_member_sha256": member_digest,
            }
        )
        result = AttachmentValidationResultV2(
            validation_result_id=("attachment-validation-result://pending"),
            validation_request_ref=attachment_validation_request_ref(request),
            artifact_id=request.artifact_id,
            output_ref=request.output_ref,
            output_sha256=request.output_sha256,
            status=AttachmentValidationStatusV2.PASSED,
            executed_validator_ids=required,
            required_validator_ids=required,
            complete_leakage_categories=(request.complete_leakage_categories),
            findings=(),
            inventory_members=(member,),
            inventory_complete=True,
            scan_complete=True,
            failure_code=None,
            policy_version="attachment-validation/r5-08-v1",
            validation_result_sha256="0" * 64,
        )
        digest = attachment_validation_result_carried_sha256(result)
        return result.model_copy(
            update={
                "validation_result_id": (f"attachment-validation-result://sha256/{digest}"),
                "validation_result_sha256": digest,
            }
        )


class _FixtureSemanticMaterialResolver:
    def resolve_review_material(
        self,
        request: AttachmentSemanticReviewRequestV2,
    ) -> ProviderSemanticReviewMaterial:
        return ProviderSemanticReviewMaterial(
            context_view_ref=request.context_view_ref,
            included_ref_inventory=tuple(
                sorted(
                    (
                        request.candidate_revision_ref,
                        request.deterministic_validation_result_ref,
                        request.context_view_ref,
                        *request.current_artifact_version_refs,
                        *request.current_output_refs,
                        *request.prior_finding_refs,
                        *request.prior_resolution_refs,
                        *request.prior_repair_plan_refs,
                        *request.prior_repair_result_refs,
                    ),
                    key=_facade_ref_key,
                )
            ),
            denied_data_families=(
                "BUILD_TRANSCRIPT",
                "HIDDEN_REASONING",
                "RAW_PRIVATE_REFERENCE",
                "RAW_TRACE",
            ),
            allowed_evidence_ref_ids=frozenset(),
            safe_payload={"round": request.round.value},
        )

    def resolve_repair_material(
        self,
        request: AttachmentRepairRequestV2,
    ) -> ProviderAttachmentRepairMaterial:
        raise AssertionError(f"clean semantic workflow cannot repair {request.repair_request_id}")


class _FixtureAcceptingSemanticBackend:
    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        del context
        return ProviderSemanticReviewDecision.accepted()


def _success(
    *,
    private_store: FactoryPrivateObjectStore,
    output_ref: ObjectRef,
) -> ProviderInvocationResult:
    return ProviderInvocationResult(
        status=GatewayInvocationStatusV2.SUCCEEDED,
        response_body_ref=private_store.put_text(
            object_type="model-response-content",
            text=f"fixture:{output_ref.object_id}",
        ),
        output_ref=output_ref,
        usage=GatewayUsageV2(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=0,
            reported_cost_micro_usd=0,
        ),
        failure_code=None,
    )


def _require_ref(
    reference: ObjectRef,
    *,
    object_type: str,
    label: str,
) -> None:
    if reference.object_type != object_type or reference.object_version != "v2":
        raise ValueError(f"{label} must reference {object_type}/v2")


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _facade_ref_key(
    value: FacadeObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "FactoryDatasetJobStoreRuntimeConfigV1",
    "FactoryDatasetSpecialistRuntimeConfigV1",
    "FactoryFixtureSpecialistComponents",
    "FactoryFixtureSpecialistContextFactory",
    "FixtureSpecialistGatewayProvider",
    "build_fixture_specialist_components",
]
