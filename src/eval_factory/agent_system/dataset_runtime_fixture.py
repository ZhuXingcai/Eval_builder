from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from env_mock_agent.facade import (
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    AttachmentExecutionStatusV2,
    AttachmentRouteCandidateKind,
    AttachmentRouteDecisionV2,
    AttachmentRouteOutcome,
    AttachmentRouteRequestV2,
    ExecutionTelemetryV2,
    FacadeObjectRef,
    WorldLedgerSnapshotRequestV2,
    WorldLedgerSnapshotV2,
    attachment_execution_request_ref,
    attachment_execution_result_carried_sha256,
    attachment_route_decision_carried_sha256,
    attachment_route_request_ref,
    world_ledger_snapshot_carried_sha256,
    world_ledger_snapshot_request_ref,
)
from eval_factory.agent_system.attachment_item_runtime import (
    FactoryAttachmentItemRuntime,
)
from eval_factory.agent_system.attachment_planner import (
    AttachmentPlanningAgentConfig,
    AttachmentPlanningInputV1,
    GatewayAttachmentPlanningAgent,
)
from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
)
from eval_factory.agent_system.attachment_preparation_builder import (
    FactoryAttachmentPreparationBuilder,
    FactoryAttachmentPreparationConfig,
)
from eval_factory.agent_system.attachment_preparation_material import (
    FactoryAttachmentPreparationMaterialStore,
)
from eval_factory.agent_system.attachment_registry import (
    AttachmentAgentRegistryConfig,
    build_attachment_agent_registry,
)
from eval_factory.agent_system.attachment_subgraph import (
    SupervisedAttachmentR5Runner,
)
from eval_factory.agent_system.core_material import FactoryCoreMaterialStore
from eval_factory.agent_system.core_vertical import CoreVerticalRunner
from eval_factory.agent_system.dataset_release_fixture import (
    FactoryDatasetReleaseRuntimeConfigV1,
    FactoryFixtureReleaseComponents,
    build_fixture_release_components,
    fixture_attachment_export_payload,
)
from eval_factory.agent_system.dataset_runtime import FactoryDatasetRuntime
from eval_factory.agent_system.dataset_specialist_fixture import (
    FactoryDatasetSpecialistRuntimeConfigV1,
    FactoryFixtureSpecialistComponents,
    FixtureSpecialistGatewayProvider,
    build_fixture_specialist_components,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.agent_system.r4_authoring_agent import (
    GatewayR4AuthoringAgent,
    R4AuthoringAgentConfig,
    RubricAgentInputV1,
    TaskDraftAgentInputV1,
    TaskEpisodeAgentInputV1,
    TaskPromptSafetyAgentInputV1,
)
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.specialists import (
    CoreSemanticInputV1,
    CoreSemanticOperationV1,
    GatewayIntentRewriteAgent,
    GatewayRequirementPlannerAgent,
    RequirementPlannerConfig,
    RequirementPlanningInputV1,
    SemanticAgentConfig,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.supervisor import ExecutionSupervisor
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridge,
)
from eval_factory.agent_system.task_authoring_material import (
    FactoryTaskAuthoringMaterialStore,
)
from eval_factory.agent_system.workspace import AgentWorkspaceManager
from eval_factory.ai_gateway.invocation import (
    EmbeddedAIGateway,
    ProviderInvocationResult,
)
from eval_factory.ai_gateway.model_catalog import ModelCatalog
from eval_factory.ai_gateway.prompt_registry import PromptRegistry
from eval_factory.ai_gateway.receipts import GatewayRecordStore
from eval_factory.ai_gateway.routing import ModelRouter
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    AttachmentGenerationPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    InferredUserIntentV2,
    IntentClaimV2,
    PlanKindV2,
    TaskRewriteCandidateV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
    ModelRoutingPolicyV2,
    PromptTemplateV2,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactRoutingPolicyV2,
    PublicSourceRetrievalPolicyV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.task import (
    AttachmentCriticality,
    EvidencePriority,
    RequirementConflict,
    RubricVisibility,
)
from eval_factory.contracts.task_v2 import (
    RubricJudgedObjectKindV2,
)
from eval_factory.task_authoring import (
    FakeRubricGenerationFixture,
    FakeRubricGenerationRunner,
    FakeTaskDraftAuthoringFixture,
    FakeTaskDraftAuthoringRunner,
    FakeTaskEpisodeGroupingFixture,
    FakeTaskEpisodeGroupingRunner,
    FakeTaskPromptSafetyFixture,
    FakeTaskPromptSafetyRunner,
    RubricAuthoringOutcome,
    RubricCriterionSelection,
    TaskDraftAttachmentFixture,
    TaskDraftAuthoringOutcome,
    TaskDraftContentFixture,
    TaskDraftRequirementFixture,
    TaskEpisodeGroupFixture,
    TaskEpisodeGroupingOutcome,
    TaskEpisodeSegmentSelection,
    TaskPromptSafetyOutcome,
)


@dataclass(frozen=True, slots=True)
class FactoryDatasetFixtureComponents:
    runtime: FactoryDatasetRuntime
    private_store: FactoryPrivateObjectStore
    gateway: EmbeddedAIGateway
    specialist: FactoryFixtureSpecialistComponents | None
    release: FactoryFixtureReleaseComponents | None


class FactoryDatasetRuntimeConfigV1(ContractModelV2):
    schema_version: Literal["eval-factory/factory-dataset-runtime-config/v1"] = (
        "eval-factory/factory-dataset-runtime-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    capabilities: tuple[AgentCapabilityV2, ...] = Field(
        min_length=1,
        max_length=256,
    )
    definitions: tuple[AgentDefinitionV2, ...] = Field(
        min_length=1,
        max_length=256,
    )
    model_profiles: tuple[ModelCapabilityProfileV2, ...] = Field(
        min_length=1,
        max_length=256,
    )
    quality_baselines: tuple[ModelQualityBaselineV2, ...] = Field(
        min_length=1,
        max_length=256,
    )
    health_snapshots: tuple[ModelHealthSnapshotV2, ...] = Field(
        min_length=1,
        max_length=256,
    )
    price_schedules: tuple[ModelPriceScheduleV2, ...] = Field(
        min_length=1,
        max_length=256,
    )
    routing_policy: ModelRoutingPolicyV2
    planning_prompt: PromptTemplateV2
    planning_agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    budget_reservation_ref: ObjectRef

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if (
            self.planning_agent_definition_ref.object_type != "agent-definition"
            or self.planning_agent_definition_ref.object_version != "v2"
        ):
            raise ValueError("planning Agent ref must be agent-definition/v2")
        if (
            self.budget_reservation_ref.object_type != "work-model-reservation"
            or self.budget_reservation_ref.object_version != "v2"
        ):
            raise ValueError("budget ref must be work-model-reservation/v2")
        profile_refs = tuple(
            sorted(
                (profile.to_ref() for profile in self.model_profiles),
                key=_ref_key,
            )
        )
        if self.allowed_model_profile_refs != tuple(
            sorted(
                set(self.allowed_model_profile_refs),
                key=_ref_key,
            )
        ) or not set(self.allowed_model_profile_refs).issubset(profile_refs):
            raise ValueError("allowed model profiles must be current and canonical")
        if self.planning_prompt.task_kind != "planning":
            raise ValueError("runtime config requires a planning prompt")
        return self


class FactoryDatasetPlanningFixtureV1(ContractModelV2):
    schema_version: Literal["eval-factory/factory-dataset-planning-fixture/v1"] = (
        "eval-factory/factory-dataset-planning-fixture/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    stage_order: tuple[str, ...] = Field(
        min_length=1,
        max_length=64,
    )
    tasks: tuple[DatasetBuildPlanTaskV2, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    required_review_kinds: tuple[PlanKindV2, ...] = Field(
        min_length=1,
        max_length=7,
    )
    total_model_requests: int = Field(
        ge=0,
        le=10_000_000,
    )
    total_model_tokens: int = Field(
        ge=0,
        le=10_000_000_000,
    )
    total_cost_micro_usd: int = Field(
        ge=0,
        le=10_000_000_000_000,
    )

    @model_validator(mode="after")
    def validate_fixture(self) -> Self:
        if (
            len(set(self.stage_order)) != len(self.stage_order)
            or tuple(
                sorted(
                    self.tasks,
                    key=lambda value: value.task_key,
                )
            )
            != self.tasks
            or len(set(self.required_review_kinds)) != len(self.required_review_kinds)
        ):
            raise ValueError("planning fixture inventories must be canonical")
        if (
            self.total_model_requests != sum(task.max_model_requests for task in self.tasks)
            or self.total_model_tokens != sum(task.max_model_tokens for task in self.tasks)
            or self.total_cost_micro_usd != sum(task.max_cost_micro_usd for task in self.tasks)
        ):
            raise ValueError("planning fixture totals differ from tasks")
        return self


class FactoryDatasetCoreRuntimeConfigV1(ContractModelV2):
    schema_version: Literal["eval-factory/factory-dataset-core-runtime-config/v1"] = (
        "eval-factory/factory-dataset-core-runtime-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    intent_prompt: PromptTemplateV2
    rewrite_prompt: PromptTemplateV2
    intent_agent_definition_ref: ObjectRef
    rewrite_agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    intent_budget_reservation_ref: ObjectRef
    rewrite_budget_reservation_ref: ObjectRef
    blocked_extracted_prompt_refs: tuple[
        ObjectRef,
        ...,
    ] = Field(default=(), max_length=1_000_000)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.intent_prompt.task_kind != "extraction" or self.rewrite_prompt.task_kind != "rewrite":
            raise ValueError("core runtime config requires extraction and rewrite prompts")
        for reference in (
            self.intent_agent_definition_ref,
            self.rewrite_agent_definition_ref,
        ):
            if reference.object_type != "agent-definition" or reference.object_version != "v2":
                raise ValueError("core runtime Agent refs must be agent-definition/v2")
        for reference in (
            self.intent_budget_reservation_ref,
            self.rewrite_budget_reservation_ref,
        ):
            if reference.object_type != "work-model-reservation" or reference.object_version != "v2":
                raise ValueError("core runtime budget refs must be work-model-reservation/v2")
        if self.allowed_model_profile_refs != tuple(
            sorted(
                set(self.allowed_model_profile_refs),
                key=_ref_key,
            )
        ):
            raise ValueError("core runtime model profiles must be canonical")
        if self.blocked_extracted_prompt_refs != tuple(
            sorted(
                set(self.blocked_extracted_prompt_refs),
                key=_ref_key,
            )
        ) or any(
            reference.object_type != "extracted-user-prompt" or reference.object_version != "v2"
            for reference in self.blocked_extracted_prompt_refs
        ):
            raise ValueError("blocked core prompt refs must be canonical")
        return self


class FactoryDatasetR4RuntimeConfigV1(ContractModelV2):
    schema_version: Literal["eval-factory/factory-dataset-r4-runtime-config/v1"] = (
        "eval-factory/factory-dataset-r4-runtime-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    episode_prompt: PromptTemplateV2
    draft_prompt: PromptTemplateV2
    safety_prompt: PromptTemplateV2
    rubric_prompt: PromptTemplateV2
    episode_agent_definition_ref: ObjectRef
    draft_agent_definition_ref: ObjectRef
    safety_agent_definition_ref: ObjectRef
    rubric_agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[
        ObjectRef,
        ...,
    ] = Field(min_length=1, max_length=256)
    episode_budget_reservation_ref: ObjectRef
    draft_budget_reservation_ref: ObjectRef
    safety_budget_reservation_ref: ObjectRef
    rubric_budget_reservation_ref: ObjectRef

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        prompts = (
            self.episode_prompt,
            self.draft_prompt,
            self.safety_prompt,
            self.rubric_prompt,
        )
        if tuple(prompt.task_kind for prompt in prompts) != (
            "task-episode-grouping",
            "task-draft-authoring",
            "task-prompt-safety",
            "rubric-authoring",
        ):
            raise ValueError("R4 runtime prompts use the wrong task kinds")
        for reference in (
            self.episode_agent_definition_ref,
            self.draft_agent_definition_ref,
            self.safety_agent_definition_ref,
            self.rubric_agent_definition_ref,
        ):
            if reference.object_type != "agent-definition" or reference.object_version != "v2":
                raise ValueError("R4 runtime Agent refs must be agent-definition/v2")
        for reference in (
            self.episode_budget_reservation_ref,
            self.draft_budget_reservation_ref,
            self.safety_budget_reservation_ref,
            self.rubric_budget_reservation_ref,
        ):
            if reference.object_type != "work-model-reservation" or reference.object_version != "v2":
                raise ValueError("R4 runtime budget refs must be work-model-reservation/v2")
        if self.allowed_model_profile_refs != tuple(
            sorted(
                set(self.allowed_model_profile_refs),
                key=_ref_key,
            )
        ):
            raise ValueError("R4 runtime model profiles must be canonical")
        return self


class FactoryDatasetAttachmentRuntimeConfigV1(ContractModelV2):
    schema_version: Literal["eval-factory/factory-dataset-attachment-runtime-config/v1"] = (
        "eval-factory/factory-dataset-attachment-runtime-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    planning_prompt: PromptTemplateV2
    planning_agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[
        ObjectRef,
        ...,
    ] = Field(min_length=1, max_length=256)
    planning_budget_reservation_ref: ObjectRef
    retrieval_policy: PublicSourceRetrievalPolicyV2
    routing_policy: ArtifactRoutingPolicyV2
    world_ledger_ref: FacadeObjectRef
    mock_prompt_ref: ObjectRef
    quality_prompt_ref: ObjectRef
    solvability_prompt_ref: ObjectRef
    mock_model_policy_ref: ObjectRef
    quality_model_policy_ref: ObjectRef
    solvability_model_policy_ref: ObjectRef

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.planning_prompt.task_kind != "attachment-planning":
            raise ValueError("attachment runtime config requires an attachment-planning prompt")
        if (
            self.planning_agent_definition_ref.object_type != "agent-definition"
            or self.planning_agent_definition_ref.object_version != "v2"
        ):
            raise ValueError("attachment runtime planning Agent ref must be agent-definition/v2")
        if (
            self.planning_budget_reservation_ref.object_type != "work-model-reservation"
            or self.planning_budget_reservation_ref.object_version != "v2"
        ):
            raise ValueError("attachment runtime budget ref must be work-model-reservation/v2")
        if self.allowed_model_profile_refs != tuple(
            sorted(
                set(self.allowed_model_profile_refs),
                key=_ref_key,
            )
        ):
            raise ValueError("attachment runtime model profiles must be canonical")
        if (
            self.world_ledger_ref.object_type != "world-ledger"
            or self.world_ledger_ref.object_version != "v2"
        ):
            raise ValueError("attachment runtime world ledger ref must be world-ledger/v2")
        for reference in (
            self.mock_prompt_ref,
            self.quality_prompt_ref,
            self.solvability_prompt_ref,
        ):
            if reference.object_type != "prompt-template" or reference.object_version != "v2":
                raise ValueError("attachment runtime prompt refs must be prompt-template/v2")
        for reference in (
            self.mock_model_policy_ref,
            self.quality_model_policy_ref,
            self.solvability_model_policy_ref,
        ):
            if reference.object_type != "model-routing-policy" or reference.object_version != "v2":
                raise ValueError("attachment runtime model policy refs must be model-routing-policy/v2")
        return self


class FixturePlanningGatewayProvider:
    def __init__(
        self,
        *,
        private_store: FactoryPrivateObjectStore,
        fixture: FactoryDatasetPlanningFixtureV1,
        planning_prompt_ref: ObjectRef,
        core_config: FactoryDatasetCoreRuntimeConfigV1 | None = None,
        r4_config: FactoryDatasetR4RuntimeConfigV1 | None = None,
        attachment_config: (FactoryDatasetAttachmentRuntimeConfigV1 | None) = None,
        specialist_provider: (FixtureSpecialistGatewayProvider | None) = None,
    ) -> None:
        self.private_store = private_store
        self.fixture = fixture
        self.planning_prompt_ref = planning_prompt_ref
        self.core_config = core_config
        self.r4_config = r4_config
        self.attachment_config = attachment_config
        self.specialist_provider = specialist_provider

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        if request.prompt_template_ref != self.planning_prompt_ref:
            if self.specialist_provider is not None and self.specialist_provider.supports(
                request.prompt_template_ref
            ):
                return await self.specialist_provider.invoke(
                    request,
                    model_profile=model_profile,
                )
            if (
                self.attachment_config is not None
                and request.prompt_template_ref == self.attachment_config.planning_prompt.to_ref()
            ):
                return self._invoke_attachment(
                    request,
                    model_profile=model_profile,
                )
            if self.r4_config is not None and request.prompt_template_ref in {
                self.r4_config.episode_prompt.to_ref(),
                self.r4_config.draft_prompt.to_ref(),
                self.r4_config.safety_prompt.to_ref(),
                self.r4_config.rubric_prompt.to_ref(),
            }:
                return self._invoke_r4(
                    request,
                    model_profile=model_profile,
                )
            return self._invoke_core(
                request,
                model_profile=model_profile,
            )
        if "planning" not in model_profile.capabilities:
            raise ValueError("fixture provider requires planning capability")
        planning = self.private_store.get_model(
            request.prompt_rendering_ref,
            RequirementPlanningInputV1,
        )
        plan = DatasetBuildPlanV2.create(
            plan_id=(f"dataset-build-plan://{planning.requirement_spec_ref.object_sha256[:32]}"),
            run_ref=planning.run_ref,
            plan_version=1,
            predecessor_plan_ref=None,
            goals=planning.goals,
            user_constraints=planning.constraints,
            assumptions=planning.assumptions,
            unresolved_questions=planning.open_questions,
            stage_order=self.fixture.stage_order,
            tasks=self.fixture.tasks,
            required_review_kinds=(self.fixture.required_review_kinds),
            total_model_requests=(self.fixture.total_model_requests),
            total_model_tokens=self.fixture.total_model_tokens,
            total_cost_micro_usd=(self.fixture.total_cost_micro_usd),
            audit=request.audit,
        )
        output_ref = self.private_store.put_model(
            object_type="dataset-build-plan",
            value=plan,
        )
        return self._success(output_ref)

    def _invoke_attachment(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        if "attachment-planning" not in model_profile.capabilities:
            raise ValueError("fixture provider requires attachment-planning capability")
        planning = self.private_store.get_model(
            request.prompt_rendering_ref,
            AttachmentPlanningInputV1,
        )
        suffix = hashlib.sha256(request.prompt_rendering_ref.object_id.encode()).hexdigest()[:32]
        plan = AttachmentGenerationPlanV2.create(
            plan_id=(f"attachment-generation-plan://fixture-provider/{suffix}"),
            run_ref=planning.run_ref,
            plan_version=1,
            predecessor_plan_ref=None,
            producer_task_view_ref=planning.producer_task_view_ref,
            evidence_bundle_ref=planning.evidence_bundle_ref,
            attachment_planning_context_ref=(planning.attachment_planning_context_ref),
            works=planning.work_templates,
            max_parallel_groups=len(planning.work_templates),
            quality_policy_ref=planning.quality_policy_ref,
            solvability_policy_ref=planning.solvability_policy_ref,
            total_model_requests=sum(value.max_model_requests for value in planning.work_templates),
            total_model_tokens=sum(value.max_model_tokens for value in planning.work_templates),
            total_cost_micro_usd=sum(value.max_cost_micro_usd for value in planning.work_templates),
            audit=request.audit,
        )
        return self._success(
            self.private_store.put_model(
                object_type="attachment-generation-plan",
                value=plan,
            )
        )

    def _invoke_r4(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        capability = next(
            (value for value in model_profile.capabilities if value != "structured-output"),
            None,
        )
        if capability == "task-episode-grouping":
            episode_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                TaskEpisodeAgentInputV1,
            )
            user_segment = next(
                (
                    segment
                    for segment in episode_input.request.segments
                    if segment.boundary_method == "user_turn"
                ),
                episode_input.request.segments[0],
            )
            episode_proposal = FakeTaskEpisodeGroupingRunner().run(
                episode_input.request,
                fixture=FakeTaskEpisodeGroupingFixture(
                    fixture_id=("r4-provider-fixture://episode"),
                    outcome=(TaskEpisodeGroupingOutcome.GROUPED),
                    groups=(
                        TaskEpisodeGroupFixture(
                            group_id=("task-episode-group://fixture-provider"),
                            selections=(
                                TaskEpisodeSegmentSelection(
                                    segment_id=(user_segment.segment_id),
                                    evidence_ref_ids=(
                                        episode_input.request.evidence_refs[0].evidence_ref_id,
                                    ),
                                ),
                            ),
                            rationale_ref=_fixture_ref(
                                "task-episode-rationale",
                                "fixture-provider",
                            ),
                        ),
                    ),
                    unresolved_reasons=frozenset(),
                ),
                audit=request.audit,
            )
            return self._success(
                self.private_store.put_model(
                    object_type=("task-episode-grouping-proposal"),
                    value=episode_proposal,
                )
            )
        elif capability == "task-draft-authoring":
            draft_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                TaskDraftAgentInputV1,
            )
            evidence_id = draft_input.request.evidence_views[0].evidence_ref_id
            episode_id = draft_input.request.task_episode_refs[0].object_id
            requirement_id = "requirement://fixture-provider/primary"
            draft_proposal = FakeTaskDraftAuthoringRunner().run(
                draft_input.request,
                fixture=FakeTaskDraftAuthoringFixture(
                    fixture_id=("r4-provider-fixture://draft"),
                    outcome=(TaskDraftAuthoringOutcome.DRAFTED),
                    content=TaskDraftContentFixture(
                        visible_prompt=(draft_input.rewritten_prompt),
                        task_intent=draft_input.task_intent,
                        evaluation_claim=(draft_input.evaluation_claim),
                        required_capability_ids=("instruction-following",),
                        allowed_tool_ids=(),
                        forbidden_outputs=(draft_input.request.required_forbidden_outputs),
                        attachment_dependencies=(
                            TaskDraftAttachmentFixture(
                                dependency_id=("attachment-dependency://fixture-provider/context"),
                                description=("Use inputs/context.txt."),
                                criticality=(AttachmentCriticality.REQUIRED),
                                evidence_priority=(EvidencePriority.EXPLICIT_REQUIREMENT),
                                evidence_ref_ids=(evidence_id,),
                            ),
                        ),
                        requirements=(
                            TaskDraftRequirementFixture(
                                requirement_id=(requirement_id),
                                statement=(draft_input.evaluation_claim),
                                criticality=(AttachmentCriticality.CRITICAL),
                                evidence_priority=(EvidencePriority.EXPLICIT_REQUIREMENT),
                                evidence_ref_ids=(evidence_id,),
                                task_episode_ids=(episode_id,),
                                conflict_status=(RequirementConflict.NONE),
                            ),
                        ),
                        prompt_requirement_ids=(requirement_id,),
                        uncertainties=(),
                    ),
                    unresolved_reasons=frozenset(),
                ),
                audit=request.audit,
            )
            return self._success(
                self.private_store.put_model(
                    object_type=("task-draft-authoring-proposal"),
                    value=draft_proposal,
                )
            )
        elif capability == "rubric-authoring":
            rubric_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                RubricAgentInputV1,
            )
            rubric_proposal = FakeRubricGenerationRunner().run(
                rubric_input.request,
                fixture=FakeRubricGenerationFixture(
                    fixture_id=("r4-provider-fixture://rubric"),
                    outcome=(RubricAuthoringOutcome.COMPILED),
                    criteria=(
                        RubricCriterionSelection(
                            selection_id=("rubric-criterion-selection://fixture-provider/response"),
                            judged_object_id=("judged-object://fixture-provider/response"),
                            judged_object_kind=(RubricJudgedObjectKindV2.CONTESTANT_RESPONSE),
                            judged_object_description=("The contestant response."),
                            description=(rubric_input.request.evaluation_claim),
                            weight=1.0,
                            prompt_requirement_ids=tuple(
                                item.requirement_id for item in rubric_input.request.prompt_requirements
                            ),
                            attachment_dependency_ids=(),
                            allowed_tool_ids=(),
                            visibility=(RubricVisibility.EVALUATOR_ONLY),
                            evaluator_binding=("evaluator-binding://factory-dataset/default"),
                        ),
                    ),
                    unresolved_reasons=frozenset(),
                    model_available=True,
                ),
                model_profile=("internal-rubric-author-v1"),
                prompt_version=("rubric-authoring-prompt/v1"),
                audit=request.audit,
            )
            return self._success(
                self.private_store.put_model(
                    object_type=("rubric-candidate-proposal"),
                    value=rubric_proposal,
                )
            )
        elif capability == "task-prompt-safety":
            safety_input = self.private_store.get_model(
                request.prompt_rendering_ref,
                TaskPromptSafetyAgentInputV1,
            )
            safety_proposal = FakeTaskPromptSafetyRunner().run(
                safety_input.request,
                fixture=FakeTaskPromptSafetyFixture(
                    fixture_id=("r4-provider-fixture://safety"),
                    outcome=(TaskPromptSafetyOutcome.PASSED),
                    findings=(),
                    unresolved_reasons=frozenset(),
                    model_available=True,
                ),
                audit=request.audit,
            )
            return self._success(
                self.private_store.put_model(
                    object_type=("task-prompt-safety-proposal"),
                    value=safety_proposal,
                )
            )
        else:
            raise ValueError("fixture provider R4 capability is unavailable")

    def _invoke_core(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        config = self.core_config
        if config is None:
            raise ValueError("fixture provider core capability is unavailable")
        semantic_input = self.private_store.get_model(
            request.prompt_rendering_ref,
            CoreSemanticInputV1,
        )
        suffix = semantic_input.extracted_prompt_ref.object_sha256[:32]
        if semantic_input.operation is CoreSemanticOperationV1.INFER_INTENT:
            if "extraction" not in model_profile.capabilities:
                raise ValueError("fixture provider requires extraction capability")
            if semantic_input.extracted_prompt_ref in config.blocked_extracted_prompt_refs:
                return ProviderInvocationResult(
                    status=GatewayInvocationStatusV2.FAILED,
                    response_body_ref=None,
                    output_ref=None,
                    usage=GatewayUsageV2(
                        input_tokens=0,
                        output_tokens=0,
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                        charged_tokens=0,
                        reported_cost_micro_usd=0,
                    ),
                    failure_code="MODEL_PROVIDER_FAILED",
                )
            intent = InferredUserIntentV2.create(
                inferred_intent_id=f"inferred-user-intent://{suffix}",
                extracted_prompt_ref=semantic_input.extracted_prompt_ref,
                claims=(
                    IntentClaimV2(
                        claim_id=f"intent-claim://{suffix}",
                        summary=("Complete the source-grounded user-requested task."),
                        evidence_refs=(semantic_input.extracted_prompt_ref,),
                        confidence_basis_points=8_500,
                        uncertain=False,
                    ),
                ),
                unresolved_requirements=(),
                abstained=False,
                audit=request.audit,
            )
            output_ref = self.private_store.put_model(
                object_type="inferred-user-intent",
                value=intent,
            )
            return self._success(output_ref)
        if "rewrite" not in model_profile.capabilities:
            raise ValueError("fixture provider requires rewrite capability")
        if semantic_input.inferred_intent_ref is None or semantic_input.rewrite_plan_ref is None:
            raise ValueError("fixture rewrite input is incomplete")
        rewritten_prompt_ref = self.private_store.put_text(
            object_type="rewritten-prompt-content",
            text=(
                "Evaluation task reconstructed from the "
                "source-bound user prompt:\n"
                f"{semantic_input.prompt_text}"
            ),
        )
        candidate = TaskRewriteCandidateV2.create(
            candidate_id=f"task-rewrite-candidate://{suffix}",
            extracted_prompt_ref=semantic_input.extracted_prompt_ref,
            inferred_intent_ref=semantic_input.inferred_intent_ref,
            rewrite_plan_ref=semantic_input.rewrite_plan_ref,
            rewritten_prompt_ref=rewritten_prompt_ref,
            evidence_refs=tuple(
                sorted(
                    (
                        semantic_input.extracted_prompt_ref,
                        semantic_input.inferred_intent_ref,
                    ),
                    key=_ref_key,
                )
            ),
            status="REVIEW_READY",
            audit=request.audit,
        )
        output_ref = self.private_store.put_model(
            object_type="task-rewrite-candidate",
            value=candidate,
        )
        return self._success(output_ref)

    def _success(
        self,
        output_ref: ObjectRef,
    ) -> ProviderInvocationResult:
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=self.private_store.put_text(
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


class _FixtureAttachmentRoutingFacade:
    def __init__(
        self,
        *,
        observed_at: datetime,
    ) -> None:
        self.observed_at = observed_at

    async def route(
        self,
        request: AttachmentRouteRequestV2,
    ) -> AttachmentRouteDecisionV2:
        decision = AttachmentRouteDecisionV2(
            route_decision_id="attachment-route-decision://pending",
            route_request_ref=attachment_route_request_ref(request),
            outcome=AttachmentRouteOutcome.SELECTED_PROVIDER,
            selected_kind=AttachmentRouteCandidateKind.PROVIDER,
            selected_id="text",
            selected_version="fixture-v1",
            satisfied_capability_ids=(request.required_provider_capability_ids),
            skipped=(),
            capability_snapshot_sha256=request.route_request_sha256,
            probed_at=self.observed_at,
            policy_version="artifact-routing/r5-05-v1",
            route_decision_sha256="0" * 64,
        )
        digest = attachment_route_decision_carried_sha256(decision)
        return decision.model_copy(
            update={
                "route_decision_id": (f"attachment-route-decision://sha256/{digest}"),
                "route_decision_sha256": digest,
            }
        )


class _FixtureAttachmentExecutionFacade:
    def __init__(
        self,
        *,
        observed_at: datetime,
    ) -> None:
        self.observed_at = observed_at

    async def snapshot_world(
        self,
        request: WorldLedgerSnapshotRequestV2,
    ) -> WorldLedgerSnapshotV2:
        snapshot = WorldLedgerSnapshotV2(
            world_ledger_snapshot_id="world-ledger-snapshot://pending",
            snapshot_request_ref=world_ledger_snapshot_request_ref(request),
            world_ledger_ref=request.world_ledger_ref,
            fact_locks=(),
            ledger_sha256=request.world_ledger_ref.object_sha256,
            policy_version="artifact-execution/r5-06-v1",
            snapshotted_at=self.observed_at,
            world_ledger_snapshot_sha256="0" * 64,
        )
        digest = world_ledger_snapshot_carried_sha256(snapshot)
        return snapshot.model_copy(
            update={
                "world_ledger_snapshot_id": (f"world-ledger-snapshot://sha256/{digest}"),
                "world_ledger_snapshot_sha256": digest,
            }
        )

    async def execute(
        self,
        request: AttachmentExecutionRequestV2,
    ) -> AttachmentExecutionResultV2:
        output_sha256 = hashlib.sha256(
            fixture_attachment_export_payload(
                request.artifact_id,
            )
        ).hexdigest()
        result = AttachmentExecutionResultV2(
            execution_result_id="attachment-execution-result://pending",
            execution_request_ref=attachment_execution_request_ref(request),
            artifact_id=request.artifact_id,
            attempt=request.attempt,
            status=AttachmentExecutionStatusV2.SUCCEEDED,
            world_ledger_snapshot_ref=request.world_ledger_snapshot_ref,
            selected_route_kind=request.selected_route_kind,
            selected_route_id=request.selected_route_id,
            worker_version="fixture-v1",
            output_ref=FacadeObjectRef(
                object_type="attachment-output",
                object_id=(f"attachment-output://sha256/{output_sha256}"),
                object_version="v2",
                object_sha256=output_sha256,
            ),
            output_sha256=output_sha256,
            retryable=False,
            failure_code=None,
            telemetry=ExecutionTelemetryV2.unavailable(
                observed_at=self.observed_at,
            ),
            policy_version="artifact-execution/r5-06-v1",
            execution_result_sha256="0" * 64,
        )
        digest = attachment_execution_result_carried_sha256(result)
        return result.model_copy(
            update={
                "execution_result_id": (f"attachment-execution-result://sha256/{digest}"),
                "execution_result_sha256": digest,
            }
        )


def build_fixture_dataset_components(
    *,
    factory_store_path: Path,
    private_store_path: Path,
    gateway_store_path: Path,
    config: FactoryDatasetRuntimeConfigV1,
    fixture: FactoryDatasetPlanningFixtureV1,
    requested_by: str,
    core_config: FactoryDatasetCoreRuntimeConfigV1 | None = None,
    r4_config: FactoryDatasetR4RuntimeConfigV1 | None = None,
    core_workspace_path: Path | None = None,
    attachment_config: (FactoryDatasetAttachmentRuntimeConfigV1 | None) = None,
    attachment_workspace_path: Path | None = None,
    specialist_config: (FactoryDatasetSpecialistRuntimeConfigV1 | None) = None,
    job_store_path: Path | None = None,
    specialist_workspace_path: Path | None = None,
    release_config: (FactoryDatasetReleaseRuntimeConfigV1 | None) = None,
    candidate_output_path: Path | None = None,
) -> FactoryDatasetFixtureComponents:
    roots = {
        factory_store_path.expanduser().resolve(),
        private_store_path.expanduser().resolve(),
        gateway_store_path.expanduser().resolve(),
    }
    if core_workspace_path is not None:
        roots.add(core_workspace_path.expanduser().resolve())
    if attachment_workspace_path is not None:
        roots.add(attachment_workspace_path.expanduser().resolve())
    if job_store_path is not None:
        roots.add(job_store_path.expanduser().resolve())
    if specialist_workspace_path is not None:
        roots.add(specialist_workspace_path.expanduser().resolve())
    if candidate_output_path is not None:
        roots.add(candidate_output_path.expanduser().resolve())
    if (core_config is None) != (core_workspace_path is None):
        raise ValueError("core runtime config and workspace must appear together")
    if r4_config is not None and core_config is None:
        raise ValueError("R4 runtime config requires core execution")
    if (attachment_config is None) != (attachment_workspace_path is None):
        raise ValueError("attachment runtime config and workspace must appear together")
    if attachment_config is not None and r4_config is None:
        raise ValueError("attachment runtime config requires R4 execution")
    specialist_values = (
        specialist_config,
        job_store_path,
        specialist_workspace_path,
    )
    if any(value is not None for value in specialist_values) and not all(
        value is not None for value in specialist_values
    ):
        raise ValueError("specialist runtime config, JobStore, and workspace must appear together")
    if specialist_config is not None and attachment_config is None:
        raise ValueError("specialist runtime config requires attachment execution")
    if (release_config is None) != (candidate_output_path is None):
        raise ValueError("release runtime config and candidate output must appear together")
    if release_config is not None and specialist_config is None:
        raise ValueError("release runtime config requires specialist execution")
    if (
        release_config is not None
        and specialist_config is not None
        and release_config.approval_policy != specialist_config.job_store.approval_policy
    ):
        raise ValueError("release approval policy must match JobStore authority")
    expected_root_count = (
        3
        + (1 if core_config is not None else 0)
        + (1 if attachment_config is not None else 0)
        + (2 if specialist_config is not None else 0)
        + (1 if release_config is not None else 0)
    )
    if len(roots) != expected_root_count:
        raise ValueError("dataset runtime stores must use distinct roots")
    profile_refs = {profile.to_ref() for profile in config.model_profiles}
    if core_config is not None and not set(core_config.allowed_model_profile_refs).issubset(profile_refs):
        raise ValueError("core runtime model profiles must belong to the catalog")
    if r4_config is not None and not set(r4_config.allowed_model_profile_refs).issubset(profile_refs):
        raise ValueError("R4 runtime model profiles must belong to the catalog")
    if attachment_config is not None and not set(attachment_config.allowed_model_profile_refs).issubset(
        profile_refs
    ):
        raise ValueError("attachment runtime model profiles must belong to the catalog")
    if specialist_config is not None:
        specialist_profile_refs = {
            reference
            for values in (
                specialist_config.solvability_allowed_model_profile_refs,
                specialist_config.criteria_planning_allowed_model_profile_refs,
                specialist_config.criteria_allowed_model_profile_refs,
                specialist_config.grading_planning_allowed_model_profile_refs,
                specialist_config.grading_allowed_model_profile_refs,
            )
            for reference in values
        }
        if not specialist_profile_refs.issubset(profile_refs):
            raise ValueError("specialist runtime model profiles must belong to the catalog")
    private_store = FactoryPrivateObjectStore(
        private_store_path,
    )
    catalog = ModelCatalog(
        profiles=config.model_profiles,
        quality_baselines=config.quality_baselines,
        health_snapshots=config.health_snapshots,
        price_schedules=config.price_schedules,
    )
    specialist_provider = (
        FixtureSpecialistGatewayProvider(
            private_store=private_store,
            config=specialist_config,
        )
        if specialist_config is not None
        else None
    )
    gateway = EmbeddedAIGateway(
        router=ModelRouter(catalog, config.routing_policy),
        catalog=catalog,
        prompts=PromptRegistry(
            (
                config.planning_prompt,
                *(
                    (
                        core_config.intent_prompt,
                        core_config.rewrite_prompt,
                    )
                    if core_config is not None
                    else ()
                ),
                *(
                    (
                        r4_config.episode_prompt,
                        r4_config.draft_prompt,
                        r4_config.safety_prompt,
                        r4_config.rubric_prompt,
                    )
                    if r4_config is not None
                    else ()
                ),
                *((attachment_config.planning_prompt,) if attachment_config is not None else ()),
                *(specialist_config.prompt_templates() if specialist_config is not None else ()),
            )
        ),
        records=GatewayRecordStore(gateway_store_path),
        provider=FixturePlanningGatewayProvider(
            private_store=private_store,
            fixture=fixture,
            planning_prompt_ref=config.planning_prompt.to_ref(),
            core_config=core_config,
            r4_config=r4_config,
            attachment_config=attachment_config,
            specialist_provider=specialist_provider,
        ),
    )
    registry = AgentRegistry(
        capabilities=config.capabilities,
        definitions=config.definitions,
    )
    compiler = DatasetBuildPlanCompiler(registry)
    store = FactoryControlStore(factory_store_path)
    reviews = PlanReviewService(
        store,
        compiler=compiler,
    )
    planner = GatewayRequirementPlannerAgent(
        gateway=gateway,
        private_store=private_store,
        config=RequirementPlannerConfig(
            prompt=config.planning_prompt,
            agent_definition_ref=(config.planning_agent_definition_ref),
            allowed_model_profile_refs=(config.allowed_model_profile_refs),
            budget_reservation_ref=(config.budget_reservation_ref),
        ),
    )
    core_runner: CoreVerticalRunner | None = None
    if core_config is not None and core_workspace_path is not None:
        core_runner = CoreVerticalRunner(
            workspace=core_workspace_path,
            private_store=private_store,
            semantic_agent=GatewayIntentRewriteAgent(
                gateway=gateway,
                private_store=private_store,
                config=SemanticAgentConfig(
                    intent_prompt=core_config.intent_prompt,
                    rewrite_prompt=core_config.rewrite_prompt,
                    intent_agent_definition_ref=(core_config.intent_agent_definition_ref),
                    rewrite_agent_definition_ref=(core_config.rewrite_agent_definition_ref),
                    allowed_model_profile_refs=(core_config.allowed_model_profile_refs),
                    intent_budget_reservation_ref=(core_config.intent_budget_reservation_ref),
                    rewrite_budget_reservation_ref=(core_config.rewrite_budget_reservation_ref),
                ),
            ),
        )
    task_bridge: FactoryTaskAuthoringBridge | None = None
    if r4_config is not None and core_runner is not None:
        task_bridge = FactoryTaskAuthoringBridge(
            core_runner=core_runner,
            private_store=private_store,
            agent=GatewayR4AuthoringAgent(
                gateway=gateway,
                private_store=private_store,
                config=R4AuthoringAgentConfig(
                    episode_prompt=r4_config.episode_prompt,
                    draft_prompt=r4_config.draft_prompt,
                    safety_prompt=r4_config.safety_prompt,
                    rubric_prompt=r4_config.rubric_prompt,
                    episode_agent_definition_ref=(r4_config.episode_agent_definition_ref),
                    draft_agent_definition_ref=(r4_config.draft_agent_definition_ref),
                    safety_agent_definition_ref=(r4_config.safety_agent_definition_ref),
                    rubric_agent_definition_ref=(r4_config.rubric_agent_definition_ref),
                    allowed_model_profile_refs=(r4_config.allowed_model_profile_refs),
                    episode_budget_reservation_ref=(r4_config.episode_budget_reservation_ref),
                    draft_budget_reservation_ref=(r4_config.draft_budget_reservation_ref),
                    safety_budget_reservation_ref=(r4_config.safety_budget_reservation_ref),
                    rubric_budget_reservation_ref=(r4_config.rubric_budget_reservation_ref),
                ),
            ),
        )
    attachment_runtime: FactoryAttachmentItemRuntime | None = None
    if attachment_config is not None and attachment_workspace_path is not None:
        attachment_registry = build_attachment_agent_registry(
            config=AttachmentAgentRegistryConfig(
                mock_prompt_ref=attachment_config.mock_prompt_ref,
                quality_prompt_ref=attachment_config.quality_prompt_ref,
                solvability_prompt_ref=(attachment_config.solvability_prompt_ref),
                mock_model_policy_ref=(attachment_config.mock_model_policy_ref),
                quality_model_policy_ref=(attachment_config.quality_model_policy_ref),
                solvability_model_policy_ref=(attachment_config.solvability_model_policy_ref),
            ),
            audit=attachment_config.planning_prompt.audit,
        )
        attachment_execution = _FixtureAttachmentExecutionFacade(
            observed_at=attachment_config.planning_prompt.audit.created_at,
        )
        attachment_runtime = FactoryAttachmentItemRuntime(
            store=store,
            plan_reviews=reviews,
            planner=GatewayAttachmentPlanningAgent(
                gateway=gateway,
                private_store=private_store,
                compiler=AttachmentGenerationPlanCompiler(attachment_registry),
                config=AttachmentPlanningAgentConfig(
                    prompt=attachment_config.planning_prompt,
                    agent_definition_ref=(attachment_config.planning_agent_definition_ref),
                    allowed_model_profile_refs=(attachment_config.allowed_model_profile_refs),
                    budget_reservation_ref=(attachment_config.planning_budget_reservation_ref),
                ),
            ),
            preparation_builder=FactoryAttachmentPreparationBuilder(
                FactoryAttachmentPreparationConfig(
                    retrieval_policy=attachment_config.retrieval_policy,
                    routing_policy=attachment_config.routing_policy,
                    world_ledger_ref=attachment_config.world_ledger_ref,
                )
            ),
            preparation_materials=(FactoryAttachmentPreparationMaterialStore(private_store)),
            routing_facade=_FixtureAttachmentRoutingFacade(
                observed_at=(attachment_config.planning_prompt.audit.created_at),
            ),
            execution_facade=attachment_execution,
            requested_by=requested_by,
            runner=SupervisedAttachmentR5Runner(
                supervisor=ExecutionSupervisor(
                    store,
                    attachment_registry,
                    workspace_manager=AgentWorkspaceManager(attachment_workspace_path),
                ),
                review_service=reviews,
            ),
        )
    specialist_components = (
        build_fixture_specialist_components(
            config=specialist_config,
            gateway=gateway,
            private_store=private_store,
            factory_store=store,
            plan_reviews=reviews,
            job_store_path=job_store_path,
            workspace_path=specialist_workspace_path,
            requested_by=requested_by,
        )
        if (
            specialist_config is not None
            and job_store_path is not None
            and specialist_workspace_path is not None
        )
        else None
    )
    release_components = (
        build_fixture_release_components(
            config=release_config,
            job_store=(specialist_components.job_store_bridge.job_store),
            store=store,
            private_store=private_store,
            plan_reviews=reviews,
            output_root=candidate_output_path,
            requested_by=requested_by,
        )
        if (
            release_config is not None
            and candidate_output_path is not None
            and specialist_components is not None
        )
        else None
    )
    runtime = FactoryDatasetRuntime(
        store=store,
        planner=planner,
        compiler=compiler,
        plan_reviews=reviews,
        requested_by=requested_by,
        core_runner=core_runner,
        core_materials=(FactoryCoreMaterialStore(store) if core_runner is not None else None),
        task_authoring_bridge=task_bridge,
        task_authoring_materials=(
            FactoryTaskAuthoringMaterialStore(private_store) if task_bridge is not None else None
        ),
        attachment_runtime=attachment_runtime,
        item_specialist_runtime=(
            specialist_components.runtime if specialist_components is not None else None
        ),
        item_specialist_context_factory=(
            specialist_components.context_factory if specialist_components is not None else None
        ),
        job_store_bridge=(
            specialist_components.job_store_bridge if specialist_components is not None else None
        ),
        batch_quality_runtime=(release_components.batch_runtime if release_components is not None else None),
        batch_quality_context_factory=(
            release_components.batch_context_factory if release_components is not None else None
        ),
        release_context_factory=(
            release_components.release_context_factory if release_components is not None else None
        ),
        job_store_witness_bridge=(
            release_components.witness_bridge if release_components is not None else None
        ),
    )
    return FactoryDatasetFixtureComponents(
        runtime=runtime,
        private_store=private_store,
        gateway=gateway,
        specialist=specialist_components,
        release=release_components,
    )


def build_fixture_dataset_runtime(
    *,
    factory_store_path: Path,
    private_store_path: Path,
    gateway_store_path: Path,
    config: FactoryDatasetRuntimeConfigV1,
    fixture: FactoryDatasetPlanningFixtureV1,
    requested_by: str,
    core_config: FactoryDatasetCoreRuntimeConfigV1 | None = None,
    r4_config: FactoryDatasetR4RuntimeConfigV1 | None = None,
    core_workspace_path: Path | None = None,
    attachment_config: (FactoryDatasetAttachmentRuntimeConfigV1 | None) = None,
    attachment_workspace_path: Path | None = None,
    specialist_config: (FactoryDatasetSpecialistRuntimeConfigV1 | None) = None,
    job_store_path: Path | None = None,
    specialist_workspace_path: Path | None = None,
    release_config: (FactoryDatasetReleaseRuntimeConfigV1 | None) = None,
    candidate_output_path: Path | None = None,
) -> FactoryDatasetRuntime:
    return build_fixture_dataset_components(
        factory_store_path=factory_store_path,
        private_store_path=private_store_path,
        gateway_store_path=gateway_store_path,
        config=config,
        fixture=fixture,
        requested_by=requested_by,
        core_config=core_config,
        r4_config=r4_config,
        core_workspace_path=core_workspace_path,
        attachment_config=attachment_config,
        attachment_workspace_path=attachment_workspace_path,
        specialist_config=specialist_config,
        job_store_path=job_store_path,
        specialist_workspace_path=specialist_workspace_path,
        release_config=release_config,
        candidate_output_path=candidate_output_path,
    ).runtime


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _fixture_ref(
    object_type: str,
    seed: str,
) -> ObjectRef:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


__all__ = [
    "FactoryDatasetAttachmentRuntimeConfigV1",
    "FactoryDatasetCoreRuntimeConfigV1",
    "FactoryDatasetFixtureComponents",
    "FactoryDatasetPlanningFixtureV1",
    "FactoryDatasetR4RuntimeConfigV1",
    "FactoryDatasetRuntimeConfigV1",
    "FixturePlanningGatewayProvider",
    "build_fixture_dataset_components",
    "build_fixture_dataset_runtime",
]
