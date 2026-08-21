from __future__ import annotations

from datetime import timedelta
from typing import ClassVar, Protocol

from pydantic import BaseModel

from env_mock_agent.facade import (
    AttachmentCrossItemSafetyScanFacade,
    AttachmentDuplicateFingerprintFacade,
    AttachmentExecutionFacade,
)
from eval_factory.agent_system.attachment_preparation import (
    AttachmentR5PreparationAuthority,
)
from eval_factory.agent_system.attachment_quality import AttachmentQualityAgent
from eval_factory.agent_system.attachment_subgraph import (
    SupervisedAttachmentR5Runner,
)
from eval_factory.agent_system.batch_quality_runtime import (
    FactoryBatchQualityContext,
    FactoryBatchQualityRuntime,
)
from eval_factory.agent_system.candidate_output import (
    AuthorizedCandidateExport,
)
from eval_factory.agent_system.candidate_projection_material import (
    FactoryCandidateProjectionResult,
)
from eval_factory.agent_system.criteria_subgraph import (
    SupervisedCriteriaRubricRunner,
)
from eval_factory.agent_system.delivery_runtime import (
    FactoryDeliveryRuntime,
    FactoryDeliveryWaitingView,
)
from eval_factory.agent_system.grading_subgraph import (
    SupervisedGradingDesignRunner,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewNotResumableError,
    PlanReviewService,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridge,
)
from eval_factory.agent_system.trace_candidate import (
    TraceCandidatePreparationService,
)
from eval_factory.agent_system.trace_candidate_store import (
    FactoryTraceCandidateMaterialStore,
)
from eval_factory.batch_quality.reports import BatchQualityItemSource
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    AttachmentGenerationPlanV2,
    AttachmentGroupResultV2,
    AttachmentQualityAssessmentV2,
    AttachmentQualityOutcomeV2,
    AttachmentSubgraphOutcomeV2,
    AttachmentSubgraphResultV2,
    CompiledAttachmentGenerationPlanV2,
    CompiledCriteriaRubricPlanV2,
    CompiledDatasetBuildPlanV2,
    CompiledGradingDesignPlanV2,
    CriteriaRubricOutcomeV2,
    CriteriaRubricPlanV2,
    CriteriaRubricResultV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    ExtractedUserPromptV2,
    FactoryRunPolicyV2,
    GradingDesignOutcomeV2,
    GradingDesignPlanV2,
    GradingDesignResultV2,
    InferredUserIntentV2,
    PlanReviewResultV2,
    SolvabilityAssessmentV2,
    TaskRewriteCandidateV2,
    TraceCandidateDecisionV2,
)
from eval_factory.contracts.ai_gateway_v2 import ModelRouteDecisionV2
from eval_factory.contracts.attachment_v2 import ArtifactExecutionBatchV2
from eval_factory.contracts.batch_quality_v2 import BatchQualityPolicyV2
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.cross_item_safety_v2 import (
    CrossItemSafetyPolicyV2,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetDeliveryManifestV2,
    CandidateDatasetOutcomeV2,
    FactoryDatasetAggregateResultV2,
)
from eval_factory.contracts.duplicate_v2 import DuplicateDetectionPolicyV2
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.contracts.orchestration_v2 import ResolvedJobWorkGraphV2
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.task_v2 import (
    EvaluatorSpecV2,
    R4TaskContractSetV2,
    ReferencePolicyV2,
    RubricSetV2,
    TaskDraftV2,
    ToolPolicyV2,
    r4_task_contract_set_ref,
)
from eval_factory.contracts.trace import TraceEnvelope
from eval_factory.contracts.validation_v2 import (
    DeterministicItemValidationResultV2,
    deterministic_item_validation_result_ref,
)
from eval_factory.harness.capability import (
    CapabilityCallV1,
    CapabilityDefinitionV1,
    CapabilityInvocationOutcomeV1,
    CapabilityProviderBindingV1,
)
from eval_factory.harness.capability_runtime import (
    CapabilityProviderExecution,
)
from eval_factory.harness.composition import StaticPackRegistrationV1
from eval_factory.harness.contracts import sorted_refs
from eval_factory.orchestration.runner import TraceIndexStageService
from eval_factory.packs.generic_agent_trace.adapter_types import (
    CapabilityMaterialResolver,
    GenericCapabilityAdapterError,
)
from eval_factory.packs.generic_agent_trace.attachment_projection import (
    GenericAgentAttachmentProjection,
)
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentQualityCapabilityRequestV1,
    AttachmentQualityCapabilityRequestV2,
    AttachmentReconstructionCapabilityRequestV1,
    BatchQualityCapabilityRequestV1,
    CriteriaRubricCapabilityRequestV1,
    DeliveryCapabilityRequestV1,
    GradingDesignCapabilityRequestV1,
    PlanReviewCapabilityRequestV1,
    RequirementPlanningCapabilityRequestV1,
    TaskAuthoringCapabilityRequestV1,
    TraceIngestionCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.plan_review_preparation import (
    GenericAgentPlanReviewPreparation,
)
from eval_factory.packs.generic_agent_trace.quality_execution import (
    GenericAgentAttachmentQualityExecution,
)
from eval_factory.packs.generic_agent_trace.specialist_projections import (
    GenericAgentCriteriaProjection,
    GenericAgentGradingProjection,
)
from eval_factory.packs.generic_agent_trace.task_authoring_projection import (
    GenericAgentTaskAuthoringProjection,
)
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    ToolCapabilityCatalog,
    VerifiedModelDomainAuthorization,
)


class _BoundGenericCapabilityProvider:
    CAPABILITY_ID: ClassVar[str]
    REQUEST_MODEL: ClassVar[type[BaseModel]]
    RESULT_MODEL: ClassVar[type[BaseModel]]

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
    ) -> None:
        self.registration = registration
        matches = tuple(
            definition
            for definition in registration.capability_definitions
            if definition.capability_id == self.CAPABILITY_ID
        )
        if len(matches) != 1:
            raise GenericCapabilityAdapterError(
                "generic Pack capability definition is not uniquely installed",
            )
        self._definition = matches[0]
        providers = tuple(
            binding
            for binding in registration.provider_bindings
            if binding.capability_definition_ref == self._definition.to_ref()
        )
        if len(providers) != 1:
            raise GenericCapabilityAdapterError(
                "generic Pack capability provider is not uniquely installed",
            )
        self._binding = providers[0]

    @property
    def definition(self) -> CapabilityDefinitionV1:
        return self._definition

    @property
    def binding(self) -> CapabilityProviderBindingV1:
        return self._binding

    @property
    def provider_binding_ref(self) -> ObjectRef:
        return self._binding.to_ref()

    @property
    def implementation_id(self) -> str:
        return self._binding.implementation_id

    @property
    def request_model(self) -> type[BaseModel]:
        return self.REQUEST_MODEL

    @property
    def request_schema_ref(self) -> ObjectRef:
        return self._definition.request_schema_ref

    @property
    def result_model(self) -> type[BaseModel]:
        return self.RESULT_MODEL

    @property
    def result_schema_ref(self) -> ObjectRef:
        return self._definition.result_schema_ref


class GenericAgentBatchContextSource(Protocol):
    def batch_context(self) -> FactoryBatchQualityContext: ...

    def commit_item_witnesses(
        self,
        *,
        audit: ContractAudit,
    ) -> None: ...

    def commit_batch_witness(
        self,
        *,
        batch_quality_ref: ObjectRef,
        audit: ContractAudit,
    ) -> None: ...


class GenericAgentDeliveryRuntimeSource(Protocol):
    def delivery_runtime(self) -> FactoryDeliveryRuntime: ...


class RequirementPlanningCapabilityProvider(
    _BoundGenericCapabilityProvider,
):
    CAPABILITY_ID = "capability.requirement-planning"
    REQUEST_MODEL = RequirementPlanningCapabilityRequestV1
    RESULT_MODEL = CompiledDatasetBuildPlanV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        compiler: DatasetBuildPlanCompiler,
        materials: CapabilityMaterialResolver,
    ) -> None:
        super().__init__(registration)
        self.compiler = compiler
        self.materials = materials

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        del call
        typed = _typed_request(request, self.REQUEST_MODEL)
        result = self.compiler.compile(
            plan=self.materials.get(
                typed.plan_ref,
                DatasetBuildPlanV2,
            ),
            policy=self.materials.get(
                typed.policy_ref,
                FactoryRunPolicyV2,
            ),
            audit=audit,
        )
        return _success(result.to_ref())


class TraceIngestionCapabilityProvider(_BoundGenericCapabilityProvider):
    CAPABILITY_ID = "capability.trace-ingestion"
    REQUEST_MODEL = TraceIngestionCapabilityRequestV1
    RESULT_MODEL = TraceEnvelope

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        service: TraceIndexStageService,
        materials: CapabilityMaterialResolver,
        candidate_preparer: TraceCandidatePreparationService | None = None,
        candidate_materials: (FactoryTraceCandidateMaterialStore | None) = None,
    ) -> None:
        super().__init__(registration)
        self.service = service
        self.materials = materials
        self.candidate_preparer = candidate_preparer
        self.candidate_materials = candidate_materials
        if (candidate_preparer is None) != (candidate_materials is None):
            raise ValueError(
                "trace candidate preparer and material store must appear together",
            )

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        typed = _typed_request(request, self.REQUEST_MODEL)
        trace = self.materials.get(
            typed.trace_source_ref,
            TraceSourceRef,
        )
        result = self.service.execute(
            trace=trace,
            audit=audit,
            job_id=typed.job_id,
            attempt=typed.attempt,
        )
        validations: tuple[ObjectRef, ...] = ()
        if self.candidate_preparer is not None and self.candidate_materials is not None:
            prepared = await self.candidate_preparer.prepare(
                indexed=result.stored_index,
                source_ref=typed.trace_source_ref,
                audit=audit,
            )
            material = self.candidate_materials.commit(
                run_id=typed.job_id,
                trace_source_ref=typed.trace_source_ref,
                trace_envelope_ref=result.trace_envelope_ref,
                preparation=prepared,
                audit=audit,
                idempotency_key=(f"{call.context.idempotency_key}.candidate"),
            )
            validations = material.preparation.validation_refs()
        return _success(
            result.trace_envelope_ref,
            validation_refs=validations,
        )


class TaskAuthoringCapabilityProvider(_BoundGenericCapabilityProvider):
    CAPABILITY_ID = "capability.task-authoring"
    REQUEST_MODEL = TaskAuthoringCapabilityRequestV1
    RESULT_MODEL = R4TaskContractSetV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        bridge: FactoryTaskAuthoringBridge,
        materials: CapabilityMaterialResolver,
        result_projection: (GenericAgentTaskAuthoringProjection | None) = None,
    ) -> None:
        super().__init__(registration)
        self.bridge = bridge
        self.materials = materials
        self.result_projection = result_projection

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        typed = _typed_request(request, self.REQUEST_MODEL)
        result = await self.bridge.run(
            decision=self.materials.get(
                typed.decision_ref,
                TraceCandidateDecisionV2,
            ),
            extracted_prompt=self.materials.get(
                typed.extracted_prompt_ref,
                ExtractedUserPromptV2,
            ),
            intent=self.materials.get(
                typed.intent_ref,
                InferredUserIntentV2,
            ),
            rewrite=self.materials.get(
                typed.rewrite_ref,
                TaskRewriteCandidateV2,
            ),
            requirement=self.materials.get(
                typed.requirement_ref,
                EvaluationRequirementSpecV2,
            ),
            audit=audit,
        )
        validations: tuple[ObjectRef, ...] = (result.task_contract_set.task_prompt_safety_gate_ref,)
        if self.result_projection is not None:
            validations = sorted_refs(
                (
                    *validations,
                    *self.result_projection.commit(
                        request=typed,
                        result=result,
                        audit=audit,
                        idempotency_key=(f"{call.context.idempotency_key}.factory"),
                    ),
                )
            )
        return _success(
            r4_task_contract_set_ref(result.task_contract_set),
            validation_refs=validations,
        )


class AttachmentReconstructionCapabilityProvider(
    _BoundGenericCapabilityProvider,
):
    CAPABILITY_ID = "capability.attachment-reconstruction"
    REQUEST_MODEL = AttachmentReconstructionCapabilityRequestV1
    RESULT_MODEL = AttachmentSubgraphResultV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        runner: SupervisedAttachmentR5Runner,
        materials: CapabilityMaterialResolver,
        facade: AttachmentExecutionFacade,
        result_projection: GenericAgentAttachmentProjection | None = None,
    ) -> None:
        super().__init__(registration)
        self.runner = runner
        self.materials = materials
        self.facade = facade
        self.result_projection = result_projection

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        typed = _typed_request(request, self.REQUEST_MODEL)
        execution = await self.runner.run(
            run_id=typed.run_id,
            plan=self.materials.get(
                typed.plan_ref,
                AttachmentGenerationPlanV2,
            ),
            compiled_plan=self.materials.get(
                typed.compiled_plan_ref,
                CompiledAttachmentGenerationPlanV2,
            ),
            preparation=self.materials.get(
                typed.preparation_ref,
                AttachmentR5PreparationAuthority,
            ),
            facade=self.facade,
            audit=audit,
            lease_duration=timedelta(
                seconds=typed.lease_duration_seconds,
            ),
            prior_batch=(
                self.materials.get(
                    typed.prior_batch_ref,
                    ArtifactExecutionBatchV2,
                )
                if typed.prior_batch_ref is not None
                else None
            ),
        )
        result = execution.subgraph_result
        if result.outcome is AttachmentSubgraphOutcomeV2.SUCCEEDED:
            validations = result.work_result_refs
            if self.result_projection is not None:
                validations = sorted_refs(
                    (
                        *validations,
                        *self.result_projection.commit(
                            request=typed,
                            execution=execution,
                            audit=audit,
                            idempotency_key=(f"{call.context.idempotency_key}.factory"),
                        ),
                    )
                )
            return _success(
                result.to_ref(),
                validation_refs=validations,
            )
        outcome = (
            CapabilityInvocationOutcomeV1.BLOCKED_POLICY
            if result.outcome is AttachmentSubgraphOutcomeV2.BLOCKED
            else CapabilityInvocationOutcomeV1.FAILED
        )
        return _non_success(
            outcome,
            _reason_code(result.reason_codes, result.outcome.value),
            validation_refs=result.work_result_refs,
        )


class CriteriaRubricCapabilityProvider(_BoundGenericCapabilityProvider):
    CAPABILITY_ID = "capability.criteria-rubric"
    REQUEST_MODEL = CriteriaRubricCapabilityRequestV1
    RESULT_MODEL = CriteriaRubricResultV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        runner: SupervisedCriteriaRubricRunner,
        materials: CapabilityMaterialResolver,
        result_projection: GenericAgentCriteriaProjection | None = None,
    ) -> None:
        super().__init__(registration)
        self.runner = runner
        self.materials = materials
        self.result_projection = result_projection

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        typed = _typed_request(request, self.REQUEST_MODEL)
        execution = await self.runner.run(
            run_id=typed.run_id,
            plan=self.materials.get(
                typed.plan_ref,
                CriteriaRubricPlanV2,
            ),
            compiled_plan=self.materials.get(
                typed.compiled_plan_ref,
                CompiledCriteriaRubricPlanV2,
            ),
            task_draft=self.materials.get(
                typed.task_draft_ref,
                TaskDraftV2,
            ),
            attachment_quality=self.materials.get(
                typed.attachment_quality_ref,
                AttachmentQualityAssessmentV2,
            ),
            solvability=self.materials.get(
                typed.solvability_ref,
                SolvabilityAssessmentV2,
            ),
            binding_definitions=self.materials.get_many(
                typed.binding_definitions_ref,
                EvaluatorBindingDefinition,
            ),
            tool_catalog=self.materials.get(
                typed.tool_catalog_ref,
                ToolCapabilityCatalog,
            ),
            audit=audit,
            lease_duration=timedelta(
                seconds=typed.lease_duration_seconds,
            ),
        )
        result = execution.result
        validations: tuple[ObjectRef, ...] = (execution.envelope.to_ref(),)
        if self.result_projection is not None:
            validations = sorted_refs(
                (
                    *validations,
                    *self.result_projection.commit(
                        request=typed,
                        execution=execution,
                        audit=audit,
                    ),
                )
            )
        if result.outcome is CriteriaRubricOutcomeV2.SUCCEEDED:
            return _success(
                result.to_ref(),
                validation_refs=validations,
            )
        if result.outcome is CriteriaRubricOutcomeV2.ABSTAINED:
            outcome = CapabilityInvocationOutcomeV1.ABSTAINED
        elif result.outcome is CriteriaRubricOutcomeV2.BLOCKED_CAPABILITY:
            outcome = CapabilityInvocationOutcomeV1.BLOCKED_CAPABILITY
        else:
            outcome = CapabilityInvocationOutcomeV1.BLOCKED_POLICY
        return _non_success(
            outcome,
            _reason_code(result.reason_codes, result.outcome.value),
            validation_refs=validations,
        )


class GradingDesignCapabilityProvider(_BoundGenericCapabilityProvider):
    CAPABILITY_ID = "capability.grading-design"
    REQUEST_MODEL = GradingDesignCapabilityRequestV1
    RESULT_MODEL = GradingDesignResultV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        runner: SupervisedGradingDesignRunner,
        materials: CapabilityMaterialResolver,
        result_projection: GenericAgentGradingProjection | None = None,
    ) -> None:
        super().__init__(registration)
        self.runner = runner
        self.materials = materials
        self.result_projection = result_projection

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        typed = _typed_request(request, self.REQUEST_MODEL)
        execution = await self.runner.run(
            run_id=typed.run_id,
            plan=self.materials.get(
                typed.plan_ref,
                GradingDesignPlanV2,
            ),
            compiled_plan=self.materials.get(
                typed.compiled_plan_ref,
                CompiledGradingDesignPlanV2,
            ),
            criteria_result=self.materials.get(
                typed.criteria_result_ref,
                CriteriaRubricResultV2,
            ),
            criteria_route=self.materials.get(
                typed.criteria_route_ref,
                ModelRouteDecisionV2,
            ),
            rubric_set=self.materials.get(
                typed.rubric_set_ref,
                RubricSetV2,
            ),
            evaluator_spec=self.materials.get(
                typed.evaluator_spec_ref,
                EvaluatorSpecV2,
            ),
            reference_policy=self.materials.get(
                typed.reference_policy_ref,
                ReferencePolicyV2,
            ),
            tool_policy=self.materials.get(
                typed.tool_policy_ref,
                ToolPolicyV2,
            ),
            model_authorizations=self.materials.get_many(
                typed.model_authorizations_ref,
                VerifiedModelDomainAuthorization,
            ),
            evaluated_at=typed.evaluated_at,
            audit=audit,
            lease_duration=timedelta(
                seconds=typed.lease_duration_seconds,
            ),
        )
        result = execution.result
        validations = sorted_refs(
            (execution.envelope.to_ref(), result.validation_ref),
        )
        if self.result_projection is not None:
            validations = sorted_refs(
                (
                    *validations,
                    *self.result_projection.commit(
                        request=typed,
                        execution=execution,
                        audit=audit,
                    ),
                )
            )
        if result.outcome is GradingDesignOutcomeV2.SUCCEEDED:
            return _success(
                result.to_ref(),
                validation_refs=validations,
            )
        if result.outcome in {
            GradingDesignOutcomeV2.ABSTAINED,
            GradingDesignOutcomeV2.ESCALATED,
        }:
            outcome = CapabilityInvocationOutcomeV1.ABSTAINED
        elif result.outcome is GradingDesignOutcomeV2.BLOCKED_CAPABILITY:
            outcome = CapabilityInvocationOutcomeV1.BLOCKED_CAPABILITY
        elif result.outcome is GradingDesignOutcomeV2.INVALID_DESIGN:
            outcome = CapabilityInvocationOutcomeV1.FAILED
        else:
            outcome = CapabilityInvocationOutcomeV1.BLOCKED_POLICY
        return _non_success(
            outcome,
            _reason_code(result.reason_codes, result.outcome.value),
            validation_refs=validations,
        )


class AttachmentQualityCapabilityProvider(_BoundGenericCapabilityProvider):
    CAPABILITY_ID = "capability.quality-review"
    REQUEST_MODEL = AttachmentQualityCapabilityRequestV1
    RESULT_MODEL = AttachmentQualityAssessmentV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        agent: AttachmentQualityAgent,
        materials: CapabilityMaterialResolver,
    ) -> None:
        super().__init__(registration)
        self.agent = agent
        self.materials = materials

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        del call
        typed = _typed_request(request, self.REQUEST_MODEL)
        result = self.agent.assess(
            subgraph_result=self.materials.get(
                typed.subgraph_result_ref,
                AttachmentSubgraphResultV2,
            ),
            group_results=tuple(
                self.materials.get(
                    reference,
                    AttachmentGroupResultV2,
                )
                for reference in typed.group_result_refs
            ),
            work_envelopes=tuple(
                self.materials.get(
                    reference,
                    AgentResultEnvelopeV2,
                )
                for reference in typed.work_envelope_refs
            ),
            source_validation=self.materials.get(
                typed.source_validation_ref,
                DeterministicItemValidationResultV2,
            ),
            item_quality=self.materials.get(
                typed.item_quality_ref,
                ItemQualityCompilationResultV2,
            ),
            audit=audit,
        )
        if result.outcome is AttachmentQualityOutcomeV2.PASSED:
            return _success(
                result.to_ref(),
                validation_refs=result.validator_result_refs,
            )
        outcome = (
            CapabilityInvocationOutcomeV1.FAILED
            if result.outcome is AttachmentQualityOutcomeV2.REPAIR_REQUIRED
            else CapabilityInvocationOutcomeV1.BLOCKED_POLICY
        )
        return _non_success(
            outcome,
            _reason_code(result.reason_codes, result.outcome.value),
            validation_refs=result.validator_result_refs,
        )


class AttachmentQualityWorkflowCapabilityProvider(
    _BoundGenericCapabilityProvider,
):
    CAPABILITY_ID = "capability.quality-review"
    REQUEST_MODEL = AttachmentQualityCapabilityRequestV2
    RESULT_MODEL = AttachmentQualityAssessmentV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        execution: GenericAgentAttachmentQualityExecution,
    ) -> None:
        super().__init__(registration)
        self.execution = execution

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        del call
        typed = _typed_request(request, self.REQUEST_MODEL)
        view = await self.execution.execute(
            typed,
            audit=audit,
        )
        result = view.result
        quality = result.finalization.attachment_quality
        validations = sorted_refs(
            (
                view.item_quality_head_ref,
                view.material_ref,
                item_quality_compilation_result_ref(
                    result.finalization.item_quality,
                ),
                result.solvability.assessment.to_ref(),
                deterministic_item_validation_result_ref(
                    result.finalization.source_validation,
                ),
            )
        )
        if quality.outcome is AttachmentQualityOutcomeV2.PASSED:
            return _success(
                quality.to_ref(),
                validation_refs=validations,
            )
        outcome = (
            CapabilityInvocationOutcomeV1.FAILED
            if quality.outcome is AttachmentQualityOutcomeV2.REPAIR_REQUIRED
            else CapabilityInvocationOutcomeV1.BLOCKED_POLICY
        )
        return _non_success(
            outcome,
            _reason_code(quality.reason_codes, quality.outcome.value),
            validation_refs=validations,
        )


class BatchQualityCapabilityProvider(_BoundGenericCapabilityProvider):
    CAPABILITY_ID = "capability.batch-quality"
    REQUEST_MODEL = BatchQualityCapabilityRequestV1
    RESULT_MODEL = FactoryDatasetAggregateResultV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        runtime: FactoryBatchQualityRuntime,
        materials: CapabilityMaterialResolver,
        duplicate_facade: (AttachmentDuplicateFingerprintFacade | None) = None,
        cross_item_facade: (AttachmentCrossItemSafetyScanFacade | None) = None,
        context_source: GenericAgentBatchContextSource | None = None,
    ) -> None:
        super().__init__(registration)
        self.runtime = runtime
        self.duplicate_facade = duplicate_facade
        self.cross_item_facade = cross_item_facade
        self.materials = materials
        self.context_source = context_source
        explicit_facades = duplicate_facade is not None and cross_item_facade is not None
        if explicit_facades == (context_source is not None):
            raise ValueError(
                "Batch provider requires one context authority source",
            )

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        del call
        typed = _typed_request(request, self.REQUEST_MODEL)
        if self.context_source is not None:
            self.context_source.commit_item_witnesses(
                audit=audit,
            )
        context = (
            self.context_source.batch_context()
            if self.context_source is not None
            else self._explicit_context(typed)
        )
        if (
            self.materials.get(
                typed.resolved_job_work_graph_ref,
                ResolvedJobWorkGraphV2,
            )
            != context.resolved_job_work_graph
            or self.materials.get(
                typed.duplicate_policy_ref,
                DuplicateDetectionPolicyV2,
            )
            != context.duplicate_policy
            or self.materials.get(
                typed.cross_item_policy_ref,
                CrossItemSafetyPolicyV2,
            )
            != context.cross_item_policy
            or self.materials.get(
                typed.batch_policy_ref,
                BatchQualityPolicyV2,
            )
            != context.batch_policy
        ):
            raise GenericCapabilityAdapterError(
                "Batch request differs from current owner context",
            )
        view = await self.runtime.advance(
            dataset_run_id=typed.dataset_run_id,
            core_vertical_result_ref=typed.core_vertical_result_ref,
            sources=self.materials.get_many(
                typed.sources_ref,
                BatchQualityItemSource,
            ),
            rejected_binding_refs=typed.rejected_binding_refs,
            blocked_binding_refs=typed.blocked_binding_refs,
            context=context,
            audit=audit,
        )
        aggregate = view.aggregate
        if self.context_source is not None and aggregate.batch_quality_ref is not None:
            self.context_source.commit_batch_witness(
                batch_quality_ref=aggregate.batch_quality_ref,
                audit=audit,
            )
        validations = (aggregate.batch_quality_ref,) if aggregate.batch_quality_ref is not None else ()
        if aggregate.outcome in {
            CandidateDatasetOutcomeV2.COMPLETE,
            CandidateDatasetOutcomeV2.PARTIAL,
        }:
            return _success(
                aggregate.to_ref(),
                validation_refs=validations,
            )
        outcome = (
            CapabilityInvocationOutcomeV1.ABSTAINED
            if aggregate.outcome is CandidateDatasetOutcomeV2.NO_ELIGIBLE_ITEMS
            else CapabilityInvocationOutcomeV1.BLOCKED_POLICY
        )
        return _non_success(
            outcome,
            _reason_code(
                aggregate.reason_codes,
                aggregate.outcome.value,
            ),
            validation_refs=validations,
        )

    def _explicit_context(
        self,
        request: BatchQualityCapabilityRequestV1,
    ) -> FactoryBatchQualityContext:
        duplicate_facade = self.duplicate_facade
        cross_item_facade = self.cross_item_facade
        assert duplicate_facade is not None
        assert cross_item_facade is not None
        return FactoryBatchQualityContext(
            resolved_job_work_graph=self.materials.get(
                request.resolved_job_work_graph_ref,
                ResolvedJobWorkGraphV2,
            ),
            duplicate_policy=self.materials.get(
                request.duplicate_policy_ref,
                DuplicateDetectionPolicyV2,
            ),
            duplicate_facade=duplicate_facade,
            cross_item_policy=self.materials.get(
                request.cross_item_policy_ref,
                CrossItemSafetyPolicyV2,
            ),
            cross_item_facade=cross_item_facade,
            batch_policy=self.materials.get(
                request.batch_policy_ref,
                BatchQualityPolicyV2,
            ),
        )


class PlanReviewCapabilityProvider(_BoundGenericCapabilityProvider):
    CAPABILITY_ID = "capability.plan-review"
    REQUEST_MODEL = PlanReviewCapabilityRequestV1
    RESULT_MODEL = PlanReviewResultV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        service: PlanReviewService,
        review_preparation: (GenericAgentPlanReviewPreparation | None) = None,
    ) -> None:
        super().__init__(registration)
        self.service = service
        self.review_preparation = review_preparation

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        del call
        typed = _typed_request(request, self.REQUEST_MODEL)
        prepared = (
            await self.review_preparation.prepare(
                typed,
                audit=audit,
            )
            if self.review_preparation is not None
            else None
        )
        try:
            view = self.service.require_resumed_review(
                run_id=(prepared.run_id if prepared is not None else typed.run_id),
                plan_kind=(prepared.plan_kind if prepared is not None else typed.plan_kind),
                plan_ref=(prepared.plan_ref if prepared is not None else typed.plan_ref),
            )
        except PlanReviewNotResumableError:
            return _non_success(
                CapabilityInvocationOutcomeV1.BLOCKED_POLICY,
                "USER_APPROVAL_REQUIRED",
            )
        validations = (view.result.decision_ref,) if view.result.decision_ref is not None else ()
        return _success(
            view.result.to_ref(),
            validation_refs=validations,
        )


class DeliveryCapabilityProvider(_BoundGenericCapabilityProvider):
    CAPABILITY_ID = "capability.delivery"
    REQUEST_MODEL = DeliveryCapabilityRequestV1
    RESULT_MODEL = CandidateDatasetDeliveryManifestV2

    def __init__(
        self,
        registration: StaticPackRegistrationV1,
        *,
        materials: CapabilityMaterialResolver,
        runtime: FactoryDeliveryRuntime | None = None,
        runtime_source: (GenericAgentDeliveryRuntimeSource | None) = None,
    ) -> None:
        super().__init__(registration)
        self.runtime = runtime
        self.materials = materials
        self.runtime_source = runtime_source
        if (runtime is None) == (runtime_source is None):
            raise ValueError(
                "Delivery provider requires one runtime authority source",
            )

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        del call
        typed = _typed_request(request, self.REQUEST_MODEL)
        candidates = tuple(
            self.materials.get(
                reference,
                FactoryCandidateProjectionResult,
            )
            for reference in typed.candidate_projection_refs
        )
        exports = tuple(
            self.materials.get(
                reference,
                AuthorizedCandidateExport,
            )
            for reference in typed.export_refs
        )
        if tuple(value.projection.to_ref() for value in candidates) != typed.candidate_projection_refs:
            raise GenericCapabilityAdapterError(
                "delivery candidate material differs from request authority",
            )
        runtime = self.runtime if self.runtime is not None else self._runtime_from_source()
        result = runtime.advance(
            dataset_run_id=typed.dataset_run_id,
            aggregate=self.materials.get(
                typed.aggregate_ref,
                FactoryDatasetAggregateResultV2,
            ),
            candidates=candidates,
            exports=exports,
            policy=self.materials.get(
                typed.policy_ref,
                FactoryRunPolicyV2,
            ),
            audit=audit,
        )
        if isinstance(result, FactoryDeliveryWaitingView):
            return _non_success(
                CapabilityInvocationOutcomeV1.BLOCKED_POLICY,
                "DELIVERY_PLAN_REVIEW_PENDING",
                validation_refs=(result.review_ref,),
            )
        return _success(
            result.manifest.to_ref(),
            validation_refs=(
                result.completion.to_ref(),
                result.inventory.to_ref(),
            ),
        )

    def _runtime_from_source(self) -> FactoryDeliveryRuntime:
        assert self.runtime_source is not None
        return self.runtime_source.delivery_runtime()


def generic_agent_trace_providers(
    *providers: _BoundGenericCapabilityProvider,
) -> tuple[_BoundGenericCapabilityProvider, ...]:
    by_capability = {value.CAPABILITY_ID: value for value in providers}
    required = {
        "capability.requirement-planning",
        "capability.trace-ingestion",
        "capability.task-authoring",
        "capability.attachment-reconstruction",
        "capability.criteria-rubric",
        "capability.grading-design",
        "capability.quality-review",
        "capability.batch-quality",
        "capability.plan-review",
        "capability.delivery",
    }
    if set(by_capability) != required or len(providers) != len(required):
        raise GenericCapabilityAdapterError(
            "generic Agent Trace provider set must contain all ten capabilities",
        )
    return tuple(by_capability[capability_id] for capability_id in sorted(by_capability))


def _typed_request[RequestT: BaseModel](
    request: BaseModel,
    request_model: type[RequestT],
) -> RequestT:
    if type(request) is not request_model:
        raise GenericCapabilityAdapterError(
            "capability adapter received another request model",
        )
    return request


def _success(
    reference: ObjectRef,
    *,
    validation_refs: tuple[ObjectRef, ...] = (),
) -> CapabilityProviderExecution:
    return CapabilityProviderExecution(
        outcome=CapabilityInvocationOutcomeV1.SUCCEEDED,
        canonical_result_ref=reference,
        content_ref=reference,
        validation_refs=sorted_refs(validation_refs),
        domain_tags=("generic-agent-trace",),
    )


def _non_success(
    outcome: CapabilityInvocationOutcomeV1,
    failure_code: str,
    *,
    validation_refs: tuple[ObjectRef, ...] = (),
) -> CapabilityProviderExecution:
    return CapabilityProviderExecution(
        outcome=outcome,
        validation_refs=sorted_refs(validation_refs),
        failure_code=failure_code,
        domain_tags=("generic-agent-trace",),
    )


def _reason_code(
    reason_codes: tuple[str, ...],
    fallback: str,
) -> str:
    return reason_codes[0] if reason_codes else fallback


__all__ = [
    "AttachmentQualityCapabilityProvider",
    "AttachmentQualityWorkflowCapabilityProvider",
    "AttachmentReconstructionCapabilityProvider",
    "BatchQualityCapabilityProvider",
    "CapabilityMaterialResolver",
    "CriteriaRubricCapabilityProvider",
    "DeliveryCapabilityProvider",
    "GenericAgentBatchContextSource",
    "GenericAgentDeliveryRuntimeSource",
    "GenericCapabilityAdapterError",
    "GradingDesignCapabilityProvider",
    "PlanReviewCapabilityProvider",
    "RequirementPlanningCapabilityProvider",
    "TaskAuthoringCapabilityProvider",
    "TraceIngestionCapabilityProvider",
    "generic_agent_trace_providers",
]
