from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from eval_factory.agent_system.attachment_preparation import (
    AttachmentR5PreparationAuthority,
)
from eval_factory.agent_system.candidate_output import (
    AuthorizedCandidateExport,
)
from eval_factory.agent_system.candidate_projection_material import (
    FactoryCandidateProjectionResult,
)
from eval_factory.batch_quality.reports import BatchQualityItemSource
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    AttachmentGenerationPlanV2,
    AttachmentGroupResultV2,
    AttachmentQualityAssessmentV2,
    AttachmentSubgraphResultV2,
    CompiledAttachmentGenerationPlanV2,
    CompiledCriteriaRubricPlanV2,
    CompiledGradingDesignPlanV2,
    CriteriaRubricPlanV2,
    CriteriaRubricResultV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    ExtractedUserPromptV2,
    FactoryRunPolicyV2,
    GradingDesignPlanV2,
    InferredUserIntentV2,
    PlanKindV2,
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
    FactoryDatasetAggregateResultV2,
)
from eval_factory.contracts.duplicate_v2 import DuplicateDetectionPolicyV2
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.contracts.orchestration_v2 import ResolvedJobWorkGraphV2
from eval_factory.contracts.quality_v2 import ItemQualityCompilationResultV2
from eval_factory.contracts.task_v2 import (
    EvaluatorSpecV2,
    ReferencePolicyV2,
    RubricSetV2,
    TaskDraftV2,
    ToolPolicyV2,
)
from eval_factory.contracts.validation_v2 import (
    DeterministicItemValidationResultV2,
)
from eval_factory.packs.generic_agent_trace.adapter_types import (
    GenericCapabilityAdapterError,
)
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentQualityCapabilityRequestV1,
    AttachmentQualityCapabilityRequestV2,
    AttachmentReconstructionCapabilityRequestV1,
    BatchQualityCapabilityRequestV1,
    CapabilityRequestV1,
    CriteriaRubricCapabilityRequestV1,
    DeliveryCapabilityRequestV1,
    GradingDesignCapabilityRequestV1,
    PlanReviewCapabilityRequestV1,
    RequirementPlanningCapabilityRequestV1,
    TaskAuthoringCapabilityRequestV1,
    TraceIngestionCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.material_resolver import (
    CompositeCapabilityMaterialResolver,
)
from eval_factory.packs.generic_agent_trace.request_preparation import (
    GenericAgentTaskRequestPreparer,
)
from eval_factory.packs.generic_agent_trace.request_store import (
    GenericAgentCapabilityRequestBindingV1,
)
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    ToolCapabilityCatalog,
    VerifiedModelDomainAuthorization,
)


@dataclass(frozen=True, slots=True)
class CapabilityOwnerMaterial[ValueT]:
    reference: ObjectRef
    value: ValueT


@dataclass(frozen=True, slots=True)
class CapabilityOwnerCollection[ValueT]:
    reference: ObjectRef
    values: tuple[ValueT, ...]


class GenericAgentCapabilityPreparationService:
    """Builds and commits Pack requests from canonical owner material."""

    def __init__(
        self,
        *,
        preparer: GenericAgentTaskRequestPreparer,
        materials: CompositeCapabilityMaterialResolver,
    ) -> None:
        self.preparer = preparer
        self.materials = materials

    def prepare_requirement_planning(
        self,
        task_id: str,
        *,
        plan: CapabilityOwnerMaterial[DatasetBuildPlanV2],
        policy: CapabilityOwnerMaterial[FactoryRunPolicyV2],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = RequirementPlanningCapabilityRequestV1.create(
            plan_ref=plan.reference,
            policy_ref=policy.reference,
            audit=audit,
        )
        return self._commit(
            task_id,
            request,
            values=((plan, DatasetBuildPlanV2), (policy, FactoryRunPolicyV2)),
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_trace_ingestion(
        self,
        task_id: str,
        *,
        trace_source: CapabilityOwnerMaterial[TraceSourceRef],
        job_id: str,
        attempt: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = TraceIngestionCapabilityRequestV1.create(
            trace_source_ref=trace_source.reference,
            job_id=job_id,
            attempt=attempt,
            audit=audit,
        )
        return self._commit(
            task_id,
            request,
            values=((trace_source, TraceSourceRef),),
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_task_authoring(
        self,
        task_id: str,
        *,
        decision: CapabilityOwnerMaterial[TraceCandidateDecisionV2],
        extracted_prompt: CapabilityOwnerMaterial[ExtractedUserPromptV2],
        intent: CapabilityOwnerMaterial[InferredUserIntentV2],
        rewrite: CapabilityOwnerMaterial[TaskRewriteCandidateV2],
        requirement: CapabilityOwnerMaterial[EvaluationRequirementSpecV2],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = TaskAuthoringCapabilityRequestV1.create(
            decision_ref=decision.reference,
            extracted_prompt_ref=extracted_prompt.reference,
            intent_ref=intent.reference,
            rewrite_ref=rewrite.reference,
            requirement_ref=requirement.reference,
            audit=audit,
        )
        return self._commit(
            task_id,
            request,
            values=(
                (decision, TraceCandidateDecisionV2),
                (extracted_prompt, ExtractedUserPromptV2),
                (intent, InferredUserIntentV2),
                (rewrite, TaskRewriteCandidateV2),
                (requirement, EvaluationRequirementSpecV2),
            ),
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_attachment_reconstruction(
        self,
        task_id: str,
        *,
        run_id: str,
        plan: CapabilityOwnerMaterial[AttachmentGenerationPlanV2],
        compiled_plan: CapabilityOwnerMaterial[CompiledAttachmentGenerationPlanV2],
        preparation: CapabilityOwnerMaterial[AttachmentR5PreparationAuthority],
        prior_batch: CapabilityOwnerMaterial[ArtifactExecutionBatchV2] | None,
        lease_duration_seconds: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = AttachmentReconstructionCapabilityRequestV1.create(
            run_id=run_id,
            plan_ref=plan.reference,
            compiled_plan_ref=compiled_plan.reference,
            preparation_ref=preparation.reference,
            prior_batch_ref=(prior_batch.reference if prior_batch is not None else None),
            lease_duration_seconds=lease_duration_seconds,
            audit=audit,
        )
        values: tuple[tuple[CapabilityOwnerMaterial[Any], type[Any]], ...] = (
            (plan, AttachmentGenerationPlanV2),
            (compiled_plan, CompiledAttachmentGenerationPlanV2),
            (preparation, AttachmentR5PreparationAuthority),
        )
        if prior_batch is not None:
            values += ((prior_batch, ArtifactExecutionBatchV2),)
        return self._commit(
            task_id,
            request,
            values=values,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_criteria_rubric(
        self,
        task_id: str,
        *,
        run_id: str,
        plan: CapabilityOwnerMaterial[CriteriaRubricPlanV2],
        compiled_plan: CapabilityOwnerMaterial[CompiledCriteriaRubricPlanV2],
        task_draft: CapabilityOwnerMaterial[TaskDraftV2],
        attachment_quality: CapabilityOwnerMaterial[AttachmentQualityAssessmentV2],
        solvability: CapabilityOwnerMaterial[SolvabilityAssessmentV2],
        binding_definitions: CapabilityOwnerCollection[EvaluatorBindingDefinition],
        tool_catalog: CapabilityOwnerMaterial[ToolCapabilityCatalog],
        lease_duration_seconds: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = CriteriaRubricCapabilityRequestV1.create(
            run_id=run_id,
            plan_ref=plan.reference,
            compiled_plan_ref=compiled_plan.reference,
            task_draft_ref=task_draft.reference,
            attachment_quality_ref=attachment_quality.reference,
            solvability_ref=solvability.reference,
            binding_definitions_ref=binding_definitions.reference,
            tool_catalog_ref=tool_catalog.reference,
            lease_duration_seconds=lease_duration_seconds,
            audit=audit,
        )
        return self._commit(
            task_id,
            request,
            values=(
                (plan, CriteriaRubricPlanV2),
                (compiled_plan, CompiledCriteriaRubricPlanV2),
                (task_draft, TaskDraftV2),
                (attachment_quality, AttachmentQualityAssessmentV2),
                (solvability, SolvabilityAssessmentV2),
                (tool_catalog, ToolCapabilityCatalog),
            ),
            collections=((binding_definitions, EvaluatorBindingDefinition),),
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_grading_design(
        self,
        task_id: str,
        *,
        run_id: str,
        plan: CapabilityOwnerMaterial[GradingDesignPlanV2],
        compiled_plan: CapabilityOwnerMaterial[CompiledGradingDesignPlanV2],
        criteria_result: CapabilityOwnerMaterial[CriteriaRubricResultV2],
        criteria_route: CapabilityOwnerMaterial[ModelRouteDecisionV2],
        rubric_set: CapabilityOwnerMaterial[RubricSetV2],
        evaluator_spec: CapabilityOwnerMaterial[EvaluatorSpecV2],
        reference_policy: CapabilityOwnerMaterial[ReferencePolicyV2],
        tool_policy: CapabilityOwnerMaterial[ToolPolicyV2],
        model_authorizations: CapabilityOwnerCollection[VerifiedModelDomainAuthorization],
        evaluated_at: datetime,
        lease_duration_seconds: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = GradingDesignCapabilityRequestV1.create(
            run_id=run_id,
            plan_ref=plan.reference,
            compiled_plan_ref=compiled_plan.reference,
            criteria_result_ref=criteria_result.reference,
            criteria_route_ref=criteria_route.reference,
            rubric_set_ref=rubric_set.reference,
            evaluator_spec_ref=evaluator_spec.reference,
            reference_policy_ref=reference_policy.reference,
            tool_policy_ref=tool_policy.reference,
            model_authorizations_ref=model_authorizations.reference,
            evaluated_at=evaluated_at,
            lease_duration_seconds=lease_duration_seconds,
            audit=audit,
        )
        return self._commit(
            task_id,
            request,
            values=(
                (plan, GradingDesignPlanV2),
                (compiled_plan, CompiledGradingDesignPlanV2),
                (criteria_result, CriteriaRubricResultV2),
                (criteria_route, ModelRouteDecisionV2),
                (rubric_set, RubricSetV2),
                (evaluator_spec, EvaluatorSpecV2),
                (reference_policy, ReferencePolicyV2),
                (tool_policy, ToolPolicyV2),
            ),
            collections=((model_authorizations, VerifiedModelDomainAuthorization),),
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_attachment_quality(
        self,
        task_id: str,
        *,
        subgraph_result: CapabilityOwnerMaterial[AttachmentSubgraphResultV2],
        group_results: tuple[CapabilityOwnerMaterial[AttachmentGroupResultV2], ...],
        work_envelopes: tuple[CapabilityOwnerMaterial[AgentResultEnvelopeV2], ...],
        source_validation: CapabilityOwnerMaterial[DeterministicItemValidationResultV2],
        item_quality: CapabilityOwnerMaterial[ItemQualityCompilationResultV2],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = AttachmentQualityCapabilityRequestV1.create(
            subgraph_result_ref=subgraph_result.reference,
            group_result_refs=tuple(value.reference for value in group_results),
            work_envelope_refs=tuple(value.reference for value in work_envelopes),
            source_validation_ref=source_validation.reference,
            item_quality_ref=item_quality.reference,
            audit=audit,
        )
        values: tuple[tuple[CapabilityOwnerMaterial[Any], type[Any]], ...] = (
            (subgraph_result, AttachmentSubgraphResultV2),
            *((value, AttachmentGroupResultV2) for value in group_results),
            *((value, AgentResultEnvelopeV2) for value in work_envelopes),
            (source_validation, DeterministicItemValidationResultV2),
            (item_quality, ItemQualityCompilationResultV2),
        )
        return self._commit(
            task_id,
            request,
            values=values,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_attachment_quality_workflow(
        self,
        task_id: str,
        *,
        item_binding_ref: ObjectRef,
        task_authoring_material_ref: ObjectRef,
        attachment_execution_material_ref: ObjectRef,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = AttachmentQualityCapabilityRequestV2.create(
            item_binding_ref=item_binding_ref,
            task_authoring_material_ref=task_authoring_material_ref,
            attachment_execution_material_ref=(attachment_execution_material_ref),
            audit=audit,
        )
        return self._commit(
            task_id,
            request,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_batch_quality(
        self,
        task_id: str,
        *,
        dataset_run_id: str,
        core_vertical_result_ref: ObjectRef,
        sources: CapabilityOwnerCollection[BatchQualityItemSource],
        rejected_binding_refs: tuple[ObjectRef, ...],
        blocked_binding_refs: tuple[ObjectRef, ...],
        resolved_job_work_graph: CapabilityOwnerMaterial[ResolvedJobWorkGraphV2],
        duplicate_policy: CapabilityOwnerMaterial[DuplicateDetectionPolicyV2],
        cross_item_policy: CapabilityOwnerMaterial[CrossItemSafetyPolicyV2],
        batch_policy: CapabilityOwnerMaterial[BatchQualityPolicyV2],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = BatchQualityCapabilityRequestV1.create(
            dataset_run_id=dataset_run_id,
            core_vertical_result_ref=core_vertical_result_ref,
            sources_ref=sources.reference,
            rejected_binding_refs=rejected_binding_refs,
            blocked_binding_refs=blocked_binding_refs,
            resolved_job_work_graph_ref=resolved_job_work_graph.reference,
            duplicate_policy_ref=duplicate_policy.reference,
            cross_item_policy_ref=cross_item_policy.reference,
            batch_policy_ref=batch_policy.reference,
            audit=audit,
        )
        return self._commit(
            task_id,
            request,
            values=(
                (resolved_job_work_graph, ResolvedJobWorkGraphV2),
                (duplicate_policy, DuplicateDetectionPolicyV2),
                (cross_item_policy, CrossItemSafetyPolicyV2),
                (batch_policy, BatchQualityPolicyV2),
            ),
            collections=((sources, BatchQualityItemSource),),
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_plan_review(
        self,
        task_id: str,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        plan_ref: ObjectRef,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = PlanReviewCapabilityRequestV1.create(
            run_id=run_id,
            plan_kind=plan_kind,
            plan_ref=plan_ref,
            audit=audit,
        )
        return self._commit(
            task_id,
            request,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def prepare_delivery(
        self,
        task_id: str,
        *,
        dataset_run_id: str,
        aggregate: CapabilityOwnerMaterial[FactoryDatasetAggregateResultV2],
        candidates: tuple[CapabilityOwnerMaterial[FactoryCandidateProjectionResult], ...],
        exports: tuple[CapabilityOwnerMaterial[AuthorizedCandidateExport], ...],
        policy: CapabilityOwnerMaterial[FactoryRunPolicyV2],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request = DeliveryCapabilityRequestV1.create(
            dataset_run_id=dataset_run_id,
            aggregate_ref=aggregate.reference,
            candidate_projection_refs=tuple(value.reference for value in candidates),
            export_refs=tuple(value.reference for value in exports),
            policy_ref=policy.reference,
            audit=audit,
        )
        values: tuple[tuple[CapabilityOwnerMaterial[Any], type[Any]], ...] = (
            (aggregate, FactoryDatasetAggregateResultV2),
            *((value, FactoryCandidateProjectionResult) for value in candidates),
            *((value, AuthorizedCandidateExport) for value in exports),
            (policy, FactoryRunPolicyV2),
        )
        return self._commit(
            task_id,
            request,
            values=values,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def _commit(
        self,
        task_id: str,
        request: CapabilityRequestV1,
        *,
        values: tuple[tuple[CapabilityOwnerMaterial[Any], type[Any]], ...] = (),
        collections: tuple[tuple[CapabilityOwnerCollection[Any], type[Any]], ...] = (),
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request_refs = request.audit.input_refs
        material_refs = tuple(value.reference for value, _ in values) + tuple(
            value.reference for value, _ in collections
        )
        if any(reference not in request_refs for reference in material_refs):
            raise GenericCapabilityAdapterError(
                "capability material is not bound by its request",
            )
        for material, expected_type in values:
            if not isinstance(material.value, expected_type):
                raise GenericCapabilityAdapterError(
                    "capability material has the wrong owner type",
                )
            self.materials.register(material.reference, material.value)
        for collection, expected_type in collections:
            if any(not isinstance(value, expected_type) for value in collection.values):
                raise GenericCapabilityAdapterError(
                    "capability material collection has the wrong owner type",
                )
            self.materials.register_many(
                collection.reference,
                collection.values,
            )
        return self.preparer.commit(
            task_id=task_id,
            request=request,
            source_authority_refs=request_refs,
            audit=audit,
            idempotency_key=idempotency_key,
        )


__all__ = [
    "CapabilityOwnerCollection",
    "CapabilityOwnerMaterial",
    "GenericAgentCapabilityPreparationService",
]
