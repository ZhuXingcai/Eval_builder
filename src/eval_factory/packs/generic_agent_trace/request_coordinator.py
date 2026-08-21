from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from eval_factory.agent_system.attachment_preparation_material import (
    FactoryAttachmentPreparationMaterialStore,
)
from eval_factory.agent_system.batch_quality_runtime import (
    FactoryBatchQualityContext,
)
from eval_factory.agent_system.candidate_output import (
    authorized_candidate_export_ref,
)
from eval_factory.agent_system.item_run_materializer import (
    FactoryItemRunMaterializer,
)
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.agent_system.trace_candidate_store import (
    FactoryTraceCandidateMaterialStore,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    CompiledAttachmentGenerationPlanV2,
    CompiledCriteriaRubricPlanV2,
    CompiledGradingDesignPlanV2,
    CriteriaRubricPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    GradingDesignPlanV2,
    PlanKindV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.batch_quality_v2 import (
    batch_quality_policy_v2_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.cross_item_safety_v2 import (
    cross_item_safety_policy_v2_ref,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
    FactoryItemRunBindingV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.duplicate_v2 import (
    duplicate_detection_policy_v2_ref,
)
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.contracts.orchestration_v2 import (
    dataset_item_id_v2,
    resolved_job_work_graph_v2_ref,
)
from eval_factory.contracts.task_v2 import (
    evaluator_spec_ref,
    reference_policy_ref,
    rubric_set_ref,
    task_draft_ref,
    tool_policy_ref,
)
from eval_factory.harness.contracts import static_object_ref
from eval_factory.packs.generic_agent_trace.candidate_aggregate import (
    GenericAgentCandidateAggregate,
)
from eval_factory.packs.generic_agent_trace.delivery_preparation import (
    GenericAgentDeliveryPreparation,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflowError,
    GenericAgentTeamTaskSpec,
)
from eval_factory.packs.generic_agent_trace.owner_refs import (
    generic_agent_tool_catalog_ref,
)
from eval_factory.packs.generic_agent_trace.plan_review_preparation import (
    GenericAgentItemReviewMaterialSource,
)
from eval_factory.packs.generic_agent_trace.request_factories import (
    CapabilityOwnerCollection,
    CapabilityOwnerMaterial,
    GenericAgentCapabilityPreparationService,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterialization,
)
from eval_factory.team import TeamSnapshot, TeamTaskStatusV1, TeamTaskWork


class GenericAgentExecutionContextSource(Protocol):
    def ensure(self) -> object: ...

    def batch_context(self) -> FactoryBatchQualityContext: ...


class GenericAgentFactoryRequestCoordinator:
    """Prepares ready-task requests from current first-party owner stores."""

    def __init__(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        requirement: EvaluationRequirementSpecV2,
        policy: FactoryRunPolicyV2,
        factory_store: FactoryControlStore,
        candidate_store: FactoryTraceCandidateMaterialStore,
        item_materializer: FactoryItemRunMaterializer,
        materialization: GenericAgentTaskGraphMaterialization,
        preparation: GenericAgentCapabilityPreparationService,
        trace_sources: Mapping[ObjectRef, TraceSourceRef],
        attachment_preparation_materials: (FactoryAttachmentPreparationMaterialStore | None) = None,
        item_materials: GenericAgentItemReviewMaterialSource | None = None,
        candidate_aggregate: GenericAgentCandidateAggregate | None = None,
        batch_quality_context: FactoryBatchQualityContext | None = None,
        delivery_preparation: GenericAgentDeliveryPreparation | None = None,
        execution_context: GenericAgentExecutionContextSource | None = None,
    ) -> None:
        self.request = request
        self.dataset_run_id = request.dataset_run_id
        self.requirement = requirement
        self.policy = policy
        self.factory_store = factory_store
        self.candidate_store = candidate_store
        self.item_materializer = item_materializer
        self.materialization = materialization
        self.preparation = preparation
        self.trace_sources = dict(trace_sources)
        self.attachment_preparation_materials = attachment_preparation_materials
        self.item_materials = item_materials
        self.candidate_aggregate = candidate_aggregate
        self.batch_quality_context = batch_quality_context
        self.delivery_preparation = delivery_preparation
        self.execution_context = execution_context
        if candidate_aggregate is None and (
            batch_quality_context is not None or execution_context is not None
        ):
            raise ValueError(
                "Batch execution context requires candidate aggregation",
            )
        if candidate_aggregate is not None and (batch_quality_context is None and execution_context is None):
            raise ValueError(
                "candidate aggregation requires a Batch execution context",
            )
        if requirement.run_id != request.dataset_run_id or set(
            self.trace_sources,
        ) != {
            instance.scope_ref
            for instance in materialization.task_instances
            if instance.capability_id == "capability.trace-ingestion"
        }:
            raise ValueError(
                "request coordinator inputs differ from materialized sources",
            )

    def prepare(
        self,
        *,
        snapshot: TeamSnapshot,
        work: TeamTaskWork,
        spec: GenericAgentTeamTaskSpec,
        audit: ContractAudit,
    ) -> None:
        if (
            snapshot.team.team_id
            != work.task.task_id.split(
                ".task-",
                1,
            )[0]
        ):
            raise GenericAgentFactoryWorkflowError(
                "request preparation task belongs to another Team",
            )
        instance = self.materialization.instance(work.task.task_id)
        if instance.capability_id != spec.capability_id:
            raise GenericAgentFactoryWorkflowError(
                "request preparation capability metadata drifted",
            )
        if instance.capability_id == "capability.requirement-planning":
            requirement_plan, _compiled = self.factory_store.get_plan(
                self.dataset_run_id,
            )
            self.preparation.prepare_requirement_planning(
                work.task.task_id,
                plan=CapabilityOwnerMaterial(
                    requirement_plan.to_ref(),
                    requirement_plan,
                ),
                policy=CapabilityOwnerMaterial(
                    self.policy.to_ref(),
                    self.policy,
                ),
                audit=audit,
                idempotency_key=self._key(
                    work,
                    requirement_plan.to_ref(),
                ),
            )
            return
        if instance.capability_id == "capability.trace-ingestion":
            source_ref = self._source_ref(instance.scope_ref)
            source = self.trace_sources[source_ref]
            attempt = max(
                1,
                work.attempt + (1 if work.status is TeamTaskStatusV1.READY else 0),
            )
            self.preparation.prepare_trace_ingestion(
                work.task.task_id,
                trace_source=CapabilityOwnerMaterial(
                    source_ref,
                    source,
                ),
                job_id=self.dataset_run_id,
                attempt=attempt,
                audit=audit,
                idempotency_key=self._key(work, source_ref),
            )
            return
        if instance.capability_id == "capability.task-authoring":
            if self.execution_context is not None:
                self.execution_context.ensure()
            source_ref = self._source_ref(instance.scope_ref)
            candidate_material = self.candidate_store.get(
                run_id=self.dataset_run_id,
                trace_source_ref=source_ref,
            )
            candidate = candidate_material.preparation
            if (
                candidate.decision.disposition is not TraceCandidateDispositionV2.CANDIDATE
                or candidate.extracted_prompt is None
                or candidate.inferred_intent is None
                or candidate.rewrite_candidate is None
            ):
                raise GenericAgentFactoryWorkflowError(
                    "task authoring requires current candidate material",
                )
            self.item_materializer.materialize(
                request=self.request,
                policy=self.policy,
                requirement=self.requirement,
                source_trace_ref=self._item_source_ref(
                    source_ref,
                ),
                candidate=candidate,
                audit=audit,
            )
            self.preparation.prepare_task_authoring(
                work.task.task_id,
                decision=CapabilityOwnerMaterial(
                    candidate.decision.to_ref(),
                    candidate.decision,
                ),
                extracted_prompt=CapabilityOwnerMaterial(
                    candidate.extracted_prompt.to_ref(),
                    candidate.extracted_prompt,
                ),
                intent=CapabilityOwnerMaterial(
                    candidate.inferred_intent.to_ref(),
                    candidate.inferred_intent,
                ),
                rewrite=CapabilityOwnerMaterial(
                    candidate.rewrite_candidate.to_ref(),
                    candidate.rewrite_candidate,
                ),
                requirement=CapabilityOwnerMaterial(
                    self.requirement.to_ref(),
                    self.requirement,
                ),
                audit=audit,
                idempotency_key=self._key(
                    work,
                    candidate_material.authority.to_ref(),
                ),
            )
            return
        if instance.capability_id == ("capability.attachment-reconstruction"):
            if self.attachment_preparation_materials is None:
                raise GenericAgentFactoryWorkflowError(
                    "attachment request preparation is unavailable",
                )
            source_ref = self._source_ref(instance.scope_ref)
            binding = self.factory_store.get_item_binding(
                self.dataset_run_id,
                dataset_item_id_v2(
                    self.dataset_run_id,
                    self._item_source_ref(source_ref),
                ),
            )
            item_run = self.factory_store.get_item_run(binding)
            attachment_plan_material = self.factory_store.get_domain_plan(
                item_run.run_id,
                PlanKindV2.ATTACHMENT_GENERATION,
            )
            attachment_plan = AttachmentGenerationPlanV2.model_validate_json(
                attachment_plan_material.plan_record_json,
            )
            compiled_attachment_plan = CompiledAttachmentGenerationPlanV2.model_validate_json(
                attachment_plan_material.compiled_plan_record_json,
            )
            preparation_ref = self.factory_store.get_domain_plan_material_ref(
                run_id=item_run.run_id,
                plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            )
            attachment_preparation = self.attachment_preparation_materials.get(
                preparation_ref,
            )
            self.preparation.prepare_attachment_reconstruction(
                work.task.task_id,
                run_id=item_run.run_id,
                plan=CapabilityOwnerMaterial(
                    attachment_plan.to_ref(),
                    attachment_plan,
                ),
                compiled_plan=CapabilityOwnerMaterial(
                    compiled_attachment_plan.to_ref(),
                    compiled_attachment_plan,
                ),
                preparation=CapabilityOwnerMaterial(
                    preparation_ref,
                    attachment_preparation,
                ),
                prior_batch=None,
                lease_duration_seconds=600,
                audit=audit,
                idempotency_key=self._key(
                    work,
                    attachment_plan.to_ref(),
                ),
            )
            return
        if instance.capability_id == "capability.quality-review":
            source_ref = self._source_ref(instance.scope_ref)
            binding = self.factory_store.get_item_binding(
                self.dataset_run_id,
                dataset_item_id_v2(
                    self.dataset_run_id,
                    self._item_source_ref(source_ref),
                ),
            )
            task_head = self.factory_store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.TASK_AUTHORING,
            )
            attachment_head = self.factory_store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.ATTACHMENT,
            )
            task_material_ref = self.factory_store.get_item_stage_material_ref(
                task_head.to_ref(),
            )
            attachment_material_ref = self.factory_store.get_item_stage_material_ref(
                attachment_head.to_ref(),
            )
            self.preparation.prepare_attachment_quality_workflow(
                work.task.task_id,
                item_binding_ref=binding.to_ref(),
                task_authoring_material_ref=task_material_ref,
                attachment_execution_material_ref=(attachment_material_ref),
                audit=audit,
                idempotency_key=self._key(
                    work,
                    attachment_head.to_ref(),
                ),
            )
            return
        if instance.capability_id == "capability.criteria-rubric":
            if self.item_materials is None:
                raise GenericAgentFactoryWorkflowError(
                    "Criteria request preparation is unavailable",
                )
            source_ref = self._source_ref(instance.scope_ref)
            binding = self._item_binding(source_ref)
            item_run = self.factory_store.get_item_run(binding)
            criteria_source = self.item_materials.criteria(
                binding.item_id,
            )
            plan_material = self.factory_store.get_domain_plan(
                item_run.run_id,
                PlanKindV2.CRITERIA_RUBRIC,
            )
            criteria_plan = CriteriaRubricPlanV2.model_validate_json(
                plan_material.plan_record_json,
            )
            compiled_criteria_plan = CompiledCriteriaRubricPlanV2.model_validate_json(
                plan_material.compiled_plan_record_json,
            )
            self.preparation.prepare_criteria_rubric(
                work.task.task_id,
                run_id=item_run.run_id,
                plan=CapabilityOwnerMaterial(
                    criteria_plan.to_ref(),
                    criteria_plan,
                ),
                compiled_plan=CapabilityOwnerMaterial(
                    compiled_criteria_plan.to_ref(),
                    compiled_criteria_plan,
                ),
                task_draft=CapabilityOwnerMaterial(
                    task_draft_ref(criteria_source.task_draft),
                    criteria_source.task_draft,
                ),
                attachment_quality=CapabilityOwnerMaterial(
                    criteria_source.quality.result.finalization.attachment_quality.to_ref(),
                    criteria_source.quality.result.finalization.attachment_quality,
                ),
                solvability=CapabilityOwnerMaterial(
                    criteria_source.quality.result.solvability.assessment.to_ref(),
                    criteria_source.quality.result.solvability.assessment,
                ),
                binding_definitions=CapabilityOwnerCollection(
                    criteria_source.binding_definitions_ref,
                    criteria_source.binding_definitions,
                ),
                tool_catalog=CapabilityOwnerMaterial(
                    generic_agent_tool_catalog_ref(
                        criteria_source.tool_catalog,
                    ),
                    criteria_source.tool_catalog,
                ),
                lease_duration_seconds=600,
                audit=audit,
                idempotency_key=self._key(
                    work,
                    criteria_plan.to_ref(),
                ),
            )
            return
        if instance.capability_id == "capability.grading-design":
            if self.item_materials is None:
                raise GenericAgentFactoryWorkflowError(
                    "Grading request preparation is unavailable",
                )
            source_ref = self._source_ref(instance.scope_ref)
            binding = self._item_binding(source_ref)
            item_run = self.factory_store.get_item_run(binding)
            grading_source = self.item_materials.grading(
                binding.item_id,
            )
            criteria_authority = grading_source.criteria.authority
            if criteria_authority is None:
                raise GenericAgentFactoryWorkflowError(
                    "Grading request requires Criteria authority",
                )
            criteria = criteria_authority.execution
            if (
                criteria.rubric_set is None
                or criteria.evaluator_spec is None
                or criteria.reference_policy is None
                or criteria.tool_policy is None
            ):
                raise GenericAgentFactoryWorkflowError(
                    "Grading request requires the complete Criteria chain",
                )
            plan_material = self.factory_store.get_domain_plan(
                item_run.run_id,
                PlanKindV2.GRADING_DESIGN,
            )
            grading_plan = GradingDesignPlanV2.model_validate_json(
                plan_material.plan_record_json,
            )
            compiled_grading_plan = CompiledGradingDesignPlanV2.model_validate_json(
                plan_material.compiled_plan_record_json,
            )
            self.preparation.prepare_grading_design(
                work.task.task_id,
                run_id=item_run.run_id,
                plan=CapabilityOwnerMaterial(
                    grading_plan.to_ref(),
                    grading_plan,
                ),
                compiled_plan=CapabilityOwnerMaterial(
                    compiled_grading_plan.to_ref(),
                    compiled_grading_plan,
                ),
                criteria_result=CapabilityOwnerMaterial(
                    criteria.result.to_ref(),
                    criteria.result,
                ),
                criteria_route=CapabilityOwnerMaterial(
                    criteria.route.to_ref(),
                    criteria.route,
                ),
                rubric_set=CapabilityOwnerMaterial(
                    rubric_set_ref(criteria.rubric_set),
                    criteria.rubric_set,
                ),
                evaluator_spec=CapabilityOwnerMaterial(
                    evaluator_spec_ref(criteria.evaluator_spec),
                    criteria.evaluator_spec,
                ),
                reference_policy=CapabilityOwnerMaterial(
                    reference_policy_ref(criteria.reference_policy),
                    criteria.reference_policy,
                ),
                tool_policy=CapabilityOwnerMaterial(
                    tool_policy_ref(criteria.tool_policy),
                    criteria.tool_policy,
                ),
                model_authorizations=CapabilityOwnerCollection(
                    grading_source.model_authorizations_ref,
                    grading_source.model_authorizations,
                ),
                evaluated_at=grading_source.evaluated_at,
                lease_duration_seconds=600,
                audit=audit,
                idempotency_key=self._key(
                    work,
                    grading_plan.to_ref(),
                ),
            )
            return
        if instance.capability_id == "capability.batch-quality":
            if (
                self.item_materials is None
                or self.candidate_aggregate is None
                or (self.batch_quality_context is None and self.execution_context is None)
            ):
                raise GenericAgentFactoryWorkflowError(
                    "Batch Quality request preparation is unavailable",
                )
            core_result = self.candidate_aggregate.build(
                audit=audit,
            )
            bindings = self.factory_store.list_item_bindings(
                self.dataset_run_id,
            )
            sources = tuple(self.item_materials.batch_source(binding.item_id) for binding in bindings)
            sources_ref = static_object_ref(
                object_type="batch-quality-sources",
                object_id=(f"batch-quality-sources://{core_result.object_sha256}"),
                object_version="v2",
                payload={
                    "item_ids": [source.item_id for source in sources],
                },
            )
            context = (
                self.batch_quality_context
                if self.batch_quality_context is not None
                else self._dynamic_batch_context()
            )
            self.preparation.prepare_batch_quality(
                work.task.task_id,
                dataset_run_id=self.dataset_run_id,
                core_vertical_result_ref=core_result.to_ref(),
                sources=CapabilityOwnerCollection(
                    sources_ref,
                    sources,
                ),
                rejected_binding_refs=(),
                blocked_binding_refs=(),
                resolved_job_work_graph=CapabilityOwnerMaterial(
                    resolved_job_work_graph_v2_ref(
                        context.resolved_job_work_graph,
                    ),
                    context.resolved_job_work_graph,
                ),
                duplicate_policy=CapabilityOwnerMaterial(
                    duplicate_detection_policy_v2_ref(
                        context.duplicate_policy,
                    ),
                    context.duplicate_policy,
                ),
                cross_item_policy=CapabilityOwnerMaterial(
                    cross_item_safety_policy_v2_ref(
                        context.cross_item_policy,
                    ),
                    context.cross_item_policy,
                ),
                batch_policy=CapabilityOwnerMaterial(
                    batch_quality_policy_v2_ref(
                        context.batch_policy,
                    ),
                    context.batch_policy,
                ),
                audit=audit,
                idempotency_key=self._key(
                    work,
                    core_result.to_ref(),
                ),
            )
            return
        if (
            instance.capability_id == "capability.plan-review"
            and instance.plan_kind is PlanKindV2.FINAL_DELIVERY
        ):
            aggregate = self.factory_store.get_dataset_aggregate(
                self.dataset_run_id,
            )
            try:
                plan_ref = self.factory_store.get_domain_plan(
                    self.dataset_run_id,
                    PlanKindV2.FINAL_DELIVERY,
                ).plan_ref
            except FactoryControlNotFoundError:
                plan_ref = aggregate.to_ref()
            self.preparation.prepare_plan_review(
                work.task.task_id,
                run_id=self.dataset_run_id,
                plan_kind=PlanKindV2.FINAL_DELIVERY,
                plan_ref=plan_ref,
                audit=audit,
                idempotency_key=self._key(work, plan_ref),
            )
            return
        if instance.capability_id == "capability.delivery":
            if self.delivery_preparation is None:
                raise GenericAgentFactoryWorkflowError(
                    "Delivery request preparation is unavailable",
                )
            material = self.delivery_preparation.material()
            candidates = tuple(
                sorted(
                    (
                        CapabilityOwnerMaterial(
                            value.projection.to_ref(),
                            value,
                        )
                        for value in material.candidates
                    ),
                    key=lambda value: self._ref_key(value.reference),
                )
            )
            exports = tuple(
                sorted(
                    (
                        CapabilityOwnerMaterial(
                            authorized_candidate_export_ref(value),
                            value,
                        )
                        for value in material.exports
                    ),
                    key=lambda value: self._ref_key(value.reference),
                )
            )
            self.preparation.prepare_delivery(
                work.task.task_id,
                dataset_run_id=self.dataset_run_id,
                aggregate=CapabilityOwnerMaterial(
                    material.aggregate.to_ref(),
                    material.aggregate,
                ),
                candidates=candidates,
                exports=exports,
                policy=CapabilityOwnerMaterial(
                    material.policy.to_ref(),
                    material.policy,
                ),
                audit=audit,
                idempotency_key=self._key(
                    work,
                    material.aggregate.to_ref(),
                ),
            )
            return
        if (
            instance.capability_id == "capability.plan-review"
            and instance.plan_kind is PlanKindV2.GLOBAL_BUILD
        ):
            global_plan, _compiled = self.factory_store.get_plan(
                self.dataset_run_id,
            )
            self.preparation.prepare_plan_review(
                work.task.task_id,
                run_id=self.dataset_run_id,
                plan_kind=PlanKindV2.GLOBAL_BUILD,
                plan_ref=global_plan.to_ref(),
                audit=audit,
                idempotency_key=self._key(
                    work,
                    global_plan.to_ref(),
                ),
            )
            return
        if instance.capability_id == "capability.plan-review" and instance.plan_kind in {
            PlanKindV2.ATTACHMENT_GENERATION,
            PlanKindV2.CRITERIA_RUBRIC,
            PlanKindV2.GRADING_DESIGN,
        }:
            source_ref = self._source_ref(instance.scope_ref)
            binding = self.factory_store.get_item_binding(
                self.dataset_run_id,
                dataset_item_id_v2(
                    self.dataset_run_id,
                    self._item_source_ref(source_ref),
                ),
            )
            item_run = self.factory_store.get_item_run(binding)
            try:
                plan_ref = self.factory_store.get_domain_plan(
                    item_run.run_id,
                    instance.plan_kind,
                ).plan_ref
            except FactoryControlNotFoundError:
                predecessor_stage = {
                    PlanKindV2.ATTACHMENT_GENERATION: (FactoryItemStageV2.TASK_AUTHORING),
                    PlanKindV2.CRITERIA_RUBRIC: (FactoryItemStageV2.ITEM_QUALITY),
                    PlanKindV2.GRADING_DESIGN: (FactoryItemStageV2.CRITERIA_RUBRIC),
                }[instance.plan_kind]
                plan_ref = self.factory_store.get_item_stage_head(
                    binding.item_id,
                    predecessor_stage,
                ).result_ref
            self.preparation.prepare_plan_review(
                work.task.task_id,
                run_id=item_run.run_id,
                plan_kind=instance.plan_kind,
                plan_ref=plan_ref,
                audit=audit,
                idempotency_key=self._key(work, plan_ref),
            )
            return
        raise GenericAgentFactoryWorkflowError(
            "first-party request preparation is unavailable for ready task",
        )

    def _dynamic_batch_context(
        self,
    ) -> FactoryBatchQualityContext:
        assert self.execution_context is not None
        return self.execution_context.batch_context()

    @staticmethod
    def _source_ref(reference: ObjectRef | None) -> ObjectRef:
        if reference is None:
            raise GenericAgentFactoryWorkflowError(
                "source-scoped task has no source authority",
            )
        return reference

    def _item_binding(
        self,
        source_ref: ObjectRef,
    ) -> FactoryItemRunBindingV2:
        return self.factory_store.get_item_binding(
            self.dataset_run_id,
            dataset_item_id_v2(
                self.dataset_run_id,
                self._item_source_ref(source_ref),
            ),
        )

    def _item_source_ref(
        self,
        source_ref: ObjectRef,
    ) -> ObjectRef:
        source = self.trace_sources[source_ref]
        return source_ref.model_copy(
            update={"object_version": source.adapter_version},
        )

    @staticmethod
    def _ref_key(reference: ObjectRef) -> tuple[str, str, str, str]:
        return (
            reference.object_type,
            reference.object_id,
            reference.object_version,
            reference.object_sha256,
        )

    @staticmethod
    def _key(
        work: TeamTaskWork,
        source_ref: ObjectRef,
    ) -> str:
        return (
            "prepare-capability-request:"
            f"{work.task.object_sha256}:"
            f"{source_ref.object_sha256}:"
            f"{max(1, work.attempt)}"
        )


__all__ = [
    "GenericAgentExecutionContextSource",
    "GenericAgentFactoryRequestCoordinator",
]
