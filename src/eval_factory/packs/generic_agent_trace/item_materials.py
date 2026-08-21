from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from eval_factory.agent_system.attachment_execution_material import (
    FactoryAttachmentExecutionMaterialStore,
)
from eval_factory.agent_system.attachment_quality_material import (
    FactoryAttachmentQualityMaterialStore,
)
from eval_factory.agent_system.attachment_quality_runtime import (
    FactoryAttachmentQualityContext,
    FactoryAttachmentQualityView,
)
from eval_factory.agent_system.criteria_authority_material import (
    FactoryCriteriaAuthorityMaterialStore,
)
from eval_factory.agent_system.criteria_item_runtime import (
    FactoryCriteriaItemState,
    FactoryCriteriaItemView,
)
from eval_factory.agent_system.grading_authority_material import (
    FactoryGradingAuthorityMaterialStore,
)
from eval_factory.agent_system.job_store_witness_bridge import (
    FactoryItemStageWitnessSource,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.release_source_builder import (
    FactoryReleaseSourceItem,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.task_authoring_material import (
    FactoryTaskAuthoringMaterialStore,
)
from eval_factory.batch_quality.reports import BatchQualityItemSource
from eval_factory.contracts.agent_system_v2 import (
    CompiledCriteriaRubricPlanV2,
    CriteriaRubricPlanV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import FactoryItemStageV2
from eval_factory.contracts.labeling_v2 import label_decision_ref
from eval_factory.contracts.quality_v2 import (
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.task_v2 import (
    r4_task_contract_set_ref,
    task_prompt_safety_gate_ref,
)
from eval_factory.packs.generic_agent_trace.plan_review_preparation import (
    GenericAgentCriteriaReviewMaterial,
    GenericAgentGradingReviewMaterial,
)
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    ToolCapabilityCatalog,
    VerifiedModelDomainAuthorization,
)


class GenericAgentFactoryItemMaterialError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GenericAgentCriteriaContext:
    quality_context: FactoryAttachmentQualityContext
    binding_definitions: tuple[EvaluatorBindingDefinition, ...]
    binding_definitions_ref: ObjectRef
    tool_catalog: ToolCapabilityCatalog


@dataclass(frozen=True, slots=True)
class GenericAgentGradingContext:
    model_authorizations: tuple[VerifiedModelDomainAuthorization, ...]
    model_authorizations_ref: ObjectRef
    evaluated_at: datetime


class GenericAgentItemContextSource(Protocol):
    def criteria_context(
        self,
        item_id: str,
    ) -> GenericAgentCriteriaContext: ...

    def grading_context(
        self,
        item_id: str,
    ) -> GenericAgentGradingContext: ...

    def source_trace_ref(self, item_id: str) -> ObjectRef: ...


class GenericAgentFactoryItemMaterialSource:
    """Rebuilds specialist request material from current owner heads."""

    def __init__(
        self,
        *,
        dataset_run_id: str,
        store: FactoryControlStore,
        plan_reviews: PlanReviewService,
        task_authoring_materials: FactoryTaskAuthoringMaterialStore,
        quality_materials: FactoryAttachmentQualityMaterialStore,
        criteria_materials: FactoryCriteriaAuthorityMaterialStore,
        attachment_materials: (FactoryAttachmentExecutionMaterialStore | None) = None,
        grading_materials: (FactoryGradingAuthorityMaterialStore | None) = None,
        criteria_contexts: (Mapping[str, GenericAgentCriteriaContext] | None) = None,
        grading_contexts: (Mapping[str, GenericAgentGradingContext] | None) = None,
        context_source: GenericAgentItemContextSource | None = None,
    ) -> None:
        self.dataset_run_id = dataset_run_id
        self.store = store
        self.plan_reviews = plan_reviews
        self.task_authoring_materials = task_authoring_materials
        self.quality_materials = quality_materials
        self.criteria_materials = criteria_materials
        self.attachment_materials = attachment_materials
        self.grading_materials = grading_materials
        self.criteria_contexts = dict(criteria_contexts or {})
        self.grading_contexts = dict(grading_contexts or {})
        self.context_source = context_source
        has_static_contexts = bool(self.criteria_contexts or self.grading_contexts)
        if has_static_contexts == (context_source is not None):
            raise ValueError(
                "item materials require one context authority source",
            )

    def get(
        self,
        item_id: str,
    ) -> FactoryAttachmentQualityContext:
        return self._criteria_context(item_id).quality_context

    def criteria(
        self,
        item_id: str,
    ) -> GenericAgentCriteriaReviewMaterial:
        context = self._criteria_context(item_id)
        task_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.TASK_AUTHORING,
        )
        quality_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.ITEM_QUALITY,
        )
        task_material_ref = self.store.get_item_stage_material_ref(
            task_head.to_ref(),
        )
        quality_material_ref = self.store.get_item_stage_material_ref(
            quality_head.to_ref(),
        )
        task_authoring = self.task_authoring_materials.get(
            task_material_ref,
        )
        quality = FactoryAttachmentQualityView(
            item_id=item_id,
            item_quality_head_ref=quality_head.to_ref(),
            material_ref=quality_material_ref,
            result=self.quality_materials.get(quality_material_ref),
        )
        return GenericAgentCriteriaReviewMaterial(
            task_draft=task_authoring.task_draft,
            task_contract_set=task_authoring.task_contract_set,
            quality=quality,
            quality_context=context.quality_context,
            binding_definitions=context.binding_definitions,
            binding_definitions_ref=context.binding_definitions_ref,
            tool_catalog=context.tool_catalog,
        )

    def grading(
        self,
        item_id: str,
    ) -> GenericAgentGradingReviewMaterial:
        context = self._grading_context(item_id)
        binding = self.store.get_item_binding(
            self.dataset_run_id,
            item_id,
        )
        item_run = self.store.get_item_run(binding)
        plan_material = self.store.get_domain_plan(
            item_run.run_id,
            PlanKindV2.CRITERIA_RUBRIC,
        )
        plan = CriteriaRubricPlanV2.model_validate_json(
            plan_material.plan_record_json,
        )
        compiled = CompiledCriteriaRubricPlanV2.model_validate_json(
            plan_material.compiled_plan_record_json,
        )
        review = self.plan_reviews.require_resumed_plan(
            run_id=item_run.run_id,
            plan_kind=PlanKindV2.CRITERIA_RUBRIC,
            plan_ref=plan.to_ref(),
        )
        stage_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.CRITERIA_RUBRIC,
        )
        material_ref = self.store.get_item_stage_material_ref(
            stage_head.to_ref(),
        )
        criteria = FactoryCriteriaItemView(
            item_id=item_id,
            plan_ref=plan.to_ref(),
            compiled_plan_ref=compiled.to_ref(),
            review_ref=review.request.to_ref(),
            stage_head_ref=stage_head.to_ref(),
            material_ref=material_ref,
            authority=self.criteria_materials.get(material_ref),
            state=FactoryCriteriaItemState.EXECUTED,
        )
        return GenericAgentGradingReviewMaterial(
            criteria=criteria,
            model_authorizations=context.model_authorizations,
            model_authorizations_ref=context.model_authorizations_ref,
            evaluated_at=context.evaluated_at,
        )

    def batch_source(self, item_id: str) -> BatchQualityItemSource:
        criteria = self.criteria(item_id)
        grading = self.grading(item_id)
        authority = grading.criteria.authority
        if (
            authority is None
            or authority.reviewed_task_contract_set is None
            or authority.reviewed_item_quality is None
        ):
            raise GenericAgentFactoryItemMaterialError(
                "Batch source requires reviewed Criteria authority",
            )
        task_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.TASK_AUTHORING,
        )
        task_material_ref = self.store.get_item_stage_material_ref(
            task_head.to_ref(),
        )
        task_authoring = self.task_authoring_materials.get(
            task_material_ref,
        )
        gate = task_authoring.prompt_safety_result.task_prompt_safety_gate
        if gate is None:
            raise GenericAgentFactoryItemMaterialError(
                "Batch source requires prompt safety authority",
            )
        if task_prompt_safety_gate_ref(gate) != task_authoring.task_contract_set.task_prompt_safety_gate_ref:
            raise GenericAgentFactoryItemMaterialError(
                "Batch source prompt safety authority drifted",
            )
        return BatchQualityItemSource(
            item_id=item_id,
            source_trace_ref=self._source_trace_ref(
                item_id,
                task_authoring.source_trace_ref,
            ),
            trace_envelope=task_authoring.trace_envelope,
            label_decisions=(task_authoring.label_decision,),
            selection_context=task_authoring.selection_context,
            task_draft=criteria.task_draft,
            task_prompt_safety_gate=gate,
            leakage_reference_set=(task_authoring.leakage_reference_set),
            task_contract_set=authority.reviewed_task_contract_set,
            item_quality=authority.reviewed_item_quality,
        )

    def release_source(
        self,
        item_id: str,
    ) -> FactoryReleaseSourceItem:
        if self.attachment_materials is None or self.grading_materials is None:
            raise GenericAgentFactoryItemMaterialError(
                "Delivery material stores are unavailable",
            )
        binding = self.store.get_item_binding(
            self.dataset_run_id,
            item_id,
        )
        task_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.TASK_AUTHORING,
        )
        attachment_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.ATTACHMENT,
        )
        criteria_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.CRITERIA_RUBRIC,
        )
        grading_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.GRADING_DESIGN,
        )
        task_authoring = self.task_authoring_materials.get(
            self.store.get_item_stage_material_ref(
                task_head.to_ref(),
            )
        )
        attachment = self.attachment_materials.get(
            self.store.get_item_stage_material_ref(
                attachment_head.to_ref(),
            )
        )
        criteria = self.criteria_materials.get(
            self.store.get_item_stage_material_ref(
                criteria_head.to_ref(),
            )
        )
        grading = self.grading_materials.get(
            self.store.get_item_stage_material_ref(
                grading_head.to_ref(),
            )
        )
        gate = task_authoring.prompt_safety_result.task_prompt_safety_gate
        if (
            gate is None
            or criteria.reviewed_task_contract_set is None
            or criteria.reviewed_item_quality is None
        ):
            raise GenericAgentFactoryItemMaterialError(
                "Delivery source authority is incomplete",
            )
        quality = self.criteria(item_id).quality
        return FactoryReleaseSourceItem(
            binding=binding,
            source_trace_ref=self._source_trace_ref(
                item_id,
                task_authoring.source_trace_ref,
            ),
            label_decisions=(task_authoring.label_decision,),
            task_draft=task_authoring.task_draft,
            task_prompt_safety_gate=gate,
            base_task_contract_set=task_authoring.task_contract_set,
            task_contract_set=criteria.reviewed_task_contract_set,
            attachment_result=(attachment.r5_execution.reconstruction_result),
            attachment_item_quality_ref=(
                item_quality_compilation_result_ref(
                    quality.result.finalization.item_quality,
                )
            ),
            item_quality=criteria.reviewed_item_quality,
            criteria_result_ref=criteria.execution.result.to_ref(),
            criteria_material_ref=(
                self.store.get_item_stage_material_ref(
                    criteria_head.to_ref(),
                )
            ),
            grading_result_ref=grading.execution.result.to_ref(),
        )

    def witness_source(
        self,
        item_id: str,
    ) -> FactoryItemStageWitnessSource:
        task_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.TASK_AUTHORING,
        )
        attachment_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.ATTACHMENT,
        )
        task_authoring = self.task_authoring_materials.get(
            self.store.get_item_stage_material_ref(
                task_head.to_ref(),
            )
        )
        criteria = self.grading(item_id).criteria.authority
        if self.attachment_materials is None or criteria is None or criteria.reviewed_item_quality is None:
            raise GenericAgentFactoryItemMaterialError(
                "JobStore witness authority is incomplete",
            )
        attachment = self.attachment_materials.get(
            self.store.get_item_stage_material_ref(
                attachment_head.to_ref(),
            )
        )
        gate = task_authoring.prompt_safety_result.task_prompt_safety_gate
        if gate is None:
            raise GenericAgentFactoryItemMaterialError(
                "JobStore witness has no prompt safety gate",
            )
        return FactoryItemStageWitnessSource(
            item_id=item_id,
            trace_index_ref=task_authoring.source_trace_ref,
            safety_ref=task_prompt_safety_gate_ref(gate),
            label_ref=label_decision_ref(
                task_authoring.label_decision,
            ),
            task_authoring_ref=r4_task_contract_set_ref(
                task_authoring.task_contract_set,
            ),
            attachment_ref=(attachment.subgraph_result.to_ref()),
            item_quality_ref=item_quality_compilation_result_ref(
                criteria.reviewed_item_quality,
            ),
        )

    def _criteria_context(
        self,
        item_id: str,
    ) -> GenericAgentCriteriaContext:
        context = self.criteria_contexts.get(item_id)
        if context is None and self.context_source is not None:
            context = self.context_source.criteria_context(item_id)
        if context is None:
            raise GenericAgentFactoryItemMaterialError(
                "Criteria context is unavailable",
            )
        return context

    def _grading_context(
        self,
        item_id: str,
    ) -> GenericAgentGradingContext:
        context = self.grading_contexts.get(item_id)
        if context is None and self.context_source is not None:
            context = self.context_source.grading_context(item_id)
        if context is None:
            raise GenericAgentFactoryItemMaterialError(
                "Grading context is unavailable",
            )
        return context

    def _source_trace_ref(
        self,
        item_id: str,
        fallback: ObjectRef,
    ) -> ObjectRef:
        if self.context_source is None:
            return fallback
        return self.context_source.source_trace_ref(item_id)


__all__ = [
    "GenericAgentCriteriaContext",
    "GenericAgentFactoryItemMaterialError",
    "GenericAgentFactoryItemMaterialSource",
    "GenericAgentGradingContext",
    "GenericAgentItemContextSource",
]
