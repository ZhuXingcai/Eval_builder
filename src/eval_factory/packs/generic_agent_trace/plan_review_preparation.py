from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from eval_factory.agent_system.attachment_item_runtime import (
    FactoryAttachmentItemRuntime,
)
from eval_factory.agent_system.attachment_quality_runtime import (
    FactoryAttachmentQualityContext,
    FactoryAttachmentQualityView,
)
from eval_factory.agent_system.criteria_item_runtime import (
    FactoryCriteriaItemRuntime,
    FactoryCriteriaItemView,
)
from eval_factory.agent_system.grading_item_runtime import (
    FactoryGradingItemRuntime,
)
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.agent_system.task_authoring_material import (
    FactoryTaskAuthoringMaterialStore,
)
from eval_factory.batch_quality.reports import BatchQualityItemSource
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunPolicyV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import FactoryItemStageV2
from eval_factory.contracts.task_v2 import R4TaskContractSetV2, TaskDraftV2
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    PlanReviewCapabilityRequestV1,
)
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    ToolCapabilityCatalog,
    VerifiedModelDomainAuthorization,
)


class GenericAgentPlanReviewPreparationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GenericAgentPreparedReview:
    run_id: str
    plan_kind: PlanKindV2
    plan_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class GenericAgentCriteriaReviewMaterial:
    task_draft: TaskDraftV2
    task_contract_set: R4TaskContractSetV2
    quality: FactoryAttachmentQualityView
    quality_context: FactoryAttachmentQualityContext
    binding_definitions: tuple[EvaluatorBindingDefinition, ...]
    binding_definitions_ref: ObjectRef
    tool_catalog: ToolCapabilityCatalog


@dataclass(frozen=True, slots=True)
class GenericAgentGradingReviewMaterial:
    criteria: FactoryCriteriaItemView
    model_authorizations: tuple[VerifiedModelDomainAuthorization, ...]
    model_authorizations_ref: ObjectRef
    evaluated_at: datetime


class GenericAgentItemReviewMaterialSource(Protocol):
    def criteria(
        self,
        item_id: str,
    ) -> GenericAgentCriteriaReviewMaterial: ...

    def grading(
        self,
        item_id: str,
    ) -> GenericAgentGradingReviewMaterial: ...

    def batch_source(
        self,
        item_id: str,
    ) -> BatchQualityItemSource: ...


class GenericAgentDeliveryReviewPreparation(Protocol):
    def prepare_review(
        self,
        *,
        audit: ContractAudit,
    ) -> object: ...


class GenericAgentPlanReviewPreparation:
    """Creates first-party reviewable plans inside the PlanReview capability."""

    def __init__(
        self,
        *,
        dataset_run_id: str,
        store: FactoryControlStore,
        policy: FactoryRunPolicyV2,
        task_authoring_materials: FactoryTaskAuthoringMaterialStore,
        attachment_runtime: FactoryAttachmentItemRuntime,
        criteria_runtime: FactoryCriteriaItemRuntime | None = None,
        grading_runtime: FactoryGradingItemRuntime | None = None,
        item_materials: GenericAgentItemReviewMaterialSource | None = None,
        delivery_preparation: (GenericAgentDeliveryReviewPreparation | None) = None,
    ) -> None:
        self.dataset_run_id = dataset_run_id
        self.store = store
        self.policy = policy
        self.task_authoring_materials = task_authoring_materials
        self.attachment_runtime = attachment_runtime
        self.criteria_runtime = criteria_runtime
        self.grading_runtime = grading_runtime
        self.item_materials = item_materials
        self.delivery_preparation = delivery_preparation
        if (criteria_runtime is None) != (item_materials is None):
            raise ValueError(
                "criteria review runtime and item materials must appear together",
            )
        if grading_runtime is not None and item_materials is None:
            raise ValueError(
                "grading review runtime requires item materials",
            )

    async def prepare(
        self,
        request: PlanReviewCapabilityRequestV1,
        *,
        audit: ContractAudit,
    ) -> GenericAgentPreparedReview:
        if request.plan_kind is PlanKindV2.GLOBAL_BUILD:
            return GenericAgentPreparedReview(
                run_id=request.run_id,
                plan_kind=request.plan_kind,
                plan_ref=request.plan_ref,
            )
        if request.plan_kind is PlanKindV2.FINAL_DELIVERY:
            if self.delivery_preparation is None:
                raise GenericAgentPlanReviewPreparationError(
                    "Final Delivery PlanReview preparation is unavailable",
                )
            aggregate = self.store.get_dataset_aggregate(
                self.dataset_run_id,
            )
            try:
                current_plan = self.store.get_domain_plan(
                    self.dataset_run_id,
                    PlanKindV2.FINAL_DELIVERY,
                ).plan_ref
            except FactoryControlNotFoundError:
                current_plan = None
            if request.plan_ref not in {
                aggregate.to_ref(),
                current_plan,
            }:
                raise GenericAgentPlanReviewPreparationError(
                    "Final Delivery request differs from current authority",
                )
            view = self.delivery_preparation.prepare_review(
                audit=audit,
            )
            plan_ref = getattr(view, "plan_ref", None)
            if not isinstance(plan_ref, ObjectRef):
                raise GenericAgentPlanReviewPreparationError(
                    "Final Delivery preparation returned no plan",
                )
            return GenericAgentPreparedReview(
                run_id=request.run_id,
                plan_kind=request.plan_kind,
                plan_ref=plan_ref,
            )
        if request.plan_kind not in {
            PlanKindV2.ATTACHMENT_GENERATION,
            PlanKindV2.CRITERIA_RUBRIC,
            PlanKindV2.GRADING_DESIGN,
        }:
            raise GenericAgentPlanReviewPreparationError(
                "domain PlanReview preparation is unavailable",
            )
        run = self.store.get_run(request.run_id)
        matches = tuple(
            binding
            for binding in self.store.list_item_bindings(
                self.dataset_run_id,
            )
            if self.store.get_item_run(binding).run_id == run.run_id
        )
        if len(matches) != 1:
            raise GenericAgentPlanReviewPreparationError(
                "PlanReview child binding is not unique",
            )
        binding = matches[0]
        predecessor_stage = {
            PlanKindV2.ATTACHMENT_GENERATION: (FactoryItemStageV2.TASK_AUTHORING),
            PlanKindV2.CRITERIA_RUBRIC: FactoryItemStageV2.ITEM_QUALITY,
            PlanKindV2.GRADING_DESIGN: FactoryItemStageV2.CRITERIA_RUBRIC,
        }[request.plan_kind]
        predecessor_head = self.store.get_item_stage_head(
            binding.item_id,
            predecessor_stage,
        )
        try:
            current_plan = self.store.get_domain_plan(
                request.run_id,
                request.plan_kind,
            ).plan_ref
        except FactoryControlNotFoundError:
            current_plan = None
        if request.plan_ref not in {
            predecessor_head.result_ref,
            current_plan,
        }:
            raise GenericAgentPlanReviewPreparationError(
                "PlanReview request differs from current review subject",
            )
        if request.plan_kind is PlanKindV2.ATTACHMENT_GENERATION:
            material_ref = self.store.get_item_stage_material_ref(
                predecessor_head.to_ref(),
            )
            task_authoring = self.task_authoring_materials.get(
                material_ref,
            )
            attachment_view = await self.attachment_runtime.prepare_review(
                binding=binding,
                task_authoring=task_authoring,
                policy=self.policy,
                audit=audit,
            )
            prepared_plan_ref = attachment_view.plan_ref
        elif request.plan_kind is PlanKindV2.CRITERIA_RUBRIC:
            if self.criteria_runtime is None or self.item_materials is None:
                raise GenericAgentPlanReviewPreparationError(
                    "Criteria PlanReview preparation is unavailable",
                )
            criteria_material = self.item_materials.criteria(binding.item_id)
            criteria_view = await self.criteria_runtime.prepare_review(
                binding=binding,
                policy=self.policy,
                task_draft=criteria_material.task_draft,
                task_contract_set=criteria_material.task_contract_set,
                quality=criteria_material.quality,
                quality_context=criteria_material.quality_context,
                binding_definitions=(criteria_material.binding_definitions),
                tool_catalog=criteria_material.tool_catalog,
                audit=audit,
            )
            prepared_plan_ref = criteria_view.plan_ref
        else:
            if self.grading_runtime is None or self.item_materials is None:
                raise GenericAgentPlanReviewPreparationError(
                    "Grading PlanReview preparation is unavailable",
                )
            grading_material = self.item_materials.grading(binding.item_id)
            grading_view = await self.grading_runtime.prepare_review(
                binding=binding,
                policy=self.policy,
                criteria=grading_material.criteria,
                model_authorizations=(grading_material.model_authorizations),
                evaluated_at=grading_material.evaluated_at,
                audit=audit,
            )
            prepared_plan_ref = grading_view.plan_ref
        return GenericAgentPreparedReview(
            run_id=request.run_id,
            plan_kind=request.plan_kind,
            plan_ref=prepared_plan_ref,
        )


__all__ = [
    "GenericAgentCriteriaReviewMaterial",
    "GenericAgentDeliveryReviewPreparation",
    "GenericAgentGradingReviewMaterial",
    "GenericAgentItemReviewMaterialSource",
    "GenericAgentPlanReviewPreparation",
    "GenericAgentPlanReviewPreparationError",
    "GenericAgentPreparedReview",
]
