from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from eval_factory.agent_system.attachment_item_runtime import (
    FactoryAttachmentItemState,
    FactoryAttachmentItemView,
)
from eval_factory.agent_system.attachment_quality_runtime import (
    FactoryAttachmentQualityContext,
    FactoryAttachmentQualityRuntime,
    FactoryAttachmentQualityView,
)
from eval_factory.agent_system.criteria_item_runtime import (
    FactoryCriteriaItemRuntime,
    FactoryCriteriaItemState,
    FactoryCriteriaItemView,
)
from eval_factory.agent_system.grading_item_runtime import (
    FactoryGradingItemRuntime,
    FactoryGradingItemState,
    FactoryGradingItemView,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import FactoryRunPolicyV2
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemRunBindingV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.task_v2 import (
    ProducerTaskViewV2,
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
    TaskDraftV2,
)
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    ToolCapabilityCatalog,
    VerifiedModelDomainAuthorization,
)


class FactoryItemSpecialistRuntimeError(RuntimeError):
    pass


class FactoryItemSpecialistState(StrEnum):
    WAITING_CRITERIA_REVIEW = "WAITING_CRITERIA_REVIEW"
    READY_FOR_GRADING = "READY_FOR_GRADING"
    WAITING_GRADING_REVIEW = "WAITING_GRADING_REVIEW"
    EXECUTED = "EXECUTED"
    BLOCKED_QUALITY = "BLOCKED_QUALITY"
    BLOCKED_CRITERIA = "BLOCKED_CRITERIA"
    BLOCKED_GRADING = "BLOCKED_GRADING"


@dataclass(frozen=True, slots=True)
class FactoryItemSpecialistContext:
    quality: FactoryAttachmentQualityContext
    evaluator_bindings: tuple[EvaluatorBindingDefinition, ...]
    tool_catalog: ToolCapabilityCatalog
    model_authorizations: tuple[
        VerifiedModelDomainAuthorization,
        ...,
    ]
    evaluated_at: datetime


@dataclass(frozen=True, slots=True)
class FactoryItemSpecialistInput:
    producer_task_view: ProducerTaskViewV2
    leakage_reference_set: PromptLeakageReferenceSetV2
    task_contract_set: R4TaskContractSetV2
    task_draft: TaskDraftV2


@dataclass(frozen=True, slots=True)
class FactoryItemSpecialistView:
    item_id: str
    state: FactoryItemSpecialistState
    pending_review_refs: tuple[ObjectRef, ...]
    quality: FactoryAttachmentQualityView
    criteria: FactoryCriteriaItemView | None
    grading: FactoryGradingItemView | None


class FactoryItemSpecialistRuntime:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        quality_runtime: FactoryAttachmentQualityRuntime,
        criteria_runtime: FactoryCriteriaItemRuntime,
        grading_runtime: FactoryGradingItemRuntime | None = None,
    ) -> None:
        self.store = store
        self.quality_runtime = quality_runtime
        self.criteria_runtime = criteria_runtime
        self.grading_runtime = grading_runtime

    async def advance(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        source: FactoryItemSpecialistInput,
        attachment: FactoryAttachmentItemView,
        policy: FactoryRunPolicyV2,
        context: FactoryItemSpecialistContext,
        audit: ContractAudit,
    ) -> FactoryItemSpecialistView:
        if (
            attachment.state is not FactoryAttachmentItemState.EXECUTED
            or attachment.stage_head_ref is None
            or attachment.supervised_execution is None
        ):
            raise FactoryItemSpecialistRuntimeError(
                "specialist runtime requires executed attachment authority"
            )
        quality = await self.quality_runtime.advance(
            binding=binding,
            producer_task_view=source.producer_task_view,
            leakage_reference_set=source.leakage_reference_set,
            task_contract_set=source.task_contract_set,
            supervised_execution=attachment.supervised_execution,
            context=context.quality,
            audit=audit,
        )
        quality_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ITEM_QUALITY,
        )
        if (
            quality_head.to_ref() != quality.item_quality_head_ref
            or quality_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
        ):
            return FactoryItemSpecialistView(
                item_id=binding.item_id,
                state=FactoryItemSpecialistState.BLOCKED_QUALITY,
                pending_review_refs=(),
                quality=quality,
                criteria=None,
                grading=None,
            )
        criteria = await self.criteria_runtime.advance(
            binding=binding,
            policy=policy,
            task_draft=source.task_draft,
            task_contract_set=source.task_contract_set,
            quality=quality,
            quality_context=context.quality,
            binding_definitions=context.evaluator_bindings,
            tool_catalog=context.tool_catalog,
            audit=audit,
        )
        if criteria.state is FactoryCriteriaItemState.WAITING_REVIEW:
            return FactoryItemSpecialistView(
                item_id=binding.item_id,
                state=(FactoryItemSpecialistState.WAITING_CRITERIA_REVIEW),
                pending_review_refs=(criteria.review_ref,),
                quality=quality,
                criteria=criteria,
                grading=None,
            )
        criteria_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.CRITERIA_RUBRIC,
        )
        if criteria_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED:
            return FactoryItemSpecialistView(
                item_id=binding.item_id,
                state=(FactoryItemSpecialistState.BLOCKED_CRITERIA),
                pending_review_refs=(),
                quality=quality,
                criteria=criteria,
                grading=None,
            )
        if self.grading_runtime is None:
            return FactoryItemSpecialistView(
                item_id=binding.item_id,
                state=FactoryItemSpecialistState.READY_FOR_GRADING,
                pending_review_refs=(),
                quality=quality,
                criteria=criteria,
                grading=None,
            )
        grading = await self.grading_runtime.advance(
            binding=binding,
            policy=policy,
            criteria=criteria,
            model_authorizations=context.model_authorizations,
            evaluated_at=context.evaluated_at,
            audit=audit,
        )
        if grading.state is FactoryGradingItemState.WAITING_REVIEW:
            return FactoryItemSpecialistView(
                item_id=binding.item_id,
                state=(FactoryItemSpecialistState.WAITING_GRADING_REVIEW),
                pending_review_refs=(grading.review_ref,),
                quality=quality,
                criteria=criteria,
                grading=grading,
            )
        grading_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.GRADING_DESIGN,
        )
        if grading_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED:
            return FactoryItemSpecialistView(
                item_id=binding.item_id,
                state=(FactoryItemSpecialistState.BLOCKED_GRADING),
                pending_review_refs=(),
                quality=quality,
                criteria=criteria,
                grading=grading,
            )
        return FactoryItemSpecialistView(
            item_id=binding.item_id,
            state=FactoryItemSpecialistState.EXECUTED,
            pending_review_refs=(),
            quality=quality,
            criteria=criteria,
            grading=grading,
        )


__all__ = [
    "FactoryItemSpecialistContext",
    "FactoryItemSpecialistInput",
    "FactoryItemSpecialistRuntime",
    "FactoryItemSpecialistRuntimeError",
    "FactoryItemSpecialistState",
    "FactoryItemSpecialistView",
]
