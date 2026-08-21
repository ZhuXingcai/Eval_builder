from __future__ import annotations

import hashlib
from dataclasses import dataclass

from env_mock_agent.facade.semantic_review_v2 import (
    AttachmentSemanticReviewFacade,
)
from env_mock_agent.facade.validation_v2 import (
    AttachmentValidationFacade,
)
from eval_factory.agent_system.attachment_finalization import (
    AttachmentR5FinalizationPipeline,
)
from eval_factory.agent_system.attachment_quality_material import (
    FactoryAttachmentQualityMaterialStore,
    FactoryAttachmentQualityResult,
)
from eval_factory.agent_system.attachment_subgraph import (
    AttachmentSupervisedExecution,
)
from eval_factory.agent_system.solvability import (
    GatewaySolvabilityAgent,
)
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.attachment_planning import (
    ItemQualityCompiler,
    SemanticReviewContextSources,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityOutcomeV2,
    SolvabilityOutcomeV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.quality_v2 import (
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.review_v2 import (
    SemanticReviewPolicyV2,
)
from eval_factory.contracts.task_v2 import (
    ProducerTaskViewV2,
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
)
from eval_factory.orchestration.job_store import JobStore
from eval_factory.provenance.redaction import ConfiguredPiiRule


class FactoryAttachmentQualityRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryAttachmentQualityContext:
    job_store: JobStore
    job_id: str
    item_id: str
    configured_pii_rules: tuple[ConfiguredPiiRule, ...]
    validation_facade: AttachmentValidationFacade
    review_policy: SemanticReviewPolicyV2
    context_sources: SemanticReviewContextSources
    review_facade: AttachmentSemanticReviewFacade
    solvability_agent: GatewaySolvabilityAgent


@dataclass(frozen=True, slots=True)
class FactoryAttachmentQualityView:
    item_id: str
    item_quality_head_ref: ObjectRef
    material_ref: ObjectRef
    result: FactoryAttachmentQualityResult


class FactoryAttachmentQualityRuntime:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        materials: FactoryAttachmentQualityMaterialStore,
        pipeline: AttachmentR5FinalizationPipeline | None = None,
    ) -> None:
        self.store = store
        self.materials = materials
        self.pipeline = pipeline or AttachmentR5FinalizationPipeline()

    async def advance(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        producer_task_view: ProducerTaskViewV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        task_contract_set: R4TaskContractSetV2,
        supervised_execution: AttachmentSupervisedExecution,
        context: FactoryAttachmentQualityContext,
        audit: ContractAudit,
    ) -> FactoryAttachmentQualityView:
        attachment_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ATTACHMENT,
        )
        if (
            attachment_head.item_binding_ref != binding.to_ref()
            or attachment_head.result_ref != supervised_execution.subgraph_result.to_ref()
            or attachment_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
        ):
            raise FactoryAttachmentQualityRuntimeError(
                "attachment quality requires the current successful attachment stage"
            )
        self._validate_delegated_item(
            binding=binding,
            context=context,
        )
        try:
            quality_head = self.store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.ITEM_QUALITY,
            )
        except FactoryControlNotFoundError:
            finalization = await self.pipeline.finalize(
                supervised_execution=supervised_execution,
                producer_task_view=producer_task_view,
                leakage_reference_set=leakage_reference_set,
                configured_pii_rules=(context.configured_pii_rules),
                validation_facade=context.validation_facade,
                task_contract_set=task_contract_set,
                review_policy=context.review_policy,
                context_sources=context.context_sources,
                review_facade=context.review_facade,
                job_store=context.job_store,
                job_id=context.job_id,
                item_id=context.item_id,
                audit=audit,
            )
            evidence_refs = finalization.attachment_quality.validator_result_refs
            solvability = await self.pipeline.assess_solvability(
                supervised_execution=supervised_execution,
                finalization=finalization,
                agent=context.solvability_agent,
                task_ref=_stable_ref(
                    "agent-task",
                    f"{binding.object_id}:solvability",
                ),
                evidence_refs=evidence_refs,
                audit=audit,
            )
            result = FactoryAttachmentQualityResult(
                finalization=finalization,
                solvability=solvability,
            )
            material_ref = self.materials.put(result)
            outcome, reasons = _quality_stage_outcome(result)
            quality_head = FactoryItemStageHeadV2.create(
                item_binding_ref=binding.to_ref(),
                item_run_ref=binding.item_run_ref,
                stage=FactoryItemStageV2.ITEM_QUALITY,
                stage_version=1,
                predecessor_head_ref=None,
                dependency_result_refs=(attachment_head.result_ref,),
                result_ref=item_quality_compilation_result_ref(finalization.item_quality),
                outcome=outcome,
                reason_codes=reasons,
                audit=audit,
            )
            quality_head = self.store.commit_item_stage_head(
                quality_head,
                idempotency_key=(f"commit-factory-item-quality-head-{quality_head.object_sha256}"),
                material_ref=material_ref,
            )
        material_ref = self.store.get_item_stage_material_ref(quality_head.to_ref())
        result = self.materials.get(material_ref)
        self._validate_current(
            binding=binding,
            attachment_head=attachment_head,
            quality_head=quality_head,
            result=result,
            task_contract_set=task_contract_set,
            context=context,
        )
        return FactoryAttachmentQualityView(
            item_id=binding.item_id,
            item_quality_head_ref=quality_head.to_ref(),
            material_ref=material_ref,
            result=result,
        )

    @staticmethod
    def _validate_delegated_item(
        *,
        binding: FactoryItemRunBindingV2,
        context: FactoryAttachmentQualityContext,
    ) -> None:
        job = context.job_store.get_job(context.job_id)
        item = context.job_store.get_item(context.item_id)
        if item.item_id != binding.item_id or item.job_id != job.job_id:
            raise FactoryAttachmentQualityRuntimeError(
                "delegated JobStore item does not bind the factory item"
            )

    @staticmethod
    def _validate_current(
        *,
        binding: FactoryItemRunBindingV2,
        attachment_head: FactoryItemStageHeadV2,
        quality_head: FactoryItemStageHeadV2,
        result: FactoryAttachmentQualityResult,
        task_contract_set: R4TaskContractSetV2,
        context: FactoryAttachmentQualityContext,
    ) -> None:
        finalization = result.finalization
        if (
            quality_head.item_binding_ref != binding.to_ref()
            or quality_head.dependency_result_refs != (attachment_head.result_ref,)
            or quality_head.result_ref != item_quality_compilation_result_ref(finalization.item_quality)
            or finalization.attachment_quality.attachment_subgraph_result_ref != attachment_head.result_ref
            or result.solvability.assessment.attachment_subgraph_result_ref != attachment_head.result_ref
            or result.solvability.assessment.quality_assessment_ref
            != finalization.attachment_quality.to_ref()
        ):
            raise FactoryAttachmentQualityRuntimeError("attachment quality stage authority drifted")
        ItemQualityCompiler().validate_current(
            finalization.item_quality,
            task_contract_set=task_contract_set,
            review_policy=context.review_policy,
            semantic_workflow=finalization.semantic_workflow,
            candidate_revision=finalization.candidate_revision,
            deterministic_validation=(finalization.deterministic_validation),
            source_deterministic_validation=(finalization.source_validation),
            job_store=context.job_store,
            job_id=context.job_id,
            item_id=context.item_id,
        )
        if (
            context.solvability_agent.gateway.get_route(result.solvability.assessment.route_decision_ref)
            != result.solvability.route
        ):
            raise FactoryAttachmentQualityRuntimeError("solvability route authority drifted")


def _quality_stage_outcome(
    result: FactoryAttachmentQualityResult,
) -> tuple[
    FactoryItemStageOutcomeV2,
    tuple[str, ...],
]:
    finalization = result.finalization
    if (
        finalization.item_quality.quality_report.approvable
        and finalization.attachment_quality.outcome is AttachmentQualityOutcomeV2.PASSED
        and result.solvability.assessment.outcome is SolvabilityOutcomeV2.SOLVABLE
    ):
        return FactoryItemStageOutcomeV2.SUCCEEDED, ()
    if finalization.attachment_quality.outcome is AttachmentQualityOutcomeV2.REJECTED:
        return (
            FactoryItemStageOutcomeV2.REJECTED,
            finalization.attachment_quality.reason_codes,
        )
    reasons = tuple(
        sorted(
            {
                *finalization.attachment_quality.reason_codes,
                *result.solvability.assessment.reason_codes,
            }
        )
    )
    return (
        FactoryItemStageOutcomeV2.FAILED,
        reasons or ("ITEM_QUALITY_NOT_APPROVABLE",),
    )


def _stable_ref(
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
    "FactoryAttachmentQualityContext",
    "FactoryAttachmentQualityRuntime",
    "FactoryAttachmentQualityRuntimeError",
    "FactoryAttachmentQualityView",
]
