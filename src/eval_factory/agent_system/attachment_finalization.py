from __future__ import annotations

from dataclasses import dataclass

from env_mock_agent.facade.semantic_review_v2 import (
    AttachmentSemanticReviewFacade,
)
from env_mock_agent.facade.validation_v2 import (
    AttachmentValidationFacade,
)
from eval_factory.agent_system.attachment_quality import (
    AttachmentQualityAgent,
)
from eval_factory.agent_system.attachment_subgraph import (
    AttachmentSupervisedExecution,
)
from eval_factory.agent_system.solvability import (
    GatewaySolvabilityAgent,
    SolvabilityAgentResult,
)
from eval_factory.attachment_planning import (
    CandidateRevisionCompiler,
    CompleteRevisionValidator,
    DeterministicValidationCompiler,
    IsolatedSemanticReviewOrchestrator,
    ItemQualityCompiler,
    SemanticReviewContextSources,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityAssessmentV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
)
from eval_factory.contracts.review_v2 import (
    AttachmentCandidateRevisionV2,
    RevisionDeterministicValidationV2,
    SemanticReviewPolicyV2,
    SemanticReviewWorkflowResultV2,
)
from eval_factory.contracts.task_v2 import (
    ProducerTaskViewV2,
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
)
from eval_factory.contracts.validation_v2 import (
    DeterministicItemValidationResultV2,
)
from eval_factory.orchestration.job_store import JobStore
from eval_factory.provenance.redaction import ConfiguredPiiRule


@dataclass(frozen=True, slots=True)
class AttachmentR5FinalizationResult:
    source_validation: DeterministicItemValidationResultV2
    candidate_revision: AttachmentCandidateRevisionV2
    deterministic_validation: RevisionDeterministicValidationV2
    semantic_workflow: SemanticReviewWorkflowResultV2
    item_quality: ItemQualityCompilationResultV2
    attachment_quality: AttachmentQualityAssessmentV2


class AttachmentR5FinalizationPipeline:
    async def finalize(
        self,
        *,
        supervised_execution: AttachmentSupervisedExecution,
        producer_task_view: ProducerTaskViewV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        configured_pii_rules: tuple[ConfiguredPiiRule, ...],
        validation_facade: AttachmentValidationFacade,
        task_contract_set: R4TaskContractSetV2,
        review_policy: SemanticReviewPolicyV2,
        context_sources: SemanticReviewContextSources,
        review_facade: AttachmentSemanticReviewFacade,
        job_store: JobStore,
        job_id: str,
        item_id: str,
        audit: ContractAudit,
    ) -> AttachmentR5FinalizationResult:
        source_validation = await DeterministicValidationCompiler().compile(
            reconstruction_result=(supervised_execution.r5_execution.reconstruction_result),
            producer_task_view=producer_task_view,
            leakage_reference_set=leakage_reference_set,
            configured_pii_rules=configured_pii_rules,
            facade=validation_facade,
            audit=audit,
        )
        candidate_revision, deterministic_validation = CandidateRevisionCompiler().compile_initial(
            reconstruction_result=(supervised_execution.r5_execution.reconstruction_result),
            deterministic_validation=source_validation,
            audit=audit,
        )
        revision_validator = CompleteRevisionValidator(
            producer_task_view=producer_task_view,
            leakage_reference_set=leakage_reference_set,
            configured_pii_rules=configured_pii_rules,
            facade=validation_facade,
            audit=audit,
        )
        semantic_workflow = await IsolatedSemanticReviewOrchestrator().run(
            job_store=job_store,
            job_id=job_id,
            item_id=item_id,
            candidate_revision=candidate_revision,
            deterministic_validation=deterministic_validation,
            review_policy=review_policy,
            context_sources=context_sources,
            review_facade=review_facade,
            revision_validator=revision_validator,
            audit=audit,
        )
        item_quality = ItemQualityCompiler().compile(
            task_contract_set=task_contract_set,
            review_policy=review_policy,
            semantic_workflow=semantic_workflow,
            candidate_revision=candidate_revision,
            deterministic_validation=deterministic_validation,
            source_deterministic_validation=source_validation,
            job_store=job_store,
            job_id=job_id,
            item_id=item_id,
            audit=audit,
        )
        attachment_quality = AttachmentQualityAgent().assess(
            subgraph_result=(supervised_execution.subgraph_result),
            group_results=supervised_execution.group_results,
            work_envelopes=supervised_execution.envelopes,
            source_validation=source_validation,
            item_quality=item_quality,
            audit=audit,
        )
        return AttachmentR5FinalizationResult(
            source_validation=source_validation,
            candidate_revision=candidate_revision,
            deterministic_validation=deterministic_validation,
            semantic_workflow=semantic_workflow,
            item_quality=item_quality,
            attachment_quality=attachment_quality,
        )

    @staticmethod
    async def assess_solvability(
        *,
        supervised_execution: AttachmentSupervisedExecution,
        finalization: AttachmentR5FinalizationResult,
        agent: GatewaySolvabilityAgent,
        task_ref: ObjectRef,
        evidence_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> SolvabilityAgentResult:
        return await agent.assess(
            task_ref=task_ref,
            subgraph_result=(supervised_execution.subgraph_result),
            quality=finalization.attachment_quality,
            evidence_refs=evidence_refs,
            audit=audit,
        )


__all__ = [
    "AttachmentR5FinalizationPipeline",
    "AttachmentR5FinalizationResult",
]
