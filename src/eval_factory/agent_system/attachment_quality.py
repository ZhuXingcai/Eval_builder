from __future__ import annotations

from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    AgentTaskOutcomeV2,
    AttachmentGroupResultV2,
    AttachmentQualityAssessmentV2,
    AttachmentQualityOutcomeV2,
    AttachmentSubgraphOutcomeV2,
    AttachmentSubgraphResultV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
    ItemQualityOutcomeV2,
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.validation_v2 import (
    DeterministicItemValidationResultV2,
    deterministic_item_validation_result_ref,
)


class AttachmentQualityError(RuntimeError):
    pass


class AttachmentQualityAgent:
    def assess(
        self,
        *,
        subgraph_result: AttachmentSubgraphResultV2,
        group_results: tuple[AttachmentGroupResultV2, ...],
        work_envelopes: tuple[AgentResultEnvelopeV2, ...],
        source_validation: DeterministicItemValidationResultV2,
        item_quality: ItemQualityCompilationResultV2,
        audit: ContractAudit,
    ) -> AttachmentQualityAssessmentV2:
        report = item_quality.quality_report
        self._validate_authority(
            subgraph_result=subgraph_result,
            group_results=group_results,
            work_envelopes=work_envelopes,
            source_validation=source_validation,
            item_quality=item_quality,
        )
        item_quality_ref = item_quality_compilation_result_ref(item_quality)
        nonwaivable = report.unresolved_non_waivable_count > 0
        reasons: tuple[str, ...]
        if nonwaivable:
            outcome = AttachmentQualityOutcomeV2.REJECTED
            reasons = ("NONWAIVABLE_FINDING",)
        elif subgraph_result.outcome is not AttachmentSubgraphOutcomeV2.SUCCEEDED:
            outcome = AttachmentQualityOutcomeV2.BLOCKED
            reasons = ("ATTACHMENT_SUBGRAPH_INCOMPLETE",)
        elif report.approvable:
            outcome = AttachmentQualityOutcomeV2.PASSED
            reasons = ()
        elif report.outcome is ItemQualityOutcomeV2.REQUIRES_REPAIR:
            outcome = AttachmentQualityOutcomeV2.REPAIR_REQUIRED
            reasons = ("ITEM_QUALITY_REPAIR_REQUIRED",)
        elif report.outcome is ItemQualityOutcomeV2.REJECTED:
            outcome = AttachmentQualityOutcomeV2.REJECTED
            reasons = ("ITEM_QUALITY_REJECTED",)
        else:
            outcome = AttachmentQualityOutcomeV2.BLOCKED
            reasons = ("ITEM_QUALITY_BLOCKED",)
        validator_refs = _sorted_refs(
            (
                report.deterministic_validation_ref,
                report.source_deterministic_validation_ref,
                *report.semantic_round_result_refs,
            )
        )
        if not validator_refs:
            raise AttachmentQualityError("item quality has no validator authority")
        return AttachmentQualityAssessmentV2.create(
            assessment_id=(f"attachment-quality-assessment://{subgraph_result.object_sha256}"),
            attachment_subgraph_result_ref=(subgraph_result.to_ref()),
            item_quality_result_ref=item_quality_ref,
            validator_result_refs=validator_refs,
            outcome=outcome,
            nonwaivable=nonwaivable,
            reason_codes=reasons,
            audit=audit,
        )

    @staticmethod
    def _validate_authority(
        *,
        subgraph_result: AttachmentSubgraphResultV2,
        group_results: tuple[AttachmentGroupResultV2, ...],
        work_envelopes: tuple[AgentResultEnvelopeV2, ...],
        source_validation: DeterministicItemValidationResultV2,
        item_quality: ItemQualityCompilationResultV2,
    ) -> None:
        if not group_results or len(group_results) != len(work_envelopes):
            raise AttachmentQualityError("attachment quality requires exact group authority")
        if {envelope.to_ref() for envelope in work_envelopes} != set(subgraph_result.work_result_refs):
            raise AttachmentQualityError("attachment quality envelopes differ from subgraph results")
        succeeded = {
            envelope.to_ref()
            for envelope in work_envelopes
            if envelope.outcome is AgentTaskOutcomeV2.SUCCEEDED
        }
        retryable = {
            envelope.to_ref() for envelope in work_envelopes if envelope.outcome.value == "RETRYABLE_FAILURE"
        }
        blocked = {
            envelope.to_ref()
            for envelope in work_envelopes
            if envelope.outcome.value in {"ABSTAINED", "BLOCKED", "FAILED"}
        }
        if (
            succeeded != set(subgraph_result.succeeded_work_result_refs)
            or retryable != set(subgraph_result.retryable_work_result_refs)
            or blocked != set(subgraph_result.blocked_work_result_refs)
        ):
            raise AttachmentQualityError("attachment quality envelope partitions drifted")
        group_refs = {group_result.to_ref() for group_result in group_results}
        output_refs = {reference for envelope in work_envelopes for reference in envelope.output_refs}
        if group_refs != output_refs:
            raise AttachmentQualityError("attachment quality group outputs differ from envelopes")
        source_ref = deterministic_item_validation_result_ref(source_validation)
        if item_quality.quality_report.source_deterministic_validation_ref != source_ref:
            raise AttachmentQualityError("item quality does not bind current deterministic validation")
        reconstruction_refs = {group_result.reconstruction_result_ref for group_result in group_results}
        if reconstruction_refs != {source_validation.attachment_reconstruction_result_ref}:
            raise AttachmentQualityError("item quality does not bind current attachment reconstruction")
        if any(group_result.plan_ref != subgraph_result.plan_ref for group_result in group_results):
            raise AttachmentQualityError("attachment group result belongs to another plan")


def _sorted_refs(
    values: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    unique = {
        (
            value.object_type,
            value.object_id,
            value.object_version,
            value.object_sha256,
        ): value
        for value in values
    }
    return tuple(unique[key] for key in sorted(unique))


__all__ = [
    "AttachmentQualityAgent",
    "AttachmentQualityError",
]
