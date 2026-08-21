from __future__ import annotations

from pathlib import Path

import pytest
from test_item_quality import _audit, _current_sources

from eval_factory.agent_system.attachment_quality import (
    AttachmentQualityAgent,
    AttachmentQualityError,
)
from eval_factory.attachment_planning.quality import ItemQualityCompiler
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    AgentTaskOutcomeV2,
    AttachmentGroupResultV2,
    AttachmentQualityOutcomeV2,
    AttachmentSubgraphOutcomeV2,
    AttachmentSubgraphResultV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.validation_v2 import (
    DeterministicItemValidationResultV2,
)

HASH = "a" * 64


def _ref(object_type: str, suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v2",
        object_sha256=HASH,
    )


def _subgraph_authority(
    outcome: AttachmentSubgraphOutcomeV2,
    *,
    reconstruction_result_ref: ObjectRef,
    suffix: str = "current",
) -> tuple[
    AttachmentSubgraphResultV2,
    tuple[AttachmentGroupResultV2, ...],
    tuple[AgentResultEnvelopeV2, ...],
]:
    if outcome is AttachmentSubgraphOutcomeV2.SUCCEEDED:
        envelope_outcomes = (
            AgentTaskOutcomeV2.SUCCEEDED,
            AgentTaskOutcomeV2.SUCCEEDED,
        )
        reasons: tuple[str, ...] = ()
    elif outcome is AttachmentSubgraphOutcomeV2.PARTIAL:
        envelope_outcomes = (
            AgentTaskOutcomeV2.SUCCEEDED,
            AgentTaskOutcomeV2.RETRYABLE_FAILURE,
        )
        reasons = ("ATTACHMENT_WORK_INCOMPLETE",)
    else:
        envelope_outcomes = (
            AgentTaskOutcomeV2.BLOCKED,
            AgentTaskOutcomeV2.BLOCKED,
        )
        reasons = ("ATTACHMENT_WORK_BLOCKED",)
    plan_ref = _ref("attachment-generation-plan", suffix)
    group_results = tuple(
        AttachmentGroupResultV2.create(
            result_id=(f"attachment-group-result://{suffix}-{index}"),
            plan_ref=plan_ref,
            work_key=f"work-{index}",
            artifact_group_ref=_ref(
                "artifact-execution-group",
                f"{suffix}-{index}",
            ),
            execution_batch_ref=_ref(
                "artifact-execution-batch",
                suffix,
            ),
            reconstruction_result_ref=reconstruction_result_ref,
            execution_receipt_refs=(
                _ref(
                    "artifact-execution-receipt",
                    f"{suffix}-{index}",
                ),
            ),
            artifact_ids=(f"artifact://{suffix}-{index}",),
            outcome=envelope_outcome,
            reason_codes=(
                () if envelope_outcome is AgentTaskOutcomeV2.SUCCEEDED else ("ATTACHMENT_GROUP_INCOMPLETE",)
            ),
            audit=_audit(),
        )
        for index, envelope_outcome in enumerate(envelope_outcomes)
    )
    envelopes = tuple(
        AgentResultEnvelopeV2.create(
            result_id=(f"agent-result-envelope://{suffix}-{index}"),
            task_ref=_ref("agent-task", f"{suffix}-{index}"),
            agent_definition_ref=_ref(
                "agent-definition",
                f"{suffix}-{index}",
            ),
            attempt=1,
            lease_id=f"agent-lease://{suffix}-{index}",
            fencing_token=1,
            workspace_receipt_ref=_ref(
                "agent-workspace-receipt",
                f"{suffix}-{index}",
            ),
            outcome=envelope_outcome,
            output_refs=(group_result.to_ref(),),
            delegated_dataset_job_result_refs=(),
            gateway_receipt_refs=(),
            validator_result_refs=(group_result.execution_receipt_refs),
            failure_code=(
                None if envelope_outcome is AgentTaskOutcomeV2.SUCCEEDED else "ATTACHMENT_GROUP_INCOMPLETE"
            ),
            safe_metrics=(),
            audit=_audit(),
        )
        for index, (
            group_result,
            envelope_outcome,
        ) in enumerate(
            zip(
                group_results,
                envelope_outcomes,
                strict=True,
            )
        )
    )
    succeeded = tuple(
        envelope.to_ref() for envelope in envelopes if envelope.outcome is AgentTaskOutcomeV2.SUCCEEDED
    )
    retryable = tuple(
        envelope.to_ref()
        for envelope in envelopes
        if envelope.outcome is AgentTaskOutcomeV2.RETRYABLE_FAILURE
    )
    blocked = tuple(
        envelope.to_ref()
        for envelope in envelopes
        if envelope.outcome
        in {
            AgentTaskOutcomeV2.ABSTAINED,
            AgentTaskOutcomeV2.BLOCKED,
            AgentTaskOutcomeV2.FAILED,
        }
    )
    subgraph = AttachmentSubgraphResultV2.create(
        result_id=f"attachment-subgraph-result://{suffix}",
        plan_ref=plan_ref,
        work_result_refs=_sorted_refs(tuple(envelope.to_ref() for envelope in envelopes)),
        succeeded_work_result_refs=_sorted_refs(succeeded),
        retryable_work_result_refs=_sorted_refs(retryable),
        blocked_work_result_refs=_sorted_refs(blocked),
        cancelled_work_result_refs=(),
        outcome=outcome,
        reason_codes=reasons,
        audit=_audit(),
    )
    return subgraph, group_results, envelopes


def _sorted_refs(
    values: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )


def _subgraph_result(
    outcome: AttachmentSubgraphOutcomeV2,
    *,
    reconstruction_result_ref: ObjectRef,
    suffix: str = "current",
) -> AttachmentSubgraphResultV2:
    return _subgraph_authority(
        outcome,
        reconstruction_result_ref=reconstruction_result_ref,
        suffix=suffix,
    )[0]


async def _item_quality(
    tmp_path: Path,
    *,
    blocked: bool = False,
    repair: bool = False,
    rejected: bool = False,
):
    sources = await _current_sources(
        tmp_path,
        blocked=blocked,
        item_scoped_repair=repair,
        artifacts=((("artifact://input", "inputs/source.txt"),) if rejected else ()),
        validation_outcomes=({"artifact://input": "secret"} if rejected else None),
    )
    quality = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )
    return quality, sources[5]


def _assess(
    *,
    item_quality,
    source_validation: DeterministicItemValidationResultV2,
    outcome: AttachmentSubgraphOutcomeV2,
    suffix: str = "current",
):
    subgraph, group_results, envelopes = _subgraph_authority(
        outcome,
        reconstruction_result_ref=(source_validation.attachment_reconstruction_result_ref),
        suffix=suffix,
    )
    assessment = AttachmentQualityAgent().assess(
        subgraph_result=subgraph,
        group_results=group_results,
        work_envelopes=envelopes,
        source_validation=source_validation,
        item_quality=item_quality,
        audit=_audit(),
    )
    return assessment, subgraph


@pytest.mark.asyncio
async def test_attachment_quality_passes_current_successful_r5_authority(
    tmp_path: Path,
) -> None:
    item_quality, source_validation = await _item_quality(tmp_path)
    assessment, subgraph = _assess(
        item_quality=item_quality,
        source_validation=source_validation,
        outcome=AttachmentSubgraphOutcomeV2.SUCCEEDED,
    )

    assert assessment.outcome is AttachmentQualityOutcomeV2.PASSED
    assert assessment.nonwaivable is False
    assert assessment.reason_codes == ()
    assert assessment.attachment_subgraph_result_ref == subgraph.to_ref()
    assert set(assessment.validator_result_refs) == {
        item_quality.quality_report.deterministic_validation_ref,
        item_quality.quality_report.source_deterministic_validation_ref,
        *item_quality.quality_report.semantic_round_result_refs,
    }


@pytest.mark.asyncio
async def test_attachment_quality_preserves_repair_and_rejection_authority(
    tmp_path: Path,
) -> None:
    repair, repair_validation = await _item_quality(
        tmp_path / "repair",
        repair=True,
    )
    rejected, rejected_validation = await _item_quality(
        tmp_path / "rejected",
        rejected=True,
    )
    repair_assessment, _ = _assess(
        item_quality=repair,
        source_validation=repair_validation,
        outcome=AttachmentSubgraphOutcomeV2.SUCCEEDED,
        suffix="repair",
    )
    rejected_assessment, _ = _assess(
        item_quality=rejected,
        source_validation=rejected_validation,
        outcome=AttachmentSubgraphOutcomeV2.SUCCEEDED,
        suffix="rejected",
    )

    assert repair_assessment.outcome is AttachmentQualityOutcomeV2.REPAIR_REQUIRED
    assert repair_assessment.reason_codes == ("ITEM_QUALITY_REPAIR_REQUIRED",)
    assert rejected_assessment.outcome is AttachmentQualityOutcomeV2.REJECTED
    assert rejected_assessment.nonwaivable is True
    assert rejected_assessment.reason_codes == ("NONWAIVABLE_FINDING",)


@pytest.mark.asyncio
async def test_attachment_quality_preserves_blocked_authority(
    tmp_path: Path,
) -> None:
    item_quality, source_validation = await _item_quality(
        tmp_path,
        blocked=True,
    )
    assessment, _ = _assess(
        item_quality=item_quality,
        source_validation=source_validation,
        outcome=AttachmentSubgraphOutcomeV2.SUCCEEDED,
    )

    assert assessment.outcome is AttachmentQualityOutcomeV2.BLOCKED
    assert assessment.reason_codes == ("ITEM_QUALITY_BLOCKED",)


@pytest.mark.asyncio
async def test_attachment_quality_does_not_pass_incomplete_subgraph(
    tmp_path: Path,
) -> None:
    item_quality, source_validation = await _item_quality(tmp_path)
    assessment, _ = _assess(
        item_quality=item_quality,
        source_validation=source_validation,
        outcome=AttachmentSubgraphOutcomeV2.PARTIAL,
    )

    assert assessment.outcome is AttachmentQualityOutcomeV2.BLOCKED
    assert assessment.reason_codes == ("ATTACHMENT_SUBGRAPH_INCOMPLETE",)


@pytest.mark.asyncio
async def test_attachment_quality_rejects_cross_reconstruction_authority(
    tmp_path: Path,
) -> None:
    item_quality, source_validation = await _item_quality(tmp_path)
    subgraph, group_results, envelopes = _subgraph_authority(
        AttachmentSubgraphOutcomeV2.SUCCEEDED,
        reconstruction_result_ref=_ref(
            "attachment-reconstruction-result",
            "other",
        ),
    )

    with pytest.raises(
        AttachmentQualityError,
        match="current attachment reconstruction",
    ):
        AttachmentQualityAgent().assess(
            subgraph_result=subgraph,
            group_results=group_results,
            work_envelopes=envelopes,
            source_validation=source_validation,
            item_quality=item_quality,
            audit=_audit(),
        )
