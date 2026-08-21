from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from attachment_r5_fixtures import build_attachment_r5_preparation

from eval_factory.agent_system.attachment_preparation import (
    AttachmentR5PreparationError,
    AttachmentR5PreparationValidator,
)
from eval_factory.attachment_planning import (
    ApprovedSourceEvidenceOutcome,
    ArtifactEvidenceModeOutcome,
    PromptOnlyDependencyOutcome,
)
from eval_factory.contracts.core import ObjectRef


@pytest.mark.asyncio
async def test_preparation_validates_current_r5_authority_without_retrieval(
    tmp_path: Path,
) -> None:
    authority = await build_attachment_r5_preparation(tmp_path)

    AttachmentR5PreparationValidator().validate_current(authority)

    assert authority.source_evidence_result.outcome is (ApprovedSourceEvidenceOutcome.NOT_REQUIRED)
    assert authority.execution_result.execution_plan is not None
    assert len(authority.execution_result.execution_plan.groups) == 2


@pytest.mark.asyncio
async def test_preparation_validates_current_compiled_source_evidence(
    tmp_path: Path,
) -> None:
    authority = await build_attachment_r5_preparation(
        tmp_path,
        descriptions=("Use inputs/source.txt.",),
        include_source_evidence=True,
    )

    AttachmentR5PreparationValidator().validate_current(authority)

    assert authority.source_evidence_result.outcome is (ApprovedSourceEvidenceOutcome.COMPILED)
    assert authority.source_evidence_result.source_evidence_set is not None
    assert authority.routing_request.build_contracts[0].source_evidence_set_ref is not None


@pytest.mark.asyncio
async def test_preparation_rejects_stale_planning_context(
    tmp_path: Path,
) -> None:
    authority = await build_attachment_r5_preparation(tmp_path)
    stale = authority.attachment_planning_context.model_copy(
        update={"policy_version": "attachment-planning/r5-01-stale"},
    )

    with pytest.raises(
        AttachmentR5PreparationError,
        match="R5-01",
    ):
        AttachmentR5PreparationValidator().validate_current(
            replace(
                authority,
                attachment_planning_context=stale,
            )
        )


@pytest.mark.asyncio
async def test_preparation_rejects_non_executable_dependency_result(
    tmp_path: Path,
) -> None:
    authority = await build_attachment_r5_preparation(tmp_path)
    blocked = authority.dependency_result.model_copy(
        update={"outcome": PromptOnlyDependencyOutcome.ABSTAIN},
    )

    with pytest.raises(
        AttachmentR5PreparationError,
        match="R5-03",
    ):
        AttachmentR5PreparationValidator().validate_current(
            replace(
                authority,
                dependency_result=blocked,
            )
        )


@pytest.mark.asyncio
async def test_preparation_rejects_non_executable_evidence_matrix(
    tmp_path: Path,
) -> None:
    authority = await build_attachment_r5_preparation(tmp_path)
    blocked = authority.evidence_mode_result.model_copy(
        update={
            "outcome": ArtifactEvidenceModeOutcome.BLOCKED_CAPABILITY,
            "artifact_evidence_matrix": None,
        },
    )

    with pytest.raises(
        AttachmentR5PreparationError,
        match="R5-02",
    ):
        AttachmentR5PreparationValidator().validate_current(
            replace(
                authority,
                evidence_mode_result=blocked,
            )
        )


@pytest.mark.asyncio
async def test_preparation_rejects_stale_compiled_source_evidence(
    tmp_path: Path,
) -> None:
    authority = await build_attachment_r5_preparation(
        tmp_path,
        descriptions=("Use inputs/source.txt.",),
        include_source_evidence=True,
    )
    assert authority.retrieval_execution_result is not None
    execution = authority.retrieval_execution_result.executions[0]
    stale_fetch = execution.fetch_result.model_copy(
        update={"bytes_count": execution.fetch_result.bytes_count + 1},
    )
    stale_execution = execution.model_copy(
        update={"fetch_result": stale_fetch},
    )
    stale_result = authority.retrieval_execution_result.model_copy(
        update={"executions": (stale_execution,)},
    )

    with pytest.raises(
        AttachmentR5PreparationError,
        match="R5-04",
    ):
        AttachmentR5PreparationValidator().validate_current(
            replace(
                authority,
                retrieval_execution_result=stale_result,
            )
        )


@pytest.mark.asyncio
async def test_preparation_rejects_source_to_build_contract_drift(
    tmp_path: Path,
) -> None:
    authority = await build_attachment_r5_preparation(tmp_path)
    contract = authority.routing_request.build_contracts[0]
    changed_contract = contract.model_copy(
        update={
            "source_evidence_set_ref": ObjectRef(
                object_type="source-evidence-set",
                object_id="source-evidence-set://wrong",
                object_version="v2",
                object_sha256="f" * 64,
            ),
        },
    )
    changed_request = authority.routing_request.model_copy(
        update={
            "build_contracts": (
                changed_contract,
                *authority.routing_request.build_contracts[1:],
            ),
        },
    )

    with pytest.raises(
        AttachmentR5PreparationError,
        match="R5-04",
    ):
        AttachmentR5PreparationValidator().validate_current(
            replace(
                authority,
                routing_request=changed_request,
            )
        )


@pytest.mark.asyncio
async def test_preparation_rejects_stale_routing_result(
    tmp_path: Path,
) -> None:
    authority = await build_attachment_r5_preparation(tmp_path)
    stale = authority.routing_result.model_copy(
        update={"result_sha256": "f" * 64},
    )

    with pytest.raises(
        AttachmentR5PreparationError,
        match="R5-05",
    ):
        AttachmentR5PreparationValidator().validate_current(
            replace(
                authority,
                routing_result=stale,
            )
        )


@pytest.mark.asyncio
async def test_preparation_rejects_stale_execution_preparation(
    tmp_path: Path,
) -> None:
    authority = await build_attachment_r5_preparation(tmp_path)
    stale = authority.execution_preparation.model_copy(
        update={"definitions": ()},
    )

    with pytest.raises(
        AttachmentR5PreparationError,
        match="R5-06",
    ):
        AttachmentR5PreparationValidator().validate_current(
            replace(
                authority,
                execution_preparation=stale,
            )
        )
