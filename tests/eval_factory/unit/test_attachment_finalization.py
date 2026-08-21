from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from attachment_gateway_fixtures import ref
from test_artifact_results import _result_audit
from test_attachment_supervised_execution import _run
from test_deterministic_validation import (
    _FakeValidationFacade,
    _reference_set,
)
from test_item_quality import _task_contract_set
from test_semantic_review import (
    _AcceptingBackend,
    _DynamicResolver,
    _NoRepairBackend,
    _policy,
    _sources,
    _store,
)
from test_solvability_agent import (
    _agent as _solvability_agent,
)

from env_mock_agent.facade.semantic_review_adapter import (
    RegistryAttachmentSemanticReviewFacade,
)
from eval_factory.agent_system.attachment_finalization import (
    AttachmentR5FinalizationPipeline,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityOutcomeV2,
    SolvabilityOutcomeV2,
)
from eval_factory.contracts.quality_v2 import (
    ItemQualityOutcomeV2,
)
from eval_factory.contracts.review_v2 import (
    SemanticReviewWorkflowOutcomeV2,
)
from eval_factory.contracts.task_v2 import producer_task_view_ref


@pytest.mark.asyncio
async def test_attachment_finalization_composes_r5_quality_and_solvability(
    tmp_path: Path,
) -> None:
    supervised, _, supervisor, producer_view = await _run(tmp_path / "supervised")
    review_policy = _policy()
    backend = _AcceptingBackend()
    review_facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: backend for role in review_policy.roles_in_order},
        repair_backend=_NoRepairBackend(),
    )
    job_store, job_id, item_id = _store(tmp_path / "quality")
    pipeline = AttachmentR5FinalizationPipeline()

    finalization = await pipeline.finalize(
        supervised_execution=supervised,
        producer_task_view=producer_view,
        leakage_reference_set=_reference_set(),
        configured_pii_rules=(),
        validation_facade=_FakeValidationFacade(),
        task_contract_set=_task_contract_set(producer_task_view_ref(producer_view)),
        review_policy=review_policy,
        context_sources=_sources(),
        review_facade=review_facade,
        job_store=job_store,
        job_id=job_id,
        item_id=item_id,
        audit=_result_audit(),
    )

    assert finalization.semantic_workflow.outcome is SemanticReviewWorkflowOutcomeV2.PASSED
    assert len(finalization.semantic_workflow.round_results) == 3
    assert finalization.item_quality.quality_report.outcome is ItemQualityOutcomeV2.PASSED
    assert finalization.item_quality.quality_report.approvable is True
    assert finalization.attachment_quality.outcome is AttachmentQualityOutcomeV2.PASSED
    assert len(backend.context_ids) == 3

    evidence_refs = (finalization.attachment_quality.validator_result_refs[0],)
    solvability_agent, _, provider, generator, judge = _solvability_agent(
        tmp_path / "solvability",
        proposal_evidence_refs=evidence_refs,
    )
    solvability = await pipeline.assess_solvability(
        supervised_execution=supervised,
        finalization=finalization,
        agent=solvability_agent,
        task_ref=ref(
            "agent-task",
            "attachment-final-solvability",
        ),
        evidence_refs=evidence_refs,
        audit=_result_audit(),
    )

    assert solvability.assessment.outcome is SolvabilityOutcomeV2.SOLVABLE
    assert solvability.route.selected_model_profile_ref == judge.to_ref()
    assert solvability.route.selected_model_profile_ref != generator.to_ref()
    assert len(provider.calls) == 1
    with sqlite3.connect(supervisor.store.path) as connection:
        states = {row[0] for row in connection.execute("SELECT state FROM plan_review_results")}
    assert {"PENDING_REVIEW", "APPROVED", "RESUMED"} <= states
