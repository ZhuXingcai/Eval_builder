from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from attachment_r5_fixtures import build_attachment_r5_preparation
from test_artifact_execution import (
    _ControlledExecutionFacade,
)
from test_artifact_results import (
    _result_audit,
)
from test_attachment_subgraph import _attachment_plan
from test_attachment_supervisor import (
    _registry,
)
from test_domain_plan_review import (
    RUN_ID,
    USER,
)
from test_domain_plan_review import (
    _setup as _domain_plan_setup,
)

from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
)
from eval_factory.agent_system.attachment_subgraph import (
    AttachmentRunnerFaultPoint,
    AttachmentRunnerInjectedCrash,
    AttachmentSubgraphError,
    StaticAttachmentRunnerFaultInjector,
    SupervisedAttachmentR5Runner,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
)
from eval_factory.agent_system.supervisor import ExecutionSupervisor
from eval_factory.agent_system.workspace import AgentWorkspaceManager
from eval_factory.contracts.agent_system_v2 import (
    AgentTaskOutcomeV2,
    AttachmentGenerationPlanV2,
    AttachmentSubgraphOutcomeV2,
    AttachmentSubgraphResultV2,
    PlanDecisionKindV2,
    PlanKindV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.task_v2 import producer_task_view_ref


def _current_plan(
    run_ref: ObjectRef,
    source: AttachmentGenerationPlanV2,
    producer_ref: ObjectRef,
) -> AttachmentGenerationPlanV2:
    return AttachmentGenerationPlanV2.create(
        plan_id="attachment-generation-plan://supervised-r5",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        producer_task_view_ref=producer_ref,
        evidence_bundle_ref=source.evidence_bundle_ref,
        attachment_planning_context_ref=(source.attachment_planning_context_ref),
        works=source.works,
        max_parallel_groups=source.max_parallel_groups,
        quality_policy_ref=source.quality_policy_ref,
        solvability_policy_ref=source.solvability_policy_ref,
        total_model_requests=source.total_model_requests,
        total_model_tokens=source.total_model_tokens,
        total_cost_micro_usd=source.total_cost_micro_usd,
        audit=_result_audit(),
    )


async def _run(
    tmp_path: Path,
    *,
    fail_first: bool = False,
):
    kwargs, facade, supervisor, producer_view = await _prepared(
        tmp_path,
        fail_first=fail_first,
    )
    result = await SupervisedAttachmentR5Runner(supervisor=supervisor).run(**kwargs)
    return result, facade, supervisor, producer_view


async def _prepared(
    tmp_path: Path,
    *,
    fail_first: bool = False,
    reviewed: bool = True,
):
    preparation = await build_attachment_r5_preparation(tmp_path / "r5")
    producer_view = preparation.producer_task_view
    producer_ref = producer_task_view_ref(producer_view)
    execution_plan = preparation.execution_result.execution_plan
    assert execution_plan is not None
    review_service, store, _ = _domain_plan_setup(tmp_path / "control")
    current = store.get_run(RUN_ID)
    registry = _registry()
    plan = _current_plan(
        current.to_ref(),
        _attachment_plan(execution_plan),
        producer_ref,
    )
    policy = store.get_policy(current.policy_ref.object_id)
    compiled = AttachmentGenerationPlanCompiler(registry).compile(
        plan=plan,
        policy=policy,
        audit=_result_audit(),
    )
    store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=current.run_version,
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        plan=plan,
        compiled_plan=compiled,
        audit=_result_audit(),
        idempotency_key="commit-supervised-attachment-plan",
    )
    if reviewed:
        opened = review_service.open_plan(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            requested_by=USER,
            idempotency_key="open-supervised-attachment-plan",
            audit=_result_audit(),
        )
        approved = review_service.decide(
            opened.request.review_request_id,
            PlanReviewDecisionSubmissionV1(
                expected_plan_version=plan.plan_version,
                decision=PlanDecisionKindV2.APPROVE,
                decided_by=USER,
                reason_code="ATTACHMENT_PLAN_APPROVED",
                idempotency_key="approve-supervised-attachment-plan",
            ),
            audit=_result_audit(),
        )
        resumed = review_service.resume(
            opened.request.review_request_id,
            PlanReviewResumeSubmissionV1(
                expected_plan_version=plan.plan_version,
                resumed_by=USER,
                idempotency_key="resume-supervised-attachment-plan",
            ),
            audit=_result_audit(),
        )
        assert approved.result.state is PlanReviewStateV2.APPROVED
        assert resumed.result.state is PlanReviewStateV2.RESUMED
    supervisor = ExecutionSupervisor(
        store,
        registry,
        workspace_manager=AgentWorkspaceManager(tmp_path / "agent-workspaces"),
    )
    fail_once = frozenset({execution_plan.routed_artifact_ids[0]}) if fail_first else frozenset()
    facade = _ControlledExecutionFacade(
        fail_once=fail_once,
        expected_parallelism=2,
    )
    return (
        {
            "run_id": RUN_ID,
            "plan": plan,
            "compiled_plan": compiled,
            "preparation": preparation,
            "facade": facade,
            "audit": _result_audit(),
        },
        facade,
        supervisor,
        producer_view,
    )


@pytest.mark.asyncio
async def test_supervised_r5_runs_two_groups_in_isolated_workspaces(
    tmp_path: Path,
) -> None:
    result, facade, supervisor, _ = await _run(tmp_path)

    assert facade.max_active == 2
    assert len(result.group_results) == 2
    assert all(group.outcome is AgentTaskOutcomeV2.SUCCEEDED for group in result.group_results)
    assert all(envelope.outcome is AgentTaskOutcomeV2.SUCCEEDED for envelope in result.envelopes)
    assert result.subgraph_result.outcome is AttachmentSubgraphOutcomeV2.SUCCEEDED
    receipts = tuple(
        supervisor.workspace_manager.resolve(envelope.workspace_receipt_ref) for envelope in result.envelopes
    )
    assert len({receipt.namespace_sha256 for receipt in receipts}) == 2
    assert all(receipt.state.value == "COMMITTED" for receipt in receipts)
    current_result = supervisor.store.get_domain_result(
        RUN_ID,
        PlanKindV2.ATTACHMENT_GENERATION,
    )
    assert (
        AttachmentSubgraphResultV2.model_validate_json(
            current_result.result_record_json,
        )
        == result.subgraph_result
    )
    with sqlite3.connect(supervisor.store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM agent_results").fetchone() == (2,)
        assert connection.execute(
            """
            SELECT COUNT(*) FROM factory_control_outbox
            WHERE event_type = 'agent-result-completed'
            """
        ).fetchone() == (2,)


@pytest.mark.asyncio
async def test_supervised_r5_isolates_retryable_group_failure(
    tmp_path: Path,
) -> None:
    result, facade, _, _ = await _run(
        tmp_path,
        fail_first=True,
    )

    assert facade.max_active == 2
    assert {group.outcome for group in result.group_results} == {
        AgentTaskOutcomeV2.RETRYABLE_FAILURE,
        AgentTaskOutcomeV2.SUCCEEDED,
    }
    assert result.subgraph_result.outcome is AttachmentSubgraphOutcomeV2.PARTIAL
    assert len(result.subgraph_result.retryable_work_result_refs) == 1
    assert len(result.subgraph_result.succeeded_work_result_refs) == 1


@pytest.mark.asyncio
async def test_supervised_r5_requires_resumed_review_before_side_effects(
    tmp_path: Path,
) -> None:
    kwargs, facade, supervisor, _ = await _prepared(
        tmp_path,
        reviewed=False,
    )
    before_database = _authority_counts(supervisor.store.path)
    before_files = _physical_files(supervisor.workspace_manager.root)

    with pytest.raises(
        AttachmentSubgraphError,
        match="resumed plan authority",
    ):
        await SupervisedAttachmentR5Runner(
            supervisor=supervisor,
        ).run(**kwargs)

    assert not facade.calls
    assert _authority_counts(supervisor.store.path) == before_database
    assert _physical_files(supervisor.workspace_manager.root) == before_files


@pytest.mark.asyncio
async def test_supervised_r5_rejects_stale_preparation_before_side_effects(
    tmp_path: Path,
) -> None:
    kwargs, facade, supervisor, _ = await _prepared(tmp_path)
    preparation = kwargs["preparation"]
    kwargs["preparation"] = replace(
        preparation,
        routing_result=preparation.routing_result.model_copy(
            update={"result_sha256": "f" * 64},
        ),
    )
    before_database = _authority_counts(supervisor.store.path)
    before_files = _physical_files(supervisor.workspace_manager.root)

    with pytest.raises(
        AttachmentSubgraphError,
        match="preparation authority",
    ):
        await SupervisedAttachmentR5Runner(supervisor=supervisor).run(**kwargs)

    assert not facade.calls
    assert _authority_counts(supervisor.store.path) == before_database
    assert _physical_files(supervisor.workspace_manager.root) == before_files


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_point",
    tuple(AttachmentRunnerFaultPoint),
)
async def test_supervised_r5_crash_recovery_and_exact_replay(
    tmp_path: Path,
    fault_point: AttachmentRunnerFaultPoint,
) -> None:
    kwargs, facade, supervisor, _ = await _prepared(tmp_path)
    crashing = SupervisedAttachmentR5Runner(
        supervisor=supervisor,
        fault_injector=StaticAttachmentRunnerFaultInjector(crash_points=frozenset({fault_point})),
    )

    with pytest.raises(
        AttachmentRunnerInjectedCrash,
        match=fault_point.value,
    ):
        await crashing.run(**kwargs)

    recovered = await SupervisedAttachmentR5Runner(supervisor=supervisor).run(**kwargs)
    provider_calls = facade.calls.copy()
    files = _physical_files(supervisor.workspace_manager.root)
    database_counts = _authority_counts(supervisor.store.path)
    workspace_refs = tuple(envelope.workspace_receipt_ref for envelope in recovered.envelopes)

    replay = await SupervisedAttachmentR5Runner(supervisor=supervisor).run(**kwargs)

    assert replay == recovered
    assert facade.calls == provider_calls
    assert _physical_files(supervisor.workspace_manager.root) == files
    assert _authority_counts(supervisor.store.path) == database_counts
    assert tuple(envelope.workspace_receipt_ref for envelope in replay.envelopes) == workspace_refs


def _physical_files(root: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _authority_counts(path: Path) -> tuple[int, ...]:
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "agent_results",
                "agent_work_events",
                "agent_work_current_heads",
                "domain_results",
                "domain_result_current_heads",
                "domain_result_idempotency",
                "factory_control_outbox",
            )
        )
