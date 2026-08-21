from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_job_store import _plan, _work_graph
from test_user_approval_requests import (
    _generation_policy,
    _job_spec,
)
from test_user_decisions import (
    _audit,
    _authentication,
    _handling_policy,
    _label_request,
)
from test_user_plan_revalidation import (
    _application_policy,
    _ref,
)

from eval_factory.approval.adjustments import LabelPlanAdjustmentCompiler
from eval_factory.approval.application_persistence import (
    UserPlanApplicationConflictError,
    UserPlanApplicationPersistenceService,
)
from eval_factory.approval.persistence import UserDecisionPersistenceService
from eval_factory.approval.revalidation import (
    RevalidationItemSource,
    RevalidationStageSource,
)
from eval_factory.contracts.approval import TypedAdjustment, UserDecision
from eval_factory.contracts.approval_application_v2 import (
    DirectedRevalidationReportOutcomeV2,
    UserPlanApplicationPolicyV2,
    revalidation_work_item_v2_ref,
)
from eval_factory.contracts.core import FailureClass, FailureRecord
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.orchestration.job_store import (
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "factory.sqlite3", clock=lambda: NOW)


def _prepared(
    tmp_path: Path,
) -> tuple[
    JobStore,
    UserPlanApplicationPersistenceService,
    str,
    object,
    tuple[RevalidationItemSource, ...],
    UserPlanApplicationPolicyV2,
]:
    store = _store(tmp_path)
    approval_policy, approval_source, request_result, request = _label_request()
    job_spec = _job_spec(approval_policy)
    store.create_planned_job(job_spec, _plan(job_spec))
    graph = _work_graph(job_spec)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-r7-06-work-graph",
    )
    adjustment = TypedAdjustment(
        target_path="label_plan.boundary",
        operation="SET",
        value="Use only current executed tool facts.",
        reason="Tighten the label boundary.",
    )
    producer = LabelPlanAdjustmentCompiler().compile(
        request=request,
        source_label_spec=approval_source.label_spec,
        source_label_plan=approval_source.label_plan,
        adjustments=(adjustment,),
        audit=_audit(),
    )
    commit = UserDecisionPersistenceService(store).commit(
        job_spec=job_spec,
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        request_compilation=request_result,
        checkpoint_sources=(approval_source,),
        request_ref=producer.effect.source_request_ref,
        handling_policy=_handling_policy(),
        authentication=_authentication(),
        decision=UserDecision.ADJUST,
        adjustments=(adjustment,),
        adjustment_effect=producer.effect,
        environment_decisions=(),
        query_packaging=None,
        reason="Apply the tightened label boundary.",
        idempotency_key="commit-r7-06-label-adjustment",
        audit=_audit(),
    )
    item_sources = (
        RevalidationItemSource(
            item_id=graph.item_ids[0],
            source_trace_ref=graph.source_trace_refs[0],
            authority_refs=(request.subject_refs[0],),
            stages=(
                RevalidationStageSource(
                    stage=StageNameV2.LABEL,
                    output_refs=(_ref("label-decision", "persisted-source"),),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.TASK_AUTHORING,
                    output_refs=(_ref("r4-task-contract-set", "persisted-source"),),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.ATTACHMENT,
                    output_refs=(
                        _ref(
                            "attachment-reconstruction-result",
                            "persisted-source",
                        ),
                    ),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.ITEM_QUALITY,
                    output_refs=(
                        _ref(
                            "item-quality-compilation-result",
                            "persisted-source",
                        ),
                    ),
                ),
            ),
        ),
    )
    return (
        store,
        UserPlanApplicationPersistenceService(store),
        commit.commit_id,
        producer.result,
        item_sources,
        _application_policy(),
    )


def _apply(
    service: UserPlanApplicationPersistenceService,
    *,
    commit_id: str,
    producer_result: object,
    item_sources: tuple[RevalidationItemSource, ...],
    policy: UserPlanApplicationPolicyV2,
    idempotency_key: str = "apply-r7-06-label-adjustment",
):
    return service.apply(
        decision_commit_id=commit_id,
        producer_result=producer_result,
        item_sources=item_sources,
        policy=policy,
        idempotency_key=idempotency_key,
        audit=_audit(),
    )


def _record_success(
    store: JobStore,
    service: UserPlanApplicationPersistenceService,
    *,
    application_id: str,
    work,
    policy: UserPlanApplicationPolicyV2,
    index: int,
):
    run = store.create_stage_run(
        stage_run_id=f"stage-run://r7-06/persistence/{index}",
        job_id="job://r7-04/requests",
        item_id=work.item_id,
        stage=work.stage,
        attempt=1,
        principal_ref=_ref("principal", f"persistence-{index}"),
        input_refs=(revalidation_work_item_v2_ref(work),),
        idempotency_key=f"create-r7-06-run-{index}",
    )
    running = store.transition_stage_run(
        run.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=run.row_version,
        idempotency_key=f"start-r7-06-run-{index}",
    )
    store.complete_stage_run(
        stage_result_id=f"stage-result://r7-06/persistence/{index}",
        stage_run_id=run.stage_run_id,
        status=StageRunStatus.SUCCEEDED,
        expected_version=running.row_version,
        idempotency_key=f"complete-r7-06-run-{index}",
        output_refs=(
            _ref(
                work.required_output_types[0],
                f"persistence-replacement-{index}",
            ),
        ),
    )
    return service.record_work_result(
        application_id=application_id,
        work_item_id=work.work_item_id,
        stage_run_id=run.stage_run_id,
        policy=policy,
        idempotency_key=f"record-r7-06-result-{index}",
        audit=_audit(),
    )


def test_apply_is_atomic_replay_safe_and_queryable(tmp_path: Path) -> None:
    (
        store,
        service,
        commit_id,
        producer_result,
        item_sources,
        policy,
    ) = _prepared(tmp_path)

    first = _apply(
        service,
        commit_id=commit_id,
        producer_result=producer_result,
        item_sources=item_sources,
        policy=policy,
    )
    replay = _apply(
        service,
        commit_id=commit_id,
        producer_result=producer_result,
        item_sources=item_sources,
        policy=policy,
    )

    assert replay == first
    assert service.get_application(first.application.application_id) == (first.application)
    assert service.get_plan(first.application.application_id) == first.plan
    assert service.list_ready_work(first.application.application_id) == (first.plan.work_items[0],)
    assert service.list_current_heads(first.application.job_id) == ()
    events = tuple(
        event for event in store.list_outbox() if event.event_type == "user-plan-application-created"
    )
    assert len(events) == 1


def test_apply_rejects_changed_replay_and_second_application(
    tmp_path: Path,
) -> None:
    (
        _,
        service,
        commit_id,
        producer_result,
        item_sources,
        policy,
    ) = _prepared(tmp_path)
    _apply(
        service,
        commit_id=commit_id,
        producer_result=producer_result,
        item_sources=item_sources,
        policy=policy,
    )
    changed_policy = _application_policy(max_work_items=99)

    with pytest.raises(IdempotencyConflictError, match="different request"):
        _apply(
            service,
            commit_id=commit_id,
            producer_result=producer_result,
            item_sources=item_sources,
            policy=changed_policy,
        )
    with pytest.raises(UserPlanApplicationConflictError, match="already"):
        _apply(
            service,
            commit_id=commit_id,
            producer_result=producer_result,
            item_sources=item_sources,
            policy=policy,
            idempotency_key="second-application-key",
        )


def test_results_advance_ready_work_and_complete_publishes_heads(
    tmp_path: Path,
) -> None:
    (
        store,
        service,
        commit_id,
        producer_result,
        item_sources,
        policy,
    ) = _prepared(tmp_path)
    compiled = _apply(
        service,
        commit_id=commit_id,
        producer_result=producer_result,
        item_sources=item_sources,
        policy=policy,
    )
    results = []
    for index, work in enumerate(compiled.plan.work_items):
        assert service.list_ready_work(compiled.application.application_id) == (work,)
        recorded = _record_success(
            store,
            service,
            application_id=compiled.application.application_id,
            work=work,
            policy=policy,
            index=index,
        )
        results.append(recorded)
        if index == 0:
            replay_result = service.record_work_result(
                application_id=compiled.application.application_id,
                work_item_id=work.work_item_id,
                stage_run_id=f"stage-run://r7-06/persistence/{index}",
                policy=policy,
                idempotency_key=f"record-r7-06-result-{index}",
                audit=_audit(),
            )
            assert replay_result == recorded
            with pytest.raises(
                UserPlanApplicationConflictError,
                match="already",
            ):
                service.record_work_result(
                    application_id=compiled.application.application_id,
                    work_item_id=work.work_item_id,
                    stage_run_id=(f"stage-run://r7-06/persistence/{index}"),
                    policy=policy,
                    idempotency_key="second-result-key",
                    audit=_audit(),
                )
            with pytest.raises(ImmutableResultError, match="policy"):
                service.record_work_result(
                    application_id=compiled.application.application_id,
                    work_item_id=work.work_item_id,
                    stage_run_id=(f"stage-run://r7-06/persistence/{index}"),
                    policy=_application_policy(max_work_items=99),
                    idempotency_key="changed-result-policy",
                    audit=_audit(),
                )
    assert service.list_ready_work(compiled.application.application_id) == ()

    report = service.complete(
        application_id=compiled.application.application_id,
        idempotency_key="complete-r7-06-application",
        audit=_audit(),
    )
    replay = service.complete(
        application_id=compiled.application.application_id,
        idempotency_key="complete-r7-06-application",
        audit=_audit(),
    )

    assert replay == report
    with pytest.raises(
        UserPlanApplicationConflictError,
        match="already",
    ):
        service.complete(
            application_id=compiled.application.application_id,
            idempotency_key="second-completion-key",
            audit=_audit(),
        )
    assert report.outcome is DirectedRevalidationReportOutcomeV2.COMPLETE
    assert service.get_report(compiled.application.application_id) == report
    assert service.get_work_item(compiled.plan.work_items[0].work_item_id) == compiled.plan.work_items[0]
    assert service.list_ready_work(compiled.application.application_id) == ()
    assert set(service.list_current_heads(compiled.application.job_id)) == set(
        report.replacement_current_refs
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            DELETE FROM object_current_heads
            WHERE job_id = ?
              AND object_type = ?
              AND object_id = ?
              AND object_version = ?
              AND object_sha256 = ?
            """,
            (
                compiled.application.job_id,
                report.replacement_current_refs[0].object_type,
                report.replacement_current_refs[0].object_id,
                report.replacement_current_refs[0].object_version,
                report.replacement_current_refs[0].object_sha256,
            ),
        )
    with pytest.raises(ImmutableResultError, match="current heads"):
        service.list_current_heads(compiled.application.job_id)
    assert set(service.rebuild_current_heads(compiled.application.job_id)) == set(
        report.replacement_current_refs
    )
    assert tuple(results) == tuple(
        service.get_work_result(work.work_item_id) for work in compiled.plan.work_items
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            DELETE FROM object_validity_events
            WHERE validity_event_id = (
                SELECT validity_event_id
                FROM object_validity_events
                WHERE job_id = ?
                ORDER BY validity_event_id
                LIMIT 1
            )
            """,
            (compiled.application.job_id,),
        )
    with pytest.raises(ImmutableResultError, match="validity events"):
        service.list_current_heads(compiled.application.job_id)


def test_blocked_report_publishes_no_current_heads(tmp_path: Path) -> None:
    (
        store,
        service,
        commit_id,
        producer_result,
        item_sources,
        policy,
    ) = _prepared(tmp_path)
    compiled = _apply(
        service,
        commit_id=commit_id,
        producer_result=producer_result,
        item_sources=item_sources,
        policy=policy,
    )
    work = compiled.plan.work_items[0]
    run = store.create_stage_run(
        stage_run_id="stage-run://r7-06/persistence/blocked",
        job_id=compiled.application.job_id,
        item_id=work.item_id,
        stage=work.stage,
        attempt=1,
        principal_ref=_ref("principal", "persistence-blocked"),
        input_refs=(revalidation_work_item_v2_ref(work),),
        idempotency_key="create-r7-06-blocked-run",
    )
    running = store.transition_stage_run(
        run.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=run.row_version,
        idempotency_key="start-r7-06-blocked-run",
    )
    store.complete_stage_run(
        stage_result_id="stage-result://r7-06/persistence/blocked",
        stage_run_id=run.stage_run_id,
        status=StageRunStatus.BLOCKED_POLICY,
        expected_version=running.row_version,
        idempotency_key="complete-r7-06-blocked-run",
        failure=FailureRecord(
            failure_class=FailureClass.POLICY,
            code="R7_06_BLOCKED_POLICY",
            message="Closed policy block.",
            retryable=False,
        ),
    )
    service.record_work_result(
        application_id=compiled.application.application_id,
        work_item_id=work.work_item_id,
        stage_run_id=run.stage_run_id,
        policy=policy,
        idempotency_key="record-r7-06-blocked-result",
        audit=_audit(),
    )

    report = service.complete(
        application_id=compiled.application.application_id,
        idempotency_key="complete-r7-06-blocked-application",
        audit=_audit(),
    )

    assert report.outcome is DirectedRevalidationReportOutcomeV2.BLOCKED
    assert report.replacement_current_refs == ()
    assert service.list_current_heads(compiled.application.job_id) == ()


def test_reopen_additive_schema_and_outbox_fault_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        store,
        service,
        commit_id,
        producer_result,
        item_sources,
        policy,
    ) = _prepared(tmp_path)

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected application outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="outbox failure"):
        _apply(
            service,
            commit_id=commit_id,
            producer_result=producer_result,
            item_sources=item_sources,
            policy=policy,
        )
    reopened = UserPlanApplicationPersistenceService(JobStore(store.path))
    with pytest.raises(UserPlanApplicationConflictError, match="not found"):
        reopened.get_application_for_decision(commit_id)

    with sqlite3.connect(store.path) as connection:
        names = {
            str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {
        "user_plan_application_policies",
        "user_plan_applications",
        "directed_revalidation_plans",
        "directed_revalidation_work_items",
        "directed_revalidation_work_results",
        "directed_revalidation_reports",
        "object_validity_events",
        "object_current_heads",
    }.issubset(names)


def test_missing_public_records_fail_closed(tmp_path: Path) -> None:
    _, service, _, _, _, _ = _prepared(tmp_path)

    with pytest.raises(RecordNotFoundError, match="UserPlanApplicationV2"):
        service.get_application("user-plan-application://missing")
    with pytest.raises(RecordNotFoundError, match="RevalidationWorkItemV2"):
        service.get_work_item("revalidation-work-item://missing")
    with pytest.raises(RecordNotFoundError, match="RevalidationWorkResultV2"):
        service.get_work_result("revalidation-work-item://missing")
    with pytest.raises(
        RecordNotFoundError,
        match="DirectedRevalidationReportV2",
    ):
        service.get_report("user-plan-application://missing")
