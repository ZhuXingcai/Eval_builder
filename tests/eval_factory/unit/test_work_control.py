from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
from test_artifact_execution import (
    _audit as _artifact_audit,
)
from test_artifact_execution import (
    _ControlledExecutionFacade,
    _definition,
    _execution_plan,
)
from test_job_store import _audit, _job_spec, _plan, _work_graph
from test_work_fanout import _planning_result

from eval_factory.attachment_planning import ArtifactGroupExecutor
from eval_factory.contracts.core import FailureClass, FailureRecord, ObjectRef
from eval_factory.contracts.observability_v2 import (
    work_attempt_metrics_v2_ref,
)
from eval_factory.contracts.orchestration import JobStatus, StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    WorkLeaseEventKindV2,
    WorkReadinessV2,
    WorkRetryDecisionKindV2,
    work_cancellation_record_v2_ref,
    work_control_policy_v2_ref,
    work_lease_event_v2_ref,
    work_lease_v2_ref,
    work_retry_decision_v2_ref,
)
from eval_factory.orchestration import (
    ArtifactGroupFanoutCompiler,
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
    StageWorkCompletion,
    StaleWorkLeaseError,
    WorkControlPolicyError,
    WorkControlService,
    WorkReadinessEvaluator,
    stage_result_record_ref,
)

NOW = datetime(2026, 7, 31, tzinfo=UTC)
HASH = "a" * 64
GOLD_PATH = (
    Path(__file__).resolve().parents[3] / "evals/golden/eval_factory/orchestration/r6-05-work-control-v1.json"
)


@dataclass
class _MutableClock:
    value: datetime = NOW

    def __call__(self) -> datetime:
        return self.value

    def advance(self, *, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _holder(suffix: str = "a") -> ObjectRef:
    return ObjectRef(
        object_type="worker-principal",
        object_id=f"worker-principal://r6-05/{suffix}",
        object_version="v1",
        object_sha256=HASH,
    )


def _failure(*, retryable: bool = True) -> FailureRecord:
    return FailureRecord(
        failure_class=FailureClass.INTERNAL,
        code="r6-05-injected-stage-failure",
        message="Typed injected work-control failure.",
        retryable=retryable,
    )


def _prepared(
    tmp_path: Path,
    *,
    max_attempts: int = 3,
    retry_delay_seconds: tuple[int, ...] = (5, 10),
    retry_lease_expiry: bool = True,
):
    clock = _MutableClock()
    store = JobStore(tmp_path / "factory.sqlite3", clock=clock)
    spec = _job_spec(job_id="job://r6-05/control")
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-r6-05-work-graph",
    )
    created = store.get_job(spec.job_id)
    store.transition_job(
        spec.job_id,
        JobStatus.RUNNING,
        expected_version=created.row_version,
        idempotency_key="start-r6-05-job",
    )
    unit = graph.work_units[0]
    readiness = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=unit,
        stage_completions=(),
        item_records=(),
        audit=_audit(),
    )
    assert readiness.readiness is WorkReadinessV2.READY
    store.record_work_readiness(
        readiness,
        idempotency_key="record-r6-05-root-readiness",
    )
    service = WorkControlService(store)
    policy = service.bind_policy(
        graph=graph,
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        retry_lease_expiry=retry_lease_expiry,
        audit=_audit(),
        idempotency_key="bind-r6-05-control-policy",
    )
    return clock, store, service, graph, unit, readiness, policy


def test_ready_work_acquires_one_fenced_running_stage_attempt(tmp_path: Path) -> None:
    _clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    holder = _holder()

    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-attempt-1",
    )
    replay = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-attempt-1",
    )

    assert replay == lease
    assert store.get_work_control_policy(graph.job_id) == policy
    assert store.get_work_lease(lease.work_lease_id) == lease
    assert service.get_lease_head(lease.work_lease_id) == (store.get_work_lease_head(lease.work_lease_id))
    assert lease.fencing_token == 1
    assert lease.attempt == 1
    assert lease.stage_run_ref is not None
    stage = store.get_stage_run(lease.stage_run_ref.object_id)
    assert stage.status is StageRunStatus.RUNNING
    assert stage.attempt == 1
    assert lease.work_unit_ref in stage.input_refs
    assert len(store.list_stage_runs(job_id=graph.job_id)) == 1

    with pytest.raises(WorkControlPolicyError, match="active lease"):
        service.acquire(
            graph=graph,
            work_unit=unit,
            readiness_snapshot=readiness,
            policy=policy,
            holder_ref=_holder("b"),
            retry_decision=None,
            audit=_audit(),
            idempotency_key="competing-r6-05-attempt-1",
        )


def test_heartbeat_extends_before_expiry_and_rejects_stale_holder(tmp_path: Path) -> None:
    clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    holder = _holder()
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-heartbeat-attempt",
    )
    clock.advance(seconds=20)
    event = service.heartbeat(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        audit=_audit(),
        idempotency_key="heartbeat-r6-05-attempt",
    )

    assert event.event_kind is WorkLeaseEventKindV2.HEARTBEAT
    assert event.lease_version == 1
    assert event.effective_expires_at == NOW + timedelta(seconds=40)
    assert store.get_work_lease_head(lease.work_lease_id).lease_version == 1

    with pytest.raises(StaleWorkLeaseError, match="holder"):
        service.heartbeat(
            lease=lease,
            policy=policy,
            holder_ref=_holder("wrong"),
            expected_lease_version=1,
            audit=_audit(),
            idempotency_key="heartbeat-r6-05-wrong-holder",
        )

    clock.advance(seconds=20)
    with pytest.raises(StaleWorkLeaseError, match="expired"):
        service.heartbeat(
            lease=lease,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=1,
            audit=_audit(),
            idempotency_key="heartbeat-r6-05-at-expiry",
        )


def test_retry_policy_delays_linked_attempt_and_increases_fence(tmp_path: Path) -> None:
    clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    holder = _holder()
    first = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-retry-attempt-1",
    )
    failed = service.complete_stage(
        lease=first,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.RETRYABLE_FAILURE,
        output_refs=(),
        failure=_failure(),
        checkpoint_ref=None,
        metrics_ref=None,
        audit=_audit(),
        idempotency_key="complete-retry-attempt-1",
    )
    decision = service.get_retry_decision(unit.resolved_work_unit_id)

    assert failed.event_kind is WorkLeaseEventKindV2.RETRYABLE_FAILURE
    assert failed.failure is not None
    assert failed.failure.message == ("Controlled work completed with a typed non-success outcome.")
    assert failed.failure.evidence_refs == ()
    assert failed.failure.detail == ()
    assert decision.decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED
    assert decision.next_attempt == 2
    assert decision.eligible_at == NOW + timedelta(seconds=5)

    with pytest.raises(WorkControlPolicyError, match="not yet eligible"):
        service.acquire(
            graph=graph,
            work_unit=unit,
            readiness_snapshot=readiness,
            policy=policy,
            holder_ref=holder,
            retry_decision=decision,
            audit=_audit(),
            idempotency_key="early-retry-attempt-2",
        )

    clock.advance(seconds=5)
    second = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=decision,
        audit=_audit(),
        idempotency_key="acquire-retry-attempt-2",
    )
    second_stage = store.get_stage_run(second.stage_run_ref.object_id)  # type: ignore[union-attr]
    assert second.fencing_token == 2
    assert second.attempt == 2
    assert second_stage.retry_of_stage_run_id == first.stage_run_ref.object_id  # type: ignore[union-attr]

    service.complete_stage(
        lease=second,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(
            ObjectRef(
                object_type="trace-envelope",
                object_id="trace-envelope://r6-05/success",
                object_version="v2",
                object_sha256=HASH,
            ),
        ),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        audit=_audit(),
        idempotency_key="complete-retry-attempt-2",
    )

    with pytest.raises(StaleWorkLeaseError, match="terminal"):
        service.complete_stage(
            lease=first,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=(),
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            audit=_audit(),
            idempotency_key="late-complete-attempt-1",
        )


def test_expiry_at_boundary_rejects_late_result_and_records_exhaustion(tmp_path: Path) -> None:
    clock, store, service, graph, unit, readiness, policy = _prepared(
        tmp_path,
        max_attempts=1,
        retry_delay_seconds=(),
    )
    holder = _holder()
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-expiring-attempt",
    )
    clock.advance(seconds=30)
    event = service.expire(
        lease=lease,
        policy=policy,
        audit=_audit(),
        idempotency_key="expire-r6-05-attempt",
    )
    decision = service.get_retry_decision(unit.resolved_work_unit_id)

    assert event.event_kind is WorkLeaseEventKindV2.EXPIRED
    assert decision.decision is WorkRetryDecisionKindV2.EXHAUSTED
    assert store.get_stage_result_for_run(lease.stage_run_ref.object_id) is not None  # type: ignore[union-attr]
    stage = store.get_stage_run(lease.stage_run_ref.object_id)  # type: ignore[union-attr]
    result = store.get_stage_result_for_run(stage.stage_run_id)
    assert result is not None
    target = next(
        candidate
        for candidate in graph.work_units
        if candidate.depends_on_work_unit_refs == (lease.work_unit_ref,)
    )
    snapshot = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=target,
        stage_completions=(
            StageWorkCompletion(
                work_unit_ref=lease.work_unit_ref,
                stage_run=stage,
                stage_result=result,
            ),
        ),
        item_records=(),
        work_retry_decisions=(decision,),
        audit=_audit(),
    )
    assert snapshot.readiness is WorkReadinessV2.BLOCKED_DEPENDENCY
    assert (
        store.record_work_readiness(
            snapshot,
            idempotency_key="record-r6-05-exhausted-readiness",
        )
        == snapshot
    )
    with pytest.raises(StaleWorkLeaseError, match="terminal"):
        service.complete_stage(
            lease=lease,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=(),
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            audit=_audit(),
            idempotency_key="late-complete-expired-attempt",
        )


def test_cancellation_closes_active_lease_and_preserves_completion_order(tmp_path: Path) -> None:
    _clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    holder = _holder()
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-cancelled-attempt",
    )
    cancellation = service.cancel_job(
        graph=graph,
        policy=policy,
        reason_code="operator-cancelled",
        audit=_audit(),
        idempotency_key="cancel-r6-05-job",
    )

    assert store.get_job(graph.job_id).status is JobStatus.CANCELLED
    assert len(cancellation.cancelled_lease_event_refs) == 1
    result = store.get_stage_result_for_run(lease.stage_run_ref.object_id)  # type: ignore[union-attr]
    assert result is not None and result.status is StageRunStatus.CANCELLED
    with pytest.raises(StaleWorkLeaseError, match="terminal"):
        service.complete_stage(
            lease=lease,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=(),
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            audit=_audit(),
            idempotency_key="late-complete-cancelled-attempt",
        )
    with pytest.raises(WorkControlPolicyError, match="cancelled"):
        service.acquire(
            graph=graph,
            work_unit=unit,
            readiness_snapshot=readiness,
            policy=policy,
            holder_ref=holder,
            retry_decision=None,
            audit=_audit(),
            idempotency_key="dispatch-after-cancel",
        )


@pytest.mark.asyncio
async def test_artifact_group_lease_has_no_stage_run_and_expires_as_control_evidence(
    tmp_path: Path,
) -> None:
    clock, store, service, graph, _unit, _readiness, policy = _prepared(
        tmp_path,
        retry_lease_expiry=False,
    )
    parent = next(unit for unit in graph.work_units if unit.stage.value == "attachment")
    execution_plan = await _execution_plan(
        tmp_path / "artifact-control",
        (("artifact://a", "inputs/a.txt"),),
        (_definition("artifact://a"),),
    )
    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=graph,
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=_planning_result(execution_plan),
        audit=_audit(),
    )
    store.create_artifact_group_fanout(
        fanout,
        idempotency_key="create-r6-05-artifact-fanout",
    )
    group = fanout.group_work_units[0]
    readiness = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=group,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        audit=_audit(),
    )
    store.record_work_readiness(
        readiness,
        idempotency_key="record-r6-05-group-readiness",
    )
    before = store.list_stage_runs(job_id=graph.job_id)
    holder = _holder("artifact-gold")
    lease = service.acquire(
        graph=graph,
        work_unit=group,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-artifact-group",
    )

    assert lease.stage_run_ref is None
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    assert work_lease_v2_ref(lease).model_dump(mode="json") == (gold["artifact_group"]["lease_ref"])
    assert store.list_stage_runs(job_id=graph.job_id) == before
    clock.advance(seconds=30)
    event = service.expire(
        lease=lease,
        policy=policy,
        audit=_audit(),
        idempotency_key="expire-r6-05-artifact-group",
    )
    assert event.event_kind is WorkLeaseEventKindV2.EXPIRED
    assert event.work_unit_ref == lease.work_unit_ref
    assert service.get_retry_decision_optional(group.resolved_work_unit_id) is None


def test_two_workers_race_to_one_active_lease(tmp_path: Path) -> None:
    _clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    barrier = Barrier(2)

    def acquire(suffix: str):
        barrier.wait(timeout=5)
        try:
            return service.acquire(
                graph=graph,
                work_unit=unit,
                readiness_snapshot=readiness,
                policy=policy,
                holder_ref=_holder(suffix),
                retry_decision=None,
                audit=_audit(),
                idempotency_key=f"race-acquire-{suffix}",
            )
        except WorkControlPolicyError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        leases = tuple(executor.map(acquire, ("a", "b")))

    winners = tuple(lease for lease in leases if lease is not None)
    assert len(winners) == 1
    assert winners[0].fencing_token == 1
    assert len(store.list_stage_runs(job_id=graph.job_id)) == 1


def test_non_ready_work_creates_no_lease_or_stage_run(tmp_path: Path) -> None:
    _clock, store, service, graph, _unit, _readiness, policy = _prepared(tmp_path)
    waiting_unit = graph.work_units[1]
    waiting = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=waiting_unit,
        stage_completions=(),
        item_records=(),
        audit=_audit(),
    )
    assert waiting.readiness is WorkReadinessV2.WAITING
    store.record_work_readiness(
        waiting,
        idempotency_key="record-r6-05-waiting-readiness",
    )

    with pytest.raises(WorkControlPolicyError, match="READY"):
        service.acquire(
            graph=graph,
            work_unit=waiting_unit,
            readiness_snapshot=waiting,
            policy=policy,
            holder_ref=_holder(),
            retry_decision=None,
            audit=_audit(),
            idempotency_key="acquire-r6-05-waiting",
        )
    assert store.list_stage_runs(job_id=graph.job_id) == ()


def test_controlled_stage_run_rejects_legacy_unfenced_completion(
    tmp_path: Path,
) -> None:
    _clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=_holder(),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-legacy-guard",
    )
    stage = store.get_stage_run(lease.stage_run_ref.object_id)  # type: ignore[union-attr]

    with pytest.raises(StaleWorkLeaseError, match="fenced"):
        store.complete_stage_run(
            stage_result_id="stage-result://r6-05/unfenced",
            stage_run_id=stage.stage_run_id,
            status=StageRunStatus.SUCCEEDED,
            expected_version=stage.row_version,
            idempotency_key="unfenced-r6-05-completion",
        )
    assert store.get_stage_result_for_run(stage.stage_run_id) is None


def test_controlled_stage_outcome_rejects_retryability_mismatch(
    tmp_path: Path,
) -> None:
    _clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=_holder(),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-outcome-mismatch",
    )

    with pytest.raises(WorkControlPolicyError, match="retryable"):
        service.complete_stage(
            lease=lease,
            policy=policy,
            holder_ref=_holder(),
            expected_lease_version=0,
            status=StageRunStatus.RETRYABLE_FAILURE,
            output_refs=(),
            failure=_failure(retryable=False),
            checkpoint_ref=None,
            metrics_ref=None,
            audit=_audit(),
            idempotency_key="complete-r6-05-outcome-mismatch",
        )
    assert (
        store.get_stage_result_for_run(
            lease.stage_run_ref.object_id  # type: ignore[union-attr]
        )
        is None
    )
    assert store.get_work_lease_head(lease.work_lease_id).state.value == ("ACTIVE")


def test_heartbeat_and_completion_outbox_faults_roll_back_every_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    holder = _holder()
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-faults",
    )
    stage_id = lease.stage_run_ref.object_id  # type: ignore[union-attr]
    original = store._append_outbox

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected work-control outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="outbox failure"):
        service.heartbeat(
            lease=lease,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            audit=_audit(),
            idempotency_key="heartbeat-r6-05-fault",
        )
    assert store.get_work_lease_head(lease.work_lease_id).lease_version == 0
    assert store.list_work_lease_events(lease.work_lease_id) == ()

    with pytest.raises(RuntimeError, match="outbox failure"):
        service.complete_stage(
            lease=lease,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            status=StageRunStatus.RETRYABLE_FAILURE,
            output_refs=(),
            failure=_failure(),
            checkpoint_ref=None,
            metrics_ref=None,
            audit=_audit(),
            idempotency_key="complete-r6-05-fault",
        )
    assert store.get_stage_result_for_run(stage_id) is None
    assert store.get_work_lease_head(lease.work_lease_id).lease_version == 0
    assert store.list_work_lease_events(lease.work_lease_id) == ()
    assert service.get_retry_decision_optional(unit.resolved_work_unit_id) is None
    with pytest.raises(RecordNotFoundError):
        store.get_work_attempt_metrics_for_lease(lease.work_lease_id)

    monkeypatch.setattr(store, "_append_outbox", original)
    event = service.complete_stage(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        audit=_audit(),
        idempotency_key="complete-r6-05-after-fault",
    )
    assert event.event_kind is WorkLeaseEventKindV2.SUCCEEDED
    metrics = store.get_work_attempt_metrics_for_lease(lease.work_lease_id)
    result = store.get_stage_result_for_run(stage_id)
    assert result is not None
    assert result.metrics_ref == work_attempt_metrics_v2_ref(metrics)


def test_completion_first_cancellation_preserves_success_and_exact_replay(
    tmp_path: Path,
) -> None:
    _clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    holder = _holder()
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-completion-first",
    )
    service.complete_stage(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        audit=_audit(),
        idempotency_key="complete-r6-05-before-cancel",
    )
    result = store.get_stage_result_for_run(lease.stage_run_ref.object_id)  # type: ignore[union-attr]
    assert result is not None

    cancellation = service.cancel_job(
        graph=graph,
        policy=policy,
        reason_code="operator-cancelled-after-completion",
        audit=_audit(),
        idempotency_key="cancel-r6-05-after-completion",
    )
    replay = service.cancel_job(
        graph=graph,
        policy=policy,
        reason_code="operator-cancelled-after-completion",
        audit=_audit(),
        idempotency_key="cancel-r6-05-after-completion",
    )

    assert replay == cancellation
    assert cancellation.cancelled_lease_event_refs == ()
    assert cancellation.preserved_result_refs == (stage_result_record_ref(result),)
    assert store.get_stage_result_for_run(result.stage_run_id) == result


def test_cancellation_fault_rolls_back_job_result_and_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=_holder(),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-cancel-fault",
    )
    original = store._append_outbox

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected cancellation outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="cancellation outbox failure"):
        service.cancel_job(
            graph=graph,
            policy=policy,
            reason_code="operator-cancelled",
            audit=_audit(),
            idempotency_key="cancel-r6-05-fault",
        )

    assert store.get_job(graph.job_id).status is JobStatus.RUNNING
    assert store.get_stage_result_for_run(lease.stage_run_ref.object_id) is None  # type: ignore[union-attr]
    assert store.get_work_lease_head(lease.work_lease_id).state.value == "ACTIVE"
    assert store.list_work_lease_events(lease.work_lease_id) == ()

    monkeypatch.setattr(store, "_append_outbox", original)
    service.cancel_job(
        graph=graph,
        policy=policy,
        reason_code="operator-cancelled",
        audit=_audit(),
        idempotency_key="cancel-r6-05-after-fault",
    )
    assert store.get_job(graph.job_id).status is JobStatus.CANCELLED


def test_lease_head_rebuild_survives_reopen_and_detects_projection_drift(
    tmp_path: Path,
) -> None:
    clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    holder = _holder()
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-rebuild",
    )
    clock.advance(seconds=10)
    service.heartbeat(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        audit=_audit(),
        idempotency_key="heartbeat-r6-05-rebuild",
    )
    expected = store.get_work_lease_head(lease.work_lease_id)
    reopened = JobStore(store.path, clock=clock)
    assert reopened.get_work_lease_head(lease.work_lease_id) == expected

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE work_lease_heads
            SET lease_version = 99
            WHERE work_lease_id = ?
            """,
            (lease.work_lease_id,),
        )
    with pytest.raises(ImmutableResultError, match=r"columns|differs"):
        reopened.get_work_lease_head(lease.work_lease_id)


@pytest.mark.asyncio
async def test_artifact_group_fenced_completion_uses_exact_r5_batch_without_stage_run(
    tmp_path: Path,
) -> None:
    _clock, store, service, graph, _unit, _readiness, policy = _prepared(tmp_path)
    parent = next(unit for unit in graph.work_units if unit.stage.value == "attachment")
    execution_plan = await _execution_plan(
        tmp_path / "artifact-fenced-completion",
        (("artifact://a", "inputs/a.txt"),),
        (_definition("artifact://a"),),
    )
    planning = _planning_result(execution_plan)
    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=graph,
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=planning,
        audit=_audit(),
    )
    store.create_artifact_group_fanout(
        fanout,
        idempotency_key="create-r6-05-completion-fanout",
    )
    group = fanout.group_work_units[0]
    readiness = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=group,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        audit=_audit(),
    )
    store.record_work_readiness(
        readiness,
        idempotency_key="record-r6-05-completion-group-ready",
    )
    holder = _holder("artifact-completion")
    lease = service.acquire(
        graph=graph,
        work_unit=group,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-completion-group",
    )
    batch = await ArtifactGroupExecutor().run(
        execution_plan,
        facade=_ControlledExecutionFacade(expected_parallelism=1),
        audit=_artifact_audit(),
    )
    event = service.complete_artifact_group(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        fanout=fanout,
        artifact_execution_planning_result=planning,
        artifact_execution_batch=batch,
        audit=_audit(),
        idempotency_key="complete-r6-05-artifact-group",
    )

    assert event.event_kind is WorkLeaseEventKindV2.SUCCEEDED
    assert len(event.result_refs) == 2
    metrics = store.get_work_attempt_metrics_for_lease(lease.work_lease_id)
    assert metrics.work_lease_ref == work_lease_v2_ref(lease)
    assert metrics.stage_run_ref is None
    assert len(metrics.artifact_slices) == 1
    artifact_slice = metrics.artifact_slices[0]
    assert artifact_slice.artifact_id == "artifact://a"
    assert artifact_slice.status == "SUCCEEDED"
    assert artifact_slice.selected_route_kind == "PROVIDER"
    assert artifact_slice.output_ref is not None
    assert metrics.output_refs == (artifact_slice.output_ref,)
    assert set(metrics.output_refs).isdisjoint(metrics.result_refs)
    assert lease.stage_run_ref is None
    assert store.list_stage_runs(job_id=graph.job_id) == ()
    with pytest.raises(RecordNotFoundError):
        service.get_retry_decision(group.resolved_work_unit_id)
    with pytest.raises(WorkControlPolicyError, match="cannot bind a StageRun"):
        service.complete_artifact_group(
            lease=lease.model_copy(
                update={
                    "stage_run_ref": ObjectRef(
                        object_type="stage-run",
                        object_id="stage-run://r6-05/invalid-group",
                        object_version="identity/v1",
                        object_sha256=HASH,
                    )
                }
            ),
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            fanout=fanout,
            artifact_execution_planning_result=planning,
            artifact_execution_batch=batch,
            audit=_audit(),
            idempotency_key="reject-r6-05-group-stage-run",
        )
    with pytest.raises(WorkControlPolicyError, match="fanout is not current"):
        service.complete_artifact_group(
            lease=lease,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            fanout=fanout.model_copy(update={"artifact_group_fanout_sha256": "f" * 64}),
            artifact_execution_planning_result=planning,
            artifact_execution_batch=batch,
            audit=_audit(),
            idempotency_key="reject-r6-05-stale-fanout",
        )
    with pytest.raises(WorkControlPolicyError, match="does not belong"):
        service.complete_artifact_group(
            lease=lease.model_copy(
                update={
                    "work_unit_ref": ObjectRef(
                        object_type="resolved-work-unit",
                        object_id="resolved-work-unit://r6-05/other",
                        object_version="v2",
                        object_sha256=HASH,
                    )
                }
            ),
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            fanout=fanout,
            artifact_execution_planning_result=planning,
            artifact_execution_batch=batch,
            audit=_audit(),
            idempotency_key="reject-r6-05-cross-group-lease",
        )
    with pytest.raises(WorkControlPolicyError, match="not current"):
        service.complete_artifact_group(
            lease=lease,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            fanout=fanout,
            artifact_execution_planning_result={},
            artifact_execution_batch=batch,
            audit=_audit(),
            idempotency_key="reject-r6-05-malformed-planning",
        )
    with pytest.raises(WorkControlPolicyError, match="attempt"):
        service.complete_artifact_group(
            lease=lease.model_copy(update={"attempt": 2}),
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            fanout=fanout,
            artifact_execution_planning_result=planning,
            artifact_execution_batch=batch,
            audit=_audit(),
            idempotency_key="reject-r6-05-stale-group-attempt",
        )
    cancellation = service.cancel_job(
        graph=graph,
        policy=policy,
        reason_code="cancel-after-artifact-success",
        audit=_audit(),
        idempotency_key="cancel-after-r6-05-artifact-success",
    )
    replay = service.cancel_job(
        graph=graph,
        policy=policy,
        reason_code="cancel-after-artifact-success",
        audit=_audit(),
        idempotency_key="cancel-after-r6-05-artifact-success",
    )
    assert replay == cancellation
    assert set(event.result_refs) <= set(cancellation.preserved_result_refs)


@pytest.mark.asyncio
async def test_artifact_retry_exhaustion_persists_parent_terminal_join(
    tmp_path: Path,
) -> None:
    _clock, store, service, graph, _unit, _readiness, policy = _prepared(
        tmp_path,
        max_attempts=1,
        retry_delay_seconds=(),
    )
    parent = next(unit for unit in graph.work_units if unit.stage.value == "attachment")
    execution_plan = await _execution_plan(
        tmp_path / "artifact-exhaustion",
        (("artifact://a", "inputs/a.txt"),),
        (_definition("artifact://a"),),
    )
    planning = _planning_result(execution_plan)
    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=graph,
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=planning,
        audit=_audit(),
    )
    store.create_artifact_group_fanout(
        fanout,
        idempotency_key="create-r6-05-exhaustion-fanout",
    )
    group = fanout.group_work_units[0]
    group_ready = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=group,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        audit=_audit(),
    )
    store.record_work_readiness(
        group_ready,
        idempotency_key="record-r6-05-exhaustion-group-ready",
    )
    holder = _holder("artifact-exhaustion")
    lease = service.acquire(
        graph=graph,
        work_unit=group,
        readiness_snapshot=group_ready,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-exhaustion-group",
    )
    batch = await ArtifactGroupExecutor().run(
        execution_plan,
        facade=_ControlledExecutionFacade(
            fail_once=frozenset({"artifact://a"}),
            expected_parallelism=1,
        ),
        audit=_artifact_audit(),
    )
    service.complete_artifact_group(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        fanout=fanout,
        artifact_execution_planning_result=planning,
        artifact_execution_batch=batch,
        audit=_audit(),
        idempotency_key="complete-r6-05-exhaustion-group",
    )
    decision = service.get_retry_decision(group.resolved_work_unit_id)
    assert decision.decision is WorkRetryDecisionKindV2.EXHAUSTED

    parent_ready = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=parent,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        artifact_execution_planning_result=planning,
        artifact_execution_batch=batch,
        work_retry_decisions=(decision,),
        audit=_audit(),
    )
    assert parent_ready.readiness is WorkReadinessV2.READY
    assert (
        store.record_work_readiness(
            parent_ready,
            idempotency_key="record-r6-05-exhausted-parent-ready",
        )
        == parent_ready
    )


def test_work_control_gold_is_deterministic_and_content_free(tmp_path: Path) -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    clock, _store, service, graph, unit, readiness, policy = _prepared(
        tmp_path,
        max_attempts=1,
        retry_delay_seconds=(),
    )
    holder = _holder("gold")
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="gold-acquire-stage",
    )
    clock.advance(seconds=10)
    heartbeat = service.heartbeat(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        audit=_audit(),
        idempotency_key="gold-heartbeat-stage",
    )
    failed = service.complete_stage(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=1,
        status=StageRunStatus.RETRYABLE_FAILURE,
        output_refs=(),
        failure=_failure(),
        checkpoint_ref=None,
        metrics_ref=None,
        audit=_audit(),
        idempotency_key="gold-complete-stage",
    )
    decision = service.get_retry_decision(unit.resolved_work_unit_id)
    cancellation = service.cancel_job(
        graph=graph,
        policy=policy,
        reason_code="gold-cancelled",
        audit=_audit(),
        idempotency_key="gold-cancel-job",
    )

    assert work_control_policy_v2_ref(policy).model_dump(mode="json") == (gold["stage"]["policy_ref"])
    assert work_lease_v2_ref(lease).model_dump(mode="json") == (gold["stage"]["lease_ref"])
    assert work_lease_event_v2_ref(heartbeat).model_dump(mode="json") == (gold["stage"]["heartbeat_ref"])
    assert work_lease_event_v2_ref(failed).model_dump(mode="json") == (gold["stage"]["retryable_event_ref"])
    assert (
        work_retry_decision_v2_ref(decision).model_dump(mode="json")
        == (gold["stage"]["exhausted_decision_ref"])
    )
    assert (
        work_cancellation_record_v2_ref(cancellation).model_dump(mode="json")
        == gold["stage"]["cancellation_ref"]
    )
    assert lease.stage_run_ref is not None
    assert lease.stage_run_ref.object_id == gold["stage"]["stage_run_id"]

    serialized = json.dumps(gold, sort_keys=True).lower()
    for forbidden in (
        "trace_text",
        "task_text",
        "private_reference",
        "grader",
        "credential",
        "physical_path",
        "runtime_transcript",
    ):
        assert forbidden not in serialized


def test_control_commands_replay_exactly_and_changed_requests_conflict(
    tmp_path: Path,
) -> None:
    clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    replayed_policy = service.bind_policy(
        graph=graph,
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=3,
        retry_delay_seconds=(5, 10),
        retry_lease_expiry=True,
        audit=_audit().model_copy(update={"created_at": NOW + timedelta(days=1)}),
        idempotency_key="bind-r6-05-control-policy",
    )
    assert replayed_policy == policy
    with pytest.raises(IdempotencyConflictError, match="different request"):
        service.bind_policy(
            graph=graph,
            lease_duration_seconds=31,
            heartbeat_extension_seconds=20,
            max_attempts=3,
            retry_delay_seconds=(5, 10),
            retry_lease_expiry=True,
            audit=_audit(),
            idempotency_key="bind-r6-05-control-policy",
        )

    holder = _holder()
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-command-replay",
    )
    clock.advance(seconds=10)
    heartbeat = service.heartbeat(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        audit=_audit(),
        idempotency_key="heartbeat-r6-05-command-replay",
    )
    replayed_heartbeat = service.heartbeat(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=0,
        audit=_audit(),
        idempotency_key="heartbeat-r6-05-command-replay",
    )
    assert replayed_heartbeat == heartbeat
    with pytest.raises(StaleWorkLeaseError, match="version"):
        service.heartbeat(
            lease=lease,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=0,
            audit=_audit(),
            idempotency_key="heartbeat-r6-05-stale-version",
        )

    completed = service.complete_stage(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=1,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        audit=_audit(),
        idempotency_key="complete-r6-05-command-replay",
    )
    replayed_completion = service.complete_stage(
        lease=lease,
        policy=policy,
        holder_ref=holder,
        expected_lease_version=1,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        audit=_audit(),
        idempotency_key="complete-r6-05-command-replay",
    )
    assert replayed_completion == completed
    with pytest.raises(IdempotencyConflictError, match="different request"):
        service.complete_stage(
            lease=lease,
            policy=policy,
            holder_ref=holder,
            expected_lease_version=1,
            status=StageRunStatus.TERMINAL_FAILURE,
            output_refs=(),
            failure=_failure(retryable=False),
            checkpoint_ref=None,
            metrics_ref=None,
            audit=_audit(),
            idempotency_key="complete-r6-05-command-replay",
        )
    assert store.get_work_lease_head(lease.work_lease_id).state.value == ("SUCCEEDED")
    assert (
        service.expire(
            lease=lease,
            policy=policy,
            audit=_audit(),
            idempotency_key="expire-after-r6-05-completion",
        )
        == completed
    )


def test_expire_before_deadline_is_rejected_without_state_change(
    tmp_path: Path,
) -> None:
    _clock, store, service, graph, unit, readiness, policy = _prepared(tmp_path)
    lease = service.acquire(
        graph=graph,
        work_unit=unit,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=_holder(),
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r6-05-premature-expiry",
    )

    with pytest.raises(WorkControlPolicyError, match="not expired"):
        service.expire(
            lease=lease,
            policy=policy,
            audit=_audit(),
            idempotency_key="expire-r6-05-before-deadline",
        )
    assert store.get_work_lease_head(lease.work_lease_id).state.value == ("ACTIVE")
    assert store.list_work_lease_events(lease.work_lease_id) == ()
