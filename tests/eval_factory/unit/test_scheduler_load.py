from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.observability_v2 import BatchAuditOutcomeV2
from eval_factory.contracts.orchestration import ItemStatus, JobStatus, StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    resolved_dataset_job_plan_v2_ref,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
    work_control_policy_v2_ref,
)
from eval_factory.contracts.scheduler_load_v2 import (
    SchedulerLoadCaseOutcomeV2,
    SchedulerLoadFaultPointV2,
    SchedulerLoadPolicyV2,
    SchedulerLoadReasonCodeV2,
)
from eval_factory.orchestration import JobStore
from eval_factory.readiness.scheduler_load import (
    R6SchedulerLoadCaseExecutor,
    SchedulerLoadExpectedFaultNotObservedError,
    SchedulerLoadMaterialIntegrityError,
    SchedulerLoadResumeFailedError,
    SchedulerLoadRunner,
    SchedulerLoadTimingError,
)
from eval_factory.readiness.scheduler_load_builder import SchedulerLoadBuilder
from eval_factory.readiness.scheduler_load_models import (
    PreparedSchedulerLoadCase,
    SchedulerLoadCaseResultV1,
)
from eval_factory.readiness.scheduler_load_store import SchedulerLoadMaterialStore

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="scheduler-load-runner-test",
        governing_versions=(
            VersionBinding(
                component="scheduler-load",
                version="scheduler-load/r8-05-v1",
            ),
        ),
    )


@pytest.fixture(scope="module")
def policy() -> SchedulerLoadPolicyV2:
    return SchedulerLoadPolicyV2.create(
        lease_duration_seconds=300,
        heartbeat_extension_seconds=60,
        max_private_bytes=100_000_000,
        max_report_bytes=100_000_000,
        audit=_audit(),
    )


@pytest.fixture(scope="module")
def workload(policy: SchedulerLoadPolicyV2):
    return SchedulerLoadBuilder().compile_workload(
        policy=policy,
        audit=_audit(),
    )


@dataclass
class _Clock:
    value: int = 1_000_000
    step: int = 1_000_000

    def __call__(self) -> int:
        current = self.value
        self.value += self.step
        return current


@pytest.mark.parametrize("fault_point", tuple(SchedulerLoadFaultPointV2))
def test_every_fault_recovers_same_attempt_and_replays_exactly(
    tmp_path: Path,
    policy: SchedulerLoadPolicyV2,
    workload,
    fault_point: SchedulerLoadFaultPointV2,
) -> None:
    builder = SchedulerLoadBuilder()
    member = next(item for item in workload.members if item.assigned_fault_point is fault_point)
    prepared = builder.prepare_case(
        member,
        workload=workload,
        policy=policy,
        audit=_audit(),
    )
    job_store_path = tmp_path / "factory.sqlite3"

    result = R6SchedulerLoadCaseExecutor(clock_ns=_Clock()).execute(
        prepared,
        workload=workload,
        policy=policy,
        job_store_path=job_store_path,
        child_root=tmp_path / "case",
        audit=_audit(),
    )

    assert result.outcome is SchedulerLoadCaseOutcomeV2.RECOVERED
    assert result.reason_code is SchedulerLoadReasonCodeV2.NONE
    assert result.assigned_fault_point is fault_point
    assert result.fault_observed is True
    assert result.resume_succeeded is True
    assert result.replay_stable is True
    assert result.attempt_count == 1
    assert result.unexpected_retry_count == 0
    assert result.duplicate_immutable_fact_count == 0
    assert result.stage_run_count == 1
    assert result.stage_result_count == 1
    assert result.attempt_metrics_count == 1
    assert result.completion_witness_count == 1
    assert result.observability_complete is True
    assert result.initial_elapsed_nanoseconds > 0
    assert result.recovery_elapsed_nanoseconds > 0
    assert result.replay_elapsed_nanoseconds > 0
    assert result.typed_side_effect_count_before_replay == result.typed_side_effect_count_after_replay
    assert result.physical_file_count_before_replay == result.physical_file_count_after_replay
    assert result.physical_bytes_before_replay == result.physical_bytes_after_replay
    assert (tmp_path / "case" / "fault-observation.json").is_file()

    store = JobStore(job_store_path)
    assert store.get_job(prepared.job_spec.job_id).status is JobStatus.RUNNING
    items = store.list_items(prepared.job_spec.job_id)
    assert len(items) == 1
    assert items[0].status is ItemStatus.CANDIDATE
    assert len(store.list_work_units(job_id=prepared.job_spec.job_id)) == 1
    assert len(store.list_work_readiness(prepared.root_work_unit.resolved_work_unit_id)) == 1
    assert len(store.list_work_leases(job_id=prepared.job_spec.job_id)) == 1
    stages = store.list_stage_runs(job_id=prepared.job_spec.job_id)
    assert len(stages) == 1
    assert stages[0].status is StageRunStatus.SUCCEEDED
    assert store.get_stage_result_for_run(stages[0].stage_run_id) is not None
    assert len(store.list_work_attempt_metrics(job_id=prepared.job_spec.job_id)) == 1
    assert (
        R6SchedulerLoadCaseExecutor.read_observability(
            store,
            prepared.job_spec.job_id,
        ).outcome
        is BatchAuditOutcomeV2.COMPLETE
    )
    completion = tuple(
        event
        for event in store.list_outbox()
        if event.aggregate_id == stages[0].stage_run_id and event.event_type == "work-lease-closed"
    )
    assert len(completion) == 1

    public = result.to_public_summary(audit=_audit())
    serialized = public.model_dump_json().casefold()
    assert prepared.job_spec.job_id.casefold() not in serialized
    assert prepared.root_work_unit.resolved_work_unit_id.casefold() not in serialized


def test_executor_rejects_non_monotonic_measurement(
    tmp_path: Path,
    policy: SchedulerLoadPolicyV2,
    workload,
) -> None:
    prepared = SchedulerLoadBuilder().prepare_case(
        workload.members[0],
        workload=workload,
        policy=policy,
        audit=_audit(),
    )

    with pytest.raises(SchedulerLoadTimingError, match="monotonic"):
        R6SchedulerLoadCaseExecutor(clock_ns=lambda: 1).execute(
            prepared,
            workload=workload,
            policy=policy,
            job_store_path=tmp_path / "factory.sqlite3",
            child_root=tmp_path / "case",
            audit=_audit(),
        )


def _ref(object_type: str, version: str, ordinal: int) -> ObjectRef:
    digest = f"{ordinal + 1:064x}"
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version=version,
        object_sha256=digest,
    )


def _fake_case(
    prepared: PreparedSchedulerLoadCase,
    *,
    workload,
    policy: SchedulerLoadPolicyV2,
    physical_bytes_after_replay: int = 1_000_000,
) -> SchedulerLoadCaseResultV1:
    ordinal = prepared.member.ordinal
    return SchedulerLoadCaseResultV1.create(
        member_ref=prepared.member.to_ref(),
        workload_ref=workload.to_ref(),
        policy_ref=policy.to_ref(),
        dataset_job_spec_ref=prepared.dataset_job_spec_ref,
        resolved_plan_ref=resolved_dataset_job_plan_v2_ref(prepared.resolved_plan),
        work_graph_ref=resolved_job_work_graph_v2_ref(prepared.work_graph),
        control_policy_ref=work_control_policy_v2_ref(prepared.control_policy),
        work_unit_ref=resolved_work_unit_v2_ref(prepared.root_work_unit),
        readiness_ref=_ref("work-readiness-snapshot", "v2", ordinal),
        lease_ref=_ref("work-lease", "v2", ordinal),
        stage_run_ref=_ref("stage-run", "identity/v1", ordinal),
        stage_result_ref=_ref("stage-result", "record/v1", ordinal),
        attempt_metrics_ref=_ref("work-attempt-metrics", "v2", ordinal),
        terminal_lease_event_ref=_ref("work-lease-event", "v2", ordinal),
        observability_report_ref=_ref("batch-audit-report", "v2", ordinal),
        output_ref=prepared.output_ref,
        job_id=prepared.job_spec.job_id,
        item_id=prepared.work_graph.item_ids[0],
        assigned_fault_point=prepared.member.assigned_fault_point,
        fault_observed=True,
        resume_succeeded=True,
        replay_stable=True,
        attempt_count=1,
        unexpected_retry_count=0,
        duplicate_immutable_fact_count=0,
        stage_run_count=1,
        stage_result_count=1,
        attempt_metrics_count=1,
        completion_witness_count=1,
        observability_complete=True,
        initial_elapsed_nanoseconds=1_000_000,
        recovery_elapsed_nanoseconds=2_000_000,
        replay_elapsed_nanoseconds=3_000_000,
        typed_side_effect_count_before_replay=14,
        typed_side_effect_count_after_replay=14,
        physical_file_count_before_replay=1,
        physical_file_count_after_replay=1,
        physical_bytes_before_replay=1_000_000,
        physical_bytes_after_replay=physical_bytes_after_replay,
        outcome=SchedulerLoadCaseOutcomeV2.RECOVERED,
        reason_code=SchedulerLoadReasonCodeV2.NONE,
        audit=_audit(),
    )


def test_private_recovered_case_allows_sqlite_wal_byte_maintenance(
    policy: SchedulerLoadPolicyV2,
    workload,
) -> None:
    prepared = SchedulerLoadBuilder().prepare_case(
        workload.members[0],
        workload=workload,
        policy=policy,
        audit=_audit(),
    )

    result = _fake_case(
        prepared,
        workload=workload,
        policy=policy,
        physical_bytes_after_replay=500_000,
    )

    assert result.outcome is SchedulerLoadCaseOutcomeV2.RECOVERED
    assert result.physical_bytes_before_replay != (result.physical_bytes_after_replay)


class _FakeExecutor:
    def __init__(
        self,
        *,
        fail_ordinal: int | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.fail_ordinal = fail_ordinal
        self.failure = failure or RuntimeError("private fake executor detail")
        self.calls: list[int] = []

    def execute(
        self,
        prepared: PreparedSchedulerLoadCase,
        *,
        workload,
        policy: SchedulerLoadPolicyV2,
        job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> SchedulerLoadCaseResultV1:
        del job_store_path, child_root, audit
        self.calls.append(prepared.member.ordinal)
        if prepared.member.ordinal == self.fail_ordinal:
            raise self.failure
        return _fake_case(
            prepared,
            workload=workload,
            policy=policy,
        )


def test_runner_closes_1000_cases_and_reuses_persisted_results(
    tmp_path: Path,
    policy: SchedulerLoadPolicyV2,
    workload,
) -> None:
    executor = _FakeExecutor()
    runner = SchedulerLoadRunner(
        executor=executor,
        verify_job_store=False,
    )

    first = runner.run(
        workload=workload,
        policy=policy,
        job_store_path=tmp_path / "factory.sqlite3",
        run_root=tmp_path / "run",
        audit=_audit(),
    )
    replay_executor = _FakeExecutor(fail_ordinal=0)
    replay = SchedulerLoadRunner(
        executor=replay_executor,
        verify_job_store=False,
    ).run(
        workload=workload,
        policy=policy,
        job_store_path=tmp_path / "factory.sqlite3",
        run_root=tmp_path / "run",
        audit=_audit(),
    )

    assert executor.calls == list(range(1000))
    assert replay_executor.calls == []
    assert replay == first
    assert first.report.outcome.value == "COMPLETE"
    assert first.report.unique_job_count == 1000
    assert first.report.total_faults_observed == 1000
    assert first.report.total_resumes == 1000
    assert first.report.total_replays == 1000
    assert first.report.total_unexpected_retries == 0
    assert tuple(summary.elapsed_nanoseconds for summary in first.report.phase_summaries) == (
        1_000_000_000,
        2_000_000_000,
        3_000_000_000,
    )
    assert (tmp_path / "run" / "run-ledger.json").is_file()
    assert len(tuple((tmp_path / "run" / "case-results").glob("*.json"))) == 1000
    material = SchedulerLoadMaterialStore(
        tmp_path / "private",
        max_private_bytes=policy.max_private_bytes,
        max_members=1000,
    )
    material.put_run_ledger(first.run_ledger)
    material.put_result_set(first.result_set)
    assert material.get_run_ledger(first.run_ledger.to_ref()) == first.run_ledger
    assert material.get_result_set(first.result_set.to_ref()) == first.result_set


def test_runner_isolates_one_internal_error_and_continues(
    tmp_path: Path,
    policy: SchedulerLoadPolicyV2,
    workload,
) -> None:
    executor = _FakeExecutor(fail_ordinal=1)
    result = SchedulerLoadRunner(
        executor=executor,
        verify_job_store=False,
    ).run(
        workload=workload,
        policy=policy,
        job_store_path=tmp_path / "factory.sqlite3",
        run_root=tmp_path / "run",
        audit=_audit(),
    )

    assert executor.calls == list(range(1000))
    assert result.report.outcome.value == "INCOMPLETE"
    assert len(result.report.infrastructure_error_case_refs) == 1
    assert len(result.report.recovered_case_refs) == 999
    failed = result.case_results[1]
    assert failed.outcome is SchedulerLoadCaseOutcomeV2.INFRASTRUCTURE_ERROR
    assert failed.reason_code is SchedulerLoadReasonCodeV2.INTERNAL_ERROR
    assert "private fake executor detail" not in failed.model_dump_json()


@pytest.mark.parametrize(
    ("failure", "expected_reason", "fault_observed"),
    (
        (
            SchedulerLoadExpectedFaultNotObservedError(),
            SchedulerLoadReasonCodeV2.EXPECTED_FAULT_NOT_OBSERVED,
            False,
        ),
        (
            SchedulerLoadResumeFailedError(),
            SchedulerLoadReasonCodeV2.RESUME_FAILED,
            True,
        ),
    ),
)
def test_runner_preserves_closed_expected_case_failures(
    tmp_path: Path,
    policy: SchedulerLoadPolicyV2,
    workload,
    failure: Exception,
    expected_reason: SchedulerLoadReasonCodeV2,
    fault_observed: bool,
) -> None:
    result = SchedulerLoadRunner(
        executor=_FakeExecutor(fail_ordinal=1, failure=failure),
        verify_job_store=False,
    ).run(
        workload=workload,
        policy=policy,
        job_store_path=tmp_path / "factory.sqlite3",
        run_root=tmp_path / "run",
        audit=_audit(),
    )

    failed = result.case_results[1]
    assert failed.outcome is SchedulerLoadCaseOutcomeV2.FAILED
    assert failed.reason_code is expected_reason
    assert failed.fault_observed is fault_observed
    assert len(result.report.failed_case_refs) == 1
    assert result.report.infrastructure_error_case_refs == ()


def test_runner_does_not_classify_material_integrity_as_resume_failure(
    tmp_path: Path,
    policy: SchedulerLoadPolicyV2,
    workload,
) -> None:
    result = SchedulerLoadRunner(
        executor=_FakeExecutor(
            fail_ordinal=1,
            failure=SchedulerLoadMaterialIntegrityError(),
        ),
        verify_job_store=False,
    ).run(
        workload=workload,
        policy=policy,
        job_store_path=tmp_path / "factory.sqlite3",
        run_root=tmp_path / "run",
        audit=_audit(),
    )

    failed = result.case_results[1]
    assert failed.outcome is SchedulerLoadCaseOutcomeV2.INFRASTRUCTURE_ERROR
    assert failed.reason_code is SchedulerLoadReasonCodeV2.MATERIAL_INTEGRITY_ERROR
    assert len(result.report.infrastructure_error_case_refs) == 1
    assert result.report.failed_case_refs == ()
