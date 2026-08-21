from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.scheduler_load_v2 import (
    SchedulerLoadCaseOutcomeV2,
    SchedulerLoadCaseSummaryV2,
    SchedulerLoadFaultPointV2,
    SchedulerLoadPhaseSummaryV2,
    SchedulerLoadPhaseV2,
    SchedulerLoadPolicyV2,
    SchedulerLoadReasonCodeV2,
    SchedulerLoadReportV2,
)
from eval_factory.readiness.scheduler_load import R6SchedulerLoadCaseExecutor
from eval_factory.readiness.scheduler_load_builder import SchedulerLoadBuilder
from eval_factory.readiness.scheduler_load_models import (
    SchedulerLoadFaultObservationV1,
)
from eval_factory.readiness.scheduler_load_store import (
    SchedulerLoadMaterialStore,
    SchedulerLoadReportStore,
    SchedulerLoadStoreConflictError,
    SchedulerLoadStoreFaultPoint,
    SchedulerLoadStoreInjectedCrash,
    SchedulerLoadStoreIntegrityError,
    SchedulerLoadStoreLimitError,
    SchedulerLoadStoreTypeError,
    StaticSchedulerLoadStoreFaultInjector,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _audit(created_at: datetime = NOW) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="scheduler-load-store-test",
        governing_versions=(
            VersionBinding(
                component="scheduler-load",
                version="scheduler-load/r8-05-v1",
            ),
        ),
    )


def _policy() -> SchedulerLoadPolicyV2:
    return SchedulerLoadPolicyV2.create(
        lease_duration_seconds=300,
        heartbeat_extension_seconds=60,
        max_private_bytes=100_000_000,
        max_report_bytes=100_000_000,
        audit=_audit(),
    )


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


def _case_result(tmp_path: Path):
    policy = _policy()
    builder = SchedulerLoadBuilder()
    workload = builder.compile_workload(policy=policy, audit=_audit())
    prepared = builder.prepare_case(
        workload.members[0],
        workload=workload,
        policy=policy,
        audit=_audit(),
    )
    result = R6SchedulerLoadCaseExecutor(clock_ns=_Clock()).execute(
        prepared,
        workload=workload,
        policy=policy,
        job_store_path=tmp_path / "factory.sqlite3",
        child_root=tmp_path / "case",
        audit=_audit(),
    )
    return policy, workload, prepared, result


def _private_ref(object_type: str, index: int) -> ObjectRef:
    digest = f"{index + 1:064x}"
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="private-v1",
        object_sha256=digest,
    )


def _report() -> SchedulerLoadReportV2:
    cases = tuple(
        SchedulerLoadCaseSummaryV2.create(
            private_case_result_ref=_private_ref(
                "scheduler-load-case-result",
                index,
            ),
            outcome=SchedulerLoadCaseOutcomeV2.RECOVERED,
            reason_code=SchedulerLoadReasonCodeV2.NONE,
            assigned_fault_point=tuple(SchedulerLoadFaultPointV2)[index % 8],
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
            audit=_audit(),
        )
        for index in range(1000)
    )
    phases = tuple(
        SchedulerLoadPhaseSummaryV2.create(
            phase=phase,
            job_count=1000,
            operation_count=8000,
            elapsed_nanoseconds=(index + 1) * 1_000_000_000,
        )
        for index, phase in enumerate(SchedulerLoadPhaseV2)
    )
    return SchedulerLoadReportV2.create(
        policy_ref=_policy().to_ref(),
        private_workload_ref=_private_ref("scheduler-load-workload", 1001),
        private_result_set_ref=_private_ref("scheduler-load-result-set", 1002),
        case_summaries=cases,
        phase_summaries=phases,
        typed_side_effect_count_before_replay=14_000,
        typed_side_effect_count_after_replay=14_000,
        physical_file_count_before_replay=1,
        physical_file_count_after_replay=1,
        physical_bytes_before_replay=1_000_000,
        physical_bytes_after_replay=1_000_000,
        audit=_audit(),
    )


def test_private_store_round_trips_workload_fault_and_case(tmp_path: Path) -> None:
    policy, workload, prepared, case = _case_result(tmp_path)
    fault = SchedulerLoadFaultObservationV1.create(
        member_ref=prepared.member.to_ref(),
        workload_ref=workload.to_ref(),
        policy_ref=policy.to_ref(),
        fault_point=prepared.member.assigned_fault_point,
        initial_elapsed_nanoseconds=case.initial_elapsed_nanoseconds,
    )
    store = SchedulerLoadMaterialStore(
        tmp_path / "private",
        max_private_bytes=policy.max_private_bytes,
        max_members=1000,
    )

    workload_write = store.put_workload(workload)
    fault_write = store.put_fault_observation(fault)
    case_write = store.put_case_result(case)
    replay = store.put_case_result(
        case.model_copy(
            update={"audit": case.audit.model_copy(update={"created_at": datetime(2026, 8, 3, tzinfo=UTC)})}
        )
    )

    assert workload_write.written is True
    assert fault_write.written is True
    assert case_write.written is True
    assert replay.written is False
    assert store.get_workload(workload.to_ref()) == workload
    assert store.get_fault_observation(fault.to_ref()) == fault
    assert store.get_case_result(case.to_ref()) == case


def test_report_store_round_trips_and_accepts_audit_only_replay(
    tmp_path: Path,
) -> None:
    report = _report()
    store = SchedulerLoadReportStore(
        tmp_path / "report",
        max_report_bytes=100_000_000,
        max_cases=1000,
    )

    first = store.put(report)
    replay = store.put(
        report.model_copy(
            update={"audit": report.audit.model_copy(update={"created_at": datetime(2026, 8, 3, tzinfo=UTC)})}
        )
    )

    assert first.written is True
    assert replay.written is False
    assert store.get(report.to_ref()) == report
    store.verify(report.to_ref())


def test_store_rejects_unsafe_root_and_limits(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(SchedulerLoadStoreTypeError):
        SchedulerLoadMaterialStore(
            symlink,
            max_private_bytes=100,
            max_members=1000,
        )

    report = _report()
    store = SchedulerLoadReportStore(
        tmp_path / "small",
        max_report_bytes=2,
        max_cases=1000,
    )
    with pytest.raises(SchedulerLoadStoreLimitError):
        store.put(report)


def test_report_store_detects_content_corruption(tmp_path: Path) -> None:
    report = _report()
    store = SchedulerLoadReportStore(
        tmp_path / "report",
        max_report_bytes=100_000_000,
        max_cases=1000,
    )
    store.put(report)
    cas_files = tuple((tmp_path / "report" / "cas").rglob("*"))
    blob = next(path for path in cas_files if path.is_file())
    blob.write_bytes(b"corrupt")

    with pytest.raises(SchedulerLoadStoreIntegrityError):
        store.get(report.to_ref())


def test_store_recovers_cas_orphan_and_committed_envelope(
    tmp_path: Path,
) -> None:
    policy = _policy()
    workload = SchedulerLoadBuilder().compile_workload(
        policy=policy,
        audit=_audit(),
    )
    cas_root = tmp_path / "cas-crash"
    crashing = SchedulerLoadMaterialStore(
        cas_root,
        max_private_bytes=policy.max_private_bytes,
        max_members=1000,
        fault_injector=StaticSchedulerLoadStoreFaultInjector(
            crash_points=frozenset({SchedulerLoadStoreFaultPoint.AFTER_CAS_WRITE})
        ),
    )
    with pytest.raises(SchedulerLoadStoreInjectedCrash):
        crashing.put_workload(workload)

    recovered = SchedulerLoadMaterialStore(
        cas_root,
        max_private_bytes=policy.max_private_bytes,
        max_members=1000,
    )
    write = recovered.put_workload(workload)
    assert write.written is True
    assert write.content_blob_written is False

    envelope_root = tmp_path / "envelope-crash"
    crashing = SchedulerLoadMaterialStore(
        envelope_root,
        max_private_bytes=policy.max_private_bytes,
        max_members=1000,
        fault_injector=StaticSchedulerLoadStoreFaultInjector(
            crash_points=frozenset({SchedulerLoadStoreFaultPoint.AFTER_ENVELOPE_WRITE})
        ),
    )
    with pytest.raises(SchedulerLoadStoreInjectedCrash):
        crashing.put_workload(workload)
    recovered = SchedulerLoadMaterialStore(
        envelope_root,
        max_private_bytes=policy.max_private_bytes,
        max_members=1000,
    )
    assert recovered.put_workload(workload).written is False
    assert recovered.get_workload(workload.to_ref()) == workload


def test_store_rejects_wrong_missing_and_malformed_envelopes(
    tmp_path: Path,
) -> None:
    policy = _policy()
    workload = SchedulerLoadBuilder().compile_workload(
        policy=policy,
        audit=_audit(),
    )
    root = tmp_path / "private"
    store = SchedulerLoadMaterialStore(
        root,
        max_private_bytes=policy.max_private_bytes,
        max_members=1000,
    )
    store.put_workload(workload)

    with pytest.raises(SchedulerLoadStoreTypeError):
        store.get_workload(_private_ref("scheduler-load-case-result", 0))
    alias = workload.to_ref().model_copy(update={"object_id": "scheduler-load-workload://sha256/" + "f" * 64})
    with pytest.raises(SchedulerLoadStoreIntegrityError):
        store.get_workload(alias)

    envelope = next(path for path in (root / "envelopes").rglob("*.json") if path.is_file())
    envelope.write_bytes(b"{}")
    with pytest.raises(SchedulerLoadStoreIntegrityError):
        store.get_workload(workload.to_ref())

    report = _report()
    report_store = SchedulerLoadReportStore(
        tmp_path / "missing-report",
        max_report_bytes=100_000_000,
        max_cases=1000,
    )
    with pytest.raises(SchedulerLoadStoreIntegrityError):
        report_store.get(report.to_ref())
    with pytest.raises(SchedulerLoadStoreTypeError):
        report_store.get(_private_ref("scheduler-load-workload", 0))


def test_store_revalidates_limits_and_report_identity(tmp_path: Path) -> None:
    with pytest.raises(SchedulerLoadStoreLimitError):
        SchedulerLoadMaterialStore(
            tmp_path / "bad-members",
            max_private_bytes=100,
            max_members=999,
        )
    with pytest.raises(SchedulerLoadStoreLimitError):
        SchedulerLoadReportStore(
            tmp_path / "bad-cases",
            max_report_bytes=100,
            max_cases=999,
        )

    report = _report()
    root = tmp_path / "report"
    store = SchedulerLoadReportStore(
        root,
        max_report_bytes=100_000_000,
        max_cases=1000,
    )
    store.put(report)
    limited = SchedulerLoadReportStore(
        root,
        max_report_bytes=100,
        max_cases=1000,
    )
    with pytest.raises(SchedulerLoadStoreLimitError):
        limited.get(report.to_ref())
    stale = report.model_copy(
        update={
            "report_id": "scheduler-load-report://sha256/" + "f" * 64,
            "report_sha256": "f" * 64,
        }
    )
    with pytest.raises(SchedulerLoadStoreConflictError):
        store.put(stale)
