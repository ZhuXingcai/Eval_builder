from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.observability_v2 import (
    BatchAuditOutcomeV2,
    BatchAuditReportV2,
    batch_audit_report_v2_ref,
    work_attempt_metrics_v2_ref,
)
from eval_factory.contracts.orchestration import (
    JobStatus,
    StageRunStatus,
)
from eval_factory.contracts.orchestration_v2 import (
    WorkLeaseEventKindV2,
    WorkReadinessV2,
    resolved_dataset_job_plan_v2_ref,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
    work_control_policy_v2_ref,
    work_lease_event_v2_ref,
    work_lease_v2_ref,
    work_readiness_snapshot_v2_ref,
)
from eval_factory.contracts.scheduler_load_v2 import (
    SchedulerLoadCaseOutcomeV2,
    SchedulerLoadFaultPointV2,
    SchedulerLoadPhaseSummaryV2,
    SchedulerLoadPhaseV2,
    SchedulerLoadPolicyV2,
    SchedulerLoadReasonCodeV2,
    SchedulerLoadReportV2,
)
from eval_factory.orchestration import (
    BatchObservabilityIntegrityError,
    BatchObservabilityService,
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    JobStoreError,
    WorkControlService,
    WorkReadinessEvaluator,
    stage_result_record_ref,
    stage_run_record_ref,
)
from eval_factory.readiness.scheduler_load_builder import SchedulerLoadBuilder
from eval_factory.readiness.scheduler_load_models import (
    PreparedSchedulerLoadCase,
    SchedulerLoadCaseResultV1,
    SchedulerLoadFaultObservationV1,
    SchedulerLoadResultSetV1,
    SchedulerLoadRunLedgerV1,
    SchedulerLoadWorkloadV1,
)


class SchedulerLoadRunError(RuntimeError):
    pass


class SchedulerLoadRunConflictError(SchedulerLoadRunError):
    pass


class SchedulerLoadTimingError(SchedulerLoadRunError):
    pass


class SchedulerLoadInjectedCrash(SchedulerLoadRunError):
    pass


class SchedulerLoadExpectedFaultNotObservedError(SchedulerLoadRunError):
    pass


class SchedulerLoadResumeFailedError(SchedulerLoadRunError):
    def __init__(self, initial_elapsed_nanoseconds: int | None = None) -> None:
        super().__init__("scheduler load recovery failed")
        self.initial_elapsed_nanoseconds = initial_elapsed_nanoseconds


class SchedulerLoadMaterialIntegrityError(SchedulerLoadRunError):
    def __init__(self, initial_elapsed_nanoseconds: int | None = None) -> None:
        super().__init__("scheduler load material failed integrity validation")
        self.initial_elapsed_nanoseconds = initial_elapsed_nanoseconds


class SchedulerLoadCaseExecutor(Protocol):
    def execute(
        self,
        prepared: PreparedSchedulerLoadCase,
        *,
        workload: SchedulerLoadWorkloadV1,
        policy: SchedulerLoadPolicyV2,
        job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> SchedulerLoadCaseResultV1: ...


@dataclass(frozen=True, slots=True)
class SchedulerLoadRunResult:
    run_ledger: SchedulerLoadRunLedgerV1
    fault_observations: tuple[SchedulerLoadFaultObservationV1, ...]
    case_results: tuple[SchedulerLoadCaseResultV1, ...]
    result_set: SchedulerLoadResultSetV1
    report: SchedulerLoadReportV2


@dataclass(frozen=True, slots=True)
class _AdvanceResult:
    observability_report: BatchAuditReportV2
    operation_count: int


@dataclass(frozen=True, slots=True)
class _Evidence:
    readiness_ref: ObjectRef | None
    lease_ref: ObjectRef | None
    stage_run_ref: ObjectRef | None
    stage_result_ref: ObjectRef | None
    attempt_metrics_ref: ObjectRef | None
    terminal_lease_event_ref: ObjectRef | None
    observability_report_ref: ObjectRef | None
    output_ref: ObjectRef | None
    attempt_count: int
    unexpected_retry_count: int
    duplicate_immutable_fact_count: int
    stage_run_count: int
    stage_result_count: int
    attempt_metrics_count: int
    completion_witness_count: int
    observability_complete: bool
    typed_side_effect_count: int


@dataclass(frozen=True, slots=True)
class _PhysicalSnapshot:
    file_count: int
    total_bytes: int


class R6SchedulerLoadCaseExecutor:
    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self._clock_ns = clock_ns

    def execute(
        self,
        prepared: PreparedSchedulerLoadCase,
        *,
        workload: SchedulerLoadWorkloadV1,
        policy: SchedulerLoadPolicyV2,
        job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> SchedulerLoadCaseResultV1:
        root = _private_root(child_root)
        fault = _read_fault_observation(
            root,
            prepared=prepared,
            workload=workload,
            policy=policy,
        )
        if fault is None:
            initial_started = self._clock_ns()
            try:
                self._advance(
                    prepared,
                    policy=policy,
                    job_store_path=job_store_path,
                    audit=audit,
                    inject_fault=True,
                )
            except SchedulerLoadInjectedCrash:
                initial_elapsed = _elapsed(
                    self._clock_ns,
                    initial_started,
                )
                fault = SchedulerLoadFaultObservationV1.create(
                    member_ref=prepared.member.to_ref(),
                    workload_ref=workload.to_ref(),
                    policy_ref=policy.to_ref(),
                    fault_point=prepared.member.assigned_fault_point,
                    initial_elapsed_nanoseconds=initial_elapsed,
                )
                _admit_fault_observation(root, fault)
            else:
                raise SchedulerLoadExpectedFaultNotObservedError

        recovery_started = self._clock_ns()
        try:
            self._advance(
                prepared,
                policy=policy,
                job_store_path=job_store_path,
                audit=audit,
                inject_fault=False,
            )
        except (
            BatchObservabilityIntegrityError,
            IdempotencyConflictError,
            ImmutableResultError,
            SchedulerLoadRunConflictError,
        ) as exc:
            raise SchedulerLoadMaterialIntegrityError(fault.initial_elapsed_nanoseconds) from exc
        except JobStoreError as exc:
            raise SchedulerLoadResumeFailedError(fault.initial_elapsed_nanoseconds) from exc
        recovery_elapsed = _elapsed(self._clock_ns, recovery_started)
        store = JobStore(job_store_path)
        before = _read_evidence(store, prepared)
        physical_before = _physical_snapshot(job_store_path)

        replay_started = self._clock_ns()
        self._advance(
            prepared,
            policy=policy,
            job_store_path=job_store_path,
            audit=audit,
            inject_fault=False,
        )
        replay_elapsed = _elapsed(self._clock_ns, replay_started)
        replay_store = JobStore(job_store_path)
        after = _read_evidence(replay_store, prepared)
        physical_after = _physical_snapshot(job_store_path)

        replay_stable = before == after and physical_before.file_count == physical_after.file_count
        outcome, reason = _classify(
            after,
            fault_observed=True,
            resume_succeeded=_complete_evidence(before),
            replay_stable=replay_stable,
        )
        return SchedulerLoadCaseResultV1.create(
            member_ref=prepared.member.to_ref(),
            workload_ref=workload.to_ref(),
            policy_ref=policy.to_ref(),
            dataset_job_spec_ref=prepared.dataset_job_spec_ref,
            resolved_plan_ref=resolved_dataset_job_plan_v2_ref(prepared.resolved_plan),
            work_graph_ref=resolved_job_work_graph_v2_ref(prepared.work_graph),
            control_policy_ref=work_control_policy_v2_ref(prepared.control_policy),
            work_unit_ref=resolved_work_unit_v2_ref(prepared.root_work_unit),
            readiness_ref=after.readiness_ref,
            lease_ref=after.lease_ref,
            stage_run_ref=after.stage_run_ref,
            stage_result_ref=after.stage_result_ref,
            attempt_metrics_ref=after.attempt_metrics_ref,
            terminal_lease_event_ref=after.terminal_lease_event_ref,
            observability_report_ref=after.observability_report_ref,
            output_ref=after.output_ref,
            job_id=prepared.job_spec.job_id,
            item_id=prepared.work_graph.item_ids[0],
            assigned_fault_point=prepared.member.assigned_fault_point,
            fault_observed=True,
            resume_succeeded=_complete_evidence(before),
            replay_stable=replay_stable,
            attempt_count=after.attempt_count,
            unexpected_retry_count=after.unexpected_retry_count,
            duplicate_immutable_fact_count=(after.duplicate_immutable_fact_count),
            stage_run_count=after.stage_run_count,
            stage_result_count=after.stage_result_count,
            attempt_metrics_count=after.attempt_metrics_count,
            completion_witness_count=after.completion_witness_count,
            observability_complete=after.observability_complete,
            initial_elapsed_nanoseconds=fault.initial_elapsed_nanoseconds,
            recovery_elapsed_nanoseconds=recovery_elapsed,
            replay_elapsed_nanoseconds=replay_elapsed,
            typed_side_effect_count_before_replay=(before.typed_side_effect_count),
            typed_side_effect_count_after_replay=(after.typed_side_effect_count),
            physical_file_count_before_replay=physical_before.file_count,
            physical_file_count_after_replay=physical_after.file_count,
            physical_bytes_before_replay=physical_before.total_bytes,
            physical_bytes_after_replay=physical_after.total_bytes,
            outcome=outcome,
            reason_code=reason,
            audit=audit,
        )

    def _advance(
        self,
        prepared: PreparedSchedulerLoadCase,
        *,
        policy: SchedulerLoadPolicyV2,
        job_store_path: Path,
        audit: ContractAudit,
        inject_fault: bool,
    ) -> _AdvanceResult:
        store = JobStore(job_store_path)
        control = WorkControlService(store)
        operations = 0

        store.create_planned_job(
            prepared.job_spec,
            prepared.resolved_plan,
        )
        operations += 1
        _maybe_crash(
            prepared,
            SchedulerLoadFaultPointV2.AFTER_JOB_CREATE,
            inject_fault=inject_fault,
        )

        graph = store.create_job_work_graph(
            prepared.work_graph,
            idempotency_key=_key(prepared, "create-work-graph"),
        )
        if graph != prepared.work_graph:
            raise SchedulerLoadRunConflictError("stored scheduler load work graph differs")
        operations += 1
        _maybe_crash(
            prepared,
            SchedulerLoadFaultPointV2.AFTER_WORK_GRAPH_CREATE,
            inject_fault=inject_fault,
        )

        running = store.transition_job(
            prepared.job_spec.job_id,
            JobStatus.RUNNING,
            expected_version=0,
            idempotency_key=_key(prepared, "start-job"),
        )
        if running.status is not JobStatus.RUNNING:
            raise SchedulerLoadRunConflictError("scheduler load Job did not enter RUNNING")
        operations += 1
        _maybe_crash(
            prepared,
            SchedulerLoadFaultPointV2.AFTER_JOB_START,
            inject_fault=inject_fault,
        )

        readiness = WorkReadinessEvaluator().evaluate(
            graph=prepared.work_graph,
            work_unit=prepared.root_work_unit,
            stage_completions=(),
            item_records=store.list_items(prepared.job_spec.job_id),
            audit=audit,
        )
        if readiness.readiness is not WorkReadinessV2.READY:
            raise SchedulerLoadRunConflictError("scheduler load root work is not READY")
        stored_readiness = store.record_work_readiness(
            readiness,
            idempotency_key=_key(prepared, "record-readiness"),
        )
        operations += 1
        _maybe_crash(
            prepared,
            SchedulerLoadFaultPointV2.AFTER_READINESS_RECORD,
            inject_fault=inject_fault,
        )

        stored_policy = control.bind_policy(
            graph=prepared.work_graph,
            lease_duration_seconds=policy.lease_duration_seconds,
            heartbeat_extension_seconds=policy.heartbeat_extension_seconds,
            max_attempts=policy.max_attempts,
            retry_delay_seconds=policy.retry_delay_seconds,
            retry_lease_expiry=policy.retry_lease_expiry,
            audit=audit,
            idempotency_key=_key(prepared, "bind-control-policy"),
        )
        if stored_policy != prepared.control_policy:
            raise SchedulerLoadRunConflictError("stored scheduler control policy differs")
        operations += 1
        _maybe_crash(
            prepared,
            SchedulerLoadFaultPointV2.AFTER_CONTROL_POLICY_BIND,
            inject_fault=inject_fault,
        )

        lease = control.acquire(
            graph=prepared.work_graph,
            work_unit=prepared.root_work_unit,
            readiness_snapshot=stored_readiness,
            policy=stored_policy,
            holder_ref=prepared.holder_ref,
            retry_decision=None,
            audit=audit,
            idempotency_key=_key(prepared, "acquire-work"),
        )
        operations += 1
        _maybe_crash(
            prepared,
            SchedulerLoadFaultPointV2.AFTER_LEASE_ACQUIRE,
            inject_fault=inject_fault,
        )

        event = control.complete_stage(
            lease=lease,
            policy=stored_policy,
            holder_ref=prepared.holder_ref,
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=(prepared.output_ref,),
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            audit=audit,
            idempotency_key=_key(prepared, "complete-work"),
        )
        if event.event_kind is not WorkLeaseEventKindV2.SUCCEEDED:
            raise SchedulerLoadRunConflictError("scheduler load work did not succeed")
        operations += 1
        _maybe_crash(
            prepared,
            SchedulerLoadFaultPointV2.AFTER_STAGE_COMPLETION,
            inject_fault=inject_fault,
        )

        report = (
            BatchObservabilityService(store)
            .refresh_job(
                job_id=prepared.job_spec.job_id,
                audit=audit,
                idempotency_key=_key(prepared, "refresh-observability"),
            )
            .report
        )
        operations += 1
        _maybe_crash(
            prepared,
            SchedulerLoadFaultPointV2.AFTER_OBSERVABILITY_REFRESH,
            inject_fault=inject_fault,
        )
        return _AdvanceResult(
            observability_report=report,
            operation_count=operations,
        )

    @staticmethod
    def read_observability(
        store: JobStore,
        job_id: str,
    ) -> BatchAuditReportV2:
        return BatchObservabilityService(store).get_audit_report(job_id)


class SchedulerLoadRunner:
    def __init__(
        self,
        *,
        builder: SchedulerLoadBuilder | None = None,
        executor: SchedulerLoadCaseExecutor | None = None,
        verify_job_store: bool = True,
    ) -> None:
        self._builder = builder or SchedulerLoadBuilder()
        self._executor = executor or R6SchedulerLoadCaseExecutor()
        self._verify_job_store = verify_job_store

    def run(
        self,
        *,
        workload: SchedulerLoadWorkloadV1,
        policy: SchedulerLoadPolicyV2,
        job_store_path: Path,
        run_root: Path,
        audit: ContractAudit,
    ) -> SchedulerLoadRunResult:
        root = _private_root(run_root)
        prepared = self._builder.prepare_cases(
            workload,
            policy=policy,
            audit=audit,
        )
        job_store_ref = _job_store_ref(job_store_path)
        ledger = SchedulerLoadRunLedgerV1.create(
            workload_ref=workload.to_ref(),
            policy_ref=policy.to_ref(),
            job_store_ref=job_store_ref,
            prepared=prepared,
        )
        _admit_ledger(root, ledger)

        results: list[SchedulerLoadCaseResultV1] = []
        faults: list[SchedulerLoadFaultObservationV1] = []
        for case in prepared:
            result_path = _case_result_path(root, case)
            if result_path.exists():
                result = _read_case_result(
                    result_path,
                    prepared=case,
                    workload=workload,
                    policy=policy,
                )
                if self._verify_job_store:
                    current = _read_evidence(JobStore(job_store_path), case)
                    _validate_persisted_evidence(result, current)
            else:
                started = perf_counter_ns()
                try:
                    result = self._executor.execute(
                        case,
                        workload=workload,
                        policy=policy,
                        job_store_path=job_store_path,
                        child_root=_case_root(root, case),
                        audit=audit,
                    )
                    _validate_case_binding(
                        case,
                        result,
                        workload=workload,
                        policy=policy,
                    )
                except SchedulerLoadTimingError:
                    raise
                except SchedulerLoadExpectedFaultNotObservedError:
                    result = _closed_case_result(
                        case,
                        workload=workload,
                        policy=policy,
                        audit=audit,
                        elapsed_nanoseconds=max(
                            1,
                            perf_counter_ns() - started,
                        ),
                        outcome=SchedulerLoadCaseOutcomeV2.FAILED,
                        reason_code=(SchedulerLoadReasonCodeV2.EXPECTED_FAULT_NOT_OBSERVED),
                        fault_observed=False,
                    )
                except SchedulerLoadResumeFailedError as exc:
                    result = _closed_case_result(
                        case,
                        workload=workload,
                        policy=policy,
                        audit=audit,
                        elapsed_nanoseconds=(
                            exc.initial_elapsed_nanoseconds
                            or max(
                                1,
                                perf_counter_ns() - started,
                            )
                        ),
                        outcome=SchedulerLoadCaseOutcomeV2.FAILED,
                        reason_code=SchedulerLoadReasonCodeV2.RESUME_FAILED,
                        fault_observed=True,
                    )
                except SchedulerLoadMaterialIntegrityError as exc:
                    result = _closed_case_result(
                        case,
                        workload=workload,
                        policy=policy,
                        audit=audit,
                        elapsed_nanoseconds=(
                            exc.initial_elapsed_nanoseconds
                            or max(
                                1,
                                perf_counter_ns() - started,
                            )
                        ),
                        outcome=SchedulerLoadCaseOutcomeV2.INFRASTRUCTURE_ERROR,
                        reason_code=(SchedulerLoadReasonCodeV2.MATERIAL_INTEGRITY_ERROR),
                        fault_observed=exc.initial_elapsed_nanoseconds is not None,
                    )
                except SchedulerLoadRunConflictError:
                    result = _closed_case_result(
                        case,
                        workload=workload,
                        policy=policy,
                        audit=audit,
                        elapsed_nanoseconds=max(
                            1,
                            perf_counter_ns() - started,
                        ),
                        outcome=SchedulerLoadCaseOutcomeV2.INFRASTRUCTURE_ERROR,
                        reason_code=(SchedulerLoadReasonCodeV2.MATERIAL_INTEGRITY_ERROR),
                        fault_observed=False,
                    )
                except Exception:
                    result = _closed_case_result(
                        case,
                        workload=workload,
                        policy=policy,
                        audit=audit,
                        elapsed_nanoseconds=max(
                            1,
                            perf_counter_ns() - started,
                        ),
                    )
                _admit_case_result(result_path, result)
            fault = SchedulerLoadFaultObservationV1.create(
                member_ref=case.member.to_ref(),
                workload_ref=workload.to_ref(),
                policy_ref=policy.to_ref(),
                fault_point=case.member.assigned_fault_point,
                initial_elapsed_nanoseconds=max(
                    1,
                    result.initial_elapsed_nanoseconds,
                ),
            )
            if result.fault_observed:
                _admit_fault_observation(_case_root(root, case), fault)
                faults.append(fault)
            results.append(result)

        case_results = tuple(results)
        summaries = tuple(result.to_public_summary(audit=audit) for result in case_results)
        result_set = SchedulerLoadResultSetV1.create(
            workload_ref=workload.to_ref(),
            policy_ref=policy.to_ref(),
            case_results=case_results,
            case_summaries=summaries,
            audit=audit,
        )
        phase_summaries = _phase_summaries(case_results)
        last = case_results[-1]
        report = SchedulerLoadReportV2.create(
            policy_ref=policy.to_ref(),
            private_workload_ref=workload.to_ref(),
            private_result_set_ref=result_set.to_ref(),
            case_summaries=summaries,
            phase_summaries=phase_summaries,
            typed_side_effect_count_before_replay=sum(
                result.typed_side_effect_count_before_replay for result in case_results
            ),
            typed_side_effect_count_after_replay=sum(
                result.typed_side_effect_count_after_replay for result in case_results
            ),
            physical_file_count_before_replay=(last.physical_file_count_before_replay),
            physical_file_count_after_replay=(last.physical_file_count_after_replay),
            physical_bytes_before_replay=(last.physical_bytes_before_replay),
            physical_bytes_after_replay=last.physical_bytes_after_replay,
            audit=audit,
        )
        return SchedulerLoadRunResult(
            run_ledger=ledger,
            fault_observations=tuple(faults),
            case_results=case_results,
            result_set=result_set,
            report=report,
        )


def _read_evidence(
    store: JobStore,
    prepared: PreparedSchedulerLoadCase,
) -> _Evidence:
    job_id = prepared.job_spec.job_id
    job = store.get_job(job_id)
    plan = store.get_resolved_job_plan(job_id)
    graph = store.get_job_work_graph(job_id)
    items = store.list_items(job_id)
    units = store.list_work_units(job_id=job_id)
    readiness = store.list_work_readiness(prepared.root_work_unit.resolved_work_unit_id)
    control_policy = store.get_work_control_policy(job_id)
    leases = store.list_work_leases(job_id=job_id)
    stages = store.list_stage_runs(job_id=job_id)
    metrics = store.list_work_attempt_metrics(job_id=job_id)
    outbox = store.list_outbox()
    retry = store.get_work_retry_decision_optional(prepared.root_work_unit.resolved_work_unit_id)
    report = R6SchedulerLoadCaseExecutor.read_observability(store, job_id)

    lease_events = store.list_work_lease_events(leases[0].work_lease_id) if len(leases) == 1 else ()
    stage_result = store.get_stage_result_for_run(stages[0].stage_run_id) if len(stages) == 1 else None
    completion_witnesses = tuple(
        event
        for event in outbox
        if len(stages) == 1
        and event.aggregate_id == stages[0].stage_run_id
        and event.event_type == "work-lease-closed"
    )
    duplicates = sum(
        max(0, count - 1)
        for count in (
            len(items),
            len(units),
            len(readiness),
            len(leases),
            len(stages),
            len(metrics),
            len(lease_events),
            len(completion_witnesses),
        )
    )
    output_ref = (
        stage_result.output_refs[0]
        if stage_result is not None and len(stage_result.output_refs) == 1
        else None
    )
    terminal_event = lease_events[0] if len(lease_events) == 1 else None
    typed_count = (
        1
        + 1
        + 1
        + len(items)
        + len(units)
        + len(readiness)
        + 1
        + len(leases)
        + len(lease_events)
        + len(stages)
        + (1 if stage_result is not None else 0)
        + len(metrics)
        + len(completion_witnesses)
        + 1
    )
    if (
        job.status is not JobStatus.RUNNING
        or plan != prepared.resolved_plan
        or graph != prepared.work_graph
        or control_policy != prepared.control_policy
    ):
        duplicates += 1
    if output_ref != prepared.output_ref:
        duplicates += 1
    if retry is not None:
        duplicates += 1
    return _Evidence(
        readiness_ref=(work_readiness_snapshot_v2_ref(readiness[0]) if len(readiness) == 1 else None),
        lease_ref=work_lease_v2_ref(leases[0]) if len(leases) == 1 else None,
        stage_run_ref=(stage_run_record_ref(stages[0]) if len(stages) == 1 else None),
        stage_result_ref=(stage_result_record_ref(stage_result) if stage_result is not None else None),
        attempt_metrics_ref=(work_attempt_metrics_v2_ref(metrics[0]) if len(metrics) == 1 else None),
        terminal_lease_event_ref=(
            work_lease_event_v2_ref(terminal_event) if terminal_event is not None else None
        ),
        observability_report_ref=batch_audit_report_v2_ref(report),
        output_ref=output_ref,
        attempt_count=len(leases),
        unexpected_retry_count=max(0, len(leases) - 1) + (1 if retry is not None else 0),
        duplicate_immutable_fact_count=duplicates,
        stage_run_count=len(stages),
        stage_result_count=1 if stage_result is not None else 0,
        attempt_metrics_count=len(metrics),
        completion_witness_count=len(completion_witnesses),
        observability_complete=(report.outcome is BatchAuditOutcomeV2.COMPLETE),
        typed_side_effect_count=typed_count,
    )


def _complete_evidence(value: _Evidence) -> bool:
    return (
        value.readiness_ref is not None
        and value.lease_ref is not None
        and value.stage_run_ref is not None
        and value.stage_result_ref is not None
        and value.attempt_metrics_ref is not None
        and value.terminal_lease_event_ref is not None
        and value.observability_report_ref is not None
        and value.output_ref is not None
        and value.attempt_count == 1
        and value.unexpected_retry_count == 0
        and value.duplicate_immutable_fact_count == 0
        and value.stage_run_count == 1
        and value.stage_result_count == 1
        and value.attempt_metrics_count == 1
        and value.completion_witness_count == 1
        and value.observability_complete
    )


def _classify(
    evidence: _Evidence,
    *,
    fault_observed: bool,
    resume_succeeded: bool,
    replay_stable: bool,
) -> tuple[SchedulerLoadCaseOutcomeV2, SchedulerLoadReasonCodeV2]:
    if not fault_observed:
        return (
            SchedulerLoadCaseOutcomeV2.FAILED,
            SchedulerLoadReasonCodeV2.EXPECTED_FAULT_NOT_OBSERVED,
        )
    if not resume_succeeded:
        return (
            SchedulerLoadCaseOutcomeV2.FAILED,
            SchedulerLoadReasonCodeV2.RESUME_FAILED,
        )
    if evidence.unexpected_retry_count:
        return (
            SchedulerLoadCaseOutcomeV2.FAILED,
            SchedulerLoadReasonCodeV2.UNEXPECTED_RETRY,
        )
    if evidence.duplicate_immutable_fact_count:
        return (
            SchedulerLoadCaseOutcomeV2.FAILED,
            SchedulerLoadReasonCodeV2.DUPLICATE_IMMUTABLE_FACT,
        )
    if not evidence.observability_complete:
        return (
            SchedulerLoadCaseOutcomeV2.FAILED,
            SchedulerLoadReasonCodeV2.OBSERVABILITY_INCOMPLETE,
        )
    if not _complete_evidence(evidence):
        return (
            SchedulerLoadCaseOutcomeV2.FAILED,
            SchedulerLoadReasonCodeV2.TERMINAL_EVIDENCE_MISMATCH,
        )
    if not replay_stable:
        return (
            SchedulerLoadCaseOutcomeV2.FAILED,
            SchedulerLoadReasonCodeV2.REPLAY_MISMATCH,
        )
    return (
        SchedulerLoadCaseOutcomeV2.RECOVERED,
        SchedulerLoadReasonCodeV2.NONE,
    )


def _maybe_crash(
    prepared: PreparedSchedulerLoadCase,
    point: SchedulerLoadFaultPointV2,
    *,
    inject_fault: bool,
) -> None:
    if inject_fault and prepared.member.assigned_fault_point is point:
        raise SchedulerLoadInjectedCrash(f"injected scheduler load crash at {point.value}")


def _key(
    prepared: PreparedSchedulerLoadCase,
    operation: str,
) -> str:
    digest = prepared.member.case_key.rsplit("/", 1)[-1]
    return f"scheduler-load-{operation}-{digest}"


def _elapsed(
    clock_ns: Callable[[], int],
    started: int,
) -> int:
    ended = clock_ns()
    if ended <= started:
        raise SchedulerLoadTimingError("scheduler load monotonic clock did not advance")
    return ended - started


def _private_root(root: Path) -> Path:
    candidate = root.expanduser()
    if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
        raise SchedulerLoadRunError("scheduler load child root must be a non-symlink directory")
    resolved = candidate.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _fault_path(root: Path) -> Path:
    return root / "fault-observation.json"


def _read_fault_observation(
    root: Path,
    *,
    prepared: PreparedSchedulerLoadCase,
    workload: SchedulerLoadWorkloadV1,
    policy: SchedulerLoadPolicyV2,
) -> SchedulerLoadFaultObservationV1 | None:
    path = _fault_path(root)
    if not path.exists():
        return None
    try:
        value = SchedulerLoadFaultObservationV1.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise SchedulerLoadRunConflictError("scheduler load fault observation is invalid") from exc
    if (
        value.member_ref != prepared.member.to_ref()
        or value.workload_ref != workload.to_ref()
        or value.policy_ref != policy.to_ref()
        or value.fault_point is not prepared.member.assigned_fault_point
    ):
        raise SchedulerLoadRunConflictError("scheduler load fault observation conflicts with this case")
    return value


def _admit_fault_observation(
    root: Path,
    value: SchedulerLoadFaultObservationV1,
) -> None:
    path = _fault_path(root)
    encoded = value.canonical_json() + b"\n"
    if path.exists():
        observed = SchedulerLoadFaultObservationV1.model_validate_json(path.read_bytes())
        if observed != value:
            raise SchedulerLoadRunConflictError("scheduler load fault observation already differs")
        return
    temporary = root / (f".fault-{value.fault_observation_sha256}-{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed = SchedulerLoadFaultObservationV1.model_validate_json(path.read_bytes())
            if observed != value:
                raise SchedulerLoadRunConflictError(
                    "scheduler load fault observation raced with another value"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _physical_snapshot(job_store_path: Path) -> _PhysicalSnapshot:
    candidates = tuple(
        path
        for path in (
            job_store_path,
            job_store_path.with_name(f"{job_store_path.name}-wal"),
            job_store_path.with_name(f"{job_store_path.name}-shm"),
            job_store_path.with_name(f"{job_store_path.name}-journal"),
        )
        if path.exists()
    )
    return _PhysicalSnapshot(
        file_count=len(candidates),
        total_bytes=sum(path.stat().st_size for path in candidates),
    )


def _phase_summaries(
    results: tuple[SchedulerLoadCaseResultV1, ...],
) -> tuple[SchedulerLoadPhaseSummaryV2, ...]:
    initial_operations = sum(
        tuple(SchedulerLoadFaultPointV2).index(result.assigned_fault_point) + 1 for result in results
    )
    return (
        SchedulerLoadPhaseSummaryV2.create(
            phase=SchedulerLoadPhaseV2.SYNTHETIC_INITIAL,
            job_count=len(results),
            operation_count=initial_operations,
            elapsed_nanoseconds=max(
                1,
                sum(result.initial_elapsed_nanoseconds for result in results),
            ),
        ),
        SchedulerLoadPhaseSummaryV2.create(
            phase=SchedulerLoadPhaseV2.CRASH_RECOVERY,
            job_count=len(results),
            operation_count=len(results) * 8,
            elapsed_nanoseconds=max(
                1,
                sum(result.recovery_elapsed_nanoseconds for result in results),
            ),
        ),
        SchedulerLoadPhaseSummaryV2.create(
            phase=SchedulerLoadPhaseV2.EXACT_REPLAY,
            job_count=len(results),
            operation_count=len(results) * 8,
            elapsed_nanoseconds=max(
                1,
                sum(result.replay_elapsed_nanoseconds for result in results),
            ),
        ),
    )


def _case_root(
    root: Path,
    prepared: PreparedSchedulerLoadCase,
) -> Path:
    digest = prepared.member.case_key.rsplit("/", 1)[-1]
    path = root / "cases" / digest
    path.mkdir(parents=True, exist_ok=True)
    return path


def _case_result_path(
    root: Path,
    prepared: PreparedSchedulerLoadCase,
) -> Path:
    digest = prepared.member.case_key.rsplit("/", 1)[-1]
    path = root / "case-results" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_case_result(
    path: Path,
    *,
    prepared: PreparedSchedulerLoadCase,
    workload: SchedulerLoadWorkloadV1,
    policy: SchedulerLoadPolicyV2,
) -> SchedulerLoadCaseResultV1:
    try:
        result = SchedulerLoadCaseResultV1.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise SchedulerLoadRunConflictError("scheduler load case result is invalid") from exc
    _validate_case_binding(
        prepared,
        result,
        workload=workload,
        policy=policy,
    )
    return result


def _validate_case_binding(
    prepared: PreparedSchedulerLoadCase,
    result: SchedulerLoadCaseResultV1,
    *,
    workload: SchedulerLoadWorkloadV1,
    policy: SchedulerLoadPolicyV2,
) -> None:
    expected = (
        result.member_ref == prepared.member.to_ref()
        and result.workload_ref == workload.to_ref()
        and result.policy_ref == policy.to_ref()
        and result.dataset_job_spec_ref == prepared.dataset_job_spec_ref
        and result.resolved_plan_ref == resolved_dataset_job_plan_v2_ref(prepared.resolved_plan)
        and result.work_graph_ref == resolved_job_work_graph_v2_ref(prepared.work_graph)
        and result.control_policy_ref == work_control_policy_v2_ref(prepared.control_policy)
        and result.work_unit_ref == resolved_work_unit_v2_ref(prepared.root_work_unit)
        and result.job_id == prepared.job_spec.job_id
        and result.item_id == prepared.work_graph.item_ids[0]
        and result.assigned_fault_point is prepared.member.assigned_fault_point
    )
    if not expected:
        raise SchedulerLoadRunConflictError("scheduler load case result binding differs")


def _validate_persisted_evidence(
    result: SchedulerLoadCaseResultV1,
    evidence: _Evidence,
) -> None:
    if (
        result.readiness_ref != evidence.readiness_ref
        or result.lease_ref != evidence.lease_ref
        or result.stage_run_ref != evidence.stage_run_ref
        or result.stage_result_ref != evidence.stage_result_ref
        or result.attempt_metrics_ref != evidence.attempt_metrics_ref
        or result.terminal_lease_event_ref != evidence.terminal_lease_event_ref
        or result.observability_report_ref != evidence.observability_report_ref
        or result.output_ref != evidence.output_ref
        or result.attempt_count != evidence.attempt_count
        or result.unexpected_retry_count != evidence.unexpected_retry_count
        or result.duplicate_immutable_fact_count != evidence.duplicate_immutable_fact_count
        or result.stage_run_count != evidence.stage_run_count
        or result.stage_result_count != evidence.stage_result_count
        or result.attempt_metrics_count != evidence.attempt_metrics_count
        or result.completion_witness_count != evidence.completion_witness_count
        or result.observability_complete != evidence.observability_complete
        or result.typed_side_effect_count_after_replay != evidence.typed_side_effect_count
    ):
        raise SchedulerLoadRunConflictError("persisted scheduler load evidence differs from JobStore")


def _admit_case_result(
    path: Path,
    result: SchedulerLoadCaseResultV1,
) -> None:
    _atomic_first_authority(
        path,
        result.canonical_json() + b"\n",
        label="scheduler load case result",
    )


def _closed_case_result(
    prepared: PreparedSchedulerLoadCase,
    *,
    workload: SchedulerLoadWorkloadV1,
    policy: SchedulerLoadPolicyV2,
    audit: ContractAudit,
    elapsed_nanoseconds: int,
    outcome: SchedulerLoadCaseOutcomeV2 = SchedulerLoadCaseOutcomeV2.INFRASTRUCTURE_ERROR,
    reason_code: SchedulerLoadReasonCodeV2 = SchedulerLoadReasonCodeV2.INTERNAL_ERROR,
    fault_observed: bool = False,
) -> SchedulerLoadCaseResultV1:
    return SchedulerLoadCaseResultV1.create(
        member_ref=prepared.member.to_ref(),
        workload_ref=workload.to_ref(),
        policy_ref=policy.to_ref(),
        dataset_job_spec_ref=prepared.dataset_job_spec_ref,
        resolved_plan_ref=resolved_dataset_job_plan_v2_ref(prepared.resolved_plan),
        work_graph_ref=resolved_job_work_graph_v2_ref(prepared.work_graph),
        control_policy_ref=work_control_policy_v2_ref(prepared.control_policy),
        work_unit_ref=resolved_work_unit_v2_ref(prepared.root_work_unit),
        readiness_ref=None,
        lease_ref=None,
        stage_run_ref=None,
        stage_result_ref=None,
        attempt_metrics_ref=None,
        terminal_lease_event_ref=None,
        observability_report_ref=None,
        output_ref=None,
        job_id=prepared.job_spec.job_id,
        item_id=prepared.work_graph.item_ids[0],
        assigned_fault_point=prepared.member.assigned_fault_point,
        fault_observed=fault_observed,
        resume_succeeded=False,
        replay_stable=False,
        attempt_count=0,
        unexpected_retry_count=0,
        duplicate_immutable_fact_count=0,
        stage_run_count=0,
        stage_result_count=0,
        attempt_metrics_count=0,
        completion_witness_count=0,
        observability_complete=False,
        initial_elapsed_nanoseconds=elapsed_nanoseconds,
        recovery_elapsed_nanoseconds=0,
        replay_elapsed_nanoseconds=0,
        typed_side_effect_count_before_replay=0,
        typed_side_effect_count_after_replay=0,
        physical_file_count_before_replay=0,
        physical_file_count_after_replay=0,
        physical_bytes_before_replay=0,
        physical_bytes_after_replay=0,
        outcome=outcome,
        reason_code=reason_code,
        audit=audit,
    )


def _job_store_ref(job_store_path: Path) -> ObjectRef:
    digest = hashlib.sha256(str(job_store_path.expanduser().resolve()).encode()).hexdigest()
    return ObjectRef(
        object_type="scheduler-load-job-store",
        object_id=f"scheduler-load-job-store://sha256/{digest}",
        object_version="private-v1",
        object_sha256=digest,
    )


def _admit_ledger(
    root: Path,
    ledger: SchedulerLoadRunLedgerV1,
) -> None:
    path = root / "run-ledger.json"
    encoded = ledger.canonical_json() + b"\n"
    if path.exists():
        try:
            observed = SchedulerLoadRunLedgerV1.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise SchedulerLoadRunConflictError("scheduler load run ledger is invalid") from exc
        if observed != ledger:
            raise SchedulerLoadRunConflictError("scheduler load run ledger conflicts with this run")
        return
    _atomic_first_authority(
        path,
        encoded,
        label="scheduler load run ledger",
    )


def _atomic_first_authority(
    path: Path,
    encoded: bytes,
    *,
    label: str,
) -> None:
    if path.exists():
        if path.read_bytes() != encoded:
            raise SchedulerLoadRunConflictError(f"{label} already differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.parent / f".{path.name}.{digest}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise SchedulerLoadRunConflictError(f"{label} raced with another value") from None
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "R6SchedulerLoadCaseExecutor",
    "SchedulerLoadCaseExecutor",
    "SchedulerLoadExpectedFaultNotObservedError",
    "SchedulerLoadInjectedCrash",
    "SchedulerLoadMaterialIntegrityError",
    "SchedulerLoadResumeFailedError",
    "SchedulerLoadRunConflictError",
    "SchedulerLoadRunError",
    "SchedulerLoadRunResult",
    "SchedulerLoadRunner",
    "SchedulerLoadTimingError",
]
