from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from concurrency_experiment_fixtures import policy
from pydantic import ValidationError
from test_canary_pipeline_driver import _manifest
from test_canary_regression_builder import RAW_ROOT, requires_private_corpus
from test_concurrency_experiment_builder import _audit, _prepared_inputs

import eval_factory.readiness.concurrency_experiment as concurrency_module
from eval_factory.cli import (
    _admit_concurrency_report,
    _load_accepted_concurrency_report,
)
from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryExecutionManifestV2,
    r6_canary_execution_manifest_v2_ref,
)
from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionCaseResultV2,
    CanaryRegressionObservedOutcomeV2,
    CanaryRegressionReasonCodeV2,
)
from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentFaultPointV2,
    ConcurrencyExperimentHostSummaryV2,
    ConcurrencyExperimentLevelOutcomeV2,
    ConcurrencyExperimentOutcomeV2,
    ConcurrencyExperimentReasonCodeV2,
    ConcurrencyExperimentRecommendationReasonV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.observability_v2 import (
    CostObservationV2,
    WorkAttemptMetricsV2,
)
from eval_factory.contracts.orchestration import (
    ItemStatus,
    JobStatus,
    StageRunStatus,
)
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    WorkLeaseEventKindV2,
    WorkUnitScopeV2,
)
from eval_factory.contracts.resource_v2 import (
    NonModelResourceVectorV2,
    ResourceUsageV2,
)
from eval_factory.orchestration.canary_errors import CanaryDriverInjectedCrash
from eval_factory.orchestration.job_store import ImmutableResultError
from eval_factory.readiness.canary_regression_models import (
    FrozenCanaryCohortV1,
    PreparedCanaryRegressionCase,
)
from eval_factory.readiness.concurrency_experiment import (
    ConcurrencyCaseExecution,
    ConcurrencyExperimentCapacityWait,
    ConcurrencyExperimentRunError,
    ConcurrencyExperimentRunner,
    ConcurrencyRecoveryExecution,
    R6ConcurrencyCaseExecutor,
    R6ConcurrencyRecoveryExecutor,
)
from eval_factory.readiness.concurrency_experiment_builder import (
    derive_recovery_execution_identity,
)
from eval_factory.readiness.concurrency_experiment_models import (
    ConcurrencyExperimentCaseSampleV1,
    ConcurrencyExperimentFaultObservationV1,
    ConcurrencyExperimentWorkloadV1,
    PreparedConcurrencyExperimentCase,
    PreparedConcurrencyRecoveryCase,
)
from eval_factory.readiness.concurrency_experiment_store import (
    ConcurrencyExperimentMaterialStore,
    ConcurrencyExperimentReportStore,
)

NOW = datetime(2026, 8, 3, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(
    object_type: str,
    value: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    digest = _digest(f"{object_type}:{value}:{version}")
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version=version,
        object_sha256=digest,
    )


def _metric_audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-06-runner-test",
        governing_versions=(
            VersionBinding(
                component="batch-observability",
                version="batch-observability/r6-06-v1",
            ),
        ),
        input_refs=tuple(
            sorted(
                set(refs),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
    )


class _Clock:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            self._value += 100
            return self._value


def _child_manifest(
    base: R6CanaryExecutionManifestV2,
    *,
    job_id: str,
    idempotency_key: str,
) -> R6CanaryExecutionManifestV2:
    job_spec = base.job_spec.model_validate(
        {
            **base.job_spec.model_dump(mode="python"),
            "job_id": job_id,
            "idempotency_key": idempotency_key,
        }
    )
    return R6CanaryExecutionManifestV2.create(
        job_spec=job_spec,
        control=base.control,
        control_plane=base.control_plane,
        seed_objects=base.seed_objects,
        trace_bindings=base.trace_bindings,
        audit=base.audit,
    )


def _prepared(
    tmp_path: Path,
) -> tuple[
    ConcurrencyExperimentWorkloadV1,
    ConcurrencyExperimentHostSummaryV2,
    FrozenCanaryCohortV1,
    dict[str, tuple[PreparedConcurrencyExperimentCase, ...]],
    tuple[PreparedConcurrencyRecoveryCase, ...],
]:
    _builder, cohort, _template, host, workload = _prepared_inputs(tmp_path)
    manifest_root = tmp_path / "manifest"
    manifest_root.mkdir()
    base = _manifest(manifest_root)
    trials: dict[str, tuple[PreparedConcurrencyExperimentCase, ...]] = {}
    for trial in workload.trials:
        cases: list[PreparedConcurrencyExperimentCase] = []
        for ordinal, frozen in enumerate(cohort.cases):
            identity = (
                f"{workload.workload_sha256}-{trial.worker_count}-{trial.trial_index}-{frozen.instance_id}"
            )
            manifest = _child_manifest(
                base,
                job_id=(
                    "job://concurrency-experiment/"
                    f"{workload.workload_sha256}/{trial.worker_count}/"
                    f"{trial.trial_index}/{frozen.instance_id}"
                ),
                idempotency_key=f"concurrency-experiment-{identity}",
            )
            cases.append(
                PreparedConcurrencyExperimentCase(
                    trial=trial,
                    case_ordinal=ordinal,
                    frozen_case=frozen,
                    prepared_case=PreparedCanaryRegressionCase(
                        cohort_case=frozen,
                        expectation=frozen.expectation,
                        raw_path=tmp_path / "private-raw",
                        child_manifest=manifest,
                    ),
                )
            )
        trials[trial.trial_id] = tuple(cases)
    recovery: list[PreparedConcurrencyRecoveryCase] = []
    for ordinal, frozen in enumerate(cohort.cases):
        derived = derive_recovery_execution_identity(
            workload=workload,
            worker_count=1,
            case_ordinal=ordinal,
            case=frozen,
        )
        manifest = _child_manifest(
            base,
            job_id=derived.execution_identity.job_id,
            idempotency_key=derived.execution_identity.idempotency_key,
        )
        recovery.append(
            PreparedConcurrencyRecoveryCase(
                worker_count=1,
                case_ordinal=ordinal,
                fault_point=derived.fault_point,
                frozen_case=frozen,
                prepared_case=PreparedCanaryRegressionCase(
                    cohort_case=frozen,
                    expectation=frozen.expectation,
                    raw_path=tmp_path / "private-raw",
                    child_manifest=manifest,
                ),
            )
        )
    return workload, host, cohort, trials, tuple(recovery)


def _attempt_metrics(
    prepared: PreparedConcurrencyExperimentCase | PreparedConcurrencyRecoveryCase,
) -> tuple[WorkAttemptMetricsV2, ...]:
    manifest = prepared.prepared_case.child_manifest
    metrics: list[WorkAttemptMetricsV2] = []
    for attempt in range(1, 11):
        prefix = f"{manifest.job_spec.job_id}:{attempt}"
        graph_ref = _ref("resolved-job-work-graph", prefix)
        unit_ref = _ref("resolved-work-unit", prefix)
        lease_ref = _ref("work-lease", prefix)
        run_ref = _ref("stage-run", prefix, version="identity/v1")
        telemetry_ref = _ref("execution-telemetry", prefix)
        result_ref = _ref("stage-result", prefix, version="identity/v1")
        ready = NOW - timedelta(days=2)
        eligible = NOW + timedelta(seconds=attempt)
        metrics.append(
            WorkAttemptMetricsV2.create(
                resolved_job_work_graph_ref=graph_ref,
                work_unit_ref=unit_ref,
                work_lease_ref=lease_ref,
                stage_run_ref=run_ref,
                job_id=manifest.job_spec.job_id,
                item_id=f"item://runner/{prepared.frozen_case.instance_id}",
                scope=WorkUnitScopeV2.ITEM,
                stage=StageNameV2.TRACE_INDEX,
                artifact_execution_group_ref=None,
                attempt=attempt,
                fencing_token=attempt,
                terminal_lease_version=1,
                terminal_event_kind=WorkLeaseEventKindV2.SUCCEEDED,
                terminal_status=StageRunStatus.SUCCEEDED,
                reason_code=None,
                ready_at=ready,
                eligible_at=eligible,
                acquired_at=eligible + timedelta(microseconds=10),
                completed_at=eligible + timedelta(microseconds=20),
                model_profile_ref=None,
                model_usage=None,
                resource_usage=ResourceUsageV2(
                    observed=NonModelResourceVectorV2(
                        processes=0,
                        renderers=0,
                        network_requests=0,
                        storage_bytes=1,
                    )
                ),
                telemetry_ref=telemetry_ref,
                tool_metrics=(),
                cost=CostObservationV2.unavailable(),
                selected_route_kind=None,
                selected_route_id=None,
                selected_route_version=None,
                worker_version=None,
                retry_of_attempt_metrics_ref=None,
                result_refs=(result_ref,),
                output_refs=(),
                artifact_slices=(),
                audit=_metric_audit(
                    graph_ref,
                    unit_ref,
                    lease_ref,
                    run_ref,
                    telemetry_ref,
                    result_ref,
                ),
            )
        )
    return tuple(metrics)


def _success(
    prepared: PreparedConcurrencyExperimentCase | PreparedConcurrencyRecoveryCase,
    audit: ContractAudit,
) -> ConcurrencyCaseExecution:
    manifest = prepared.prepared_case.child_manifest
    prefix = manifest.job_spec.job_id
    stage_refs = tuple(_ref("stage-result", f"{prefix}:{index}", version="identity/v1") for index in range(9))
    result = CanaryRegressionCaseResultV2.create(
        instance_id=prepared.frozen_case.instance_id,
        source_trace_ref=manifest.trace_bindings[0].source_trace_ref,
        signal_quality=prepared.frozen_case.signal_quality,
        expectation=prepared.prepared_case.expectation,
        observed_outcome=CanaryRegressionObservedOutcomeV2.SUCCEEDED,
        reason_code=CanaryRegressionReasonCodeV2.NONE,
        highest_reached_stage=StageNameV2.ITEM_QUALITY,
        child_manifest_ref=r6_canary_execution_manifest_v2_ref(manifest),
        child_dataset_result_ref=_ref("r6-canary-dataset-result", prefix),
        job_id=manifest.job_spec.job_id,
        job_status=JobStatus.SUCCEEDED,
        item_id=f"item://runner/{prepared.frozen_case.instance_id}",
        item_status=ItemStatus.APPROVED,
        stage_result_refs=stage_refs,
        quality_result_refs=(_ref("item-quality-compilation-result", prefix),),
        package_manifest_refs=(_ref("final-package-manifest", prefix),),
        audit_report_ref=_ref("batch-audit-report", prefix),
        work_unit_count=10,
        work_lease_count=10,
        stage_run_count=9,
        attempt_count=10,
        retry_count=0,
        resume_count=0,
        provider_invocation_count=1,
        audit=audit,
    )
    return ConcurrencyCaseExecution(
        case_result=result,
        attempt_metrics=_attempt_metrics(prepared),
    )


def _partial_infrastructure(
    prepared: PreparedConcurrencyExperimentCase,
    audit: ContractAudit,
) -> ConcurrencyCaseExecution:
    manifest = prepared.prepared_case.child_manifest
    result = CanaryRegressionCaseResultV2.create(
        instance_id=prepared.frozen_case.instance_id,
        source_trace_ref=manifest.trace_bindings[0].source_trace_ref,
        signal_quality=prepared.frozen_case.signal_quality,
        expectation=prepared.prepared_case.expectation,
        observed_outcome=(CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR),
        reason_code=CanaryRegressionReasonCodeV2.INTERNAL_ERROR,
        highest_reached_stage=None,
        child_manifest_ref=r6_canary_execution_manifest_v2_ref(manifest),
        child_dataset_result_ref=None,
        job_id=manifest.job_spec.job_id,
        job_status=JobStatus.RUNNING,
        item_id=None,
        item_status=None,
        stage_result_refs=(),
        quality_result_refs=(),
        package_manifest_refs=(),
        audit_report_ref=None,
        work_unit_count=1,
        work_lease_count=1,
        stage_run_count=1,
        attempt_count=1,
        retry_count=0,
        resume_count=0,
        provider_invocation_count=0,
        audit=audit,
    )
    return ConcurrencyCaseExecution(
        case_result=result,
        attempt_metrics=(),
    )


class _Executor:
    def __init__(
        self,
        *,
        fail: frozenset[tuple[int, int, int]] = frozenset(),
        partial: frozenset[tuple[int, int, int]] = frozenset(),
        wait_once: frozenset[tuple[int, int, int]] = frozenset(),
        errors: dict[tuple[int, int, int], Exception] | None = None,
    ) -> None:
        self.fail = fail
        self.partial = partial
        self.wait_once = wait_once
        self.errors = errors or {}
        self.calls: dict[tuple[int, int, int], int] = {}
        self.active: dict[int, int] = {}
        self.max_active: dict[int, int] = {}
        self._lock = threading.Lock()

    def execute(
        self,
        prepared: PreparedConcurrencyExperimentCase,
        *,
        shared_job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> ConcurrencyCaseExecution:
        del shared_job_store_path, child_root
        key = (
            prepared.trial.worker_count,
            prepared.trial.trial_index,
            prepared.case_ordinal,
        )
        with self._lock:
            self.calls[key] = self.calls.get(key, 0) + 1
            active = self.active.get(prepared.trial.worker_count, 0) + 1
            self.active[prepared.trial.worker_count] = active
            self.max_active[prepared.trial.worker_count] = max(
                active,
                self.max_active.get(prepared.trial.worker_count, 0),
            )
        try:
            time.sleep(0.0005)
            if key in self.errors:
                raise self.errors[key]
            if key in self.fail:
                raise RuntimeError("private failure detail")
            if key in self.partial:
                return _partial_infrastructure(prepared, audit)
            if key in self.wait_once and self.calls[key] == 1:
                raise ConcurrencyExperimentCapacityWait
            return _success(prepared, audit)
        finally:
            with self._lock:
                self.active[prepared.trial.worker_count] -= 1


class _RecoveryExecutor:
    def __init__(
        self,
        *,
        fail_ordinal: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.fail_ordinal = fail_ordinal
        self.error = error

    def execute(
        self,
        prepared: PreparedConcurrencyRecoveryCase,
        *,
        shared_job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> ConcurrencyRecoveryExecution:
        del shared_job_store_path, child_root
        if prepared.case_ordinal == self.fail_ordinal:
            raise self.error or RuntimeError("private recovery detail")
        execution = _success(prepared, audit)
        return ConcurrencyRecoveryExecution(
            fault_observation=ConcurrencyExperimentFaultObservationV1.create(
                child_manifest_ref=r6_canary_execution_manifest_v2_ref(prepared.prepared_case.child_manifest),
                fault_point=prepared.fault_point,
                observed=True,
            ),
            case_result=execution.case_result,
            attempt_metrics=execution.attempt_metrics,
            provider_invocation_count=1,
            typed_before=10,
            typed_after=10,
            physical_files_before=3,
            physical_files_after=3,
            physical_bytes_before=1_000,
            physical_bytes_after=1_000,
            replay_stable=True,
        )


def test_runner_closes_full_staircase_and_bounds_workers(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)
    executor = _Executor()
    experiment_policy = policy()
    material = ConcurrencyExperimentMaterialStore(
        tmp_path / "private",
        max_private_bytes=100_000_000,
        max_members=1_000,
    )
    reports = ConcurrencyExperimentReportStore(
        tmp_path / "report",
        max_report_bytes=100_000_000,
    )
    host_write = material.put_host(host)
    material.put_workload(workload)
    run_root = tmp_path / "run"

    result = ConcurrencyExperimentRunner(
        executor=executor,
        recovery_executor=_RecoveryExecutor(),
        clock_ns=_Clock(),
        material_store=material,
    ).run(
        workload=workload,
        policy=experiment_policy,
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=recovery,
        run_root=run_root,
        audit=workload.audit,
    )

    assert len(result.case_samples) == 360
    assert len(result.trial_results) == 15
    assert len(result.level_results) == 5
    assert len(result.recovery_results) == 24
    assert len(result.run_ledger.child_manifest_refs) == 384
    assert len(result.run_ledger.trial_job_store_refs) == 16
    assert all(executor.max_active[worker_count] <= worker_count for worker_count in (1, 2, 4, 8, 16))
    assert all(
        tuple(sample.case_ordinal for sample in result.case_samples[offset : offset + 24]) == tuple(range(24))
        for offset in range(0, 360, 24)
    )
    assert result.report.outcome is ConcurrencyExperimentOutcomeV2.NON_MODEL_RECOMMENDED
    assert result.report.recommendation is not None
    assert result.report.recommendation.recommended_worker_count == 1
    level_1 = next(level.summary for level in result.level_results if level.worker_count == 1)
    assert level_1.attempt_active_total_microseconds.p50 == 20
    assert (
        result.case_samples[0].attempt_metrics[0].total_duration_us
        > level_1.attempt_active_total_microseconds.p50
    )
    assert material.get_host(host_write.object_ref) == host
    assert material.get_case_sample(result.case_samples[0].to_ref()) == (result.case_samples[0])
    assert material.get_trial_result(result.trial_results[0].to_ref()) == (result.trial_results[0])
    assert material.get_level_result(result.level_results[0].to_ref()) == (result.level_results[0])
    assert (
        material.get_fault_observation(result.recovery_results[0].fault_observation_ref).to_ref()
        == result.recovery_results[0].fault_observation_ref
    )
    assert material.get_recovery_result(result.recovery_results[0].to_ref()) == result.recovery_results[0]
    assert material.get_run_ledger(result.run_ledger.to_ref()) == (result.run_ledger)
    assert material.get_result_set(result.result_set.to_ref()) == (result.result_set)
    _admit_concurrency_report(run_root, result.report)
    reports.put(result.report)
    calls_before_replay = dict(executor.calls)
    replay = _load_accepted_concurrency_report(
        run_root=run_root,
        policy=experiment_policy,
        workload_ref=workload.to_ref(),
        material_store=material,
        report_store=reports,
    )
    assert replay == result.report
    assert executor.calls == calls_before_replay


def test_waiting_case_is_readmitted_after_peer_progress(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)
    waiting_key = (1, 0, 0)
    executor = _Executor(wait_once=frozenset({waiting_key}))

    result = ConcurrencyExperimentRunner(
        executor=executor,
        recovery_executor=_RecoveryExecutor(),
        clock_ns=_Clock(),
    ).run(
        workload=workload,
        policy=policy(),
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=recovery,
        run_root=tmp_path / "run",
        audit=workload.audit,
    )

    assert executor.calls[waiting_key] == 2
    assert len(result.case_samples) == 360


def test_one_case_failure_does_not_suppress_other_cases(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)
    executor = _Executor(fail=frozenset({(16, 2, 0)}))

    result = ConcurrencyExperimentRunner(
        executor=executor,
        recovery_executor=_RecoveryExecutor(),
        clock_ns=_Clock(),
    ).run(
        workload=workload,
        policy=policy(),
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=recovery,
        run_root=tmp_path / "run",
        audit=workload.audit,
    )

    assert len(result.case_samples) == 360
    level_16 = next(level.summary for level in result.level_results if level.worker_count == 16)
    assert level_16.succeeded_case_count == 71
    assert level_16.infrastructure_error_count == 1
    assert level_16.outcome is ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_ERROR
    assert result.report.outcome is ConcurrencyExperimentOutcomeV2.NON_MODEL_RECOMMENDED


def test_level_separates_durable_attempts_from_attempt_metrics(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)

    result = ConcurrencyExperimentRunner(
        executor=_Executor(partial=frozenset({(4, 0, 0)})),
        recovery_executor=_RecoveryExecutor(),
        clock_ns=_Clock(),
    ).run(
        workload=workload,
        policy=policy(),
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=recovery,
        run_root=tmp_path / "run",
        audit=workload.audit,
    )

    level_4 = next(level.summary for level in result.level_results if level.worker_count == 4)
    assert level_4.attempt_count == 711
    assert level_4.attempt_metric_count == 710
    assert level_4.queue_wait_microseconds.sample_count == 710
    assert level_4.outcome is ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_ERROR


def test_trial_preserves_typed_material_integrity_failure(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)

    result = ConcurrencyExperimentRunner(
        executor=_Executor(
            errors={
                (16, 2, 0): ImmutableResultError("private integrity detail"),
            }
        ),
        recovery_executor=_RecoveryExecutor(),
        clock_ns=_Clock(),
    ).run(
        workload=workload,
        policy=policy(),
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=recovery,
        run_root=tmp_path / "run",
        audit=workload.audit,
    )

    level_16 = next(level.summary for level in result.level_results if level.worker_count == 16)
    assert level_16.outcome is ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_INTEGRITY
    assert level_16.reason_codes == (ConcurrencyExperimentReasonCodeV2.MATERIAL_INTEGRITY_ERROR,)
    assert result.report.outcome is ConcurrencyExperimentOutcomeV2.INCOMPLETE
    assert result.report.recommendation is None


def test_recovery_propagates_typed_material_integrity_failure(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)

    with pytest.raises(ImmutableResultError, match="integrity"):
        ConcurrencyExperimentRunner(
            executor=_Executor(),
            recovery_executor=_RecoveryExecutor(
                fail_ordinal=0,
                error=ImmutableResultError("private integrity detail"),
            ),
            clock_ns=_Clock(),
        ).run(
            workload=workload,
            policy=policy(),
            host_summary=host,
            prepared_trials=trials,
            recovery_cases=recovery,
            run_root=tmp_path / "run",
            audit=workload.audit,
        )


def test_recovery_failure_blocks_recommendation_without_crashing(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)

    result = ConcurrencyExperimentRunner(
        executor=_Executor(),
        recovery_executor=_RecoveryExecutor(fail_ordinal=0),
        clock_ns=_Clock(),
    ).run(
        workload=workload,
        policy=policy(),
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=recovery,
        run_root=tmp_path / "run",
        audit=workload.audit,
    )

    assert result.report.outcome is ConcurrencyExperimentOutcomeV2.RECOMMENDATION_PENDING
    assert result.report.recommendation is None
    assert result.report.recovery_summary is not None
    assert result.report.recovery_summary.recovery_stable is False
    assert ConcurrencyExperimentRecommendationReasonV2.RECOVERY_VALIDATION_FAILED in (
        result.report.reason_codes
    )


@requires_private_corpus
def test_real_r6_executors_use_shared_job_store_and_exact_replay(
    tmp_path: Path,
) -> None:
    builder, cohort, template, _host, workload = _prepared_inputs(tmp_path)
    trial = workload.trials[0]
    prepared = builder.prepare_case(
        workload=workload,
        trial=trial,
        case_ordinal=0,
        case=cohort.cases[0],
        template=template,
        raw_root=RAW_ROOT,
        preparation_root=tmp_path / "preparation",
        audit=_audit(),
    )
    clean = R6ConcurrencyCaseExecutor().execute(
        prepared,
        shared_job_store_path=tmp_path / "clean.sqlite3",
        child_root=tmp_path / "clean-child",
        audit=_audit(),
    )

    assert clean.case_result.observed_outcome is (CanaryRegressionObservedOutcomeV2.SUCCEEDED)
    assert len(clean.attempt_metrics) == 10

    recovery_case = PreparedConcurrencyRecoveryCase(
        worker_count=1,
        case_ordinal=0,
        fault_point=ConcurrencyExperimentFaultPointV2.BEFORE_HANDLER,
        frozen_case=prepared.frozen_case,
        prepared_case=prepared.prepared_case,
    )
    recovered = R6ConcurrencyRecoveryExecutor().execute(
        recovery_case,
        shared_job_store_path=tmp_path / "recovery.sqlite3",
        child_root=tmp_path / "recovery-child",
        audit=_audit(),
    )

    assert recovered.fault_observation.observed is True
    assert recovered.case_result.observed_outcome is (CanaryRegressionObservedOutcomeV2.SUCCEEDED)
    assert len(recovered.attempt_metrics) == 10
    assert recovered.provider_invocation_count == 1
    assert recovered.typed_before == recovered.typed_after
    assert recovered.physical_files_before == recovered.physical_files_after
    assert recovered.replay_stable is True


def test_private_sample_preserves_partial_non_success_metrics(
    tmp_path: Path,
) -> None:
    _workload, _host, _cohort, trials, _recovery = _prepared(tmp_path)
    prepared = next(iter(trials.values()))[0]
    partial = _partial_infrastructure(prepared, _audit())

    sample = ConcurrencyExperimentCaseSampleV1.create(
        trial_ref=prepared.trial.to_ref(),
        case_ordinal=prepared.case_ordinal,
        child_manifest=prepared.prepared_case.child_manifest,
        case_result=partial.case_result,
        attempt_metrics=(),
        submitted_nanoseconds=1,
        started_nanoseconds=2,
        completed_nanoseconds=3,
        provider_invocation_count=0,
        audit=_audit(),
    )
    assert sample.case_result.attempt_count == 1
    assert sample.attempt_metrics == ()

    success = _success(prepared, _audit())
    with pytest.raises(ValidationError, match="complete attempt metrics"):
        ConcurrencyExperimentCaseSampleV1.create(
            trial_ref=prepared.trial.to_ref(),
            case_ordinal=prepared.case_ordinal,
            child_manifest=prepared.prepared_case.child_manifest,
            case_result=success.case_result,
            attempt_metrics=success.attempt_metrics[:-1],
            submitted_nanoseconds=1,
            started_nanoseconds=2,
            completed_nanoseconds=3,
            provider_invocation_count=1,
            audit=_audit(),
        )


def test_runner_rejects_mutually_exclusive_recovery_sources(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)

    with pytest.raises(ConcurrencyExperimentRunError, match="mutually exclusive"):
        ConcurrencyExperimentRunner(
            executor=_Executor(),
            recovery_executor=_RecoveryExecutor(),
            clock_ns=_Clock(),
        ).run(
            workload=workload,
            policy=policy(),
            host_summary=host,
            prepared_trials=trials,
            recovery_cases=recovery,
            recovery_case_factory=lambda _worker_count: recovery,
            run_root=tmp_path / "run",
            audit=workload.audit,
        )


def test_runner_rejects_missing_trial_before_execution(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, _recovery = _prepared(tmp_path)
    trials.pop(workload.trials[0].trial_id)

    with pytest.raises(ConcurrencyExperimentRunError, match="trial is missing"):
        ConcurrencyExperimentRunner(
            executor=_Executor(),
            clock_ns=_Clock(),
        ).run(
            workload=workload,
            policy=policy(),
            host_summary=host,
            prepared_trials=trials,
            recovery_cases=(),
            run_root=tmp_path / "run",
            audit=workload.audit,
        )


def test_trial_rejects_wrong_case_count_and_mixed_trials(
    tmp_path: Path,
) -> None:
    workload, _host, _cohort, trials, _recovery = _prepared(tmp_path)
    first = workload.trials[0]
    second = workload.trials[1]
    runner = ConcurrencyExperimentRunner(
        executor=_Executor(),
        clock_ns=_Clock(),
    )

    with pytest.raises(ConcurrencyExperimentRunError, match="exactly 24"):
        runner._run_trial(
            trials[first.trial_id][:-1],
            policy=policy(),
            job_store_path=tmp_path / "short.sqlite3",
            trial_root=tmp_path / "short",
            audit=workload.audit,
        )

    mixed = list(trials[first.trial_id])
    mixed[0] = trials[second.trial_id][0]
    with pytest.raises(ConcurrencyExperimentRunError, match="mixes"):
        runner._run_trial(
            tuple(mixed),
            policy=policy(),
            job_store_path=tmp_path / "mixed.sqlite3",
            trial_root=tmp_path / "mixed",
            audit=workload.audit,
        )


def test_trial_rejects_duplicate_ordinals_before_execution(
    tmp_path: Path,
) -> None:
    workload, _host, _cohort, trials, _recovery = _prepared(tmp_path)
    trial = workload.trials[0]
    duplicated = list(trials[trial.trial_id])
    duplicated[1] = replace(duplicated[1], case_ordinal=0)

    with pytest.raises(
        ConcurrencyExperimentRunError,
        match="canonical case ordinals",
    ):
        ConcurrencyExperimentRunner(
            executor=_Executor(),
            clock_ns=_Clock(),
        )._run_trial(
            tuple(duplicated),
            policy=policy(),
            job_store_path=tmp_path / "duplicate.sqlite3",
            trial_root=tmp_path / "duplicate",
            audit=workload.audit,
        )


def test_trial_closes_all_waiting_cases_without_attempts(
    tmp_path: Path,
) -> None:
    workload, _host, _cohort, trials, _recovery = _prepared(tmp_path)
    trial = workload.trials[0]
    always_wait = {
        (trial.worker_count, trial.trial_index, ordinal): (ConcurrencyExperimentCapacityWait())
        for ordinal in range(24)
    }

    samples, result = ConcurrencyExperimentRunner(
        executor=_Executor(errors=always_wait),
        clock_ns=_Clock(),
    )._run_trial(
        trials[trial.trial_id],
        policy=policy(),
        job_store_path=tmp_path / "waiting.sqlite3",
        trial_root=tmp_path / "waiting",
        audit=workload.audit,
    )

    assert len(samples) == 24
    assert all(sample.attempt_metrics == () for sample in samples)
    assert all(
        sample.case_result.reason_code is CanaryRegressionReasonCodeV2.RESOURCE_WAITING for sample in samples
    )
    assert result.typed_side_effect_count == 0


def test_runner_uses_recovery_factory_and_can_publish_pending_without_recovery(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)
    requested_workers: list[int] = []

    with_recovery = ConcurrencyExperimentRunner(
        executor=_Executor(),
        recovery_executor=_RecoveryExecutor(),
        clock_ns=_Clock(),
    ).run(
        workload=workload,
        policy=policy(),
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=(),
        recovery_case_factory=lambda worker_count: requested_workers.append(worker_count) or recovery,
        run_root=tmp_path / "with-recovery",
        audit=workload.audit,
    )
    assert requested_workers == [1]
    assert len(with_recovery.recovery_results) == 24

    without_recovery = ConcurrencyExperimentRunner(
        executor=_Executor(),
        clock_ns=_Clock(),
    ).run(
        workload=workload,
        policy=policy(),
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=(),
        run_root=tmp_path / "without-recovery",
        audit=workload.audit,
    )
    assert without_recovery.report.outcome is (ConcurrencyExperimentOutcomeV2.RECOMMENDATION_PENDING)
    assert without_recovery.report.recovery_summary is None
    assert without_recovery.report.recommendation is None
    assert len(without_recovery.run_ledger.trial_job_store_refs) == 15


def test_recovery_rejects_wrong_count_and_candidate_worker(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)
    runner = ConcurrencyExperimentRunner(
        executor=_Executor(),
        recovery_executor=_RecoveryExecutor(),
        clock_ns=_Clock(),
    )

    with pytest.raises(ConcurrencyExperimentRunError, match="exactly 24 probes"):
        runner._run_recovery(
            recovery[:-1],
            candidate_worker_count=1,
            job_store_path=tmp_path / "short-recovery.sqlite3",
            recovery_root=tmp_path / "short-recovery",
            audit=workload.audit,
        )

    wrong_worker = tuple(replace(case, worker_count=2) for case in recovery)
    with pytest.raises(
        ConcurrencyExperimentRunError,
        match="another candidate worker count",
    ):
        runner.run(
            workload=workload,
            policy=policy(),
            host_summary=host,
            prepared_trials=trials,
            recovery_cases=wrong_worker,
            run_root=tmp_path / "wrong-worker",
            audit=workload.audit,
        )


def test_runner_rejects_file_and_symlink_roots(tmp_path: Path) -> None:
    workload, host, _cohort, trials, _recovery = _prepared(tmp_path)
    runner = ConcurrencyExperimentRunner(
        executor=_Executor(),
        clock_ns=_Clock(),
    )
    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ConcurrencyExperimentRunError, match="non-symlink"):
        runner.run(
            workload=workload,
            policy=policy(),
            host_summary=host,
            prepared_trials=trials,
            recovery_cases=(),
            run_root=file_root,
            audit=workload.audit,
        )

    actual = tmp_path / "actual-root"
    actual.mkdir()
    alias = tmp_path / "alias-root"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ConcurrencyExperimentRunError, match="non-symlink"):
        runner.run(
            workload=workload,
            policy=policy(),
            host_summary=host,
            prepared_trials=trials,
            recovery_cases=(),
            run_root=alias,
            audit=workload.audit,
        )


def test_level_with_no_attempt_metrics_remains_reportable(
    tmp_path: Path,
) -> None:
    workload, host, _cohort, trials, recovery = _prepared(tmp_path)
    errors = {
        (16, trial_index, ordinal): RuntimeError("private failure")
        for trial_index in range(3)
        for ordinal in range(24)
    }

    result = ConcurrencyExperimentRunner(
        executor=_Executor(errors=errors),
        recovery_executor=_RecoveryExecutor(),
        clock_ns=_Clock(),
    ).run(
        workload=workload,
        policy=policy(),
        host_summary=host,
        prepared_trials=trials,
        recovery_cases=recovery,
        run_root=tmp_path / "run",
        audit=workload.audit,
    )

    level = next(value.summary for value in result.level_results if value.worker_count == 16)
    assert level.outcome is ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_ERROR
    assert level.infrastructure_error_count == 72
    assert level.attempt_count == 0
    assert level.attempt_metric_count == 0
    assert level.queue_wait_microseconds.sample_count == 0
    assert (
        level.queue_wait_microseconds.minimum,
        level.queue_wait_microseconds.p50,
        level.queue_wait_microseconds.p95,
        level.queue_wait_microseconds.maximum,
    ) == (0, 0, 0, 0)


def test_recovery_recompiles_provider_evidence_after_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workload, _host, _cohort, _trials, recovery = _prepared(tmp_path)
    prepared = recovery[0]
    marker = object()

    class _InjectedDriver:
        job_store = object()
        stage_store = object()

        async def run(self, _manifest: object) -> object:
            raise CanaryDriverInjectedCrash("expected")

    class _ResumedDriver:
        job_store = object()
        stage_store = object()

        async def run(self, _manifest: object) -> object:
            return marker

    class _Compiler:
        def __init__(self) -> None:
            self.calls = 0

        def compile(self, *_args: object, **_kwargs: object):
            self.calls += 1
            result = _success(prepared, _audit()).case_result
            if self.calls == 2:
                return result.model_copy(update={"provider_invocation_count": 2})
            return result

    class _Observability:
        def __init__(self, _store: object) -> None:
            pass

        def list_attempt_metrics(
            self,
            _job_id: str,
        ) -> tuple[WorkAttemptMetricsV2, ...]:
            return ()

    drivers = iter((_InjectedDriver(), _ResumedDriver()))
    monkeypatch.setattr(
        concurrency_module,
        "_driver",
        lambda *_args, **_kwargs: next(drivers),
    )
    monkeypatch.setattr(
        concurrency_module,
        "BatchObservabilityService",
        _Observability,
    )
    monkeypatch.setattr(
        concurrency_module,
        "_job_evidence_count",
        lambda *_args: 10,
    )
    monkeypatch.setattr(
        concurrency_module,
        "_recovery_physical_snapshot",
        lambda *_args: (3, 100),
    )

    observed = R6ConcurrencyRecoveryExecutor(
        compiler=_Compiler(),  # type: ignore[arg-type]
    ).execute(
        prepared,
        shared_job_store_path=tmp_path / "job.sqlite3",
        child_root=tmp_path / "child",
        audit=_audit(),
    )

    assert observed.provider_invocation_count == 2
    assert observed.replay_stable is False


def test_recovery_physical_snapshot_includes_child_files(
    tmp_path: Path,
) -> None:
    job_store = tmp_path / "job.sqlite3"
    job_store.write_bytes(b"job")
    child = tmp_path / "child"
    child.mkdir()
    (child / "first.json").write_text("first", encoding="utf-8")
    before = concurrency_module._recovery_physical_snapshot(job_store, child)
    (child / "second.json").write_text("second", encoding="utf-8")
    after = concurrency_module._recovery_physical_snapshot(job_store, child)

    assert before[0] == 2
    assert after[0] == 3
    assert after[1] > before[1]
