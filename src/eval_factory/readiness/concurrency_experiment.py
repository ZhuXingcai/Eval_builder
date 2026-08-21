from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryDatasetResultV2,
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
    ConcurrencyExperimentLevelSummaryV2,
    ConcurrencyExperimentPercentileSummaryV2,
    ConcurrencyExperimentPolicyV2,
    ConcurrencyExperimentReasonCodeV2,
    ConcurrencyExperimentRecoverySummaryV2,
    ConcurrencyExperimentReportV2,
    ConcurrencyExperimentUnitV2,
    NonModelResourceRecommendationV2,
    select_balanced_worker_level,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.observability_v2 import (
    CacheUseV2,
    MetricAvailabilityV2,
    WorkAttemptMetricsV2,
)
from eval_factory.contracts.resource_v2 import ResourceAdmissionOutcomeV2
from eval_factory.orchestration.canary_driver import (
    CanaryDriverFaultPoint,
    CanaryPipelineDriver,
    StaticCanaryDriverFaultInjector,
    create_canary_stage_store,
)
from eval_factory.orchestration.canary_errors import (
    CanaryDriverInjectedCrash,
    CanaryResourceAdmissionError,
)
from eval_factory.orchestration.job_store import JobStore
from eval_factory.orchestration.observability import BatchObservabilityService
from eval_factory.readiness.canary_regression import (
    CanaryRegressionCaseEvidenceCompiler,
    _error_reason,
)
from eval_factory.readiness.concurrency_experiment_models import (
    ConcurrencyExperimentCaseSampleV1,
    ConcurrencyExperimentFaultObservationV1,
    ConcurrencyExperimentLevelResultV1,
    ConcurrencyExperimentRecoveryResultV1,
    ConcurrencyExperimentResultSetV1,
    ConcurrencyExperimentRunLedgerV1,
    ConcurrencyExperimentTrialResultV1,
    ConcurrencyExperimentTrialSpecV1,
    ConcurrencyExperimentWorkloadV1,
    PreparedConcurrencyExperimentCase,
    PreparedConcurrencyRecoveryCase,
)
from eval_factory.readiness.concurrency_experiment_store import (
    ConcurrencyExperimentMaterialStore,
)
from eval_factory.trace import TraceIndexStore, TraceSourceRegistry


class ConcurrencyExperimentRunError(RuntimeError):
    pass


class ConcurrencyExperimentCapacityWait(ConcurrencyExperimentRunError):
    pass


@dataclass(frozen=True, slots=True)
class ConcurrencyCaseExecution:
    case_result: CanaryRegressionCaseResultV2
    attempt_metrics: tuple[WorkAttemptMetricsV2, ...]


@dataclass(frozen=True, slots=True)
class ConcurrencyRecoveryExecution:
    fault_observation: ConcurrencyExperimentFaultObservationV1
    case_result: CanaryRegressionCaseResultV2
    attempt_metrics: tuple[WorkAttemptMetricsV2, ...]
    provider_invocation_count: int
    typed_before: int
    typed_after: int
    physical_files_before: int
    physical_files_after: int
    physical_bytes_before: int
    physical_bytes_after: int
    replay_stable: bool


class ConcurrencyCaseExecutor(Protocol):
    def execute(
        self,
        prepared: PreparedConcurrencyExperimentCase,
        *,
        shared_job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> ConcurrencyCaseExecution: ...


class ConcurrencyRecoveryExecutor(Protocol):
    def execute(
        self,
        prepared: PreparedConcurrencyRecoveryCase,
        *,
        shared_job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> ConcurrencyRecoveryExecution: ...


class R6ConcurrencyCaseExecutor:
    def __init__(
        self,
        *,
        compiler: CanaryRegressionCaseEvidenceCompiler | None = None,
    ) -> None:
        self._compiler = compiler or CanaryRegressionCaseEvidenceCompiler()

    def execute(
        self,
        prepared: PreparedConcurrencyExperimentCase,
        *,
        shared_job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> ConcurrencyCaseExecution:
        root = _private_root(child_root)
        driver = _driver(shared_job_store_path, root)
        child_result: R6CanaryDatasetResultV2 | None = None
        error: Exception | None = None
        try:
            child_result = asyncio.run(driver.run(prepared.prepared_case.child_manifest))
        except CanaryResourceAdmissionError as exc:
            if exc.outcome is ResourceAdmissionOutcomeV2.WAITING_CAPACITY:
                raise ConcurrencyExperimentCapacityWait from exc
            error = exc
        except Exception as exc:
            error = exc
        result = self._compiler.compile(
            prepared.prepared_case,
            job_store=driver.job_store,
            stage_store=driver.stage_store,
            child_result=child_result,
            error=error,
            audit=audit,
        )
        metrics = BatchObservabilityService(driver.job_store).list_attempt_metrics(result.job_id)
        return ConcurrencyCaseExecution(
            case_result=result,
            attempt_metrics=metrics,
        )


class R6ConcurrencyRecoveryExecutor:
    def __init__(
        self,
        *,
        compiler: CanaryRegressionCaseEvidenceCompiler | None = None,
    ) -> None:
        self._compiler = compiler or CanaryRegressionCaseEvidenceCompiler()

    def execute(
        self,
        prepared: PreparedConcurrencyRecoveryCase,
        *,
        shared_job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> ConcurrencyRecoveryExecution:
        root = _private_root(child_root)
        injected = _driver(
            shared_job_store_path,
            root,
            fault_injector=StaticCanaryDriverFaultInjector(
                crash_points=frozenset({CanaryDriverFaultPoint(prepared.fault_point.value.casefold())})
            ),
        )
        observed = False
        try:
            asyncio.run(injected.run(prepared.prepared_case.child_manifest))
        except CanaryDriverInjectedCrash:
            observed = True
        fault = ConcurrencyExperimentFaultObservationV1.create(
            child_manifest_ref=r6_canary_execution_manifest_v2_ref(prepared.prepared_case.child_manifest),
            fault_point=prepared.fault_point,
            observed=observed,
        )
        resumed = _driver(shared_job_store_path, root)
        child_result: R6CanaryDatasetResultV2 | None = None
        error: Exception | None = None
        try:
            child_result = asyncio.run(resumed.run(prepared.prepared_case.child_manifest))
        except Exception as exc:
            error = exc
        result = self._compiler.compile(
            prepared.prepared_case,
            job_store=resumed.job_store,
            stage_store=resumed.stage_store,
            child_result=child_result,
            error=error,
            audit=audit,
        )
        metrics = BatchObservabilityService(resumed.job_store).list_attempt_metrics(result.job_id)
        before = _job_evidence_count(resumed.job_store, result.job_id)
        physical_before = _recovery_physical_snapshot(
            shared_job_store_path,
            root,
        )
        replay_result = asyncio.run(resumed.run(prepared.prepared_case.child_manifest))
        replayed = self._compiler.compile(
            prepared.prepared_case,
            job_store=resumed.job_store,
            stage_store=resumed.stage_store,
            child_result=replay_result,
            error=None,
            audit=audit,
        )
        after = _job_evidence_count(resumed.job_store, result.job_id)
        physical_after = _recovery_physical_snapshot(
            shared_job_store_path,
            root,
        )
        replay_stable = (
            child_result is not None
            and replay_result == child_result
            and replayed == result
            and before == after
            and physical_before[0] == physical_after[0]
        )
        return ConcurrencyRecoveryExecution(
            fault_observation=fault,
            case_result=result,
            attempt_metrics=metrics,
            provider_invocation_count=replayed.provider_invocation_count,
            typed_before=before,
            typed_after=after,
            physical_files_before=physical_before[0],
            physical_files_after=physical_after[0],
            physical_bytes_before=physical_before[1],
            physical_bytes_after=physical_after[1],
            replay_stable=replay_stable,
        )


@dataclass(frozen=True, slots=True)
class ConcurrencyExperimentRunResult:
    run_ledger: ConcurrencyExperimentRunLedgerV1
    case_samples: tuple[ConcurrencyExperimentCaseSampleV1, ...]
    trial_results: tuple[ConcurrencyExperimentTrialResultV1, ...]
    level_results: tuple[ConcurrencyExperimentLevelResultV1, ...]
    recovery_results: tuple[ConcurrencyExperimentRecoveryResultV1, ...]
    result_set: ConcurrencyExperimentResultSetV1
    report: ConcurrencyExperimentReportV2


class ConcurrencyExperimentRunner:
    def __init__(
        self,
        *,
        executor: ConcurrencyCaseExecutor | None = None,
        recovery_executor: ConcurrencyRecoveryExecutor | None = None,
        clock_ns: Callable[[], int] = perf_counter_ns,
        material_store: ConcurrencyExperimentMaterialStore | None = None,
    ) -> None:
        self._executor = executor or R6ConcurrencyCaseExecutor()
        self._recovery_executor = recovery_executor or R6ConcurrencyRecoveryExecutor()
        self._clock_ns = clock_ns
        self._material_store = material_store

    def run(
        self,
        *,
        workload: ConcurrencyExperimentWorkloadV1,
        policy: ConcurrencyExperimentPolicyV2,
        host_summary: ConcurrencyExperimentHostSummaryV2,
        prepared_trials: dict[
            str,
            tuple[PreparedConcurrencyExperimentCase, ...],
        ],
        recovery_cases: tuple[PreparedConcurrencyRecoveryCase, ...],
        run_root: Path,
        audit: ContractAudit,
        recovery_case_factory: (
            Callable[
                [int],
                tuple[PreparedConcurrencyRecoveryCase, ...],
            ]
            | None
        ) = None,
    ) -> ConcurrencyExperimentRunResult:
        root = _private_root(run_root)
        if recovery_cases and recovery_case_factory is not None:
            raise ConcurrencyExperimentRunError("recovery cases and recovery factory are mutually exclusive")
        case_samples: list[ConcurrencyExperimentCaseSampleV1] = []
        trial_results: list[ConcurrencyExperimentTrialResultV1] = []
        samples_by_trial: dict[str, tuple[ConcurrencyExperimentCaseSampleV1, ...]] = {}
        for trial in workload.trials:
            prepared = prepared_trials.get(trial.trial_id)
            if prepared is None:
                raise ConcurrencyExperimentRunError("prepared trial is missing")
            trial_samples, trial_result = self._run_trial(
                prepared,
                policy=policy,
                job_store_path=(
                    root
                    / "trials"
                    / f"workers-{trial.worker_count}"
                    / f"trial-{trial.trial_index}"
                    / "job.sqlite3"
                ),
                trial_root=(root / "trials" / f"workers-{trial.worker_count}" / f"trial-{trial.trial_index}"),
                audit=audit,
            )
            samples_by_trial[trial.trial_id] = trial_samples
            case_samples.extend(trial_samples)
            trial_results.append(trial_result)
        level_results = self._compile_levels(
            workload,
            tuple(trial_results),
            samples_by_trial,
            audit=audit,
        )
        summaries = tuple(result.summary for result in level_results)
        candidate = select_balanced_worker_level(summaries, policy=policy)
        recovery_results: tuple[ConcurrencyExperimentRecoveryResultV1, ...] = ()
        recovery_summary: ConcurrencyExperimentRecoverySummaryV2 | None = None
        recommendation: NonModelResourceRecommendationV2 | None = None
        selected_recovery_cases: tuple[
            PreparedConcurrencyRecoveryCase,
            ...,
        ] = ()
        if candidate is not None and recovery_case_factory is not None:
            selected_recovery_cases = recovery_case_factory(candidate.worker_count)
        elif candidate is not None:
            selected_recovery_cases = recovery_cases
        if candidate is not None and selected_recovery_cases:
            if any(case.worker_count != candidate.worker_count for case in selected_recovery_cases):
                raise ConcurrencyExperimentRunError("recovery cases use another candidate worker count")
            recovery_results, recovery_summary = self._run_recovery(
                selected_recovery_cases,
                candidate_worker_count=candidate.worker_count,
                job_store_path=root / "recovery" / "job.sqlite3",
                recovery_root=root / "recovery",
                audit=audit,
            )
            if recovery_summary.recovery_stable:
                recommendation = NonModelResourceRecommendationV2.create(
                    selected_level=candidate,
                    level_summaries=summaries,
                    recovery_summary=recovery_summary,
                    recommended_storage_bytes_per_job=(workload.retained_storage_bytes_per_case),
                    recommended_storage_bytes_per_cohort=(workload.retained_storage_bytes_per_cohort),
                    policy=policy,
                    audit=audit,
                )
        all_manifests = tuple(
            case.prepared_case.child_manifest
            for trial in workload.trials
            for case in prepared_trials.get(trial.trial_id, ())
        ) + tuple(case.prepared_case.child_manifest for case in selected_recovery_cases)
        job_store_refs = tuple(_job_store_ref(workload, trial) for trial in workload.trials) + (
            (_recovery_job_store_ref(workload),) if selected_recovery_cases else ()
        )
        ledger = ConcurrencyExperimentRunLedgerV1.create(
            workload_ref=workload.to_ref(),
            trial_job_store_refs=job_store_refs,
            child_manifests=all_manifests,
        )
        result_set = ConcurrencyExperimentResultSetV1.create(
            workload_ref=workload.to_ref(),
            level_results=level_results,
            recovery_results=recovery_results,
            audit=audit,
        )
        report = ConcurrencyExperimentReportV2.create(
            policy_ref=policy.to_ref(),
            host_summary=host_summary,
            cohort_manifest_ref=policy.cohort_manifest_ref,
            template_ref=workload.template_ref,
            private_workload_ref=workload.to_ref(),
            private_result_set_ref=result_set.to_ref(),
            level_summaries=summaries,
            recovery_summary=recovery_summary,
            recommendation=recommendation,
            audit=audit,
        )
        self._persist_aggregate(
            ledger=ledger,
            level_results=level_results,
            recovery_results=recovery_results,
            result_set=result_set,
        )
        return ConcurrencyExperimentRunResult(
            run_ledger=ledger,
            case_samples=tuple(case_samples),
            trial_results=tuple(trial_results),
            level_results=level_results,
            recovery_results=recovery_results,
            result_set=result_set,
            report=report,
        )

    def _run_trial(
        self,
        prepared: tuple[PreparedConcurrencyExperimentCase, ...],
        *,
        policy: ConcurrencyExperimentPolicyV2,
        job_store_path: Path,
        trial_root: Path,
        audit: ContractAudit,
    ) -> tuple[
        tuple[ConcurrencyExperimentCaseSampleV1, ...],
        ConcurrencyExperimentTrialResultV1,
    ]:
        if len(prepared) != 24:
            raise ConcurrencyExperimentRunError("concurrency trial requires exactly 24 cases")
        trial = prepared[0].trial
        if any(case.trial != trial for case in prepared):
            raise ConcurrencyExperimentRunError("concurrency trial mixes trial identities")
        if {case.case_ordinal for case in prepared} != set(range(24)):
            raise ConcurrencyExperimentRunError("concurrency trial requires canonical case ordinals")
        trial_started = self._clock_ns()
        submitted = {case.case_ordinal: self._clock_ns() for case in prepared}
        readmissions = {case.case_ordinal: 0 for case in prepared}
        pending = {case.case_ordinal: case for case in prepared}
        samples: dict[int, ConcurrencyExperimentCaseSampleV1] = {}
        while pending:
            progressed = False
            waiting_cases: dict[int, PreparedConcurrencyExperimentCase] = {}
            with ThreadPoolExecutor(
                max_workers=trial.worker_count,
            ) as pool:
                futures: dict[
                    Future[
                        tuple[
                            PreparedConcurrencyExperimentCase,
                            int,
                            int,
                            ConcurrencyCaseExecution | Exception,
                        ]
                    ],
                    PreparedConcurrencyExperimentCase,
                ] = {
                    pool.submit(
                        self._execute_timed,
                        case,
                        submitted[case.case_ordinal],
                        job_store_path,
                        trial_root / "children" / case.frozen_case.instance_id,
                        audit,
                    ): case
                    for case in pending.values()
                }
                active = set(futures)
                while active:
                    done, active = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        case, started, completed, execution = future.result()
                        if isinstance(
                            execution,
                            ConcurrencyExperimentCapacityWait,
                        ):
                            waiting_cases[case.case_ordinal] = case
                            continue
                        if isinstance(execution, Exception):
                            execution = _closed_exception_execution(
                                case,
                                error=execution,
                                audit=audit,
                            )
                        sample = ConcurrencyExperimentCaseSampleV1.create(
                            trial_ref=trial.to_ref(),
                            case_ordinal=case.case_ordinal,
                            child_manifest=(case.prepared_case.child_manifest),
                            case_result=execution.case_result,
                            attempt_metrics=execution.attempt_metrics,
                            submitted_nanoseconds=(submitted[case.case_ordinal]),
                            started_nanoseconds=started,
                            completed_nanoseconds=completed,
                            provider_invocation_count=(execution.case_result.provider_invocation_count),
                            audit=audit,
                        )
                        samples[case.case_ordinal] = sample
                        if self._material_store is not None:
                            self._material_store.put_case_sample(sample)
                        progressed = True
            if not waiting_cases:
                break
            for ordinal in waiting_cases:
                readmissions[ordinal] += 1
            exhausted = tuple(
                ordinal
                for ordinal in waiting_cases
                if readmissions[ordinal] > policy.max_capacity_readmissions
            )
            if exhausted or not progressed:
                for ordinal, case in waiting_cases.items():
                    execution = _closed_wait_execution(case, audit=audit)
                    started = self._clock_ns()
                    completed = max(started + 1, self._clock_ns())
                    sample = ConcurrencyExperimentCaseSampleV1.create(
                        trial_ref=trial.to_ref(),
                        case_ordinal=case.case_ordinal,
                        child_manifest=case.prepared_case.child_manifest,
                        case_result=execution.case_result,
                        attempt_metrics=(),
                        submitted_nanoseconds=submitted[ordinal],
                        started_nanoseconds=started,
                        completed_nanoseconds=completed,
                        provider_invocation_count=0,
                        audit=audit,
                    )
                    samples[ordinal] = sample
                break
            pending = waiting_cases
        trial_completed = max(trial_started + 1, self._clock_ns())
        ordered = tuple(samples[index] for index in range(24))
        physical = _physical_snapshot(job_store_path)
        result = ConcurrencyExperimentTrialResultV1.create(
            trial_ref=trial.to_ref(),
            case_samples=ordered,
            elapsed_nanoseconds=trial_completed - trial_started,
            typed_side_effect_count=sum(len(value.attempt_metrics) for value in ordered),
            physical_file_count=physical[0],
            physical_bytes=physical[1],
            audit=audit,
        )
        if self._material_store is not None:
            self._material_store.put_trial_result(result)
        return ordered, result

    def _execute_timed(
        self,
        case: PreparedConcurrencyExperimentCase,
        submitted: int,
        job_store_path: Path,
        child_root: Path,
        audit: ContractAudit,
    ) -> tuple[
        PreparedConcurrencyExperimentCase,
        int,
        int,
        ConcurrencyCaseExecution | Exception,
    ]:
        del submitted
        started = self._clock_ns()
        try:
            execution: ConcurrencyCaseExecution | Exception = self._executor.execute(
                case,
                shared_job_store_path=job_store_path,
                child_root=child_root,
                audit=audit,
            )
        except Exception as exc:
            execution = exc
        completed = max(started + 1, self._clock_ns())
        return case, started, completed, execution

    def _compile_levels(
        self,
        workload: ConcurrencyExperimentWorkloadV1,
        trial_results: tuple[ConcurrencyExperimentTrialResultV1, ...],
        samples_by_trial: dict[
            str,
            tuple[ConcurrencyExperimentCaseSampleV1, ...],
        ],
        *,
        audit: ContractAudit,
    ) -> tuple[ConcurrencyExperimentLevelResultV1, ...]:
        results: list[ConcurrencyExperimentLevelResultV1] = []
        by_ref = {result.trial_ref: result for result in trial_results}
        for worker_count in (1, 2, 4, 8, 16):
            trials = tuple(trial for trial in workload.trials if trial.worker_count == worker_count)
            level_trial_results = tuple(by_ref[trial.to_ref()] for trial in trials)
            samples = tuple(sample for trial in trials for sample in samples_by_trial[trial.trial_id])
            summary = _level_summary(
                worker_count,
                level_trial_results,
                samples,
            )
            level = ConcurrencyExperimentLevelResultV1.create(
                worker_count=worker_count,
                trial_results=level_trial_results,
                summary=summary,
                audit=audit,
            )
            if self._material_store is not None:
                self._material_store.put_level_result(level)
            results.append(level)
        return tuple(results)

    def _run_recovery(
        self,
        prepared: tuple[PreparedConcurrencyRecoveryCase, ...],
        *,
        candidate_worker_count: int,
        job_store_path: Path,
        recovery_root: Path,
        audit: ContractAudit,
    ) -> tuple[
        tuple[ConcurrencyExperimentRecoveryResultV1, ...],
        ConcurrencyExperimentRecoverySummaryV2,
    ]:
        if len(prepared) != 24:
            raise ConcurrencyExperimentRunError("recovery validation requires exactly 24 probes")
        with ThreadPoolExecutor(
            max_workers=candidate_worker_count,
        ) as pool:
            futures = tuple(
                (
                    case,
                    pool.submit(
                        self._recovery_executor.execute,
                        case,
                        shared_job_store_path=job_store_path,
                        child_root=(recovery_root / "children" / case.frozen_case.instance_id),
                        audit=audit,
                    ),
                )
                for case in prepared
            )
            executions_list: list[ConcurrencyRecoveryExecution] = []
            for case, future in futures:
                try:
                    executions_list.append(future.result())
                except Exception as exc:
                    if _error_reason(exc) is CanaryRegressionReasonCodeV2.MATERIAL_INTEGRITY_ERROR:
                        raise
                    fault = ConcurrencyExperimentFaultObservationV1.create(
                        child_manifest_ref=(
                            r6_canary_execution_manifest_v2_ref(case.prepared_case.child_manifest)
                        ),
                        fault_point=case.fault_point,
                        observed=False,
                    )
                    closed = _closed_exception_execution(
                        case,
                        error=exc,
                        audit=audit,
                    )
                    executions_list.append(
                        ConcurrencyRecoveryExecution(
                            fault_observation=fault,
                            case_result=closed.case_result,
                            attempt_metrics=(),
                            provider_invocation_count=0,
                            typed_before=0,
                            typed_after=0,
                            physical_files_before=0,
                            physical_files_after=0,
                            physical_bytes_before=0,
                            physical_bytes_after=0,
                            replay_stable=False,
                        )
                    )
            executions = tuple(executions_list)
        points = tuple(ConcurrencyExperimentFaultPointV2)
        summary = ConcurrencyExperimentRecoverySummaryV2.create(
            selected_worker_count=candidate_worker_count,
            fault_counts={
                point: sum(execution.fault_observation.fault_point is point for execution in executions)
                for point in points
            },
            fault_observed_count=sum(value.fault_observation.observed for value in executions),
            resume_success_count=sum(
                value.case_result.observed_outcome is CanaryRegressionObservedOutcomeV2.SUCCEEDED
                for value in executions
            ),
            resume_failure_count=sum(
                value.case_result.observed_outcome is not CanaryRegressionObservedOutcomeV2.SUCCEEDED
                for value in executions
            ),
            unexpected_retry_count=sum(value.case_result.retry_count for value in executions),
            duplicate_provider_fact_count=sum(
                max(0, value.provider_invocation_count - 1) for value in executions
            ),
            provider_invocation_count=sum(value.provider_invocation_count for value in executions),
            typed_side_effect_count_before_replay=sum(value.typed_before for value in executions),
            typed_side_effect_count_after_replay=sum(value.typed_after for value in executions),
            physical_file_count_before_replay=sum(value.physical_files_before for value in executions),
            physical_file_count_after_replay=sum(value.physical_files_after for value in executions),
            physical_bytes_before_replay=sum(value.physical_bytes_before for value in executions),
            physical_bytes_after_replay=sum(value.physical_bytes_after for value in executions),
            exact_replay_stable=all(value.replay_stable for value in executions),
        )
        results = tuple(
            ConcurrencyExperimentRecoveryResultV1.create(
                fault_observation=execution.fault_observation,
                case_result=execution.case_result,
                summary=summary,
                audit=audit,
            )
            for execution in executions
        )
        if self._material_store is not None:
            for execution, result in zip(executions, results, strict=True):
                self._material_store.put_fault_observation(execution.fault_observation)
                self._material_store.put_recovery_result(result)
        return results, summary

    def _persist_aggregate(
        self,
        *,
        ledger: ConcurrencyExperimentRunLedgerV1,
        level_results: tuple[ConcurrencyExperimentLevelResultV1, ...],
        recovery_results: tuple[ConcurrencyExperimentRecoveryResultV1, ...],
        result_set: ConcurrencyExperimentResultSetV1,
    ) -> None:
        del level_results, recovery_results
        if self._material_store is not None:
            self._material_store.put_run_ledger(ledger)
            self._material_store.put_result_set(result_set)


def _level_summary(
    worker_count: int,
    trial_results: tuple[ConcurrencyExperimentTrialResultV1, ...],
    samples: tuple[ConcurrencyExperimentCaseSampleV1, ...],
) -> ConcurrencyExperimentLevelSummaryV2:
    metrics = tuple(metric for sample in samples for metric in sample.attempt_metrics)
    stage_run_count = sum(sample.case_result.stage_run_count for sample in samples)
    stage_result_count = sum(len(sample.case_result.stage_result_refs) for sample in samples)
    provider_invocation_count = sum(sample.provider_invocation_count for sample in samples)
    retry_count = sum(sample.case_result.retry_count for sample in samples)
    resume_count = sum(sample.case_result.resume_count for sample in samples)
    attempt_count = sum(sample.case_result.attempt_count for sample in samples)
    succeeded = sum(
        sample.case_result.observed_outcome is CanaryRegressionObservedOutcomeV2.SUCCEEDED
        for sample in samples
    )
    product_errors = sum(
        sample.case_result.observed_outcome
        in {
            CanaryRegressionObservedOutcomeV2.BLOCKED,
            CanaryRegressionObservedOutcomeV2.FAILED,
        }
        for sample in samples
    )
    infrastructure = sum(
        sample.case_result.observed_outcome is CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR
        for sample in samples
    )
    incomplete = sum(
        sample.case_result.observed_outcome is CanaryRegressionObservedOutcomeV2.INCOMPLETE
        for sample in samples
    )
    metrics_complete = (
        len(metrics) == 720
        and all(metric.model_usage is None for metric in metrics)
        and all(metric.cache_use is CacheUseV2.NOT_APPLICABLE for metric in metrics)
        and all(metric.cost.availability is MetricAvailabilityV2.UNAVAILABLE for metric in metrics)
        and all(metric.resource_usage is not None for metric in metrics)
    )
    exact_r6_counts = stage_run_count == 648 and stage_result_count == 648 and provider_invocation_count == 72
    reasons: tuple[ConcurrencyExperimentReasonCodeV2, ...]
    if succeeded == 72 and metrics_complete and exact_r6_counts and retry_count == 0 and resume_count == 0:
        outcome = ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE
        reasons = (ConcurrencyExperimentReasonCodeV2.NONE,)
    elif incomplete:
        outcome = ConcurrencyExperimentLevelOutcomeV2.INCOMPLETE
        reasons = (ConcurrencyExperimentReasonCodeV2.CASE_INCOMPLETE,)
    elif infrastructure:
        integrity_failure = any(
            sample.case_result.reason_code is CanaryRegressionReasonCodeV2.MATERIAL_INTEGRITY_ERROR
            for sample in samples
        )
        outcome = (
            ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_INTEGRITY
            if integrity_failure
            else ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_ERROR
        )
        reasons = (
            (
                ConcurrencyExperimentReasonCodeV2.MATERIAL_INTEGRITY_ERROR
                if integrity_failure
                else ConcurrencyExperimentReasonCodeV2.CASE_INFRASTRUCTURE_ERROR
            ),
        )
    elif not metrics_complete or not exact_r6_counts:
        outcome = ConcurrencyExperimentLevelOutcomeV2.INCOMPLETE
        reasons = (ConcurrencyExperimentReasonCodeV2.METRICS_INCOMPLETE,)
    elif retry_count or resume_count:
        outcome = ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_ERROR
        reasons = (ConcurrencyExperimentReasonCodeV2.UNEXPECTED_RETRY,)
    else:
        outcome = ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_ERROR
        reasons = (ConcurrencyExperimentReasonCodeV2.CASE_FAILED,)
    model_requests = sum(metric.model_usage.requests for metric in metrics if metric.model_usage is not None)
    input_tokens = sum(
        metric.model_usage.input_tokens for metric in metrics if metric.model_usage is not None
    )
    output_tokens = sum(
        metric.model_usage.output_tokens for metric in metrics if metric.model_usage is not None
    )
    resources = tuple(
        metric.resource_usage.observed for metric in metrics if metric.resource_usage is not None
    )
    return ConcurrencyExperimentLevelSummaryV2.create(
        worker_count=worker_count,
        trial_count=len(trial_results),
        unique_case_count=24,
        executed_case_count=len(samples),
        succeeded_case_count=succeeded,
        product_error_count=product_errors,
        infrastructure_error_count=infrastructure,
        incomplete_case_count=incomplete,
        attempt_count=attempt_count,
        attempt_metric_count=len(metrics),
        stage_run_count=stage_run_count,
        stage_result_count=stage_result_count,
        provider_invocation_count=provider_invocation_count,
        retry_count=retry_count,
        resume_count=resume_count,
        model_request_count=model_requests,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        process_start_count=sum(value.processes for value in resources),
        renderer_operation_count=sum(value.renderers for value in resources),
        network_request_count=sum(value.network_requests for value in resources),
        retained_storage_bytes=sum(value.storage_bytes for value in resources),
        tool_call_count=sum(tool.calls for metric in metrics for tool in metric.tool_metrics)
        + sum(
            tool.calls
            for metric in metrics
            for artifact in metric.artifact_slices
            for tool in artifact.tool_metrics
        ),
        unavailable_cost_attempt_count=sum(
            metric.cost.availability is MetricAvailabilityV2.UNAVAILABLE for metric in metrics
        ),
        elapsed_nanoseconds=_summary(
            tuple(value.elapsed_nanoseconds for value in trial_results),
            ConcurrencyExperimentUnitV2.NANOSECONDS,
        ),
        throughput_millijobs_per_second=_summary(
            tuple(value.throughput_millijobs_per_second for value in trial_results),
            ConcurrencyExperimentUnitV2.MILLIJOBS_PER_SECOND,
        ),
        case_dispatch_wait_nanoseconds=_summary(
            tuple(value.dispatch_wait_nanoseconds for value in samples),
            ConcurrencyExperimentUnitV2.NANOSECONDS,
        ),
        case_wall_latency_nanoseconds=_summary(
            tuple(value.wall_latency_nanoseconds for value in samples),
            ConcurrencyExperimentUnitV2.NANOSECONDS,
        ),
        queue_wait_microseconds=_summary(
            tuple(value.queue_wait_us for value in metrics),
            ConcurrencyExperimentUnitV2.MICROSECONDS,
        ),
        execution_microseconds=_summary(
            tuple(value.execution_duration_us for value in metrics),
            ConcurrencyExperimentUnitV2.MICROSECONDS,
        ),
        attempt_active_total_microseconds=_summary(
            tuple(value.queue_wait_us + value.execution_duration_us for value in metrics),
            ConcurrencyExperimentUnitV2.MICROSECONDS,
        ),
        outcome=outcome,
        reason_codes=reasons,
    )


def _summary(
    values: tuple[int, ...],
    unit: ConcurrencyExperimentUnitV2,
) -> ConcurrencyExperimentPercentileSummaryV2:
    return ConcurrencyExperimentPercentileSummaryV2.create(
        samples=values,
        unit=unit,
    )


def _closed_exception_execution(
    prepared: (PreparedConcurrencyExperimentCase | PreparedConcurrencyRecoveryCase),
    *,
    error: Exception,
    audit: ContractAudit,
) -> ConcurrencyCaseExecution:
    return _closed_execution(
        prepared,
        reason=_error_reason(error).value,
        audit=audit,
    )


def _closed_wait_execution(
    prepared: PreparedConcurrencyExperimentCase,
    *,
    audit: ContractAudit,
) -> ConcurrencyCaseExecution:
    return _closed_execution(
        prepared,
        reason="RESOURCE_WAITING",
        audit=audit,
    )


def _closed_execution(
    prepared: (PreparedConcurrencyExperimentCase | PreparedConcurrencyRecoveryCase),
    *,
    reason: str,
    audit: ContractAudit,
) -> ConcurrencyCaseExecution:
    from eval_factory.contracts.canary_execution_v2 import (
        r6_canary_execution_manifest_v2_ref,
    )
    from eval_factory.contracts.canary_regression_v2 import (
        CanaryRegressionReasonCodeV2,
    )

    manifest = prepared.prepared_case.child_manifest
    binding = manifest.trace_bindings[0]
    result = CanaryRegressionCaseResultV2.create(
        instance_id=prepared.frozen_case.instance_id,
        source_trace_ref=binding.source_trace_ref,
        signal_quality=prepared.frozen_case.signal_quality,
        expectation=prepared.prepared_case.expectation,
        observed_outcome=(CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR),
        reason_code=CanaryRegressionReasonCodeV2(reason),
        highest_reached_stage=None,
        child_manifest_ref=r6_canary_execution_manifest_v2_ref(manifest),
        child_dataset_result_ref=None,
        job_id=manifest.job_spec.job_id,
        job_status=None,
        item_id=None,
        item_status=None,
        stage_result_refs=(),
        quality_result_refs=(),
        package_manifest_refs=(),
        audit_report_ref=None,
        work_unit_count=0,
        work_lease_count=0,
        stage_run_count=0,
        attempt_count=0,
        retry_count=0,
        resume_count=0,
        provider_invocation_count=0,
        audit=audit,
    )
    return ConcurrencyCaseExecution(case_result=result, attempt_metrics=())


def _driver(
    job_store_path: Path,
    root: Path,
    *,
    fault_injector: StaticCanaryDriverFaultInjector | None = None,
) -> CanaryPipelineDriver:
    return CanaryPipelineDriver(
        job_store=JobStore(job_store_path),
        source_registry=TraceSourceRegistry(root / "source.sqlite3"),
        trace_store=TraceIndexStore(root / "trace-store"),
        stage_store=create_canary_stage_store(root / "stage-store"),
        fault_injector=fault_injector,
    )


def _job_evidence_count(store: JobStore, job_id: str) -> int:
    runs = store.list_stage_runs(job_id=job_id)
    work_units = store.list_work_units(job_id=job_id)
    return sum(
        [
            1,
            len(store.list_items(job_id)),
            len(work_units),
            sum(len(store.list_work_readiness(unit.resolved_work_unit_id)) for unit in work_units),
            len(store.list_work_leases(job_id=job_id)),
            len(runs),
            sum(store.get_stage_result_for_run(run.stage_run_id) is not None for run in runs),
            len(store.list_work_attempt_metrics(job_id=job_id)),
            sum(event.aggregate_id == job_id for event in store.list_outbox()),
        ]
    )


def _physical_snapshot(path: Path) -> tuple[int, int]:
    files = tuple(
        candidate
        for candidate in (
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
        )
        if candidate.is_file()
    )
    return len(files), sum(candidate.stat().st_size for candidate in files)


def _recovery_physical_snapshot(
    job_store_path: Path,
    child_root: Path,
) -> tuple[int, int]:
    job_files, job_bytes = _physical_snapshot(job_store_path)
    child_files = tuple(
        candidate for candidate in child_root.rglob("*") if candidate.is_file() or candidate.is_symlink()
    )
    return (
        job_files + len(child_files),
        job_bytes + sum(candidate.lstat().st_size for candidate in child_files),
    )


def _job_store_ref(
    workload: ConcurrencyExperimentWorkloadV1,
    trial: ConcurrencyExperimentTrialSpecV1,
) -> ObjectRef:
    digest = hashlib.sha256(
        json.dumps(
            {
                "workload": workload.workload_sha256,
                "trial": trial.trial_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return ObjectRef(
        object_type="concurrency-experiment-job-store",
        object_id=f"concurrency-experiment-job-store://sha256/{digest}",
        object_version="private-v1",
        object_sha256=digest,
    )


def _recovery_job_store_ref(
    workload: ConcurrencyExperimentWorkloadV1,
) -> ObjectRef:
    digest = hashlib.sha256(
        json.dumps(
            {
                "workload": workload.workload_sha256,
                "partition": "recovery",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return ObjectRef(
        object_type="concurrency-experiment-job-store",
        object_id=f"concurrency-experiment-job-store://sha256/{digest}",
        object_version="private-v1",
        object_sha256=digest,
    )


def _private_root(path: Path) -> Path:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ConcurrencyExperimentRunError("concurrency experiment root must be a non-symlink directory")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


__all__ = [
    "ConcurrencyCaseExecution",
    "ConcurrencyCaseExecutor",
    "ConcurrencyExperimentCapacityWait",
    "ConcurrencyExperimentRunError",
    "ConcurrencyExperimentRunResult",
    "ConcurrencyExperimentRunner",
    "ConcurrencyRecoveryExecution",
    "ConcurrencyRecoveryExecutor",
    "R6ConcurrencyCaseExecutor",
    "R6ConcurrencyRecoveryExecutor",
]
