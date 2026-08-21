from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Self, cast

from pydantic import Field, model_validator

from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryExecutionManifestV2,
    r6_canary_execution_manifest_v2_ref,
)
from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionCaseResultV2,
    CanaryRegressionObservedOutcomeV2,
    canary_regression_case_result_v2_ref,
)
from eval_factory.contracts.concurrency_experiment_v2 import (
    CONCURRENCY_EXPERIMENT_CASES,
    CONCURRENCY_EXPERIMENT_POLICY_VERSION,
    ConcurrencyExperimentFaultPointV2,
    ConcurrencyExperimentHostSummaryV2,
    ConcurrencyExperimentLevelSummaryV2,
    ConcurrencyExperimentRecoverySummaryV2,
    concurrency_experiment_host_summary_v2_carried_sha256,
)
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.observability_v2 import (
    WorkAttemptMetricsV2,
    work_attempt_metrics_v2_ref,
)
from eval_factory.readiness.canary_regression_models import (
    FrozenCanaryCaseV1,
    PreparedCanaryRegressionCase,
)

CONCURRENCY_TRIAL_SCHEDULE: tuple[tuple[int, int], ...] = (
    (1, 0),
    (2, 0),
    (4, 0),
    (8, 0),
    (16, 0),
    (16, 1),
    (8, 1),
    (4, 1),
    (2, 1),
    (1, 1),
    (4, 2),
    (8, 2),
    (16, 2),
    (1, 2),
    (2, 2),
)


class ConcurrencyExperimentTrialSpecV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-trial-spec/private-v1"] = (
        "eval-factory/concurrency-experiment-trial-spec/private-v1"
    )
    trial_id: Identifier
    schedule_ordinal: int = Field(ge=0, lt=15)
    worker_count: Literal[1, 2, 4, 8, 16]
    trial_index: int = Field(ge=0, lt=3)
    trial_sha256: Sha256

    @model_validator(mode="after")
    def validate_trial(self) -> Self:
        if (
            self.worker_count,
            self.trial_index,
        ) != CONCURRENCY_TRIAL_SCHEDULE[self.schedule_ordinal]:
            raise ValueError("concurrency trial schedule is not canonical")
        _validate_identity(
            self.trial_id,
            self.trial_sha256,
            "concurrency-experiment-trial",
            concurrency_experiment_trial_spec_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        schedule_ordinal: int,
        worker_count: int,
        trial_index: int,
    ) -> ConcurrencyExperimentTrialSpecV1:
        value = cls(
            trial_id="concurrency-experiment-trial://pending",
            schedule_ordinal=schedule_ordinal,
            worker_count=cast(Literal[1, 2, 4, 8, 16], worker_count),
            trial_index=trial_index,
            trial_sha256="0" * 64,
        )
        return _finalize(
            value,
            "trial_id",
            "trial_sha256",
            "concurrency-experiment-trial",
            concurrency_experiment_trial_spec_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return _private_ref(
            "concurrency-experiment-trial",
            self.trial_id,
            self.trial_sha256,
        )


class ConcurrencyExperimentWorkloadV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-workload/private-v1"] = (
        "eval-factory/concurrency-experiment-workload/private-v1"
    )
    workload_id: Identifier
    policy_ref: ObjectRef
    cohort_ref: ObjectRef
    template_ref: ObjectRef
    host_summary_sha256: Sha256
    trials: tuple[ConcurrencyExperimentTrialSpecV1, ...] = Field(
        min_length=15,
        max_length=15,
    )
    retained_storage_bytes_per_case: int = Field(ge=1, le=10**12)
    retained_storage_bytes_per_cohort: int = Field(ge=24, le=10**15)
    policy_version: Literal["concurrency-experiment/r8-06-v1"] = CONCURRENCY_EXPERIMENT_POLICY_VERSION
    workload_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_workload(self) -> Self:
        _require_ref(
            self.policy_ref,
            "concurrency-experiment-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.cohort_ref,
            "frozen-canary-cohort",
            "private-v1",
            "cohort_ref",
        )
        _require_ref(
            self.template_ref,
            "canary-regression-template",
            "private-v1",
            "template_ref",
        )
        if (
            tuple((trial.worker_count, trial.trial_index) for trial in self.trials)
            != CONCURRENCY_TRIAL_SCHEDULE
        ):
            raise ValueError("concurrency workload trials are not canonical")
        if self.retained_storage_bytes_per_cohort < (
            self.retained_storage_bytes_per_case * CONCURRENCY_EXPERIMENT_CASES
        ):
            raise ValueError("concurrency workload cohort storage is too small")
        refs = (self.policy_ref, self.cohort_ref, self.template_ref)
        _require_audit(self.audit, refs, "concurrency workload")
        _validate_identity(
            self.workload_id,
            self.workload_sha256,
            "concurrency-experiment-workload",
            concurrency_experiment_workload_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        cohort_ref: ObjectRef,
        template_ref: ObjectRef,
        host_summary: ConcurrencyExperimentHostSummaryV2,
        retained_storage_bytes_per_case: int,
        retained_storage_bytes_per_cohort: int,
        audit: ContractAudit,
    ) -> ConcurrencyExperimentWorkloadV1:
        trials = tuple(
            ConcurrencyExperimentTrialSpecV1.create(
                schedule_ordinal=ordinal,
                worker_count=worker_count,
                trial_index=trial_index,
            )
            for ordinal, (worker_count, trial_index) in enumerate(CONCURRENCY_TRIAL_SCHEDULE)
        )
        refs = (policy_ref, cohort_ref, template_ref)
        value = cls(
            workload_id="concurrency-experiment-workload://pending",
            policy_ref=policy_ref,
            cohort_ref=cohort_ref,
            template_ref=template_ref,
            host_summary_sha256=(concurrency_experiment_host_summary_v2_carried_sha256(host_summary)),
            trials=trials,
            retained_storage_bytes_per_case=retained_storage_bytes_per_case,
            retained_storage_bytes_per_cohort=retained_storage_bytes_per_cohort,
            workload_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "workload_id",
            "workload_sha256",
            "concurrency-experiment-workload",
            concurrency_experiment_workload_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_workload_v1_ref(self)


@dataclass(frozen=True, slots=True)
class PreparedConcurrencyExperimentCase:
    trial: ConcurrencyExperimentTrialSpecV1
    case_ordinal: int
    frozen_case: FrozenCanaryCaseV1
    prepared_case: PreparedCanaryRegressionCase

    def __post_init__(self) -> None:
        if not 0 <= self.case_ordinal < CONCURRENCY_EXPERIMENT_CASES:
            raise ValueError("concurrency case ordinal is out of range")
        if self.frozen_case != self.prepared_case.cohort_case:
            raise ValueError("prepared concurrency case differs from frozen case")


@dataclass(frozen=True, slots=True)
class PreparedConcurrencyRecoveryCase:
    worker_count: int
    case_ordinal: int
    fault_point: ConcurrencyExperimentFaultPointV2
    frozen_case: FrozenCanaryCaseV1
    prepared_case: PreparedCanaryRegressionCase

    def __post_init__(self) -> None:
        if self.worker_count not in {1, 2, 4, 8, 16}:
            raise ValueError("recovery worker count is not in the staircase")
        if not 0 <= self.case_ordinal < CONCURRENCY_EXPERIMENT_CASES:
            raise ValueError("recovery case ordinal is out of range")
        if self.frozen_case != self.prepared_case.cohort_case:
            raise ValueError("prepared recovery case differs from frozen case")


class ConcurrencyExperimentCaseSampleV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-case-sample/private-v1"] = (
        "eval-factory/concurrency-experiment-case-sample/private-v1"
    )
    case_sample_id: Identifier
    trial_ref: ObjectRef
    case_ordinal: int = Field(ge=0, lt=24)
    instance_id: Identifier
    child_manifest_ref: ObjectRef
    case_result: CanaryRegressionCaseResultV2
    attempt_metrics: tuple[WorkAttemptMetricsV2, ...]
    submitted_nanoseconds: int = Field(ge=0, le=10**18)
    started_nanoseconds: int = Field(ge=0, le=10**18)
    completed_nanoseconds: int = Field(ge=1, le=10**18)
    dispatch_wait_nanoseconds: int = Field(ge=0, le=10**18)
    wall_latency_nanoseconds: int = Field(ge=1, le=10**18)
    provider_invocation_count: int = Field(ge=0, le=100_000)
    policy_version: Literal["concurrency-experiment/r8-06-v1"] = CONCURRENCY_EXPERIMENT_POLICY_VERSION
    case_sample_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_sample(self) -> Self:
        _require_ref(
            self.trial_ref,
            "concurrency-experiment-trial",
            "private-v1",
            "trial_ref",
        )
        _require_ref(
            self.child_manifest_ref,
            "r6-canary-execution-manifest",
            "v2",
            "child_manifest_ref",
        )
        if not (self.submitted_nanoseconds <= self.started_nanoseconds < self.completed_nanoseconds):
            raise ValueError("concurrency case timing is not monotonic")
        if (
            self.dispatch_wait_nanoseconds != self.started_nanoseconds - self.submitted_nanoseconds
            or self.wall_latency_nanoseconds != self.completed_nanoseconds - self.started_nanoseconds
        ):
            raise ValueError("concurrency case timing fields are not derived")
        if self.case_result.instance_id != self.instance_id:
            raise ValueError("concurrency case result uses another instance")
        if self.case_result.child_manifest_ref != self.child_manifest_ref:
            raise ValueError("concurrency case result uses another manifest")
        metric_count = len(self.attempt_metrics)
        if metric_count > self.case_result.attempt_count:
            raise ValueError("concurrency metrics exceed durable attempts")
        if (
            self.case_result.observed_outcome is CanaryRegressionObservedOutcomeV2.SUCCEEDED
            and metric_count != self.case_result.attempt_count
        ):
            raise ValueError("successful concurrency case requires complete attempt metrics")
        refs = (
            self.trial_ref,
            self.child_manifest_ref,
            canary_regression_case_result_v2_ref(self.case_result),
            *(work_attempt_metrics_v2_ref(value) for value in self.attempt_metrics),
        )
        _require_audit(self.audit, refs, "concurrency case sample")
        _validate_identity(
            self.case_sample_id,
            self.case_sample_sha256,
            "concurrency-experiment-case-sample",
            concurrency_experiment_case_sample_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        trial_ref: ObjectRef,
        case_ordinal: int,
        child_manifest: R6CanaryExecutionManifestV2,
        case_result: CanaryRegressionCaseResultV2,
        attempt_metrics: tuple[WorkAttemptMetricsV2, ...],
        submitted_nanoseconds: int,
        started_nanoseconds: int,
        completed_nanoseconds: int,
        provider_invocation_count: int,
        audit: ContractAudit,
    ) -> ConcurrencyExperimentCaseSampleV1:
        child_ref = r6_canary_execution_manifest_v2_ref(child_manifest)
        metrics = tuple(
            sorted(
                attempt_metrics,
                key=lambda value: value.work_attempt_metrics_id,
            )
        )
        refs = (
            trial_ref,
            child_ref,
            canary_regression_case_result_v2_ref(case_result),
            *(work_attempt_metrics_v2_ref(value) for value in metrics),
        )
        value = cls(
            case_sample_id="concurrency-experiment-case-sample://pending",
            trial_ref=trial_ref,
            case_ordinal=case_ordinal,
            instance_id=case_result.instance_id,
            child_manifest_ref=child_ref,
            case_result=case_result,
            attempt_metrics=metrics,
            submitted_nanoseconds=submitted_nanoseconds,
            started_nanoseconds=started_nanoseconds,
            completed_nanoseconds=completed_nanoseconds,
            dispatch_wait_nanoseconds=(started_nanoseconds - submitted_nanoseconds),
            wall_latency_nanoseconds=(completed_nanoseconds - started_nanoseconds),
            provider_invocation_count=provider_invocation_count,
            case_sample_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "case_sample_id",
            "case_sample_sha256",
            "concurrency-experiment-case-sample",
            concurrency_experiment_case_sample_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_case_sample_v1_ref(self)


class ConcurrencyExperimentTrialResultV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-trial-result/private-v1"] = (
        "eval-factory/concurrency-experiment-trial-result/private-v1"
    )
    trial_result_id: Identifier
    trial_ref: ObjectRef
    case_sample_refs: tuple[ObjectRef, ...] = Field(
        min_length=24,
        max_length=24,
    )
    elapsed_nanoseconds: int = Field(ge=1, le=10**18)
    throughput_millijobs_per_second: int = Field(ge=1, le=10**18)
    typed_side_effect_count: int = Field(ge=0, le=10**12)
    physical_file_count: int = Field(ge=0, le=10**9)
    physical_bytes: int = Field(ge=0, le=10**18)
    trial_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_trial_result(self) -> Self:
        _require_ref(
            self.trial_ref,
            "concurrency-experiment-trial",
            "private-v1",
            "trial_ref",
        )
        _require_sorted_unique_refs(
            self.case_sample_refs,
            "concurrency-experiment-case-sample",
            "case_sample_refs",
        )
        expected = CONCURRENCY_EXPERIMENT_CASES * 1_000_000_000_000 // self.elapsed_nanoseconds
        if expected < 1 or self.throughput_millijobs_per_second != expected:
            raise ValueError("concurrency trial throughput is not derived")
        refs = (self.trial_ref, *self.case_sample_refs)
        _require_audit(self.audit, refs, "concurrency trial result")
        _validate_identity(
            self.trial_result_id,
            self.trial_result_sha256,
            "concurrency-experiment-trial-result",
            concurrency_experiment_trial_result_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        trial_ref: ObjectRef,
        case_samples: tuple[ConcurrencyExperimentCaseSampleV1, ...],
        elapsed_nanoseconds: int,
        typed_side_effect_count: int,
        physical_file_count: int,
        physical_bytes: int,
        audit: ContractAudit,
    ) -> ConcurrencyExperimentTrialResultV1:
        refs = _sorted_refs(tuple(value.to_ref() for value in case_samples))
        value = cls(
            trial_result_id="concurrency-experiment-trial-result://pending",
            trial_ref=trial_ref,
            case_sample_refs=refs,
            elapsed_nanoseconds=elapsed_nanoseconds,
            throughput_millijobs_per_second=(
                CONCURRENCY_EXPERIMENT_CASES * 1_000_000_000_000 // elapsed_nanoseconds
                if elapsed_nanoseconds > 0
                else 0
            ),
            typed_side_effect_count=typed_side_effect_count,
            physical_file_count=physical_file_count,
            physical_bytes=physical_bytes,
            trial_result_sha256="0" * 64,
            audit=_safe_audit(audit, (trial_ref, *refs)),
        )
        return _finalize(
            value,
            "trial_result_id",
            "trial_result_sha256",
            "concurrency-experiment-trial-result",
            concurrency_experiment_trial_result_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_trial_result_v1_ref(self)


class ConcurrencyExperimentLevelResultV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-level-result/private-v1"] = (
        "eval-factory/concurrency-experiment-level-result/private-v1"
    )
    level_result_id: Identifier
    worker_count: Literal[1, 2, 4, 8, 16]
    trial_result_refs: tuple[ObjectRef, ...] = Field(
        min_length=3,
        max_length=3,
    )
    summary: ConcurrencyExperimentLevelSummaryV2
    level_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_level_result(self) -> Self:
        _require_sorted_unique_refs(
            self.trial_result_refs,
            "concurrency-experiment-trial-result",
            "trial_result_refs",
        )
        if self.summary.worker_count != self.worker_count:
            raise ValueError("level result summary uses another worker count")
        _require_audit(
            self.audit,
            self.trial_result_refs,
            "concurrency level result",
        )
        _validate_identity(
            self.level_result_id,
            self.level_result_sha256,
            "concurrency-experiment-level-result",
            concurrency_experiment_level_result_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        worker_count: int,
        trial_results: tuple[ConcurrencyExperimentTrialResultV1, ...],
        summary: ConcurrencyExperimentLevelSummaryV2,
        audit: ContractAudit,
    ) -> ConcurrencyExperimentLevelResultV1:
        refs = _sorted_refs(tuple(value.to_ref() for value in trial_results))
        value = cls(
            level_result_id="concurrency-experiment-level-result://pending",
            worker_count=cast(Literal[1, 2, 4, 8, 16], worker_count),
            trial_result_refs=refs,
            summary=summary,
            level_result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "level_result_id",
            "level_result_sha256",
            "concurrency-experiment-level-result",
            concurrency_experiment_level_result_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_level_result_v1_ref(self)


class ConcurrencyExperimentFaultObservationV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-fault-observation/private-v1"] = (
        "eval-factory/concurrency-experiment-fault-observation/private-v1"
    )
    fault_observation_id: Identifier
    child_manifest_ref: ObjectRef
    fault_point: ConcurrencyExperimentFaultPointV2
    observed: bool
    fault_observation_sha256: Sha256

    @model_validator(mode="after")
    def validate_fault(self) -> Self:
        _require_ref(
            self.child_manifest_ref,
            "r6-canary-execution-manifest",
            "v2",
            "child_manifest_ref",
        )
        _validate_identity(
            self.fault_observation_id,
            self.fault_observation_sha256,
            "concurrency-experiment-fault-observation",
            concurrency_experiment_fault_observation_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        child_manifest_ref: ObjectRef,
        fault_point: ConcurrencyExperimentFaultPointV2,
        observed: bool,
    ) -> ConcurrencyExperimentFaultObservationV1:
        value = cls(
            fault_observation_id=("concurrency-experiment-fault-observation://pending"),
            child_manifest_ref=child_manifest_ref,
            fault_point=fault_point,
            observed=observed,
            fault_observation_sha256="0" * 64,
        )
        return _finalize(
            value,
            "fault_observation_id",
            "fault_observation_sha256",
            "concurrency-experiment-fault-observation",
            concurrency_experiment_fault_observation_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_fault_observation_v1_ref(self)


class ConcurrencyExperimentRecoveryResultV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-recovery-result/private-v1"] = (
        "eval-factory/concurrency-experiment-recovery-result/private-v1"
    )
    recovery_result_id: Identifier
    fault_observation_ref: ObjectRef
    case_result_ref: ObjectRef
    summary: ConcurrencyExperimentRecoverySummaryV2
    recovery_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_recovery(self) -> Self:
        _require_ref(
            self.fault_observation_ref,
            "concurrency-experiment-fault-observation",
            "private-v1",
            "fault_observation_ref",
        )
        _require_ref(
            self.case_result_ref,
            "canary-regression-case-result",
            "v2",
            "case_result_ref",
        )
        refs = (self.fault_observation_ref, self.case_result_ref)
        _require_audit(self.audit, refs, "concurrency recovery result")
        _validate_identity(
            self.recovery_result_id,
            self.recovery_result_sha256,
            "concurrency-experiment-recovery-result",
            concurrency_experiment_recovery_result_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        fault_observation: ConcurrencyExperimentFaultObservationV1,
        case_result: CanaryRegressionCaseResultV2,
        summary: ConcurrencyExperimentRecoverySummaryV2,
        audit: ContractAudit,
    ) -> ConcurrencyExperimentRecoveryResultV1:
        refs = (
            fault_observation.to_ref(),
            canary_regression_case_result_v2_ref(case_result),
        )
        value = cls(
            recovery_result_id="concurrency-experiment-recovery-result://pending",
            fault_observation_ref=refs[0],
            case_result_ref=refs[1],
            summary=summary,
            recovery_result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "recovery_result_id",
            "recovery_result_sha256",
            "concurrency-experiment-recovery-result",
            concurrency_experiment_recovery_result_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_recovery_result_v1_ref(self)


class ConcurrencyExperimentResultSetV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-result-set/private-v1"] = (
        "eval-factory/concurrency-experiment-result-set/private-v1"
    )
    result_set_id: Identifier
    workload_ref: ObjectRef
    level_result_refs: tuple[ObjectRef, ...] = Field(min_length=5, max_length=5)
    recovery_result_refs: tuple[ObjectRef, ...] = ()
    result_set_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result_set(self) -> Self:
        _require_ref(
            self.workload_ref,
            "concurrency-experiment-workload",
            "private-v1",
            "workload_ref",
        )
        _require_sorted_unique_refs(
            self.level_result_refs,
            "concurrency-experiment-level-result",
            "level_result_refs",
        )
        if self.recovery_result_refs:
            _require_sorted_unique_refs(
                self.recovery_result_refs,
                "concurrency-experiment-recovery-result",
                "recovery_result_refs",
            )
        refs = (
            self.workload_ref,
            *self.level_result_refs,
            *self.recovery_result_refs,
        )
        _require_audit(self.audit, refs, "concurrency result set")
        _validate_identity(
            self.result_set_id,
            self.result_set_sha256,
            "concurrency-experiment-result-set",
            concurrency_experiment_result_set_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        workload_ref: ObjectRef,
        level_results: tuple[ConcurrencyExperimentLevelResultV1, ...],
        recovery_results: tuple[ConcurrencyExperimentRecoveryResultV1, ...],
        audit: ContractAudit,
    ) -> ConcurrencyExperimentResultSetV1:
        level_refs = _sorted_refs(tuple(value.to_ref() for value in level_results))
        recovery_refs = _sorted_refs(tuple(value.to_ref() for value in recovery_results))
        refs = (workload_ref, *level_refs, *recovery_refs)
        value = cls(
            result_set_id="concurrency-experiment-result-set://pending",
            workload_ref=workload_ref,
            level_result_refs=level_refs,
            recovery_result_refs=recovery_refs,
            result_set_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "result_set_id",
            "result_set_sha256",
            "concurrency-experiment-result-set",
            concurrency_experiment_result_set_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_result_set_v1_ref(self)


class ConcurrencyExperimentRunLedgerV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-run-ledger/private-v1"] = (
        "eval-factory/concurrency-experiment-run-ledger/private-v1"
    )
    run_ledger_id: Identifier
    workload_ref: ObjectRef
    trial_job_store_refs: tuple[ObjectRef, ...] = Field(
        min_length=15,
        max_length=16,
    )
    child_manifest_refs: tuple[ObjectRef, ...] = Field(
        min_length=360,
        max_length=384,
    )
    run_ledger_sha256: Sha256

    @model_validator(mode="after")
    def validate_ledger(self) -> Self:
        _require_ref(
            self.workload_ref,
            "concurrency-experiment-workload",
            "private-v1",
            "workload_ref",
        )
        _require_sorted_unique_refs(
            self.trial_job_store_refs,
            "concurrency-experiment-job-store",
            "trial_job_store_refs",
        )
        _require_sorted_unique_refs(
            self.child_manifest_refs,
            "r6-canary-execution-manifest",
            "child_manifest_refs",
            version="v2",
        )
        _validate_identity(
            self.run_ledger_id,
            self.run_ledger_sha256,
            "concurrency-experiment-run-ledger",
            concurrency_experiment_run_ledger_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        workload_ref: ObjectRef,
        trial_job_store_refs: tuple[ObjectRef, ...],
        child_manifests: tuple[R6CanaryExecutionManifestV2, ...],
    ) -> ConcurrencyExperimentRunLedgerV1:
        value = cls(
            run_ledger_id="concurrency-experiment-run-ledger://pending",
            workload_ref=workload_ref,
            trial_job_store_refs=_sorted_refs(trial_job_store_refs),
            child_manifest_refs=_sorted_refs(
                tuple(r6_canary_execution_manifest_v2_ref(manifest) for manifest in child_manifests)
            ),
            run_ledger_sha256="0" * 64,
        )
        return _finalize(
            value,
            "run_ledger_id",
            "run_ledger_sha256",
            "concurrency-experiment-run-ledger",
            concurrency_experiment_run_ledger_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_run_ledger_v1_ref(self)


def concurrency_experiment_trial_spec_v1_carried_sha256(
    value: ConcurrencyExperimentTrialSpecV1,
) -> str:
    return _carried(value, {"trial_id", "trial_sha256"})


def concurrency_experiment_workload_v1_carried_sha256(
    value: ConcurrencyExperimentWorkloadV1,
) -> str:
    return _carried(value, {"workload_id", "workload_sha256", "audit"})


def concurrency_experiment_case_sample_v1_carried_sha256(
    value: ConcurrencyExperimentCaseSampleV1,
) -> str:
    return _carried(value, {"case_sample_id", "case_sample_sha256", "audit"})


def concurrency_experiment_trial_result_v1_carried_sha256(
    value: ConcurrencyExperimentTrialResultV1,
) -> str:
    return _carried(value, {"trial_result_id", "trial_result_sha256", "audit"})


def concurrency_experiment_level_result_v1_carried_sha256(
    value: ConcurrencyExperimentLevelResultV1,
) -> str:
    return _carried(value, {"level_result_id", "level_result_sha256", "audit"})


def concurrency_experiment_fault_observation_v1_carried_sha256(
    value: ConcurrencyExperimentFaultObservationV1,
) -> str:
    return _carried(
        value,
        {"fault_observation_id", "fault_observation_sha256"},
    )


def concurrency_experiment_recovery_result_v1_carried_sha256(
    value: ConcurrencyExperimentRecoveryResultV1,
) -> str:
    return _carried(
        value,
        {"recovery_result_id", "recovery_result_sha256", "audit"},
    )


def concurrency_experiment_result_set_v1_carried_sha256(
    value: ConcurrencyExperimentResultSetV1,
) -> str:
    return _carried(value, {"result_set_id", "result_set_sha256", "audit"})


def concurrency_experiment_run_ledger_v1_carried_sha256(
    value: ConcurrencyExperimentRunLedgerV1,
) -> str:
    return _carried(value, {"run_ledger_id", "run_ledger_sha256"})


def concurrency_experiment_workload_v1_ref(
    value: ConcurrencyExperimentWorkloadV1,
) -> ObjectRef:
    return _validated_private_ref(
        value.workload_id,
        value.workload_sha256,
        "concurrency-experiment-workload",
        concurrency_experiment_workload_v1_carried_sha256(value),
    )


def concurrency_experiment_case_sample_v1_ref(
    value: ConcurrencyExperimentCaseSampleV1,
) -> ObjectRef:
    return _validated_private_ref(
        value.case_sample_id,
        value.case_sample_sha256,
        "concurrency-experiment-case-sample",
        concurrency_experiment_case_sample_v1_carried_sha256(value),
    )


def concurrency_experiment_trial_result_v1_ref(
    value: ConcurrencyExperimentTrialResultV1,
) -> ObjectRef:
    return _validated_private_ref(
        value.trial_result_id,
        value.trial_result_sha256,
        "concurrency-experiment-trial-result",
        concurrency_experiment_trial_result_v1_carried_sha256(value),
    )


def concurrency_experiment_level_result_v1_ref(
    value: ConcurrencyExperimentLevelResultV1,
) -> ObjectRef:
    return _validated_private_ref(
        value.level_result_id,
        value.level_result_sha256,
        "concurrency-experiment-level-result",
        concurrency_experiment_level_result_v1_carried_sha256(value),
    )


def concurrency_experiment_fault_observation_v1_ref(
    value: ConcurrencyExperimentFaultObservationV1,
) -> ObjectRef:
    return _validated_private_ref(
        value.fault_observation_id,
        value.fault_observation_sha256,
        "concurrency-experiment-fault-observation",
        concurrency_experiment_fault_observation_v1_carried_sha256(value),
    )


def concurrency_experiment_recovery_result_v1_ref(
    value: ConcurrencyExperimentRecoveryResultV1,
) -> ObjectRef:
    return _validated_private_ref(
        value.recovery_result_id,
        value.recovery_result_sha256,
        "concurrency-experiment-recovery-result",
        concurrency_experiment_recovery_result_v1_carried_sha256(value),
    )


def concurrency_experiment_result_set_v1_ref(
    value: ConcurrencyExperimentResultSetV1,
) -> ObjectRef:
    return _validated_private_ref(
        value.result_set_id,
        value.result_set_sha256,
        "concurrency-experiment-result-set",
        concurrency_experiment_result_set_v1_carried_sha256(value),
    )


def concurrency_experiment_run_ledger_v1_ref(
    value: ConcurrencyExperimentRunLedgerV1,
) -> ObjectRef:
    return _validated_private_ref(
        value.run_ledger_id,
        value.run_ledger_sha256,
        "concurrency-experiment-run-ledger",
        concurrency_experiment_run_ledger_v1_carried_sha256(value),
    )


def _validated_private_ref(
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
) -> ObjectRef:
    _validate_identity(
        object_id,
        object_sha256,
        prefix,
        observed,
        allow_pending=False,
    )
    return _private_ref(prefix, object_id, object_sha256)


def _private_ref(
    object_type: str,
    object_id: str,
    object_sha256: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="private-v1",
        object_sha256=object_sha256,
    )


def _require_ref(
    value: ObjectRef,
    object_type: str,
    version: str,
    name: str,
) -> None:
    if value.object_type != object_type or value.object_version != version:
        raise ValueError(f"{name} has the wrong type or version")


def _require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    object_type: str,
    name: str,
    *,
    version: str = "private-v1",
) -> None:
    if values != _sorted_refs(values) or len(values) != len(set(values)):
        raise ValueError(f"{name} must be sorted and unique")
    if any(value.object_type != object_type or value.object_version != version for value in values):
        raise ValueError(f"{name} contains an invalid ref")


def _require_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError(f"{label} audit refs are incomplete")
    versions = tuple(
        binding for binding in audit.governing_versions if binding.component == "concurrency-experiment"
    )
    if len(versions) != 1 or versions[0].version != CONCURRENCY_EXPERIMENT_POLICY_VERSION:
        raise ValueError(f"{label} audit is missing concurrency policy")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(values), key=_ref_key))


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _carried(value: ContractModelV2, exclude: set[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(
                value.model_dump(
                    mode="python",
                    exclude=exclude,
                    exclude_none=False,
                )
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    id_field: str,
    hash_field: str,
    prefix: str,
    digest: str,
) -> ModelT:
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            hash_field: digest,
        }
    )


def _validate_identity(
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
    *,
    allow_pending: bool = True,
) -> None:
    if allow_pending and object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_id != f"{prefix}://sha256/{observed}" or object_sha256 != observed:
        raise ValueError(f"{prefix} identity is stale")


__all__ = [
    "CONCURRENCY_TRIAL_SCHEDULE",
    "ConcurrencyExperimentCaseSampleV1",
    "ConcurrencyExperimentFaultObservationV1",
    "ConcurrencyExperimentLevelResultV1",
    "ConcurrencyExperimentRecoveryResultV1",
    "ConcurrencyExperimentResultSetV1",
    "ConcurrencyExperimentRunLedgerV1",
    "ConcurrencyExperimentTrialResultV1",
    "ConcurrencyExperimentTrialSpecV1",
    "ConcurrencyExperimentWorkloadV1",
    "PreparedConcurrencyExperimentCase",
    "PreparedConcurrencyRecoveryCase",
    "concurrency_experiment_case_sample_v1_ref",
    "concurrency_experiment_fault_observation_v1_ref",
    "concurrency_experiment_level_result_v1_ref",
    "concurrency_experiment_recovery_result_v1_ref",
    "concurrency_experiment_result_set_v1_ref",
    "concurrency_experiment_run_ledger_v1_ref",
    "concurrency_experiment_trial_result_v1_ref",
    "concurrency_experiment_workload_v1_ref",
]
