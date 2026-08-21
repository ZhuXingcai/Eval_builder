from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self, cast

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2

CONCURRENCY_EXPERIMENT_POLICY_VERSION: Literal["concurrency-experiment/r8-06-v1"] = (
    "concurrency-experiment/r8-06-v1"
)
CONCURRENCY_EXPERIMENT_CLAIM_SCOPE: Literal["CANARY_WORKER_STORAGE_ONLY"] = "CANARY_WORKER_STORAGE_ONLY"
CONCURRENCY_EXPERIMENT_LEVELS: tuple[Literal[1, 2, 4, 8, 16], ...] = (
    1,
    2,
    4,
    8,
    16,
)
CONCURRENCY_EXPERIMENT_TRIALS: Literal[3] = 3
CONCURRENCY_EXPERIMENT_CASES: Literal[24] = 24
CONCURRENCY_EXPERIMENT_EXECUTIONS_PER_LEVEL: Literal[72] = 72
CONCURRENCY_EXPERIMENT_RECOVERY_PROBES: Literal[24] = 24

R8_03_REPORT_SHA256: Literal["375b519dc3b40d3c8aab09d9a04574deca73ec80138c2fcaa2f1714eb1edbd4f"] = (
    "375b519dc3b40d3c8aab09d9a04574deca73ec80138c2fcaa2f1714eb1edbd4f"
)
R8_04_REPORT_SHA256: Literal["730f151435d2472cfe456e9df350c2663e8702951bd1eb1e8f0dcb45b315374f"] = (
    "730f151435d2472cfe456e9df350c2663e8702951bd1eb1e8f0dcb45b315374f"
)
R8_05_REPORT_SHA256: Literal["52061b69b7b418874709bddc71450111680e7931f4e04a337103b34b9f379bac"] = (
    "52061b69b7b418874709bddc71450111680e7931f4e04a337103b34b9f379bac"
)
R8_CANARY_COHORT_SHA256: Literal["1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6"] = (
    "1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6"
)


class ConcurrencyExperimentFaultPointV2(StrEnum):
    BEFORE_HANDLER = "BEFORE_HANDLER"
    AFTER_OBJECT_PERSISTENCE = "AFTER_OBJECT_PERSISTENCE"
    BEFORE_LEASE_COMPLETION = "BEFORE_LEASE_COMPLETION"
    AFTER_LEASE_COMPLETION = "AFTER_LEASE_COMPLETION"


class ConcurrencyExperimentOutcomeV2(StrEnum):
    NON_MODEL_RECOMMENDED = "NON_MODEL_RECOMMENDED"
    RECOMMENDATION_PENDING = "RECOMMENDATION_PENDING"
    INCOMPLETE = "INCOMPLETE"


class ConcurrencyExperimentLevelOutcomeV2(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE_ERROR = "INELIGIBLE_ERROR"
    INELIGIBLE_INTEGRITY = "INELIGIBLE_INTEGRITY"
    INCOMPLETE = "INCOMPLETE"


class ConcurrencyExperimentMetricAvailabilityV2(StrEnum):
    MEASURED = "MEASURED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class ConcurrencyExperimentRecommendationReasonV2(StrEnum):
    BALANCED_KNEE = "BALANCED_KNEE"
    NO_ELIGIBLE_LEVEL = "NO_ELIGIBLE_LEVEL"
    THROUGHPUT_THRESHOLD_NOT_MET = "THROUGHPUT_THRESHOLD_NOT_MET"
    LATENCY_THRESHOLD_NOT_MET = "LATENCY_THRESHOLD_NOT_MET"
    RECOVERY_VALIDATION_FAILED = "RECOVERY_VALIDATION_FAILED"
    MODEL_EVIDENCE_PENDING = "MODEL_EVIDENCE_PENDING"


class ConcurrencyExperimentReasonCodeV2(StrEnum):
    NONE = "NONE"
    CASE_FAILED = "CASE_FAILED"
    CASE_INFRASTRUCTURE_ERROR = "CASE_INFRASTRUCTURE_ERROR"
    CASE_INCOMPLETE = "CASE_INCOMPLETE"
    UNEXPECTED_RETRY = "UNEXPECTED_RETRY"
    METRICS_INCOMPLETE = "METRICS_INCOMPLETE"
    SAMPLE_COUNT_MISMATCH = "SAMPLE_COUNT_MISMATCH"
    CAPACITY_WAIT_EXHAUSTED = "CAPACITY_WAIT_EXHAUSTED"
    RECOVERY_FAILED = "RECOVERY_FAILED"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"
    MATERIAL_INTEGRITY_ERROR = "MATERIAL_INTEGRITY_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ConcurrencyExperimentUnitV2(StrEnum):
    NANOSECONDS = "NANOSECONDS"
    MICROSECONDS = "MICROSECONDS"
    MILLIJOBS_PER_SECOND = "MILLIJOBS_PER_SECOND"


class ConcurrencyExperimentPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-policy/v2"] = (
        "eval-factory/concurrency-experiment-policy/v2"
    )
    policy_id: Identifier
    canary_regression_report_ref: ObjectRef
    real_trace_stability_report_ref: ObjectRef
    scheduler_load_report_ref: ObjectRef
    cohort_manifest_ref: ObjectRef
    worker_levels: tuple[Literal[1, 2, 4, 8, 16], ...] = CONCURRENCY_EXPERIMENT_LEVELS
    trial_count_per_level: Literal[3] = CONCURRENCY_EXPERIMENT_TRIALS
    case_count_per_trial: Literal[24] = CONCURRENCY_EXPERIMENT_CASES
    recovery_probe_count: Literal[24] = CONCURRENCY_EXPERIMENT_RECOVERY_PROBES
    fault_points: tuple[ConcurrencyExperimentFaultPointV2, ...] = tuple(ConcurrencyExperimentFaultPointV2)
    throughput_efficiency_basis_points: Literal[9000] = 9_000
    p95_baseline_limit_basis_points: Literal[15000] = 15_000
    max_capacity_readmissions: int = Field(ge=1, le=10_000)
    max_private_bytes: int = Field(ge=2, le=10**12)
    max_report_bytes: int = Field(ge=2, le=100_000_000)
    policy_version: Literal["concurrency-experiment/r8-06-v1"] = CONCURRENCY_EXPERIMENT_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        expected = (
            (
                self.canary_regression_report_ref,
                "canary-regression-report",
                "v2",
                R8_03_REPORT_SHA256,
            ),
            (
                self.real_trace_stability_report_ref,
                "real-trace-stability-report",
                "v2",
                R8_04_REPORT_SHA256,
            ),
            (
                self.scheduler_load_report_ref,
                "scheduler-load-report",
                "v2",
                R8_05_REPORT_SHA256,
            ),
            (
                self.cohort_manifest_ref,
                "development-canary-manifest",
                "v4",
                R8_CANARY_COHORT_SHA256,
            ),
        )
        for ref, object_type, version, digest in expected:
            _require_ref(ref, object_type, version, object_type)
            if ref.object_sha256 != digest:
                raise ValueError(f"{object_type} does not bind accepted evidence")
        if self.worker_levels != CONCURRENCY_EXPERIMENT_LEVELS:
            raise ValueError("concurrency experiment worker levels are not canonical")
        if self.fault_points != tuple(ConcurrencyExperimentFaultPointV2):
            raise ValueError("concurrency experiment fault points are not canonical")
        refs = tuple(row[0] for row in expected)
        _require_audit(self.audit, refs, "concurrency experiment policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "concurrency-experiment-policy",
            concurrency_experiment_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        canary_regression_report_ref: ObjectRef,
        real_trace_stability_report_ref: ObjectRef,
        scheduler_load_report_ref: ObjectRef,
        cohort_manifest_ref: ObjectRef,
        max_capacity_readmissions: int,
        max_private_bytes: int,
        max_report_bytes: int,
        audit: ContractAudit,
    ) -> ConcurrencyExperimentPolicyV2:
        refs = (
            canary_regression_report_ref,
            real_trace_stability_report_ref,
            scheduler_load_report_ref,
            cohort_manifest_ref,
        )
        value = cls(
            policy_id="concurrency-experiment-policy://pending",
            canary_regression_report_ref=canary_regression_report_ref,
            real_trace_stability_report_ref=real_trace_stability_report_ref,
            scheduler_load_report_ref=scheduler_load_report_ref,
            cohort_manifest_ref=cohort_manifest_ref,
            max_capacity_readmissions=max_capacity_readmissions,
            max_private_bytes=max_private_bytes,
            max_report_bytes=max_report_bytes,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "concurrency-experiment-policy",
            concurrency_experiment_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_policy_v2_ref(self)


class ConcurrencyExperimentHostSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-host-summary/v2"] = (
        "eval-factory/concurrency-experiment-host-summary/v2"
    )
    platform_family: Literal["DARWIN"]
    architecture: Literal["ARM64"]
    logical_cpu_count: int = Field(ge=1, le=4096)
    physical_cpu_count: int = Field(ge=1, le=4096)
    memory_bytes: int = Field(ge=1, le=10**16)
    python_version: str = Field(min_length=1, max_length=64)
    host_summary_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_host(self) -> Self:
        if self.physical_cpu_count > self.logical_cpu_count:
            raise ValueError("physical CPU count cannot exceed logical CPU count")
        _require_audit(self.audit, (), "concurrency experiment host")
        observed = concurrency_experiment_host_summary_v2_carried_sha256(self)
        if self.host_summary_sha256 not in {"0" * 64, observed}:
            raise ValueError("concurrency experiment host identity is stale")
        return self

    @classmethod
    def create(
        cls,
        *,
        platform_family: Literal["DARWIN"],
        architecture: Literal["ARM64"],
        logical_cpu_count: int,
        physical_cpu_count: int,
        memory_bytes: int,
        python_version: str,
        audit: ContractAudit,
    ) -> ConcurrencyExperimentHostSummaryV2:
        value = cls(
            platform_family=platform_family,
            architecture=architecture,
            logical_cpu_count=logical_cpu_count,
            physical_cpu_count=physical_cpu_count,
            memory_bytes=memory_bytes,
            python_version=python_version,
            host_summary_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return value.model_copy(
            update={"host_summary_sha256": (concurrency_experiment_host_summary_v2_carried_sha256(value))}
        )


class ConcurrencyExperimentPercentileSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-percentile-summary/v2"] = (
        "eval-factory/concurrency-experiment-percentile-summary/v2"
    )
    sample_count: int = Field(ge=0, le=1_000_000_000)
    minimum: int = Field(ge=0, le=10**18)
    p50: int = Field(ge=0, le=10**18)
    p95: int = Field(ge=0, le=10**18)
    maximum: int = Field(ge=0, le=10**18)
    unit: ConcurrencyExperimentUnitV2

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.sample_count == 0 and any((self.minimum, self.p50, self.p95, self.maximum)):
            raise ValueError("empty percentile summary values must be zero")
        if not self.minimum <= self.p50 <= self.p95 <= self.maximum:
            raise ValueError("percentile summary values are not monotonic")
        return self

    @classmethod
    def create(
        cls,
        *,
        samples: tuple[int, ...],
        unit: ConcurrencyExperimentUnitV2,
    ) -> ConcurrencyExperimentPercentileSummaryV2:
        if not samples:
            return cls(
                sample_count=0,
                minimum=0,
                p50=0,
                p95=0,
                maximum=0,
                unit=unit,
            )
        if any(value < 0 for value in samples):
            raise ValueError("percentile samples must be non-negative")
        ordered = tuple(sorted(samples))
        return cls(
            sample_count=len(ordered),
            minimum=ordered[0],
            p50=_nearest_rank(ordered, 50),
            p95=_nearest_rank(ordered, 95),
            maximum=ordered[-1],
            unit=unit,
        )


class ConcurrencyExperimentLevelSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-level-summary/v2"] = (
        "eval-factory/concurrency-experiment-level-summary/v2"
    )
    worker_count: Literal[1, 2, 4, 8, 16]
    trial_count: Literal[3]
    unique_case_count: Literal[24]
    executed_case_count: Literal[72]
    succeeded_case_count: int = Field(ge=0, le=72)
    product_error_count: int = Field(ge=0, le=72)
    infrastructure_error_count: int = Field(ge=0, le=72)
    incomplete_case_count: int = Field(ge=0, le=72)
    error_rate_basis_points: int = Field(ge=0, le=10_000)
    attempt_count: int = Field(ge=0, le=100_000)
    attempt_metric_count: int = Field(ge=0, le=100_000)
    stage_run_count: int = Field(ge=0, le=100_000)
    stage_result_count: int = Field(ge=0, le=100_000)
    provider_invocation_count: int = Field(ge=0, le=100_000)
    retry_count: int = Field(ge=0, le=100_000)
    resume_count: int = Field(ge=0, le=100_000)
    model_request_count: Literal[0] = 0
    input_token_count: Literal[0] = 0
    output_token_count: Literal[0] = 0
    model_availability: Literal[ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE] = (
        ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE
    )
    cache_availability: Literal[ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE] = (
        ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE
    )
    cost_availability: Literal[ConcurrencyExperimentMetricAvailabilityV2.UNAVAILABLE] = (
        ConcurrencyExperimentMetricAvailabilityV2.UNAVAILABLE
    )
    process_start_count: Literal[0] = 0
    renderer_operation_count: Literal[0] = 0
    network_request_count: Literal[0] = 0
    retained_storage_bytes: int = Field(ge=0, le=10**15)
    tool_call_count: int = Field(ge=0, le=10**12)
    unavailable_cost_attempt_count: int = Field(ge=0, le=100_000)
    elapsed_nanoseconds: ConcurrencyExperimentPercentileSummaryV2
    throughput_millijobs_per_second: ConcurrencyExperimentPercentileSummaryV2
    case_dispatch_wait_nanoseconds: ConcurrencyExperimentPercentileSummaryV2
    case_wall_latency_nanoseconds: ConcurrencyExperimentPercentileSummaryV2
    queue_wait_microseconds: ConcurrencyExperimentPercentileSummaryV2
    execution_microseconds: ConcurrencyExperimentPercentileSummaryV2
    attempt_active_total_microseconds: ConcurrencyExperimentPercentileSummaryV2
    outcome: ConcurrencyExperimentLevelOutcomeV2
    reason_codes: tuple[ConcurrencyExperimentReasonCodeV2, ...]

    @model_validator(mode="after")
    def validate_level(self) -> Self:
        if self.worker_count not in CONCURRENCY_EXPERIMENT_LEVELS:
            raise ValueError("worker count is not in the experiment staircase")
        classified = (
            self.succeeded_case_count
            + self.product_error_count
            + self.infrastructure_error_count
            + self.incomplete_case_count
        )
        if classified != self.executed_case_count:
            raise ValueError("level case outcomes do not close over executions")
        expected_rate = (
            (self.product_error_count + self.infrastructure_error_count + self.incomplete_case_count)
            * 10_000
            // self.executed_case_count
        )
        if self.error_rate_basis_points != expected_rate:
            raise ValueError("level error rate differs from case outcomes")
        expected_units = (
            (self.elapsed_nanoseconds, CONCURRENCY_EXPERIMENT_TRIALS),
            (
                self.throughput_millijobs_per_second,
                CONCURRENCY_EXPERIMENT_TRIALS,
            ),
            (
                self.case_dispatch_wait_nanoseconds,
                CONCURRENCY_EXPERIMENT_EXECUTIONS_PER_LEVEL,
            ),
            (
                self.case_wall_latency_nanoseconds,
                CONCURRENCY_EXPERIMENT_EXECUTIONS_PER_LEVEL,
            ),
            (self.queue_wait_microseconds, self.attempt_metric_count),
            (self.execution_microseconds, self.attempt_metric_count),
            (
                self.attempt_active_total_microseconds,
                self.attempt_metric_count,
            ),
        )
        for summary, count in expected_units:
            if summary.sample_count != count:
                raise ValueError("level metric sample count is incomplete")
        if self.attempt_metric_count > self.attempt_count:
            raise ValueError("attempt metrics exceed durable attempts")
        if self.unavailable_cost_attempt_count != self.attempt_metric_count:
            raise ValueError("every observed attempt metric requires unavailable cost")
        if self.retry_count > self.attempt_count or self.resume_count > self.attempt_count:
            raise ValueError("level retry or resume count exceeds attempts")
        if self.outcome is ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE:
            if (
                self.reason_codes != (ConcurrencyExperimentReasonCodeV2.NONE,)
                or self.succeeded_case_count != self.executed_case_count
                or self.attempt_count != self.executed_case_count * 10
                or self.attempt_metric_count != self.attempt_count
                or self.stage_run_count != self.executed_case_count * 9
                or self.stage_result_count != self.executed_case_count * 9
                or self.provider_invocation_count != self.executed_case_count
                or self.retry_count
                or self.resume_count
            ):
                raise ValueError("eligible level evidence is incomplete")
        elif not self.reason_codes or ConcurrencyExperimentReasonCodeV2.NONE in self.reason_codes:
            raise ValueError("ineligible level requires closed non-success reasons")
        return self

    @classmethod
    def create(
        cls,
        *,
        worker_count: int,
        trial_count: int,
        unique_case_count: int,
        executed_case_count: int,
        succeeded_case_count: int,
        product_error_count: int,
        infrastructure_error_count: int,
        incomplete_case_count: int,
        attempt_count: int,
        attempt_metric_count: int,
        stage_run_count: int,
        stage_result_count: int,
        provider_invocation_count: int,
        retry_count: int,
        resume_count: int,
        model_request_count: int,
        input_token_count: int,
        output_token_count: int,
        process_start_count: int,
        renderer_operation_count: int,
        network_request_count: int,
        retained_storage_bytes: int,
        tool_call_count: int,
        unavailable_cost_attempt_count: int,
        elapsed_nanoseconds: ConcurrencyExperimentPercentileSummaryV2,
        throughput_millijobs_per_second: ConcurrencyExperimentPercentileSummaryV2,
        case_dispatch_wait_nanoseconds: ConcurrencyExperimentPercentileSummaryV2,
        case_wall_latency_nanoseconds: ConcurrencyExperimentPercentileSummaryV2,
        queue_wait_microseconds: ConcurrencyExperimentPercentileSummaryV2,
        execution_microseconds: ConcurrencyExperimentPercentileSummaryV2,
        attempt_active_total_microseconds: ConcurrencyExperimentPercentileSummaryV2,
        outcome: ConcurrencyExperimentLevelOutcomeV2,
        reason_codes: tuple[ConcurrencyExperimentReasonCodeV2, ...],
    ) -> ConcurrencyExperimentLevelSummaryV2:
        errors = product_error_count + infrastructure_error_count + incomplete_case_count
        return cls(
            worker_count=cast(Literal[1, 2, 4, 8, 16], worker_count),
            trial_count=cast(Literal[3], trial_count),
            unique_case_count=cast(Literal[24], unique_case_count),
            executed_case_count=cast(Literal[72], executed_case_count),
            succeeded_case_count=succeeded_case_count,
            product_error_count=product_error_count,
            infrastructure_error_count=infrastructure_error_count,
            incomplete_case_count=incomplete_case_count,
            error_rate_basis_points=(errors * 10_000 // executed_case_count if executed_case_count else 0),
            attempt_count=attempt_count,
            attempt_metric_count=attempt_metric_count,
            stage_run_count=stage_run_count,
            stage_result_count=stage_result_count,
            provider_invocation_count=provider_invocation_count,
            retry_count=retry_count,
            resume_count=resume_count,
            model_request_count=cast(Literal[0], model_request_count),
            input_token_count=cast(Literal[0], input_token_count),
            output_token_count=cast(Literal[0], output_token_count),
            process_start_count=cast(Literal[0], process_start_count),
            renderer_operation_count=cast(Literal[0], renderer_operation_count),
            network_request_count=cast(Literal[0], network_request_count),
            retained_storage_bytes=retained_storage_bytes,
            tool_call_count=tool_call_count,
            unavailable_cost_attempt_count=unavailable_cost_attempt_count,
            elapsed_nanoseconds=elapsed_nanoseconds,
            throughput_millijobs_per_second=throughput_millijobs_per_second,
            case_dispatch_wait_nanoseconds=case_dispatch_wait_nanoseconds,
            case_wall_latency_nanoseconds=case_wall_latency_nanoseconds,
            queue_wait_microseconds=queue_wait_microseconds,
            execution_microseconds=execution_microseconds,
            attempt_active_total_microseconds=attempt_active_total_microseconds,
            outcome=outcome,
            reason_codes=reason_codes,
        )


class ConcurrencyExperimentFaultCountV2(ContractModelV2):
    fault_point: ConcurrencyExperimentFaultPointV2
    count: Literal[6] = 6


class ConcurrencyExperimentRecoverySummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-recovery-summary/v2"] = (
        "eval-factory/concurrency-experiment-recovery-summary/v2"
    )
    selected_worker_count: Literal[1, 2, 4, 8, 16]
    probe_count: Literal[24] = CONCURRENCY_EXPERIMENT_RECOVERY_PROBES
    fault_counts: tuple[ConcurrencyExperimentFaultCountV2, ...] = Field(
        min_length=4,
        max_length=4,
    )
    fault_observed_count: int = Field(ge=0, le=24)
    resume_success_count: int = Field(ge=0, le=24)
    resume_failure_count: int = Field(ge=0, le=24)
    unexpected_retry_count: int = Field(ge=0, le=100)
    duplicate_provider_fact_count: int = Field(ge=0, le=100_000)
    provider_invocation_count: int = Field(ge=0, le=100_000)
    typed_side_effect_count_before_replay: int = Field(ge=0, le=10**12)
    typed_side_effect_count_after_replay: int = Field(ge=0, le=10**12)
    physical_file_count_before_replay: int = Field(ge=0, le=10**9)
    physical_file_count_after_replay: int = Field(ge=0, le=10**9)
    physical_bytes_before_replay: int = Field(default=0, ge=0, le=10**18)
    physical_bytes_after_replay: int = Field(default=0, ge=0, le=10**18)
    exact_replay_stable: bool
    recovery_stable: bool

    @model_validator(mode="after")
    def validate_recovery(self) -> Self:
        if self.selected_worker_count not in CONCURRENCY_EXPERIMENT_LEVELS:
            raise ValueError("recovery worker count is not in the staircase")
        expected_points = tuple(ConcurrencyExperimentFaultPointV2)
        if tuple(row.fault_point for row in self.fault_counts) != expected_points:
            raise ValueError("recovery fault rows are not canonical")
        if any(row.count != 6 for row in self.fault_counts):
            raise ValueError("recovery requires six probes per fault point")
        if self.resume_success_count + self.resume_failure_count != self.probe_count:
            raise ValueError("recovery resume counts do not close over probes")
        expected_stable = (
            self.fault_observed_count == self.probe_count
            and self.resume_success_count == self.probe_count
            and self.resume_failure_count == 0
            and self.unexpected_retry_count == 0
            and self.duplicate_provider_fact_count == 0
            and self.provider_invocation_count == self.probe_count
            and self.typed_side_effect_count_before_replay == self.typed_side_effect_count_after_replay
            and self.physical_file_count_before_replay == self.physical_file_count_after_replay
            and self.exact_replay_stable
        )
        if self.recovery_stable is not expected_stable:
            raise ValueError("recovery stability differs from nested evidence")
        return self

    @classmethod
    def create(
        cls,
        *,
        selected_worker_count: int,
        fault_counts: dict[ConcurrencyExperimentFaultPointV2, int],
        fault_observed_count: int,
        resume_success_count: int,
        resume_failure_count: int,
        unexpected_retry_count: int,
        duplicate_provider_fact_count: int,
        provider_invocation_count: int,
        typed_side_effect_count_before_replay: int,
        typed_side_effect_count_after_replay: int,
        physical_file_count_before_replay: int,
        physical_file_count_after_replay: int,
        exact_replay_stable: bool,
        physical_bytes_before_replay: int = 0,
        physical_bytes_after_replay: int = 0,
    ) -> ConcurrencyExperimentRecoverySummaryV2:
        rows = tuple(
            ConcurrencyExperimentFaultCountV2(
                fault_point=point,
                count=cast(Literal[6], fault_counts.get(point, 0)),
            )
            for point in ConcurrencyExperimentFaultPointV2
        )
        stable = (
            fault_observed_count == CONCURRENCY_EXPERIMENT_RECOVERY_PROBES
            and resume_success_count == CONCURRENCY_EXPERIMENT_RECOVERY_PROBES
            and resume_failure_count == 0
            and unexpected_retry_count == 0
            and duplicate_provider_fact_count == 0
            and provider_invocation_count == CONCURRENCY_EXPERIMENT_RECOVERY_PROBES
            and typed_side_effect_count_before_replay == typed_side_effect_count_after_replay
            and physical_file_count_before_replay == physical_file_count_after_replay
            and exact_replay_stable
        )
        return cls(
            selected_worker_count=cast(
                Literal[1, 2, 4, 8, 16],
                selected_worker_count,
            ),
            fault_counts=rows,
            fault_observed_count=fault_observed_count,
            resume_success_count=resume_success_count,
            resume_failure_count=resume_failure_count,
            unexpected_retry_count=unexpected_retry_count,
            duplicate_provider_fact_count=duplicate_provider_fact_count,
            provider_invocation_count=provider_invocation_count,
            typed_side_effect_count_before_replay=(typed_side_effect_count_before_replay),
            typed_side_effect_count_after_replay=(typed_side_effect_count_after_replay),
            physical_file_count_before_replay=physical_file_count_before_replay,
            physical_file_count_after_replay=physical_file_count_after_replay,
            physical_bytes_before_replay=physical_bytes_before_replay,
            physical_bytes_after_replay=physical_bytes_after_replay,
            exact_replay_stable=exact_replay_stable,
            recovery_stable=stable,
        )


class NonModelResourceRecommendationV2(ContractModelV2):
    schema_version: Literal["eval-factory/non-model-resource-recommendation/v2"] = (
        "eval-factory/non-model-resource-recommendation/v2"
    )
    recommendation_id: Identifier
    policy_ref: ObjectRef
    recommended_worker_count: Literal[1, 2, 4, 8, 16]
    recommended_storage_bytes_per_job: int = Field(ge=1, le=10**15)
    recommended_storage_bytes_per_cohort: int = Field(ge=1, le=10**15)
    process_capacity: None = None
    renderer_capacity: None = None
    network_capacity: None = None
    model_concurrency: None = None
    model_request_budget: None = None
    model_token_budget: None = None
    selected_median_throughput_millijobs_per_second: int = Field(ge=1)
    maximum_median_throughput_millijobs_per_second: int = Field(ge=1)
    throughput_efficiency_basis_points: int = Field(ge=0, le=10_000)
    selected_p95_case_latency_nanoseconds: int = Field(ge=0)
    baseline_p95_case_latency_nanoseconds: int = Field(ge=1)
    p95_baseline_ratio_basis_points: int = Field(ge=0)
    throughput_threshold_basis_points: Literal[9000] = 9_000
    p95_limit_basis_points: Literal[15000] = 15_000
    reason: Literal[ConcurrencyExperimentRecommendationReasonV2.BALANCED_KNEE] = (
        ConcurrencyExperimentRecommendationReasonV2.BALANCED_KNEE
    )
    pending_dimensions: tuple[
        Literal["MODEL", "PROCESS", "RENDERER", "NETWORK"],
        ...,
    ] = ("MODEL", "PROCESS", "RENDERER", "NETWORK")
    claim_scope: Literal["CANARY_WORKER_STORAGE_ONLY"] = CONCURRENCY_EXPERIMENT_CLAIM_SCOPE
    policy_version: Literal["concurrency-experiment/r8-06-v1"] = CONCURRENCY_EXPERIMENT_POLICY_VERSION
    recommendation_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_recommendation(self) -> Self:
        _require_ref(
            self.policy_ref,
            "concurrency-experiment-policy",
            "v2",
            "policy_ref",
        )
        if self.recommended_storage_bytes_per_cohort < (
            self.recommended_storage_bytes_per_job * CONCURRENCY_EXPERIMENT_CASES
        ):
            raise ValueError("cohort storage recommendation is too small")
        expected_efficiency = (
            self.selected_median_throughput_millijobs_per_second
            * 10_000
            // self.maximum_median_throughput_millijobs_per_second
        )
        expected_ratio = (
            self.selected_p95_case_latency_nanoseconds * 10_000 // self.baseline_p95_case_latency_nanoseconds
        )
        if (
            self.throughput_efficiency_basis_points != expected_efficiency
            or self.p95_baseline_ratio_basis_points != expected_ratio
            or expected_efficiency < self.throughput_threshold_basis_points
            or expected_ratio > self.p95_limit_basis_points
        ):
            raise ValueError("recommendation does not satisfy balanced thresholds")
        if self.pending_dimensions != ("MODEL", "PROCESS", "RENDERER", "NETWORK"):
            raise ValueError("recommendation pending dimensions are not canonical")
        _require_audit(self.audit, (self.policy_ref,), "resource recommendation")
        _validate_identity(
            self.recommendation_id,
            self.recommendation_sha256,
            "non-model-resource-recommendation",
            non_model_resource_recommendation_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        selected_level: ConcurrencyExperimentLevelSummaryV2,
        level_summaries: tuple[ConcurrencyExperimentLevelSummaryV2, ...],
        recovery_summary: ConcurrencyExperimentRecoverySummaryV2,
        recommended_storage_bytes_per_job: int,
        recommended_storage_bytes_per_cohort: int,
        policy: ConcurrencyExperimentPolicyV2,
        audit: ContractAudit,
    ) -> NonModelResourceRecommendationV2:
        selected = select_balanced_worker_level(level_summaries, policy=policy)
        if selected != selected_level:
            raise ValueError("selected level differs from balanced recommendation")
        if (
            recovery_summary.selected_worker_count != selected_level.worker_count
            or not recovery_summary.recovery_stable
        ):
            raise ValueError("recommendation requires exact recovery closure")
        eligible = tuple(
            level
            for level in level_summaries
            if level.outcome is ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE
        )
        baseline = next(
            (level for level in eligible if level.worker_count == 1),
            None,
        )
        if baseline is None:
            raise ValueError("recommendation requires eligible single-worker baseline")
        maximum = max(level.throughput_millijobs_per_second.p50 for level in eligible)
        selected_throughput = selected_level.throughput_millijobs_per_second.p50
        selected_latency = selected_level.case_wall_latency_nanoseconds.p95
        baseline_latency = baseline.case_wall_latency_nanoseconds.p95
        value = cls(
            recommendation_id="non-model-resource-recommendation://pending",
            policy_ref=policy.to_ref(),
            recommended_worker_count=selected_level.worker_count,
            recommended_storage_bytes_per_job=recommended_storage_bytes_per_job,
            recommended_storage_bytes_per_cohort=(recommended_storage_bytes_per_cohort),
            selected_median_throughput_millijobs_per_second=selected_throughput,
            maximum_median_throughput_millijobs_per_second=maximum,
            throughput_efficiency_basis_points=(selected_throughput * 10_000 // maximum),
            selected_p95_case_latency_nanoseconds=selected_latency,
            baseline_p95_case_latency_nanoseconds=baseline_latency,
            p95_baseline_ratio_basis_points=(selected_latency * 10_000 // baseline_latency),
            recommendation_sha256="0" * 64,
            audit=_safe_audit(audit, (policy.to_ref(),)),
        )
        return _finalize(
            value,
            "recommendation_id",
            "recommendation_sha256",
            "non-model-resource-recommendation",
            non_model_resource_recommendation_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return non_model_resource_recommendation_v2_ref(self)


class ConcurrencyExperimentReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-report/v2"] = (
        "eval-factory/concurrency-experiment-report/v2"
    )
    report_id: Identifier
    policy_ref: ObjectRef
    host_summary: ConcurrencyExperimentHostSummaryV2
    cohort_manifest_ref: ObjectRef
    template_ref: ObjectRef
    private_workload_ref: ObjectRef
    private_result_set_ref: ObjectRef
    level_summaries: tuple[ConcurrencyExperimentLevelSummaryV2, ...] = Field(
        min_length=5,
        max_length=5,
    )
    recovery_summary: ConcurrencyExperimentRecoverySummaryV2 | None = None
    recommendation: NonModelResourceRecommendationV2 | None = None
    model_availability: Literal[ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE] = (
        ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE
    )
    cache_availability: Literal[ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE] = (
        ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE
    )
    cost_availability: Literal[ConcurrencyExperimentMetricAvailabilityV2.UNAVAILABLE] = (
        ConcurrencyExperimentMetricAvailabilityV2.UNAVAILABLE
    )
    outcome: ConcurrencyExperimentOutcomeV2
    reason_codes: tuple[ConcurrencyExperimentRecommendationReasonV2, ...]
    claim_scope: Literal["CANARY_WORKER_STORAGE_ONLY"] = CONCURRENCY_EXPERIMENT_CLAIM_SCOPE
    satisfies_sc_012: Literal[False] = False
    satisfies_model_resource_policy: Literal[False] = False
    satisfies_operations_slo: Literal[False] = False
    authorizes_safety_or_privacy_approval: Literal[False] = False
    authorizes_attestation: Literal[False] = False
    authorizes_production_release: Literal[False] = False
    policy_version: Literal["concurrency-experiment/r8-06-v1"] = CONCURRENCY_EXPERIMENT_POLICY_VERSION
    report_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.policy_ref,
            "concurrency-experiment-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.cohort_manifest_ref,
            "development-canary-manifest",
            "v4",
            "cohort_manifest_ref",
        )
        _require_ref(
            self.template_ref,
            "canary-regression-template",
            "private-v1",
            "template_ref",
        )
        _require_ref(
            self.private_workload_ref,
            "concurrency-experiment-workload",
            "private-v1",
            "private_workload_ref",
        )
        _require_ref(
            self.private_result_set_ref,
            "concurrency-experiment-result-set",
            "private-v1",
            "private_result_set_ref",
        )
        if tuple(level.worker_count for level in self.level_summaries) != (CONCURRENCY_EXPERIMENT_LEVELS):
            raise ValueError("report levels are not canonical")
        if self.host_summary.host_summary_sha256 != concurrency_experiment_host_summary_v2_carried_sha256(
            self.host_summary
        ):
            raise ValueError("report host summary identity is pending or stale")
        expected_outcome, expected_reasons = _report_outcome(
            self.level_summaries,
            self.recovery_summary,
            self.recommendation,
        )
        if self.outcome is not expected_outcome or self.reason_codes != expected_reasons:
            raise ValueError("report outcome differs from nested evidence")
        if self.recommendation is not None:
            validate_non_model_resource_recommendation_v2_identity(self.recommendation)
            if self.recommendation.policy_ref != self.policy_ref:
                raise ValueError("report recommendation uses another policy")
            selected = _select_balanced_worker_level(
                self.level_summaries,
                throughput_efficiency_basis_points=(self.recommendation.throughput_threshold_basis_points),
                p95_baseline_limit_basis_points=self.recommendation.p95_limit_basis_points,
            )
            if selected is None or selected.worker_count != self.recommendation.recommended_worker_count:
                raise ValueError("report recommendation differs from balanced level evidence")
            eligible = tuple(
                level
                for level in self.level_summaries
                if level.outcome is ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE
            )
            baseline = next(level for level in eligible if level.worker_count == 1)
            maximum = max(level.throughput_millijobs_per_second.p50 for level in eligible)
            if (
                self.recommendation.selected_median_throughput_millijobs_per_second
                != selected.throughput_millijobs_per_second.p50
                or self.recommendation.maximum_median_throughput_millijobs_per_second != maximum
                or self.recommendation.selected_p95_case_latency_nanoseconds
                != selected.case_wall_latency_nanoseconds.p95
                or self.recommendation.baseline_p95_case_latency_nanoseconds
                != baseline.case_wall_latency_nanoseconds.p95
            ):
                raise ValueError("report recommendation metrics differ from level evidence")
            if (
                self.recovery_summary is None
                or self.recovery_summary.selected_worker_count != self.recommendation.recommended_worker_count
            ):
                raise ValueError("report recovery uses another worker level")
        refs = (
            self.policy_ref,
            self.cohort_manifest_ref,
            self.template_ref,
            self.private_workload_ref,
            self.private_result_set_ref,
            *((self.recommendation.to_ref(),) if self.recommendation is not None else ()),
        )
        _require_audit(self.audit, refs, "concurrency experiment report")
        _validate_identity(
            self.report_id,
            self.report_sha256,
            "concurrency-experiment-report",
            concurrency_experiment_report_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        host_summary: ConcurrencyExperimentHostSummaryV2,
        cohort_manifest_ref: ObjectRef,
        template_ref: ObjectRef,
        private_workload_ref: ObjectRef,
        private_result_set_ref: ObjectRef,
        level_summaries: tuple[ConcurrencyExperimentLevelSummaryV2, ...],
        recovery_summary: ConcurrencyExperimentRecoverySummaryV2 | None,
        recommendation: NonModelResourceRecommendationV2 | None,
        audit: ContractAudit,
    ) -> ConcurrencyExperimentReportV2:
        ordered = tuple(sorted(level_summaries, key=lambda level: level.worker_count))
        outcome, reasons = _report_outcome(
            ordered,
            recovery_summary,
            recommendation,
        )
        refs = (
            policy_ref,
            cohort_manifest_ref,
            template_ref,
            private_workload_ref,
            private_result_set_ref,
            *((recommendation.to_ref(),) if recommendation is not None else ()),
        )
        value = cls(
            report_id="concurrency-experiment-report://pending",
            policy_ref=policy_ref,
            host_summary=host_summary,
            cohort_manifest_ref=cohort_manifest_ref,
            template_ref=template_ref,
            private_workload_ref=private_workload_ref,
            private_result_set_ref=private_result_set_ref,
            level_summaries=ordered,
            recovery_summary=recovery_summary,
            recommendation=recommendation,
            outcome=outcome,
            reason_codes=reasons,
            report_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "report_id",
            "report_sha256",
            "concurrency-experiment-report",
            concurrency_experiment_report_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return concurrency_experiment_report_v2_ref(self)


def select_balanced_worker_level(
    levels: tuple[ConcurrencyExperimentLevelSummaryV2, ...],
    *,
    policy: ConcurrencyExperimentPolicyV2,
) -> ConcurrencyExperimentLevelSummaryV2 | None:
    return _select_balanced_worker_level(
        levels,
        throughput_efficiency_basis_points=policy.throughput_efficiency_basis_points,
        p95_baseline_limit_basis_points=policy.p95_baseline_limit_basis_points,
    )


def _select_balanced_worker_level(
    levels: tuple[ConcurrencyExperimentLevelSummaryV2, ...],
    *,
    throughput_efficiency_basis_points: int,
    p95_baseline_limit_basis_points: int,
) -> ConcurrencyExperimentLevelSummaryV2 | None:
    ordered = tuple(sorted(levels, key=lambda level: level.worker_count))
    if any(
        level.outcome
        in {
            ConcurrencyExperimentLevelOutcomeV2.INCOMPLETE,
            ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_INTEGRITY,
        }
        for level in ordered
    ):
        return None
    eligible = tuple(
        level for level in ordered if level.outcome is ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE
    )
    baseline = next(
        (level for level in eligible if level.worker_count == 1),
        None,
    )
    if baseline is None or not eligible:
        return None
    maximum = max(level.throughput_millijobs_per_second.p50 for level in eligible)
    baseline_p95 = baseline.case_wall_latency_nanoseconds.p95
    if maximum <= 0 or baseline_p95 <= 0:
        return None
    for level in eligible:
        throughput_bp = level.throughput_millijobs_per_second.p50 * 10_000 // maximum
        latency_bp = level.case_wall_latency_nanoseconds.p95 * 10_000 // baseline_p95
        if (
            throughput_bp >= throughput_efficiency_basis_points
            and latency_bp <= p95_baseline_limit_basis_points
        ):
            return level
    return None


def concurrency_experiment_policy_v2_carried_sha256(
    value: ConcurrencyExperimentPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def concurrency_experiment_host_summary_v2_carried_sha256(
    value: ConcurrencyExperimentHostSummaryV2,
) -> str:
    return _carried(value, {"host_summary_sha256", "audit"})


def non_model_resource_recommendation_v2_carried_sha256(
    value: NonModelResourceRecommendationV2,
) -> str:
    return _carried(
        value,
        {"recommendation_id", "recommendation_sha256", "audit"},
    )


def concurrency_experiment_report_v2_carried_sha256(
    value: ConcurrencyExperimentReportV2,
) -> str:
    return _carried(value, {"report_id", "report_sha256", "audit"})


def concurrency_experiment_policy_v2_ref(
    value: ConcurrencyExperimentPolicyV2,
) -> ObjectRef:
    validate_concurrency_experiment_policy_v2_identity(value)
    return _ref(
        "concurrency-experiment-policy",
        value.policy_id,
        value.policy_sha256,
    )


def non_model_resource_recommendation_v2_ref(
    value: NonModelResourceRecommendationV2,
) -> ObjectRef:
    validate_non_model_resource_recommendation_v2_identity(value)
    return _ref(
        "non-model-resource-recommendation",
        value.recommendation_id,
        value.recommendation_sha256,
    )


def concurrency_experiment_report_v2_ref(
    value: ConcurrencyExperimentReportV2,
) -> ObjectRef:
    validate_concurrency_experiment_report_v2_identity(value)
    return _ref(
        "concurrency-experiment-report",
        value.report_id,
        value.report_sha256,
    )


def validate_concurrency_experiment_policy_v2_identity(
    value: ConcurrencyExperimentPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "concurrency-experiment-policy",
        concurrency_experiment_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_non_model_resource_recommendation_v2_identity(
    value: NonModelResourceRecommendationV2,
) -> None:
    _validate_identity(
        value.recommendation_id,
        value.recommendation_sha256,
        "non-model-resource-recommendation",
        non_model_resource_recommendation_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_concurrency_experiment_report_v2_identity(
    value: ConcurrencyExperimentReportV2,
) -> None:
    _validate_identity(
        value.report_id,
        value.report_sha256,
        "concurrency-experiment-report",
        concurrency_experiment_report_v2_carried_sha256(value),
        allow_pending=False,
    )


def _report_outcome(
    levels: tuple[ConcurrencyExperimentLevelSummaryV2, ...],
    recovery: ConcurrencyExperimentRecoverySummaryV2 | None,
    recommendation: NonModelResourceRecommendationV2 | None,
) -> tuple[
    ConcurrencyExperimentOutcomeV2,
    tuple[ConcurrencyExperimentRecommendationReasonV2, ...],
]:
    if any(
        level.outcome
        in {
            ConcurrencyExperimentLevelOutcomeV2.INCOMPLETE,
            ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_INTEGRITY,
        }
        for level in levels
    ):
        return (
            ConcurrencyExperimentOutcomeV2.INCOMPLETE,
            (ConcurrencyExperimentRecommendationReasonV2.NO_ELIGIBLE_LEVEL,),
        )
    if recommendation is not None:
        if recovery is None or not recovery.recovery_stable:
            raise ValueError("recommendation requires stable recovery evidence")
        return (
            ConcurrencyExperimentOutcomeV2.NON_MODEL_RECOMMENDED,
            (
                ConcurrencyExperimentRecommendationReasonV2.BALANCED_KNEE,
                ConcurrencyExperimentRecommendationReasonV2.MODEL_EVIDENCE_PENDING,
            ),
        )
    if recovery is not None and not recovery.recovery_stable:
        return (
            ConcurrencyExperimentOutcomeV2.RECOMMENDATION_PENDING,
            (
                ConcurrencyExperimentRecommendationReasonV2.RECOVERY_VALIDATION_FAILED,
                ConcurrencyExperimentRecommendationReasonV2.MODEL_EVIDENCE_PENDING,
            ),
        )
    return (
        ConcurrencyExperimentOutcomeV2.RECOMMENDATION_PENDING,
        _pending_recommendation_reasons(levels),
    )


def _pending_recommendation_reasons(
    levels: tuple[ConcurrencyExperimentLevelSummaryV2, ...],
) -> tuple[ConcurrencyExperimentRecommendationReasonV2, ...]:
    eligible = tuple(
        level for level in levels if level.outcome is ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE
    )
    baseline = next(
        (level for level in eligible if level.worker_count == 1),
        None,
    )
    if not eligible or baseline is None:
        return (
            ConcurrencyExperimentRecommendationReasonV2.NO_ELIGIBLE_LEVEL,
            ConcurrencyExperimentRecommendationReasonV2.MODEL_EVIDENCE_PENDING,
        )
    maximum = max(level.throughput_millijobs_per_second.p50 for level in eligible)
    baseline_p95 = baseline.case_wall_latency_nanoseconds.p95
    if maximum <= 0 or baseline_p95 <= 0:
        return (
            ConcurrencyExperimentRecommendationReasonV2.NO_ELIGIBLE_LEVEL,
            ConcurrencyExperimentRecommendationReasonV2.MODEL_EVIDENCE_PENDING,
        )
    throughput_qualified = {
        level.worker_count
        for level in eligible
        if (level.throughput_millijobs_per_second.p50 * 10_000 // maximum >= 9_000)
    }
    latency_qualified = {
        level.worker_count
        for level in eligible
        if (level.case_wall_latency_nanoseconds.p95 * 10_000 // baseline_p95 <= 15_000)
    }
    if throughput_qualified & latency_qualified:
        return (
            ConcurrencyExperimentRecommendationReasonV2.NO_ELIGIBLE_LEVEL,
            ConcurrencyExperimentRecommendationReasonV2.MODEL_EVIDENCE_PENDING,
        )
    return (
        ConcurrencyExperimentRecommendationReasonV2.THROUGHPUT_THRESHOLD_NOT_MET,
        ConcurrencyExperimentRecommendationReasonV2.LATENCY_THRESHOLD_NOT_MET,
        ConcurrencyExperimentRecommendationReasonV2.MODEL_EVIDENCE_PENDING,
    )


def _nearest_rank(samples: tuple[int, ...], percentile: int) -> int:
    rank = max(1, (percentile * len(samples) + 99) // 100)
    return samples[rank - 1]


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": tuple(sorted(set(refs), key=_ref_key))})


def _require_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != tuple(sorted(set(refs), key=_ref_key)):
        raise ValueError(f"{label} audit refs are incomplete")
    versions = tuple(
        binding for binding in audit.governing_versions if binding.component == "concurrency-experiment"
    )
    if len(versions) != 1 or versions[0].version != CONCURRENCY_EXPERIMENT_POLICY_VERSION:
        raise ValueError(f"{label} audit is missing concurrency policy")


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} has the wrong type or version")


def _ref(
    object_type: str,
    object_id: str,
    object_sha256: str,
) -> ObjectRef:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        raise ValueError(f"{object_type} identity is pending")
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v2",
        object_sha256=object_sha256,
    )


def _carried(value: ContractModelV2, exclude: set[str]) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude=exclude,
            exclude_none=False,
        )
    )


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
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


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "CONCURRENCY_EXPERIMENT_CASES",
    "CONCURRENCY_EXPERIMENT_CLAIM_SCOPE",
    "CONCURRENCY_EXPERIMENT_EXECUTIONS_PER_LEVEL",
    "CONCURRENCY_EXPERIMENT_LEVELS",
    "CONCURRENCY_EXPERIMENT_POLICY_VERSION",
    "CONCURRENCY_EXPERIMENT_RECOVERY_PROBES",
    "CONCURRENCY_EXPERIMENT_TRIALS",
    "ConcurrencyExperimentFaultPointV2",
    "ConcurrencyExperimentHostSummaryV2",
    "ConcurrencyExperimentLevelOutcomeV2",
    "ConcurrencyExperimentLevelSummaryV2",
    "ConcurrencyExperimentMetricAvailabilityV2",
    "ConcurrencyExperimentOutcomeV2",
    "ConcurrencyExperimentPercentileSummaryV2",
    "ConcurrencyExperimentPolicyV2",
    "ConcurrencyExperimentReasonCodeV2",
    "ConcurrencyExperimentRecommendationReasonV2",
    "ConcurrencyExperimentRecoverySummaryV2",
    "ConcurrencyExperimentReportV2",
    "ConcurrencyExperimentUnitV2",
    "NonModelResourceRecommendationV2",
    "concurrency_experiment_host_summary_v2_carried_sha256",
    "concurrency_experiment_policy_v2_carried_sha256",
    "concurrency_experiment_policy_v2_ref",
    "concurrency_experiment_report_v2_carried_sha256",
    "concurrency_experiment_report_v2_ref",
    "non_model_resource_recommendation_v2_carried_sha256",
    "non_model_resource_recommendation_v2_ref",
    "select_balanced_worker_level",
    "validate_concurrency_experiment_policy_v2_identity",
    "validate_concurrency_experiment_report_v2_identity",
    "validate_non_model_resource_recommendation_v2_identity",
]
