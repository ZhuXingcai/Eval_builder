from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from typing import Literal, Self, TypedDict, cast

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.observability_v2 import R6_OBSERVABILITY_POLICY_VERSION
from eval_factory.contracts.orchestration_v2 import (
    R6_WORK_CONTROL_POLICY_VERSION,
    R6_WORK_GRAPH_POLICY_VERSION,
    StageNameV2,
)

SCHEDULER_LOAD_POLICY_VERSION: Literal["scheduler-load/r8-05-v1"] = "scheduler-load/r8-05-v1"
SCHEDULER_LOAD_CLAIM_SCOPE: Literal["SCHEDULER_LOAD_ONLY"] = "SCHEDULER_LOAD_ONLY"
SCHEDULER_LOAD_JOB_COUNT: Literal[1000] = 1000
SCHEDULER_LOAD_JOBS_PER_FAULT: Literal[125] = 125


class SchedulerLoadFaultPointV2(StrEnum):
    AFTER_JOB_CREATE = "AFTER_JOB_CREATE"
    AFTER_WORK_GRAPH_CREATE = "AFTER_WORK_GRAPH_CREATE"
    AFTER_JOB_START = "AFTER_JOB_START"
    AFTER_READINESS_RECORD = "AFTER_READINESS_RECORD"
    AFTER_CONTROL_POLICY_BIND = "AFTER_CONTROL_POLICY_BIND"
    AFTER_LEASE_ACQUIRE = "AFTER_LEASE_ACQUIRE"
    AFTER_STAGE_COMPLETION = "AFTER_STAGE_COMPLETION"
    AFTER_OBSERVABILITY_REFRESH = "AFTER_OBSERVABILITY_REFRESH"


class SchedulerLoadPhaseV2(StrEnum):
    SYNTHETIC_INITIAL = "SYNTHETIC_INITIAL"
    CRASH_RECOVERY = "CRASH_RECOVERY"
    EXACT_REPLAY = "EXACT_REPLAY"


class SchedulerLoadCaseOutcomeV2(StrEnum):
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    INCOMPLETE = "INCOMPLETE"


class SchedulerLoadReasonCodeV2(StrEnum):
    NONE = "NONE"
    EXPECTED_FAULT_NOT_OBSERVED = "EXPECTED_FAULT_NOT_OBSERVED"
    RESUME_FAILED = "RESUME_FAILED"
    TERMINAL_EVIDENCE_MISMATCH = "TERMINAL_EVIDENCE_MISMATCH"
    UNEXPECTED_RETRY = "UNEXPECTED_RETRY"
    DUPLICATE_IMMUTABLE_FACT = "DUPLICATE_IMMUTABLE_FACT"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"
    OBSERVABILITY_INCOMPLETE = "OBSERVABILITY_INCOMPLETE"
    MATERIAL_INTEGRITY_ERROR = "MATERIAL_INTEGRITY_ERROR"
    CHILD_STATE_INCOMPLETE = "CHILD_STATE_INCOMPLETE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class SchedulerLoadReportOutcomeV2(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class SchedulerLoadPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-policy/v2"] = "eval-factory/scheduler-load-policy/v2"
    policy_id: Identifier
    workload_seed: Literal["scheduler-load/r8-05-v1"] = "scheduler-load/r8-05-v1"
    required_job_count: Literal[1000] = SCHEDULER_LOAD_JOB_COUNT
    worker_count: Literal[1] = 1
    requested_stages: tuple[StageNameV2, ...] = (StageNameV2.TRACE_INDEX,)
    require_fault_per_job: Literal[True] = True
    require_resume_success: Literal[True] = True
    require_exact_replay: Literal[True] = True
    max_unexpected_retries: Literal[0] = 0
    required_stage_results_per_job: Literal[1] = 1
    required_attempt_metrics_per_job: Literal[1] = 1
    fault_points: tuple[SchedulerLoadFaultPointV2, ...] = tuple(SchedulerLoadFaultPointV2)
    plan_policy_version: Literal["dataset-job-stage-policy/r6-01-v1"] = "dataset-job-stage-policy/r6-01-v1"
    work_graph_policy_version: Literal["dataset-work-graph/r6-02-v1"] = R6_WORK_GRAPH_POLICY_VERSION
    work_control_policy_version: Literal["work-control/r6-05-v1"] = R6_WORK_CONTROL_POLICY_VERSION
    observability_policy_version: Literal["batch-observability/r6-06-v1"] = R6_OBSERVABILITY_POLICY_VERSION
    lease_duration_seconds: int = Field(ge=1, le=86_400)
    heartbeat_extension_seconds: int = Field(ge=1, le=86_400)
    max_attempts: Literal[1] = 1
    retry_delay_seconds: tuple[int, ...] = ()
    retry_lease_expiry: Literal[False] = False
    max_private_bytes: int = Field(ge=2, le=1_000_000_000)
    max_report_bytes: int = Field(ge=2, le=100_000_000)
    policy_version: Literal["scheduler-load/r8-05-v1"] = SCHEDULER_LOAD_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.requested_stages != (StageNameV2.TRACE_INDEX,):
            raise ValueError("scheduler load must request TRACE_INDEX only")
        if self.fault_points != tuple(SchedulerLoadFaultPointV2):
            raise ValueError("scheduler load must bind all fault points in canonical order")
        if self.retry_delay_seconds:
            raise ValueError("single-attempt scheduler load cannot bind retry delays")
        if self.heartbeat_extension_seconds > self.lease_duration_seconds:
            raise ValueError("heartbeat extension cannot exceed lease duration")
        _require_audit(self.audit, (), "scheduler load policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "scheduler-load-policy",
            scheduler_load_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        lease_duration_seconds: int,
        heartbeat_extension_seconds: int,
        max_private_bytes: int,
        max_report_bytes: int,
        audit: ContractAudit,
    ) -> SchedulerLoadPolicyV2:
        value = cls(
            policy_id="scheduler-load-policy://pending",
            lease_duration_seconds=lease_duration_seconds,
            heartbeat_extension_seconds=heartbeat_extension_seconds,
            max_private_bytes=max_private_bytes,
            max_report_bytes=max_report_bytes,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "scheduler-load-policy",
            scheduler_load_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return scheduler_load_policy_v2_ref(self)


class SchedulerLoadPhaseSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-phase-summary/v2"] = (
        "eval-factory/scheduler-load-phase-summary/v2"
    )
    phase: SchedulerLoadPhaseV2
    job_count: Literal[1000] = SCHEDULER_LOAD_JOB_COUNT
    operation_count: int = Field(ge=1, le=1_000_000_000)
    elapsed_nanoseconds: int = Field(ge=1, le=10**18)
    throughput_millijobs_per_second: int = Field(ge=1, le=10**18)

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        expected = self.job_count * 1_000_000_000_000 // self.elapsed_nanoseconds
        if expected < 1 or self.throughput_millijobs_per_second != expected:
            raise ValueError("scheduler load throughput does not match elapsed time")
        return self

    @classmethod
    def create(
        cls,
        *,
        phase: SchedulerLoadPhaseV2,
        job_count: int,
        operation_count: int,
        elapsed_nanoseconds: int,
    ) -> SchedulerLoadPhaseSummaryV2:
        throughput = job_count * 1_000_000_000_000 // elapsed_nanoseconds if elapsed_nanoseconds > 0 else 0
        return cls(
            phase=phase,
            job_count=cast(Literal[1000], job_count),
            operation_count=operation_count,
            elapsed_nanoseconds=elapsed_nanoseconds,
            throughput_millijobs_per_second=throughput,
        )


class SchedulerLoadFaultCountV2(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-fault-count/v2"] = (
        "eval-factory/scheduler-load-fault-count/v2"
    )
    fault_point: SchedulerLoadFaultPointV2
    count: Literal[125] = SCHEDULER_LOAD_JOBS_PER_FAULT


class SchedulerLoadCaseSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-case-summary/v2"] = (
        "eval-factory/scheduler-load-case-summary/v2"
    )
    case_summary_id: Identifier
    private_case_result_ref: ObjectRef
    outcome: SchedulerLoadCaseOutcomeV2
    reason_code: SchedulerLoadReasonCodeV2
    assigned_fault_point: SchedulerLoadFaultPointV2
    fault_observed: bool
    resume_succeeded: bool
    replay_stable: bool
    attempt_count: int = Field(ge=0, le=10)
    unexpected_retry_count: int = Field(ge=0, le=10)
    duplicate_immutable_fact_count: int = Field(ge=0, le=1_000_000)
    stage_run_count: int = Field(ge=0, le=10)
    stage_result_count: int = Field(ge=0, le=10)
    attempt_metrics_count: int = Field(ge=0, le=10)
    completion_witness_count: int = Field(ge=0, le=100)
    observability_complete: bool
    policy_version: Literal["scheduler-load/r8-05-v1"] = SCHEDULER_LOAD_POLICY_VERSION
    case_summary_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        _require_ref(
            self.private_case_result_ref,
            "scheduler-load-case-result",
            "private-v1",
            "private_case_result_ref",
        )
        validate_scheduler_load_case_classification(
            self.outcome,
            self.reason_code,
            replay_stable=self.replay_stable,
        )
        if self.outcome is SchedulerLoadCaseOutcomeV2.RECOVERED and not _recovered_case_fields(self):
            raise ValueError("recovered scheduler load case evidence is incomplete")
        if self.unexpected_retry_count > max(0, self.attempt_count - 1):
            raise ValueError("unexpected retries exceed scheduler attempts")
        _require_audit(
            self.audit,
            (self.private_case_result_ref,),
            "scheduler load case summary",
        )
        _validate_identity(
            self.case_summary_id,
            self.case_summary_sha256,
            "scheduler-load-case-summary",
            scheduler_load_case_summary_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        private_case_result_ref: ObjectRef,
        outcome: SchedulerLoadCaseOutcomeV2,
        reason_code: SchedulerLoadReasonCodeV2,
        assigned_fault_point: SchedulerLoadFaultPointV2,
        fault_observed: bool,
        resume_succeeded: bool,
        replay_stable: bool,
        attempt_count: int,
        unexpected_retry_count: int,
        duplicate_immutable_fact_count: int,
        stage_run_count: int,
        stage_result_count: int,
        attempt_metrics_count: int,
        completion_witness_count: int,
        observability_complete: bool,
        audit: ContractAudit,
    ) -> SchedulerLoadCaseSummaryV2:
        value = cls(
            case_summary_id="scheduler-load-case-summary://pending",
            private_case_result_ref=private_case_result_ref,
            outcome=outcome,
            reason_code=reason_code,
            assigned_fault_point=assigned_fault_point,
            fault_observed=fault_observed,
            resume_succeeded=resume_succeeded,
            replay_stable=replay_stable,
            attempt_count=attempt_count,
            unexpected_retry_count=unexpected_retry_count,
            duplicate_immutable_fact_count=duplicate_immutable_fact_count,
            stage_run_count=stage_run_count,
            stage_result_count=stage_result_count,
            attempt_metrics_count=attempt_metrics_count,
            completion_witness_count=completion_witness_count,
            observability_complete=observability_complete,
            case_summary_sha256="0" * 64,
            audit=_safe_audit(audit, (private_case_result_ref,)),
        )
        return _finalize(
            value,
            "case_summary_id",
            "case_summary_sha256",
            "scheduler-load-case-summary",
            scheduler_load_case_summary_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return scheduler_load_case_summary_v2_ref(self)


class _DerivedReportFields(TypedDict):
    case_summary_refs: tuple[ObjectRef, ...]
    recovered_case_refs: tuple[ObjectRef, ...]
    failed_case_refs: tuple[ObjectRef, ...]
    infrastructure_error_case_refs: tuple[ObjectRef, ...]
    incomplete_case_refs: tuple[ObjectRef, ...]
    fault_counts: tuple[SchedulerLoadFaultCountV2, ...]
    unique_job_count: Literal[1000]
    total_faults_assigned: int
    total_faults_observed: int
    total_resumes: int
    total_replays: int
    total_attempts: int
    total_unexpected_retries: int
    total_duplicate_immutable_facts: int
    total_stage_runs: int
    total_stage_results: int
    total_attempt_metrics: int
    total_completion_witnesses: int
    typed_replay_stable: bool
    physical_replay_stable: bool
    outcome: SchedulerLoadReportOutcomeV2


class SchedulerLoadReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-report/v2"] = "eval-factory/scheduler-load-report/v2"
    report_id: Identifier
    policy_ref: ObjectRef
    private_workload_ref: ObjectRef
    private_result_set_ref: ObjectRef
    case_summaries: tuple[SchedulerLoadCaseSummaryV2, ...] = Field(
        min_length=1000,
        max_length=1000,
    )
    case_summary_refs: tuple[ObjectRef, ...] = Field(
        min_length=1000,
        max_length=1000,
    )
    recovered_case_refs: tuple[ObjectRef, ...] = ()
    failed_case_refs: tuple[ObjectRef, ...] = ()
    infrastructure_error_case_refs: tuple[ObjectRef, ...] = ()
    incomplete_case_refs: tuple[ObjectRef, ...] = ()
    fault_counts: tuple[SchedulerLoadFaultCountV2, ...] = Field(
        min_length=8,
        max_length=8,
    )
    phase_summaries: tuple[SchedulerLoadPhaseSummaryV2, ...] = Field(
        min_length=3,
        max_length=3,
    )
    unique_job_count: Literal[1000] = SCHEDULER_LOAD_JOB_COUNT
    total_faults_assigned: int = Field(ge=0, le=1000)
    total_faults_observed: int = Field(ge=0, le=1000)
    total_resumes: int = Field(ge=0, le=1000)
    total_replays: int = Field(ge=0, le=1000)
    total_attempts: int = Field(ge=0, le=10_000)
    total_unexpected_retries: int = Field(ge=0, le=10_000)
    total_duplicate_immutable_facts: int = Field(ge=0, le=1_000_000)
    total_stage_runs: int = Field(ge=0, le=10_000)
    total_stage_results: int = Field(ge=0, le=10_000)
    total_attempt_metrics: int = Field(ge=0, le=10_000)
    total_completion_witnesses: int = Field(ge=0, le=100_000)
    typed_side_effect_count_before_replay: int = Field(ge=0, le=1_000_000_000)
    typed_side_effect_count_after_replay: int = Field(ge=0, le=1_000_000_000)
    physical_file_count_before_replay: int = Field(ge=0, le=1_000_000)
    physical_file_count_after_replay: int = Field(ge=0, le=1_000_000)
    physical_bytes_before_replay: int = Field(ge=0, le=10**15)
    physical_bytes_after_replay: int = Field(ge=0, le=10**15)
    typed_replay_stable: bool
    physical_replay_stable: bool
    claim_scope: Literal["SCHEDULER_LOAD_ONLY"] = SCHEDULER_LOAD_CLAIM_SCOPE
    satisfies_real_trace_stability: Literal[False] = False
    satisfies_label_quality: Literal[False] = False
    satisfies_concurrency_policy: Literal[False] = False
    satisfies_operations_slo: Literal[False] = False
    authorizes_attestation: Literal[False] = False
    authorizes_production_release: Literal[False] = False
    outcome: SchedulerLoadReportOutcomeV2
    policy_version: Literal["scheduler-load/r8-05-v1"] = SCHEDULER_LOAD_POLICY_VERSION
    report_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.policy_ref,
            "scheduler-load-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.private_workload_ref,
            "scheduler-load-workload",
            "private-v1",
            "private_workload_ref",
        )
        _require_ref(
            self.private_result_set_ref,
            "scheduler-load-result-set",
            "private-v1",
            "private_result_set_ref",
        )
        fields = _derive_report_fields(
            self.case_summaries,
            self.phase_summaries,
            typed_before=self.typed_side_effect_count_before_replay,
            typed_after=self.typed_side_effect_count_after_replay,
            physical_files_before=self.physical_file_count_before_replay,
            physical_files_after=self.physical_file_count_after_replay,
            physical_bytes_before=self.physical_bytes_before_replay,
            physical_bytes_after=self.physical_bytes_after_replay,
        )
        for field_name, expected in fields.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"scheduler load report {field_name} differs from nested evidence")
        refs = (
            self.policy_ref,
            self.private_workload_ref,
            self.private_result_set_ref,
            *self.case_summary_refs,
        )
        _require_audit(self.audit, refs, "scheduler load report")
        _validate_identity(
            self.report_id,
            self.report_sha256,
            "scheduler-load-report",
            scheduler_load_report_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        private_workload_ref: ObjectRef,
        private_result_set_ref: ObjectRef,
        case_summaries: tuple[SchedulerLoadCaseSummaryV2, ...],
        phase_summaries: tuple[SchedulerLoadPhaseSummaryV2, ...],
        typed_side_effect_count_before_replay: int,
        typed_side_effect_count_after_replay: int,
        physical_file_count_before_replay: int,
        physical_file_count_after_replay: int,
        physical_bytes_before_replay: int,
        physical_bytes_after_replay: int,
        audit: ContractAudit,
    ) -> SchedulerLoadReportV2:
        ordered = tuple(sorted(case_summaries, key=lambda item: item.case_summary_id))
        fields = _derive_report_fields(
            ordered,
            phase_summaries,
            typed_before=typed_side_effect_count_before_replay,
            typed_after=typed_side_effect_count_after_replay,
            physical_files_before=physical_file_count_before_replay,
            physical_files_after=physical_file_count_after_replay,
            physical_bytes_before=physical_bytes_before_replay,
            physical_bytes_after=physical_bytes_after_replay,
        )
        refs = (
            policy_ref,
            private_workload_ref,
            private_result_set_ref,
            *fields["case_summary_refs"],
        )
        value = cls(
            report_id="scheduler-load-report://pending",
            policy_ref=policy_ref,
            private_workload_ref=private_workload_ref,
            private_result_set_ref=private_result_set_ref,
            case_summaries=ordered,
            phase_summaries=phase_summaries,
            typed_side_effect_count_before_replay=(typed_side_effect_count_before_replay),
            typed_side_effect_count_after_replay=(typed_side_effect_count_after_replay),
            physical_file_count_before_replay=physical_file_count_before_replay,
            physical_file_count_after_replay=physical_file_count_after_replay,
            physical_bytes_before_replay=physical_bytes_before_replay,
            physical_bytes_after_replay=physical_bytes_after_replay,
            **fields,
            report_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "report_id",
            "report_sha256",
            "scheduler-load-report",
            scheduler_load_report_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return scheduler_load_report_v2_ref(self)


def scheduler_load_policy_v2_carried_sha256(
    value: SchedulerLoadPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def scheduler_load_case_summary_v2_carried_sha256(
    value: SchedulerLoadCaseSummaryV2,
) -> str:
    return _carried(
        value,
        {"case_summary_id", "case_summary_sha256", "audit"},
    )


def scheduler_load_report_v2_carried_sha256(
    value: SchedulerLoadReportV2,
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={
            "report_id",
            "report_sha256",
            "audit",
            "case_summaries",
        },
        exclude_none=False,
    )
    return _payload_sha256(payload)


def scheduler_load_policy_v2_ref(
    value: SchedulerLoadPolicyV2,
) -> ObjectRef:
    validate_scheduler_load_policy_v2_identity(value)
    return _ref(
        "scheduler-load-policy",
        value.policy_id,
        value.policy_sha256,
    )


def scheduler_load_case_summary_v2_ref(
    value: SchedulerLoadCaseSummaryV2,
) -> ObjectRef:
    validate_scheduler_load_case_summary_v2_identity(value)
    return _ref(
        "scheduler-load-case-summary",
        value.case_summary_id,
        value.case_summary_sha256,
    )


def scheduler_load_report_v2_ref(
    value: SchedulerLoadReportV2,
) -> ObjectRef:
    validate_scheduler_load_report_v2_identity(value)
    return _ref(
        "scheduler-load-report",
        value.report_id,
        value.report_sha256,
    )


def validate_scheduler_load_policy_v2_identity(
    value: SchedulerLoadPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "scheduler-load-policy",
        scheduler_load_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_scheduler_load_case_summary_v2_identity(
    value: SchedulerLoadCaseSummaryV2,
) -> None:
    _validate_identity(
        value.case_summary_id,
        value.case_summary_sha256,
        "scheduler-load-case-summary",
        scheduler_load_case_summary_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_scheduler_load_report_v2_identity(
    value: SchedulerLoadReportV2,
) -> None:
    _validate_identity(
        value.report_id,
        value.report_sha256,
        "scheduler-load-report",
        scheduler_load_report_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_scheduler_load_case_classification(
    outcome: SchedulerLoadCaseOutcomeV2,
    reason_code: SchedulerLoadReasonCodeV2,
    *,
    replay_stable: bool,
) -> None:
    if outcome is SchedulerLoadCaseOutcomeV2.RECOVERED:
        allowed = frozenset({SchedulerLoadReasonCodeV2.NONE})
    elif outcome is SchedulerLoadCaseOutcomeV2.FAILED:
        allowed = frozenset(
            {
                SchedulerLoadReasonCodeV2.EXPECTED_FAULT_NOT_OBSERVED,
                SchedulerLoadReasonCodeV2.RESUME_FAILED,
                SchedulerLoadReasonCodeV2.TERMINAL_EVIDENCE_MISMATCH,
                SchedulerLoadReasonCodeV2.UNEXPECTED_RETRY,
                SchedulerLoadReasonCodeV2.DUPLICATE_IMMUTABLE_FACT,
                SchedulerLoadReasonCodeV2.REPLAY_MISMATCH,
                SchedulerLoadReasonCodeV2.OBSERVABILITY_INCOMPLETE,
            }
        )
    elif outcome is SchedulerLoadCaseOutcomeV2.INFRASTRUCTURE_ERROR:
        allowed = frozenset(
            {
                SchedulerLoadReasonCodeV2.MATERIAL_INTEGRITY_ERROR,
                SchedulerLoadReasonCodeV2.INTERNAL_ERROR,
            }
        )
    else:
        allowed = frozenset({SchedulerLoadReasonCodeV2.CHILD_STATE_INCOMPLETE})
    if reason_code not in allowed:
        raise ValueError("scheduler load case outcome and reason code disagree")
    if (
        outcome
        in {
            SchedulerLoadCaseOutcomeV2.INFRASTRUCTURE_ERROR,
            SchedulerLoadCaseOutcomeV2.INCOMPLETE,
        }
        and replay_stable
    ):
        raise ValueError("untrustworthy scheduler load case cannot claim replay")


def _recovered_case_fields(value: SchedulerLoadCaseSummaryV2) -> bool:
    return (
        value.fault_observed
        and value.resume_succeeded
        and value.replay_stable
        and value.attempt_count == 1
        and value.unexpected_retry_count == 0
        and value.duplicate_immutable_fact_count == 0
        and value.stage_run_count == 1
        and value.stage_result_count == 1
        and value.attempt_metrics_count == 1
        and value.completion_witness_count == 1
        and value.observability_complete
    )


def _derive_report_fields(
    cases: tuple[SchedulerLoadCaseSummaryV2, ...],
    phases: tuple[SchedulerLoadPhaseSummaryV2, ...],
    *,
    typed_before: int,
    typed_after: int,
    physical_files_before: int,
    physical_files_after: int,
    physical_bytes_before: int,
    physical_bytes_after: int,
) -> _DerivedReportFields:
    if len(cases) != SCHEDULER_LOAD_JOB_COUNT:
        raise ValueError("scheduler load report requires exactly 1000 cases")
    refs = tuple(scheduler_load_case_summary_v2_ref(item) for item in cases)
    if refs != tuple(sorted(refs, key=_ref_key)) or len(refs) != len(set(refs)):
        raise ValueError("scheduler load case summaries must be sorted and unique")
    private_refs = tuple(item.private_case_result_ref for item in cases)
    if len(private_refs) != len(set(private_refs)):
        raise ValueError("scheduler load cases reuse private results")
    if tuple(item.phase for item in phases) != tuple(SchedulerLoadPhaseV2):
        raise ValueError("scheduler load phases must use canonical order")
    if any(item.job_count != SCHEDULER_LOAD_JOB_COUNT for item in phases):
        raise ValueError("scheduler load phase counts differ from workload")

    fault_counts = Counter(item.assigned_fault_point for item in cases)
    expected_faults = Counter({fault: SCHEDULER_LOAD_JOBS_PER_FAULT for fault in SchedulerLoadFaultPointV2})
    if fault_counts != expected_faults:
        raise ValueError("scheduler load fault distribution differs from canonical workload")
    fault_rows = tuple(
        SchedulerLoadFaultCountV2(
            fault_point=fault,
            count=SCHEDULER_LOAD_JOBS_PER_FAULT,
        )
        for fault in SchedulerLoadFaultPointV2
    )
    partitions = {
        outcome: tuple(ref for item, ref in zip(cases, refs, strict=True) if item.outcome is outcome)
        for outcome in SchedulerLoadCaseOutcomeV2
    }
    typed_stable = typed_before == typed_after
    # WAL checkpoints may move bytes between SQLite artifacts without adding
    # authority. File-count closure plus exact typed facts is the replay gate;
    # byte totals remain observational.
    physical_stable = physical_files_before == physical_files_after
    complete = (
        len(partitions[SchedulerLoadCaseOutcomeV2.RECOVERED]) == SCHEDULER_LOAD_JOB_COUNT
        and typed_stable
        and physical_stable
    )
    outcome = SchedulerLoadReportOutcomeV2.COMPLETE if complete else SchedulerLoadReportOutcomeV2.INCOMPLETE
    return {
        "case_summary_refs": refs,
        "recovered_case_refs": partitions[SchedulerLoadCaseOutcomeV2.RECOVERED],
        "failed_case_refs": partitions[SchedulerLoadCaseOutcomeV2.FAILED],
        "infrastructure_error_case_refs": partitions[SchedulerLoadCaseOutcomeV2.INFRASTRUCTURE_ERROR],
        "incomplete_case_refs": partitions[SchedulerLoadCaseOutcomeV2.INCOMPLETE],
        "fault_counts": fault_rows,
        "unique_job_count": SCHEDULER_LOAD_JOB_COUNT,
        "total_faults_assigned": len(cases),
        "total_faults_observed": sum(item.fault_observed for item in cases),
        "total_resumes": sum(item.resume_succeeded for item in cases),
        "total_replays": sum(item.replay_stable for item in cases),
        "total_attempts": sum(item.attempt_count for item in cases),
        "total_unexpected_retries": sum(item.unexpected_retry_count for item in cases),
        "total_duplicate_immutable_facts": sum(item.duplicate_immutable_fact_count for item in cases),
        "total_stage_runs": sum(item.stage_run_count for item in cases),
        "total_stage_results": sum(item.stage_result_count for item in cases),
        "total_attempt_metrics": sum(item.attempt_metrics_count for item in cases),
        "total_completion_witnesses": sum(item.completion_witness_count for item in cases),
        "typed_replay_stable": typed_stable,
        "physical_replay_stable": physical_stable,
        "outcome": outcome,
    }


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
    expected = tuple(sorted(set(refs), key=_ref_key))
    if audit.input_refs != expected:
        raise ValueError(f"{label} audit refs are incomplete")
    versions = tuple(binding for binding in audit.governing_versions if binding.component == "scheduler-load")
    if len(versions) != 1 or versions[0].version != SCHEDULER_LOAD_POLICY_VERSION:
        raise ValueError(f"{label} audit is missing scheduler load policy")


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
    "SCHEDULER_LOAD_CLAIM_SCOPE",
    "SCHEDULER_LOAD_JOBS_PER_FAULT",
    "SCHEDULER_LOAD_JOB_COUNT",
    "SCHEDULER_LOAD_POLICY_VERSION",
    "SchedulerLoadCaseOutcomeV2",
    "SchedulerLoadCaseSummaryV2",
    "SchedulerLoadFaultCountV2",
    "SchedulerLoadFaultPointV2",
    "SchedulerLoadPhaseSummaryV2",
    "SchedulerLoadPhaseV2",
    "SchedulerLoadPolicyV2",
    "SchedulerLoadReasonCodeV2",
    "SchedulerLoadReportOutcomeV2",
    "SchedulerLoadReportV2",
    "scheduler_load_case_summary_v2_carried_sha256",
    "scheduler_load_case_summary_v2_ref",
    "scheduler_load_policy_v2_carried_sha256",
    "scheduler_load_policy_v2_ref",
    "scheduler_load_report_v2_carried_sha256",
    "scheduler_load_report_v2_ref",
    "validate_scheduler_load_case_classification",
    "validate_scheduler_load_case_summary_v2_identity",
    "validate_scheduler_load_policy_v2_identity",
    "validate_scheduler_load_report_v2_identity",
]
