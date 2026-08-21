from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedDatasetJobPlanV2,
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    WorkControlPolicyV2,
)
from eval_factory.contracts.scheduler_load_v2 import (
    SCHEDULER_LOAD_JOB_COUNT,
    SCHEDULER_LOAD_JOBS_PER_FAULT,
    SCHEDULER_LOAD_POLICY_VERSION,
    SchedulerLoadCaseOutcomeV2,
    SchedulerLoadCaseSummaryV2,
    SchedulerLoadFaultPointV2,
    SchedulerLoadReasonCodeV2,
    scheduler_load_case_summary_v2_ref,
    validate_scheduler_load_case_classification,
)


class SchedulerLoadMemberV1(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-member/private-v1"] = (
        "eval-factory/scheduler-load-member/private-v1"
    )
    case_key: Identifier
    ordinal: int = Field(ge=0, lt=1000)
    descriptor_sha256: Sha256
    assigned_fault_point: SchedulerLoadFaultPointV2

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        observed = scheduler_load_member_v1_carried_sha256(self)
        if self.case_key not in {
            "scheduler-load-case://pending",
            f"scheduler-load-case://sha256/{observed}",
        }:
            raise ValueError("scheduler load member case key is stale")
        return self

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        descriptor_sha256: str,
        assigned_fault_point: SchedulerLoadFaultPointV2,
    ) -> SchedulerLoadMemberV1:
        value = cls(
            case_key="scheduler-load-case://pending",
            ordinal=ordinal,
            descriptor_sha256=descriptor_sha256,
            assigned_fault_point=assigned_fault_point,
        )
        digest = scheduler_load_member_v1_carried_sha256(value)
        return value.model_copy(update={"case_key": f"scheduler-load-case://sha256/{digest}"})

    def to_ref(self) -> ObjectRef:
        return scheduler_load_member_v1_ref(self)


class SchedulerLoadWorkloadV1(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-workload/private-v1"] = (
        "eval-factory/scheduler-load-workload/private-v1"
    )
    workload_id: Identifier
    policy_ref: ObjectRef
    members: tuple[SchedulerLoadMemberV1, ...] = Field(
        min_length=1000,
        max_length=1000,
    )
    policy_version: Literal["scheduler-load/r8-05-v1"] = SCHEDULER_LOAD_POLICY_VERSION
    workload_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_workload(self) -> Self:
        _require_ref(
            self.policy_ref,
            "scheduler-load-policy",
            "v2",
            "policy_ref",
        )
        if tuple(member.ordinal for member in self.members) != tuple(range(SCHEDULER_LOAD_JOB_COUNT)):
            raise ValueError("scheduler load members must use canonical ordinals")
        if len({member.case_key for member in self.members}) != len(self.members):
            raise ValueError("scheduler load member case keys must be unique")
        if len({member.descriptor_sha256 for member in self.members}) != len(self.members):
            raise ValueError("scheduler load descriptors must be unique")
        expected_faults = tuple(SchedulerLoadFaultPointV2)
        if tuple(member.assigned_fault_point for member in self.members) != tuple(
            expected_faults[index % len(expected_faults)] for index in range(SCHEDULER_LOAD_JOB_COUNT)
        ):
            raise ValueError("scheduler load fault assignment is not canonical")
        for fault in expected_faults:
            if (
                sum(member.assigned_fault_point is fault for member in self.members)
                != SCHEDULER_LOAD_JOBS_PER_FAULT
            ):
                raise ValueError("scheduler load fault buckets are not balanced")
        _require_audit(self.audit, (self.policy_ref,), "scheduler load workload")
        _validate_identity(
            self.workload_id,
            self.workload_sha256,
            "scheduler-load-workload",
            scheduler_load_workload_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        members: tuple[SchedulerLoadMemberV1, ...],
        audit: ContractAudit,
    ) -> SchedulerLoadWorkloadV1:
        ordered = tuple(sorted(members, key=lambda member: member.ordinal))
        value = cls(
            workload_id="scheduler-load-workload://pending",
            policy_ref=policy_ref,
            members=ordered,
            workload_sha256="0" * 64,
            audit=_safe_audit(audit, (policy_ref,)),
        )
        return _finalize(
            value,
            "workload_id",
            "workload_sha256",
            "scheduler-load-workload",
            scheduler_load_workload_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return scheduler_load_workload_v1_ref(self)


@dataclass(frozen=True, slots=True)
class PreparedSchedulerLoadCase:
    member: SchedulerLoadMemberV1
    job_spec: DatasetJobSpecV2
    dataset_job_spec_ref: ObjectRef
    resolved_plan: ResolvedDatasetJobPlanV2
    work_graph: ResolvedJobWorkGraphV2
    control_policy: WorkControlPolicyV2
    root_work_unit: ResolvedWorkUnitV2
    holder_ref: ObjectRef
    output_ref: ObjectRef


class SchedulerLoadFaultObservationV1(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-fault-observation/private-v1"] = (
        "eval-factory/scheduler-load-fault-observation/private-v1"
    )
    fault_observation_id: Identifier
    member_ref: ObjectRef
    workload_ref: ObjectRef
    policy_ref: ObjectRef
    fault_point: SchedulerLoadFaultPointV2
    initial_elapsed_nanoseconds: int = Field(ge=1, le=10**18)
    policy_version: Literal["scheduler-load/r8-05-v1"] = SCHEDULER_LOAD_POLICY_VERSION
    fault_observation_sha256: Sha256

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        _require_ref(
            self.member_ref,
            "scheduler-load-member",
            "private-v1",
            "member_ref",
        )
        _require_ref(
            self.workload_ref,
            "scheduler-load-workload",
            "private-v1",
            "workload_ref",
        )
        _require_ref(
            self.policy_ref,
            "scheduler-load-policy",
            "v2",
            "policy_ref",
        )
        _validate_identity(
            self.fault_observation_id,
            self.fault_observation_sha256,
            "scheduler-load-fault-observation",
            scheduler_load_fault_observation_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        member_ref: ObjectRef,
        workload_ref: ObjectRef,
        policy_ref: ObjectRef,
        fault_point: SchedulerLoadFaultPointV2,
        initial_elapsed_nanoseconds: int,
    ) -> SchedulerLoadFaultObservationV1:
        value = cls(
            fault_observation_id="scheduler-load-fault-observation://pending",
            member_ref=member_ref,
            workload_ref=workload_ref,
            policy_ref=policy_ref,
            fault_point=fault_point,
            initial_elapsed_nanoseconds=initial_elapsed_nanoseconds,
            fault_observation_sha256="0" * 64,
        )
        return _finalize(
            value,
            "fault_observation_id",
            "fault_observation_sha256",
            "scheduler-load-fault-observation",
            scheduler_load_fault_observation_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return scheduler_load_fault_observation_v1_ref(self)


class SchedulerLoadCaseResultV1(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-case-result/private-v1"] = (
        "eval-factory/scheduler-load-case-result/private-v1"
    )
    case_result_id: Identifier
    member_ref: ObjectRef
    workload_ref: ObjectRef
    policy_ref: ObjectRef
    dataset_job_spec_ref: ObjectRef
    resolved_plan_ref: ObjectRef
    work_graph_ref: ObjectRef
    control_policy_ref: ObjectRef
    work_unit_ref: ObjectRef
    readiness_ref: ObjectRef | None = None
    lease_ref: ObjectRef | None = None
    stage_run_ref: ObjectRef | None = None
    stage_result_ref: ObjectRef | None = None
    attempt_metrics_ref: ObjectRef | None = None
    terminal_lease_event_ref: ObjectRef | None = None
    observability_report_ref: ObjectRef | None = None
    output_ref: ObjectRef | None = None
    job_id: Identifier
    item_id: Identifier
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
    initial_elapsed_nanoseconds: int = Field(ge=0, le=10**18)
    recovery_elapsed_nanoseconds: int = Field(ge=0, le=10**18)
    replay_elapsed_nanoseconds: int = Field(ge=0, le=10**18)
    typed_side_effect_count_before_replay: int = Field(ge=0, le=1_000_000_000)
    typed_side_effect_count_after_replay: int = Field(ge=0, le=1_000_000_000)
    physical_file_count_before_replay: int = Field(ge=0, le=1_000_000)
    physical_file_count_after_replay: int = Field(ge=0, le=1_000_000)
    physical_bytes_before_replay: int = Field(ge=0, le=10**15)
    physical_bytes_after_replay: int = Field(ge=0, le=10**15)
    outcome: SchedulerLoadCaseOutcomeV2
    reason_code: SchedulerLoadReasonCodeV2
    policy_version: Literal["scheduler-load/r8-05-v1"] = SCHEDULER_LOAD_POLICY_VERSION
    case_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        for ref, object_type, version, name in (
            (
                self.member_ref,
                "scheduler-load-member",
                "private-v1",
                "member_ref",
            ),
            (
                self.workload_ref,
                "scheduler-load-workload",
                "private-v1",
                "workload_ref",
            ),
            (self.policy_ref, "scheduler-load-policy", "v2", "policy_ref"),
            (
                self.dataset_job_spec_ref,
                "dataset-job-spec",
                "v2",
                "dataset_job_spec_ref",
            ),
            (
                self.resolved_plan_ref,
                "resolved-dataset-job-plan",
                "v2",
                "resolved_plan_ref",
            ),
            (
                self.work_graph_ref,
                "resolved-job-work-graph",
                "v2",
                "work_graph_ref",
            ),
            (
                self.control_policy_ref,
                "work-control-policy",
                "v2",
                "control_policy_ref",
            ),
            (
                self.work_unit_ref,
                "resolved-work-unit",
                "v2",
                "work_unit_ref",
            ),
        ):
            _require_ref(ref, object_type, version, name)
        optional_types = (
            (self.readiness_ref, "work-readiness-snapshot"),
            (self.lease_ref, "work-lease"),
            (self.stage_run_ref, "stage-run"),
            (self.stage_result_ref, "stage-result"),
            (self.attempt_metrics_ref, "work-attempt-metrics"),
            (self.terminal_lease_event_ref, "work-lease-event"),
            (self.observability_report_ref, "batch-audit-report"),
            (self.output_ref, "scheduler-load-output"),
        )
        for optional_ref, object_type in optional_types:
            if optional_ref is not None and optional_ref.object_type != object_type:
                raise ValueError(f"private case {object_type} ref is invalid")
        validate_scheduler_load_case_classification(
            self.outcome,
            self.reason_code,
            replay_stable=self.replay_stable,
        )
        if self.outcome is SchedulerLoadCaseOutcomeV2.RECOVERED and not _recovered_private_fields(self):
            raise ValueError("recovered private scheduler load case is incomplete")
        refs = _case_input_refs(self)
        _require_audit(self.audit, refs, "scheduler load case result")
        _validate_identity(
            self.case_result_id,
            self.case_result_sha256,
            "scheduler-load-case-result",
            scheduler_load_case_result_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        member_ref: ObjectRef,
        workload_ref: ObjectRef,
        policy_ref: ObjectRef,
        dataset_job_spec_ref: ObjectRef,
        resolved_plan_ref: ObjectRef,
        work_graph_ref: ObjectRef,
        control_policy_ref: ObjectRef,
        work_unit_ref: ObjectRef,
        readiness_ref: ObjectRef | None,
        lease_ref: ObjectRef | None,
        stage_run_ref: ObjectRef | None,
        stage_result_ref: ObjectRef | None,
        attempt_metrics_ref: ObjectRef | None,
        terminal_lease_event_ref: ObjectRef | None,
        observability_report_ref: ObjectRef | None,
        output_ref: ObjectRef | None,
        job_id: str,
        item_id: str,
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
        initial_elapsed_nanoseconds: int,
        recovery_elapsed_nanoseconds: int,
        replay_elapsed_nanoseconds: int,
        typed_side_effect_count_before_replay: int,
        typed_side_effect_count_after_replay: int,
        physical_file_count_before_replay: int,
        physical_file_count_after_replay: int,
        physical_bytes_before_replay: int,
        physical_bytes_after_replay: int,
        outcome: SchedulerLoadCaseOutcomeV2,
        reason_code: SchedulerLoadReasonCodeV2,
        audit: ContractAudit,
    ) -> SchedulerLoadCaseResultV1:
        refs = _sorted_refs(
            (
                member_ref,
                workload_ref,
                policy_ref,
                dataset_job_spec_ref,
                resolved_plan_ref,
                work_graph_ref,
                control_policy_ref,
                work_unit_ref,
                *(
                    ref
                    for ref in (
                        readiness_ref,
                        lease_ref,
                        stage_run_ref,
                        stage_result_ref,
                        attempt_metrics_ref,
                        terminal_lease_event_ref,
                        observability_report_ref,
                        output_ref,
                    )
                    if ref is not None
                ),
            )
        )
        value = cls(
            case_result_id="scheduler-load-case-result://pending",
            member_ref=member_ref,
            workload_ref=workload_ref,
            policy_ref=policy_ref,
            dataset_job_spec_ref=dataset_job_spec_ref,
            resolved_plan_ref=resolved_plan_ref,
            work_graph_ref=work_graph_ref,
            control_policy_ref=control_policy_ref,
            work_unit_ref=work_unit_ref,
            readiness_ref=readiness_ref,
            lease_ref=lease_ref,
            stage_run_ref=stage_run_ref,
            stage_result_ref=stage_result_ref,
            attempt_metrics_ref=attempt_metrics_ref,
            terminal_lease_event_ref=terminal_lease_event_ref,
            observability_report_ref=observability_report_ref,
            output_ref=output_ref,
            job_id=job_id,
            item_id=item_id,
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
            initial_elapsed_nanoseconds=initial_elapsed_nanoseconds,
            recovery_elapsed_nanoseconds=recovery_elapsed_nanoseconds,
            replay_elapsed_nanoseconds=replay_elapsed_nanoseconds,
            typed_side_effect_count_before_replay=(typed_side_effect_count_before_replay),
            typed_side_effect_count_after_replay=(typed_side_effect_count_after_replay),
            physical_file_count_before_replay=physical_file_count_before_replay,
            physical_file_count_after_replay=physical_file_count_after_replay,
            physical_bytes_before_replay=physical_bytes_before_replay,
            physical_bytes_after_replay=physical_bytes_after_replay,
            outcome=outcome,
            reason_code=reason_code,
            case_result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "case_result_id",
            "case_result_sha256",
            "scheduler-load-case-result",
            scheduler_load_case_result_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return scheduler_load_case_result_v1_ref(self)

    def to_public_summary(
        self,
        *,
        audit: ContractAudit,
    ) -> SchedulerLoadCaseSummaryV2:
        return SchedulerLoadCaseSummaryV2.create(
            private_case_result_ref=self.to_ref(),
            outcome=self.outcome,
            reason_code=self.reason_code,
            assigned_fault_point=self.assigned_fault_point,
            fault_observed=self.fault_observed,
            resume_succeeded=self.resume_succeeded,
            replay_stable=self.replay_stable,
            attempt_count=self.attempt_count,
            unexpected_retry_count=self.unexpected_retry_count,
            duplicate_immutable_fact_count=self.duplicate_immutable_fact_count,
            stage_run_count=self.stage_run_count,
            stage_result_count=self.stage_result_count,
            attempt_metrics_count=self.attempt_metrics_count,
            completion_witness_count=self.completion_witness_count,
            observability_complete=self.observability_complete,
            audit=audit,
        )


class SchedulerLoadResultSetV1(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-result-set/private-v1"] = (
        "eval-factory/scheduler-load-result-set/private-v1"
    )
    result_set_id: Identifier
    workload_ref: ObjectRef
    policy_ref: ObjectRef
    case_result_refs: tuple[ObjectRef, ...] = Field(
        min_length=1000,
        max_length=1000,
    )
    case_summary_refs: tuple[ObjectRef, ...] = Field(
        min_length=1000,
        max_length=1000,
    )
    policy_version: Literal["scheduler-load/r8-05-v1"] = SCHEDULER_LOAD_POLICY_VERSION
    result_set_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result_set(self) -> Self:
        _require_ref(
            self.workload_ref,
            "scheduler-load-workload",
            "private-v1",
            "workload_ref",
        )
        _require_ref(
            self.policy_ref,
            "scheduler-load-policy",
            "v2",
            "policy_ref",
        )
        _require_sorted_unique_refs(
            self.case_result_refs,
            "case_result_refs",
            object_type="scheduler-load-case-result",
            version="private-v1",
        )
        _require_sorted_unique_refs(
            self.case_summary_refs,
            "case_summary_refs",
            object_type="scheduler-load-case-summary",
            version="v2",
        )
        refs = (
            self.workload_ref,
            self.policy_ref,
            *self.case_result_refs,
            *self.case_summary_refs,
        )
        _require_audit(self.audit, refs, "scheduler load result set")
        _validate_identity(
            self.result_set_id,
            self.result_set_sha256,
            "scheduler-load-result-set",
            scheduler_load_result_set_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        workload_ref: ObjectRef,
        policy_ref: ObjectRef,
        case_results: tuple[SchedulerLoadCaseResultV1, ...],
        case_summaries: tuple[SchedulerLoadCaseSummaryV2, ...],
        audit: ContractAudit,
    ) -> SchedulerLoadResultSetV1:
        result_refs = _sorted_refs(tuple(result.to_ref() for result in case_results))
        summary_refs = _sorted_refs(
            tuple(scheduler_load_case_summary_v2_ref(summary) for summary in case_summaries)
        )
        refs = (
            workload_ref,
            policy_ref,
            *result_refs,
            *summary_refs,
        )
        value = cls(
            result_set_id="scheduler-load-result-set://pending",
            workload_ref=workload_ref,
            policy_ref=policy_ref,
            case_result_refs=result_refs,
            case_summary_refs=summary_refs,
            result_set_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "result_set_id",
            "result_set_sha256",
            "scheduler-load-result-set",
            scheduler_load_result_set_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return scheduler_load_result_set_v1_ref(self)


class SchedulerLoadRunLedgerV1(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-run-ledger/private-v1"] = (
        "eval-factory/scheduler-load-run-ledger/private-v1"
    )
    run_ledger_id: Identifier
    workload_ref: ObjectRef
    policy_ref: ObjectRef
    job_store_ref: ObjectRef
    prepared_authority_refs: tuple[ObjectRef, ...] = Field(
        min_length=6000,
        max_length=6000,
    )
    policy_version: Literal["scheduler-load/r8-05-v1"] = SCHEDULER_LOAD_POLICY_VERSION
    run_ledger_sha256: Sha256

    @model_validator(mode="after")
    def validate_ledger(self) -> Self:
        _require_ref(
            self.workload_ref,
            "scheduler-load-workload",
            "private-v1",
            "workload_ref",
        )
        _require_ref(
            self.policy_ref,
            "scheduler-load-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.job_store_ref,
            "scheduler-load-job-store",
            "private-v1",
            "job_store_ref",
        )
        if self.prepared_authority_refs != _sorted_refs(self.prepared_authority_refs):
            raise ValueError("prepared scheduler authorities must be sorted")
        if len(set(self.prepared_authority_refs)) != 6000:
            raise ValueError("prepared scheduler authorities must be unique")
        _validate_identity(
            self.run_ledger_id,
            self.run_ledger_sha256,
            "scheduler-load-run-ledger",
            scheduler_load_run_ledger_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        workload_ref: ObjectRef,
        policy_ref: ObjectRef,
        job_store_ref: ObjectRef,
        prepared: tuple[PreparedSchedulerLoadCase, ...],
    ) -> SchedulerLoadRunLedgerV1:
        refs = _sorted_refs(
            tuple(
                ref
                for case in prepared
                for ref in (
                    case.member.to_ref(),
                    case.dataset_job_spec_ref,
                    _ref_for_plan(case.resolved_plan),
                    _ref_for_graph(case.work_graph),
                    _ref_for_policy(case.control_policy),
                    _ref_for_work_unit(case.root_work_unit),
                )
            )
        )
        value = cls(
            run_ledger_id="scheduler-load-run-ledger://pending",
            workload_ref=workload_ref,
            policy_ref=policy_ref,
            job_store_ref=job_store_ref,
            prepared_authority_refs=refs,
            run_ledger_sha256="0" * 64,
        )
        return _finalize(
            value,
            "run_ledger_id",
            "run_ledger_sha256",
            "scheduler-load-run-ledger",
            scheduler_load_run_ledger_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return scheduler_load_run_ledger_v1_ref(self)


def scheduler_load_member_v1_carried_sha256(
    value: SchedulerLoadMemberV1,
) -> str:
    return _carried(value, {"case_key"})


def scheduler_load_workload_v1_carried_sha256(
    value: SchedulerLoadWorkloadV1,
) -> str:
    return _carried(value, {"workload_id", "workload_sha256", "audit"})


def scheduler_load_fault_observation_v1_carried_sha256(
    value: SchedulerLoadFaultObservationV1,
) -> str:
    return _carried(
        value,
        {"fault_observation_id", "fault_observation_sha256"},
    )


def scheduler_load_case_result_v1_carried_sha256(
    value: SchedulerLoadCaseResultV1,
) -> str:
    return _carried(value, {"case_result_id", "case_result_sha256", "audit"})


def scheduler_load_result_set_v1_carried_sha256(
    value: SchedulerLoadResultSetV1,
) -> str:
    return _carried(value, {"result_set_id", "result_set_sha256", "audit"})


def scheduler_load_run_ledger_v1_carried_sha256(
    value: SchedulerLoadRunLedgerV1,
) -> str:
    return _carried(value, {"run_ledger_id", "run_ledger_sha256"})


def scheduler_load_member_v1_ref(value: SchedulerLoadMemberV1) -> ObjectRef:
    return _private_ref(
        "scheduler-load-member",
        value.case_key,
        scheduler_load_member_v1_carried_sha256(value),
    )


def scheduler_load_workload_v1_ref(
    value: SchedulerLoadWorkloadV1,
) -> ObjectRef:
    _validate_identity(
        value.workload_id,
        value.workload_sha256,
        "scheduler-load-workload",
        scheduler_load_workload_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _private_ref(
        "scheduler-load-workload",
        value.workload_id,
        value.workload_sha256,
    )


def scheduler_load_fault_observation_v1_ref(
    value: SchedulerLoadFaultObservationV1,
) -> ObjectRef:
    _validate_identity(
        value.fault_observation_id,
        value.fault_observation_sha256,
        "scheduler-load-fault-observation",
        scheduler_load_fault_observation_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _private_ref(
        "scheduler-load-fault-observation",
        value.fault_observation_id,
        value.fault_observation_sha256,
    )


def scheduler_load_case_result_v1_ref(
    value: SchedulerLoadCaseResultV1,
) -> ObjectRef:
    _validate_identity(
        value.case_result_id,
        value.case_result_sha256,
        "scheduler-load-case-result",
        scheduler_load_case_result_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _private_ref(
        "scheduler-load-case-result",
        value.case_result_id,
        value.case_result_sha256,
    )


def scheduler_load_result_set_v1_ref(
    value: SchedulerLoadResultSetV1,
) -> ObjectRef:
    _validate_identity(
        value.result_set_id,
        value.result_set_sha256,
        "scheduler-load-result-set",
        scheduler_load_result_set_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _private_ref(
        "scheduler-load-result-set",
        value.result_set_id,
        value.result_set_sha256,
    )


def scheduler_load_run_ledger_v1_ref(
    value: SchedulerLoadRunLedgerV1,
) -> ObjectRef:
    _validate_identity(
        value.run_ledger_id,
        value.run_ledger_sha256,
        "scheduler-load-run-ledger",
        scheduler_load_run_ledger_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _private_ref(
        "scheduler-load-run-ledger",
        value.run_ledger_id,
        value.run_ledger_sha256,
    )


def _recovered_private_fields(value: SchedulerLoadCaseResultV1) -> bool:
    return (
        all(
            ref is not None
            for ref in (
                value.readiness_ref,
                value.lease_ref,
                value.stage_run_ref,
                value.stage_result_ref,
                value.attempt_metrics_ref,
                value.terminal_lease_event_ref,
                value.observability_report_ref,
                value.output_ref,
            )
        )
        and value.fault_observed
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
        and value.initial_elapsed_nanoseconds > 0
        and value.recovery_elapsed_nanoseconds > 0
        and value.replay_elapsed_nanoseconds > 0
        and value.typed_side_effect_count_before_replay == value.typed_side_effect_count_after_replay
        and value.physical_file_count_before_replay == value.physical_file_count_after_replay
    )


def _case_input_refs(
    value: SchedulerLoadCaseResultV1,
) -> tuple[ObjectRef, ...]:
    return _sorted_refs(
        (
            value.member_ref,
            value.workload_ref,
            value.policy_ref,
            value.dataset_job_spec_ref,
            value.resolved_plan_ref,
            value.work_graph_ref,
            value.control_policy_ref,
            value.work_unit_ref,
            *(
                ref
                for ref in (
                    value.readiness_ref,
                    value.lease_ref,
                    value.stage_run_ref,
                    value.stage_result_ref,
                    value.attempt_metrics_ref,
                    value.terminal_lease_event_ref,
                    value.observability_report_ref,
                    value.output_ref,
                )
                if ref is not None
            ),
        )
    )


def _ref_for_plan(value: ResolvedDatasetJobPlanV2) -> ObjectRef:
    from eval_factory.contracts.orchestration_v2 import (
        resolved_dataset_job_plan_v2_ref,
    )

    return resolved_dataset_job_plan_v2_ref(value)


def _ref_for_graph(value: ResolvedJobWorkGraphV2) -> ObjectRef:
    from eval_factory.contracts.orchestration_v2 import (
        resolved_job_work_graph_v2_ref,
    )

    return resolved_job_work_graph_v2_ref(value)


def _ref_for_policy(value: WorkControlPolicyV2) -> ObjectRef:
    from eval_factory.contracts.orchestration_v2 import work_control_policy_v2_ref

    return work_control_policy_v2_ref(value)


def _ref_for_work_unit(value: ResolvedWorkUnitV2) -> ObjectRef:
    from eval_factory.contracts.orchestration_v2 import resolved_work_unit_v2_ref

    return resolved_work_unit_v2_ref(value)


def _require_sorted_unique_refs(
    refs: tuple[ObjectRef, ...],
    field_name: str,
    *,
    object_type: str,
    version: str,
) -> None:
    if refs != _sorted_refs(refs) or len(refs) != len(set(refs)):
        raise ValueError(f"{field_name} must be sorted and unique")
    if any(ref.object_type != object_type or ref.object_version != version for ref in refs):
        raise ValueError(f"{field_name} contains an invalid ref")


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} has the wrong type or version")


def _require_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError(f"{label} audit refs are incomplete")
    versions = tuple(binding for binding in audit.governing_versions if binding.component == "scheduler-load")
    if len(versions) != 1 or versions[0].version != SCHEDULER_LOAD_POLICY_VERSION:
        raise ValueError(f"{label} audit is missing scheduler load policy")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


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


def _sorted_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(refs), key=_ref_key))


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _carried(value: ContractModelV2, exclude: set[str]) -> str:
    payload = value.model_dump(
        mode="python",
        exclude=exclude,
        exclude_none=False,
    )
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(payload),
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
    "PreparedSchedulerLoadCase",
    "SchedulerLoadCaseResultV1",
    "SchedulerLoadFaultObservationV1",
    "SchedulerLoadMemberV1",
    "SchedulerLoadResultSetV1",
    "SchedulerLoadRunLedgerV1",
    "SchedulerLoadWorkloadV1",
    "scheduler_load_case_result_v1_carried_sha256",
    "scheduler_load_case_result_v1_ref",
    "scheduler_load_fault_observation_v1_carried_sha256",
    "scheduler_load_fault_observation_v1_ref",
    "scheduler_load_member_v1_carried_sha256",
    "scheduler_load_member_v1_ref",
    "scheduler_load_result_set_v1_carried_sha256",
    "scheduler_load_result_set_v1_ref",
    "scheduler_load_run_ledger_v1_carried_sha256",
    "scheduler_load_run_ledger_v1_ref",
    "scheduler_load_workload_v1_carried_sha256",
    "scheduler_load_workload_v1_ref",
]
