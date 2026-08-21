from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.scheduler_load_v2 import (
    SCHEDULER_LOAD_CLAIM_SCOPE,
    SchedulerLoadCaseOutcomeV2,
    SchedulerLoadCaseSummaryV2,
    SchedulerLoadFaultPointV2,
    SchedulerLoadPhaseSummaryV2,
    SchedulerLoadPhaseV2,
    SchedulerLoadPolicyV2,
    SchedulerLoadReasonCodeV2,
    SchedulerLoadReportOutcomeV2,
    SchedulerLoadReportV2,
    scheduler_load_case_summary_v2_ref,
    validate_scheduler_load_policy_v2_identity,
    validate_scheduler_load_report_v2_identity,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 2, tzinfo=UTC)
GOLD_PATH = (
    Path(__file__).resolve().parents[3] / "evals/golden/eval_factory/readiness/r8-05-scheduler-load-v1.json"
)


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="scheduler-load-contract-test",
        governing_versions=(
            VersionBinding(
                component="scheduler-load",
                version="scheduler-load/r8-05-v1",
            ),
        ),
        input_refs=tuple(refs),
    )


def _private_ref(object_type: str, suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{suffix:0>64}",
        object_version="private-v1",
        object_sha256=f"{suffix:0>64}",
    )


def _policy() -> SchedulerLoadPolicyV2:
    return SchedulerLoadPolicyV2.create(
        lease_duration_seconds=300,
        heartbeat_extension_seconds=60,
        max_private_bytes=100_000_000,
        max_report_bytes=100_000_000,
        audit=_audit(),
    )


def _case(index: int) -> SchedulerLoadCaseSummaryV2:
    return SchedulerLoadCaseSummaryV2.create(
        private_case_result_ref=_private_ref(
            "scheduler-load-case-result",
            f"{index + 1:064x}",
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


def _phases() -> tuple[SchedulerLoadPhaseSummaryV2, ...]:
    return tuple(
        SchedulerLoadPhaseSummaryV2.create(
            phase=phase,
            job_count=1000,
            operation_count=1000 * (index + 1),
            elapsed_nanoseconds=1_000_000_000 * (index + 1),
        )
        for index, phase in enumerate(SchedulerLoadPhaseV2)
    )


def _report(
    *,
    replace_index: int | None = None,
    replacement_outcome: SchedulerLoadCaseOutcomeV2 = SchedulerLoadCaseOutcomeV2.INCOMPLETE,
    replacement_reason: SchedulerLoadReasonCodeV2 = SchedulerLoadReasonCodeV2.CHILD_STATE_INCOMPLETE,
) -> SchedulerLoadReportV2:
    cases = list(_case(index) for index in range(1000))
    if replace_index is not None:
        original = cases[replace_index]
        cases[replace_index] = SchedulerLoadCaseSummaryV2.create(
            private_case_result_ref=original.private_case_result_ref,
            outcome=replacement_outcome,
            reason_code=replacement_reason,
            assigned_fault_point=original.assigned_fault_point,
            fault_observed=False,
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
            audit=_audit(),
        )
    return SchedulerLoadReportV2.create(
        policy_ref=_policy().to_ref(),
        private_workload_ref=_private_ref("scheduler-load-workload", "b" * 64),
        private_result_set_ref=_private_ref("scheduler-load-result-set", "c" * 64),
        case_summaries=tuple(cases),
        phase_summaries=_phases(),
        typed_side_effect_count_before_replay=10_000,
        typed_side_effect_count_after_replay=10_000,
        physical_file_count_before_replay=1,
        physical_file_count_after_replay=1,
        physical_bytes_before_replay=1_000_000,
        physical_bytes_after_replay=1_000_000,
        audit=_audit(),
    )


def test_policy_binds_exact_fixed_scheduler_load_profile() -> None:
    policy = _policy()

    validate_scheduler_load_policy_v2_identity(policy)
    assert policy.required_job_count == 1000
    assert policy.worker_count == 1
    assert tuple(stage.value for stage in policy.requested_stages) == ("trace_index",)
    assert policy.require_fault_per_job is True
    assert policy.require_resume_success is True
    assert policy.require_exact_replay is True
    assert policy.max_attempts == 1
    assert policy.retry_delay_seconds == ()
    assert policy.max_unexpected_retries == 0
    assert policy.fault_points == tuple(SchedulerLoadFaultPointV2)
    assert policy.plan_policy_version == "dataset-job-stage-policy/r6-01-v1"
    assert policy.work_graph_policy_version == "dataset-work-graph/r6-02-v1"
    assert policy.work_control_policy_version == "work-control/r6-05-v1"
    assert policy.observability_policy_version == "batch-observability/r6-06-v1"


def test_contracts_are_strict_and_versions_are_literal() -> None:
    policy = _policy()
    payload = policy.model_dump(mode="python")
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        SchedulerLoadPolicyV2.model_validate(payload)

    payload = policy.model_dump(mode="python")
    payload["schema_version"] = "eval-factory/scheduler-load-policy/v1"
    with pytest.raises(ValidationError):
        SchedulerLoadPolicyV2.model_validate(payload)


def test_policy_rejects_profile_drift() -> None:
    policy = _policy()
    for field, value in (
        ("required_job_count", 999),
        ("worker_count", 2),
        ("require_fault_per_job", False),
        ("max_attempts", 2),
        ("retry_delay_seconds", (1,)),
        ("fault_points", tuple(reversed(tuple(SchedulerLoadFaultPointV2)))),
    ):
        payload = policy.model_dump(mode="python")
        payload[field] = value
        with pytest.raises(ValidationError):
            SchedulerLoadPolicyV2.model_validate(payload)


def test_phase_summary_uses_integer_measured_throughput() -> None:
    value = SchedulerLoadPhaseSummaryV2.create(
        phase=SchedulerLoadPhaseV2.SYNTHETIC_INITIAL,
        job_count=1000,
        operation_count=8000,
        elapsed_nanoseconds=2_000_000_000,
    )

    assert value.throughput_millijobs_per_second == 500_000
    with pytest.raises(ValidationError):
        SchedulerLoadPhaseSummaryV2.create(
            phase=SchedulerLoadPhaseV2.SYNTHETIC_INITIAL,
            job_count=1000,
            operation_count=8000,
            elapsed_nanoseconds=0,
        )


@pytest.mark.parametrize(
    ("outcome", "reason"),
    (
        (
            SchedulerLoadCaseOutcomeV2.RECOVERED,
            SchedulerLoadReasonCodeV2.RESUME_FAILED,
        ),
        (
            SchedulerLoadCaseOutcomeV2.FAILED,
            SchedulerLoadReasonCodeV2.INTERNAL_ERROR,
        ),
        (
            SchedulerLoadCaseOutcomeV2.INFRASTRUCTURE_ERROR,
            SchedulerLoadReasonCodeV2.REPLAY_MISMATCH,
        ),
        (
            SchedulerLoadCaseOutcomeV2.INCOMPLETE,
            SchedulerLoadReasonCodeV2.MATERIAL_INTEGRITY_ERROR,
        ),
    ),
)
def test_case_outcome_reason_matrix_is_closed(
    outcome: SchedulerLoadCaseOutcomeV2,
    reason: SchedulerLoadReasonCodeV2,
) -> None:
    payload = _case(0).model_dump(mode="python")
    payload.update(
        {
            "outcome": outcome,
            "reason_code": reason,
            "case_summary_id": "scheduler-load-case-summary://pending",
            "case_summary_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValidationError, match="outcome and reason"):
        SchedulerLoadCaseSummaryV2.model_validate(payload)


def test_recovered_case_requires_exact_complete_evidence() -> None:
    base = _case(0)
    for field, value in (
        ("fault_observed", False),
        ("resume_succeeded", False),
        ("replay_stable", False),
        ("attempt_count", 2),
        ("unexpected_retry_count", 1),
        ("duplicate_immutable_fact_count", 1),
        ("stage_run_count", 0),
        ("stage_result_count", 0),
        ("attempt_metrics_count", 0),
        ("completion_witness_count", 0),
        ("observability_complete", False),
    ):
        payload = base.model_dump(mode="python")
        payload.update(
            {
                field: value,
                "case_summary_id": "scheduler-load-case-summary://pending",
                "case_summary_sha256": "0" * 64,
            }
        )
        with pytest.raises(ValidationError, match="recovered"):
            SchedulerLoadCaseSummaryV2.model_validate(payload)


def test_complete_report_closes_exact_1000_job_fault_distribution() -> None:
    report = _report()

    validate_scheduler_load_report_v2_identity(report)
    assert report.outcome is SchedulerLoadReportOutcomeV2.COMPLETE
    assert report.claim_scope == SCHEDULER_LOAD_CLAIM_SCOPE
    assert report.unique_job_count == 1000
    assert len(report.case_summary_refs) == 1000
    assert len(report.recovered_case_refs) == 1000
    assert report.failed_case_refs == ()
    assert report.infrastructure_error_case_refs == ()
    assert report.incomplete_case_refs == ()
    assert Counter({row.fault_point: row.count for row in report.fault_counts}) == Counter(
        {fault: 125 for fault in SchedulerLoadFaultPointV2}
    )
    assert report.total_faults_assigned == 1000
    assert report.total_faults_observed == 1000
    assert report.total_resumes == 1000
    assert report.total_replays == 1000
    assert report.total_attempts == 1000
    assert report.total_stage_runs == 1000
    assert report.total_stage_results == 1000
    assert report.total_attempt_metrics == 1000
    assert report.total_completion_witnesses == 1000
    assert report.total_unexpected_retries == 0
    assert report.total_duplicate_immutable_facts == 0
    assert report.typed_replay_stable is True
    assert report.physical_replay_stable is True
    assert report.satisfies_real_trace_stability is False
    assert report.satisfies_label_quality is False
    assert report.satisfies_concurrency_policy is False
    assert report.satisfies_operations_slo is False
    assert report.authorizes_attestation is False
    assert report.authorizes_production_release is False


def test_report_requires_canonical_fault_rotation() -> None:
    cases = list(_case(index) for index in range(1000))
    original = cases[-1]
    cases[-1] = SchedulerLoadCaseSummaryV2.create(
        private_case_result_ref=original.private_case_result_ref,
        outcome=original.outcome,
        reason_code=original.reason_code,
        assigned_fault_point=SchedulerLoadFaultPointV2.AFTER_JOB_CREATE,
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
    with pytest.raises(ValueError, match="fault distribution"):
        SchedulerLoadReportV2.create(
            policy_ref=_policy().to_ref(),
            private_workload_ref=_private_ref("scheduler-load-workload", "b" * 64),
            private_result_set_ref=_private_ref("scheduler-load-result-set", "c" * 64),
            case_summaries=tuple(cases),
            phase_summaries=_phases(),
            typed_side_effect_count_before_replay=10_000,
            typed_side_effect_count_after_replay=10_000,
            physical_file_count_before_replay=1,
            physical_file_count_after_replay=1,
            physical_bytes_before_replay=1_000_000,
            physical_bytes_after_replay=1_000_000,
            audit=_audit(),
        )


def test_untrustworthy_case_makes_report_incomplete() -> None:
    report = _report(replace_index=0)

    assert report.outcome is SchedulerLoadReportOutcomeV2.INCOMPLETE
    assert len(report.incomplete_case_refs) == 1
    assert len(report.recovered_case_refs) == 999


def test_sqlite_byte_maintenance_is_observational_not_replay_authority() -> None:
    report = _report()
    changed_bytes = SchedulerLoadReportV2.create(
        policy_ref=report.policy_ref,
        private_workload_ref=report.private_workload_ref,
        private_result_set_ref=report.private_result_set_ref,
        case_summaries=report.case_summaries,
        phase_summaries=report.phase_summaries,
        typed_side_effect_count_before_replay=10_000,
        typed_side_effect_count_after_replay=10_000,
        physical_file_count_before_replay=3,
        physical_file_count_after_replay=3,
        physical_bytes_before_replay=10_000_000,
        physical_bytes_after_replay=6_000_000,
        audit=_audit(),
    )

    assert changed_bytes.outcome is SchedulerLoadReportOutcomeV2.COMPLETE
    assert changed_bytes.physical_replay_stable is True
    assert changed_bytes.physical_bytes_before_replay != (changed_bytes.physical_bytes_after_replay)


def test_report_identity_binds_public_case_summaries_and_measurements() -> None:
    report = _report()
    changed_phase = SchedulerLoadPhaseSummaryV2.create(
        phase=SchedulerLoadPhaseV2.SYNTHETIC_INITIAL,
        job_count=1000,
        operation_count=1000,
        elapsed_nanoseconds=2_000_000_000,
    )
    changed = SchedulerLoadReportV2.create(
        policy_ref=report.policy_ref,
        private_workload_ref=report.private_workload_ref,
        private_result_set_ref=report.private_result_set_ref,
        case_summaries=report.case_summaries,
        phase_summaries=(changed_phase, *report.phase_summaries[1:]),
        typed_side_effect_count_before_replay=10_000,
        typed_side_effect_count_after_replay=10_000,
        physical_file_count_before_replay=1,
        physical_file_count_after_replay=1,
        physical_bytes_before_replay=1_000_000,
        physical_bytes_after_replay=1_000_000,
        audit=_audit(),
    )

    assert changed.report_sha256 != report.report_sha256
    assert (
        tuple(scheduler_load_case_summary_v2_ref(value) for value in report.case_summaries)
        == report.case_summary_refs
    )


def test_public_report_is_content_free_and_load_only() -> None:
    serialized = _report().model_dump_json().casefold()

    for forbidden in (
        "job_id",
        "item_id",
        "work_unit_id",
        "lease_id",
        "stage_run_id",
        "source_uri",
        "raw_sha",
        "physical_path",
        "final_output",
        "grader",
        "credential",
        "exception",
    ):
        assert forbidden not in serialized
    assert "scheduler_load_only" in serialized


def test_content_free_gold_binds_deterministic_public_fixture() -> None:
    report = _report()
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    refs = [ref.model_dump(mode="json") for ref in report.case_summary_refs]
    case_set_sha256 = hashlib.sha256(
        json.dumps(
            refs,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    assert payload["policy_sha256"] == _policy().policy_sha256
    assert payload["report_sha256"] == report.report_sha256
    assert payload["case_summary_set_sha256"] == case_set_sha256
    assert payload["unique_job_count"] == 1000
    assert payload["claim_scope"] == "SCHEDULER_LOAD_ONLY"
    assert payload["authority"] == {
        "authorizes_attestation": False,
        "authorizes_production_release": False,
        "satisfies_concurrency_policy": False,
        "satisfies_label_quality": False,
        "satisfies_operations_slo": False,
        "satisfies_real_trace_stability": False,
    }
    serialized = GOLD_PATH.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "job_id",
        "item_id",
        "work_unit_id",
        "lease_id",
        "stage_run_id",
        "source_uri",
        "raw_sha",
        "physical_path",
        "credential",
        "exception",
    ):
        assert forbidden not in serialized
