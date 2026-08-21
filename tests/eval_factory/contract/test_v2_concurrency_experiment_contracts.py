from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.concurrency_experiment_v2 import (
    CONCURRENCY_EXPERIMENT_CLAIM_SCOPE,
    CONCURRENCY_EXPERIMENT_LEVELS,
    ConcurrencyExperimentFaultPointV2,
    ConcurrencyExperimentHostSummaryV2,
    ConcurrencyExperimentLevelOutcomeV2,
    ConcurrencyExperimentLevelSummaryV2,
    ConcurrencyExperimentMetricAvailabilityV2,
    ConcurrencyExperimentOutcomeV2,
    ConcurrencyExperimentPercentileSummaryV2,
    ConcurrencyExperimentPolicyV2,
    ConcurrencyExperimentReasonCodeV2,
    ConcurrencyExperimentRecommendationReasonV2,
    ConcurrencyExperimentRecoverySummaryV2,
    ConcurrencyExperimentReportV2,
    ConcurrencyExperimentUnitV2,
    NonModelResourceRecommendationV2,
    select_balanced_worker_level,
    validate_concurrency_experiment_policy_v2_identity,
    validate_concurrency_experiment_report_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

NOW = datetime(2026, 8, 3, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _ref(object_type: str, suffix: str, *, version: str = "v2") -> ObjectRef:
    digest = suffix.rjust(64, "0")
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-06-contract-test",
        governing_versions=(
            VersionBinding(
                component="concurrency-experiment",
                version="concurrency-experiment/r8-06-v1",
            ),
        ),
        input_refs=tuple(sorted(refs, key=lambda value: value.object_id)),
    )


def _predecessors() -> tuple[ObjectRef, ObjectRef, ObjectRef, ObjectRef]:
    return (
        _ref(
            "canary-regression-report",
            "375b519dc3b40d3c8aab09d9a04574deca73ec80138c2fcaa2f1714eb1edbd4f",
        ),
        _ref(
            "real-trace-stability-report",
            "730f151435d2472cfe456e9df350c2663e8702951bd1eb1e8f0dcb45b315374f",
        ),
        _ref(
            "scheduler-load-report",
            "52061b69b7b418874709bddc71450111680e7931f4e04a337103b34b9f379bac",
        ),
        ObjectRef(
            object_type="development-canary-manifest",
            object_id="development-canary-manifest://v4",
            object_version="v4",
            object_sha256="1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6",
        ),
    )


def _policy() -> ConcurrencyExperimentPolicyV2:
    r8_03, r8_04, r8_05, cohort = _predecessors()
    return ConcurrencyExperimentPolicyV2.create(
        canary_regression_report_ref=r8_03,
        real_trace_stability_report_ref=r8_04,
        scheduler_load_report_ref=r8_05,
        cohort_manifest_ref=cohort,
        max_capacity_readmissions=24,
        max_private_bytes=1_000_000_000,
        max_report_bytes=100_000_000,
        audit=_audit(),
    )


def _percentiles(
    values: tuple[int, ...],
    *,
    unit: ConcurrencyExperimentUnitV2 = ConcurrencyExperimentUnitV2.NANOSECONDS,
) -> ConcurrencyExperimentPercentileSummaryV2:
    return ConcurrencyExperimentPercentileSummaryV2.create(
        samples=values,
        unit=unit,
    )


def _level(
    worker_count: int,
    *,
    throughput: tuple[int, int, int],
    case_p95_ns: int,
    outcome: ConcurrencyExperimentLevelOutcomeV2 = ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE,
    reason_code: ConcurrencyExperimentReasonCodeV2 = ConcurrencyExperimentReasonCodeV2.NONE,
) -> ConcurrencyExperimentLevelSummaryV2:
    return ConcurrencyExperimentLevelSummaryV2.create(
        worker_count=worker_count,
        trial_count=3,
        unique_case_count=24,
        executed_case_count=72,
        succeeded_case_count=(72 if outcome is ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE else 71),
        product_error_count=(0 if outcome is ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE else 1),
        infrastructure_error_count=0,
        incomplete_case_count=0,
        attempt_count=720,
        attempt_metric_count=720,
        stage_run_count=648,
        stage_result_count=648,
        provider_invocation_count=72,
        retry_count=0,
        resume_count=0,
        model_request_count=0,
        input_token_count=0,
        output_token_count=0,
        process_start_count=0,
        renderer_operation_count=0,
        network_request_count=0,
        retained_storage_bytes=72 * 25,
        tool_call_count=72,
        unavailable_cost_attempt_count=720,
        elapsed_nanoseconds=_percentiles((10, 11, 12)),
        throughput_millijobs_per_second=_percentiles(
            throughput,
            unit=ConcurrencyExperimentUnitV2.MILLIJOBS_PER_SECOND,
        ),
        case_dispatch_wait_nanoseconds=_percentiles(tuple(range(1, 73))),
        case_wall_latency_nanoseconds=ConcurrencyExperimentPercentileSummaryV2(
            sample_count=72,
            minimum=case_p95_ns // 2,
            p50=case_p95_ns * 3 // 4,
            p95=case_p95_ns,
            maximum=case_p95_ns,
            unit=ConcurrencyExperimentUnitV2.NANOSECONDS,
        ),
        queue_wait_microseconds=_percentiles(
            tuple(range(1, 721)),
            unit=ConcurrencyExperimentUnitV2.MICROSECONDS,
        ),
        execution_microseconds=_percentiles(
            tuple(range(1, 721)),
            unit=ConcurrencyExperimentUnitV2.MICROSECONDS,
        ),
        attempt_active_total_microseconds=_percentiles(
            tuple(range(1, 721)),
            unit=ConcurrencyExperimentUnitV2.MICROSECONDS,
        ),
        outcome=outcome,
        reason_codes=(reason_code,),
    )


def _eligible_levels() -> tuple[ConcurrencyExperimentLevelSummaryV2, ...]:
    return (
        _level(1, throughput=(99_000, 100_000, 101_000), case_p95_ns=100),
        _level(2, throughput=(179_000, 180_000, 181_000), case_p95_ns=110),
        _level(4, throughput=(194_000, 195_000, 196_000), case_p95_ns=120),
        _level(8, throughput=(198_000, 200_000, 202_000), case_p95_ns=140),
        _level(16, throughput=(197_000, 199_000, 201_000), case_p95_ns=170),
    )


def _recovery(worker_count: int = 2) -> ConcurrencyExperimentRecoverySummaryV2:
    return ConcurrencyExperimentRecoverySummaryV2.create(
        selected_worker_count=worker_count,
        fault_counts={point: 6 for point in ConcurrencyExperimentFaultPointV2},
        fault_observed_count=24,
        resume_success_count=24,
        resume_failure_count=0,
        unexpected_retry_count=0,
        duplicate_provider_fact_count=0,
        provider_invocation_count=24,
        typed_side_effect_count_before_replay=1_000,
        typed_side_effect_count_after_replay=1_000,
        physical_file_count_before_replay=3,
        physical_file_count_after_replay=3,
        exact_replay_stable=True,
    )


def _recommendation(
    levels: tuple[ConcurrencyExperimentLevelSummaryV2, ...],
) -> NonModelResourceRecommendationV2:
    selected = select_balanced_worker_level(levels, policy=_policy())
    assert selected is not None
    return NonModelResourceRecommendationV2.create(
        selected_level=selected,
        level_summaries=levels,
        recovery_summary=_recovery(selected.worker_count),
        recommended_storage_bytes_per_job=25,
        recommended_storage_bytes_per_cohort=600,
        policy=_policy(),
        audit=_audit(),
    )


def _report() -> ConcurrencyExperimentReportV2:
    policy = _policy()
    host = ConcurrencyExperimentHostSummaryV2.create(
        platform_family="DARWIN",
        architecture="ARM64",
        logical_cpu_count=10,
        physical_cpu_count=10,
        memory_bytes=17_179_869_184,
        python_version="3.12.13",
        audit=_audit(),
    )
    levels = _eligible_levels()
    recovery = _recovery()
    recommendation = _recommendation(levels)
    return ConcurrencyExperimentReportV2.create(
        policy_ref=policy.to_ref(),
        host_summary=host,
        cohort_manifest_ref=policy.cohort_manifest_ref,
        template_ref=_ref(
            "canary-regression-template",
            "4",
            version="private-v1",
        ),
        private_workload_ref=_ref(
            "concurrency-experiment-workload",
            "5",
            version="private-v1",
        ),
        private_result_set_ref=_ref(
            "concurrency-experiment-result-set",
            "6",
            version="private-v1",
        ),
        level_summaries=levels,
        recovery_summary=recovery,
        recommendation=recommendation,
        audit=_audit(),
    )


def test_policy_binds_exact_non_model_experiment() -> None:
    policy = _policy()

    validate_concurrency_experiment_policy_v2_identity(policy)
    assert policy.worker_levels == CONCURRENCY_EXPERIMENT_LEVELS
    assert policy.trial_count_per_level == 3
    assert policy.case_count_per_trial == 24
    assert policy.recovery_probe_count == 24
    assert policy.throughput_efficiency_basis_points == 9_000
    assert policy.p95_baseline_limit_basis_points == 15_000
    assert policy.fault_points == tuple(ConcurrencyExperimentFaultPointV2)


def test_contracts_are_strict_and_policy_literals_reject_drift() -> None:
    payload = _policy().model_dump(mode="python")
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        ConcurrencyExperimentPolicyV2.model_validate(payload)

    for field, value in (
        ("worker_levels", (1, 2, 4, 8)),
        ("trial_count_per_level", 2),
        ("case_count_per_trial", 23),
        ("throughput_efficiency_basis_points", 8_999),
        ("p95_baseline_limit_basis_points", 14_999),
    ):
        payload = _policy().model_dump(mode="python")
        payload[field] = value
        with pytest.raises(ValidationError):
            ConcurrencyExperimentPolicyV2.model_validate(payload)


@pytest.mark.parametrize(
    ("samples", "p50", "p95"),
    (
        ((7,), 7, 7),
        ((1, 2), 1, 2),
        ((1, 2, 3), 2, 3),
        ((1, 2, 3, 4), 2, 4),
        (tuple(range(1, 21)), 10, 19),
    ),
)
def test_percentiles_use_integer_nearest_rank(
    samples: tuple[int, ...],
    p50: int,
    p95: int,
) -> None:
    summary = _percentiles(samples)

    assert summary.minimum == min(samples)
    assert summary.p50 == p50
    assert summary.p95 == p95
    assert summary.maximum == max(samples)


def test_percentiles_represent_empty_samples_without_fabricating_values() -> None:
    empty = _percentiles(())

    assert empty.sample_count == 0
    assert (empty.minimum, empty.p50, empty.p95, empty.maximum) == (0, 0, 0, 0)
    payload = empty.model_dump(mode="python")
    payload["p95"] = 1
    with pytest.raises(ValidationError, match="empty percentile"):
        ConcurrencyExperimentPercentileSummaryV2.model_validate(payload)

    payload = _percentiles((1, 2, 3)).model_dump(mode="python")
    payload["p95"] = 0
    with pytest.raises(ValidationError):
        ConcurrencyExperimentPercentileSummaryV2.model_validate(payload)


def test_level_summary_has_exact_counts_and_unavailable_semantics() -> None:
    level = _eligible_levels()[0]

    assert level.outcome is ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE
    assert level.executed_case_count == 72
    assert level.error_rate_basis_points == 0
    assert level.attempt_count == 720
    assert level.attempt_metric_count == 720
    assert level.model_availability is ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE
    assert level.cache_availability is ConcurrencyExperimentMetricAvailabilityV2.NOT_APPLICABLE
    assert level.cost_availability is ConcurrencyExperimentMetricAvailabilityV2.UNAVAILABLE
    assert level.model_request_count == 0
    assert level.input_token_count == 0
    assert level.output_token_count == 0


def test_eligible_level_rejects_inexact_r6_evidence_counts() -> None:
    level = _eligible_levels()[0]
    mutations = (
        {
            "attempt_count": 719,
            "attempt_metric_count": 719,
            "unavailable_cost_attempt_count": 719,
        },
        {"stage_run_count": 647},
        {"stage_result_count": 647},
        {"provider_invocation_count": 71},
    )

    for mutation in mutations:
        payload = level.model_dump(mode="python")
        payload.update(mutation)
        if "attempt_metric_count" in mutation:
            for field in (
                "queue_wait_microseconds",
                "execution_microseconds",
                "attempt_active_total_microseconds",
            ):
                payload[field]["sample_count"] = 719
        with pytest.raises(ValidationError, match="eligible level evidence"):
            ConcurrencyExperimentLevelSummaryV2.model_validate(payload)


def test_balanced_knee_selects_lowest_level_at_efficiency_and_latency_gate() -> None:
    levels = _eligible_levels()

    selected = select_balanced_worker_level(levels, policy=_policy())

    assert selected is not None
    assert selected.worker_count == 2


def test_recovery_requires_exact_six_by_four_closure() -> None:
    recovery = _recovery()

    assert recovery.probe_count == 24
    assert recovery.recovery_stable is True
    assert all(row.count == 6 for row in recovery.fault_counts)

    payload = recovery.model_dump(mode="python")
    payload["resume_success_count"] = 23
    with pytest.raises(ValidationError):
        ConcurrencyExperimentRecoverySummaryV2.model_validate(payload)


def test_recovery_stability_requires_one_provider_invocation_per_probe() -> None:
    payload = _recovery().model_dump(mode="python")
    payload["provider_invocation_count"] = 23

    with pytest.raises(ValidationError, match="recovery stability"):
        ConcurrencyExperimentRecoverySummaryV2.model_validate(payload)


def test_report_is_scoped_and_cannot_grant_downstream_authority() -> None:
    report = _report()

    validate_concurrency_experiment_report_v2_identity(report)
    assert report.outcome is ConcurrencyExperimentOutcomeV2.NON_MODEL_RECOMMENDED
    assert report.claim_scope == CONCURRENCY_EXPERIMENT_CLAIM_SCOPE
    assert report.recommendation is not None
    assert report.recommendation.recommended_worker_count == 2
    assert report.satisfies_sc_012 is False
    assert report.satisfies_model_resource_policy is False
    assert report.satisfies_operations_slo is False
    assert report.authorizes_safety_or_privacy_approval is False
    assert report.authorizes_attestation is False
    assert report.authorizes_production_release is False


def test_report_binds_recommendation_and_recovery_to_level_evidence() -> None:
    report = _report()
    levels = _eligible_levels()
    alternate_levels = (
        levels[0],
        _level(
            2,
            throughput=(179_000, 180_000, 181_000),
            case_p95_ns=160,
        ),
        *levels[2:],
    )
    alternate_selected = select_balanced_worker_level(
        alternate_levels,
        policy=_policy(),
    )
    assert alternate_selected is not None
    assert alternate_selected.worker_count == 4
    alternate_recommendation = NonModelResourceRecommendationV2.create(
        selected_level=alternate_selected,
        level_summaries=alternate_levels,
        recovery_summary=_recovery(4),
        recommended_storage_bytes_per_job=25,
        recommended_storage_bytes_per_cohort=600,
        policy=_policy(),
        audit=_audit(),
    )

    with pytest.raises(ValidationError, match="balanced level evidence"):
        ConcurrencyExperimentReportV2.create(
            policy_ref=report.policy_ref,
            host_summary=report.host_summary,
            cohort_manifest_ref=report.cohort_manifest_ref,
            template_ref=report.template_ref,
            private_workload_ref=report.private_workload_ref,
            private_result_set_ref=report.private_result_set_ref,
            level_summaries=levels,
            recovery_summary=_recovery(4),
            recommendation=alternate_recommendation,
            audit=_audit(),
        )

    with pytest.raises(ValidationError, match="recovery uses another worker"):
        ConcurrencyExperimentReportV2.create(
            policy_ref=report.policy_ref,
            host_summary=report.host_summary,
            cohort_manifest_ref=report.cohort_manifest_ref,
            template_ref=report.template_ref,
            private_workload_ref=report.private_workload_ref,
            private_result_set_ref=report.private_result_set_ref,
            level_summaries=levels,
            recovery_summary=_recovery(4),
            recommendation=report.recommendation,
            audit=_audit(),
        )


def test_report_rejects_pending_host_summary_identity() -> None:
    report = _report()
    pending_host = report.host_summary.model_copy(update={"host_summary_sha256": "0" * 64})

    with pytest.raises(ValidationError, match="host summary identity"):
        ConcurrencyExperimentReportV2.create(
            policy_ref=report.policy_ref,
            host_summary=pending_host,
            cohort_manifest_ref=report.cohort_manifest_ref,
            template_ref=report.template_ref,
            private_workload_ref=report.private_workload_ref,
            private_result_set_ref=report.private_result_set_ref,
            level_summaries=report.level_summaries,
            recovery_summary=report.recovery_summary,
            recommendation=report.recommendation,
            audit=_audit(),
        )


def test_report_without_recommendation_is_pending() -> None:
    report = _report()
    pending = ConcurrencyExperimentReportV2.create(
        policy_ref=report.policy_ref,
        host_summary=report.host_summary,
        cohort_manifest_ref=report.cohort_manifest_ref,
        template_ref=report.template_ref,
        private_workload_ref=report.private_workload_ref,
        private_result_set_ref=report.private_result_set_ref,
        level_summaries=tuple(
            level.model_copy(
                update={
                    "outcome": ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_ERROR,
                    "reason_codes": (ConcurrencyExperimentReasonCodeV2.CASE_FAILED,),
                }
            )
            for level in report.level_summaries
        ),
        recovery_summary=None,
        recommendation=None,
        audit=_audit(),
    )

    assert pending.outcome is ConcurrencyExperimentOutcomeV2.RECOMMENDATION_PENDING
    assert pending.recommendation is None


def test_pending_report_names_both_balanced_threshold_failures() -> None:
    report = _report()
    levels = (
        _level(1, throughput=(74, 75, 84), case_p95_ns=23_620_108_084),
        _level(2, throughput=(87, 91, 92), case_p95_ns=45_556_228_125),
        *(
            _level(
                worker_count,
                throughput=throughput,
                case_p95_ns=case_p95_ns,
                outcome=ConcurrencyExperimentLevelOutcomeV2.INELIGIBLE_ERROR,
                reason_code=ConcurrencyExperimentReasonCodeV2.CASE_FAILED,
            )
            for worker_count, throughput, case_p95_ns in (
                (4, (88, 92, 94), 78_890_391_375),
                (8, (87, 90, 93), 118_744_168_542),
                (16, (103, 111, 114), 184_008_397_834),
            )
        ),
    )

    pending = ConcurrencyExperimentReportV2.create(
        policy_ref=report.policy_ref,
        host_summary=report.host_summary,
        cohort_manifest_ref=report.cohort_manifest_ref,
        template_ref=report.template_ref,
        private_workload_ref=report.private_workload_ref,
        private_result_set_ref=report.private_result_set_ref,
        level_summaries=levels,
        recovery_summary=None,
        recommendation=None,
        audit=_audit(),
    )

    assert pending.reason_codes == (
        ConcurrencyExperimentRecommendationReasonV2.THROUGHPUT_THRESHOLD_NOT_MET,
        ConcurrencyExperimentRecommendationReasonV2.LATENCY_THRESHOLD_NOT_MET,
        ConcurrencyExperimentRecommendationReasonV2.MODEL_EVIDENCE_PENDING,
    )


def test_public_report_excludes_private_and_content_fields() -> None:
    serialized = _report().model_dump_json().casefold()

    for forbidden in (
        "instance_id",
        "job_id",
        "item_id",
        "work_unit_id",
        "lease_id",
        "stage_run_id",
        "source_uri",
        "raw_sha",
        "hostname",
        "physical_path",
        "final_output",
        "grader",
        "credential",
        "exception",
        "prompt_body",
        "rubric_body",
    ):
        assert forbidden not in serialized


def test_approved_policy_and_content_free_gold_are_current() -> None:
    approved = ConcurrencyExperimentPolicyV2.model_validate_json(
        (REPO_ROOT / "evals/manifests/r8-06-concurrency-experiment-policy-v1.json").read_bytes()
    )
    assert approved.policy_sha256 == _policy().policy_sha256

    gold_path = REPO_ROOT / "evals/golden/eval_factory/readiness/r8-06-concurrency-experiment-v1.json"
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    assert payload["claim_scope"] == "CANARY_WORKER_STORAGE_ONLY"
    assert payload["authority"] == {
        "satisfies_sc_012": False,
        "satisfies_model_resource_policy": False,
        "satisfies_operations_slo": False,
        "authorizes_safety_or_privacy_approval": False,
        "authorizes_attestation": False,
        "authorizes_production_release": False,
    }
    serialized = gold_path.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "instance_id",
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
        "final_output",
        "grader_rule",
        "hidden_condition",
    ):
        assert forbidden not in serialized
