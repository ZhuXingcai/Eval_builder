from __future__ import annotations

from datetime import UTC, datetime

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
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

NOW = datetime(2026, 8, 3, tzinfo=UTC)


def ref(object_type: str, digest: str, *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version=version,
        object_sha256=digest,
    )


def audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-06-test-fixture",
        governing_versions=(
            VersionBinding(
                component="concurrency-experiment",
                version="concurrency-experiment/r8-06-v1",
            ),
        ),
    )


def policy() -> ConcurrencyExperimentPolicyV2:
    return ConcurrencyExperimentPolicyV2.create(
        canary_regression_report_ref=ref(
            "canary-regression-report",
            "375b519dc3b40d3c8aab09d9a04574deca73ec80138c2fcaa2f1714eb1edbd4f",
        ),
        real_trace_stability_report_ref=ref(
            "real-trace-stability-report",
            "730f151435d2472cfe456e9df350c2663e8702951bd1eb1e8f0dcb45b315374f",
        ),
        scheduler_load_report_ref=ref(
            "scheduler-load-report",
            "52061b69b7b418874709bddc71450111680e7931f4e04a337103b34b9f379bac",
        ),
        cohort_manifest_ref=ObjectRef(
            object_type="development-canary-manifest",
            object_id="development-canary-manifest://v4",
            object_version="v4",
            object_sha256="1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6",
        ),
        max_capacity_readmissions=24,
        max_private_bytes=1_000_000_000,
        max_report_bytes=100_000_000,
        audit=audit(),
    )


def percentile(
    *,
    sample_count: int,
    p50: int,
    p95: int,
    unit: ConcurrencyExperimentUnitV2,
) -> ConcurrencyExperimentPercentileSummaryV2:
    return ConcurrencyExperimentPercentileSummaryV2(
        sample_count=sample_count,
        minimum=min(p50, p95),
        p50=p50,
        p95=p95,
        maximum=max(p50, p95),
        unit=unit,
    )


def level(
    worker_count: int,
    *,
    throughput: int,
    case_p95: int,
) -> ConcurrencyExperimentLevelSummaryV2:
    return ConcurrencyExperimentLevelSummaryV2.create(
        worker_count=worker_count,
        trial_count=3,
        unique_case_count=24,
        executed_case_count=72,
        succeeded_case_count=72,
        product_error_count=0,
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
        retained_storage_bytes=1_800,
        tool_call_count=72,
        unavailable_cost_attempt_count=720,
        elapsed_nanoseconds=percentile(
            sample_count=3,
            p50=10,
            p95=12,
            unit=ConcurrencyExperimentUnitV2.NANOSECONDS,
        ),
        throughput_millijobs_per_second=percentile(
            sample_count=3,
            p50=throughput,
            p95=throughput,
            unit=ConcurrencyExperimentUnitV2.MILLIJOBS_PER_SECOND,
        ),
        case_dispatch_wait_nanoseconds=percentile(
            sample_count=72,
            p50=1,
            p95=2,
            unit=ConcurrencyExperimentUnitV2.NANOSECONDS,
        ),
        case_wall_latency_nanoseconds=percentile(
            sample_count=72,
            p50=case_p95 * 3 // 4,
            p95=case_p95,
            unit=ConcurrencyExperimentUnitV2.NANOSECONDS,
        ),
        queue_wait_microseconds=percentile(
            sample_count=720,
            p50=1,
            p95=2,
            unit=ConcurrencyExperimentUnitV2.MICROSECONDS,
        ),
        execution_microseconds=percentile(
            sample_count=720,
            p50=1,
            p95=2,
            unit=ConcurrencyExperimentUnitV2.MICROSECONDS,
        ),
        attempt_active_total_microseconds=percentile(
            sample_count=720,
            p50=2,
            p95=3,
            unit=ConcurrencyExperimentUnitV2.MICROSECONDS,
        ),
        outcome=ConcurrencyExperimentLevelOutcomeV2.ELIGIBLE,
        reason_codes=(ConcurrencyExperimentReasonCodeV2.NONE,),
    )


def levels() -> tuple[ConcurrencyExperimentLevelSummaryV2, ...]:
    return (
        level(1, throughput=100_000, case_p95=100),
        level(2, throughput=180_000, case_p95=110),
        level(4, throughput=195_000, case_p95=120),
        level(8, throughput=200_000, case_p95=140),
        level(16, throughput=199_000, case_p95=170),
    )


def recovery(worker_count: int = 2) -> ConcurrencyExperimentRecoverySummaryV2:
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


def report() -> ConcurrencyExperimentReportV2:
    experiment_policy = policy()
    rows = levels()
    selected = select_balanced_worker_level(rows, policy=experiment_policy)
    assert selected is not None
    recovery_summary = recovery(selected.worker_count)
    recommendation = NonModelResourceRecommendationV2.create(
        selected_level=selected,
        level_summaries=rows,
        recovery_summary=recovery_summary,
        recommended_storage_bytes_per_job=25,
        recommended_storage_bytes_per_cohort=600,
        policy=experiment_policy,
        audit=audit(),
    )
    return ConcurrencyExperimentReportV2.create(
        policy_ref=experiment_policy.to_ref(),
        host_summary=ConcurrencyExperimentHostSummaryV2.create(
            platform_family="DARWIN",
            architecture="ARM64",
            logical_cpu_count=10,
            physical_cpu_count=10,
            memory_bytes=17_179_869_184,
            python_version="3.12.13",
            audit=audit(),
        ),
        cohort_manifest_ref=experiment_policy.cohort_manifest_ref,
        template_ref=ref(
            "canary-regression-template",
            "4".rjust(64, "0"),
            version="private-v1",
        ),
        private_workload_ref=ref(
            "concurrency-experiment-workload",
            "5".rjust(64, "0"),
            version="private-v1",
        ),
        private_result_set_ref=ref(
            "concurrency-experiment-result-set",
            "6".rjust(64, "0"),
            version="private-v1",
        ),
        level_summaries=rows,
        recovery_summary=recovery_summary,
        recommendation=recommendation,
        audit=audit(),
    )
