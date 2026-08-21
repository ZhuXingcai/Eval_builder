from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.cli_v2 import (
    MAX_PIPELINE_PAGE_SIZE,
    PipelineCancellationModeV2,
    PipelineCancellationResultV2,
    PipelineControlConfigV2,
    PipelineEventsResultV2,
    PipelineMetricSliceSummaryV2,
    PipelineMetricsResultV2,
    PipelineObservabilityAvailabilityV2,
    PipelinePlanResultV2,
    PipelineResumeActionV2,
    PipelineResumeResultV2,
    PipelineStatusResultV2,
    PipelineWorkUnitStatusV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.observability_v2 import (
    BatchAuditEventKindV2,
    BatchAuditEventV2,
    BatchAuditOutcomeV2,
    MetricScopeV2,
)
from eval_factory.contracts.orchestration import JobStatus
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    WorkLeaseStateV2,
    WorkReadinessV2,
    WorkRetryDecisionKindV2,
    WorkUnitScopeV2,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-07/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _unit_status() -> PipelineWorkUnitStatusV2:
    return PipelineWorkUnitStatusV2(
        work_unit_ref=_ref("resolved-work-unit", "unit"),
        scope=WorkUnitScopeV2.ITEM,
        stage=StageNameV2.TRACE_INDEX,
        item_id="item://r6-07/item",
        readiness=WorkReadinessV2.READY,
        readiness_ref=_ref("work-readiness-snapshot", "ready"),
        lease_ref=_ref("work-lease", "lease"),
        lease_state=WorkLeaseStateV2.RETRYABLE_FAILURE,
        lease_version=1,
        retry_decision_ref=_ref("work-retry-decision", "retry"),
        retry_decision=WorkRetryDecisionKindV2.RETRY_SCHEDULED,
        attempt_metrics_ref=_ref("work-attempt-metrics", "metrics"),
    )


def _status() -> PipelineStatusResultV2:
    return PipelineStatusResultV2(
        job_id="job://r6-07/job",
        job_status=JobStatus.RUNNING,
        job_version=1,
        resolved_plan_ref=_ref("resolved-dataset-job-plan", "plan"),
        work_graph_ref=_ref("resolved-job-work-graph", "graph"),
        work_control_policy_ref=_ref("work-control-policy", "control"),
        item_count=1,
        work_unit_count=1,
        terminal_attempt_count=1,
        work_unit_offset=0,
        work_unit_limit=100,
        work_units=(_unit_status(),),
        observability_availability=(PipelineObservabilityAvailabilityV2.AVAILABLE),
        audit_report_ref=_ref("batch-audit-report", "report"),
        audit_outcome=BatchAuditOutcomeV2.COMPLETE,
    )


def _event() -> BatchAuditEventV2:
    source_ref = _ref("work-lease-event", "terminal")
    return BatchAuditEventV2.create(
        job_id="job://r6-07/job",
        item_id="item://r6-07/item",
        work_unit_ref=_ref("resolved-work-unit", "unit"),
        stage=StageNameV2.TRACE_INDEX,
        artifact_id=None,
        attempt=1,
        event_kind=BatchAuditEventKindV2.WORK_LEASE_TERMINAL,
        status="SUCCEEDED",
        reason_code=None,
        occurred_at=NOW,
        source_family="WORK_LEASE_EVENT",
        source_version=1,
        source_ref=source_ref,
        outbox_event_id="outbox-event://r6-07/terminal",
        related_refs=(),
        result_refs=(),
        audit=ContractAudit(
            created_at=NOW,
            created_by="r6-07-contract-test",
            governing_versions=(
                VersionBinding(
                    component="batch-observability",
                    version="batch-observability/r6-06-v1",
                ),
            ),
            input_refs=(source_ref,),
        ),
    )


def test_control_config_is_strict_and_matches_retry_shape() -> None:
    value = PipelineControlConfigV2(
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=3,
        retry_delay_seconds=(5, 10),
        retry_lease_expiry=True,
    )

    assert value.max_attempts == 3
    with pytest.raises(ValidationError, match="retry"):
        value.model_copy(
            update={"retry_delay_seconds": (5,)},
        ).__class__.model_validate(
            {
                **value.model_dump(mode="python"),
                "retry_delay_seconds": (5,),
            }
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        PipelineControlConfigV2.model_validate(
            {
                **value.model_dump(mode="python"),
                "provider_token": "forbidden",
            }
        )


def test_plan_and_status_results_bind_exact_refs_and_pagination() -> None:
    unit = _unit_status()
    plan = PipelinePlanResultV2(
        job_id="job://r6-07/job",
        job_spec_ref=_ref("dataset-job-spec", "job"),
        resolved_plan_ref=_ref("resolved-dataset-job-plan", "plan"),
        work_graph_ref=_ref("resolved-job-work-graph", "graph"),
        resolved_stages=(StageNameV2.TRACE_INDEX,),
        work_unit_count=1,
        work_unit_offset=0,
        work_unit_limit=100,
        work_units=(unit,),
    )
    status = _status()

    assert plan.work_units == (unit,)
    assert status.audit_outcome is BatchAuditOutcomeV2.COMPLETE
    with pytest.raises(ValidationError, match="less than or equal"):
        PipelinePlanResultV2.model_validate(
            {
                **plan.model_dump(mode="python"),
                "work_unit_limit": MAX_PIPELINE_PAGE_SIZE + 1,
            }
        )
    with pytest.raises(ValidationError, match="observability"):
        PipelineStatusResultV2.model_validate(
            {
                **status.model_dump(mode="python"),
                "observability_availability": (PipelineObservabilityAvailabilityV2.REFRESH_REQUIRED),
            }
        )


def test_work_status_rejects_partial_lease_and_retry_bindings() -> None:
    value = _unit_status()

    with pytest.raises(ValidationError, match="lease"):
        PipelineWorkUnitStatusV2.model_validate(
            {
                **value.model_dump(mode="python"),
                "lease_ref": None,
            }
        )
    with pytest.raises(ValidationError, match="retry"):
        PipelineWorkUnitStatusV2.model_validate(
            {
                **value.model_dump(mode="python"),
                "retry_decision": None,
            }
        )


def test_resume_and_cancel_results_preserve_closed_truth() -> None:
    resumed = PipelineResumeResultV2(
        action=PipelineResumeActionV2.REOPENED,
        status=_status(),
    )
    cancelled = PipelineCancellationResultV2(
        job_id="job://r6-07/job",
        mode=PipelineCancellationModeV2.COMBINED,
        cancellation_ref=_ref("work-cancellation-record", "cancel"),
        preserved_result_refs=(_ref("stage-result", "result"),),
        model_cancellation_request_refs=(_ref("work-model-cancellation-request", "model"),),
        resource_termination_request_refs=(_ref("work-resource-termination-request", "resource"),),
        pending_physical_acknowledgements=2,
    )

    assert resumed.action is PipelineResumeActionV2.REOPENED
    assert cancelled.pending_physical_acknowledgements == len(
        cancelled.model_cancellation_request_refs
    ) + len(cancelled.resource_termination_request_refs)
    with pytest.raises(ValidationError, match="pending"):
        PipelineCancellationResultV2.model_validate(
            {
                **cancelled.model_dump(mode="python"),
                "pending_physical_acknowledgements": 1,
            }
        )


def test_events_and_metrics_are_bounded_and_content_free() -> None:
    report_ref = _ref("batch-audit-report", "report")
    event = _event()
    events = PipelineEventsResultV2(
        job_id="job://r6-07/job",
        report_ref=report_ref,
        audit_outcome=BatchAuditOutcomeV2.COMPLETE,
        finding_codes=(),
        total_events=1,
        event_offset=0,
        event_limit=100,
        events=(event,),
    )
    metric = PipelineMetricSliceSummaryV2(
        scope=MetricScopeV2.JOB,
        scope_id="job://r6-07/job",
        work_units=1,
        terminal_attempts=1,
        succeeded_attempts=1,
        retryable_attempts=0,
        terminal_non_success_attempts=0,
        cancelled_attempts=0,
        retry_attempts=0,
        resume_attempts=0,
        model_requests=1,
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=15,
        reported_usage_attempts=1,
        conservative_usage_attempts=0,
        cache_hit_attempts=0,
        cache_miss_attempts=1,
        cache_unavailable_attempts=0,
        process_starts=0,
        renderer_operations=0,
        network_requests=0,
        retained_storage_bytes=0,
        tool_calls=0,
        reported_cost_microusd=0,
        unavailable_cost_attempts=1,
        backpressure_events=0,
        error_events=0,
        queue_wait_sample_count=1,
        execution_sample_count=1,
        total_sample_count=1,
    )
    metrics = PipelineMetricsResultV2(
        job_id="job://r6-07/job",
        item_id=None,
        report_ref=report_ref,
        audit_outcome=BatchAuditOutcomeV2.COMPLETE,
        finding_codes=(),
        total_attempts=1,
        attempt_offset=0,
        attempt_limit=100,
        attempt_metrics_refs=(_ref("work-attempt-metrics", "attempt"),),
        metric_slices=(metric,),
    )

    assert events.events == (event,)
    assert metrics.metric_slices[0].queue_wait_sample_count == 1
    serialized = metrics.model_dump_json()
    for forbidden in (
        "queue_wait_samples_us",
        "prompt",
        "credential",
        "private_reference",
        "grader_rule",
        "final_answer",
    ):
        assert forbidden not in serialized
    with pytest.raises(ValidationError, match="Extra inputs"):
        PipelineMetricsResultV2.model_validate(
            {
                **metrics.model_dump(mode="python"),
                "raw_trace": "forbidden",
            }
        )
