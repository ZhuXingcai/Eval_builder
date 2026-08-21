from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.observability_v2 import (
    BatchAuditEventV2,
    BatchAuditFindingCodeV2,
    BatchAuditOutcomeV2,
    MetricScopeV2,
    MetricSliceV2,
)
from eval_factory.contracts.orchestration import JobStatus
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    WorkLeaseStateV2,
    WorkReadinessV2,
    WorkRetryDecisionKindV2,
    WorkUnitScopeV2,
)

PIPELINE_CLI_POLICY_VERSION: Literal["pipeline-cli/r6-07-v1"] = "pipeline-cli/r6-07-v1"
MAX_PIPELINE_PAGE_SIZE = 500


class PipelineObservabilityAvailabilityV2(StrEnum):
    AVAILABLE = "AVAILABLE"
    REFRESH_REQUIRED = "REFRESH_REQUIRED"


class PipelineResumeActionV2(StrEnum):
    STARTED = "STARTED"
    REOPENED = "REOPENED"
    ALREADY_RUNNING = "ALREADY_RUNNING"


class PipelineCancellationModeV2(StrEnum):
    WORK = "WORK"
    RESOURCE = "RESOURCE"
    MODEL = "MODEL"
    COMBINED = "COMBINED"


class PipelineControlConfigV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-control-config/v2"] = (
        "eval-factory/pipeline-control-config/v2"
    )
    lease_duration_seconds: int = Field(ge=1)
    heartbeat_extension_seconds: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    retry_delay_seconds: tuple[int, ...]
    retry_lease_expiry: bool
    policy_version: Literal["pipeline-cli/r6-07-v1"] = PIPELINE_CLI_POLICY_VERSION

    @model_validator(mode="after")
    def validate_retry_shape(self) -> Self:
        if len(self.retry_delay_seconds) != self.max_attempts - 1:
            raise ValueError("retry_delay_seconds must contain one delay per retry")
        if any(value < 0 for value in self.retry_delay_seconds):
            raise ValueError("retry delays must be non-negative")
        return self


class PipelineWorkUnitStatusV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-work-unit-status/v2"] = (
        "eval-factory/pipeline-work-unit-status/v2"
    )
    work_unit_ref: ObjectRef
    scope: WorkUnitScopeV2
    stage: StageNameV2
    item_id: Identifier | None = None
    readiness: WorkReadinessV2 | None = None
    readiness_ref: ObjectRef | None = None
    lease_ref: ObjectRef | None = None
    lease_state: WorkLeaseStateV2 | None = None
    lease_version: int | None = Field(default=None, ge=0)
    retry_decision_ref: ObjectRef | None = None
    retry_decision: WorkRetryDecisionKindV2 | None = None
    attempt_metrics_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        _require_ref(
            self.work_unit_ref,
            "resolved-work-unit",
            "v2",
            "work_unit_ref",
        )
        item_scope = self.scope is not WorkUnitScopeV2.JOB
        if item_scope != (self.item_id is not None):
            raise ValueError("work-unit scope and item_id disagree")
        if (self.readiness is None) != (self.readiness_ref is None):
            raise ValueError("readiness and readiness_ref must be paired")
        if self.readiness_ref is not None:
            _require_ref(
                self.readiness_ref,
                "work-readiness-snapshot",
                "v2",
                "readiness_ref",
            )
        lease_values = (
            self.lease_ref,
            self.lease_state,
            self.lease_version,
        )
        if any(value is None for value in lease_values) and any(value is not None for value in lease_values):
            raise ValueError("lease ref, state, and version must be paired")
        if self.lease_ref is not None:
            _require_ref(
                self.lease_ref,
                "work-lease",
                "v2",
                "lease_ref",
            )
        if (self.retry_decision_ref is None) != (self.retry_decision is None):
            raise ValueError("retry decision and ref must be paired")
        if self.retry_decision_ref is not None:
            _require_ref(
                self.retry_decision_ref,
                "work-retry-decision",
                "v2",
                "retry_decision_ref",
            )
        if self.attempt_metrics_ref is not None:
            _require_ref(
                self.attempt_metrics_ref,
                "work-attempt-metrics",
                "v2",
                "attempt_metrics_ref",
            )
        return self


class PipelinePlanResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-plan-result/v2"] = "eval-factory/pipeline-plan-result/v2"
    job_id: Identifier
    job_spec_ref: ObjectRef
    resolved_plan_ref: ObjectRef
    work_graph_ref: ObjectRef
    resolved_stages: tuple[StageNameV2, ...]
    work_unit_count: int = Field(ge=0)
    work_unit_offset: int = Field(ge=0)
    work_unit_limit: int = Field(
        ge=1,
        le=MAX_PIPELINE_PAGE_SIZE,
    )
    work_units: tuple[PipelineWorkUnitStatusV2, ...]

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        _require_ref(
            self.job_spec_ref,
            "dataset-job-spec",
            "v2",
            "job_spec_ref",
        )
        _require_ref(
            self.resolved_plan_ref,
            "resolved-dataset-job-plan",
            "v2",
            "resolved_plan_ref",
        )
        _require_ref(
            self.work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "work_graph_ref",
        )
        _require_page(
            "work-unit page",
            total=self.work_unit_count,
            offset=self.work_unit_offset,
            limit=self.work_unit_limit,
            returned=len(self.work_units),
        )
        return self


class PipelineCreateResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-create-result/v2"] = (
        "eval-factory/pipeline-create-result/v2"
    )
    job_id: Identifier
    job_status: JobStatus
    job_version: int = Field(ge=0)
    resolved_plan_ref: ObjectRef
    work_graph_ref: ObjectRef
    work_control_policy_ref: ObjectRef
    item_count: int = Field(ge=0)
    work_unit_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_create(self) -> Self:
        if self.job_status is not JobStatus.CREATED:
            raise ValueError("pipeline create result must leave the Job CREATED")
        _require_pipeline_refs(
            self.resolved_plan_ref,
            self.work_graph_ref,
            self.work_control_policy_ref,
        )
        return self


class PipelineStatusResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-status-result/v2"] = (
        "eval-factory/pipeline-status-result/v2"
    )
    job_id: Identifier
    job_status: JobStatus
    job_version: int = Field(ge=0)
    resolved_plan_ref: ObjectRef
    work_graph_ref: ObjectRef
    work_control_policy_ref: ObjectRef
    item_count: int = Field(ge=0)
    work_unit_count: int = Field(ge=0)
    terminal_attempt_count: int = Field(ge=0)
    work_unit_offset: int = Field(ge=0)
    work_unit_limit: int = Field(
        ge=1,
        le=MAX_PIPELINE_PAGE_SIZE,
    )
    work_units: tuple[PipelineWorkUnitStatusV2, ...]
    observability_availability: PipelineObservabilityAvailabilityV2
    audit_report_ref: ObjectRef | None = None
    audit_outcome: BatchAuditOutcomeV2 | None = None

    @model_validator(mode="after")
    def validate_pipeline_status(self) -> Self:
        _require_pipeline_refs(
            self.resolved_plan_ref,
            self.work_graph_ref,
            self.work_control_policy_ref,
        )
        _require_page(
            "work-unit page",
            total=self.work_unit_count,
            offset=self.work_unit_offset,
            limit=self.work_unit_limit,
            returned=len(self.work_units),
        )
        available = self.observability_availability is PipelineObservabilityAvailabilityV2.AVAILABLE
        if available != (self.audit_report_ref is not None):
            raise ValueError("observability availability and report ref disagree")
        if available != (self.audit_outcome is not None):
            raise ValueError("observability availability and outcome disagree")
        if self.audit_report_ref is not None:
            _require_ref(
                self.audit_report_ref,
                "batch-audit-report",
                "v2",
                "audit_report_ref",
            )
        return self


class PipelineResumeResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-resume-result/v2"] = (
        "eval-factory/pipeline-resume-result/v2"
    )
    action: PipelineResumeActionV2
    status: PipelineStatusResultV2

    @model_validator(mode="after")
    def validate_resume(self) -> Self:
        if self.status.job_status is not JobStatus.RUNNING:
            raise ValueError("pipeline resume result must describe a RUNNING Job")
        return self


class PipelineCancellationResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-cancellation-result/v2"] = (
        "eval-factory/pipeline-cancellation-result/v2"
    )
    job_id: Identifier
    mode: PipelineCancellationModeV2
    cancellation_ref: ObjectRef
    preserved_result_refs: tuple[ObjectRef, ...] = ()
    model_cancellation_request_refs: tuple[ObjectRef, ...] = ()
    resource_termination_request_refs: tuple[ObjectRef, ...] = ()
    pending_physical_acknowledgements: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_cancellation(self) -> Self:
        _require_ref(
            self.cancellation_ref,
            "work-cancellation-record",
            "v2",
            "cancellation_ref",
        )
        _require_sorted_unique_refs(
            "preserved_result_refs",
            self.preserved_result_refs,
        )
        _require_refs(
            self.model_cancellation_request_refs,
            "work-model-cancellation-request",
            "v2",
            "model_cancellation_request_refs",
        )
        _require_refs(
            self.resource_termination_request_refs,
            "work-resource-termination-request",
            "v2",
            "resource_termination_request_refs",
        )
        expected_pending = len(self.model_cancellation_request_refs) + len(
            self.resource_termination_request_refs
        )
        if self.pending_physical_acknowledgements != expected_pending:
            raise ValueError("pending physical acknowledgements are not exact")
        if self.mode is PipelineCancellationModeV2.WORK and expected_pending:
            raise ValueError("work-only cancellation cannot carry physical requests")
        if self.mode is PipelineCancellationModeV2.MODEL and self.resource_termination_request_refs:
            raise ValueError("model cancellation cannot carry resource requests")
        if self.mode is PipelineCancellationModeV2.RESOURCE and self.model_cancellation_request_refs:
            raise ValueError("resource cancellation cannot carry model requests")
        return self


class PipelineEventsResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-events-result/v2"] = (
        "eval-factory/pipeline-events-result/v2"
    )
    job_id: Identifier
    report_ref: ObjectRef
    audit_outcome: BatchAuditOutcomeV2
    finding_codes: tuple[BatchAuditFindingCodeV2, ...] = ()
    total_events: int = Field(ge=0)
    event_offset: int = Field(ge=0)
    event_limit: int = Field(
        ge=1,
        le=MAX_PIPELINE_PAGE_SIZE,
    )
    events: tuple[BatchAuditEventV2, ...]

    @model_validator(mode="after")
    def validate_events(self) -> Self:
        _require_ref(
            self.report_ref,
            "batch-audit-report",
            "v2",
            "report_ref",
        )
        _require_findings(
            self.audit_outcome,
            self.finding_codes,
        )
        _require_page(
            "event page",
            total=self.total_events,
            offset=self.event_offset,
            limit=self.event_limit,
            returned=len(self.events),
        )
        if any(event.job_id != self.job_id for event in self.events):
            raise ValueError("event page contains another Job")
        return self


class PipelineMetricSliceSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-metric-slice-summary/v2"] = (
        "eval-factory/pipeline-metric-slice-summary/v2"
    )
    scope: MetricScopeV2
    scope_id: Identifier
    work_units: int = Field(ge=0)
    terminal_attempts: int = Field(ge=0)
    succeeded_attempts: int = Field(ge=0)
    retryable_attempts: int = Field(ge=0)
    terminal_non_success_attempts: int = Field(ge=0)
    cancelled_attempts: int = Field(ge=0)
    retry_attempts: int = Field(ge=0)
    resume_attempts: int = Field(ge=0)
    model_requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    charged_tokens: int = Field(ge=0)
    reported_usage_attempts: int = Field(ge=0)
    conservative_usage_attempts: int = Field(ge=0)
    cache_hit_attempts: int = Field(ge=0)
    cache_miss_attempts: int = Field(ge=0)
    cache_unavailable_attempts: int = Field(ge=0)
    process_starts: int = Field(ge=0)
    renderer_operations: int = Field(ge=0)
    network_requests: int = Field(ge=0)
    retained_storage_bytes: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    reported_cost_microusd: int = Field(ge=0)
    unavailable_cost_attempts: int = Field(ge=0)
    backpressure_events: int = Field(ge=0)
    error_events: int = Field(ge=0)
    queue_wait_sample_count: int = Field(ge=0)
    execution_sample_count: int = Field(ge=0)
    total_sample_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_metric_summary(self) -> Self:
        if (
            self.succeeded_attempts
            + self.retryable_attempts
            + self.terminal_non_success_attempts
            + self.cancelled_attempts
            != self.terminal_attempts
        ):
            raise ValueError("attempt outcomes must classify terminal attempts")
        if (
            self.queue_wait_sample_count != self.terminal_attempts
            or self.execution_sample_count != self.terminal_attempts
            or self.total_sample_count != self.terminal_attempts
        ):
            raise ValueError("sample counts must equal terminal attempts")
        return self

    @classmethod
    def from_metric_slice(
        cls,
        value: MetricSliceV2,
    ) -> PipelineMetricSliceSummaryV2:
        return cls(
            scope=value.scope,
            scope_id=value.scope_id,
            work_units=value.work_units,
            terminal_attempts=value.terminal_attempts,
            succeeded_attempts=value.succeeded_attempts,
            retryable_attempts=value.retryable_attempts,
            terminal_non_success_attempts=(value.terminal_non_success_attempts),
            cancelled_attempts=value.cancelled_attempts,
            retry_attempts=value.retry_attempts,
            resume_attempts=value.resume_attempts,
            model_requests=value.model_requests,
            input_tokens=value.input_tokens,
            output_tokens=value.output_tokens,
            cache_creation_input_tokens=(value.cache_creation_input_tokens),
            cache_read_input_tokens=(value.cache_read_input_tokens),
            charged_tokens=value.charged_tokens,
            reported_usage_attempts=(value.reported_usage_attempts),
            conservative_usage_attempts=(value.conservative_usage_attempts),
            cache_hit_attempts=value.cache_hit_attempts,
            cache_miss_attempts=value.cache_miss_attempts,
            cache_unavailable_attempts=(value.cache_unavailable_attempts),
            process_starts=value.process_starts,
            renderer_operations=value.renderer_operations,
            network_requests=value.network_requests,
            retained_storage_bytes=value.retained_storage_bytes,
            tool_calls=value.tool_calls,
            reported_cost_microusd=value.reported_cost_microusd,
            unavailable_cost_attempts=(value.unavailable_cost_attempts),
            backpressure_events=value.backpressure_events,
            error_events=value.error_events,
            queue_wait_sample_count=len(value.queue_wait_samples_us),
            execution_sample_count=len(value.execution_samples_us),
            total_sample_count=len(value.total_samples_us),
        )


class PipelineMetricsResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/pipeline-metrics-result/v2"] = (
        "eval-factory/pipeline-metrics-result/v2"
    )
    job_id: Identifier
    item_id: Identifier | None = None
    report_ref: ObjectRef
    audit_outcome: BatchAuditOutcomeV2
    finding_codes: tuple[BatchAuditFindingCodeV2, ...] = ()
    total_attempts: int = Field(ge=0)
    attempt_offset: int = Field(ge=0)
    attempt_limit: int = Field(
        ge=1,
        le=MAX_PIPELINE_PAGE_SIZE,
    )
    attempt_metrics_refs: tuple[ObjectRef, ...]
    metric_slices: tuple[PipelineMetricSliceSummaryV2, ...]

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        _require_ref(
            self.report_ref,
            "batch-audit-report",
            "v2",
            "report_ref",
        )
        _require_findings(
            self.audit_outcome,
            self.finding_codes,
        )
        _require_page(
            "attempt page",
            total=self.total_attempts,
            offset=self.attempt_offset,
            limit=self.attempt_limit,
            returned=len(self.attempt_metrics_refs),
        )
        _require_refs(
            self.attempt_metrics_refs,
            "work-attempt-metrics",
            "v2",
            "attempt_metrics_refs",
        )
        slice_keys = tuple((value.scope.value, value.scope_id) for value in self.metric_slices)
        if slice_keys != tuple(sorted(set(slice_keys))):
            raise ValueError("metric slices must be sorted and unique")
        return self


def _require_pipeline_refs(
    plan_ref: ObjectRef,
    graph_ref: ObjectRef,
    policy_ref: ObjectRef,
) -> None:
    _require_ref(
        plan_ref,
        "resolved-dataset-job-plan",
        "v2",
        "resolved_plan_ref",
    )
    _require_ref(
        graph_ref,
        "resolved-job-work-graph",
        "v2",
        "work_graph_ref",
    )
    _require_ref(
        policy_ref,
        "work-control-policy",
        "v2",
        "work_control_policy_ref",
    )


def _require_page(
    label: str,
    *,
    total: int,
    offset: int,
    limit: int,
    returned: int,
) -> None:
    if offset > total:
        raise ValueError(f"{label} offset exceeds total")
    if returned > limit or offset + returned > total:
        raise ValueError(f"{label} is not bounded")


def _require_findings(
    outcome: BatchAuditOutcomeV2,
    findings: tuple[BatchAuditFindingCodeV2, ...],
) -> None:
    if findings != tuple(sorted(set(findings), key=lambda value: value.value)):
        raise ValueError("finding codes must be sorted and unique")
    complete = outcome is BatchAuditOutcomeV2.COMPLETE
    if complete == bool(findings):
        raise ValueError("COMPLETE requires no findings; INCOMPLETE requires findings")


def _require_ref(
    ref: ObjectRef,
    object_type: str,
    object_version: str,
    name: str,
) -> None:
    if ref.object_type != object_type or ref.object_version != object_version:
        raise ValueError(f"{name} must reference {object_type} {object_version}")


def _require_refs(
    refs: tuple[ObjectRef, ...],
    object_type: str,
    object_version: str,
    name: str,
) -> None:
    _require_sorted_unique_refs(name, refs)
    for ref in refs:
        _require_ref(
            ref,
            object_type,
            object_version,
            name,
        )


def _require_sorted_unique_refs(
    name: str,
    refs: tuple[ObjectRef, ...],
) -> None:
    if refs != tuple(sorted(set(refs), key=_ref_key)):
        raise ValueError(f"{name} must be sorted and unique")


def _ref_key(
    ref: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )
