from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.model_control_v2 import (
    ModelUsageSourceV2,
    ModelUsageV2,
)
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    WorkLeaseEventKindV2,
    WorkUnitScopeV2,
)
from eval_factory.contracts.resource_v2 import ResourceUsageV2

R6_OBSERVABILITY_POLICY_VERSION: Literal["batch-observability/r6-06-v1"] = "batch-observability/r6-06-v1"
_Fingerprint = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
_ToolFamily = Literal["FILE", "SHELL", "SEARCH", "FETCH", "RENDERER", "OTHER"]
_RouteKind = Literal["PROVIDER", "RUNTIME"]


class MetricAvailabilityV2(StrEnum):
    REPORTED = "REPORTED"
    UNAVAILABLE = "UNAVAILABLE"


class CacheUseV2(StrEnum):
    HIT = "HIT"
    MISS = "MISS"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class MetricScopeV2(StrEnum):
    JOB = "JOB"
    ITEM = "ITEM"
    STAGE = "STAGE"
    MODEL_PROFILE = "MODEL_PROFILE"
    TOOL_FAMILY = "TOOL_FAMILY"
    ARTIFACT = "ARTIFACT"


class BatchAuditEventKindV2(StrEnum):
    JOB_CREATED = "JOB_CREATED"
    JOB_STATUS_CHANGED = "JOB_STATUS_CHANGED"
    ITEM_CREATED = "ITEM_CREATED"
    ITEM_STATUS_CHANGED = "ITEM_STATUS_CHANGED"
    WORK_GRAPH_CREATED = "WORK_GRAPH_CREATED"
    WORK_CONTROL_POLICY_BOUND = "WORK_CONTROL_POLICY_BOUND"
    WORK_READINESS_RECORDED = "WORK_READINESS_RECORDED"
    WORK_DISPATCHED = "WORK_DISPATCHED"
    WORK_LEASE_ACQUIRED = "WORK_LEASE_ACQUIRED"
    WORK_LEASE_HEARTBEAT = "WORK_LEASE_HEARTBEAT"
    WORK_LEASE_TERMINAL = "WORK_LEASE_TERMINAL"
    WORK_RETRY_DECIDED = "WORK_RETRY_DECIDED"
    WORK_CANCELLED = "WORK_CANCELLED"
    MODEL_POLICY_BOUND = "MODEL_POLICY_BOUND"
    MODEL_ADMISSION_DECIDED = "MODEL_ADMISSION_DECIDED"
    MODEL_USAGE_RECORDED = "MODEL_USAGE_RECORDED"
    MODEL_BACKPRESSURE_RECORDED = "MODEL_BACKPRESSURE_RECORDED"
    MODEL_CANCELLATION_RECORDED = "MODEL_CANCELLATION_RECORDED"
    RESOURCE_POLICY_BOUND = "RESOURCE_POLICY_BOUND"
    RESOURCE_ADMISSION_DECIDED = "RESOURCE_ADMISSION_DECIDED"
    RESOURCE_USAGE_RECORDED = "RESOURCE_USAGE_RECORDED"
    RESOURCE_TERMINATION_RECORDED = "RESOURCE_TERMINATION_RECORDED"
    STAGE_RUN_CREATED = "STAGE_RUN_CREATED"
    STAGE_RUN_COMPLETED = "STAGE_RUN_COMPLETED"
    ARTIFACT_RESULT_RECORDED = "ARTIFACT_RESULT_RECORDED"
    ATTEMPT_METRICS_RECORDED = "ATTEMPT_METRICS_RECORDED"


class BatchAuditOutcomeV2(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class BatchAuditFindingCodeV2(StrEnum):
    MISSING_OUTBOX_WITNESS = "MISSING_OUTBOX_WITNESS"
    DUPLICATE_OUTBOX_WITNESS = "DUPLICATE_OUTBOX_WITNESS"
    MISSING_TERMINAL_METRICS = "MISSING_TERMINAL_METRICS"
    METRICS_BINDING_MISMATCH = "METRICS_BINDING_MISMATCH"
    MISSING_MODEL_EVENT = "MISSING_MODEL_EVENT"
    MISSING_RESOURCE_EVENT = "MISSING_RESOURCE_EVENT"
    LEASE_EVENT_GAP = "LEASE_EVENT_GAP"
    RESERVATION_EVENT_GAP = "RESERVATION_EVENT_GAP"
    RETRY_CHAIN_MISMATCH = "RETRY_CHAIN_MISMATCH"
    CROSS_SCOPE_REFERENCE = "CROSS_SCOPE_REFERENCE"
    CLOCK_REGRESSION = "CLOCK_REGRESSION"
    PROJECTION_DRIFT = "PROJECTION_DRIFT"
    UNEXPECTED_SOURCE_RECORD = "UNEXPECTED_SOURCE_RECORD"


class ToolMetricV2(ContractModelV2):
    tool_family: _ToolFamily
    calls: int = Field(ge=1)


class CostObservationV2(ContractModelV2):
    availability: MetricAvailabilityV2
    currency: Literal["USD"] = "USD"
    amount_microusd: int | None = Field(default=None, ge=0)
    source_ref: ObjectRef | None = None
    rounding: Literal["HALF_EVEN_MICRO_USD"] | None = None

    @model_validator(mode="after")
    def validate_cost(self) -> Self:
        reported = self.availability is MetricAvailabilityV2.REPORTED
        if reported != (self.amount_microusd is not None):
            raise ValueError("reported cost availability and amount_microusd disagree")
        if reported != (self.source_ref is not None):
            raise ValueError("reported cost availability and source_ref disagree")
        if reported != (self.rounding is not None):
            raise ValueError("reported cost availability and rounding disagree")
        return self

    @classmethod
    def reported(
        cls,
        *,
        amount_microusd: int,
        source_ref: ObjectRef,
    ) -> CostObservationV2:
        return cls(
            availability=MetricAvailabilityV2.REPORTED,
            amount_microusd=amount_microusd,
            source_ref=source_ref,
            rounding="HALF_EVEN_MICRO_USD",
        )

    @classmethod
    def unavailable(cls) -> CostObservationV2:
        return cls(availability=MetricAvailabilityV2.UNAVAILABLE)


class ArtifactMetricSliceV2(ContractModelV2):
    artifact_id: Identifier
    status: Identifier
    telemetry_ref: ObjectRef | None = None
    selected_route_kind: _RouteKind | None = None
    selected_route_id: Identifier | None = None
    selected_route_version: str | None = Field(default=None, min_length=1, max_length=256)
    worker_version: str | None = Field(default=None, min_length=1, max_length=256)
    output_ref: ObjectRef | None = None
    tool_metrics: tuple[ToolMetricV2, ...] = ()
    cost: CostObservationV2

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        _require_route_pair(
            self.selected_route_kind,
            self.selected_route_id,
            self.selected_route_version,
        )
        if (self.selected_route_id is None) != (self.telemetry_ref is None):
            raise ValueError("attempted artifact metrics require telemetry")
        if self.telemetry_ref is not None:
            _require_ref(
                self.telemetry_ref,
                "execution-telemetry",
                "v2",
                "telemetry_ref",
            )
        _require_tool_metrics(self.tool_metrics)
        return self


class WorkAttemptMetricsV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-attempt-metrics/v2"] = "eval-factory/work-attempt-metrics/v2"
    work_attempt_metrics_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    work_unit_ref: ObjectRef
    work_lease_ref: ObjectRef
    stage_run_ref: ObjectRef | None = None
    job_id: Identifier
    item_id: Identifier | None = None
    scope: WorkUnitScopeV2
    stage: StageNameV2
    artifact_execution_group_ref: ObjectRef | None = None
    attempt: int = Field(ge=1)
    fencing_token: int = Field(ge=1)
    terminal_lease_version: int = Field(ge=1)
    terminal_event_kind: WorkLeaseEventKindV2
    terminal_status: StageRunStatus
    reason_code: Identifier | None = None
    ready_at: datetime
    eligible_at: datetime
    acquired_at: datetime
    completed_at: datetime
    queue_wait_us: int = Field(ge=0)
    execution_duration_us: int = Field(ge=0)
    total_duration_us: int = Field(ge=0)
    model_profile_ref: ObjectRef | None = None
    model_usage: ModelUsageV2 | None = None
    resource_usage: ResourceUsageV2 | None = None
    telemetry_ref: ObjectRef
    cache_use: CacheUseV2
    tool_metrics: tuple[ToolMetricV2, ...] = ()
    cost: CostObservationV2
    selected_route_kind: _RouteKind | None = None
    selected_route_id: Identifier | None = None
    selected_route_version: str | None = Field(default=None, min_length=1, max_length=256)
    worker_version: str | None = Field(default=None, min_length=1, max_length=256)
    retry_of_attempt_metrics_ref: ObjectRef | None = None
    result_refs: tuple[ObjectRef, ...] = ()
    output_refs: tuple[ObjectRef, ...] = ()
    artifact_slices: tuple[ArtifactMetricSliceV2, ...] = ()
    policy_version: Literal["batch-observability/r6-06-v1"] = R6_OBSERVABILITY_POLICY_VERSION
    work_attempt_metrics_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        for ref, object_type, version, name in (
            (
                self.resolved_job_work_graph_ref,
                "resolved-job-work-graph",
                "v2",
                "resolved_job_work_graph_ref",
            ),
            (self.work_unit_ref, "resolved-work-unit", "v2", "work_unit_ref"),
            (self.work_lease_ref, "work-lease", "v2", "work_lease_ref"),
        ):
            _require_ref(ref, object_type, version, name)
        if self.stage_run_ref is not None:
            _require_ref(self.stage_run_ref, "stage-run", "identity/v1", "stage_run_ref")
        artifact_scope = self.scope is WorkUnitScopeV2.ARTIFACT_GROUP
        if artifact_scope != (self.artifact_execution_group_ref is not None):
            raise ValueError("artifact-group scope and artifact_execution_group_ref disagree")
        if artifact_scope:
            if (
                self.stage_run_ref is not None
                or self.item_id is None
                or self.stage is not StageNameV2.ATTACHMENT
            ):
                raise ValueError("artifact-group metrics require item attachment work without StageRun")
        else:
            if self.stage_run_ref is None:
                raise ValueError("job/item attempt metrics require a StageRun")
            if self.artifact_slices:
                raise ValueError("job/item attempt metrics cannot carry artifact slices")
        if self.artifact_execution_group_ref is not None:
            _require_ref(
                self.artifact_execution_group_ref,
                "artifact-execution-group",
                "v2",
                "artifact_execution_group_ref",
            )
        if self.terminal_event_kind is WorkLeaseEventKindV2.HEARTBEAT:
            raise ValueError("attempt metrics require a terminal lease event kind")
        if self.terminal_event_kind is WorkLeaseEventKindV2.SUCCEEDED:
            if self.terminal_status is not StageRunStatus.SUCCEEDED or self.reason_code is not None:
                raise ValueError("successful terminal metrics have mismatched status or reason")
        elif self.reason_code is None:
            raise ValueError("non-success terminal metrics require reason_code")
        _require_aware_ordered_timestamps(
            self.ready_at,
            self.eligible_at,
            self.acquired_at,
            self.completed_at,
        )
        expected_queue = _duration_us(max(self.ready_at, self.eligible_at), self.acquired_at)
        expected_execution = _duration_us(self.acquired_at, self.completed_at)
        expected_total = _duration_us(self.ready_at, self.completed_at)
        if (
            self.queue_wait_us != expected_queue
            or self.execution_duration_us != expected_execution
            or self.total_duration_us != expected_total
        ):
            raise ValueError("attempt metric durations do not match authoritative timestamps")
        expected_cache = _cache_use(self.model_usage)
        if self.cache_use is not expected_cache:
            raise ValueError("cache_use does not match model usage source and cache reads")
        if (self.model_profile_ref is None) != (self.model_usage is None):
            raise ValueError("model profile and model usage must be paired")
        if self.model_profile_ref is not None and (
            self.model_profile_ref.object_type != "model-profile"
            or self.model_profile_ref.object_version not in {"v1", "v2"}
        ):
            raise ValueError("model_profile_ref must reference a versioned model-profile")
        _require_ref(
            self.telemetry_ref,
            "execution-telemetry",
            "v2",
            "telemetry_ref",
        )
        _require_route_pair(
            self.selected_route_kind,
            self.selected_route_id,
            self.selected_route_version,
        )
        _require_tool_metrics(self.tool_metrics)
        _require_sorted_unique_refs("result_refs", self.result_refs)
        _require_sorted_unique_refs("output_refs", self.output_refs)
        if self.retry_of_attempt_metrics_ref is not None:
            _require_ref(
                self.retry_of_attempt_metrics_ref,
                "work-attempt-metrics",
                "v2",
                "retry_of_attempt_metrics_ref",
            )
        artifact_ids = tuple(item.artifact_id for item in self.artifact_slices)
        if artifact_ids != tuple(sorted(set(artifact_ids))):
            raise ValueError("artifact slices must be sorted and unique")
        refs = _unique_sorted_refs(
            (
                self.resolved_job_work_graph_ref,
                self.work_unit_ref,
                self.work_lease_ref,
                *((self.stage_run_ref,) if self.stage_run_ref else ()),
                *((self.artifact_execution_group_ref,) if self.artifact_execution_group_ref else ()),
                *((self.model_profile_ref,) if self.model_profile_ref else ()),
                self.telemetry_ref,
                *((self.retry_of_attempt_metrics_ref,) if self.retry_of_attempt_metrics_ref else ()),
                *self.result_refs,
                *self.output_refs,
                *(
                    ref
                    for artifact in self.artifact_slices
                    for ref in (
                        *((artifact.telemetry_ref,) if artifact.telemetry_ref else ()),
                        *((artifact.output_ref,) if artifact.output_ref else ()),
                        *((artifact.cost.source_ref,) if artifact.cost.source_ref else ()),
                    )
                ),
                *((self.cost.source_ref,) if self.cost.source_ref else ()),
            )
        )
        _require_observability_audit(self.audit, refs)
        validate_work_attempt_metrics_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        work_unit_ref: ObjectRef,
        work_lease_ref: ObjectRef,
        stage_run_ref: ObjectRef | None,
        job_id: Identifier,
        item_id: Identifier | None,
        scope: WorkUnitScopeV2,
        stage: StageNameV2,
        artifact_execution_group_ref: ObjectRef | None,
        attempt: int,
        fencing_token: int,
        terminal_lease_version: int,
        terminal_event_kind: WorkLeaseEventKindV2,
        terminal_status: StageRunStatus,
        reason_code: Identifier | None,
        ready_at: datetime,
        eligible_at: datetime,
        acquired_at: datetime,
        completed_at: datetime,
        model_profile_ref: ObjectRef | None,
        model_usage: ModelUsageV2 | None,
        resource_usage: ResourceUsageV2 | None,
        telemetry_ref: ObjectRef,
        tool_metrics: tuple[ToolMetricV2, ...],
        cost: CostObservationV2,
        selected_route_kind: _RouteKind | None,
        selected_route_id: Identifier | None,
        selected_route_version: str | None,
        worker_version: str | None,
        retry_of_attempt_metrics_ref: ObjectRef | None,
        result_refs: tuple[ObjectRef, ...],
        output_refs: tuple[ObjectRef, ...],
        artifact_slices: tuple[ArtifactMetricSliceV2, ...],
        audit: ContractAudit,
    ) -> WorkAttemptMetricsV2:
        value = cls(
            work_attempt_metrics_id="work-attempt-metrics://pending",
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            work_unit_ref=work_unit_ref,
            work_lease_ref=work_lease_ref,
            stage_run_ref=stage_run_ref,
            job_id=job_id,
            item_id=item_id,
            scope=scope,
            stage=stage,
            artifact_execution_group_ref=artifact_execution_group_ref,
            attempt=attempt,
            fencing_token=fencing_token,
            terminal_lease_version=terminal_lease_version,
            terminal_event_kind=terminal_event_kind,
            terminal_status=terminal_status,
            reason_code=reason_code,
            ready_at=ready_at,
            eligible_at=eligible_at,
            acquired_at=acquired_at,
            completed_at=completed_at,
            queue_wait_us=_duration_us(max(ready_at, eligible_at), acquired_at),
            execution_duration_us=_duration_us(acquired_at, completed_at),
            total_duration_us=_duration_us(ready_at, completed_at),
            model_profile_ref=model_profile_ref,
            model_usage=model_usage,
            resource_usage=resource_usage,
            telemetry_ref=telemetry_ref,
            cache_use=_cache_use(model_usage),
            tool_metrics=tuple(sorted(tool_metrics, key=lambda item: item.tool_family)),
            cost=cost,
            selected_route_kind=selected_route_kind,
            selected_route_id=selected_route_id,
            selected_route_version=selected_route_version,
            worker_version=worker_version,
            retry_of_attempt_metrics_ref=retry_of_attempt_metrics_ref,
            result_refs=_unique_sorted_refs(result_refs),
            output_refs=_unique_sorted_refs(output_refs),
            artifact_slices=tuple(sorted(artifact_slices, key=lambda item: item.artifact_id)),
            work_attempt_metrics_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_attempt_metrics_id",
            "work_attempt_metrics_sha256",
            "work-attempt-metrics",
        )


class BatchAuditEventV2(ContractModelV2):
    schema_version: Literal["eval-factory/batch-audit-event/v2"] = "eval-factory/batch-audit-event/v2"
    batch_audit_event_id: Identifier
    job_id: Identifier
    item_id: Identifier | None = None
    work_unit_ref: ObjectRef | None = None
    stage: StageNameV2 | None = None
    artifact_id: Identifier | None = None
    attempt: int | None = Field(default=None, ge=1)
    event_kind: BatchAuditEventKindV2
    status: Identifier | None = None
    reason_code: Identifier | None = None
    occurred_at: datetime
    source_family: Identifier
    source_version: int = Field(ge=0)
    source_ref: ObjectRef
    outbox_event_id: Identifier | None = None
    related_refs: tuple[ObjectRef, ...] = ()
    result_refs: tuple[ObjectRef, ...] = ()
    policy_version: Literal["batch-observability/r6-06-v1"] = R6_OBSERVABILITY_POLICY_VERSION
    batch_audit_event_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        if self.attempt is not None and self.work_unit_ref is None:
            raise ValueError("attempt-scoped audit events require a work unit")
        if self.work_unit_ref is not None:
            _require_ref(self.work_unit_ref, "resolved-work-unit", "v2", "work_unit_ref")
        _require_sorted_unique_refs("related_refs", self.related_refs)
        _require_sorted_unique_refs("result_refs", self.result_refs)
        refs = _unique_sorted_refs((self.source_ref, *self.related_refs, *self.result_refs))
        _require_observability_audit(self.audit, refs)
        validate_batch_audit_event_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: Identifier,
        item_id: Identifier | None,
        work_unit_ref: ObjectRef | None,
        stage: StageNameV2 | None,
        artifact_id: Identifier | None,
        attempt: int | None,
        event_kind: BatchAuditEventKindV2,
        status: Identifier | None,
        reason_code: Identifier | None,
        occurred_at: datetime,
        source_family: Identifier,
        source_version: int,
        source_ref: ObjectRef,
        outbox_event_id: Identifier | None,
        related_refs: tuple[ObjectRef, ...],
        result_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> BatchAuditEventV2:
        value = cls(
            batch_audit_event_id="batch-audit-event://pending",
            job_id=job_id,
            item_id=item_id,
            work_unit_ref=work_unit_ref,
            stage=stage,
            artifact_id=artifact_id,
            attempt=attempt,
            event_kind=event_kind,
            status=status,
            reason_code=reason_code,
            occurred_at=occurred_at,
            source_family=source_family,
            source_version=source_version,
            source_ref=source_ref,
            outbox_event_id=outbox_event_id,
            related_refs=_unique_sorted_refs(related_refs),
            result_refs=_unique_sorted_refs(result_refs),
            batch_audit_event_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "batch_audit_event_id",
            "batch_audit_event_sha256",
            "batch-audit-event",
        )


class MetricSliceV2(ContractModelV2):
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
    queue_wait_samples_us: tuple[int, ...] = ()
    execution_samples_us: tuple[int, ...] = ()
    total_samples_us: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_slice(self) -> Self:
        for samples in (
            self.queue_wait_samples_us,
            self.execution_samples_us,
            self.total_samples_us,
        ):
            if any(item < 0 for item in samples) or samples != tuple(sorted(samples)):
                raise ValueError("latency samples must be non-negative and sorted")
            if len(samples) != self.terminal_attempts:
                raise ValueError("each terminal attempt requires one latency sample")
        if (
            self.succeeded_attempts
            + self.retryable_attempts
            + self.terminal_non_success_attempts
            + self.cancelled_attempts
            != self.terminal_attempts
        ):
            raise ValueError("attempt outcome counts must classify terminal attempts")
        if self.retry_attempts > self.terminal_attempts:
            raise ValueError("retry attempts cannot exceed terminal attempts")
        if self.resume_attempts > self.retry_attempts:
            raise ValueError("resume attempts cannot exceed retry attempts")
        if self.reported_usage_attempts + self.conservative_usage_attempts > self.terminal_attempts:
            raise ValueError("model usage attempts cannot exceed terminal attempts")
        if (
            self.cache_hit_attempts + self.cache_miss_attempts + self.cache_unavailable_attempts
            != self.reported_usage_attempts + self.conservative_usage_attempts
        ):
            raise ValueError("cache counts must classify every model usage attempt")
        return self


class ItemMetricsSnapshotV2(ContractModelV2):
    schema_version: Literal["eval-factory/item-metrics-snapshot/v2"] = "eval-factory/item-metrics-snapshot/v2"
    item_metrics_snapshot_id: Identifier
    subject_ref: ObjectRef
    predecessor_ref: ObjectRef | None = None
    source_fingerprint: _Fingerprint
    attempt_metrics_refs: tuple[ObjectRef, ...]
    audit_event_refs: tuple[ObjectRef, ...]
    metric_slices: tuple[MetricSliceV2, ...]
    policy_version: Literal["batch-observability/r6-06-v1"] = R6_OBSERVABILITY_POLICY_VERSION
    item_metrics_snapshot_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        _require_ref(self.subject_ref, "item-record", "record/v1", "subject_ref")
        if self.predecessor_ref is not None:
            _require_ref(
                self.predecessor_ref,
                "item-metrics-snapshot",
                "v2",
                "predecessor_ref",
            )
        _require_metric_refs(self.attempt_metrics_refs, self.audit_event_refs)
        _require_metric_slices(self.metric_slices)
        refs = _unique_sorted_refs(
            (
                self.subject_ref,
                *((self.predecessor_ref,) if self.predecessor_ref else ()),
                *self.attempt_metrics_refs,
                *self.audit_event_refs,
            )
        )
        _require_observability_audit(self.audit, refs)
        validate_item_metrics_snapshot_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        subject_ref: ObjectRef,
        predecessor_ref: ObjectRef | None,
        source_fingerprint: str,
        attempt_metrics_refs: tuple[ObjectRef, ...],
        audit_event_refs: tuple[ObjectRef, ...],
        metric_slices: tuple[MetricSliceV2, ...],
        audit: ContractAudit,
    ) -> ItemMetricsSnapshotV2:
        value = cls(
            item_metrics_snapshot_id="item-metrics-snapshot://pending",
            subject_ref=subject_ref,
            predecessor_ref=predecessor_ref,
            source_fingerprint=source_fingerprint,
            attempt_metrics_refs=_unique_sorted_refs(attempt_metrics_refs),
            audit_event_refs=_unique_sorted_refs(audit_event_refs),
            metric_slices=tuple(sorted(metric_slices, key=_metric_slice_key)),
            item_metrics_snapshot_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "item_metrics_snapshot_id",
            "item_metrics_snapshot_sha256",
            "item-metrics-snapshot",
        )


class BatchMetricsSnapshotV2(ContractModelV2):
    schema_version: Literal["eval-factory/batch-metrics-snapshot/v2"] = (
        "eval-factory/batch-metrics-snapshot/v2"
    )
    batch_metrics_snapshot_id: Identifier
    subject_ref: ObjectRef
    predecessor_ref: ObjectRef | None = None
    source_fingerprint: _Fingerprint
    attempt_metrics_refs: tuple[ObjectRef, ...]
    audit_event_refs: tuple[ObjectRef, ...]
    item_metrics_refs: tuple[ObjectRef, ...]
    metric_slices: tuple[MetricSliceV2, ...]
    policy_version: Literal["batch-observability/r6-06-v1"] = R6_OBSERVABILITY_POLICY_VERSION
    batch_metrics_snapshot_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        _require_ref(
            self.subject_ref,
            "resolved-job-work-graph",
            "v2",
            "subject_ref",
        )
        if self.predecessor_ref is not None:
            _require_ref(
                self.predecessor_ref,
                "batch-metrics-snapshot",
                "v2",
                "predecessor_ref",
            )
        _require_metric_refs(self.attempt_metrics_refs, self.audit_event_refs)
        _require_refs(
            self.item_metrics_refs,
            "item-metrics-snapshot",
            "v2",
            "item_metrics_refs",
        )
        _require_metric_slices(self.metric_slices)
        refs = _unique_sorted_refs(
            (
                self.subject_ref,
                *((self.predecessor_ref,) if self.predecessor_ref else ()),
                *self.attempt_metrics_refs,
                *self.audit_event_refs,
                *self.item_metrics_refs,
            )
        )
        _require_observability_audit(self.audit, refs)
        validate_batch_metrics_snapshot_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        subject_ref: ObjectRef,
        predecessor_ref: ObjectRef | None,
        source_fingerprint: str,
        attempt_metrics_refs: tuple[ObjectRef, ...],
        audit_event_refs: tuple[ObjectRef, ...],
        item_metrics_refs: tuple[ObjectRef, ...],
        metric_slices: tuple[MetricSliceV2, ...],
        audit: ContractAudit,
    ) -> BatchMetricsSnapshotV2:
        value = cls(
            batch_metrics_snapshot_id="batch-metrics-snapshot://pending",
            subject_ref=subject_ref,
            predecessor_ref=predecessor_ref,
            source_fingerprint=source_fingerprint,
            attempt_metrics_refs=_unique_sorted_refs(attempt_metrics_refs),
            audit_event_refs=_unique_sorted_refs(audit_event_refs),
            item_metrics_refs=_unique_sorted_refs(item_metrics_refs),
            metric_slices=tuple(sorted(metric_slices, key=_metric_slice_key)),
            batch_metrics_snapshot_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "batch_metrics_snapshot_id",
            "batch_metrics_snapshot_sha256",
            "batch-metrics-snapshot",
        )


class BatchAuditFindingV2(ContractModelV2):
    code: BatchAuditFindingCodeV2
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_finding(self) -> Self:
        _require_sorted_unique_refs("finding subject_refs", self.subject_refs)
        return self


class BatchAuditReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/batch-audit-report/v2"] = "eval-factory/batch-audit-report/v2"
    batch_audit_report_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    source_fingerprint: _Fingerprint
    outcome: BatchAuditOutcomeV2
    expected_work_unit_refs: tuple[ObjectRef, ...]
    audit_event_refs: tuple[ObjectRef, ...]
    attempt_metrics_refs: tuple[ObjectRef, ...]
    item_metrics_refs: tuple[ObjectRef, ...]
    batch_metrics_ref: ObjectRef
    findings: tuple[BatchAuditFindingV2, ...]
    policy_version: Literal["batch-observability/r6-06-v1"] = R6_OBSERVABILITY_POLICY_VERSION
    batch_audit_report_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        _require_refs(
            self.expected_work_unit_refs,
            "resolved-work-unit",
            "v2",
            "expected_work_unit_refs",
        )
        _require_metric_refs(self.attempt_metrics_refs, self.audit_event_refs)
        _require_refs(
            self.item_metrics_refs,
            "item-metrics-snapshot",
            "v2",
            "item_metrics_refs",
        )
        _require_ref(
            self.batch_metrics_ref,
            "batch-metrics-snapshot",
            "v2",
            "batch_metrics_ref",
        )
        finding_keys = tuple(
            (finding.code.value, tuple(_ref_key(ref) for ref in finding.subject_refs))
            for finding in self.findings
        )
        if finding_keys != tuple(sorted(set(finding_keys))):
            raise ValueError("findings must be sorted and unique")
        complete = self.outcome is BatchAuditOutcomeV2.COMPLETE
        if complete == bool(self.findings):
            raise ValueError("COMPLETE report requires no findings; INCOMPLETE requires findings")
        refs = _unique_sorted_refs(
            (
                self.resolved_job_work_graph_ref,
                *self.expected_work_unit_refs,
                *self.audit_event_refs,
                *self.attempt_metrics_refs,
                *self.item_metrics_refs,
                self.batch_metrics_ref,
                *(ref for finding in self.findings for ref in finding.subject_refs),
            )
        )
        _require_observability_audit(self.audit, refs)
        validate_batch_audit_report_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        source_fingerprint: str,
        outcome: BatchAuditOutcomeV2,
        expected_work_unit_refs: tuple[ObjectRef, ...],
        audit_event_refs: tuple[ObjectRef, ...],
        attempt_metrics_refs: tuple[ObjectRef, ...],
        item_metrics_refs: tuple[ObjectRef, ...],
        batch_metrics_ref: ObjectRef,
        findings: tuple[BatchAuditFindingV2, ...],
        audit: ContractAudit,
    ) -> BatchAuditReportV2:
        ordered_findings = tuple(
            sorted(
                findings,
                key=lambda item: (
                    item.code.value,
                    tuple(_ref_key(ref) for ref in item.subject_refs),
                ),
            )
        )
        value = cls(
            batch_audit_report_id="batch-audit-report://pending",
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            source_fingerprint=source_fingerprint,
            outcome=outcome,
            expected_work_unit_refs=_unique_sorted_refs(expected_work_unit_refs),
            audit_event_refs=_unique_sorted_refs(audit_event_refs),
            attempt_metrics_refs=_unique_sorted_refs(attempt_metrics_refs),
            item_metrics_refs=_unique_sorted_refs(item_metrics_refs),
            batch_metrics_ref=batch_metrics_ref,
            findings=ordered_findings,
            batch_audit_report_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "batch_audit_report_id",
            "batch_audit_report_sha256",
            "batch-audit-report",
        )


_REF_FIELDS: dict[type[ContractModelV2], tuple[str, str, str]] = {
    WorkAttemptMetricsV2: (
        "work_attempt_metrics_id",
        "work_attempt_metrics_sha256",
        "work-attempt-metrics",
    ),
    BatchAuditEventV2: (
        "batch_audit_event_id",
        "batch_audit_event_sha256",
        "batch-audit-event",
    ),
    ItemMetricsSnapshotV2: (
        "item_metrics_snapshot_id",
        "item_metrics_snapshot_sha256",
        "item-metrics-snapshot",
    ),
    BatchMetricsSnapshotV2: (
        "batch_metrics_snapshot_id",
        "batch_metrics_snapshot_sha256",
        "batch-metrics-snapshot",
    ),
    BatchAuditReportV2: (
        "batch_audit_report_id",
        "batch_audit_report_sha256",
        "batch-audit-report",
    ),
}


def work_attempt_metrics_v2_ref(value: WorkAttemptMetricsV2) -> ObjectRef:
    return _ref_for(value)


def batch_audit_event_v2_ref(value: BatchAuditEventV2) -> ObjectRef:
    return _ref_for(value)


def item_metrics_snapshot_v2_ref(value: ItemMetricsSnapshotV2) -> ObjectRef:
    return _ref_for(value)


def batch_metrics_snapshot_v2_ref(value: BatchMetricsSnapshotV2) -> ObjectRef:
    return _ref_for(value)


def batch_audit_report_v2_ref(value: BatchAuditReportV2) -> ObjectRef:
    return _ref_for(value)


def validate_work_attempt_metrics_v2_identity(value: WorkAttemptMetricsV2) -> None:
    _validate(value, *_REF_FIELDS[WorkAttemptMetricsV2])


def validate_batch_audit_event_v2_identity(value: BatchAuditEventV2) -> None:
    _validate(value, *_REF_FIELDS[BatchAuditEventV2])


def validate_item_metrics_snapshot_v2_identity(value: ItemMetricsSnapshotV2) -> None:
    _validate(value, *_REF_FIELDS[ItemMetricsSnapshotV2])


def validate_batch_metrics_snapshot_v2_identity(value: BatchMetricsSnapshotV2) -> None:
    _validate(value, *_REF_FIELDS[BatchMetricsSnapshotV2])


def validate_batch_audit_report_v2_identity(value: BatchAuditReportV2) -> None:
    _validate(value, *_REF_FIELDS[BatchAuditReportV2])


def _carried(value: ContractModelV2, id_field: str, sha_field: str) -> str:
    excluded_fields = {id_field, sha_field, "audit"}
    if isinstance(value, BatchAuditEventV2):
        excluded_fields.add("outbox_event_id")
    encoded = json.dumps(
        canonical_value_v2(
            value.model_dump(
                mode="python",
                exclude=excluded_fields,
            )
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    id_field: str,
    sha_field: str,
    prefix: str,
) -> ModelT:
    digest = _carried(value, id_field, sha_field)
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            sha_field: digest,
        }
    )


def _validate(
    value: ContractModelV2,
    id_field: str,
    sha_field: str,
    prefix: str,
) -> None:
    object_id = str(getattr(value, id_field))
    object_sha256 = str(getattr(value, sha_field))
    if object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    digest = _carried(value, id_field, sha_field)
    if object_id != f"{prefix}://sha256/{digest}" or object_sha256 != digest:
        raise ValueError(f"{prefix} identity is stale or invalid")


def _ref_for(value: ContractModelV2) -> ObjectRef:
    id_field, sha_field, prefix = _REF_FIELDS[type(value)]
    _validate(value, id_field, sha_field, prefix)
    return ObjectRef(
        object_type=prefix,
        object_id=str(getattr(value, id_field)),
        object_version="v2",
        object_sha256=str(getattr(value, sha_field)),
    )


def _duration_us(start: datetime, end: datetime) -> int:
    if start.tzinfo is None or end.tzinfo is None or end < start:
        raise ValueError("attempt timestamps must be timezone-aware and nondecreasing")
    return int((end - start).total_seconds() * 1_000_000)


def _require_aware_ordered_timestamps(
    ready_at: datetime,
    eligible_at: datetime,
    acquired_at: datetime,
    completed_at: datetime,
) -> None:
    if any(value.tzinfo is None for value in (ready_at, eligible_at, acquired_at, completed_at)):
        raise ValueError("attempt timestamps must be timezone-aware")
    if eligible_at < ready_at or acquired_at < max(ready_at, eligible_at) or completed_at < acquired_at:
        raise ValueError("attempt timestamps must be nondecreasing")


def _cache_use(usage: ModelUsageV2 | None) -> CacheUseV2:
    if usage is None:
        return CacheUseV2.NOT_APPLICABLE
    if usage.source is not ModelUsageSourceV2.REPORTED:
        return CacheUseV2.UNAVAILABLE
    return CacheUseV2.HIT if usage.cache_read_input_tokens else CacheUseV2.MISS


def _require_route_pair(
    kind: _RouteKind | None,
    route_id: Identifier | None,
    version: str | None,
) -> None:
    if (kind is None) != (route_id is None):
        raise ValueError("selected route kind and ID must be paired")
    if version is not None and route_id is None:
        raise ValueError("selected route version requires a route")


def _require_tool_metrics(values: tuple[ToolMetricV2, ...]) -> None:
    families = tuple(item.tool_family for item in values)
    if families != tuple(sorted(set(families))):
        raise ValueError("tool metrics must be sorted and unique")


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)


def _unique_sorted_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(refs), key=_ref_key))


def _require_sorted_unique_refs(name: str, refs: tuple[ObjectRef, ...]) -> None:
    if refs != _unique_sorted_refs(refs):
        raise ValueError(f"{name} must be sorted and unique")


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
        _require_ref(ref, object_type, object_version, name)


def _require_metric_refs(
    attempt_refs: tuple[ObjectRef, ...],
    event_refs: tuple[ObjectRef, ...],
) -> None:
    _require_refs(
        attempt_refs,
        "work-attempt-metrics",
        "v2",
        "attempt_metrics_refs",
    )
    _require_refs(event_refs, "batch-audit-event", "v2", "audit_event_refs")


def _metric_slice_key(value: MetricSliceV2) -> tuple[str, str]:
    return (value.scope.value, value.scope_id)


def _require_metric_slices(values: tuple[MetricSliceV2, ...]) -> None:
    keys = tuple(_metric_slice_key(value) for value in values)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("metric slices must be sorted and unique")


def _require_observability_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> None:
    if audit.input_refs != _unique_sorted_refs(refs):
        raise ValueError("observability audit refs are incomplete")
    bindings = tuple(
        binding for binding in audit.governing_versions if binding.component == "batch-observability"
    )
    if len(bindings) != 1 or bindings[0].version != R6_OBSERVABILITY_POLICY_VERSION:
        raise ValueError("observability audit is missing the current policy")
