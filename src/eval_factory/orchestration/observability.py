from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ValidationError

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
)
from eval_factory.contracts.core_v2 import (
    canonical_value_v2,
)
from eval_factory.contracts.model_control_v2 import (
    JobModelPolicyV2,
    ModelAdmissionDecisionV2,
    ModelUsageSourceV2,
    ProviderBackpressureRecordV2,
    WorkModelCancellationRequestV2,
    WorkModelCancellationResultV2,
    WorkModelDemandV2,
    WorkModelEventKindV2,
    WorkModelEventV2,
    WorkModelReservationV2,
    job_model_policy_v2_ref,
    model_admission_decision_v2_ref,
    provider_backpressure_record_v2_ref,
    work_model_cancellation_request_v2_ref,
    work_model_cancellation_result_v2_ref,
    work_model_demand_v2_ref,
    work_model_event_v2_ref,
    work_model_reservation_v2_ref,
)
from eval_factory.contracts.observability_v2 import (
    BatchAuditEventKindV2,
    BatchAuditEventV2,
    BatchAuditFindingCodeV2,
    BatchAuditFindingV2,
    BatchAuditOutcomeV2,
    BatchAuditReportV2,
    BatchMetricsSnapshotV2,
    CacheUseV2,
    ItemMetricsSnapshotV2,
    MetricAvailabilityV2,
    MetricScopeV2,
    MetricSliceV2,
    WorkAttemptMetricsV2,
    batch_audit_event_v2_ref,
    batch_metrics_snapshot_v2_ref,
    item_metrics_snapshot_v2_ref,
    work_attempt_metrics_v2_ref,
)
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    WorkCancellationRecordV2,
    WorkControlPolicyV2,
    WorkDispatchDecisionV2,
    WorkLeaseEventKindV2,
    WorkLeaseEventV2,
    WorkLeaseV2,
    WorkReadinessSnapshotV2,
    WorkRetryDecisionV2,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
    work_cancellation_record_v2_ref,
    work_control_policy_v2_ref,
    work_dispatch_decision_v2_ref,
    work_lease_event_v2_ref,
    work_lease_v2_ref,
    work_readiness_snapshot_v2_ref,
    work_retry_decision_v2_ref,
)
from eval_factory.contracts.resource_v2 import (
    JobResourcePolicyV2,
    ResourceAdmissionDecisionV2,
    WorkResourceDemandV2,
    WorkResourceEventV2,
    WorkResourceReservationV2,
    WorkResourceTerminationRequestV2,
    WorkResourceTerminationResultV2,
    job_resource_policy_v2_ref,
    resource_admission_decision_v2_ref,
    work_resource_demand_v2_ref,
    work_resource_event_v2_ref,
    work_resource_reservation_v2_ref,
    work_resource_termination_request_v2_ref,
    work_resource_termination_result_v2_ref,
)
from eval_factory.orchestration.job_store import (
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    JobStoreError,
    RecordNotFoundError,
    _observability_audit,
    _request_sha256,
)
from eval_factory.orchestration.model_control import ModelControlService
from eval_factory.orchestration.models import (
    ItemRecord,
    JobRecord,
    OutboxEvent,
    StageResultRecord,
    StageRunRecord,
    item_record_ref,
    stage_result_record_ref,
    stage_run_record_ref,
)
from eval_factory.orchestration.resources import ResourceControlService


class BatchObservabilityPolicyError(JobStoreError):
    pass


class BatchObservabilityIntegrityError(JobStoreError):
    pass


@dataclass(frozen=True, slots=True)
class BatchObservabilityResult:
    audit_events: tuple[BatchAuditEventV2, ...]
    attempt_metrics: tuple[WorkAttemptMetricsV2, ...]
    item_metrics: tuple[ItemMetricsSnapshotV2, ...]
    batch_metrics: BatchMetricsSnapshotV2
    report: BatchAuditReportV2


@dataclass(frozen=True, slots=True)
class _CompiledSources:
    graph: ResolvedJobWorkGraphV2
    units: tuple[ResolvedWorkUnitV2, ...]
    items: tuple[ItemRecord, ...]
    attempt_metrics: tuple[WorkAttemptMetricsV2, ...]
    events: tuple[BatchAuditEventV2, ...]
    findings: tuple[BatchAuditFindingV2, ...]
    source_fingerprint: str


_OUTBOX_EVENT_KINDS: dict[str, BatchAuditEventKindV2] = {
    "job-created": BatchAuditEventKindV2.JOB_CREATED,
    "job-status-changed": BatchAuditEventKindV2.JOB_STATUS_CHANGED,
    "item-created": BatchAuditEventKindV2.ITEM_CREATED,
    "item-status-changed": BatchAuditEventKindV2.ITEM_STATUS_CHANGED,
}
_SOURCE_FAMILY_RANK = {
    family: rank
    for rank, family in enumerate(
        (
            "JOB_OUTBOX",
            "ITEM_OUTBOX",
            "WORK_GRAPH",
            "WORK_CONTROL_POLICY",
            "WORK_READINESS",
            "WORK_DISPATCH",
            "WORK_LEASE",
            "STAGE_RUN",
            "WORK_LEASE_EVENT",
            "WORK_RETRY",
            "WORK_CANCELLATION",
            "MODEL_POLICY",
            "MODEL_ADMISSION",
            "MODEL_EVENT",
            "MODEL_BACKPRESSURE",
            "MODEL_CANCELLATION",
            "RESOURCE_POLICY",
            "RESOURCE_ADMISSION",
            "RESOURCE_EVENT",
            "RESOURCE_TERMINATION",
            "STAGE_RESULT",
            "ARTIFACT_RESULT",
            "ATTEMPT_METRICS",
        )
    )
}


class BatchObservabilityService:
    def __init__(self, job_store: JobStore) -> None:
        self.job_store = job_store

    def refresh_job(
        self,
        *,
        job_id: Identifier,
        audit: ContractAudit,
        idempotency_key: Identifier,
    ) -> BatchObservabilityResult:
        with self.job_store._transaction() as connection:
            sources = self._compile_sources(
                connection,
                job_id=job_id,
                audit=audit,
            )
            request_sha256 = _request_sha256(
                {
                    "job_id": job_id,
                    "source_fingerprint": sources.source_fingerprint,
                }
            )
            scope = f"refresh-batch-observability:{job_id}"
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            result = self._materialize(
                connection,
                sources=sources,
                audit=audit,
                require_existing=False,
            )
            if prior is not None:
                if prior[0] != "BATCH_AUDIT_REPORT" or prior[1] != result.report.batch_audit_report_id:
                    raise IdempotencyConflictError("batch observability replay response is corrupt")
                return result
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="BATCH_AUDIT_REPORT",
                response_id=result.report.batch_audit_report_id,
                created_at=self.job_store._clock(),
            )
            return result

    def rebuild_job(self, job_id: Identifier) -> BatchObservabilityResult:
        with self.job_store._transaction() as connection:
            report = self._get_current_report(connection, job_id)
            sources = self._compile_sources(
                connection,
                job_id=job_id,
                audit=report.audit,
            )
            result = self._materialize(
                connection,
                sources=sources,
                audit=report.audit,
                require_existing=True,
            )
            stored_ids = {
                str(row["batch_audit_event_id"])
                for row in connection.execute(
                    """
                    SELECT batch_audit_event_id
                    FROM batch_audit_events
                    WHERE job_id = ?
                    """,
                    (job_id,),
                )
            }
            expected_ids = {event.batch_audit_event_id for event in result.audit_events}
            if stored_ids != expected_ids:
                raise BatchObservabilityIntegrityError("stored audit event projection differs from rebuild")
            if result.report != report:
                raise BatchObservabilityIntegrityError("stored audit report projection differs from rebuild")
            return result

    def list_audit_events(
        self,
        job_id: Identifier,
    ) -> tuple[BatchAuditEventV2, ...]:
        with self.job_store._connect() as connection:
            self.job_store._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            rows = connection.execute(
                """
                SELECT batch_audit_event_id, job_id, source_family,
                       source_id, source_version, occurred_at, record_json
                FROM batch_audit_events
                WHERE job_id = ?
                ORDER BY occurred_at, source_family, source_version, source_id
                """,
                (job_id,),
            ).fetchall()
            return tuple(
                sorted(
                    (self._load_audit_event_row(row) for row in rows),
                    key=self._event_key,
                )
            )

    def list_attempt_metrics(
        self,
        job_id: Identifier,
    ) -> tuple[WorkAttemptMetricsV2, ...]:
        return self.job_store.list_work_attempt_metrics(job_id=job_id)

    def get_item_metrics(
        self,
        item_id: Identifier,
    ) -> ItemMetricsSnapshotV2:
        with self.job_store._connect() as connection:
            item = self.job_store._get_record(
                connection,
                "items",
                "item_id",
                item_id,
                ItemRecord,
            )
            report = self._get_current_report(connection, item.job_id)
            for ref in report.item_metrics_refs:
                row = connection.execute(
                    """
                    SELECT item_metrics_snapshot_id, job_id, item_id,
                           source_fingerprint, record_json
                    FROM item_metrics_snapshots
                    WHERE item_metrics_snapshot_id = ?
                    """,
                    (ref.object_id,),
                ).fetchone()
                if row is not None and str(row["item_id"]) == item_id:
                    return self._load_item_snapshot_row(row)
            raise RecordNotFoundError(f"current ItemMetricsSnapshotV2 not found: {item_id}")

    def get_batch_metrics(
        self,
        job_id: Identifier,
    ) -> BatchMetricsSnapshotV2:
        with self.job_store._connect() as connection:
            report = self._get_current_report(connection, job_id)
            row = connection.execute(
                """
                SELECT batch_metrics_snapshot_id, job_id,
                       source_fingerprint, record_json
                FROM batch_metrics_snapshots
                WHERE batch_metrics_snapshot_id = ?
                """,
                (report.batch_metrics_ref.object_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"current BatchMetricsSnapshotV2 not found: {job_id}")
            return self._load_batch_snapshot_row(row)

    def get_audit_report(
        self,
        job_id: Identifier,
    ) -> BatchAuditReportV2:
        with self.job_store._connect() as connection:
            return self._get_current_report(connection, job_id)

    def projection_is_current(
        self,
        job_id: Identifier,
    ) -> bool:
        with self.job_store._transaction() as connection:
            report = self._get_current_report(connection, job_id)
            sources = self._compile_sources(
                connection,
                job_id=job_id,
                audit=report.audit,
            )
            return sources.source_fingerprint == report.source_fingerprint

    def _compile_sources(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: Identifier,
        audit: ContractAudit,
    ) -> _CompiledSources:
        graph = self._get_graph(connection, job_id)
        units = self._get_units(connection, job_id)
        unit_by_id = {unit.resolved_work_unit_id: unit for unit in units}
        items = self._get_items(connection, job_id)
        item_ids = {item.item_id for item in items}
        stage_runs = self._get_stage_runs(connection, job_id)
        stage_run_by_id = {stage.stage_run_id: stage for stage in stage_runs}
        outbox = self._get_outbox(
            connection,
            job_id=job_id,
            item_ids=item_ids,
            stage_run_ids=set(stage_run_by_id),
        )
        attempt_metrics = self._get_attempt_metrics(connection, job_id)
        metrics_by_lease = {value.work_lease_ref.object_id: value for value in attempt_metrics}
        findings: list[BatchAuditFindingV2] = []
        events: list[BatchAuditEventV2] = []
        source_refs: list[ObjectRef] = [
            resolved_job_work_graph_v2_ref(graph),
            *(resolved_work_unit_v2_ref(unit) for unit in units),
            *(item_record_ref(item) for item in items),
            *(self._outbox_ref(value) for value in outbox),
            *(work_attempt_metrics_v2_ref(value) for value in attempt_metrics),
        ]

        self._compile_outbox_state_events(
            events,
            outbox=outbox,
            items={item.item_id: item for item in items},
            stages=stage_run_by_id,
            audit=audit,
            job_id=job_id,
        )
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        graph_outbox = self._match_outbox(
            outbox,
            event_types=("job-work-graph-created",),
            attribute=("work-graph-id", graph.resolved_job_work_graph_id),
            source_ref=graph_ref,
            findings=findings,
        )
        events.append(
            self._event(
                job_id=job_id,
                item_id=None,
                work_unit=None,
                attempt=None,
                event_kind=BatchAuditEventKindV2.WORK_GRAPH_CREATED,
                status=None,
                reason_code=None,
                occurred_at=(graph_outbox.created_at if graph_outbox is not None else graph.audit.created_at),
                source_family="WORK_GRAPH",
                source_version=0,
                source_ref=graph_ref,
                outbox=graph_outbox,
                related_refs=tuple(resolved_work_unit_v2_ref(unit) for unit in units),
                result_refs=(),
                audit=audit,
            )
        )

        work_policies = self._get_records(
            connection,
            """
            SELECT record_json
            FROM work_control_policies
            WHERE resolved_job_work_graph_id = ?
            """,
            (graph.resolved_job_work_graph_id,),
            WorkControlPolicyV2,
        )
        for work_policy in work_policies:
            source_ref = work_control_policy_v2_ref(work_policy)
            source_refs.append(source_ref)
            witness = self._match_outbox(
                outbox,
                event_types=("work-control-policy-bound",),
                attribute=(
                    "policy-id",
                    work_policy.work_control_policy_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=None,
                    work_unit=None,
                    attempt=None,
                    event_kind=(BatchAuditEventKindV2.WORK_CONTROL_POLICY_BOUND),
                    status="BOUND",
                    reason_code=None,
                    occurred_at=(witness.created_at if witness is not None else work_policy.audit.created_at),
                    source_family="WORK_CONTROL_POLICY",
                    source_version=0,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=work_policy.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )
        readiness_values = self._get_records(
            connection,
            """
            SELECT readiness.record_json
            FROM work_readiness_snapshots AS readiness
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id =
                 readiness.resolved_work_unit_id
            WHERE units.job_id = ?
            """,
            (job_id,),
            WorkReadinessSnapshotV2,
        )
        for readiness in readiness_values:
            source_ref = work_readiness_snapshot_v2_ref(readiness)
            source_refs.append(source_ref)
            readiness_unit = unit_by_id[readiness.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=("work-readiness-recorded",),
                attribute=(
                    "snapshot-id",
                    readiness.work_readiness_snapshot_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=readiness_unit.item_id,
                    work_unit=readiness_unit,
                    attempt=None,
                    event_kind=(BatchAuditEventKindV2.WORK_READINESS_RECORDED),
                    status=readiness.readiness.value,
                    reason_code=None,
                    occurred_at=(witness.created_at if witness is not None else readiness.audit.created_at),
                    source_family="WORK_READINESS",
                    source_version=0,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=readiness.audit.input_refs,
                    result_refs=(
                        *readiness.succeeded_dependency_result_refs,
                        *readiness.retryable_dependency_result_refs,
                        *readiness.terminal_non_success_result_refs,
                    ),
                    audit=audit,
                )
            )

        dispatches = self._get_records(
            connection,
            """
            SELECT decisions.record_json
            FROM work_dispatch_decisions AS decisions
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id =
                 decisions.resolved_work_unit_id
            WHERE units.job_id = ?
            """,
            (job_id,),
            WorkDispatchDecisionV2,
        )
        dispatch_by_id = {value.work_dispatch_decision_id: value for value in dispatches}
        for dispatch in dispatches:
            source_ref = work_dispatch_decision_v2_ref(dispatch)
            source_refs.append(source_ref)
            dispatch_unit = unit_by_id[dispatch.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=("work-lease-acquired",),
                attributes=(
                    (
                        "work-unit-id",
                        dispatch_unit.resolved_work_unit_id,
                    ),
                    ("attempt", dispatch.attempt),
                ),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=dispatch_unit.item_id,
                    work_unit=dispatch_unit,
                    attempt=dispatch.attempt,
                    event_kind=BatchAuditEventKindV2.WORK_DISPATCHED,
                    status="DISPATCHED",
                    reason_code=None,
                    occurred_at=(witness.created_at if witness is not None else dispatch.eligible_at),
                    source_family="WORK_DISPATCH",
                    source_version=dispatch.attempt,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=dispatch.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )

        leases = self._get_records(
            connection,
            """
            SELECT leases.record_json
            FROM work_leases AS leases
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id =
                 leases.resolved_work_unit_id
            WHERE units.job_id = ?
            """,
            (job_id,),
            WorkLeaseV2,
        )
        lease_by_id = {value.work_lease_id: value for value in leases}
        lease_by_stage_run = {
            value.stage_run_ref.object_id: value for value in leases if value.stage_run_ref is not None
        }
        for lease in leases:
            source_ref = work_lease_v2_ref(lease)
            source_refs.append(source_ref)
            lease_unit = unit_by_id[lease.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=("work-lease-acquired",),
                attribute=("work-lease-id", lease.work_lease_id),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=lease_unit.item_id,
                    work_unit=lease_unit,
                    attempt=lease.attempt,
                    event_kind=(BatchAuditEventKindV2.WORK_LEASE_ACQUIRED),
                    status="ACTIVE",
                    reason_code=None,
                    occurred_at=lease.acquired_at,
                    source_family="WORK_LEASE",
                    source_version=0,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=lease.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )
            matched_dispatch = dispatch_by_id.get(lease.work_dispatch_decision_ref.object_id)
            if (
                matched_dispatch is None
                or matched_dispatch.work_unit_ref != lease.work_unit_ref
                or matched_dispatch.attempt != lease.attempt
            ):
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )

        for stage_run in stage_runs:
            source_ref = stage_run_record_ref(stage_run)
            source_refs.append(source_ref)
            stage_lease = lease_by_stage_run.get(stage_run.stage_run_id)
            stage_unit = self._unit_for_stage(stage_run, unit_by_id)
            if stage_lease is None or stage_unit is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            witness = self._match_outbox(
                outbox,
                event_types=("work-lease-acquired",),
                attribute=("work-lease-id", stage_lease.work_lease_id),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=stage_run.item_id,
                    work_unit=stage_unit,
                    attempt=stage_run.attempt,
                    event_kind=BatchAuditEventKindV2.STAGE_RUN_CREATED,
                    status="RUNNING",
                    reason_code=None,
                    occurred_at=stage_run.created_at,
                    source_family="STAGE_RUN",
                    source_version=0,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=(work_lease_v2_ref(stage_lease),),
                    result_refs=(),
                    audit=audit,
                )
            )

        lease_events = self._get_records(
            connection,
            """
            SELECT events.record_json
            FROM work_lease_events AS events
            JOIN work_leases AS leases
              ON leases.work_lease_id = events.work_lease_id
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id =
                 leases.resolved_work_unit_id
            WHERE units.job_id = ?
            ORDER BY events.work_lease_id, events.lease_version
            """,
            (job_id,),
            WorkLeaseEventV2,
        )
        terminal_event_by_lease: dict[str, WorkLeaseEventV2] = {}
        versions_by_lease: dict[str, list[int]] = defaultdict(list)
        for lease_event in lease_events:
            source_ref = work_lease_event_v2_ref(lease_event)
            source_refs.append(source_ref)
            event_lease = lease_by_id.get(lease_event.work_lease_ref.object_id)
            if event_lease is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            unit = unit_by_id[event_lease.work_unit_ref.object_id]
            versions_by_lease[event_lease.work_lease_id].append(lease_event.lease_version)
            terminal = lease_event.event_kind is not WorkLeaseEventKindV2.HEARTBEAT
            if terminal:
                terminal_event_by_lease[event_lease.work_lease_id] = lease_event
            witness = self._match_outbox(
                outbox,
                event_types=(
                    "work-lease-heartbeat",
                    "work-lease-closed",
                ),
                attribute=(
                    "lease-event-id",
                    lease_event.work_lease_event_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=event_lease.attempt,
                    event_kind=(
                        BatchAuditEventKindV2.WORK_LEASE_TERMINAL
                        if terminal
                        else BatchAuditEventKindV2.WORK_LEASE_HEARTBEAT
                    ),
                    status=lease_event.event_kind.value,
                    reason_code=(lease_event.failure.code if lease_event.failure is not None else None),
                    occurred_at=lease_event.audit.created_at,
                    source_family="WORK_LEASE_EVENT",
                    source_version=lease_event.lease_version,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=lease_event.audit.input_refs,
                    result_refs=lease_event.result_refs,
                    audit=audit,
                )
            )
        for lease_id, versions in versions_by_lease.items():
            if versions != list(range(1, len(versions) + 1)):
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.LEASE_EVENT_GAP,
                        work_lease_v2_ref(lease_by_id[lease_id]),
                    )
                )
        for lease in leases:
            stored_head = self.job_store._get_work_lease_head(
                connection,
                lease.work_lease_id,
            )
            rebuilt_head = self.job_store._rebuild_work_lease_head(
                connection,
                lease,
            )
            if stored_head != rebuilt_head:
                raise ImmutableResultError("materialized work lease head differs from immutable events")

        retry_values = self._get_records(
            connection,
            """
            SELECT retries.record_json
            FROM work_retry_decisions AS retries
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id =
                 retries.resolved_work_unit_id
            WHERE units.job_id = ?
            """,
            (job_id,),
            WorkRetryDecisionV2,
        )
        for retry in retry_values:
            source_ref = work_retry_decision_v2_ref(retry)
            source_refs.append(source_ref)
            retry_unit = unit_by_id[retry.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=("work-lease-closed",),
                attribute=(
                    "lease-event-id",
                    retry.prior_lease_event_ref.object_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=retry_unit.item_id,
                    work_unit=retry_unit,
                    attempt=retry.completed_attempt,
                    event_kind=BatchAuditEventKindV2.WORK_RETRY_DECIDED,
                    status=retry.decision.value,
                    reason_code=None,
                    occurred_at=retry.audit.created_at,
                    source_family="WORK_RETRY",
                    source_version=retry.completed_attempt,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=retry.audit.input_refs,
                    result_refs=retry.prior_result_refs,
                    audit=audit,
                )
            )

        cancellation_values = self._get_records(
            connection,
            """
            SELECT record_json
            FROM work_cancellations
            WHERE job_id = ?
            """,
            (job_id,),
            WorkCancellationRecordV2,
        )
        for cancellation in cancellation_values:
            source_ref = work_cancellation_record_v2_ref(cancellation)
            source_refs.append(source_ref)
            witness = self._match_outbox(
                outbox,
                event_types=("job-work-cancelled",),
                attribute=(
                    "cancellation-id",
                    cancellation.work_cancellation_record_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=None,
                    work_unit=None,
                    attempt=None,
                    event_kind=BatchAuditEventKindV2.WORK_CANCELLED,
                    status="CANCELLED",
                    reason_code=cancellation.reason_code,
                    occurred_at=cancellation.audit.created_at,
                    source_family="WORK_CANCELLATION",
                    source_version=0,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=cancellation.audit.input_refs,
                    result_refs=cancellation.preserved_result_refs,
                    audit=audit,
                )
            )

        stage_results = self._get_stage_results(connection, job_id)
        for stage_result in stage_results:
            source_ref = stage_result_record_ref(stage_result)
            source_refs.append(source_ref)
            stage = stage_run_by_id[stage_result.stage_run_id]
            stage_unit = self._unit_for_stage(stage, unit_by_id)
            witness = self._match_outbox(
                outbox,
                event_types=("stage-run-completed",),
                attribute=("result-id", stage_result.stage_result_id),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=stage.item_id,
                    work_unit=stage_unit,
                    attempt=stage.attempt,
                    event_kind=(BatchAuditEventKindV2.STAGE_RUN_COMPLETED),
                    status=stage_result.status.value,
                    reason_code=(stage_result.failure.code if stage_result.failure is not None else None),
                    occurred_at=stage_result.created_at,
                    source_family="STAGE_RESULT",
                    source_version=stage_result.stage_run_version,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=(stage_run_record_ref(stage),),
                    result_refs=stage_result.output_refs,
                    audit=audit,
                )
            )

        stage_result_by_run = {value.stage_run_id: value for value in stage_results}
        for lease_id, terminal_event in terminal_event_by_lease.items():
            lease = lease_by_id[lease_id]
            metrics = metrics_by_lease.get(lease_id)
            if metrics is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.MISSING_TERMINAL_METRICS,
                        work_lease_event_v2_ref(terminal_event),
                    )
                )
                continue
            metrics_ref = work_attempt_metrics_v2_ref(metrics)
            source_refs.append(metrics_ref)
            if (
                metrics.work_lease_ref != work_lease_v2_ref(lease)
                or metrics.terminal_lease_version != terminal_event.lease_version
                or metrics.terminal_event_kind is not terminal_event.event_kind
                or metrics.fencing_token != lease.fencing_token
            ):
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.METRICS_BINDING_MISMATCH,
                        metrics_ref,
                        work_lease_event_v2_ref(terminal_event),
                    )
                )
            if lease.stage_run_ref is not None:
                bound_result = stage_result_by_run.get(lease.stage_run_ref.object_id)
                if bound_result is None or bound_result.metrics_ref != metrics_ref:
                    findings.append(
                        self._finding(
                            (BatchAuditFindingCodeV2.METRICS_BINDING_MISMATCH),
                            metrics_ref,
                            lease.stage_run_ref,
                        )
                    )
            unit = unit_by_id[lease.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=("work-lease-closed",),
                attribute=("work-lease-id", lease_id),
                source_ref=metrics_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=metrics.attempt,
                    event_kind=(BatchAuditEventKindV2.ATTEMPT_METRICS_RECORDED),
                    status=metrics.terminal_status.value,
                    reason_code=metrics.reason_code,
                    occurred_at=metrics.completed_at,
                    source_family="ATTEMPT_METRICS",
                    source_version=metrics.terminal_lease_version,
                    source_ref=metrics_ref,
                    outbox=witness,
                    related_refs=(metrics.work_lease_ref,),
                    result_refs=metrics.result_refs,
                    audit=audit,
                )
            )
            for artifact in metrics.artifact_slices:
                artifact_ref = self._artifact_slice_ref(
                    metrics_ref,
                    artifact,
                )
                source_refs.append(artifact_ref)
                related_refs = (
                    metrics_ref,
                    *((artifact.cost.source_ref,) if artifact.cost.source_ref is not None else ()),
                )
                result_refs = (artifact.output_ref,) if artifact.output_ref is not None else ()
                events.append(
                    self._event(
                        job_id=job_id,
                        item_id=unit.item_id,
                        work_unit=unit,
                        attempt=metrics.attempt,
                        event_kind=(BatchAuditEventKindV2.ARTIFACT_RESULT_RECORDED),
                        status=artifact.status,
                        reason_code=(None if artifact.status == "SUCCEEDED" else artifact.status),
                        occurred_at=metrics.completed_at,
                        source_family="ARTIFACT_RESULT",
                        source_version=metrics.terminal_lease_version,
                        source_ref=artifact_ref,
                        outbox=witness,
                        related_refs=related_refs,
                        result_refs=result_refs,
                        audit=audit,
                        artifact_id=artifact.artifact_id,
                    )
                )
        for lease_id, metrics in metrics_by_lease.items():
            if lease_id not in terminal_event_by_lease:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.UNEXPECTED_SOURCE_RECORD,
                        work_attempt_metrics_v2_ref(metrics),
                    )
                )

        self._compile_model_sources(
            connection,
            job_id=job_id,
            audit=audit,
            outbox=outbox,
            unit_by_id=unit_by_id,
            lease_by_id=lease_by_id,
            terminal_event_by_lease=terminal_event_by_lease,
            metrics_by_lease=metrics_by_lease,
            events=events,
            findings=findings,
            source_refs=source_refs,
        )
        self._compile_resource_sources(
            connection,
            job_id=job_id,
            audit=audit,
            outbox=outbox,
            unit_by_id=unit_by_id,
            lease_by_id=lease_by_id,
            terminal_event_by_lease=terminal_event_by_lease,
            metrics_by_lease=metrics_by_lease,
            events=events,
            findings=findings,
            source_refs=source_refs,
        )
        source_fingerprint = self._fingerprint(source_refs)
        return _CompiledSources(
            graph=graph,
            units=units,
            items=items,
            attempt_metrics=attempt_metrics,
            events=tuple(sorted(events, key=self._event_key)),
            findings=self._unique_findings(findings),
            source_fingerprint=source_fingerprint,
        )

    def _compile_model_sources(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        audit: ContractAudit,
        outbox: tuple[OutboxEvent, ...],
        unit_by_id: dict[str, ResolvedWorkUnitV2],
        lease_by_id: dict[str, WorkLeaseV2],
        terminal_event_by_lease: dict[str, WorkLeaseEventV2],
        metrics_by_lease: dict[str, WorkAttemptMetricsV2],
        events: list[BatchAuditEventV2],
        findings: list[BatchAuditFindingV2],
        source_refs: list[ObjectRef],
    ) -> None:
        policies = self._get_records(
            connection,
            """
            SELECT record_json
            FROM job_model_policies
            WHERE job_id = ?
            """,
            (job_id,),
            JobModelPolicyV2,
        )
        model_control = ModelControlService(self.job_store)
        for policy in policies:
            model_control._get_job_head(connection, policy)
            for pool_ref in policy.model_rate_pool_policy_refs:
                pool_policy = model_control._get_pool_policy(
                    connection,
                    pool_ref.object_id,
                )
                model_control._get_pool_head(
                    connection,
                    pool_policy,
                )
            source_ref = job_model_policy_v2_ref(policy)
            source_refs.append(source_ref)
            witness = self._match_outbox(
                outbox,
                event_types=("job-model-policy-bound",),
                attribute=("policy-id", policy.job_model_policy_id),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=None,
                    work_unit=None,
                    attempt=None,
                    event_kind=BatchAuditEventKindV2.MODEL_POLICY_BOUND,
                    status="BOUND",
                    reason_code=None,
                    occurred_at=(witness.created_at if witness is not None else policy.audit.created_at),
                    source_family="MODEL_POLICY",
                    source_version=0,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=policy.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )

        demands = self._get_records(
            connection,
            """
            SELECT demands.record_json
            FROM work_model_demands AS demands
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id =
                 demands.resolved_work_unit_id
            WHERE units.job_id = ?
            """,
            (job_id,),
            WorkModelDemandV2,
        )
        demand_by_id = {value.work_model_demand_id: value for value in demands}
        source_refs.extend(work_model_demand_v2_ref(value) for value in demands)
        decisions = self._get_records(
            connection,
            """
            SELECT decisions.record_json
            FROM model_admission_decisions AS decisions
            JOIN work_model_demands AS demands
              ON demands.work_model_demand_id =
                 decisions.work_model_demand_id
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id =
                 demands.resolved_work_unit_id
            WHERE units.job_id = ?
            """,
            (job_id,),
            ModelAdmissionDecisionV2,
        )
        admission_witness_by_id: dict[str, OutboxEvent | None] = {}
        for decision in decisions:
            source_ref = model_admission_decision_v2_ref(decision)
            source_refs.append(source_ref)
            demand = demand_by_id.get(decision.work_model_demand_ref.object_id)
            if demand is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            unit = unit_by_id[demand.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=(
                    "model-admission-decided",
                    "combined-control-admission-decided",
                ),
                attribute=(
                    "model-admission-decision-id",
                    decision.model_admission_decision_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            admission_witness_by_id[decision.model_admission_decision_id] = witness
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=demand.attempt,
                    event_kind=(BatchAuditEventKindV2.MODEL_ADMISSION_DECIDED),
                    status=decision.outcome.value,
                    reason_code=(
                        decision.constrained_dimensions[0].value if decision.constrained_dimensions else None
                    ),
                    occurred_at=decision.decided_at,
                    source_family="MODEL_ADMISSION",
                    source_version=demand.attempt,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=decision.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )

        reservations = self._get_records(
            connection,
            """
            SELECT reservations.record_json
            FROM work_model_reservations AS reservations
            WHERE reservations.job_id = ?
            """,
            (job_id,),
            WorkModelReservationV2,
        )
        reservation_by_id = {value.work_model_reservation_id: value for value in reservations}
        reservation_by_lease = {value.work_lease_ref.object_id: value for value in reservations}
        for reservation in reservations:
            model_control._get_reservation_head(
                connection,
                reservation,
            )
        source_refs.extend(work_model_reservation_v2_ref(value) for value in reservations)
        model_events = self._get_records(
            connection,
            """
            SELECT events.record_json
            FROM work_model_events AS events
            JOIN work_model_reservations AS reservations
              ON reservations.work_model_reservation_id =
                 events.work_model_reservation_id
            WHERE reservations.job_id = ?
            ORDER BY events.work_model_reservation_id,
                     events.reservation_version
            """,
            (job_id,),
            WorkModelEventV2,
        )
        events_by_reservation: dict[str, list[WorkModelEventV2]] = defaultdict(list)
        for model_event in model_events:
            source_ref = work_model_event_v2_ref(model_event)
            source_refs.append(source_ref)
            event_reservation = reservation_by_id.get(model_event.work_model_reservation_ref.object_id)
            if event_reservation is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            events_by_reservation[event_reservation.work_model_reservation_id].append(model_event)
            demand = demand_by_id.get(event_reservation.work_model_demand_ref.object_id)
            lease = lease_by_id.get(event_reservation.work_lease_ref.object_id)
            if demand is None or lease is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            unit = unit_by_id[demand.work_unit_ref.object_id]
            if model_event.event_kind is WorkModelEventKindV2.ADMITTED:
                witness = admission_witness_by_id.get(reservation.model_admission_decision_ref.object_id)
                if witness is None:
                    findings.append(
                        self._finding(
                            (BatchAuditFindingCodeV2.MISSING_OUTBOX_WITNESS),
                            source_ref,
                        )
                    )
            elif model_event.work_lease_event_ref is not None:
                witness = self._match_outbox(
                    outbox,
                    event_types=("work-lease-closed",),
                    attribute=(
                        "lease-event-id",
                        model_event.work_lease_event_ref.object_id,
                    ),
                    source_ref=source_ref,
                    findings=findings,
                )
            else:
                witness = None
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=demand.attempt,
                    event_kind=(BatchAuditEventKindV2.MODEL_USAGE_RECORDED),
                    status=model_event.event_kind.value,
                    reason_code=None,
                    occurred_at=model_event.effective_at,
                    source_family="MODEL_EVENT",
                    source_version=model_event.reservation_version,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=model_event.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )
        for reservation_id, values in events_by_reservation.items():
            versions = [value.reservation_version for value in values]
            if versions != list(range(len(versions))):
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.RESERVATION_EVENT_GAP,
                        work_model_reservation_v2_ref(reservation_by_id[reservation_id]),
                    )
                )

        backpressure_values = self._get_records(
            connection,
            """
            SELECT records.record_json
            FROM provider_backpressure_records AS records
            JOIN work_model_reservations AS reservations
              ON reservations.work_model_reservation_id =
                 records.work_model_reservation_id
            WHERE reservations.job_id = ?
            """,
            (job_id,),
            ProviderBackpressureRecordV2,
        )
        for backpressure in backpressure_values:
            source_ref = provider_backpressure_record_v2_ref(backpressure)
            source_refs.append(source_ref)
            backpressure_reservation = reservation_by_id.get(
                backpressure.work_model_reservation_ref.object_id
            )
            if backpressure_reservation is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            demand = demand_by_id[backpressure_reservation.work_model_demand_ref.object_id]
            unit = unit_by_id[demand.work_unit_ref.object_id]
            linked = next(
                (
                    value
                    for value in events_by_reservation[reservation.work_model_reservation_id]
                    if value.provider_backpressure_ref == source_ref
                ),
                None,
            )
            witness = (
                self._match_outbox(
                    outbox,
                    event_types=("work-lease-closed",),
                    attribute=(
                        "lease-event-id",
                        linked.work_lease_event_ref.object_id,
                    ),
                    source_ref=source_ref,
                    findings=findings,
                )
                if linked is not None and linked.work_lease_event_ref is not None
                else None
            )
            if linked is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.MISSING_MODEL_EVENT,
                        source_ref,
                    )
                )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=demand.attempt,
                    event_kind=(BatchAuditEventKindV2.MODEL_BACKPRESSURE_RECORDED),
                    status=backpressure.kind.value,
                    reason_code=backpressure.kind.value,
                    occurred_at=backpressure.observed_at,
                    source_family="MODEL_BACKPRESSURE",
                    source_version=(linked.reservation_version if linked is not None else 0),
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=backpressure.audit.input_refs,
                    result_refs=(backpressure.facade_receipt_ref,),
                    audit=audit,
                )
            )

        cancellation_requests = self._get_records(
            connection,
            """
            SELECT requests.record_json
            FROM work_model_cancellation_requests AS requests
            JOIN work_model_reservations AS reservations
              ON reservations.work_model_reservation_id =
                 requests.work_model_reservation_id
            WHERE reservations.job_id = ?
            """,
            (job_id,),
            WorkModelCancellationRequestV2,
        )
        request_by_id = {value.work_model_cancellation_request_id: value for value in cancellation_requests}
        for request in cancellation_requests:
            source_ref = work_model_cancellation_request_v2_ref(request)
            source_refs.append(source_ref)
            reservation = reservation_by_id[request.work_model_reservation_ref.object_id]
            demand = demand_by_id[reservation.work_model_demand_ref.object_id]
            unit = unit_by_id[demand.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=("work-lease-closed",),
                attribute=(
                    "lease-event-id",
                    request.work_lease_event_ref.object_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=demand.attempt,
                    event_kind=(BatchAuditEventKindV2.MODEL_CANCELLATION_RECORDED),
                    status="REQUESTED",
                    reason_code=request.reason.value,
                    occurred_at=request.requested_at,
                    source_family="MODEL_CANCELLATION",
                    source_version=1,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=request.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )
        cancellation_results = self._get_records(
            connection,
            """
            SELECT results.record_json
            FROM work_model_cancellation_results AS results
            JOIN work_model_cancellation_requests AS requests
              ON requests.work_model_cancellation_request_id =
                 results.work_model_cancellation_request_id
            JOIN work_model_reservations AS reservations
              ON reservations.work_model_reservation_id =
                 requests.work_model_reservation_id
            WHERE reservations.job_id = ?
            """,
            (job_id,),
            WorkModelCancellationResultV2,
        )
        for result in cancellation_results:
            source_ref = work_model_cancellation_result_v2_ref(result)
            source_refs.append(source_ref)
            matched_request = request_by_id.get(result.cancellation_request_ref.object_id)
            if matched_request is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            reservation = reservation_by_id[matched_request.work_model_reservation_ref.object_id]
            demand = demand_by_id[reservation.work_model_demand_ref.object_id]
            unit = unit_by_id[demand.work_unit_ref.object_id]
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=demand.attempt,
                    event_kind=(BatchAuditEventKindV2.MODEL_CANCELLATION_RECORDED),
                    status=result.outcome.value,
                    reason_code=None,
                    occurred_at=result.completed_at,
                    source_family="MODEL_CANCELLATION",
                    source_version=2,
                    source_ref=source_ref,
                    outbox=None,
                    related_refs=result.audit.input_refs,
                    result_refs=(result.facade_cancellation_result_ref,),
                    audit=audit,
                )
            )

        for lease_id, reservation in reservation_by_lease.items():
            terminal = terminal_event_by_lease.get(lease_id)
            if terminal is None:
                continue
            terminal_ref = work_lease_event_v2_ref(terminal)
            matching = tuple(
                value
                for value in events_by_reservation[reservation.work_model_reservation_id]
                if value.work_lease_event_ref == terminal_ref
            )
            if not matching:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.MISSING_MODEL_EVENT,
                        terminal_ref,
                        work_model_reservation_v2_ref(reservation),
                    )
                )
                continue
            metrics = metrics_by_lease.get(lease_id)
            first_terminal = min(
                matching,
                key=lambda value: value.reservation_version,
            )
            if metrics is not None and metrics.model_usage != first_terminal.usage:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.METRICS_BINDING_MISMATCH,
                        work_attempt_metrics_v2_ref(metrics),
                        work_model_event_v2_ref(first_terminal),
                    )
                )
        for lease_id, metrics in metrics_by_lease.items():
            if metrics.model_usage is not None and lease_id not in reservation_by_lease:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.UNEXPECTED_SOURCE_RECORD,
                        work_attempt_metrics_v2_ref(metrics),
                    )
                )

    def _compile_resource_sources(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        audit: ContractAudit,
        outbox: tuple[OutboxEvent, ...],
        unit_by_id: dict[str, ResolvedWorkUnitV2],
        lease_by_id: dict[str, WorkLeaseV2],
        terminal_event_by_lease: dict[str, WorkLeaseEventV2],
        metrics_by_lease: dict[str, WorkAttemptMetricsV2],
        events: list[BatchAuditEventV2],
        findings: list[BatchAuditFindingV2],
        source_refs: list[ObjectRef],
    ) -> None:
        policies = self._get_records(
            connection,
            """
            SELECT record_json
            FROM job_resource_policies
            WHERE job_id = ?
            """,
            (job_id,),
            JobResourcePolicyV2,
        )
        resource_control = ResourceControlService(self.job_store)
        for policy in policies:
            resource_control._get_pool_head(
                connection,
                policy.resource_pool_policy_ref.object_id,
            )
            resource_control._get_job_head(
                connection,
                policy.job_resource_policy_id,
            )
            source_ref = job_resource_policy_v2_ref(policy)
            source_refs.append(source_ref)
            witness = self._match_outbox(
                outbox,
                event_types=("job-resource-policy-bound",),
                attribute=("policy-id", policy.job_resource_policy_id),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=None,
                    work_unit=None,
                    attempt=None,
                    event_kind=(BatchAuditEventKindV2.RESOURCE_POLICY_BOUND),
                    status="BOUND",
                    reason_code=None,
                    occurred_at=(witness.created_at if witness is not None else policy.audit.created_at),
                    source_family="RESOURCE_POLICY",
                    source_version=0,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=policy.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )

        demands = self._get_records(
            connection,
            """
            SELECT demands.record_json
            FROM work_resource_demands AS demands
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id =
                 demands.resolved_work_unit_id
            WHERE units.job_id = ?
            """,
            (job_id,),
            WorkResourceDemandV2,
        )
        demand_by_id = {value.work_resource_demand_id: value for value in demands}
        source_refs.extend(work_resource_demand_v2_ref(value) for value in demands)
        decisions = self._get_records(
            connection,
            """
            SELECT decisions.record_json
            FROM resource_admission_decisions AS decisions
            JOIN work_resource_demands AS demands
              ON demands.work_resource_demand_id =
                 decisions.work_resource_demand_id
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id =
                 demands.resolved_work_unit_id
            WHERE units.job_id = ?
            """,
            (job_id,),
            ResourceAdmissionDecisionV2,
        )
        admission_witness_by_id: dict[str, OutboxEvent | None] = {}
        for decision in decisions:
            source_ref = resource_admission_decision_v2_ref(decision)
            source_refs.append(source_ref)
            demand = demand_by_id.get(decision.work_resource_demand_ref.object_id)
            if demand is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            unit = unit_by_id[demand.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=(
                    "resource-admission-decided",
                    "combined-control-admission-decided",
                ),
                attribute=(
                    "resource-admission-decision-id",
                    decision.resource_admission_decision_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            admission_witness_by_id[decision.resource_admission_decision_id] = witness
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=demand.attempt,
                    event_kind=(BatchAuditEventKindV2.RESOURCE_ADMISSION_DECIDED),
                    status=decision.outcome.value,
                    reason_code=(
                        decision.constrained_resources[0].value if decision.constrained_resources else None
                    ),
                    occurred_at=decision.decided_at,
                    source_family="RESOURCE_ADMISSION",
                    source_version=demand.attempt,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=decision.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )

        reservations = self._get_records(
            connection,
            """
            SELECT record_json
            FROM work_resource_reservations
            WHERE job_id = ?
            """,
            (job_id,),
            WorkResourceReservationV2,
        )
        reservation_by_id = {value.work_resource_reservation_id: value for value in reservations}
        reservation_by_lease = {value.work_lease_ref.object_id: value for value in reservations}
        for reservation in reservations:
            resource_control._get_reservation_head(
                connection,
                reservation.work_resource_reservation_id,
            )
        source_refs.extend(work_resource_reservation_v2_ref(value) for value in reservations)
        resource_events = self._get_records(
            connection,
            """
            SELECT events.record_json
            FROM work_resource_events AS events
            JOIN work_resource_reservations AS reservations
              ON reservations.work_resource_reservation_id =
                 events.work_resource_reservation_id
            WHERE reservations.job_id = ?
            ORDER BY events.work_resource_reservation_id,
                     events.reservation_version
            """,
            (job_id,),
            WorkResourceEventV2,
        )
        events_by_reservation: dict[
            str,
            list[WorkResourceEventV2],
        ] = defaultdict(list)
        for resource_event in resource_events:
            source_ref = work_resource_event_v2_ref(resource_event)
            source_refs.append(source_ref)
            event_reservation = reservation_by_id.get(resource_event.work_resource_reservation_ref.object_id)
            if event_reservation is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            events_by_reservation[event_reservation.work_resource_reservation_id].append(resource_event)
            demand = demand_by_id.get(event_reservation.work_resource_demand_ref.object_id)
            lease = lease_by_id.get(event_reservation.work_lease_ref.object_id)
            if demand is None or lease is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            unit = unit_by_id[demand.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=("work-lease-closed",),
                attribute=(
                    "lease-event-id",
                    resource_event.work_lease_event_ref.object_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=demand.attempt,
                    event_kind=(BatchAuditEventKindV2.RESOURCE_USAGE_RECORDED),
                    status=resource_event.event_kind.value,
                    reason_code=None,
                    occurred_at=resource_event.audit.created_at,
                    source_family="RESOURCE_EVENT",
                    source_version=resource_event.reservation_version,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=resource_event.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )
        for reservation_id, values in events_by_reservation.items():
            versions = [value.reservation_version for value in values]
            if versions != list(range(1, len(versions) + 1)):
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.RESERVATION_EVENT_GAP,
                        work_resource_reservation_v2_ref(reservation_by_id[reservation_id]),
                    )
                )

        termination_requests = self._get_records(
            connection,
            """
            SELECT requests.record_json
            FROM work_resource_termination_requests AS requests
            JOIN work_resource_reservations AS reservations
              ON reservations.work_resource_reservation_id =
                 requests.work_resource_reservation_id
            WHERE reservations.job_id = ?
            """,
            (job_id,),
            WorkResourceTerminationRequestV2,
        )
        request_by_id = {value.work_resource_termination_request_id: value for value in termination_requests}
        for request in termination_requests:
            source_ref = work_resource_termination_request_v2_ref(request)
            source_refs.append(source_ref)
            reservation = reservation_by_id[request.work_resource_reservation_ref.object_id]
            demand = demand_by_id[reservation.work_resource_demand_ref.object_id]
            unit = unit_by_id[demand.work_unit_ref.object_id]
            witness = self._match_outbox(
                outbox,
                event_types=("work-lease-closed",),
                attribute=(
                    "lease-event-id",
                    request.work_lease_event_ref.object_id,
                ),
                source_ref=source_ref,
                findings=findings,
            )
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=demand.attempt,
                    event_kind=(BatchAuditEventKindV2.RESOURCE_TERMINATION_RECORDED),
                    status="REQUESTED",
                    reason_code=request.reason.value,
                    occurred_at=request.requested_at,
                    source_family="RESOURCE_TERMINATION",
                    source_version=1,
                    source_ref=source_ref,
                    outbox=witness,
                    related_refs=request.audit.input_refs,
                    result_refs=(),
                    audit=audit,
                )
            )
        termination_results = self._get_records(
            connection,
            """
            SELECT results.record_json
            FROM work_resource_termination_results AS results
            JOIN work_resource_termination_requests AS requests
              ON requests.work_resource_termination_request_id =
                 results.work_resource_termination_request_id
            JOIN work_resource_reservations AS reservations
              ON reservations.work_resource_reservation_id =
                 requests.work_resource_reservation_id
            WHERE reservations.job_id = ?
            """,
            (job_id,),
            WorkResourceTerminationResultV2,
        )
        for result in termination_results:
            source_ref = work_resource_termination_result_v2_ref(result)
            source_refs.append(source_ref)
            matched_request = request_by_id.get(result.termination_request_ref.object_id)
            if matched_request is None:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.CROSS_SCOPE_REFERENCE,
                        source_ref,
                    )
                )
                continue
            reservation = reservation_by_id[matched_request.work_resource_reservation_ref.object_id]
            demand = demand_by_id[reservation.work_resource_demand_ref.object_id]
            unit = unit_by_id[demand.work_unit_ref.object_id]
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=unit.item_id,
                    work_unit=unit,
                    attempt=demand.attempt,
                    event_kind=(BatchAuditEventKindV2.RESOURCE_TERMINATION_RECORDED),
                    status=result.outcome.value,
                    reason_code=None,
                    occurred_at=result.completed_at,
                    source_family="RESOURCE_TERMINATION",
                    source_version=2,
                    source_ref=source_ref,
                    outbox=None,
                    related_refs=result.audit.input_refs,
                    result_refs=(result.facade_termination_result_ref,),
                    audit=audit,
                )
            )

        for lease_id, reservation in reservation_by_lease.items():
            terminal = terminal_event_by_lease.get(lease_id)
            if terminal is None:
                continue
            terminal_ref = work_lease_event_v2_ref(terminal)
            matching = tuple(
                value
                for value in events_by_reservation[reservation.work_resource_reservation_id]
                if value.work_lease_event_ref == terminal_ref
            )
            if not matching:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.MISSING_RESOURCE_EVENT,
                        terminal_ref,
                        work_resource_reservation_v2_ref(reservation),
                    )
                )
                continue
            metrics = metrics_by_lease.get(lease_id)
            first_terminal = min(
                matching,
                key=lambda value: value.reservation_version,
            )
            if metrics is not None and metrics.resource_usage != first_terminal.charged_usage:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.METRICS_BINDING_MISMATCH,
                        work_attempt_metrics_v2_ref(metrics),
                        work_resource_event_v2_ref(first_terminal),
                    )
                )
        for lease_id, metrics in metrics_by_lease.items():
            if metrics.resource_usage is not None and lease_id not in reservation_by_lease:
                findings.append(
                    self._finding(
                        BatchAuditFindingCodeV2.UNEXPECTED_SOURCE_RECORD,
                        work_attempt_metrics_v2_ref(metrics),
                    )
                )

    def _materialize(
        self,
        connection: sqlite3.Connection,
        *,
        sources: _CompiledSources,
        audit: ContractAudit,
        require_existing: bool,
    ) -> BatchObservabilityResult:
        events = tuple(
            self._persist_event(
                connection,
                event,
                require_existing=require_existing,
            )
            for event in sources.events
        )
        stored_event_ids = {
            str(row["batch_audit_event_id"])
            for row in connection.execute(
                """
                SELECT batch_audit_event_id
                FROM batch_audit_events
                WHERE job_id = ?
                """,
                (sources.graph.job_id,),
            )
        }
        expected_event_ids = {value.batch_audit_event_id for value in events}
        if stored_event_ids != expected_event_ids:
            raise BatchObservabilityIntegrityError("stored audit event projection contains unexpected rows")
        event_refs = tuple(batch_audit_event_v2_ref(value) for value in events)
        attempt_refs = tuple(work_attempt_metrics_v2_ref(value) for value in sources.attempt_metrics)
        units_by_item: dict[str, tuple[ResolvedWorkUnitV2, ...]] = {
            item.item_id: tuple(unit for unit in sources.units if unit.item_id == item.item_id)
            for item in sources.items
        }
        item_snapshots: list[ItemMetricsSnapshotV2] = []
        for item in sources.items:
            metrics = tuple(value for value in sources.attempt_metrics if value.item_id == item.item_id)
            item_events = tuple(value for value in events if value.item_id == item.item_id)
            item_refs = (
                item_record_ref(item),
                *(work_attempt_metrics_v2_ref(value) for value in metrics),
                *(batch_audit_event_v2_ref(value) for value in item_events),
            )
            fingerprint = self._fingerprint(item_refs)
            predecessor = self._item_snapshot_tip(
                connection,
                item.item_id,
                source_fingerprint=fingerprint,
            )
            snapshot = ItemMetricsSnapshotV2.create(
                subject_ref=item_record_ref(item),
                predecessor_ref=predecessor,
                source_fingerprint=fingerprint,
                attempt_metrics_refs=tuple(work_attempt_metrics_v2_ref(value) for value in metrics),
                audit_event_refs=tuple(batch_audit_event_v2_ref(value) for value in item_events),
                metric_slices=self._metric_slices(
                    metrics=metrics,
                    events=item_events,
                    units=units_by_item[item.item_id],
                    job_id=sources.graph.job_id,
                    item_id=item.item_id,
                    include_job=False,
                ),
                audit=_observability_audit(audit, item_refs),
            )
            item_snapshots.append(
                self._persist_item_snapshot(
                    connection,
                    job_id=sources.graph.job_id,
                    item_id=item.item_id,
                    snapshot=snapshot,
                    require_existing=require_existing,
                )
            )
        item_refs = tuple(item_metrics_snapshot_v2_ref(value) for value in item_snapshots)
        graph_ref = resolved_job_work_graph_v2_ref(sources.graph)
        batch_predecessor = self._batch_snapshot_tip(
            connection,
            sources.graph.job_id,
            source_fingerprint=sources.source_fingerprint,
        )
        batch_refs = (
            graph_ref,
            *attempt_refs,
            *event_refs,
            *item_refs,
        )
        batch_snapshot = BatchMetricsSnapshotV2.create(
            subject_ref=graph_ref,
            predecessor_ref=batch_predecessor,
            source_fingerprint=sources.source_fingerprint,
            attempt_metrics_refs=attempt_refs,
            audit_event_refs=event_refs,
            item_metrics_refs=item_refs,
            metric_slices=self._metric_slices(
                metrics=sources.attempt_metrics,
                events=events,
                units=sources.units,
                job_id=sources.graph.job_id,
                item_id=None,
                include_job=True,
            ),
            audit=_observability_audit(audit, batch_refs),
        )
        batch_snapshot = self._persist_batch_snapshot(
            connection,
            job_id=sources.graph.job_id,
            snapshot=batch_snapshot,
            require_existing=require_existing,
        )
        batch_ref = batch_metrics_snapshot_v2_ref(batch_snapshot)
        findings = sources.findings
        report_refs = (
            graph_ref,
            *(resolved_work_unit_v2_ref(unit) for unit in sources.units),
            *event_refs,
            *attempt_refs,
            *item_refs,
            batch_ref,
            *(ref for finding in findings for ref in finding.subject_refs),
        )
        report = BatchAuditReportV2.create(
            resolved_job_work_graph_ref=graph_ref,
            source_fingerprint=sources.source_fingerprint,
            outcome=(BatchAuditOutcomeV2.COMPLETE if not findings else BatchAuditOutcomeV2.INCOMPLETE),
            expected_work_unit_refs=tuple(resolved_work_unit_v2_ref(unit) for unit in sources.units),
            audit_event_refs=event_refs,
            attempt_metrics_refs=attempt_refs,
            item_metrics_refs=item_refs,
            batch_metrics_ref=batch_ref,
            findings=findings,
            audit=_observability_audit(audit, report_refs),
        )
        report = self._persist_report(
            connection,
            job_id=sources.graph.job_id,
            report=report,
            require_existing=require_existing,
        )
        return BatchObservabilityResult(
            audit_events=events,
            attempt_metrics=sources.attempt_metrics,
            item_metrics=tuple(item_snapshots),
            batch_metrics=batch_snapshot,
            report=report,
        )

    def _metric_slices(
        self,
        *,
        metrics: tuple[WorkAttemptMetricsV2, ...],
        events: tuple[BatchAuditEventV2, ...],
        units: tuple[ResolvedWorkUnitV2, ...],
        job_id: str,
        item_id: str | None,
        include_job: bool,
    ) -> tuple[MetricSliceV2, ...]:
        groups: dict[
            tuple[MetricScopeV2, str],
            list[WorkAttemptMetricsV2],
        ] = defaultdict(list)
        grouped_metric_ids: dict[
            tuple[MetricScopeV2, str],
            set[str],
        ] = defaultdict(set)
        unit_counts: dict[tuple[MetricScopeV2, str], set[str]] = defaultdict(set)

        def add_metric(
            key: tuple[MetricScopeV2, str],
            value: WorkAttemptMetricsV2,
        ) -> None:
            if value.work_attempt_metrics_id in grouped_metric_ids[key]:
                return
            grouped_metric_ids[key].add(value.work_attempt_metrics_id)
            groups[key].append(value)

        if include_job:
            key = (MetricScopeV2.JOB, job_id)
            for value in metrics:
                add_metric(key, value)
            unit_counts[key].update(unit.resolved_work_unit_id for unit in units)
        if item_id is not None:
            key = (MetricScopeV2.ITEM, item_id)
            for value in metrics:
                add_metric(key, value)
            unit_counts[key].update(unit.resolved_work_unit_id for unit in units)
        elif include_job:
            for value in metrics:
                if value.item_id is not None:
                    add_metric(
                        (MetricScopeV2.ITEM, value.item_id),
                        value,
                    )
            for unit in units:
                if unit.item_id is not None:
                    unit_counts[(MetricScopeV2.ITEM, unit.item_id)].add(unit.resolved_work_unit_id)
        for unit in units:
            stage_key = (MetricScopeV2.STAGE, unit.stage.value)
            groups[stage_key]
            unit_counts[stage_key].add(unit.resolved_work_unit_id)
        for value in metrics:
            add_metric(
                (MetricScopeV2.STAGE, value.stage.value),
                value,
            )
            unit_counts[(MetricScopeV2.STAGE, value.stage.value)].add(value.work_unit_ref.object_id)
            if value.model_profile_ref is not None:
                key = (
                    MetricScopeV2.MODEL_PROFILE,
                    value.model_profile_ref.object_id,
                )
                add_metric(key, value)
                unit_counts[key].add(value.work_unit_ref.object_id)
            tool_metrics = (
                tuple(tool for artifact in value.artifact_slices for tool in artifact.tool_metrics)
                if value.artifact_slices
                else value.tool_metrics
            )
            for tool in tool_metrics:
                key = (MetricScopeV2.TOOL_FAMILY, tool.tool_family)
                add_metric(key, value)
                unit_counts[key].add(value.work_unit_ref.object_id)
            for artifact in value.artifact_slices:
                key = (MetricScopeV2.ARTIFACT, artifact.artifact_id)
                add_metric(key, value)
                unit_counts[key].add(value.work_unit_ref.object_id)
        return tuple(
            self._metric_slice(
                scope=key[0],
                scope_id=key[1],
                metrics=tuple(values),
                events=self._events_for_metric_slice(
                    events,
                    scope=key[0],
                    scope_id=key[1],
                    metrics=tuple(values),
                ),
                work_units=len(unit_counts[key]),
            )
            for key, values in sorted(
                groups.items(),
                key=lambda item: (
                    item[0][0].value,
                    item[0][1],
                ),
            )
        )

    @staticmethod
    def _metric_slice(
        *,
        scope: MetricScopeV2,
        scope_id: str,
        metrics: tuple[WorkAttemptMetricsV2, ...],
        events: tuple[BatchAuditEventV2, ...],
        work_units: int,
    ) -> MetricSliceV2:
        artifact_scope = scope is MetricScopeV2.ARTIFACT
        model_usage = tuple(
            value.model_usage for value in metrics if not artifact_scope and value.model_usage is not None
        )
        resources = tuple(
            value.resource_usage.observed
            for value in metrics
            if not artifact_scope and value.resource_usage is not None
        )
        artifact_slices = tuple(
            artifact
            for value in metrics
            for artifact in value.artifact_slices
            if (scope is MetricScopeV2.ARTIFACT and artifact.artifact_id == scope_id)
        )
        if artifact_slices:
            succeeded_attempts = sum(value.status == "SUCCEEDED" for value in artifact_slices)
            retryable_attempts = sum(value.status == "RETRYABLE_FAILURE" for value in artifact_slices)
            terminal_non_success_attempts = sum(
                value.status
                in {
                    "BLOCKED_CAPABILITY",
                    "BLOCKED_DEPENDENCY",
                    "BLOCKED_POLICY",
                    "TERMINAL_FAILURE",
                }
                for value in artifact_slices
            )
            cancelled_attempts = 0
            tool_calls = sum(tool.calls for value in artifact_slices for tool in value.tool_metrics)
            reported_cost_microusd = sum(
                value.cost.amount_microusd or 0
                for value in artifact_slices
                if value.cost.availability is MetricAvailabilityV2.REPORTED
            )
            unavailable_cost_attempts = sum(
                value.cost.availability is MetricAvailabilityV2.UNAVAILABLE for value in artifact_slices
            )
        else:
            succeeded_attempts = sum(value.terminal_status is StageRunStatus.SUCCEEDED for value in metrics)
            retryable_attempts = sum(
                value.terminal_status is StageRunStatus.RETRYABLE_FAILURE for value in metrics
            )
            terminal_non_success_attempts = sum(
                value.terminal_status
                in {
                    StageRunStatus.BLOCKED_POLICY,
                    StageRunStatus.BLOCKED_CAPABILITY,
                    StageRunStatus.TERMINAL_FAILURE,
                }
                for value in metrics
            )
            cancelled_attempts = sum(value.terminal_status is StageRunStatus.CANCELLED for value in metrics)
            if scope is MetricScopeV2.TOOL_FAMILY:
                tool_calls = sum(
                    tool.calls
                    for value in metrics
                    for tool in (
                        tuple(
                            artifact_tool
                            for artifact in value.artifact_slices
                            for artifact_tool in artifact.tool_metrics
                        )
                        if value.artifact_slices
                        else value.tool_metrics
                    )
                    if tool.tool_family == scope_id
                )
            else:
                tool_calls = sum(
                    tool.calls
                    for value in metrics
                    for tool in (
                        tuple(
                            artifact_tool
                            for artifact in value.artifact_slices
                            for artifact_tool in artifact.tool_metrics
                        )
                        if value.artifact_slices
                        else value.tool_metrics
                    )
                )
            reported_cost_microusd = sum(
                cost.amount_microusd or 0
                for value in metrics
                for cost in (
                    tuple(artifact.cost for artifact in value.artifact_slices)
                    if value.artifact_slices
                    else (value.cost,)
                )
                if cost.availability is MetricAvailabilityV2.REPORTED
            )
            unavailable_cost_attempts = sum(
                any(
                    cost.availability is MetricAvailabilityV2.UNAVAILABLE
                    for cost in (
                        tuple(artifact.cost for artifact in value.artifact_slices)
                        if value.artifact_slices
                        else (value.cost,)
                    )
                )
                for value in metrics
            )
        return MetricSliceV2(
            scope=scope,
            scope_id=scope_id,
            work_units=work_units,
            terminal_attempts=len(metrics),
            succeeded_attempts=succeeded_attempts,
            retryable_attempts=retryable_attempts,
            terminal_non_success_attempts=(terminal_non_success_attempts),
            cancelled_attempts=cancelled_attempts,
            retry_attempts=sum(value.retry_of_attempt_metrics_ref is not None for value in metrics),
            resume_attempts=0,
            model_requests=sum(value.requests for value in model_usage),
            input_tokens=sum(value.input_tokens for value in model_usage),
            output_tokens=sum(value.output_tokens for value in model_usage),
            cache_creation_input_tokens=sum(value.cache_creation_input_tokens for value in model_usage),
            cache_read_input_tokens=sum(value.cache_read_input_tokens for value in model_usage),
            charged_tokens=sum(value.charged_tokens for value in model_usage),
            reported_usage_attempts=sum(value.source is ModelUsageSourceV2.REPORTED for value in model_usage),
            conservative_usage_attempts=sum(
                value.source is ModelUsageSourceV2.CONSERVATIVE_ALLOWANCE for value in model_usage
            ),
            cache_hit_attempts=sum(
                not artifact_scope and value.cache_use is CacheUseV2.HIT for value in metrics
            ),
            cache_miss_attempts=sum(
                not artifact_scope and value.cache_use is CacheUseV2.MISS for value in metrics
            ),
            cache_unavailable_attempts=sum(
                not artifact_scope and value.cache_use is CacheUseV2.UNAVAILABLE for value in metrics
            ),
            process_starts=sum(value.processes for value in resources),
            renderer_operations=sum(value.renderers for value in resources),
            network_requests=sum(value.network_requests for value in resources),
            retained_storage_bytes=sum(value.storage_bytes for value in resources),
            tool_calls=tool_calls,
            reported_cost_microusd=reported_cost_microusd,
            unavailable_cost_attempts=unavailable_cost_attempts,
            backpressure_events=sum(
                value.event_kind is BatchAuditEventKindV2.MODEL_BACKPRESSURE_RECORDED for value in events
            ),
            error_events=sum(value.reason_code is not None for value in events),
            queue_wait_samples_us=tuple(sorted(value.queue_wait_us for value in metrics)),
            execution_samples_us=tuple(sorted(value.execution_duration_us for value in metrics)),
            total_samples_us=tuple(sorted(value.total_duration_us for value in metrics)),
        )

    @staticmethod
    def _events_for_metric_slice(
        events: tuple[BatchAuditEventV2, ...],
        *,
        scope: MetricScopeV2,
        scope_id: str,
        metrics: tuple[WorkAttemptMetricsV2, ...],
    ) -> tuple[BatchAuditEventV2, ...]:
        if scope is MetricScopeV2.JOB:
            return events
        if scope is MetricScopeV2.ITEM:
            return tuple(value for value in events if value.item_id == scope_id)
        if scope is MetricScopeV2.STAGE:
            return tuple(
                value for value in events if value.stage is not None and value.stage.value == scope_id
            )
        if scope is MetricScopeV2.ARTIFACT:
            return tuple(value for value in events if value.artifact_id == scope_id)
        work_unit_ids = {value.work_unit_ref.object_id for value in metrics}
        return tuple(
            value
            for value in events
            if value.work_unit_ref is not None and value.work_unit_ref.object_id in work_unit_ids
        )

    def _compile_outbox_state_events(
        self,
        events: list[BatchAuditEventV2],
        *,
        outbox: tuple[OutboxEvent, ...],
        items: dict[str, ItemRecord],
        stages: dict[str, StageRunRecord],
        audit: ContractAudit,
        job_id: str,
    ) -> None:
        for value in outbox:
            event_kind = _OUTBOX_EVENT_KINDS.get(value.event_type)
            if event_kind is None:
                continue
            source_ref = self._outbox_ref(value)
            item_id = value.aggregate_id if value.aggregate_type == "ITEM" else None
            stage = stages.get(value.aggregate_id) if value.aggregate_type == "STAGE_RUN" else None
            unit = None
            attempt = None
            related_refs: tuple[ObjectRef, ...] = ()
            if item_id is not None and item_id in items:
                related_refs = (item_record_ref(items[item_id]),)
            if stage is not None:
                related_refs = (stage_run_record_ref(stage),)
                attempt = stage.attempt
            raw_status = self._attribute(value, "to-status")
            events.append(
                self._event(
                    job_id=job_id,
                    item_id=item_id or (stage.item_id if stage else None),
                    work_unit=unit,
                    attempt=attempt if unit is not None else None,
                    event_kind=event_kind,
                    status=(raw_status if isinstance(raw_status, str) else None),
                    reason_code=None,
                    occurred_at=value.created_at,
                    source_family=("ITEM_OUTBOX" if value.aggregate_type == "ITEM" else "JOB_OUTBOX"),
                    source_version=value.aggregate_version,
                    source_ref=source_ref,
                    outbox=value,
                    related_refs=related_refs,
                    result_refs=(),
                    audit=audit,
                )
            )

    def _event(
        self,
        *,
        job_id: str,
        item_id: str | None,
        work_unit: ResolvedWorkUnitV2 | None,
        attempt: int | None,
        event_kind: BatchAuditEventKindV2,
        status: str | None,
        reason_code: str | None,
        occurred_at: datetime,
        source_family: str,
        source_version: int,
        source_ref: ObjectRef,
        outbox: OutboxEvent | None,
        related_refs: tuple[ObjectRef, ...],
        result_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
        artifact_id: str | None = None,
    ) -> BatchAuditEventV2:
        refs = (source_ref, *related_refs, *result_refs)
        return BatchAuditEventV2.create(
            job_id=job_id,
            item_id=item_id,
            work_unit_ref=(resolved_work_unit_v2_ref(work_unit) if work_unit is not None else None),
            stage=work_unit.stage if work_unit is not None else None,
            artifact_id=artifact_id,
            attempt=attempt,
            event_kind=event_kind,
            status=status,
            reason_code=reason_code,
            occurred_at=occurred_at,
            source_family=source_family,
            source_version=source_version,
            source_ref=source_ref,
            outbox_event_id=outbox.event_id if outbox else None,
            related_refs=related_refs,
            result_refs=result_refs,
            audit=_observability_audit(audit, refs),
        )

    def _match_outbox(
        self,
        outbox: tuple[OutboxEvent, ...],
        *,
        event_types: tuple[str, ...],
        source_ref: ObjectRef,
        findings: list[BatchAuditFindingV2],
        attribute: tuple[str, object] | None = None,
        attributes: tuple[tuple[str, object], ...] = (),
        aggregate_id: str | None = None,
    ) -> OutboxEvent | None:
        expected = (
            *((attribute,) if attribute is not None else ()),
            *attributes,
        )
        matches = tuple(
            value
            for value in outbox
            if value.event_type in event_types
            and (aggregate_id is None or value.aggregate_id == aggregate_id)
            and all(self._attribute(value, key) == expected_value for key, expected_value in expected)
        )
        if not matches:
            findings.append(
                self._finding(
                    BatchAuditFindingCodeV2.MISSING_OUTBOX_WITNESS,
                    source_ref,
                )
            )
            return None
        if len(matches) > 1:
            findings.append(
                self._finding(
                    BatchAuditFindingCodeV2.DUPLICATE_OUTBOX_WITNESS,
                    source_ref,
                    *(self._outbox_ref(value) for value in matches),
                )
            )
        return sorted(matches, key=lambda value: value.event_id)[0]

    @staticmethod
    def _attribute(event: OutboxEvent, key: str) -> object | None:
        return next(
            (attribute.value for attribute in event.attributes if attribute.key == key),
            None,
        )

    @staticmethod
    def _outbox_ref(value: OutboxEvent) -> ObjectRef:
        encoded = json.dumps(
            canonical_value_v2(
                value.model_dump(
                    mode="python",
                    exclude={"event_id", "published_at"},
                )
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        return ObjectRef(
            object_type="outbox-event",
            object_id=f"outbox-event://sha256/{digest}",
            object_version="record/v1",
            object_sha256=digest,
        )

    @staticmethod
    def _artifact_slice_ref(
        metrics_ref: ObjectRef,
        value: object,
    ) -> ObjectRef:
        payload = canonical_value_v2(
            {
                "work_attempt_metrics_ref": metrics_ref,
                "artifact_metric_slice": value,
            }
        )
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        return ObjectRef(
            object_type="artifact-metric-slice",
            object_id=f"artifact-metric-slice://sha256/{digest}",
            object_version="v2",
            object_sha256=digest,
        )

    @staticmethod
    def _finding(
        code: BatchAuditFindingCodeV2,
        *refs: ObjectRef,
    ) -> BatchAuditFindingV2:
        return BatchAuditFindingV2(
            code=code,
            subject_refs=tuple(sorted(set(refs), key=_ref_key)),
        )

    @staticmethod
    def _unique_findings(
        values: Iterable[BatchAuditFindingV2],
    ) -> tuple[BatchAuditFindingV2, ...]:
        return tuple(
            sorted(
                set(values),
                key=lambda value: (
                    value.code.value,
                    tuple(_ref_key(ref) for ref in value.subject_refs),
                ),
            )
        )

    @staticmethod
    def _fingerprint(refs: Iterable[ObjectRef]) -> str:
        payload = tuple(
            sorted(
                set(refs),
                key=_ref_key,
            )
        )
        encoded = json.dumps(
            canonical_value_v2(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _event_key(
        value: BatchAuditEventV2,
    ) -> tuple[datetime, int, int, str]:
        return (
            value.occurred_at,
            _SOURCE_FAMILY_RANK[value.source_family],
            value.source_version,
            value.source_ref.object_id,
        )

    @staticmethod
    def _unit_for_stage(
        stage: StageRunRecord,
        units: dict[str, ResolvedWorkUnitV2],
    ) -> ResolvedWorkUnitV2 | None:
        return next(
            (
                units[ref.object_id]
                for ref in stage.input_refs
                if ref.object_type == "resolved-work-unit" and ref.object_id in units
            ),
            None,
        )

    def _get_graph(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> ResolvedJobWorkGraphV2:
        row = connection.execute(
            """
            SELECT record_json
            FROM resolved_job_work_graphs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise BatchObservabilityPolicyError(f"planned R6 work graph not found: {job_id}")
        return self._load(row, ResolvedJobWorkGraphV2)

    def _get_units(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[ResolvedWorkUnitV2, ...]:
        return self._get_records(
            connection,
            """
            SELECT record_json
            FROM resolved_work_units
            WHERE job_id = ?
            ORDER BY resolved_work_unit_id
            """,
            (job_id,),
            ResolvedWorkUnitV2,
        )

    def _get_items(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[ItemRecord, ...]:
        return self._get_records(
            connection,
            """
            SELECT record_json
            FROM items
            WHERE job_id = ?
            ORDER BY item_id
            """,
            (job_id,),
            ItemRecord,
        )

    def _get_stage_runs(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[StageRunRecord, ...]:
        return self._get_records(
            connection,
            """
            SELECT record_json
            FROM stage_runs
            WHERE job_id = ?
            ORDER BY stage_run_id
            """,
            (job_id,),
            StageRunRecord,
        )

    def _get_stage_results(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[StageResultRecord, ...]:
        return self._get_records(
            connection,
            """
            SELECT results.record_json
            FROM stage_results AS results
            JOIN stage_runs AS runs
              ON runs.stage_run_id = results.stage_run_id
            WHERE runs.job_id = ?
            ORDER BY results.stage_result_id
            """,
            (job_id,),
            StageResultRecord,
        )

    def _get_outbox(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        item_ids: set[str],
        stage_run_ids: set[str],
    ) -> tuple[OutboxEvent, ...]:
        aggregate_ids = {job_id, *item_ids, *stage_run_ids}
        placeholders = ",".join("?" for _ in aggregate_ids)
        return self._get_records(
            connection,
            f"""
            SELECT record_json
            FROM outbox_events
            WHERE aggregate_id IN ({placeholders})
            ORDER BY event_id
            """,
            tuple(sorted(aggregate_ids)),
            OutboxEvent,
        )

    def _get_attempt_metrics(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[WorkAttemptMetricsV2, ...]:
        rows = connection.execute(
            """
            SELECT work_attempt_metrics_id, job_id,
                   resolved_work_unit_id, work_lease_id,
                   attempt, record_json
            FROM work_attempt_metrics
            WHERE job_id = ?
            ORDER BY resolved_work_unit_id, attempt
            """,
            (job_id,),
        ).fetchall()
        return tuple(
            self.job_store._load_work_attempt_metrics_row(
                connection,
                row,
            )
            for row in rows
        )

    def _get_current_report(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> BatchAuditReportV2:
        rows = connection.execute(
            """
            SELECT batch_audit_report_id, job_id,
                   source_fingerprint, outcome, record_json
            FROM batch_audit_reports
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchall()
        if not rows:
            raise RecordNotFoundError(f"BatchAuditReportV2 not found: {job_id}")
        reports = tuple(self._load_report_row(row) for row in rows)
        stored_event_ids = {
            str(row["batch_audit_event_id"])
            for row in connection.execute(
                """
                SELECT batch_audit_event_id
                FROM batch_audit_events
                WHERE job_id = ?
                """,
                (job_id,),
            )
        }
        matching = tuple(
            report
            for report in reports
            if {ref.object_id for ref in report.audit_event_refs} == stored_event_ids
        )
        if len(matching) == 1:
            return matching[0]
        if len(reports) == 1:
            return reports[0]
        raise BatchObservabilityIntegrityError("current audit report projection is ambiguous")

    def _persist_event(
        self,
        connection: sqlite3.Connection,
        event: BatchAuditEventV2,
        *,
        require_existing: bool,
    ) -> BatchAuditEventV2:
        row = connection.execute(
            """
            SELECT batch_audit_event_id, job_id, source_family,
                   source_id, source_version, occurred_at, record_json
            FROM batch_audit_events
            WHERE job_id = ? AND source_family = ? AND source_id = ?
            """,
            (
                event.job_id,
                event.source_family,
                event.source_ref.object_id,
            ),
        ).fetchone()
        if row is not None:
            stored = self._load_audit_event_row(row)
            if stored.batch_audit_event_id != event.batch_audit_event_id:
                raise BatchObservabilityIntegrityError("stored audit event projection differs from source")
            if require_existing and stored != event:
                raise BatchObservabilityIntegrityError("stored audit event projection differs from rebuild")
            return stored
        if require_existing:
            raise BatchObservabilityIntegrityError("required audit event projection is missing")
        connection.execute(
            """
            INSERT INTO batch_audit_events (
                batch_audit_event_id, job_id, source_family,
                source_id, source_version, occurred_at, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.batch_audit_event_id,
                event.job_id,
                event.source_family,
                event.source_ref.object_id,
                event.source_version,
                event.occurred_at.isoformat(),
                self.job_store._record_json(event),
            ),
        )
        return event

    def _persist_item_snapshot(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        item_id: str,
        snapshot: ItemMetricsSnapshotV2,
        require_existing: bool,
    ) -> ItemMetricsSnapshotV2:
        row = connection.execute(
            """
            SELECT item_metrics_snapshot_id, job_id, item_id,
                   source_fingerprint, record_json
            FROM item_metrics_snapshots
            WHERE item_id = ? AND source_fingerprint = ?
            """,
            (item_id, snapshot.source_fingerprint),
        ).fetchone()
        if row is not None:
            stored = self._load_item_snapshot_row(row)
            if stored.item_metrics_snapshot_id != snapshot.item_metrics_snapshot_id:
                raise BatchObservabilityIntegrityError("stored item metrics projection differs from source")
            if require_existing and stored != snapshot:
                raise BatchObservabilityIntegrityError("stored item metrics projection differs from rebuild")
            return stored
        if require_existing:
            raise BatchObservabilityIntegrityError("required item metrics projection is missing")
        connection.execute(
            """
            INSERT INTO item_metrics_snapshots (
                item_metrics_snapshot_id, job_id, item_id,
                source_fingerprint, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot.item_metrics_snapshot_id,
                job_id,
                item_id,
                snapshot.source_fingerprint,
                self.job_store._record_json(snapshot),
            ),
        )
        return snapshot

    def _persist_batch_snapshot(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        snapshot: BatchMetricsSnapshotV2,
        require_existing: bool,
    ) -> BatchMetricsSnapshotV2:
        row = connection.execute(
            """
            SELECT batch_metrics_snapshot_id, job_id,
                   source_fingerprint, record_json
            FROM batch_metrics_snapshots
            WHERE job_id = ? AND source_fingerprint = ?
            """,
            (job_id, snapshot.source_fingerprint),
        ).fetchone()
        if row is not None:
            stored = self._load_batch_snapshot_row(row)
            if stored.batch_metrics_snapshot_id != snapshot.batch_metrics_snapshot_id:
                raise BatchObservabilityIntegrityError("stored batch metrics projection differs from source")
            if require_existing and stored != snapshot:
                raise BatchObservabilityIntegrityError("stored batch metrics projection differs from rebuild")
            return stored
        if require_existing:
            raise BatchObservabilityIntegrityError("required batch metrics projection is missing")
        connection.execute(
            """
            INSERT INTO batch_metrics_snapshots (
                batch_metrics_snapshot_id, job_id,
                source_fingerprint, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                snapshot.batch_metrics_snapshot_id,
                job_id,
                snapshot.source_fingerprint,
                self.job_store._record_json(snapshot),
            ),
        )
        return snapshot

    def _persist_report(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        report: BatchAuditReportV2,
        require_existing: bool,
    ) -> BatchAuditReportV2:
        row = connection.execute(
            """
            SELECT batch_audit_report_id, job_id,
                   source_fingerprint, outcome, record_json
            FROM batch_audit_reports
            WHERE job_id = ? AND source_fingerprint = ?
            """,
            (job_id, report.source_fingerprint),
        ).fetchone()
        if row is not None:
            stored = self._load_report_row(row)
            if stored.batch_audit_report_id != report.batch_audit_report_id:
                raise BatchObservabilityIntegrityError("stored audit report projection differs from source")
            if require_existing and stored != report:
                raise BatchObservabilityIntegrityError("stored audit report projection differs from rebuild")
            return stored
        if require_existing:
            raise BatchObservabilityIntegrityError("required audit report projection is missing")
        connection.execute(
            """
            INSERT INTO batch_audit_reports (
                batch_audit_report_id, job_id,
                source_fingerprint, outcome, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                report.batch_audit_report_id,
                job_id,
                report.source_fingerprint,
                report.outcome.value,
                self.job_store._record_json(report),
            ),
        )
        return report

    def _item_snapshot_tip(
        self,
        connection: sqlite3.Connection,
        item_id: str,
        *,
        source_fingerprint: str,
    ) -> ObjectRef | None:
        rows = connection.execute(
            """
            SELECT item_metrics_snapshot_id, job_id, item_id,
                   source_fingerprint, record_json
            FROM item_metrics_snapshots
            WHERE item_id = ?
            """,
            (item_id,),
        ).fetchall()
        values = tuple(self._load_item_snapshot_row(row) for row in rows)
        exact = tuple(value for value in values if value.source_fingerprint == source_fingerprint)
        if exact:
            return exact[0].predecessor_ref
        referenced = {
            value.predecessor_ref.object_id for value in values if value.predecessor_ref is not None
        }
        tips = tuple(value for value in values if value.item_metrics_snapshot_id not in referenced)
        if len(tips) > 1:
            raise BatchObservabilityIntegrityError("item metrics snapshot chain has multiple tips")
        return item_metrics_snapshot_v2_ref(tips[0]) if tips else None

    def _batch_snapshot_tip(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        *,
        source_fingerprint: str,
    ) -> ObjectRef | None:
        rows = connection.execute(
            """
            SELECT batch_metrics_snapshot_id, job_id,
                   source_fingerprint, record_json
            FROM batch_metrics_snapshots
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchall()
        values = tuple(self._load_batch_snapshot_row(row) for row in rows)
        exact = tuple(value for value in values if value.source_fingerprint == source_fingerprint)
        if exact:
            return exact[0].predecessor_ref
        referenced = {
            value.predecessor_ref.object_id for value in values if value.predecessor_ref is not None
        }
        tips = tuple(value for value in values if value.batch_metrics_snapshot_id not in referenced)
        if len(tips) > 1:
            raise BatchObservabilityIntegrityError("batch metrics snapshot chain has multiple tips")
        return batch_metrics_snapshot_v2_ref(tips[0]) if tips else None

    @staticmethod
    def _load[RecordT: BaseModel](
        row: sqlite3.Row,
        model_type: type[RecordT],
    ) -> RecordT:
        try:
            return model_type.model_validate_json(str(row["record_json"]))
        except ValidationError as exc:
            raise ImmutableResultError(f"stored {model_type.__name__} is invalid") from exc

    def _get_records[RecordT: BaseModel](
        self,
        connection: sqlite3.Connection,
        query: str,
        parameters: tuple[object, ...],
        model_type: type[RecordT],
    ) -> tuple[RecordT, ...]:
        rows = connection.execute(query, parameters).fetchall()
        try:
            return tuple(model_type.model_validate_json(str(row["record_json"])) for row in rows)
        except ValidationError as exc:
            raise ImmutableResultError(f"stored {model_type.__name__} is invalid") from exc

    @staticmethod
    def _load_audit_event_row(
        row: sqlite3.Row,
    ) -> BatchAuditEventV2:
        try:
            value = BatchAuditEventV2.model_validate_json(str(row["record_json"]))
        except ValidationError as exc:
            raise BatchObservabilityIntegrityError("stored audit event projection is invalid") from exc
        if (
            str(row["batch_audit_event_id"]) != value.batch_audit_event_id
            or str(row["job_id"]) != value.job_id
            or str(row["source_family"]) != value.source_family
            or str(row["source_id"]) != value.source_ref.object_id
            or int(row["source_version"]) != value.source_version
            or str(row["occurred_at"]) != value.occurred_at.isoformat()
        ):
            raise BatchObservabilityIntegrityError("stored audit event columns differ from record")
        return value

    @staticmethod
    def _load_item_snapshot_row(
        row: sqlite3.Row,
    ) -> ItemMetricsSnapshotV2:
        try:
            value = ItemMetricsSnapshotV2.model_validate_json(str(row["record_json"]))
        except ValidationError as exc:
            raise BatchObservabilityIntegrityError("stored item metrics projection is invalid") from exc
        if (
            str(row["item_metrics_snapshot_id"]) != value.item_metrics_snapshot_id
            or str(row["item_id"]) != value.subject_ref.object_id
            or str(row["source_fingerprint"]) != value.source_fingerprint
        ):
            raise BatchObservabilityIntegrityError("stored item metrics columns differ from record")
        return value

    @staticmethod
    def _load_batch_snapshot_row(
        row: sqlite3.Row,
    ) -> BatchMetricsSnapshotV2:
        try:
            value = BatchMetricsSnapshotV2.model_validate_json(str(row["record_json"]))
        except ValidationError as exc:
            raise BatchObservabilityIntegrityError("stored batch metrics projection is invalid") from exc
        if (
            str(row["batch_metrics_snapshot_id"]) != value.batch_metrics_snapshot_id
            or str(row["source_fingerprint"]) != value.source_fingerprint
        ):
            raise BatchObservabilityIntegrityError("stored batch metrics columns differ from record")
        return value

    @staticmethod
    def _load_report_row(row: sqlite3.Row) -> BatchAuditReportV2:
        try:
            value = BatchAuditReportV2.model_validate_json(str(row["record_json"]))
        except ValidationError as exc:
            raise BatchObservabilityIntegrityError("stored audit report projection is invalid") from exc
        if (
            str(row["batch_audit_report_id"]) != value.batch_audit_report_id
            or str(row["source_fingerprint"]) != value.source_fingerprint
            or str(row["outcome"]) != value.outcome.value
        ):
            raise BatchObservabilityIntegrityError("stored audit report columns differ from record")
        return value


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )
