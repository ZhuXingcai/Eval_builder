from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from env_mock_agent.facade import (
    ExecutionTelemetryV2,
    FacadeObjectRef,
    TelemetryAvailabilityV2,
    execution_telemetry_ref,
)
from eval_factory.contracts.canary_execution_v2 import (
    SemanticReviewFanoutV2,
    semantic_review_fanout_v2_ref,
)
from eval_factory.contracts.checkpoint_interaction_v2 import (
    UserCheckpointInteractionStateV2,
    UserCheckpointInteractionV2,
    validate_user_checkpoint_interaction_v2_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    FailureClass,
    FailureRecord,
    Identifier,
    ObjectRef,
    ScalarValue,
    TypedAttribute,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import (
    canonical_json_v2,
    canonical_sha256_v2,
    canonical_value_v2,
)
from eval_factory.contracts.model_control_v2 import (
    ModelAdmissionDecisionV2,
    ModelAdmissionOutcomeV2,
    ModelUsageV2,
    WorkModelDemandV2,
    WorkModelReservationV2,
    model_admission_decision_v2_ref,
    work_model_demand_v2_ref,
)
from eval_factory.contracts.observability_v2 import (
    R6_OBSERVABILITY_POLICY_VERSION,
    ArtifactMetricSliceV2,
    CostObservationV2,
    ToolMetricV2,
    WorkAttemptMetricsV2,
    work_attempt_metrics_v2_ref,
)
from eval_factory.contracts.orchestration import ItemStatus, JobStatus, StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    R6_WORK_CONTROL_FAILURE_MESSAGE,
    R6_WORK_CONTROL_POLICY_VERSION,
    ArtifactGroupFanoutV2,
    DatasetJobSpecV2,
    ResolvedDatasetJobPlanV2,
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkCancellationRecordV2,
    WorkControlPolicyV2,
    WorkDispatchDecisionV2,
    WorkLeaseEventKindV2,
    WorkLeaseEventV2,
    WorkLeaseStateV2,
    WorkLeaseV2,
    WorkReadinessSnapshotV2,
    WorkReadinessV2,
    WorkRetryDecisionKindV2,
    WorkRetryDecisionV2,
    WorkUnitScopeV2,
    artifact_group_fanout_v2_ref,
    resolved_dataset_job_plan_v2_ref,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
    work_control_policy_v2_ref,
    work_dispatch_decision_v2_ref,
    work_lease_event_v2_ref,
    work_lease_v2_ref,
    work_readiness_snapshot_v2_ref,
    work_retry_decision_v2_ref,
)
from eval_factory.contracts.resource_v2 import (
    ResourceAdmissionDecisionV2,
    ResourceAdmissionOutcomeV2,
    ResourceUsageV2,
    WorkResourceDemandV2,
    WorkResourceReservationV2,
    resource_admission_decision_v2_ref,
    work_resource_demand_v2_ref,
)
from eval_factory.orchestration.fanout import (
    DatasetJobWorkGraphCompiler,
    SemanticReviewFanoutCompiler,
    WorkFanoutPolicyError,
)
from eval_factory.orchestration.models import (
    TERMINAL_STAGE_STATUSES,
    ItemRecord,
    JobRecord,
    OutboxEvent,
    StageResultRecord,
    StageRunRecord,
    WorkLeaseHeadRecord,
    item_record_ref,
    stage_result_record_ref,
    stage_run_record_ref,
)
from eval_factory.orchestration.planning import DatasetJobPlanCompiler

RecordT = TypeVar("RecordT", bound=BaseModel)


class JobStoreError(RuntimeError):
    pass


class RecordNotFoundError(JobStoreError):
    pass


class IllegalTransitionError(JobStoreError):
    pass


class ConcurrencyConflictError(JobStoreError):
    pass


class IdempotencyConflictError(JobStoreError):
    pass


class ImmutableResultError(JobStoreError):
    pass


class WorkControlPolicyError(JobStoreError):
    pass


class StaleWorkLeaseError(JobStoreError):
    pass


JOB_TRANSITIONS = {
    JobStatus.CREATED: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.BLOCKED,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.BLOCKED: frozenset(
        {
            JobStatus.RUNNING,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
}
ITEM_TRANSITIONS = {
    ItemStatus.CANDIDATE: frozenset(
        {
            ItemStatus.RUNNING,
            ItemStatus.REJECTED,
            ItemStatus.FAILED,
        }
    ),
    ItemStatus.RUNNING: frozenset(
        {
            ItemStatus.NEEDS_REVIEW,
            ItemStatus.APPROVED,
            ItemStatus.REJECTED,
            ItemStatus.FAILED,
        }
    ),
    ItemStatus.NEEDS_REVIEW: frozenset(
        {
            ItemStatus.RUNNING,
            ItemStatus.APPROVED,
            ItemStatus.REJECTED,
            ItemStatus.FAILED,
        }
    ),
    ItemStatus.APPROVED: frozenset({ItemStatus.RELEASED, ItemStatus.REJECTED}),
    ItemStatus.RELEASED: frozenset({ItemStatus.REVOKED}),
}
STAGE_TRANSITIONS = {
    StageRunStatus.PENDING: frozenset(
        {
            StageRunStatus.RUNNING,
            StageRunStatus.CANCELLED,
        }
    ),
    StageRunStatus.RUNNING: TERMINAL_STAGE_STATUSES,
}
RETRYABLE_STAGE_STATUSES = frozenset(
    {
        StageRunStatus.BLOCKED_POLICY,
        StageRunStatus.BLOCKED_CAPABILITY,
        StageRunStatus.RETRYABLE_FAILURE,
    }
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _request_sha256(value: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _attributes(**values: ScalarValue) -> tuple[TypedAttribute, ...]:
    return tuple(
        TypedAttribute(key=key.replace("_", "-"), value=value) for key, value in sorted(values.items())
    )


class JobStore:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    row_version INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    job_spec_json TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resolved_job_plans (
                    resolved_job_plan_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    plan_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resolved_job_work_graphs (
                    resolved_job_work_graph_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    graph_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS items (
                    item_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    status TEXT NOT NULL,
                    row_version INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS items_job_id ON items(job_id);

                CREATE TABLE IF NOT EXISTS resolved_work_units (
                    resolved_work_unit_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT REFERENCES items(item_id),
                    stage TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    owner_ref TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS resolved_work_units_job_id
                ON resolved_work_units(job_id);
                CREATE INDEX IF NOT EXISTS resolved_work_units_owner_ref
                ON resolved_work_units(owner_ref);

                CREATE TABLE IF NOT EXISTS artifact_group_fanouts (
                    artifact_group_fanout_id TEXT PRIMARY KEY,
                    parent_work_unit_id TEXT NOT NULL UNIQUE
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS semantic_review_fanouts (
                    semantic_review_fanout_id TEXT PRIMARY KEY,
                    parent_work_unit_id TEXT NOT NULL UNIQUE
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_readiness_snapshots (
                    work_readiness_snapshot_id TEXT PRIMARY KEY,
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS work_readiness_by_unit
                ON work_readiness_snapshots(resolved_work_unit_id);

                CREATE TABLE IF NOT EXISTS work_control_policies (
                    work_control_policy_id TEXT PRIMARY KEY,
                    resolved_job_work_graph_id TEXT NOT NULL UNIQUE
                        REFERENCES resolved_job_work_graphs(resolved_job_work_graph_id),
                    policy_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_dispatch_decisions (
                    work_dispatch_decision_id TEXT PRIMARY KEY,
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    attempt INTEGER NOT NULL,
                    retry_decision_id TEXT
                        REFERENCES work_retry_decisions(work_retry_decision_id),
                    record_json TEXT NOT NULL,
                    UNIQUE(resolved_work_unit_id, attempt)
                );

                CREATE TABLE IF NOT EXISTS work_lease_fences (
                    resolved_work_unit_id TEXT PRIMARY KEY
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    last_fencing_token INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_leases (
                    work_lease_id TEXT PRIMARY KEY,
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    work_dispatch_decision_id TEXT NOT NULL UNIQUE
                        REFERENCES work_dispatch_decisions(work_dispatch_decision_id),
                    attempt INTEGER NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(resolved_work_unit_id, fencing_token)
                );
                CREATE INDEX IF NOT EXISTS work_leases_by_unit
                ON work_leases(resolved_work_unit_id, attempt);

                CREATE TABLE IF NOT EXISTS work_lease_heads (
                    work_lease_id TEXT PRIMARY KEY
                        REFERENCES work_leases(work_lease_id),
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    state TEXT NOT NULL,
                    lease_version INTEGER NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS work_lease_one_active
                ON work_lease_heads(resolved_work_unit_id)
                WHERE state = 'ACTIVE';

                CREATE TABLE IF NOT EXISTS work_lease_events (
                    work_lease_event_id TEXT PRIMARY KEY,
                    work_lease_id TEXT NOT NULL
                        REFERENCES work_leases(work_lease_id),
                    lease_version INTEGER NOT NULL,
                    event_kind TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(work_lease_id, lease_version)
                );

                CREATE TABLE IF NOT EXISTS work_retry_decisions (
                    work_retry_decision_id TEXT PRIMARY KEY,
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    prior_lease_event_id TEXT NOT NULL UNIQUE
                        REFERENCES work_lease_events(work_lease_event_id),
                    decision TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS work_retry_decisions_by_unit
                ON work_retry_decisions(resolved_work_unit_id);

                CREATE TABLE IF NOT EXISTS work_cancellations (
                    work_cancellation_record_id TEXT PRIMARY KEY,
                    resolved_job_work_graph_id TEXT NOT NULL UNIQUE
                        REFERENCES resolved_job_work_graphs(resolved_job_work_graph_id),
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resource_pool_policies (
                    resource_pool_policy_id TEXT PRIMARY KEY,
                    resource_domain_id TEXT NOT NULL UNIQUE,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_resource_policies (
                    job_resource_policy_id TEXT PRIMARY KEY,
                    resolved_job_work_graph_id TEXT NOT NULL UNIQUE
                        REFERENCES resolved_job_work_graphs(resolved_job_work_graph_id),
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    resource_pool_policy_id TEXT NOT NULL
                        REFERENCES resource_pool_policies(resource_pool_policy_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_resource_demands (
                    work_resource_demand_id TEXT PRIMARY KEY,
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    attempt INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(resolved_work_unit_id, attempt)
                );

                CREATE TABLE IF NOT EXISTS resource_admission_decisions (
                    resource_admission_decision_id TEXT PRIMARY KEY,
                    work_resource_demand_id TEXT NOT NULL
                        REFERENCES work_resource_demands(work_resource_demand_id),
                    outcome TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS resource_admission_by_demand
                ON resource_admission_decisions(work_resource_demand_id);

                CREATE TABLE IF NOT EXISTS work_resource_reservations (
                    work_resource_reservation_id TEXT PRIMARY KEY,
                    resource_admission_decision_id TEXT NOT NULL UNIQUE
                        REFERENCES resource_admission_decisions(resource_admission_decision_id),
                    work_resource_demand_id TEXT NOT NULL
                        REFERENCES work_resource_demands(work_resource_demand_id),
                    work_lease_id TEXT NOT NULL UNIQUE REFERENCES work_leases(work_lease_id),
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_resource_events (
                    work_resource_event_id TEXT PRIMARY KEY,
                    work_resource_reservation_id TEXT NOT NULL
                        REFERENCES work_resource_reservations(work_resource_reservation_id),
                    reservation_version INTEGER NOT NULL,
                    event_kind TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(work_resource_reservation_id, reservation_version)
                );

                CREATE TABLE IF NOT EXISTS resource_pool_heads (
                    resource_pool_policy_id TEXT PRIMARY KEY
                        REFERENCES resource_pool_policies(resource_pool_policy_id),
                    row_version INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_resource_heads (
                    job_resource_policy_id TEXT PRIMARY KEY
                        REFERENCES job_resource_policies(job_resource_policy_id),
                    row_version INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_resource_reservation_heads (
                    work_resource_reservation_id TEXT PRIMARY KEY
                        REFERENCES work_resource_reservations(work_resource_reservation_id),
                    state TEXT NOT NULL,
                    reservation_version INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_resource_termination_requests (
                    work_resource_termination_request_id TEXT PRIMARY KEY,
                    work_resource_reservation_id TEXT NOT NULL UNIQUE
                        REFERENCES work_resource_reservations(work_resource_reservation_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_resource_termination_results (
                    work_resource_termination_result_id TEXT PRIMARY KEY,
                    work_resource_termination_request_id TEXT NOT NULL UNIQUE
                        REFERENCES work_resource_termination_requests(
                            work_resource_termination_request_id
                        ),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_rate_pool_policies (
                    model_rate_pool_policy_id TEXT PRIMARY KEY,
                    provider_bucket_id TEXT NOT NULL UNIQUE,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_model_policies (
                    job_model_policy_id TEXT PRIMARY KEY,
                    resolved_job_work_graph_id TEXT NOT NULL UNIQUE
                        REFERENCES resolved_job_work_graphs(resolved_job_work_graph_id),
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_model_policy_pools (
                    job_model_policy_id TEXT NOT NULL
                        REFERENCES job_model_policies(job_model_policy_id),
                    model_rate_pool_policy_id TEXT NOT NULL
                        REFERENCES model_rate_pool_policies(model_rate_pool_policy_id),
                    PRIMARY KEY (job_model_policy_id, model_rate_pool_policy_id)
                );

                CREATE TABLE IF NOT EXISTS work_model_demands (
                    work_model_demand_id TEXT PRIMARY KEY,
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    attempt INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(resolved_work_unit_id, attempt)
                );

                CREATE TABLE IF NOT EXISTS model_admission_decisions (
                    model_admission_decision_id TEXT PRIMARY KEY,
                    work_model_demand_id TEXT NOT NULL
                        REFERENCES work_model_demands(work_model_demand_id),
                    outcome TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS model_admission_by_demand
                ON model_admission_decisions(work_model_demand_id);

                CREATE TABLE IF NOT EXISTS work_model_reservations (
                    work_model_reservation_id TEXT PRIMARY KEY,
                    model_admission_decision_id TEXT NOT NULL UNIQUE
                        REFERENCES model_admission_decisions(model_admission_decision_id),
                    work_model_demand_id TEXT NOT NULL
                        REFERENCES work_model_demands(work_model_demand_id),
                    work_lease_id TEXT NOT NULL UNIQUE REFERENCES work_leases(work_lease_id),
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    model_rate_pool_policy_id TEXT NOT NULL
                        REFERENCES model_rate_pool_policies(model_rate_pool_policy_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_model_events (
                    work_model_event_id TEXT PRIMARY KEY,
                    work_model_reservation_id TEXT NOT NULL
                        REFERENCES work_model_reservations(work_model_reservation_id),
                    model_rate_pool_policy_id TEXT NOT NULL
                        REFERENCES model_rate_pool_policies(model_rate_pool_policy_id),
                    reservation_version INTEGER NOT NULL,
                    bucket_version INTEGER NOT NULL,
                    event_kind TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(work_model_reservation_id, reservation_version),
                    UNIQUE(model_rate_pool_policy_id, bucket_version)
                );

                CREATE TABLE IF NOT EXISTS model_rate_pool_heads (
                    model_rate_pool_policy_id TEXT PRIMARY KEY
                        REFERENCES model_rate_pool_policies(model_rate_pool_policy_id),
                    row_version INTEGER NOT NULL,
                    bucket_version INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_model_heads (
                    job_model_policy_id TEXT PRIMARY KEY
                        REFERENCES job_model_policies(job_model_policy_id),
                    row_version INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_model_reservation_heads (
                    work_model_reservation_id TEXT PRIMARY KEY
                        REFERENCES work_model_reservations(work_model_reservation_id),
                    state TEXT NOT NULL,
                    reservation_version INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS provider_backpressure_records (
                    provider_backpressure_record_id TEXT PRIMARY KEY,
                    work_model_reservation_id TEXT NOT NULL
                        REFERENCES work_model_reservations(work_model_reservation_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_model_cancellation_requests (
                    work_model_cancellation_request_id TEXT PRIMARY KEY,
                    work_model_reservation_id TEXT NOT NULL UNIQUE
                        REFERENCES work_model_reservations(work_model_reservation_id),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_model_cancellation_results (
                    work_model_cancellation_result_id TEXT PRIMARY KEY,
                    work_model_cancellation_request_id TEXT NOT NULL UNIQUE
                        REFERENCES work_model_cancellation_requests(
                            work_model_cancellation_request_id
                        ),
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stage_runs (
                    stage_run_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT REFERENCES items(item_id),
                    stage TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    row_version INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    retry_of_stage_run_id TEXT REFERENCES stage_runs(stage_run_id),
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS stage_runs_job_id ON stage_runs(job_id);
                CREATE UNIQUE INDEX IF NOT EXISTS stage_runs_one_active
                ON stage_runs(job_id, COALESCE(item_id, ''), stage)
                WHERE status IN ('PENDING', 'RUNNING');

                CREATE TABLE IF NOT EXISTS stage_results (
                    stage_result_id TEXT PRIMARY KEY,
                    stage_run_id TEXT NOT NULL UNIQUE REFERENCES stage_runs(stage_run_id),
                    result_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_attempt_metrics (
                    work_attempt_metrics_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    resolved_work_unit_id TEXT NOT NULL
                        REFERENCES resolved_work_units(resolved_work_unit_id),
                    work_lease_id TEXT NOT NULL UNIQUE
                        REFERENCES work_leases(work_lease_id),
                    attempt INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS work_attempt_metrics_job_id
                ON work_attempt_metrics(job_id, attempt);

                CREATE TABLE IF NOT EXISTS batch_audit_events (
                    batch_audit_event_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    source_family TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_version INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(job_id, source_family, source_id)
                );
                CREATE INDEX IF NOT EXISTS batch_audit_events_job_order
                ON batch_audit_events(
                    job_id, occurred_at, source_family,
                    source_version, source_id
                );

                CREATE TABLE IF NOT EXISTS item_metrics_snapshots (
                    item_metrics_snapshot_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT NOT NULL REFERENCES items(item_id),
                    source_fingerprint TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(item_id, source_fingerprint)
                );
                CREATE INDEX IF NOT EXISTS item_metrics_snapshots_job
                ON item_metrics_snapshots(job_id, item_id);

                CREATE TABLE IF NOT EXISTS batch_metrics_snapshots (
                    batch_metrics_snapshot_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    source_fingerprint TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(job_id, source_fingerprint)
                );

                CREATE TABLE IF NOT EXISTS batch_audit_reports (
                    batch_audit_report_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    source_fingerprint TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(job_id, source_fingerprint)
                );

                CREATE TABLE IF NOT EXISTS user_approval_request_compilations (
                    result_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    checkpoint TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_approval_requests (
                    request_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    compilation_result_id TEXT NOT NULL
                        REFERENCES user_approval_request_compilations(result_id),
                    checkpoint TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_approval_request_revisions (
                    revision_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    source_request_id TEXT NOT NULL
                        REFERENCES user_approval_requests(request_id),
                    revision_number INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    revision_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(source_request_id, revision_number)
                );

                CREATE TABLE IF NOT EXISTS user_decision_records (
                    decision_record_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    request_id TEXT NOT NULL UNIQUE
                        REFERENCES user_approval_requests(request_id),
                    authenticated_user TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    record_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_decision_commits (
                    commit_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    request_id TEXT NOT NULL UNIQUE
                        REFERENCES user_approval_requests(request_id),
                    decision_record_id TEXT NOT NULL UNIQUE
                        REFERENCES user_decision_records(decision_record_id),
                    outcome TEXT NOT NULL,
                    commit_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_checkpoint_interaction_policies (
                    policy_id TEXT PRIMARY KEY,
                    policy_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_checkpoint_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    chain_id TEXT NOT NULL,
                    interaction_version INTEGER NOT NULL,
                    checkpoint TEXT NOT NULL,
                    state TEXT NOT NULL,
                    request_id TEXT NOT NULL
                        REFERENCES user_approval_requests(request_id),
                    decision_commit_id TEXT
                        REFERENCES user_decision_commits(commit_id),
                    interaction_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(chain_id, interaction_version)
                );
                CREATE INDEX IF NOT EXISTS user_checkpoint_interactions_job
                ON user_checkpoint_interactions(
                    job_id, chain_id, interaction_version
                );

                CREATE TABLE IF NOT EXISTS user_checkpoint_results (
                    result_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    result_type TEXT NOT NULL,
                    interaction_id TEXT
                        REFERENCES user_checkpoint_interactions(interaction_id),
                    result_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS user_checkpoint_results_job
                ON user_checkpoint_results(job_id, result_type, result_id);

                CREATE TABLE IF NOT EXISTS user_checkpoint_current_heads (
                    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id),
                    interaction_id TEXT NOT NULL UNIQUE
                        REFERENCES user_checkpoint_interactions(interaction_id),
                    chain_id TEXT NOT NULL,
                    interaction_version INTEGER NOT NULL,
                    state TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_plan_application_policies (
                    policy_id TEXT PRIMARY KEY,
                    policy_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_plan_applications (
                    application_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    decision_commit_id TEXT NOT NULL UNIQUE
                        REFERENCES user_decision_commits(commit_id),
                    policy_id TEXT NOT NULL
                        REFERENCES user_plan_application_policies(policy_id),
                    checkpoint TEXT NOT NULL,
                    application_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS user_plan_applications_job
                ON user_plan_applications(job_id, application_id);

                CREATE TABLE IF NOT EXISTS directed_revalidation_plans (
                    plan_id TEXT PRIMARY KEY,
                    application_id TEXT NOT NULL UNIQUE
                        REFERENCES user_plan_applications(application_id),
                    resolved_job_work_graph_id TEXT NOT NULL
                        REFERENCES resolved_job_work_graphs(
                            resolved_job_work_graph_id
                        ),
                    plan_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS directed_revalidation_work_items (
                    work_item_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL
                        REFERENCES directed_revalidation_plans(plan_id),
                    application_id TEXT NOT NULL
                        REFERENCES user_plan_applications(application_id),
                    item_id TEXT REFERENCES items(item_id),
                    stage TEXT NOT NULL,
                    work_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS directed_revalidation_work_plan
                ON directed_revalidation_work_items(plan_id, stage, item_id);

                CREATE TABLE IF NOT EXISTS directed_revalidation_work_results (
                    result_id TEXT PRIMARY KEY,
                    work_item_id TEXT NOT NULL UNIQUE
                        REFERENCES directed_revalidation_work_items(work_item_id),
                    stage_result_id TEXT NOT NULL UNIQUE
                        REFERENCES stage_results(stage_result_id),
                    outcome TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS directed_revalidation_reports (
                    report_id TEXT PRIMARY KEY,
                    application_id TEXT NOT NULL UNIQUE
                        REFERENCES user_plan_applications(application_id),
                    plan_id TEXT NOT NULL UNIQUE
                        REFERENCES directed_revalidation_plans(plan_id),
                    outcome TEXT NOT NULL,
                    report_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS object_validity_events (
                    validity_event_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    application_id TEXT NOT NULL
                        REFERENCES user_plan_applications(application_id),
                    report_id TEXT
                        REFERENCES directed_revalidation_reports(report_id),
                    event_kind TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    object_version TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    object_ref_json TEXT NOT NULL,
                    UNIQUE(
                        application_id, event_kind, object_type,
                        object_id, object_version, object_sha256
                    )
                );
                CREATE INDEX IF NOT EXISTS object_validity_events_job
                ON object_validity_events(job_id, application_id, event_kind);

                CREATE TABLE IF NOT EXISTS object_current_heads (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    object_version TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    report_id TEXT NOT NULL
                        REFERENCES directed_revalidation_reports(report_id),
                    object_ref_json TEXT NOT NULL,
                    PRIMARY KEY (
                        job_id, object_type, object_id,
                        object_version, object_sha256
                    )
                );

                CREATE TABLE IF NOT EXISTS release_projection_policies (
                    policy_id TEXT PRIMARY KEY,
                    policy_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_query_specs (
                    query_spec_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT NOT NULL REFERENCES items(item_id),
                    task_draft_id TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(item_id, task_draft_id, prompt_sha256)
                );

                CREATE TABLE IF NOT EXISTS evaluation_item_release_subjects (
                    release_subject_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT NOT NULL REFERENCES items(item_id),
                    policy_id TEXT NOT NULL
                        REFERENCES release_projection_policies(policy_id),
                    query_spec_id TEXT NOT NULL
                        REFERENCES evaluation_query_specs(query_spec_id),
                    release_subject_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evaluation_release_subjects_item
                ON evaluation_item_release_subjects(item_id, release_subject_id);

                CREATE TABLE IF NOT EXISTS release_decisions_v2 (
                    release_decision_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT NOT NULL REFERENCES items(item_id),
                    release_subject_id TEXT NOT NULL
                        REFERENCES evaluation_item_release_subjects(
                            release_subject_id
                        ),
                    chain_id TEXT NOT NULL,
                    previous_decision_id TEXT
                        REFERENCES release_decisions_v2(release_decision_id),
                    action TEXT NOT NULL,
                    state TEXT NOT NULL,
                    decision_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS release_decisions_v2_chain
                ON release_decisions_v2(item_id, chain_id, release_decision_id);

                CREATE TABLE IF NOT EXISTS evaluation_items_v2 (
                    evaluation_item_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT NOT NULL REFERENCES items(item_id),
                    release_decision_id TEXT NOT NULL UNIQUE
                        REFERENCES release_decisions_v2(release_decision_id),
                    item_version TEXT NOT NULL,
                    item_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evaluation_items_v2_item
                ON evaluation_items_v2(item_id, evaluation_item_id);

                CREATE TABLE IF NOT EXISTS item_release_projection_events (
                    projection_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT NOT NULL REFERENCES items(item_id),
                    projection_revision INTEGER NOT NULL,
                    previous_projection_id TEXT
                        REFERENCES item_release_projection_events(projection_id),
                    release_subject_id TEXT NOT NULL
                        REFERENCES evaluation_item_release_subjects(
                            release_subject_id
                        ),
                    evaluation_item_id TEXT NOT NULL UNIQUE
                        REFERENCES evaluation_items_v2(evaluation_item_id),
                    release_decision_id TEXT NOT NULL UNIQUE
                        REFERENCES release_decisions_v2(release_decision_id),
                    release_state TEXT NOT NULL,
                    item_status TEXT NOT NULL,
                    projection_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(item_id, projection_revision)
                );

                CREATE TABLE IF NOT EXISTS release_projection_results (
                    result_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT NOT NULL REFERENCES items(item_id),
                    phase TEXT NOT NULL,
                    release_subject_id TEXT NOT NULL
                        REFERENCES evaluation_item_release_subjects(
                            release_subject_id
                        ),
                    query_spec_id TEXT NOT NULL
                        REFERENCES evaluation_query_specs(query_spec_id),
                    release_decision_id TEXT NOT NULL UNIQUE
                        REFERENCES release_decisions_v2(release_decision_id),
                    evaluation_item_id TEXT NOT NULL UNIQUE
                        REFERENCES evaluation_items_v2(evaluation_item_id),
                    projection_id TEXT NOT NULL UNIQUE
                        REFERENCES item_release_projection_events(projection_id),
                    previous_result_id TEXT
                        REFERENCES release_projection_results(result_id),
                    stage_result_id TEXT
                        REFERENCES stage_results(stage_result_id),
                    result_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS release_projection_results_item
                ON release_projection_results(item_id, result_id);

                CREATE TABLE IF NOT EXISTS item_release_current_projections (
                    item_id TEXT PRIMARY KEY REFERENCES items(item_id),
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    projection_id TEXT NOT NULL UNIQUE
                        REFERENCES item_release_projection_events(projection_id),
                    result_id TEXT NOT NULL UNIQUE
                        REFERENCES release_projection_results(result_id),
                    projection_revision INTEGER NOT NULL,
                    release_state TEXT NOT NULL,
                    item_status TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS release_publication_policies (
                    policy_id TEXT PRIMARY KEY,
                    policy_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lh_release_manifest_items (
                    item_manifest_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT NOT NULL REFERENCES items(item_id),
                    approved_result_id TEXT NOT NULL
                        REFERENCES release_projection_results(result_id),
                    policy_id TEXT NOT NULL
                        REFERENCES release_publication_policies(policy_id),
                    item_manifest_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(item_id, item_manifest_id)
                );

                CREATE TABLE IF NOT EXISTS nonproduction_release_manifests (
                    manifest_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    channel TEXT NOT NULL,
                    registry TEXT NOT NULL,
                    policy_id TEXT NOT NULL
                        REFERENCES release_publication_policies(policy_id),
                    manifest_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lh_export_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    manifest_id TEXT NOT NULL
                        REFERENCES nonproduction_release_manifests(manifest_id),
                    item_manifest_id TEXT NOT NULL UNIQUE
                        REFERENCES lh_release_manifest_items(item_manifest_id),
                    workspace_export_result_id TEXT NOT NULL,
                    bundle_sha256 TEXT NOT NULL,
                    receipt_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS nonproduction_registry_entries (
                    registry_entry_id TEXT PRIMARY KEY,
                    manifest_id TEXT NOT NULL UNIQUE
                        REFERENCES nonproduction_release_manifests(manifest_id),
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    channel TEXT NOT NULL,
                    registry TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    entry_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS item_publication_projection_events (
                    projection_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    item_id TEXT NOT NULL REFERENCES items(item_id),
                    approved_result_id TEXT NOT NULL
                        REFERENCES release_projection_results(result_id),
                    manifest_id TEXT NOT NULL
                        REFERENCES nonproduction_release_manifests(manifest_id),
                    receipt_id TEXT NOT NULL UNIQUE
                        REFERENCES lh_export_receipts(receipt_id),
                    registry_entry_id TEXT NOT NULL
                        REFERENCES nonproduction_registry_entries(
                            registry_entry_id
                        ),
                    publish_decision_id TEXT NOT NULL UNIQUE
                        REFERENCES release_decisions_v2(release_decision_id),
                    published_evaluation_item_id TEXT NOT NULL UNIQUE
                        REFERENCES evaluation_items_v2(evaluation_item_id),
                    item_status TEXT NOT NULL,
                    projection_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(item_id, manifest_id)
                );

                CREATE TABLE IF NOT EXISTS release_publication_results (
                    result_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    manifest_id TEXT NOT NULL UNIQUE
                        REFERENCES nonproduction_release_manifests(manifest_id),
                    registry_entry_id TEXT NOT NULL UNIQUE
                        REFERENCES nonproduction_registry_entries(
                            registry_entry_id
                        ),
                    policy_id TEXT NOT NULL
                        REFERENCES release_publication_policies(policy_id),
                    result_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS item_publication_current_projections (
                    item_id TEXT PRIMARY KEY REFERENCES items(item_id),
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    projection_id TEXT NOT NULL UNIQUE
                        REFERENCES item_publication_projection_events(
                            projection_id
                        ),
                    result_id TEXT NOT NULL
                        REFERENCES release_publication_results(result_id),
                    item_status TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_release_acceptances (
                    acceptance_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
                    result_id TEXT NOT NULL UNIQUE,
                    result_sha256 TEXT NOT NULL,
                    registry_entry_id TEXT NOT NULL UNIQUE,
                    policy_id TEXT NOT NULL,
                    attestation_authority_id TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_item_publication_current (
                    item_id TEXT PRIMARY KEY REFERENCES items(item_id),
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    projection_id TEXT NOT NULL UNIQUE,
                    result_id TEXT NOT NULL,
                    acceptance_id TEXT NOT NULL
                        REFERENCES production_release_acceptances(acceptance_id),
                    item_status TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS controlled_stage_runs (
                    stage_run_id TEXT PRIMARY KEY
                        REFERENCES stage_runs(stage_run_id),
                    work_lease_id TEXT NOT NULL UNIQUE
                        REFERENCES work_leases(work_lease_id),
                    fencing_token INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS idempotency_records (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_type TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (scope, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS outbox_events (
                    event_id TEXT PRIMARY KEY,
                    aggregate_type TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    aggregate_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    published_at TEXT
                );
                CREATE INDEX IF NOT EXISTS outbox_unpublished
                ON outbox_events(event_id)
                WHERE published_at IS NULL;
                """
            )

    @contextmanager
    def _read_snapshot(self) -> Generator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.rollback()
            connection.close()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _record_json(value: BaseModel) -> str:
        return canonical_json_v2(value).decode()

    @staticmethod
    def _load_record(
        row: sqlite3.Row,
        model_type: type[RecordT],
    ) -> RecordT:
        return model_type.model_validate_json(str(row["record_json"]))

    def _get_record(
        self,
        connection: sqlite3.Connection,
        table: str,
        id_column: str,
        record_id: str,
        model_type: type[RecordT],
    ) -> RecordT:
        row = connection.execute(
            f"SELECT record_json FROM {table} WHERE {id_column} = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"{model_type.__name__} not found: {record_id}")
        return self._load_record(row, model_type)

    def _idempotent_response(
        self,
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
    ) -> tuple[str, str] | None:
        row = connection.execute(
            """
            SELECT request_sha256, response_type, response_id
            FROM idempotency_records
            WHERE scope = ? AND idempotency_key = ?
            """,
            (scope, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if str(row["request_sha256"]) != request_sha256:
            raise IdempotencyConflictError(
                f"idempotency key {idempotency_key!r} was used for a different request"
            )
        return str(row["response_type"]), str(row["response_id"])

    def _record_idempotency(
        self,
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_type: str,
        response_id: str,
        created_at: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO idempotency_records (
                scope, idempotency_key, request_sha256,
                response_type, response_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                scope,
                idempotency_key,
                request_sha256,
                response_type,
                response_id,
                created_at.isoformat(),
            ),
        )

    def _append_outbox(
        self,
        connection: sqlite3.Connection,
        *,
        aggregate_type: Literal["JOB", "ITEM", "STAGE_RUN"],
        aggregate_id: str,
        aggregate_version: int,
        event_type: str,
        attributes: tuple[TypedAttribute, ...] = (),
    ) -> OutboxEvent:
        event = OutboxEvent(
            event_id=f"outbox-event://{uuid4().hex}",
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            aggregate_version=aggregate_version,
            event_type=event_type,
            attributes=attributes,
            created_at=self._clock(),
        )
        connection.execute(
            """
            INSERT INTO outbox_events (
                event_id, aggregate_type, aggregate_id,
                aggregate_version, event_type, record_json, published_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                event.event_id,
                event.aggregate_type,
                event.aggregate_id,
                event.aggregate_version,
                event.event_type,
                self._record_json(event),
            ),
        )
        return event

    def create_job(self, spec: DatasetJobSpecV2) -> JobRecord:
        return self._create_job(spec=spec, plan=None)

    def create_planned_job(
        self,
        spec: DatasetJobSpecV2,
        plan: ResolvedDatasetJobPlanV2,
    ) -> JobRecord:
        DatasetJobPlanCompiler().validate_current(plan, job_spec=spec)
        return self._create_job(spec=spec, plan=plan)

    def _create_job(
        self,
        *,
        spec: DatasetJobSpecV2,
        plan: ResolvedDatasetJobPlanV2 | None,
    ) -> JobRecord:
        spec_sha256 = canonical_sha256_v2(spec)
        request_sha256 = (
            spec_sha256
            if plan is None
            else _request_sha256(
                {
                    "job_spec_sha256": spec_sha256,
                    "resolved_job_plan_ref": resolved_dataset_job_plan_v2_ref(plan),
                }
            )
        )
        scope = "create-job"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=spec.idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, response_id = prior
                if response_type != "JOB":
                    raise IdempotencyConflictError("create-job response type is corrupt")
                record = self._get_record(
                    connection,
                    "jobs",
                    "job_id",
                    response_id,
                    JobRecord,
                )
                if plan is not None:
                    try:
                        stored_plan = self._get_resolved_job_plan(
                            connection,
                            response_id,
                        )
                    except RecordNotFoundError as exc:
                        raise ImmutableResultError("stored resolved plan is missing or invalid") from exc
                    if stored_plan != plan:
                        raise ImmutableResultError("stored resolved plan differs from the requested plan")
                return record

            now = self._clock()
            record = JobRecord(
                job_id=spec.job_id,
                job_spec_sha256=spec_sha256,
                row_version=0,
                idempotency_key=spec.idempotency_key,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, status, row_version, idempotency_key,
                    job_spec_json, record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    record.status,
                    record.row_version,
                    record.idempotency_key,
                    self._record_json(spec),
                    self._record_json(record),
                ),
            )
            if plan is not None:
                connection.execute(
                    """
                    INSERT INTO resolved_job_plans (
                        resolved_job_plan_id, job_id, plan_sha256, record_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        plan.resolved_job_plan_id,
                        record.job_id,
                        plan.resolved_job_plan_sha256,
                        self._record_json(plan),
                    ),
                )
            self._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=record.job_id,
                aggregate_version=record.row_version,
                event_type="job-created",
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=spec.idempotency_key,
                request_sha256=request_sha256,
                response_type="JOB",
                response_id=record.job_id,
                created_at=now,
            )
            return record

    def get_job(self, job_id: Identifier) -> JobRecord:
        with self._connect() as connection:
            return self._get_record(connection, "jobs", "job_id", job_id, JobRecord)

    def has_current_checkpoint_interaction(
        self,
        job_id: Identifier,
    ) -> bool:
        with self._connect() as connection:
            self._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            row = connection.execute(
                """
                SELECT interaction_id, chain_id, interaction_version, state
                FROM user_checkpoint_current_heads
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            latest = connection.execute(
                """
                SELECT interaction_id, job_id, chain_id,
                       interaction_version, state, interaction_sha256,
                       record_json
                FROM user_checkpoint_interactions
                WHERE job_id = ?
                ORDER BY interaction_version DESC
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if latest is None:
                if row is not None:
                    raise ImmutableResultError("checkpoint current head has no immutable interaction")
                return False
            if row is None:
                raise ImmutableResultError("checkpoint current head is missing")
            if any(
                row[column] != latest[column]
                for column in (
                    "interaction_id",
                    "chain_id",
                    "interaction_version",
                    "state",
                )
            ):
                raise ImmutableResultError("checkpoint current head differs from latest interaction")
            try:
                interaction = UserCheckpointInteractionV2.model_validate_json(str(latest["record_json"]))
                validate_user_checkpoint_interaction_v2_identity(interaction)
            except (ValidationError, ValueError) as exc:
                raise ImmutableResultError("latest checkpoint interaction is malformed") from exc
            if (
                str(latest["interaction_id"]) != interaction.interaction_id
                or str(latest["job_id"]) != interaction.job_id
                or str(latest["chain_id"]) != interaction.chain_id
                or int(latest["interaction_version"]) != interaction.interaction_version
                or str(latest["state"]) != interaction.state.value
                or str(latest["interaction_sha256"]) != interaction.interaction_sha256
            ):
                raise ImmutableResultError("latest checkpoint interaction columns are inconsistent")
            return interaction.state not in {
                UserCheckpointInteractionStateV2.RESUMED,
                UserCheckpointInteractionStateV2.TERMINATED,
            }

    def _list_incomplete_work_refs_in_transaction(
        self,
        connection: sqlite3.Connection,
        job_id: Identifier,
    ) -> tuple[ObjectRef, ...]:
        self._get_record(
            connection,
            "jobs",
            "job_id",
            job_id,
            JobRecord,
        )
        rows = connection.execute(
            """
            SELECT resolved_work_unit_id, job_id, item_id, stage, scope,
                   owner_ref, record_json
            FROM resolved_work_units
            WHERE job_id = ?
            ORDER BY rowid
            """,
            (job_id,),
        ).fetchall()
        incomplete: list[ObjectRef] = []
        for row in rows:
            unit, _ = self._load_work_unit_row(row)
            retry_row = connection.execute(
                """
                SELECT work_retry_decision_id, resolved_work_unit_id,
                       prior_lease_event_id, decision, record_json
                FROM work_retry_decisions
                WHERE resolved_work_unit_id = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (unit.resolved_work_unit_id,),
            ).fetchone()
            if retry_row is not None:
                retry = self._load_work_retry_decision_row(
                    connection,
                    retry_row,
                )
                if retry.decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED:
                    incomplete.append(resolved_work_unit_v2_ref(unit))
                    continue
            lease = self._get_latest_work_lease_for_unit(
                connection,
                unit.resolved_work_unit_id,
            )
            if lease is not None:
                head = self._get_work_lease_head(
                    connection,
                    lease.work_lease_id,
                )
                if head.state is WorkLeaseStateV2.ACTIVE:
                    incomplete.append(resolved_work_unit_v2_ref(unit))
                continue
            readiness_row = connection.execute(
                """
                SELECT work_readiness_snapshot_id, resolved_work_unit_id,
                       record_json
                FROM work_readiness_snapshots
                WHERE resolved_work_unit_id = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (unit.resolved_work_unit_id,),
            ).fetchone()
            if readiness_row is None:
                incomplete.append(resolved_work_unit_v2_ref(unit))
                continue
            readiness = self._load_work_readiness_row(readiness_row)
            if readiness.readiness in {
                WorkReadinessV2.READY,
                WorkReadinessV2.WAITING,
            }:
                incomplete.append(resolved_work_unit_v2_ref(unit))
        return tuple(incomplete)

    def get_job_spec(self, job_id: Identifier) -> DatasetJobSpecV2:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT job_spec_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"JobRecord not found: {job_id}")
            return DatasetJobSpecV2.model_validate_json(str(row["job_spec_json"]))

    def get_resolved_job_plan(
        self,
        job_id: Identifier,
    ) -> ResolvedDatasetJobPlanV2:
        with self._connect() as connection:
            return self._get_resolved_job_plan(connection, job_id)

    def _get_resolved_job_plan(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> ResolvedDatasetJobPlanV2:
        row = connection.execute(
            """
            SELECT resolved_job_plan_id, job_id, plan_sha256, record_json
            FROM resolved_job_plans
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"ResolvedDatasetJobPlanV2 not found: {job_id}")
        try:
            plan = self._load_record(row, ResolvedDatasetJobPlanV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored resolved plan record is invalid") from exc
        if (
            str(row["resolved_job_plan_id"]) != plan.resolved_job_plan_id
            or str(row["plan_sha256"]) != plan.resolved_job_plan_sha256
            or str(row["job_id"]) != plan.dataset_job_spec_ref.object_id
        ):
            raise ImmutableResultError("stored resolved plan columns are inconsistent")
        return plan

    def create_job_work_graph(
        self,
        graph: ResolvedJobWorkGraphV2,
        *,
        idempotency_key: Identifier,
    ) -> ResolvedJobWorkGraphV2:
        canonical = ResolvedJobWorkGraphV2.model_validate(graph.model_dump(mode="python"))
        graph_ref = resolved_job_work_graph_v2_ref(canonical)
        request_sha256 = _request_sha256({"resolved_job_work_graph_ref": graph_ref})
        scope = f"create-job-work-graph:{canonical.job_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, response_id = prior
                if response_type != "JOB_WORK_GRAPH":
                    raise IdempotencyConflictError("create-job-work-graph response type is corrupt")
                stored = self._get_job_work_graph_by_id(connection, response_id)
                if stored != canonical:
                    raise ImmutableResultError("stored job work graph differs from the requested graph")
                return stored

            job = self._get_record(
                connection,
                "jobs",
                "job_id",
                canonical.job_id,
                JobRecord,
            )
            if job.status in {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                raise IllegalTransitionError(f"cannot expand terminal job {canonical.job_id}")
            spec = self._get_job_spec(connection, canonical.job_id)
            try:
                plan = self._get_resolved_job_plan(connection, canonical.job_id)
            except RecordNotFoundError as exc:
                raise WorkFanoutPolicyError("job work graph requires a persisted resolved plan") from exc
            DatasetJobWorkGraphCompiler().validate_current(
                canonical,
                job_spec=spec,
                resolved_plan=plan,
            )
            if connection.execute(
                """
                SELECT 1
                FROM resolved_job_work_graphs
                WHERE job_id = ? OR resolved_job_work_graph_id = ?
                """,
                (canonical.job_id, canonical.resolved_job_work_graph_id),
            ).fetchone():
                raise ImmutableResultError(f"job {canonical.job_id} already has an immutable work graph")
            if connection.execute(
                "SELECT 1 FROM items WHERE job_id = ? LIMIT 1",
                (canonical.job_id,),
            ).fetchone():
                raise ImmutableResultError("job work graph expansion requires an empty Item set")

            now = self._clock()
            items = tuple(
                ItemRecord(
                    item_id=item_id,
                    job_id=canonical.job_id,
                    row_version=0,
                    idempotency_key=self._work_item_idempotency_key(
                        canonical.resolved_job_work_graph_id,
                        item_id,
                    ),
                    created_at=now,
                    updated_at=now,
                )
                for item_id in canonical.item_ids
            )
            connection.execute(
                """
                INSERT INTO resolved_job_work_graphs (
                    resolved_job_work_graph_id, job_id, graph_sha256, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    canonical.resolved_job_work_graph_id,
                    canonical.job_id,
                    canonical.resolved_job_work_graph_sha256,
                    self._record_json(canonical),
                ),
            )
            for item in items:
                connection.execute(
                    """
                    INSERT INTO items (
                        item_id, job_id, status, row_version,
                        idempotency_key, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.item_id,
                        item.job_id,
                        item.status,
                        item.row_version,
                        item.idempotency_key,
                        self._record_json(item),
                    ),
                )
            self._insert_work_units(
                connection,
                canonical.work_units,
                owner_ref=canonical.resolved_job_work_graph_id,
            )
            for item in items:
                self._append_outbox(
                    connection,
                    aggregate_type="ITEM",
                    aggregate_id=item.item_id,
                    aggregate_version=item.row_version,
                    event_type="item-created",
                    attributes=_attributes(
                        job_id=canonical.job_id,
                        work_graph_id=canonical.resolved_job_work_graph_id,
                    ),
                )
            self._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=canonical.job_id,
                aggregate_version=job.row_version,
                event_type="job-work-graph-created",
                attributes=_attributes(
                    work_graph_id=canonical.resolved_job_work_graph_id,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="JOB_WORK_GRAPH",
                response_id=canonical.resolved_job_work_graph_id,
                created_at=now,
            )
            return canonical

    def get_job_work_graph(
        self,
        job_id: Identifier,
    ) -> ResolvedJobWorkGraphV2:
        with self._connect() as connection:
            return self._get_job_work_graph(connection, job_id)

    def list_work_units(
        self,
        *,
        job_id: Identifier,
        item_id: Identifier | None = None,
        scope: WorkUnitScopeV2 | None = None,
    ) -> tuple[ResolvedWorkUnitV2, ...]:
        query = """
            SELECT resolved_work_unit_id, job_id, item_id, stage, scope,
                   owner_ref, record_json
            FROM resolved_work_units
            WHERE job_id = ?
            """
        parameters: list[object] = [job_id]
        if item_id is not None:
            query += " AND item_id = ?"
            parameters.append(item_id)
        if scope is not None:
            query += " AND scope = ?"
            parameters.append(scope)
        query += " ORDER BY rowid"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
            loaded = tuple(self._load_work_unit_row(row) for row in rows)
            if not loaded:
                return ()
            graph = self._get_job_work_graph(connection, job_id)
            for unit, owner_ref in loaded:
                if owner_ref == graph.resolved_job_work_graph_id:
                    if unit not in graph.work_units:
                        raise ImmutableResultError("job work unit owner is inconsistent")
                    continue
                if unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP:
                    fanout = self._get_artifact_group_fanout_by_id(
                        connection,
                        owner_ref,
                    )
                    if (
                        fanout.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph)
                        or unit not in fanout.group_work_units
                    ):
                        raise ImmutableResultError("artifact work unit owner is inconsistent")
                    continue
                semantic_fanout = self._get_semantic_review_fanout_by_id(
                    connection,
                    owner_ref,
                )
                if semantic_fanout.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(
                    graph
                ) or unit not in tuple(value.work_unit for value in semantic_fanout.round_work):
                    raise ImmutableResultError("semantic review work unit owner is inconsistent")
            return tuple(unit for unit, _ in loaded)

    def create_artifact_group_fanout(
        self,
        fanout: ArtifactGroupFanoutV2,
        *,
        idempotency_key: Identifier,
    ) -> ArtifactGroupFanoutV2:
        canonical = ArtifactGroupFanoutV2.model_validate(fanout.model_dump(mode="python"))
        fanout_ref = artifact_group_fanout_v2_ref(canonical)
        request_sha256 = _request_sha256({"artifact_group_fanout_ref": fanout_ref})
        parent_id = canonical.parent_attachment_work_unit_ref.object_id
        scope = f"create-artifact-group-fanout:{parent_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, response_id = prior
                if response_type != "ARTIFACT_GROUP_FANOUT":
                    raise IdempotencyConflictError("create-artifact-group-fanout response type is corrupt")
                stored = self._get_artifact_group_fanout_by_id(
                    connection,
                    response_id,
                )
                if stored != canonical:
                    raise ImmutableResultError("stored artifact group fanout differs from the request")
                return stored

            parent, owner_ref = self._get_work_unit(connection, parent_id)
            if resolved_work_unit_v2_ref(parent) != (canonical.parent_attachment_work_unit_ref):
                raise WorkFanoutPolicyError("artifact fanout parent work unit is stale")
            graph = self._get_job_work_graph_by_id(
                connection,
                canonical.resolved_job_work_graph_ref.object_id,
            )
            if (
                resolved_job_work_graph_v2_ref(graph) != canonical.resolved_job_work_graph_ref
                or owner_ref != graph.resolved_job_work_graph_id
                or parent not in graph.work_units
            ):
                raise WorkFanoutPolicyError("artifact fanout does not bind the persisted parent graph")
            for unit in canonical.group_work_units:
                if (
                    unit.job_id != graph.job_id
                    or unit.item_id != parent.item_id
                    or unit.scope is not WorkUnitScopeV2.ARTIFACT_GROUP
                ):
                    raise WorkFanoutPolicyError("artifact group work unit crosses its parent scope")
            if connection.execute(
                """
                SELECT 1
                FROM artifact_group_fanouts
                WHERE artifact_group_fanout_id = ? OR parent_work_unit_id = ?
                """,
                (canonical.artifact_group_fanout_id, parent_id),
            ).fetchone():
                raise ImmutableResultError("attachment work unit already has an immutable artifact fanout")

            now = self._clock()
            connection.execute(
                """
                INSERT INTO artifact_group_fanouts (
                    artifact_group_fanout_id, parent_work_unit_id, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    canonical.artifact_group_fanout_id,
                    parent_id,
                    self._record_json(canonical),
                ),
            )
            self._insert_work_units(
                connection,
                canonical.group_work_units,
                owner_ref=canonical.artifact_group_fanout_id,
            )
            if parent.item_id is None:
                raise WorkFanoutPolicyError("artifact fanout parent must belong to an Item")
            item = self._get_record(
                connection,
                "items",
                "item_id",
                parent.item_id,
                ItemRecord,
            )
            self._append_outbox(
                connection,
                aggregate_type="ITEM",
                aggregate_id=item.item_id,
                aggregate_version=item.row_version,
                event_type="artifact-group-fanout-created",
                attributes=_attributes(
                    fanout_id=canonical.artifact_group_fanout_id,
                    parent_work_unit_id=parent_id,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="ARTIFACT_GROUP_FANOUT",
                response_id=canonical.artifact_group_fanout_id,
                created_at=now,
            )
            return canonical

    def get_artifact_group_fanout(
        self,
        parent_work_unit_id: Identifier,
    ) -> ArtifactGroupFanoutV2:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT artifact_group_fanout_id
                FROM artifact_group_fanouts
                WHERE parent_work_unit_id = ?
                """,
                (parent_work_unit_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"ArtifactGroupFanoutV2 not found: {parent_work_unit_id}")
            return self._get_artifact_group_fanout_by_id(
                connection,
                str(row["artifact_group_fanout_id"]),
            )

    def create_semantic_review_fanout(
        self,
        fanout: SemanticReviewFanoutV2,
        *,
        idempotency_key: Identifier,
    ) -> SemanticReviewFanoutV2:
        canonical = SemanticReviewFanoutV2.model_validate(fanout.model_dump(mode="python"))
        fanout_ref = semantic_review_fanout_v2_ref(canonical)
        request_sha256 = _request_sha256({"semantic_review_fanout_ref": fanout_ref})
        parent_id = canonical.parent_item_quality_work_unit_ref.object_id
        scope = f"create-semantic-review-fanout:{parent_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "SEMANTIC_REVIEW_FANOUT":
                    raise IdempotencyConflictError("semantic review fanout response type is corrupt")
                stored = self._get_semantic_review_fanout_by_id(
                    connection,
                    prior[1],
                )
                if stored != canonical:
                    raise ImmutableResultError("stored semantic review fanout differs from the request")
                return stored

            parent, owner_ref = self._get_work_unit(
                connection,
                parent_id,
            )
            if resolved_work_unit_v2_ref(parent) != canonical.parent_item_quality_work_unit_ref:
                raise WorkFanoutPolicyError("semantic review fanout parent is stale")
            graph = self._get_job_work_graph_by_id(
                connection,
                canonical.resolved_job_work_graph_ref.object_id,
            )
            if (
                resolved_job_work_graph_v2_ref(graph) != canonical.resolved_job_work_graph_ref
                or owner_ref != graph.resolved_job_work_graph_id
                or parent not in graph.work_units
            ):
                raise WorkFanoutPolicyError("semantic review fanout does not bind its persisted parent graph")
            SemanticReviewFanoutCompiler().validate_current(
                canonical,
                graph=graph,
                parent_item_quality_work_unit=parent,
            )
            if connection.execute(
                """
                SELECT 1
                FROM semantic_review_fanouts
                WHERE semantic_review_fanout_id = ?
                   OR parent_work_unit_id = ?
                """,
                (canonical.semantic_review_fanout_id, parent_id),
            ).fetchone():
                raise ImmutableResultError(
                    "ITEM_QUALITY work already has an immutable semantic review fanout"
                )

            connection.execute(
                """
                INSERT INTO semantic_review_fanouts (
                    semantic_review_fanout_id,
                    parent_work_unit_id,
                    record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    canonical.semantic_review_fanout_id,
                    parent_id,
                    self._record_json(canonical),
                ),
            )
            round_units = tuple(value.work_unit for value in canonical.round_work)
            self._insert_work_units(
                connection,
                round_units,
                owner_ref=canonical.semantic_review_fanout_id,
            )
            if parent.item_id is None:
                raise WorkFanoutPolicyError("semantic review parent must belong to an Item")
            item = self._get_record(
                connection,
                "items",
                "item_id",
                parent.item_id,
                ItemRecord,
            )
            now = self._clock()
            self._append_outbox(
                connection,
                aggregate_type="ITEM",
                aggregate_id=item.item_id,
                aggregate_version=item.row_version,
                event_type="semantic-review-fanout-created",
                attributes=_attributes(
                    fanout_id=canonical.semantic_review_fanout_id,
                    parent_work_unit_id=parent_id,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="SEMANTIC_REVIEW_FANOUT",
                response_id=canonical.semantic_review_fanout_id,
                created_at=now,
            )
            return canonical

    def get_semantic_review_fanout(
        self,
        parent_work_unit_id: Identifier,
    ) -> SemanticReviewFanoutV2:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT semantic_review_fanout_id
                FROM semantic_review_fanouts
                WHERE parent_work_unit_id = ?
                """,
                (parent_work_unit_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"SemanticReviewFanoutV2 not found: {parent_work_unit_id}")
            return self._get_semantic_review_fanout_by_id(
                connection,
                str(row["semantic_review_fanout_id"]),
            )

    def record_work_readiness(
        self,
        snapshot: WorkReadinessSnapshotV2,
        *,
        idempotency_key: Identifier,
    ) -> WorkReadinessSnapshotV2:
        canonical = WorkReadinessSnapshotV2.model_validate(snapshot.model_dump(mode="python"))
        snapshot_ref = work_readiness_snapshot_v2_ref(canonical)
        request_sha256 = _request_sha256({"work_readiness_snapshot_ref": snapshot_ref})
        work_unit_id = canonical.work_unit_ref.object_id
        scope = f"record-work-readiness:{work_unit_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, response_id = prior
                if response_type != "WORK_READINESS":
                    raise IdempotencyConflictError("record-work-readiness response type is corrupt")
                stored = self._get_work_readiness_snapshot(
                    connection,
                    response_id,
                )
                if stored != canonical:
                    raise ImmutableResultError("stored work readiness differs from the request")
                return stored

            graph = self._get_job_work_graph_by_id(
                connection,
                canonical.resolved_job_work_graph_ref.object_id,
            )
            if resolved_job_work_graph_v2_ref(graph) != (canonical.resolved_job_work_graph_ref):
                raise WorkFanoutPolicyError("work readiness graph reference is stale")
            unit, owner_ref = self._get_work_unit(connection, work_unit_id)
            if resolved_work_unit_v2_ref(unit) != canonical.work_unit_ref:
                raise WorkFanoutPolicyError("work readiness unit reference is stale")
            if canonical.semantic_review_fanout_ref is not None:
                semantic_fanout = self._get_semantic_review_fanout_by_id(
                    connection,
                    canonical.semantic_review_fanout_ref.object_id,
                )
                if (
                    semantic_review_fanout_v2_ref(semantic_fanout) != canonical.semantic_review_fanout_ref
                    or semantic_fanout.resolved_job_work_graph_ref != canonical.resolved_job_work_graph_ref
                ):
                    raise WorkFanoutPolicyError("work readiness semantic fanout reference is stale")
                round_units = tuple(value.work_unit for value in semantic_fanout.round_work)
                if owner_ref == graph.resolved_job_work_graph_id:
                    if canonical.work_unit_ref != semantic_fanout.parent_item_quality_work_unit_ref:
                        raise WorkFanoutPolicyError(
                            "semantic fanout readiness must target its finalizer or round work"
                        )
                elif owner_ref != semantic_fanout.semantic_review_fanout_id or unit not in round_units:
                    raise WorkFanoutPolicyError(
                        "work readiness unit is not owned by the semantic review fanout"
                    )
            elif canonical.artifact_group_fanout_ref is None:
                if owner_ref != graph.resolved_job_work_graph_id:
                    raise WorkFanoutPolicyError("artifact group readiness requires its fanout reference")
            else:
                fanout = self._get_artifact_group_fanout_by_id(
                    connection,
                    canonical.artifact_group_fanout_ref.object_id,
                )
                if (
                    artifact_group_fanout_v2_ref(fanout) != canonical.artifact_group_fanout_ref
                    or fanout.resolved_job_work_graph_ref != canonical.resolved_job_work_graph_ref
                ):
                    raise WorkFanoutPolicyError("work readiness artifact fanout reference is stale")
                if owner_ref == graph.resolved_job_work_graph_id:
                    if canonical.work_unit_ref != (fanout.parent_attachment_work_unit_ref):
                        raise WorkFanoutPolicyError("fanout readiness must target its parent attachment unit")
                elif owner_ref != fanout.artifact_group_fanout_id:
                    raise WorkFanoutPolicyError("work readiness unit is not owned by the artifact fanout")
            self._validate_readiness_evidence(
                connection,
                canonical,
                unit,
            )
            if connection.execute(
                """
                SELECT 1
                FROM work_readiness_snapshots
                WHERE work_readiness_snapshot_id = ?
                """,
                (canonical.work_readiness_snapshot_id,),
            ).fetchone():
                raise ImmutableResultError("work readiness snapshot already exists under another command")

            now = self._clock()
            connection.execute(
                """
                INSERT INTO work_readiness_snapshots (
                    work_readiness_snapshot_id, resolved_work_unit_id, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    canonical.work_readiness_snapshot_id,
                    work_unit_id,
                    self._record_json(canonical),
                ),
            )
            if unit.item_id is None:
                aggregate_type: Literal["JOB", "ITEM"] = "JOB"
                job = self._get_record(
                    connection,
                    "jobs",
                    "job_id",
                    unit.job_id,
                    JobRecord,
                )
                aggregate_id = job.job_id
                aggregate_version = job.row_version
            else:
                aggregate_type = "ITEM"
                item = self._get_record(
                    connection,
                    "items",
                    "item_id",
                    unit.item_id,
                    ItemRecord,
                )
                aggregate_id = item.item_id
                aggregate_version = item.row_version
            self._append_outbox(
                connection,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                aggregate_version=aggregate_version,
                event_type="work-readiness-recorded",
                attributes=_attributes(
                    readiness=canonical.readiness,
                    snapshot_id=canonical.work_readiness_snapshot_id,
                    work_unit_id=work_unit_id,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_READINESS",
                response_id=canonical.work_readiness_snapshot_id,
                created_at=now,
            )
            return canonical

    def list_work_readiness(
        self,
        resolved_work_unit_id: Identifier,
    ) -> tuple[WorkReadinessSnapshotV2, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT work_readiness_snapshot_id, resolved_work_unit_id,
                       record_json
                FROM work_readiness_snapshots
                WHERE resolved_work_unit_id = ?
                ORDER BY rowid
                """,
                (resolved_work_unit_id,),
            ).fetchall()
            return tuple(self._load_work_readiness_row(row) for row in rows)

    def bind_work_control_policy(
        self,
        policy: WorkControlPolicyV2,
        *,
        idempotency_key: Identifier,
    ) -> WorkControlPolicyV2:
        canonical = WorkControlPolicyV2.model_validate(policy.model_dump(mode="python"))
        policy_ref = work_control_policy_v2_ref(canonical)
        graph_id = canonical.resolved_job_work_graph_ref.object_id
        request_sha256 = _request_sha256({"work_control_policy_ref": policy_ref})
        scope = f"bind-work-control-policy:{graph_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_CONTROL_POLICY":
                    raise IdempotencyConflictError("work control policy response type is corrupt")
                stored = self._get_work_control_policy_by_id(connection, prior[1])
                if work_control_policy_v2_ref(stored) != policy_ref:
                    raise ImmutableResultError("stored work control policy identity differs from the request")
                return stored

            graph = self._get_job_work_graph_by_id(connection, graph_id)
            if resolved_job_work_graph_v2_ref(graph) != canonical.resolved_job_work_graph_ref:
                raise WorkControlPolicyError("work control policy graph reference is stale")
            if connection.execute(
                """
                SELECT 1
                FROM work_control_policies
                WHERE resolved_job_work_graph_id = ?
                   OR work_control_policy_id = ?
                """,
                (graph_id, canonical.work_control_policy_id),
            ).fetchone():
                raise ImmutableResultError("work graph already has an immutable control policy")

            now = self._clock()
            connection.execute(
                """
                INSERT INTO work_control_policies (
                    work_control_policy_id, resolved_job_work_graph_id,
                    policy_sha256, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    canonical.work_control_policy_id,
                    graph_id,
                    canonical.work_control_policy_sha256,
                    self._record_json(canonical),
                ),
            )
            job = self._get_record(
                connection,
                "jobs",
                "job_id",
                graph.job_id,
                JobRecord,
            )
            self._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job.job_id,
                aggregate_version=job.row_version,
                event_type="work-control-policy-bound",
                attributes=_attributes(policy_id=canonical.work_control_policy_id),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_CONTROL_POLICY",
                response_id=canonical.work_control_policy_id,
                created_at=now,
            )
            return canonical

    def get_work_control_policy(
        self,
        job_id: Identifier,
    ) -> WorkControlPolicyV2:
        with self._connect() as connection:
            graph = self._get_job_work_graph(connection, job_id)
            row = connection.execute(
                """
                SELECT work_control_policy_id
                FROM work_control_policies
                WHERE resolved_job_work_graph_id = ?
                """,
                (graph.resolved_job_work_graph_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"WorkControlPolicyV2 not found: {job_id}")
            return self._get_work_control_policy_by_id(
                connection,
                str(row["work_control_policy_id"]),
            )

    def get_work_lease(self, work_lease_id: Identifier) -> WorkLeaseV2:
        with self._connect() as connection:
            return self._get_work_lease(connection, work_lease_id)

    def list_work_leases(
        self,
        *,
        job_id: Identifier,
    ) -> tuple[WorkLeaseV2, ...]:
        with self._connect() as connection:
            self._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            rows = connection.execute(
                """
                SELECT leases.work_lease_id
                FROM work_leases AS leases
                JOIN resolved_work_units AS units
                  ON units.resolved_work_unit_id =
                     leases.resolved_work_unit_id
                WHERE units.job_id = ?
                ORDER BY leases.resolved_work_unit_id,
                         leases.attempt,
                         leases.work_lease_id
                """,
                (job_id,),
            ).fetchall()
            leases = tuple(
                self._get_work_lease(
                    connection,
                    str(row["work_lease_id"]),
                )
                for row in rows
            )
            for lease in leases:
                head = self._get_work_lease_head(
                    connection,
                    lease.work_lease_id,
                )
                if head != self._rebuild_work_lease_head(
                    connection,
                    lease,
                ):
                    raise ImmutableResultError("materialized work lease head differs from immutable events")
            return leases

    def get_work_lease_head(
        self,
        work_lease_id: Identifier,
    ) -> WorkLeaseHeadRecord:
        with self._connect() as connection:
            lease = self._get_work_lease(connection, work_lease_id)
            head = self._get_work_lease_head(connection, work_lease_id)
            rebuilt = self._rebuild_work_lease_head(connection, lease)
            if head != rebuilt:
                raise ImmutableResultError("materialized work lease head differs from immutable events")
            return head

    def list_work_lease_events(
        self,
        work_lease_id: Identifier,
    ) -> tuple[WorkLeaseEventV2, ...]:
        with self._connect() as connection:
            self._get_work_lease(connection, work_lease_id)
            rows = connection.execute(
                """
                SELECT work_lease_event_id, work_lease_id, lease_version,
                       event_kind, record_json
                FROM work_lease_events
                WHERE work_lease_id = ?
                ORDER BY lease_version
                """,
                (work_lease_id,),
            ).fetchall()
            return tuple(self._load_work_lease_event_row(row) for row in rows)

    def get_work_attempt_metrics_for_lease(
        self,
        work_lease_id: Identifier,
    ) -> WorkAttemptMetricsV2:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT work_attempt_metrics_id, job_id,
                       resolved_work_unit_id, work_lease_id,
                       attempt, record_json
                FROM work_attempt_metrics
                WHERE work_lease_id = ?
                """,
                (work_lease_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"WorkAttemptMetricsV2 not found for lease: {work_lease_id}")
            return self._load_work_attempt_metrics_row(connection, row)

    def list_work_attempt_metrics(
        self,
        *,
        job_id: Identifier,
    ) -> tuple[WorkAttemptMetricsV2, ...]:
        with self._connect() as connection:
            self._get_record(connection, "jobs", "job_id", job_id, JobRecord)
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
            return tuple(self._load_work_attempt_metrics_row(connection, row) for row in rows)

    def get_work_retry_decision(
        self,
        work_unit_id: Identifier,
    ) -> WorkRetryDecisionV2:
        decision = self.get_work_retry_decision_optional(work_unit_id)
        if decision is None:
            raise RecordNotFoundError(f"WorkRetryDecisionV2 not found: {work_unit_id}")
        return decision

    def get_work_retry_decision_optional(
        self,
        work_unit_id: Identifier,
    ) -> WorkRetryDecisionV2 | None:
        with self._connect() as connection:
            self._get_work_unit(connection, work_unit_id)
            row = connection.execute(
                """
                SELECT work_retry_decision_id, resolved_work_unit_id,
                       prior_lease_event_id, decision, record_json
                FROM work_retry_decisions
                WHERE resolved_work_unit_id = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (work_unit_id,),
            ).fetchone()
            if row is None:
                return None
            return self._load_work_retry_decision_row(connection, row)

    def acquire_work_lease(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        readiness_snapshot: WorkReadinessSnapshotV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        retry_decision: WorkRetryDecisionV2 | None,
        audit: ContractAudit,
        idempotency_key: Identifier,
    ) -> WorkLeaseV2:
        return self._acquire_work_lease(
            graph=graph,
            work_unit=work_unit,
            readiness_snapshot=readiness_snapshot,
            policy=policy,
            holder_ref=holder_ref,
            retry_decision=retry_decision,
            audit=audit,
            idempotency_key=idempotency_key,
            connection=None,
            control_authorizations=frozenset(),
        )

    def _acquire_work_lease(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        readiness_snapshot: WorkReadinessSnapshotV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        retry_decision: WorkRetryDecisionV2 | None,
        audit: ContractAudit,
        idempotency_key: Identifier,
        connection: sqlite3.Connection | None,
        control_authorizations: frozenset[Literal["RESOURCE", "MODEL"]],
    ) -> WorkLeaseV2:
        canonical_graph = ResolvedJobWorkGraphV2.model_validate(graph.model_dump(mode="python"))
        canonical_unit = ResolvedWorkUnitV2.model_validate(work_unit.model_dump(mode="python"))
        canonical_readiness = WorkReadinessSnapshotV2.model_validate(
            readiness_snapshot.model_dump(mode="python")
        )
        canonical_policy = WorkControlPolicyV2.model_validate(policy.model_dump(mode="python"))
        canonical_retry = (
            WorkRetryDecisionV2.model_validate(retry_decision.model_dump(mode="python"))
            if retry_decision is not None
            else None
        )
        request = {
            "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(canonical_graph),
            "work_unit_ref": resolved_work_unit_v2_ref(canonical_unit),
            "work_readiness_snapshot_ref": work_readiness_snapshot_v2_ref(canonical_readiness),
            "work_control_policy_ref": work_control_policy_v2_ref(canonical_policy),
            "holder_ref": holder_ref,
            "retry_decision_ref": (
                work_retry_decision_v2_ref(canonical_retry) if canonical_retry is not None else None
            ),
        }
        request_sha256 = _request_sha256(request)
        unit_id = canonical_unit.resolved_work_unit_id
        scope = f"acquire-work-lease:{unit_id}"
        transaction = self._transaction() if connection is None else nullcontext(connection)
        with transaction as connection:
            self._reject_control_graph_bypass(
                connection,
                canonical_graph.resolved_job_work_graph_id,
                control_authorizations=control_authorizations,
            )
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_LEASE":
                    raise IdempotencyConflictError("work lease response type is corrupt")
                return self._get_work_lease(connection, prior[1])

            prior_lease, attempt, eligible_at, retry_ref, job = self._validate_work_lease_admission(
                connection,
                graph=canonical_graph,
                work_unit=canonical_unit,
                readiness_snapshot=canonical_readiness,
                policy=canonical_policy,
                retry_decision=canonical_retry,
            )

            now = self._clock()
            fence = self._allocate_work_fence(connection, unit_id)
            graph_ref = resolved_job_work_graph_v2_ref(canonical_graph)
            unit_ref = resolved_work_unit_v2_ref(canonical_unit)
            readiness_ref = work_readiness_snapshot_v2_ref(canonical_readiness)
            policy_ref = work_control_policy_v2_ref(canonical_policy)
            dispatch = WorkDispatchDecisionV2.create(
                resolved_job_work_graph_ref=graph_ref,
                work_unit_ref=unit_ref,
                work_readiness_snapshot_ref=readiness_ref,
                work_control_policy_ref=policy_ref,
                retry_decision_ref=retry_ref,
                attempt=attempt,
                eligible_at=eligible_at,
                audit=_control_audit(
                    audit,
                    (
                        graph_ref,
                        unit_ref,
                        readiness_ref,
                        policy_ref,
                        *((retry_ref,) if retry_ref else ()),
                    ),
                ),
            )
            connection.execute(
                """
                INSERT INTO work_dispatch_decisions (
                    work_dispatch_decision_id, resolved_work_unit_id,
                    attempt, retry_decision_id, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    dispatch.work_dispatch_decision_id,
                    unit_id,
                    attempt,
                    canonical_retry.work_retry_decision_id if canonical_retry is not None else None,
                    self._record_json(dispatch),
                ),
            )
            dispatch_ref = work_dispatch_decision_v2_ref(dispatch)
            stage_run: StageRunRecord | None = None
            if canonical_unit.scope is not WorkUnitScopeV2.ARTIFACT_GROUP:
                retry_of = (
                    prior_lease.stage_run_ref.object_id
                    if prior_lease is not None and prior_lease.stage_run_ref is not None
                    else None
                )
                self._validate_retry(
                    connection,
                    job_id=canonical_unit.job_id,
                    item_id=canonical_unit.item_id,
                    stage=canonical_unit.stage,
                    attempt=attempt,
                    retry_of_stage_run_id=retry_of,
                )
                stage_run = StageRunRecord(
                    stage_run_id=_work_stage_run_id(unit_id, attempt),
                    job_id=canonical_unit.job_id,
                    item_id=canonical_unit.item_id,
                    stage=canonical_unit.stage,
                    attempt=attempt,
                    status=StageRunStatus.RUNNING,
                    principal_ref=holder_ref,
                    input_refs=tuple(
                        sorted(
                            (unit_ref, dispatch_ref),
                            key=_object_ref_key,
                        )
                    ),
                    retry_of_stage_run_id=retry_of,
                    row_version=1,
                    idempotency_key=idempotency_key,
                    created_at=now,
                    started_at=now,
                )
                connection.execute(
                    """
                    INSERT INTO stage_runs (
                        stage_run_id, job_id, item_id, stage, attempt,
                        status, row_version, idempotency_key,
                        retry_of_stage_run_id, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stage_run.stage_run_id,
                        stage_run.job_id,
                        stage_run.item_id,
                        stage_run.stage,
                        stage_run.attempt,
                        stage_run.status,
                        stage_run.row_version,
                        stage_run.idempotency_key,
                        stage_run.retry_of_stage_run_id,
                        self._record_json(stage_run),
                    ),
                )

            stage_run_ref = stage_run_record_ref(stage_run) if stage_run is not None else None
            lease = WorkLeaseV2.create(
                work_dispatch_decision_ref=dispatch_ref,
                work_unit_ref=unit_ref,
                work_control_policy_ref=policy_ref,
                holder_ref=holder_ref,
                fencing_token=fence,
                attempt=attempt,
                stage_run_ref=stage_run_ref,
                acquired_at=now,
                expires_at=now + timedelta(seconds=canonical_policy.lease_duration_seconds),
                audit=_control_audit(
                    audit,
                    (
                        dispatch_ref,
                        unit_ref,
                        policy_ref,
                        holder_ref,
                        *((stage_run_ref,) if stage_run_ref else ()),
                    ),
                ),
            )
            self._insert_work_lease(connection, lease)
            head = WorkLeaseHeadRecord(
                work_lease_id=lease.work_lease_id,
                work_unit_id=unit_id,
                state=WorkLeaseStateV2.ACTIVE,
                lease_version=0,
                fencing_token=fence,
                expires_at=lease.expires_at,
            )
            self._insert_work_lease_head(connection, head)
            if stage_run is not None:
                connection.execute(
                    """
                    INSERT INTO controlled_stage_runs (
                        stage_run_id, work_lease_id, fencing_token
                    ) VALUES (?, ?, ?)
                    """,
                    (stage_run.stage_run_id, lease.work_lease_id, fence),
                )

            aggregate_type: Literal["JOB", "ITEM", "STAGE_RUN"]
            if stage_run is not None:
                aggregate_type = "STAGE_RUN"
                aggregate_id = stage_run.stage_run_id
                aggregate_version = stage_run.row_version
            elif canonical_unit.item_id is not None:
                aggregate_type = "ITEM"
                item = self._get_record(
                    connection,
                    "items",
                    "item_id",
                    canonical_unit.item_id,
                    ItemRecord,
                )
                aggregate_id = item.item_id
                aggregate_version = item.row_version
            else:
                aggregate_type = "JOB"
                aggregate_id = job.job_id
                aggregate_version = job.row_version
            self._append_outbox(
                connection,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                aggregate_version=aggregate_version,
                event_type="work-lease-acquired",
                attributes=_attributes(
                    attempt=attempt,
                    fencing_token=fence,
                    work_lease_id=lease.work_lease_id,
                    work_unit_id=unit_id,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_LEASE",
                response_id=lease.work_lease_id,
                created_at=now,
            )
            return lease

    def heartbeat_work_lease(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        audit: ContractAudit,
        idempotency_key: Identifier,
    ) -> WorkLeaseEventV2:
        canonical_lease = WorkLeaseV2.model_validate(lease.model_dump(mode="python"))
        canonical_policy = WorkControlPolicyV2.model_validate(policy.model_dump(mode="python"))
        request = {
            "work_lease_ref": work_lease_v2_ref(canonical_lease),
            "work_control_policy_ref": work_control_policy_v2_ref(canonical_policy),
            "holder_ref": holder_ref,
            "expected_lease_version": expected_lease_version,
        }
        request_sha256 = _request_sha256(request)
        scope = f"heartbeat-work-lease:{canonical_lease.work_lease_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_work_lease_event(connection, prior[1])
            stored_lease, stored_policy, head = self._require_active_work_lease(
                connection,
                canonical_lease,
                canonical_policy,
                holder_ref=holder_ref,
                expected_lease_version=expected_lease_version,
            )
            now = self._clock()
            if now >= head.expires_at:
                raise StaleWorkLeaseError("work lease is expired")
            effective_expires_at = max(
                head.expires_at,
                now
                + timedelta(
                    seconds=stored_policy.heartbeat_extension_seconds,
                ),
            )
            lease_ref = work_lease_v2_ref(stored_lease)
            policy_ref = work_control_policy_v2_ref(stored_policy)
            event = WorkLeaseEventV2.create(
                work_lease_ref=lease_ref,
                work_unit_ref=stored_lease.work_unit_ref,
                event_kind=WorkLeaseEventKindV2.HEARTBEAT,
                lease_version=head.lease_version + 1,
                fencing_token=stored_lease.fencing_token,
                effective_expires_at=effective_expires_at,
                result_refs=(),
                failure=None,
                work_control_policy_ref=policy_ref,
                audit=_control_audit(
                    audit,
                    (
                        lease_ref,
                        stored_lease.work_unit_ref,
                        policy_ref,
                    ),
                ),
            )
            self._append_work_lease_event(connection, event)
            updated_head = head.model_copy(
                update={
                    "lease_version": event.lease_version,
                    "expires_at": effective_expires_at,
                }
            )
            self._update_work_lease_head(connection, updated_head, head.lease_version)
            self._append_work_control_outbox(
                connection,
                stored_lease,
                event_type="work-lease-heartbeat",
                event=event,
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_LEASE_EVENT",
                response_id=event.work_lease_event_id,
                created_at=now,
            )
            return event

    def complete_stage_work_lease(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        status: StageRunStatus,
        output_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        checkpoint_ref: ObjectRef | None,
        metrics_ref: ObjectRef | None,
        audit: ContractAudit,
        idempotency_key: Identifier,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> WorkLeaseEventV2:
        return self._complete_stage_work_lease(
            lease=lease,
            policy=policy,
            holder_ref=holder_ref,
            expected_lease_version=expected_lease_version,
            status=status,
            output_refs=output_refs,
            failure=failure,
            checkpoint_ref=checkpoint_ref,
            metrics_ref=metrics_ref,
            audit=audit,
            idempotency_key=idempotency_key,
            connection=None,
            control_authorizations=frozenset(),
            metrics_telemetry=telemetry,
        )

    def _complete_stage_work_lease(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        status: StageRunStatus,
        output_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        checkpoint_ref: ObjectRef | None,
        metrics_ref: ObjectRef | None,
        audit: ContractAudit,
        idempotency_key: Identifier,
        connection: sqlite3.Connection | None,
        control_authorizations: frozenset[Literal["RESOURCE", "MODEL"]],
        metrics_telemetry: ExecutionTelemetryV2 | None = None,
        metrics_model_profile_ref: ObjectRef | None = None,
        metrics_model_usage: ModelUsageV2 | None = None,
        metrics_resource_usage: ResourceUsageV2 | None = None,
        metrics_selected_route_kind: Literal["PROVIDER", "RUNTIME"] | None = None,
        metrics_selected_route_id: Identifier | None = None,
        metrics_selected_route_version: str | None = None,
        metrics_worker_version: str | None = None,
        metrics_artifact_slices: tuple[ArtifactMetricSliceV2, ...] = (),
    ) -> WorkLeaseEventV2:
        canonical_lease = WorkLeaseV2.model_validate(lease.model_dump(mode="python"))
        canonical_policy = WorkControlPolicyV2.model_validate(policy.model_dump(mode="python"))
        _validate_control_stage_outcome(status, failure)
        request = {
            "work_lease_ref": work_lease_v2_ref(canonical_lease),
            "work_control_policy_ref": work_control_policy_v2_ref(canonical_policy),
            "holder_ref": holder_ref,
            "expected_lease_version": expected_lease_version,
            "status": status,
            "output_refs": output_refs,
            "failure": failure,
            "checkpoint_ref": checkpoint_ref,
            "metrics_ref": metrics_ref,
            "metrics_telemetry": metrics_telemetry,
            "metrics_model_profile_ref": metrics_model_profile_ref,
            "metrics_model_usage": metrics_model_usage,
            "metrics_resource_usage": metrics_resource_usage,
            "metrics_selected_route_kind": metrics_selected_route_kind,
            "metrics_selected_route_id": metrics_selected_route_id,
            "metrics_selected_route_version": metrics_selected_route_version,
            "metrics_worker_version": metrics_worker_version,
            "metrics_artifact_slices": metrics_artifact_slices,
        }
        request_sha256 = _request_sha256(request)
        scope = f"complete-stage-work-lease:{canonical_lease.work_lease_id}"
        transaction = self._transaction() if connection is None else nullcontext(connection)
        with transaction as connection:
            self._reject_control_lease_bypass(
                connection,
                canonical_lease.work_lease_id,
                control_authorizations=control_authorizations,
            )
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_work_lease_event(connection, prior[1])
            stored_lease, stored_policy, head = self._require_active_work_lease(
                connection,
                canonical_lease,
                canonical_policy,
                holder_ref=holder_ref,
                expected_lease_version=expected_lease_version,
            )
            now = self._clock()
            if now >= head.expires_at:
                raise StaleWorkLeaseError("work lease is expired")
            if stored_lease.stage_run_ref is None:
                raise WorkControlPolicyError("stage completion requires a controlled StageRun")
            if metrics_ref is not None:
                raise WorkControlPolicyError("controlled completion metrics are compiler-owned")
            controlled = connection.execute(
                """
                SELECT work_lease_id, fencing_token
                FROM controlled_stage_runs
                WHERE stage_run_id = ?
                """,
                (stored_lease.stage_run_ref.object_id,),
            ).fetchone()
            if (
                controlled is None
                or str(controlled["work_lease_id"]) != stored_lease.work_lease_id
                or int(controlled["fencing_token"]) != stored_lease.fencing_token
            ):
                raise StaleWorkLeaseError("controlled StageRun fencing is stale")
            current_stage = self._get_record(
                connection,
                "stage_runs",
                "stage_run_id",
                stored_lease.stage_run_ref.object_id,
                StageRunRecord,
            )
            event_kind = _stage_status_event_kind(status, failure)
            attempt_metrics = self._create_attempt_metrics(
                connection,
                lease=stored_lease,
                policy=stored_policy,
                head=head,
                terminal_event_kind=event_kind,
                terminal_status=status,
                failure=failure,
                completed_at=now,
                model_profile_ref=metrics_model_profile_ref,
                model_usage=metrics_model_usage,
                resource_usage=metrics_resource_usage,
                telemetry=metrics_telemetry,
                selected_route_kind=metrics_selected_route_kind,
                selected_route_id=metrics_selected_route_id,
                selected_route_version=metrics_selected_route_version,
                worker_version=metrics_worker_version,
                artifact_slices=metrics_artifact_slices,
                result_refs=output_refs,
                output_refs=output_refs,
                audit=audit,
            )
            attempt_metrics_ref = work_attempt_metrics_v2_ref(attempt_metrics)
            result = self._complete_stage_run_in_transaction(
                connection,
                stage_result_id=_work_stage_result_id(
                    current_stage.stage_run_id,
                    status,
                ),
                current=current_stage,
                status=status,
                output_refs=output_refs,
                failure=failure,
                checkpoint_ref=checkpoint_ref,
                metrics_ref=attempt_metrics_ref,
                now=now,
            )
            result_ref = stage_result_record_ref(result)
            event = self._close_work_lease(
                connection,
                lease=stored_lease,
                policy=stored_policy,
                head=head,
                event_kind=event_kind,
                result_refs=(result_ref,),
                failure=failure,
                audit=audit,
            )
            if event_kind is WorkLeaseEventKindV2.RETRYABLE_FAILURE:
                self._create_work_retry_decision(
                    connection,
                    stored_lease,
                    stored_policy,
                    event,
                    audit,
                )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_LEASE_EVENT",
                response_id=event.work_lease_event_id,
                created_at=now,
            )
            return event

    def complete_artifact_group_work_lease(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        event_kind: WorkLeaseEventKindV2,
        result_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        audit: ContractAudit,
        idempotency_key: Identifier,
        telemetry: ExecutionTelemetryV2 | None = None,
        artifact_slices: tuple[ArtifactMetricSliceV2, ...] = (),
    ) -> WorkLeaseEventV2:
        return self._complete_artifact_group_work_lease(
            lease=lease,
            policy=policy,
            holder_ref=holder_ref,
            expected_lease_version=expected_lease_version,
            event_kind=event_kind,
            result_refs=result_refs,
            failure=failure,
            audit=audit,
            idempotency_key=idempotency_key,
            connection=None,
            control_authorizations=frozenset(),
            metrics_telemetry=telemetry,
            metrics_artifact_slices=artifact_slices,
        )

    def _complete_artifact_group_work_lease(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        event_kind: WorkLeaseEventKindV2,
        result_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        audit: ContractAudit,
        idempotency_key: Identifier,
        connection: sqlite3.Connection | None,
        control_authorizations: frozenset[Literal["RESOURCE", "MODEL"]],
        metrics_telemetry: ExecutionTelemetryV2 | None = None,
        metrics_model_profile_ref: ObjectRef | None = None,
        metrics_model_usage: ModelUsageV2 | None = None,
        metrics_resource_usage: ResourceUsageV2 | None = None,
        metrics_selected_route_kind: Literal["PROVIDER", "RUNTIME"] | None = None,
        metrics_selected_route_id: Identifier | None = None,
        metrics_selected_route_version: str | None = None,
        metrics_worker_version: str | None = None,
        metrics_artifact_slices: tuple[ArtifactMetricSliceV2, ...] = (),
    ) -> WorkLeaseEventV2:
        canonical_lease = WorkLeaseV2.model_validate(lease.model_dump(mode="python"))
        canonical_policy = WorkControlPolicyV2.model_validate(policy.model_dump(mode="python"))
        request = {
            "work_lease_ref": work_lease_v2_ref(canonical_lease),
            "work_control_policy_ref": work_control_policy_v2_ref(canonical_policy),
            "holder_ref": holder_ref,
            "expected_lease_version": expected_lease_version,
            "event_kind": event_kind,
            "result_refs": result_refs,
            "failure": failure,
            "metrics_telemetry": metrics_telemetry,
            "metrics_model_profile_ref": metrics_model_profile_ref,
            "metrics_model_usage": metrics_model_usage,
            "metrics_resource_usage": metrics_resource_usage,
            "metrics_selected_route_kind": metrics_selected_route_kind,
            "metrics_selected_route_id": metrics_selected_route_id,
            "metrics_selected_route_version": metrics_selected_route_version,
            "metrics_worker_version": metrics_worker_version,
            "metrics_artifact_slices": metrics_artifact_slices,
        }
        request_sha256 = _request_sha256(request)
        scope = f"complete-artifact-work-lease:{canonical_lease.work_lease_id}"
        transaction = self._transaction() if connection is None else nullcontext(connection)
        with transaction as connection:
            self._reject_control_lease_bypass(
                connection,
                canonical_lease.work_lease_id,
                control_authorizations=control_authorizations,
            )
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_work_lease_event(connection, prior[1])
            stored_lease, stored_policy, head = self._require_active_work_lease(
                connection,
                canonical_lease,
                canonical_policy,
                holder_ref=holder_ref,
                expected_lease_version=expected_lease_version,
            )
            now = self._clock()
            if now >= head.expires_at:
                raise StaleWorkLeaseError("work lease is expired")
            unit, _ = self._get_work_unit(
                connection,
                stored_lease.work_unit_ref.object_id,
            )
            if unit.scope is not WorkUnitScopeV2.ARTIFACT_GROUP or stored_lease.stage_run_ref is not None:
                raise WorkControlPolicyError("artifact completion requires a StageRun-free group lease")
            self._create_attempt_metrics(
                connection,
                lease=stored_lease,
                policy=stored_policy,
                head=head,
                terminal_event_kind=event_kind,
                terminal_status=_artifact_terminal_status(event_kind),
                failure=failure,
                completed_at=now,
                model_profile_ref=metrics_model_profile_ref,
                model_usage=metrics_model_usage,
                resource_usage=metrics_resource_usage,
                telemetry=metrics_telemetry,
                selected_route_kind=metrics_selected_route_kind,
                selected_route_id=metrics_selected_route_id,
                selected_route_version=metrics_selected_route_version,
                worker_version=metrics_worker_version,
                artifact_slices=metrics_artifact_slices,
                result_refs=result_refs,
                output_refs=tuple(
                    artifact.output_ref
                    for artifact in metrics_artifact_slices
                    if artifact.output_ref is not None
                ),
                audit=audit,
            )
            event = self._close_work_lease(
                connection,
                lease=stored_lease,
                policy=stored_policy,
                head=head,
                event_kind=event_kind,
                result_refs=result_refs,
                failure=failure,
                audit=audit,
            )
            if event_kind is WorkLeaseEventKindV2.RETRYABLE_FAILURE:
                self._create_work_retry_decision(
                    connection,
                    stored_lease,
                    stored_policy,
                    event,
                    audit,
                )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_LEASE_EVENT",
                response_id=event.work_lease_event_id,
                created_at=now,
            )
            return event

    def expire_work_lease(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        audit: ContractAudit,
        idempotency_key: Identifier,
    ) -> WorkLeaseEventV2:
        return self._expire_work_lease(
            lease=lease,
            policy=policy,
            audit=audit,
            idempotency_key=idempotency_key,
            connection=None,
            control_authorizations=frozenset(),
        )

    def _expire_work_lease(
        self,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        audit: ContractAudit,
        idempotency_key: Identifier,
        connection: sqlite3.Connection | None,
        control_authorizations: frozenset[Literal["RESOURCE", "MODEL"]],
    ) -> WorkLeaseEventV2:
        canonical_lease = WorkLeaseV2.model_validate(lease.model_dump(mode="python"))
        canonical_policy = WorkControlPolicyV2.model_validate(policy.model_dump(mode="python"))
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(canonical_lease),
                "work_control_policy_ref": work_control_policy_v2_ref(canonical_policy),
            }
        )
        scope = f"expire-work-lease:{canonical_lease.work_lease_id}"
        transaction = self._transaction() if connection is None else nullcontext(connection)
        with transaction as connection:
            self._reject_control_lease_bypass(
                connection,
                canonical_lease.work_lease_id,
                control_authorizations=control_authorizations,
            )
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_work_lease_event(connection, prior[1])
            stored_lease = self._get_work_lease(
                connection,
                canonical_lease.work_lease_id,
            )
            stored_policy = self._get_work_control_policy_by_id(
                connection,
                canonical_policy.work_control_policy_id,
            )
            if stored_lease != canonical_lease or stored_policy != canonical_policy:
                raise StaleWorkLeaseError("work lease or policy is stale")
            head = self._get_work_lease_head(
                connection,
                stored_lease.work_lease_id,
            )
            if head != self._rebuild_work_lease_head(
                connection,
                stored_lease,
            ):
                raise ImmutableResultError("materialized work lease head is corrupt")
            if head.state is not WorkLeaseStateV2.ACTIVE:
                event = self._get_latest_work_lease_event(
                    connection,
                    stored_lease.work_lease_id,
                )
                now = self._clock()
                self._record_idempotency(
                    connection,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    response_type="WORK_LEASE_EVENT",
                    response_id=event.work_lease_event_id,
                    created_at=now,
                )
                return event
            stored_lease, stored_policy, head = self._require_active_work_lease(
                connection,
                canonical_lease,
                canonical_policy,
                holder_ref=None,
                expected_lease_version=None,
            )
            now = self._clock()
            if now < head.expires_at:
                raise WorkControlPolicyError("work lease is not expired")
            retryable = stored_policy.retry_lease_expiry
            failure = FailureRecord(
                failure_class=FailureClass.INTERNAL,
                code=("work-lease-expired-retryable" if retryable else "work-lease-expired-terminal"),
                message="Work lease expired before fenced completion.",
                retryable=retryable,
            )
            status = StageRunStatus.RETRYABLE_FAILURE if retryable else StageRunStatus.TERMINAL_FAILURE
            attempt_metrics = self._create_attempt_metrics(
                connection,
                lease=stored_lease,
                policy=stored_policy,
                head=head,
                terminal_event_kind=WorkLeaseEventKindV2.EXPIRED,
                terminal_status=status,
                failure=failure,
                completed_at=now,
                model_profile_ref=None,
                model_usage=None,
                resource_usage=None,
                telemetry=None,
                selected_route_kind=None,
                selected_route_id=None,
                selected_route_version=None,
                worker_version=None,
                artifact_slices=(),
                result_refs=(),
                output_refs=(),
                audit=audit,
            )
            attempt_metrics_ref = work_attempt_metrics_v2_ref(attempt_metrics)
            result_refs: tuple[ObjectRef, ...] = ()
            if stored_lease.stage_run_ref is not None:
                stage = self._get_record(
                    connection,
                    "stage_runs",
                    "stage_run_id",
                    stored_lease.stage_run_ref.object_id,
                    StageRunRecord,
                )
                result = self._complete_stage_run_in_transaction(
                    connection,
                    stage_result_id=_work_stage_result_id(
                        stage.stage_run_id,
                        status,
                    ),
                    current=stage,
                    status=status,
                    output_refs=(),
                    failure=failure,
                    checkpoint_ref=None,
                    metrics_ref=attempt_metrics_ref,
                    now=now,
                )
                result_refs = (stage_result_record_ref(result),)
            event = self._close_work_lease(
                connection,
                lease=stored_lease,
                policy=stored_policy,
                head=head,
                event_kind=WorkLeaseEventKindV2.EXPIRED,
                result_refs=result_refs,
                failure=failure,
                audit=audit,
            )
            if retryable:
                self._create_work_retry_decision(
                    connection,
                    stored_lease,
                    stored_policy,
                    event,
                    audit,
                )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_LEASE_EVENT",
                response_id=event.work_lease_event_id,
                created_at=now,
            )
            return event

    def cancel_job_work(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        policy: WorkControlPolicyV2,
        reason_code: Identifier,
        audit: ContractAudit,
        idempotency_key: Identifier,
    ) -> WorkCancellationRecordV2:
        return self._cancel_job_work(
            graph=graph,
            policy=policy,
            reason_code=reason_code,
            audit=audit,
            idempotency_key=idempotency_key,
            connection=None,
            control_authorizations=frozenset(),
        )

    def _cancel_job_work(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        policy: WorkControlPolicyV2,
        reason_code: Identifier,
        audit: ContractAudit,
        idempotency_key: Identifier,
        connection: sqlite3.Connection | None,
        control_authorizations: frozenset[Literal["RESOURCE", "MODEL"]],
    ) -> WorkCancellationRecordV2:
        canonical_graph = ResolvedJobWorkGraphV2.model_validate(graph.model_dump(mode="python"))
        canonical_policy = WorkControlPolicyV2.model_validate(policy.model_dump(mode="python"))
        request_sha256 = _request_sha256(
            {
                "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(canonical_graph),
                "work_control_policy_ref": work_control_policy_v2_ref(canonical_policy),
                "reason_code": reason_code,
            }
        )
        scope = f"cancel-job-work:{canonical_graph.job_id}"
        transaction = self._transaction() if connection is None else nullcontext(connection)
        with transaction as connection:
            self._reject_control_graph_bypass(
                connection,
                canonical_graph.resolved_job_work_graph_id,
                control_authorizations=control_authorizations,
            )
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_work_cancellation(connection, prior[1])
            stored_graph = self._get_job_work_graph_by_id(
                connection,
                canonical_graph.resolved_job_work_graph_id,
            )
            stored_policy = self._get_work_control_policy_by_id(
                connection,
                canonical_policy.work_control_policy_id,
            )
            if stored_graph != canonical_graph or stored_policy != canonical_policy:
                raise WorkControlPolicyError("cancellation graph or policy is stale")
            job = self._get_record(
                connection,
                "jobs",
                "job_id",
                canonical_graph.job_id,
                JobRecord,
            )
            if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
                raise IllegalTransitionError("terminal job cannot be cancelled")
            if job.status is JobStatus.CANCELLED:
                raise WorkControlPolicyError("job is already cancelled")

            preserved_rows = connection.execute(
                """
                SELECT sr.record_json
                FROM stage_results AS sr
                JOIN stage_runs AS run
                  ON run.stage_run_id = sr.stage_run_id
                WHERE run.job_id = ?
                ORDER BY sr.rowid
                """,
                (job.job_id,),
            ).fetchall()
            preserved_stage_refs = tuple(
                stage_result_record_ref(self._load_record(row, StageResultRecord)) for row in preserved_rows
            )
            terminal_event_rows = connection.execute(
                """
                SELECT events.work_lease_event_id, events.work_lease_id,
                       events.lease_version, events.event_kind,
                       events.record_json
                FROM work_lease_events AS events
                JOIN work_leases AS leases
                  ON leases.work_lease_id = events.work_lease_id
                JOIN resolved_work_units AS units
                  ON units.resolved_work_unit_id =
                     leases.resolved_work_unit_id
                WHERE units.job_id = ?
                  AND events.event_kind != 'HEARTBEAT'
                ORDER BY events.rowid
                """,
                (job.job_id,),
            ).fetchall()
            preserved_control_refs = tuple(
                result_ref
                for row in terminal_event_rows
                for result_ref in self._load_work_lease_event_row(row).result_refs
            )
            preserved_refs = tuple(
                sorted(
                    set((*preserved_stage_refs, *preserved_control_refs)),
                    key=_object_ref_key,
                )
            )
            active_rows = connection.execute(
                """
                SELECT heads.work_lease_id
                FROM work_lease_heads AS heads
                JOIN resolved_work_units AS units
                  ON units.resolved_work_unit_id = heads.resolved_work_unit_id
                WHERE units.job_id = ? AND heads.state = 'ACTIVE'
                ORDER BY heads.rowid
                """,
                (job.job_id,),
            ).fetchall()
            cancelled_event_refs: list[ObjectRef] = []
            now = self._clock()
            for row in active_rows:
                active_lease = self._get_work_lease(
                    connection,
                    str(row["work_lease_id"]),
                )
                head = self._get_work_lease_head(
                    connection,
                    active_lease.work_lease_id,
                )
                failure = FailureRecord(
                    failure_class=FailureClass.INTERNAL,
                    code="work-cancelled",
                    message="Work cancelled by a durable Job cancellation request.",
                    retryable=False,
                )
                attempt_metrics = self._create_attempt_metrics(
                    connection,
                    lease=active_lease,
                    policy=stored_policy,
                    head=head,
                    terminal_event_kind=WorkLeaseEventKindV2.CANCELLED,
                    terminal_status=StageRunStatus.CANCELLED,
                    failure=failure,
                    completed_at=now,
                    model_profile_ref=None,
                    model_usage=None,
                    resource_usage=None,
                    telemetry=None,
                    selected_route_kind=None,
                    selected_route_id=None,
                    selected_route_version=None,
                    worker_version=None,
                    artifact_slices=(),
                    result_refs=(),
                    output_refs=(),
                    audit=audit,
                )
                attempt_metrics_ref = work_attempt_metrics_v2_ref(attempt_metrics)
                result_refs: tuple[ObjectRef, ...] = ()
                if active_lease.stage_run_ref is not None:
                    stage = self._get_record(
                        connection,
                        "stage_runs",
                        "stage_run_id",
                        active_lease.stage_run_ref.object_id,
                        StageRunRecord,
                    )
                    result = self._complete_stage_run_in_transaction(
                        connection,
                        stage_result_id=_work_stage_result_id(
                            stage.stage_run_id,
                            StageRunStatus.CANCELLED,
                        ),
                        current=stage,
                        status=StageRunStatus.CANCELLED,
                        output_refs=(),
                        failure=failure,
                        checkpoint_ref=None,
                        metrics_ref=attempt_metrics_ref,
                        now=now,
                    )
                    result_refs = (stage_result_record_ref(result),)
                event = self._close_work_lease(
                    connection,
                    lease=active_lease,
                    policy=stored_policy,
                    head=head,
                    event_kind=WorkLeaseEventKindV2.CANCELLED,
                    result_refs=result_refs,
                    failure=failure,
                    audit=audit,
                )
                cancelled_event_refs.append(work_lease_event_v2_ref(event))

            graph_ref = resolved_job_work_graph_v2_ref(stored_graph)
            policy_ref = work_control_policy_v2_ref(stored_policy)
            cancellation = WorkCancellationRecordV2.create(
                resolved_job_work_graph_ref=graph_ref,
                work_control_policy_ref=policy_ref,
                reason_code=reason_code,
                cancelled_lease_event_refs=tuple(cancelled_event_refs),
                preserved_result_refs=preserved_refs,
                audit=_control_audit(
                    audit,
                    (
                        graph_ref,
                        policy_ref,
                        *tuple(cancelled_event_refs),
                        *preserved_refs,
                    ),
                ),
            )
            connection.execute(
                """
                INSERT INTO work_cancellations (
                    work_cancellation_record_id,
                    resolved_job_work_graph_id, job_id, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    cancellation.work_cancellation_record_id,
                    stored_graph.resolved_job_work_graph_id,
                    job.job_id,
                    self._record_json(cancellation),
                ),
            )
            updated_job = JobRecord.model_validate(
                {
                    **job.model_dump(mode="python"),
                    "status": JobStatus.CANCELLED,
                    "row_version": job.row_version + 1,
                    "updated_at": now,
                }
            )
            self._update_record(
                connection,
                "jobs",
                "job_id",
                job.job_id,
                updated_job.status,
                updated_job.row_version,
                updated_job,
                job.row_version,
            )
            self._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job.job_id,
                aggregate_version=updated_job.row_version,
                event_type="job-work-cancelled",
                attributes=_attributes(
                    cancellation_id=cancellation.work_cancellation_record_id,
                    reason_code=reason_code,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_CANCELLATION",
                response_id=cancellation.work_cancellation_record_id,
                created_at=now,
            )
            return cancellation

    def transition_job(
        self,
        job_id: Identifier,
        target: JobStatus,
        *,
        expected_version: int,
        idempotency_key: Identifier,
        reason: str | None = None,
    ) -> JobRecord:
        request_sha256 = _request_sha256(
            {
                "job_id": job_id,
                "target": target,
                "expected_version": expected_version,
                "reason": reason,
            }
        )
        scope = f"transition-job:{job_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_record(connection, "jobs", "job_id", prior[1], JobRecord)
            updated = self._transition_job_in_transaction(
                connection,
                job_id=job_id,
                target=target,
                expected_version=expected_version,
                reason=reason,
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="JOB",
                response_id=updated.job_id,
                created_at=updated.updated_at,
            )
            return updated

    def _transition_job_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: Identifier,
        target: JobStatus,
        expected_version: int,
        reason: str | None,
    ) -> JobRecord:
        current = self._get_record(
            connection,
            "jobs",
            "job_id",
            job_id,
            JobRecord,
        )
        self._require_version(current.row_version, expected_version, job_id)
        self._require_transition(JOB_TRANSITIONS, current.status, target, "job")
        now = self._clock()
        updated = JobRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "status": target,
                "row_version": current.row_version + 1,
                "updated_at": now,
            }
        )
        self._update_record(
            connection,
            "jobs",
            "job_id",
            updated.job_id,
            updated.status,
            updated.row_version,
            updated,
            expected_version,
        )
        self._append_outbox(
            connection,
            aggregate_type="JOB",
            aggregate_id=updated.job_id,
            aggregate_version=updated.row_version,
            event_type="job-status-changed",
            attributes=_attributes(
                from_status=current.status,
                to_status=target,
                reason=reason,
            ),
        )
        return updated

    def create_item(
        self,
        job_id: Identifier,
        item_id: Identifier,
        *,
        idempotency_key: Identifier,
    ) -> ItemRecord:
        request_sha256 = _request_sha256(
            {
                "job_id": job_id,
                "item_id": item_id,
            }
        )
        scope = f"create-item:{job_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_record(connection, "items", "item_id", prior[1], ItemRecord)
            job = self._get_record(connection, "jobs", "job_id", job_id, JobRecord)
            if job.status in {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                raise IllegalTransitionError(f"cannot add item to terminal job {job_id}")
            now = self._clock()
            record = ItemRecord(
                item_id=item_id,
                job_id=job_id,
                row_version=0,
                idempotency_key=idempotency_key,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                """
                INSERT INTO items (
                    item_id, job_id, status, row_version,
                    idempotency_key, record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.item_id,
                    record.job_id,
                    record.status,
                    record.row_version,
                    record.idempotency_key,
                    self._record_json(record),
                ),
            )
            self._append_outbox(
                connection,
                aggregate_type="ITEM",
                aggregate_id=record.item_id,
                aggregate_version=record.row_version,
                event_type="item-created",
                attributes=_attributes(job_id=job_id),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="ITEM",
                response_id=record.item_id,
                created_at=now,
            )
            return record

    def get_item(self, item_id: Identifier) -> ItemRecord:
        with self._connect() as connection:
            return self._get_record(connection, "items", "item_id", item_id, ItemRecord)

    def list_items(self, job_id: Identifier) -> tuple[ItemRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT item_id, job_id, status, row_version,
                       idempotency_key, record_json
                FROM items
                WHERE job_id = ?
                ORDER BY item_id
                """,
                (job_id,),
            ).fetchall()
            items = tuple(self._load_item_row(row) for row in rows)
            graph_row = connection.execute(
                """
                SELECT resolved_job_work_graph_id
                FROM resolved_job_work_graphs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if graph_row is None:
                return items
            graph = self._get_job_work_graph_by_id(
                connection,
                str(graph_row["resolved_job_work_graph_id"]),
            )
            by_id = {item.item_id: item for item in items}
            if set(by_id) != set(graph.item_ids):
                raise ImmutableResultError("stored Item set differs from the job work graph")
            return tuple(by_id[item_id] for item_id in graph.item_ids)

    def transition_item(
        self,
        item_id: Identifier,
        target: ItemStatus,
        *,
        expected_version: int,
        idempotency_key: Identifier,
        reason: str | None = None,
    ) -> ItemRecord:
        request_sha256 = _request_sha256(
            {
                "item_id": item_id,
                "target": target,
                "expected_version": expected_version,
                "reason": reason,
            }
        )
        scope = f"transition-item:{item_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_record(connection, "items", "item_id", prior[1], ItemRecord)
            current = self._get_record(connection, "items", "item_id", item_id, ItemRecord)
            self._require_version(current.row_version, expected_version, item_id)
            self._require_transition(ITEM_TRANSITIONS, current.status, target, "item")
            now = self._clock()
            updated = ItemRecord.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "status": target,
                    "row_version": current.row_version + 1,
                    "updated_at": now,
                }
            )
            self._update_record(
                connection,
                "items",
                "item_id",
                updated.item_id,
                updated.status,
                updated.row_version,
                updated,
                expected_version,
            )
            self._append_outbox(
                connection,
                aggregate_type="ITEM",
                aggregate_id=updated.item_id,
                aggregate_version=updated.row_version,
                event_type="item-status-changed",
                attributes=_attributes(
                    from_status=current.status,
                    to_status=target,
                    reason=reason,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="ITEM",
                response_id=updated.item_id,
                created_at=now,
            )
            return updated

    def create_stage_run(
        self,
        *,
        stage_run_id: Identifier,
        job_id: Identifier,
        item_id: Identifier | None,
        stage: StageNameV2,
        attempt: int,
        principal_ref: ObjectRef,
        input_refs: tuple[ObjectRef, ...],
        idempotency_key: Identifier,
        retry_of_stage_run_id: Identifier | None = None,
    ) -> StageRunRecord:
        request = {
            "stage_run_id": stage_run_id,
            "job_id": job_id,
            "item_id": item_id,
            "stage": stage,
            "attempt": attempt,
            "principal_ref": principal_ref,
            "input_refs": input_refs,
            "retry_of_stage_run_id": retry_of_stage_run_id,
        }
        request_sha256 = _request_sha256(request)
        scope = f"create-stage-run:{job_id}:{item_id or '-'}:{stage}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_record(
                    connection,
                    "stage_runs",
                    "stage_run_id",
                    prior[1],
                    StageRunRecord,
                )
            self._get_record(connection, "jobs", "job_id", job_id, JobRecord)
            if item_id is not None:
                item = self._get_record(connection, "items", "item_id", item_id, ItemRecord)
                if item.job_id != job_id:
                    raise JobStoreError(f"item {item_id} does not belong to job {job_id}")
            self._validate_retry(
                connection,
                job_id=job_id,
                item_id=item_id,
                stage=stage,
                attempt=attempt,
                retry_of_stage_run_id=retry_of_stage_run_id,
            )
            now = self._clock()
            record = StageRunRecord(
                stage_run_id=stage_run_id,
                job_id=job_id,
                item_id=item_id,
                stage=stage,
                attempt=attempt,
                principal_ref=principal_ref,
                input_refs=input_refs,
                retry_of_stage_run_id=retry_of_stage_run_id,
                row_version=0,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            try:
                connection.execute(
                    """
                    INSERT INTO stage_runs (
                        stage_run_id, job_id, item_id, stage, attempt,
                        status, row_version, idempotency_key,
                        retry_of_stage_run_id, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.stage_run_id,
                        record.job_id,
                        record.item_id,
                        record.stage,
                        record.attempt,
                        record.status,
                        record.row_version,
                        record.idempotency_key,
                        record.retry_of_stage_run_id,
                        self._record_json(record),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise IllegalTransitionError(
                    f"an active {stage} stage run already exists for {job_id}/{item_id}"
                ) from exc
            self._append_outbox(
                connection,
                aggregate_type="STAGE_RUN",
                aggregate_id=record.stage_run_id,
                aggregate_version=record.row_version,
                event_type="stage-run-created",
                attributes=_attributes(stage=stage, attempt=attempt),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="STAGE_RUN",
                response_id=record.stage_run_id,
                created_at=now,
            )
            return record

    def get_stage_run(self, stage_run_id: Identifier) -> StageRunRecord:
        with self._connect() as connection:
            return self._get_record(
                connection,
                "stage_runs",
                "stage_run_id",
                stage_run_id,
                StageRunRecord,
            )

    def list_stage_runs(
        self,
        *,
        job_id: Identifier,
        item_id: Identifier | None = None,
        stage: StageNameV2 | None = None,
    ) -> tuple[StageRunRecord, ...]:
        query = """
            SELECT record_json
            FROM stage_runs
            WHERE job_id = ?
            """
        parameters: list[object] = [job_id]
        if item_id is not None:
            query += " AND item_id = ?"
            parameters.append(item_id)
        if stage is not None:
            query += " AND stage = ?"
            parameters.append(stage)
        query += " ORDER BY attempt, stage_run_id"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
            return tuple(self._load_record(row, StageRunRecord) for row in rows)

    def transition_stage_run(
        self,
        stage_run_id: Identifier,
        target: StageRunStatus,
        *,
        expected_version: int,
        idempotency_key: Identifier,
    ) -> StageRunRecord:
        if target != StageRunStatus.RUNNING:
            raise IllegalTransitionError(
                "terminal stage transitions require complete_stage_run and an immutable result"
            )
        request_sha256 = _request_sha256(
            {
                "stage_run_id": stage_run_id,
                "target": target,
                "expected_version": expected_version,
            }
        )
        scope = f"transition-stage-run:{stage_run_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_record(
                    connection,
                    "stage_runs",
                    "stage_run_id",
                    prior[1],
                    StageRunRecord,
                )
            current = self._get_record(
                connection,
                "stage_runs",
                "stage_run_id",
                stage_run_id,
                StageRunRecord,
            )
            self._require_version(current.row_version, expected_version, stage_run_id)
            self._require_transition(STAGE_TRANSITIONS, current.status, target, "stage run")
            now = self._clock()
            updated = StageRunRecord.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "status": target,
                    "row_version": current.row_version + 1,
                    "started_at": now,
                    "ended_at": current.ended_at,
                }
            )
            self._update_record(
                connection,
                "stage_runs",
                "stage_run_id",
                updated.stage_run_id,
                updated.status,
                updated.row_version,
                updated,
                expected_version,
            )
            self._append_outbox(
                connection,
                aggregate_type="STAGE_RUN",
                aggregate_id=updated.stage_run_id,
                aggregate_version=updated.row_version,
                event_type="stage-run-status-changed",
                attributes=_attributes(
                    from_status=current.status,
                    to_status=target,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="STAGE_RUN",
                response_id=updated.stage_run_id,
                created_at=now,
            )
            return updated

    def complete_stage_run(
        self,
        *,
        stage_result_id: Identifier,
        stage_run_id: Identifier,
        status: StageRunStatus,
        expected_version: int,
        idempotency_key: Identifier,
        output_refs: tuple[ObjectRef, ...] = (),
        failure: FailureRecord | None = None,
        checkpoint_ref: ObjectRef | None = None,
        metrics_ref: ObjectRef | None = None,
    ) -> StageResultRecord:
        request = {
            "stage_result_id": stage_result_id,
            "stage_run_id": stage_run_id,
            "status": status,
            "expected_version": expected_version,
            "output_refs": output_refs,
            "failure": failure,
            "checkpoint_ref": checkpoint_ref,
            "metrics_ref": metrics_ref,
        }
        request_sha256 = _request_sha256(request)
        scope = f"complete-stage-run:{stage_run_id}"
        with self._transaction() as connection:
            prior = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._get_record(
                    connection,
                    "stage_results",
                    "stage_result_id",
                    prior[1],
                    StageResultRecord,
                )
            if connection.execute(
                """
                SELECT 1
                FROM controlled_stage_runs
                WHERE stage_run_id = ?
                """,
                (stage_run_id,),
            ).fetchone():
                raise StaleWorkLeaseError("controlled StageRun requires fenced lease completion")
            current = self._get_record(
                connection,
                "stage_runs",
                "stage_run_id",
                stage_run_id,
                StageRunRecord,
            )
            self._require_version(current.row_version, expected_version, stage_run_id)
            now = self._clock()
            result = self._complete_stage_run_in_transaction(
                connection,
                stage_result_id=stage_result_id,
                current=current,
                status=status,
                output_refs=output_refs,
                failure=failure,
                checkpoint_ref=checkpoint_ref,
                metrics_ref=metrics_ref,
                now=now,
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="STAGE_RESULT",
                response_id=result.stage_result_id,
                created_at=now,
            )
            return result

    def get_stage_result(self, stage_result_id: Identifier) -> StageResultRecord:
        with self._connect() as connection:
            return self._get_record(
                connection,
                "stage_results",
                "stage_result_id",
                stage_result_id,
                StageResultRecord,
            )

    def get_stage_result_for_run(self, stage_run_id: Identifier) -> StageResultRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_json
                FROM stage_results
                WHERE stage_run_id = ?
                """,
                (stage_run_id,),
            ).fetchone()
            if row is None:
                return None
            return self._load_record(row, StageResultRecord)

    def list_outbox(self) -> tuple[OutboxEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_json
                FROM outbox_events
                ORDER BY rowid
                """
            ).fetchall()
            return tuple(self._load_record(row, OutboxEvent) for row in rows)

    def _get_job_spec(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> DatasetJobSpecV2:
        row = connection.execute(
            "SELECT job_spec_json FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"JobRecord not found: {job_id}")
        try:
            return DatasetJobSpecV2.model_validate_json(str(row["job_spec_json"]))
        except ValidationError as exc:
            raise ImmutableResultError("stored job spec is invalid") from exc

    def _get_job_work_graph(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> ResolvedJobWorkGraphV2:
        row = connection.execute(
            """
            SELECT resolved_job_work_graph_id
            FROM resolved_job_work_graphs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"ResolvedJobWorkGraphV2 not found: {job_id}")
        return self._get_job_work_graph_by_id(
            connection,
            str(row["resolved_job_work_graph_id"]),
        )

    def _get_job_work_graph_by_id(
        self,
        connection: sqlite3.Connection,
        graph_id: str,
    ) -> ResolvedJobWorkGraphV2:
        row = connection.execute(
            """
            SELECT resolved_job_work_graph_id, job_id, graph_sha256, record_json
            FROM resolved_job_work_graphs
            WHERE resolved_job_work_graph_id = ?
            """,
            (graph_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"ResolvedJobWorkGraphV2 not found: {graph_id}")
        try:
            graph = self._load_record(row, ResolvedJobWorkGraphV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored job work graph record is invalid") from exc
        if (
            str(row["resolved_job_work_graph_id"]) != graph.resolved_job_work_graph_id
            or str(row["job_id"]) != graph.job_id
            or str(row["graph_sha256"]) != graph.resolved_job_work_graph_sha256
        ):
            raise ImmutableResultError("stored job work graph columns are inconsistent")
        units = self._list_work_units_by_owner(
            connection,
            graph.resolved_job_work_graph_id,
        )
        if units != graph.work_units:
            raise ImmutableResultError("materialized work units differ from the job work graph")
        item_rows = connection.execute(
            """
            SELECT item_id, job_id, status, row_version,
                   idempotency_key, record_json
            FROM items
            WHERE job_id = ?
            ORDER BY item_id
            """,
            (graph.job_id,),
        ).fetchall()
        item_ids = {self._load_item_row(row).item_id for row in item_rows}
        if item_ids != set(graph.item_ids):
            raise ImmutableResultError("materialized Item set differs from the job work graph")
        return graph

    def _get_artifact_group_fanout_by_id(
        self,
        connection: sqlite3.Connection,
        fanout_id: str,
    ) -> ArtifactGroupFanoutV2:
        row = connection.execute(
            """
            SELECT artifact_group_fanout_id, parent_work_unit_id, record_json
            FROM artifact_group_fanouts
            WHERE artifact_group_fanout_id = ?
            """,
            (fanout_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"ArtifactGroupFanoutV2 not found: {fanout_id}")
        try:
            fanout = self._load_record(row, ArtifactGroupFanoutV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored artifact group fanout record is invalid") from exc
        if (
            str(row["artifact_group_fanout_id"]) != fanout.artifact_group_fanout_id
            or str(row["parent_work_unit_id"]) != fanout.parent_attachment_work_unit_ref.object_id
        ):
            raise ImmutableResultError("stored artifact group fanout columns are inconsistent")
        units = self._list_work_units_by_owner(
            connection,
            fanout.artifact_group_fanout_id,
        )
        if units != fanout.group_work_units:
            raise ImmutableResultError("materialized work units differ from the artifact fanout")
        return fanout

    def _get_semantic_review_fanout_by_id(
        self,
        connection: sqlite3.Connection,
        fanout_id: str,
    ) -> SemanticReviewFanoutV2:
        row = connection.execute(
            """
            SELECT semantic_review_fanout_id,
                   parent_work_unit_id,
                   record_json
            FROM semantic_review_fanouts
            WHERE semantic_review_fanout_id = ?
            """,
            (fanout_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"SemanticReviewFanoutV2 not found: {fanout_id}")
        try:
            fanout = self._load_record(
                row,
                SemanticReviewFanoutV2,
            )
        except ValidationError as exc:
            raise ImmutableResultError("stored semantic review fanout is invalid") from exc
        if (
            str(row["semantic_review_fanout_id"]) != fanout.semantic_review_fanout_id
            or str(row["parent_work_unit_id"]) != fanout.parent_item_quality_work_unit_ref.object_id
        ):
            raise ImmutableResultError("stored semantic review fanout columns are inconsistent")
        units = self._list_work_units_by_owner(
            connection,
            fanout.semantic_review_fanout_id,
        )
        expected = tuple(value.work_unit for value in fanout.round_work)
        if units != expected:
            raise ImmutableResultError("materialized work units differ from the semantic review fanout")
        return fanout

    def _get_work_readiness_snapshot(
        self,
        connection: sqlite3.Connection,
        snapshot_id: str,
    ) -> WorkReadinessSnapshotV2:
        row = connection.execute(
            """
            SELECT work_readiness_snapshot_id, resolved_work_unit_id,
                   record_json
            FROM work_readiness_snapshots
            WHERE work_readiness_snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"WorkReadinessSnapshotV2 not found: {snapshot_id}")
        return self._load_work_readiness_row(row)

    def _load_work_readiness_row(
        self,
        row: sqlite3.Row,
    ) -> WorkReadinessSnapshotV2:
        try:
            snapshot = self._load_record(row, WorkReadinessSnapshotV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored work readiness record is invalid") from exc
        if (
            str(row["work_readiness_snapshot_id"]) != snapshot.work_readiness_snapshot_id
            or str(row["resolved_work_unit_id"]) != snapshot.work_unit_ref.object_id
        ):
            raise ImmutableResultError("stored work readiness columns are inconsistent")
        return snapshot

    def _insert_work_units(
        self,
        connection: sqlite3.Connection,
        units: tuple[ResolvedWorkUnitV2, ...],
        *,
        owner_ref: str,
    ) -> None:
        for unit in units:
            connection.execute(
                """
                INSERT INTO resolved_work_units (
                    resolved_work_unit_id, job_id, item_id, stage,
                    scope, owner_ref, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    unit.resolved_work_unit_id,
                    unit.job_id,
                    unit.item_id,
                    unit.stage,
                    unit.scope,
                    owner_ref,
                    self._record_json(unit),
                ),
            )

    def _list_work_units_by_owner(
        self,
        connection: sqlite3.Connection,
        owner_ref: str,
    ) -> tuple[ResolvedWorkUnitV2, ...]:
        rows = connection.execute(
            """
            SELECT resolved_work_unit_id, job_id, item_id, stage, scope,
                   owner_ref, record_json
            FROM resolved_work_units
            WHERE owner_ref = ?
            ORDER BY rowid
            """,
            (owner_ref,),
        ).fetchall()
        return tuple(self._load_work_unit_row(row)[0] for row in rows)

    def _get_work_unit(
        self,
        connection: sqlite3.Connection,
        work_unit_id: str,
    ) -> tuple[ResolvedWorkUnitV2, str]:
        row = connection.execute(
            """
            SELECT resolved_work_unit_id, job_id, item_id, stage, scope,
                   owner_ref, record_json
            FROM resolved_work_units
            WHERE resolved_work_unit_id = ?
            """,
            (work_unit_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"ResolvedWorkUnitV2 not found: {work_unit_id}")
        return self._load_work_unit_row(row)

    def _load_work_unit_row(
        self,
        row: sqlite3.Row,
    ) -> tuple[ResolvedWorkUnitV2, str]:
        try:
            unit = self._load_record(row, ResolvedWorkUnitV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored resolved work unit is invalid") from exc
        row_item_id = str(row["item_id"]) if row["item_id"] is not None else None
        if (
            str(row["resolved_work_unit_id"]) != unit.resolved_work_unit_id
            or str(row["job_id"]) != unit.job_id
            or row_item_id != unit.item_id
            or str(row["stage"]) != unit.stage
            or str(row["scope"]) != unit.scope
        ):
            raise ImmutableResultError("stored resolved work unit columns are inconsistent")
        return unit, str(row["owner_ref"])

    def _load_item_row(self, row: sqlite3.Row) -> ItemRecord:
        try:
            item = self._load_record(row, ItemRecord)
        except ValidationError as exc:
            raise ImmutableResultError("stored Item record is invalid") from exc
        if (
            str(row["item_id"]) != item.item_id
            or str(row["job_id"]) != item.job_id
            or str(row["status"]) != item.status
            or int(row["row_version"]) != item.row_version
            or str(row["idempotency_key"]) != item.idempotency_key
        ):
            raise ImmutableResultError("stored Item columns are inconsistent")
        return item

    def _get_work_control_policy_by_id(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> WorkControlPolicyV2:
        row = connection.execute(
            """
            SELECT work_control_policy_id, resolved_job_work_graph_id,
                   policy_sha256, record_json
            FROM work_control_policies
            WHERE work_control_policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"WorkControlPolicyV2 not found: {policy_id}")
        try:
            policy = self._load_record(row, WorkControlPolicyV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored work control policy is invalid") from exc
        if (
            str(row["work_control_policy_id"]) != policy.work_control_policy_id
            or str(row["resolved_job_work_graph_id"]) != policy.resolved_job_work_graph_ref.object_id
            or str(row["policy_sha256"]) != policy.work_control_policy_sha256
        ):
            raise ImmutableResultError("stored work control policy columns are inconsistent")
        graph = self._get_job_work_graph_by_id(
            connection,
            policy.resolved_job_work_graph_ref.object_id,
        )
        if resolved_job_work_graph_v2_ref(graph) != policy.resolved_job_work_graph_ref:
            raise ImmutableResultError("stored work control policy graph ref is stale")
        return policy

    def _get_work_dispatch_decision(
        self,
        connection: sqlite3.Connection,
        decision_id: str,
    ) -> WorkDispatchDecisionV2:
        row = connection.execute(
            """
            SELECT work_dispatch_decision_id, resolved_work_unit_id,
                   attempt, retry_decision_id, record_json
            FROM work_dispatch_decisions
            WHERE work_dispatch_decision_id = ?
            """,
            (decision_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"WorkDispatchDecisionV2 not found: {decision_id}")
        try:
            decision = self._load_record(row, WorkDispatchDecisionV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored work dispatch decision is invalid") from exc
        retry_id = str(row["retry_decision_id"]) if row["retry_decision_id"] is not None else None
        if (
            str(row["work_dispatch_decision_id"]) != decision.work_dispatch_decision_id
            or str(row["resolved_work_unit_id"]) != decision.work_unit_ref.object_id
            or int(row["attempt"]) != decision.attempt
            or retry_id
            != (decision.retry_decision_ref.object_id if decision.retry_decision_ref is not None else None)
        ):
            raise ImmutableResultError("stored work dispatch columns are inconsistent")
        return decision

    def _get_work_lease(
        self,
        connection: sqlite3.Connection,
        lease_id: str,
    ) -> WorkLeaseV2:
        row = connection.execute(
            """
            SELECT work_lease_id, resolved_work_unit_id,
                   work_dispatch_decision_id, attempt,
                   fencing_token, record_json
            FROM work_leases
            WHERE work_lease_id = ?
            """,
            (lease_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"WorkLeaseV2 not found: {lease_id}")
        try:
            lease = self._load_record(row, WorkLeaseV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored work lease is invalid") from exc
        if (
            str(row["work_lease_id"]) != lease.work_lease_id
            or str(row["resolved_work_unit_id"]) != lease.work_unit_ref.object_id
            or str(row["work_dispatch_decision_id"]) != lease.work_dispatch_decision_ref.object_id
            or int(row["attempt"]) != lease.attempt
            or int(row["fencing_token"]) != lease.fencing_token
        ):
            raise ImmutableResultError("stored work lease columns are inconsistent")
        dispatch = self._get_work_dispatch_decision(
            connection,
            lease.work_dispatch_decision_ref.object_id,
        )
        if (
            work_dispatch_decision_v2_ref(dispatch) != lease.work_dispatch_decision_ref
            or dispatch.work_unit_ref != lease.work_unit_ref
            or dispatch.attempt != lease.attempt
        ):
            raise ImmutableResultError("stored work lease dispatch binding is stale")
        unit, _ = self._get_work_unit(connection, lease.work_unit_ref.object_id)
        if resolved_work_unit_v2_ref(unit) != lease.work_unit_ref:
            raise ImmutableResultError("stored work lease unit ref is stale")
        policy = self._get_work_control_policy_by_id(
            connection,
            lease.work_control_policy_ref.object_id,
        )
        if work_control_policy_v2_ref(policy) != lease.work_control_policy_ref:
            raise ImmutableResultError("stored work lease policy ref is stale")
        if lease.stage_run_ref is not None:
            stage = self._get_record(
                connection,
                "stage_runs",
                "stage_run_id",
                lease.stage_run_ref.object_id,
                StageRunRecord,
            )
            if stage_run_record_ref(stage) != lease.stage_run_ref:
                raise ImmutableResultError("stored work lease StageRun ref is stale")
        return lease

    def _get_work_lease_head(
        self,
        connection: sqlite3.Connection,
        lease_id: str,
    ) -> WorkLeaseHeadRecord:
        row = connection.execute(
            """
            SELECT work_lease_id, resolved_work_unit_id, state,
                   lease_version, fencing_token, expires_at, record_json
            FROM work_lease_heads
            WHERE work_lease_id = ?
            """,
            (lease_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"WorkLeaseHeadRecord not found: {lease_id}")
        try:
            head = self._load_record(row, WorkLeaseHeadRecord)
        except ValidationError as exc:
            raise ImmutableResultError("stored work lease head is invalid") from exc
        if (
            str(row["work_lease_id"]) != head.work_lease_id
            or str(row["resolved_work_unit_id"]) != head.work_unit_id
            or str(row["state"]) != head.state
            or int(row["lease_version"]) != head.lease_version
            or int(row["fencing_token"]) != head.fencing_token
            or str(row["expires_at"]) != head.expires_at.isoformat()
        ):
            raise ImmutableResultError("stored work lease head columns are inconsistent")
        return head

    def _get_work_lease_event(
        self,
        connection: sqlite3.Connection,
        event_id: str,
    ) -> WorkLeaseEventV2:
        row = connection.execute(
            """
            SELECT work_lease_event_id, work_lease_id, lease_version,
                   event_kind, record_json
            FROM work_lease_events
            WHERE work_lease_event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"WorkLeaseEventV2 not found: {event_id}")
        return self._load_work_lease_event_row(row)

    def _get_latest_work_lease_event(
        self,
        connection: sqlite3.Connection,
        lease_id: str,
    ) -> WorkLeaseEventV2:
        row = connection.execute(
            """
            SELECT work_lease_event_id, work_lease_id, lease_version,
                   event_kind, record_json
            FROM work_lease_events
            WHERE work_lease_id = ?
            ORDER BY lease_version DESC
            LIMIT 1
            """,
            (lease_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("terminal work lease has no immutable event")
        return self._load_work_lease_event_row(row)

    def _load_work_lease_event_row(
        self,
        row: sqlite3.Row,
    ) -> WorkLeaseEventV2:
        try:
            event = self._load_record(row, WorkLeaseEventV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored work lease event is invalid") from exc
        if (
            str(row["work_lease_event_id"]) != event.work_lease_event_id
            or str(row["work_lease_id"]) != event.work_lease_ref.object_id
            or int(row["lease_version"]) != event.lease_version
            or str(row["event_kind"]) != event.event_kind
        ):
            raise ImmutableResultError("stored work lease event columns are inconsistent")
        return event

    def _load_work_retry_decision_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> WorkRetryDecisionV2:
        try:
            decision = self._load_record(row, WorkRetryDecisionV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored work retry decision is invalid") from exc
        if (
            str(row["work_retry_decision_id"]) != decision.work_retry_decision_id
            or str(row["resolved_work_unit_id"]) != decision.work_unit_ref.object_id
            or str(row["prior_lease_event_id"]) != decision.prior_lease_event_ref.object_id
            or str(row["decision"]) != decision.decision
        ):
            raise ImmutableResultError("stored work retry decision columns are inconsistent")
        unit, _ = self._get_work_unit(
            connection,
            decision.work_unit_ref.object_id,
        )
        event = self._get_work_lease_event(
            connection,
            decision.prior_lease_event_ref.object_id,
        )
        policy = self._get_work_control_policy_by_id(
            connection,
            decision.work_control_policy_ref.object_id,
        )
        if (
            resolved_work_unit_v2_ref(unit) != decision.work_unit_ref
            or work_lease_event_v2_ref(event) != decision.prior_lease_event_ref
            or event.work_unit_ref != decision.work_unit_ref
            or event.result_refs != decision.prior_result_refs
            or work_control_policy_v2_ref(policy) != decision.work_control_policy_ref
        ):
            raise ImmutableResultError("stored work retry decision binding is stale")
        return decision

    def _get_latest_work_retry_decision(
        self,
        connection: sqlite3.Connection,
        unit_id: str,
    ) -> WorkRetryDecisionV2:
        row = connection.execute(
            """
            SELECT work_retry_decision_id, resolved_work_unit_id,
                   prior_lease_event_id, decision, record_json
            FROM work_retry_decisions
            WHERE resolved_work_unit_id = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (unit_id,),
        ).fetchone()
        if row is None:
            raise WorkControlPolicyError("work unit has no retry decision")
        return self._load_work_retry_decision_row(connection, row)

    def _get_latest_work_readiness(
        self,
        connection: sqlite3.Connection,
        unit_id: str,
    ) -> WorkReadinessSnapshotV2:
        row = connection.execute(
            """
            SELECT work_readiness_snapshot_id, resolved_work_unit_id,
                   record_json
            FROM work_readiness_snapshots
            WHERE resolved_work_unit_id = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (unit_id,),
        ).fetchone()
        if row is None:
            raise WorkControlPolicyError("work lease requires persisted readiness")
        return self._load_work_readiness_row(row)

    def _get_latest_work_lease_for_unit(
        self,
        connection: sqlite3.Connection,
        unit_id: str,
    ) -> WorkLeaseV2 | None:
        row = connection.execute(
            """
            SELECT work_lease_id
            FROM work_leases
            WHERE resolved_work_unit_id = ?
            ORDER BY attempt DESC
            LIMIT 1
            """,
            (unit_id,),
        ).fetchone()
        if row is None:
            return None
        return self._get_work_lease(connection, str(row["work_lease_id"]))

    def _get_work_cancellation(
        self,
        connection: sqlite3.Connection,
        cancellation_id: str,
    ) -> WorkCancellationRecordV2:
        row = connection.execute(
            """
            SELECT work_cancellation_record_id,
                   resolved_job_work_graph_id, job_id, record_json
            FROM work_cancellations
            WHERE work_cancellation_record_id = ?
            """,
            (cancellation_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"WorkCancellationRecordV2 not found: {cancellation_id}")
        try:
            record = self._load_record(row, WorkCancellationRecordV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored work cancellation is invalid") from exc
        graph = self._get_job_work_graph_by_id(
            connection,
            record.resolved_job_work_graph_ref.object_id,
        )
        if (
            str(row["work_cancellation_record_id"]) != record.work_cancellation_record_id
            or str(row["resolved_job_work_graph_id"]) != record.resolved_job_work_graph_ref.object_id
            or str(row["job_id"]) != graph.job_id
            or record.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph)
        ):
            raise ImmutableResultError("stored work cancellation columns are inconsistent")
        policy = self._get_work_control_policy_by_id(
            connection,
            record.work_control_policy_ref.object_id,
        )
        if work_control_policy_v2_ref(policy) != record.work_control_policy_ref:
            raise ImmutableResultError("stored work cancellation policy ref is stale")
        for event_ref in record.cancelled_lease_event_refs:
            event = self._get_work_lease_event(
                connection,
                event_ref.object_id,
            )
            if (
                work_lease_event_v2_ref(event) != event_ref
                or event.event_kind is not WorkLeaseEventKindV2.CANCELLED
            ):
                raise ImmutableResultError("stored work cancellation event ref is stale")
        stage_rows = connection.execute(
            """
            SELECT results.record_json
            FROM stage_results AS results
            JOIN stage_runs AS runs
              ON runs.stage_run_id = results.stage_run_id
            WHERE runs.job_id = ?
            """,
            (graph.job_id,),
        ).fetchall()
        available_refs = {
            stage_result_record_ref(self._load_record(stage_row, StageResultRecord))
            for stage_row in stage_rows
        }
        event_rows = connection.execute(
            """
            SELECT events.work_lease_event_id, events.work_lease_id,
                   events.lease_version, events.event_kind,
                   events.record_json
            FROM work_lease_events AS events
            JOIN work_leases AS leases
              ON leases.work_lease_id = events.work_lease_id
            JOIN resolved_work_units AS units
              ON units.resolved_work_unit_id = leases.resolved_work_unit_id
            WHERE units.job_id = ?
            """,
            (graph.job_id,),
        ).fetchall()
        available_refs.update(
            result_ref
            for event_row in event_rows
            for result_ref in self._load_work_lease_event_row(event_row).result_refs
        )
        if not set(record.preserved_result_refs) <= available_refs:
            raise ImmutableResultError("stored work cancellation preserved refs are stale")
        return record

    def _validate_work_unit_graph_owner(
        self,
        connection: sqlite3.Connection,
        graph: ResolvedJobWorkGraphV2,
        unit: ResolvedWorkUnitV2,
        owner_ref: str,
    ) -> None:
        if owner_ref == graph.resolved_job_work_graph_id:
            if unit not in graph.work_units:
                raise WorkControlPolicyError("work unit is not owned by the graph")
            return
        if unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP:
            fanout = self._get_artifact_group_fanout_by_id(connection, owner_ref)
            if (
                fanout.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph)
                or unit not in fanout.group_work_units
            ):
                raise WorkControlPolicyError("artifact work unit is not owned by the graph")
            return
        semantic_fanout = self._get_semantic_review_fanout_by_id(
            connection,
            owner_ref,
        )
        if semantic_fanout.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(
            graph
        ) or unit not in tuple(value.work_unit for value in semantic_fanout.round_work):
            raise WorkControlPolicyError("semantic review work unit is not owned by the graph")

    def _allocate_work_fence(
        self,
        connection: sqlite3.Connection,
        unit_id: str,
    ) -> int:
        row = connection.execute(
            """
            SELECT last_fencing_token
            FROM work_lease_fences
            WHERE resolved_work_unit_id = ?
            """,
            (unit_id,),
        ).fetchone()
        if row is None:
            fence = 1
            connection.execute(
                """
                INSERT INTO work_lease_fences (
                    resolved_work_unit_id, last_fencing_token
                ) VALUES (?, ?)
                """,
                (unit_id, fence),
            )
            return fence
        fence = int(row["last_fencing_token"]) + 1
        connection.execute(
            """
            UPDATE work_lease_fences
            SET last_fencing_token = ?
            WHERE resolved_work_unit_id = ?
            """,
            (fence, unit_id),
        )
        return fence

    def _insert_work_lease(
        self,
        connection: sqlite3.Connection,
        lease: WorkLeaseV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO work_leases (
                work_lease_id, resolved_work_unit_id,
                work_dispatch_decision_id, attempt,
                fencing_token, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                lease.work_lease_id,
                lease.work_unit_ref.object_id,
                lease.work_dispatch_decision_ref.object_id,
                lease.attempt,
                lease.fencing_token,
                self._record_json(lease),
            ),
        )

    def _insert_work_lease_head(
        self,
        connection: sqlite3.Connection,
        head: WorkLeaseHeadRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO work_lease_heads (
                work_lease_id, resolved_work_unit_id, state,
                lease_version, fencing_token, expires_at, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                head.work_lease_id,
                head.work_unit_id,
                head.state,
                head.lease_version,
                head.fencing_token,
                head.expires_at.isoformat(),
                self._record_json(head),
            ),
        )

    def _update_work_lease_head(
        self,
        connection: sqlite3.Connection,
        head: WorkLeaseHeadRecord,
        expected_version: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE work_lease_heads
            SET state = ?, lease_version = ?, expires_at = ?, record_json = ?
            WHERE work_lease_id = ? AND lease_version = ?
            """,
            (
                head.state,
                head.lease_version,
                head.expires_at.isoformat(),
                self._record_json(head),
                head.work_lease_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ConcurrencyConflictError(f"concurrent work lease update: {head.work_lease_id}")

    def _require_active_work_lease(
        self,
        connection: sqlite3.Connection,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        *,
        holder_ref: ObjectRef | None,
        expected_lease_version: int | None,
    ) -> tuple[WorkLeaseV2, WorkControlPolicyV2, WorkLeaseHeadRecord]:
        stored_lease = self._get_work_lease(connection, lease.work_lease_id)
        stored_policy = self._get_work_control_policy_by_id(
            connection,
            policy.work_control_policy_id,
        )
        if stored_lease != lease or stored_policy != policy:
            raise StaleWorkLeaseError("work lease or policy is stale")
        if work_control_policy_v2_ref(stored_policy) != stored_lease.work_control_policy_ref:
            raise StaleWorkLeaseError("work lease policy binding is stale")
        head = self._get_work_lease_head(connection, stored_lease.work_lease_id)
        rebuilt = self._rebuild_work_lease_head(connection, stored_lease)
        if head != rebuilt:
            raise ImmutableResultError("materialized work lease head is corrupt")
        if head.state is not WorkLeaseStateV2.ACTIVE:
            raise StaleWorkLeaseError("work lease is terminal")
        if holder_ref is not None and holder_ref != stored_lease.holder_ref:
            raise StaleWorkLeaseError("work lease holder does not match")
        if expected_lease_version is not None and head.lease_version != expected_lease_version:
            raise StaleWorkLeaseError("work lease version is stale")
        if head.fencing_token != stored_lease.fencing_token:
            raise StaleWorkLeaseError("work lease fencing token is stale")
        return stored_lease, stored_policy, head

    def _append_work_lease_event(
        self,
        connection: sqlite3.Connection,
        event: WorkLeaseEventV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO work_lease_events (
                work_lease_event_id, work_lease_id,
                lease_version, event_kind, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.work_lease_event_id,
                event.work_lease_ref.object_id,
                event.lease_version,
                event.event_kind,
                self._record_json(event),
            ),
        )

    def _create_attempt_metrics(
        self,
        connection: sqlite3.Connection,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        head: WorkLeaseHeadRecord,
        terminal_event_kind: WorkLeaseEventKindV2,
        terminal_status: StageRunStatus,
        failure: FailureRecord | None,
        completed_at: datetime,
        model_profile_ref: ObjectRef | None,
        model_usage: ModelUsageV2 | None,
        resource_usage: ResourceUsageV2 | None,
        telemetry: ExecutionTelemetryV2 | None,
        selected_route_kind: Literal["PROVIDER", "RUNTIME"] | None,
        selected_route_id: Identifier | None,
        selected_route_version: str | None,
        worker_version: str | None,
        artifact_slices: tuple[ArtifactMetricSliceV2, ...],
        result_refs: tuple[ObjectRef, ...],
        output_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> WorkAttemptMetricsV2:
        unit, _ = self._get_work_unit(connection, lease.work_unit_ref.object_id)
        dispatch = self._get_work_dispatch_decision(
            connection,
            lease.work_dispatch_decision_ref.object_id,
        )
        readiness = self._get_work_readiness_snapshot(
            connection,
            dispatch.work_readiness_snapshot_ref.object_id,
        )
        if (
            dispatch.work_unit_ref != lease.work_unit_ref
            or readiness.work_unit_ref != lease.work_unit_ref
            or readiness.readiness is not WorkReadinessV2.READY
            or policy.resolved_job_work_graph_ref != readiness.resolved_job_work_graph_ref
        ):
            raise ImmutableResultError("attempt metrics source lineage is stale or inconsistent")
        if model_usage is None:
            model_row = connection.execute(
                """
                SELECT work_model_reservation_id
                FROM work_model_reservations
                WHERE work_lease_id = ?
                """,
                (lease.work_lease_id,),
            ).fetchone()
            if model_row is not None:
                model_reservation = self._get_record(
                    connection,
                    "work_model_reservations",
                    "work_model_reservation_id",
                    str(model_row["work_model_reservation_id"]),
                    WorkModelReservationV2,
                )
                demand = self._get_record(
                    connection,
                    "work_model_demands",
                    "work_model_demand_id",
                    model_reservation.work_model_demand_ref.object_id,
                    WorkModelDemandV2,
                )
                model_profile_ref = demand.model_profile_ref
                model_usage = ModelUsageV2.conservative(
                    requests=model_reservation.request_allowance,
                    tokens=model_reservation.token_allowance,
                )
        if resource_usage is None:
            resource_row = connection.execute(
                """
                SELECT work_resource_reservation_id
                FROM work_resource_reservations
                WHERE work_lease_id = ?
                """,
                (lease.work_lease_id,),
            ).fetchone()
            if resource_row is not None:
                resource_reservation = self._get_record(
                    connection,
                    "work_resource_reservations",
                    "work_resource_reservation_id",
                    str(resource_row["work_resource_reservation_id"]),
                    WorkResourceReservationV2,
                )
                resource_usage = ResourceUsageV2(observed=resource_reservation.budget_allowance)
        observed = telemetry or ExecutionTelemetryV2.unavailable(observed_at=completed_at)
        observed_ref = _object_ref_from_facade(execution_telemetry_ref(observed))
        if observed.reported_cost.availability is TelemetryAvailabilityV2.REPORTED:
            amount = observed.reported_cost.amount_microusd
            if amount is None:
                raise ImmutableResultError("reported telemetry cost has no amount")
            cost = CostObservationV2.reported(
                amount_microusd=amount,
                source_ref=observed_ref,
            )
        else:
            cost = CostObservationV2.unavailable()
        tool_metrics = tuple(
            ToolMetricV2(
                tool_family=item.tool_family.value,
                calls=item.calls,
            )
            for item in observed.tool_family_counts
        )
        prior_metrics_ref: ObjectRef | None = None
        if lease.attempt > 1:
            prior_row = connection.execute(
                """
                SELECT metrics.work_attempt_metrics_id, metrics.job_id,
                       metrics.resolved_work_unit_id, metrics.work_lease_id,
                       metrics.attempt, metrics.record_json
                FROM work_attempt_metrics AS metrics
                JOIN work_leases AS leases
                  ON leases.work_lease_id = metrics.work_lease_id
                WHERE leases.resolved_work_unit_id = ?
                  AND leases.attempt = ?
                """,
                (unit.resolved_work_unit_id, lease.attempt - 1),
            ).fetchone()
            if prior_row is None:
                raise ImmutableResultError("retry attempt metrics are missing the prior attempt")
            prior_metrics_ref = work_attempt_metrics_v2_ref(
                self._load_work_attempt_metrics_row(connection, prior_row)
            )
        refs = (
            policy.resolved_job_work_graph_ref,
            lease.work_unit_ref,
            work_lease_v2_ref(lease),
            *((lease.stage_run_ref,) if lease.stage_run_ref else ()),
            *((unit.artifact_execution_group_ref,) if unit.artifact_execution_group_ref else ()),
            *((model_profile_ref,) if model_profile_ref else ()),
            *((prior_metrics_ref,) if prior_metrics_ref else ()),
            *result_refs,
            *output_refs,
            *(
                ref
                for artifact in artifact_slices
                for ref in (
                    *((artifact.telemetry_ref,) if artifact.telemetry_ref else ()),
                    *((artifact.output_ref,) if artifact.output_ref else ()),
                    *((artifact.cost.source_ref,) if artifact.cost.source_ref else ()),
                )
            ),
            observed_ref,
        )
        metrics = WorkAttemptMetricsV2.create(
            resolved_job_work_graph_ref=policy.resolved_job_work_graph_ref,
            work_unit_ref=lease.work_unit_ref,
            work_lease_ref=work_lease_v2_ref(lease),
            stage_run_ref=lease.stage_run_ref,
            job_id=unit.job_id,
            item_id=unit.item_id,
            scope=unit.scope,
            stage=unit.stage,
            artifact_execution_group_ref=unit.artifact_execution_group_ref,
            attempt=lease.attempt,
            fencing_token=lease.fencing_token,
            terminal_lease_version=head.lease_version + 1,
            terminal_event_kind=terminal_event_kind,
            terminal_status=terminal_status,
            reason_code=failure.code if failure is not None else None,
            ready_at=readiness.audit.created_at,
            eligible_at=dispatch.eligible_at,
            acquired_at=lease.acquired_at,
            completed_at=completed_at,
            model_profile_ref=model_profile_ref,
            model_usage=model_usage,
            resource_usage=resource_usage,
            telemetry_ref=observed_ref,
            tool_metrics=tool_metrics,
            cost=cost,
            selected_route_kind=selected_route_kind,
            selected_route_id=selected_route_id,
            selected_route_version=selected_route_version,
            worker_version=worker_version,
            retry_of_attempt_metrics_ref=prior_metrics_ref,
            result_refs=result_refs,
            output_refs=output_refs,
            artifact_slices=artifact_slices,
            audit=_observability_audit(audit, tuple(set(refs))),
        )
        connection.execute(
            """
            INSERT INTO work_attempt_metrics (
                work_attempt_metrics_id, job_id,
                resolved_work_unit_id, work_lease_id,
                attempt, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                metrics.work_attempt_metrics_id,
                metrics.job_id,
                metrics.work_unit_ref.object_id,
                metrics.work_lease_ref.object_id,
                metrics.attempt,
                self._record_json(metrics),
            ),
        )
        return metrics

    def _load_work_attempt_metrics_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> WorkAttemptMetricsV2:
        try:
            metrics = self._load_record(row, WorkAttemptMetricsV2)
        except ValidationError as exc:
            raise ImmutableResultError("stored work attempt metrics are invalid") from exc
        if (
            str(row["work_attempt_metrics_id"]) != metrics.work_attempt_metrics_id
            or str(row["job_id"]) != metrics.job_id
            or str(row["resolved_work_unit_id"]) != metrics.work_unit_ref.object_id
            or str(row["work_lease_id"]) != metrics.work_lease_ref.object_id
            or int(row["attempt"]) != metrics.attempt
        ):
            raise ImmutableResultError("stored work attempt metrics columns are inconsistent")
        lease = self._get_work_lease(
            connection,
            metrics.work_lease_ref.object_id,
        )
        if (
            work_lease_v2_ref(lease) != metrics.work_lease_ref
            or lease.work_unit_ref != metrics.work_unit_ref
            or lease.attempt != metrics.attempt
            or lease.fencing_token != metrics.fencing_token
        ):
            raise ImmutableResultError("stored work attempt metrics lease binding is stale")
        return metrics

    def _close_work_lease(
        self,
        connection: sqlite3.Connection,
        *,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        head: WorkLeaseHeadRecord,
        event_kind: WorkLeaseEventKindV2,
        result_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        audit: ContractAudit,
    ) -> WorkLeaseEventV2:
        if event_kind is WorkLeaseEventKindV2.HEARTBEAT:
            raise WorkControlPolicyError("terminal lease close cannot be a heartbeat")
        lease_ref = work_lease_v2_ref(lease)
        policy_ref = work_control_policy_v2_ref(policy)
        event_failure = _content_free_control_failure(failure) if failure is not None else None
        event = WorkLeaseEventV2.create(
            work_lease_ref=lease_ref,
            work_unit_ref=lease.work_unit_ref,
            event_kind=event_kind,
            lease_version=head.lease_version + 1,
            fencing_token=lease.fencing_token,
            effective_expires_at=None,
            result_refs=result_refs,
            failure=event_failure,
            work_control_policy_ref=policy_ref,
            audit=_control_audit(
                audit,
                (
                    lease_ref,
                    lease.work_unit_ref,
                    policy_ref,
                    *result_refs,
                ),
            ),
        )
        self._append_work_lease_event(connection, event)
        updated_head = head.model_copy(
            update={
                "state": _lease_state_for_event(event_kind),
                "lease_version": event.lease_version,
            }
        )
        self._update_work_lease_head(connection, updated_head, head.lease_version)
        self._append_work_control_outbox(
            connection,
            lease,
            event_type="work-lease-closed",
            event=event,
        )
        return event

    def _create_work_retry_decision(
        self,
        connection: sqlite3.Connection,
        lease: WorkLeaseV2,
        policy: WorkControlPolicyV2,
        event: WorkLeaseEventV2,
        audit: ContractAudit,
    ) -> WorkRetryDecisionV2:
        if lease.attempt < policy.max_attempts:
            decision_kind = WorkRetryDecisionKindV2.RETRY_SCHEDULED
            next_attempt = lease.attempt + 1
            eligible_at = self._clock() + timedelta(seconds=policy.retry_delay_seconds[lease.attempt - 1])
        else:
            decision_kind = WorkRetryDecisionKindV2.EXHAUSTED
            next_attempt = None
            eligible_at = None
        event_ref = work_lease_event_v2_ref(event)
        policy_ref = work_control_policy_v2_ref(policy)
        decision = WorkRetryDecisionV2.create(
            work_unit_ref=lease.work_unit_ref,
            prior_lease_event_ref=event_ref,
            prior_result_refs=event.result_refs,
            work_control_policy_ref=policy_ref,
            decision=decision_kind,
            completed_attempt=lease.attempt,
            next_attempt=next_attempt,
            eligible_at=eligible_at,
            audit=_control_audit(
                audit,
                (
                    lease.work_unit_ref,
                    event_ref,
                    policy_ref,
                    *event.result_refs,
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO work_retry_decisions (
                work_retry_decision_id, resolved_work_unit_id,
                prior_lease_event_id, decision, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                decision.work_retry_decision_id,
                lease.work_unit_ref.object_id,
                event.work_lease_event_id,
                decision.decision,
                self._record_json(decision),
            ),
        )
        return decision

    def _append_work_control_outbox(
        self,
        connection: sqlite3.Connection,
        lease: WorkLeaseV2,
        *,
        event_type: str,
        event: WorkLeaseEventV2,
    ) -> None:
        unit, _ = self._get_work_unit(connection, lease.work_unit_ref.object_id)
        if lease.stage_run_ref is not None:
            stage = self._get_record(
                connection,
                "stage_runs",
                "stage_run_id",
                lease.stage_run_ref.object_id,
                StageRunRecord,
            )
            aggregate_type: Literal["JOB", "ITEM", "STAGE_RUN"] = "STAGE_RUN"
            aggregate_id = stage.stage_run_id
            aggregate_version = stage.row_version
        elif unit.item_id is not None:
            item = self._get_record(
                connection,
                "items",
                "item_id",
                unit.item_id,
                ItemRecord,
            )
            aggregate_type = "ITEM"
            aggregate_id = item.item_id
            aggregate_version = item.row_version
        else:
            job = self._get_record(
                connection,
                "jobs",
                "job_id",
                unit.job_id,
                JobRecord,
            )
            aggregate_type = "JOB"
            aggregate_id = job.job_id
            aggregate_version = job.row_version
        self._append_outbox(
            connection,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            aggregate_version=aggregate_version,
            event_type=event_type,
            attributes=_attributes(
                event_kind=event.event_kind,
                lease_event_id=event.work_lease_event_id,
                work_lease_id=lease.work_lease_id,
                work_unit_id=lease.work_unit_ref.object_id,
            ),
        )

    def _rebuild_work_lease_head(
        self,
        connection: sqlite3.Connection,
        lease: WorkLeaseV2,
    ) -> WorkLeaseHeadRecord:
        state = WorkLeaseStateV2.ACTIVE
        version = 0
        expires_at = lease.expires_at
        rows = connection.execute(
            """
            SELECT work_lease_event_id, work_lease_id, lease_version,
                   event_kind, record_json
            FROM work_lease_events
            WHERE work_lease_id = ?
            ORDER BY lease_version
            """,
            (lease.work_lease_id,),
        ).fetchall()
        for row in rows:
            event = self._load_work_lease_event_row(row)
            if event.lease_version != version + 1:
                raise ImmutableResultError("work lease event versions are not contiguous")
            if (
                event.work_lease_ref != work_lease_v2_ref(lease)
                or event.work_unit_ref != lease.work_unit_ref
                or event.work_control_policy_ref != lease.work_control_policy_ref
                or event.fencing_token != lease.fencing_token
            ):
                raise ImmutableResultError("work lease event binding is inconsistent")
            if state is not WorkLeaseStateV2.ACTIVE:
                raise ImmutableResultError("work lease has an event after terminal state")
            version = event.lease_version
            if event.event_kind is WorkLeaseEventKindV2.HEARTBEAT:
                if event.effective_expires_at is None:
                    raise ImmutableResultError("heartbeat event has no effective expiry")
                expires_at = event.effective_expires_at
            else:
                state = _lease_state_for_event(event.event_kind)
        return WorkLeaseHeadRecord(
            work_lease_id=lease.work_lease_id,
            work_unit_id=lease.work_unit_ref.object_id,
            state=state,
            lease_version=version,
            fencing_token=lease.fencing_token,
            expires_at=expires_at,
        )

    def _complete_stage_run_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        stage_result_id: str,
        current: StageRunRecord,
        status: StageRunStatus,
        output_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        checkpoint_ref: ObjectRef | None,
        metrics_ref: ObjectRef | None,
        now: datetime,
    ) -> StageResultRecord:
        existing = connection.execute(
            "SELECT stage_result_id FROM stage_results WHERE stage_run_id = ?",
            (current.stage_run_id,),
        ).fetchone()
        if existing is not None:
            raise ImmutableResultError(f"stage run {current.stage_run_id} already has an immutable result")
        self._require_transition(
            STAGE_TRANSITIONS,
            current.status,
            status,
            "stage run",
        )
        if status not in TERMINAL_STAGE_STATUSES:
            raise IllegalTransitionError("stage completion requires a terminal status")
        result_payload = {
            "stage_result_id": stage_result_id,
            "stage_run_id": current.stage_run_id,
            "stage_run_version": current.row_version + 1,
            "status": status,
            "output_refs": output_refs,
            "failure": failure,
            "checkpoint_ref": checkpoint_ref,
            "metrics_ref": metrics_ref,
            "created_at": now,
        }
        result = StageResultRecord(
            stage_result_id=stage_result_id,
            stage_run_id=current.stage_run_id,
            stage_run_version=current.row_version + 1,
            status=status,
            output_refs=output_refs,
            failure=failure,
            checkpoint_ref=checkpoint_ref,
            metrics_ref=metrics_ref,
            result_sha256=_request_sha256(result_payload),
            created_at=now,
        )
        updated = StageRunRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "status": status,
                "row_version": current.row_version + 1,
                "ended_at": now,
            }
        )
        self._update_record(
            connection,
            "stage_runs",
            "stage_run_id",
            updated.stage_run_id,
            updated.status,
            updated.row_version,
            updated,
            current.row_version,
        )
        connection.execute(
            """
            INSERT INTO stage_results (
                stage_result_id, stage_run_id, result_sha256, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                result.stage_result_id,
                result.stage_run_id,
                result.result_sha256,
                self._record_json(result),
            ),
        )
        self._append_outbox(
            connection,
            aggregate_type="STAGE_RUN",
            aggregate_id=updated.stage_run_id,
            aggregate_version=updated.row_version,
            event_type="stage-run-completed",
            attributes=_attributes(status=status, result_id=stage_result_id),
        )
        return result

    def _validate_readiness_evidence(
        self,
        connection: sqlite3.Connection,
        snapshot: WorkReadinessSnapshotV2,
        unit: ResolvedWorkUnitV2,
    ) -> None:
        dependency_by_ref: dict[ObjectRef, ResolvedWorkUnitV2] = {}
        dependency_refs = unit.depends_on_work_unit_refs
        if snapshot.semantic_review_fanout_ref is not None:
            fanout = self._get_semantic_review_fanout_by_id(
                connection,
                snapshot.semantic_review_fanout_ref.object_id,
            )
            if semantic_review_fanout_v2_ref(fanout) != snapshot.semantic_review_fanout_ref:
                raise WorkFanoutPolicyError("semantic review readiness fanout is stale")
            if snapshot.work_unit_ref == fanout.parent_item_quality_work_unit_ref:
                dependency_refs = (
                    *dependency_refs,
                    *(resolved_work_unit_v2_ref(value.work_unit) for value in fanout.round_work),
                )
        if snapshot.artifact_group_fanout_ref is not None:
            artifact_fanout = self._get_artifact_group_fanout_by_id(
                connection,
                snapshot.artifact_group_fanout_ref.object_id,
            )
            if artifact_group_fanout_v2_ref(artifact_fanout) != snapshot.artifact_group_fanout_ref:
                raise WorkFanoutPolicyError("artifact readiness fanout is stale")
            if snapshot.work_unit_ref == artifact_fanout.parent_attachment_work_unit_ref:
                dependency_refs = tuple(
                    resolved_work_unit_v2_ref(value) for value in artifact_fanout.group_work_units
                )
        for dependency_ref in dependency_refs:
            dependency, _ = self._get_work_unit(
                connection,
                dependency_ref.object_id,
            )
            if resolved_work_unit_v2_ref(dependency) != dependency_ref:
                raise WorkFanoutPolicyError("work readiness dependency identity is stale")
            dependency_by_ref[dependency_ref] = dependency

        covered: set[ObjectRef] = set()
        for waiting_ref in snapshot.waiting_dependency_work_unit_refs:
            if waiting_ref not in dependency_by_ref:
                raise WorkFanoutPolicyError("waiting readiness ref is not a direct dependency")
            covered.add(waiting_ref)
        for observed_item_ref in snapshot.skipped_item_refs:
            item = self._get_record(
                connection,
                "items",
                "item_id",
                observed_item_ref.object_id,
                ItemRecord,
            )
            if item_record_ref(item) != observed_item_ref:
                raise WorkFanoutPolicyError("skipped Item readiness ref is stale")
            matches = tuple(
                dependency_ref
                for dependency_ref, dependency in dependency_by_ref.items()
                if dependency.item_id == item.item_id
            )
            if len(matches) != 1:
                raise WorkFanoutPolicyError("skipped Item does not match one direct dependency")
            covered.add(matches[0])

        result_refs = (
            *snapshot.succeeded_dependency_result_refs,
            *snapshot.retryable_dependency_result_refs,
            *snapshot.terminal_non_success_result_refs,
        )
        for observed_result_ref in result_refs:
            if observed_result_ref.object_type == "work-retry-decision":
                row = connection.execute(
                    """
                    SELECT work_retry_decision_id, resolved_work_unit_id,
                           prior_lease_event_id, decision, record_json
                    FROM work_retry_decisions
                    WHERE work_retry_decision_id = ?
                    """,
                    (observed_result_ref.object_id,),
                ).fetchone()
                if row is None:
                    raise RecordNotFoundError(
                        f"WorkRetryDecisionV2 not found: {observed_result_ref.object_id}"
                    )
                decision = self._load_work_retry_decision_row(
                    connection,
                    row,
                )
                if work_retry_decision_v2_ref(decision) != observed_result_ref:
                    raise WorkFanoutPolicyError("work retry decision readiness ref is stale")
                event = self._get_work_lease_event(
                    connection,
                    decision.prior_lease_event_ref.object_id,
                )
                if (
                    work_lease_event_v2_ref(event) != decision.prior_lease_event_ref
                    or decision.prior_result_refs != event.result_refs
                ):
                    raise WorkFanoutPolicyError("work retry decision source evidence is stale")
                if (
                    observed_result_ref in snapshot.retryable_dependency_result_refs
                    and decision.decision is not WorkRetryDecisionKindV2.RETRY_SCHEDULED
                ) or (
                    observed_result_ref in snapshot.terminal_non_success_result_refs
                    and decision.decision is not WorkRetryDecisionKindV2.EXHAUSTED
                ):
                    raise WorkFanoutPolicyError("work retry decision is in the wrong readiness partition")
                self._cover_control_readiness_work(
                    connection,
                    snapshot,
                    dependency_by_ref,
                    covered,
                    decision.work_unit_ref,
                )
                continue
            if observed_result_ref.object_type == "work-lease-event":
                event = self._get_work_lease_event(
                    connection,
                    observed_result_ref.object_id,
                )
                if work_lease_event_v2_ref(event) != observed_result_ref:
                    raise WorkFanoutPolicyError("work lease event readiness ref is stale")
                if (
                    observed_result_ref not in snapshot.terminal_non_success_result_refs
                    or event.event_kind
                    not in {
                        WorkLeaseEventKindV2.TERMINAL_FAILURE,
                        WorkLeaseEventKindV2.EXPIRED,
                        WorkLeaseEventKindV2.CANCELLED,
                    }
                    or event.failure is None
                    or event.failure.retryable
                ):
                    raise WorkFanoutPolicyError("work lease event is not terminal readiness evidence")
                self._cover_control_readiness_work(
                    connection,
                    snapshot,
                    dependency_by_ref,
                    covered,
                    event.work_unit_ref,
                )
                continue
            if observed_result_ref.object_type == "resource-admission-decision":
                resource_decision = self._get_record(
                    connection,
                    "resource_admission_decisions",
                    "resource_admission_decision_id",
                    observed_result_ref.object_id,
                    ResourceAdmissionDecisionV2,
                )
                if resource_admission_decision_v2_ref(resource_decision) != observed_result_ref:
                    raise WorkFanoutPolicyError("resource admission readiness ref is stale")
                if (
                    observed_result_ref not in snapshot.terminal_non_success_result_refs
                    or resource_decision.outcome
                    not in {
                        ResourceAdmissionOutcomeV2.BUDGET_EXHAUSTED,
                        ResourceAdmissionOutcomeV2.UNSATISFIABLE_DEMAND,
                    }
                ):
                    raise WorkFanoutPolicyError("resource admission is not terminal readiness evidence")
                resource_demand = self._get_record(
                    connection,
                    "work_resource_demands",
                    "work_resource_demand_id",
                    resource_decision.work_resource_demand_ref.object_id,
                    WorkResourceDemandV2,
                )
                if work_resource_demand_v2_ref(resource_demand) != (
                    resource_decision.work_resource_demand_ref
                ):
                    raise WorkFanoutPolicyError("resource admission demand evidence is stale")
                self._cover_control_readiness_work(
                    connection,
                    snapshot,
                    dependency_by_ref,
                    covered,
                    resource_demand.work_unit_ref,
                )
                continue
            if observed_result_ref.object_type == "model-admission-decision":
                model_decision = self._get_record(
                    connection,
                    "model_admission_decisions",
                    "model_admission_decision_id",
                    observed_result_ref.object_id,
                    ModelAdmissionDecisionV2,
                )
                if model_admission_decision_v2_ref(model_decision) != (observed_result_ref):
                    raise WorkFanoutPolicyError("model admission readiness ref is stale")
                if (
                    observed_result_ref not in snapshot.terminal_non_success_result_refs
                    or model_decision.outcome
                    not in {
                        ModelAdmissionOutcomeV2.BUDGET_EXHAUSTED,
                        ModelAdmissionOutcomeV2.UNSATISFIABLE_DEMAND,
                    }
                ):
                    raise WorkFanoutPolicyError("model admission is not terminal readiness evidence")
                model_demand = self._get_record(
                    connection,
                    "work_model_demands",
                    "work_model_demand_id",
                    model_decision.work_model_demand_ref.object_id,
                    WorkModelDemandV2,
                )
                if work_model_demand_v2_ref(model_demand) != (model_decision.work_model_demand_ref):
                    raise WorkFanoutPolicyError("model admission demand evidence is stale")
                self._cover_control_readiness_work(
                    connection,
                    snapshot,
                    dependency_by_ref,
                    covered,
                    model_demand.work_unit_ref,
                )
                continue
            if observed_result_ref.object_type != "stage-result":
                continue
            result = self._get_record(
                connection,
                "stage_results",
                "stage_result_id",
                observed_result_ref.object_id,
                StageResultRecord,
            )
            if stage_result_record_ref(result) != observed_result_ref:
                raise WorkFanoutPolicyError("StageResult readiness ref is stale")
            stage_run = self._get_record(
                connection,
                "stage_runs",
                "stage_run_id",
                result.stage_run_id,
                StageRunRecord,
            )
            matches = tuple(
                dependency_ref
                for dependency_ref, dependency in dependency_by_ref.items()
                if (
                    dependency_ref in stage_run.input_refs
                    and dependency.job_id == stage_run.job_id
                    and dependency.item_id == stage_run.item_id
                    and dependency.stage is stage_run.stage
                )
            )
            if len(matches) != 1:
                raise WorkFanoutPolicyError("StageResult does not match one direct dependency")
            covered.add(matches[0])
        if snapshot.artifact_group_fanout_ref is None and covered != set(dependency_by_ref):
            raise WorkFanoutPolicyError("readiness evidence does not exactly cover direct dependencies")

    def _cover_control_readiness_work(
        self,
        connection: sqlite3.Connection,
        snapshot: WorkReadinessSnapshotV2,
        dependency_by_ref: dict[ObjectRef, ResolvedWorkUnitV2],
        covered: set[ObjectRef],
        observed_work_unit_ref: ObjectRef,
    ) -> None:
        unit, owner_ref = self._get_work_unit(
            connection,
            observed_work_unit_ref.object_id,
        )
        if resolved_work_unit_v2_ref(unit) != observed_work_unit_ref:
            raise WorkFanoutPolicyError("control readiness work ref is stale")
        if observed_work_unit_ref in dependency_by_ref:
            covered.add(observed_work_unit_ref)
            return
        if snapshot.artifact_group_fanout_ref is None:
            raise WorkFanoutPolicyError("control readiness work is not a direct dependency")
        fanout = self._get_artifact_group_fanout_by_id(
            connection,
            snapshot.artifact_group_fanout_ref.object_id,
        )
        if (
            artifact_group_fanout_v2_ref(fanout) != snapshot.artifact_group_fanout_ref
            or owner_ref != fanout.artifact_group_fanout_id
            or unit not in fanout.group_work_units
            or snapshot.work_unit_ref != fanout.parent_attachment_work_unit_ref
        ):
            raise WorkFanoutPolicyError("control readiness work is not owned by the artifact fanout")

    @staticmethod
    def _work_item_idempotency_key(
        graph_id: str,
        item_id: str,
    ) -> str:
        digest = _request_sha256(
            {
                "resolved_job_work_graph_id": graph_id,
                "item_id": item_id,
            }
        )
        return f"work-graph-item://sha256/{digest}"

    @staticmethod
    def _require_version(observed: int, expected: int, aggregate_id: str) -> None:
        if observed != expected:
            raise ConcurrencyConflictError(
                f"stale aggregate {aggregate_id}: expected version {expected}, observed {observed}"
            )

    @staticmethod
    def _require_transition(
        transitions: dict[Any, frozenset[Any]],
        source: Any,
        target: Any,
        aggregate: str,
    ) -> None:
        if target not in transitions.get(source, frozenset()):
            raise IllegalTransitionError(f"illegal {aggregate} transition: {source} -> {target}")

    def _update_record(
        self,
        connection: sqlite3.Connection,
        table: str,
        id_column: str,
        record_id: str,
        status: str,
        row_version: int,
        record: BaseModel,
        expected_version: int,
    ) -> None:
        cursor = connection.execute(
            f"""
            UPDATE {table}
            SET status = ?, row_version = ?, record_json = ?
            WHERE {id_column} = ? AND row_version = ?
            """,
            (
                status,
                row_version,
                self._record_json(record),
                record_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ConcurrencyConflictError(f"concurrent update detected for {record_id}")

    def _validate_work_lease_admission(
        self,
        connection: sqlite3.Connection,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        readiness_snapshot: WorkReadinessSnapshotV2,
        policy: WorkControlPolicyV2,
        retry_decision: WorkRetryDecisionV2 | None,
    ) -> tuple[
        WorkLeaseV2 | None,
        int,
        datetime,
        ObjectRef | None,
        JobRecord,
    ]:
        stored_graph = self._get_job_work_graph_by_id(
            connection,
            graph.resolved_job_work_graph_id,
        )
        if stored_graph != graph:
            raise WorkControlPolicyError("work lease graph is not current")
        stored_unit, owner_ref = self._get_work_unit(
            connection,
            work_unit.resolved_work_unit_id,
        )
        if stored_unit != work_unit:
            raise WorkControlPolicyError("work lease unit is not current")
        self._validate_work_unit_graph_owner(
            connection,
            graph,
            work_unit,
            owner_ref,
        )
        stored_policy = self._get_work_control_policy_by_id(
            connection,
            policy.work_control_policy_id,
        )
        if stored_policy != policy or policy.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(
            graph
        ):
            raise WorkControlPolicyError("work lease policy is not current")

        latest_readiness = self._get_latest_work_readiness(
            connection,
            work_unit.resolved_work_unit_id,
        )
        if (
            latest_readiness != readiness_snapshot
            or readiness_snapshot.work_unit_ref != resolved_work_unit_v2_ref(work_unit)
            or readiness_snapshot.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph)
        ):
            raise WorkControlPolicyError("work lease requires the latest readiness snapshot")
        if readiness_snapshot.readiness is not WorkReadinessV2.READY:
            raise WorkControlPolicyError("work lease requires READY readiness")

        job = self._get_record(
            connection,
            "jobs",
            "job_id",
            graph.job_id,
            JobRecord,
        )
        if (
            job.status is JobStatus.CANCELLED
            or connection.execute(
                "SELECT 1 FROM work_cancellations WHERE job_id = ?",
                (job.job_id,),
            ).fetchone()
        ):
            raise WorkControlPolicyError("cancelled job cannot dispatch work")
        if job.status is not JobStatus.RUNNING:
            raise WorkControlPolicyError("work lease requires a RUNNING job")
        if connection.execute(
            """
            SELECT 1
            FROM work_lease_heads
            WHERE resolved_work_unit_id = ? AND state = 'ACTIVE'
            """,
            (work_unit.resolved_work_unit_id,),
        ).fetchone():
            raise WorkControlPolicyError("work unit already has an active lease")

        prior_lease = self._get_latest_work_lease_for_unit(
            connection,
            work_unit.resolved_work_unit_id,
        )
        if prior_lease is None:
            if retry_decision is not None:
                raise WorkControlPolicyError("initial work attempt cannot bind retry evidence")
            return prior_lease, 1, self._clock(), None, job

        if retry_decision is None:
            raise WorkControlPolicyError("later work attempt requires a retry decision")
        latest_retry = self._get_latest_work_retry_decision(
            connection,
            work_unit.resolved_work_unit_id,
        )
        if (
            latest_retry != retry_decision
            or retry_decision.decision is not WorkRetryDecisionKindV2.RETRY_SCHEDULED
            or retry_decision.next_attempt != prior_lease.attempt + 1
        ):
            raise WorkControlPolicyError("retry decision is stale or not dispatchable")
        if retry_decision.eligible_at is None:
            raise WorkControlPolicyError("retry decision has no eligible time")
        if self._clock() < retry_decision.eligible_at:
            raise WorkControlPolicyError("retry is not yet eligible")
        if connection.execute(
            """
            SELECT 1
            FROM work_dispatch_decisions
            WHERE retry_decision_id = ?
            """,
            (retry_decision.work_retry_decision_id,),
        ).fetchone():
            raise WorkControlPolicyError("retry decision was already consumed")
        return (
            prior_lease,
            retry_decision.next_attempt,
            retry_decision.eligible_at,
            work_retry_decision_v2_ref(retry_decision),
            job,
        )

    @staticmethod
    def _reject_control_graph_bypass(
        connection: sqlite3.Connection,
        graph_id: str,
        *,
        control_authorizations: frozenset[Literal["RESOURCE", "MODEL"]],
    ) -> None:
        required: set[Literal["RESOURCE", "MODEL"]] = set()
        resource_row = connection.execute(
            """
            SELECT 1
            FROM job_resource_policies
            WHERE resolved_job_work_graph_id = ?
            """,
            (graph_id,),
        ).fetchone()
        if resource_row is not None:
            required.add("RESOURCE")
        model_row = connection.execute(
            """
            SELECT 1
            FROM job_model_policies
            WHERE resolved_job_work_graph_id = ?
            """,
            (graph_id,),
        ).fetchone()
        if model_row is not None:
            required.add("MODEL")
        missing = required.difference(control_authorizations)
        if missing == {"RESOURCE"}:
            raise WorkControlPolicyError("resource-controlled graph requires resource-aware work control")
        if missing == {"MODEL"}:
            raise WorkControlPolicyError("model-controlled graph requires model-aware work control")
        if missing:
            raise WorkControlPolicyError(
                "controlled graph requires all bound control families: " + ",".join(sorted(missing))
            )

    @staticmethod
    def _reject_control_lease_bypass(
        connection: sqlite3.Connection,
        lease_id: str,
        *,
        control_authorizations: frozenset[Literal["RESOURCE", "MODEL"]],
    ) -> None:
        required: set[Literal["RESOURCE", "MODEL"]] = set()
        resource_row = connection.execute(
            """
            SELECT 1
            FROM work_resource_reservations
            WHERE work_lease_id = ?
            """,
            (lease_id,),
        ).fetchone()
        if resource_row is not None:
            required.add("RESOURCE")
        model_row = connection.execute(
            """
            SELECT 1
            FROM work_model_reservations
            WHERE work_lease_id = ?
            """,
            (lease_id,),
        ).fetchone()
        if model_row is not None:
            required.add("MODEL")
        missing = required.difference(control_authorizations)
        if missing == {"RESOURCE"}:
            raise WorkControlPolicyError("resource-controlled lease requires resource-aware work control")
        if missing == {"MODEL"}:
            raise WorkControlPolicyError("model-controlled lease requires model-aware work control")
        if missing:
            raise WorkControlPolicyError(
                "controlled lease requires all bound control families: " + ",".join(sorted(missing))
            )

    def _validate_retry(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        item_id: str | None,
        stage: StageNameV2,
        attempt: int,
        retry_of_stage_run_id: str | None,
    ) -> None:
        if attempt == 1:
            if retry_of_stage_run_id is not None:
                raise IllegalTransitionError("first stage attempt cannot bind retry_of")
            return
        if retry_of_stage_run_id is None:
            raise IllegalTransitionError("later stage attempt requires retry_of")
        prior = self._get_record(
            connection,
            "stage_runs",
            "stage_run_id",
            retry_of_stage_run_id,
            StageRunRecord,
        )
        if (
            prior.job_id != job_id
            or prior.item_id != item_id
            or prior.stage is not stage
            or prior.attempt + 1 != attempt
        ):
            raise IllegalTransitionError("retry stage identity or attempt does not match prior run")
        if prior.status not in RETRYABLE_STAGE_STATUSES:
            raise IllegalTransitionError(f"stage status {prior.status} is not retryable")


def _control_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    versions = tuple(binding for binding in audit.governing_versions if binding.component != "work-control")
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=(
            *versions,
            VersionBinding(
                component="work-control",
                version=R6_WORK_CONTROL_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(sorted(refs, key=_object_ref_key)),
    )


def _observability_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    versions = tuple(
        binding for binding in audit.governing_versions if binding.component != "batch-observability"
    )
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=(
            *versions,
            VersionBinding(
                component="batch-observability",
                version=R6_OBSERVABILITY_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(sorted(set(refs), key=_object_ref_key)),
    )


def _object_ref_from_facade(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _object_ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _work_stage_run_id(work_unit_id: str, attempt: int) -> str:
    digest = _request_sha256(
        {
            "work_unit_id": work_unit_id,
            "attempt": attempt,
            "policy_version": R6_WORK_CONTROL_POLICY_VERSION,
        }
    )
    return f"stage-run://r6-05/sha256/{digest}"


def _work_stage_result_id(
    stage_run_id: str,
    status: StageRunStatus,
) -> str:
    digest = _request_sha256(
        {
            "stage_run_id": stage_run_id,
            "status": status,
            "policy_version": R6_WORK_CONTROL_POLICY_VERSION,
        }
    )
    return f"stage-result://r6-05/sha256/{digest}"


def _stage_status_event_kind(
    status: StageRunStatus,
    failure: FailureRecord | None,
) -> WorkLeaseEventKindV2:
    if status is StageRunStatus.SUCCEEDED:
        return WorkLeaseEventKindV2.SUCCEEDED
    if status is StageRunStatus.CANCELLED:
        return WorkLeaseEventKindV2.CANCELLED
    if failure is not None and failure.retryable:
        return WorkLeaseEventKindV2.RETRYABLE_FAILURE
    return WorkLeaseEventKindV2.TERMINAL_FAILURE


def _artifact_terminal_status(
    event_kind: WorkLeaseEventKindV2,
) -> StageRunStatus:
    mapping = {
        WorkLeaseEventKindV2.SUCCEEDED: StageRunStatus.SUCCEEDED,
        WorkLeaseEventKindV2.RETRYABLE_FAILURE: StageRunStatus.RETRYABLE_FAILURE,
        WorkLeaseEventKindV2.TERMINAL_FAILURE: StageRunStatus.TERMINAL_FAILURE,
        WorkLeaseEventKindV2.EXPIRED: StageRunStatus.RETRYABLE_FAILURE,
        WorkLeaseEventKindV2.CANCELLED: StageRunStatus.CANCELLED,
    }
    try:
        return mapping[event_kind]
    except KeyError as exc:
        raise WorkControlPolicyError("artifact completion cannot use heartbeat") from exc


def _validate_control_stage_outcome(
    status: StageRunStatus,
    failure: FailureRecord | None,
) -> None:
    if status is StageRunStatus.SUCCEEDED:
        if failure is not None:
            raise WorkControlPolicyError("successful controlled stage cannot carry failure")
        return
    if failure is None:
        raise WorkControlPolicyError("non-success controlled stage requires failure")
    if status is StageRunStatus.RETRYABLE_FAILURE and not failure.retryable:
        raise WorkControlPolicyError("retryable stage status requires retryable failure")
    if (
        status
        in {
            StageRunStatus.TERMINAL_FAILURE,
            StageRunStatus.CANCELLED,
        }
        and failure.retryable
    ):
        raise WorkControlPolicyError("terminal stage status cannot carry retryable failure")


def _content_free_control_failure(
    failure: FailureRecord,
) -> FailureRecord:
    return FailureRecord(
        failure_class=failure.failure_class,
        code=failure.code,
        message=R6_WORK_CONTROL_FAILURE_MESSAGE,
        retryable=failure.retryable,
    )


def _lease_state_for_event(
    event_kind: WorkLeaseEventKindV2,
) -> WorkLeaseStateV2:
    mapping = {
        WorkLeaseEventKindV2.SUCCEEDED: WorkLeaseStateV2.SUCCEEDED,
        WorkLeaseEventKindV2.RETRYABLE_FAILURE: (WorkLeaseStateV2.RETRYABLE_FAILURE),
        WorkLeaseEventKindV2.TERMINAL_FAILURE: (WorkLeaseStateV2.TERMINAL_FAILURE),
        WorkLeaseEventKindV2.EXPIRED: WorkLeaseStateV2.EXPIRED,
        WorkLeaseEventKindV2.CANCELLED: WorkLeaseStateV2.CANCELLED,
    }
    try:
        return mapping[event_kind]
    except KeyError as exc:
        raise WorkControlPolicyError("heartbeat cannot become a terminal lease state") from exc
