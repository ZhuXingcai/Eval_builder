from __future__ import annotations

import hashlib
import sqlite3

from pydantic import model_validator

from env_mock_agent.facade import ExecutionTelemetryV2, FacadeObjectRef
from env_mock_agent.facade.resource_v2 import (
    AttachmentResourceGrantV2,
    AttachmentResourceTerminationRequestV2,
    AttachmentResourceTerminationResultV2,
    ResourceBoundAttachmentExecutionFacade,
    attachment_resource_termination_request_ref,
    attachment_resource_termination_result_ref,
    validate_attachment_resource_termination_result_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    FailureRecord,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.observability_v2 import ArtifactMetricSliceV2
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    WorkCancellationRecordV2,
    WorkControlPolicyV2,
    WorkLeaseEventKindV2,
    WorkLeaseEventV2,
    WorkLeaseV2,
    WorkReadinessSnapshotV2,
    WorkRetryDecisionV2,
    WorkUnitScopeV2,
    dataset_job_spec_v2_ref,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
    work_control_policy_v2_ref,
    work_lease_event_v2_ref,
    work_lease_v2_ref,
    work_readiness_snapshot_v2_ref,
    work_retry_decision_v2_ref,
)
from eval_factory.contracts.resource_v2 import (
    R6_RESOURCE_CONTROL_POLICY_VERSION,
    JobResourcePolicyV2,
    NonModelResourceVectorV2,
    ResourceAdmissionDecisionV2,
    ResourceAdmissionOutcomeV2,
    ResourceKindV2,
    ResourcePoolPolicyV2,
    ResourceUsageV2,
    WorkResourceDemandV2,
    WorkResourceEventKindV2,
    WorkResourceEventV2,
    WorkResourceReservationStateV2,
    WorkResourceReservationV2,
    WorkResourceTerminationOutcomeV2,
    WorkResourceTerminationReasonV2,
    WorkResourceTerminationRequestV2,
    WorkResourceTerminationResultV2,
    job_resource_policy_v2_ref,
    resource_admission_decision_v2_ref,
    resource_pool_policy_v2_ref,
    validate_job_resource_policy_v2_identity,
    validate_resource_pool_policy_v2_identity,
    work_resource_demand_v2_ref,
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
    _attributes,
    _request_sha256,
)
from eval_factory.orchestration.models import (
    JobRecord,
    JobResourceHeadRecord,
    ResourcePoolHeadRecord,
    WorkResourceReservationHeadRecord,
)


class ResourceControlPolicyError(JobStoreError):
    pass


class StaleResourceReservationError(JobStoreError):
    pass


class ResourceAdmissionResult(ContractModelV2):
    decision: ResourceAdmissionDecisionV2
    lease: WorkLeaseV2 | None = None
    reservation: WorkResourceReservationV2 | None = None

    @model_validator(mode="after")
    def validate_result(self) -> ResourceAdmissionResult:
        admitted = self.decision.outcome is ResourceAdmissionOutcomeV2.ADMITTED
        if admitted != (self.lease is not None and self.reservation is not None):
            raise ValueError("admission result must contain lease and reservation exactly when admitted")
        return self


class ResourceCompletionResult(ContractModelV2):
    lease_event: WorkLeaseEventV2
    resource_event: WorkResourceEventV2


class ResourceExpiryResult(ContractModelV2):
    lease_event: WorkLeaseEventV2
    resource_event: WorkResourceEventV2
    termination_request: WorkResourceTerminationRequestV2 | None = None


class ResourceCancellationResult(ContractModelV2):
    cancellation: WorkCancellationRecordV2
    resource_events: tuple[WorkResourceEventV2, ...]
    termination_requests: tuple[WorkResourceTerminationRequestV2, ...]


class ResourceTerminationResult(ContractModelV2):
    termination_result: WorkResourceTerminationResultV2
    resource_event: WorkResourceEventV2


class ResourceControlService:
    def __init__(self, job_store: JobStore) -> None:
        self.job_store = job_store

    def bind_pool_policy(
        self,
        *,
        resource_domain_ref: ObjectRef,
        capacity: NonModelResourceVectorV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ResourcePoolPolicyV2:
        policy = ResourcePoolPolicyV2.create(
            resource_domain_ref=resource_domain_ref,
            capacity=capacity,
            audit=_resource_audit(audit, (resource_domain_ref,)),
        )
        policy_ref = resource_pool_policy_v2_ref(policy)
        request_sha256 = policy_ref.object_sha256
        scope = f"bind-resource-pool:{resource_domain_ref.object_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "RESOURCE_POOL_POLICY":
                    raise IdempotencyConflictError("resource pool policy response type is corrupt")
                stored = self._get_pool_policy(connection, prior[1])
                if stored != policy:
                    raise ImmutableResultError("stored resource pool policy differs from request")
                return stored
            row = connection.execute(
                """
                SELECT resource_pool_policy_id
                FROM resource_pool_policies
                WHERE resource_domain_id = ?
                """,
                (resource_domain_ref.object_id,),
            ).fetchone()
            if row is not None:
                raise ImmutableResultError("resource domain already has an immutable pool policy")
            connection.execute(
                """
                INSERT INTO resource_pool_policies (
                    resource_pool_policy_id, resource_domain_id, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    policy.resource_pool_policy_id,
                    resource_domain_ref.object_id,
                    self.job_store._record_json(policy),
                ),
            )
            head = ResourcePoolHeadRecord(
                resource_pool_policy_id=policy.resource_pool_policy_id,
                active_capacity=NonModelResourceVectorV2.zero(),
                row_version=0,
            )
            connection.execute(
                """
                INSERT INTO resource_pool_heads (
                    resource_pool_policy_id, row_version, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    policy.resource_pool_policy_id,
                    head.row_version,
                    self.job_store._record_json(head),
                ),
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="RESOURCE_POOL_POLICY",
                response_id=policy.resource_pool_policy_id,
                created_at=self.job_store._clock(),
            )
        return policy

    def bind_job_policy(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        pool_policy: ResourcePoolPolicyV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> JobResourcePolicyV2:
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        stored_spec = self.job_store.get_job_spec(graph.job_id)
        spec_ref = dataset_job_spec_v2_ref(stored_spec)
        pool_ref = resource_pool_policy_v2_ref(pool_policy)
        job_concurrency = NonModelResourceVectorV2(
            processes=stored_spec.concurrency.processes,
            renderers=stored_spec.concurrency.renderers,
            network_requests=stored_spec.concurrency.network_requests,
            storage_bytes=stored_spec.budget.max_storage_bytes,
        )
        job_budget = NonModelResourceVectorV2(
            processes=stored_spec.budget.max_processes,
            renderers=stored_spec.budget.max_renderers,
            network_requests=stored_spec.budget.max_network_requests,
            storage_bytes=stored_spec.budget.max_storage_bytes,
        )
        policy = JobResourcePolicyV2.create(
            dataset_job_spec_ref=spec_ref,
            resolved_job_work_graph_ref=graph_ref,
            resource_pool_policy_ref=pool_ref,
            job_concurrency=job_concurrency,
            job_budget=job_budget,
            audit=_resource_audit(audit, (spec_ref, graph_ref, pool_ref)),
        )
        request_sha256 = job_resource_policy_v2_ref(policy).object_sha256
        scope = f"bind-job-resource-policy:{graph.job_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "JOB_RESOURCE_POLICY":
                    raise IdempotencyConflictError("job resource policy response type is corrupt")
                stored = self._get_job_policy(connection, prior[1])
                if stored != policy:
                    raise ImmutableResultError("stored Job resource policy differs from request")
                return stored
            stored_graph = self.job_store._get_job_work_graph_by_id(
                connection,
                graph.resolved_job_work_graph_id,
            )
            if stored_graph != graph:
                raise ResourceControlPolicyError("Job resource policy graph is not current")
            stored_pool = self._get_pool_policy(connection, pool_policy.resource_pool_policy_id)
            if stored_pool != pool_policy:
                raise ResourceControlPolicyError("resource pool policy is not current")
            if connection.execute(
                """
                SELECT 1
                FROM work_dispatch_decisions AS decisions
                JOIN resolved_work_units AS units
                  ON units.resolved_work_unit_id = decisions.resolved_work_unit_id
                WHERE units.job_id = ?
                LIMIT 1
                """,
                (graph.job_id,),
            ).fetchone():
                raise ResourceControlPolicyError("Job resource policy must bind before the first dispatch")
            connection.execute(
                """
                INSERT INTO job_resource_policies (
                    job_resource_policy_id, resolved_job_work_graph_id,
                    job_id, resource_pool_policy_id, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    policy.job_resource_policy_id,
                    graph.resolved_job_work_graph_id,
                    graph.job_id,
                    pool_policy.resource_pool_policy_id,
                    self.job_store._record_json(policy),
                ),
            )
            head = JobResourceHeadRecord(
                job_resource_policy_id=policy.job_resource_policy_id,
                active_capacity=NonModelResourceVectorV2.zero(),
                reserved_budget=NonModelResourceVectorV2.zero(),
                consumed_budget=NonModelResourceVectorV2.zero(),
                row_version=0,
            )
            connection.execute(
                """
                INSERT INTO job_resource_heads (
                    job_resource_policy_id, row_version, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    policy.job_resource_policy_id,
                    head.row_version,
                    self.job_store._record_json(head),
                ),
            )
            job = self.job_store._get_record(
                connection,
                "jobs",
                "job_id",
                graph.job_id,
                JobRecord,
            )
            self.job_store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job.job_id,
                aggregate_version=job.row_version,
                event_type="job-resource-policy-bound",
                attributes=_attributes(
                    policy_id=policy.job_resource_policy_id,
                ),
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="JOB_RESOURCE_POLICY",
                response_id=policy.job_resource_policy_id,
                created_at=self.job_store._clock(),
            )
        return policy

    def create_demand(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        job_policy: JobResourcePolicyV2,
        execution_profile_ref: ObjectRef,
        attempt: int,
        capacity_units: NonModelResourceVectorV2,
        budget_allowance: NonModelResourceVectorV2,
        termination_required: bool,
        audit: ContractAudit,
    ) -> WorkResourceDemandV2:
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        unit_ref = resolved_work_unit_v2_ref(work_unit)
        policy_ref = job_resource_policy_v2_ref(job_policy)
        graph_owns_unit = work_unit in graph.work_units
        fanout_unit = work_unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP and work_unit.job_id == graph.job_id
        semantic_review_unit = (
            work_unit.semantic_review_round is not None and work_unit.job_id == graph.job_id
        )
        if (
            not (graph_owns_unit or fanout_unit or semantic_review_unit)
            or job_policy.resolved_job_work_graph_ref != graph_ref
        ):
            raise ResourceControlPolicyError("resource demand graph or work unit is not current")
        return WorkResourceDemandV2.create(
            resolved_job_work_graph_ref=graph_ref,
            work_unit_ref=unit_ref,
            job_resource_policy_ref=policy_ref,
            execution_profile_ref=execution_profile_ref,
            attempt=attempt,
            capacity_units=capacity_units,
            budget_allowance=budget_allowance,
            termination_required=termination_required,
            audit=_resource_audit(
                audit,
                (graph_ref, unit_ref, policy_ref, execution_profile_ref),
            ),
        )

    def acquire(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        readiness_snapshot: WorkReadinessSnapshotV2,
        control_policy: WorkControlPolicyV2,
        job_policy: JobResourcePolicyV2,
        demand: WorkResourceDemandV2,
        holder_ref: ObjectRef,
        retry_decision: WorkRetryDecisionV2 | None,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ResourceAdmissionResult:
        request_sha256 = _request_sha256(
            {
                "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(graph),
                "work_unit_ref": resolved_work_unit_v2_ref(work_unit),
                "work_readiness_snapshot_ref": work_readiness_snapshot_v2_ref(readiness_snapshot),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "job_resource_policy_ref": job_resource_policy_v2_ref(job_policy),
                "work_resource_demand_ref": work_resource_demand_v2_ref(demand),
                "holder_ref": holder_ref,
                "retry_decision_ref": (
                    work_retry_decision_v2_ref(retry_decision) if retry_decision is not None else None
                ),
            }
        )
        scope = f"acquire-work-with-resources:{work_unit.resolved_work_unit_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "RESOURCE_ADMISSION_DECISION":
                    raise IdempotencyConflictError("resource admission response type is corrupt")
                return self._load_admission_result(connection, prior[1])
            self._validate_acquisition_sources(
                connection,
                graph=graph,
                work_unit=work_unit,
                job_policy=job_policy,
                demand=demand,
            )
            _prior_lease, intended_attempt, _eligible_at, _retry_ref, _job = (
                self.job_store._validate_work_lease_admission(
                    connection,
                    graph=graph,
                    work_unit=work_unit,
                    readiness_snapshot=readiness_snapshot,
                    policy=control_policy,
                    retry_decision=retry_decision,
                )
            )
            if demand.attempt != intended_attempt:
                raise ResourceControlPolicyError(
                    "resource demand attempt does not match the dispatchable attempt"
                )
            self._insert_or_validate_demand(connection, demand)
            if connection.execute(
                """
                SELECT 1
                FROM resource_admission_decisions
                WHERE work_resource_demand_id = ?
                  AND outcome IN ('BUDGET_EXHAUSTED', 'UNSATISFIABLE_DEMAND')
                LIMIT 1
                """,
                (demand.work_resource_demand_id,),
            ).fetchone():
                raise ResourceControlPolicyError("resource demand already has a terminal admission decision")
            pool_policy = self._get_pool_policy(
                connection,
                job_policy.resource_pool_policy_ref.object_id,
            )
            pool_head = self._get_pool_head(connection, pool_policy.resource_pool_policy_id)
            job_head = self._get_job_head(connection, job_policy.job_resource_policy_id)
            outcome, constrained = _classify(
                pool_policy=pool_policy,
                job_policy=job_policy,
                pool_head=pool_head,
                job_head=job_head,
                demand=demand,
            )
            policy_ref = job_resource_policy_v2_ref(job_policy)
            demand_ref = work_resource_demand_v2_ref(demand)
            decision = ResourceAdmissionDecisionV2.create(
                job_resource_policy_ref=policy_ref,
                work_resource_demand_ref=demand_ref,
                outcome=outcome,
                constrained_resources=constrained,
                pool_head_version_before=pool_head.row_version,
                job_head_version_before=job_head.row_version,
                decided_at=self.job_store._clock(),
                audit=_resource_audit(audit, (policy_ref, demand_ref)),
            )
            connection.execute(
                """
                INSERT INTO resource_admission_decisions (
                    resource_admission_decision_id, work_resource_demand_id,
                    outcome, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    decision.resource_admission_decision_id,
                    demand.work_resource_demand_id,
                    decision.outcome,
                    self.job_store._record_json(decision),
                ),
            )
            lease: WorkLeaseV2 | None = None
            reservation: WorkResourceReservationV2 | None = None
            if outcome is ResourceAdmissionOutcomeV2.ADMITTED:
                lease = self.job_store._acquire_work_lease(
                    graph=graph,
                    work_unit=work_unit,
                    readiness_snapshot=readiness_snapshot,
                    policy=control_policy,
                    holder_ref=holder_ref,
                    retry_decision=retry_decision,
                    audit=audit,
                    idempotency_key=_internal_key(idempotency_key, "lease"),
                    connection=connection,
                    control_authorizations=frozenset({"RESOURCE"}),
                )
                dispatch_ref = lease.work_dispatch_decision_ref
                lease_ref = work_lease_v2_ref(lease)
                decision_ref = resource_admission_decision_v2_ref(decision)
                handle_ref = _execution_handle_ref(lease)
                reservation = WorkResourceReservationV2.create(
                    resource_admission_decision_ref=decision_ref,
                    work_resource_demand_ref=demand_ref,
                    work_lease_ref=lease_ref,
                    work_dispatch_decision_ref=dispatch_ref,
                    holder_ref=holder_ref,
                    fencing_token=lease.fencing_token,
                    execution_handle_ref=handle_ref,
                    capacity_units=demand.capacity_units,
                    budget_allowance=demand.budget_allowance,
                    acquired_at=lease.acquired_at,
                    audit=_resource_audit(
                        audit,
                        (
                            decision_ref,
                            demand_ref,
                            lease_ref,
                            dispatch_ref,
                            holder_ref,
                            handle_ref,
                        ),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO work_resource_reservations (
                        work_resource_reservation_id,
                        resource_admission_decision_id,
                        work_resource_demand_id, work_lease_id, job_id,
                        resolved_work_unit_id, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reservation.work_resource_reservation_id,
                        decision.resource_admission_decision_id,
                        demand.work_resource_demand_id,
                        lease.work_lease_id,
                        graph.job_id,
                        work_unit.resolved_work_unit_id,
                        self.job_store._record_json(reservation),
                    ),
                )
                reservation_head = WorkResourceReservationHeadRecord(
                    work_resource_reservation_id=reservation.work_resource_reservation_id,
                    state=WorkResourceReservationStateV2.ACTIVE,
                    reservation_version=0,
                )
                connection.execute(
                    """
                    INSERT INTO work_resource_reservation_heads (
                        work_resource_reservation_id, state,
                        reservation_version, record_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        reservation.work_resource_reservation_id,
                        reservation_head.state,
                        reservation_head.reservation_version,
                        self.job_store._record_json(reservation_head),
                    ),
                )
                self._update_pool_head(
                    connection,
                    pool_head.model_copy(
                        update={
                            "active_capacity": pool_head.active_capacity.add(demand.capacity_units),
                            "row_version": pool_head.row_version + 1,
                        }
                    ),
                    expected_version=pool_head.row_version,
                )
                self._update_job_head(
                    connection,
                    job_head.model_copy(
                        update={
                            "active_capacity": job_head.active_capacity.add(demand.capacity_units),
                            "reserved_budget": job_head.reserved_budget.add(demand.budget_allowance),
                            "row_version": job_head.row_version + 1,
                        }
                    ),
                    expected_version=job_head.row_version,
                )
            job = self.job_store._get_record(
                connection,
                "jobs",
                "job_id",
                graph.job_id,
                JobRecord,
            )
            self.job_store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job.job_id,
                aggregate_version=job.row_version,
                event_type="resource-admission-decided",
                attributes=_attributes(
                    attempt=demand.attempt,
                    resource_admission_decision_id=decision.resource_admission_decision_id,
                    work_unit_id=demand.work_unit_ref.object_id,
                ),
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="RESOURCE_ADMISSION_DECISION",
                response_id=decision.resource_admission_decision_id,
                created_at=self.job_store._clock(),
            )
            return ResourceAdmissionResult(
                decision=decision,
                lease=lease,
                reservation=reservation,
            )

    def complete_stage(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        reservation: WorkResourceReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        status: StageRunStatus,
        output_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        checkpoint_ref: ObjectRef | None,
        metrics_ref: ObjectRef | None,
        usage: ResourceUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> ResourceCompletionResult:
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_resource_reservation_ref": (work_resource_reservation_v2_ref(reservation)),
                "holder_ref": holder_ref,
                "expected_lease_version": expected_lease_version,
                "status": status,
                "output_refs": output_refs,
                "failure": failure,
                "checkpoint_ref": checkpoint_ref,
                "metrics_ref": metrics_ref,
                "usage": usage,
                "telemetry": telemetry,
            }
        )
        scope = f"complete-resource-stage:{reservation.work_resource_reservation_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_RESOURCE_EVENT":
                    raise IdempotencyConflictError("resource completion response type is corrupt")
                event = self._get_resource_event(connection, prior[1])
                lease_event = self.job_store._get_work_lease_event(
                    connection,
                    event.work_lease_event_ref.object_id,
                )
                return ResourceCompletionResult(
                    lease_event=lease_event,
                    resource_event=event,
                )
            stored_reservation = self._get_reservation(
                connection,
                reservation.work_resource_reservation_id,
            )
            if stored_reservation != reservation or reservation.work_lease_ref != work_lease_v2_ref(lease):
                raise StaleResourceReservationError("resource reservation is stale")
            head = self._get_reservation_head(
                connection,
                reservation.work_resource_reservation_id,
            )
            if head.state is not WorkResourceReservationStateV2.ACTIVE:
                raise StaleResourceReservationError("resource reservation is not active")
            exceeded = usage.observed.exceeds(reservation.budget_allowance)
            if exceeded:
                raise ResourceControlPolicyError(
                    f"observed resource usage exceeds grant: {','.join(item.value for item in exceeded)}"
                )
            lease_event = self.job_store._complete_stage_work_lease(
                lease=lease,
                policy=control_policy,
                holder_ref=holder_ref,
                expected_lease_version=expected_lease_version,
                status=status,
                output_refs=output_refs,
                failure=failure,
                checkpoint_ref=checkpoint_ref,
                metrics_ref=metrics_ref,
                audit=audit,
                idempotency_key=_internal_key(idempotency_key, "lease-event"),
                connection=connection,
                control_authorizations=frozenset({"RESOURCE"}),
                metrics_resource_usage=usage,
                metrics_telemetry=telemetry,
            )
            resource_event = self._release_completed(
                connection,
                reservation=reservation,
                lease_event=lease_event,
                usage=usage,
                audit=audit,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_RESOURCE_EVENT",
                response_id=resource_event.work_resource_event_id,
                created_at=self.job_store._clock(),
            )
            return ResourceCompletionResult(
                lease_event=lease_event,
                resource_event=resource_event,
            )

    def complete_artifact_group(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        reservation: WorkResourceReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        event_kind: WorkLeaseEventKindV2,
        result_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        usage: ResourceUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
        artifact_slices: tuple[ArtifactMetricSliceV2, ...] = (),
    ) -> ResourceCompletionResult:
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_resource_reservation_ref": (work_resource_reservation_v2_ref(reservation)),
                "holder_ref": holder_ref,
                "expected_lease_version": expected_lease_version,
                "event_kind": event_kind,
                "result_refs": result_refs,
                "failure": failure,
                "usage": usage,
                "telemetry": telemetry,
                "artifact_slices": artifact_slices,
            }
        )
        scope = f"complete-resource-artifact:{reservation.work_resource_reservation_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_RESOURCE_EVENT":
                    raise IdempotencyConflictError("resource artifact completion response type is corrupt")
                resource_event = self._get_resource_event(connection, prior[1])
                lease_event = self.job_store._get_work_lease_event(
                    connection,
                    resource_event.work_lease_event_ref.object_id,
                )
                return ResourceCompletionResult(
                    lease_event=lease_event,
                    resource_event=resource_event,
                )
            self._require_active_reservation(connection, lease, reservation)
            exceeded = usage.observed.exceeds(reservation.budget_allowance)
            if exceeded:
                raise ResourceControlPolicyError(
                    f"observed resource usage exceeds grant: {','.join(item.value for item in exceeded)}"
                )
            lease_event = self.job_store._complete_artifact_group_work_lease(
                lease=lease,
                policy=control_policy,
                holder_ref=holder_ref,
                expected_lease_version=expected_lease_version,
                event_kind=event_kind,
                result_refs=result_refs,
                failure=failure,
                audit=audit,
                idempotency_key=_internal_key(idempotency_key, "lease-event"),
                connection=connection,
                control_authorizations=frozenset({"RESOURCE"}),
                metrics_resource_usage=usage,
                metrics_telemetry=telemetry,
                metrics_artifact_slices=artifact_slices,
            )
            resource_event = self._release_completed(
                connection,
                reservation=reservation,
                lease_event=lease_event,
                usage=usage,
                audit=audit,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_RESOURCE_EVENT",
                response_id=resource_event.work_resource_event_id,
                created_at=self.job_store._clock(),
            )
            return ResourceCompletionResult(
                lease_event=lease_event,
                resource_event=resource_event,
            )

    def expire(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        reservation: WorkResourceReservationV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ResourceExpiryResult:
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_resource_reservation_ref": (work_resource_reservation_v2_ref(reservation)),
            }
        )
        scope = f"expire-resource-lease:{reservation.work_resource_reservation_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_RESOURCE_EVENT":
                    raise IdempotencyConflictError("resource expiry response type is corrupt")
                resource_event = self._get_resource_event(connection, prior[1])
                lease_event = self.job_store._get_work_lease_event(
                    connection,
                    resource_event.work_lease_event_ref.object_id,
                )
                request = self._get_termination_request_for_reservation(
                    connection,
                    reservation.work_resource_reservation_id,
                )
                return ResourceExpiryResult(
                    lease_event=lease_event,
                    resource_event=resource_event,
                    termination_request=request,
                )
            self._require_active_reservation(connection, lease, reservation)
            lease_event = self.job_store._expire_work_lease(
                lease=lease,
                policy=control_policy,
                audit=audit,
                idempotency_key=_internal_key(idempotency_key, "lease-event"),
                connection=connection,
                control_authorizations=frozenset({"RESOURCE"}),
            )
            resource_event, request = self._close_aborted_reservation(
                connection,
                reservation=reservation,
                lease_event=lease_event,
                reason=WorkResourceTerminationReasonV2.EXPIRED,
                audit=audit,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_RESOURCE_EVENT",
                response_id=resource_event.work_resource_event_id,
                created_at=self.job_store._clock(),
            )
            return ResourceExpiryResult(
                lease_event=lease_event,
                resource_event=resource_event,
                termination_request=request,
            )

    def cancel_job(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        control_policy: WorkControlPolicyV2,
        job_policy: JobResourcePolicyV2,
        reason_code: str,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ResourceCancellationResult:
        request_sha256 = _request_sha256(
            {
                "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(graph),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "job_resource_policy_ref": job_resource_policy_v2_ref(job_policy),
                "reason_code": reason_code,
            }
        )
        scope = f"cancel-job-resources:{graph.job_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_CANCELLATION":
                    raise IdempotencyConflictError("resource cancellation response type is corrupt")
                cancellation = self.job_store._get_work_cancellation(connection, prior[1])
                prior_events, prior_requests = self._list_job_resource_closures(
                    connection,
                    graph.job_id,
                    cancellation,
                )
                return ResourceCancellationResult(
                    cancellation=cancellation,
                    resource_events=prior_events,
                    termination_requests=prior_requests,
                )
            stored_policy = self._get_job_policy(connection, job_policy.job_resource_policy_id)
            if (
                stored_policy != job_policy
                or job_policy.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph)
            ):
                raise ResourceControlPolicyError("Job resource policy is not current")
            active_rows = connection.execute(
                """
                SELECT reservations.work_resource_reservation_id,
                       reservations.work_lease_id
                FROM work_resource_reservations AS reservations
                JOIN work_resource_reservation_heads AS heads
                  ON heads.work_resource_reservation_id =
                     reservations.work_resource_reservation_id
                WHERE reservations.job_id = ? AND heads.state = 'ACTIVE'
                ORDER BY reservations.work_resource_reservation_id
                """,
                (graph.job_id,),
            ).fetchall()
            active = tuple(
                (
                    self._get_reservation(
                        connection,
                        str(row["work_resource_reservation_id"]),
                    ),
                    str(row["work_lease_id"]),
                )
                for row in active_rows
            )
            cancellation = self.job_store._cancel_job_work(
                graph=graph,
                policy=control_policy,
                reason_code=reason_code,
                audit=audit,
                idempotency_key=_internal_key(idempotency_key, "cancellation"),
                connection=connection,
                control_authorizations=frozenset({"RESOURCE"}),
            )
            events: list[WorkResourceEventV2] = []
            requests: list[WorkResourceTerminationRequestV2] = []
            for reservation, lease_id in active:
                lease_event = self.job_store._get_latest_work_lease_event(
                    connection,
                    lease_id,
                )
                event, request = self._close_aborted_reservation(
                    connection,
                    reservation=reservation,
                    lease_event=lease_event,
                    reason=WorkResourceTerminationReasonV2.CANCELLED,
                    audit=audit,
                )
                events.append(event)
                if request is not None:
                    requests.append(request)
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_CANCELLATION",
                response_id=cancellation.work_cancellation_record_id,
                created_at=self.job_store._clock(),
            )
            return ResourceCancellationResult(
                cancellation=cancellation,
                resource_events=tuple(events),
                termination_requests=tuple(requests),
            )

    async def terminate(
        self,
        *,
        request: WorkResourceTerminationRequestV2,
        facade: ResourceBoundAttachmentExecutionFacade,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ResourceTerminationResult:
        facade_request = AttachmentResourceTerminationRequestV2.create(
            reservation_ref=_facade_ref(request.work_resource_reservation_ref),
            execution_handle_ref=_facade_ref(request.execution_handle_ref),
            fencing_token=request.fencing_token,
            reason=request.reason.value,
            requested_at=request.requested_at,
        )
        facade_result = await facade.terminate_resources(facade_request)
        validate_attachment_resource_termination_result_identity(facade_result)
        if facade_result.termination_request_ref != (
            attachment_resource_termination_request_ref(facade_request)
        ):
            raise ResourceControlPolicyError("facade termination result does not bind its exact request")
        return self.record_termination(
            request=request,
            facade_result=facade_result,
            audit=audit,
            idempotency_key=idempotency_key,
        )

    def record_termination(
        self,
        *,
        request: WorkResourceTerminationRequestV2,
        facade_result: AttachmentResourceTerminationResultV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ResourceTerminationResult:
        validate_attachment_resource_termination_result_identity(facade_result)
        facade_request = AttachmentResourceTerminationRequestV2.create(
            reservation_ref=_facade_ref(request.work_resource_reservation_ref),
            execution_handle_ref=_facade_ref(request.execution_handle_ref),
            fencing_token=request.fencing_token,
            reason=request.reason.value,
            requested_at=request.requested_at,
        )
        if facade_result.termination_request_ref != (
            attachment_resource_termination_request_ref(facade_request)
        ):
            raise ResourceControlPolicyError("facade termination result does not bind its exact request")
        facade_termination_result_ref = _object_ref(attachment_resource_termination_result_ref(facade_result))
        outcome = WorkResourceTerminationOutcomeV2(facade_result.outcome.value)
        request_sha256 = _request_sha256(
            {
                "work_resource_termination_request_ref": (work_resource_termination_request_v2_ref(request)),
                "facade_termination_result_ref": (facade_termination_result_ref),
                "outcome": outcome,
            }
        )
        scope = f"record-resource-termination:{request.work_resource_termination_request_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_RESOURCE_TERMINATION_RESULT":
                    raise IdempotencyConflictError("termination response type is corrupt")
                result = self.job_store._get_record(
                    connection,
                    "work_resource_termination_results",
                    "work_resource_termination_result_id",
                    prior[1],
                    WorkResourceTerminationResultV2,
                )
                if result.termination_request_ref != (
                    work_resource_termination_request_v2_ref(request)
                ) or result.facade_termination_result_ref != (facade_termination_result_ref):
                    raise ImmutableResultError("stored termination result request binding is stale")
                event = self._get_latest_resource_event(
                    connection,
                    request.work_resource_reservation_ref.object_id,
                )
                if event.resource_termination_result_ref != (work_resource_termination_result_v2_ref(result)):
                    raise ImmutableResultError("stored resource event termination result is stale")
                return ResourceTerminationResult(
                    termination_result=result,
                    resource_event=event,
                )
            stored_request = self.job_store._get_record(
                connection,
                "work_resource_termination_requests",
                "work_resource_termination_request_id",
                request.work_resource_termination_request_id,
                WorkResourceTerminationRequestV2,
            )
            if stored_request != request:
                raise StaleResourceReservationError("termination request is stale")
            reservation = self._get_reservation(
                connection,
                request.work_resource_reservation_ref.object_id,
            )
            if (
                reservation.execution_handle_ref != request.execution_handle_ref
                or reservation.fencing_token != request.fencing_token
            ):
                raise StaleResourceReservationError("termination request handle or fence is stale")
            head = self._get_reservation_head(
                connection,
                reservation.work_resource_reservation_id,
            )
            if head.state is not WorkResourceReservationStateV2.RELEASE_PENDING_TERMINATION:
                raise StaleResourceReservationError("resource reservation is not awaiting termination")
            request_ref = work_resource_termination_request_v2_ref(request)
            if (
                facade_termination_result_ref.object_type != "attachment-resource-termination-result"
                or facade_termination_result_ref.object_version != "v2"
            ):
                raise ResourceControlPolicyError("facade termination result reference is invalid")
            result = WorkResourceTerminationResultV2.create(
                termination_request_ref=request_ref,
                facade_termination_result_ref=facade_termination_result_ref,
                outcome=outcome,
                completed_at=self.job_store._clock(),
                audit=_resource_audit(
                    audit,
                    (
                        request_ref,
                        facade_termination_result_ref,
                    ),
                ),
            )
            connection.execute(
                """
                INSERT INTO work_resource_termination_results (
                    work_resource_termination_result_id,
                    work_resource_termination_request_id, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    result.work_resource_termination_result_id,
                    request.work_resource_termination_request_id,
                    self.job_store._record_json(result),
                ),
            )
            succeeded = outcome in {
                WorkResourceTerminationOutcomeV2.TERMINATED,
                WorkResourceTerminationOutcomeV2.ALREADY_STOPPED,
            }
            released = (
                NonModelResourceVectorV2(
                    processes=reservation.capacity_units.processes,
                    renderers=reservation.capacity_units.renderers,
                    network_requests=reservation.capacity_units.network_requests,
                    storage_bytes=0,
                )
                if succeeded
                else NonModelResourceVectorV2.zero()
            )
            lease_event = self.job_store._get_work_lease_event(
                connection,
                request.work_lease_event_ref.object_id,
            )
            policy = self._get_job_policy_for_reservation(connection, reservation)
            pool_head = self._get_pool_head(
                connection,
                policy.resource_pool_policy_ref.object_id,
            )
            job_head = self._get_job_head(
                connection,
                policy.job_resource_policy_id,
            )
            reservation_ref = work_resource_reservation_v2_ref(reservation)
            lease_event_ref = work_lease_event_v2_ref(lease_event)
            termination_result_ref = work_resource_termination_result_v2_ref(result)
            resource_event = WorkResourceEventV2.create(
                work_resource_reservation_ref=reservation_ref,
                work_lease_event_ref=lease_event_ref,
                event_kind=(
                    WorkResourceEventKindV2.TERMINATED_RELEASED
                    if succeeded
                    else WorkResourceEventKindV2.QUARANTINED
                ),
                reservation_version=head.reservation_version + 1,
                charged_usage=ResourceUsageV2(observed=NonModelResourceVectorV2.zero()),
                released_capacity=released,
                resource_termination_request_ref=None,
                audit=_resource_audit(
                    audit,
                    (
                        reservation_ref,
                        lease_event_ref,
                        termination_result_ref,
                    ),
                ),
                resource_termination_result_ref=termination_result_ref,
            )
            self._insert_resource_event(connection, reservation, resource_event)
            self._update_pool_head(
                connection,
                pool_head.model_copy(
                    update={
                        "active_capacity": pool_head.active_capacity.subtract(released),
                        "row_version": pool_head.row_version + 1,
                    }
                ),
                expected_version=pool_head.row_version,
            )
            self._update_job_head(
                connection,
                job_head.model_copy(
                    update={
                        "active_capacity": job_head.active_capacity.subtract(released),
                        "row_version": job_head.row_version + 1,
                    }
                ),
                expected_version=job_head.row_version,
            )
            terminal_head = WorkResourceReservationHeadRecord(
                work_resource_reservation_id=reservation.work_resource_reservation_id,
                state=(
                    WorkResourceReservationStateV2.RELEASED
                    if succeeded
                    else WorkResourceReservationStateV2.QUARANTINED
                ),
                reservation_version=head.reservation_version + 1,
            )
            self._update_reservation_head(
                connection,
                terminal_head,
                expected_version=head.reservation_version,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_RESOURCE_TERMINATION_RESULT",
                response_id=result.work_resource_termination_result_id,
                created_at=self.job_store._clock(),
            )
            return ResourceTerminationResult(
                termination_result=result,
                resource_event=resource_event,
            )

    def to_facade_grant(
        self,
        reservation: WorkResourceReservationV2,
        *,
        operation_ref: FacadeObjectRef,
    ) -> AttachmentResourceGrantV2:
        return AttachmentResourceGrantV2.create(
            reservation_ref=_facade_ref(work_resource_reservation_v2_ref(reservation)),
            execution_handle_ref=_facade_ref(reservation.execution_handle_ref),
            operation_ref=operation_ref,
            fencing_token=reservation.fencing_token,
            process_limit=reservation.budget_allowance.processes,
            renderer_limit=reservation.budget_allowance.renderers,
            network_request_limit=reservation.budget_allowance.network_requests,
            storage_byte_limit=reservation.budget_allowance.storage_bytes,
        )

    def get_pool_head(self, pool_policy_id: str) -> ResourcePoolHeadRecord:
        with self.job_store._read_snapshot() as connection:
            return self._get_pool_head(connection, pool_policy_id)

    def get_job_policy(self, job_id: str) -> JobResourcePolicyV2:
        with self.job_store._read_snapshot() as connection:
            self.job_store._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            rows = connection.execute(
                """
                SELECT job_resource_policy_id
                FROM job_resource_policies
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchall()
            if not rows:
                raise RecordNotFoundError(f"JobResourcePolicyV2 not found: {job_id}")
            if len(rows) != 1:
                raise ImmutableResultError("Job has multiple resource policies")
            return self._get_job_policy(
                connection,
                str(rows[0]["job_resource_policy_id"]),
            )

    def get_job_head(self, job_policy_id: str) -> JobResourceHeadRecord:
        with self.job_store._read_snapshot() as connection:
            return self._get_job_head(connection, job_policy_id)

    def get_reservation_head(
        self,
        reservation_id: str,
    ) -> WorkResourceReservationHeadRecord:
        with self.job_store._read_snapshot() as connection:
            return self._get_reservation_head(connection, reservation_id)

    def get_reservation_for_lease(
        self,
        work_lease_id: str,
    ) -> WorkResourceReservationV2:
        with self.job_store._read_snapshot() as connection:
            rows = connection.execute(
                """
                SELECT work_resource_reservation_id
                FROM work_resource_reservations
                WHERE work_lease_id = ?
                """,
                (work_lease_id,),
            ).fetchall()
            if len(rows) != 1:
                raise ImmutableResultError("work lease must have exactly one resource reservation")
            reservation = self._get_reservation(
                connection,
                str(rows[0]["work_resource_reservation_id"]),
            )
            self._get_reservation_head(
                connection,
                reservation.work_resource_reservation_id,
            )
            return reservation

    def _validate_acquisition_sources(
        self,
        connection: sqlite3.Connection,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        job_policy: JobResourcePolicyV2,
        demand: WorkResourceDemandV2,
    ) -> None:
        stored_policy = self._get_job_policy(connection, job_policy.job_resource_policy_id)
        if stored_policy != job_policy:
            raise ResourceControlPolicyError("Job resource policy is not current")
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        unit_ref = resolved_work_unit_v2_ref(work_unit)
        if (
            job_policy.resolved_job_work_graph_ref != graph_ref
            or demand.resolved_job_work_graph_ref != graph_ref
            or demand.work_unit_ref != unit_ref
            or demand.job_resource_policy_ref != job_resource_policy_v2_ref(job_policy)
        ):
            raise ResourceControlPolicyError("resource demand source binding is stale")
        stored_unit, _ = self.job_store._get_work_unit(
            connection,
            work_unit.resolved_work_unit_id,
        )
        if stored_unit != work_unit:
            raise ResourceControlPolicyError("resource demand work unit is not current")

    def _prepare_admission_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        job_policy: JobResourcePolicyV2,
        demand: WorkResourceDemandV2,
        intended_attempt: int,
        audit: ContractAudit,
    ) -> tuple[
        ResourceAdmissionDecisionV2,
        ResourcePoolPolicyV2,
        ResourcePoolHeadRecord,
        JobResourceHeadRecord,
    ]:
        self._validate_acquisition_sources(
            connection,
            graph=graph,
            work_unit=work_unit,
            job_policy=job_policy,
            demand=demand,
        )
        if demand.attempt != intended_attempt:
            raise ResourceControlPolicyError(
                "resource demand attempt does not match the dispatchable attempt"
            )
        self._insert_or_validate_demand(connection, demand)
        if connection.execute(
            """
            SELECT 1
            FROM resource_admission_decisions
            WHERE work_resource_demand_id = ?
              AND outcome IN ('BUDGET_EXHAUSTED', 'UNSATISFIABLE_DEMAND')
            LIMIT 1
            """,
            (demand.work_resource_demand_id,),
        ).fetchone():
            raise ResourceControlPolicyError("resource demand already has a terminal admission decision")
        pool_policy = self._get_pool_policy(
            connection,
            job_policy.resource_pool_policy_ref.object_id,
        )
        pool_head = self._get_pool_head(
            connection,
            pool_policy.resource_pool_policy_id,
        )
        job_head = self._get_job_head(
            connection,
            job_policy.job_resource_policy_id,
        )
        outcome, constrained = _classify(
            pool_policy=pool_policy,
            job_policy=job_policy,
            pool_head=pool_head,
            job_head=job_head,
            demand=demand,
        )
        policy_ref = job_resource_policy_v2_ref(job_policy)
        demand_ref = work_resource_demand_v2_ref(demand)
        decision = ResourceAdmissionDecisionV2.create(
            job_resource_policy_ref=policy_ref,
            work_resource_demand_ref=demand_ref,
            outcome=outcome,
            constrained_resources=constrained,
            pool_head_version_before=pool_head.row_version,
            job_head_version_before=job_head.row_version,
            decided_at=self.job_store._clock(),
            audit=_resource_audit(audit, (policy_ref, demand_ref)),
        )
        return decision, pool_policy, pool_head, job_head

    def _insert_admission_decision_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        decision: ResourceAdmissionDecisionV2,
        demand: WorkResourceDemandV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO resource_admission_decisions (
                resource_admission_decision_id, work_resource_demand_id,
                outcome, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                decision.resource_admission_decision_id,
                demand.work_resource_demand_id,
                decision.outcome,
                self.job_store._record_json(decision),
            ),
        )

    def _insert_admitted_reservation_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        holder_ref: ObjectRef,
        demand: WorkResourceDemandV2,
        decision: ResourceAdmissionDecisionV2,
        lease: WorkLeaseV2,
        pool_head: ResourcePoolHeadRecord,
        job_head: JobResourceHeadRecord,
        audit: ContractAudit,
    ) -> WorkResourceReservationV2:
        if decision.outcome is not ResourceAdmissionOutcomeV2.ADMITTED:
            raise ResourceControlPolicyError("only admitted resource decision can create a reservation")
        demand_ref = work_resource_demand_v2_ref(demand)
        decision_ref = resource_admission_decision_v2_ref(decision)
        lease_ref = work_lease_v2_ref(lease)
        dispatch_ref = lease.work_dispatch_decision_ref
        handle_ref = _execution_handle_ref(lease)
        reservation = WorkResourceReservationV2.create(
            resource_admission_decision_ref=decision_ref,
            work_resource_demand_ref=demand_ref,
            work_lease_ref=lease_ref,
            work_dispatch_decision_ref=dispatch_ref,
            holder_ref=holder_ref,
            fencing_token=lease.fencing_token,
            execution_handle_ref=handle_ref,
            capacity_units=demand.capacity_units,
            budget_allowance=demand.budget_allowance,
            acquired_at=lease.acquired_at,
            audit=_resource_audit(
                audit,
                (
                    decision_ref,
                    demand_ref,
                    lease_ref,
                    dispatch_ref,
                    holder_ref,
                    handle_ref,
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO work_resource_reservations (
                work_resource_reservation_id,
                resource_admission_decision_id,
                work_resource_demand_id, work_lease_id, job_id,
                resolved_work_unit_id, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reservation.work_resource_reservation_id,
                decision.resource_admission_decision_id,
                demand.work_resource_demand_id,
                lease.work_lease_id,
                graph.job_id,
                work_unit.resolved_work_unit_id,
                self.job_store._record_json(reservation),
            ),
        )
        reservation_head = WorkResourceReservationHeadRecord(
            work_resource_reservation_id=reservation.work_resource_reservation_id,
            state=WorkResourceReservationStateV2.ACTIVE,
            reservation_version=0,
        )
        connection.execute(
            """
            INSERT INTO work_resource_reservation_heads (
                work_resource_reservation_id, state,
                reservation_version, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                reservation.work_resource_reservation_id,
                reservation_head.state,
                reservation_head.reservation_version,
                self.job_store._record_json(reservation_head),
            ),
        )
        self._update_pool_head(
            connection,
            pool_head.model_copy(
                update={
                    "active_capacity": pool_head.active_capacity.add(demand.capacity_units),
                    "row_version": pool_head.row_version + 1,
                }
            ),
            expected_version=pool_head.row_version,
        )
        self._update_job_head(
            connection,
            job_head.model_copy(
                update={
                    "active_capacity": job_head.active_capacity.add(demand.capacity_units),
                    "reserved_budget": job_head.reserved_budget.add(demand.budget_allowance),
                    "row_version": job_head.row_version + 1,
                }
            ),
            expected_version=job_head.row_version,
        )
        return reservation

    def _insert_or_validate_demand(
        self,
        connection: sqlite3.Connection,
        demand: WorkResourceDemandV2,
    ) -> None:
        row = connection.execute(
            """
            SELECT record_json
            FROM work_resource_demands
            WHERE resolved_work_unit_id = ? AND attempt = ?
            """,
            (demand.work_unit_ref.object_id, demand.attempt),
        ).fetchone()
        if row is not None:
            stored = WorkResourceDemandV2.model_validate_json(str(row["record_json"]))
            if stored != demand:
                raise ImmutableResultError("work attempt already has a different resource demand")
            return
        connection.execute(
            """
            INSERT INTO work_resource_demands (
                work_resource_demand_id, resolved_work_unit_id,
                attempt, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                demand.work_resource_demand_id,
                demand.work_unit_ref.object_id,
                demand.attempt,
                self.job_store._record_json(demand),
            ),
        )

    def _release_completed(
        self,
        connection: sqlite3.Connection,
        *,
        reservation: WorkResourceReservationV2,
        lease_event: WorkLeaseEventV2,
        usage: ResourceUsageV2,
        audit: ContractAudit,
    ) -> WorkResourceEventV2:
        policy = self._get_job_policy_for_reservation(connection, reservation)
        pool_head = self._get_pool_head(
            connection,
            policy.resource_pool_policy_ref.object_id,
        )
        job_head = self._get_job_head(connection, policy.job_resource_policy_id)
        released = NonModelResourceVectorV2(
            processes=reservation.capacity_units.processes,
            renderers=reservation.capacity_units.renderers,
            network_requests=reservation.capacity_units.network_requests,
            storage_bytes=(reservation.capacity_units.storage_bytes - usage.observed.storage_bytes),
        )
        reservation_ref = work_resource_reservation_v2_ref(reservation)
        lease_event_ref = work_lease_event_v2_ref(lease_event)
        resource_event = WorkResourceEventV2.create(
            work_resource_reservation_ref=reservation_ref,
            work_lease_event_ref=lease_event_ref,
            event_kind=WorkResourceEventKindV2.COMPLETED_RELEASED,
            reservation_version=1,
            charged_usage=usage,
            released_capacity=released,
            resource_termination_request_ref=None,
            audit=_resource_audit(audit, (reservation_ref, lease_event_ref)),
        )
        connection.execute(
            """
            INSERT INTO work_resource_events (
                work_resource_event_id, work_resource_reservation_id,
                reservation_version, event_kind, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                resource_event.work_resource_event_id,
                reservation.work_resource_reservation_id,
                resource_event.reservation_version,
                resource_event.event_kind,
                self.job_store._record_json(resource_event),
            ),
        )
        self._update_pool_head(
            connection,
            pool_head.model_copy(
                update={
                    "active_capacity": pool_head.active_capacity.subtract(released),
                    "row_version": pool_head.row_version + 1,
                }
            ),
            expected_version=pool_head.row_version,
        )
        self._update_job_head(
            connection,
            job_head.model_copy(
                update={
                    "active_capacity": job_head.active_capacity.subtract(released),
                    "reserved_budget": job_head.reserved_budget.subtract(reservation.budget_allowance),
                    "consumed_budget": job_head.consumed_budget.add(usage.observed),
                    "row_version": job_head.row_version + 1,
                }
            ),
            expected_version=job_head.row_version,
        )
        released_head = WorkResourceReservationHeadRecord(
            work_resource_reservation_id=reservation.work_resource_reservation_id,
            state=WorkResourceReservationStateV2.RELEASED,
            reservation_version=1,
        )
        connection.execute(
            """
            UPDATE work_resource_reservation_heads
            SET state = ?, reservation_version = ?, record_json = ?
            WHERE work_resource_reservation_id = ?
              AND reservation_version = 0
            """,
            (
                released_head.state,
                released_head.reservation_version,
                self.job_store._record_json(released_head),
                reservation.work_resource_reservation_id,
            ),
        )
        return resource_event

    def _close_aborted_reservation(
        self,
        connection: sqlite3.Connection,
        *,
        reservation: WorkResourceReservationV2,
        lease_event: WorkLeaseEventV2,
        reason: WorkResourceTerminationReasonV2,
        audit: ContractAudit,
    ) -> tuple[WorkResourceEventV2, WorkResourceTerminationRequestV2 | None]:
        demand = self.job_store._get_record(
            connection,
            "work_resource_demands",
            "work_resource_demand_id",
            reservation.work_resource_demand_ref.object_id,
            WorkResourceDemandV2,
        )
        policy = self._get_job_policy(
            connection,
            demand.job_resource_policy_ref.object_id,
        )
        pool_head = self._get_pool_head(
            connection,
            policy.resource_pool_policy_ref.object_id,
        )
        job_head = self._get_job_head(connection, policy.job_resource_policy_id)
        reservation_ref = work_resource_reservation_v2_ref(reservation)
        lease_event_ref = work_lease_event_v2_ref(lease_event)
        termination_request: WorkResourceTerminationRequestV2 | None = None
        if demand.termination_required:
            termination_request = WorkResourceTerminationRequestV2.create(
                work_resource_reservation_ref=reservation_ref,
                work_lease_event_ref=lease_event_ref,
                execution_handle_ref=reservation.execution_handle_ref,
                fencing_token=reservation.fencing_token,
                reason=reason,
                requested_at=self.job_store._clock(),
                audit=_resource_audit(
                    audit,
                    (
                        reservation_ref,
                        lease_event_ref,
                        reservation.execution_handle_ref,
                    ),
                ),
            )
            connection.execute(
                """
                INSERT INTO work_resource_termination_requests (
                    work_resource_termination_request_id,
                    work_resource_reservation_id, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    termination_request.work_resource_termination_request_id,
                    reservation.work_resource_reservation_id,
                    self.job_store._record_json(termination_request),
                ),
            )
            request_ref = work_resource_termination_request_v2_ref(termination_request)
            event_kind = WorkResourceEventKindV2.RELEASE_PENDING_TERMINATION
            released = NonModelResourceVectorV2.zero()
            event_refs: tuple[ObjectRef, ...] = (
                reservation_ref,
                lease_event_ref,
                request_ref,
            )
            next_state = WorkResourceReservationStateV2.RELEASE_PENDING_TERMINATION
        else:
            request_ref = None
            event_kind = WorkResourceEventKindV2.COMPLETED_RELEASED
            released = NonModelResourceVectorV2(
                processes=reservation.capacity_units.processes,
                renderers=reservation.capacity_units.renderers,
                network_requests=reservation.capacity_units.network_requests,
                storage_bytes=0,
            )
            event_refs = (reservation_ref, lease_event_ref)
            next_state = WorkResourceReservationStateV2.RELEASED
        resource_event = WorkResourceEventV2.create(
            work_resource_reservation_ref=reservation_ref,
            work_lease_event_ref=lease_event_ref,
            event_kind=event_kind,
            reservation_version=1,
            charged_usage=ResourceUsageV2(observed=reservation.budget_allowance),
            released_capacity=released,
            resource_termination_request_ref=request_ref,
            audit=_resource_audit(audit, event_refs),
        )
        self._insert_resource_event(connection, reservation, resource_event)
        self._update_pool_head(
            connection,
            pool_head.model_copy(
                update={
                    "active_capacity": pool_head.active_capacity.subtract(released),
                    "row_version": pool_head.row_version + 1,
                }
            ),
            expected_version=pool_head.row_version,
        )
        updated_job_head = job_head.model_copy(
            update={
                "active_capacity": job_head.active_capacity.subtract(released),
                "reserved_budget": job_head.reserved_budget.subtract(reservation.budget_allowance),
                "consumed_budget": job_head.consumed_budget.add(reservation.budget_allowance),
                "row_version": job_head.row_version + 1,
            }
        )
        self._update_job_head(
            connection,
            updated_job_head,
            expected_version=job_head.row_version,
        )
        next_head = WorkResourceReservationHeadRecord(
            work_resource_reservation_id=reservation.work_resource_reservation_id,
            state=next_state,
            reservation_version=1,
        )
        self._update_reservation_head(
            connection,
            next_head,
            expected_version=0,
        )
        return resource_event, termination_request

    def _require_active_reservation(
        self,
        connection: sqlite3.Connection,
        lease: WorkLeaseV2,
        reservation: WorkResourceReservationV2,
    ) -> None:
        stored = self._get_reservation(
            connection,
            reservation.work_resource_reservation_id,
        )
        if stored != reservation or reservation.work_lease_ref != work_lease_v2_ref(lease):
            raise StaleResourceReservationError("resource reservation is stale")
        head = self._get_reservation_head(
            connection,
            reservation.work_resource_reservation_id,
        )
        if head.state is not WorkResourceReservationStateV2.ACTIVE:
            raise StaleResourceReservationError("resource reservation is not active")

    def _insert_resource_event(
        self,
        connection: sqlite3.Connection,
        reservation: WorkResourceReservationV2,
        event: WorkResourceEventV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO work_resource_events (
                work_resource_event_id, work_resource_reservation_id,
                reservation_version, event_kind, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.work_resource_event_id,
                reservation.work_resource_reservation_id,
                event.reservation_version,
                event.event_kind,
                self.job_store._record_json(event),
            ),
        )

    def _update_reservation_head(
        self,
        connection: sqlite3.Connection,
        head: WorkResourceReservationHeadRecord,
        *,
        expected_version: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE work_resource_reservation_heads
            SET state = ?, reservation_version = ?, record_json = ?
            WHERE work_resource_reservation_id = ?
              AND reservation_version = ?
            """,
            (
                head.state,
                head.reservation_version,
                self.job_store._record_json(head),
                head.work_resource_reservation_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleResourceReservationError("resource reservation head changed concurrently")

    def _get_termination_request_for_reservation(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> WorkResourceTerminationRequestV2 | None:
        row = connection.execute(
            """
            SELECT work_resource_termination_request_id, record_json
            FROM work_resource_termination_requests
            WHERE work_resource_reservation_id = ?
            """,
            (reservation_id,),
        ).fetchone()
        if row is None:
            return None
        request = WorkResourceTerminationRequestV2.model_validate_json(str(row["record_json"]))
        reservation = self._get_reservation(connection, reservation_id)
        lease_event = self.job_store._get_work_lease_event(
            connection,
            request.work_lease_event_ref.object_id,
        )
        if (
            str(row["work_resource_termination_request_id"]) != request.work_resource_termination_request_id
            or request.work_resource_reservation_ref != work_resource_reservation_v2_ref(reservation)
            or request.work_lease_event_ref != work_lease_event_v2_ref(lease_event)
            or request.execution_handle_ref != reservation.execution_handle_ref
            or request.fencing_token != reservation.fencing_token
        ):
            raise ImmutableResultError("resource termination request binding is stale")
        return request

    def _get_latest_resource_event(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> WorkResourceEventV2:
        row = connection.execute(
            """
            SELECT work_resource_event_id
            FROM work_resource_events
            WHERE work_resource_reservation_id = ?
            ORDER BY reservation_version DESC
            LIMIT 1
            """,
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("resource reservation has no immutable event")
        return self._get_resource_event(
            connection,
            str(row["work_resource_event_id"]),
        )

    def _list_job_resource_closures(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        cancellation: WorkCancellationRecordV2,
    ) -> tuple[
        tuple[WorkResourceEventV2, ...],
        tuple[WorkResourceTerminationRequestV2, ...],
    ]:
        rows = connection.execute(
            """
            SELECT events.work_resource_event_id
            FROM work_resource_events AS events
            JOIN work_resource_reservations AS reservations
              ON reservations.work_resource_reservation_id =
                 events.work_resource_reservation_id
            WHERE reservations.job_id = ?
            ORDER BY events.work_resource_reservation_id,
                     events.reservation_version
            """,
            (job_id,),
        ).fetchall()
        cancelled_event_refs = set(cancellation.cancelled_lease_event_refs)
        observed_events = tuple(
            self._get_resource_event(
                connection,
                str(row["work_resource_event_id"]),
            )
            for row in rows
        )
        events = tuple(
            event
            for event in observed_events
            if event.reservation_version == 1 and event.work_lease_event_ref in cancelled_event_refs
        )
        request_rows = connection.execute(
            """
            SELECT requests.work_resource_termination_request_id,
                   requests.work_resource_reservation_id
            FROM work_resource_termination_requests AS requests
            JOIN work_resource_reservations AS reservations
              ON reservations.work_resource_reservation_id =
                 requests.work_resource_reservation_id
            WHERE reservations.job_id = ?
            ORDER BY requests.work_resource_termination_request_id
            """,
            (job_id,),
        ).fetchall()
        observed_requests = tuple(
            self._get_termination_request_for_reservation(
                connection,
                str(row["work_resource_reservation_id"]),
            )
            for row in request_rows
        )
        requests = tuple(
            request
            for request in observed_requests
            if request is not None and request.work_lease_event_ref in cancelled_event_refs
        )
        return events, requests

    def _get_pool_policy(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> ResourcePoolPolicyV2:
        row = connection.execute(
            """
            SELECT resource_pool_policy_id, resource_domain_id, record_json
            FROM resource_pool_policies
            WHERE resource_pool_policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if row is None:
            raise ResourceControlPolicyError("resource pool policy is missing")
        policy = ResourcePoolPolicyV2.model_validate_json(str(row["record_json"]))
        validate_resource_pool_policy_v2_identity(policy)
        if (
            str(row["resource_pool_policy_id"]) != policy.resource_pool_policy_id
            or str(row["resource_domain_id"]) != policy.resource_domain_ref.object_id
        ):
            raise ImmutableResultError("resource pool policy columns are inconsistent")
        return policy

    def _get_job_policy(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> JobResourcePolicyV2:
        row = connection.execute(
            """
            SELECT job_resource_policy_id, resolved_job_work_graph_id,
                   job_id, resource_pool_policy_id, record_json
            FROM job_resource_policies
            WHERE job_resource_policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if row is None:
            raise ResourceControlPolicyError("Job resource policy is missing")
        policy = JobResourcePolicyV2.model_validate_json(str(row["record_json"]))
        validate_job_resource_policy_v2_identity(policy)
        if (
            str(row["job_resource_policy_id"]) != policy.job_resource_policy_id
            or str(row["resolved_job_work_graph_id"]) != policy.resolved_job_work_graph_ref.object_id
            or str(row["job_id"]) != policy.dataset_job_spec_ref.object_id
            or str(row["resource_pool_policy_id"]) != policy.resource_pool_policy_ref.object_id
        ):
            raise ImmutableResultError("Job resource policy columns are inconsistent")
        pool = self._get_pool_policy(
            connection,
            policy.resource_pool_policy_ref.object_id,
        )
        if resource_pool_policy_v2_ref(pool) != policy.resource_pool_policy_ref:
            raise ImmutableResultError("Job resource policy pool ref is stale")
        job_spec = self.job_store._get_job_spec(
            connection,
            policy.dataset_job_spec_ref.object_id,
        )
        graph = self.job_store._get_job_work_graph_by_id(
            connection,
            policy.resolved_job_work_graph_ref.object_id,
        )
        if (
            dataset_job_spec_v2_ref(job_spec) != policy.dataset_job_spec_ref
            or resolved_job_work_graph_v2_ref(graph) != policy.resolved_job_work_graph_ref
            or graph.job_id != job_spec.job_id
        ):
            raise ImmutableResultError("Job resource policy source refs are stale")
        return policy

    def _get_pool_head(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> ResourcePoolHeadRecord:
        row = connection.execute(
            """
            SELECT row_version, record_json
            FROM resource_pool_heads
            WHERE resource_pool_policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if row is None:
            raise ResourceControlPolicyError("resource pool head is missing")
        head = ResourcePoolHeadRecord.model_validate_json(str(row["record_json"]))
        if head.resource_pool_policy_id != policy_id or head.row_version != int(row["row_version"]):
            raise ImmutableResultError("resource pool head columns are inconsistent")
        reservation_rows = connection.execute(
            """
            SELECT reservations.work_resource_reservation_id
            FROM work_resource_reservations AS reservations
            JOIN job_resource_policies AS job_policies
              ON job_policies.job_id = reservations.job_id
            WHERE job_policies.resource_pool_policy_id = ?
            ORDER BY reservations.work_resource_reservation_id
            """,
            (policy_id,),
        ).fetchall()
        expected_active = NonModelResourceVectorV2.zero()
        expected_version = 0
        for reservation_row in reservation_rows:
            active, _reserved, _consumed, event_count, _state = self._reservation_effective_state(
                connection,
                str(reservation_row["work_resource_reservation_id"]),
            )
            expected_active = expected_active.add(active)
            expected_version += 1 + event_count
        if head.active_capacity != expected_active or head.row_version != expected_version:
            raise ImmutableResultError("resource pool head does not rebuild from immutable reservations")
        return head

    def _get_job_head(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> JobResourceHeadRecord:
        row = connection.execute(
            """
            SELECT row_version, record_json
            FROM job_resource_heads
            WHERE job_resource_policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if row is None:
            raise ResourceControlPolicyError("Job resource head is missing")
        head = JobResourceHeadRecord.model_validate_json(str(row["record_json"]))
        if head.job_resource_policy_id != policy_id or head.row_version != int(row["row_version"]):
            raise ImmutableResultError("Job resource head columns are inconsistent")
        job_row = connection.execute(
            """
            SELECT job_id
            FROM job_resource_policies
            WHERE job_resource_policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if job_row is None:
            raise ImmutableResultError("Job resource policy owner is missing")
        reservation_rows = connection.execute(
            """
            SELECT work_resource_reservation_id
            FROM work_resource_reservations
            WHERE job_id = ?
            ORDER BY work_resource_reservation_id
            """,
            (str(job_row["job_id"]),),
        ).fetchall()
        expected_active = NonModelResourceVectorV2.zero()
        expected_reserved = NonModelResourceVectorV2.zero()
        expected_consumed = NonModelResourceVectorV2.zero()
        expected_version = 0
        for reservation_row in reservation_rows:
            active, reserved, consumed, event_count, _state = self._reservation_effective_state(
                connection,
                str(reservation_row["work_resource_reservation_id"]),
            )
            expected_active = expected_active.add(active)
            expected_reserved = expected_reserved.add(reserved)
            expected_consumed = expected_consumed.add(consumed)
            expected_version += 1 + event_count
        if (
            head.active_capacity != expected_active
            or head.reserved_budget != expected_reserved
            or head.consumed_budget != expected_consumed
            or head.row_version != expected_version
        ):
            raise ImmutableResultError("Job resource head does not rebuild from immutable reservations")
        return head

    def _get_reservation_head(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> WorkResourceReservationHeadRecord:
        row = connection.execute(
            """
            SELECT state, reservation_version, record_json
            FROM work_resource_reservation_heads
            WHERE work_resource_reservation_id = ?
            """,
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise StaleResourceReservationError("resource reservation head is missing")
        head = WorkResourceReservationHeadRecord.model_validate_json(str(row["record_json"]))
        if (
            head.work_resource_reservation_id != reservation_id
            or head.state.value != str(row["state"])
            or head.reservation_version != int(row["reservation_version"])
        ):
            raise ImmutableResultError("resource reservation head columns are inconsistent")
        _active, _reserved, _consumed, event_count, state = self._reservation_effective_state(
            connection, reservation_id
        )
        if head.reservation_version != event_count or head.state is not state:
            raise ImmutableResultError("resource reservation head does not rebuild from immutable events")
        return head

    def _reservation_effective_state(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> tuple[
        NonModelResourceVectorV2,
        NonModelResourceVectorV2,
        NonModelResourceVectorV2,
        int,
        WorkResourceReservationStateV2,
    ]:
        reservation = self._get_reservation(connection, reservation_id)
        event_rows = connection.execute(
            """
            SELECT work_resource_event_id, reservation_version,
                   event_kind, record_json
            FROM work_resource_events
            WHERE work_resource_reservation_id = ?
            ORDER BY reservation_version
            """,
            (reservation_id,),
        ).fetchall()
        released = NonModelResourceVectorV2.zero()
        consumed = NonModelResourceVectorV2.zero()
        expected_version = 1
        state = WorkResourceReservationStateV2.ACTIVE
        for event_row in event_rows:
            event = WorkResourceEventV2.model_validate_json(str(event_row["record_json"]))
            if (
                str(event_row["work_resource_event_id"]) != event.work_resource_event_id
                or str(event_row["event_kind"]) != event.event_kind.value
                or event.work_resource_reservation_ref != work_resource_reservation_v2_ref(reservation)
                or event.reservation_version != expected_version
                or int(event_row["reservation_version"]) != expected_version
            ):
                raise ImmutableResultError("resource event sequence or reservation binding is stale")
            allowed = {
                WorkResourceReservationStateV2.ACTIVE: {
                    WorkResourceEventKindV2.COMPLETED_RELEASED,
                    WorkResourceEventKindV2.RELEASE_PENDING_TERMINATION,
                },
                WorkResourceReservationStateV2.RELEASE_PENDING_TERMINATION: {
                    WorkResourceEventKindV2.TERMINATED_RELEASED,
                    WorkResourceEventKindV2.QUARANTINED,
                },
            }
            if event.event_kind not in allowed.get(state, set()):
                raise ImmutableResultError("resource event transition is not allowed")
            if event.resource_termination_request_ref is not None:
                termination_request = self.job_store._get_record(
                    connection,
                    "work_resource_termination_requests",
                    "work_resource_termination_request_id",
                    event.resource_termination_request_ref.object_id,
                    WorkResourceTerminationRequestV2,
                )
                if (
                    work_resource_termination_request_v2_ref(termination_request)
                    != event.resource_termination_request_ref
                    or termination_request.work_resource_reservation_ref
                    != event.work_resource_reservation_ref
                    or termination_request.work_lease_event_ref != event.work_lease_event_ref
                ):
                    raise ImmutableResultError("resource termination request event binding is stale")
            if event.resource_termination_result_ref is not None:
                termination_result = self.job_store._get_record(
                    connection,
                    "work_resource_termination_results",
                    "work_resource_termination_result_id",
                    event.resource_termination_result_ref.object_id,
                    WorkResourceTerminationResultV2,
                )
                termination_request = self.job_store._get_record(
                    connection,
                    "work_resource_termination_requests",
                    "work_resource_termination_request_id",
                    termination_result.termination_request_ref.object_id,
                    WorkResourceTerminationRequestV2,
                )
                if (
                    work_resource_termination_result_v2_ref(termination_result)
                    != event.resource_termination_result_ref
                    or work_resource_termination_request_v2_ref(termination_request)
                    != termination_result.termination_request_ref
                    or termination_request.work_resource_reservation_ref
                    != event.work_resource_reservation_ref
                    or termination_request.work_lease_event_ref != event.work_lease_event_ref
                ):
                    raise ImmutableResultError("resource termination result event binding is stale")
            released = released.add(event.released_capacity)
            consumed = consumed.add(event.charged_usage.observed)
            if event.event_kind is WorkResourceEventKindV2.RELEASE_PENDING_TERMINATION:
                state = WorkResourceReservationStateV2.RELEASE_PENDING_TERMINATION
            elif event.event_kind is WorkResourceEventKindV2.QUARANTINED:
                state = WorkResourceReservationStateV2.QUARANTINED
            else:
                state = WorkResourceReservationStateV2.RELEASED
            expected_version += 1
        try:
            active = reservation.capacity_units.subtract(released)
        except ValueError as exc:
            raise ImmutableResultError("resource events release more capacity than reserved") from exc
        if consumed.exceeds(reservation.budget_allowance):
            raise ImmutableResultError("resource events charge more budget than reserved")
        reserved = reservation.budget_allowance if not event_rows else NonModelResourceVectorV2.zero()
        return active, reserved, consumed, len(event_rows), state

    def _get_reservation(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> WorkResourceReservationV2:
        row = connection.execute(
            """
            SELECT work_resource_reservation_id,
                   resource_admission_decision_id,
                   work_resource_demand_id, work_lease_id,
                   job_id, resolved_work_unit_id, record_json
            FROM work_resource_reservations
            WHERE work_resource_reservation_id = ?
            """,
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise StaleResourceReservationError("resource reservation is missing")
        reservation = WorkResourceReservationV2.model_validate_json(str(row["record_json"]))
        if (
            str(row["work_resource_reservation_id"]) != reservation.work_resource_reservation_id
            or str(row["resource_admission_decision_id"])
            != reservation.resource_admission_decision_ref.object_id
            or str(row["work_resource_demand_id"]) != reservation.work_resource_demand_ref.object_id
            or str(row["work_lease_id"]) != reservation.work_lease_ref.object_id
        ):
            raise ImmutableResultError("resource reservation columns are inconsistent")
        decision = self.job_store._get_record(
            connection,
            "resource_admission_decisions",
            "resource_admission_decision_id",
            reservation.resource_admission_decision_ref.object_id,
            ResourceAdmissionDecisionV2,
        )
        demand = self.job_store._get_record(
            connection,
            "work_resource_demands",
            "work_resource_demand_id",
            reservation.work_resource_demand_ref.object_id,
            WorkResourceDemandV2,
        )
        lease = self.job_store._get_work_lease(
            connection,
            reservation.work_lease_ref.object_id,
        )
        if (
            resource_admission_decision_v2_ref(decision) != reservation.resource_admission_decision_ref
            or decision.outcome is not ResourceAdmissionOutcomeV2.ADMITTED
            or decision.work_resource_demand_ref != reservation.work_resource_demand_ref
            or decision.job_resource_policy_ref != demand.job_resource_policy_ref
            or work_resource_demand_v2_ref(demand) != reservation.work_resource_demand_ref
            or work_lease_v2_ref(lease) != reservation.work_lease_ref
            or lease.work_dispatch_decision_ref != reservation.work_dispatch_decision_ref
            or lease.holder_ref != reservation.holder_ref
            or lease.fencing_token != reservation.fencing_token
            or lease.attempt != demand.attempt
            or reservation.capacity_units != demand.capacity_units
            or reservation.budget_allowance != demand.budget_allowance
            or reservation.acquired_at != lease.acquired_at
            or reservation.execution_handle_ref != _execution_handle_ref(lease)
        ):
            raise ImmutableResultError("resource reservation source binding is stale")
        unit, _owner = self.job_store._get_work_unit(
            connection,
            str(row["resolved_work_unit_id"]),
        )
        if (
            resolved_work_unit_v2_ref(unit) != demand.work_unit_ref
            or lease.work_unit_ref != demand.work_unit_ref
            or unit.job_id != str(row["job_id"])
        ):
            raise ImmutableResultError("resource reservation work owner is stale")
        return reservation

    def _get_resource_event(
        self,
        connection: sqlite3.Connection,
        event_id: str,
    ) -> WorkResourceEventV2:
        event = self.job_store._get_record(
            connection,
            "work_resource_events",
            "work_resource_event_id",
            event_id,
            WorkResourceEventV2,
        )
        self._get_reservation_head(
            connection,
            event.work_resource_reservation_ref.object_id,
        )
        return event

    def _get_job_policy_for_reservation(
        self,
        connection: sqlite3.Connection,
        reservation: WorkResourceReservationV2,
    ) -> JobResourcePolicyV2:
        demand = self.job_store._get_record(
            connection,
            "work_resource_demands",
            "work_resource_demand_id",
            reservation.work_resource_demand_ref.object_id,
            WorkResourceDemandV2,
        )
        return self._get_job_policy(
            connection,
            demand.job_resource_policy_ref.object_id,
        )

    def _load_admission_result(
        self,
        connection: sqlite3.Connection,
        decision_id: str,
    ) -> ResourceAdmissionResult:
        decision = self.job_store._get_record(
            connection,
            "resource_admission_decisions",
            "resource_admission_decision_id",
            decision_id,
            ResourceAdmissionDecisionV2,
        )
        demand = self.job_store._get_record(
            connection,
            "work_resource_demands",
            "work_resource_demand_id",
            decision.work_resource_demand_ref.object_id,
            WorkResourceDemandV2,
        )
        policy = self._get_job_policy(
            connection,
            decision.job_resource_policy_ref.object_id,
        )
        if (
            work_resource_demand_v2_ref(demand) != decision.work_resource_demand_ref
            or job_resource_policy_v2_ref(policy) != decision.job_resource_policy_ref
            or demand.job_resource_policy_ref != decision.job_resource_policy_ref
        ):
            raise ImmutableResultError("resource admission decision source binding is stale")
        row = connection.execute(
            """
            SELECT work_resource_reservation_id, work_lease_id
            FROM work_resource_reservations
            WHERE resource_admission_decision_id = ?
            """,
            (decision_id,),
        ).fetchone()
        if row is None:
            return ResourceAdmissionResult(decision=decision)
        reservation = self._get_reservation(
            connection,
            str(row["work_resource_reservation_id"]),
        )
        lease = self.job_store._get_work_lease(connection, str(row["work_lease_id"]))
        return ResourceAdmissionResult(
            decision=decision,
            lease=lease,
            reservation=reservation,
        )

    def _update_pool_head(
        self,
        connection: sqlite3.Connection,
        head: ResourcePoolHeadRecord,
        *,
        expected_version: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE resource_pool_heads
            SET row_version = ?, record_json = ?
            WHERE resource_pool_policy_id = ? AND row_version = ?
            """,
            (
                head.row_version,
                self.job_store._record_json(head),
                head.resource_pool_policy_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ResourceControlPolicyError("resource pool head changed concurrently")

    def _update_job_head(
        self,
        connection: sqlite3.Connection,
        head: JobResourceHeadRecord,
        *,
        expected_version: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE job_resource_heads
            SET row_version = ?, record_json = ?
            WHERE job_resource_policy_id = ? AND row_version = ?
            """,
            (
                head.row_version,
                self.job_store._record_json(head),
                head.job_resource_policy_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ResourceControlPolicyError("Job resource head changed concurrently")


def _classify(
    *,
    pool_policy: ResourcePoolPolicyV2,
    job_policy: JobResourcePolicyV2,
    pool_head: ResourcePoolHeadRecord,
    job_head: JobResourceHeadRecord,
    demand: WorkResourceDemandV2,
) -> tuple[ResourceAdmissionOutcomeV2, tuple[ResourceKindV2, ...]]:
    unsatisfiable = set(demand.capacity_units.exceeds(pool_policy.capacity))
    unsatisfiable.update(demand.capacity_units.exceeds(job_policy.job_concurrency))
    if unsatisfiable:
        return (
            ResourceAdmissionOutcomeV2.UNSATISFIABLE_DEMAND,
            tuple(sorted(unsatisfiable, key=lambda item: item.value)),
        )
    budget_total = job_head.consumed_budget.add(job_head.reserved_budget).add(demand.budget_allowance)
    exhausted = budget_total.exceeds(job_policy.job_budget)
    if exhausted:
        return ResourceAdmissionOutcomeV2.BUDGET_EXHAUSTED, exhausted
    occupied = set(pool_head.active_capacity.add(demand.capacity_units).exceeds(pool_policy.capacity))
    occupied.update(job_head.active_capacity.add(demand.capacity_units).exceeds(job_policy.job_concurrency))
    if occupied:
        return (
            ResourceAdmissionOutcomeV2.WAITING_CAPACITY,
            tuple(sorted(occupied, key=lambda item: item.value)),
        )
    return ResourceAdmissionOutcomeV2.ADMITTED, ()


def _resource_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    versions = tuple(item for item in audit.governing_versions if item.component != "resource-control")
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=(
            *versions,
            VersionBinding(
                component="resource-control",
                version=R6_RESOURCE_CONTROL_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(
            sorted(
                refs,
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
    )


def _execution_handle_ref(lease: WorkLeaseV2) -> ObjectRef:
    digest = _digest(f"{lease.work_lease_id}|{lease.fencing_token}|{lease.holder_ref.object_sha256}")
    return ObjectRef(
        object_type="resource-execution-handle",
        object_id=f"resource-execution-handle://sha256/{digest}",
        object_version="v1",
        object_sha256=digest,
    )


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _object_ref(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _internal_key(idempotency_key: str, suffix: str) -> str:
    return f"resource-control://sha256/{_digest(f'{idempotency_key}|{suffix}')}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
