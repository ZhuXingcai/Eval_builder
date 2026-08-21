from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta

from pydantic import model_validator

from env_mock_agent.facade import (
    ExecutionTelemetryV2,
    FacadeObjectRef,
    ModelControlGrantV2,
    ModelDemandModeFacadeV2,
    ModelInvocationCancellationRequestV2,
    ModelInvocationCancellationResultV2,
    ModelInvocationDescriptorV2,
    model_invocation_cancellation_request_ref,
    model_invocation_cancellation_result_ref,
    model_invocation_descriptor_ref,
    validate_model_invocation_cancellation_result_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    FailureClass,
    FailureRecord,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.model_control_v2 import (
    R6_MODEL_CONTROL_POLICY_VERSION,
    JobModelPolicyV2,
    ModelAdmissionDecisionV2,
    ModelAdmissionOutcomeV2,
    ModelDemandModeV2,
    ModelRateDimensionV2,
    ModelRatePoolPolicyV2,
    ModelUsageSourceV2,
    ModelUsageV2,
    ProviderBackpressureKindV2,
    ProviderBackpressureRecordV2,
    WorkModelCancellationOutcomeV2,
    WorkModelCancellationReasonV2,
    WorkModelCancellationRequestV2,
    WorkModelCancellationResultV2,
    WorkModelDemandV2,
    WorkModelEventKindV2,
    WorkModelEventV2,
    WorkModelReservationStateV2,
    WorkModelReservationV2,
    job_model_policy_v2_ref,
    model_admission_decision_v2_ref,
    model_rate_pool_policy_v2_ref,
    provider_backpressure_record_v2_ref,
    validate_job_model_policy_v2_identity,
    validate_model_rate_pool_policy_v2_identity,
    work_model_cancellation_request_v2_ref,
    work_model_cancellation_result_v2_ref,
    work_model_demand_v2_ref,
    work_model_event_v2_ref,
    work_model_reservation_v2_ref,
)
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
    JobResourcePolicyV2,
    ResourceAdmissionDecisionV2,
    ResourceAdmissionOutcomeV2,
    ResourceUsageV2,
    WorkResourceDemandV2,
    WorkResourceEventV2,
    WorkResourceReservationV2,
    WorkResourceTerminationReasonV2,
    WorkResourceTerminationRequestV2,
    job_resource_policy_v2_ref,
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
    JobModelHeadRecord,
    JobRecord,
    ModelRatePoolHeadRecord,
    WorkModelReservationHeadRecord,
)
from eval_factory.orchestration.resources import ResourceControlService

RATE_CREDITS_PER_UNIT = 60_000_000


class ModelControlPolicyError(JobStoreError):
    pass


class StaleModelReservationError(JobStoreError):
    pass


class ModelAdmissionResult(ContractModelV2):
    decision: ModelAdmissionDecisionV2
    lease: WorkLeaseV2 | None = None
    reservation: WorkModelReservationV2 | None = None

    @model_validator(mode="after")
    def validate_result(self) -> ModelAdmissionResult:
        admitted = self.decision.outcome is ModelAdmissionOutcomeV2.ADMITTED
        if admitted != (self.lease is not None and self.reservation is not None):
            raise ValueError("model admission result requires lease and reservation exactly when admitted")
        return self


class ModelCompletionResult(ContractModelV2):
    lease_event: WorkLeaseEventV2
    model_event: WorkModelEventV2


class CombinedCompletionResult(ContractModelV2):
    lease_event: WorkLeaseEventV2
    model_event: WorkModelEventV2
    resource_event: WorkResourceEventV2


class ModelExpiryResult(ContractModelV2):
    lease_event: WorkLeaseEventV2
    model_event: WorkModelEventV2
    cancellation_request: WorkModelCancellationRequestV2


class ModelCancellationResult(ContractModelV2):
    cancellation_result: WorkModelCancellationResultV2
    model_event: WorkModelEventV2


class ModelJobCancellationResult(ContractModelV2):
    cancellation: WorkCancellationRecordV2
    model_events: tuple[WorkModelEventV2, ...]
    cancellation_requests: tuple[WorkModelCancellationRequestV2, ...]


class CombinedJobCancellationResult(ContractModelV2):
    cancellation: WorkCancellationRecordV2
    model_events: tuple[WorkModelEventV2, ...]
    model_cancellation_requests: tuple[WorkModelCancellationRequestV2, ...]
    resource_events: tuple[WorkResourceEventV2, ...]
    resource_termination_requests: tuple[WorkResourceTerminationRequestV2, ...]


class CombinedAdmissionResult(ContractModelV2):
    model_decision: ModelAdmissionDecisionV2 | None = None
    resource_decision: ResourceAdmissionDecisionV2 | None = None
    lease: WorkLeaseV2 | None = None
    model_reservation: WorkModelReservationV2 | None = None
    resource_reservation: WorkResourceReservationV2 | None = None

    @model_validator(mode="after")
    def validate_result(self) -> CombinedAdmissionResult:
        admitted = (
            self.model_decision is not None
            and self.model_decision.outcome is ModelAdmissionOutcomeV2.ADMITTED
            and self.resource_decision is not None
            and self.resource_decision.outcome is ResourceAdmissionOutcomeV2.ADMITTED
        )
        has_positive_records = (
            self.lease is not None
            and self.model_reservation is not None
            and self.resource_reservation is not None
        )
        if admitted != has_positive_records:
            raise ValueError(
                "combined admission requires one lease and both reservations exactly when admitted"
            )
        if self.model_decision is None and self.resource_decision is None:
            raise ValueError("combined admission requires a model or resource decision")
        if has_positive_records and (
            self.model_reservation is None
            or self.resource_reservation is None
            or self.model_reservation.work_lease_ref != self.resource_reservation.work_lease_ref
        ):
            raise ValueError("combined reservations must share one work lease")
        return self


class CombinedExpiryResult(ContractModelV2):
    lease_event: WorkLeaseEventV2
    model_event: WorkModelEventV2
    resource_event: WorkResourceEventV2
    model_cancellation_request: WorkModelCancellationRequestV2
    resource_termination_request: WorkResourceTerminationRequestV2 | None = None


class ModelControlService:
    def __init__(self, job_store: JobStore) -> None:
        self.job_store = job_store

    def bind_pool_policy(
        self,
        *,
        provider_bucket_ref: ObjectRef,
        allowed_model_profile_refs: tuple[ObjectRef, ...],
        requests_per_minute: int,
        tokens_per_minute: int,
        request_burst: int,
        token_burst: int,
        max_concurrent_requests: int,
        minimum_backpressure_seconds: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ModelRatePoolPolicyV2:
        refs = (provider_bucket_ref, *allowed_model_profile_refs)
        policy = ModelRatePoolPolicyV2.create(
            provider_bucket_ref=provider_bucket_ref,
            allowed_model_profile_refs=allowed_model_profile_refs,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            request_burst=request_burst,
            token_burst=token_burst,
            max_concurrent_requests=max_concurrent_requests,
            minimum_backpressure_seconds=minimum_backpressure_seconds,
            audit=_model_audit(audit, refs),
        )
        request_sha256 = model_rate_pool_policy_v2_ref(policy).object_sha256
        scope = f"bind-model-rate-pool:{provider_bucket_ref.object_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "MODEL_RATE_POOL_POLICY":
                    raise IdempotencyConflictError("model rate pool policy response type is corrupt")
                stored = self._get_pool_policy(connection, prior[1])
                if stored != policy:
                    raise ImmutableResultError("stored model rate pool policy differs from request")
                return stored
            row = connection.execute(
                """
                SELECT model_rate_pool_policy_id
                FROM model_rate_pool_policies
                WHERE provider_bucket_id = ?
                """,
                (provider_bucket_ref.object_id,),
            ).fetchone()
            if row is not None:
                raise ImmutableResultError("provider bucket already has an immutable model rate policy")
            connection.execute(
                """
                INSERT INTO model_rate_pool_policies (
                    model_rate_pool_policy_id, provider_bucket_id, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    policy.model_rate_pool_policy_id,
                    provider_bucket_ref.object_id,
                    self.job_store._record_json(policy),
                ),
            )
            head = ModelRatePoolHeadRecord(
                model_rate_pool_policy_id=policy.model_rate_pool_policy_id,
                request_credits=policy.request_burst * RATE_CREDITS_PER_UNIT,
                token_credits=policy.token_burst * RATE_CREDITS_PER_UNIT,
                active_concurrency=0,
                blocked_until=None,
                effective_at=policy.audit.created_at,
                bucket_version=0,
                row_version=0,
            )
            connection.execute(
                """
                INSERT INTO model_rate_pool_heads (
                    model_rate_pool_policy_id, row_version, bucket_version, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    policy.model_rate_pool_policy_id,
                    head.row_version,
                    head.bucket_version,
                    self.job_store._record_json(head),
                ),
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="MODEL_RATE_POOL_POLICY",
                response_id=policy.model_rate_pool_policy_id,
                created_at=self.job_store._clock(),
            )
        return policy

    def bind_job_policy(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        pool_policies: tuple[ModelRatePoolPolicyV2, ...],
        model_profile_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> JobModelPolicyV2:
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        stored_spec = self.job_store.get_job_spec(graph.job_id)
        spec_ref = dataset_job_spec_v2_ref(stored_spec)
        pool_refs = tuple(
            sorted(
                (model_rate_pool_policy_v2_ref(item) for item in pool_policies),
                key=_ref_key,
            )
        )
        profile_refs = tuple(sorted(model_profile_refs, key=_ref_key))
        if tuple(sorted(stored_spec.model_profiles)) != tuple(sorted(ref.object_id for ref in profile_refs)):
            raise ModelControlPolicyError(
                "Job model profile refs do not match DatasetJobSpecV2.model_profiles"
            )
        policy = JobModelPolicyV2.create(
            dataset_job_spec_ref=spec_ref,
            resolved_job_work_graph_ref=graph_ref,
            model_rate_pool_policy_refs=pool_refs,
            model_profile_refs=profile_refs,
            max_concurrent_requests=stored_spec.concurrency.model_requests,
            max_model_requests=stored_spec.budget.max_model_requests,
            max_model_tokens=stored_spec.budget.max_model_tokens,
            audit=_model_audit(
                audit,
                (spec_ref, graph_ref, *pool_refs, *profile_refs),
            ),
        )
        request_sha256 = job_model_policy_v2_ref(policy).object_sha256
        scope = f"bind-job-model-policy:{graph.job_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "JOB_MODEL_POLICY":
                    raise IdempotencyConflictError("Job model policy response type is corrupt")
                stored = self._get_job_policy(connection, prior[1])
                if stored != policy:
                    raise ImmutableResultError("stored Job model policy differs from request")
                return stored
            stored_graph = self.job_store._get_job_work_graph_by_id(
                connection,
                graph.resolved_job_work_graph_id,
            )
            if stored_graph != graph:
                raise ModelControlPolicyError("Job model policy graph is not current")
            for pool in pool_policies:
                if (
                    self._get_pool_policy(
                        connection,
                        pool.model_rate_pool_policy_id,
                    )
                    != pool
                ):
                    raise ModelControlPolicyError("model rate pool policy is not current")
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
                raise ModelControlPolicyError("Job model policy must bind before the first dispatch")
            connection.execute(
                """
                INSERT INTO job_model_policies (
                    job_model_policy_id, resolved_job_work_graph_id, job_id, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    policy.job_model_policy_id,
                    graph.resolved_job_work_graph_id,
                    graph.job_id,
                    self.job_store._record_json(policy),
                ),
            )
            connection.executemany(
                """
                INSERT INTO job_model_policy_pools (
                    job_model_policy_id, model_rate_pool_policy_id
                ) VALUES (?, ?)
                """,
                tuple(
                    (policy.job_model_policy_id, ref.object_id) for ref in policy.model_rate_pool_policy_refs
                ),
            )
            head = JobModelHeadRecord(
                job_model_policy_id=policy.job_model_policy_id,
                active_concurrency=0,
                reserved_requests=0,
                reserved_tokens=0,
                consumed_requests=0,
                consumed_tokens=0,
                row_version=0,
            )
            connection.execute(
                """
                INSERT INTO job_model_heads (
                    job_model_policy_id, row_version, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    policy.job_model_policy_id,
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
                event_type="job-model-policy-bound",
                attributes=_attributes(
                    policy_id=policy.job_model_policy_id,
                ),
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="JOB_MODEL_POLICY",
                response_id=policy.job_model_policy_id,
                created_at=self.job_store._clock(),
            )
        return policy

    def create_demand(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        job_policy: JobModelPolicyV2,
        pool_policy: ModelRatePoolPolicyV2,
        model_profile_ref: ObjectRef,
        operation_ref: ObjectRef,
        model_execution_profile_ref: ObjectRef,
        attempt: int,
        mode: ModelDemandModeV2,
        request_allowance: int,
        input_token_allowance: int,
        output_token_allowance: int,
        cache_token_allowance: int,
        concurrency_units: int,
        audit: ContractAudit,
    ) -> WorkModelDemandV2:
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        unit_ref = resolved_work_unit_v2_ref(work_unit)
        policy_ref = job_model_policy_v2_ref(job_policy)
        pool_ref = model_rate_pool_policy_v2_ref(pool_policy)
        graph_owns_unit = work_unit in graph.work_units
        fanout_unit = work_unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP and work_unit.job_id == graph.job_id
        if (
            not (graph_owns_unit or fanout_unit)
            or job_policy.resolved_job_work_graph_ref != graph_ref
            or pool_ref not in job_policy.model_rate_pool_policy_refs
            or model_profile_ref not in job_policy.model_profile_refs
            or model_profile_ref not in pool_policy.allowed_model_profile_refs
        ):
            raise ModelControlPolicyError("model demand graph, work unit, pool, or profile is not current")
        refs = (
            graph_ref,
            unit_ref,
            policy_ref,
            pool_ref,
            model_profile_ref,
            operation_ref,
            model_execution_profile_ref,
        )
        return WorkModelDemandV2.create(
            resolved_job_work_graph_ref=graph_ref,
            work_unit_ref=unit_ref,
            job_model_policy_ref=policy_ref,
            model_rate_pool_policy_ref=pool_ref,
            model_profile_ref=model_profile_ref,
            operation_ref=operation_ref,
            model_execution_profile_ref=model_execution_profile_ref,
            attempt=attempt,
            mode=mode,
            request_allowance=request_allowance,
            input_token_allowance=input_token_allowance,
            output_token_allowance=output_token_allowance,
            cache_token_allowance=cache_token_allowance,
            concurrency_units=concurrency_units,
            audit=_model_audit(audit, refs),
        )

    def to_facade_controls(
        self,
        *,
        demand: WorkModelDemandV2,
        reservation: WorkModelReservationV2,
        idempotency_key: str,
    ) -> tuple[ModelInvocationDescriptorV2, ModelControlGrantV2]:
        with self.job_store._connect() as connection:
            stored_demand = self._get_demand(
                connection,
                demand.work_model_demand_id,
            )
            stored_reservation = self._get_reservation(
                connection,
                reservation.work_model_reservation_id,
            )
            if (
                stored_demand != demand
                or stored_reservation != reservation
                or reservation.work_model_demand_ref != work_model_demand_v2_ref(demand)
            ):
                raise StaleModelReservationError("model demand or reservation is stale")
            if (
                self._get_reservation_head(connection, reservation).state
                is not WorkModelReservationStateV2.ACTIVE
            ):
                raise StaleModelReservationError("model reservation is not active")
        descriptor = ModelInvocationDescriptorV2.create(
            operation_ref=_facade_ref(demand.operation_ref),
            model_profile_ref=_facade_ref(demand.model_profile_ref),
            mode=ModelDemandModeFacadeV2(demand.mode.value),
            request_allowance=demand.request_allowance,
            input_token_allowance=demand.input_token_allowance,
            output_token_allowance=demand.output_token_allowance,
            cache_token_allowance=demand.cache_token_allowance,
            idempotency_key=idempotency_key,
        )
        grant = ModelControlGrantV2.create(
            reservation_ref=_facade_ref(work_model_reservation_v2_ref(reservation)),
            invocation_handle_ref=_facade_ref(reservation.invocation_handle_ref),
            descriptor_ref=model_invocation_descriptor_ref(descriptor),
            model_profile_ref=descriptor.model_profile_ref,
            fencing_token=reservation.fencing_token,
            request_allowance=reservation.request_allowance,
            token_allowance=reservation.token_allowance,
            output_token_limit=demand.output_token_allowance,
        )
        return descriptor, grant

    def acquire(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        readiness_snapshot: WorkReadinessSnapshotV2,
        control_policy: WorkControlPolicyV2,
        job_policy: JobModelPolicyV2,
        demand: WorkModelDemandV2,
        holder_ref: ObjectRef,
        retry_decision: WorkRetryDecisionV2 | None,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ModelAdmissionResult:
        request_sha256 = _request_sha256(
            {
                "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(graph),
                "work_unit_ref": resolved_work_unit_v2_ref(work_unit),
                "work_readiness_snapshot_ref": work_readiness_snapshot_v2_ref(readiness_snapshot),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "job_model_policy_ref": job_model_policy_v2_ref(job_policy),
                "work_model_demand_ref": work_model_demand_v2_ref(demand),
                "holder_ref": holder_ref,
                "retry_decision_ref": (
                    work_retry_decision_v2_ref(retry_decision) if retry_decision is not None else None
                ),
            }
        )
        scope = f"acquire-work-with-model-control:{work_unit.resolved_work_unit_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "MODEL_ADMISSION_DECISION":
                    raise IdempotencyConflictError("model admission response type is corrupt")
                return self._load_admission_result(connection, prior[1])
            self._validate_acquisition_sources(
                connection,
                graph=graph,
                work_unit=work_unit,
                job_policy=job_policy,
                demand=demand,
            )
            _prior, intended_attempt, _eligible, _retry_ref, job = (
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
                raise ModelControlPolicyError("model demand attempt does not match dispatchable attempt")
            self._insert_or_validate_demand(connection, demand)
            if connection.execute(
                """
                SELECT 1
                FROM model_admission_decisions
                WHERE work_model_demand_id = ?
                  AND outcome IN ('BUDGET_EXHAUSTED', 'UNSATISFIABLE_DEMAND')
                LIMIT 1
                """,
                (demand.work_model_demand_id,),
            ).fetchone():
                raise ModelControlPolicyError("model demand already has a terminal admission decision")
            pool_policy = self._get_pool_policy(
                connection,
                demand.model_rate_pool_policy_ref.object_id,
            )
            stored_pool_head = self._get_pool_head(connection, pool_policy)
            job_head = self._get_job_head(connection, job_policy)
            now = self.job_store._clock()
            effective_pool_head = _refill_pool_head(
                stored_pool_head,
                pool_policy,
                now,
            )
            outcome, constrained, eligible_at = _classify(
                pool_policy=pool_policy,
                job_policy=job_policy,
                pool_head=effective_pool_head,
                job_head=job_head,
                demand=demand,
                now=now,
            )
            policy_ref = job_model_policy_v2_ref(job_policy)
            demand_ref = work_model_demand_v2_ref(demand)
            decision = ModelAdmissionDecisionV2.create(
                job_model_policy_ref=policy_ref,
                work_model_demand_ref=demand_ref,
                outcome=outcome,
                constrained_dimensions=constrained,
                eligible_at=eligible_at,
                pool_head_version_before=stored_pool_head.row_version,
                job_head_version_before=job_head.row_version,
                decided_at=now,
                audit=_model_audit(audit, (policy_ref, demand_ref)),
            )
            connection.execute(
                """
                INSERT INTO model_admission_decisions (
                    model_admission_decision_id, work_model_demand_id,
                    outcome, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    decision.model_admission_decision_id,
                    demand.work_model_demand_id,
                    decision.outcome,
                    self.job_store._record_json(decision),
                ),
            )
            lease: WorkLeaseV2 | None = None
            reservation: WorkModelReservationV2 | None = None
            if outcome is ModelAdmissionOutcomeV2.ADMITTED:
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
                    control_authorizations=frozenset({"MODEL"}),
                )
                reservation = self._insert_admitted_reservation(
                    connection,
                    graph=graph,
                    work_unit=work_unit,
                    holder_ref=holder_ref,
                    demand=demand,
                    decision=decision,
                    lease=lease,
                    pool_policy=pool_policy,
                    pool_head=effective_pool_head,
                    job_head=job_head,
                    now=now,
                    audit=audit,
                )
            self.job_store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job.job_id,
                aggregate_version=job.row_version,
                event_type="model-admission-decided",
                attributes=_attributes(
                    attempt=demand.attempt,
                    model_admission_decision_id=decision.model_admission_decision_id,
                    work_unit_id=demand.work_unit_ref.object_id,
                ),
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="MODEL_ADMISSION_DECISION",
                response_id=decision.model_admission_decision_id,
                created_at=now,
            )
            return ModelAdmissionResult(
                decision=decision,
                lease=lease,
                reservation=reservation,
            )

    def acquire_combined(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        readiness_snapshot: WorkReadinessSnapshotV2,
        control_policy: WorkControlPolicyV2,
        job_policy: JobModelPolicyV2,
        demand: WorkModelDemandV2,
        resource_job_policy: JobResourcePolicyV2,
        resource_demand: WorkResourceDemandV2,
        holder_ref: ObjectRef,
        retry_decision: WorkRetryDecisionV2 | None,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> CombinedAdmissionResult:
        request_sha256 = _request_sha256(
            {
                "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(graph),
                "work_unit_ref": resolved_work_unit_v2_ref(work_unit),
                "work_readiness_snapshot_ref": work_readiness_snapshot_v2_ref(readiness_snapshot),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "job_model_policy_ref": job_model_policy_v2_ref(job_policy),
                "work_model_demand_ref": work_model_demand_v2_ref(demand),
                "job_resource_policy_ref": resource_job_policy,
                "work_resource_demand_ref": resource_demand,
                "holder_ref": holder_ref,
                "retry_decision_ref": (
                    work_retry_decision_v2_ref(retry_decision) if retry_decision is not None else None
                ),
            }
        )
        scope = f"acquire-work-with-combined-control:{work_unit.resolved_work_unit_id}"
        resources = ResourceControlService(self.job_store)
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._load_combined_admission_result(
                    connection,
                    response_type=prior[0],
                    response_id=prior[1],
                    resources=resources,
                )
            _prior, intended_attempt, _eligible, _retry_ref, job = (
                self.job_store._validate_work_lease_admission(
                    connection,
                    graph=graph,
                    work_unit=work_unit,
                    readiness_snapshot=readiness_snapshot,
                    policy=control_policy,
                    retry_decision=retry_decision,
                )
            )
            (
                model_decision,
                model_pool_policy,
                model_pool_head,
                model_job_head,
            ) = self._prepare_admission_in_transaction(
                connection,
                graph=graph,
                work_unit=work_unit,
                job_policy=job_policy,
                demand=demand,
                intended_attempt=intended_attempt,
                audit=audit,
            )
            response_type: str
            response_id: str
            resource_decision: ResourceAdmissionDecisionV2 | None = None
            lease: WorkLeaseV2 | None = None
            model_reservation: WorkModelReservationV2 | None = None
            resource_reservation: WorkResourceReservationV2 | None = None
            if model_decision.outcome is not ModelAdmissionOutcomeV2.ADMITTED:
                self._insert_admission_decision_in_transaction(
                    connection,
                    decision=model_decision,
                    demand=demand,
                )
                response_type = "COMBINED_MODEL_ADMISSION"
                response_id = model_decision.model_admission_decision_id
            else:
                (
                    resource_decision,
                    _resource_pool_policy,
                    resource_pool_head,
                    resource_job_head,
                ) = resources._prepare_admission_in_transaction(
                    connection,
                    graph=graph,
                    work_unit=work_unit,
                    job_policy=resource_job_policy,
                    demand=resource_demand,
                    intended_attempt=intended_attempt,
                    audit=audit,
                )
                if resource_decision.outcome is not ResourceAdmissionOutcomeV2.ADMITTED:
                    resources._insert_admission_decision_in_transaction(
                        connection,
                        decision=resource_decision,
                        demand=resource_demand,
                    )
                    response_type = "COMBINED_RESOURCE_ADMISSION"
                    response_id = resource_decision.resource_admission_decision_id
                else:
                    self._insert_admission_decision_in_transaction(
                        connection,
                        decision=model_decision,
                        demand=demand,
                    )
                    resources._insert_admission_decision_in_transaction(
                        connection,
                        decision=resource_decision,
                        demand=resource_demand,
                    )
                    lease = self.job_store._acquire_work_lease(
                        graph=graph,
                        work_unit=work_unit,
                        readiness_snapshot=readiness_snapshot,
                        policy=control_policy,
                        holder_ref=holder_ref,
                        retry_decision=retry_decision,
                        audit=audit,
                        idempotency_key=_internal_key(
                            idempotency_key,
                            "combined-lease",
                        ),
                        connection=connection,
                        control_authorizations=frozenset({"MODEL", "RESOURCE"}),
                    )
                    resource_reservation = resources._insert_admitted_reservation_in_transaction(
                        connection,
                        graph=graph,
                        work_unit=work_unit,
                        holder_ref=holder_ref,
                        demand=resource_demand,
                        decision=resource_decision,
                        lease=lease,
                        pool_head=resource_pool_head,
                        job_head=resource_job_head,
                        audit=audit,
                    )
                    model_reservation = self._insert_admitted_reservation(
                        connection,
                        graph=graph,
                        work_unit=work_unit,
                        holder_ref=holder_ref,
                        demand=demand,
                        decision=model_decision,
                        lease=lease,
                        pool_policy=model_pool_policy,
                        pool_head=model_pool_head,
                        job_head=model_job_head,
                        now=self.job_store._clock(),
                        audit=audit,
                    )
                    response_type = "COMBINED_MODEL_ADMISSION"
                    response_id = model_decision.model_admission_decision_id
            self.job_store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job.job_id,
                aggregate_version=job.row_version,
                event_type="combined-control-admission-decided",
                attributes=_attributes(
                    attempt=demand.attempt,
                    model_admission_decision_id=(
                        model_decision.model_admission_decision_id if model_decision is not None else None
                    ),
                    resource_admission_decision_id=(
                        resource_decision.resource_admission_decision_id
                        if resource_decision is not None
                        else None
                    ),
                    work_unit_id=demand.work_unit_ref.object_id,
                ),
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type=response_type,
                response_id=response_id,
                created_at=self.job_store._clock(),
            )
            return CombinedAdmissionResult(
                model_decision=(
                    model_decision
                    if resource_decision is None
                    or resource_decision.outcome is ResourceAdmissionOutcomeV2.ADMITTED
                    else None
                ),
                resource_decision=resource_decision,
                lease=lease,
                model_reservation=model_reservation,
                resource_reservation=resource_reservation,
            )

    def complete_stage(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        reservation: WorkModelReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        status: StageRunStatus,
        output_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        checkpoint_ref: ObjectRef | None,
        metrics_ref: ObjectRef | None,
        usage: ModelUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> ModelCompletionResult:
        _require_model_receipt_ref(output_refs)
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(reservation),
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
        scope = f"complete-model-stage:{reservation.work_model_reservation_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_MODEL_EVENT":
                    raise IdempotencyConflictError("model completion response type is corrupt")
                model_event = self._get_model_event(connection, prior[1])
                if model_event.work_lease_event_ref is None:
                    raise ImmutableResultError("model completion event is missing lease event")
                lease_event = self.job_store._get_work_lease_event(
                    connection,
                    model_event.work_lease_event_ref.object_id,
                )
                return ModelCompletionResult(
                    lease_event=lease_event,
                    model_event=model_event,
                )
            stored = self._get_reservation(
                connection,
                reservation.work_model_reservation_id,
            )
            if stored != reservation or reservation.work_lease_ref != work_lease_v2_ref(lease):
                raise StaleModelReservationError("model reservation is stale")
            reservation_head = self._get_reservation_head(connection, reservation)
            if reservation_head.state is not WorkModelReservationStateV2.ACTIVE:
                raise StaleModelReservationError("model reservation is not active")
            if (
                usage.requests > reservation.request_allowance
                or usage.charged_tokens > reservation.token_allowance
            ):
                raise ModelControlPolicyError("observed model usage exceeds grant")
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
                control_authorizations=frozenset({"MODEL"}),
                metrics_model_profile_ref=self._get_demand(
                    connection,
                    reservation.work_model_demand_ref.object_id,
                ).model_profile_ref,
                metrics_model_usage=usage,
                metrics_telemetry=telemetry,
            )
            model_event = self._release_completed(
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
                response_type="WORK_MODEL_EVENT",
                response_id=model_event.work_model_event_id,
                created_at=self.job_store._clock(),
            )
            return ModelCompletionResult(
                lease_event=lease_event,
                model_event=model_event,
            )

    def complete_artifact_group(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        reservation: WorkModelReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        event_kind: WorkLeaseEventKindV2,
        result_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        usage: ModelUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
        artifact_slices: tuple[ArtifactMetricSliceV2, ...] = (),
    ) -> ModelCompletionResult:
        _require_model_receipt_ref(result_refs)
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(reservation),
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
        scope = f"complete-model-artifact:{reservation.work_model_reservation_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_MODEL_EVENT":
                    raise IdempotencyConflictError("model artifact completion response type is corrupt")
                model_event = self._get_model_event(connection, prior[1])
                if model_event.work_lease_event_ref is None:
                    raise ImmutableResultError("model artifact completion event is missing lease event")
                lease_event = self.job_store._get_work_lease_event(
                    connection,
                    model_event.work_lease_event_ref.object_id,
                )
                return ModelCompletionResult(
                    lease_event=lease_event,
                    model_event=model_event,
                )
            self._require_active_reservation(connection, lease, reservation)
            if (
                usage.requests > reservation.request_allowance
                or usage.charged_tokens > reservation.token_allowance
            ):
                raise ModelControlPolicyError("observed model usage exceeds grant")
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
                control_authorizations=frozenset({"MODEL"}),
                metrics_model_profile_ref=self._get_demand(
                    connection,
                    reservation.work_model_demand_ref.object_id,
                ).model_profile_ref,
                metrics_model_usage=usage,
                metrics_telemetry=telemetry,
                metrics_artifact_slices=artifact_slices,
            )
            model_event = self._release_completed(
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
                response_type="WORK_MODEL_EVENT",
                response_id=model_event.work_model_event_id,
                created_at=self.job_store._clock(),
            )
            return ModelCompletionResult(
                lease_event=lease_event,
                model_event=model_event,
            )

    def complete_artifact_group_combined(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        model_reservation: WorkModelReservationV2,
        resource_reservation: WorkResourceReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        event_kind: WorkLeaseEventKindV2,
        result_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        model_usage: ModelUsageV2,
        resource_usage: ResourceUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
        artifact_slices: tuple[ArtifactMetricSliceV2, ...] = (),
    ) -> CombinedCompletionResult:
        _require_model_receipt_ref(result_refs)
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(model_reservation),
                "work_resource_reservation_ref": resource_reservation,
                "holder_ref": holder_ref,
                "expected_lease_version": expected_lease_version,
                "event_kind": event_kind,
                "result_refs": result_refs,
                "failure": failure,
                "model_usage": model_usage,
                "resource_usage": resource_usage,
                "telemetry": telemetry,
                "artifact_slices": artifact_slices,
            }
        )
        scope = f"complete-combined-artifact:{lease.work_lease_id}"
        resources = ResourceControlService(self.job_store)
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._load_combined_completion_result(
                    connection,
                    response_type=prior[0],
                    model_event_id=prior[1],
                    resource_reservation=resource_reservation,
                    resources=resources,
                )
            self._require_active_reservation(
                connection,
                lease,
                model_reservation,
            )
            resources._require_active_reservation(
                connection,
                lease,
                resource_reservation,
            )
            if (
                model_usage.requests > model_reservation.request_allowance
                or model_usage.charged_tokens > model_reservation.token_allowance
            ):
                raise ModelControlPolicyError("observed model usage exceeds grant")
            exceeded = resource_usage.observed.exceeds(resource_reservation.budget_allowance)
            if exceeded:
                raise ModelControlPolicyError("observed resource usage exceeds combined grant")
            lease_event = self.job_store._complete_artifact_group_work_lease(
                lease=lease,
                policy=control_policy,
                holder_ref=holder_ref,
                expected_lease_version=expected_lease_version,
                event_kind=event_kind,
                result_refs=result_refs,
                failure=failure,
                audit=audit,
                idempotency_key=_internal_key(
                    idempotency_key,
                    "combined-artifact-lease-event",
                ),
                connection=connection,
                control_authorizations=frozenset({"MODEL", "RESOURCE"}),
                metrics_model_profile_ref=self._get_demand(
                    connection,
                    model_reservation.work_model_demand_ref.object_id,
                ).model_profile_ref,
                metrics_model_usage=model_usage,
                metrics_resource_usage=resource_usage,
                metrics_telemetry=telemetry,
                metrics_artifact_slices=artifact_slices,
            )
            resource_event = resources._release_completed(
                connection,
                reservation=resource_reservation,
                lease_event=lease_event,
                usage=resource_usage,
                audit=audit,
            )
            model_event = self._release_completed(
                connection,
                reservation=model_reservation,
                lease_event=lease_event,
                usage=model_usage,
                audit=audit,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_MODEL_EVENT",
                response_id=model_event.work_model_event_id,
                created_at=self.job_store._clock(),
            )
            return CombinedCompletionResult(
                lease_event=lease_event,
                model_event=model_event,
                resource_event=resource_event,
            )

    def complete_stage_backpressured(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        reservation: WorkModelReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        facade_receipt_ref: ObjectRef,
        kind: ProviderBackpressureKindV2,
        eligible_at: datetime,
        usage: ModelUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> ModelCompletionResult:
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(reservation),
                "holder_ref": holder_ref,
                "expected_lease_version": expected_lease_version,
                "facade_receipt_ref": facade_receipt_ref,
                "kind": kind,
                "eligible_at": eligible_at,
                "usage": usage,
                "telemetry": telemetry,
            }
        )
        scope = f"complete-model-backpressure:{reservation.work_model_reservation_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_MODEL_EVENT":
                    raise IdempotencyConflictError("model backpressure response type is corrupt")
                model_event = self._get_model_event(connection, prior[1])
                if model_event.work_lease_event_ref is None:
                    raise ImmutableResultError("model backpressure event is missing lease event")
                lease_event = self.job_store._get_work_lease_event(
                    connection,
                    model_event.work_lease_event_ref.object_id,
                )
                return ModelCompletionResult(
                    lease_event=lease_event,
                    model_event=model_event,
                )
            stored = self._get_reservation(
                connection,
                reservation.work_model_reservation_id,
            )
            if stored != reservation or reservation.work_lease_ref != work_lease_v2_ref(lease):
                raise StaleModelReservationError("model reservation is stale")
            if (
                self._get_reservation_head(connection, reservation).state
                is not WorkModelReservationStateV2.ACTIVE
            ):
                raise StaleModelReservationError("model reservation is not active")
            if (
                usage.requests > reservation.request_allowance
                or usage.charged_tokens > reservation.token_allowance
            ):
                raise ModelControlPolicyError("backpressure usage exceeds grant")
            now = self.job_store._clock()
            eligible_at = self._normalize_backpressure_eligible_at(
                connection,
                reservation=reservation,
                provider_eligible_at=eligible_at,
                now=now,
            )
            reservation_ref = work_model_reservation_v2_ref(reservation)
            backpressure = ProviderBackpressureRecordV2.create(
                work_model_reservation_ref=reservation_ref,
                facade_receipt_ref=facade_receipt_ref,
                kind=kind,
                usage=usage,
                observed_at=now,
                eligible_at=eligible_at,
                audit=_model_audit(
                    audit,
                    (reservation_ref, facade_receipt_ref),
                ),
            )
            connection.execute(
                """
                INSERT INTO provider_backpressure_records (
                    provider_backpressure_record_id,
                    work_model_reservation_id, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    backpressure.provider_backpressure_record_id,
                    reservation.work_model_reservation_id,
                    self.job_store._record_json(backpressure),
                ),
            )
            failure = FailureRecord(
                failure_class=FailureClass.CAPABILITY,
                code="MODEL_PROVIDER_BACKPRESSURE",
                message="model provider requested bounded retry",
                retryable=True,
            )
            lease_event = self.job_store._complete_stage_work_lease(
                lease=lease,
                policy=control_policy,
                holder_ref=holder_ref,
                expected_lease_version=expected_lease_version,
                status=StageRunStatus.RETRYABLE_FAILURE,
                output_refs=(),
                failure=failure,
                checkpoint_ref=None,
                metrics_ref=None,
                audit=audit,
                idempotency_key=_internal_key(
                    idempotency_key,
                    "backpressure-lease-event",
                ),
                connection=connection,
                control_authorizations=frozenset({"MODEL"}),
                metrics_model_profile_ref=self._get_demand(
                    connection,
                    reservation.work_model_demand_ref.object_id,
                ).model_profile_ref,
                metrics_model_usage=usage,
                metrics_telemetry=telemetry,
            )
            model_event = self._release_completed(
                connection,
                reservation=reservation,
                lease_event=lease_event,
                usage=usage,
                audit=audit,
                event_kind=WorkModelEventKindV2.BACKPRESSURED_RELEASED,
                provider_backpressure_ref=provider_backpressure_record_v2_ref(backpressure),
                blocked_until=eligible_at,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_MODEL_EVENT",
                response_id=model_event.work_model_event_id,
                created_at=now,
            )
            return ModelCompletionResult(
                lease_event=lease_event,
                model_event=model_event,
            )

    def complete_stage_combined(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        model_reservation: WorkModelReservationV2,
        resource_reservation: WorkResourceReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        status: StageRunStatus,
        output_refs: tuple[ObjectRef, ...],
        failure: FailureRecord | None,
        checkpoint_ref: ObjectRef | None,
        metrics_ref: ObjectRef | None,
        model_usage: ModelUsageV2,
        resource_usage: ResourceUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> CombinedCompletionResult:
        _require_model_receipt_ref(output_refs)
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(model_reservation),
                "work_resource_reservation_ref": resource_reservation,
                "holder_ref": holder_ref,
                "expected_lease_version": expected_lease_version,
                "status": status,
                "output_refs": output_refs,
                "failure": failure,
                "checkpoint_ref": checkpoint_ref,
                "metrics_ref": metrics_ref,
                "model_usage": model_usage,
                "resource_usage": resource_usage,
                "telemetry": telemetry,
            }
        )
        scope = f"complete-combined-stage:{lease.work_lease_id}"
        resources = ResourceControlService(self.job_store)
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_MODEL_EVENT":
                    raise IdempotencyConflictError("combined completion response type is corrupt")
                model_event = self._get_model_event(connection, prior[1])
                if model_event.work_lease_event_ref is None:
                    raise ImmutableResultError("combined model event is missing lease event")
                lease_event = self.job_store._get_work_lease_event(
                    connection,
                    model_event.work_lease_event_ref.object_id,
                )
                row = connection.execute(
                    """
                    SELECT work_resource_event_id
                    FROM work_resource_events
                    WHERE work_resource_reservation_id = ?
                    ORDER BY reservation_version DESC
                    LIMIT 1
                    """,
                    (resource_reservation.work_resource_reservation_id,),
                ).fetchone()
                if row is None:
                    raise ImmutableResultError("combined completion is missing resource event")
                resource_event = resources._get_resource_event(
                    connection,
                    str(row["work_resource_event_id"]),
                )
                return CombinedCompletionResult(
                    lease_event=lease_event,
                    model_event=model_event,
                    resource_event=resource_event,
                )
            stored_model = self._get_reservation(
                connection,
                model_reservation.work_model_reservation_id,
            )
            if stored_model != model_reservation or model_reservation.work_lease_ref != work_lease_v2_ref(
                lease
            ):
                raise StaleModelReservationError("model reservation is stale")
            model_head = self._get_reservation_head(
                connection,
                model_reservation,
            )
            if model_head.state is not WorkModelReservationStateV2.ACTIVE:
                raise StaleModelReservationError("model reservation is not active")
            resources._require_active_reservation(
                connection,
                lease,
                resource_reservation,
            )
            if (
                model_usage.requests > model_reservation.request_allowance
                or model_usage.charged_tokens > model_reservation.token_allowance
            ):
                raise ModelControlPolicyError("observed model usage exceeds grant")
            exceeded = resource_usage.observed.exceeds(resource_reservation.budget_allowance)
            if exceeded:
                raise ModelControlPolicyError("observed resource usage exceeds combined grant")
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
                idempotency_key=_internal_key(
                    idempotency_key,
                    "combined-lease-event",
                ),
                connection=connection,
                control_authorizations=frozenset({"MODEL", "RESOURCE"}),
                metrics_model_profile_ref=self._get_demand(
                    connection,
                    model_reservation.work_model_demand_ref.object_id,
                ).model_profile_ref,
                metrics_model_usage=model_usage,
                metrics_resource_usage=resource_usage,
                metrics_telemetry=telemetry,
            )
            resource_event = resources._release_completed(
                connection,
                reservation=resource_reservation,
                lease_event=lease_event,
                usage=resource_usage,
                audit=audit,
            )
            model_event = self._release_completed(
                connection,
                reservation=model_reservation,
                lease_event=lease_event,
                usage=model_usage,
                audit=audit,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_MODEL_EVENT",
                response_id=model_event.work_model_event_id,
                created_at=self.job_store._clock(),
            )
            return CombinedCompletionResult(
                lease_event=lease_event,
                model_event=model_event,
                resource_event=resource_event,
            )

    def complete_stage_combined_backpressured(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        model_reservation: WorkModelReservationV2,
        resource_reservation: WorkResourceReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        facade_receipt_ref: ObjectRef,
        kind: ProviderBackpressureKindV2,
        eligible_at: datetime,
        model_usage: ModelUsageV2,
        resource_usage: ResourceUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> CombinedCompletionResult:
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(model_reservation),
                "work_resource_reservation_ref": resource_reservation,
                "holder_ref": holder_ref,
                "expected_lease_version": expected_lease_version,
                "facade_receipt_ref": facade_receipt_ref,
                "kind": kind,
                "eligible_at": eligible_at,
                "model_usage": model_usage,
                "resource_usage": resource_usage,
                "telemetry": telemetry,
            }
        )
        scope = f"complete-combined-backpressure:{lease.work_lease_id}"
        resources = ResourceControlService(self.job_store)
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._load_combined_completion_result(
                    connection,
                    response_type=prior[0],
                    model_event_id=prior[1],
                    resource_reservation=resource_reservation,
                    resources=resources,
                )
            self._require_active_reservation(
                connection,
                lease,
                model_reservation,
            )
            resources._require_active_reservation(
                connection,
                lease,
                resource_reservation,
            )
            if (
                model_usage.requests > model_reservation.request_allowance
                or model_usage.charged_tokens > model_reservation.token_allowance
            ):
                raise ModelControlPolicyError("backpressure usage exceeds grant")
            exceeded = resource_usage.observed.exceeds(resource_reservation.budget_allowance)
            if exceeded:
                raise ModelControlPolicyError("observed resource usage exceeds combined grant")
            now = self.job_store._clock()
            eligible_at = self._normalize_backpressure_eligible_at(
                connection,
                reservation=model_reservation,
                provider_eligible_at=eligible_at,
                now=now,
            )
            reservation_ref = work_model_reservation_v2_ref(model_reservation)
            backpressure = ProviderBackpressureRecordV2.create(
                work_model_reservation_ref=reservation_ref,
                facade_receipt_ref=facade_receipt_ref,
                kind=kind,
                usage=model_usage,
                observed_at=now,
                eligible_at=eligible_at,
                audit=_model_audit(
                    audit,
                    (reservation_ref, facade_receipt_ref),
                ),
            )
            connection.execute(
                """
                INSERT INTO provider_backpressure_records (
                    provider_backpressure_record_id,
                    work_model_reservation_id, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    backpressure.provider_backpressure_record_id,
                    model_reservation.work_model_reservation_id,
                    self.job_store._record_json(backpressure),
                ),
            )
            failure = FailureRecord(
                failure_class=FailureClass.CAPABILITY,
                code="MODEL_PROVIDER_BACKPRESSURE",
                message="model provider requested bounded retry",
                retryable=True,
            )
            lease_event = self.job_store._complete_stage_work_lease(
                lease=lease,
                policy=control_policy,
                holder_ref=holder_ref,
                expected_lease_version=expected_lease_version,
                status=StageRunStatus.RETRYABLE_FAILURE,
                output_refs=(),
                failure=failure,
                checkpoint_ref=None,
                metrics_ref=None,
                audit=audit,
                idempotency_key=_internal_key(
                    idempotency_key,
                    "combined-backpressure-lease-event",
                ),
                connection=connection,
                control_authorizations=frozenset({"MODEL", "RESOURCE"}),
                metrics_model_profile_ref=self._get_demand(
                    connection,
                    model_reservation.work_model_demand_ref.object_id,
                ).model_profile_ref,
                metrics_model_usage=model_usage,
                metrics_resource_usage=resource_usage,
                metrics_telemetry=telemetry,
            )
            resource_event = resources._release_completed(
                connection,
                reservation=resource_reservation,
                lease_event=lease_event,
                usage=resource_usage,
                audit=audit,
            )
            model_event = self._release_completed(
                connection,
                reservation=model_reservation,
                lease_event=lease_event,
                usage=model_usage,
                audit=audit,
                event_kind=WorkModelEventKindV2.BACKPRESSURED_RELEASED,
                provider_backpressure_ref=provider_backpressure_record_v2_ref(backpressure),
                blocked_until=eligible_at,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_MODEL_EVENT",
                response_id=model_event.work_model_event_id,
                created_at=now,
            )
            return CombinedCompletionResult(
                lease_event=lease_event,
                model_event=model_event,
                resource_event=resource_event,
            )

    def complete_usage_overrun(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        reservation: WorkModelReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        facade_receipt_ref: ObjectRef,
        usage: ModelUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
        artifact_slices: tuple[ArtifactMetricSliceV2, ...] = (),
    ) -> ModelCompletionResult:
        _require_model_receipt_ref((facade_receipt_ref,))
        if (
            usage.requests <= reservation.request_allowance
            and usage.charged_tokens <= reservation.token_allowance
        ):
            raise ModelControlPolicyError("usage overrun completion requires observed usage above the grant")
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(reservation),
                "holder_ref": holder_ref,
                "expected_lease_version": expected_lease_version,
                "facade_receipt_ref": facade_receipt_ref,
                "usage": usage,
                "telemetry": telemetry,
                "artifact_slices": artifact_slices,
            }
        )
        scope = f"complete-model-overrun:{reservation.work_model_reservation_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_MODEL_EVENT":
                    raise IdempotencyConflictError("model usage overrun response type is corrupt")
                model_event = self._get_model_event(connection, prior[1])
                if model_event.work_lease_event_ref is None:
                    raise ImmutableResultError("model usage overrun event is missing lease event")
                return ModelCompletionResult(
                    lease_event=self.job_store._get_work_lease_event(
                        connection,
                        model_event.work_lease_event_ref.object_id,
                    ),
                    model_event=model_event,
                )
            self._require_active_reservation(connection, lease, reservation)
            failure = FailureRecord(
                failure_class=FailureClass.POLICY,
                code="MODEL_USAGE_OVERRUN",
                message="model usage exceeded the committed grant",
                retryable=False,
            )
            if lease.stage_run_ref is None:
                lease_event = self.job_store._complete_artifact_group_work_lease(
                    lease=lease,
                    policy=control_policy,
                    holder_ref=holder_ref,
                    expected_lease_version=expected_lease_version,
                    event_kind=WorkLeaseEventKindV2.TERMINAL_FAILURE,
                    result_refs=(facade_receipt_ref,),
                    failure=failure,
                    audit=audit,
                    idempotency_key=_internal_key(
                        idempotency_key,
                        "overrun-lease-event",
                    ),
                    connection=connection,
                    control_authorizations=frozenset({"MODEL"}),
                    metrics_model_profile_ref=self._get_demand(
                        connection,
                        reservation.work_model_demand_ref.object_id,
                    ).model_profile_ref,
                    metrics_model_usage=usage,
                    metrics_telemetry=telemetry,
                    metrics_artifact_slices=artifact_slices,
                )
            else:
                lease_event = self.job_store._complete_stage_work_lease(
                    lease=lease,
                    policy=control_policy,
                    holder_ref=holder_ref,
                    expected_lease_version=expected_lease_version,
                    status=StageRunStatus.TERMINAL_FAILURE,
                    output_refs=(facade_receipt_ref,),
                    failure=failure,
                    checkpoint_ref=None,
                    metrics_ref=None,
                    audit=audit,
                    idempotency_key=_internal_key(
                        idempotency_key,
                        "overrun-lease-event",
                    ),
                    connection=connection,
                    control_authorizations=frozenset({"MODEL"}),
                    metrics_model_profile_ref=self._get_demand(
                        connection,
                        reservation.work_model_demand_ref.object_id,
                    ).model_profile_ref,
                    metrics_model_usage=usage,
                    metrics_telemetry=telemetry,
                    metrics_artifact_slices=artifact_slices,
                )
            model_event = self._release_overrun(
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
                response_type="WORK_MODEL_EVENT",
                response_id=model_event.work_model_event_id,
                created_at=self.job_store._clock(),
            )
            return ModelCompletionResult(
                lease_event=lease_event,
                model_event=model_event,
            )

    def complete_usage_overrun_combined(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        model_reservation: WorkModelReservationV2,
        resource_reservation: WorkResourceReservationV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        facade_receipt_ref: ObjectRef,
        model_usage: ModelUsageV2,
        resource_usage: ResourceUsageV2,
        audit: ContractAudit,
        idempotency_key: str,
        telemetry: ExecutionTelemetryV2 | None = None,
        artifact_slices: tuple[ArtifactMetricSliceV2, ...] = (),
    ) -> CombinedCompletionResult:
        _require_model_receipt_ref((facade_receipt_ref,))
        if (
            model_usage.requests <= model_reservation.request_allowance
            and model_usage.charged_tokens <= model_reservation.token_allowance
        ):
            raise ModelControlPolicyError("usage overrun completion requires observed usage above the grant")
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(model_reservation),
                "work_resource_reservation_ref": resource_reservation,
                "holder_ref": holder_ref,
                "expected_lease_version": expected_lease_version,
                "facade_receipt_ref": facade_receipt_ref,
                "model_usage": model_usage,
                "resource_usage": resource_usage,
                "telemetry": telemetry,
                "artifact_slices": artifact_slices,
            }
        )
        scope = f"complete-combined-overrun:{lease.work_lease_id}"
        resources = ResourceControlService(self.job_store)
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._load_combined_completion_result(
                    connection,
                    response_type=prior[0],
                    model_event_id=prior[1],
                    resource_reservation=resource_reservation,
                    resources=resources,
                )
            self._require_active_reservation(
                connection,
                lease,
                model_reservation,
            )
            resources._require_active_reservation(
                connection,
                lease,
                resource_reservation,
            )
            exceeded = resource_usage.observed.exceeds(resource_reservation.budget_allowance)
            if exceeded:
                raise ModelControlPolicyError("observed resource usage exceeds combined grant")
            failure = FailureRecord(
                failure_class=FailureClass.POLICY,
                code="MODEL_USAGE_OVERRUN",
                message="model usage exceeded the committed grant",
                retryable=False,
            )
            if lease.stage_run_ref is None:
                lease_event = self.job_store._complete_artifact_group_work_lease(
                    lease=lease,
                    policy=control_policy,
                    holder_ref=holder_ref,
                    expected_lease_version=expected_lease_version,
                    event_kind=WorkLeaseEventKindV2.TERMINAL_FAILURE,
                    result_refs=(facade_receipt_ref,),
                    failure=failure,
                    audit=audit,
                    idempotency_key=_internal_key(
                        idempotency_key,
                        "combined-overrun-lease-event",
                    ),
                    connection=connection,
                    control_authorizations=frozenset({"MODEL", "RESOURCE"}),
                    metrics_model_profile_ref=self._get_demand(
                        connection,
                        model_reservation.work_model_demand_ref.object_id,
                    ).model_profile_ref,
                    metrics_model_usage=model_usage,
                    metrics_resource_usage=resource_usage,
                    metrics_telemetry=telemetry,
                    metrics_artifact_slices=artifact_slices,
                )
            else:
                lease_event = self.job_store._complete_stage_work_lease(
                    lease=lease,
                    policy=control_policy,
                    holder_ref=holder_ref,
                    expected_lease_version=expected_lease_version,
                    status=StageRunStatus.TERMINAL_FAILURE,
                    output_refs=(facade_receipt_ref,),
                    failure=failure,
                    checkpoint_ref=None,
                    metrics_ref=None,
                    audit=audit,
                    idempotency_key=_internal_key(
                        idempotency_key,
                        "combined-overrun-lease-event",
                    ),
                    connection=connection,
                    control_authorizations=frozenset({"MODEL", "RESOURCE"}),
                    metrics_model_profile_ref=self._get_demand(
                        connection,
                        model_reservation.work_model_demand_ref.object_id,
                    ).model_profile_ref,
                    metrics_model_usage=model_usage,
                    metrics_resource_usage=resource_usage,
                    metrics_telemetry=telemetry,
                    metrics_artifact_slices=artifact_slices,
                )
            resource_event = resources._release_completed(
                connection,
                reservation=resource_reservation,
                lease_event=lease_event,
                usage=resource_usage,
                audit=audit,
            )
            model_event = self._release_overrun(
                connection,
                reservation=model_reservation,
                lease_event=lease_event,
                usage=model_usage,
                audit=audit,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_MODEL_EVENT",
                response_id=model_event.work_model_event_id,
                created_at=self.job_store._clock(),
            )
            return CombinedCompletionResult(
                lease_event=lease_event,
                model_event=model_event,
                resource_event=resource_event,
            )

    def expire(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        reservation: WorkModelReservationV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ModelExpiryResult:
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(reservation),
            }
        )
        scope = f"expire-model-lease:{reservation.work_model_reservation_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_MODEL_EVENT":
                    raise IdempotencyConflictError("model expiry response type is corrupt")
                model_event = self._get_model_event(connection, prior[1])
                if (
                    model_event.work_lease_event_ref is None
                    or model_event.model_cancellation_request_ref is None
                ):
                    raise ImmutableResultError("model expiry event is missing control refs")
                lease_event = self.job_store._get_work_lease_event(
                    connection,
                    model_event.work_lease_event_ref.object_id,
                )
                cancellation_request = self._get_cancellation_request(
                    connection,
                    model_event.model_cancellation_request_ref.object_id,
                )
                return ModelExpiryResult(
                    lease_event=lease_event,
                    model_event=model_event,
                    cancellation_request=cancellation_request,
                )
            stored = self._get_reservation(
                connection,
                reservation.work_model_reservation_id,
            )
            if stored != reservation or reservation.work_lease_ref != work_lease_v2_ref(lease):
                raise StaleModelReservationError("model reservation is stale")
            if (
                self._get_reservation_head(connection, reservation).state
                is not WorkModelReservationStateV2.ACTIVE
            ):
                raise StaleModelReservationError("model reservation is not active")
            lease_event = self.job_store._expire_work_lease(
                lease=lease,
                policy=control_policy,
                audit=audit,
                idempotency_key=_internal_key(
                    idempotency_key,
                    "expiry-lease-event",
                ),
                connection=connection,
                control_authorizations=frozenset({"MODEL"}),
            )
            event, cancellation_request = self._close_aborted_reservation(
                connection,
                reservation=reservation,
                lease_event=lease_event,
                reason=WorkModelCancellationReasonV2.EXPIRED,
                audit=audit,
            )
            now = self.job_store._clock()
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_MODEL_EVENT",
                response_id=event.work_model_event_id,
                created_at=now,
            )
            return ModelExpiryResult(
                lease_event=lease_event,
                model_event=event,
                cancellation_request=cancellation_request,
            )

    def expire_combined(
        self,
        *,
        lease: WorkLeaseV2,
        control_policy: WorkControlPolicyV2,
        model_reservation: WorkModelReservationV2,
        resource_reservation: WorkResourceReservationV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> CombinedExpiryResult:
        request_sha256 = _request_sha256(
            {
                "work_lease_ref": work_lease_v2_ref(lease),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "work_model_reservation_ref": work_model_reservation_v2_ref(model_reservation),
                "work_resource_reservation_ref": resource_reservation,
            }
        )
        scope = f"expire-combined-lease:{lease.work_lease_id}"
        resources = ResourceControlService(self.job_store)
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_MODEL_EVENT":
                    raise IdempotencyConflictError("combined expiry response type is corrupt")
                model_event = self._get_model_event(connection, prior[1])
                if (
                    model_event.work_lease_event_ref is None
                    or model_event.model_cancellation_request_ref is None
                ):
                    raise ImmutableResultError("combined expiry model event is missing control refs")
                lease_event = self.job_store._get_work_lease_event(
                    connection,
                    model_event.work_lease_event_ref.object_id,
                )
                model_request = self._get_cancellation_request(
                    connection,
                    model_event.model_cancellation_request_ref.object_id,
                )
                resource_row = connection.execute(
                    """
                    SELECT work_resource_event_id
                    FROM work_resource_events
                    WHERE work_resource_reservation_id = ?
                      AND reservation_version = 1
                    """,
                    (resource_reservation.work_resource_reservation_id,),
                ).fetchone()
                if resource_row is None:
                    raise ImmutableResultError("combined expiry is missing initial resource event")
                resource_event = resources._get_resource_event(
                    connection,
                    str(resource_row["work_resource_event_id"]),
                )
                if resource_event.work_lease_event_ref != (work_lease_event_v2_ref(lease_event)):
                    raise ImmutableResultError("combined expiry resource event binds another lease event")
                return CombinedExpiryResult(
                    lease_event=lease_event,
                    model_event=model_event,
                    resource_event=resource_event,
                    model_cancellation_request=model_request,
                    resource_termination_request=(
                        resources._get_termination_request_for_reservation(
                            connection,
                            resource_reservation.work_resource_reservation_id,
                        )
                    ),
                )
            self._require_active_reservation(
                connection,
                lease,
                model_reservation,
            )
            resources._require_active_reservation(
                connection,
                lease,
                resource_reservation,
            )
            lease_event = self.job_store._expire_work_lease(
                lease=lease,
                policy=control_policy,
                audit=audit,
                idempotency_key=_internal_key(
                    idempotency_key,
                    "combined-expiry-lease-event",
                ),
                connection=connection,
                control_authorizations=frozenset({"MODEL", "RESOURCE"}),
            )
            model_event, model_request = self._close_aborted_reservation(
                connection,
                reservation=model_reservation,
                lease_event=lease_event,
                reason=WorkModelCancellationReasonV2.EXPIRED,
                audit=audit,
            )
            resource_event, resource_request = resources._close_aborted_reservation(
                connection,
                reservation=resource_reservation,
                lease_event=lease_event,
                reason=WorkResourceTerminationReasonV2.EXPIRED,
                audit=audit,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_MODEL_EVENT",
                response_id=model_event.work_model_event_id,
                created_at=self.job_store._clock(),
            )
            return CombinedExpiryResult(
                lease_event=lease_event,
                model_event=model_event,
                resource_event=resource_event,
                model_cancellation_request=model_request,
                resource_termination_request=resource_request,
            )

    def cancel_job(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        control_policy: WorkControlPolicyV2,
        job_policy: JobModelPolicyV2,
        reason_code: str,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ModelJobCancellationResult:
        request_sha256 = _request_sha256(
            {
                "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(graph),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "job_model_policy_ref": job_model_policy_v2_ref(job_policy),
                "reason_code": reason_code,
            }
        )
        scope = f"cancel-job-model:{graph.job_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_CANCELLATION":
                    raise IdempotencyConflictError("model cancellation response type is corrupt")
                cancellation = self.job_store._get_work_cancellation(
                    connection,
                    prior[1],
                )
                prior_events, prior_requests = self._list_job_model_closures(
                    connection,
                    graph.job_id,
                    cancellation,
                )
                return ModelJobCancellationResult(
                    cancellation=cancellation,
                    model_events=prior_events,
                    cancellation_requests=prior_requests,
                )
            stored_policy = self._get_job_policy(
                connection,
                job_policy.job_model_policy_id,
            )
            if (
                stored_policy != job_policy
                or job_policy.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph)
            ):
                raise ModelControlPolicyError("Job model policy is not current")
            active_rows = connection.execute(
                """
                SELECT reservations.work_model_reservation_id,
                       reservations.work_lease_id
                FROM work_model_reservations AS reservations
                JOIN work_model_reservation_heads AS heads
                  ON heads.work_model_reservation_id =
                     reservations.work_model_reservation_id
                WHERE reservations.job_id = ? AND heads.state = 'ACTIVE'
                ORDER BY reservations.work_model_reservation_id
                """,
                (graph.job_id,),
            ).fetchall()
            active = tuple(
                (
                    self._get_reservation(
                        connection,
                        str(row["work_model_reservation_id"]),
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
                idempotency_key=_internal_key(
                    idempotency_key,
                    "cancellation",
                ),
                connection=connection,
                control_authorizations=frozenset({"MODEL"}),
            )
            events: list[WorkModelEventV2] = []
            requests: list[WorkModelCancellationRequestV2] = []
            for reservation, lease_id in active:
                lease_event = self.job_store._get_latest_work_lease_event(
                    connection,
                    lease_id,
                )
                event, request = self._close_aborted_reservation(
                    connection,
                    reservation=reservation,
                    lease_event=lease_event,
                    reason=WorkModelCancellationReasonV2.CANCELLED,
                    audit=audit,
                )
                events.append(event)
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
            return ModelJobCancellationResult(
                cancellation=cancellation,
                model_events=tuple(events),
                cancellation_requests=tuple(requests),
            )

    def cancel_job_combined(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        control_policy: WorkControlPolicyV2,
        model_job_policy: JobModelPolicyV2,
        resource_job_policy: JobResourcePolicyV2,
        reason_code: str,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> CombinedJobCancellationResult:
        request_sha256 = _request_sha256(
            {
                "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(graph),
                "work_control_policy_ref": work_control_policy_v2_ref(control_policy),
                "job_model_policy_ref": job_model_policy_v2_ref(model_job_policy),
                "job_resource_policy_ref": job_resource_policy_v2_ref(resource_job_policy),
                "reason_code": reason_code,
            }
        )
        scope = f"cancel-job-combined:{graph.job_id}"
        resources = ResourceControlService(self.job_store)
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_CANCELLATION":
                    raise IdempotencyConflictError("combined cancellation response type is corrupt")
                cancellation = self.job_store._get_work_cancellation(
                    connection,
                    prior[1],
                )
                prior_model_events, prior_model_requests = self._list_job_model_closures(
                    connection,
                    graph.job_id,
                    cancellation,
                )
                prior_resource_events, prior_resource_requests = resources._list_job_resource_closures(
                    connection,
                    graph.job_id,
                    cancellation,
                )
                return CombinedJobCancellationResult(
                    cancellation=cancellation,
                    model_events=prior_model_events,
                    model_cancellation_requests=prior_model_requests,
                    resource_events=prior_resource_events,
                    resource_termination_requests=prior_resource_requests,
                )
            stored_model_policy = self._get_job_policy(
                connection,
                model_job_policy.job_model_policy_id,
            )
            stored_resource_policy = resources._get_job_policy(
                connection,
                resource_job_policy.job_resource_policy_id,
            )
            graph_ref = resolved_job_work_graph_v2_ref(graph)
            if (
                stored_model_policy != model_job_policy
                or model_job_policy.resolved_job_work_graph_ref != graph_ref
            ):
                raise ModelControlPolicyError("Job model policy is not current")
            if (
                stored_resource_policy != resource_job_policy
                or resource_job_policy.resolved_job_work_graph_ref != graph_ref
            ):
                raise ModelControlPolicyError("Job resource policy is not current")
            model_rows = connection.execute(
                """
                SELECT reservations.work_model_reservation_id,
                       reservations.work_lease_id
                FROM work_model_reservations AS reservations
                JOIN work_model_reservation_heads AS heads
                  ON heads.work_model_reservation_id =
                     reservations.work_model_reservation_id
                WHERE reservations.job_id = ? AND heads.state = 'ACTIVE'
                ORDER BY reservations.work_model_reservation_id
                """,
                (graph.job_id,),
            ).fetchall()
            resource_rows = connection.execute(
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
            active_models = tuple(
                (
                    self._get_reservation(
                        connection,
                        str(row["work_model_reservation_id"]),
                    ),
                    str(row["work_lease_id"]),
                )
                for row in model_rows
            )
            active_resources = tuple(
                (
                    resources._get_reservation(
                        connection,
                        str(row["work_resource_reservation_id"]),
                    ),
                    str(row["work_lease_id"]),
                )
                for row in resource_rows
            )
            if {item[1] for item in active_models} != {item[1] for item in active_resources}:
                raise ImmutableResultError("combined active reservations do not share the same leases")
            cancellation = self.job_store._cancel_job_work(
                graph=graph,
                policy=control_policy,
                reason_code=reason_code,
                audit=audit,
                idempotency_key=_internal_key(
                    idempotency_key,
                    "combined-cancellation",
                ),
                connection=connection,
                control_authorizations=frozenset({"MODEL", "RESOURCE"}),
            )
            closed_model_events: list[WorkModelEventV2] = []
            closed_model_requests: list[WorkModelCancellationRequestV2] = []
            for model_reservation, lease_id in active_models:
                lease_event = self.job_store._get_latest_work_lease_event(
                    connection,
                    lease_id,
                )
                model_event, model_request = self._close_aborted_reservation(
                    connection,
                    reservation=model_reservation,
                    lease_event=lease_event,
                    reason=WorkModelCancellationReasonV2.CANCELLED,
                    audit=audit,
                )
                closed_model_events.append(model_event)
                closed_model_requests.append(model_request)
            closed_resource_events: list[WorkResourceEventV2] = []
            closed_resource_requests: list[WorkResourceTerminationRequestV2] = []
            for resource_reservation, lease_id in active_resources:
                lease_event = self.job_store._get_latest_work_lease_event(
                    connection,
                    lease_id,
                )
                resource_event, resource_request = resources._close_aborted_reservation(
                    connection,
                    reservation=resource_reservation,
                    lease_event=lease_event,
                    reason=WorkResourceTerminationReasonV2.CANCELLED,
                    audit=audit,
                )
                closed_resource_events.append(resource_event)
                if resource_request is not None:
                    closed_resource_requests.append(resource_request)
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_CANCELLATION",
                response_id=cancellation.work_cancellation_record_id,
                created_at=self.job_store._clock(),
            )
            return CombinedJobCancellationResult(
                cancellation=cancellation,
                model_events=tuple(closed_model_events),
                model_cancellation_requests=tuple(closed_model_requests),
                resource_events=tuple(closed_resource_events),
                resource_termination_requests=tuple(closed_resource_requests),
            )

    def to_facade_cancellation_request(
        self,
        request: WorkModelCancellationRequestV2,
    ) -> ModelInvocationCancellationRequestV2:
        return ModelInvocationCancellationRequestV2.create(
            reservation_ref=_facade_ref(request.work_model_reservation_ref),
            invocation_handle_ref=_facade_ref(request.invocation_handle_ref),
            fencing_token=request.fencing_token,
            reason=request.reason.value,
            requested_at=request.requested_at,
        )

    def record_cancellation(
        self,
        *,
        request: WorkModelCancellationRequestV2,
        facade_result: ModelInvocationCancellationResultV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ModelCancellationResult:
        request_sha256 = _request_sha256(
            {
                "work_model_cancellation_request_ref": (work_model_cancellation_request_v2_ref(request)),
                "facade_cancellation_result_ref": (model_invocation_cancellation_result_ref(facade_result)),
            }
        )
        scope = f"record-model-cancellation:{request.work_model_cancellation_request_id}"
        with self.job_store._transaction() as connection:
            prior = self.job_store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "WORK_MODEL_CANCELLATION_RESULT":
                    raise IdempotencyConflictError("model cancellation response type is corrupt")
                result = self._get_cancellation_result(connection, prior[1])
                event = self._get_event_for_cancellation_result(
                    connection,
                    result.work_model_cancellation_result_id,
                )
                return ModelCancellationResult(
                    cancellation_result=result,
                    model_event=event,
                )
            stored_request = self._get_cancellation_request(
                connection,
                request.work_model_cancellation_request_id,
            )
            if stored_request != request:
                raise StaleModelReservationError("model cancellation request is stale")
            validate_model_invocation_cancellation_result_identity(facade_result)
            expected_facade_request = self.to_facade_cancellation_request(request)
            if facade_result.cancellation_request_ref != (
                model_invocation_cancellation_request_ref(expected_facade_request)
            ):
                raise ModelControlPolicyError("facade cancellation result does not bind the request")
            reservation = self._get_reservation(
                connection,
                request.work_model_reservation_ref.object_id,
            )
            head = self._get_reservation_head(connection, reservation)
            if head.state is not WorkModelReservationStateV2.RELEASE_PENDING_ACK:
                raise StaleModelReservationError("model reservation is not pending acknowledgement")
            factory_outcome = WorkModelCancellationOutcomeV2(facade_result.outcome.value)
            request_ref = work_model_cancellation_request_v2_ref(request)
            facade_result_ref = _factory_ref(model_invocation_cancellation_result_ref(facade_result))
            result = WorkModelCancellationResultV2.create(
                cancellation_request_ref=request_ref,
                facade_cancellation_result_ref=facade_result_ref,
                outcome=factory_outcome,
                completed_at=facade_result.completed_at,
                audit=_model_audit(
                    audit,
                    (request_ref, facade_result_ref),
                ),
            )
            connection.execute(
                """
                INSERT INTO work_model_cancellation_results (
                    work_model_cancellation_result_id,
                    work_model_cancellation_request_id, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    result.work_model_cancellation_result_id,
                    request.work_model_cancellation_request_id,
                    self.job_store._record_json(result),
                ),
            )
            demand = self._get_demand(
                connection,
                reservation.work_model_demand_ref.object_id,
            )
            pool_policy = self._get_pool_policy(
                connection,
                demand.model_rate_pool_policy_ref.object_id,
            )
            job_policy = self._get_job_policy(
                connection,
                demand.job_model_policy_ref.object_id,
            )
            now = self.job_store._clock()
            pool_head = _refill_pool_head(
                self._get_pool_head(connection, pool_policy),
                pool_policy,
                now,
            )
            job_head = self._get_job_head(connection, job_policy)
            released = factory_outcome in {
                WorkModelCancellationOutcomeV2.CANCELLED,
                WorkModelCancellationOutcomeV2.ALREADY_FINISHED,
            }
            result_ref = work_model_cancellation_result_v2_ref(result)
            reservation_ref = work_model_reservation_v2_ref(reservation)
            event = WorkModelEventV2.create(
                work_model_reservation_ref=reservation_ref,
                work_lease_event_ref=request.work_lease_event_ref,
                event_kind=(
                    WorkModelEventKindV2.ACKNOWLEDGED_RELEASED
                    if released
                    else WorkModelEventKindV2.QUARANTINED
                ),
                reservation_version=2,
                bucket_version=pool_head.bucket_version + 1,
                effective_at=now,
                usage=_zero_usage(),
                refunded_request_allowance=0,
                refunded_token_allowance=0,
                released_concurrency=(reservation.concurrency_units if released else 0),
                provider_backpressure_ref=None,
                model_cancellation_request_ref=None,
                model_cancellation_result_ref=result_ref,
                blocked_until=None,
                audit=_model_audit(
                    audit,
                    (
                        reservation_ref,
                        request.work_lease_event_ref,
                        result_ref,
                    ),
                ),
            )
            self._insert_event(
                connection,
                event,
                pool_policy.model_rate_pool_policy_id,
            )
            self._update_pool_head(
                connection,
                pool_head.model_copy(
                    update={
                        "active_concurrency": (
                            pool_head.active_concurrency - (reservation.concurrency_units if released else 0)
                        ),
                        "bucket_version": event.bucket_version,
                        "row_version": pool_head.row_version + 1,
                    }
                ),
                expected_version=pool_head.row_version,
            )
            self._update_job_head(
                connection,
                job_head.model_copy(
                    update={
                        "active_concurrency": (
                            job_head.active_concurrency - (reservation.concurrency_units if released else 0)
                        ),
                        "row_version": job_head.row_version + 1,
                    }
                ),
                expected_version=job_head.row_version,
            )
            self._update_reservation_head(
                connection,
                WorkModelReservationHeadRecord(
                    work_model_reservation_id=(reservation.work_model_reservation_id),
                    state=(
                        WorkModelReservationStateV2.RELEASED
                        if released
                        else WorkModelReservationStateV2.QUARANTINED
                    ),
                    reservation_version=2,
                ),
                expected_version=1,
            )
            self.job_store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="WORK_MODEL_CANCELLATION_RESULT",
                response_id=result.work_model_cancellation_result_id,
                created_at=now,
            )
            return ModelCancellationResult(
                cancellation_result=result,
                model_event=event,
            )

    def get_pool_head(self, model_rate_pool_policy_id: str) -> ModelRatePoolHeadRecord:
        with self.job_store._connect() as connection:
            policy = self._get_pool_policy(connection, model_rate_pool_policy_id)
            return self._get_pool_head(connection, policy)

    def get_job_policy(self, job_id: str) -> JobModelPolicyV2:
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
                SELECT job_model_policy_id,
                       resolved_job_work_graph_id,
                       job_id
                FROM job_model_policies
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchall()
            if not rows:
                raise RecordNotFoundError(f"JobModelPolicyV2 not found: {job_id}")
            if len(rows) != 1:
                raise ImmutableResultError("Job has multiple model policies")
            row = rows[0]
            policy = self._get_job_policy(
                connection,
                str(row["job_model_policy_id"]),
            )
            if (
                str(row["resolved_job_work_graph_id"]) != policy.resolved_job_work_graph_ref.object_id
                or str(row["job_id"]) != policy.dataset_job_spec_ref.object_id
            ):
                raise ImmutableResultError("stored Job model policy columns are inconsistent")
            graph = self.job_store._get_job_work_graph_by_id(
                connection,
                policy.resolved_job_work_graph_ref.object_id,
            )
            spec = self.job_store._get_job_spec(
                connection,
                job_id,
            )
            if (
                resolved_job_work_graph_v2_ref(graph) != policy.resolved_job_work_graph_ref
                or dataset_job_spec_v2_ref(spec) != policy.dataset_job_spec_ref
            ):
                raise ImmutableResultError("stored Job model policy binding is stale")
            for pool_ref in policy.model_rate_pool_policy_refs:
                pool = self._get_pool_policy(
                    connection,
                    pool_ref.object_id,
                )
                if model_rate_pool_policy_v2_ref(pool) != pool_ref:
                    raise ImmutableResultError("stored Job model pool binding is stale")
            return policy

    def get_job_head(self, job_model_policy_id: str) -> JobModelHeadRecord:
        with self.job_store._connect() as connection:
            policy = self._get_job_policy(connection, job_model_policy_id)
            return self._get_job_head(connection, policy)

    def get_reservation_head(
        self,
        work_model_reservation_id: str,
    ) -> WorkModelReservationHeadRecord:
        with self.job_store._connect() as connection:
            reservation = self._get_reservation(
                connection,
                work_model_reservation_id,
            )
            return self._get_reservation_head(connection, reservation)

    def _prepare_admission_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        job_policy: JobModelPolicyV2,
        demand: WorkModelDemandV2,
        intended_attempt: int,
        audit: ContractAudit,
    ) -> tuple[
        ModelAdmissionDecisionV2,
        ModelRatePoolPolicyV2,
        ModelRatePoolHeadRecord,
        JobModelHeadRecord,
    ]:
        self._validate_acquisition_sources(
            connection,
            graph=graph,
            work_unit=work_unit,
            job_policy=job_policy,
            demand=demand,
        )
        if demand.attempt != intended_attempt:
            raise ModelControlPolicyError("model demand attempt does not match dispatchable attempt")
        self._insert_or_validate_demand(connection, demand)
        if connection.execute(
            """
            SELECT 1
            FROM model_admission_decisions
            WHERE work_model_demand_id = ?
              AND outcome IN ('BUDGET_EXHAUSTED', 'UNSATISFIABLE_DEMAND')
            LIMIT 1
            """,
            (demand.work_model_demand_id,),
        ).fetchone():
            raise ModelControlPolicyError("model demand already has a terminal admission decision")
        pool_policy = self._get_pool_policy(
            connection,
            demand.model_rate_pool_policy_ref.object_id,
        )
        stored_pool_head = self._get_pool_head(connection, pool_policy)
        job_head = self._get_job_head(connection, job_policy)
        now = self.job_store._clock()
        effective_pool_head = _refill_pool_head(
            stored_pool_head,
            pool_policy,
            now,
        )
        outcome, constrained, eligible_at = _classify(
            pool_policy=pool_policy,
            job_policy=job_policy,
            pool_head=effective_pool_head,
            job_head=job_head,
            demand=demand,
            now=now,
        )
        policy_ref = job_model_policy_v2_ref(job_policy)
        demand_ref = work_model_demand_v2_ref(demand)
        decision = ModelAdmissionDecisionV2.create(
            job_model_policy_ref=policy_ref,
            work_model_demand_ref=demand_ref,
            outcome=outcome,
            constrained_dimensions=constrained,
            eligible_at=eligible_at,
            pool_head_version_before=stored_pool_head.row_version,
            job_head_version_before=job_head.row_version,
            decided_at=now,
            audit=_model_audit(audit, (policy_ref, demand_ref)),
        )
        return decision, pool_policy, effective_pool_head, job_head

    def _insert_admission_decision_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        decision: ModelAdmissionDecisionV2,
        demand: WorkModelDemandV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO model_admission_decisions (
                model_admission_decision_id, work_model_demand_id,
                outcome, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                decision.model_admission_decision_id,
                demand.work_model_demand_id,
                decision.outcome,
                self.job_store._record_json(decision),
            ),
        )

    def _load_combined_admission_result(
        self,
        connection: sqlite3.Connection,
        *,
        response_type: str,
        response_id: str,
        resources: ResourceControlService,
    ) -> CombinedAdmissionResult:
        if response_type == "COMBINED_RESOURCE_ADMISSION":
            resource = resources._load_admission_result(connection, response_id)
            return CombinedAdmissionResult(resource_decision=resource.decision)
        if response_type != "COMBINED_MODEL_ADMISSION":
            raise IdempotencyConflictError("combined admission response type is corrupt")
        model = self._load_admission_result(connection, response_id)
        if model.lease is None:
            return CombinedAdmissionResult(model_decision=model.decision)
        row = connection.execute(
            """
            SELECT resource_admission_decision_id
            FROM work_resource_reservations
            WHERE work_lease_id = ?
            """,
            (model.lease.work_lease_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("combined model admission is missing resource reservation")
        resource = resources._load_admission_result(
            connection,
            str(row["resource_admission_decision_id"]),
        )
        return CombinedAdmissionResult(
            model_decision=model.decision,
            resource_decision=resource.decision,
            lease=model.lease,
            model_reservation=model.reservation,
            resource_reservation=resource.reservation,
        )

    def _load_combined_completion_result(
        self,
        connection: sqlite3.Connection,
        *,
        response_type: str,
        model_event_id: str,
        resource_reservation: WorkResourceReservationV2,
        resources: ResourceControlService,
    ) -> CombinedCompletionResult:
        if response_type != "WORK_MODEL_EVENT":
            raise IdempotencyConflictError("combined completion response type is corrupt")
        model_event = self._get_model_event(connection, model_event_id)
        if model_event.work_lease_event_ref is None:
            raise ImmutableResultError("combined model event is missing lease event")
        lease_event = self.job_store._get_work_lease_event(
            connection,
            model_event.work_lease_event_ref.object_id,
        )
        resource_event = resources._get_latest_resource_event(
            connection,
            resource_reservation.work_resource_reservation_id,
        )
        if resource_event.work_lease_event_ref != work_lease_event_v2_ref(lease_event):
            raise ImmutableResultError("combined resource event binds another lease event")
        return CombinedCompletionResult(
            lease_event=lease_event,
            model_event=model_event,
            resource_event=resource_event,
        )

    def _insert_admitted_reservation(
        self,
        connection: sqlite3.Connection,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        holder_ref: ObjectRef,
        demand: WorkModelDemandV2,
        decision: ModelAdmissionDecisionV2,
        lease: WorkLeaseV2,
        pool_policy: ModelRatePoolPolicyV2,
        pool_head: ModelRatePoolHeadRecord,
        job_head: JobModelHeadRecord,
        now: datetime,
        audit: ContractAudit,
    ) -> WorkModelReservationV2:
        demand_ref = work_model_demand_v2_ref(demand)
        decision_ref = model_admission_decision_v2_ref(decision)
        lease_ref = work_lease_v2_ref(lease)
        handle_ref = _invocation_handle_ref(lease)
        reservation = WorkModelReservationV2.create(
            model_admission_decision_ref=decision_ref,
            work_model_demand_ref=demand_ref,
            work_lease_ref=lease_ref,
            work_dispatch_decision_ref=lease.work_dispatch_decision_ref,
            holder_ref=holder_ref,
            fencing_token=lease.fencing_token,
            invocation_handle_ref=handle_ref,
            request_allowance=demand.request_allowance,
            token_allowance=demand.token_allowance,
            concurrency_units=demand.concurrency_units,
            acquired_at=lease.acquired_at,
            audit=_model_audit(
                audit,
                (
                    decision_ref,
                    demand_ref,
                    lease_ref,
                    lease.work_dispatch_decision_ref,
                    holder_ref,
                    handle_ref,
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO work_model_reservations (
                work_model_reservation_id, model_admission_decision_id,
                work_model_demand_id, work_lease_id, job_id,
                resolved_work_unit_id, model_rate_pool_policy_id, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reservation.work_model_reservation_id,
                decision.model_admission_decision_id,
                demand.work_model_demand_id,
                lease.work_lease_id,
                graph.job_id,
                work_unit.resolved_work_unit_id,
                pool_policy.model_rate_pool_policy_id,
                self.job_store._record_json(reservation),
            ),
        )
        reservation_ref = work_model_reservation_v2_ref(reservation)
        event = WorkModelEventV2.create(
            work_model_reservation_ref=reservation_ref,
            work_lease_event_ref=None,
            event_kind=WorkModelEventKindV2.ADMITTED,
            reservation_version=0,
            bucket_version=pool_head.bucket_version + 1,
            effective_at=now,
            usage=_zero_usage(),
            refunded_request_allowance=0,
            refunded_token_allowance=0,
            released_concurrency=0,
            provider_backpressure_ref=None,
            model_cancellation_request_ref=None,
            model_cancellation_result_ref=None,
            blocked_until=None,
            audit=_model_audit(audit, (reservation_ref,)),
        )
        self._insert_event(
            connection,
            event,
            pool_policy.model_rate_pool_policy_id,
        )
        reservation_head = WorkModelReservationHeadRecord(
            work_model_reservation_id=reservation.work_model_reservation_id,
            state=WorkModelReservationStateV2.ACTIVE,
            reservation_version=0,
        )
        connection.execute(
            """
            INSERT INTO work_model_reservation_heads (
                work_model_reservation_id, state, reservation_version, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                reservation.work_model_reservation_id,
                reservation_head.state,
                reservation_head.reservation_version,
                self.job_store._record_json(reservation_head),
            ),
        )
        request_debit = demand.request_allowance * RATE_CREDITS_PER_UNIT
        token_debit = demand.token_allowance * RATE_CREDITS_PER_UNIT
        new_pool_head = pool_head.model_copy(
            update={
                "request_credits": pool_head.request_credits - request_debit,
                "token_credits": pool_head.token_credits - token_debit,
                "active_concurrency": (pool_head.active_concurrency + demand.concurrency_units),
                "effective_at": now,
                "bucket_version": event.bucket_version,
                "row_version": pool_head.row_version + 1,
            }
        )
        self._update_pool_head(
            connection,
            new_pool_head,
            expected_version=pool_head.row_version,
        )
        new_job_head = job_head.model_copy(
            update={
                "active_concurrency": (job_head.active_concurrency + demand.concurrency_units),
                "reserved_requests": (job_head.reserved_requests + demand.request_allowance),
                "reserved_tokens": job_head.reserved_tokens + demand.token_allowance,
                "row_version": job_head.row_version + 1,
            }
        )
        self._update_job_head(
            connection,
            new_job_head,
            expected_version=job_head.row_version,
        )
        return reservation

    def _require_active_reservation(
        self,
        connection: sqlite3.Connection,
        lease: WorkLeaseV2,
        reservation: WorkModelReservationV2,
    ) -> None:
        stored = self._get_reservation(
            connection,
            reservation.work_model_reservation_id,
        )
        if stored != reservation or reservation.work_lease_ref != work_lease_v2_ref(lease):
            raise StaleModelReservationError("model reservation is stale")
        head = self._get_reservation_head(connection, reservation)
        if head.state is not WorkModelReservationStateV2.ACTIVE:
            raise StaleModelReservationError("model reservation is not active")

    def _normalize_backpressure_eligible_at(
        self,
        connection: sqlite3.Connection,
        *,
        reservation: WorkModelReservationV2,
        provider_eligible_at: datetime,
        now: datetime,
    ) -> datetime:
        if provider_eligible_at <= now:
            raise ModelControlPolicyError("provider backpressure eligibility must be in the future")
        demand = self._get_demand(
            connection,
            reservation.work_model_demand_ref.object_id,
        )
        pool_policy = self._get_pool_policy(
            connection,
            demand.model_rate_pool_policy_ref.object_id,
        )
        return max(
            provider_eligible_at,
            now
            + timedelta(
                seconds=pool_policy.minimum_backpressure_seconds,
            ),
        )

    def _release_completed(
        self,
        connection: sqlite3.Connection,
        *,
        reservation: WorkModelReservationV2,
        lease_event: WorkLeaseEventV2,
        usage: ModelUsageV2,
        audit: ContractAudit,
        event_kind: WorkModelEventKindV2 = WorkModelEventKindV2.COMPLETED_RELEASED,
        provider_backpressure_ref: ObjectRef | None = None,
        blocked_until: datetime | None = None,
    ) -> WorkModelEventV2:
        demand = self._get_demand(
            connection,
            reservation.work_model_demand_ref.object_id,
        )
        pool_policy = self._get_pool_policy(
            connection,
            demand.model_rate_pool_policy_ref.object_id,
        )
        job_policy = self._get_job_policy(
            connection,
            demand.job_model_policy_ref.object_id,
        )
        pool_head = _refill_pool_head(
            self._get_pool_head(connection, pool_policy),
            pool_policy,
            self.job_store._clock(),
        )
        job_head = self._get_job_head(connection, job_policy)
        now = pool_head.effective_at
        reservation_ref = work_model_reservation_v2_ref(reservation)
        lease_event_ref = work_lease_event_v2_ref(lease_event)
        request_refund = reservation.request_allowance - usage.requests
        token_refund = reservation.token_allowance - usage.charged_tokens
        event = WorkModelEventV2.create(
            work_model_reservation_ref=reservation_ref,
            work_lease_event_ref=lease_event_ref,
            event_kind=event_kind,
            reservation_version=1,
            bucket_version=pool_head.bucket_version + 1,
            effective_at=now,
            usage=usage,
            refunded_request_allowance=request_refund,
            refunded_token_allowance=token_refund,
            released_concurrency=reservation.concurrency_units,
            provider_backpressure_ref=provider_backpressure_ref,
            model_cancellation_request_ref=None,
            model_cancellation_result_ref=None,
            blocked_until=blocked_until,
            audit=_model_audit(
                audit,
                (
                    reservation_ref,
                    lease_event_ref,
                    *((provider_backpressure_ref,) if provider_backpressure_ref else ()),
                ),
            ),
        )
        self._insert_event(
            connection,
            event,
            pool_policy.model_rate_pool_policy_id,
        )
        new_pool_head = pool_head.model_copy(
            update={
                "request_credits": min(
                    pool_policy.request_burst * RATE_CREDITS_PER_UNIT,
                    pool_head.request_credits + request_refund * RATE_CREDITS_PER_UNIT,
                ),
                "token_credits": min(
                    pool_policy.token_burst * RATE_CREDITS_PER_UNIT,
                    pool_head.token_credits + token_refund * RATE_CREDITS_PER_UNIT,
                ),
                "active_concurrency": (pool_head.active_concurrency - reservation.concurrency_units),
                "blocked_until": (
                    blocked_until
                    if blocked_until is not None
                    and (pool_head.blocked_until is None or blocked_until > pool_head.blocked_until)
                    else pool_head.blocked_until
                ),
                "bucket_version": event.bucket_version,
                "row_version": pool_head.row_version + 1,
            }
        )
        self._update_pool_head(
            connection,
            new_pool_head,
            expected_version=pool_head.row_version,
        )
        new_job_head = job_head.model_copy(
            update={
                "active_concurrency": (job_head.active_concurrency - reservation.concurrency_units),
                "reserved_requests": (job_head.reserved_requests - reservation.request_allowance),
                "reserved_tokens": (job_head.reserved_tokens - reservation.token_allowance),
                "consumed_requests": job_head.consumed_requests + usage.requests,
                "consumed_tokens": job_head.consumed_tokens + usage.charged_tokens,
                "row_version": job_head.row_version + 1,
            }
        )
        self._update_job_head(
            connection,
            new_job_head,
            expected_version=job_head.row_version,
        )
        reservation_head = WorkModelReservationHeadRecord(
            work_model_reservation_id=reservation.work_model_reservation_id,
            state=WorkModelReservationStateV2.RELEASED,
            reservation_version=1,
        )
        self._update_reservation_head(
            connection,
            reservation_head,
            expected_version=0,
        )
        return event

    def _release_overrun(
        self,
        connection: sqlite3.Connection,
        *,
        reservation: WorkModelReservationV2,
        lease_event: WorkLeaseEventV2,
        usage: ModelUsageV2,
        audit: ContractAudit,
    ) -> WorkModelEventV2:
        demand = self._get_demand(
            connection,
            reservation.work_model_demand_ref.object_id,
        )
        pool_policy = self._get_pool_policy(
            connection,
            demand.model_rate_pool_policy_ref.object_id,
        )
        job_policy = self._get_job_policy(
            connection,
            demand.job_model_policy_ref.object_id,
        )
        pool_head = _refill_pool_head(
            self._get_pool_head(connection, pool_policy),
            pool_policy,
            self.job_store._clock(),
        )
        job_head = self._get_job_head(connection, job_policy)
        reservation_ref = work_model_reservation_v2_ref(reservation)
        lease_event_ref = work_lease_event_v2_ref(lease_event)
        event = WorkModelEventV2.create(
            work_model_reservation_ref=reservation_ref,
            work_lease_event_ref=lease_event_ref,
            event_kind=WorkModelEventKindV2.USAGE_OVERRUN_RELEASED,
            reservation_version=1,
            bucket_version=pool_head.bucket_version + 1,
            effective_at=pool_head.effective_at,
            usage=usage,
            refunded_request_allowance=0,
            refunded_token_allowance=0,
            released_concurrency=reservation.concurrency_units,
            provider_backpressure_ref=None,
            model_cancellation_request_ref=None,
            model_cancellation_result_ref=None,
            blocked_until=None,
            audit=_model_audit(
                audit,
                (reservation_ref, lease_event_ref),
            ),
        )
        self._insert_event(
            connection,
            event,
            pool_policy.model_rate_pool_policy_id,
        )
        self._update_pool_head(
            connection,
            pool_head.model_copy(
                update={
                    "request_credits": (
                        pool_head.request_credits
                        - max(
                            usage.requests - reservation.request_allowance,
                            0,
                        )
                        * RATE_CREDITS_PER_UNIT
                    ),
                    "token_credits": (
                        pool_head.token_credits
                        - max(
                            usage.charged_tokens - reservation.token_allowance,
                            0,
                        )
                        * RATE_CREDITS_PER_UNIT
                    ),
                    "active_concurrency": (pool_head.active_concurrency - reservation.concurrency_units),
                    "bucket_version": event.bucket_version,
                    "row_version": pool_head.row_version + 1,
                }
            ),
            expected_version=pool_head.row_version,
        )
        self._update_job_head(
            connection,
            job_head.model_copy(
                update={
                    "active_concurrency": (job_head.active_concurrency - reservation.concurrency_units),
                    "reserved_requests": (job_head.reserved_requests - reservation.request_allowance),
                    "reserved_tokens": (job_head.reserved_tokens - reservation.token_allowance),
                    "consumed_requests": (job_head.consumed_requests + usage.requests),
                    "consumed_tokens": (job_head.consumed_tokens + usage.charged_tokens),
                    "row_version": job_head.row_version + 1,
                }
            ),
            expected_version=job_head.row_version,
        )
        self._update_reservation_head(
            connection,
            WorkModelReservationHeadRecord(
                work_model_reservation_id=(reservation.work_model_reservation_id),
                state=WorkModelReservationStateV2.RELEASED,
                reservation_version=1,
            ),
            expected_version=0,
        )
        return event

    def _close_aborted_reservation(
        self,
        connection: sqlite3.Connection,
        *,
        reservation: WorkModelReservationV2,
        lease_event: WorkLeaseEventV2,
        reason: WorkModelCancellationReasonV2,
        audit: ContractAudit,
    ) -> tuple[WorkModelEventV2, WorkModelCancellationRequestV2]:
        head = self._get_reservation_head(connection, reservation)
        if head.state is not WorkModelReservationStateV2.ACTIVE:
            raise StaleModelReservationError("model reservation is not active")
        now = self.job_store._clock()
        reservation_ref = work_model_reservation_v2_ref(reservation)
        lease_event_ref = work_lease_event_v2_ref(lease_event)
        cancellation_request = WorkModelCancellationRequestV2.create(
            work_model_reservation_ref=reservation_ref,
            work_lease_event_ref=lease_event_ref,
            invocation_handle_ref=reservation.invocation_handle_ref,
            fencing_token=reservation.fencing_token,
            reason=reason,
            requested_at=now,
            audit=_model_audit(
                audit,
                (
                    reservation_ref,
                    lease_event_ref,
                    reservation.invocation_handle_ref,
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO work_model_cancellation_requests (
                work_model_cancellation_request_id,
                work_model_reservation_id, record_json
            ) VALUES (?, ?, ?)
            """,
            (
                cancellation_request.work_model_cancellation_request_id,
                reservation.work_model_reservation_id,
                self.job_store._record_json(cancellation_request),
            ),
        )
        demand = self._get_demand(
            connection,
            reservation.work_model_demand_ref.object_id,
        )
        pool_policy = self._get_pool_policy(
            connection,
            demand.model_rate_pool_policy_ref.object_id,
        )
        job_policy = self._get_job_policy(
            connection,
            demand.job_model_policy_ref.object_id,
        )
        pool_head = _refill_pool_head(
            self._get_pool_head(connection, pool_policy),
            pool_policy,
            now,
        )
        job_head = self._get_job_head(connection, job_policy)
        cancellation_ref = work_model_cancellation_request_v2_ref(cancellation_request)
        usage = ModelUsageV2.conservative(
            requests=reservation.request_allowance,
            tokens=reservation.token_allowance,
        )
        event = WorkModelEventV2.create(
            work_model_reservation_ref=reservation_ref,
            work_lease_event_ref=lease_event_ref,
            event_kind=WorkModelEventKindV2.RELEASE_PENDING_ACK,
            reservation_version=1,
            bucket_version=pool_head.bucket_version + 1,
            effective_at=now,
            usage=usage,
            refunded_request_allowance=0,
            refunded_token_allowance=0,
            released_concurrency=0,
            provider_backpressure_ref=None,
            model_cancellation_request_ref=cancellation_ref,
            model_cancellation_result_ref=None,
            blocked_until=None,
            audit=_model_audit(
                audit,
                (reservation_ref, lease_event_ref, cancellation_ref),
            ),
        )
        self._insert_event(
            connection,
            event,
            pool_policy.model_rate_pool_policy_id,
        )
        self._update_pool_head(
            connection,
            pool_head.model_copy(
                update={
                    "bucket_version": event.bucket_version,
                    "row_version": pool_head.row_version + 1,
                }
            ),
            expected_version=pool_head.row_version,
        )
        self._update_job_head(
            connection,
            job_head.model_copy(
                update={
                    "reserved_requests": (job_head.reserved_requests - reservation.request_allowance),
                    "reserved_tokens": (job_head.reserved_tokens - reservation.token_allowance),
                    "consumed_requests": (job_head.consumed_requests + usage.requests),
                    "consumed_tokens": (job_head.consumed_tokens + usage.charged_tokens),
                    "row_version": job_head.row_version + 1,
                }
            ),
            expected_version=job_head.row_version,
        )
        self._update_reservation_head(
            connection,
            WorkModelReservationHeadRecord(
                work_model_reservation_id=(reservation.work_model_reservation_id),
                state=WorkModelReservationStateV2.RELEASE_PENDING_ACK,
                reservation_version=1,
            ),
            expected_version=0,
        )
        return event, cancellation_request

    def _validate_acquisition_sources(
        self,
        connection: sqlite3.Connection,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        job_policy: JobModelPolicyV2,
        demand: WorkModelDemandV2,
    ) -> None:
        stored_graph = self.job_store._get_job_work_graph_by_id(
            connection,
            graph.resolved_job_work_graph_id,
        )
        if stored_graph != graph:
            raise ModelControlPolicyError("model admission graph is not current")
        stored_policy = self._get_job_policy(
            connection,
            job_policy.job_model_policy_id,
        )
        if stored_policy != job_policy:
            raise ModelControlPolicyError("Job model policy is not current")
        stored_demand_policy = demand.job_model_policy_ref
        if stored_demand_policy != job_model_policy_v2_ref(job_policy):
            raise ModelControlPolicyError("model demand Job policy is stale")
        pool = self._get_pool_policy(
            connection,
            demand.model_rate_pool_policy_ref.object_id,
        )
        if (
            demand.model_rate_pool_policy_ref not in job_policy.model_rate_pool_policy_refs
            or demand.model_profile_ref not in job_policy.model_profile_refs
            or demand.model_profile_ref not in pool.allowed_model_profile_refs
        ):
            raise ModelControlPolicyError("model demand pool or profile is stale")
        graph_owns_unit = work_unit in graph.work_units
        fanout_unit = work_unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP and work_unit.job_id == graph.job_id
        if (
            not (graph_owns_unit or fanout_unit)
            or demand.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph)
            or demand.work_unit_ref != resolved_work_unit_v2_ref(work_unit)
        ):
            raise ModelControlPolicyError("model demand work binding is stale")

    def _insert_or_validate_demand(
        self,
        connection: sqlite3.Connection,
        demand: WorkModelDemandV2,
    ) -> None:
        row = connection.execute(
            """
            SELECT record_json
            FROM work_model_demands
            WHERE work_model_demand_id = ?
            """,
            (demand.work_model_demand_id,),
        ).fetchone()
        if row is not None:
            stored = self.job_store._load_record(row, WorkModelDemandV2)
            if stored != demand:
                raise ImmutableResultError("stored model demand differs from request")
            return
        conflict = connection.execute(
            """
            SELECT work_model_demand_id
            FROM work_model_demands
            WHERE resolved_work_unit_id = ? AND attempt = ?
            """,
            (demand.work_unit_ref.object_id, demand.attempt),
        ).fetchone()
        if conflict is not None:
            raise ImmutableResultError("work unit attempt already has another immutable model demand")
        connection.execute(
            """
            INSERT INTO work_model_demands (
                work_model_demand_id, resolved_work_unit_id, attempt, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                demand.work_model_demand_id,
                demand.work_unit_ref.object_id,
                demand.attempt,
                self.job_store._record_json(demand),
            ),
        )

    def _load_admission_result(
        self,
        connection: sqlite3.Connection,
        decision_id: str,
    ) -> ModelAdmissionResult:
        decision = self.job_store._get_record(
            connection,
            "model_admission_decisions",
            "model_admission_decision_id",
            decision_id,
            ModelAdmissionDecisionV2,
        )
        row = connection.execute(
            """
            SELECT work_model_reservation_id, work_lease_id
            FROM work_model_reservations
            WHERE model_admission_decision_id = ?
            """,
            (decision_id,),
        ).fetchone()
        if row is None:
            return ModelAdmissionResult(decision=decision)
        reservation = self._get_reservation(
            connection,
            str(row["work_model_reservation_id"]),
        )
        lease = self.job_store._get_work_lease(connection, str(row["work_lease_id"]))
        return ModelAdmissionResult(
            decision=decision,
            lease=lease,
            reservation=reservation,
        )

    def _get_pool_policy(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> ModelRatePoolPolicyV2:
        policy = self.job_store._get_record(
            connection,
            "model_rate_pool_policies",
            "model_rate_pool_policy_id",
            policy_id,
            ModelRatePoolPolicyV2,
        )
        validate_model_rate_pool_policy_v2_identity(policy)
        row = connection.execute(
            """
            SELECT provider_bucket_id
            FROM model_rate_pool_policies
            WHERE model_rate_pool_policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if row is None or str(row["provider_bucket_id"]) != (policy.provider_bucket_ref.object_id):
            raise ImmutableResultError("stored model rate pool policy columns are inconsistent")
        return policy

    def _get_job_policy(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> JobModelPolicyV2:
        policy = self.job_store._get_record(
            connection,
            "job_model_policies",
            "job_model_policy_id",
            policy_id,
            JobModelPolicyV2,
        )
        validate_job_model_policy_v2_identity(policy)
        return policy

    def _get_demand(
        self,
        connection: sqlite3.Connection,
        demand_id: str,
    ) -> WorkModelDemandV2:
        demand = self.job_store._get_record(
            connection,
            "work_model_demands",
            "work_model_demand_id",
            demand_id,
            WorkModelDemandV2,
        )
        if work_model_demand_v2_ref(demand).object_id != demand_id:
            raise ImmutableResultError("stored model demand identity is inconsistent")
        return demand

    def _get_reservation(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> WorkModelReservationV2:
        reservation = self.job_store._get_record(
            connection,
            "work_model_reservations",
            "work_model_reservation_id",
            reservation_id,
            WorkModelReservationV2,
        )
        row = connection.execute(
            """
            SELECT model_admission_decision_id, work_model_demand_id,
                   work_lease_id
            FROM work_model_reservations
            WHERE work_model_reservation_id = ?
            """,
            (reservation_id,),
        ).fetchone()
        if row is None or (
            str(row["model_admission_decision_id"]) != reservation.model_admission_decision_ref.object_id
            or str(row["work_model_demand_id"]) != reservation.work_model_demand_ref.object_id
            or str(row["work_lease_id"]) != reservation.work_lease_ref.object_id
        ):
            raise ImmutableResultError("stored model reservation columns are inconsistent")
        return reservation

    def _get_model_event(
        self,
        connection: sqlite3.Connection,
        event_id: str,
    ) -> WorkModelEventV2:
        event = self.job_store._get_record(
            connection,
            "work_model_events",
            "work_model_event_id",
            event_id,
            WorkModelEventV2,
        )
        if work_model_event_v2_ref(event).object_id != event_id:
            raise ImmutableResultError("stored model event identity is inconsistent")
        return event

    def _get_cancellation_request(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> WorkModelCancellationRequestV2:
        request = self.job_store._get_record(
            connection,
            "work_model_cancellation_requests",
            "work_model_cancellation_request_id",
            request_id,
            WorkModelCancellationRequestV2,
        )
        if work_model_cancellation_request_v2_ref(request).object_id != request_id:
            raise ImmutableResultError("stored model cancellation request identity is inconsistent")
        return request

    def _get_cancellation_result(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> WorkModelCancellationResultV2:
        result = self.job_store._get_record(
            connection,
            "work_model_cancellation_results",
            "work_model_cancellation_result_id",
            result_id,
            WorkModelCancellationResultV2,
        )
        if work_model_cancellation_result_v2_ref(result).object_id != result_id:
            raise ImmutableResultError("stored model cancellation result identity is inconsistent")
        return result

    def _get_event_for_cancellation_result(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> WorkModelEventV2:
        result = self._get_cancellation_result(connection, result_id)
        request = self._get_cancellation_request(
            connection,
            result.cancellation_request_ref.object_id,
        )
        rows = connection.execute(
            """
            SELECT record_json
            FROM work_model_events
            WHERE work_model_reservation_id = ?
            ORDER BY reservation_version
            """,
            (request.work_model_reservation_ref.object_id,),
        ).fetchall()
        result_ref = work_model_cancellation_result_v2_ref(result)
        matches = tuple(
            event
            for row in rows
            for event in (self.job_store._load_record(row, WorkModelEventV2),)
            if event.model_cancellation_result_ref == result_ref
        )
        if len(matches) != 1:
            raise ImmutableResultError("model cancellation result event binding is inconsistent")
        return matches[0]

    def _list_job_model_closures(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        cancellation: WorkCancellationRecordV2,
    ) -> tuple[
        tuple[WorkModelEventV2, ...],
        tuple[WorkModelCancellationRequestV2, ...],
    ]:
        event_rows = connection.execute(
            """
            SELECT events.work_model_event_id
            FROM work_model_events AS events
            JOIN work_model_reservations AS reservations
              ON reservations.work_model_reservation_id =
                 events.work_model_reservation_id
            WHERE reservations.job_id = ?
            ORDER BY events.work_model_reservation_id,
                     events.reservation_version
            """,
            (job_id,),
        ).fetchall()
        cancelled_event_refs = set(cancellation.cancelled_lease_event_refs)
        observed_events = tuple(
            self._get_model_event(
                connection,
                str(row["work_model_event_id"]),
            )
            for row in event_rows
        )
        events = tuple(
            event
            for event in observed_events
            if event.reservation_version == 1
            and event.event_kind is WorkModelEventKindV2.RELEASE_PENDING_ACK
            and event.work_lease_event_ref in cancelled_event_refs
        )
        request_rows = connection.execute(
            """
            SELECT requests.work_model_cancellation_request_id
            FROM work_model_cancellation_requests AS requests
            JOIN work_model_reservations AS reservations
              ON reservations.work_model_reservation_id =
                 requests.work_model_reservation_id
            WHERE reservations.job_id = ?
            ORDER BY requests.work_model_cancellation_request_id
            """,
            (job_id,),
        ).fetchall()
        requests = tuple(
            request
            for row in request_rows
            for request in (
                self._get_cancellation_request(
                    connection,
                    str(row["work_model_cancellation_request_id"]),
                ),
            )
            if request.work_lease_event_ref in cancelled_event_refs
        )
        if len(events) != len(requests):
            raise ImmutableResultError("model Job cancellation closure bindings are inconsistent")
        return events, requests

    def _get_pool_head(
        self,
        connection: sqlite3.Connection,
        policy: ModelRatePoolPolicyV2,
    ) -> ModelRatePoolHeadRecord:
        row = connection.execute(
            """
            SELECT model_rate_pool_policy_id, row_version, bucket_version, record_json
            FROM model_rate_pool_heads
            WHERE model_rate_pool_policy_id = ?
            """,
            (policy.model_rate_pool_policy_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("model rate pool head is missing")
        stored = self.job_store._load_record(row, ModelRatePoolHeadRecord)
        if (
            stored.model_rate_pool_policy_id != str(row["model_rate_pool_policy_id"])
            or stored.row_version != int(row["row_version"])
            or stored.bucket_version != int(row["bucket_version"])
        ):
            raise ImmutableResultError("stored model rate pool head columns are inconsistent")
        rebuilt = self._rebuild_pool_head(connection, policy)
        if stored != rebuilt:
            raise ImmutableResultError("model rate pool head differs from immutable events")
        return stored

    def _rebuild_pool_head(
        self,
        connection: sqlite3.Connection,
        policy: ModelRatePoolPolicyV2,
    ) -> ModelRatePoolHeadRecord:
        head = ModelRatePoolHeadRecord(
            model_rate_pool_policy_id=policy.model_rate_pool_policy_id,
            request_credits=policy.request_burst * RATE_CREDITS_PER_UNIT,
            token_credits=policy.token_burst * RATE_CREDITS_PER_UNIT,
            active_concurrency=0,
            blocked_until=None,
            effective_at=policy.audit.created_at,
            bucket_version=0,
            row_version=0,
        )
        rows = connection.execute(
            """
            SELECT work_model_event_id, bucket_version, record_json
            FROM work_model_events
            WHERE model_rate_pool_policy_id = ?
            ORDER BY bucket_version
            """,
            (policy.model_rate_pool_policy_id,),
        ).fetchall()
        for row in rows:
            event = self.job_store._load_record(row, WorkModelEventV2)
            if (
                event.bucket_version != head.bucket_version + 1
                or int(row["bucket_version"]) != event.bucket_version
                or str(row["work_model_event_id"]) != event.work_model_event_id
            ):
                raise ImmutableResultError("model rate pool event sequence is inconsistent")
            reservation = self._get_reservation(
                connection,
                event.work_model_reservation_ref.object_id,
            )
            demand = self._get_demand(
                connection,
                reservation.work_model_demand_ref.object_id,
            )
            if demand.model_rate_pool_policy_ref.object_id != policy.model_rate_pool_policy_id:
                raise ImmutableResultError("model rate pool event belongs to another policy")
            head = _refill_pool_head(head, policy, event.effective_at)
            if event.event_kind is WorkModelEventKindV2.ADMITTED:
                request_credits = head.request_credits - reservation.request_allowance * RATE_CREDITS_PER_UNIT
                token_credits = head.token_credits - reservation.token_allowance * RATE_CREDITS_PER_UNIT
                active = head.active_concurrency + reservation.concurrency_units
            elif event.event_kind is WorkModelEventKindV2.USAGE_OVERRUN_RELEASED:
                request_credits = (
                    head.request_credits
                    - max(
                        event.usage.requests - reservation.request_allowance,
                        0,
                    )
                    * RATE_CREDITS_PER_UNIT
                )
                token_credits = (
                    head.token_credits
                    - max(
                        event.usage.charged_tokens - reservation.token_allowance,
                        0,
                    )
                    * RATE_CREDITS_PER_UNIT
                )
                active = head.active_concurrency - event.released_concurrency
            else:
                request_credits = min(
                    policy.request_burst * RATE_CREDITS_PER_UNIT,
                    head.request_credits + event.refunded_request_allowance * RATE_CREDITS_PER_UNIT,
                )
                token_credits = min(
                    policy.token_burst * RATE_CREDITS_PER_UNIT,
                    head.token_credits + event.refunded_token_allowance * RATE_CREDITS_PER_UNIT,
                )
                active = head.active_concurrency - event.released_concurrency
            admission_overdraw = event.event_kind is WorkModelEventKindV2.ADMITTED and (
                request_credits < 0 or token_credits < 0
            )
            if admission_overdraw or active < 0:
                raise ImmutableResultError("model rate pool event overdraws or over-releases")
            blocked_until = head.blocked_until
            if event.blocked_until is not None and (
                blocked_until is None or event.blocked_until > blocked_until
            ):
                blocked_until = event.blocked_until
            head = head.model_copy(
                update={
                    "request_credits": request_credits,
                    "token_credits": token_credits,
                    "active_concurrency": active,
                    "blocked_until": blocked_until,
                    "bucket_version": event.bucket_version,
                    "row_version": head.row_version + 1,
                }
            )
        return head

    def _get_job_head(
        self,
        connection: sqlite3.Connection,
        policy: JobModelPolicyV2,
    ) -> JobModelHeadRecord:
        row = connection.execute(
            """
            SELECT job_model_policy_id, row_version, record_json
            FROM job_model_heads
            WHERE job_model_policy_id = ?
            """,
            (policy.job_model_policy_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("Job model head is missing")
        stored = self.job_store._load_record(row, JobModelHeadRecord)
        if stored.job_model_policy_id != str(row["job_model_policy_id"]) or stored.row_version != int(
            row["row_version"]
        ):
            raise ImmutableResultError("stored Job model head columns are inconsistent")
        rebuilt = self._rebuild_job_head(connection, policy)
        if stored != rebuilt:
            raise ImmutableResultError("Job model head differs from immutable events")
        return stored

    def _rebuild_job_head(
        self,
        connection: sqlite3.Connection,
        policy: JobModelPolicyV2,
    ) -> JobModelHeadRecord:
        graph = self.job_store._get_job_work_graph_by_id(
            connection,
            policy.resolved_job_work_graph_ref.object_id,
        )
        head = JobModelHeadRecord(
            job_model_policy_id=policy.job_model_policy_id,
            active_concurrency=0,
            reserved_requests=0,
            reserved_tokens=0,
            consumed_requests=0,
            consumed_tokens=0,
            row_version=0,
        )
        rows = connection.execute(
            """
            SELECT work_model_reservation_id
            FROM work_model_reservations
            WHERE job_id = ?
            ORDER BY rowid
            """,
            (graph.job_id,),
        ).fetchall()
        for row in rows:
            reservation = self._get_reservation(
                connection,
                str(row["work_model_reservation_id"]),
            )
            demand = self._get_demand(
                connection,
                reservation.work_model_demand_ref.object_id,
            )
            if demand.job_model_policy_ref != job_model_policy_v2_ref(policy):
                raise ImmutableResultError("model reservation belongs to another Job model policy")
            head = head.model_copy(
                update={
                    "active_concurrency": (head.active_concurrency + reservation.concurrency_units),
                    "reserved_requests": (head.reserved_requests + reservation.request_allowance),
                    "reserved_tokens": (head.reserved_tokens + reservation.token_allowance),
                    "row_version": head.row_version + 1,
                }
            )
            event_rows = connection.execute(
                """
                SELECT record_json
                FROM work_model_events
                WHERE work_model_reservation_id = ? AND reservation_version > 0
                ORDER BY reservation_version
                """,
                (reservation.work_model_reservation_id,),
            ).fetchall()
            for event_row in event_rows:
                event = self.job_store._load_record(event_row, WorkModelEventV2)
                consumed = event.usage
                release_budget = event.reservation_version == 1
                head = head.model_copy(
                    update={
                        "active_concurrency": (head.active_concurrency - event.released_concurrency),
                        "reserved_requests": (
                            head.reserved_requests - (reservation.request_allowance if release_budget else 0)
                        ),
                        "reserved_tokens": (
                            head.reserved_tokens - (reservation.token_allowance if release_budget else 0)
                        ),
                        "consumed_requests": (
                            head.consumed_requests + (consumed.requests if release_budget else 0)
                        ),
                        "consumed_tokens": (
                            head.consumed_tokens + (consumed.charged_tokens if release_budget else 0)
                        ),
                        "row_version": head.row_version + 1,
                    }
                )
        if (
            min(
                head.active_concurrency,
                head.reserved_requests,
                head.reserved_tokens,
            )
            < 0
        ):
            raise ImmutableResultError("Job model event over-releases reserved state")
        return head

    def _get_reservation_head(
        self,
        connection: sqlite3.Connection,
        reservation: WorkModelReservationV2,
    ) -> WorkModelReservationHeadRecord:
        row = connection.execute(
            """
            SELECT work_model_reservation_id, state, reservation_version, record_json
            FROM work_model_reservation_heads
            WHERE work_model_reservation_id = ?
            """,
            (reservation.work_model_reservation_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("model reservation head is missing")
        stored = self.job_store._load_record(
            row,
            WorkModelReservationHeadRecord,
        )
        if (
            stored.work_model_reservation_id != str(row["work_model_reservation_id"])
            or stored.state.value != str(row["state"])
            or stored.reservation_version != int(row["reservation_version"])
        ):
            raise ImmutableResultError("stored model reservation head columns are inconsistent")
        rebuilt = self._rebuild_reservation_head(connection, reservation)
        if stored != rebuilt:
            raise ImmutableResultError("model reservation head differs from immutable events")
        return stored

    def _rebuild_reservation_head(
        self,
        connection: sqlite3.Connection,
        reservation: WorkModelReservationV2,
    ) -> WorkModelReservationHeadRecord:
        rows = connection.execute(
            """
            SELECT reservation_version, record_json
            FROM work_model_events
            WHERE work_model_reservation_id = ?
            ORDER BY reservation_version
            """,
            (reservation.work_model_reservation_id,),
        ).fetchall()
        state: WorkModelReservationStateV2 | None = None
        version = -1
        for row in rows:
            event = self.job_store._load_record(row, WorkModelEventV2)
            if (
                event.reservation_version != version + 1
                or int(row["reservation_version"]) != event.reservation_version
            ):
                raise ImmutableResultError("model reservation event sequence is inconsistent")
            if event.event_kind is WorkModelEventKindV2.ADMITTED:
                if state is not None:
                    raise ImmutableResultError("model reservation has duplicate admission")
                state = WorkModelReservationStateV2.ACTIVE
            elif event.event_kind is WorkModelEventKindV2.RELEASE_PENDING_ACK:
                if state is not WorkModelReservationStateV2.ACTIVE:
                    raise ImmutableResultError("model reservation pending transition is illegal")
                state = WorkModelReservationStateV2.RELEASE_PENDING_ACK
            elif event.event_kind is WorkModelEventKindV2.QUARANTINED:
                if state is not WorkModelReservationStateV2.RELEASE_PENDING_ACK:
                    raise ImmutableResultError("model reservation quarantine transition is illegal")
                state = WorkModelReservationStateV2.QUARANTINED
            elif event.event_kind is WorkModelEventKindV2.ACKNOWLEDGED_RELEASED:
                if state is not WorkModelReservationStateV2.RELEASE_PENDING_ACK:
                    raise ImmutableResultError("model reservation acknowledgement transition is illegal")
                state = WorkModelReservationStateV2.RELEASED
            elif event.event_kind in {
                WorkModelEventKindV2.COMPLETED_RELEASED,
                WorkModelEventKindV2.BACKPRESSURED_RELEASED,
                WorkModelEventKindV2.USAGE_OVERRUN_RELEASED,
            }:
                if state is not WorkModelReservationStateV2.ACTIVE:
                    raise ImmutableResultError("model reservation release transition is illegal")
                state = WorkModelReservationStateV2.RELEASED
            else:
                raise ImmutableResultError("unknown model reservation event kind")
            version = event.reservation_version
        if state is None or version < 0:
            raise ImmutableResultError("model reservation has no immutable admission event")
        return WorkModelReservationHeadRecord(
            work_model_reservation_id=reservation.work_model_reservation_id,
            state=state,
            reservation_version=version,
        )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        event: WorkModelEventV2,
        pool_policy_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO work_model_events (
                work_model_event_id, work_model_reservation_id,
                model_rate_pool_policy_id, reservation_version,
                bucket_version, event_kind, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.work_model_event_id,
                event.work_model_reservation_ref.object_id,
                pool_policy_id,
                event.reservation_version,
                event.bucket_version,
                event.event_kind,
                self.job_store._record_json(event),
            ),
        )

    def _update_pool_head(
        self,
        connection: sqlite3.Connection,
        head: ModelRatePoolHeadRecord,
        *,
        expected_version: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE model_rate_pool_heads
            SET row_version = ?, bucket_version = ?, record_json = ?
            WHERE model_rate_pool_policy_id = ? AND row_version = ?
            """,
            (
                head.row_version,
                head.bucket_version,
                self.job_store._record_json(head),
                head.model_rate_pool_policy_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ModelControlPolicyError("model rate pool head changed concurrently")

    def _update_job_head(
        self,
        connection: sqlite3.Connection,
        head: JobModelHeadRecord,
        *,
        expected_version: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE job_model_heads
            SET row_version = ?, record_json = ?
            WHERE job_model_policy_id = ? AND row_version = ?
            """,
            (
                head.row_version,
                self.job_store._record_json(head),
                head.job_model_policy_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ModelControlPolicyError("Job model head changed concurrently")

    def _update_reservation_head(
        self,
        connection: sqlite3.Connection,
        head: WorkModelReservationHeadRecord,
        *,
        expected_version: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE work_model_reservation_heads
            SET state = ?, reservation_version = ?, record_json = ?
            WHERE work_model_reservation_id = ? AND reservation_version = ?
            """,
            (
                head.state,
                head.reservation_version,
                self.job_store._record_json(head),
                head.work_model_reservation_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleModelReservationError("model reservation head changed concurrently")


def _classify(
    *,
    pool_policy: ModelRatePoolPolicyV2,
    job_policy: JobModelPolicyV2,
    pool_head: ModelRatePoolHeadRecord,
    job_head: JobModelHeadRecord,
    demand: WorkModelDemandV2,
    now: datetime,
) -> tuple[
    ModelAdmissionOutcomeV2,
    tuple[ModelRateDimensionV2, ...],
    datetime | None,
]:
    unsatisfiable: set[ModelRateDimensionV2] = set()
    if (
        demand.request_allowance > pool_policy.request_burst
        or demand.request_allowance > job_policy.max_model_requests
    ):
        unsatisfiable.add(ModelRateDimensionV2.REQUESTS)
    if (
        demand.token_allowance > pool_policy.token_burst
        or demand.token_allowance > job_policy.max_model_tokens
    ):
        unsatisfiable.add(ModelRateDimensionV2.TOKENS)
    if (
        demand.concurrency_units > pool_policy.max_concurrent_requests
        or demand.concurrency_units > job_policy.max_concurrent_requests
    ):
        unsatisfiable.add(ModelRateDimensionV2.CONCURRENCY)
    if unsatisfiable:
        return (
            ModelAdmissionOutcomeV2.UNSATISFIABLE_DEMAND,
            _sorted_dimensions(unsatisfiable),
            None,
        )
    exhausted: set[ModelRateDimensionV2] = set()
    if (
        job_head.consumed_requests + job_head.reserved_requests + demand.request_allowance
        > job_policy.max_model_requests
    ):
        exhausted.add(ModelRateDimensionV2.REQUESTS)
    if (
        job_head.consumed_tokens + job_head.reserved_tokens + demand.token_allowance
        > job_policy.max_model_tokens
    ):
        exhausted.add(ModelRateDimensionV2.TOKENS)
    if exhausted:
        return (
            ModelAdmissionOutcomeV2.BUDGET_EXHAUSTED,
            _sorted_dimensions(exhausted),
            None,
        )
    if pool_head.blocked_until is not None and pool_head.blocked_until > now:
        return (
            ModelAdmissionOutcomeV2.WAITING_PROVIDER,
            (ModelRateDimensionV2.PROVIDER_BACKPRESSURE,),
            pool_head.blocked_until,
        )
    rate_waits: list[tuple[ModelRateDimensionV2, int]] = []
    request_debit = demand.request_allowance * RATE_CREDITS_PER_UNIT
    if request_debit > pool_head.request_credits:
        deficit = request_debit - pool_head.request_credits
        rate_waits.append(
            (
                ModelRateDimensionV2.REQUESTS,
                _ceil_div(deficit, pool_policy.requests_per_minute),
            )
        )
    token_debit = demand.token_allowance * RATE_CREDITS_PER_UNIT
    if token_debit > pool_head.token_credits:
        deficit = token_debit - pool_head.token_credits
        rate_waits.append(
            (
                ModelRateDimensionV2.TOKENS,
                _ceil_div(deficit, pool_policy.tokens_per_minute),
            )
        )
    if rate_waits:
        wait_us = max(item[1] for item in rate_waits)
        return (
            ModelAdmissionOutcomeV2.WAITING_RATE,
            _sorted_dimensions({item[0] for item in rate_waits}),
            now + timedelta(microseconds=wait_us),
        )
    if (
        pool_head.active_concurrency + demand.concurrency_units > pool_policy.max_concurrent_requests
        or job_head.active_concurrency + demand.concurrency_units > job_policy.max_concurrent_requests
    ):
        return (
            ModelAdmissionOutcomeV2.WAITING_CONCURRENCY,
            (ModelRateDimensionV2.CONCURRENCY,),
            None,
        )
    return ModelAdmissionOutcomeV2.ADMITTED, (), None


def _refill_pool_head(
    head: ModelRatePoolHeadRecord,
    policy: ModelRatePoolPolicyV2,
    effective_at: datetime,
) -> ModelRatePoolHeadRecord:
    if effective_at < head.effective_at:
        raise ImmutableResultError("model rate pool clock moved backwards")
    elapsed = effective_at - head.effective_at
    elapsed_microseconds = elapsed.days * 86_400_000_000 + elapsed.seconds * 1_000_000 + elapsed.microseconds
    return head.model_copy(
        update={
            "request_credits": min(
                policy.request_burst * RATE_CREDITS_PER_UNIT,
                head.request_credits + elapsed_microseconds * policy.requests_per_minute,
            ),
            "token_credits": min(
                policy.token_burst * RATE_CREDITS_PER_UNIT,
                head.token_credits + elapsed_microseconds * policy.tokens_per_minute,
            ),
            "effective_at": effective_at,
        }
    )


def _model_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    versions = tuple(item for item in audit.governing_versions if item.component != "model-control")
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=(
            *versions,
            VersionBinding(
                component="model-control",
                version=R6_MODEL_CONTROL_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(sorted(refs, key=_ref_key)),
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)


def _sorted_dimensions(
    dimensions: set[ModelRateDimensionV2],
) -> tuple[ModelRateDimensionV2, ...]:
    return tuple(sorted(dimensions, key=lambda item: item.value))


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _zero_usage() -> ModelUsageV2:
    return ModelUsageV2(
        requests=0,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=0,
        source=ModelUsageSourceV2.REPORTED,
    )


def _invocation_handle_ref(lease: WorkLeaseV2) -> ObjectRef:
    digest = hashlib.sha256(f"{lease.work_lease_id}:{lease.fencing_token}:model".encode()).hexdigest()
    return ObjectRef(
        object_type="model-invocation-handle",
        object_id=f"model-invocation-handle://sha256/{digest}",
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


def _factory_ref(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _require_model_receipt_ref(refs: tuple[ObjectRef, ...]) -> None:
    receipts = tuple(
        ref for ref in refs if ref.object_type == "model-control-receipt" and ref.object_version == "v2"
    )
    if len(receipts) != 1:
        raise ModelControlPolicyError("model completion requires exactly one model-control-receipt v2 ref")


def _internal_key(key: str, suffix: str) -> str:
    digest = hashlib.sha256(f"{key}:{suffix}".encode()).hexdigest()
    return f"internal-model-control:{digest}"
