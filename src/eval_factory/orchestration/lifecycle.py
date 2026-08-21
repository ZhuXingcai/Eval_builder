from __future__ import annotations

from dataclasses import dataclass

from eval_factory.contracts.cli_v2 import (
    MAX_PIPELINE_PAGE_SIZE,
    PipelineCancellationModeV2,
    PipelineCancellationResultV2,
    PipelineControlConfigV2,
    PipelineCreateResultV2,
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
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.model_control_v2 import (
    JobModelPolicyV2,
    WorkModelReservationStateV2,
    work_model_cancellation_request_v2_ref,
)
from eval_factory.contracts.observability_v2 import (
    BatchAuditFindingCodeV2,
    BatchAuditReportV2,
    MetricScopeV2,
    batch_audit_report_v2_ref,
    work_attempt_metrics_v2_ref,
)
from eval_factory.contracts.orchestration import JobStatus
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedDatasetJobPlanV2,
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    WorkControlPolicyV2,
    WorkLeaseStateV2,
    WorkRetryDecisionKindV2,
    resolved_dataset_job_plan_v2_ref,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
    work_cancellation_record_v2_ref,
    work_control_policy_v2_ref,
    work_lease_v2_ref,
    work_readiness_snapshot_v2_ref,
    work_retry_decision_v2_ref,
)
from eval_factory.contracts.resource_v2 import (
    JobResourcePolicyV2,
    WorkResourceReservationStateV2,
    work_resource_termination_request_v2_ref,
)
from eval_factory.orchestration.control import WorkControlService
from eval_factory.orchestration.fanout import (
    DatasetJobWorkGraphCompiler,
)
from eval_factory.orchestration.job_store import (
    JobStore,
    JobStoreError,
    RecordNotFoundError,
)
from eval_factory.orchestration.model_control import (
    CombinedJobCancellationResult,
    ModelControlService,
    ModelJobCancellationResult,
)
from eval_factory.orchestration.models import (
    ItemRecord,
    JobRecord,
)
from eval_factory.orchestration.observability import (
    BatchObservabilityResult,
    BatchObservabilityService,
)
from eval_factory.orchestration.planning import (
    DatasetJobPlanCompiler,
)
from eval_factory.orchestration.resources import (
    ResourceCancellationResult,
    ResourceControlService,
)


class PipelineLifecyclePolicyError(JobStoreError):
    pass


class PipelineObservabilityRefreshRequiredError(PipelineLifecyclePolicyError):
    pass


class PipelineCheckpointRequiredError(PipelineLifecyclePolicyError):
    pass


@dataclass(frozen=True, slots=True)
class _CompiledPipeline:
    plan: ResolvedDatasetJobPlanV2
    graph: ResolvedJobWorkGraphV2
    control_policy: WorkControlPolicyV2


@dataclass(frozen=True, slots=True)
class _StoredPipeline:
    job: JobRecord
    spec: DatasetJobSpecV2
    plan: ResolvedDatasetJobPlanV2
    graph: ResolvedJobWorkGraphV2
    control_policy: WorkControlPolicyV2
    items: tuple[ItemRecord, ...]
    work_units: tuple[ResolvedWorkUnitV2, ...]


class PipelineLifecycleService:
    def __init__(self, job_store: JobStore) -> None:
        self.job_store = job_store
        self.work_control = WorkControlService(job_store)
        self.model_control = ModelControlService(job_store)
        self.resource_control = ResourceControlService(job_store)
        self.observability = BatchObservabilityService(job_store)

    @staticmethod
    def plan(
        *,
        job_spec: DatasetJobSpecV2,
        audit: ContractAudit,
    ) -> PipelinePlanResultV2:
        spec = DatasetJobSpecV2.model_validate(job_spec.model_dump(mode="python"))
        plan = DatasetJobPlanCompiler().compile(
            job_spec=spec,
            audit=audit,
        )
        graph = DatasetJobWorkGraphCompiler().compile(
            job_spec=spec,
            resolved_plan=plan,
            audit=audit,
        )
        limit = min(
            max(len(graph.work_units), 1),
            MAX_PIPELINE_PAGE_SIZE,
        )
        page = graph.work_units[:limit]
        return PipelinePlanResultV2(
            job_id=spec.job_id,
            job_spec_ref=plan.dataset_job_spec_ref,
            resolved_plan_ref=resolved_dataset_job_plan_v2_ref(plan),
            work_graph_ref=resolved_job_work_graph_v2_ref(graph),
            resolved_stages=plan.resolved_stages,
            work_unit_count=len(graph.work_units),
            work_unit_offset=0,
            work_unit_limit=limit,
            work_units=tuple(PipelineLifecycleService._bare_work_status(unit) for unit in page),
        )

    def create(
        self,
        *,
        job_spec: DatasetJobSpecV2,
        control: PipelineControlConfigV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> PipelineCreateResultV2:
        spec = DatasetJobSpecV2.model_validate(job_spec.model_dump(mode="python"))
        config = PipelineControlConfigV2.model_validate(control.model_dump(mode="python"))
        if idempotency_key != spec.idempotency_key:
            raise PipelineLifecyclePolicyError(
                "pipeline create idempotency key must match DatasetJobSpecV2.idempotency_key"
            )
        compiled = self._compile(
            job_spec=spec,
            control=config,
            audit=audit,
        )

        job = self.job_store.create_planned_job(
            spec,
            compiled.plan,
        )
        graph = self.job_store.create_job_work_graph(
            compiled.graph,
            idempotency_key=idempotency_key,
        )
        policy = self.work_control.bind_policy(
            graph=graph,
            lease_duration_seconds=config.lease_duration_seconds,
            heartbeat_extension_seconds=(config.heartbeat_extension_seconds),
            max_attempts=config.max_attempts,
            retry_delay_seconds=config.retry_delay_seconds,
            retry_lease_expiry=config.retry_lease_expiry,
            audit=audit,
            idempotency_key=idempotency_key,
        )
        if policy != compiled.control_policy:
            raise PipelineLifecyclePolicyError(
                "persisted work control policy differs from the validated create input"
            )
        return PipelineCreateResultV2(
            job_id=job.job_id,
            job_status=JobStatus.CREATED,
            job_version=0,
            resolved_plan_ref=resolved_dataset_job_plan_v2_ref(compiled.plan),
            work_graph_ref=resolved_job_work_graph_v2_ref(graph),
            work_control_policy_ref=work_control_policy_v2_ref(policy),
            item_count=len(graph.item_ids),
            work_unit_count=len(graph.work_units),
        )

    def status(
        self,
        *,
        job_id: str,
        offset: int,
        limit: int,
    ) -> PipelineStatusResultV2:
        return self._status(
            job_id=job_id,
            offset=offset,
            limit=limit,
            incomplete_only=False,
        )

    def resume(
        self,
        *,
        job_id: str,
        expected_job_version: int,
        idempotency_key: str,
        offset: int,
        limit: int,
    ) -> PipelineResumeResultV2:
        _validate_page(offset=offset, limit=limit)
        current = self.job_store.get_job(job_id)
        if current.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            raise PipelineLifecyclePolicyError("terminal Job cannot be resumed")

        if current.status is JobStatus.CREATED:
            action = PipelineResumeActionV2.STARTED
            self.job_store.transition_job(
                job_id,
                JobStatus.RUNNING,
                expected_version=expected_job_version,
                idempotency_key=idempotency_key,
                reason="pipeline-resume",
            )
        elif current.status is JobStatus.BLOCKED:
            if self.job_store.has_current_checkpoint_interaction(job_id):
                raise PipelineCheckpointRequiredError("BLOCKED Job requires checkpoint-aware resume")
            action = PipelineResumeActionV2.REOPENED
            self.job_store.transition_job(
                job_id,
                JobStatus.RUNNING,
                expected_version=expected_job_version,
                idempotency_key=idempotency_key,
                reason="pipeline-resume",
            )
        else:
            action = PipelineResumeActionV2.ALREADY_RUNNING
            if current.row_version != expected_job_version:
                self.job_store.transition_job(
                    job_id,
                    JobStatus.RUNNING,
                    expected_version=expected_job_version,
                    idempotency_key=idempotency_key,
                    reason="pipeline-resume",
                )

        return PipelineResumeResultV2(
            action=action,
            status=self._status(
                job_id=job_id,
                offset=offset,
                limit=limit,
                incomplete_only=True,
            ),
        )

    def cancel(
        self,
        *,
        job_id: str,
        reason_code: str,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> PipelineCancellationResultV2:
        stored = self._load_pipeline(job_id)
        if stored.job.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
        }:
            raise PipelineLifecyclePolicyError("terminal Job cannot be cancelled")
        model_policy = self._model_policy_optional(job_id)
        resource_policy = self._resource_policy_optional(job_id)

        if model_policy is not None and resource_policy is not None:
            combined = self.model_control.cancel_job_combined(
                graph=stored.graph,
                control_policy=stored.control_policy,
                model_job_policy=model_policy,
                resource_job_policy=resource_policy,
                reason_code=reason_code,
                audit=audit,
                idempotency_key=idempotency_key,
            )
            return self._combined_cancellation(job_id, combined)
        if model_policy is not None:
            model = self.model_control.cancel_job(
                graph=stored.graph,
                control_policy=stored.control_policy,
                job_policy=model_policy,
                reason_code=reason_code,
                audit=audit,
                idempotency_key=idempotency_key,
            )
            return self._model_cancellation(job_id, model)
        if resource_policy is not None:
            resource = self.resource_control.cancel_job(
                graph=stored.graph,
                control_policy=stored.control_policy,
                job_policy=resource_policy,
                reason_code=reason_code,
                audit=audit,
                idempotency_key=idempotency_key,
            )
            return self._resource_cancellation(job_id, resource)

        cancellation = self.work_control.cancel_job(
            graph=stored.graph,
            policy=stored.control_policy,
            reason_code=reason_code,
            audit=audit,
            idempotency_key=idempotency_key,
        )
        return PipelineCancellationResultV2(
            job_id=job_id,
            mode=PipelineCancellationModeV2.WORK,
            cancellation_ref=work_cancellation_record_v2_ref(cancellation),
            preserved_result_refs=cancellation.preserved_result_refs,
            pending_physical_acknowledgements=0,
        )

    def events(
        self,
        *,
        job_id: str,
        offset: int,
        limit: int,
        refresh: bool,
        audit: ContractAudit | None,
        idempotency_key: str | None,
    ) -> PipelineEventsResultV2:
        _validate_page(offset=offset, limit=limit)
        report = self._report(
            job_id=job_id,
            refresh=refresh,
            audit=audit,
            idempotency_key=idempotency_key,
        )
        events = self.observability.list_audit_events(job_id)
        page = _page(events, offset=offset, limit=limit)
        return PipelineEventsResultV2(
            job_id=job_id,
            report_ref=batch_audit_report_v2_ref(report),
            audit_outcome=report.outcome,
            finding_codes=_finding_codes(report),
            total_events=len(events),
            event_offset=offset,
            event_limit=limit,
            events=page,
        )

    def metrics(
        self,
        *,
        job_id: str,
        item_id: str | None,
        scope: MetricScopeV2 | None,
        offset: int,
        limit: int,
        refresh: bool,
        audit: ContractAudit | None,
        idempotency_key: str | None,
    ) -> PipelineMetricsResultV2:
        _validate_page(offset=offset, limit=limit)
        report = self._report(
            job_id=job_id,
            refresh=refresh,
            audit=audit,
            idempotency_key=idempotency_key,
        )
        attempts = self.observability.list_attempt_metrics(job_id)
        if item_id is not None:
            item = self.job_store.get_item(item_id)
            if item.job_id != job_id:
                raise PipelineLifecyclePolicyError("metrics item belongs to another Job")
            attempts = tuple(value for value in attempts if value.item_id == item_id)
            metric_slices = self.observability.get_item_metrics(item_id).metric_slices
        else:
            metric_slices = self.observability.get_batch_metrics(job_id).metric_slices

        attempt_refs = tuple(
            sorted(
                (work_attempt_metrics_v2_ref(value) for value in attempts),
                key=_ref_key,
            )
        )
        attempt_page = _page(
            attempt_refs,
            offset=offset,
            limit=limit,
        )
        slices = tuple(
            PipelineMetricSliceSummaryV2.from_metric_slice(value)
            for value in metric_slices
            if scope is None or value.scope is scope
        )
        return PipelineMetricsResultV2(
            job_id=job_id,
            item_id=item_id,
            report_ref=batch_audit_report_v2_ref(report),
            audit_outcome=report.outcome,
            finding_codes=_finding_codes(report),
            total_attempts=len(attempt_refs),
            attempt_offset=offset,
            attempt_limit=limit,
            attempt_metrics_refs=attempt_page,
            metric_slices=tuple(
                sorted(
                    slices,
                    key=lambda value: (
                        value.scope.value,
                        value.scope_id,
                    ),
                )
            ),
        )

    def _compile(
        self,
        *,
        job_spec: DatasetJobSpecV2,
        control: PipelineControlConfigV2,
        audit: ContractAudit,
    ) -> _CompiledPipeline:
        plan = DatasetJobPlanCompiler().compile(
            job_spec=job_spec,
            audit=audit,
        )
        graph = DatasetJobWorkGraphCompiler().compile(
            job_spec=job_spec,
            resolved_plan=plan,
            audit=audit,
        )
        policy = self.work_control.compile_policy(
            graph=graph,
            lease_duration_seconds=control.lease_duration_seconds,
            heartbeat_extension_seconds=(control.heartbeat_extension_seconds),
            max_attempts=control.max_attempts,
            retry_delay_seconds=control.retry_delay_seconds,
            retry_lease_expiry=control.retry_lease_expiry,
            audit=audit,
        )
        return _CompiledPipeline(
            plan=plan,
            graph=graph,
            control_policy=policy,
        )

    def _load_pipeline(self, job_id: str) -> _StoredPipeline:
        job = self.job_store.get_job(job_id)
        spec = self.job_store.get_job_spec(job_id)
        plan = self.job_store.get_resolved_job_plan(job_id)
        graph = self.job_store.get_job_work_graph(job_id)
        control_policy = self.job_store.get_work_control_policy(job_id)
        DatasetJobPlanCompiler().validate_current(
            plan,
            job_spec=spec,
        )
        DatasetJobWorkGraphCompiler().validate_current(
            graph,
            job_spec=spec,
            resolved_plan=plan,
        )
        if control_policy.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph):
            raise PipelineLifecyclePolicyError("work control policy is not current")
        return _StoredPipeline(
            job=job,
            spec=spec,
            plan=plan,
            graph=graph,
            control_policy=control_policy,
            items=self.job_store.list_items(job_id),
            work_units=self.job_store.list_work_units(job_id=job_id),
        )

    def _status(
        self,
        *,
        job_id: str,
        offset: int,
        limit: int,
        incomplete_only: bool,
    ) -> PipelineStatusResultV2:
        _validate_page(offset=offset, limit=limit)
        stored = self._load_pipeline(job_id)
        leases = self.job_store.list_work_leases(job_id=job_id)
        lease_by_unit = {lease.work_unit_ref.object_id: lease for lease in leases}
        metrics_by_lease = {
            value.work_lease_ref.object_id: value
            for value in self.job_store.list_work_attempt_metrics(job_id=job_id)
        }
        statuses: list[PipelineWorkUnitStatusV2] = []
        for unit in stored.work_units:
            readiness_values = self.job_store.list_work_readiness(unit.resolved_work_unit_id)
            readiness = readiness_values[-1] if readiness_values else None
            lease = lease_by_unit.get(unit.resolved_work_unit_id)
            head = self.job_store.get_work_lease_head(lease.work_lease_id) if lease is not None else None
            retry = self.work_control.get_retry_decision_optional(unit.resolved_work_unit_id)
            metrics = metrics_by_lease.get(lease.work_lease_id) if lease is not None else None
            value = PipelineWorkUnitStatusV2(
                work_unit_ref=resolved_work_unit_v2_ref(unit),
                scope=unit.scope,
                stage=unit.stage,
                item_id=unit.item_id,
                readiness=(readiness.readiness if readiness is not None else None),
                readiness_ref=(work_readiness_snapshot_v2_ref(readiness) if readiness is not None else None),
                lease_ref=(work_lease_v2_ref(lease) if lease is not None else None),
                lease_state=head.state if head is not None else None,
                lease_version=(head.lease_version if head is not None else None),
                retry_decision_ref=(work_retry_decision_v2_ref(retry) if retry is not None else None),
                retry_decision=(retry.decision if retry is not None else None),
                attempt_metrics_ref=(work_attempt_metrics_v2_ref(metrics) if metrics is not None else None),
            )
            if not incomplete_only or _is_incomplete(value):
                statuses.append(value)

        page = _page(
            tuple(statuses),
            offset=offset,
            limit=limit,
        )
        try:
            report = self.observability.get_audit_report(job_id)
        except RecordNotFoundError:
            availability = PipelineObservabilityAvailabilityV2.REFRESH_REQUIRED
            report_ref = None
            audit_outcome = None
        else:
            if self.observability.projection_is_current(job_id):
                availability = PipelineObservabilityAvailabilityV2.AVAILABLE
                report_ref = batch_audit_report_v2_ref(report)
                audit_outcome = report.outcome
            else:
                availability = PipelineObservabilityAvailabilityV2.REFRESH_REQUIRED
                report_ref = None
                audit_outcome = None

        current_job = self.job_store.get_job(job_id)
        return PipelineStatusResultV2(
            job_id=job_id,
            job_status=current_job.status,
            job_version=current_job.row_version,
            resolved_plan_ref=resolved_dataset_job_plan_v2_ref(stored.plan),
            work_graph_ref=resolved_job_work_graph_v2_ref(stored.graph),
            work_control_policy_ref=work_control_policy_v2_ref(stored.control_policy),
            item_count=len(stored.items),
            work_unit_count=len(statuses),
            terminal_attempt_count=len(metrics_by_lease),
            work_unit_offset=offset,
            work_unit_limit=limit,
            work_units=page,
            observability_availability=availability,
            audit_report_ref=report_ref,
            audit_outcome=audit_outcome,
        )

    @staticmethod
    def _bare_work_status(
        unit: ResolvedWorkUnitV2,
    ) -> PipelineWorkUnitStatusV2:
        return PipelineWorkUnitStatusV2(
            work_unit_ref=resolved_work_unit_v2_ref(unit),
            scope=unit.scope,
            stage=unit.stage,
            item_id=unit.item_id,
        )

    def _report(
        self,
        *,
        job_id: str,
        refresh: bool,
        audit: ContractAudit | None,
        idempotency_key: str | None,
    ) -> BatchAuditReportV2:
        self.job_store.get_job(job_id)
        if refresh:
            if audit is None or not idempotency_key:
                raise PipelineLifecyclePolicyError("observability refresh requires audit and idempotency key")
            result: BatchObservabilityResult = self.observability.refresh_job(
                job_id=job_id,
                audit=audit,
                idempotency_key=idempotency_key,
            )
            return result.report
        try:
            report = self.observability.get_audit_report(job_id)
        except RecordNotFoundError as exc:
            raise PipelineObservabilityRefreshRequiredError(
                "observability projection is unavailable; explicit refresh is required"
            ) from exc
        if not self.observability.projection_is_current(job_id):
            raise PipelineObservabilityRefreshRequiredError(
                "observability projection is stale; explicit refresh is required"
            )
        return report

    def _model_policy_optional(
        self,
        job_id: str,
    ) -> JobModelPolicyV2 | None:
        try:
            return self.model_control.get_job_policy(job_id)
        except RecordNotFoundError:
            return None

    def _resource_policy_optional(
        self,
        job_id: str,
    ) -> JobResourcePolicyV2 | None:
        try:
            return self.resource_control.get_job_policy(job_id)
        except RecordNotFoundError:
            return None

    def _model_cancellation(
        self,
        job_id: str,
        value: ModelJobCancellationResult,
    ) -> PipelineCancellationResultV2:
        pending = tuple(
            request
            for request in value.cancellation_requests
            if self.model_control.get_reservation_head(request.work_model_reservation_ref.object_id).state
            is WorkModelReservationStateV2.RELEASE_PENDING_ACK
        )
        request_refs = _sorted_refs(
            tuple(work_model_cancellation_request_v2_ref(request) for request in pending)
        )
        return PipelineCancellationResultV2(
            job_id=job_id,
            mode=PipelineCancellationModeV2.MODEL,
            cancellation_ref=work_cancellation_record_v2_ref(value.cancellation),
            preserved_result_refs=(value.cancellation.preserved_result_refs),
            model_cancellation_request_refs=request_refs,
            pending_physical_acknowledgements=len(request_refs),
        )

    def _resource_cancellation(
        self,
        job_id: str,
        value: ResourceCancellationResult,
    ) -> PipelineCancellationResultV2:
        pending = tuple(
            request
            for request in value.termination_requests
            if self.resource_control.get_reservation_head(
                request.work_resource_reservation_ref.object_id
            ).state
            is WorkResourceReservationStateV2.RELEASE_PENDING_TERMINATION
        )
        request_refs = _sorted_refs(
            tuple(work_resource_termination_request_v2_ref(request) for request in pending)
        )
        return PipelineCancellationResultV2(
            job_id=job_id,
            mode=PipelineCancellationModeV2.RESOURCE,
            cancellation_ref=work_cancellation_record_v2_ref(value.cancellation),
            preserved_result_refs=(value.cancellation.preserved_result_refs),
            resource_termination_request_refs=request_refs,
            pending_physical_acknowledgements=len(request_refs),
        )

    def _combined_cancellation(
        self,
        job_id: str,
        value: CombinedJobCancellationResult,
    ) -> PipelineCancellationResultV2:
        pending_model = tuple(
            request
            for request in value.model_cancellation_requests
            if self.model_control.get_reservation_head(request.work_model_reservation_ref.object_id).state
            is WorkModelReservationStateV2.RELEASE_PENDING_ACK
        )
        pending_resource = tuple(
            request
            for request in value.resource_termination_requests
            if self.resource_control.get_reservation_head(
                request.work_resource_reservation_ref.object_id
            ).state
            is WorkResourceReservationStateV2.RELEASE_PENDING_TERMINATION
        )
        model_refs = _sorted_refs(
            tuple(work_model_cancellation_request_v2_ref(request) for request in pending_model)
        )
        resource_refs = _sorted_refs(
            tuple(work_resource_termination_request_v2_ref(request) for request in pending_resource)
        )
        return PipelineCancellationResultV2(
            job_id=job_id,
            mode=PipelineCancellationModeV2.COMBINED,
            cancellation_ref=work_cancellation_record_v2_ref(value.cancellation),
            preserved_result_refs=(value.cancellation.preserved_result_refs),
            model_cancellation_request_refs=model_refs,
            resource_termination_request_refs=resource_refs,
            pending_physical_acknowledgements=(len(model_refs) + len(resource_refs)),
        )


def _validate_page(*, offset: int, limit: int) -> None:
    if offset < 0:
        raise PipelineLifecyclePolicyError("pagination offset must be non-negative")
    if limit < 1 or limit > MAX_PIPELINE_PAGE_SIZE:
        raise PipelineLifecyclePolicyError(f"pagination limit must be between 1 and {MAX_PIPELINE_PAGE_SIZE}")


def _page[T](
    values: tuple[T, ...],
    *,
    offset: int,
    limit: int,
) -> tuple[T, ...]:
    if offset > len(values):
        raise PipelineLifecyclePolicyError("pagination offset exceeds total")
    return values[offset : offset + limit]


def _finding_codes(
    report: BatchAuditReportV2,
) -> tuple[BatchAuditFindingCodeV2, ...]:
    return tuple(
        sorted(
            {finding.code for finding in report.findings},
            key=lambda value: value.value,
        )
    )


def _is_incomplete(value: PipelineWorkUnitStatusV2) -> bool:
    if value.retry_decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED:
        return True
    if value.lease_state is WorkLeaseStateV2.ACTIVE:
        return True
    if value.lease_state is not None:
        return False
    return value.readiness is None or value.readiness.value in {
        "READY",
        "WAITING",
    }


def _sorted_refs(
    refs: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(refs), key=_ref_key))


def _ref_key(
    ref: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )
