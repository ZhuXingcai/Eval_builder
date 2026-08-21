from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from eval_factory.attachment_planning import (
    ArtifactExecutionPlanningResult,
)
from eval_factory.contracts.attachment_v2 import ArtifactExecutionBatchV2
from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryDatasetResultV2,
    R6CanaryExecutionManifestV2,
    R6CanaryObjectCodecV2,
    R6CanaryTraceBindingV2,
    SemanticReviewFanoutV2,
    r6_canary_execution_manifest_v2_ref,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.observability_v2 import (
    BatchAuditReportV2,
)
from eval_factory.contracts.orchestration import (
    ItemStatus,
    JobStatus,
    StageRunStatus,
)
from eval_factory.contracts.orchestration_v2 import (
    ArtifactGroupFanoutV2,
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkLeaseEventKindV2,
    WorkLeaseStateV2,
    WorkLeaseV2,
    WorkReadinessV2,
    WorkUnitScopeV2,
    resolved_work_unit_v2_ref,
)
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
    final_package_manifest_ref,
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.resource_v2 import (
    JobResourcePolicyV2,
    NonModelResourceVectorV2,
    ResourceAdmissionOutcomeV2,
    ResourceUsageV2,
    WorkResourceDemandV2,
)
from eval_factory.orchestration.canary_errors import (
    CanaryDriverError,
    CanaryDriverInjectedCrash,
    CanaryResourceAdmissionError,
)
from eval_factory.orchestration.canary_profile import (
    CanaryHandlerContext,
    CanaryStageHandler,
    R6CanaryLabelOutput,
    R6CanaryStageHandler,
    R6CanaryTaskFixture,
    _artifact_group_outputs,
    _attachment_preparation,
    canary_profile_codec_registry,
)
from eval_factory.orchestration.fanout import (
    SemanticReviewFanoutCompiler,
    StageWorkCompletion,
    WorkReadinessEvaluator,
)
from eval_factory.orchestration.job_store import (
    JobStore,
    RecordNotFoundError,
)
from eval_factory.orchestration.lifecycle import (
    PipelineLifecycleService,
)
from eval_factory.orchestration.observability import (
    BatchObservabilityService,
)
from eval_factory.orchestration.resources import (
    ResourceControlService,
)
from eval_factory.orchestration.runner import TraceIndexStageService
from eval_factory.orchestration.stage_object_store import (
    StageObjectStore,
)
from eval_factory.trace.source_registry import TraceSourceRegistry
from eval_factory.trace.storage import TraceIndexStore


class CanaryDriverFaultPoint(StrEnum):
    BEFORE_HANDLER = "before_handler"
    AFTER_OBJECT_PERSISTENCE = "after_object_persistence"
    BEFORE_LEASE_COMPLETION = "before_lease_completion"
    AFTER_LEASE_COMPLETION = "after_lease_completion"


class CanaryDriverFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: CanaryDriverFaultPoint,
        *,
        work_unit_id: str,
        attempt: int,
    ) -> None: ...


@dataclass(frozen=True)
class StaticCanaryDriverFaultInjector:
    crash_points: frozenset[CanaryDriverFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: CanaryDriverFaultPoint,
        *,
        work_unit_id: str,
        attempt: int,
    ) -> None:
        del work_unit_id, attempt
        if point in self.crash_points:
            raise CanaryDriverInjectedCrash(f"injected canary driver crash at {point.value}")


class CanaryPipelineDriver:
    def __init__(
        self,
        *,
        job_store: JobStore,
        source_registry: TraceSourceRegistry,
        trace_store: TraceIndexStore,
        stage_store: StageObjectStore,
        handler: CanaryStageHandler | None = None,
        fault_injector: CanaryDriverFaultInjector | None = None,
    ) -> None:
        self.job_store = job_store
        self.stage_store = stage_store
        self.handler = handler or R6CanaryStageHandler()
        self.fault_injector = fault_injector
        self.trace_index_service = TraceIndexStageService(
            source_registry=source_registry,
            trace_store=trace_store,
        )
        self.lifecycle = PipelineLifecycleService(job_store)
        self.resources = ResourceControlService(job_store)

    async def run(
        self,
        manifest: R6CanaryExecutionManifestV2,
    ) -> R6CanaryDatasetResultV2:
        canonical = R6CanaryExecutionManifestV2.model_validate(manifest.model_dump(mode="python"))
        for seed in canonical.seed_objects:
            self.stage_store.admit_seed(seed)
        self.lifecycle.create(
            job_spec=canonical.job_spec,
            control=canonical.control,
            audit=canonical.audit,
            idempotency_key=canonical.job_spec.idempotency_key,
        )
        graph = self.job_store.get_job_work_graph(canonical.job_spec.job_id)
        resource_policy = self._ensure_resource_policy(
            graph,
            canonical,
        )
        semantic_by_item = self._ensure_semantic_fanouts(
            graph,
            canonical,
        )
        current = self.job_store.get_job(graph.job_id)
        if current.status is JobStatus.SUCCEEDED:
            report = (
                BatchObservabilityService(self.job_store)
                .refresh_job(
                    job_id=graph.job_id,
                    audit=canonical.audit,
                    idempotency_key=(f"r6-canary-observability:{canonical.manifest_sha256}"),
                )
                .report
            )
            return self._result(canonical, report)
        self.lifecycle.resume(
            job_id=graph.job_id,
            expected_job_version=current.row_version,
            idempotency_key=(f"r6-canary-resume:{canonical.manifest_sha256}:{current.row_version}"),
            offset=0,
            limit=500,
        )
        policy = self.job_store.get_work_control_policy(graph.job_id)

        while True:
            units = self.job_store.list_work_units(job_id=graph.job_id)
            artifact_by_unit = self._artifact_fanouts_by_unit(graph)
            completions = self._stage_completions(
                units,
                graph.job_id,
            )
            self._reconcile_terminal_items(
                units,
                completions,
            )
            active_by_unit = self._active_leases(graph.job_id)
            terminal_unit_ids = self._terminal_unit_ids(graph.job_id)
            terminal_item_ids = {
                item.item_id
                for item in self.job_store.list_items(graph.job_id)
                if item.status
                in {
                    ItemStatus.APPROVED,
                    ItemStatus.REJECTED,
                    ItemStatus.FAILED,
                }
            }
            incomplete = tuple(
                unit
                for unit in units
                if (
                    unit.resolved_work_unit_id not in terminal_unit_ids
                    and unit.item_id not in terminal_item_ids
                )
            )
            if not incomplete:
                break
            progressed = False
            for unit in incomplete:
                semantic = (
                    semantic_by_item.get(unit.item_id or "")
                    if unit.stage is StageNameV2.ITEM_QUALITY
                    else None
                )
                artifact_fanout = artifact_by_unit.get(unit.resolved_work_unit_id)
                active = active_by_unit.get(unit.resolved_work_unit_id)
                is_artifact_parent = (
                    artifact_fanout is not None
                    and resolved_work_unit_v2_ref(unit) == artifact_fanout.parent_attachment_work_unit_ref
                )
                if active is not None and artifact_fanout is not None and is_artifact_parent:
                    planning, batch = self._artifact_join_evidence(
                        artifact_fanout,
                        terminal_unit_ids=terminal_unit_ids,
                    )
                    snapshot = WorkReadinessEvaluator().evaluate(
                        graph=graph,
                        work_unit=unit,
                        stage_completions=(),
                        item_records=self.job_store.list_items(graph.job_id),
                        artifact_group_fanout=artifact_fanout,
                        artifact_execution_planning_result=planning,
                        artifact_execution_batch=batch,
                        audit=canonical.audit,
                    )
                    self.job_store.record_work_readiness(
                        snapshot,
                        idempotency_key=(f"r6-canary-readiness:{snapshot.work_readiness_snapshot_sha256}"),
                    )
                    if snapshot.readiness is not WorkReadinessV2.READY:
                        continue
                if active is None:
                    snapshot = WorkReadinessEvaluator().evaluate(
                        graph=graph,
                        work_unit=unit,
                        stage_completions=(() if artifact_fanout is not None else completions),
                        item_records=self.job_store.list_items(graph.job_id),
                        artifact_group_fanout=artifact_fanout,
                        semantic_review_fanout=semantic,
                        audit=canonical.audit,
                    )
                    self.job_store.record_work_readiness(
                        snapshot,
                        idempotency_key=(f"r6-canary-readiness:{snapshot.work_readiness_snapshot_sha256}"),
                    )
                    if snapshot.readiness is not WorkReadinessV2.READY:
                        continue
                    self._start_item(unit)
                    demand = self._resource_demand(
                        graph=graph,
                        work_unit=unit,
                        resource_policy=resource_policy,
                        manifest=canonical,
                        attempt=1,
                        artifact_group_fanout=artifact_fanout,
                    )
                    pool_head = self.resources.get_pool_head(
                        resource_policy.resource_pool_policy_ref.object_id
                    )
                    admission = self.resources.acquire(
                        graph=graph,
                        work_unit=unit,
                        readiness_snapshot=snapshot,
                        control_policy=policy,
                        job_policy=resource_policy,
                        demand=demand,
                        holder_ref=(canonical.control_plane.worker_principal_ref),
                        retry_decision=None,
                        audit=canonical.audit,
                        idempotency_key=(
                            "r6-canary-resource-acquire:"
                            f"{unit.resolved_work_unit_sha256}:1:{pool_head.row_version}"
                        ),
                    )
                    if (
                        admission.decision.outcome is not ResourceAdmissionOutcomeV2.ADMITTED
                        or admission.lease is None
                        or admission.reservation is None
                    ):
                        raise CanaryResourceAdmissionError(admission.decision.outcome)
                    lease = admission.lease
                    reservation = admission.reservation
                    head_version = 0
                else:
                    lease, head_version = active
                    reservation = self.resources.get_reservation_for_lease(lease.work_lease_id)
                if lease.stage_run_ref is None:
                    if unit.scope is not WorkUnitScopeV2.ARTIFACT_GROUP:
                        raise CanaryDriverError("non-artifact canary lease has no StageRun")
                    stage_run = None
                else:
                    stage_run = self.job_store.get_stage_run(lease.stage_run_ref.object_id)
                dependency_refs = _dependency_outputs(
                    unit,
                    completions,
                    semantic,
                )
                item_refs = _item_result_refs(
                    unit.item_id,
                    completions,
                )
                binding = self._binding_for_unit(
                    canonical,
                    unit,
                    graph=graph,
                    artifact_group_fanout=artifact_fanout,
                )
                self._maybe_raise(
                    CanaryDriverFaultPoint.BEFORE_HANDLER,
                    unit,
                    lease.attempt,
                )
                execution = await self.handler.execute(
                    CanaryHandlerContext(
                        manifest=canonical,
                        binding=binding,
                        graph=graph,
                        work_unit=unit,
                        stage_run=stage_run,
                        stage_store=self.stage_store,
                        job_store=self.job_store,
                        trace_index_service=self.trace_index_service,
                        dependency_output_refs=dependency_refs,
                        item_result_refs=item_refs,
                        resource_control=self.resources,
                        resource_reservation=reservation,
                        artifact_group_fanout=artifact_fanout,
                        private_root=self.stage_store.root,
                    )
                )
                self._maybe_raise(
                    CanaryDriverFaultPoint.AFTER_OBJECT_PERSISTENCE,
                    unit,
                    lease.attempt,
                )
                if execution.deferred:
                    progressed = True
                    break
                self._maybe_raise(
                    CanaryDriverFaultPoint.BEFORE_LEASE_COMPLETION,
                    unit,
                    lease.attempt,
                )
                usage = ResourceUsageV2(
                    observed=(execution.resource_usage or NonModelResourceVectorV2.zero())
                )
                if unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP:
                    self.resources.complete_artifact_group(
                        lease=lease,
                        control_policy=policy,
                        reservation=reservation,
                        holder_ref=lease.holder_ref,
                        expected_lease_version=head_version,
                        event_kind=WorkLeaseEventKindV2.SUCCEEDED,
                        result_refs=execution.output_refs,
                        failure=None,
                        usage=usage,
                        audit=canonical.audit,
                        idempotency_key=(
                            f"r6-canary-resource-complete:{lease.work_lease_id}:{head_version + 1}"
                        ),
                    )
                else:
                    self.resources.complete_stage(
                        lease=lease,
                        control_policy=policy,
                        reservation=reservation,
                        holder_ref=lease.holder_ref,
                        expected_lease_version=head_version,
                        status=StageRunStatus.SUCCEEDED,
                        output_refs=execution.output_refs,
                        failure=None,
                        checkpoint_ref=execution.checkpoint_ref,
                        metrics_ref=None,
                        usage=usage,
                        audit=canonical.audit,
                        idempotency_key=(
                            f"r6-canary-resource-complete:{lease.work_lease_id}:{head_version + 1}"
                        ),
                    )
                self._maybe_raise(
                    CanaryDriverFaultPoint.AFTER_LEASE_COMPLETION,
                    unit,
                    lease.attempt,
                )
                if execution.item_terminal_status is not None:
                    self._finish_item(
                        unit,
                        execution.item_terminal_status,
                    )
                if unit.stage is StageNameV2.ITEM_QUALITY and unit.semantic_review_round is None:
                    self._approve_item(unit)
                progressed = True
                break
            if not progressed:
                raise CanaryDriverError("canary pipeline has incomplete work but no ready or active unit")

        self._complete_job(graph)
        observability = BatchObservabilityService(self.job_store).refresh_job(
            job_id=graph.job_id,
            audit=canonical.audit,
            idempotency_key=(f"r6-canary-observability:{canonical.manifest_sha256}"),
        )
        return self._result(
            canonical,
            observability.report,
        )

    def _ensure_resource_policy(
        self,
        graph: ResolvedJobWorkGraphV2,
        manifest: R6CanaryExecutionManifestV2,
    ) -> JobResourcePolicyV2:
        pool = self.resources.bind_pool_policy(
            resource_domain_ref=(manifest.control_plane.resource_domain_ref),
            capacity=manifest.control_plane.resource_capacity,
            audit=manifest.audit,
            idempotency_key=(
                f"r6-canary-resource-pool:{manifest.control_plane.resource_domain_ref.object_id}"
            ),
        )
        return self.resources.bind_job_policy(
            graph=graph,
            pool_policy=pool,
            audit=manifest.audit,
            idempotency_key=(f"r6-canary-job-resource-policy:{graph.job_id}"),
        )

    def _resource_demand(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        resource_policy: JobResourcePolicyV2,
        manifest: R6CanaryExecutionManifestV2,
        attempt: int,
        artifact_group_fanout: ArtifactGroupFanoutV2 | None,
    ) -> WorkResourceDemandV2:
        capacity = NonModelResourceVectorV2.zero()
        if work_unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP:
            binding = self._binding_for_unit(
                manifest,
                work_unit,
                graph=graph,
                artifact_group_fanout=artifact_group_fanout,
            )
            fixture_refs = {
                seed.object_ref: seed
                for seed in manifest.seed_objects
                if seed.codec is R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE
            }
            matches = tuple(ref for ref in binding.task_fixture_refs if ref in fixture_refs)
            if len(matches) != 1:
                raise CanaryDriverError("ATTACHMENT work has no exact task fixture")
            fixture = self.stage_store.get_as(
                matches[0],
                codec=(R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE),
                model_type=R6CanaryTaskFixture,
            )
            if fixture.text_provider_content is not None:
                storage_bytes = len((fixture.text_provider_content.rstrip() + "\n").encode("utf-8"))
                capacity = NonModelResourceVectorV2(
                    processes=0,
                    renderers=0,
                    network_requests=0,
                    storage_bytes=storage_bytes,
                )
        profile_ref = ObjectRef(
            object_type="resource-execution-profile",
            object_id="resource-execution-profile://r6-canary/r1-r5-v1",
            object_version="v1",
            object_sha256=manifest.manifest_sha256,
        )
        return self.resources.create_demand(
            graph=graph,
            work_unit=work_unit,
            job_policy=resource_policy,
            execution_profile_ref=profile_ref,
            attempt=attempt,
            capacity_units=capacity,
            budget_allowance=capacity,
            termination_required=False,
            audit=manifest.audit,
        )

    def _ensure_semantic_fanouts(
        self,
        graph: ResolvedJobWorkGraphV2,
        manifest: R6CanaryExecutionManifestV2,
    ) -> dict[str, SemanticReviewFanoutV2]:
        result: dict[str, SemanticReviewFanoutV2] = {}
        for parent in graph.work_units:
            if parent.stage is not StageNameV2.ITEM_QUALITY:
                continue
            fanout = SemanticReviewFanoutCompiler().compile(
                graph=graph,
                parent_item_quality_work_unit=parent,
                audit=manifest.audit,
            )
            stored = self.job_store.create_semantic_review_fanout(
                fanout,
                idempotency_key=(f"r6-canary-semantic-fanout:{parent.resolved_work_unit_sha256}"),
            )
            if parent.item_id is None:
                raise CanaryDriverError("ITEM_QUALITY fanout has no Item")
            result[parent.item_id] = stored
        return result

    def _artifact_fanouts_by_unit(
        self,
        graph: ResolvedJobWorkGraphV2,
    ) -> dict[str, ArtifactGroupFanoutV2]:
        result: dict[str, ArtifactGroupFanoutV2] = {}
        for parent in graph.work_units:
            if parent.stage is not StageNameV2.ATTACHMENT:
                continue
            try:
                fanout = self.job_store.get_artifact_group_fanout(parent.resolved_work_unit_id)
            except RecordNotFoundError:
                continue
            result[parent.resolved_work_unit_id] = fanout
            result.update({unit.resolved_work_unit_id: fanout for unit in fanout.group_work_units})
        return result

    def _artifact_join_evidence(
        self,
        fanout: ArtifactGroupFanoutV2,
        *,
        terminal_unit_ids: set[str],
    ) -> tuple[
        ArtifactExecutionPlanningResult | None,
        ArtifactExecutionBatchV2 | None,
    ]:
        if len(fanout.group_work_units) != 1:
            raise CanaryDriverError("text canary requires exactly one artifact group")
        group = fanout.group_work_units[0]
        if group.item_id is None:
            raise CanaryDriverError("artifact group has no Item identity")
        if group.resolved_work_unit_id not in terminal_unit_ids:
            return None, None
        preparation = _attachment_preparation(
            self.stage_store,
            job_id=group.job_id,
            item_id=group.item_id,
        )
        outputs = _artifact_group_outputs(
            self.stage_store,
            job_id=group.job_id,
            item_id=group.item_id,
        )
        if len(outputs) != 1 or outputs[0].work_unit_ref != resolved_work_unit_v2_ref(group):
            raise CanaryDriverError("terminal artifact group has no exact durable journal")
        return (
            preparation.execution_result,
            outputs[0].execution_batch,
        )

    def _stage_completions(
        self,
        units: tuple[ResolvedWorkUnitV2, ...],
        job_id: str,
    ) -> tuple[StageWorkCompletion, ...]:
        by_ref = {resolved_work_unit_v2_ref(unit): unit for unit in units}
        completions: list[StageWorkCompletion] = []
        for run in self.job_store.list_stage_runs(job_id=job_id):
            result = self.job_store.get_stage_result_for_run(run.stage_run_id)
            if result is None:
                continue
            matches = tuple(ref for ref in run.input_refs if ref in by_ref)
            if len(matches) != 1:
                continue
            completions.append(
                StageWorkCompletion(
                    work_unit_ref=matches[0],
                    stage_run=run,
                    stage_result=result,
                )
            )
        return tuple(completions)

    def _active_leases(
        self,
        job_id: str,
    ) -> dict[str, tuple[WorkLeaseV2, int]]:
        result: dict[str, tuple[WorkLeaseV2, int]] = {}
        for lease in self.job_store.list_work_leases(job_id=job_id):
            head = self.job_store.get_work_lease_head(lease.work_lease_id)
            if head.state is WorkLeaseStateV2.ACTIVE:
                result[lease.work_unit_ref.object_id] = (
                    lease,
                    head.lease_version,
                )
        return result

    def _terminal_unit_ids(self, job_id: str) -> set[str]:
        terminal: set[str] = set()
        for lease in self.job_store.list_work_leases(job_id=job_id):
            head = self.job_store.get_work_lease_head(lease.work_lease_id)
            if head.state is not WorkLeaseStateV2.ACTIVE:
                terminal.add(lease.work_unit_ref.object_id)
        return terminal

    def _reconcile_terminal_items(
        self,
        units: tuple[ResolvedWorkUnitV2, ...],
        completions: tuple[StageWorkCompletion, ...],
    ) -> None:
        by_ref = {resolved_work_unit_v2_ref(unit): unit for unit in units}
        for completion in completions:
            unit = by_ref.get(completion.work_unit_ref)
            if unit is None or unit.item_id is None:
                continue
            item = self.job_store.get_item(unit.item_id)
            if item.status is not ItemStatus.RUNNING:
                continue
            if unit.stage is StageNameV2.LABEL and completion.stage_result.status is StageRunStatus.SUCCEEDED:
                output_refs = tuple(
                    ref
                    for ref in completion.stage_result.output_refs
                    if ref.object_type == "r6-canary-label-output"
                )
                if len(output_refs) != 1:
                    raise CanaryDriverError("terminal LABEL result has no exact durable output")
                output = self.stage_store.get_as(
                    output_refs[0],
                    codec=R6CanaryObjectCodecV2.LABEL_OUTPUT,
                    model_type=R6CanaryLabelOutput,
                )
                if output.item_id != unit.item_id:
                    raise CanaryDriverError("terminal LABEL output crosses Item identity")
                if output.merge_result.label_decision.decision.value != "MATCH":
                    self._finish_item(unit, ItemStatus.REJECTED)
                continue
            if (
                unit.stage is StageNameV2.ITEM_QUALITY
                and unit.semantic_review_round is None
                and completion.stage_result.status is StageRunStatus.SUCCEEDED
            ):
                output_refs = tuple(
                    ref
                    for ref in completion.stage_result.output_refs
                    if ref.object_type == "item-quality-compilation-result"
                )
                if len(output_refs) != 1:
                    raise CanaryDriverError("terminal ITEM_QUALITY result has no exact durable output")
                quality = self.stage_store.get_as(
                    output_refs[0],
                    codec=R6CanaryObjectCodecV2.ITEM_QUALITY_OUTPUT,
                    model_type=ItemQualityCompilationResultV2,
                )
                if not quality.quality_report.approvable:
                    raise CanaryDriverError("canary ITEM_QUALITY finalizer is not approvable")
                self._approve_item(unit)

    @staticmethod
    def _binding_for_unit(
        manifest: R6CanaryExecutionManifestV2,
        unit: ResolvedWorkUnitV2,
        *,
        graph: ResolvedJobWorkGraphV2,
        artifact_group_fanout: ArtifactGroupFanoutV2 | None,
    ) -> R6CanaryTraceBindingV2:
        source_trace_ref = unit.source_trace_ref
        if source_trace_ref is None:
            if artifact_group_fanout is None:
                raise CanaryDriverError("source-free work unit has no artifact fanout")
            parents = tuple(
                candidate
                for candidate in graph.work_units
                if resolved_work_unit_v2_ref(candidate)
                == artifact_group_fanout.parent_attachment_work_unit_ref
            )
            if len(parents) != 1:
                raise CanaryDriverError("artifact group has no exact parent work unit")
            source_trace_ref = parents[0].source_trace_ref
        matches = tuple(
            binding for binding in manifest.trace_bindings if binding.source_trace_ref == source_trace_ref
        )
        if len(matches) != 1:
            raise CanaryDriverError("canary work unit has no exact trace binding")
        return matches[0]

    def _start_item(self, unit: ResolvedWorkUnitV2) -> None:
        if unit.item_id is None:
            return
        item = self.job_store.get_item(unit.item_id)
        if item.status is ItemStatus.CANDIDATE:
            self.job_store.transition_item(
                item.item_id,
                ItemStatus.RUNNING,
                expected_version=item.row_version,
                idempotency_key=(f"r6-canary-item-running:{item.item_id}"),
            )

    def _approve_item(self, unit: ResolvedWorkUnitV2) -> None:
        if unit.item_id is None:
            raise CanaryDriverError("ITEM_QUALITY finalizer has no Item")
        item = self.job_store.get_item(unit.item_id)
        if item.status is ItemStatus.RUNNING:
            self.job_store.transition_item(
                item.item_id,
                ItemStatus.APPROVED,
                expected_version=item.row_version,
                idempotency_key=(f"r6-canary-item-approved:{item.item_id}"),
            )

    def _finish_item(
        self,
        unit: ResolvedWorkUnitV2,
        status: ItemStatus,
    ) -> None:
        if unit.item_id is None:
            raise CanaryDriverError("terminal canary result has no Item")
        item = self.job_store.get_item(unit.item_id)
        if item.status is ItemStatus.RUNNING:
            self.job_store.transition_item(
                item.item_id,
                status,
                expected_version=item.row_version,
                idempotency_key=(f"r6-canary-item-{status.value.casefold()}:{item.item_id}"),
            )

    def _complete_job(
        self,
        graph: ResolvedJobWorkGraphV2,
    ) -> None:
        job = self.job_store.get_job(graph.job_id)
        if job.status is JobStatus.SUCCEEDED:
            return
        items = self.job_store.list_items(graph.job_id)
        terminal = {
            ItemStatus.APPROVED,
            ItemStatus.REJECTED,
            ItemStatus.FAILED,
        }
        if (
            not items
            or not any(item.status is ItemStatus.APPROVED for item in items)
            or any(item.status not in terminal for item in items)
        ):
            raise CanaryDriverError("canary Job cannot succeed without approved Items")
        self.job_store.transition_job(
            graph.job_id,
            JobStatus.SUCCEEDED,
            expected_version=job.row_version,
            idempotency_key=f"r6-canary-job-succeeded:{graph.job_id}",
        )

    def _result(
        self,
        manifest: R6CanaryExecutionManifestV2,
        audit_report: BatchAuditReportV2,
    ) -> R6CanaryDatasetResultV2:
        items = self.job_store.list_items(manifest.job_spec.job_id)
        graph = self.job_store.get_job_work_graph(manifest.job_spec.job_id)
        finalizer_refs = {
            resolved_work_unit_v2_ref(unit)
            for unit in graph.work_units
            if (unit.stage is StageNameV2.ITEM_QUALITY and unit.semantic_review_round is None)
        }
        quality_output_refs: list[ObjectRef] = []
        for run in self.job_store.list_stage_runs(
            job_id=manifest.job_spec.job_id,
            stage=StageNameV2.ITEM_QUALITY,
        ):
            if not finalizer_refs.intersection(run.input_refs):
                continue
            result = self.job_store.get_stage_result_for_run(run.stage_run_id)
            if result is None:
                raise CanaryDriverError("ITEM_QUALITY finalizer has no StageResult")
            quality_output_refs.extend(
                ref for ref in result.output_refs if ref.object_type == "item-quality-compilation-result"
            )
        quality_values = tuple(
            self.stage_store.get_as(
                ref,
                codec=R6CanaryObjectCodecV2.ITEM_QUALITY_OUTPUT,
                model_type=ItemQualityCompilationResultV2,
            )
            for ref in quality_output_refs
        )
        quality_refs = tuple(item_quality_compilation_result_ref(value) for value in quality_values)
        package_refs = tuple(
            final_package_manifest_ref(value.final_package_manifest)
            for value in quality_values
            if value.final_package_manifest is not None
        )
        from eval_factory.contracts.observability_v2 import (
            batch_audit_report_v2_ref,
        )

        return R6CanaryDatasetResultV2(
            manifest_ref=(r6_canary_execution_manifest_v2_ref(manifest)),
            job_id=manifest.job_spec.job_id,
            job_status=self.job_store.get_job(manifest.job_spec.job_id).status,
            item_ids=tuple(sorted(item.item_id for item in items)),
            succeeded_item_ids=tuple(
                sorted(item.item_id for item in items if item.status is ItemStatus.APPROVED)
            ),
            blocked_item_ids=tuple(
                sorted(item.item_id for item in items if item.status is ItemStatus.REJECTED)
            ),
            failed_item_ids=tuple(sorted(item.item_id for item in items if item.status is ItemStatus.FAILED)),
            quality_result_refs=tuple(sorted(quality_refs, key=_ref_key)),
            package_manifest_refs=tuple(sorted(package_refs, key=_ref_key)),
            audit_report_ref=batch_audit_report_v2_ref(audit_report),
            audit_outcome=audit_report.outcome,
        )

    def _maybe_raise(
        self,
        point: CanaryDriverFaultPoint,
        unit: ResolvedWorkUnitV2,
        attempt: int,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(
                point,
                work_unit_id=unit.resolved_work_unit_id,
                attempt=attempt,
            )


def create_canary_stage_store(root: Path) -> StageObjectStore:
    return StageObjectStore(
        root,
        registry=canary_profile_codec_registry(),
    )


def _dependency_outputs(
    unit: ResolvedWorkUnitV2,
    completions: tuple[StageWorkCompletion, ...],
    semantic: SemanticReviewFanoutV2 | None,
) -> tuple[ObjectRef, ...]:
    dependencies = set(unit.depends_on_work_unit_refs)
    if semantic is not None and resolved_work_unit_v2_ref(unit) == semantic.parent_item_quality_work_unit_ref:
        dependencies.update(resolved_work_unit_v2_ref(value.work_unit) for value in semantic.round_work)
    return tuple(
        ref
        for completion in completions
        if completion.work_unit_ref in dependencies
        for ref in completion.stage_result.output_refs
    )


def _item_result_refs(
    item_id: str | None,
    completions: tuple[StageWorkCompletion, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(
        ref
        for completion in completions
        if completion.stage_run.item_id == item_id
        for ref in completion.stage_result.output_refs
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )
