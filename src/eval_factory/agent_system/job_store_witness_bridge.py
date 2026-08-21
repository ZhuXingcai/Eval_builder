from __future__ import annotations

import hashlib
from dataclasses import dataclass

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkControlPolicyV2,
    WorkReadinessV2,
    WorkUnitScopeV2,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
)
from eval_factory.orchestration.control import WorkControlService
from eval_factory.orchestration.fanout import (
    StageWorkCompletion,
    WorkReadinessEvaluator,
)
from eval_factory.orchestration.job_store import JobStore


class FactoryJobStoreWitnessBridgeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryItemStageWitnessSource:
    item_id: str
    trace_index_ref: ObjectRef
    safety_ref: ObjectRef
    label_ref: ObjectRef
    task_authoring_ref: ObjectRef
    attachment_ref: ObjectRef
    item_quality_ref: ObjectRef

    def output_for(
        self,
        stage: StageNameV2,
    ) -> ObjectRef:
        values = {
            StageNameV2.TRACE_INDEX: self.trace_index_ref,
            StageNameV2.SAFETY: self.safety_ref,
            StageNameV2.LABEL: self.label_ref,
            StageNameV2.TASK_AUTHORING: self.task_authoring_ref,
            StageNameV2.ATTACHMENT: self.attachment_ref,
            StageNameV2.ITEM_QUALITY: self.item_quality_ref,
        }
        try:
            return values[stage]
        except KeyError as exc:
            raise FactoryJobStoreWitnessBridgeError("item witness stage is unsupported") from exc


class FactoryJobStoreWitnessBridge:
    def __init__(
        self,
        *,
        job_store: JobStore,
    ) -> None:
        self.job_store = job_store
        self.control = WorkControlService(job_store)

    def commit_items(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        sources: tuple[FactoryItemStageWitnessSource, ...],
        audit: ContractAudit,
    ) -> tuple[StageWorkCompletion, ...]:
        self._validate_graph(graph)
        source_item_ids = tuple(value.item_id for value in sources)
        if len(set(source_item_ids)) != len(source_item_ids) or set(source_item_ids) != set(graph.item_ids):
            raise FactoryJobStoreWitnessBridgeError("item witness inventory differs from JobStore graph")
        by_item = {value.item_id: value for value in sources}
        completions = list(self._load_completions(graph))
        policy = self._policy(graph, audit=audit)
        for unit in graph.work_units:
            if unit.scope is not WorkUnitScopeV2.ITEM:
                continue
            if unit.item_id is None:
                raise FactoryJobStoreWitnessBridgeError("item work unit lacks Item authority")
            output_ref = by_item[unit.item_id].output_for(unit.stage)
            completion = self._complete(
                graph=graph,
                unit=unit,
                output_refs=(output_ref,),
                completions=tuple(completions),
                policy=policy,
                audit=audit,
            )
            if completion not in completions:
                completions.append(completion)
        return tuple(completions)

    def commit_batch(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        batch_quality_ref: ObjectRef,
        audit: ContractAudit,
    ) -> StageWorkCompletion:
        self._validate_graph(graph)
        units = tuple(
            unit
            for unit in graph.work_units
            if (unit.scope is WorkUnitScopeV2.JOB and unit.stage is StageNameV2.BATCH_QUALITY)
        )
        if len(units) != 1:
            raise FactoryJobStoreWitnessBridgeError("JobStore graph requires one BatchQuality work unit")
        completions = self._load_completions(graph)
        return self._complete(
            graph=graph,
            unit=units[0],
            output_refs=(batch_quality_ref,),
            completions=completions,
            policy=self._policy(graph, audit=audit),
            audit=audit,
        )

    def _complete(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        unit: ResolvedWorkUnitV2,
        output_refs: tuple[ObjectRef, ...],
        completions: tuple[StageWorkCompletion, ...],
        policy: WorkControlPolicyV2,
        audit: ContractAudit,
    ) -> StageWorkCompletion:
        unit_ref = resolved_work_unit_v2_ref(unit)
        existing = tuple(value for value in completions if value.work_unit_ref == unit_ref)
        if existing:
            if (
                len(existing) != 1
                or existing[0].stage_result.status is not StageRunStatus.SUCCEEDED
                or existing[0].stage_result.output_refs != output_refs
            ):
                raise FactoryJobStoreWitnessBridgeError("persisted JobStore StageResult witness differs")
            return existing[0]
        readiness = WorkReadinessEvaluator().evaluate(
            graph=graph,
            work_unit=unit,
            stage_completions=completions,
            item_records=self.job_store.list_items(graph.job_id),
            audit=audit,
        )
        if readiness.readiness is not WorkReadinessV2.READY:
            raise FactoryJobStoreWitnessBridgeError("JobStore work unit is not ready for witness commit")
        readiness = self.job_store.record_work_readiness(
            readiness,
            idempotency_key=(f"factory-witness-ready:{unit.resolved_work_unit_id}"),
        )
        lease = self.control.acquire(
            graph=graph,
            work_unit=unit,
            readiness_snapshot=readiness,
            policy=policy,
            holder_ref=_holder_ref(graph.job_id),
            retry_decision=None,
            audit=audit,
            idempotency_key=(f"factory-witness-acquire:{unit.resolved_work_unit_id}"),
        )
        self.control.complete_stage(
            lease=lease,
            policy=policy,
            holder_ref=_holder_ref(graph.job_id),
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=output_refs,
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            audit=audit,
            idempotency_key=(f"factory-witness-complete:{unit.resolved_work_unit_id}"),
        )
        if lease.stage_run_ref is None:
            raise FactoryJobStoreWitnessBridgeError("controlled work lease lacks StageRun authority")
        stage_run = self.job_store.get_stage_run(lease.stage_run_ref.object_id)
        stage_result = self.job_store.get_stage_result_for_run(stage_run.stage_run_id)
        if (
            stage_result is None
            or stage_result.status is not StageRunStatus.SUCCEEDED
            or stage_result.output_refs != output_refs
        ):
            raise FactoryJobStoreWitnessBridgeError("JobStore StageResult witness commit is incomplete")
        return StageWorkCompletion(
            work_unit_ref=unit_ref,
            stage_run=stage_run,
            stage_result=stage_result,
        )

    def _policy(
        self,
        graph: ResolvedJobWorkGraphV2,
        *,
        audit: ContractAudit,
    ) -> WorkControlPolicyV2:
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        return self.control.bind_policy(
            graph=graph,
            lease_duration_seconds=300,
            heartbeat_extension_seconds=60,
            max_attempts=1,
            retry_delay_seconds=(),
            retry_lease_expiry=False,
            audit=audit,
            idempotency_key=(f"factory-witness-policy:{graph_ref.object_sha256}"),
        )

    def _load_completions(
        self,
        graph: ResolvedJobWorkGraphV2,
    ) -> tuple[StageWorkCompletion, ...]:
        values = []
        for lease in self.job_store.list_work_leases(
            job_id=graph.job_id,
        ):
            if lease.stage_run_ref is None:
                continue
            result = self.job_store.get_stage_result_for_run(lease.stage_run_ref.object_id)
            if result is None:
                continue
            values.append(
                StageWorkCompletion(
                    work_unit_ref=lease.work_unit_ref,
                    stage_run=self.job_store.get_stage_run(lease.stage_run_ref.object_id),
                    stage_result=result,
                )
            )
        refs = tuple(value.work_unit_ref for value in values)
        if len(refs) != len(set(refs)):
            raise FactoryJobStoreWitnessBridgeError(
                "JobStore has multiple terminal witnesses for one work unit"
            )
        return tuple(
            sorted(
                values,
                key=lambda value: _ref_key(value.work_unit_ref),
            )
        )

    def _validate_graph(
        self,
        graph: ResolvedJobWorkGraphV2,
    ) -> None:
        current = self.job_store.get_job_work_graph(graph.job_id)
        if current != graph or resolved_job_work_graph_v2_ref(current) != resolved_job_work_graph_v2_ref(
            graph
        ):
            raise FactoryJobStoreWitnessBridgeError("JobStore witness graph is not current")


def _holder_ref(job_id: str) -> ObjectRef:
    digest = hashlib.sha256(job_id.encode()).hexdigest()
    return ObjectRef(
        object_type="worker-principal",
        object_id=f"worker-principal://factory-witness/{digest}",
        object_version="v1",
        object_sha256=digest,
    )


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "FactoryItemStageWitnessSource",
    "FactoryJobStoreWitnessBridge",
    "FactoryJobStoreWitnessBridgeError",
]
