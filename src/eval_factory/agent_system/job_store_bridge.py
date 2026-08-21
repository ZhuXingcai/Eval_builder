from __future__ import annotations

import hashlib
from dataclasses import dataclass

from eval_factory.contracts.approval import (
    UserApprovalPolicy,
)
from eval_factory.contracts.approval_v2 import (
    user_approval_policy_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ItemStatus,
    JobStatus,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedDatasetJobPlanV2,
    ResolvedJobWorkGraphV2,
    StageNameV2,
)
from eval_factory.orchestration.fanout import (
    DatasetJobWorkGraphCompiler,
)
from eval_factory.orchestration.job_store import (
    JobStore,
    RecordNotFoundError,
)
from eval_factory.orchestration.planning import (
    DatasetJobPlanCompiler,
)


class FactoryJobStoreBridgeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryJobStoreTemplate:
    requested_stages: tuple[StageNameV2, ...]
    privacy_profile: str
    model_profiles: tuple[str, ...]
    budget: ResourceBudget
    concurrency: ConcurrencyLimit
    approval_policy: UserApprovalPolicy
    export_target: ExportTarget
    adapter_name: str = "raw-traj-v1"

    def __post_init__(self) -> None:
        stages = self.requested_stages
        positions = [list(StageNameV2).index(stage) for stage in stages]
        if (
            not stages
            or stages[0] is not StageNameV2.TRACE_INDEX
            or stages[-1] is not StageNameV2.RELEASE
            or positions != sorted(set(positions))
            or self.export_target.channel == "PRODUCTION"
        ):
            raise ValueError("Factory Job template requires canonical non-production trace-to-release stages")


@dataclass(frozen=True, slots=True)
class FactoryJobStoreAuthority:
    job_spec: DatasetJobSpecV2
    plan: ResolvedDatasetJobPlanV2
    graph: ResolvedJobWorkGraphV2


class FactoryJobStoreBridge:
    def __init__(
        self,
        *,
        job_store: JobStore,
        template: FactoryJobStoreTemplate,
    ) -> None:
        self.job_store = job_store
        self.template = template

    def prepare(
        self,
        *,
        dataset_run_id: str,
        source_trace_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> FactoryJobStoreAuthority:
        refs = _canonical_source_refs(source_trace_refs)
        spec = self._job_spec(
            dataset_run_id=dataset_run_id,
            source_trace_refs=refs,
            audit=audit,
        )
        plan = DatasetJobPlanCompiler().compile(
            job_spec=spec,
            audit=audit,
        )
        graph = DatasetJobWorkGraphCompiler().compile(
            job_spec=spec,
            resolved_plan=plan,
            audit=audit,
        )
        expected = FactoryJobStoreAuthority(
            job_spec=spec,
            plan=plan,
            graph=graph,
        )
        try:
            stored_spec = self.job_store.get_job_spec(dataset_run_id)
        except RecordNotFoundError:
            self._create(expected)
        else:
            observed = FactoryJobStoreAuthority(
                job_spec=stored_spec,
                plan=self.job_store.get_resolved_job_plan(dataset_run_id),
                graph=self.job_store.get_job_work_graph(dataset_run_id),
            )
            if observed != expected:
                raise FactoryJobStoreBridgeError(
                    "Factory JobStore authority differs from current parent inputs"
                )
        self._validate_running(expected)
        return expected

    def _create(
        self,
        authority: FactoryJobStoreAuthority,
    ) -> None:
        spec = authority.job_spec
        job = self.job_store.create_planned_job(
            spec,
            authority.plan,
        )
        self.job_store.create_job_work_graph(
            authority.graph,
            idempotency_key=(f"create-factory-work-graph:{authority.graph.resolved_job_work_graph_sha256}"),
        )
        self.job_store.transition_job(
            job.job_id,
            JobStatus.RUNNING,
            expected_version=job.row_version,
            idempotency_key=(f"start-factory-job:{spec.canonical_sha256()}"),
        )
        for item in self.job_store.list_items(spec.job_id):
            self.job_store.transition_item(
                item.item_id,
                ItemStatus.RUNNING,
                expected_version=item.row_version,
                idempotency_key=(f"start-factory-item:{item.item_id}"),
            )

    def _validate_running(
        self,
        authority: FactoryJobStoreAuthority,
    ) -> None:
        job = self.job_store.get_job(authority.job_spec.job_id)
        items = self.job_store.list_items(job.job_id)
        if (
            job.status is not JobStatus.RUNNING
            or tuple(item.item_id for item in items) != authority.graph.item_ids
            or any(item.status is not ItemStatus.RUNNING for item in items)
        ):
            raise FactoryJobStoreBridgeError(
                "Factory JobStore runtime state differs from current parent authority"
            )

    def _job_spec(
        self,
        *,
        dataset_run_id: str,
        source_trace_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> DatasetJobSpecV2:
        template = self.template
        traces = tuple(
            TraceSourceRef(
                source_trace_id=reference.object_id,
                source_uri=(f"factory-trace://sha256/{reference.object_sha256}"),
                raw_sha256=reference.object_sha256,
                adapter_name=template.adapter_name,
                adapter_version=(reference.object_version),
                processing_class=("RESTRICTED_TRACE_RAW"),
            )
            for reference in source_trace_refs
        )
        digest = hashlib.sha256(
            (
                dataset_run_id + "|" + "|".join(reference.object_sha256 for reference in source_trace_refs)
            ).encode()
        ).hexdigest()
        return DatasetJobSpecV2(
            job_id=dataset_run_id,
            traces=traces,
            requested_stages=(template.requested_stages),
            privacy_profile=template.privacy_profile,
            model_profiles=template.model_profiles,
            budget=template.budget,
            concurrency=template.concurrency,
            selection_spec_ref=None,
            approval_policy_ref=user_approval_policy_ref(template.approval_policy),
            approval_mode=template.approval_policy.mode,
            enabled_checkpoints=(template.approval_policy.enabled_checkpoints),
            export_target=template.export_target,
            idempotency_key=(f"factory-job-bridge-{digest}"),
            audit=audit,
        )


def _canonical_source_refs(
    values: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    refs = tuple(
        sorted(
            values,
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )
    if not refs or len(refs) != len(set(refs)) or any(value.object_type != "trace-source" for value in refs):
        raise FactoryJobStoreBridgeError("Factory JobStore sources must be unique trace-source refs")
    return refs


__all__ = [
    "FactoryJobStoreAuthority",
    "FactoryJobStoreBridge",
    "FactoryJobStoreBridgeError",
    "FactoryJobStoreTemplate",
]
