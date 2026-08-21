from __future__ import annotations

import hashlib
import json

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    StageNameV2,
    WorkControlPolicyV2,
    dataset_job_spec_v2_ref,
    resolved_job_work_graph_v2_ref,
)
from eval_factory.contracts.scheduler_load_v2 import (
    SCHEDULER_LOAD_JOB_COUNT,
    SchedulerLoadPolicyV2,
    validate_scheduler_load_policy_v2_identity,
)
from eval_factory.orchestration.fanout import DatasetJobWorkGraphCompiler
from eval_factory.orchestration.planning import DatasetJobPlanCompiler
from eval_factory.readiness.scheduler_load_models import (
    PreparedSchedulerLoadCase,
    SchedulerLoadMemberV1,
    SchedulerLoadWorkloadV1,
)


class SchedulerLoadBuilderError(RuntimeError):
    pass


class SchedulerLoadPolicyError(SchedulerLoadBuilderError):
    pass


class SchedulerLoadBuilder:
    def compile_workload(
        self,
        *,
        policy: SchedulerLoadPolicyV2,
        audit: ContractAudit,
    ) -> SchedulerLoadWorkloadV1:
        _validate_policy(policy)
        policy_ref = policy.to_ref()
        members = tuple(
            SchedulerLoadMemberV1.create(
                ordinal=ordinal,
                descriptor_sha256=_descriptor_sha256(
                    policy_ref=policy_ref,
                    workload_seed=policy.workload_seed,
                    ordinal=ordinal,
                ),
                assigned_fault_point=policy.fault_points[ordinal % len(policy.fault_points)],
            )
            for ordinal in range(SCHEDULER_LOAD_JOB_COUNT)
        )
        return SchedulerLoadWorkloadV1.create(
            policy_ref=policy_ref,
            members=members,
            audit=audit,
        )

    def prepare_cases(
        self,
        workload: SchedulerLoadWorkloadV1,
        *,
        policy: SchedulerLoadPolicyV2,
        audit: ContractAudit,
    ) -> tuple[PreparedSchedulerLoadCase, ...]:
        _validate_workload(workload, policy=policy)
        return tuple(
            self.prepare_case(
                member,
                workload=workload,
                policy=policy,
                audit=audit,
            )
            for member in workload.members
        )

    @staticmethod
    def prepare_case(
        member: SchedulerLoadMemberV1,
        *,
        workload: SchedulerLoadWorkloadV1,
        policy: SchedulerLoadPolicyV2,
        audit: ContractAudit,
    ) -> PreparedSchedulerLoadCase:
        _validate_workload(workload, policy=policy)
        if member not in workload.members:
            raise SchedulerLoadPolicyError("scheduler load member is not in the workload")
        policy_ref = policy.to_ref()
        workload_ref = workload.to_ref()
        member_ref = member.to_ref()
        digest = member.case_key.rsplit("/", 1)[-1]
        approval_ref = ObjectRef(
            object_type="user-approval-policy",
            object_id="user-approval-policy://scheduler-load/none/v2",
            object_version="v2",
            object_sha256=policy.policy_sha256,
        )
        trace = TraceSourceRef(
            source_trace_id=f"source-trace://scheduler-load/{digest}",
            source_uri=f"scheduler-load://descriptor/{member.descriptor_sha256}",
            raw_sha256=member.descriptor_sha256,
            adapter_name="scheduler-load",
            adapter_version="r8-05-v1",
            processing_class="RESTRICTED_TRACE_RAW",
        )
        spec_audit = audit.model_copy(
            update={"input_refs": _sorted_refs((policy_ref, workload_ref, member_ref, approval_ref))}
        )
        spec = DatasetJobSpecV2(
            job_id=f"job://scheduler-load/{digest}",
            traces=(trace,),
            requested_stages=(StageNameV2.TRACE_INDEX,),
            privacy_profile="scheduler-load-private",
            model_profiles=(),
            budget=ResourceBudget(
                max_model_requests=0,
                max_model_tokens=0,
                max_processes=0,
                max_renderers=0,
                max_network_requests=0,
                max_storage_bytes=0,
            ),
            concurrency=ConcurrencyLimit(
                model_requests=1,
                processes=1,
                renderers=1,
                network_requests=1,
                artifacts_per_item=1,
                items=1,
            ),
            selection_spec_ref=None,
            approval_policy_ref=approval_ref,
            approval_mode=ApprovalMode.NONE,
            enabled_checkpoints=frozenset(),
            export_target=ExportTarget(
                profile="LH",
                profile_version="scheduler-load-only",
                channel="CANARY",
                registry="registry://scheduler-load/disabled",
            ),
            idempotency_key=f"scheduler-load-create-{digest}",
            audit=spec_audit,
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
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        control_policy = WorkControlPolicyV2.create(
            resolved_job_work_graph_ref=graph_ref,
            lease_duration_seconds=policy.lease_duration_seconds,
            heartbeat_extension_seconds=policy.heartbeat_extension_seconds,
            max_attempts=policy.max_attempts,
            retry_delay_seconds=policy.retry_delay_seconds,
            retry_lease_expiry=policy.retry_lease_expiry,
            audit=_control_audit(audit, graph_ref=graph_ref),
        )
        if len(graph.work_units) != 1:
            raise SchedulerLoadPolicyError("scheduler load graph must contain one work unit")
        holder_ref = ObjectRef(
            object_type="worker-principal",
            object_id=f"worker-principal://scheduler-load/{digest}",
            object_version="v1",
            object_sha256=_payload_sha256(
                {
                    "member_ref": member_ref,
                    "policy_ref": policy_ref,
                    "role": "scheduler-load-worker",
                }
            ),
        )
        output_ref = ObjectRef(
            object_type="scheduler-load-output",
            object_id=f"scheduler-load-output://sha256/{member.descriptor_sha256}",
            object_version="private-v1",
            object_sha256=_payload_sha256(
                {
                    "member_ref": member_ref,
                    "workload_ref": workload_ref,
                    "policy_ref": policy_ref,
                    "kind": "synthetic-trace-index-completion",
                }
            ),
        )
        return PreparedSchedulerLoadCase(
            member=member,
            job_spec=spec,
            dataset_job_spec_ref=dataset_job_spec_v2_ref(spec),
            resolved_plan=plan,
            work_graph=graph,
            control_policy=control_policy,
            root_work_unit=graph.work_units[0],
            holder_ref=holder_ref,
            output_ref=output_ref,
        )


def _validate_policy(policy: SchedulerLoadPolicyV2) -> None:
    try:
        validate_scheduler_load_policy_v2_identity(policy)
    except ValueError as exc:
        raise SchedulerLoadPolicyError("scheduler load policy is stale") from exc


def _validate_workload(
    workload: SchedulerLoadWorkloadV1,
    *,
    policy: SchedulerLoadPolicyV2,
) -> None:
    _validate_policy(policy)
    try:
        workload_ref = workload.to_ref()
    except ValueError as exc:
        raise SchedulerLoadPolicyError("scheduler load workload is stale") from exc
    del workload_ref
    if workload.policy_ref != policy.to_ref():
        raise SchedulerLoadPolicyError("scheduler load workload policy binding is stale")


def _control_audit(
    audit: ContractAudit,
    *,
    graph_ref: ObjectRef,
) -> ContractAudit:
    bindings = tuple(
        sorted(
            (
                *(binding for binding in audit.governing_versions if binding.component != "work-control"),
                VersionBinding(
                    component="work-control",
                    version="work-control/r6-05-v1",
                ),
            ),
            key=lambda binding: (
                binding.component,
                binding.version,
                binding.sha256 or "",
            ),
        )
    )
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=bindings,
        input_refs=(graph_ref,),
    )


def _descriptor_sha256(
    *,
    policy_ref: ObjectRef,
    workload_seed: str,
    ordinal: int,
) -> str:
    return _payload_sha256(
        {
            "policy_ref": policy_ref,
            "workload_seed": workload_seed,
            "ordinal": ordinal,
            "kind": "scheduler-load-synthetic-descriptor",
        }
    )


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _sorted_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            set(refs),
            key=lambda ref: (
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )
    )


__all__ = [
    "SchedulerLoadBuilder",
    "SchedulerLoadBuilderError",
    "SchedulerLoadPolicyError",
]
