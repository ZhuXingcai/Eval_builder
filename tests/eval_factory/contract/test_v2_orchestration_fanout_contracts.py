from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkDependencyJoinModeV2,
    WorkReadinessSnapshotV2,
    WorkReadinessV2,
    WorkUnitScopeV2,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
)
from eval_factory.orchestration.fanout import DatasetJobWorkGraphCompiler
from eval_factory.orchestration.planning import DatasetJobPlanCompiler

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)
FULL_STAGES = (
    StageNameV2.TRACE_INDEX,
    StageNameV2.SAFETY,
    StageNameV2.LABEL,
    StageNameV2.TASK_AUTHORING,
    StageNameV2.ATTACHMENT,
    StageNameV2.ITEM_QUALITY,
    StageNameV2.BATCH_QUALITY,
    StageNameV2.RELEASE,
)


def _ref(name: str) -> ObjectRef:
    return ObjectRef(
        object_type=name,
        object_id=f"{name}://r6-02/contract",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit(
    *,
    created_at: datetime = NOW,
    created_by: str = "r6-02-contract-test",
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by=created_by,
        governing_versions=(VersionBinding(component="eval-factory-spec", version="approved-v2"),),
    )


def _trace(suffix: str) -> TraceSourceRef:
    return TraceSourceRef(
        source_trace_id=f"source-trace://r6-02/{suffix}",
        source_uri=f"raw-traj://r6-02/{suffix}",
        raw_sha256=(suffix[0] * 64),
        adapter_name="raw-traj-v1",
        adapter_version="v1",
        processing_class="RESTRICTED_TRACE_RAW",
    )


def _spec() -> DatasetJobSpecV2:
    return DatasetJobSpecV2(
        job_id="job://r6-02/contract",
        traces=(_trace("a"), _trace("b")),
        requested_stages=FULL_STAGES,
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=10,
            max_model_tokens=1000,
            max_processes=2,
            max_renderers=1,
            max_network_requests=5,
            max_storage_bytes=1024 * 1024,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=1,
            items=2,
        ),
        approval_policy_ref=_ref("user-approval-policy"),
        approval_mode=ApprovalMode.NONE,
        enabled_checkpoints=frozenset(),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        idempotency_key="create-job-r6-02-contract",
        audit=_audit(),
    )


def _graph() -> ResolvedJobWorkGraphV2:
    spec = _spec()
    plan = DatasetJobPlanCompiler().compile(job_spec=spec, audit=_audit())
    return DatasetJobWorkGraphCompiler().compile(
        job_spec=spec,
        resolved_plan=plan,
        audit=_audit(),
    )


def test_work_graph_contract_is_strict_frozen_and_round_trips() -> None:
    graph = _graph()

    assert ResolvedJobWorkGraphV2.model_validate_json(graph.model_dump_json()) == graph
    assert resolved_job_work_graph_v2_ref(graph).object_sha256 == (graph.resolved_job_work_graph_sha256)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResolvedJobWorkGraphV2.model_validate(
            {
                **graph.model_dump(mode="python"),
                "trace_text": "must-not-cross-the-contract",
            }
        )
    with pytest.raises(ValidationError, match="frozen"):
        graph.policy_version = "changed"  # type: ignore[misc]


def test_work_unit_scope_matrix_rejects_cross_scope_fields() -> None:
    graph = _graph()
    item_unit = next(unit for unit in graph.work_units if unit.scope is WorkUnitScopeV2.ITEM)
    values = item_unit.model_dump(mode="python")
    values["artifact_execution_group_ref"] = _ref("artifact-execution-group")

    with pytest.raises(ValidationError, match="ITEM work unit"):
        ResolvedWorkUnitV2.model_validate(values)

    job_unit = next(unit for unit in graph.work_units if unit.scope is WorkUnitScopeV2.JOB)
    values = job_unit.model_dump(mode="python")
    values["item_id"] = graph.item_ids[0]
    with pytest.raises(ValidationError, match="JOB work unit"):
        ResolvedWorkUnitV2.model_validate(values)


def test_work_graph_rejects_stale_unit_and_graph_identity() -> None:
    graph = _graph()
    unit = graph.work_units[0]
    stale_unit = unit.model_copy(update={"resolved_work_unit_sha256": "b" * 64})
    payload = graph.model_dump(mode="python")
    payload["work_units"] = (stale_unit, *graph.work_units[1:])

    with pytest.raises(ValidationError, match="work unit identity is stale"):
        ResolvedJobWorkGraphV2.model_validate(payload)

    payload = graph.model_dump(mode="python")
    payload["resolved_job_work_graph_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="work graph identity is stale"):
        ResolvedJobWorkGraphV2.model_validate(payload)


def test_readiness_contract_rejects_overlapping_partitions() -> None:
    graph = _graph()
    target = next(
        unit
        for unit in graph.work_units
        if unit.scope is WorkUnitScopeV2.ITEM and unit.stage is StageNameV2.SAFETY
    )
    dependency = target.depends_on_work_unit_refs[0]
    result_ref = ObjectRef(
        object_type="stage-result",
        object_id="stage-result://r6-02/contract",
        object_version="record/v1",
        object_sha256=HASH,
    )
    base = {
        "work_readiness_snapshot_id": "work-readiness-snapshot://pending",
        "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(graph),
        "artifact_group_fanout_ref": None,
        "work_unit_ref": resolved_work_unit_v2_ref(target),
        "readiness": WorkReadinessV2.WAITING,
        "succeeded_dependency_result_refs": (result_ref,),
        "retryable_dependency_result_refs": (result_ref,),
        "terminal_non_success_result_refs": (),
        "waiting_dependency_work_unit_refs": (dependency,),
        "skipped_item_refs": (),
        "policy_version": graph.policy_version,
        "work_readiness_snapshot_sha256": "0" * 64,
        "audit": _audit(),
    }

    with pytest.raises(ValidationError, match="partitions must be disjoint"):
        WorkReadinessSnapshotV2.model_validate(base)


def test_readiness_contract_rejects_untyped_partition_refs() -> None:
    graph = _graph()
    target = next(
        unit
        for unit in graph.work_units
        if unit.scope is WorkUnitScopeV2.ITEM and unit.stage is StageNameV2.SAFETY
    )
    base = {
        "work_readiness_snapshot_id": "work-readiness-snapshot://pending",
        "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(graph),
        "artifact_group_fanout_ref": None,
        "work_unit_ref": resolved_work_unit_v2_ref(target),
        "readiness": WorkReadinessV2.READY,
        "succeeded_dependency_result_refs": (_ref("untyped-result"),),
        "retryable_dependency_result_refs": (),
        "terminal_non_success_result_refs": (),
        "waiting_dependency_work_unit_refs": (),
        "skipped_item_refs": (),
        "policy_version": graph.policy_version,
        "work_readiness_snapshot_sha256": "0" * 64,
        "audit": _audit(),
    }

    with pytest.raises(
        ValidationError,
        match="result partitions contain an invalid ref",
    ):
        WorkReadinessSnapshotV2.model_validate(base)


def test_audit_actor_and_time_do_not_change_work_identity() -> None:
    spec = _spec()
    plan = DatasetJobPlanCompiler().compile(job_spec=spec, audit=_audit())
    compiler = DatasetJobWorkGraphCompiler()
    first = compiler.compile(job_spec=spec, resolved_plan=plan, audit=_audit())
    second = compiler.compile(
        job_spec=spec,
        resolved_plan=plan,
        audit=_audit(
            created_at=datetime(2030, 1, 1, tzinfo=UTC),
            created_by="different-r6-02-operator",
        ),
    )

    assert first.audit != second.audit
    assert first.resolved_job_work_graph_id == second.resolved_job_work_graph_id
    assert first.resolved_job_work_graph_sha256 == second.resolved_job_work_graph_sha256
    assert [
        (
            resolved_work_unit_v2_ref(unit),
            unit.join_mode is WorkDependencyJoinModeV2.ALL_TERMINAL,
        )
        for unit in first.work_units
    ] == [
        (
            resolved_work_unit_v2_ref(unit),
            unit.join_mode is WorkDependencyJoinModeV2.ALL_TERMINAL,
        )
        for unit in second.work_units
    ]
