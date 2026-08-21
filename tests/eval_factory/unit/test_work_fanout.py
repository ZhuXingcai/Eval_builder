from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_artifact_execution import (
    _audit as _artifact_audit,
)
from test_artifact_execution import (
    _ControlledExecutionFacade,
    _definition,
    _execution_plan,
)
from test_job_planning import BASE_CHAIN, _audit, _spec

from eval_factory.attachment_planning import (
    ArtifactExecutionPlanningOutcome,
    ArtifactExecutionPlanningResult,
    ArtifactGroupExecutor,
)
from eval_factory.contracts.approval import ApprovalCheckpoint, ApprovalMode
from eval_factory.contracts.attachment_v2 import (
    ArtifactExecutionPlanV2,
    artifact_execution_batch_ref,
    artifact_execution_plan_ref,
    artifact_execution_receipt_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    FailureClass,
    FailureRecord,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.orchestration import ItemStatus, StageRunStatus, TraceSourceRef
from eval_factory.contracts.orchestration_v2 import (
    R6_WORK_CONTROL_FAILURE_MESSAGE,
    ArtifactGroupFanoutV2,
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkDependencyJoinModeV2,
    WorkLeaseEventKindV2,
    WorkLeaseEventV2,
    WorkReadinessV2,
    WorkRetryDecisionKindV2,
    WorkRetryDecisionV2,
    WorkUnitScopeV2,
    resolved_work_unit_v2_ref,
    work_lease_event_v2_ref,
    work_retry_decision_v2_ref,
)
from eval_factory.orchestration.fanout import (
    ArtifactGroupFanoutCompiler,
    DatasetJobWorkGraphCompiler,
    StageWorkCompletion,
    WorkFanoutPolicyError,
    WorkReadinessEvaluator,
)
from eval_factory.orchestration.models import (
    ItemRecord,
    StageResultRecord,
    StageRunRecord,
    stage_result_record_carried_sha256,
    stage_result_record_ref,
)
from eval_factory.orchestration.planning import DatasetJobPlanCompiler

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/orchestration" / "r6-02-work-fanout-v1.json"


def _gold(case_id: str) -> dict[str, object]:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return next(case for case in payload["cases"] if case["case_id"] == case_id)


def _trace(suffix: str) -> TraceSourceRef:
    return TraceSourceRef(
        source_trace_id=f"source-trace://r6-02/{suffix}",
        source_uri=f"raw-traj://r6-02/{suffix}",
        raw_sha256=suffix[0] * 64,
        adapter_name="raw-traj-v1",
        adapter_version="v1",
        processing_class="RESTRICTED_TRACE_RAW",
    )


def _two_trace_spec(
    requested_stages: tuple[StageNameV2, ...] = BASE_CHAIN,
    *,
    approval_mode: ApprovalMode = ApprovalMode.NONE,
    enabled_checkpoints: frozenset[ApprovalCheckpoint] = frozenset(),
) -> object:
    base = _spec(
        requested_stages,
        approval_mode=approval_mode,
        enabled_checkpoints=enabled_checkpoints,
        job_id="job://r6-02/fanout",
        idempotency_key="create-job-r6-02-fanout",
    )
    return type(base).model_validate(
        {
            **base.model_dump(mode="python"),
            "traces": (_trace("a"), _trace("b")),
        }
    )


def _graph(
    requested_stages: tuple[StageNameV2, ...] = BASE_CHAIN,
    *,
    approval_mode: ApprovalMode = ApprovalMode.NONE,
    enabled_checkpoints: frozenset[ApprovalCheckpoint] = frozenset(),
) -> ResolvedJobWorkGraphV2:
    spec = _two_trace_spec(
        requested_stages,
        approval_mode=approval_mode,
        enabled_checkpoints=enabled_checkpoints,
    )
    plan = DatasetJobPlanCompiler().compile(job_spec=spec, audit=_audit())
    return DatasetJobWorkGraphCompiler().compile(
        job_spec=spec,
        resolved_plan=plan,
        audit=_audit(),
    )


def _planning_result(
    plan: ArtifactExecutionPlanV2,
) -> ArtifactExecutionPlanningResult:
    return ArtifactExecutionPlanningResult(
        outcome=ArtifactExecutionPlanningOutcome.PLANNED,
        world_ledger_snapshot=plan.world_ledger_snapshot,
        execution_plan=plan,
        audit=_artifact_audit(artifact_execution_plan_ref(plan)),
    )


def _unit(
    graph: ResolvedJobWorkGraphV2,
    stage: StageNameV2,
    *,
    item_id: str | None = None,
    scope: WorkUnitScopeV2 | None = None,
) -> ResolvedWorkUnitV2:
    matches = tuple(
        unit
        for unit in graph.work_units
        if unit.stage is stage
        and (item_id is None or unit.item_id == item_id)
        and (scope is None or unit.scope is scope)
    )
    assert len(matches) == 1
    return matches[0]


def _completion(
    unit: ResolvedWorkUnitV2,
    *,
    status: StageRunStatus,
    retryable: bool = False,
    suffix: str,
) -> StageWorkCompletion:
    run = StageRunRecord(
        stage_run_id=f"stage-run://r6-02/{suffix}",
        job_id=unit.job_id,
        item_id=unit.item_id,
        stage=unit.stage,
        attempt=1,
        status=status,
        principal_ref=ObjectRef(
            object_type="stage-principal",
            object_id=f"stage-principal://r6-02/{suffix}",
            object_version="v1",
            object_sha256=HASH,
        ),
        input_refs=(resolved_work_unit_v2_ref(unit),),
        row_version=1,
        idempotency_key=f"stage-run-r6-02-{suffix}",
        created_at=NOW,
        started_at=NOW,
        ended_at=NOW,
    )
    failure = (
        None
        if status is StageRunStatus.SUCCEEDED
        else FailureRecord(
            failure_class=FailureClass.INTERNAL,
            code=f"r6-02-{suffix}-failure",
            message="Typed fanout test failure.",
            retryable=retryable,
        )
    )
    result = StageResultRecord(
        stage_result_id=f"stage-result://r6-02/{suffix}",
        stage_run_id=run.stage_run_id,
        stage_run_version=run.row_version,
        status=status,
        failure=failure,
        result_sha256="0" * 64,
        created_at=NOW,
    )
    result = result.model_copy(
        update={
            "result_sha256": stage_result_record_carried_sha256(result),
        }
    )
    return StageWorkCompletion(
        work_unit_ref=resolved_work_unit_v2_ref(unit),
        stage_run=run,
        stage_result=result,
    )


def _control_audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r6-05-fanout-test",
        governing_versions=(
            VersionBinding(
                component="work-control",
                version="work-control/r6-05-v1",
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


def _retry_decision(
    unit: ResolvedWorkUnitV2,
    completion: StageWorkCompletion,
    *,
    decision: WorkRetryDecisionKindV2,
) -> WorkRetryDecisionV2:
    unit_ref = resolved_work_unit_v2_ref(unit)
    result_ref = stage_result_record_ref(completion.stage_result)
    event_ref = ObjectRef(
        object_type="work-lease-event",
        object_id=f"work-lease-event://r6-05/{decision.value.lower()}",
        object_version="v2",
        object_sha256=HASH,
    )
    policy_ref = ObjectRef(
        object_type="work-control-policy",
        object_id="work-control-policy://r6-05/fanout",
        object_version="v2",
        object_sha256=HASH,
    )
    return WorkRetryDecisionV2.create(
        work_unit_ref=unit_ref,
        prior_lease_event_ref=event_ref,
        prior_result_refs=(result_ref,),
        work_control_policy_ref=policy_ref,
        decision=decision,
        completed_attempt=1,
        next_attempt=(2 if decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED else None),
        eligible_at=(NOW if decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED else None),
        audit=_control_audit(unit_ref, event_ref, result_ref, policy_ref),
    )


def _terminal_control_event(unit: ResolvedWorkUnitV2) -> WorkLeaseEventV2:
    unit_ref = resolved_work_unit_v2_ref(unit)
    lease_ref = ObjectRef(
        object_type="work-lease",
        object_id="work-lease://r6-05/artifact-expired",
        object_version="v2",
        object_sha256=HASH,
    )
    policy_ref = ObjectRef(
        object_type="work-control-policy",
        object_id="work-control-policy://r6-05/fanout",
        object_version="v2",
        object_sha256=HASH,
    )
    return WorkLeaseEventV2.create(
        work_lease_ref=lease_ref,
        work_unit_ref=unit_ref,
        event_kind=WorkLeaseEventKindV2.EXPIRED,
        lease_version=1,
        fencing_token=1,
        effective_expires_at=None,
        result_refs=(),
        failure=FailureRecord(
            failure_class=FailureClass.INTERNAL,
            code="artifact-group-lease-expired",
            message=R6_WORK_CONTROL_FAILURE_MESSAGE,
            retryable=False,
        ),
        work_control_policy_ref=policy_ref,
        audit=_control_audit(lease_ref, unit_ref, policy_ref),
    )


def _artifact_retry_decision(
    unit: ResolvedWorkUnitV2,
    batch_ref: ObjectRef,
    receipt_ref: ObjectRef,
    *,
    decision: WorkRetryDecisionKindV2,
) -> WorkRetryDecisionV2:
    unit_ref = resolved_work_unit_v2_ref(unit)
    event_ref = ObjectRef(
        object_type="work-lease-event",
        object_id=f"work-lease-event://r6-05/artifact-{decision.value.lower()}",
        object_version="v2",
        object_sha256=HASH,
    )
    policy_ref = ObjectRef(
        object_type="work-control-policy",
        object_id="work-control-policy://r6-05/artifact-fanout",
        object_version="v2",
        object_sha256=HASH,
    )
    prior_refs = tuple(
        sorted(
            (batch_ref, receipt_ref),
            key=lambda ref: (
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )
    )
    return WorkRetryDecisionV2.create(
        work_unit_ref=unit_ref,
        prior_lease_event_ref=event_ref,
        prior_result_refs=prior_refs,
        work_control_policy_ref=policy_ref,
        decision=decision,
        completed_attempt=1,
        next_attempt=(2 if decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED else None),
        eligible_at=(NOW if decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED else None),
        audit=_control_audit(unit_ref, event_ref, policy_ref, *prior_refs),
    )


def test_compile_two_trace_no_checkpoint_graph_has_exact_scope_and_joins() -> None:
    graph = _graph()
    gold = _gold("two-item-no-checkpoint-graph")

    assert len(graph.item_ids) == 2
    assert len(graph.work_units) == 14
    assert sum(unit.scope is WorkUnitScopeV2.ITEM for unit in graph.work_units) == 12
    assert sum(unit.scope is WorkUnitScopeV2.JOB for unit in graph.work_units) == 2

    for item_id in graph.item_ids:
        trace = _unit(graph, StageNameV2.TRACE_INDEX, item_id=item_id)
        safety = _unit(graph, StageNameV2.SAFETY, item_id=item_id)
        quality = _unit(graph, StageNameV2.ITEM_QUALITY, item_id=item_id)
        assert trace.depends_on_work_unit_refs == ()
        assert safety.depends_on_work_unit_refs == (resolved_work_unit_v2_ref(trace),)
        assert quality.join_mode is WorkDependencyJoinModeV2.ALL_TERMINAL

    batch = _unit(
        graph,
        StageNameV2.BATCH_QUALITY,
        scope=WorkUnitScopeV2.JOB,
    )
    assert batch.join_mode is WorkDependencyJoinModeV2.ALL_TERMINAL
    assert len(batch.depends_on_work_unit_refs) == 2
    assert graph.resolved_job_work_graph_sha256 == (gold["resolved_job_work_graph_sha256"])


def test_checkpoint_fanout_preserves_global_gate_and_item_local_lineage() -> None:
    checkpoints = frozenset(
        {
            ApprovalCheckpoint.LABEL_PLAN,
            ApprovalCheckpoint.TASK_REWRITE_PLAN,
            ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        }
    )
    requested = (
        StageNameV2.TRACE_INDEX,
        StageNameV2.LABEL_PLAN,
        StageNameV2.TASK_REWRITE_PLAN,
        StageNameV2.ENVIRONMENT_STRATEGY,
        StageNameV2.RELEASE,
    )
    graph = _graph(
        requested,
        approval_mode=ApprovalMode.PLAN_GATES,
        enabled_checkpoints=checkpoints,
    )
    gold = _gold("plan-checkpoint-graph")
    label_plan = _unit(
        graph,
        StageNameV2.LABEL_PLAN,
        scope=WorkUnitScopeV2.JOB,
    )

    for item_id in graph.item_ids:
        safety = _unit(graph, StageNameV2.SAFETY, item_id=item_id)
        label = _unit(graph, StageNameV2.LABEL, item_id=item_id)
        assert set(label.depends_on_work_unit_refs) == {
            resolved_work_unit_v2_ref(safety),
            resolved_work_unit_v2_ref(label_plan),
        }
    assert graph.resolved_job_work_graph_sha256 == (gold["resolved_job_work_graph_sha256"])


def test_readiness_waits_then_unlocks_only_exact_item_descendant() -> None:
    graph = _graph()
    first_item, second_item = graph.item_ids
    first_trace = _unit(graph, StageNameV2.TRACE_INDEX, item_id=first_item)
    first_safety = _unit(graph, StageNameV2.SAFETY, item_id=first_item)
    second_safety = _unit(graph, StageNameV2.SAFETY, item_id=second_item)
    evaluator = WorkReadinessEvaluator()

    waiting = evaluator.evaluate(
        graph=graph,
        work_unit=first_safety,
        stage_completions=(),
        item_records=(),
        audit=_audit(),
    )
    ready = evaluator.evaluate(
        graph=graph,
        work_unit=first_safety,
        stage_completions=(
            _completion(
                first_trace,
                status=StageRunStatus.SUCCEEDED,
                suffix="a-success",
            ),
        ),
        item_records=(),
        audit=_audit(),
    )
    unrelated = evaluator.evaluate(
        graph=graph,
        work_unit=second_safety,
        stage_completions=(
            _completion(
                first_trace,
                status=StageRunStatus.SUCCEEDED,
                suffix="a-success",
            ),
        ),
        item_records=(),
        audit=_audit(),
    )

    assert waiting.readiness is WorkReadinessV2.WAITING
    assert ready.readiness is WorkReadinessV2.READY
    assert unrelated.readiness is WorkReadinessV2.WAITING


def test_readiness_rejects_stale_stage_result_identity() -> None:
    graph = _graph()
    trace = _unit(
        graph,
        StageNameV2.TRACE_INDEX,
        item_id=graph.item_ids[0],
    )
    safety = _unit(
        graph,
        StageNameV2.SAFETY,
        item_id=graph.item_ids[0],
    )
    completion = _completion(
        trace,
        status=StageRunStatus.SUCCEEDED,
        suffix="stale-result",
    )
    stale = completion.model_copy(
        update={"stage_result": completion.stage_result.model_copy(update={"result_sha256": "f" * 64})}
    )

    with pytest.raises(
        WorkFanoutPolicyError,
        match="exact work unit",
    ):
        WorkReadinessEvaluator().evaluate(
            graph=graph,
            work_unit=safety,
            stage_completions=(stale,),
            item_records=(),
            audit=_audit(),
        )


def test_all_terminal_join_allows_mixed_nonretryable_but_waits_for_retryable() -> None:
    graph = _graph()
    gold = _gold("mixed-item-terminal-fan-in")
    first_quality = _unit(
        graph,
        StageNameV2.ITEM_QUALITY,
        item_id=graph.item_ids[0],
    )
    second_quality = _unit(
        graph,
        StageNameV2.ITEM_QUALITY,
        item_id=graph.item_ids[1],
    )
    batch = _unit(
        graph,
        StageNameV2.BATCH_QUALITY,
        scope=WorkUnitScopeV2.JOB,
    )
    evaluator = WorkReadinessEvaluator()
    success = _completion(
        first_quality,
        status=StageRunStatus.SUCCEEDED,
        suffix="a-quality-success",
    )
    terminal = _completion(
        second_quality,
        status=StageRunStatus.TERMINAL_FAILURE,
        retryable=False,
        suffix="b-quality-terminal",
    )
    retryable = _completion(
        second_quality,
        status=StageRunStatus.RETRYABLE_FAILURE,
        retryable=True,
        suffix="b-quality-retryable",
    )

    mixed = evaluator.evaluate(
        graph=graph,
        work_unit=batch,
        stage_completions=(success, terminal),
        item_records=(),
        audit=_audit(),
    )
    waiting = evaluator.evaluate(
        graph=graph,
        work_unit=batch,
        stage_completions=(success, retryable),
        item_records=(),
        audit=_audit(),
    )

    assert mixed.readiness is WorkReadinessV2.READY
    assert len(mixed.terminal_non_success_result_refs) == 1
    assert waiting.readiness is WorkReadinessV2.WAITING
    assert len(waiting.retryable_dependency_result_refs) == 1
    assert mixed.work_readiness_snapshot_sha256 == (gold["work_readiness_snapshot_sha256"])
    assert waiting.work_readiness_snapshot_sha256 == (gold["retryable_snapshot_sha256"])


def test_retry_decision_keeps_scheduled_work_waiting_and_makes_exhaustion_terminal() -> None:
    graph = _graph()
    first_quality = _unit(
        graph,
        StageNameV2.ITEM_QUALITY,
        item_id=graph.item_ids[0],
    )
    second_quality = _unit(
        graph,
        StageNameV2.ITEM_QUALITY,
        item_id=graph.item_ids[1],
    )
    batch = _unit(
        graph,
        StageNameV2.BATCH_QUALITY,
        scope=WorkUnitScopeV2.JOB,
    )
    first_success = _completion(
        first_quality,
        status=StageRunStatus.SUCCEEDED,
        suffix="r6-05-first-quality-success",
    )
    retryable = _completion(
        second_quality,
        status=StageRunStatus.RETRYABLE_FAILURE,
        retryable=True,
        suffix="r6-05-second-quality-retryable",
    )
    scheduled = _retry_decision(
        second_quality,
        retryable,
        decision=WorkRetryDecisionKindV2.RETRY_SCHEDULED,
    )
    exhausted = _retry_decision(
        second_quality,
        retryable,
        decision=WorkRetryDecisionKindV2.EXHAUSTED,
    )
    evaluator = WorkReadinessEvaluator()

    waiting = evaluator.evaluate(
        graph=graph,
        work_unit=batch,
        stage_completions=(first_success, retryable),
        item_records=(),
        work_retry_decisions=(scheduled,),
        audit=_audit(),
    )
    terminal = evaluator.evaluate(
        graph=graph,
        work_unit=batch,
        stage_completions=(first_success, retryable),
        item_records=(),
        work_retry_decisions=(exhausted,),
        audit=_audit(),
    )

    assert waiting.readiness is WorkReadinessV2.WAITING
    assert waiting.retryable_dependency_result_refs == (work_retry_decision_v2_ref(scheduled),)
    assert terminal.readiness is WorkReadinessV2.READY
    assert terminal.terminal_non_success_result_refs == (work_retry_decision_v2_ref(exhausted),)


def test_rejected_item_satisfies_batch_terminal_fan_in() -> None:
    graph = _graph()
    first_quality = _unit(
        graph,
        StageNameV2.ITEM_QUALITY,
        item_id=graph.item_ids[0],
    )
    batch = _unit(
        graph,
        StageNameV2.BATCH_QUALITY,
        scope=WorkUnitScopeV2.JOB,
    )
    rejected = ItemRecord(
        item_id=graph.item_ids[1],
        job_id=graph.job_id,
        status=ItemStatus.REJECTED,
        row_version=1,
        idempotency_key="r6-02-rejected-fan-in",
        created_at=NOW,
        updated_at=NOW,
    )

    snapshot = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=batch,
        stage_completions=(
            _completion(
                first_quality,
                status=StageRunStatus.SUCCEEDED,
                suffix="first-quality-success",
            ),
        ),
        item_records=(rejected,),
        audit=_audit(),
    )

    assert snapshot.readiness is WorkReadinessV2.READY
    assert len(snapshot.succeeded_dependency_result_refs) == 1
    assert len(snapshot.skipped_item_refs) == 1


def test_item_quality_waits_for_retryable_attachment_but_accepts_terminal() -> None:
    graph = _graph()
    attachment = _unit(
        graph,
        StageNameV2.ATTACHMENT,
        item_id=graph.item_ids[0],
    )
    quality = _unit(
        graph,
        StageNameV2.ITEM_QUALITY,
        item_id=graph.item_ids[0],
    )
    evaluator = WorkReadinessEvaluator()

    terminal = evaluator.evaluate(
        graph=graph,
        work_unit=quality,
        stage_completions=(
            _completion(
                attachment,
                status=StageRunStatus.TERMINAL_FAILURE,
                retryable=False,
                suffix="attachment-terminal",
            ),
        ),
        item_records=(),
        audit=_audit(),
    )
    retryable = evaluator.evaluate(
        graph=graph,
        work_unit=quality,
        stage_completions=(
            _completion(
                attachment,
                status=StageRunStatus.RETRYABLE_FAILURE,
                retryable=True,
                suffix="attachment-retryable",
            ),
        ),
        item_records=(),
        audit=_audit(),
    )

    assert terminal.readiness is WorkReadinessV2.READY
    assert len(terminal.terminal_non_success_result_refs) == 1
    assert retryable.readiness is WorkReadinessV2.WAITING
    assert len(retryable.retryable_dependency_result_refs) == 1


def test_terminal_item_skips_future_work_without_fabricated_stage_run() -> None:
    graph = _graph()
    item_id = graph.item_ids[0]
    task = _unit(graph, StageNameV2.TASK_AUTHORING, item_id=item_id)
    item = ItemRecord(
        item_id=item_id,
        job_id=graph.job_id,
        status=ItemStatus.REJECTED,
        row_version=1,
        idempotency_key="r6-02-rejected-item",
        created_at=NOW,
        updated_at=NOW,
    )

    snapshot = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=task,
        stage_completions=(),
        item_records=(item,),
        audit=_audit(),
    )

    assert snapshot.readiness is WorkReadinessV2.SKIPPED_ITEM
    assert len(snapshot.skipped_item_refs) == 1


def test_readiness_rejects_cross_graph_item_evidence() -> None:
    graph = _graph()
    task = _unit(
        graph,
        StageNameV2.TASK_AUTHORING,
        item_id=graph.item_ids[0],
    )
    item = ItemRecord(
        item_id=graph.item_ids[0],
        job_id="job://r6-02/other",
        status=ItemStatus.REJECTED,
        row_version=1,
        idempotency_key="r6-02-cross-job-item",
        created_at=NOW,
        updated_at=NOW,
    )

    with pytest.raises(
        WorkFanoutPolicyError,
        match="duplicate or cross-graph",
    ):
        WorkReadinessEvaluator().evaluate(
            graph=graph,
            work_unit=task,
            stage_completions=(),
            item_records=(item,),
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_artifact_fanout_reuses_exact_r5_groups_without_stage_runs(
    tmp_path: object,
) -> None:
    graph = _graph()
    parent = _unit(
        graph,
        StageNameV2.ATTACHMENT,
        item_id=graph.item_ids[0],
    )
    plan = await _execution_plan(
        tmp_path,
        (
            ("artifact://a", "inputs/a.txt"),
            ("artifact://b", "inputs/b.txt"),
        ),
        (
            _definition("artifact://a"),
            _definition("artifact://b"),
        ),
    )

    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=graph,
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=_planning_result(plan),
        audit=_audit(),
    )
    gold = _gold("exact-r5-group-fanout-and-join")

    assert isinstance(fanout, ArtifactGroupFanoutV2)
    assert len(fanout.group_work_units) == len(plan.groups) == 2
    assert all(
        unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP
        and unit.stage is StageNameV2.ATTACHMENT
        and unit.item_id == parent.item_id
        and unit.depends_on_work_unit_refs == ()
        for unit in fanout.group_work_units
    )
    assert fanout.artifact_group_fanout_sha256 == (gold["artifact_group_fanout_sha256"])

    with pytest.raises(WorkFanoutPolicyError, match="ATTACHMENT"):
        ArtifactGroupFanoutCompiler().compile(
            graph=graph,
            parent_attachment_work_unit=_unit(
                graph,
                StageNameV2.ITEM_QUALITY,
                item_id=graph.item_ids[0],
            ),
            artifact_execution_planning_result=_planning_result(plan),
            audit=_audit(),
        )

    stale_planning = _planning_result(plan).model_copy(update={"audit": _artifact_audit()})
    with pytest.raises(
        WorkFanoutPolicyError,
        match="planning result audit is stale",
    ):
        ArtifactGroupFanoutCompiler().compile(
            graph=graph,
            parent_attachment_work_unit=parent,
            artifact_execution_planning_result=stale_planning,
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_parent_attachment_joins_exact_r5_batch_without_group_stage_runs(
    tmp_path: object,
) -> None:
    graph = _graph()
    parent = _unit(
        graph,
        StageNameV2.ATTACHMENT,
        item_id=graph.item_ids[0],
    )
    plan = await _execution_plan(
        tmp_path,
        (
            ("artifact://a", "inputs/a.txt"),
            ("artifact://b", "inputs/b.txt"),
        ),
        (
            _definition("artifact://a"),
            _definition("artifact://b"),
        ),
    )
    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=graph,
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=_planning_result(plan),
        audit=_audit(),
    )
    gold = _gold("exact-r5-group-fanout-and-join")
    evaluator = WorkReadinessEvaluator()

    before_batch = evaluator.evaluate(
        graph=graph,
        work_unit=parent,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        audit=_audit(),
    )
    success_batch = await ArtifactGroupExecutor().run(
        plan,
        facade=_ControlledExecutionFacade(expected_parallelism=2),
        audit=_artifact_audit(),
    )
    completed = evaluator.evaluate(
        graph=graph,
        work_unit=parent,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        artifact_execution_planning_result=_planning_result(plan),
        artifact_execution_batch=success_batch,
        audit=_audit(),
    )
    retryable_batch = await ArtifactGroupExecutor().run(
        plan,
        facade=_ControlledExecutionFacade(
            fail_once=frozenset({"artifact://b"}),
            expected_parallelism=2,
        ),
        audit=_artifact_audit(),
    )
    retryable = evaluator.evaluate(
        graph=graph,
        work_unit=parent,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        artifact_execution_planning_result=_planning_result(plan),
        artifact_execution_batch=retryable_batch,
        audit=_audit(),
    )
    retryable_receipt = next(
        receipt for receipt in retryable_batch.receipts if receipt.outcome.value == "RETRYABLE_FAILURE"
    )
    retryable_group = next(
        unit
        for unit in fanout.group_work_units
        if unit.artifact_execution_group_ref == retryable_receipt.artifact_execution_group_ref
    )
    exhausted_decision = _artifact_retry_decision(
        retryable_group,
        artifact_execution_batch_ref(retryable_batch),
        artifact_execution_receipt_ref(retryable_receipt),
        decision=WorkRetryDecisionKindV2.EXHAUSTED,
    )
    exhausted = evaluator.evaluate(
        graph=graph,
        work_unit=parent,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        artifact_execution_planning_result=_planning_result(plan),
        artifact_execution_batch=retryable_batch,
        work_retry_decisions=(exhausted_decision,),
        audit=_audit(),
    )

    assert before_batch.readiness is WorkReadinessV2.WAITING
    assert set(before_batch.waiting_dependency_work_unit_refs) == {
        resolved_work_unit_v2_ref(unit) for unit in fanout.group_work_units
    }
    assert completed.readiness is WorkReadinessV2.READY
    assert set(completed.succeeded_dependency_result_refs) == {
        artifact_execution_batch_ref(success_batch),
        *(artifact_execution_receipt_ref(receipt) for receipt in success_batch.receipts),
    }
    assert retryable.readiness is WorkReadinessV2.WAITING
    assert artifact_execution_batch_ref(retryable_batch) in (retryable.retryable_dependency_result_refs)
    assert exhausted.readiness is WorkReadinessV2.READY
    assert work_retry_decision_v2_ref(exhausted_decision) in (exhausted.terminal_non_success_result_refs)
    assert artifact_execution_batch_ref(retryable_batch) in (exhausted.terminal_non_success_result_refs)
    assert plan.artifact_execution_plan_sha256 == (gold["artifact_execution_plan_sha256"])
    assert [unit.resolved_work_unit_sha256 for unit in fanout.group_work_units] == gold["group_unit_sha256"]
    assert before_batch.work_readiness_snapshot_sha256 == (gold["before_batch_snapshot_sha256"])
    assert completed.work_readiness_snapshot_sha256 == (gold["joined_snapshot_sha256"])
    assert retryable.work_readiness_snapshot_sha256 == (gold["retryable_snapshot_sha256"])


@pytest.mark.asyncio
async def test_artifact_control_terminal_event_satisfies_parent_terminal_join(
    tmp_path: Path,
) -> None:
    graph = _graph()
    parent = _unit(
        graph,
        StageNameV2.ATTACHMENT,
        item_id=graph.item_ids[0],
    )
    plan = await _execution_plan(
        tmp_path / "r6-05-control-terminal",
        (("artifact://a", "inputs/a.txt"),),
        (_definition("artifact://a"),),
    )
    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=graph,
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=_planning_result(plan),
        audit=_audit(),
    )
    event = _terminal_control_event(fanout.group_work_units[0])

    snapshot = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=parent,
        stage_completions=(),
        item_records=(),
        artifact_group_fanout=fanout,
        work_lease_events=(event,),
        audit=_audit(),
    )

    assert snapshot.readiness is WorkReadinessV2.READY
    assert snapshot.terminal_non_success_result_refs == (work_lease_event_v2_ref(event),)
    assert snapshot.waiting_dependency_work_unit_refs == ()


def test_work_fanout_gold_is_content_free_and_policy_bound() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == ("eval-factory/r6-02-work-fanout-gold/v1")
    assert payload["policy_version"] == "dataset-work-graph/r6-02-v1"
    assert payload["claim_scope"] == ("CONTENT_FREE_WORK_GRAPH_AND_READINESS_ONLY")
    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in (
        "trace_text",
        "task_text",
        "private_reference",
        "grader",
        "credential",
        "physical_path",
    ):
        assert forbidden not in serialized
