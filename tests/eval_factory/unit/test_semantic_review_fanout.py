from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_job_store import _audit, _job_spec, _plan, _work_graph
from test_work_fanout import _completion

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.orchestration import (
    JobStatus,
    StageRunStatus,
)
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    WorkDependencyJoinModeV2,
    WorkReadinessV2,
    resolved_work_unit_v2_ref,
)
from eval_factory.contracts.review_v2 import SemanticReviewRoundV2
from eval_factory.orchestration import (
    ImmutableResultError,
    JobStore,
    SemanticReviewFanoutCompiler,
    WorkControlService,
    WorkReadinessEvaluator,
)

HASH = "a" * 64


def _parent(graph):
    return next(unit for unit in graph.work_units if unit.stage is StageNameV2.ITEM_QUALITY)


def _fanout(graph):
    return SemanticReviewFanoutCompiler().compile(
        graph=graph,
        parent_item_quality_work_unit=_parent(graph),
        audit=_audit(),
    )


def _prepared_store(tmp_path: Path):
    store = JobStore(tmp_path / "factory.sqlite3")
    spec = _job_spec(
        job_id="job://r6-parent/semantic-review",
        idempotency_key="create-r6-parent-semantic-review",
    )
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-semantic-review-work-graph",
    )
    return store, spec, graph


def test_compiler_creates_three_distinct_round_units() -> None:
    graph = _work_graph(_job_spec())
    parent = _parent(graph)
    fanout = _fanout(graph)

    assert tuple(value.round for value in fanout.round_work) == tuple(SemanticReviewRoundV2)
    assert all(value.work_unit.semantic_review_round is value.round for value in fanout.round_work)
    refs = tuple(resolved_work_unit_v2_ref(value.work_unit) for value in fanout.round_work)
    assert len(set(refs)) == 3
    assert resolved_work_unit_v2_ref(parent) not in refs
    assert fanout.round_work[0].work_unit.depends_on_work_unit_refs == parent.depends_on_work_unit_refs
    assert fanout.round_work[0].work_unit.join_mode is WorkDependencyJoinModeV2.ALL_SUCCEEDED
    assert fanout.round_work[1].work_unit.depends_on_work_unit_refs == (refs[0],)
    assert fanout.round_work[2].work_unit.depends_on_work_unit_refs == (refs[1],)


def test_parent_readiness_dynamically_joins_all_review_rounds() -> None:
    graph = _work_graph(_job_spec())
    parent = _parent(graph)
    fanout = _fanout(graph)
    attachment = next(unit for unit in graph.work_units if unit.stage is StageNameV2.ATTACHMENT)
    attachment_completion = _completion(
        attachment,
        status=StageRunStatus.SUCCEEDED,
        suffix="semantic-attachment",
    )
    round_completions = tuple(
        _completion(
            value.work_unit,
            status=StageRunStatus.SUCCEEDED,
            suffix=f"semantic-{value.round.value.lower()}",
        )
        for value in fanout.round_work
    )
    evaluator = WorkReadinessEvaluator()

    waiting = evaluator.evaluate(
        graph=graph,
        work_unit=parent,
        stage_completions=(
            attachment_completion,
            *round_completions[:2],
        ),
        item_records=(),
        semantic_review_fanout=fanout,
        audit=_audit(),
    )
    ready = evaluator.evaluate(
        graph=graph,
        work_unit=parent,
        stage_completions=(
            attachment_completion,
            *round_completions,
        ),
        item_records=(),
        semantic_review_fanout=fanout,
        audit=_audit(),
    )

    assert waiting.readiness is WorkReadinessV2.WAITING
    assert waiting.semantic_review_fanout_ref is not None
    assert waiting.waiting_dependency_work_unit_refs == (
        resolved_work_unit_v2_ref(fanout.round_work[2].work_unit),
    )
    assert ready.readiness is WorkReadinessV2.READY
    assert len(ready.succeeded_dependency_result_refs) == 4


def test_job_store_persists_replays_and_validates_fanout(
    tmp_path: Path,
) -> None:
    store, spec, graph = _prepared_store(tmp_path)
    fanout = _fanout(graph)

    first = store.create_semantic_review_fanout(
        fanout,
        idempotency_key="create-semantic-review-fanout",
    )
    replay = store.create_semantic_review_fanout(
        fanout,
        idempotency_key="create-semantic-review-fanout",
    )

    assert replay == first == fanout
    assert store.get_semantic_review_fanout(fanout.parent_item_quality_work_unit_ref.object_id) == fanout
    assert store.list_work_units(job_id=spec.job_id) == (
        *graph.work_units,
        *(value.work_unit for value in fanout.round_work),
    )

    with pytest.raises(ValidationError, match="identity"):
        store.create_semantic_review_fanout(
            fanout.model_copy(update={"semantic_review_fanout_id": ("semantic-review-fanout://different")}),
            idempotency_key="create-semantic-review-fanout",
        )

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE resolved_work_units
            SET owner_ref = ?
            WHERE resolved_work_unit_id = ?
            """,
            (
                graph.resolved_job_work_graph_id,
                fanout.round_work[0].work_unit.resolved_work_unit_id,
            ),
        )
    with pytest.raises(ImmutableResultError, match="materialized"):
        store.list_work_units(job_id=spec.job_id)


def test_controlled_round_unit_can_acquire_its_own_stage_run(
    tmp_path: Path,
) -> None:
    store, spec, graph = _prepared_store(tmp_path)
    fanout = store.create_semantic_review_fanout(
        _fanout(graph),
        idempotency_key="create-controlled-review-fanout",
    )
    parent = _parent(graph)
    attachment = next(unit for unit in graph.work_units if unit.stage is StageNameV2.ATTACHMENT)
    attachment_ref = resolved_work_unit_v2_ref(attachment)
    run = store.create_stage_run(
        stage_run_id="stage-run://semantic-review/attachment",
        job_id=spec.job_id,
        item_id=parent.item_id,
        stage=StageNameV2.ATTACHMENT,
        attempt=1,
        principal_ref=ObjectRef(
            object_type="stage-principal",
            object_id="stage-principal://semantic-review/attachment",
            object_version="v1",
            object_sha256=HASH,
        ),
        input_refs=(attachment_ref,),
        idempotency_key="create-semantic-review-attachment-run",
    )
    running = store.transition_stage_run(
        run.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-semantic-review-attachment-run",
    )
    result = store.complete_stage_run(
        stage_result_id="stage-result://semantic-review/attachment",
        stage_run_id=running.stage_run_id,
        status=StageRunStatus.SUCCEEDED,
        expected_version=1,
        output_refs=(
            ObjectRef(
                object_type="attachment-reconstruction-result",
                object_id="attachment-reconstruction-result://semantic",
                object_version="v2",
                object_sha256=HASH,
            ),
        ),
        failure=None,
        checkpoint_ref=None,
        metrics_ref=None,
        idempotency_key="complete-semantic-review-attachment-run",
    )
    coverage = fanout.round_work[0].work_unit
    readiness = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=coverage,
        stage_completions=(
            _completion(
                attachment,
                status=StageRunStatus.SUCCEEDED,
                suffix="semantic-coverage-dependency",
            ).model_copy(
                update={
                    "stage_run": store.get_stage_run(running.stage_run_id),
                    "stage_result": result,
                }
            ),
        ),
        item_records=(),
        semantic_review_fanout=fanout,
        audit=_audit(),
    )
    store.record_work_readiness(
        readiness,
        idempotency_key="record-semantic-review-coverage-ready",
    )
    created = store.get_job(spec.job_id)
    store.transition_job(
        spec.job_id,
        JobStatus.RUNNING,
        expected_version=created.row_version,
        idempotency_key="start-semantic-review-job",
    )
    control = WorkControlService(store)
    policy = control.bind_policy(
        graph=graph,
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=1,
        retry_delay_seconds=(),
        retry_lease_expiry=False,
        audit=_audit(),
        idempotency_key="bind-semantic-review-control",
    )
    holder = ObjectRef(
        object_type="worker-principal",
        object_id="worker-principal://semantic-review/coverage",
        object_version="v1",
        object_sha256=HASH,
    )

    lease = control.acquire(
        graph=graph,
        work_unit=coverage,
        readiness_snapshot=readiness,
        policy=policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-semantic-review-coverage",
    )

    assert lease.stage_run_ref is not None
    controlled = store.get_stage_run(lease.stage_run_ref.object_id)
    assert controlled.stage is StageNameV2.ITEM_QUALITY
    assert coverage.semantic_review_round is (SemanticReviewRoundV2.COVERAGE_SOLVABILITY)
    assert lease.work_unit_ref in controlled.input_refs
