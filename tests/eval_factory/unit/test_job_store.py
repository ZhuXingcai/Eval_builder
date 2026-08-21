from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_artifact_execution import _definition, _execution_plan
from test_work_fanout import _completion, _planning_result

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import (
    ContractAudit,
    FailureClass,
    FailureRecord,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ItemStatus,
    JobStatus,
    ResourceBudget,
    StageRunStatus,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedDatasetJobPlanV2,
    ResolvedJobWorkGraphV2,
    StageNameV2,
)
from eval_factory.orchestration import (
    ConcurrencyConflictError,
    IdempotencyConflictError,
    IllegalTransitionError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
    stage_result_record_carried_sha256,
    stage_result_record_ref,
    stage_run_record_ref,
)
from eval_factory.orchestration.fanout import (
    ArtifactGroupFanoutCompiler,
    DatasetJobWorkGraphCompiler,
    WorkFanoutPolicyError,
    WorkReadinessEvaluator,
)
from eval_factory.orchestration.planning import DatasetJobPlanCompiler

HASH = "a" * 64
NOW = datetime(2026, 7, 21, tzinfo=UTC)


def _ref(name: str) -> ObjectRef:
    return ObjectRef(
        object_type=name,
        object_id=f"{name}://example/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="job-store-test",
        governing_versions=(VersionBinding(component="eval-factory-spec", version="approved-v2"),),
    )


def _job_spec(
    *,
    job_id: str = "job://example/v2",
    idempotency_key: str = "create-job-example-v2",
) -> DatasetJobSpecV2:
    return DatasetJobSpecV2(
        job_id=job_id,
        traces=(
            TraceSourceRef(
                source_trace_id="source-trace://example",
                source_uri="raw-traj://example",
                raw_sha256=HASH,
                adapter_name="raw-traj-v1",
                adapter_version="v1",
                processing_class="RESTRICTED_TRACE_RAW",
            ),
        ),
        requested_stages=(
            StageNameV2.TRACE_INDEX,
            StageNameV2.SAFETY,
            StageNameV2.LABEL,
            StageNameV2.TASK_AUTHORING,
            StageNameV2.ATTACHMENT,
            StageNameV2.ITEM_QUALITY,
            StageNameV2.BATCH_QUALITY,
            StageNameV2.RELEASE,
        ),
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
            items=1,
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
        idempotency_key=idempotency_key,
        audit=_audit(),
    )


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "factory.sqlite3", clock=lambda: NOW)


def _plan(spec: DatasetJobSpecV2) -> ResolvedDatasetJobPlanV2:
    return DatasetJobPlanCompiler().compile(job_spec=spec, audit=_audit())


def _work_graph(spec: DatasetJobSpecV2) -> ResolvedJobWorkGraphV2:
    return DatasetJobWorkGraphCompiler().compile(
        job_spec=spec,
        resolved_plan=_plan(spec),
        audit=_audit(),
    )


def _failure(code: str = "retryable-stage-error") -> FailureRecord:
    return FailureRecord(
        failure_class=FailureClass.INTERNAL,
        code=code,
        message="Injected stage failure.",
        retryable=True,
    )


def test_create_job_is_idempotent_and_binds_exact_request(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _job_spec()

    first = store.create_job(spec)
    replay = store.create_job(spec)

    assert replay == first
    assert store.get_job_spec(first.job_id) == spec
    assert [event.event_type for event in store.list_outbox()] == ["job-created"]

    different = _job_spec(job_id="job://different/v2")
    with pytest.raises(IdempotencyConflictError, match="different request"):
        store.create_job(different)


def test_job_transitions_enforce_legal_graph_version_and_replay(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.create_job(_job_spec())

    running = store.transition_job(
        job.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="job-transition-running-v1",
    )
    replay = store.transition_job(
        job.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="job-transition-running-v1",
    )
    assert replay == running
    assert running.row_version == 1

    with pytest.raises(ConcurrencyConflictError, match="stale aggregate"):
        store.transition_job(
            job.job_id,
            JobStatus.BLOCKED,
            expected_version=0,
            idempotency_key="job-transition-stale-v1",
        )
    with pytest.raises(IllegalTransitionError, match="illegal job transition"):
        store.transition_job(
            job.job_id,
            JobStatus.CREATED,
            expected_version=1,
            idempotency_key="job-transition-illegal-v1",
        )

    succeeded = store.transition_job(
        job.job_id,
        JobStatus.SUCCEEDED,
        expected_version=1,
        idempotency_key="job-transition-succeeded-v1",
    )
    assert succeeded.status is JobStatus.SUCCEEDED
    with pytest.raises(IllegalTransitionError):
        store.transition_job(
            job.job_id,
            JobStatus.RUNNING,
            expected_version=2,
            idempotency_key="job-transition-after-terminal-v1",
        )


def test_item_state_machine_is_job_bound_and_versioned(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.create_job(_job_spec())
    item = store.create_item(
        job.job_id,
        "item://example/v1",
        idempotency_key="create-item-example-v1",
    )
    assert (
        store.create_item(
            job.job_id,
            item.item_id,
            idempotency_key="create-item-example-v1",
        )
        == item
    )

    running = store.transition_item(
        item.item_id,
        ItemStatus.RUNNING,
        expected_version=0,
        idempotency_key="item-running-v1",
    )
    approved = store.transition_item(
        item.item_id,
        ItemStatus.APPROVED,
        expected_version=running.row_version,
        idempotency_key="item-approved-v1",
    )
    released = store.transition_item(
        item.item_id,
        ItemStatus.RELEASED,
        expected_version=approved.row_version,
        idempotency_key="item-released-v1",
    )
    revoked = store.transition_item(
        item.item_id,
        ItemStatus.REVOKED,
        expected_version=released.row_version,
        idempotency_key="item-revoked-v1",
    )
    assert revoked.status is ItemStatus.REVOKED
    with pytest.raises(IllegalTransitionError):
        store.transition_item(
            item.item_id,
            ItemStatus.RUNNING,
            expected_version=revoked.row_version,
            idempotency_key="item-after-revoked-v1",
        )


def test_stage_result_is_immutable_and_commits_with_terminal_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.create_job(_job_spec())
    item = store.create_item(
        job.job_id,
        "item://stage/v1",
        idempotency_key="create-item-stage-v1",
    )
    stage = store.create_stage_run(
        stage_run_id="stage-run://example/1",
        job_id=job.job_id,
        item_id=item.item_id,
        stage=StageNameV2.TRACE_INDEX,
        attempt=1,
        principal_ref=_ref("stage-principal"),
        input_refs=(_ref("raw-trace"),),
        idempotency_key="create-stage-run-example-1",
    )
    running = store.transition_stage_run(
        stage.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=stage.row_version,
        idempotency_key="start-stage-run-example-1",
    )
    with pytest.raises(IllegalTransitionError, match="immutable result"):
        store.transition_stage_run(
            stage.stage_run_id,
            StageRunStatus.SUCCEEDED,
            expected_version=running.row_version,
            idempotency_key="bypass-stage-result-example-1",
        )
    result = store.complete_stage_run(
        stage_result_id="stage-result://example/1",
        stage_run_id=stage.stage_run_id,
        status=StageRunStatus.SUCCEEDED,
        expected_version=running.row_version,
        idempotency_key="complete-stage-run-example-1",
        output_refs=(_ref("trace-envelope"),),
    )
    replay = store.complete_stage_run(
        stage_result_id="stage-result://example/1",
        stage_run_id=stage.stage_run_id,
        status=StageRunStatus.SUCCEEDED,
        expected_version=running.row_version,
        idempotency_key="complete-stage-run-example-1",
        output_refs=(_ref("trace-envelope"),),
    )
    assert replay == result
    assert store.get_stage_result(result.stage_result_id) == result
    completed = store.get_stage_run(stage.stage_run_id)
    assert completed.status is StageRunStatus.SUCCEEDED
    assert completed.row_version == 2
    assert completed.ended_at == NOW

    with pytest.raises(ImmutableResultError):
        store.complete_stage_run(
            stage_result_id="stage-result://example/other",
            stage_run_id=stage.stage_run_id,
            status=StageRunStatus.SUCCEEDED,
            expected_version=completed.row_version,
            idempotency_key="complete-stage-run-example-other",
        )


def test_stage_run_identity_ref_is_stable_across_status_transitions(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    job = store.create_job(_job_spec())
    item = store.create_item(
        job.job_id,
        "item://stage-identity/v1",
        idempotency_key="create-item-stage-identity-v1",
    )
    stage = store.create_stage_run(
        stage_run_id="stage-run://identity/1",
        job_id=job.job_id,
        item_id=item.item_id,
        stage=StageNameV2.ITEM_QUALITY,
        attempt=1,
        principal_ref=_ref("coverage-reviewer"),
        input_refs=(_ref("coverage-review-view"),),
        idempotency_key="create-stage-identity-1",
    )
    pending_ref = stage_run_record_ref(stage)
    running = store.transition_stage_run(
        stage.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=stage.row_version,
        idempotency_key="start-stage-identity-1",
    )
    running_ref = stage_run_record_ref(running)
    result = store.complete_stage_run(
        stage_result_id="stage-result://identity/1",
        stage_run_id=stage.stage_run_id,
        status=StageRunStatus.SUCCEEDED,
        expected_version=running.row_version,
        idempotency_key="complete-stage-identity-1",
        output_refs=(_ref("semantic-review-round-result"),),
    )
    completed_ref = stage_run_record_ref(store.get_stage_run(stage.stage_run_id))

    assert pending_ref == running_ref == completed_ref
    assert pending_ref.object_version == "identity/v1"
    assert stage_result_record_ref(result).object_sha256 == result.result_sha256
    assert stage_result_record_carried_sha256(result) == result.result_sha256
    assert stage_result_record_ref(result).object_version == "record/v1"


def test_stage_active_uniqueness_and_retry_chain_are_enforced(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.create_job(_job_spec())
    first = store.create_stage_run(
        stage_run_id="stage-run://retry/1",
        job_id=job.job_id,
        item_id=None,
        stage=StageNameV2.TRACE_INDEX,
        attempt=1,
        principal_ref=_ref("stage-principal"),
        input_refs=(_ref("raw-trace"),),
        idempotency_key="create-stage-retry-1",
    )
    with pytest.raises(IllegalTransitionError, match="active"):
        store.create_stage_run(
            stage_run_id="stage-run://retry/parallel",
            job_id=job.job_id,
            item_id=None,
            stage=StageNameV2.TRACE_INDEX,
            attempt=1,
            principal_ref=_ref("stage-principal"),
            input_refs=(_ref("raw-trace"),),
            idempotency_key="create-stage-retry-parallel",
        )

    running = store.transition_stage_run(
        first.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-stage-retry-1",
    )
    store.complete_stage_run(
        stage_result_id="stage-result://retry/1",
        stage_run_id=first.stage_run_id,
        status=StageRunStatus.RETRYABLE_FAILURE,
        expected_version=running.row_version,
        idempotency_key="complete-stage-retry-1",
        failure=_failure(),
    )
    retry = store.create_stage_run(
        stage_run_id="stage-run://retry/2",
        job_id=job.job_id,
        item_id=None,
        stage=StageNameV2.TRACE_INDEX,
        attempt=2,
        principal_ref=_ref("stage-principal"),
        input_refs=(_ref("raw-trace"),),
        idempotency_key="create-stage-retry-2",
        retry_of_stage_run_id=first.stage_run_id,
    )
    assert retry.retry_of_stage_run_id == first.stage_run_id

    with pytest.raises(IllegalTransitionError, match="identity or attempt"):
        store.create_stage_run(
            stage_run_id="stage-run://retry/wrong",
            job_id=job.job_id,
            item_id=None,
            stage=StageNameV2.SAFETY,
            attempt=2,
            principal_ref=_ref("stage-principal"),
            input_refs=(_ref("raw-trace"),),
            idempotency_key="create-stage-retry-wrong",
            retry_of_stage_run_id=first.stage_run_id,
        )


def test_aggregate_and_outbox_roll_back_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    original = store._append_outbox

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="injected outbox failure"):
        store.create_job(_job_spec())
    with pytest.raises(RecordNotFoundError):
        store.get_job("job://example/v2")
    assert store.list_outbox() == ()

    monkeypatch.setattr(store, "_append_outbox", original)
    created = store.create_job(_job_spec())
    assert created.status is JobStatus.CREATED
    assert len(store.list_outbox()) == 1


def test_non_success_stage_result_requires_failure(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.create_job(_job_spec())
    stage = store.create_stage_run(
        stage_run_id="stage-run://failure/1",
        job_id=job.job_id,
        item_id=None,
        stage=StageNameV2.TRACE_INDEX,
        attempt=1,
        principal_ref=_ref("stage-principal"),
        input_refs=(_ref("raw-trace"),),
        idempotency_key="create-stage-failure-1",
    )
    running = store.transition_stage_run(
        stage.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-stage-failure-1",
    )
    with pytest.raises(ValueError, match="requires failure"):
        store.complete_stage_run(
            stage_result_id="stage-result://failure/1",
            stage_run_id=stage.stage_run_id,
            status=StageRunStatus.RETRYABLE_FAILURE,
            expected_version=running.row_version,
            idempotency_key="complete-stage-failure-1",
        )
    assert store.get_stage_run(stage.stage_run_id).status is StageRunStatus.RUNNING
    with pytest.raises(RecordNotFoundError):
        store.get_stage_result("stage-result://failure/1")


def test_stage_completion_result_idempotency_and_outbox_roll_back_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    job = store.create_job(_job_spec())
    stage = store.create_stage_run(
        stage_run_id="stage-run://atomic/1",
        job_id=job.job_id,
        item_id=None,
        stage=StageNameV2.TRACE_INDEX,
        attempt=1,
        principal_ref=_ref("stage-principal"),
        input_refs=(_ref("raw-trace"),),
        idempotency_key="create-stage-atomic-1",
    )
    running = store.transition_stage_run(
        stage.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-stage-atomic-1",
    )
    outbox_before = store.list_outbox()
    original = store._append_outbox

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected completion outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="completion outbox failure"):
        store.complete_stage_run(
            stage_result_id="stage-result://atomic/1",
            stage_run_id=stage.stage_run_id,
            status=StageRunStatus.SUCCEEDED,
            expected_version=running.row_version,
            idempotency_key="complete-stage-atomic-1",
            output_refs=(_ref("trace-envelope"),),
        )

    assert store.get_stage_run(stage.stage_run_id) == running
    with pytest.raises(RecordNotFoundError):
        store.get_stage_result("stage-result://atomic/1")
    assert store.list_outbox() == outbox_before

    monkeypatch.setattr(store, "_append_outbox", original)
    result = store.complete_stage_run(
        stage_result_id="stage-result://atomic/1",
        stage_run_id=stage.stage_run_id,
        status=StageRunStatus.SUCCEEDED,
        expected_version=running.row_version,
        idempotency_key="complete-stage-atomic-1",
        output_refs=(_ref("trace-envelope"),),
    )
    assert store.get_stage_result(result.stage_result_id) == result
    assert len(store.list_outbox()) == len(outbox_before) + 1


def test_planned_job_creation_persists_exact_plan_and_replays_without_new_events(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)

    first = store.create_planned_job(spec, plan)
    replay = store.create_planned_job(spec, plan)

    assert replay == first
    assert store.get_job_spec(first.job_id) == spec
    assert store.get_resolved_job_plan(first.job_id) == plan
    assert [event.event_type for event in store.list_outbox()] == ["job-created"]


def test_planned_job_replay_rejects_conflicting_request_or_plan(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    store.create_planned_job(spec, plan)

    different_spec = _job_spec(job_id="job://different/v2")
    with pytest.raises(IdempotencyConflictError, match="different request"):
        store.create_planned_job(different_spec, _plan(different_spec))

    different_audit_plan = DatasetJobPlanCompiler().compile(
        job_spec=spec,
        audit=ContractAudit(
            created_at=datetime(2030, 1, 1, tzinfo=UTC),
            created_by="different-r6-planner",
            governing_versions=_audit().governing_versions,
        ),
    )
    assert different_audit_plan.resolved_job_plan_id == plan.resolved_job_plan_id
    with pytest.raises(ImmutableResultError, match="stored resolved plan"):
        store.create_planned_job(spec, different_audit_plan)


def test_unplanned_r1_job_has_no_fabricated_resolved_plan(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = store.create_job(_job_spec())

    with pytest.raises(RecordNotFoundError, match="ResolvedDatasetJobPlanV2"):
        store.get_resolved_job_plan(job.job_id)


def test_planned_job_and_plan_roll_back_when_outbox_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    original = store._append_outbox

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected planned-create outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="planned-create outbox failure"):
        store.create_planned_job(spec, plan)

    with pytest.raises(RecordNotFoundError):
        store.get_job(spec.job_id)
    with pytest.raises(RecordNotFoundError):
        store.get_resolved_job_plan(spec.job_id)
    assert store.list_outbox() == ()

    monkeypatch.setattr(store, "_append_outbox", original)
    created = store.create_planned_job(spec, plan)
    assert store.get_resolved_job_plan(created.job_id) == plan


def test_planned_job_rolls_back_when_plan_insert_fails(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    with sqlite3.connect(store.path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER fail_resolved_plan_insert
            BEFORE INSERT ON resolved_job_plans
            BEGIN
                SELECT RAISE(ABORT, 'injected resolved plan failure');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="resolved plan failure"):
        store.create_planned_job(spec, plan)

    with pytest.raises(RecordNotFoundError):
        store.get_job(spec.job_id)
    with pytest.raises(RecordNotFoundError):
        store.get_resolved_job_plan(spec.job_id)
    assert store.list_outbox() == ()


def test_resolved_plan_read_rejects_persisted_column_drift(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    store.create_planned_job(spec, plan)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE resolved_job_plans
            SET plan_sha256 = ?
            WHERE job_id = ?
            """,
            ("b" * 64, spec.job_id),
        )

    with pytest.raises(ImmutableResultError, match="columns are inconsistent"):
        store.get_resolved_job_plan(spec.job_id)
    with pytest.raises(ImmutableResultError, match="columns are inconsistent"):
        store.create_planned_job(spec, plan)


def test_existing_store_reopens_with_additive_plan_storage(tmp_path: Path) -> None:
    database = tmp_path / "factory.sqlite3"
    original = JobStore(database, clock=lambda: NOW)
    legacy_spec = _job_spec()
    legacy_job = original.create_job(legacy_spec)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE work_readiness_snapshots")
        connection.execute("DROP TABLE artifact_group_fanouts")
        connection.execute("DROP TABLE resolved_work_units")
        connection.execute("DROP TABLE resolved_job_work_graphs")
        connection.execute("DROP TABLE resolved_job_plans")

    reopened = JobStore(database, clock=lambda: NOW)
    assert reopened.get_job(legacy_job.job_id) == legacy_job
    assert reopened.get_job_spec(legacy_job.job_id) == legacy_spec
    with pytest.raises(RecordNotFoundError):
        reopened.get_resolved_job_plan(legacy_job.job_id)

    new_spec = _job_spec(
        job_id="job://r6-01/additive",
        idempotency_key="create-job-r6-01-additive",
    )
    new_plan = _plan(new_spec)
    new_job = reopened.create_planned_job(new_spec, new_plan)
    assert reopened.get_resolved_job_plan(new_job.job_id) == new_plan


def test_job_work_graph_read_rejects_materialized_column_drift(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-work-graph-corruption-r6-02",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE resolved_work_units
            SET stage = ?
            WHERE resolved_work_unit_id = ?
            """,
            (
                StageNameV2.RELEASE,
                graph.work_units[0].resolved_work_unit_id,
            ),
        )

    with pytest.raises(
        ImmutableResultError,
        match="work unit columns are inconsistent",
    ):
        store.get_job_work_graph(spec.job_id)


def test_job_work_graph_read_rejects_materialized_item_drift(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-work-graph-item-corruption-r6-02",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE items
            SET status = ?
            WHERE item_id = ?
            """,
            (ItemStatus.REJECTED, graph.item_ids[0]),
        )

    with pytest.raises(
        ImmutableResultError,
        match="Item columns are inconsistent",
    ):
        store.get_job_work_graph(spec.job_id)


def test_job_work_graph_atomically_creates_items_units_and_replays(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    base = _job_spec()
    second_trace = TraceSourceRef(
        source_trace_id="source-trace://second",
        source_uri="raw-traj://second",
        raw_sha256="b" * 64,
        adapter_name="raw-traj-v1",
        adapter_version="v1",
        processing_class="RESTRICTED_TRACE_RAW",
    )
    spec = DatasetJobSpecV2.model_validate(
        {
            **base.model_dump(mode="python"),
            "traces": (*base.traces, second_trace),
        }
    )
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)

    first = store.create_job_work_graph(
        graph,
        idempotency_key="create-work-graph-r6-02",
    )
    replay = store.create_job_work_graph(
        graph,
        idempotency_key="create-work-graph-r6-02",
    )

    assert replay == first == graph
    assert store.get_job_work_graph(spec.job_id) == graph
    assert tuple(item.item_id for item in store.list_items(spec.job_id)) == (graph.item_ids)
    assert store.list_work_units(job_id=spec.job_id) == graph.work_units
    event_types = [event.event_type for event in store.list_outbox()]
    assert event_types.count("job-created") == 1
    assert event_types.count("item-created") == len(graph.item_ids)
    assert event_types.count("job-work-graph-created") == 1


def test_job_work_graph_requires_planned_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    graph = _work_graph(spec)
    store.create_job(spec)

    with pytest.raises(WorkFanoutPolicyError, match="resolved plan"):
        store.create_job_work_graph(
            graph,
            idempotency_key="create-work-graph-unplanned-r6-02",
        )


def test_job_work_graph_rolls_back_items_units_and_outbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    outbox_before = store.list_outbox()
    original = store._append_outbox

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected work graph outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="work graph outbox failure"):
        store.create_job_work_graph(
            graph,
            idempotency_key="create-work-graph-rollback-r6-02",
        )

    assert store.list_items(spec.job_id) == ()
    assert store.list_work_units(job_id=spec.job_id) == ()
    with pytest.raises(RecordNotFoundError):
        store.get_job_work_graph(spec.job_id)
    assert store.list_outbox() == outbox_before

    monkeypatch.setattr(store, "_append_outbox", original)
    assert (
        store.create_job_work_graph(
            graph,
            idempotency_key="create-work-graph-rollback-r6-02",
        )
        == graph
    )


def test_work_readiness_snapshot_is_append_only_and_idempotent(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-work-graph-readiness-r6-02",
    )
    root = graph.work_units[0]
    snapshot = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=root,
        stage_completions=(),
        item_records=(),
        audit=_audit(),
    )

    first = store.record_work_readiness(
        snapshot,
        idempotency_key="record-work-readiness-r6-02",
    )
    replay = store.record_work_readiness(
        snapshot,
        idempotency_key="record-work-readiness-r6-02",
    )
    safety = next(unit for unit in graph.work_units if unit.stage is StageNameV2.SAFETY)
    waiting = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=safety,
        stage_completions=(),
        item_records=(),
        audit=_audit(),
    )
    persisted_waiting = store.record_work_readiness(
        waiting,
        idempotency_key="record-work-readiness-waiting-r6-02",
    )

    assert replay == first == snapshot
    assert store.list_work_readiness(root.resolved_work_unit_id) == (snapshot,)
    assert persisted_waiting == waiting
    assert store.list_work_readiness(safety.resolved_work_unit_id) == (waiting,)


def test_work_readiness_rolls_back_snapshot_and_idempotency_with_outbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-work-graph-readiness-rollback-r6-02",
    )
    root = graph.work_units[0]
    snapshot = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=root,
        stage_completions=(),
        item_records=(),
        audit=_audit(),
    )
    outbox_before = store.list_outbox()
    original = store._append_outbox

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected readiness outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(RuntimeError, match="readiness outbox failure"):
        store.record_work_readiness(
            snapshot,
            idempotency_key="record-readiness-rollback-r6-02",
        )

    assert store.list_work_readiness(root.resolved_work_unit_id) == ()
    assert store.list_outbox() == outbox_before

    monkeypatch.setattr(store, "_append_outbox", original)
    assert (
        store.record_work_readiness(
            snapshot,
            idempotency_key="record-readiness-rollback-r6-02",
        )
        == snapshot
    )


def test_work_readiness_rejects_unpersisted_stage_result_evidence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-work-graph-readiness-evidence-r6-02",
    )
    trace = next(unit for unit in graph.work_units if unit.stage is StageNameV2.TRACE_INDEX)
    safety = next(unit for unit in graph.work_units if unit.stage is StageNameV2.SAFETY)
    snapshot = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=safety,
        stage_completions=(
            _completion(
                trace,
                status=StageRunStatus.SUCCEEDED,
                suffix="unpersisted-stage-result",
            ),
        ),
        item_records=(),
        audit=_audit(),
    )

    with pytest.raises(
        RecordNotFoundError,
        match="StageResultRecord",
    ):
        store.record_work_readiness(
            snapshot,
            idempotency_key="record-unpersisted-readiness-r6-02",
        )


@pytest.mark.asyncio
async def test_artifact_group_fanout_persists_exact_units_without_stage_runs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-work-graph-artifacts-r6-02",
    )
    parent = next(unit for unit in graph.work_units if unit.stage is StageNameV2.ATTACHMENT)
    execution_plan = await _execution_plan(
        tmp_path / "artifact-fanout",
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
        artifact_execution_planning_result=_planning_result(execution_plan),
        audit=_audit(),
    )

    first = store.create_artifact_group_fanout(
        fanout,
        idempotency_key="create-artifact-fanout-r6-02",
    )
    replay = store.create_artifact_group_fanout(
        fanout,
        idempotency_key="create-artifact-fanout-r6-02",
    )

    assert first == replay == fanout
    assert store.get_artifact_group_fanout(parent.resolved_work_unit_id) == fanout
    assert store.list_work_units(job_id=spec.job_id) == (
        *graph.work_units,
        *fanout.group_work_units,
    )
    assert store.list_stage_runs(job_id=spec.job_id) == ()
    assert [event.event_type for event in store.list_outbox()].count("artifact-group-fanout-created") == 1


@pytest.mark.asyncio
async def test_artifact_group_fanout_rolls_back_units_and_outbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    spec = _job_spec()
    plan = _plan(spec)
    graph = _work_graph(spec)
    store.create_planned_job(spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-work-graph-artifact-rollback-r6-02",
    )
    parent = next(unit for unit in graph.work_units if unit.stage is StageNameV2.ATTACHMENT)
    execution_plan = await _execution_plan(
        tmp_path / "artifact-fanout-rollback",
        (("artifact://a", "inputs/a.txt"),),
        (_definition("artifact://a"),),
    )
    fanout = ArtifactGroupFanoutCompiler().compile(
        graph=graph,
        parent_attachment_work_unit=parent,
        artifact_execution_planning_result=_planning_result(execution_plan),
        audit=_audit(),
    )
    outbox_before = store.list_outbox()
    original = store._append_outbox

    def fail_outbox(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected artifact fanout outbox failure")

    monkeypatch.setattr(store, "_append_outbox", fail_outbox)
    with pytest.raises(
        RuntimeError,
        match="artifact fanout outbox failure",
    ):
        store.create_artifact_group_fanout(
            fanout,
            idempotency_key="create-artifact-fanout-rollback-r6-02",
        )

    with pytest.raises(RecordNotFoundError):
        store.get_artifact_group_fanout(parent.resolved_work_unit_id)
    assert store.list_work_units(job_id=spec.job_id) == graph.work_units
    assert store.list_outbox() == outbox_before

    monkeypatch.setattr(store, "_append_outbox", original)
    assert (
        store.create_artifact_group_fanout(
            fanout,
            idempotency_key="create-artifact-fanout-rollback-r6-02",
        )
        == fanout
    )
