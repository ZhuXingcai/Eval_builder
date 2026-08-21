from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest
from test_release_projection import (
    _audit,
    _policy,
    _ref,
    _source,
)
from test_user_approval_requests import _generation_policy
from test_user_decisions import (
    _authentication,
    _handling_policy,
)

from eval_factory.approval.persistence import (
    UserDecisionPersistenceService,
)
from eval_factory.approval.requests import (
    FinalDatasetReviewApprovalSource,
    UserApprovalRequestCompiler,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ApprovalMode,
    FinalReviewScope,
    UserDecision,
)
from eval_factory.contracts.approval_v2 import (
    FinalDatasetReviewPreviewV2,
    user_approval_request_ref,
)
from eval_factory.contracts.batch_quality_v2 import (
    BatchQualityReportV2,
    batch_quality_report_v2_ref,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.orchestration import (
    ItemStatus,
    JobStatus,
    StageRunStatus,
)
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    WorkLeaseStateV2,
    WorkReadinessV2,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
)
from eval_factory.contracts.quality_v2 import (
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.release_v2 import ReleaseStateV2
from eval_factory.dataset import (
    ReleaseProjectionConflictError,
    ReleaseProjectionIntegrityError,
    ReleaseProjectionPendingError,
    ReleaseProjectionPersistenceService,
)
from eval_factory.orchestration import (
    DatasetJobWorkGraphCompiler,
    IdempotencyConflictError,
    JobStore,
    RecordNotFoundError,
    StageWorkCompletion,
    WorkControlService,
    WorkReadinessEvaluator,
)
from eval_factory.orchestration.models import (
    StageResultRecord,
    stage_result_record_carried_sha256,
)
from eval_factory.orchestration.planning import DatasetJobPlanCompiler


def test_release_projection_schema_is_additive_and_reopen_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "factory.sqlite3"
    JobStore(path)
    JobStore(path)

    with sqlite3.connect(path) as connection:
        names = {
            str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert {
        "release_projection_policies",
        "evaluation_query_specs",
        "evaluation_item_release_subjects",
        "release_decisions_v2",
        "evaluation_items_v2",
        "item_release_projection_events",
        "release_projection_results",
        "item_release_current_projections",
    }.issubset(names)


def _stage_output(unit_id: str) -> ObjectRef:
    digest = hashlib.sha256(unit_id.encode()).hexdigest()
    return ObjectRef(
        object_type="stage-output",
        object_id=f"stage-output://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _rebind_batch_quality(
    source,
    *,
    graph,
) -> BatchQualityReportV2:
    report = source.batch_quality
    return BatchQualityReportV2.create(
        batch_id=report.batch_id,
        resolved_job_work_graph_ref=resolved_job_work_graph_v2_ref(graph),
        policy_ref=report.policy_ref,
        duplicate_policy_ref=report.duplicate_policy_ref,
        duplicate_result_ref=report.duplicate_result_ref,
        cross_item_policy_ref=report.cross_item_policy_ref,
        cross_item_result_ref=report.cross_item_result_ref,
        item_ids=report.item_ids,
        item_quality_result_refs=report.item_quality_result_refs,
        base_item_quality_report_refs=report.base_item_quality_report_refs,
        current_item_quality_report_refs=(report.current_item_quality_report_refs),
        lineage_audits=report.lineage_audits,
        batch_lineage_checks=report.batch_lineage_checks,
        findings=report.findings,
        item_quality_report_revisions=(report.item_quality_report_revisions),
        duplicate_pair_refs=report.duplicate_pair_refs,
        duplicate_cluster_refs=report.duplicate_cluster_refs,
        visible_match_refs=report.visible_match_refs,
        answer_reuse_pair_refs=report.answer_reuse_pair_refs,
        safety_cluster_refs=report.safety_cluster_refs,
        invalidated_result_refs=report.invalidated_result_refs,
        audit=_audit(),
    )


async def _prepared_release_job(
    tmp_path: Path,
    *,
    mode: ApprovalMode = ApprovalMode.NONE,
):
    final_scope = (
        FinalReviewScope.SELECTED_ITEMS
        if mode in {ApprovalMode.FINAL_ONLY, ApprovalMode.PLAN_AND_FINAL}
        else FinalReviewScope.NONE
    )
    original_sources = tuple(
        [
            await _source(
                mode=mode,
                final_scope=final_scope,
                item_index=0,
            ),
            await _source(
                mode=mode,
                final_scope=final_scope,
                item_index=1,
            ),
        ]
    )
    job_spec = original_sources[0].job_spec
    plan = DatasetJobPlanCompiler().compile(
        job_spec=job_spec,
        audit=_audit(),
    )
    graph = DatasetJobWorkGraphCompiler().compile(
        job_spec=job_spec,
        resolved_plan=plan,
        audit=_audit(),
    )
    assert graph.item_ids == tuple(source.item.item_id for source in original_sources)
    batch_quality = _rebind_batch_quality(
        original_sources[0],
        graph=graph,
    )

    store = JobStore(
        tmp_path / "factory.sqlite3",
        clock=lambda: _audit().created_at,
    )
    store.create_planned_job(job_spec, plan)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-r7-08-work-graph",
    )
    job = store.get_job(job_spec.job_id)
    store.transition_job(
        job.job_id,
        JobStatus.RUNNING,
        expected_version=job.row_version,
        idempotency_key="start-r7-08-job",
    )

    sources = []
    for original in original_sources:
        item = store.get_item(original.item.item_id)
        running = store.transition_item(
            item.item_id,
            ItemStatus.RUNNING,
            expected_version=item.row_version,
            idempotency_key=f"start-r7-08-item:{item.item_id}",
        )
        sources.append(
            replace(
                original,
                item=running,
                resolved_job_work_graph=graph,
                batch_quality=batch_quality,
            )
        )
    current_sources = tuple(sources)

    control = WorkControlService(store)
    control_policy = control.bind_policy(
        graph=graph,
        lease_duration_seconds=300,
        heartbeat_extension_seconds=60,
        max_attempts=1,
        retry_delay_seconds=(),
        retry_lease_expiry=False,
        audit=_audit(),
        idempotency_key="bind-r7-08-control-policy",
    )
    holder = ObjectRef(
        object_type="worker-principal",
        object_id="worker-principal://r7-08/release",
        object_version="v1",
        object_sha256="a" * 64,
    )
    completions: list[StageWorkCompletion] = []
    release_unit = next(unit for unit in graph.work_units if unit.stage is StageNameV2.RELEASE)
    for unit in graph.work_units:
        if unit is release_unit:
            continue
        readiness = WorkReadinessEvaluator().evaluate(
            graph=graph,
            work_unit=unit,
            stage_completions=tuple(completions),
            item_records=store.list_items(graph.job_id),
            audit=_audit(),
        )
        assert readiness.readiness is WorkReadinessV2.READY
        store.record_work_readiness(
            readiness,
            idempotency_key=(f"ready-r7-08:{unit.resolved_work_unit_id}"),
        )
        lease = control.acquire(
            graph=graph,
            work_unit=unit,
            readiness_snapshot=readiness,
            policy=control_policy,
            holder_ref=holder,
            retry_decision=None,
            audit=_audit(),
            idempotency_key=(f"acquire-r7-08:{unit.resolved_work_unit_id}"),
        )
        if unit.stage is StageNameV2.ITEM_QUALITY:
            source = next(value for value in current_sources if value.item.item_id == unit.item_id)
            output_refs = (item_quality_compilation_result_ref(source.item_quality),)
        elif unit.stage is StageNameV2.BATCH_QUALITY:
            output_refs = (batch_quality_report_v2_ref(current_sources[0].batch_quality),)
        else:
            output_refs = (_stage_output(unit.resolved_work_unit_id),)
        control.complete_stage(
            lease=lease,
            policy=control_policy,
            holder_ref=holder,
            expected_lease_version=0,
            status=StageRunStatus.SUCCEEDED,
            output_refs=output_refs,
            failure=None,
            checkpoint_ref=None,
            metrics_ref=None,
            audit=_audit(),
            idempotency_key=(f"complete-r7-08:{unit.resolved_work_unit_id}"),
        )
        assert lease.stage_run_ref is not None
        stage_run = store.get_stage_run(lease.stage_run_ref.object_id)
        stage_result = store.get_stage_result_for_run(stage_run.stage_run_id)
        assert stage_result is not None
        completions.append(
            StageWorkCompletion(
                work_unit_ref=resolved_work_unit_v2_ref(unit),
                stage_run=stage_run,
                stage_result=stage_result,
            )
        )

    release_readiness = WorkReadinessEvaluator().evaluate(
        graph=graph,
        work_unit=release_unit,
        stage_completions=tuple(completions),
        item_records=store.list_items(graph.job_id),
        audit=_audit(),
    )
    assert release_readiness.readiness is WorkReadinessV2.READY
    store.record_work_readiness(
        release_readiness,
        idempotency_key="ready-r7-08-release",
    )
    return (
        store,
        current_sources,
        control,
        control_policy,
        holder,
        release_unit,
        release_readiness,
    )


def _commit_final_decision(
    store: JobStore,
    *,
    source,
    candidate,
    decision: UserDecision,
    suffix: str,
):
    preview = FinalDatasetReviewPreviewV2.create(
        scope=FinalReviewScope.SELECTED_ITEMS,
        subject_refs=(candidate.evaluation_item_ref,),
        prompt_projection_refs=(),
        selected_item_projection_refs=(_ref("user-item-projection", suffix),),
        dataset_projection_ref=None,
        quality_summary_refs=(batch_quality_report_v2_ref(source.batch_quality),),
        open_low_severity_finding_refs=(),
        sample_navigation_refs=(_ref("approval-navigation-sample", suffix),),
        projection_policy_version=("user-approval-projection/r7-04-v1"),
        audit=_audit(),
    )
    checkpoint_source = FinalDatasetReviewApprovalSource(preview=preview)
    request_result = UserApprovalRequestCompiler().compile(
        job_spec=source.job_spec,
        approval_policy=source.approval_policy,
        generation_policy=_generation_policy(),
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(checkpoint_source,),
        requested_by="requesting-user",
        audit=_audit(),
    )
    request = request_result.requests[0]
    return UserDecisionPersistenceService(store).commit(
        job_spec=source.job_spec,
        approval_policy=source.approval_policy,
        generation_policy=_generation_policy(),
        request_compilation=request_result,
        checkpoint_sources=(checkpoint_source,),
        request_ref=user_approval_request_ref(request),
        handling_policy=_handling_policy(),
        authentication=_authentication(),
        decision=decision,
        adjustments=(),
        adjustment_effect=None,
        environment_decisions=(),
        query_packaging=None,
        reason=f"{decision.value} final release candidate.",
        idempotency_key=f"commit-r7-08-final-{suffix}",
        audit=_audit(),
    )


@pytest.mark.asyncio
async def test_candidate_persistence_is_atomic_and_exactly_replayable(
    tmp_path: Path,
) -> None:
    store, sources, *_ = await _prepared_release_job(tmp_path)
    service = ReleaseProjectionPersistenceService(store)

    candidate = service.request_release(
        source=sources[0],
        policy=_policy(),
        idempotency_key="request-r7-08-item-0",
        audit=_audit(),
    )
    outbox_count = len(store.list_outbox())
    replay = service.request_release(
        source=sources[0],
        policy=_policy(),
        idempotency_key="request-r7-08-item-0",
        audit=_audit(),
    )

    assert replay == candidate
    assert service.get_current_result(sources[0].item.item_id) == candidate
    assert store.get_item(sources[0].item.item_id).status is ItemStatus.RUNNING
    assert len(store.list_outbox()) == outbox_count
    with sqlite3.connect(store.path) as connection:
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "release_projection_policies",
                "evaluation_query_specs",
                "evaluation_item_release_subjects",
                "release_decisions_v2",
                "evaluation_items_v2",
                "item_release_projection_events",
                "release_projection_results",
                "item_release_current_projections",
            )
        }
    assert set(counts.values()) == {1}


@pytest.mark.asyncio
async def test_candidate_conflicts_on_changed_key_or_changed_same_key(
    tmp_path: Path,
) -> None:
    store, sources, *_ = await _prepared_release_job(tmp_path)
    service = ReleaseProjectionPersistenceService(store)
    service.request_release(
        source=sources[0],
        policy=_policy(),
        idempotency_key="request-r7-08-conflict",
        audit=_audit(),
    )

    with pytest.raises(
        ReleaseProjectionConflictError,
        match="already represents",
    ):
        service.request_release(
            source=sources[0],
            policy=_policy(),
            idempotency_key="request-r7-08-second-authority",
            audit=_audit(),
        )

    base_policy = _policy()
    changed_policy = type(base_policy).create(
        max_source_trace_refs=base_policy.max_source_trace_refs,
        max_label_decision_refs=base_policy.max_label_decision_refs,
        max_user_decision_refs=base_policy.max_user_decision_refs,
        max_revalidation_reports=base_policy.max_revalidation_reports,
        max_current_head_refs=base_policy.max_current_head_refs,
        max_chain_depth=base_policy.max_chain_depth - 1,
        allowed_channels=base_policy.allowed_channels,
        allowed_export_profiles=base_policy.allowed_export_profiles,
        audit=_audit(),
    )
    with pytest.raises(
        IdempotencyConflictError,
        match="different request",
    ):
        service.request_release(
            source=sources[0],
            policy=changed_policy,
            idempotency_key="request-r7-08-conflict",
            audit=_audit(),
        )


@pytest.mark.parametrize(
    ("keys", "expected_conflicts"),
    (
        (
            ("request-r7-08-race", "request-r7-08-race"),
            0,
        ),
        (
            (
                "request-r7-08-race-a",
                "request-r7-08-race-b",
            ),
            1,
        ),
    ),
)
@pytest.mark.asyncio
async def test_two_worker_candidate_race_has_one_authority(
    tmp_path: Path,
    keys: tuple[str, str],
    expected_conflicts: int,
) -> None:
    store, sources, *_ = await _prepared_release_job(tmp_path)
    barrier = Barrier(2)

    def request(key: str) -> str:
        barrier.wait()
        try:
            result = ReleaseProjectionPersistenceService(store).request_release(
                source=sources[0],
                policy=_policy(),
                idempotency_key=key,
                audit=_audit(),
            )
        except ReleaseProjectionConflictError:
            return "CONFLICT"
        return result.result_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        observed = tuple(executor.map(request, keys))

    assert observed.count("CONFLICT") == expected_conflicts
    committed_ids = {value for value in observed if value != "CONFLICT"}
    assert len(committed_ids) == 1
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM release_projection_results").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_candidate_requires_successful_quality_stage_witnesses(
    tmp_path: Path,
) -> None:
    store, sources, *_ = await _prepared_release_job(tmp_path)
    item_quality_unit = next(
        unit
        for unit in sources[0].resolved_job_work_graph.work_units
        if unit.stage is StageNameV2.ITEM_QUALITY and unit.item_id == sources[0].item.item_id
    )
    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            """
            SELECT results.stage_result_id, results.record_json
            FROM work_leases AS leases
            JOIN controlled_stage_runs AS controlled
              ON controlled.work_lease_id = leases.work_lease_id
            JOIN stage_results AS results
              ON results.stage_run_id = controlled.stage_run_id
            WHERE leases.resolved_work_unit_id = ?
            """,
            (item_quality_unit.resolved_work_unit_id,),
        ).fetchone()
        assert row is not None
        stage_result_id = str(row[0])
        original_json = str(row[1])
        connection.execute(
            """
            UPDATE stage_results
            SET record_json = '{}'
            WHERE stage_result_id = ?
            """,
            (stage_result_id,),
        )
    service = ReleaseProjectionPersistenceService(store)
    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="quality StageResult is malformed",
    ):
        service.request_release(
            source=sources[0],
            policy=_policy(),
            idempotency_key="request-r7-08-malformed-witness",
            audit=_audit(),
        )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE stage_results
            SET record_json = ?
            WHERE stage_result_id = ?
            """,
            (original_json, stage_result_id),
        )

    with store._transaction() as connection:
        row = connection.execute(
            """
            SELECT results.record_json
            FROM work_leases AS leases
            JOIN controlled_stage_runs AS controlled
              ON controlled.work_lease_id = leases.work_lease_id
            JOIN stage_results AS results
              ON results.stage_run_id = controlled.stage_run_id
            WHERE leases.resolved_work_unit_id = ?
            """,
            (item_quality_unit.resolved_work_unit_id,),
        ).fetchone()
        assert row is not None
        result = StageResultRecord.model_validate_json(str(row["record_json"]))
        changed = result.model_copy(
            update={
                "output_refs": (),
                "result_sha256": "0" * 64,
            }
        )
        changed = changed.model_copy(update={"result_sha256": (stage_result_record_carried_sha256(changed))})
        connection.execute(
            """
            UPDATE stage_results
            SET result_sha256 = ?, record_json = ?
            WHERE stage_result_id = ?
            """,
            (
                changed.result_sha256,
                store._record_json(changed),
                changed.stage_result_id,
            ),
        )

    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="StageResult witness",
    ):
        service.request_release(
            source=sources[0],
            policy=_policy(),
            idempotency_key="request-r7-08-missing-witness",
            audit=_audit(),
        )
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM release_projection_results").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_public_guards_reject_missing_projection_and_bad_job_inputs(
    tmp_path: Path,
) -> None:
    (
        store,
        sources,
        control,
        control_policy,
        holder,
        release_unit,
        release_readiness,
    ) = await _prepared_release_job(tmp_path)
    service = ReleaseProjectionPersistenceService(store)
    candidate = service.request_release(
        source=sources[0],
        policy=_policy(),
        idempotency_key="request-r7-08-public-guards",
        audit=_audit(),
    )
    assert service.list_item_results(sources[0].item.item_id) == (candidate,)
    with pytest.raises(RecordNotFoundError, match="current"):
        service.get_current_result(sources[1].item.item_id)
    with pytest.raises(RecordNotFoundError, match="history"):
        service.rebuild_item_projection(sources[1].item.item_id)

    changed_spec = sources[1].job_spec.model_copy(update={"idempotency_key": "changed-release-source"})
    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="persisted Job",
    ):
        service.request_release(
            source=replace(sources[1], job_spec=changed_spec),
            policy=_policy(),
            idempotency_key="request-r7-08-stale-job",
            audit=_audit(),
        )

    lease = control.acquire(
        graph=sources[0].resolved_job_work_graph,
        work_unit=release_unit,
        readiness_snapshot=release_readiness,
        policy=control_policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r7-08-public-guards",
    )
    with pytest.raises(
        ReleaseProjectionConflictError,
        match="at least one",
    ):
        service.finalize_job(
            sources=(),
            policy=_policy(),
            lease=lease,
            work_control_policy=control_policy,
            holder_ref=holder,
            expected_lease_version=0,
            idempotency_key="finalize-r7-08-empty",
            audit=_audit(),
        )
    cross_job_source = replace(
        sources[1],
        item=sources[1].item.model_copy(update={"job_id": "dataset-job://other"}),
    )
    with pytest.raises(
        ReleaseProjectionConflictError,
        match="one Job",
    ):
        service.finalize_job(
            sources=(sources[0], cross_job_source),
            policy=_policy(),
            lease=lease,
            work_control_policy=control_policy,
            holder_ref=holder,
            expected_lease_version=0,
            idempotency_key="finalize-r7-08-cross-job",
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionConflictError,
        match="sorted and unique",
    ):
        service.finalize_job(
            sources=(sources[0], sources[0]),
            policy=_policy(),
            lease=lease,
            work_control_policy=control_policy,
            holder_ref=holder,
            expected_lease_version=0,
            idempotency_key="finalize-r7-08-duplicate-item",
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionConflictError,
        match="cover every current",
    ):
        service.finalize_job(
            sources=sources,
            policy=_policy(),
            lease=lease,
            work_control_policy=control_policy,
            holder_ref=holder,
            expected_lease_version=0,
            idempotency_key="finalize-r7-08-missing-candidate",
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_final_review_pending_rolls_back_entire_job_finalization(
    tmp_path: Path,
) -> None:
    (
        store,
        sources,
        control,
        control_policy,
        holder,
        release_unit,
        release_readiness,
    ) = await _prepared_release_job(
        tmp_path,
        mode=ApprovalMode.FINAL_ONLY,
    )
    service = ReleaseProjectionPersistenceService(store)
    for index, source in enumerate(sources):
        candidate = service.request_release(
            source=source,
            policy=_policy(),
            idempotency_key=f"request-r7-08-pending-{index}",
            audit=_audit(),
        )
        assert candidate.item_projection.item_status is ItemStatus.NEEDS_REVIEW
    lease = control.acquire(
        graph=sources[0].resolved_job_work_graph,
        work_unit=release_unit,
        readiness_snapshot=release_readiness,
        policy=control_policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r7-08-pending-release",
    )

    with pytest.raises(
        ReleaseProjectionPendingError,
        match="pending",
    ):
        service.finalize_job(
            sources=sources,
            policy=_policy(),
            lease=lease,
            work_control_policy=control_policy,
            holder_ref=holder,
            expected_lease_version=0,
            idempotency_key="finalize-r7-08-pending",
            audit=_audit(),
        )

    assert all(store.get_item(source.item.item_id).status is ItemStatus.NEEDS_REVIEW for source in sources)
    assert store.get_work_lease_head(lease.work_lease_id).state is WorkLeaseStateV2.ACTIVE
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM release_projection_results
                WHERE phase = 'TERMINAL'
                """
            ).fetchone()[0]
            == 0
        )


@pytest.mark.asyncio
async def test_current_projection_drift_requires_explicit_rebuild(
    tmp_path: Path,
) -> None:
    store, sources, *_ = await _prepared_release_job(tmp_path)
    service = ReleaseProjectionPersistenceService(store)
    candidate = service.request_release(
        source=sources[0],
        policy=_policy(),
        idempotency_key="request-r7-08-rebuild",
        audit=_audit(),
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE item_release_current_projections
            SET item_status = 'APPROVED'
            WHERE item_id = ?
            """,
            (sources[0].item.item_id,),
        )

    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="columns",
    ):
        service.get_current_result(sources[0].item.item_id)

    rebuilt = service.rebuild_item_projection(sources[0].item.item_id)
    assert rebuilt == candidate
    assert service.get_current_result(sources[0].item.item_id) == candidate


@pytest.mark.asyncio
async def test_replay_metadata_stage_binding_and_item_projection_corruption(
    tmp_path: Path,
) -> None:
    store, sources, *_ = await _prepared_release_job(tmp_path)
    service = ReleaseProjectionPersistenceService(store)
    candidate = service.request_release(
        source=sources[0],
        policy=_policy(),
        idempotency_key="request-r7-08-corruption",
        audit=_audit(),
    )

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE idempotency_records
            SET response_type = 'WRONG_RESPONSE'
            WHERE scope = ? AND idempotency_key = ?
            """,
            (
                f"request-release:{sources[0].item.item_id}",
                "request-r7-08-corruption",
            ),
        )
    with pytest.raises(
        IdempotencyConflictError,
        match="response type",
    ):
        service.request_release(
            source=sources[0],
            policy=_policy(),
            idempotency_key="request-r7-08-corruption",
            audit=_audit(),
        )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE idempotency_records
            SET response_type = 'RELEASE_PROJECTION_CANDIDATE'
            WHERE scope = ? AND idempotency_key = ?
            """,
            (
                f"request-release:{sources[0].item.item_id}",
                "request-r7-08-corruption",
            ),
        )
        stage_result_id = str(
            connection.execute("SELECT stage_result_id FROM stage_results LIMIT 1").fetchone()[0]
        )
        connection.execute(
            """
            UPDATE release_projection_results
            SET stage_result_id = ?
            WHERE result_id = ?
            """,
            (stage_result_id, candidate.result_id),
        )
    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="candidate release result",
    ):
        service.get_result(candidate.result_id)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE release_projection_results
            SET stage_result_id = NULL
            WHERE result_id = ?
            """,
            (candidate.result_id,),
        )

    item = store.get_item(sources[0].item.item_id)
    drifted_item = type(item).model_validate(
        {
            **item.model_dump(mode="python"),
            "status": ItemStatus.NEEDS_REVIEW,
            "row_version": item.row_version + 1,
        }
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE items
            SET status = ?, row_version = ?, record_json = ?
            WHERE item_id = ?
            """,
            (
                drifted_item.status.value,
                drifted_item.row_version,
                store._record_json(drifted_item),
                drifted_item.item_id,
            ),
        )
    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="Item status differs",
    ):
        service.get_current_result(drifted_item.item_id)
    assert service.rebuild_item_projection(drifted_item.item_id) == candidate
    assert store.get_item(drifted_item.item_id).status is ItemStatus.RUNNING

    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM release_projection_policies")
    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="missing its projection policy",
    ):
        service.get_result(candidate.result_id)


@pytest.mark.parametrize(
    ("table", "column", "value", "message"),
    (
        (
            "evaluation_query_specs",
            "prompt_sha256",
            "f" * 64,
            "QuerySpec columns",
        ),
        (
            "evaluation_item_release_subjects",
            "release_subject_sha256",
            "f" * 64,
            "subject columns",
        ),
        (
            "release_decisions_v2",
            "state",
            "REJECTED",
            "ReleaseDecision columns",
        ),
        (
            "evaluation_items_v2",
            "item_sha256",
            "f" * 64,
            "EvaluationItem columns",
        ),
        (
            "item_release_projection_events",
            "release_state",
            "APPROVED",
            "projection columns",
        ),
        (
            "release_projection_results",
            "result_sha256",
            "f" * 64,
            "result columns",
        ),
    ),
)
@pytest.mark.asyncio
async def test_immutable_release_row_drift_is_detected(
    tmp_path: Path,
    table: str,
    column: str,
    value: str,
    message: str,
) -> None:
    store, sources, *_ = await _prepared_release_job(tmp_path)
    service = ReleaseProjectionPersistenceService(store)
    candidate = service.request_release(
        source=sources[0],
        policy=_policy(),
        idempotency_key="request-r7-08-row-drift",
        audit=_audit(),
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            f"UPDATE {table} SET {column} = ?",
            (value,),
        )

    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match=message,
    ):
        service.get_result(candidate.result_id)


@pytest.mark.asyncio
async def test_malformed_release_json_is_rejected_at_every_read_boundary(
    tmp_path: Path,
) -> None:
    store, sources, *_ = await _prepared_release_job(tmp_path)
    service = ReleaseProjectionPersistenceService(store)
    candidate = service.request_release(
        source=sources[0],
        policy=_policy(),
        idempotency_key="request-r7-08-malformed-json",
        audit=_audit(),
    )
    cases = (
        (
            "release_projection_policies",
            "projection policy is malformed",
            False,
        ),
        ("evaluation_query_specs", "QuerySpec is malformed", False),
        (
            "evaluation_item_release_subjects",
            "release subject is malformed",
            False,
        ),
        (
            "release_decisions_v2",
            "ReleaseDecision is malformed",
            False,
        ),
        (
            "evaluation_items_v2",
            "EvaluationItem is malformed",
            False,
        ),
        (
            "item_release_projection_events",
            "Item release projection is malformed",
            False,
        ),
        (
            "release_projection_results",
            "projection result is malformed",
            False,
        ),
        (
            "item_release_current_projections",
            "current release projection is malformed",
            True,
        ),
    )
    for table, message, current_only in cases:
        with sqlite3.connect(store.path) as connection:
            row = connection.execute(f"SELECT record_json FROM {table}").fetchone()
            assert row is not None
            original = str(row[0])
            connection.execute(f"UPDATE {table} SET record_json = '{{}}'")
        with pytest.raises(
            ReleaseProjectionIntegrityError,
            match=message,
        ):
            if current_only:
                service.get_current_result(sources[0].item.item_id)
            else:
                service.get_result(candidate.result_id)
        with sqlite3.connect(store.path) as connection:
            connection.execute(
                f"UPDATE {table} SET record_json = ?",
                (original,),
            )


@pytest.mark.asyncio
async def test_candidate_write_fault_rolls_back_every_release_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, sources, *_ = await _prepared_release_job(tmp_path)
    original_append = store._append_outbox

    def fail_release_outbox(
        connection,
        **kwargs,
    ):
        if kwargs["event_type"] == "item-release-candidate-projected":
            raise RuntimeError("injected candidate release outbox failure")
        return original_append(connection, **kwargs)

    monkeypatch.setattr(store, "_append_outbox", fail_release_outbox)
    service = ReleaseProjectionPersistenceService(store)
    with pytest.raises(
        RuntimeError,
        match="injected candidate",
    ):
        service.request_release(
            source=sources[0],
            policy=_policy(),
            idempotency_key="request-r7-08-fault",
            audit=_audit(),
        )

    with sqlite3.connect(store.path) as connection:
        for table in (
            "release_projection_policies",
            "evaluation_query_specs",
            "evaluation_item_release_subjects",
            "release_decisions_v2",
            "evaluation_items_v2",
            "item_release_projection_events",
            "release_projection_results",
            "item_release_current_projections",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert store.get_item(sources[0].item.item_id).status is ItemStatus.RUNNING


@pytest.mark.asyncio
async def test_terminal_write_fault_rolls_back_items_stage_and_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        store,
        sources,
        control,
        control_policy,
        holder,
        release_unit,
        release_readiness,
    ) = await _prepared_release_job(tmp_path)
    service = ReleaseProjectionPersistenceService(store)
    for index, source in enumerate(sources):
        service.request_release(
            source=source,
            policy=_policy(),
            idempotency_key=f"request-r7-08-fault-{index}",
            audit=_audit(),
        )
    lease = control.acquire(
        graph=sources[0].resolved_job_work_graph,
        work_unit=release_unit,
        readiness_snapshot=release_readiness,
        policy=control_policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r7-08-fault-release",
    )
    with sqlite3.connect(store.path) as connection:
        stage_result_count = int(connection.execute("SELECT COUNT(*) FROM stage_results").fetchone()[0])

    original_append = store._append_outbox

    def fail_job_outbox(
        connection,
        **kwargs,
    ):
        if kwargs["event_type"] == "release-job-finalized":
            raise RuntimeError("injected terminal release outbox failure")
        return original_append(connection, **kwargs)

    monkeypatch.setattr(store, "_append_outbox", fail_job_outbox)
    with pytest.raises(
        RuntimeError,
        match="injected terminal",
    ):
        service.finalize_job(
            sources=sources,
            policy=_policy(),
            lease=lease,
            work_control_policy=control_policy,
            holder_ref=holder,
            expected_lease_version=0,
            idempotency_key="finalize-r7-08-fault",
            audit=_audit(),
        )

    assert all(store.get_item(source.item.item_id).status is ItemStatus.RUNNING for source in sources)
    assert store.get_work_lease_head(lease.work_lease_id).state is WorkLeaseStateV2.ACTIVE
    assert lease.stage_run_ref is not None
    assert store.get_stage_result_for_run(lease.stage_run_ref.object_id) is None
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM release_projection_results
                WHERE phase = 'TERMINAL'
                """
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT COUNT(*) FROM stage_results").fetchone()[0] == stage_result_count


@pytest.mark.asyncio
async def test_finalize_job_commits_all_items_and_one_release_stage_result(
    tmp_path: Path,
) -> None:
    (
        store,
        sources,
        control,
        control_policy,
        holder,
        release_unit,
        release_readiness,
    ) = await _prepared_release_job(tmp_path)
    service = ReleaseProjectionPersistenceService(store)
    for index, source in enumerate(sources):
        service.request_release(
            source=source,
            policy=_policy(),
            idempotency_key=f"request-r7-08-item-{index}",
            audit=_audit(),
        )
    lease = control.acquire(
        graph=sources[0].resolved_job_work_graph,
        work_unit=release_unit,
        readiness_snapshot=release_readiness,
        policy=control_policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r7-08-release",
    )

    terminal = service.finalize_job(
        sources=sources,
        policy=_policy(),
        lease=lease,
        work_control_policy=control_policy,
        holder_ref=holder,
        expected_lease_version=0,
        idempotency_key="finalize-r7-08-job",
        audit=_audit(),
    )
    replay = service.finalize_job(
        sources=sources,
        policy=_policy(),
        lease=lease,
        work_control_policy=control_policy,
        holder_ref=holder,
        expected_lease_version=0,
        idempotency_key="finalize-r7-08-job",
        audit=_audit(),
    )

    assert replay == terminal
    assert len(terminal) == 2
    assert {result.release_decision.state for result in terminal} == {ReleaseStateV2.APPROVED}
    assert all(store.get_item(source.item.item_id).status is ItemStatus.APPROVED for source in sources)
    assert lease.stage_run_ref is not None
    stage_result = store.get_stage_result_for_run(lease.stage_run_ref.object_id)
    assert stage_result is not None
    assert stage_result.status is StageRunStatus.SUCCEEDED
    assert stage_result.output_refs == tuple(result.to_ref() for result in terminal)
    terminal_by_item = {result.release_subject.item_id: result for result in terminal}
    for source in sources:
        history = service.list_item_results(source.item.item_id)
        assert history == (
            history[0],
            terminal_by_item[source.item.item_id],
        )
    with sqlite3.connect(store.path) as connection:
        stage_result_ids = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT stage_result_id
                FROM release_projection_results
                WHERE phase = 'TERMINAL'
                """
            )
        }
    assert stage_result_ids == {stage_result.stage_result_id}

    with pytest.raises(
        ReleaseProjectionConflictError,
        match="non-terminal Job Item",
    ):
        service.finalize_job(
            sources=sources,
            policy=_policy(),
            lease=lease,
            work_control_policy=control_policy,
            holder_ref=holder,
            expected_lease_version=0,
            idempotency_key="finalize-r7-08-second-authority",
            audit=_audit(),
        )

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE release_projection_results
            SET stage_result_id = NULL
            WHERE result_id = ?
            """,
            (terminal[0].result_id,),
        )
    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="StageResult binding",
    ):
        service.finalize_job(
            sources=sources,
            policy=_policy(),
            lease=lease,
            work_control_policy=control_policy,
            holder_ref=holder,
            expected_lease_version=0,
            idempotency_key="finalize-r7-08-job",
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_finalize_job_persists_mixed_accepted_and_user_rejected_items(
    tmp_path: Path,
) -> None:
    (
        store,
        sources,
        control,
        control_policy,
        holder,
        release_unit,
        release_readiness,
    ) = await _prepared_release_job(
        tmp_path,
        mode=ApprovalMode.FINAL_ONLY,
    )
    service = ReleaseProjectionPersistenceService(store)
    candidates = tuple(
        service.request_release(
            source=source,
            policy=_policy(),
            idempotency_key=f"request-r7-08-mixed-{index}",
            audit=_audit(),
        )
        for index, source in enumerate(sources)
    )
    commits = (
        _commit_final_decision(
            store,
            source=sources[0],
            candidate=candidates[0],
            decision=UserDecision.ACCEPT,
            suffix="accepted",
        ),
        _commit_final_decision(
            store,
            source=sources[1],
            candidate=candidates[1],
            decision=UserDecision.REJECT,
            suffix="rejected",
        ),
    )
    terminal_sources = tuple(
        replace(source, decision_commits=(commit,)) for source, commit in zip(sources, commits, strict=True)
    )
    lease = control.acquire(
        graph=sources[0].resolved_job_work_graph,
        work_unit=release_unit,
        readiness_snapshot=release_readiness,
        policy=control_policy,
        holder_ref=holder,
        retry_decision=None,
        audit=_audit(),
        idempotency_key="acquire-r7-08-mixed-release",
    )

    terminal = service.finalize_job(
        sources=terminal_sources,
        policy=_policy(),
        lease=lease,
        work_control_policy=control_policy,
        holder_ref=holder,
        expected_lease_version=0,
        idempotency_key="finalize-r7-08-mixed",
        audit=_audit(),
    )

    states = {result.release_subject.item_id: result.release_decision.state for result in terminal}
    assert states == {
        sources[0].item.item_id: ReleaseStateV2.APPROVED,
        sources[1].item.item_id: ReleaseStateV2.REJECTED,
    }
    assert store.get_item(sources[0].item.item_id).status is ItemStatus.APPROVED
    assert store.get_item(sources[1].item.item_id).status is ItemStatus.REJECTED
    assert lease.stage_run_ref is not None
    stage_result = store.get_stage_result_for_run(lease.stage_run_ref.object_id)
    assert stage_result is not None
    assert stage_result.status is StageRunStatus.SUCCEEDED
    assert len(stage_result.output_refs) == 2
