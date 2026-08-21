from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest
from test_release_publication import (
    NOW,
    _audit,
    _decision,
    _item,
    _policy,
    _ref,
    _source,
)

from env_mock_agent.facade.release_export_adapter import (
    MappingLHWorkspaceExportMaterialResolver,
    RegistryLHWorkspaceExportFacade,
)
from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ItemStatus,
    ResourceBudget,
    StageRunStatus,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    StageNameV2,
)
from eval_factory.contracts.release import ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    ItemReleaseProjectionV2,
    ReleaseProjectionPhaseV2,
    ReleaseProjectionPolicyV2,
    ReleaseProjectionResultV2,
    evaluation_item_v2_ref,
    release_decision_v2_ref,
    release_projection_result_v2_ref,
)
from eval_factory.contracts.release_v2 import (
    ReleaseActionV2,
    ReleaseStateV2,
)
from eval_factory.dataset.bundle_store import (
    NonProductionReleaseBundleStore,
)
from eval_factory.dataset.persistence import (
    ReleaseProjectionIntegrityError,
    ReleaseProjectionPersistenceService,
)
from eval_factory.dataset.publication import (
    ReleasePublicationConflictError,
    ReleasePublicationIntegrityError,
    ReleasePublicationPersistenceService,
)
from eval_factory.orchestration import (
    IdempotencyConflictError,
    JobStore,
    RecordNotFoundError,
)


def test_publication_schema_is_additive_and_reopen_safe(
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
        "release_publication_policies",
        "lh_release_manifest_items",
        "nonproduction_release_manifests",
        "lh_export_receipts",
        "item_publication_projection_events",
        "nonproduction_registry_entries",
        "release_publication_results",
        "item_publication_current_projections",
    }.issubset(names)


def _release_policy() -> ReleaseProjectionPolicyV2:
    return ReleaseProjectionPolicyV2.create(
        max_source_trace_refs=8,
        max_label_decision_refs=32,
        max_user_decision_refs=32,
        max_revalidation_reports=32,
        max_current_head_refs=256,
        max_chain_depth=32,
        allowed_channels=frozenset(
            {
                ReleaseChannel.CANARY,
                ReleaseChannel.INTERNAL_REVIEW,
            }
        ),
        allowed_export_profiles=frozenset({"LH"}),
        audit=_audit(),
    )


def _job_spec(
    result: ReleaseProjectionResultV2,
) -> DatasetJobSpecV2:
    trace = result.release_subject.source_trace_refs[0]
    return DatasetJobSpecV2(
        job_id=result.release_subject.job_id,
        traces=(
            TraceSourceRef(
                source_trace_id=trace.object_id,
                source_uri="source://r7-09/item-a",
                raw_sha256=trace.object_sha256,
                adapter_name="raw-traj-v1",
                adapter_version=trace.object_version,
                processing_class="RESTRICTED_TRACE_RAW",
            ),
        ),
        requested_stages=(StageNameV2.TRACE_INDEX,),
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=10,
            max_model_tokens=10_000,
            max_processes=2,
            max_renderers=2,
            max_network_requests=10,
            max_storage_bytes=1_000_000,
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
        approval_policy_ref=(result.release_subject.user_approval_policy_ref),
        approval_mode=ApprovalMode.NONE,
        enabled_checkpoints=frozenset(),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry=result.release_decision.registry,
        ),
        idempotency_key="job-r7-09",
        audit=_audit(),
    )


def _release_chain(
    approved: ReleaseProjectionResultV2,
) -> tuple[ReleaseProjectionResultV2, ReleaseProjectionResultV2]:
    subject = approved.release_subject
    bindings = subject.satisfied_checkpoint_bindings
    candidate_decision = _decision(
        subject=subject,
        action=ReleaseActionV2.REQUEST_RELEASE,
        state=ReleaseStateV2.CANDIDATE,
        previous_decision_ref=None,
        item_version="1",
        checkpoint_bindings=bindings,
        decided_at=NOW,
    )
    candidate_item = _item(
        subject=subject,
        decision=candidate_decision,
    )
    candidate_projection = ItemReleaseProjectionV2.create(
        job_id=subject.job_id,
        item_id=subject.item_id,
        projection_revision=1,
        chain_id=candidate_decision.chain_id,
        previous_projection_ref=None,
        release_subject_ref=subject.to_ref(),
        evaluation_item_ref=evaluation_item_v2_ref(candidate_item),
        release_decision_ref=release_decision_v2_ref(candidate_decision),
        release_state=ReleaseStateV2.CANDIDATE,
        pending_checkpoints=(),
        item_status=ItemStatus.RUNNING,
        audit=_audit(),
    )
    candidate = ReleaseProjectionResultV2.create(
        phase=ReleaseProjectionPhaseV2.CANDIDATE,
        release_subject=subject,
        query_spec=approved.query_spec,
        release_decision=candidate_decision,
        evaluation_item=candidate_item,
        item_projection=candidate_projection,
        previous_result_ref=None,
        policy_ref=subject.policy_ref,
        audit=_audit(),
    )
    approved_decision = _decision(
        subject=subject,
        action=ReleaseActionV2.APPROVE,
        state=ReleaseStateV2.APPROVED,
        previous_decision_ref=candidate.release_decision_ref,
        item_version="2",
        checkpoint_bindings=bindings,
        decided_at=NOW,
    )
    approved_item = _item(
        subject=subject,
        decision=approved_decision,
    )
    approved_projection = ItemReleaseProjectionV2.create(
        job_id=subject.job_id,
        item_id=subject.item_id,
        projection_revision=2,
        chain_id=approved_decision.chain_id,
        previous_projection_ref=candidate.item_projection_ref,
        release_subject_ref=subject.to_ref(),
        evaluation_item_ref=evaluation_item_v2_ref(approved_item),
        release_decision_ref=release_decision_v2_ref(approved_decision),
        release_state=ReleaseStateV2.APPROVED,
        pending_checkpoints=(),
        item_status=ItemStatus.APPROVED,
        audit=_audit(),
    )
    terminal = ReleaseProjectionResultV2.create(
        phase=ReleaseProjectionPhaseV2.TERMINAL,
        release_subject=subject,
        query_spec=approved.query_spec,
        release_decision=approved_decision,
        evaluation_item=approved_item,
        item_projection=approved_projection,
        previous_result_ref=candidate.to_ref(),
        policy_ref=subject.policy_ref,
        audit=_audit(),
    )
    return candidate, terminal


def _prepared(
    tmp_path: Path,
    *,
    fault_injector=None,
):
    release_policy = _release_policy()
    source, _ = _source(release_projection_policy_ref=release_policy.to_ref())
    candidate, approved = _release_chain(source.approved_result)
    source = replace(source, approved_result=approved)
    store = JobStore(
        tmp_path / "factory.sqlite3",
        clock=lambda: NOW,
    )
    store.create_job(_job_spec(approved))
    item = store.create_item(
        approved.release_subject.job_id,
        approved.release_subject.item_id,
        idempotency_key="create-item-r7-09",
    )
    item = store.transition_item(
        item.item_id,
        ItemStatus.RUNNING,
        expected_version=item.row_version,
        idempotency_key="start-item-r7-09",
    )
    item = store.transition_item(
        item.item_id,
        ItemStatus.APPROVED,
        expected_version=item.row_version,
        idempotency_key="approve-item-r7-09",
    )
    run = store.create_stage_run(
        stage_run_id="stage-run://r7-09/release",
        job_id=approved.release_subject.job_id,
        item_id=None,
        stage=StageNameV2.RELEASE,
        attempt=1,
        principal_ref=_ref("worker-principal", "release"),
        input_refs=(approved.release_subject_ref,),
        idempotency_key="create-release-stage-r7-09",
    )
    run = store.transition_stage_run(
        run.stage_run_id,
        StageRunStatus.RUNNING,
        expected_version=run.row_version,
        idempotency_key="start-release-stage-r7-09",
    )
    stage_result = store.complete_stage_run(
        stage_result_id="stage-result://r7-09/release",
        stage_run_id=run.stage_run_id,
        status=StageRunStatus.SUCCEEDED,
        expected_version=run.row_version,
        idempotency_key="complete-release-stage-r7-09",
        output_refs=(release_projection_result_v2_ref(approved),),
    )
    release_service = ReleaseProjectionPersistenceService(store)
    with store._transaction() as connection:
        release_service._persist_result(
            connection,
            result=candidate,
            policy=release_policy,
            stage_result_id=None,
        )
        release_service._persist_result(
            connection,
            result=approved,
            policy=release_policy,
            stage_result_id=stage_result.stage_result_id,
        )
        release_service._insert_current_projection(
            connection,
            approved,
        )
    bundle_store = NonProductionReleaseBundleStore(
        canary_root=tmp_path / "canary",
        internal_review_root=tmp_path / "internal",
    )
    service = ReleasePublicationPersistenceService(
        store,
        bundle_store=bundle_store,
        workspace_facade=RegistryLHWorkspaceExportFacade(
            resolver=MappingLHWorkspaceExportMaterialResolver(outputs={})
        ),
        fault_injector=fault_injector,
    )
    return service, source, _policy(), store


def _publish(
    service: ReleasePublicationPersistenceService,
    source,
    policy,
    *,
    key: str = "publish-r7-09",
    rejected_item_ids: tuple[str, ...] = (),
):
    return service.publish_job(
        item_sources=(source,),
        rejected_item_ids=rejected_item_ids,
        policy=policy,
        base_contract_manifest_ref=_ref(
            "contract-manifest",
            "v1",
            version="manifest/v1",
        ),
        overlay_contract_manifest_ref=_ref(
            "contract-manifest",
            "v2",
            version="manifest/v2",
        ),
        release_profile_decision_ref=_ref(
            "release-profile-decision",
            "lh-v1",
            version="decision/v1",
        ),
        idempotency_key=key,
        audit=_audit(),
    )


def test_publish_job_is_atomic_replayable_and_successor_aware(
    tmp_path: Path,
) -> None:
    service, source, policy, store = _prepared(tmp_path)
    outbox_before = len(store.list_outbox())
    stages_before = len(store.list_stage_runs(job_id=source.approved_result.release_subject.job_id))

    first = _publish(service, source, policy)
    replay = _publish(service, source, policy)

    assert replay == first
    with sqlite3.connect(store.path) as connection:
        publication_json = str(
            connection.execute(
                """
                SELECT record_json
                FROM release_publication_results
                WHERE result_id = ?
                """,
                (first.result_id,),
            ).fetchone()[0]
        )
    assert source.approved_result.query_spec.prompt not in publication_json
    assert "approved_source_result_refs" in publication_json
    assert store.get_item(source.approved_result.release_subject.item_id).status is ItemStatus.RELEASED
    assert len(store.list_stage_runs(job_id=source.approved_result.release_subject.job_id)) == stages_before
    assert len(store.list_outbox()) == outbox_before + 2
    assert service.get_publication(first.result_id) == first
    assert service.get_registry_entry(first.registry_entry.registry_entry_id) == first.registry_entry
    assert (
        service.get_current_item_publication(source.approved_result.release_subject.item_id)
        == first.item_projections[0]
    )
    assert service.list_registry(
        channel=ReleaseChannel.CANARY,
        registry=first.registry_entry.registry,
    ) == (first.registry_entry,)
    assert service.resolve_bundle(
        registry_entry_id=first.registry_entry.registry_entry_id,
        item_id=source.approved_result.release_subject.item_id,
    ).is_dir()
    assert (
        ReleaseProjectionPersistenceService(store).get_current_result(
            source.approved_result.release_subject.item_id
        )
        == source.approved_result
    )


def test_publish_conflicts_on_changed_key_and_second_authority(
    tmp_path: Path,
) -> None:
    service, source, policy, _store = _prepared(tmp_path)
    _publish(service, source, policy)
    changed_policy = _policy(max_manifest_refs=2_000)

    with pytest.raises(IdempotencyConflictError):
        _publish(
            service,
            source,
            changed_policy,
            key="publish-r7-09",
        )
    with pytest.raises(ReleasePublicationConflictError, match="already"):
        _publish(
            service,
            source,
            policy,
            key="second-publisher",
        )


def test_two_publishers_select_one_authority_and_exact_replay(
    tmp_path: Path,
) -> None:
    service, source, policy, _store = _prepared(tmp_path)
    barrier = Barrier(2)

    def publish(key: str):
        barrier.wait()
        try:
            return _publish(service, source, policy, key=key)
        except ReleasePublicationConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        same_key = tuple(
            executor.map(
                publish,
                ("race-same", "race-same"),
            )
        )
    assert same_key[0] == same_key[1]

    second_service, second_source, second_policy, _ = _prepared(tmp_path / "different-key")
    barrier = Barrier(2)

    def publish_different(key: str):
        barrier.wait()
        try:
            return _publish(
                second_service,
                second_source,
                second_policy,
                key=key,
            )
        except ReleasePublicationConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        different_keys = tuple(
            executor.map(
                publish_different,
                ("race-a", "race-b"),
            )
        )
    assert sum(isinstance(value, ReleasePublicationConflictError) for value in different_keys) == 1
    assert sum(not isinstance(value, ReleasePublicationConflictError) for value in different_keys) == 1


@pytest.mark.parametrize(
    "fault_point",
    (
        "after_manifest",
        "after_registry",
        "after_current_projections",
        "after_item_statuses",
        "after_outbox",
        "after_idempotency",
    ),
)
def test_publication_write_fault_rolls_back_every_visible_effect(
    tmp_path: Path,
    fault_point: str,
) -> None:
    def fault(point: str) -> None:
        if point == fault_point:
            raise RuntimeError(f"fault:{point}")

    service, source, policy, store = _prepared(
        tmp_path,
        fault_injector=fault,
    )
    outbox_before = len(store.list_outbox())

    with pytest.raises(RuntimeError, match="fault:"):
        _publish(service, source, policy)

    with sqlite3.connect(store.path) as connection:
        for table in (
            "release_publication_policies",
            "lh_release_manifest_items",
            "nonproduction_release_manifests",
            "lh_export_receipts",
            "item_publication_projection_events",
            "nonproduction_registry_entries",
            "release_publication_results",
            "item_publication_current_projections",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert (
            connection.execute(
                """
            SELECT COUNT(*)
            FROM release_decisions_v2
            WHERE action = 'PUBLISH'
            """
            ).fetchone()[0]
            == 0
        )
    assert store.get_item(source.approved_result.release_subject.item_id).status is ItemStatus.APPROVED
    assert len(store.list_outbox()) == outbox_before


def test_current_projection_rebuild_and_immutable_drift(
    tmp_path: Path,
) -> None:
    service, source, policy, store = _prepared(tmp_path)
    result = _publish(service, source, policy)
    item_id = source.approved_result.release_subject.item_id
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            DELETE FROM item_publication_current_projections
            WHERE item_id = ?
            """,
            (item_id,),
        )

    with pytest.raises(ReleasePublicationIntegrityError):
        service.get_publication(result.result_id)
    assert service.rebuild_job_publication(result.release_manifest.job_id) == result
    assert service.get_current_item_publication(item_id) == (result.item_projections[0])

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE nonproduction_registry_entries
            SET registry = 'registry://r7-09/drift'
            WHERE registry_entry_id = ?
            """,
            (result.registry_entry.registry_entry_id,),
        )
    with pytest.raises(
        ReleasePublicationIntegrityError,
        match="columns",
    ):
        service.get_publication(result.result_id)


def test_bundle_corruption_blocks_registry_reads(
    tmp_path: Path,
) -> None:
    service, source, policy, _store = _prepared(tmp_path)
    result = _publish(service, source, policy)
    bundle = service.resolve_bundle(
        registry_entry_id=result.registry_entry.registry_entry_id,
        item_id=source.approved_result.release_subject.item_id,
    )
    (bundle / "query.yaml").write_text("corrupt", encoding="utf-8")

    with pytest.raises(
        ReleasePublicationIntegrityError,
        match="bundle",
    ):
        service.get_registry_entry(result.registry_entry.registry_entry_id)


def test_unexpected_empty_bundle_directory_blocks_registry_reads(
    tmp_path: Path,
) -> None:
    service, source, policy, _store = _prepared(tmp_path)
    result = _publish(service, source, policy)
    bundle = service.resolve_bundle(
        registry_entry_id=result.registry_entry.registry_entry_id,
        item_id=source.approved_result.release_subject.item_id,
    )
    (bundle / "workspace/unexpected-empty").mkdir()

    with pytest.raises(
        ReleasePublicationIntegrityError,
        match="bundle",
    ):
        service.get_registry_entry(result.registry_entry.registry_entry_id)


def test_r7_released_item_without_successor_is_projection_drift(
    tmp_path: Path,
) -> None:
    service, source, _policy_value, store = _prepared(tmp_path)
    del service
    item = store.get_item(source.approved_result.release_subject.item_id)
    store.transition_item(
        item.item_id,
        ItemStatus.RELEASED,
        expected_version=item.row_version,
        idempotency_key="illegal-release-without-publication",
    )

    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="successor",
    ):
        ReleaseProjectionPersistenceService(store).get_current_result(item.item_id)


def test_r7_successor_read_rejects_publication_event_drift(
    tmp_path: Path,
) -> None:
    service, source, policy, store = _prepared(tmp_path)
    result = _publish(service, source, policy)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE item_publication_projection_events
            SET item_status = 'APPROVED'
            WHERE projection_id = ?
            """,
            (result.item_projections[0].projection_id,),
        )

    with pytest.raises(
        ReleaseProjectionIntegrityError,
        match="event columns",
    ):
        ReleaseProjectionPersistenceService(store).get_current_result(
            source.approved_result.release_subject.item_id
        )


def test_publication_rejects_empty_cross_job_and_inexact_partitions(
    tmp_path: Path,
) -> None:
    service, source, policy, _store = _prepared(tmp_path)
    common = {
        "rejected_item_ids": (),
        "policy": policy,
        "base_contract_manifest_ref": _ref(
            "contract-manifest",
            "v1",
            version="manifest/v1",
        ),
        "overlay_contract_manifest_ref": _ref(
            "contract-manifest",
            "v2",
            version="manifest/v2",
        ),
        "release_profile_decision_ref": _ref(
            "release-profile-decision",
            "lh-v1",
            version="decision/v1",
        ),
        "idempotency_key": "invalid-publication",
        "audit": _audit(),
    }
    with pytest.raises(ReleasePublicationConflictError, match="at least one"):
        service.publish_job(item_sources=(), **common)

    other_subject = source.approved_result.release_subject.model_copy(update={"job_id": "job://r7-09/other"})
    other_result = source.approved_result.model_copy(update={"release_subject": other_subject})
    with pytest.raises(ReleasePublicationConflictError, match="one Job"):
        service.publish_job(
            item_sources=(
                source,
                replace(source, approved_result=other_result),
            ),
            **common,
        )

    with pytest.raises(ReleasePublicationConflictError, match="partition"):
        _publish(
            service,
            source,
            policy,
            rejected_item_ids=("item://r7-09/unknown",),
        )


def test_registry_read_bounds_and_missing_objects(
    tmp_path: Path,
) -> None:
    service, source, policy, _store = _prepared(tmp_path)
    result = _publish(service, source, policy)

    with pytest.raises(ReleasePublicationConflictError, match="non-production"):
        service.list_registry(
            channel=ReleaseChannel.PRODUCTION,
            registry="registry://production",
        )
    with pytest.raises(ReleasePublicationConflictError, match="pagination"):
        service.list_registry(
            channel=ReleaseChannel.CANARY,
            registry=result.registry_entry.registry,
            offset=-1,
        )
    with pytest.raises(RecordNotFoundError, match="registry entry not found"):
        service.get_registry_entry("nonproduction-registry-entry://missing")
    with pytest.raises(RecordNotFoundError, match="registry entry not found"):
        service.resolve_bundle(
            registry_entry_id="nonproduction-registry-entry://missing",
            item_id=source.approved_result.release_subject.item_id,
        )
    with pytest.raises(RecordNotFoundError, match="published Item not found"):
        service.resolve_bundle(
            registry_entry_id=result.registry_entry.registry_entry_id,
            item_id="item://r7-09/missing",
        )
    with pytest.raises(RecordNotFoundError, match="publication not found"):
        service.rebuild_job_publication("job://r7-09/missing")


def test_stale_approved_source_is_rejected_against_current_projection(
    tmp_path: Path,
) -> None:
    service, source, policy, store = _prepared(tmp_path)
    with store._connect() as connection:
        current = service.release_projection._load_current_result_required(
            connection,
            source.approved_result.release_subject.item_id,
        )
        candidate = service.release_projection._load_previous_result(
            connection,
            current,
        )
    assert candidate is not None

    with pytest.raises(
        ReleasePublicationIntegrityError,
        match="current R7-08",
    ):
        _publish(
            service,
            replace(source, approved_result=candidate),
            policy,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("record_json", "{}", "malformed"),
        ("item_status", "APPROVED", "columns"),
    ),
)
def test_current_publication_projection_drift_is_detected(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    service, source, policy, store = _prepared(tmp_path)
    _publish(service, source, policy)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            f"""
            UPDATE item_publication_current_projections
            SET {column} = ?
            WHERE item_id = ?
            """,
            (value, source.approved_result.release_subject.item_id),
        )

    with pytest.raises(ReleasePublicationIntegrityError, match=message):
        service.get_current_item_publication(source.approved_result.release_subject.item_id)


@pytest.mark.parametrize(
    ("table", "column", "value", "message"),
    (
        (
            "release_publication_policies",
            "record_json",
            "{}",
            "policy is malformed",
        ),
        (
            "lh_export_receipts",
            "bundle_sha256",
            "f" * 64,
            "receipt columns",
        ),
    ),
)
def test_publication_child_row_drift_is_detected(
    tmp_path: Path,
    table: str,
    column: str,
    value: str,
    message: str,
) -> None:
    service, source, policy, store = _prepared(tmp_path)
    result = _publish(service, source, policy)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            f"UPDATE {table} SET {column} = ?",
            (value,),
        )

    with pytest.raises(ReleasePublicationIntegrityError, match=message):
        service.get_publication(result.result_id)
