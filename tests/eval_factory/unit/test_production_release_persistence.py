from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Event

import pytest
from production_attestation_fixtures import production_inputs
from test_production_release import (
    NOW,
    ROOT,
    _audit,
    _gate_policy,
)
from test_release_publication_persistence import (
    _prepared,
)
from test_release_publication_persistence import (
    _publish as _publish_nonproduction,
)

from env_mock_agent.facade.release_export_adapter import (
    MappingLHWorkspaceExportMaterialResolver,
    RegistryLHWorkspaceExportFacade,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.contracts.production_attestation_v2 import (
    ProductionReadinessAttestationPolicyV2,
    ProductionReadinessAttestationProjectionV2,
)
from eval_factory.contracts.production_release_v2 import (
    ProductionReleaseOutcomeV2,
    ProductionReleasePolicyV2,
    ProductionReleaseReasonCodeV2,
    ProductionReleaseResultV2,
)
from eval_factory.contracts.release_projection_v2 import (
    release_decision_v2_ref,
)
from eval_factory.dataset.export import ReleasePublicationItemSource
from eval_factory.dataset.persistence import (
    ReleaseProjectionPersistenceService,
)
from eval_factory.dataset.production_bundle_store import (
    ProductionReleaseBundleStore,
)
from eval_factory.dataset.production_publication import (
    ProductionReleaseAcceptanceV1,
    ProductionReleasePersistenceService,
)
from eval_factory.dataset.production_release import (
    ProductionReleaseAttestationGate,
    ProductionReleaseConflictError,
    ProductionReleaseIntegrityError,
)
from eval_factory.dataset.production_release_store import (
    ProductionRegistryStore,
)
from eval_factory.orchestration import (
    IdempotencyConflictError,
    JobStore,
    RecordNotFoundError,
)
from eval_factory.readiness.production_attestation import (
    ProductionAttestationEvaluator,
)
from eval_factory.readiness.production_attestation_builder import (
    ProductionAttestationBuilder,
)
from eval_factory.readiness.production_attestation_store import (
    ProductionAttestationAcceptedRunStore,
    ProductionAttestationMaterialStore,
    ProductionAttestationReportStore,
    ProductionAttestationStoreCodec,
)


def test_production_release_schema_is_additive_and_reopen_safe(
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
        "production_release_acceptances",
        "production_item_publication_current",
    }.issubset(names)


def test_direct_production_publish_is_atomic_replayable_and_successor_aware(
    tmp_path: Path,
) -> None:
    (
        service,
        source,
        policy,
        store,
        bundle_store,
        registry_store,
        profile_payload,
    ) = _production_service(tmp_path)
    outbox_before = len(store.list_outbox())

    first = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )
    inventory_after_first = (
        _inventory(bundle_store.root),
        _inventory(registry_store.root),
    )
    replay = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )

    acceptance = ProductionReleaseAcceptanceV1.create(
        job_id=source.approved_result.release_subject.job_id,
        result=first,
    )
    assert first.outcome is ProductionReleaseOutcomeV2.PUBLISHED
    assert replay == first
    assert service.get_publication(acceptance.acceptance_id) == first
    assert (
        service.get_current_item_publication(source.approved_result.release_subject.item_id)
        == first.item_projections[0]
    )
    assert service.resolve_bundle(
        acceptance_id=acceptance.acceptance_id,
        item_id=source.approved_result.release_subject.item_id,
    ).is_dir()
    assert store.get_item(source.approved_result.release_subject.item_id).status is ItemStatus.RELEASED
    assert len(store.list_outbox()) == outbox_before + 2
    assert (
        _inventory(bundle_store.root),
        _inventory(registry_store.root),
    ) == inventory_after_first
    assert (
        ReleaseProjectionPersistenceService(store).get_current_result(
            source.approved_result.release_subject.item_id
        )
        == source.approved_result
    )
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM production_release_acceptances").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM production_item_publication_current").fetchone()[0] == 1
        )
        assert (
            connection.execute(
                """
            SELECT COUNT(*)
            FROM idempotency_records
            WHERE response_type = 'PRODUCTION_RELEASE'
            """
            ).fetchone()[0]
            == 1
        )


def test_production_publish_conflicts_without_duplicate_authority(
    tmp_path: Path,
) -> None:
    (
        service,
        source,
        policy,
        store,
        bundle_store,
        registry_store,
        profile_payload,
    ) = _production_service(tmp_path)
    first = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )
    outbox_after_first = len(store.list_outbox())
    inventory_after_first = (
        _inventory(bundle_store.root),
        _inventory(registry_store.root),
    )

    with pytest.raises(IdempotencyConflictError):
        _publish_production(
            service,
            source,
            policy,
            profile_payload,
            previous_attestation_acceptance_key=("production-attestation-acceptance://changed"),
        )
    with pytest.raises(ProductionReleaseConflictError, match="already"):
        _publish_production(
            service,
            source,
            policy,
            profile_payload,
            key="publish-production-r8-09-second",
        )

    acceptance = ProductionReleaseAcceptanceV1.create(
        job_id=source.approved_result.release_subject.job_id,
        result=first,
    )
    assert service.get_publication(acceptance.acceptance_id) == first
    assert len(store.list_outbox()) == outbox_after_first
    assert (
        _inventory(bundle_store.root),
        _inventory(registry_store.root),
    ) == inventory_after_first


def test_concurrent_publishers_select_one_exact_authority(
    tmp_path: Path,
) -> None:
    (
        same_service,
        same_source,
        same_policy,
        same_store,
        _same_bundle_store,
        _same_registry_store,
        same_profile_payload,
    ) = _production_service(tmp_path / "same-key")

    with ThreadPoolExecutor(max_workers=2) as executor:
        same_results = tuple(
            executor.map(
                lambda _: _publish_production(
                    same_service,
                    same_source,
                    same_policy,
                    same_profile_payload,
                ),
                range(2),
            )
        )

    assert same_results[0] == same_results[1]
    with sqlite3.connect(same_store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM production_release_acceptances").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM production_item_publication_current").fetchone()[0] == 1
        )

    (
        different_service,
        different_source,
        different_policy,
        different_store,
        _different_bundle_store,
        _different_registry_store,
        different_profile_payload,
    ) = _production_service(tmp_path / "different-key")

    def invoke(key: str) -> ProductionReleaseResultV2 | Exception:
        try:
            return _publish_production(
                different_service,
                different_source,
                different_policy,
                different_profile_payload,
                key=key,
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        different_results = tuple(
            executor.map(
                invoke,
                (
                    "publish-production-r8-09-a",
                    "publish-production-r8-09-b",
                ),
            )
        )

    assert sum(isinstance(value, ProductionReleaseResultV2) for value in different_results) == 1
    assert sum(isinstance(value, ProductionReleaseConflictError) for value in different_results) == 1
    with sqlite3.connect(different_store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM production_release_acceptances").fetchone()[0] == 1


def test_historical_replay_survives_invalidation_but_new_key_blocks(
    tmp_path: Path,
) -> None:
    (
        service,
        source,
        policy,
        store,
        bundle_store,
        registry_store,
        profile_payload,
    ) = _production_service(tmp_path)
    first = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )
    outbox_after_first = len(store.list_outbox())
    inventory_after_first = (
        _inventory(bundle_store.root),
        _inventory(registry_store.root),
    )
    accepted = service.attestation_gate.accepted_store
    attestation_result = accepted.get_accepted_result("production-attestation-acceptance://issued")
    projection = attestation_result.projection
    assert projection is not None
    attestation_policy = accepted.report_store.get(
        attestation_result.policy_ref,
        ProductionAttestationStoreCodec.POLICY,
        ProductionReadinessAttestationPolicyV2,
    )
    invalidation_record, invalidated = ProductionAttestationEvaluator().evaluate_currentness(
        projection=projection,
        prior_version_set=attestation_result.version_set,
        observed_version_set=attestation_result.version_set,
        observed_issuer_registry_ref=projection.issuer_registry_ref,
        observed_at=NOW + timedelta(days=1),
        explicit_revocation_kind="RELEASE",
        explicit_revocation_ref=ObjectRef(
            object_type="production-attestation-revocation",
            object_id=("production-attestation-revocation://r8-09/release"),
            object_version="v1",
            object_sha256="f" * 64,
        ),
        audit=_audit(),
    )
    assert invalidation_record is not None
    accepted.report_store.publish_invalidation(
        series_key=attestation_result.attestation_series_id,
        invalidation_policy=attestation_policy.invalidation_policy,
        prior=projection,
        invalidation_record=invalidation_record,
        successor=invalidated,
    )

    replay = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )
    blocked = _publish_production(
        service,
        source,
        policy,
        profile_payload,
        key="publish-production-r8-09-after-invalidation",
    )

    assert replay == first
    assert blocked.outcome is (ProductionReleaseOutcomeV2.PRODUCTION_RELEASE_BLOCKED)
    assert blocked.reason_codes == (ProductionReleaseReasonCodeV2.ATTESTATION_NOT_CURRENT,)
    assert blocked.satisfies_sc_015 is False
    assert len(store.list_outbox()) == outbox_after_first
    assert (
        _inventory(bundle_store.root),
        _inventory(registry_store.root),
    ) == inventory_after_first


def test_invalidation_writer_waits_for_production_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        service,
        source,
        policy,
        _store,
        _bundle_store,
        _registry_store,
        profile_payload,
    ) = _production_service(tmp_path)
    accepted = service.attestation_gate.accepted_store
    attestation_result = accepted.get_accepted_result("production-attestation-acceptance://issued")
    projection = attestation_result.projection
    assert projection is not None
    attestation_policy = accepted.report_store.get(
        attestation_result.policy_ref,
        ProductionAttestationStoreCodec.POLICY,
        ProductionReadinessAttestationPolicyV2,
    )
    invalidation_record, invalidated = ProductionAttestationEvaluator().evaluate_currentness(
        projection=projection,
        prior_version_set=attestation_result.version_set,
        observed_version_set=attestation_result.version_set,
        observed_issuer_registry_ref=projection.issuer_registry_ref,
        observed_at=NOW + timedelta(days=1),
        explicit_revocation_kind="RELEASE",
        explicit_revocation_ref=ObjectRef(
            object_type="production-attestation-revocation",
            object_id=("production-attestation-revocation://r8-09/concurrent-release"),
            object_version="v1",
            object_sha256="e" * 64,
        ),
        audit=_audit(),
    )
    assert invalidation_record is not None

    transaction_entered = Event()
    release_transaction = Event()
    exclusive_attempted = Event()
    persist_current = service._persist_current
    publish_current = accepted.report_store.publish_current_projection

    def hold_transaction(
        connection: sqlite3.Connection,
        *,
        acceptance: ProductionReleaseAcceptanceV1,
        result: ProductionReleaseResultV2,
    ) -> None:
        persist_current(
            connection,
            acceptance=acceptance,
            result=result,
        )
        transaction_entered.set()
        if not release_transaction.wait(timeout=5):
            raise AssertionError("production transaction test barrier timed out")

    def observe_exclusive_attempt(
        *,
        series_key: str,
        projection: ProductionReadinessAttestationProjectionV2,
        expected_prior_ref: ObjectRef | None,
    ) -> ProductionReadinessAttestationProjectionV2:
        exclusive_attempted.set()
        return publish_current(
            series_key=series_key,
            projection=projection,
            expected_prior_ref=expected_prior_ref,
        )

    monkeypatch.setattr(service, "_persist_current", hold_transaction)
    monkeypatch.setattr(
        accepted.report_store,
        "publish_current_projection",
        observe_exclusive_attempt,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        publication = executor.submit(
            _publish_production,
            service,
            source,
            policy,
            profile_payload,
        )
        assert transaction_entered.wait(timeout=5)
        invalidation = executor.submit(
            accepted.report_store.publish_invalidation,
            series_key=attestation_result.attestation_series_id,
            invalidation_policy=attestation_policy.invalidation_policy,
            prior=projection,
            invalidation_record=invalidation_record,
            successor=invalidated,
        )
        assert exclusive_attempted.wait(timeout=5)
        try:
            assert not invalidation.done()
        finally:
            release_transaction.set()
        published = publication.result(timeout=5)
        invalidated_current = invalidation.result(timeout=5)

    assert published.outcome is ProductionReleaseOutcomeV2.PUBLISHED
    assert invalidated_current == invalidated
    blocked = _publish_production(
        service,
        source,
        policy,
        profile_payload,
        key="publish-production-r8-09-after-concurrent-invalidation",
    )
    assert blocked.outcome is ProductionReleaseOutcomeV2.PRODUCTION_RELEASE_BLOCKED
    assert blocked.reason_codes == (ProductionReleaseReasonCodeV2.ATTESTATION_NOT_CURRENT,)


def test_nonproduction_publication_is_the_production_predecessor(
    tmp_path: Path,
) -> None:
    r7_service, source, r7_policy, store = _prepared(tmp_path / "r7")
    store._clock = lambda: NOW
    nonproduction = _publish_nonproduction(
        r7_service,
        source,
        r7_policy,
    )
    policy, gate = _accepted_gate(tmp_path / "attestation")
    service = ProductionReleasePersistenceService(
        store,
        bundle_store=ProductionReleaseBundleStore(tmp_path / "production/bundles"),
        registry_store=ProductionRegistryStore(
            tmp_path / "production/registry",
            max_registry_bytes=policy.max_registry_bytes,
        ),
        workspace_facade=RegistryLHWorkspaceExportFacade(
            resolver=MappingLHWorkspaceExportMaterialResolver(outputs={})
        ),
        attestation_gate=gate,
    )
    profile_payload = (ROOT / "specs/002-eval-dataset-factory/release-profiles/v1/decision.json").read_bytes()
    item_before = store.get_item(source.approved_result.release_subject.item_id)

    production = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )

    projection = production.item_projections[0]
    decision = production.published_decisions[0]
    assert projection.source_nonproduction_publication_ref == (nonproduction.to_ref())
    assert decision.previous_decision_ref == release_decision_v2_ref(nonproduction.published_decisions[0])
    assert int(decision.item_version) == (int(nonproduction.published_decisions[0].item_version) + 1)
    assert store.get_item(source.approved_result.release_subject.item_id) == item_before
    assert (
        ReleaseProjectionPersistenceService(store).get_current_result(
            source.approved_result.release_subject.item_id
        )
        == source.approved_result
    )


def test_production_publish_fault_rolls_back_and_reuses_orphan_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        service,
        source,
        policy,
        store,
        bundle_store,
        registry_store,
        profile_payload,
    ) = _production_service(tmp_path)
    outbox_before = len(store.list_outbox())
    with sqlite3.connect(store.path) as connection:
        decision_count_before = connection.execute("SELECT COUNT(*) FROM release_decisions_v2").fetchone()[0]
        item_count_before = connection.execute("SELECT COUNT(*) FROM evaluation_items_v2").fetchone()[0]
    persist_current = service._persist_current

    def fail_after_current(
        connection: sqlite3.Connection,
        *,
        acceptance: ProductionReleaseAcceptanceV1,
        result: ProductionReleaseResultV2,
    ) -> None:
        persist_current(
            connection,
            acceptance=acceptance,
            result=result,
        )
        raise RuntimeError("fault:after-production-current")

    monkeypatch.setattr(service, "_persist_current", fail_after_current)
    with pytest.raises(RuntimeError, match="fault:after-production-current"):
        _publish_production(
            service,
            source,
            policy,
            profile_payload,
        )

    orphan_inventory = (
        _inventory(bundle_store.root),
        _inventory(registry_store.root),
    )
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM release_decisions_v2").fetchone()[0]
            == decision_count_before
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM evaluation_items_v2").fetchone()[0] == item_count_before
        )
        assert connection.execute("SELECT COUNT(*) FROM production_release_acceptances").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM production_item_publication_current").fetchone()[0] == 0
        )
        assert (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM idempotency_records
                WHERE response_type = 'PRODUCTION_RELEASE'
                """
            ).fetchone()[0]
            == 0
        )
    assert store.get_item(source.approved_result.release_subject.item_id).status is ItemStatus.APPROVED
    assert len(store.list_outbox()) == outbox_before

    monkeypatch.setattr(service, "_persist_current", persist_current)
    result = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )

    assert result.outcome is ProductionReleaseOutcomeV2.PUBLISHED
    assert (
        _inventory(bundle_store.root),
        _inventory(registry_store.root),
    ) == orphan_inventory


def test_production_acceptance_pointer_drift_fails_closed(
    tmp_path: Path,
) -> None:
    (
        service,
        source,
        policy,
        store,
        _bundle_store,
        _registry_store,
        profile_payload,
    ) = _production_service(tmp_path)
    result = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )
    acceptance = ProductionReleaseAcceptanceV1.create(
        job_id=source.approved_result.release_subject.job_id,
        result=result,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE production_release_acceptances
            SET published_at = ?
            WHERE acceptance_id = ?
            """,
            (
                (NOW + timedelta(seconds=1)).isoformat(),
                acceptance.acceptance_id,
            ),
        )

    with pytest.raises(
        ProductionReleaseIntegrityError,
        match="columns",
    ):
        service.get_publication(acceptance.acceptance_id)


def test_production_current_projection_rebuild_and_item_reconciliation(
    tmp_path: Path,
) -> None:
    (
        service,
        source,
        policy,
        store,
        _bundle_store,
        _registry_store,
        profile_payload,
    ) = _production_service(tmp_path)
    item_id = source.approved_result.release_subject.item_id
    approved_item = store.get_item(item_id)
    result = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )
    acceptance = ProductionReleaseAcceptanceV1.create(
        job_id=source.approved_result.release_subject.job_id,
        result=result,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            DELETE FROM production_item_publication_current
            WHERE item_id = ?
            """,
            (item_id,),
        )
        connection.execute(
            """
            UPDATE items
            SET status = ?, row_version = ?, record_json = ?
            WHERE item_id = ?
            """,
            (
                approved_item.status.value,
                approved_item.row_version,
                store._record_json(approved_item),
                item_id,
            ),
        )

    with pytest.raises(
        ProductionReleaseIntegrityError,
        match="current projection",
    ):
        service.get_publication(acceptance.acceptance_id)
    assert service.rebuild_job_publication(source.approved_result.release_subject.job_id) == result
    assert service.get_current_item_publication(item_id) == result.item_projections[0]
    assert store.get_item(item_id).status is ItemStatus.RELEASED
    with pytest.raises(ProductionReleaseIntegrityError, match="columns"):
        with sqlite3.connect(store.path) as connection:
            connection.execute(
                """
                UPDATE production_release_acceptances
                SET result_sha256 = ?
                WHERE acceptance_id = ?
                """,
                ("0" * 64, acceptance.acceptance_id),
            )
        service.rebuild_job_publication(source.approved_result.release_subject.job_id)
    with pytest.raises(RecordNotFoundError, match="not found"):
        service.rebuild_job_publication("job://r8-09/missing")


def _production_service(
    root: Path,
) -> tuple[
    ProductionReleasePersistenceService,
    ReleasePublicationItemSource,
    ProductionReleasePolicyV2,
    JobStore,
    ProductionReleaseBundleStore,
    ProductionRegistryStore,
    bytes,
]:
    _r7_service, source, _r7_policy, store = _prepared(root / "r7")
    store._clock = lambda: NOW
    policy, gate = _accepted_gate(root / "attestation")
    bundle_store = ProductionReleaseBundleStore(root / "production/bundles")
    registry_store = ProductionRegistryStore(
        root / "production/registry",
        max_registry_bytes=policy.max_registry_bytes,
    )
    service = ProductionReleasePersistenceService(
        store,
        bundle_store=bundle_store,
        registry_store=registry_store,
        workspace_facade=RegistryLHWorkspaceExportFacade(
            resolver=MappingLHWorkspaceExportMaterialResolver(outputs={})
        ),
        attestation_gate=gate,
    )
    profile_payload = (ROOT / "specs/002-eval-dataset-factory/release-profiles/v1/decision.json").read_bytes()
    return (
        service,
        source,
        policy,
        store,
        bundle_store,
        registry_store,
        profile_payload,
    )


def _publish_production(
    service: ProductionReleasePersistenceService,
    source: ReleasePublicationItemSource,
    policy: ProductionReleasePolicyV2,
    profile_payload: bytes,
    *,
    key: str = "publish-production-r8-09",
    previous_attestation_acceptance_key: str | None = None,
) -> ProductionReleaseResultV2:
    return service.publish_job(
        item_sources=(source,),
        policy=policy,
        release_profile_payload=profile_payload,
        registry="registry://production/lh-v1",
        attestation_acceptance_key=("production-attestation-acceptance://issued"),
        previous_attestation_acceptance_key=(previous_attestation_acceptance_key),
        idempotency_key=key,
        audit=_audit(),
    )


def _accepted_gate(
    root: Path,
) -> tuple[ProductionReleasePolicyV2, ProductionReleaseAttestationGate]:
    policy = _gate_policy()
    attestation_policy, request, registry = production_inputs(
        root / "inputs",
        extra_policy_refs=(policy.to_ref(),),
        release_profile_decision_ref=policy.release_profile_decision_ref,
    )
    compilation = ProductionAttestationBuilder().compile_full(
        policy=attestation_policy,
        request=request,
        trusted_registry=registry,
    )
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=request.audit,
    )
    material = ProductionAttestationMaterialStore(
        root / "private",
        max_private_bytes=100_000_000,
        max_members=1_000,
    )
    reports = ProductionAttestationReportStore(
        root / "public",
        max_report_bytes=10_000_000,
    )
    accepted = ProductionAttestationAcceptedRunStore(
        material_store=material,
        report_store=reports,
    )
    accepted.persist(
        acceptance_key="production-attestation-acceptance://issued",
        request_sha256=request.request_sha256,
        compilation=compilation,
        result=result,
        frozen=frozen,
        audit=request.audit,
    )
    return policy, ProductionReleaseAttestationGate(accepted)


def _inventory(root: Path) -> tuple[int, int]:
    files = tuple(path for path in root.rglob("*") if path.is_file())
    return len(files), sum(path.stat().st_size for path in files)
