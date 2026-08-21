from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_independent_label_test_set_freeze import (
    _access_policy,
    _audit,
    _candidate,
    _candidate_pool,
    _label_spec,
    _partition,
    _policy,
    _ref,
)

from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2
from eval_factory.contracts.statistics_v2 import (
    LabelTestSetAccessPurposeV2,
    LabelTestSetAccessReasonV2,
    LabelTestSetFreezeOutcomeV2,
    LabelTestSetPartitionKindV2,
)
from eval_factory.statistics.material_store import IndependentLabelMaterialStore
from eval_factory.statistics.models import (
    IndependentLabelTestSetAccessRequestV1,
    TrustedIndependentLabelTestSetPrincipalV1,
)
from eval_factory.statistics.persistence import (
    IndependentLabelAccessResult,
    IndependentLabelHeadRebuild,
    IndependentLabelTestSetAuthorizationError,
    IndependentLabelTestSetConflictError,
    IndependentLabelTestSetIntegrityError,
    IndependentLabelTestSetPersistenceInjectedCrash,
    IndependentLabelTestSetPersistenceService,
    StaticIndependentLabelTestSetPersistenceFaultInjector,
)


def _material_store(root: Path) -> IndependentLabelMaterialStore:
    return IndependentLabelMaterialStore(
        root,
        max_candidate_pool_bytes=10_000_000,
        max_partition_bytes=2_000_000,
        max_test_set_bytes=10_000_000,
        max_members=10_000,
    )


def _service(
    tmp_path: Path,
    *,
    fault_points: frozenset[str] = frozenset(),
) -> IndependentLabelTestSetPersistenceService:
    return IndependentLabelTestSetPersistenceService(
        tmp_path / "statistics.sqlite3",
        material_store=_material_store(tmp_path / "private-material"),
        fault_injector=StaticIndependentLabelTestSetPersistenceFaultInjector(crash_points=fault_points),
    )


def _source(
    *,
    count: int,
    dataset_version: str = "v1",
    supersedes_dataset_ref=None,
    supersedes_freeze_ref=None,
):
    spec = _label_spec("recovery", semantic=True)
    candidates = tuple(
        _candidate(
            spec,
            index=index,
            decision=(
                LabelDecisionValueV2.MATCH
                if index % 3 == 0
                else (LabelDecisionValueV2.NO_MATCH if index % 3 == 1 else LabelDecisionValueV2.ABSTAIN)
            ),
        )
        for index in range(count)
    )
    train_manifest, train_material = _partition(LabelTestSetPartitionKindV2.TRAIN)
    development_manifest, development_material = _partition(LabelTestSetPartitionKindV2.DEVELOPMENT)
    return {
        "dataset_series_id": "independent-label-test-set://persistence",
        "dataset_version": dataset_version,
        "label_specs": (spec,),
        "policy": _policy((spec,)),
        "access_policy": _access_policy(),
        "candidate_pool": _candidate_pool(candidates),
        "train_partition_manifest": train_manifest,
        "train_partition_material": train_material,
        "development_partition_manifest": development_manifest,
        "development_partition_material": development_material,
        "supersedes_dataset_ref": supersedes_dataset_ref,
        "supersedes_freeze_ref": supersedes_freeze_ref,
        "audit": _audit(),
    }


def _access_context(
    frozen,
    *,
    key: str,
    allowed_purposes=frozenset({LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION}),
    allowed_dataset_refs=None,
    principal_max: int = 1_000,
    request_purpose=LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION,
    request_policy_ref=None,
    request_max: int = 1_000,
):
    manifest = frozen.dataset_manifest
    assert manifest is not None
    principal = TrustedIndependentLabelTestSetPrincipalV1(
        principal_ref=_ref("statistical-principal", "evaluator", version="v1"),
        allowed_purposes=allowed_purposes,
        allowed_dataset_refs=(
            frozenset({manifest.to_ref()}) if allowed_dataset_refs is None else allowed_dataset_refs
        ),
        max_members=principal_max,
        audit_case_ref=None,
    )
    request = IndependentLabelTestSetAccessRequestV1(
        dataset_manifest_ref=manifest.to_ref(),
        access_policy_ref=(frozen.access_policy_ref if request_policy_ref is None else request_policy_ref),
        purpose=request_purpose,
        max_members=request_max,
        idempotency_key=key,
    )
    return principal, request


def test_pending_is_immutable_but_does_not_create_dataset_head(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    pending = service.freeze(
        idempotency_key="freeze-pending",
        **_source(count=91),
    )

    assert pending.freeze_record.outcome is LabelTestSetFreezeOutcomeV2.STATISTICAL_GATE_PENDING
    assert pending.dataset_manifest is None
    assert service.get_result(pending.to_ref()) == pending
    assert service.list_results(limit=10) == (pending,)
    with pytest.raises(IndependentLabelTestSetConflictError, match="frozen dataset"):
        service.get_current("independent-label-test-set://persistence")


def test_sufficient_freeze_replays_exactly_and_survives_reopen(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    source = _source(count=120)

    first = service.freeze(idempotency_key="freeze-sufficient", **source)
    replay = service.freeze(idempotency_key="freeze-sufficient", **source)
    reopened = _service(tmp_path)

    assert first == replay
    assert first.dataset_manifest is not None
    assert reopened.get_result(first.to_ref()) == first
    assert reopened.get_current("independent-label-test-set://persistence") == first
    with pytest.raises(IndependentLabelTestSetConflictError, match="idempotency"):
        service.freeze(
            idempotency_key="freeze-sufficient",
            **_source(count=119),
        )
    stale_spec_source = dict(source)
    stale_spec_source["label_specs"] = (
        source["label_specs"][0].model_copy(update={"requirement": "Changed without rehash."}),
    )
    with pytest.raises(IndependentLabelTestSetConflictError, match="idempotency"):
        service.freeze(
            idempotency_key="freeze-sufficient",
            **stale_spec_source,
        )
    with pytest.raises(IndependentLabelTestSetConflictError, match="version"):
        service.freeze(idempotency_key="another-key", **source)


def test_operations_close_sqlite_connections(tmp_path: Path) -> None:
    service = _service(tmp_path)
    frozen = service.freeze(
        idempotency_key="freeze-connection-lifecycle",
        **_source(count=120),
    )

    service.get_result(frozen.to_ref())
    service.list_results(limit=10)

    assert not service.path.with_name(f"{service.path.name}-wal").exists()
    assert not service.path.with_name(f"{service.path.name}-shm").exists()


def test_pending_can_be_superseded_by_new_frozen_version(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    pending = service.freeze(
        idempotency_key="pending-v1",
        **_source(count=91, dataset_version="v1"),
    )

    frozen = service.freeze(
        idempotency_key="frozen-v2",
        **_source(
            count=120,
            dataset_version="v2",
            supersedes_freeze_ref=pending.freeze_record.to_ref(),
        ),
    )

    assert frozen.dataset_manifest is not None
    assert frozen.freeze_record.supersedes_freeze_ref == (pending.freeze_record.to_ref())
    assert service.get_current("independent-label-test-set://persistence") == frozen


def test_authorized_access_returns_private_material_and_denied_returns_none(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    frozen = service.freeze(
        idempotency_key="freeze-access",
        **_source(count=120),
    )
    manifest = frozen.dataset_manifest
    assert manifest is not None
    principal_ref = _ref("statistical-principal", "evaluator", version="v1")
    principal = TrustedIndependentLabelTestSetPrincipalV1(
        principal_ref=principal_ref,
        allowed_purposes=frozenset({LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION}),
        allowed_dataset_refs=frozenset({manifest.to_ref()}),
        max_members=1_000,
        audit_case_ref=None,
    )
    request = IndependentLabelTestSetAccessRequestV1(
        dataset_manifest_ref=manifest.to_ref(),
        access_policy_ref=frozen.access_policy_ref,
        purpose=LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION,
        max_members=1_000,
        idempotency_key="access-granted",
    )

    granted = service.authorize_and_get_material(
        principal=principal,
        request=request,
        audit=_audit(),
    )

    assert isinstance(granted, IndependentLabelAccessResult)
    assert granted.receipt.material_verified is True
    assert len(granted.material.members) == 100

    denied_principal = principal.model_copy(
        update={
            "principal_ref": _ref(
                "statistical-principal",
                "other",
                version="v1",
            )
        }
    )
    with pytest.raises(
        IndependentLabelTestSetAuthorizationError,
        match="not authorized",
    ) as denied:
        service.authorize_and_get_material(
            principal=denied_principal,
            request=request.model_copy(update={"idempotency_key": "access-denied"}),
            audit=_audit(),
        )
    assert denied.value.receipt.returned_member_count == 0


def test_freeze_fault_rolls_back_database_authority(tmp_path: Path) -> None:
    crashing = _service(
        tmp_path,
        fault_points=frozenset({"after_freeze_record"}),
    )
    source = _source(count=120)

    with pytest.raises(
        IndependentLabelTestSetPersistenceInjectedCrash,
        match="after_freeze_record",
    ):
        crashing.freeze(idempotency_key="freeze-fault", **source)

    clean = _service(tmp_path)
    assert clean.list_results(limit=10) == ()
    recovered = clean.freeze(idempotency_key="freeze-fault", **source)
    assert recovered.dataset_manifest is not None


def test_two_freezers_publish_one_authority(tmp_path: Path) -> None:
    source = _source(count=120)

    def freeze(key: str):
        return _service(tmp_path).freeze(idempotency_key=key, **source)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(freeze, "race-one"),
            executor.submit(freeze, "race-two"),
        ]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except IndependentLabelTestSetConflictError:
            outcomes.append(None)

    assert sum(value is not None for value in outcomes) == 1
    assert (
        _service(tmp_path).get_current("independent-label-test-set://persistence").dataset_manifest
        is not None
    )


def test_current_head_rebuild_and_immutable_drift_detection(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    frozen = service.freeze(
        idempotency_key="freeze-rebuild",
        **_source(count=120),
    )
    with sqlite3.connect(tmp_path / "statistics.sqlite3") as connection:
        connection.execute("DELETE FROM label_test_set_current_heads")

    with pytest.raises(IndependentLabelTestSetIntegrityError, match="head"):
        service.get_current("independent-label-test-set://persistence")
    rebuilt = service.rebuild_current_heads()
    assert isinstance(rebuilt, IndependentLabelHeadRebuild)
    assert rebuilt.head_count == 1
    assert service.get_current("independent-label-test-set://persistence") == frozen

    with sqlite3.connect(tmp_path / "statistics.sqlite3") as connection:
        connection.execute(
            """
            UPDATE label_test_set_freeze_records
            SET outcome = 'STATISTICAL_GATE_PENDING'
            WHERE freeze_id = ?
            """,
            (frozen.freeze_record.freeze_id,),
        )
    with pytest.raises(
        IndependentLabelTestSetIntegrityError,
        match="materialized columns",
    ):
        service.get_result(frozen.to_ref())


def test_access_replay_and_closed_denial_matrix(tmp_path: Path) -> None:
    service = _service(tmp_path)
    frozen = service.freeze(
        idempotency_key="freeze-access-matrix",
        **_source(count=120),
    )
    principal, request = _access_context(frozen, key="access-replay")

    first = service.authorize_and_get_material(
        principal=principal,
        request=request,
        audit=_audit(),
    )
    replay = service.authorize_and_get_material(
        principal=principal,
        request=request,
        audit=_audit(),
    )
    assert first.receipt == replay.receipt

    manifest = frozen.dataset_manifest
    assert manifest is not None
    cases = (
        (
            _access_context(
                frozen,
                key="deny-purpose",
                request_purpose=(LabelTestSetAccessPurposeV2.R8_LABEL_TEST_SET_INTEGRITY_AUDIT),
            ),
            LabelTestSetAccessReasonV2.PURPOSE_NOT_AUTHORIZED,
        ),
        (
            _access_context(
                frozen,
                key="deny-dataset",
                allowed_dataset_refs=frozenset({_ref("independent-label-test-set-manifest", "other")}),
            ),
            LabelTestSetAccessReasonV2.DATASET_NOT_AUTHORIZED,
        ),
        (
            _access_context(
                frozen,
                key="deny-limit",
                principal_max=50,
            ),
            LabelTestSetAccessReasonV2.MEMBER_LIMIT_EXCEEDED,
        ),
        (
            _access_context(
                frozen,
                key="deny-policy",
                request_policy_ref=_ref(
                    "independent-label-test-set-access-policy",
                    "stale",
                ),
            ),
            LabelTestSetAccessReasonV2.ACCESS_POLICY_STALE,
        ),
    )
    for (case_principal, case_request), expected_reason in cases:
        with pytest.raises(IndependentLabelTestSetAuthorizationError) as denied:
            service.authorize_and_get_material(
                principal=case_principal,
                request=case_request,
                audit=_audit(),
            )
        assert denied.value.receipt.reason is expected_reason
        with pytest.raises(IndependentLabelTestSetAuthorizationError) as replayed:
            service.authorize_and_get_material(
                principal=case_principal,
                request=case_request,
                audit=_audit(),
            )
        assert replayed.value.receipt == denied.value.receipt


def test_access_idempotency_and_receipt_drift_fail_closed(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    frozen = service.freeze(
        idempotency_key="freeze-access-drift",
        **_source(count=120),
    )
    principal, request = _access_context(frozen, key="access-drift")
    granted = service.authorize_and_get_material(
        principal=principal,
        request=request,
        audit=_audit(),
    )

    with pytest.raises(IndependentLabelTestSetConflictError, match="idempotency"):
        service.authorize_and_get_material(
            principal=principal,
            request=request.model_copy(update={"max_members": 999}),
            audit=_audit(),
        )

    with sqlite3.connect(tmp_path / "statistics.sqlite3") as connection:
        connection.execute(
            """
            UPDATE label_test_set_access_receipts
            SET reason = 'MEMBER_LIMIT_EXCEEDED'
            WHERE receipt_id = ?
            """,
            (granted.receipt.receipt_id,),
        )
    with pytest.raises(
        IndependentLabelTestSetIntegrityError,
        match="materialized columns",
    ):
        service.authorize_and_get_material(
            principal=principal,
            request=request,
            audit=_audit(),
        )


def test_supersession_guards_fail_closed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(
        IndependentLabelTestSetConflictError,
        match="first freeze",
    ):
        service.freeze(
            idempotency_key="bad-first-supersession",
            **_source(
                count=91,
                supersedes_freeze_ref=_ref(
                    "independent-label-test-set-freeze-record",
                    "missing",
                ),
            ),
        )

    pending = service.freeze(
        idempotency_key="pending-for-supersession",
        **_source(count=91),
    )
    with pytest.raises(
        IndependentLabelTestSetConflictError,
        match="latest attempt",
    ):
        service.freeze(
            idempotency_key="missing-freeze-supersession",
            **_source(count=120, dataset_version="v2"),
        )

    frozen = service.freeze(
        idempotency_key="frozen-after-pending",
        **_source(
            count=120,
            dataset_version="v2",
            supersedes_freeze_ref=pending.freeze_record.to_ref(),
        ),
    )
    with pytest.raises(
        IndependentLabelTestSetConflictError,
        match="current authority",
    ):
        service.freeze(
            idempotency_key="missing-dataset-supersession",
            **_source(
                count=120,
                dataset_version="v3",
                supersedes_freeze_ref=frozen.freeze_record.to_ref(),
            ),
        )


@pytest.mark.parametrize(
    "fault_point",
    (
        "after_material",
        "after_policy",
        "after_access_policy",
        "after_partitions",
        "after_manifest",
        "after_result",
        "after_head",
        "after_idempotency",
    ),
)
def test_every_freeze_fault_point_has_no_database_authority(
    tmp_path: Path,
    fault_point: str,
) -> None:
    root = tmp_path / fault_point
    crashing = _service(root, fault_points=frozenset({fault_point}))

    with pytest.raises(
        IndependentLabelTestSetPersistenceInjectedCrash,
        match=fault_point,
    ):
        crashing.freeze(
            idempotency_key=f"freeze-{fault_point}",
            **_source(count=120),
        )

    clean = _service(root)
    assert clean.list_results(limit=10) == ()


def test_page_ref_head_and_policy_drift_fail_closed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    frozen = service.freeze(
        idempotency_key="freeze-drift-matrix",
        **_source(count=120),
    )
    with pytest.raises(IndependentLabelTestSetConflictError, match="page limit"):
        service.list_results(limit=0)
    with pytest.raises(
        IndependentLabelTestSetIntegrityError,
        match="requested reference",
    ):
        service.get_result(frozen.to_ref().model_copy(update={"object_sha256": "f" * 64}))

    with sqlite3.connect(tmp_path / "statistics.sqlite3") as connection:
        connection.execute(
            """
            UPDATE label_test_set_current_heads
            SET freeze_sequence = freeze_sequence + 1
            WHERE dataset_series_id = ?
            """,
            (frozen.freeze_record.dataset_series_id,),
        )
    with pytest.raises(IndependentLabelTestSetIntegrityError, match="stale"):
        service.get_current(frozen.freeze_record.dataset_series_id)
    service.rebuild_current_heads()

    with sqlite3.connect(tmp_path / "statistics.sqlite3") as connection:
        connection.execute(
            "DELETE FROM label_test_set_policies WHERE policy_id = ?",
            (frozen.policy_ref.object_id,),
        )
    with pytest.raises(IndependentLabelTestSetIntegrityError, match="policy"):
        service.get_result(frozen.to_ref())


def test_empty_current_head_rebuild_is_valid(tmp_path: Path) -> None:
    rebuilt = _service(tmp_path).rebuild_current_heads()
    assert rebuilt.head_count == 0
