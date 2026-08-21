from __future__ import annotations

import json
from datetime import timedelta
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
)

from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2
from eval_factory.contracts.statistics_v2 import LabelTestSetPartitionKindV2
from eval_factory.statistics.material_store import (
    IndependentLabelMaterialInjectedCrash,
    IndependentLabelMaterialIntegrityError,
    IndependentLabelMaterialLimitError,
    IndependentLabelMaterialStore,
    IndependentLabelMaterialStoreFaultPoint,
    IndependentLabelMaterialTypeError,
    StaticIndependentLabelMaterialFaultInjector,
)
from eval_factory.statistics.models import (
    IndependentLabelCandidatePoolV1,
    IndependentLabelTestSetMaterialV1,
)


def _values():
    spec = _label_spec("search", semantic=False)
    candidate = _candidate(
        spec,
        index=1,
        decision=LabelDecisionValueV2.MATCH,
    )
    pool = _candidate_pool((candidate,))
    train_manifest, train = _partition(LabelTestSetPartitionKindV2.TRAIN)
    development_manifest, development = _partition(LabelTestSetPartitionKindV2.DEVELOPMENT)
    material = IndependentLabelTestSetMaterialV1.create(
        dataset_series_id="independent-label-test-set://store-test",
        dataset_version="v1",
        members=(candidate,),
        candidate_pool_ref=pool.to_ref(),
        train_partition_ref=train_manifest.to_ref(),
        development_partition_ref=development_manifest.to_ref(),
        policy_ref=_policy((spec,)).to_ref(),
        access_policy_ref=_access_policy().to_ref(),
        audit=_audit(),
    )
    return pool, train, development, material


def _store(
    root: Path,
    *,
    fault_injector: StaticIndependentLabelMaterialFaultInjector | None = None,
) -> IndependentLabelMaterialStore:
    return IndependentLabelMaterialStore(
        root,
        max_candidate_pool_bytes=2_000_000,
        max_partition_bytes=1_000_000,
        max_test_set_bytes=2_000_000,
        max_members=1_000,
        fault_injector=fault_injector,
    )


def test_store_round_trips_closed_private_codecs_and_exact_replay(
    tmp_path: Path,
) -> None:
    pool, train, development, material = _values()
    store = _store(tmp_path / "statistics-material")

    pool_write = store.put_candidate_pool(pool)
    train_write = store.put_partition(train)
    development_write = store.put_partition(development)
    material_write = store.put_test_set(material)
    replay = store.put_test_set(material)

    assert pool_write.written is True
    assert train_write.written is True
    assert development_write.written is True
    assert material_write.written is True
    assert replay.written is False
    assert store.get_candidate_pool(pool.to_ref()) == pool
    assert store.get_partition(train.to_ref()) == train
    assert store.get_partition(development.to_ref()) == development
    assert store.get_test_set(material.to_ref()) == material
    assert not hasattr(store, "envelope_path")
    with pytest.raises(IndependentLabelMaterialTypeError):
        store.get_test_set(pool.to_ref())


def test_audit_only_replay_reuses_first_authoritative_envelope(
    tmp_path: Path,
) -> None:
    pool, _, _, _ = _values()
    changed = IndependentLabelCandidatePoolV1.create(
        candidates=pool.candidates,
        source_population_ref=pool.source_population_ref,
        audit=pool.audit.model_copy(
            update={
                "created_at": pool.audit.created_at + timedelta(days=1),
                "created_by": "another-r8-01-actor",
            }
        ),
    )
    assert changed.to_ref() == pool.to_ref()
    store = _store(tmp_path / "statistics-material")

    store.put_candidate_pool(pool)
    replay = store.put_candidate_pool(changed)

    assert replay.written is False
    assert store.get_candidate_pool(pool.to_ref()) == pool


def test_limits_missing_and_corruption_fail_closed(tmp_path: Path) -> None:
    pool, _, _, _ = _values()
    with pytest.raises(IndependentLabelMaterialLimitError):
        IndependentLabelMaterialStore(
            tmp_path / "invalid",
            max_candidate_pool_bytes=1,
            max_partition_bytes=1,
            max_test_set_bytes=1,
            max_members=0,
        )
    limited = IndependentLabelMaterialStore(
        tmp_path / "limited",
        max_candidate_pool_bytes=2,
        max_partition_bytes=2,
        max_test_set_bytes=2,
        max_members=1,
    )
    with pytest.raises(IndependentLabelMaterialLimitError):
        limited.put_candidate_pool(pool)
    second = _candidate(
        _label_spec("search", semantic=False),
        index=2,
        decision=LabelDecisionValueV2.MATCH,
    )
    member_limited = IndependentLabelMaterialStore(
        tmp_path / "member-limited",
        max_candidate_pool_bytes=2_000_000,
        max_partition_bytes=1_000_000,
        max_test_set_bytes=2_000_000,
        max_members=1,
    )
    with pytest.raises(IndependentLabelMaterialLimitError, match="member"):
        member_limited.put_candidate_pool(_candidate_pool((pool.candidates[0], second)))

    root = tmp_path / "corrupt"
    store = _store(root)
    write = store.put_candidate_pool(pool)
    envelope_path = (
        root
        / "envelopes"
        / "candidate-pool"
        / "sha256"
        / write.object_ref.object_sha256[:2]
        / f"{write.object_ref.object_sha256}.json"
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    digest = envelope["content_blob_ref"]["object_sha256"]
    content_path = root / "cas" / "sha256" / digest[:2] / digest
    content_path.write_bytes(b"{}")
    with pytest.raises(IndependentLabelMaterialIntegrityError, match="corrupt"):
        store.get_candidate_pool(write.object_ref)

    envelope_path.write_text("{}", encoding="utf-8")
    with pytest.raises(IndependentLabelMaterialIntegrityError, match="invalid"):
        store.get_candidate_pool(write.object_ref)


def test_cas_orphan_is_invisible_and_reused_after_retry(tmp_path: Path) -> None:
    pool, _, _, _ = _values()
    root = tmp_path / "statistics-material"
    crashing = _store(
        root,
        fault_injector=StaticIndependentLabelMaterialFaultInjector(
            crash_points=frozenset({IndependentLabelMaterialStoreFaultPoint.AFTER_CAS_WRITE})
        ),
    )

    with pytest.raises(IndependentLabelMaterialInjectedCrash, match="after_cas_write"):
        crashing.put_candidate_pool(pool)

    clean = _store(root)
    with pytest.raises(IndependentLabelMaterialIntegrityError, match="envelope"):
        clean.get_candidate_pool(pool.to_ref())
    retried = clean.put_candidate_pool(pool)
    assert retried.content_blob_written is False
    assert retried.written is True


def test_envelope_write_crash_leaves_valid_private_material(tmp_path: Path) -> None:
    pool, _, _, _ = _values()
    root = tmp_path / "statistics-material"
    crashing = _store(
        root,
        fault_injector=StaticIndependentLabelMaterialFaultInjector(
            crash_points=frozenset({IndependentLabelMaterialStoreFaultPoint.AFTER_ENVELOPE_WRITE})
        ),
    )

    with pytest.raises(
        IndependentLabelMaterialInjectedCrash,
        match="after_envelope_write",
    ):
        crashing.put_candidate_pool(pool)

    clean = _store(root)
    assert clean.get_candidate_pool(pool.to_ref()) == pool
    assert clean.put_candidate_pool(pool).written is False
