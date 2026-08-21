from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryObjectCodecV2,
    R6CanarySeedObjectV2,
)
from eval_factory.contracts.labeling_v2 import LabelUnresolvedReason
from eval_factory.labeling.semantic import (
    FakeSemanticResidualFixture,
    SemanticResidualOutcome,
)
from eval_factory.orchestration.stage_object_store import (
    StageObjectCodecError,
    StageObjectConflictError,
    StageObjectCorruptionError,
    StageObjectInjectedCrash,
    StageObjectNotFoundError,
    StageObjectStore,
    StageObjectStoreFaultPoint,
    StaticStageObjectStoreFaultInjector,
    canary_stage_object_codec_registry,
)


def _fixture(
    *,
    fixture_id: str = "semantic-fixture://stage-store/example",
    confidence: float = 0.4,
) -> FakeSemanticResidualFixture:
    return FakeSemanticResidualFixture(
        fixture_id=fixture_id,
        outcome=SemanticResidualOutcome.ABSTAIN,
        confidence=confidence,
        selected_evidence_ref_ids=(),
        unresolved_reasons=frozenset({LabelUnresolvedReason.AMBIGUOUS_EVIDENCE}),
        explanation_code="semantic-fixture-ambiguous",
        model_available=True,
    )


def _seed(
    value: FakeSemanticResidualFixture,
) -> R6CanarySeedObjectV2:
    registry = canary_stage_object_codec_registry()
    codec = R6CanaryObjectCodecV2.SEMANTIC_RESIDUAL_FIXTURE
    return R6CanarySeedObjectV2(
        object_ref=registry.reference(codec, value),
        codec=codec,
        payload_json=value.canonical_json().decode(),
    )


def _object_log(root: Path) -> Path:
    return root / "objects.jsonl"


def _blob_path(root: Path, digest: str) -> Path:
    return root / "cas" / "sha256" / digest[:2] / digest


def test_seed_admission_replays_exact_typed_object(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-store"
    value = _fixture()
    seed = _seed(value)
    store = StageObjectStore(root)

    first = store.admit_seed(seed)
    second = store.admit_seed(seed)
    loaded = store.get_as(
        seed.object_ref,
        codec=seed.codec,
        model_type=FakeSemanticResidualFixture,
    )

    assert first.object_ref == seed.object_ref
    assert first.envelope_appended is True
    assert first.content_blob_written is True
    assert second.object_ref == seed.object_ref
    assert second.envelope_appended is False
    assert second.content_blob_written is False
    assert loaded == value
    assert store.list_envelopes() == (first.envelope,)
    assert _object_log(root).read_text(encoding="utf-8").count("\n") == 1


def test_envelope_is_content_free_and_payload_lives_only_in_cas(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-store"
    seed = _seed(_fixture())
    write = StageObjectStore(root).admit_seed(seed)

    envelope_text = _object_log(root).read_text(encoding="utf-8")
    assert "selected_evidence_ref_ids" not in envelope_text
    assert "semantic-fixture-ambiguous" not in envelope_text
    assert (
        _blob_path(
            root,
            write.envelope.content_blob_ref.object_sha256,
        ).read_text(encoding="utf-8")
        == seed.payload_json
    )


def test_wrong_seed_ref_and_same_key_conflict_fail_closed(
    tmp_path: Path,
) -> None:
    store = StageObjectStore(tmp_path / "stage-store")
    first = _fixture()
    store.admit_seed(_seed(first))

    changed = _fixture(confidence=0.6)
    with pytest.raises(StageObjectConflictError, match="immutable"):
        store.admit_seed(_seed(changed))

    wrong = _seed(first).model_copy(
        update={"object_ref": _seed(first).object_ref.model_copy(update={"object_type": "label-spec"})}
    )
    with pytest.raises(StageObjectCodecError, match="reference"):
        store.admit_seed(wrong)


def test_cas_orphan_is_invisible_and_reused_after_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-store"
    seed = _seed(_fixture())
    crashing = StageObjectStore(
        root,
        fault_injector=StaticStageObjectStoreFaultInjector(
            crash_points=frozenset({StageObjectStoreFaultPoint.AFTER_CAS_WRITE})
        ),
    )

    with pytest.raises(StageObjectInjectedCrash, match="after_cas_write"):
        crashing.admit_seed(seed)

    assert not _object_log(root).exists()
    clean = StageObjectStore(root)
    with pytest.raises(StageObjectNotFoundError):
        clean.get_as(
            seed.object_ref,
            codec=seed.codec,
            model_type=FakeSemanticResidualFixture,
        )
    retried = clean.admit_seed(seed)
    assert retried.content_blob_written is False
    assert retried.envelope_appended is True


def test_envelope_append_crash_requires_projection_rebuild(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-store"
    seed = _seed(_fixture())
    crashing = StageObjectStore(
        root,
        fault_injector=StaticStageObjectStoreFaultInjector(
            crash_points=frozenset({StageObjectStoreFaultPoint.AFTER_ENVELOPE_APPEND})
        ),
    )

    with pytest.raises(
        StageObjectInjectedCrash,
        match="after_envelope_append",
    ):
        crashing.admit_seed(seed)

    clean = StageObjectStore(root)
    with pytest.raises(StageObjectCorruptionError, match="projection"):
        clean.get_as(
            seed.object_ref,
            codec=seed.codec,
            model_type=FakeSemanticResidualFixture,
        )
    rebuilt = clean.rebuild_projection()
    assert rebuilt.object_count == 1
    assert (
        clean.get_as(
            seed.object_ref,
            codec=seed.codec,
            model_type=FakeSemanticResidualFixture,
        )
        == _fixture()
    )


def test_projection_drift_and_cas_corruption_are_detected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-store"
    seed = _seed(_fixture())
    store = StageObjectStore(root)
    write = store.admit_seed(seed)

    with sqlite3.connect(root / "projection.sqlite3") as connection:
        connection.execute("DELETE FROM stage_objects")
    with pytest.raises(StageObjectCorruptionError, match="projection"):
        store.get_as(
            seed.object_ref,
            codec=seed.codec,
            model_type=FakeSemanticResidualFixture,
        )
    store.rebuild_projection()

    blob = _blob_path(
        root,
        write.envelope.content_blob_ref.object_sha256,
    )
    blob.write_bytes(b"corrupted")
    with pytest.raises(StageObjectCorruptionError, match="CAS"):
        store.get_as(
            seed.object_ref,
            codec=seed.codec,
            model_type=FakeSemanticResidualFixture,
        )


def test_malformed_or_conflicting_envelope_log_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-store"
    seed = _seed(_fixture())
    store = StageObjectStore(root)
    store.admit_seed(seed)
    log = _object_log(root)

    with log.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")
    with pytest.raises(StageObjectCorruptionError, match="JSONL"):
        store.rebuild_projection()

    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text(lines[0] + "\n", encoding="utf-8")
    envelope = json.loads(lines[0])
    envelope["content_blob_ref"]["object_id"] = f"stage-object-content://sha256/{'b' * 64}"
    envelope["content_blob_ref"]["object_sha256"] = "b" * 64
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                envelope,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    with pytest.raises(StageObjectConflictError, match="conflicting"):
        store.rebuild_projection()


def test_crash_after_projection_keeps_committed_object_visible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-store"
    seed = _seed(_fixture())
    crashing = StageObjectStore(
        root,
        fault_injector=StaticStageObjectStoreFaultInjector(
            crash_points=frozenset({StageObjectStoreFaultPoint.AFTER_PROJECTION_REBUILD})
        ),
    )

    with pytest.raises(
        StageObjectInjectedCrash,
        match="after_projection_rebuild",
    ):
        crashing.admit_seed(seed)

    clean = StageObjectStore(root)
    assert (
        clean.get_as(
            seed.object_ref,
            codec=seed.codec,
            model_type=FakeSemanticResidualFixture,
        )
        == _fixture()
    )
