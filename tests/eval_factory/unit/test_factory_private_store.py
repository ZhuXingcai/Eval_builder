from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.contracts.agent_system_v2 import EvaluationRequirementSpecV2
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 6, tzinfo=UTC),
        created_by="factory-private-store-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="private-store-v1",
                sha256=HASH,
            ),
        ),
    )


def _ref(object_type: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://example/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _requirement() -> EvaluationRequirementSpecV2:
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://private-store",
        run_id="factory-run://private-store",
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Preserve source-grounded inputs.",),
        constraints=("Do not disclose private payloads.",),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )


def _stored_path(root: Path, reference: ObjectRef) -> Path:
    return root / "sha256" / reference.object_sha256[:2] / reference.object_sha256


def test_private_store_replays_exact_bytes_and_rejects_corrupt_existing_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    store = FactoryPrivateObjectStore(root)
    payload = b"private-content"
    reference = store.put_bytes(object_type="private-content", payload=payload)

    assert store.put_bytes(object_type="private-content", payload=payload) == reference
    assert store.get_bytes(reference) == payload

    _stored_path(root, reference).write_bytes(b"corrupt")
    with pytest.raises(FactoryPrivateObjectError, match="content is corrupt"):
        store.put_bytes(object_type="private-content", payload=payload)
    with pytest.raises(FactoryPrivateObjectError, match="hash is corrupt"):
        store.get_bytes(reference)


def test_private_store_fails_closed_for_missing_and_non_utf8_content(
    tmp_path: Path,
) -> None:
    store = FactoryPrivateObjectStore(tmp_path / "private")
    missing = ObjectRef(
        object_type="private-content",
        object_id="private-content://missing/v2",
        object_version="v2",
        object_sha256="b" * 64,
    )
    with pytest.raises(FactoryPrivateObjectError, match="object is missing"):
        store.get_bytes(missing)

    binary = store.put_bytes(object_type="private-content", payload=b"\xff")
    with pytest.raises(FactoryPrivateObjectError, match="not UTF-8"):
        store.get_text(binary)


def test_private_store_requires_valid_canonical_models(tmp_path: Path) -> None:
    store = FactoryPrivateObjectStore(tmp_path / "private")
    requirement = _requirement()
    stored = store.put_model(
        object_type="evaluation-requirement-spec",
        value=requirement,
    )
    assert store.get_model(stored, EvaluationRequirementSpecV2) == requirement

    invalid = store.put_bytes(
        object_type="evaluation-requirement-spec",
        payload=b"{}",
    )
    with pytest.raises(FactoryPrivateObjectError, match="model is invalid"):
        store.get_model(invalid, EvaluationRequirementSpecV2)

    noncanonical = store.put_bytes(
        object_type="evaluation-requirement-spec",
        payload=requirement.model_dump_json(indent=2).encode(),
    )
    with pytest.raises(FactoryPrivateObjectError, match="model is not canonical"):
        store.get_model(noncanonical, EvaluationRequirementSpecV2)
