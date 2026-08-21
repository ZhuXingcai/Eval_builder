from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.readiness.external_evidence_admission import (
    ExternalCorpusAdmissionBuilder,
    ExternalEvidencePartitionError,
    ExternalEvidenceSourceError,
)
from eval_factory.readiness.external_evidence_models import (
    ExternalPartitionKindV1,
    ExternalPartitionMemberV1,
)
from eval_factory.trace.adapters.runtime_snapshot_v1 import (
    RuntimeSnapshotV1Adapter,
)

NOW = datetime(2026, 8, 9, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _audit(
    *,
    created_at: datetime = NOW,
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="external-admission-test",
        governing_versions=(
            VersionBinding(
                component="external-evidence-admission",
                version="external-evidence/r8-10-v1",
            ),
        ),
    )


def _authorization_ref() -> ObjectRef:
    return ObjectRef(
        object_type="external-source-authorization",
        object_id="external-source-authorization://tests/runtime-snapshot",
        object_version="v2",
        object_sha256=_digest("external-source-authorization:tests"),
    )


def _snapshot(
    *,
    sid: str,
    runtime: str,
    ordinal: int,
) -> dict[str, object]:
    api_type = {
        "claude_code": "Message",
        "codex": "Response",
        "hermes": "Chat",
    }.get(runtime, "Unknown")
    return {
        "sid": sid,
        "event_time": f"2026-07-{17 + ordinal % 7:02d} 00:00:00",
        "api_type": api_type,
        "business": "CodingPlan",
        "real_model": f"{runtime}-real-model",
        "request_model": f"{runtime}-request-model",
        "request": "{}",
        "response": "{}",
    }


def _write_snapshot(
    root: Path,
    *,
    runtime: str,
    ordinal: int,
    sid: str | None = None,
    payload_updates: dict[str, object] | None = None,
) -> Path:
    directory = root / runtime
    directory.mkdir(parents=True, exist_ok=True)
    active_sid = sid or f"sid-{ordinal:04d}"
    payload = _snapshot(
        sid=active_sid,
        runtime=runtime,
        ordinal=ordinal,
    )
    if payload_updates:
        payload.update(payload_updates)
    path = directory / f"req_{ordinal:016x}_raw.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _write_population(
    root: Path,
    *,
    count: int = 100,
) -> tuple[Path, ...]:
    runtimes = ("claude_code", "codex", "hermes")
    return tuple(
        _write_snapshot(
            root,
            runtime=runtimes[index % len(runtimes)],
            ordinal=index,
        )
        for index in range(count)
    )


def _expected_commitment(
    root: Path,
    paths: tuple[Path, ...],
) -> str:
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        payload = path.read_bytes()
        outer = json.loads(payload)
        sid = outer["sid"]
        assert isinstance(sid, str)
        rows.append(
            {
                "runtime": path.parent.name,
                "relative_name": path.relative_to(root).as_posix(),
                "sid_sha256": hashlib.sha256(sid.encode()).hexdigest(),
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    canonical = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def test_compile_inventory_preserves_original_bytes_and_commitment(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    paths = _write_population(raw_root)
    before = {path: path.read_bytes() for path in paths}

    compilation = ExternalCorpusAdmissionBuilder().compile_inventory(
        raw_root=raw_root,
        source_authorization_ref=_authorization_ref(),
        audit=_audit(),
        max_sources=1_000,
        max_source_bytes=1_000_000,
        max_total_source_bytes=100_000_000,
    )

    inventory = compilation.inventory
    assert len(inventory.members) == 100
    assert inventory.source_count == 100
    assert inventory.unique_trace_count == 100
    assert inventory.unique_raw_hash_count == 100
    assert inventory.inventory_sha256 == _expected_commitment(raw_root, paths)
    assert inventory.to_ref().object_sha256 == inventory.inventory_sha256
    assert compilation.source_population_ref.object_sha256 == (inventory.inventory_sha256)
    assert sum(member.size_bytes for member in inventory.members) == (inventory.total_raw_bytes)
    assert tuple(member.relative_name for member in inventory.members) == tuple(
        sorted(member.relative_name for member in inventory.members)
    )
    assert {member.runtime for member in inventory.members} == {
        "claude_code",
        "codex",
        "hermes",
    }
    assert all(path.read_bytes() == before[path] for path in paths)
    assert compilation.path_for(inventory.members[0].source_trace_id).is_file()


def test_inventory_identity_is_stable_across_audit_time(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_population(raw_root)
    builder = ExternalCorpusAdmissionBuilder()

    first = builder.compile_inventory(
        raw_root=raw_root,
        source_authorization_ref=_authorization_ref(),
        audit=_audit(),
        max_sources=100,
        max_source_bytes=1_000_000,
        max_total_source_bytes=100_000_000,
    )
    second = builder.compile_inventory(
        raw_root=raw_root,
        source_authorization_ref=_authorization_ref(),
        audit=_audit(created_at=datetime(2026, 8, 10, tzinfo=UTC)),
        max_sources=100,
        max_source_bytes=1_000_000,
        max_total_source_bytes=100_000_000,
    )

    assert second.inventory.inventory_id == first.inventory.inventory_id
    assert second.inventory.inventory_sha256 == first.inventory.inventory_sha256
    assert second.inventory.canonical_sha256() != first.inventory.canonical_sha256()


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-file",
        "wrong-runtime",
        "nested-directory",
        "wrong-extension",
        "symlink",
    ],
)
def test_inventory_rejects_non_exact_source_roots(
    tmp_path: Path,
    mutation: str,
) -> None:
    raw_root = tmp_path / "raw"
    _write_population(raw_root)
    if mutation == "extra-file":
        (raw_root / "README.md").write_text("extra", encoding="utf-8")
    elif mutation == "wrong-runtime":
        _write_snapshot(raw_root, runtime="other", ordinal=101)
    elif mutation == "nested-directory":
        nested = raw_root / "codex" / "nested"
        nested.mkdir()
        (nested / "value.json").write_text("{}", encoding="utf-8")
    elif mutation == "wrong-extension":
        (raw_root / "codex" / "value.jsonl").write_text(
            "{}",
            encoding="utf-8",
        )
    else:
        source = next((raw_root / "codex").glob("*.json"))
        (raw_root / "codex" / "linked.json").symlink_to(source)

    with pytest.raises(ExternalEvidenceSourceError):
        ExternalCorpusAdmissionBuilder().compile_inventory(
            raw_root=raw_root,
            source_authorization_ref=_authorization_ref(),
            audit=_audit(),
            max_sources=1_000,
            max_source_bytes=1_000_000,
            max_total_source_bytes=100_000_000,
        )


def test_inventory_rejects_duplicate_sid_and_invalid_outer_schema(
    tmp_path: Path,
) -> None:
    duplicate_root = tmp_path / "duplicate"
    _write_population(duplicate_root)
    _write_snapshot(
        duplicate_root,
        runtime="codex",
        ordinal=101,
        sid="sid-0000",
    )
    with pytest.raises(ExternalEvidenceSourceError, match="SID"):
        ExternalCorpusAdmissionBuilder().compile_inventory(
            raw_root=duplicate_root,
            source_authorization_ref=_authorization_ref(),
            audit=_audit(),
            max_sources=1_000,
            max_source_bytes=1_000_000,
            max_total_source_bytes=100_000_000,
        )

    invalid_root = tmp_path / "invalid"
    _write_population(invalid_root)
    target = next((invalid_root / "codex").glob("*.json"))
    outer = json.loads(target.read_bytes())
    outer["unexpected"] = "closed"
    target.write_text(
        json.dumps(outer, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ExternalEvidenceSourceError, match="adapter"):
        ExternalCorpusAdmissionBuilder().compile_inventory(
            raw_root=invalid_root,
            source_authorization_ref=_authorization_ref(),
            audit=_audit(),
            max_sources=1_000,
            max_source_bytes=1_000_000,
            max_total_source_bytes=100_000_000,
        )


class _MutatingAdapter(RuntimeSnapshotV1Adapter):
    def probe(self, source: Path):  # type: ignore[no-untyped-def]
        result = super().probe(source)
        if source.name.endswith("0000000000000000_raw.json"):
            source.write_bytes(source.read_bytes() + b" ")
        return result


def test_inventory_rejects_source_change_after_probe(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_population(raw_root)

    with pytest.raises(ExternalEvidenceSourceError, match="changed"):
        ExternalCorpusAdmissionBuilder(
            adapter=_MutatingAdapter(),
        ).compile_inventory(
            raw_root=raw_root,
            source_authorization_ref=_authorization_ref(),
            audit=_audit(),
            max_sources=1_000,
            max_source_bytes=1_000_000,
            max_total_source_bytes=100_000_000,
        )


def test_compile_partitions_produces_exact_test_remainder(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_population(raw_root, count=120)
    builder = ExternalCorpusAdmissionBuilder()
    compilation = builder.compile_inventory(
        raw_root=raw_root,
        source_authorization_ref=_authorization_ref(),
        audit=_audit(),
        max_sources=1_000,
        max_source_bytes=1_000_000,
        max_total_source_bytes=100_000_000,
    )
    members = compilation.inventory.members
    train = tuple(
        ExternalPartitionMemberV1(
            source_trace_id=member.source_trace_id,
            raw_sha256=member.raw_sha256,
        )
        for member in members[:10]
    )
    development = tuple(
        ExternalPartitionMemberV1(
            source_trace_id=member.source_trace_id,
            raw_sha256=member.raw_sha256,
        )
        for member in members[10:20]
    )

    partitions = builder.compile_partitions(
        inventory=compilation.inventory,
        train_members=train,
        development_members=development,
        source_authority_refs=(
            compilation.inventory.to_ref(),
            compilation.source_population_ref,
        ),
        audit=_audit(),
        minimum_test_sources=100,
    )

    assert partitions.train.partition_kind is ExternalPartitionKindV1.TRAIN
    assert partitions.development.partition_kind is (ExternalPartitionKindV1.DEVELOPMENT)
    assert partitions.test.partition_kind is ExternalPartitionKindV1.TEST
    assert len(partitions.train.members) == 10
    assert len(partitions.development.members) == 10
    assert len(partitions.test.members) == 100
    assert partitions.eligible_source_count == 100
    assert partitions.excluded_source_count == 20
    assert partitions.test.to_public_ref().object_type == ("external-partition-manifest")


def test_partitions_reject_trace_or_hash_overlap(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_population(raw_root, count=101)
    builder = ExternalCorpusAdmissionBuilder()
    compilation = builder.compile_inventory(
        raw_root=raw_root,
        source_authorization_ref=_authorization_ref(),
        audit=_audit(),
        max_sources=1_000,
        max_source_bytes=1_000_000,
        max_total_source_bytes=100_000_000,
    )
    member = compilation.inventory.members[0]
    partition_member = ExternalPartitionMemberV1(
        source_trace_id=member.source_trace_id,
        raw_sha256=member.raw_sha256,
    )

    with pytest.raises(ExternalEvidencePartitionError, match="overlap"):
        builder.compile_partitions(
            inventory=compilation.inventory,
            train_members=(partition_member,),
            development_members=(partition_member,),
            source_authority_refs=(compilation.inventory.to_ref(),),
            audit=_audit(),
            minimum_test_sources=100,
        )

    alias = ExternalPartitionMemberV1(
        source_trace_id="source-trace://external/alias",
        raw_sha256=member.raw_sha256,
    )
    with pytest.raises(ExternalEvidencePartitionError, match="alias"):
        builder.compile_partitions(
            inventory=compilation.inventory,
            train_members=(alias,),
            development_members=(),
            source_authority_refs=(compilation.inventory.to_ref(),),
            audit=_audit(),
            minimum_test_sources=100,
        )
