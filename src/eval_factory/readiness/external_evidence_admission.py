from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.readiness.external_evidence_models import (
    ExternalCorpusInventoryV1,
    ExternalCorpusMemberV1,
    ExternalPartitionInventoryV1,
    ExternalPartitionKindV1,
    ExternalPartitionMemberV1,
    ExternalRuntimeV1,
)
from eval_factory.trace.adapters.runtime_snapshot_v1 import (
    RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS,
    RuntimeSnapshotV1Adapter,
)
from eval_factory.trace.models import TraceProbeResultV2, TraceProbeStatus


class ExternalEvidenceAdmissionError(RuntimeError):
    pass


class ExternalEvidenceSourceError(ExternalEvidenceAdmissionError):
    pass


class ExternalEvidencePartitionError(ExternalEvidenceAdmissionError):
    pass


class ExternalEvidenceLimitError(ExternalEvidenceAdmissionError):
    pass


class _SnapshotAdapter(Protocol):
    name: str
    version: str

    def probe(self, source: Path) -> TraceProbeResultV2: ...


@dataclass(frozen=True, slots=True)
class ExternalCorpusInventoryCompilation:
    inventory: ExternalCorpusInventoryV1
    source_population_ref: ObjectRef
    raw_paths: tuple[tuple[str, Path], ...]

    def path_for(self, source_trace_id: str) -> Path:
        matches = tuple(
            path for observed_trace_id, path in self.raw_paths if observed_trace_id == source_trace_id
        )
        if len(matches) != 1:
            raise ExternalEvidenceSourceError("external source path binding is missing or ambiguous")
        return matches[0]


@dataclass(frozen=True, slots=True)
class ExternalPartitionCompilation:
    train: ExternalPartitionInventoryV1
    development: ExternalPartitionInventoryV1
    test: ExternalPartitionInventoryV1
    eligible_source_count: int
    excluded_source_count: int


class ExternalCorpusAdmissionBuilder:
    def __init__(
        self,
        *,
        adapter: _SnapshotAdapter | None = None,
    ) -> None:
        self.adapter = adapter or RuntimeSnapshotV1Adapter()
        if (
            self.adapter.name != RuntimeSnapshotV1Adapter.name
            or self.adapter.version != RuntimeSnapshotV1Adapter.version
        ):
            raise ExternalEvidenceSourceError("external corpus adapter identity is not approved")

    def compile_inventory(
        self,
        *,
        raw_root: Path,
        source_authorization_ref: ObjectRef,
        audit: ContractAudit,
        max_sources: int,
        max_source_bytes: int,
        max_total_source_bytes: int,
    ) -> ExternalCorpusInventoryCompilation:
        _validate_limits(
            max_sources=max_sources,
            max_source_bytes=max_source_bytes,
            max_total_source_bytes=max_total_source_bytes,
        )
        root = raw_root.expanduser().absolute()
        paths = _source_paths(root, max_sources=max_sources)
        members: list[ExternalCorpusMemberV1] = []
        raw_paths: list[tuple[str, Path]] = []
        total_bytes = 0
        seen_sids: set[str] = set()
        seen_sid_commitments: set[str] = set()
        seen_raw_hashes: set[str] = set()
        seen_trace_ids: set[str] = set()

        for path in paths:
            probe = self.adapter.probe(path)
            if probe.status is not TraceProbeStatus.SUPPORTED:
                raise ExternalEvidenceSourceError("external source adapter rejected a corpus member")
            if (
                probe.raw_sha256 is None
                or probe.size_bytes is None
                or probe.record_count != 1
                or set(probe.outer_fields) != RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS
            ):
                raise ExternalEvidenceSourceError("external source adapter evidence is incomplete")
            if probe.size_bytes > max_source_bytes:
                raise ExternalEvidenceLimitError("external source exceeds the per-member byte limit")
            payload = _read_verified_source(
                path,
                expected_size=probe.size_bytes,
                expected_sha256=probe.raw_sha256,
            )
            total_bytes += len(payload)
            if total_bytes > max_total_source_bytes:
                raise ExternalEvidenceLimitError("external source population exceeds the total byte limit")
            outer = _parse_outer(payload)
            runtime = ExternalRuntimeV1(path.parent.name)
            try:
                member = ExternalCorpusMemberV1.create(
                    runtime=runtime,
                    relative_name=path.relative_to(root).as_posix(),
                    sid=outer["sid"],
                    raw_sha256=probe.raw_sha256,
                    size_bytes=probe.size_bytes,
                    event_time=outer["event_time"],
                    api_type=outer["api_type"],
                    business=outer["business"],
                    real_model=outer["real_model"],
                    request_model=outer["request_model"],
                )
            except (ValidationError, ValueError) as exc:
                raise ExternalEvidenceSourceError("external source member metadata is invalid") from exc
            _require_new(member.sid, seen_sids, "SID")
            _require_new(
                member.sid_sha256,
                seen_sid_commitments,
                "SID commitment",
            )
            _require_new(member.raw_sha256, seen_raw_hashes, "raw hash")
            _require_new(member.source_trace_id, seen_trace_ids, "trace ID")
            members.append(member)
            raw_paths.append((member.source_trace_id, path.resolve()))

        try:
            inventory = ExternalCorpusInventoryV1.create(
                source_authorization_ref=source_authorization_ref,
                members=tuple(members),
                audit=audit,
            )
        except (ValidationError, ValueError) as exc:
            raise ExternalEvidenceSourceError("external corpus inventory is invalid") from exc
        return ExternalCorpusInventoryCompilation(
            inventory=inventory,
            source_population_ref=inventory.to_source_population_ref(),
            raw_paths=tuple(sorted(raw_paths)),
        )

    def compile_partitions(
        self,
        *,
        inventory: ExternalCorpusInventoryV1,
        train_members: tuple[ExternalPartitionMemberV1, ...],
        development_members: tuple[ExternalPartitionMemberV1, ...],
        source_authority_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
        minimum_test_sources: int,
    ) -> ExternalPartitionCompilation:
        if minimum_test_sources < 100 or minimum_test_sources > 100_000:
            raise ExternalEvidenceLimitError("external minimum test-source count is invalid")
        _require_partition_one_to_one(train_members, "TRAIN")
        _require_partition_one_to_one(
            development_members,
            "DEVELOPMENT",
        )
        _require_disjoint(
            train_members,
            development_members,
            "TRAIN/DEVELOPMENT overlap",
        )

        inventory_members = tuple(
            ExternalPartitionMemberV1(
                source_trace_id=member.source_trace_id,
                raw_sha256=member.raw_sha256,
            )
            for member in inventory.members
        )
        _require_global_one_to_one(
            inventory_members,
            train_members,
            development_members,
        )
        excluded = {
            (member.source_trace_id, member.raw_sha256) for member in (*train_members, *development_members)
        }
        test_members = tuple(
            member
            for member in inventory_members
            if (member.source_trace_id, member.raw_sha256) not in excluded
        )
        if len(test_members) < minimum_test_sources:
            raise ExternalEvidencePartitionError("external TEST partition has insufficient unique sources")
        _require_disjoint(train_members, test_members, "TRAIN/TEST overlap")
        _require_disjoint(
            development_members,
            test_members,
            "DEVELOPMENT/TEST overlap",
        )
        try:
            train = ExternalPartitionInventoryV1.create(
                partition_kind=ExternalPartitionKindV1.TRAIN,
                members=train_members,
                source_authority_refs=source_authority_refs,
                audit=audit,
            )
            development = ExternalPartitionInventoryV1.create(
                partition_kind=ExternalPartitionKindV1.DEVELOPMENT,
                members=development_members,
                source_authority_refs=source_authority_refs,
                audit=audit,
            )
            test = ExternalPartitionInventoryV1.create(
                partition_kind=ExternalPartitionKindV1.TEST,
                members=test_members,
                source_authority_refs=source_authority_refs,
                audit=audit,
            )
        except (ValidationError, ValueError) as exc:
            raise ExternalEvidencePartitionError("external partition authority is invalid") from exc
        excluded_count = len(inventory_members) - len(test_members)
        return ExternalPartitionCompilation(
            train=train,
            development=development,
            test=test,
            eligible_source_count=len(test_members),
            excluded_source_count=excluded_count,
        )


def _source_paths(
    root: Path,
    *,
    max_sources: int,
) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ExternalEvidenceSourceError("external source root must be a non-symlink directory")
    expected_runtime_names = {
        ExternalRuntimeV1.CLAUDE_CODE.value,
        ExternalRuntimeV1.CODEX.value,
        ExternalRuntimeV1.HERMES.value,
    }
    try:
        runtime_entries = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    except OSError as exc:
        raise ExternalEvidenceSourceError("external source root is unreadable") from exc
    if {path.name for path in runtime_entries} != expected_runtime_names or any(
        path.is_symlink() or not path.is_dir() for path in runtime_entries
    ):
        raise ExternalEvidenceSourceError("external source runtime inventory is not exact")
    paths: list[Path] = []
    for runtime_directory in runtime_entries:
        try:
            entries = tuple(sorted(runtime_directory.iterdir(), key=lambda path: path.name))
        except OSError as exc:
            raise ExternalEvidenceSourceError("external source runtime directory is unreadable") from exc
        for path in entries:
            if path.is_symlink() or not path.is_file():
                raise ExternalEvidenceSourceError("external source root contains a non-regular member")
            if (
                path.suffix != ".json"
                or not path.name.startswith("req_")
                or not path.name.endswith("_raw.json")
            ):
                raise ExternalEvidenceSourceError("external source root contains an unexpected member")
            paths.append(path)
            if len(paths) > max_sources:
                raise ExternalEvidenceLimitError("external source population exceeds the member limit")
    if len(paths) < 100:
        raise ExternalEvidenceSourceError("external source population contains fewer than 100 members")
    return tuple(sorted(paths))


def _read_verified_source(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> bytes:
    try:
        if path.is_symlink():
            raise ExternalEvidenceSourceError("external source became a symbolic link")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
    except ExternalEvidenceSourceError:
        raise
    except OSError as exc:
        raise ExternalEvidenceSourceError("external source bytes are unreadable") from exc
    identity = _stat_identity(before)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        identity != _stat_identity(after)
        or identity != _stat_identity(current)
        or before.st_size != expected_size
        or len(payload) != expected_size
        or observed_sha256 != expected_sha256
    ):
        raise ExternalEvidenceSourceError("external source changed during admission")
    return payload


def _parse_outer(payload: bytes) -> dict[str, str]:
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ExternalEvidenceSourceError("external source outer JSON is invalid") from exc
    if (
        not isinstance(value, dict)
        or set(value) != RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS
        or any(not isinstance(value.get(field), str) for field in RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS)
    ):
        raise ExternalEvidenceSourceError("external source outer schema is invalid")
    return {field: value[field] for field in RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _stat_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


def _validate_limits(
    *,
    max_sources: int,
    max_source_bytes: int,
    max_total_source_bytes: int,
) -> None:
    if (
        max_sources < 100
        or max_sources > 100_000
        or max_source_bytes < 1
        or max_source_bytes > 1_000_000_000
        or max_total_source_bytes < max_source_bytes
        or max_total_source_bytes > 1_000_000_000_000
    ):
        raise ExternalEvidenceLimitError("external source admission limits are invalid")


def _require_new(
    value: str,
    seen: set[str],
    label: str,
) -> None:
    if value in seen:
        raise ExternalEvidenceSourceError(f"external corpus contains duplicate {label}")
    seen.add(value)


def _require_partition_one_to_one(
    members: tuple[ExternalPartitionMemberV1, ...],
    label: str,
) -> None:
    traces: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for member in members:
        prior_hash = traces.setdefault(
            member.source_trace_id,
            member.raw_sha256,
        )
        prior_trace = hashes.setdefault(
            member.raw_sha256,
            member.source_trace_id,
        )
        if prior_hash != member.raw_sha256 or prior_trace != member.source_trace_id:
            raise ExternalEvidencePartitionError(f"{label} partition contains a trace/raw-hash alias")
    if len(members) != len(set(members)):
        raise ExternalEvidencePartitionError(f"{label} partition contains duplicate members")


def _require_disjoint(
    left: tuple[ExternalPartitionMemberV1, ...],
    right: tuple[ExternalPartitionMemberV1, ...],
    label: str,
) -> None:
    left_traces = {member.source_trace_id for member in left}
    right_traces = {member.source_trace_id for member in right}
    left_hashes = {member.raw_sha256 for member in left}
    right_hashes = {member.raw_sha256 for member in right}
    if left_traces.intersection(right_traces) or left_hashes.intersection(right_hashes):
        raise ExternalEvidencePartitionError(label)


def _require_global_one_to_one(
    *partitions: tuple[ExternalPartitionMemberV1, ...],
) -> None:
    traces: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for member in (item for partition in partitions for item in partition):
        prior_hash = traces.setdefault(
            member.source_trace_id,
            member.raw_sha256,
        )
        prior_trace = hashes.setdefault(
            member.raw_sha256,
            member.source_trace_id,
        )
        if prior_hash != member.raw_sha256 or prior_trace != member.source_trace_id:
            raise ExternalEvidencePartitionError(
                "external partition authority contains a trace/raw-hash alias"
            )


__all__ = [
    "ExternalCorpusAdmissionBuilder",
    "ExternalCorpusInventoryCompilation",
    "ExternalEvidenceAdmissionError",
    "ExternalEvidenceLimitError",
    "ExternalEvidencePartitionError",
    "ExternalEvidenceSourceError",
    "ExternalPartitionCompilation",
]
