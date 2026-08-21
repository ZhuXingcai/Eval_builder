from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    RelativePath,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.readiness.external_evidence_admission import (
    ExternalCorpusInventoryCompilation,
)
from eval_factory.readiness.external_evidence_models import (
    ExternalCorpusInventoryV1,
    ExternalCorpusMemberV1,
    ExternalRuntimeV1,
)
from eval_factory.trace.adapters.curated_trajectory_v1 import (
    CURATED_TRAJECTORY_V1_REQUIRED_FIELDS,
    CuratedTrajectoryV1Adapter,
    validate_curated_trajectory_v1,
)
from eval_factory.trace.models import TraceProbeStatus

CURATED_TRAJECTORY_REQUIRED_COLUMNS = (
    "sid",
    "account",
    "event_time",
    "model",
    "n_user_turns",
    "n_steps",
    "n_ps_errors",
    "up_首轮用户请求",
    "up_list_全部用户请求(JSON)",
    "error_categories_命令用错类型",
    "trajectory_完整轨迹(JSON)",
    "error_location_命令用错位置",
    "乱码报错",
    "question_suggestion_出题建议",
    "核心责任归因",
    "核心子标签",
    "核心判定原因",
    "模型问题错误数",
    "非模型问题错误数",
    "边界模糊错误数",
    "模型问题子标签汇总",
    "非模型问题子标签汇总",
    "边界模糊子标签汇总",
    "执行类工具调用数",
    "有效PowerShell调用数",
    "PowerShell调用占比",
    "PowerShell启动分层",
    "出题Query",
    "criterion",
)

CURATED_TRAJECTORY_CONTAINER_POLICY_VERSION: Literal["curated-trajectory-container/r8-10-v1"] = (
    "curated-trajectory-container/r8-10-v1"
)


class CuratedTrajectoryContainerError(RuntimeError):
    pass


class CuratedTrajectoryContainerMemberV1(ContractModelV2):
    schema_version: Literal["eval-factory/curated-trajectory-container-member/private-v1"] = (
        "eval-factory/curated-trajectory-container-member/private-v1"
    )
    member_id: Identifier
    row_index: int = Field(ge=0, le=100_000)
    relative_name: RelativePath
    source_trace_id: Identifier
    sid: str = Field(min_length=1, max_length=256)
    sid_sha256: Sha256
    raw_sha256: Sha256
    size_bytes: int = Field(ge=1, le=1_000_000_000)
    event_time: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    member_sha256: Sha256

    @classmethod
    def create(
        cls,
        *,
        row_index: int,
        sid: str,
        raw_sha256: str,
        size_bytes: int,
        event_time: str,
        model: str,
    ) -> CuratedTrajectoryContainerMemberV1:
        sid_sha256 = hashlib.sha256(sid.encode()).hexdigest()
        relative_name = f"claude_curated/req_{sid_sha256[:16]}_raw.json"
        source_trace_id = f"source-trace://external/sha256/{sid_sha256}"
        value = cls.model_construct(
            member_id="curated-trajectory-container-member://pending",
            row_index=row_index,
            relative_name=relative_name,
            source_trace_id=source_trace_id,
            sid=sid,
            sid_sha256=sid_sha256,
            raw_sha256=raw_sha256,
            size_bytes=size_bytes,
            event_time=event_time,
            model=model,
            member_sha256="0" * 64,
        )
        digest = _carried_sha256(
            value,
            exclude={"member_id", "member_sha256"},
        )
        return cls(
            member_id=(f"curated-trajectory-container-member://sha256/{digest}"),
            row_index=row_index,
            relative_name=relative_name,
            source_trace_id=source_trace_id,
            sid=sid,
            sid_sha256=sid_sha256,
            raw_sha256=raw_sha256,
            size_bytes=size_bytes,
            event_time=event_time,
            model=model,
            member_sha256=digest,
        )

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        sid_sha256 = hashlib.sha256(self.sid.encode()).hexdigest()
        expected_name = f"claude_curated/req_{sid_sha256[:16]}_raw.json"
        if (
            self.sid_sha256 != sid_sha256
            or self.source_trace_id != f"source-trace://external/sha256/{sid_sha256}"
            or self.relative_name != expected_name
        ):
            raise ValueError("curated trajectory member identity is stale")
        digest = _carried_sha256(
            self,
            exclude={"member_id", "member_sha256"},
        )
        if (
            self.member_sha256 != digest
            or self.member_id != f"curated-trajectory-container-member://sha256/{digest}"
        ):
            raise ValueError("curated trajectory member hash is stale")
        return self


class CuratedTrajectoryContainerInventoryV1(ContractModelV2):
    schema_version: Literal["eval-factory/curated-trajectory-container-inventory/private-v1"] = (
        "eval-factory/curated-trajectory-container-inventory/private-v1"
    )
    inventory_id: Identifier
    container_sha256: Sha256
    container_size_bytes: int = Field(ge=1, le=1_000_000_000)
    header_sha256: Sha256
    members: tuple[CuratedTrajectoryContainerMemberV1, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    member_count: int = Field(ge=1, le=100_000)
    unique_sid_count: int = Field(ge=1, le=100_000)
    unique_member_hash_count: int = Field(ge=1, le=100_000)
    total_member_bytes: int = Field(ge=1, le=1_000_000_000_000)
    adapter_name: Literal["curated_trajectory_v1"] = CuratedTrajectoryV1Adapter.name
    adapter_version: Literal["1.0.0"] = CuratedTrajectoryV1Adapter.version
    policy_version: Literal["curated-trajectory-container/r8-10-v1"] = (
        CURATED_TRAJECTORY_CONTAINER_POLICY_VERSION
    )
    inventory_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(
        cls,
        *,
        container_sha256: str,
        container_size_bytes: int,
        members: tuple[CuratedTrajectoryContainerMemberV1, ...],
        audit: ContractAudit,
    ) -> CuratedTrajectoryContainerInventoryV1:
        ordered = tuple(sorted(members, key=lambda member: member.row_index))
        header_sha256 = hashlib.sha256(
            json.dumps(
                CURATED_TRAJECTORY_REQUIRED_COLUMNS,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        value = cls.model_construct(
            inventory_id="curated-trajectory-container-inventory://pending",
            container_sha256=container_sha256,
            container_size_bytes=container_size_bytes,
            header_sha256=header_sha256,
            members=ordered,
            member_count=len(ordered),
            unique_sid_count=len({member.sid for member in ordered}),
            unique_member_hash_count=len({member.raw_sha256 for member in ordered}),
            total_member_bytes=sum(member.size_bytes for member in ordered),
            inventory_sha256="0" * 64,
            audit=audit,
        )
        digest = _carried_sha256(
            value,
            exclude={"inventory_id", "inventory_sha256", "audit"},
        )
        return cls(
            inventory_id=(f"curated-trajectory-container-inventory://sha256/{digest}"),
            container_sha256=container_sha256,
            container_size_bytes=container_size_bytes,
            header_sha256=header_sha256,
            members=ordered,
            member_count=len(ordered),
            unique_sid_count=len({member.sid for member in ordered}),
            unique_member_hash_count=len({member.raw_sha256 for member in ordered}),
            total_member_bytes=sum(member.size_bytes for member in ordered),
            inventory_sha256=digest,
            audit=audit,
        )

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        ordered = tuple(sorted(self.members, key=lambda member: member.row_index))
        if self.members != ordered:
            raise ValueError("curated trajectory members must use row order")
        for values, label in (
            ((member.row_index for member in self.members), "row index"),
            ((member.relative_name for member in self.members), "relative name"),
            ((member.source_trace_id for member in self.members), "trace ID"),
            ((member.sid for member in self.members), "SID"),
            ((member.raw_sha256 for member in self.members), "raw hash"),
        ):
            observed = tuple(values)
            if len(observed) != len(set(observed)):
                raise ValueError(f"curated trajectory inventory contains duplicate {label}")
        expected_header_sha256 = hashlib.sha256(
            json.dumps(
                CURATED_TRAJECTORY_REQUIRED_COLUMNS,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if (
            self.header_sha256 != expected_header_sha256
            or self.member_count != len(self.members)
            or self.unique_sid_count != len({member.sid for member in self.members})
            or self.unique_member_hash_count != len({member.raw_sha256 for member in self.members})
            or self.total_member_bytes != sum(member.size_bytes for member in self.members)
        ):
            raise ValueError("curated trajectory inventory aggregate is stale")
        digest = _carried_sha256(
            self,
            exclude={"inventory_id", "inventory_sha256", "audit"},
        )
        if (
            self.inventory_sha256 != digest
            or self.inventory_id != f"curated-trajectory-container-inventory://sha256/{digest}"
        ):
            raise ValueError("curated trajectory inventory identity is stale")
        return self

    def to_source_authorization_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-source-authorization",
            object_id=(f"external-source-authorization://curated-trajectory/sha256/{self.container_sha256}"),
            object_version="v2",
            object_sha256=self.container_sha256,
        )


@dataclass(frozen=True, slots=True)
class CuratedTrajectoryContainerCompilation:
    inventory: CuratedTrajectoryContainerInventoryV1
    raw_paths: tuple[tuple[str, Path], ...]

    def path_for(self, source_trace_id: str) -> Path:
        matches = tuple(
            path for observed_trace_id, path in self.raw_paths if observed_trace_id == source_trace_id
        )
        if len(matches) != 1:
            raise CuratedTrajectoryContainerError("curated trajectory path binding is missing or ambiguous")
        return matches[0]


@dataclass(frozen=True, slots=True)
class _PendingMember:
    member: CuratedTrajectoryContainerMemberV1
    raw: bytes


class CuratedTrajectoryContainerBuilder:
    def compile(
        self,
        *,
        container_path: Path,
        material_root: Path,
        audit: ContractAudit,
        max_sources: int,
        max_member_bytes: int,
        max_total_member_bytes: int,
        expected_container_sha256: str,
    ) -> CuratedTrajectoryContainerCompilation:
        _validate_limits(
            max_sources=max_sources,
            max_member_bytes=max_member_bytes,
            max_total_member_bytes=max_total_member_bytes,
        )
        container = container_path.expanduser().absolute()
        raw_container = _read_container(
            container,
            expected_sha256=expected_container_sha256,
        )
        pending = _parse_members(
            raw_container,
            max_sources=max_sources,
            max_member_bytes=max_member_bytes,
            max_total_member_bytes=max_total_member_bytes,
        )
        root = _material_root(
            material_root,
            container=container,
        )
        raw_paths: list[tuple[str, Path]] = []
        expected_paths: set[Path] = set()
        for item in pending:
            path = root / item.member.relative_name
            _write_exact(path, item.raw)
            expected_paths.add(path.resolve())
            raw_paths.append((item.member.source_trace_id, path.resolve()))
        _require_exact_material_inventory(root, expected_paths)
        adapter = CuratedTrajectoryV1Adapter()
        for item in pending:
            path = next(
                path for source_trace_id, path in raw_paths if source_trace_id == item.member.source_trace_id
            )
            probe = adapter.probe(path)
            if (
                probe.status is not TraceProbeStatus.SUPPORTED
                or probe.raw_sha256 != item.member.raw_sha256
                or probe.size_bytes != item.member.size_bytes
                or probe.record_count != 1
                or set(probe.outer_fields) != CURATED_TRAJECTORY_V1_REQUIRED_FIELDS
            ):
                raise CuratedTrajectoryContainerError("curated trajectory material verification failed")
        inventory = CuratedTrajectoryContainerInventoryV1.create(
            container_sha256=expected_container_sha256,
            container_size_bytes=len(raw_container),
            members=tuple(item.member for item in pending),
            audit=audit,
        )
        return CuratedTrajectoryContainerCompilation(
            inventory=inventory,
            raw_paths=tuple(sorted(raw_paths)),
        )

    def compile_external_inventory(
        self,
        *,
        compilation: CuratedTrajectoryContainerCompilation,
        audit: ContractAudit,
    ) -> ExternalCorpusInventoryCompilation:
        members = tuple(
            ExternalCorpusMemberV1.create(
                runtime=ExternalRuntimeV1.CLAUDE_CURATED,
                relative_name=member.relative_name,
                sid=member.sid,
                raw_sha256=member.raw_sha256,
                size_bytes=member.size_bytes,
                event_time=member.event_time,
                api_type="Message",
                business="CuratedPowerShell",
                real_model=member.model,
                request_model=member.model,
                adapter_name=CuratedTrajectoryV1Adapter.name,
                adapter_version=CuratedTrajectoryV1Adapter.version,
            )
            for member in compilation.inventory.members
        )
        inventory = ExternalCorpusInventoryV1.create(
            source_authorization_ref=(compilation.inventory.to_source_authorization_ref()),
            members=members,
            audit=audit,
        )
        raw_paths = tuple(
            (
                member.source_trace_id,
                compilation.path_for(member.source_trace_id),
            )
            for member in members
        )
        return ExternalCorpusInventoryCompilation(
            inventory=inventory,
            source_population_ref=inventory.to_source_population_ref(),
            raw_paths=tuple(sorted(raw_paths)),
        )


def _read_container(
    path: Path,
    *,
    expected_sha256: str,
) -> bytes:
    if not path.exists() or path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".csv":
        raise CuratedTrajectoryContainerError("curated trajectory container must be a regular CSV file")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
    except OSError as exc:
        raise CuratedTrajectoryContainerError("curated trajectory container is unreadable") from exc
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(before) != _stat_identity(current)
        or len(raw) != before.st_size
        or observed_sha256 != expected_sha256
    ):
        raise CuratedTrajectoryContainerError("curated trajectory container hash or metadata changed")
    return raw


def _parse_members(
    raw_container: bytes,
    *,
    max_sources: int,
    max_member_bytes: int,
    max_total_member_bytes: int,
) -> tuple[_PendingMember, ...]:
    try:
        text = raw_container.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CuratedTrajectoryContainerError("curated trajectory container is not UTF-8") from exc
    old_limit = csv.field_size_limit()
    csv.field_size_limit(min(sys.maxsize, max_member_bytes * 4))
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if (
            reader.fieldnames is None
            or tuple(reader.fieldnames) != CURATED_TRAJECTORY_REQUIRED_COLUMNS
            or len(reader.fieldnames) != len(set(reader.fieldnames))
        ):
            raise CuratedTrajectoryContainerError("curated trajectory container header is not exact")
        pending: list[_PendingMember] = []
        total_bytes = 0
        for row_index, row in enumerate(reader):
            if (
                None in row
                or set(row) != set(CURATED_TRAJECTORY_REQUIRED_COLUMNS)
                or any(not isinstance(value, str) for value in row.values())
            ):
                raise CuratedTrajectoryContainerError("curated trajectory container row is invalid")
            if row_index >= max_sources:
                raise CuratedTrajectoryContainerError("curated trajectory source count exceeds the limit")
            sid = row["sid"].strip()
            event_time = row["event_time"].strip()
            model = row["model"].strip()
            raw = row["trajectory_完整轨迹(JSON)"].encode()
            if not sid or not event_time or not model:
                raise CuratedTrajectoryContainerError("curated trajectory source metadata is incomplete")
            if not raw or len(raw) > max_member_bytes:
                raise CuratedTrajectoryContainerError("curated trajectory logical member exceeds the limit")
            total_bytes += len(raw)
            if total_bytes > max_total_member_bytes:
                raise CuratedTrajectoryContainerError(
                    "curated trajectory logical population exceeds the limit"
                )
            _validate_logical_member(raw)
            pending.append(
                _PendingMember(
                    member=CuratedTrajectoryContainerMemberV1.create(
                        row_index=row_index,
                        sid=sid,
                        raw_sha256=hashlib.sha256(raw).hexdigest(),
                        size_bytes=len(raw),
                        event_time=event_time,
                        model=model,
                    ),
                    raw=raw,
                )
            )
    except csv.Error as exc:
        raise CuratedTrajectoryContainerError("curated trajectory CSV is malformed") from exc
    finally:
        csv.field_size_limit(old_limit)
    if not pending:
        raise CuratedTrajectoryContainerError("curated trajectory container is empty")
    for values, label in (
        ((item.member.sid for item in pending), "SID"),
        ((item.member.raw_sha256 for item in pending), "logical member hash"),
        ((item.member.relative_name for item in pending), "materialized path"),
    ):
        observed = tuple(values)
        if len(observed) != len(set(observed)):
            raise CuratedTrajectoryContainerError(f"curated trajectory container contains duplicate {label}")
    return tuple(pending)


def _validate_logical_member(raw: bytes) -> None:
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise CuratedTrajectoryContainerError("curated trajectory logical member JSON is invalid") from exc
    fields, diagnostics = validate_curated_trajectory_v1(value)
    if set(fields) != CURATED_TRAJECTORY_V1_REQUIRED_FIELDS or diagnostics:
        raise CuratedTrajectoryContainerError("curated trajectory logical member shape is invalid")


def _material_root(
    path: Path,
    *,
    container: Path,
) -> Path:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
        raise CuratedTrajectoryContainerError("curated trajectory material root is invalid")
    resolved = candidate.resolve()
    container_resolved = container.resolve()
    if (
        resolved == container_resolved.parent
        or container_resolved == resolved
        or resolved in container_resolved.parents
    ):
        raise CuratedTrajectoryContainerError("curated trajectory material root overlaps source authority")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate.resolve()


def _write_exact(
    path: Path,
    raw: bytes,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CuratedTrajectoryContainerError("curated trajectory material contains a symlink")
    if path.exists():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise CuratedTrajectoryContainerError("curated trajectory material is unreadable") from exc
        if observed != raw:
            raise CuratedTrajectoryContainerError("curated trajectory material conflicts with logical member")
        return
    temporary = path.parent / (f".{path.name}.{hashlib.sha256(raw).hexdigest()}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise CuratedTrajectoryContainerError(
                    "curated trajectory material conflicts with logical member"
                ) from None
    except OSError as exc:
        raise CuratedTrajectoryContainerError("curated trajectory material could not be persisted") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _require_exact_material_inventory(
    root: Path,
    expected_paths: set[Path],
) -> None:
    observed: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CuratedTrajectoryContainerError("curated trajectory material contains a symlink")
        if path.is_file():
            observed.add(path.resolve())
        elif not path.is_dir():
            raise CuratedTrajectoryContainerError("curated trajectory material contains a special file")
    if observed != expected_paths:
        raise CuratedTrajectoryContainerError("curated trajectory material inventory is not exact")


def _validate_limits(
    *,
    max_sources: int,
    max_member_bytes: int,
    max_total_member_bytes: int,
) -> None:
    if (
        max_sources < 1
        or max_sources > 100_000
        or max_member_bytes < 1
        or max_member_bytes > 1_000_000_000
        or max_total_member_bytes < max_member_bytes
        or max_total_member_bytes > 1_000_000_000_000
    ):
        raise CuratedTrajectoryContainerError("curated trajectory container limits are invalid")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


def _carried_sha256(
    value: ContractModelV2,
    *,
    exclude: set[str],
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={"schema_version", *exclude},
    )
    canonical = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


__all__ = [
    "CURATED_TRAJECTORY_CONTAINER_POLICY_VERSION",
    "CURATED_TRAJECTORY_REQUIRED_COLUMNS",
    "CuratedTrajectoryContainerBuilder",
    "CuratedTrajectoryContainerCompilation",
    "CuratedTrajectoryContainerError",
    "CuratedTrajectoryContainerInventoryV1",
    "CuratedTrajectoryContainerMemberV1",
]
