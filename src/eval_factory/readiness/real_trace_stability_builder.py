from __future__ import annotations

import csv
import hashlib
import io
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    StageNameV2,
    dataset_job_spec_v2_ref,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    REAL_TRACE_STABILITY_SOURCE_MANIFEST_SHA256,
    RealTraceStabilityCategoryCountV2,
    RealTraceStabilityCorpusSummaryV2,
    RealTraceStabilityPolicyV2,
    validate_real_trace_stability_policy_v2_identity,
)
from eval_factory.readiness.real_trace_stability_models import (
    PreparedRealTraceStabilityCase,
    RealTraceStabilityInventoryV1,
    RealTraceStabilityMemberV1,
)

_MANIFEST_FIELDS = (
    "instance_id",
    "sid",
    "p_date",
    "business",
    "category",
    "pool_id",
    "pool_category",
)
_CURRENT_SOURCE_COUNT = 91


class RealTraceStabilityBuilderError(RuntimeError):
    pass


class RealTraceStabilityManifestError(RealTraceStabilityBuilderError):
    pass


class RealTraceStabilitySourceError(RealTraceStabilityBuilderError):
    pass


class RealTraceStabilityPolicyError(RealTraceStabilityBuilderError):
    pass


@dataclass(frozen=True, slots=True)
class RealTraceStabilityInventoryCompilation:
    inventory: RealTraceStabilityInventoryV1
    corpus_summary: RealTraceStabilityCorpusSummaryV2
    raw_paths: tuple[tuple[str, Path], ...]

    def path_for(self, instance_id: str) -> Path:
        matches = tuple(
            path for observed_instance_id, path in self.raw_paths if observed_instance_id == instance_id
        )
        if len(matches) != 1:
            raise RealTraceStabilitySourceError("stability inventory path binding is missing or ambiguous")
        return matches[0]


@dataclass(frozen=True, slots=True)
class _ManifestRow:
    instance_id: str
    sid: str
    business: str
    category: str
    pool_id: str
    pool_category: str


class RealTraceStabilityBuilder:
    def compile_inventory(
        self,
        *,
        source_manifest: Path,
        raw_root: Path,
        policy: RealTraceStabilityPolicyV2,
        audit: ContractAudit,
    ) -> RealTraceStabilityInventoryCompilation:
        try:
            validate_real_trace_stability_policy_v2_identity(policy)
        except ValueError as exc:
            raise RealTraceStabilityPolicyError("real-trace stability policy is stale") from exc
        manifest = source_manifest.expanduser()
        if manifest.is_symlink() or not manifest.is_file():
            raise RealTraceStabilityManifestError("source manifest must be a regular non-symlink file")
        payload = manifest.read_bytes()
        if (
            len(payload) > policy.max_private_bytes
            or hashlib.sha256(payload).hexdigest() != REAL_TRACE_STABILITY_SOURCE_MANIFEST_SHA256
            or hashlib.sha256(payload).hexdigest() != policy.source_manifest_ref.object_sha256
        ):
            raise RealTraceStabilityManifestError("source manifest differs from approved authority")
        rows = self.parse_manifest(payload, max_sources=policy.max_sources)
        if len(rows) != _CURRENT_SOURCE_COUNT:
            raise RealTraceStabilityManifestError("source manifest does not contain the current 91 traces")

        root = raw_root.expanduser()
        if root.is_symlink() or not root.is_dir():
            raise RealTraceStabilitySourceError("raw root must be a regular non-symlink directory")
        resolved_root = root.resolve()
        expected_names = {f"{row.instance_id}_{row.sid}.jsonl" for row in rows}
        observed = tuple(resolved_root.glob("*.jsonl"))
        if len(observed) != len(expected_names) or {path.name for path in observed} != expected_names:
            raise RealTraceStabilitySourceError("raw source inventory differs from the source manifest")

        material: list[tuple[_ManifestRow, Path, str, int]] = []
        total_bytes = 0
        raw_hashes: set[str] = set()
        for row in rows:
            candidate = resolved_root / f"{row.instance_id}_{row.sid}.jsonl"
            resolved = candidate.resolve()
            if candidate.is_symlink() or not resolved.is_file() or resolved.parent != resolved_root:
                raise RealTraceStabilitySourceError("raw source escapes the approved root")
            try:
                size = resolved.stat().st_size
            except OSError as exc:
                raise RealTraceStabilitySourceError("raw source metadata is unreadable") from exc
            if size < 1 or size > policy.max_source_bytes:
                raise RealTraceStabilitySourceError("raw source exceeds the approved byte limit")
            raw_sha256 = _file_sha256(resolved, expected_size=size)
            if raw_sha256 in raw_hashes:
                raise RealTraceStabilitySourceError("raw source inventory contains a hash alias")
            raw_hashes.add(raw_sha256)
            total_bytes += size
            if total_bytes > policy.max_total_source_bytes:
                raise RealTraceStabilitySourceError("raw source inventory exceeds the aggregate byte limit")
            material.append((row, resolved, raw_sha256, size))

        ordered = tuple(sorted(material, key=lambda item: (item[0].instance_id, item[2])))
        members = tuple(
            RealTraceStabilityMemberV1.create(
                instance_id=row.instance_id,
                sid=row.sid,
                business=row.business,
                category=row.category,
                pool_id=row.pool_id,
                pool_category=row.pool_category,
                raw_sha256=raw_sha256,
                size_bytes=size,
                assigned_fault_point=policy.fault_points[index % len(policy.fault_points)],
            )
            for index, (row, _path, raw_sha256, size) in enumerate(ordered)
        )
        inventory = RealTraceStabilityInventoryV1.create(
            source_manifest_ref=policy.source_manifest_ref,
            members=members,
            audit=audit,
        )
        category_counts: dict[str, int] = {}
        for member in members:
            category_counts[member.category] = category_counts.get(member.category, 0) + 1
        sizes = tuple(member.size_bytes for member in members)
        corpus = RealTraceStabilityCorpusSummaryV2.create(
            source_manifest_ref=policy.source_manifest_ref,
            policy_ref=policy.to_ref(),
            private_inventory_ref=inventory.to_ref(),
            eligible_unique_trace_count=len(members),
            unique_instance_count=len({item.instance_id for item in members}),
            unique_raw_hash_count=len({item.raw_sha256 for item in members}),
            total_raw_bytes=sum(sizes),
            minimum_raw_bytes=min(sizes),
            maximum_raw_bytes=max(sizes),
            category_counts=tuple(
                RealTraceStabilityCategoryCountV2(
                    category=category,
                    count=count,
                )
                for category, count in sorted(category_counts.items())
            ),
            audit=audit,
        )
        path_by_instance = {row.instance_id: path for row, path, _raw_sha256, _size in ordered}
        return RealTraceStabilityInventoryCompilation(
            inventory=inventory,
            corpus_summary=corpus,
            raw_paths=tuple(sorted(path_by_instance.items())),
        )

    @staticmethod
    def parse_manifest(
        payload: bytes,
        *,
        max_sources: int,
    ) -> tuple[_ManifestRow, ...]:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RealTraceStabilityManifestError("source manifest is not UTF-8") from exc
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != _MANIFEST_FIELDS:
            raise RealTraceStabilityManifestError("source manifest header is invalid")
        rows: list[_ManifestRow] = []
        for raw in reader:
            if len(rows) >= max_sources:
                raise RealTraceStabilityManifestError("source manifest exceeds the row limit")
            if None in raw or set(raw) != set(_MANIFEST_FIELDS):
                raise RealTraceStabilityManifestError("source manifest row fields are invalid")
            values = {key: (raw[key] or "").strip() for key in _MANIFEST_FIELDS}
            if any(not value or len(value) > 256 for value in values.values()):
                raise RealTraceStabilityManifestError("source manifest row value is invalid")
            try:
                date.fromisoformat(values["p_date"])
            except ValueError as exc:
                raise RealTraceStabilityManifestError("source manifest date is invalid") from exc
            if not values["instance_id"].startswith("LH_"):
                raise RealTraceStabilityManifestError("source manifest instance is invalid")
            rows.append(
                _ManifestRow(
                    instance_id=values["instance_id"],
                    sid=values["sid"],
                    business=values["business"],
                    category=values["category"],
                    pool_id=values["pool_id"],
                    pool_category=values["pool_category"],
                )
            )
        if not rows:
            raise RealTraceStabilityManifestError("source manifest is empty")
        for identities, label in (
            ((row.instance_id for row in rows), "instance"),
            ((row.sid for row in rows), "SID"),
            ((row.pool_id for row in rows), "pool"),
        ):
            observed_values = tuple(identities)
            if len(observed_values) != len(set(observed_values)):
                raise RealTraceStabilityManifestError(f"source manifest contains duplicate {label}")
        return tuple(sorted(rows, key=lambda row: (row.instance_id, row.sid)))

    def prepare_cases(
        self,
        compilation: RealTraceStabilityInventoryCompilation,
        *,
        policy: RealTraceStabilityPolicyV2,
        audit: ContractAudit,
    ) -> tuple[PreparedRealTraceStabilityCase, ...]:
        return tuple(
            self.prepare_case(
                member,
                raw_path=compilation.path_for(member.instance_id),
                policy=policy,
                audit=audit,
            )
            for member in compilation.inventory.members
        )

    @staticmethod
    def prepare_case(
        member: RealTraceStabilityMemberV1,
        *,
        raw_path: Path,
        policy: RealTraceStabilityPolicyV2,
        audit: ContractAudit,
    ) -> PreparedRealTraceStabilityCase:
        try:
            validate_real_trace_stability_policy_v2_identity(policy)
        except ValueError as exc:
            raise RealTraceStabilityPolicyError("real-trace stability policy is stale") from exc
        case_digest = member.case_key.rsplit("/", 1)[-1]
        adapter_name, separator, adapter_version = policy.adapter_version.partition("/")
        if not separator:
            raise RealTraceStabilityPolicyError("real-trace stability adapter version is invalid")
        trace = TraceSourceRef(
            source_trace_id=f"source-trace://real-stability/{case_digest}",
            source_uri=raw_path.resolve().as_uri(),
            raw_sha256=member.raw_sha256,
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            processing_class="RESTRICTED_TRACE_RAW",
        )
        policy_ref = policy.to_ref()
        member_ref = member.to_ref()
        spec_audit = audit.model_copy(
            update={
                "input_refs": tuple(
                    sorted(
                        {
                            policy.source_manifest_ref,
                            policy_ref,
                            member_ref,
                        },
                        key=_ref_key,
                    )
                )
            }
        )
        spec = DatasetJobSpecV2(
            job_id=f"job://real-trace-stability/{case_digest}",
            traces=(trace,),
            requested_stages=(StageNameV2.TRACE_INDEX,),
            privacy_profile="trusted-monitored-local",
            model_profiles=(),
            budget=ResourceBudget(
                max_model_requests=0,
                max_model_tokens=0,
                max_processes=1,
                max_renderers=0,
                max_network_requests=0,
                max_storage_bytes=policy.max_source_bytes * 4,
            ),
            concurrency=ConcurrencyLimit(
                model_requests=1,
                processes=1,
                renderers=1,
                network_requests=1,
                artifacts_per_item=1,
                items=1,
            ),
            approval_policy_ref=_static_ref(
                "user-approval-policy",
                "real-trace-stability-none",
            ),
            approval_mode=ApprovalMode.NONE,
            enabled_checkpoints=frozenset(),
            export_target=ExportTarget(
                profile="LH",
                profile_version="v1",
                channel="CANARY",
                registry="registry://real-trace-stability-unused",
            ),
            idempotency_key=f"real-trace-stability-{case_digest}",
            audit=spec_audit,
        )
        return PreparedRealTraceStabilityCase(
            member=member,
            raw_path=raw_path.resolve(),
            trace=trace,
            policy=policy,
            job_spec=spec,
            dataset_job_spec_ref=dataset_job_spec_v2_ref(spec),
        )


def _file_sha256(
    path: Path,
    *,
    expected_size: int,
) -> str:
    digest = hashlib.sha256()
    observed_size = 0
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(1024 * 1024):
                observed_size += len(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        current = path.stat()
    except OSError as exc:
        raise RealTraceStabilitySourceError("raw source bytes are unreadable") from exc
    before_identity = _stat_identity(before)
    if (
        before.st_size != expected_size
        or observed_size != expected_size
        or before_identity != _stat_identity(after)
        or before_identity != _stat_identity(current)
    ):
        raise RealTraceStabilitySourceError("raw source changed during admission")
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


def _static_ref(object_type: str, suffix: str) -> ObjectRef:
    digest = hashlib.sha256(f"{object_type}:{suffix}:v1".encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v1",
        object_sha256=digest,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "RealTraceStabilityBuilder",
    "RealTraceStabilityBuilderError",
    "RealTraceStabilityInventoryCompilation",
    "RealTraceStabilityManifestError",
    "RealTraceStabilityPolicyError",
    "RealTraceStabilitySourceError",
]
