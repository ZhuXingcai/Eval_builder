from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
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
from eval_factory.contracts.external_evidence_v2 import (
    EXTERNAL_EVIDENCE_POLICY_VERSION,
)

_SOURCE_NAME = re.compile(r"^req_[0-9a-f]{16}_raw\.json$")


class ExternalRuntimeV1(StrEnum):
    CLAUDE_CODE = "claude_code"
    CLAUDE_CURATED = "claude_curated"
    CODEX = "codex"
    HERMES = "hermes"


class ExternalPartitionKindV1(StrEnum):
    TRAIN = "TRAIN"
    DEVELOPMENT = "DEVELOPMENT"
    TEST = "TEST"


class ExternalCorpusMemberV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-corpus-member/private-v1"] = (
        "eval-factory/external-corpus-member/private-v1"
    )
    member_id: Identifier
    runtime: ExternalRuntimeV1
    relative_name: RelativePath
    source_trace_id: Identifier
    sid: str = Field(min_length=1, max_length=256)
    sid_sha256: Sha256
    raw_sha256: Sha256
    size_bytes: int = Field(ge=1, le=1_000_000_000)
    event_time: str = Field(min_length=1, max_length=128)
    api_type: str = Field(min_length=1, max_length=128)
    business: str = Field(min_length=1, max_length=128)
    real_model: str = Field(min_length=1, max_length=256)
    request_model: str = Field(min_length=1, max_length=256)
    adapter_name: Literal[
        "curated_trajectory_v1",
        "runtime_snapshot_v1",
    ] = "runtime_snapshot_v1"
    adapter_version: Literal["1.0.0"] = "1.0.0"
    member_sha256: Sha256

    @classmethod
    def create(
        cls,
        *,
        runtime: ExternalRuntimeV1,
        relative_name: str,
        sid: str,
        raw_sha256: str,
        size_bytes: int,
        event_time: str,
        api_type: str,
        business: str,
        real_model: str,
        request_model: str,
        adapter_name: Literal[
            "curated_trajectory_v1",
            "runtime_snapshot_v1",
        ] = "runtime_snapshot_v1",
        adapter_version: Literal["1.0.0"] = "1.0.0",
    ) -> ExternalCorpusMemberV1:
        sid_sha256 = hashlib.sha256(sid.encode()).hexdigest()
        source_trace_id = f"source-trace://external/sha256/{sid_sha256}"
        value = cls.model_construct(
            member_id="external-corpus-member://pending",
            runtime=runtime,
            relative_name=relative_name,
            source_trace_id=source_trace_id,
            sid=sid,
            sid_sha256=sid_sha256,
            raw_sha256=raw_sha256,
            size_bytes=size_bytes,
            event_time=event_time,
            api_type=api_type,
            business=business,
            real_model=real_model,
            request_model=request_model,
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            member_sha256="0" * 64,
        )
        digest = _carried_sha256(
            value,
            exclude={"member_id", "member_sha256"},
        )
        return cls(
            member_id=f"external-corpus-member://sha256/{digest}",
            runtime=runtime,
            relative_name=relative_name,
            source_trace_id=source_trace_id,
            sid=sid,
            sid_sha256=sid_sha256,
            raw_sha256=raw_sha256,
            size_bytes=size_bytes,
            event_time=event_time,
            api_type=api_type,
            business=business,
            real_model=real_model,
            request_model=request_model,
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            member_sha256=digest,
        )

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        if self.relative_name.count("/") != 1:
            raise ValueError("external corpus member must be one runtime level deep")
        runtime, filename = self.relative_name.split("/", 1)
        if runtime != self.runtime.value or _SOURCE_NAME.fullmatch(filename) is None:
            raise ValueError("external corpus member path is invalid")
        expected_sid_sha256 = hashlib.sha256(self.sid.encode()).hexdigest()
        if (
            self.sid_sha256 != expected_sid_sha256
            or self.source_trace_id != f"source-trace://external/sha256/{expected_sid_sha256}"
        ):
            raise ValueError("external corpus member SID identity is stale")
        expected_api_type = {
            ExternalRuntimeV1.CLAUDE_CODE: "Message",
            ExternalRuntimeV1.CLAUDE_CURATED: "Message",
            ExternalRuntimeV1.CODEX: "Response",
            ExternalRuntimeV1.HERMES: "Chat",
        }[self.runtime]
        if self.api_type != expected_api_type:
            raise ValueError("external corpus runtime and API type disagree")
        expected_adapter = {
            ExternalRuntimeV1.CLAUDE_CODE: "runtime_snapshot_v1",
            ExternalRuntimeV1.CLAUDE_CURATED: "curated_trajectory_v1",
            ExternalRuntimeV1.CODEX: "runtime_snapshot_v1",
            ExternalRuntimeV1.HERMES: "runtime_snapshot_v1",
        }[self.runtime]
        if self.adapter_name != expected_adapter:
            raise ValueError("external corpus runtime and adapter disagree")
        digest = _carried_sha256(
            self,
            exclude={"member_id", "member_sha256"},
        )
        if self.member_sha256 != digest or self.member_id != f"external-corpus-member://sha256/{digest}":
            raise ValueError("external corpus member identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-corpus-member",
            object_id=self.member_id,
            object_version="private-v1",
            object_sha256=self.member_sha256,
        )


class ExternalCorpusInventoryV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-corpus-inventory/private-v1"] = (
        "eval-factory/external-corpus-inventory/private-v1"
    )
    inventory_id: Identifier
    source_authorization_ref: ObjectRef
    members: tuple[ExternalCorpusMemberV1, ...] = Field(
        min_length=100,
        max_length=100_000,
    )
    source_count: int = Field(ge=100, le=100_000)
    unique_trace_count: int = Field(ge=100, le=100_000)
    unique_raw_hash_count: int = Field(ge=100, le=100_000)
    total_raw_bytes: int = Field(ge=1, le=1_000_000_000_000)
    adapter_name: Literal[
        "curated_trajectory_v1",
        "runtime_snapshot_v1",
    ] = "runtime_snapshot_v1"
    adapter_version: Literal["1.0.0"] = "1.0.0"
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION
    inventory_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(
        cls,
        *,
        source_authorization_ref: ObjectRef,
        members: tuple[ExternalCorpusMemberV1, ...],
        audit: ContractAudit,
    ) -> ExternalCorpusInventoryV1:
        ordered = tuple(sorted(members, key=lambda member: member.relative_name))
        adapters = {(member.adapter_name, member.adapter_version) for member in ordered}
        if len(adapters) != 1:
            raise ValueError("external corpus inventory must use one adapter")
        adapter_name, adapter_version = next(iter(adapters))
        digest = external_corpus_inventory_commitment(ordered)
        return cls(
            inventory_id=f"external-corpus-inventory://sha256/{digest}",
            source_authorization_ref=source_authorization_ref,
            members=ordered,
            source_count=len(ordered),
            unique_trace_count=len({member.source_trace_id for member in ordered}),
            unique_raw_hash_count=len({member.raw_sha256 for member in ordered}),
            total_raw_bytes=sum(member.size_bytes for member in ordered),
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            inventory_sha256=digest,
            audit=_audit_with_refs(audit, (source_authorization_ref,)),
        )

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        _require_ref(
            self.source_authorization_ref,
            "external-source-authorization",
            "v2",
            "source_authorization_ref",
        )
        ordered = tuple(sorted(self.members, key=lambda member: member.relative_name))
        if self.members != ordered:
            raise ValueError("external corpus members must be sorted")
        for values, label in (
            ((member.relative_name for member in self.members), "relative name"),
            ((member.source_trace_id for member in self.members), "trace ID"),
            ((member.sid for member in self.members), "SID"),
            ((member.sid_sha256 for member in self.members), "SID commitment"),
            ((member.raw_sha256 for member in self.members), "raw hash"),
            ((member.member_id for member in self.members), "member ID"),
        ):
            observed = tuple(values)
            if len(observed) != len(set(observed)):
                raise ValueError(f"external corpus contains duplicate {label}")
        if (
            self.source_count != len(self.members)
            or self.unique_trace_count != len({member.source_trace_id for member in self.members})
            or self.unique_raw_hash_count != len({member.raw_sha256 for member in self.members})
            or self.total_raw_bytes != sum(member.size_bytes for member in self.members)
        ):
            raise ValueError("external corpus aggregate counts are stale")
        if {(member.adapter_name, member.adapter_version) for member in self.members} != {
            (self.adapter_name, self.adapter_version)
        }:
            raise ValueError("external corpus inventory adapter is stale")
        digest = external_corpus_inventory_commitment(self.members)
        if (
            self.inventory_sha256 != digest
            or self.inventory_id != f"external-corpus-inventory://sha256/{digest}"
        ):
            raise ValueError("external corpus inventory identity is stale")
        if self.audit.input_refs != (self.source_authorization_ref,):
            raise ValueError("external corpus inventory audit refs are stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-corpus-inventory",
            object_id=self.inventory_id,
            object_version="private-v1",
            object_sha256=self.inventory_sha256,
        )

    def to_source_population_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-source-population",
            object_id=(f"external-source-population://sha256/{self.inventory_sha256}"),
            object_version="v2",
            object_sha256=self.inventory_sha256,
        )


class ExternalPartitionMemberV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-partition-member/private-v1"] = (
        "eval-factory/external-partition-member/private-v1"
    )
    source_trace_id: Identifier
    raw_sha256: Sha256


class ExternalPartitionInventoryV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-partition-inventory/private-v1"] = (
        "eval-factory/external-partition-inventory/private-v1"
    )
    partition_id: Identifier
    partition_kind: ExternalPartitionKindV1
    members: tuple[ExternalPartitionMemberV1, ...]
    source_authority_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION
    partition_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(
        cls,
        *,
        partition_kind: ExternalPartitionKindV1,
        members: tuple[ExternalPartitionMemberV1, ...],
        source_authority_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> ExternalPartitionInventoryV1:
        ordered_members = tuple(
            sorted(
                members,
                key=lambda member: (
                    member.source_trace_id,
                    member.raw_sha256,
                ),
            )
        )
        refs = _sorted_refs(source_authority_refs)
        safe_audit = _audit_with_refs(audit, refs)
        value = cls.model_construct(
            partition_id="external-partition-inventory://pending",
            partition_kind=partition_kind,
            members=ordered_members,
            source_authority_refs=refs,
            partition_sha256="0" * 64,
            audit=safe_audit,
        )
        digest = _carried_sha256(
            value,
            exclude={"partition_id", "partition_sha256", "audit"},
        )
        return cls(
            partition_id=f"external-partition-inventory://sha256/{digest}",
            partition_kind=partition_kind,
            members=ordered_members,
            source_authority_refs=refs,
            partition_sha256=digest,
            audit=safe_audit,
        )

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        ordered = tuple(
            sorted(
                self.members,
                key=lambda member: (
                    member.source_trace_id,
                    member.raw_sha256,
                ),
            )
        )
        if self.members != ordered or len(self.members) != len(set(self.members)):
            raise ValueError("external partition members must be sorted and unique")
        _require_one_to_one(self.members, "external partition")
        if self.source_authority_refs != _sorted_refs(self.source_authority_refs):
            raise ValueError("external partition source refs must be sorted")
        if self.audit.input_refs != self.source_authority_refs:
            raise ValueError("external partition audit refs are stale")
        digest = _carried_sha256(
            self,
            exclude={"partition_id", "partition_sha256", "audit"},
        )
        if (
            self.partition_sha256 != digest
            or self.partition_id != f"external-partition-inventory://sha256/{digest}"
        ):
            raise ValueError("external partition identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-partition-inventory",
            object_id=self.partition_id,
            object_version="private-v1",
            object_sha256=self.partition_sha256,
        )

    def to_public_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-partition-manifest",
            object_id=(
                f"external-partition-manifest://"
                f"{self.partition_kind.value.casefold()}/sha256/"
                f"{self.partition_sha256}"
            ),
            object_version="v2",
            object_sha256=self.partition_sha256,
        )


class ExternalEvidenceMaterialClosureV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-evidence-material-closure/private-v1"] = (
        "eval-factory/external-evidence-material-closure/private-v1"
    )
    closure_id: Identifier
    inventory_ref: ObjectRef
    partition_refs: tuple[ObjectRef, ...] = Field(
        min_length=3,
        max_length=3,
    )
    reference_set_ref: ObjectRef
    observation_set_ref: ObjectRef
    source_count: int = Field(ge=100, le=100_000)
    total_pair_count: int = Field(ge=1, le=30_000_000)
    material_object_count: Literal[6] = 6
    material_canonical_bytes: int = Field(
        ge=2,
        le=5_000_000_000,
    )
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION
    closure_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(
        cls,
        *,
        inventory_ref: ObjectRef,
        partition_refs: tuple[ObjectRef, ...],
        reference_set_ref: ObjectRef,
        observation_set_ref: ObjectRef,
        source_count: int,
        total_pair_count: int,
        material_canonical_bytes: int,
        audit: ContractAudit,
    ) -> ExternalEvidenceMaterialClosureV1:
        partitions = _sorted_refs(partition_refs)
        refs = _sorted_refs(
            (
                inventory_ref,
                *partitions,
                reference_set_ref,
                observation_set_ref,
            )
        )
        safe_audit = _audit_with_refs(audit, refs)
        value = cls.model_construct(
            closure_id="external-evidence-material-closure://pending",
            inventory_ref=inventory_ref,
            partition_refs=partitions,
            reference_set_ref=reference_set_ref,
            observation_set_ref=observation_set_ref,
            source_count=source_count,
            total_pair_count=total_pair_count,
            material_canonical_bytes=material_canonical_bytes,
            closure_sha256="0" * 64,
            audit=safe_audit,
        )
        digest = _carried_sha256(
            value,
            exclude={"closure_id", "closure_sha256", "audit"},
        )
        return cls(
            closure_id=(f"external-evidence-material-closure://sha256/{digest}"),
            inventory_ref=inventory_ref,
            partition_refs=partitions,
            reference_set_ref=reference_set_ref,
            observation_set_ref=observation_set_ref,
            source_count=source_count,
            total_pair_count=total_pair_count,
            material_canonical_bytes=material_canonical_bytes,
            closure_sha256=digest,
            audit=safe_audit,
        )

    @model_validator(mode="after")
    def validate_closure(self) -> Self:
        _require_ref(
            self.inventory_ref,
            "external-corpus-inventory",
            "private-v1",
            "inventory_ref",
        )
        if self.partition_refs != _sorted_refs(self.partition_refs) or len(set(self.partition_refs)) != 3:
            raise ValueError("external material partitions must be sorted and unique")
        for ref in self.partition_refs:
            _require_ref(
                ref,
                "external-partition-inventory",
                "private-v1",
                "partition_refs",
            )
        _require_ref(
            self.reference_set_ref,
            "external-reference-set",
            "private-v1",
            "reference_set_ref",
        )
        _require_ref(
            self.observation_set_ref,
            "blind-label-observation-set",
            "private-v1",
            "observation_set_ref",
        )
        refs = _sorted_refs(
            (
                self.inventory_ref,
                *self.partition_refs,
                self.reference_set_ref,
                self.observation_set_ref,
            )
        )
        if self.audit.input_refs != refs:
            raise ValueError("external material closure audit refs are stale")
        digest = _carried_sha256(
            self,
            exclude={"closure_id", "closure_sha256", "audit"},
        )
        if (
            self.closure_sha256 != digest
            or self.closure_id != f"external-evidence-material-closure://sha256/{digest}"
        ):
            raise ValueError("external material closure identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-evidence-material-closure",
            object_id=self.closure_id,
            object_version="private-v1",
            object_sha256=self.closure_sha256,
        )


def external_corpus_inventory_commitment(
    members: tuple[ExternalCorpusMemberV1, ...],
) -> str:
    projection = [
        {
            "runtime": member.runtime.value,
            "relative_name": member.relative_name,
            "sid_sha256": member.sid_sha256,
            "raw_sha256": member.raw_sha256,
            "size_bytes": member.size_bytes,
        }
        for member in members
    ]
    payload = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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


def _require_one_to_one(
    members: tuple[ExternalPartitionMemberV1, ...],
    label: str,
) -> None:
    trace_to_hash: dict[str, str] = {}
    hash_to_trace: dict[str, str] = {}
    for member in members:
        prior_hash = trace_to_hash.setdefault(
            member.source_trace_id,
            member.raw_sha256,
        )
        prior_trace = hash_to_trace.setdefault(
            member.raw_sha256,
            member.source_trace_id,
        )
        if prior_hash != member.raw_sha256 or prior_trace != member.source_trace_id:
            raise ValueError(f"{label} trace/raw-hash mapping is not one-to-one")


def _audit_with_refs(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _sorted_refs(
    refs: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(refs), key=_ref_key))


def _ref_key(
    ref: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _require_ref(
    ref: ObjectRef,
    object_type: str,
    object_version: str,
    label: str,
) -> None:
    if ref.object_type != object_type or ref.object_version != object_version:
        raise ValueError(f"{label} must reference {object_type}/{object_version}")


__all__ = [
    "ExternalCorpusInventoryV1",
    "ExternalCorpusMemberV1",
    "ExternalEvidenceMaterialClosureV1",
    "ExternalPartitionInventoryV1",
    "ExternalPartitionKindV1",
    "ExternalPartitionMemberV1",
    "ExternalRuntimeV1",
    "external_corpus_inventory_commitment",
]
