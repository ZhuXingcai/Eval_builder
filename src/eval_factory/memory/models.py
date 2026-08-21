from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from eval_factory.contracts.core import Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_sha256_v2

MemoryContent = Annotated[str, StringConstraints(min_length=1, max_length=32_768)]
MemoryQueryText = Annotated[str, StringConstraints(min_length=1, max_length=4_096)]


class MemoryKindV1(StrEnum):
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"


class MemoryVisibilityV1(StrEnum):
    PROJECT_SHARED = "PROJECT_SHARED"
    AGENT_PRIVATE = "AGENT_PRIVATE"


class MemorySensitivityV1(StrEnum):
    INTERNAL = "INTERNAL"
    RESTRICTED = "RESTRICTED"
    FORBIDDEN = "FORBIDDEN"


class MemoryHeadStateV1(StrEnum):
    ACTIVE = "ACTIVE"
    TOMBSTONED = "TOMBSTONED"


class MemoryRetrievalModeV1(StrEnum):
    LEXICAL = "LEXICAL"
    SEMANTIC_REQUIRED = "SEMANTIC_REQUIRED"
    HYBRID_REQUIRED = "HYBRID_REQUIRED"


class MemoryNamespaceV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-memory-namespace/private-v1"] = (
        "eval-factory/agent-memory-namespace/private-v1"
    )
    tenant_id: Identifier
    project_id: Identifier
    subject_id: Identifier
    visibility: MemoryVisibilityV1
    owner_agent_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_visibility(self) -> MemoryNamespaceV1:
        if self.visibility is MemoryVisibilityV1.AGENT_PRIVATE:
            if self.owner_agent_id is None:
                raise ValueError("Agent-private memory requires owner_agent_id")
        elif self.owner_agent_id is not None:
            raise ValueError("project-shared memory cannot set owner_agent_id")
        return self


class MemoryAccessContextV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-memory-access/private-v1"] = (
        "eval-factory/agent-memory-access/private-v1"
    )
    tenant_id: Identifier
    project_id: Identifier
    subject_id: Identifier
    requester_agent_id: Identifier

    def authorizes(self, namespace: MemoryNamespaceV1) -> bool:
        return (
            self.tenant_id == namespace.tenant_id
            and self.project_id == namespace.project_id
            and self.subject_id == namespace.subject_id
            and (
                namespace.visibility is MemoryVisibilityV1.PROJECT_SHARED
                or self.requester_agent_id == namespace.owner_agent_id
            )
        )


class MemoryCandidateV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-memory-candidate/private-v1"] = (
        "eval-factory/agent-memory-candidate/private-v1"
    )
    memory_id: Identifier
    namespace: MemoryNamespaceV1
    kind: MemoryKindV1
    content: MemoryContent
    tags: tuple[Identifier, ...] = Field(default=(), max_length=32)
    source_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=32)
    valid_from: datetime
    valid_until: datetime | None = None
    expires_at: datetime | None = None
    importance_basis_points: int = Field(default=5_000, ge=0, le=10_000)
    sensitivity: MemorySensitivityV1 = MemorySensitivityV1.INTERNAL
    approval_ref: ObjectRef | None = None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("memory tags must be sorted and unique")
        return value

    @field_validator("source_refs")
    @classmethod
    def validate_source_refs(
        cls,
        value: tuple[ObjectRef, ...],
    ) -> tuple[ObjectRef, ...]:
        keys = tuple(_ref_key(item) for item in value)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("memory source refs must be sorted and unique")
        return value

    @field_validator("valid_from", "valid_until", "expires_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("memory timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> MemoryCandidateV1:
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be later than valid_from")
        if self.expires_at is not None and self.expires_at <= self.valid_from:
            raise ValueError("expires_at must be later than valid_from")
        if self.sensitivity is MemorySensitivityV1.RESTRICTED:
            if self.approval_ref is None or self.approval_ref.object_type != "memory-admission-approval":
                raise ValueError("restricted memory requires a memory-admission-approval ref")
        elif self.approval_ref is not None:
            raise ValueError("only restricted memory may bind approval_ref")
        return self


class MemoryRecordV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-memory-record/private-v1"] = (
        "eval-factory/agent-memory-record/private-v1"
    )
    memory_record_id: Identifier
    object_sha256: Sha256
    memory_id: Identifier
    revision: int = Field(ge=1)
    predecessor_ref: ObjectRef | None = None
    namespace: MemoryNamespaceV1
    kind: MemoryKindV1
    tags: tuple[Identifier, ...] = Field(default=(), max_length=32)
    source_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=32)
    content_sha256: Sha256
    valid_from: datetime
    valid_until: datetime | None = None
    expires_at: datetime | None = None
    importance_basis_points: int = Field(ge=0, le=10_000)
    sensitivity: MemorySensitivityV1
    approval_ref: ObjectRef | None = None
    created_at: datetime

    @field_validator("valid_from", "valid_until", "expires_at", "created_at")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("memory record timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> MemoryRecordV1:
        expected = _memory_record_sha256(self)
        if self.object_sha256 != expected:
            raise ValueError("memory record object_sha256 is stale")
        if self.memory_record_id != f"agent-memory-record://sha256/{expected}":
            raise ValueError("memory record ID is stale")
        if self.revision == 1 and self.predecessor_ref is not None:
            raise ValueError("first memory revision cannot have a predecessor")
        if self.revision > 1 and (
            self.predecessor_ref is None or self.predecessor_ref.object_type != "agent-memory-record"
        ):
            raise ValueError("successor memory revision requires an Agent memory predecessor")
        return self

    @classmethod
    def create(
        cls,
        *,
        candidate: MemoryCandidateV1,
        revision: int,
        predecessor_ref: ObjectRef | None,
        created_at: datetime,
    ) -> MemoryRecordV1:
        content_sha256 = hashlib.sha256(candidate.content.encode("utf-8")).hexdigest()
        payload = {
            "schema_version": "eval-factory/agent-memory-record/private-v1",
            "memory_id": candidate.memory_id,
            "revision": revision,
            "predecessor_ref": predecessor_ref,
            "namespace": candidate.namespace,
            "kind": candidate.kind,
            "tags": candidate.tags,
            "source_refs": candidate.source_refs,
            "content_sha256": content_sha256,
            "valid_from": candidate.valid_from,
            "valid_until": candidate.valid_until,
            "expires_at": candidate.expires_at,
            "importance_basis_points": candidate.importance_basis_points,
            "sensitivity": candidate.sensitivity,
            "approval_ref": candidate.approval_ref,
            "created_at": created_at,
        }
        digest = _payload_sha256(payload)
        return cls(
            memory_record_id=f"agent-memory-record://sha256/{digest}",
            object_sha256=digest,
            memory_id=candidate.memory_id,
            revision=revision,
            predecessor_ref=predecessor_ref,
            namespace=candidate.namespace,
            kind=candidate.kind,
            tags=candidate.tags,
            source_refs=candidate.source_refs,
            content_sha256=content_sha256,
            valid_from=candidate.valid_from,
            valid_until=candidate.valid_until,
            expires_at=candidate.expires_at,
            importance_basis_points=candidate.importance_basis_points,
            sensitivity=candidate.sensitivity,
            approval_ref=candidate.approval_ref,
            created_at=created_at,
        )

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="agent-memory-record",
            object_id=self.memory_record_id,
            object_version="private-v1",
            object_sha256=self.object_sha256,
        )


class MemoryTombstoneV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-memory-tombstone/private-v1"] = (
        "eval-factory/agent-memory-tombstone/private-v1"
    )
    tombstone_id: Identifier
    object_sha256: Sha256
    memory_id: Identifier
    final_revision: int = Field(ge=1)
    prior_record_ref: ObjectRef
    reason_code: Identifier
    deleted_at: datetime

    @field_validator("deleted_at")
    @classmethod
    def validate_deleted_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("tombstone timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> MemoryTombstoneV1:
        expected = _tombstone_sha256(self)
        if self.object_sha256 != expected:
            raise ValueError("memory tombstone object_sha256 is stale")
        if self.tombstone_id != f"agent-memory-tombstone://sha256/{expected}":
            raise ValueError("memory tombstone ID is stale")
        if self.prior_record_ref.object_type != "agent-memory-record":
            raise ValueError("memory tombstone requires an Agent memory record")
        return self

    @classmethod
    def create(
        cls,
        *,
        memory_id: str,
        final_revision: int,
        prior_record_ref: ObjectRef,
        reason_code: str,
        deleted_at: datetime,
    ) -> MemoryTombstoneV1:
        payload = {
            "schema_version": "eval-factory/agent-memory-tombstone/private-v1",
            "memory_id": memory_id,
            "final_revision": final_revision,
            "prior_record_ref": prior_record_ref,
            "reason_code": reason_code,
            "deleted_at": deleted_at,
        }
        digest = _payload_sha256(payload)
        return cls(
            tombstone_id=f"agent-memory-tombstone://sha256/{digest}",
            object_sha256=digest,
            memory_id=memory_id,
            final_revision=final_revision,
            prior_record_ref=prior_record_ref,
            reason_code=reason_code,
            deleted_at=deleted_at,
        )

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="agent-memory-tombstone",
            object_id=self.tombstone_id,
            object_version="private-v1",
            object_sha256=self.object_sha256,
        )


class StoredMemoryV1(ContractModelV2):
    schema_version: Literal["eval-factory/stored-agent-memory/private-v1"] = (
        "eval-factory/stored-agent-memory/private-v1"
    )
    record: MemoryRecordV1
    content: MemoryContent

    @model_validator(mode="after")
    def validate_content(self) -> StoredMemoryV1:
        observed = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if observed != self.record.content_sha256:
            raise ValueError("stored memory content differs from record")
        return self


class MemoryRecallQueryV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-memory-recall-query/private-v1"] = (
        "eval-factory/agent-memory-recall-query/private-v1"
    )
    access: MemoryAccessContextV1
    query_text: MemoryQueryText
    mode: MemoryRetrievalModeV1 = MemoryRetrievalModeV1.LEXICAL
    kinds: tuple[MemoryKindV1, ...] = ()
    required_tags: tuple[Identifier, ...] = Field(default=(), max_length=32)
    limit: int = Field(default=10, ge=1, le=100)
    evaluated_at: datetime
    recency_half_life_seconds: int = Field(default=2_592_000, ge=1, le=315_576_000)

    @field_validator("kinds")
    @classmethod
    def validate_kinds(
        cls,
        value: tuple[MemoryKindV1, ...],
    ) -> tuple[MemoryKindV1, ...]:
        if value != tuple(sorted(set(value), key=lambda item: item.value)):
            raise ValueError("memory kinds must be sorted and unique")
        return value

    @field_validator("required_tags")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("required memory tags must be sorted and unique")
        return value

    @field_validator("evaluated_at")
    @classmethod
    def validate_evaluated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory recall timestamp must be timezone-aware")
        return value


class MemoryRecallMatchV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-memory-recall-match/private-v1"] = (
        "eval-factory/agent-memory-recall-match/private-v1"
    )
    memory: StoredMemoryV1
    total_score_basis_points: int = Field(ge=0, le=10_000)
    lexical_score_basis_points: int = Field(ge=0, le=10_000)
    semantic_score_basis_points: int | None = Field(default=None, ge=0, le=10_000)
    recency_score_basis_points: int = Field(ge=0, le=10_000)
    importance_score_basis_points: int = Field(ge=0, le=10_000)


class MemoryRecallResultV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-memory-recall-result/private-v1"] = (
        "eval-factory/agent-memory-recall-result/private-v1"
    )
    query_sha256: Sha256
    mode: MemoryRetrievalModeV1
    evaluated_at: datetime
    matches: tuple[MemoryRecallMatchV1, ...]
    candidate_count: int = Field(ge=0)


def _memory_record_sha256(value: MemoryRecordV1) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude={"memory_record_id", "object_sha256"},
        )
    )


def _tombstone_sha256(value: MemoryTombstoneV1) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude={"tombstone_id", "object_sha256"},
        )
    )


def _payload_sha256(value: object) -> str:
    return canonical_sha256_v2(_CanonicalPayload(value=value))


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


class _CanonicalPayload(ContractModelV2):
    value: object


__all__ = [
    "MemoryAccessContextV1",
    "MemoryCandidateV1",
    "MemoryHeadStateV1",
    "MemoryKindV1",
    "MemoryNamespaceV1",
    "MemoryRecallMatchV1",
    "MemoryRecallQueryV1",
    "MemoryRecallResultV1",
    "MemoryRecordV1",
    "MemoryRetrievalModeV1",
    "MemorySensitivityV1",
    "MemoryTombstoneV1",
    "MemoryVisibilityV1",
    "StoredMemoryV1",
]
