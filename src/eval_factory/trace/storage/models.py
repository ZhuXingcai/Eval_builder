from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractModel,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.trace import (
    FileObservation,
    InteractionSegment,
    SourceSpan,
    ToolCallRecord,
    TraceEvent,
)
from eval_factory.trace.normalization import NormalizedContentBlob

TRACE_FACT_ENVELOPE_VERSION: Literal["eval-factory/trace-fact-envelope/v1"] = (
    "eval-factory/trace-fact-envelope/v1"
)
TRACE_STORE_MANIFEST_VERSION: Literal["eval-factory/stored-trace-manifest/v1"] = (
    "eval-factory/stored-trace-manifest/v1"
)
CONTENT_BLOB_RECORD_VERSION: Literal["eval-factory/stored-content-blob/v1"] = (
    "eval-factory/stored-content-blob/v1"
)


class TraceStoreError(RuntimeError):
    pass


class TraceFactConflictError(TraceStoreError):
    pass


class TraceCasConflictError(TraceStoreError):
    pass


class TraceStoreCorruptionError(TraceStoreError):
    pass


class TraceProjectionError(TraceStoreError):
    pass


class TraceObjectNotFoundError(TraceStoreError):
    pass


class TraceFactKind(StrEnum):
    TRACE_ENVELOPE = "trace-envelope"
    SOURCE_SPAN = "source-span"
    REPAIR_MAP = "repair-map"
    TRACE_EVENT = "trace-event"
    TOOL_CALL_RECORD = "tool-call-record"
    FILE_OBSERVATION = "file-observation"
    INTERACTION_SEGMENT = "interaction-segment"
    CONTENT_BLOB = "content-blob"


class StoredTraceManifest(ContractModel):
    schema_version: Literal["eval-factory/stored-trace-manifest/v1"] = TRACE_STORE_MANIFEST_VERSION
    trace_ir_version_id: Identifier
    source_trace_id: Identifier
    source_uri: str = Field(min_length=3, max_length=1024)
    raw_sha256: Sha256
    adapter_name: Identifier
    adapter_version: str = Field(min_length=1, max_length=128)
    trace_ir_schema_version: str = Field(min_length=1, max_length=128)
    repair_policy_version: str = Field(min_length=1, max_length=128)
    recovery_policy_version: str = Field(min_length=1, max_length=128)
    normalization_policy_version: str = Field(min_length=1, max_length=128)
    tool_family_policy_version: str = Field(min_length=1, max_length=128)
    file_observation_policy_version: str = Field(min_length=1, max_length=128)
    segmentation_policy_version: str = Field(min_length=1, max_length=128)
    parse_quality: str = Field(min_length=1, max_length=64)
    object_refs: tuple[ObjectRef, ...]


class StoredContentBlobRecord(ContractModel):
    schema_version: Literal["eval-factory/stored-content-blob/v1"] = CONTENT_BLOB_RECORD_VERSION
    content_ref: ObjectRef
    media_type: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_ref(self) -> StoredContentBlobRecord:
        if self.content_ref.object_type != "content-blob":
            raise ValueError("stored content blob requires a content-blob ref")
        return self


class TraceFactEnvelope(ContractModel):
    schema_version: Literal["eval-factory/trace-fact-envelope/v1"] = TRACE_FACT_ENVELOPE_VERSION
    fact_kind: TraceFactKind
    trace_ir_version_id: Identifier
    object_type: Identifier
    object_id: Identifier
    object_version: str = Field(min_length=1, max_length=128)
    object_sha256: Sha256
    payload_json: str = Field(alias="canonical_json", min_length=2)
    jsonl_sequence: int = Field(ge=0)

    @field_validator("fact_kind", mode="before")
    @classmethod
    def parse_fact_kind(cls, value: object) -> TraceFactKind:
        if isinstance(value, TraceFactKind):
            return value
        if isinstance(value, str):
            return TraceFactKind(value)
        raise TypeError("fact_kind must be a trace fact kind")

    @model_validator(mode="after")
    def validate_payload_hash(self) -> TraceFactEnvelope:
        if self.fact_kind is TraceFactKind.CONTENT_BLOB:
            return self
        observed = hashlib.sha256(self.payload_json.encode()).hexdigest()
        if observed != self.object_sha256:
            raise ValueError(
                f"fact payload hash mismatch for {self.object_type}/{self.object_id}: "
                f"expected {self.object_sha256}, observed {observed}"
            )
        return self

    @property
    def object_key(self) -> tuple[str, str, str]:
        return (self.object_type, self.object_id, self.object_version)

    def jsonl_bytes(self) -> bytes:
        return self.canonical_json_for_log() + b"\n"

    def canonical_json_for_log(self) -> bytes:
        value = self.model_dump(mode="json", by_alias=True, exclude_none=False)
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()


@dataclass(frozen=True)
class TraceFactPayload:
    fact_kind: TraceFactKind
    trace_ir_version_id: str
    object_ref: ObjectRef
    canonical_json: str

    @property
    def object_key(self) -> tuple[str, str, str]:
        return (
            self.object_ref.object_type,
            self.object_ref.object_id,
            self.object_ref.object_version,
        )


@dataclass(frozen=True)
class TraceStoreCommit:
    trace_ir_version_id: str
    batch_sha256: str
    appended_fact_count: int
    existing_fact_count: int
    written_blob_count: int
    existing_blob_count: int


@dataclass(frozen=True)
class TraceProjectionRebuild:
    fact_count: int
    projection_sha256: str


@dataclass(frozen=True)
class TraceTextHit:
    object_id: str
    object_sha256: str
    media_type: str


@dataclass(frozen=True)
class StoredTraceIndex:
    trace_ir_version_id: str
    manifest: StoredTraceManifest
    source_spans: tuple[SourceSpan, ...]
    events: tuple[TraceEvent, ...]
    tool_call_records: tuple[ToolCallRecord, ...]
    file_observations: tuple[FileObservation, ...]
    interaction_segments: tuple[InteractionSegment, ...]
    content_blobs: tuple[NormalizedContentBlob, ...]

    def event_summary(self) -> list[tuple[int, str, str | None]]:
        return [(item.sequence, item.event_type.value, item.role) for item in self.events]

    def tool_call_summary(self) -> list[tuple[int, str, str]]:
        return [(item.sequence, item.tool_family.value, item.status.value) for item in self.tool_call_records]

    def file_observation_summary(self) -> list[tuple[int, str, str, str]]:
        return [
            (item.sequence, item.logical_path, item.operation.value, item.completeness.value)
            for item in self.file_observations
        ]

    def segment_summary(self) -> list[tuple[int, int, str]]:
        return [
            (item.sequence_start, item.sequence_end, item.boundary_method)
            for item in self.interaction_segments
        ]

    def content_blob_summary(self) -> list[tuple[str, str, int]]:
        return [
            (item.content_ref.object_id, item.media_type, len(item.canonical_bytes))
            for item in self.content_blobs
        ]
