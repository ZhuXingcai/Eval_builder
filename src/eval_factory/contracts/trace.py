from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    RelativePath,
    Sha256,
    TypedAttribute,
)


class ParseQuality(StrEnum):
    STRICT = "STRICT"
    REPAIRED = "REPAIRED"
    PARTIAL = "PARTIAL"
    UNPARSEABLE = "UNPARSEABLE"


class CapabilityStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"


class TraceCapability(ContractModel):
    schema_version: Literal["eval-factory/trace-capability/v1"] = "eval-factory/trace-capability/v1"
    capability: Literal[
        "conversation_events",
        "tool_events",
        "call_result_pairing",
        "file_timeline",
        "final_response",
        "attachment_observations",
    ]
    status: CapabilityStatus
    reason_codes: tuple[Identifier, ...] = ()


class SourceSpan(ContractModel):
    schema_version: Literal["eval-factory/source-span/v1"] = "eval-factory/source-span/v1"
    span_id: Identifier
    source_uri: str = Field(min_length=3, max_length=1024)
    source_trace_id: Identifier
    outer_record_index: int = Field(ge=0)
    field: Literal["request", "response", "extra", "outer"]
    raw_byte_start: int = Field(ge=0)
    raw_byte_end: int = Field(ge=0)
    decoded_char_start: int | None = Field(default=None, ge=0)
    decoded_char_end: int | None = Field(default=None, ge=0)
    repair_map_ref: ObjectRef | None = None
    approximate: bool
    raw_sha256: Sha256

    @model_validator(mode="after")
    def validate_ranges(self) -> SourceSpan:
        if self.raw_byte_end < self.raw_byte_start:
            raise ValueError("raw byte range must be ordered")
        if (self.decoded_char_start is None) != (self.decoded_char_end is None):
            raise ValueError("decoded range endpoints must both be present or absent")
        if (
            self.decoded_char_start is not None
            and self.decoded_char_end is not None
            and self.decoded_char_end < self.decoded_char_start
        ):
            raise ValueError("decoded character range must be ordered")
        return self


class RepairSegment(ContractModel):
    schema_version: Literal["eval-factory/repair-segment/v1"] = "eval-factory/repair-segment/v1"
    raw_byte_start: int = Field(ge=0)
    raw_byte_end: int = Field(ge=0)
    decoded_char_start: int = Field(ge=0)
    decoded_char_end: int = Field(ge=0)
    transform: Identifier
    rule_id: Identifier
    exact: bool


class RepairMap(ContractModel):
    schema_version: Literal["eval-factory/repair-map/v1"] = "eval-factory/repair-map/v1"
    repair_map_id: Identifier
    source_trace_id: Identifier
    field: Literal["request", "response", "extra"]
    segments: tuple[RepairSegment, ...] = Field(min_length=1)
    semantic_critical: bool
    audit: ContractAudit


class TraceEventType(StrEnum):
    USER_TEXT = "user_text"
    ASSISTANT_TEXT = "assistant_text"
    SYSTEM_CONTEXT = "system_context"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RUNTIME_ERROR = "runtime_error"
    CONTINUATION_MARKER = "continuation_marker"
    ATTACHMENT_REFERENCE = "attachment_reference"


class TraceEvent(ContractModel):
    schema_version: Literal["eval-factory/trace-event/v1"] = "eval-factory/trace-event/v1"
    event_id: Identifier
    trace_ir_version_id: Identifier
    sequence: int = Field(ge=0)
    event_type: TraceEventType
    role: Literal["user", "assistant", "system", "tool", "runtime"] | None = None
    content_ref: ObjectRef | None = None
    source_spans: tuple[ObjectRef, ...] = Field(min_length=1)
    attributes: tuple[TypedAttribute, ...] = ()


class ToolFamily(StrEnum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    FILE_LIST = "file_list"
    SEARCH = "search"
    FETCH = "fetch"
    SHELL = "shell"
    CODE_EXECUTION = "code_execution"
    USER_INTERACTION = "user_interaction"
    SUBAGENT = "subagent"
    TASK_MANAGEMENT = "task_management"
    UNKNOWN = "unknown"


class ToolCallStatus(StrEnum):
    PAIRED_SUCCESS = "PAIRED_SUCCESS"
    PAIRED_ERROR = "PAIRED_ERROR"
    ORPHAN_CALL = "ORPHAN_CALL"
    ORPHAN_RESULT = "ORPHAN_RESULT"
    AMBIGUOUS = "AMBIGUOUS"
    DUPLICATE_ID = "DUPLICATE_ID"


class ToolCallRecord(ContractModel):
    schema_version: Literal["eval-factory/tool-call-record/v1"] = "eval-factory/tool-call-record/v1"
    tool_call_record_id: Identifier
    trace_ir_version_id: Identifier
    call_id: str | None = Field(default=None, max_length=255)
    original_name: str = Field(min_length=1, max_length=255)
    tool_family: ToolFamily
    arguments_ref: ObjectRef | None = None
    call_event_ref: ObjectRef | None = None
    result_event_ref: ObjectRef | None = None
    status: ToolCallStatus
    error_signature: str | None = Field(default=None, max_length=1000)
    sequence: int = Field(ge=0)


class FileOperation(StrEnum):
    READ = "READ"
    WRITE = "WRITE"
    EDIT = "EDIT"
    COPY = "COPY"
    MOVE = "MOVE"
    DELETE = "DELETE"
    LIST = "LIST"
    UNKNOWN_MUTATION = "UNKNOWN_MUTATION"


class Completeness(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class FileObservation(ContractModel):
    schema_version: Literal["eval-factory/file-observation/v1"] = "eval-factory/file-observation/v1"
    observation_id: Identifier
    trace_ir_version_id: Identifier
    logical_path: RelativePath
    raw_path_ref: ObjectRef
    operation: FileOperation
    sequence: int = Field(ge=0)
    observed_start: int | None = Field(default=None, ge=0)
    observed_end: int | None = Field(default=None, ge=0)
    completeness: Completeness
    truncated: bool | None
    content_ref: ObjectRef | None
    content_sha256: Sha256 | None
    file_version_id: Identifier
    source_event_refs: tuple[ObjectRef, ...] = Field(min_length=1)


class InteractionSegment(ContractModel):
    schema_version: Literal["eval-factory/interaction-segment/v1"] = "eval-factory/interaction-segment/v1"
    segment_id: Identifier
    trace_ir_version_id: Identifier
    boundary_method: Identifier
    sequence_start: int = Field(ge=0)
    sequence_end: int = Field(ge=0)
    member_event_refs: tuple[ObjectRef, ...] = Field(min_length=1)


class TraceEnvelope(ContractModel):
    schema_version: Literal["eval-factory/trace-envelope/v1"] = "eval-factory/trace-envelope/v1"
    source_trace_id: Identifier
    trace_ir_version_id: Identifier
    source_uri: str = Field(min_length=3, max_length=1024)
    raw_sha256: Sha256
    adapter_name: Identifier
    adapter_version: str = Field(min_length=1, max_length=128)
    trace_ir_schema_version: Literal["v1"]
    repair_policy_version: str = Field(min_length=1, max_length=128)
    segmentation_policy_version: str = Field(min_length=1, max_length=128)
    parse_quality: ParseQuality
    capabilities: tuple[TraceCapability, ...] = Field(min_length=1)
    event_refs: tuple[ObjectRef, ...]
    tool_call_refs: tuple[ObjectRef, ...]
    file_observation_refs: tuple[ObjectRef, ...]
    segment_refs: tuple[ObjectRef, ...]
    repair_map_refs: tuple[ObjectRef, ...] = ()
    unrecoverable_span_refs: tuple[ObjectRef, ...] = ()
    audit: ContractAudit
