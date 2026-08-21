from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    Sha256,
    TypedAttribute,
)
from eval_factory.contracts.trace import (
    SourceSpan,
    ToolCallRecord,
    TraceEvent,
)
from eval_factory.trace.parsing.recovery import TraceRecoveryResult

TRACE_IR_SCHEMA_VERSION = "v1"
NORMALIZATION_POLICY_VERSION = "raw-traj-event-normalization/v1"
TOOL_FAMILY_POLICY_VERSION = "raw-traj-tool-family/v1"
DECLARED_SEGMENTATION_POLICY_VERSION = "deterministic-interaction-segments/v1"

type JsonValue = None | bool | int | float | str | tuple[JsonValue, ...] | Mapping[str, JsonValue]
type MediaType = Literal["application/json", "text/plain; charset=utf-8"]


@dataclass(frozen=True)
class NormalizationDiagnostic:
    code: str
    outer_record_index: int
    field: str
    source_span_ref: ObjectRef | None = None
    event_ref: ObjectRef | None = None
    attributes: tuple[TypedAttribute, ...] = ()


@dataclass(frozen=True)
class NormalizedContentBlob:
    content_ref: ObjectRef
    media_type: MediaType
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        observed = hashlib.sha256(self.canonical_bytes).hexdigest()
        if observed != self.content_ref.object_sha256:
            raise ValueError("content ref hash does not match canonical bytes")


@dataclass(frozen=True)
class TraceNormalizationResult:
    recovery_result: TraceRecoveryResult
    trace_ir_version_id: str
    trace_ir_schema_version: str
    normalization_policy_version: str
    tool_family_policy_version: str
    declared_segmentation_policy_version: str
    events: tuple[TraceEvent, ...]
    tool_call_records: tuple[ToolCallRecord, ...]
    source_spans: tuple[SourceSpan, ...]
    content_blobs: tuple[NormalizedContentBlob, ...]
    diagnostics: tuple[NormalizationDiagnostic, ...]
    unrecoverable_spans: tuple[SourceSpan, ...]
    audit: ContractAudit

    def __post_init__(self) -> None:
        if self.trace_ir_schema_version != TRACE_IR_SCHEMA_VERSION:
            raise ValueError("unexpected TraceIR schema version")
        if self.normalization_policy_version != NORMALIZATION_POLICY_VERSION:
            raise ValueError("unexpected normalization policy version")
        if self.tool_family_policy_version != TOOL_FAMILY_POLICY_VERSION:
            raise ValueError("unexpected tool-family policy version")
        if self.declared_segmentation_policy_version != DECLARED_SEGMENTATION_POLICY_VERSION:
            raise ValueError("unexpected declared segmentation policy version")
        if self.unrecoverable_spans != self.recovery_result.unrecoverable_spans:
            raise ValueError("normalization cannot change unrecoverable spans")
        _require_unique_contracts(
            "source span",
            ((item.span_id, item.canonical_sha256()) for item in self.source_spans),
        )
        _require_unique_contracts(
            "event",
            ((item.event_id, item.canonical_sha256()) for item in self.events),
        )
        _require_unique_contracts(
            "tool call record",
            ((item.tool_call_record_id, item.canonical_sha256()) for item in self.tool_call_records),
        )
        _require_unique_contracts(
            "content blob",
            ((item.content_ref.object_id, item.content_ref.object_sha256) for item in self.content_blobs),
        )
        source_refs = {
            item.span_id: ObjectRef(
                object_type="source-span",
                object_id=item.span_id,
                object_version="v1",
                object_sha256=item.canonical_sha256(),
            )
            for item in self.source_spans
        }
        content_refs = {item.content_ref.object_id: item.content_ref for item in self.content_blobs}
        event_refs = {
            item.event_id: ObjectRef(
                object_type="trace-event",
                object_id=item.event_id,
                object_version=TRACE_IR_SCHEMA_VERSION,
                object_sha256=item.canonical_sha256(),
            )
            for item in self.events
        }
        for event in self.events:
            for ref in event.source_spans:
                expected = source_refs.get(ref.object_id)
                if expected != ref:
                    raise ValueError(f"event references unresolved source span: {ref.object_id}")
            if (
                event.content_ref is not None
                and content_refs.get(event.content_ref.object_id) != event.content_ref
            ):
                raise ValueError(f"event references unresolved content blob: {event.content_ref.object_id}")
        for record in self.tool_call_records:
            if (
                record.arguments_ref is not None
                and content_refs.get(record.arguments_ref.object_id) != record.arguments_ref
            ):
                raise ValueError("tool call record references unresolved arguments blob")
            if (
                record.call_event_ref is not None
                and event_refs.get(record.call_event_ref.object_id) != record.call_event_ref
            ):
                raise ValueError("tool call record references unresolved call event")
            if (
                record.result_event_ref is not None
                and event_refs.get(record.result_event_ref.object_id) != record.result_event_ref
            ):
                raise ValueError("tool call record references unresolved result event")


def canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        _json_plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def content_ref_for_bytes(
    value: bytes,
    *,
    media_type: MediaType,
) -> ObjectRef:
    digest = hashlib.sha256(value).hexdigest()
    return ObjectRef(
        object_type="content-blob",
        object_id=f"content://sha256/{digest}",
        object_version=media_type,
        object_sha256=digest,
    )


def object_ref_for_event(event: TraceEvent) -> ObjectRef:
    return ObjectRef(
        object_type="trace-event",
        object_id=event.event_id,
        object_version=TRACE_IR_SCHEMA_VERSION,
        object_sha256=event.canonical_sha256(),
    )


def object_ref_for_span(span: SourceSpan) -> ObjectRef:
    return ObjectRef(
        object_type="source-span",
        object_id=span.span_id,
        object_version="v1",
        object_sha256=span.canonical_sha256(),
    )


def stable_hash(value: Mapping[str, object]) -> Sha256:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_id(prefix: str, value: Mapping[str, object]) -> str:
    return f"{prefix}://sha256/{stable_hash(value)}"


def typed_attr(key: str, value: str | int | float | bool | None) -> TypedAttribute:
    return TypedAttribute(key=key, value=value)


def _json_plain(value: JsonValue) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_json_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in sorted(value.items())}
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _require_unique_contracts(
    label: str,
    entries: Iterable[tuple[str, str]],
) -> None:
    seen: dict[str, str] = {}
    for object_id, digest in entries:
        existing = seen.get(object_id)
        if existing is not None and existing != digest:
            raise ValueError(f"duplicate {label} id with different hash: {object_id}")
        seen[object_id] = digest
