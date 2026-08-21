from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.trace import (
    Completeness,
    FileObservation,
    FileOperation,
    ToolCallRecord,
    ToolCallStatus,
    ToolFamily,
    TraceEvent,
)
from eval_factory.trace.indexing.models import (
    FILE_OBSERVATION_POLICY_VERSION,
    IndexDiagnostic,
)
from eval_factory.trace.indexing.paths import extract_raw_path, project_logical_path
from eval_factory.trace.normalization import TraceNormalizationResult
from eval_factory.trace.normalization.models import (
    NormalizedContentBlob,
    content_ref_for_bytes,
    object_ref_for_event,
    stable_id,
    typed_attr,
)

_FAMILY_TO_OPERATION: dict[ToolFamily, FileOperation] = {
    ToolFamily.FILE_READ: FileOperation.READ,
    ToolFamily.FILE_WRITE: FileOperation.WRITE,
    ToolFamily.FILE_EDIT: FileOperation.EDIT,
    ToolFamily.FILE_LIST: FileOperation.LIST,
}


@dataclass(frozen=True)
class FileObservationBuild:
    observations: tuple[FileObservation, ...]
    raw_path_blobs: tuple[NormalizedContentBlob, ...]
    diagnostics: tuple[IndexDiagnostic, ...]


def build_file_observations(
    normalization_result: TraceNormalizationResult,
) -> FileObservationBuild:
    events = {event.event_id: event for event in normalization_result.events}
    blobs = {blob.content_ref.object_id: blob for blob in normalization_result.content_blobs}
    diagnostics: list[IndexDiagnostic] = []
    observations: list[FileObservation] = []
    raw_path_blobs: dict[str, NormalizedContentBlob] = {}

    for record in normalization_result.tool_call_records:
        operation = _FAMILY_TO_OPERATION.get(record.tool_family)
        if operation is None:
            continue
        call_event = _event_from_ref(record.call_event_ref, events)
        result_event = _event_from_ref(record.result_event_ref, events)
        if call_event is None:
            diagnostics.append(
                _diagnostic(
                    "file-observation-arguments-missing",
                    record,
                    result_event,
                )
            )
            continue
        arguments = _json_from_ref(record.arguments_ref, blobs)
        if arguments is None:
            diagnostics.append(_diagnostic("file-observation-arguments-missing", record, call_event))
            continue
        raw_path = extract_raw_path(arguments)
        if raw_path is None:
            diagnostics.append(_diagnostic("file-observation-path-missing", record, call_event))
            continue
        projected = project_logical_path(raw_path)
        if projected is None:
            diagnostics.append(_diagnostic("file-observation-path-unsafe", record, call_event))
            continue
        raw_path_ref, raw_path_blob = _raw_path_blob(projected.raw)
        raw_path_blobs.setdefault(raw_path_ref.object_id, raw_path_blob)
        content_ref = _content_ref_for_operation(
            operation=operation,
            arguments=arguments,
            record=record,
            result_event=result_event,
        )
        completeness = _completeness(
            operation=operation,
            record=record,
            call_event=call_event,
            result_event=result_event,
            content_ref=content_ref,
        )
        truncated = _truncated(record=record, call_event=call_event, result_event=result_event)
        source_events = _source_event_refs(call_event, result_event)
        observation_id = stable_id(
            "file-observation",
            {
                "trace_ir_version_id": normalization_result.trace_ir_version_id,
                "file_observation_policy_version": FILE_OBSERVATION_POLICY_VERSION,
                "logical_path": projected.value,
                "operation": operation.value,
                "sequence": record.sequence,
                "source_event_ids": [ref.object_id for ref in source_events],
            },
        )
        file_version_id = stable_id(
            "file-version",
            {
                "trace_ir_version_id": normalization_result.trace_ir_version_id,
                "file_observation_policy_version": FILE_OBSERVATION_POLICY_VERSION,
                "logical_path": projected.value,
                "operation": operation.value,
                "sequence": record.sequence,
                "source_event_ids": [ref.object_id for ref in source_events],
            },
        )
        observations.append(
            FileObservation(
                observation_id=observation_id,
                trace_ir_version_id=normalization_result.trace_ir_version_id,
                logical_path=projected.value,
                raw_path_ref=raw_path_ref,
                operation=operation,
                sequence=record.sequence,
                observed_start=None,
                observed_end=None,
                completeness=completeness,
                truncated=truncated,
                content_ref=content_ref,
                content_sha256=(content_ref.object_sha256 if content_ref is not None else None),
                file_version_id=file_version_id,
                source_event_refs=source_events,
            )
        )

    return FileObservationBuild(
        observations=tuple(sorted(observations, key=lambda item: item.sequence)),
        raw_path_blobs=tuple(sorted(raw_path_blobs.values(), key=lambda item: item.content_ref.object_id)),
        diagnostics=tuple(diagnostics),
    )


def _json_from_ref(
    ref: ObjectRef | None,
    blobs: Mapping[str, NormalizedContentBlob],
) -> dict[str, object] | None:
    if ref is None:
        return None
    blob = blobs.get(ref.object_id)
    if blob is None or blob.media_type != "application/json":
        return None
    value = json.loads(blob.canonical_bytes)
    return value if isinstance(value, dict) else None


def _event_from_ref(
    ref: ObjectRef | None,
    events: Mapping[str, TraceEvent],
) -> TraceEvent | None:
    if ref is None:
        return None
    return events.get(ref.object_id)


def _raw_path_blob(raw_path: str) -> tuple[ObjectRef, NormalizedContentBlob]:
    encoded = raw_path.encode()
    ref = content_ref_for_bytes(encoded, media_type="text/plain; charset=utf-8")
    return ref, NormalizedContentBlob(
        content_ref=ref,
        media_type="text/plain; charset=utf-8",
        canonical_bytes=encoded,
    )


def _content_ref_for_operation(
    *,
    operation: FileOperation,
    arguments: Mapping[str, object],
    record: ToolCallRecord,
    result_event: TraceEvent | None,
) -> ObjectRef | None:
    if operation in {FileOperation.WRITE, FileOperation.EDIT}:
        if "content" in arguments:
            return record.arguments_ref
        if "new_string" in arguments:
            return record.arguments_ref
    if result_event is not None and record.status in {
        ToolCallStatus.PAIRED_SUCCESS,
        ToolCallStatus.PAIRED_ERROR,
    }:
        return result_event.content_ref
    return None


def _completeness(
    *,
    operation: FileOperation,
    record: ToolCallRecord,
    call_event: TraceEvent,
    result_event: TraceEvent | None,
    content_ref: ObjectRef | None,
) -> Completeness:
    if _has_truncation(call_event) or (result_event is not None and _has_truncation(result_event)):
        return Completeness.PARTIAL
    if record.status is ToolCallStatus.ORPHAN_CALL or record.status is ToolCallStatus.ORPHAN_RESULT:
        return Completeness.UNKNOWN
    if record.status in {ToolCallStatus.AMBIGUOUS, ToolCallStatus.DUPLICATE_ID}:
        return Completeness.UNKNOWN
    if operation in {FileOperation.WRITE, FileOperation.EDIT} and content_ref is not None:
        return Completeness.COMPLETE
    if operation in {FileOperation.READ, FileOperation.LIST} and result_event is not None:
        return Completeness.PARTIAL
    return Completeness.UNKNOWN


def _truncated(
    *,
    record: ToolCallRecord,
    call_event: TraceEvent,
    result_event: TraceEvent | None,
) -> bool | None:
    if _has_truncation(call_event) or (result_event is not None and _has_truncation(result_event)):
        return True
    if record.status in {ToolCallStatus.PAIRED_SUCCESS, ToolCallStatus.PAIRED_ERROR}:
        return False
    return None


def _has_truncation(event: TraceEvent) -> bool:
    return any(item.key in {"truncated", "is_truncated"} and item.value is True for item in event.attributes)


def _source_event_refs(
    call_event: TraceEvent,
    result_event: TraceEvent | None,
) -> tuple[ObjectRef, ...]:
    refs = [object_ref_for_event(call_event)]
    if result_event is not None and result_event.event_id != call_event.event_id:
        refs.append(object_ref_for_event(result_event))
    return tuple(refs)


def _diagnostic(
    code: str,
    record: ToolCallRecord,
    event: TraceEvent | None,
) -> IndexDiagnostic:
    return IndexDiagnostic(
        code=code,
        event_ref=(object_ref_for_event(event) if event is not None else None),
        tool_call_record_ref=ObjectRef(
            object_type="tool-call-record",
            object_id=record.tool_call_record_id,
            object_version="v1",
            object_sha256=record.canonical_sha256(),
        ),
        attributes=(
            typed_attr("tool_family", record.tool_family.value),
            typed_attr("status", record.status.value),
        ),
    )
