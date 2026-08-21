from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from eval_factory.contracts.core import ObjectRef
from eval_factory.trace.indexing import TraceIndexResult
from eval_factory.trace.indexing.models import (
    object_ref_for_observation,
    object_ref_for_segment,
)
from eval_factory.trace.normalization.models import (
    object_ref_for_event,
    object_ref_for_span,
)
from eval_factory.trace.storage.models import (
    StoredContentBlobRecord,
    StoredTraceManifest,
    TraceFactConflictError,
    TraceFactEnvelope,
    TraceFactKind,
    TraceFactPayload,
    TraceStoreCorruptionError,
)

FACT_LOG_RELATIVE_PATH = Path("facts") / "trace_facts.jsonl"


class CanonicalJsonValue(Protocol):
    def canonical_json(self) -> bytes: ...


def collect_fact_payloads(result: TraceIndexResult) -> tuple[TraceFactPayload, ...]:
    manifest = build_manifest(result)
    payloads: list[TraceFactPayload] = [
        _payload(
            TraceFactKind.TRACE_ENVELOPE,
            result.normalization_result.trace_ir_version_id,
            ObjectRef(
                object_type="trace-envelope",
                object_id=manifest.trace_ir_version_id,
                object_version="stored-manifest/v1",
                object_sha256=manifest.canonical_sha256(),
            ),
            manifest,
        )
    ]
    payloads.extend(
        _payload(
            TraceFactKind.SOURCE_SPAN,
            result.normalization_result.trace_ir_version_id,
            object_ref_for_span(item),
            item,
        )
        for item in result.normalization_result.source_spans
    )
    payloads.extend(
        _payload(
            TraceFactKind.REPAIR_MAP,
            result.normalization_result.trace_ir_version_id,
            ObjectRef(
                object_type="repair-map",
                object_id=item.repair_map_id,
                object_version="v1",
                object_sha256=item.canonical_sha256(),
            ),
            item,
        )
        for item in _repair_maps(result)
    )
    payloads.extend(
        _payload(
            TraceFactKind.TRACE_EVENT,
            result.normalization_result.trace_ir_version_id,
            object_ref_for_event(item),
            item,
        )
        for item in result.normalization_result.events
    )
    payloads.extend(
        _payload(
            TraceFactKind.TOOL_CALL_RECORD,
            result.normalization_result.trace_ir_version_id,
            ObjectRef(
                object_type="tool-call-record",
                object_id=item.tool_call_record_id,
                object_version="v1",
                object_sha256=item.canonical_sha256(),
            ),
            item,
        )
        for item in result.normalization_result.tool_call_records
    )
    payloads.extend(
        _payload(
            TraceFactKind.FILE_OBSERVATION,
            result.normalization_result.trace_ir_version_id,
            object_ref_for_observation(item),
            item,
        )
        for item in result.file_observations
    )
    payloads.extend(
        _payload(
            TraceFactKind.INTERACTION_SEGMENT,
            result.normalization_result.trace_ir_version_id,
            object_ref_for_segment(item),
            item,
        )
        for item in result.interaction_segments
    )
    payloads.extend(
        _payload(
            TraceFactKind.CONTENT_BLOB,
            result.normalization_result.trace_ir_version_id,
            item.content_ref,
            StoredContentBlobRecord(
                content_ref=item.content_ref,
                media_type=item.media_type,
                size_bytes=len(item.canonical_bytes),
            ),
        )
        for item in result.normalization_result.content_blobs
    )
    return tuple(
        sorted(
            payloads,
            key=lambda item: (
                _fact_order(item.fact_kind),
                item.object_ref.object_type,
                item.object_ref.object_id,
                item.object_ref.object_version,
            ),
        )
    )


def build_manifest(result: TraceIndexResult) -> StoredTraceManifest:
    registered = result.normalization_result.recovery_result.base_parse_result.registered_source
    recovery = result.normalization_result.recovery_result
    refs: list[ObjectRef] = []
    refs.extend(object_ref_for_span(item) for item in result.normalization_result.source_spans)
    refs.extend(object_ref_for_event(item) for item in result.normalization_result.events)
    refs.extend(
        ObjectRef(
            object_type="tool-call-record",
            object_id=item.tool_call_record_id,
            object_version="v1",
            object_sha256=item.canonical_sha256(),
        )
        for item in result.normalization_result.tool_call_records
    )
    refs.extend(object_ref_for_observation(item) for item in result.file_observations)
    refs.extend(object_ref_for_segment(item) for item in result.interaction_segments)
    refs.extend(item.content_ref for item in result.normalization_result.content_blobs)
    return StoredTraceManifest(
        trace_ir_version_id=result.normalization_result.trace_ir_version_id,
        source_trace_id=registered.source.source_trace_id,
        source_uri=registered.source.source_uri,
        raw_sha256=registered.source.raw_sha256,
        adapter_name=registered.source.adapter_name,
        adapter_version=registered.source.adapter_version,
        trace_ir_schema_version=result.normalization_result.trace_ir_schema_version,
        repair_policy_version=recovery.base_parse_result.repair_policy_version,
        recovery_policy_version=recovery.recovery_policy_version,
        normalization_policy_version=result.normalization_result.normalization_policy_version,
        tool_family_policy_version=result.normalization_result.tool_family_policy_version,
        file_observation_policy_version=result.file_observation_policy_version,
        segmentation_policy_version=result.segmentation_policy_version,
        parse_quality=recovery.parse_quality.value,
        object_refs=tuple(
            sorted(
                refs,
                key=lambda item: (item.object_type, item.object_id, item.object_version),
            )
        ),
    )


def read_fact_log(path: Path) -> tuple[TraceFactEnvelope, ...]:
    if not path.exists():
        return ()
    facts: list[TraceFactEnvelope] = []
    by_key: dict[tuple[str, str, str], TraceFactEnvelope] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceStoreCorruptionError(f"malformed fact JSONL at line {index + 1}") from exc
            try:
                envelope = TraceFactEnvelope.model_validate(value)
            except ValidationError as exc:
                raise TraceStoreCorruptionError(f"invalid fact envelope at line {index + 1}") from exc
            if envelope.jsonl_sequence != index:
                raise TraceStoreCorruptionError(
                    f"fact sequence mismatch at line {index + 1}: expected {index}, observed "
                    f"{envelope.jsonl_sequence}"
                )
            existing = by_key.get(envelope.object_key)
            if existing is not None and existing.object_sha256 != envelope.object_sha256:
                raise TraceFactConflictError(
                    f"conflicting fact for {envelope.object_type}/{envelope.object_id}"
                )
            by_key.setdefault(envelope.object_key, envelope)
            facts.append(envelope)
    return tuple(facts)


def append_missing_facts(
    path: Path,
    *,
    existing: tuple[TraceFactEnvelope, ...],
    payloads: tuple[TraceFactPayload, ...],
) -> tuple[tuple[TraceFactEnvelope, ...], int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key = {item.object_key: item for item in existing}
    appended: list[TraceFactEnvelope] = []
    next_sequence = len(existing)
    for payload in payloads:
        prior = by_key.get(payload.object_key)
        if prior is not None:
            if prior.object_sha256 != payload.object_ref.object_sha256:
                raise TraceFactConflictError(
                    f"conflicting fact for {payload.object_ref.object_type}/{payload.object_ref.object_id}"
                )
            continue
        envelope = TraceFactEnvelope(
            fact_kind=payload.fact_kind,
            trace_ir_version_id=payload.trace_ir_version_id,
            object_type=payload.object_ref.object_type,
            object_id=payload.object_ref.object_id,
            object_version=payload.object_ref.object_version,
            object_sha256=payload.object_ref.object_sha256,
            canonical_json=payload.canonical_json,
            jsonl_sequence=next_sequence,
        )
        appended.append(envelope)
        by_key[envelope.object_key] = envelope
        next_sequence += 1
    if appended:
        with path.open("ab") as handle:
            for item in appended:
                handle.write(item.jsonl_bytes())
    return tuple(appended), len(payloads) - len(appended)


def batch_sha256(facts: tuple[TraceFactEnvelope, ...]) -> str:
    encoded = b"".join(item.canonical_json_for_log() for item in facts)
    return hashlib.sha256(encoded).hexdigest()


def projection_sha256(rows: object) -> str:
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _payload(
    fact_kind: TraceFactKind,
    trace_ir_version_id: str,
    ref: ObjectRef,
    value: CanonicalJsonValue,
) -> TraceFactPayload:
    return TraceFactPayload(
        fact_kind=fact_kind,
        trace_ir_version_id=trace_ir_version_id,
        object_ref=ref,
        canonical_json=value.canonical_json().decode(),
    )


def _repair_maps(result: TraceIndexResult) -> tuple[Any, ...]:
    seen: dict[str, Any] = {}
    for item in (
        *result.normalization_result.recovery_result.base_parse_result.repair_maps,
        *result.normalization_result.recovery_result.recovery_maps,
    ):
        seen.setdefault(item.repair_map_id, item)
    return tuple(seen.values())


def _fact_order(kind: TraceFactKind) -> int:
    return {
        TraceFactKind.TRACE_ENVELOPE: 0,
        TraceFactKind.SOURCE_SPAN: 1,
        TraceFactKind.REPAIR_MAP: 2,
        TraceFactKind.TRACE_EVENT: 3,
        TraceFactKind.TOOL_CALL_RECORD: 4,
        TraceFactKind.FILE_OBSERVATION: 5,
        TraceFactKind.INTERACTION_SEGMENT: 6,
        TraceFactKind.CONTENT_BLOB: 7,
    }[kind]
