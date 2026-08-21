from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.trace import (
    FileObservation,
    InteractionSegment,
    SourceSpan,
    ToolCallRecord,
    TraceEvent,
)
from eval_factory.trace.normalization.models import object_ref_for_event, object_ref_for_span
from eval_factory.trace.query.models import (
    QueryAuthorizationError,
    QueryBudgetExceededError,
    QueryKind,
    QueryValidationError,
    TraceEvidenceRequest,
    TraceQueryRequest,
    TraceQueryResponse,
    TraceTextFragment,
    query_id_for,
)
from eval_factory.trace.storage import TraceIndexStore
from eval_factory.trace.storage.models import StoredTraceIndex


class TraceQueryService:
    def __init__(self, store: TraceIndexStore) -> None:
        self.store = store

    def query(self, request: TraceQueryRequest) -> TraceQueryResponse:
        self._authorize(request)
        stored = self.store.load(request.trace_ir_version_id)
        refs, spans, fragments, truncated = self._select(stored, request)
        fragments, text_truncated = _budget_fragments(fragments, request.max_characters)
        truncated = truncated or text_truncated
        return TraceQueryResponse(
            query_id=query_id_for(request, refs),
            trace_ir_version_id=request.trace_ir_version_id,
            principal_id=request.principal.principal_id,
            consumer_stage=request.principal.consumer_stage,
            consumer_agent=request.principal.consumer_agent,
            purpose=request.purpose,
            query_kind=request.query_kind,
            result_refs=refs,
            source_span_refs=spans,
            text_fragments=fragments,
            returned_characters=sum(len(item.text) for item in fragments),
            max_characters=request.max_characters,
            truncated=truncated,
            audit=request.principal.audit,
        )

    def evidence_bundle(self, request: TraceEvidenceRequest) -> EvidenceBundle:
        response = self.query(request.query)
        stored = self.store.load(request.query.trace_ir_version_id)
        source_trace_id = stored.manifest.source_trace_id
        evidence = tuple(
            EvidenceRef(
                evidence_ref_id=_stable_id(
                    "evidence-ref",
                    {
                        "query_id": response.query_id,
                        "subject": ref.model_dump(mode="json", exclude_none=False),
                        "spans": [
                            span.model_dump(mode="json", exclude_none=False)
                            for span in response.source_span_refs
                        ],
                    },
                ),
                subject_ref=ref,
                source_spans=response.source_span_refs or (_fallback_span_ref(stored),),
                polarity=EvidencePolarity.POSITIVE,
                capability="trace-query",
                capability_complete=False,
            )
            for ref in response.result_refs
        )
        policy_ref = _projection_policy_ref(request.query.principal.consumer_stage, request.query.purpose)
        bundle_seed = {
            "query_id": response.query_id,
            "refs": [item.model_dump(mode="json", exclude_none=False) for item in response.result_refs],
            "spans": [item.model_dump(mode="json", exclude_none=False) for item in response.source_span_refs],
            "returned_characters": response.returned_characters,
            "max_characters": response.max_characters,
        }
        return EvidenceBundle(
            evidence_bundle_id=_stable_id("evidence-bundle", bundle_seed),
            source_trace_id=source_trace_id,
            trace_ir_version_id=response.trace_ir_version_id,
            consumer_stage=response.consumer_stage,
            purpose=response.purpose,
            projection_policy_ref=policy_ref,
            evidence=evidence,
            returned_characters=response.returned_characters,
            max_characters=response.max_characters,
            tainted_content_included=False,
            bundle_sha256=_stable_hash(bundle_seed),
            audit=ContractAudit(
                created_at=datetime(2026, 7, 22, tzinfo=UTC),
                created_by="trace-query-service",
                governing_versions=(VersionBinding(component="trace-query-service", version="v1"),),
                input_refs=response.result_refs,
            ),
        )

    def _authorize(self, request: TraceQueryRequest) -> None:
        principal = request.principal
        if request.trace_ir_version_id not in principal.allowed_trace_ir_version_ids:
            raise QueryAuthorizationError("trace scope is not authorized for principal")
        if request.purpose not in principal.allowed_purposes:
            raise QueryAuthorizationError("purpose is not authorized for principal")
        if request.limit > principal.max_events:
            raise QueryBudgetExceededError("requested event limit exceeds principal event limit")
        if request.max_characters > principal.max_characters:
            raise QueryBudgetExceededError("requested character budget exceeds principal character budget")
        if request.allow_tainted and not principal.audit:
            raise QueryAuthorizationError("tainted content requires an audit principal")

    def _select(
        self,
        stored: StoredTraceIndex,
        request: TraceQueryRequest,
    ) -> tuple[tuple[ObjectRef, ...], tuple[SourceSpanRef, ...], tuple[TraceTextFragment, ...], bool]:
        if request.query_kind is QueryKind.MANIFEST:
            ref = ObjectRef(
                object_type="trace-envelope",
                object_id=stored.manifest.trace_ir_version_id,
                object_version="stored-manifest/v1",
                object_sha256=stored.manifest.canonical_sha256(),
            )
            return (ref,), (), (), False
        if request.query_kind is QueryKind.EVENTS:
            return self._events(stored, request)
        if request.query_kind is QueryKind.EVENT:
            event_id = _required_str(request, "event_id")
            return self._event(stored, event_id)
        if request.query_kind is QueryKind.SOURCE_SPAN:
            span_id = _required_str(request, "span_id")
            return self._span(stored, span_id)
        if request.query_kind is QueryKind.TOOL_CALLS:
            return self._tool_calls(stored, request)
        if request.query_kind is QueryKind.FILE_OBSERVATIONS:
            return self._file_observations(stored, request)
        if request.query_kind is QueryKind.SEGMENTS:
            return self._segments(stored, request)
        if request.query_kind is QueryKind.TEXT_SEARCH:
            text = _required_str(request, "text")
            return self._text_search(stored, request, text)
        raise QueryValidationError(f"unsupported query kind: {request.query_kind}")

    def _events(
        self,
        stored: StoredTraceIndex,
        request: TraceQueryRequest,
    ) -> tuple[tuple[ObjectRef, ...], tuple[SourceSpanRef, ...], tuple[TraceTextFragment, ...], bool]:
        event_type = request.filters.get("event_type")
        events = [
            item for item in stored.events if event_type is None or item.event_type.value == str(event_type)
        ]
        selected, truncated = _limit(events, request.limit)
        refs = tuple(object_ref_for_event(item) for item in selected)
        spans = _span_refs_for_events(stored, selected)
        fragments = _fragments_for_events(stored, selected)
        return refs, spans, fragments, truncated

    def _event(
        self,
        stored: StoredTraceIndex,
        event_id: str,
    ) -> tuple[tuple[ObjectRef, ...], tuple[SourceSpanRef, ...], tuple[TraceTextFragment, ...], bool]:
        event = next((item for item in stored.events if item.event_id == event_id), None)
        if event is None:
            return (), (), (), False
        return (
            (object_ref_for_event(event),),
            _span_refs_for_events(stored, (event,)),
            _fragments_for_events(stored, (event,)),
            False,
        )

    def _span(
        self,
        stored: StoredTraceIndex,
        span_id: str,
    ) -> tuple[tuple[ObjectRef, ...], tuple[SourceSpanRef, ...], tuple[TraceTextFragment, ...], bool]:
        span = next((item for item in stored.source_spans if item.span_id == span_id), None)
        if span is None:
            return (), (), (), False
        return (object_ref_for_span(span),), (_source_span_ref(span),), (), False

    def _tool_calls(
        self,
        stored: StoredTraceIndex,
        request: TraceQueryRequest,
    ) -> tuple[tuple[ObjectRef, ...], tuple[SourceSpanRef, ...], tuple[TraceTextFragment, ...], bool]:
        family = request.filters.get("tool_family")
        status = request.filters.get("status")
        records = [
            item
            for item in stored.tool_call_records
            if (family is None or item.tool_family.value == str(family))
            and (status is None or item.status.value == str(status))
        ]
        selected, truncated = _limit(records, request.limit)
        return tuple(_tool_ref(item) for item in selected), (), (), truncated

    def _file_observations(
        self,
        stored: StoredTraceIndex,
        request: TraceQueryRequest,
    ) -> tuple[tuple[ObjectRef, ...], tuple[SourceSpanRef, ...], tuple[TraceTextFragment, ...], bool]:
        path = request.filters.get("logical_path")
        operation = request.filters.get("operation")
        records = [
            item
            for item in stored.file_observations
            if (path is None or item.logical_path == str(path))
            and (operation is None or item.operation.value == str(operation))
        ]
        selected, truncated = _limit(records, request.limit)
        return tuple(_file_ref(item) for item in selected), (), (), truncated

    def _segments(
        self,
        stored: StoredTraceIndex,
        request: TraceQueryRequest,
    ) -> tuple[tuple[ObjectRef, ...], tuple[SourceSpanRef, ...], tuple[TraceTextFragment, ...], bool]:
        boundary = request.filters.get("boundary_method")
        records = [
            item
            for item in stored.interaction_segments
            if boundary is None or item.boundary_method == str(boundary)
        ]
        selected, truncated = _limit(records, request.limit)
        return tuple(_segment_ref(item) for item in selected), (), (), truncated

    def _text_search(
        self,
        stored: StoredTraceIndex,
        request: TraceQueryRequest,
        text: str,
    ) -> tuple[tuple[ObjectRef, ...], tuple[SourceSpanRef, ...], tuple[TraceTextFragment, ...], bool]:
        hits = self.store.search_text(stored.trace_ir_version_id, text)
        selected, truncated = _limit(list(hits), request.limit)
        refs = tuple(
            ObjectRef(
                object_type="content-blob",
                object_id=item.object_id,
                object_version=item.media_type,
                object_sha256=item.object_sha256,
            )
            for item in selected
        )
        blobs = {item.content_ref.object_id: item for item in stored.content_blobs}
        fragments = tuple(
            TraceTextFragment(
                subject_ref=ref,
                text=blobs[ref.object_id].canonical_bytes.decode("utf-8", errors="replace"),
            )
            for ref in refs
            if ref.object_id in blobs and blobs[ref.object_id].media_type == "text/plain; charset=utf-8"
        )
        return refs, (), fragments, truncated


def _required_str(request: TraceQueryRequest, key: str) -> str:
    value = request.filters.get(key)
    if not isinstance(value, str) or not value:
        raise QueryValidationError(f"{request.query_kind.value} query requires filter {key}")
    return value


def _limit[T](items: list[T], limit: int) -> tuple[tuple[T, ...], bool]:
    return tuple(items[:limit]), len(items) > limit


def _budget_fragments(
    fragments: tuple[TraceTextFragment, ...],
    max_characters: int,
) -> tuple[tuple[TraceTextFragment, ...], bool]:
    accepted: list[TraceTextFragment] = []
    used = 0
    truncated = False
    for fragment in fragments:
        length = len(fragment.text)
        if used + length > max_characters:
            truncated = True
            continue
        accepted.append(fragment)
        used += length
    return tuple(accepted), truncated


def _span_refs_for_events(
    stored: StoredTraceIndex,
    events: tuple[TraceEvent, ...],
) -> tuple[SourceSpanRef, ...]:
    spans = {item.span_id: item for item in stored.source_spans}
    result: list[SourceSpanRef] = []
    seen: set[str] = set()
    for event in events:
        for ref in event.source_spans:
            span = spans.get(ref.object_id)
            if span is not None and span.span_id not in seen:
                result.append(_source_span_ref(span))
                seen.add(span.span_id)
    return tuple(result)


def _fragments_for_events(
    stored: StoredTraceIndex,
    events: tuple[TraceEvent, ...],
) -> tuple[TraceTextFragment, ...]:
    blobs = {item.content_ref.object_id: item for item in stored.content_blobs}
    fragments: list[TraceTextFragment] = []
    for event in events:
        if event.content_ref is None:
            continue
        blob = blobs.get(event.content_ref.object_id)
        if blob is None or blob.media_type != "text/plain; charset=utf-8":
            continue
        fragments.append(
            TraceTextFragment(
                subject_ref=object_ref_for_event(event),
                text=blob.canonical_bytes.decode("utf-8", errors="replace"),
            )
        )
    return tuple(fragments)


def _source_span_ref(span: SourceSpan) -> SourceSpanRef:
    return SourceSpanRef(
        span_id=span.span_id,
        source_trace_id=span.source_trace_id,
        raw_sha256=span.raw_sha256,
        approximate=span.approximate,
    )


def _fallback_span_ref(stored: StoredTraceIndex) -> SourceSpanRef:
    if not stored.source_spans:
        raise QueryValidationError("stored trace has no source spans for evidence bundle")
    return _source_span_ref(stored.source_spans[0])


def _tool_ref(item: ToolCallRecord) -> ObjectRef:
    return ObjectRef(
        object_type="tool-call-record",
        object_id=item.tool_call_record_id,
        object_version="v1",
        object_sha256=item.canonical_sha256(),
    )


def _file_ref(item: FileObservation) -> ObjectRef:
    return ObjectRef(
        object_type="file-observation",
        object_id=item.observation_id,
        object_version="v1",
        object_sha256=item.canonical_sha256(),
    )


def _segment_ref(item: InteractionSegment) -> ObjectRef:
    return ObjectRef(
        object_type="interaction-segment",
        object_id=item.segment_id,
        object_version="deterministic-interaction-segments/v1",
        object_sha256=item.canonical_sha256(),
    )


def _projection_policy_ref(stage: str, purpose: str) -> ObjectRef:
    seed = {"stage": stage, "purpose": purpose, "policy": "trace-query-r1/v1"}
    return ObjectRef(
        object_type="projection-policy",
        object_id=_stable_id("projection-policy", seed),
        object_version="trace-query-r1/v1",
        object_sha256=_stable_hash(seed),
    )


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}://sha256/{_stable_hash(value)}"


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
