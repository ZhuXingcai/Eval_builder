from __future__ import annotations

from pathlib import Path
from typing import cast

from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.trace import (
    FileObservation,
    InteractionSegment,
    SourceSpan,
    ToolCallRecord,
    TraceEvent,
)
from eval_factory.trace.indexing import TraceIndexResult
from eval_factory.trace.normalization import NormalizedContentBlob
from eval_factory.trace.normalization.models import MediaType
from eval_factory.trace.storage.cas import ContentAddressedStore
from eval_factory.trace.storage.facts import (
    FACT_LOG_RELATIVE_PATH,
    append_missing_facts,
    batch_sha256,
    build_manifest,
    collect_fact_payloads,
    read_fact_log,
)
from eval_factory.trace.storage.models import (
    StoredContentBlobRecord,
    StoredTraceIndex,
    StoredTraceManifest,
    TraceFactEnvelope,
    TraceFactKind,
    TraceObjectNotFoundError,
    TraceProjectionRebuild,
    TraceStoreCommit,
    TraceStoreCorruptionError,
    TraceTextHit,
)
from eval_factory.trace.storage.projection import TraceProjection


class TraceIndexStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.fact_log_path = self.root / FACT_LOG_RELATIVE_PATH
        self.cas = ContentAddressedStore(self.root / "cas")
        self.projection = TraceProjection(self.root / "projection.sqlite3")

    def persist(
        self,
        result: TraceIndexResult,
        *,
        audit: ContractAudit,
    ) -> TraceStoreCommit:
        del audit
        written_blobs = 0
        existing_blobs = 0
        for blob in result.normalization_result.content_blobs:
            if self.cas.write(blob):
                written_blobs += 1
            else:
                existing_blobs += 1
        existing = read_fact_log(self.fact_log_path)
        payloads = collect_fact_payloads(result)
        appended, existing_count = append_missing_facts(
            self.fact_log_path,
            existing=existing,
            payloads=payloads,
        )
        facts = (*existing, *appended)
        self.projection.rebuild(facts)
        self.projection.populate_fts(
            self._content_for_fts(
                facts,
                materialized=result.normalization_result.content_blobs,
            )
        )
        return TraceStoreCommit(
            trace_ir_version_id=result.normalization_result.trace_ir_version_id,
            batch_sha256=batch_sha256(facts),
            appended_fact_count=len(appended),
            existing_fact_count=existing_count,
            written_blob_count=written_blobs,
            existing_blob_count=existing_blobs,
        )

    def persist_and_materialize(
        self,
        result: TraceIndexResult,
        *,
        audit: ContractAudit,
    ) -> tuple[TraceStoreCommit, StoredTraceIndex]:
        commit = self.persist(result, audit=audit)
        return commit, self.materialize(result)

    @staticmethod
    def materialize(result: TraceIndexResult) -> StoredTraceIndex:
        return _stored_index_from_result(result)

    def load(self, trace_ir_version_id: str) -> StoredTraceIndex:
        facts = read_fact_log(self.fact_log_path)
        selected = tuple(fact for fact in facts if fact.trace_ir_version_id == trace_ir_version_id)
        if not selected:
            raise TraceObjectNotFoundError(f"stored trace index not found: {trace_ir_version_id}")
        self.projection.ensure_consistent(facts)
        return self._load_from_facts(trace_ir_version_id, selected)

    def load_manifest(
        self,
        trace_ir_version_id: str,
    ) -> StoredTraceManifest:
        facts = read_fact_log(self.fact_log_path)
        selected = tuple(
            fact
            for fact in facts
            if fact.trace_ir_version_id == trace_ir_version_id
            and fact.fact_kind is TraceFactKind.TRACE_ENVELOPE
        )
        if not selected:
            raise TraceObjectNotFoundError(f"stored trace manifest not found: {trace_ir_version_id}")
        if len(selected) != 1:
            raise TraceStoreCorruptionError(f"stored trace manifest is ambiguous: {trace_ir_version_id}")
        fact = selected[0]
        manifest = StoredTraceManifest.model_validate_json(fact.payload_json)
        if (
            fact.object_id != trace_ir_version_id
            or manifest.trace_ir_version_id != trace_ir_version_id
            or manifest.canonical_sha256() != fact.object_sha256
        ):
            raise TraceStoreCorruptionError(f"stored trace manifest differs from fact: {trace_ir_version_id}")
        return manifest

    def rebuild_projection(self) -> TraceProjectionRebuild:
        facts = read_fact_log(self.fact_log_path)
        rebuild = self.projection.rebuild(facts)
        self.projection.populate_fts(self._content_for_fts(facts))
        return rebuild

    def search_text(self, trace_ir_version_id: str, query: str) -> tuple[TraceTextHit, ...]:
        return self.projection.search_text(trace_ir_version_id, query)

    def _load_from_facts(
        self,
        trace_ir_version_id: str,
        facts: tuple[TraceFactEnvelope, ...],
    ) -> StoredTraceIndex:
        manifest: StoredTraceManifest | None = None
        source_spans: list[SourceSpan] = []
        events: list[TraceEvent] = []
        tool_records: list[ToolCallRecord] = []
        file_observations: list[FileObservation] = []
        segments: list[InteractionSegment] = []
        content_blobs: list[NormalizedContentBlob] = []

        for fact in sorted(facts, key=lambda item: item.jsonl_sequence):
            if fact.fact_kind is TraceFactKind.TRACE_ENVELOPE:
                manifest = StoredTraceManifest.model_validate_json(fact.payload_json)
            elif fact.fact_kind is TraceFactKind.SOURCE_SPAN:
                source_spans.append(SourceSpan.model_validate_json(fact.payload_json))
            elif fact.fact_kind is TraceFactKind.TRACE_EVENT:
                events.append(TraceEvent.model_validate_json(fact.payload_json))
            elif fact.fact_kind is TraceFactKind.TOOL_CALL_RECORD:
                tool_records.append(ToolCallRecord.model_validate_json(fact.payload_json))
            elif fact.fact_kind is TraceFactKind.FILE_OBSERVATION:
                file_observations.append(FileObservation.model_validate_json(fact.payload_json))
            elif fact.fact_kind is TraceFactKind.INTERACTION_SEGMENT:
                segments.append(InteractionSegment.model_validate_json(fact.payload_json))
            elif fact.fact_kind is TraceFactKind.CONTENT_BLOB:
                record = StoredContentBlobRecord.model_validate_json(fact.payload_json)
                raw = self.cas.read(
                    record.content_ref.object_id,
                    record.content_ref.object_sha256,
                )
                if len(raw) != record.size_bytes:
                    raise TraceStoreCorruptionError(
                        f"CAS blob size mismatch for {record.content_ref.object_id}"
                    )
                content_blobs.append(
                    NormalizedContentBlob(
                        content_ref=record.content_ref,
                        media_type=cast(MediaType, record.media_type),
                        canonical_bytes=raw,
                    )
                )
            elif fact.fact_kind is TraceFactKind.REPAIR_MAP:
                continue
        if manifest is None:
            raise TraceStoreCorruptionError(f"missing stored trace manifest for {trace_ir_version_id}")
        return StoredTraceIndex(
            trace_ir_version_id=trace_ir_version_id,
            manifest=manifest,
            source_spans=tuple(sorted(source_spans, key=lambda item: item.span_id)),
            events=tuple(sorted(events, key=lambda item: item.sequence)),
            tool_call_records=tuple(sorted(tool_records, key=lambda item: item.sequence)),
            file_observations=tuple(sorted(file_observations, key=lambda item: item.sequence)),
            interaction_segments=tuple(
                sorted(
                    segments,
                    key=lambda item: (
                        item.sequence_start,
                        item.sequence_end,
                        item.boundary_method,
                        item.segment_id,
                    ),
                )
            ),
            content_blobs=tuple(sorted(content_blobs, key=lambda item: item.content_ref.object_id)),
        )

    def _content_for_fts(
        self,
        facts: tuple[TraceFactEnvelope, ...],
        *,
        materialized: tuple[NormalizedContentBlob, ...] = (),
    ) -> dict[str, tuple[str, str, bytes]]:
        available = {blob.content_ref.object_id: blob for blob in materialized}
        content: dict[str, tuple[str, str, bytes]] = {}
        for fact in facts:
            if fact.fact_kind is not TraceFactKind.CONTENT_BLOB:
                continue
            record = StoredContentBlobRecord.model_validate_json(fact.payload_json)
            blob = available.get(record.content_ref.object_id)
            if blob is not None:
                if (
                    blob.content_ref != record.content_ref
                    or blob.media_type != record.media_type
                    or len(blob.canonical_bytes) != record.size_bytes
                ):
                    raise TraceStoreCorruptionError(
                        f"materialized content differs for {record.content_ref.object_id}"
                    )
                raw = blob.canonical_bytes
            else:
                raw = self.cas.read(
                    record.content_ref.object_id,
                    record.content_ref.object_sha256,
                )
            content[record.content_ref.object_id] = (
                fact.trace_ir_version_id,
                record.media_type,
                raw,
            )
        return content


def _stored_index_from_result(
    result: TraceIndexResult,
) -> StoredTraceIndex:
    normalization = result.normalization_result
    return StoredTraceIndex(
        trace_ir_version_id=normalization.trace_ir_version_id,
        manifest=build_manifest(result),
        source_spans=tuple(
            sorted(
                normalization.source_spans,
                key=lambda item: item.span_id,
            )
        ),
        events=tuple(
            sorted(
                normalization.events,
                key=lambda item: item.sequence,
            )
        ),
        tool_call_records=tuple(
            sorted(
                normalization.tool_call_records,
                key=lambda item: item.sequence,
            )
        ),
        file_observations=tuple(
            sorted(
                result.file_observations,
                key=lambda item: item.sequence,
            )
        ),
        interaction_segments=tuple(
            sorted(
                result.interaction_segments,
                key=lambda item: (
                    item.sequence_start,
                    item.sequence_end,
                    item.boundary_method,
                    item.segment_id,
                ),
            )
        ),
        content_blobs=tuple(
            sorted(
                normalization.content_blobs,
                key=lambda item: item.content_ref.object_id,
            )
        ),
    )
