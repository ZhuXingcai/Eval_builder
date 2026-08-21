from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from eval_factory.contracts.core import ContractAudit, ObjectRef, TypedAttribute
from eval_factory.contracts.trace import (
    ParseQuality,
    SourceSpan,
    ToolCallRecord,
    ToolCallStatus,
    ToolFamily,
    TraceEvent,
    TraceEventType,
)
from eval_factory.trace.adapters import (
    CuratedTrajectoryV1Adapter,
    RawTrajV1Adapter,
    RuntimeSnapshotV1Adapter,
    TraceAdapter,
)
from eval_factory.trace.normalization.models import (
    DECLARED_SEGMENTATION_POLICY_VERSION,
    NORMALIZATION_POLICY_VERSION,
    TOOL_FAMILY_POLICY_VERSION,
    TRACE_IR_SCHEMA_VERSION,
    JsonValue,
    NormalizationDiagnostic,
    NormalizedContentBlob,
    TraceNormalizationResult,
    canonical_json_bytes,
    content_ref_for_bytes,
    object_ref_for_span,
    stable_id,
    typed_attr,
)
from eval_factory.trace.normalization.pairing import (
    ToolObservation,
    build_tool_records,
)
from eval_factory.trace.normalization.tools import (
    classify_tool_family,
    extract_call_id,
    extract_result_id,
    normalized_tool_name,
)
from eval_factory.trace.parsing.models import FrozenJsonObject
from eval_factory.trace.parsing.raw_traj_v1 import read_registered_source_bytes
from eval_factory.trace.parsing.recovery import (
    RecoveredJsonUnit,
    RecoveryFieldStatus,
    TraceRecoveryResult,
)
from eval_factory.trace.parsing.streaming import RecoveryUnitKind


class RawTrajV1Normalizer:
    normalization_policy_version = NORMALIZATION_POLICY_VERSION
    tool_family_policy_version = TOOL_FAMILY_POLICY_VERSION
    trace_ir_schema_version = TRACE_IR_SCHEMA_VERSION
    declared_segmentation_policy_version = DECLARED_SEGMENTATION_POLICY_VERSION

    def __init__(
        self,
        *,
        adapter: TraceAdapter | None = None,
    ) -> None:
        self.adapter = adapter or RawTrajV1Adapter()

    def normalize(
        self,
        source_path: Path,
        *,
        recovery_result: TraceRecoveryResult,
        audit: ContractAudit,
    ) -> TraceNormalizationResult:
        read_registered_source_bytes(
            source_path,
            registered_source=recovery_result.base_parse_result.registered_source,
            adapter=self.adapter,
        )
        builder = _NormalizationBuilder(
            recovery_result=recovery_result,
            audit=audit,
        )
        return builder.build()


class _NormalizationBuilder:
    def __init__(
        self,
        *,
        recovery_result: TraceRecoveryResult,
        audit: ContractAudit,
    ) -> None:
        self.recovery_result = recovery_result
        self.audit = audit
        self.runtime_snapshot = recovery_result.base_parse_result.registered_source.source.adapter_name in {
            CuratedTrajectoryV1Adapter.name,
            RuntimeSnapshotV1Adapter.name,
        }
        self.trace_ir_version_id = _trace_ir_version_id(recovery_result)
        self.events: list[TraceEvent] = []
        self.tool_observations: list[ToolObservation] = []
        self.content_blobs: dict[str, NormalizedContentBlob] = {}
        self.source_spans: dict[str, SourceSpan] = {}
        self.diagnostics: list[NormalizationDiagnostic] = []

    def build(self) -> TraceNormalizationResult:
        if self.recovery_result.parse_quality is ParseQuality.UNPARSEABLE:
            self.diagnostics.append(
                NormalizationDiagnostic(
                    code="normalization-unparseable-trace",
                    outer_record_index=0,
                    field="trace",
                )
            )
            return self._result(tool_call_records=())

        for record in self.recovery_result.records:
            for recovered_field in record.fields:
                field = recovered_field.original.field
                if field == "extra":
                    continue
                if recovered_field.status is RecoveryFieldStatus.COMPLETE:
                    self._normalize_complete_field(
                        value=recovered_field.original.value,
                        field=field,
                        outer_record_index=record.outer_record_index,
                        span=recovered_field.original.source_span,
                    )
                    continue
                for unit in recovered_field.units:
                    self._normalize_recovered_unit(unit)

        tool_call_records = build_tool_records(
            tuple(self.tool_observations),
            trace_ir_version_id=self.trace_ir_version_id,
        )
        if self.runtime_snapshot:
            tool_call_records = tuple(
                _runtime_snapshot_error_signature(record) for record in tool_call_records
            )
        return self._result(
            tool_call_records=tool_call_records,
        )

    def _result(
        self,
        *,
        tool_call_records: tuple[ToolCallRecord, ...],
    ) -> TraceNormalizationResult:
        return TraceNormalizationResult(
            recovery_result=self.recovery_result,
            trace_ir_version_id=self.trace_ir_version_id,
            trace_ir_schema_version=TRACE_IR_SCHEMA_VERSION,
            normalization_policy_version=NORMALIZATION_POLICY_VERSION,
            tool_family_policy_version=TOOL_FAMILY_POLICY_VERSION,
            declared_segmentation_policy_version=DECLARED_SEGMENTATION_POLICY_VERSION,
            events=tuple(self.events),
            tool_call_records=tool_call_records,
            source_spans=tuple(sorted(self.source_spans.values(), key=lambda item: item.span_id)),
            content_blobs=tuple(
                sorted(self.content_blobs.values(), key=lambda item: item.content_ref.object_id)
            ),
            diagnostics=tuple(self.diagnostics),
            unrecoverable_spans=self.recovery_result.unrecoverable_spans,
            audit=self.audit,
        )

    def _normalize_complete_field(
        self,
        *,
        value: FrozenJsonObject | None,
        field: str,
        outer_record_index: int,
        span: SourceSpan,
    ) -> None:
        if value is None:
            return
        if field == "request":
            messages = value.get("messages")
            if isinstance(messages, tuple):
                for message in messages:
                    if isinstance(message, Mapping):
                        self._normalize_message(
                            message,
                            outer_record_index=outer_record_index,
                            field=field,
                            span=span,
                        )
                    else:
                        self._diagnostic(
                            "normalization-invalid-message",
                            outer_record_index,
                            field,
                            span,
                        )
                return
            items = value.get("input")
            if self.runtime_snapshot and isinstance(items, tuple):
                for item in items:
                    if isinstance(item, Mapping):
                        self._normalize_responses_item(
                            item,
                            outer_record_index=outer_record_index,
                            field=field,
                            span=span,
                        )
                    else:
                        self._diagnostic(
                            "normalization-invalid-responses-item",
                            outer_record_index,
                            field,
                            span,
                        )
                return
            self._diagnostic(
                "normalization-request-messages-missing",
                outer_record_index,
                field,
                span,
            )
            return
        if field == "response":
            role = _string_or_default(value.get("role"), "assistant")
            stop_reason = value.get("stop_reason")
            content = value.get("content")
            if isinstance(content, tuple):
                for block in content:
                    if isinstance(block, Mapping):
                        self._normalize_block(
                            block,
                            role=role,
                            outer_record_index=outer_record_index,
                            field=field,
                            span=span,
                            container_stop_reason=stop_reason if isinstance(stop_reason, str) else None,
                        )
                    else:
                        self._diagnostic(
                            "normalization-invalid-response-block", outer_record_index, field, span
                        )
                return
            if isinstance(content, str):
                self._emit_text(
                    content,
                    role=role,
                    outer_record_index=outer_record_index,
                    field=field,
                    span=span,
                )
                return
            output = value.get("output")
            if self.runtime_snapshot and isinstance(output, tuple):
                for item in output:
                    if isinstance(item, Mapping):
                        self._normalize_responses_item(
                            item,
                            outer_record_index=outer_record_index,
                            field=field,
                            span=span,
                        )
                    else:
                        self._diagnostic(
                            "normalization-invalid-responses-item",
                            outer_record_index,
                            field,
                            span,
                        )
                return
            choices = value.get("choices")
            if self.runtime_snapshot and isinstance(choices, tuple):
                for choice in choices:
                    if not isinstance(choice, Mapping):
                        self._diagnostic(
                            "normalization-invalid-choice",
                            outer_record_index,
                            field,
                            span,
                        )
                        continue
                    message = choice.get("message")
                    if isinstance(message, Mapping):
                        self._normalize_message(
                            message,
                            outer_record_index=outer_record_index,
                            field=field,
                            span=span,
                        )
                    else:
                        self._diagnostic(
                            "normalization-choice-message-missing",
                            outer_record_index,
                            field,
                            span,
                        )
                return
            self._diagnostic("normalization-response-content-missing", outer_record_index, field, span)

    def _normalize_responses_item(
        self,
        item: Mapping[str, JsonValue],
        *,
        outer_record_index: int,
        field: str,
        span: SourceSpan,
    ) -> None:
        item_type = item.get("type")
        if item_type == "message":
            self._normalize_message(
                item,
                outer_record_index=outer_record_index,
                field=field,
                span=span,
            )
            return
        if item_type == "function_call":
            call_id = item.get("call_id")
            if call_id is None:
                call_id = item.get("id")
            call: dict[str, JsonValue] = {
                "type": "tool_use",
                "id": call_id,
                "name": item.get("name"),
                "input": item.get("arguments"),
            }
            self._emit_tool_call(
                call,
                outer_record_index=outer_record_index,
                field=field,
                span=span,
            )
            return
        if item_type == "function_call_output":
            status = item.get("status")
            result: dict[str, JsonValue] = {
                "type": "tool_result",
                "tool_use_id": item.get("call_id"),
                "content": item.get("output"),
                "is_error": status in {"error", "failed"},
            }
            self._emit_tool_result(
                result,
                outer_record_index=outer_record_index,
                field=field,
                span=span,
            )
            return
        if item_type == "reasoning":
            self._diagnostic(
                "normalization-reasoning-omitted",
                outer_record_index,
                field,
                span,
            )
            return
        self._normalize_block(
            item,
            role="assistant" if field == "response" else "unknown",
            outer_record_index=outer_record_index,
            field=field,
            span=span,
            container_stop_reason=None,
        )

    def _normalize_recovered_unit(self, unit: RecoveredJsonUnit) -> None:
        if unit.kind is RecoveryUnitKind.MESSAGE:
            self._normalize_message(
                unit.value,
                outer_record_index=unit.outer_record_index,
                field=unit.field,
                span=unit.source_span,
            )
            return
        role = unit.observed_role or ("assistant" if unit.field == "response" else "unknown")
        self._normalize_block(
            unit.value,
            role=role,
            outer_record_index=unit.outer_record_index,
            field=unit.field,
            span=unit.source_span,
            container_stop_reason=None,
        )

    def _normalize_message(
        self,
        message: Mapping[str, JsonValue],
        *,
        outer_record_index: int,
        field: str,
        span: SourceSpan,
    ) -> None:
        role = message.get("role")
        if not isinstance(role, str):
            self._diagnostic("normalization-message-role-missing", outer_record_index, field, span)
            return
        normalized_role = "system" if self.runtime_snapshot and role == "developer" else role
        content = message.get("content")
        if self.runtime_snapshot and role == "tool":
            result: dict[str, JsonValue] = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id"),
                "content": content,
                "is_error": message.get("status") in {"error", "failed"},
            }
            self._emit_tool_result(
                result,
                outer_record_index=outer_record_index,
                field=field,
                span=span,
            )
            return

        emitted = False
        if isinstance(content, str):
            self._emit_text(
                content,
                role=normalized_role,
                outer_record_index=outer_record_index,
                field=field,
                span=span,
            )
            emitted = True
        elif isinstance(content, tuple):
            for block in content:
                if isinstance(block, Mapping):
                    self._normalize_block(
                        block,
                        role=normalized_role,
                        outer_record_index=outer_record_index,
                        field=field,
                        span=span,
                        container_stop_reason=None,
                    )
                else:
                    self._diagnostic("normalization-invalid-content-block", outer_record_index, field, span)
            emitted = True

        tool_calls = message.get("tool_calls")
        if self.runtime_snapshot and isinstance(tool_calls, tuple):
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    self._diagnostic(
                        "normalization-invalid-tool-call",
                        outer_record_index,
                        field,
                        span,
                    )
                    continue
                function = tool_call.get("function")
                if not isinstance(function, Mapping):
                    self._diagnostic(
                        "normalization-tool-call-function-missing",
                        outer_record_index,
                        field,
                        span,
                    )
                    continue
                call: dict[str, JsonValue] = {
                    "type": "tool_use",
                    "id": tool_call.get("id"),
                    "name": function.get("name"),
                    "input": function.get("arguments"),
                }
                self._emit_tool_call(
                    call,
                    outer_record_index=outer_record_index,
                    field=field,
                    span=span,
                )
            emitted = True

        if not emitted:
            self._diagnostic(
                "normalization-message-content-missing",
                outer_record_index,
                field,
                span,
            )

    def _normalize_block(
        self,
        block: Mapping[str, JsonValue],
        *,
        role: str,
        outer_record_index: int,
        field: str,
        span: SourceSpan,
        container_stop_reason: str | None,
    ) -> None:
        block_type = block.get("type")
        if not isinstance(block_type, str):
            self._diagnostic("normalization-block-type-missing", outer_record_index, field, span)
            return
        if self.runtime_snapshot and block_type == "":
            recovered_type = _recover_empty_block_type(block)
            if recovered_type is not None:
                block_type = recovered_type
                self._diagnostic(
                    "normalization-empty-block-type-recovered",
                    outer_record_index,
                    field,
                    span,
                    attributes=(typed_attr("recovered_type", recovered_type),),
                )
        if block_type == "thinking" or (
            self.runtime_snapshot and block_type in {"reasoning", "summary_text"}
        ):
            self._diagnostic("normalization-thinking-omitted", outer_record_index, field, span)
            return
        if block_type == "text" or (self.runtime_snapshot and block_type in {"input_text", "output_text"}):
            text = block.get("text")
            if isinstance(text, str):
                self._emit_text(
                    text,
                    role=role,
                    outer_record_index=outer_record_index,
                    field=field,
                    span=span,
                    extra_attributes=_truncation_attributes(block, container_stop_reason),
                )
                return
            self._diagnostic("normalization-text-missing", outer_record_index, field, span)
            return
        if block_type == "tool_use":
            self._emit_tool_call(
                block,
                outer_record_index=outer_record_index,
                field=field,
                span=span,
            )
            return
        if block_type == "tool_result":
            self._emit_tool_result(
                block,
                outer_record_index=outer_record_index,
                field=field,
                span=span,
            )
            return
        if self.runtime_snapshot and block_type in {"function_call", "function_call_output", "message"}:
            self._normalize_responses_item(
                block,
                outer_record_index=outer_record_index,
                field=field,
                span=span,
            )
            return
        if block_type in {"continuation", "continuation_marker"}:
            content_ref = self._json_blob(block)
            self._add_event(
                TraceEventType.CONTINUATION_MARKER,
                role=_event_role(role),
                content_ref=content_ref,
                span=span,
                attributes=(typed_attr("raw_type", block_type),),
            )
            return
        if block_type in {
            "attachment",
            "document",
            "file",
            "image",
        } or (self.runtime_snapshot and block_type == "image_url"):
            content_ref = self._json_blob(block)
            self._add_event(
                TraceEventType.ATTACHMENT_REFERENCE,
                role=_event_role(role),
                content_ref=content_ref,
                span=span,
                attributes=(typed_attr("raw_type", block_type),),
            )
            return
        self._diagnostic(
            "normalization-unknown-block-type",
            outer_record_index,
            field,
            span,
            attributes=(typed_attr("raw_type", block_type),),
        )

    def _emit_text(
        self,
        value: str,
        *,
        role: str,
        outer_record_index: int,
        field: str,
        span: SourceSpan,
        extra_attributes: tuple[TypedAttribute, ...] = (),
    ) -> None:
        event_type = _text_event_type(role)
        if event_type is None:
            self._diagnostic(
                "normalization-unknown-text-role",
                outer_record_index,
                field,
                span,
                attributes=(typed_attr("role", role),),
            )
            return
        self._add_event(
            event_type,
            role=_event_role(role),
            content_ref=self._text_blob(value),
            span=span,
            attributes=extra_attributes,
        )

    def _emit_tool_call(
        self,
        block: Mapping[str, JsonValue],
        *,
        outer_record_index: int,
        field: str,
        span: SourceSpan,
    ) -> None:
        name = normalized_tool_name(block.get("name"))
        family = classify_tool_family(name)
        explicit = extract_call_id(
            block.get("id"),
            fallback_seed=_fallback_seed(
                trace_ir_version_id=self.trace_ir_version_id,
                kind="call",
                sequence=len(self.events),
                span=span,
            ),
        )
        arguments_ref = self._json_blob(block.get("input") if "input" in block else {})
        attributes = [
            typed_attr("raw_type", "tool_use"),
            typed_attr("tool_name", name),
            typed_attr("tool_family", family.value),
            typed_attr("call_id", explicit.value),
        ]
        if explicit.synthetic:
            attributes.append(typed_attr("synthetic_id", True))
            assert explicit.reason is not None
            self._diagnostic(
                explicit.reason, outer_record_index, field, span, attributes=(typed_attr("kind", "call"),)
            )
        event = self._add_event(
            TraceEventType.TOOL_CALL,
            role="assistant",
            content_ref=arguments_ref,
            span=span,
            attributes=tuple(attributes),
        )
        self.tool_observations.append(
            ToolObservation(
                kind="call",
                relation_id=explicit,
                ambiguous=False,
                is_error=False,
                event=event,
                original_name=name,
                tool_family=family,
                arguments_ref=arguments_ref,
                result_ref=None,
                error_signature=None,
            )
        )

    def _emit_tool_result(
        self,
        block: Mapping[str, JsonValue],
        *,
        outer_record_index: int,
        field: str,
        span: SourceSpan,
    ) -> None:
        result_id = extract_result_id(
            tool_use_id=block.get("tool_use_id"),
            tool_call_id=block.get("tool_call_id"),
            fallback_seed=_fallback_seed(
                trace_ir_version_id=self.trace_ir_version_id,
                kind="result",
                sequence=len(self.events),
                span=span,
            ),
        )
        content_ref = self._content_value_blob(block.get("content") if "content" in block else block)
        is_error = block.get("is_error") is True
        attributes = [
            typed_attr("raw_type", "tool_result"),
            typed_attr("call_id", result_id.relation_id.value),
            typed_attr("is_error", is_error),
        ]
        if result_id.relation_id.synthetic:
            attributes.append(typed_attr("synthetic_id", True))
            assert result_id.relation_id.reason is not None
            self._diagnostic(
                result_id.relation_id.reason,
                outer_record_index,
                field,
                span,
                attributes=(typed_attr("kind", "result"),),
            )
        if result_id.ambiguous:
            self._diagnostic(
                "conflicting-result-id-fields",
                outer_record_index,
                field,
                span,
                attributes=(typed_attr("kind", "result"),),
            )
        event = self._add_event(
            TraceEventType.TOOL_RESULT,
            role="tool",
            content_ref=content_ref,
            span=span,
            attributes=tuple(attributes),
        )
        error_signature = None
        if is_error:
            error_signature = _error_signature(ToolFamily.UNKNOWN, content_ref)
            self._add_event(
                TraceEventType.RUNTIME_ERROR,
                role="runtime",
                content_ref=content_ref,
                span=span,
                attributes=(
                    typed_attr("raw_type", "tool_result"),
                    typed_attr("error_signature", error_signature),
                ),
            )
        self.tool_observations.append(
            ToolObservation(
                kind="result",
                relation_id=result_id.relation_id,
                ambiguous=result_id.ambiguous,
                is_error=is_error,
                event=event,
                original_name="unknown",
                tool_family=ToolFamily.UNKNOWN,
                arguments_ref=None,
                result_ref=content_ref,
                error_signature=error_signature,
            )
        )

    def _add_event(
        self,
        event_type: TraceEventType,
        *,
        role: Literal["user", "assistant", "system", "tool", "runtime"] | None,
        content_ref: ObjectRef | None,
        span: SourceSpan,
        attributes: tuple[TypedAttribute, ...] = (),
    ) -> TraceEvent:
        self.source_spans.setdefault(span.span_id, span)
        sequence = len(self.events)
        span_ref = object_ref_for_span(span)
        event_id = stable_id(
            "trace-event",
            {
                "trace_ir_version_id": self.trace_ir_version_id,
                "sequence": sequence,
                "event_type": event_type.value,
                "source_span_ids": [span.span_id],
            },
        )
        event = TraceEvent(
            event_id=event_id,
            trace_ir_version_id=self.trace_ir_version_id,
            sequence=sequence,
            event_type=event_type,
            role=role,
            content_ref=content_ref,
            source_spans=(span_ref,),
            attributes=attributes,
        )
        self.events.append(event)
        return event

    def _text_blob(self, value: str) -> ObjectRef:
        encoded = value.encode()
        return self._blob(encoded, media_type="text/plain; charset=utf-8")

    def _json_blob(self, value: JsonValue) -> ObjectRef:
        return self._blob(canonical_json_bytes(value), media_type="application/json")

    def _content_value_blob(self, value: JsonValue) -> ObjectRef:
        if isinstance(value, str):
            return self._text_blob(value)
        return self._json_blob(value)

    def _blob(
        self,
        value: bytes,
        *,
        media_type: Literal["application/json", "text/plain; charset=utf-8"],
    ) -> ObjectRef:
        ref = content_ref_for_bytes(value, media_type=media_type)
        self.content_blobs.setdefault(
            ref.object_id,
            NormalizedContentBlob(
                content_ref=ref,
                media_type=media_type,
                canonical_bytes=value,
            ),
        )
        return ref

    def _diagnostic(
        self,
        code: str,
        outer_record_index: int,
        field: str,
        span: SourceSpan,
        *,
        attributes: tuple[TypedAttribute, ...] = (),
    ) -> None:
        self.diagnostics.append(
            NormalizationDiagnostic(
                code=code,
                outer_record_index=outer_record_index,
                field=field,
                source_span_ref=object_ref_for_span(span),
                attributes=attributes,
            )
        )


def _trace_ir_version_id(
    recovery_result: TraceRecoveryResult,
) -> str:
    registered = recovery_result.base_parse_result.registered_source
    source = registered.source
    return stable_id(
        "trace-ir-version",
        {
            "source_trace_id": source.source_trace_id,
            "raw_sha256": source.raw_sha256,
            "adapter_name": source.adapter_name,
            "adapter_version": source.adapter_version,
            "trace_ir_schema_version": TRACE_IR_SCHEMA_VERSION,
            "repair_policy_version": recovery_result.base_parse_result.repair_policy_version,
            "recovery_policy_version": recovery_result.recovery_policy_version,
            "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
            "tool_family_policy_version": TOOL_FAMILY_POLICY_VERSION,
            "segmentation_policy_version": DECLARED_SEGMENTATION_POLICY_VERSION,
        },
    )


def _text_event_type(role: str) -> TraceEventType | None:
    if role == "user":
        return TraceEventType.USER_TEXT
    if role == "assistant":
        return TraceEventType.ASSISTANT_TEXT
    if role == "system":
        return TraceEventType.SYSTEM_CONTEXT
    return None


def _event_role(
    role: str,
) -> Literal["user", "assistant", "system", "tool", "runtime"] | None:
    if role in {"user", "assistant", "system", "tool", "runtime"}:
        return role  # type: ignore[return-value]
    return None


def _recover_empty_block_type(
    block: Mapping[str, JsonValue],
) -> str | None:
    keys = set(block)
    if (
        {"id", "input", "name", "type"}.issubset(keys)
        and isinstance(block.get("id"), str)
        and isinstance(block.get("name"), str)
        and isinstance(block.get("input"), Mapping)
    ):
        return "tool_use"
    if keys <= {"text", "type"} and isinstance(block.get("text"), str):
        return "text"
    if keys <= {"signature", "thinking", "type"} and isinstance(block.get("thinking"), str):
        return "thinking"
    return None


def _string_or_default(value: JsonValue | None, default: str) -> str:
    return value if isinstance(value, str) else default


def _fallback_seed(
    *,
    trace_ir_version_id: str,
    kind: str,
    sequence: int,
    span: SourceSpan,
) -> dict[str, object]:
    return {
        "trace_ir_version_id": trace_ir_version_id,
        "kind": kind,
        "sequence": sequence,
        "span_id": span.span_id,
    }


def _truncation_attributes(
    block: Mapping[str, JsonValue],
    stop_reason: str | None,
) -> tuple[TypedAttribute, ...]:
    attributes: list[TypedAttribute] = []
    for key in ("truncated", "is_truncated"):
        value = block.get(key)
        if isinstance(value, bool):
            attributes.append(typed_attr(key, value))
    if stop_reason is not None:
        attributes.append(typed_attr("stop_reason", stop_reason))
        if stop_reason == "max_tokens":
            attributes.append(typed_attr("truncated", True))
    return tuple(attributes)


def _runtime_snapshot_error_signature(
    record: ToolCallRecord,
) -> ToolCallRecord:
    if (
        record.status is not ToolCallStatus.PAIRED_ERROR
        or record.original_name.strip().casefold() != "powershell"
    ):
        return record
    digest = (
        record.result_event_ref.object_sha256[:16] if record.result_event_ref is not None else "no-result"
    )
    return record.model_copy(
        update={
            "error_signature": f"powershell:{digest}:paired_error",
        }
    )


def _error_signature(
    family: ToolFamily,
    ref: ObjectRef | None,
) -> str:
    digest = ref.object_sha256[:16] if ref is not None else "no-content"
    return f"tool-result-error:{family.value}:{digest}"
