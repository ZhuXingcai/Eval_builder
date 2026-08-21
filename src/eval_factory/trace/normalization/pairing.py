from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.trace import (
    ToolCallRecord,
    ToolCallStatus,
    ToolFamily,
    TraceEvent,
)
from eval_factory.trace.normalization.models import object_ref_for_event, stable_id
from eval_factory.trace.normalization.tools import ExplicitId


@dataclass(frozen=True)
class ToolObservation:
    kind: Literal["call", "result"]
    relation_id: ExplicitId
    ambiguous: bool
    is_error: bool
    event: TraceEvent
    original_name: str
    tool_family: ToolFamily
    arguments_ref: ObjectRef | None
    result_ref: ObjectRef | None
    error_signature: str | None


def build_tool_records(
    observations: tuple[ToolObservation, ...],
    *,
    trace_ir_version_id: str,
) -> tuple[ToolCallRecord, ...]:
    records: list[ToolCallRecord] = []
    invalid_or_synthetic: list[ToolObservation] = []
    calls: dict[str, list[ToolObservation]] = defaultdict(list)
    results: dict[str, list[ToolObservation]] = defaultdict(list)
    for item in observations:
        if item.ambiguous or not item.relation_id.valid:
            invalid_or_synthetic.append(item)
            continue
        target = calls if item.kind == "call" else results
        target[item.relation_id.value].append(item)

    for item in sorted(invalid_or_synthetic, key=lambda obs: obs.event.sequence):
        status = (
            ToolCallStatus.AMBIGUOUS
            if item.ambiguous
            else (ToolCallStatus.ORPHAN_CALL if item.kind == "call" else ToolCallStatus.ORPHAN_RESULT)
        )
        records.append(
            _tool_record(
                status=status,
                relation_id=item.relation_id.value,
                item=item,
                trace_ir_version_id=trace_ir_version_id,
            )
        )

    relation_ids = sorted(
        set(calls) | set(results),
        key=lambda value: min(obs.event.sequence for obs in (*calls.get(value, ()), *results.get(value, ()))),
    )
    for relation_id in relation_ids:
        relation_calls = calls.get(relation_id, [])
        relation_results = results.get(relation_id, [])
        if len(relation_calls) > 1 or len(relation_results) > 1:
            for item in sorted((*relation_calls, *relation_results), key=lambda obs: obs.event.sequence):
                records.append(
                    _tool_record(
                        status=ToolCallStatus.DUPLICATE_ID,
                        relation_id=relation_id,
                        item=item,
                        trace_ir_version_id=trace_ir_version_id,
                    )
                )
            continue
        if relation_calls and relation_results:
            call = relation_calls[0]
            result = relation_results[0]
            records.append(
                _tool_record(
                    status=(
                        ToolCallStatus.PAIRED_ERROR if result.is_error else ToolCallStatus.PAIRED_SUCCESS
                    ),
                    relation_id=relation_id,
                    item=call,
                    result=result,
                    trace_ir_version_id=trace_ir_version_id,
                )
            )
            continue
        if relation_calls:
            records.append(
                _tool_record(
                    status=ToolCallStatus.ORPHAN_CALL,
                    relation_id=relation_id,
                    item=relation_calls[0],
                    trace_ir_version_id=trace_ir_version_id,
                )
            )
            continue
        records.append(
            _tool_record(
                status=ToolCallStatus.ORPHAN_RESULT,
                relation_id=relation_id,
                item=relation_results[0],
                trace_ir_version_id=trace_ir_version_id,
            )
        )
    return tuple(sorted(records, key=lambda record: record.sequence))


def _tool_record(
    *,
    status: ToolCallStatus,
    relation_id: str,
    item: ToolObservation,
    trace_ir_version_id: str,
    result: ToolObservation | None = None,
) -> ToolCallRecord:
    call = item if item.kind == "call" else None
    observed_result = result if result is not None else (item if item.kind == "result" else None)
    if call is None and observed_result is None:
        raise ValueError("tool record requires a call or result observation")
    original_name = call.original_name if call is not None else "unknown"
    family = call.tool_family if call is not None else ToolFamily.UNKNOWN
    error_signature = observed_result.error_signature if observed_result is not None else None
    if observed_result is not None and observed_result.is_error and error_signature is None:
        error_signature = _error_signature(family, observed_result.result_ref)
    call_event_ref = object_ref_for_event(call.event) if call is not None else None
    result_event_ref = object_ref_for_event(observed_result.event) if observed_result is not None else None
    record_id = stable_id(
        "tool-call-record",
        {
            "trace_ir_version_id": trace_ir_version_id,
            "status": status.value,
            "relation_id": relation_id,
            "call_event_id": call.event.event_id if call is not None else None,
            "result_event_id": observed_result.event.event_id if observed_result is not None else None,
        },
    )
    if call is not None:
        sequence = call.event.sequence
    else:
        assert observed_result is not None
        sequence = observed_result.event.sequence
    return ToolCallRecord(
        tool_call_record_id=record_id,
        trace_ir_version_id=trace_ir_version_id,
        call_id=relation_id,
        original_name=original_name,
        tool_family=family,
        arguments_ref=call.arguments_ref if call is not None else None,
        call_event_ref=call_event_ref,
        result_event_ref=result_event_ref,
        status=status,
        error_signature=error_signature,
        sequence=sequence,
    )


def _error_signature(
    family: ToolFamily,
    ref: ObjectRef | None,
) -> str:
    digest = ref.object_sha256[:16] if ref is not None else "no-content"
    return f"tool-result-error:{family.value}:{digest}"
