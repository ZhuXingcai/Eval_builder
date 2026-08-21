from __future__ import annotations

from collections.abc import Iterable

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.trace import (
    FileObservation,
    InteractionSegment,
    ToolCallRecord,
    TraceEvent,
    TraceEventType,
)
from eval_factory.trace.indexing.models import SEGMENTATION_POLICY_VERSION, object_ref_for_observation
from eval_factory.trace.normalization.models import object_ref_for_event, stable_id


def build_interaction_segments(
    *,
    events: tuple[TraceEvent, ...],
    tool_call_records: tuple[ToolCallRecord, ...],
    file_observations: tuple[FileObservation, ...],
) -> tuple[InteractionSegment, ...]:
    segments: list[InteractionSegment] = []
    claimed_context_sequences: set[int] = set()
    by_id = {event.event_id: event for event in events}

    for event in events:
        if event.event_type is TraceEventType.USER_TEXT:
            segments.append(_segment("user_turn", (event,)))
            claimed_context_sequences.add(event.sequence)
        elif event.event_type is TraceEventType.CONTINUATION_MARKER:
            segments.append(_segment("continuation", (event,)))
            claimed_context_sequences.add(event.sequence)

    for record in tool_call_records:
        if record.call_event_ref is None or record.result_event_ref is None:
            continue
        members = _events_for_refs(
            (record.call_event_ref, record.result_event_ref),
            by_id,
        )
        if members:
            segments.append(_segment("tool_window", members))
            claimed_context_sequences.update(event.sequence for event in members)

    for observation in file_observations:
        members = _events_for_refs(observation.source_event_refs, by_id)
        if members:
            segments.append(_segment("file_operation", members, observation=observation))
            claimed_context_sequences.update(event.sequence for event in members)

    context_run: list[TraceEvent] = []
    for event in events:
        if event.sequence in claimed_context_sequences:
            if context_run:
                segments.append(_segment("context", tuple(context_run)))
                context_run = []
            continue
        if event.event_type in {
            TraceEventType.ASSISTANT_TEXT,
            TraceEventType.SYSTEM_CONTEXT,
            TraceEventType.RUNTIME_ERROR,
            TraceEventType.ATTACHMENT_REFERENCE,
        }:
            context_run.append(event)
            continue
        if context_run:
            segments.append(_segment("context", tuple(context_run)))
            context_run = []
    if context_run:
        segments.append(_segment("context", tuple(context_run)))

    return tuple(
        sorted(
            _dedupe_segments(segments),
            key=lambda item: (item.sequence_start, item.sequence_end, item.boundary_method, item.segment_id),
        )
    )


def _events_for_refs(
    refs: Iterable[ObjectRef],
    by_id: dict[str, TraceEvent],
) -> tuple[TraceEvent, ...]:
    events: list[TraceEvent] = []
    seen: set[str] = set()
    for ref in refs:
        event = by_id.get(ref.object_id)
        if event is None or event.event_id in seen:
            continue
        events.append(event)
        seen.add(event.event_id)
    return tuple(sorted(events, key=lambda item: item.sequence))


def _segment(
    boundary_method: str,
    events: tuple[TraceEvent, ...],
    *,
    observation: FileObservation | None = None,
) -> InteractionSegment:
    if not events:
        raise ValueError("segment requires at least one event")
    refs = tuple(object_ref_for_event(event) for event in events)
    seed: dict[str, object] = {
        "trace_ir_version_id": events[0].trace_ir_version_id,
        "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
        "boundary_method": boundary_method,
        "member_event_ids": [ref.object_id for ref in refs],
    }
    if observation is not None:
        seed["observation_id"] = observation.observation_id
        seed["observation_ref"] = object_ref_for_observation(observation).model_dump(
            mode="json",
            exclude_none=False,
        )
    return InteractionSegment(
        segment_id=stable_id("interaction-segment", seed),
        trace_ir_version_id=events[0].trace_ir_version_id,
        boundary_method=boundary_method,
        sequence_start=events[0].sequence,
        sequence_end=events[-1].sequence,
        member_event_refs=refs,
    )


def _dedupe_segments(
    segments: list[InteractionSegment],
) -> tuple[InteractionSegment, ...]:
    by_id: dict[str, InteractionSegment] = {}
    for segment in segments:
        by_id.setdefault(segment.segment_id, segment)
    return tuple(by_id.values())
