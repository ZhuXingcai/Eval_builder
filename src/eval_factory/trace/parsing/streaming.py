from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum


class RecoveryUnitKind(StrEnum):
    MESSAGE = "MESSAGE"
    CONTENT_BLOCK = "CONTENT_BLOCK"


@dataclass(frozen=True)
class RecoveryCandidate:
    kind: RecoveryUnitKind
    start: int
    end: int
    observed_role: str | None = None


@dataclass(frozen=True)
class StreamingScan:
    candidates: tuple[RecoveryCandidate, ...]
    unrecoverable_ranges: tuple[tuple[int, int], ...]
    target_array_found: bool
    target_array_complete: bool


@dataclass(frozen=True)
class _ArrayScan:
    object_ranges: tuple[tuple[int, int], ...]
    invalid_ranges: tuple[tuple[int, int], ...]
    incomplete_object_start: int | None
    end_index: int | None


def scan_request_units(value: str) -> StreamingScan:
    messages_start = _find_object_member(value, 0, "messages")
    if messages_start is None or messages_start >= len(value) or value[messages_start] != "[":
        return StreamingScan(
            candidates=(),
            unrecoverable_ranges=((0, len(value)),) if value else (),
            target_array_found=False,
            target_array_complete=False,
        )

    messages = _scan_object_array(value, messages_start)
    candidates = [
        RecoveryCandidate(
            kind=RecoveryUnitKind.MESSAGE,
            start=start,
            end=end,
        )
        for start, end in messages.object_ranges
    ]
    invalid_ranges = list(messages.invalid_ranges)

    if messages.incomplete_object_start is not None:
        message_start = messages.incomplete_object_start
        role = _decode_string_member(value, message_start, "role")
        content_start = _find_object_member(value, message_start, "content")
        if content_start is not None and content_start < len(value) and value[content_start] == "[":
            content = _scan_object_array(value, content_start)
            candidates.extend(
                RecoveryCandidate(
                    kind=RecoveryUnitKind.CONTENT_BLOCK,
                    start=start,
                    end=end,
                    observed_role=role,
                )
                for start, end in content.object_ranges
            )
            invalid_ranges.extend(content.invalid_ranges)

    if messages.end_index is not None and messages.end_index < len(value):
        invalid_ranges.append((messages.end_index, len(value)))
    normalized = _subtract_candidates(
        invalid_ranges,
        tuple((candidate.start, candidate.end) for candidate in candidates),
    )
    return StreamingScan(
        candidates=tuple(sorted(candidates, key=lambda item: (item.start, item.end, item.kind))),
        unrecoverable_ranges=normalized,
        target_array_found=True,
        target_array_complete=messages.end_index is not None,
    )


def scan_response_units(value: str) -> StreamingScan:
    content_start = _find_object_member(value, 0, "content")
    if content_start is None or content_start >= len(value) or value[content_start] != "[":
        return StreamingScan(
            candidates=(),
            unrecoverable_ranges=((0, len(value)),) if value else (),
            target_array_found=False,
            target_array_complete=False,
        )
    content = _scan_object_array(value, content_start)
    candidates = tuple(
        RecoveryCandidate(
            kind=RecoveryUnitKind.CONTENT_BLOCK,
            start=start,
            end=end,
            observed_role="assistant",
        )
        for start, end in content.object_ranges
    )
    invalid_ranges = list(content.invalid_ranges)
    if content.end_index is not None and content.end_index < len(value):
        invalid_ranges.append((content.end_index, len(value)))
    return StreamingScan(
        candidates=candidates,
        unrecoverable_ranges=_subtract_candidates(
            invalid_ranges,
            tuple((candidate.start, candidate.end) for candidate in candidates),
        ),
        target_array_found=True,
        target_array_complete=content.end_index is not None,
    )


def _scan_object_array(value: str, start: int) -> _ArrayScan:
    cursor = start + 1
    objects: list[tuple[int, int]] = []
    invalid: list[tuple[int, int]] = []

    while cursor < len(value):
        cursor = _skip_whitespace(value, cursor)
        while cursor < len(value) and value[cursor] == ",":
            cursor = _skip_whitespace(value, cursor + 1)
        if cursor >= len(value):
            return _ArrayScan(tuple(objects), tuple(invalid), None, None)
        if value[cursor] == "]":
            return _ArrayScan(tuple(objects), tuple(invalid), None, cursor + 1)
        if value[cursor] == "{":
            end = _balanced_end(value, cursor)
            if end is None:
                invalid.append((cursor, len(value)))
                return _ArrayScan(tuple(objects), tuple(invalid), cursor, None)
            objects.append((cursor, end))
            cursor = end
            continue

        scalar_end = _skip_value(value, cursor)
        if scalar_end is None:
            invalid.append((cursor, len(value)))
            return _ArrayScan(tuple(objects), tuple(invalid), None, None)
        invalid.append((cursor, scalar_end))
        cursor = scalar_end

    return _ArrayScan(tuple(objects), tuple(invalid), None, None)


def _find_object_member(
    value: str,
    object_start: int,
    member: str,
) -> int | None:
    if object_start >= len(value) or value[object_start] != "{":
        return None
    cursor = object_start + 1
    while cursor < len(value):
        cursor = _skip_whitespace(value, cursor)
        while cursor < len(value) and value[cursor] == ",":
            cursor = _skip_whitespace(value, cursor + 1)
        if cursor >= len(value) or value[cursor] == "}":
            return None
        key_end = _string_end(value, cursor)
        if key_end is None:
            return None
        try:
            key = json.loads(value[cursor:key_end])
        except json.JSONDecodeError:
            return None
        if not isinstance(key, str):
            return None
        colon = _skip_whitespace(value, key_end)
        if colon >= len(value) or value[colon] != ":":
            return None
        member_start = _skip_whitespace(value, colon + 1)
        if key == member:
            return member_start
        member_end = _skip_value(value, member_start)
        if member_end is None:
            return None
        cursor = member_end
    return None


def _decode_string_member(
    value: str,
    object_start: int,
    member: str,
) -> str | None:
    start = _find_object_member(value, object_start, member)
    if start is None:
        return None
    end = _string_end(value, start)
    if end is None:
        return None
    try:
        decoded = json.loads(value[start:end])
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, str) else None


def _skip_value(value: str, start: int) -> int | None:
    if start >= len(value):
        return None
    char = value[start]
    if char == '"':
        return _string_end(value, start)
    if char in "[{":
        return _balanced_end(value, start)
    cursor = start
    while cursor < len(value) and value[cursor] not in ",]} \t\r\n":
        cursor += 1
    return cursor if cursor > start else None


def _balanced_end(value: str, start: int) -> int | None:
    opening = value[start]
    if opening not in "[{":
        return None
    stack = ["]" if opening == "[" else "}"]
    cursor = start + 1
    in_string = False

    while cursor < len(value):
        char = value[cursor]
        if in_string:
            if char == "\\":
                cursor += 2
                continue
            if char == '"':
                in_string = False
            cursor += 1
            continue
        if char == '"':
            in_string = True
            cursor += 1
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return cursor + 1
        cursor += 1
    return None


def _string_end(value: str, start: int) -> int | None:
    if start >= len(value) or value[start] != '"':
        return None
    cursor = start + 1
    while cursor < len(value):
        char = value[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == '"':
            return cursor + 1
        cursor += 1
    return None


def _skip_whitespace(value: str, cursor: int) -> int:
    while cursor < len(value) and value[cursor] in " \t\r\n":
        cursor += 1
    return cursor


def _subtract_candidates(
    ranges: list[tuple[int, int]],
    candidates: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    pending = [(max(0, start), max(0, end)) for start, end in ranges if end > start]
    for candidate_start, candidate_end in sorted(candidates):
        next_pending: list[tuple[int, int]] = []
        for start, end in pending:
            if candidate_end <= start or candidate_start >= end:
                next_pending.append((start, end))
                continue
            if start < candidate_start:
                next_pending.append((start, candidate_start))
            if candidate_end < end:
                next_pending.append((candidate_end, end))
        pending = next_pending
    return _merge_ranges(pending)


def _merge_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)
