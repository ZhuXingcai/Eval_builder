from __future__ import annotations

from dataclasses import dataclass

INVALID_BACKSLASH_RULE_ID = "json-string-invalid-backslash/v1"
INVALID_BACKSLASH_TRANSFORM = "escape-invalid-backslash"
REPAIR_POLICY_VERSION = "raw-traj-local-repair/v1"

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')


@dataclass(frozen=True)
class RepairEdit:
    source_char_start: int
    source_char_end: int
    repaired_char_start: int
    repaired_char_end: int


@dataclass(frozen=True)
class LocalRepairResult:
    value: str
    edits: tuple[RepairEdit, ...]


def repair_invalid_json_string_backslashes(value: str) -> LocalRepairResult:
    output: list[str] = []
    edits: list[RepairEdit] = []
    in_string = False
    source_index = 0
    repaired_index = 0

    while source_index < len(value):
        char = value[source_index]
        if not in_string:
            output.append(char)
            source_index += 1
            repaired_index += 1
            if char == '"':
                in_string = True
            continue
        if char == '"':
            output.append(char)
            source_index += 1
            repaired_index += 1
            in_string = False
            continue
        if char != "\\":
            output.append(char)
            source_index += 1
            repaired_index += 1
            continue

        next_char = value[source_index + 1 : source_index + 2]
        unicode_escape = (
            next_char == "u"
            and source_index + 6 <= len(value)
            and all(item in _HEX_DIGITS for item in value[source_index + 2 : source_index + 6])
        )
        if next_char in _SIMPLE_ESCAPES or unicode_escape:
            width = 6 if unicode_escape else 2
            output.extend(value[source_index : source_index + width])
            source_index += width
            repaired_index += width
            continue

        output.extend(("\\", "\\"))
        edits.append(
            RepairEdit(
                source_char_start=source_index,
                source_char_end=source_index + 1,
                repaired_char_start=repaired_index,
                repaired_char_end=repaired_index + 2,
            )
        )
        source_index += 1
        repaired_index += 2

    return LocalRepairResult(value="".join(output), edits=tuple(edits))


def repaired_boundary_to_source(
    source: str,
    repair: LocalRepairResult,
) -> tuple[int, ...]:
    boundaries = [0]
    source_index = 0
    for edit in repair.edits:
        while source_index < edit.source_char_start:
            source_index += 1
            boundaries.append(source_index)
        replacement_width = edit.repaired_char_end - edit.repaired_char_start
        for replacement_index in range(replacement_width):
            boundary = (
                edit.source_char_end if replacement_index == replacement_width - 1 else edit.source_char_start
            )
            boundaries.append(boundary)
        source_index = edit.source_char_end
    while source_index < len(source):
        source_index += 1
        boundaries.append(source_index)
    if len(boundaries) != len(repair.value) + 1:
        raise ValueError("repair coordinate map does not match repaired value")
    return tuple(boundaries)
