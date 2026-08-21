from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

TARGET_FIELDS = frozenset({"request", "response", "extra"})
_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


class OuterCoordinateError(ValueError):
    pass


@dataclass(frozen=True)
class LocatedJsonString:
    value: str
    raw_content_start: int
    raw_content_end: int
    raw_char_boundaries: tuple[int, ...]

    def raw_range_for_char(self, index: int) -> tuple[int, int]:
        if index < 0 or index + 1 >= len(self.raw_char_boundaries):
            raise OuterCoordinateError(f"decoded character index is out of range: {index}")
        return self.raw_char_boundaries[index], self.raw_char_boundaries[index + 1]

    def raw_range_for_chars(self, start: int, end: int) -> tuple[int, int]:
        if start < 0 or end < start or end >= len(self.raw_char_boundaries):
            raise OuterCoordinateError(f"decoded character range is out of bounds: {start}:{end}")
        return self.raw_char_boundaries[start], self.raw_char_boundaries[end]


@dataclass(frozen=True)
class _DecodedString:
    value: str
    end_index: int
    raw_char_boundaries: tuple[int, ...]


def locate_top_level_strings(
    raw_line: bytes,
    *,
    line_byte_start: int,
    expected_outer: dict[str, Any],
    target_fields: frozenset[str] = TARGET_FIELDS,
) -> dict[str, LocatedJsonString]:
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OuterCoordinateError("accepted outer record is not valid UTF-8") from exc

    byte_boundaries = _text_byte_boundaries(text)
    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    index = _skip_whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        raise OuterCoordinateError("accepted outer record is not an object")
    index += 1
    located: dict[str, LocatedJsonString] = {}

    while True:
        index = _skip_whitespace(text, index)
        if index >= len(text):
            raise OuterCoordinateError("outer object ended before its closing delimiter")
        if text[index] == "}":
            index += 1
            break
        key_token = _decode_string_at(text, index, byte_boundaries, line_byte_start)
        key = key_token.value
        index = _skip_whitespace(text, key_token.end_index)
        if index >= len(text) or text[index] != ":":
            raise OuterCoordinateError("outer object key is missing its colon")
        index = _skip_whitespace(text, index + 1)
        value_start = index
        try:
            value, value_end = decoder.raw_decode(text, value_start)
        except (json.JSONDecodeError, ValueError) as exc:
            raise OuterCoordinateError("accepted outer value could not be located") from exc

        if key in target_fields:
            if value_start >= len(text) or text[value_start] != '"':
                raise OuterCoordinateError(f"outer field {key!r} is not a string")
            token = _decode_string_at(text, value_start, byte_boundaries, line_byte_start)
            if token.end_index != value_end or token.value != value:
                raise OuterCoordinateError(f"coordinate decoder disagrees for outer field {key!r}")
            located[key] = LocatedJsonString(
                value=token.value,
                raw_content_start=line_byte_start + byte_boundaries[value_start + 1],
                raw_content_end=line_byte_start + byte_boundaries[value_end - 1],
                raw_char_boundaries=token.raw_char_boundaries,
            )

        index = _skip_whitespace(text, value_end)
        if index >= len(text):
            raise OuterCoordinateError("outer object ended before a separator")
        if text[index] == ",":
            index += 1
            continue
        if text[index] == "}":
            index += 1
            break
        raise OuterCoordinateError("outer object contains an invalid separator")

    if text[index:].strip():
        raise OuterCoordinateError("outer record contains trailing non-whitespace data")
    missing = sorted(target_fields - located.keys())
    if missing:
        raise OuterCoordinateError(f"outer record is missing target fields: {', '.join(missing)}")
    for field in target_fields:
        if located[field].value != expected_outer[field]:
            raise OuterCoordinateError(f"coordinate decoder disagrees with outer parser for {field!r}")
    return located


def _decode_string_at(
    text: str,
    start: int,
    byte_boundaries: tuple[int, ...],
    line_byte_start: int,
) -> _DecodedString:
    if start >= len(text) or text[start] != '"':
        raise OuterCoordinateError("JSON string token must begin with a quote")
    chars: list[str] = []
    boundaries = [line_byte_start + byte_boundaries[start + 1]]
    index = start + 1

    while index < len(text):
        char = text[index]
        if char == '"':
            return _DecodedString(
                value="".join(chars),
                end_index=index + 1,
                raw_char_boundaries=tuple(boundaries),
            )
        if ord(char) < 0x20:
            raise OuterCoordinateError("JSON string contains an unescaped control character")
        if char != "\\":
            chars.append(char)
            index += 1
            boundaries.append(line_byte_start + byte_boundaries[index])
            continue

        escape_start = index
        if index + 1 >= len(text):
            raise OuterCoordinateError("JSON string ends inside an escape")
        escape = text[index + 1]
        if escape in _SIMPLE_ESCAPES:
            chars.append(_SIMPLE_ESCAPES[escape])
            index += 2
            boundaries.append(line_byte_start + byte_boundaries[index])
            continue
        if escape != "u":
            raise OuterCoordinateError(f"JSON string contains invalid escape at index {escape_start}")
        first, index = _decode_unicode_escape(text, index)
        if 0xD800 <= first <= 0xDBFF and text[index : index + 2] == "\\u":
            second, second_end = _decode_unicode_escape(text, index)
            if 0xDC00 <= second <= 0xDFFF:
                codepoint = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                chars.append(chr(codepoint))
                index = second_end
                boundaries.append(line_byte_start + byte_boundaries[index])
                continue
        chars.append(chr(first))
        boundaries.append(line_byte_start + byte_boundaries[index])

    raise OuterCoordinateError("JSON string is missing its closing quote")


def _decode_unicode_escape(text: str, start: int) -> tuple[int, int]:
    end = start + 6
    digits = text[start + 2 : end]
    if len(digits) != 4 or any(char not in _HEX_DIGITS for char in digits):
        raise OuterCoordinateError(f"invalid Unicode escape at index {start}")
    return int(digits, 16), end


def _text_byte_boundaries(text: str) -> tuple[int, ...]:
    boundaries = [0]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
    return tuple(boundaries)


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
