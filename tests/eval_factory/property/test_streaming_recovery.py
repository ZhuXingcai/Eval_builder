from __future__ import annotations

import json
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from eval_factory.trace.parsing.streaming import (
    scan_request_units,
    scan_response_units,
)

JSON_SCALAR = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=30)
)
JSON_VALUE = st.recursive(
    JSON_SCALAR,
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=20), children, max_size=5)
    ),
    max_leaves=20,
)
JSON_OBJECT = st.dictionaries(st.text(max_size=20), JSON_VALUE, max_size=5)


@given(st.lists(JSON_OBJECT, max_size=12))
def test_request_scanner_emits_only_complete_strict_objects(
    messages: list[dict[str, Any]],
) -> None:
    encoded = ",".join(
        json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for message in messages
    )
    malformed = f'{{"messages":[{encoded}]'

    first = scan_request_units(malformed)
    replay = scan_request_units(malformed)

    assert first == replay
    assert len(first.candidates) == len(messages)
    for candidate in first.candidates:
        decoded = json.loads(malformed[candidate.start : candidate.end])
        assert isinstance(decoded, dict)
    _assert_non_overlapping(first)


@given(st.lists(JSON_OBJECT, max_size=12))
def test_response_scanner_emits_only_complete_strict_objects(
    blocks: list[dict[str, Any]],
) -> None:
    encoded = ",".join(
        json.dumps(
            block,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for block in blocks
    )
    malformed = f'{{"content":[{encoded}]'

    scan = scan_response_units(malformed)

    assert len(scan.candidates) == len(blocks)
    for candidate in scan.candidates:
        decoded = json.loads(malformed[candidate.start : candidate.end])
        assert isinstance(decoded, dict)
    _assert_non_overlapping(scan)


def _assert_non_overlapping(scan) -> None:
    candidates = sorted((item.start, item.end) for item in scan.candidates)
    for index, (start, end) in enumerate(candidates):
        assert start < end
        if index:
            assert candidates[index - 1][1] <= start
        for missing_start, missing_end in scan.unrecoverable_ranges:
            assert end <= missing_start or start >= missing_end
