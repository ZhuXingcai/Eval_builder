from __future__ import annotations

import json
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from eval_factory.trace import repair_invalid_json_string_backslashes

JSON_SCALAR = (
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text()
)
JSON_VALUE = st.recursive(
    JSON_SCALAR,
    lambda children: (
        st.lists(children, max_size=8) | st.dictionaries(st.text(max_size=30), children, max_size=8)
    ),
    max_leaves=30,
)
JSON_OBJECT = st.dictionaries(st.text(max_size=30), JSON_VALUE, max_size=8)


@given(JSON_OBJECT)
def test_valid_standard_json_is_never_changed(value: dict[str, Any]) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    repaired = repair_invalid_json_string_backslashes(encoded)

    assert repaired.value == encoded
    assert repaired.edits == ()


@given(st.text(max_size=100))
def test_successful_invalid_backslash_repair_is_idempotent(suffix: str) -> None:
    encoded_suffix = json.dumps(suffix, ensure_ascii=False)[1:-1]
    malformed = f'{{"path":"C:\\q{encoded_suffix}"}}'

    first = repair_invalid_json_string_backslashes(malformed)
    second = repair_invalid_json_string_backslashes(first.value)

    assert len(first.edits) == 1
    assert json.loads(first.value)["path"].startswith("C:\\q")
    assert second.value == first.value
    assert second.edits == ()
