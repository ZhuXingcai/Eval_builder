from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.trace import (
    NestedFieldStatus,
    RawTrajParseOutcome,
    RawTrajV1Parser,
    RegisteredSourceMismatchError,
    TraceSourceRegistry,
    repair_invalid_json_string_backslashes,
)


def _outer(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "ParserGold",
        "endpoint": "offline",
        "event_time": "2026-07-21T00:00:00Z",
        "extra": "{}",
        "mm_urls": "",
        "model": "offline-model",
        "p_date": "2026-07-21",
        "request": '{"messages":[]}',
        "response": '{"choices":[]}',
        "sid": "synthetic-session",
        "source": None,
    }
    value.update(overrides)
    return value


def _write_outer(
    path: Path,
    *records: dict[str, object],
    ensure_ascii: bool = False,
) -> bytes:
    raw = (
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=ensure_ascii,
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        )
        + "\n"
    ).encode()
    path.write_bytes(raw)
    return raw


def _audit(
    created_at: datetime = datetime(2026, 7, 21, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="raw-traj-parser-test",
        governing_versions=(
            VersionBinding(
                component="raw-traj-local-repair",
                version="v1",
            ),
        ),
    )


def _parse(
    source: Path,
    tmp_path: Path,
    *,
    source_trace_id: str = "source-trace://parser-test",
) -> tuple[bytes, Any]:
    before = source.read_bytes()
    registered = TraceSourceRegistry(tmp_path / "registry.sqlite3").register(
        source,
        source_trace_id=source_trace_id,
        source_uri=f"raw-traj://{source_trace_id.rsplit('/', 1)[-1]}",
    )
    result = RawTrajV1Parser().parse(
        source,
        registered_source=registered,
        audit=_audit(),
    )
    return before, result


def test_synthetic_parser_gold_cases(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (root / "evals/golden/eval_factory/parser/v1/cases.json").read_text(encoding="utf-8")
    )

    for index, case in enumerate(payload["cases"]):
        source = tmp_path / f"{case['case_id']}.jsonl"
        raw = _write_outer(
            source,
            _outer(
                request=case["request"],
                response=case["response"],
                extra=case["extra"],
            ),
        )
        before, result = _parse(
            source,
            tmp_path,
            source_trace_id=f"source-trace://gold-{index}",
        )
        request = result.records[0].fields[0]

        assert result.outcome.value == case["expected_outcome"]
        assert (
            len(request.repair_map.segments if request.repair_map else ()) == case["expected_request_repairs"]
        )
        assert before == raw
        assert source.read_bytes() == raw


def test_repair_map_has_absolute_raw_and_decoded_coordinates(tmp_path: Path) -> None:
    nested = '{"messages":[{"content":"prefix 前缀 C:\\qa D:\\work"}]}'
    source = tmp_path / "repair.jsonl"
    raw = _write_outer(source, _outer(request=nested))

    _, first = _parse(source, tmp_path)
    registered = first.registered_source
    replay = RawTrajV1Parser().parse(
        source,
        registered_source=registered,
        audit=_audit(),
    )
    request = first.records[0].fields[0]
    assert request.status is NestedFieldStatus.REPAIRED
    assert request.repair_map is not None
    assert len(request.repair_map.segments) == 2
    assert request.repair_map.semantic_critical is True
    assert request.source_span.approximate is True
    assert request.source_span.repair_map_ref is not None
    request.source_span.repair_map_ref.require_matching_hash(request.repair_map)

    repaired = repair_invalid_json_string_backslashes(nested)
    for segment, edit in zip(request.repair_map.segments, repaired.edits, strict=True):
        assert raw[segment.raw_byte_start : segment.raw_byte_end] == b"\\\\"
        assert repaired.value[segment.decoded_char_start : segment.decoded_char_end] == "\\\\"
        assert segment.decoded_char_start == edit.repaired_char_start
        assert segment.decoded_char_end == edit.repaired_char_end
        assert segment.exact is False

    prefix = b'"request":"'
    expected_start = raw.index(prefix) + len(prefix)
    expected_end = raw.index(b'","response":')
    assert request.source_span.raw_byte_start == expected_start
    assert request.source_span.raw_byte_end == expected_end
    assert request.source_span.decoded_char_start == 0
    assert request.source_span.decoded_char_end == len(repaired.value)
    assert replay == first


def test_audit_time_does_not_participate_in_factual_ids(tmp_path: Path) -> None:
    source = tmp_path / "audit-time.jsonl"
    _write_outer(source, _outer(request='{"path":"C:\\qa"}'))
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id="source-trace://audit-time",
    )
    parser = RawTrajV1Parser()

    first = parser.parse(
        source,
        registered_source=registered,
        audit=_audit(datetime(2026, 7, 21, tzinfo=UTC)),
    )
    second = parser.parse(
        source,
        registered_source=registered,
        audit=_audit(datetime(2026, 7, 22, tzinfo=UTC)),
    )

    first_field = first.records[0].fields[0]
    second_field = second.records[0].fields[0]
    assert first_field.source_span.span_id == second_field.source_span.span_id
    assert first_field.repair_map is not None
    assert second_field.repair_map is not None
    assert first_field.repair_map.repair_map_id == second_field.repair_map.repair_map_id
    assert first_field.repair_map.canonical_sha256() != second_field.repair_map.canonical_sha256()


def test_outer_unicode_escape_coordinate_decoder_matches_standard_json(
    tmp_path: Path,
) -> None:
    nested = json.dumps({"message": "emoji 😀"}, ensure_ascii=False)
    source = tmp_path / "unicode.jsonl"
    _write_outer(source, _outer(request=nested), ensure_ascii=True)

    _, result = _parse(source, tmp_path)

    request = result.records[0].fields[0]
    assert result.outcome is RawTrajParseOutcome.STRICT
    assert request.status is NestedFieldStatus.STRICT
    assert request.value is not None
    assert request.value["message"] == "emoji 😀"


def test_wrong_nested_root_and_unrepairable_structure_need_streaming(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid.jsonl"
    _write_outer(
        source,
        _outer(
            request="[]",
            response='{"choices":[}',
        ),
    )

    _, result = _parse(source, tmp_path)

    request, response, extra = result.records[0].fields
    assert result.outcome is RawTrajParseOutcome.NEEDS_STREAMING
    assert request.status is NestedFieldStatus.NEEDS_STREAMING
    assert request.diagnostics[0].code == "nested-json-root-not-object"
    assert response.status is NestedFieldStatus.NEEDS_STREAMING
    assert response.diagnostics[0].code == "nested-json-invalid"
    assert request.value is None
    assert response.value is None
    assert request.repair_map is None
    assert response.repair_map is None
    assert extra.status is NestedFieldStatus.STRICT


def test_failed_repair_does_not_publish_candidate_map(tmp_path: Path) -> None:
    source = tmp_path / "truncated.jsonl"
    _write_outer(
        source,
        _outer(request='{"messages":[{"content":"C:\\qa"}]'),
    )

    _, result = _parse(source, tmp_path)

    request = result.records[0].fields[0]
    assert request.status is NestedFieldStatus.NEEDS_STREAMING
    assert request.repair_map is None
    assert request.diagnostics[0].code == "nested-json-local-repair-insufficient"
    assert request.source_span.approximate is False


def test_parsed_field_rejects_inconsistent_repair_span_binding(tmp_path: Path) -> None:
    source = tmp_path / "binding.jsonl"
    _write_outer(source, _outer(request='{"path":"C:\\qa"}'))
    _, result = _parse(source, tmp_path)
    request = result.records[0].fields[0]
    assert request.repair_map is not None

    with pytest.raises(ValueError, match="approximate span"):
        replace(
            request,
            source_span=request.source_span.model_copy(update={"repair_map_ref": None, "approximate": False}),
        )


def test_changed_registered_source_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "changed.jsonl"
    _write_outer(source, _outer(sid="before"))
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id="source-trace://changed",
    )
    _write_outer(source, _outer(sid="after"))

    with pytest.raises(RegisteredSourceMismatchError, match="does not match"):
        RawTrajV1Parser().parse(
            source,
            registered_source=registered,
            audit=_audit(),
        )


def test_multiple_records_keep_non_empty_indexes_and_absolute_offsets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multiple.jsonl"
    first = json.dumps(_outer(sid="first"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    second = json.dumps(
        _outer(sid="second", request='{"path":"C:\\qa"}'),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    raw = f"{first}\n\n{second}\n".encode()
    source.write_bytes(raw)

    _, result = _parse(source, tmp_path)

    assert tuple(record.outer_record_index for record in result.records) == (0, 1)
    repaired = result.records[1].fields[0]
    assert repaired.status is NestedFieldStatus.REPAIRED
    assert repaired.repair_map is not None
    segment = repaired.repair_map.segments[0]
    assert segment.raw_byte_start > len(first.encode())
    assert raw[segment.raw_byte_start : segment.raw_byte_end] == b"\\\\"


def test_local_raw_traj_corpus_matches_frozen_parse_counts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    raw_root = root.parent / "raw_traj"
    paths = sorted(raw_root.glob("*.jsonl"))
    if len(paths) != 91:
        pytest.skip("complete local raw_traj corpus is unavailable")
    registry = TraceSourceRegistry(tmp_path / "corpus.sqlite3")
    counts: Counter[tuple[str, NestedFieldStatus]] = Counter()
    parser = RawTrajV1Parser()

    for index, source in enumerate(paths):
        registered = registry.register(
            source,
            source_trace_id=f"source-trace://corpus-{index:03d}",
            source_uri=f"raw-traj://corpus-{index:03d}",
        )
        result = parser.parse(
            source,
            registered_source=registered,
            audit=_audit(),
        )
        for record in result.records:
            counts.update((field.field, field.status) for field in record.fields)

    assert counts == Counter(
        {
            ("request", NestedFieldStatus.STRICT): 19,
            ("request", NestedFieldStatus.REPAIRED): 72,
            ("response", NestedFieldStatus.STRICT): 90,
            ("response", NestedFieldStatus.REPAIRED): 1,
            ("extra", NestedFieldStatus.STRICT): 91,
        }
    )
