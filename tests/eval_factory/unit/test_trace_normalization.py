from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.contracts.trace import ToolCallStatus, ToolFamily, TraceEventType
from eval_factory.trace import (
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RawTrajV1Parser,
    TraceSourceRegistry,
)
from eval_factory.trace.normalization import TraceNormalizationResult
from eval_factory.trace.parsing.models import RawTrajParseResult
from eval_factory.trace.parsing.recovery import TraceRecoveryResult


def _outer(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "NormalizationGold",
        "endpoint": "offline",
        "event_time": "2026-07-22T00:00:00Z",
        "extra": '{"req_lost_number":0,"resp_lost_number":0}',
        "mm_urls": "",
        "model": "offline-model",
        "p_date": "2026-07-22",
        "request": '{"messages":[]}',
        "response": '{"content":[],"role":"assistant"}',
        "sid": "synthetic-session",
        "source": None,
    }
    value.update(overrides)
    return value


def _write_outer(path: Path, *records: dict[str, object]) -> bytes:
    raw = (
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
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
    created_at: datetime = datetime(2026, 7, 22, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="trace-normalization-test",
        governing_versions=(VersionBinding(component="raw-traj-event-normalization", version="v1"),),
    )


def _normalize(
    source: Path,
    tmp_path: Path,
    *,
    source_trace_id: str = "source-trace://normalization-test",
    audit: ContractAudit | None = None,
) -> tuple[TraceRecoveryResult, TraceNormalizationResult]:
    active_audit = audit or _audit()
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id=source_trace_id,
        source_uri=f"raw-traj://{source_trace_id.rsplit('/', 1)[-1]}",
    )
    parsed = RawTrajV1Parser().parse(
        source,
        registered_source=registered,
        audit=active_audit,
    )
    recovered = RawTrajRecovery().recover(
        source,
        parse_result=parsed,
        audit=active_audit,
    )
    normalized = RawTrajV1Normalizer().normalize(
        source,
        recovery_result=recovered,
        audit=active_audit,
    )
    return recovered, normalized


def test_normalizes_all_event_types_and_omits_thinking(tmp_path: Path) -> None:
    request = {
        "messages": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": [{"type": "text", "text": "user asks"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "do not expose"},
                    {"type": "text", "text": "assistant text"},
                    {"type": "continuation_marker", "reason": "continued"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                    {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"file_path": "input.txt"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "ok",
                    }
                ],
            },
        ]
    }
    response = {
        "role": "assistant",
        "stop_reason": "max_tokens",
        "content": [{"type": "text", "text": "final", "truncated": True}],
    }
    source = tmp_path / "events.jsonl"
    _write_outer(
        source,
        _outer(request=json.dumps(request), response=json.dumps(response)),
    )

    _, result = _normalize(source, tmp_path)

    counts = Counter(event.event_type for event in result.events)
    assert counts == Counter(
        {
            TraceEventType.SYSTEM_CONTEXT: 1,
            TraceEventType.USER_TEXT: 1,
            TraceEventType.ASSISTANT_TEXT: 2,
            TraceEventType.CONTINUATION_MARKER: 1,
            TraceEventType.ATTACHMENT_REFERENCE: 1,
            TraceEventType.TOOL_CALL: 1,
            TraceEventType.TOOL_RESULT: 1,
        }
    )
    assert "do not expose" not in b"".join(blob.canonical_bytes for blob in result.content_blobs).decode()
    assert any(item.code == "normalization-thinking-omitted" for item in result.diagnostics)
    assert all(event.source_spans for event in result.events)
    assert all(blob.content_ref.object_id.startswith("content://sha256/") for blob in result.content_blobs)
    assert result.tool_call_records[0].status is ToolCallStatus.PAIRED_SUCCESS
    assert result.tool_call_records[0].tool_family is ToolFamily.FILE_READ


def test_exact_tool_family_aliases_and_unknown_tool(tmp_path: Path) -> None:
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "read", "name": "read_file", "input": {}},
                    {"type": "tool_use", "id": "write", "name": "Write", "input": {}},
                    {"type": "tool_use", "id": "edit", "name": "multi_edit", "input": {}},
                    {"type": "tool_use", "id": "list", "name": "Glob", "input": {}},
                    {"type": "tool_use", "id": "search", "name": "WebSearch", "input": {}},
                    {"type": "tool_use", "id": "fetch", "name": "WebFetch", "input": {}},
                    {"type": "tool_use", "id": "shell", "name": "PowerShell", "input": {}},
                    {"type": "tool_use", "id": "user", "name": "AskUserQuestion", "input": {}},
                    {"type": "tool_use", "id": "agent", "name": "Agent", "input": {}},
                    {"type": "tool_use", "id": "task", "name": "TaskUpdate", "input": {}},
                    {"type": "tool_use", "id": "unknown", "name": "MadeUpTool", "input": {}},
                ],
            }
        ]
    }
    source = tmp_path / "families.jsonl"
    _write_outer(source, _outer(request=json.dumps(request)))

    _, result = _normalize(source, tmp_path)

    observed = {record.call_id: record.tool_family for record in result.tool_call_records}
    assert observed == {
        "read": ToolFamily.FILE_READ,
        "write": ToolFamily.FILE_WRITE,
        "edit": ToolFamily.FILE_EDIT,
        "list": ToolFamily.FILE_LIST,
        "search": ToolFamily.SEARCH,
        "fetch": ToolFamily.FETCH,
        "shell": ToolFamily.SHELL,
        "user": ToolFamily.USER_INTERACTION,
        "agent": ToolFamily.SUBAGENT,
        "task": ToolFamily.TASK_MANAGEMENT,
        "unknown": ToolFamily.UNKNOWN,
    }


def test_explicit_id_pairing_orphan_duplicate_ambiguous_and_error(
    tmp_path: Path,
) -> None:
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "ok", "name": "Read", "input": {}},
                    {"type": "tool_use", "id": "err", "name": "Bash", "input": {}},
                    {"type": "tool_use", "id": "orphan", "name": "Write", "input": {}},
                    {"type": "tool_use", "id": "dup", "name": "Read", "input": {}},
                    {"type": "tool_use", "id": "dup", "name": "Read", "input": {}},
                    {"type": "tool_use", "name": "Read", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "ok", "content": "ok"},
                    {"type": "tool_result", "tool_use_id": "err", "content": "boom", "is_error": True},
                    {"type": "tool_result", "tool_use_id": "result-only", "content": "late"},
                    {"type": "tool_result", "tool_use_id": "dup", "content": "duplicate"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "one",
                        "tool_call_id": "two",
                        "content": "ambiguous",
                    },
                    {"type": "tool_result", "content": "missing"},
                ],
            },
        ]
    }
    source = tmp_path / "pairing.jsonl"
    _write_outer(source, _outer(request=json.dumps(request)))

    _, result = _normalize(source, tmp_path)

    statuses = Counter(record.status for record in result.tool_call_records)
    assert statuses[ToolCallStatus.PAIRED_SUCCESS] == 1
    assert statuses[ToolCallStatus.PAIRED_ERROR] == 1
    assert statuses[ToolCallStatus.ORPHAN_CALL] == 2
    assert statuses[ToolCallStatus.ORPHAN_RESULT] == 2
    assert statuses[ToolCallStatus.DUPLICATE_ID] == 3
    assert statuses[ToolCallStatus.AMBIGUOUS] == 1
    assert Counter(event.event_type for event in result.events)[TraceEventType.RUNTIME_ERROR] == 1
    error_record = next(
        record for record in result.tool_call_records if record.status is ToolCallStatus.PAIRED_ERROR
    )
    assert error_record.error_signature
    assert "boom" not in error_record.error_signature


def test_repaired_partial_input_uses_approximate_source_refs(tmp_path: Path) -> None:
    request = '{"messages":[{"role":"user","content":"C:\\qa"}]'
    source = tmp_path / "repaired-partial.jsonl"
    _write_outer(source, _outer(request=request))

    recovered, result = _normalize(source, tmp_path)

    assert recovered.unrecoverable_spans
    assert result.events
    assert result.source_spans[0].approximate is True
    event_ref = result.events[0].source_spans[0]
    assert event_ref.object_id == result.source_spans[0].span_id
    assert result.unrecoverable_spans == recovered.unrecoverable_spans


def test_unparseable_trace_returns_empty_factual_result(tmp_path: Path) -> None:
    source = tmp_path / "unparseable.jsonl"
    _write_outer(
        source,
        _outer(
            request='{"messages":[{"role":"user","content":"broken',
            response='{"content":[{"type":"text","text":"broken',
        ),
    )

    _, result = _normalize(source, tmp_path)

    assert result.events == ()
    assert result.tool_call_records == ()
    assert result.content_blobs == ()
    assert any(item.code == "normalization-unparseable-trace" for item in result.diagnostics)


def test_normalization_ids_ignore_audit_time(tmp_path: Path) -> None:
    request = {"messages": [{"role": "user", "content": "same"}]}
    source = tmp_path / "audit.jsonl"
    _write_outer(source, _outer(request=json.dumps(request)))

    _, first = _normalize(
        source,
        tmp_path,
        source_trace_id="source-trace://audit",
        audit=_audit(datetime(2026, 7, 22, tzinfo=UTC)),
    )
    _, second = _normalize(
        source,
        tmp_path,
        source_trace_id="source-trace://audit-second",
        audit=_audit(datetime(2026, 7, 23, tzinfo=UTC)),
    )

    assert first.trace_ir_version_id != second.trace_ir_version_id

    _, third = _normalize(
        source,
        tmp_path,
        source_trace_id="source-trace://audit",
        audit=_audit(datetime(2026, 7, 23, tzinfo=UTC)),
    )
    assert first.trace_ir_version_id == third.trace_ir_version_id
    assert [event.event_id for event in first.events] == [event.event_id for event in third.events]
    assert [blob.content_ref.object_sha256 for blob in first.content_blobs] == [
        blob.content_ref.object_sha256 for blob in third.content_blobs
    ]


def test_local_corpus_normalization_reconciles_structured_tool_counts(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root.parent / "raw_traj").glob("*.jsonl"))
    if len(paths) != 91:
        pytest.skip("complete local raw_traj corpus is unavailable")
    registry = TraceSourceRegistry(tmp_path / "corpus.sqlite3")
    parser = RawTrajV1Parser()
    recovery = RawTrajRecovery()
    normalizer = RawTrajV1Normalizer()
    tool_call_events = 0
    tool_result_events = 0
    expected_tool_calls = 0
    expected_tool_results = 0
    tool_records = 0
    families: Counter[ToolFamily] = Counter()

    for index, source in enumerate(paths):
        registered = registry.register(
            source,
            source_trace_id=f"source-trace://normalization-corpus-{index:03d}",
            source_uri=f"raw-traj://normalization-corpus-{index:03d}",
        )
        audit = _audit()
        parsed = parser.parse(source, registered_source=registered, audit=audit)
        recovered = recovery.recover(source, parse_result=parsed, audit=audit)
        result = normalizer.normalize(source, recovery_result=recovered, audit=audit)
        expected_tool_calls += _count_structured_blocks(parsed, "tool_use")
        expected_tool_results += _count_structured_blocks(parsed, "tool_result")
        counts = Counter(event.event_type for event in result.events)
        tool_call_events += counts[TraceEventType.TOOL_CALL]
        tool_result_events += counts[TraceEventType.TOOL_RESULT]
        tool_records += len(result.tool_call_records)
        families.update(record.tool_family for record in result.tool_call_records)

    assert tool_call_events == expected_tool_calls
    assert tool_result_events == expected_tool_results
    assert tool_records >= tool_call_events
    assert {
        ToolFamily.FILE_READ,
        ToolFamily.FILE_WRITE,
        ToolFamily.FILE_EDIT,
        ToolFamily.FILE_LIST,
        ToolFamily.SEARCH,
        ToolFamily.FETCH,
        ToolFamily.SHELL,
        ToolFamily.USER_INTERACTION,
        ToolFamily.SUBAGENT,
        ToolFamily.TASK_MANAGEMENT,
        ToolFamily.UNKNOWN,
    } <= set(families)


def _count_structured_blocks(parsed: RawTrajParseResult, block_type: str) -> int:
    count = 0
    for record in parsed.records:
        for field in record.fields:
            if field.value is None:
                continue
            if field.field == "request":
                messages = field.value.get("messages")
                if isinstance(messages, tuple):
                    for message in messages:
                        if isinstance(message, Mapping):
                            content = message.get("content")
                            count += _count_blocks(content, block_type)
            elif field.field == "response":
                count += _count_blocks(field.value.get("content"), block_type)
    return count


def _count_blocks(value: object, block_type: str) -> int:
    if not isinstance(value, tuple):
        return 0
    return sum(1 for item in value if isinstance(item, Mapping) and item.get("type") == block_type)
