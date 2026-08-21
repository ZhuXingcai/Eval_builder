from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.contracts.trace import (
    CapabilityStatus,
    ParseQuality,
    ToolCallStatus,
    TraceEventType,
)
from eval_factory.orchestration.runner import TraceIndexStageService
from eval_factory.trace import (
    RawTrajParseResult,
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RawTrajV1Parser,
    RegisteredSourceMismatchError,
    RegisteredTraceSource,
    RuntimeSnapshotV1Adapter,
    RuntimeSnapshotV1Parser,
    TraceIndexStore,
    TraceNormalizationResult,
    TraceRecoveryResult,
    TraceSourceRegistry,
)


def _audit(
    created_at: datetime = datetime(2026, 8, 9, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="runtime-snapshot-ingestion-test",
        governing_versions=(
            VersionBinding(
                component="runtime-snapshot-adapter",
                version="v1",
            ),
        ),
    )


def _write_snapshot(path: Path) -> bytes:
    request = {
        "messages": [
            {"role": "user", "content": "inspect the repository"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "Read",
                        "input": {"file_path": "README.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "repository overview",
                    }
                ],
            },
        ]
    }
    response = {
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
    }
    outer = {
        "sid": "snapshot-session",
        "event_time": "2026-07-17 00:00:08",
        "api_type": "Message",
        "business": "CodingPlan",
        "real_model": "model-a",
        "request_model": "model-request-a",
        "request": json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "response": json.dumps(
            response,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    raw = json.dumps(
        outer,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    path.write_bytes(raw)
    return raw


def _write_runtime_snapshot(
    path: Path,
    *,
    api_type: str,
    request: dict[str, object],
    response: dict[str, object],
) -> bytes:
    outer = {
        "sid": "snapshot-session",
        "event_time": "2026-07-17 00:00:08",
        "api_type": api_type,
        "business": "CodingPlan",
        "real_model": "model-a",
        "request_model": "model-request-a",
        "request": json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "response": json.dumps(
            response,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    raw = json.dumps(
        outer,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    path.write_bytes(raw)
    return raw


def _ingest(
    source: Path,
    tmp_path: Path,
) -> tuple[
    RegisteredTraceSource,
    RawTrajParseResult,
    TraceRecoveryResult,
    TraceNormalizationResult,
]:
    adapter = RuntimeSnapshotV1Adapter()
    registered = TraceSourceRegistry(
        tmp_path / "registry.sqlite3",
        adapter=adapter,
    ).register(
        source,
        source_trace_id="source-trace://runtime-snapshot/example",
        source_uri="runtime-snapshot://claude-code/example",
    )
    parsed = RuntimeSnapshotV1Parser().parse(
        source,
        registered_source=registered,
        audit=_audit(),
    )
    recovered = RawTrajRecovery(adapter=adapter).recover(
        source,
        parse_result=parsed,
        audit=_audit(),
    )
    normalized = RawTrajV1Normalizer(adapter=adapter).normalize(
        source,
        recovery_result=recovered,
        audit=_audit(),
    )
    return registered, parsed, recovered, normalized


def test_snapshot_parse_recovery_and_normalization_use_original_source_spans(
    tmp_path: Path,
) -> None:
    source = tmp_path / "snapshot.json"
    raw = _write_snapshot(source)

    registered, parsed, recovered, normalized = _ingest(
        source,
        tmp_path,
    )

    assert registered.source.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert parsed.outcome.value == "STRICT"
    assert tuple(field.field for field in parsed.records[0].fields) == (
        "request",
        "response",
    )
    assert all(
        field.source_span.raw_sha256 == registered.source.raw_sha256 for field in parsed.records[0].fields
    )
    assert all(
        raw[field.source_span.raw_byte_start : field.source_span.raw_byte_end]
        for field in parsed.records[0].fields
    )
    assert recovered.parse_quality is ParseQuality.STRICT
    assert {value.status for value in recovered.capabilities} == {CapabilityStatus.COMPLETE}
    assert {value.event_type for value in normalized.events} >= {
        TraceEventType.USER_TEXT,
        TraceEventType.ASSISTANT_TEXT,
        TraceEventType.TOOL_CALL,
        TraceEventType.TOOL_RESULT,
    }
    assert normalized.tool_call_records
    assert source.read_bytes() == raw


def test_snapshot_trace_ir_identity_binds_adapter_and_original_hash(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.json"
    first_raw = _write_snapshot(first_source)
    _, _, _, first = _ingest(first_source, tmp_path / "first")

    second_source = tmp_path / "second.json"
    second_raw = _write_snapshot(second_source)
    _, _, _, second = _ingest(second_source, tmp_path / "second")

    assert first_raw == second_raw
    assert first.trace_ir_version_id == second.trace_ir_version_id
    assert first.recovery_result.base_parse_result.registered_source.source.adapter_name == (
        "runtime_snapshot_v1"
    )


def test_snapshot_powershell_paired_error_gets_approved_signature(
    tmp_path: Path,
) -> None:
    source = tmp_path / "powershell.json"
    _write_runtime_snapshot(
        source,
        api_type="Message",
        request={
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "powershell-1",
                            "name": "PowerShell",
                            "input": {"command": "Get-Item missing"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "powershell-1",
                            "is_error": True,
                            "content": "structured failure",
                        }
                    ],
                },
            ]
        },
        response={
            "role": "assistant",
            "content": [{"type": "text", "text": "recovering"}],
        },
    )

    _, _, _, normalized = _ingest(source, tmp_path / "powershell")

    record = normalized.tool_call_records[0]
    assert record.status is ToolCallStatus.PAIRED_ERROR
    assert record.error_signature is not None
    assert record.error_signature.startswith("powershell:")
    assert record.error_signature.endswith(":paired_error")


def test_snapshot_parser_rejects_raw_traj_registration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    snapshot = json.loads(_write_snapshot(tmp_path / "snapshot.json"))
    snapshot.update(
        {
            "account": "account",
            "endpoint": "endpoint",
            "extra": "{}",
            "mm_urls": "",
            "model": snapshot["real_model"],
            "p_date": "2026-07-17",
            "source": None,
        }
    )
    snapshot.pop("real_model")
    snapshot.pop("request_model")
    source.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
    registered = TraceSourceRegistry(
        tmp_path / "raw-registry.sqlite3",
    ).register(
        source,
        source_trace_id="source-trace://raw/example",
    )

    with pytest.raises(
        RegisteredSourceMismatchError,
        match="adapter identity",
    ):
        RuntimeSnapshotV1Parser().parse(
            source,
            registered_source=registered,
            audit=_audit(),
        )


def test_snapshot_parser_detects_mutation_after_registration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "snapshot.json"
    _write_snapshot(source)
    adapter = RuntimeSnapshotV1Adapter()
    registered = TraceSourceRegistry(
        tmp_path / "registry.sqlite3",
        adapter=adapter,
    ).register(
        source,
        source_trace_id="source-trace://runtime-snapshot/example",
    )
    source.write_bytes(source.read_bytes() + b" ")

    with pytest.raises(
        RegisteredSourceMismatchError,
        match="immutable source registration",
    ):
        RuntimeSnapshotV1Parser().parse(
            source,
            registered_source=registered,
            audit=_audit(),
        )


def test_raw_parser_still_rejects_snapshot_registration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "snapshot.json"
    _write_snapshot(source)
    adapter = RuntimeSnapshotV1Adapter()
    registered = TraceSourceRegistry(
        tmp_path / "registry.sqlite3",
        adapter=adapter,
    ).register(
        source,
        source_trace_id="source-trace://runtime-snapshot/example",
    )

    with pytest.raises(
        RegisteredSourceMismatchError,
        match="adapter identity",
    ):
        RawTrajV1Parser().parse(
            source,
            registered_source=registered,
            audit=_audit(),
        )


def test_trace_index_stage_service_executes_and_replays_snapshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "snapshot.json"
    raw = _write_snapshot(source)
    adapter = RuntimeSnapshotV1Adapter()
    probe = adapter.probe(source)
    assert probe.raw_sha256 is not None
    registry = TraceSourceRegistry(
        tmp_path / "registry.sqlite3",
        adapter=adapter,
    )
    trace_store = TraceIndexStore(tmp_path / "trace-store")
    service = TraceIndexStageService(
        source_registry=registry,
        trace_store=trace_store,
        parser=RuntimeSnapshotV1Parser(),
        recovery=RawTrajRecovery(adapter=adapter),
        normalizer=RawTrajV1Normalizer(adapter=adapter),
    )
    trace = TraceSourceRef(
        source_trace_id="source-trace://runtime-snapshot/stage",
        source_uri=source.resolve().as_uri(),
        raw_sha256=probe.raw_sha256,
        adapter_name=adapter.name,
        adapter_version=adapter.version,
        processing_class="RESTRICTED_TRACE_RAW",
    )

    first = service.execute(
        trace=trace,
        audit=_audit(),
        job_id="job://runtime-snapshot/stage",
        attempt=1,
    )
    replay = service.execute(
        trace=trace,
        audit=_audit(datetime(2026, 8, 10, tzinfo=UTC)),
        job_id="job://runtime-snapshot/stage",
        attempt=1,
    )

    manifest = first.stored_index.manifest
    assert replay.trace_envelope_ref == first.trace_envelope_ref
    assert replay.stored_index.manifest == manifest
    assert manifest.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert manifest.adapter_name == adapter.name
    assert manifest.adapter_version == adapter.version
    assert manifest.parse_quality == ParseQuality.STRICT.value
    assert trace_store.load(manifest.trace_ir_version_id).manifest == manifest


def test_codex_responses_shape_normalizes_messages_and_function_calls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "codex.json"
    _write_runtime_snapshot(
        source,
        api_type="Response",
        request={
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "developer context",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "inspect the repository",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "read",
                    "arguments": '{"file_path":"README.md"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "repository overview",
                },
            ]
        },
        response={
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "private"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "done",
                        }
                    ],
                },
            ]
        },
    )

    _, _, recovered, normalized = _ingest(source, tmp_path)

    assert recovered.parse_quality is ParseQuality.STRICT
    assert {value.status for value in recovered.capabilities} == {CapabilityStatus.COMPLETE}
    assert [value.event_type for value in normalized.events] == [
        TraceEventType.SYSTEM_CONTEXT,
        TraceEventType.USER_TEXT,
        TraceEventType.TOOL_CALL,
        TraceEventType.TOOL_RESULT,
        TraceEventType.ASSISTANT_TEXT,
    ]
    assert len(normalized.tool_call_records) == 1
    assert normalized.tool_call_records[0].call_id == "call-1"
    assert all(b"private" not in blob.canonical_bytes for blob in normalized.content_blobs)


def test_hermes_chat_shape_normalizes_choices_and_tool_messages(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hermes.json"
    _write_runtime_snapshot(
        source,
        api_type="Chat",
        request={
            "messages": [
                {"role": "system", "content": "system context"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect the repository"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.invalid/input.png"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"file_path":"README.md"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "repository overview",
                },
            ]
        },
        response={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    )

    _, _, recovered, normalized = _ingest(source, tmp_path)

    assert recovered.parse_quality is ParseQuality.STRICT
    assert {value.status for value in recovered.capabilities} == {CapabilityStatus.COMPLETE}
    assert [value.event_type for value in normalized.events] == [
        TraceEventType.SYSTEM_CONTEXT,
        TraceEventType.USER_TEXT,
        TraceEventType.ATTACHMENT_REFERENCE,
        TraceEventType.TOOL_CALL,
        TraceEventType.TOOL_RESULT,
        TraceEventType.ASSISTANT_TEXT,
    ]
    assert len(normalized.tool_call_records) == 1
    assert normalized.tool_call_records[0].call_id == "call-1"


def test_claude_empty_block_types_recover_only_closed_structural_shapes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "claude-empty-type.json"
    _write_runtime_snapshot(
        source,
        api_type="Message",
        request={
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "",
                            "id": "call-1",
                            "name": "Read",
                            "input": {"file_path": "README.md"},
                        },
                        {"type": "", "text": "visible"},
                        {"type": "", "thinking": "private"},
                        {"type": "", "unexpected": "unknown"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "repository overview",
                        }
                    ],
                },
            ]
        },
        response={
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
        },
    )

    _, _, _, normalized = _ingest(source, tmp_path)

    assert [value.event_type for value in normalized.events] == [
        TraceEventType.TOOL_CALL,
        TraceEventType.ASSISTANT_TEXT,
        TraceEventType.TOOL_RESULT,
        TraceEventType.ASSISTANT_TEXT,
    ]
    assert len(normalized.tool_call_records) == 1
    assert all(b"private" not in blob.canonical_bytes for blob in normalized.content_blobs)
    codes = [value.code for value in normalized.diagnostics]
    assert codes.count("normalization-empty-block-type-recovered") == 3
    assert codes.count("normalization-unknown-block-type") == 1
