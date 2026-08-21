from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.trace import (
    RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS,
    RuntimeSnapshotV1Adapter,
    TraceProbeResultV2,
    TraceProbeStatus,
    TraceSourceConflictError,
    TraceSourceRegistry,
    UnsupportedTraceSourceError,
)


def _snapshot(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "sid": "session-1",
        "event_time": "2026-07-17 00:00:08",
        "api_type": "Message",
        "business": "CodingPlan",
        "real_model": "model-a",
        "request_model": "model-request-a",
        "request": '{"messages":[]}',
        "response": '{"content":[]}',
    }
    value.update(overrides)
    return value


def _write_snapshot(path: Path, **overrides: object) -> bytes:
    raw = json.dumps(
        _snapshot(**overrides),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    path.write_bytes(raw)
    return raw


def test_probe_accepts_runtime_snapshot_and_preserves_original_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "req_0123456789abcdef_raw.json"
    before = _write_snapshot(source)

    result = RuntimeSnapshotV1Adapter().probe(source)

    assert result == TraceProbeResultV2(
        adapter_name="runtime_snapshot_v1",
        adapter_version="1.0.0",
        source_uri=source.resolve().as_uri(),
        status=TraceProbeStatus.SUPPORTED,
        raw_sha256=hashlib.sha256(before).hexdigest(),
        size_bytes=len(before),
        record_count=1,
        outer_fields=tuple(sorted(RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS)),
        diagnostics=(),
    )
    assert source.read_bytes() == before


def test_registry_persists_snapshot_adapter_and_original_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "req_0123456789abcdef_raw.json"
    before = _write_snapshot(source)
    registry = TraceSourceRegistry(
        tmp_path / "registry.sqlite3",
        adapter=RuntimeSnapshotV1Adapter(),
    )

    first = registry.register(
        source,
        source_trace_id="source-trace://runtime-snapshot/session-1",
        source_uri="runtime-snapshot://claude-code/session-1",
    )
    replay = registry.register(
        source,
        source_trace_id="source-trace://runtime-snapshot/session-1",
        source_uri="runtime-snapshot://claude-code/session-1",
    )

    assert replay == first
    assert first.source.raw_sha256 == hashlib.sha256(before).hexdigest()
    assert first.source.adapter_name == "runtime_snapshot_v1"
    assert first.source.adapter_version == "1.0.0"
    assert first.outer_fields == tuple(sorted(RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS))
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    ("name", "value", "expected_code", "expected_status"),
    [
        (
            "snapshot.jsonl",
            _snapshot(),
            "runtime-snapshot-extension-mismatch",
            TraceProbeStatus.UNSUPPORTED,
        ),
        (
            "snapshot.json",
            [],
            "runtime-snapshot-outer-not-object",
            TraceProbeStatus.INVALID,
        ),
        (
            "snapshot.json",
            {"sid": "missing-fields"},
            "runtime-snapshot-envelope-fields-mismatch",
            TraceProbeStatus.INVALID,
        ),
        (
            "snapshot.json",
            {**_snapshot(), "unknown": "field"},
            "runtime-snapshot-envelope-fields-mismatch",
            TraceProbeStatus.INVALID,
        ),
        (
            "snapshot.json",
            _snapshot(request={"messages": []}),
            "runtime-snapshot-envelope-field-type-invalid",
            TraceProbeStatus.INVALID,
        ),
    ],
)
def test_probe_rejects_non_snapshot_sources(
    tmp_path: Path,
    name: str,
    value: object,
    expected_code: str,
    expected_status: TraceProbeStatus,
) -> None:
    source = tmp_path / name
    source.write_text(json.dumps(value), encoding="utf-8")

    result = RuntimeSnapshotV1Adapter().probe(source)

    assert result.status is expected_status
    assert {item.code for item in result.diagnostics} == {expected_code}


def test_probe_rejects_malformed_missing_directory_and_symlink(
    tmp_path: Path,
) -> None:
    adapter = RuntimeSnapshotV1Adapter()
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json}", encoding="utf-8")
    directory = tmp_path / "directory.json"
    directory.mkdir()
    target = tmp_path / "target.json"
    _write_snapshot(target)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)

    results = (
        adapter.probe(malformed),
        adapter.probe(tmp_path / "missing.json"),
        adapter.probe(directory),
        adapter.probe(symlink),
    )

    assert [value.status for value in results] == [
        TraceProbeStatus.INVALID,
        TraceProbeStatus.UNREADABLE,
        TraceProbeStatus.UNSUPPORTED,
        TraceProbeStatus.UNSUPPORTED,
    ]
    assert [value.diagnostics[0].code for value in results] == [
        "runtime-snapshot-outer-json-invalid",
        "runtime-snapshot-source-missing",
        "runtime-snapshot-not-regular-file",
        "runtime-snapshot-symlink-unsupported",
    ]


def test_registry_rejects_snapshot_after_source_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "snapshot.json"
    _write_snapshot(source, sid="first")
    registry = TraceSourceRegistry(
        tmp_path / "registry.sqlite3",
        adapter=RuntimeSnapshotV1Adapter(),
    )
    registry.register(
        source,
        source_trace_id="source-trace://runtime-snapshot/example",
    )
    _write_snapshot(source, sid="changed")

    with pytest.raises(
        TraceSourceConflictError,
        match="different source metadata",
    ):
        registry.register(
            source,
            source_trace_id="source-trace://runtime-snapshot/example",
        )


def test_probe_contract_is_strict_and_adapter_identity_is_generic() -> None:
    value = TraceProbeResultV2(
        adapter_name="runtime_snapshot_v1",
        adapter_version="1.0.0",
        source_uri="runtime-snapshot://example",
        status=TraceProbeStatus.SUPPORTED,
        raw_sha256="a" * 64,
        size_bytes=1,
        record_count=1,
        outer_fields=("request",),
        diagnostics=(),
    )
    payload = value.model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        TraceProbeResultV2.model_validate(payload)


def test_raw_traj_v1_still_rejects_snapshot_extension(tmp_path: Path) -> None:
    source = tmp_path / "snapshot.json"
    _write_snapshot(source)

    with pytest.raises(UnsupportedTraceSourceError) as error:
        TraceSourceRegistry(tmp_path / "raw.sqlite3").register(
            source,
            source_trace_id="source-trace://must-not-fallback",
        )

    assert error.value.probe.diagnostics[0].code == "raw-traj-extension-mismatch"
