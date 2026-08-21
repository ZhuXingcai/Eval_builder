from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

import eval_factory.trace.source_registry as source_registry_module
from eval_factory.trace import (
    RAW_TRAJ_V1_REQUIRED_FIELDS,
    RawTrajV1Adapter,
    RegisteredTraceSource,
    TraceProbeStatus,
    TraceSourceConflictError,
    TraceSourceNotFoundError,
    TraceSourceRegistry,
    UnsupportedTraceSourceError,
)


def _outer(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account": "account",
        "api_type": "chat",
        "business": "CodingPlan",
        "endpoint": "internal",
        "event_time": "2026-07-21T00:00:00Z",
        "extra": "{}",
        "mm_urls": "",
        "model": "model",
        "p_date": "2026-07-21",
        "request": '{"messages":[]}',
        "response": "[]",
        "sid": "session",
        "source": None,
    }
    value.update(overrides)
    return value


def _write_jsonl(path: Path, *records: object, blank_line: bool = False) -> bytes:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    if blank_line:
        lines.insert(1, "")
    content = ("\n".join(lines) + "\n").encode()
    path.write_bytes(content)
    return content


def test_probe_accepts_outer_envelope_without_parsing_nested_json(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    raw = _write_jsonl(
        path,
        _outer(request=r'{"messages":[{"content":"C:\oops"}]}'),
    )

    result = RawTrajV1Adapter().probe(path)

    assert result.status is TraceProbeStatus.SUPPORTED
    assert result.record_count == 1
    assert result.size_bytes == len(raw)
    assert result.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert set(result.outer_fields) == RAW_TRAJ_V1_REQUIRED_FIELDS
    assert result.diagnostics == ()


def test_probe_supports_multiple_records_and_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    raw = _write_jsonl(path, _outer(sid="one"), _outer(sid="two"), blank_line=True)

    result = RawTrajV1Adapter().probe(path)

    assert result.status is TraceProbeStatus.SUPPORTED
    assert result.record_count == 2
    assert result.raw_sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    ("name", "content", "expected_code"),
    [
        ("trace.txt", b"{}\n", "raw-traj-extension-mismatch"),
        ("empty.jsonl", b"", "raw-traj-empty"),
        ("invalid.jsonl", b"{not-json}\n", "raw-traj-outer-json-invalid"),
        ("array.jsonl", b"[]\n", "raw-traj-outer-not-object"),
        (
            "missing.jsonl",
            (json.dumps({"request": "{}", "response": "[]", "extra": "{}"}) + "\n").encode(),
            "raw-traj-envelope-fields-missing",
        ),
        (
            "wrong-type.jsonl",
            (json.dumps(_outer(request={"messages": []})) + "\n").encode(),
            "raw-traj-envelope-field-type-invalid",
        ),
    ],
)
def test_probe_rejects_non_raw_traj_sources(
    tmp_path: Path,
    name: str,
    content: bytes,
    expected_code: str,
) -> None:
    path = tmp_path / name
    path.write_bytes(content)

    result = RawTrajV1Adapter().probe(path)

    expected_status = (
        TraceProbeStatus.UNSUPPORTED
        if expected_code == "raw-traj-extension-mismatch"
        else TraceProbeStatus.INVALID
    )
    assert result.status is expected_status
    assert expected_code in {item.code for item in result.diagnostics}


def test_probe_rejects_missing_directory_and_symlink(tmp_path: Path) -> None:
    adapter = RawTrajV1Adapter()
    missing = adapter.probe(tmp_path / "missing.jsonl")
    directory = tmp_path / "directory.jsonl"
    directory.mkdir()
    not_file = adapter.probe(directory)
    target = tmp_path / "target.jsonl"
    _write_jsonl(target, _outer())
    symlink = tmp_path / "symlink.jsonl"
    symlink.symlink_to(target)
    linked = adapter.probe(symlink)

    assert missing.status is TraceProbeStatus.UNREADABLE
    assert missing.diagnostics[0].code == "raw-traj-source-missing"
    assert not_file.status is TraceProbeStatus.UNSUPPORTED
    assert not_file.diagnostics[0].code == "raw-traj-not-regular-file"
    assert linked.status is TraceProbeStatus.UNSUPPORTED
    assert linked.diagnostics[0].code == "raw-traj-symlink-unsupported"


def test_registration_is_idempotent_and_does_not_modify_source(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    before = _write_jsonl(source, _outer())
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")

    first = registry.register(
        source,
        source_trace_id="source-trace://example",
        source_uri="raw-traj://example",
    )
    replay = registry.register(
        source,
        source_trace_id="source-trace://example",
        source_uri="raw-traj://example",
    )

    assert replay == first
    assert registry.get(first.source.source_trace_id) == first
    assert registry.list_sources() == (first,)
    assert registry.content_blob_count() == 1
    assert source.read_bytes() == before
    assert first.source.raw_sha256 == hashlib.sha256(before).hexdigest()
    assert first.source.adapter_name == "raw_traj_v1"
    assert first.source.adapter_version == "1.0.0"
    assert first.source.processing_class == "RESTRICTED_TRACE_RAW"


def test_registry_closes_every_sqlite_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(source_registry_module.sqlite3, "connect", tracked_connect)
    source = tmp_path / "trace.jsonl"
    _write_jsonl(source, _outer())
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id="source-trace://example",
    )

    assert registry.get(registered.source.source_trace_id) == registered
    assert registry.list_sources() == (registered,)
    assert registry.content_blob_count() == 1
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_same_content_can_have_distinct_source_ids_without_duplicate_blob(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write_jsonl(first_path, _outer())
    shutil.copyfile(first_path, second_path)
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")

    first = registry.register(
        first_path,
        source_trace_id="source-trace://first",
        source_uri="raw-traj://first",
    )
    second = registry.register(
        second_path,
        source_trace_id="source-trace://second",
        source_uri="raw-traj://second",
    )

    assert first.source.raw_sha256 == second.source.raw_sha256
    assert registry.content_blob_count() == 1
    assert len(registry.list_sources()) == 2


def test_changed_file_or_conflicting_uri_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    _write_jsonl(source, _outer(sid="first"))
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registry.register(
        source,
        source_trace_id="source-trace://example",
        source_uri="raw-traj://example",
    )

    _write_jsonl(source, _outer(sid="changed"))
    with pytest.raises(TraceSourceConflictError, match="different source metadata"):
        registry.register(
            source,
            source_trace_id="source-trace://example",
            source_uri="raw-traj://example",
        )
    with pytest.raises(TraceSourceConflictError, match="already registered"):
        registry.register(
            source,
            source_trace_id="source-trace://different",
            source_uri="raw-traj://example",
        )


def test_registration_rejects_unsupported_source_and_missing_lookup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")

    with pytest.raises(UnsupportedTraceSourceError) as error:
        registry.register(source, source_trace_id="source-trace://bad")
    assert error.value.probe.status is TraceProbeStatus.INVALID
    with pytest.raises(TraceSourceNotFoundError):
        registry.get("source-trace://missing")


def test_registered_source_hash_is_self_validating(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    _write_jsonl(source, _outer())
    record = TraceSourceRegistry(tmp_path / "registry.sqlite3").register(
        source,
        source_trace_id="source-trace://example",
    )
    payload = record.model_dump(mode="json")
    payload["registration_sha256"] = "b" * 64

    with pytest.raises(ValidationError, match="registration hash mismatch"):
        RegisteredTraceSource.model_validate_json(json.dumps(payload))


def test_probe_matches_frozen_canary_source_hashes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    raw_root = repo_root.parent / "raw_traj"
    if not raw_root.is_dir():
        pytest.skip("local raw_traj corpus is unavailable")
    manifest = json.loads(
        (repo_root / "evals/golden/eval_factory/canary_manifest.v4.json").read_text(encoding="utf-8")
    )
    by_instance = {
        path.name.split("_", 2)[0] + "_" + path.name.split("_", 2)[1]: path
        for path in raw_root.glob("*.jsonl")
    }
    adapter = RawTrajV1Adapter()

    for item in manifest["items"]:
        source = by_instance[item["instance_id"]]
        before = source.read_bytes()
        result = adapter.probe(source)
        assert result.status is TraceProbeStatus.SUPPORTED
        assert result.raw_sha256 == item["raw_sha256"]
        assert result.size_bytes == item["size_bytes"]
        assert source.read_bytes() == before
