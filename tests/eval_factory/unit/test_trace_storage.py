from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import eval_factory.trace.storage.projection as projection_module
from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.trace import (
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RawTrajV1Parser,
    TraceIndexBuilder,
    TraceIndexResult,
    TraceSourceRegistry,
)
from eval_factory.trace.storage import (
    TraceCasConflictError,
    TraceFactConflictError,
    TraceIndexStore,
    TraceStoreCorruptionError,
)


def _outer(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "StorageGold",
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


def _write_outer(path: Path, *records: dict[str, object]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        created_by="trace-storage-test",
        governing_versions=(VersionBinding(component="trace-index-storage", version="v1"),),
    )


def _index(
    source: Path,
    tmp_path: Path,
    *,
    source_trace_id: str = "source-trace://storage-test",
) -> TraceIndexResult:
    audit = _audit()
    registry = TraceSourceRegistry(tmp_path / "registry.sqlite3")
    registered = registry.register(
        source,
        source_trace_id=source_trace_id,
        source_uri=f"raw-traj://{source_trace_id.rsplit('/', 1)[-1]}",
    )
    parsed = RawTrajV1Parser().parse(source, registered_source=registered, audit=audit)
    recovered = RawTrajRecovery().recover(source, parse_result=parsed, audit=audit)
    normalized = RawTrajV1Normalizer().normalize(source, recovery_result=recovered, audit=audit)
    return TraceIndexBuilder().build(normalization_result=normalized, audit=audit)


def _storage_index(tmp_path: Path) -> TraceIndexResult:
    request = {
        "messages": [
            {"role": "system", "content": "storage policy"},
            {"role": "user", "content": "NeedleAlpha please inspect input.txt"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will read it."},
                    {
                        "type": "tool_use",
                        "id": "read-1",
                        "name": "Read",
                        "input": {"file_path": "input.txt"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "read-1",
                        "content": "NeedleAlpha file body",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "write-1",
                        "name": "Write",
                        "input": {"file_path": "out.txt", "content": "final input state"},
                    }
                ],
            },
        ]
    }
    source = tmp_path / "storage.jsonl"
    _write_outer(source, _outer(request=json.dumps(request)))
    return _index(source, tmp_path)


def _fact_line_count(store_root: Path) -> int:
    return sum(1 for _ in (store_root / "facts" / "trace_facts.jsonl").open(encoding="utf-8"))


def _blob_path(store_root: Path, digest: str) -> Path:
    return store_root / "cas" / "sha256" / digest[:2] / digest


def test_persist_load_rebuild_and_search_are_query_equivalent(tmp_path: Path) -> None:
    result = _storage_index(tmp_path)
    store = TraceIndexStore(tmp_path / "store")

    commit = store.persist(result, audit=_audit())
    manifest = store.load_manifest(
        result.normalization_result.trace_ir_version_id,
    )
    loaded = store.load(result.normalization_result.trace_ir_version_id)
    rebuilt = store.rebuild_projection()
    reloaded = store.load(result.normalization_result.trace_ir_version_id)

    assert commit.appended_fact_count > 0
    assert commit.written_blob_count == len(result.normalization_result.content_blobs)
    assert manifest == loaded.manifest
    assert rebuilt.fact_count == commit.appended_fact_count
    assert loaded.event_summary() == [
        (item.sequence, item.event_type.value, item.role) for item in result.normalization_result.events
    ]
    assert reloaded.event_summary() == loaded.event_summary()
    assert loaded.tool_call_summary() == [
        (item.sequence, item.tool_family.value, item.status.value)
        for item in result.normalization_result.tool_call_records
    ]
    assert loaded.file_observation_summary() == [
        (item.sequence, item.logical_path, item.operation.value, item.completeness.value)
        for item in result.file_observations
    ]
    assert loaded.segment_summary() == [
        (item.sequence_start, item.sequence_end, item.boundary_method) for item in result.interaction_segments
    ]
    assert loaded.content_blob_summary() == [
        (item.content_ref.object_id, item.media_type, len(item.canonical_bytes))
        for item in result.normalization_result.content_blobs
    ]

    hits = store.search_text(result.normalization_result.trace_ir_version_id, "NeedleAlpha")
    assert {hit.object_id for hit in hits}
    assert {hit.object_id for hit in hits} <= {
        blob.content_ref.object_id for blob in result.normalization_result.content_blobs
    }


def test_persisted_materialization_matches_disk_reload(tmp_path: Path) -> None:
    result = _storage_index(tmp_path)
    store = TraceIndexStore(tmp_path / "store")

    commit, materialized = store.persist_and_materialize(
        result,
        audit=_audit(),
    )
    loaded = store.load(result.normalization_result.trace_ir_version_id)

    assert commit.appended_fact_count > 0
    assert materialized == loaded


def test_projection_closes_every_sqlite_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _storage_index(tmp_path)
    connections: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(projection_module.sqlite3, "connect", tracked_connect)
    store = TraceIndexStore(tmp_path / "store")

    store.persist(result, audit=_audit())
    store.load(result.normalization_result.trace_ir_version_id)
    store.search_text(result.normalization_result.trace_ir_version_id, "NeedleAlpha")

    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_persist_replay_is_idempotent(tmp_path: Path) -> None:
    result = _storage_index(tmp_path)
    store_root = tmp_path / "store"
    store = TraceIndexStore(store_root)

    first = store.persist(result, audit=_audit())
    line_count = _fact_line_count(store_root)
    second = store.persist(result, audit=_audit())

    assert second.appended_fact_count == 0
    assert second.existing_fact_count == first.appended_fact_count
    assert second.written_blob_count == 0
    assert _fact_line_count(store_root) == line_count


def test_cas_conflict_fails_closed_without_rewriting(tmp_path: Path) -> None:
    result = _storage_index(tmp_path)
    store_root = tmp_path / "store"
    store = TraceIndexStore(store_root)
    store.persist(result, audit=_audit())
    digest = result.normalization_result.content_blobs[0].content_ref.object_sha256
    path = _blob_path(store_root, digest)
    path.write_bytes(b"corrupted bytes")

    with pytest.raises(TraceCasConflictError, match=digest):
        store.persist(result, audit=_audit())


def test_duplicate_conflicting_fact_fails_closed(tmp_path: Path) -> None:
    result = _storage_index(tmp_path)
    store_root = tmp_path / "store"
    store = TraceIndexStore(store_root)
    store.persist(result, audit=_audit())
    fact_log = store_root / "facts" / "trace_facts.jsonl"
    lines = fact_log.read_text(encoding="utf-8").splitlines()
    original = json.loads(next(line for line in lines if '"fact_kind":"trace-event"' in line))
    payload = json.loads(original["canonical_json"])
    payload["role"] = "runtime"
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    original["canonical_json"] = canonical
    original["object_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    original["jsonl_sequence"] = len(lines)
    fact_log.write_text(
        "\n".join([*lines, json.dumps(original, sort_keys=True, separators=(",", ":"))]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TraceFactConflictError, match="conflicting fact"):
        store.load(result.normalization_result.trace_ir_version_id)


def test_corruption_cases_fail_closed_and_rebuild_repairs_projection(tmp_path: Path) -> None:
    result = _storage_index(tmp_path)
    store_root = tmp_path / "store"
    store = TraceIndexStore(store_root)
    store.persist(result, audit=_audit())
    trace_ir_version_id = result.normalization_result.trace_ir_version_id

    with sqlite3.connect(store_root / "projection.sqlite3") as connection:
        connection.execute("DELETE FROM objects")
    with pytest.raises(TraceStoreCorruptionError, match="projection drift"):
        store.load(trace_ir_version_id)

    store.rebuild_projection()
    assert store.load(trace_ir_version_id).event_summary()

    digest = result.normalization_result.content_blobs[0].content_ref.object_sha256
    _blob_path(store_root, digest).unlink()
    with pytest.raises(TraceStoreCorruptionError, match="missing CAS blob"):
        store.load(trace_ir_version_id)


def test_malformed_jsonl_fails_closed(tmp_path: Path) -> None:
    result = _storage_index(tmp_path)
    store_root = tmp_path / "store"
    store = TraceIndexStore(store_root)
    store.persist(result, audit=_audit())
    fact_log = store_root / "facts" / "trace_facts.jsonl"
    fact_log.write_text(fact_log.read_text(encoding="utf-8") + "{not-json}\n", encoding="utf-8")

    with pytest.raises(TraceStoreCorruptionError, match="malformed fact JSONL"):
        store.load(result.normalization_result.trace_ir_version_id)
