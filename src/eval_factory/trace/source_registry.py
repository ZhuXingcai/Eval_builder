from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.trace.adapters import RawTrajV1Adapter, TraceAdapter
from eval_factory.trace.models import (
    RegisteredTraceSource,
    TraceProbeResult,
    TraceProbeResultV2,
    TraceProbeStatus,
    registered_trace_source_sha256,
)


class TraceSourceRegistryError(RuntimeError):
    pass


class UnsupportedTraceSourceError(TraceSourceRegistryError):
    def __init__(
        self,
        probe: TraceProbeResult | TraceProbeResultV2,
    ) -> None:
        self.probe = probe
        codes = ", ".join(item.code for item in probe.diagnostics) or probe.status
        super().__init__(f"trace source is not registerable: {codes}")


class TraceSourceConflictError(TraceSourceRegistryError):
    pass


class TraceSourceNotFoundError(TraceSourceRegistryError):
    pass


class TraceSourceRegistry:
    def __init__(
        self,
        path: Path,
        *,
        adapter: TraceAdapter | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter or RawTrajV1Adapter()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS content_blobs (
                    raw_sha256 TEXT PRIMARY KEY,
                    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0)
                );

                CREATE TABLE IF NOT EXISTS trace_sources (
                    source_trace_id TEXT PRIMARY KEY,
                    source_uri TEXT NOT NULL UNIQUE,
                    raw_sha256 TEXT NOT NULL REFERENCES content_blobs(raw_sha256),
                    adapter_name TEXT NOT NULL,
                    adapter_version TEXT NOT NULL,
                    processing_class TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS trace_sources_raw_sha256
                ON trace_sources(raw_sha256);
                """
            )

    def register(
        self,
        source_path: Path,
        *,
        source_trace_id: str,
        source_uri: str | None = None,
    ) -> RegisteredTraceSource:
        probe = self.adapter.probe(source_path)
        if probe.status is not TraceProbeStatus.SUPPORTED:
            raise UnsupportedTraceSourceError(probe)
        assert probe.raw_sha256 is not None
        assert probe.size_bytes is not None

        trace_source_ref = TraceSourceRef(
            source_trace_id=source_trace_id,
            source_uri=source_uri or probe.source_uri,
            raw_sha256=probe.raw_sha256,
            adapter_name=self.adapter.name,
            adapter_version=self.adapter.version,
            processing_class="RESTRICTED_TRACE_RAW",
        )
        record = RegisteredTraceSource(
            source=trace_source_ref,
            size_bytes=probe.size_bytes,
            record_count=probe.record_count,
            outer_fields=probe.outer_fields,
            registration_sha256=registered_trace_source_sha256(
                trace_source_ref,
                probe.size_bytes,
                probe.record_count,
                probe.outer_fields,
            ),
        )

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._find_by_id(connection, source_trace_id)
            if existing is not None:
                if existing == record:
                    connection.commit()
                    return existing
                raise TraceSourceConflictError(
                    f"source_trace_id {source_trace_id!r} is already bound to different source metadata"
                )
            uri_owner = connection.execute(
                """
                SELECT source_trace_id, raw_sha256
                FROM trace_sources
                WHERE source_uri = ?
                """,
                (trace_source_ref.source_uri,),
            ).fetchone()
            if uri_owner is not None:
                raise TraceSourceConflictError(
                    f"source_uri {trace_source_ref.source_uri!r} is already registered by "
                    f"{uri_owner['source_trace_id']!r}"
                )
            blob = connection.execute(
                """
                SELECT size_bytes
                FROM content_blobs
                WHERE raw_sha256 = ?
                """,
                (record.source.raw_sha256,),
            ).fetchone()
            if blob is None:
                connection.execute(
                    """
                    INSERT INTO content_blobs (raw_sha256, size_bytes)
                    VALUES (?, ?)
                    """,
                    (record.source.raw_sha256, record.size_bytes),
                )
            elif int(blob["size_bytes"]) != record.size_bytes:
                raise TraceSourceConflictError(
                    f"content hash {record.source.raw_sha256} has conflicting byte size"
                )
            connection.execute(
                """
                INSERT INTO trace_sources (
                    source_trace_id, source_uri, raw_sha256,
                    adapter_name, adapter_version, processing_class, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.source.source_trace_id,
                    record.source.source_uri,
                    record.source.raw_sha256,
                    record.source.adapter_name,
                    record.source.adapter_version,
                    record.source.processing_class,
                    record.canonical_json().decode(),
                ),
            )
            connection.commit()
            return record
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, source_trace_id: str) -> RegisteredTraceSource:
        with closing(self._connect()) as connection, connection:
            record = self._find_by_id(connection, source_trace_id)
            if record is None:
                raise TraceSourceNotFoundError(f"trace source not found: {source_trace_id}")
            return record

    def list_sources(self) -> tuple[RegisteredTraceSource, ...]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT record_json
                FROM trace_sources
                ORDER BY source_trace_id
                """
            ).fetchall()
            return tuple(RegisteredTraceSource.model_validate_json(str(row["record_json"])) for row in rows)

    def content_blob_count(self) -> int:
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM content_blobs").fetchone()
            assert row is not None
            return int(row["count"])

    @staticmethod
    def _find_by_id(
        connection: sqlite3.Connection,
        source_trace_id: str,
    ) -> RegisteredTraceSource | None:
        row = connection.execute(
            """
            SELECT record_json
            FROM trace_sources
            WHERE source_trace_id = ?
            """,
            (source_trace_id,),
        ).fetchone()
        if row is None:
            return None
        return RegisteredTraceSource.model_validate_json(str(row["record_json"]))
