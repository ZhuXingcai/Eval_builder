from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from eval_factory.trace.storage.facts import projection_sha256
from eval_factory.trace.storage.models import (
    TraceFactEnvelope,
    TraceFactKind,
    TraceProjectionError,
    TraceProjectionRebuild,
    TraceStoreCorruptionError,
    TraceTextHit,
)


class TraceProjection:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
            self._create_schema(connection)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def rebuild(self, facts: tuple[TraceFactEnvelope, ...]) -> TraceProjectionRebuild:
        try:
            with self._transaction() as connection:
                self._drop_projection(connection)
                self._create_schema(connection)
                for fact in facts:
                    self._insert_fact(connection, fact)
                digest = self._projection_digest(connection)
        except sqlite3.Error as exc:
            raise TraceProjectionError("failed to rebuild trace projection") from exc
        return TraceProjectionRebuild(fact_count=len(facts), projection_sha256=digest)

    def ensure_consistent(self, facts: tuple[TraceFactEnvelope, ...]) -> None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM objects").fetchone()
            assert row is not None
            if int(row["count"]) != len(facts):
                raise TraceStoreCorruptionError("projection drift: object count differs from fact log")
            rows = connection.execute(
                """
                SELECT object_type, object_id, object_version, object_sha256, jsonl_sequence
                FROM objects
                ORDER BY jsonl_sequence
                """
            ).fetchall()
        observed = tuple(
            (
                str(row["object_type"]),
                str(row["object_id"]),
                str(row["object_version"]),
                str(row["object_sha256"]),
                int(row["jsonl_sequence"]),
            )
            for row in rows
        )
        expected = tuple(
            (
                fact.object_type,
                fact.object_id,
                fact.object_version,
                fact.object_sha256,
                fact.jsonl_sequence,
            )
            for fact in facts
        )
        if observed != expected:
            raise TraceStoreCorruptionError("projection drift: object metadata differs from fact log")

    def search_text(self, trace_ir_version_id: str, query: str) -> tuple[TraceTextHit, ...]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT object_id, object_sha256, media_type
                FROM content_fts
                WHERE trace_ir_version_id = ? AND content_fts MATCH ?
                ORDER BY object_id
                """,
                (trace_ir_version_id, query),
            ).fetchall()
        return tuple(
            TraceTextHit(
                object_id=str(row["object_id"]),
                object_sha256=str(row["object_sha256"]),
                media_type=str(row["media_type"]),
            )
            for row in rows
        )

    def _drop_projection(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            DROP TABLE IF EXISTS trace_versions;
            DROP TABLE IF EXISTS objects;
            DROP TABLE IF EXISTS content_blobs;
            DROP TABLE IF EXISTS source_spans;
            DROP TABLE IF EXISTS events;
            DROP TABLE IF EXISTS tool_call_records;
            DROP TABLE IF EXISTS file_observations;
            DROP TABLE IF EXISTS interaction_segments;
            DROP TABLE IF EXISTS segment_members;
            DROP TABLE IF EXISTS fact_batches;
            DROP TABLE IF EXISTS content_fts;
            """
        )

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS trace_versions (
                trace_ir_version_id TEXT PRIMARY KEY,
                source_trace_id TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                raw_sha256 TEXT NOT NULL,
                parse_quality TEXT NOT NULL,
                policy_json TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS objects (
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                object_version TEXT NOT NULL,
                object_sha256 TEXT NOT NULL,
                trace_ir_version_id TEXT NOT NULL,
                fact_kind TEXT NOT NULL,
                jsonl_sequence INTEGER NOT NULL,
                canonical_json TEXT NOT NULL,
                PRIMARY KEY (object_type, object_id, object_version)
            );
            CREATE INDEX IF NOT EXISTS objects_trace_version ON objects(trace_ir_version_id);

            CREATE TABLE IF NOT EXISTS content_blobs (
                object_id TEXT PRIMARY KEY,
                object_sha256 TEXT NOT NULL,
                media_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                cas_path TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_spans (
                span_id TEXT PRIMARY KEY,
                trace_ir_version_id TEXT NOT NULL,
                source_trace_id TEXT NOT NULL,
                field TEXT NOT NULL,
                raw_byte_start INTEGER NOT NULL,
                raw_byte_end INTEGER NOT NULL,
                approximate INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                trace_ir_version_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                role TEXT,
                content_ref_id TEXT
            );
            CREATE INDEX IF NOT EXISTS events_trace_type ON events(trace_ir_version_id, event_type, sequence);

            CREATE TABLE IF NOT EXISTS tool_call_records (
                tool_call_record_id TEXT PRIMARY KEY,
                trace_ir_version_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                tool_family TEXT NOT NULL,
                status TEXT NOT NULL,
                call_event_id TEXT,
                result_event_id TEXT
            );

            CREATE TABLE IF NOT EXISTS file_observations (
                observation_id TEXT PRIMARY KEY,
                trace_ir_version_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                logical_path TEXT NOT NULL,
                operation TEXT NOT NULL,
                completeness TEXT NOT NULL,
                file_version_id TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS file_observations_path
            ON file_observations(trace_ir_version_id, logical_path);

            CREATE TABLE IF NOT EXISTS interaction_segments (
                segment_id TEXT PRIMARY KEY,
                trace_ir_version_id TEXT NOT NULL,
                boundary_method TEXT NOT NULL,
                sequence_start INTEGER NOT NULL,
                sequence_end INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS segment_members (
                segment_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                member_index INTEGER NOT NULL,
                PRIMARY KEY (segment_id, member_index)
            );

            CREATE TABLE IF NOT EXISTS fact_batches (
                batch_id TEXT PRIMARY KEY,
                trace_ir_version_id TEXT NOT NULL,
                batch_sha256 TEXT NOT NULL,
                status TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                object_id UNINDEXED,
                object_sha256 UNINDEXED,
                trace_ir_version_id UNINDEXED,
                media_type UNINDEXED,
                text
            );
            """
        )

    def _insert_fact(self, connection: sqlite3.Connection, fact: TraceFactEnvelope) -> None:
        connection.execute(
            """
            INSERT INTO objects (
                object_type, object_id, object_version, object_sha256,
                trace_ir_version_id, fact_kind, jsonl_sequence, canonical_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.object_type,
                fact.object_id,
                fact.object_version,
                fact.object_sha256,
                fact.trace_ir_version_id,
                fact.fact_kind.value,
                fact.jsonl_sequence,
                fact.payload_json,
            ),
        )
        payload = json.loads(fact.payload_json)
        if fact.fact_kind is TraceFactKind.TRACE_ENVELOPE:
            self._insert_manifest(connection, fact, payload)
        elif fact.fact_kind is TraceFactKind.SOURCE_SPAN:
            connection.execute(
                """
                INSERT INTO source_spans (
                    span_id, trace_ir_version_id, source_trace_id, field,
                    raw_byte_start, raw_byte_end, approximate
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["span_id"],
                    fact.trace_ir_version_id,
                    payload["source_trace_id"],
                    payload["field"],
                    payload["raw_byte_start"],
                    payload["raw_byte_end"],
                    int(bool(payload["approximate"])),
                ),
            )
        elif fact.fact_kind is TraceFactKind.TRACE_EVENT:
            connection.execute(
                """
                INSERT INTO events (
                    event_id, trace_ir_version_id, sequence, event_type, role, content_ref_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["event_id"],
                    fact.trace_ir_version_id,
                    payload["sequence"],
                    payload["event_type"],
                    payload.get("role"),
                    (payload["content_ref"]["object_id"] if payload.get("content_ref") else None),
                ),
            )
        elif fact.fact_kind is TraceFactKind.TOOL_CALL_RECORD:
            connection.execute(
                """
                INSERT INTO tool_call_records (
                    tool_call_record_id, trace_ir_version_id, sequence, tool_family,
                    status, call_event_id, result_event_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["tool_call_record_id"],
                    fact.trace_ir_version_id,
                    payload["sequence"],
                    payload["tool_family"],
                    payload["status"],
                    (payload["call_event_ref"]["object_id"] if payload.get("call_event_ref") else None),
                    (payload["result_event_ref"]["object_id"] if payload.get("result_event_ref") else None),
                ),
            )
        elif fact.fact_kind is TraceFactKind.FILE_OBSERVATION:
            connection.execute(
                """
                INSERT INTO file_observations (
                    observation_id, trace_ir_version_id, sequence, logical_path,
                    operation, completeness, file_version_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["observation_id"],
                    fact.trace_ir_version_id,
                    payload["sequence"],
                    payload["logical_path"],
                    payload["operation"],
                    payload["completeness"],
                    payload["file_version_id"],
                ),
            )
        elif fact.fact_kind is TraceFactKind.INTERACTION_SEGMENT:
            connection.execute(
                """
                INSERT INTO interaction_segments (
                    segment_id, trace_ir_version_id, boundary_method, sequence_start, sequence_end
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload["segment_id"],
                    fact.trace_ir_version_id,
                    payload["boundary_method"],
                    payload["sequence_start"],
                    payload["sequence_end"],
                ),
            )
            for index, ref in enumerate(payload["member_event_refs"]):
                connection.execute(
                    """
                    INSERT INTO segment_members (segment_id, event_id, member_index)
                    VALUES (?, ?, ?)
                    """,
                    (payload["segment_id"], ref["object_id"], index),
                )
        elif fact.fact_kind is TraceFactKind.CONTENT_BLOB:
            media_type = payload["media_type"]
            connection.execute(
                """
                INSERT INTO content_blobs (
                    object_id, object_sha256, media_type, size_bytes, cas_path
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload["content_ref"]["object_id"],
                    payload["content_ref"]["object_sha256"],
                    media_type,
                    payload["size_bytes"],
                    f"cas/sha256/{payload['content_ref']['object_sha256'][:2]}/"
                    f"{payload['content_ref']['object_sha256']}",
                ),
            )
        elif fact.fact_kind is TraceFactKind.REPAIR_MAP:
            return

    def populate_fts(self, content: dict[str, tuple[str, str, bytes]]) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM content_fts")
            for object_id, (trace_ir_version_id, media_type, raw) in sorted(content.items()):
                if media_type != "text/plain; charset=utf-8":
                    continue
                connection.execute(
                    """
                    INSERT INTO content_fts (
                        object_id, object_sha256, trace_ir_version_id, media_type, text
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        object_id,
                        _sha256(raw),
                        trace_ir_version_id,
                        media_type,
                        raw.decode("utf-8", errors="replace"),
                    ),
                )

    def _insert_manifest(
        self,
        connection: sqlite3.Connection,
        fact: TraceFactEnvelope,
        payload: dict[str, object],
    ) -> None:
        policy = {
            "adapter_name": payload["adapter_name"],
            "adapter_version": payload["adapter_version"],
            "trace_ir_schema_version": payload["trace_ir_schema_version"],
            "repair_policy_version": payload["repair_policy_version"],
            "recovery_policy_version": payload["recovery_policy_version"],
            "normalization_policy_version": payload["normalization_policy_version"],
            "tool_family_policy_version": payload["tool_family_policy_version"],
            "file_observation_policy_version": payload["file_observation_policy_version"],
            "segmentation_policy_version": payload["segmentation_policy_version"],
        }
        connection.execute(
            """
            INSERT INTO trace_versions (
                trace_ir_version_id, source_trace_id, source_uri, raw_sha256,
                parse_quality, policy_json, manifest_json, manifest_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.trace_ir_version_id,
                payload["source_trace_id"],
                payload["source_uri"],
                payload["raw_sha256"],
                payload["parse_quality"],
                json.dumps(policy, sort_keys=True, separators=(",", ":")),
                fact.payload_json,
                fact.object_sha256,
            ),
        )

    def _projection_digest(self, connection: sqlite3.Connection) -> str:
        rows = {
            "objects": [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT object_type, object_id, object_version, object_sha256, fact_kind,
                           trace_ir_version_id, jsonl_sequence
                    FROM objects
                    ORDER BY jsonl_sequence
                    """
                )
            ],
            "events": [dict(row) for row in connection.execute("SELECT * FROM events ORDER BY sequence")],
            "files": [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM file_observations ORDER BY sequence, observation_id"
                )
            ],
            "segments": [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM interaction_segments ORDER BY sequence_start, segment_id"
                )
            ],
        }
        return projection_sha256(rows)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
