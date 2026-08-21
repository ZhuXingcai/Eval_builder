from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.memory.models import (
    MemoryAccessContextV1,
    MemoryCandidateV1,
    MemoryHeadStateV1,
    MemoryRecordV1,
    MemoryTombstoneV1,
    MemoryVisibilityV1,
    StoredMemoryV1,
)
from eval_factory.memory.policy import (
    MemoryAdmissionPolicy,
    MemoryAuthorizationError,
    authorize_namespace,
)


class AgentMemoryStoreError(RuntimeError):
    pass


class MemoryNotFoundError(AgentMemoryStoreError):
    pass


class MemoryTombstonedError(AgentMemoryStoreError):
    pass


class MemoryConcurrencyError(AgentMemoryStoreError):
    pass


class MemoryIdempotencyConflictError(AgentMemoryStoreError):
    pass


class MemoryIntegrityError(AgentMemoryStoreError):
    pass


class AgentMemoryStore:
    """SQLite authority for private cross-run Agent memories."""

    def __init__(
        self,
        path: Path,
        *,
        trusted_approval_refs: frozenset[ObjectRef] = frozenset(),
        trusted_access_contexts: frozenset[MemoryAccessContextV1] = frozenset(),
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if any(reference.object_type != "memory-admission-approval" for reference in trusted_approval_refs):
            raise ValueError("trusted memory approvals require memory-admission-approval refs")
        self._trusted_approval_refs = trusted_approval_refs
        self._trusted_access_contexts = trusted_access_contexts
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute("PRAGMA journal_size_limit = 0")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_revisions (
                    memory_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    memory_record_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    predecessor_record_id TEXT,
                    tenant_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    owner_agent_id TEXT,
                    kind TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    expires_at TEXT,
                    importance_basis_points INTEGER NOT NULL,
                    sensitivity TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY(memory_id, revision)
                );

                CREATE TABLE IF NOT EXISTS memory_contents (
                    memory_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    content_text TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    PRIMARY KEY(memory_id, revision),
                    FOREIGN KEY(memory_id, revision)
                        REFERENCES memory_revisions(memory_id, revision)
                );

                CREATE TABLE IF NOT EXISTS memory_heads (
                    memory_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    memory_record_id TEXT NOT NULL,
                    tombstone_id TEXT,
                    tenant_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    owner_agent_id TEXT,
                    kind TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS erased_memory_revision_digests (
                    memory_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    memory_record_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    predecessor_record_id TEXT,
                    content_sha256 TEXT NOT NULL,
                    PRIMARY KEY(memory_id, revision)
                );

                CREATE TABLE IF NOT EXISTS memory_tombstones (
                    tombstone_id TEXT PRIMARY KEY,
                    object_sha256 TEXT NOT NULL,
                    memory_id TEXT NOT NULL UNIQUE,
                    final_revision INTEGER NOT NULL,
                    prior_record_id TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    deleted_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_idempotency (
                    operation_scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_type TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    PRIMARY KEY(operation_scope, idempotency_key)
                );

                CREATE INDEX IF NOT EXISTS memory_heads_scope_idx
                ON memory_heads(
                    tenant_id, project_id, subject_id, state,
                    visibility, owner_agent_id
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

    @contextmanager
    def _read_snapshot(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.rollback()
            connection.close()

    def remember(
        self,
        candidate: MemoryCandidateV1,
        *,
        access: MemoryAccessContextV1,
        expected_revision: int,
        idempotency_key: str,
        created_at: datetime,
    ) -> MemoryRecordV1:
        if expected_revision < 0:
            raise ValueError("expected memory revision must be non-negative")
        _require_aware(created_at, "memory creation timestamp")
        self._authorize_access_context(access)
        MemoryAdmissionPolicy.validate(
            candidate,
            access=access,
            trusted_approval_refs=self._trusted_approval_refs,
        )
        request_sha256 = _request_sha256(
            {
                "operation": "remember",
                "candidate": candidate,
                "access": access,
                "expected_revision": expected_revision,
                "created_at": created_at,
            }
        )
        scope = _operation_scope("remember", access)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_replay(
                connection,
                operation_scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-memory-record",
                model_type=MemoryRecordV1,
            )
            if replay is not None:
                connection.rollback()
                if not isinstance(replay, MemoryRecordV1):
                    raise MemoryIntegrityError("memory idempotency response has the wrong type")
                return replay

            head = connection.execute(
                "SELECT * FROM memory_heads WHERE memory_id = ?",
                (candidate.memory_id,),
            ).fetchone()
            predecessor: MemoryRecordV1 | None = None
            if head is None:
                if expected_revision != 0:
                    raise MemoryConcurrencyError("new memory requires expected revision zero")
                revision = 1
            else:
                self._authorize_head(head, access=access)
                if str(head["state"]) == MemoryHeadStateV1.TOMBSTONED.value:
                    raise MemoryTombstonedError("tombstoned memory cannot be reactivated")
                if int(head["revision"]) != expected_revision:
                    raise MemoryConcurrencyError("memory revision is stale")
                predecessor = self._load_record_row(
                    connection.execute(
                        """
                        SELECT * FROM memory_revisions
                        WHERE memory_id = ? AND revision = ?
                        """,
                        (candidate.memory_id, expected_revision),
                    ).fetchone()
                )
                if predecessor.namespace != candidate.namespace:
                    raise MemoryConcurrencyError("memory namespace cannot change")
                if predecessor.kind is not candidate.kind:
                    raise MemoryConcurrencyError("memory kind cannot change")
                if created_at < predecessor.created_at:
                    raise MemoryConcurrencyError("memory successor timestamp cannot precede its predecessor")
                revision = expected_revision + 1

            record = MemoryRecordV1.create(
                candidate=candidate,
                revision=revision,
                predecessor_ref=predecessor.to_ref() if predecessor is not None else None,
                created_at=created_at,
            )
            self._insert_record(connection, record=record, content=candidate.content)
            connection.execute(
                """
                INSERT INTO memory_heads (
                    memory_id, revision, state, memory_record_id, tombstone_id,
                    tenant_id, project_id, subject_id, visibility,
                    owner_agent_id, kind
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    revision = excluded.revision,
                    state = excluded.state,
                    memory_record_id = excluded.memory_record_id,
                    tombstone_id = NULL,
                    tenant_id = excluded.tenant_id,
                    project_id = excluded.project_id,
                    subject_id = excluded.subject_id,
                    visibility = excluded.visibility,
                    owner_agent_id = excluded.owner_agent_id,
                    kind = excluded.kind
                """,
                (
                    record.memory_id,
                    record.revision,
                    MemoryHeadStateV1.ACTIVE.value,
                    record.memory_record_id,
                    record.namespace.tenant_id,
                    record.namespace.project_id,
                    record.namespace.subject_id,
                    record.namespace.visibility.value,
                    record.namespace.owner_agent_id,
                    record.kind.value,
                ),
            )
            self._insert_idempotency(
                connection,
                operation_scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-memory-record",
                response_id=record.memory_record_id,
                response_json=record.canonical_json().decode("utf-8"),
            )
            connection.commit()
            return record
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise MemoryConcurrencyError("memory write conflicted") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def forget(
        self,
        memory_id: str,
        *,
        access: MemoryAccessContextV1,
        expected_revision: int,
        reason_code: str,
        idempotency_key: str,
        deleted_at: datetime,
    ) -> MemoryTombstoneV1:
        if expected_revision < 1:
            raise ValueError("forget requires a positive expected revision")
        _require_aware(deleted_at, "memory deletion timestamp")
        self._authorize_access_context(access)
        request_sha256 = _request_sha256(
            {
                "operation": "forget",
                "memory_id": memory_id,
                "access": access,
                "expected_revision": expected_revision,
                "reason_code": reason_code,
                "deleted_at": deleted_at,
            }
        )
        scope = _operation_scope("forget", access)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_replay(
                connection,
                operation_scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-memory-tombstone",
                model_type=MemoryTombstoneV1,
            )
            if replay is not None:
                connection.rollback()
                if not isinstance(replay, MemoryTombstoneV1):
                    raise MemoryIntegrityError("memory idempotency response has the wrong type")
                self._truncate_wal(connection)
                return replay

            head = connection.execute(
                "SELECT * FROM memory_heads WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if head is None:
                raise MemoryNotFoundError("memory authority is missing")
            self._authorize_head(head, access=access)
            if str(head["state"]) == MemoryHeadStateV1.TOMBSTONED.value:
                raise MemoryTombstonedError("memory is already tombstoned")
            if int(head["revision"]) != expected_revision:
                raise MemoryConcurrencyError("memory revision is stale")
            current = self._load_record_row(
                connection.execute(
                    """
                    SELECT * FROM memory_revisions
                    WHERE memory_id = ? AND revision = ?
                    """,
                    (memory_id, expected_revision),
                ).fetchone()
            )
            authorize_namespace(current.namespace, access=access)
            if deleted_at < current.created_at:
                raise MemoryConcurrencyError("memory deletion timestamp cannot precede the current revision")
            tombstone = MemoryTombstoneV1.create(
                memory_id=memory_id,
                final_revision=expected_revision,
                prior_record_ref=current.to_ref(),
                reason_code=reason_code,
                deleted_at=deleted_at,
            )
            revision_rows = connection.execute(
                """
                SELECT memory_id, revision, memory_record_id, object_sha256,
                       predecessor_record_id, content_sha256
                FROM memory_revisions
                WHERE memory_id = ?
                ORDER BY revision
                """,
                (memory_id,),
            ).fetchall()
            if len(revision_rows) != expected_revision:
                raise MemoryIntegrityError("memory revision history is incomplete")
            for revision_row in revision_rows:
                connection.execute(
                    """
                    INSERT INTO erased_memory_revision_digests (
                        memory_id, revision, memory_record_id, object_sha256,
                        predecessor_record_id, content_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(revision_row["memory_id"]),
                        int(revision_row["revision"]),
                        str(revision_row["memory_record_id"]),
                        str(revision_row["object_sha256"]),
                        (
                            str(revision_row["predecessor_record_id"])
                            if revision_row["predecessor_record_id"] is not None
                            else None
                        ),
                        str(revision_row["content_sha256"]),
                    ),
                )
            connection.execute(
                """
                INSERT INTO memory_tombstones (
                    tombstone_id, object_sha256, memory_id, final_revision,
                    prior_record_id, reason_code, deleted_at, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tombstone.tombstone_id,
                    tombstone.object_sha256,
                    tombstone.memory_id,
                    tombstone.final_revision,
                    tombstone.prior_record_ref.object_id,
                    tombstone.reason_code,
                    _timestamp(tombstone.deleted_at),
                    tombstone.canonical_json().decode("utf-8"),
                ),
            )
            connection.execute(
                """
                UPDATE memory_heads
                SET state = ?, tombstone_id = ?
                WHERE memory_id = ? AND revision = ? AND state = ?
                """,
                (
                    MemoryHeadStateV1.TOMBSTONED.value,
                    tombstone.tombstone_id,
                    memory_id,
                    expected_revision,
                    MemoryHeadStateV1.ACTIVE.value,
                ),
            )
            connection.execute(
                "DELETE FROM memory_contents WHERE memory_id = ?",
                (memory_id,),
            )
            connection.execute(
                "DELETE FROM memory_revisions WHERE memory_id = ?",
                (memory_id,),
            )
            record_ids = tuple(str(row["memory_record_id"]) for row in revision_rows)
            connection.executemany(
                """
                DELETE FROM memory_idempotency
                WHERE response_type = ? AND response_id = ?
                """,
                (("agent-memory-record", record_id) for record_id in record_ids),
            )
            self._insert_idempotency(
                connection,
                operation_scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-memory-tombstone",
                response_id=tombstone.tombstone_id,
                response_json=tombstone.canonical_json().decode("utf-8"),
            )
            connection.commit()
            self._truncate_wal(connection)
            return tombstone
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise MemoryConcurrencyError("memory deletion conflicted") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_current(
        self,
        memory_id: str,
        *,
        access: MemoryAccessContextV1,
    ) -> StoredMemoryV1:
        self._authorize_access_context(access)
        with self._read_snapshot() as connection:
            head = connection.execute(
                "SELECT * FROM memory_heads WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if head is None:
                raise MemoryNotFoundError("memory authority is missing")
            self._authorize_head(head, access=access)
            if str(head["state"]) == MemoryHeadStateV1.TOMBSTONED.value:
                self._verify_tombstone_head(connection, head)
                raise MemoryTombstonedError("memory is tombstoned")
            value = self._load_stored_memory(
                connection,
                memory_id=memory_id,
                revision=int(head["revision"]),
            )
            self._verify_head(head, value.record)
            authorize_namespace(value.record.namespace, access=access)
            return value

    def list_current(
        self,
        *,
        access: MemoryAccessContextV1,
    ) -> tuple[StoredMemoryV1, ...]:
        self._authorize_access_context(access)
        with self._read_snapshot() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memory_heads
                WHERE tenant_id = ?
                  AND project_id = ?
                  AND subject_id = ?
                  AND state = ?
                  AND (
                    visibility = ?
                    OR (visibility = ? AND owner_agent_id = ?)
                  )
                ORDER BY memory_id
                """,
                (
                    access.tenant_id,
                    access.project_id,
                    access.subject_id,
                    MemoryHeadStateV1.ACTIVE.value,
                    "PROJECT_SHARED",
                    "AGENT_PRIVATE",
                    access.requester_agent_id,
                ),
            ).fetchall()
            result: list[StoredMemoryV1] = []
            for head in rows:
                value = self._load_stored_memory(
                    connection,
                    memory_id=str(head["memory_id"]),
                    revision=int(head["revision"]),
                )
                self._verify_head(head, value.record)
                authorize_namespace(value.record.namespace, access=access)
                result.append(value)
            return tuple(result)

    def get_tombstone(
        self,
        memory_id: str,
        *,
        access: MemoryAccessContextV1,
    ) -> MemoryTombstoneV1:
        self._authorize_access_context(access)
        with self._read_snapshot() as connection:
            head = connection.execute(
                "SELECT * FROM memory_heads WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if head is None:
                raise MemoryNotFoundError("memory authority is missing")
            self._authorize_head(head, access=access)
            row = connection.execute(
                "SELECT * FROM memory_tombstones WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError("memory tombstone is missing")
            value = self._load_tombstone_row(row)
            self._verify_tombstone_head(connection, head)
            return value

    def content_row_count(self, memory_id: str) -> int:
        with self._read_snapshot() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM memory_contents WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            assert row is not None
            return int(row["count"])

    def erased_revision_digest_count(self, memory_id: str) -> int:
        with self._read_snapshot() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM erased_memory_revision_digests
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
            assert row is not None
            return int(row["count"])

    def _insert_record(
        self,
        connection: sqlite3.Connection,
        *,
        record: MemoryRecordV1,
        content: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_revisions (
                memory_id, revision, memory_record_id, object_sha256,
                predecessor_record_id, tenant_id, project_id, subject_id,
                visibility, owner_agent_id, kind, content_sha256,
                valid_from, valid_until, expires_at,
                importance_basis_points, sensitivity, created_at, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.memory_id,
                record.revision,
                record.memory_record_id,
                record.object_sha256,
                record.predecessor_ref.object_id if record.predecessor_ref else None,
                record.namespace.tenant_id,
                record.namespace.project_id,
                record.namespace.subject_id,
                record.namespace.visibility.value,
                record.namespace.owner_agent_id,
                record.kind.value,
                record.content_sha256,
                _timestamp(record.valid_from),
                _optional_timestamp(record.valid_until),
                _optional_timestamp(record.expires_at),
                record.importance_basis_points,
                record.sensitivity.value,
                _timestamp(record.created_at),
                record.canonical_json().decode("utf-8"),
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_contents (
                memory_id, revision, content_text, content_sha256
            ) VALUES (?, ?, ?, ?)
            """,
            (
                record.memory_id,
                record.revision,
                content,
                record.content_sha256,
            ),
        )

    def _load_stored_memory(
        self,
        connection: sqlite3.Connection,
        *,
        memory_id: str,
        revision: int,
    ) -> StoredMemoryV1:
        record = self._load_record_row(
            connection.execute(
                """
                SELECT * FROM memory_revisions
                WHERE memory_id = ? AND revision = ?
                """,
                (memory_id, revision),
            ).fetchone()
        )
        content_row = connection.execute(
            """
            SELECT * FROM memory_contents
            WHERE memory_id = ? AND revision = ?
            """,
            (memory_id, revision),
        ).fetchone()
        if content_row is None:
            raise MemoryIntegrityError("memory content is missing")
        content = str(content_row["content_text"])
        observed = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if observed != record.content_sha256 or str(content_row["content_sha256"]) != observed:
            raise MemoryIntegrityError("memory content hash drifted")
        try:
            return StoredMemoryV1(record=record, content=content)
        except ValidationError as exc:
            raise MemoryIntegrityError("stored memory is corrupt") from exc

    def _load_record_row(self, row: sqlite3.Row | None) -> MemoryRecordV1:
        if row is None:
            raise MemoryIntegrityError("memory revision is missing")
        try:
            value = MemoryRecordV1.model_validate_json(str(row["record_json"]))
        except ValidationError as exc:
            raise MemoryIntegrityError("memory revision is corrupt") from exc
        expected = (
            value.memory_id,
            value.revision,
            value.memory_record_id,
            value.object_sha256,
            value.predecessor_ref.object_id if value.predecessor_ref else None,
            value.namespace.tenant_id,
            value.namespace.project_id,
            value.namespace.subject_id,
            value.namespace.visibility.value,
            value.namespace.owner_agent_id,
            value.kind.value,
            value.content_sha256,
            _timestamp(value.valid_from),
            _optional_timestamp(value.valid_until),
            _optional_timestamp(value.expires_at),
            value.importance_basis_points,
            value.sensitivity.value,
            _timestamp(value.created_at),
        )
        observed = (
            str(row["memory_id"]),
            int(row["revision"]),
            str(row["memory_record_id"]),
            str(row["object_sha256"]),
            str(row["predecessor_record_id"]) if row["predecessor_record_id"] is not None else None,
            str(row["tenant_id"]),
            str(row["project_id"]),
            str(row["subject_id"]),
            str(row["visibility"]),
            str(row["owner_agent_id"]) if row["owner_agent_id"] is not None else None,
            str(row["kind"]),
            str(row["content_sha256"]),
            str(row["valid_from"]),
            str(row["valid_until"]) if row["valid_until"] is not None else None,
            str(row["expires_at"]) if row["expires_at"] is not None else None,
            int(row["importance_basis_points"]),
            str(row["sensitivity"]),
            str(row["created_at"]),
        )
        if observed != expected:
            raise MemoryIntegrityError("memory revision columns drifted")
        return value

    def _load_tombstone_row(self, row: sqlite3.Row) -> MemoryTombstoneV1:
        try:
            value = MemoryTombstoneV1.model_validate_json(str(row["record_json"]))
        except ValidationError as exc:
            raise MemoryIntegrityError("memory tombstone is corrupt") from exc
        expected = (
            value.tombstone_id,
            value.object_sha256,
            value.memory_id,
            value.final_revision,
            value.prior_record_ref.object_id,
            value.reason_code,
            _timestamp(value.deleted_at),
        )
        observed = (
            str(row["tombstone_id"]),
            str(row["object_sha256"]),
            str(row["memory_id"]),
            int(row["final_revision"]),
            str(row["prior_record_id"]),
            str(row["reason_code"]),
            str(row["deleted_at"]),
        )
        if observed != expected:
            raise MemoryIntegrityError("memory tombstone columns drifted")
        return value

    def _verify_head(
        self,
        head: sqlite3.Row,
        record: MemoryRecordV1,
    ) -> None:
        expected = (
            record.memory_id,
            record.revision,
            MemoryHeadStateV1.ACTIVE.value,
            record.memory_record_id,
            None,
            record.namespace.tenant_id,
            record.namespace.project_id,
            record.namespace.subject_id,
            record.namespace.visibility.value,
            record.namespace.owner_agent_id,
            record.kind.value,
        )
        observed = (
            str(head["memory_id"]),
            int(head["revision"]),
            str(head["state"]),
            str(head["memory_record_id"]),
            str(head["tombstone_id"]) if head["tombstone_id"] is not None else None,
            str(head["tenant_id"]),
            str(head["project_id"]),
            str(head["subject_id"]),
            str(head["visibility"]),
            str(head["owner_agent_id"]) if head["owner_agent_id"] is not None else None,
            str(head["kind"]),
        )
        if observed != expected:
            raise MemoryIntegrityError("memory head drifted")

    def _verify_tombstone_head(
        self,
        connection: sqlite3.Connection,
        head: sqlite3.Row,
    ) -> None:
        row = connection.execute(
            "SELECT * FROM memory_tombstones WHERE memory_id = ?",
            (str(head["memory_id"]),),
        ).fetchone()
        if row is None:
            raise MemoryIntegrityError("memory tombstone head is incomplete")
        tombstone = self._load_tombstone_row(row)
        if (
            int(head["revision"]) != tombstone.final_revision
            or str(head["memory_record_id"]) != tombstone.prior_record_ref.object_id
            or str(head["tombstone_id"]) != tombstone.tombstone_id
        ):
            raise MemoryIntegrityError("memory tombstone head drifted")
        erased = connection.execute(
            """
            SELECT * FROM erased_memory_revision_digests
            WHERE memory_id = ?
            ORDER BY revision
            """,
            (tombstone.memory_id,),
        ).fetchall()
        if len(erased) != tombstone.final_revision:
            raise MemoryIntegrityError("erased memory revision history is incomplete")
        final = erased[-1]
        if (
            int(final["revision"]) != tombstone.final_revision
            or str(final["memory_record_id"]) != tombstone.prior_record_ref.object_id
            or str(final["object_sha256"]) != tombstone.prior_record_ref.object_sha256
        ):
            raise MemoryIntegrityError("erased memory revision digest drifted")
        detailed = connection.execute(
            "SELECT COUNT(*) AS count FROM memory_revisions WHERE memory_id = ?",
            (tombstone.memory_id,),
        ).fetchone()
        content = connection.execute(
            "SELECT COUNT(*) AS count FROM memory_contents WHERE memory_id = ?",
            (tombstone.memory_id,),
        ).fetchone()
        assert detailed is not None and content is not None
        if int(detailed["count"]) != 0 or int(content["count"]) != 0:
            raise MemoryIntegrityError("forgotten memory retained private material")

    @staticmethod
    def _authorize_head(
        head: sqlite3.Row,
        *,
        access: MemoryAccessContextV1,
    ) -> None:
        same_scope = (
            str(head["tenant_id"]) == access.tenant_id
            and str(head["project_id"]) == access.project_id
            and str(head["subject_id"]) == access.subject_id
        )
        private_allowed = str(head["visibility"]) == MemoryVisibilityV1.PROJECT_SHARED.value or (
            str(head["visibility"]) == MemoryVisibilityV1.AGENT_PRIVATE.value
            and str(head["owner_agent_id"]) == access.requester_agent_id
        )
        if not same_scope or not private_allowed:
            raise MemoryAuthorizationError("memory namespace is not authorized")

    def _authorize_access_context(
        self,
        access: MemoryAccessContextV1,
    ) -> None:
        if access not in self._trusted_access_contexts:
            raise MemoryAuthorizationError("memory access context is not trusted")

    def _idempotent_replay(
        self,
        connection: sqlite3.Connection,
        *,
        operation_scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_type: str,
        model_type: type[MemoryRecordV1] | type[MemoryTombstoneV1],
    ) -> MemoryRecordV1 | MemoryTombstoneV1 | None:
        row = connection.execute(
            """
            SELECT * FROM memory_idempotency
            WHERE operation_scope = ? AND idempotency_key = ?
            """,
            (operation_scope, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if str(row["request_sha256"]) != request_sha256 or str(row["response_type"]) != response_type:
            raise MemoryIdempotencyConflictError("memory idempotency key was reused")
        try:
            value = model_type.model_validate_json(str(row["response_json"]))
        except ValidationError as exc:
            raise MemoryIntegrityError("memory idempotency response is corrupt") from exc
        response_id = value.memory_record_id if isinstance(value, MemoryRecordV1) else value.tombstone_id
        if str(row["response_id"]) != response_id:
            raise MemoryIntegrityError("memory idempotency response drifted")
        return value

    @staticmethod
    def _insert_idempotency(
        connection: sqlite3.Connection,
        *,
        operation_scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_type: str,
        response_id: str,
        response_json: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_idempotency (
                operation_scope, idempotency_key, request_sha256,
                response_type, response_id, response_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                operation_scope,
                idempotency_key,
                request_sha256,
                response_type,
                response_id,
                response_json,
            ),
        )

    @staticmethod
    def _truncate_wal(connection: sqlite3.Connection) -> None:
        row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if row is None or int(row[0]) != 0:
            raise MemoryIntegrityError("memory content erasure checkpoint is incomplete")


def _operation_scope(operation: str, access: MemoryAccessContextV1) -> str:
    return ":".join(
        (
            operation,
            access.tenant_id,
            access.project_id,
            access.subject_id,
            access.requester_agent_id,
        )
    )


def _request_sha256(value: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: datetime) -> str:
    rendered = value.isoformat()
    return rendered.removesuffix("+00:00") + "Z" if rendered.endswith("+00:00") else rendered


def _optional_timestamp(value: datetime | None) -> str | None:
    return _timestamp(value) if value is not None else None


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


__all__ = [
    "AgentMemoryStore",
    "AgentMemoryStoreError",
    "MemoryConcurrencyError",
    "MemoryIdempotencyConflictError",
    "MemoryIntegrityError",
    "MemoryNotFoundError",
    "MemoryTombstonedError",
]
