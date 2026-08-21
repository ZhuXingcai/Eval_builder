from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from eval_factory.agent_system._graph_journal_types import (
    FactoryGraphJournalFaultPoint,
)


class FactoryGraphJournalDatabase:
    def __init__(
        self,
        path: Path,
        *,
        authority_paths: tuple[Path, ...] = (),
        fault_injector: (Callable[[FactoryGraphJournalFaultPoint], None] | None) = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for authority_path in authority_paths:
            if self.path == authority_path.expanduser().resolve():
                raise ValueError(
                    "Graph journal must be separate from authority stores",
                )
        self._fault = fault_injector or (lambda _point: None)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.rollback()
            connection.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
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

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_execution_bindings (
                    object_id TEXT PRIMARY KEY,
                    object_sha256 TEXT NOT NULL,
                    binding_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(binding_id, object_sha256)
                );

                CREATE TABLE IF NOT EXISTS graph_binding_current_heads (
                    binding_id TEXT PRIMARY KEY,
                    object_id TEXT NOT NULL UNIQUE
                        REFERENCES graph_execution_bindings(object_id)
                );

                CREATE TABLE IF NOT EXISTS graph_checkpoints (
                    object_id TEXT PRIMARY KEY,
                    object_sha256 TEXT NOT NULL,
                    binding_id TEXT NOT NULL,
                    binding_object_id TEXT NOT NULL
                        REFERENCES graph_execution_bindings(object_id),
                    phase TEXT NOT NULL,
                    node TEXT NOT NULL,
                    transition_number INTEGER NOT NULL,
                    predecessor_object_id TEXT,
                    record_json TEXT NOT NULL,
                    UNIQUE(binding_id, transition_number, phase)
                );

                CREATE TABLE IF NOT EXISTS graph_checkpoint_current_heads (
                    binding_id TEXT PRIMARY KEY,
                    checkpoint_object_id TEXT NOT NULL UNIQUE
                        REFERENCES graph_checkpoints(object_id)
                );

                CREATE TABLE IF NOT EXISTS graph_checkpoint_outbox (
                    event_id TEXT PRIMARY KEY,
                    binding_id TEXT NOT NULL,
                    checkpoint_object_id TEXT NOT NULL
                        REFERENCES graph_checkpoints(object_id),
                    session_ref_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_checkpoint_deliveries (
                    checkpoint_object_id TEXT NOT NULL
                        REFERENCES graph_checkpoints(object_id),
                    session_object_id TEXT NOT NULL,
                    PRIMARY KEY(checkpoint_object_id, session_object_id)
                );

                CREATE TABLE IF NOT EXISTS graph_journal_idempotency (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_type TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    PRIMARY KEY(scope, idempotency_key)
                );
                """
            )
            connection.commit()
        finally:
            connection.close()


__all__ = ["FactoryGraphJournalDatabase"]
