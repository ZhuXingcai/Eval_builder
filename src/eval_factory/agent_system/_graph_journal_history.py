from __future__ import annotations

import sqlite3

from eval_factory.agent_system._graph_journal_codec import parse_record
from eval_factory.agent_system._graph_journal_db import (
    FactoryGraphJournalDatabase,
)
from eval_factory.agent_system._graph_journal_types import (
    FactoryGraphJournalConflictError,
    FactoryGraphJournalIntegrityError,
    FactoryGraphJournalNotFoundError,
)
from eval_factory.harness.graph_models import (
    GraphCheckpointPhaseV1,
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)


class FactoryGraphJournalHistory(FactoryGraphJournalDatabase):
    @staticmethod
    def _validate_binding_successor(
        prior: HarnessGraphExecutionBindingV1,
        current: HarnessGraphExecutionBindingV1,
    ) -> None:
        invariant_fields = (
            "session_ref",
            "requirement_ref",
            "requirement_policy_ref",
            "factory_request_ref",
            "factory_policy_ref",
            "pack_manifest_ref",
            "composition_ref",
            "blueprint_ref",
            "thread_id",
            "max_transitions",
        )
        if any(getattr(prior, field) != getattr(current, field) for field in invariant_fields):
            raise FactoryGraphJournalConflictError(
                "Graph binding successor changed pinned authority",
            )

    @staticmethod
    def _validate_checkpoint_successor(
        *,
        prior: HarnessGraphCheckpointV1 | None,
        checkpoint: HarnessGraphCheckpointV1,
    ) -> None:
        if prior is None:
            if (
                checkpoint.phase is not GraphCheckpointPhaseV1.PRE_TRANSITION
                or checkpoint.predecessor_checkpoint_ref is not None
                or checkpoint.transition_number != 1
            ):
                raise FactoryGraphJournalConflictError(
                    "first Graph checkpoint must be transition-one pre",
                )
            return
        if checkpoint.predecessor_checkpoint_ref != prior.to_ref():
            raise FactoryGraphJournalConflictError(
                "Graph checkpoint predecessor is stale",
            )
        if prior.phase is GraphCheckpointPhaseV1.PRE_TRANSITION:
            if (
                checkpoint.phase
                not in {
                    GraphCheckpointPhaseV1.POST_TRANSITION,
                    GraphCheckpointPhaseV1.RECONCILED,
                }
                or checkpoint.transition_number != prior.transition_number
                or checkpoint.node != prior.node
            ):
                raise FactoryGraphJournalConflictError(
                    "pre checkpoint requires matching post/reconciled successor",
                )
        else:
            is_reconciliation = (
                checkpoint.phase is GraphCheckpointPhaseV1.RECONCILED
                and checkpoint.transition_number == prior.transition_number
                and checkpoint.node == prior.node
            )
            is_next_pre = (
                checkpoint.phase is GraphCheckpointPhaseV1.PRE_TRANSITION
                and checkpoint.transition_number == prior.transition_number + 1
            )
            if not (is_reconciliation or is_next_pre):
                raise FactoryGraphJournalConflictError(
                    "terminal checkpoint requires reconciliation or next-transition pre",
                )

    def _current_binding_or_none(
        self,
        connection: sqlite3.Connection,
        binding_id: str,
    ) -> HarnessGraphExecutionBindingV1 | None:
        row = connection.execute(
            """
            SELECT object_id FROM graph_binding_current_heads
            WHERE binding_id = ?
            """,
            (binding_id,),
        ).fetchone()
        return self._load_binding(connection, str(row["object_id"])) if row is not None else None

    def _current_checkpoint_or_none(
        self,
        connection: sqlite3.Connection,
        binding_id: str,
    ) -> HarnessGraphCheckpointV1 | None:
        row = connection.execute(
            """
            SELECT checkpoint_object_id
            FROM graph_checkpoint_current_heads
            WHERE binding_id = ?
            """,
            (binding_id,),
        ).fetchone()
        return (
            self._load_checkpoint(
                connection,
                str(row["checkpoint_object_id"]),
            )
            if row is not None
            else None
        )

    def _load_binding(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> HarnessGraphExecutionBindingV1:
        row = connection.execute(
            """
            SELECT * FROM graph_execution_bindings
            WHERE object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryGraphJournalNotFoundError(
                f"Graph binding not found: {object_id}",
            )
        value = parse_record(
            HarnessGraphExecutionBindingV1,
            str(row["record_json"]),
            "Graph binding",
        )
        if (
            value.object_sha256 != row["object_sha256"]
            or value.binding_id != row["binding_id"]
            or value.thread_id != row["thread_id"]
        ):
            raise FactoryGraphJournalIntegrityError(
                "Graph binding columns differ from immutable record",
            )
        return value

    def _load_checkpoint(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> HarnessGraphCheckpointV1:
        row = connection.execute(
            """
            SELECT * FROM graph_checkpoints
            WHERE object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryGraphJournalNotFoundError(
                f"Graph checkpoint not found: {object_id}",
            )
        value = parse_record(
            HarnessGraphCheckpointV1,
            str(row["record_json"]),
            "Graph checkpoint",
        )
        binding = self._load_binding(
            connection,
            str(row["binding_object_id"]),
        )
        if (
            value.object_sha256 != row["object_sha256"]
            or value.binding_ref != binding.to_ref()
            or value.phase.value != row["phase"]
            or value.node != row["node"]
            or value.transition_number != row["transition_number"]
            or (value.predecessor_checkpoint_ref.object_id if value.predecessor_checkpoint_ref else None)
            != row["predecessor_object_id"]
        ):
            raise FactoryGraphJournalIntegrityError(
                "Graph checkpoint columns differ from immutable record",
            )
        return value

    def _replay(
        self,
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_type: str,
    ) -> str | None:
        row = connection.execute(
            """
            SELECT * FROM graph_journal_idempotency
            WHERE scope = ? AND idempotency_key = ?
            """,
            (scope, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if row["request_sha256"] != request_sha256 or row["response_type"] != response_type:
            raise FactoryGraphJournalConflictError(
                "Graph journal idempotency request changed",
            )
        return str(row["response_id"])

    @staticmethod
    def _remember(
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_type: str,
        response_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO graph_journal_idempotency (
                scope, idempotency_key, request_sha256,
                response_type, response_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                scope,
                idempotency_key,
                request_sha256,
                response_type,
                response_id,
            ),
        )

    def _rebuild_binding_heads(
        self,
        connection: sqlite3.Connection,
    ) -> dict[str, str]:
        heads: dict[str, str] = {}
        rows = connection.execute(
            """
            SELECT binding_id, object_id, record_json
            FROM graph_execution_bindings
            ORDER BY rowid
            """,
        ).fetchall()
        prior_by_id: dict[
            str,
            HarnessGraphExecutionBindingV1,
        ] = {}
        for row in rows:
            value = parse_record(
                HarnessGraphExecutionBindingV1,
                str(row["record_json"]),
                "Graph binding",
            )
            prior = prior_by_id.get(value.binding_id)
            if prior is not None:
                self._validate_binding_successor(prior, value)
            prior_by_id[value.binding_id] = value
            heads[value.binding_id] = value.object_id
        return heads

    def _rebuild_checkpoint_heads(
        self,
        connection: sqlite3.Connection,
    ) -> dict[str, str]:
        heads: dict[str, str] = {}
        prior_by_id: dict[str, HarnessGraphCheckpointV1] = {}
        rows = connection.execute(
            """
            SELECT binding_id, record_json FROM graph_checkpoints
            ORDER BY rowid
            """,
        ).fetchall()
        for row in rows:
            binding_id = str(row["binding_id"])
            value = parse_record(
                HarnessGraphCheckpointV1,
                str(row["record_json"]),
                "Graph checkpoint",
            )
            self._validate_checkpoint_successor(
                prior=prior_by_id.get(binding_id),
                checkpoint=value,
            )
            prior_by_id[binding_id] = value
            heads[binding_id] = value.object_id
        return heads


__all__ = ["FactoryGraphJournalHistory"]
