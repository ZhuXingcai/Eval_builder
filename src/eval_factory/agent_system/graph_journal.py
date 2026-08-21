from __future__ import annotations

from eval_factory.agent_system._graph_journal_codec import (
    parse_record,
    record_json,
    request_sha256,
)
from eval_factory.agent_system._graph_journal_history import (
    FactoryGraphJournalHistory,
)
from eval_factory.agent_system._graph_journal_types import (
    FactoryGraphCheckpointDelivery,
    FactoryGraphJournalConflictError,
    FactoryGraphJournalError,
    FactoryGraphJournalFaultPoint,
    FactoryGraphJournalHeadRebuild,
    FactoryGraphJournalInjectedCrash,
    FactoryGraphJournalIntegrityError,
    FactoryGraphJournalNotFoundError,
    StaticFactoryGraphJournalFaultInjector,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.harness.graph_models import (
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)


class FactoryGraphJournalStore(FactoryGraphJournalHistory):
    def commit_binding(
        self,
        binding: HarnessGraphExecutionBindingV1,
        *,
        idempotency_key: str,
    ) -> HarnessGraphExecutionBindingV1:
        digest = request_sha256(binding.to_ref())
        scope = f"graph-binding:{binding.binding_id}"
        with self._write() as connection:
            replay = self._replay(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=digest,
                response_type="GRAPH_BINDING",
            )
            if replay is not None:
                return self._load_binding(connection, replay)
            prior = self._current_binding_or_none(
                connection,
                binding.binding_id,
            )
            if prior is not None:
                self._validate_binding_successor(prior, binding)
                if prior == binding:
                    self._remember(
                        connection,
                        scope=scope,
                        idempotency_key=idempotency_key,
                        request_sha256=digest,
                        response_type="GRAPH_BINDING",
                        response_id=prior.object_id,
                    )
                    return prior
            connection.execute(
                """
                INSERT INTO graph_execution_bindings (
                    object_id, object_sha256, binding_id,
                    thread_id, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    binding.object_id,
                    binding.object_sha256,
                    binding.binding_id,
                    binding.thread_id,
                    record_json(binding),
                ),
            )
            self._fault(
                FactoryGraphJournalFaultPoint.AFTER_BINDING_RECORD,
            )
            connection.execute(
                """
                INSERT INTO graph_binding_current_heads (
                    binding_id, object_id
                ) VALUES (?, ?)
                ON CONFLICT(binding_id) DO UPDATE SET
                    object_id = excluded.object_id
                """,
                (binding.binding_id, binding.object_id),
            )
            self._fault(
                FactoryGraphJournalFaultPoint.AFTER_BINDING_HEAD,
            )
            self._remember(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=digest,
                response_type="GRAPH_BINDING",
                response_id=binding.object_id,
            )
            self._fault(
                FactoryGraphJournalFaultPoint.AFTER_IDEMPOTENCY,
            )
            return binding

    def get_binding(
        self,
        binding_id: str,
    ) -> HarnessGraphExecutionBindingV1:
        with self._read() as connection:
            binding = self._current_binding_or_none(
                connection,
                binding_id,
            )
            if binding is None:
                raise FactoryGraphJournalNotFoundError(
                    f"Graph binding not found: {binding_id}",
                )
            return binding

    def get_binding_by_ref(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1:
        if reference.object_type != "graph-execution-binding":
            raise FactoryGraphJournalIntegrityError(
                "Graph binding ref has the wrong type",
            )
        with self._read() as connection:
            value = self._load_binding(connection, reference.object_id)
            if value.to_ref() != reference:
                raise FactoryGraphJournalIntegrityError(
                    "Graph binding differs from its reference",
                )
            return value

    def commit_checkpoint(
        self,
        checkpoint: HarnessGraphCheckpointV1,
        *,
        idempotency_key: str,
    ) -> HarnessGraphCheckpointV1:
        binding = self.get_binding_by_ref(checkpoint.binding_ref)
        digest = request_sha256(checkpoint.to_ref())
        scope = f"graph-checkpoint:{binding.binding_id}"
        with self._write() as connection:
            replay = self._replay(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=digest,
                response_type="GRAPH_CHECKPOINT",
            )
            if replay is not None:
                return self._load_checkpoint(connection, replay)
            current_binding = self._current_binding_or_none(
                connection,
                binding.binding_id,
            )
            if current_binding is None or current_binding.to_ref() != binding.to_ref():
                raise FactoryGraphJournalConflictError(
                    "Graph checkpoint uses a stale binding",
                )
            prior = self._current_checkpoint_or_none(
                connection,
                binding.binding_id,
            )
            self._validate_checkpoint_successor(
                prior=prior,
                checkpoint=checkpoint,
            )
            connection.execute(
                """
                INSERT INTO graph_checkpoints (
                    object_id, object_sha256, binding_id,
                    binding_object_id, phase, node,
                    transition_number, predecessor_object_id,
                    record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.object_id,
                    checkpoint.object_sha256,
                    binding.binding_id,
                    binding.object_id,
                    checkpoint.phase.value,
                    checkpoint.node,
                    checkpoint.transition_number,
                    (
                        checkpoint.predecessor_checkpoint_ref.object_id
                        if checkpoint.predecessor_checkpoint_ref
                        else None
                    ),
                    record_json(checkpoint),
                ),
            )
            self._fault(
                FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_RECORD,
            )
            connection.execute(
                """
                INSERT INTO graph_checkpoint_current_heads (
                    binding_id, checkpoint_object_id
                ) VALUES (?, ?)
                ON CONFLICT(binding_id) DO UPDATE SET
                    checkpoint_object_id = excluded.checkpoint_object_id
                """,
                (binding.binding_id, checkpoint.object_id),
            )
            self._fault(
                FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_HEAD,
            )
            event_id = f"graph-checkpoint-event://{checkpoint.object_sha256}"
            connection.execute(
                """
                INSERT INTO graph_checkpoint_outbox (
                    event_id, binding_id, checkpoint_object_id,
                    session_ref_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    event_id,
                    binding.binding_id,
                    checkpoint.object_id,
                    binding.session_ref.model_dump_json(),
                ),
            )
            self._fault(
                FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_OUTBOX,
            )
            self._remember(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=digest,
                response_type="GRAPH_CHECKPOINT",
                response_id=checkpoint.object_id,
            )
            self._fault(
                FactoryGraphJournalFaultPoint.AFTER_IDEMPOTENCY,
            )
            return checkpoint

    def current_checkpoint(
        self,
        binding_id: str,
    ) -> HarnessGraphCheckpointV1 | None:
        with self._read() as connection:
            return self._current_checkpoint_or_none(
                connection,
                binding_id,
            )

    def get_checkpoint_by_ref(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphCheckpointV1:
        if reference.object_type != "graph-checkpoint":
            raise FactoryGraphJournalIntegrityError(
                "Graph checkpoint ref has the wrong type",
            )
        with self._read() as connection:
            value = self._load_checkpoint(
                connection,
                reference.object_id,
            )
            if value.to_ref() != reference:
                raise FactoryGraphJournalIntegrityError(
                    "Graph checkpoint differs from its reference",
                )
            return value

    def list_checkpoints(
        self,
        binding_id: str,
    ) -> tuple[HarnessGraphCheckpointV1, ...]:
        with self._read() as connection:
            rows = connection.execute(
                """
                SELECT record_json FROM graph_checkpoints
                WHERE binding_id = ?
                ORDER BY transition_number, rowid
                """,
                (binding_id,),
            ).fetchall()
            return tuple(
                parse_record(
                    HarnessGraphCheckpointV1,
                    str(row["record_json"]),
                    "Graph checkpoint",
                )
                for row in rows
            )

    def pending_deliveries(
        self,
        binding_id: str,
        *,
        limit: int = 500,
    ) -> tuple[FactoryGraphCheckpointDelivery, ...]:
        if not 1 <= limit <= 500:
            raise ValueError("Graph delivery limit is out of bounds")
        with self._read() as connection:
            rows = connection.execute(
                """
                SELECT o.checkpoint_object_id, o.session_ref_json
                FROM graph_checkpoint_outbox o
                LEFT JOIN graph_checkpoint_deliveries d
                  ON d.checkpoint_object_id = o.checkpoint_object_id
                 AND d.session_object_id = json_extract(
                     o.session_ref_json, '$.object_id'
                 )
                WHERE o.binding_id = ?
                  AND d.checkpoint_object_id IS NULL
                ORDER BY o.rowid
                LIMIT ?
                """,
                (binding_id, limit),
            ).fetchall()
            return tuple(
                FactoryGraphCheckpointDelivery(
                    checkpoint=self._load_checkpoint(
                        connection,
                        str(row["checkpoint_object_id"]),
                    ),
                    session_ref=ObjectRef.model_validate_json(
                        str(row["session_ref_json"]),
                    ),
                )
                for row in rows
            )

    def mark_delivered(
        self,
        *,
        checkpoint_ref: ObjectRef,
        session_ref: ObjectRef,
    ) -> None:
        with self._write() as connection:
            checkpoint = self._load_checkpoint(
                connection,
                checkpoint_ref.object_id,
            )
            if checkpoint.to_ref() != checkpoint_ref:
                raise FactoryGraphJournalIntegrityError(
                    "Graph delivery checkpoint ref drifted",
                )
            binding = self._load_binding(
                connection,
                checkpoint.binding_ref.object_id,
            )
            if binding.session_ref != session_ref:
                raise FactoryGraphJournalConflictError(
                    "Graph checkpoint delivery uses another session",
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO graph_checkpoint_deliveries (
                    checkpoint_object_id, session_object_id
                ) VALUES (?, ?)
                """,
                (checkpoint.object_id, session_ref.object_id),
            )
            self._fault(
                FactoryGraphJournalFaultPoint.AFTER_DELIVERY,
            )

    def rebuild_current_heads(
        self,
        *,
        repair: bool = False,
    ) -> FactoryGraphJournalHeadRebuild:
        with self._write() as connection:
            bindings = self._rebuild_binding_heads(connection)
            checkpoints = self._rebuild_checkpoint_heads(connection)
            observed_bindings = {
                str(row["binding_id"]): str(row["object_id"])
                for row in connection.execute(
                    "SELECT binding_id, object_id FROM graph_binding_current_heads",
                )
            }
            observed_checkpoints = {
                str(row["binding_id"]): str(
                    row["checkpoint_object_id"],
                )
                for row in connection.execute(
                    "SELECT binding_id, checkpoint_object_id FROM graph_checkpoint_current_heads",
                )
            }
            if observed_bindings != bindings or observed_checkpoints != checkpoints:
                if not repair:
                    raise FactoryGraphJournalIntegrityError(
                        "Graph journal current heads differ from history",
                    )
                connection.execute(
                    "DELETE FROM graph_binding_current_heads",
                )
                connection.executemany(
                    """
                    INSERT INTO graph_binding_current_heads (
                        binding_id, object_id
                    ) VALUES (?, ?)
                    """,
                    sorted(bindings.items()),
                )
                connection.execute(
                    "DELETE FROM graph_checkpoint_current_heads",
                )
                connection.executemany(
                    """
                    INSERT INTO graph_checkpoint_current_heads (
                        binding_id, checkpoint_object_id
                    ) VALUES (?, ?)
                    """,
                    sorted(checkpoints.items()),
                )
            return FactoryGraphJournalHeadRebuild(
                binding_count=len(bindings),
                checkpoint_count=len(checkpoints),
            )


__all__ = [
    "FactoryGraphCheckpointDelivery",
    "FactoryGraphJournalConflictError",
    "FactoryGraphJournalError",
    "FactoryGraphJournalFaultPoint",
    "FactoryGraphJournalHeadRebuild",
    "FactoryGraphJournalInjectedCrash",
    "FactoryGraphJournalIntegrityError",
    "FactoryGraphJournalNotFoundError",
    "FactoryGraphJournalStore",
    "StaticFactoryGraphJournalFaultInjector",
]
