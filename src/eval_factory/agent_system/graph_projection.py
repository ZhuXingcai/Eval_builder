from __future__ import annotations

from typing import Protocol

from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.contracts import static_object_ref
from eval_factory.harness.session_models import (
    SessionCheckpointEventKindV1,
    SessionCheckpointPayloadV1,
    SessionEventV1,
)


class FactoryGraphSessionEventSink(Protocol):
    def append_team_event(
        self,
        *,
        session_ref: ObjectRef,
        source_outbox_ref: ObjectRef,
        team_authority_version: int,
        payload: SessionCheckpointPayloadV1,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> SessionEventV1: ...


class FactoryGraphSessionReconciler:
    """Projects graph-journal outbox records into Harness sessions."""

    def __init__(
        self,
        *,
        journal: FactoryGraphJournalStore,
        session_sink: FactoryGraphSessionEventSink,
    ) -> None:
        self._journal = journal
        self._session_sink = session_sink

    def reconcile(
        self,
        binding_id: str,
        *,
        audit: ContractAudit,
        limit: int = 500,
    ) -> tuple[SessionEventV1, ...]:
        committed: list[SessionEventV1] = []
        for delivery in self._journal.pending_deliveries(
            binding_id,
            limit=limit,
        ):
            checkpoint = delivery.checkpoint
            binding = self._journal.get_binding_by_ref(
                checkpoint.binding_ref,
            )
            source_outbox_ref = static_object_ref(
                object_type="graph-checkpoint-outbox",
                object_id=(f"graph-checkpoint-outbox://{checkpoint.object_sha256}"),
                object_version="v1",
                payload={
                    "checkpoint_ref": checkpoint.to_ref(),
                    "session_ref": delivery.session_ref,
                },
            )
            event = self._session_sink.append_team_event(
                session_ref=delivery.session_ref,
                source_outbox_ref=source_outbox_ref,
                team_authority_version=(binding.expected_authority_version),
                payload=SessionCheckpointPayloadV1(
                    event_kind=(SessionCheckpointEventKindV1.GRAPH_CHECKPOINTED),
                    checkpoint_ref=checkpoint.to_ref(),
                ),
                audit=audit,
                idempotency_key=(
                    f"graph-checkpoint.{checkpoint.object_sha256}.{delivery.session_ref.object_sha256}"
                ),
            )
            self._journal.mark_delivered(
                checkpoint_ref=checkpoint.to_ref(),
                session_ref=delivery.session_ref,
            )
            committed.append(event)
        return tuple(committed)


__all__ = [
    "FactoryGraphSessionEventSink",
    "FactoryGraphSessionReconciler",
]
