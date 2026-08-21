from __future__ import annotations

from typing import Protocol

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.session_models import (
    SessionCheckpointEventKindV1,
    SessionCheckpointPayloadV1,
    SessionEventV1,
    SessionTeamEventKindV1,
    SessionTeamPayloadV1,
)
from eval_factory.team.models import TeamOutboxKindV1, TeamOutboxRecordV1
from eval_factory.team.store import TeamStore


class TeamSessionEventSink(Protocol):
    def append_team_event(
        self,
        *,
        session_ref: ObjectRef,
        source_outbox_ref: ObjectRef,
        team_authority_version: int,
        payload: SessionTeamPayloadV1 | SessionCheckpointPayloadV1,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> SessionEventV1: ...


class TeamSessionReconciler:
    """Projects TeamStore-first outbox evidence into independent sessions."""

    def __init__(
        self,
        *,
        team_store: TeamStore,
        session_sink: TeamSessionEventSink,
    ) -> None:
        self.team_store = team_store
        self.session_sink = session_sink

    def reconcile(
        self,
        team_id: str,
        *,
        audit: ContractAudit,
        limit: int = 500,
    ) -> tuple[SessionEventV1, ...]:
        committed: list[SessionEventV1] = []
        for outbox, session_ref in self.team_store.pending_outbox(
            team_id,
            limit=limit,
        ):
            source_snapshot = self.team_store.get_snapshot_at(
                outbox.aggregate_ref,
            )
            event = self.session_sink.append_team_event(
                session_ref=session_ref,
                source_outbox_ref=outbox.to_ref(),
                team_authority_version=(source_snapshot.authority.authority_version),
                payload=_session_payload(outbox),
                audit=audit,
                idempotency_key=(f"team-outbox.{outbox.object_sha256}.{session_ref.object_sha256}"),
            )
            self.team_store.mark_outbox_delivered(
                outbox_ref=outbox.to_ref(),
                session_ref=session_ref,
            )
            committed.append(event)
        return tuple(committed)


def _session_payload(
    outbox: TeamOutboxRecordV1,
) -> SessionTeamPayloadV1 | SessionCheckpointPayloadV1:
    if outbox.event_kind is TeamOutboxKindV1.TEAM_CHECKPOINTED:
        return SessionCheckpointPayloadV1(
            event_kind=SessionCheckpointEventKindV1.TEAM_CHECKPOINTED,
            checkpoint_ref=outbox.record_ref,
        )
    kinds = {
        TeamOutboxKindV1.TEAM_CHANGED: SessionTeamEventKindV1.TEAM_CHANGED,
        TeamOutboxKindV1.TEAM_MESSAGE_POSTED: (SessionTeamEventKindV1.TEAM_MESSAGE_POSTED),
        TeamOutboxKindV1.ARTIFACT_HEAD_CHANGED: (SessionTeamEventKindV1.ARTIFACT_HEAD_CHANGED),
        TeamOutboxKindV1.CONFLICT_OPENED: SessionTeamEventKindV1.CONFLICT_OPENED,
        TeamOutboxKindV1.CONFLICT_RESOLVED: (SessionTeamEventKindV1.CONFLICT_RESOLVED),
    }
    return SessionTeamPayloadV1(
        event_kind=kinds[outbox.event_kind],
        record_ref=outbox.record_ref,
    )


__all__ = [
    "TeamSessionEventSink",
    "TeamSessionReconciler",
]
