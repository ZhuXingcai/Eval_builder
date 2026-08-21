from __future__ import annotations

from collections.abc import AsyncIterator

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.requirement_agent import GatewayRequirementAgentLoop
from eval_factory.harness.runtime_models import (
    HarnessEventPageV1,
    HarnessSessionPageV1,
    HarnessSessionProjectionV1,
    HarnessTurnResultV1,
)
from eval_factory.harness.session_models import SessionEventV1
from eval_factory.harness.session_store import HarnessSessionStore


class HarnessSessionService:
    def __init__(
        self,
        *,
        store: HarnessSessionStore,
        requirement_agent: GatewayRequirementAgentLoop,
    ) -> None:
        self.store = store
        self.requirement_agent = requirement_agent

    def create_session(
        self,
        *,
        session_id: str,
        incarnation_id: str,
        composition_ref: ObjectRef,
        created_by: str,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> HarnessSessionProjectionV1:
        return self.store.create_session(
            session_id=session_id,
            incarnation_id=incarnation_id,
            composition_ref=composition_ref,
            created_by=created_by,
            idempotency_key=idempotency_key,
            audit=audit,
        )

    def get_session(self, session_id: str) -> HarnessSessionProjectionV1:
        return self.store.get_projection(session_id)

    def list_sessions(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> HarnessSessionPageV1:
        return self.store.list_sessions(offset=offset, limit=limit)

    def list_events(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> HarnessEventPageV1:
        return self.store.list_events(
            session_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    async def post_message(
        self,
        *,
        session_id: str,
        expected_session_version: int,
        principal_ref: ObjectRef,
        content: str,
        artifact_envelope_refs: tuple[ObjectRef, ...],
        idempotency_key: str,
        audit: ContractAudit,
    ) -> HarnessTurnResultV1:
        start = self.store.start_turn(
            session_id=session_id,
            expected_session_version=expected_session_version,
            principal_ref=principal_ref,
            content=content,
            artifact_envelope_refs=artifact_envelope_refs,
            idempotency_key=idempotency_key,
            audit=audit,
        )
        return await self.requirement_agent.run(start, audit=audit)

    async def stream_turn(
        self,
        *,
        session_id: str,
        expected_session_version: int,
        principal_ref: ObjectRef,
        content: str,
        artifact_envelope_refs: tuple[ObjectRef, ...],
        idempotency_key: str,
        audit: ContractAudit,
    ) -> AsyncIterator[SessionEventV1]:
        start = self.store.start_turn(
            session_id=session_id,
            expected_session_version=expected_session_version,
            principal_ref=principal_ref,
            content=content,
            artifact_envelope_refs=artifact_envelope_refs,
            idempotency_key=idempotency_key,
            audit=audit,
        )
        yield start.user_event
        await self.requirement_agent.run(start, audit=audit)
        page = self.store.list_events(
            session_id,
            after_sequence=start.user_event.sequence,
            limit=500,
        )
        for event in page.events:
            if event.command_ref == start.command.to_ref():
                yield event

    def reconcile(
        self,
        command_ref: ObjectRef,
        *,
        audit: ContractAudit,
    ) -> HarnessTurnResultV1:
        start = self.store.get_turn_start(command_ref)
        return self.requirement_agent.reconcile(start, audit=audit)

    def rebuild_projection(
        self,
        session_id: str,
    ) -> HarnessSessionProjectionV1:
        return self.store.rebuild_projection(session_id)


__all__ = ["HarnessSessionService"]
