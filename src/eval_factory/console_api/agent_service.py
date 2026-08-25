from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from eval_factory.console_api.agent_contracts import (
    AgentShellCloseSessionCommandV1,
    AgentShellCreateSessionCommandV1,
    AgentShellDeliverySummaryV1,
    AgentShellEventFamilyV1,
    AgentShellEventPageV1,
    AgentShellEventV1,
    AgentShellExecutionPhaseV1,
    AgentShellMemberProjectionV1,
    AgentShellPostMessageCommandV1,
    AgentShellProjectionV1,
    AgentShellReconcileCommandV1,
    AgentShellSessionPageV1,
    AgentShellSessionSummaryV1,
    AgentShellTeamSummaryV1,
    AgentShellTranscriptEntryV1,
    AgentShellWorkspaceDescriptorV1,
    AgentShellWorkspaceKindV1,
    AgentShellWorkspaceStatusV1,
)
from eval_factory.console_api.composition_types import (
    AgentShellCompositionNotConfiguredError,
    AgentShellCompositionNotFoundError,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    HarnessSessionConflictError,
    HarnessSessionProjectionV1,
    HarnessSessionService,
    SessionEventV1,
    SessionRuntimePayloadV1,
)

if TYPE_CHECKING:
    from eval_factory.console_api.composition_service import (
        AgentShellCompositionService,
    )


class AgentShellService:
    def __init__(
        self,
        *,
        sessions: HarnessSessionService,
        composition_ref: ObjectRef,
        principal_resolver: Callable[[str], ObjectRef],
        governing_versions: tuple[VersionBinding, ...],
        composition: AgentShellCompositionService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._sessions = sessions
        self._composition_ref = composition_ref
        self._principal_resolver = principal_resolver
        self._governing_versions = governing_versions
        self._composition = composition
        self._clock = clock or (lambda: datetime.now(UTC))

    def create_session(
        self,
        command: AgentShellCreateSessionCommandV1,
        *,
        principal: str,
    ) -> AgentShellProjectionV1:
        projection = self._sessions.create_session(
            session_id=command.session_id,
            incarnation_id=command.incarnation_id,
            composition_ref=self._composition_ref,
            created_by=principal,
            idempotency_key=command.idempotency_key,
            audit=self._audit(principal, (self._composition_ref,)),
        )
        return self._projection(projection)

    def close_session(
        self,
        session_id: str,
        command: AgentShellCloseSessionCommandV1,
        *,
        principal: str,
    ) -> AgentShellProjectionV1:
        principal_ref = self._principal_resolver(principal)
        current = self._sessions.get_session(session_id)
        projection = self._sessions.close_session(
            session_id=session_id,
            expected_session_version=command.expected_session_version,
            principal_ref=principal_ref,
            idempotency_key=command.idempotency_key,
            audit=self._audit(
                principal,
                (principal_ref, current.session.session_ref),
            ),
        )
        return self._projection(projection)

    def get_session(self, session_id: str) -> AgentShellProjectionV1:
        return self._projection(self._sessions.get_session(session_id))

    def list_sessions(
        self,
        *,
        offset: int,
        limit: int,
    ) -> AgentShellSessionPageV1:
        page = self._sessions.list_sessions(offset=offset, limit=limit)
        summaries = tuple(
            self._summary(
                self._sessions.get_session(state.session_id),
            )
            for state in page.sessions
        )
        return AgentShellSessionPageV1(
            offset=page.offset,
            limit=page.limit,
            total=page.total,
            sessions=summaries,
        )

    async def post_message(
        self,
        session_id: str,
        command: AgentShellPostMessageCommandV1,
        *,
        principal: str,
    ) -> AgentShellProjectionV1:
        principal_ref = self._principal_resolver(principal)
        if self._composition is not None:
            self._composition.before_turn(
                session_id,
                idempotency_key=command.idempotency_key,
            )
        audit = self._audit(
            principal,
            (principal_ref, *command.artifact_envelope_refs),
        )
        turn = await self._sessions.post_message(
            session_id=session_id,
            expected_session_version=command.expected_session_version,
            principal_ref=principal_ref,
            content=command.content,
            artifact_envelope_refs=command.artifact_envelope_refs,
            idempotency_key=command.idempotency_key,
            audit=audit,
        )
        if self._composition is not None:
            await self._composition.after_turn(
                session_id,
                turn=turn,
                principal=principal,
                audit=audit,
            )
        return self.get_session(session_id)

    async def reconcile(
        self,
        session_id: str,
        command: AgentShellReconcileCommandV1,
        *,
        principal: str,
    ) -> AgentShellProjectionV1:
        composition = self._require_composition()
        principal_ref = self._principal_resolver(principal)
        claim = self._sessions.claim_shell_reconcile(
            session_id=session_id,
            expected_session_version=command.expected_session_version,
            principal_ref=principal_ref,
            idempotency_key=command.idempotency_key,
            effect_ref=composition.reconcile_binding_ref(session_id),
        )
        projection = self._sessions.get_session(session_id)
        input_refs = [
            principal_ref,
            projection.session.session_ref,
        ]
        if claim.effect_ref is not None:
            input_refs.append(claim.effect_ref)
        await composition.reconcile(
            session_id,
            expected_binding_ref=claim.effect_ref,
            audit=self._audit(
                principal,
                tuple(input_refs),
            ),
        )
        return self.get_session(session_id)

    def team(self, session_id: str) -> AgentShellTeamSummaryV1:
        self._require_composition()
        projection = self.get_session(session_id)
        if projection.team is None:
            raise AgentShellCompositionNotFoundError(
                "Agent Shell Team projection is unavailable",
            )
        return projection.team

    def member(
        self,
        session_id: str,
        member_id: str,
    ) -> AgentShellMemberProjectionV1:
        return self._require_composition().member_projection(
            session_id,
            member_id,
        )

    def workspace(
        self,
        session_id: str,
        kind: AgentShellWorkspaceKindV1,
    ) -> AgentShellWorkspaceDescriptorV1:
        self._require_composition()
        projection = self.get_session(session_id)
        matches = tuple(value for value in projection.workspaces if value.kind is kind)
        if len(matches) != 1:
            raise AgentShellCompositionNotFoundError(
                "Agent Shell workspace is unavailable",
            )
        return matches[0]

    def delivery(
        self,
        session_id: str,
    ) -> AgentShellDeliverySummaryV1:
        self._require_composition()
        projection = self.get_session(session_id)
        if projection.delivery is None:
            raise AgentShellCompositionNotFoundError(
                "Agent Shell delivery is unavailable",
            )
        return projection.delivery

    def list_events(
        self,
        session_id: str,
        *,
        after_sequence: int,
        limit: int,
    ) -> AgentShellEventPageV1:
        projection = self._sessions.get_session(session_id)
        watermark = projection.session.last_event_sequence
        if after_sequence > watermark:
            raise HarnessSessionConflictError(
                "event cursor is ahead of committed session history",
            )
        return self._event_page(
            session_id,
            after_sequence=after_sequence,
            limit=limit,
            watermark=watermark,
        )

    def _event_page(
        self,
        session_id: str,
        *,
        after_sequence: int,
        limit: int,
        watermark: int,
    ) -> AgentShellEventPageV1:
        page = self._sessions.list_events(
            session_id,
            after_sequence=after_sequence,
            limit=limit,
        )
        events = tuple(event for event in page.events if event.sequence <= watermark)
        return AgentShellEventPageV1(
            session_id=session_id,
            after_sequence=after_sequence,
            watermark=watermark,
            events=tuple(self._event(event) for event in events),
            next_sequence=(events[-1].sequence + 1 if events and events[-1].sequence < watermark else None),
        )

    def _projection(
        self,
        value: HarnessSessionProjectionV1,
    ) -> AgentShellProjectionV1:
        page = self._event_page(
            value.session.session_id,
            after_sequence=max(0, value.session.last_event_sequence - 500),
            limit=500,
            watermark=value.session.last_event_sequence,
        )
        composition = self._composition.project(value) if self._composition is not None else None
        return AgentShellProjectionV1(
            session=self._summary(value),
            transcript=tuple(
                AgentShellTranscriptEntryV1(
                    message_id=item.message_ref.object_id,
                    role=item.role,
                    content=item.content,
                    artifact_envelope_refs=item.artifact_envelope_refs,
                    created_at=item.created_at,
                )
                for item in value.transcript
            ),
            activity=page.events,
            requirement_outcome=(composition.requirement_outcome if composition is not None else None),
            evidence_class=(composition.evidence_class if composition is not None else None),
            source_admission_ref=(composition.source_admission_ref if composition is not None else None),
            execution_phase=(
                composition.execution_phase
                if composition is not None
                else AgentShellExecutionPhaseV1.NOT_CONFIGURED
            ),
            factory=(composition.factory if composition is not None else None),
            graph=(composition.graph if composition is not None else None),
            team=(composition.team if composition is not None else None),
            plan_reviews=(composition.plan_reviews if composition is not None else ()),
            pending_interactions=(composition.pending_interactions if composition is not None else ()),
            workspaces=(composition.workspaces if composition is not None else _unconfigured_workspaces()),
            delivery=(composition.delivery if composition is not None else None),
            reconnect_cursor=value.session.last_event_sequence,
            source_fingerprint=(
                composition.source_fingerprint if composition is not None else value.source_fingerprint
            ),
        )

    def _summary(
        self,
        value: HarnessSessionProjectionV1,
    ) -> AgentShellSessionSummaryV1:
        title = next(
            (item.content[:160] for item in value.transcript if item.role.value == "USER"),
            None,
        )
        return AgentShellSessionSummaryV1(
            session_id=value.session.session_id,
            title=title,
            status=value.session.status,
            session_version=value.session.session_version,
            last_event_sequence=value.session.last_event_sequence,
            created_at=value.session.created_at,
            updated_at=value.session.updated_at,
        )

    def _event(self, value: SessionEventV1) -> AgentShellEventV1:
        payload = value.payload
        if isinstance(payload, SessionRuntimePayloadV1):
            runtime = self._sessions.get_runtime_event(
                payload.runtime_event_ref,
            )
            return AgentShellEventV1(
                event_id=value.event_id,
                sequence=value.sequence,
                authority_version=value.authority_version,
                family=AgentShellEventFamilyV1.RUNTIME,
                event_kind=runtime.kind,
                occurred_at=value.occurred_at,
                turn_id=value.turn_id,
                step_id=value.step_id,
                runtime_id=runtime.runtime_id,
                tool_family=runtime.tool_family,
                usage=runtime.usage,
                failure_code=runtime.failure_code,
            )
        return AgentShellEventV1(
            event_id=value.event_id,
            sequence=value.sequence,
            authority_version=value.authority_version,
            family=AgentShellEventFamilyV1(payload.family),
            event_kind=payload.event_kind,
            occurred_at=value.occurred_at,
            turn_id=value.turn_id,
            step_id=value.step_id,
        )

    def _audit(
        self,
        principal: str,
        input_refs: tuple[ObjectRef, ...],
    ) -> ContractAudit:
        return ContractAudit(
            created_at=self._clock(),
            created_by=principal,
            governing_versions=self._governing_versions,
            input_refs=tuple(
                sorted(
                    set(input_refs),
                    key=lambda item: (
                        item.object_type,
                        item.object_id,
                        item.object_version,
                        item.object_sha256,
                    ),
                )
            ),
        )

    def _require_composition(self) -> AgentShellCompositionService:
        if self._composition is None:
            raise AgentShellCompositionNotConfiguredError(
                "Agent Shell composition is not configured",
            )
        return self._composition


def _unconfigured_workspaces() -> tuple[
    AgentShellWorkspaceDescriptorV1,
    ...,
]:
    return tuple(
        AgentShellWorkspaceDescriptorV1(
            kind=kind,
            status=(
                AgentShellWorkspaceStatusV1.AVAILABLE
                if kind
                in {
                    AgentShellWorkspaceKindV1.CONVERSATION,
                    AgentShellWorkspaceKindV1.ACTIVITY,
                }
                else AgentShellWorkspaceStatusV1.EMPTY
            ),
            owner_refs=(),
            item_count=0,
            reason_codes=(),
        )
        for kind in AgentShellWorkspaceKindV1
    )


__all__ = ["AgentShellService"]
