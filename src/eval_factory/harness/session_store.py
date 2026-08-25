from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from env_mock_agent.facade import UnifiedRuntimeEventV2
from eval_factory.contracts.core import ContractAudit, ContractModel, ObjectRef
from eval_factory.harness.contracts import sorted_refs
from eval_factory.harness.runtime_models import (
    GatewayJournalStateV1,
    HarnessEventPageV1,
    HarnessGatewayJournalV1,
    HarnessGatewaySummaryV1,
    HarnessMessageCommandV1,
    HarnessMessageRoleV1,
    HarnessMessageV1,
    HarnessRequirementPolicyV1,
    HarnessSessionIdentityV1,
    HarnessSessionPageV1,
    HarnessSessionProjectionV1,
    HarnessSessionStateV1,
    HarnessSessionStatusV1,
    HarnessTranscriptMessageV1,
    HarnessTurnOutcomeV1,
    HarnessTurnResultV1,
    RequirementInterpretationOutcomeV1,
    RequirementInterpretationV1,
)
from eval_factory.harness.session_models import (
    HarnessSessionRefV1,
    SessionCheckpointPayloadV1,
    SessionEventV1,
    SessionGatewayEventKindV1,
    SessionGatewayPayloadV1,
    SessionLifecycleEventKindV1,
    SessionLifecyclePayloadV1,
    SessionMessageEventKindV1,
    SessionMessagePayloadV1,
    SessionRequirementEventKindV1,
    SessionRequirementPayloadV1,
    SessionRuntimePayloadV1,
    SessionTeamPayloadV1,
)


class HarnessSessionStoreError(RuntimeError):
    pass


class HarnessSessionNotFoundError(HarnessSessionStoreError):
    pass


class HarnessSessionConflictError(HarnessSessionStoreError):
    pass


class HarnessSessionConcurrencyError(HarnessSessionStoreError):
    pass


class HarnessSessionIntegrityError(HarnessSessionStoreError):
    pass


class HarnessUnknownInvocationOutcomeError(HarnessSessionStoreError):
    pass


@dataclass(frozen=True, slots=True)
class HarnessTurnStart:
    state: HarnessSessionStateV1
    command: HarnessMessageCommandV1
    user_message: HarnessMessageV1
    user_event: SessionEventV1
    replay: HarnessTurnResultV1 | None


@dataclass(frozen=True, slots=True)
class HarnessShellReconcileClaim:
    replay: bool
    effect_ref: ObjectRef | None


class HarnessSessionStore:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._fault = fault_injector or (lambda _: None)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS harness_sessions (
                    session_id TEXT PRIMARY KEY,
                    identity_object_id TEXT NOT NULL UNIQUE,
                    incarnation_id TEXT NOT NULL UNIQUE,
                    composition_object_id TEXT NOT NULL,
                    session_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    last_event_sequence INTEGER NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    identity_json TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS harness_commands (
                    command_object_id TEXT PRIMARY KEY,
                    command_sha256 TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES harness_sessions(session_id),
                    idempotency_key TEXT NOT NULL,
                    user_message_object_id TEXT NOT NULL UNIQUE,
                    user_event_object_id TEXT NOT NULL UNIQUE,
                    command_json TEXT NOT NULL,
                    UNIQUE(session_id, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS harness_messages (
                    message_object_id TEXT PRIMARY KEY,
                    message_sha256 TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES harness_sessions(session_id),
                    event_sequence INTEGER NOT NULL,
                    final INTEGER NOT NULL,
                    message_json TEXT NOT NULL,
                    UNIQUE(session_id, event_sequence)
                );

                CREATE TABLE IF NOT EXISTS harness_events (
                    session_id TEXT NOT NULL REFERENCES harness_sessions(session_id),
                    sequence INTEGER NOT NULL,
                    event_object_id TEXT NOT NULL UNIQUE,
                    event_sha256 TEXT NOT NULL,
                    command_object_id TEXT,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS harness_runtime_events (
                    runtime_event_object_id TEXT PRIMARY KEY,
                    runtime_event_sha256 TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES harness_sessions(session_id),
                    runtime_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    source_sequence INTEGER NOT NULL,
                    event_kind TEXT NOT NULL,
                    session_event_object_id TEXT NOT NULL UNIQUE
                        REFERENCES harness_events(event_object_id),
                    runtime_event_json TEXT NOT NULL,
                    UNIQUE (
                        session_id, runtime_id, run_id, work_id, source_sequence
                    )
                );

                CREATE TABLE IF NOT EXISTS harness_gateway_journals (
                    journal_object_id TEXT PRIMARY KEY,
                    journal_sha256 TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES harness_sessions(session_id),
                    command_object_id TEXT NOT NULL REFERENCES harness_commands(command_object_id),
                    state TEXT NOT NULL,
                    journal_json TEXT NOT NULL,
                    UNIQUE(command_object_id, state)
                );

                CREATE TABLE IF NOT EXISTS harness_gateway_journal_heads (
                    command_object_id TEXT PRIMARY KEY REFERENCES harness_commands(command_object_id),
                    journal_object_id TEXT NOT NULL UNIQUE
                        REFERENCES harness_gateway_journals(journal_object_id)
                );

                CREATE TABLE IF NOT EXISTS harness_interpretations (
                    interpretation_object_id TEXT PRIMARY KEY,
                    interpretation_sha256 TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES harness_sessions(session_id),
                    command_object_id TEXT NOT NULL UNIQUE
                        REFERENCES harness_commands(command_object_id),
                    interpretation_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS harness_requirement_policies (
                    policy_object_id TEXT PRIMARY KEY,
                    policy_sha256 TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES harness_sessions(session_id),
                    interpretation_object_id TEXT NOT NULL UNIQUE
                        REFERENCES harness_interpretations(interpretation_object_id),
                    policy_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS harness_turns (
                    turn_object_id TEXT PRIMARY KEY,
                    turn_sha256 TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES harness_sessions(session_id),
                    command_object_id TEXT NOT NULL UNIQUE
                        REFERENCES harness_commands(command_object_id),
                    turn_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS harness_idempotency (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_type TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    PRIMARY KEY(scope, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS harness_session_projections (
                    session_id TEXT PRIMARY KEY REFERENCES harness_sessions(session_id),
                    source_fingerprint TEXT NOT NULL,
                    projection_json TEXT NOT NULL
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

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
        request_sha256 = _hash_payload(
            {
                "session_id": session_id,
                "incarnation_id": incarnation_id,
                "composition_ref": composition_ref,
                "created_by": created_by,
            }
        )
        scope = "create-harness-session"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="harness-session",
            )
            if replay_id is not None:
                projection = self._load_projection(connection, replay_id)
                connection.rollback()
                return projection
            now = self._clock()
            identity = HarnessSessionIdentityV1.create(
                session_id=session_id,
                incarnation_id=incarnation_id,
                composition_ref=composition_ref,
                created_by=created_by,
                created_at=now,
                audit=audit,
            )
            state = HarnessSessionStateV1.create(
                identity=identity,
                session_version=1,
                status=HarnessSessionStatusV1.ACTIVE,
                last_event_sequence=1,
                current_interpretation_ref=None,
                current_requirement_policy_ref=None,
                updated_at=now,
            )
            event = SessionEventV1.create(
                event_id=f"session-event://{session_id}/1",
                session=state.to_session_ref(),
                sequence=1,
                authority_version=1,
                turn_id=None,
                step_id=None,
                command_ref=None,
                payload=SessionLifecyclePayloadV1(
                    event_kind=SessionLifecycleEventKindV1.SESSION_OPENED,
                ),
                occurred_at=now,
                audit=audit,
            )
            connection.execute(
                """
                INSERT INTO harness_sessions (
                    session_id, identity_object_id, incarnation_id,
                    composition_object_id, session_version, status,
                    last_event_sequence, state_sha256, identity_json, state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    identity.object_id,
                    incarnation_id,
                    composition_ref.object_id,
                    state.session_version,
                    state.status.value,
                    state.last_event_sequence,
                    state.state_sha256,
                    _json(identity),
                    _json(state),
                ),
            )
            self._insert_event(connection, session_id, event, None)
            projection = self._build_projection(connection, session_id)
            self._write_projection(connection, projection)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="harness-session",
                response_id=session_id,
            )
            self._fault("before_create_session_commit")
            connection.commit()
            return projection
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise HarnessSessionConflictError("session authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def close_session(
        self,
        *,
        session_id: str,
        expected_session_version: int,
        principal_ref: ObjectRef,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> HarnessSessionProjectionV1:
        request_sha256 = _hash_payload(
            {
                "session_id": session_id,
                "expected_session_version": expected_session_version,
                "principal_ref": principal_ref,
            }
        )
        scope = f"close-harness-session:{session_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="harness-session",
            )
            if replay_id is not None:
                projection = self._load_projection(connection, replay_id)
                if projection.session.status is not HarnessSessionStatusV1.CLOSED:
                    raise HarnessSessionIntegrityError(
                        "closed session replay is not closed",
                    )
                connection.rollback()
                return projection
            state, identity = self._load_session(connection, session_id)
            if state.status is HarnessSessionStatusV1.CLOSED:
                raise HarnessSessionConflictError("session is already closed")
            if state.session_version != expected_session_version:
                raise HarnessSessionConcurrencyError("session version is stale")
            now = self._clock()
            next_state = HarnessSessionStateV1.create(
                identity=identity,
                session_version=state.session_version + 1,
                status=HarnessSessionStatusV1.CLOSED,
                last_event_sequence=state.last_event_sequence + 1,
                current_interpretation_ref=state.current_interpretation_ref,
                current_requirement_policy_ref=state.current_requirement_policy_ref,
                updated_at=now,
            )
            event = SessionEventV1.create(
                event_id=(
                    f"session-event://{session_id}/"
                    f"{next_state.last_event_sequence}"
                ),
                session=next_state.to_session_ref(),
                sequence=next_state.last_event_sequence,
                authority_version=next_state.session_version,
                turn_id=None,
                step_id=None,
                command_ref=None,
                payload=SessionLifecyclePayloadV1(
                    event_kind=SessionLifecycleEventKindV1.SESSION_CLOSED,
                ),
                occurred_at=now,
                audit=audit,
            )
            self._update_state(connection, next_state)
            self._insert_event(connection, session_id, event, None)
            projection = self._build_projection(connection, session_id)
            self._write_projection(connection, projection)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="harness-session",
                response_id=session_id,
            )
            self._fault("before_close_session_commit")
            connection.commit()
            return projection
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim_shell_reconcile(
        self,
        *,
        session_id: str,
        expected_session_version: int,
        principal_ref: ObjectRef,
        idempotency_key: str,
        effect_ref: ObjectRef | None,
    ) -> HarnessShellReconcileClaim:
        """Persist a replayable reconcile intent before cross-store effects."""
        request_sha256 = _hash_payload(
            {
                "session_id": session_id,
                "expected_session_version": expected_session_version,
                "principal_ref": principal_ref,
            }
        )
        scope = f"agent-shell-reconcile:{session_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-shell-reconcile-intent",
            )
            if replay_id is not None:
                persisted_effect_ref = _shell_reconcile_effect_ref(
                    replay_id,
                )
                connection.rollback()
                return HarnessShellReconcileClaim(
                    replay=True,
                    effect_ref=persisted_effect_ref,
                )
            state, _identity = self._load_session(
                connection,
                session_id,
            )
            if state.session_version != expected_session_version:
                raise HarnessSessionConcurrencyError(
                    "Agent Shell reconcile uses stale session authority",
                )
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-shell-reconcile-intent",
                response_id=_shell_reconcile_effect_id(effect_ref),
            )
            self._fault("before_shell_reconcile_claim_commit")
            connection.commit()
            return HarnessShellReconcileClaim(
                replay=False,
                effect_ref=effect_ref,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def has_message_command(
        self,
        *,
        session_id: str,
        idempotency_key: str,
    ) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._load_session(connection, session_id)
            row = connection.execute(
                """
                SELECT 1
                FROM harness_commands
                WHERE session_id = ? AND idempotency_key = ?
                """,
                (session_id, idempotency_key),
            ).fetchone()
            connection.rollback()
            return row is not None
        finally:
            connection.close()

    def start_turn(
        self,
        *,
        session_id: str,
        expected_session_version: int,
        principal_ref: ObjectRef,
        content: str,
        artifact_envelope_refs: tuple[ObjectRef, ...],
        idempotency_key: str,
        audit: ContractAudit,
    ) -> HarnessTurnStart:
        normalized_content = content.strip()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            state, identity = self._load_session(connection, session_id)
            existing = connection.execute(
                """
                SELECT * FROM harness_commands
                WHERE session_id = ? AND idempotency_key = ?
                """,
                (session_id, idempotency_key),
            ).fetchone()
            command_id = f"session-command://{session_id}/{_suffix(idempotency_key)}"
            command = HarnessMessageCommandV1.create(
                command_id=command_id,
                session_ref=state.session_ref,
                expected_session_version=expected_session_version,
                principal_ref=principal_ref,
                content_sha256=hashlib.sha256(normalized_content.encode()).hexdigest(),
                artifact_envelope_refs=sorted_refs(artifact_envelope_refs),
                idempotency_key=idempotency_key,
                audit=audit,
            )
            if existing is not None:
                stored_command = _parse(
                    HarnessMessageCommandV1,
                    str(existing["command_json"]),
                    "session command",
                )
                if stored_command.to_ref() != command.to_ref():
                    raise HarnessSessionConflictError("session idempotency key already binds another command")
                user_message = self._load_message_by_id(
                    connection,
                    str(existing["user_message_object_id"]),
                )
                user_event = self._load_event_by_id(
                    connection,
                    str(existing["user_event_object_id"]),
                )
                replay = self._load_turn_by_command(
                    connection,
                    command.object_id,
                )
                connection.rollback()
                return HarnessTurnStart(
                    state=state,
                    command=command,
                    user_message=user_message,
                    user_event=user_event,
                    replay=replay,
                )
            if state.status is not HarnessSessionStatusV1.ACTIVE:
                raise HarnessSessionConflictError("session is not active")
            if state.session_version != expected_session_version:
                raise HarnessSessionConcurrencyError("session version is stale")
            now = self._clock()
            next_version = state.session_version + 1
            next_sequence = state.last_event_sequence + 1
            user_message = HarnessMessageV1.create(
                message_id=f"harness-message://{session_id}/{next_sequence}/user",
                session_ref=state.session_ref,
                role=HarnessMessageRoleV1.USER,
                content=normalized_content,
                artifact_envelope_refs=artifact_envelope_refs,
                chunk_index=None,
                final=True,
                created_at=now,
                audit=audit,
            )
            next_state = HarnessSessionStateV1.create(
                identity=identity,
                session_version=next_version,
                status=HarnessSessionStatusV1.ACTIVE,
                last_event_sequence=next_sequence,
                current_interpretation_ref=state.current_interpretation_ref,
                current_requirement_policy_ref=state.current_requirement_policy_ref,
                updated_at=now,
            )
            user_event = SessionEventV1.create(
                event_id=f"session-event://{session_id}/{next_sequence}",
                session=next_state.to_session_ref(),
                sequence=next_sequence,
                authority_version=next_version,
                turn_id=_turn_id(command),
                step_id=None,
                command_ref=command.to_ref(),
                payload=SessionMessagePayloadV1(
                    event_kind=SessionMessageEventKindV1.USER_MESSAGE,
                    message_ref=user_message.to_ref(),
                    model_visible_artifact_refs=user_message.artifact_envelope_refs,
                ),
                occurred_at=now,
                audit=audit,
            )
            connection.execute(
                """
                INSERT INTO harness_commands (
                    command_object_id, command_sha256, session_id,
                    idempotency_key, user_message_object_id,
                    user_event_object_id, command_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command.object_id,
                    command.object_sha256,
                    session_id,
                    command.idempotency_key,
                    user_message.object_id,
                    user_event.object_id,
                    _json(command),
                ),
            )
            self._insert_message(
                connection,
                session_id,
                user_message,
                next_sequence,
            )
            self._insert_event(
                connection,
                session_id,
                user_event,
                command.object_id,
            )
            self._update_state(connection, next_state)
            self._write_projection(
                connection,
                self._build_projection(connection, session_id),
            )
            self._fault("before_turn_prefix_commit")
            connection.commit()
            return HarnessTurnStart(
                state=next_state,
                command=command,
                user_message=user_message,
                user_event=user_event,
                replay=None,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_prepared_journal(
        self,
        *,
        session_id: str,
        command: HarnessMessageCommandV1,
        journal: HarnessGatewayJournalV1,
    ) -> HarnessGatewayJournalV1:
        if journal.state is not GatewayJournalStateV1.PREPARED:
            raise HarnessSessionConflictError("initial Gateway journal must be PREPARED")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            state, identity = self._load_session(connection, session_id)
            existing = self._load_journal_head(connection, command.object_id)
            if existing is not None:
                if existing != journal:
                    raise HarnessSessionConflictError("Gateway journal already has different authority")
                connection.rollback()
                return existing
            self._insert_journal(connection, session_id, command.object_id, journal)
            connection.execute(
                """
                INSERT INTO harness_gateway_journal_heads (
                    command_object_id, journal_object_id
                ) VALUES (?, ?)
                """,
                (command.object_id, journal.object_id),
            )
            next_state = HarnessSessionStateV1.create(
                identity=identity,
                session_version=state.session_version + 1,
                status=state.status,
                last_event_sequence=state.last_event_sequence,
                current_interpretation_ref=state.current_interpretation_ref,
                current_requirement_policy_ref=state.current_requirement_policy_ref,
                updated_at=self._clock(),
            )
            self._update_state(connection, next_state)
            self._write_projection(
                connection,
                self._build_projection(connection, session_id),
            )
            self._fault("before_prepared_journal_commit")
            connection.commit()
            return journal
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def complete_turn(
        self,
        *,
        session_id: str,
        command: HarnessMessageCommandV1,
        prepared_journal: HarnessGatewayJournalV1,
        committed_journal: HarnessGatewayJournalV1,
        assistant_messages: tuple[HarnessMessageV1, ...],
        interpretation: RequirementInterpretationV1,
        requirement_policy: HarnessRequirementPolicyV1 | None,
        outcome: HarnessTurnOutcomeV1,
        audit: ContractAudit,
    ) -> HarnessTurnResultV1:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._load_turn_by_command(connection, command.object_id)
            if replay is not None:
                connection.rollback()
                return replay
            state, identity = self._load_session(connection, session_id)
            head = self._load_journal_head(connection, command.object_id)
            if head != prepared_journal:
                raise HarnessSessionIntegrityError("turn completion does not bind current prepared journal")
            if committed_journal.predecessor_journal_ref != prepared_journal.to_ref():
                raise HarnessSessionConflictError(
                    "Gateway journal successor does not bind prepared authority"
                )
            final_messages = [value for value in assistant_messages if value.final]
            if len(final_messages) != 1:
                raise HarnessSessionConflictError("turn completion requires one final assistant message")
            next_version = state.session_version + 1
            sequence = state.last_event_sequence
            event_values: list[SessionEventV1] = []
            for message in assistant_messages:
                sequence += 1
                event_kind = (
                    SessionMessageEventKindV1.ASSISTANT_MESSAGE
                    if message.final
                    else SessionMessageEventKindV1.ASSISTANT_CHUNK
                )
                event = SessionEventV1.create(
                    event_id=f"session-event://{session_id}/{sequence}",
                    session=HarnessSessionRefV1(
                        session_ref=state.session_ref,
                        incarnation_id=state.incarnation_id,
                        session_version=next_version,
                        composition_ref=state.composition_ref,
                    ),
                    sequence=sequence,
                    authority_version=next_version,
                    turn_id=_turn_id(command),
                    step_id=None,
                    command_ref=command.to_ref(),
                    payload=SessionMessagePayloadV1(
                        event_kind=event_kind,
                        message_ref=message.to_ref(),
                        model_visible_artifact_refs=(message.artifact_envelope_refs),
                    ),
                    occurred_at=message.created_at,
                    audit=audit,
                )
                self._insert_message(connection, session_id, message, sequence)
                self._insert_event(
                    connection,
                    session_id,
                    event,
                    command.object_id,
                )
                event_values.append(event)
            sequence += 1
            if committed_journal.state is GatewayJournalStateV1.UNKNOWN_OUTCOME:
                gateway_kind = SessionGatewayEventKindV1.GATEWAY_UNKNOWN
            elif committed_journal.failure_code is None:
                gateway_kind = SessionGatewayEventKindV1.GATEWAY_COMPLETED
            else:
                gateway_kind = SessionGatewayEventKindV1.GATEWAY_FAILED
            gateway_event = SessionEventV1.create(
                event_id=f"session-event://{session_id}/{sequence}",
                session=HarnessSessionRefV1(
                    session_ref=state.session_ref,
                    incarnation_id=state.incarnation_id,
                    session_version=next_version,
                    composition_ref=state.composition_ref,
                ),
                sequence=sequence,
                authority_version=next_version,
                turn_id=_turn_id(command),
                step_id=None,
                command_ref=command.to_ref(),
                payload=SessionGatewayPayloadV1(
                    event_kind=gateway_kind,
                    journal_ref=committed_journal.to_ref(),
                ),
                occurred_at=self._clock(),
                audit=audit,
            )
            self._insert_event(
                connection,
                session_id,
                gateway_event,
                command.object_id,
            )
            event_values.append(gateway_event)
            sequence += 1
            requirement_kind = {
                HarnessTurnOutcomeV1.READY: (SessionRequirementEventKindV1.REQUIREMENT_READY),
                HarnessTurnOutcomeV1.CLARIFICATION_REQUIRED: (
                    SessionRequirementEventKindV1.CLARIFICATION_REQUIRED
                ),
            }.get(
                outcome,
                SessionRequirementEventKindV1.REQUIREMENT_BLOCKED,
            )
            requirement_event = SessionEventV1.create(
                event_id=f"session-event://{session_id}/{sequence}",
                session=HarnessSessionRefV1(
                    session_ref=state.session_ref,
                    incarnation_id=state.incarnation_id,
                    session_version=next_version,
                    composition_ref=state.composition_ref,
                ),
                sequence=sequence,
                authority_version=next_version,
                turn_id=_turn_id(command),
                step_id=None,
                command_ref=command.to_ref(),
                payload=SessionRequirementPayloadV1(
                    event_kind=requirement_kind,
                    interpretation_ref=interpretation.to_ref(),
                ),
                occurred_at=self._clock(),
                audit=audit,
            )
            self._insert_event(
                connection,
                session_id,
                requirement_event,
                command.object_id,
            )
            event_values.append(requirement_event)
            turn = HarnessTurnResultV1.create(
                turn_id=_turn_id(command),
                session_ref=state.session_ref,
                command_ref=command.to_ref(),
                user_message_ref=interpretation.user_message_ref,
                assistant_message_ref=final_messages[0].to_ref(),
                interpretation_ref=interpretation.to_ref(),
                requirement_policy_ref=(
                    requirement_policy.to_ref() if requirement_policy is not None else None
                ),
                gateway_journal_ref=committed_journal.to_ref(),
                event_refs=sorted_refs(
                    (
                        self._load_user_event_ref(connection, command),
                        *(value.to_ref() for value in event_values),
                    )
                ),
                outcome=outcome,
                reason_codes=interpretation.reason_codes,
                audit=audit,
            )
            self._insert_journal(
                connection,
                session_id,
                command.object_id,
                committed_journal,
            )
            connection.execute(
                """
                UPDATE harness_gateway_journal_heads
                SET journal_object_id = ?
                WHERE command_object_id = ?
                """,
                (committed_journal.object_id, command.object_id),
            )
            connection.execute(
                """
                INSERT INTO harness_interpretations (
                    interpretation_object_id, interpretation_sha256,
                    session_id, command_object_id, interpretation_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    interpretation.object_id,
                    interpretation.object_sha256,
                    session_id,
                    command.object_id,
                    _json(interpretation),
                ),
            )
            if requirement_policy is not None:
                connection.execute(
                    """
                    INSERT INTO harness_requirement_policies (
                        policy_object_id, policy_sha256, session_id,
                        interpretation_object_id, policy_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        requirement_policy.object_id,
                        requirement_policy.object_sha256,
                        session_id,
                        interpretation.object_id,
                        _json(requirement_policy),
                    ),
                )
            connection.execute(
                """
                INSERT INTO harness_turns (
                    turn_object_id, turn_sha256, session_id,
                    command_object_id, turn_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    turn.object_id,
                    turn.object_sha256,
                    session_id,
                    command.object_id,
                    _json(turn),
                ),
            )
            next_state = HarnessSessionStateV1.create(
                identity=identity,
                session_version=next_version,
                status=(
                    HarnessSessionStatusV1.ACTIVE
                    if outcome
                    in {
                        HarnessTurnOutcomeV1.READY,
                        HarnessTurnOutcomeV1.CLARIFICATION_REQUIRED,
                    }
                    else HarnessSessionStatusV1.BLOCKED
                ),
                last_event_sequence=sequence,
                current_interpretation_ref=interpretation.to_ref(),
                current_requirement_policy_ref=(
                    requirement_policy.to_ref()
                    if requirement_policy is not None
                    else state.current_requirement_policy_ref
                ),
                updated_at=self._clock(),
            )
            self._update_state(connection, next_state)
            self._write_projection(
                connection,
                self._build_projection(connection, session_id),
            )
            self._insert_idempotency(
                connection,
                scope=f"session-turn:{session_id}",
                idempotency_key=command.idempotency_key,
                request_sha256=command.object_sha256,
                response_type="harness-turn-result",
                response_id=turn.object_id,
            )
            self._fault("before_turn_completion_commit")
            connection.commit()
            return turn
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_projection(self, session_id: str) -> HarnessSessionProjectionV1:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            stored = self._load_projection(connection, session_id)
            rebuilt = self._build_projection(connection, session_id)
            if stored != rebuilt:
                raise HarnessSessionIntegrityError("session projection differs from immutable authority")
            connection.rollback()
            return stored
        finally:
            connection.close()

    def get_current_requirement(
        self,
        session_id: str,
    ) -> tuple[RequirementInterpretationV1, HarnessRequirementPolicyV1]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            state, _identity = self._load_session(connection, session_id)
            if state.current_interpretation_ref is None or state.current_requirement_policy_ref is None:
                raise HarnessSessionNotFoundError(
                    "session has no current READY requirement authority",
                )
            interpretation_row = connection.execute(
                """
                SELECT * FROM harness_interpretations
                WHERE interpretation_object_id = ?
                """,
                (state.current_interpretation_ref.object_id,),
            ).fetchone()
            policy_row = connection.execute(
                """
                SELECT * FROM harness_requirement_policies
                WHERE policy_object_id = ?
                """,
                (state.current_requirement_policy_ref.object_id,),
            ).fetchone()
            if interpretation_row is None or policy_row is None:
                raise HarnessSessionIntegrityError(
                    "current READY requirement authority is incomplete",
                )
            interpretation = _parse(
                RequirementInterpretationV1,
                str(interpretation_row["interpretation_json"]),
                "requirement interpretation",
            )
            policy = _parse(
                HarnessRequirementPolicyV1,
                str(policy_row["policy_json"]),
                "requirement policy",
            )
            if (
                interpretation.to_ref() != state.current_interpretation_ref
                or policy.to_ref() != state.current_requirement_policy_ref
                or policy.interpretation_ref != interpretation.to_ref()
                or str(interpretation_row["interpretation_sha256"]) != interpretation.object_sha256
                or str(interpretation_row["session_id"]) != session_id
                or str(policy_row["policy_sha256"]) != policy.object_sha256
                or str(policy_row["session_id"]) != session_id
            ):
                raise HarnessSessionIntegrityError(
                    "current READY requirement authority drifted",
                )
            connection.rollback()
            return interpretation, policy
        finally:
            connection.close()

    def get_interpretation(
        self,
        reference: ObjectRef,
    ) -> RequirementInterpretationV1:
        if reference.object_type != "requirement-interpretation":
            raise HarnessSessionIntegrityError(
                "requirement interpretation ref has the wrong type",
            )
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM harness_interpretations
                WHERE interpretation_object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise HarnessSessionNotFoundError(
                    "requirement interpretation was not found",
                )
            interpretation = _parse(
                RequirementInterpretationV1,
                str(row["interpretation_json"]),
                "requirement interpretation",
            )
            if (
                interpretation.to_ref() != reference
                or str(row["interpretation_sha256"]) != interpretation.object_sha256
            ):
                raise HarnessSessionIntegrityError(
                    "requirement interpretation authority drifted",
                )
            return interpretation
        finally:
            connection.close()

    def get_projection_by_ref(
        self,
        reference: ObjectRef,
    ) -> HarnessSessionProjectionV1:
        if reference.object_type != "harness-session":
            raise HarnessSessionIntegrityError(
                "session ref has the wrong object type",
            )
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT session_id
                FROM harness_sessions
                WHERE identity_object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise HarnessSessionNotFoundError(
                    "session authority was not found",
                )
            session_id = str(row["session_id"])
            stored = self._load_projection(connection, session_id)
            rebuilt = self._build_projection(connection, session_id)
            if stored != rebuilt:
                raise HarnessSessionIntegrityError(
                    "session projection differs from immutable authority",
                )
            if stored.session.session_ref != reference:
                raise HarnessSessionIntegrityError(
                    "session authority differs from its reference",
                )
            connection.rollback()
            return stored
        finally:
            connection.close()

    def append_runtime_event(
        self,
        *,
        session_ref: ObjectRef,
        runtime_event: UnifiedRuntimeEventV2,
        audit: ContractAudit,
        idempotency_key: str,
        turn_id: str | None = None,
        step_id: str | None = None,
        command_ref: ObjectRef | None = None,
    ) -> SessionEventV1:
        runtime_ref = _runtime_event_ref(runtime_event)
        request_sha256 = _hash_payload(
            {
                "session_ref": session_ref,
                "runtime_event": runtime_event,
                "turn_id": turn_id,
                "step_id": step_id,
                "command_ref": command_ref,
            }
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            identity_row = connection.execute(
                """
                SELECT session_id FROM harness_sessions
                WHERE identity_object_id = ?
                """,
                (session_ref.object_id,),
            ).fetchone()
            if identity_row is None:
                raise HarnessSessionNotFoundError(
                    "Harness session was not found",
                )
            session_id = str(identity_row["session_id"])
            scope = f"append-runtime-event:{session_id}"
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="session-event",
            )
            if replay_id is not None:
                event = self._load_event_by_id(connection, replay_id)
                connection.rollback()
                return event

            existing = connection.execute(
                """
                SELECT *
                FROM harness_runtime_events
                WHERE runtime_event_object_id = ?
                """,
                (runtime_event.runtime_event_id,),
            ).fetchone()
            if existing is not None:
                stored = _parse_runtime_event(
                    str(existing["runtime_event_json"]),
                )
                if stored != runtime_event:
                    raise HarnessSessionConflictError(
                        "runtime event identity already binds another value",
                    )
                if str(existing["session_id"]) != session_id:
                    raise HarnessSessionConflictError(
                        "runtime event is already projected by another session",
                    )
                if (
                    stored.runtime_event_sha256 != str(existing["runtime_event_sha256"])
                    or stored.runtime_id != str(existing["runtime_id"])
                    or stored.run_id != str(existing["run_id"])
                    or stored.work_id != str(existing["work_id"])
                    or stored.sequence != int(existing["source_sequence"])
                    or stored.kind.value != str(existing["event_kind"])
                ):
                    raise HarnessSessionIntegrityError(
                        "runtime event columns drifted",
                    )
                event = self._load_event_by_id(
                    connection,
                    str(existing["session_event_object_id"]),
                )
                if (
                    not isinstance(event.payload, SessionRuntimePayloadV1)
                    or event.payload.runtime_event_ref != runtime_ref
                    or event.payload.runtime_id != runtime_event.runtime_id
                    or event.payload.event_kind is not runtime_event.kind
                ):
                    raise HarnessSessionIntegrityError(
                        "runtime event projection drifted",
                    )
                self._insert_idempotency(
                    connection,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    response_type="session-event",
                    response_id=event.object_id,
                )
                connection.commit()
                return event

            state, identity = self._load_session(connection, session_id)
            if state.session_ref != session_ref:
                raise HarnessSessionConcurrencyError(
                    "runtime event uses stale Harness session authority",
                )
            prior = connection.execute(
                """
                SELECT source_sequence, event_kind
                FROM harness_runtime_events
                WHERE session_id = ? AND runtime_id = ?
                  AND run_id = ? AND work_id = ?
                ORDER BY source_sequence DESC
                LIMIT 1
                """,
                (
                    session_id,
                    runtime_event.runtime_id,
                    runtime_event.run_id,
                    runtime_event.work_id,
                ),
            ).fetchone()
            expected_source_sequence = 1 if prior is None else int(prior["source_sequence"]) + 1
            if runtime_event.sequence != expected_source_sequence:
                raise HarnessSessionConflictError(
                    "runtime event source sequence is not contiguous",
                )
            if prior is None and runtime_event.kind.value != "RUN_STARTED":
                raise HarnessSessionConflictError(
                    "runtime event stream must start with RUN_STARTED",
                )
            if prior is not None and runtime_event.kind.value == "RUN_STARTED":
                raise HarnessSessionConflictError(
                    "runtime event stream cannot restart",
                )
            if prior is not None and str(prior["event_kind"]) in {
                "RUN_COMPLETED",
                "RUN_FAILED",
            }:
                raise HarnessSessionConflictError(
                    "runtime event cannot follow a terminal event",
                )
            if runtime_event.kind.value in {"RUN_COMPLETED", "RUN_FAILED"}:
                usage_row = connection.execute(
                    """
                    SELECT 1
                    FROM harness_runtime_events
                    WHERE session_id = ? AND runtime_id = ?
                      AND run_id = ? AND work_id = ?
                      AND event_kind = 'USAGE_REPORTED'
                    LIMIT 1
                    """,
                    (
                        session_id,
                        runtime_event.runtime_id,
                        runtime_event.run_id,
                        runtime_event.work_id,
                    ),
                ).fetchone()
                if usage_row is None:
                    raise HarnessSessionConflictError(
                        "runtime terminal event requires prior usage",
                    )

            next_version = state.session_version + 1
            next_sequence = state.last_event_sequence + 1
            next_state = HarnessSessionStateV1.create(
                identity=identity,
                session_version=next_version,
                status=state.status,
                last_event_sequence=next_sequence,
                current_interpretation_ref=state.current_interpretation_ref,
                current_requirement_policy_ref=(state.current_requirement_policy_ref),
                updated_at=self._clock(),
            )
            event = SessionEventV1.create(
                event_id=(f"session-event://{session_id}/runtime/{_suffix(runtime_event.runtime_event_id)}"),
                session=next_state.to_session_ref(),
                sequence=next_sequence,
                authority_version=next_version,
                turn_id=turn_id,
                step_id=step_id,
                command_ref=command_ref,
                payload=SessionRuntimePayloadV1(
                    event_kind=runtime_event.kind,
                    runtime_id=runtime_event.runtime_id,
                    runtime_event_ref=runtime_ref,
                ),
                occurred_at=runtime_event.occurred_at,
                audit=audit,
            )
            self._insert_event(
                connection,
                session_id,
                event,
                command_ref.object_id if command_ref is not None else None,
            )
            connection.execute(
                """
                INSERT INTO harness_runtime_events (
                    runtime_event_object_id, runtime_event_sha256,
                    session_id, runtime_id, run_id, work_id,
                    source_sequence, event_kind, session_event_object_id,
                    runtime_event_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    runtime_event.runtime_event_id,
                    runtime_event.runtime_event_sha256,
                    session_id,
                    runtime_event.runtime_id,
                    runtime_event.run_id,
                    runtime_event.work_id,
                    runtime_event.sequence,
                    runtime_event.kind.value,
                    event.object_id,
                    runtime_event.model_dump_json(),
                ),
            )
            self._update_state(connection, next_state)
            self._write_projection(
                connection,
                self._build_projection(connection, session_id),
            )
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="session-event",
                response_id=event.object_id,
            )
            self._fault("before_runtime_event_commit")
            connection.commit()
            return event
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_runtime_event(
        self,
        reference: ObjectRef,
    ) -> UnifiedRuntimeEventV2:
        if reference.object_type != "runtime-event" or reference.object_version != "v2":
            raise HarnessSessionIntegrityError(
                "runtime event ref has the wrong type or version",
            )
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM harness_runtime_events
                WHERE runtime_event_object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise HarnessSessionNotFoundError(
                    "runtime event was not found",
                )
            value = _parse_runtime_event(
                str(row["runtime_event_json"]),
            )
            if (
                _runtime_event_ref(value) != reference
                or value.runtime_event_sha256 != str(row["runtime_event_sha256"])
                or value.runtime_id != str(row["runtime_id"])
                or value.run_id != str(row["run_id"])
                or value.work_id != str(row["work_id"])
                or value.sequence != int(row["source_sequence"])
                or value.kind.value != str(row["event_kind"])
            ):
                raise HarnessSessionIntegrityError(
                    "runtime event columns drifted",
                )
            return value
        finally:
            connection.close()

    def append_team_event(
        self,
        *,
        session_ref: ObjectRef,
        source_outbox_ref: ObjectRef,
        team_authority_version: int,
        payload: SessionTeamPayloadV1 | SessionCheckpointPayloadV1,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> SessionEventV1:
        request_sha256 = _hash_payload(
            {
                "session_ref": session_ref,
                "source_outbox_ref": source_outbox_ref,
                "team_authority_version": team_authority_version,
                "payload": payload,
            },
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            identity_row = connection.execute(
                """
                SELECT session_id FROM harness_sessions
                WHERE identity_object_id = ?
                """,
                (session_ref.object_id,),
            ).fetchone()
            if identity_row is None:
                raise HarnessSessionNotFoundError("Harness session was not found")
            session_id = str(identity_row["session_id"])
            scope = f"append-team-event:{session_id}"
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="session-event",
            )
            if replay_id is not None:
                row = connection.execute(
                    """
                    SELECT event_json FROM harness_events
                    WHERE event_object_id = ?
                    """,
                    (replay_id,),
                ).fetchone()
                if row is None:
                    raise HarnessSessionIntegrityError(
                        "idempotent Team event is missing",
                    )
                event = _parse(
                    SessionEventV1,
                    str(row["event_json"]),
                    "session Team event",
                )
                connection.rollback()
                return event
            state, identity = self._load_session(connection, session_id)
            if state.session_ref != session_ref:
                raise HarnessSessionConcurrencyError(
                    "Team event uses stale Harness session authority",
                )
            next_version = state.session_version + 1
            next_sequence = state.last_event_sequence + 1
            now = self._clock()
            prior_event_row = connection.execute(
                """
                SELECT event_json FROM harness_events
                WHERE session_id = ? AND sequence = ?
                """,
                (session_id, state.last_event_sequence),
            ).fetchone()
            if prior_event_row is None:
                raise HarnessSessionIntegrityError(
                    "session event head is missing",
                )
            prior_event = _parse(
                SessionEventV1,
                str(prior_event_row["event_json"]),
                "session event",
            )
            next_state = HarnessSessionStateV1.create(
                identity=identity,
                session_version=next_version,
                status=state.status,
                last_event_sequence=next_sequence,
                current_interpretation_ref=state.current_interpretation_ref,
                current_requirement_policy_ref=state.current_requirement_policy_ref,
                updated_at=now,
            )
            event = SessionEventV1.create(
                event_id=(f"session-event://{session_id}/team/{_suffix(source_outbox_ref.object_id)}"),
                session=next_state.to_session_ref(),
                sequence=next_sequence,
                authority_version=max(
                    next_version,
                    team_authority_version,
                    prior_event.authority_version,
                ),
                turn_id=None,
                step_id=None,
                command_ref=None,
                payload=payload,
                occurred_at=now,
                audit=audit,
            )
            self._insert_event(connection, session_id, event, None)
            self._update_state(connection, next_state)
            self._write_projection(
                connection,
                self._build_projection(connection, session_id),
            )
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="session-event",
                response_id=event.object_id,
            )
            self._fault("before_team_event_commit")
            connection.commit()
            return event
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def rebuild_projection(self, session_id: str) -> HarnessSessionProjectionV1:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            projection = self._build_projection(connection, session_id)
            self._write_projection(connection, projection)
            connection.commit()
            return projection
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def list_sessions(
        self,
        *,
        offset: int,
        limit: int,
        include_closed: bool = False,
    ) -> HarnessSessionPageV1:
        if offset < 0 or limit < 1 or limit > 500:
            raise ValueError("session page is outside bounds")
        connection = self._connect()
        try:
            where_clause = "" if include_closed else "WHERE status != ?"
            parameters: tuple[object, ...] = (
                ()
                if include_closed
                else (HarnessSessionStatusV1.CLOSED.value,)
            )
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM harness_sessions {where_clause}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM harness_sessions
                {where_clause}
                ORDER BY session_id
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
            return HarnessSessionPageV1(
                offset=offset,
                limit=limit,
                total=total,
                sessions=tuple(self._parse_state_row(row) for row in rows),
            )
        finally:
            connection.close()

    def list_events(
        self,
        session_id: str,
        *,
        after_sequence: int,
        limit: int,
    ) -> HarnessEventPageV1:
        if after_sequence < 0 or limit < 1 or limit > 500:
            raise ValueError("event page is outside bounds")
        connection = self._connect()
        try:
            state, _ = self._load_session(connection, session_id)
            rows = connection.execute(
                """
                SELECT * FROM harness_events
                WHERE session_id = ? AND sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (session_id, after_sequence, limit),
            ).fetchall()
            events = tuple(_parse(SessionEventV1, str(row["event_json"]), "session event") for row in rows)
            next_sequence = (
                events[-1].sequence + 1
                if events and events[-1].sequence < state.last_event_sequence
                else None
            )
            return HarnessEventPageV1(
                session_ref=state.session_ref,
                after_sequence=after_sequence,
                limit=limit,
                events=events,
                next_sequence=next_sequence,
            )
        finally:
            connection.close()

    def get_message(self, reference: ObjectRef) -> HarnessMessageV1:
        connection = self._connect()
        try:
            value = self._load_message_by_id(connection, reference.object_id)
            if value.to_ref() != reference:
                raise HarnessSessionIntegrityError("message ref drifted")
            return value
        finally:
            connection.close()

    def get_turn_for_command(
        self,
        command_ref: ObjectRef,
    ) -> HarnessTurnResultV1 | None:
        connection = self._connect()
        try:
            return self._load_turn_by_command(connection, command_ref.object_id)
        finally:
            connection.close()

    def get_turn_start(
        self,
        command_ref: ObjectRef,
    ) -> HarnessTurnStart:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM harness_commands
                WHERE command_object_id = ?
                """,
                (command_ref.object_id,),
            ).fetchone()
            if row is None:
                raise HarnessSessionNotFoundError("session command was not found")
            command = _parse(
                HarnessMessageCommandV1,
                str(row["command_json"]),
                "session command",
            )
            if command.to_ref() != command_ref:
                raise HarnessSessionIntegrityError("session command ref drifted")
            state, _ = self._load_session(connection, str(row["session_id"]))
            return HarnessTurnStart(
                state=state,
                command=command,
                user_message=self._load_message_by_id(
                    connection,
                    str(row["user_message_object_id"]),
                ),
                user_event=self._load_event_by_id(
                    connection,
                    str(row["user_event_object_id"]),
                ),
                replay=self._load_turn_by_command(
                    connection,
                    command.object_id,
                ),
            )
        finally:
            connection.close()

    def get_journal_for_command(
        self,
        command_ref: ObjectRef,
    ) -> HarnessGatewayJournalV1 | None:
        connection = self._connect()
        try:
            return self._load_journal_head(connection, command_ref.object_id)
        finally:
            connection.close()

    def get_journal(self, reference: ObjectRef) -> HarnessGatewayJournalV1:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM harness_gateway_journals
                WHERE journal_object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise HarnessSessionIntegrityError("Gateway journal is missing")
            value = _parse(
                HarnessGatewayJournalV1,
                str(row["journal_json"]),
                "Gateway journal",
            )
            if value.to_ref() != reference:
                raise HarnessSessionIntegrityError("Gateway journal ref drifted")
            return value
        finally:
            connection.close()

    def _load_session(
        self,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> tuple[HarnessSessionStateV1, HarnessSessionIdentityV1]:
        row = connection.execute(
            "SELECT * FROM harness_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise HarnessSessionNotFoundError("Harness session was not found")
        state = self._parse_state_row(row)
        identity = _parse(
            HarnessSessionIdentityV1,
            str(row["identity_json"]),
            "session identity",
        )
        if (
            identity.object_id != row["identity_object_id"]
            or identity.session_id != session_id
            or identity.incarnation_id != row["incarnation_id"]
            or identity.composition_ref.object_id != row["composition_object_id"]
        ):
            raise HarnessSessionIntegrityError("session identity columns drifted")
        return state, identity

    def _parse_state_row(self, row: sqlite3.Row) -> HarnessSessionStateV1:
        state = _parse(
            HarnessSessionStateV1,
            str(row["state_json"]),
            "session state",
        )
        if (
            state.session_id != row["session_id"]
            or state.incarnation_id != row["incarnation_id"]
            or state.composition_ref.object_id != row["composition_object_id"]
            or state.session_version != row["session_version"]
            or state.status.value != row["status"]
            or state.last_event_sequence != row["last_event_sequence"]
            or state.state_sha256 != row["state_sha256"]
        ):
            raise HarnessSessionIntegrityError("session state columns drifted")
        return state

    def _update_state(
        self,
        connection: sqlite3.Connection,
        state: HarnessSessionStateV1,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE harness_sessions
            SET session_version = ?, status = ?, last_event_sequence = ?,
                state_sha256 = ?, state_json = ?
            WHERE session_id = ?
            """,
            (
                state.session_version,
                state.status.value,
                state.last_event_sequence,
                state.state_sha256,
                _json(state),
                state.session_id,
            ),
        )
        if cursor.rowcount != 1:
            raise HarnessSessionNotFoundError("Harness session was not found")

    def _insert_message(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        message: HarnessMessageV1,
        event_sequence: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO harness_messages (
                message_object_id, message_sha256, session_id,
                event_sequence, final, message_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message.object_id,
                message.object_sha256,
                session_id,
                event_sequence,
                int(message.final),
                _json(message),
            ),
        )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        event: SessionEventV1,
        command_object_id: str | None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO harness_events (
                session_id, sequence, event_object_id,
                event_sha256, command_object_id, event_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                event.sequence,
                event.object_id,
                event.object_sha256,
                command_object_id,
                _json(event),
            ),
        )

    def _insert_journal(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        command_object_id: str,
        journal: HarnessGatewayJournalV1,
    ) -> None:
        connection.execute(
            """
            INSERT INTO harness_gateway_journals (
                journal_object_id, journal_sha256, session_id,
                command_object_id, state, journal_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                journal.object_id,
                journal.object_sha256,
                session_id,
                command_object_id,
                journal.state.value,
                _json(journal),
            ),
        )

    def _load_journal_head(
        self,
        connection: sqlite3.Connection,
        command_object_id: str,
    ) -> HarnessGatewayJournalV1 | None:
        row = connection.execute(
            """
            SELECT j.*
            FROM harness_gateway_journal_heads h
            JOIN harness_gateway_journals j
              ON j.journal_object_id = h.journal_object_id
            WHERE h.command_object_id = ?
            """,
            (command_object_id,),
        ).fetchone()
        if row is None:
            return None
        value = _parse(
            HarnessGatewayJournalV1,
            str(row["journal_json"]),
            "Gateway journal",
        )
        if (
            value.object_id != row["journal_object_id"]
            or value.object_sha256 != row["journal_sha256"]
            or value.state.value != row["state"]
        ):
            raise HarnessSessionIntegrityError("Gateway journal columns drifted")
        return value

    def _load_message_by_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> HarnessMessageV1:
        row = connection.execute(
            """
            SELECT * FROM harness_messages
            WHERE message_object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise HarnessSessionIntegrityError("session message is missing")
        value = _parse(
            HarnessMessageV1,
            str(row["message_json"]),
            "session message",
        )
        if (
            value.object_id != row["message_object_id"]
            or value.object_sha256 != row["message_sha256"]
            or int(value.final) != row["final"]
        ):
            raise HarnessSessionIntegrityError("session message columns drifted")
        return value

    def _load_event_by_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> SessionEventV1:
        row = connection.execute(
            """
            SELECT * FROM harness_events
            WHERE event_object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise HarnessSessionIntegrityError("session event is missing")
        return _parse(
            SessionEventV1,
            str(row["event_json"]),
            "session event",
        )

    def _load_user_event_ref(
        self,
        connection: sqlite3.Connection,
        command: HarnessMessageCommandV1,
    ) -> ObjectRef:
        row = connection.execute(
            """
            SELECT user_event_object_id FROM harness_commands
            WHERE command_object_id = ?
            """,
            (command.object_id,),
        ).fetchone()
        if row is None:
            raise HarnessSessionIntegrityError("session command is missing")
        return self._load_event_by_id(
            connection,
            str(row["user_event_object_id"]),
        ).to_ref()

    def _load_turn_by_command(
        self,
        connection: sqlite3.Connection,
        command_object_id: str,
    ) -> HarnessTurnResultV1 | None:
        row = connection.execute(
            """
            SELECT * FROM harness_turns
            WHERE command_object_id = ?
            """,
            (command_object_id,),
        ).fetchone()
        if row is None:
            return None
        value = _parse(
            HarnessTurnResultV1,
            str(row["turn_json"]),
            "session turn",
        )
        if value.object_id != row["turn_object_id"] or value.object_sha256 != row["turn_sha256"]:
            raise HarnessSessionIntegrityError("session turn columns drifted")
        return value

    def _build_projection(
        self,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> HarnessSessionProjectionV1:
        state, _ = self._load_session(connection, session_id)
        message_rows = connection.execute(
            """
            SELECT * FROM harness_messages
            WHERE session_id = ? AND final = 1
            ORDER BY event_sequence
            """,
            (session_id,),
        ).fetchall()
        messages = tuple(
            self._load_message_by_id(
                connection,
                str(row["message_object_id"]),
            )
            for row in message_rows
        )
        event_rows = connection.execute(
            """
            SELECT * FROM harness_events
            WHERE session_id = ?
            ORDER BY sequence
            """,
            (session_id,),
        ).fetchall()
        events = tuple(
            _parse(
                SessionEventV1,
                str(row["event_json"]),
                "session event",
            )
            for row in event_rows
        )
        interpretation = None
        pending_questions: tuple[str, ...] = ()
        if state.current_interpretation_ref is not None:
            row = connection.execute(
                """
                SELECT * FROM harness_interpretations
                WHERE interpretation_object_id = ?
                """,
                (state.current_interpretation_ref.object_id,),
            ).fetchone()
            if row is None:
                raise HarnessSessionIntegrityError("current interpretation is missing")
            interpretation = _parse(
                RequirementInterpretationV1,
                str(row["interpretation_json"]),
                "requirement interpretation",
            )
            if interpretation.to_ref() != state.current_interpretation_ref:
                raise HarnessSessionIntegrityError("current interpretation ref drifted")
            if interpretation.outcome is RequirementInterpretationOutcomeV1.CLARIFICATION_REQUIRED:
                pending_questions = interpretation.clarification_questions
        journal_row = connection.execute(
            """
            SELECT j.*
            FROM harness_gateway_journal_heads h
            JOIN harness_gateway_journals j
              ON j.journal_object_id = h.journal_object_id
            JOIN harness_commands c
              ON c.command_object_id = h.command_object_id
            WHERE c.session_id = ?
            ORDER BY c.rowid DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        latest_gateway = None
        journal_ref = None
        if journal_row is not None:
            journal = _parse(
                HarnessGatewayJournalV1,
                str(journal_row["journal_json"]),
                "Gateway journal",
            )
            journal_ref = journal.to_ref()
            latest_gateway = HarnessGatewaySummaryV1(
                journal_ref=journal_ref,
                state=journal.state,
                evidence_class=journal.evidence_class,
                route_ref=journal.route_ref,
                invocation_result_ref=journal.invocation_result_ref,
                gateway_status=journal.gateway_status,
                model_profile_ref=journal.model_profile_ref,
                prompt_template_ref=journal.prompt_template_ref,
                usage=journal.usage,
                failure_code=journal.failure_code,
            )
        fingerprint = _hash_payload(
            {
                "state": state,
                "messages": [message.to_ref() for message in messages],
                "events": [event.to_ref() for event in events],
                "interpretation_ref": state.current_interpretation_ref,
                "policy_ref": state.current_requirement_policy_ref,
                "journal_ref": journal_ref,
            }
        )
        return HarnessSessionProjectionV1(
            session=state,
            transcript=tuple(
                HarnessTranscriptMessageV1(
                    message_ref=message.to_ref(),
                    role=message.role,
                    content=message.content,
                    artifact_envelope_refs=message.artifact_envelope_refs,
                    created_at=message.created_at,
                )
                for message in messages
            ),
            event_refs=tuple(event.to_ref() for event in events),
            current_interpretation_ref=state.current_interpretation_ref,
            current_requirement_policy_ref=state.current_requirement_policy_ref,
            latest_gateway=latest_gateway,
            pending_clarification_questions=pending_questions,
            source_fingerprint=fingerprint,
        )

    def _write_projection(
        self,
        connection: sqlite3.Connection,
        projection: HarnessSessionProjectionV1,
    ) -> None:
        connection.execute(
            """
            INSERT INTO harness_session_projections (
                session_id, source_fingerprint, projection_json
            ) VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                source_fingerprint = excluded.source_fingerprint,
                projection_json = excluded.projection_json
            """,
            (
                projection.session.session_id,
                projection.source_fingerprint,
                _json(projection),
            ),
        )

    def _load_projection(
        self,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> HarnessSessionProjectionV1:
        row = connection.execute(
            """
            SELECT * FROM harness_session_projections
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            exists = connection.execute(
                "SELECT 1 FROM harness_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if exists is None:
                raise HarnessSessionNotFoundError("Harness session was not found")
            raise HarnessSessionIntegrityError("session projection is missing")
        value = _parse(
            HarnessSessionProjectionV1,
            str(row["projection_json"]),
            "session projection",
        )
        if value.source_fingerprint != row["source_fingerprint"]:
            raise HarnessSessionIntegrityError("session projection fingerprint drifted")
        return value

    def _check_idempotency(
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
            SELECT * FROM harness_idempotency
            WHERE scope = ? AND idempotency_key = ?
            """,
            (scope, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if row["request_sha256"] != request_sha256 or row["response_type"] != response_type:
            raise HarnessSessionConflictError("idempotency key already binds another request")
        return str(row["response_id"])

    def _insert_idempotency(
        self,
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
            INSERT INTO harness_idempotency (
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


def _turn_id(command: HarnessMessageCommandV1) -> str:
    return f"harness-turn://{_suffix(command.object_id)}"


def _suffix(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _json(value: ContractModel) -> str:
    return value.canonical_json().decode()


def _parse[ModelT: ContractModel](
    model_type: type[ModelT],
    payload: str,
    label: str,
) -> ModelT:
    try:
        value = model_type.model_validate_json(payload)
    except ValidationError as exc:
        raise HarnessSessionIntegrityError(f"{label} is invalid") from exc
    if value.canonical_json().decode() != payload:
        raise HarnessSessionIntegrityError(f"{label} is not canonical")
    return value


def _shell_reconcile_effect_id(effect_ref: ObjectRef | None) -> str:
    return "NO_GRAPH_BINDING" if effect_ref is None else _json(effect_ref)


def _shell_reconcile_effect_ref(response_id: str) -> ObjectRef | None:
    if response_id == "NO_GRAPH_BINDING":
        return None
    return _parse(
        ObjectRef,
        response_id,
        "Agent Shell reconcile effect ref",
    )


def _hash_payload(value: object) -> str:
    encoded = json.dumps(
        value,
        default=lambda item: (
            item.model_dump(mode="json", exclude_none=False) if isinstance(item, BaseModel) else str(item)
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _runtime_event_ref(value: UnifiedRuntimeEventV2) -> ObjectRef:
    return ObjectRef(
        object_type="runtime-event",
        object_id=value.runtime_event_id,
        object_version="v2",
        object_sha256=value.runtime_event_sha256,
    )


def _parse_runtime_event(payload: str) -> UnifiedRuntimeEventV2:
    try:
        value = UnifiedRuntimeEventV2.model_validate_json(payload)
    except ValidationError as exc:
        raise HarnessSessionIntegrityError(
            "runtime event is invalid",
        ) from exc
    if value.model_dump_json() != payload:
        raise HarnessSessionIntegrityError(
            "runtime event is not canonical",
        )
    return value


__all__ = [
    "HarnessSessionConcurrencyError",
    "HarnessSessionConflictError",
    "HarnessSessionIntegrityError",
    "HarnessSessionNotFoundError",
    "HarnessSessionStore",
    "HarnessSessionStoreError",
    "HarnessTurnStart",
    "HarnessUnknownInvocationOutcomeError",
]
