from __future__ import annotations

import json
import sqlite3

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.artifacts import ArtifactEnvelopeV1, ArtifactHeadV1
from eval_factory.harness.capability import (
    CapabilityCallV1,
    CapabilityInvocationOutcomeV1,
    CapabilityResultV1,
    CapabilitySideEffectV1,
)
from eval_factory.harness.capability_runtime import CapabilityRuntimeInvocation
from eval_factory.harness.contracts import sorted_refs
from eval_factory.harness.interaction import (
    GraphMutationKindV1,
    PermissionOutcomeV1,
)
from eval_factory.team._store_authority import TeamStoreAuthority
from eval_factory.team._store_codec import parse_record, record_json, request_sha256
from eval_factory.team._store_types import (
    TeamBlackboardError,
    TeamConcurrencyError,
    TeamIdempotencyConflictError,
    TeamIntegrityError,
    TeamMailboxError,
    TeamMessagePage,
    TeamSnapshot,
    TeamTaskCompletion,
    TeamTaskWork,
)
from eval_factory.team.models import (
    ArtifactSubscriptionV1,
    TeamConflictStatusV1,
    TeamConflictV1,
    TeamMessageAudienceV1,
    TeamMessageKindV1,
    TeamMessageV1,
    TeamOutboxKindV1,
    TeamTaskEventV1,
    TeamTaskV1,
    TeamV1,
)


class TeamStoreCollaboration(TeamStoreAuthority):
    """Owns P2P mailbox and Artifact Blackboard invariants."""

    def seed_artifact_head(
        self,
        *,
        team_ref: ObjectRef,
        head_id: str,
        envelope: ArtifactEnvelopeV1,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ArtifactHeadV1:
        team_id = self._team_id(team_ref)
        digest = request_sha256(
            {
                "team_ref": team_ref,
                "head_id": head_id,
                "envelope_ref": envelope.to_ref(),
            },
        )
        scope = f"seed-artifact:{team_id}:{head_id}"
        with self._write() as connection:
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "ARTIFACT_HEAD",
                    "artifact_heads",
                    ArtifactHeadV1,
                )
            snapshot = self._snapshot(connection, team_id)
            input_heads = {value for task in snapshot.graph.tasks for value in task.input_artifact_head_ids}
            output_heads = {value for task in snapshot.graph.tasks for value in task.output_artifact_head_ids}
            if (
                snapshot.team.to_ref() != team_ref
                or head_id not in input_heads
                or head_id in output_heads
                or envelope.revision != 1
                or envelope.predecessor_envelope_ref is not None
                or envelope.producer_task_ref is not None
                or self._artifact_head(connection, team_id, head_id) is not None
            ):
                raise TeamBlackboardError(
                    "seed Artifact Head is not an unowned current Team input",
                )
            connection.execute(
                "INSERT INTO envelopes VALUES (?, ?, ?, ?, ?, ?)",
                (
                    envelope.object_id,
                    team_id,
                    "__team-input__",
                    envelope.artifact_id,
                    envelope.revision,
                    record_json(envelope),
                ),
            )
            self._fault("seed_artifact_head.after_envelope")
            sequence = (
                int(
                    connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) FROM artifact_heads WHERE team_id = ?",
                        (team_id,),
                    ).fetchone()[0],
                )
                + 1
            )
            head = ArtifactHeadV1.create(
                head_id=head_id,
                team_ref=team_ref,
                semantic_role=envelope.semantic_role,
                revision=1,
                envelope_ref=envelope.to_ref(),
                predecessor_head_ref=None,
                audit=audit,
            )
            connection.execute(
                "INSERT INTO artifact_heads VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    head.object_id,
                    team_id,
                    head_id,
                    1,
                    sequence,
                    envelope.object_id,
                    record_json(head),
                ),
            )
            connection.execute(
                "INSERT INTO artifact_current VALUES (?, ?, ?)",
                (team_id, head_id, head.object_id),
            )
            self._fault("seed_artifact_head.after_head")
            next_snapshot = self._advance_data_scope(
                connection,
                snapshot,
                audit,
            )
            self._fault("seed_artifact_head.after_authority")
            self._outbox(
                connection,
                next_snapshot,
                TeamOutboxKindV1.ARTIFACT_HEAD_CHANGED,
                head.to_ref(),
            )
            self._fault("seed_artifact_head.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "ARTIFACT_HEAD",
                head.object_id,
            )
            self._fault("seed_artifact_head.after_idempotency")
            return head

    def send_message(
        self,
        message: TeamMessageV1,
        *,
        idempotency_key: str,
    ) -> TeamMessageV1:
        team_id = self._team_id(message.team_ref)
        digest = request_sha256({"message_ref": message.to_ref()})
        scope = f"send-message:{team_id}"
        with self._write() as connection:
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "MESSAGE",
                    "messages",
                    TeamMessageV1,
                )
            snapshot = self._snapshot(connection, team_id)
            if message.team_ref != snapshot.team.to_ref():
                raise TeamConcurrencyError("message uses stale Team authority")
            sender = self._member_ref(snapshot, message.sender_member_ref)
            recipients = tuple(self._member_ref(snapshot, ref) for ref in message.recipient_member_refs)
            if sender.to_ref() in message.recipient_member_refs:
                raise TeamMailboxError("direct message cannot address its sender")
            system_message = (
                message.audience is TeamMessageAudienceV1.SYSTEM
                and message.message_kind is TeamMessageKindV1.SYSTEM
            )
            if sender.is_coordinator and not system_message:
                raise TeamMailboxError(
                    "Coordinator cannot relay ordinary peer messages",
                )
            if (
                message.audience is TeamMessageAudienceV1.SYSTEM
                or message.message_kind is TeamMessageKindV1.SYSTEM
            ) and (not sender.is_coordinator or not system_message):
                raise TeamMailboxError("only Coordinator may send system messages")
            task = None
            task_id = None
            if message.task_ref is not None:
                task = self._task_ref(snapshot, message.task_ref)
                task_id = task.task_id
                participant_ids = {
                    sender.member_id,
                    *(member.member_id for member in recipients),
                }
                if task.assigned_member_id not in participant_ids and not system_message:
                    raise TeamMailboxError(
                        "message task is outside participant scope",
                    )
            if message.reply_to_message_ref is not None:
                reply = self._load(
                    connection,
                    "messages",
                    message.reply_to_message_ref.object_id,
                    TeamMessageV1,
                )
                if reply.to_ref() != message.reply_to_message_ref:
                    raise TeamMailboxError("reply message reference drifted")
                reply_team = connection.execute(
                    "SELECT team_id FROM teams WHERE object_id = ?",
                    (reply.team_ref.object_id,),
                ).fetchone()
                if reply_team is None or reply_team["team_id"] != team_id:
                    raise TeamMailboxError("reply chain crosses Team authority")
                if (
                    reply.audience is TeamMessageAudienceV1.DIRECT
                    and sender.to_ref() != reply.sender_member_ref
                    and sender.to_ref() not in reply.recipient_member_refs
                ):
                    raise TeamMailboxError(
                        "sender is outside the direct reply chain",
                    )
                if message.task_ref != reply.task_ref:
                    raise TeamMailboxError("reply chain changed task scope")
            visible = self._visible_envelopes(
                connection,
                snapshot,
                sender.member_id,
            )
            if not set(message.artifact_envelope_refs).issubset(visible):
                raise TeamMailboxError("message carries unauthorized artifact ref")
            audience_members = (
                recipients if message.audience is TeamMessageAudienceV1.DIRECT else snapshot.roster.members
            )
            if any(
                not set(message.artifact_envelope_refs).issubset(
                    self._visible_envelopes(
                        connection,
                        snapshot,
                        member.member_id,
                    ),
                )
                for member in audience_members
            ):
                raise TeamMailboxError(
                    "message exposes artifact outside recipient visibility",
                )
            sequence = (
                int(
                    connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) FROM messages WHERE team_id = ?",
                        (team_id,),
                    ).fetchone()[0],
                )
                + 1
            )
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message.object_id,
                    team_id,
                    sequence,
                    sender.member_id,
                    task_id,
                    message.message_kind.value,
                    record_json(message),
                ),
            )
            self._fault("send_message.after_message")
            session_refs = (
                sorted_refs(
                    (
                        sender.independent_session_ref,
                        *(member.independent_session_ref for member in recipients),
                    ),
                )
                if message.audience is TeamMessageAudienceV1.DIRECT
                else tuple(member.independent_session_ref for member in snapshot.roster.members)
            )
            self._outbox_for_sessions(
                connection,
                snapshot,
                TeamOutboxKindV1.TEAM_MESSAGE_POSTED,
                message.to_ref(),
                session_refs,
            )
            self._fault("send_message.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "MESSAGE",
                message.object_id,
            )
            self._fault("send_message.after_idempotency")
            return message

    def list_messages(
        self,
        team_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> TeamMessagePage:
        if after_sequence < 0 or not 1 <= limit <= 500:
            raise ValueError("message page is out of bounds")
        with self._read() as connection:
            self._snapshot(connection, team_id)
            rows = connection.execute(
                """
                SELECT sequence, record_json FROM messages
                WHERE team_id = ? AND sequence > ?
                ORDER BY sequence LIMIT ?
                """,
                (team_id, after_sequence, limit + 1),
            ).fetchall()
            values = tuple(
                parse_record(
                    TeamMessageV1,
                    str(row["record_json"]),
                    "Team message",
                )
                for row in rows[:limit]
            )
            return TeamMessagePage(
                values,
                int(rows[limit - 1]["sequence"]) if len(rows) > limit else None,
            )

    def current_artifact_heads(
        self,
        team_id: str,
    ) -> tuple[ArtifactHeadV1, ...]:
        with self._read() as connection:
            self._snapshot(connection, team_id)
            return self._artifact_heads(connection, team_id)

    def current_artifact_envelopes(
        self,
        team_id: str,
    ) -> tuple[ArtifactEnvelopeV1, ...]:
        return tuple(envelope for _, envelope in self.artifact_head_envelopes(team_id))

    def artifact_head_envelopes(
        self,
        team_id: str,
    ) -> tuple[tuple[ArtifactHeadV1, ArtifactEnvelopeV1], ...]:
        with self._read() as connection:
            self._snapshot(connection, team_id)
            return tuple(
                (head, self._head_envelope(connection, head))
                for head in self._artifact_heads(connection, team_id)
            )

    def task_input_envelopes(
        self,
        team_id: str,
        task_id: str,
    ) -> tuple[ArtifactEnvelopeV1, ...]:
        with self._read() as connection:
            snapshot = self._snapshot(connection, team_id)
            task = self._task(snapshot, task_id)
            heads = {head.head_id: head for head in self._artifact_heads(connection, team_id)}
            if not set(task.input_artifact_head_ids).issubset(heads):
                raise TeamBlackboardError("task input Artifact Head is missing")
            return tuple(
                self._head_envelope(connection, heads[head_id]) for head_id in task.input_artifact_head_ids
            )

    def replay_task_completion(
        self,
        *,
        team_id: str,
        task_id: str,
        invocation_key: str,
        request_ref: ObjectRef,
    ) -> TeamTaskCompletion | None:
        with self._read() as connection:
            snapshot = self._snapshot(connection, team_id)
            self._task(snapshot, task_id)
            rows = connection.execute(
                """
                SELECT c.record_json call_json, r.object_id result_id
                FROM capability_calls c
                LEFT JOIN capability_results r ON r.call_id = c.object_id
                WHERE c.team_id = ? AND c.task_id = ?
                """,
                (team_id, task_id),
            ).fetchall()
            for row in rows:
                call = parse_record(
                    CapabilityCallV1,
                    str(row["call_json"]),
                    "Capability call",
                )
                if call.context.idempotency_key != invocation_key:
                    continue
                if call.request_ref != request_ref:
                    raise TeamIdempotencyConflictError(
                        "Team operation key binds another Capability request",
                    )
                if row["result_id"] is None:
                    raise TeamIntegrityError(
                        "Capability completion result is missing",
                    )
                return self._completion(
                    connection,
                    team_id,
                    str(row["result_id"]),
                )
            return None

    def create_subscription(
        self,
        subscription: ArtifactSubscriptionV1,
        *,
        idempotency_key: str,
    ) -> ArtifactSubscriptionV1:
        team_id = self._team_id(subscription.team_ref)
        digest = request_sha256({"subscription_ref": subscription.to_ref()})
        scope = f"create-subscription:{team_id}:{subscription.subscription_id}"
        with self._write() as connection:
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "SUBSCRIPTION",
                    "subscriptions",
                    ArtifactSubscriptionV1,
                )
            snapshot = self._snapshot(connection, team_id)
            self._validate_pair_head(
                connection,
                team_id=team_id,
                history_table="subscriptions",
                current_table="subscription_current",
                key_column="subscription_id",
                version_column="revision",
            )
            if subscription.team_ref != snapshot.team.to_ref():
                raise TeamConcurrencyError("subscription uses stale Team authority")
            self._member_ref(snapshot, subscription.member_ref)
            task_ids = {task.task_id for task in snapshot.graph.tasks}
            head_ids = {
                head_id
                for task in snapshot.graph.tasks
                for head_id in (
                    *task.input_artifact_head_ids,
                    *task.output_artifact_head_ids,
                )
            }
            if (
                subscription.cursor != 0
                or not set(subscription.task_ids).issubset(task_ids)
                or not set(subscription.artifact_head_ids).issubset(head_ids)
            ):
                raise TeamBlackboardError("subscription selector is not current")
            self._insert_subscription(connection, team_id, subscription, 1)
            self._fault("create_subscription.after_record")
            connection.execute(
                "INSERT INTO subscription_current VALUES (?, ?, ?)",
                (team_id, subscription.subscription_id, subscription.object_id),
            )
            self._fault("create_subscription.after_head")
            self._outbox(
                connection,
                snapshot,
                TeamOutboxKindV1.TEAM_CHANGED,
                snapshot.team.to_ref(),
            )
            self._fault("create_subscription.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "SUBSCRIPTION",
                subscription.object_id,
            )
            self._fault("create_subscription.after_idempotency")
            return subscription

    def advance_subscription(
        self,
        *,
        current_ref: ObjectRef,
        cursor: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> ArtifactSubscriptionV1:
        digest = request_sha256(
            {"current_ref": current_ref, "cursor": cursor},
        )
        with self._write() as connection:
            row = connection.execute(
                "SELECT team_id, subscription_id, revision FROM subscriptions WHERE object_id = ?",
                (current_ref.object_id,),
            ).fetchone()
            if row is None:
                raise TeamBlackboardError("subscription was not found")
            team_id, subscription_id = (
                str(row["team_id"]),
                str(row["subscription_id"]),
            )
            scope = f"advance-subscription:{team_id}:{subscription_id}"
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "SUBSCRIPTION",
                    "subscriptions",
                    ArtifactSubscriptionV1,
                )
            current = self._load(
                connection,
                "subscriptions",
                current_ref.object_id,
                ArtifactSubscriptionV1,
            )
            snapshot = self._snapshot(connection, team_id)
            self._validate_pair_head(
                connection,
                team_id=team_id,
                history_table="subscriptions",
                current_table="subscription_current",
                key_column="subscription_id",
                version_column="revision",
            )
            member = self._member_ref(snapshot, current.member_ref)
            head = connection.execute(
                "SELECT object_id FROM subscription_current WHERE team_id = ? AND subscription_id = ?",
                (team_id, subscription_id),
            ).fetchone()
            latest = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM artifact_heads WHERE team_id = ?",
                    (team_id,),
                ).fetchone()[0],
            )
            matching = self._matching_sequences(
                connection,
                team_id,
                current,
                current.cursor,
            )
            task_ids = {task.task_id for task in snapshot.graph.tasks}
            head_ids = {
                head_id
                for task in snapshot.graph.tasks
                for head_id in (
                    *task.input_artifact_head_ids,
                    *task.output_artifact_head_ids,
                )
            }
            if (
                current.to_ref() != current_ref
                or head is None
                or head["object_id"] != current.object_id
                or not set(current.task_ids).issubset(task_ids)
                or not set(current.artifact_head_ids).issubset(head_ids)
                or cursor < current.cursor
                or cursor > latest
                or (matching and cursor > matching[0])
            ):
                raise TeamBlackboardError("subscription cursor is stale or skips evidence")
            successor = ArtifactSubscriptionV1.create(
                subscription_id=current.subscription_id,
                team_ref=snapshot.team.to_ref(),
                member_ref=member.to_ref(),
                task_ids=current.task_ids,
                artifact_head_ids=current.artifact_head_ids,
                semantic_roles=current.semantic_roles,
                cursor=cursor,
                audit=audit,
            )
            self._insert_subscription(
                connection,
                team_id,
                successor,
                int(row["revision"]) + 1,
            )
            self._fault("advance_subscription.after_record")
            connection.execute(
                "UPDATE subscription_current SET object_id = ? WHERE team_id = ? AND subscription_id = ?",
                (successor.object_id, team_id, subscription_id),
            )
            self._fault("advance_subscription.after_head")
            self._outbox(
                connection,
                snapshot,
                TeamOutboxKindV1.TEAM_CHANGED,
                snapshot.team.to_ref(),
            )
            self._fault("advance_subscription.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "SUBSCRIPTION",
                successor.object_id,
            )
            self._fault("advance_subscription.after_idempotency")
            return successor

    def open_conflict(
        self,
        conflict: TeamConflictV1,
        *,
        opened_by_member_ref: ObjectRef,
        idempotency_key: str,
    ) -> TeamConflictV1:
        if conflict.status is not TeamConflictStatusV1.OPEN:
            raise TeamBlackboardError("new conflict must be OPEN")
        return self._write_conflict(
            conflict,
            opened_by_member_ref,
            None,
            idempotency_key,
        )

    def close_conflict(
        self,
        conflict: TeamConflictV1,
        *,
        current_ref: ObjectRef,
        closed_by_member_ref: ObjectRef,
        idempotency_key: str,
    ) -> TeamConflictV1:
        if conflict.status is TeamConflictStatusV1.OPEN:
            raise TeamBlackboardError("conflict successor must close")
        return self._write_conflict(
            conflict,
            closed_by_member_ref,
            current_ref,
            idempotency_key,
        )

    def open_conflicts(self, team_id: str) -> tuple[TeamConflictV1, ...]:
        with self._read() as connection:
            self._snapshot(connection, team_id)
            self._validate_pair_head(
                connection,
                team_id=team_id,
                history_table="conflicts",
                current_table="conflict_current",
                key_column="conflict_id",
                version_column="revision",
            )
            rows = connection.execute(
                """
                SELECT c.record_json FROM conflict_current h
                JOIN conflicts c ON c.object_id = h.object_id
                WHERE h.team_id = ? AND c.status = ? ORDER BY c.conflict_id
                """,
                (team_id, TeamConflictStatusV1.OPEN.value),
            ).fetchall()
            return tuple(
                parse_record(
                    TeamConflictV1,
                    str(row["record_json"]),
                    "Team conflict",
                )
                for row in rows
            )

    def _write_conflict(
        self,
        conflict: TeamConflictV1,
        member_ref: ObjectRef,
        current_ref: ObjectRef | None,
        operation_key: str,
    ) -> TeamConflictV1:
        team_id = self._team_id(conflict.team_ref)
        digest = request_sha256(
            {
                "conflict_ref": conflict.to_ref(),
                "member_ref": member_ref,
                "current_ref": current_ref,
            },
        )
        scope = f"conflict:{team_id}:{conflict.conflict_id}"
        with self._write() as connection:
            replay = self._replay(connection, scope, operation_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "CONFLICT",
                    "conflicts",
                    TeamConflictV1,
                )
            snapshot = self._snapshot(connection, team_id)
            self._validate_pair_head(
                connection,
                team_id=team_id,
                history_table="conflicts",
                current_table="conflict_current",
                key_column="conflict_id",
                version_column="revision",
            )
            member = self._member_ref(snapshot, member_ref)
            if conflict.team_ref != snapshot.team.to_ref():
                raise TeamConcurrencyError("conflict uses stale Team authority")
            self._task_ref(snapshot, conflict.task_ref)
            current_envelope_refs = {
                self._head_envelope(connection, artifact_head).to_ref()
                for artifact_head in self._artifact_heads(connection, team_id)
            }
            if not set(conflict.artifact_envelope_refs).issubset(
                current_envelope_refs,
            ):
                raise TeamBlackboardError(
                    "conflict subjects are not current Artifact Envelopes",
                )
            if (
                conflict.resolution_artifact_ref is not None
                and conflict.resolution_artifact_ref not in current_envelope_refs
            ):
                raise TeamBlackboardError(
                    "conflict resolution is not current typed evidence",
                )
            if (
                conflict.resolved_by_member_ref is not None
                and conflict.resolved_by_member_ref != member.to_ref()
            ):
                raise TeamBlackboardError("conflict resolver is stale")
            head = connection.execute(
                "SELECT object_id FROM conflict_current WHERE team_id = ? AND conflict_id = ?",
                (team_id, conflict.conflict_id),
            ).fetchone()
            if current_ref is None:
                if head is not None:
                    raise TeamBlackboardError("conflict already exists")
                revision = 1
            else:
                if head is None or head["object_id"] != current_ref.object_id:
                    raise TeamConcurrencyError("conflict current head is stale")
                current = self._load(
                    connection,
                    "conflicts",
                    current_ref.object_id,
                    TeamConflictV1,
                )
                if current.to_ref() != current_ref or current.status is not TeamConflictStatusV1.OPEN:
                    raise TeamBlackboardError("only open conflict may close")
                if (
                    conflict.conflict_id != current.conflict_id
                    or conflict.task_ref != current.task_ref
                    or conflict.artifact_envelope_refs != current.artifact_envelope_refs
                ):
                    raise TeamBlackboardError(
                        "conflict successor changed its subject authority",
                    )
                revision = (
                    int(
                        connection.execute(
                            "SELECT revision FROM conflicts WHERE object_id = ?",
                            (current.object_id,),
                        ).fetchone()[0],
                    )
                    + 1
                )
            connection.execute(
                "INSERT INTO conflicts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    conflict.object_id,
                    team_id,
                    conflict.conflict_id,
                    revision,
                    conflict.status.value,
                    record_json(conflict),
                ),
            )
            self._fault("write_conflict.after_record")
            connection.execute(
                """
                INSERT INTO conflict_current VALUES (?, ?, ?)
                ON CONFLICT(team_id, conflict_id) DO UPDATE SET
                  object_id = excluded.object_id
                """,
                (team_id, conflict.conflict_id, conflict.object_id),
            )
            self._fault("write_conflict.after_head")
            self._outbox(
                connection,
                snapshot,
                (
                    TeamOutboxKindV1.CONFLICT_OPENED
                    if current_ref is None
                    else TeamOutboxKindV1.CONFLICT_RESOLVED
                ),
                conflict.to_ref(),
            )
            self._fault("write_conflict.after_outbox")
            self._remember(
                connection,
                scope,
                operation_key,
                digest,
                "CONFLICT",
                conflict.object_id,
            )
            self._fault("write_conflict.after_idempotency")
            return conflict

    # Completion and Blackboard internals called by TeamStoreAuthority.

    def _validate_invocation(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
        task: TeamTaskV1,
        work: TeamTaskWork,
        invocation: CapabilityRuntimeInvocation,
        output_bindings: tuple[tuple[str, ArtifactEnvelopeV1], ...],
    ) -> None:
        assert work.claim is not None
        member = self._member_ref(snapshot, work.claim.member_ref)
        context = invocation.call.context
        action = invocation.permission_action
        decision = invocation.permission_decision
        grant = next(value for value in snapshot.authority.grants if value.member_id == member.member_id)
        if (
            context.team_ref != snapshot.team.to_ref()
            or context.member_ref != member.to_ref()
            or context.task_ref != task.to_ref()
            or context.session_ref != member.independent_session_ref
            or context.authority_ref != snapshot.authority.to_ref()
            or context.principal_ref != grant.principal_ref
            or context.data_purpose != action.data_purpose
            or context.data_classification != action.data_classification
            or invocation.call.capability_definition_ref != task.capability_definition_ref
            or invocation.call.capability_definition_ref != action.capability_definition_ref
            or invocation.call.provider_binding_ref != action.provider_binding_ref
            or invocation.result.call_ref != invocation.call.to_ref()
            or action.authority_ref != snapshot.authority.to_ref()
            or action.permission_policy_ref != snapshot.authority.permission_policy_ref
            or action.member_id != member.member_id
            or action.principal_ref != grant.principal_ref
            or action.task_id != task.task_id
            or decision.authority_ref != snapshot.authority.to_ref()
            or decision.policy_ref != snapshot.authority.permission_policy_ref
            or decision.action_ref != action.to_ref()
        ):
            raise TeamConcurrencyError("Capability invocation authority is stale")
        inputs = self._input_refs(connection, snapshot.team.team_id, task)
        if invocation.call.input_artifact_refs != inputs or action.data_scope_refs != inputs:
            raise TeamConcurrencyError("Capability inputs are stale")
        if (
            action.capability_definition_ref not in grant.capability_definition_refs
            or action.provider_binding_ref not in grant.provider_binding_refs
            or action.task_id not in grant.task_ids
            or not set(action.data_scope_refs).issubset(grant.data_scope_refs)
            or action.data_purpose not in grant.data_purposes
            or action.data_classification not in grant.data_classifications
            or action.side_effect not in grant.allowed_side_effects
        ):
            raise TeamIntegrityError("Capability action bypassed member grant")
        if decision.outcome is PermissionOutcomeV1.ALLOW and (
            action.source_admission
            or action.external_execution
            or action.side_effect
            in {
                CapabilitySideEffectV1.EXTERNAL_EFFECT,
                CapabilitySideEffectV1.IRREVERSIBLE_WRITE,
            }
            or action.graph_mutation is GraphMutationKindV1.AUTHORITY_WIDENING
            or action.destructive
            or action.release
            or (
                action.side_effect is CapabilitySideEffectV1.REVERSIBLE_WRITE
                and not (
                    action.local_execution
                    and action.deterministic
                    and action.graph_mutation
                    in {
                        GraphMutationKindV1.NONE,
                        GraphMutationKindV1.IN_SCOPE,
                    }
                )
            )
            or snapshot.authority.used_model_requests + action.model_requests_delta
            > snapshot.authority.max_model_requests
            or snapshot.authority.used_model_tokens + action.model_tokens_delta
            > snapshot.authority.max_model_tokens
            or snapshot.authority.used_cost_micro_usd + action.cost_micro_usd_delta
            > snapshot.authority.max_cost_micro_usd
        ):
            raise TeamIntegrityError(
                "Capability action bypassed Balanced Autonomy",
            )
        usage = self._task_usage(
            connection,
            snapshot.team.team_id,
            task.task_id,
        )
        deltas = self._usage_deltas(invocation)
        if any(
            used + delta > maximum
            for used, delta, maximum in zip(
                usage,
                deltas,
                (
                    task.max_model_requests,
                    task.max_model_tokens,
                    task.max_cost_micro_usd,
                ),
                strict=True,
            )
        ):
            raise TeamIntegrityError("Capability action exceeded task budget")
        output_refs = sorted_refs(envelope.to_ref() for _, envelope in output_bindings)
        head_ids = tuple(head_id for head_id, _ in output_bindings)
        if invocation.result.outcome is CapabilityInvocationOutcomeV1.SUCCEEDED:
            if (
                member.is_coordinator
                or output_refs != invocation.result.output_artifact_refs
                or output_refs != sorted_refs(artifact.to_ref() for artifact in invocation.output_artifacts)
                or tuple(sorted(head_ids)) != head_ids
                or len(set(head_ids)) != len(head_ids)
                or set(head_ids) != set(task.output_artifact_head_ids)
            ):
                raise TeamBlackboardError("successful output binding is invalid")
        elif output_bindings or invocation.output_artifacts:
            raise TeamBlackboardError("non-success cannot publish Artifact Head")
        if (
            invocation.permission_decision.outcome is not PermissionOutcomeV1.ALLOW
            and invocation.result.outcome is not CapabilityInvocationOutcomeV1.BLOCKED_POLICY
        ):
            raise TeamIntegrityError("Capability result bypassed permission")

    def _publish_head(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
        task: TeamTaskV1,
        head_id: str,
        envelope: ArtifactEnvelopeV1,
        audit: ContractAudit,
    ) -> ArtifactHeadV1:
        if (
            envelope.producer_task_ref != task.to_ref()
            or envelope.producer_capability_ref != task.capability_definition_ref
        ):
            raise TeamBlackboardError("Artifact producer is not task owner")
        current = self._artifact_head(connection, snapshot.team.team_id, head_id)
        predecessor = self._head_envelope(connection, current) if current else None
        revision = current.revision + 1 if current else 1
        if (
            envelope.revision != revision
            or envelope.predecessor_envelope_ref != (predecessor.to_ref() if predecessor else None)
            or (
                predecessor is not None
                and (
                    envelope.artifact_id != predecessor.artifact_id
                    or envelope.semantic_role != predecessor.semantic_role
                )
            )
        ):
            raise TeamBlackboardError("Artifact successor lineage is stale")
        connection.execute(
            "INSERT INTO envelopes VALUES (?, ?, ?, ?, ?, ?)",
            (
                envelope.object_id,
                snapshot.team.team_id,
                task.task_id,
                envelope.artifact_id,
                envelope.revision,
                record_json(envelope),
            ),
        )
        sequence = (
            int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM artifact_heads WHERE team_id = ?",
                    (snapshot.team.team_id,),
                ).fetchone()[0],
            )
            + 1
        )
        head = ArtifactHeadV1.create(
            head_id=head_id,
            team_ref=snapshot.team.to_ref(),
            semantic_role=envelope.semantic_role,
            revision=revision,
            envelope_ref=envelope.to_ref(),
            predecessor_head_ref=current.to_ref() if current else None,
            audit=audit,
        )
        connection.execute(
            "INSERT INTO artifact_heads VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                head.object_id,
                snapshot.team.team_id,
                head_id,
                revision,
                sequence,
                envelope.object_id,
                record_json(head),
            ),
        )
        connection.execute(
            """
            INSERT INTO artifact_current VALUES (?, ?, ?)
            ON CONFLICT(team_id, head_id) DO UPDATE SET
              object_id = excluded.object_id
            """,
            (snapshot.team.team_id, head_id, head.object_id),
        )
        return head

    def _completion(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        result_id: str,
    ) -> TeamTaskCompletion:
        result = self._load(
            connection,
            "capability_results",
            result_id,
            CapabilityResultV1,
        )
        row = connection.execute(
            """
            SELECT event_id, team_object_id, artifact_head_ids_json
            FROM completion_responses
            WHERE result_id = ? AND team_id = ?
            """,
            (result_id, team_id),
        ).fetchone()
        if row is None:
            raise TeamIntegrityError("Capability completion response is missing")
        event = self._load(
            connection,
            "task_events",
            str(row["event_id"]),
            TeamTaskEventV1,
        )
        if event.capability_result_ref != result.to_ref():
            raise TeamIntegrityError("Capability completion event drifted")
        try:
            head_ids = json.loads(str(row["artifact_head_ids_json"]))
        except json.JSONDecodeError as exc:
            raise TeamIntegrityError(
                "Capability completion head inventory is invalid",
            ) from exc
        if (
            not isinstance(head_ids, list)
            or any(not isinstance(value, str) for value in head_ids)
            or len(head_ids) != len(set(head_ids))
        ):
            raise TeamIntegrityError(
                "Capability completion head inventory is invalid",
            )
        heads = tuple(
            self._load(
                connection,
                "artifact_heads",
                object_id,
                ArtifactHeadV1,
            )
            for object_id in head_ids
        )
        if (
            sorted_refs(self._head_envelope(connection, head).to_ref() for head in heads)
            != result.output_artifact_refs
        ):
            raise TeamIntegrityError("Capability completion heads drifted")
        team_row = connection.execute(
            "SELECT record_json FROM teams WHERE object_id = ?",
            (str(row["team_object_id"]),),
        ).fetchone()
        if team_row is None:
            raise TeamIntegrityError("Capability completion Team is missing")
        team = parse_record(TeamV1, str(team_row["record_json"]), "Team")
        return TeamTaskCompletion(
            result,
            event,
            heads,
            self._snapshot_at(connection, team.to_ref()),
        )

    def _artifact_heads(
        self,
        connection: sqlite3.Connection,
        team_id: str,
    ) -> tuple[ArtifactHeadV1, ...]:
        expected = {
            str(row["head_id"]): str(row["object_id"])
            for row in connection.execute(
                """
                SELECT h.head_id, h.object_id
                FROM artifact_heads h
                JOIN (
                  SELECT head_id, MAX(revision) revision
                  FROM artifact_heads WHERE team_id = ?
                  GROUP BY head_id
                ) latest
                  ON latest.head_id = h.head_id
                 AND latest.revision = h.revision
                WHERE h.team_id = ?
                """,
                (team_id, team_id),
            ).fetchall()
        }
        rows = connection.execute(
            """
            SELECT c.head_id, c.object_id, h.revision,
                   h.envelope_id, h.record_json
            FROM artifact_current c
            JOIN artifact_heads h ON h.object_id = c.object_id
            WHERE c.team_id = ? ORDER BY c.head_id
            """,
            (team_id,),
        ).fetchall()
        observed = {str(row["head_id"]): str(row["object_id"]) for row in rows}
        if observed != expected:
            raise TeamIntegrityError("Artifact current heads drifted")
        values = tuple(
            parse_record(
                ArtifactHeadV1,
                str(row["record_json"]),
                "Artifact Head",
            )
            for row in rows
        )
        if any(
            value.head_id != row["head_id"]
            or value.object_id != row["object_id"]
            or value.revision != row["revision"]
            or value.envelope_ref.object_id != row["envelope_id"]
            for value, row in zip(values, rows, strict=True)
        ):
            raise TeamIntegrityError("Artifact current head columns drifted")
        return values

    def _artifact_head(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        head_id: str,
    ) -> ArtifactHeadV1 | None:
        return next(
            (head for head in self._artifact_heads(connection, team_id) if head.head_id == head_id),
            None,
        )

    def _head_envelope(
        self,
        connection: sqlite3.Connection,
        head: ArtifactHeadV1,
    ) -> ArtifactEnvelopeV1:
        return self._envelope(connection, head.envelope_ref)

    def _envelope(
        self,
        connection: sqlite3.Connection,
        reference: ObjectRef,
    ) -> ArtifactEnvelopeV1:
        value = self._load(
            connection,
            "envelopes",
            reference.object_id,
            ArtifactEnvelopeV1,
        )
        if value.to_ref() != reference:
            raise TeamIntegrityError("Artifact Envelope reference drifted")
        return value

    def _input_refs(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        task: TeamTaskV1,
    ) -> tuple[ObjectRef, ...]:
        heads = {head.head_id: head for head in self._artifact_heads(connection, team_id)}
        if not set(task.input_artifact_head_ids).issubset(heads):
            raise TeamBlackboardError("task input Artifact Head is missing")
        return sorted_refs(
            self._head_envelope(connection, heads[head_id]).to_ref()
            for head_id in task.input_artifact_head_ids
        )

    def _visible_envelopes(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
        member_id: str,
    ) -> set[ObjectRef]:
        head_ids = {
            head_id
            for task in snapshot.graph.tasks
            if task.assigned_member_id == member_id
            for head_id in (
                *task.input_artifact_head_ids,
                *task.output_artifact_head_ids,
            )
        }
        return {
            self._head_envelope(connection, head).to_ref()
            for head in self._artifact_heads(connection, snapshot.team.team_id)
            if head.head_id in head_ids
        }

    def _matching_sequences(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        subscription: ArtifactSubscriptionV1,
        cursor: int,
    ) -> tuple[int, ...]:
        rows = connection.execute(
            """
            SELECT h.sequence, h.head_id, h.record_json head_json,
                   e.task_id, e.record_json envelope_json
            FROM artifact_heads h JOIN envelopes e ON e.object_id = h.envelope_id
            WHERE h.team_id = ? AND h.sequence > ? ORDER BY h.sequence
            """,
            (team_id, cursor),
        ).fetchall()
        matches = []
        for row in rows:
            head = parse_record(
                ArtifactHeadV1,
                str(row["head_json"]),
                "Artifact Head",
            )
            envelope = parse_record(
                ArtifactEnvelopeV1,
                str(row["envelope_json"]),
                "Artifact Envelope",
            )
            if (
                str(row["task_id"]) in subscription.task_ids
                or head.head_id in subscription.artifact_head_ids
                or envelope.semantic_role in subscription.semantic_roles
            ):
                matches.append(int(row["sequence"]))
        return tuple(matches)

    @staticmethod
    def _insert_subscription(
        connection: sqlite3.Connection,
        team_id: str,
        value: ArtifactSubscriptionV1,
        revision: int,
    ) -> None:
        connection.execute(
            "INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, ?)",
            (
                value.object_id,
                team_id,
                value.subscription_id,
                revision,
                value.cursor,
                record_json(value),
            ),
        )

    def _outbox_for_sessions(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
        kind: TeamOutboxKindV1,
        record_ref: ObjectRef,
        session_refs: tuple[ObjectRef, ...],
    ) -> None:
        record = self._outbox(connection, snapshot, kind, record_ref)
        if record.intended_member_session_refs == session_refs:
            return
        replacement = type(record).create(
            outbox_key=record.outbox_key,
            team_id=record.team_id,
            sequence=record.sequence,
            event_kind=record.event_kind,
            aggregate_ref=record.aggregate_ref,
            record_ref=record.record_ref,
            intended_member_session_refs=session_refs,
            occurred_at=record.occurred_at,
            audit=record.audit,
        )
        connection.execute(
            "UPDATE outbox SET object_id = ?, record_json = ? WHERE object_id = ?",
            (replacement.object_id, record_json(replacement), record.object_id),
        )


__all__ = ["TeamStoreCollaboration"]
