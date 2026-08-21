from __future__ import annotations

import sqlite3
from typing import Any

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.artifacts import ArtifactEnvelopeV1, ArtifactHeadV1
from eval_factory.harness.contracts import sorted_refs
from eval_factory.harness.interaction import ExecutionAuthorityV1
from eval_factory.team._store_codec import parse_record, record_json, request_sha256
from eval_factory.team._store_collaboration import TeamStoreCollaboration
from eval_factory.team._store_types import (
    TeamConcurrencyError,
    TeamConvergenceSource,
    TeamHeadRebuild,
    TeamIntegrityError,
    TeamProjectionSource,
    TeamSnapshot,
    TeamTaskStateError,
)
from eval_factory.team.models import (
    ArtifactSubscriptionV1,
    MemberContextProjectionV1,
    TeamCheckpointV1,
    TeamConflictStatusV1,
    TeamConflictV1,
    TeamConvergenceDecisionV1,
    TeamLifecycleV1,
    TeamMessageAudienceV1,
    TeamMessageV1,
    TeamOutboxKindV1,
    TeamOutboxRecordV1,
    TeamRosterV1,
    TeamTaskClaimV1,
    TeamTaskEventKindV1,
    TeamTaskEventV1,
    TeamTaskGraphV1,
    TeamTaskLeaseV1,
    TeamTaskStatusV1,
    TeamV1,
    validate_team_task_event_log,
)


class TeamStoreProjection(TeamStoreCollaboration):
    """Owns context caches, checkpoints, outbox delivery, rework, and rebuild."""

    def promote_narrowed_graph(
        self,
        team_id: str,
        *,
        expected_team_ref: ObjectRef,
        expected_graph_ref: ObjectRef,
        expected_authority_ref: ObjectRef,
        successor: TeamSnapshot,
        idempotency_key: str,
    ) -> TeamSnapshot:
        digest = request_sha256(
            {
                "team_id": team_id,
                "team_ref": expected_team_ref,
                "graph_ref": expected_graph_ref,
                "authority_ref": expected_authority_ref,
                "successor_team_ref": successor.team.to_ref(),
                "successor_graph_ref": successor.graph.to_ref(),
                "successor_authority_ref": (successor.authority.to_ref()),
            },
        )
        scope = f"narrow-graph:{team_id}"
        with self._write() as connection:
            replay = self._replay(
                connection,
                scope,
                idempotency_key,
                digest,
            )
            if replay is not None:
                if replay[0] != "TEAM":
                    raise TeamIntegrityError(
                        "graph narrowing replay type drifted",
                    )
                row = connection.execute(
                    "SELECT record_json FROM teams WHERE object_id = ?",
                    (replay[1],),
                ).fetchone()
                if row is None:
                    raise TeamIntegrityError(
                        "graph narrowing response Team is missing",
                    )
                replay_team = parse_record(
                    TeamV1,
                    str(row["record_json"]),
                    "Team",
                )
                return self._snapshot_at(
                    connection,
                    replay_team.to_ref(),
                )
            current = self._snapshot(connection, team_id)
            if (
                current.team.to_ref() != expected_team_ref
                or current.graph.to_ref() != expected_graph_ref
                or current.authority.to_ref() != expected_authority_ref
            ):
                raise TeamConcurrencyError(
                    "graph narrowing Team authority is stale",
                )
            self._validate_graph_narrowing(
                connection,
                current=current,
                successor=successor,
            )
            graph = successor.graph
            authority = successor.authority
            team = successor.team
            connection.execute(
                "INSERT INTO graphs VALUES (?, ?, ?, ?)",
                (
                    graph.object_id,
                    team_id,
                    graph.revision,
                    record_json(graph),
                ),
            )
            connection.execute(
                "INSERT INTO authorities VALUES (?, ?, ?, ?)",
                (
                    authority.object_id,
                    team_id,
                    authority.authority_version,
                    record_json(authority),
                ),
            )
            connection.execute(
                "INSERT INTO teams VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    team.object_id,
                    team_id,
                    team.team_version,
                    current.roster.object_id,
                    graph.object_id,
                    authority.object_id,
                    record_json(team),
                ),
            )
            self._fault("promote_narrowed_graph.after_records")
            connection.execute(
                """
                UPDATE team_heads
                SET team_id_ref = ?, graph_id = ?, authority_id = ?
                WHERE team_id = ?
                """,
                (
                    team.object_id,
                    graph.object_id,
                    authority.object_id,
                    team_id,
                ),
            )
            self._fault("promote_narrowed_graph.after_head")
            self._outbox(
                connection,
                successor,
                TeamOutboxKindV1.TEAM_CHANGED,
                team.to_ref(),
            )
            self._fault("promote_narrowed_graph.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "TEAM",
                team.object_id,
            )
            self._fault(
                "promote_narrowed_graph.after_idempotency",
            )
            return successor

    @staticmethod
    def _validate_graph_narrowing(
        connection: sqlite3.Connection,
        *,
        current: TeamSnapshot,
        successor: TeamSnapshot,
    ) -> None:
        graph = successor.graph
        authority = successor.authority
        team = successor.team
        if (
            successor.roster.to_ref() != current.roster.to_ref()
            or graph.graph_id != current.graph.graph_id
            or graph.team_id != current.graph.team_id
            or graph.revision != current.graph.revision + 1
            or graph.predecessor_graph_ref != current.graph.to_ref()
            or graph.roster_ref != current.roster.to_ref()
            or graph.member_ids != current.graph.member_ids
            or authority.authority_id != current.authority.authority_id
            or authority.authority_version != current.authority.authority_version + 1
            or authority.predecessor_authority_ref != current.authority.to_ref()
            or authority.team_id != current.authority.team_id
            or authority.team_incarnation_id != current.authority.team_incarnation_id
            or authority.roster_ref != current.roster.to_ref()
            or authority.task_graph_ref != graph.to_ref()
            or authority.permission_policy_ref != current.authority.permission_policy_ref
            or authority.max_model_requests != current.authority.max_model_requests
            or authority.max_model_tokens != current.authority.max_model_tokens
            or authority.max_cost_micro_usd != current.authority.max_cost_micro_usd
            or authority.used_model_requests != current.authority.used_model_requests
            or authority.used_model_tokens != current.authority.used_model_tokens
            or authority.used_cost_micro_usd != current.authority.used_cost_micro_usd
            or team.team_id != current.team.team_id
            or team.team_incarnation_id != current.team.team_incarnation_id
            or team.team_version != current.team.team_version + 1
            or team.goal_ref != current.team.goal_ref
            or team.composition_ref != current.team.composition_ref
            or team.roster_ref != current.roster.to_ref()
            or team.task_graph_ref != graph.to_ref()
            or team.authority_ref != authority.to_ref()
            or team.coordinator_member_id != current.team.coordinator_member_id
            or team.lifecycle != current.team.lifecycle
        ):
            raise TeamIntegrityError(
                "graph narrowing changed fixed Team authority",
            )
        current_tasks = {task.task_id: task for task in current.graph.tasks}
        successor_tasks = {task.task_id: task for task in graph.tasks}
        if not successor_tasks or not set(successor_tasks).issubset(current_tasks):
            raise TeamIntegrityError(
                "graph narrowing added or removed every task",
            )
        changed_or_removed = set(current_tasks)
        for task_id, task in successor_tasks.items():
            source = current_tasks[task_id]
            if (
                task.task_id != source.task_id
                or task.task_kind != source.task_kind
                or task.assigned_member_id != source.assigned_member_id
                or task.capability_definition_ref != source.capability_definition_ref
                or not set(task.dependency_task_ids).issubset(
                    source.dependency_task_ids,
                )
                or not set(task.input_artifact_head_ids).issubset(
                    source.input_artifact_head_ids,
                )
                or task.output_artifact_head_ids != source.output_artifact_head_ids
                or task.acceptance_check_refs != source.acceptance_check_refs
                or task.status != source.status
                or task.max_attempts != source.max_attempts
                or task.max_model_requests != source.max_model_requests
                or task.max_model_tokens != source.max_model_tokens
                or task.max_cost_micro_usd != source.max_cost_micro_usd
            ):
                raise TeamIntegrityError(
                    "graph narrowing widened or rewrote a task",
                )
            if task.to_ref() == source.to_ref():
                changed_or_removed.discard(task_id)
        if changed_or_removed:
            placeholders = ",".join("?" for _ in changed_or_removed)
            event = connection.execute(
                f"""
                SELECT task_id FROM task_events
                WHERE team_id = ? AND task_id IN ({placeholders})
                LIMIT 1
                """,
                (current.team.team_id, *sorted(changed_or_removed)),
            ).fetchone()
            if event is not None:
                raise TeamTaskStateError(
                    "graph narrowing cannot rewrite task event history",
                )
            removed_heads = {
                head_id
                for task_id in changed_or_removed
                for head_id in current_tasks[task_id].output_artifact_head_ids
            }
            if removed_heads:
                head_placeholders = ",".join("?" for _ in removed_heads)
                head = connection.execute(
                    f"""
                    SELECT head_id FROM artifact_current
                    WHERE team_id = ?
                      AND head_id IN ({head_placeholders})
                    LIMIT 1
                    """,
                    (
                        current.team.team_id,
                        *sorted(removed_heads),
                    ),
                ).fetchone()
                if head is not None:
                    raise TeamTaskStateError(
                        "graph narrowing cannot remove Artifact authority",
                    )
        current_grants = {grant.member_id: grant for grant in current.authority.grants}
        successor_grants = {grant.member_id: grant for grant in authority.grants}
        if set(successor_grants) != set(current_grants):
            raise TeamIntegrityError(
                "graph narrowing changed grant members",
            )
        assigned = {
            member_id: tuple(
                sorted(task.task_id for task in graph.tasks if task.assigned_member_id == member_id)
            )
            for member_id in successor_grants
        }
        for member_id, source_grant in current_grants.items():
            target = successor_grants[member_id]
            if (
                target.principal_ref != source_grant.principal_ref
                or target.capability_definition_refs != source_grant.capability_definition_refs
                or target.provider_binding_refs != source_grant.provider_binding_refs
                or target.task_ids != assigned[member_id]
                or not set(target.task_ids).issubset(
                    source_grant.task_ids,
                )
                or target.data_scope_refs != source_grant.data_scope_refs
                or target.data_purposes != source_grant.data_purposes
                or target.data_classifications != source_grant.data_classifications
                or target.allowed_side_effects != source_grant.allowed_side_effects
            ):
                raise TeamIntegrityError(
                    "graph narrowing widened or rewrote a grant",
                )

    def projection_source(
        self,
        team_id: str,
        member_id: str,
    ) -> TeamProjectionSource:
        with self._read() as connection:
            return self._projection_source(connection, team_id, member_id)

    def _projection_source(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        member_id: str,
    ) -> TeamProjectionSource:
        snapshot = self._snapshot(connection, team_id)
        member = self._member(snapshot, member_id)
        assigned = {task.task_id for task in snapshot.graph.tasks if task.assigned_member_id == member_id}
        neighborhood = set(assigned)
        for task in snapshot.graph.tasks:
            if task.task_id in assigned:
                neighborhood.update(task.dependency_task_ids)
            if assigned.intersection(task.dependency_task_ids):
                neighborhood.add(task.task_id)
        tasks = tuple(task for task in snapshot.graph.tasks if task.task_id in neighborhood)
        head_ids = {
            head_id
            for task in tasks
            for head_id in (
                *task.input_artifact_head_ids,
                *task.output_artifact_head_ids,
            )
        }
        envelopes = tuple(
            self._head_envelope(connection, head)
            for head in self._artifact_heads(connection, team_id)
            if head.head_id in head_ids
        )
        task_by_ref = {task.to_ref(): task.task_id for task in tasks}
        messages = []
        for row in connection.execute(
            "SELECT record_json FROM messages WHERE team_id = ? ORDER BY sequence",
            (team_id,),
        ).fetchall():
            message = parse_record(
                TeamMessageV1,
                str(row["record_json"]),
                "Team message",
            )
            direct = (
                message.sender_member_ref == member.to_ref()
                or member.to_ref() in message.recipient_member_refs
            )
            broadcast = message.audience is not TeamMessageAudienceV1.DIRECT and (
                message.task_ref is None or task_by_ref.get(message.task_ref) in neighborhood
            )
            if direct or broadcast:
                messages.append(message)
        subscriptions = tuple(
            value for value in self._subscriptions(connection, team_id) if value.member_ref == member.to_ref()
        )
        acceptance_refs = tuple(
            sorted(
                {reference for task in tasks for reference in task.acceptance_check_refs},
                key=lambda value: (
                    value.object_type,
                    value.object_id,
                    value.object_version,
                    value.object_sha256,
                ),
            ),
        )
        fingerprint = request_sha256(
            {
                "team_ref": snapshot.team.to_ref(),
                "member_ref": member.to_ref(),
                "task_refs": tuple(task.to_ref() for task in tasks),
                "artifact_refs": tuple(value.to_ref() for value in envelopes),
                "message_refs": tuple(value.to_ref() for value in messages),
                "subscription_refs": tuple(value.to_ref() for value in subscriptions),
                "acceptance_refs": acceptance_refs,
            },
        )
        return TeamProjectionSource(
            snapshot,
            member,
            tasks,
            envelopes,
            tuple(messages),
            subscriptions,
            acceptance_refs,
            fingerprint,
        )

    @staticmethod
    def _projection_from_source(
        source: TeamProjectionSource,
        *,
        audit: ContractAudit,
    ) -> MemberContextProjectionV1:
        return MemberContextProjectionV1.create(
            projection_id=(
                f"{source.snapshot.team.team_id}.{source.member.member_id}."
                f"{source.snapshot.team.team_version}."
                f"{source.source_fingerprint[:16]}"
            ),
            team_ref=source.snapshot.team.to_ref(),
            member_ref=source.member.to_ref(),
            authority_ref=source.snapshot.authority.to_ref(),
            task_refs=sorted_refs(value.to_ref() for value in source.tasks),
            artifact_envelope_refs=sorted_refs(value.to_ref() for value in source.artifact_envelopes),
            message_refs=sorted_refs(value.to_ref() for value in source.messages),
            subscription_refs=sorted_refs(value.to_ref() for value in source.subscriptions),
            acceptance_check_refs=source.acceptance_check_refs,
            audit=audit,
        )

    def persist_projection(
        self,
        projection: MemberContextProjectionV1,
        *,
        source_fingerprint: str,
    ) -> MemberContextProjectionV1:
        team_id = self._team_id(projection.team_ref)
        with self._write() as connection:
            historical = self._snapshot_at(connection, projection.team_ref)
            member = self._member_ref(historical, projection.member_ref)
            digest = request_sha256(
                {
                    "projection_ref": projection.to_ref(),
                    "source_fingerprint": source_fingerprint,
                },
            )
            scope = f"persist-projection:{team_id}:{member.member_id}"
            replay = self._replay(
                connection,
                scope,
                source_fingerprint,
                digest,
            )
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "PROJECTION",
                    "projections",
                    MemberContextProjectionV1,
                )
            self._validate_pair_head(
                connection,
                team_id=team_id,
                history_table="projections",
                current_table="projection_current",
                key_column="member_id",
                version_column="revision",
            )
            source = self._projection_source(
                connection,
                team_id,
                member.member_id,
            )
            expected = self._projection_from_source(
                source,
                audit=projection.audit,
            )
            if projection != expected or source_fingerprint != source.source_fingerprint:
                raise TeamIntegrityError(
                    "member projection is not the current Store projection",
                )
            revision = (
                int(
                    connection.execute(
                        "SELECT COALESCE(MAX(revision), 0) FROM projections "
                        "WHERE team_id = ? AND member_id = ?",
                        (team_id, member.member_id),
                    ).fetchone()[0],
                )
                + 1
            )
            connection.execute(
                "INSERT INTO projections VALUES (?, ?, ?, ?, ?, ?)",
                (
                    projection.object_id,
                    team_id,
                    member.member_id,
                    revision,
                    source_fingerprint,
                    record_json(projection),
                ),
            )
            self._fault("persist_projection.after_record")
            connection.execute(
                """
                INSERT INTO projection_current VALUES (?, ?, ?)
                ON CONFLICT(team_id, member_id) DO UPDATE SET
                  object_id = excluded.object_id
                """,
                (team_id, member.member_id, projection.object_id),
            )
            self._fault("persist_projection.after_head")
            self._outbox(
                connection,
                source.snapshot,
                TeamOutboxKindV1.TEAM_CHANGED,
                source.snapshot.team.to_ref(),
            )
            self._fault("persist_projection.after_outbox")
            self._remember(
                connection,
                scope,
                source_fingerprint,
                digest,
                "PROJECTION",
                projection.object_id,
            )
            self._fault("persist_projection.after_idempotency")
            return projection

    def stored_projection(
        self,
        team_id: str,
        member_id: str,
    ) -> tuple[MemberContextProjectionV1, str] | None:
        with self._read() as connection:
            self._snapshot(connection, team_id)
            self._validate_pair_head(
                connection,
                team_id=team_id,
                history_table="projections",
                current_table="projection_current",
                key_column="member_id",
                version_column="revision",
            )
            row = connection.execute(
                """
                SELECT p.record_json, p.fingerprint FROM projection_current h
                JOIN projections p ON p.object_id = h.object_id
                WHERE h.team_id = ? AND h.member_id = ?
                """,
                (team_id, member_id),
            ).fetchone()
            if row is None:
                return None
            return (
                parse_record(
                    MemberContextProjectionV1,
                    str(row["record_json"]),
                    "member projection",
                ),
                str(row["fingerprint"]),
            )

    def validate_projection_rebuild(
        self,
        team_id: str,
        member_id: str,
    ) -> MemberContextProjectionV1:
        with self._read() as connection:
            self._snapshot(connection, team_id)
            self._validate_pair_head(
                connection,
                team_id=team_id,
                history_table="projections",
                current_table="projection_current",
                key_column="member_id",
                version_column="revision",
            )
            row = connection.execute(
                """
                SELECT p.record_json, p.fingerprint FROM projection_current h
                JOIN projections p ON p.object_id = h.object_id
                WHERE h.team_id = ? AND h.member_id = ?
                """,
                (team_id, member_id),
            ).fetchone()
            if row is None:
                raise TeamIntegrityError("member context projection is missing")
            stored = parse_record(
                MemberContextProjectionV1,
                str(row["record_json"]),
                "member projection",
            )
            source = self._projection_source(
                connection,
                team_id,
                member_id,
            )
            expected = self._projection_from_source(
                source,
                audit=stored.audit,
            )
            if stored != expected or str(row["fingerprint"]) != source.source_fingerprint:
                raise TeamIntegrityError(
                    "member context projection differs from immutable Team authority",
                )
            return stored

    def create_checkpoint(
        self,
        team_id: str,
        *,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> TeamCheckpointV1:
        digest = request_sha256({"team_id": team_id})
        scope = f"checkpoint:{team_id}"
        with self._write() as connection:
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "CHECKPOINT",
                    "checkpoints",
                    TeamCheckpointV1,
                )
            self._validate_checkpoint_head(connection, team_id)
            snapshot = self._snapshot(connection, team_id)
            heads = self._artifact_heads(connection, team_id)
            subscriptions = self._subscriptions(connection, team_id)
            cursor = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM messages WHERE team_id = ?",
                    (team_id,),
                ).fetchone()[0],
            )
            sequence = (
                int(
                    connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) FROM checkpoints WHERE team_id = ?",
                        (team_id,),
                    ).fetchone()[0],
                )
                + 1
            )
            checkpoint = TeamCheckpointV1.create(
                checkpoint_id=f"{team_id}.checkpoint-{sequence}",
                team_ref=snapshot.team.to_ref(),
                roster_ref=snapshot.roster.to_ref(),
                task_graph_ref=snapshot.graph.to_ref(),
                authority_ref=snapshot.authority.to_ref(),
                artifact_head_refs=sorted_refs(head.to_ref() for head in heads),
                subscription_refs=sorted_refs(value.to_ref() for value in subscriptions),
                member_session_refs=sorted_refs(
                    member.independent_session_ref for member in snapshot.roster.members
                ),
                message_cursor=cursor,
                audit=audit,
            )
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                (
                    checkpoint.object_id,
                    team_id,
                    sequence,
                    snapshot.team.team_version,
                    record_json(checkpoint),
                ),
            )
            self._fault("create_checkpoint.after_record")
            connection.execute(
                """
                INSERT INTO checkpoint_current VALUES (?, ?)
                ON CONFLICT(team_id) DO UPDATE SET object_id = excluded.object_id
                """,
                (team_id, checkpoint.object_id),
            )
            self._fault("create_checkpoint.after_head")
            self._outbox(
                connection,
                snapshot,
                TeamOutboxKindV1.TEAM_CHECKPOINTED,
                checkpoint.to_ref(),
            )
            self._fault("create_checkpoint.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "CHECKPOINT",
                checkpoint.object_id,
            )
            self._fault("create_checkpoint.after_idempotency")
            return checkpoint

    def current_checkpoint(self, team_id: str) -> TeamCheckpointV1 | None:
        with self._read() as connection:
            self._snapshot(connection, team_id)
            self._validate_checkpoint_head(connection, team_id)
            row = connection.execute(
                "SELECT object_id FROM checkpoint_current WHERE team_id = ?",
                (team_id,),
            ).fetchone()
            return (
                self._load(
                    connection,
                    "checkpoints",
                    str(row["object_id"]),
                    TeamCheckpointV1,
                )
                if row
                else None
            )

    def revise_for_rework(
        self,
        team_id: str,
        *,
        expected_team_ref: ObjectRef,
        expected_graph_ref: ObjectRef,
        expected_authority_ref: ObjectRef,
        reset_task_ids: tuple[str, ...],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> TeamSnapshot:
        task_ids = tuple(sorted(set(reset_task_ids)))
        digest = request_sha256(
            {
                "team_id": team_id,
                "team_ref": expected_team_ref,
                "graph_ref": expected_graph_ref,
                "authority_ref": expected_authority_ref,
                "reset_task_ids": task_ids,
            },
        )
        scope = f"rework:{team_id}"
        with self._write() as connection:
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                if replay[0] != "TEAM":
                    raise TeamIntegrityError("rework replay type drifted")
                row = connection.execute(
                    "SELECT record_json FROM teams WHERE object_id = ?",
                    (replay[1],),
                ).fetchone()
                if row is None:
                    raise TeamIntegrityError("rework response Team is missing")
                replay_team = parse_record(
                    TeamV1,
                    str(row["record_json"]),
                    "Team",
                )
                return self._snapshot_at(connection, replay_team.to_ref())
            current = self._snapshot(connection, team_id)
            if (
                current.team.to_ref() != expected_team_ref
                or current.graph.to_ref() != expected_graph_ref
                or current.authority.to_ref() != expected_authority_ref
            ):
                raise TeamConcurrencyError(
                    "rework Team authority is stale",
                )
            known = {task.task_id: task for task in current.graph.tasks}
            work = {value.task.task_id: value for value in self._work(connection, current)}
            if not task_ids or not set(task_ids).issubset(known):
                raise TeamTaskStateError("rework task inventory is invalid")
            if any(
                work[task_id].status
                not in {
                    TeamTaskStatusV1.COMPLETED,
                    TeamTaskStatusV1.FAILED,
                    TeamTaskStatusV1.CANCELLED,
                }
                or work[task_id].attempt >= known[task_id].max_attempts
                for task_id in task_ids
            ):
                raise TeamTaskStateError("rework task is not terminal/retryable")
            graph = type(current.graph).create(
                graph_id=current.graph.graph_id,
                team_id=team_id,
                revision=current.graph.revision + 1,
                predecessor_graph_ref=current.graph.to_ref(),
                roster_ref=current.roster.to_ref(),
                member_ids=current.graph.member_ids,
                tasks=current.graph.tasks,
                audit=audit,
            )
            authority = ExecutionAuthorityV1.create(
                authority_id=current.authority.authority_id,
                authority_version=current.authority.authority_version + 1,
                predecessor_authority_ref=current.authority.to_ref(),
                team_id=team_id,
                team_incarnation_id=current.team.team_incarnation_id,
                roster_ref=current.roster.to_ref(),
                task_graph_ref=graph.to_ref(),
                permission_policy_ref=current.authority.permission_policy_ref,
                grants=current.authority.grants,
                max_model_requests=current.authority.max_model_requests,
                max_model_tokens=current.authority.max_model_tokens,
                max_cost_micro_usd=current.authority.max_cost_micro_usd,
                used_model_requests=current.authority.used_model_requests,
                used_model_tokens=current.authority.used_model_tokens,
                used_cost_micro_usd=current.authority.used_cost_micro_usd,
                audit=audit,
            )
            team = TeamV1.create(
                team_id=team_id,
                team_incarnation_id=current.team.team_incarnation_id,
                team_version=current.team.team_version + 1,
                goal_ref=current.team.goal_ref,
                composition_ref=current.team.composition_ref,
                roster_ref=current.roster.to_ref(),
                task_graph_ref=graph.to_ref(),
                authority_ref=authority.to_ref(),
                coordinator_member_id=current.team.coordinator_member_id,
                lifecycle=TeamLifecycleV1.ACTIVE,
                audit=audit,
            )
            successor = TeamSnapshot(team, current.roster, graph, authority)
            connection.execute(
                "INSERT INTO graphs VALUES (?, ?, ?, ?)",
                (graph.object_id, team_id, graph.revision, record_json(graph)),
            )
            connection.execute(
                "INSERT INTO authorities VALUES (?, ?, ?, ?)",
                (
                    authority.object_id,
                    team_id,
                    authority.authority_version,
                    record_json(authority),
                ),
            )
            connection.execute(
                "INSERT INTO teams VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    team.object_id,
                    team_id,
                    team.team_version,
                    current.roster.object_id,
                    graph.object_id,
                    authority.object_id,
                    record_json(team),
                ),
            )
            self._fault("revise_for_rework.after_records")
            connection.execute(
                "UPDATE team_heads SET team_id_ref = ?, graph_id = ?, authority_id = ? WHERE team_id = ?",
                (team.object_id, graph.object_id, authority.object_id, team_id),
            )
            self._fault("revise_for_rework.after_head")
            for task_id in task_ids:
                task = known[task_id]
                prior = work[task_id]
                status = (
                    TeamTaskStatusV1.READY
                    if all(
                        work[dependency].status is TeamTaskStatusV1.COMPLETED and dependency not in task_ids
                        for dependency in task.dependency_task_ids
                    )
                    else TeamTaskStatusV1.PENDING
                )
                event = self._event(
                    successor,
                    task,
                    prior.event_version + 1,
                    prior.attempt,
                    TeamTaskEventKindV1.RETRY_SCHEDULED,
                    status,
                    audit,
                )
                self._append_task_event(connection, team_id, task, event)
            self._fault("revise_for_rework.after_events")
            self._outbox(
                connection,
                successor,
                TeamOutboxKindV1.TEAM_CHANGED,
                team.to_ref(),
            )
            self._fault("revise_for_rework.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "TEAM",
                team.object_id,
            )
            self._fault("revise_for_rework.after_idempotency")
            return successor

    def record_convergence(
        self,
        decision: TeamConvergenceDecisionV1,
        *,
        idempotency_key: str,
    ) -> TeamConvergenceDecisionV1:
        team_id = self._team_id(decision.successor_team_ref or decision.team_ref)
        digest = request_sha256({"decision_ref": decision.to_ref()})
        scope = f"convergence:{team_id}"
        with self._write() as connection:
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "CONVERGENCE",
                    "convergence",
                    TeamConvergenceDecisionV1,
                )
            snapshot = self._snapshot(connection, team_id)
            if (
                (decision.successor_team_ref or decision.team_ref) != snapshot.team.to_ref()
                or (decision.successor_graph_ref or decision.graph_ref) != snapshot.graph.to_ref()
                or (decision.successor_authority_ref or decision.authority_ref) != snapshot.authority.to_ref()
            ):
                raise TeamIntegrityError("convergence decision is stale")
            checkpoint = self._load(
                connection,
                "checkpoints",
                decision.checkpoint_ref.object_id,
                TeamCheckpointV1,
            )
            if checkpoint.to_ref() != decision.checkpoint_ref:
                raise TeamIntegrityError("convergence checkpoint drifted")
            checkpoint_head = connection.execute(
                "SELECT object_id FROM checkpoint_current WHERE team_id = ?",
                (team_id,),
            ).fetchone()
            if (
                checkpoint_head is None
                or checkpoint_head["object_id"] != checkpoint.object_id
                or checkpoint.team_ref != snapshot.team.to_ref()
                or checkpoint.roster_ref != snapshot.roster.to_ref()
                or checkpoint.task_graph_ref != snapshot.graph.to_ref()
                or checkpoint.authority_ref != snapshot.authority.to_ref()
            ):
                raise TeamIntegrityError(
                    "convergence checkpoint is not current Team authority",
                )
            if decision.successor_team_ref is not None:
                prior_row = connection.execute(
                    """
                    SELECT record_json FROM teams
                    WHERE team_id = ? AND version = ?
                    """,
                    (team_id, snapshot.team.team_version - 1),
                ).fetchone()
                if prior_row is None:
                    raise TeamIntegrityError(
                        "rework predecessor Team is missing",
                    )
                prior_team = parse_record(
                    TeamV1,
                    str(prior_row["record_json"]),
                    "Team",
                )
                if (
                    prior_team.to_ref() != decision.team_ref
                    or snapshot.graph.predecessor_graph_ref != decision.graph_ref
                    or snapshot.authority.predecessor_authority_ref != decision.authority_ref
                ):
                    raise TeamIntegrityError(
                        "rework convergence predecessor is stale",
                    )
            sequence = (
                int(
                    connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) FROM convergence WHERE team_id = ?",
                        (team_id,),
                    ).fetchone()[0],
                )
                + 1
            )
            connection.execute(
                "INSERT INTO convergence VALUES (?, ?, ?, ?, ?)",
                (
                    decision.object_id,
                    team_id,
                    sequence,
                    decision.outcome.value,
                    record_json(decision),
                ),
            )
            self._fault("record_convergence.after_record")
            self._outbox(
                connection,
                snapshot,
                TeamOutboxKindV1.TEAM_CHANGED,
                snapshot.team.to_ref(),
            )
            self._fault("record_convergence.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "CONVERGENCE",
                decision.object_id,
            )
            self._fault("record_convergence.after_idempotency")
            return decision

    def convergence_source(
        self,
        team_id: str,
    ) -> TeamConvergenceSource:
        with self._read() as connection:
            snapshot = self._snapshot(connection, team_id)
            self._validate_pair_head(
                connection,
                team_id=team_id,
                history_table="conflicts",
                current_table="conflict_current",
                key_column="conflict_id",
                version_column="revision",
            )
            messages = tuple(
                parse_record(
                    TeamMessageV1,
                    str(row["record_json"]),
                    "Team message",
                )
                for row in connection.execute(
                    """
                    SELECT record_json FROM messages
                    WHERE team_id = ? ORDER BY sequence
                    """,
                    (team_id,),
                ).fetchall()
            )
            conflicts = tuple(
                parse_record(
                    TeamConflictV1,
                    str(row["record_json"]),
                    "Team conflict",
                )
                for row in connection.execute(
                    """
                    SELECT c.record_json FROM conflict_current h
                    JOIN conflicts c ON c.object_id = h.object_id
                    WHERE h.team_id = ? AND c.status = ?
                    ORDER BY c.conflict_id
                    """,
                    (team_id, TeamConflictStatusV1.OPEN.value),
                ).fetchall()
            )
            return TeamConvergenceSource(
                snapshot=snapshot,
                work=self._work(connection, snapshot),
                messages=messages,
                conflicts=conflicts,
                artifact_heads=self._artifact_heads(connection, team_id),
            )

    @staticmethod
    def _validate_checkpoint_head(
        connection: sqlite3.Connection,
        team_id: str,
    ) -> None:
        expected = connection.execute(
            """
            SELECT object_id FROM checkpoints
            WHERE team_id = ? ORDER BY sequence DESC LIMIT 1
            """,
            (team_id,),
        ).fetchone()
        observed = connection.execute(
            """
            SELECT object_id FROM checkpoint_current
            WHERE team_id = ?
            """,
            (team_id,),
        ).fetchone()
        if (expected is None) != (observed is None) or (
            expected is not None and observed is not None and expected["object_id"] != observed["object_id"]
        ):
            raise TeamIntegrityError(
                "checkpoint current head differs from immutable history",
            )

    def list_outbox(
        self,
        team_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> tuple[TeamOutboxRecordV1, ...]:
        if after_sequence < 0 or not 1 <= limit <= 1_000:
            raise ValueError("outbox page is out of bounds")
        with self._read() as connection:
            self._snapshot(connection, team_id)
            return tuple(
                parse_record(
                    TeamOutboxRecordV1,
                    str(row["record_json"]),
                    "Team outbox",
                )
                for row in connection.execute(
                    "SELECT record_json FROM outbox WHERE team_id = ? "
                    "AND sequence > ? ORDER BY sequence LIMIT ?",
                    (team_id, after_sequence, limit),
                ).fetchall()
            )

    def pending_outbox(
        self,
        team_id: str,
        *,
        limit: int = 500,
    ) -> tuple[tuple[TeamOutboxRecordV1, ObjectRef], ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("outbox page is out of bounds")
        pending: list[tuple[TeamOutboxRecordV1, ObjectRef]] = []
        with self._read() as connection:
            self._snapshot(connection, team_id)
            after_sequence = 0
            while len(pending) < limit:
                rows = connection.execute(
                    """
                    SELECT sequence, record_json FROM outbox
                    WHERE team_id = ? AND sequence > ?
                    ORDER BY sequence LIMIT ?
                    """,
                    (team_id, after_sequence, limit),
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    record = parse_record(
                        TeamOutboxRecordV1,
                        str(row["record_json"]),
                        "Team outbox",
                    )
                    delivered = {
                        str(delivery["session_id"])
                        for delivery in connection.execute(
                            "SELECT session_id FROM outbox_delivery WHERE outbox_id = ?",
                            (record.object_id,),
                        ).fetchall()
                    }
                    pending.extend(
                        (record, session_ref)
                        for session_ref in record.intended_member_session_refs
                        if session_ref.object_id not in delivered
                    )
                    if len(pending) >= limit:
                        break
                after_sequence = int(rows[-1]["sequence"])
                if len(rows) < limit:
                    break
        return tuple(pending[:limit])

    def mark_outbox_delivered(
        self,
        *,
        outbox_ref: ObjectRef,
        session_ref: ObjectRef,
    ) -> None:
        with self._write() as connection:
            record = self._load(
                connection,
                "outbox",
                outbox_ref.object_id,
                TeamOutboxRecordV1,
            )
            if record.to_ref() != outbox_ref or session_ref not in record.intended_member_session_refs:
                raise TeamIntegrityError("outbox delivery authority is stale")
            connection.execute(
                "INSERT OR IGNORE INTO outbox_delivery VALUES (?, ?, ?)",
                (
                    record.object_id,
                    session_ref.object_id,
                    self._clock().isoformat(),
                ),
            )

    def rebuild_current_heads(self, *, repair: bool = False) -> TeamHeadRebuild:
        context = self._write() if repair else self._read()
        with context as connection:
            self._validate_rebuild_sources(connection)
            expected: dict[str, dict[Any, Any]] = {
                "team_heads": self._latest(connection, "teams", "team_id", "version"),
                "task_heads": self._latest_pair(
                    connection,
                    "task_events",
                    "team_id",
                    "task_id",
                    "version",
                ),
                "artifact_current": self._latest_pair(
                    connection,
                    "artifact_heads",
                    "team_id",
                    "head_id",
                    "revision",
                ),
                "subscription_current": self._latest_pair(
                    connection,
                    "subscriptions",
                    "team_id",
                    "subscription_id",
                    "revision",
                ),
                "conflict_current": self._latest_pair(
                    connection,
                    "conflicts",
                    "team_id",
                    "conflict_id",
                    "revision",
                ),
                "projection_current": self._latest_pair(
                    connection,
                    "projections",
                    "team_id",
                    "member_id",
                    "revision",
                ),
                "checkpoint_current": self._latest(
                    connection,
                    "checkpoints",
                    "team_id",
                    "sequence",
                ),
            }
            if repair:
                self._repair_heads(connection, expected)
            else:
                self._compare_heads(connection, expected)
            for team_id in expected["team_heads"]:
                self._snapshot(connection, team_id)
            return TeamHeadRebuild(
                len(expected["team_heads"]),
                len(expected["task_heads"]),
                len(expected["artifact_current"]),
                len(expected["subscription_current"]),
                len(expected["conflict_current"]),
                len(expected["projection_current"]),
                len(expected["checkpoint_current"]),
            )

    def recover_expired_leases(
        self,
        team_id: str,
        *,
        audit: ContractAudit,
    ) -> tuple[TeamTaskEventV1, ...]:
        self.rebuild_current_heads()
        results = []
        for work in self.list_task_work(team_id):
            if (
                work.status is TeamTaskStatusV1.ACTIVE
                and work.lease is not None
                and work.fencing_token is not None
                and work.effective_expires_at is not None
                and self._clock() >= work.effective_expires_at
                and (
                    work.latest_event is None
                    or work.latest_event.event_kind is not TeamTaskEventKindV1.CANCEL_REQUESTED
                )
            ):
                results.append(
                    self.expire_lease(
                        lease_ref=work.lease.to_ref(),
                        fencing_token=work.fencing_token,
                        audit=audit,
                        idempotency_key=(f"recover.{work.lease.object_sha256}.{work.fencing_token}"),
                    ),
                )
        return tuple(results)

    # Projection/rebuild helpers.

    def _validate_rebuild_sources(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        task_ids_by_ref: dict[ObjectRef, str] = {}
        member_ids_by_ref: dict[ObjectRef, str] = {}
        prior_graphs: dict[str, TeamTaskGraphV1] = {}
        for row in connection.execute(
            "SELECT * FROM graphs ORDER BY team_id, revision",
        ).fetchall():
            graph = parse_record(
                TeamTaskGraphV1,
                str(row["record_json"]),
                "Team graph",
            )
            if (
                graph.object_id != row["object_id"]
                or graph.team_id != row["team_id"]
                or graph.revision != row["revision"]
                or graph.revision
                != (prior_graphs[graph.team_id].revision + 1 if graph.team_id in prior_graphs else 1)
                or graph.predecessor_graph_ref
                != (prior_graphs[graph.team_id].to_ref() if graph.team_id in prior_graphs else None)
            ):
                raise TeamIntegrityError("Team graph history drifted")
            prior_graphs[graph.team_id] = graph
            for task in graph.tasks:
                prior = task_ids_by_ref.get(task.to_ref())
                if prior is not None and prior != task.task_id:
                    raise TeamIntegrityError("Team task identity drifted")
                task_ids_by_ref[task.to_ref()] = task.task_id

        prior_rosters: dict[str, TeamRosterV1] = {}
        for row in connection.execute(
            "SELECT * FROM rosters ORDER BY team_id, revision",
        ).fetchall():
            roster = parse_record(
                TeamRosterV1,
                str(row["record_json"]),
                "Team roster",
            )
            prior_roster = prior_rosters.get(roster.team_id)
            if (
                roster.object_id != row["object_id"]
                or roster.team_id != row["team_id"]
                or roster.revision != row["revision"]
                or roster.revision != (prior_roster.revision + 1 if prior_roster else 1)
                or roster.predecessor_roster_ref != (prior_roster.to_ref() if prior_roster else None)
            ):
                raise TeamIntegrityError("Team roster history drifted")
            prior_rosters[roster.team_id] = roster
            for member in roster.members:
                prior_member_id = member_ids_by_ref.get(member.to_ref())
                if prior_member_id is not None and prior_member_id != member.member_id:
                    raise TeamIntegrityError("Team member identity drifted")
                member_ids_by_ref[member.to_ref()] = member.member_id

        prior_authorities: dict[str, ExecutionAuthorityV1] = {}
        for row in connection.execute(
            "SELECT * FROM authorities ORDER BY team_id, version",
        ).fetchall():
            authority = parse_record(
                ExecutionAuthorityV1,
                str(row["record_json"]),
                "execution authority",
            )
            prior_authority = prior_authorities.get(authority.team_id)
            if (
                authority.object_id != row["object_id"]
                or authority.team_id != row["team_id"]
                or authority.authority_version != row["version"]
                or authority.authority_version
                != (prior_authority.authority_version + 1 if prior_authority else 1)
                or authority.predecessor_authority_ref
                != (prior_authority.to_ref() if prior_authority else None)
            ):
                raise TeamIntegrityError(
                    "execution authority history drifted",
                )
            prior_authorities[authority.team_id] = authority

        expected_team_versions: dict[str, int] = {}
        for row in connection.execute(
            "SELECT * FROM teams ORDER BY team_id, version",
        ).fetchall():
            team = parse_record(TeamV1, str(row["record_json"]), "Team")
            expected_version = expected_team_versions.get(team.team_id, 1)
            if (
                team.object_id != row["object_id"]
                or team.team_id != row["team_id"]
                or team.team_version != row["version"]
                or team.team_version != expected_version
            ):
                raise TeamIntegrityError("Team history drifted")
            expected_team_versions[team.team_id] = expected_version + 1
            self._snapshot_at(connection, team.to_ref())

        claim_attempts: dict[ObjectRef, int] = {}
        claim_scopes: dict[ObjectRef, tuple[str, str, int]] = {}
        for row in connection.execute(
            "SELECT * FROM task_claims ORDER BY team_id, task_id, attempt",
        ).fetchall():
            claim = parse_record(
                TeamTaskClaimV1,
                str(row["record_json"]),
                "task claim",
            )
            if (
                claim.object_id != row["object_id"]
                or self._team_id_for_ref(connection, claim.team_ref) != row["team_id"]
                or task_ids_by_ref.get(claim.task_ref) != row["task_id"]
                or member_ids_by_ref.get(claim.member_ref) is None
            ):
                raise TeamIntegrityError("task claim history drifted")
            claim_attempts[claim.to_ref()] = int(row["attempt"])
            claim_scopes[claim.to_ref()] = (
                str(row["team_id"]),
                str(row["task_id"]),
                int(row["attempt"]),
            )

        lease_attempts: dict[ObjectRef, int] = {}
        for row in connection.execute(
            "SELECT * FROM task_leases ORDER BY team_id, task_id, attempt",
        ).fetchall():
            lease = parse_record(
                TeamTaskLeaseV1,
                str(row["record_json"]),
                "task lease",
            )
            if (
                lease.object_id != row["object_id"]
                or lease.fencing_token != row["fence"]
                or claim_attempts.get(lease.claim_ref) != row["attempt"]
                or claim_scopes.get(lease.claim_ref)
                != (
                    str(row["team_id"]),
                    str(row["task_id"]),
                    int(row["attempt"]),
                )
            ):
                raise TeamIntegrityError("task lease history drifted")
            lease_attempts[lease.to_ref()] = int(row["attempt"])

        event_logs: dict[tuple[str, str], list[TeamTaskEventV1]] = {}
        for row in connection.execute(
            "SELECT * FROM task_events ORDER BY team_id, task_id, version",
        ).fetchall():
            event = parse_record(
                TeamTaskEventV1,
                str(row["record_json"]),
                "task event",
            )
            if (
                event.object_id != row["object_id"]
                or event.task_event_version != row["version"]
                or event.graph_revision != row["graph_revision"]
                or event.event_kind.value != row["kind"]
                or task_ids_by_ref.get(event.task_ref) != row["task_id"]
                or self._team_id_for_ref(connection, event.team_ref) != row["team_id"]
                or (event.claim_ref is not None and claim_attempts.get(event.claim_ref) != event.attempt)
                or (event.lease_ref is not None and lease_attempts.get(event.lease_ref) != event.attempt)
            ):
                raise TeamIntegrityError("task event history drifted")
            key = (str(row["team_id"]), str(row["task_id"]))
            event_logs.setdefault(key, []).append(event)
        for events in event_logs.values():
            try:
                validate_team_task_event_log(tuple(events))
            except ValueError as exc:
                raise TeamIntegrityError(
                    "task event history is invalid",
                ) from exc

        prior_heads: dict[tuple[str, str], ArtifactHeadV1] = {}
        prior_envelopes: dict[tuple[str, str], ArtifactEnvelopeV1] = {}
        next_sequence: dict[str, int] = {}
        for row in connection.execute(
            """
            SELECT h.*, e.record_json envelope_json
            FROM artifact_heads h
            JOIN envelopes e ON e.object_id = h.envelope_id
            ORDER BY h.team_id, h.sequence
            """,
        ).fetchall():
            team_id = str(row["team_id"])
            head_id = str(row["head_id"])
            head = parse_record(
                ArtifactHeadV1,
                str(row["record_json"]),
                "Artifact Head",
            )
            envelope = parse_record(
                ArtifactEnvelopeV1,
                str(row["envelope_json"]),
                "Artifact Envelope",
            )
            expected_sequence = next_sequence.get(team_id, 1)
            prior_head = prior_heads.get((team_id, head_id))
            prior_envelope = prior_envelopes.get((team_id, head_id))
            if (
                head.object_id != row["object_id"]
                or head.head_id != head_id
                or head.revision != row["revision"]
                or int(row["sequence"]) != expected_sequence
                or head.envelope_ref != envelope.to_ref()
                or envelope.object_id != row["envelope_id"]
                or self._team_id_for_ref(connection, head.team_ref) != team_id
                or head.revision != (prior_head.revision + 1 if prior_head else 1)
                or head.predecessor_head_ref != (prior_head.to_ref() if prior_head else None)
                or envelope.revision != head.revision
                or envelope.predecessor_envelope_ref != (prior_envelope.to_ref() if prior_envelope else None)
                or (
                    prior_envelope is not None
                    and (
                        envelope.artifact_id != prior_envelope.artifact_id
                        or envelope.semantic_role != prior_envelope.semantic_role
                    )
                )
            ):
                raise TeamIntegrityError("Artifact Head history drifted")
            next_sequence[team_id] = expected_sequence + 1
            prior_heads[(team_id, head_id)] = head
            prior_envelopes[(team_id, head_id)] = envelope

        self._validate_revision_rows(
            connection,
            table="subscriptions",
            key_column="subscription_id",
            model=ArtifactSubscriptionV1,
        )
        self._validate_revision_rows(
            connection,
            table="conflicts",
            key_column="conflict_id",
            model=TeamConflictV1,
        )
        self._validate_revision_rows(
            connection,
            table="projections",
            key_column="member_id",
            model=MemberContextProjectionV1,
        )
        for row in connection.execute(
            """
            SELECT u.*, r.team_id result_team_id, r.task_id result_task_id
            FROM task_usage_events u
            JOIN capability_results r ON r.object_id = u.result_id
            """,
        ).fetchall():
            if (
                row["team_id"] != row["result_team_id"]
                or row["task_id"] != row["result_task_id"]
                or int(row["model_requests"]) < 0
                or int(row["model_tokens"]) < 0
                or int(row["cost_micro_usd"]) < 0
            ):
                raise TeamIntegrityError("task usage history drifted")
        for row in connection.execute(
            "SELECT result_id, team_id FROM completion_responses",
        ).fetchall():
            self._completion(
                connection,
                str(row["team_id"]),
                str(row["result_id"]),
            )
        for table in ("messages", "checkpoints", "convergence", "outbox"):
            self._validate_sequence_rows(connection, table)

    def _team_id_for_ref(
        self,
        connection: sqlite3.Connection,
        reference: ObjectRef,
    ) -> str:
        snapshot = self._snapshot_at(connection, reference)
        return snapshot.team.team_id

    @staticmethod
    def _validate_revision_rows(
        connection: sqlite3.Connection,
        *,
        table: str,
        key_column: str,
        model: type[Any],
    ) -> None:
        expected: dict[tuple[str, str], int] = {}
        for row in connection.execute(
            f"SELECT * FROM {table} ORDER BY team_id, {key_column}, revision",
        ).fetchall():
            value = parse_record(model, str(row["record_json"]), table)
            key = (str(row["team_id"]), str(row[key_column]))
            revision = expected.get(key, 1)
            if value.object_id != row["object_id"] or int(row["revision"]) != revision:
                raise TeamIntegrityError(f"{table} history drifted")
            expected[key] = revision + 1

    @staticmethod
    def _validate_sequence_rows(
        connection: sqlite3.Connection,
        table: str,
    ) -> None:
        expected: dict[str, int] = {}
        for row in connection.execute(
            f"SELECT team_id, sequence FROM {table} ORDER BY team_id, sequence",
        ).fetchall():
            team_id = str(row["team_id"])
            sequence = expected.get(team_id, 1)
            if int(row["sequence"]) != sequence:
                raise TeamIntegrityError(f"{table} sequence drifted")
            expected[team_id] = sequence + 1

    def _subscriptions(
        self,
        connection: sqlite3.Connection,
        team_id: str,
    ) -> tuple[ArtifactSubscriptionV1, ...]:
        self._validate_pair_head(
            connection,
            team_id=team_id,
            history_table="subscriptions",
            current_table="subscription_current",
            key_column="subscription_id",
            version_column="revision",
        )
        return tuple(
            parse_record(
                ArtifactSubscriptionV1,
                str(row["record_json"]),
                "artifact subscription",
            )
            for row in connection.execute(
                """
                SELECT s.record_json FROM subscription_current h
                JOIN subscriptions s ON s.object_id = h.object_id
                WHERE h.team_id = ? ORDER BY h.subscription_id
                """,
                (team_id,),
            ).fetchall()
        )

    @staticmethod
    def _latest(
        connection: sqlite3.Connection,
        table: str,
        key: str,
        version: str,
    ) -> dict[str, str]:
        return {
            str(row[key]): str(row["object_id"])
            for row in connection.execute(
                f"""
                SELECT s.* FROM {table} s JOIN (
                  SELECT {key}, MAX({version}) version FROM {table} GROUP BY {key}
                ) x ON x.{key} = s.{key} AND x.version = s.{version}
                """
            ).fetchall()
        }

    @staticmethod
    def _latest_pair(
        connection: sqlite3.Connection,
        table: str,
        first: str,
        second: str,
        version: str,
    ) -> dict[tuple[str, str], str]:
        return {
            (str(row[first]), str(row[second])): str(row["object_id"])
            for row in connection.execute(
                f"""
                SELECT s.* FROM {table} s JOIN (
                  SELECT {first}, {second}, MAX({version}) version
                  FROM {table} GROUP BY {first}, {second}
                ) x ON x.{first} = s.{first} AND x.{second} = s.{second}
                   AND x.version = s.{version}
                """
            ).fetchall()
        }

    def _compare_heads(
        self,
        connection: sqlite3.Connection,
        expected: dict[str, dict[Any, Any]],
    ) -> None:
        observed = {
            "team_heads": {
                str(row["team_id"]): str(row["team_id_ref"])
                for row in connection.execute("SELECT * FROM team_heads")
            },
            "task_heads": {
                (str(row["team_id"]), str(row["task_id"])): str(row["event_id"])
                for row in connection.execute("SELECT * FROM task_heads")
            },
            **{
                table: {
                    (str(row["team_id"]), str(row[key])): str(row["object_id"])
                    for row in connection.execute(f"SELECT * FROM {table}")
                }
                for table, key in (
                    ("artifact_current", "head_id"),
                    ("subscription_current", "subscription_id"),
                    ("conflict_current", "conflict_id"),
                    ("projection_current", "member_id"),
                )
            },
            "checkpoint_current": {
                str(row["team_id"]): str(row["object_id"])
                for row in connection.execute("SELECT * FROM checkpoint_current")
            },
        }
        if observed != expected:
            raise TeamIntegrityError("Team current heads differ from history")

    def _repair_heads(
        self,
        connection: sqlite3.Connection,
        expected: dict[str, dict[Any, Any]],
    ) -> None:
        for table in expected:
            connection.execute(f"DELETE FROM {table}")
        for team_id, object_id in expected["team_heads"].items():
            team = parse_record(
                TeamV1,
                str(
                    connection.execute(
                        "SELECT record_json FROM teams WHERE object_id = ?",
                        (object_id,),
                    ).fetchone()[0],
                ),
                "Team",
            )
            connection.execute(
                "INSERT INTO team_heads VALUES (?, ?, ?, ?, ?)",
                (
                    team_id,
                    team.object_id,
                    team.roster_ref.object_id,
                    team.task_graph_ref.object_id,
                    team.authority_ref.object_id,
                ),
            )
        for (team_id, task_id), object_id in expected["task_heads"].items():
            event = self._load(
                connection,
                "task_events",
                object_id,
                TeamTaskEventV1,
            )
            connection.execute(
                "INSERT INTO task_heads VALUES (?, ?, ?, ?)",
                (team_id, task_id, event.task_ref.object_id, object_id),
            )
        for table, key in (
            ("artifact_current", "head_id"),
            ("subscription_current", "subscription_id"),
            ("conflict_current", "conflict_id"),
            ("projection_current", "member_id"),
        ):
            connection.executemany(
                f"INSERT INTO {table} (team_id, {key}, object_id) VALUES (?, ?, ?)",
                ((*pair, value) for pair, value in expected[table].items()),
            )
        connection.executemany(
            "INSERT INTO checkpoint_current VALUES (?, ?)",
            expected["checkpoint_current"].items(),
        )


__all__ = ["TeamStoreProjection"]
