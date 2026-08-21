from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.artifacts import ArtifactHeadV1
from eval_factory.harness.capability import (
    CapabilityInvocationOutcomeV1,
)
from eval_factory.harness.capability_runtime import CapabilityRuntimeInvocation
from eval_factory.harness.interaction import (
    ExecutionAuthorityV1,
    MemberExecutionGrantV1,
    PermissionOutcomeV1,
)
from eval_factory.team._store_codec import parse_record, record_json, request_sha256
from eval_factory.team._store_db import TeamStoreDatabase
from eval_factory.team._store_types import (
    TeamConcurrencyError,
    TeamIntegrityError,
    TeamSnapshot,
    TeamStaleLeaseError,
    TeamTaskCompletion,
    TeamTaskStateError,
    TeamTaskWork,
)
from eval_factory.team.models import (
    TeamOutboxKindV1,
    TeamRosterV1,
    TeamTaskClaimV1,
    TeamTaskEventKindV1,
    TeamTaskEventV1,
    TeamTaskGraphV1,
    TeamTaskLeaseV1,
    TeamTaskStatusV1,
    TeamTaskV1,
    TeamV1,
)


class TeamStoreAuthority(TeamStoreDatabase):
    """Owns immutable Team composition and fenced task lifecycle authority."""

    def create_team(
        self,
        *,
        team: TeamV1,
        roster: TeamRosterV1,
        graph: TeamTaskGraphV1,
        authority: ExecutionAuthorityV1,
        idempotency_key: str,
    ) -> TeamSnapshot:
        snapshot = TeamSnapshot(team, roster, graph, authority)
        self._validate_bundle(snapshot)
        digest = request_sha256(
            {
                "team_ref": team.to_ref(),
                "roster_ref": roster.to_ref(),
                "graph_ref": graph.to_ref(),
                "authority_ref": authority.to_ref(),
            },
        )
        scope = f"create-team:{team.team_id}"
        with self._write() as connection:
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                if replay != ("TEAM", team.object_id):
                    raise TeamIntegrityError("Team replay response drifted")
                return self._snapshot_at(connection, team.to_ref())
            self._insert_bundle(connection, snapshot)
            self._fault("create_team.after_records")
            connection.execute(
                "INSERT INTO team_heads VALUES (?, ?, ?, ?, ?)",
                (
                    team.team_id,
                    team.object_id,
                    roster.object_id,
                    graph.object_id,
                    authority.object_id,
                ),
            )
            self._fault("create_team.after_head")
            self._outbox(
                connection,
                snapshot,
                TeamOutboxKindV1.TEAM_CHANGED,
                team.to_ref(),
            )
            self._fault("create_team.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "TEAM",
                team.object_id,
            )
            self._fault("create_team.after_idempotency")
            return snapshot

    def get_snapshot(self, team_id: str) -> TeamSnapshot:
        with self._read() as connection:
            return self._snapshot(connection, team_id)

    def get_snapshot_at(self, team_ref: ObjectRef) -> TeamSnapshot:
        with self._read() as connection:
            return self._snapshot_at(connection, team_ref)

    def list_task_work(self, team_id: str) -> tuple[TeamTaskWork, ...]:
        with self._read() as connection:
            return self._work(connection, self._snapshot(connection, team_id))

    def get_task_work(self, team_id: str, task_id: str) -> TeamTaskWork:
        for work in self.list_task_work(team_id):
            if work.task.task_id == task_id:
                return work
        raise TeamTaskStateError("Team task was not found")

    def ready_tasks(self, team_id: str) -> tuple[TeamTaskWork, ...]:
        return tuple(work for work in self.list_task_work(team_id) if work.status is TeamTaskStatusV1.READY)

    def task_usage(
        self,
        team_id: str,
        task_id: str,
    ) -> tuple[int, int, int]:
        with self._read() as connection:
            snapshot = self._snapshot(connection, team_id)
            self._task(snapshot, task_id)
            return self._task_usage(connection, team_id, task_id)

    def claim_task(
        self,
        *,
        team_ref: ObjectRef,
        graph_ref: ObjectRef,
        authority_ref: ObjectRef,
        task_ref: ObjectRef,
        member_ref: ObjectRef,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> TeamTaskClaimV1:
        team_id = self._team_id(team_ref)
        digest = request_sha256(
            {
                "team_ref": team_ref,
                "graph_ref": graph_ref,
                "authority_ref": authority_ref,
                "task_ref": task_ref,
                "member_ref": member_ref,
            },
        )
        scope = f"claim-task:{team_id}:{task_ref.object_id}"
        with self._write() as connection:
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "CLAIM",
                    "task_claims",
                    TeamTaskClaimV1,
                )
            snapshot = self._require_refs(
                connection,
                team_id,
                team_ref,
                graph_ref,
                authority_ref,
            )
            task = self._task_ref(snapshot, task_ref)
            member = self._member_ref(snapshot, member_ref)
            work = self._work_one(connection, snapshot, task.task_id)
            if task.assigned_member_id != member.member_id:
                raise TeamTaskStateError("task is assigned to another member")
            if work.status is not TeamTaskStatusV1.READY:
                raise TeamTaskStateError("only ready work may be claimed")
            attempt = work.attempt + 1
            if attempt > task.max_attempts:
                raise TeamTaskStateError("task attempt budget is exhausted")
            claim = TeamTaskClaimV1.create(
                claim_id=f"{team_id}.{task.task_id}.attempt-{attempt}",
                team_ref=team_ref,
                task_ref=task_ref,
                member_ref=member_ref,
                graph_revision=snapshot.graph.revision,
                claimed_at=self._clock(),
                audit=audit,
            )
            connection.execute(
                "INSERT INTO task_claims VALUES (?, ?, ?, ?, ?)",
                (
                    claim.object_id,
                    team_id,
                    task.task_id,
                    attempt,
                    record_json(claim),
                ),
            )
            self._fault("claim_task.after_claim")
            event = self._event(
                snapshot,
                task,
                work.event_version + 1,
                attempt,
                TeamTaskEventKindV1.CLAIMED,
                TeamTaskStatusV1.CLAIMED,
                audit,
                claim=claim,
            )
            self._append_task_event(connection, team_id, task, event)
            self._fault("claim_task.after_event")
            self._outbox(
                connection,
                snapshot,
                TeamOutboxKindV1.TEAM_CHANGED,
                snapshot.team.to_ref(),
            )
            self._fault("claim_task.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "CLAIM",
                claim.object_id,
            )
            self._fault("claim_task.after_idempotency")
            return claim

    def acquire_lease(
        self,
        *,
        claim_ref: ObjectRef,
        lease_duration_seconds: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> TeamTaskLeaseV1:
        if not 1 <= lease_duration_seconds <= 86_400:
            raise ValueError("lease duration is out of bounds")
        digest = request_sha256(
            {
                "claim_ref": claim_ref,
                "lease_duration_seconds": lease_duration_seconds,
            },
        )
        with self._write() as connection:
            row = connection.execute(
                "SELECT team_id, task_id FROM task_claims WHERE object_id = ?",
                (claim_ref.object_id,),
            ).fetchone()
            if row is None:
                raise TeamTaskStateError("task claim was not found")
            team_id, task_id = str(row["team_id"]), str(row["task_id"])
            scope = f"acquire-lease:{team_id}:{task_id}"
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "LEASE",
                    "task_leases",
                    TeamTaskLeaseV1,
                )
            claim = self._load(
                connection,
                "task_claims",
                claim_ref.object_id,
                TeamTaskClaimV1,
            )
            if claim.to_ref() != claim_ref:
                raise TeamIntegrityError("task claim reference drifted")
            snapshot = self._snapshot(connection, team_id)
            work = self._work_one(connection, snapshot, task_id)
            if (
                claim.team_ref != snapshot.team.to_ref()
                or work.status is not TeamTaskStatusV1.CLAIMED
                or work.claim != claim
            ):
                raise TeamConcurrencyError("task claim is no longer current")
            fence = (
                int(
                    connection.execute(
                        "SELECT COALESCE(MAX(fence), 0) FROM task_leases WHERE team_id = ? AND task_id = ?",
                        (team_id, task_id),
                    ).fetchone()[0],
                )
                + 1
            )
            now = self._clock()
            lease = TeamTaskLeaseV1.create(
                lease_id=f"{team_id}.{task_id}.fence-{fence}",
                claim_ref=claim_ref,
                fencing_token=fence,
                acquired_at=now,
                expires_at=now + timedelta(seconds=lease_duration_seconds),
                audit=audit,
            )
            connection.execute(
                "INSERT INTO task_leases VALUES (?, ?, ?, ?, ?, ?)",
                (
                    lease.object_id,
                    team_id,
                    task_id,
                    work.attempt,
                    fence,
                    record_json(lease),
                ),
            )
            self._fault("acquire_lease.after_lease")
            task = self._task(snapshot, task_id)
            event = self._event(
                snapshot,
                task,
                work.event_version + 1,
                work.attempt,
                TeamTaskEventKindV1.LEASE_ACQUIRED,
                TeamTaskStatusV1.ACTIVE,
                audit,
                claim=claim,
                lease=lease,
                expires_at=lease.expires_at,
            )
            self._append_task_event(connection, team_id, task, event)
            self._fault("acquire_lease.after_event")
            self._outbox(
                connection,
                snapshot,
                TeamOutboxKindV1.TEAM_CHANGED,
                snapshot.team.to_ref(),
            )
            self._fault("acquire_lease.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "LEASE",
                lease.object_id,
            )
            self._fault("acquire_lease.after_idempotency")
            return lease

    def heartbeat_lease(
        self,
        *,
        lease_ref: ObjectRef,
        fencing_token: int,
        extension_seconds: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> TeamTaskEventV1:
        if not 1 <= extension_seconds <= 86_400:
            raise ValueError("heartbeat extension is out of bounds")
        return self._lease_event(
            lease_ref=lease_ref,
            fencing_token=fencing_token,
            audit=audit,
            idempotency_key=idempotency_key,
            operation="heartbeat",
            extension_seconds=extension_seconds,
        )

    def expire_lease(
        self,
        *,
        lease_ref: ObjectRef,
        fencing_token: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> TeamTaskEventV1:
        return self._lease_event(
            lease_ref=lease_ref,
            fencing_token=fencing_token,
            audit=audit,
            idempotency_key=idempotency_key,
            operation="expire",
        )

    def request_cancellation(
        self,
        *,
        lease_ref: ObjectRef,
        fencing_token: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> TeamTaskEventV1:
        return self._lease_event(
            lease_ref=lease_ref,
            fencing_token=fencing_token,
            audit=audit,
            idempotency_key=idempotency_key,
            operation="cancel-request",
        )

    def acknowledge_cancellation(
        self,
        *,
        lease_ref: ObjectRef,
        fencing_token: int,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> TeamTaskEventV1:
        return self._lease_event(
            lease_ref=lease_ref,
            fencing_token=fencing_token,
            audit=audit,
            idempotency_key=idempotency_key,
            operation="cancel-ack",
        )

    def _lease_event(
        self,
        *,
        lease_ref: ObjectRef,
        fencing_token: int,
        audit: ContractAudit,
        idempotency_key: str,
        operation: str,
        extension_seconds: int = 0,
    ) -> TeamTaskEventV1:
        digest = request_sha256(
            {
                "lease_ref": lease_ref,
                "fence": fencing_token,
                "operation": operation,
                "extension_seconds": extension_seconds,
            },
        )
        with self._write() as connection:
            team_id, task_id, lease = self._lease_identity(connection, lease_ref)
            scope = f"{operation}:{lease.object_id}"
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                return self._typed_replay(
                    connection,
                    replay,
                    "TASK_EVENT",
                    "task_events",
                    TeamTaskEventV1,
                )
            snapshot, task, work = self._active_lease(
                connection,
                team_id,
                task_id,
                lease,
                fencing_token,
                allow_expired=operation == "expire",
                allow_expired_cancellation=operation == "cancel-ack",
                allow_cancel=operation == "cancel-ack",
            )
            assert work.claim is not None
            assert work.effective_expires_at is not None
            now = self._clock()
            if operation == "heartbeat":
                expires = max(
                    work.effective_expires_at,
                    now + timedelta(seconds=extension_seconds),
                )
                event = self._event(
                    snapshot,
                    task,
                    work.event_version + 1,
                    work.attempt,
                    TeamTaskEventKindV1.HEARTBEAT,
                    TeamTaskStatusV1.ACTIVE,
                    audit,
                    claim=work.claim,
                    lease=lease,
                    expires_at=expires,
                )
            elif operation == "expire":
                if now < work.effective_expires_at:
                    raise TeamStaleLeaseError("task lease has not expired")
                expired = self._event(
                    snapshot,
                    task,
                    work.event_version + 1,
                    work.attempt,
                    TeamTaskEventKindV1.EXPIRED,
                    TeamTaskStatusV1.BLOCKED,
                    audit,
                    claim=work.claim,
                    lease=lease,
                    expires_at=work.effective_expires_at,
                    reason="LEASE_EXPIRED",
                )
                self._insert_event(connection, team_id, task.task_id, expired)
                event = self._retry_or_exhaust(
                    snapshot,
                    task,
                    expired.task_event_version + 1,
                    work.attempt,
                    audit,
                    claim=work.claim,
                    lease=lease,
                    expires_at=work.effective_expires_at,
                    reason="LEASE_EXPIRED",
                )
            elif operation == "cancel-request":
                event = self._event(
                    snapshot,
                    task,
                    work.event_version + 1,
                    work.attempt,
                    TeamTaskEventKindV1.CANCEL_REQUESTED,
                    TeamTaskStatusV1.ACTIVE,
                    audit,
                    claim=work.claim,
                    lease=lease,
                    expires_at=work.effective_expires_at,
                )
            else:
                if (
                    work.latest_event is None
                    or work.latest_event.event_kind is not TeamTaskEventKindV1.CANCEL_REQUESTED
                ):
                    raise TeamTaskStateError("cancellation was not requested")
                event = self._event(
                    snapshot,
                    task,
                    work.event_version + 1,
                    work.attempt,
                    TeamTaskEventKindV1.CANCELLED,
                    TeamTaskStatusV1.CANCELLED,
                    audit,
                    claim=work.claim,
                    lease=lease,
                    expires_at=work.effective_expires_at,
                    reason="CANCELLATION_ACKNOWLEDGED",
                )
            self._append_task_event(connection, team_id, task, event)
            self._fault(f"{operation}.after_event")
            self._outbox(
                connection,
                snapshot,
                TeamOutboxKindV1.TEAM_CHANGED,
                snapshot.team.to_ref(),
            )
            self._fault(f"{operation}.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "TASK_EVENT",
                event.object_id,
            )
            self._fault(f"{operation}.after_idempotency")
            return event

    def complete_task(
        self,
        *,
        team_ref: ObjectRef,
        graph_ref: ObjectRef,
        authority_ref: ObjectRef,
        lease_ref: ObjectRef,
        fencing_token: int,
        invocation: CapabilityRuntimeInvocation,
        output_bindings: tuple[tuple[str, Any], ...],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> TeamTaskCompletion:
        team_id = self._team_id(team_ref)
        digest = request_sha256(
            {
                "team_ref": team_ref,
                "graph_ref": graph_ref,
                "authority_ref": authority_ref,
                "lease_ref": lease_ref,
                "fence": fencing_token,
                "call_ref": invocation.call.to_ref(),
                "result_ref": invocation.result.to_ref(),
                "outputs": tuple((head_id, envelope.to_ref()) for head_id, envelope in output_bindings),
            },
        )
        with self._write() as connection:
            _, task_id, lease = self._lease_identity(connection, lease_ref)
            scope = f"complete-task:{team_id}:{task_id}"
            replay = self._replay(connection, scope, idempotency_key, digest)
            if replay is not None:
                if replay[0] != "RESULT":
                    raise TeamIntegrityError("completion replay type drifted")
                return self._completion(connection, team_id, replay[1])
            snapshot = self._require_refs(
                connection,
                team_id,
                team_ref,
                graph_ref,
                authority_ref,
            )
            _, task, work = self._active_lease(
                connection,
                team_id,
                task_id,
                lease,
                fencing_token,
            )
            self._validate_invocation(
                connection,
                snapshot,
                task,
                work,
                invocation,
                output_bindings,
            )
            connection.execute(
                "INSERT INTO capability_calls VALUES (?, ?, ?, ?)",
                (
                    invocation.call.object_id,
                    team_id,
                    task_id,
                    record_json(invocation.call),
                ),
            )
            self._fault("complete_task.after_call")
            connection.execute(
                "INSERT INTO capability_results VALUES (?, ?, ?, ?, ?)",
                (
                    invocation.result.object_id,
                    team_id,
                    task_id,
                    invocation.call.object_id,
                    record_json(invocation.result),
                ),
            )
            self._fault("complete_task.after_result")
            usage_deltas = self._usage_deltas(invocation)
            connection.execute(
                "INSERT INTO task_usage_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    invocation.result.object_id,
                    team_id,
                    task_id,
                    *usage_deltas,
                ),
            )
            self._fault("complete_task.after_usage")
            heads: tuple[ArtifactHeadV1, ...] = ()
            if invocation.result.outcome is CapabilityInvocationOutcomeV1.SUCCEEDED:
                heads = tuple(
                    self._publish_head(
                        connection,
                        snapshot,
                        task,
                        head_id,
                        envelope,
                        audit,
                    )
                    for head_id, envelope in output_bindings
                )
            self._fault("complete_task.after_heads")
            assert work.claim is not None
            assert work.lease is not None
            status = (
                TeamTaskStatusV1.COMPLETED
                if invocation.result.outcome is CapabilityInvocationOutcomeV1.SUCCEEDED
                else TeamTaskStatusV1.FAILED
            )
            completed = self._event(
                snapshot,
                task,
                work.event_version + 1,
                work.attempt,
                TeamTaskEventKindV1.COMPLETED,
                status,
                audit,
                claim=work.claim,
                lease=work.lease,
                expires_at=work.effective_expires_at,
                call_result=(invocation.call.to_ref(), invocation.result.to_ref()),
                reason=(invocation.result.failure_code if status is TeamTaskStatusV1.FAILED else None),
            )
            final = completed
            if status is TeamTaskStatusV1.FAILED:
                self._insert_event(connection, team_id, task_id, completed)
                final = self._retry_or_exhaust(
                    snapshot,
                    task,
                    completed.task_event_version + 1,
                    work.attempt,
                    audit,
                    claim=work.claim,
                    lease=work.lease,
                    expires_at=work.effective_expires_at,
                    reason=completed.reason_code,
                )
            self._append_task_event(connection, team_id, task, final)
            self._fault("complete_task.after_task_head")
            next_snapshot = self._advance_usage(
                connection,
                snapshot,
                invocation,
                audit,
            )
            connection.execute(
                "INSERT INTO completion_responses VALUES (?, ?, ?, ?, ?)",
                (
                    invocation.result.object_id,
                    team_id,
                    completed.object_id,
                    next_snapshot.team.object_id,
                    json.dumps(
                        [head.object_id for head in heads],
                        separators=(",", ":"),
                    ),
                ),
            )
            self._fault("complete_task.after_response")
            for head in heads:
                self._outbox(
                    connection,
                    next_snapshot,
                    TeamOutboxKindV1.ARTIFACT_HEAD_CHANGED,
                    head.to_ref(),
                )
            self._outbox(
                connection,
                next_snapshot,
                TeamOutboxKindV1.TEAM_CHANGED,
                next_snapshot.team.to_ref(),
            )
            self._fault("complete_task.after_outbox")
            self._remember(
                connection,
                scope,
                idempotency_key,
                digest,
                "RESULT",
                invocation.result.object_id,
            )
            self._fault("complete_task.after_idempotency")
            return TeamTaskCompletion(
                invocation.result,
                completed,
                heads,
                next_snapshot,
            )

    # Lifecycle replay helpers.

    def _work(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
    ) -> tuple[TeamTaskWork, ...]:
        values = {
            task.task_id: self._stored_work(connection, snapshot, task) for task in snapshot.graph.tasks
        }
        statuses = {task_id: work.status for task_id, work in values.items()}
        for task in snapshot.graph.tasks:
            if statuses[task.task_id] in {
                TeamTaskStatusV1.PENDING,
                TeamTaskStatusV1.READY,
            }:
                statuses[task.task_id] = (
                    TeamTaskStatusV1.READY
                    if all(
                        self._successfully_completed(values[dependency])
                        for dependency in task.dependency_task_ids
                    )
                    else TeamTaskStatusV1.PENDING
                )
        return tuple(
            TeamTaskWork(
                work.task,
                statuses[task_id],
                work.attempt,
                work.event_version,
                work.graph_revision,
                work.claim,
                work.lease,
                work.fencing_token,
                work.effective_expires_at,
                work.latest_event,
            )
            for task_id, work in values.items()
        )

    def _work_one(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
        task_id: str,
    ) -> TeamTaskWork:
        return next(work for work in self._work(connection, snapshot) if work.task.task_id == task_id)

    def _stored_work(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
        task: TeamTaskV1,
    ) -> TeamTaskWork:
        row = connection.execute(
            """
            SELECT task_ref, event_id FROM task_heads
            WHERE team_id = ? AND task_id = ?
            """,
            (snapshot.team.team_id, task.task_id),
        ).fetchone()
        latest = connection.execute(
            """
            SELECT object_id FROM task_events
            WHERE team_id = ? AND task_id = ?
            ORDER BY version DESC LIMIT 1
            """,
            (snapshot.team.team_id, task.task_id),
        ).fetchone()
        if row is None:
            if latest is not None:
                raise TeamIntegrityError("task current head is missing")
            coordinator_control = task.assigned_member_id == snapshot.team.coordinator_member_id
            return TeamTaskWork(
                task,
                (
                    task.status
                    if coordinator_control
                    and task.status
                    in {
                        TeamTaskStatusV1.COMPLETED,
                        TeamTaskStatusV1.FAILED,
                        TeamTaskStatusV1.CANCELLED,
                    }
                    else TeamTaskStatusV1.PENDING
                ),
                0,
                0,
                snapshot.graph.revision,
                None,
                None,
                None,
                None,
                None,
            )
        if latest is None or row["event_id"] != latest["object_id"] or row["task_ref"] != task.object_id:
            raise TeamIntegrityError("task current head drifted")
        event = self._load(
            connection,
            "task_events",
            str(row["event_id"]),
            TeamTaskEventV1,
        )
        claim = (
            self._load(
                connection,
                "task_claims",
                event.claim_ref.object_id,
                TeamTaskClaimV1,
            )
            if event.claim_ref
            else None
        )
        lease = (
            self._load(
                connection,
                "task_leases",
                event.lease_ref.object_id,
                TeamTaskLeaseV1,
            )
            if event.lease_ref
            else None
        )
        if ((event.claim_ref is not None and claim is not None) and claim.to_ref() != event.claim_ref) or (
            (event.lease_ref is not None and lease is not None) and lease.to_ref() != event.lease_ref
        ):
            raise TeamIntegrityError("task event lease authority drifted")
        if lease is not None and (
            claim is None or lease.claim_ref != claim.to_ref() or event.fencing_token != lease.fencing_token
        ):
            raise TeamIntegrityError("task event fence authority drifted")
        return TeamTaskWork(
            task,
            event.status_after,
            event.attempt,
            event.task_event_version,
            event.graph_revision,
            claim,
            lease,
            event.fencing_token,
            event.effective_expires_at,
            event,
        )

    @staticmethod
    def _successfully_completed(work: TeamTaskWork) -> bool:
        return (
            work.status is TeamTaskStatusV1.COMPLETED
            and work.latest_event is not None
            and work.latest_event.event_kind is TeamTaskEventKindV1.COMPLETED
            and work.latest_event.capability_result_ref is not None
        )

    def _event(
        self,
        snapshot: TeamSnapshot,
        task: TeamTaskV1,
        version: int,
        attempt: int,
        kind: TeamTaskEventKindV1,
        status: TeamTaskStatusV1,
        audit: ContractAudit,
        *,
        claim: TeamTaskClaimV1 | None = None,
        lease: TeamTaskLeaseV1 | None = None,
        expires_at: datetime | None = None,
        call_result: tuple[ObjectRef, ObjectRef] | None = None,
        reason: str | None = None,
    ) -> TeamTaskEventV1:
        return TeamTaskEventV1.create(
            event_key=f"{snapshot.team.team_id}.{task.task_id}.event-{version}",
            team_ref=snapshot.team.to_ref(),
            task_ref=task.to_ref(),
            graph_ref=snapshot.graph.to_ref(),
            graph_revision=snapshot.graph.revision,
            task_event_version=version,
            attempt=max(1, attempt),
            event_kind=kind,
            status_after=status,
            claim_ref=claim.to_ref() if claim else None,
            lease_ref=lease.to_ref() if lease else None,
            fencing_token=lease.fencing_token if lease else None,
            effective_expires_at=expires_at,
            capability_call_ref=call_result[0] if call_result else None,
            capability_result_ref=call_result[1] if call_result else None,
            reason_code=reason,
            occurred_at=self._clock(),
            audit=audit,
        )

    def _append_task_event(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        task: TeamTaskV1,
        event: TeamTaskEventV1,
    ) -> None:
        self._insert_event(connection, team_id, task.task_id, event)
        connection.execute(
            """
            INSERT INTO task_heads VALUES (?, ?, ?, ?)
            ON CONFLICT(team_id, task_id) DO UPDATE SET
              task_ref = excluded.task_ref, event_id = excluded.event_id
            """,
            (team_id, task.task_id, task.object_id, event.object_id),
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        team_id: str,
        task_id: str,
        event: TeamTaskEventV1,
    ) -> None:
        connection.execute(
            "INSERT INTO task_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.object_id,
                team_id,
                task_id,
                event.task_event_version,
                event.graph_revision,
                event.event_kind.value,
                record_json(event),
            ),
        )

    def _retry_or_exhaust(
        self,
        snapshot: TeamSnapshot,
        task: TeamTaskV1,
        version: int,
        attempt: int,
        audit: ContractAudit,
        *,
        claim: TeamTaskClaimV1,
        lease: TeamTaskLeaseV1,
        expires_at: datetime | None,
        reason: str | None,
    ) -> TeamTaskEventV1:
        if attempt < task.max_attempts:
            return self._event(
                snapshot,
                task,
                version,
                attempt,
                TeamTaskEventKindV1.RETRY_SCHEDULED,
                TeamTaskStatusV1.READY,
                audit,
                reason=reason,
            )
        return self._event(
            snapshot,
            task,
            version,
            attempt,
            TeamTaskEventKindV1.EXHAUSTED,
            TeamTaskStatusV1.FAILED,
            audit,
            claim=claim,
            lease=lease,
            expires_at=expires_at,
            reason="ATTEMPT_EXHAUSTED",
        )

    def _lease_identity(
        self,
        connection: sqlite3.Connection,
        reference: ObjectRef,
    ) -> tuple[str, str, TeamTaskLeaseV1]:
        row = connection.execute(
            "SELECT team_id, task_id, record_json FROM task_leases WHERE object_id = ?",
            (reference.object_id,),
        ).fetchone()
        if row is None:
            raise TeamTaskStateError("task lease was not found")
        lease = parse_record(
            TeamTaskLeaseV1,
            str(row["record_json"]),
            "task lease",
        )
        if lease.to_ref() != reference:
            raise TeamIntegrityError("task lease reference drifted")
        return str(row["team_id"]), str(row["task_id"]), lease

    def _active_lease(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        task_id: str,
        lease: TeamTaskLeaseV1,
        fence: int,
        *,
        allow_expired: bool = False,
        allow_expired_cancellation: bool = False,
        allow_cancel: bool = False,
    ) -> tuple[TeamSnapshot, TeamTaskV1, TeamTaskWork]:
        snapshot = self._snapshot(connection, team_id)
        task = self._task(snapshot, task_id)
        work = self._work_one(connection, snapshot, task_id)
        if work.status is not TeamTaskStatusV1.ACTIVE or work.lease != lease or work.fencing_token != fence:
            raise TeamStaleLeaseError("task lease or fence is stale")
        if (
            not allow_cancel
            and work.latest_event is not None
            and work.latest_event.event_kind is TeamTaskEventKindV1.CANCEL_REQUESTED
        ):
            raise TeamStaleLeaseError("task cancellation is pending")
        if (
            not allow_expired
            and not allow_expired_cancellation
            and work.effective_expires_at is not None
            and self._clock() >= work.effective_expires_at
        ):
            raise TeamStaleLeaseError("task lease is expired")
        return snapshot, task, work

    def _typed_replay[RecordT: BaseModel](
        self,
        connection: sqlite3.Connection,
        replay: tuple[str, str],
        expected: str,
        table: str,
        model: type[RecordT],
    ) -> RecordT:
        if replay[0] != expected:
            raise TeamIntegrityError("idempotent response type drifted")
        return self._load(connection, table, replay[1], model)

    @staticmethod
    def _insert_bundle(
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
    ) -> None:
        connection.execute(
            "INSERT INTO rosters VALUES (?, ?, ?, ?)",
            (
                snapshot.roster.object_id,
                snapshot.roster.team_id,
                snapshot.roster.revision,
                record_json(snapshot.roster),
            ),
        )
        connection.execute(
            "INSERT INTO graphs VALUES (?, ?, ?, ?)",
            (
                snapshot.graph.object_id,
                snapshot.graph.team_id,
                snapshot.graph.revision,
                record_json(snapshot.graph),
            ),
        )
        connection.execute(
            "INSERT INTO authorities VALUES (?, ?, ?, ?)",
            (
                snapshot.authority.object_id,
                snapshot.authority.team_id,
                snapshot.authority.authority_version,
                record_json(snapshot.authority),
            ),
        )
        connection.execute(
            "INSERT INTO teams VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.team.object_id,
                snapshot.team.team_id,
                snapshot.team.team_version,
                snapshot.roster.object_id,
                snapshot.graph.object_id,
                snapshot.authority.object_id,
                record_json(snapshot.team),
            ),
        )

    # Implemented by the collaboration module.
    def _publish_head(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _artifact_heads(
        self,
        connection: sqlite3.Connection,
        team_id: str,
    ) -> tuple[ArtifactHeadV1, ...]:
        raise NotImplementedError

    def _head_envelope(
        self,
        connection: sqlite3.Connection,
        head: ArtifactHeadV1,
    ) -> Any:
        raise NotImplementedError

    def _validate_invocation(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def _completion(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        result_id: str,
    ) -> TeamTaskCompletion:
        raise NotImplementedError

    def _advance_usage(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
        invocation: CapabilityRuntimeInvocation,
        audit: ContractAudit,
    ) -> TeamSnapshot:
        return self._advance_data_scope(
            connection,
            snapshot,
            audit,
            deltas=self._usage_deltas(invocation),
        )

    @staticmethod
    def _usage_deltas(
        invocation: CapabilityRuntimeInvocation,
    ) -> tuple[int, int, int]:
        action = invocation.permission_action
        charge = (
            invocation.permission_decision.outcome is PermissionOutcomeV1.ALLOW
            and invocation.result.outcome is not CapabilityInvocationOutcomeV1.BLOCKED_POLICY
        )
        return (
            action.model_requests_delta if charge else 0,
            action.model_tokens_delta if charge else 0,
            action.cost_micro_usd_delta if charge else 0,
        )

    @staticmethod
    def _task_usage(
        connection: sqlite3.Connection,
        team_id: str,
        task_id: str,
    ) -> tuple[int, int, int]:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(model_requests), 0),
                   COALESCE(SUM(model_tokens), 0),
                   COALESCE(SUM(cost_micro_usd), 0)
            FROM task_usage_events
            WHERE team_id = ? AND task_id = ?
            """,
            (team_id, task_id),
        ).fetchone()
        return int(row[0]), int(row[1]), int(row[2])

    def _advance_data_scope(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
        audit: ContractAudit,
        *,
        deltas: tuple[int, int, int] = (0, 0, 0),
    ) -> TeamSnapshot:
        current_envelopes = {
            head.head_id: self._head_envelope(connection, head).to_ref()
            for head in self._artifact_heads(connection, snapshot.team.team_id)
        }
        grants = []
        for grant in snapshot.authority.grants:
            input_ids = {
                head_id
                for task in snapshot.graph.tasks
                if task.task_id in grant.task_ids
                for head_id in task.input_artifact_head_ids
            }
            stable_refs = tuple(
                ref for ref in grant.data_scope_refs if ref.object_type != "artifact-envelope"
            )
            grants.append(
                MemberExecutionGrantV1(
                    member_id=grant.member_id,
                    principal_ref=grant.principal_ref,
                    capability_definition_refs=grant.capability_definition_refs,
                    provider_binding_refs=grant.provider_binding_refs,
                    task_ids=grant.task_ids,
                    data_scope_refs=tuple(
                        sorted(
                            (
                                *stable_refs,
                                *(
                                    current_envelopes[head_id]
                                    for head_id in sorted(input_ids)
                                    if head_id in current_envelopes
                                ),
                            ),
                            key=lambda ref: (
                                ref.object_type,
                                ref.object_id,
                                ref.object_version,
                                ref.object_sha256,
                            ),
                        ),
                    ),
                    data_purposes=grant.data_purposes,
                    data_classifications=grant.data_classifications,
                    allowed_side_effects=grant.allowed_side_effects,
                ),
            )
        normalized_grants = tuple(grants)
        if normalized_grants == snapshot.authority.grants and deltas == (0, 0, 0):
            return snapshot
        authority = ExecutionAuthorityV1.create(
            authority_id=snapshot.authority.authority_id,
            authority_version=snapshot.authority.authority_version + 1,
            predecessor_authority_ref=snapshot.authority.to_ref(),
            team_id=snapshot.team.team_id,
            team_incarnation_id=snapshot.team.team_incarnation_id,
            roster_ref=snapshot.roster.to_ref(),
            task_graph_ref=snapshot.graph.to_ref(),
            permission_policy_ref=snapshot.authority.permission_policy_ref,
            grants=normalized_grants,
            max_model_requests=snapshot.authority.max_model_requests,
            max_model_tokens=snapshot.authority.max_model_tokens,
            max_cost_micro_usd=snapshot.authority.max_cost_micro_usd,
            used_model_requests=(snapshot.authority.used_model_requests + deltas[0]),
            used_model_tokens=snapshot.authority.used_model_tokens + deltas[1],
            used_cost_micro_usd=(snapshot.authority.used_cost_micro_usd + deltas[2]),
            audit=audit,
        )
        team = TeamV1.create(
            team_id=snapshot.team.team_id,
            team_incarnation_id=snapshot.team.team_incarnation_id,
            team_version=snapshot.team.team_version + 1,
            goal_ref=snapshot.team.goal_ref,
            composition_ref=snapshot.team.composition_ref,
            roster_ref=snapshot.roster.to_ref(),
            task_graph_ref=snapshot.graph.to_ref(),
            authority_ref=authority.to_ref(),
            coordinator_member_id=snapshot.team.coordinator_member_id,
            lifecycle=snapshot.team.lifecycle,
            audit=audit,
        )
        connection.execute(
            "INSERT INTO authorities VALUES (?, ?, ?, ?)",
            (
                authority.object_id,
                authority.team_id,
                authority.authority_version,
                record_json(authority),
            ),
        )
        connection.execute(
            "INSERT INTO teams VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                team.object_id,
                team.team_id,
                team.team_version,
                snapshot.roster.object_id,
                snapshot.graph.object_id,
                authority.object_id,
                record_json(team),
            ),
        )
        connection.execute(
            "UPDATE team_heads SET team_id_ref = ?, authority_id = ? WHERE team_id = ?",
            (team.object_id, authority.object_id, team.team_id),
        )
        return TeamSnapshot(
            team,
            snapshot.roster,
            snapshot.graph,
            authority,
        )


__all__ = ["TeamStoreAuthority"]
