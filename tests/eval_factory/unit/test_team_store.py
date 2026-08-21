from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from test_support.team_runtime_fixtures import (
    audit,
    ref,
    seed_envelope,
    team_fixture,
)

from eval_factory.harness.contracts import sorted_refs
from eval_factory.team import (
    ArtifactSubscriptionV1,
    TeamBlackboardError,
    TeamConcurrencyError,
    TeamConflictStatusV1,
    TeamConflictV1,
    TeamConvergenceOutcomeV1,
    TeamCoordinator,
    TeamIdempotencyConflictError,
    TeamIntegrityError,
    TeamMailboxError,
    TeamMessageAudienceV1,
    TeamMessageKindV1,
    TeamMessageV1,
    TeamStaleLeaseError,
    TeamStore,
    TeamTaskStatusV1,
)


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 18, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _store(tmp_path, clock: _Clock | None = None) -> tuple[TeamStore, object]:
    fixture = team_fixture()
    store = TeamStore(tmp_path / "team.sqlite3", clock=clock)
    snapshot = store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    return store, snapshot


def test_create_replay_ready_claim_and_fenced_heartbeat(tmp_path) -> None:
    clock = _Clock()
    store, snapshot = _store(tmp_path, clock)
    fixture = team_fixture()
    replay = store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    assert replay == snapshot
    assert [work.task.task_id for work in store.ready_tasks("team-stage3")] == [
        "task-requirement",
    ]

    task = next(value for value in snapshot.graph.tasks if value.task_id == "task-requirement")
    member = next(value for value in snapshot.roster.members if value.member_id == "member-requirement")
    claim = store.claim_task(
        team_ref=snapshot.team.to_ref(),
        graph_ref=snapshot.graph.to_ref(),
        authority_ref=snapshot.authority.to_ref(),
        task_ref=task.to_ref(),
        member_ref=member.to_ref(),
        audit=audit(),
        idempotency_key="claim-requirement",
    )
    lease = store.acquire_lease(
        claim_ref=claim.to_ref(),
        lease_duration_seconds=30,
        audit=audit(),
        idempotency_key="lease-requirement",
    )
    clock.now += timedelta(seconds=5)
    heartbeat = store.heartbeat_lease(
        lease_ref=lease.to_ref(),
        fencing_token=lease.fencing_token,
        extension_seconds=60,
        audit=audit(),
        idempotency_key="heartbeat-requirement",
    )
    assert heartbeat.status_after is TeamTaskStatusV1.ACTIVE
    assert heartbeat.effective_expires_at == clock.now + timedelta(seconds=60)
    assert (
        store.get_task_work(
            "team-stage3",
            "task-requirement",
        ).fencing_token
        == 1
    )


def test_idempotency_conflict_and_expiry_retry_reject_stale_fence(tmp_path) -> None:
    clock = _Clock()
    store, snapshot = _store(tmp_path, clock)
    task = next(value for value in snapshot.graph.tasks if value.task_id == "task-requirement")
    member = next(value for value in snapshot.roster.members if value.member_id == "member-requirement")
    claim = store.claim_task(
        team_ref=snapshot.team.to_ref(),
        graph_ref=snapshot.graph.to_ref(),
        authority_ref=snapshot.authority.to_ref(),
        task_ref=task.to_ref(),
        member_ref=member.to_ref(),
        audit=audit(),
        idempotency_key="claim-trace",
    )
    with pytest.raises(TeamIdempotencyConflictError):
        store.claim_task(
            team_ref=snapshot.team.to_ref(),
            graph_ref=snapshot.graph.to_ref(),
            authority_ref=snapshot.authority.to_ref(),
            task_ref=task.to_ref(),
            member_ref=next(
                value for value in snapshot.roster.members if value.member_id == "member-trace"
            ).to_ref(),
            audit=audit(),
            idempotency_key="claim-trace",
        )
    lease = store.acquire_lease(
        claim_ref=claim.to_ref(),
        lease_duration_seconds=10,
        audit=audit(),
        idempotency_key="lease-trace",
    )
    clock.now += timedelta(seconds=10)
    retry = store.expire_lease(
        lease_ref=lease.to_ref(),
        fencing_token=lease.fencing_token,
        audit=audit(),
        idempotency_key="expire-trace",
    )
    assert retry.status_after is TeamTaskStatusV1.READY
    assert retry.reason_code == "LEASE_EXPIRED"
    with pytest.raises(TeamStaleLeaseError):
        store.heartbeat_lease(
            lease_ref=lease.to_ref(),
            fencing_token=lease.fencing_token,
            extension_seconds=30,
            audit=audit(),
            idempotency_key="late-heartbeat",
        )


def test_direct_message_is_contiguous_and_projection_scoped(tmp_path) -> None:
    store, snapshot = _store(tmp_path)
    quality = next(value for value in snapshot.roster.members if value.member_id == "member-quality")
    task_member = next(value for value in snapshot.roster.members if value.member_id == "member-task")
    task = next(value for value in snapshot.graph.tasks if value.task_id == "task-author")
    message = TeamMessageV1.create(
        message_id="quality-to-task-challenge",
        team_ref=snapshot.team.to_ref(),
        sender_member_ref=quality.to_ref(),
        audience=TeamMessageAudienceV1.DIRECT,
        recipient_member_refs=(task_member.to_ref(),),
        message_kind=TeamMessageKindV1.CHALLENGE,
        task_ref=task.to_ref(),
        message_body_ref=ref("team-message-body", "challenge"),
        artifact_envelope_refs=(),
        reply_to_message_ref=None,
        audit=audit(),
    )
    assert store.send_message(message, idempotency_key="send-challenge") == message
    page = store.list_messages("team-stage3")
    assert page.messages == (message,)
    assert page.next_sequence is None
    task_projection = store.projection_source("team-stage3", "member-task")
    trace_projection = store.projection_source("team-stage3", "member-trace")
    assert message in task_projection.messages
    assert message not in trace_projection.messages
    assert all(member.member_id != "member-coordinator" for member in (quality, task_member))
    assert (
        store.send_message(
            message,
            idempotency_key="send-challenge",
        )
        == message
    )
    with pytest.raises(ValueError, match="out of bounds"):
        store.list_messages("team-stage3", limit=0)
    unauthorized_system = TeamMessageV1.create(
        message_id="quality-system-message",
        team_ref=snapshot.team.to_ref(),
        sender_member_ref=quality.to_ref(),
        audience=TeamMessageAudienceV1.SYSTEM,
        recipient_member_refs=(),
        message_kind=TeamMessageKindV1.SYSTEM,
        task_ref=None,
        message_body_ref=ref("team-message-body", "system"),
        artifact_envelope_refs=(),
        reply_to_message_ref=None,
        audit=audit(),
    )
    with pytest.raises(TeamMailboxError, match="Coordinator"):
        store.send_message(
            unauthorized_system,
            idempotency_key="unauthorized-system",
        )


def test_checkpoint_restart_and_head_rebuild(tmp_path) -> None:
    store, snapshot = _store(tmp_path)
    checkpoint = store.create_checkpoint(
        "team-stage3",
        audit=audit(),
        idempotency_key="checkpoint-1",
    )
    restarted = TeamStore(tmp_path / "team.sqlite3")
    assert restarted.get_snapshot("team-stage3") == snapshot
    assert restarted.current_checkpoint("team-stage3") == checkpoint
    rebuilt = restarted.rebuild_current_heads()
    assert rebuilt.team_count == 1
    assert rebuilt.task_count == 0
    assert restarted.list_outbox("team-stage3")


def test_subscription_cursor_and_conflict_lifecycle(tmp_path) -> None:
    store, snapshot = _store(tmp_path)
    task_member = next(member for member in snapshot.roster.members if member.member_id == "member-task")
    subscription = ArtifactSubscriptionV1.create(
        subscription_id="subscription-task",
        team_ref=snapshot.team.to_ref(),
        member_ref=task_member.to_ref(),
        task_ids=("task-author",),
        artifact_head_ids=("head-goal",),
        semantic_roles=("evaluation-requirement",),
        cursor=0,
        audit=audit(),
    )
    current = store.create_subscription(
        subscription,
        idempotency_key="create-subscription",
    )
    fixture = team_fixture()
    envelope_a = seed_envelope(
        fixture,
        artifact_id="artifact-goal",
        semantic_role="evaluation-requirement",
        producer_capability_id="capability.requirement-planning",
    )
    envelope_b = seed_envelope(
        fixture,
        artifact_id="artifact-attachment",
        semantic_role="attachment-package",
        producer_capability_id="capability.attachment-reconstruction",
    )
    store.seed_artifact_head(
        team_ref=snapshot.team.to_ref(),
        head_id="head-goal",
        envelope=envelope_a,
        audit=audit(),
        idempotency_key="seed-conflict-goal",
    )
    assert (
        store.seed_artifact_head(
            team_ref=snapshot.team.to_ref(),
            head_id="head-goal",
            envelope=envelope_a,
            audit=audit(),
            idempotency_key="seed-conflict-goal",
        ).head_id
        == "head-goal"
    )
    with pytest.raises(TeamBlackboardError, match="unowned current"):
        store.seed_artifact_head(
            team_ref=snapshot.team.to_ref(),
            head_id="head-goal",
            envelope=envelope_a,
            audit=audit(),
            idempotency_key="seed-conflict-goal-again",
        )
    snapshot = store.get_snapshot("team-stage3")
    store.seed_artifact_head(
        team_ref=snapshot.team.to_ref(),
        head_id="head-attachment",
        envelope=envelope_b,
        audit=audit(),
        idempotency_key="seed-conflict-attachment",
    )
    current = store.advance_subscription(
        current_ref=current.to_ref(),
        cursor=1,
        audit=audit(),
        idempotency_key="advance-one",
    )
    assert current.cursor == 1
    assert (
        store.create_subscription(
            subscription,
            idempotency_key="create-subscription",
        )
        == subscription
    )
    current_snapshot = store.get_snapshot("team-stage3")
    invalid_subscription = ArtifactSubscriptionV1.create(
        subscription_id="subscription-invalid",
        team_ref=current_snapshot.team.to_ref(),
        member_ref=task_member.to_ref(),
        task_ids=("task-author",),
        artifact_head_ids=("head-goal",),
        semantic_roles=("evaluation-requirement",),
        cursor=1,
        audit=audit(),
    )
    with pytest.raises(TeamBlackboardError, match="selector"):
        store.create_subscription(
            invalid_subscription,
            idempotency_key="create-invalid-subscription",
        )
    with pytest.raises(TeamBlackboardError, match="stale or skips"):
        store.advance_subscription(
            current_ref=current.to_ref(),
            cursor=0,
            audit=audit(),
            idempotency_key="advance-backward",
        )

    task = next(task for task in current_snapshot.graph.tasks if task.task_id == "task-author")
    conflict = TeamConflictV1.create(
        conflict_id="conflict-task",
        team_ref=current_snapshot.team.to_ref(),
        task_ref=task.to_ref(),
        artifact_envelope_refs=(
            envelope_a.to_ref(),
            envelope_b.to_ref(),
        ),
        status=TeamConflictStatusV1.OPEN,
        resolution_artifact_ref=None,
        resolved_by_member_ref=None,
        audit=audit(),
    )
    conflict = store.open_conflict(
        conflict,
        opened_by_member_ref=task_member.to_ref(),
        idempotency_key="open-conflict",
    )
    assert store.open_conflicts("team-stage3") == (conflict,)
    assert (
        store.open_conflict(
            TeamConflictV1.create(
                conflict_id="conflict-task",
                team_ref=current_snapshot.team.to_ref(),
                task_ref=task.to_ref(),
                artifact_envelope_refs=(
                    envelope_a.to_ref(),
                    envelope_b.to_ref(),
                ),
                status=TeamConflictStatusV1.OPEN,
                resolution_artifact_ref=None,
                resolved_by_member_ref=None,
                audit=audit(),
            ),
            opened_by_member_ref=task_member.to_ref(),
            idempotency_key="open-conflict",
        )
        == conflict
    )
    with pytest.raises(TeamBlackboardError, match="must close"):
        store.close_conflict(
            conflict,
            current_ref=conflict.to_ref(),
            closed_by_member_ref=task_member.to_ref(),
            idempotency_key="close-with-open",
        )
    resolved = TeamConflictV1.create(
        conflict_id=conflict.conflict_id,
        team_ref=current_snapshot.team.to_ref(),
        task_ref=task.to_ref(),
        artifact_envelope_refs=conflict.artifact_envelope_refs,
        status=TeamConflictStatusV1.RESOLVED,
        resolution_artifact_ref=envelope_b.to_ref(),
        resolved_by_member_ref=task_member.to_ref(),
        audit=audit(),
    )
    store.close_conflict(
        resolved,
        current_ref=conflict.to_ref(),
        closed_by_member_ref=task_member.to_ref(),
        idempotency_key="resolve-conflict",
    )
    assert store.open_conflicts("team-stage3") == ()
    with pytest.raises(TeamBlackboardError, match="resolver"):
        store.close_conflict(
            TeamConflictV1.create(
                conflict_id="conflict-task",
                team_ref=current_snapshot.team.to_ref(),
                task_ref=task.to_ref(),
                artifact_envelope_refs=conflict.artifact_envelope_refs,
                status=TeamConflictStatusV1.RESOLVED,
                resolution_artifact_ref=envelope_b.to_ref(),
                resolved_by_member_ref=next(
                    member
                    for member in current_snapshot.roster.members
                    if member.member_id == "member-quality"
                ).to_ref(),
                audit=audit(),
            ),
            current_ref=resolved.to_ref(),
            closed_by_member_ref=task_member.to_ref(),
            idempotency_key="stale-resolver",
        )


def test_blackboard_read_helpers_reject_missing_inputs(tmp_path) -> None:
    store, _ = _store(tmp_path)
    assert store.current_artifact_envelopes("team-stage3") == ()
    with pytest.raises(TeamBlackboardError, match="missing"):
        store.task_input_envelopes("team-stage3", "task-author")


def test_cancellation_and_explicit_rebuild_repair(tmp_path) -> None:
    clock = _Clock()
    store, snapshot = _store(tmp_path, clock)
    task = next(value for value in snapshot.graph.tasks if value.task_id == "task-requirement")
    member = next(value for value in snapshot.roster.members if value.member_id == "member-requirement")
    claim = store.claim_task(
        team_ref=snapshot.team.to_ref(),
        graph_ref=snapshot.graph.to_ref(),
        authority_ref=snapshot.authority.to_ref(),
        task_ref=task.to_ref(),
        member_ref=member.to_ref(),
        audit=audit(),
        idempotency_key="claim-cancel",
    )
    lease = store.acquire_lease(
        claim_ref=claim.to_ref(),
        lease_duration_seconds=30,
        audit=audit(),
        idempotency_key="lease-cancel",
    )
    requested = store.request_cancellation(
        lease_ref=lease.to_ref(),
        fencing_token=lease.fencing_token,
        audit=audit(),
        idempotency_key="request-cancel",
    )
    assert requested.status_after is TeamTaskStatusV1.ACTIVE
    clock.now += timedelta(seconds=30)
    assert (
        store.recover_expired_leases(
            "team-stage3",
            audit=audit(),
        )
        == ()
    )
    cancelled = store.acknowledge_cancellation(
        lease_ref=lease.to_ref(),
        fencing_token=lease.fencing_token,
        audit=audit(),
        idempotency_key="ack-cancel",
    )
    assert cancelled.status_after is TeamTaskStatusV1.CANCELLED

    store.create_checkpoint(
        "team-stage3",
        audit=audit(),
        idempotency_key="checkpoint-cancel",
    )
    connection = sqlite3.connect(store.path)
    try:
        connection.execute("DELETE FROM checkpoint_current")
        connection.execute("DELETE FROM task_heads")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(TeamIntegrityError, match="task current head"):
        store.get_task_work("team-stage3", "task-requirement")
    with pytest.raises(TeamIntegrityError, match="checkpoint current head"):
        store.current_checkpoint("team-stage3")
    store.rebuild_current_heads(repair=True)
    assert store.get_snapshot("team-stage3") == snapshot
    assert store.get_task_work("team-stage3", "task-requirement").status is TeamTaskStatusV1.CANCELLED
    assert store.current_checkpoint("team-stage3") is not None


def test_declared_terminal_task_does_not_unblock_dependency(tmp_path) -> None:
    fixture = team_fixture()
    requirement = next(task for task in fixture.graph.tasks if task.task_id == "task-requirement")
    declared_complete = type(requirement).create(
        task_id=requirement.task_id,
        task_kind=requirement.task_kind,
        dependency_task_ids=requirement.dependency_task_ids,
        assigned_member_id=requirement.assigned_member_id,
        capability_definition_ref=requirement.capability_definition_ref,
        input_artifact_head_ids=requirement.input_artifact_head_ids,
        output_artifact_head_ids=requirement.output_artifact_head_ids,
        acceptance_check_refs=requirement.acceptance_check_refs,
        status=TeamTaskStatusV1.COMPLETED,
        max_attempts=requirement.max_attempts,
        max_model_requests=requirement.max_model_requests,
        max_model_tokens=requirement.max_model_tokens,
        max_cost_micro_usd=requirement.max_cost_micro_usd,
        audit=audit(),
    )
    graph = type(fixture.graph).create(
        graph_id=fixture.graph.graph_id,
        team_id=fixture.graph.team_id,
        revision=1,
        predecessor_graph_ref=None,
        roster_ref=fixture.roster.to_ref(),
        member_ids=fixture.graph.member_ids,
        tasks=tuple(declared_complete if task == requirement else task for task in fixture.graph.tasks),
        audit=audit(),
    )
    authority = type(fixture.authority).create(
        authority_id=fixture.authority.authority_id,
        authority_version=1,
        predecessor_authority_ref=None,
        team_id=fixture.authority.team_id,
        team_incarnation_id=fixture.authority.team_incarnation_id,
        roster_ref=fixture.roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        permission_policy_ref=fixture.authority.permission_policy_ref,
        grants=fixture.authority.grants,
        max_model_requests=fixture.authority.max_model_requests,
        max_model_tokens=fixture.authority.max_model_tokens,
        max_cost_micro_usd=fixture.authority.max_cost_micro_usd,
        used_model_requests=0,
        used_model_tokens=0,
        used_cost_micro_usd=0,
        audit=audit(),
    )
    team = type(fixture.team).create(
        team_id=fixture.team.team_id,
        team_incarnation_id=fixture.team.team_incarnation_id,
        team_version=1,
        goal_ref=fixture.team.goal_ref,
        composition_ref=fixture.team.composition_ref,
        roster_ref=fixture.roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        authority_ref=authority.to_ref(),
        coordinator_member_id=fixture.team.coordinator_member_id,
        lifecycle=fixture.team.lifecycle,
        audit=audit(),
    )
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=team,
        roster=fixture.roster,
        graph=graph,
        authority=authority,
        idempotency_key="create-team",
    )

    assert store.get_task_work("team-stage3", "task-trace").status is TeamTaskStatusV1.PENDING


def test_mailbox_rejects_coordinator_relay_scope_and_visibility_bypasses(
    tmp_path,
) -> None:
    store, snapshot = _store(tmp_path)
    members = {member.member_id: member for member in snapshot.roster.members}
    tasks = {task.task_id: task for task in snapshot.graph.tasks}

    coordinator_relay = TeamMessageV1.create(
        message_id="coordinator-relay",
        team_ref=snapshot.team.to_ref(),
        sender_member_ref=members["member-coordinator"].to_ref(),
        audience=TeamMessageAudienceV1.DIRECT,
        recipient_member_refs=(members["member-task"].to_ref(),),
        message_kind=TeamMessageKindV1.HANDOFF,
        task_ref=tasks["task-author"].to_ref(),
        message_body_ref=ref("team-message-body", "relay"),
        artifact_envelope_refs=(),
        reply_to_message_ref=None,
        audit=audit(),
    )
    with pytest.raises(TeamMailboxError, match="cannot relay"):
        store.send_message(
            coordinator_relay,
            idempotency_key="coordinator-relay",
        )

    unrelated = TeamMessageV1.create(
        message_id="unrelated-challenge",
        team_ref=snapshot.team.to_ref(),
        sender_member_ref=members["member-quality"].to_ref(),
        audience=TeamMessageAudienceV1.DIRECT,
        recipient_member_refs=(members["member-trace"].to_ref(),),
        message_kind=TeamMessageKindV1.CHALLENGE,
        task_ref=tasks["task-author"].to_ref(),
        message_body_ref=ref("team-message-body", "unrelated"),
        artifact_envelope_refs=(),
        reply_to_message_ref=None,
        audit=audit(),
    )
    with pytest.raises(TeamMailboxError, match="participant scope"):
        store.send_message(unrelated, idempotency_key="unrelated-challenge")

    envelope = seed_envelope(
        team_fixture(),
        artifact_id="artifact-goal",
        semantic_role="evaluation-requirement",
        producer_capability_id="capability.requirement-planning",
    )
    store.seed_artifact_head(
        team_ref=snapshot.team.to_ref(),
        head_id="head-goal",
        envelope=envelope,
        audit=audit(),
        idempotency_key="seed-goal",
    )
    current = store.get_snapshot("team-stage3")
    leaked = TeamMessageV1.create(
        message_id="recipient-visibility-bypass",
        team_ref=current.team.to_ref(),
        sender_member_ref=members["member-requirement"].to_ref(),
        audience=TeamMessageAudienceV1.DIRECT,
        recipient_member_refs=(members["member-trace"].to_ref(),),
        message_kind=TeamMessageKindV1.HANDOFF,
        task_ref=tasks["task-requirement"].to_ref(),
        message_body_ref=ref("team-message-body", "leak"),
        artifact_envelope_refs=(envelope.to_ref(),),
        reply_to_message_ref=None,
        audit=audit(),
    )
    with pytest.raises(TeamMailboxError, match="recipient visibility"):
        store.send_message(leaked, idempotency_key="recipient-visibility")


def test_outbox_scan_does_not_starve_after_delivered_prefix(tmp_path) -> None:
    store, _ = _store(tmp_path)
    first = store.list_outbox("team-stage3", limit=1)[0]
    for session_ref in first.intended_member_session_refs:
        store.mark_outbox_delivered(
            outbox_ref=first.to_ref(),
            session_ref=session_ref,
        )
    checkpoint = store.create_checkpoint(
        "team-stage3",
        audit=audit(),
        idempotency_key="checkpoint-after-delivery",
    )

    pending = store.pending_outbox("team-stage3", limit=1)

    assert len(pending) == 1
    assert pending[0][0].record_ref == checkpoint.to_ref()


def test_conflict_requires_current_team_and_current_artifacts(tmp_path) -> None:
    store, snapshot = _store(tmp_path)
    task_member = next(member for member in snapshot.roster.members if member.member_id == "member-task")
    task = next(task for task in snapshot.graph.tasks if task.task_id == "task-author")
    envelope = seed_envelope(
        team_fixture(),
        artifact_id="artifact-goal",
        semantic_role="evaluation-requirement",
        producer_capability_id="capability.requirement-planning",
    )
    store.seed_artifact_head(
        team_ref=snapshot.team.to_ref(),
        head_id="head-goal",
        envelope=envelope,
        audit=audit(),
        idempotency_key="seed-goal",
    )
    stale = TeamConflictV1.create(
        conflict_id="stale-conflict",
        team_ref=snapshot.team.to_ref(),
        task_ref=task.to_ref(),
        artifact_envelope_refs=sorted_refs(
            (
                envelope.to_ref(),
                ref("artifact-envelope", "not-current"),
            ),
        ),
        status=TeamConflictStatusV1.OPEN,
        resolution_artifact_ref=None,
        resolved_by_member_ref=None,
        audit=audit(),
    )
    with pytest.raises(TeamConcurrencyError, match="stale Team"):
        store.open_conflict(
            stale,
            opened_by_member_ref=task_member.to_ref(),
            idempotency_key="stale-conflict",
        )

    current = store.get_snapshot("team-stage3")
    noncurrent = TeamConflictV1.create(
        conflict_id="noncurrent-conflict",
        team_ref=current.team.to_ref(),
        task_ref=task.to_ref(),
        artifact_envelope_refs=sorted_refs(
            (
                envelope.to_ref(),
                ref("artifact-envelope", "not-current"),
            ),
        ),
        status=TeamConflictStatusV1.OPEN,
        resolution_artifact_ref=None,
        resolved_by_member_ref=None,
        audit=audit(),
    )
    with pytest.raises(TeamBlackboardError, match="not current"):
        store.open_conflict(
            noncurrent,
            opened_by_member_ref=task_member.to_ref(),
            idempotency_key="noncurrent-conflict",
        )


def test_rebuild_rejects_drifted_immutable_head_history(tmp_path) -> None:
    store, snapshot = _store(tmp_path)
    store.seed_artifact_head(
        team_ref=snapshot.team.to_ref(),
        head_id="head-goal",
        envelope=seed_envelope(
            team_fixture(),
            artifact_id="artifact-goal",
            semantic_role="evaluation-requirement",
            producer_capability_id="capability.requirement-planning",
        ),
        audit=audit(),
        idempotency_key="seed-goal",
    )
    connection = sqlite3.connect(store.path)
    try:
        connection.execute(
            "UPDATE artifact_heads SET revision = 9 WHERE head_id = ?",
            ("head-goal",),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(TeamIntegrityError, match="Head history"):
        store.rebuild_current_heads(repair=True)


@pytest.mark.parametrize(
    "fault_point",
    (
        "seed_artifact_head.after_envelope",
        "seed_artifact_head.after_head",
        "seed_artifact_head.after_authority",
        "seed_artifact_head.after_outbox",
        "seed_artifact_head.after_idempotency",
    ),
)
def test_seed_transaction_rolls_back_at_every_boundary(
    tmp_path,
    fault_point: str,
) -> None:
    fixture = team_fixture()

    def inject(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("injected seed fault")

    store = TeamStore(
        tmp_path / "team.sqlite3",
        fault_injector=inject,
    )
    snapshot = store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    with pytest.raises(RuntimeError, match="seed fault"):
        store.seed_artifact_head(
            team_ref=snapshot.team.to_ref(),
            head_id="head-goal",
            envelope=seed_envelope(
                fixture,
                artifact_id="artifact-goal",
                semantic_role="evaluation-requirement",
                producer_capability_id="capability.requirement-planning",
            ),
            audit=audit(),
            idempotency_key="seed-goal",
        )

    restarted = TeamStore(tmp_path / "team.sqlite3")
    assert restarted.get_snapshot("team-stage3") == snapshot
    assert restarted.current_artifact_heads("team-stage3") == ()
    assert len(restarted.list_outbox("team-stage3")) == 1


@pytest.mark.parametrize(
    "fault_point",
    (
        "claim_task.after_claim",
        "claim_task.after_event",
        "claim_task.after_outbox",
        "claim_task.after_idempotency",
    ),
)
def test_claim_transaction_rolls_back_at_every_boundary(
    tmp_path,
    fault_point: str,
) -> None:
    fixture = team_fixture()

    def inject(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("injected claim fault")

    store = TeamStore(
        tmp_path / "team.sqlite3",
        fault_injector=inject,
    )
    snapshot = store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    task = next(task for task in snapshot.graph.tasks if task.task_id == "task-requirement")
    member = next(member for member in snapshot.roster.members if member.member_id == "member-requirement")
    with pytest.raises(RuntimeError, match="claim fault"):
        store.claim_task(
            team_ref=snapshot.team.to_ref(),
            graph_ref=snapshot.graph.to_ref(),
            authority_ref=snapshot.authority.to_ref(),
            task_ref=task.to_ref(),
            member_ref=member.to_ref(),
            audit=audit(),
            idempotency_key="claim-requirement",
        )

    restarted = TeamStore(tmp_path / "team.sqlite3")
    assert (
        restarted.get_task_work(
            "team-stage3",
            "task-requirement",
        ).status
        is TeamTaskStatusV1.READY
    )
    assert len(restarted.list_outbox("team-stage3")) == 1


def test_coordinator_uses_current_conflict_work_and_head_authority(
    tmp_path,
) -> None:
    store, _ = _store(tmp_path)
    coordinator = TeamCoordinator(store)
    assert (
        coordinator.converge(
            "team-stage3",
            audit=audit(),
            decision_key="continue-ready",
        ).outcome
        is TeamConvergenceOutcomeV1.CONTINUE
    )

    fixture = team_fixture()
    envelopes = (
        seed_envelope(
            fixture,
            artifact_id="artifact-goal",
            semantic_role="evaluation-requirement",
            producer_capability_id="capability.requirement-planning",
        ),
        seed_envelope(
            fixture,
            artifact_id="artifact-attachment",
            semantic_role="attachment-package",
            producer_capability_id="capability.attachment-reconstruction",
        ),
    )
    for head_id, envelope in zip(
        ("head-goal", "head-attachment"),
        envelopes,
        strict=True,
    ):
        store.seed_artifact_head(
            team_ref=store.get_snapshot("team-stage3").team.to_ref(),
            head_id=head_id,
            envelope=envelope,
            audit=audit(),
            idempotency_key=f"seed-{head_id}",
        )
    current = store.get_snapshot("team-stage3")
    task_member = next(member for member in current.roster.members if member.member_id == "member-task")
    task = next(task for task in current.graph.tasks if task.task_id == "task-author")
    conflict = TeamConflictV1.create(
        conflict_id="coordinator-blocking-conflict",
        team_ref=current.team.to_ref(),
        task_ref=task.to_ref(),
        artifact_envelope_refs=sorted_refs(envelope.to_ref() for envelope in envelopes),
        status=TeamConflictStatusV1.OPEN,
        resolution_artifact_ref=None,
        resolved_by_member_ref=None,
        audit=audit(),
    )
    conflict = store.open_conflict(
        conflict,
        opened_by_member_ref=task_member.to_ref(),
        idempotency_key="open-coordinator-conflict",
    )
    assert (
        coordinator.converge(
            "team-stage3",
            audit=audit(),
            decision_key="blocked-conflict",
        ).outcome
        is TeamConvergenceOutcomeV1.BLOCKED
    )
    store.close_conflict(
        TeamConflictV1.create(
            conflict_id=conflict.conflict_id,
            team_ref=current.team.to_ref(),
            task_ref=task.to_ref(),
            artifact_envelope_refs=conflict.artifact_envelope_refs,
            status=TeamConflictStatusV1.RESOLVED,
            resolution_artifact_ref=envelopes[1].to_ref(),
            resolved_by_member_ref=task_member.to_ref(),
            audit=audit(),
        ),
        current_ref=conflict.to_ref(),
        closed_by_member_ref=task_member.to_ref(),
        idempotency_key="close-coordinator-conflict",
    )

    requirement = next(task for task in current.graph.tasks if task.task_id == "task-requirement")
    requirement_member = next(
        member for member in current.roster.members if member.member_id == "member-requirement"
    )
    claim = store.claim_task(
        team_ref=current.team.to_ref(),
        graph_ref=current.graph.to_ref(),
        authority_ref=current.authority.to_ref(),
        task_ref=requirement.to_ref(),
        member_ref=requirement_member.to_ref(),
        audit=audit(),
        idempotency_key="claim-terminal",
    )
    lease = store.acquire_lease(
        claim_ref=claim.to_ref(),
        lease_duration_seconds=30,
        audit=audit(),
        idempotency_key="lease-terminal",
    )
    store.request_cancellation(
        lease_ref=lease.to_ref(),
        fencing_token=lease.fencing_token,
        audit=audit(),
        idempotency_key="request-terminal",
    )
    store.acknowledge_cancellation(
        lease_ref=lease.to_ref(),
        fencing_token=lease.fencing_token,
        audit=audit(),
        idempotency_key="ack-terminal",
    )
    assert (
        coordinator.converge(
            "team-stage3",
            audit=audit(),
            decision_key="blocked-terminal",
        ).outcome
        is TeamConvergenceOutcomeV1.BLOCKED
    )


def test_public_reads_reject_missing_team_and_subscription_heads(
    tmp_path,
) -> None:
    store, snapshot = _store(tmp_path)
    task_member = next(member for member in snapshot.roster.members if member.member_id == "member-task")
    subscription = ArtifactSubscriptionV1.create(
        subscription_id="subscription-task",
        team_ref=snapshot.team.to_ref(),
        member_ref=task_member.to_ref(),
        task_ids=("task-author",),
        artifact_head_ids=(),
        semantic_roles=(),
        cursor=0,
        audit=audit(),
    )
    store.create_subscription(
        subscription,
        idempotency_key="create-subscription",
    )
    connection = sqlite3.connect(store.path)
    try:
        connection.execute("DELETE FROM subscription_current")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(TeamIntegrityError, match="subscription_current"):
        store.projection_source("team-stage3", "member-task")

    store.rebuild_current_heads(repair=True)
    connection = sqlite3.connect(store.path)
    try:
        connection.execute("DELETE FROM team_heads")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(TeamIntegrityError, match="head is missing"):
        store.get_snapshot("team-stage3")
    store.rebuild_current_heads(repair=True)
    assert store.get_snapshot("team-stage3").team == snapshot.team
