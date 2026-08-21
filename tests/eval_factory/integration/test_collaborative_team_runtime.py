from __future__ import annotations

import pytest
from test_support.team_runtime_fixtures import (
    audit,
    capability_runtime,
    ref,
    seed_envelope,
    team_fixture,
)

from eval_factory.harness import HarnessSessionStore
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentQualityCapabilityRequestV1,
    RequirementPlanningCapabilityRequestV1,
    TaskAuthoringCapabilityRequestV1,
    TraceIngestionCapabilityRequestV1,
)
from eval_factory.team import (
    TeamCapabilityRunner,
    TeamConvergenceOutcomeV1,
    TeamCoordinator,
    TeamMessageAudienceV1,
    TeamMessageKindV1,
    TeamMessageV1,
    TeamSessionReconciler,
    TeamStore,
)


async def test_peer_challenge_rework_converges_and_recovers(tmp_path) -> None:
    base = team_fixture()
    session_store = HarnessSessionStore(tmp_path / "sessions.sqlite3")
    session_refs = {}
    for role in ("coordinator", "requirement", "trace", "task", "quality"):
        projection = session_store.create_session(
            session_id=f"session-{role}",
            incarnation_id=f"session-{role}-incarnation-001",
            composition_ref=base.registration.composition.to_ref(),
            created_by="team-runtime-integration",
            idempotency_key=f"create-session-{role}",
            audit=audit(),
        )
        session_refs[role] = projection.session.session_ref
    fixture = team_fixture(session_refs)
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    for head_id, envelope in (
        (
            "head-goal",
            seed_envelope(
                fixture,
                artifact_id="artifact-goal",
                semantic_role="evaluation-requirement",
                producer_capability_id="capability.requirement-planning",
            ),
        ),
        (
            "head-attachment",
            seed_envelope(
                fixture,
                artifact_id="artifact-attachment",
                semantic_role="attachment-package",
                producer_capability_id="capability.attachment-reconstruction",
            ),
        ),
    ):
        store.seed_artifact_head(
            team_ref=store.get_snapshot("team-stage3").team.to_ref(),
            head_id=head_id,
            envelope=envelope,
            audit=audit(),
            idempotency_key=f"seed-{head_id}",
        )
    registry, runtime, providers = capability_runtime(fixture)
    runner = TeamCapabilityRunner(
        store=store,
        runtime=runtime,
        registry=registry,
    )
    await runner.execute(
        team_id="team-stage3",
        task_id="task-requirement",
        request=RequirementPlanningCapabilityRequestV1.create(
            plan_ref=ref("dataset-build-plan", "plan", version="v2"),
            policy_ref=ref("factory-run-policy", "policy", version="v2"),
            audit=audit(),
        ),
        audit=audit(),
    )
    await runner.execute(
        team_id="team-stage3",
        task_id="task-trace",
        request=TraceIngestionCapabilityRequestV1.create(
            trace_source_ref=ref("trace-source", "source", version="v2"),
            job_id="job-stage3",
            attempt=1,
            audit=audit(),
        ),
        audit=audit(),
    )
    task_request = TaskAuthoringCapabilityRequestV1.create(
        decision_ref=ref("trace-candidate-decision", "decision", version="v2"),
        extracted_prompt_ref=ref(
            "extracted-user-prompt",
            "prompt",
            version="v2",
        ),
        intent_ref=ref("inferred-user-intent", "intent", version="v2"),
        rewrite_ref=ref("task-rewrite-candidate", "rewrite", version="v2"),
        requirement_ref=ref(
            "evaluation-requirement-spec",
            "requirement",
            version="v2",
        ),
        audit=audit(),
    )
    await runner.execute(
        team_id="team-stage3",
        task_id="task-author",
        request=task_request,
        audit=audit(),
    )
    quality_request = AttachmentQualityCapabilityRequestV1.create(
        subgraph_result_ref=ref(
            "attachment-subgraph-result",
            "subgraph",
            version="v2",
        ),
        group_result_refs=(ref("attachment-group-result", "group", version="v2"),),
        work_envelope_refs=(ref("agent-result-envelope", "work", version="v2"),),
        source_validation_ref=ref(
            "deterministic-item-validation-result",
            "validation",
            version="v2",
        ),
        item_quality_ref=ref(
            "item-quality-compilation-result",
            "quality",
            version="v2",
        ),
        audit=audit(),
    )
    await runner.execute(
        team_id="team-stage3",
        task_id="task-quality",
        request=quality_request,
        audit=audit(),
    )

    snapshot = store.get_snapshot("team-stage3")
    quality = next(member for member in snapshot.roster.members if member.member_id == "member-quality")
    task_member = next(member for member in snapshot.roster.members if member.member_id == "member-task")
    task = next(task for task in snapshot.graph.tasks if task.task_id == "task-author")
    task_envelope = next(
        envelope
        for head, envelope in store.artifact_head_envelopes("team-stage3")
        if head.head_id == "head-task"
    )
    challenge = TeamMessageV1.create(
        message_id="quality-challenge-task-v1",
        team_ref=snapshot.team.to_ref(),
        sender_member_ref=quality.to_ref(),
        audience=TeamMessageAudienceV1.DIRECT,
        recipient_member_refs=(task_member.to_ref(),),
        message_kind=TeamMessageKindV1.CHALLENGE,
        task_ref=task.to_ref(),
        message_body_ref=ref("team-message-body", "challenge"),
        artifact_envelope_refs=(task_envelope.to_ref(),),
        reply_to_message_ref=None,
        audit=audit(),
    )
    store.send_message(challenge, idempotency_key="quality-challenge")
    decision = TeamCoordinator(store).converge(
        "team-stage3",
        audit=audit(),
        decision_key="convergence-rework-1",
    )
    assert decision.outcome is TeamConvergenceOutcomeV1.REWORK
    assert decision.successor_graph_ref is not None
    assert store.get_snapshot("team-stage3").graph.revision == 2

    await runner.execute(
        team_id="team-stage3",
        task_id="task-author",
        request=task_request,
        audit=audit(),
    )
    await runner.execute(
        team_id="team-stage3",
        task_id="task-quality",
        request=quality_request,
        audit=audit(),
    )
    current = store.get_snapshot("team-stage3")
    response = TeamMessageV1.create(
        message_id="quality-close-task-v2",
        team_ref=current.team.to_ref(),
        sender_member_ref=quality.to_ref(),
        audience=TeamMessageAudienceV1.DIRECT,
        recipient_member_refs=(task_member.to_ref(),),
        message_kind=TeamMessageKindV1.REVIEW_RESPONSE,
        task_ref=task.to_ref(),
        message_body_ref=ref("team-message-body", "closed"),
        artifact_envelope_refs=(),
        reply_to_message_ref=challenge.to_ref(),
        audit=audit(),
    )
    store.send_message(response, idempotency_key="quality-close")
    complete = TeamCoordinator(store).converge(
        "team-stage3",
        audit=audit(),
        decision_key="convergence-complete-2",
    )
    assert complete.outcome is TeamConvergenceOutcomeV1.COMPLETE
    revisions = {head.head_id: head.revision for head in store.current_artifact_heads("team-stage3")}
    assert revisions["head-task"] == 2
    assert revisions["head-quality"] == 2
    assert providers["capability.task-authoring"].calls == 2
    assert providers["capability.quality-review"].calls == 2
    replayed_rework = store.revise_for_rework(
        "team-stage3",
        expected_team_ref=decision.team_ref,
        expected_graph_ref=decision.graph_ref,
        expected_authority_ref=decision.authority_ref,
        reset_task_ids=("task-author", "task-quality"),
        audit=audit(),
        idempotency_key="convergence-rework-1.rework",
    )
    assert replayed_rework.team.to_ref() == decision.successor_team_ref
    replayed_checkpoint = store.create_checkpoint(
        "team-stage3",
        audit=audit(),
        idempotency_key="convergence-rework-1.checkpoint",
    )
    assert replayed_checkpoint.to_ref() == decision.checkpoint_ref

    events = TeamSessionReconciler(
        team_store=store,
        session_sink=session_store,
    ).reconcile("team-stage3", audit=audit())
    assert events
    assert not store.pending_outbox("team-stage3")
    restarted = TeamStore(tmp_path / "team.sqlite3")
    rebuilt = restarted.rebuild_current_heads()
    assert rebuilt.artifact_count == 6
    assert restarted.current_checkpoint("team-stage3") is not None
    assert restarted.list_messages("team-stage3").messages == (
        challenge,
        response,
    )


class _CommitThenCrash:
    def __init__(self, store: HarnessSessionStore) -> None:
        self.store = store
        self.event = None

    def append_team_event(self, **kwargs):
        self.event = self.store.append_team_event(**kwargs)
        raise RuntimeError("session projection crash")


def test_session_outbox_retry_uses_historical_team_authority(tmp_path) -> None:
    base = team_fixture()
    session_store = HarnessSessionStore(tmp_path / "sessions.sqlite3")
    session_refs = {}
    for role in ("coordinator", "requirement", "trace", "task", "quality"):
        projection = session_store.create_session(
            session_id=f"session-{role}",
            incarnation_id=f"session-{role}-incarnation-001",
            composition_ref=base.registration.composition.to_ref(),
            created_by="team-runtime-integration",
            idempotency_key=f"create-session-{role}",
            audit=audit(),
        )
        session_refs[role] = projection.session.session_ref
    fixture = team_fixture(session_refs)
    team_store = TeamStore(tmp_path / "team.sqlite3")
    team_store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    crash_sink = _CommitThenCrash(session_store)
    with pytest.raises(RuntimeError, match="projection crash"):
        TeamSessionReconciler(
            team_store=team_store,
            session_sink=crash_sink,
        ).reconcile("team-stage3", audit=audit(), limit=1)
    assert crash_sink.event is not None

    team_store.seed_artifact_head(
        team_ref=fixture.team.to_ref(),
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
    retried = TeamSessionReconciler(
        team_store=team_store,
        session_sink=session_store,
    ).reconcile("team-stage3", audit=audit(), limit=1)

    assert retried == (crash_sink.event,)
