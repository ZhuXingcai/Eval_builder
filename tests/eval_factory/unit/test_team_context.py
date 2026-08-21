from __future__ import annotations

import sqlite3

import pytest
from test_support.team_runtime_fixtures import audit, ref, team_fixture

from eval_factory.harness.contracts import sorted_refs
from eval_factory.team import (
    TeamContextProjector,
    TeamIntegrityError,
    TeamMessageAudienceV1,
    TeamMessageKindV1,
    TeamMessageV1,
    TeamStore,
)


def test_member_projection_is_scoped_persisted_and_rebuildable(tmp_path) -> None:
    fixture = team_fixture()
    store = TeamStore(tmp_path / "team.sqlite3")
    snapshot = store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    quality = next(member for member in snapshot.roster.members if member.member_id == "member-quality")
    task_member = next(member for member in snapshot.roster.members if member.member_id == "member-task")
    task = next(task for task in snapshot.graph.tasks if task.task_id == "task-author")
    message = TeamMessageV1.create(
        message_id="challenge-task",
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
    store.send_message(message, idempotency_key="challenge-task")

    projector = TeamContextProjector(store)
    projection = projector.project(
        "team-stage3",
        "member-task",
        audit=audit(),
    )
    assert message.to_ref() in projection.message_refs
    assert all(reference.object_type != "execution-authority" for reference in projection.message_refs)
    assert (
        projector.validate_rebuild(
            "team-stage3",
            "member-task",
            audit=audit(),
        )
        == projection
    )


def test_projection_drift_is_rejected_not_auto_repaired(tmp_path) -> None:
    fixture = team_fixture()
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    projector = TeamContextProjector(store)
    projector.project("team-stage3", "member-task", audit=audit())
    connection = sqlite3.connect(store.path)
    try:
        connection.execute(
            "UPDATE projections SET fingerprint = ? WHERE member_id = ?",
            ("0" * 64, "member-task"),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(TeamIntegrityError, match="differs"):
        projector.validate_rebuild(
            "team-stage3",
            "member-task",
            audit=audit(),
        )


def test_store_rejects_caller_authored_projection_membership(tmp_path) -> None:
    fixture = team_fixture()
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    source = store.projection_source("team-stage3", "member-task")
    valid = TeamContextProjector(store).project(
        "team-stage3",
        "member-task",
        audit=audit(),
        persist=False,
    )
    coordinator_task = next(task for task in source.snapshot.graph.tasks if task.task_id == "task-coordinate")
    forged = type(valid).create(
        projection_id="forged-member-projection",
        team_ref=valid.team_ref,
        member_ref=valid.member_ref,
        authority_ref=valid.authority_ref,
        task_refs=sorted_refs(
            (*valid.task_refs, coordinator_task.to_ref()),
        ),
        artifact_envelope_refs=valid.artifact_envelope_refs,
        message_refs=valid.message_refs,
        subscription_refs=valid.subscription_refs,
        acceptance_check_refs=valid.acceptance_check_refs,
        audit=audit(),
    )

    with pytest.raises(TeamIntegrityError, match="Store projection"):
        store.persist_projection(
            forged,
            source_fingerprint=source.source_fingerprint,
        )
