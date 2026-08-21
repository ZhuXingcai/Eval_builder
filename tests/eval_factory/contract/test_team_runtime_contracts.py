from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.team import (
    TeamConflictStatusV1,
    TeamConflictV1,
    TeamConvergenceDecisionV1,
    TeamConvergenceOutcomeV1,
    TeamOutboxKindV1,
    TeamOutboxRecordV1,
    TeamTaskEventKindV1,
    TeamTaskEventV1,
    TeamTaskStatusV1,
    validate_team_task_event_log,
)

HASH = "a" * 64


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
        created_by="team-runtime-contract-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage3",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _ref(object_type: str, suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v1",
        object_sha256=HASH,
    )


def _claimed(version: int = 1) -> TeamTaskEventV1:
    return TeamTaskEventV1.create(
        event_key=f"task-event-{version}",
        team_ref=_ref("agent-team", "team"),
        task_ref=_ref("team-task", "task"),
        graph_ref=_ref("team-task-graph", "graph"),
        graph_revision=1,
        task_event_version=version,
        attempt=1,
        event_kind=TeamTaskEventKindV1.CLAIMED,
        status_after=TeamTaskStatusV1.CLAIMED,
        claim_ref=_ref("team-task-claim", "claim"),
        lease_ref=None,
        fencing_token=None,
        effective_expires_at=None,
        capability_call_ref=None,
        capability_result_ref=None,
        reason_code=None,
        occurred_at=datetime(2026, 8, 18, tzinfo=UTC),
        audit=_audit(),
    )


def test_runtime_records_are_strict_frozen_and_ref_only() -> None:
    event = _claimed()
    with pytest.raises(ValidationError):
        TeamTaskEventV1.model_validate(
            {**event.model_dump(mode="python"), "inline_result": {}},
        )
    with pytest.raises(ValidationError, match="completed event"):
        TeamTaskEventV1.create(
            event_key="invalid-completion",
            team_ref=event.team_ref,
            task_ref=event.task_ref,
            graph_ref=event.graph_ref,
            graph_revision=1,
            task_event_version=2,
            attempt=1,
            event_kind=TeamTaskEventKindV1.COMPLETED,
            status_after=TeamTaskStatusV1.COMPLETED,
            claim_ref=_ref("team-task-claim", "claim"),
            lease_ref=_ref("team-task-lease", "lease"),
            fencing_token=1,
            effective_expires_at=datetime(2026, 8, 18, tzinfo=UTC),
            capability_call_ref=None,
            capability_result_ref=None,
            reason_code=None,
            occurred_at=datetime(2026, 8, 18, tzinfo=UTC),
            audit=_audit(),
        )


def test_task_event_log_rejects_gaps_and_illegal_transitions() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        validate_team_task_event_log((_claimed(version=2),))

    invalid = _claimed().model_copy(
        update={
            "event_kind": TeamTaskEventKindV1.HEARTBEAT,
            "status_after": TeamTaskStatusV1.ACTIVE,
        },
    )
    with pytest.raises(ValueError, match="illegal"):
        validate_team_task_event_log((invalid,))


@pytest.mark.parametrize(
    "reason_code",
    (None, "USER_APPROVAL_REQUIRED"),
)
def test_retry_event_preserves_optional_closed_cause(
    reason_code: str | None,
) -> None:
    event = TeamTaskEventV1.create(
        event_key="task-event-retry",
        team_ref=_ref("agent-team", "team"),
        task_ref=_ref("team-task", "task"),
        graph_ref=_ref("team-task-graph", "graph"),
        graph_revision=1,
        task_event_version=2,
        attempt=1,
        event_kind=TeamTaskEventKindV1.RETRY_SCHEDULED,
        status_after=TeamTaskStatusV1.READY,
        claim_ref=None,
        lease_ref=None,
        fencing_token=None,
        effective_expires_at=None,
        capability_call_ref=None,
        capability_result_ref=None,
        reason_code=reason_code,
        occurred_at=datetime(2026, 8, 18, tzinfo=UTC),
        audit=_audit(),
    )

    assert event.reason_code == reason_code


def test_convergence_rework_requires_all_successor_authorities() -> None:
    with pytest.raises(ValidationError, match="all successor"):
        TeamConvergenceDecisionV1.create(
            decision_key="rework-1",
            team_ref=_ref("agent-team", "team"),
            graph_ref=_ref("team-task-graph", "graph"),
            authority_ref=_ref("execution-authority", "authority"),
            checkpoint_ref=_ref("team-checkpoint", "checkpoint"),
            outcome=TeamConvergenceOutcomeV1.REWORK,
            blocking_refs=(_ref("team-message", "challenge"),),
            successor_team_ref=_ref("agent-team", "team-2"),
            successor_graph_ref=None,
            successor_authority_ref=None,
            reason_codes=("DIRECT_CHALLENGE_REWORK",),
            audit=_audit(),
        )


def test_outbox_kind_cannot_mislabel_message_or_checkpoint_authority() -> None:
    with pytest.raises(ValidationError, match="record type"):
        TeamOutboxRecordV1.create(
            outbox_key="outbox-1",
            team_id="team-1",
            sequence=1,
            event_kind=TeamOutboxKindV1.TEAM_MESSAGE_POSTED,
            aggregate_ref=_ref("agent-team", "team"),
            record_ref=_ref("execution-authority", "forbidden"),
            intended_member_session_refs=(_ref("harness-session", "member"),),
            occurred_at=datetime(2026, 8, 18, tzinfo=UTC),
            audit=_audit(),
        )


def test_superseded_conflict_requires_typed_resolution_evidence() -> None:
    with pytest.raises(ValidationError, match="closed conflict"):
        TeamConflictV1.create(
            conflict_id="conflict-1",
            team_ref=_ref("agent-team", "team"),
            task_ref=_ref("team-task", "task"),
            artifact_envelope_refs=(
                _ref("artifact-envelope", "a"),
                _ref("artifact-envelope", "b"),
            ),
            status=TeamConflictStatusV1.SUPERSEDED,
            resolution_artifact_ref=None,
            resolved_by_member_ref=None,
            audit=_audit(),
        )
