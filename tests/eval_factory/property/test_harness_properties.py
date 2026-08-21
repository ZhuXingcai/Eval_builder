from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    HarnessSessionRefV1,
    SessionEventV1,
    SessionLifecycleEventKindV1,
    SessionLifecyclePayloadV1,
    validate_session_event_log,
)
from eval_factory.team import (
    TeamTaskGraphV1,
    TeamTaskStatusV1,
    TeamTaskV1,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 17, tzinfo=UTC),
        created_by="harness-property-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage0",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


@given(st.integers(min_value=1, max_value=30))
def test_session_event_log_accepts_only_contiguous_order(length: int) -> None:
    session = HarnessSessionRefV1(
        session_ref=_ref("harness-session"),
        incarnation_id="session-incarnation-property",
        session_version=1,
        composition_ref=_ref("harness-composition"),
    )
    events = tuple(
        SessionEventV1.create(
            event_id=f"session-event-{sequence:03d}",
            session=session,
            sequence=sequence,
            authority_version=1,
            turn_id=None,
            step_id=None,
            command_ref=None,
            payload=SessionLifecyclePayloadV1(
                event_kind=SessionLifecycleEventKindV1.SESSION_OPENED,
            ),
            occurred_at=datetime(2026, 8, 17, tzinfo=UTC) + timedelta(seconds=sequence),
            audit=_audit(),
        )
        for sequence in range(1, length + 1)
    )
    validate_session_event_log(events)

    broken = list(events)
    broken[-1] = broken[-1].model_copy(update={"sequence": length + 1})
    with pytest.raises(ValueError, match="contiguous"):
        validate_session_event_log(tuple(broken))


@given(st.integers(min_value=2, max_value=25))
def test_team_task_graph_chain_is_acyclic_and_back_edge_is_rejected(
    length: int,
) -> None:
    tasks = tuple(
        TeamTaskV1.create(
            task_id=f"task-{index:03d}",
            task_kind=f"kind-{index:03d}",
            dependency_task_ids=(() if index == 0 else (f"task-{index - 1:03d}",)),
            assigned_member_id="member-property",
            capability_definition_ref=_ref(
                "harness-capability-definition",
                f"capability-{index:03d}",
            ),
            input_artifact_head_ids=(() if index == 0 else (f"head-{index - 1:03d}",)),
            output_artifact_head_ids=(f"head-{index:03d}",),
            acceptance_check_refs=(_ref("acceptance-check", f"check-{index:03d}"),),
            status=(TeamTaskStatusV1.READY if index == 0 else TeamTaskStatusV1.PENDING),
            max_attempts=2,
            max_model_requests=0,
            max_model_tokens=0,
            max_cost_micro_usd=0,
            audit=_audit(),
        )
        for index in range(length)
    )
    graph = TeamTaskGraphV1.create(
        graph_id="team-task-graph.property",
        team_id="team-property",
        revision=1,
        predecessor_graph_ref=None,
        roster_ref=_ref("team-roster"),
        member_ids=("member-property",),
        tasks=tasks,
        audit=_audit(),
    )
    assert len(graph.tasks) == length

    first = tasks[0]
    cyclic_first = TeamTaskV1.create(
        task_id=first.task_id,
        task_kind=first.task_kind,
        dependency_task_ids=(f"task-{length - 1:03d}",),
        assigned_member_id=first.assigned_member_id,
        capability_definition_ref=first.capability_definition_ref,
        input_artifact_head_ids=first.input_artifact_head_ids,
        output_artifact_head_ids=first.output_artifact_head_ids,
        acceptance_check_refs=first.acceptance_check_refs,
        status=first.status,
        max_attempts=first.max_attempts,
        max_model_requests=first.max_model_requests,
        max_model_tokens=first.max_model_tokens,
        max_cost_micro_usd=first.max_cost_micro_usd,
        audit=_audit(),
    )
    cyclic_tasks = (cyclic_first, *tasks[1:])
    with pytest.raises(ValidationError, match="acyclic"):
        TeamTaskGraphV1.create(
            graph_id="team-task-graph.cycle",
            team_id="team-property",
            revision=1,
            predecessor_graph_ref=None,
            roster_ref=_ref("team-roster"),
            member_ids=("member-property",),
            tasks=cyclic_tasks,
            audit=_audit(),
        )
