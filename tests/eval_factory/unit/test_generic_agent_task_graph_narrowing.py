from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import audit, ref

from eval_factory.packs import (
    build_generic_agent_evaluation_blueprint,
    build_generic_agent_trace_execution_pack,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflow,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterializer,
)
from eval_factory.packs.generic_agent_trace.task_graph_narrowing import (
    GenericAgentTaskGraphNarrower,
    GenericAgentTaskGraphNarrowingError,
)
from eval_factory.team import TeamStore


def _materialized():
    sources = (
        ref("trace-source", "narrow-a", version="v1"),
        ref("trace-source", "narrow-b", version="v1"),
    )
    registration = build_generic_agent_trace_execution_pack(
        audit=audit(),
    )
    materialized = GenericAgentTaskGraphMaterializer().materialize(
        registration=registration,
        requirement_ref=ref(
            "evaluation-requirement-spec",
            "narrow",
            version="v2",
        ),
        source_refs=sources,
        independent_session_refs={
            role: ref("harness-session", f"narrow-{role}")
            for role in (
                "control",
                "coordinator",
                "quality",
                "requirement",
                "task",
                "trace",
            )
        },
        team_id="team-narrow",
        team_incarnation_id="team-narrow-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    return materialized, sources


def test_task_graph_narrowing_keeps_only_candidate_branch() -> None:
    materialized, sources = _materialized()
    result = GenericAgentTaskGraphNarrower().narrow(
        snapshot=materialized.authority.snapshot(),
        materialization=materialized,
        candidate_source_refs=(sources[0],),
        audit=audit(),
    )

    assert result.changed is True
    assert len(materialized.authority.graph.tasks) == 23
    assert len(result.snapshot.graph.tasks) == 15
    assert len(result.removed_task_ids) == 8
    assert result.snapshot.graph.predecessor_graph_ref == (materialized.authority.graph.to_ref())
    assert result.snapshot.team.team_incarnation_id == (materialized.authority.team.team_incarnation_id)
    batch = next(task for task in result.snapshot.graph.tasks if task.task_kind == "batch-quality")
    assert len(batch.dependency_task_ids) == 6
    retained = {task.task_id for task in result.snapshot.graph.tasks}
    assert all(set(grant.task_ids).issubset(retained) for grant in result.snapshot.authority.grants)

    replay = GenericAgentTaskGraphNarrower().narrow(
        snapshot=result.snapshot,
        materialization=materialized,
        candidate_source_refs=(sources[0],),
        audit=audit(),
    )
    assert replay.changed is False
    assert replay.snapshot == result.snapshot


def test_task_graph_narrowing_rejects_unknown_candidate() -> None:
    materialized, _sources = _materialized()
    with pytest.raises(
        GenericAgentTaskGraphNarrowingError,
        match="sorted subset",
    ):
        GenericAgentTaskGraphNarrower().narrow(
            snapshot=materialized.authority.snapshot(),
            materialization=materialized,
            candidate_source_refs=(ref("trace-source", "unknown", version="v1"),),
            audit=audit(),
        )


def test_factory_workflow_accepts_committed_narrowing(
    tmp_path,
) -> None:
    materialized, sources = _materialized()
    registration = build_generic_agent_trace_execution_pack(
        audit=audit(),
    )
    source = materialized.authority.snapshot()
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=source.team,
        roster=source.roster,
        graph=source.graph,
        authority=source.authority,
        idempotency_key="create-workflow-narrow",
    )
    narrowing = GenericAgentTaskGraphNarrower().narrow(
        snapshot=source,
        materialization=materialized,
        candidate_source_refs=(sources[0],),
        audit=audit(),
    )
    store.promote_narrowed_graph(
        source.team.team_id,
        expected_team_ref=source.team.to_ref(),
        expected_graph_ref=source.graph.to_ref(),
        expected_authority_ref=source.authority.to_ref(),
        successor=narrowing.snapshot,
        idempotency_key="promote-workflow-narrow",
    )
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=registration,
        requirement_ref=source.team.goal_ref,
        team_ref=source.team.to_ref(),
        roster_ref=source.roster.to_ref(),
        task_graph_ref=source.graph.to_ref(),
        authority_ref=source.authority.to_ref(),
        audit=audit(),
    )
    workflow = GenericAgentFactoryWorkflow(
        blueprint=blueprint,
        store=store,
        runner=cast(
            Any,
            SimpleNamespace(
                registry=SimpleNamespace(
                    registration=registration,
                ),
            ),
        ),
        request_source=cast(Any, object()),
    )

    validated = workflow.validate_team(source.team.team_id)
    assert validated == narrowing.snapshot
