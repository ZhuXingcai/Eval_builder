from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel
from test_support.team_runtime_fixtures import (
    audit,
    capability_runtime,
    ref,
    team_fixture,
)

from eval_factory.blueprints import EvaluationBlueprintV1
from eval_factory.packs import (
    build_generic_agent_evaluation_blueprint,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflow,
    GenericAgentFactoryWorkflowError,
    GenericAgentTeamTaskSpec,
    compile_generic_agent_team_task_specs,
    materialize_generic_agent_team,
)
from eval_factory.team import (
    TeamCapabilityRunner,
    TeamSnapshot,
    TeamStore,
    TeamTaskWork,
)


class _UnusedRequestSource:
    def build(
        self,
        *,
        snapshot: TeamSnapshot,
        work: TeamTaskWork,
        spec: GenericAgentTeamTaskSpec,
    ) -> BaseModel:
        del snapshot, work, spec
        raise AssertionError(
            "invalid Team must fail before request materialization",
        )


def _blueprint() -> EvaluationBlueprintV1:
    fixture = team_fixture()
    return build_generic_agent_evaluation_blueprint(
        registration=fixture.registration,
        requirement_ref=fixture.team.goal_ref,
        team_ref=fixture.team.to_ref(),
        roster_ref=fixture.roster.to_ref(),
        task_graph_ref=fixture.graph.to_ref(),
        authority_ref=fixture.authority.to_ref(),
        audit=audit(),
    )


def test_blueprint_compiles_exact_ten_capability_dag() -> None:
    blueprint = _blueprint()
    specs = compile_generic_agent_team_task_specs(blueprint)

    assert len(specs) == 10
    assert len({value.capability_id for value in specs}) == 10
    observed_edges = {
        (dependency, spec.capability_id) for spec in specs for dependency in spec.dependency_capability_ids
    }
    expected_edges = {
        (
            edge.producer_capability_id,
            edge.consumer_capability_id,
        )
        for edge in blueprint.data_flow_edges
    }
    assert observed_edges == expected_edges


def test_pack_materializes_exact_ten_task_team() -> None:
    fixture = team_fixture()
    sessions = {
        role: ref("harness-session", f"stage4-{role}")
        for role in (
            "control",
            "coordinator",
            "quality",
            "requirement",
            "task",
            "trace",
        )
    }
    authority = materialize_generic_agent_team(
        registration=fixture.registration,
        requirement_ref=fixture.team.goal_ref,
        independent_session_refs=sessions,
        team_id="team-stage4",
        team_incarnation_id="team-stage4-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=fixture.registration,
        requirement_ref=fixture.team.goal_ref,
        team_ref=authority.team.to_ref(),
        roster_ref=authority.roster.to_ref(),
        task_graph_ref=authority.graph.to_ref(),
        authority_ref=authority.authority.to_ref(),
        audit=audit(),
    )
    capability_by_ref = {
        value.to_ref(): value.capability_id for value in fixture.registration.capability_definitions
    }
    capability_by_task = {
        task.task_id: capability_by_ref[task.capability_definition_ref] for task in authority.graph.tasks
    }
    observed = {
        capability_by_task[task.task_id]: tuple(
            sorted(capability_by_task[value] for value in task.dependency_task_ids)
        )
        for task in authority.graph.tasks
    }
    expected = {
        value.capability_id: value.dependency_capability_ids
        for value in compile_generic_agent_team_task_specs(
            blueprint,
        )
    }

    assert len(authority.graph.tasks) == 10
    assert authority.snapshot().team == authority.team
    assert observed == expected
    assert sum(task.status.value == "READY" for task in authority.graph.tasks) == 1
    assert authority.team.goal_ref == fixture.team.goal_ref


def test_workflow_rejects_incomplete_team_before_dispatch(
    tmp_path: Path,
) -> None:
    fixture = team_fixture()
    registry, runtime, _providers = capability_runtime(fixture)
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    workflow = GenericAgentFactoryWorkflow(
        blueprint=build_generic_agent_evaluation_blueprint(
            registration=fixture.registration,
            requirement_ref=fixture.team.goal_ref,
            team_ref=fixture.team.to_ref(),
            roster_ref=fixture.roster.to_ref(),
            task_graph_ref=fixture.graph.to_ref(),
            authority_ref=fixture.authority.to_ref(),
            audit=audit(),
        ),
        store=store,
        runner=TeamCapabilityRunner(
            store=store,
            runtime=runtime,
            registry=registry,
        ),
        request_source=_UnusedRequestSource(),
    )

    with pytest.raises(
        GenericAgentFactoryWorkflowError,
        match="exact ten",
    ):
        workflow.validate_team(fixture.team.team_id)

    assert not store.current_artifact_heads(
        fixture.team.team_id,
    )
    assert store.ready_tasks(fixture.team.team_id)
