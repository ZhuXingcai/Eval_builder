from __future__ import annotations

from pathlib import Path

import pytest
from test_support.team_runtime_fixtures import audit, ref

from eval_factory.harness import ExecutionAuthorityV1
from eval_factory.packs import build_generic_agent_trace_execution_pack
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterializer,
)
from eval_factory.packs.generic_agent_trace.task_graph_narrowing import (
    GenericAgentTaskGraphNarrower,
)
from eval_factory.team import (
    TeamConcurrencyError,
    TeamIntegrityError,
    TeamSnapshot,
    TeamStore,
    TeamV1,
)


def _fixture(tmp_path: Path):
    sources = (
        ref("trace-source", "store-narrow-a", version="v1"),
        ref("trace-source", "store-narrow-b", version="v1"),
    )
    materialized = GenericAgentTaskGraphMaterializer().materialize(
        registration=build_generic_agent_trace_execution_pack(
            audit=audit(),
        ),
        requirement_ref=ref(
            "evaluation-requirement-spec",
            "store-narrow",
            version="v2",
        ),
        source_refs=sources,
        independent_session_refs={
            role: ref("harness-session", f"store-narrow-{role}")
            for role in (
                "control",
                "coordinator",
                "quality",
                "requirement",
                "task",
                "trace",
            )
        },
        team_id="team-store-narrow",
        team_incarnation_id="team-store-narrow-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    store = TeamStore(tmp_path / "team.sqlite3")
    source = materialized.authority.snapshot()
    store.create_team(
        team=source.team,
        roster=source.roster,
        graph=source.graph,
        authority=source.authority,
        idempotency_key="create-team-store-narrow",
    )
    narrowing = GenericAgentTaskGraphNarrower().narrow(
        snapshot=source,
        materialization=materialized,
        candidate_source_refs=(sources[0],),
        audit=audit(),
    )
    return store, source, narrowing


def test_team_store_commits_and_replays_graph_narrowing(
    tmp_path: Path,
) -> None:
    store, source, narrowing = _fixture(tmp_path)

    committed = store.promote_narrowed_graph(
        source.team.team_id,
        expected_team_ref=source.team.to_ref(),
        expected_graph_ref=source.graph.to_ref(),
        expected_authority_ref=source.authority.to_ref(),
        successor=narrowing.snapshot,
        idempotency_key="promote-store-narrow",
    )
    replay = store.promote_narrowed_graph(
        source.team.team_id,
        expected_team_ref=source.team.to_ref(),
        expected_graph_ref=source.graph.to_ref(),
        expected_authority_ref=source.authority.to_ref(),
        successor=narrowing.snapshot,
        idempotency_key="promote-store-narrow",
    )

    assert replay == committed
    assert store.get_snapshot(source.team.team_id) == committed
    assert len(committed.graph.tasks) == 15
    with pytest.raises(
        TeamConcurrencyError,
        match="stale",
    ):
        store.promote_narrowed_graph(
            source.team.team_id,
            expected_team_ref=source.team.to_ref(),
            expected_graph_ref=source.graph.to_ref(),
            expected_authority_ref=source.authority.to_ref(),
            successor=narrowing.snapshot,
            idempotency_key="stale-store-narrow",
        )


def test_team_store_rejects_budget_widening(
    tmp_path: Path,
) -> None:
    store, source, narrowing = _fixture(tmp_path)
    target = narrowing.snapshot
    widened_authority = ExecutionAuthorityV1.create(
        authority_id=target.authority.authority_id,
        authority_version=target.authority.authority_version,
        predecessor_authority_ref=(target.authority.predecessor_authority_ref),
        team_id=target.authority.team_id,
        team_incarnation_id=target.authority.team_incarnation_id,
        roster_ref=target.authority.roster_ref,
        task_graph_ref=target.authority.task_graph_ref,
        permission_policy_ref=(target.authority.permission_policy_ref),
        grants=target.authority.grants,
        max_model_requests=(target.authority.max_model_requests + 1),
        max_model_tokens=target.authority.max_model_tokens,
        max_cost_micro_usd=(target.authority.max_cost_micro_usd),
        used_model_requests=target.authority.used_model_requests,
        used_model_tokens=target.authority.used_model_tokens,
        used_cost_micro_usd=(target.authority.used_cost_micro_usd),
        audit=audit(),
    )
    widened_team = TeamV1.create(
        team_id=target.team.team_id,
        team_incarnation_id=target.team.team_incarnation_id,
        team_version=target.team.team_version,
        goal_ref=target.team.goal_ref,
        composition_ref=target.team.composition_ref,
        roster_ref=target.team.roster_ref,
        task_graph_ref=target.team.task_graph_ref,
        authority_ref=widened_authority.to_ref(),
        coordinator_member_id=target.team.coordinator_member_id,
        lifecycle=target.team.lifecycle,
        audit=audit(),
    )
    widened = TeamSnapshot(
        team=widened_team,
        roster=target.roster,
        graph=target.graph,
        authority=widened_authority,
    )

    with pytest.raises(
        TeamIntegrityError,
        match="fixed Team authority",
    ):
        store.promote_narrowed_graph(
            source.team.team_id,
            expected_team_ref=source.team.to_ref(),
            expected_graph_ref=source.graph.to_ref(),
            expected_authority_ref=source.authority.to_ref(),
            successor=widened,
            idempotency_key="widen-store-narrow",
        )
