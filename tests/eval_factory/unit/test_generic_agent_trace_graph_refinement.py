from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import audit, ref

from eval_factory.contracts.agent_system_v2 import (
    TraceCandidateDispositionV2,
)
from eval_factory.packs import build_generic_agent_trace_execution_pack
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterializer,
)
from eval_factory.packs.generic_agent_trace.trace_graph_refinement import (
    GenericAgentTraceGraphRefinement,
)
from eval_factory.team import TeamCheckpointV1, TeamTaskStatusV1


class _Store:
    def __init__(
        self,
        snapshot: object,
        *,
        completed: bool,
        fail_checkpoint_once: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.completed = completed
        self.fail_checkpoint_once = fail_checkpoint_once
        self.promotions = 0
        self.checkpoint_attempts = 0
        self.checkpoint = None

    def get_snapshot(self, team_id: str):
        return self.snapshot

    def get_task_work(self, team_id: str, task_id: str):
        return SimpleNamespace(
            status=(TeamTaskStatusV1.COMPLETED if self.completed else TeamTaskStatusV1.READY),
        )

    def promote_narrowed_graph(self, team_id: str, **kwargs: object):
        self.promotions += 1
        self.snapshot = kwargs["successor"]
        return self.snapshot

    def current_artifact_heads(self, team_id: str):
        return ()

    def current_checkpoint(self, team_id: str):
        return self.checkpoint

    def create_checkpoint(
        self,
        team_id: str,
        *,
        audit,
        idempotency_key: str,
    ):
        self.checkpoint_attempts += 1
        if self.fail_checkpoint_once:
            self.fail_checkpoint_once = False
            raise RuntimeError("checkpoint fault")
        snapshot = self.snapshot
        self.checkpoint = TeamCheckpointV1.create(
            checkpoint_id=f"{team_id}.checkpoint-{self.checkpoint_attempts}",
            team_ref=snapshot.team.to_ref(),
            roster_ref=snapshot.roster.to_ref(),
            task_graph_ref=snapshot.graph.to_ref(),
            authority_ref=snapshot.authority.to_ref(),
            artifact_head_refs=(),
            subscription_refs=(),
            member_session_refs=tuple(member.independent_session_ref for member in snapshot.roster.members),
            message_cursor=0,
            audit=audit,
        )
        return self.checkpoint


def _fixture():
    sources = (
        ref("trace-source", "refine-a", version="v1"),
        ref("trace-source", "refine-b", version="v1"),
    )
    materialized = GenericAgentTaskGraphMaterializer().materialize(
        registration=build_generic_agent_trace_execution_pack(
            audit=audit(),
        ),
        requirement_ref=ref(
            "evaluation-requirement-spec",
            "refine",
            version="v2",
        ),
        source_refs=sources,
        independent_session_refs={
            role: ref("harness-session", f"refine-{role}")
            for role in (
                "control",
                "coordinator",
                "quality",
                "requirement",
                "task",
                "trace",
            )
        },
        team_id="team-refine",
        team_incarnation_id="team-refine-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    materials = {
        source: SimpleNamespace(
            authority=SimpleNamespace(
                trace_source_ref=source,
                to_ref=lambda source=source: ref(
                    "factory-trace-candidate-authority",
                    source.object_sha256,
                    version="v1",
                ),
            ),
            preparation=SimpleNamespace(
                decision=SimpleNamespace(
                    disposition=(
                        TraceCandidateDispositionV2.CANDIDATE
                        if source == sources[0]
                        else TraceCandidateDispositionV2.NON_CANDIDATE
                    ),
                ),
            ),
        )
        for source in sources
    }
    return materialized, materials


def test_trace_graph_refinement_waits_then_commits() -> None:
    materialized, materials = _fixture()
    store = _Store(
        materialized.authority.snapshot(),
        completed=False,
    )
    refinement = GenericAgentTraceGraphRefinement(
        dataset_run_id="dataset-refine",
        store=cast(Any, store),
        candidate_store=cast(
            Any,
            SimpleNamespace(
                get=lambda **kwargs: materials[kwargs["trace_source_ref"]],
            ),
        ),
        materialization=materialized,
    )

    assert (
        refinement.refine(
            materialized.authority.team.team_id,
            audit=audit(),
        )
        is None
    )
    store.completed = True
    narrowed = refinement.refine(
        materialized.authority.team.team_id,
        audit=audit(),
    )
    replay = refinement.refine(
        materialized.authority.team.team_id,
        audit=audit(),
    )

    assert narrowed is not None and narrowed.changed is True
    assert replay is not None and replay.changed is False
    assert store.promotions == 1
    assert store.checkpoint_attempts == 1


def test_trace_graph_refinement_recovers_checkpoint_after_promotion() -> None:
    materialized, materials = _fixture()
    store = _Store(
        materialized.authority.snapshot(),
        completed=True,
        fail_checkpoint_once=True,
    )
    refinement = GenericAgentTraceGraphRefinement(
        dataset_run_id="dataset-refine",
        store=cast(Any, store),
        candidate_store=cast(
            Any,
            SimpleNamespace(
                get=lambda **kwargs: materials[kwargs["trace_source_ref"]],
            ),
        ),
        materialization=materialized,
    )

    with pytest.raises(RuntimeError, match="checkpoint fault"):
        refinement.refine(
            materialized.authority.team.team_id,
            audit=audit(),
        )
    recovered = refinement.refine(
        materialized.authority.team.team_id,
        audit=audit(),
    )
    replay = refinement.refine(
        materialized.authority.team.team_id,
        audit=audit(),
    )

    assert recovered is not None and recovered.changed is False
    assert replay is not None and replay.changed is False
    assert store.promotions == 1
    assert store.checkpoint_attempts == 2
    assert store.checkpoint is not None
