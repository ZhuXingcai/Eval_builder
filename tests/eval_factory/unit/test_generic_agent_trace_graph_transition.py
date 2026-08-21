from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_factory_graph_bootstrap import _factory_inputs
from test_support.team_runtime_fixtures import HASH, audit, ref

from eval_factory.agent_system.graph import (
    FactoryGraphDriftError,
    FactoryGraphSignalV2,
    FactoryGraphState,
)
from eval_factory.agent_system.graph_authority import (
    FactoryGraphAuthorityResolver,
)
from eval_factory.agent_system.graph_journal import FactoryGraphJournalStore
from eval_factory.agent_system.graph_transition import (
    FactoryGraphTransitionService,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunStatusV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.harness import (
    HarnessGraphExecutionBindingV1,
    HarnessSessionIdentityV1,
    HarnessSessionProjectionV1,
    HarnessSessionStateV1,
    HarnessSessionStatusV1,
    StaticPackRegistry,
)
from eval_factory.packs import (
    build_generic_agent_evaluation_blueprint,
    build_generic_agent_trace_execution_pack,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterializer,
)
from eval_factory.packs.generic_agent_trace.trace_graph_refinement import (
    GenericAgentTraceGraphRefinement,
)
from eval_factory.team import TeamSnapshot, TeamStore, TeamTaskStatusV1


@dataclass(frozen=True, slots=True)
class _SessionSource:
    projection: HarnessSessionProjectionV1

    def get_projection_by_ref(
        self,
        reference: ObjectRef,
    ) -> HarnessSessionProjectionV1:
        assert reference == self.projection.session.session_ref
        return self.projection


class _CompletedTraceTeamStore:
    def __init__(self, store: TeamStore) -> None:
        self.store = store
        self.promotions = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def get_task_work(
        self,
        team_id: str,
        task_id: str,
    ) -> SimpleNamespace:
        del team_id, task_id
        return SimpleNamespace(status=TeamTaskStatusV1.COMPLETED)

    def promote_narrowed_graph(
        self,
        team_id: str,
        **values: object,
    ) -> TeamSnapshot:
        self.promotions += 1
        return self.store.promote_narrowed_graph(
            team_id,
            expected_team_ref=cast(ObjectRef, values["expected_team_ref"]),
            expected_graph_ref=cast(ObjectRef, values["expected_graph_ref"]),
            expected_authority_ref=cast(
                ObjectRef,
                values["expected_authority_ref"],
            ),
            successor=cast(TeamSnapshot, values["successor"]),
            idempotency_key=cast(str, values["idempotency_key"]),
        )


def _candidate_material(
    source_ref: ObjectRef,
    *,
    disposition: TraceCandidateDispositionV2,
) -> SimpleNamespace:
    return SimpleNamespace(
        authority=SimpleNamespace(
            trace_source_ref=source_ref,
            to_ref=lambda: ref(
                "factory-trace-candidate-authority",
                source_ref.object_sha256,
                version="v1",
            ),
        ),
        preparation=SimpleNamespace(
            decision=SimpleNamespace(disposition=disposition),
        ),
    )


def test_trace_narrowing_advances_graph_binding_and_replays(
    tmp_path: Path,
) -> None:
    policy, requirement, request = _factory_inputs()
    factory_store = FactoryControlStore(tmp_path / "factory.sqlite3")
    run = factory_store.create_run(
        policy=policy,
        requirement=requirement,
        idempotency_key="create-run",
    )
    factory_store.commit_dataset_request(
        request,
        idempotency_key="commit-request",
    )
    registration = build_generic_agent_trace_execution_pack(audit=audit())
    sources = (
        ref("trace-source", "graph-transition-a", version="v2"),
        ref("trace-source", "graph-transition-b", version="v2"),
    )
    materialization = GenericAgentTaskGraphMaterializer().materialize(
        registration=registration,
        requirement_ref=requirement.to_ref(),
        source_refs=sources,
        independent_session_refs={
            role: ref("harness-session", f"graph-transition-{role}")
            for role in (
                "control",
                "coordinator",
                "quality",
                "requirement",
                "task",
                "trace",
            )
        },
        team_id="team-graph-transition",
        team_incarnation_id="team-graph-transition-incarnation-1",
        max_model_requests=policy.max_model_requests,
        max_model_tokens=policy.max_model_tokens,
        max_cost_micro_usd=policy.max_cost_micro_usd,
        audit=audit(),
    )
    team_store = TeamStore(tmp_path / "team.sqlite3")
    source_authority = materialization.authority
    team_store.create_team(
        team=source_authority.team,
        roster=source_authority.roster,
        graph=source_authority.graph,
        authority=source_authority.authority,
        idempotency_key="create-team",
    )
    source_checkpoint = team_store.create_checkpoint(
        source_authority.team.team_id,
        audit=audit(),
        idempotency_key="source-checkpoint",
    )
    source_snapshot = team_store.get_snapshot(
        source_authority.team.team_id,
    )
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=registration,
        requirement_ref=requirement.to_ref(),
        team_ref=source_snapshot.team.to_ref(),
        roster_ref=source_snapshot.roster.to_ref(),
        task_graph_ref=source_snapshot.graph.to_ref(),
        authority_ref=source_snapshot.authority.to_ref(),
        audit=audit(),
    )

    requirement_policy_ref = ref(
        "harness-requirement-policy",
        "graph-transition-ready",
    )
    identity = HarnessSessionIdentityV1.create(
        session_id="session-graph-transition",
        incarnation_id="session-graph-transition-incarnation-1",
        composition_ref=registration.composition.to_ref(),
        created_by="graph-transition-test",
        created_at=audit().created_at,
        audit=audit(),
    )
    session = HarnessSessionStateV1.create(
        identity=identity,
        session_version=1,
        status=HarnessSessionStatusV1.ACTIVE,
        last_event_sequence=1,
        current_interpretation_ref=ref(
            "requirement-interpretation",
            "graph-transition-ready",
        ),
        current_requirement_policy_ref=requirement_policy_ref,
        updated_at=audit().created_at,
    )
    session_source = _SessionSource(
        HarnessSessionProjectionV1(
            session=session,
            transcript=(),
            event_refs=(),
            current_interpretation_ref=session.current_interpretation_ref,
            current_requirement_policy_ref=requirement_policy_ref,
            latest_gateway=None,
            pending_clarification_questions=(),
            source_fingerprint=HASH,
        ),
    )
    binding = HarnessGraphExecutionBindingV1.create(
        binding_id="binding-graph-transition",
        session_ref=identity.to_ref(),
        requirement_ref=requirement.to_ref(),
        requirement_policy_ref=requirement_policy_ref,
        factory_run_ref=run.to_ref(),
        factory_request_ref=request.to_ref(),
        factory_policy_ref=policy.to_ref(),
        expected_factory_run_version=run.run_version,
        pack_manifest_ref=registration.manifest.to_ref(),
        composition_ref=registration.composition.to_ref(),
        blueprint_ref=blueprint.to_ref(),
        team_ref=source_snapshot.team.to_ref(),
        roster_ref=source_snapshot.roster.to_ref(),
        task_graph_ref=source_snapshot.graph.to_ref(),
        execution_authority_ref=source_snapshot.authority.to_ref(),
        team_checkpoint_ref=source_checkpoint.to_ref(),
        expected_team_version=source_snapshot.team.team_version,
        expected_task_graph_revision=source_snapshot.graph.revision,
        expected_authority_version=(source_snapshot.authority.authority_version),
        blackboard_head_refs=(),
        thread_id="thread-graph-transition",
        max_transitions=request.max_transitions,
        audit=audit(),
    )
    journal = FactoryGraphJournalStore(tmp_path / "journal.sqlite3")
    journal.commit_binding(binding, idempotency_key="commit-binding")
    resolver = FactoryGraphAuthorityResolver(
        journal=journal,
        session_source=session_source,
        factory_source=factory_store,
        team_source=team_store,
        pack_registry=StaticPackRegistry((registration,)),
        blueprints=(blueprint,),
    )
    transition = FactoryGraphTransitionService(
        journal=journal,
        authority=resolver,
        audit=audit(),
    )
    state: FactoryGraphState = {
        "factory_run_id": request.dataset_run_id,
        "expected_run_version": run.run_version,
        "transition_count": 1,
        "signal": FactoryGraphSignalV2.CONTINUE,
        "graph_binding_ref": binding.to_ref(),
        "run_ref": run.to_ref(),
        "run_status": FactoryRunStatusV2.RUNNING,
    }
    pre_ref = transition.begin_transition(
        node="dispatch_ready_work",
        state=state,
    )

    completed_store = _CompletedTraceTeamStore(team_store)
    materials = {
        sources[0]: _candidate_material(
            sources[0],
            disposition=TraceCandidateDispositionV2.CANDIDATE,
        ),
        sources[1]: _candidate_material(
            sources[1],
            disposition=TraceCandidateDispositionV2.NON_CANDIDATE,
        ),
    }
    refinement = GenericAgentTraceGraphRefinement(
        dataset_run_id=request.dataset_run_id,
        store=cast(Any, completed_store),
        candidate_store=cast(
            Any,
            SimpleNamespace(
                get=lambda **values: materials[values["trace_source_ref"]],
            ),
        ),
        materialization=materialization,
    )
    narrowing = refinement.refine(
        source_snapshot.team.team_id,
        audit=audit(),
    )
    narrowed_checkpoint = team_store.current_checkpoint(
        source_snapshot.team.team_id,
    )
    assert narrowing is not None and narrowing.changed is True
    assert len(narrowing.snapshot.graph.tasks) == 15
    assert narrowed_checkpoint is not None
    assert narrowed_checkpoint.task_graph_ref == narrowing.snapshot.graph.to_ref()

    update = transition.complete_transition(
        node="dispatch_ready_work",
        state=state,
        pre_checkpoint_ref=pre_ref,
    )
    successor_ref = cast(ObjectRef, update["graph_binding_ref"])
    successor = journal.get_binding_by_ref(successor_ref)
    post = journal.current_checkpoint(binding.binding_id)
    assert successor.team_ref == narrowing.snapshot.team.to_ref()
    assert successor.task_graph_ref == narrowing.snapshot.graph.to_ref()
    assert successor.execution_authority_ref == (narrowing.snapshot.authority.to_ref())
    assert successor.team_checkpoint_ref == narrowed_checkpoint.to_ref()
    assert post is not None and post.binding_ref == successor.to_ref()
    with pytest.raises(
        FactoryGraphDriftError,
        match="not current",
    ):
        resolver.resolve_current(binding.to_ref())
    assert resolver.resolve_current(successor.to_ref()) == successor

    replay = refinement.refine(
        source_snapshot.team.team_id,
        audit=audit(),
    )
    assert replay is not None and replay.changed is False
    assert completed_store.promotions == 1
    assert team_store.current_checkpoint(source_snapshot.team.team_id) == narrowed_checkpoint
