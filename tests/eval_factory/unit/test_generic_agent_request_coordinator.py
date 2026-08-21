from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_factory_control_graph import _plan
from test_factory_core_material import _execution
from test_factory_graph_bootstrap import _factory_inputs
from test_support.team_runtime_fixtures import audit, ref, team_fixture

from eval_factory.agent_system.store import FactoryControlNotFoundError
from eval_factory.contracts.dataset_runtime_v2 import FactoryItemStageV2
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflowError,
    GenericAgentTeamTaskSpec,
)
from eval_factory.packs.generic_agent_trace.request_coordinator import (
    GenericAgentFactoryRequestCoordinator,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterializer,
)
from eval_factory.team import TeamStore


class _Preparation:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _record(self, name: str, kwargs: dict[str, object]) -> None:
        self.calls.append((name, kwargs))

    def prepare_requirement_planning(
        self,
        task_id: str,
        **kwargs: object,
    ) -> None:
        self._record("requirement", {"task_id": task_id, **kwargs})

    def prepare_trace_ingestion(
        self,
        task_id: str,
        **kwargs: object,
    ) -> None:
        self._record("trace", {"task_id": task_id, **kwargs})

    def prepare_task_authoring(
        self,
        task_id: str,
        **kwargs: object,
    ) -> None:
        self._record("task", {"task_id": task_id, **kwargs})

    def prepare_plan_review(
        self,
        task_id: str,
        **kwargs: object,
    ) -> None:
        self._record("review", {"task_id": task_id, **kwargs})


class _FactoryStore:
    def __init__(self, plan: object, compiled: object) -> None:
        self.plan = plan
        self.compiled = compiled
        self.binding = SimpleNamespace(
            item_id="item-request-coordinator",
        )
        self.item_run = SimpleNamespace(
            run_id="factory-run://request-coordinator-item",
        )

    def get_plan(self, run_id: str):
        del run_id
        return self.plan, self.compiled

    def get_item_binding(self, dataset_run_id: str, item_id: str):
        del dataset_run_id, item_id
        return self.binding

    def get_item_run(self, binding: object):
        assert binding is self.binding
        return self.item_run

    def get_domain_plan(self, run_id: str, plan_kind: object):
        del run_id, plan_kind
        raise FactoryControlNotFoundError("not found")

    def get_item_stage_head(self, item_id: str, stage: FactoryItemStageV2):
        assert item_id == self.binding.item_id
        return SimpleNamespace(
            result_ref=ref(
                {
                    FactoryItemStageV2.TASK_AUTHORING: ("r4-task-contract-set"),
                    FactoryItemStageV2.ITEM_QUALITY: ("item-quality-compilation-result"),
                    FactoryItemStageV2.CRITERIA_RUBRIC: ("criteria-rubric-result"),
                }[stage],
                stage.value.lower(),
                version="v2",
            ),
        )


def _fixture(tmp_path):
    policy, requirement, request = _factory_inputs()
    source_ref = _execution().decisions[0].source_ref
    base = team_fixture()
    materialized = GenericAgentTaskGraphMaterializer().materialize(
        registration=base.registration,
        requirement_ref=requirement.to_ref(),
        source_refs=(source_ref,),
        independent_session_refs={
            role: ref("harness-session", f"coordinator-{role}")
            for role in (
                "control",
                "coordinator",
                "quality",
                "requirement",
                "task",
                "trace",
            )
        },
        team_id="team-request-coordinator",
        team_incarnation_id="team-request-coordinator-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    team_store = TeamStore(tmp_path / "team.sqlite3")
    authority = materialized.authority
    team_store.create_team(
        team=authority.team,
        roster=authority.roster,
        graph=authority.graph,
        authority=authority.authority,
        idempotency_key="create-team",
    )
    plan, compiled = _plan(
        run_ref=ref("factory-run", "request-coordinator", version="v2"),
        policy_ref=policy.to_ref(),
    )
    preparation = _Preparation()
    candidate = _execution()
    candidate_material = SimpleNamespace(
        authority=SimpleNamespace(
            to_ref=lambda: ref(
                "factory-trace-candidate-authority",
                "candidate",
            ),
        ),
        preparation=SimpleNamespace(
            decision=candidate.decisions[0],
            extracted_prompt=candidate.extracted_prompts[0],
            inferred_intent=candidate.inferred_intents[0],
            rewrite_candidate=candidate.rewrite_candidates[0],
        ),
    )
    factory_store = _FactoryStore(plan, compiled)
    coordinator = GenericAgentFactoryRequestCoordinator(
        request=request,
        requirement=requirement,
        policy=policy,
        factory_store=cast(Any, factory_store),
        candidate_store=cast(
            Any,
            SimpleNamespace(get=lambda **kwargs: candidate_material),
        ),
        item_materializer=cast(
            Any,
            SimpleNamespace(materialize=lambda **kwargs: object()),
        ),
        materialization=materialized,
        preparation=cast(Any, preparation),
        trace_sources={
            source_ref: TraceSourceRef(
                source_trace_id="source-trace://material",
                source_uri="file:///tmp/material.jsonl",
                raw_sha256="a" * 64,
                adapter_name="raw_traj_v1",
                adapter_version="v1",
                processing_class="RESTRICTED_TRACE_RAW",
            ),
        },
    )
    return coordinator, preparation, team_store, materialized


def _spec(capability_id: str) -> GenericAgentTeamTaskSpec:
    return GenericAgentTeamTaskSpec(
        capability_id=capability_id,
        dependency_capability_ids=(),
        input_schema_refs=(),
        output_schema_refs=(),
    )


def test_request_coordinator_prepares_current_frontier_requests(
    tmp_path,
) -> None:
    coordinator, preparation, team_store, materialized = _fixture(
        tmp_path,
    )
    snapshot = team_store.get_snapshot(
        materialized.authority.team.team_id,
    )
    by_capability: dict[str, list[object]] = {}
    for instance in materialized.task_instances:
        by_capability.setdefault(instance.capability_id, []).append(
            team_store.get_task_work(
                snapshot.team.team_id,
                instance.task_id,
            ),
        )

    for capability_id in (
        "capability.requirement-planning",
        "capability.trace-ingestion",
        "capability.task-authoring",
    ):
        coordinator.prepare(
            snapshot=snapshot,
            work=by_capability[capability_id][0],  # type: ignore[arg-type]
            spec=_spec(capability_id),
            audit=audit(),
        )
    global_review = next(
        team_store.get_task_work(
            snapshot.team.team_id,
            instance.task_id,
        )
        for instance in materialized.task_instances
        if instance.plan_kind is not None
        and instance.scope_ref is None
        and instance.capability_id == "capability.plan-review"
        and instance.plan_kind.value == "GLOBAL_BUILD"
    )
    coordinator.prepare(
        snapshot=snapshot,
        work=global_review,
        spec=_spec("capability.plan-review"),
        audit=audit(),
    )
    for instance in materialized.task_instances:
        if instance.capability_id == "capability.plan-review" and instance.scope_ref is not None:
            coordinator.prepare(
                snapshot=snapshot,
                work=team_store.get_task_work(
                    snapshot.team.team_id,
                    instance.task_id,
                ),
                spec=_spec("capability.plan-review"),
                audit=audit(),
            )

    assert [name for name, _ in preparation.calls] == [
        "requirement",
        "trace",
        "task",
        "review",
        "review",
        "review",
        "review",
    ]
    assert preparation.calls[1][1]["attempt"] == 1
    assert {call["plan_kind"].value for name, call in preparation.calls if name == "review"} == {
        "GLOBAL_BUILD",
        "ATTACHMENT_GENERATION",
        "CRITERIA_RUBRIC",
        "GRADING_DESIGN",
    }


def test_request_coordinator_fails_closed_for_unprepared_domain(
    tmp_path,
) -> None:
    coordinator, _preparation, team_store, materialized = _fixture(
        tmp_path,
    )
    snapshot = team_store.get_snapshot(
        materialized.authority.team.team_id,
    )
    instance = next(
        value
        for value in materialized.task_instances
        if value.capability_id == "capability.attachment-reconstruction"
    )
    with pytest.raises(
        GenericAgentFactoryWorkflowError,
        match="unavailable",
    ):
        coordinator.prepare(
            snapshot=snapshot,
            work=team_store.get_task_work(
                snapshot.team.team_id,
                instance.task_id,
            ),
            spec=_spec(instance.capability_id),
            audit=audit(),
        )

    with pytest.raises(
        GenericAgentFactoryWorkflowError,
        match="metadata drifted",
    ):
        coordinator.prepare(
            snapshot=snapshot,
            work=team_store.get_task_work(
                snapshot.team.team_id,
                instance.task_id,
            ),
            spec=_spec("capability.delivery"),
            audit=audit(),
        )
