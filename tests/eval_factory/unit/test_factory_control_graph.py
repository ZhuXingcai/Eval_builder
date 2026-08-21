from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

import pytest

from eval_factory.agent_system.graph import (
    FactoryControlGraph,
    FactoryGraphDriftError,
    FactoryGraphHandlerError,
    FactoryGraphSignalV2,
    FactoryGraphState,
    FactoryGraphTransitionLimitError,
    factory_graph_checkpointer,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    CompiledDatasetBuildPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 6, tzinfo=UTC),
        created_by="factory-control-graph-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="phase-a",
                sha256=HASH,
            ),
        ),
    )


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://core-v1",
        allowed_task_kinds=("planning", "trace-extraction"),
        max_transitions=64,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=100,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )


def _requirement() -> EvaluationRequirementSpecV2:
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://example",
        run_id="factory-run://example",
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Build source-grounded rewrite candidates.",),
        constraints=("Do not expose raw traces.",),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )


def _store(tmp_path: Path) -> tuple[FactoryControlStore, str]:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    run = store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run",
    )
    return store, run.run_id


def _plan(
    *,
    run_ref: ObjectRef,
    policy_ref: ObjectRef,
) -> tuple[DatasetBuildPlanV2, CompiledDatasetBuildPlanV2]:
    task = DatasetBuildPlanTaskV2(
        task_key="extract",
        stage="core",
        task_kind="trace-extraction",
        agent_role="trace-extraction-agent",
        dependency_task_keys=(),
        input_object_types=("requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        required_capability_ids=("capability://trace-extraction",),
        acceptance_check_refs=(_ref("acceptance-check"),),
        plan_review_kind=None,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )
    plan = DatasetBuildPlanV2.create(
        plan_id="dataset-build-plan://example",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        goals=("Build source-grounded rewrite candidates.",),
        user_constraints=(),
        assumptions=(),
        unresolved_questions=(),
        stage_order=("core",),
        tasks=(task,),
        required_review_kinds=(),
        total_model_requests=1,
        total_model_tokens=4096,
        total_cost_micro_usd=100_000,
        audit=_audit(),
    )
    compiled = CompiledDatasetBuildPlanV2.create(
        compiled_plan_id="compiled-dataset-build-plan://example",
        source_plan_ref=plan.to_ref(),
        policy_ref=policy_ref,
        tasks=plan.tasks,
        topological_task_keys=("extract",),
        audit=_audit(),
    )
    return plan, compiled


def _initial_state(run_id: str) -> FactoryGraphState:
    return {
        "factory_run_id": run_id,
        "expected_run_version": 0,
        "transition_count": 0,
        "signal": FactoryGraphSignalV2.CONTINUE,
    }


def test_graph_state_is_closed_to_refs_and_bounded_scalars() -> None:
    hints = get_type_hints(FactoryGraphState)
    forbidden = {
        "attachment_bytes",
        "credential",
        "model_output_body",
        "prompt_body",
        "rag_text",
        "raw_trace",
        "retrieved_text",
        "runtime_transcript",
    }
    assert forbidden.isdisjoint(hints)
    assert "factory_run_id" in FactoryGraphState.__required_keys__
    assert "expected_run_version" in FactoryGraphState.__required_keys__
    assert "run_ref" in FactoryGraphState.__optional_keys__
    assert all("dict" not in str(annotation).casefold() for annotation in hints.values())


def test_parent_graph_has_stable_control_and_error_topology(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    compiled = FactoryControlGraph(store).compile()
    graph = compiled.get_graph()

    assert {
        "load_authority",
        "admit_request",
        "requirement_planner",
        "review_global_plan",
        "compile_global_plan",
        "dispatch_ready_work",
        "validate_core_vertical",
        "planner_assessment",
        "assemble_core_output",
        "review_final_delivery",
        "export_core_output",
        "blocked",
    }.issubset(graph.nodes)
    edge_pairs = {(edge.source, edge.target) for edge in graph.edges}
    assert ("__start__", "load_authority") in edge_pairs
    assert ("admit_request", "requirement_planner") in edge_pairs
    assert ("planner_assessment", "__end__") in edge_pairs
    assert ("planner_assessment", "review_global_plan") not in edge_pairs
    assert ("export_core_output", "__end__") in edge_pairs
    assert ("blocked", "__end__") in edge_pairs


def test_sqlite_checkpoint_resumes_after_interrupt_without_replaying_prior_node(
    tmp_path: Path,
) -> None:
    store, run_id = _store(tmp_path)
    planner_calls = 0

    def planner(state: FactoryGraphState) -> dict[str, object]:
        nonlocal planner_calls
        planner_calls += 1
        assert state["factory_run_id"] == run_id
        return {"signal": FactoryGraphSignalV2.WAIT}

    checkpoint_path = tmp_path / "factory-graph-checkpoint.sqlite3"
    with factory_graph_checkpointer(
        checkpoint_path,
        control_store_path=store.path,
    ) as checkpointer:
        graph = FactoryControlGraph(
            store,
            handlers={"requirement_planner": planner},
        ).compile(
            checkpointer=checkpointer,
            interrupt_after=["admit_request"],
        )
        config = {"configurable": {"thread_id": run_id}}
        interrupted = graph.invoke(_initial_state(run_id), config)
        assert interrupted["transition_count"] == 2
        assert planner_calls == 0

        resumed = graph.invoke(None, config)
        assert resumed["signal"] is FactoryGraphSignalV2.WAIT
        assert resumed["transition_count"] == 3
        assert planner_calls == 1

    assert checkpoint_path.exists()
    assert checkpoint_path != store.path


@pytest.mark.asyncio
async def test_async_handler_requires_ainvoke(
    tmp_path: Path,
) -> None:
    store, run_id = _store(tmp_path)
    calls = 0

    async def planner(
        state: FactoryGraphState,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert state["factory_run_id"] == run_id
        return {"signal": FactoryGraphSignalV2.WAIT}

    graph = FactoryControlGraph(
        store,
        handlers={"requirement_planner": planner},
    ).compile()

    result = await graph.ainvoke(_initial_state(run_id))

    assert result["signal"] is FactoryGraphSignalV2.WAIT
    assert calls == 1
    with pytest.raises(
        FactoryGraphHandlerError,
        match="requires ainvoke",
    ):
        graph.invoke(_initial_state(run_id))
    assert calls == 1


def test_resume_rejects_store_checkpoint_drift(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    with factory_graph_checkpointer(
        tmp_path / "factory-graph-checkpoint.sqlite3",
        control_store_path=store.path,
    ) as checkpointer:
        graph = FactoryControlGraph(store).compile(
            checkpointer=checkpointer,
            interrupt_after=["load_authority"],
        )
        config = {"configurable": {"thread_id": run_id}}
        graph.invoke(_initial_state(run_id), config)

        current = store.get_run(run_id)
        plan, compiled_plan = _plan(
            run_ref=current.to_ref(),
            policy_ref=current.policy_ref,
        )
        store.commit_plan(
            run_id=run_id,
            expected_run_version=0,
            plan=plan,
            compiled_plan=compiled_plan,
            audit=_audit(),
            idempotency_key="commit-plan",
        )

        with pytest.raises(FactoryGraphDriftError, match="run version"):
            graph.invoke(None, config)


def test_checkpoint_path_cannot_alias_control_store(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    with (
        pytest.raises(ValueError, match="separate"),
        factory_graph_checkpointer(
            store.path,
            control_store_path=store.path,
        ),
    ):
        pass


def test_transition_limit_fails_closed(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    graph = FactoryControlGraph(store, transition_limit=1).compile()
    with pytest.raises(FactoryGraphTransitionLimitError, match="transition limit"):
        graph.invoke(_initial_state(run_id))
