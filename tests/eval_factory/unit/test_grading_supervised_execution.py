from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from attachment_gateway_fixtures import (
    audit,
    profile,
    ref,
)
from test_domain_plan_review import (
    RUN_ID,
    USER,
)
from test_grading_agent import (
    _candidate_agent,
    _criteria_authority,
    _plan,
    _prompts,
    _proposal,
)

from eval_factory.agent_system.grading_planning import (
    GradingDesignPlanCompiler,
)
from eval_factory.agent_system.grading_registry import (
    GradingDesignAgentRegistryConfig,
    build_grading_design_agent_registry,
)
from eval_factory.agent_system.grading_subgraph import (
    GradingDesignRunnerFaultPoint,
    GradingDesignRunnerInjectedCrash,
    GradingDesignSubgraphError,
    StaticGradingDesignRunnerFaultInjector,
    SupervisedGradingDesignRunner,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.store import (
    FactoryControlInjectedCrash,
    FactoryControlNotFoundError,
    FactoryControlStore,
    FactoryControlStoreFaultPoint,
    StaticFactoryControlStoreFaultInjector,
)
from eval_factory.agent_system.supervisor import (
    ExecutionSupervisor,
)
from eval_factory.agent_system.workspace import (
    AgentWorkspaceManager,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentTaskOutcomeV2,
    GradingDesignOutcomeV2,
    GradingDesignResultV2,
    PlanDecisionKindV2,
    PlanKindV2,
)


async def _prepared(
    tmp_path: Path,
    *,
    generator_only: bool = False,
    reviewed: bool = True,
):
    authority = await _criteria_authority(tmp_path / "criteria")
    _, judge_prompt = _prompts()
    judge = profile(
        "supervised-grading-judge",
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
    )
    allowed_refs = (authority.generator_profile.to_ref(),) if generator_only else (judge.to_ref(),)
    plan = _plan(
        authority,
        judge_prompt=judge_prompt,
        allowed_judge_model_profile_refs=allowed_refs,
    )
    registry = build_grading_design_agent_registry(
        config=GradingDesignAgentRegistryConfig(
            prompt_ref=judge_prompt.to_ref(),
            model_policy_ref=plan.model_policy_ref,
        ),
        audit=audit(),
    )
    definition = registry.resolve(
        plan.agent_role,
        "grading-design",
    )
    agent, _, _, provider = _candidate_agent(
        tmp_path / "agent",
        authority=authority,
        source_plan=plan,
        proposal=_proposal(plan),
        judge_profile=judge,
        agent_definition_ref=definition.to_ref(),
    )
    store = authority.store
    run = store.get_run(RUN_ID)
    compiled = GradingDesignPlanCompiler(registry).compile(
        plan=plan,
        policy=store.get_policy(run.policy_ref.object_id),
        audit=audit(),
    )
    store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=run.run_version,
        plan_kind=PlanKindV2.GRADING_DESIGN,
        plan=plan,
        compiled_plan=compiled,
        audit=audit(),
        idempotency_key="commit-grading-plan-supervised",
    )
    if reviewed:
        review_service = PlanReviewService(store)
        opened = review_service.open_plan(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.GRADING_DESIGN,
            requested_by=USER,
            idempotency_key="open-grading-plan",
            audit=audit(),
        )
        review_service.decide(
            opened.request.review_request_id,
            PlanReviewDecisionSubmissionV1(
                expected_plan_version=plan.plan_version,
                decision=PlanDecisionKindV2.APPROVE,
                decided_by=USER,
                reason_code="GRADING_PLAN_APPROVED",
                idempotency_key="approve-grading-plan",
            ),
            audit=audit(),
        )
        review_service.resume(
            opened.request.review_request_id,
            PlanReviewResumeSubmissionV1(
                expected_plan_version=plan.plan_version,
                resumed_by=USER,
                idempotency_key="resume-grading-plan",
            ),
            audit=audit(),
        )
    supervisor = ExecutionSupervisor(
        store,
        registry,
        workspace_manager=AgentWorkspaceManager(tmp_path / "workspaces"),
    )
    kwargs = {
        "run_id": RUN_ID,
        "plan": plan,
        "compiled_plan": compiled,
        "criteria_result": authority.result,
        "criteria_route": authority.route,
        "rubric_set": authority.rubric_set,
        "evaluator_spec": authority.evaluator_spec,
        "reference_policy": authority.reference_policy,
        "tool_policy": authority.tool_policy,
        "model_authorizations": (),
        "evaluated_at": datetime(2026, 8, 6, tzinfo=UTC),
        "audit": audit(),
    }
    return agent, provider, supervisor, kwargs


async def _author_result(agent, kwargs):
    execution = await agent.author(
        task_ref=ref("agent-task", "grading-store-test"),
        plan=kwargs["plan"],
        criteria_result=kwargs["criteria_result"],
        criteria_route=kwargs["criteria_route"],
        rubric_set=kwargs["rubric_set"],
        evaluator_spec=kwargs["evaluator_spec"],
        reference_policy=kwargs["reference_policy"],
        tool_policy=kwargs["tool_policy"],
        model_authorizations=kwargs["model_authorizations"],
        evaluated_at=kwargs["evaluated_at"],
        audit=kwargs["audit"],
    )
    return execution.result


@pytest.mark.asyncio
async def test_supervised_grading_commits_agent_and_domain_once(
    tmp_path: Path,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(tmp_path)
    runner = SupervisedGradingDesignRunner(
        supervisor=supervisor,
        agent=agent,
    )

    first = await runner.run(**kwargs)
    files = _physical_files(supervisor.workspace_manager.root)
    counts = _authority_counts(supervisor.store.path)
    replay = await runner.run(**kwargs)

    assert first.result.outcome is (GradingDesignOutcomeV2.SUCCEEDED)
    assert first.envelope.outcome is (AgentTaskOutcomeV2.SUCCEEDED)
    assert replay == first
    assert len(provider.calls) == 1
    assert _physical_files(supervisor.workspace_manager.root) == files
    assert _authority_counts(supervisor.store.path) == counts
    material = supervisor.store.get_domain_result(
        RUN_ID,
        PlanKindV2.GRADING_DESIGN,
    )
    assert GradingDesignResultV2.model_validate_json(material.result_record_json) == first.result


@pytest.mark.asyncio
async def test_supervised_grading_persists_collision_as_typed_block(
    tmp_path: Path,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(
        tmp_path,
        generator_only=True,
    )

    execution = await SupervisedGradingDesignRunner(
        supervisor=supervisor,
        agent=agent,
    ).run(**kwargs)

    assert execution.result.outcome is (GradingDesignOutcomeV2.BLOCKED_POLICY)
    assert execution.result.reason_codes == ("GENERATOR_JUDGE_COLLISION",)
    assert execution.envelope.outcome is (AgentTaskOutcomeV2.BLOCKED)
    assert execution.envelope.gateway_receipt_refs == ()
    assert provider.calls == []
    assert (
        supervisor.store.get_domain_result(
            RUN_ID,
            PlanKindV2.GRADING_DESIGN,
        ).result_ref
        == execution.result.to_ref()
    )


@pytest.mark.asyncio
async def test_supervised_grading_requires_resumed_review(
    tmp_path: Path,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(
        tmp_path,
        reviewed=False,
    )
    before_counts = _authority_counts(supervisor.store.path)
    before_files = _physical_files(supervisor.workspace_manager.root)

    with pytest.raises(
        GradingDesignSubgraphError,
        match="resumed plan authority",
    ):
        await SupervisedGradingDesignRunner(
            supervisor=supervisor,
            agent=agent,
        ).run(**kwargs)

    assert provider.calls == []
    assert _authority_counts(supervisor.store.path) == before_counts
    assert _physical_files(supervisor.workspace_manager.root) == before_files


@pytest.mark.asyncio
async def test_supervised_grading_rejects_stale_source_before_side_effects(
    tmp_path: Path,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(tmp_path)
    criteria_result = kwargs["criteria_result"]
    kwargs["criteria_result"] = criteria_result.model_copy(
        update={
            "route_decision_ref": ref(
                "model-route-decision",
                "stale-supervised",
            )
        }
    )
    before_counts = _authority_counts(supervisor.store.path)
    before_files = _physical_files(supervisor.workspace_manager.root)

    with pytest.raises(
        RuntimeError,
        match="stale",
    ):
        await SupervisedGradingDesignRunner(
            supervisor=supervisor,
            agent=agent,
        ).run(**kwargs)

    assert provider.calls == []
    assert _authority_counts(supervisor.store.path) == before_counts
    assert _physical_files(supervisor.workspace_manager.root) == before_files


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_point",
    tuple(GradingDesignRunnerFaultPoint),
)
async def test_supervised_grading_crash_recovery_and_exact_replay(
    tmp_path: Path,
    fault_point: GradingDesignRunnerFaultPoint,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(tmp_path)
    crashing = SupervisedGradingDesignRunner(
        supervisor=supervisor,
        agent=agent,
        fault_injector=(StaticGradingDesignRunnerFaultInjector(crash_points=frozenset({fault_point}))),
    )

    with pytest.raises(
        GradingDesignRunnerInjectedCrash,
        match=fault_point.value,
    ):
        await crashing.run(**kwargs)

    runner = SupervisedGradingDesignRunner(
        supervisor=supervisor,
        agent=agent,
    )
    recovered = await runner.run(**kwargs)
    files = _physical_files(supervisor.workspace_manager.root)
    counts = _authority_counts(supervisor.store.path)
    replay = await runner.run(**kwargs)

    assert replay == recovered
    assert len(provider.calls) == 1
    assert _physical_files(supervisor.workspace_manager.root) == files
    assert _authority_counts(supervisor.store.path) == counts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_point",
    (
        FactoryControlStoreFaultPoint.AFTER_DOMAIN_RESULT,
        FactoryControlStoreFaultPoint.AFTER_DOMAIN_RESULT_HEAD,
        FactoryControlStoreFaultPoint.AFTER_OUTBOX,
        FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
    ),
)
async def test_grading_result_store_faults_roll_back_all_authority(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    agent, _, supervisor, kwargs = await _prepared(tmp_path)
    result = await _author_result(agent, kwargs)
    before = _authority_counts(supervisor.store.path)
    faulted = FactoryControlStore(
        supervisor.store.path,
        fault_injector=StaticFactoryControlStoreFaultInjector(crash_points=frozenset({fault_point})),
    )

    with pytest.raises(
        FactoryControlInjectedCrash,
        match=fault_point.value,
    ):
        faulted.commit_domain_result(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.GRADING_DESIGN,
            result=result,
            idempotency_key="fault-grading-result",
        )

    assert _authority_counts(supervisor.store.path) == before
    with pytest.raises(
        FactoryControlNotFoundError,
        match="current domain result",
    ):
        supervisor.store.get_domain_result(
            RUN_ID,
            PlanKindV2.GRADING_DESIGN,
        )


@pytest.mark.asyncio
async def test_grading_result_head_rebuilds_from_immutable_history(
    tmp_path: Path,
) -> None:
    agent, _, supervisor, kwargs = await _prepared(tmp_path)
    execution = await SupervisedGradingDesignRunner(
        supervisor=supervisor,
        agent=agent,
    ).run(**kwargs)
    with sqlite3.connect(supervisor.store.path) as connection:
        connection.execute(
            """
            DELETE FROM domain_result_current_heads
            WHERE run_id = ? AND plan_kind = ?
            """,
            (RUN_ID, PlanKindV2.GRADING_DESIGN.value),
        )

    with pytest.raises(
        FactoryControlNotFoundError,
        match="current domain result",
    ):
        supervisor.store.get_domain_result(
            RUN_ID,
            PlanKindV2.GRADING_DESIGN,
        )
    rebuilt = supervisor.store.rebuild_current_heads()

    assert rebuilt.domain_result_head_count == 2
    assert (
        supervisor.store.get_domain_result(
            RUN_ID,
            PlanKindV2.GRADING_DESIGN,
        ).result_ref
        == execution.result.to_ref()
    )


def _physical_files(
    root: Path,
) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _authority_counts(
    path: Path,
) -> tuple[int, ...]:
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "agent_tasks",
                "agent_work_leases",
                "agent_work_events",
                "agent_results",
                "domain_results",
                "domain_result_current_heads",
                "domain_result_idempotency",
                "factory_control_outbox",
            )
        )
