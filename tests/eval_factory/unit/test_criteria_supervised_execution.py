from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from test_criteria_agent import (
    _attachment_result,
    _binding_definitions,
    _candidate_agent,
    _catalog,
    _draft,
    _plan,
    _proposal,
    _quality,
    _solvability,
)
from test_criteria_planning import _registry
from test_domain_plan_review import (
    RUN_ID,
    USER,
    _commit_attachment_plan,
)
from test_domain_plan_review import (
    _setup as _domain_setup,
)

from eval_factory.agent_system.criteria_agent import (
    CriteriaRubricAgentError,
)
from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
)
from eval_factory.agent_system.criteria_subgraph import (
    CriteriaRubricRunnerFaultPoint,
    CriteriaRubricRunnerInjectedCrash,
    CriteriaRubricSubgraphError,
    StaticCriteriaRubricRunnerFaultInjector,
    SupervisedCriteriaRubricRunner,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.supervisor import ExecutionSupervisor
from eval_factory.agent_system.workspace import AgentWorkspaceManager
from eval_factory.contracts.agent_system_v2 import (
    AgentTaskOutcomeV2,
    CriteriaRubricOutcomeV2,
    CriteriaRubricResultV2,
    PlanDecisionKindV2,
    PlanKindV2,
)
from eval_factory.task_authoring import (
    RubricAuthoringOutcome,
    RubricAuthoringReason,
)


async def _prepared(
    tmp_path: Path,
    *,
    proposal=None,
    failed: bool = False,
    reviewed: bool = True,
):
    task_draft = _draft()
    catalog = _catalog()
    _, source_store, attachment_compiler = _domain_setup(
        tmp_path / "source-control",
    )
    attachment_plan = _commit_attachment_plan(
        source_store,
        attachment_compiler,
    )
    attachment_result = _attachment_result(
        attachment_plan.to_ref(),
    )
    source_store.commit_domain_result(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        result=attachment_result,
        idempotency_key="commit-attachment-result-for-criteria",
    )
    quality = _quality(attachment_result)
    solvability = _solvability(quality)
    plan = _plan(
        run_ref_value=source_store.get_run(RUN_ID).to_ref(),
        task_draft=task_draft,
        quality=quality,
        solvability=solvability,
        catalog=catalog,
    )
    agent, _, provider, _, store = _candidate_agent(
        tmp_path / "agent",
        plan=plan,
        proposal=proposal or _proposal(),
        failed=failed,
        control_store=source_store,
    )
    if reviewed:
        review_service = PlanReviewService(store)
        opened = review_service.open_plan(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.CRITERIA_RUBRIC,
            requested_by=USER,
            idempotency_key="open-criteria-plan",
            audit=plan.audit,
        )
        review_service.decide(
            opened.request.review_request_id,
            PlanReviewDecisionSubmissionV1(
                expected_plan_version=plan.plan_version,
                decision=PlanDecisionKindV2.APPROVE,
                decided_by=USER,
                reason_code="CRITERIA_PLAN_APPROVED",
                idempotency_key="approve-criteria-plan",
            ),
            audit=plan.audit,
        )
        review_service.resume(
            opened.request.review_request_id,
            PlanReviewResumeSubmissionV1(
                expected_plan_version=plan.plan_version,
                resumed_by=USER,
                idempotency_key="resume-criteria-plan",
            ),
            audit=plan.audit,
        )
    compiled = CriteriaRubricPlanCompiler(_registry()).compile(
        plan=plan,
        policy=store.get_policy(store.get_run(RUN_ID).policy_ref.object_id),
        audit=plan.audit,
    )
    supervisor = ExecutionSupervisor(
        store,
        _registry(),
        workspace_manager=AgentWorkspaceManager(tmp_path / "workspaces"),
    )
    kwargs = {
        "run_id": RUN_ID,
        "plan": plan,
        "compiled_plan": compiled,
        "task_draft": task_draft,
        "attachment_quality": quality,
        "solvability": solvability,
        "binding_definitions": _binding_definitions(),
        "tool_catalog": catalog,
        "audit": plan.audit,
    }
    return agent, provider, supervisor, kwargs


@pytest.mark.asyncio
async def test_supervised_criteria_commits_agent_and_domain_authority_once(
    tmp_path: Path,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(tmp_path)
    runner = SupervisedCriteriaRubricRunner(
        supervisor=supervisor,
        agent=agent,
    )

    first = await runner.run(**kwargs)
    files = _physical_files(supervisor.workspace_manager.root)
    counts = _authority_counts(supervisor.store.path)
    replay = await runner.run(**kwargs)

    assert first.result.outcome is CriteriaRubricOutcomeV2.SUCCEEDED
    assert first.envelope.outcome is AgentTaskOutcomeV2.SUCCEEDED
    assert replay == first
    assert len(provider.calls) == 1
    assert _physical_files(supervisor.workspace_manager.root) == files
    assert _authority_counts(supervisor.store.path) == counts
    material = supervisor.store.get_domain_result(
        RUN_ID,
        PlanKindV2.CRITERIA_RUBRIC,
    )
    assert CriteriaRubricResultV2.model_validate_json(material.result_record_json) == first.result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("proposal", "failed", "expected_result", "expected_envelope"),
    (
        (
            _proposal(
                outcome=RubricAuthoringOutcome.ABSTAIN,
                reasons=frozenset({RubricAuthoringReason.AMBIGUOUS_RUBRIC}),
            ),
            False,
            CriteriaRubricOutcomeV2.ABSTAINED,
            AgentTaskOutcomeV2.ABSTAINED,
        ),
        (
            _proposal(),
            True,
            CriteriaRubricOutcomeV2.BLOCKED_CAPABILITY,
            AgentTaskOutcomeV2.BLOCKED,
        ),
    ),
)
async def test_supervised_criteria_persists_typed_non_success(
    tmp_path: Path,
    proposal,
    failed: bool,
    expected_result: CriteriaRubricOutcomeV2,
    expected_envelope: AgentTaskOutcomeV2,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(
        tmp_path,
        proposal=proposal,
        failed=failed,
    )

    execution = await SupervisedCriteriaRubricRunner(
        supervisor=supervisor,
        agent=agent,
    ).run(**kwargs)

    assert execution.result.outcome is expected_result
    assert execution.envelope.outcome is expected_envelope
    assert execution.envelope.output_refs == (execution.result.to_ref(),)
    assert len(provider.calls) == 1
    assert (
        supervisor.store.get_domain_result(
            RUN_ID,
            PlanKindV2.CRITERIA_RUBRIC,
        ).result_ref
        == execution.result.to_ref()
    )


@pytest.mark.asyncio
async def test_supervised_criteria_rejects_stale_source_before_side_effects(
    tmp_path: Path,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(tmp_path)
    task_draft = kwargs["task_draft"]
    kwargs["task_draft"] = task_draft.model_copy(
        update={"visible_prompt": (task_draft.visible_prompt + " changed")}
    )
    before_counts = _authority_counts(supervisor.store.path)
    before_files = _physical_files(supervisor.workspace_manager.root)

    with pytest.raises(
        CriteriaRubricAgentError,
        match="stale",
    ):
        await SupervisedCriteriaRubricRunner(
            supervisor=supervisor,
            agent=agent,
        ).run(**kwargs)

    assert provider.calls == []
    assert _authority_counts(supervisor.store.path) == before_counts
    assert _physical_files(supervisor.workspace_manager.root) == before_files


@pytest.mark.asyncio
async def test_supervised_criteria_requires_resumed_review(
    tmp_path: Path,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(
        tmp_path,
        reviewed=False,
    )
    before_counts = _authority_counts(supervisor.store.path)
    before_files = _physical_files(supervisor.workspace_manager.root)

    with pytest.raises(
        CriteriaRubricSubgraphError,
        match="resumed plan authority",
    ):
        await SupervisedCriteriaRubricRunner(
            supervisor=supervisor,
            agent=agent,
        ).run(**kwargs)

    assert provider.calls == []
    assert _authority_counts(supervisor.store.path) == before_counts
    assert _physical_files(supervisor.workspace_manager.root) == before_files


@pytest.mark.asyncio
async def test_supervised_criteria_requires_current_attachment_result(
    tmp_path: Path,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(tmp_path)
    quality = kwargs["attachment_quality"]
    kwargs["attachment_quality"] = quality.model_copy(
        update={
            "attachment_subgraph_result_ref": (_attachment_result().to_ref()),
        },
    )
    before_counts = _authority_counts(supervisor.store.path)
    before_files = _physical_files(supervisor.workspace_manager.root)

    with pytest.raises(
        CriteriaRubricSubgraphError,
        match="current attachment authority",
    ):
        await SupervisedCriteriaRubricRunner(
            supervisor=supervisor,
            agent=agent,
        ).run(**kwargs)

    assert provider.calls == []
    assert _authority_counts(supervisor.store.path) == before_counts
    assert _physical_files(supervisor.workspace_manager.root) == before_files


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_point",
    tuple(CriteriaRubricRunnerFaultPoint),
)
async def test_supervised_criteria_crash_recovery_and_exact_replay(
    tmp_path: Path,
    fault_point: CriteriaRubricRunnerFaultPoint,
) -> None:
    agent, provider, supervisor, kwargs = await _prepared(tmp_path)
    crashing = SupervisedCriteriaRubricRunner(
        supervisor=supervisor,
        agent=agent,
        fault_injector=(StaticCriteriaRubricRunnerFaultInjector(crash_points=frozenset({fault_point}))),
    )

    with pytest.raises(
        CriteriaRubricRunnerInjectedCrash,
        match=fault_point.value,
    ):
        await crashing.run(**kwargs)

    runner = SupervisedCriteriaRubricRunner(
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
