from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.agent_system.planner_assessment import (
    FactoryPlannerAssessmentCompiler,
    FactoryPlannerAssessmentFacts,
)
from eval_factory.agent_system.store import (
    FactoryControlConcurrencyError,
    FactoryControlInjectedCrash,
    FactoryControlStore,
    FactoryControlStoreFaultPoint,
    StaticFactoryControlStoreFaultInjector,
)
from eval_factory.contracts.agent_system_v2 import (
    CompiledDatasetBuildPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    FactoryRunV2,
    PlannerAssessmentActionV2,
    PlannerAssessmentV2,
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
        created_at=datetime(2026, 8, 19, tzinfo=UTC),
        created_by="planner-assessment-persistence-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage4",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy.stage4",
        allowed_task_kinds=("task-authoring",),
        max_transitions=64,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=10,
        max_model_tokens=10_000,
        max_cost_micro_usd=100_000,
        audit=_audit(),
    )


def _requirement() -> EvaluationRequirementSpecV2:
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec.stage4",
        run_id="factory-run-stage4",
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Build evaluation data.",),
        constraints=(),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )


def _prepare(
    store: FactoryControlStore,
) -> tuple[
    FactoryRunV2,
    CompiledDatasetBuildPlanV2,
]:
    run = store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run",
    )
    task = DatasetBuildPlanTaskV2(
        task_key="task",
        stage="factory",
        task_kind="task-authoring",
        agent_role="task",
        dependency_task_keys=(),
        input_object_types=("trace-evidence",),
        output_object_types=("task-candidate",),
        required_capability_ids=("capability.task-authoring",),
        acceptance_check_refs=(_ref("acceptance-check"),),
        plan_review_kind=None,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=1_000,
        max_cost_micro_usd=10_000,
    )
    plan = DatasetBuildPlanV2.create(
        plan_id="dataset-build-plan.stage4",
        run_ref=run.to_ref(),
        plan_version=1,
        predecessor_plan_ref=None,
        goals=("Build evaluation data.",),
        user_constraints=(),
        assumptions=(),
        unresolved_questions=(),
        stage_order=("factory",),
        tasks=(task,),
        required_review_kinds=(),
        total_model_requests=1,
        total_model_tokens=1_000,
        total_cost_micro_usd=10_000,
        audit=_audit(),
    )
    compiled = CompiledDatasetBuildPlanV2.create(
        compiled_plan_id="compiled-dataset-build-plan.stage4",
        source_plan_ref=plan.to_ref(),
        policy_ref=_policy().to_ref(),
        tasks=(task,),
        topological_task_keys=(task.task_key,),
        audit=_audit(),
    )
    current = store.commit_plan(
        run_id=run.run_id,
        expected_run_version=run.run_version,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-plan",
    )
    return current, compiled


def _assessment(
    run: FactoryRunV2,
    compiled: CompiledDatasetBuildPlanV2,
) -> PlannerAssessmentV2:
    return FactoryPlannerAssessmentCompiler().compile(
        FactoryPlannerAssessmentFacts(
            run=run,
            compiled_plan=compiled,
            validator_result_refs=(_ref("validator-result"),),
            ready_task_refs=(_ref("team-task"),),
        ),
        audit=_audit(),
    )


def test_assessment_commit_advances_run_and_replays(
    tmp_path: Path,
) -> None:
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    current, compiled = _prepare(store)
    assessment = _assessment(current, compiled)

    successor = store.commit_planner_assessment(
        run_id=current.run_id,
        expected_run_version=current.run_version,
        assessment=assessment,
        audit=_audit(),
        idempotency_key="commit-assessment",
    )
    replay = store.commit_planner_assessment(
        run_id=current.run_id,
        expected_run_version=current.run_version,
        assessment=assessment,
        audit=_audit(),
        idempotency_key="commit-assessment",
    )

    assert replay == successor
    assert successor.run_version == current.run_version + 1
    assert successor.planner_assessment_ref == assessment.to_ref()
    assert store.get_planner_assessment(assessment.to_ref()) == assessment
    assert store.list_planner_assessments(successor.run_id) == (assessment,)
    assert assessment.proposed_action is PlannerAssessmentActionV2.CONTINUE


def test_assessment_commit_rejects_stale_run(
    tmp_path: Path,
) -> None:
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    current, compiled = _prepare(store)
    assessment = _assessment(current, compiled)

    with pytest.raises(
        FactoryControlConcurrencyError,
        match="stale",
    ):
        store.commit_planner_assessment(
            run_id=current.run_id,
            expected_run_version=current.run_version - 1,
            assessment=assessment,
            audit=_audit(),
            idempotency_key="stale-assessment",
        )
    assert store.list_planner_assessments(current.run_id) == ()


@pytest.mark.parametrize(
    "fault_point",
    (
        FactoryControlStoreFaultPoint.AFTER_PLANNER_ASSESSMENT,
        FactoryControlStoreFaultPoint.AFTER_RUN,
        FactoryControlStoreFaultPoint.AFTER_RUN_HEAD,
        FactoryControlStoreFaultPoint.AFTER_OUTBOX,
        FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
    ),
)
def test_assessment_commit_fault_rolls_back_every_record(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    path = tmp_path / "factory.sqlite3"
    store = FactoryControlStore(path)
    current, compiled = _prepare(store)
    assessment = _assessment(current, compiled)
    crashing = FactoryControlStore(
        path,
        fault_injector=StaticFactoryControlStoreFaultInjector(
            frozenset(
                {
                    fault_point,
                }
            ),
        ),
    )

    with pytest.raises(
        FactoryControlInjectedCrash,
        match=fault_point.value,
    ):
        crashing.commit_planner_assessment(
            run_id=current.run_id,
            expected_run_version=current.run_version,
            assessment=assessment,
            audit=_audit(),
            idempotency_key="crash-assessment",
        )

    reopened = FactoryControlStore(path)
    assert reopened.get_run(current.run_id) == current
    assert reopened.list_planner_assessments(current.run_id) == ()
