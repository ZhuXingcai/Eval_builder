from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from test_support.team_runtime_fixtures import (
    audit as team_audit,
)
from test_support.team_runtime_fixtures import (
    team_fixture,
)

from eval_factory.agent_system.dataset_graph_handlers import (
    FactoryDatasetGraphHandlers,
)
from eval_factory.agent_system.graph import (
    FactoryGraphDriftError,
    FactoryGraphSignalV2,
    FactoryGraphState,
)
from eval_factory.agent_system.planner_assessment import (
    FactoryPlannerAssessmentCompiler,
    FactoryPlannerAssessmentError,
    FactoryPlannerAssessmentFacts,
)
from eval_factory.agent_system.planner_assessment_source import (
    FactoryPlannerAssessmentSource,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    CompiledDatasetBuildPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    FactoryRunStatusV2,
    FactoryRunV2,
    PlannerAssessmentActionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetNextActionV2,
    FactoryDatasetRunViewV2,
)
from eval_factory.team import TeamStore

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 19, tzinfo=UTC),
        created_by="planner-assessment-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage4",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _plan() -> tuple[
    DatasetBuildPlanV2,
    CompiledDatasetBuildPlanV2,
]:
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
    run_ref = _ref("factory-run", "run-1")
    plan = DatasetBuildPlanV2.create(
        plan_id="dataset-build-plan.stage4",
        run_ref=run_ref,
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
        policy_ref=_ref("factory-run-policy"),
        tasks=(task,),
        topological_task_keys=(task.task_key,),
        audit=_audit(),
    )
    return plan, compiled


def _run(
    compiled: CompiledDatasetBuildPlanV2,
    *,
    delivery_manifest_ref: ObjectRef | None = None,
) -> FactoryRunV2:
    return FactoryRunV2.create(
        run_id="factory-run-stage4",
        run_version=2,
        status=(
            FactoryRunStatusV2.COMPLETED if delivery_manifest_ref is not None else FactoryRunStatusV2.RUNNING
        ),
        policy_ref=compiled.policy_ref,
        requirement_spec_ref=_ref("evaluation-requirement-spec"),
        current_plan_ref=compiled.source_plan_ref,
        compiled_plan_ref=compiled.to_ref(),
        active_task_refs=(),
        result_refs=(),
        pending_review_ref=None,
        planner_assessment_ref=None,
        completion_ref=(_ref("factory-run-completion") if delivery_manifest_ref is not None else None),
        delivery_manifest_ref=delivery_manifest_ref,
        transition_count=2,
        model_requests_used=0,
        model_tokens_used=0,
        cost_micro_usd_used=0,
        audit=_audit(),
    )


def _facts(
    *,
    pending: tuple[ObjectRef, ...] = (),
    ready: tuple[ObjectRef, ...] = (),
    retryable: tuple[ObjectRef, ...] = (),
    terminal: tuple[ObjectRef, ...] = (),
    invalidated: tuple[ObjectRef, ...] = (),
    approval: tuple[ObjectRef, ...] = (),
    unknown: tuple[ObjectRef, ...] = (),
    conflicts: tuple[ObjectRef, ...] = (),
    missing: tuple[ObjectRef, ...] = (),
    manifest: ObjectRef | None = None,
) -> FactoryPlannerAssessmentFacts:
    _plan_value, compiled = _plan()
    return FactoryPlannerAssessmentFacts(
        run=_run(compiled, delivery_manifest_ref=manifest),
        compiled_plan=compiled,
        validator_result_refs=(_ref("validator-result"),),
        pending_task_refs=pending,
        ready_task_refs=ready,
        retryable_task_refs=retryable,
        terminal_failure_task_refs=terminal,
        invalidated_task_refs=invalidated,
        approval_required_task_refs=approval,
        unknown_outcome_refs=unknown,
        open_conflict_refs=conflicts,
        missing_output_refs=missing,
        delivery_manifest_ref=manifest,
    )


@pytest.mark.parametrize(
    ("facts", "action", "rationale"),
    (
        (
            _facts(unknown=(_ref("gateway-invocation"),)),
            PlannerAssessmentActionV2.ESCALATE,
            ("UNKNOWN_OUTCOME_VERIFICATION_REQUIRED",),
        ),
        (
            _facts(conflicts=(_ref("team-conflict"),)),
            PlannerAssessmentActionV2.ESCALATE,
            ("OPEN_TEAM_CONFLICT",),
        ),
        (
            _facts(approval=(_ref("team-task"),)),
            PlannerAssessmentActionV2.ESCALATE,
            ("USER_APPROVAL_REQUIRED",),
        ),
        (
            _facts(terminal=(_ref("team-task"),)),
            PlannerAssessmentActionV2.ESCALATE,
            ("TERMINAL_WORK_FAILED",),
        ),
        (
            _facts(invalidated=(_ref("team-task"),)),
            PlannerAssessmentActionV2.REPLAN,
            ("CURRENT_PLAN_INVALIDATED",),
        ),
        (
            _facts(retryable=(_ref("team-task"),)),
            PlannerAssessmentActionV2.RETRY,
            ("RETRYABLE_WORK_REMAINS",),
        ),
        (
            _facts(ready=(_ref("team-task"),)),
            PlannerAssessmentActionV2.CONTINUE,
            ("APPROVED_WORK_REMAINS",),
        ),
        (
            _facts(missing=(_ref("artifact-head"),)),
            PlannerAssessmentActionV2.CONTINUE,
            ("REQUIRED_OUTPUT_MISSING",),
        ),
        (
            _facts(),
            PlannerAssessmentActionV2.FINISH,
            ("EXECUTION_FRONTIER_COMPLETE",),
        ),
        (
            _facts(
                manifest=_ref("candidate-dataset-delivery-manifest"),
            ),
            PlannerAssessmentActionV2.FINISH,
            ("ALL_REQUIRED_AUTHORITY_CURRENT",),
        ),
    ),
)
def test_assessment_compiler_uses_closed_precedence(
    facts: FactoryPlannerAssessmentFacts,
    action: PlannerAssessmentActionV2,
    rationale: tuple[str, ...],
) -> None:
    result = FactoryPlannerAssessmentCompiler().compile(
        facts,
        audit=_audit(),
    )

    assert result.proposed_action is action
    assert result.rationale_codes == rationale
    assert result.run_ref == facts.run.to_ref()
    assert result.compiled_plan_ref == facts.compiled_plan.to_ref()


@pytest.mark.parametrize(
    ("action", "signal"),
    (
        (
            PlannerAssessmentActionV2.CONTINUE,
            FactoryGraphSignalV2.CONTINUE,
        ),
        (
            PlannerAssessmentActionV2.RETRY,
            FactoryGraphSignalV2.RETRY,
        ),
        (
            PlannerAssessmentActionV2.REPLAN,
            FactoryGraphSignalV2.REPLAN,
        ),
        (
            PlannerAssessmentActionV2.ESCALATE,
            FactoryGraphSignalV2.ESCALATE,
        ),
        (
            PlannerAssessmentActionV2.FINISH,
            FactoryGraphSignalV2.FINISH,
        ),
    ),
)
def test_graph_signal_comes_from_assessment_action(
    action: PlannerAssessmentActionV2,
    signal: FactoryGraphSignalV2,
) -> None:
    facts = _facts(
        manifest=(
            _ref("candidate-dataset-delivery-manifest")
            if action is PlannerAssessmentActionV2.FINISH
            else None
        ),
    )
    assessment = (
        FactoryPlannerAssessmentCompiler()
        .compile(
            facts,
            audit=_audit(),
        )
        .model_copy(update={"proposed_action": action})
    )

    assert (
        FactoryDatasetGraphHandlers.signal_from_assessment(
            assessment,
        )
        is signal
    )


def test_assessment_precedence_escalates_before_replan_retry() -> None:
    facts = _facts(
        approval=(_ref("team-task", "approval"),),
        invalidated=(_ref("team-task", "invalidated"),),
        retryable=(_ref("team-task", "retry"),),
    )

    result = FactoryPlannerAssessmentCompiler().compile(
        facts,
        audit=_audit(),
    )

    assert result.proposed_action is PlannerAssessmentActionV2.ESCALATE
    assert result.affected_task_refs == facts.approval_required_task_refs


def test_assessment_source_projects_current_team_authority(
    tmp_path: Path,
) -> None:
    _plan_value, compiled = _plan()
    run = _run(compiled)

    class _FactorySource:
        def get_run(self, run_id: str) -> FactoryRunV2:
            assert run_id == run.run_id
            return run

        def get_plan(
            self,
            run_id: str,
        ) -> tuple[
            DatasetBuildPlanV2,
            CompiledDatasetBuildPlanV2,
        ]:
            assert run_id == run.run_id
            return _plan_value, compiled

    fixture = team_fixture()
    team_store = TeamStore(tmp_path / "team.sqlite3")
    team_store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    checkpoint = team_store.create_checkpoint(
        fixture.team.team_id,
        audit=team_audit(),
        idempotency_key="checkpoint",
    )
    view = FactoryDatasetRunViewV2.create(
        request_ref=_ref("factory-dataset-run-request"),
        dataset_run_ref=run.to_ref(),
        status=run.status,
        pending_review_refs=(),
        item_binding_refs=(),
        candidate_count=0,
        rejected_count=0,
        blocked_count=0,
        incomplete_count=0,
        aggregate_result_ref=None,
        delivery_manifest_ref=None,
        next_action=FactoryDatasetNextActionV2.ADVANCE,
        audit=_audit(),
    )
    source = FactoryPlannerAssessmentSource(
        factory_store=cast(
            FactoryControlStore,
            _FactorySource(),
        ),
        team_store=team_store,
        team_id=fixture.team.team_id,
        audit=_audit(),
    )

    valid_state: FactoryGraphState = {
        "factory_run_id": run.run_id,
        "expected_run_version": run.run_version,
        "transition_count": 1,
        "signal": FactoryGraphSignalV2.CONTINUE,
        "run_ref": run.to_ref(),
        "team_ref": fixture.team.to_ref(),
        "team_checkpoint_ref": checkpoint.to_ref(),
    }
    result = source(
        valid_state,
        view,
        run,
    )

    assert result.proposed_action is PlannerAssessmentActionV2.CONTINUE
    assert result.validator_result_refs == (checkpoint.to_ref(),)
    assert result.affected_task_refs

    for field_name, value, message in (
        ("run_ref", _ref("factory-run", "stale"), "Factory"),
        ("team_ref", _ref("agent-team", "stale"), "Team"),
        (
            "team_checkpoint_ref",
            _ref("team-checkpoint", "stale", version="v1"),
            "checkpoint",
        ),
    ):
        changed = dict(valid_state)
        changed[field_name] = value
        with pytest.raises(FactoryGraphDriftError, match=message):
            source(
                cast(FactoryGraphState, changed),
                view,
                run,
            )


def test_assessment_rejects_stale_plan_and_missing_evidence() -> None:
    facts = _facts()
    stale_run = facts.run.model_copy(
        update={
            "compiled_plan_ref": _ref(
                "compiled-dataset-build-plan",
                "stale",
            ),
        },
    )
    with pytest.raises(
        FactoryPlannerAssessmentError,
        match="not current",
    ):
        FactoryPlannerAssessmentCompiler().compile(
            FactoryPlannerAssessmentFacts(
                run=stale_run,
                compiled_plan=facts.compiled_plan,
                validator_result_refs=facts.validator_result_refs,
            ),
            audit=_audit(),
        )

    with pytest.raises(
        FactoryPlannerAssessmentError,
        match="validator",
    ):
        FactoryPlannerAssessmentCompiler().compile(
            FactoryPlannerAssessmentFacts(
                run=facts.run,
                compiled_plan=facts.compiled_plan,
                validator_result_refs=(),
            ),
            audit=_audit(),
        )


def test_assessment_facts_require_sorted_disjoint_tasks() -> None:
    facts = _facts()
    with pytest.raises(ValueError, match="sorted"):
        FactoryPlannerAssessmentFacts(
            run=facts.run,
            compiled_plan=facts.compiled_plan,
            validator_result_refs=(
                _ref("validator-result", "z"),
                _ref("validator-result", "a"),
            ),
        )
    task = _ref("team-task")
    with pytest.raises(ValueError, match="disjoint"):
        FactoryPlannerAssessmentFacts(
            run=facts.run,
            compiled_plan=facts.compiled_plan,
            validator_result_refs=facts.validator_result_refs,
            ready_task_refs=(task,),
            terminal_failure_task_refs=(task,),
        )
