from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from test_criteria_planning import (
    _plan as _criteria_plan,
)
from test_criteria_planning import (
    _registry as _criteria_registry,
)
from test_domain_plan_review import (
    RUN_ID,
    USER,
    _audit,
)
from test_domain_plan_review import (
    _registry as _domain_registry,
)
from test_domain_plan_review import (
    _setup as _domain_setup,
)

from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
)
from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
)
from eval_factory.agent_system.plan_adapters import (
    AttachmentGenerationPlanAdapter,
    CriteriaRubricPlanAdapter,
    GlobalBuildPlanAdapter,
    ReviewablePlanAdapterRegistry,
)
from eval_factory.agent_system.plan_review import (
    CriteriaRubricPlanReviewEditSubmissionV1,
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.store import (
    FactoryControlConflictError,
    FactoryControlInjectedCrash,
    FactoryControlIntegrityError,
    FactoryControlNotFoundError,
    FactoryControlStore,
    FactoryControlStoreFaultPoint,
    StaticFactoryControlStoreFaultInjector,
)
from eval_factory.contracts.agent_system_v2 import (
    CriteriaRubricOutcomeV2,
    CriteriaRubricResultV2,
    PlanDecisionKindV2,
    PlanKindV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.core import ObjectRef

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "current",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _service(
    tmp_path: Path,
) -> tuple[
    PlanReviewService,
    CriteriaRubricPlanCompiler,
]:
    _, store, _ = _domain_setup(tmp_path)
    domain = _domain_registry()
    criteria = _criteria_registry()
    registry = AgentRegistry(
        capabilities=(
            *domain.capabilities,
            *criteria.capabilities,
        ),
        definitions=(
            *domain.definitions,
            *criteria.definitions,
        ),
    )
    global_compiler = DatasetBuildPlanCompiler(registry)
    attachment_compiler = AttachmentGenerationPlanCompiler(registry)
    criteria_compiler = CriteriaRubricPlanCompiler(registry)
    service = PlanReviewService(
        store,
        compiler=global_compiler,
        adapter_registry=ReviewablePlanAdapterRegistry(
            (
                GlobalBuildPlanAdapter(global_compiler),
                AttachmentGenerationPlanAdapter(attachment_compiler),
                CriteriaRubricPlanAdapter(criteria_compiler),
            )
        ),
    )
    return service, criteria_compiler


def _commit(
    service: PlanReviewService,
    compiler: CriteriaRubricPlanCompiler,
):
    run = service.store.get_run(RUN_ID)
    plan = _criteria_plan(
        updates={"run_ref": run.to_ref()},
    )
    compiled = compiler.compile(
        plan=plan,
        policy=service.store.get_policy(run.policy_ref.object_id),
        audit=_audit(),
    )
    service.store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=run.run_version,
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-criteria-rubric-plan",
    )
    return plan


def _open(service: PlanReviewService):
    return service.open_plan(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        requested_by=USER,
        idempotency_key="open-criteria-rubric-review",
        audit=_audit(),
    )


def _result(
    plan,
    *,
    suffix: str = "current",
) -> CriteriaRubricResultV2:
    return CriteriaRubricResultV2.create(
        result_id=f"criteria-rubric-result://{suffix}",
        plan_ref=plan.to_ref(),
        task_draft_ref=plan.task_draft_ref,
        attachment_quality_ref=plan.attachment_quality_ref,
        solvability_ref=plan.solvability_ref,
        route_decision_ref=_ref("model-route-decision", suffix),
        gateway_receipt_ref=_ref("gateway-receipt", suffix),
        gateway_invocation_result_ref=_ref(
            "gateway-invocation-result",
            suffix,
        ),
        proposal_ref=_ref("criteria-rubric-proposal", suffix),
        rubric_set_ref=_ref("rubric-set", suffix),
        evaluator_spec_ref=_ref("evaluator-spec", suffix),
        reference_policy_ref=_ref("reference-policy", suffix),
        tool_policy_ref=_ref("tool-policy", suffix),
        contestant_tool_policy_ref=_ref(
            "contestant-tool-policy",
            suffix,
        ),
        outcome=CriteriaRubricOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=_audit(),
    )


def test_criteria_plan_approve_and_resume_uses_shared_state_machine(
    tmp_path: Path,
) -> None:
    service, compiler = _service(tmp_path)
    plan = _commit(service, compiler)
    opened = _open(service)

    assert opened.plan == plan
    assert opened.request.plan_kind is PlanKindV2.CRITERIA_RUBRIC
    assert "criterion_goals" in opened.presentation.editable_paths
    approved = service.decide(
        opened.request.review_request_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by=USER,
            reason_code="CRITERIA_RUBRIC_APPROVED",
            idempotency_key="approve-criteria-rubric",
        ),
        audit=_audit(),
    )
    resumed = service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-criteria-rubric",
        ),
        audit=_audit(),
    )

    assert approved.result.state is PlanReviewStateV2.APPROVED
    assert resumed.result.state is PlanReviewStateV2.RESUMED
    assert (
        service.store.get_domain_plan(
            RUN_ID,
            PlanKindV2.CRITERIA_RUBRIC,
        ).plan_ref
        == plan.to_ref()
    )


def test_criteria_plan_edit_invalidates_only_criteria_authority(
    tmp_path: Path,
) -> None:
    service, compiler = _service(tmp_path)
    base = _commit(service, compiler)
    committed_result = service.store.commit_domain_result(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        result=_result(base),
        idempotency_key="commit-criteria-result-before-edit",
    )
    opened = _open(service)
    waiting_run = service.store.get_run(RUN_ID)
    changed_goal = base.criterion_goals[0].model_copy(
        update={"goal_summary": "Judge the revised visible response goal."}
    )
    successor = _criteria_plan(
        updates={
            "run_ref": waiting_run.to_ref(),
            "plan_version": 2,
            "predecessor_plan_ref": base.to_ref(),
            "criterion_goals": (changed_goal,),
            "max_model_tokens": 8_000,
        },
    )
    edited = service.edit_plan(
        opened.request.review_request_id,
        CriteriaRubricPlanReviewEditSubmissionV1(
            expected_plan_version=1,
            edited_plan=successor,
            changed_paths=("criterion_goals", "max_model_tokens"),
            decided_by=USER,
            reason_code="REFINE_CRITERIA_GOALS",
            idempotency_key="edit-criteria-rubric",
        ),
        audit=_audit(),
    )

    assert edited.result.state is PlanReviewStateV2.REVISION_REQUESTED
    assert edited.revision is not None
    assert edited.revision.invalidated_object_refs == (base.to_ref(),)
    assert all(
        ref.object_type
        not in {
            "attachment-quality-assessment",
            "attachment-subgraph-result",
            "solvability-assessment",
        }
        for ref in edited.revision.invalidated_object_refs
    )
    resumed = service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-edited-criteria-rubric",
        ),
        audit=_audit(),
    )
    assert resumed.result.state is PlanReviewStateV2.RESUMED
    assert (
        service.store.get_domain_plan(
            RUN_ID,
            PlanKindV2.CRITERIA_RUBRIC,
        ).plan_ref
        == successor.to_ref()
    )
    assert committed_result.plan_ref == base.to_ref()
    with pytest.raises(FactoryControlNotFoundError):
        service.store.get_domain_result(
            RUN_ID,
            PlanKindV2.CRITERIA_RUBRIC,
        )


def test_criteria_result_store_replay_rebuild_conflict_and_drift(
    tmp_path: Path,
) -> None:
    service, compiler = _service(tmp_path)
    plan = _commit(service, compiler)
    result = _result(plan)

    first = service.store.commit_domain_result(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        result=result,
        idempotency_key="commit-criteria-result",
    )
    replay = service.store.commit_domain_result(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        result=result,
        idempotency_key="commit-criteria-result",
    )

    assert replay == first
    with pytest.raises(FactoryControlConflictError):
        service.store.commit_domain_result(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.CRITERIA_RUBRIC,
            result=_result(plan, suffix="changed"),
            idempotency_key="commit-criteria-result",
        )
    with pytest.raises(FactoryControlConflictError):
        service.store.commit_domain_result(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.CRITERIA_RUBRIC,
            result=result,
            idempotency_key="second-criteria-result",
        )

    with sqlite3.connect(service.store.path) as connection:
        connection.execute("DELETE FROM domain_result_current_heads")
    rebuild = service.store.rebuild_current_heads()
    assert rebuild.domain_result_head_count == 1
    assert (
        service.store.get_domain_result(
            RUN_ID,
            PlanKindV2.CRITERIA_RUBRIC,
        )
        == first
    )

    with sqlite3.connect(service.store.path) as connection:
        connection.execute(
            """
            UPDATE domain_results
            SET record_json_sha256 = ?
            WHERE result_object_id = ?
            """,
            ("f" * 64, result.object_id),
        )
    with pytest.raises(
        FactoryControlIntegrityError,
        match="record hash drifted",
    ):
        service.store.get_domain_result(
            RUN_ID,
            PlanKindV2.CRITERIA_RUBRIC,
        )


@pytest.mark.parametrize(
    "fault_point",
    (
        FactoryControlStoreFaultPoint.AFTER_DOMAIN_RESULT,
        FactoryControlStoreFaultPoint.AFTER_DOMAIN_RESULT_HEAD,
        FactoryControlStoreFaultPoint.AFTER_OUTBOX,
    ),
)
def test_criteria_result_faults_roll_back_all_authority(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    service, compiler = _service(tmp_path)
    plan = _commit(service, compiler)
    faulted = FactoryControlStore(
        service.store.path,
        fault_injector=StaticFactoryControlStoreFaultInjector(
            crash_points=frozenset({fault_point}),
        ),
    )

    with pytest.raises(
        FactoryControlInjectedCrash,
        match=fault_point.value,
    ):
        faulted.commit_domain_result(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.CRITERIA_RUBRIC,
            result=_result(plan),
            idempotency_key="fault-criteria-result",
        )

    with sqlite3.connect(service.store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM domain_results").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM domain_result_current_heads").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM domain_result_idempotency").fetchone() == (0,)
